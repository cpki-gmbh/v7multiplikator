; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .text
; memcpy dequant slice fill, next-slice check
%define DM_NEXT_SLICE_OFF 1                    ; [ebx + 1] = i + 1 for last-slice remainder check
%define DM_ERMS_THRESHOLD 65536                ; sizes >= 64KB dispatch to worker pool, below use ERMS
%define QA_CEIL_ADJUST 1                       ; subtract 1 in lea for ceil-division: [eax + ecx - 1]
%define QA_Q_STRIDE_SHIFT 6                    ; log2,64 : int8 stride per group in q array
%define QA_X_STRIDE_SHIFT 8                    ; log2,256 : float stride per group in x array, 64*4
%define DM_ALIGN_MASK ~31                      ; mask to round down to 32-byte alignment
     global v7d_memcpy                         ; exported: dispatched memcpy operation
     SF_START v7d_memcpy, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles, reserve frame
     SF_QWORD dm_dst                           ; destination pointer, stack slot
     SF_QWORD dm_src                           ; source pointer, stack slot
     SF_DWORD dm_byte_count                    ; total bytes to copy, stack slot
     SF_DWORD dm_num_slices                    ; computed slice count, stack slot
     SF_DWORD dm_bytes_per_slice               ; bytes per dispatch slice, stack slot
     SF_DWORD dm_caller_end                    ; caller steal loop bound, stack slot
     SF_DWORD dm_caller_count                  ; caller's completed job count, stack slot
     SF_COMMIT                                 ; finalize stack frame layout
; Save args
     mov dm_dst, rcx                           ; dst pointer
     mov dm_src, rdx                           ; src pointer
     mov dm_byte_count, r8d                    ; total byte count
; Init thread pool, runs once via CAS guard
     call initialize_once                      ; ensure workers are ready
     SET_FTZ_DAZ                               ; enable Flush-To-Zero and Denormals-Are-Zero
; Load byte_count for ERMS/dispatch decision
     mov eax, dm_byte_count                    ; reload byte_count from stack
; Simple threshold: < 64KB = ERMS, single-threaded, >= 64KB = DISPATCH, parallel
     cmp eax, DM_ERMS_THRESHOLD                ; byte_count >= 64KB?
     jae .dm_proceed_dispatch                  ; yes, split across workers
; no: fall through to single-threaded ERMS
.dm_call_erms:
; Reload args from stack, saved at function entry above
; API_CALL overwrites volatile regs, so reload unconditionally
     mov rcx, dm_dst                           ; rcx = original dst
     mov rdx, dm_src                           ; rdx = original src
     mov r8d, dm_byte_count                    ; r8d = original byte_count
     call v7s_memcpy_chunk                     ; ERMS memcpy kernel
     jmp .dm_done                              ; done
.dm_proceed_dispatch:
; Compute num_slices = min(byte_count / 1024, num_workers + 1), the caller participates
     mov eax, dm_byte_count                    ; eax = byte_count
     shr eax, 10                               ; eax = byte_count / 1024, fast divide by power of 2
; Clamp to num_workers + 1 (caller participates)
     mov ecx, [num_workers]                    ; worker count
     inc ecx                                   ; + caller
     cmp eax, ecx                              ; slices > participants?
     cmova eax, ecx                            ; branchless clamp: eax = min(byte_count/1024, workers+1)
     mov dm_num_slices, eax                    ; final slice count
     mov eax, dm_byte_count                    ; eax = byte_count
     xor edx, edx                              ; zero for div
     mov ecx, dm_num_slices                    ; ecx = num_slices
     div ecx                                   ; eax = byte_count / num_slices
     and eax, DM_ALIGN_MASK                    ; round down to 32-byte alignment
; === Fill jobs_arr with MATMUL_MEMCPY jobs ===
     xor ebx, ebx                              ; i = 0
     mov r12d, dm_num_slices                   ; num_slices, loop bound
     mov r13, dm_dst                           ; running dst pointer
     mov r14, dm_src                           ; running src pointer
     mov r15d, eax                             ; aligned bytes per slice remain in EAX
     xor edi, edi                              ; running_byte_offset = 0
     xor r9d, r9d                              ; actual_slices_filled = 0
     mov r8d, dm_byte_count                    ; loop-invariant total hoisted to the free register, eliminates 1 stack read per iteration; r8's only other use is the ERMS path, which returns before this loop
.dm_fill_loop:
; at least one slice: byte_count >= DM_ERMS_THRESHOLD, slice count clamped to num_workers+1 >= 1
; Compute byte_count = min,bytes_per_slice, byte_count - offset
     mov eax, r8d                              ; total bytes, register cache
     sub eax, edi                              ; remaining = total - running_offset, always > 0 below (proof: stride >= 1024 and remaining >= stride at every loop head)
; Last-slice remainder absorption: if this is the final slice, take all remaining bytes
     lea ecx, [ebx + DM_NEXT_SLICE_OFF]        ; ecx = i + 1
     cmp ecx, r12d                             ; i+1 == num_slices, last slice ?
     je .dm_set_count                          ; yes, take all remaining bytes
     cmp eax, r15d                             ; remaining > bytes_per_slice?
     jbe .dm_set_count                         ; remaining <= stride: keep eax=remaining, fall-through = full stride, zero taken branches on the common path
.dm_use_full_slice:
     mov eax, r15d                             ; eax = bytes_per_slice
.dm_set_count:
; Compute &jobs_arr[i] via JOB_ADDR macro, rsi = &jobs_arr[i]
     JOB_ADDR rbx                              ; rsi = &jobs_arr[i]
     JOB_SET_OUTPUT r13                        ; dst pointer for this slice
     JOB_SET_INPUT_Q r14                       ; src pointer for this slice
     JOB_SET_INPUT_ELEMENTS eax                ; byte_count for this slice
     mov r10d, eax                             ; save byte_count before eax is overwritten
     mov eax, MATMUL_MEMCPY                    ; job mode
     JOB_SET_MODE eax                          ; JOB_MODE = MATMUL_MEMCPY
     xor eax, eax                              ; JOB_SCALE = 0, ERMS kernel
     JOB_SET_SCALE eax                         ; store kernel selection, always ERMS
; Advance running offset and pointers by actual byte_count
     add edi, r10d                             ; running_offset += byte_count
     add r13, r10                              ; dst += byte_count, 64-bit pointer advance
     add r14, r10                              ; src += byte_count, 64-bit pointer advance
     inc rbx                                   ; i++
     inc r9d                                   ; actual_slices_filled++
     cmp ebx, r12d                             ; every slice staged?
     jl .dm_fill_loop                          ; continue to next slice
.dm_fill_done:
     mov r12d, r9d                             ; use actual filled count for dispatch
; ====== DISPATCH_EXECUTE: publish, wake, steal, wait, track ======
     mov eax, r12d                             ; eax = num_slices, job count
     DISPATCH_EXECUTE eax, dm_caller_count, dm_caller_end ; publish jobs, wake workers, steal, wait, track
.dm_done:
     vzeroupper                                ; clean dirty YMM uppers, caller-steal path ran AVX kernels on this thread; Win64: uppers volatile, vzeroupper is the project exit idiom
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; epilogue: restore non-volatiles, return
%define QA_DIRECT_THRESHOLD 64                 ; total_groups <= this -> direct quantizer, 64*64=4096 elements
%define CFG_FIX_QCG_CLAMP 128                  ; config_was_fixed bit 7: quantize_chunk_groups clamped to total_groups, too big for this shape
     global v7d_quantize                       ; exported: dispatched quantize operation
     SF_START v7d_quantize, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles, reserve frame
     SF_QWORD qa_out_q                         ; int8_t* output quantized data, stack slot
     SF_QWORD qa_in_x                          ; float* input activations, stack slot
     SF_QWORD qa_out_s                         ; float* output scales, stack slot
     SF_DWORD qa_n                             ; total element count, stack slot
     SF_DWORD qa_total_groups                  ; total groups = ceil,n/64, stack slot
     SF_DWORD qa_chunk_groups                  ; groups per dispatch chunk, stack slot
     SF_DWORD qa_num_chunks                    ; computed chunk count, stack slot
     SF_DWORD qa_caller_count                  ; caller's completed job count, stack slot
     SF_DWORD qa_caller_end                    ; caller steal loop bound, stack slot
     SF_COMMIT                                 ; finalize stack frame layout
     mov qa_out_q, rcx                         ; save out_q, int8_t* output to stack slot
     mov qa_in_x, rdx                          ; save in_x, float* input to stack slot
     mov qa_out_s, r8                          ; save out_s, float* scales to stack slot
     mov qa_n, r9d                             ; save n, total element count to stack slot
     call initialize_once                      ; ensure DLL init done, SMC patching, worker pool
     SET_FTZ_DAZ                               ; enable Flush-To-Zero and Denormals-Are-Zero
     mov eax, qa_n                             ; load n for group computation
     add eax, GS-1                             ; round up: n +, GS-1
     shr eax, QA_Q_STRIDE_SHIFT                ; divide by 64 = groups of GS floats, log2,GS
     mov qa_total_groups, eax                  ; save total groups count
     cmp eax, QA_DIRECT_THRESHOLD              ; <= 64 groups (4096 elements): bypass worker dispatch
     jbe .qa_direct_call                       ; yes, bypass dispatch, call kernel on caller thread
     mov ecx, [quantize_chunk_groups]          ; load INI-configured chunk groups, per-shape, slot 12
     test ecx, ecx                             ; 0, old "auto" doc value or negative sign bit?
     jle .qa_qcg_range_fatal                   ; yes, config range violation, FATAL, no auto impl exists
     mov qa_chunk_groups, ecx                  ; save chunk_groups
; --- Compute num_chunks = ceil,total_groups / chunk_groups ---
; ecx already holds chunk_groups, divisor; eax still holds the total, dividend
     lea eax, [eax + ecx - QA_CEIL_ADJUST]     ; round up: total + divisor - 1
     xor edx, edx                              ; clear high 32 bits for div
     div ecx                                   ; eax = ceil,total_groups / chunk_groups
     mov qa_num_chunks, eax                    ; save number of chunks
     cmp eax, MAX_JOBS                         ; num_chunks <= MAX_JOBS?
     jbe .qa_chunks_ok                         ; yes, proceed
     FATAL "v7d_quantize: num_chunks > MAX_JOBS" ; contract violation: too many chunks for job array
.qa_chunks_ok:
; --- Fill jobs_arr with quantize JOBs ---
     xor ebx, ebx                              ; i = 0, loop counter
     xor r12d, r12d                            ; group_start = 0, running group offset
     mov r15d, qa_num_chunks                   ; r15d = num_chunks, loop bound
     mov r13d, qa_chunk_groups                 ; r13d = chunk_groups, loop-invariant cache, r13 unused elsewhere in this function; eliminates 1 stack read + 1 mov per iteration
     mov eax, qa_total_groups                  ; eax = ceil,n/64, the largest group count that exists
     cmp r13d, eax                             ; chunk_groups > total_groups?, INI permits any value >= 1
     cmova r13d, eax                           ; clamp: chunk_groups = min,chunk_groups, total_groups, caps chunk_groups*64 under 2^31 so the 32-bit element count below can never wrap to 0, a zero-element job silently skips quantization ; qa_num_chunks stays computed from the UNCLAMPED value so the documented one-chunk overshoot behaviour is unchanged
     cmp r13d, ecx                             ; clamp fired?, ecx = unclamped INI chunk_groups, still live from the config load above
     je .qa_qcg_applied                        ; no, value applied as configured, no flag
     or qword [config_was_fixed], CFG_FIX_QCG_CLAMP ; flag bit: quantize_chunk_groups clamped for this shape
.qa_qcg_applied:
     mov r14d, qa_n                            ; loop-invariant n hoisted to the free register, eliminates 1 stack read per iteration, same technique as the r13d chunk_groups cache
.fill_loop:
; at least one chunk: reached only on the dispatched path, num_chunks = ceil(total_groups / chunk_groups) >= 1
; Compute job address via JOB_ADDR macro, rsi = &jobs_arr[i]
     JOB_ADDR rbx                              ; rsi = &jobs_arr[i]
     mov rax, qa_out_q                         ; rax = out_q base pointer, preserve across offset add
     mov rcx, r12                              ; rcx = group_start, 64-bit index, does NOT overwrite rax
     shl rcx, QA_Q_STRIDE_SHIFT                ; rcx = group_start * 64, byte offset into q array, 64-bit
     add rax, rcx                              ; rax = out_q + group_start*64, correct base + offset
     JOB_SET_OUTPUT rax                        ; JOB_OUTPUT = out_q pointer for this chunk
     mov rax, qa_in_x                          ; rax = in_x base pointer, preserve across offset add
     mov rcx, r12                              ; rcx = group_start, 64-bit index, does NOT overwrite rax
     shl rcx, QA_X_STRIDE_SHIFT                ; rcx = group_start * 256, 64-bit byte offset: 64 floats * 4 bytes
     add rax, rcx                              ; rax = in_x + group_start*256, correct base + offset
     JOB_SET_INPUT_Q rax                       ; JOB_INPUT_Q = in_x pointer for this chunk
     mov rax, qa_out_s                         ; load out_s base pointer
     JOB_SET_INPUT_S rax                       ; JOB_INPUT_S = out_s base, common across chunks
     mov eax, r13d                             ; eax = chunk_groups, register cache, in groups
     shl eax, QA_Q_STRIDE_SHIFT                ; eax = chunk_groups * 64 = element count per full chunk, 32-bit: safe, the clamp above caps chunk_groups at total_groups, and total_groups*64 < 2^31
     mov rdx, r12                              ; rdx = group_start, 64-bit index
     shl rdx, QA_Q_STRIDE_SHIFT                ; rdx = group_start * 64, 64-bit element offset
     mov edi, r14d                             ; edi = total element count, n, hoisted register cache, JOB_ADDR overwrites r11
     sub edi, edx                              ; edi = n - offset = remaining elements, low 32 of rdx identical < 2^31
     cmp eax, edi                              ; chunk_size > remaining?
     cmova eax, edi                            ; yes, clamp to remaining
     JOB_SET_INPUT_ELEMENTS eax                ; JOB_INPUT_ELEMENTS = n_slice for this job
     JOB_SET_FEATURE_DIMENSION r12d            ; JOB_FEATURE_DIMENSION = group_start offset
     xor eax, eax                              ; eax = 0, no per-chunk scale offset
     JOB_SET_SCALE eax                         ; JOB_SCALE = 0
     mov eax, MATMUL_QUANTIZE                  ; mode
     JOB_SET_MODE eax                          ; JOB_MODE = MATMUL_QUANTIZE
     inc rbx                                   ; i++
     add r12d, r13d                            ; group_start += chunk_groups, register cache
     cmp ebx, r15d                             ; all chunks staged?
     jl .fill_loop                             ; next chunk if any
.fill_done:
; ====== DISPATCH_EXECUTE: publish, wake, steal, wait, track ======
     mov eax, r15d                             ; eax = num_chunks, job count
     DISPATCH_EXECUTE eax, qa_caller_count, qa_caller_end ; publish jobs, wake workers, steal, wait, track
     jmp .qa_done                              ; skip the direct-call path
.qa_direct_call:
     mov rcx, qa_out_q                         ; arg1: quantized output base
     mov rdx, qa_in_x                          ; arg2: float input base
     mov r8, qa_out_s                          ; arg3: scales output base
     mov r9d, qa_n                             ; arg4: total element count, one slice covers all
     xor eax, eax                              ; group_start = 0, single slice, starts at group 0
     mov [rsp + APICALL_ARG5], rax             ; arg5: group_start, stack-passed
     call v7s_quantize_chunk                   ; direct kernel call, no dispatch overhead
.qa_done:
     vzeroupper                                ; clean dirty YMM uppers, caller-steal path ran AVX kernels on this thread; project exit idiom
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; epilogue: restore non-volatiles, return
.qa_qcg_range_fatal:
     FATAL "v7d_quantize: quantize_chunk_groups <= 0, fix threading_proxy.ini (values > total_groups are legal: one chunk)" ; INI range violation: reject, never div-by-zero or silently quantize nothing
; ==============================================================================
; Experimental: dispatched four-token Q8 batch (speed_fix plan Phase 2.3).
; One job per contiguous output-row range across the worker pool; every job
; streams its own weight-row range ONCE for all four input vectors, so the
; weight data is loaded rows/workers times instead of 4x. Job encoding:
; JOB_OUTPUT = first row index (integer semantics for this mode),
; JOB_INPUT_ELEMENTS = row count, JOB_MODE = MATMUL_Q8_BATCH4. The whole
; matrix's args block is published in the BSS q8b_args_ptr, valid for the
; whole blocking dispatch. No per-shape INI block: the geometry is a trivial
; contiguous row split, not the autotuner's split-geometry space.
; ==============================================================================
     global v7d_q8_batch4                      ; exported: dispatched four-token Q8 batch
v7d_q8_batch4:
     xor r8d, r8d                              ; Public input rows are ordinary signed bytes
     mov edx, 1                                ; Public entry describes exactly one four-token tile
     jmp v7d_q8_batch4_panel                   ; Use the shared row dispatcher
     SF_START v7d_q8_batch4_panel, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles, reserve frame
     SF_QWORD q4b_args                         ; caller's whole-matrix args block pointer
     SF_DWORD q4b_packed                       ; Input rows contain precomputed magnitudes
     SF_DWORD q4b_tiles                        ; Number of four-token tiles sharing the row ranges
     SF_DWORD q4b_num_slices                   ; slice count = min(rows, num_workers + 1), the caller steals slices too
     SF_DWORD q4b_caller_count                 ; caller's completed job count, stack slot
     SF_DWORD q4b_caller_end                   ; caller steal loop bound, stack slot
     SF_COMMIT                                 ; finalize stack frame layout
     mov q4b_packed, r8d                       ; Preserve the packed-input flag
     mov q4b_tiles, edx                        ; Preserve the panel length across initialization
     mov q4b_args, rcx                         ; save the caller's args block pointer
     call initialize_once                      ; ensure workers are ready
     SET_FTZ_DAZ                               ; enable Flush-To-Zero and Denormals-Are-Zero
; Publish the whole args block for process_job's slice builder
%define Q4B_PANEL_WEIGHT_SHIFT 14              ; Target 16 KiB of weight bytes per panel job
%define Q4B_ARGS_N_OFF 112                     ; Input elements in the shared descriptor
%define Q4B_ARGS_D_OFF 116                     ; byte offset of the dword row-count field inside v7_q8_batch4_args
     mov edx, q4b_packed                       ; Load the input layout flag
     mov [rel q8b_input_packed], edx           ; Publish the input layout for workers
     mov rax, q4b_args                         ; rax = caller's args block pointer
     mov edx, q4b_tiles                        ; Load the number of descriptors
     imul rdx, Q8_BATCH_ARGS_BYTES             ; Compute the size of the descriptor array
     add rdx, rax                              ; Compute the end of the immutable panel
     mov [rel q8b_args_end], rdx               ; Workers stop here when processing their row range
     mov [rel q8b_args_ptr], rax               ; BSS publish, valid until DISPATCH_EXECUTE returns
; Slice geometry: contiguous row ranges over the output rows
     mov r11d, [rax + Q4B_ARGS_N_OFF]          ; Weight bytes per output row
     mov eax, [rax + Q4B_ARGS_D_OFF]           ; Total output rows
     mov ecx, [num_workers]                    ; worker count, permanent invariant > 0 after init (a zero-row caller bug reaches the div and crashes visibly, Zydis catches it)
     inc ecx                                   ; + caller = participant count, the caller steals slices too
     cmp dword q4b_tiles, 1                    ; Do row slices serve more than one tile?
     jbe .q4b_limit_rows                       ; Keep the original split for one tile
     shl ecx, 1                                ; Offer at least two jobs per participant
     mov edx, eax                              ; Total output rows
     imul rdx, r11                             ; Total weight bytes
     dec rdx                                   ; Bias the ceiling division
     shr rdx, Q4B_PANEL_WEIGHT_SHIFT           ; Divide by target weight bytes per job
     inc rdx                                   ; Include a final partial weight slice
     cmp rcx, rdx                              ; Do weight slices require more jobs?
     cmovb rcx, rdx                            ; Keep the full-width count until capped
     mov edx, MAX_JOBS                         ; Capacity of the shared job array
     cmp rcx, rdx                              ; Does the requested count exceed capacity?
     cmova rcx, rdx                            ; Stay within the job array
.q4b_limit_rows:
     cmp eax, ecx                              ; rows > participants?
     ja .q4b_have_slices                       ; yes: cap the slice count at the participant count
     mov ecx, eax                              ; no: one slice per row, no idle jobs
.q4b_have_slices:
     mov q4b_num_slices, ecx                   ; save the slice count
     xor edx, edx                              ; clear high half for div
     div ecx                                   ; eax = base rows per slice, edx = remainder = extras
; Fill jobs_arr: one job per contiguous row range
     xor ebx, ebx                              ; i = 0
     xor r12d, r12d                            ; running first row of the next slice
     mov r13d, q4b_num_slices                  ; slice loop bound
     mov r14d, eax                             ; base row count from DIV
     mov r15d, edx                             ; remainder from DIV
.q4b_fill_loop:
; at least one slice: q4b_num_slices = min(rows, num_workers+1) >= 1 on a valid call
     JOB_ADDR rbx                              ; rsi = &jobs_arr[i]
     JOB_SET_OUTPUT r12                        ; JOB_OUTPUT = first row index of this range
     mov eax, r14d                             ; base rows for every slice
     test r15d, r15d                           ; extras remaining?
     jz .q4b_store_count                       ; no: this slice gets the base count
     inc eax                                   ; one extra row for the first (rows % slices) slices
     dec r15d                                  ; one fewer extra to hand out
.q4b_store_count:
     JOB_SET_INPUT_ELEMENTS eax                ; JOB_INPUT_ELEMENTS = row count
     add r12d, eax                             ; advance the running first row past this slice
     mov eax, MATMUL_Q8_BATCH4                 ; eax = the batch4 job mode constant
     JOB_SET_MODE eax                          ; JOB_MODE = batch4
     inc rbx                                   ; i++
     cmp ebx, r13d                             ; every slice staged?
     jl .q4b_fill_loop                         ; next slice
.q4b_fill_done:
; ====== DISPATCH_EXECUTE: publish, wake, steal, wait, track ======
     mov eax, r13d                             ; eax = num_slices, job count
     DISPATCH_EXECUTE eax, q4b_caller_count, q4b_caller_end ; publish jobs, wake workers, steal, wait, track
.q4b_done:
     vzeroupper                                ; clean dirty YMM uppers, caller-steal path ran AVX kernels on this thread; project exit idiom
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; epilogue: restore non-volatiles, return
%undef Q4B_PANEL_WEIGHT_SHIFT
%undef Q4B_ARGS_N_OFF
%undef q4b_packed
%undef q4b_packed_raw
%undef q4b_tiles
%undef q4b_tiles_raw
%undef Q4B_ARGS_D_OFF                          ; args-block row-count field offset, file-local
; %undef cleanup: file-local defines leave scope here --
%undef DM_NEXT_SLICE_OFF                       ; [ebx + 1] last-slice remainder offset
%undef DM_ERMS_THRESHOLD                       ; 64KB dispatch/ERMS cutover
%undef DM_ALIGN_MASK                           ; ~31 32-byte alignment mask
%undef QA_CEIL_ADJUST                          ; 1 in the ceil-division lea
%undef QA_Q_STRIDE_SHIFT                       ; log2,64 int8 group stride
%undef QA_X_STRIDE_SHIFT                       ; log2,256 float group stride
%undef QA_DIRECT_THRESHOLD                     ; single-call group cutoff
%undef CFG_FIX_QCG_CLAMP                       ; config_was_fixed bit 7
