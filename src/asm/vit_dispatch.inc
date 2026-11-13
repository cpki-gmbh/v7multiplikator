; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%include "vit_structs.inc"
%include "inf_structs.inc"
     section .text
%define VLD_BLK4_SCALE 4                       ; +4-column unit scale, chunk_d stays a multiple of 4
%define VLD_M_TAIL_MASK 3                      ; mask extracting M & 3, ragged tail the 4-unit split drops
%define VD_F4_SCALE 4                          ; float element scale for the per-head stride
%define VD_ALIGN64_BIAS 63                     ; 64-byte align bias folded into the stride LEA
     global v7d_vit_linear                     ; export ViT linear matmul dispatch entry point
v7d_vit_linear:
     SF_START v7d_vit_linear, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles, reserve frame
     SF_QWORD vld_bias                         ; bias base, or NULL
     SF_DWORD vld_nrows                        ; n_rows
     SF_DWORD vld_m                            ; M, output columns
     SF_DWORD vld_nchunks                      ; number of chunks
     SF_DWORD vld_caller_count                 ; caller job count
     SF_DWORD vld_caller_end                   ; caller batch end
     SF_COMMIT                                 ; finalize stack frame layout
; Save args: out/in/w go straight into their non-volatile register caches,
; which survive the initialize_once call below
     mov r12, rcx                              ; r12 = output base, constant
     mov r13, rdx                              ; r13 = input base, constant
     mov r15, r8                               ; r15 = weight base, constant
     mov vld_bias, r9                          ; bias base
     mov eax, [rbp + VK_ARG5]                  ; n_rows
     mov vld_nrows, eax                        ; save
     mov eax, [rbp + VK_ARG6]                  ; M
     mov vld_m, eax                            ; save
; Init thread pool
     call initialize_once                      ; ensure workers ready
     SET_FTZ_DAZ                               ; match worker FTZ/DAZ mode
; Set globals for chunk kernel
     mov eax, [rbp + VK_ARG7]                  ; K, AFTER initialize_once: the store below goes through eax, which the call overwrites
     mov [rel g_vit_k], eax                    ; store K global
     mov eax, vld_m                            ; M
     mov [rel g_vit_m], eax                    ; store M global
     mov r8d, [rel vit_min_chunk_cols]         ; min columns per chunk, tuned; >= 1 is the builder's creation-time contract
     xor edx, edx                              ; clear for div
     div r8d                                   ; eax = floor,M / min_chunk, correct cap semantics, nchunks may not exceed M/min
     test eax, eax                             ; M < min_chunk, ceiling 0 ?
     jz .vld_chunks_one                        ; yes: one chunk covers the whole layer
     mov vld_nchunks, eax                      ; tentative count = ceiling
     mov eax, [rel vit_max_chunks]             ; tuned target count, >= 1 is the builder's creation-time contract
     mov ecx, [num_workers]                    ; worker count
     inc ecx                                   ; +1, caller participates
     cmp eax, ecx                              ; target below workers+1?
     jge .vld_chunks_workers_ok                ; no, keep the tuned target, >= workers+1
     mov eax, ecx                              ; yes -> raise to workers+1
     or qword [config_was_fixed], CFG_FIX_VIT_CHUNKS ; bit 2: vit chunk count raised to workers+1
.vld_chunks_workers_ok:
     cmp eax, vld_nchunks                      ; target above the M/min ceiling?
     jle .vld_chunks_ceiling_ok                ; no, keep it
     mov eax, vld_nchunks                      ; yes -> clamp down to the ceiling
     or qword [config_was_fixed], CFG_FIX_VIT_CHUNKS ; bit 2: vit chunk count clamped to the M/min ceiling
.vld_chunks_ceiling_ok:
     mov ecx, vld_m                            ; M = output columns of the layer
     shr ecx, 2                                ; ecx = M/4, total 4-column units
     cmp eax, ecx                              ; chunk count fits the 4-column budget?
     jle .vld_nchunks_ok                       ; yes, chunk_d >= 4 guaranteed for all
     mov eax, ecx                              ; no, clamp to the budget
     or qword [config_was_fixed], CFG_FIX_VIT_CHUNKS ; bit 2: vit chunk count clamped to the M/4 budget
     test eax, eax                             ; 4-column unit budget, M>>2 exhausted to zero? only REACHABLE when the tuned vit_min_chunk_cols is 1..3 and M < 4, min_chunk >= 4 diverts to .vld_chunks_one at the ceiling test above
     jz .vld_chunks_one                        ; M < 4 with a 1..3-wide min: degenerate to ONE whole-M chunk, the tail fold rebuilds its chunk_d from M&3, the kernel's documented <4-column path ; legal nearest degeneration per the config policy, not a hot-path abort
.vld_nchunks_nonzero:
     cmp eax, MAX_JOBS                         ; count within the fixed job pool?
     jbe .vld_nchunks_ok                       ; yes, the fill below cannot overrun jobs_arr
     FATAL "v7d_vit_linear: M/4 column units exceed MAX_JOBS (4096), the fixed job pool cannot hold the chunk count"
.vld_nchunks_ok:
     mov vld_nchunks, eax                      ; save the final, possibly capped count
     mov eax, vld_m                            ; M
     shr eax, 2                                ; M / 4, total 4-column units
     xor edx, edx                              ; clear for div
     mov ecx, vld_nchunks                      ; divisor
     div ecx                                   ; eax = base4 units, edx = rem4 units
     shl eax, 2                                ; base = base4 * 4, multiple of 4
     mov r10d, vld_nchunks                     ; loop-B bound, cached, no per-iteration stack load
     mov ecx, eax                              ; aligned chunk base is still in EAX
     lea r8d, [rcx + VLD_BLK4_SCALE]           ; loop-A chunk_d = base + 4
     mov r14, vld_bias                         ; bias base, constant
     mov edi, vld_nrows                        ; n_rows, constant
     xor r9d, r9d                              ; col_start = 0, register induction var
     xor ebx, ebx                              ; chunk index i = 0, the chunk loop counter
     JOB_BASE                                  ; rsi = &jobs_arr[0], the index is known zero
     mov r11d, MATMUL_VIT_LINEAR               ; job mode, hoisted; AFTER JOB_BASE, which needs no scratch
     test edx, edx                             ; any +4 chunks?, rem may be 0
     jz .vld_fill_loop_b                       ; no, skip to base chunks
.vld_fill_loop_a:
     JOB_SET_OUTPUT r12                        ; output base, same for all jobs
     JOB_SET_INPUT_Q r13                       ; input base, same for all jobs
     JOB_SET_INPUT_S r14                       ; bias, or NULL
     JOB_SET_WEIGHT_Q r15                      ; weight base, same for all jobs
     JOB_SET_INPUT_ELEMENTS edi                ; n_rows
     JOB_SET_COLUMN_OFFSET r9d                 ; col_start for this chunk, register
     JOB_SET_FEATURE_DIMENSION r8d             ; chunk_d = base + 4, register, no reload
     JOB_SET_MODE r11d                         ; ViT linear mode, hoisted loop-invariant
     add r9d, r8d                              ; col_start += chunk_d
     JOB_NEXT rsi                              ; next job slot, the one permitted spelling
     inc ebx                                   ; i++
     cmp ebx, edx                              ; i < rem?, bottom test, no back-edge jmp
     jb .vld_fill_loop_a                       ; yes, next +4 chunk
.vld_fill_loop_b:
     JOB_SET_OUTPUT r12                        ; output base, same for all jobs
     JOB_SET_INPUT_Q r13                       ; input base, same for all jobs
     JOB_SET_INPUT_S r14                       ; bias, or NULL
     JOB_SET_WEIGHT_Q r15                      ; weight base, same for all jobs
     JOB_SET_INPUT_ELEMENTS edi                ; n_rows
     JOB_SET_COLUMN_OFFSET r9d                 ; col_start for this chunk, register
     JOB_SET_FEATURE_DIMENSION ecx             ; chunk_d = base, register, no reload
     JOB_SET_MODE r11d                         ; ViT linear mode, hoisted loop-invariant
     add r9d, ecx                              ; col_start += chunk_d
     JOB_NEXT rsi                              ; next job slot, the one permitted spelling
     inc ebx                                   ; i++
     cmp ebx, r10d                             ; i < nchunks?, cached bound; bottom test
     jb .vld_fill_loop_b                       ; yes, next base chunk
.vld_fill_tail:
; Fold the M&3 ragged tail into the FINAL chunk, index nchunks-1 >= rem,
; so it sits in the base region
     dec ebx                                   ; ebx = final chunk index, nchunks >= 1 guaranteed by the zero-ceiling divert, the .vld_chunks_one degeneration, and the builder's >= 1 min/max contract
     JOB_ADDR rbx                              ; rsi = &jobs_arr[nchunks-1], r11 scratch is dead, fill done
     JOB_LOAD_FEATURE_DIMENSION eax            ; eax = final chunk's chunk_d, = base
     mov edx, vld_m                            ; M, output columns
     and edx, VLD_M_TAIL_MASK                  ; M & 3 = leftover columns the 4-unit split dropped
     add eax, edx                              ; final chunk_d += ragged tail, kernel slow path
     JOB_SET_FEATURE_DIMENSION eax             ; patch the final job
.vld_fill_done:
; Dispatch all chunks through the central scheduler
     mov eax, vld_nchunks                      ; job count
     DISPATCH_EXECUTE eax, vld_caller_count, vld_caller_end ; publish jobs, wake workers, steal, wait, track
     vzeroupper                                ; clean AVX state
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore non-volatiles, release frame
.vld_chunks_one:
     mov eax, 1                                ; one chunk covering all M columns
     or qword [config_was_fixed], CFG_FIX_VIT_CHUNKS ; bit 2: vit M < min_chunk degenerated to a single chunk
     jmp .vld_nchunks_ok                       ; skip the three target clamps, 1 is the floor; for M >= 4 it is under the M/4 budget, and for M < 4 the tail fold rebuilds the single chunk's chunk_d from M&3
     global v7d_vit_attention                  ; export ViT attention dispatch entry point
v7d_vit_attention:
     SF_START v7d_vit_attention, rbx, rsi, r12, r13, r14, r15 ; prologue: save non-volatiles, r13 = cached VCFG_HEADS; r14/r15 needed by DISPATCH_EXECUTE; rdi unused, reserve frame
     SF_QWORD vad_attn                         ; attn output base
     SF_QWORD vad_qkv                          ; qkv buffer base
     SF_QWORD vad_scores                       ; scores scratch base
     SF_DWORD vad_n                            ; patch count
     SF_DWORD vad_caller_count                 ; caller job count
     SF_DWORD vad_caller_end                   ; caller batch end
     SF_COMMIT                                 ; finalize stack frame layout
     mov vad_attn, rcx                         ; attn output base
     mov vad_qkv, rdx                          ; qkv buffer base
     mov vad_scores, r8                        ; scores scratch
     mov vad_n, r9d                            ; patch count
     call initialize_once                      ; ensure workers ready
     SET_FTZ_DAZ                               ; match worker FTZ/DAZ mode
; Fill VCFG_HEADS jobs, one per head
     xor ebx, ebx                              ; h = 0, the head loop counter
     JOB_BASE                                  ; rsi = &jobs_arr[0], the index is known zero
; Cache loop-invariants in callee-saved registers:
; r14 = attn output base, r15 = qkv base, r13d = VCFG_HEADS
     mov r14, vad_attn                         ; attn output base, constant
     mov r15, vad_qkv                          ; qkv buffer base, constant
     mov r13d, dword [rel g_vit_cfg + VCFG_HEADS] ; r13d = number of attention heads, job count; within MAX_JOBS is the builder's contract
     mov eax, vad_n                            ; n, zero-extends rax for 64-bit math
     lea rax, [rax*VD_F4_SCALE + VD_ALIGN64_BIAS] ; n * sizeof,float + 63 for 64-align, one op
     and rax, -64                              ; clear low 6 bits = ALIGN64, S45: 64-bit math; full-width mask, no low-byte truncation
     mov r12, rax                              ; r12 = per-head stride, constant
     mov r10, vad_scores                       ; r10 = scores slice for head 0, running pointer
     mov r11d, vad_n                           ; r11d = n, patch count, constant across heads, hoisted out of loop
     mov eax, MATMUL_VIT_ATTENTION             ; job mode, loop-invariant, hoisted
     xor edx, edx                              ; edx = 0, JOB_FEATURE_DIMENSION unused, constant across heads, hoisted out of the loop
.vad_fill_loop:
; Fill job fields via register intermediaries, JOB_SET_* requires registers, not SF locals
     JOB_SET_OUTPUT r14                        ; attn output, cached, same for all heads
     JOB_SET_INPUT_Q r15                       ; qkv buffer, cached, same for all heads
     JOB_SET_INPUT_S r10                       ; unique scores pointer for this head
     JOB_SET_INPUT_ELEMENTS r11d               ; n, cached in register, no stack load per head
     JOB_SET_COLUMN_OFFSET ebx                 ; h, head index
     JOB_SET_FEATURE_DIMENSION edx             ; unused, 0, hoisted
     JOB_SET_MODE eax                          ; ViT attention mode, hoisted loop-invariant
     add r10, r12                              ; advance to next head's scores slice, 1-cycle add
     JOB_NEXT rsi                              ; next job slot, the one permitted spelling
     inc ebx                                   ; h++
     cmp ebx, r13d                             ; h < heads?, bottom test, no back-edge jmp
     jl .vad_fill_loop                         ; yes, next head
.vad_fill_done:
; Dispatch all head jobs
     mov eax, r13d                             ; eax = num heads = job count, cached
     DISPATCH_EXECUTE eax, vad_caller_count, vad_caller_end ; publish all head jobs, wake workers, steal, wait
     vzeroupper                                ; clean AVX state
     SF_END rbx, rsi, r12, r13, r14, r15       ; restore non-volatiles, incl. r13 = VCFG_HEADS cache, release frame
%undef VLD_M_TAIL_MASK                         ; mask extracting M & 3, ragged tail the 4-unit split drops
%undef VLD_BLK4_SCALE                          ; +4-column unit scale, chunk_d stays a multiple of 4
%undef VD_F4_SCALE                             ; float element scale for the per-head stride
%undef VD_ALIGN64_BIAS                         ; 64-byte align bias folded into the stride LEA
