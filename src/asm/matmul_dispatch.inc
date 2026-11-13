; v7 - part of the v7 inference engine
; Copyright, C 2026 G??rg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .text
; ============================================================================
; Named offset constants, replaces all raw [reg+N] patterns for checker compliance
; ============================================================================
; Local constants for array indexing
%define IDX_1 1                                ; Index for block 1
%define IDX_2 2                                ; Index for block 2
%define PF_SCALE 4                             ; bytes per element of a dword array: the prefetch-slot stride, AND the float scale of xout in the job-fill body, both elements are 4 bytes, so the two uses are equal by element size, not by design
%define GSS_DENOM_BIAS 2                       ; 2 * (P + 1) fold: the +2 turning P+1 into the 2P+2 GSS denominator
; Fill-strategy routing sentinels stored in mm_chunk_split_pct, snapshot at fill time
; Uniform is the ABSENCE of a sentinel: the preps store a literal 0, and the routing
; below is a two-way test, FILL_SPLIT / FILL_TAPER whose fall-through is uniform.
%define FILL_SPLIT 1                           ; two-phase split fill, chunk_split_pct == 1; legacy pct>0 collapses here
%define FILL_TAPER 2                           ; exponentially-decaying, tapered chunk fill
; Taper fixed-point constants, taper_shift is in units of 1/16 of a power of two
%define TAPER_SHIFT_MAX 112                    ; max taper_shift value, k=7.0 = 7*16
%define TAPER_FRAC_MASK 15                     ; mask for fractional bits, taper_shift & 15 = f
%define TAPER_FIXED_SHIFT 4                    ; bits for integer part: top = taper_shift >> 4
%define TAPER_MUL_SCALE 4                      ; bytes per taper_mul_table entry, dword
%define CFG_FIX_TAPER_CAP 1                    ; bit 0: taper CAP < W_min at this P/shape -> uniform fill
%define CFG_FIX_NUM_CHUNKS 2                   ; bit 1: num_chunks > D/min_chunk_cols -> capped to max_chunks
%define CFG_FIX_VIT_CHUNKS 4                   ; bit 2: vit linear M<64 chunk tail, vit dispatch clamps its chunk count
%define CFG_FIX_SPLIT_G1_COLS 8                ; bit 3: chunk_group_1_count > g1_cols -> clamped to one chunk per column
%define CFG_FIX_SPLIT_G1_POOL 16               ; bit 4: chunk_group_1_count capped to the job pool ceiling
%define CFG_FIX_SPLIT_G2_COLS 32               ; bit 5: chunk_group_2_count > g2_cols -> clamped to one chunk per column
%define CFG_FIX_SPLIT_G2_POOL 64               ; bit 6: chunk_group_2_count capped to the pool headroom left after g1
; matrix_multiplication_mode, caller stack args, rbp frame
%define MM_ARG_INPUT_ELS 48                    ; Offset for input_elements, n, 5th caller arg
%define MM_ARG_FEAT_DIM 56                     ; Offset for feature_dimension, d, 6th caller arg
; Export stub tensor layout, v7d_f32, v7d_fused_f32, v7d_swiglu, v7d_swiglu_q8
; All stubs do sub rsp,96 then build fake tensors at these offsets
%define STUB_TENSOR_INPUT_Q 56                 ; [rsp + STUB_TENSOR_INPUT_Q] = fake_input.q
%define STUB_TENSOR_INPUT_S 64                 ; [rsp + STUB_TENSOR_INPUT_S] = fake_input.s, or NULL for F32
%define STUB_TENSOR_WEIGHT_Q 80                ; Offset for fake_weight.q, within stub stack frame
%define STUB_TENSOR_WEIGHT_S 88                ; Offset for fake_weight.s, within stub stack frame
%define STUB_TENSOR_PAD 96                     ; sub rsp, 96 / add rsp, 96, padding for descriptors
; v7d_f32 / v7d_fused_f32 / v7d_swiglu / v7d_swiglu_q8, 5th caller stack arg
%define STUB_FEAT_DIM 48                       ; Offset for feature dimension d, 5th caller arg
; FUSED_MM_TAIL mode_const
; Shared tail for stub-tensor dispatch entries: fake-weight build, xout park,
; arg setup, mode dispatch, pad teardown, epilogue. Contract: fake INPUT
; tensor already built at [rsp+STUB_TENSOR_INPUT_*]; stack args n/d already
; written to [rsp+APICALL_ARG5/6]; r8 = QuantizedTensor* weight; rcx = xout
; still live ; rdx dead. Overwrites rax, rcx, rdx, r8, r9.

%macro FUSED_MM_TAIL 1
     mov rax, [r8]                             ; rax = weight->quantized_data
     mov [rsp + STUB_TENSOR_WEIGHT_Q], rax     ; fake_weight.q = weight q data
     mov rax, [r8 + TENSOR_S_OFF]              ; rax = weight->scales
     mov [rsp + STUB_TENSOR_WEIGHT_S], rax     ; fake_weight.s = weight scales
     mov rdx, rcx                              ; xout -> rdx, 2nd arg
     lea r8, [rsp + STUB_TENSOR_INPUT_Q]       ; &fake_input -> r8, 3rd arg
     lea r9, [rsp + STUB_TENSOR_WEIGHT_Q]      ; &fake_weight -> r9, 4th arg
     mov ecx, %1                               ; mode, 1st arg
     call matrix_multiplication_mode           ; dispatch to mode handler
     add rsp, STUB_TENSOR_PAD                  ; remove extra padding, pairs with the entry's sub
     SF_END                                    ; epilogue: restore frame and return
%endmacro

%macro STUB_PASS_ND 0
     mov [rsp + APICALL_ARG5], r9d             ; 5th arg = n, callee reads dword
     mov eax, [rbp + STUB_FEAT_DIM]            ; d, 32-bit read, zero-extends
     mov [rsp + APICALL_ARG6], eax             ; 6th arg = d, callee reads dword
%endmacro
     MATMUL_WRAPPER v7d_q8, MATMUL_Q8          ; Q8xQ8 matmul wrapper
     SF_START v7d_f32                          ; F32xQ8 matmul prologue
     SF_COMMIT                                 ; Commit stack frame
     sub rsp, STUB_TENSOR_PAD                  ; Extra padding for fake tensor descriptors, up to rsp+STUB_TENSOR_PAD-1
     STUB_PASS_ND                              ; forward n, r9d + d to the callee's stack args
     mov rdx, [rdx]                            ; dereference: rdx = xt->x, float* activations
     mov [rsp + STUB_TENSOR_INPUT_Q], rdx      ; fake_input.q = x
     mov qword [rsp + STUB_TENSOR_INPUT_S], 0  ; fake_input.s = NULL
     FUSED_MM_TAIL MATMUL_F32                  ; weight build + arg setup + dispatch + epilogue
     MATMUL_WRAPPER matrix_multiplication, MATMUL_Q8 ; Compatibility alias wrapper
; ============================================================================
; matrix_multiplication_mode,int mode, float* xout, quantized_tensor* input,
; quantized_tensor* weight, int input_elements, int feature_dimension
; C++ ref: matmul_refactor_v7.cpp lines 793-979
; Win64: rcx=mode, rdx=xout, r8=input, r9=weight, [rsp + APICALL_ARG5]=input_elements, [rsp + APICALL_ARG6]=feature_dimension

; SF_* named slots:
; mm_xout qword - xout base pointer
; mm_input qword - input tensor ptr
; mm_weight qword - weight tensor ptr
; mm_weight_s qword - weight->scales, cached once before fill; rbx is a
; working register in every fill prep, so weight->s
; lives in this slot instead of a register
; mm_input_elements dword - input_elements, n
; mm_feature_dimension dword - feature_dimension, d
; mm_chunk_count dword - chunk_count_local
; mm_base dword - feature_dimension / chunk_count_local
; mm_rem dword - feature_dimension % chunk_count_local
; mm_caller_count dword - jobs caller processed
; mm_caller_end dword - end bound for caller job loop
; mm_mode dword - MATMUL_Q8 or MATMUL_F32

; Register allocation in caller steal loop:
; r14d = caller_batch, non-volatile, permanent during steal
; r15d = dispatched_count, non-volatile, permanent during steal
; ============================================================================
     SF_START matrix_multiplication_mode, rbx, rsi, rdi, r12, r13, r14, r15 ; Prologue, save non-volatiles
     SF_QWORD mm_xout                          ; xout base pointer
     SF_QWORD mm_input                         ; input tensor ptr
     SF_QWORD mm_weight                        ; weight tensor ptr
     SF_QWORD mm_weight_s                      ; weight->scales cached before fill, loaded into rbx at each fill entry
     SF_DWORD mm_input_elements                ; input_elements, n
     SF_DWORD mm_feature_dimension             ; feature_dimension, d
     SF_DWORD mm_chunk_count                   ; chunk_count_local
     SF_DWORD mm_base                          ; feature_dimension / chunk_count_local
     SF_DWORD mm_rem                           ; feature_dimension % chunk_count_local
     SF_DWORD mm_caller_count                  ; jobs caller processed
     SF_DWORD mm_caller_end                    ; end bound for caller job loop
     SF_DWORD mm_mode                          ; MATMUL_Q8 or MATMUL_F32
; Two-phase chunk split temps
     SF_DWORD mm_g1_chunk_dim                  ; group 1 chunk dimension
     SF_DWORD mm_g1_rem                        ; group 1 remainder
     SF_DWORD mm_g2_chunk_dim                  ; group 2 chunk dimension
     SF_DWORD mm_g2_rem                        ; group 2 remainder
     SF_DWORD mm_chunk_split_pct               ; snapshot of chunk_split_pct, race-safe for fill loop
     SF_DWORD mm_taper_mul                     ; precomputed 16.16 fixed-point multiplier for IMUL decay
     SF_DWORD mm_align_div                     ; 1 = W_min not a power of two -> div-based AlignDown, e.g. head_dim 96
     SF_COMMIT                                 ; Commit stack frame
%ifdef ZYDIS_TRACE
     INT3_DBG_STR "matrix_multiplication_mode entry" ; Debug trace
     INT3_DBG_REG 1                            ; RCX = mode
     INT3_DBG_REG 2                            ; RDX = xout
     INT3_DBG_REG 8                            ; R8 = input
     INT3_DBG_REG 9                            ; R9 = weight, n and d are stack args 5 and 6, read from the frame slots below
; Mode needs no validation here, the v7d_* entry points pass a compile-time
; MATMUL_* constant. It is parked in mm_mode because rcx is a scratch from the div on.
%endif
; Save entry args for matrix_multiplication_mode
; Win64: rcx=mode, rdx=xout, r8=input, r9=weight, [rsp + APICALL_ARG5]=n, [rsp + APICALL_ARG6]=d
     mov mm_mode, ecx                          ; mode, MATMUL_Q8 or MATMUL_F32
     mov mm_xout, rdx                          ; xout
     mov mm_input, r8                          ; input
     mov mm_weight, r9                         ; weight
     mov eax, [rbp + MM_ARG_INPUT_ELS]         ; n, input_elements from stack arg 5
     mov mm_input_elements, eax                ; save input_elements, n
     mov eax, [rbp + MM_ARG_FEAT_DIM]          ; d, feature_dimension from stack arg 6
     mov mm_feature_dimension, eax             ; save feature_dimension, d
; Init thread pool, runs once via CAS guard, must run before any path
; that calls v7s_q8_chunk, every Q8 dispatch needs the SMC patch applied
     call initialize_once                      ; ensure thread pool & SMC patching initialized
     SET_FTZ_DAZ                               ; Enable FTZ+DAZ, denormals-to-zero
%ifdef PROFILE_MODE
     mov rax, [caller_jobs_done]               ; caller jobs of the closing period
     add [caller_jobs_total], rax              ; sticky session total, every dispatch type contributes before its reset
     xor eax, eax                              ; zero register for reset
     mov [caller_jobs_done], rax               ; reset caller jobs done counter
     mov [caller_fetch_adds], rax              ; reset caller fetch_add counter
     mov [caller_spins_wait], rax              ; reset caller spin-wait counter
%ifdef ZYDIS_TRACE
     INT3_DBG_STR "mm: reset_instr done"       ; Debug trace
%endif
     lea rbx, [rel worker_jobs_done]           ; base addr of worker_jobs_done array
     lea rdi, [rel worker_fetch_adds]          ; base addr of worker_fetch_adds array
     lea r12, [rel worker_wake_spins]          ; base addr of worker_wake_spins array
     lea r13, [rel worker_sleeps]              ; base addr of worker_sleeps array
     xor edx, edx                              ; the zero for the four stores below, rdx is dead here and the loop makes no calls
     xor r14d, r14d                            ; i = 0
     lea r15, [rel worker_jobs_peak]           ; peak array base, dead after this loop, DISPATCH_EXECUTE re-arms r15d itself
.reset_instr:
     cmp r14d, [num_workers]                   ; i < num_workers?
     jge .instr_done                           ; done if all workers reset
     mov eax, r14d                             ; zero-extend to 64-bit for the stores below
     shl eax, 6                                ; rax = id * 64, u32 exact: id < MAX_WORKERS, cache-line stride
     mov rcx, [rbx + rax]                      ; this worker's job count from the closing period
     cmp rcx, [r15 + rax]                      ; a new per-period high-water mark?
     jbe .peak_keep                            ; no, the peak already stands
     mov [r15 + rax], rcx                      ; yes, remember it before the reset wipes the period count
.peak_keep:
     mov qword [rbx + rax], rdx                ; Worker[i].jobs_done = 0
     mov [rdi + rax], rdx                      ; Worker[i].fetch_adds = 0
     mov [r12 + rax], rdx                      ; Worker[i].wake_spins = 0
     mov [r13 + rax], rdx                      ; Worker[i].sleeps = 0
     inc r14d                                  ; i++
     jmp .reset_instr                          ; continue reset loop
.instr_done:
%endif
; --- Level 1 cache: parameter block ---
     mov eax, mm_input_elements                ; load n from frame slot
     cmp eax, [cached_shape_n]                 ; same n as last dispatch?
     jne .config_change                        ; no -> cold path
     mov eax, mm_feature_dimension             ; load d from frame slot
     cmp eax, [cached_shape_d]                 ; same d as last dispatch?
     je .warm_tune_gate                        ; warm path: skip the rescan/copy, but the turn flip must still run
.config_change:
     mov eax, mm_input_elements                ; load n
     mov [cached_shape_n], eax                 ; update the level-1 shape cache, n
     mov ebx, eax                              ; ebx = n, non-volatile across API calls
     mov eax, mm_feature_dimension             ; load d
     mov [cached_shape_d], eax                 ; update the level-1 shape cache, d
     mov r12d, eax                             ; r12d = d, save for prefetch cache check
     mov ecx, eax                              ; ecx = d, for comparison
     shl rcx, 32                               ; f64: pack d into upper 32 bits for fused qword compare
     or rcx, rbx                               ; f64: rcx = [d | n] packed compare value
     lea r15, [shape_blocks + IDX_1*SHAPE_BLOCK_SIZE] ; r15 = block 1, default fallback
     lea r14, [shape_blocks + IDX_2*SHAPE_BLOCK_SIZE] ; r14 = start of block 2
     lea rax, [shape_blocks + SHAPE_BLOCKS_TOTAL] ; one-past-last block, BSS end bound, loop-invariant
.scan_loop:
     cmp dword [r14], 0                        ; n==0?, unpopulated block = end of list
     je .scan_done                             ; yes, stop scanning, no match, use default
     cmp [r14], rcx                            ; f64: compare block_n + block_d as single qword
     jne .next_block                           ; no, try next
; Match found! Use this block instead of default
     mov r15, r14                              ; r15 = matched block pointer
     jmp .scan_done                            ; stop scanning
.next_block:
     add r14, SHAPE_BLOCK_SIZE                 ; advance to next block
     cmp r14, rax                              ; cursor past the block array?
     jb .scan_loop                             ; no: continue scanning
.scan_done:
; [default], block 1 is never a legal dispatch source. Every shape
; the engine dispatches must carry its own per-shape [NxD] config block --
; falling through to defaults would silently run the shape on unrelated
; parameters, sentineling tuning and benchmarks alike. r15 was preset to block 1
; before the scan, so equality here means the scan found nothing.
     lea rax, [shape_blocks + IDX_1*SHAPE_BLOCK_SIZE] ; block 1 address, the preset
     cmp r15, rax                              ; scan matched a per-shape block?
     jne .scan_matched                         ; yes: proceed with the copy
     WRITE_UINT64 rbx                          ; print the missing shape's n, rbx = zero-extended mm_input_elements
     WRITE_CHR 'x'                             ; separator: render the pair as its NxD INI-section form
     WRITE_UINT64 r12                          ; print the missing shape's d, r12 = zero-extended mm_feature_dimension
     WRITE_NL                                  ; end the resolution line before the FATAL text
     FATAL "dispatch: no per-shape config block for this shape (resolution printed above), add its [NxD] INI section and rebuild the DB"
.scan_matched:
     mov [rel tune_scanned_block], r15         ; cached shape match independent of tuner selection
; Copy matched block params to block 0, active via rep movsq
     lea rsi, [r15 + SHAPE_BLOCK_PARAM_OFF]    ; source block params
     lea rdi, [param_set_0]                    ; dest = active set, block 0 params
     mov ecx, PARAM_SET_QWORDS                 ; 12 qwords = 96 bytes
     MEMORY_COPY_QWORD                         ; copy params -> block 0
     mov dword [rel cycle_tuner_installed_slot], -1 ; block 0 now holds this shape's INI config: the delay
; phase must reinstall its genome before it counts
; Snapshot param_set_0 to debug buffer after per-shape block copy
     lea rsi, [param_set_0]                    ; source = block 0, active params
     lea rdi, [rel debug_param_set_0]          ; dest = BSS debug array
     mov ecx, PARAM_SET_QWORDS                 ; 12 qwords
     MEMORY_COPY_QWORD                         ; copy active params to debug buffer
; ---- active-shape registry: record every TUNABLE shape that dispatches ----
; The cycle tuner steps one machine per registered shape, so the registry
; only ever holds shapes that REALLY dispatch in the loaded model, the
; embedding-gather orientation "151936x1024", a row gather, never a matmul
; is absent from the registry and can never be staged, stalled on, or
; rotated to. Minimal scan per the block-header packing: one qword compare
; per entry against [d | n].
; every dispatching shape matched a per-shape block, a
; [default] fallback FATALs above, so registration always runs here.
; The 64-slot registry is a DESIGNED CEILING, not a guarded buffer: the
; shipped shape universe is 48 INI sections / ~45 supported resolutions,
; far below 64, and distinct dispatching shapes can only come from that
; universe, the walk is intentionally unbounded, zero sentinel IS the
; count because a bound check would cost the cold path instructions for a
; state that cannot occur. Fast code: no capacity check, by intent.
     mov ecx, r12d                             ; d into the low half, zero-extends
     shl rcx, 32                               ; d into the high half: packed [d | n]
     or rcx, rbx                               ; rcx = packed [d | n], same layout as the block headers
     lea rdx, [rel g_active_shapes]            ; registry base, 64 packed qwords, zero-terminated
.scan_active_next:
     mov rax, [rdx]                            ; registry entry
     cmp rax, rcx                              ; this shape already registered?
     je .scan_active_done                      ; yes -> stop
     test rax, rax                             ; zero sentinel = end of list, the only termination
     jz .scan_active_append                    ; yes -> append here
     add rdx, 8                                ; next entry
     jmp .scan_active_next                     ; walk until the zero sentinel
.scan_active_append:
     mov [rdx], rcx                            ; append, the next entry is still zero -> sentinel moves forward
     lea rax, [rel shape_blocks]               ; rax = block array base
     mov r8, r15                               ; r8 = matched block pointer
     sub r8, rax                               ; r8 = byte offset of the matched block
     shr r8, 9                                 ; r8 = the matched block's index
     call cycle_tuner_register_shape           ; rcx = packed entry, rdx = entry ptr, r8d = block index: seeds the machine
.scan_active_done:
     jmp .tune_off                             ; cold path joins the warm path after its copy
.warm_tune_gate:
     mov ebx, mm_input_elements                ; the tuner-off tail expects ebx = n, the cold path set it at .config_change
     mov r12d, mm_feature_dimension            ; ...and r12d = d, prefetch cache compare
     mov r15, [rel tune_scanned_block]         ; re-pin the actual scanned block
.tune_off:
     cmp ebx, [cached_prefetch_n]              ; ebx = n, matches prefetch n?
     jne .patch_prefetch                       ; no -> patch prefetch
     cmp r12d, [cached_prefetch_d]             ; r12d = d, matches prefetch d?
     je .config_done                           ; yes -> skip SMC patching
.patch_prefetch:
; Use block 0 as the authoritative installed-byte mirror. The first pass is
; forced because initial zero-valued mirrors do not mean the kernel is NOPped.
     lea r14, [rel shape_blocks]               ; block 0: current installed prefetch configuration
     xor r13d, r13d                            ; any instruction bytes actually modified? initially no
     xor edi, edi                              ; clear all 32 flag bits, initial full-patch flag
     cmp dword [rel cached_prefetch_n], 0      ; first shape ever installed?
     sete dil                                  ; flag = 1 iff cached_prefetch_n == 0: first install must patch all 41 slots
.pf_cache_initialized:
; All slots 0..38 have a 9-byte instruction and a displacement at byte 5.
; PREFETCH_SLOT labels immediately follow the instruction, making the 4-byte
; displacement the four bytes immediately before .pfN. No table walk, calls,
; complete-instruction restore or opcode rewrite is needed in enabled->enabled
; transitions. The two attention slots (39,40) have no displacement.

%macro PF_WRITE_DISP_DIRECT 1
%if %1 < 8
     mov dword [rel v7s_q8_chunk.pf%1 - PF_DISP_END_BACK], eax ; write only the Q8 instruction's disp32
%elif %1 < 11
     mov dword [rel v7s_f32_chunk.pf%1 - PF_DISP_END_BACK], eax ; write only the F32 instruction's disp32
%elif %1 < 13
     mov dword [rel v7s_fused_f32_chunk.pf%1 - PF_DISP_END_BACK], eax ; write only the fused-F32 disp32
%elif %1 < 27
     mov dword [rel swk_kernel.pf%1 - PF_DISP_END_BACK], eax ; write only the SwiGLU-F32 disp32
%elif %1 < 39
     mov dword [rel v7s_swiglu_q8_chunk.pf%1 - PF_DISP_END_BACK], eax ; write only the SwiGLU-Q8 disp32
%endif
%endmacro

%macro PATCH_PREFETCH_DIFF_SLOT 1
     mov eax, [r15 + SHAPE_BLOCK_PF_DIST_OFF + %1*PF_SCALE] ; new distance
     test edi, edi                             ; first installation is not represented by block 0
     jnz %%full_restore                        ; perform restore-and-patch on the initial pass
     cmp eax, [r14 + SHAPE_BLOCK_PF_DIST_OFF + %1*PF_SCALE] ; distance unchanged?
     jne %%distance_changed                    ; different distance, inspect enabled state
     mov eax, [r15 + SHAPE_BLOCK_PF_TYPE_OFF + %1*PF_SCALE] ; new type
     cmp eax, [r14 + SHAPE_BLOCK_PF_TYPE_OFF + %1*PF_SCALE] ; type unchanged too?
     je %%next                                 ; identical configuration, no mirror or code write
     cmp dword [r14 + SHAPE_BLOCK_PF_DIST_OFF + %1*PF_SCALE], 32 ; code currently NOPped?
     jb %%mirror_type                          ; type change has no effect while disabled
     mov ecx, %1                               ; slot for the type setter
     mov edx, eax                              ; new type
     call set_prefetch_type_fast               ; type-only change, one patching call
     mov r13d, 1                               ; actual executable-code modification
%%mirror_type:
     mov eax, [r15 + SHAPE_BLOCK_PF_TYPE_OFF + %1*PF_SCALE] ; reload after caller-clobbering call
     mov [r14 + SHAPE_BLOCK_PF_TYPE_OFF + %1*PF_SCALE], eax ; mirror the new type
     jmp %%next                                ; finish this slot
%%distance_changed:
     cmp eax, 32                               ; new distance disables the prefetch?
     jb %%to_disabled                          ; yes, distinguish enabled->disabled from NOP->NOP
     cmp dword [r14 + SHAPE_BLOCK_PF_DIST_OFF + %1*PF_SCALE], 32 ; was it disabled?
     jb %%full_restore                         ; yes: restore instruction before enabling
%if %1 < 39
     PF_WRITE_DISP_DIRECT %1                   ; enabled->enabled: direct four-byte store
     mov r13d, 1                               ; actual executable-code modification
%endif
     mov eax, [r15 + SHAPE_BLOCK_PF_TYPE_OFF + %1*PF_SCALE] ; new type
     cmp eax, [r14 + SHAPE_BLOCK_PF_TYPE_OFF + %1*PF_SCALE] ; same opcode/ModRM selection?
     je %%mirror_both                          ; yes, skip type setter entirely
     mov ecx, %1                               ; slot for type setter
     mov edx, eax                              ; new type
     call set_prefetch_type_fast               ; type patch without restoring the instruction
     mov r13d, 1                               ; actual executable-code modification
     jmp %%mirror_both                         ; update both mirror fields
%%to_disabled:
     cmp dword [r14 + SHAPE_BLOCK_PF_DIST_OFF + %1*PF_SCALE], 32 ; already a NOP?
     jb %%mirror_both                          ; NOP->NOP: no executable bytes change at all
     mov ecx, %1                               ; slot index
     mov edx, eax                              ; disabled distance
     call set_prefetch_distance_fast           ; enabled->disabled, replace with the NOP template
     mov r13d, 1                               ; actual executable-code modification
     jmp %%mirror_both                         ; no type setter when new instruction is NOP
%%full_restore:
     mov ecx, %1                               ; slot index
     mov edx, eax                              ; new distance
     call set_prefetch_distance_fast           ; first install or disabled->enabled
     mov r13d, 1                               ; actual executable-code modification
     cmp dword [r15 + SHAPE_BLOCK_PF_DIST_OFF + %1*PF_SCALE], 32 ; new code NOPped?
     jb %%mirror_both                          ; if disabled, no opcode/type modification needed
     mov ecx, %1                               ; slot index after volatile clobbers
     mov edx, [r15 + SHAPE_BLOCK_PF_TYPE_OFF + %1*PF_SCALE] ; new type
     call set_prefetch_type_fast               ; restore defaults then select type
%%mirror_both:
     mov eax, [r15 + SHAPE_BLOCK_PF_DIST_OFF + %1*PF_SCALE] ; requested distance
     mov [r14 + SHAPE_BLOCK_PF_DIST_OFF + %1*PF_SCALE], eax ; mirror the distance
     mov eax, [r15 + SHAPE_BLOCK_PF_TYPE_OFF + %1*PF_SCALE] ; requested type
     mov [r14 + SHAPE_BLOCK_PF_TYPE_OFF + %1*PF_SCALE], eax ; mirror the type
%%next:
%endmacro
%assign PF_SLOT 0                              ; generate a straight-line 41-slot differential installer
%rep 41
     PATCH_PREFETCH_DIFF_SLOT PF_SLOT          ; patch this slot only if necessary
%assign PF_SLOT PF_SLOT+1                      ; advance to the next prefetch slot
%endrep
%unmacro PATCH_PREFETCH_DIFF_SLOT 1            ; retire the helper: the unrolled pass above is its only user
%unmacro PF_WRITE_DISP_DIRECT 1                ; retire the direct-store helper with it
     test r13d, r13d                           ; did any executable bytes change?
     jz .pf_batch_clean                        ; no: zero serialization work
     call smc_prefetch_serialize               ; one CPUID per changed batch, none if code bytes unchanged
.pf_batch_clean:
     mov [cached_prefetch_n], ebx              ; mark the mirrored configuration installed for this shape
     mov [cached_prefetch_d], r12d             ; cache d only after installation completes
.config_done:
; ====== END PER-SHAPE CONFIG SELECTION ======
; ====== the in-process cycle tuner: advance the machine, stage the timer ======
; The flip runs AFTER the config selection: .patch_prefetch has either just
; patched all 41 slots from the matched block or the cache already holds
; this shape's full configuration, so the install's cache stamp stays
; truthful. The install re-patches the q8-owned slots 0-7 and the
; swiglu_q8-owned slots 27-38 from the step genome on top of that full set.
; A newly registered shape loads its population immediately, including
; on the first dispatch made by a tool or the first chat turn.
     cmp dword mm_mode, MATMUL_Q8              ; the tuner tunes the q8 kernel family only
     je .tune_eligible                         ; plain Q8: an F32 dispatch advances no machine, measures nothing it patched
     cmp dword mm_mode, MATMUL_SWIGLU_Q8       ; fused Q8 SwiGLU shares the registry shapes and the Q8 population
     jne .tune_skip                            ; every other mode advances no machine
.tune_eligible:
     mov rcx, r15                              ; rcx = matched block pointer
     lea rax, [rel shape_blocks]               ; rax = block array base
     sub rcx, rax                              ; rcx = byte offset of the matched block
     shr rcx, 9                                ; rcx = the matched block's index
     call cycle_tuner_flip                     ; advance the machine, install the step genome
     cmp dword [rel cycle_tuner_step_live], 0  ; did THIS dispatch stage an episode step?
     je .tune_skip                             ; no -> normal dispatch, no timer
     lfence                                    ; fence pending memory ops before the
; TSC read so the timer cannot lap pipeline work
     rdtsc                                     ; timer start, rax/rdx dead here
     lfence                                    ; and none after it laps the timing edge, LFENCE;RDTSC;LFENCE, the documented invariant-TSC measurement idiom
     shl rdx, 32                               ; move high TSC half into position
     or rdx, rax                               ; rdx = packed 64-bit timestamp
     mov [rel cycle_tuner_tsc_start], rdx      ; start of the timed step
.tune_skip:
; ====== End in-process cycle tuner injection ======
     mov ebx, [num_chunks]                     ; chunk_count_local = num_chunks
     mov eax, mm_feature_dimension             ; feature_dimension
     mov ecx, [min_chunk_cols]                 ; min_chunk_cols from INI, default 16
     test ecx, ecx                             ; min_chunk_cols <= 0?, signed compare, also rejects the negatives a zero-only test let through, and whose div ecx would sign-flip
     jle .min_chunk_cols_bad                   ; misconfigured INI -> FATAL
     xor edx, edx                              ; clear for div
     div ecx                                   ; eax = max_chunks = feature_dimension / min_chunk_cols
     cmp ebx, eax                              ; chunk_count_local > max_chunks?
     jle .chunk_count_capped                   ; no, within the per-dispatch legal count
     mov ebx, eax                              ; yes, degrade to the nearest legal chunk count, max_chunks
     or qword [config_was_fixed], CFG_FIX_NUM_CHUNKS ; flag bit 1: num_chunks engine-fixed for this system/shape
.chunk_count_capped:
; Cap chunk_count_local to MAX_JOBS to prevent jobs_arr buffer overflow.
; Runs on EVERY path: num_chunks <= max_chunks does NOT imply <= MAX_JOBS
; max_chunks = d/min_chunk_cols can dwarf the job pool on wide shapes .
     cmp ebx, MAX_JOBS                         ; chunk_count > MAX_JOBS?, immediate cmp, no reg load needed
     jle .chunk_count_ok                       ; no, within bounds
     FATAL "dispatch: chunk_count > MAX_JOBS"  ; Fatal error: too many jobs
.min_chunk_cols_bad:
     FATAL "dispatch: min_chunk_cols < 1, fix per-shape INI" ; Fatal error: invalid INI config
.chunk_count_ok:
     mov mm_chunk_count, ebx                   ; save clamped chunk_count_local
     test ebx, ebx                             ; chunk_count_local == 0?, feature_dimension < min_chunk_cols
     jnz .chunk_count_fill                     ; no: proceed with fill
     FATAL "dispatch: chunk_count == 0 (feature_dimension < min_chunk_cols)"
.chunk_count_fill:
     mov r12, mm_xout                          ; r12 = xout base, float*
     mov rax, mm_input                         ; rax = input tensor ptr, quantized_tensor*
     mov r14, [rax]                            ; r14 = input->quantized_data
     mov r13, [rax + TENSOR_S_OFF]             ; r13 = input->scales
     mov rax, mm_weight                        ; rax = weight tensor ptr, quantized_tensor*
     mov rdi, [rax]                            ; rdi = weight->quantized_data, cached across fill
     mov rax, [rax + TENSOR_S_OFF]             ; rax = weight->scales
     mov mm_weight_s, rax                      ; cache weight->s in frame slot
     xor r8d, r8d                              ; column_offset = 0
     xor ecx, ecx                              ; i = 0, looped_count
     mov r15d, mm_input_elements               ; r15d = input_elements, n saved across div
     JOB_BASE                                  ; rsi = first descriptor; fill routes advance it
     mov r11d, mm_mode                         ; cache job mode; fill preps preserve r11
; Choose fill strategy
     cmp dword mm_mode, MATMUL_FUSED_F32       ; fused mode?
     je .fused_fill_prep                       ; yes -> use head-aligned chunking, skip split
     mov eax, [chunk_split_pct]                ; snapshot global for race-safe fill loop
     mov mm_chunk_split_pct, eax               ; save to stack, mode cannot change mid-fill
     test eax, eax                             ; two-phase split enabled?
     jnz .split_fill_prep                      ; yes, use two-phase fill
; Either uniform or taper, depends on taper_shift INI value
     cmp dword [taper_shift], 0                ; k=0 -> uniform; k>0 -> tapered decay, read only for the zero test
     jnz .taper_fill_entry                     ; k>0 -> exponentially-decaying chunks
; ====== Uniform fill, k=0: legacy behavior, kept for fallback/measurement ======
; Compute base and rem for even distribution
; C++ ref lines 873-874
.uniform_fill_prep:
     mov eax, mm_feature_dimension             ; feature_dimension
     xor edx, edx                              ; clear edx for div
     div ebx                                   ; eax = base, edx = rem
     mov mm_base, eax                          ; base
     mov mm_rem, edx                           ; rem
     jmp .fill_jobs_entry                      ; skip fused prep
.fused_fill_prep:
; Head-aligned chunking: each chunk covers one or more complete heads.
; Every job's column_offset and chunk_dim are multiples of head_dim.
; Taper check: if taper_shift > 0, use tapered fused fill, W_min = head_dim ; else uniform.
     cmp dword [taper_shift], 0                ; k=0 -> uniform fused; k>0 -> tapered fused, read only for the zero test
     jnz .taper_fused_prep                     ; yes, branch to taper-specific setup, skip uniform fused
%ifdef ZYDIS_TRACE
     mov eax, [num_workers]                    ; num_workers
     INT3_DBG_STR "nw="                        ; Debug trace: num_workers
     INT3_DBG_REG 0                            ; eax = num_workers, loaded at the mov above
     mov eax, mm_feature_dimension             ; d_total
     INT3_DBG_STR "dt="                        ; Debug trace: d_total
     INT3_DBG_REG 0                            ; d_total
%endif
     mov eax, [fused_head_dim]                 ; head dimension from global
%ifdef ZYDIS_TRACE
     INT3_DBG_STR "hd="                        ; Debug trace: head_dim, traced after the load, so eax really holds it
     INT3_DBG_REG 0                            ; head_dim
%endif
.fused_geom:
     mov ecx, eax                              ; ecx = head_dim, divisor for n_heads
     mov eax, mm_feature_dimension             ; feature_dimension, d
     xor edx, edx                              ; clear for div
     div ecx                                   ; eax = n_heads, edx = remainder, d % head_dim == 0 is the import's weight contract
; Chunks = min(n_heads, num_workers + 1), the caller is a participant
; Use the participant count P so every P-th processor finds work; each chunk
; still spans multiple heads, the kernel processes them in its head loop
     mov ebx, [num_workers]                    ; worker count
     inc ebx                                   ; + caller = participant count P
     cmp eax, ebx                              ; n_heads vs P
     jge .fused_chunks_ok                      ; n_heads >= P -> keep P chunks
     mov ebx, eax                              ; n_heads < P -> cap chunks to n_heads
.fused_chunks_ok:
     mov mm_chunk_count, ebx                   ; save capped chunk count
%ifdef ZYDIS_TRACE
     push rax                                  ; save n_heads
     mov eax, ebx                              ; final chunk_count
     INT3_DBG_REG 0                            ; chunk_count
     pop rax                                   ; restore n_heads
%endif
; base =, n_heads / chunks * head_dim, rem_heads = n_heads % chunks
     xor edx, edx                              ; clear for div
     div ebx                                   ; eax = heads_per_chunk, edx = rem_heads
%ifdef ZYDIS_TRACE
     push rax                                  ; save heads_per_chunk
     mov eax, edx                              ; rem_heads
     INT3_DBG_REG 0                            ; rem_heads
     pop rax                                   ; restore heads_per_chunk
%endif
     imul eax, ecx                             ; base_cols = heads_per_chunk * head_dim
%ifdef ZYDIS_TRACE
     push rax                                  ; save base_cols
     INT3_DBG_REG 0                            ; base_cols
     pop rax                                   ; restore base_cols
%endif
     mov mm_base, eax                          ; save base columns
     mov mm_rem, edx                           ; save rem_heads, chunks that get +head_dim
     xor r8d, r8d                              ; column_offset = 0
     xor ecx, ecx                              ; i = 0
     mov dword mm_chunk_split_pct, ecx         ; fused uses uniform fill loop, not split
     jmp .fill_jobs_entry                      ; use uniform fill loop, extra=+head_dim for fused
.taper_fill_entry:
     mov eax, [min_chunk_cols]                 ; configured W_min, power of two
     cmp eax, 4                                ; configured W_min < Q8 4-row block?
     jge .taper_fill_common                    ; no, use configured value
     FATAL "taper min_chunk_cols < 4 contradicts Q8 4-row kernel block, fix INI"
; --- Fused taper setup, W_min from fused_head_dim global ---
.taper_fused_prep:
     mov eax, [fused_head_dim]                 ; head_dim for this model, power of two, e.g. 128
; --- Common tail: complete stack-slot init shared by fused and non-fused ---
.taper_fill_common:
     mov mm_rem, eax                           ; W_min
     blsr ecx, eax                             ; ecx = W_min & (W_min-1), nonzero iff not a power of two
     mov mm_align_div, ecx                     ; 0 = mask align, pow2, nonzero = div align, head_dim 96 etc.
     neg eax                                   ; align_mask = -W_min, two's complement
     mov mm_g1_chunk_dim, eax                  ; align_mask, for & masking
     mov eax, [num_workers]                    ; worker thread count, invariant > 0
     lea ecx, [rax + rax + GSS_DENOM_BIAS]     ; ecx = 2 * (P + 1) = 2P + 2, GSS denominator, one LEA
     mov eax, mm_feature_dimension             ; D, total column count
     xor edx, edx                              ; clear high half of 64-bit dividend
     div ecx                                   ; eax = D /, 2P, floor
     cmp dword mm_align_div, 0                 ; W_min not a power of two, e.g. head_dim 96 ?
     jne .taper_cap_div                        ; yes: div/imul AlignDown of the RAW value
     and eax, mm_g1_chunk_dim                  ; pow2: mask IS exact AlignDown
     jmp .taper_cap_aligned                    ; rejoin with the aligned CAP
.taper_cap_div:
     xor edx, edx                              ; clear high half of dividend
     div dword mm_rem                          ; eax =, D/,2P / W_min, edx = remainder
     imul eax, mm_rem                          ; CAP = quotient * W_min, true AlignDown of the raw value
.taper_cap_aligned:
     cmp eax, mm_rem                           ; CAP < W_min?
     jge .taper_cap_ok                         ; no, CAP is legal, use it
     or qword [config_was_fixed], CFG_FIX_TAPER_CAP ; flag bit 0: taper degenerated to uniform on this system/shape
     cmp dword mm_mode, MATMUL_FUSED_F32       ; fused mode, W_min = head_dim ?
     jne .taper_cap_degen_uniform              ; non-fused: ebx = num_chunks still valid, use uniform fill
     mov eax, [fused_head_dim]                 ; fused: reload head_dim for the uniform fused prep
     jmp .fused_geom                           ; re-enter the fused geometry sequence, recomputes chunks from n_heads, nearest legal
.taper_cap_degen_uniform:
; Reset fill loop counters overwritten by the taper setup, ecx = 2P divisor,
; r8d preserved but re-zeroed for the uniform-fill entry contract .
     xor ecx, ecx                              ; i = 0, uniform fill loop counter
     xor r8d, r8d                              ; column_offset = 0
     jmp .uniform_fill_prep                    ; non-fused: uniform fill with the configured num_chunks, nearest legal
.taper_cap_ok:
     mov mm_g2_chunk_dim, eax                  ; store CAP
     mov eax, [taper_shift]                    ; fixed-point k, units of 1/16, 0=uniform
     cmp eax, TAPER_SHIFT_MAX                  ; unsigned bounds check: accepts exactly [1..112]. Signed jle let a negative taper_shift through, which zeroed the shift and made every chunk W_min
     jbe .taper_k_ok                           ; within range
     FATAL "taper_shift outside [1..112], fix per-shape INI"
.taper_k_ok:
     mov ecx, eax                              ; copy for decomposition
     shr ecx, TAPER_FIXED_SHIFT                ; top = integer part, shift count
     mov mm_g1_rem, ecx                        ; store top for shr cl in hot loop
     and eax, TAPER_FRAC_MASK                  ; f = fractional part, 0..15, zero-extends to rax
     lea rcx, [rel taper_mul_table]            ; multiplier table base
     mov eax, dword [rcx + rax*TAPER_MUL_SCALE] ; load 16.16 fixed-point multiplier for f
     mov mm_taper_mul, eax                     ; store for IMUL in hot loop
     xor r8d, r8d                              ; column_offset = 0
     xor ecx, ecx                              ; i = 0, unused by taper loop body but kept for fill_job_body contract
     mov dword mm_chunk_split_pct, FILL_TAPER  ; route return dispatch to .taper_fill_loop
     mov rbx, mm_weight_s                      ; rbx = weight->scales for taper fill, rbx unused by taper code
     jmp .taper_fill_loop                      ; begin the decay loop
; --- Taper loop: recompute R = D - column_offset each iteration, Option B ---
     align 32                                  ; Align taper loop entry to 32 bytes for hot dispatch path
.taper_fill_loop:
     mov eax, mm_feature_dimension             ; D, total column count
     sub eax, r8d                              ; R = D - column_offset, r8d accumulates chunk_dim
     mov mm_base, eax                          ; snapshot R, for flush path / debug
     cmp eax, mm_rem                           ; R <= W_min?
     jle .taper_flush                          ; yes, emit final unaligned remainder
; Compute chunk = AlignDown,min,R * 2^-k, CAP, clamped to >= W_min, branchless
; Fixed-point IMUL decay: chunk =, R >> top * taper_mul[f] >> 16
     mov edx, eax                              ; edx = R, saved for new_R computation
     mov cl, byte mm_g1_rem_raw                ; top, integer part of shift
     shr eax, cl                               ; eax = R >> top, partial decay
     mov ecx, dword mm_taper_mul               ; 16.16 fixed-point multiplier, zero-extends to rcx
     imul rax, rcx                             ; rax =, R >> top * multiplier, 2-operand, no rdx overwrite
     shr rax, 16                               ; convert from 16.16 fixed-point to integer column count
; Same mask-then-div hazard as the CAP path, pow2 masks, non-pow2
; div-aligns the RAW decayed value.
     cmp dword mm_align_div, 0                 ; W_min not a power of two, head_dim 96 ?
     jne .taper_align_div                      ; yes: div/imul AlignDown of the RAW decayed value
     and eax, mm_g1_chunk_dim                  ; pow2: mask IS exact AlignDown
     jmp .taper_chunk_aligned                  ; rejoin with the aligned chunk, edx still holds R
.taper_align_div:
     xor edx, edx                              ; clear high half of dividend, R snapshot preserved in mm_base
     div dword mm_rem                          ; eax = raw / W_min, edx = raw % W_min
     imul eax, mm_rem                          ; chunk = floor,raw/W_min *W_min, true AlignDown
     mov edx, mm_base                          ; restore R for the new_R computation below
.taper_chunk_aligned:
     cmp eax, mm_g2_chunk_dim                  ; aligned chunk > CAP?
     cmovg eax, mm_g2_chunk_dim                ; chunk = min,chunk, CAP, GSS bound: no job > D/,2P
     cmp eax, mm_rem                           ; aligned chunk < W_min?
     cmovl eax, mm_rem                         ; chunk = max,chunk, W_min, branchless clamp
; Check if remainder would be too small, R - chunk < W_min -> absorb whole R
     sub edx, eax                              ; edx = new_R = R - chunk
     cmp edx, mm_rem                           ; new_R >= W_min?
     jge .taper_store_dim                      ; yes, chunk stays as-is
     mov eax, mm_base                          ; no, absorb full remainder R unaligned, flush already uses R raw
.taper_store_dim:
     mov edx, eax                              ; edx = chunk_dim, .fill_job_body contract: edx holds chunk_dim
     jmp .fill_job_body                        ; store job, update column offset
; --- Taper flush: emit the unaligned remaining columns as the final job ---
.taper_flush:
     mov eax, mm_base                          ; R, remaining after the last non-flush chunk
     test eax, eax                             ; R == 0?
     jz .fill_done                             ; nothing to flush, dispatch what we have
     mov edx, eax                              ; edx = R, .fill_job_body contract
     jmp .fill_job_body                        ; store final job; routing sends us back to taper loop,
; which recomputes R = D - column_offset; if that was the
; flush, column_offset == D now, so R = 0 -> .fill_done
     align 32                                  ; Align fill jobs entry to 32 bytes
.fill_jobs_entry:
     mov rbx, mm_weight_s                      ; rbx = weight->scales for the whole fill, preps done, rbx dead in loops
.fill_jobs:
     cmp ecx, mm_chunk_count                   ; i < chunk_count_local?
     jge .fill_done                            ; done filling jobs
; Compute chunk_dim = base +, i < rem ? head_dim_or_1 : 0
     mov eax, mm_base                          ; base
     cmp ecx, mm_rem                           ; i < rem?
     jge .no_rem_extra                         ; no extra for this chunk
; For fused mode, extra = head_dim, one more head.For normal, extra = 1 column.
     cmp dword mm_mode, MATMUL_FUSED_F32       ; fused mode?
     jne .add_one_extra                        ; no -> extra = 1 column
     add eax, [fused_head_dim]                 ; extra = head_dim, one complete head
     jmp .no_rem_extra                         ; skip single-column path
.add_one_extra:
     inc eax                                   ; chunk_dim = base + 1, one extra column
.no_rem_extra:
; chunk_dim is >= 1 by construction on this path too: base = D / chunk_count and both
; branches above add at least 1, inc eax, or fused_head_dim, so no test is warranted.
     mov edx, eax                              ; edx = chunk_dim, avoid overwriting rdi=weight->q
     jmp .fill_job_body                        ; store job fields for this chunk
; ====== Two-phase split fill preparation ======
     align 16
.split_fill_prep:
     mov dword mm_chunk_split_pct, FILL_SPLIT  ; route return dispatch to split fill, raw pct would misroute to taper/uniform
     mov eax, [chunk_split_pct]                ; pct, raw global, 0-99 by contract
     cmp eax, 100                              ; unsigned bound also rejects negative percentages
     jae .split_pct_fatal                      ; invalid percentage
     mov ecx, [chunk_group_1_count]            ; configured first-group count
     mov edx, [chunk_group_2_count]            ; configured second-group count
     mov r10d, ecx                             ; use the combined sign bits
     or r10d, edx                              ; SF set iff either count is negative
     js .split_count_neg                       ; invalid configured count
     test ecx, ecx                             ; group one absent?
     jz .split_bnd_zero                        ; all columns go to group two
     test edx, edx                             ; group two absent?
     jz .split_bnd_full                        ; all columns go to group one
     imul eax, mm_feature_dimension            ; percentage times the feature count, pct is bounded to [0..99] and d <= 151936, so the product stays under 2^24
     xor edx, edx                              ; unsigned dividend high half
     mov ecx, 100                              ; percentage divisor
     div ecx                                   ; boundary between two nonempty groups
     jmp .split_bnd_store                      ; save computed boundary
.split_bnd_zero:
     xor eax, eax                              ; group one has no columns
     jmp .split_bnd_store                      ; save zero boundary
.split_bnd_full:
     mov eax, mm_feature_dimension             ; group one receives all columns
.split_bnd_store:
     mov mm_base, eax                          ; save g1_cols in mm_base, reused; eax still holds it for the g1 div below
     mov ecx, [chunk_group_1_count]            ; configured g1 count
     cmp ecx, mm_base                          ; g1_count > g1_cols?
     jle .split_g1_pool_chk                    ; no column clamp needed, the pool cap below must still run
     mov ecx, mm_base                          ; nearest legal count: one column per chunk
     or qword [config_was_fixed], CFG_FIX_SPLIT_G1_COLS ; flag bit 3: group-1 column count was engine-fixed for this system/shape
.split_g1_pool_chk:
     cmp ecx, MAX_JOBS - 1                     ; pool must hold g1 PLUS one g2 job: capping g1 at MAX_JOBS leaves the headroom below equal to 0, which zeroes g2_count
     jle .split_g1_budget_store                ; yes: the column count is also pool-legal
     mov ecx, MAX_JOBS - 1                     ; g1_cols exceeds the pool, cap at the ceiling minus g2's reserved slot
     or qword [config_was_fixed], CFG_FIX_SPLIT_G1_POOL ; flag bit 4: group-1 count capped by the job pool ceiling
.split_g1_budget_store:
     mov [chunk_group_1_count], ecx            ; publish the clamped count for every later reader
.split_g1_budget_ok:
; Compute g1_chunk_dim and g1_rem, if g1_count > 0
     test ecx, ecx                             ; g1_count == 0?
     jz .split_g1_zero                         ; yes, skip g1 computation
     mov ebx, ecx                              ; the clamped group-one count is still in ECX
     xor edx, edx                              ; clear for div
     div ebx                                   ; eax = g1_chunk_dim, edx = g1_rem
     mov mm_g1_chunk_dim, eax                  ; save g1 chunk dimension
     mov mm_g1_rem, edx                        ; save g1 remainder
     jmp .split_g2_prep                        ; skip zero case
.split_g1_zero:
     mov dword mm_g1_chunk_dim, 0              ; no g1 chunks
     mov dword mm_g1_rem, 0                    ; no remainder
.split_g2_prep:
; Compute remaining = d - g1_cols
     mov eax, mm_feature_dimension             ; load d again
     sub eax, mm_base                          ; eax = remaining cols for group 2
; Clamp group 2's count the same way, leaving room for group 1 inside MAX_JOBS
     mov ecx, [chunk_group_2_count]            ; configured g2 count
     cmp ecx, eax                              ; g2_count > g2_cols?, eax = remaining cols
     jle .split_g2_pool_chk                    ; no column clamp needed, the pool headroom check below must still run
     mov ecx, eax                              ; nearest legal count: one column per chunk
     or qword [config_was_fixed], CFG_FIX_SPLIT_G2_COLS ; flag bit 5: group-2 column count was engine-fixed for this system/shape
.split_g2_pool_chk:
     mov edx, MAX_JOBS                         ; pool ceiling
     sub edx, [chunk_group_1_count]            ; ceiling = MAX_JOBS - g1_count
     cmp ecx, edx                              ; does the column count also leave room in the pool?
     jle .split_g2_budget_store                ; yes: the column count is also pool-legal
     mov ecx, edx                              ; no: take whatever the pool has left after g1
     or qword [config_was_fixed], CFG_FIX_SPLIT_G2_POOL ; flag bit 6: group-2 count capped by the pool headroom left after g1
.split_g2_budget_store:
     mov [chunk_group_2_count], ecx            ; publish the clamped count for every later reader
.split_g2_budget_ok:
     mov mm_rem, eax                           ; remaining cols for group 2, free in the split path; the taper path's W_min use is finished by here
     test ecx, ecx                             ; g2_count == 0?
     jz .split_g2_zero                         ; yes, skip g2 computation
     xor edx, edx                              ; clear for div
     mov ebx, ecx                              ; the clamped group-two count already in ECX
     div ebx                                   ; eax = g2_chunk_dim, edx = g2_rem
     mov mm_g2_chunk_dim, eax                  ; save g2 chunk dimension
     mov mm_g2_rem, edx                        ; save g2 remainder
     jmp .split_total_count                    ; skip zero case
.split_g2_zero:
     mov dword mm_g2_chunk_dim, 0              ; no g2 chunks
     mov dword mm_g2_rem, 0                    ; no remainder
.split_total_count:
; total = g1_count + g2_count, overwrite mm_chunk_count
; The total can never exceed MAX_JOBS: both published counts are double-clamped
; before this point, g1 <= MAX_JOBS-1 by the pool clamp, g2 <= MAX_JOBS-g1 because
; its pool-clamp ceiling reads the already-published g1, g1+g2 <= MAX_JOBS is
; an invariant of the clamp arithmetic itself, so no overflow guard is needed here.
     mov eax, [chunk_group_1_count]            ; g1 count
     add eax, [chunk_group_2_count]            ; + g2 count
.split_total_ok:
     mov mm_chunk_count, eax                   ; save total chunk count for loop bound
; Reset counters for split fill loop
     xor r8d, r8d                              ; column_offset = 0
     xor ecx, ecx                              ; i = 0, looped_count
     mov rbx, mm_weight_s                      ; rbx = weight->scales for split fill, preps done
; ====== Two-phase split fill loop ======
     align 32
.split_fill_entry:
     cmp ecx, mm_chunk_count                   ; i < total?
     jge .fill_done                            ; done filling jobs
; Determine which group this chunk belongs to
     cmp ecx, [chunk_group_1_count]            ; i < g1_count?
     jae .split_g2_chunk                       ; no, phase 2
; Phase 1: group 1 chunk
     mov eax, mm_g1_chunk_dim                  ; g1 base
     cmp ecx, mm_g1_rem                        ; i < g1_rem?
     jae .split_store_dim                      ; no extra
     inc eax                                   ; chunk_dim = g1 + 1 for remainder distribution
     jmp .split_store_dim                      ; store chunk dimension
.split_g2_chunk:
; Phase 2: group 2 chunk
     mov eax, mm_g2_chunk_dim                  ; g2 base
; relative index within phase 2: i - g1_count
     mov edx, ecx                              ; copy i for relative index
     sub edx, [chunk_group_1_count]            ; subtract g1 count to get g2-relative index
     cmp edx, mm_g2_rem                        ; relative_idx < g2_rem?
     jae .split_store_dim                      ; no extra
     inc eax                                   ; chunk_dim = g2 + 1
.split_store_dim:
     mov edx, eax                              ; edx = chunk_dim, save before test, avoid overwriting rdi=weight->q
; ====== Shared job fill body: uniform, split and taper ======
.fill_job_body:
; rsi addresses the next free 64-byte job.
; jobs[i].output = xout + column_offset, float pointer arithmetic
     lea r9, [r12 + r8*PF_SCALE]               ; r9 = xout + column_offset*4
     JOB_SET_OUTPUT r9                         ; jobs[i].output = xout + column_offset
; jobs[i].input_quantized = input_quantized_data, shared, same for all
     JOB_SET_INPUT_Q r14                       ; jobs[i].input_quantized = input->q
; jobs[i].input_scales = input_scales_data, shared
     JOB_SET_INPUT_S r13                       ; jobs[i].input_scales = input->s
; column_offset * input_elements computed once, reused for both pointers
     mov r9d, r8d                              ; column_offset, 32-bit, zero-extends to 64-bit
     imul r9, r15                              ; r9 = column_offset * input_elements, 2-operand imul
; jobs[i].weight_quantized = weight->q + product, folded mov+add into one lea
     lea r10, [rdi + r9]                       ; r10 = weight->q + offset
     JOB_SET_WEIGHT_Q r10                      ; jobs[i].weight_quantized = weight->q + offset
; jobs[i].weight_scales = weight->s +, product >> 4
     shr r9, 4                                 ; / 16, = /64 * 4, combined shift
     lea r10, [rbx + r9]                       ; r10 = weight->scales +, product >> 4, rbx-cached base, one lea
     JOB_SET_WEIGHT_S r10                      ; jobs[i].weight_scales = weight->s + offset
; jobs[i].input_elements = input_elements
     JOB_SET_INPUT_ELEMENTS r15d               ; jobs[i].input_elements = n
; jobs[i].column_offset = column_offset
     JOB_SET_COLUMN_OFFSET r8d                 ; jobs[i].column_offset = column_offset
; jobs[i].feature_dimension = chunk_dim
     JOB_SET_FEATURE_DIMENSION edx             ; jobs[i].feature_dimension = chunk_dim
     add r8d, edx                              ; column_offset += chunk_dim, edx still holds it, before mode staging
; jobs[i].mode = mode, cached at fill entry, invariant for the whole fill
     JOB_SET_MODE r11d                         ; jobs[i].mode = cached mode
     JOB_NEXT rsi                              ; next descriptor, only after a stored job
     inc ecx                                   ; loop_count++
; Return to the correct fill loop. Two sentinels are tested and uniform is the
; fall-through, no FILL_UNIFORM constant exists: the uniform preps store a literal 0 .
     cmp dword mm_chunk_split_pct, FILL_SPLIT  ; split mode, == 1 ?
     je .split_fill_entry                      ; yes, split loop back
     cmp dword mm_chunk_split_pct, FILL_TAPER  ; taper mode, == 2 ?
     je .taper_fill_loop                       ; yes, taper loop back, R recompute via D - r8d
     jmp .fill_jobs                            ; uniform mode, no sentinel set loop back
.split_pct_fatal:
     FATAL "dispatch: chunk_split_pct outside [0..99]"
.split_count_neg:
     FATAL "dispatch: chunk_group_1_count/chunk_group_2_count < 0, fix per-shape INI"
.split_zero_chunks_fatal:
     FATAL "dispatch: split config produced zero chunks (g1+g2 == 0)" ; contract violation: matmul requested but no jobs filled, xout would never be written
.fill_done:
%ifdef ZYDIS_TRACE
     INT3_DBG_STR "mm: fill_done"              ; Debug trace
%endif
     cmp dword mm_mode, MATMUL_Q8              ; this dispatch a plain Q8 matmul?
     jne .skip_q8_patch                        ; no, v7s_q8_chunk never runs, skip the code-page writes
     mov eax, mm_input_elements                ; input_elements, n
     PATCH_Q8_N v7s_q8_chunk, eax, r10         ; patch shared n and three n-derived scale displacements; d remains per-job
.skip_q8_patch:
     mov rax, rsi                              ; one-past-last job
     lea rdx, [rel jobs_arr]                   ; descriptor array base
     sub rax, rdx                              ; bytes of stored descriptors
     shr rax, JOB_SIZE_SHL                     ; eax = actual emitted job count
     test eax, eax                             ; job count zero?
     jz .split_zero_chunks_fatal               ; yes: fill produced no jobs, xout would never be written
     DISPATCH_EXECUTE eax, mm_caller_count, mm_caller_end ; Dispatch jobs with caller participation
; ====== the in-process cycle tuner: rdtsc stop + fold + verdict ======
; The stop timestamp is captured HERE, before any prologue can run, so the
; measured window ends at DISPATCH_EXECUTE's last instruction. Ordinary
; dispatches (no staged step) skip both the capture and the call.
     cmp dword [rel cycle_tuner_step_live], 0  ; a timed step staged by this dispatch's flip?
     je .tune_no_stop                          ; no: plain dispatch, nothing recorded
     rdtscp                                    ; stop timer, rdtscp waits for older instructions
     lfence                                    ; serializing tail of the RDTSCP;LFENCE idiom
     shl rdx, 32                               ; high half into position
     or rdx, rax                               ; rdx = packed 64-bit stop timestamp
     mov [rel cycle_tuner_tsc_stop], rdx       ; stage it for the stop hook
     call cycle_tuner_stop                     ; record the delta, fold, advance, verdict at state 4
.tune_no_stop:
; ====== End in-process cycle tuner injection 2 ======
%ifdef ZYDIS_TRACE
     INT3_DBG_STR "caller: returning to C++ caller" ; Debug trace
%endif
     vzeroupper                                ; clear dirty YMM upper bits from kernel
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; Epilogue, restore non-volatiles
%undef CFG_FIX_SPLIT_G1_COLS                   ; config_was_fixed bit 3
%undef CFG_FIX_SPLIT_G1_POOL                   ; config_was_fixed bit 4
%undef CFG_FIX_SPLIT_G2_COLS                   ; config_was_fixed bit 5
%undef CFG_FIX_SPLIT_G2_POOL                   ; config_was_fixed bit 6
%undef MM_ARG_INPUT_ELS                        ; caller stack arg 5, input_elements n
%undef MM_ARG_FEAT_DIM                         ; caller stack arg 6, feature_dimension d
%undef CFG_FIX_TAPER_CAP                       ; config_was_fixed bit 0
%undef CFG_FIX_NUM_CHUNKS                      ; config_was_fixed bit 1
%undef IDX_2                                   ; fill-plan block index 2
%undef IDX_1                                   ; fill-plan block index 1
%undef PF_SCALE                                ; bytes per dword-array element, prefetch slots, xout float stride
%undef FILL_TAPER                              ; exponentially-decaying fill mode
%undef FILL_SPLIT                              ; two-phase split fill mode
%undef TAPER_MUL_SCALE                         ; bytes per taper_mul_table entry
%undef TAPER_FIXED_SHIFT                       ; bits for the integer taper part
%undef TAPER_FRAC_MASK                         ; mask for the fractional taper bits
%undef TAPER_SHIFT_MAX                         ; max taper_shift value
%undef mm_xout                                 ; frame slot: xout base pointer
%undef mm_xout_raw
%undef mm_input                                ; frame slot: input tensor ptr
%undef mm_input_raw
%undef mm_weight                               ; frame slot: weight tensor ptr
%undef mm_weight_raw
%undef mm_weight_s                             ; frame slot: cached weight->scales
%undef mm_weight_s_raw
%undef mm_input_elements                       ; frame slot: input_elements, n
%undef mm_input_elements_raw
%undef mm_feature_dimension                    ; frame slot: feature_dimension, d
%undef mm_feature_dimension_raw
%undef mm_chunk_count                          ; frame slot: chunk_count_local
%undef mm_chunk_count_raw
%undef mm_base                                 ; frame slot: base / g1_cols / R snapshot
%undef mm_base_raw
%undef mm_rem                                  ; frame slot: rem / W_min / g2_cols
%undef mm_rem_raw
%undef mm_caller_count                         ; frame slot: jobs the caller processed
%undef mm_caller_count_raw
%undef mm_caller_end                           ; frame slot: caller batch end bound
%undef mm_caller_end_raw
%undef mm_mode                                 ; frame slot: MATMUL_Q8 / MATMUL_F32
%undef mm_mode_raw
%undef mm_g1_chunk_dim                         ; frame slot: group 1 chunk dimension / align mask
%undef mm_g1_chunk_dim_raw
%undef mm_g1_rem                               ; frame slot: group 1 remainder / taper top
%undef mm_g1_rem_raw
%undef mm_g2_chunk_dim                         ; frame slot: group 2 chunk dimension / taper CAP
%undef mm_g2_chunk_dim_raw
%undef mm_g2_rem                               ; frame slot: group 2 remainder
%undef mm_g2_rem_raw
%undef mm_chunk_split_pct                      ; frame slot: fill-strategy sentinel
%undef mm_chunk_split_pct_raw
%undef mm_taper_mul                            ; frame slot: 16.16 decay multiplier
%undef mm_taper_mul_raw
%undef mm_align_div                            ; frame slot: W_min pow2 flag
%undef mm_align_div_raw
