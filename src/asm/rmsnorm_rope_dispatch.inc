; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define RN_COS_TABLE 48                        ; Cos table pointer, 5th stack arg
%define RN_SIN_TABLE 56                        ; Sin table pointer, 6th stack arg
%define RN_SEQ_POS 64                          ; Sequence position, 7th stack arg
%define RN_ROPE_MAX_DIM 72                     ; RoPE max dimension, 8th stack arg
%define RN_FLOAT4_SCALE 4                      ; scale factor for float pointer arithmetic, sizeof,float
     section .text
     SF_START v7s_rmsnorm_rope_heads, rbx, rsi, r12, r14, r15 ; prologue: rdi/r13 dropped, never written here ; r14 SAVED only because DISPATCH_EXECUTE overwrites it, r14d = caller_batch, dropping it from this list would silently corrupt a non-volatile the caller holds
     SF_DWORD rn_de_caller_count               ; caller job counter, dispatch path
     SF_DWORD rn_de_caller_end                 ; caller batch end bound
     SF_COMMIT                                 ; commit stack frame locals
     mov r15d, edx                             ; r15d = n_heads, head loop bound, invariant
     mov [rmsnorm_head_dim], r8d               ; publish head_dim, dword global, read by workers
     mov [rmsnorm_norm_weight], r9             ; publish norm weight pointer, global, read by workers
; Publish, float head_dim once, head_dim is invariant across all heads --
; the kernel divides by it, single correct rounding, matches the C reference
     vcvtsi2ss xmm0, xmm0, r8d                 ; xmm0 =, float head_dim, merge bits [127:32] come from xmm0's own residue, dead: only [31:0] is read, by the vmovss store below
     vmovss [rmsnorm_head_dim_float], xmm0     ; store for workers
     mov r10, [rbp+RN_COS_TABLE]               ; r10 = cos table base, direct load, no rn_cos spill
     mov r11, [rbp+RN_SIN_TABLE]               ; r11 = sin table base, direct load, no rn_sin spill
     mov eax, [rbp+RN_SEQ_POS]                 ; load pos, stack arg 7, 4-byte load ZERO-EXTENDS into rax, which the imul below turns into a table pointer, so the 32-bit pos value MUST stay non-negative
     mov [rmsnorm_pos], eax                    ; pos global
; Save buf in r12, non-volatile before anything overwrites rcx
     mov r12, rcx                              ; r12 = buf, preserved across calls
     mov r8d, [rbp+RN_ROPE_MAX_DIM]            ; rope_max_dim = elements per table position
     imul r8, rax                              ; row offset = pos * rope_max_dim, elements
     lea r10, [r10 + r8*RN_FLOAT4_SCALE]       ; r10 = offset cos pointer, fused SIB
     lea r11, [r11 + r8*RN_FLOAT4_SCALE]       ; r11 = offset sin pointer, fused SIB
     mov [rmsnorm_cos_table], r10              ; update global for workers, final write
     mov [rmsnorm_sin_table], r11              ; update global, final write
; Init worker pool & SMC, runs once via CAS guard
     call initialize_once                      ; ensure thread pool & SMC patching initialized
; Set FTZ/DAZ on the calling thread to match worker threads
     SET_FTZ_DAZ                               ; flush denormals to zero for consistent arithmetic
     mov eax, MATMUL_RMSNORM_ROPE              ; job mode, loop-invariant, hoisted out of the loop
     xor ebx, ebx                              ; h = 0, head loop counter
     JOB_BASE                                  ; rsi = jobs_arr[0] base for the fill loop
.rn_fill_loop:
     JOB_SET_OUTPUT r12                        ; JOB_OUTPUT = buf
     JOB_SET_COLUMN_OFFSET ebx                 ; JOB_COLUMN_OFFSET = h, dword field, low dword of the counter
     JOB_SET_MODE eax                          ; JOB_MODE = RMSNORM_ROPE
     JOB_NEXT rsi                              ; advance to next job, the one permitted spelling; 1 instruction
     inc ebx                                   ; h++, 32-bit inc: zero-extends, matches 32-bit cmp
     cmp ebx, r15d                             ; h < n_heads?
     jl .rn_fill_loop                          ; yes, next head
; ====== Central dispatch: publish, steal, wait, track ======
     DISPATCH_EXECUTE r15d, rn_de_caller_count, rn_de_caller_end ; publish, steal, wait, r15d = n_heads, cached in the prologue, fill loop never touches r15
     vzeroupper                                ; clear dirty YMM upper bits before return
     SF_END rbx, rsi, r12, r14, r15            ; restore non-volatiles and return, rdi/r13 dropped, never written here
%undef RN_FLOAT4_SCALE                         ; file-local scale for float pointer arithmetic
%undef RN_ROPE_MAX_DIM                         ; file-local 8th stack arg offset
%undef RN_SEQ_POS                              ; file-local 7th stack arg offset
%undef RN_SIN_TABLE                            ; file-local 6th stack arg offset
%undef RN_COS_TABLE                            ; file-local 5th stack arg offset
