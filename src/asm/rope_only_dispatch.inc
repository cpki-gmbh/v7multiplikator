; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define RO_D_SIN_TABLE 48                      ; Sin table pointer, 5th arg, first stack arg
%define RO_D_SEQ_POS 56                        ; Sequence position, 6th arg, second stack arg
%define RO_D_ROPE_MAX_DIM 64                   ; RoPE max dimension, 7th arg, third stack arg
%define RO_D_FLOAT4_SCALE 4                    ; scale factor for float pointer arithmetic, sizeof,float
     section .text
     SF_START v7s_rope_only_heads, rbx, rsi, r12, r14, r15 ; prologue: r14 SAVED only because DISPATCH_EXECUTE overwrites it, never written in this body, same rationale as rmsnorm_rope_dispatch.inc:11 ; r15d = n_heads, fill bound + dispatch count ; r12 = buf; rsi = the job cursor
     SF_DWORD rod_de_caller_count              ; caller job counter, dispatch path
     SF_DWORD rod_de_caller_end                ; caller batch end bound
     SF_COMMIT                                 ; commit stack frame locals
     mov r12, rcx                              ; r12 = buf, save first, frees rcx
     mov r15d, edx                             ; r15d = n_heads, non-volatile home
     mov [rmsnorm_head_dim], r8d               ; head_dim global, kernel readback
     mov r11, [rbp+RO_D_SIN_TABLE]             ; r11 = sin table base, stack arg 5
     mov eax, [rbp+RO_D_SEQ_POS]               ; eax = pos, stack arg 6
     mov [rmsnorm_pos], eax                    ; pos global
     mov r8d, [rbp+RO_D_ROPE_MAX_DIM]          ; r8d = rope_max_dim, stack arg 7, zero-extends r8; rax still holds pos
     imul r8, rax                              ; row offset = pos * rope_max_dim, elements
     lea r10, [r9 + r8*RO_D_FLOAT4_SCALE]      ; r10 = pos-offset cos pointer, fused SIB, straight from arg4, r9 untouched since entry
     lea r11, [r11 + r8*RO_D_FLOAT4_SCALE]     ; r11 = pos-offset sin pointer, fused SIB
     mov [rmsnorm_cos_table], r10              ; publish offset cos global, final write
     mov [rmsnorm_sin_table], r11              ; publish offset sin global, final write
; Init worker pool & SMC, runs once via CAS guard; guarantees init_ok==1
; on every return path, so no init_ok gate is needed here
     call initialize_once                      ; thread pool + SMC, r10/r11/rdx dead at this point
; Set FTZ/DAZ on the calling thread to match worker threads, requires init: g_mxcsr image is live
     SET_FTZ_DAZ                               ; flush denormals to zero for consistent arithmetic
; --- Fill jobs_arr with per-head MATMUL_ROPE_ONLY jobs ---
     mov eax, MATMUL_ROPE_ONLY                 ; job mode constant hoisted, loop-invariant
     xor ebx, ebx                              ; h = 0, head loop counter
     JOB_BASE                                  ; rsi = jobs_arr[0] base for the fill loop
.rod_fill_loop:
; Fill JOB fields via JOB_SET_* macros, implicit rsi
     JOB_SET_OUTPUT r12                        ; JOB_OUTPUT = buf
     JOB_SET_COLUMN_OFFSET ebx                 ; JOB_COLUMN_OFFSET = h, direct, no eax round-trip
     JOB_SET_MODE eax                          ; JOB_MODE = ROPE_ONLY, hoisted constant
     JOB_NEXT rsi                              ; advance to next job, the one permitted spelling; 1 instruction
     inc rbx                                   ; h++
     cmp ebx, r15d                             ; h < n_heads?
     jl .rod_fill_loop                         ; yes, next head, bottom-tested do-while
; ====== Central dispatch: publish, steal, wait, track ======
     DISPATCH_EXECUTE r15d, rod_de_caller_count, rod_de_caller_end ; publish, steal, wait, falls through to epilogue
     vzeroupper                                ; clear dirty YMM upper bits before return
     SF_END rbx, rsi, r12, r14, r15            ; restore non-volatiles and return
; %undef cleanup: file-local defines leave scope here --
%undef RO_D_FLOAT4_SCALE                       ; file-local scale for float pointer arithmetic
%undef RO_D_ROPE_MAX_DIM                       ; file-local 7th stack arg offset
%undef RO_D_SEQ_POS                            ; file-local 6th stack arg offset
%undef RO_D_SIN_TABLE                          ; file-local 5th stack arg offset
