; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; qk_fork_kernel.inc -- Q8 per-head QK RMSNorm+RoPE, reentrant for fork/main overlap.
; Enabled only for has_qk_norm=1 models; the existing RoPE-only path is unchanged.
; One pointer argument (RCX) to an immutable descriptor on the caller's stack.
; Shared RMSNorm BSS globals are NOT touched: both threads run concurrently.

     struc qk_fork_args                        ; fork kernel descriptor layout, 48 bytes
.buf:
     resq 1                                    ; head buffer base, s->q slice
.weight:
     resq 1                                    ; per-layer norm weight base
.cos:
     resq 1                                    ; cos table, already offset to sequence position
.sin:
     resq 1                                    ; sin table, already offset to sequence position
.head_dim:
     resd 1                                    ; dimension per attention head
.heads:
     resd 1                                    ; head count for this descriptor
.pos:
     resd 1                                    ; RoPE sequence position
.unused:
     resd 1                                    ; pad to 48 bytes
     endstruc                                  ; defines qk_fork_args_size

%if qk_fork_args_size != 48
%error qk_fork_args_size must be 48            ; layout guard: descriptor slots are addressed by these offsets
%endif

%define QKF_FLOAT4_SCALE 4                     ; sizeof,float for SIB scale in head addressing

     section .text
; ============================================================================
; v7s_qk_fork_heads -- RMSNorm+RoPE over a head range, descriptor-driven
; RCX = qk_fork_args* descriptor, read-only on both threads
; ============================================================================
     SF_START v7s_qk_fork_heads, rbx, r12, r13, r14 ; prologue: save non-volatiles, r12/r13 = kernel body homes
     SF_DWORD qkf_dim_float                    ; thread-private float head_dim scalar for COMPUTE_RMS_SCALE
     SF_XMM qkf_xmm14                          ; save xmm14 low 128 bits, sole callee-saved XMM touched by RMSNORM_ROPE_BODY
     SF_COMMIT                                 ; finalize stack frame layout
     vmovdqu qkf_xmm14, xmm14                  ; save xmm14 LOW 128 bits for the RMSNorm scale broadcast
     mov r14, rcx                              ; r14 = descriptor base, remains read-only on both threads
     mov edx, [r14 + qk_fork_args.head_dim]    ; edx = head_dim, for the float conversion
     vcvtsi2ss xmm0, xmm0, edx                 ; xmm0 = float(head_dim)
     vmovss qkf_dim_float, xmm0                ; store the thread-private dimension scalar
     mov r12, [r14 + qk_fork_args.buf]         ; r12 = first head's buffer, RMSNORM_ROPE_BODY uses r12 in place and never modifies it
     xor ebx, ebx                              ; ebx = head index
.qkf_head:
     mov r13d, [r14 + qk_fork_args.head_dim]   ; r13d = head_dim, RMSNORM_ROPE_BODY's non-volatile home
     RMSNORM_ROPE_BODY qkf_dim_float, [r14 + qk_fork_args.weight], [r14 + qk_fork_args.cos], [r14 + qk_fork_args.sin], dword [r14 + qk_fork_args.pos], .qkf_done_head ; inline RMSNorm+RoPE for this head, all inputs from the descriptor
.qkf_done_head:
     mov eax, [r14 + qk_fork_args.head_dim]    ; eax = head_dim elements
     lea r12, [r12 + rax*QKF_FLOAT4_SCALE]     ; advance to the next head's contiguous region, no per-head multiply or base reload
     inc ebx                                   ; head index++
     cmp ebx, [r14 + qk_fork_args.heads]       ; all heads of this descriptor done?
     jb .qkf_head                              ; next head
     CLEANUP_YMM                               ; vzeroupper BEFORE restoring callee-saved XMMs
     vmovdqu xmm14, qkf_xmm14                  ; restore xmm14 LOW 128 bits; uppers stay zeroed
     SF_END rbx, r12, r13, r14                 ; restore non-volatiles and return
; ---- %undef cleanup: keep the preprocessor namespace clean for later includes ----
%undef qkf_dim_float                           ; SF_DWORD slot alias
%undef qkf_dim_float_raw                       ; SF_DWORD raw slot alias
%undef qkf_xmm14                               ; SF_XMM spill slot alias
%undef QKF_FLOAT4_SCALE                        ; file-local float byte scale
