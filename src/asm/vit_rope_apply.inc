; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     global vit_rope_apply                     ; export vit_rope_apply entry point
%define VRA_ROW_SCALE 2                        ; table-row byte stride scale = rbx*2, rbx = rope_half*4; head_dim*4 == 2*rope_half*4, definitional
%define VRA_YMM32 32                           ; byte offset for YMM lane 1
%define VRA_YMM64 64                           ; byte offset for YMM lane 2
%define VRA_YMM96 96                           ; byte offset for YMM lane 3
%define VRA_LO0 -128                           ; high-half base + this = &Q/K[h*64 + 0], = -,rope_half*4 = -VCFG_ROPE_HALF_F4
%define VRA_LO32 -96                           ; ... + 8, VRA_YMM32 + VRA_LO0
%define VRA_LO64 -64                           ; ... + 16
%define VRA_LO96 -32                           ; ... + 24
; ============================================================================
; VRA_ROTATE_QK q_base, k_base, lo_disp, hi_disp, cos_reg, sin_reg

; Rotate one dim pair of BOTH the Q and K rows in place using house RoPE
; algebra: lower' = lower*cos - upper*sin, upper' = upper*cos + lower*sin.
; All four Q/K loads issue before any store, so the two independent streams
; fetch in parallel while the arithmetic proceeds group by group.

; Overwrites: ymm0, Q lower, ymm1, Q upper, ymm2, K lower, ymm3, K upper,
; ymm4, low result, ymm5, high result. Each row's ymm2/ymm3 pair stays live
; until its own second FMA. The two outputs split across ymm4/ymm5 so both
; multiplies issue before either FMA.
; ============================================================================

%macro VRA_ROTATE_QK 6                         ; q_base, k_base, lo_disp, hi_disp, cos_reg, sin_reg
     vmovups ymm0, [%1 + %3]                   ; Q lower, both Q and K read before any store
     vmovups ymm1, [%1 + %4]                   ; Q upper
     vmovups ymm2, [%2 + %3]                   ; K lower, independent stream
     vmovups ymm3, [%2 + %4]                   ; K upper, independent stream
     vmulps ymm4, ymm0, %5                     ; Q lower*cos
     vmulps ymm5, ymm1, %5                     ; Q upper*cos
     vmulps ymm14, ymm2, %5                    ; K lower*cos, independent of the Q chain
     vmulps ymm15, ymm3, %5                    ; K upper*cos, independent of the Q chain
     vfnmadd231ps ymm4, ymm1, %6               ; Q rotated lower
     vfmadd231ps ymm5, ymm0, %6                ; Q rotated upper
     vfnmadd231ps ymm14, ymm3, %6              ; independent K work before the Q stores
     vfmadd231ps ymm15, ymm2, %6               ; K rotated upper
     vmovups [%1 + %3], ymm4                   ; Q low store
     vmovups [%1 + %4], ymm5                   ; Q high store
     vmovups [%2 + %3], ymm14                  ; K low store
     vmovups [%2 + %4], ymm15                  ; K high store
%endmacro
vit_rope_apply:
     SF_START vit_rope_apply, rbx, r12, r13, r14, r15 ; prologue: save non-volatiles, reserve frame, rsi/rdi are not used, the low half is addressed off rax/r9 via the negative disp8 displacements
     SF_QWORD vra_qkv                          ; qkv buffer base
     SF_QWORD vra_cos                          ; cos table base
     SF_QWORD vra_sin                          ; sin table base
     SF_DWORD vra_n                            ; patch count
     SF_XMM vra_xmm6                           ; callee-saved xmm6 low half, hoisted cos[0..7] across 16 heads
     SF_XMM vra_xmm7                           ; callee-saved xmm7 low half, hoisted cos[8..15]
     SF_XMM vra_xmm8                           ; callee-saved xmm8 low half, hoisted cos[16..23]
     SF_XMM vra_xmm9                           ; callee-saved xmm9 low half, hoisted cos[24..31]
     SF_XMM vra_xmm10                          ; callee-saved xmm10 low half, hoisted sin[0..7]
     SF_XMM vra_xmm11                          ; callee-saved xmm11 low half, hoisted sin[8..15]
     SF_XMM vra_xmm12                          ; callee-saved xmm12 low half, hoisted sin[16..23]
     SF_XMM vra_xmm13                          ; callee-saved xmm13 low half, hoisted sin[24..31]
     SF_XMM vra_xmm14                          ; callee-saved xmm14 low half, K-lower rotation result
     SF_XMM vra_xmm15                          ; callee-saved xmm15 low half, K-upper rotation result
     SF_COMMIT                                 ; finalize stack frame layout
     mov vra_qkv, rcx                          ; save qkv base
     mov vra_cos, rdx                          ; save cos base
     mov vra_sin, r8                           ; save sin base
     mov vra_n, r9d                            ; save n
     vmovdqu vra_xmm6, xmm6                    ; save xmm6 low half
     vmovdqu vra_xmm7, xmm7                    ; save xmm7 low half
     vmovdqu vra_xmm8, xmm8                    ; save xmm8 low half
     vmovdqu vra_xmm9, xmm9                    ; save xmm9 low half
     vmovdqu vra_xmm10, xmm10                  ; save xmm10 low half
     vmovdqu vra_xmm11, xmm11                  ; save xmm11 low half
     vmovdqu vra_xmm12, xmm12                  ; save xmm12 low half
     vmovdqu vra_xmm13, xmm13                  ; save xmm13 low half
     vmovdqu vra_xmm14, xmm14                  ; save xmm14 low half
     vmovdqu vra_xmm15, xmm15                  ; save xmm15 low half
; Cache cfg globals, constant within call into volatile regs, leaf function,
; no calls so rdx/r8/r10/r11 survive the whole body. Eliminates ~116K RIP-rel loads.
     mov rbx, qword [rel g_vit_cfg + VCFG_ROPE_HALF_F4] ; rbx = rope_half*4, 128, high-half byte offset
     mov r11d, dword [rel g_vit_cfg + VCFG_HEADS] ; r11d = head count, 16, dword ON PURPOSE: it bounds the 32-bit head counter r8d at the loop test, and nothing 64-bit consumes it
     mov rdx, qword [rel g_vit_cfg + VCFG_QKV_DIM] ; rdx = qkv_dim, full row width; head_dim == 64 is the import's contract for this RoPE loop's strides
     shl rdx, 2                                ; x4 = qkv row byte stride, running-pointer advance
     mov r10, qword [rel g_vit_cfg + VCFG_HIDDEN_F4] ; r10 = hidden * 4, K offset within qkv; QWORD field, loop-invariant, hoisted, it is consumed by the 64-bit lea that builds the K base each patch
     xor ecx, ecx                              ; p = 0, patch counter in dead rcx, leaf function, no calls, so no frame slot is needed for it
     mov r12, vra_qkv                          ; r12 = &qkv[p][0], running pointer, advanced by the qkv row stride at the loop tail, exactly like r14/r15 below
     mov r14, vra_cos                          ; r14 = &cos[p][0], running pointer, advanced one table row per patch
     mov r15, vra_sin                          ; r15 = &sin[p][0], running pointer, same row stride
.vra_patch_loop:
; K base for this patch: r12 + hidden * 4, r10 hoisted above
     lea r13, [r12 + r10]                      ; r13 = &qkv[p][hidden] = K base
     vmovups ymm6, [r14]                       ; cos[0..7]
     vmovups ymm7, [r14 + VRA_YMM32]           ; cos[8..15]
     vmovups ymm8, [r14 + VRA_YMM64]           ; cos[16..23]
     vmovups ymm9, [r14 + VRA_YMM96]           ; cos[24..31]
     vmovups ymm10, [r15]                      ; sin[0..7]
     vmovups ymm11, [r15 + VRA_YMM32]          ; sin[8..15]
     vmovups ymm12, [r15 + VRA_YMM64]          ; sin[16..23]
     vmovups ymm13, [r15 + VRA_YMM96]          ; sin[24..31]
     lea rax, [r12 + rbx]                      ; rax = &Q[0][rope_half], high-half base for the current head
     lea r9, [r13 + rbx]                       ; r9 = &K[0][rope_half], same scheme
     xor r8d, r8d                              ; h = 0, register head counter
.vra_head_loop:
; Fully unrolled dim loop, rope_half=32, step=8, exactly 4 iterations
; head_dim=64 is the import's contract; eliminates 12 loop-overhead instructions per head
; --- Iteration 0, d=0 ---
; Q: q0 = Q[0..7], q1 = Q[32..39], cos/sin read directly from the preload regs
     VRA_ROTATE_QK rax, r9, VRA_LO0, 0, ymm6, ymm10 ; rotate Q and K, d=0
; --- Iteration 1, d=8 ---
; Q: q0 = Q[8..15], q1 = Q[40..47]
     VRA_ROTATE_QK rax, r9, VRA_LO32, VRA_YMM32, ymm7, ymm11 ; rotate Q and K, d=8
; --- Iteration 2, d=16 ---
; Q: q0 = Q[16..23], q1 = Q[48..55]
     VRA_ROTATE_QK rax, r9, VRA_LO64, VRA_YMM64, ymm8, ymm12 ; rotate Q and K, d=16
; --- Iteration 3, d=24 ---
; Q: q0 = Q[24..31], q1 = Q[56..63]
     VRA_ROTATE_QK rax, r9, VRA_LO96, VRA_YMM96, ymm9, ymm13 ; rotate Q and K, d=24
     lea rax, [rax + rbx*VRA_ROW_SCALE]        ; advance Q high-half head pointer, head_dim*4 = rbx*2, definitional
     lea r9, [r9 + rbx*VRA_ROW_SCALE]          ; advance K high-half head pointer
     inc r8d                                   ; h++, register counter
     cmp r8d, r11d                             ; h < heads?, cached
     jl .vra_head_loop                         ; next head, bottom test
     add r12, rdx                              ; Q base advances one qkv row, qkv_dim*4, cached in rdx at :72-73
     add r14, rbx                              ; compact cos row = rope_half*4 bytes, only the first half is read
     add r15, rbx                              ; compact sin row = rope_half*4 bytes
     inc ecx                                   ; p++, register counter
     cmp ecx, vra_n                            ; p < n?, memory-operand compare
     jl .vra_patch_loop                        ; next patch, bottom test
.vra_done:
     vzeroupper                                ; clear the dirty YMM uppers before the XMM restores and the return, uppers are volatile per Win64; the caller never sees ours, and this is the project idiom for every AVX leaf
     vmovdqu xmm6, vra_xmm6                    ; restore callee-saved xmm6 low half
     vmovdqu xmm7, vra_xmm7                    ; restore callee-saved xmm7 low half
     vmovdqu xmm8, vra_xmm8                    ; restore callee-saved xmm8 low half
     vmovdqu xmm9, vra_xmm9                    ; restore callee-saved xmm9 low half
     vmovdqu xmm10, vra_xmm10                  ; restore callee-saved xmm10 low half
     vmovdqu xmm11, vra_xmm11                  ; restore callee-saved xmm11 low half
     vmovdqu xmm12, vra_xmm12                  ; restore callee-saved xmm12 low half
     vmovdqu xmm13, vra_xmm13                  ; restore callee-saved xmm13 low half
     vmovdqu xmm14, vra_xmm14                  ; restore callee-saved xmm14 low half
     vmovdqu xmm15, vra_xmm15                  ; restore callee-saved xmm15 low half
     SF_END rbx, r12, r13, r14, r15            ; restore non-volatiles, return
%undef VRA_YMM96
%undef VRA_YMM64
%undef VRA_YMM32
%undef VRA_LO0
%undef VRA_LO32
%undef VRA_LO64
%undef VRA_LO96
%unmacro VRA_ROTATE_QK 6                       ; remove the per-dim QK rotation macro, all four expansion sites above
