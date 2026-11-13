; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define VLN_DWORD_SCALE 4                      ; SIB scale: dim * sizeof,float row stride

     global vit_layernorm                      ; export vit_layernorm entry point
vit_layernorm:
     SF_START vit_layernorm, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles, reserve frame
     SF_XMM vln_xmm6                           ; callee-saved xmm6 spill slot, per-row shift broadcast
     SF_XMM vln_xmm7                           ; callee-saved xmm7 spill slot, 1.0 constant cache
     SF_XMM vln_xmm8                           ; callee-saved xmm8 spill slot, block-B scratch
     SF_COMMIT                                 ; finalize stack frame layout
     vmovdqu vln_xmm6, xmm6                    ; save callee-saved xmm6, leaf fn: nothing rewrites this spill slot; epilogue :107 restores the entry value
     vmovdqu vln_xmm7, xmm7                    ; save callee-saved xmm7, leaf fn: nothing rewrites this spill slot; overwriteed in-body at :17, epilogue :108 restores the entry value
     vmovdqu vln_xmm8, xmm8                    ; save callee-saved xmm8
     mov edi, [rbp + VK_ARG6]                  ; dim, register cache for all loop bounds; positive multiple of 16 is the import's contract for the 2-way unrolled loops
     xor esi, esi                              ; esi = row counter r = 0, live in register, no stack slot
     vcvtsi2ss xmm5, xmm5, edi                 ; float dim in scalar lane
     vshufps xmm5, xmm5, xmm5, 0x00            ; all four lanes = (float)dim, row-invariant divisor
     vmovss xmm7, [rel vit_gelu_one]           ; xmm7 = 1.0, row-invariant; feeds inv_std division
; Row stride in bytes = dim * 4
     lea r13, [rdi*VLN_DWORD_SCALE]            ; r13 = dim * 4, 64-bit row stride; rdi already holds the zero-extended dim from `mov edi` at :13, so this is the same addressing result with lower latency
     mov r12, rdx                              ; r12 = running input ptr, rdx live since entry
     mov r14, rcx                              ; r14 = running output ptr, rcx live since entry
     mov r15, r8                               ; r15 = gamma base, same for all rows; r8 live
     mov rbx, r9                               ; rbx = beta base, same for all rows; r9 live
     mov r8d, [rbp + VK_ARG5]                  ; n_rows cached in r8d, after the pointer caches
     mov eax, edi                              ; dim copy for the block count, row-invariant hoist: eax free in the row body
     shr eax, 4                                ; eax = dim/16 block count, exact: import contract dim % 16 == 0, see :13
; Row loop is a do-while: every validated caller path passes n_rows >= 1
; the encode FATALs zero-token inputs; the anyres tile path always has >= 1
; row.If that ever changes, add a zero-row guard here, with n_rows == 0
; this shape would process one garbage row before the bottom test exits.
.vln_row_loop:
     vbroadcastss ymm6, dword [r12]            ; shift = first x value, broadcast to all lanes, low half read by the mean recovery at :116
     vxorps ymm0, ymm0, ymm0                   ; shifted sum accumulator A = 0
     vxorps ymm1, ymm1, ymm1                   ; shifted square sum accumulator A = 0
     vxorps ymm2, ymm2, ymm2                   ; shifted sum accumulator B = 0
     vxorps ymm3, ymm3, ymm3                   ; shifted square sum accumulator B = 0
     mov edx, eax                              ; block countdown = dim/16, row-invariant, hoisted into eax
     xor ecx, ecx                              ; d = 0, register counter
     align 16
.vln_stats_loop:
     vmovups ymm4, [r12 + rcx*VK_F4]           ; x[d..d+7], read ONCE for both stats
     vmovups ymm8, [r12 + rcx*VK_F4 + VK_8F_BYTES] ; x[d+8..d+15], independent scratch, no WAR on ymm4
     vsubps ymm4, ymm4, ymm6                   ; x - shift for the first 8 elements
     vsubps ymm8, ymm8, ymm6                   ; x - shift for the second 8 elements
     vaddps ymm0, ymm0, ymm4                   ; shifted sum A
     vaddps ymm2, ymm2, ymm8                   ; shifted sum B
     vfmadd231ps ymm1, ymm4, ymm4              ; shifted square sum A
     vfmadd231ps ymm3, ymm8, ymm8              ; shifted square sum B
     add ecx, 16                               ; d += 16, two YMM blocks
     dec edx                                   ; one 16-element block done, independent chain
     jnz .vln_stats_loop                       ; more blocks, bottom-tested
; Merge A+B accumulators before the scalar reductions
     vaddps ymm0, ymm0, ymm2                   ; sum_x = A + B
     vaddps ymm1, ymm1, ymm3                   ; sum_x2 = A + B
; Interleave two independent six-op scalar reductions; keep the original tree.
     vextractf128 xmm3, ymm0, 1                ; upper half, sum
     vextractf128 xmm4, ymm1, 1                ; upper half, square sum
     vaddps xmm0, xmm0, xmm3                   ; sum: merge 8 -> 4
     vaddps xmm1, xmm1, xmm4                   ; square sum: merge 8 -> 4
     vmovhlps xmm3, xmm0, xmm0                 ; high pair sum, [2,3,2,3] == vunpckhpd when both sources alias
     vmovhlps xmm4, xmm1, xmm1                 ; high pair square sum
     vaddps xmm0, xmm0, xmm3                   ; merge pairs sum (unchanged)
     vaddps xmm1, xmm1, xmm4                   ; merge pairs square sum (unchanged)
     vmovshdup xmm3, xmm0                      ; lane 1 sum, [1,1,3,3], only lane0 consumed
     vmovshdup xmm4, xmm1                      ; lane 1 square sum
     vaddss xmm0, xmm0, xmm3                   ; scalar sum
     vaddss xmm1, xmm1, xmm4                   ; scalar square sum
; Recover variance and original mean from the shifted statistics
     vunpcklps xmm2, xmm0, xmm1                ; {sum, sqsum, ...}
     vdivps xmm2, xmm2, xmm5                   ; both divides, one instruction, same FP32 rounding, row-invariant divisor broadcast at entry
     vmovaps xmm0, xmm2                        ; shifted mean = lane 0
     vpermilps xmm1, xmm2, 0x55                ; E[(x-shift)^2]/dim = lane 1
     vaddss xmm3, xmm0, xmm6                   ; original mean = shifted_mean + shift, into dead xmm3, independent of the variance FMA
     vbroadcastss ymm4, xmm3                   ; ymm4 = broadcast mean
     vfnmadd231ss xmm1, xmm0, xmm0             ; variance still uses the SHIFTED mean in xmm0, one rounding
     vaddss xmm1, xmm1, dword [rel g_vit_cfg + VCFG_LN_EPS] ; variance + eps, data-driven from DB
     vsqrtss xmm1, xmm1, xmm1                  ; sqrt,variance + eps = std
; Exact 1/std via vdivss, not vrcpss approximation, need PyTorch-matching output
     vdivss xmm3, xmm7, xmm1                   ; xmm3 = 1.0 / std = inv_std, overwrites the dead mean scratch
     vbroadcastss ymm2, xmm3                   ; ymm2 = broadcast inv_std, ymm2/ymm3 held the merged accumulators, dead since the reduction
     mov edx, eax                              ; block countdown = dim/16, row-invariant, hoisted into eax
     xor ecx, ecx                              ; d = 0, register counter
     align 16
.vln_norm_loop:
     vmovups ymm0, [r12 + rcx*VK_F4]           ; x[d..d+7]
     vmovups ymm1, [r12 + rcx*VK_F4 + VK_8F_BYTES] ; x[d+8..d+15]
     vmovups ymm3, [r15 + rcx*VK_F4]           ; gamma[d..d+7]
     vmovups ymm6, [r15 + rcx*VK_F4 + VK_8F_BYTES] ; gamma[d+8..d+15]
     vsubps ymm0, ymm0, ymm4                   ; x - mean, block A
     vsubps ymm1, ymm1, ymm4                   ; x - mean, block B
     vmulps ymm0, ymm0, ymm2                   ; same exact inv_std multiplication, block A
     vmulps ymm1, ymm1, ymm2                   ; same exact inv_std multiplication, block B
     vfmadd213ps ymm0, ymm3, [rbx + rcx*VK_F4] ; y = gamma * normalized + beta, block A
     vfmadd213ps ymm1, ymm6, [rbx + rcx*VK_F4 + VK_8F_BYTES] ; block B, gamma-B now in ymm6
     vmovups [r14 + rcx*VK_F4], ymm0           ; store output A
     vmovups [r14 + rcx*VK_F4 + VK_8F_BYTES], ymm1 ; store output B
     add ecx, 16                               ; d += 16, two YMM blocks
     dec edx                                   ; one 16-element block done, independent chain
     jnz .vln_norm_loop                        ; more blocks, bottom-tested
; Advance to next row
     add r12, r13                              ; in += dim*4
     add r14, r13                              ; out += dim*4
     inc esi                                   ; r++, register counter, no stack traffic
     cmp esi, r8d                              ; r < n_rows?, cached in r8d after the pointer caches
     jl .vln_row_loop                          ; next row
.vln_done:
     vzeroupper                                ; clean AVX state
     vmovdqu xmm6, vln_xmm6                    ; restore callee-saved xmm6
     vmovdqu xmm7, vln_xmm7                    ; restore callee-saved xmm7
     vmovdqu xmm8, vln_xmm8                    ; restore callee-saved xmm8
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore non-volatiles, return
