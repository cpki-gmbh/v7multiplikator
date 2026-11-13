; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; Reciprocal contract, aligned with kt_quantize_c_ref and v7d_quantize :
; the stored scale is wmax / 127.0f, line below, and the multiply
; reciprocal is 1.0f / scale computed over the ALREADY-ROUNDED stored
; scale, the same double-rounding form the C reference and the standalone
; quantize kernel use, so the fused kernel matches the
; v7s_rmsnorm -> v7d_quantize chain bit-for-bit.
; ============================================================================

; ============================================================================
; Constants for the fused kernel
; ============================================================================
%define RNQ_YMM_BLOCK 32                       ; 32-byte stride between consecutive YMM blocks
%define RNQ_GS 64                              ; group size for quantization, 64 floats
%define RNQ_GS_SHIFT 6                         ; log2,RNQ_GS : n/64 = n>>6
%define RNQ_GS_BYTES 256                       ; GS * sizeof,float = 64 * 4 = 256 bytes per group
%define RNQ_FLOAT4_SCALE 4                     ; scale factor for float pointer arithmetic, sizeof,float
%define RNQ_QK_INT8_2 32                       ; byte offset for XMM int8 block 2, elements 32..47
%define RNQ_ARG5_OFF 48                        ; 5th argument offset from rbp, caller's stack frame
%define RNQ_YMM_2X 64                          ; RNQ_YMM_BLOCK * 2, offset for block 2
%define RNQ_YMM_3X 96                          ; RNQ_YMM_BLOCK * 3, offset for block 3
%define RNQ_YMM_4X 128                         ; RNQ_YMM_BLOCK * 4, offset for block 4
%define RNQ_YMM_5X 160                         ; RNQ_YMM_BLOCK * 5, offset for block 5
%define RNQ_YMM_6X 192                         ; RNQ_YMM_BLOCK * 6, offset for block 6
%define RNQ_YMM_7X 224                         ; RNQ_YMM_BLOCK * 7, offset for block 7
%define RNQ_YMM_8X 256                         ; RNQ_YMM_BLOCK * 8, one 64-float iteration advance

%macro RNQ_QUANTIZE_REG 5
     vmulps ymm2, ymm1, %1                     ; block 0 * reciprocal
%if %5 = 0
     vmovdqa ymm6, [rel qk_perm_0415]          ; %1 is now dead: first expansion caches the permutation in ymm6
%endif
     vmulps ymm3, ymm1, %2                     ; block 1 * reciprocal
     vcvtps2dq ymm2, ymm2                      ; convert block 0 while later multiplications issue
     vmulps ymm4, ymm1, %3                     ; block 2 * reciprocal
     vcvtps2dq ymm3, ymm3                      ; convert block 1
     vmulps ymm5, ymm1, %4                     ; block 3 * reciprocal
     vcvtps2dq ymm4, ymm4                      ; convert block 2
     vcvtps2dq ymm5, ymm5                      ; convert block 3
     vpackssdw ymm2, ymm2, ymm3                ; unchanged first int16 pack
     vpackssdw ymm4, ymm4, ymm5                ; pack 1
     vpacksswb ymm2, ymm2, ymm4                ; 32 int16 -> 32 int8, saturating
     vpmaxsb ymm2, ymm2, [rel qk_min127_b8]    ; -128 -> -127 across all 32 byte lanes
     vpermd ymm2, ymm6, ymm2                   ; restore linear byte order; mask cached in ymm6 from the first expansion
     vmovdqu [rcx + %5], ymm2                  ; store 32 int8 values
%endmacro

%macro RNQ_STAGE1_SCALE 2
     vmulps %1, ymm0, [rdx + rax + %2]         ; dst = scale * x, folded load, matches C ref order
%endmacro

%macro RNQ_STAGE2_WEIGHT 2
     vmulps %1, %1, [r12 + rax + %2]           ; dst *= weight -> norm[block]
%endmacro
     section .text                             ; self-declare: last kernel in the block, its trailing section decides where the SMC-region end label, v7_kernels.inc lands, so it must be .text explicitly
     SF_START v7s_rmsnorm_quantize, rbx, r12   ; prologue: emits the global + entry label, saves the non-volatiles used
; Win64 preserves only the LOW 128 bits of xmm6-15, upper YMM halves are
; volatile, ten 16-byte SF_XMM slots halve the save/restore traffic
     SF_XMM rnq_xmm6                           ; xmm6 low half spill slot, norm[0..7]
     SF_XMM rnq_xmm7                           ; xmm7 low half spill slot, norm[8..15]
     SF_XMM rnq_xmm8                           ; xmm8 low half spill slot, norm[16..23]
     SF_XMM rnq_xmm9                           ; xmm9 low half spill slot, norm[24..31]
     SF_XMM rnq_xmm10                          ; xmm10 low half spill slot, norm[32..39]
     SF_XMM rnq_xmm11                          ; xmm11 low half spill slot, norm[40..47]
     SF_XMM rnq_xmm12                          ; xmm12 low half spill slot, norm[48..55]
     SF_XMM rnq_xmm13                          ; xmm13 low half spill slot, norm[56..63]
     SF_XMM rnq_xmm14                          ; xmm14 low half spill slot, abs_mask
     SF_XMM rnq_xmm15                          ; xmm15 low half spill slot, hoisted 127.0f
     SF_COMMIT                                 ; commit stack frame locals
; Save callee-saved XMM low halves that this kernel uses, xmm6-15
     vmovdqu rnq_xmm6, xmm6                    ; save xmm6, norm[0..7]
     vmovdqu rnq_xmm7, xmm7                    ; save xmm7, norm[8..15]
     vmovdqu rnq_xmm8, xmm8                    ; save xmm8, norm[16..23]
     vmovdqu rnq_xmm9, xmm9                    ; save xmm9, norm[24..31]
     vmovdqu rnq_xmm10, xmm10                  ; save xmm10, norm[32..39]
     vmovdqu rnq_xmm11, xmm11                  ; save xmm11, norm[40..47]
     vmovdqu rnq_xmm12, xmm12                  ; save xmm12, norm[48..55]
     vmovdqu rnq_xmm13, xmm13                  ; save xmm13, norm[56..63]
     vmovdqu rnq_xmm14, xmm14                  ; save xmm14, abs_mask
     vmovdqu rnq_xmm15, xmm15                  ; save xmm15, hoisted 127.0f
; === Precompute, float n for RMSNorm division, keeps single rounding ===
     vcvtsi2ss xmm14, xmm14, r9d               ; scalar float n; upper lanes are never used before the full ymm14 abs-mask reload
; === Save input parameters into non-volatile registers ===
     mov rbx, r8                               ; rbx = out_s, scales base, stable across all calls
     xor eax, eax                              ; rax = byte offset from the fixed bases, reset for Phase 2
     mov r12, [rbp + RNQ_ARG5_OFF]             ; r12 = weight, 5th arg from caller's stack frame
     vxorps ymm0, ymm0, ymm0                   ; sumsq accumulator A, blocks 0,4,8,...
     vxorps ymm1, ymm1, ymm1                   ; sumsq accumulator B, blocks 1,5,9,...
     vxorps ymm10, ymm10, ymm10                ; sumsq accumulator C, blocks 2,6,10,...
     vxorps ymm11, ymm11, ymm11                ; sumsq accumulator D, blocks 3,7,11,...
     mov r8d, r9d                              ; r8d = n, total dimension, multiple of 64
     shr r8d, RNQ_GS_SHIFT                     ; r8d = n / 64, 8-block iterations, always >= 1
     align 16
.rnq_p1_vec_loop:
     vmovups ymm2, [rdx + rax]                 ; x[block0]
     vmovups ymm4, [rdx + rax + RNQ_YMM_BLOCK] ; x[block1]
     vmovups ymm3, [rdx + rax + RNQ_YMM_2X]    ; x[block2]
     vmovups ymm5, [rdx + rax + RNQ_YMM_3X]    ; x[block3]
     vfmadd231ps ymm0, ymm2, ymm2              ; sumsq_A += x[block0]^2
     vfmadd231ps ymm1, ymm4, ymm4              ; sumsq_B += x[block1]^2
     vfmadd231ps ymm10, ymm3, ymm3             ; sumsq_C += x[block2]^2
     vfmadd231ps ymm11, ymm5, ymm5             ; sumsq_D += x[block3]^2
     vmovups ymm6, [rdx + rax + RNQ_YMM_4X]    ; x[block4]
     vmovups ymm7, [rdx + rax + RNQ_YMM_5X]    ; x[block5]
     vmovups ymm8, [rdx + rax + RNQ_YMM_6X]    ; x[block6]
     vmovups ymm9, [rdx + rax + RNQ_YMM_7X]    ; x[block7]
     vfmadd231ps ymm0, ymm6, ymm6              ; sumsq_A += x[block4]^2
     vfmadd231ps ymm1, ymm7, ymm7              ; sumsq_B += x[block5]^2
     vfmadd231ps ymm10, ymm8, ymm8             ; sumsq_C += x[block6]^2
     vfmadd231ps ymm11, ymm9, ymm9             ; sumsq_D += x[block7]^2
     add rax, RNQ_YMM_8X                       ; advance offset by 8 blocks, 256 bytes = 64 floats
     dec r8d                                   ; one fewer iteration, dec+jne macro-fuses on Zen 4, same 3 bytes
     jne .rnq_p1_vec_loop                      ; continue unrolled loop
.rnq_p1_reduce:
; Pairwise merge of the four accumulators, identical tree to v7s_rmsnorm
     vaddps ymm0, ymm0, ymm1                   ; merge A+B
     vaddps ymm10, ymm10, ymm11                ; merge C+D
     vaddps ymm0, ymm0, ymm10                  ; merge the pairs
     HADD_YMM xmm0, ymm0, xmm1                 ; Reduce to scalar
; ===== Compute RMSNorm scale via exact sqrt+div, IEEE-754, 0 ULP error =====
     vdivss xmm0, xmm0, xmm14                  ; mean_sq = sumsq / n, single rounding, matches C ref
     vaddss xmm0, xmm0, [rel g_rms_eps]        ; val = mean_sq + eps, per-model, from DB via BSS
     vsqrtss xmm0, xmm0, xmm0                  ; sqrt,val, Inf stays Inf, scale becomes 0 ; NaN only if sumsq was NaN
     vmovss xmm1, [rel rpe_one]                ; 1.0f numerator for the scale divide
     vdivss xmm0, xmm1, xmm0                   ; scale = 1.0 / sqrt,val, IEEE-754 exact
     vbroadcastss ymm0, xmm0                   ; ymm0 = scale, broadcast once, live across all groups
; ===== Phase 2: Normalize 64 floats into ymm6-13, in-register, no memory store =====
; Reset input pointer to start
     xor eax, eax                              ; rax = byte offset from the fixed bases, in_x = rdx, weight = r12
; Compute number of 64-float groups, = n / 64; r9d still holds n from entry
     shr r9d, RNQ_GS_SHIFT                     ; r9d = ngroups, >= 1 guaranteed: n>=64, n%64==0
     vmovdqa ymm14, [rel qk_abs_mask]          ; ymm14 = |mask|, reused by all 8 vandps per group
     vbroadcastss ymm15, [rel qk_f127]         ; ymm15 = 127.0f, numerator for the group reciprocal divide
     align 16
.rnq_group_loop:
; RNQ_STAGE1_SCALE / RNQ_STAGE2_WEIGHT are defined beside RNQ_QUANTIZE_REG
; at the top of the file, hoisted out of this loop, %macro emits nothing .
; ---- Pass 1/3: dst = scale * x[block] * weight[block], paired per register ----
     RNQ_STAGE1_SCALE ymm6, 0                  ; norm[0..7]
     RNQ_STAGE1_SCALE ymm7, RNQ_YMM_BLOCK      ; norm[8..15]
     RNQ_STAGE2_WEIGHT ymm6, 0                 ; norm[0..7] *= weight
     RNQ_STAGE2_WEIGHT ymm7, RNQ_YMM_BLOCK     ; norm[8..15] *= weight
     RNQ_STAGE1_SCALE ymm8, RNQ_YMM_2X         ; norm[16..23]
     RNQ_STAGE1_SCALE ymm9, RNQ_YMM_3X         ; norm[24..31]
     RNQ_STAGE2_WEIGHT ymm8, RNQ_YMM_2X        ; norm[16..23] *= weight
     RNQ_STAGE2_WEIGHT ymm9, RNQ_YMM_3X        ; norm[24..31] *= weight
     RNQ_STAGE1_SCALE ymm10, RNQ_YMM_4X        ; norm[32..39]
     RNQ_STAGE1_SCALE ymm11, RNQ_YMM_5X        ; norm[40..47]
     RNQ_STAGE2_WEIGHT ymm10, RNQ_YMM_4X       ; norm[32..39] *= weight
     RNQ_STAGE2_WEIGHT ymm11, RNQ_YMM_5X       ; norm[40..47] *= weight
     RNQ_STAGE1_SCALE ymm12, RNQ_YMM_6X        ; norm[48..55]
     RNQ_STAGE1_SCALE ymm13, RNQ_YMM_7X        ; norm[56..63]
     RNQ_STAGE2_WEIGHT ymm12, RNQ_YMM_6X       ; norm[48..55] *= weight
     RNQ_STAGE2_WEIGHT ymm13, RNQ_YMM_7X       ; norm[56..63] *= weight
     vandps ymm2, ymm6, ymm14                  ; ymm2 = |norm[0..7]|
     vandps ymm3, ymm7, ymm14                  ; ymm3 = |norm[8..15]|
     vmaxps ymm2, ymm2, ymm3                   ; ymm2 = max,|0..15|
     vandps ymm3, ymm8, ymm14                  ; ymm3 = |norm[16..23]|
     vandps ymm4, ymm9, ymm14                  ; ymm4 = |norm[24..31]|
     vmaxps ymm3, ymm3, ymm4                   ; ymm3 = max,|16..31|
     vmaxps ymm2, ymm2, ymm3                   ; ymm2 = max,|0..31|
     vandps ymm3, ymm10, ymm14                 ; ymm3 = |norm[32..39]|
     vandps ymm4, ymm11, ymm14                 ; ymm4 = |norm[40..47]|
     vmaxps ymm3, ymm3, ymm4                   ; ymm3 = max,|32..47|
     vandps ymm4, ymm12, ymm14                 ; ymm4 = |norm[48..55]|
     vandps ymm5, ymm13, ymm14                 ; ymm5 = |norm[56..63]|
     vmaxps ymm4, ymm4, ymm5                   ; ymm4 = max,|48..63|
     vmaxps ymm3, ymm3, ymm4                   ; ymm3 = max,|32..63|
; Level 3: final combine -> ymm2, ymm0 = scale stays live for next group
; NaN policy, same exposure as quantize_kernel's main path, whose header
; comment documents it precisely : vmaxps selects src2 on unordered pairs, so a
; NaN at the final-lane position can sentinel wmax into the eps floor and
; squeeze the whole group to +-127. NaN activations are an impossible state
; for the engine, every producer clamps or is itself quantized, so the
; position-dependent behavior is DOCUMENTED here, never handled.
     vmaxps ymm2, ymm2, ymm3                   ; ymm2 = max,|0..63| across entire group
     vextractf128 xmm3, ymm2, 1                ; xmm3 = upper 4 maxes
     vmaxps xmm2, xmm2, xmm3                   ; max across lower+upper 128-bit lanes
     vmovhlps xmm3, xmm2, xmm2                 ; pair swap, [2,3,2,3]: lanes 0/1 match 0x4E, upper scratch differs
     vmaxps xmm2, xmm2, xmm3                   ; max of pairs
     vmovshdup xmm3, xmm2                      ; xmm3[0] = xmm2[1] = b1, odd-lane duplicate
     vmaxss xmm2, xmm2, xmm3                   ; lane 0 = max,b0, b1 = scalar wmax
     vmaxss xmm2, xmm2, [rel scalar_epsilon]   ; xmm2 = max,wmax, eps, DAZ-invariant
     vmovss xmm3, [rel rpe_one]                ; 1.0f numerator, xmm3 dead after the max tree
     vdivss xmm4, xmm2, xmm15                  ; scale = wmax / 127.0f
     vmovss [rbx], xmm4                        ; out_s[group_idx] = scale
     vdivss xmm1, xmm3, xmm4                   ; 1.0f / scale, exact IEEE division
     vbroadcastss ymm1, xmm1                   ; reciprocal scale, broadcast to all lanes
; --- Blocks 0..3: ymm6-9 -> [out_q + 0] ---
     RNQ_QUANTIZE_REG ymm6, ymm7, ymm8, ymm9, 0 ; quantize norms 0..31 -> [out_q+0]
; --- Blocks 4..7: ymm10-13 -> [out_q + 32] ---
     RNQ_QUANTIZE_REG ymm10, ymm11, ymm12, ymm13, RNQ_QK_INT8_2 ; quantize norms 32..63 -> [out_q+32]
; === Advance pointers for next group ===
     add rcx, RNQ_GS                           ; out_q += 64, bytes for int8
     add rax, RNQ_GS_BYTES                     ; advance the shared offset by one group, in_x and weight advance together
; weight += 256 bytes, 64 floats
     add rbx, RNQ_FLOAT4_SCALE                 ; advance scale pointer, saves r10d register & inc
     dec r9d                                   ; ngroups_remaining--, dec+jnz macro-fuses on Zen 4, same 3 bytes
     jnz .rnq_group_loop                       ; next group
.rnq_done:
; Restore callee-saved XMM low halves, vzeroupper first: avoids AVX-SSE
; transition penalties in the caller; upper halves are volatile anyway
     CLEANUP_YMM                               ; vzeroupper for AVX ABI compliance
     vmovdqu xmm6, rnq_xmm6                    ; restore xmm6, norm[0..7]
     vmovdqu xmm7, rnq_xmm7                    ; restore xmm7, norm[8..15]
     vmovdqu xmm8, rnq_xmm8                    ; restore xmm8, norm[16..23]
     vmovdqu xmm9, rnq_xmm9                    ; restore xmm9, norm[24..31]
     vmovdqu xmm10, rnq_xmm10                  ; restore xmm10, norm[32..39]
     vmovdqu xmm11, rnq_xmm11                  ; restore xmm11, norm[40..47]
     vmovdqu xmm12, rnq_xmm12                  ; restore xmm12, norm[48..55]
     vmovdqu xmm13, rnq_xmm13                  ; restore xmm13, norm[56..63]
     vmovdqu xmm14, rnq_xmm14                  ; restore xmm14, abs_mask
     vmovdqu xmm15, rnq_xmm15                  ; restore xmm15, hoisted 127.0f
     SF_END rbx, r12                           ; epilogue: restore non-volatiles and return
%undef RNQ_YMM_BLOCK                           ; 32-byte stride between consecutive YMM blocks
%undef RNQ_GS                                  ; quantization group size, 64 floats
%undef RNQ_GS_SHIFT                            ; log2,RNQ_GS
%undef RNQ_GS_BYTES                            ; 256 bytes per group
%undef RNQ_FLOAT4_SCALE                        ; sizeof,float for pointer arithmetic
%undef RNQ_QK_INT8_2                           ; int8 output byte offset 32
%undef RNQ_ARG5_OFF                            ; 5th argument offset from rbp
%undef RNQ_YMM_2X                              ; block 2 byte offset
%undef RNQ_YMM_3X                              ; block 3 byte offset
%undef RNQ_YMM_4X                              ; block 4 byte offset
%undef RNQ_YMM_5X                              ; block 5 byte offset
%undef RNQ_YMM_6X                              ; block 6 byte offset
%undef RNQ_YMM_7X                              ; block 7 byte offset
%undef RNQ_YMM_8X                              ; 64-float iteration advance
%unmacro RNQ_QUANTIZE_REG 5                    ; quantize 4 YMMs -> 32 int8 macro, arity MUST match
%unmacro RNQ_STAGE1_SCALE 2                    ; normalize stage 1 macro
%unmacro RNQ_STAGE2_WEIGHT 2                   ; normalize stage 2 macro
