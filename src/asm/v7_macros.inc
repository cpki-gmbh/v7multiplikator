; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%ifndef V7_MACROS_INC
%define V7_MACROS_INC                          ; include guard, file scanned once per translation unit
%define V7M_FLOAT_SCALE 4                      ; SIB scale for float-array element addressing, sizeof,float
%define V7M_Q8_HI_OFF 32                       ; byte offset of the high 32B half inside a 64B Q8 weight load
%define V7M_Q8_PF_OWN 288                      ; own-row prefetch distance, bytes past the current 64B group
%define V7M_Q8_PF_NEXT 256                     ; next-row prefetch distance, bytes past the current 64B group
%define V7M_WSUB1_OFF 8                        ; int8 sub-block 1 byte offset inside the 32-byte weight group
%define V7M_WSUB2_OFF 16                       ; int8 sub-block 2 byte offset inside the 32-byte weight group
%define V7M_WSUB3_OFF 24                       ; int8 sub-block 3 byte offset inside the 32-byte weight group
%define TENSOR_S_OFF 8                         ; byte offset of the .scales pointer in v7_q8_tensor, struct in src/v7_lib.h, shared by q8_kernel, swiglu_kernel, matmul_dispatch, fused_dispatch
%define V7M_ACT_SUB1_OFF 32                    ; activation sub-block 1 byte offset, 8 floats * 4 bytes
%define V7M_ACT_SUB2_OFF 64                    ; activation sub-block 2 byte offset
%define V7M_ACT_SUB3_OFF 96                    ; activation sub-block 3 byte offset
%define PQ_BIAS_ELEMS 32                       ; PQ_BIAS_ADD: floats consumed per iteration, init FATALs unless dsum and kv_dim are multiples of 32
%define PQ_BIAS_BLOCK 128                      ; PQ_BIAS_ADD: byte advance per iteration, 32 floats * 4 bytes
%define PQ_BIAS_STRIDE1 32                     ; PQ_BIAS_ADD: byte offset of floats 8..15 inside the block
%define PQ_BIAS_STRIDE2 64                     ; PQ_BIAS_ADD: byte offset of floats 16..23 inside the block
%define PQ_BIAS_STRIDE3 96                     ; PQ_BIAS_ADD: byte offset of floats 24..31 inside the block
%define SL_F32_NOT_SIGN_MASK 0x7FFFFFFF        ; binary32 with the sign bit STRIPPED; use for "|x|" and "magnitude zero". The sign bit ITSELF is 0x80000000
%define SL_F32_EXP_MASK 0x7F800000             ; binary32 exponent field all ones = Inf/NaN class
; ============================================================================
; HADD_YMM dst_xmm, src_ymm, tmp_xmm
; Reduces 8 packed floats in src_ymm to a scalar in the low lane of dst_xmm.
; REQUIRES: dst_xmm IS the low 128 bits of src_ymm. Under that contract the
; low-half extract is a self-copy, VEX.128 vextractf128 writes the
; 128 bits the destination already holds and leaves its upper half
; alone, so it is omitted and the vaddps reads bits 127:0 of dst_xmm
; directly. Every call site in the tree satisfies this; a destination
; that is not src_ymm's low half must reduce into the volatile alias
; Uses AMD Zen-friendly shuffle sequence, no vhaddps .
; Overwrites: tmp_xmm, flags.
; ============================================================================

%macro HADD_YMM 3
     vextractf128 %3, %2, 1                    ; %3 = high 4 floats of src_ymm
     vaddps %1, %1, %3                         ; %1 = low + high, 128-bit VEX form: reads only bits 127:0 of %1
     vmovshdup %3, %1                          ; %3 = [e1, e1, e3, e3]
     vaddps %1, %1, %3                         ; %1 = [e0+e1, e1+e1, e2+e3, e3+e3]
     vmovhlps %3, %1, %1                       ; %3 = [e2+e3, e3+e3] in low qword, only element 0 used; 3-byte VEX vs 4-byte vunpckhpd
     vaddss %1, %1, %3                         ; %1 = scalar sum
%endmacro
; HADD4_YMM: reduce ymm0..ymm3 to xmm0..xmm3 with four independent
; shuffle/add chains. Scratch xmm4..xmm7 are destroyed. Each accumulator uses
; exactly the same addition tree as HADD_YMM. Only use at sites whose prologue
; saves the non-volatile low 128 bits of xmm6/xmm7 and that hold no live
; xmm4..xmm7 input.

%macro HADD4_YMM 0
     vextractf128 xmm4, ymm0, 1                ; acc0 high 4 floats
     vextractf128 xmm5, ymm1, 1                ; acc1 high 4 floats
     vextractf128 xmm6, ymm2, 1                ; acc2 high 4 floats
     vextractf128 xmm7, ymm3, 1                ; acc3 high 4 floats
     vaddps xmm0, xmm0, xmm4                   ; acc0 low + high
     vaddps xmm1, xmm1, xmm5                   ; acc1 low + high
     vaddps xmm2, xmm2, xmm6                   ; acc2 low + high
     vaddps xmm3, xmm3, xmm7                   ; acc3 low + high
     vmovshdup xmm4, xmm0                      ; acc0 odd-lane pair
     vmovshdup xmm5, xmm1                      ; acc1 odd-lane pair
     vmovshdup xmm6, xmm2                      ; acc2 odd-lane pair
     vmovshdup xmm7, xmm3                      ; acc3 odd-lane pair
     vaddps xmm0, xmm0, xmm4                   ; acc0 merge pairs
     vaddps xmm1, xmm1, xmm5                   ; acc1 merge pairs
     vaddps xmm2, xmm2, xmm6                   ; acc2 merge pairs
     vaddps xmm3, xmm3, xmm7                   ; acc3 merge pairs
     vmovhlps xmm4, xmm0, xmm0                 ; acc0 high qword
     vmovhlps xmm5, xmm1, xmm1                 ; acc1 high qword
     vmovhlps xmm6, xmm2, xmm2                 ; acc2 high qword
     vmovhlps xmm7, xmm3, xmm3                 ; acc3 high qword
     vaddss xmm0, xmm0, xmm4                   ; acc0 scalar sum
     vaddss xmm1, xmm1, xmm5                   ; acc1 scalar sum
     vaddss xmm2, xmm2, xmm6                   ; acc2 scalar sum
     vaddss xmm3, xmm3, xmm7                   ; acc3 scalar sum
%endmacro
; ============================================================================
; REDUCE_YMM_TO_SCALAR_FAST dst_xmm, src_ymm, tmp1_xmm, tmp2_xmm
; Alternate reduction: delegates to HADD_YMM for fast shuffle-based reduction.
; tmp2_xmm is unused, retained for API compatibility .
; Overwrites: tmp1_xmm, flags.
; ============================================================================

%macro REDUCE_YMM_TO_SCALAR_FAST 4
     HADD_YMM %1, %2, %3                       ; the fast path IS the hadd path, delegate, arg4 retained for call-site compatibility
%endmacro
; ============================================================================
; EVAL_SIGMOID regprefix, ss|ps, sign_mask_src
; sigmoid,x = 1/,1+exp,-x via clamped 4th-order minimax polynomial
; clamp to [-80,+80] keeps the polynomial in range and an EXACT final
; division, vdivss/vdivps, no rcp approximation.regprefix is "x" or
; "y"; the macro addresses xmm/ymm0..5 as %1mm0..%1mm5.
; Overwrites: xmm0-xmm5, or ymm0-ymm5, flags; in ss mode ALSO eax.
; sign_mask_src is the sign-bit mask constant for the negation XOR.
; ============================================================================

%macro EVAL_SIGMOID 3
; Step 1: negate x
     vxorps %1mm0, %1mm0, %3                   ; Negate via sign-bit XOR
; Step 2: clamp to [-80, +80], safe for 4th-order minimax polynomial
%ifidn %2, ss
     vmaxss %1mm0, %1mm0, [rel scalar_m80]     ; Clamp lower: x >= -80, memory operand, no staging load
     vminss %1mm0, %1mm0, [rel scalar_80]      ; Clamp upper: x <= +80, memory operand, no staging load
%else
     vmaxps %1mm0, %1mm0, [rel ymm_m80]        ; Clamp lower: x >= -80 per lane, memory operand, no vbroadcastss
     vminps %1mm0, %1mm0, [rel ymm_80]         ; Clamp upper: x <= +80 per lane, memory operand, no vbroadcastss
%endif
; Step 3: fx = x * log2,e, ipart = round-to-nearest,fx, fpart = fx - ipart
%ifidn %2, ss
     vmulss %1mm1, %1mm0, [rel scalar_log2e]   ; fx = x * log2,e
     vroundss %1mm2, %1mm1, %1mm1, 0           ; ipart = round-to-nearest,fx
     vcvttss2si eax, %1mm2                     ; eax = int,ipart
     vsubss %1mm3, %1mm1, %1mm2                ; fpart = fx - ipart
     add eax, 127                              ; biased exponent, computed while the scalar Horner chain runs
     shl eax, 23                               ; shift into the float exponent field
     vmovd %1mm4, eax                          ; 2^ipart as float, xmm4 stays dead throughout the ss scalar polynomial, the poly chain lives in xmm5
%else
     vmulps %1mm1, %1mm0, [rel s_log2e_v]      ; fx = x * log2,e, all lanes, memory-folded constant
     vroundps %1mm2, %1mm1, 0                  ; ipart = round-to-nearest,fx, all lanes
     vsubps %1mm3, %1mm1, %1mm2                ; fpart = fx - ipart, all lanes
     vcvttps2dq %1mm5, %1mm2                   ; ipart as int32, all lanes
     vpaddd %1mm5, %1mm5, [rel swk_bias_127]   ; Add exponent bias 127
     vpslld %1mm5, %1mm5, 23                   ; Shift into float exponent field
%endif
; Step 4: poly,fpart =, ,,,,C5*fpart + C4 *fpart + C3 *fpart + C2 *fpart + C1 *fpart + 1.0
; 5th-order minimax from math_data.inc, the same set the attention INLINE_EXP
; uses, 1.27e-7 max rel err
%ifidn %2, ss
     vmovss %1mm5, [rel attn_c5]               ; scalar C5 load; only lane 0 participates in the ss chain
     vfmadd213ss %1mm5, %1mm3, [rel attn_c4]   ; xmm5 = fpart*C5 + C4
     vfmadd213ss %1mm5, %1mm3, [rel attn_c3]   ; xmm5 = fpart*prev + C3
     vfmadd213ss %1mm5, %1mm3, [rel attn_c2]   ; xmm5 = fpart*prev + C2
     vfmadd213ss %1mm5, %1mm3, [rel attn_c1]   ; xmm5 = fpart*prev + C1
     vfmadd213ss %1mm5, %1mm3, [rel rpe_one]   ; xmm5 = fpart*prev + 1.0
%else
     vbroadcastss %1mm4, [rel attn_c5]         ; Broadcast C5 into the FMA accumulator
     vfmadd213ps %1mm4, %1mm3, [rel attn_c4_vec] ; poly = fpart*acc + same coefficient in every lane
     vfmadd213ps %1mm4, %1mm3, [rel attn_c3_vec] ; poly = fpart*acc + same coefficient in every lane
     vfmadd213ps %1mm4, %1mm3, [rel attn_c2_vec] ; poly = fpart*acc + same coefficient in every lane
     vfmadd213ps %1mm4, %1mm3, [rel attn_c1_vec] ; poly = fpart*acc + same coefficient in every lane
     vfmadd213ps %1mm4, %1mm3, [rel ymm_one]   ; poly = fpart*prev + 1.0, ymm_one is all-ones, so every lane reads 1.0, identical to a broadcast
%endif
; Step 5: exp,-x = 2^ipart * poly
%ifidn %2, ss
     vmulss %1mm0, %1mm4, %1mm5                ; exp,-x = 2^ipart * poly, poly lives in xmm5 on the ss branch
%else
     vmulps %1mm0, %1mm5, %1mm4                ; exp,-x = 2^ipart * poly, all lanes
%endif
; Step 6: sigmoid = 1/,1+exp,-x via exact division, 0 ULP
%ifidn %2, ss
     vaddss %1mm0, %1mm0, [rel rpe_one]        ; 1 + exp,-x, memory operand
     vmovss %1mm1, [rel rpe_one]               ; scalar numerator 1.0; only lane 0 is read by VDIVSS
     vdivss %1mm0, %1mm1, %1mm0                ; sigmoid = 1/,1+exp,-x
%else
     vbroadcastss %1mm1, [rel rpe_one]         ; broadcast 1.0 once, register feeds both add and div
     vaddps %1mm0, %1mm0, %1mm1                ; 1 + exp,-x, all lanes, register operand, no ymm_one load
     vdivps %1mm0, %1mm1, %1mm0                ; sigmoid = 1/,1+exp,-x, all lanes
%endif
%endmacro
; ============================================================================
; SELECT_BATCH_COMMON nj_reg32, dst_reg32
; Internal: shared micro-batching + INI fallback for both SELECT_BATCH
; and CALLER_SELECT_BATCH. Uses caller-supplied global names.
; nj_reg32 = job count, preserved
; dst_reg32 = output batch size, overwritten then set
; %3 = small_global, batch_small or caller_batch_small
; %4 = med_global, batch_medium or caller_batch_medium
; %5 = large_global, batch_large or caller_batch_large
; %6 = huge_global, batch_huge or caller_batch_huge
; %7 = med_thresh, batch_med_thresh or caller_batch_med_thresh
; %8 = large_thresh, batch_large_thresh or caller_batch_large_thresh
; %9 = huge_thresh, batch_huge_thresh or caller_batch_huge_thresh
; Overwrites: flags only. Preserves nj_reg32.
; ============================================================================

%macro SELECT_BATCH_COMMON 9
; Micro-batching: workers*2->1, *4->2, *8->4, *16->8
     mov %2, [rel num_workers]                 ; worker count, RIP-relative, consistent with the dispatch-side RIP-relative accesses, no base reloc
     add %2, %2                                ; workers * 2
     cmp %1, %2                                ; num_jobs < workers*2?
     jge %%tier_4                              ; no, check next tier
     mov %2, 1                                 ; batch = 1
     jmp %%done                                ; tier hit, batch decided
%%tier_4:
     add %2, %2                                ; workers * 4
     cmp %1, %2                                ; num_jobs < workers*4?
     jge %%tier_8                              ; no, check next tier
     mov %2, 2                                 ; batch = 2
     jmp %%done                                ; tier hit, batch decided
%%tier_8:
     add %2, %2                                ; workers * 8
     cmp %1, %2                                ; num_jobs < workers*8?
     jge %%tier_16                             ; no, check next tier
     mov %2, 4                                 ; batch = 4
     jmp %%done                                ; tier hit, batch decided
%%tier_16:
     add %2, %2                                ; workers * 16
     cmp %1, %2                                ; num_jobs < workers*16?
     jge %%tier_normal                         ; no, use INI thresholds
     mov %2, 8                                 ; batch = 8
     jmp %%done                                ; tier hit, batch decided
%%tier_normal:
; Descending threshold checks: huge -> large -> medium -> small
     mov %2, dword [%3]                        ; default small
     cmp %1, dword [%9]                        ; nj >= huge_thresh?
     jge %%use_huge                            ; yes
     cmp %1, dword [%8]                        ; nj >= large_thresh?
     jge %%use_large                           ; yes
     cmp %1, dword [%7]                        ; nj >= med_thresh?
     jl %%done                                 ; no, keep small
     mov %2, dword [%4]                        ; medium batch
     jmp %%done                                ; batch decided
%%use_large:
     mov %2, dword [%5]                        ; large batch
     jmp %%done                                ; batch decided
%%use_huge:
     mov %2, dword [%6]                        ; huge batch
%%done:
%endmacro
; ============================================================================
; SELECT_BATCH nj_reg32, dst_reg32
; Worker-side batch size selection, uses batch_* globals .
; Overwrites: flags only. Preserves both operand registers.
; ============================================================================

%macro SELECT_BATCH 2
     SELECT_BATCH_COMMON %1, %2, batch_small, batch_medium, batch_large, batch_huge, batch_med_thresh, batch_large_thresh, batch_huge_thresh ; worker-side INI globals
%endmacro
; ============================================================================
; CALLER_SELECT_BATCH nj_reg32, dst_reg32
; Caller-side batch size selection, uses caller_batch_* globals .
; Overwrites: flags only. Preserves both operand registers.
; ============================================================================

%macro CALLER_SELECT_BATCH 2
     SELECT_BATCH_COMMON %1, %2, caller_batch_small, caller_batch_medium, caller_batch_large, caller_batch_huge, caller_batch_med_thresh, caller_batch_large_thresh, caller_batch_huge_thresh ; caller-side INI globals
%endmacro

%macro F32_MADD_32 3
     vpmovsxbd ymm9, [r10 + rcx + %1]          ; load+sign-extend 8 int8 weights
     vpmovsxbd ymm10, [r10 + rcx + %1 + V7M_WSUB1_OFF] ; load next 8 int8 weights
     vcvtdq2ps ymm9, ymm9                      ; convert weights 0..7 early
     vcvtdq2ps ymm10, ymm10                    ; convert weights 8..15 early
     vmovups ymm4, [rdi + rcx*V7M_FLOAT_SCALE + %2] ; load 8 float activations
     vpmovsxbd ymm11, [r10 + rcx + %1 + V7M_WSUB2_OFF] ; load next 8 int8 weights
     vmovups ymm5, [rdi + rcx*V7M_FLOAT_SCALE + %2 + V7M_ACT_SUB1_OFF] ; load next 8 float activations
     vpmovsxbd ymm12, [r10 + rcx + %1 + V7M_WSUB3_OFF] ; load next 8 int8 weights
     vmovups ymm6, [rdi + rcx*V7M_FLOAT_SCALE + %2 + V7M_ACT_SUB2_OFF] ; load next 8 float activations
     vmovups ymm7, [rdi + rcx*V7M_FLOAT_SCALE + %2 + V7M_ACT_SUB3_OFF] ; load next 8 float activations
     vmulps ymm9, ymm9, %3                     ; weights 0..7 *= scale
     vcvtdq2ps ymm11, ymm11                    ; weights 16..23 to float
     vmulps ymm10, ymm10, %3                   ; weights 8..15 *= scale
     vcvtdq2ps ymm12, ymm12                    ; weights 24..31 to float
     vfmadd231ps ymm0, ymm4, ymm9              ; accum0 += activation * weight_float
     vmulps ymm11, ymm11, %3                   ; weights 16..23 *= scale
     vfmadd231ps ymm1, ymm5, ymm10             ; accum1 += activation * weight_float
     vmulps ymm12, ymm12, %3                   ; weights 24..31 *= scale
     vfmadd231ps ymm2, ymm6, ymm11             ; accum2 += activation * weight_float
     vfmadd231ps ymm3, ymm7, ymm12             ; accum3 += activation * weight_float
%endmacro

%macro RMSNORM_ROPE_BODY 6
; Pass 1: Sum of squares, 4 accumulators, 32-element unrolled
; r13d = head_dim, loaded by the caller before macro expansion, no reload here
     xor ecx, ecx                              ; element counter
     vxorps ymm0, ymm0, ymm0                   ; acc A, 256-bit: VEX.128 preserves bits 255:128, which the FMA below and HADD_YMM both read
     vxorps ymm1, ymm1, ymm1                   ; acc B, 256-bit, same reason
     vxorps ymm2, ymm2, ymm2                   ; acc C, 256-bit, same reason
     vxorps ymm3, ymm3, ymm3                   ; acc D, 256-bit, same reason
     mov edx, r13d                             ; head_dim
     shr edx, 5                                ; edx = 32-element block count for pass 1
%%ss_loop:
     vmovups ymm4, [r12 + rcx*V7M_FLOAT_SCALE] ; load elements [0..7]
     vmovups ymm5, [r12 + rcx*V7M_FLOAT_SCALE + V7M_ACT_SUB1_OFF] ; load elements [8..15]
     vfmadd231ps ymm0, ymm4, ymm4              ; sum into acc A
     vfmadd231ps ymm1, ymm5, ymm5              ; sum into acc B
     vmovups ymm4, [r12 + rcx*V7M_FLOAT_SCALE + V7M_ACT_SUB2_OFF] ; load elements [16..23]
     vmovups ymm5, [r12 + rcx*V7M_FLOAT_SCALE + V7M_ACT_SUB3_OFF] ; load elements [24..31]
     vfmadd231ps ymm2, ymm4, ymm4              ; sum into acc C
     vfmadd231ps ymm3, ymm5, ymm5              ; sum into acc D
     add ecx, 32                               ; advance by 32 elements
     dec edx                                   ; one block done, bottom-tested, no head_dim test per iteration
     jnz %%ss_loop                             ; continue unrolled loop
; Combine 4 accumulators into ymm0
     vaddps ymm0, ymm0, ymm1                   ; ymm0 += ymm1
     vaddps ymm2, ymm2, ymm3                   ; ymm2 += ymm3
     vaddps ymm0, ymm0, ymm2                   ; ymm0 += ymm2
     HADD_YMM xmm0, ymm0, xmm4                 ; reduce 8 floats in ymm0 to scalar
; Compute RMSNorm scale via exact sqrt+div, 0 ULP
     COMPUTE_RMS_SCALE xmm0, %1, xmm0          ; mean_sq = sumsq/n, + eps, sqrt, reciprocal
     vbroadcastss ymm14, xmm0                  ; broadcast scale to all lanes
; Skip RoPE if pos == 0, cos=1, sin=0 => rotation is no-op
     cmp %5, 0                                 ; pos == 0?
     jz %%rmsnorm_only                         ; yes, skip rotation
; Pass 2: Fused RMSNorm scaling + RoPE rotation
     mov rdx, %2                               ; load norm weights pointer, volatile home, no non-volatile save needed
     mov r10, %3                               ; load cos table pointer, volatile home, no non-volatile save needed
     mov r11, %4                               ; load sin table pointer, volatile home, no non-volatile save needed
     shr r13d, 1                               ; half = head_dim / 2, pair count
     lea r8, [r12 + r13*V7M_FLOAT_SCALE]       ; r8 = second half of head, half floats offset, fused SIB; r13 zero-extended
     lea r9, [rdx + r13*V7M_FLOAT_SCALE]       ; r9 = second half of norm weights, fused SIB
     xor ecx, ecx                              ; pair counter
     shr r13d, 3                               ; block count = half/8, exact, the import's head_dim%32 FATAL, db_build_tool.c:1927: half=hd/2 is %16 so half%8==0; head_dim>=64 -> >=4 blocks
%%rp_loop:
     vmovups ymm4, [r10 + rcx*V7M_FLOAT_SCALE] ; load cos values, early: overlaps data loads
     vmovups ymm5, [r11 + rcx*V7M_FLOAT_SCALE] ; load sin values, early: overlaps data loads
     vmovups ymm0, [r12 + rcx*V7M_FLOAT_SCALE] ; load first half, even pairs
     vmovups ymm1, [r8 + rcx*V7M_FLOAT_SCALE]  ; load second half, odd pairs
     vmulps ymm0, ymm0, ymm14                  ; x_even *= rms_scale, scale first
     vmulps ymm1, ymm1, ymm14                  ; x_odd *= rms_scale, scale first
     vmulps ymm0, ymm0, [rdx + rcx*V7M_FLOAT_SCALE] ; x_even *= norm_weight[even], load folded
     vmulps ymm1, ymm1, [r9 + rcx*V7M_FLOAT_SCALE] ; x_odd *= norm_weight[odd], independent chain
     vmulps ymm2, ymm0, ymm4                   ; result_even = even * cos
     vmulps ymm3, ymm0, ymm5                   ; result_odd = even * sin
     vfnmadd231ps ymm2, ymm1, ymm5             ; result_even = even*cos - odd*sin
     vfmadd231ps ymm3, ymm1, ymm4              ; result_odd = even*sin + odd*cos
     vmovups [r12 + rcx*V7M_FLOAT_SCALE], ymm2 ; store rotated even
     vmovups [r8 + rcx*V7M_FLOAT_SCALE], ymm3  ; store rotated odd
     add ecx, 8                                ; advance by 8 pairs
     dec r13d                                  ; one block done, independent chain, removes serial cmp-on-ecx recurrence
     jnz %%rp_loop                             ; more blocks, continue, bottom-tested
     jmp %6                                    ; rotation complete, exit, MANDATORY: fall-through would enter %%rmsnorm_only and double-normalize
; pos=0 fast path: RMSNorm only, no RoPE rotation
%%rmsnorm_only:
     mov rdx, %2                               ; load norm weights pointer, volatile home, no non-volatile save needed
     xor ecx, ecx                              ; element counter
     mov eax, r13d                             ; head_dim zero-extends into rax, intact: reached via jz BEFORE the rotation path's shr
     shr eax, 3                                ; 8-element block count, u32 exact: head_dim is u32 and right shift stays u32
%%rn_loop:
     vmovups ymm0, [r12 + rcx*V7M_FLOAT_SCALE] ; load 8 elements
     vmulps ymm0, ymm0, ymm14                  ; x *= rms_scale, scale first, C ref order
     vmulps ymm0, ymm0, [rdx + rcx*V7M_FLOAT_SCALE] ; x *= norm_weight, same FP order, load folded
     vmovups [r12 + rcx*V7M_FLOAT_SCALE], ymm0 ; store result
     add ecx, 8                                ; advance by 8
     dec rax                                   ; one block done, independent chain
     jnz %%rn_loop                             ; continue
     jmp %6                                    ; done, MANDATORY exit, nothing follows the macro body's last path
%endmacro
; ============================================================================
; COMPUTE_RMS_SCALE sumsq_xmm, dim_src, dst_xmm
; scale = 1/sqrt,sumsq/dim + eps via exact IEEE ops, vdivss, vsqrtss,
; vdivss, no rcp/rsqrt approximation.dim_src may be an XMM register
; holding, float dim or a [rel float] memory operand.
; Overwrites: xmm1, flags. dst_xmm and dim_src are written/read only.
; ============================================================================

%macro COMPUTE_RMS_SCALE 3
     vdivss %1, %1, %2                         ; mean_sq = sumsq / dim, single correct rounding, matches C ref
     vaddss %1, %1, [rel g_rms_eps]            ; val = mean_sq + eps, per-model
     vmovss xmm1, [rel rpe_one]                ; 1.0f, prepared while the sqrt latency is outstanding
     vsqrtss %3, %1, %1                        ; sqrt_v = sqrt,val, exact
     vdivss %3, xmm1, %3                       ; scale = 1.0 / sqrt_v, exact div
%endmacro

%macro ALIGN_DESTINATION_POINTER 3
     mov ecx, %1                               ; Copy dest address low bits to scratch
     neg ecx                                   ; -dest: negation folds the wraparound into one mask
     and ecx, %3 - 1                           ; -dest &, align-1 = head bytes, 0 when aligned
     jz %%_aligned                             ; Already aligned, skip head copy
     sub %2, ecx                               ; Deduct head from remaining count
     rep movsb                                 ; Copy head bytes, rdi/rsi advanced; DF=0 per ABI
%%_aligned:
%endmacro

%macro PQ_BIAS_ADD 0
%%loop:
     vmovups ymm0, [rdx]                       ; load output floats 0..7
     vmovups ymm1, [rdx + PQ_BIAS_STRIDE1]     ; load output floats 8..15
     vmovups ymm2, [rdx + PQ_BIAS_STRIDE2]     ; load output floats 16..23
     vmovups ymm3, [rdx + PQ_BIAS_STRIDE3]     ; load output floats 24..31
     vaddps ymm0, ymm0, [rax]                  ; add bias floats 0..7
     vaddps ymm1, ymm1, [rax + PQ_BIAS_STRIDE1] ; add bias floats 8..15
     vaddps ymm2, ymm2, [rax + PQ_BIAS_STRIDE2] ; add bias floats 16..23
     vaddps ymm3, ymm3, [rax + PQ_BIAS_STRIDE3] ; add bias floats 24..31
     vmovups [rdx], ymm0                       ; store
     vmovups [rdx + PQ_BIAS_STRIDE1], ymm1     ; store
     vmovups [rdx + PQ_BIAS_STRIDE2], ymm2     ; store
     vmovups [rdx + PQ_BIAS_STRIDE3], ymm3     ; store
     sub rax, -PQ_BIAS_BLOCK                   ; advance bias pointer, imm8 encoding of the same +128
     sub rdx, -PQ_BIAS_BLOCK                   ; advance output pointer, imm8 encoding of the same +128
     sub ecx, PQ_BIAS_ELEMS                    ; decrement count by 32
     jg %%loop                                 ; loop while remaining > 0
%endmacro

%macro PQ_SET_FT_X 0
     mov pq_ft_x, rax                          ; ft.x = rax, input activations
     lea rdx, pq_ft_x_raw                      ; rdx = &ft, FloatTensor*, .s stays NULL
%endmacro

%macro PQ_QK_ROPE 2
     mov edx, %1                               ; rdx = head count, arg2
     mov r8d, pq_head_dim                      ; r8 = head_dim, arg3
     cmp dword pq_has_qk_norm, 0               ; has_qk_norm == 0?, Llama/SmolLM2
     je %%rope_only                            ; yes -> RoPE only, no per-head RMSNorm
     mov rax, [r13 + %2]                       ; rax = w->norm weights base
     add rax, pq_layer_hd_off                  ; rax = norm_weights[layer], byte offset
     mov r9, rax                               ; r9 = norm_weights[layer], arg4
     mov r10, [fused_rope_cos]                 ; r10 = cosine lookup table pointer, arg5
     mov r11, [fused_rope_sin]                 ; r11 = sine lookup table pointer, arg6
     API_CALL v7s_rmsnorm_rope_heads, rcx, rdx, r8, r9, r10, r11, pq_pos_slot, pq_rope_max_slot ; Apply RMSNorm+RoPE in-place
     jmp %%done                                ; full norm+RoPE done
%%rope_only:
     mov r9, [fused_rope_cos]                  ; r9 = cos_table, arg4, no norm_weight
     mov r10, [fused_rope_sin]                 ; r10 = sin_table, arg5
     API_CALL v7s_rope_only_heads, rcx, rdx, r8, r9, r10, pq_pos_slot, pq_rope_max_slot ; RoPE-only, no RMSNorm
%%done:
%endmacro
%endif                                         ; V7_MACROS_INC
