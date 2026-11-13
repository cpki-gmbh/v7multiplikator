; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define VG_CHAIN_BYTES 32                      ; one 8-float chain in bytes
%define VG_CHAIN2_BYTES 64                     ; two chains in bytes, chain C offset
%define VG_CHAIN3_BYTES 96                     ; three chains in bytes, chain D offset
%define VG_VEC_ALIGN_MASK -32                  ; mask for floor,n, 32 alignment
%define VG_FOUR_CHAIN_BYTES 128                ; all four chains per iteration
%define VG_FOUR_CHAIN_BYTES_NEG -VG_FOUR_CHAIN_BYTES ; negated stride for the [reg - N] advance form
%define VG_FOUR_CHAIN_FLOATS 32                ; floats consumed per .vg_loop4 iteration, NOT bytes, VG_FOUR_CHAIN_BYTES is the byte size
%define VG_TAIL_MASK 24                        ; count % 32 for the tail; also the "nothing left" test
vit_gelu:
     SF_START vit_gelu, rbx, r12, r14          ; prologue: save non-volatiles, r14d = total count
     SF_XMM vg_x6                              ; spill slot: xmm6 low half, holds constant
     SF_XMM vg_x7                              ; spill slot: xmm7 low half, chain A x
     SF_XMM vg_x8                              ; spill slot: xmm8 low half, holds constant
     SF_XMM vg_x9                              ; spill slot: xmm9 low half, chain B x
     SF_XMM vg_x10                             ; spill slot: xmm10 low half, holds constant
     SF_XMM vg_x11                             ; spill slot: xmm11 low half, chain C x
     SF_XMM vg_x13                             ; spill slot: xmm13 low half, chain D x
     SF_XMM vg_x12                             ; spill slot: xmm12 low half, holds c6 broadcast
     SF_XMM vg_x14                             ; spill slot: xmm14 low half, ln2_hi vector
     SF_XMM vg_x15                             ; spill slot: xmm15 low half, ln2_lo vector
     SF_COMMIT                                 ; finalize stack frame layout
     vmovdqu vg_x6, xmm6                       ; save callee-saved xmm6 low, holds constant
     vmovdqu vg_x7, xmm7                       ; save callee-saved xmm7 low, chain A x
     vmovdqu vg_x8, xmm8                       ; save callee-saved xmm8 low, holds constant
     vmovdqu vg_x9, xmm9                       ; save callee-saved xmm9 low, chain B x
     vmovdqu vg_x10, xmm10                     ; save callee-saved xmm10 low, holds constant
     vmovdqu vg_x11, xmm11                     ; save callee-saved xmm11 low, chain C x
     vmovdqu vg_x13, xmm13                     ; save callee-saved xmm13 low, chain D x
     vmovdqu vg_x12, xmm12                     ; save callee-saved xmm12 low, holds c6 broadcast
     vmovdqu vg_x14, xmm14                     ; save callee-saved xmm14 low, ln2_hi vector
     vmovdqu vg_x15, xmm15                     ; save callee-saved xmm15 low, ln2_lo vector
     mov r12, rcx                              ; r12 = running pointer, register for addressing
     mov r14d, edx                             ; r14d = total float count, kept for the tail
     mov ebx, r14d                             ; ebx = total count
     and ebx, VG_VEC_ALIGN_MASK                ; ebx = vec_count = floor,n/32 *32, 4 chains/iter ; n<32 falls to the tail's 0/1/2/3-chunk path
     jz .vg_tail                               ; < 32 floats, tail only
     vbroadcastss ymm12, [rel vit_gelu_k]      ; k = 0.044715, cubic coefficient for the four-vector body
     vbroadcastss ymm13, [rel vit_gelu_c2]     ; c2 = 2*sqrt,2/pi, argument scale for the four-vector body
     vbroadcastss ymm14, [rel s_c6_v]          ; c6, Horner seed for the four-vector body
     vbroadcastss ymm15, [rel vit_gelu_one]    ; 1.0, unity for the polynomial and denominator

%macro GELU_CHAIN 2
     vmovups %2, [%1]                          ; x-reg = x[0..7] of this chain
     vmulps ymm0, %2, %2                       ; ymm0 = x^2
     vmulps ymm0, ymm0, %2                     ; ymm0 = x^3
     vfmadd213ps ymm0, ymm6, %2                ; ymm0 = 0.044715*x^3 + x
     vmulps ymm0, ymm0, ymm10                  ; ymm0 = 2*sqrt,2/pi *,... = 2u
     EXP_INLINE                                ; ymm0 = exp,2u, the inlined clamped exp chain, writes ymm0/2/3/4
     vaddps ymm5, ymm0, ymm8                   ; ymm5 = e + 1
     vdivps ymm0, ymm0, ymm5                   ; ymm0 = e/,e+1 = GELU factor, x2/x0.5 cancel
     vmulps ymm0, ymm0, %2                     ; ymm0 = x * e/,e+1 = GELU,x
     vmovups [%1], ymm0                        ; store this chain
%endmacro

; GELU_PAIR_PRE dstA, dstB, inA, inB, two independent chains interleaved with
; the x loads omitted: the caller issues all four chain loads at loop entry so
; the serial exp/divide chains start with their operands already resident and
; the two chains' independent work overlaps

%macro GELU_PAIR_PRE 4                         ; dstA, dstB, inputA, inputB
     vmulps ymm0, %3, %3                       ; chain A: x^2
     vmulps ymm1, %4, %4                       ; chain B: x^2
     vmulps ymm0, ymm0, %3                     ; chain A: x^3
     vmulps ymm1, ymm1, %4                     ; chain B: x^3
     vfmadd213ps ymm0, ymm6, %3                ; chain A: 0.044715*x^3 + x
     vfmadd213ps ymm1, ymm6, %4                ; chain B: same
     vmulps ymm0, ymm0, ymm10                  ; chain A: 2u
     vmulps ymm1, ymm1, ymm10                  ; chain B: 2u
     vminps ymm0, ymm0, [rel s_expclamp_v]     ; chain A: clamp upper
     vminps ymm1, ymm1, [rel s_expclamp_v]     ; chain B: clamp upper
     vmaxps ymm0, ymm0, [rel s_negclamp_v]     ; chain A: clamp lower
     vmaxps ymm1, ymm1, [rel s_negclamp_v]     ; chain B: clamp lower
     vmulps ymm2, ymm0, [rel s_log2e_v]        ; chain A: x/ln2
     vmulps ymm3, ymm1, [rel s_log2e_v]        ; chain B: x/ln2
     vroundps ymm2, ymm2, 0                    ; chain A: n float
     vroundps ymm3, ymm3, 0                    ; chain B: n float
     vmovaps ymm4, ymm12                       ; chain A: c6 seed, issued in the reduction latency window
     vmovaps ymm5, ymm12                       ; chain B: c6 seed
     vfnmadd231ps ymm0, ymm2, ymm14            ; chain A: x - n*ln2_hi, in place, no seed copy
     vfnmadd231ps ymm1, ymm3, ymm14            ; chain B: same
     vfnmadd231ps ymm0, ymm2, ymm15            ; chain A: f, single rounding
     vfnmadd231ps ymm1, ymm3, ymm15            ; chain B: f, single rounding
     vcvttps2dq ymm2, ymm2                     ; chain A: n as int32
     vcvttps2dq ymm3, ymm3                     ; chain B: n as int32
     vpaddd ymm2, ymm2, [rel swk_bias_127]     ; chain A: exponent bias
     vpaddd ymm3, ymm3, [rel swk_bias_127]     ; chain B: exponent bias
     vfmadd213ps ymm4, ymm0, [rel s_c5_v]      ; chain A: start Horner
     vfmadd213ps ymm5, ymm1, [rel s_c5_v]      ; chain B: start Horner
     vpslld ymm2, ymm2, 23                     ; chain A: exponent field
     vpslld ymm3, ymm3, 23                     ; chain B: exponent field
     vfmadd213ps ymm4, ymm0, [rel s_c4_v]      ; chain A
     vfmadd213ps ymm5, ymm1, [rel s_c4_v]      ; chain B
     vfmadd213ps ymm4, ymm0, [rel s_c3_v]      ; chain A
     vfmadd213ps ymm5, ymm1, [rel s_c3_v]      ; chain B
     vfmadd213ps ymm4, ymm0, [rel s_half_v]    ; chain A
     vfmadd213ps ymm5, ymm1, [rel s_half_v]    ; chain B
     vfmadd213ps ymm4, ymm0, ymm8              ; chain A
     vfmadd213ps ymm5, ymm1, ymm8              ; chain B
     vfmadd213ps ymm4, ymm0, ymm8              ; chain A
     vfmadd213ps ymm5, ymm1, ymm8              ; chain B
     vmulps ymm4, ymm4, ymm2                   ; chain A: exp factor
     vmulps ymm5, ymm5, ymm3                   ; chain B: exp factor
     vaddps ymm0, ymm4, ymm8                   ; chain A: e + 1 (ymm0's residual died in the Horner above)
     vaddps ymm1, ymm5, ymm8                   ; chain B: e + 1
     vdivps ymm4, ymm4, ymm0                   ; chain A: e/(e+1)
     vdivps ymm5, ymm5, ymm1                   ; chain B: e/(e+1)
     vmulps ymm4, ymm4, %3                     ; chain A: x * factor
     vmulps ymm5, ymm5, %4                     ; chain B: x * factor
     vmovups [%1], ymm4                        ; store chain A
     vmovups [%2], ymm5                        ; store chain B
%endmacro

%macro EXP_INLINE 0
     vminps ymm0, ymm0, [rel s_expclamp_v]     ; clamp upper to avoid overflow
     vmaxps ymm0, ymm0, [rel s_negclamp_v]     ; clamp lower to avoid underflow
     vmulps ymm3, ymm0, [rel s_log2e_v]        ; x / ln,2
     vroundps ymm3, ymm3, 0                    ; n = round,x/ln2
     vcvttps2dq ymm2, ymm3                     ; n as int32
     vpaddd ymm2, ymm2, [rel swk_bias_127]     ; exponent bias
     vfnmadd231ps ymm0, ymm3, ymm14            ; x - n*ln2_hi
     vpslld ymm2, ymm2, 23                     ; exponent field
     vfnmadd231ps ymm0, ymm3, ymm15            ; f, single rounding
     vmovaps ymm4, ymm12                       ; c6 seed
     vfmadd213ps ymm4, ymm0, [rel s_c5_v]      ; f*c6 + c5
     vfmadd213ps ymm4, ymm0, [rel s_c4_v]      ; f*prev + c4
     vfmadd213ps ymm4, ymm0, [rel s_c3_v]      ; f*prev + c3
     vfmadd213ps ymm4, ymm0, [rel s_half_v]    ; f*prev + 0.5, c2
     vfmadd213ps ymm4, ymm0, ymm8              ; c1
     vfmadd213ps ymm4, ymm0, ymm8              ; c0
     vmulps ymm0, ymm4, ymm2                   ; exp,x = poly * 2^n, final result still in ymm0
%endmacro
.vg_loop4:
     vmovups ymm8, [r12]                       ; four disjoint input vectors, A: x[0..7]
     vmovups ymm9, [r12 + VG_CHAIN_BYTES]      ; B: x[8..15]
     vmovups ymm10, [r12 + VG_CHAIN2_BYTES]    ; C: x[16..23]
     vmovups ymm11, [r12 + VG_CHAIN3_BYTES]    ; D: x[24..31]
     vmulps ymm0, ymm8, ymm8                   ; A: x^2
     vmulps ymm1, ymm9, ymm9                   ; B: x^2
     vmulps ymm2, ymm10, ymm10                 ; C: x^2
     vmulps ymm3, ymm11, ymm11                 ; D: x^2
     vmulps ymm0, ymm0, ymm8                   ; A: x^3
     vmulps ymm1, ymm1, ymm9                   ; B: x^3
     vmulps ymm2, ymm2, ymm10                  ; C: x^3
     vmulps ymm3, ymm3, ymm11                  ; D: x^3
     vfmadd213ps ymm0, ymm12, ymm8             ; A: k*x^3 + x, one rounding
     vfmadd213ps ymm1, ymm12, ymm9             ; B: k*x^3 + x
     vfmadd213ps ymm2, ymm12, ymm10            ; C: k*x^3 + x
     vfmadd213ps ymm3, ymm12, ymm11            ; D: k*x^3 + x
     vmulps ymm0, ymm0, ymm13                  ; A: * c2 = 2u
     vmulps ymm1, ymm1, ymm13                  ; B: * c2
     vmulps ymm2, ymm2, ymm13                  ; C: * c2
     vmulps ymm3, ymm3, ymm13                  ; D: * c2
     vmovups ymm8, [rel s_ln2hi_v]             ; ln2_hi after the x registers die
     vmovups ymm9, [rel s_ln2lo_v]             ; ln2_lo
     vminps ymm0, ymm0, [rel s_expclamp_v]     ; A: clamp upper
     vminps ymm1, ymm1, [rel s_expclamp_v]     ; B: clamp upper
     vminps ymm2, ymm2, [rel s_expclamp_v]     ; C: clamp upper
     vminps ymm3, ymm3, [rel s_expclamp_v]     ; D: clamp upper
     vmaxps ymm0, ymm0, [rel s_negclamp_v]     ; A: clamp lower
     vmaxps ymm1, ymm1, [rel s_negclamp_v]     ; B: clamp lower
     vmaxps ymm2, ymm2, [rel s_negclamp_v]     ; C: clamp lower
     vmaxps ymm3, ymm3, [rel s_negclamp_v]     ; D: clamp lower
     vmulps ymm4, ymm0, [rel s_log2e_v]        ; A: x/ln2
     vmulps ymm5, ymm1, [rel s_log2e_v]        ; B: x/ln2
     vmulps ymm6, ymm2, [rel s_log2e_v]        ; C: x/ln2
     vmulps ymm7, ymm3, [rel s_log2e_v]        ; D: x/ln2
     vroundps ymm4, ymm4, 0                    ; A: n float, round nearest
     vroundps ymm5, ymm5, 0                    ; B: n float
     vroundps ymm6, ymm6, 0                    ; C: n float
     vroundps ymm7, ymm7, 0                    ; D: n float
     vfnmadd231ps ymm0, ymm4, ymm8             ; A: f -= n*ln2_hi
     vfnmadd231ps ymm1, ymm5, ymm8             ; B: f -= n*ln2_hi
     vfnmadd231ps ymm2, ymm6, ymm8             ; C: f -= n*ln2_hi
     vfnmadd231ps ymm3, ymm7, ymm8             ; D: f -= n*ln2_hi
     vfnmadd231ps ymm0, ymm4, ymm9             ; A: f -= n*ln2_lo, single rounding
     vfnmadd231ps ymm1, ymm5, ymm9             ; B: f -= n*ln2_lo
     vfnmadd231ps ymm2, ymm6, ymm9             ; C: f -= n*ln2_lo
     vfnmadd231ps ymm3, ymm7, ymm9             ; D: f -= n*ln2_lo
     vcvttps2dq ymm4, ymm4                     ; A: n as int32
     vcvttps2dq ymm5, ymm5                     ; B: n as int32
     vcvttps2dq ymm6, ymm6                     ; C: n as int32
     vcvttps2dq ymm7, ymm7                     ; D: n as int32
     vpaddd ymm4, ymm4, [rel swk_bias_127]     ; A: exponent bias
     vpaddd ymm5, ymm5, [rel swk_bias_127]     ; B: exponent bias
     vpaddd ymm6, ymm6, [rel swk_bias_127]     ; C: exponent bias
     vpaddd ymm7, ymm7, [rel swk_bias_127]     ; D: exponent bias
     vpslld ymm4, ymm4, 23                     ; A: 2^n bit pattern
     vpslld ymm5, ymm5, 23                     ; B: 2^n bit pattern
     vpslld ymm6, ymm6, 23                     ; C: 2^n bit pattern
     vpslld ymm7, ymm7, 23                     ; D: 2^n bit pattern
     vmovaps ymm8, ymm14                       ; A: c6 Horner seed
     vmovaps ymm9, ymm14                       ; B: c6 Horner seed
     vmovaps ymm10, ymm14                      ; C: c6 Horner seed
     vmovaps ymm11, ymm14                      ; D: c6 Horner seed
     vfmadd213ps ymm8, ymm0, [rel s_c5_v]      ; A: c5
     vfmadd213ps ymm9, ymm1, [rel s_c5_v]      ; B: c5
     vfmadd213ps ymm10, ymm2, [rel s_c5_v]     ; C: c5
     vfmadd213ps ymm11, ymm3, [rel s_c5_v]     ; D: c5
     vfmadd213ps ymm8, ymm0, [rel s_c4_v]      ; A: c4
     vfmadd213ps ymm9, ymm1, [rel s_c4_v]      ; B: c4
     vfmadd213ps ymm10, ymm2, [rel s_c4_v]     ; C: c4
     vfmadd213ps ymm11, ymm3, [rel s_c4_v]     ; D: c4
     vfmadd213ps ymm8, ymm0, [rel s_c3_v]      ; A: c3
     vfmadd213ps ymm9, ymm1, [rel s_c3_v]      ; B: c3
     vfmadd213ps ymm10, ymm2, [rel s_c3_v]     ; C: c3
     vfmadd213ps ymm11, ymm3, [rel s_c3_v]     ; D: c3
     vfmadd213ps ymm8, ymm0, [rel s_half_v]    ; A: c2 = 0.5
     vfmadd213ps ymm9, ymm1, [rel s_half_v]    ; B: c2
     vfmadd213ps ymm10, ymm2, [rel s_half_v]   ; C: c2
     vfmadd213ps ymm11, ymm3, [rel s_half_v]   ; D: c2
     vfmadd213ps ymm8, ymm0, ymm15             ; A: c1
     vfmadd213ps ymm9, ymm1, ymm15             ; B: c1
     vfmadd213ps ymm10, ymm2, ymm15            ; C: c1
     vfmadd213ps ymm11, ymm3, ymm15            ; D: c1
     vfmadd213ps ymm8, ymm0, ymm15             ; A: c0
     vfmadd213ps ymm9, ymm1, ymm15             ; B: c0
     vfmadd213ps ymm10, ymm2, ymm15            ; C: c0
     vfmadd213ps ymm11, ymm3, ymm15            ; D: c0
     vmulps ymm8, ymm8, ymm4                   ; A: exp = poly * 2^n
     vmulps ymm9, ymm9, ymm5                   ; B: exp
     vmulps ymm10, ymm10, ymm6                 ; C: exp
     vmulps ymm11, ymm11, ymm7                 ; D: exp
     vaddps ymm0, ymm8, ymm15                  ; A: e + 1
     vaddps ymm1, ymm9, ymm15                  ; B: e + 1
     vaddps ymm2, ymm10, ymm15                 ; C: e + 1
     vaddps ymm3, ymm11, ymm15                 ; D: e + 1
     vdivps ymm8, ymm8, ymm0                   ; A: e/(e+1)
     vdivps ymm9, ymm9, ymm1                   ; B: e/(e+1)
     vdivps ymm10, ymm10, ymm2                 ; C: e/(e+1)
     vdivps ymm11, ymm11, ymm3                 ; D: e/(e+1)
     vmulps ymm8, ymm8, [r12]                  ; A: * original x, cache-line reread before any store
     vmulps ymm9, ymm9, [r12 + VG_CHAIN_BYTES] ; B: * original x
     vmulps ymm10, ymm10, [r12 + VG_CHAIN2_BYTES] ; C: * original x
     vmulps ymm11, ymm11, [r12 + VG_CHAIN3_BYTES] ; D: * original x
     vmovups [r12], ymm8                       ; store A
     vmovups [r12 + VG_CHAIN_BYTES], ymm9      ; store B
     vmovups [r12 + VG_CHAIN2_BYTES], ymm10    ; store C
     vmovups [r12 + VG_CHAIN3_BYTES], ymm11    ; store D
     sub r12, VG_FOUR_CHAIN_BYTES_NEG          ; advance past all four chains, 128 bytes FORWARD, sub -128 is the 4-byte sign-extended imm8 form, against 7 for 'add r12, 128'
     sub ebx, VG_FOUR_CHAIN_FLOATS             ; vec_count -= 32 floats
     jg .vg_loop4                              ; next quad
.vg_tail:
     mov ebx, r14d                             ; ebx = total count
     and ebx, VG_TAIL_MASK                     ; remainder = count % 32, provably 0/8/16/24
     jz .vg_done                               ; nothing left
     vbroadcastss ymm6, [rel vit_gelu_k]       ; ymm6 = 0.044715, reused every tail iteration
     vbroadcastss ymm8, [rel vit_gelu_one]     ; ymm8 = 1.0, denominator constant: e+1 for GELU factor e/,e+1
     vbroadcastss ymm10, [rel vit_gelu_c2]     ; ymm10 = 2*sqrt,2/pi, pre-doubled constant
     vbroadcastss ymm12, [rel s_c6_v]          ; ymm12 = c6, polynomial seed, reused by every chain's exp
     vmovups ymm14, [rel s_ln2hi_v]            ; ymm14 = ln2_hi vector, invariant for the tail
     vmovups ymm15, [rel s_ln2lo_v]            ; ymm15 = ln2_lo vector, invariant for the tail
     cmp ebx, VG_CHAIN2_BYTES / 4              ; at least two 8-float chunks, 16 floats?
     jb .vg_tail_one                           ; no, single chain
     vmovups ymm7, [r12]                       ; chain A x
     vmovups ymm9, [r12 + VG_CHAIN_BYTES]      ; chain B x
     GELU_PAIR_PRE r12, {r12 + VG_CHAIN_BYTES}, ymm7, ymm9 ; chains A+B, paired
     add r12, VG_CHAIN2_BYTES                  ; advance two chunks
     sub ebx, VG_CHAIN2_BYTES / 4              ; 0 = done, 8 = one chunk remains
     jz .vg_done                               ; the pair covered the whole tail
.vg_tail_one:
     GELU_CHAIN r12, ymm7                      ; one ragged chunk: x[0..7] at the tail pointer
.vg_done:
     vzeroupper                                ; clean AVX state before XMM restore, upper halves were dirtied mid-function
     vmovdqu xmm6, vg_x6                       ; restore callee-saved xmm6 low
     vmovdqu xmm7, vg_x7                       ; restore callee-saved xmm7 low
     vmovdqu xmm8, vg_x8                       ; restore callee-saved xmm8 low
     vmovdqu xmm9, vg_x9                       ; restore callee-saved xmm9 low
     vmovdqu xmm10, vg_x10                     ; restore callee-saved xmm10 low
     vmovdqu xmm11, vg_x11                     ; restore callee-saved xmm11 low
     vmovdqu xmm13, vg_x13                     ; restore callee-saved xmm13 low
     vmovdqu xmm12, vg_x12                     ; restore callee-saved xmm12 low
     vmovdqu xmm14, vg_x14                     ; restore callee-saved xmm14 low
     vmovdqu xmm15, vg_x15                     ; restore callee-saved xmm15 low
     SF_END rbx, r12, r14                      ; restore non-volatiles, return
%unmacro GELU_CHAIN 2                          ; arity MUST match the %macro declaration, two operands
%unmacro GELU_PAIR_PRE 4                       ; arity MUST match the declaration, four operands
%unmacro EXP_INLINE 0                          ; the inlined exp body, expansion sites are inside GELU_CHAIN
%undef VG_FOUR_CHAIN_BYTES_NEG
%undef VG_FOUR_CHAIN_BYTES
%undef VG_FOUR_CHAIN_FLOATS
%undef VG_TAIL_MASK
%undef VG_VEC_ALIGN_MASK
%undef VG_CHAIN3_BYTES
%undef VG_CHAIN2_BYTES
%undef VG_CHAIN_BYTES
