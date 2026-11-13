; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .rdata align=32
; --- RMSNorm/RoPE constants ---
     align 4
rpe_epsilon:
     dd 1e-6                                   ; 1e-6 contract floor for v7_set_sampler_temperature's sub-epsilon/NaN clamp, public_api.inc, RMSNorm's epsilon is per-model from the DB, __sys_rms_eps -> g_rms_eps, never this constant
rpe_one:
     dd 1.0                                    ; 1.0f, the universal one, RMS scale numerator, sigmoid addend, exp-polynomial c1/c0, attention scale recurrence, sampler bounds
rpe_half:
     dd 0.5                                    ; 0.5f, RoPE exponent denominator, theta^,-2i/d and sigmoid input scale
; --- EVAL_SIGMOID scalar constants, loaded via vmovss/vbroadcastss ---
     align 4
scalar_m80:
     dd -80.0                                  ; -80.0 clamp threshold for exp, safe for softmax under FTZ
scalar_80:
     dd 80.0                                   ; +80.0 upper clamp for exp, safe for softmax under FTZ
scalar_log2e:
     dd 1.4426950408889634                     ; log2,e for 2^x decomposition
scalar_neg_flt_max:
     dd 0xFF7FFFFF                             ; -FLT_MAX for softmax m0 initialization / max tracker
     align 4
attn_c5:
     dd 0x3AADD303                             ; degree-5 centered fit on [-0.5,+0.5], C5, P(0)=1
attn_c4:
     dd 0x3C1E7411                             ; degree-5 centered fit on [-0.5,+0.5], C4
attn_c3:
     dd 0x3D635B9B                             ; degree-5 centered fit on [-0.5,+0.5], C3
attn_c2:
     dd 0x3E75FCE0                             ; degree-5 centered fit on [-0.5,+0.5], C2
attn_c1:
     dd 0x3F317215                             ; degree-5 centered fit on [-0.5,+0.5], C1
; --- EVAL_SIGMOID packed all-lane coefficient vectors: bit-duplicates of the scalar dd literals above,
; required because a packed FMA folded memory operand feeds lane i from mem[i] ---
     align 32
attn_c4_vec:
     times 8 dd 0x3C1E7411                     ; C4 vector, == attn_c4
attn_c3_vec:
     times 8 dd 0x3D635B9B                     ; C3 vector, == attn_c3
attn_c2_vec:
     times 8 dd 0x3E75FCE0                     ; C2 vector, == attn_c2
attn_c1_vec:
     times 8 dd 0x3F317215                     ; C1 vector, == attn_c1
; --- EVAL_SIGMOID packed YMM constants, 8xf32 ---
     align 32
     ymm_one: times 8 dd 1.0                   ; YMM broadcast of 1.0
     ymm_m80: times 8 dd -80.0                 ; YMM broadcast of the -80.0 clamp threshold, exp underflow guard
     ymm_80: times 8 dd 80.0                   ; YMM broadcast of the +80.0 clamp threshold, exp overflow guard
; --- Packed exponent bias ---
     align 32
     swk_bias_127: times 8 dd 127              ; float exponent bias for 2^ipart
; --- Public API border constants, public_api.inc setters ---
     align 4
pub_float_absmask:
     dd 0x7FFFFFFF                             ; float abs, mask, AND clears the sign bit
pub_pen_limit:
     dd 2.0                                    ; presence-penalty magnitude limit, [-2, 2] contract
pub_float_inf:
     dd 0x7F800000                             ; +infinity, the temperature setter refuses it
