; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .text
%define RH_FLOAT4_SCALE 4                      ; sizeof,float for SIB scale in addressing
     SF_START v7s_rmsnorm_rope_head, r12, r13  ; r12 = head output base, r13 = head_dim, both non-volatile homes for the kernel body
     SF_XMM rh_xmm14                           ; save xmm14 low 128 bits, sole callee-saved XMM touched by RMSNORM_ROPE_BODY; YMM upper bits volatile per Win64 ABI, full-width restore would re-dirty after vzeroupper
     SF_COMMIT                                 ; finalize stack frame layout
     vmovdqu rh_xmm14, xmm14                   ; save xmm14 LOW 128 bits for RMSNorm scale
; Load head_dim from global into non-volatile register
     mov r13d, [rmsnorm_head_dim]              ; r13d = head dimension, avoids stack spill, reused by head-pointer imul
; RoPE table pointers are pre-offset by dispatcher, v7s_rmsnorm_rope_heads
; No per-head offset computation needed
; Compute head pointer: buf + head_idx * head_dim * sizeof,float
     imul rdx, r13                             ; rdx = head_idx * head_dim, 2-operand form, §45: 64-bit offset arithmetic; product <= 8192
     lea r12, [rcx + rdx*RH_FLOAT4_SCALE]      ; r12 = &buf[head_idx * head_dim], SIB scale = sizeof,float
; === Per-head RMSNorm + RoPE via shared macro ===
     RMSNORM_ROPE_BODY dword [rmsnorm_head_dim_float], [rmsnorm_norm_weight], [rmsnorm_cos_table], [rmsnorm_sin_table], dword [rmsnorm_pos], .rh_done ; inline RMSNorm+RoPE for this head, all globals read directly from BSS, zero stack-slot round-trips
.rh_done:
     CLEANUP_YMM                               ; vzeroupper BEFORE restoring callee-saved XMMs
     vmovdqu xmm14, rh_xmm14                   ; restore xmm14 LOW 128 bits; uppers stay zeroed, vzeroupper not undone
     SF_END r12, r13                           ; restore non-volatiles and return
; ---- %undef cleanup: keep the preprocessor namespace clean for later includes ----
%undef rh_xmm14
%undef RH_FLOAT4_SCALE
