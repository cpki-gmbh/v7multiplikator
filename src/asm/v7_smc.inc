; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%ifdef V7_LINUX
     section .data.rel.ro                      ; ELF loader relocates kernel pointers before RELRO
%else
     section .rdata                            ; kernel_bases is read-only data, not code, win64 default section is .text
%endif
     align 8
; kernel base address table for SMC patching, internal only, no DLL export,
; never resolved via GetProcAddress; every consumer is same-module RIP-relative
kernel_bases:
     dq v7s_q8_chunk                           ; kernel_id 0: Q8 kernel base address
     dq v7s_f32_chunk                          ; kernel_id 1: F32 kernel base address
     dq v7s_fused_f32_chunk                    ; kernel_id 2: Fused F32 kernel base address
     dq swk_kernel                             ; kernel_id 3: SwiGLU F32 base, MUST be the same label the swiglu-f32 PREFETCH_ROWs in v7.asm (slots 13-26) use for .pfN offset computation; v7s_swiglu_chunk, defined at the top of swiglu_kernel.inc, is a same-address export alias (SF_START emits the swk_kernel label immediately after it), but this entry must track the ROW label, not the alias, or every kid-3 patch write mis-targets if the two ever diverge
     dq v7s_swiglu_q8_chunk                    ; kernel_id 4: SwiGLU Q8 kernel base address
     dq v7s_attention_inner                    ; kernel_id 5: Attention kernel base address
     section .text                             ; close the .rdata this file opened
