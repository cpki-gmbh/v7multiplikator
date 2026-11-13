; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; v7_kernels.inc, kernel-block aggregator. The include order below is
; LOAD-BEARING: v7s_q8_chunk, q8_kernel.inc, included FIRST and the
; v7_kernels_end label, after the LAST include bracket the SMC VirtualProtect
; region that v7_init.inc step 6 makes RWX. Every PREFETCH_SLOT site and the
; PATCH_Q8_N scale-displacement writes must stay between the two labels.
; Rules:
; 1. q8_kernel.inc must remain the first include, v7s_q8_chunk is the
; region START anchor; a patched kernel placed before it falls outside
; the protected pages, v7_init.inc covers only the partial first page
; below the label and the first SMC write access-faults at runtime.
; 2. Any kernel file containing PREFETCH_SLOT sites must be included ONLY
; inside this block, all 41 slots live here today .
; 3. The end label must stay after the last include, and the last include
; must end its emission in section .text, the label lands in whatever
; section is current where it is emitted, see rmsnorm_quantize_kernel.inc,
; which self-declares section .text .
; ============================================================================
     section .text
%include "q8_kernel.inc"
%include "f32_kernel.inc"
%include "attention_kernel.inc"
%include "fused_f32_kernel.inc"
%include "swiglu_kernel.inc"
%include "swiglu_q8_kernel.inc"
%include "add_rmsnorm_kernel.inc"
%include "rmsnorm_kernel.inc"
%include "add_kernel.inc"
%include "memcpy_kernel.inc"
%include "quantize_kernel.inc"
%include "rmsnorm_rope_kernel.inc"
%include "rope_only_kernel.inc"
%include "vit_kernel.inc"
%include "rmsnorm_quantize_kernel.inc"
     global v7_kernels_end                     ; end-of-kernel-BLOCK label: pairs with v7s_q8_chunk, region start, q8_kernel.inc as the SMC VirtualProtect bracket; consumed by v7_init.inc step 6 only
v7_kernels_end:
