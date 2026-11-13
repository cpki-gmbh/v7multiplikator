; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%include "vit_structs.inc"
%include "inf_structs.inc"
%include "vit_layernorm.inc"
%include "vit_gelu.inc"
%include "vit_gelu_fork.inc"                   ; two-way in-place GELU for vision layer MLPs
%include "vit_add.inc"
%include "vit_pos_embed.inc"
%include "vit_rope_build.inc"
%include "vit_rope_apply.inc"
%include "vit_pixel_shuffle.inc"
     section .rdata align=32
vit_gelu_c2:
     dd 1.5957691216057308                     ; GELU tanh-approx c2, =sqrt,2/pi
vit_gelu_k:
     dd 0.044715                               ; cubic coefficient for GELU
vit_gelu_one:
     dd 1.0                                    ; 1.0 for GELU / LayerNorm
     section .text
; ============================================================================
; Call-frame constants for internal function calls within v7s_vit_forward
; ============================================================================
%define VIT_CALL7_FRAME 64                     ; 32 shadow + 24 stack args + 8 pad = 64, 16-aligned
%define VIT_CALL8_FRAME 64                     ; 8-arg call: 32 shadow + 4 stack args * 8 = 64, 16-aligned
%define VIT_CALL6_FRAME 48                     ; 32 shadow + 16 stack args = 48, 16-aligned
%define VIT_CALL4_FRAME 32                     ; 32 shadow only = 32, 16-aligned
%define VIT_CALL_ARG5 32                       ; byte offset of 5th arg from rsp in call frame
%define VIT_CALL_ARG6 40                       ; byte offset of 6th arg from rsp
%define VIT_CALL_ARG7 48                       ; byte offset of 7th arg from rsp
%define VIT_CALL_ARG8 56                       ; byte offset of 8th arg from rsp
%define VIT_YMM_BYTES 32                       ; bytes per AVX2 YMM vector, 8 floats
%define VIT_RGB_VEC_BYTES 24                   ; bytes consumed per vector normalize iteration, 8 px x 3 bytes
%define VIT_RGB_LO_HALF 16                     ; byte offset of the hi-8-bytes load within a 24-byte RGB block
; ============================================================================
; .rdata constants for v7s_vit_encode_callback
; ============================================================================
     section .rdata align=32
sz_vit_w_key:
     db "__sys_vit_weights", 0                 ; microdb key for ViT weights blob
sz_vit_proj_key:
     db "__sys_vit_projector", 0               ; microdb key for projector weights, SigLIP
vit_inv_1275:
     dd 0x3C008080                             ; tuned 1/127.5 coefficient: one ULP below the nearest binary32 to 2/255, chosen for the fma(pixel, c, -1.0f) byte->[-1,1] mapping
vit_one_float:
     dd 1.0                                    ; 1.0 subtracted after normalization
ar_mask_r_lo:
     db 0, 3, 6, 9, 12, 15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 ; R taps from lo bytes, pixels 0..5 -> lanes 0..5
ar_mask_r_hi:
     db 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 2, 5, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 ; R taps from hi bytes, stream 18,21 = pixels 6..7 -> lanes 6..7
ar_mask_g_lo:
     db 1, 4, 7, 10, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 ; G taps from lo bytes, pixels 0..4 -> lanes 0..4
ar_mask_g_hi:
     db 0x80, 0x80, 0x80, 0x80, 0x80, 0, 3, 6, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 ; G taps from hi bytes, stream 16,19,22 = pixels 5..7 -> lanes 5..7
ar_mask_b_lo:
     db 2, 5, 8, 11, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 ; B taps from lo bytes, pixels 0..4 -> lanes 0..4
ar_mask_b_hi:
     db 0x80, 0x80, 0x80, 0x80, 0x80, 1, 4, 7, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80 ; B taps from hi bytes, stream 17,20,23 = pixels 5..7 -> lanes 5..7
     section .text
     global v7s_vit_forward                    ; export v7s_vit_forward entry point
v7s_vit_forward:
     SF_START v7s_vit_forward, rbx, rsi, rdi, r12, r13, r14 ; prologue: save non-volatiles, reserve frame, rbx carries the layer index, it must be non-volatile, the loop body's 13 calls may overwrite esi; r15 is neither written nor read here
     SF_QWORD vvf_pixels                       ; const float* pixels, patch features input
     SF_DWORD vvf_gh                           ; grid_h
     SF_DWORD vvf_gw                           ; grid_w
     SF_DWORD vvf_nmerged                      ; n_merged = n / 4
     SF_QWORD vvf_w                            ; weight struct pointer, 27 ptrs = 216 bytes
     SF_QWORD vvf_out_main                     ; output main merger [n_merged * 2560]
     SF_QWORD vvf_out_ds0                      ; output deepstack 0
     SF_QWORD vvf_out_ds1                      ; output deepstack 1
     SF_QWORD vvf_out_ds2                      ; output deepstack 2
     SF_QWORD vvf_hidden                       ; working: hidden state [n * 1024]
     SF_QWORD vvf_qkv                          ; working: qkv buffer [n * 3072]
     SF_QWORD vvf_norm                         ; working: normalized buffer [n * 1024]
     SF_QWORD vvf_mlp                          ; working: mlp intermediate [n * 4096]
     SF_QWORD vvf_scores                       ; working: attention scores [heads * ALIGN64,n ]
     SF_QWORD vvf_attn                         ; working: attention output [n * 1024]
     SF_QWORD vvf_pos                          ; working: pos embeddings [n * 1024]
     SF_QWORD vvf_cos                          ; working: RoPE cos table [n * 64]
     SF_QWORD vvf_sin                          ; working: RoPE sin table [n * 64]
     SF_QWORD vvf_packed                       ; working: packed 2x2 patches [n_merged * 4096]
     SF_QWORD vvf_mmlp                         ; working: merger mlp intermediate [n_merged * 4096]
     SF_QWORD vvf_n1g                          ; running: norm1_gamma[L], advanced by VCFG_HIDDEN_F4 per layer
     SF_QWORD vvf_n1b                          ; running: norm1_beta[L]
     SF_QWORD vvf_qkvw                         ; running: qkv_w[L], advanced by VCFG_L_QKV_W per layer
     SF_QWORD vvf_qkvb                         ; running: qkv_b[L], advanced by VCFG_L_QKV_B per layer
     SF_QWORD vvf_pjw                          ; running: proj_w[L], advanced by VCFG_L_PROJ_W per layer
     SF_QWORD vvf_pjb                          ; running: proj_b[L], advanced by VCFG_L_PROJ_B per layer
     SF_QWORD vvf_n2g                          ; running: norm2_gamma[L], advanced by VCFG_HIDDEN_F4 per layer
     SF_QWORD vvf_n2b                          ; running: norm2_beta[L]
     SF_QWORD vvf_f1w                          ; running: fc1_w[L], advanced by VCFG_L_FC1_W per layer
     SF_QWORD vvf_f1b                          ; running: fc1_b[L], advanced by VCFG_L_FC1_B per layer
     SF_QWORD vvf_f2w                          ; running: fc2_w[L], advanced by VCFG_L_FC2_W per layer
     SF_QWORD vvf_f2b                          ; running: fc2_b[L], advanced by VCFG_L_FC2_B per layer
     SF_COMMIT                                 ; finalize stack frame layout
; Save input arguments
     mov vvf_pixels, rcx                       ; save pixels pointer
     mov vvf_gh, edx                           ; save grid_h
     mov vvf_gw, r8d                           ; save grid_w
     mov vvf_w, r9                             ; save weight struct pointer
     mov rax, [rbp + VK_ARG5]                  ; load out_main from stack arg
     mov vvf_out_main, rax                     ; save out_main
     mov rax, [rbp + VK_ARG6]                  ; load out_ds0
     mov vvf_out_ds0, rax                      ; save out_ds0
     mov rax, [rbp + VK_ARG7]                  ; load out_ds1
     mov vvf_out_ds1, rax                      ; save out_ds1
     mov rax, [rbp + VK_ARG8]                  ; load out_ds2
     mov vvf_out_ds2, rax                      ; save out_ds2
; Compute n = gh * gw, n_merged = n / merge*merge
; Data-driven: respects VCFG_MERGE, Qwen3-VL merge=2 -> /4, SigLIP merge=1 -> /1
     mov eax, edx                              ; original grid_h remains in EDX
     imul eax, r8d                             ; original grid_w remains in R8D
     mov r14d, eax                             ; cache n in r14d, non-volatile for entire forward
     xor edx, edx                              ; zero upper dividend bits for div
     div dword [rel g_vit_cfg + VCFG_MERGE_UNIT] ; n / merge^2, precomputed in DB, offset 184 = n_merged
     mov vvf_nmerged, eax                      ; save n_merged, data-driven, not hardcoded /4
     mov rax, [rel g_vit_ptr_hidden]           ; load hidden buffer pointer
     mov vvf_hidden, rax                       ; save to local frame
     mov rax, [rel g_vit_ptr_qkv]              ; load qkv buffer pointer
     mov vvf_qkv, rax                          ; save to local frame
     mov rax, [rel g_vit_ptr_norm]             ; load norm buffer pointer
     mov vvf_norm, rax                         ; save to local frame
     mov rax, [rel g_vit_ptr_mlp]              ; load mlp buffer pointer
     mov vvf_mlp, rax                          ; save to local frame
     mov rax, [rel g_vit_ptr_scores]           ; load scores scratch pointer
     mov vvf_scores, rax                       ; save to local frame
     mov rax, [rel g_vit_ptr_attn]             ; load attn buffer pointer
     mov vvf_attn, rax                         ; save to local frame
     mov rax, [rel g_vit_ptr_pos]              ; load pos buffer pointer
     mov vvf_pos, rax                          ; save to local frame
     mov rax, [rel g_vit_ptr_cos]              ; load cos table pointer
     mov vvf_cos, rax                          ; save to local frame
     mov rax, [rel g_vit_ptr_sin]              ; load sin table pointer
     mov vvf_sin, rax                          ; save to local frame
     TRACE_STR "vvf: scratch ptrs loaded"      ; trace: all scratch buffer pointers loaded OK
     mov rax, [rel g_vit_ptr_packed]           ; load packed buffer pointer
     mov vvf_packed, rax                       ; save to local frame
     mov rax, [rel g_vit_ptr_mmlp]             ; load mmlp buffer pointer
     mov vvf_mmlp, rax                         ; save to local frame
; --- Phase 1: patch_embed linear, 1536 -> 1024 ---
     TRACE_STR "vvf: phase1 set_spin(1)"       ; trace: about to call set_spin for ViT
     API_CALL v7s_set_spin, 1                  ; workers commit to spinning for ViT dispatch
     TRACE_STR "vvf: phase1 patch_embed setup" ; trace: setting up patch_embed matmul args
     sub rsp, VIT_CALL7_FRAME                  ; allocate 7-arg call frame
     mov rcx, vvf_hidden                       ; load hidden buffer as output ; out = hidden
     mov rdx, vvf_pixels                       ; load pixels/patches base ; in = pixels
     mov rax, vvf_w                            ; weight struct base
     mov r8, [rax + VW_PE_W]                   ; w = pe_w
     mov r9, [rax + VW_PE_B]                   ; bias = pe_b
     mov dword [rsp + VIT_CALL_ARG5], r14d     ; arg5 = n_rows, store directly, r14d is non-volatile, holds n for the whole function
     mov eax, dword [rel g_vit_cfg + VCFG_HIDDEN] ; load VCFG_HIDDEN via scratch
     mov dword [rsp + VIT_CALL_ARG6], eax      ; arg6 = M = 1024
     mov eax, dword [rel g_vit_cfg + VCFG_PATCH_DIM] ; load VCFG_PATCH_DIM via scratch
     mov dword [rsp + VIT_CALL_ARG7], eax      ; arg7 = K = 1536
     TRACE_STR "vvf: phase1 calling v7d_vit_linear" ; trace: about to call patch_embed matmul
     call v7d_vit_linear                       ; hidden = pe_w * pixels + pe_b
     add rsp, VIT_CALL7_FRAME                  ; restore stack
     TRACE_STR "vvf: phase1 patch_embed DONE"  ; trace: patch_embed matmul returned OK
; --- Phase 2: position embeddings ---
     TRACE_STR "vvf: phase2 pos_embed"         ; trace: entering position embedding phase
; POS_EMBED_TYPE=0: Qwen3-VL bilinear-resample the 48x48 grid to gh x gw
; POS_EMBED_TYPE=1: SigLIP additive, pos table is already [n x hidden], copy directly
     cmp dword [rel g_vit_cfg + VCFG_POS_EMBED_TYPE], 0 ; 0=bilinear,Qwen3-VL ?
     jne .pos_additive                         ; 1=additive,SigLIP
     sub rsp, VIT_CALL4_FRAME                  ; allocate 4-arg call frame
     mov rcx, vvf_pos                          ; load pos buffer base ; out = pos buffer
     mov rax, vvf_w                            ; weight struct
     mov rdx, [rax + VW_POS_EMBED]             ; table = pos_embed weights
     mov r8d, vvf_gh                           ; grid_h
     mov r9d, vvf_gw                           ; grid_w
     call vit_pos_embed                        ; fill pos with bilinear-interpolated embeddings
     add rsp, VIT_CALL4_FRAME                  ; restore stack after 4-arg call
     jmp .pos_add_common                       ; skip additive path, go to shared add
.pos_additive:
; SigLIP: the immutable DB table already has the required [n x hidden] layout.
     TRACE_STR "vvf: pos_additive (SigLIP direct table)" ; no intermediate copy
     mov rax, vvf_w                            ; weight struct base
     mov rax, [rax + VW_POS_EMBED]             ; direct pointer to the position-embedding table
     mov vvf_pos, rax                          ; only the local input pointer changes
.pos_add_common:
     TRACE_STR "vvf: pos_add_common (hidden+=pos)" ; trace: about to add pos to hidden
; hidden += pos, element-wise add, SAME for both pos_embed types
     sub rsp, VIT_CALL4_FRAME                  ; allocate 4-arg call frame, 32 shadow
     mov rcx, vvf_hidden                       ; load hidden buffer as output, dst = hidden
     mov rdx, vvf_pos                          ; src = pos
     mov r8d, r14d                             ; n, row count, :111 wrote r14d, and a 32-bit count is exact here, n*hidden fits 32 bits for every supported model
     imul r8d, dword [rel g_vit_cfg + VCFG_HIDDEN] ; count = n * hidden, 2-operand memory form, legacy-encoded
     call vit_add                              ; hidden[i] += pos[i]
     add rsp, VIT_CALL4_FRAME                  ; restore stack after 4-arg call
     TRACE_STR "vvf: phase2 DONE (hidden+=pos)" ; trace: position embedding add complete
; --- Phase 3: build RoPE tables, Qwen3-VL only; SigLIP skips RoPE ---
     cmp dword [rel g_vit_cfg + VCFG_USE_ROPE], 0 ; RoPE enabled, 1=Qwen3-VL ?
     je .skip_rope_build                       ; 0=SigLIP, skip table build entirely
     sub rsp, VIT_CALL4_FRAME                  ; allocate 4-arg call frame, 32 shadow
     mov rcx, vvf_cos                          ; load cos table base, cos table output
     mov rdx, vvf_sin                          ; sin table output
     mov r8d, vvf_gh                           ; grid_h
     mov r9d, vvf_gw                           ; grid_w
     call vit_rope_build                       ; fill cos/sin tables
     add rsp, VIT_CALL4_FRAME                  ; restore stack after 4-arg call
.skip_rope_build:
; --- Phase 4: 24-layer transformer loop ---
     TRACE_STR "vvf: entering layer loop"      ; trace: about to start transformer layer loop
     xor ebx, ebx                              ; layer = 0, rbx is non-volatile: none of the loop body's 13 calls can disturb it
     mov rax, vvf_w                            ; weight struct base, kept in rax across the seeds
     mov rdx, [rax + VW_BLK_NORM1_GAMMA]       ; norm1_gamma base, layer 0
     mov vvf_n1g, rdx                          ; seed running pointer
     mov rdx, [rax + VW_BLK_NORM1_BETA]        ; norm1_beta base
     mov vvf_n1b, rdx                          ; seed running pointer
     mov rdx, [rax + VW_BLK_QKV_W]             ; qkv_w base
     mov vvf_qkvw, rdx                         ; seed running pointer
     mov rdx, [rax + VW_BLK_QKV_B]             ; qkv_b base
     mov vvf_qkvb, rdx                         ; seed running pointer
     mov rdx, [rax + VW_BLK_PROJ_W]            ; proj_w base
     mov vvf_pjw, rdx                          ; seed running pointer
     mov rdx, [rax + VW_BLK_PROJ_B]            ; proj_b base
     mov vvf_pjb, rdx                          ; seed running pointer
     mov rdx, [rax + VW_BLK_NORM2_GAMMA]       ; norm2_gamma base
     mov vvf_n2g, rdx                          ; seed running pointer
     mov rdx, [rax + VW_BLK_NORM2_BETA]        ; norm2_beta base
     mov vvf_n2b, rdx                          ; seed running pointer
     mov rdx, [rax + VW_BLK_FC1_W]             ; fc1_w base
     mov vvf_f1w, rdx                          ; seed running pointer
     mov rdx, [rax + VW_BLK_FC1_B]             ; fc1_b base
     mov vvf_f1b, rdx                          ; seed running pointer
     mov rdx, [rax + VW_BLK_FC2_W]             ; fc2_w base
     mov vvf_f2w, rdx                          ; seed running pointer
     mov rdx, [rax + VW_BLK_FC2_B]             ; fc2_b base
     mov vvf_f2b, rdx                          ; seed running pointer
.vvf_layer_loop:
; ===== Attention branch =====
; 1. norm = LayerNorm,hidden, norm1_g[L], norm1_b[L], n, dim
     mov r8, vvf_n1g                           ; gamma = norm1_gamma[L], running pointer
     mov r9, vvf_n1b                           ; beta = norm1_beta[L]
     sub rsp, VIT_CALL6_FRAME                  ; allocate 6-arg call frame, shadow + stack args
     mov rcx, vvf_norm                         ; out = norm buffer
     mov rdx, vvf_hidden                       ; in = hidden state
; r8 = gamma, r9 = beta already set
     mov dword [rsp + VIT_CALL_ARG5], r14d     ; n_rows, store directly, r14d non-volatile
     mov eax, dword [rel g_vit_cfg + VCFG_HIDDEN] ; load VCFG_HIDDEN via scratch
     mov dword [rsp + VIT_CALL_ARG6], eax      ; dim = hidden
     call vit_layernorm                        ; norm = LN,hidden, gamma, beta, n, dim
     add rsp, VIT_CALL6_FRAME                  ; restore stack after 6-arg call
; 2. qkv = linear,norm, qkv_w[L], qkv_b[L], n, 3072, 1024
     mov r8, vvf_qkvw                          ; weight = qkv_w[L], running pointer
     mov r9, vvf_qkvb                          ; bias = qkv_b[L], running pointer
     sub rsp, VIT_CALL7_FRAME                  ; 7-arg call frame
     mov rcx, vvf_qkv                          ; load qkv buffer as output ; out = qkv
     mov rdx, vvf_norm                         ; load norm buffer, LN output
; r8 = weight, r9 = bias already set
     mov dword [rsp + VIT_CALL_ARG5], r14d     ; arg5, store directly, r14d non-volatile
     mov eax, dword [rel g_vit_cfg + VCFG_QKV_DIM] ; load VCFG_QKV_DIM via scratch
     mov dword [rsp + VIT_CALL_ARG6], eax      ; arg6 = M = qkv_dim, 3072
     mov eax, dword [rel g_vit_cfg + VCFG_HIDDEN] ; load VCFG_HIDDEN via scratch
     mov dword [rsp + VIT_CALL_ARG7], eax      ; arg7 = K = hidden, 1024
     call v7d_vit_linear                       ; qkv = qkv_w * norm + qkv_b, fast path
     add rsp, VIT_CALL7_FRAME                  ; restore stack after 7-arg call
; 3. apply RoPE to Q and K in qkv buffer, Qwen3-VL only; SigLIP skips
     cmp dword [rel g_vit_cfg + VCFG_USE_ROPE], 0 ; RoPE enabled?
     je .skip_rope_apply                       ; 0=SigLIP, skip RoPE apply
     sub rsp, VIT_CALL4_FRAME                  ; allocate 4-arg call frame, 32 shadow
     mov rcx, vvf_qkv                          ; qkv buffer, Q at 0, K at 1024
     mov rdx, vvf_cos                          ; cos table
     mov r8, vvf_sin                           ; sin table
     mov r9d, r14d                             ; n patches
     call vit_rope_apply                       ; rotate Q and K by cos/sin
     add rsp, VIT_CALL4_FRAME                  ; restore stack after 4-arg call
.skip_rope_apply:
; 4. attention: attn = softmax,QK^T * scale @ V
     sub rsp, VIT_CALL4_FRAME                  ; allocate 4-arg call frame, 32 shadow
     mov rcx, vvf_attn                         ; attn output buffer
     mov rdx, vvf_qkv                          ; qkv buffer, Q,K,V interleaved
     mov r8, vvf_scores                        ; per-head scores scratch
     mov r9d, r14d                             ; n patches
     call v7d_vit_attention                    ; dispatched 16-head bidirectional attention
     add rsp, VIT_CALL4_FRAME                  ; restore stack after 4-arg call
     mov r8, vvf_pjw                           ; weight = proj_w[L], running pointer
     mov r9, vvf_pjb                           ; bias = proj_b[L], running pointer
     mov rax, vvf_hidden                       ; load hidden pointer for fused residual
     mov qword [rel g_vit_residual], rax       ; set g_vit_residual = hidden, enable fused add
     sub rsp, VIT_CALL7_FRAME                  ; allocate 7-arg call frame, shadow + stack args
     mov rcx, vvf_hidden                       ; out = hidden, fused: write directly to hidden, not norm
     mov rdx, vvf_attn                         ; in = attn
     mov dword [rsp + VIT_CALL_ARG5], r14d     ; n_rows, store directly, r14d non-volatile
     mov eax, dword [rel g_vit_cfg + VCFG_HIDDEN] ; load VCFG_HIDDEN via scratch
     mov dword [rsp + VIT_CALL_ARG6], eax      ; M = hidden, 1024
     mov dword [rsp + VIT_CALL_ARG7], eax      ; K = hidden, 1024
     call v7d_vit_linear                       ; hidden = proj_w * attn + proj_b + old_hidden, fused
     add rsp, VIT_CALL7_FRAME                  ; restore stack after 7-arg call
     mov qword [rel g_vit_residual], 0         ; clear g_vit_residual, disable fused add
; 6. norm = layernorm,hidden, norm2_g[L], norm2_b[L], n, hidden
     mov r8, vvf_n2g                           ; gamma = norm2_gamma[L], running pointer
     mov r9, vvf_n2b                           ; beta = norm2_beta[L], running pointer
     sub rsp, VIT_CALL6_FRAME                  ; allocate 6-arg call frame, shadow + stack args
     mov rcx, vvf_norm                         ; out = norm
     mov rdx, vvf_hidden                       ; in = hidden
     mov dword [rsp + VIT_CALL_ARG5], r14d     ; n_rows, store directly, r14d non-volatile
     mov eax, dword [rel g_vit_cfg + VCFG_HIDDEN] ; load VCFG_HIDDEN via scratch
     mov dword [rsp + VIT_CALL_ARG6], eax      ; dim = hidden, 1024
     call vit_layernorm                        ; LayerNorm over n rows
     add rsp, VIT_CALL6_FRAME                  ; restore stack after 6-arg call
; 7. mlp = linear,norm, fc1_w[L], fc1_b[L], n, 4096, 1024
     mov r8, vvf_f1w                           ; weight = fc1_w[L], running pointer
     mov r9, vvf_f1b                           ; bias = fc1_b[L], running pointer
     sub rsp, VIT_CALL7_FRAME                  ; allocate 7-arg call frame, shadow + stack args
     mov rcx, vvf_mlp                          ; out = mlp
     mov rdx, vvf_norm                         ; in = norm
     mov dword [rsp + VIT_CALL_ARG5], r14d     ; n_rows, store directly, r14d non-volatile
     mov eax, dword [rel g_vit_cfg + VCFG_INTER] ; load VCFG_INTER via scratch
     mov dword [rsp + VIT_CALL_ARG6], eax      ; M = inter, 4096
     mov eax, dword [rel g_vit_cfg + VCFG_HIDDEN] ; load VCFG_HIDDEN via scratch
     mov dword [rsp + VIT_CALL_ARG7], eax      ; K = hidden, 1024
     call v7d_vit_linear                       ; mlp = fc1_w * norm + fc1_b
     add rsp, VIT_CALL7_FRAME - VIT_CALL4_FRAME ; 7-arg frame torn down, 4-arg frame allocated, one adjustment
; 8. GELU,mlp in-place
     mov rcx, vvf_mlp                          ; x = mlp
     mov edx, r14d                             ; n, row count, straight into the arg register
     imul edx, dword [rel g_vit_cfg + VCFG_INTER] ; count = n * 4096
     call vit_gelu_fork                        ; GELU split between caller and W0
     add rsp, VIT_CALL4_FRAME                  ; restore stack after 4-arg call
; 9. hidden = fc2_w * mlp + fc2_b + hidden, fused residual add
; Same fused-residual mechanism as step 5 above, with fc2_w/fc2_b and mlp in place of proj_w/proj_b and attn
     mov r8, vvf_f2w                           ; weight = fc2_w[L], running pointer
     mov r9, vvf_f2b                           ; bias = fc2_b[L], running pointer
     mov rax, vvf_hidden                       ; load hidden pointer for fused residual
     mov qword [rel g_vit_residual], rax       ; set g_vit_residual = hidden, enable fused add
     sub rsp, VIT_CALL7_FRAME                  ; allocate 7-arg call frame, shadow + stack args
     mov rcx, vvf_hidden                       ; out = hidden, fused: write directly to hidden, not norm
     mov rdx, vvf_mlp                          ; in = mlp
     mov dword [rsp + VIT_CALL_ARG5], r14d     ; n_rows, store directly, r14d non-volatile
     mov eax, dword [rel g_vit_cfg + VCFG_HIDDEN] ; load VCFG_HIDDEN via scratch
     mov dword [rsp + VIT_CALL_ARG6], eax      ; M = 1024
     mov eax, dword [rel g_vit_cfg + VCFG_INTER] ; load VCFG_INTER via scratch
     mov dword [rsp + VIT_CALL_ARG7], eax      ; K = 4096
     call v7d_vit_linear                       ; hidden = fc2_w * mlp + fc2_b + old_hidden, fused
     add rsp, VIT_CALL7_FRAME                  ; restore stack after 7-arg call
     mov qword [rel g_vit_residual], 0         ; clear g_vit_residual, disable fused add
     cmp qword [rel g_vit_cfg + VCFG_N_DEEPSTACK], 0 ; model has DeepStack taps? zero -> the three tap compares below cannot be wanted
     je .vvf_layer_next                        ; no DeepStack -> next layer, same target the chain's own no-match jne uses
     cmp qword vvf_out_ds0, 0                  ; NULL = no requested DeepStack output; avoid all unused tap work
     je .vvf_layer_next                        ; skip DeepStack merger when the caller discards it
     cmp ebx, dword [rel g_vit_cfg + VCFG_DS_TAP0] ; layer == ds_tap0, 5 ?
     jne .vvf_chk_ds1                          ; not ds_tap0 -> check ds_tap1
; Run deepstack merger 0 -> out_ds0
     mov r12, vvf_out_ds0                      ; output target = ds0
     xor r13d, r13d                            ; deepstack index = 0
     jmp .vvf_do_deepstack                     ; run merger -> out_ds0
.vvf_chk_ds1:
     cmp ebx, dword [rel g_vit_cfg + VCFG_DS_TAP1] ; layer == ds_tap1, 11 ?
     jne .vvf_chk_ds2                          ; not ds_tap1 -> check ds_tap2
     mov r12, vvf_out_ds1                      ; output target = ds1
     mov r13d, 1                               ; deepstack index = 1
     jmp .vvf_do_deepstack                     ; run merger -> out_ds1
.vvf_chk_ds2:
     cmp ebx, dword [rel g_vit_cfg + VCFG_DS_TAP2] ; layer == ds_tap2, 17 ?
     jne .vvf_layer_next                       ; not ds_tap2 -> next layer
     mov r12, vvf_out_ds2                      ; output target = ds2
     mov r13d, 2                               ; deepstack index = 2
.vvf_do_deepstack:
; --- Deepstack merger for index r13d, output to r12 ---
; Normalize the existing flat hidden layout directly into packed scratch.
; Grouping consecutive patches into merge-unit rows changes no element order.
; 1. layernorm,packed, hidden, ds_norm_g[d], ds_norm_b[d], n_merged, packed_dim
     mov rax, vvf_w                            ; load weight struct base pointer
     mov r8, [rax + VW_DS_NORM_GAMMA]          ; ds norm gamma base
     mov r9, [rax + VW_DS_NORM_BETA]           ; ds norm beta base
     mov eax, r13d                             ; d, deepstack index
     imul rax, qword [rel g_vit_cfg + VCFG_PACKED_DIM_F4] ; d * packed_dim*4, per-tap norm stride
     add r8, rax                               ; r8 = ds_norm_g[d]
     add r9, rax                               ; r9 = ds_norm_b[d]
     sub rsp, VIT_CALL6_FRAME                  ; allocate 6-arg call frame, shadow + stack args
     mov rcx, vvf_packed                       ; out = packed scratch
     mov rdx, vvf_hidden                       ; in = hidden, identical flat element order
     mov eax, vvf_nmerged                      ; load n_merged, row count
     mov dword [rsp + VIT_CALL_ARG5], eax      ; n_rows = n_merged
     mov eax, dword [rel g_vit_cfg + VCFG_PACKED_DIM] ; load VCFG_PACKED_DIM via scratch
     mov dword [rsp + VIT_CALL_ARG6], eax      ; dim = packed_dim
     call vit_layernorm                        ; LayerNorm over n rows
     add rsp, VIT_CALL6_FRAME                  ; restore stack after 6-arg call
; VVF_LINEAR_FC1 -- the shared packed-dim FC1 linear call: out = mmlp, in =
; packed, M = K = packed_dim. Expects r8/r9 pointing at the selected weights
; and bias; leaves rsp on the 4-arg frame vit_gelu_fork expects.

%macro VVF_LINEAR_FC1 0
     sub rsp, VIT_CALL7_FRAME                  ; allocate 7-arg call frame, shadow + stack args
     mov rcx, vvf_mmlp                         ; out = mmlp
     mov rdx, vvf_packed                       ; in = packed
     mov eax, vvf_nmerged                      ; load n_merged, row count
     mov dword [rsp + VIT_CALL_ARG5], eax      ; n_rows
     mov eax, dword [rel g_vit_cfg + VCFG_PACKED_DIM] ; load VCFG_PACKED_DIM via scratch, reused for both M and K
     mov dword [rsp + VIT_CALL_ARG6], eax      ; M = packed_dim
     mov dword [rsp + VIT_CALL_ARG7], eax      ; K = packed_dim, eax still holds it
     call v7d_vit_linear                       ; dispatched F32xF32 matmul, out = W * in + b
     add rsp, VIT_CALL7_FRAME - VIT_CALL4_FRAME ; 7-arg frame torn down, 4-arg frame allocated, one adjustment
%endmacro
; VVF_LINEAR_FC2 out -- the shared FC2 linear call: in = mmlp, M = out_hidden,
; K = packed_dim. %1 = the output (register or frame slot); r8/r9 preselected.

%macro VVF_LINEAR_FC2 1
     sub rsp, VIT_CALL7_FRAME                  ; allocate 7-arg call frame, shadow + stack args
     mov rcx, %1                               ; out, branch-specific
     mov rdx, vvf_mmlp                         ; in = mmlp
     mov eax, vvf_nmerged                      ; load n_merged, row count
     mov dword [rsp + VIT_CALL_ARG5], eax      ; n_rows
     mov eax, dword [rel g_vit_cfg + VCFG_OUT_HIDDEN] ; load VCFG_OUT_HIDDEN via scratch
     mov dword [rsp + VIT_CALL_ARG6], eax      ; M = out_hidden
     mov eax, dword [rel g_vit_cfg + VCFG_PACKED_DIM] ; load VCFG_PACKED_DIM via scratch
     mov dword [rsp + VIT_CALL_ARG7], eax      ; K = packed_dim
     call v7d_vit_linear                       ; dispatched F32xF32 matmul, out = W * in + b
     add rsp, VIT_CALL7_FRAME                  ; restore stack after 7-arg call
%endmacro
; 2. mmlp = linear,packed, ds_fc1_w[d], ds_fc1_b[d], n_merged, 4096, 4096
     mov rax, vvf_w                            ; load weight struct base pointer
     mov r8, [rax + VW_DS_FC1_W]               ; ds fc1 weight base
     mov r9, [rax + VW_DS_FC1_B]               ; ds fc1 bias base
     mov eax, r13d                             ; d, deepstack index
     imul rax, qword [rel g_vit_cfg + VCFG_PACKED_DIM_F4] ; d * packed_dim*4, base product
     add r9, rax                               ; r9 = ds_fc1_b[d], bias offset first, reuses the base product
     imul rax, qword [rel g_vit_cfg + VCFG_PACKED_DIM] ; * packed_dim = weight stride, scales the same product in-register
     add r8, rax                               ; r8 = ds_fc1_w[d]
     VVF_LINEAR_FC1                            ; mmlp = fc1_w * packed + b, 7-arg frame -> 4-arg frame
; 4. GELU,Mmlp
     mov rcx, vvf_mmlp                         ; x = mmlp
     mov edx, vvf_nmerged                      ; n_merged, row count, straight into the arg register
     imul edx, dword [rel g_vit_cfg + VCFG_PACKED_DIM] ; count = n_merged * packed_dim
     call vit_gelu_fork                        ; layer GELU while worker pool is spinning
     add rsp, VIT_CALL4_FRAME                  ; restore stack after 4-arg call
; 5. out_ds[d] = linear,mmlp, ds_fc2_w[d], ds_fc2_b[d], n_merged, 2560, 4096
     mov rax, vvf_w                            ; load weight struct base pointer
     mov r8, [rax + VW_DS_FC2_W]               ; ds fc2 weight base
     mov r9, [rax + VW_DS_FC2_B]               ; ds fc2 bias base
     mov eax, r13d                             ; d, deepstack index
     imul rax, qword [rel g_vit_cfg + VCFG_OUT_HIDDEN] ; d * out_hidden, base product
     lea r10, [rax*VK_F4]                      ; r10 = d * out_hidden * 4, bias stride, reuses the base product
     add r9, r10                               ; r9 = ds_fc2_b[d]
     imul rax, qword [rel g_vit_cfg + VCFG_PACKED_DIM_F4] ; * packed_dim*4 = weight stride, scales the same product in-register
     add r8, rax                               ; r8 = ds_fc2_w[d]
     VVF_LINEAR_FC2 r12                        ; out_ds[d] = fc2_w * mmlp + b, r12 from branch above
.vvf_layer_next:
     inc ebx                                   ; layer++, register inc, 1 uop vs the RMW frame slot
     mov rax, qword [rel g_vit_cfg + VCFG_HIDDEN_F4] ; norm stride, hidden*4
     add vvf_n1g, rax                          ; norm1_gamma -> [L+1]
     add vvf_n1b, rax                          ; norm1_beta -> [L+1]
     add vvf_n2g, rax                          ; norm2_gamma -> [L+1]
     add vvf_n2b, rax                          ; norm2_beta -> [L+1]
     mov rax, qword [rel g_vit_cfg + VCFG_L_QKV_W] ; qkv_w per-layer stride
     add vvf_qkvw, rax                         ; qkv_w -> [L+1]
     mov rax, qword [rel g_vit_cfg + VCFG_L_QKV_B] ; qkv_b per-layer stride
     add vvf_qkvb, rax                         ; qkv_b -> [L+1]
     mov rax, qword [rel g_vit_cfg + VCFG_L_PROJ_W] ; proj_w per-layer stride
     add vvf_pjw, rax                          ; proj_w -> [L+1]
     mov rax, qword [rel g_vit_cfg + VCFG_L_PROJ_B] ; proj_b per-layer stride
     add vvf_pjb, rax                          ; proj_b -> [L+1]
     mov rax, qword [rel g_vit_cfg + VCFG_L_FC1_W] ; fc1_w per-layer stride
     add vvf_f1w, rax                          ; fc1_w -> [L+1]
     mov rax, qword [rel g_vit_cfg + VCFG_L_FC1_B] ; fc1_b per-layer stride
     add vvf_f1b, rax                          ; fc1_b -> [L+1]
     mov rax, qword [rel g_vit_cfg + VCFG_L_FC2_W] ; fc2_w per-layer stride
     add vvf_f2w, rax                          ; fc2_w -> [L+1]
     mov rax, qword [rel g_vit_cfg + VCFG_L_FC2_B] ; fc2_b per-layer stride
     add vvf_f2b, rax                          ; fc2_b -> [L+1]
     cmp ebx, dword [rel g_vit_cfg + VCFG_DEPTH] ; layer < depth, 24 ?
     jl .vvf_layer_loop                        ; next layer
.vvf_layers_done:
; --- Phase 5: merger, Qwen3-VL or post_ln, SigLIP ---
; Branch on spatial merge: MERGE>=2 = Qwen3-VL merger; MERGE==1 = SigLIP post_ln
     cmp dword [rel g_vit_cfg + VCFG_MERGE], 1 ; merge==1, no spatial merge = SigLIP ?
     je .vvf_post_ln                           ; SigLIP: skip merger, do post_ln + copy
; === Qwen3-VL main merger, LN directly into packed -> FC1 -> GELU -> FC2 -> out_main ===
; 1. packed = layernorm,hidden, mg_norm_g, mg_norm_b, n, 1024
     mov rax, vvf_w                            ; load weight struct base pointer
     mov r8, [rax + VW_MG_NORM_GAMMA]          ; main merger norm gamma
     mov r9, [rax + VW_MG_NORM_BETA]           ; main merger norm beta
     sub rsp, VIT_CALL6_FRAME                  ; allocate 6-arg call frame, shadow + stack args
     mov rcx, vvf_packed                       ; out directly into packed scratch
     mov rdx, vvf_hidden                       ; in = hidden
     mov dword [rsp + VIT_CALL_ARG5], r14d     ; n_rows, store directly, r14d non-volatile
     mov eax, dword [rel g_vit_cfg + VCFG_HIDDEN] ; load VCFG_HIDDEN via scratch
     mov dword [rsp + VIT_CALL_ARG6], eax      ; dim = hidden
     call vit_layernorm                        ; identical flat layout as the pack2x2 output
     add rsp, VIT_CALL6_FRAME                  ; restore stack after 6-arg call
; 2. mmlp = linear,packed, mg_fc1_w, mg_fc1_b, n_merged, 4096, 4096
     mov rax, vvf_w                            ; load weight struct base pointer
     mov r8, [rax + VW_MG_FC1_W]               ; main merger fc1 weight base
     mov r9, [rax + VW_MG_FC1_B]               ; main merger fc1 bias base
     VVF_LINEAR_FC1                            ; mmlp = fc1_w * packed + b, 7-arg frame -> 4-arg frame
; 4. GELU,Mmlp
     mov rcx, vvf_mmlp                         ; x = mmlp
     mov edx, vvf_nmerged                      ; n_merged, row count, straight into the arg register
     imul edx, dword [rel g_vit_cfg + VCFG_PACKED_DIM] ; count = n_merged * packed_dim
     call vit_gelu_fork                        ; GELU split between caller and W0, same as every vision GELU
     add rsp, VIT_CALL4_FRAME                  ; restore stack after 4-arg call
; 5. out_main = linear,mmlp, mg_fc2_w, mg_fc2_b, n_merged, 2560, 4096
     mov rax, vvf_w                            ; load weight struct base pointer
     mov r8, [rax + VW_MG_FC2_W]               ; main merger fc2 weight base
     mov r9, [rax + VW_MG_FC2_B]               ; main merger fc2 bias base
     VVF_LINEAR_FC2 vvf_out_main               ; out_main = fc2_w * mmlp + b
; --- No per-call buffer free: scratch block is pre-allocated at init ---
     jmp .vvf_forward_done                     ; Qwen3-VL merger done, skip post_ln path
.vvf_post_ln:
     cmp dword [rel g_vit_cfg + VCFG_APPLY_POST_LN], 0 ; apply post_layernorm?
     jne .vvf_post_ln_apply                    ; yes, SmolVLM/Idefics3 path
; --- LLaVA-OV: copy raw hidden -> out_main, NO post_layernorm ---
     TRACE_STR "vvf: post_ln (raw copy)"       ; trace: LLaVA-OV raw-copy path
     sub rsp, VIT_CALL4_FRAME                  ; 3-arg call frame, 32 shadow
     mov rcx, vvf_out_main                     ; dst = out_main
     mov rdx, vvf_hidden                       ; src = raw last-layer hidden
     mov eax, r14d                             ; n patches
     imul eax, dword [rel g_vit_cfg + VCFG_HIDDEN] ; n * hidden elements, fits 32 bits like every other count in this file
     shl eax, 2                                ; byte count
     mov r8d, eax                              ; arg3 = byte_count, dispatched memcpy ABI is rcx/rdx/r8d
     call v7d_memcpy                           ; dispatched copy, threshold decides ERMS vs workers
     add rsp, VIT_CALL4_FRAME                  ; restore stack
     jmp .vvf_forward_done                     ; LLaVA-OV path done
.vvf_post_ln_apply:
; --- SmolVLM/Idefics3: out_main = post_layernorm,hidden, mg_norm_g/b, n, hidden ---
     mov rax, vvf_w                            ; load weight struct base pointer
     mov r8, [rax + VW_MG_NORM_GAMMA]          ; post_layernorm gamma, stored in mg_norm slot
     mov r9, [rax + VW_MG_NORM_BETA]           ; post_layernorm beta, same slot
     sub rsp, VIT_CALL6_FRAME                  ; allocate 6-arg call frame
     mov rcx, vvf_out_main                     ; out = out_main
     mov rdx, vvf_hidden                       ; in = hidden, raw last-layer output
     mov eax, r14d                             ; n patches
     mov dword [rsp + VIT_CALL_ARG5], eax      ; n_rows = n
     mov eax, dword [rel g_vit_cfg + VCFG_HIDDEN] ; dim = hidden, 768 SmolVLM-256M
     mov dword [rsp + VIT_CALL_ARG6], eax      ; dim
     call vit_layernorm                        ; out_main = LN,hidden, gamma, beta, n, hidden
     add rsp, VIT_CALL6_FRAME                  ; restore stack after 6-arg call
.vvf_forward_done:
     vzeroupper                                ; clean AVX state
     SF_END rbx, rsi, rdi, r12, r13, r14       ; restore non-volatiles, return, list matches the prologue
%include "vit_encode_callback.inc"
