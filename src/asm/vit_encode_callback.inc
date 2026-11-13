; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define VCB_RGB_SCALE 2                        ; SIB scale for *3 (reg + reg*2), RGB channels
%define VCB_DWORD_SCALE 4                      ; SIB scale: one float / one dword element
%define VCB_XMM_STASH_BYTES 160                ; ten 16-byte xmm stash slots below rsp for the mask-holding vector run
%define VCB_STASH_OFF0 0                       ; stash slot of xmm6
%define VCB_STASH_OFF1 16                      ; stash slot of xmm7
%define VCB_STASH_OFF2 32                      ; stash slot of xmm8
%define VCB_STASH_OFF3 48                      ; stash slot of xmm9
%define VCB_STASH_OFF4 64                      ; stash slot of xmm10
%define VCB_STASH_OFF5 80                      ; stash slot of xmm11
%define VCB_STASH_OFF6 96                      ; stash slot of xmm12
%define VCB_STASH_OFF7 112                     ; stash slot of xmm13
%define VCB_STASH_OFF8 128                     ; stash slot of xmm14
%define VCB_STASH_OFF9 144                     ; stash slot of xmm15

; VCB_LAYOUT_WEIGHTS -- lay out the 27 vit_weights_t pointers over the weights
; blob. IN: rax = running blob pointer, positioned past the magic dword.
; OUT: rax advanced past ds_fc2_b, vcb_weights fully populated. Clobbers rcx.

%macro VCB_LAYOUT_WEIGHTS 0
     lea rcx, vcb_weights                      ; struct base on stack
     mov [rcx + VW_PE_W], rax                  ; pe_w
     add rax, qword [rel g_vit_cfg + VCFG_PE_W_STRIDE] ; advance past pe_w
     mov [rcx + VW_PE_B], rax                  ; pe_b
     add rax, qword [rel g_vit_cfg + VCFG_HIDDEN_F4] ; advance past pe_b
     mov [rcx + VW_POS_EMBED], rax             ; pos_embed
     add rax, qword [rel g_vit_cfg + VCFG_POS_EMBED_BYTES] ; advance past pos_embed
     mov [rcx + VW_BLK_NORM1_GAMMA], rax       ; blk_norm1_gamma
     add rax, qword [rel g_vit_cfg + VCFG_BLK_NORM1_STRIDE] ; advance past norm1_gamma
     mov [rcx + VW_BLK_NORM1_BETA], rax        ; blk_norm1_beta
     add rax, qword [rel g_vit_cfg + VCFG_BLK_NORM1_STRIDE] ; advance past norm1_beta
     mov [rcx + VW_BLK_QKV_W], rax             ; blk_qkv_w
     add rax, qword [rel g_vit_cfg + VCFG_BLK_QKV_W_STRIDE] ; advance past qkv_w
     mov [rcx + VW_BLK_QKV_B], rax             ; blk_qkv_b
     add rax, qword [rel g_vit_cfg + VCFG_BLK_QKV_B_STRIDE] ; advance past qkv_b
     mov [rcx + VW_BLK_PROJ_W], rax            ; blk_proj_w
     add rax, qword [rel g_vit_cfg + VCFG_BLK_PROJ_W_STRIDE] ; advance past proj_w
     mov [rcx + VW_BLK_PROJ_B], rax            ; blk_proj_b
     add rax, qword [rel g_vit_cfg + VCFG_BLK_PROJ_B_STRIDE] ; advance past proj_b
     mov [rcx + VW_BLK_NORM2_GAMMA], rax       ; blk_norm2_gamma
     add rax, qword [rel g_vit_cfg + VCFG_BLK_NORM1_STRIDE] ; advance past norm2_gamma, =norm1 stride: no VCFG_BLK_NORM2_STRIDE exists, both LayerNorms are depth x hidden
     mov [rcx + VW_BLK_NORM2_BETA], rax        ; blk_norm2_beta
     add rax, qword [rel g_vit_cfg + VCFG_BLK_NORM1_STRIDE] ; advance past norm2_beta, =norm1 stride
     mov [rcx + VW_BLK_FC1_W], rax             ; blk_fc1_w
     add rax, qword [rel g_vit_cfg + VCFG_BLK_FC1_W_STRIDE] ; advance past fc1_w
     mov [rcx + VW_BLK_FC1_B], rax             ; blk_fc1_b
     add rax, qword [rel g_vit_cfg + VCFG_BLK_FC1_B_STRIDE] ; advance past fc1_b
     mov [rcx + VW_BLK_FC2_W], rax             ; blk_fc2_w
     add rax, qword [rel g_vit_cfg + VCFG_BLK_FC2_W_STRIDE] ; advance past fc2_w
     mov [rcx + VW_BLK_FC2_B], rax             ; blk_fc2_b
     add rax, qword [rel g_vit_cfg + VCFG_BLK_FC2_B_STRIDE] ; advance past fc2_b
     mov [rcx + VW_MG_NORM_GAMMA], rax         ; mg_norm_gamma
     add rax, qword [rel g_vit_cfg + VCFG_HIDDEN_F4] ; advance past mg_norm_gamma
     mov [rcx + VW_MG_NORM_BETA], rax          ; mg_norm_beta
     add rax, qword [rel g_vit_cfg + VCFG_HIDDEN_F4] ; advance past mg_norm_beta
     mov [rcx + VW_MG_FC1_W], rax              ; mg_fc1_w
     add rax, qword [rel g_vit_cfg + VCFG_MG_FC1_W_STRIDE] ; advance past mg_fc1_w
     mov [rcx + VW_MG_FC1_B], rax              ; mg_fc1_b
     add rax, qword [rel g_vit_cfg + VCFG_MG_FC1_B_STRIDE] ; advance past mg_fc1_b
     mov [rcx + VW_MG_FC2_W], rax              ; mg_fc2_w
     add rax, qword [rel g_vit_cfg + VCFG_MG_FC2_W_STRIDE] ; advance past mg_fc2_w
     mov [rcx + VW_MG_FC2_B], rax              ; mg_fc2_b
     add rax, qword [rel g_vit_cfg + VCFG_MG_FC2_B_STRIDE] ; advance past mg_fc2_b
     mov [rcx + VW_DS_NORM_GAMMA], rax         ; ds_norm_gamma
     add rax, qword [rel g_vit_cfg + VCFG_DS_FC1_B_STRIDE] ; advance past ds_norm_gamma, = n_ds * packed * 4
     mov [rcx + VW_DS_NORM_BETA], rax          ; ds_norm_beta
     add rax, qword [rel g_vit_cfg + VCFG_DS_FC1_B_STRIDE] ; advance past ds_norm_beta, = n_ds * packed * 4
     mov [rcx + VW_DS_FC1_W], rax              ; ds_fc1_w
     add rax, qword [rel g_vit_cfg + VCFG_DS_FC1_W_STRIDE] ; advance past ds_fc1_w
     mov [rcx + VW_DS_FC1_B], rax              ; ds_fc1_b
     add rax, qword [rel g_vit_cfg + VCFG_DS_FC1_B_STRIDE] ; advance past ds_fc1_b
     mov [rcx + VW_DS_FC2_W], rax              ; ds_fc2_w
     add rax, qword [rel g_vit_cfg + VCFG_DS_FC2_W_STRIDE] ; advance past ds_fc2_w
     mov [rcx + VW_DS_FC2_B], rax              ; ds_fc2_b
%endmacro

; VCB_RGB8_UNPACK -- unpack 8 interleaved RGB bytes pairs into planar float
; triples. IN: xmm0 = RGB bytes 0..15, xmm1 = bytes 16..23 (8-byte load),
; xmm6..xmm11 = the six planar shuffle masks. OUT: ymm2 = R float,
; ymm12 = G float, ymm14 = B float, all /1 unscaled. Clobbers xmm2/3/12-15.

%macro VCB_RGB8_UNPACK 0
     vpshufb xmm2, xmm0, xmm6                  ; R low bytes
     vpshufb xmm3, xmm1, xmm7                  ; R high bytes
     vpshufb xmm12, xmm0, xmm8                 ; G low bytes
     vpshufb xmm13, xmm1, xmm9                 ; G high bytes
     vpshufb xmm14, xmm0, xmm10                ; B low bytes
     vpshufb xmm15, xmm1, xmm11                ; B high bytes
     vpor xmm2, xmm2, xmm3                     ; combine R bytes
     vpor xmm12, xmm12, xmm13                  ; combine G bytes
     vpor xmm14, xmm14, xmm15                  ; combine B bytes
     vpmovzxbd ymm2, xmm2                      ; expand R to int32
     vpmovzxbd ymm12, xmm12                    ; expand G to int32
     vpmovzxbd ymm14, xmm14                    ; expand B to int32
     vcvtdq2ps ymm2, ymm2                      ; R float
     vcvtdq2ps ymm12, ymm12                    ; G float
     vcvtdq2ps ymm14, ymm14                    ; B float
%endmacro

     global v7s_vit_encode_callback            ; export v7s_vit_encode_callback entry point
v7s_vit_encode_callback:
     SF_START v7s_vit_encode_callback, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles, reserve frame
     SF_QWORD vcb_raw                          ; const uint8_t* raw_rgb, caller-owned, NOT freed by us
     SF_DWORD vcb_img_w                        ; image width, from caller
     SF_DWORD vcb_img_h                        ; image height, from caller
     SF_QWORD vcb_out_emb                      ; float** out_emb
     SF_QWORD vcb_out_cnt                      ; int* out_count
     SF_QWORD vcb_out_ds0                      ; float** out_ds0
     SF_QWORD vcb_out_ds1                      ; float** out_ds1
     SF_QWORD vcb_out_ds2                      ; float** out_ds2
     SF_QWORD vcb_blob                         ; vit_weights blob from ht_find
     SF_QWORD vcb_res                          ; normalized planar image buffer
     SF_QWORD vcb_patches                      ; preprocessed patch features [n * 1536]
     SF_QWORD vcb_out_main                     ; main output buffer, ViT scratch block
     SF_QWORD vcb_ds0_buf                      ; ds0 buffer, ViT scratch block
     SF_QWORD vcb_ds1_buf                      ; ds1 buffer, ViT scratch block
     SF_QWORD vcb_ds2_buf                      ; ds2 buffer, ViT scratch block
     SF_DWORD vcb_gh                           ; grid_h = img_h / 16
     SF_DWORD vcb_gw                           ; grid_w = img_w / 16
     SF_DWORD vcb_n                            ; n = gh * gw
     SF_DWORD vcb_nmerged                      ; n_merged = n / 4
     SF_DWORD vcb_total_px                     ; img_w * img_h total pixels
     SF_DWORD vcb_h1                           ; patchify outer loop: h1 = 0..gh/merge-1
     SF_DWORD vcb_w1                           ; patchify inner loop: w1 = 0..gw/merge-1
     SF_DWORD vcb_w1_bound                     ; gw / merge, loop-invariant w1 bound, hoisted
     SF_DWORD vcb_h1_bound                     ; gh / merge, loop-invariant h1 bound, hoisted
     SF_DWORD vcb_mh                           ; patchify merge-y: mh = 0..1
     SF_DWORD vcb_mw                           ; patchify merge-x: mw = 0..1
     SF_QWORD vcb_fc1_w_bytes                  ; fc1_w byte size, shared by FC1 + FC2 projector setup
     SF_QWORD vcb_proj_blob                    ; projector weights blob from ht_find, SigLIP only
     SF_QWORD vcb_proj_inter_buf               ; projector FC1 intermediate buffer
     SF_QWORD vcb_proj_out_buf                 ; projector FC2 output buffer, final embeddings
     SF_DWORD vcb_grid_h_orig                  ; anyres A7: original grid rows = best_h / VCFG_IMAGE_SIZE, g_ar_grid_h, computed at A4, the only anyres derived value still parked in a frame slot; the rest live in registers and this code only READS the canonical g_ar_grid_* BSS, so they stay untouched
     SF_QWORD vcb_ar_fc1_w                     ; anyres projector FC1 weight base, hoisted once per encode
     SF_QWORD vcb_ar_fc1_b                     ; anyres projector FC1 bias base, hoisted
     SF_QWORD vcb_ar_fc2_w                     ; anyres projector FC2 weight base, hoisted
     SF_QWORD vcb_ar_fc2_b                     ; anyres projector FC2 bias base, hoisted
     SF_DWORD vcb_img_sz                       ; anyres tile side, DB-owned VCFG_IMAGE_SIZE copy, ALL anyres tile geometry reads this slot, never a compile-time 384
     SF_QWORD vcb_ps2_f4                       ; merge patchify: ps^2*4, the per-channel dst cursor advance, slot, not a register, every GPR is live in that nest
     SF_ARRAY vcb_weights, SIZEOF_VIT_WEIGHTS  ; vit_weights_t struct, 216 bytes
     SF_COMMIT                                 ; finalize stack frame layout
; Save input arguments, 8-arg ABI: rcx,rdx,r8,r9 + 4 stack args
     mov vcb_raw, rcx                          ; save raw_rgb pointer, caller-owned
     mov vcb_img_w, edx                        ; save image width
     mov vcb_img_h, r8d                        ; save image height
     mov vcb_out_emb, r9                       ; save out_emb pointer
     mov rax, [rbp + VK_ARG5]                  ; load out_count
     mov vcb_out_cnt, rax                      ; save out_count pointer
     mov rax, [rbp + VK_ARG6]                  ; load out_ds0
     mov vcb_out_ds0, rax                      ; save out_ds0 pointer
     mov rax, [rbp + VK_ARG7]                  ; load out_ds1
     mov vcb_out_ds1, rax                      ; save out_ds1 pointer
     mov rax, [rbp + VK_ARG8]                  ; load out_ds2
     mov vcb_out_ds2, rax                      ; save out_ds2 pointer
     cmp qword [rel g_vit_cfg + VCFG_ANYRES_ENABLED], 0 ; anyres model?
     jne .vcb_anyres                           ; yes: take the multi-tile anyres path
     mov rcx, vcb_raw                          ; arg1 = src_rgb, caller-owned
     mov edx, vcb_img_w                        ; arg2 = src_w
     mov r8d, vcb_img_h                        ; arg3 = src_h
     call vit_input_fn                         ; rax = rescaled RGB buffer, or src, if same size
     mov vcb_raw, rax                          ; use the rescaled buffer from here on
     TRACE_STR "vcb: after vit_input_fn"       ; debug: scaler returned
     TRACE_REG "raw", rax                      ; debug: raw RGB buffer, rescaled
     mov rax, [rel g_vis_target_w]             ; target width chosen by the scaler
     mov vcb_img_w, eax                        ; image width = target width
     TRACE_REG "img_w", rax                    ; debug: scaled image width
     mov rax, [rel g_vis_target_h]             ; target height chosen by the scaler
     mov vcb_img_h, eax                        ; image height = target height
     TRACE_REG "img_h", rax                    ; debug: scaled image height
     TRACE_STR "vcb: step1 ht_find vit_weights" ; trace: about to look up ViT weights
     lea rcx, [rel sz_vit_w_key]               ; key = "__sys_vit_weights"
     xor edx, edx                              ; no out_len
     call ht_find_or_fatal                     ; rax = blob ptr, FATALs with the key name on miss
     mov vcb_blob, rax                         ; save blob pointer
     TRACE_REG "vcb: vit_blob", rax            ; trace: ViT weights blob base address
; --- Step 2: Lay out vit_weights_t struct over the blob ---
     mov rax, vcb_blob                         ; load blob pointer from stack
     add rax, 4                                ; skip reserved dword, fptr = blob + 4
; rax now holds the running pointer; the macro walks every stride below
     VCB_LAYOUT_WEIGHTS                        ; lay out all 27 pointers with the original stride sequence
     add rax, qword [rel g_vit_cfg + VCFG_DS_FC2_B_STRIDE] ; advance past ds_fc2_b, single-tile-only tail stride
     TRACE_REG "vcb: fptr_end", rax            ; trace: running pointer after all weight strides
; --- Step 3: Compute grid + total pixels from caller-supplied dimensions ---
; No file I/O, raw_rgb buffer + img_w + img_h are passed by the caller.
     mov eax, vcb_img_w                        ; load image width
     imul eax, vcb_img_h                       ; w * h pixels
     mov vcb_total_px, eax                     ; save total pixel count
; --- Step 4: Preprocess, normalize + patchify ---
; Compute grid dimensions, data-driven patch_size, not hardcoded /16
     mov eax, vcb_img_h                        ; load image height
     xor edx, edx                              ; zero upper dividend bits
     div dword [rel g_vit_cfg + VCFG_PATCH_SIZE] ; img_h / patch_size = grid_h
     mov vcb_gh, eax                           ; save grid_h
     mov eax, vcb_img_w                        ; load image width
     xor edx, edx                              ; zero upper dividend bits
     div dword [rel g_vit_cfg + VCFG_PATCH_SIZE] ; img_w / patch_size = grid_w
     mov vcb_gw, eax                           ; save grid_w
     mov eax, vcb_gh                           ; reload grid_h
     imul eax, vcb_gw                          ; gh * gw
     mov vcb_n, eax                            ; n = gh * gw
; n_merged = n /, merge*merge.Qwen3-VL merge=2: n/4. SigLIP merge=1: n/1 = n.
     xor edx, edx                              ; zero upper dividend bits
     div dword [rel g_vit_cfg + VCFG_MERGE_UNIT] ; n / merge^2, precomputed in DB, offset 184 = n_merged
     mov vcb_nmerged, eax                      ; save n_merged
     mov eax, vcb_n                            ; reload n for tracing
     TRACE_REG "vcb: n", rax                   ; trace: total patch count, gh*gw
     mov eax, vcb_nmerged                      ; reload n_merged for tracing
     TRACE_REG "vcb: n_merged", rax            ; trace: merged token count
; Allocate normalized planar image: res[3 * total_px] floats = 3 * total_px * 4 bytes
; All image-size products use 64-bit arithmetic to avoid overflowing intermediate byte counts.
     mov eax, vcb_total_px                     ; pixel count per channel
     shl rax, 2                                ; * sizeof,float = bytes per channel, 64-bit
     lea rcx, [rax + rax*VCB_RGB_SCALE]        ; * 3 channels = total bytes for res, 64-bit, *3 of nonnegative bytes
     call mdb_alloc_mem                        ; rax = res buffer
     mov vcb_res, rax                          ; save res pointer
; Allocate patches buffer: n * patch_dim * 4 bytes
     mov ecx, dword [rel g_vit_cfg + VCFG_PATCH_DIM_F4] ; bytes per patch, patch_dim * sizeof,float
     mov eax, vcb_n                            ; n patches
     imul rcx, rax                             ; total bytes = per_patch * count
     call mdb_alloc_mem                        ; rax = patches buffer
     mov vcb_patches, rax                      ; save patches pointer
     mov r12, vcb_raw                          ; r12 = running raw RGB pointer
     mov r13, vcb_res                          ; r13 = res planar base
     mov r14d, vcb_total_px                    ; r14d = total_px, plane stride in floats; rdi/r15 deltas from rsi are constant per call
     mov ebx, r14d                             ; ebx = remaining pixel count, mirror from reg
     mov rsi, r13                              ; rsi = running plane-0, R output ptr, single cursor, G/B stores are rsi-indexed
     vbroadcastss ymm4, [rel vit_inv_1275]     ; 1/127.5 broadcast, loop-invariant
     vbroadcastss ymm5, [rel vit_one_float]    ; 1.0 broadcast, loop-invariant
     mov eax, ebx                              ; total pixels
     and eax, -8                               ; vector trip count, multiple of 8
     jz .vcb_norm_vec_done                     ; no vector iterations, the jz MUST precede the 'and ebx,7' below, which overwrites ZF
     and ebx, 7                                ; scalar tail count, 0..7
; stash the six non-volatile XMM the vector run holds for masks
     sub rsp, VCB_XMM_STASH_BYTES              ; six 16-byte slots below rsp, RBP-relative SF slots keep working
     vmovdqu [rsp + VCB_STASH_OFF0], xmm6      ; stash xmm6, restored after the loop
     vmovdqu [rsp + VCB_STASH_OFF1], xmm7      ; stash xmm7
     vmovdqu [rsp + VCB_STASH_OFF2], xmm8      ; stash xmm8
     vmovdqu [rsp + VCB_STASH_OFF3], xmm9      ; stash xmm9
     vmovdqu [rsp + VCB_STASH_OFF4], xmm10     ; stash xmm10
     vmovdqu [rsp + VCB_STASH_OFF5], xmm11     ; stash xmm11
     vmovdqu [rsp + VCB_STASH_OFF6], xmm12     ; stash xmm12, G-low intermediate
     vmovdqu [rsp + VCB_STASH_OFF7], xmm13     ; stash xmm13, G-high intermediate
     vmovdqu [rsp + VCB_STASH_OFF8], xmm14     ; stash xmm14, B-low intermediate
     vmovdqu [rsp + VCB_STASH_OFF9], xmm15     ; stash xmm15, B-high intermediate
     vmovdqu xmm6, [rel ar_mask_r_lo]          ; xmm6 = R-low planar mask, invariant over the whole vector run
     vmovdqu xmm7, [rel ar_mask_r_hi]          ; xmm7 = R-high planar mask
     vmovdqu xmm8, [rel ar_mask_g_lo]          ; xmm8 = G-low planar mask
     vmovdqu xmm9, [rel ar_mask_g_hi]          ; xmm9 = G-high planar mask
     vmovdqu xmm10, [rel ar_mask_b_lo]         ; xmm10 = B-low planar mask
     vmovdqu xmm11, [rel ar_mask_b_hi]         ; xmm11 = B-high planar mask
.vcb_norm_vec_loop:
     vmovdqu xmm0, [r12]                       ; RGB bytes 0..15, raw lo, survives all six shuffles
     vmovq xmm1, [r12 + VIT_RGB_LO_HALF]       ; RGB bytes 16..23 only, 8-byte load, zero-extends hi 64 bits; a 16-byte vmovdqu would overread past the raw buffer end on the last iteration
     VCB_RGB8_UNPACK                           ; planar R/G/B floats into ymm2/ymm12/ymm14
     vfmsub132ps ymm2, ymm5, ymm4              ; ymm2 = ymm2*ymm4 - ymm5 =, float R *, 1/127.5 - 1.0, FMSUB132: dst = dst*src3 - src2; one rounding
     vfmsub132ps ymm12, ymm5, ymm4             ; ymm12 =, float G *, 1/127.5 - 1.0, one rounding
     vfmsub132ps ymm14, ymm5, ymm4             ; ymm14 =, float B *, 1/127.5 - 1.0, one rounding
     vmovups [rsi], ymm2                       ; store 8 R floats, plane 0 at cursor
     vmovups [rsi + r14*VK_F4], ymm12          ; store 8 G floats, plane 1 = cursor + 1 stride, indexed
     vmovups [rsi + r14*VK_F4 + r14*VK_F4], ymm14 ; store 8 B floats, plane 2 = cursor + 2 strides, indexed
     add r12, VIT_RGB_VEC_BYTES                ; src += 24, 8 pixels x 3 bytes
     add rsi, VIT_YMM_BYTES                    ; single output cursor += 32, 8 floats
     sub eax, 8                                ; 8 fewer vector pixels
     jnz .vcb_norm_vec_loop                    ; next 8 pixels
     vmovdqu xmm6, [rsp + VCB_STASH_OFF0]      ; restore the vector-run-stashed non-volatile XMMs
     vmovdqu xmm7, [rsp + VCB_STASH_OFF1]      ; restore xmm7
     vmovdqu xmm8, [rsp + VCB_STASH_OFF2]      ; restore xmm8
     vmovdqu xmm9, [rsp + VCB_STASH_OFF3]      ; restore xmm9
     vmovdqu xmm10, [rsp + VCB_STASH_OFF4]     ; restore xmm10
     vmovdqu xmm11, [rsp + VCB_STASH_OFF5]     ; restore xmm11
     vmovdqu xmm12, [rsp + VCB_STASH_OFF6]     ; restore xmm12
     vmovdqu xmm13, [rsp + VCB_STASH_OFF7]     ; restore xmm13
     vmovdqu xmm14, [rsp + VCB_STASH_OFF8]     ; restore xmm14
     vmovdqu xmm15, [rsp + VCB_STASH_OFF9]     ; restore xmm15
     add rsp, VCB_XMM_STASH_BYTES              ; release the ten stash slots
.vcb_norm_vec_done:
     lea rdi, [rsi + r14*VK_F4]                ; reconstruct plane-1, G tail cursor = plane-0 cursor + 1 stride
     lea r15, [rsi + r14*VK_F4 + r14*VK_F4]    ; reconstruct plane-2, B tail cursor = plane-0 cursor + 2 strides
.vcb_norm_loop:
     test ebx, ebx                             ; any pixels remaining?
     jle .vcb_norm_done                        ; no, all pixels processed
     movzx eax, byte [r12]                     ; load R byte, 0..255
     vcvtsi2ss xmm1, xmm1, eax                 ; float R
     movzx eax, byte [r12 + VIT_RAW_G_OFF]     ; load G byte
     vcvtsi2ss xmm2, xmm2, eax                 ; float G
     movzx eax, byte [r12 + VIT_RAW_B_OFF]     ; load B byte
     vcvtsi2ss xmm3, xmm3, eax                 ; float B
     vfmsub132ss xmm1, xmm5, xmm4              ; R * 1/127.5 - 1.0, one rounding
     vfmsub132ss xmm2, xmm5, xmm4              ; G * 1/127.5 - 1.0, one rounding
     vfmsub132ss xmm3, xmm5, xmm4              ; B * 1/127.5 - 1.0, one rounding
     vmovss [rsi], xmm1                        ; store R to res plane 0
     vmovss [rdi], xmm2                        ; store G to res plane 1
     vmovss [r15], xmm3                        ; store B to res plane 2
     add r12, VIT_RAW_RGB_CHANS                ; raw += 3 bytes, next pixel
     add rsi, VK_F4                            ; plane 0 out += 4 bytes
     add rdi, VK_F4                            ; plane 1 out += 4 bytes
     add r15, VK_F4                            ; plane 2 out += 4 bytes
     dec ebx                                   ; remaining pixel count--
     jnz .vcb_norm_loop                        ; next pixel
.vcb_norm_done:
     mov r12, vcb_patches                      ; running patches output pointer
     mov r13, vcb_res                          ; res planar image base
     mov r14d, vcb_img_w                       ; image width = row stride
     mov r15d, vcb_total_px                    ; total_px = plane element stride
     shl r15, 2                                ; r15 = plane BYTE stride, total_px*4, the per-channel src advance, mov r15d above zero-extends, so this shift is the 64-bit product
; Branch: MERGE>=2 = Qwen3-VL 4-loop, h1/w1/mh/mw ; MERGE==1 = SigLIP row-major
     cmp dword [rel g_vit_cfg + VCFG_MERGE], 1 ; merge==1, SigLIP ?
     je .vcb_patchify_simple                   ; yes, simple row-major patchify
.vcb_patchify_ps_ok:
; Hoist the loop-invariant division bounds, gw/merge, gh/merge out of the
; h1/w1 loops.
     mov eax, vcb_gw                           ; grid_w
     xor edx, edx                              ; zero upper dividend bits for div
     div dword [rel g_vit_cfg + VCFG_MERGE]    ; eax = gw / merge
     mov vcb_w1_bound, eax                     ; stash w1 loop bound
     mov eax, vcb_gh                           ; grid_h
     xor edx, edx                              ; zero upper dividend bits for div
     div dword [rel g_vit_cfg + VCFG_MERGE]    ; eax = gh / merge
     mov vcb_h1_bound, eax                     ; stash h1 loop bound
; Per-row cursor advances for the additive patchify cursors are loop-constant.
; No calls anywhere in the patchify nest, so the volatile regs survive intact.
     lea r10d, [r14*VCB_DWORD_SCALE]           ; r10 = img_w*4 = src advance per ph row, loop-constant
     mov r11d, dword [rel g_vit_cfg + VCFG_PATCH_SIZE_F4] ; r11 = ps*4 = dst advance per ph row, loop-constant; precomputed field, low 32 bits exact
     mov eax, dword [rel g_vit_cfg + VCFG_PATCH_SIZE] ; ps, dword read zero-extends rax
     imul rax, rax                             ; rax = ps^2
     shl rax, 2                                ; ps^2*4 = per-channel dst cursor advance, 32-bit shl zero-extends rax
     mov vcb_ps2_f4, rax                       ; once per encode, in a FRAME SLOT: every GPR is live in this nest, and 'add r64, m64' is the one form that costs no extra instruction
; h1 = 0..gh/merge-1, outer loop
     mov dword vcb_h1, 0                       ; h1 = 0
.vcb_h1_loop:
; w1 = 0..gw/merge-1
     mov dword vcb_w1, 0                       ; w1 = 0
.vcb_w1_loop:
; mh = 0..1, merge-y within the 2x2 patch-merge block
     mov dword vcb_mh, 0                       ; mh = 0
.vcb_mh_loop:
; mw = 0..1, merge-x within the 2x2 patch-merge block
     mov dword vcb_mw, 0                       ; mw = 0
.vcb_mw_loop:
; y_base =, h1*merge + mh * patch_size, top row of this patch in the source image
     mov eax, vcb_h1                           ; h1
     imul eax, dword [rel g_vit_cfg + VCFG_MERGE] ; h1 * merge
     add eax, vcb_mh                           ; h1*merge + mh
     imul eax, dword [rel g_vit_cfg + VCFG_PATCH_SIZE] ; * patch_size = y_base
     mov r9d, eax                              ; r9d = y_base, volatile; no calls until inner loop
; x_base =, w1*merge + mw * patch_size, left col of this patch
     mov eax, vcb_w1                           ; w1
     imul eax, dword [rel g_vit_cfg + VCFG_MERGE] ; w1 * merge
     add eax, vcb_mw                           ; w1*merge + mw
     imul eax, dword [rel g_vit_cfg + VCFG_PATCH_SIZE] ; * patch_size = x_base
     mov ebx, eax                              ; ebx = x_base, non-volatile across inner loops
     mov eax, r9d                              ; y_base
     imul eax, r14d                            ; y_base * img_w, row stride
     add eax, ebx                              ; + x_base
     lea rax, [r13 + rax*VCB_DWORD_SCALE]      ; * sizeof,float + res base = src base for c = 0, add eax above zero-extended the pixel offset
     mov rbx, rax                              ; rbx = running src base, x_base is dead from here
     mov rdi, r12                              ; rdi = t=0 row-0 cursor for c = 0, t0base,c = r12 +, 2c *ps^2*4
     mov r8, rdi                               ; r8 = copy of t0base,0
     add r8, vcb_ps2_f4                        ; r8 = t=1 row-0 cursor for c = 0, t1base,c = t0base,c + ps^2*4
     xor ecx, ecx                              ; c = 0
.vcb_c_loop:
     mov rsi, rbx                              ; src row ph=0 cursor for this channel
     add rbx, r15                              ; advance to the next channel's src base
     mov edx, dword [rel g_vit_cfg + VCFG_PATCH_SIZE] ; edx = remaining patch rows, countdown; ps == 16 is the import's contract for merge>=2 models
.vcb_ph_loop:
     vmovups ymm0, [rsi]                       ; first 8 floats
     vmovups ymm1, [rsi + VIT_YMM_BYTES]       ; second 8 floats
; Store to t=0 destination, rdi cursor
     vmovups [rdi], ymm0                       ; t=0 first 8 floats
     vmovups [rdi + VIT_YMM_BYTES], ymm1       ; t=0 second 8 floats
; Store same data to t=1 destination, no reload
     vmovups [r8], ymm0                        ; t=1 first 8 floats
     vmovups [r8 + VIT_YMM_BYTES], ymm1        ; t=1 second 8 floats
     add rsi, r10                              ; src += img_w*4, next source row
     add rdi, r11                              ; t=0 dst += ps*4, next patch row
     add r8, r11                               ; t=1 dst += ps*4, next patch row
     dec edx                                   ; one patch row done
     jnz .vcb_ph_loop                          ; next row, last iteration falls through to the exit
.vcb_c_next:
     add rdi, vcb_ps2_f4                       ; next channel's t=0 row-0 cursor, this channel's t=1 base plus ps^2*4
     add r8, vcb_ps2_f4                        ; next channel's t=1 row-0 cursor
     inc ecx                                   ; c++
     cmp ecx, VIT_RAW_RGB_CHANS                ; c < 3?
     jl .vcb_c_loop                            ; next channel
.vcb_patch_done:
; Advance to next patch, output is block-major: patch_dim floats per patch
     add r12, qword [rel g_vit_cfg + VCFG_PATCH_DIM_F4] ; patches pointer += patch_dim*4 bytes
     inc dword vcb_mw                          ; mw++
     mov eax, dword [rel g_vit_cfg + VCFG_MERGE] ; load merge, 2 into scratch
     cmp dword vcb_mw, eax                     ; mw < merge?
     jl .vcb_mw_loop                           ; next mw
.vcb_mh_next:
     inc dword vcb_mh                          ; mh++
     mov eax, dword [rel g_vit_cfg + VCFG_MERGE] ; load merge, 2 into scratch
     cmp dword vcb_mh, eax                     ; mh < merge?
     jl .vcb_mh_loop                           ; next mh
.vcb_w1_next:
     inc dword vcb_w1                          ; w1++
     mov eax, vcb_w1_bound                     ; loop bound, hoisted gw/merge, cmp cannot take two memory operands, so the bound is staged in eax
     cmp vcb_w1, eax                           ; w1 < gw/merge?
     jl .vcb_w1_loop                           ; next w1
.vcb_h1_next:
     inc dword vcb_h1                          ; h1++
     mov eax, vcb_h1_bound                     ; loop bound, hoisted gh/merge, staged for the same reason
     cmp vcb_h1, eax                           ; h1 < gh/merge?
     jl .vcb_h1_loop                           ; next h1
     jmp .vcb_patchify_done                    ; Qwen3-VL patchify done
.vcb_patchify_simple:
     lea r10d, [r14*VCB_DWORD_SCALE]           ; r10 = img_w*4 = src advance per ph row, loop-constant; 8 <= ps <= 16 with a non-negative ps*4-32 tail offset is the import's contract: the two-ymm row copy needs 32 <= ps*4 <= 64
     mov r11d, dword [rel g_vit_cfg + VCFG_PATCH_SIZE_F4] ; r11 = ps*4 = dst advance per ph row, loop-constant; precomputed field, low 32 bits exact
     lea r8d, [r11 - VIT_YMM_BYTES]            ; r8 = ps*4 - 32 = second vmovups offset, runtime tail, one address-generation
     mov rdi, r12                              ; dst cursor, seeded ONCE per encode: after each patch's 3 channels it has advanced exactly patch_dim*4, the same amount r12 advances at the w-loop tail, so it re-enters every patch start equal to r12; the ph loop's 'add rdi, r11' x ps also carries it channel-to-channel: rdi_start,c+1 = rdi_start,c + ps^2*4
     mov dword vcb_h1, 0                       ; h = 0
.vcb_ps_h_loop:
     mov dword vcb_w1, 0                       ; w = 0
.vcb_ps_w_loop:
; y_base = h * patch_size
     mov eax, vcb_h1                           ; h
     imul eax, dword [rel g_vit_cfg + VCFG_PATCH_SIZE] ; h * patch_size
     mov r9d, eax                              ; r9d = y_base
; x_base = w * patch_size
     mov eax, vcb_w1                           ; w
     imul eax, dword [rel g_vit_cfg + VCFG_PATCH_SIZE] ; w * patch_size
     mov ebx, eax                              ; ebx = x_base, non-volatile across inner
     mov eax, r9d                              ; y_base
     imul eax, r14d                            ; y_base * img_w, row stride
     add eax, ebx                              ; + x_base
     lea rbx, [r13 + rax*VCB_DWORD_SCALE]      ; rbx = running src base for c = 0: * sizeof,float + res base, add eax above zero-extended the pixel offset; x_base is dead from here (pixel-shuffle path)
     xor ecx, ecx                              ; c = 0
.vcb_ps_c_loop:
     mov rsi, rbx                              ; src row ph=0 cursor for this channel
     add rbx, r15                              ; advance to the next channel's src base
     mov edx, dword [rel g_vit_cfg + VCFG_PATCH_SIZE] ; edx = remaining patch rows, countdown; 8 <= ps <= 16 is the import's contract for SigLIP models
.vcb_ps_ph_loop:
     vmovups ymm0, [rsi]                       ; bytes 0-31
     vmovups ymm1, [rsi + r8]                  ; last 32 bytes of the same patch row, independent load
     vmovups [rdi], ymm0                       ; store first block
     vmovups [rdi + r8], ymm1                  ; store last block, may overlap first for ps=8
     add rsi, r10                              ; src += img_w*4, next source row
     add rdi, r11                              ; dst += ps*4, next patch row
     dec edx                                   ; one patch row done
     jnz .vcb_ps_ph_loop                       ; next ph, last iteration falls through to the exit
.vcb_ps_c_next:
     inc ecx                                   ; c = next color channel
     cmp ecx, VIT_RAW_RGB_CHANS                ; c < 3 channels?
     jl .vcb_ps_c_loop                         ; next c
; advance output ptr by patch_dim*4
     add r12, qword [rel g_vit_cfg + VCFG_PATCH_DIM_F4] ; patches pointer += patch_dim*4 bytes, next patch
; w++
     inc dword vcb_w1                          ; w = next patch column
     mov eax, vcb_w1                           ; reload w for grid-width compare, cmp cannot take two memory operands, so w is staged in eax
     cmp eax, vcb_gw                           ; w < gw?
     jl .vcb_ps_w_loop                         ; next w
; h++
     inc dword vcb_h1                          ; h = next patch row
     mov eax, vcb_h1                           ; reload h for grid-height compare, staged for the same reason
     cmp eax, vcb_gh                           ; h < gh?
     jl .vcb_ps_h_loop                         ; next h
.vcb_patchify_done:
     cmp qword [rel g_vit_cfg + VCFG_PROJ_IN_DIM], 0 ; separate projector present?
     jne .vcb_alloc_proj                       ; yes, SigLIP: alloc for hidden dim
; Qwen3-VL: out_main = alloc,n_merged * out_hidden * 4
     mov ecx, dword [rel g_vit_cfg + VCFG_OUT_HIDDEN] ; out_hidden, 2560 Qwen3-VL
     jmp .vcb_alloc_common                     ; bytes-per-row in ecx, alloc out_main
.vcb_alloc_proj:
; SigLIP: out_main = alloc,n * hidden * 4, holds ViT post_ln output
     mov ecx, dword [rel g_vit_cfg + VCFG_HIDDEN] ; hidden, 1152 SigLIP
.vcb_alloc_common:
     shl ecx, 2                                ; * sizeof,float = bytes per row
     mov eax, vcb_nmerged                      ; token count for out_main
     imul rcx, rax                             ; total bytes = per_row * count
     call mdb_alloc_mem                        ; allocate buffer, rax = ptr
     mov vcb_out_main, rax                     ; save out_main buffer pointer
     cmp qword [rel g_vit_cfg + VCFG_N_DEEPSTACK], 0 ; model has DeepStack taps?
     jne .vcb_ds_alloc_real                    ; yes, allocate the 3 ds buffers
     xor eax, eax                              ; out_main already saved to vcb_out_main, no live value in RAX
     mov qword vcb_ds0_buf, rax                ; no DeepStack: NULL ds buffers, frame slots must not hold garbage
     mov qword vcb_ds1_buf, rax                ; NULL ds1
     mov qword vcb_ds2_buf, rax                ; NULL ds2
     jmp .vcb_ds_alloc_done                    ; skip the 3 allocations entirely
.vcb_ds_alloc_real:
; ds0 = alloc,n_merged * out_hidden * 4
     mov ecx, dword [rel g_vit_cfg + VCFG_OUT_HIDDEN] ; out_hidden
     shl ecx, 2                                ; * sizeof,float = bytes per row
     mov eax, vcb_nmerged                      ; token count for ds0
     imul rcx, rax                             ; total bytes = per_row * count
     call mdb_alloc_mem                        ; allocate buffer, rax = ptr
     mov vcb_ds0_buf, rax                      ; save ds0 buffer pointer
; ds1 = alloc,n_merged * out_hidden * 4
     mov ecx, dword [rel g_vit_cfg + VCFG_OUT_HIDDEN] ; out_hidden
     shl ecx, 2                                ; * sizeof,float = bytes per row
     mov eax, vcb_nmerged                      ; token count for ds1
     imul rcx, rax                             ; total bytes = per_row * count
     call mdb_alloc_mem                        ; allocate buffer, rax = ptr
     mov vcb_ds1_buf, rax                      ; save ds1 buffer pointer
; ds2 = alloc,n_merged * out_hidden * 4
     mov ecx, dword [rel g_vit_cfg + VCFG_OUT_HIDDEN] ; out_hidden
     shl ecx, 2                                ; * sizeof,float = bytes per row
     mov eax, vcb_nmerged                      ; token count for ds2
     imul rcx, rax                             ; total bytes = per_row * count
     call mdb_alloc_mem                        ; allocate buffer, rax = ptr
     mov vcb_ds2_buf, rax                      ; save ds2 buffer pointer
.vcb_ds_alloc_done:
     sub rsp, VIT_CALL8_FRAME                  ; 8-arg call frame: 32 shadow + 4 stack args
     mov rcx, vcb_patches                      ; arg1: pixels = patches
     mov edx, vcb_gh                           ; arg2: grid_h
     mov r8d, vcb_gw                           ; arg3: grid_w
     lea r9, vcb_weights                       ; arg4: weight struct address on stack
     mov rax, vcb_out_main                     ; arg5 value = out_main
     mov [rsp + VIT_CALL_ARG5], rax            ; arg5: out_main
     mov rax, vcb_ds0_buf                      ; arg6 value = out_ds0
     mov [rsp + VIT_CALL_ARG6], rax            ; arg6: out_ds0
     mov rax, vcb_ds1_buf                      ; arg7 value = out_ds1
     mov [rsp + VIT_CALL_ARG7], rax            ; arg7: out_ds1
     mov rax, vcb_ds2_buf                      ; arg8 value = out_ds2
     mov [rsp + VIT_CALL_ARG8], rax            ; arg8: out_ds2, last stack arg, fills frame exactly
     call v7s_vit_forward                      ; run ViT encoder on merged patches
     add rsp, VIT_CALL8_FRAME                  ; restore stack after 8-arg call
     mov eax, dword [rel g_vit_cfg + VCFG_PROJ_TYPE] ; load projector type, 0/1/2
     test eax, eax                             ; type == 0, Qwen3-VL, the primary model ?
     jz .vcb_no_projector                      ; yes, 1 taken branch, merger already projected
     cmp eax, 1                                ; MLP projector?
     jne .vcb_proj_pixel_shuffle               ; not 1: type 2 is the SmolVLM/Idefics3 pixel-shuffle path; type 1 falls into the MLP
.vcb_proj_mlp:
; === MLP Projector, SigLIP/LLaVA-OV : FC1 -> GELU -> FC2 ===
     lea rcx, [rel sz_vit_proj_key]            ; key = "__sys_vit_projector"
     xor edx, edx                              ; no out_len
     call ht_find_or_fatal                     ; rax = blob ptr, FATALs with the key name on miss
     mov vcb_proj_blob, rax                    ; save blob base
     TRACE_REG "vcb: proj_blob", rax           ; trace: projector blob base address, reads rax before any overwrite
; Allocate proj_inter_buf = n * proj_inter * 4
     mov ecx, dword [rel g_vit_cfg + VCFG_PROJ_INTER] ; proj_inter, 896
     shl ecx, 2                                ; * sizeof,float
     mov eax, vcb_nmerged                      ; n tokens
     imul rcx, rax                             ; total bytes
     call mdb_alloc_mem                        ; allocate
     mov vcb_proj_inter_buf, rax               ; save FC1 intermediate buffer
; Allocate proj_out_buf = n * proj_out * 4
     mov ecx, dword [rel g_vit_cfg + VCFG_PROJ_OUT_DIM] ; proj_out, 896
     shl ecx, 2                                ; * sizeof,float
     mov eax, vcb_nmerged                      ; n tokens
     imul rcx, rax                             ; total bytes
     call mdb_alloc_mem                        ; allocate
     mov vcb_proj_out_buf, rax                 ; save FC2 output buffer
     API_CALL v7s_set_spin, 1                  ; workers commit to spinning for projector FC1+FC2, before r8/r9 load
     mov rax, vcb_proj_blob                    ; blob base
     add rax, VIT_PROJ_HDR_BYTES               ; skip the projector header, fc1_w starts here
     mov r8, rax                               ; fc1_w base
; advance past fc1_w: proj_inter * proj_in * 4
     mov rcx, qword [rel g_vit_cfg + VCFG_PROJ_INTER] ; fc1_w cols = proj_inter
     imul rcx, qword [rel g_vit_cfg + VCFG_PROJ_IN_DIM] ; fc1_w elements = proj_inter * proj_in
     shl rcx, 2                                ; fc1_w bytes = elements * 4
     mov vcb_fc1_w_bytes, rcx                  ; stash fc1_w bytes, reused by FC2 setup below
     add rax, rcx                              ; rax = fc1_b base
     mov r9, rax                               ; fc1_b base
     sub rsp, VIT_CALL7_FRAME                  ; 7-arg call frame
     mov rcx, vcb_proj_inter_buf               ; out = proj_inter
     mov rdx, vcb_out_main                     ; in = out_main, ViT output
     mov eax, vcb_nmerged                      ; n_rows
     mov dword [rsp + VIT_CALL_ARG5], eax      ; arg5: n_rows
     mov eax, dword [rel g_vit_cfg + VCFG_PROJ_INTER] ; M = proj_inter
     mov dword [rsp + VIT_CALL_ARG6], eax      ; arg6: M = proj_inter
     mov eax, dword [rel g_vit_cfg + VCFG_PROJ_IN_DIM] ; K = proj_in
     mov dword [rsp + VIT_CALL_ARG7], eax      ; arg7: K = proj_in
     call v7d_vit_linear                       ; proj_inter = fc1_w * out_main + fc1_b
     add rsp, VIT_CALL7_FRAME                  ; tear down 7-arg call frame
; GELU,proj_inter in-place
     sub rsp, VIT_CALL4_FRAME                  ; 4-arg call frame for vit_gelu
     mov rcx, vcb_proj_inter_buf               ; arg1: in/out buffer = proj_inter
     mov edx, vcb_nmerged                      ; n_tokens, straight into the arg register
     imul edx, dword [rel g_vit_cfg + VCFG_PROJ_INTER] ; total elements = n_tokens * proj_inter
     call vit_gelu                             ; GELU in place
     add rsp, VIT_CALL4_FRAME                  ; tear down 4-arg call frame
     mov rax, vcb_proj_blob                    ; blob base
     add rax, VIT_PROJ_HDR_BYTES               ; skip the projector header, fc1_b base below
; advance past fc1_w: proj_inter * proj_in * 4, hoisted at FC1 setup
     add rax, vcb_fc1_w_bytes                  ; rax past fc1_w, m64 add of the precomputed byte count
; advance past fc1_b: proj_inter * 4
     mov rcx, qword [rel g_vit_cfg + VCFG_PROJ_INTER] ; fc1_b elements = proj_inter
     lea rax, [rax + rcx*VCB_DWORD_SCALE]      ; rax = fc2_w base, fc1_b bytes = proj_inter * 4
     mov r8, rax                               ; fc2_w base
; advance past fc2_w: proj_out * proj_inter * 4
     mov rcx, qword [rel g_vit_cfg + VCFG_PROJ_OUT_DIM] ; fc2_w cols = proj_out
     imul rcx, qword [rel g_vit_cfg + VCFG_PROJ_INTER] ; fc2_w elements = proj_out * proj_inter
     lea rax, [rax + rcx*VCB_DWORD_SCALE]      ; rax = fc2_b base, fc2_w bytes = elements * 4
     mov r9, rax                               ; fc2_b base
     sub rsp, VIT_CALL7_FRAME                  ; 7-arg call frame
     mov rcx, vcb_proj_out_buf                 ; out = proj_out
     mov rdx, vcb_proj_inter_buf               ; in = proj_inter, GELU'd
     mov eax, vcb_nmerged                      ; n_rows
     mov dword [rsp + VIT_CALL_ARG5], eax      ; arg5: n_rows
     mov eax, dword [rel g_vit_cfg + VCFG_PROJ_OUT_DIM] ; M = proj_out
     mov dword [rsp + VIT_CALL_ARG6], eax      ; arg6: M = proj_out
     mov eax, dword [rel g_vit_cfg + VCFG_PROJ_INTER] ; K = proj_inter
     mov dword [rsp + VIT_CALL_ARG7], eax      ; arg7: K = proj_inter
     call v7d_vit_linear                       ; proj_out = fc2_w * proj_inter + fc2_b
     add rsp, VIT_CALL7_FRAME                  ; tear down 7-arg call frame
; Free intermediate buffer; out_main is replaced by proj_out_buf
     mov rcx, vcb_proj_inter_buf               ; free GELU'd intermediate
     call mdb_free_mem                         ; release FC1 output buffer
     mov rcx, vcb_out_main                     ; free old ViT-output buffer
     call mdb_free_mem                         ; release pre-projector output
     mov rax, vcb_proj_out_buf                 ; proj_out becomes the embedding
     mov vcb_out_main, rax                     ; repoint out_main to projected output
     jmp .vcb_no_projector                     ; MLP path done
.vcb_proj_pixel_shuffle:
; === Pixel-shuffle Projector, SmolVLM/Idefics3 : unshuffle + Linear ===
; Load projector weights from DB, same key as MLP: __sys_vit_projector
     lea rcx, [rel sz_vit_proj_key]            ; key = "__sys_vit_projector"
     xor edx, edx                              ; no out_len
     call ht_find_or_fatal                     ; rax = blob ptr, FATALs with the key name on miss
     mov vcb_proj_blob, rax                    ; save blob base for vit_pixel_shuffle
     TRACE_REG "vcb: ps_proj_blob", rax        ; trace: pixel-shuffle projector blob
; Allocate output buffer = n_tokens * proj_out_dim * 4
     mov ecx, dword [rel g_vit_cfg + VCFG_PROJ_OUT_DIM] ; text_hidden, 576 for SmolVLM-256M
     shl ecx, 2                                ; * sizeof,float
     mov eax, dword [rel g_vit_cfg + VCFG_PS_N_TOKENS] ; n_tokens, 64
     imul rcx, rax                             ; total bytes
     call mdb_alloc_mem                        ; allocate
     mov vcb_proj_out_buf, rax                 ; save projected output buffer
; Call vit_pixel_shuffle,out, vit_out, proj_blob
     sub rsp, VIT_CALL4_FRAME                  ; 4-arg call frame, 32 shadow
     mov rcx, rax                              ; new projection buffer, still in RAX
     mov rdx, vcb_out_main                     ; arg2: in = ViT post-LN output
     mov r8, vcb_proj_blob                     ; arg3: proj_blob, reserved hdr + W
     call vit_pixel_shuffle                    ; pixel-shuffle + Linear, kernel handles scratch alloc
     add rsp, VIT_CALL4_FRAME                  ; restore stack after 4-arg call
; Replace out_main with projected output, update token count
     mov rcx, vcb_out_main                     ; free old ViT-output buffer, pre-shuffle
     call mdb_free_mem                         ; release
     mov rax, vcb_proj_out_buf                 ; proj_out becomes the embedding
     mov vcb_out_main, rax                     ; repoint out_main to pixel-shuffle output
     mov eax, dword [rel g_vit_cfg + VCFG_PS_N_TOKENS] ; n_tokens, 64 for SmolVLM-256M
     mov vcb_nmerged, eax                      ; update token count for downstream
.vcb_no_projector:
; --- Step 8: Set output pointers ---
     mov rdx, vcb_out_emb                      ; &out_emb
     mov rax, vcb_out_main                     ; main embedding buffer
     mov [rdx], rax                            ; *out_emb = out_main
     mov rdx, vcb_out_cnt                      ; &out_count
     mov eax, vcb_nmerged                      ; n_merged tokens
     mov [rdx], eax                            ; *out_count = n_merged
     mov rdx, vcb_out_ds0                      ; &out_ds0
     mov rax, vcb_ds0_buf                      ; ds0 buffer pointer
     mov [rdx], rax                            ; *out_ds0 = ds0
     mov rdx, vcb_out_ds1                      ; &out_ds1
     mov rax, vcb_ds1_buf                      ; ds1 buffer pointer
     mov [rdx], rax                            ; *out_ds1 = ds1
     mov rdx, vcb_out_ds2                      ; &out_ds2
     mov rax, vcb_ds2_buf                      ; ds2 buffer pointer
     mov [rdx], rax                            ; *out_ds2 = ds2
     TRACE_STR "vit_cb: encode SUCCESS"        ; diagnostic: callback reached success exit
     mov eax, vcb_nmerged                      ; load merged vision token count for tracing
     TRACE_REG "vit_cb: n_merged", rax         ; print token count delivered to chat_image
     mov rcx, vcb_res                          ; free normalized planar image buffer
     call mdb_free_mem                         ; release res buffer
     mov rcx, vcb_patches                      ; free preprocessed patch features
     call mdb_free_mem                         ; release patches buffer
     xor eax, eax                              ; return code = 0
     jmp .vcb_done                             ; common exit path
%include "vit_anyres.inc"
%unmacro VCB_LAYOUT_WEIGHTS 0                  ; both uses expanded, file-local macro leaves scope
%unmacro VCB_RGB8_UNPACK 0                     ; both uses expanded, file-local macro leaves scope
.vcb_done:
     vzeroupper                                ; clean AVX state
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore non-volatiles, return
%undef VIT_CALL7_FRAME
%undef VIT_CALL8_FRAME
%undef VIT_CALL6_FRAME
%undef VIT_CALL4_FRAME
%undef VIT_CALL_ARG5
%undef VIT_CALL_ARG6
%undef VIT_CALL_ARG7
%undef VIT_CALL_ARG8
%undef VIT_YMM_BYTES
%undef VIT_RGB_VEC_BYTES
%undef VIT_RGB_LO_HALF
%undef vcb_raw
%undef vcb_raw_raw
%undef vcb_img_w
%undef vcb_img_w_raw
%undef vcb_img_h
%undef vcb_img_h_raw
%undef vcb_out_emb
%undef vcb_out_emb_raw
%undef vcb_out_cnt
%undef vcb_out_cnt_raw
%undef vcb_out_ds0
%undef vcb_out_ds0_raw
%undef vcb_out_ds1
%undef vcb_out_ds1_raw
%undef vcb_out_ds2
%undef vcb_out_ds2_raw
%undef vcb_blob
%undef vcb_blob_raw
%undef vcb_res
%undef vcb_res_raw
%undef vcb_patches
%undef vcb_patches_raw
%undef vcb_out_main
%undef vcb_out_main_raw
%undef vcb_ds0_buf
%undef vcb_ds0_buf_raw
%undef vcb_ds1_buf
%undef vcb_ds1_buf_raw
%undef vcb_ds2_buf
%undef vcb_ds2_buf_raw
%undef vcb_gh
%undef vcb_gh_raw
%undef vcb_gw
%undef vcb_gw_raw
%undef vcb_n
%undef vcb_n_raw
%undef vcb_nmerged
%undef vcb_nmerged_raw
%undef vcb_total_px
%undef vcb_total_px_raw
%undef vcb_h1
%undef vcb_h1_raw
%undef vcb_w1
%undef vcb_w1_raw
%undef vcb_w1_bound
%undef vcb_w1_bound_raw
%undef vcb_h1_bound
%undef vcb_h1_bound_raw
%undef vcb_mh
%undef vcb_mh_raw
%undef vcb_mw
%undef vcb_mw_raw
%undef vcb_fc1_w_bytes
%undef vcb_fc1_w_bytes_raw
%undef vcb_proj_blob
%undef vcb_proj_blob_raw
%undef vcb_proj_inter_buf
%undef vcb_proj_inter_buf_raw
%undef vcb_proj_out_buf
%undef vcb_proj_out_buf_raw
%undef vcb_grid_h_orig
%undef vcb_grid_h_orig_raw
%undef vcb_ar_fc1_w
%undef vcb_ar_fc1_w_raw
%undef vcb_ar_fc1_b
%undef vcb_ar_fc1_b_raw
%undef vcb_ar_fc2_w
%undef vcb_ar_fc2_w_raw
%undef vcb_ar_fc2_b
%undef vcb_ar_fc2_b_raw
%undef vcb_img_sz
%undef vcb_img_sz_raw
%undef vcb_ps2_f4
%undef vcb_ps2_f4_raw
%undef vcb_weights
%undef vcb_weights_raw
