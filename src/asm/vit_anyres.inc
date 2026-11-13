; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define ANYRES_RGB_CH 3                        ; RGB channels per pixel
%define ANYRES_N_PTS 36                        ; number of pinpoints, 6x6 grid
%define ANYRES_DW3 2                           ; LEA scale for *3, reg + reg*2
%define AR_DWORD_SCALE 4                       ; SIB scale: one float / one dword element
%define AR_PIN_W_OFF 4                         ; byte offset of width in, h,w pinpoint pair
%define AR_PIN_PAIR_BYTES 8                    ; bytes per, h,w pinpoint pair
%define RGB_R_OFF 0                            ; byte offset of R channel in packed RGB pixel
%define RGB_G_OFF 1                            ; byte offset of G channel in packed RGB pixel
%define RGB_B_OFF 2                            ; byte offset of B channel in packed RGB pixel
%define AR_FLOAT_BYTES 4                       ; sizeof,float, anyres-local prefix: bare FLOAT_BYTES is the exact silent-redefinition hazard the %undef preamble below describes
%define AR_FLOAT_SCALE 4                       ; SIB scale for float-element addressing, anyres-local prefix
%define AR_FLOAT_SCALE_2X 8                    ; 2 planes in floats, G/B-plane store offsets
%define AR_YMM_TAIL 32                         ; one YMM store width: the two-vmovups patch copy's tail offset is, ps*4 - AR_YMM_TAIL
%define VITP_HDR_BYTES 8                       ; bytes per .vit file header slot, qword
%define MAGIC_DWORD_BYTES 4                    ; bytes per .vit header magic dword
%define AR_RGB_VEC_BYTES 24                    ; 8 pixels x 3 bytes, one vector-iteration source span
%define AR_RGB_HI_OFF 16                       ; byte offset of the high source half-load, RGB bytes 16..23
%define AR_YMM_BYTES 32                        ; 8 floats, one vector-iteration plane store
%define AR_CALL_FRAME 48                       ; resize call frame: 32 shadow + 16 for stack args 5,6, 16-aligned
%define AR_ARG5_OFF 32                         ; 5th arg, dst_w offset from rsp in the resize call frame
%define AR_ARG6_OFF 40                         ; 6th arg, dst_h offset from rsp in the resize call frame
%define AR_FIT_CEIL_ADJ 1                      ; ceil-div adjustment: numerator + divisor - AR_FIT_CEIL_ADJ before the divide
%define AR_XMM_STASH_BYTES 160                 ; ten 16-byte xmm stash slots below rsp for the mask-holding vector run
%define AR_STASH_OFF0 0                        ; stash slot of xmm6
%define AR_STASH_OFF1 16                       ; stash slot of xmm7
%define AR_STASH_OFF2 32                       ; stash slot of xmm8
%define AR_STASH_OFF3 48                       ; stash slot of xmm9
%define AR_STASH_OFF4 64                       ; stash slot of xmm10
%define AR_STASH_OFF5 80                       ; stash slot of xmm11
%define AR_STASH_OFF6 96                       ; stash slot of xmm12
%define AR_STASH_OFF7 112                      ; stash slot of xmm13
%define AR_STASH_OFF8 128                      ; stash slot of xmm14
%define AR_STASH_OFF9 144                      ; stash slot of xmm15
; AR_FREE_RESET buffer, success-path free + zero-reset of one BSS buffer
; the 3-line triplet is hand-copied 8 times; macro dedup candidate

%macro AR_FREE_RESET 1
     mov rcx, [rel %1]                         ; buffer to free
     call mdb_free_mem                         ; success-path free
     mov qword [rel %1], 0                     ; reset for the next encode
%endmacro
.vcb_anyres:
; --- Step A1: Load ViT weights from DB ---
     lea rcx, [rel sz_vit_w_key]               ; key = "__sys_vit_weights"
     xor edx, edx                              ; no out_len
     call ht_find_or_fatal                     ; rax = blob ptr, FATALs naming the key on miss, the key symbol and the message cannot drift apart
     mov vcb_blob, rax                         ; save blob pointer
.vcb_ar_a2:
; --- Step A2: Lay out vit_weights_t struct, same pointer arithmetic as single-tile ---
     add rax, MAGIC_DWORD_BYTES                ; skip the .vit magic dword, weights blob starts after it
     VCB_LAYOUT_WEIGHTS                        ; lay out all 27 pointers with the original stride sequence, no tail stride here
     mov r12d, vcb_img_w                       ; r12d = img_w
     mov r13d, vcb_img_h                       ; r13d = img_h
     mov ecx, dword [rel g_vit_cfg + VCFG_IMAGE_SIZE] ; tile side, DB-owned, never a compile-time 384
     mov dword vcb_img_sz, ecx                 ; park in the frame slot every later consumer reads
     xor ebx, ebx                              ; max_effective = 0
     mov esi, 0x7FFFFFFF                       ; min_wasted = INT_MAX sentinel, the first candidate always records, wasted = w*h <= 2304*2304 < INT_MAX, so the tie-break at :131 always fires
     xor r14d, r14d                            ; best_h = 0
     xor r15d, r15d                            ; best_w = 0
     xor edi, edi                              ; idx = 0
     lea r11, [rel g_anyres_pinpoints]         ; r11 = 36 x, h,w pairs base
     mov r10d, r12d                            ; r10d = img_w * img_h, hoisted out of the candidate loop, r10 is free inside it
     imul r10d, r13d                           ; 32-bit product, same width as the per-candidate imul it replaces
     vcvtsi2sd xmm1, xmm1, r12d                ; xmm1 =, double img_w, loop-invariant, hoisted, r12d is never rewritten inside the loop
     vcvtsi2sd xmm3, xmm3, r13d                ; xmm3 =, double img_h, loop-invariant, same
.vcb_ar_sbr_loop:
; Load candidate, h, w, the pair index folds into both operands, SIB scale, so no
; address register is needed and rax stays free below
     mov ecx, [r11 + rdi * AR_PIN_PAIR_BYTES]  ; candidate_h
     mov edx, [r11 + rdi * AR_PIN_PAIR_BYTES + AR_PIN_W_OFF] ; candidate_w
     vcvtsi2sd xmm0, xmm0, edx                 ; double candidate_w
     vcvtsi2sd xmm2, xmm2, ecx                 ; double candidate_h
     vunpcklpd xmm4, xmm0, xmm2                ; {candidate_w, candidate_h}
     vunpcklpd xmm5, xmm1, xmm3                ; {img_w, img_h}
     vdivpd xmm4, xmm4, xmm5                   ; both binary64 ratios, one packed divide
     vmovapd xmm0, xmm4                        ; scale_x in the low lane
     vunpckhpd xmm2, xmm4, xmm4                ; scale_y in the low lane
     vminsd xmm0, xmm0, xmm2                   ; scale = min,sx, sy
     vmulsd xmm2, xmm1, xmm0                   ; img_w * scale, xmm1 = hoisted, double img_w
     vmulsd xmm4, xmm3, xmm0                   ; img_h * scale, independent, into spare xmm4
     vcvttsd2si r8d, xmm2                      ; r8d = dw, truncation matches Python int,
     vcvttsd2si r9d, xmm4                      ; r9d = dh
; effective = min,dw * dh, img_w * img_h
     mov eax, r8d                              ; dw
     imul eax, r9d                             ; dw * dh
     cmp eax, r10d                             ; dw*dh vs img_w*img_h, r10d = the hoisted invariant
     cmovg eax, r10d                           ; effective = min,dw*dh, img_w*img_h
; wasted =, w * h - effective
     mov r8d, edx                              ; w
     imul r8d, ecx                             ; w * h
     sub r8d, eax                              ; wasted = w*h - effective
; Compare with best, flags from the single cmp serve both decisions
     cmp eax, ebx                              ; effective > max_effective?
     jg .vcb_ar_sbr_better                     ; yes, new best
     jne .vcb_ar_sbr_next                      ; neither win nor tie, skip, common case: 2 branches
     cmp r8d, esi                              ; tie, wasted < min_wasted?
     jge .vcb_ar_sbr_next                      ; no, skip
.vcb_ar_sbr_better:
     mov ebx, eax                              ; max_effective = effective
     mov esi, r8d                              ; min_wasted = wasted
     mov r14d, ecx                             ; best_h = candidate_h
     mov r15d, edx                             ; best_w = candidate_w
.vcb_ar_sbr_next:
     inc edi                                   ; idx++
     cmp edi, ANYRES_N_PTS                     ; idx < 36?
     jl .vcb_ar_sbr_loop                       ; yes, next candidate
; Save best resolution to BSS
     mov dword [rel g_ar_best_h], r14d         ; save best_h
     mov dword [rel g_ar_best_w], r15d         ; save best_w
; --- Step A4: Compute grid dimensions ---
; grid_h = best_h / image_size, grid_w = best_w / image_size
     mov eax, r14d                             ; best_h
     xor edx, edx                              ; zero upper dividend
     div dword [rel g_vit_cfg + VCFG_IMAGE_SIZE] ; best_h / tile side, 384 for LLaVA-OV
     mov dword [rel g_ar_grid_h], eax          ; grid_h
     mov eax, r15d                             ; best_w
     xor edx, edx                              ; zero upper dividend
     div dword [rel g_vit_cfg + VCFG_IMAGE_SIZE] ; best_w / tile side, 384 for LLaVA-OV
     mov dword [rel g_ar_grid_w], eax          ; grid_w
; n_tiles = grid_h * grid_w + 1
     mov eax, dword [rel g_ar_grid_h]          ; load grid_h
     imul eax, dword [rel g_ar_grid_w]         ; grid_h * grid_w
     inc eax                                   ; + 1, base tile
     mov dword [rel g_ar_n_tiles], eax         ; save n_tiles
; --- Step A5: Allocate all-features buffer ---
     mov ecx, dword [rel g_vit_cfg + VCFG_AR_BASE_TOK] ; base tokens per tile, 729, DB-precomputed
     imul ecx, dword [rel g_vit_cfg + VCFG_PROJ_OUT_DIM] ; 729 * 896
     shl rcx, 2                                ; x4 = bytes per tile, projected features
     imul rcx, rax                             ; 64-bit: n_tiles * bytes_per_tile
     call mdb_alloc_mem                        ; rax = buffer
     mov [rel g_ar_all_features], rax          ; save all-features buffer
     lea rcx, [rel sz_vit_proj_key]            ; key = "__sys_vit_projector"
     xor edx, edx                              ; no out_len
     call ht_find_or_fatal                     ; rax = blob ptr, FATALs naming the key on miss
     add rax, VITP_HDR_BYTES                   ; skip VITP header, 8 bytes
     mov vcb_ar_fc1_w, rax                     ; fc1_w base
     mov rcx, qword [rel g_vit_cfg + VCFG_PROJ_INTER] ; proj_inter
     imul rcx, qword [rel g_vit_cfg + VCFG_PROJ_IN_DIM] ; proj_inter * proj_in
     lea rax, [rax + rcx*AR_DWORD_SCALE]       ; past fc1_w, * 4 = fc1_w bytes
     mov vcb_ar_fc1_b, rax                     ; fc1_b base
     mov rcx, qword [rel g_vit_cfg + VCFG_PROJ_INTER] ; proj_inter
     lea rax, [rax + rcx*AR_DWORD_SCALE]       ; past fc1_b, * 4 = fc1_b bytes
     mov vcb_ar_fc2_w, rax                     ; fc2_w base
     mov rcx, qword [rel g_vit_cfg + VCFG_PROJ_OUT_DIM] ; proj_out
     imul rcx, qword [rel g_vit_cfg + VCFG_PROJ_INTER] ; proj_out * proj_inter
     lea rax, [rax + rcx*AR_DWORD_SCALE]       ; past fc2_w, * 4 = fc2_w bytes
     mov vcb_ar_fc2_b, rax                     ; fc2_b base
     xor eax, eax                              ; tile_idx = 0
     mov dword [rel g_ar_tile_idx], eax        ; save
     mov ecx, dword vcb_img_sz                 ; tile side, DB-owned
     imul ecx, ecx                             ; side^2
     lea ecx, [rcx + rcx * ANYRES_DW3]         ; * 3 channels = tile RGB bytes
     call mdb_alloc_mem                        ; rax = tile RGB buffer
     mov [rel g_ar_tile_rgb], rax              ; save tile RGB buffer
; --- tile_res: 3 * side^2 * 4 bytes, 1769472 at 384 ---
     mov ecx, dword vcb_img_sz                 ; tile side, DB-owned
     imul ecx, ecx                             ; side^2
     shl ecx, 2                                ; * 4 = bytes per channel
     lea ecx, [rcx + rcx * ANYRES_DW3]         ; * 3 = total planar bytes
     call mdb_alloc_mem                        ; rax = planar float buffer
     mov [rel g_ar_tile_res], rax              ; save
; --- tile_patches: base_tok * patch_dim * 4 ---
     mov ecx, dword [rel g_vit_cfg + VCFG_PATCH_DIM_F4] ; patch_dim * 4
     mov eax, dword [rel g_vit_cfg + VCFG_AR_BASE_TOK] ; base tokens per tile, 729, DB-precomputed
     imul rcx, rax                             ; 64-bit: base_tok * patch_dim * 4
     call mdb_alloc_mem                        ; rax = patches buffer
     mov [rel g_ar_tile_patches], rax          ; save
; --- tile_vit_out: base_tok * hidden * 4 ---
     mov ecx, dword [rel g_vit_cfg + VCFG_HIDDEN] ; hidden dim
     shl ecx, 2                                ; * 4 = bytes per row
     mov eax, dword [rel g_vit_cfg + VCFG_AR_BASE_TOK] ; base tokens per tile, 729, DB-precomputed
     imul rcx, rax                             ; 64-bit: base_tok * hidden * 4
     call mdb_alloc_mem                        ; rax = ViT output buffer
     mov [rel g_ar_tile_vit_out], rax          ; save
; --- proj_inter: base_tok * proj_inter * 4 ---
     mov ecx, dword [rel g_vit_cfg + VCFG_PROJ_INTER] ; proj_inter dim
     shl ecx, 2                                ; * 4
     mov eax, dword [rel g_vit_cfg + VCFG_AR_BASE_TOK] ; base tokens per tile, 729, DB-precomputed
     imul rcx, rax                             ; 64-bit size
     call mdb_alloc_mem                        ; rax = proj_inter buffer
     mov [rel g_ar_proj_inter], rax            ; save proj_inter
; DeepStack outputs are not consumed by AnyRes. The forward caller passes NULL
; to suppress all three unused tap pipelines and their scratch allocations.
.vcb_ar_tile_loop:
     cmp dword [rel g_ar_tile_idx], 0          ; tile 0, base ?, the tile index is read only for the zero test here
     jnz .vcb_ar_grid_tile                     ; no, grid tile
; === Base tile: bicubic resize original to 384x384 ===
     sub rsp, AR_CALL_FRAME                    ; 48 bytes: 32 shadow + 16 for stack args
     mov rcx, vcb_raw                          ; arg1 = src RGB, original raw image
     mov edx, vcb_img_w                        ; arg2 = src width
     mov r8d, vcb_img_h                        ; arg3 = src height
     mov r9, [rel g_ar_tile_rgb]               ; arg4 = dst = tile RGB buffer
     mov eax, dword vcb_img_sz                 ; tile side, DB-owned
     mov dword [rsp + AR_ARG5_OFF], eax        ; arg5 = dst_w = side
     mov dword [rsp + AR_ARG6_OFF], eax        ; arg6 = dst_h = side
     call v7s_bicubic_resize                   ; bicubic resize, Catmull-Rom, PIL resample=3
     add rsp, AR_CALL_FRAME                    ; restore stack
     mov r12, [rel g_ar_tile_rgb]              ; base tile RGB starts at the contiguous buffer
     xor r10d, r10d                            ; base rows have no source skip
     jmp .vcb_ar_have_rgb                      ; normalize the base tile
.vcb_ar_grid_tile:
; === Grid tile: resize to best_res, then crop 384x384 ===
; First time: resize original to, best_w, best_h
     cmp qword [rel g_ar_resized_buf], 0       ; already resized?
     jne .vcb_ar_have_resized                  ; yes, reuse
     mov eax, dword [rel g_ar_best_w]          ; best_w, zero-extends to rax
     mov ecx, dword [rel g_ar_best_h]          ; best_h, zero-extends to rcx
     imul rax, rcx                             ; best_w * best_h, 64-bit
     lea rcx, [rax + rax * ANYRES_DW3]         ; * 3 = total bytes
     call mdb_alloc_mem                        ; rax = resized buffer
     mov [rel g_ar_resized_buf], rax           ; save
; Bicubic resize original -> fit block -> centered black pad onto the, best_w, best_h canvas
; HF _resize_for_patching + _pad_for_patching: aspect-PRESERVING fit, never a stretch
; fit dims via the 64-bit cross-multiply, no division for the branch :
; width-limited iff best_w * img_h < best_h * img_w
; height-limited iff best_w * img_h >= best_h * img_w, equality: fit == candidate, pad == 0
     mov r8d, dword [rel g_ar_best_w]          ; r8d = best_w, fit_w for the width-limited branch
     mov r9d, dword [rel g_ar_best_h]          ; r9d = best_h, fit_h for the height-limited branch
     mov eax, vcb_img_w                        ; eax = img_w
     mov ecx, vcb_img_h                        ; ecx = img_h
     mov r10d, r8d                             ; r10d = best_w, zero-extends for the 64-bit imul below
     imul r10, rcx                             ; r10 = best_w * img_h, 64-bit
     mov r11d, r9d                             ; r11d = best_h, zero-extends for the 64-bit imul below
     imul r11, rax                             ; r11 = best_h * img_w, 64-bit
     cmp r10, r11                              ; best_w*img_h < best_h*img_w?
     jae .vcb_ar_fit_height                    ; no -> height-limited, equality: fit == candidate, zero pad
; width-limited: fit_w = best_w, fit_h = ceil_div,img_h * best_w, img_w ; numerator = r10, commuted product
     mov r11, rax                              ; r11 = img_w, divisor park; rax is the div dividend half
     lea rax, [r10 + r11 - AR_FIT_CEIL_ADJ]    ; numerator + divisor - 1, the ceil-div adjustment; 64-bit
     xor edx, edx                              ; zero upper dividend, 64-bit div
     div r11                                   ; rax = fit_h, quotient fits u32 and is <= best_h by the branch condition
     mov r9d, eax                              ; fit_h
     jmp .vcb_ar_fit_have                      ; r8d = fit_w, r9d = fit_h
.vcb_ar_fit_height:
; height-limited: fit_h = best_h, fit_w = ceil_div,img_w * best_h, img_h ; numerator = r11, commuted product
     mov r10, rcx                              ; r10 = img_h, divisor park
     lea rax, [r11 + r10 - AR_FIT_CEIL_ADJ]    ; numerator + divisor - 1, ceil-div adjustment
     xor edx, edx                              ; zero upper dividend, 64-bit div
     div r10                                   ; rax = fit_w, <= best_w by the branch condition
     mov r8d, eax                              ; fit_w
; r9d already = best_h = fit_h
.vcb_ar_fit_have:
; Write into the centered rectangle of the zero-filled RGB canvas.
     sub rsp, AR_CALL_FRAME                    ; shadow space and two stack arguments
     mov [rsp + AR_ARG5_OFF], r8d              ; fit width
     mov [rsp + AR_ARG6_OFF], r9d              ; fit height
     mov edx, [rel g_ar_best_w]                ; canvas width
     mov r10d, edx                             ; preserve canvas width for the row skip
     sub r10d, r8d                             ; pixels outside the fitted row
     mov ecx, r10d                             ; horizontal padding on both sides
     shr ecx, 1                                ; left padding, rounded down
     mov eax, [rel g_ar_best_h]                ; canvas height
     sub eax, r9d                              ; vertical padding on both sides
     shr eax, 1                                ; top padding, rounded down
     imul rax, rdx                             ; top padding in pixels
     add rax, rcx                              ; centered origin in pixels
     lea rax, [rax + rax*ANYRES_DW3]           ; centered origin in bytes
     lea r10d, [r10 + r10*ANYRES_DW3]          ; byte skip between fitted rows
     mov r9, [rel g_ar_resized_buf]            ; canvas base
     add r9, rax                               ; destination at the fitted rectangle
     mov rcx, vcb_raw                          ; original RGB
     mov edx, vcb_img_w                        ; original width
     mov r8d, vcb_img_h                        ; original height
     call v7s_bicubic_resize_strided           ; keep row padding outside the output
     add rsp, AR_CALL_FRAME                    ; release the call frame
.vcb_ar_have_resized:
; Crop 384x384 from resized_buf at, tile_x, tile_y
     mov eax, dword [rel g_ar_tile_idx]        ; tile_idx
     dec eax                                   ; grid_tile_idx = tile_idx - 1
     xor edx, edx                              ; zero upper dividend
     div dword [rel g_ar_grid_w]               ; eax = tile_row, edx = tile_col
     imul eax, dword [rel g_ar_best_w]         ; tile_row * canvas width
     add eax, edx                              ; add tile_col
     imul eax, dword vcb_img_sz                ; origin in pixels
     lea eax, [rax + rax*ANYRES_DW3]           ; origin in RGB bytes
     mov r12, [rel g_ar_resized_buf]           ; resized canvas base
     add r12, rax                              ; source at the tile's first pixel
     mov r10d, [rel g_ar_best_w]               ; canvas row width in pixels
     sub r10d, vcb_img_sz                      ; pixels outside this tile on a canvas row
     lea r10d, [r10 + r10*ANYRES_DW3]          ; source skip after one tile row
.vcb_ar_have_rgb:
     mov rsi, [rel g_ar_tile_res]              ; running R-plane destination
     mov ebx, dword vcb_img_sz                 ; pixels per row
     mov r11d, ebx                             ; rows to normalize
     mov r14d, ebx                             ; tile side
     imul r14d, ebx                            ; pixel count gives the G/B plane offsets
     mov eax, 1                                ; contiguous input can be processed as one row
     test r10d, r10d                           ; does the input have a row skip?
     cmovz ebx, r14d                           ; contiguous vector groups can cross row boundaries
     cmovz r11d, eax                           ; process the contiguous run once
     vbroadcastss ymm4, [rel vit_inv_1275]     ; 1/127.5 broadcast, loop-invariant, volatile reg
     vbroadcastss ymm5, [rel vit_one_float]    ; 1.0 broadcast, loop-invariant, volatile reg
; Save the nonvolatile lanes used by the RGB unpacker.
     sub rsp, AR_XMM_STASH_BYTES               ; six 16-byte slots below rsp, RBP-relative SF slots keep working
     vmovdqu [rsp + AR_STASH_OFF0], xmm6       ; stash xmm6, restored after the loop
     vmovdqu [rsp + AR_STASH_OFF1], xmm7       ; stash xmm7
     vmovdqu [rsp + AR_STASH_OFF2], xmm8       ; stash xmm8
     vmovdqu [rsp + AR_STASH_OFF3], xmm9       ; stash xmm9
     vmovdqu [rsp + AR_STASH_OFF4], xmm10      ; stash xmm10
     vmovdqu [rsp + AR_STASH_OFF5], xmm11      ; stash xmm11
     vmovdqu [rsp + AR_STASH_OFF6], xmm12      ; stash xmm12, G-low intermediate
     vmovdqu [rsp + AR_STASH_OFF7], xmm13      ; stash xmm13, G-high intermediate
     vmovdqu [rsp + AR_STASH_OFF8], xmm14      ; stash xmm14, B-low intermediate
     vmovdqu [rsp + AR_STASH_OFF9], xmm15      ; stash xmm15, B-high intermediate
     vmovdqu xmm6, [rel ar_mask_r_lo]          ; xmm6 = R-low planar mask, invariant over the whole vector run
     vmovdqu xmm7, [rel ar_mask_r_hi]          ; xmm7 = R-high planar mask
     vmovdqu xmm8, [rel ar_mask_g_lo]          ; xmm8 = G-low planar mask
     vmovdqu xmm9, [rel ar_mask_g_hi]          ; xmm9 = G-high planar mask
     vmovdqu xmm10, [rel ar_mask_b_lo]         ; xmm10 = B-low planar mask
     vmovdqu xmm11, [rel ar_mask_b_hi]         ; xmm11 = B-high planar mask
.vcb_ar_norm_row:
     mov eax, ebx                              ; pixels in this row
     and eax, -8                               ; complete groups of eight pixels
     jz .vcb_ar_norm_tail                      ; a narrow row uses only the scalar loop
.vcb_ar_norm_vec_loop:
     vmovdqu xmm0, [r12]                       ; RGB bytes 0..15, raw lo, survives all six shuffles
     vmovq xmm1, [r12 + AR_RGB_HI_OFF]         ; RGB bytes 16..23 only, 8-byte load, zero-extends hi 64 bits; hi masks tap indices 0..7, a 16-byte vmovdqu would read 8 bytes past the tile buffer on the last iteration and can fault on a page boundary
     VCB_RGB8_UNPACK                           ; planar R/G/B floats into ymm2/ymm12/ymm14
     vfmsub132ps ymm2, ymm5, ymm4              ; float R *, 1/127.5 - 1.0 in ONE rounding, FMSUB132: dst = dst*src3 - src2, the same fused spelling the single-tile path uses; mul+sub would round twice
     vfmsub132ps ymm12, ymm5, ymm4             ; float G *, 1/127.5 - 1.0 in ONE rounding, fused, as above
     vfmsub132ps ymm14, ymm5, ymm4             ; float B *, 1/127.5 - 1.0 in ONE rounding, fused, as above
     vmovups [rsi], ymm2                       ; store 8 R floats, plane 0 at cursor
     vmovups [rsi + r14 * AR_FLOAT_SCALE], ymm12 ; store 8 G floats, plane 1 = cursor + 1 stride, indexed, unaligned form: the stride is DB-owned side*side, not provably 32-aligned
     vmovups [rsi + r14 * AR_FLOAT_SCALE_2X], ymm14 ; store 8 B floats, plane 2 = cursor + 2 strides, indexed, unaligned form, as above
     add r12, AR_RGB_VEC_BYTES                 ; src += 24, 8 pixels x 3 bytes
     add rsi, AR_YMM_BYTES                     ; R plane cursor += 32, 8 floats ; G/B advance implicitly via the r14-indexed plane offsets
     sub eax, 8                                ; 8 fewer vector pixels
     jnz .vcb_ar_norm_vec_loop                 ; next 8 pixels
.vcb_ar_norm_tail:
     mov eax, ebx                              ; full row width
     and eax, 7                                ; pixels remaining after the vector groups
     jz .vcb_ar_norm_row_done                  ; no scalar pixels in this row
.vcb_ar_norm_loop:
     movzx ecx, byte [r12 + RGB_R_OFF]         ; R channel byte
     vcvtsi2ss xmm0, xmm0, ecx                 ; R byte to exact float
     vfmsub132ss xmm0, xmm5, xmm4              ; R * scale - 1 in one rounding
     vmovss [rsi], xmm0                        ; R plane
     movzx ecx, byte [r12 + RGB_G_OFF]         ; G channel byte
     vcvtsi2ss xmm0, xmm0, ecx                 ; G byte to exact float
     vfmsub132ss xmm0, xmm5, xmm4              ; G * scale - 1 in one rounding
     vmovss [rsi + r14*AR_FLOAT_SCALE], xmm0   ; G plane
     movzx ecx, byte [r12 + RGB_B_OFF]         ; B channel byte
     vcvtsi2ss xmm0, xmm0, ecx                 ; B byte to exact float
     vfmsub132ss xmm0, xmm5, xmm4              ; B * scale - 1 in one rounding
     vmovss [rsi + r14*AR_FLOAT_SCALE_2X], xmm0 ; B plane
     add r12, ANYRES_RGB_CH                    ; next source pixel
     add rsi, AR_FLOAT_BYTES                   ; next destination pixel in each plane
     dec eax                                   ; one fewer scalar pixel
     jnz .vcb_ar_norm_loop                     ; finish the row
.vcb_ar_norm_row_done:
     add r12, r10                              ; skip canvas pixels outside this tile
     dec r11d                                  ; one fewer row in the tile
     jnz .vcb_ar_norm_row                      ; process the next tile row
     vmovdqu xmm6, [rsp + AR_STASH_OFF0]       ; restore the vector-run-stashed non-volatile XMMs
     vmovdqu xmm7, [rsp + AR_STASH_OFF1]       ; restore xmm7
     vmovdqu xmm8, [rsp + AR_STASH_OFF2]       ; restore xmm8
     vmovdqu xmm9, [rsp + AR_STASH_OFF3]       ; restore xmm9
     vmovdqu xmm10, [rsp + AR_STASH_OFF4]      ; restore xmm10
     vmovdqu xmm11, [rsp + AR_STASH_OFF5]      ; restore xmm11
     vmovdqu xmm12, [rsp + AR_STASH_OFF6]      ; restore xmm12
     vmovdqu xmm13, [rsp + AR_STASH_OFF7]      ; restore xmm13
     vmovdqu xmm14, [rsp + AR_STASH_OFF8]      ; restore xmm14
     vmovdqu xmm15, [rsp + AR_STASH_OFF9]      ; restore xmm15
     add rsp, AR_XMM_STASH_BYTES               ; release the ten stash slots
.vcb_ar_norm_done:
; --- A6d: Patchify, SigLIP row-major, merge=1; patches buffer hoisted out of the tile loop ---
; Patchify: h=0..26, w=0..26, c=0..2, ph=0..13, additive cursors, no per-iteration imuls
     mov r12, [rel g_ar_tile_patches]          ; running patches output, BSS reload, buffer hoisted
     mov r14d, dword vcb_img_sz                ; img_w = tile side, DB-owned, row stride; 384 at SigLIP
     mov r15d, r14d                            ; tile side
     imul r15d, r15d                           ; total_px = side^2, plane stride; 147456 at 384
     shl r15d, 2                               ; total_px*4 = BYTES to the next colour plane, 32-bit shl zero-extends, so r15 is a byte stride
     lea r8d, [r14*AR_DWORD_SCALE]             ; r8d = img_w*4 = src advance per ph row, 1536, loop-constant
     mov r9d, dword [rel g_vit_cfg + VCFG_PATCH_SIZE] ; patch_size, hoisted, loop-invariant; ps >= 8 with a non-negative ps*4-32 tail offset is the import's contract
     lea r10d, [r9*AR_DWORD_SCALE]             ; r10d = patch_size*4 = dst advance per ph row, 56, loop-constant
     mov eax, r9d                              ; patch_size
     imul eax, eax                             ; ps^2
     shl eax, 2                                ; ps^2 * 4 = dst channel step, bytes
     mov r11d, eax                             ; r11d = dst channel step, register, r11 is unused across the patchify block
     mov dword vcb_h1, 0                       ; h = 0
.vcb_ar_pf_h:
; per-h row term: y_base * img_w, constant for all 27*27*3 iterations of this h
     mov eax, vcb_h1                           ; h
     mov r14d, dword vcb_img_sz                ; img_w reload, r14d ends each w pass as the channel accumulator
     imul eax, r9d                             ; y_base = h * patch_size
     imul eax, r14d                            ; y_base * img_w, w/c-invariant row term
     shl eax, 2                                ; ... expressed in bytes
     mov ebx, eax                              ; ebx = src byte offset of patch, h, w=0, c=0, the WHOLE src offset is carried from here
     mov dword vcb_w1, 0                       ; w = 0
.vcb_ar_pf_w:
; per-w channel accumulator, advanced at the c tail; reset for each patch
     xor r14d, r14d                            ; dst channel offset accumulator = 0, bytes, in r14d
     xor ecx, ecx                              ; c = 0
.vcb_ar_pf_c:
; src cursor for this, patch, channel : ebx already carries
; y_base*img_w + x_base + c*total_px * 4 as a pure byte offset, accumulated, never recomputed
     lea rsi, [r13 + rbx]                      ; rsi = src row ph=0, advanced additively below
; dst cursor for this channel: patches + patch_base + c*ps^2*4, r12 = patch base, cdst in bytes
     lea rdi, [r12 + r14]                      ; rdi = dst row ph=0 = patch base + c*ps^2*4, r14d zero-extends into r14
     mov edx, r9d                              ; ph countdown = patch_size, ps >= 1 gated at init
.vcb_ar_pf_ph:
     vmovups ymm0, [rsi]                       ; bytes 0..31
     vmovups ymm1, [rsi + r10 - AR_YMM_TAIL]   ; bytes ps*4-32..ps*4-1, independent demand load
     vmovups [rdi], ymm0                       ; store to dst
     vmovups [rdi + r10 - AR_YMM_TAIL], ymm1   ; store to dst
     add rsi, r8                               ; src += img_w*4, next source row, replaces the offset recompute chain
     add rdi, r10                              ; dst += patch_size*4, next patch row
     dec edx                                   ; one ph row done, dec sets ZF at zero
     jnz .vcb_ar_pf_ph                         ; next ph
; channel complete: advance the accumulators, once per CHANNEL, placed before inc ecx,
; not at the c-loop head; a per-patch placement would leave channels 1/2 at offset 0
     add ebx, r15d                             ; src: c-offset += total_px*4, next colour plane
     add r14d, r11d                            ; dst offset += ps^2 * 4, bytes
     inc ecx                                   ; c++
     cmp ecx, ANYRES_RGB_CH                    ; c < 3?
     jl .vcb_ar_pf_c                           ; next c
; Advance to next patch
     add r12, qword [rel g_vit_cfg + VCFG_PATCH_DIM_F4] ; output += patch_dim * 4
; the c loop advanced ebx across three colour planes, drop them, then step one patch right
     lea eax, [r15 + r15*ANYRES_DW3]           ; 3 colour-plane strides in one op
     sub ebx, eax                              ; drop all three planes, next w column
     add ebx, r10d                             ; x_base += patch_size*4, next w column
     inc dword vcb_w1                          ; w++
     mov eax, dword [rel g_vit_cfg + VCFG_AR_GRID_SIDE] ; patch grid side, 27, DB-precomputed; eax scratch at both tails
     cmp vcb_w1, eax                           ; w < grid side?
     jl .vcb_ar_pf_w                           ; next w
     inc dword vcb_h1                          ; h++
     cmp vcb_h1, eax                           ; h < grid side?, eax still holds the grid side from the w tail above, nothing between overwrites it
     jl .vcb_ar_pf_h                           ; next h row, THE back-edge of the patchify h-loop: without it only patch row h=0 is written and rows 1..26 of every tile stay zero vectors
; --- A6e: v7s_vit_forward, out buffer hoisted out of the tile loop; ds slots hoisted per encode ---
; v7s_vit_forward,patches, grid_side, grid_side, &weights, out_main, ds0, ds1, ds2
     sub rsp, VIT_CALL8_FRAME                  ; 8-arg call frame
     mov rcx, [rel g_ar_tile_patches]          ; arg1: patches
     mov edx, dword [rel g_vit_cfg + VCFG_AR_GRID_SIDE] ; arg2: grid_h = patch grid side, 27, DB-precomputed
     mov r8d, edx                              ; arg3: grid_w = same patch grid side, zero-extends like the dword load it replaces
     lea r9, vcb_weights                       ; arg4: weight struct on stack
     mov rax, [rel g_ar_tile_vit_out]          ; arg5: out_main
     mov [rsp + VIT_CALL_ARG5], rax            ; store arg5
     xor eax, eax                              ; all DeepStack outputs are discarded on AnyRes
     mov [rsp + VIT_CALL_ARG6], rax            ; arg6: no ds0
     mov [rsp + VIT_CALL_ARG7], rax            ; arg7: no ds1
     mov [rsp + VIT_CALL_ARG8], rax            ; arg8: no ds2
     call v7s_vit_forward                      ; run ViT 26-layer forward
     add rsp, VIT_CALL8_FRAME                  ; restore stack
     API_CALL v7s_set_spin, 1                  ; workers spin for projector, before r8/r9 load
; FC1 weight pointers, hoisted frame cache, set before the tile loop
     mov r8, vcb_ar_fc1_w                      ; fc1_w base, frame cache
     mov r9, vcb_ar_fc1_b                      ; fc1_b base, frame cache
; FC1: proj_inter = linear,vit_out, fc1_w, fc1_b, base_tok, proj_inter, proj_in
     sub rsp, VIT_CALL7_FRAME                  ; 7-arg call frame
     mov rcx, [rel g_ar_proj_inter]            ; arg1: out = proj_inter
     mov rdx, [rel g_ar_tile_vit_out]          ; arg2: in = ViT output
     mov eax, dword [rel g_vit_cfg + VCFG_AR_BASE_TOK] ; n_rows = base tokens per tile, 729, DB-precomputed
     mov dword [rsp + VIT_CALL_ARG5], eax      ; arg5: n_rows
     mov eax, dword [rel g_vit_cfg + VCFG_PROJ_INTER] ; M = proj_inter
     mov dword [rsp + VIT_CALL_ARG6], eax      ; arg6: M
     mov eax, dword [rel g_vit_cfg + VCFG_PROJ_IN_DIM] ; K = proj_in
     mov dword [rsp + VIT_CALL_ARG7], eax      ; arg7: K
     call v7d_vit_linear                       ; proj_inter = fc1_w * vit_out + fc1_b
     add rsp, VIT_CALL7_FRAME                  ; restore stack
; GELU in-place
     mov rcx, [rel g_ar_proj_inter]            ; in/out = proj_inter
     mov eax, dword [rel g_vit_cfg + VCFG_AR_BASE_TOK] ; base tokens per tile, 729, DB-precomputed
     imul eax, dword [rel g_vit_cfg + VCFG_PROJ_INTER] ; base_tok * proj_inter
     mov edx, eax                              ; element count
     call vit_gelu                             ; GELU activation in-place
; FC2 writes its tile directly into the feature array.
     mov r8, vcb_ar_fc2_w                      ; fc2_w base, frame cache
     mov r9, vcb_ar_fc2_b                      ; fc2_b base, frame cache
; FC2: proj_out = linear,proj_inter, fc2_w, fc2_b, base_tok, proj_out, proj_inter
     sub rsp, VIT_CALL7_FRAME                  ; 7-arg call frame
     mov eax, [rel g_ar_tile_idx]              ; current tile index
     mov ecx, [rel g_vit_cfg + VCFG_AR_BASE_TOK] ; rows per tile
     imul ecx, [rel g_vit_cfg + VCFG_PROJ_OUT_DIM] ; floats per tile
     imul rax, rcx                             ; offset in floats to this tile
     mov rcx, [rel g_ar_all_features]          ; owning feature array
     lea rcx, [rcx + rax*AR_FLOAT_SCALE]       ; write FC2 directly to the final tile slice
     mov rdx, [rel g_ar_proj_inter]            ; arg2: in = proj_inter, GELU'd
     mov eax, dword [rel g_vit_cfg + VCFG_AR_BASE_TOK] ; n_rows = base tokens per tile, 729, DB-precomputed
     mov dword [rsp + VIT_CALL_ARG5], eax      ; arg5: n_rows
     mov eax, dword [rel g_vit_cfg + VCFG_PROJ_OUT_DIM] ; M = proj_out
     mov dword [rsp + VIT_CALL_ARG6], eax      ; arg6: M
     mov eax, dword [rel g_vit_cfg + VCFG_PROJ_INTER] ; K = proj_inter
     mov dword [rsp + VIT_CALL_ARG7], eax      ; arg7: K
     call v7d_vit_linear                       ; proj_out = fc2_w * proj_inter + fc2_b
     add rsp, VIT_CALL7_FRAME                  ; restore stack
; Next tile, reuse the temporary buffers.
     inc dword [rel g_ar_tile_idx]             ; tile_idx++
     mov eax, dword [rel g_ar_tile_idx]        ; reload incremented index
     cmp eax, dword [rel g_ar_n_tiles]         ; tile_idx < n_tiles?
     jb .vcb_ar_tile_loop                      ; next tile, bottom test
.vcb_ar_tiles_done:
; All tiles done; workers stay spinning until chat completion
; Free the per-encode temporary buffers.
     AR_FREE_RESET g_ar_tile_rgb               ; free + zero-reset, macro: mov rcx/call mdb_free_mem/zero-pin
     AR_FREE_RESET g_ar_tile_res               ; free + zero-reset
     AR_FREE_RESET g_ar_tile_patches           ; free + zero-reset
     AR_FREE_RESET g_ar_tile_vit_out           ; free + zero-reset
     AR_FREE_RESET g_ar_proj_inter             ; free + zero-reset
     mov eax, dword [rel g_ar_grid_h]          ; canonical grid_h
     imul eax, dword [rel g_vit_cfg + VCFG_AR_GRID_SIDE] ; feat_h = grid_h * grid side, 27, DB-precomputed
     mov ecx, dword [rel g_ar_grid_w]          ; canonical grid_w
     imul ecx, dword [rel g_vit_cfg + VCFG_AR_GRID_SIDE] ; feat_w = grid_w * grid side, 27, DB-precomputed
; total = base_tok + feat_h *, feat_w + 1
     inc ecx                                   ; feat_w + 1, for newline
     imul eax, ecx                             ; feat_h *, feat_w + 1
     add eax, dword [rel g_vit_cfg + VCFG_AR_BASE_TOK] ; + base tile tokens, 729, DB-precomputed
     mov dword [rel g_ar_final_cnt], eax       ; save total token count
     mov ecx, dword [rel g_vit_cfg + VCFG_PROJ_OUT_DIM] ; proj_out
     shl rcx, 2                                ; x4 = bytes per token row
     imul rcx, rax                             ; 64-bit: total * proj_out * 4, eax still holds the total from just above
     call mdb_alloc_mem                        ; rax = final embedding buffer
     mov [rel g_ar_final_emb], rax             ; save final embedding buffer
; Copy base tile, tile 0 features to output[0..728]
; src = all_features, dst = final_emb, count = base_tok * proj_out * 4
     mov rsi, [rel g_ar_all_features]          ; src = all_features, tile 0 at offset 0
     mov rdi, rax                              ; newly allocated final embedding buffer
     mov ecx, dword [rel g_vit_cfg + VCFG_AR_BASE_TOK] ; base tokens per tile, 729, DB-precomputed
     imul ecx, dword [rel g_vit_cfg + VCFG_PROJ_OUT_DIM] ; base_tok * proj_out
     shl ecx, 2                                ; * 4 = bytes
     rep movsb                                 ; copy base tile features
; Load canonical grid_w, g_ar_grid_w = best_w / VCFG_IMAGE_SIZE, set once at A4
     mov ebx, dword [rel g_ar_grid_w]          ; ebx = grid_w, non-volatile; = best_w / tile side, computed at A4
; Load constant pointers
     mov r12, [rel g_ar_all_features]          ; r12 = all_features base
     mov r13, [rel g_anyres_newline_emb]       ; r13 = newline embedding base
; r15 = proj_out * 4, bytes per feature vector
     mov ecx, dword [rel g_vit_cfg + VCFG_PROJ_OUT_DIM] ; proj_out
     shl ecx, 2                                ; * 4
     mov r15d, ecx                             ; r15 = bytes per feature
; Load canonical grid_h, g_ar_grid_h = best_h / VCFG_IMAGE_SIZE, set once at A4
     mov eax, dword [rel g_ar_grid_h]          ; canonical grid_h, = best_h / tile side, computed at A4
     mov vcb_grid_h_orig, eax                  ; frame-local grid rows, g_ar_n_tiles stays canonical
; Output cursor starts after base tile, base_tok * proj_out * 4 bytes
     mov r14, [rel g_ar_final_emb]             ; r14 = output base
     mov ecx, dword [rel g_vit_cfg + VCFG_AR_BASE_TOK] ; base tokens per tile, 729, DB-precomputed
     imul ecx, dword [rel g_vit_cfg + VCFG_PROJ_OUT_DIM] ; base_tok * proj_out
     shl ecx, 2                                ; * 4
     add r14, rcx                              ; r14 = output past base tile
     xor r8d, r8d                              ; r8d = tile_row = 0
; Loop-invariant products, hoisted out of all three loops, r11/rdx are used nowhere
; else in the arrange phase, and rep movsb overwrites only rcx/rsi/rdi
     mov r11d, r15d                            ; r11d = bytes per feature
     imul r11d, dword [rel g_vit_cfg + VCFG_AR_TC_ADV] ; *, 729-27 = src advance per tile_col, DB-precomputed
     mov edx, r15d                             ; edx = bytes per feature
     imul edx, dword [rel g_vit_cfg + VCFG_AR_GRID_SIDE] ; * 27 = bytes for one output row of 27 features, DB-precomputed
.vcb_ar_arr_tr:
     xor r9d, r9d                              ; r9d = local_row = 0
     mov eax, r8d                              ; tile_row
     imul eax, ebx                             ; * grid_w
     inc eax                                   ; + 1, base tile offset
     imul eax, dword [rel g_vit_cfg + VCFG_AR_BASE_TOK] ; * base_tok, 729, DB-precomputed; feature index, 32-bit < 4B
     mov r10, r15                              ; bytes per feature, 64-bit; r15d was written 32-bit, zero-extended
     imul r10, rax                             ; 64-bit byte offset
.vcb_ar_arr_lr:
     lea rsi, [r12 + r10]                      ; current tile's input-row pointer, offset carried in r10
     mov eax, ebx                              ; eax = grid_w countdown, >= 1: every candidate is recorded on the first pass, so grid_h/grid_w >= 1; survives rep movsb
.vcb_ar_arr_tc:
; rsi already positioned at this tile-row's first feature
     mov rdi, r14                              ; dst = output cursor
     mov ecx, edx                              ; rcx = 27 * bpf, zero-extends
     rep movsb                                 ; copy all 27 contiguous features, advances rsi by 27*bpf
     add r14, rdx                              ; advance output cursor by one row of 27 features, rdx's upper half is zero: edx is written by 32-bit ops only
     add rsi, r11                              ; src: skip to next tile_col's same local_row, 702*bpf more
     dec eax                                   ; one tile_col done, dec sets ZF at zero
     jnz .vcb_ar_arr_tc                        ; next tile_col, bottom test
; fall through = all tile_cols done -> newline embedding copy follows
; Append newline embedding for this output row
     mov rsi, r13                              ; src = newline embedding base
     mov rdi, r14                              ; dst = output cursor
     mov ecx, r15d                             ; bytes = proj_out * 4
     rep movsb                                 ; copy newline embedding
     add r14, r15                              ; advance output cursor
     add r10, rdx                              ; next local row byte offset, no multiply needed
     inc r9d                                   ; local_row++
     cmp r9d, dword [rel g_vit_cfg + VCFG_AR_GRID_SIDE] ; local_row < grid side?, memory operand: no per-row hoist needed, r9d zero-extends at its init
     jb .vcb_ar_arr_lr                         ; next local_row, bottom test
     inc r8d                                   ; tile_row++
     mov eax, vcb_grid_h_orig                  ; grid_h_orig, frame local
     cmp r8d, eax                              ; tile_row < grid_h_orig?
     jb .vcb_ar_arr_tr                         ; next tile_row, bottom test
.vcb_ar_arr_done:
; --- Step A8: Set output pointers ---
     mov rdx, vcb_out_emb                      ; &out_emb
     mov rax, [rel g_ar_final_emb]             ; final embedding buffer
     mov [rdx], rax                            ; *out_emb = final_emb
     mov rdx, vcb_out_cnt                      ; &out_count
     mov eax, dword [rel g_ar_final_cnt]       ; total token count
     mov [rdx], eax                            ; *out_count = total
     xor eax, eax                              ; 0 for NULL
     mov rdx, vcb_out_ds0                      ; &out_ds0
     mov qword [rdx], rax                      ; *out_ds0 = NULL
     mov rdx, vcb_out_ds1                      ; &out_ds1
     mov qword [rdx], rax                      ; *out_ds1 = NULL
     mov rdx, vcb_out_ds2                      ; &out_ds2
     mov qword [rdx], rax                      ; *out_ds2 = NULL
; Free all_features buffer, final_emb holds the copy
     AR_FREE_RESET g_ar_all_features           ; free + zero-reset
     AR_FREE_RESET g_ar_resized_buf            ; free + zero-reset
     xor eax, eax                              ; return code = 0
     jmp .vcb_done                             ; common exit path
%unmacro AR_FREE_RESET 1                       ; file-local free+reset triplet macro, all eight invocations above
%undef AR_ARG6_OFF
%undef AR_ARG5_OFF
%undef AR_CALL_FRAME
%undef AR_FIT_CEIL_ADJ
%undef AR_YMM_BYTES
%undef AR_RGB_HI_OFF
%undef AR_RGB_VEC_BYTES
%undef MAGIC_DWORD_BYTES
%undef VITP_HDR_BYTES
%undef AR_YMM_TAIL
%undef AR_FLOAT_SCALE_2X
%undef AR_FLOAT_SCALE
%undef AR_FLOAT_BYTES
%undef RGB_B_OFF
%undef RGB_G_OFF
%undef RGB_R_OFF
%undef AR_PIN_PAIR_BYTES
%undef AR_PIN_W_OFF
%undef ANYRES_DW3
%undef ANYRES_N_PTS
%undef ANYRES_RGB_CH
