; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%macro ALIGN64_RAX 0
     add rax, 63                               ; round up for alignment
     and rax, -64                              ; clear low 6 bits, 64-byte align, full 64-bit mask
%endmacro
; ============================================================================
; CACHE_CFG cfg_off_const, global

; Load one dword config field from r12, = Config* into a global.
; Interface: reads r12, Config*, writes eax and the named global.
; Callers relying on eax validity for a follow-on validation MUST place that
; validation immediately after the invocation, no other macro/call in between .
; ============================================================================

%macro CACHE_CFG 2
     mov eax, [r12 + %1]                       ; Load config dword from r12, Config*
     mov %2, eax                               ; Cache into global
%endmacro
; V7_init,filename must be called before v7s_inference_create.
     PUB_CPU_GATE                              ; refuse to run on hardware below the v7 baseline
; === Step 1: Get config from microdb ===
     lea rcx, [rel inf_key_config]             ; rcx = "__sys_config" string key
     xor edx, edx                              ; out_len = NULL
     call ht_find_or_fatal                     ; Find config entry, FATAL if DB not loaded
     mov inf_cfg_ptr, rax                      ; Save config pointer for later use
; Step 1: config is valid — reserved fields at offsets 0,4 are always 0 per §96
; === Step 2: Get mem_layout from microdb ===
     lea rcx, [rel inf_key_mem_layout]         ; rcx = "__sys_mem_layout" string key
     xor edx, edx                              ; out_len = NULL
     call ht_find_or_fatal                     ; Find mem_layout entry, FATAL with key name on miss
     mov inf_ml_ptr, rax                       ; Save mem_layout pointer
; mem_layout is valid — reserved field at offset 0 is always 0 per §96
; === Step 3: Cache config values ===
; CRITICAL: use r12 as preserved config pointer to avoid eax zero-extension overwrite
     mov r12, inf_cfg_ptr                      ; r12 = config pointer, preserved by SF_START, preserved across eax writes
     CACHE_CFG CFG_VOCAB_SIZE, inf_vocab_size  ; vocab_size -> global
     CACHE_CFG CFG_N_LAYERS, inf_n_layers      ; Cache n_layers from config, build-time invariant, per §96/§57d
     CACHE_CFG CFG_DIM, inf_dim                ; Cache dim from config
     CACHE_CFG CFG_HIDDEN_DIM, inf_hidden_dim  ; hidden_dim -> global
     CACHE_CFG CFG_HEAD_DIM, inf_head_dim      ; Cache head_dim from config
     CACHE_CFG CFG_N_HEADS, inf_n_heads        ; Cache n_heads from config
     CACHE_CFG CFG_N_KV_HEADS, inf_n_kv_heads  ; n_kv_heads -> global
     CACHE_CFG CFG_SEQ_LEN, inf_seq_len        ; Cache seq_len from config
     CACHE_CFG CFG_SHARED_CLASSIFIER, inf_shared_cls ; Cache shared_classifier flag from config at +40
     CACHE_CFG CFG_GROUP_SIZE, inf_group_size  ; group_size -> global, +44
     lea rcx, [rel inf_key_rms_eps]            ; rcx = "__sys_rms_eps" key
     xor edx, edx                              ; out_len = NULL
     call ht_find_or_fatal                     ; rax = data ptr, FATAL with key name on miss
     mov eax, [rax]                            ; load eps float32 from DB
     mov [rel g_rms_eps], eax                  ; store in BSS global for kernels
     lea rcx, [rel inf_key_vision_config]      ; rcx = "__sys_vision_config" key
     lea rdx, [rel g_vit_cfg_len]              ; out_len = BSS scratch qword
     call ht_find                              ; rax = data ptr or NULL, non-fatal lookup
     test rax, rax                             ; vision config present in DB?
     jz .no_vision_config                      ; no, text-only model, zero all vision globals
     mov rsi, rax                              ; rsi = DB source pointer, movsq copies FROM rsi
     lea rdi, [rel g_vit_cfg]                  ; rdi = BSS destination block, movsq writes TO rdi
     mov ecx, VCFG_QWORDS                      ; 93 qwords = 744 bytes to copy
     rep movsq                                 ; blob copy, overwrites rsi/rdi, safe in init
     mov r10d, [rel g_vit_cfg + VCONF_IMAGE_TOKEN_ID] ; image placeholder token id
     mov [rel g_image_token_id], r10d          ; set the active override flag, fresh pool follows at set_vision_embeddings
     mov [rel g_model_image_token_id], r10d    ; persist the model's image token id, re-applied by set_vision_embeddings
     mov r10d, [rel g_vit_cfg + VCONF_HAS_DEEPSTACK] ; DeepStack flag
     mov [rel g_cfg_has_deepstack], r10d       ; store global, 1 = apply decoder-layer adds
%define VPPF_NUL_BYTES 1                       ; sizeof,NUL byte, skip past terminator to next string
     lea rcx, [rel inf_key_vis_prompt_fmt]     ; key = "__sys_vision_prompt_fmt"
     lea rdx, [rel g_vit_cfg_len]              ; out_len scratch, cfg length already consumed above
     call ht_find_or_fatal                     ; rax = blob pointer, mandatory key: a vision DB without it is stale
     mov rbx, [rel g_vit_cfg_len]              ; blob byte length, ht_find stores the full 64-bit E_DLEN, so the qword read is exact; the builder caps the blob at 64 bytes
     mov rsi, rax                              ; src = pool blob
     lea rdi, [rel g_vis_prompt_buf]           ; dst = BSS copy, pool-independent
     mov rcx, rbx                              ; byte count
     rep movsb                                 ; copy blob out of the pool, overwrites rsi/rdi, init only
.vppf_copy_done:
; Split the copy into its three NUL-terminated strings, prefix, marker, suffix .
; The builder writes exactly three terminated strings, the scans below are
; the PARSE, not validation: a missing NUL is the builder's failure, not ours.
; The prefix scan MUST test blob[0] FIRST: the prefix may be EMPTY, LLaVA-OV's
; blob is NUL "<image>" NUL "\n" NUL, an empty prefix is legal, and an
; inc-first scan would skip it, parse the marker as the prefix, and the image
; placeholder would never reach the prompt, the model answers blind .
     lea rax, [rel g_vis_prompt_buf]           ; rax = BSS copy base, prefix = blob[0]
     mov rcx, rax                              ; rcx = prefix scan cursor, starts AT blob[0]
.vppf_scan_prefix:
     cmp byte [rcx], 0                         ; NUL terminator found?, empty prefix: fires immediately
     je .vppf_scan_prefix_done                 ; yes, rcx = &prefix NUL
     inc rcx                                   ; advance cursor
     jmp .vppf_scan_prefix                     ; keep scanning
.vppf_scan_prefix_done:
     lea rdx, [rcx + VPPF_NUL_BYTES]           ; rdx = marker start, skip past the prefix NUL byte
     xor r11d, r11d                            ; r11d = marker body length = 0
.vppf_strlen_marker:
     cmp byte [rdx + r11], 0                   ; NUL at marker[length]?
     je .vppf_marker_len_done                  ; yes, r11d holds the body length
     inc r11d                                  ; length++
     jmp .vppf_strlen_marker                   ; continue scanning
.vppf_marker_len_done:
     lea r10, [rdx + r11 + VPPF_NUL_BYTES]     ; r10 = suffix start, marker NUL + 1
; Publish the parsed pointers/length
     mov [rel g_vis_prompt_prefix], rax        ; prefix = blob[0], first string
     mov [rel g_vis_prompt_marker], rdx        ; cache marker pointer
     mov [rel g_vis_prompt_marker_len], r11d   ; cache marker body length, EMIT_IMAGE_PADS bound
     mov [rel g_vis_prompt_suffix], r10        ; cache suffix pointer
.vppf_fmt_ok:
%define PINPOINT_H_OFF 0                       ; byte offset of height field in, h,w pair
%define PINPOINT_W_OFF 4                       ; byte offset of width field in, h,w pair
%define PINPOINT_PAIR_BYTES 8                  ; bytes per, h,w pair
%define PINPOINT_NEXT 1                        ; index step to the NEXT tile, pinpoints are 1-based:, i+1 *image_size
     cmp qword [rel g_vit_cfg + VCFG_ANYRES_ENABLED], 0 ; anyres model?
     je .vis_no_anyres                         ; no, skip newline + pinpoints
     mov r11d, dword [rel g_vit_cfg + VCFG_IMAGE_SIZE] ; r11d = anyres base image_size, positive divisor is the builder's contract
     lea rdi, [rel g_anyres_pinpoints]         ; rdi = destination cursor
     xor r9d, r9d                              ; r9d = i = 0, height loop index
.vppf_pin_h:
     lea eax, [r9+PINPOINT_NEXT]               ; i+1, pinpoints are 1-based tiles
     imul eax, r11d                            ; height =, i+1 * image_size
     mov r10d, eax                             ; r10d = height, constant per outer loop
     mov eax, r11d                             ; width starts at 1 * image_size
     mov r8d, 6                                ; six widths per height
.vppf_pin_w:
     mov dword [rdi + PINPOINT_H_OFF], r10d    ; store height
     mov dword [rdi + PINPOINT_W_OFF], eax     ; store width
     add eax, r11d                             ; next width by finite difference
     add rdi, PINPOINT_PAIR_BYTES              ; advance to next, h,w pair
     dec r8d                                   ; width counter, bottom-tested
     jnz .vppf_pin_w                           ; next width
     inc r9d                                   ; i++
     cmp r9d, 6                                ; i < 6?
     jl .vppf_pin_h                            ; next height
     lea rcx, [rel inf_key_img_newline]        ; key = "__sys_image_newline_emb"
     lea rdx, [rel g_vit_cfg_len]              ; out_len scratch, prompt-fmt length already consumed
     call ht_find_or_fatal                     ; rax = data ptr, mandatory with ANYRES=1: pack_image_features derefs it unconditionally
     mov rbx, [rel g_vit_cfg_len]              ; blob byte length, ht_find stores the full 64-bit E_DLEN, so the qword read is exact; fits the BSS buffer by the builder's cap
     mov rsi, rax                              ; src = pool blob
     lea rdi, [rel g_anyres_newline_buf]       ; dst = BSS copy
     mov rcx, rbx                              ; byte count
     rep movsb                                 ; copy embedding out of the pool
     lea rax, [rel g_anyres_newline_buf]       ; rax = BSS copy base
     mov [rel g_anyres_newline_emb], rax       ; cache pointer to the BSS copy
     TRACE_STR "anyres: pinpoints filled + newline loaded" ; diagnostic
     jmp .vision_config_done                   ; anyres path is done, the zeroing below is for the non-anyres case only
.vis_no_anyres:
     mov qword [rel g_anyres_newline_emb], 0   ; no anyres newline embedding, a non-anyres vision model must not inherit the pointer a prior anyres model cached
     jmp .vision_config_done                   ; skip zeroing, vision config was loaded
.no_vision_config:
; Text-only model: zero all vision globals + g_vit_cfg to prevent stale state from prior loads
     xor eax, eax                              ; zero source for block clear
     lea rdi, [rel g_vit_cfg]                  ; rdi = start of config block
     mov ecx, VCFG_QWORDS                      ; clear 93 qwords
     rep stosq                                 ; zero the whole g_vit_cfg block
     mov [rel g_vit_cfg_len], rax              ; no config copied, length stays 0
     mov [rel g_image_token_id], eax           ; no active image placeholder token
     mov [rel g_model_image_token_id], eax     ; no model image token id, text-only model
     mov [rel g_cfg_has_deepstack], eax        ; no DeepStack injection
     mov [rel g_vision_embeddings], rax        ; no vision embedding buffer
     mov [rel g_vision_emb_count], eax         ; no vision tokens
     mov [rel g_vision_emb_consumed], eax      ; reset consumption counter
; 3 qwords, so each store clears exactly one slot pointer.
     mov [rel g_deepstack_features + VDS_SLOT0], rax ; no DS feature slot 0
     mov [rel g_deepstack_features + VDS_SLOT1], rax ; no DS feature slot 1
     mov [rel g_deepstack_features + VDS_SLOT2], rax ; no DS feature slot 2
     mov [rel g_anyres_newline_emb], rax       ; no anyres newline embedding, text-only
     mov [rel g_vis_prompt_prefix], rax        ; no prefix string
     mov [rel g_vis_prompt_marker], rax        ; no marker string
     mov [rel g_vis_prompt_suffix], rax        ; no suffix string
     mov [rel g_vis_prompt_marker_len], eax    ; no marker length
.vision_config_done:
%define VIT_HIDDEN_PAIR_SCALE 2                ; SIB scale for two hidden-sized buffers in one LEA
%define VIT_COS_PAIR_SCALE 2                   ; SIB scale for cos+sin sharing one aligned size
%define TRANSFORMER_ALLOC_BYTES 384            ; byte size of the Transformer struct, cfg + weights + runstate pointer blocks
%define VIT_SCRATCH_ROUND_ADD 65535            ; add-mask that rounds the ViT scratch total up to the next 64KB page
%define VIT_SCRATCH_ROUND_CLR -65536           ; matching clear-mask, the same 64KB page, negated so it is a native imm32
%define VITPTR_HIDDEN 0                        ; g_vit_ptr_hidden offset
%define VITPTR_QKV 8                           ; g_vit_ptr_qkv offset
%define VITPTR_NORM 16                         ; g_vit_ptr_norm offset
%define VITPTR_MLP 24                          ; g_vit_ptr_mlp offset
%define VITPTR_SCORES 32                       ; g_vit_ptr_scores offset
%define VITPTR_ATTN 40                         ; g_vit_ptr_attn offset
%define VITPTR_POS 48                          ; g_vit_ptr_pos offset
%define VITPTR_COS 56                          ; g_vit_ptr_cos offset
%define VITPTR_SIN 64                          ; g_vit_ptr_sin offset
%define VITPTR_PACKED 72                       ; g_vit_ptr_packed offset
%define VITPTR_MMLP 80                         ; g_vit_ptr_mmlp offset
     cmp dword [rel g_image_token_id], 0       ; vision model?
     je .vit_scratch_skip                      ; text-only, no scratch block needed
     mov r12, qword [rel g_vit_cfg + VCFG_NUM_POS] ; r12 = num_pos, patch count, factor of every sub-buffer below; nonzero is the builder's contract
; --- Compute aligned sub-buffer sizes ---
; hidden = num_pos * hidden_f4
     mov rax, qword [rel g_vit_cfg + VCFG_HIDDEN_F4] ; hidden*4 bytes per patch
     imul rax, r12                             ; * num_pos patches
     ALIGN64_RAX                               ; round up rax to 64-byte boundary
     mov r13, rax                              ; r13 = hidden_aligned
; qkv = num_pos * qkv_stride
     mov rax, qword [rel g_vit_cfg + VCFG_QKV_STRIDE] ; qkv bytes per patch row
     imul rax, r12                             ; * num_pos patches
     ALIGN64_RAX                               ; round up rax to 64-byte boundary
     mov r14, rax                              ; r14 = qkv_aligned
; mlp = num_pos * inter * 4
     mov rax, qword [rel g_vit_cfg + VCFG_INTER] ; inter dimension
     shl rax, 2                                ; inter * 4 = bytes per row
     imul rax, r12                             ; * num_pos patches
     ALIGN64_RAX                               ; round up rax to 64-byte boundary
     mov r15, rax                              ; r15 = mlp_aligned
; cos = num_pos * rope_half_f4
     mov rax, qword [rel g_vit_cfg + VCFG_ROPE_HALF_F4] ; rope_half*4 bytes per row, only the first half is consumed
     imul rax, r12                             ; * num_pos patches
     ALIGN64_RAX                               ; round up rax to 64-byte boundary
     mov inf_vit_push_cos, rax                 ; save cos_aligned, frame slot, read back after the mdb_alloc_mem call below, which overwrites the volatile carriers
     mov rax, qword [rel g_vit_cfg + VCFG_NUM_POS_F4] ; rax = num_pos * 4 = one softmax score row in bytes, per-head scratch; the heads multiply below yields one row per head, matches vit_dispatch's ALIGN64,n*4 per-head stride
     ALIGN64_RAX                               ; round up rax to 64-byte boundary, 27x27=729 case needs it
     imul rax, qword [rel g_vit_cfg + VCFG_HEADS] ; * num heads, product of a 64-multiple stays 64-multiple, no re-align needed
     mov r12, rax                              ; r12 = scores_aligned, carried to the carve
; --- Accumulate total ---
; 3*hidden + qkv + mlp + scores + 2*cos; cache-bearing pos stays separate
     lea rbx, [r14 + r13*VIT_HIDDEN_PAIR_SCALE] ; 2*hidden, hidden + norm, same size each, plus qkv
     add rbx, r13                              ; + pos, third hidden-sized buffer
     add rbx, r15                              ; + mlp
     add rbx, r12                              ; + scores
     mov r11, inf_vit_push_cos                 ; r11 = cos/sin aligned size, frame slot, same for both
     lea rbx, [rbx + r11*VIT_COS_PAIR_SCALE + VIT_SCRATCH_ROUND_ADD] ; + cos/sin and the 64KB round-up offset in one LEA
; Round total up to 64KB page for mdb_alloc_mem
     and rbx, VIT_SCRATCH_ROUND_CLR            ; 64KB align, clears low 16 bits of rbx
     mov rcx, rbx                              ; arg1: total bytes
     call mdb_alloc_mem                        ; rax = block base, FATALs on OOM
     mov [rel g_vit_scratch_block], rax        ; save base pointer
     mov r11, inf_vit_push_cos                 ; r11 = cos/sin aligned size, frame slot, same
     lea r9, [rel g_vit_ptr_hidden]            ; r9 = base of contiguous ViT ptr globals, 11 qwords
     mov [r9 + VITPTR_HIDDEN], rax             ; store hidden pointer
     add rax, r13                              ; advance past hidden
; qkv
     mov [r9 + VITPTR_QKV], rax                ; store qkv pointer
     add rax, r14                              ; advance past qkv
; norm, ==hidden size
     mov [r9 + VITPTR_NORM], rax               ; store norm pointer
     add rax, r13                              ; advance past norm
; mlp
     mov [r9 + VITPTR_MLP], rax                ; store mlp pointer
     add rax, r15                              ; advance past mlp
; scores, r12 = scores_aligned from phase 1
     mov [r9 + VITPTR_SCORES], rax             ; scores ptr
     add rax, r12                              ; advance past scores
; pos, ==hidden
     mov [r9 + VITPTR_POS], rax                ; store pos pointer
     add rax, r13                              ; advance past pos
; Reuse norm for attn and packed: their contents never overlap with norm.
     mov rdx, [r9 + VITPTR_NORM]               ; shared hidden-sized scratch
     mov [r9 + VITPTR_ATTN], rdx               ; attn overwrites dead norm after QKV
     mov [r9 + VITPTR_PACKED], rdx             ; packed used only at DeepStack/main merger
; Reuse qkv for merger output: QKV is dead by each merger invocation.
     mov rdx, [r9 + VITPTR_QKV]                ; qkv size >= packed merger output size
     mov [r9 + VITPTR_MMLP], rdx               ; mmlp never overlaps a live qkv tensor
; cos, r11 = cos_aligned from phase 1
     mov [r9 + VITPTR_COS], rax                ; cos ptr
     add rax, r11                              ; advance past cos
; sin, ==cos size
     mov [r9 + VITPTR_SIN], rax                ; store sin pointer
     add rax, r11                              ; advance past sin
.vit_scratch_skip:
     mov qword [rel g_cached_pe_pos], 0        ; 0 never equals a real buffer address: the first pos_embed call of every context recomputes
     mov rcx, TRANSFORMER_ALLOC_BYTES          ; arg1: Transformer struct size
     call mdb_alloc_mem                        ; rax = block base, FATALs on OOM
     mov inf_transformer, rax                  ; Cache pointer in SF frame slot
     mov [g_inf_transformer], rax              ; Track in BSS for cleanup
; %undef cleanup: file-local defines leave scope here, house convention --
%unmacro CACHE_CFG 2                           ; file-local config->global cache macro, all ten invocations are above
%unmacro ALIGN64_RAX 0                         ; file-local 64-byte round-up macro, all six invocations are above
%undef VITPTR_MMLP                             ; g_vit_ptr_mmlp offset
%undef VITPTR_PACKED                           ; g_vit_ptr_packed offset
%undef VITPTR_SIN                              ; g_vit_ptr_sin offset
%undef VITPTR_COS                              ; g_vit_ptr_cos offset
%undef VITPTR_POS                              ; g_vit_ptr_pos offset
%undef VITPTR_ATTN                             ; g_vit_ptr_attn offset
%undef VITPTR_SCORES                           ; g_vit_ptr_scores offset
%undef VITPTR_MLP                              ; g_vit_ptr_mlp offset
%undef VITPTR_NORM                             ; g_vit_ptr_norm offset
%undef VITPTR_QKV                              ; g_vit_ptr_qkv offset
%undef VITPTR_HIDDEN                           ; g_vit_ptr_hidden offset
%undef PINPOINT_PAIR_BYTES                     ; bytes per, h,w pair
%undef PINPOINT_W_OFF                          ; byte offset of width field in, h,w pair
%undef PINPOINT_H_OFF                          ; byte offset of height field in, h,w pair
%undef PINPOINT_NEXT                           ; index step to the next tile, file-local pinpoint constant
%undef VPPF_NUL_BYTES                          ; sizeof,NUL byte

%undef VIT_SCRATCH_ROUND_CLR
%undef VIT_SCRATCH_ROUND_ADD
%undef TRANSFORMER_ALLOC_BYTES
