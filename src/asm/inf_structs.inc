; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; Config struct, v7_config, 16 ints x 4 bytes = 64 bytes, power-of-2 padded
%define CFG_RESERVED0 0                        ; reserved0, uint32_t, always 0
%define CFG_RESERVED1 4                        ; reserved1, uint32_t, always 0
%define CFG_DIM 8                              ; dim field offset
%define CFG_HIDDEN_DIM 12                      ; hidden_dim field offset
%define CFG_N_LAYERS 16                        ; n_layers field offset
%define CFG_N_HEADS 20                         ; n_heads field offset
%define CFG_N_KV_HEADS 24                      ; n_kv_heads field offset
%define CFG_VOCAB_SIZE 28                      ; vocab_size field offset
%define CFG_SEQ_LEN 32                         ; seq_len field offset
%define CFG_HEAD_DIM 36                        ; head_dim field offset
%define CFG_SHARED_CLASSIFIER 40               ; shared_classifier field offset
%define CFG_GROUP_SIZE 44                      ; group_size field offset
%define CFG_HAS_QK_NORM 48                     ; has_qk_norm field offset, 1 if Qwen3 uses QK-RMSNorm
%define CFG_ROPE_BASE 52                       ; rope_base field offset, integer theta, e.g. 1000000; the float base the kernels use is the separate __sys_rope_base DB key
; sizeof,Config, 16 int32 fields, offsets CFG_RESERVED0, 0 .. CFG_RESERVED3, 60
%define CFG_SIZE_QWORDS 8                      ; sizeof,Config / 8 = 64 / 8, must equal TRANSFORMER_WEIGHTS: Config is immediately followed by Weights
; Transformer struct offsets, after 64-byte Config
%define TRANSFORMER_WEIGHTS 64                 ; Weights pointer offset, Config is 64 bytes
%define TRANSFORMER_STATE 184                  ; RunState struct offset, INLINE embedded struct, NOT a pointer, weights is exactly 120 bytes = 15 pointers
%define W_Q_TOKENS 0                           ; q_tokens pointer offset
%define W_RMS_ATT_WEIGHT 16                    ; rms_att_weight pointer offset
%define W_RMS_FFN_WEIGHT 24                    ; rms_ffn_weight pointer offset
%define W_WQ 32                                ; wq pointer offset
%define W_WK 40                                ; wk pointer offset
%define W_WV 48                                ; wv pointer offset
%define W_WO 56                                ; wo pointer offset
%define W_Q_NORM_WEIGHTS 64                    ; q_norm_weights pointer offset
%define W_K_NORM_WEIGHTS 72                    ; k_norm_weights pointer offset
%define W_W1 80                                ; w1 pointer offset
%define W_W2 88                                ; w2 pointer offset
%define W_W3 96                                ; w3 pointer offset
%define W_RMS_FINAL_WEIGHT 104                 ; rms_final_weight pointer offset
%define W_WCLS 112                             ; wcls pointer offset
%define S_X 0                                  ; x pointer offset
%define S_XB 8                                 ; xb pointer offset
%define S_HB 16                                ; hb pointer offset
%define S_HB2 24                               ; hb2 pointer offset
%define S_XQ_Q 32                              ; xq.q pointer offset
%define S_XQ_S 40                              ; xq.s pointer offset
%define S_HQ_Q 48                              ; hq.q pointer offset
%define S_HQ_S 56                              ; hq.s pointer offset
%define S_Q 64                                 ; q pointer offset
%define S_K 72                                 ; k pointer offset
%define S_V 80                                 ; v pointer offset
%define S_ATT 88                               ; att pointer offset
%define S_LOGITS 96                            ; logits pointer offset
%define S_KEY_CACHE 104                        ; key_cache pointer offset
%define S_VALUE_CACHE 112                      ; value_cache pointer offset
%define S_X_STORAGE 120                        ; x_storage pointer offset
%define S_Q_STORAGE 128                        ; q_storage pointer offset
%define S_ATT_OUT 136                          ; att_out pointer offset
%define ML_OFF_X 16                            ; size_t
%define ML_OFF_XB 24                           ; size_t
%define ML_OFF_HB 32                           ; size_t
%define ML_OFF_HB2 40                          ; size_t
%define ML_OFF_XQ_Q 48                         ; size_t
%define ML_OFF_XQ_S 56                         ; size_t
%define ML_OFF_HQ_Q 64                         ; size_t
%define ML_OFF_HQ_S 72                         ; size_t
%define ML_OFF_Q 80                            ; size_t
%define ML_OFF_ATT 88                          ; size_t
%define ML_OFF_LOGITS 96                       ; size_t
%define ML_OFF_KEY_CACHE 104                   ; size_t
%define ML_OFF_X_STORAGE 120                   ; size_t
%define ML_OFF_Q_STORAGE 128                   ; size_t
%define ML_OFF_ATT_OUT 136                     ; size_t
%define ML_EMB_TABLE_BYTES 144                 ; size_t
%define WM_FILE_OFFSET 0                       ; size_t
%define WM_ENTRY_SIZE 24                       ; padded to size_t alignment
%define WM_RMS_ATT 0                           ; index of rms_att_weight, fp32, n_layers*dim
%define WM_RMS_FFN 1                           ; index of rms_ffn_weight, fp32, n_layers*dim
%define WM_RMS_FINAL 2                         ; index of rms_final_weight, fp32, dim
%define WM_Q_NORM 3                            ; index of q_norm_weights, fp32, n_layers*head_dim
%define WM_K_NORM 4                            ; index of k_norm_weights, fp32, n_layers*head_dim
%define WM_Q_TOKENS 5                          ; index of q_tokens, Q8, vocab_size*dim
%define WM_WQ 6                                ; index of wq, Q8, n_layers*dim*n_heads*head_dim
%define WM_WK 7                                ; index of wk, Q8, n_layers*dim*n_kv_heads*head_dim
%define WM_WV 8                                ; index of wv, Q8, n_layers*dim*n_kv_heads*head_dim
%define WM_WO 9                                ; index of wo, Q8, n_layers*n_heads*head_dim*dim
%define WM_W1 10                               ; index of w1, Q8, n_layers*dim*hidden_dim
%define WM_W2 11                               ; index of w2, Q8, n_layers*hidden_dim*dim
%define WM_W3 12                               ; index of w3, Q8, n_layers*dim*hidden_dim
%define WM_Q_TOKENS_DUP 13                     ; index of the q_tokens duplicate slot, db_build_tool.c:1381 copies wm[5]; no ASM consumer reads it
%define WM_WCLS 14                             ; index of wcls, Q8; db_build_tool.c:1383-1386 copies wm[5] when shared_classifier, else its own dim*vocab_size block
%define WM_EMB_TABLE 15                        ; index of the embedding table slot: rows/is_quantized set, file_offset left 0, dequantized from model_weights, not mapped from the file
%define QT_Q 0                                 ; qword, pointer to int8 quantized data
%define QT_S 8                                 ; qword, pointer to float scales
%define QT_SIZE 16                             ; Size of QuantizedTensor struct, bytes
; ============================================================================
; InferenceContext struct offsets, matches v7_lib.h
; ============================================================================
%define CTX_TRANSFORMER 0                      ; qword, points to Transformer
%define CTX_TOKENIZER 8                        ; qword, points to Tokenizer
%define CTX_SAMPLER 16                         ; qword, points to Sampler
%define CTX_PROMPT_TOKENS 24                   ; qword, points to int prompt_tokens buffer
%define CTX_OUT_TOKENS 32                      ; qword, points to int out_tokens buffer
%define CTX_MAX_PROMPT 40                      ; dword, max prompt tokens capacity
%define CTX_MAX_GEN 44                         ; dword, max generated tokens capacity
%define CTX_POS 48                             ; dword, current sequence position, for KV-cache
%define CTX_QUANTIZE 52                        ; dword, quantize mode flag, 0=F32, 1=Q8
%define CTX_HEADER_SIZE 56                     ; InferenceContext header size, bytes
%define TOK_VOCAB 0                            ; qword, char** vocab, ptr to ptr array
%define TOK_MERGE_SCORES 8                     ; qword, float* merge_scores
%define TOK_VOCAB_SIZE 16                      ; dword, int vocab_size
%define TOK_MAX_TOKEN_LEN 20                   ; dword, unsigned int max_token_length
%define TOK_BOS_ID 24                          ; dword, unsigned int bos_token_id
%define TOK_EOS_ID 28                          ; dword, unsigned int eos_token_id
%define TOK_PROMPT_TEMPLATE 32                 ; 1024 bytes
%define TOK_SYSTEM_TEMPLATE 1056               ; 1024 bytes
%define TOK_IS_SP 2080                         ; dword, int is_sentencepiece, 1=Space->U+2581 + dummy prefix
%define TOK_SIZE 2088                          ; Tokenizer struct size, bytes, incl. 4 pad bytes
; Sampler struct offsets, internal DLL layout, 40 bytes.Field-for-field identical to the C Sampler: SAM_RESERVED at 4 is the 4-byte pad that puts SAM_PROBINDEX on an 8-byte boundary
%define SAM_VOCAB_SIZE 0                       ; dword, int vocab_size
%define SAM_PROBINDEX 8                        ; qword, ProbIndex* probindex
%define SAM_TEMPERATURE 16                     ; float, 4 bytes
%define SAM_TOPP 20                            ; float, 4 bytes
%define SAM_PRESENCE_PENALTY 24                ; float, 4 bytes, presence penalty
%define SAM_TOPK 28                            ; dword, 4 bytes, top-k limit
%define SAM_RNG_STATE 32                       ; qword, unsigned long long rng_state
%define SAM_SIZE 40                            ; Sampler struct size, bytes
%define HDR_BOS_ID 4                           ; dword, bos_token_id
%define HDR_EOS_ID 8                           ; dword, eos_token_id
%define VB_HEADER_SIZE 4                       ; bytes before first entry
%define VB_ENTRY_HEADER 4                      ; bytes from entry start to string data
%define VCONF_IMAGE_TOKEN_ID 0                 ; int32, image placeholder token id, 151655
%define VCONF_VISION_START 4                   ; int32, <|vision_start|> id, 151652
%define VCONF_VISION_END 8                     ; int32, <|vision_end|> id, 151653
%define VCONF_HAS_DEEPSTACK 12                 ; int32, 1 = apply DeepStack adds in decoder layers 0/1/2
%define VCFG_HIDDEN 64                         ; qword, vis_hidden, 1024
%define VCFG_HIDDEN_F4 72                      ; qword, hidden x 4, 4096
%define VCFG_INTER 80                          ; qword, inter, 4096
%define VCFG_QKV_DIM 88                        ; qword, 3 x hidden, 3072
%define VCFG_QKV_STRIDE 96                     ; qword, qkv_dim x 4, 12288
%define VCFG_HEADS 104                         ; qword, heads, 16
%define VCFG_HEAD_DIM 112                      ; qword, hidden / heads, 64
%define VCFG_HEAD_DIM_F4 120                   ; qword, head_dim x 4, 256
%define VCFG_DEPTH 128                         ; qword, depth, 24
%define VCFG_NUM_POS 136                       ; qword, num_pos, 2304
%define VCFG_NUM_POS_F4 144                    ; qword, num_pos x 4, 9216
%define VCFG_GRID_SIDE 152                     ; qword, isqrt,num_pos, 48
%define VCFG_GRID_MINUS1_I 160                 ; qword, grid_side - 1 int, 47
%define VCFG_GRID_MINUS1_F 168                 ; qword, float bits of, float, grid_side-1, 0x423C0000
%define VCFG_MERGE 176                         ; qword, merge, 2
%define VCFG_MERGE_UNIT 184                    ; qword, merge x merge, 4
%define VCFG_PATCH_SIZE 192                    ; qword, patch_size, 16
%define VCFG_PATCH_SIZE_F4 200                 ; qword, patch_size x 4, 64
%define VCFG_OUT_HIDDEN 208                    ; qword, out_hidden, 2560
%define VCFG_PATCH_DIM 216                     ; qword, 3 x temporal x patch_size^2, 1536
%define VCFG_PATCH_DIM_F4 224                  ; qword, patch_dim x 4, 6144
%define VCFG_PACKED_DIM 232                    ; qword, hidden x merge_unit, 4096
%define VCFG_PACKED_DIM_F4 240                 ; qword, packed_dim x 4, 16384
%define VCFG_N_DEEPSTACK 248                   ; qword, DeepStack tap count, 3 Qwen3-VL, 0 SigLIP
%define VCFG_ROPE_QTR 256                      ; qword, head_dim / 4, 16
%define VCFG_ROPE_HALF 264                     ; qword, head_dim / 2, 32
%define VCFG_ROPE_HALF_F4 272                  ; qword, rope_half x 4, 128
%define VCFG_DS_TAP0 280                       ; qword, ds_tap0, 5
%define VCFG_DS_TAP1 288                       ; qword, ds_tap1, 11
%define VCFG_DS_TAP2 296                       ; qword, ds_tap2, 17
%define VCFG_L_QKV_W 304                       ; qword, qkv_dim x hidden x 4, 12582912
%define VCFG_L_QKV_B 312                       ; qword, qkv_dim x 4, 12288
%define VCFG_L_PROJ_W 320                      ; qword, hidden^2 x 4, 4194304
%define VCFG_L_PROJ_B 328                      ; qword, hidden x 4, 4096
%define VCFG_L_FC1_W 336                       ; qword, inter x hidden x 4, 16777216
%define VCFG_L_FC1_B 344                       ; qword, inter x 4, 16384
%define VCFG_L_FC2_W 352                       ; qword, hidden x inter x 4, 16777216
%define VCFG_L_FC2_B 360                       ; qword, hidden x 4, 4096
%define VCFG_BLK_NORM1_STRIDE 368              ; qword, depth x hidden x 4, 98304
%define VCFG_BLK_QKV_W_STRIDE 376              ; qword, depth x qkv_dim x hidden x 4, 301989888
%define VCFG_BLK_QKV_B_STRIDE 384              ; qword, depth x qkv_dim x 4, 294912
%define VCFG_BLK_PROJ_W_STRIDE 392             ; qword, depth x hidden^2 x 4, 100663296
%define VCFG_BLK_PROJ_B_STRIDE 400             ; qword, depth x hidden x 4, 98304
%define VCFG_BLK_FC1_W_STRIDE 408              ; qword, depth x inter x hidden x 4, 402653184
%define VCFG_BLK_FC1_B_STRIDE 416              ; qword, depth x inter x 4, 393216
%define VCFG_BLK_FC2_W_STRIDE 424              ; qword, depth x hidden x inter x 4, 402653184
%define VCFG_BLK_FC2_B_STRIDE 432              ; qword, depth x hidden x 4, 98304
%define VCFG_MG_FC1_W_STRIDE 440               ; qword, packed_dim^2 x 4, 67108864
%define VCFG_MG_FC1_B_STRIDE 448               ; qword, packed_dim x 4, 16384
%define VCFG_MG_FC2_W_STRIDE 456               ; qword, out_hidden x packed_dim x 4, 41943040
%define VCFG_MG_FC2_B_STRIDE 464               ; qword, out_hidden x 4, 10240
%define VCFG_DS_FC1_W_STRIDE 472               ; qword, 3 x packed_dim^2 x 4, 201326592
%define VCFG_DS_FC1_B_STRIDE 480               ; qword, 3 x packed_dim x 4, 49152
%define VCFG_DS_FC2_W_STRIDE 488               ; qword, 3 x out_hidden x packed_dim x 4, 125829120
%define VCFG_DS_FC2_B_STRIDE 496               ; qword, 3 x out_hidden x 4, 30720
%define VCFG_PE_W_STRIDE 504                   ; qword, patch_dim x hidden x 4, 6291456
%define VCFG_POS_EMBED_BYTES 512               ; qword, num_pos x hidden x 4, 9437184
%define VCFG_ATTN_SCALE 520                    ; qword, IEEE-754 bits of 1/sqrt,head_dim, 0x3E000000 for 0.125, head_dim 64, 0x3DF15BEF for 0.1178511302, head_dim 72.Per-§119 arithmetic is precomputed in C; kernels load with a single RIP-relative read.
; --- Cross-model feature fields, offsets 528-728, from .vision header slots ---
%define VCFG_NORM_TYPE 528                     ; qword, 0=RMSNorm, Qwen3-VL, 1=LayerNorm, SigLIP
%define VCFG_FFN_TYPE 536                      ; qword, 0=SwiGLU, Qwen3-VL, 1=GELU MLP, SigLIP
%define VCFG_HAS_CLS 544                       ; qword, 1 = prepend class token, 0 Qwen3-VL
%define VCFG_FEATURE_SELECT 552                ; qword, 0=cls only, 1=full patch embeddings, 1
%define VCFG_PROJ_HIDDEN 560                   ; qword, projector output dim, Qwen3-VL merger out = 2560
%define VCFG_PROJ_LAYERS 568                   ; qword, projector MLP layer count, 2 Qwen3-VL
%define VCFG_PROJ_ACTIVATION 576               ; qword, 1=GELU, 0=none, 1 Qwen3-VL
%define VCFG_POS_EMBED_TYPE 584                ; qword, 0=Qwen3-VL bilinear-grid, 1=learned additive, SigLIP
%define VCFG_IMAGE_SIZE 592                    ; qword, image size, SigLIP 384; 0 = Qwen3-VL position grid
; offset 600 reserved, stays 0 in the blob : VCFG_N_DEEPSTACK at 248 carries the tap count
%define VCFG_ROPE_THETA_LOG2 608               ; dword, IEEE-754 bits of log2,theta, 0x41549A78 for theta=10000 ; precomputed in C, loaded with a single RIP-relative read, no runtime math
%define VCFG_USE_ROPE 616                      ; dword, 1=apply RoPE to Q,K each layer, Qwen3-VL ; 0=skip RoPE entirely, SigLIP
%define VCFG_PROJ_IN_DIM 624                   ; qword, projector input dim, 0=no separate projector / Qwen3-VL merger; 1152 SigLIP
%define VCFG_PROJ_OUT_DIM 632                  ; qword, projector output dim, 0=no projector; 896 SigLIP = text hidden
%define VCFG_PROJ_INTER 640                    ; qword, projector intermediate dim, 0=no projector; 896 SigLIP
%define VCFG_PROJ_TYPE 648                     ; dword, 0=none,Qwen3-VL merger, 1=MLP,SigLIP/LLaVA-OV FC1+GELU+FC2, 2=pixel-shuffle,SmolVLM/Idefics3
%define VCFG_APPLY_POST_LN 656                 ; dword, 1=apply post_layernorm after layer loop, SmolVLM/Idefics3, 0=copy raw hidden, LLaVA-OV
%define VCFG_PS_FACTOR 664                     ; qword, pixel-shuffle block size F, 4 for SmolVLM-256M, 3 for SmolVLM-2B
%define VCFG_PS_GRID_SIDE 672                  ; qword, image_size / patch_size, 32 for SmolVLM-256M
%define VCFG_PS_FLAT_DIM 680                   ; qword, vision_hidden * F * F, 12288 for SmolVLM-256M
%define VCFG_PS_N_TOKENS 688                   ; qword,, grid_side/F ^2, output token count, 64 for SmolVLM-256M
%define VCFG_ANYRES_ENABLED 696                ; qword, 1 = anyres multi-tile, LLaVA-OV, 0 = single-tile
%define VCFG_LN_EPS 704                        ; dword, IEEE-754 float bits of ViT LayerNorm epsilon, 1e-5 SigLIP, 1e-6 Qwen3-VL.Read as dword by vit_layernorm.
%define VCFG_AR_GRID_SIDE 712                  ; qword, anyres tile patch grid side = image_size / patch_size, truncated, 27 for SigLIP 384/14; HF-reference truncation semantics
%define VCFG_AR_BASE_TOK 720                   ; qword, ar_grid_side^2 = base tokens per anyres tile, 729 for SigLIP
%define VCFG_AR_TC_ADV 728                     ; qword, ar_base_tok - ar_grid_side, 702: src advance per tile_col in the anyres arrange loop
%define VCFG_PS_OUT_W 736                      ; qword, grid_side / F: output tokens per side (out_h == out_w: n_tokens = out_w^2, square grid)
; struct size = VCFG_PS_OUT_W + 8 = 744; block holds VCFG_QWORDS qwords
%define VCFG_QWORDS 93                         ; qwords in the g_vit_cfg BSS block, 93 used, 0 spare
