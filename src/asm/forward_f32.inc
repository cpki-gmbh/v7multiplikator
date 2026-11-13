; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define FWD_F32_FLOAT_BYTES 4                  ; Bytes per float element, SIB scale for float array indexing
%define FWD_F32_FP16_BYTES 2                   ; Bytes per FP16 cache element, SIB scale for packed KV addressing
     global forward_f32                        ; Export F32 forward function
forward_f32:
     xor r9d, r9d                              ; public callers request logits
     SF_START forward_f32_mode, rbx, r15, r12, r13, r14 ; Begin forward_f32 stack frame
     SF_PTR ff_cfg_ptr                         ; Config*
     SF_PTR ff_w_ptr                           ; TransformerWeights*
     SF_DWORD ff_dim                           ; Stack slot: model dimension
     SF_DWORD ff_hidden_dim                    ; Stack slot: hidden dimension
     SF_DWORD ff_n_layers                      ; Stack slot: number of layers
     SF_DWORD ff_seq_len                       ; Stack slot: max sequence length
     SF_DWORD ff_n_heads                       ; Stack slot: number of query heads
     SF_DWORD ff_n_kv_heads                    ; Stack slot: number of key/value heads
     SF_DWORD ff_head_dim                      ; Stack slot: head dimension
     SF_DWORD ff_vocab_size                    ; Stack slot: vocabulary size
     SF_QWORD ff_kv_dim                        ; Stack slot: n_kv_heads * head_dim, key/value width
     SF_DWORD ff_kv_mul                        ; Stack slot: KV head multiplier
     SF_QWORD ff_all_heads_dim                 ; Stack slot: n_heads * head_dim, query width
     SF_DWORD ff_layer_idx                     ; Stack slot: current layer index
     SF_QWORD ff_layer_stride                  ; Stack slot: layer stride, bytes
     SF_QWORD ff_loff                          ; Stack slot: layer*seq_len*kv_dim offset in FLOAT-ELEMENT units, multiply by 4 for bytes at the use site
     SF_QWORD ff_seq_len_kv                    ; Stack slot: cached seq_len * kv_dim, invariant, float units per layer
     SF_QWORD ff_pos_kv                        ; Stack slot: cached pos * kv_dim in FLOAT-ELEMENT units, invariant across layers; the *4 belongs in the consuming LEA
     SF_DWORD ff_pos_local                     ; Stack slot: local position in sequence
     SF_DWORD ff_token_local                   ; Stack slot: current token index
     SF_QWORD ff_ds_row_local                  ; Stack slot: vision row offset in ELEMENTS, row * dim; the *4 byte scale sits in the consuming LEA
     SF_DWORD ff_do_ds                         ; Stack slot: folded DeepStack gate, 1 = has_deepstack AND token==image_token; per-call constant
     SF_DWORD ff_has_qk_norm                   ; Stack slot: has_qk_norm flag, 0=no per-head QK norm
     SF_DWORD ff_dim_bytes                     ; Stack slot: dim * 4, precomputed memcpy byte count
     SF_QWORD ff_dsum                          ; Stack slot: all_heads_dim + 2*kv_dim, fused [Q|K|V] bias row width
     SF_QWORD ff_layer_dim_off                 ; Stack slot: incremental layer*dim offset, avoids imul per layer
     SF_QWORD ff_layer_hd_off                  ; Stack slot: incremental layer*head_dim offset, avoids imul per layer
     SF_QWORD ff_layer_bias_off                ; Stack slot: incremental layer*dsum offset, avoids imul per layer
     SF_QWORD ff_bias_base                     ; Stack slot: cached g_qkv_bias base, BSS invariant, loaded once per call
; tensor_f32 temp struct, 2 qwords: {float* data, float* scales}
; Declare scales FIRST so data is at lower address = struct offset 0
     SF_QWORD ff_tensor_scale                  ; Stack slot: tensor_f32 scales pointer
     SF_QWORD ff_tensor_data                   ; Stack slot: tensor_f32 data pointer
     SF_DWORD ff_cache_only                    ; nonzero when only the KV writes are needed
     SF_COMMIT                                 ; Commit forward_f32 stack frame
     mov ff_cache_only, r9d                    ; preserve the private replay mode
; === Save inputs ===
     mov ff_token_local, edx                   ; Cache token index for embedding lookup
     mov ff_pos_local, r8d                     ; Cache sequence position for KV-cache addressing
; === Extract struct pointers ===
     mov ff_cfg_ptr, rcx                       ; Config* at offset 0
     lea rax, [rcx + TRANSFORMER_WEIGHTS]      ; TransformerWeights* at offset 64
     mov ff_w_ptr, rax                         ; Store TransformerWeights* for weight access
     lea rax, [rcx + TRANSFORMER_STATE]        ; RunState* at offset 184
     mov r13, rax                              ; Cache RunState* in non-volatile reg r13, preserved across API_CALLs
     lea r12, ff_tensor_data_raw               ; Cache tensor_f32 struct slot address in non-volatile reg r12
; === Cache config values ===
     mov rbx, rcx                              ; rbx = Config*, non-volatile; rcx unchanged since the ff_cfg_ptr store above
     mov eax, [rbx + CFG_DIM]                  ; Load dim, model embedding dimension from Config
     mov ff_dim, eax                           ; Cache dim for subsequent operations
     shl eax, 2                                ; eax = dim * 4, memcpy byte count
     mov ff_dim_bytes, eax                     ; Cache precomputed dim_bytes for embedding copy sites
     mov eax, [rbx + CFG_HIDDEN_DIM]           ; Load hidden_dim, FFN intermediate dimension
     mov ff_hidden_dim, eax                    ; Cache hidden_dim for SwiGLU operations
     mov eax, [rbx + CFG_N_LAYERS]             ; Load n_layers, number of transformer layers
     mov ff_n_layers, eax                      ; Cache n_layers for layer loop bounds
     mov eax, [rbx + CFG_N_HEADS]              ; Load n_heads, number of query heads
     mov ff_n_heads, eax                       ; Cache n_heads for attention head computation
     mov eax, [rbx + CFG_N_KV_HEADS]           ; Load n_kv_heads, number of key/value heads
     mov ff_n_kv_heads, eax                    ; Cache n_kv_heads for GQA computation
     mov eax, [rbx + CFG_VOCAB_SIZE]           ; Load vocab_size for classifier output dimension
     mov ff_vocab_size, eax                    ; Cache vocab_size for final projection
     mov eax, [rbx + CFG_SEQ_LEN]              ; Load seq_len, maximum sequence length
     mov ff_seq_len, eax                       ; Cache seq_len for KV-cache addressing
     mov eax, [rbx + CFG_HEAD_DIM]             ; Load head_dim, dimension per attention head
     mov ff_head_dim, eax                      ; Cache head_dim for attention kernel dispatch
     mov eax, [rbx + CFG_HAS_QK_NORM]          ; Load has_qk_norm, 1=Qwen3 QK-RMSNorm, 0=Llama/SmolLM2
     mov ff_has_qk_norm, eax                   ; Cache has_qk_norm for Q/K projection decisions
     mov rbx, ff_w_ptr                         ; Repurpose rbx: cache TransformerWeights* in non-volatile reg
     mov eax, ff_n_kv_heads                    ; Load n_kv_heads, 32-bit read zero-extends rax
     mov r10d, ff_head_dim                     ; Load head_dim, 32-bit read zero-extends r10; scratch through this block
     imul rax, r10                             ; rax = kv_dim = n_kv_heads * head_dim, 64-bit product
     mov ff_kv_dim, rax                        ; kv_dim, qword slot
     mov rcx, rax                              ; rcx = kv_dim copy for the dsum fold below
     mov eax, ff_n_heads                       ; Load n_heads for kv_mul division
     xor edx, edx                              ; Zero-extend eax->edx:eax for unsigned division
     mov r11d, ff_n_kv_heads                   ; Load n_kv_heads as divisor, 32-bit read
     div r11d                                  ; kv_mul = n_heads / n_kv_heads, unsigned
     mov ff_kv_mul, eax                        ; kv_mul = n_heads / n_kv_heads
     mov eax, ff_n_heads                       ; Load n_heads, 32-bit read zero-extends rax
     imul rax, r10                             ; rax = all_heads_dim = n_heads * head_dim, 64-bit product
     mov ff_all_heads_dim, rax                 ; all_heads_dim, qword slot
     add rcx, rcx                              ; rcx = 2 * kv_dim, 64-bit
     add rax, rcx                              ; rax = all_heads_dim + 2*kv_dim = dsum, 64-bit
     mov ff_dsum, rax                          ; Cache dsum for per-layer QKV bias offset, qword slot
; kv_dim, all_heads_dim and dsum depend only on the config, which cannot change
; for the lifetime of the context. Every product is validated once at init when
; the values are first computed; none is re-checked per token. The same 64-bit
; discipline that governs the kernel ABI covers dsum too: the sum can wrap 32
; bits even when each addend passes on its own, hence the qword slot and the
; 64-bit adds above.
     call initialize_once                      ; initialize the pool before replacing its first text-row copy
     API_CALL v7s_set_spin, 1                  ; Set spin=1 before memcpy, pool threads commit to spinning on signal
; === Step 0: select source, then copy exactly one row ===
     mov eax, [rel g_image_token_id]           ; Load the image-pad token id once
     test eax, eax                             ; zero = image override inactive, text-only / post-vision-turn
     jz .ff_embed_normal                       ; No, standard token embedding stays
     cmp ff_token_local, eax                   ; Is the current token the image-pad token?
     jne .ff_embed_normal                      ; No, plain token, keep table embedding
     mov eax, [rel g_vision_emb_consumed]      ; row index, zero-extends to rax; never negative
     cmp eax, dword [rel g_vision_emb_count]   ; consumed < count?
     jge .ff_step0_exhausted                   ; no, FATAL, more image tokens than embeddings
     mov r10, [rel g_vision_embeddings]        ; base pointer, set/cleared together with the id
     mov r11d, dword ff_dim                    ; r11 = dim, sign-extended for the row product below
     imul r11, rax                             ; r11 = row * dim = embedding row offset in elements
     mov ff_ds_row_local, r11                  ; Cache full 64-bit offset for the DeepStack add
     lea rdx, [r10 + r11*FWD_F32_FLOAT_BYTES]  ; rdx = src = embeddings + row*dim*4
     mov rcx, [r13]                            ; rcx = s->x as memcpy destination
     mov r8d, ff_dim_bytes                     ; r8 = dim*4, precomputed memcpy byte count
     API_CALL v7d_memcpy, rcx, rdx, r8         ; Copy vision embedding row into s->x
     inc dword [rel g_vision_emb_consumed]     ; Advance consumed index for next image token
     jmp .ff_step0_text                        ; Vision copy done, continue with text flow
.ff_step0_exhausted:
     FATAL "vision: embedding pool exhausted in forward_f32" ; Contract violation: more image tokens than embeddings
.ff_remove_fatal:
     FATAL "streaming: eviction log overflow, no chat session is draining it (raw forward loop with streaming enabled)"
.ff_embed_normal:
     mov rcx, [r13]                            ; output FP32 embedding row
     mov edx, ff_token_local                   ; token ID
     mov r8d, ff_dim                           ; row length in elements
     mov r9, [rbx + W_Q_TOKENS]                ; packed embedding descriptor
     call v7s_embedding_row                    ; dequantize exactly this token into the destination
.ff_step0_text:
; === Build tensor_f32 {.data=s->xb, .scales=NULL} on stack ===
     mov rcx, [r13 + S_XB]                     ; rcx = s->xb for tensor data pointer
     mov [r12], rcx                            ; Store s->xb as tensor_f32.data
     mov qword ff_tensor_scale, 0              ; scales = NULL, unquantized F32 path; single 7-byte imm64 store, an xor+mov pair would be 10
; === Hoisted per-call constants, pos and head_dim never change across layers ===
; === StreamingLLM: resolve this token's cache slot + eviction bookkeeping ===
; slot = pos, streaming off, or pos < C: linear fill
; slot = S +, pos - S mod W, wrapped: overwrite the oldest window row
; ONE g_stream_enable test covers resolver + bookkeeping + rope select.
; eax carries the slot from here to the RoPE chain; the nested-generate
; mechanism swaps g_stream_enable to 0 for the whole nested chat, so every
; consumer, slot, rope, bookkeeping, kernel, routing, checks sees the
; legacy path with all legacy backstops. One mechanism, one state word.
     V7_STREAM_RECORD_SLOT ff_pos_local, ff_token_local, .ff_remove_fatal ; inline shared ring handling, rope position follows below
; === RoPE position: the fused tables are indexed by SLOT in wrapped mode ===
; K is rotated once at its slot and stays valid forever; the query arrives
; at the same position h, the attention kernel only delta-rotates q for
; the sink/stale segments. eax still carries the slot: the off and linear
; paths hold pos, which equals the slot there, no reload, no test.
; slot != pos means WRAPPED: the rope base is the session-local slot and the
; M-RoPE compression delta must NOT fold into it, a slot can sit below
; |delta|, slot+delta would wrap the 32-bit table index negative; K and Q
; skip the delta TOGETHER, so every produced distance stays delta-free-exact
     mov ecx, eax                              ; ecx = slot, == pos when streaming off or pos < C
     cmp eax, ff_pos_local                     ; slot != pos -> wrapped regime
     jne .ff_rope_pos_done                     ; wrapped: rope = slot, no delta
     cmp qword [rel g_mrope_saved_cos], 0      ; M-RoPE scratch tables swapped in, prefill active ?
     jne .ff_rope_pos_done                     ; yes, scratch rows are indexed by raw pos, no delta
     add ecx, dword [rel g_mrope_delta]        ; no, generation: use compacted M-RoPE position, cancels in every q-k pair
.ff_rope_pos_done:
     mov edx, ff_head_dim                      ; rdx = head_dim for RoPE pair count
     API_CALL v7s_set_fused_rope, rcx, rdx     ; Configure RoPE tables once for ALL layers, hoisted out of .ff_layer_loop, pos and head_dim are layer-invariant
; === Hoist KV-cache invariants out of the layer loop ===
     mov ecx, ff_seq_len                       ; rcx = seq_len, zero-extended, positive
     mov r10d, ff_kv_dim_raw                   ; r10 = kv_dim, 32-bit read of the qword slot; zero-extends, always positive
     imul rcx, r10                             ; rcx = seq_len * kv_dim, 64-bit, no truncation
     mov ff_seq_len_kv, rcx                    ; Cache seq_len * kv_dim, float units per layer
     mov eax, [rel g_stream_slot]              ; rax = this token's cache slot, == pos when streaming off or pos < C
     imul rax, r10                             ; rax = slot * kv_dim, 64-bit, no truncation
     mov ff_pos_kv, rax                        ; Cache pos * kv_dim in float units
; === Layer loop ===
     xor eax, eax                              ; full RAX = 0, pos*kv_dim already stored to ff_pos_kv above
     mov ff_layer_idx, eax                     ; layer loop counter, SF_DWORD slot
     mov qword ff_layer_stride, rax            ; layer stride
     mov qword ff_loff, rax                    ; layer KV offset
     mov qword ff_layer_dim_off, rax           ; incremental layer*dim offset
     mov qword ff_layer_hd_off, rax            ; incremental layer*head_dim offset
     mov qword ff_layer_bias_off, rax          ; incremental layer*dsum offset
     mov rax, [rel g_qkv_bias]                 ; Cache QKV bias base once per call, BSS invariant, set at init; always valid, never NULL
     mov qword ff_bias_base, rax               ; Q/K/V sites read the frame slot below
; DeepStack gate fold, per-call constant : has_deepstack AND token==image_token.
; One flag replaces the two BSS loads plus three compares per layer.
     xor edx, edx                              ; folded gate defaults to 0
     cmp dword [rel g_cfg_has_deepstack], 0    ; DeepStack feature injection enabled?, zero = text-only model
     jz .ff_do_ds_done                         ; -> folded flag stays 0
     mov eax, [rel g_image_token_id]           ; image-pad token id
     test eax, eax                             ; zero = image override inactive, post-vision-turn
     jz .ff_do_ds_done                         ; no DeepStack row is valid
     cmp ff_token_local, eax                   ; current token is the image-pad token?
     sete dl                                   ; 1 = DeepStack runs for layers 0..2
.ff_do_ds_done:
     mov ff_do_ds, edx                         ; cache the folded gate for the layer loop
.ff_layer_loop:
; === layer_stride and loff are incrementally maintained, updated at loop bottom ===
; One FP32 staging block shared with F32 prefill, full-precision fused K and V ops.
     mov r15, [rel g_inf_kv_scratch]           ; r15 = K FP32 staging row
     mov [r13 + S_K], r15                      ; s->k = K staging row
     mov eax, ff_kv_dim_raw                    ; eax = kv_dim, V row starts one row later
     lea r14, [r15 + rax*FWD_F32_FLOAT_BYTES]  ; r14 = V FP32 staging row, K row + kv_dim floats
     mov [r13 + S_V], r14                      ; s->v = V staging row
; === RMSNorm: s->x -> s->xb ===
     mov rcx, [r13 + S_XB]                     ; rcx = s->xb as RMSNorm output buffer
     mov rdx, [r13]                            ; rdx = s->x as RMSNorm input buffer
     mov rax, [rbx + W_RMS_ATT_WEIGHT]         ; rax = w->rms_att_weight, via cached rbx
     mov r10, ff_layer_dim_off                 ; Load incremental layer*dim offset, avoids imul
     lea r8, [rax + r10*FWD_F32_FLOAT_BYTES]   ; r8 = rms_att_weight[layer], float offset
     mov r9d, ff_dim                           ; r9 = dim as vector length for RMSNorm
     API_CALL v7s_rmsnorm, rcx, rdx, r8, r9    ; RMSNorm,s->xb, s->x, rms_att_weight[l], dim
; === Fused Q: v7d_fused_f32,s->q, &xb, wq[l], dim, all_heads_dim, q_norm[l], 1 ===
; Set Q bias pointer in BSS: fused_bias_ptr = g_qkv_bias + layer_idx*dsum, Q slice starts at offset 0
     mov rax, qword ff_bias_base               ; bias base, never NULL: bias-less models get an all-zeros block at init
     mov rcx, ff_layer_bias_off                ; layer*dsum float offset, incremental
     lea rax, [rax + rcx*FWD_F32_FLOAT_BYTES]  ; rax = &qkv_bias[layer*dsum] = Q slice start
     mov [rel fused_bias_ptr], rax             ; publish Q bias pointer for fused kernel
     mov rcx, [r13 + S_Q]                      ; rcx = s->q as fused Q output buffer
     mov rdx, r12                              ; rdx = &xb_tensor as input
     mov r8, [rbx + W_WQ]                      ; r8 = w->wq, via cached rbx
     add r8, ff_layer_stride                   ; r8 = wq[layer] = wq + layer_idx * 16
     mov r9d, ff_dim                           ; n = dim, input dimension
     mov r10d, ff_all_heads_dim_raw            ; d = all_heads_dim, 32-bit read of the qword slot
     cmp dword ff_has_qk_norm, 0               ; has_qk_norm == 0?
     je .ff_q_no_norm                          ; no per-head QK norm, skip pointer construction, Llama/SmolLM2/Phi3/MiniCPM5
     mov rax, [rbx + W_Q_NORM_WEIGHTS]         ; has norm: load q_norm_weights
     mov r11, ff_layer_hd_off                  ; has norm: load incremental layer*head_dim offset, avoids imul
     lea r11, [rax + r11*FWD_F32_FLOAT_BYTES]  ; has norm: form q_norm_weights[layer]
     jmp .ff_q_call                            ; pointer valid, fall into the Q call
.ff_q_no_norm:
     xor r11d, r11d                            ; NULL norm_weights tells fused kernel to apply RoPE only, no per-head RMSNorm
.ff_q_call:
     API_CALL v7d_fused_f32, rcx, rdx, r8, r9, r10, r11, 1 ; Fused matmul+norm+RoPE+Q-bias for Q
; === Fused K: v7d_fused_f32,s->k, &xb_tensor, wk[l], dim, kv_dim, k_norm[l], 1 ===
; Set K bias pointer: fused_bias_ptr = g_qkv_bias + layer_idx*dsum + all_heads_dim, K bias starts after Q
     mov rax, qword ff_bias_base               ; bias base, never NULL: bias-less models get an all-zeros block at init
     mov rcx, ff_all_heads_dim                 ; K bias starts after the full Q slice
     add rcx, qword ff_layer_bias_off          ; K slice = layer*dsum + all_heads_dim, floats
     lea rax, [rax + rcx*FWD_F32_FLOAT_BYTES]  ; rax = K bias byte address
     mov [rel fused_bias_ptr], rax             ; publish K bias pointer for fused kernel
     mov rcx, r15                              ; rcx = s->k as fused K output buffer, cached non-volatile
     mov rdx, r12                              ; rdx = &xb_tensor as input
     mov r8, [rbx + W_WK]                      ; r8 = w->wk, via cached rbx
     add r8, ff_layer_stride                   ; r8 = wk[layer] = wk + layer_idx * 16
     mov r9d, ff_dim                           ; n = dim, input dimension
     mov r10d, ff_kv_dim_raw                   ; d = kv_dim, 32-bit read of the qword slot
     cmp dword ff_has_qk_norm, 0               ; has_qk_norm == 0?
     je .ff_k_no_norm                          ; no per-head QK norm, skip pointer construction
     mov rax, [rbx + W_K_NORM_WEIGHTS]         ; has norm: load k_norm_weights
     mov r11, ff_layer_hd_off                  ; has norm: load incremental layer*head_dim offset, avoids imul
     lea r11, [rax + r11*FWD_F32_FLOAT_BYTES]  ; has norm: form k_norm_weights[layer]
     jmp .ff_k_call                            ; pointer valid, fall into the K call
.ff_k_no_norm:
     xor r11d, r11d                            ; NULL norm_weights tells fused kernel to apply RoPE only
.ff_k_call:
     API_CALL v7d_fused_f32, rcx, rdx, r8, r9, r10, r11, 1 ; Fused matmul+norm+RoPE+K-bias for K
; === V: v7d_f32,s->v, &xb_tensor, wv[l], dim, kv_dim + V bias add ===
; V uses v7d_f32, not fused, bias handled by the explicit add below
     mov rcx, r14                              ; rcx = s->v as V projection output buffer, cached non-volatile
     mov rdx, r12                              ; rdx = &xb_tensor as input
     mov r8, [rbx + W_WV]                      ; r8 = w->wv, via cached rbx
     add r8, ff_layer_stride                   ; r8 = wv[layer] = wv + layer_idx * 16
     mov r9d, ff_dim                           ; n = dim, input dimension
     mov r10d, ff_kv_dim_raw                   ; d = kv_dim, 32-bit read of the qword slot
     API_CALL v7d_f32, rcx, rdx, r8, r9, r10   ; F32 matmul for V projection, no RoPE
     cmp dword [rel g_qkv_bias_active], 0      ; bias block contains real data?
     je .ff_v_bias_done                        ; no, skip the whole block: for bias-less models the address computation below AND the loop operands are unused on arrival, every layer, every token
     mov rax, qword ff_bias_base               ; bias base, never NULL: bias-less models get an all-zeros block at init
     mov rcx, ff_all_heads_dim                 ; V bias starts after the full Q slice
     mov r10, ff_kv_dim                        ; kv_dim = Q+K slice width, skipped together
     add rcx, r10                              ; + kv_dim, skip Q+K slices
     add rcx, qword ff_layer_bias_off          ; + layer*dsum -> V slice, floats
     lea rax, [rax + rcx*FWD_F32_FLOAT_BYTES]  ; rax = V bias byte address
     mov rdx, r14                              ; rdx = s->v, V output, cached in r14
     mov ecx, ff_kv_dim_raw                    ; ecx = kv_dim, 32-bit read of the qword slot; element count
.ff_v_bias_loop:
     vmovups ymm0, [rdx]                       ; load first 8 V output floats
     vmovups ymm1, [rdx + Q_8FLT_STRIDE]       ; load next 8 V output floats
     vmovups ymm2, [rdx + Q_16FLT_STRIDE]      ; load third 8 V output floats
     vmovups ymm3, [rdx + Q_24FLT_STRIDE]      ; load fourth 8 V output floats
     vaddps ymm0, ymm0, [rax]                  ; add first 8 bias floats
     vaddps ymm1, ymm1, [rax + Q_8FLT_STRIDE]  ; add next 8 bias floats
     vaddps ymm2, ymm2, [rax + Q_16FLT_STRIDE] ; add third 8 bias floats
     vaddps ymm3, ymm3, [rax + Q_24FLT_STRIDE] ; add fourth 8 bias floats
     vmovups [rdx], ymm0                       ; store first 8 bias-corrected V floats
     vmovups [rdx + Q_8FLT_STRIDE], ymm1       ; store next 8 bias-corrected V floats
     vmovups [rdx + Q_16FLT_STRIDE], ymm2      ; store third 8 bias-corrected V floats
     vmovups [rdx + Q_24FLT_STRIDE], ymm3      ; store fourth 8 bias-corrected V floats
     sub rax, -Q_32FLT_STRIDE                  ; advance bias pointer by 32 floats, imm8 encoding of the same +128
     sub rdx, -Q_32FLT_STRIDE                  ; advance V output pointer by 32 floats, imm8 encoding of the same +128
     sub ecx, 32                               ; decrement by 32
     jg .ff_v_bias_loop                        ; loop while remaining > 0
.ff_v_bias_done:
     mov rax, ff_loff                          ; layer KV base, element offset
     add rax, ff_pos_kv                        ; packed cache logical element offset
     mov rcx, [r13 + S_KEY_CACHE]              ; rcx = packed key cache base
     lea rcx, [rcx + rax*FWD_F32_FP16_BYTES]   ; KEY cache byte address: base + element_offset * sizeof(fp16)
     mov rdx, r15                              ; K FP32 staging row, after norm/RoPE
     mov r8d, ff_kv_dim_raw                    ; count = kv_dim elements
     API_CALL v7s_kv_pack_f16, rcx, rdx, r8    ; pack K row FP32 -> FP16 cache
     mov rax, ff_loff                          ; layer KV base, element offset
     add rax, ff_pos_kv                        ; packed cache logical element offset
     mov rcx, [r13 + S_VALUE_CACHE]            ; rcx = packed value cache base
     lea rcx, [rcx + rax*FWD_F32_FP16_BYTES]   ; VALUE cache byte address: base + element_offset * sizeof(fp16)
     mov rdx, r14                              ; V FP32 staging row, after bias
     mov r8d, ff_kv_dim_raw                    ; count = kv_dim elements
     API_CALL v7s_kv_pack_f16, rcx, rdx, r8    ; pack V row FP32 -> FP16 cache
; === Attention: v7d_attention,s, p, pos, loff, kv_dim, kv_mul ===
     mov rcx, r13                              ; arg1 = RunState*, cached r13 for attention kernel
     mov rdx, ff_cfg_ptr_raw                   ; arg2 = Config* for model hyperparameters
     mov r8d, ff_pos_local                     ; arg3 = current sequence position for causal mask
     mov r9, ff_loff                           ; arg4 = layer KV-cache offset for cache addressing
     mov r10d, ff_kv_dim_raw                   ; arg5 = KV dimension, 32-bit read of the qword slot
     mov r11d, ff_kv_mul                       ; arg6 = number of query heads per KV head, GQA
     API_CALL v7d_attention, rcx, rdx, r8, r9, r10, r11 ; Multithreaded attention via v7 scheduler
; === WO: v7d_f32,s->hb2, &xb_tensor, wo[l], all_heads_dim, dim ===
     mov rcx, [r13 + S_HB2]                    ; rcx = s->hb2 as WO output buffer
     mov rdx, r12                              ; rdx = &xb_tensor as input
     mov r8, [rbx + W_WO]                      ; r8 = w->wo, via cached rbx
     add r8, ff_layer_stride                   ; r8 = wo[layer] = wo + layer_idx * 16
     mov r9d, ff_all_heads_dim_raw             ; n = all_heads_dim, 32-bit read of the qword slot
     mov r10d, ff_dim                          ; d = dim, output back to embedding size
     API_CALL v7d_f32, rcx, rdx, r8, r9, r10   ; F32 matmul for WO output projection
; === Fused residual + FFN RMSNorm: s->x += s->hb2; s->xb = norm,s->x ===
     mov rcx, [r13 + S_XB]                     ; arg1: o = s->xb, norm output
     mov rdx, [r13]                            ; arg2: residual = s->x, updated in place
     mov r8, [r13 + S_HB2]                     ; arg3: add = s->hb2
     mov rax, [rbx + W_RMS_FFN_WEIGHT]         ; rax = w->rms_ffn_weight, via cached rbx
     mov r10, ff_layer_dim_off                 ; incremental layer*dim offset, avoids imul
     lea r9, [rax + r10*FWD_F32_FLOAT_BYTES]   ; arg4: rms_ffn_weight[layer], float offset
     mov r11d, ff_dim                          ; r11d = dim, staged: API_CALL cannot move a dword slot to rax
     API_CALL v7s_fused_add_rmsnorm, rcx, rdx, r8, r9, r11 ; fused add + RMSNorm, one pass over s->x
; === SwiGLU: v7d_swiglu,s->hb, &xb_tensor, w1[l], dim, hidden_dim, w3[l] ===
     mov rcx, [r13 + S_HB]                     ; rcx = s->hb as SwiGLU output buffer
     mov rdx, r12                              ; rdx = &xb_tensor as input
     mov r8, [rbx + W_W1]                      ; r8 = w->w1, via cached rbx
     add r8, ff_layer_stride                   ; r8 = w1[layer] = w1 + layer_idx * 16
     mov r9d, ff_dim                           ; n = dim, input dimension
     mov r10d, ff_hidden_dim                   ; d = hidden_dim, intermediate dimension
     mov r11, [rbx + W_W3]                     ; r11 = w->w3, via cached rbx
     add r11, ff_layer_stride                  ; r11 = w3[layer] = w3 + layer_idx * 16
     API_CALL v7d_swiglu, rcx, rdx, r8, r9, r10, r11 ; SwiGLU,s->hb, &xb, w1[l], dim, hidden_dim, w3[l]
; === Build hb_tensor {.data=s->hb, .scales=NULL} for W2 ===
     mov rcx, [r13 + S_HB]                     ; rcx = s->hb as tensor data pointer
     mov [r12], rcx                            ; Store s->hb as tensor_f32.data for W2 input
; === W2: v7d_f32,s->xb, &hb_tensor, w2[l], hidden_dim, dim ===
     mov rcx, [r13 + S_XB]                     ; rcx = s->xb as W2 output buffer
     mov rdx, r12                              ; rdx = &hb_tensor as input
     mov r8, [rbx + W_W2]                      ; r8 = w->w2, via cached rbx
     add r8, ff_layer_stride                   ; r8 = w2[layer] = w2 + layer_idx * 16
     mov r9d, ff_hidden_dim                    ; n = hidden_dim, input from SwiGLU
     mov r10d, ff_dim                          ; d = dim, output back to embedding size
     API_CALL v7d_f32, rcx, rdx, r8, r9, r10   ; F32 matmul for W2 down-projection
; === Residual: s->x += s->xb ===
     mov rcx, [r13]                            ; rcx = s->x as base for residual add
     mov rdx, [r13 + S_XB]                     ; rdx = s->xb as residual to add
     mov r8d, ff_dim                           ; r8 = vector length for elementwise addition
     API_CALL v7s_add_inplace, rcx, rdx, r8    ; s->x += s->xb, residual connection
     cmp ff_do_ds, 0                           ; folded gate: DeepStack active for this token?
     je .ff_ds_done                            ; No, text-only path, nothing to inject
     cmp ff_layer_idx, 3                       ; Is this one of the first 3 layers?
     jge .ff_ds_done                           ; No, only layers 0/1/2 receive features
     mov rdx, ff_ds_row_local                  ; rdx = cached image-row offset, elements inside the DeepStack buffer
     mov rcx, [r13]                            ; rcx = s->x as DeepStack add destination
     mov r8d, ff_dim                           ; r8 = vector length for add_inplace
; layer_idx is guaranteed in {0,1,2} by the cmp ff_layer_idx, 3 / jge above
     mov eax, ff_layer_idx                     ; eax = layer index, zero-extended
     lea r10, [rel g_deepstack_features]       ; r10 = base of the per-layer DeepStack pointer table
     mov rax, [r10 + rax*VDS_PTR_STRIDE]       ; rax = g_deepstack_features[layer_idx] pointer, slot table is filled for layers 0..2 whenever the ds gate is set, never NULL here
     lea rdx, [rax + rdx*FWD_F32_FLOAT_BYTES]  ; rdx = ds + row*dim*4, byte address
     API_CALL v7s_add_inplace, rcx, rdx, r8    ; s->x += ds[layer][row], DeepStack feature add
.ff_ds_done:
; === Restore xb_tensor for next layer iteration ===
     mov rcx, [r13 + S_XB]                     ; rcx = current s->xb pointer
     mov [r12], rcx                            ; Update tensor_f32.data for next iteration
; === Next layer ===
     inc ff_layer_idx                          ; Advance to next transformer layer
     add qword ff_layer_stride, 16             ; Advance weight stride by sizeof,tensor_f32
     mov rax, ff_seq_len_kv                    ; Load invariant seq_len * kv_dim
     add qword ff_loff, rax                    ; Advance layer KV offset by one layer's stride
     mov eax, ff_dim                           ; Load dim for incremental offset update
     add qword ff_layer_dim_off, rax           ; Advance layer*dim offset by dim
     mov eax, ff_head_dim                      ; Load head_dim for incremental offset update
     add qword ff_layer_hd_off, rax            ; Advance layer*head_dim offset by head_dim
     mov rax, ff_dsum                          ; rax = dsum, bias row width for the incremental offset update
     add qword ff_layer_bias_off, rax          ; Advance layer*dsum offset by dsum
     mov eax, ff_layer_idx                     ; Load next layer index for the bottom test, mem-to-mem cmp is illegal
     cmp eax, ff_n_layers                      ; All layers processed?
     jl .ff_layer_loop                         ; No, next layer iteration, fall-through exits after the last
     cmp ff_cache_only, 0                      ; are these logits consumed?
     jne .return_logits                        ; replay returns after the layer KV writes
; === Final RMSNorm: s->x -> s->x ===
     mov rcx, [r13]                            ; rcx = s->x as in-place output buffer
     mov rdx, rcx                              ; rdx = same s->x as input buffer, register copy
     mov r8, [rbx + W_RMS_FINAL_WEIGHT]        ; r8 = final RMSNorm weight pointer
     mov r9d, ff_dim                           ; r9 = dim as vector length for RMSNorm
     API_CALL v7s_rmsnorm, rcx, rdx, r8, r9    ; Final RMSNorm,s->x, s->x, rms_final_weight, dim
; === Final classifier uses the active worker pool ===
; === Classifier ===
; Build x_tensor {.data=s->x, .scales=NULL}
     mov rcx, [r13]                            ; rcx = s->x as tensor_f32.data
     mov [r12], rcx                            ; Store s->x as tensor data for classifier
; v7d_f32,s->logits, &x_tensor, wcls, dim, vocab_size
     mov rcx, [r13 + S_LOGITS]                 ; rcx = s->logits as classifier output buffer
     mov rdx, r12                              ; rdx = &x_tensor as input
     mov r8, [rbx + W_WCLS]                    ; r8 = classifier weight tensor, via cached rbx
     mov r9d, ff_dim                           ; n = dim, input dimension
     mov r10d, ff_vocab_size                   ; d = vocab_size, output = vocabulary size
     API_CALL v7d_f32, rcx, rdx, r8, r9, r10   ; F32 matmul for classifier projection
; === Return s->logits ===
.return_logits:
     mov rax, [r13 + S_LOGITS]                 ; rax = s->logits pointer as return value
     vzeroupper                                ; clear the upper YMM lanes before returning to SSE-aware C callers
     SF_END rbx, r15, r12, r13, r14            ; Epilogue: restore rbx, r15, r12, r13, r14 and return
; ============================================================================
; Clean up SF_* slot names from forward_f32
; ============================================================================
%undef ff_cache_only
%undef ff_cache_only_raw
%undef FWD_F32_FLOAT_BYTES
%undef FWD_F32_FP16_BYTES
%undef ff_cfg_ptr
%undef ff_cfg_ptr_raw
%undef ff_w_ptr
%undef ff_w_ptr_raw
%undef ff_dim
%undef ff_dim_raw
%undef ff_dim_bytes
%undef ff_dim_bytes_raw
%undef ff_hidden_dim
%undef ff_hidden_dim_raw
%undef ff_n_layers
%undef ff_n_layers_raw
%undef ff_seq_len
%undef ff_seq_len_raw
%undef ff_n_heads
%undef ff_n_heads_raw
%undef ff_n_kv_heads
%undef ff_n_kv_heads_raw
%undef ff_head_dim
%undef ff_head_dim_raw
%undef ff_vocab_size
%undef ff_vocab_size_raw
%undef ff_kv_dim
%undef ff_kv_dim_raw
%undef ff_kv_mul
%undef ff_kv_mul_raw
%undef ff_all_heads_dim
%undef ff_all_heads_dim_raw
%undef ff_layer_idx
%undef ff_layer_idx_raw
%undef ff_layer_stride
%undef ff_layer_stride_raw
%undef ff_loff
%undef ff_loff_raw
%undef ff_seq_len_kv
%undef ff_seq_len_kv_raw
%undef ff_pos_kv
%undef ff_pos_kv_raw
%undef ff_pos_local
%undef ff_pos_local_raw
%undef ff_token_local
%undef ff_token_local_raw
%undef ff_ds_row_local
%undef ff_ds_row_local_raw
%undef ff_do_ds
%undef ff_do_ds_raw
%undef ff_tensor_scale
%undef ff_tensor_scale_raw
%undef ff_tensor_data
%undef ff_tensor_data_raw
%undef ff_dsum
%undef ff_dsum_raw
%undef ff_layer_dim_off
%undef ff_layer_dim_off_raw
%undef ff_layer_hd_off
%undef ff_layer_hd_off_raw
%undef ff_layer_bias_off
%undef ff_layer_bias_off_raw
%undef ff_has_qk_norm
%undef ff_has_qk_norm_raw
%undef ff_bias_base
%undef ff_bias_base_raw

