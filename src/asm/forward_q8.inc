; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .text
%define FWD_Q8_FLOAT_BYTES 4                   ; Bytes per float element, scale factor for byte-offset arithmetic
%define FWD_Q8_FP16_BYTES 2                    ; Bytes per FP16 cache element, scale factor for packed KV addressing
%define FWD_Q8_KV_TENSORS 2                    ; K and V, two tensors in the fused [wq|wk|wv] matrix
%define FQ8_TENSOR_BYTES 16                    ; sizeof,tensor_f32 per-layer weight pointer stride
%define FQ8_BIAS_STRIDE1 32                    ; bias unroll byte offset: floats 8..15 of the 32-float block
%define FQ8_BIAS_STRIDE2 64                    ; bias unroll byte offset: floats 16..23 of the 32-float block
%define FQ8_BIAS_STRIDE3 96                    ; bias unroll byte offset: floats 24..31 of the 32-float block
%define FQ8_BIAS_BLOCK 128                     ; bias unroll advance: 32 floats * 4 bytes per iteration
%define FQ8_BIAS_ELEMS 32                      ; bias unroll element count consumed per iteration

%macro FQ8_RMSNORM_QUANTIZE 1
     mov rcx, [r15 + S_XQ_Q]                   ; rcx = s->xq.q as quantized output
     mov rdx, [r15]                            ; rdx = s->x via cached r15 as RMSNorm input
     mov r8, [r15 + S_XQ_S]                    ; r8 = s->xq.s as scales output
     mov r9d, dim                              ; r9 = dim as vector length for fused kernel
     mov rax, [rbx + %1]                       ; rax = per-layer RMS weight row base
     mov r10, layer_dim_off                    ; layer_idx * dim element offset
     lea r10, [rax + r10*FWD_Q8_FLOAT_BYTES]   ; r10 = rms_weight[layer]
     API_CALL v7s_rmsnorm_quantize, rcx, rdx, r8, r9, r10 ; fused RMSNorm+quantize: x -> xq, eps from BSS
%endmacro

%macro FQ8_QK_ROPE 2
     mov edx, %2                               ; rdx = head count as RoPE head count, arg2
     mov r8d, head_dim                         ; r8 = head_dim for RoPE pair computation, arg3
     cmp dword qk_norm_flag, 0                 ; has_qk_norm == 0?, Llama/SmolLM2
     je %%rope_only                            ; yes -> RoPE only, no per-head RMSNorm
; has_qk_norm=1: full RMSNorm+RoPE
     mov rax, [rbx + %1]                       ; rax = per-head norm weight base, via cached rbx
     mov r10, layer_head_dim_off               ; Pre-computed layer_idx * head_dim offset
     lea r9, [rax + r10*FWD_Q8_FLOAT_BYTES]    ; r9 = norm_weights[layer], arg4
     mov r10, [fused_rope_cos]                 ; r10 = cosine lookup table pointer, arg5
     mov r11, [fused_rope_sin]                 ; r11 = sine lookup table pointer, arg6
     API_CALL v7s_rmsnorm_rope_heads, rcx, rdx, r8, r9, r10, r11, pos_slot, rope_max_slot ; Fused RMSNorm+RoPE on the heads
     jmp %%done                                ; Skip RoPE-only branch, full norm+RoPE done
%%rope_only:
; has_qk_norm=0: RoPE only, no norm_weight arg
     mov r9, [fused_rope_cos]                  ; r9 = cos_table, arg4, no norm_weight
     mov r10, [fused_rope_sin]                 ; r10 = sin_table, arg5
     API_CALL v7s_rope_only_heads, rcx, rdx, r8, r9, r10, pos_slot, rope_max_slot ; RoPE-only on the heads, no RMSNorm
%%done:
%endmacro

; Construct per-thread descriptor from a buffer register and a model weight key.
; Uses R9/R10/R11/RAX; no writes to the global RMSNorm dispatcher BSS.

%macro FQ8_FORK_DESC 4
     lea r9, %1                                ; r9 = descriptor slot address
     mov [r9 + qk_fork_args.buf], %2           ; store the head buffer pointer
     mov r10, [rbx + %3]                       ; r10 = per-layer norm weight base, via cached rbx
     mov r11, layer_head_dim_off               ; r11 = pre-computed layer_idx * head_dim offset
     lea r10, [r10 + r11*FWD_Q8_FLOAT_BYTES]   ; r10 = norm_weights[layer]
     mov [r9 + qk_fork_args.weight], r10       ; store the norm weight pointer
     mov r11, pos_slot                         ; r11 = sequence position
     imul r11, rope_max_slot                   ; r11 = position * rope table row stride
     mov r10, [fused_rope_cos]                 ; r10 = cosine lookup table pointer
     lea r10, [r10 + r11*FWD_Q8_FLOAT_BYTES]   ; r10 = cos + pos*rope_max_dim floats
     mov [r9 + qk_fork_args.cos], r10          ; store the pre-offset cos pointer
     mov r10, [fused_rope_sin]                 ; r10 = sine lookup table pointer
     lea r10, [r10 + r11*FWD_Q8_FLOAT_BYTES]   ; r10 = sin + pos*rope_max_dim floats
     mov [r9 + qk_fork_args.sin], r10          ; store the pre-offset sin pointer
     mov eax, head_dim                         ; eax = head_dim
     mov [r9 + qk_fork_args.head_dim], eax     ; store head_dim
     mov eax, %4                               ; eax = head count argument
     mov [r9 + qk_fork_args.heads], eax        ; store head count
     mov rax, pos_slot                         ; rax = sequence position
     mov [r9 + qk_fork_args.pos], eax          ; store sequence position
%endmacro

%macro FQ8_CRC32_X 1
     mov rsi, [r15]                            ; rsi = s->x, the hashed buffer
     mov ecx, dim_bytes                        ; model dimension in bytes, multiple of 256
     shr ecx, 3                                ; qword count, at least one
     xor eax, eax                              ; CRC32C initial state remains zero
%%crc_loop:
     crc32 rax, qword [rsi]                    ; one qword CRC32C == eight little-endian byte CRC32C steps
     add rsi, 8                                ; next qword
     dec ecx                                   ; decrease qword count
     jnz %%crc_loop                            ; repeat until all dim_bytes bytes hashed
     mov [rel %1], eax                         ; publish the hash
%endmacro
     global forward_q8                         ; Export: Q8 per-token forward pass
forward_q8:
     xor r9d, r9d                              ; public callers request logits
     SF_START forward_q8_mode, rbx, r15, rsi, r12, r13 ; Function prologue: save non-volatiles, rsi because FQ8_CRC32_X clobbers it, r12/r13 for the FP32 K/V slices
     SF_PTR cfg_ptr                            ; Config* = transformer + 0
     SF_PTR w_ptr                              ; TransformerWeights* = transformer + 64
     SF_DWORD dim                              ; Model dimension, hidden size
     SF_DWORD dim_bytes                        ; dim * 4, pre-computed byte count for embedding/memcpy sites
     SF_DWORD hidden_dim                       ; FFN hidden dimension
     SF_DWORD n_layers                         ; Number of transformer layers
     SF_DWORD seq_len                          ; Maximum sequence length
     SF_DWORD n_heads                          ; Number of query heads
     SF_DWORD n_kv_heads                       ; Number of key/value heads
     SF_DWORD head_dim                         ; Dimension per attention head
     SF_DWORD vocab_size                       ; Vocabulary size
     SF_DWORD kv_dim                           ; KV cache dimension
     SF_QWORD kv_bytes                         ; kv_dim * 4, byte count per K/V slice for the fast-string copy
     SF_DWORD kv_mul                           ; KV multiplicity factor, n_heads / n_kv_heads
     SF_QWORD all_heads_dim                    ; Total dimension across all heads, n_heads * head_dim
     SF_DWORD layer_idx                        ; Current transformer layer index
     SF_QWORD layer_bytes                      ; layer_idx * 16, pre-computed byte offset for weight tensors
     SF_QWORD loff                             ; Layer offset into the KV cache, layer_idx * seq_len * kv_dim, float elements
     SF_DWORD pos_local                        ; Current sequence position
     SF_DWORD token_local                      ; Current input token id
     SF_QWORD ds_row_local                     ; Cached vision row offset in elements, consumed * dim, for DeepStack add
     SF_QWORD rope_max_slot                    ; RoPE max dimension, stored for API_CALL
     SF_QWORD pos_slot                         ; RoPE position, stored for API_CALL
     SF_QWORD xq_ptr                           ; Pre-computed &s->xq, r15 + S_XQ_Q offset, invariant per call
     SF_DWORD qk_norm_flag                     ; has_qk_norm flag, 0=no per-head QK norm, 1=Qwen3-style
     SF_QWORD layer_dim_off                    ; Pre-computed layer_idx * dim, element offset, for RMS weight access
     SF_QWORD layer_head_dim_off               ; Pre-computed layer_idx * head_dim, element offset, for norm weight access
; Permanent FloatTensor slots for v7d_q8_fused calls, WO/W2/classifier .
; Declare scales FIRST so data sits at the lower address = struct offset 0.
     SF_QWORD ft_s                             ; FloatTensor scales pointer, zeroed once at entry = NULL, unquantized
     SF_QWORD ft_x                             ; FloatTensor data pointer, rewritten before each q8_fused call
     SF_QWORD qkv_desc_base                    ; Fused QKV descriptor array base, from g_inf_qkv_desc, built by Step 9d
     SF_QWORD dsum                             ; Fused QKV output rows = all_heads_dim + 2*kv_dim
     SF_QWORD loff_stride                      ; Per-layer KV offset stride, seq_len * kv_dim, incremental
     SF_QWORD dim_off_stride                   ; Per-layer element stride, = dim, incremental
     SF_QWORD hd_off_stride                    ; Per-layer element stride, = head_dim, incremental
     SF_QWORD pos_kv                           ; pos * kv_dim element offset, constant across layers
     SF_QWORD qkv_bias_off                     ; Incremental layer*dsum offset, QKV bias slice start, floats
     SF_DWORD fq8_do_ds                        ; Folded DeepStack gate: 1 = has_deepstack AND token==image_token, per-call constant
     SF_ARRAY fq8_fork_q_desc, qk_fork_args_size ; Immutable Q descriptor, alive until FORK_WAIT
     SF_ARRAY fq8_fork_k_desc, qk_fork_args_size ; Immutable K descriptor, called on main thread
     SF_DWORD fq8_fork_active                  ; 1 only when worker 0 owns Q processing
     SF_DWORD fq8_cache_only                   ; nonzero when only the KV writes are needed
     SF_COMMIT                                 ; Commit stack frame, lock in offsets
     mov fq8_cache_only, r9d                   ; preserve the private replay mode
; === Save inputs ===
     mov token_local, edx                      ; Cache token index for embedding lookup
     mov pos_local, r8d                        ; Cache sequence position for KV-cache addressing
     mov ft_s, 0                               ; FloatTensor.s = NULL once, reused by all three q8_fused calls
; === Extract struct pointers ===
     mov cfg_ptr, rcx                          ; Store Config* pointer, offset 0 from Transformer
     lea rax, [rcx + TRANSFORMER_WEIGHTS]      ; Compute TransformerWeights* at offset 64
     mov w_ptr, rax                            ; Store TransformerWeights* for weight access
     lea rax, [rcx + TRANSFORMER_STATE]        ; Compute RunState* at offset 184
     mov r15, rax                              ; Cache RunState* in r15, non-volatile, saves 25+ stack loads
     lea rax, [r15 + S_XQ_Q]                   ; Compute &s->xq once, invariant across the entire call
     mov xq_ptr, rax                           ; Cache &s->xq for fused QKV + SwiGLU v7d_q8 call sites
     mov rbx, rcx                              ; Transformer/Config pointer still in RCX
     mov eax, [rbx + CFG_DIM]                  ; Load dim, model embedding dimension from Config
     mov dim, eax                              ; Cache dim for subsequent operations
     shl eax, 2                                ; eax = dim * 4, byte count for memcpy sites
     mov dim_bytes, eax                        ; Cache dim*4 once for every memcpy site
     mov eax, [rbx + CFG_HIDDEN_DIM]           ; Load hidden_dim, FFN intermediate dimension
     mov hidden_dim, eax                       ; Cache hidden_dim for SwiGLU operations
     mov eax, [rbx + CFG_N_LAYERS]             ; Load n_layers, number of transformer layers
     mov n_layers, eax                         ; Cache n_layers for layer loop bounds
     mov eax, [rbx + CFG_N_HEADS]              ; Load n_heads, number of query heads
     mov n_heads, eax                          ; Cache n_heads for attention head computation
     mov eax, [rbx + CFG_N_KV_HEADS]           ; Load n_kv_heads, number of key/value heads
     mov n_kv_heads, eax                       ; Cache n_kv_heads for GQA computation
     mov eax, [rbx + CFG_VOCAB_SIZE]           ; Load vocab_size for classifier output dimension
     mov vocab_size, eax                       ; Cache vocab_size for final projection
     mov eax, [rbx + CFG_SEQ_LEN]              ; Load seq_len, maximum sequence length
     mov seq_len, eax                          ; Cache seq_len for KV-cache addressing
     mov eax, [rbx + CFG_HEAD_DIM]             ; Load head_dim, dimension per attention head
     mov head_dim, eax                         ; Cache head_dim for attention kernel dispatch
     mov eax, [rbx + CFG_HAS_QK_NORM]          ; Load has_qk_norm, 1=Qwen3 QK-RMSNorm, 0=Llama/SmolLM2
     mov qk_norm_flag, eax                     ; Cache has_qk_norm for Q/K post-processing decisions
     mov rbx, w_ptr                            ; Repurpose rbx: cache TransformerWeights* pointer, non-volatile, callee-saved
     mov eax, n_kv_heads                       ; Load n_kv_heads, 32-bit read zero-extends rax
     mov r10d, head_dim                        ; Load head_dim, 32-bit read zero-extends r10
     imul rax, r10                             ; kv_dim = n_kv_heads * head_dim, fits 32 bits, stored to dword slot below
     mov kv_dim, eax                           ; Cache kv_dim, dword for the K/V projection dimension
     shl rax, 2                                ; kv_dim * 4 = byte count for the K/V copy
     mov kv_bytes, rax                         ; Cache the byte count for the fast-string K/V copy
     mov eax, n_heads                          ; Load n_heads for kv_mul computation
     xor edx, edx                              ; Zero-extend eax->edx:eax for unsigned division
     div n_kv_heads                            ; kv_mul = n_heads / n_kv_heads, unsigned; n_kv_heads is a dword stack slot, DIV takes it directly
     mov kv_mul, eax                           ; Cache kv_mul for GQA head grouping
     mov eax, n_heads                          ; Load n_heads, 32-bit read zero-extends rax
     imul rax, r10                             ; all_heads_dim = n_heads * head_dim, 64-bit product
     mov all_heads_dim, rax                    ; Cache all_heads_dim, 64-bit for WO projection
; === Load fused QKV descriptor base, built by v7s_inference_create Step 9d ===
     mov rax, [rel g_inf_qkv_desc]             ; rax = fused QKV descriptor array base, always built before any forward can run
     mov qkv_desc_base, rax                    ; Cache fused descriptor base for the per-layer fused call
     mov rax, all_heads_dim                    ; Load all_heads_dim, 64-bit for dsum computation
     mov ecx, kv_dim                           ; Load kv_dim, zero-extends to 64-bit, always positive
     lea r10, [rax + rcx*FWD_Q8_KV_TENSORS]    ; r10 = all_heads_dim + 2*kv_dim, fused QKV row count
     mov dsum, r10                             ; Cache fused output dimension for the fused call
; === Signal pool threads, BEFORE first dispatch so they spin from the start ===
     call initialize_once                      ; initialize the pool before replacing its first text-row copy
     API_CALL v7s_set_spin, 1                  ; Set spin=1 before memcpy, pool threads commit to spinning on signal
; === Step 0: choose vision or normal embedding BEFORE copying ===
     mov eax, [rel g_image_token_id]           ; load the image-pad token id once
     test eax, eax                             ; zero = image override inactive, text-only / post-vision-turn
     jz .fq8_embed_normal                      ; no image token possible, keep the table embedding
     cmp token_local, eax                      ; Is the current token the image-pad token?
     jne .fq8_embed_normal                     ; No, plain token: keep the table embedding
     mov r10d, [rel g_vision_emb_consumed]     ; r10 = next vision embedding row index
     cmp r10d, dword [rel g_vision_emb_count]  ; consumed index vs pool count, compared in memory
     jl .fq8_step0_ok                          ; Index in range, proceed with vision copy
     FATAL "vision: embedding pool exhausted in forward_q8" ; Contract violation: more image tokens than embeddings
.fq8_remove_fatal:
     FATAL "streaming: eviction log overflow, no chat session is draining it (raw forward loop with streaming enabled)"
.fq8_step0_ok:
     mov eax, dword dim                        ; rax = dim, zero-extended
     imul r10, rax                             ; r10 = consumed * dim = vision row offset in elements
     mov ds_row_local, r10                     ; Cache full 64-bit offset for DeepStack
     mov rcx, [rel g_vision_embeddings]        ; rcx = vision embeddings base pointer
     lea rdx, [rcx + r10*FWD_Q8_FLOAT_BYTES]   ; rdx = src = embeddings + consumed*dim*4
     mov rcx, [r15]                            ; rcx = s->x as memcpy destination
     mov r8d, dim_bytes                        ; r8 = dim * 4, pre-computed byte count
     API_CALL v7d_memcpy, rcx, rdx, r8         ; Copy vision embedding row into s->x
     inc dword [rel g_vision_emb_consumed]     ; Advance consumed index for next image token
     jmp .fq8_step0_text                       ; Vision copy done, continue with text flow
.fq8_embed_normal:
     mov rcx, [r15]                            ; output FP32 embedding row
     mov edx, token_local                      ; token ID
     mov r8d, dim                              ; row length in elements
     mov r9, [rbx + W_Q_TOKENS]                ; packed embedding descriptor
     call v7s_embedding_row                    ; dequantize exactly this token into the destination
.fq8_step0_text:
     mov eax, [rope_max_dim]                   ; Load rope_max_dim value, 32-bit
     mov rope_max_slot, rax                    ; Save to stable stack slot, zero-extended for 64-bit ABI
; === StreamingLLM: resolve this token's cache slot + eviction bookkeeping ===
; Identical fused shape to forward_f32, resolver + bookkeeping + publish in
; ONE g_stream_enable test.eax carries the slot into the mrope chain below,
; so pos_slot = slot + delta automatically. The eviction ring and pending log
; are shared with F32, both paths must track, or a mixed session loses
; evictions.
     V7_STREAM_RECORD_SLOT pos_local, token_local, .fq8_remove_fatal ; inline shared ring handling, identical to the F32 body; slot carries into the mrope chain below
; slot != pos means WRAPPED: rope = the session-local slot, NO mrope delta
; a slot can sit below |delta|, slot+delta would wrap the 32-bit table
; index negative; K and Q skip the delta TOGETHER, so every produced
; distance stays delta-free-exact
     cmp eax, pos_local                        ; slot != pos -> wrapped regime
     jne .fq8_rope_pos_done                    ; wrapped: pos_slot = slot, no delta
     cmp qword [rel g_mrope_saved_cos], 0      ; M-RoPE scratch tables swapped in, prefill active ?
     jne .fq8_rope_pos_done                    ; yes, scratch rows are indexed by raw pos, no delta
     add eax, dword [rel g_mrope_delta]        ; no, generation: use compacted M-RoPE position, cancels in every q-k pair
.fq8_rope_pos_done:
     mov pos_slot, rax                         ; pos is saved to a stable qword slot, survives rax overwriting below
; === Layer loop ===
     mov layer_idx, 0                          ; Initialize layer loop counter to first layer
; Incremental per-layer offsets: compute strides ONCE, init slots to layer-0
; values, advance by +stride at each loop tail.
     mov ecx, seq_len                          ; Load seq_len, zero-extends, always positive
     mov r10d, kv_dim                          ; Load kv_dim, zero-extends, always positive
     imul rcx, r10                             ; rcx = seq_len * kv_dim, 64-bit
     mov loff_stride, rcx                      ; Save KV-cache offset stride, float units
     mov ecx, dim                              ; dim, zero-extended: positive config dim, declare-time int -> 64-bit
     mov dim_off_stride, rcx                   ; Save element stride for RMS weight offsets
     mov ecx, head_dim                         ; head_dim, zero-extended
     mov hd_off_stride, rcx                    ; Save element stride for norm weight offsets
     xor eax, eax                              ; 0 = layer-0 value for all offsets
     mov loff, rax                             ; loff,0 = 0
     mov layer_bytes, rax                      ; layer_bytes,0 = 0
     mov layer_dim_off, rax                    ; layer_dim_off,0 = 0
     mov layer_head_dim_off, rax               ; layer_head_dim_off,0 = 0
     mov qkv_bias_off, 0                       ; qkv_bias_off,0 = 0, incremental offset for QKV bias
     mov eax, [rel g_stream_slot]              ; eax = this token's cache slot, == pos when streaming off or pos < C
     imul rax, r10                             ; rax = slot * kv_dim, 64-bit, layer-invariant
     mov pos_kv, rax                           ; Cache for KV pointer setup, once per call
     mov rsi, [r15 + S_Q]                      ; rsi = s->q QKV output base
     mov rax, all_heads_dim                    ; Load Q block length, floats
     lea r12, [rsi + rax*FWD_Q8_FLOAT_BYTES]   ; r12 = &s->q[all_heads_dim] = K FP32 slice base, loop-invariant
     mov r13, r12                              ; r13 = V FP32 slice base
     add r13, kv_bytes                         ; V slice follows K by kv_dim floats
     mov [r15 + S_K], r12                      ; s->k = K FP32 slice
     mov [r15 + S_V], r13                      ; s->v = V FP32 slice
; DeepStack gate fold, per-call constant : has_deepstack AND token==image_token.
; One flag replaces the two BSS loads plus three compares per layer.
     xor edx, edx                              ; folded gate defaults to 0
     cmp dword [rel g_cfg_has_deepstack], 0    ; DeepStack feature injection enabled?, zero = text-only model
     jz .fq8_ds_fold_done                      ; -> folded flag stays 0
     mov eax, [rel g_image_token_id]           ; image-pad token id
     test eax, eax                             ; zero = image override inactive, post-vision-turn
     jz .fq8_ds_fold_done                      ; no DeepStack row is valid
     cmp token_local, eax                      ; current token is the image-pad token?
     sete dl                                   ; 1 = DeepStack runs for layers 0..2
.fq8_ds_fold_done:
     mov fq8_do_ds, edx                        ; cache the folded gate for the layer loop
.layer_loop:
; K and V will stay in the FP32 fused-QKV output until K RoPE finishes.
; Cache slots are addressed only at the final FP16 pack boundary.
     FQ8_RMSNORM_QUANTIZE W_RMS_ATT_WEIGHT     ; fused RMSNorm+quantize: x -> xq with rms_att_weight[layer]
     mov rcx, [r15 + S_Q]                      ; rcx = xout = s->q, contiguous QKV output buffer
     mov rdx, xq_ptr                           ; rdx = &s->xq, pre-computed, invariant
     mov r8, qkv_desc_base                     ; r8 = fused QKV descriptor array base
     add r8, layer_bytes                       ; r8 = qkv[layer] = qkv + layer_idx * 16
     mov r9d, dim                              ; r9 = n, input dim
     mov r10, dsum                             ; r10 = d, output rows = all_heads_dim + 2*kv_dim
     API_CALL v7d_q8, rcx, rdx, r8, r9, r10    ; Q8 matmul: QKV projection, dim->all_heads_dim+2*kv_dim
     cmp dword [rel g_qkv_bias_active], 0      ; bias block contains real data?
     je .fq8_bias_done                         ; no, skip the add loop entirely
     mov rax, [rel g_qkv_bias]                 ; rax = qkv_bias base pointer, never NULL, points into the DB blob
     mov rcx, qkv_bias_off                     ; rcx = incremental offset = layer_idx * dsum, 64-bit, pre-computed
     lea rax, [rax + rcx*FWD_Q8_FLOAT_BYTES]   ; rax = &qkv_bias[layer_idx * dsum * 4]
     mov rdx, [r15 + S_Q]                      ; rdx = s->q, QKV output to add bias into
     mov ecx, dword dsum                       ; ecx = dsum, element count, multiple of 32
.q8_bias_loop:
     vmovups ymm0, [rdx]                       ; load QKV floats 0..7
     vaddps ymm0, ymm0, [rax]                  ; add bias floats 0..7
     vmovups [rdx], ymm0                       ; store back
     vmovups ymm1, [rdx + FQ8_BIAS_STRIDE1]    ; load QKV floats 8..15
     vaddps ymm1, ymm1, [rax + FQ8_BIAS_STRIDE1] ; add bias floats 8..15
     vmovups [rdx + FQ8_BIAS_STRIDE1], ymm1    ; store back
     vmovups ymm2, [rdx + FQ8_BIAS_STRIDE2]    ; load QKV floats 16..23
     vaddps ymm2, ymm2, [rax + FQ8_BIAS_STRIDE2] ; add bias floats 16..23
     vmovups [rdx + FQ8_BIAS_STRIDE2], ymm2    ; store back
     vmovups ymm3, [rdx + FQ8_BIAS_STRIDE3]    ; load QKV floats 24..31
     vaddps ymm3, ymm3, [rax + FQ8_BIAS_STRIDE3] ; add bias floats 24..31
     vmovups [rdx + FQ8_BIAS_STRIDE3], ymm3    ; store back
     sub rax, -FQ8_BIAS_BLOCK                  ; advance bias pointer, imm8 encoding of the same +128
     sub rdx, -FQ8_BIAS_BLOCK                  ; advance output pointer, imm8 encoding of the same +128
     sub ecx, FQ8_BIAS_ELEMS                   ; decrement count by 32
     jg .q8_bias_loop                          ; loop while remaining > 0
.fq8_bias_done:
; v7s_rmsnorm_rope_heads,s->q, n_heads, head_dim, q_norm_weights[l], cos, sin, pos, rope_max_dim
; pos_slot and rope_max_slot are pre-cached before the layer loop, see above .
; The fork has no disable switch and no asleep-worker bypass: the forward
; entry's v7s_set_spin(1) guarantees a fork-receptive pool for the whole call.
     mov fq8_fork_active, 0                    ; reset the per-layer fork gate
     cmp qk_norm_flag, 0                       ; RoPE-only models keep their original dispatch implementation
     je .fq8_qk_legacy                         ; has_qk_norm == 0: no per-head norm tensors to fork, config branch
     mov rax, [r15 + S_Q]                      ; rax = s->q via r15 as the Q head buffer
     FQ8_FORK_DESC fq8_fork_q_desc, rax, W_Q_NORM_WEIGHTS, n_heads ; build the Q descriptor
     FQ8_FORK_DESC fq8_fork_k_desc, r12, W_K_NORM_WEIGHTS, n_kv_heads ; build the K descriptor, r12 = s->k slice
     lea rcx, fq8_fork_q_desc                  ; only four GPRs transfer to worker 0
     FORK 0                                    ; parent ZF=1, worker 0 ZF=0
     jz .fq8_qk_parent                         ; parent continues with K heads
; Worker W0: Q heads, no access to forward_q8 stack except read-only descriptor.
     call v7s_qk_fork_heads                    ; Q heads: RMSNorm+RoPE from the Q descriptor
     CONTINUE_DISPATCH 0                       ; return worker 0 to its spin loop
.fq8_qk_parent:
     mov fq8_fork_active, 1                    ; worker 0 owns Q processing until FORK_WAIT
     lea rcx, fq8_fork_k_desc                  ; rcx = K descriptor
     call v7s_qk_fork_heads                    ; Caller: K heads; Q heads run concurrently
     jmp .fq8_qk_ready                         ; skip the legacy path
.fq8_qk_legacy:
     mov rcx, [r15 + S_Q]                      ; rcx = s->q via r15 as RoPE input/output buffer, arg1
     FQ8_QK_ROPE W_Q_NORM_WEIGHTS, n_heads     ; original per-head dispatcher path
     mov rcx, r12                              ; rcx = s->k, cached in r12 at the K/V pointer setup, preserved across API_CALLs, arg1
     FQ8_QK_ROPE W_K_NORM_WEIGHTS, n_kv_heads  ; original per-head dispatcher path
.fq8_qk_ready:
     mov rax, pos_kv                           ; packed cache row destination, element offset
     add rax, loff                             ; + layer KV offset, elements
     mov rcx, [r15 + S_KEY_CACHE]              ; rcx = packed key cache base
     lea rcx, [rcx + rax*FWD_Q8_FP16_BYTES]    ; KEY cache byte address: base + element_offset * sizeof(fp16)
     mov rdx, r12                              ; K FP32 slice, after RoPE
     mov r8d, kv_dim                           ; count = kv_dim elements
     API_CALL v7s_kv_pack_f16, rcx, rdx, r8    ; pack K slice FP32 -> FP16 cache
     mov rax, pos_kv                           ; packed cache row destination, element offset
     add rax, loff                             ; + layer KV offset, elements
     mov rcx, [r15 + S_VALUE_CACHE]            ; rcx = packed value cache base
     lea rcx, [rcx + rax*FWD_Q8_FP16_BYTES]    ; VALUE cache byte address: base + element_offset * sizeof(fp16)
     mov rdx, r13                              ; V FP32 slice, bias applied
     mov r8d, kv_dim                           ; count = kv_dim elements
     API_CALL v7s_kv_pack_f16, rcx, rdx, r8    ; pack V slice FP32 -> FP16 cache
     cmp fq8_fork_active, 0                    ; worker 0 still owns the Q heads?
     je .fq8_qk_joined                         ; no, legacy path already joined
     FORK_WAIT 0                               ; Q heads complete before attention / next regular dispatch
.fq8_qk_joined:
; === Step 4: Attention ===
     mov rcx, r15                              ; arg1 = RunState* via cached r15 for attention kernel
     mov rdx, cfg_ptr                          ; arg2 = Config* for model hyperparameters
     mov r8d, pos_local                        ; arg3 = current sequence position for causal mask
     mov r9, loff                              ; arg4 = layer KV-cache offset for cache addressing
     mov r10d, kv_dim                          ; arg5 = KV dimension for cache stride
     mov r11d, kv_mul                          ; arg6 = number of query heads per KV head, GQA
     API_CALL v7d_attention, rcx, rdx, r8, r9, r10, r11 ; Multi-head attention, KV cache + causal mask
; === Step 5: WO projection, fused quantize+Q8 ===
; v7d_q8_fused,s->hb2, &ft_x, wo[l], all_heads_dim, dim
     mov rax, [r15 + S_XB]                     ; rax = s->xb, float* activations
     mov ft_x, rax                             ; ft.x = s->xb
     mov rcx, [r15 + S_HB2]                    ; rcx = xout = s->hb2
     lea rdx, ft_x_raw                         ; rdx = &ft, FloatTensor*, .s stays NULL
     mov r8, [rbx + W_WO]                      ; r8 = w->wo, via cached rbx
     add r8, layer_bytes                       ; r8 = wo[layer] = wo + layer_idx * 16
     mov r9d, dword all_heads_dim              ; r9 = n, input dim, fits in 32 bits, zero-extends
     mov r10d, dim                             ; r10 = d, output dim, 5th arg
     API_CALL v7d_q8_fused, rcx, rdx, r8, r9, r10 ; Fused quantize+Q8 matmul: WO projection, all_heads_dim->dim
; === Step 6: Residual add ===
     mov rcx, [r15]                            ; rcx = s->x via r15 as base for residual add
     mov rdx, [r15 + S_HB2]                    ; rdx = s->hb2 via r15 as residual to add
     mov r8d, dim                              ; r8 = vector length for elementwise addition
     API_CALL v7s_add_inplace, rcx, rdx, r8    ; Residual add: x += hb2, attention output
     FQ8_RMSNORM_QUANTIZE W_RMS_FFN_WEIGHT     ; fused RMSNorm+quantize: x -> xq with rms_ffn_weight[layer]
; === Step 8: SwiGLU Q8 ===
; v7d_swiglu_q8,s->hb, &s->xq, w1[l], dim, hidden_dim, w3[l]
     mov rcx, [r15 + S_HB]                     ; rcx = s->hb via r15 as SwiGLU output buffer
     mov rdx, xq_ptr                           ; rdx = &s->xq, pre-computed, invariant
     mov r8, [rbx + W_W1]                      ; r8 = w->w1, via cached rbx
     add r8, layer_bytes                       ; r8 = w1[layer] = w1 + layer_idx * 16
     mov r9d, dim                              ; r9 = input dimension for SwiGLU matmul
     mov r11, [rbx + W_W3]                     ; r11 = w->w3, via cached rbx
     add r11, layer_bytes                      ; r11 = w3[layer] = w3 + layer_idx * 16
     mov r10d, hidden_dim                      ; r10 = hidden dimension for SwiGLU intermediate
     API_CALL v7d_swiglu_q8, rcx, rdx, r8, r9, r10, r11 ; SwiGLU Q8 matmul: FFN up+gate, dim->hidden_dim
; === Step 9: W2 projection, fused quantize+Q8 ===
; v7d_q8_fused,s->xb, &ft_hb, w2[l], hidden_dim, dim
     mov rax, [r15 + S_HB]                     ; rax = s->hb, float* SwiGLU output
     mov ft_x, rax                             ; ft.x = s->hb
     mov rcx, [r15 + S_XB]                     ; rcx = xout = s->xb
     lea rdx, ft_x_raw                         ; rdx = &ft, FloatTensor*, .s stays NULL
     mov r8, [rbx + W_W2]                      ; r8 = w->w2, via cached rbx
     add r8, layer_bytes                       ; r8 = w2[layer] = w2 + layer_idx * 16
     mov r9d, hidden_dim                       ; r9 = n, input dim
     mov r10d, dim                             ; r10 = d, output dim, 5th arg
     API_CALL v7d_q8_fused, rcx, rdx, r8, r9, r10 ; Fused quantize+Q8 matmul: W2 projection, hidden_dim->dim
; === Step 10: Residual add ===
     mov rcx, [r15]                            ; rcx = s->x via r15 as base for residual add
     mov rdx, [r15 + S_XB]                     ; rdx = s->xb via r15 as residual to add
     mov r8d, dim                              ; r8 = vector length for elementwise addition
     API_CALL v7s_add_inplace, rcx, rdx, r8    ; Residual add: x += xb, FFN output
     cmp fq8_do_ds, 0                          ; folded gate: DeepStack active for this token?
     je .fq8_ds_done                           ; No, text-only path, nothing to inject
     cmp layer_idx, 3                          ; Is this one of the first 3 layers?
     jge .fq8_ds_done                          ; No, only layers 0/1/2 receive features
     mov rax, ds_row_local                     ; rax = cached vision row offset, elements inside the DeepStack buffer
     mov rcx, [r15]                            ; rcx = s->x as DeepStack add destination
     mov r8d, dim                              ; r8 = vector length for add_inplace
     mov r9d, layer_idx                        ; r9d = layer index, guaranteed in {0,1,2} by the check above
     lea r10, [rel g_deepstack_features]       ; r10 = base of the per-layer DeepStack pointer table
     mov r9, [r10 + r9*VDS_PTR_STRIDE]         ; r9 = ds[layer_idx] tensor base pointer, indexed load; slots are filled for layers 0..2 whenever the gate is set
     lea rdx, [r9 + rax*FWD_Q8_FLOAT_BYTES]    ; rdx = ds + row*dim*4, byte address
     API_CALL v7s_add_inplace, rcx, rdx, r8    ; s->x += ds[layer][row], DeepStack feature add
.fq8_ds_done:
; === Next layer: advance incremental offsets, then loop ===
     inc layer_idx                             ; Advance layer_idx to next transformer layer
     cmp layer_idx, 1                          ; layer 0 finished, idx advanced to 1 ?
     jne .fq8_l0hash_done                      ; no: skip entirely, most common path
     cmp dword [rel enable_token_dump], 0      ; logits dump enabled?
     je .fq8_l0hash_done                       ; no: skip the hash
     FQ8_CRC32_X dbg_layer0_crc                ; layer-0 CRC32C over dim*4 bytes of s->x
.fq8_l0hash_done:
     mov rax, loff_stride                      ; Load KV stride, mem-to-mem add needs a register hop
     add loff, rax                             ; loff += seq_len * kv_dim, float units
     add qword layer_bytes, FQ8_TENSOR_BYTES   ; layer_bytes += 16, sizeof tensor_f32
     mov rax, dim_off_stride                   ; Load RMS weight element stride
     add layer_dim_off, rax                    ; layer_dim_off += dim, element units
     mov rax, hd_off_stride                    ; Load norm weight element stride
     add layer_head_dim_off, rax               ; layer_head_dim_off += head_dim, element units
     mov rax, dsum                             ; rax = dsum, QKV bias row width for the incremental offset update
     add qkv_bias_off, rax                     ; qkv_bias_off += dsum, float elements, 64-bit add
     mov eax, layer_idx                        ; Load updated layer index for bounds check
     cmp eax, n_layers                         ; Compare against total number of layers
     jl .layer_loop                            ; Continue if more layers remain
.after_layers:
     cmp fq8_cache_only, 0                     ; are these logits consumed?
     jne .return_logits                        ; replay returns after the layer KV writes
; === Step 11: Final RMSNorm ===
     mov rcx, [r15]                            ; rcx = s->x via r15 as in-place output buffer
     mov rdx, rcx                              ; rdx = same s->x as input buffer, register copy
     mov r8, [rbx + W_RMS_FINAL_WEIGHT]        ; r8 = w->rms_final_weight, via cached rbx
     mov r9d, dim                              ; r9 = dim as vector length for RMSNorm
     API_CALL v7s_rmsnorm, rcx, rdx, r8, r9    ; Final RMSNorm: x with rms_final_weight
     cmp dword [rel enable_token_dump], 0      ; logits dump enabled?
     je .fq8_nohash                            ; no: skip the byte loop entirely
     FQ8_CRC32_X dbg_hidden_crc                ; hidden-state CRC32C over dim*4 bytes of s->x
.fq8_nohash:
; === Step 12: Classifier, fused quantize+Q8 ===
; v7d_q8_fused,s->logits, &ft_x, wcls, dim, vocab_size
     mov rax, [r15]                            ; rax = s->x, float* final activations
     mov ft_x, rax                             ; ft.x = s->x
     mov rcx, [r15 + S_LOGITS]                 ; rcx = xout = s->logits
     lea rdx, ft_x_raw                         ; rdx = &ft, FloatTensor*, .s stays NULL
     mov r8, [rbx + W_WCLS]                    ; r8 = weight tensor wcls
     mov r9d, dim                              ; r9 = n, input dim
     mov r10d, vocab_size                      ; r10 = d, output dim, 5th arg
     API_CALL v7d_q8_fused, rcx, rdx, r8, r9, r10 ; Fused quantize+Q8 matmul: classifier, dim->vocab_size
; === Return s->logits ===
.return_logits:
     mov rax, [r15 + S_LOGITS]                 ; rax = s->logits pointer as return value, via cached r15
     vzeroupper                                ; clean dirty YMM uppers once at exit, callers reach WriteFile/CRT; hoisted out of the layer loop, VEX-only kernels inside
     SF_END rbx, r15, rsi, r12, r13            ; Function epilogue: restore non-volatiles and return
; ============================================================================
; Clean up SF_* slot names from forward_q8
; ============================================================================
%unmacro FQ8_RMSNORM_QUANTIZE 1                ; remove: file-local, both invocations are above
%unmacro FQ8_QK_ROPE 2                         ; remove: file-local, both invocations are above
%unmacro FQ8_FORK_DESC 4                       ; remove: file-local, both invocations are above
%undef fq8_fork_q_desc                         ; SF_ARRAY slot alias
%undef fq8_fork_k_desc                         ; SF_ARRAY slot alias
%undef fq8_fork_active                         ; SF_DWORD slot alias
%undef fq8_fork_active_raw                     ; SF_DWORD raw slot alias
%unmacro FQ8_CRC32_X 1                         ; remove: file-local, both invocations are above
%undef FQ8_BIAS_ELEMS                          ; bias unroll element count consumed per iteration
%undef FQ8_BIAS_BLOCK                          ; bias unroll advance: 32 floats * 4 bytes per iteration
%undef FQ8_BIAS_STRIDE3                        ; bias unroll byte offset: floats 24..31 of the 32-float block
%undef FQ8_BIAS_STRIDE2                        ; bias unroll byte offset: floats 16..23 of the 32-float block
%undef FQ8_BIAS_STRIDE1                        ; bias unroll byte offset: floats 8..15 of the 32-float block
%undef FQ8_TENSOR_BYTES                        ; sizeof,tensor_f32 per-layer weight pointer stride
%undef FWD_Q8_KV_TENSORS                       ; K and V, two tensors in the fused [wq|wk|wv] matrix
%undef fq8_cache_only
%undef fq8_cache_only_raw
%undef FWD_Q8_FLOAT_BYTES                      ; bytes per float element
%undef FWD_Q8_FP16_BYTES                       ; bytes per FP16 cache element
%undef cfg_ptr
%undef cfg_ptr_raw
%undef w_ptr
%undef w_ptr_raw
%undef dim
%undef dim_raw
%undef dim_bytes
%undef dim_bytes_raw
%undef hidden_dim
%undef hidden_dim_raw
%undef n_layers
%undef n_layers_raw
%undef seq_len
%undef seq_len_raw
%undef n_heads
%undef n_heads_raw
%undef n_kv_heads
%undef n_kv_heads_raw
%undef head_dim
%undef head_dim_raw
%undef vocab_size
%undef vocab_size_raw
%undef kv_dim
%undef kv_dim_raw
%undef kv_bytes
%undef kv_bytes_raw
%undef kv_mul
%undef kv_mul_raw
%undef all_heads_dim
%undef all_heads_dim_raw
%undef layer_idx
%undef layer_idx_raw
%undef layer_bytes
%undef layer_bytes_raw
%undef loff
%undef loff_raw
%undef pos_local
%undef pos_local_raw
%undef token_local
%undef token_local_raw
%undef ds_row_local
%undef ds_row_local_raw
%undef qkv_bias_off
%undef qkv_bias_off_raw
%undef rope_max_slot
%undef rope_max_slot_raw
%undef pos_slot
%undef pos_slot_raw
%undef xq_ptr
%undef xq_ptr_raw
%undef layer_dim_off
%undef layer_dim_off_raw
%undef layer_head_dim_off
%undef layer_head_dim_off_raw
%undef ft_s
%undef ft_s_raw
%undef ft_x
%undef ft_x_raw
%undef qkv_desc_base
%undef qkv_desc_base_raw
%undef dsum
%undef dsum_raw
%undef qk_norm_flag
%undef qk_norm_flag_raw
%undef loff_stride
%undef loff_stride_raw
%undef dim_off_stride
%undef dim_off_stride_raw
%undef hd_off_stride
%undef hd_off_stride_raw
%undef pos_kv
%undef pos_kv_raw
%undef fq8_do_ds
%undef fq8_do_ds_raw
; ============================================================================
; BSS: hidden-state hash published for the sampler's debug dump line
; ============================================================================
     section .bss                              ; the two debug CRC slots below
     align 4
dbg_hidden_crc:
     resd 1                                    ; CRC32C over s->x after the final RMSNorm, written when enable_token_dump
dbg_layer0_crc:
     resd 1                                    ; CRC32C over s->x after layer 0, written when enable_token_dump
     section .text
; ============================================================================
; ============================================================================

