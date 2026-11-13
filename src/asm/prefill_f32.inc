; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define PF_F4_SCALE 4                          ; Bytes per float/int32 element, for *4 indexed addressing
%define PF_FP16_BYTES 2                        ; Bytes per FP16 cache element, for packed KV addressing
%define PF_KV_PAIR_SCALE 2                     ; K+V pair multiplier for the dsum lea, 2*kv_dim scale
%define PF_YMM32 32                            ; Byte offset of the second YMM group in the unrolled V-bias loop
%define PF_YMM64 64                            ; Byte offset of the third YMM group in the unrolled V-bias loop, 2*32
%define PF_YMM96 96                            ; Byte offset of the fourth YMM group in the unrolled V-bias loop, 3*32
%define PF_VBIAS_PASS_FLOATS 32                ; floats per V-bias pass, 4 x 8 ; kv_dim is always a multiple of 32, so this stride leaves no tail
%define PF_VBIAS_PASS_BYTES 128                ; byte stride of one V-bias pass, 32 floats * 4
     global prefill_forward_f32                ; export prefill F32 entry point
prefill_forward_f32:
     SF_START prefill_forward_f32, rbx, rdi, r15, r12, r13, r14 ; prologue, save non-volatiles, r12-r14 hold hoisted wq/wk/wv bases
     SF_QWORD pf_cfg_ptr                       ; Config*
     SF_QWORD pf_w_ptr                         ; TransformerWeights*
     SF_QWORD pf_s_ptr                         ; RunState*
     SF_DWORD pf_dim                           ; model dimension
     SF_DWORD pf_hidden_dim                    ; FFN hidden dimension
     SF_DWORD pf_n_layers                      ; number of transformer layers
     SF_DWORD pf_seq_len                       ; max sequence length
     SF_DWORD pf_n_heads                       ; number of attention heads
     SF_DWORD pf_n_kv_heads                    ; number of KV heads, grouped query
     SF_DWORD pf_head_dim                      ; dimension per attention head
     SF_DWORD pf_vocab_size                    ; vocabulary size
     SF_DWORD pf_kv_dim                        ; KV dimension, n_kv_heads * head_dim
     SF_DWORD pf_kv_mul                        ; n_heads / n_kv_heads, grouped query ratio
     SF_QWORD pf_all_heads_dim                 ; n_heads * head_dim, total Q dimension, 64-bit
     SF_DWORD pf_n_tokens                      ; n_tokens argument, from r8d
     SF_DWORD pf_start_pos                     ; start_pos argument, from r9d
     SF_QWORD pf_tokens_ptr                    ; int* tokens argument, from rdx
     SF_DWORD pf_layer_idx                     ; layer loop counter l
     SF_QWORD pf_loff                          ; layer KV-cache offset
     SF_QWORD pf_layer_off_16                  ; layer_idx * 16, tensor_f32 stride for Phase 3 weight select
     SF_QWORD pf_rms_att_w_off                 ; layer_idx * dim * 4, RMS att weight byte offset
     SF_QWORD pf_rms_ffn_w_off                 ; layer_idx * dim * 4, RMS ffn weight byte offset
     SF_QWORD pf_loff_stride                   ; seq_len * kv_dim, per-layer KV offset increment, computed once
     SF_QWORD pf_rms_w_stride                  ; dim * 4, per-layer RMS weight byte-offset increment, computed once
     SF_QWORD pf_x_storage                     ; s->x_storage
     SF_QWORD pf_q_storage                     ; s->q_storage
     SF_QWORD pf_att_out_storage               ; s->att_out_storage
     SF_QWORD pf_saved_xb                      ; saved original s->xb
     SF_QWORD pf_saved_q                       ; saved original s->q
     SF_QWORD pf_saved_x                       ; saved original s->x
     SF_QWORD pf_tensor_scale                  ; tensor_f32.scales, offset 8 in struct, declare BEFORE data: SF_QWORD allocates downward, so data must be the LAST of the pair to sit at struct offset 0
     SF_QWORD pf_tensor_data                   ; tensor_f32.data, offset 0 in struct
     SF_QWORD pf_x_ptr                         ; per-token x pointer
     SF_DWORD pf_has_qk_norm                   ; has_qk_norm flag, 0=no per-head QK norm
     SF_QWORD pf_dsum                          ; dsum = all_heads_dim + 2*kv_dim, QKV bias stride per layer, 64-bit, layer_idx * dsum is an element offset
     SF_DWORD pf_emb_base                      ; vision embedding pool index at prefill entry, rows already used
     SF_DWORD pf_ds_enabled                    ; cached g_cfg_has_deepstack flag, DeepStack injection gate
     SF_DWORD pf_ds_image_token_id             ; cached g_image_token_id, image-pad token id
     SF_DWORD pf_ds_active                     ; per-layer gate: ds_enabled && image_id != 0 && layer_idx < 3
     SF_DWORD pf_bias_active                   ; cached g_qkv_bias_active, 0 = bias-less model: V-bias add is a no-op
     SF_QWORD pf_q_bias_base                   ; hoisted per-layer Q bias base, always valid: zeros block for bias-less models
     SF_QWORD pf_k_bias_base                   ; hoisted per-layer K bias base, always valid: zeros block for bias-less models
     SF_QWORD pf_v_bias_base                   ; hoisted per-layer V bias base, always valid: zeros block for bias-less models
     SF_QWORD pf_q_norm_ptr                    ; hoisted per-layer Q norm-weight pointer, NULL when no QK norm
     SF_QWORD pf_k_norm_ptr                    ; hoisted per-layer K norm-weight pointer, NULL when no QK norm
     SF_COMMIT                                 ; commit stack frame locals
     INT3_DBG_STR "pf:entry"                   ; Entry into prefill_forward_f32
; === Save arguments ===
     mov pf_n_tokens, r8d                      ; Save n_tokens, number of prompt tokens
     mov pf_start_pos, r9d                     ; Save start_pos, global KV-cache position
     mov pf_tokens_ptr, rdx                    ; Save tokens pointer for embedding lookup
; === Extract struct pointers ===
     mov pf_cfg_ptr, rcx                       ; Store Config* pointer directly from rcx
     lea rax, [rcx + TRANSFORMER_WEIGHTS]      ; Compute TransformerWeights* at offset 64
     mov pf_w_ptr, rax                         ; Store TransformerWeights* for weight access
     lea rax, [rcx + TRANSFORMER_STATE]        ; Compute RunState* at offset 184
     mov pf_s_ptr, rax                         ; Store RunState* for runtime state access
; === Cache config values ===
     mov eax, [rcx + CFG_DIM]                  ; Load dim, model embedding dimension from Config
     mov pf_dim, eax                           ; Cache dim for subsequent operations
     mov eax, [rcx + CFG_HIDDEN_DIM]           ; Load hidden_dim, FFN intermediate dimension
     mov pf_hidden_dim, eax                    ; Cache hidden_dim for SwiGLU operations
     mov eax, [rcx + CFG_N_LAYERS]             ; Load n_layers, number of transformer layers
     mov pf_n_layers, eax                      ; Cache n_layers for layer loop bounds
     mov eax, [rcx + CFG_N_HEADS]              ; Load n_heads, number of query heads
     mov pf_n_heads, eax                       ; Cache n_heads for attention head computation
     mov eax, [rcx + CFG_N_KV_HEADS]           ; Load n_kv_heads, number of key/value heads
     mov pf_n_kv_heads, eax                    ; Cache n_kv_heads for GQA computation
     mov eax, [rcx + CFG_VOCAB_SIZE]           ; Load vocab_size for classifier output dimension
     mov pf_vocab_size, eax                    ; Cache vocab_size for final projection
     mov eax, [rcx + CFG_SEQ_LEN]              ; Load seq_len, maximum sequence length
     mov pf_seq_len, eax                       ; Cache seq_len for KV-cache addressing
     mov eax, [rcx + CFG_HEAD_DIM]             ; Load head_dim, dimension per attention head
     mov pf_head_dim, eax                      ; Cache head_dim for attention kernel dispatch
     mov eax, [rcx + CFG_HAS_QK_NORM]          ; Load has_qk_norm, 1=Qwen3 QK-RMSNorm, 0=Llama/SmolLM2
     mov pf_has_qk_norm, eax                   ; Cache has_qk_norm for Q/K projection decisions
     mov eax, [rel g_cfg_has_deepstack]        ; Load DeepStack enable flag for Phase 3 gating
     mov pf_ds_enabled, eax                    ; Cache DeepStack gate, call-invariant, avoids per-token RIP read
     mov eax, [rel g_image_token_id]           ; Load image-pad token id for Phase 3 matching
     mov pf_ds_image_token_id, eax             ; Cache image token id, call-invariant, avoids per-token RIP read
     mov eax, [rel g_qkv_bias_active]          ; 1 = bias block has real data, init-time OR-scan
     mov pf_bias_active, eax                   ; cache once per call, drives the V-bias gate
     mov rcx, pf_tokens_ptr                    ; arg1: tokens array
     mov edx, pf_n_tokens                      ; arg2: token count this call
     mov r8d, pf_start_pos                     ; arg3: absolute start position
     call v7s_mrope_prepare                    ; prepare merged tables + delta, early-out if N/A
; === Compute derived values ===
     mov eax, pf_n_kv_heads                    ; Load n_kv_heads for kv_dim computation
     imul eax, pf_head_dim                     ; Multiply by head_dim for KV projection size
     mov pf_kv_dim, eax                        ; kv_dim = n_kv_heads * head_dim
     mov eax, pf_n_heads                       ; Load n_heads for kv_mul division
     xor edx, edx                              ; Zero-extend eax->edx:eax for unsigned division
     mov ecx, pf_n_kv_heads                    ; Load n_kv_heads as divisor for division
     div ecx                                   ; kv_mul = n_heads / n_kv_heads, unsigned
     mov pf_kv_mul, eax                        ; kv_mul = n_heads / n_kv_heads
     mov eax, dword pf_n_heads                 ; n_heads, zero-extended: positive config dim, 64-bit to prevent truncation
     mov ecx, dword pf_head_dim                ; head_dim, zero-extended for safe multiply
     imul rax, rcx                             ; rax = n_heads * head_dim, 64-bit, no truncation
     mov pf_all_heads_dim, rax                 ; Store full 64-bit product, no truncation
; dsum = all_heads_dim + 2*kv_dim, for QKV bias offset computation
     mov edx, pf_kv_dim                        ; edx = kv_dim, 32-bit load zero-extends, a qword add off the _raw address would read 4 bytes PAST this dword slot into the neighbor
     lea rcx, [rax + rdx*PF_KV_PAIR_SCALE]     ; rcx = all_heads_dim + 2*kv_dim, rax still holds all_heads_dim; one scaled lea
     mov pf_dsum, rcx                          ; Cache dsum for per-layer QKV bias offset, 64-bit: layer_idx * dsum below
; n_tokens/seq bounds are chat's outer-border contract, its KV-bounds FATAL, not re-checked here
     INT3_DBG_STR "pf:guards_done"             ; After guards site, guards removed
; === Get storage pointers from RunState ===
     mov rax, pf_s_ptr                         ; Load RunState* for storage field access
     mov rcx, [rax + S_X_STORAGE]              ; rcx = s->x_storage for batched embeddings
     mov pf_x_storage, rcx                     ; Cache x_storage pointer for per-token offsets
     mov rcx, [rax + S_Q_STORAGE]              ; rcx = s->q_storage for batched Q projections
     mov pf_q_storage, rcx                     ; Cache q_storage pointer for per-token offsets
     mov rcx, [rax + S_ATT_OUT]                ; rcx = s->att_out_storage for batched attention output
     mov pf_att_out_storage, rcx               ; Cache att_out_storage pointer for per-token offsets
     INT3_DBG_STR "pf:storage_ptr"             ; After storage pointers loaded
; === Worker spin contract ===
     call initialize_once                      ; initialize the pool before replacing its first text-row copy
     API_CALL v7s_set_spin, 1                  ; Workers commit to spinning before the first dispatch, memcpy embedding copies
     INT3_DBG_STR "pf:spin_done"               ; After v7s_set_spin returns
; === Build xb_tensor {s->xb, NULL} ===
     mov rax, pf_s_ptr                         ; Load RunState* for xb field access
     mov rcx, [rax + S_XB]                     ; rcx = s->xb for tensor data pointer
     mov pf_tensor_data, rcx                   ; Store s->xb as tensor_f32.data
     xor eax, eax                              ; Clear eax for NULL scales
     mov pf_tensor_scale, rax                  ; scales = NULL, tensor_f32.scales = 0
     mov eax, [rel g_vision_emb_consumed]      ; Load vision pool cursor at prefill entry
     mov pf_emb_base, eax                      ; pf_emb_base = rows consumed before this call
     xor r15d, r15d                            ; Initialize token counter t = 0 in r15d, non-volatile
     mov ebx, pf_ds_image_token_id             ; hoist image-pad id into free non-volatile rbx
     mov r12, pf_x_storage_raw                 ; r12 = running dst = x_storage + t*dim*4, strength-reduced, advanced at the loop tail
     mov r13d, dword pf_dim                    ; r13d = dim, 32-bit load: zero-extends, a 64-bit mov off the _raw address would read 4 bytes PAST this dword slot into the neighbor
     shl r13, 2                                ; r13 = per-token byte stride = dim*4
     mov rdi, pf_tokens_ptr_raw                ; rdi = tokens array, loop-invariant; rdi is not written again until well after this loop exits, and SF_START/SF_END already save/restore it
     jmp .pf_emb_chk                           ; jump to the bottom-test on entry, loop inversion
.pf_embed_loop:
     mov edx, [rdi + r15*PF_F4_SCALE]          ; Load tokens[t] as int, r15 zero-extended: 32-bit ops keep upper bits 0
     mov r14d, edx                             ; cache tokens[t] in non-volatile r14, survives API_CALL
     cmp r14d, ebx                             ; tokens[t], r14 vs image-pad id, ebx
     jne .pf_emb_normal                        ; No, every token when id==0, common path
     test ebx, ebx                             ; Matched, but id==0 means override inactive, token 0 is real
     jz .pf_emb_normal                         ; Rare: token 0 on a text-only model, keep table embedding
     mov r11d, [rel g_vision_emb_consumed]     ; r11 = next vision embedding row index
     cmp r11d, dword [rel g_vision_emb_count]  ; consumed index vs pool count, compared in memory
     jl .pf_emb_vision_ok                      ; Index in range, proceed with vision copy
     FATAL "vision: embedding pool exhausted in prefill_f32" ; Contract violation: more image tokens than embeddings
.pf_emb_vision_ok:
     mov eax, r11d                             ; 64-bit consumed row index, bounds-checked non-negative
     mov ecx, dword pf_dim                     ; dim, zero-extended: positive config dim
     imul rax, rcx                             ; rax = consumed * dim, element offset into the embedding pool
     mov rdx, rax                              ; rdx = full 64-bit element offset into embedding pool
     mov rax, [rel g_vision_embeddings]        ; rax = vision embeddings base pointer
     lea rdx, [rax + rdx*PF_F4_SCALE]          ; rdx = src = embeddings + consumed*dim*4
     mov rcx, r12                              ; rcx = dst = x_storage + t*dim*4
     mov r8, r13                               ; r8 = dim * 4, byte count
     API_CALL v7d_memcpy, rcx, rdx, r8         ; Copy vision embedding row into x_storage[t]
     inc dword [rel g_vision_emb_consumed]     ; Advance consumed index for next image token
     jmp .pf_emb_vision_done                   ; This token is fully written, continue the loop tail
.pf_emb_normal:
     mov rcx, r12                              ; output FP32 embedding row
     mov r8d, pf_dim                           ; row length in elements
     mov rax, pf_w_ptr_raw                     ; TransformerWeights base
     mov r9, [rax + W_Q_TOKENS]                ; packed embedding descriptor
     call v7s_embedding_row                    ; dequantize exactly this token into the destination
.pf_emb_vision_done:
     inc r15d                                  ; Advance token counter in register, no stack traffic!
     add r12, r13                              ; Advance the running dst by dim*4, strength-reduced
.pf_emb_chk:
     cmp r15d, pf_n_tokens                     ; Compare token counter against total, direct register cmp
     jl .pf_embed_loop                         ; Continue while more tokens remain
.pf_embed_done:
     INT3_DBG_STR "pf:embed_done"              ; Embedding loop complete
; === Layer loop ===
     mov pf_layer_idx, 0                       ; Initialize layer loop counter to first layer
     INT3_DBG_STR "pf:layer_loop_start"        ; Starting layer loop
     mov rax, pf_s_ptr_raw                     ; Load RunState* for field access
     mov rcx, [rax]                            ; rcx = original s->x pointer, offset 0
     mov pf_saved_x, rcx                       ; Save original s->x for later restoration
     mov rcx, [rax + S_XB]                     ; rcx = original s->xb pointer
     mov pf_saved_xb, rcx                      ; Save original s->xb for later restoration
     mov rcx, [rax + S_Q]                      ; rcx = original s->q pointer
     mov pf_saved_q, rcx                       ; Save original s->q for later restoration
     mov eax, dword pf_seq_len                 ; rax = seq_len, sign-extended to 64-bit
     mov ecx, dword pf_kv_dim                  ; rcx = kv_dim, sign-extended to 64-bit
     imul rax, rcx                             ; rax = seq_len * kv_dim, 64-bit
     mov pf_loff_stride, rax                   ; per-layer KV offset increment
     mov eax, dword pf_dim                     ; rax = dim, sign-extended to 64-bit
     shl rax, 2                                ; rax = dim * 4, 64-bit byte offset
     mov pf_rms_w_stride, rax                  ; per-layer RMS weight byte-offset increment
     xor eax, eax                              ; layer 0: all offsets zero
     mov pf_loff, rax                          ; loff,0 = 0, float units
     mov pf_layer_off_16, rax                  ; layer_off_16,0 = 0
     mov pf_rms_att_w_off, rax                 ; rms att weight offset,0 = 0
     mov pf_rms_ffn_w_off, rax                 ; rms ffn weight offset,0 = 0
.pf_layer_loop:
     mov eax, pf_layer_idx                     ; Load current layer index for bounds check
     cmp eax, pf_n_layers                      ; Compare against total number of layers
     jge .pf_after_layers                      ; Exit loop if all layers processed
     xor edi, edi                              ; Initialize token counter t = 0, register, API calls preserve edi
; Hoist wq/wk/wv per-layer bases for the Phase 1 token loop, r12/r13/r14, non-volatile
     mov rax, pf_w_ptr_raw                     ; Load TransformerWeights* for weight access
     mov r12, [rax + W_WQ]                     ; r12 = w->wq base tensor
     mov r13, [rax + W_WK]                     ; r13 = w->wk base tensor
     mov r14, [rax + W_WV]                     ; r14 = w->wv base tensor
     add r12, pf_layer_off_16                  ; r12 = wq[layer], cached layer_idx*16 slot, no recompute
     add r13, pf_layer_off_16                  ; r13 = wk[layer]
     add r14, pf_layer_off_16                  ; r14 = wv[layer]
; Hoist Q/K/V bias bases per layer, all inputs are layer invariants, removes
; ~25 instructions of per-token recomputes from the Phase 1 loop below
     mov rax, [rel g_qkv_bias]                 ; Load QKV bias base pointer, always valid: init points bias-less models at an all-zeros block, never NULL
     mov ecx, pf_layer_idx                     ; Load current layer index, zero-extends to rcx
     mov r11, pf_dsum                          ; r11 = dsum per layer, 64-bit; r11 free here
     imul rcx, r11                             ; rcx = layer_idx * dsum, bias row offset in floats
     lea rax, [rax + rcx*PF_F4_SCALE]          ; rax = &qkv_bias[layer_idx * dsum] = Q bias start
     mov pf_q_bias_base, rax                   ; Save Q bias base for the token loop
     mov rdx, pf_all_heads_dim                 ; K offset = Q base + all_heads_dim*4, full 64-bit read of the SF_QWORD slot, like all other readers of this slot
     lea rax, [rax + rdx*PF_F4_SCALE]          ; rax = Q base + all_heads_dim*4 = K bias start
     mov pf_k_bias_base, rax                   ; Save K bias base
     mov edx, dword pf_kv_dim                  ; kv_dim sign-extended
     lea rax, [rax + rdx*PF_F4_SCALE]          ; rax = K base + kv_dim*4 = V bias start
     mov pf_v_bias_base, rax                   ; Save V bias base
     mov rax, pf_w_ptr_raw                     ; Load TransformerWeights*
     xor r10d, r10d                            ; default = NULL, no-QK-norm models
     cmp dword pf_has_qk_norm, 0               ; model has per-head QK norm?, layer-invariant
     jne .pf_qk_ptrs                           ; yes: compute the real pointers
     mov pf_q_norm_ptr, r10                    ; park NULL
     mov pf_k_norm_ptr, r10                    ; park NULL
     jmp .pf_qk_hoist_done                     ; skip pointer computation, NULLs are already parked
.pf_qk_ptrs:
     mov r11d, pf_layer_idx                    ; Layer index
     imul r11d, pf_head_dim                    ; * head_dim, weight float offset; fits 32-bit: 28*128 max
     mov r10, [rax + W_Q_NORM_WEIGHTS]         ; r10 = w->q_norm_weights base
     lea r10, [r10 + r11*PF_F4_SCALE]          ; r10 = &q_norm_weights[layer_idx*head_dim]
     mov pf_q_norm_ptr, r10                    ; park per-layer Q norm pointer
     mov r10, [rax + W_K_NORM_WEIGHTS]         ; r10 = w->k_norm_weights base, same offset formula
     lea r10, [r10 + r11*PF_F4_SCALE]          ; r10 = &k_norm_weights[layer_idx*head_dim]
     mov pf_k_norm_ptr, r10                    ; park per-layer K norm pointer
.pf_qk_hoist_done:
     jmp .pf1_chk                              ; jump to the bottom-test on entry, loop inversion
.pf_phase1_loop:
     mov ecx, edi                              ; ecx = t, preserve across abs_pos calc
; abs_pos = start_pos + t, kept in ebx, free in Phase 1, preserved across API calls
     mov eax, pf_start_pos                     ; Load base start position
     add eax, ecx                              ; Add token offset for absolute position
     mov ebx, eax                              ; ebx = absolute position, no stack traffic
; x_ptr = x_storage + t * dim, r11 carries it to the RMSNorm arg2 below, r11 is
; dead from here until the arg7 staging at :393, so no slot round-trip is needed
     mov r11, pf_x_storage_raw                 ; r11 = x_storage base pointer
     mov r10d, pf_dim                          ; Load dim for stride computation
     imul rcx, r10                             ; rcx = t * dim, float units
     lea r11, [r11 + rcx*PF_F4_SCALE]          ; r11 = x_storage + t*dim*4, byte offset
; Project K/V into FP32 scratch. Packed cache rows are written after V bias.
     mov rcx, pf_s_ptr_raw                     ; rcx = RunState*
     mov rax, [rel g_inf_kv_scratch]           ; rax = K/V FP32 staging block base
     mov [rcx + S_K], rax                      ; s->k = K FP32 staging row
     mov edx, pf_kv_dim                        ; edx = kv_dim
     lea rax, [rax + rdx*PF_F4_SCALE]          ; V row begins one kv_dim float later
     mov [rcx + S_V], rax                      ; s->v = V FP32 staging row
; === RMSNorm: x_ptr -> s->xb ===
     mov rcx, [rcx + S_XB]                     ; rcx = s->xb as RMSNorm output buffer
     mov rdx, r11                              ; rdx = x_ptr as RMSNorm input buffer, from r11
     mov rax, pf_w_ptr_raw                     ; Load TransformerWeights* for weight access
     mov r8, [rax + W_RMS_ATT_WEIGHT]          ; r8 = w->rms_att_weight
     add r8, pf_rms_att_w_off                  ; r8 = rms_att_weight[layer], memory-operand add
     mov r9d, pf_dim                           ; r9 = dim as vector length for RMSNorm
     API_CALL v7s_rmsnorm, rcx, rdx, r8, r9    ; RMSNorm,s->xb, x_ptr, rms_att_weight[l], dim
; === Fused Q: v7s_set_fused_rope + v7d_fused_f32 ===
; Set Q bias pointer, hoisted per layer; 0 = bias-less -> fused kernel adds zeros
     mov rax, pf_q_bias_base                   ; Q bias base for this layer
     mov [rel fused_bias_ptr], rax             ; Set Q bias pointer for fused kernel
     mov ecx, ebx                              ; rcx = absolute position for RoPE frequency setup
     cmp qword [rel g_mrope_saved_cos], 0      ; M-RoPE scratch tables swapped in?
     jne .pf_rope_pos_done                     ; yes, scratch rows are indexed by raw pos, no delta
     add ecx, dword [rel g_mrope_delta]        ; no, compacted M-RoPE position, mirrors forward_f32 generation
.pf_rope_pos_done:
     mov edx, pf_head_dim                      ; rdx = head_dim for RoPE pair count
     API_CALL v7s_set_fused_rope, rcx, rdx     ; Configure RoPE tables for fused Q projection
; q_dst = q_storage + t * all_heads_dim
     mov rax, pf_q_storage_raw                 ; Load q_storage base pointer
     mov ecx, edi                              ; Load token index, register counter for offset computation
     mov r10, pf_all_heads_dim                 ; r10 = all_heads_dim: stride below AND d, output dim for the call, survives the arg staging, only rcx/rdx/r8/r9 are reloaded between
     imul rcx, r10                             ; rcx = t * all_heads_dim, float units
     lea rcx, [rax + rcx*PF_F4_SCALE]          ; rcx = q_storage + t*all_heads_dim*4, output addr, self-overwrite LEA
     lea rdx, pf_tensor_data_raw               ; rdx = &xb_tensor as input
     mov r8, r12                               ; r8 = wq[l], hoisted per-layer weight base
     mov r9d, pf_dim                           ; n = dim, input dimension
     mov r11, pf_q_norm_ptr                    ; arg7 = q_norm_weights[layer] or NULL, hoisted per layer
     mov r15d, 1                               ; is_q = 1, enable RoPE on output
     API_CALL v7d_fused_f32, rcx, rdx, r8, r9, r10, r11, r15 ; Fused matmul+RMSNorm+RoPE for Q
; === Fused K: v7d_fused_f32,s->k, &xb_tensor, wk[l], dim, kv_dim, k_norm[l], 1 ===
; Set K bias pointer, hoisted per layer
     mov rax, pf_k_bias_base                   ; K bias base for this layer
     mov [rel fused_bias_ptr], rax             ; Set K bias pointer for fused kernel
     mov rax, pf_s_ptr_raw                     ; Load RunState* for k buffer field access
     mov rcx, [rax + S_K]                      ; rcx = s->k as fused K output buffer
     lea rdx, pf_tensor_data_raw               ; rdx = &xb_tensor as input
     mov r8, r13                               ; r8 = wk[l], hoisted per-layer weight base
     mov r9d, pf_dim                           ; n = dim, input dimension
     mov r10d, pf_kv_dim                       ; d = kv_dim, output dimension
     mov r11, pf_k_norm_ptr                    ; arg7 = k_norm_weights[layer] or NULL, hoisted per layer
     mov r15d, 1                               ; is_q = 1, enable RoPE on output
     API_CALL v7d_fused_f32, rcx, rdx, r8, r9, r10, r11, r15 ; Fused matmul+RMSNorm+RoPE for K
     mov rax, pf_s_ptr_raw                     ; Load RunState* for v buffer field access
     mov rcx, [rax + S_V]                      ; rcx = s->v as V projection output buffer
     lea rdx, pf_tensor_data_raw               ; rdx = &xb_tensor as input
     mov r8, r14                               ; r8 = wv[l], hoisted per-layer weight base
     mov r9d, pf_dim                           ; n = dim, input dimension
     mov r10d, pf_kv_dim                       ; d = kv_dim, output dimension
     API_CALL v7d_f32, rcx, rdx, r8, r9, r10   ; F32 matmul for V projection, no RoPE
; === V bias add, Qwen2 attention bias; zeros for bias-less models, safe no-op ===
     cmp dword pf_bias_active, 0               ; bias block contains real data?
     je .pf_v_bias_done                        ; no, skip the dead zero-add loop
     mov rax, pf_v_bias_base                   ; V bias base, hoisted per layer; always valid, zeros block for bias-less models
     mov rdx, pf_s_ptr_raw                     ; Load RunState*
     mov rdx, [rdx + S_V]                      ; rdx = s->v, V output buffer
     mov ecx, pf_kv_dim                        ; ecx = kv_dim, element count
.pf_v_bias_loop:
     vmovups ymm0, [rdx]                       ; V output floats 0..7
     vaddps ymm0, ymm0, [rax]                  ; + bias 0..7
     vmovups [rdx], ymm0                       ; store back
     vmovups ymm1, [rdx + PF_YMM32]            ; V output floats 8..15
     vaddps ymm1, ymm1, [rax + PF_YMM32]       ; + bias 8..15
     vmovups [rdx + PF_YMM32], ymm1            ; store back
     vmovups ymm2, [rdx + PF_YMM64]            ; V output floats 16..23
     vaddps ymm2, ymm2, [rax + PF_YMM64]       ; + bias 16..23
     vmovups [rdx + PF_YMM64], ymm2            ; store back
     vmovups ymm3, [rdx + PF_YMM96]            ; V output floats 24..31
     vaddps ymm3, ymm3, [rax + PF_YMM96]       ; + bias 24..31
     vmovups [rdx + PF_YMM96], ymm3            ; store back
     sub rax, -PF_VBIAS_PASS_BYTES             ; advance bias pointer, imm8 encoding of the same +128
     sub rdx, -PF_VBIAS_PASS_BYTES             ; advance output pointer, imm8 encoding of the same +128
     sub ecx, PF_VBIAS_PASS_FLOATS             ; decrement by 32
     jg .pf_v_bias_loop                        ; Loop while remaining > 0
.pf_v_bias_done:
     mov eax, ebx                              ; absolute token position, not M-RoPE-adjusted
     mov edx, pf_kv_dim                        ; edx = kv_dim
     imul rax, rdx                             ; rax = abs_pos * kv_dim, elements
     add rax, pf_loff                          ; + layer KV offset, elements
     mov r11, pf_s_ptr_raw                     ; do NOT clobber r12, hoisted WQ!
     mov rcx, [r11 + S_KEY_CACHE]              ; rcx = packed key cache base
     lea rcx, [rcx + rax*PF_FP16_BYTES]        ; KEY cache byte address: base + element_offset * sizeof(fp16)
     mov rdx, [r11 + S_K]                      ; K FP32 staging row
     mov r8d, pf_kv_dim                        ; count = kv_dim elements
     API_CALL v7s_kv_pack_f16, rcx, rdx, r8    ; pack K row FP32 -> FP16 cache
     mov eax, ebx                              ; absolute token position, not M-RoPE-adjusted
     mov edx, pf_kv_dim                        ; edx = kv_dim
     imul rax, rdx                             ; rax = abs_pos * kv_dim, elements
     add rax, pf_loff                          ; + layer KV offset, elements
     mov r11, pf_s_ptr_raw                     ; volatile r11 was clobbered by pack call
     mov rcx, [r11 + S_VALUE_CACHE]            ; rcx = packed value cache base
     lea rcx, [rcx + rax*PF_FP16_BYTES]        ; VALUE cache byte address: base + element_offset * sizeof(fp16)
     mov rdx, [r11 + S_V]                      ; V FP32 staging row
     mov r8d, pf_kv_dim                        ; count = kv_dim elements
     API_CALL v7s_kv_pack_f16, rcx, rdx, r8    ; pack V row FP32 -> FP16 cache
; === Next Phase 1 token ===
     inc edi                                   ; Advance token counter to next position, register, no stack traffic
.pf1_chk:
     cmp edi, pf_n_tokens                      ; Compare token counter against total, direct register cmp
     jl .pf_phase1_loop                        ; Continue while more tokens remain
.pf_phase1_done:
     vzeroupper                                ; Clean AVX state once per layer, YMM0-5 volatile per ABI, dirty across calls is legal
; Token loop for attention
     xor edi, edi                              ; Initialize token counter t = 0, register, API calls preserve edi
     mov r12, pf_s_ptr_raw                     ; r12 = RunState*, invariant
     mov r13, pf_q_storage_raw                 ; r13 = q_storage cursor, base for t = 0
     mov r14, pf_att_out_storage_raw           ; r14 = att_out_storage cursor, base for t = 0
     mov rbx, pf_all_heads_dim                 ; rbx = all_heads_dim
     shl rbx, 2                                ; rbx = all_heads_dim * 4, the per-token byte stride
     mov r15, pf_cfg_ptr_raw                   ; r15 = Config*, invariant
     jmp .pf_attn_chk                          ; jump to the bottom-test on entry, loop inversion
.pf_attn_loop:
     mov [r12 + S_Q], r13                      ; s->q = this token's Q vector, cursor at base + t*stride
     mov [r12 + S_XB], r14                     ; s->xb = this token's attention output slot
     mov eax, pf_start_pos                     ; Load base start position
     add eax, edi                              ; eax = start_pos + t, abs_pos
; Call v7d_attention,s, p, start_pos + t, loff, kv_dim, kv_mul
     mov rcx, r12                              ; arg1 = RunState* for attention kernel, hoisted
     mov rdx, r15                              ; arg2 = Config* for model hyperparameters, hoisted
     mov r8d, eax                              ; arg3 = absolute position, eax live from the KV block above
     mov r9, pf_loff                           ; arg4 = layer KV-cache offset for cache addressing
     mov r10d, pf_kv_dim                       ; arg5 = KV dimension for cache stride
     mov r11d, pf_kv_mul                       ; arg6 = number of query heads per KV head, GQA
     API_CALL v7d_attention, rcx, rdx, r8, r9, r10, r11 ; Multithreaded attention via v7 scheduler
     add r13, rbx                              ; q cursor += all_heads_dim*4
     add r14, rbx                              ; att_out cursor += all_heads_dim*4
     inc edi                                   ; Advance token counter to next position, register
.pf_attn_chk:
     cmp edi, pf_n_tokens                      ; Compare token counter against total, direct register cmp
     jl .pf_attn_loop                          ; Continue while more tokens remain
.pf_attn_done:
; Restore original s->xb and s->q, s->k/s->v were never touched
     mov rax, pf_s_ptr_raw                     ; Load RunState* for field restoration
     mov rcx, pf_saved_xb                      ; Load saved original s->xb pointer
     mov [rax + S_XB], rcx                     ; Restore s->xb to original value
     mov rcx, pf_saved_q                       ; Load saved original s->q pointer
     mov [rax + S_Q], rcx                      ; Restore s->q to original value
; Rebuild xb_tensor {s->xb, NULL}
     mov rcx, [rax + S_XB]                     ; rcx = restored s->xb pointer
     mov pf_tensor_data, rcx                   ; Update tensor_f32.data to restored s->xb
; pf_tensor_scale stays NULL, written once at the init block, never modified after
     mov rax, pf_w_ptr_raw                     ; Load TransformerWeights* for weight access
     mov rbx, [rax + W_WO]                     ; rbx = wo base
     add rbx, pf_layer_off_16                  ; rbx = wo[layer]
     mov r12, [rax + W_W1]                     ; r12 = w1 base
     add r12, pf_layer_off_16                  ; r12 = w1[layer], gate_w
     mov r13, [rax + W_W3]                     ; r13 = w3 base
     add r13, pf_layer_off_16                  ; r13 = w3[layer], up_w
     mov r14, [rax + W_W2]                     ; r14 = w2 base
     add r14, pf_layer_off_16                  ; r14 = w2[layer]
     xor edi, edi                              ; Reset per-layer DeepStack image-token row counter, kept in edi
     xor r15d, r15d                            ; Initialize token counter t = 0, register, API calls preserve r15
; ds_active = deepstack enabled AND image-token override active AND layer < 3, one gate replaces 3 in-loop tests
     xor eax, eax                              ; default: inactive
     cmp dword pf_ds_enabled, 0                ; DeepStack disabled?
     je .pf3_ds_set                            ; yes: keep eax=0
     cmp dword pf_ds_image_token_id, 0         ; no image token override?
     je .pf3_ds_set                            ; yes: keep eax=0
     cmp dword pf_layer_idx, 3                 ; layer >= 3, past DeepStack taps ?
     jge .pf3_ds_set                           ; yes: keep eax=0
     mov eax, 1                                ; all three conditions met: ds_active=1
.pf3_ds_set:
     mov pf_ds_active, eax                     ; store per-layer DeepStack gate result
     jmp .pf3_chk                              ; jump to the bottom-test on entry, loop inversion
.pf_phase3_loop:
; x_ptr = x_storage + t * dim, ecx = t stays live across both stride products
     mov ecx, r15d                             ; ecx = t, register counter, zero-extended
     mov r10d, pf_dim                          ; Load dim for stride computation
     imul r10, rcx                             ; r10 = t * dim, float units
     mov rax, pf_x_storage_raw                 ; Load x_storage base pointer
     lea rax, [rax + r10*PF_F4_SCALE]          ; rax = x_storage + t*dim*4, byte offset
     mov pf_x_ptr, rax                         ; Cache per-token x pointer for residual add input
; att_ptr = att_out_storage + t * all_heads_dim
     mov r10, pf_all_heads_dim                 ; Load all_heads_dim for stride computation
     imul r10, rcx                             ; r10 = t * all_heads_dim, float units
     mov rax, pf_att_out_storage_raw           ; Load att_out_storage base pointer
     lea rax, [rax + r10*PF_F4_SCALE]          ; rax = att_out_storage + t*all_heads_dim*4
; Build att_tensor {att_ptr, NULL}
     mov pf_tensor_data, rax                   ; Store att_ptr as tensor_f32.data for WO input
; pf_tensor_scale stays NULL, written once at the init block, never modified after
; === WO: v7d_f32,s->hb2, &att_tensor, wo[l], all_heads_dim, dim ===
     mov rax, pf_s_ptr_raw                     ; Load RunState* for hb2 buffer field access
     mov rcx, [rax + S_HB2]                    ; rcx = s->hb2 as WO output buffer
     lea rdx, pf_tensor_data_raw               ; rdx = &att_tensor as input
     mov r8, rbx                               ; r8 = wo[layer], hoisted per layer
     mov r9, pf_all_heads_dim                  ; n = all_heads_dim, input from attention
     mov r10d, pf_dim                          ; d = dim, output back to embedding size
     API_CALL v7d_f32, rcx, rdx, r8, r9, r10   ; F32 matmul for WO output projection
; === Fused residual + FFN RMSNorm: x += hb2; s->xb = norm,x ===
     mov rax, pf_s_ptr_raw                     ; Load RunState* for xb and hb2 field access
     mov rcx, [rax + S_XB]                     ; arg1: o = s->xb, norm output
     mov rdx, pf_x_ptr_raw                     ; arg2: residual = per-token x pointer, updated in place
     mov r8, [rax + S_HB2]                     ; arg3: add = s->hb2
     mov rax, pf_w_ptr_raw                     ; Load TransformerWeights* for weight access
     mov rax, [rax + W_RMS_FFN_WEIGHT]         ; rax = w->rms_ffn_weight
     add rax, pf_rms_ffn_w_off                 ; rax = rms_ffn_weight[layer], memory-operand add
     mov r11d, pf_dim                          ; r11d = dim, staged: API_CALL cannot move a dword slot to rax
     API_CALL v7s_fused_add_rmsnorm, rcx, rdx, r8, rax, r11 ; fused add + RMSNorm, one pass over x[t]
; Rebuild xb_tensor {s->xb, NULL}
     mov rax, pf_s_ptr_raw                     ; Load RunState* for xb field access
     mov rcx, [rax + S_XB]                     ; rcx = current s->xb pointer
     mov pf_tensor_data, rcx                   ; Update tensor_f32.data for SwiGLU input
; pf_tensor_scale stays NULL, written once at the init block, never modified after
; === SwiGLU: v7d_swiglu,s->hb, &xb_tensor, w1[l], dim, hidden_dim, w3[l] ===
     mov rcx, [rax + S_HB]                     ; rcx = s->hb as SwiGLU output buffer
     lea rdx, pf_tensor_data_raw               ; rdx = &xb_tensor as input
     mov r8, r12                               ; r8 = w1[layer], gate_w, hoisted per layer
     mov r9d, pf_dim                           ; n = dim, input dimension
     mov r10d, pf_hidden_dim                   ; d = hidden_dim, intermediate dimension
     mov r11, r13                              ; r11 = w3[layer], up_w, hoisted per layer
     API_CALL v7d_swiglu, rcx, rdx, r8, r9, r10, r11 ; SwiGLU,s->hb, &xb, w1[l], dim, hidden_dim, w3[l]
; Build hb_tensor {s->hb, NULL}
     mov rax, pf_s_ptr_raw                     ; Load RunState* for hb field access
     mov rcx, [rax + S_HB]                     ; rcx = s->hb as tensor data pointer
     mov pf_tensor_data, rcx                   ; Store s->hb as tensor_f32.data for W2 input
; pf_tensor_scale stays NULL, written once at the init block, never modified after
; === W2: v7d_f32,s->xb, &hb_tensor, w2[l], hidden_dim, dim ===
     mov rcx, [rax + S_XB]                     ; rcx = s->xb as W2 output buffer
     lea rdx, pf_tensor_data_raw               ; rdx = &hb_tensor as input
     mov r8, r14                               ; r8 = w2[layer], hoisted per layer
     mov r9d, pf_hidden_dim                    ; n = hidden_dim, input from SwiGLU
     mov r10d, pf_dim                          ; d = dim, output back to embedding size
     API_CALL v7d_f32, rcx, rdx, r8, r9, r10   ; F32 matmul for W2 down-projection
; === Residual: x += xb ===
     mov rax, pf_s_ptr_raw                     ; Load RunState* for xb field access
     mov rcx, pf_x_ptr_raw                     ; rcx = per-token x pointer as base for residual
     mov rdx, [rax + S_XB]                     ; rdx = s->xb as residual to add
     mov r8d, pf_dim                           ; r8 = vector length for elementwise addition
     API_CALL v7s_add_inplace, rcx, rdx, r8    ; x[t] += s->xb, residual connection
     cmp dword pf_ds_active, 0                 ; Is DeepStack feature injection active for this layer?
     je .pf_ds_done                            ; No, hoisted gate, ds_enabled && image_id && layer < 3
     mov r11d, pf_ds_image_token_id            ; Load cached image-pad token id for comparison
     mov rax, pf_tokens_ptr_raw                ; Load tokens array pointer for tokens[t] access
     cmp dword [rax + r15*PF_F4_SCALE], r11d   ; is the current token the image-pad token?, tokens[t] compared in memory
     jne .pf_ds_done                           ; No, plain token, skip DeepStack
     mov r10d, pf_emb_base                     ; r10 = embedding rows consumed before this prefill
     add r10d, edi                             ; r10 = row index of this image token in DeepStack tensors
     mov eax, r10d                             ; 64-bit row index into the DeepStack tensor
     mov ecx, dword pf_dim                     ; dim, zero-extended: positive config dim
     imul rax, rcx                             ; rax = row * dim, element offset
     mov rdx, rax                              ; rdx = full 64-bit element offset into DeepStack tensor
     mov rcx, pf_x_ptr_raw                     ; rcx = per-token x pointer as DeepStack add destination
     mov r8d, pf_dim                           ; r8 = vector length for add_inplace
     mov r9d, pf_layer_idx                     ; r9 = layer index, ds[] slot, moved to the rare path
     lea rax, [rel g_deepstack_features]       ; rax = array base
     mov rax, [rax + r9*VDS_PTR_STRIDE]        ; rax = ds[layer_idx], r9 zero-extended, qword scale; slot table is filled for layers 0..2 whenever the ds gate is set, never NULL here
     lea rdx, [rax + rdx*PF_F4_SCALE]          ; rdx = ds + row*dim*4, byte address
     API_CALL v7s_add_inplace, rcx, rdx, r8    ; x[t] += ds[layer][row], DeepStack feature add
     inc edi                                   ; Advance per-layer image-token row counter
.pf_ds_done:
; === Next Phase 3 token ===
     inc r15d                                  ; Advance token counter to next position, register
.pf3_chk:
     cmp r15d, pf_n_tokens                     ; Compare token counter against total, direct register cmp
     jl .pf_phase3_loop                        ; Continue while more tokens remain
.pf_phase3_done:
; === Rebuild xb_tensor for next layer iteration ===
     mov rax, pf_s_ptr_raw                     ; Load RunState* for xb field access
     mov rcx, [rax + S_XB]                     ; rcx = s->xb, restored pointer
     mov pf_tensor_data, rcx                   ; Reset tensor data to s->xb for next layer's Q/K/V input
; pf_tensor_scale stays NULL, written once at the init block, never modified after
; === Next layer, incremental offset advance, no per-layer imul chains ===
     inc pf_layer_idx                          ; Advance to next transformer layer
     mov rax, pf_loff_stride                   ; Per-layer KV offset increment
     add pf_loff, rax                          ; loff += seq_len * kv_dim, 64-bit
     add qword pf_layer_off_16, 16             ; layer_off_16 += 16, tensor_f32 stride
     mov rax, pf_rms_w_stride                  ; Per-layer RMS weight byte offset increment
     add pf_rms_att_w_off, rax                 ; rms att offset += dim * 4
     add pf_rms_ffn_w_off, rax                 ; rms ffn offset += dim * 4
     jmp .pf_layer_loop                        ; Continue to next layer iteration
.pf_after_layers:
     INT3_DBG_STR "pf:layers_done"             ; All layers complete
; === Done with layers ===
; === Final RMSNorm on last position ===
; x_last = x_storage +, n_tokens - 1 * dim
     mov rax, pf_x_storage_raw                 ; Load x_storage base pointer
     mov ecx, pf_n_tokens                      ; Load n_tokens for last position computation
     dec ecx                                   ; ecx = n_tokens - 1, last token index
     mov r10d, pf_dim                          ; Load dim for stride computation
     imul rcx, r10                             ; rcx =, n_tokens - 1 * dim, float units
     lea rax, [rax + rcx*PF_F4_SCALE]          ; rax = x_storage +, n_tokens-1 *dim*4, last token ptr
     mov rcx, pf_s_ptr_raw                     ; Load RunState* for s->x pointer update
     mov [rcx], rax                            ; s->x = x_last, reuse s->x pointer for last token
     mov rdx, rax                              ; rdx = x_last, same as s->x now
     mov rax, pf_w_ptr_raw                     ; Load TransformerWeights* for final rms weight
     mov r8, [rax + W_RMS_FINAL_WEIGHT]        ; r8 = w->rms_final_weight
     mov r9d, pf_dim                           ; r9 = dim as vector length for RMSNorm
     API_CALL v7s_rmsnorm, rdx, rdx, r8, r9    ; Final RMSNorm on last token: s->x = RMSNorm,x_last
; === Classifier on last position ===
; Build x_tensor {s->x, NULL}
     mov rax, pf_s_ptr_raw                     ; Load RunState* for s->x pointer access
     mov rcx, [rax]                            ; rcx = s->x, last token after RMSNorm
     mov pf_tensor_data, rcx                   ; Store s->x as tensor_f32.data for classifier
; pf_tensor_scale stays NULL, written once at the init block, never modified after
     mov rcx, [rax + S_LOGITS]                 ; rcx = s->logits as classifier output buffer
     lea rdx, pf_tensor_data_raw               ; rdx = &x_tensor as input
     mov rax, pf_w_ptr_raw                     ; Load TransformerWeights* for wcls access
     mov r8, [rax + W_WCLS]                    ; r8 = classifier weight tensor
     mov r9d, pf_dim                           ; n = dim, input dimension
     mov r10d, pf_vocab_size                   ; d = vocab_size, output = vocabulary size
     API_CALL v7d_f32, rcx, rdx, r8, r9, r10   ; F32 matmul for classifier projection
; === Restore original s->x pointer, was hijacked to x_storage for final RMSNorm ===
     mov rax, pf_s_ptr_raw                     ; Load RunState* for s->x field restoration
     mov rcx, pf_saved_x                       ; Load saved original s->x pointer
     mov [rax], rcx                            ; Restore s->x to original standalone buffer
     INT3_DBG_STR "pf:return"                  ; Before returning logits
; === Return s->logits ===
     call v7s_mrope_restore                    ; put base rope tables back, no-op when not swapped
     mov rax, pf_s_ptr_raw                     ; Load RunState* for return value fetch
     mov rax, [rax + S_LOGITS]                 ; rax = s->logits pointer as return value
     SF_END rbx, rdi, r15, r12, r13, r14       ; Epilogue: restore rbx, rdi, r15, r12, r13, r14 and return
; ============================================================================
; Clean up SF_* slot names from prefill_forward_f32
; ============================================================================
%undef pf_cfg_ptr
%undef pf_cfg_ptr_raw
%undef pf_w_ptr
%undef pf_w_ptr_raw
%undef pf_s_ptr
%undef pf_s_ptr_raw
%undef pf_dim
%undef pf_dim_raw
%undef pf_hidden_dim
%undef pf_hidden_dim_raw
%undef pf_n_layers
%undef pf_n_layers_raw
%undef pf_seq_len
%undef pf_seq_len_raw
%undef pf_n_heads
%undef pf_n_heads_raw
%undef pf_n_kv_heads
%undef pf_n_kv_heads_raw
%undef pf_head_dim
%undef pf_head_dim_raw
%undef pf_vocab_size
%undef pf_vocab_size_raw
%undef pf_kv_dim
%undef pf_kv_dim_raw
%undef pf_kv_mul
%undef pf_kv_mul_raw
%undef pf_all_heads_dim
%undef pf_all_heads_dim_raw
%undef pf_n_tokens
%undef pf_n_tokens_raw
%undef pf_start_pos
%undef pf_start_pos_raw
%undef pf_tokens_ptr
%undef pf_tokens_ptr_raw
%undef pf_layer_idx
%undef pf_layer_idx_raw
%undef pf_loff
%undef pf_loff_raw
%undef pf_layer_off_16
%undef pf_layer_off_16_raw
%undef pf_rms_att_w_off
%undef pf_rms_att_w_off_raw
%undef pf_rms_ffn_w_off
%undef pf_rms_ffn_w_off_raw
%undef pf_loff_stride
%undef pf_loff_stride_raw
%undef pf_rms_w_stride
%undef pf_rms_w_stride_raw
%undef pf_x_storage
%undef pf_x_storage_raw
%undef pf_q_storage
%undef pf_q_storage_raw
%undef pf_att_out_storage
%undef pf_att_out_storage_raw
%undef pf_saved_xb
%undef pf_saved_xb_raw
%undef pf_saved_q
%undef pf_saved_q_raw
%undef pf_saved_x
%undef pf_saved_x_raw
%undef pf_tensor_scale
%undef pf_tensor_scale_raw
%undef pf_tensor_data
%undef pf_tensor_data_raw
%undef pf_x_ptr
%undef pf_x_ptr_raw
%undef pf_has_qk_norm
%undef pf_has_qk_norm_raw
%undef pf_dsum
%undef pf_dsum_raw
%undef pf_emb_base
%undef pf_emb_base_raw
%undef pf_ds_enabled
%undef pf_ds_enabled_raw
%undef pf_ds_active
%undef pf_ds_active_raw
%undef pf_ds_image_token_id
%undef pf_ds_image_token_id_raw
%undef pf_bias_active
%undef pf_bias_active_raw
%undef pf_q_bias_base
%undef pf_q_bias_base_raw
%undef pf_k_bias_base
%undef pf_k_bias_base_raw
%undef pf_v_bias_base
%undef pf_v_bias_base_raw
%undef pf_q_norm_ptr
%undef pf_q_norm_ptr_raw
%undef pf_k_norm_ptr
%undef pf_k_norm_ptr_raw
%undef PF_F4_SCALE                             ; bytes per float/int32 element
%undef PF_FP16_BYTES                           ; bytes per FP16 cache element
%undef PF_KV_PAIR_SCALE                        ; 2*kv_dim multiplier for the dsum lea
%undef PF_YMM32                                ; second YMM group offset in the unrolled V-bias loop
%undef PF_YMM64                                ; third YMM group offset in the unrolled V-bias loop
%undef PF_YMM96                                ; fourth YMM group offset in the unrolled V-bias loop
%undef PF_VBIAS_PASS_FLOATS                    ; floats consumed per V-bias pass
%undef PF_VBIAS_PASS_BYTES                     ; byte stride of one V-bias pass
