; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .text
%define INT32_SCALE 4                          ; Scale factor for int32 array indexing, *4 byte offset
%define PQ_FP16_BYTES 2                        ; Bytes per FP16 cache element, for packed KV addressing
%define PQ_KV_DOUBLED 2                        ; Scale factor for 2*kv_dim, K+V combined block
%define PQ_LAYER_TENSOR_BYTES 16               ; sizeof,tensor_f32 per-layer weight pointer stride
; QKV tiles carry four projection inputs. Partial tiles reuse the last real
; input and apply RoPE and cache writes only to real members.
%define PQ4_TILE_TOKENS 4                      ; tokens per batch-4 tile, the kernel's fixed width
%define PQ4_FORK_MEMBER1_OFF pq4_fork_args_size ; descriptor 1 byte offset in the fork member array
%define PQ4_FORK_MEMBER2_OFF (2 * pq4_fork_args_size) ; descriptor 2 byte offset
%define PQ4_FORK_MEMBER3_OFF (3 * pq4_fork_args_size) ; descriptor 3 byte offset
%define PQ4_API_PAD 128                        ; bytes between rsp and the scratch base, covers the deepest API_CALL frame in this function (8 args = 64) so the staging region below rsp never touches the scratch
%define PQ_ATT_PANEL_FRAME 80                  ; Win64 shadow, five stack arguments and alignment
%define PQ_ATT_PANEL_ARG9 64                   ; Ninth argument offset before the call
%define PQ_FFN_SIMD_BYTES 32                   ; One YMM vector of FP32 activations
%define PQ_FFN_SIMD_FLOATS 8                   ; FP32 lanes per YMM vector
%define PQ4_PROBE_PAGE 4096                    ; page size for the guard-page probe of the runtime scratch sub
%define PQ_PANEL_TILES 8                       ; Maximum four-token tiles in a projection panel
%define PQ_PANEL_TOKENS (PQ_PANEL_TILES * PQ4_TILE_TOKENS) ; Maximum tokens in the panel
; R8 is the available output workspace, RCX is one packed QKV row's byte
; width and RAX is the packed input stride. Return the bounded tile width.
%macro PQ_PANEL_CAPACITY 1
     mov r11, rax                              ; Save bytes per quantized input
     lea rax, [rcx*PQ4_TILE_TOKENS]            ; Capacity of the retired packed QKV scratch
     xor edx, edx                              ; Clear the high half of the input capacity dividend
     div r11                                   ; Number of packed inputs that fit
     mov r11, rax                              ; Retain the input token limit
     mov rax, r8                               ; Read the available output bytes
     xor edx, edx                              ; Clear the high half of the output capacity dividend
     div %1                                    ; Number of output rows that fit
     cmp rax, r11                              ; Is the output workspace the tighter constraint?
     cmova rax, r11                            ; Cap by the input workspace as well
     mov edx, PQ_PANEL_TOKENS                  ; Maximum number of staged tokens
     cmp rax, rdx                              ; Do the workspaces exceed the panel cap?
     cmova rax, rdx                            ; Limit to the descriptor array size
     and eax, Q4B_TILE_MASK                    ; Round down to complete four-token tiles
     mov edx, PQ4_TILE_TOKENS                  ; A single tile can use the original scratch
     cmp eax, edx                              ; Is the computed panel smaller than one tile?
     cmovb eax, edx                            ; Keep the single-tile route available
%endmacro

; EAX carries the token index. Leave R10 at its descriptor and EAX at its slot.
%macro PQ_PANEL_MEMBER 0
     mov r10d, eax                             ; Copy the token index
     shr r10d, 2                               ; Divide by four to select the tile
     imul r10, Q8_BATCH_ARGS_BYTES             ; Convert the tile index into a descriptor offset
     lea rdx, pq_ffn_args                      ; Address the descriptor panel
     add r10, rdx                              ; Address the tile descriptor
     and eax, PQ4_TILE_TOKENS - 1              ; Select the token's pointer slot
%endmacro

%define PQ_PREVIOUS_ROW 1                      ; Distance from the first unused slot to the last real row
%macro PQ_PAD_INPUTS 0
     mov ecx, pq_ffn_count                     ; First unused slot after the real input rows
     mov r11d, pq_ffn_tiles                    ; Number of descriptors consumed by the kernel
     shl r11d, 2                               ; Four input slots per descriptor
     cmp ecx, r11d                             ; Does the final descriptor already have four inputs?
     jae %%done                                ; All descriptors are complete
     lea eax, [rcx - PQ_PREVIOUS_ROW]          ; Index the final real input
     PQ_PANEL_MEMBER                           ; Find the valid quantized input and scales
     mov r8, [r10 + rax*Q4B_ARGS_QW + Q4B_ARGS_XQ0] ; Share the last real quantized row
     mov r9, [r10 + rax*Q4B_ARGS_QW + Q4B_ARGS_XS0] ; Share its scales
%%next:
     mov eax, ecx                              ; Select the next unused slot
     PQ_PANEL_MEMBER                           ; Locate the destination descriptor
     mov [r10 + rax*Q4B_ARGS_QW + Q4B_ARGS_XQ0], r8 ; Stage a valid input pointer
     mov [r10 + rax*Q4B_ARGS_QW + Q4B_ARGS_XS0], r9 ; Stage the matching scale pointer
     inc ecx                                   ; Advance within the final tile
     cmp ecx, r11d                             ; Any unused slots left?
     jb %%next                                 ; Fill the remaining slots
%%done:
%endmacro

%define PQ_PACK_EXPANSION 2                    ; Signed and magnitude bytes per activation byte
%define PQ_PACK_VECTOR_BYTES 32                ; Bytes in each signed or magnitude block
; Expand a quantized row backwards so the source bytes can share the destination.
; The descriptor and original width are already staged in this phase's frame.
%macro PQ_PACK_INPUT 1
     mov eax, pq4_j                            ; Read the current panel member
     PQ_PANEL_MEMBER                           ; Find this member's input pointer
     mov rcx, [r10 + rax*Q4B_ARGS_QW + Q4B_ARGS_XQ0] ; Read the quantized input base
     mov edx, dword %1                         ; Read the original row length
%%pack:
     sub edx, PQ_PACK_VECTOR_BYTES             ; Start with the last unread vector
     vmovdqu ymm0, [rcx + rdx]                 ; Load signed quantized activations
     vpabsb ymm1, ymm0                         ; Compute the unsigned magnitudes once
     vmovdqu [rcx + rdx*PQ_PACK_EXPANSION], ymm0 ; Store the signed vector at its expanded offset
     vmovdqu [rcx + rdx*PQ_PACK_EXPANSION + PQ_PACK_VECTOR_BYTES], ymm1 ; Store magnitudes next to the signed bytes
     test edx, edx                             ; Have all signed vectors been expanded?
     jnz %%pack                                ; Work backward to the next input vector
     vzeroupper                                ; Clear upper lanes before the next API call
%endmacro

; Weight member, output workspace, output width, input width, tile count.
%macro PQ_FFN_PROJECT 6
     lea rcx, pq_ffn_args                      ; Address the descriptor panel
     mov r10, rcx                              ; Walk the descriptor array
     mov r11d, %5                              ; Number of active four-token descriptors
     mov rdx, %2                               ; Start of the output workspace
%%describe:
     mov rax, [r13 + %1]                       ; Load the selected projection weight array
     add rax, pq_layer_bytes                   ; Select this layer's tensor
     mov r8, [rax + QT_Q]                      ; Load quantized weights
     mov [r10 + Q4B_ARGS_WQ], r8               ; Publish the weight bytes
     mov r8, [rax + QT_S]                      ; Load weight scales
     mov [r10 + Q4B_ARGS_WS], r8               ; Publish the scale table
     mov eax, dword %4                         ; Load the projection input width
     mov [r10 + Q4B_ARGS_N], eax               ; Set the descriptor input width
     mov eax, %3                               ; Load the projection output width
     mov [r10 + Q4B_ARGS_D], eax               ; Set the descriptor row count
     shl rax, 2                                ; Convert the output stride to bytes
     mov [r10 + Q4B_ARGS_OUT0], rdx            ; First token's output
     add rdx, rax                              ; Advance to the next output row
     mov [r10 + Q4B_ARGS_OUT1], rdx            ; Second token's output
     add rdx, rax                              ; Advance to the next output row
     mov [r10 + Q4B_ARGS_OUT2], rdx            ; Third token's output
     add rdx, rax                              ; Advance to the next output row
     mov [r10 + Q4B_ARGS_OUT3], rdx            ; Fourth token's output
     add rdx, rax                              ; Advance to the next tile's first row
     add r10, Q8_BATCH_ARGS_BYTES              ; Advance to the next tile descriptor
     dec r11d                                  ; Count the described tiles
     jnz %%describe                            ; Continue if the panel has another tile
     mov edx, %5                               ; Pass the tile count
     mov r8d, %6                               ; Tell workers how the input rows are laid out
     API_CALL v7d_q8_batch4_panel, rcx, rdx, r8 ; Process the panel with cache-sized weight slices
%endmacro

%define Q4B_ARGS_OUT0 0                        ; args.out[0], packed QKV output block for tile token 0
%define Q4B_ARGS_OUT1 8                        ; args.out[1]
%define Q4B_ARGS_OUT2 16                       ; args.out[2]
%define Q4B_ARGS_OUT3 24                       ; args.out[3]
%define Q4B_ARGS_XQ0 32                        ; args.xq[0], quantized input for tile token 0
%define Q4B_ARGS_XQ1 40                        ; args.xq[1]
%define Q4B_ARGS_XQ2 48                        ; args.xq[2]
%define Q4B_ARGS_XQ3 56                        ; args.xq[3]
%define Q4B_ARGS_XS0 64                        ; args.xs[0], activation scales for tile token 0
%define Q4B_ARGS_XS1 72                        ; args.xs[1]
%define Q4B_ARGS_XS2 80                        ; args.xs[2]
%define Q4B_ARGS_XS3 88                        ; args.xs[3]
%define Q4B_ARGS_WQ 96                         ; args.wq, packed QKV weight bytes for this layer
%define Q4B_ARGS_WS 104                        ; args.ws, packed QKV weight scales for this layer
%define Q4B_ARGS_N 112                         ; args.n, elements per input = dim
%define Q4B_ARGS_D 116                         ; args.d, packed output rows = dsum
%define Q4B_ARGS_QW 8                          ; qword stride between the per-token pointer slots
%define Q4B_ROUND 15                           ; scratch byte count rounding bias, 16-byte align
%define Q4B_ALIGN ~15                          ; scratch byte count alignment mask
%define Q4B_TILE_MASK -4                       ; tile-count mask: n_tokens & -4 = full-tile token count

; PQ_PACK_KV_F16 -- commit one FP32 K or V staging row to its FP16 cache row.
; %1 = RunState cache-field, %2 = RunState staging-field, optional %3 = the
; extra member offset of the batch-4 tile path. Reads edi (tile index or
; token counter) and the pq_* frame slots; clobbers rax/rdx/rcx/r8.

%macro PQ_PACK_KV_F16 2-3
     mov eax, pq_start_pos                     ; starting absolute position
     add eax, edi                              ; token index / tile base
%if %0 = 3
     add eax, %3                               ; only the batch-4 path has a member offset
%endif
     mov edx, pq_kv_dim                        ; elements per row
     imul rax, rdx                             ; position * kv_dim
     add rax, pq_loff                          ; layer's KV offset
     mov rcx, [r14 + %1]                       ; FP16 cache base
     lea rcx, [rcx + rax*PQ_FP16_BYTES]        ; destination row
     mov rdx, [r14 + %2]                       ; FP32 source staging row
     mov r8d, pq_kv_dim                        ; element count
     API_CALL v7s_kv_pack_f16, rcx, rdx, r8    ; pack one row
%endmacro

     SF_START prefill_forward_q8, rbx, rsi, rdi, r15, r12, r13, r14 ; Function prologue: save the 7 non-volatiles the body writes; emits global + entry label
; rsi carries g_image_token_id through the embed loop, the attention countdown, then wo[l];
; r12 carries the embedding cursor, then the attention position, WO token counter and DeepStack flag.
     SF_QWORD pq_cfg_ptr                       ; Config*
     SF_DWORD pq_dim                           ; Model embedding dimension, config
     SF_DWORD pq_hidden_dim                    ; FFN intermediate dimension, config
     SF_DWORD pq_n_layers                      ; Total transformer layers, config
     SF_DWORD pq_seq_len                       ; Maximum sequence length, config
     SF_DWORD pq_n_heads                       ; Number of query attention heads, config
     SF_DWORD pq_n_kv_heads                    ; Number of key/value attention heads, config
     SF_DWORD pq_head_dim                      ; Dimension per attention head, config
     SF_DWORD pq_vocab_size                    ; Vocabulary size, config
     SF_DWORD pq_kv_dim                        ; KV projection dimension = n_kv_heads * head_dim
     SF_DWORD pq_kv_mul                        ; GQA ratio = n_heads / n_kv_heads
     SF_QWORD pq_all_heads_dim                 ; Total attention output dim = n_heads * head_dim, 64-bit
     SF_DWORD pq_n_tokens                      ; n_tokens argument, from r8d
     SF_DWORD pq_start_pos                     ; start_pos argument, from r9d
     SF_QWORD pq_tokens_ptr                    ; int* tokens argument, from rdx
     SF_DWORD pq_layer_idx                     ; layer loop counter l
     SF_QWORD pq_layer_bytes                   ; layer_idx * 16, pre-computed byte offset
     SF_QWORD pq_layer_dim_off                 ; layer_idx * dim * 4, pre-computed RMS weight byte offset
     SF_QWORD pq_layer_hd_off                  ; layer_idx * head_dim * 4, pre-computed norm weight byte offset
     SF_QWORD pq_loff                          ; layer KV-cache offset
     SF_QWORD pq_x_storage                     ; s->x_storage
     SF_QWORD pq_q_storage                     ; s->q_storage
     SF_QWORD pq_att_out_storage               ; s->att_out_storage
     SF_QWORD pq_saved_xb                      ; saved original s->xb
     SF_QWORD pq_saved_q                       ; saved original s->q
     SF_QWORD pq_saved_x                       ; saved original s->x
     SF_QWORD pq_x_ptr                         ; per-token x pointer
     SF_QWORD pq_rope_max_slot                 ; stack slot for rope_max_dim, arg8 to rmsnorm_rope_heads
     SF_QWORD pq_pos_slot                      ; stack slot for abs_pos, arg7, stable qword source, safe for API_CALL arg staging
     SF_DWORD pq_bias_active                   ; cached g_qkv_bias_active, 0 = bias-less model: skip all 3 PQ_BIAS_ADD no-ops
     SF_DWORD pq_has_qk_norm                   ; has_qk_norm flag, 0=no per-head QK norm, 1=Qwen3-style
     SF_DWORD pq_emb_base                      ; vision embedding pool index at prefill entry, rows already used
     SF_QWORD pq_ft_s                          ; FloatTensor scales pointer, zeroed once at entry = NULL, unquantized
     SF_QWORD pq_ft_x                          ; FloatTensor data pointer, rewritten before each q8_fused call
     SF_QWORD pq_q_dst                         ; cached q_dst = q_storage + t*all_heads_dim*4, Q matmul, Q bias, RoPE
     SF_QWORD pq_q_bias_base                   ; per-layer Q bias base, g_qkv_bias + layer*dsum*4; always valid: inf_weights points bias-less models at an all-zeros block
     SF_QWORD pq_k_bias_base                   ; per-layer K bias base, = q_bias_base + all_heads_dim*4
     SF_QWORD pq_v_bias_base                   ; per-layer V bias base, = k_bias_base + kv_dim*4
     SF_QWORD pq_loff_stride                   ; per-layer KV-cache offset stride, seq_len * kv_dim, incremental
     SF_QWORD pq_dim_off_stride                ; per-layer dim-weight byte stride, dim * 4, incremental
     SF_QWORD pq_hd_off_stride                 ; per-layer head_dim-weight byte stride, head_dim * 4, incremental
     SF_QWORD pq_dsum_bytes                    ; per-layer QKV bias byte stride, dsum * 4, incremental
     SF_QWORD pq_ahd_bytes                     ; all_heads_dim * 4, Q/att per-token byte stride, shared by Phases 1/3wo/2
     SF_QWORD pq4_base                         ; batch-4 scratch base, stack block below the frame, runtime-sized
     SF_DWORD pq4_j                            ; tile member counter, stack slot: every volatile register is clobbered by the API_CALLs inside the loop bodies
     SF_ARRAY pq4_fork_members, PQ4_TILE_TOKENS * pq4_fork_args_size ; immutable per-member descriptors through FORK_WAIT
%assign PQ4_FORK_RBP_DISP __sf_arr_end         ; capture the end-anchored array displacement at declaration time
     SF_ARRAY pq4_args, 120                    ; v7_q8_batch4_args block: out/xq/xs/n/d constant per prefill, wq/ws per layer
     SF_DWORD pq_ffn_limit                     ; Maximum FFN panel width for this prompt
     SF_DWORD pq_ffn_count                     ; Active token count of the current panel
     SF_DWORD pq_ffn_packed                    ; Whether FFN input rows contain magnitudes
     SF_DWORD pq_ffn_tiles                     ; Active four-token descriptors
     SF_QWORD pq_ffn_input                     ; Cursor through packed FFN activations
     SF_QWORD pq_ffn_input_stride              ; Bytes per packed input and its scales
     SF_QWORD pq_c_packed                      ; First packed output for QKV postprocessing
     SF_QWORD pq_c_desc                        ; Base of the token descriptor array
     SF_DWORD pq_c_count                       ; Number of tokens described in this group
     SF_DWORD pq_c_panel                       ; Whether the descriptor array is a full panel
     SF_QWORD pq_qkv_scratch                   ; Original packed QKV scratch
     SF_DWORD pq_qkv_limit                     ; Maximum QKV panel width for this prompt
     SF_DWORD pq_qkv_pending                   ; Number of QKV tiles still awaiting postprocessing
     SF_QWORD pq_qkv_next                      ; Next completed QKV tile descriptor
     SF_QWORD pq_qkv_out                       ; Cursor through packed QKV panel output
     SF_ARRAY pq_ffn_args, PQ_PANEL_TILES * Q8_BATCH_ARGS_BYTES ; Four-token FFN batch descriptor
     SF_DWORD pq_ffn_tile_end                  ; Real-token bound, zero when the workspace cannot hold four outputs
     SF_DWORD pq_wo_limit                      ; Maximum output-projection panel width
     SF_QWORD pq_wo_input_stride               ; Bytes per quantized attention row and its scales
     SF_QWORD pq_ffn_max_output                ; Larger of the hidden and residual row strides
     SF_QWORD pq_wo_out                        ; Current WO projection output row
     SF_QWORD pq_ffn_stride                    ; Hidden activation stride in bytes
     SF_COMMIT                                 ; Commit stack frame: locals allocated
; === Save arguments ===
     mov pq_n_tokens, r8d                      ; Save n_tokens, number of prompt tokens
     mov pq_start_pos, r9d                     ; Save start_pos, global KV-cache position
     mov pq_tokens_ptr, rdx                    ; Save tokens pointer for embedding lookup
     mov pq_cfg_ptr, rcx                       ; Save Transformer* BEFORE any call overwrites rcx
     mov pq_ft_s, 0                            ; FloatTensor.s = NULL once, reused by the WO/W2/classifier q8_fused calls
; n_tokens/seq bounds are chat's outer-border contract, its KV-bounds FATAL, not re-checked here
; === M-RoPE: build merged rope tables for this prompt range, no-op for
; models without text M-RoPE; swaps fused_rope_cos/sin on image turns .
     mov rcx, rdx                              ; arg1 = original tokens pointer
     mov edx, r8d                              ; arg2 = original token count
     mov r8d, r9d                              ; arg3 = original start position
     call v7s_mrope_prepare                    ; prepare merged tables + delta, early-out if N/A
; === Extract struct pointers, reload rcx, overwritten by mrope_prepare call ===
     mov rcx, pq_cfg_ptr                       ; Reload Transformer*, rcx was overwritten by the call
     lea r13, [rcx + TRANSFORMER_WEIGHTS]      ; Cache w_ptr in non-volatile r13, hoisted for the whole body; per-layer only by the wq/wk/wv hoist
     lea r14, [rcx + TRANSFORMER_STATE]        ; Cache s_ptr in non-volatile r14, base for every s_* storage load in the body
; === Cache config values ===
     mov rbx, rcx                              ; rbx = Config*, non-volatile, callee-saved; rcx still holds it
     mov eax, [rbx + CFG_DIM]                  ; Load dim, model embedding dimension
     mov pq_dim, eax                           ; Cache dim
     mov eax, [rbx + CFG_HIDDEN_DIM]           ; Load hidden_dim, FFN intermediate dimension
     mov pq_hidden_dim, eax                    ; Cache hidden_dim
     mov eax, [rbx + CFG_N_LAYERS]             ; Load n_layers
     mov pq_n_layers, eax                      ; Cache n_layers
     mov eax, [rbx + CFG_N_HEADS]              ; Load n_heads
     mov pq_n_heads, eax                       ; Cache n_heads
     mov eax, [rbx + CFG_N_KV_HEADS]           ; Load n_kv_heads
     mov pq_n_kv_heads, eax                    ; Cache n_kv_heads
     mov eax, [rbx + CFG_VOCAB_SIZE]           ; Load vocab_size
     mov pq_vocab_size, eax                    ; Cache vocab_size
     mov eax, [rbx + CFG_SEQ_LEN]              ; Load seq_len
     mov pq_seq_len, eax                       ; Cache seq_len
     mov eax, [rbx + CFG_HEAD_DIM]             ; Load head_dim
     mov pq_head_dim, eax                      ; Cache head_dim
     mov eax, [rbx + CFG_HAS_QK_NORM]          ; Load has_qk_norm, 1=Qwen3 QK-RMSNorm, 0=Llama/SmolLM2
     mov pq_has_qk_norm, eax                   ; Cache has_qk_norm for Q/K post-processing decisions
; === Compute derived values, 64-bit products with signed-int kernel-ABI bound ===
     mov eax, dword pq_n_kv_heads              ; n_kv_heads, zero-extended: a positive config dim, same 64-bit value
     mov ecx, dword pq_head_dim                ; head_dim, zero-extended
     imul rax, rcx                             ; kv_dim = n_kv_heads * head_dim, 64-bit product, dword-stored below
     mov pq_kv_dim, eax                        ; kv_dim, dword; bounded at init
     mov eax, pq_n_heads                       ; n_heads for kv_mul division
     xor edx, edx                              ; Zero-extend eax->edx:eax for unsigned division
     mov ecx, pq_n_kv_heads                    ; Load n_kv_heads as divisor
     div ecx                                   ; kv_mul = n_heads / n_kv_heads, unsigned
     mov pq_kv_mul, eax                        ; kv_mul = n_heads / n_kv_heads
     mov eax, dword pq_n_heads                 ; n_heads, zero-extended: positive config dim
     mov ecx, dword pq_head_dim                ; head_dim, zero-extended for the 64-bit multiply
     imul rax, rcx                             ; all_heads_dim = n_heads * head_dim, 64-bit
     mov pq_all_heads_dim, rax                 ; Cache all_heads_dim, 64-bit, S45
; === Get storage pointers from RunState ===
     mov rcx, [r14 + S_X_STORAGE]              ; rcx = s->x_storage
     mov pq_x_storage, rcx                     ; Cache x_storage
     mov rcx, [r14 + S_Q_STORAGE]              ; rcx = s->q_storage
     mov pq_q_storage, rcx                     ; Cache q_storage
     mov rcx, [r14 + S_ATT_OUT]                ; rcx = s->att_out_storage
     mov pq_att_out_storage, rcx               ; Cache att_out_storage
; === Wake workers ===
     call initialize_once                      ; initialize the pool before replacing its first text-row copy
     API_CALL v7s_set_spin, 1                  ; Wake v7 worker threads
     mov eax, [rel g_vision_emb_consumed]      ; Load vision pool cursor at prefill entry
     mov pq_emb_base, eax                      ; pq_emb_base = rows consumed before this call
     xor r15d, r15d                            ; Initialize token counter t = 0 in r15d, non-volatile
     mov edi, dword pq_dim                     ; dim, zero-extend: dim always positive; shorter than movsxd
     shl rdi, 2                                ; byte count = dim*4, rdi zero-extended by the 32-bit mov
     mov r12, pq_x_storage                     ; r12 = dst = x_storage at t=0, non-volatile carrier, advances incrementally below
     mov esi, dword [rel g_image_token_id]     ; cache loop-invariant image-token id in rsi, survives API_CALLs: non-volatile, in save list
     jmp .pq_emb_chk                           ; jump to the bottom-test on entry, loop inversion
.pq_embed_loop:
     mov rax, pq_tokens_ptr                    ; Load tokens array pointer
     mov edx, [rax + r15*INT32_SCALE]          ; Load tokens[t] as int, token ID; r15 zero-extended index
     test esi, esi                             ; image-token override active?, rsi caches g_image_token_id across the API_CALLs
     jz .pq_emb_normal                         ; No, standard token embedding stays
     cmp edx, esi                              ; Is the current token the image-pad token?, register compare
     jne .pq_emb_normal                        ; No, plain token, keep table embedding
     mov r11d, [rel g_vision_emb_consumed]     ; r11 = next vision embedding row index
     cmp r11d, dword [rel g_vision_emb_count]  ; consumed index vs pool count, compared in memory
     jl .pq_emb_vision_ok                      ; Index in range, proceed with vision copy
     FATAL "vision: embedding pool exhausted in prefill_q8" ; Contract violation: more image tokens than embeddings
.pq_emb_vision_ok:
     mov eax, r11d                             ; 64-bit consumed row index, bounds-checked non-negative
     mov ecx, dword pq_dim                     ; dim, zero-extended: positive config dim
     imul rax, rcx                             ; rax = consumed * dim, element offset into the embedding pool
     mov rdx, rax                              ; rdx = full 64-bit element offset into embedding pool
     mov rax, [rel g_vision_embeddings]        ; rax = vision embeddings base pointer
     lea rdx, [rax + rdx*INT32_SCALE]          ; rdx = src = embeddings + consumed*dim*4
     mov rcx, r12                              ; dst = x_storage + t*dim*4, carried in r12
     API_CALL v7d_memcpy, rcx, rdx, rdi        ; Copy vision embedding row into x_storage[t], size = dim*4 in rdi
     inc dword [rel g_vision_emb_consumed]     ; Advance consumed index for next image token
     jmp .pq_emb_vision_done                   ; This token is fully written, continue the loop tail
.pq_emb_normal:
     mov rcx, r12                              ; output FP32 embedding row
     mov r8d, pq_dim                           ; row length in elements
     mov r9, [r13 + W_Q_TOKENS]                ; packed embedding descriptor
     call v7s_embedding_row                    ; dequantize exactly this token into the destination
.pq_emb_vision_done:
     add r12, rdi                              ; dst += dim*4, incremental, r12 carries the running x_storage pointer
     inc r15d                                  ; Advance token counter in register, no stack traffic!
.pq_emb_chk:
     cmp r15d, pq_n_tokens                     ; Compare loop counter against total, direct register cmp
     jl .pq_embed_loop                         ; Continue while more tokens remain
.pq_embed_done:
; === Save original s->x before layer loop, needed for restore at end ===
     mov rcx, [r14]                            ; rcx = original s->x pointer, offset 0
     mov pq_saved_x, rcx                       ; Save original s->x, safe even if n_layers==0
; === Layer loop ===
     mov pq_layer_idx, 0                       ; Initialize layer loop counter
     mov ecx, dword pq_seq_len                 ; Load seq_len, 32-bit load zero-extends
     mov r10d, dword pq_kv_dim                 ; Load kv_dim, 32-bit load zero-extends
     imul rcx, r10                             ; rcx = seq_len * kv_dim = loff stride, 64-bit
     mov pq_loff_stride, rcx                   ; Save KV-cache offset stride
     mov ecx, dword pq_dim                     ; Load dim
     shl rcx, 2                                ; rcx = dim * 4
     mov pq_dim_off_stride, rcx                ; Save RMS weight byte stride
     mov ecx, dword pq_head_dim                ; Load head_dim
     shl rcx, 2                                ; rcx = head_dim * 4
     mov pq_hd_off_stride, rcx                 ; Save norm weight byte stride
     mov rcx, pq_all_heads_dim                 ; Load all_heads_dim, QWORD slot, Q block length
     mov edx, dword pq_kv_dim                  ; kv_dim
     lea rcx, [rcx + rdx*PQ_KV_DOUBLED]        ; dsum = all_heads_dim + 2*kv_dim, one address-gen op
     shl rcx, 2                                ; dsum * 4 bytes
     mov pq_dsum_bytes, rcx                    ; Save QKV bias byte stride
     mov rax, pq_all_heads_dim                 ; all_heads_dim, 64-bit
     shl rax, 2                                ; ahd_bytes = all_heads_dim * 4, 64-bit per S45
     mov pq_ahd_bytes, rax                     ; Q/att per-token byte stride, shared by Phases 1/3wo/2
; Q and attention storage are both available after the WO phase.
; Each must hold four hidden vectors, and the gate storage later holds four W2 outputs.
     mov eax, pq_hidden_dim                    ; Hidden width in floats
     shl rax, 2                                ; Hidden activation stride in bytes
     mov pq_ffn_stride, rax                    ; Save the stride for the activation loop
     mov r8d, pq_n_tokens                      ; Number of initialized prompt rows
     imul r8, pq_ahd_bytes                     ; Bytes available in each reusable workspace
     mov r9, pq_dim_off_stride                 ; Residual row width in bytes
     cmp rax, r9                               ; Is a hidden vector wider than a residual vector?
     cmova r9, rax                             ; Use the larger width for the capacity check
     mov dword pq_ffn_tile_end, 0              ; Default to the per-token FFN
     mov pq_ffn_max_output, r9                 ; Both hidden and down-projection outputs must fit
     shl r9, 2                                 ; Bytes needed for four rows
     cmp r8, r9                                ; Do both reusable workspaces hold one FFN tile?
     jb .pq_panel_limits                       ; Keep the per-token WO and FFN routes
     mov edx, pq_n_tokens                      ; Token count for the full-tile bound
     mov pq_ffn_tile_end, edx                  ; Save the full-tile bound
.pq_panel_limits:
     mov eax, pq_dim                           ; Quantized activation bytes per QKV or FFN input
     mov edx, eax                              ; Compute the scale width separately
     shr edx, 4                                ; Four-byte scale per 64 activation bytes
     lea rax, [rax*PQ_PACK_EXPANSION + rdx]    ; Space for signed bytes, magnitudes and scales
     mov pq_ffn_input_stride, rax              ; Save the shared per-token input stride
     mov edx, pq4_fork_args_size               ; Bytes needed per postprocessing descriptor
     cmp rax, rdx                              ; Is the descriptor larger than its retired input?
     cmovb rax, rdx                            ; Size scratch for both lifetimes
     PQ_PANEL_CAPACITY pq_dsum_bytes           ; Limit QKV by the packed output row width
     mov pq_qkv_limit, eax                     ; Save the maximum QKV panel width
     mov rax, pq_ffn_input_stride              ; Read the packed FFN input stride
     PQ_PANEL_CAPACITY pq_ffn_max_output       ; Limit FFN by its widest projection output
     mov pq_ffn_limit, eax                     ; Save the maximum FFN panel width
     mov rax, pq_all_heads_dim                 ; Quantized bytes per attention row
     mov rdx, rax                              ; Compute the scale width separately
     shr rdx, 4                                ; Four-byte scale per 64 activation bytes
     lea rax, [rax*PQ_PACK_EXPANSION + rdx]    ; Space for signed bytes, magnitudes and scales
     mov pq_wo_input_stride, rax               ; Save the packed attention row stride
     PQ_PANEL_CAPACITY pq_dim_off_stride       ; Limit WO by its residual-width output
     mov pq_wo_limit, eax                      ; Save the maximum WO panel width
; === Phase-1 batch-4 scratch: runtime-sized stack block below the frame ===
; layout: [xq0..xq3, dim bytes each][xs0..xs3, dim/16 bytes each][qkv0..3, dsum*4 bytes each].
; The sub jumps guard pages, so each page is touched once (compressor probe pattern);
; SF_END recovers rsp from rbp, no matching add on any exit path.
; dim is a multiple of 64 (the Q8 group contract), so the xs stride dim/16 is exact.
     mov eax, pq_dim                           ; dim, zero-extends to rax
     shl rax, 2                                ; rax = 4*dim, the four quantized inputs
     mov edx, pq_dim                           ; dim again for the scale-array size
     shr rdx, 2                                ; rdx = dim/4, the four scale arrays: 4 slots of dim/16 bytes each
     add rax, rdx                              ; rax += the scale bytes
     mov rdx, rcx                              ; dsum*4 still in RCX
     lea rax, [rax + rdx*INT32_SCALE + Q4B_ROUND] ; + 4*dsum*4, biased for the 16-round-up below
     and rax, Q4B_ALIGN                        ; rax = 16-aligned scratch byte count
     sub rax, -PQ4_API_PAD                     ; + pad, imm8 encoding of the same +128
     mov rdx, rax                              ; rdx = total bytes to lower rsp
.pq4_scratch_probe:
     cmp rdx, PQ4_PROBE_PAGE                   ; more than one page left?
     jbe .pq4_scratch_last                     ; no: final partial chunk
     sub rsp, PQ4_PROBE_PAGE                   ; lower rsp by one page
     test [rsp], rsp                           ; touch the new page, guard-page commit
     sub rdx, PQ4_PROBE_PAGE                   ; one page fewer to reserve
     jmp .pq4_scratch_probe                    ; continue until <= one page remains
.pq4_scratch_last:
     sub rsp, rdx                              ; the final partial chunk
     test [rsp], rsp                           ; touch it
     lea rax, [rsp + PQ4_API_PAD]              ; rax = scratch base, above the API pad
     mov pq4_base, rax                         ; stage the base for the args fill below
; args constant fields: the four out/xq/xs pointer tables (they ARE the scratch
; layout), plus n/d. wq/ws are per-layer, stored in the layer-loop head.
     lea rdx, pq4_args                         ; rdx = args block address, the SF_ARRAY name already carries the brackets
     mov r9, pq4_base                          ; r9 = scratch cursor
     mov [rdx + Q4B_ARGS_XQ0], r9              ; xq[0] = base
     mov eax, pq_dim                           ; eax = dim, zero-extends: r10 = dim bytes per quantized input
     mov r10, rax                              ; r10 = dim bytes, xq row stride
     add r9, r10                               ; advance one xq row
     mov [rdx + Q4B_ARGS_XQ1], r9              ; xq[1]
     add r9, r10                               ; advance one xq row
     mov [rdx + Q4B_ARGS_XQ2], r9              ; xq[2]
     add r9, r10                               ; advance one xq row
     mov [rdx + Q4B_ARGS_XQ3], r9              ; xq[3]
     add r9, r10                               ; r9 = xs base = base + 4*dim
     mov r10d, pq_dim                          ; r10d = dim, zero-extends
     shr r10, 4                                ; r10 = dim/16 = xs stride bytes per token
     mov [rdx + Q4B_ARGS_XS0], r9              ; xs[0]
     add r9, r10                               ; advance one xs slot
     mov [rdx + Q4B_ARGS_XS1], r9              ; xs[1]
     add r9, r10                               ; advance one xs slot
     mov [rdx + Q4B_ARGS_XS2], r9              ; xs[2]
     add r9, r10                               ; advance one xs slot
     mov [rdx + Q4B_ARGS_XS3], r9              ; xs[3]
     add r9, r10                               ; r9 = qkv base = xs base + all four xs slots
     mov r10, pq_dsum_bytes                    ; r10 = dsum*4 bytes per packed output row
     mov pq_qkv_scratch, r9                    ; Save the original QKV output scratch
     mov [rdx + Q4B_ARGS_OUT0], r9             ; out[0]
     add r9, r10                               ; advance one packed row
     mov [rdx + Q4B_ARGS_OUT1], r9             ; out[1]
     add r9, r10                               ; advance one packed row
     mov [rdx + Q4B_ARGS_OUT2], r9             ; out[2]
     add r9, r10                               ; advance one packed row
     mov [rdx + Q4B_ARGS_OUT3], r9             ; out[3]
     mov r10d, pq_dim                          ; r10d = dim, zero-extends
     mov [rdx + Q4B_ARGS_N], r10d              ; args.n = dim, DWORD field, a qword store would spill into args.d
     mov rax, pq_dsum_bytes                    ; rax = dsum*4, SF_QWORD slot
     shr rax, 2                                ; rax = dsum = packed output rows
     mov [rdx + Q4B_ARGS_D], eax               ; args.d = dsum, DWORD field
     xor eax, eax                              ; 0 = layer-0 value for all four offsets
     mov pq_loff, rax                          ; loff,0 = 0
     mov pq_layer_bytes, rax                   ; layer_bytes,0 = 0
     mov pq_layer_dim_off, rax                 ; layer_dim_off,0 = 0
     mov pq_layer_hd_off, rax                  ; layer_hd_off,0 = 0
     mov rax, [rel g_qkv_bias]                 ; rax = g_qkv_bias base, always valid: inf_weights points bias-less models at an all-zeros block, the bias loops dereference it unconditionally, NULL would fault
     mov r11d, [rel g_qkv_bias_active]         ; r11d = active flag. MUST NOT be read into eax here: a 32-bit write
     mov pq_bias_active, r11d                  ; 0 = bias-less model: the adds touch an all-zeros block, skip them
     mov pq_q_bias_base, rax                   ; q_bias_base,0 = g_qkv_bias
     mov rcx, pq_all_heads_dim                 ; rcx = all_heads_dim, 64-bit
     lea rax, [rax + rcx*INT32_SCALE]          ; rax = g_qkv_bias + all_heads_dim*4
     mov pq_k_bias_base, rax                   ; k_bias_base,0 = q base + all_heads_dim*4
     mov ecx, dword pq_kv_dim                  ; ecx = kv_dim, dword read zero-extends rcx
     lea rax, [rax + rcx*INT32_SCALE]          ; rax = k base + kv_dim*4
     mov pq_v_bias_base, rax                   ; v_bias_base,0 = k base + kv_dim*4
; rope table element count, BSS global, loop-invariant across ALL layers, so it
; is staged once before the layer loop and every layer reuses the same slot
     mov eax, [rope_max_dim]                   ; Load rope_max_dim once, BSS global never changes during prefill
     mov pq_rope_max_slot, rax                 ; Save to stable stack slot, dword load zero-extends into rax
.pq_layer_loop:
     mov eax, pq_layer_idx                     ; Load current layer index
     cmp eax, pq_n_layers                      ; Compare against total number of layers
     jge .pq_after_layers                      ; Exit loop if all layers processed
; packed QKV desc for this layer: batch-4 args wq/ws (g_inf_qkv_desc[l])
     mov rax, [rel g_inf_qkv_desc]             ; rax = packed descriptor block base
     add rax, pq_layer_bytes                   ; rax = desc[layer], pq_layer_bytes == layer*QT_SIZE
     mov r9, [rax + QT_Q]                      ; r9 = packed weight bytes, [Q|K|V] rows
     mov r10, [rax + QT_S]                     ; r10 = packed scale block
     lea rax, pq4_args                         ; rax = args block address
     mov [rax + Q4B_ARGS_WQ], r9               ; args.wq = packed bytes for this layer
     mov [rax + Q4B_ARGS_WS], r10              ; args.ws = packed scales for this layer
; pq_q/k/v_bias_base are maintained incrementally, init + stride at loop tail --
; the K/V bases start pre-offset past the Q, and K bias sections.
     mov dword pq_qkv_pending, 0               ; No completed tiles are pending at layer entry
     xor edi, edi                              ; Token counter t = 0 in edi, register; edi carries the TILE BASE index through the tile path, the scatter's rep movsb cursor clobbers are undone by the push/pop in stage C
     mov rax, pq_x_storage                     ; x_ptr,0 = x_storage, advanced incrementally at the loop tail
     mov pq_x_ptr, rax                         ; Start the QKV input cursor
     mov rax, pq_q_storage                     ; q_dst,0 = q_storage
     mov pq_q_dst, rax                         ; Start the query output cursor
; Quantize and project packed QKV, then apply bias and RoPE to real tokens.
     jmp .pq4_tile_chk                         ; jump to the bottom-test on entry, loop inversion
.pq4_tile_loop:
     cmp dword pq_qkv_pending, 0               ; Did the preceding dispatch finish another tile?
     jne .pq_qkv_consume                       ; Postprocess the completed tile without a dispatch
     mov eax, pq_n_tokens                      ; Total real prompt tokens
     sub eax, edi                              ; Count remaining real inputs
     cmp eax, PQ4_TILE_TOKENS                  ; Is the final tile incomplete?
     jb .pq_qkv_one_tile                       ; A partial tile uses the original scratch
     cmp dword pq_qkv_limit, PQ4_TILE_TOKENS   ; Does the workspace hold multiple tiles?
     je .pq_qkv_one_tile                       ; Use the original scratch for one tile
     and eax, Q4B_TILE_MASK                    ; Larger panels contain only full tiles
     cmp eax, pq_qkv_limit                     ; Is the remaining panel shorter than the cap?
     cmova eax, pq_qkv_limit                   ; Cap the panel width
     mov pq_ffn_count, eax                     ; Save the active panel token count
     shr eax, 2                                ; Convert tokens to four-token tiles
     mov pq_qkv_pending, eax                   ; Save the number of tiles to postprocess
     mov rax, pq_qkv_scratch                   ; Retired QKV output scratch holds the packed inputs
     mov pq_ffn_input, rax                     ; Initialize the packed input cursor
     mov rax, pq_att_out_storage               ; Attention storage is unused until QKV completes
     mov pq_qkv_out, rax                       ; Initialize the packed output cursor
     mov dword pq4_j, 0                        ; Start at the first panel member
.pq_qkv_prepare:
     mov eax, pq4_j                            ; Read the member's index
     PQ_PANEL_MEMBER                           ; Select the member's descriptor and slot
     mov rcx, pq_ffn_input                     ; Read the packed input destination
     mov edx, pq_dim                           ; Quantized activation width
     lea r8, [rcx + rdx*PQ_PACK_EXPANSION]     ; Leave room for the expanded activation
     mov [r10 + rax*Q4B_ARGS_QW + Q4B_ARGS_XQ0], rcx ; Set the descriptor's input row
     mov [r10 + rax*Q4B_ARGS_QW + Q4B_ARGS_XS0], r8 ; Set the descriptor's scale row
     mov rdx, pq_qkv_out                       ; Read the output destination
     mov [r10 + rax*Q4B_ARGS_QW + Q4B_ARGS_OUT0], rdx ; Set the member's packed QKV output
     mov rdx, pq_dsum_bytes                    ; Read the packed output byte stride
     add pq_qkv_out, rdx                       ; Advance the packed output cursor
     mov rdx, pq_ffn_input_stride              ; Read the packed input byte stride
     add pq_ffn_input, rdx                     ; Advance the packed input cursor
     lea r11, pq4_args                         ; Read the common projection dimensions and weights
     mov rdx, [r11 + Q4B_ARGS_WQ]              ; Load the packed QKV weight bytes
     mov [r10 + Q4B_ARGS_WQ], rdx              ; Set the tile's weight bytes
     mov rdx, [r11 + Q4B_ARGS_WS]              ; Load the packed QKV weight scales
     mov [r10 + Q4B_ARGS_WS], rdx              ; Set the tile's weight scales
     mov rdx, [r11 + Q4B_ARGS_N]               ; Load the adjacent input and output widths
     mov [r10 + Q4B_ARGS_N], rdx               ; Set both dimensions for this descriptor
     mov edx, pq4_j                            ; Reload the member's index
     imul rdx, pq_dim_off_stride               ; Compute the offset from the current residual
     add rdx, pq_x_ptr                         ; Address this member's residual row
     mov r9d, pq_dim                           ; RMSNorm width in floats
     mov r10, [r13 + W_RMS_ATT_WEIGHT]         ; Read the attention normalization weights
     add r10, pq_layer_dim_off                 ; Select this layer's weights
     API_CALL v7s_rmsnorm_quantize, rcx, rdx, r8, r9, r10 ; Normalize and quantize the input
     PQ_PACK_INPUT pq_dim                      ; Precompute magnitudes for this QKV input
     inc dword pq4_j                           ; Count completed panel members
     mov eax, pq4_j                            ; Read the completed count
     cmp eax, pq_ffn_count                     ; Are all panel members ready?
     jb .pq_qkv_prepare                        ; Prepare the next token
     lea rcx, pq_ffn_args                      ; Address the complete QKV descriptor panel
     mov pq_qkv_next, rcx                      ; Remember the first tile for postprocessing
     mov edx, pq_qkv_pending                   ; Number of tile descriptors
     mov r8d, 1                                ; Every QKV panel input has packed magnitudes
     API_CALL v7d_q8_batch4_panel, rcx, rdx, r8 ; Dispatch cache-sized slices for all tiles
.pq_qkv_consume:
     cmp dword pq_has_qk_norm, 0               ; Can the postprocessing use private descriptors?
     je .pq_qkv_legacy_consume                 ; RoPE-only models retain the serial scatter
     mov rax, pq_att_out_storage               ; First packed output of the completed panel
     mov pq_c_packed, rax                      ; Use the entire contiguous output panel
     mov rax, pq_qkv_scratch                   ; Reuse the packed input memory after projection
     mov pq_c_desc, rax                        ; Place private descriptors in that workspace
     mov eax, pq_ffn_count                     ; Read the token count of the completed panel
     mov pq_c_count, eax                       ; Describe every token before dispatch
     mov dword pq_c_panel, 1                   ; Select the full-pool dispatch after setup
     jmp .pq4_stage_c_setup                    ; Share the existing descriptor construction
.pq_qkv_legacy_consume:
     mov dword pq_c_count, PQ4_TILE_TOKENS     ; A completed panel tile has four real rows
     mov rax, pq_qkv_next                      ; Read the completed tile descriptor
     lea rdx, pq4_args                         ; Stage C consumes the original descriptor shape
     vmovdqu ymm0, [rax + Q4B_ARGS_OUT0]       ; Read this tile's four packed output pointers
     vmovdqu [rdx + Q4B_ARGS_OUT0], ymm0       ; Publish the pointers for Stage C
     vzeroupper                                ; Clear upper lanes for Stage C's calls
     add rax, Q8_BATCH_ARGS_BYTES              ; Advance to the next descriptor
     mov pq_qkv_next, rax                      ; Save the next tile cursor
     dec dword pq_qkv_pending                  ; Account for the tile handed to Stage C
     jmp .pq4_postprocess                      ; Run the original scatter and RoPE
.pq_qkv_one_tile:
     mov eax, pq_n_tokens                      ; Total real prompt tokens
     sub eax, edi                              ; Count remaining real rows
     mov edx, PQ4_TILE_TOKENS                  ; Four rows fit in the tile scratch
     cmp eax, edx                              ; Does a complete tile remain?
     cmova eax, edx                            ; Limit to one tile
     mov pq_c_count, eax                       ; Only these members are postprocessed
; --- Stage A: quantize the four tokens of this tile into the scratch ---
; the member counter lives in an SF slot: API_CALL clobbers every volatile register
     mov dword pq4_j, 0                        ; j = 0
.pq4_stage_a_loop:
     mov r11d, dword pq4_j                     ; j, fresh each iteration, the call below clobbers it
     lea rax, pq4_args                         ; rax = args block, the per-token pointer tables live in it
     mov rcx, [rax + r11*Q4B_ARGS_QW + Q4B_ARGS_XQ0] ; xq[j], quantized output for tile member j
     mov r8, [rax + r11*Q4B_ARGS_QW + Q4B_ARGS_XS0] ; xs[j], scale output for tile member j
     mov rdx, pq_x_ptr                         ; rdx = token edi's x
     mov r9, pq_dim_off_stride                 ; r9 = dim*4 bytes per token
     mov eax, pq_c_count                       ; Number of real inputs
     dec eax                                   ; Index the last valid input
     cmp r11d, eax                             ; Is this slot padding?
     cmova r11d, eax                           ; Reuse the last real row
     imul r9, r11                              ; Byte offset of the valid input
     add rdx, r9                               ; rdx = x[edi+j]
     mov r9d, pq_dim                           ; r9 = dim, vector length
     mov r10, [r13 + W_RMS_ATT_WEIGHT]         ; r10 = w->rms_att_weight base
     add r10, pq_layer_dim_off                 ; r10 = rms_att_weight[layer]
     API_CALL v7s_rmsnorm_quantize, rcx, rdx, r8, r9, r10 ; fused RMSNorm+quantize: x[edi+j] -> scratch xq[j], eps from BSS
     inc dword pq4_j                           ; j++, slot, survives the call
     cmp dword pq4_j, PQ4_TILE_TOKENS          ; all four members quantized?
     jb .pq4_stage_a_loop                      ; continue
; --- Stage B: ONE dispatched packed-QKV batch call for the whole tile ---
     lea rcx, pq4_args                         ; rcx = args block, wq/ws point at this layer's packed desc
     API_CALL v7d_q8_batch4, rcx               ; four packed QKV blocks, row-sliced across the worker pool
.pq4_postprocess:
; --- Stage C: four-way register fork; the serial stage remains for RoPE-only models. ---
; The shared RMSNorm dispatcher BSS is not reentrant. Only QK-norm models
; use the descriptor-driven head kernel; RoPE-only models keep legacy Stage C.
; The fork has no disable switch and no asleep-worker bypass: the prefill
; entry's v7s_set_spin(1) guarantees a fork-receptive pool for the whole call.
; Forks are published only to workers that exist (W0 by the init invariant,
; W1 from two workers, W2 from three); the caller executes every member whose
; helper is absent, after its own member 3. There is no whole-tile fallback.
     cmp dword pq_c_count, PQ4_TILE_TOKENS     ; Does the tile have all four real rows?
     jb .pq4_stage_c_legacy                    ; Only real rows receive RoPE and cache writes
     cmp dword pq_has_qk_norm, 0               ; model has per-head QK norm?
     je .pq4_stage_c_legacy                    ; no: RoPE-only models keep the serial stage, config branch
     lea rax, pq4_args                         ; Read the original tile descriptor
     mov rax, [rax + Q4B_ARGS_OUT0]            ; First packed output of the tile
     mov pq_c_packed, rax                      ; Use the four contiguous packed rows
     lea rax, pq4_fork_members                 ; Original four-token descriptor array
     mov pq_c_desc, rax                        ; Use the stack descriptors for one tile
     mov dword pq_c_count, PQ4_TILE_TOKENS     ; Describe the four tile members
     mov dword pq_c_panel, 0                   ; Select the original fork path after setup
.pq4_stage_c_setup:
     SET_FTZ_DAZ                               ; match normal Q/K dispatcher caller MXCSR
; Stage the original S_K/S_V pointers for this tile, consistent with legacy.
; Only caller token 3 writes their FP32 contents, while W0..W2 use their
; private packed QKV blocks. All four write disjoint Q and FP16 cache rows.
     mov rax, [rel g_inf_kv_scratch]           ; rax = K/V FP32 staging block base
     mov [r14 + S_K], rax                      ; s->k = staging K row, caller's member 3 fills it
     mov ecx, pq_kv_dim                        ; ecx = kv_dim floats per K row
     lea rax, [rax + rcx*INT32_SCALE]          ; V row begins one kv_dim float later
     mov [r14 + S_V], rax                      ; s->v = staging V row
     mov dword pq4_j, 0                        ; j = 0, descriptor fill loop counter
.pq4_fork_desc_loop:
     mov r10d, pq4_j                           ; tile member index 0..3
     mov r9, pq_dsum_bytes                     ; Packed output row stride
     imul r9, r10                              ; Compute this member's output byte offset
     add r9, pq_c_packed                       ; Address this token's packed QKV output
     mov r11, r10                              ; Copy the member index
     shl r11, 7                                ; Convert to a 128-byte descriptor offset
     mov r8, pq_c_desc                         ; Read the descriptor array base
     add r8, r11                               ; Address the member's private descriptor
     mov [r8 + pq4_fork_args.out], r9          ; packed QKV source block
     mov rax, pq_ahd_bytes                     ; rax = Q row byte stride, all_heads_dim*4
     imul rax, r10                             ; member * Q row stride
     add rax, pq_q_dst                         ; rax = this member's Q row base
     mov [r8 + pq4_fork_args.q_dst], rax       ; persistent float Q destination
     mov rax, pq_q_bias_base                   ; rax = Q bias base for this layer
     mov [r8 + pq4_fork_args.q_bias], rax      ; per-member Q bias
     mov rax, pq_k_bias_base                   ; rax = K bias base for this layer
     mov [r8 + pq4_fork_args.k_bias], rax      ; per-member K bias
     mov rax, pq_v_bias_base                   ; rax = V bias base for this layer
     mov [r8 + pq4_fork_args.v_bias], rax      ; per-member V bias
     mov rax, [r13 + W_Q_NORM_WEIGHTS]         ; rax = Q head norm weights base
     add rax, pq_layer_hd_off                  ; rax = q_norm[layer]
     mov [r8 + pq4_fork_args.q_weight], rax    ; per-member Q norm weights
     mov rax, [r13 + W_K_NORM_WEIGHTS]         ; rax = K head norm weights base
     add rax, pq_layer_hd_off                  ; rax = k_norm[layer]
     mov [r8 + pq4_fork_args.k_weight], rax    ; per-member K norm weights
     mov rax, pq_all_heads_dim                 ; full 64-bit count, model-bounded before storing dword
     mov [r8 + pq4_fork_args.all_heads], eax   ; Q float count
     mov eax, pq_kv_dim                        ; eax = kv_dim
     mov [r8 + pq4_fork_args.kv_dim], eax      ; K and V float count each
     mov eax, pq_head_dim                      ; eax = head_dim
     mov [r8 + pq4_fork_args.head_dim], eax    ; per-head dimension
     mov eax, pq_n_heads                       ; eax = n_heads
     mov [r8 + pq4_fork_args.n_heads], eax     ; number of Q heads
     mov eax, pq_n_kv_heads                    ; eax = n_kv_heads
     mov [r8 + pq4_fork_args.n_kv_heads], eax  ; number of K heads
     mov eax, pq_bias_active                   ; eax = cached bias flag
     mov [r8 + pq4_fork_args.bias_active], eax ; bias adds skipped when model has none
     xor edx, edx                              ; edx = 0, last_member default
     mov eax, pq_c_count                       ; Number of members in the current group
     dec eax                                   ; Index of the last member
     cmp pq4_j, eax                            ; Is this the final member?
     sete dl                                   ; dl = 1 only for the final member
     mov [r8 + pq4_fork_args.last_member], edx ; member 3 writes the shared K/V staging row
; KV cache uses RAW token position. RoPE may use the compacted M-RoPE
; position, matching the original Stage C semantics.
     mov r9d, pq_start_pos                     ; r9d = base start position
     add r9d, edi                              ; + tile base token index
     add r9d, pq4_j                            ; r9 = raw absolute position
     mov eax, r9d                              ; eax = raw position copy for the delta test
     cmp qword [rel g_mrope_saved_cos], 0      ; M-RoPE scratch tables swapped in?
     jne .pq4_fork_pos_raw                     ; yes: raw positions index the scratch rows
     add eax, [rel g_mrope_delta]              ; no: compacted M-RoPE position, 0 for plain text
.pq4_fork_pos_raw:
     mov [r8 + pq4_fork_args.pos], eax         ; RoPE position, raw or delta-corrected
     mov ecx, eax                              ; ecx = position copy for the table offset
     imul rcx, pq_rope_max_slot                ; elements into fused cos/sin tables
     mov r11, [fused_rope_cos]                 ; r11 = cos table base
     lea r11, [r11 + rcx*INT32_SCALE]          ; r11 = cos row at this position
     mov [r8 + pq4_fork_args.cos], r11         ; pre-offset cos pointer
     mov r11, [fused_rope_sin]                 ; r11 = sin table base
     lea r11, [r11 + rcx*INT32_SCALE]          ; r11 = sin row at this position
     mov [r8 + pq4_fork_args.sin], r11         ; pre-offset sin pointer
     mov eax, r9d                              ; eax = raw position, cache rows are raw-indexed
     mov edx, pq_kv_dim                        ; edx = kv_dim
     imul rax, rdx                             ; rax = raw_pos * kv_dim, elements
     add rax, pq_loff                          ; cache element index = loff + raw_pos * kv_dim
     mov r11, [r14 + S_KEY_CACHE]              ; r11 = packed key cache base
     lea r11, [r11 + rax*PQ_FP16_BYTES]        ; r11 = this member's key cache row bytes
     mov [r8 + pq4_fork_args.key_cache], r11   ; FP16 key destination
     mov r11, [r14 + S_VALUE_CACHE]            ; r11 = packed value cache base
     lea r11, [r11 + rax*PQ_FP16_BYTES]        ; r11 = this member's value cache row bytes
     mov [r8 + pq4_fork_args.value_cache], r11 ; FP16 value destination
     inc dword pq4_j                           ; j++, slot, next member descriptor
     mov eax, pq4_j                            ; Number of descriptors already built
     cmp eax, pq_c_count                       ; Are all descriptors ready?
     jb .pq4_fork_desc_loop                    ; Prepare the next descriptor
     cmp dword pq_c_panel, 0                   ; Does the whole panel have private descriptors?
     je .pq4_fork_tile                         ; A single tile uses the original fork
     mov rcx, pq_c_desc                        ; Start of the private descriptor array
     mov edx, pq_c_count                       ; Number of token jobs
     API_CALL v7d_qkv_post_panel, rcx, rdx     ; Run the unchanged postprocessing in parallel
     mov eax, pq_c_count                       ; Number of completed tokens
     add edi, eax                              ; Advance the absolute prompt cursor
     mov rdx, pq_dim_off_stride                ; Bytes per residual row
     imul rdx, rax                             ; Bytes in the completed residual span
     add pq_x_ptr, rdx                         ; Advance the residual cursor
     imul rax, pq_ahd_bytes                    ; Bytes in the completed query span
     add pq_q_dst, rax                         ; Advance the query output cursor
     mov dword pq_qkv_pending, 0               ; The dispatch postprocessed the entire panel
     jmp .pq4_tile_chk                         ; Prepare another panel or the tail
.pq4_fork_tile:
; Forks are published only to workers that exist: W2 first (three or more
; workers), then W1 (two or more), then W0 (the init invariant guarantees at
; least one). The caller runs member 3 plus every member whose helper is
; absent, then joins the published helpers in the same descending order.
     cmp dword [num_workers], 3                ; W2 exists?
     jb .pq4_fork_one                          ; no: caller will run member 2 itself
     lea rcx, [rbp - PQ4_FORK_RBP_DISP + PQ4_FORK_MEMBER2_OFF] ; rcx = descriptor 2
     FORK 2                                    ; hand member 2 to worker 2
     jz .pq4_fork_parent2                      ; parent path continues below
     call v7s_pq4_fork_member                  ; worker 2: member 2 postprocess
     CONTINUE_DISPATCH 2                       ; return worker 2 to its spin loop
.pq4_fork_parent2:
.pq4_fork_one:
     cmp dword [num_workers], 2                ; W1 exists?
     jb .pq4_fork_zero                         ; no: caller will run member 1 itself
     lea rcx, [rbp - PQ4_FORK_RBP_DISP + PQ4_FORK_MEMBER1_OFF] ; rcx = descriptor 1
     FORK 1                                    ; hand member 1 to worker 1
     jz .pq4_fork_parent1                      ; parent path continues below
     call v7s_pq4_fork_member                  ; worker 1: member 1 postprocess
     CONTINUE_DISPATCH 1                       ; return worker 1 to its spin loop
.pq4_fork_parent1:
.pq4_fork_zero:
     lea rcx, pq4_fork_members                 ; rcx = descriptor 0, arg1 via the FORK transfer
     FORK 0                                    ; hand member 0 to worker 0, exists by the init invariant
     jz .pq4_fork_parent0                      ; parent path continues below
     call v7s_pq4_fork_member                  ; worker 0: member 0 postprocess
     CONTINUE_DISPATCH 0                       ; return worker 0 to its spin loop
.pq4_fork_parent0:
     lea rcx, [rbp - PQ4_FORK_RBP_DISP + PQ4_FORK_MEMBER3_OFF] ; rcx = descriptor 3, caller's own member
     call v7s_pq4_fork_member                  ; caller: member 3 postprocess, writes shared K/V scratch
     cmp dword [num_workers], 3                ; W2 absent: caller also runs member 2
     jae .pq4_caller_one                       ; three or more workers: member 2 already forked to W2
     lea rcx, [rbp - PQ4_FORK_RBP_DISP + PQ4_FORK_MEMBER2_OFF] ; rcx = descriptor 2
     call v7s_pq4_fork_member                  ; caller: member 2 postprocess
.pq4_caller_one:
     cmp dword [num_workers], 2                ; W1 absent: caller also runs member 1
     jae .pq4_join_two                         ; two or more workers: member 1 already forked to W1
     lea rcx, [rbp - PQ4_FORK_RBP_DISP + PQ4_FORK_MEMBER1_OFF] ; rcx = descriptor 1
     call v7s_pq4_fork_member                  ; caller: member 1 postprocess
.pq4_join_two:
     cmp dword [num_workers], 3                ; join in the publish order: W2 first when it exists
     jb .pq4_join_one                          ; no W2: nothing to join on slot 2
     FORK_WAIT 2                               ; join worker 2
.pq4_join_one:
     cmp dword [num_workers], 2                ; W1 first when it exists
     jb .pq4_join_zero                         ; no W1: nothing to join on slot 1
     FORK_WAIT 1                               ; join worker 1
.pq4_join_zero:
     FORK_WAIT 0                               ; join worker 0, always published
; Legacy path increments these pointers once per token; fork consumes 4.
     mov rax, pq_dim_off_stride                ; rax = x stride, dim*4
     shl rax, 2                                ; four tokens consumed by this tile
     add pq_x_ptr, rax                         ; running x pointer advances four tokens
     mov rax, pq_ahd_bytes                     ; rax = Q stride, all_heads_dim*4
     shl rax, 2                                ; four tokens consumed by this tile
     add pq_q_dst, rax                         ; running Q output pointer advances four tokens
     jmp .pq4_stage_c_joined                   ; skip the serial loop, tile complete
.pq4_stage_c_legacy:
; --- Stage C: per member: scatter Q/K/V, then the identical bias+QK-RoPE blocks ---
; member counter in the SF slot again: the rope calls clobber every volatile
     mov dword pq4_j, 0                        ; j = 0
.pq4_stage_c_loop:
     mov r11d, dword pq4_j                     ; j, fresh each iteration
     lea rax, pq4_args                         ; rax = args block
     mov r9, [rax + r11*Q4B_ARGS_QW + Q4B_ARGS_OUT0] ; r9 = out[j], this member's packed QKV block, survives all three scatters (rep movsb touches only rsi/rdi/rcx)
; abs_pos for token edi+j: start_pos + edi + j, M-RoPE delta-corrected, identical to the original block
     lea eax, [rdi + r11]                      ; edi + member j in one address computation
     add eax, pq_start_pos                     ; + base start position
     mov ecx, eax                              ; position copy for the delta test
     cmp qword [rel g_mrope_saved_cos], 0      ; M-RoPE scratch tables swapped in?
     jne .pq4c_pos_raw                         ; yes, raw positions
     add ecx, dword [rel g_mrope_delta]        ; compacted M-RoPE position
.pq4c_pos_raw:
     mov pq_pos_slot, rcx                      ; stage the RoPE position for this member's PQ_QK_ROPE calls
; K/V are writable in place: out[j] = contiguous Q | K | V FP32 block, and the
; batch4 fork path already establishes that these packed blocks are disjoint,
; mutually private per member. Only Q needs a copy to its persistent row.
     mov rcx, pq_ahd_bytes                     ; rcx = Q block bytes
     lea r8, [r9 + rcx]                        ; r8 = K starts after Q inside out[j]
     mov [r14 + S_K], r8                       ; s->k = K row inside the packed block
     mov eax, pq_kv_dim                        ; eax = kv_dim
     lea rdx, [r8 + rax*INT32_SCALE]           ; V row follows K (kv_dim floats later)
     mov [r14 + S_V], rdx                      ; s->v = V row inside the packed block
; scatter Q: out[j][0..ahd) -> this member's Q row
; push/pop the tile counter ACROSS the copy: edi carries the tile's
; base token index and the rep movsb overwrites rdi/edi as the
; destination cursor; without the pop the next member computes its abs_pos
; and KV rows from a heap address's low 32 bits
     push rdi                                  ; save edi = tile base token index
     mov rsi, r9                               ; src = packed block start
     mov rdi, pq_q_dst                         ; dst = this member's Q output row
     rep movsb                                 ; Q block, rcx already = Q bytes
     pop rdi                                   ; restore edi = tile base token index
; Q bias add: identical to the original block
     cmp dword pq_bias_active, 0               ; bias block contains real data?
     je .pq4c_qb_skip                          ; no, the zeros-block add is a no-op, skip it
     mov rdx, pq_q_dst                         ; rdx = this member's Q row
     mov rax, pq_q_bias_base                   ; rax = Q bias base for this layer
     mov ecx, dword pq_all_heads_dim           ; ecx = all_heads_dim, loop count
     PQ_BIAS_ADD                               ; 32-float bias add loop
     vzeroupper                                ; clean AVX state before next API_CALL
.pq4c_qb_skip:
     mov rcx, pq_q_dst                         ; rcx = Q row as RoPE input/output
     PQ_QK_ROPE pq_n_heads, W_Q_NORM_WEIGHTS   ; Q: RMSNorm+RoPE or RoPE-only
; K bias add: identical to the original block
     mov rdx, [r14 + S_K]                      ; rdx = key row
     cmp dword pq_bias_active, 0               ; bias block contains real data?
     je .pq4c_kb_skip                          ; no, skip
     mov rax, pq_k_bias_base                   ; rax = K bias base for this layer
     mov ecx, dword pq_kv_dim                  ; ecx = kv_dim, loop count
     PQ_BIAS_ADD                               ; 32-float bias add loop
     vzeroupper                                ; clean AVX state before next API_CALL
.pq4c_kb_skip:
     mov rcx, [r14 + S_K]                      ; rcx = key row as RoPE input/output
     PQ_QK_ROPE pq_n_kv_heads, W_K_NORM_WEIGHTS ; K: RMSNorm+RoPE or RoPE-only
; V bias add: identical to the original block
     mov rdx, [r14 + S_V]                      ; rdx = value row
     cmp dword pq_bias_active, 0               ; bias block contains real data?
     je .pq4c_vb_skip                          ; no, skip
     mov rax, pq_v_bias_base                   ; rax = V bias base for this layer
     mov ecx, dword pq_kv_dim                  ; ecx = kv_dim, loop count
     PQ_BIAS_ADD                               ; 32-float bias add loop
     vzeroupper                                ; clean AVX state before next API_CALL
.pq4c_vb_skip:
; Commit this member before scratch is overwritten by the next tile member.
     PQ_PACK_KV_F16 S_KEY_CACHE, S_K, pq4_j    ; commit member K
     PQ_PACK_KV_F16 S_VALUE_CACHE, S_V, pq4_j  ; commit member V
; Preserve the original serial-stage postcondition: after a four-token tile,
; S_K/S_V point at the FINAL token's fully biased/normed/rotated FP32 rows
; in g_inf_kv_scratch. The final member's K/V are the only rows any later
; consumer reads, so copy exactly one K+V pair back per tile.
     mov eax, pq_c_count                       ; Count of real tile members
     dec eax                                   ; Index the final real member
     cmp pq4_j, eax                            ; Does this row own shared K/V staging?
     jne .pq4c_keep_private_kv                 ; no: members 0..2 stay private in out[j]
     push rdi                                  ; preserve the outer edi tile counter
     mov rsi, [r14 + S_K]                      ; K | V are contiguous in out[3]
     mov rdi, [rel g_inf_kv_scratch]           ; dst = shared K/V FP32 staging block
     mov ecx, pq_kv_dim                        ; kv_dim
     shl rcx, 3                                ; 2 * kv_dim * sizeof(float) bytes
     rep movsb                                 ; ONE copy per entire four-token tile
     pop rdi                                   ; restore edi = tile base token index
     mov rax, [rel g_inf_kv_scratch]           ; rax = staging block base
     mov [r14 + S_K], rax                      ; s->k = shared staging row again
     mov ecx, pq_kv_dim                        ; kv_dim
     lea rax, [rax + rcx*INT32_SCALE]          ; V row begins one kv_dim float later
     mov [r14 + S_V], rax                      ; s->v = shared staging row again
.pq4c_keep_private_kv:
; Advance the running pointers for the next real member.
     mov rax, pq_dim_off_stride                ; x stride, dim*4
     add pq_x_ptr, rax                         ; running x pointer advances one token
     mov rax, pq_ahd_bytes                     ; Q stride, all_heads_dim*4
     add pq_q_dst, rax                         ; running Q output pointer advances one token
     inc dword pq4_j                           ; j++, slot, survives the calls
     mov eax, pq_c_count                       ; Number of real tile members
     cmp pq4_j, eax                            ; Did every real row finish postprocessing?
     jb .pq4_stage_c_loop                      ; continue
.pq4_stage_c_joined:
     add edi, pq_c_count                       ; Advance by the real token count
.pq4_tile_chk:
     cmp edi, pq_n_tokens                      ; Are there real tokens left to project?
     jl .pq4_tile_loop                         ; continue
.pq_phase1_done:
     cmp dword pq_qkv_limit, PQ4_TILE_TOKENS   ; Did QKV temporarily borrow attention storage?
     je .pq_attention_setup                    ; Original descriptor still has its stack pointers
     lea rdx, pq4_args                         ; Restore the original descriptor outputs
     mov rax, pq_qkv_scratch                   ; First original packed QKV slot
     mov r9, pq_dsum_bytes                     ; Packed output stride
     mov [rdx + Q4B_ARGS_OUT0], rax            ; Restore slot zero
     add rax, r9                               ; Advance to the next slot
     mov [rdx + Q4B_ARGS_OUT1], rax            ; Restore slot one
     add rax, r9                               ; Advance to the next slot
     mov [rdx + Q4B_ARGS_OUT2], rax            ; Restore slot two
     add rax, r9                               ; Advance to the next slot
     mov [rdx + Q4B_ARGS_OUT3], rax            ; Restore slot three
.pq_attention_setup:
     cmp dword pq_n_tokens, PQ4_TILE_TOKENS    ; Is there enough work to fan out independent tokens?
     jb .pq_attention_scalar                   ; Keep the short-prompt attention policy
     mov r8, pq_q_storage                      ; Query storage for the prompt
     mov r9, pq_att_out_storage                ; Output storage for the prompt
     mov rcx, r14                              ; Original RunState
     mov rdx, rbx                              ; Model configuration
     sub rsp, PQ_ATT_PANEL_FRAME               ; Reserve Win64 shadow and five stack arguments
     mov eax, pq_start_pos                     ; First absolute position
     mov [rsp + APICALL_ARG5], rax             ; Stage the start position
     mov eax, pq_n_tokens                      ; Number of prompt tokens
     mov [rsp + APICALL_ARG6], rax             ; Stage the token count
     mov rax, pq_loff                          ; Read the layer KV offset
     mov [rsp + APICALL_ARG7], rax             ; Stage the layer KV offset
     mov eax, pq_kv_dim                        ; Read the KV row width
     mov [rsp + APICALL_ARG8], rax             ; Stage the KV row width
     mov eax, pq_kv_mul                        ; Read query heads per KV head
     mov [rsp + PQ_ATT_PANEL_ARG9], rax        ; Stage the query head grouping
     call v7d_q8_attention_panel               ; Dispatch independent causal attention tokens
     add rsp, PQ_ATT_PANEL_FRAME               ; Release the panel's call frame
     mov rdi, pq_ahd_bytes                     ; Restore the stride used by WO
     jmp .pq_wo_setup                          ; The panel leaves the original RunState intact
.pq_attention_scalar:
     mov rcx, [r14 + S_XB]                     ; rcx = original s->xb pointer
     mov pq_saved_xb, rcx                      ; Save original s->xb
     mov rcx, [r14 + S_Q]                      ; rcx = original s->q pointer
     mov pq_saved_q, rcx                       ; Save original s->q
     mov esi, dword pq_n_tokens                ; countdown = n_tokens, >= 1, guarded at entry in non-volatile esi
     mov r12d, dword pq_start_pos              ; abs_pos register, init start_pos, inc per token
     xor r15d, r15d                            ; byte_offset register, init 0, += ahd_bytes per token
     mov rdi, pq_ahd_bytes                     ; stride register = all_heads_dim*4, non-volatile, survives API_CALL; hoisted slot reuse
.pq_attn_loop:
; byte_offset, r15 indexes q_storage/att_out_storage; abs_pos, r12d drives the KV row + arg3
     mov rcx, pq_q_storage                     ; Load q_storage base pointer
     add rcx, r15                              ; rcx = q_storage + byte_offset, per-token Q
     mov [r14 + S_Q], rcx                      ; s->q points to this token's Q vector
     mov rcx, pq_att_out_storage               ; Load att_out_storage base pointer
     add rcx, r15                              ; rcx = att_out_storage + byte_offset
     mov [r14 + S_XB], rcx                     ; s->xb points to this token's attention output slot
     mov r8d, r12d                             ; r8d = abs_pos for the attention call arg3
; Call v7d_attention,s, cfg, abs_pos, loff, kv_dim, kv_mul
     mov rcx, r14                              ; arg1 = RunState*, cached in non-volatile
     mov rdx, rbx                              ; arg2 = Config*, cached in non-volatile
     mov r9, pq_loff                           ; arg4 = layer KV-cache offset
     mov r10d, pq_kv_dim                       ; arg5 = KV dimension
     mov r11d, pq_kv_mul                       ; arg6 = query heads per KV head, GQA
     API_CALL v7d_attention, rcx, rdx, r8, r9, r10, r11 ; Multithreaded attention
     add r15, rdi                              ; byte_offset += all_heads_dim*4
     inc r12d                                  ; abs_pos += 1
     dec esi                                   ; countdown, register, esi is non-volatile, so no stack RMW
     jnz .pq_attn_loop                         ; Continue until all attention calls complete
.pq_attn_done:
; Restore original s->xb and s->q, s->k/s->v were never touched
     mov rcx, pq_saved_xb                      ; Load saved original s->xb pointer
     mov [r14 + S_XB], rcx                     ; Restore s->xb
     mov rcx, pq_saved_q                       ; Load saved original s->q pointer
     mov [r14 + S_Q], rcx                      ; Restore s->q
.pq_wo_setup:
     xor r12d, r12d                            ; Token counter t = 0 in r12d, register; r12 dead after Phase 2
     mov rax, pq_x_storage                     ; x_ptr,0 = x_storage, advanced incrementally at the loop tail
     mov pq_x_ptr, rax                         ; init Phase-3wo running x pointer
     mov r15, pq_att_out_storage               ; att cursor,0 = att_out_storage, r15 dead since Phase 2
     mov rsi, [r13 + W_WO]                     ; rsi = w->wo base, loop-invariant; hoisted out of the token loop, rsi free after Phase 2
     add rsi, pq_layer_bytes                   ; rsi = wo[l] = wo + layer_idx * 16, per-layer constant: tensor_f32 POINTER stride, NOT the dim*4 data stride, that is pq_dim_off_stride, a different constant in this same function
     cmp dword pq_ffn_tile_end, 0              ; Does Q storage hold four output rows?
     je .pq3wo_chk                             ; Run the existing projection for small prompts
.pq_wo_tile:
     mov eax, pq_ffn_tile_end                  ; Read the full-tile prompt bound
     sub eax, r12d                             ; Count tokens still eligible for this panel
     cmp eax, pq_wo_limit                      ; Can all remaining tokens fit?
     cmova eax, pq_wo_limit                    ; Use only the available workspace
     mov pq_ffn_count, eax                     ; Save the active panel width
     add eax, PQ4_TILE_TOKENS - 1              ; Round up for the final partial tile
     shr eax, 2                                ; Convert to descriptor count
     mov pq_ffn_tiles, eax                     ; Save the active descriptor count
     mov rax, pq_qkv_scratch                   ; Retired QKV scratch holds the packed attention inputs
     mov pq_ffn_input, rax                     ; Start at the first packed input
     mov dword pq4_j, 0                        ; Start with the first token of the panel
.pq_wo_quantize:
     mov eax, pq4_j                            ; Read the token's panel index
     PQ_PANEL_MEMBER                           ; Select the descriptor and pointer slot
     mov rcx, pq_ffn_input                     ; Read the current packed input address
     mov r9, pq_all_heads_dim                  ; Read the WO input width
     lea r8, [rcx + r9*PQ_PACK_EXPANSION]      ; Leave room for the expanded attention input
     mov [r10 + rax*Q4B_ARGS_QW + Q4B_ARGS_XQ0], rcx ; Set the quantized input for this token
     mov [r10 + rax*Q4B_ARGS_QW + Q4B_ARGS_XS0], r8 ; Set the input scales for this token
     API_CALL v7d_quantize, rcx, r15, r8, r9   ; Quantize this token's attention output
     PQ_PACK_INPUT pq_all_heads_dim            ; Precompute magnitudes for the WO dot products
     mov rax, pq_wo_input_stride               ; Read the packed input stride
     add pq_ffn_input, rax                     ; Advance the packed input cursor
     add r15, rdi                              ; Advance to the next attention output row
     inc dword pq4_j                           ; Count the prepared inputs
     mov eax, pq4_j                            ; Read the completed count
     cmp eax, pq_ffn_count                     ; Are all panel inputs ready?
     jb .pq_wo_quantize                        ; Quantize another row
     PQ_PAD_INPUTS                             ; Complete the final descriptor before dispatch
     PQ_FFN_PROJECT W_WO, pq_q_storage, pq_dim, pq_all_heads_dim, pq_ffn_tiles, 1 ; Project the panel to residual width
     mov rax, pq_q_storage                     ; First projection output row
     mov pq_wo_out, rax                        ; Keep the output cursor across vector adds
     mov dword pq4_j, 0                        ; Begin with the first output row
.pq_wo_residual:
     mov rcx, pq_x_ptr                         ; Read the next residual destination
     mov rdx, pq_wo_out                        ; Read the next WO output
     mov r8d, pq_dim                           ; Length of each residual row
     API_CALL v7s_add_inplace, rcx, rdx, r8    ; Add the projection to the residual
     mov rax, pq_dim_off_stride                ; Residual and WO rows share a stride
     add pq_x_ptr, rax                         ; Advance the residual destination
     add pq_wo_out, rax                        ; Advance the projection output cursor
     inc dword pq4_j                           ; Count completed residual rows
     mov eax, pq4_j                            ; Read the completed count
     cmp eax, pq_ffn_count                     ; Is the panel complete?
     jb .pq_wo_residual                        ; Update the next residual
     add r12d, pq_ffn_count                    ; Count the completed tokens
     cmp r12d, pq_ffn_tile_end                 ; Is another full tile available?
     jl .pq_wo_tile                            ; Dispatch the next projection panel
     jmp .pq3wo_chk                            ; Handle the remaining tokens
.pq_phase3wo_loop:
; x_ptr and att_ptr are maintained incrementally, slots advanced at the loop tail
; === WO: fused quantize+Q8,att_out[t], wo[l] -> hb2 ===
; v7d_q8_fused,s->hb2, &ft_x, wo[layer], all_heads_dim, dim
     mov rax, r15                              ; rax = att_out[t], float* attention output
     mov rcx, [r14 + S_HB2]                    ; rcx = s->hb2 as output buffer
     PQ_SET_FT_X                               ; ft.x = att_out[t]; rdx = &ft, staging macro
     mov r8, rsi                               ; r8 = wo[l], hoisted base, no per-token reload
     mov r9, pq_all_heads_dim                  ; r9 = n, input dimension, from attention
     mov r10d, pq_dim                          ; r10 = d, output dimension, back to embedding size
     API_CALL v7d_q8_fused, rcx, rdx, r8, r9, r10 ; Fused WO projection: quantize+int8 matmul, output to hb2
; === Residual: x += hb2 ===
     mov rcx, pq_x_ptr                         ; rcx = per-token x pointer
     mov rdx, [r14 + S_HB2]                    ; rdx = s->hb2 as residual to add
     mov r8d, pq_dim                           ; r8 = vector length
     API_CALL v7s_add_inplace, rcx, rdx, r8    ; x[t] += s->hb2, residual connection
; === Next WO token: advance incremental pointers ===
     mov rax, pq_dim_off_stride                ; x_ptr += dim*4
     add pq_x_ptr, rax                         ; advance the running x pointer
     add r15, rdi                              ; att cursor += all_heads_dim*4, exact stride
     inc r12d                                  ; Advance token counter, register
.pq3wo_chk:
     cmp r12d, pq_n_tokens                     ; Compare token counter against total, direct register cmp
     jl .pq_phase3wo_loop                      ; Continue while more tokens remain
.pq_phase3wo_done:
     xor edi, edi                              ; DeepStack img counter t_img = 0, edi; incremented only for image tokens
     xor r15d, r15d                            ; Token counter t = 0 in r15d, register; r15 dead after Phase 2
     mov rax, pq_x_storage                     ; x_ptr,0 = x_storage, advanced incrementally at the loop tail
     mov pq_x_ptr, rax                         ; init Phase-3ffn running x pointer
; DeepStack gate, loop-invariant per layer : active only for layers 0/1/2 with feature injection on
; hoisted out of the per-token loop: the per-token cost would be ~11 instructions
     xor r12d, r12d                            ; ds_active = 0, default: skip
     cmp dword [rel g_cfg_has_deepstack], 0    ; Is DeepStack feature injection enabled?
     je .pq_ds_gate_done                       ; No, text-only model
     cmp dword [rel g_image_token_id], 0       ; Is the vision image-token override active?
     je .pq_ds_gate_done                       ; No, image id unset
     mov eax, pq_layer_idx                     ; Load layer index for the DeepStack layer gate
     cmp eax, 3                                ; Is this one of the first 3 layers?
     jge .pq_ds_gate_done                      ; No, only layers 0/1/2 receive features
     mov r12d, 1                               ; ds_active = 1: this layer injects image features
.pq_ds_gate_done:
     mov esi, dword [rel g_image_token_id]     ; cache the image-pad id in the dead non-volatile, rsi is last written well above this point and not read again until SF_END, so the per-token compare below costs no L1 load, the same hoist the embed loop does at its entry
     test r12d, r12d                           ; Does this layer inject per-token DeepStack features?
     jnz .pq3ffn_chk                           ; Use the existing per-token route for those layers
     cmp r15d, pq_ffn_tile_end                 ; Is a full FFN tile available?
     jge .pq3ffn_chk                           ; Process the remaining tokens individually
.pq_ffn_tile:
     mov eax, pq_ffn_tile_end                  ; Read the full-tile bound
     sub eax, r15d                             ; Count tokens still eligible for the batch
     cmp eax, pq_ffn_limit                     ; Are there enough for a complete panel?
     cmova eax, pq_ffn_limit                   ; Cap the active panel length
     mov pq_ffn_count, eax                     ; Record the number of active tokens
     add eax, PQ4_TILE_TOKENS - 1              ; Round up for the final partial tile
     shr eax, 2                                ; Four slots per descriptor
     mov pq_ffn_tiles, eax                     ; Record the number of active descriptors
     mov dword pq_ffn_packed, 0                ; Shared one-tile inputs retain their original layout
     lea rax, pq4_args                         ; QKV owns the scratch used by the FFN
     cmp dword pq_ffn_limit, PQ4_TILE_TOKENS   ; Is this model limited to a single tile?
     jne .pq_ffn_panel_inputs                  ; Set up the larger packed input workspace
     lea rcx, pq_ffn_args                      ; Address the one-tile descriptor
     vmovdqu ymm0, [rax + Q4B_ARGS_XQ0]        ; Read the four original Q8 input pointers
     vmovdqu [rcx + Q4B_ARGS_XQ0], ymm0        ; Share the QKV input slots
     vmovdqu ymm0, [rax + Q4B_ARGS_XS0]        ; Read the four original scale pointers
     vmovdqu [rcx + Q4B_ARGS_XS0], ymm0        ; Share the QKV scale slots
     vzeroupper                                ; Clear upper lanes before normalizing
     jmp .pq_ffn_begin_norm                    ; The one-tile pointers are ready
.pq_ffn_panel_inputs:
     mov dword pq_ffn_packed, 1                ; The larger panel stages expanded input rows
     mov rax, [rax + Q4B_ARGS_OUT0]            ; Use the retired contiguous QKV outputs
     mov pq_ffn_input, rax                     ; First packed input of this panel
.pq_ffn_begin_norm:
     mov rdi, pq_x_ptr                         ; Walk the four residual input rows
     mov dword pq4_j, 0                        ; Start with the first tile member
.pq_ffn_norm:
     mov eax, pq4_j                            ; Read the member index
     PQ_PANEL_MEMBER                           ; Select this token's descriptor and slot
     cmp dword pq_ffn_limit, PQ4_TILE_TOKENS   ; Does the panel share the four original inputs?
     je .pq_ffn_shared_input                   ; Read those pointers directly
     mov rcx, pq_ffn_input                     ; Read the current packed input address
     mov edx, pq_dim                           ; Quantized activation bytes
     lea r8, [rcx + rdx*PQ_PACK_EXPANSION]     ; Leave room for the expanded activation
     mov [r10 + rax*Q4B_ARGS_QW + Q4B_ARGS_XQ0], rcx ; Stage the activation pointer
     mov [r10 + rax*Q4B_ARGS_QW + Q4B_ARGS_XS0], r8 ; Stage the scale pointer
     mov rdx, pq_ffn_input_stride              ; Load the packed activation stride
     add pq_ffn_input, rdx                     ; Advance the packed input cursor
     jmp .pq_ffn_quantize                      ; Normalize the row into the staged pointers
.pq_ffn_shared_input:
     mov rcx, [r10 + rax*Q4B_ARGS_QW + Q4B_ARGS_XQ0] ; Read the shared input pointer
     mov r8, [r10 + rax*Q4B_ARGS_QW + Q4B_ARGS_XS0] ; Read the shared scale pointer
.pq_ffn_quantize:
     mov rdx, rdi                              ; Residual row to normalize
     mov r9d, pq_dim                           ; Normalization width
     mov r10, [r13 + W_RMS_FFN_WEIGHT]         ; Load the FFN normalization weights
     add r10, pq_layer_dim_off                 ; Select the current layer's weights
     API_CALL v7s_rmsnorm_quantize, rcx, rdx, r8, r9, r10 ; Normalize and quantize this row
     cmp dword pq_ffn_packed, 0                ; Does the FFN panel carry magnitudes?
     je .pq_ffn_norm_done                      ; The shared one-tile input has no packing space
     PQ_PACK_INPUT pq_dim                      ; Precompute magnitudes for gate and up
.pq_ffn_norm_done:
     add rdi, pq_dim_off_stride                ; Advance to the next residual input row
     inc dword pq4_j                           ; Advance the member index
     mov eax, pq4_j                            ; Read the completed member count
     cmp eax, pq_ffn_count                     ; Are all panel inputs ready?
     jb .pq_ffn_norm                           ; Prepare the next input
     PQ_PAD_INPUTS                             ; Complete the final descriptor before dispatch
     PQ_FFN_PROJECT W_W1, pq_q_storage, pq_hidden_dim, pq_dim, pq_ffn_tiles, pq_ffn_packed ; Gate projection panel
     PQ_FFN_PROJECT W_W3, pq_att_out_storage, pq_hidden_dim, pq_dim, pq_ffn_tiles, pq_ffn_packed ; Up projection panel
     mov rdi, pq_q_storage                     ; First tile member's gate vector
     mov dword pq4_j, 0                        ; Start with the first tile member
.pq_ffn_activate:
     mov eax, pq4_j                            ; Read the member index
     imul rax, pq_ffn_stride                   ; Compute the up vector offset
     add rax, pq_att_out_storage               ; Locate this member's up vector
     mov edx, pq_hidden_dim                    ; Number of float lanes to process
.pq_ffn_silu:
     vmovups ymm0, [rdi + rdx*INT32_SCALE - PQ_FFN_SIMD_BYTES] ; Read eight gate values
     EVAL_SIGMOID y, ps, [rel swk_sign_mask_ymm] ; Use the fused kernel's sigmoid arithmetic
     vmulps ymm0, ymm0, [rdi + rdx*INT32_SCALE - PQ_FFN_SIMD_BYTES] ; Multiply sigmoid by gate
     vmulps ymm0, ymm0, [rax + rdx*INT32_SCALE - PQ_FFN_SIMD_BYTES] ; Multiply by the up projection
     vmovups [rdi + rdx*INT32_SCALE - PQ_FFN_SIMD_BYTES], ymm0 ; Overwrite the gate vector with SwiGLU output
     sub edx, PQ_FFN_SIMD_FLOATS               ; Walk back by eight floats
     jnz .pq_ffn_silu                          ; Continue until the vector is complete
     vzeroupper                                ; Clear upper lanes before the quantizer call
     mov rcx, rax                              ; Up row is now available for the quantized W2 input
     mov r9d, pq_hidden_dim                    ; Quantize the full hidden vector
     lea r8, [rcx + r9*PQ_PACK_EXPANSION]      ; Packed values and scales fit inside the retired up row
     cmp dword pq_ffn_packed, 0                ; Are the down-projection inputs packed too?
     jne .pq_ffn_scales_ready                  ; Keep the expanded scale offset
     lea r8, [rcx + r9]                        ; Use the original single-tile scale offset
.pq_ffn_scales_ready:
     mov eax, pq4_j                            ; Read the member index
     PQ_PANEL_MEMBER                           ; Select this token's W2 descriptor and slot
     mov rdx, rdi                              ; SwiGLU output to quantize
     mov [r10 + rax*Q4B_ARGS_QW + Q4B_ARGS_XQ0], rcx ; Stage the W2 activation pointer
     mov [r10 + rax*Q4B_ARGS_QW + Q4B_ARGS_XS0], r8 ; Stage the W2 scale pointer
     API_CALL v7d_quantize, rcx, rdx, r8, r9   ; Prepare this input for the down projection
     cmp dword pq_ffn_packed, 0                ; Does the down-projection panel use packed inputs?
     je .pq_ffn_activation_done                ; Keep the original one-tile input
     PQ_PACK_INPUT pq_hidden_dim               ; Precompute down-projection activation magnitudes
.pq_ffn_activation_done:
     add rdi, pq_ffn_stride                    ; Advance to the next gate vector
     inc dword pq4_j                           ; Advance the member index
     mov eax, pq4_j                            ; Read the completed member count
     cmp eax, pq_ffn_count                     ; Are all panel hidden vectors quantized?
     jb .pq_ffn_activate                       ; Apply SwiGLU to the next member
     PQ_PAD_INPUTS                             ; Complete the final descriptor before dispatch
     PQ_FFN_PROJECT W_W2, pq_q_storage, pq_dim, pq_hidden_dim, pq_ffn_tiles, pq_ffn_packed ; Down projection panel
     mov rdi, pq_q_storage                     ; First tile member's down-projection output
     mov dword pq4_j, 0                        ; Start with the first tile member
.pq_ffn_residual:
     mov rcx, pq_x_ptr                         ; Residual destination for this member
     mov rdx, rdi                              ; Down-projection output for this member
     mov r8d, pq_dim                           ; Vector length in floats
     API_CALL v7s_add_inplace, rcx, rdx, r8    ; Add W2 output to the residual
     mov rax, pq_dim_off_stride                ; Residual and W2 rows have the same stride
     add pq_x_ptr, rax                         ; Advance the residual row pointer
     add rdi, rax                              ; Advance the W2 output pointer
     inc dword pq4_j                           ; Advance the member index
     mov eax, pq4_j                            ; Read the completed member count
     cmp eax, pq_ffn_count                     ; Are all panel residuals updated?
     jb .pq_ffn_residual                       ; Add the next tile member's output
     add r15d, pq_ffn_count                    ; Advance past the completed panel tokens
     cmp r15d, pq_ffn_tile_end                 ; Is another full FFN tile available?
     jl .pq_ffn_tile                           ; Process the next tile
     jmp .pq3ffn_chk                           ; jump to the bottom-test on entry, loop inversion
.pq_phase3ffn_loop:
; x_ptr is maintained incrementally, slot advanced at the loop tail
; === FFN RMSNorm + quantize, fused v7s_rmsnorm_quantize : x[t] -> xq ===
     mov rcx, [r14 + S_XQ_Q]                   ; rcx = s->xq.q as quantized output
     mov rdx, pq_x_ptr                         ; rdx = per-token x pointer as RMSNorm input
     mov r8, [r14 + S_XQ_S]                    ; r8 = s->xq.s as scales output
     mov r9d, pq_dim                           ; r9 = dim as vector length
     mov rax, [r13 + W_RMS_FFN_WEIGHT]         ; rax = w->rms_ffn_weight
     add rax, pq_layer_dim_off                 ; rax = rms_ffn_weight[layer], byte offset
     mov r10, rax                              ; r10 = rms_ffn_weight[layer], arg5
     API_CALL v7s_rmsnorm_quantize, rcx, rdx, r8, r9, r10 ; fused RMSNorm+quantize: x[t] -> xq, eps from BSS
; === SwiGLU Q8: swiglu_q8,xq, w1[l], w3[l] -> hb ===
; v7d_swiglu_q8,s->hb, &s->xq, w1[layer], dim, hidden_dim, w3[layer]
     mov rax, pq_layer_bytes                   ; rax = per-layer tensor-slot byte stride, 16, added to each LOADED
; pointer below; it must NEVER be folded into the r13 struct base
; that would read a DIFFERENT TransformerWeights field per layer
     mov rcx, [r14 + S_HB]                     ; rcx = s->hb as SwiGLU output buffer
     lea rdx, [r14 + S_XQ_Q]                   ; rdx = &s->xq as quantized input tensor
     mov r8, [r13 + W_W1]                      ; r8 = w->w1 base, gated up-projection
     add r8, rax                               ; r8 = w1[l] = w1 + layer_idx * 16
     mov r9d, pq_dim                           ; r9 = input dimension
     mov r10d, pq_hidden_dim                   ; r10 = hidden dimension
     mov r11, [r13 + W_W3]                     ; r11 = w->w3 base, gate activation
     add r11, rax                              ; r11 = w3[l] = w3 + layer_idx * 16
     API_CALL v7d_swiglu_q8, rcx, rdx, r8, r9, r10, r11 ; SwiGLU Q8 on gated FFN up-projection
; === W2: fused quantize+Q8,hb, w2[layer] -> xb ===
; v7d_q8_fused,s->xb, &ft_x, w2[layer], hidden_dim, dim
     mov rax, [r14 + S_HB]                     ; rax = s->hb, float* SwiGLU output
     mov rcx, [r14 + S_XB]                     ; rcx = s->xb as output buffer
     PQ_SET_FT_X                               ; ft.x = s->hb; rdx = &ft, staging macro
     mov r8, [r13 + W_W2]                      ; r8 = w->w2 base, down-projection
     add r8, pq_layer_bytes                    ; r8 = w2[l] = w2 + layer_idx * 16, slot reload: rax was overwritten by the W2 staging above
     mov r9d, pq_hidden_dim                    ; r9 = n, input dimension, from SwiGLU
     mov r10d, pq_dim                          ; r10 = d, output dimension, back to embedding size
     API_CALL v7d_q8_fused, rcx, rdx, r8, r9, r10 ; Fused W2 down-projection: quantize+int8 matmul, output to xb
; === Residual: x += xb ===
     mov rcx, pq_x_ptr                         ; rcx = per-token x pointer
     mov rdx, [r14 + S_XB]                     ; rdx = s->xb as residual to add
     mov r8d, pq_dim                           ; r8 = vector length
     API_CALL v7s_add_inplace, rcx, rdx, r8    ; x[t] += s->xb, residual connection
     test r12d, r12d                           ; DeepStack active for this layer?
     jz .pq_ds_done                            ; No, skip everything, 2 instr instead of ~8
     mov rax, pq_tokens_ptr                    ; Load tokens array pointer for tokens[t] access
     cmp dword [rax + r15*INT32_SCALE], esi    ; is tokens[t] the image-pad token?, ESI caches the id from the ds_active gate above
     jne .pq_ds_done                           ; No, plain token, skip DeepStack
     mov r10d, pq_emb_base                     ; r10 = embedding rows consumed before this prefill
     add r10d, edi                             ; r10 = row index of this image token in DeepStack tensors
     mov eax, r10d                             ; 64-bit row index into the DeepStack tensor
     mov ecx, dword pq_dim                     ; dim, zero-extended: positive config dim
     imul rax, rcx                             ; rax = row * dim, element offset
     mov rdx, rax                              ; rdx = element offset into DeepStack tensor
     mov rcx, pq_x_ptr                         ; rcx = per-token x pointer as DeepStack add destination
     mov r8d, pq_dim                           ; r8 = vector length for add_inplace
     lea rax, [rel g_deepstack_features]       ; rax = base of 3-qword DeepStack pointer array
     mov r9d, pq_layer_idx                     ; r9 = layer index, reloaded here: only live on the image-token path
     mov rax, [rax + r9*VDS_PTR_STRIDE]        ; indexed load: slot 0/1/2 at qword offsets 0/8/16, slots are filled for layers 0..2 whenever the gate is set
     lea rdx, [rax + rdx*INT32_SCALE]          ; rdx = ds + row*dim*4, byte address
     API_CALL v7s_add_inplace, rcx, rdx, r8    ; x[t] += ds[layer][row], DeepStack feature add
     inc edi                                   ; Advance per-layer image-token row counter
.pq_ds_done:
; === Next FFN token: advance the incremental x pointer ===
     mov rax, pq_dim_off_stride                ; x_ptr += dim*4
     add pq_x_ptr, rax                         ; advance the running x pointer
     inc r15d                                  ; Advance token counter, register
.pq3ffn_chk:
     cmp r15d, pq_n_tokens                     ; Compare token counter against total, direct register cmp
     jl .pq_phase3ffn_loop                     ; Continue while more tokens remain
.pq_phase3_done:
; === Next layer: advance the incremental per-layer offsets, then loop ===
     inc dword pq_layer_idx                    ; Advance to next transformer layer
     mov rax, pq_loff_stride                   ; Load KV stride, mem-to-mem add needs a register hop
     add pq_loff, rax                          ; loff += seq_len * kv_dim, float units
     add qword pq_layer_bytes, PQ_LAYER_TENSOR_BYTES ; layer_bytes += 16, sizeof tensor_f32
     mov rax, pq_dim_off_stride                ; Load RMS weight stride
     add pq_layer_dim_off, rax                 ; layer_dim_off += dim * 4
     mov rax, pq_hd_off_stride                 ; Load norm weight stride
     add pq_layer_hd_off, rax                  ; layer_hd_off += head_dim * 4
     mov rax, pq_dsum_bytes                    ; Load bias stride
     add pq_q_bias_base, rax                   ; q_bias_base += dsum * 4
     add pq_k_bias_base, rax                   ; k_bias_base += dsum * 4
     add pq_v_bias_base, rax                   ; v_bias_base += dsum * 4
     jmp .pq_layer_loop                        ; Continue to next layer iteration
.pq_after_layers:
; === Done with layers ===
; === Final RMSNorm on last position ===
; x_last = x_storage +, n_tokens - 1 * dim
     mov rax, pq_x_storage                     ; Load x_storage base pointer
     mov ecx, pq_n_tokens                      ; Load n_tokens
     dec ecx                                   ; ecx = n_tokens - 1, last token index
     mov r10d, pq_dim                          ; Load dim
     imul rcx, r10                             ; rcx =, n_tokens - 1 * dim, float units
     lea rax, [rax + rcx*INT32_SCALE]          ; rax = x_storage +, n_tokens-1 *dim*4, last token ptr
     mov rcx, r14                              ; rcx = RunState*, cached in non-volatile
     mov [rcx], rax                            ; s->x = x_last
     mov rdx, rax                              ; rdx = x_last, same as s->x now
     mov rax, [r13 + W_RMS_FINAL_WEIGHT]       ; rax = w->rms_final_weight, direct from cached w_ptr
     mov r8, rax                               ; r8 = final RMSNorm weight pointer
     mov r9d, pq_dim                           ; r9 = dim as vector length
     API_CALL v7s_rmsnorm, rdx, rdx, r8, r9    ; Final RMSNorm on last token: s->x = RMSNorm,x_last
; === Classifier on last position, fused ===
; v7d_q8_fused,s->logits, &ft_x, wcls, dim, vocab_size
     mov rax, [r14]                            ; rax = s->x, float* final activations
     mov rcx, [r14 + S_LOGITS]                 ; rcx = s->logits as output buffer
     PQ_SET_FT_X                               ; ft.x = s->x; rdx = &ft, staging macro
     mov r8, [r13 + W_WCLS]                    ; r8 = classifier weight tensor, w->wcls
     mov r9d, pq_dim                           ; r9 = n, input dimension
     mov r10d, pq_vocab_size                   ; r10 = d, output dimension, vocabulary size
     API_CALL v7d_q8_fused, rcx, rdx, r8, r9, r10 ; Fused classifier: quantize+int8 matmul, output logits for last token
; === Restore original s->x pointer, was hijacked to x_storage for final RMSNorm ===
     mov rcx, pq_saved_x                       ; Load saved original s->x pointer
     mov [r14], rcx                            ; Restore s->x to original standalone buffer
; === Return s->logits ===
     call v7s_mrope_restore                    ; put base rope tables back, no-op when not swapped
     mov rax, [r14 + S_LOGITS]                 ; rax = s->logits pointer as return value
     SF_END rbx, rsi, rdi, r15, r12, r13, r14  ; Epilogue: restore non-volatile regs, incl. rsi and return
; ============================================================================
; Clean up SF_* slot names from prefill_forward_q8
; ============================================================================
%undef pq4_fork_members                        ; file-local batch-4 fork descriptor array slot
%undef PQ4_FORK_RBP_DISP                       ; captured array displacement, sites expanded
%undef PQ4_FORK_MEMBER1_OFF                    ; descriptor byte offsets, sites expanded
%undef PQ4_FORK_MEMBER2_OFF
%undef PQ4_FORK_MEMBER3_OFF
%undef pq_cfg_ptr
%undef pq_cfg_ptr_raw
%undef pq_dim
%undef pq_dim_raw
%undef pq_hidden_dim
%undef pq_hidden_dim_raw
%undef pq_n_layers
%undef pq_n_layers_raw
%undef pq_seq_len
%undef pq_seq_len_raw
%undef pq_n_heads
%undef pq_n_heads_raw
%undef pq_n_kv_heads
%undef pq_n_kv_heads_raw
%undef pq_head_dim
%undef pq_head_dim_raw
%undef pq_vocab_size
%undef pq_vocab_size_raw
%undef pq_kv_dim
%undef pq_kv_dim_raw
%undef pq_kv_mul
%undef pq_kv_mul_raw
%undef pq_all_heads_dim
%undef pq_all_heads_dim_raw
%undef pq_n_tokens
%undef pq_n_tokens_raw
%undef pq_start_pos
%undef pq_start_pos_raw
%undef pq_tokens_ptr
%undef pq_tokens_ptr_raw
%undef pq_layer_idx
%undef pq_layer_idx_raw
%undef pq_layer_bytes
%undef pq_layer_bytes_raw
%undef pq_layer_dim_off
%undef pq_layer_dim_off_raw
%undef pq_layer_hd_off
%undef pq_layer_hd_off_raw
%undef pq_loff
%undef pq_loff_raw
%undef pq_x_storage
%undef pq_x_storage_raw
%undef pq_q_storage
%undef pq_q_storage_raw
%undef pq_att_out_storage
%undef pq_att_out_storage_raw
%undef pq_saved_xb
%undef pq_saved_xb_raw
%undef pq_saved_q
%undef pq_saved_q_raw
%undef pq_saved_x
%undef pq_saved_x_raw
%undef pq_x_ptr
%undef pq_x_ptr_raw
%undef pq_rope_max_slot
%undef pq_rope_max_slot_raw
%undef pq_pos_slot
%undef pq_pos_slot_raw
%undef pq_has_qk_norm
%undef pq_has_qk_norm_raw
%undef pq_bias_active
%undef pq_bias_active_raw
%undef pq_emb_base
%undef pq_emb_base_raw
%undef pq_ft_s
%undef pq_ft_s_raw
%undef pq_ft_x
%undef pq_ft_x_raw
%undef pq_q_bias_base
%undef pq_q_bias_base_raw
%undef pq_k_bias_base
%undef pq_k_bias_base_raw
%undef pq_v_bias_base
%undef pq_v_bias_base_raw
%undef pq_q_dst
%undef pq_q_dst_raw
%undef pq_loff_stride
%undef pq_loff_stride_raw
%undef pq_dim_off_stride
%undef pq_dim_off_stride_raw
%undef pq_hd_off_stride
%undef pq_hd_off_stride_raw
%undef pq_dsum_bytes
%undef pq_dsum_bytes_raw
%undef pq_ahd_bytes
%undef pq_ahd_bytes_raw
; ---- %undef cleanup: file-local constants, the SF block above removes all 90 slot names ----
%undef INT32_SCALE
%undef PQ_FP16_BYTES
%undef PQ_KV_DOUBLED
%undef PQ_LAYER_TENSOR_BYTES
%undef PQ4_TILE_TOKENS                         ; tokens per batch-4 tile
%unmacro PQ_PACK_KV_F16 2-3                    ; all four commit sites expanded, file-local macro leaves scope
%undef PQ4_API_PAD                             ; scratch/API pad below rsp
%undef PQ4_PROBE_PAGE                          ; guard-page probe chunk
%undef Q4B_ARGS_OUT0                           ; args block offsets, batch-4 tile path
%undef Q4B_ARGS_OUT1
%undef Q4B_ARGS_OUT2
%undef Q4B_ARGS_OUT3
%undef Q4B_ARGS_XQ0
%undef Q4B_ARGS_XQ1
%undef Q4B_ARGS_XQ2
%undef Q4B_ARGS_XQ3
%undef Q4B_ARGS_XS0
%undef Q4B_ARGS_XS1
%undef Q4B_ARGS_XS2
%undef Q4B_ARGS_XS3
%undef Q4B_ARGS_WQ
%undef Q4B_ARGS_WS
%undef Q4B_ARGS_N
%undef Q4B_ARGS_D
%undef Q4B_ARGS_QW
%undef Q4B_ROUND
%undef Q4B_ALIGN
%undef Q4B_TILE_MASK

%undef pq_ffn_args
%undef pq_ffn_tile_end
%undef pq_ffn_tile_end_raw
%undef pq_ffn_stride
%undef pq_ffn_stride_raw
%unmacro PQ_FFN_PROJECT 6                      ; End the FFN projection macro scope
%undef PQ_FFN_SIMD_BYTES
%undef PQ_FFN_SIMD_FLOATS
%undef pq_wo_out
%undef pq_wo_out_raw
%undef PQ_ATT_PANEL_FRAME
%undef PQ_ATT_PANEL_ARG9

%unmacro PQ_PANEL_MEMBER 0                     ; End the descriptor addressing macro scope
%undef PQ_PANEL_TILES
%undef PQ_PANEL_TOKENS
%undef pq_ffn_limit
%undef pq_ffn_limit_raw
%undef pq_ffn_count
%undef pq_ffn_count_raw
%undef pq_ffn_tiles
%undef pq_ffn_tiles_raw
%undef pq_ffn_input
%undef pq_ffn_input_raw
%undef pq_ffn_input_stride
%undef pq_ffn_input_stride_raw
%undef pq_qkv_scratch
%undef pq_qkv_scratch_raw
%undef pq_qkv_limit
%undef pq_qkv_limit_raw
%undef pq_qkv_pending
%undef pq_qkv_pending_raw
%undef pq_qkv_next
%undef pq_qkv_next_raw
%undef pq_qkv_out
%undef pq_qkv_out_raw

%unmacro PQ_PANEL_CAPACITY 1                   ; End the workspace sizing macro scope
%undef pq_wo_limit
%undef pq_wo_limit_raw
%undef pq_wo_input_stride
%undef pq_wo_input_stride_raw
%undef pq_ffn_max_output
%undef pq_ffn_max_output_raw
%undef pq_c_packed
%undef pq_c_packed_raw
%undef pq_c_desc
%undef pq_c_desc_raw
%undef pq_c_count
%undef pq_c_count_raw
%undef pq_c_panel
%undef pq_c_panel_raw

%unmacro PQ_PACK_INPUT 1                       ; End the activation packing macro scope
%undef PQ_PACK_VECTOR_BYTES
%undef pq_ffn_packed
%undef pq_ffn_packed_raw

%unmacro PQ_PAD_INPUTS 0                       ; End the final-tile pointer helper
%undef PQ_PREVIOUS_ROW
