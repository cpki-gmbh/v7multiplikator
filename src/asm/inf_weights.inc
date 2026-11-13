; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; Local constants for offset arithmetic
%define IDX_0 0                                ; Zero index for array access
%define ONE_INT 1                              ; Literal 1 for arithmetic
%define WM_DWORD_SCALE 4                       ; SIB scale: one float / one dword element
%define WM_CEIL_BIAS (GS - 1)                  ; ceiling alignment bias for the group-scale count, GS - 1
%define FUSED_KV_COUNT 2                       ; K and V, two tensors in the fused QKV weight block
%define BIAS_SCAN_QWORD_BYTES 8                ; qword advance of the bias-block OR-scan cursor
%define BIAS_SCAN_GRP_BYTES 32                 ; bytes per 256-bit OR group, one 32-byte YMM load
%define BIAS_SCAN_GRP2_OFF 64                  ; byte offset of the 3rd OR group within the iteration, 2*32
%define BIAS_SCAN_GRP3_OFF 96                  ; byte offset of the 4th OR group within the iteration, 3*32
%define BIAS_SCAN_ITER_BYTES 128               ; bytes per full OR-scan iteration, 4 groups of 32

%macro BUILD_QT_PER_LAYER 3-4
     mov r13, r14                              ; r13 = desc_base, saved for weights[] store
     mov eax, inf_dim                          ; eax = dim, zero-extends to rax
     mov r11d, %3                              ; r11d = middle factor, n_heads / n_kv_heads / hidden_dim
     imul rax, r11                             ; rax = dim * mid, 64-bit
%if %0 > 3
     mov r11d, %4                              ; r11d = head_dim, only for 4-arg form
     imul rax, r11                             ; rax = dim * mid * head_dim = pe_q, 64-bit
%endif
     mov r15, rax                              ; r15 = pe_q, full 64-bit
     lea rax, [r15 + r8 - ONE_INT]             ; rax = pe_q + GS - 1, 64-bit
     shr rax, 6                                ; rax = ceil,pe_q / 64, power-of-2 divide, 1 cycle vs 35-80 for div
     lea rax, [r15 + rax*WM_DWORD_SCALE]       ; rax = stride_bytes = s_bytes + pe_q, 64-bit
     mov r9, rax                               ; r9 = stride, full 64-bit
     mov rax, [rsi + %1 * WM_ENTRY_SIZE + WM_FILE_OFFSET] ; rax = file offset
     add rax, rbx                              ; rax = layer 0 q_ptr = model_weights + file_off
     mov r10d, r12d                            ; r10d = n_layers counter
%%qt_loop:
     mov [r14 + QT_Q], rax                     ; qt[l].q = q_ptr
     lea r11, [rax + r15]                      ; r11 = q_ptr + pe_q = s_ptr
     mov [r14 + QT_S], r11                     ; qt[l].s = s_ptr
     add rax, r9                               ; q_ptr += stride
     add r14, QT_SIZE                          ; desc_ptr++
     dec r10d                                  ; --n_layers counter
     jnz %%qt_loop                             ; loop while layers remain
     mov [rdi + %2], r13                       ; weights[<arg2>] = desc_base
%endmacro
; SET_FP32_PTR wm_index, w_store_offset
; One FP32 weight pointer: weights[W] = model_weights + wm[W].file_offset.
; Leaves the pointer in rax, the k_norm invocation feeds the qkv_bias math below .

%macro SET_FP32_PTR 2
     mov rax, [rsi + %1 * WM_ENTRY_SIZE + WM_FILE_OFFSET] ; file offset from weight_meta
     add rax, rbx                              ; + model_weights base
     mov [rdi + %2], rax                       ; store pointer into weights
%endmacro
; FUSED_QKV_COPY desc_reg, field_const, count_reg
; One block copy inside .qkv_loop: rsi = src, rcx = qword count, pre-shifted
; before the loop, rdi = persistent write cursor, rep movsq advances it .

%macro FUSED_QKV_COPY 3
     mov rsi, [%1 + %2]                        ; src = per-tensor q or s data
     mov rcx, %3                               ; qword count, hoisted shift
     rep movsq                                 ; copy block; rdi advances
%endmacro
; === Step 6: Copy config into Transformer ===
     mov rax, inf_transformer                  ; Reload pointer, fresh at file boundary, prior step ended with BSS/frame stores
     mov rdi, rax                              ; dst = Transformer
     mov rsi, inf_cfg_ptr                      ; src = microdb Config, fresh pointer
     mov ecx, CFG_SIZE_QWORDS                  ; qword count of the Config struct
     rep movsq                                 ; Copy config from microdb to Transformer
; === Step 7: Get model_weights from microdb ===
     lea rcx, [rel inf_key_model_weights]      ; rcx = "__sys_model_weights" string key
     xor edx, edx                              ; out_len = NULL
     call ht_find_or_fatal                     ; Find model_weights entry, FATAL with key name on miss
     mov inf_mw_ptr, rax                       ; Direct pointer to weight data in microdb heap
; === Step 8: Get weight_meta from microdb ===
     lea rcx, [rel inf_key_weight_meta]        ; rcx = "__sys_weight_meta" string key
     xor edx, edx                              ; out_len = NULL
     call ht_find_or_fatal                     ; Find weight_meta entry, FATAL with key name on miss
     mov inf_wm_ptr, rax                       ; Save weight_meta pointer
     mov eax, inf_n_layers                     ; Load n_layers count, 32-bit read zero-extends rax
     imul rax, rax, 7                          ; 7 quantized per-layer tensors, wq/wk/wv/wo/w1/w2/w3
     add rax, 2                                ; rax = 7 * n_layers + 2
     shl rax, 4                                ; rax =, 7*n_layers + 2 * 16, bytes
     mov rcx, rax                              ; arg1: descriptor block size
     call mdb_alloc_mem                        ; rax = block base, FATALs internally on OOM
     mov inf_qt_desc, rax                      ; Cache pointer in SF frame slot
     mov [g_inf_qt_desc], rax                  ; Track in BSS for cleanup
     mov rdi, inf_transformer                  ; Load Transformer base
     lea rdi, [rdi + TRANSFORMER_WEIGHTS]      ; rdi = &weights
; === Step 9a: Set up FP32 weight pointers ===
     mov rsi, inf_wm_ptr                       ; rsi = &wm[0]
     mov rbx, inf_mw_ptr                       ; rbx = model_weights
; wm[0]: rms_att_weight
     SET_FP32_PTR WM_RMS_ATT, W_RMS_ATT_WEIGHT ; weights->rms_att_weight = mw + wm[0].file_off
; wm[1]: rms_ffn_weight
     SET_FP32_PTR WM_RMS_FFN, W_RMS_FFN_WEIGHT ; weights->rms_ffn_weight = mw + wm[1].file_off
; wm[2]: rms_final_weight
     SET_FP32_PTR WM_RMS_FINAL, W_RMS_FINAL_WEIGHT ; weights->rms_final_weight = mw + wm[2].file_off
; wm[3]: q_norm_weights
     SET_FP32_PTR WM_Q_NORM, W_Q_NORM_WEIGHTS  ; weights->q_norm_weights = mw + wm[3].file_off
; wm[4]: k_norm_weights, rax left holding the pointer: feeds qkv_bias below
     SET_FP32_PTR WM_K_NORM, W_K_NORM_WEIGHTS  ; weights->k_norm_weights = mw + wm[4].file_off
     mov ecx, inf_n_layers                     ; ecx = n_layers, zero-extends to rcx
     mov edx, inf_head_dim                     ; rdx = head_dim, zero-extends to 64-bit, matches the file's other config loads
     imul rcx, rdx                             ; rcx = n_layers * head_dim, 64-bit per S45
     lea rax, [rax + rcx*WM_DWORD_SCALE]       ; rax = k_norm_weights + n_layers*head_dim*4 = qkv_bias base
     mov [rel g_qkv_bias], rax                 ; Store qkv_bias base pointer in BSS global
     mov rdx, rax                              ; rdx = scan cursor = bias base, LIVE through the scan, r12 is the chain scratch
     mov eax, inf_n_heads                      ; eax = n_heads, zero-extends to rax
     mov ecx, inf_n_kv_heads                   ; ecx = n_kv_heads, zero-extends to rcx
     lea rax, [rax + rcx*FUSED_KV_COUNT]       ; rax = n_heads + 2*n_kv_heads, Q + K + V slice count
     mov r8d, inf_head_dim                     ; r8d = head_dim, zero-extends to r8
     imul rax, r8                              ; rax = dsum =, n_heads + 2*n_kv_heads * head_dim, 64-bit product
     mov r12d, inf_n_layers                    ; r12d = n_layers, zero-extends to r12; re-bound at Step 9b below
     imul rax, r12                             ; rax = total bias floats = n_layers * dsum, 64-bit
     shr rax, 1                                ; floats -> qwords, 2 floats per qword for the scan end
     lea r9, [rdx + rax*BIAS_SCAN_QWORD_BYTES] ; r9 = scan end, one past the last qword, 64-bit index
     vpxor ymm0, ymm0, ymm0                    ; OR accumulator A, streams 0 and 2
     vpxor ymm1, ymm1, ymm1                    ; OR accumulator B, streams 1 and 3, volatile Win64 XMM1
.bias_scan_loop:
     vpor ymm0, ymm0, [rdx]                    ; stream A, first 32 bytes
     vpor ymm1, ymm1, [rdx + BIAS_SCAN_GRP_BYTES] ; stream B, next 32 bytes
     vpor ymm0, ymm0, [rdx + BIAS_SCAN_GRP2_OFF] ; stream A, third 32 bytes
     vpor ymm1, ymm1, [rdx + BIAS_SCAN_GRP3_OFF] ; stream B, fourth 32 bytes
     sub rdx, -BIAS_SCAN_ITER_BYTES            ; advance 128 bytes, imm8 encoding of the same +128
     cmp rdx, r9                               ; cursor reached the end?
     jb .bias_scan_loop                        ; no: keep scanning
     vpor ymm0, ymm0, ymm1                     ; combine the two OR chains, same byte mask
     xor eax, eax                              ; eax = 0, bias inactive
     vptest ymm0, ymm0                         ; ZF set iff EVERY scanned byte is zero, vptest with identical operands ANDs ymm0 with itself
     setne al                                  ; eax = 1 when a real bias exists
     mov [rel g_qkv_bias_active], eax          ; publish activity flag for forward_q8
     CLEANUP_YMM                               ; vzeroupper, this is the first YMM register v7s_inference_create touches, the file's other steps are integer rep movsq, and the ABI requires the upper halves cleared before returning to the caller
; === Step 9b: q_tokens, wm[5], 1 instance ===
     mov r14, inf_qt_desc                      ; r14 = qt_descs base
     mov r12d, inf_n_layers                    ; r12d = n_layers for per-layer stride calc
; q_tokens: wm[5], 1 instance, stored at qt_descs + 0
; pe_q = vocab_size * dim, int8 elements before scales start
     mov eax, inf_vocab_size                   ; eax = vocab_size, zero-extends to rax
     mov r15d, inf_dim                         ; r15d = dim, zero-extends to r15
     imul rax, r15                             ; rax = vocab_size * dim = pe_q, 64-bit
     mov r15, rax                              ; r15 = pe_q, full 64-bit
     mov rax, [rsi + WM_Q_TOKENS * WM_ENTRY_SIZE + WM_FILE_OFFSET] ; q_tokens file offset
     add rax, rbx                              ; q_ptr = model_weights + file_off
     mov [r14 + IDX_0*QT_SIZE + QT_Q], rax     ; q_tokens[0].q = q_ptr
     lea r11, [rax + r15]                      ; r11 = q_ptr + pe_q = s_ptr
     mov [r14 + IDX_0*QT_SIZE + QT_S], r11     ; q_tokens[0].s = q_ptr + pe_q
; Store q_tokens descriptor pointer in weights struct
     mov [rdi + W_Q_TOKENS], r14               ; w->q_tokens = qt_descs + 0
; === Step 9c: Build per-layer quantized tensor descriptors ===
; Advance past q_tokens, 1 entry at offset 0
     add r14, QT_SIZE                          ; r14 = start of wq descriptors
     mov r8d, inf_group_size                   ; r8d = GS, the macro's ceil-div input below, GS == 64 is the builder's contract
     BUILD_QT_PER_LAYER WM_WQ, W_WQ, inf_n_heads, inf_head_dim ; wq: pe_q = dim * n_heads * head_dim
     BUILD_QT_PER_LAYER WM_WK, W_WK, inf_n_kv_heads, inf_head_dim ; wk: pe_q = dim * n_kv_heads * head_dim
     BUILD_QT_PER_LAYER WM_WV, W_WV, inf_n_kv_heads, inf_head_dim ; wv: pe_q = dim * n_kv_heads * head_dim
     BUILD_QT_PER_LAYER WM_WO, W_WO, inf_n_heads, inf_head_dim ; wo: pe_q = dim * n_heads * head_dim
     BUILD_QT_PER_LAYER WM_W1, W_W1, inf_hidden_dim ; w1: pe_q = dim * hidden_dim
     BUILD_QT_PER_LAYER WM_W2, W_W2, inf_hidden_dim ; w2: pe_q = dim * hidden_dim
     BUILD_QT_PER_LAYER WM_W3, W_W3, inf_hidden_dim ; w3: pe_q = dim * hidden_dim
     cmp inf_shared_cls, 0                     ; check if classifier is shared
     jne .qt_wcls_shared                       ; if shared, reuse q_tokens desc
; Non-shared wcls: pe_q = dim * vocab_size
     mov r13, r14                              ; r13 = wcls_desc_base
     mov eax, inf_dim                          ; eax = dim for pe_q, zero-extends to rax
     mov r15d, inf_vocab_size                  ; r15d = vocab_size, zero-extends to r15
     imul rax, r15                             ; rax = dim * vocab_size = pe_q, 64-bit
     mov r15, rax                              ; r15 = pe_q, full 64-bit
     mov rax, [rsi + WM_WCLS * WM_ENTRY_SIZE + WM_FILE_OFFSET] ; rax = wcls file offset
     add rax, rbx                              ; rax = wcls q_ptr
     mov [r14 + QT_Q], rax                     ; qt.wcls.q = q_ptr
     lea r11, [rax + r15]                      ; r11 = q_ptr + pe_q = s_ptr
     mov [r14 + QT_S], r11                     ; qt.wcls.s = s_ptr
     add r14, QT_SIZE                          ; desc_ptr++
     mov [rdi + W_WCLS], r13                   ; w->wcls = wcls_desc_base
     jmp .qt_wcls_done                         ; done with wcls descriptor
.qt_wcls_shared:
     mov rax, [rdi + W_Q_TOKENS]               ; w->q_tokens pointer
     mov [rdi + W_WCLS], rax                   ; w->wcls = w->q_tokens
.qt_wcls_done:
     mov eax, inf_n_heads                      ; eax = n_heads, 32-bit read zero-extends rax
     mov ecx, inf_head_dim                     ; ecx = head_dim, 32-bit read zero-extends rcx
     imul rax, rcx                             ; rax = all_heads_dim
     mov r8d, inf_n_kv_heads                   ; r8 = n_kv_heads, 32-bit read zero-extends; volatile scratch here
     mov r10d, inf_head_dim                    ; r10 = head_dim, 32-bit read zero-extends
     imul r8, r10                              ; r8 = kv_dim
     mov r9d, inf_dim                          ; r9d = dim, zero-extends r9
     imul rax, r9                              ; rax = pe_q_wq = all_heads_dim*dim, 64-bit
     mov rbx, rax                              ; stash pe_q_wq in rbx, survives alloc
     imul r8, r9                               ; r8 = pe_q_wk = kv_dim*dim, 64-bit
     mov r13, r8                               ; stash pe_q_wk in r13, survives alloc
; --- s_bytes: ceil,pe/GS *4 per tensor ---
     lea rax, [rbx + WM_CEIL_BIAS]             ; pe_q_wq + GS - 1, ceiling alignment
     shr rax, 6                                ; rax = ceil,pe_q_wq / 64, 1 cycle vs 35-80 for div
     shl rax, 2                                ; rax = s_wq, scale bytes
     mov r14, rax                              ; stash s_wq in r14, survives alloc
     lea rax, [r13 + WM_CEIL_BIAS]             ; pe_q_wk + GS - 1, ceiling alignment
     shr rax, 6                                ; rax = ceil,pe_q_wk / 64, 1 cycle vs 35-80 for div
     shl rax, 2                                ; rax = s_wk, scale bytes
     mov r15, rax                              ; stash s_wk in r15, survives alloc
; --- total = n_layers *, pe_q_wq + 2*pe_q_wk + s_wq + 2*s_wk ---
     lea rax, [rbx + r13*FUSED_KV_COUNT]       ; rax = pe_q_wq + 2*pe_q_wk
     add rax, r14                              ; rax += s_wq
     lea rax, [rax + r15*FUSED_KV_COUNT]       ; rax = fused_stride, bytes per layer
     mov ecx, inf_n_layers                     ; ecx = n_layers
     imul rax, rcx                             ; rax = total fused data bytes, 64-bit
     mov rcx, rax                              ; arg1: total fused data bytes
     call mdb_alloc_mem                        ; rax = fused data buffer, FATALs internally on OOM
     mov inf_qkv_data, rax                     ; Cache pointer in SF frame slot
     mov [g_inf_qkv_data], rax                 ; Track in BSS for cleanup, destroy frees it
     mov eax, inf_n_layers                     ; eax = n_layers, 32-bit read zero-extends rax
     shl rax, 4                                ; rax = n_layers * 16 bytes, one tensor_f32 descriptor per layer
     mov rcx, rax                              ; arg1: descriptor block size
     call mdb_alloc_mem                        ; rax = descriptor block, FATALs internally on OOM
     mov inf_qkv_desc, rax                     ; Cache pointer in SF frame slot
     mov [g_inf_qkv_desc], rax                 ; Track in BSS for cleanup
; --- reload sizes from stashes, alloc calls overwritten the volatile regs ---
     mov rdx, rbx                              ; rdx = pe_q_wq
     mov r8, r13                               ; r8 = pe_q_wk
     mov r10, r14                              ; r10 = s_wq
     mov r11, r15                              ; r11 = s_wk
; --- hoist loop invariants: qword counts + per-layer s_ptr delta ---
     lea r15, [rdx + r8*FUSED_KV_COUNT]        ; r15 = pe_q_wq + 2*pe_q_wk, total q bytes per layer = s_ptr delta; BEFORE the shr's convert rdx/r8 to qwords
     shr rdx, 3                                ; rdx = pe_q_wq qwords
     shr r8, 3                                 ; r8 = pe_q_wk qwords
     shr r10, 3                                ; r10 = s_wq qwords
     shr r11, 3                                ; r11 = s_wk qwords
     mov rbx, [rdi + W_WQ]                     ; rbx = wq descriptor base, layer 0
     mov r13, [rdi + W_WK]                     ; r13 = wk descriptor base, layer 0
     mov r14, [rdi + W_WV]                     ; r14 = wv descriptor base, layer 0
     mov rdi, inf_qkv_data                     ; rdi = fused data write cursor, rep movsq advances it
     mov r12, inf_qkv_desc                     ; r12 = fused desc write ptr, advances
     mov r9d, inf_n_layers                     ; r9d = layer counter
.qkv_loop:
     mov qword [r12 + QT_Q], rdi               ; desc[l].q = fused layer data start, write cursor
     lea rax, [rdi + r15]                      ; rax = start + pe_q_wq + 2*pe_q_wk, delta hoisted into r15
     mov qword [r12 + QT_S], rax               ; desc[l].s = scale block start
     add r12, QT_SIZE                          ; desc_ptr += 16
     FUSED_QKV_COPY rbx, QT_Q, rdx             ; copy wq q block
     FUSED_QKV_COPY r13, QT_Q, r8              ; copy wk q block
     FUSED_QKV_COPY r14, QT_Q, r8              ; copy wv q block
     FUSED_QKV_COPY rbx, QT_S, r10             ; copy wq scale block
     FUSED_QKV_COPY r13, QT_S, r11             ; copy wk scale block
     FUSED_QKV_COPY r14, QT_S, r11             ; copy wv scale block
     add rbx, QT_SIZE                          ; wq desc advances to next layer
     add r13, QT_SIZE                          ; wk desc advances to next layer
     add r14, QT_SIZE                          ; wv desc advances to next layer
     dec r9d                                   ; --layer counter
     jnz .qkv_loop                             ; more layers to fuse
%undef BIAS_SCAN_ITER_BYTES
%undef BIAS_SCAN_GRP_BYTES
%undef BIAS_SCAN_GRP2_OFF
%undef BIAS_SCAN_GRP3_OFF
%undef BIAS_SCAN_QWORD_BYTES
%undef FUSED_KV_COUNT
%undef ONE_INT
%undef IDX_0
