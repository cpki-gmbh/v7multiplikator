; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; Local constants for offset arithmetic, float scale 4 bytes, array index 0
%define IDX_0 0                                ; Zero index for array access
%define FK_FLOAT_BYTES 4                       ; Bytes per float element, for byte-offset arithmetic
%define KV_PRETOUCH_TOKENS 1500                ; initial token positions backed by resident pages per KV layer
%define KV_STAGING_ROW_PAIR_BYTES 8            ; two FP32 staging rows, 2 * sizeof(float), per kv_dim element
%define KV_PRETOUCH_PAGE 4096                  ; Windows x64 memory page stride for initial KV writes
%define KV_PRETOUCH_LAST -1                    ; byte displacement of the touched prefix's final byte
; ============================================================================
; WIRE_RS ml_off_const, s_off_const

; Load a RunState field's offset from rsi, = MemLayout* and convert it to
; an absolute pointer inside the rs_block at rbx, then store it into
; [rdi + S_*], = state->field.Collapses the 14 rs_block-relative pointer
; fields into 14 invocations. Overwrites rax only. KEY_CACHE / VALUE_CACHE are
; written by hand instead, they are mdb_alloc_mem block bases, not rs_block
; offsets, and K / V are never wired at init: they are per-token scratch
; Precondition, set up by caller : rsi=&MemLayout, rbx=rs_block base, rdi=&state
; ============================================================================

%macro WIRE_RS 2
     mov rax, [rsi + %1]                       ; Load field offset from MemLayout
     add rax, rbx                              ; Convert offset to absolute RunState pointer, 3 bytes vs lea 4
     mov [rdi + %2], rax                       ; Store pointer in RunState field
%endmacro
     mov rsi, inf_ml_ptr                       ; Pin MemLayout*, Step 10 entry
     mov rbx, [rsi + ML_OFF_KEY_CACHE]         ; size of the existing working-buffer prefix
     add rbx, 63                               ; reserve padding for the FP32 staging alignment
     and rbx, -64                              ; staging starts on a 64-byte boundary
     mov eax, inf_n_kv_heads                   ; KV head count
     mov ecx, inf_head_dim                     ; elements per KV head
     imul rax, rcx                             ; FP32 elements in each staging row
     lea rcx, [rbx + rax*KV_STAGING_ROW_PAIR_BYTES] ; prefix plus two FP32 rows
     call mdb_alloc_mem                        ; one working block, fbm fallback on OOM; never NULL
     mov [g_inf_rs_block], rax                 ; publish the only owning pointer
     mov inf_rs_block, rax                     ; retain the unchanged RunState base
     add rax, rbx                              ; locate the aligned interior staging buffer
     mov [g_inf_kv_scratch], rax               ; publish its borrowed pointer for forward and prefill
; KV allocations are independent of old FP32 MicroDB metadata (working-block layout unchanged).
; A persistent cache element is exactly 2 bytes, and every dimension is init-validated.
     mov eax, inf_n_layers                     ; eax = n_layers
     mov ecx, inf_seq_len                      ; ecx = seq_len
     imul rax, rcx                             ; rax = n_layers * seq_len
     mov ecx, inf_n_kv_heads                   ; ecx = n_kv_heads
     imul rax, rcx                             ; rax = n_layers * seq_len * n_kv_heads
     mov ecx, inf_head_dim                     ; ecx = head_dim
     imul rax, rcx                             ; rax = total KV elements per cache
     shl rax, 1                                ; bytes in ONE packed FP16 K/V cache
     mov rbx, rax                              ; preserve packed cache size across allocator calls
     lea rcx, [rbx + rbx]                      ; bytes for both packed KV planes
     call mdb_alloc_mem                        ; one owning allocation, same as generate
     mov [g_inf_key_cache], rax                ; cleanup owns the combined base
     mov inf_key_cache, rax                    ; key plane starts at the owning base
     add rax, rbx                              ; value plane starts after all key layers
     mov inf_value_cache, rax                  ; cache interior pointer for RunState setup
; Make the first min(seq_len, 1500) token positions resident in each layer's
; independent FP16 K/V allocation. The separate generate KV block is untouched.
     mov eax, inf_n_kv_heads                   ; kv elements per token = n_kv_heads * head_dim
     imul eax, inf_head_dim                    ; element count fits the validated model dimensions
     shl eax, 1                                ; FP16 bytes per token
     mov ecx, inf_seq_len                      ; full sequence length in tokens
     mov edx, ecx                              ; preserve full length for the layer stride
     cmp ecx, KV_PRETOUCH_TOKENS               ; is the context shorter than the requested prefix?
     jbe .kv_touch_count                       ; yes: touch only the allocated tokens
     mov ecx, KV_PRETOUCH_TOKENS               ; otherwise touch the first 1500 tokens
.kv_touch_count:
     imul rcx, rax                             ; bytes to touch in each layer
     imul rdx, rax                             ; full layer stride in bytes
     mov r10, inf_key_cache                    ; first key-cache layer
     mov r11, inf_value_cache                  ; first value-cache layer
     mov r8d, inf_n_layers                     ; layer count
.kv_touch_layer:
     xor r9d, r9d                              ; offset of first page in this layer
.kv_touch_page:
     mov byte [r10 + r9], 0                    ; fault in key-cache page
     mov byte [r11 + r9], 0                    ; fault in value-cache page
     add r9, KV_PRETOUCH_PAGE                  ; advance by one memory page
     cmp r9, rcx                               ; have all regularly spaced pages been touched?
     jb .kv_touch_page                         ; no: touch the next page
     mov byte [r10 + rcx + KV_PRETOUCH_LAST], 0 ; include the final partial key page when unaligned
     mov byte [r11 + rcx + KV_PRETOUCH_LAST], 0 ; include the final partial value page when unaligned
     add r10, rdx                              ; next key-cache layer
     add r11, rdx                              ; next value-cache layer
     dec r8d                                   ; one fewer layer to process
     jnz .kv_touch_layer                       ; repeat for all layers
; === Step 11: Set up RunState pointers ===
; RunState is at TRANSFORMER_STATE, 184 within Transformer
     mov rdi, inf_transformer                  ; Load Transformer base
     lea rdi, [rdi + TRANSFORMER_STATE]        ; rdi = &state
     mov rbx, inf_rs_block                     ; rbx = rs_block base, rsi = MemLayout* already pinned since Step 10
; Set each RunState field from mem_layout offsets
     WIRE_RS ML_OFF_X, S_X                     ; state->x = rs_block + off_x
     WIRE_RS ML_OFF_XB, S_XB                   ; state->xb = rs_block + off_xb
     WIRE_RS ML_OFF_HB, S_HB                   ; state->hb = rs_block + off_hb
     WIRE_RS ML_OFF_HB2, S_HB2                 ; state->hb2 = rs_block + off_hb2
     WIRE_RS ML_OFF_XQ_Q, S_XQ_Q               ; state->xq_q = rs_block + off_xq_q
     WIRE_RS ML_OFF_XQ_S, S_XQ_S               ; state->xq_s = rs_block + off_xq_s
     WIRE_RS ML_OFF_HQ_Q, S_HQ_Q               ; state->hq_q = rs_block + off_hq_q
     WIRE_RS ML_OFF_HQ_S, S_HQ_S               ; state->hq_s = rs_block + off_hq_s
     WIRE_RS ML_OFF_Q, S_Q                     ; state->q = rs_block + off_q
     WIRE_RS ML_OFF_ATT, S_ATT                 ; state->att = rs_block + off_att
     WIRE_RS ML_OFF_LOGITS, S_LOGITS           ; state->logits = rs_block + off_logits
     mov rax, inf_key_cache                    ; key cache block, mdb_alloc_mem, not carved from rs_block
     mov [rdi + S_KEY_CACHE], rax              ; state->key_cache = key cache block base
     mov rax, inf_value_cache                  ; value cache block
     mov [rdi + S_VALUE_CACHE], rax            ; state->value_cache = value cache block base
     mov rax, [g_inf_kv_scratch]               ; rax = K/V staging block base
     mov [rdi + S_K], rax                      ; FP32 row, NEVER a pointer into packed cache
     mov ecx, inf_n_kv_heads                   ; ecx = n_kv_heads
     imul ecx, inf_head_dim                    ; ecx = kv_dim
     lea rax, [rax + rcx*FK_FLOAT_BYTES]       ; V row begins one kv_dim float later
     mov [rdi + S_V], rax                      ; adjacent FP32 row
     WIRE_RS ML_OFF_X_STORAGE, S_X_STORAGE     ; state->x_storage = rs_block + off_x_storage
     WIRE_RS ML_OFF_Q_STORAGE, S_Q_STORAGE     ; state->q_storage = rs_block + off_q_storage
     WIRE_RS ML_OFF_ATT_OUT, S_ATT_OUT         ; state->att_out = rs_block + off_att_out
; === Step 12: Capture the embedding conversion mode ===
; The token embedding table stays packed; each text path dequantizes its own
; requested row through v7s_embedding_row under this exact MXCSR mode.
     stmxcsr [rel g_inf_embed_mxcsr]           ; mode used by the on-demand row reader
; === Step 13: Initialize RoPE tables, ASM, via v7s_rope_init ===
     mov ecx, inf_head_dim                     ; arg1: head_dim for table builder
     call v7s_rope_init                        ; Set globals, free old tables, build new tables, tail-calls the table builder internally; returns here; every return path publishes both fused_rope_cos/sin non-NULL, bad head_dim FATALs inside, allocation cannot fail
; rope_max_dim = head_dim/2, elements per position in cos/sin tables, stride
     mov rcx, [fused_rope_cos]                 ; arg1: cos table pointer
     mov rdx, [fused_rope_sin]                 ; arg2: sin table pointer
     mov r8d, inf_head_dim                     ; arg3 source: head_dim
     shr r8d, 1                                ; rope_max_dim = head_dim / 2, elements per table position
     call set_rope_tables                      ; publish cos/sin base + stride for all forward kernels
%unmacro WIRE_RS 2                             ; release the RunState pointer-wiring macro
%undef FK_FLOAT_BYTES
%undef IDX_0
