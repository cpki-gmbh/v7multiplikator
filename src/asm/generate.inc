; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; Canonical inference-structure offsets
; ============================================================================
%include "inf_structs.inc"
     section .text
     global v7s_generate                       ; Export: raw generate, no chat template, EOS-only stop
v7s_generate:
     SF_START v7s_generate, r12                ; save r12, the only non-volatile written in the body
     SF_PTR g_ctx                              ; InferenceContext* ctx, rcx
     SF_PTR g_prompt                           ; char* prompt, rdx
     SF_PTR g_output                           ; char* output, r8
     SF_DWORD g_output_len                     ; int output_len, r9d
     SF_PTR g_saved_key_cache                  ; original RunState->key_cache
     SF_PTR g_saved_value_cache                ; original RunState->value_cache
     SF_QWORD g_kv_single                      ; byte size of one KV cache
     SF_PTR g_rs                               ; RunState*, cached
     SF_DWORD g_saved_pos                      ; original ctx->pos, saved before swap
     SF_QWORD g_saved_rng_state                ; saved sampler->rng_state, PRNG state
     SF_DWORD g_saved_mrope_run                ; chat session's M-RoPE running counter, Qwen3-VL
     SF_DWORD g_saved_mrope_delta              ; chat session's M-RoPE generation rope delta
     SF_PTR g_saved_out_tokens                 ; original ctx->out_tokens, the chat session's token history
     SF_COMMIT                                 ; finalize stack frame
; Save register arguments
     TRACE_STR "[GEN] v7s_generate entry"      ; trace: function entry
     TRACE_HEX rcx                             ; trace: ctx ptr arg
     TRACE_HEX rdx                             ; trace: prompt ptr arg
     TRACE_HEX r8                              ; trace: output ptr arg
     TRACE_HEX r9                              ; trace: output_len arg
     mov g_ctx, rcx                            ; save InferenceContext*
     mov g_prompt, rdx                         ; save user prompt ptr
     mov g_output, r8                          ; save output buffer ptr
     mov g_output_len, r9d                     ; save output buffer len
; Validate output buffer
     cmp g_output, 0                           ; check output != NULL
     je .g_fatal_null_output                   ; FATAL if NULL
     cmp g_output_len, 1                       ; check output_len >= 1
     jl .g_fatal_output_len                    ; FATAL if too small
     mov r12, g_ctx                            ; r12 = ctx
     mov r12, [r12 + CTX_TRANSFORMER]          ; r12 = Transformer*, offset 0 in the context header, this DEREFERENCE is required: the frame slot holds ctx, not the Transformer
     TRACE_STR "[GEN] resolved Transformer*"   ; trace: got Transformer*
     TRACE_HEX r12                             ; trace: Transformer* value
; Config is at Transformer+0
     mov eax, [r12 + CFG_N_LAYERS]             ; eax = n_layers, first factor of the KV size product
     mov ecx, [r12 + CFG_SEQ_LEN]              ; rcx = seq_len, zero-extended
     imul rax, rcx                             ; rax = n_layers * seq_len, 64-bit product
     mov ecx, [r12 + CFG_N_KV_HEADS]           ; rcx = n_kv_heads, zero-extended
     imul rax, rcx                             ; rax = n_layers * seq_len * n_kv_heads
     mov ecx, [r12 + CFG_HEAD_DIM]             ; rcx = head_dim, zero-extended
     imul rax, rcx                             ; rax = n_layers * seq_len * kv_dim, 64-bit
     shl rax, 1                                ; packed FP16 cache, two bytes per element
     mov g_kv_single, rax                      ; cache kv_single
     TRACE_STR "[GEN] kv_single computed"      ; trace: kv_single computed
     TRACE_HEX rax                             ; trace: kv_single byte count, rax = kv_single, cached to the g_kv_single frame slot immediately above
     mov ecx, [r12 + CFG_SEQ_LEN]              ; one dword of history per context position
     lea rcx, [rax + rcx*DWORD_SCALE]          ; add the token-history bytes to one KV plane
     add rcx, rax                              ; include the other KV plane
     TRACE_STR "[GEN] allocating private KV and token history" ; trace the single private allocation
     call mdb_alloc_mem                        ; rax = merged KV block base = key region, fbm fallback on OOM; never NULL
     mov [rel g_generate_kv_owner], rax        ; publish BSS resource holder = merged KV base = key region, destroy path releases on a caught FATAL
     TRACE_STR "[GEN] merged KV block allocated OK" ; trace: allocation succeeded
     TRACE_HEX rax                             ; trace: block base, rax = mdb_alloc_mem result, published just above
     lea rax, [r12 + TRANSFORMER_STATE]        ; rax = &RunState, embedded at offset 184
     mov g_rs, rax                             ; cache RunState*
     TRACE_STR "[GEN] resolved RunState*"      ; trace: got RunState*
     TRACE_HEX rax                             ; trace: RunState* value
     mov rcx, [rax + S_KEY_CACHE]              ; rcx = original key_cache
     mov g_saved_key_cache, rcx                ; save it
     TRACE_STR "[GEN] original key_cache"      ; trace: old key_cache
     TRACE_HEX rcx                             ; trace: old key_cache value
     mov rcx, [rax + S_VALUE_CACHE]            ; rcx = original value_cache
     mov g_saved_value_cache, rcx              ; save it
     TRACE_STR "[GEN] original value_cache"    ; trace: old value_cache
     TRACE_HEX rcx                             ; trace: old value_cache value
     mov rcx, [rel g_generate_kv_owner]        ; rcx = merged mapping base = generate key region
     mov [rax + S_KEY_CACHE], rcx              ; RunState->key_cache = gen_key_cache
     TRACE_STR "[GEN] swapped key_cache -> gen" ; trace: key_cache swapped
     TRACE_HEX rcx                             ; trace: new key_cache value
     add rcx, g_kv_single                      ; rcx = mapping base + kv_single = generate value region, the value region sits kv_single bytes past the key region
     mov [rax + S_VALUE_CACHE], rcx            ; RunState->value_cache = gen_value_cache
     TRACE_STR "[GEN] swapped value_cache -> gen" ; trace: value_cache swapped
     TRACE_HEX rcx                             ; trace: new value_cache value
     mov rax, g_ctx                            ; rax = ctx
     TRACE_STR "[GEN] ctx ptr for pos save"    ; trace: about to read ctx->pos
     TRACE_HEX rax                             ; trace: ctx pointer
     mov ecx, [rax + CTX_POS]                  ; ecx = original ctx->pos, rax preserved
     TRACE_STR "[GEN] original ctx->pos"       ; trace: old pos value
     TRACE_HEX rcx                             ; trace: pos value, the ecx store clears rcx's upper 32, so all 64 bits are exact
     mov g_saved_pos, ecx                      ; save original position
     mov dword [rax + CTX_POS], 0              ; ctx->pos = 0 for generate
     TRACE_STR "[GEN] ctx->pos zeroed"         ; trace: pos set to 0
     mov rax, [rax + CTX_SAMPLER]              ; rax = Sampler*, rax still holds ctx from the pos load above, nothing overwrote it in between
     TRACE_STR "[GEN] resolved Sampler*"       ; trace: got Sampler*
     TRACE_HEX rax                             ; trace: Sampler* value
     mov rax, [rax + SAM_RNG_STATE]            ; rax = rng_state value, u64
     TRACE_STR "[GEN] original rng_state"      ; trace: old rng_state
     TRACE_HEX rax                             ; trace: rng_state value
     mov g_saved_rng_state, rax                ; save original PRNG state
     mov eax, [rel g_mrope_run]                ; chat session's running counter
     mov g_saved_mrope_run, eax                ; save before the inner chat resets it
     mov eax, [rel g_mrope_delta]              ; chat session's generation delta
     mov g_saved_mrope_delta, eax              ; save before the inner chat overwrites it
     mov rax, g_kv_single                      ; bytes per KV plane
     add rax, rax                              ; history follows both KV planes
     add rax, [rel g_generate_kv_owner]        ; form the interior history pointer
     mov rcx, g_ctx                            ; reload the inference context
     mov rdx, [rcx + CTX_OUT_TOKENS]           ; save the outer history before the swap
     mov g_saved_out_tokens, rdx               ; retain the outer history pointer
     mov [rcx + CTX_OUT_TOKENS], rax           ; nested chat uses the zero-filled tail
     TRACE_STR "[GEN] === ABOUT TO CALL v7s_chat ===" ; trace: pre-chat barrier
     TRACE_STR "[GEN] arg3 = NULL (0)"         ; trace: arg3 = NULL
     TRACE_STR "[GEN] arg6 = 1 (reset)"        ; trace: arg6 = reset=1
     mov r11d, g_output_len                    ; arg5: output buffer length, API_CALL overwrites volatile regs
     mov eax, [rel g_stream_enable]            ; swap streaming OFF for the nested chat, every legacy
     mov [rel g_stream_enable_saved], eax      ; backstop returns, prompt seq_len FATAL, decode capacity stop
     mov dword [rel g_stream_enable], 0        ; and the nested chat cannot ADD evictions to the session ring/log
     mov eax, [rel g_stream_otk_live]          ; the outer session's token-history validity, a KV-loaded session has none
     mov [rel g_stream_otk_live_saved], eax    ; save it, the nested chat's reset sets the flag unconditionally
; bookkeeping off.The entry resets are also skipped while enable==0, so a
; pending outer batch and the warm latch SURVIVE the nested call: the nested
; turn-end flush delivers the pending batch, correct tags, exactly-once
     API_CALL v7s_chat, g_ctx, g_prompt, 0, g_output, r11, 1 ; reset=1 for fresh generate KV
     TRACE_STR "[GEN] === v7s_chat returned ===" ; trace: post-chat barrier
     TRACE_HEX rax                             ; trace: v7s_chat return value
     mov eax, [rel g_stream_enable_saved]      ; restore the outer session's mode; a caught FATAL that skips
     mov [rel g_stream_enable], eax            ; this store leaves streaming OFF, safe degradation, never OOB
     TRACE_STR "[GEN] === RESTORING STATE ===" ; trace: restore phase
     mov rax, g_rs                             ; rax = RunState*
     TRACE_STR "[GEN] restoring RunState*"     ; trace: restoring
     TRACE_HEX rax                             ; trace: RunState* value
     mov rcx, g_saved_key_cache                ; rcx = original key_cache
     mov [rax + S_KEY_CACHE], rcx              ; restore RunState->key_cache
     TRACE_STR "[GEN] restored key_cache"      ; trace: key restored
     TRACE_HEX rcx                             ; trace: restored key_cache value
     mov rcx, g_saved_value_cache              ; rcx = original value_cache
     mov [rax + S_VALUE_CACHE], rcx            ; restore RunState->value_cache
     TRACE_STR "[GEN] restored value_cache"    ; trace: value restored
     TRACE_HEX rcx                             ; trace: restored value_cache value
; Restore ctx->pos so the chat session continues from where it left off
     mov rax, g_ctx                            ; rax = ctx
     mov ecx, g_saved_pos                      ; ecx = original position
     mov [rax + CTX_POS], ecx                  ; ctx->pos = original pos
     TRACE_STR "[GEN] restored ctx->pos"       ; trace: pos restored
     TRACE_HEX rcx                             ; trace: restored pos value
; Restore sampler->rng_state so chat's PRNG continues from where it left off
     mov rax, [rax + CTX_SAMPLER]              ; rax = Sampler*
     mov rcx, g_saved_rng_state                ; rcx = saved rng_state value
     mov [rax + SAM_RNG_STATE], rcx            ; restore rng_state
     TRACE_STR "[GEN] restored rng_state"      ; trace: rng_state restored
     TRACE_HEX rcx                             ; trace: restored rng_state value
     mov eax, g_saved_mrope_run                ; chat session's running counter
     mov [rel g_mrope_run], eax                ; undo the generate session's reset/advance
     mov eax, g_saved_mrope_delta              ; chat session's generation delta
     mov [rel g_mrope_delta], eax              ; undo the generate session's overwrite
     mov eax, [rel g_stream_otk_live_saved]    ; the outer session's token-history validity
     mov [rel g_stream_otk_live], eax          ; undo the nested chat's unconditional history-live set
     mov rcx, g_ctx                            ; rcx = ctx
     mov rdx, g_saved_out_tokens               ; rdx = original out_tokens
     mov [rcx + CTX_OUT_TOKENS], rdx           ; ctx->out_tokens = chat history again
     TRACE_STR "[GEN] === FREEING CACHES ==="  ; trace: free phase
     mov rcx, [rel g_generate_kv_owner]        ; rcx = merged KV base, one block holds key + value regions
     TRACE_HEX rcx                             ; trace: block to free
     call mdb_free_mem                         ; release the merged KV block
     mov qword [rel g_generate_kv_owner], 0    ; clear the BSS resource holder, normal path, destroy path sees nothing to free
     TRACE_STR "[GEN] merged KV block freed, all done" ; trace: all done
; Dump context state at exit, callee gates on enable_dump internally, no output when dumps are off
     mov rcx, g_ctx                            ; rcx = InferenceContext*
     call dump_inference_context               ; dump full context to stdout
     SF_END r12                                ; restore r12 and return
; ============================================================================
; FATAL error handlers
; ============================================================================
.g_fatal_null_output:
     FATAL "v7s_generate: output is NULL"      ; contract violation
.g_fatal_output_len:
     FATAL "v7s_generate: output_len < 1"      ; contract violation
%undef g_ctx
%undef g_ctx_raw
%undef g_prompt
%undef g_prompt_raw
%undef g_output
%undef g_output_raw
%undef g_output_len
%undef g_output_len_raw
%undef g_saved_key_cache
%undef g_saved_key_cache_raw
%undef g_saved_value_cache
%undef g_saved_value_cache_raw
%undef g_kv_single
%undef g_kv_single_raw
%undef g_rs
%undef g_rs_raw
%undef g_saved_pos
%undef g_saved_pos_raw
%undef g_saved_rng_state
%undef g_saved_rng_state_raw
%undef g_saved_mrope_run
%undef g_saved_mrope_run_raw
%undef g_saved_mrope_delta
%undef g_saved_mrope_delta_raw
%undef g_saved_out_tokens
%undef g_saved_out_tokens_raw

