; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; chat_session.inc, v7s_chat session setup .
; ANCHOR RULE: local labels only, the non-local label anchor stays v7s_chat, SF_START in chat.inc .
; Fall-through entry from chat.inc's frame commit; falls through into chat_prompt.inc.

     mov c_ctx, rcx                            ; InferenceContext* ctx
     mov c_user_prompt, rdx                    ; User prompt string
     mov c_system_prompt, r8                   ; System prompt string
     mov c_output, r9                          ; Output buffer
     mov eax, [rbp + CHAT_SARG_OUTPUT_LEN]     ; Load output_len from stack, 5th arg
     mov c_output_len, eax                     ; Cache output capacity
     mov eax, [rbp + CHAT_SARG_RESET]          ; Load reset flag from stack, 6th arg
     mov c_reset, eax                          ; Cache reset flag, 0=continue, !=0=fresh session
; ====== Per-chat sidecar activation: THE DATABASE KEY IS THE SWITCH ======
; The host manages the "logo:sidecar" DB key via v7_db_add -- add a new
; script or overwrite the old one to SWITCH sidecars. This per-chat check
; picks the change up: key absent = hook off, script changed = re-parse.
     call v7s_sidecar_refresh                  ; re-read the "logo:sidecar" DB key: absent = hook off, changed = re-parse
; ====== End per-chat sidecar activation ======
; Phase 0 timestamp: chat entry, rax/rdx dead here, args all cached
     RDTSC_TO_MEM rdtsc_chat_start             ; statistics: TSC at v7s_chat entry
; Early validation: output must be non-NULL and output_len > 0
     cmp c_output, 0                           ; output == NULL?
     je .chat_null_output_fatal                ; FATAL: NULL output buffer
     cmp c_output_len, 1                       ; output_len < 1?, capacity check
     jl .chat_output_len_fatal                 ; FATAL: invalid output length
     cmp c_output_len, SL_STREAM_BUF_SIZE - 1  ; output_len within the 1 MiB stream snapshot capacity, room for the in-place NUL at str_len ?
     jbe .chat_len_clamped                     ; yes: use verbatim
     mov c_output_len, SL_STREAM_BUF_SIZE - 1  ; chat-border correction: silently clamp an oversized request; generation truncates at the clamped capacity, buffer-end truncation is intended behaviour
.chat_len_clamped:
     cmp c_reset, 0                            ; reset != 0, caller wants fresh session ?
     je .chat_no_reset                         ; No, keep ctx->pos as-is for multi-turn
     mov rax, c_ctx                            ; Load InferenceContext*, must be after cmp, before use
     mov dword [rax + CTX_POS], 0              ; Write ctx->pos = 0, fresh session: KV cache
; slots get overwritten as pos walks from 0 up,
; no need to zero the cache memory itself
     cmp dword [rel g_stream_enable], 0        ; streaming session?, a nested v7s_generate runs with
     je .chat_reset_nostream                   ; enable swapped OFF and must NOT touch the outer
; session's eviction state, and a post-KV-load warm
; latch would otherwise be lost before it burns down
     xor r10d, r10d                            ; zero source for the reset stores
     mov dword [rel g_stream_remove_warm], r10d ; fresh streaming session: no post-load warm-up may carry over
     mov eax, [rel g_stream_window]            ; W of the live config: the reset repopulates the cache linearly
     mov [rel g_stream_win_valid], eax         ; from 0, so the first wrapped write sees a FULL window, a tier-2 shaved value below W must not survive a fresh session
     mov dword [rel g_stream_otk_live], r10d   ; fresh session: no recorded history until this session's prompt fill republishes it
     mov dword [rel g_stream_remove_count], r10d ; pending evictions belong to the PREVIOUS session, drop them
.chat_reset_nostream:
     mov c_pos, 0                              ; c_pos := 0, pure write avoids RMW stall
     jmp .chat_pos_done                        ; Skip multi-turn pos read
.chat_no_reset:
     mov rax, c_ctx                            ; Load InferenceContext* for pos read
     mov eax, [rax + CTX_POS]                  ; Read ctx->pos, current KV-cache position
     mov c_pos, eax                            ; Use saved pos from previous turns
.chat_pos_done:
     mov c_out_pos, 0                          ; byte cursor restarts at buffer origin for every turn, the output buffer is per-turn
     mov c_rw_tier2, 0                         ; no rewrite in flight: the sample/finish unwinds test this flag on every chat exit
     mov c_rw_muted, 0                         ; the warm-latch mute is ALSO read unconditionally at finish, a stale stack slot here would restore g_stream_remove_warm from garbage c_rw_latch on a no-rewrite turn, c_rw_latch itself stays uninitialized: it is only read behind the mute gate a genuine rewrite parked
     cmp c_pos, 0                              ; does this turn's timeline start at absolute 0?, a reset session, or a no-reset call on a fresh ctx
     jne .chat_otk_live_skip                   ; no: continuation, either the prefix [0, pos was live-built by this process's earlier turns, flag stays 1 or it is a KV load's foreign rows, flag stays 0: the load cleared it and this turn's fill only writes [pos, pos+P
     mov dword [rel g_stream_otk_live], 1      ; the fill below writes out_tokens[0..P contiguously from 0, the id history the tier-2 fill-in reads is real
.chat_otk_live_skip:
     mov rax, c_ctx                            ; Load InferenceContext*
     mov rax, [rax + CTX_SAMPLER]              ; Load Sampler* from ctx->sampler
     mov ecx, [rax + SAM_VOCAB_SIZE]           ; Load vocabulary size from Sampler
     mov c_vocab_size, ecx                     ; Cache vocab_size for v7s_sample, import guarantees vocab fits the bitmap
     mov ecx, [rax + SAM_TEMPERATURE]          ; Load temperature float bits, GPR move, no xmm round-trip
     mov c_temperature_bits, ecx               ; Cache temperature bits for v7s_sample
     mov ecx, [rax + SAM_TOPP]                 ; Load top-p threshold float bits
     mov c_topp_bits, ecx                      ; Cache topp bits for v7s_sample
     mov ecx, [rax + SAM_PRESENCE_PENALTY]     ; Load presence_penalty float bits
     mov c_presence_penalty_bits, ecx          ; Cache for penalty application
     lea rcx, [rax + SAM_RNG_STATE]            ; Compute &sampler->rng_state
     mov c_rng_state_ptr, rcx                  ; Cache rng_state address
     mov rax, [rax + SAM_PROBINDEX]            ; Load probindex workspace pointer
     mov c_probindex_ptr, rax                  ; Cache probindex for v7s_sample
     mov rax, c_ctx                            ; Load InferenceContext*
     mov rax, [rax + CTX_TRANSFORMER]          ; Load Transformer* from ctx->transformer
     mov ecx, [rax + CFG_SEQ_LEN]              ; Load seq_len from embedded config
     mov c_seq_len, ecx                        ; Cache seq_len for KV-cache bounds check
; Read max_gen from ctx, max generated tokens, set by caller
     mov rax, c_ctx                            ; Load InferenceContext*
     mov eax, [rax + CTX_MAX_GEN]              ; Load ctx->max_gen token limit
     mov c_max_gen, eax                        ; Cache max_gen for generation loop bound
; Cache tokenizer fields via ctx->tokenizer
     mov rax, c_ctx                            ; Load InferenceContext*
     mov rax, [rax + CTX_TOKENIZER]            ; Load Tokenizer* from ctx->tokenizer
     mov ecx, [rax + TOK_EOS_ID]               ; Load EOS token ID
     mov c_eos_token_id, ecx                   ; Cache EOS for stop condition
     mov [rel g_chat_eos_token], ecx           ; publish EOS id once for V7-EOS primitive reads, loop-invariant
     mov ecx, [rel g_stop_id_count]            ; stop-ID count, chat-invariant, cache once
     mov c_stop_count, ecx                     ; cache for the per-token stop-set scan
     mov eax, [rel g_raw_generation]           ; raw-generation flag, public v7_generate arg; 0 = normal
     mov c_raw_mode, eax                       ; cache ONCE at entry, Step 1 render + stop scan branch on it
     mov rax, c_ctx                            ; Load InferenceContext*
     mov eax, [rax + CTX_QUANTIZE]             ; ctx->quantize flag, 0=F32, 1=Q8
     mov c_quantize, eax                       ; cache ONCE, five load sites read the slot below
     mov rax, c_ctx                            ; Load InferenceContext*
     mov rax, [rax + CTX_TRANSFORMER]          ; ctx->transformer, Transformer*
     mov c_transformer, rax                    ; cache ONCE, 8 forward/prefill call sites read the slot
     mov rax, c_ctx                            ; Load InferenceContext*
     mov rax, [rax + CTX_OUT_TOKENS]           ; out_tokens int*, seq_len capacity
     mov c_out_tokens, rax                     ; cache for the fill sites below
     mov dword [rel g_live_gen_count], 0       ; reset the per-turn generated-token counter, v7_get_gen_count reads it live; the main loop increments per generated token and the rewrite re-prefill loop increments per replacement token, GenCount is 0 exactly at the first hook call of this chat call and grows with every token the round processes
     mov ecx, c_pos                            ; snapshot the chat call's ENTRY position BEFORE the prompt prefill: GetTokenPosition = c_pos - base is local to this query, the first hook call returns the prompt token count, a short rewrite rolls it back, the next chat call resets it
     mov [rel g_query_base_pos], ecx           ; publish the per-query base for the GetTokenPosition primitive
     mov r14, rsp                              ; Temporary top bound for probe loop, overwritten after probe
     sub rsp, CHAT_RENDERED_BUF + CHAT_PROMPT_TOKENS_BYTES + BITMAP_BYTES ; Allocate ~180KB on stack, rendered+ptok+bitmap
; Stack probe: touch every 4KB page TOP-DOWN to trigger guard pages sequentially
     lea rcx, [r14 - CHAT_PROBE_PAGE]          ; Start at highest new page, just below old RSP
.probe_loop:
     mov dword [rcx], 0                        ; Write touch to commit guard page, guaranteed write, never no-op
     sub rcx, CHAT_PROBE_PAGE                  ; Move down to next 4KB page
     cmp rcx, rsp                              ; Reached or passed new RSP?
     jae .probe_loop                           ; If >= new RSP, keep probing downward
     mov dword [rsp], 0                        ; Ensure page containing RSP is committed, probe may miss it if RSP unaligned
     mov r14, rsp                              ; r14 = rendered buffer base, lowest address
     lea rax, [r14 + CHAT_RENDERED_BUF]        ; prompt_tokens buffer = rendered_base + 32KB
     mov c_prompt_tokens, rax                  ; Cache prompt_tokens pointer
     lea rax, [r14 + CHAT_RENDERED_BUF + CHAT_PROMPT_TOKENS_BYTES] ; bitmap = after prompt_tokens
     mov c_bitmap_base, rax                    ; Cache bitmap pointer
     cmp dword c_presence_penalty_bits, 0      ; penalty == +0.0f, disabled?, bits are read only for the zero test here
     jz .chat_bitmap_zero_done                 ; yes -> nothing can be set, nothing to clear
     mov rdi, rax                              ; destination = bitmap base
     xor eax, eax                              ; value = 0
     mov ecx, BITMAP_DWORDS                    ; 5120 dwords
     rep stosd                                 ; zero bitmap
.chat_bitmap_zero_done:
