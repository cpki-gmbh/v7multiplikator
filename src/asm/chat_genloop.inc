; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; chat_genloop.inc, generation loop: first sample, hook dispatch, decode+append, forward, resample .
; ANCHOR RULE: local labels only, the non-local label anchor stays v7s_chat, SF_START in chat.inc .
; Entered by fall-through from chat_prefill.inc, .chat_sample_first and by jumps from chat_rewrite.inc; ends with the loop back-edge jmp .chat_gen_loop.

; Phase 2 timestamp: prompt processed, prefill or single-token forward done
     RDTSC_TO_MEM rdtsc_after_prefill          ; statistics: TSC before first sample
; --- Apply presence_penalty to logits, skip if penalty == 0 or -0.0f ---
     APPLY_PRESENCE_PENALTY .chat_skip_pp_first ; apply penalty to first-token logits
.chat_skip_pp_first:
; next = v7s_sample,logits, vocab_size, temperature, topp, &rng_state, probindex
     CHAT_SAMPLE_TO_NEXT                       ; sample with current logits, temperature, top-p and RNG
     mov ecx, c_pos                            ; ecx = current position, base for this turn
     call v7s_stream_init_turn                 ; zero stream state, record base_pos
; Token stream diagnostic dump, guard first: dump_token_stream no-ops when disabled
     cmp dword [rel enable_token_dump], 0      ; token dump enabled?
     je .chat_first_dump_skip                  ; no, skip the 5-instruction call setup
     mov rax, c_ctx                            ; Load InferenceContext*, stack local
     mov rcx, [rax + CTX_TOKENIZER]            ; arg1: Tokenizer* from ctx
     mov edx, c_next_token                     ; arg2: token_id
     mov r8d, c_pos                            ; arg3: position
     call dump_token_stream                    ; dump token to stdout
.chat_first_dump_skip:
     xor r15d, r15d                            ; r15d = gen_count = 0, active counter register, incremented in the gen loop and the rewrite re-prefill loop, set to T_diverge on rewrite rollback, compared against max_gen in both loops
; Cache vocab pointer and seq_len in non-volatile registers, avoid repeated memory loads in hot loop
     mov rax, c_ctx                            ; rax = InferenceContext*
     mov rbx, [rax + CTX_TOKENIZER]            ; rbx = Tokenizer*
     mov rbx, [rbx + TOK_VOCAB]                ; rbx = tok->vocab, char**, cached for gen loop
.chat_gen_loop:
; --- Restore max_gen, decode loop overwrites r14d ---
     mov r14d, c_max_gen                       ; r14d = max_gen, reload, decode overwrites r14d
     cmp dword [rel sl_hook_enabled], 0        ; hook enabled?
     je .chat_hook_accept                      ; no: skip hook, zero-overhead gate
     mov r10d, c_pos                           ; r10d = absolute position, zero-extends r10
     mov edx, c_next_token                     ; edx = trial token ID, doubles as arg2 below, needed on both paths
     cmp r10d, c_seq_len                       ; pos >= seq_len, KV cache full ?
     jae .chat_skip_otk_fill                   ; yes: skip the fill, slot is out of bounds
     mov rax, c_out_tokens                     ; rax = out_tokens base, V7-GET-TOKEN history fill
     mov [rax + r10*DWORD_SCALE], edx          ; out_tokens[pos] = trial token id
.chat_skip_otk_fill:
; Call v7s_sectorlogo_on_token,ctx, next_token, pos, &next_token, &temp, &topp
     mov rcx, c_ctx                            ; arg1: InferenceContext*
; edx already = arg2: trial token ID, loaded above for the out_tokens fill
     mov r8d, c_pos                            ; arg3: current sequence position
     lea r9, c_next_token_raw                  ; arg4: &c_next_token, in-place override
     sub rsp, CHAT_TEMP_FRAME_2                ; shadow, 32 + 2 stack args, 16
     lea rax, c_temperature_bits_raw           ; &temperature float bits
     mov [rsp + CHAT_TEMP_ARG5], rax           ; arg5: &temperature
     lea rax, c_topp_bits_raw                  ; &topp float bits
     mov [rsp + CHAT_TEMP_ARG6], rax           ; arg6: &topp
     call v7s_sectorlogo_on_token              ; returns action code in EAX
     add rsp, CHAT_TEMP_FRAME_2                ; restore stack
     test eax, eax                             ; SL_ACTION_ACCEPT, 0 ?
     je .chat_hook_accept                      ; yes: accept, the shared history fill below records the token
     cmp eax, 1                                ; SL_ACTION_RESAMPLE, 1 ?
     je .chat_hook_resample                    ; yes: reject and resample
     cmp eax, 2                                ; SL_ACTION_EOS, 2 ?
     je .chat_finish                           ; yes: force stop generation
     cmp eax, 3                                ; SL_ACTION_REWRITE, 3 ?
     je .chat_hook_rewrite                     ; yes: automated diff/rollback/prefill
; the hook is the engine's own v7s_sectorlogo_on_token, its return is
; always one of the four actions above, no foreign producer exists
.chat_hook_accept:
     mov edx, c_next_token                     ; the accepted token, the hook may have overridden the trial in place
     mov r10d, c_pos                           ; r10d = absolute position, zero-extends r10
     cmp r10d, c_seq_len                       ; pos >= seq_len, KV cache full ?
     jae .chat_hist_done                       ; slot out of bounds: skip the history fill
     mov rax, c_out_tokens                     ; rax = out_tokens base, V7-GET-TOKEN history
     mov [rax + r10*DWORD_SCALE], edx          ; out_tokens[pos] = accepted token id, recorded on every path, hook on or off
.chat_hist_done:
     mov eax, c_eos_token_id                   ; Load EOS token ID, cached; .data copy published at init
     cmp c_next_token, eax                     ; next == eos_token_id?
     je .chat_finish                           ; Yes, stop generating
     cmp c_raw_mode, 0                         ; raw generation mode, v7_generate: no template, EOS-only stops ?
     jne .chat_no_extra_stops                  ; yes, EOS-only stop set, skip the extra scan
     mov edx, c_stop_count                     ; stop count, single cached read
     cmp edx, 1                                ; Only eos_token_id?
     jbe .chat_no_extra_stops                  ; Yes, skip extra checks
     lea r9, [rel g_stop_ids]                  ; r9 = base of stop IDs array
     mov ecx, 1                                ; Start index at 1, skip eos_token_id, slot 0
     mov r8d, c_next_token                     ; r8d = cached next_token for repeated compares
.chat_check_extra_stops:
     cmp r8d, dword [r9 + rcx*DWORD_SCALE]     ; next_token matches extra stop ID?, compared in memory
     je .chat_finish                           ; Yes, stop generating
     inc ecx                                   ; Advance to next slot
     cmp ecx, edx                              ; Reached end of array?, edx = hoisted count
     jb .chat_check_extra_stops                ; No, keep checking
.chat_no_extra_stops:
; --- Check max_gen limit, generated token cap ---
     test r14d, r14d                           ; max_gen == 0, unlimited ?
     jz .chat_skip_max_gen                     ; Skip check if unlimited
     cmp r15d, r14d                            ; gen_count >= max_gen?, register vs register
     jae .chat_finish                          ; Yes, stop generating, limit reached
.chat_skip_max_gen:
; --- Decode token: piece = vocab[next] -> copy to output ---
     mov ecx, c_next_token                     ; Load token ID for vocab lookup, rcx = index
     mov r12, [rbx + rcx*VOCAB_PTR_SCALE]      ; r12 = vocab[next], char* piece, cached vocab ptr
; Get piece length via inline strlen, fast, avoids API_CALL kernel transition
     mov rdi, r12                              ; rdi = piece pointer
     xor eax, eax                              ; al = 0, search for null terminator
     mov ecx, -1                               ; rcx = max scan count, 0xFFFFFFFF
     repne scasb                               ; scan for null byte
     not ecx                                   ; ecx = length including null terminator
     dec ecx                                   ; ecx = string length, excluding null
     mov r13d, ecx                             ; r13d = slen
; Check output buffer capacity: out_pos + slen >= output_len?
     mov eax, c_out_pos                        ; Load current output write position
     add eax, r13d                             ; eax = out_pos + raw slen, decoded length is always <= raw
     cmp eax, c_output_len                     ; Raw length fits?
     jb .chat_decode_fits                      ; yes -> the decoded piece cannot overflow either
     mov rsi, r12                              ; rsi = piece source, scan input
     mov r14d, r13d                            ; r14d = slen, r14d's max_gen duty ended at the cap check
     CHAT_DECODE_LEN                           ; r9d = decoded byte count, write-free scan
     mov eax, c_out_pos                        ; reload out_pos, scan overwritten eax
     add eax, r9d                              ; eax = out_pos + decoded length
     cmp eax, c_output_len                     ; Decoded length fits?
     jae .chat_finish                          ; no even decoded: truncate for real
.chat_decode_fits:
; Copy piece bytes to output with decode conversions, via macro
     mov eax, c_out_pos                        ; eax = out_pos for dest offset
     mov rdi, c_output                         ; rdi = output buffer base
     add rdi, rax                              ; rdi = output + out_pos, destination
     mov rsi, r12                              ; rsi = piece source pointer
     mov r14d, r13d                            ; r14d = slen, source length
     CHAT_DECODE_PIECE                         ; shared decoder returns the byte count in r13d
; out_pos += decoded_length
     add c_out_pos, r13d                       ; out_pos += decoded length
     mov r8d, r13d                             ; r8d = decoded piece length
     mov eax, c_out_pos                        ; eax = current output position
     sub eax, r8d                              ; eax = start of decoded text
     mov rdx, c_output                         ; rdx = output buffer base
     add rdx, rax                              ; rdx = &decoded_text in output buffer
     mov ecx, c_next_token                     ; ecx = token ID
     call v7s_stream_append_token              ; append to tri-array stream
     cmp dword c_presence_penalty_bits, 0      ; penalty == +0.0f, disabled?, bits are read only for the zero test, per-token hot path
     jz .chat_mark_emitted_done                ; skip: bitmap never read when penalty off
     mov rax, c_bitmap_base                    ; rax = bitmap base
     mov ecx, c_next_token                     ; ecx = token ID
     bts dword [rax], ecx                      ; hardware bit-addressing: dword [rax + 4*,tok/32 ], bit tok%32
.chat_mark_emitted_done:
     cmp dword [rel g_stream_remove_count], remove_BATCH ; pending evictions at the supply/consume threshold?
     jb .chat_remove_ok                        ; below -> no flush needed
     call v7s_stream_remove_flush              ; supply/consume ~every remove_BATCH evictions, callback latency bound
.chat_remove_ok:
     mov esi, c_seq_len                        ; Reload seq_len, rsi used as piece source in decode
     cmp dword [rel g_stream_enable], 0        ; streaming enabled?
     jne .chat_do_forward                      ; streaming: the ring cannot fill, max_gen/EOS stop instead
     cmp c_pos, esi                            ; pos >= seq_len?
     jae .chat_finish                          ; KV-cache full, stop, output already has this token
.chat_do_forward:
; --- Forward pass: logits = forward_*,transformer, next, pos ---
     mov eax, c_quantize                       ; cached ctx->quantize flag
     mov edx, c_next_token                     ; arg2: token ID, identical in both branches, hoisted above the test
     mov r8d, c_pos                            ; arg3: sequence position, identical in both branches, hoisted above the test
     test eax, eax                             ; F32 mode?
     jz .chat_gen_f32                          ; Jump to F32 if quantize==0
; Q8 generation
     mov rcx, c_transformer                    ; arg1: Transformer* from ctx
     sub rsp, 32                               ; Shadow space for Win64 ABI
     call forward_q8                           ; Returns logits in RAX
     add rsp, 32                               ; Restore RSP after shadow space
     jmp .chat_gen_done                        ; Skip F32 path
.chat_gen_f32:
     mov rcx, c_transformer                    ; arg1: Transformer* from ctx
     sub rsp, 32                               ; Shadow space for Win64 ABI
     call forward_f32                          ; Returns logits in RAX
     add rsp, 32                               ; Restore RSP after shadow space
.chat_gen_done:
     mov c_logits, rax                         ; Cache logits pointer
; pos++; gen_count++
     inc c_pos                                 ; pos++, consistent with all other c_pos accesses
     inc r15d                                  ; gen_count++ in register, track generated tokens for max_gen limit
     inc dword [rel g_live_gen_count]          ; true generated-token count for v7_get_gen_count: the main loop counts per generated token; the rewrite re-prefill loop counts its replacement tokens separately so the per-turn counter is monotonic, see that loop's inc
; --- Apply presence_penalty to logits ---
     APPLY_PRESENCE_PENALTY .chat_skip_pp_loop ; apply penalty to loop-token logits
.chat_skip_pp_loop:
; --- Sample next token ---
     CHAT_SAMPLE_TO_NEXT                       ; sample with current logits, temperature, top-p and RNG
; Token stream diagnostic dump, guard first: dump_token_stream no-ops when disabled
     cmp dword [rel enable_token_dump], 0      ; token dump enabled?
     je .chat_loop_dump_skip                   ; no, skip the 5-instruction call setup
     mov rax, c_ctx                            ; Load InferenceContext*, stack local
     mov rcx, [rax + CTX_TOKENIZER]            ; arg1: Tokenizer* from ctx
     mov edx, c_next_token                     ; arg2: token_id
     mov r8d, c_pos                            ; arg3: position
     call dump_token_stream                    ; dump token to stdout
.chat_loop_dump_skip:
     jmp .chat_gen_loop                        ; Continue generating
.chat_hook_resample:
; the sampler is NON-DESTRUCTIVE, scratch-table design : c_logits still holds
; the model's raw logits, so the re-sample reads the same distribution the
; first pass sampled from, the hook's fresh temperature/top-p apply directly
     CHAT_SAMPLE_TO_NEXT                       ; resample, hook may have changed temperature/top-p
     jmp .chat_gen_loop                        ; re-enter loop, hook sees new token
