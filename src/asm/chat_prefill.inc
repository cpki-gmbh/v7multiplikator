; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; chat_prefill.inc, ingest routes: StreamingLLM wrap, KV check, prefill/single forward, history+ring fill .
; ANCHOR RULE: local labels only, the non-local label anchor stays v7s_chat, SF_START in chat.inc .
; Falls through from chat_prompt.inc; ends by falling into .chat_sample_first, kept at this file's tail: the label sits with the jumps that target it .

     cmp dword [rel g_stream_enable], 0        ; StreamingLLM enabled?
     jne .chat_stream_route                    ; streaming: the seq_len FATAL does not apply
     mov r8d, c_seq_len                        ; Load seq_len for KV-cache bounds check
     mov r10d, c_pos                           ; Load current position
     add r10d, eax                             ; r10d = c_pos + num_prompt_tokens
     cmp r10d, r8d                             ; would exceed KV-cache capacity?
     ja .chat_kvcache_fatal                    ; FATAL: prompt exceeds KV-cache capacity
     jmp .chat_route_std                       ; legacy: continue with the standard route
.chat_stream_route:
     mov r10d, c_pos                           ; r10d = session position
     add r10d, eax                             ; r10d = turn end, the last prompt token sits at r10d - 1
     cmp r10d, [rel g_stream_cap]              ; turn end <= C, last token at <= C-1 ?
     jbe .chat_route_std                       ; -> legacy prefill, fast path, bit-identical
; fall through: wrapped turn -> per-token ingestion
.chat_stream_ingest:
     mov rsi, c_prompt_tokens                  ; rsi = int* prompt tokens, non-volatile, preserved across calls
     mov r15d, c_pos                           ; r15d = running position, dump-loop parking convention
     mov ebx, c_num_prompt_tokens              ; hoist the loop bound, rbx unused until the gen loop re-parks vocab
     xor edi, edi                              ; edi = 0, loop index
.chat_ingest_lp:
     cmp edi, ebx                              ; i < n?, register compare
     jae .chat_ingest_done                     ; loop finished
     mov edx, [rsi + rdi*DWORD_SCALE]          ; edx = prompt_tokens[i], arg2 for forward
     cmp r15d, c_seq_len                       ; out_tokens clamp: positions >= seq_len
     jae .chat_ingest_nootk                    ; have no history slot, documented GET-TOKEN limit
     mov rax, c_out_tokens                     ; rax = out_tokens array base
     mov ecx, r15d                             ; ecx = position
     mov [rax + rcx*DWORD_SCALE], edx          ; out_tokens[pos] = token id
.chat_ingest_nootk:
     mov rcx, c_transformer                    ; arg1 = Transformer*
     mov r8d, r15d                             ; arg3 = position, forward resolves slot + rope itself
     cmp dword c_quantize, 0                   ; 0 = F32, else Q8, read only for the zero test
     jne .chat_ingest_q8                       ; no -> Q8 forward
     sub rsp, 32                               ; shadow space, identical to the prefill call sites
     call forward_f32                          ; logits = forward_f32,transformer, token, pos
     add rsp, 32                               ; restore shadow space
     jmp .chat_ingest_flush                    ; store logits + flush check
.chat_ingest_q8:
     sub rsp, 32                               ; shadow space
     call forward_q8                           ; logits = forward_q8,transformer, token, pos
     add rsp, 32                               ; restore shadow space
.chat_ingest_flush:
     mov c_logits, rax                         ; the LAST prompt token's logits drive sampling, same store as the prefill path
     cmp dword [rel g_stream_remove_count], remove_BATCH ; pending evictions at the supply/consume threshold?
     jb .chat_ingest_next                      ; no -> next token
     call v7s_stream_remove_flush              ; wrapped prompts supply/consume the pending log too, cap never reached
.chat_ingest_next:
     inc edi                                   ; i is incremented
     inc r15d                                  ; pos++
     jmp .chat_ingest_lp                       ; next prompt token
.chat_ingest_done:
     mov c_pos, r15d                           ; session position advances past the prompt
     jmp .chat_sample_first                    ; never fall into the prefill dispatch
.chat_route_std:                               ; standard route, legacy check done; single/multi below
     cmp eax, 1                                ; num_prompt_tokens == 1?
     je .chat_single_prompt                    ; Exactly one, use per-token forward
; Prefill path: logits = prefill_forward_*,transformer, tokens, n_tokens, start_pos
     mov rcx, c_transformer                    ; shared arg1: Transformer*
     mov rdx, c_prompt_tokens                  ; shared arg2: prompt_tokens array
     mov r8d, c_num_prompt_tokens              ; shared arg3: number of prompt tokens
     mov r9d, c_pos                            ; shared arg4: start_pos = ctx->pos, multi-turn
     cmp dword c_quantize, 0                   ; select Q8/F32, read only for the zero test
     jz .chat_prefill_f32                      ; quantize==0: F32 prefill
     sub rsp, 32                               ; one Win64 shadow allocation for either call
     call prefill_forward_q8                   ; Returns logits pointer in RAX
     jmp .chat_prefill_return                  ; skip F32 prefill path
.chat_prefill_f32:
     sub rsp, 32                               ; own Win64 shadow allocation on the F32 path
     call prefill_forward_f32                  ; Returns logits pointer in RAX
.chat_prefill_return:
     add rsp, 32                               ; shared shadow release
.chat_prefill_done:
     mov c_logits, rax                         ; Cache logits pointer
; Fill out_tokens[start_pos + i] = prompt_tokens[i] so V7-GET-TOKEN sees
; the full prompt history, absolute positions; seq_len bound FATAL-checked above
; StreamingLLM: this loop ALSO fills the eviction ring, prefill never runs the
; forward slot resolver, so a legacy-routed prompt's ring rows would otherwise
; stay zero and a later eviction would report vocab[0] as the evicted text.
     mov r11, c_out_tokens                     ; destination history base
     mov r10d, c_pos                           ; first linear position in both arrays
     mov rdx, c_prompt_tokens                  ; source token array
     mov rsi, rdx                              ; load source for the history copy
     lea rdi, [r11 + r10*DWORD_SCALE]          ; destination history slice
     mov ecx, c_num_prompt_tokens              ; number of tokens to copy
     rep movsd                                 ; copy the history slice in one pass
     cmp dword [rel g_stream_enable], 0        ; is there an active streaming ID ring?
     je .chat_otk_done                         ; the history copy is sufficient
     mov r9, [rel g_stream_remove_ids]         ; ring base, still in linear fill
     mov rsi, rdx                              ; restore original source
     lea rdi, [r9 + r10*DWORD_SCALE]           ; ring slice uses the same starting position
     mov ecx, c_num_prompt_tokens              ; same token count
     rep movsd                                 ; fill the ring slice in one pass
.chat_otk_done:
; pos += num_prompt_tokens
     mov eax, c_num_prompt_tokens              ; Load prompt token count
     add c_pos, eax                            ; pos += num_prompt_tokens, preserve previous pos
     jmp .chat_sample_first                    ; Go straight to sampling
.chat_single_prompt:
     mov rcx, c_transformer                    ; shared arg1: Transformer*
     mov rax, c_prompt_tokens                  ; prompt token array
     mov edx, [rax]                            ; shared arg2: prompt_tokens[0]
     mov r8d, c_pos                            ; shared arg3: pos = ctx->pos, multi-turn
     cmp dword c_quantize, 0                   ; select Q8/F32, read only for the zero test
     jz .chat_single_f32                       ; quantize==0: F32 forward
     sub rsp, 32                               ; shared Win64 shadow allocation
     call forward_q8                           ; Returns logits pointer in RAX
     jmp .chat_single_return                   ; skip F32 path
.chat_single_f32:
     sub rsp, 32                               ; own Win64 shadow allocation on the F32 path
     call forward_f32                          ; Returns logits pointer in RAX
.chat_single_return:
     add rsp, 32                               ; shared shadow release
.chat_single_done:
     mov c_logits, rax                         ; Cache logits pointer
; Fill out_tokens[pos] = prompt_tokens[0], V7-GET-TOKEN history; pos not yet advanced
     mov r11, c_out_tokens                     ; r11 = out_tokens base
     mov rax, c_prompt_tokens                  ; rax = prompt_tokens array
     mov eax, [rax]                            ; eax = prompt_tokens[0]
     mov r10d, c_pos                           ; r10d = absolute position, zero-extends r10 for indexing
     mov [r11 + r10*DWORD_SCALE], eax          ; out_tokens[pos] = token id
; pos++, multi-turn aware, increments existing position
     inc c_pos                                 ; pos++ after single prompt token, preserve previous pos
; Fall through to sample first token
.chat_sample_first:
