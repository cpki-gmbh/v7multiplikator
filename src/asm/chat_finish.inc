; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; chat_finish.inc, finish path, all FATAL labels, function epilogue .
; ANCHOR RULE: local labels only, the non-local label anchor stays v7s_chat, SF_START in chat.inc .
; Entered from every stop site across the fragments; ends .chat_done + SF_END, the function epilogue .

.chat_finish:
     cmp c_rw_muted, 0                         ; a rewrite parked the warm latch, either tier, mid-suffix stop ?
     je .chat_finish_rw_latch                  ; no: zero-overhead path
     mov eax, c_rw_latch                       ; eax = the saved warm-latch value
     mov [rel g_stream_remove_warm], eax       ; restore the eviction latch: a missed restore here drops future evictions, the latch would stay 0x7FFFFFFF and every wrapped forward would just decrement it
     mov c_rw_muted, 0                         ; the latch is back
.chat_finish_rw_latch:
     cmp c_rw_tier2, 0                         ; a TIER-2 rewrite in flight, the tracker needs the re-derive ?
     je .chat_finish_rw_done                   ; no: the tracker was never touched
     mov eax, c_pos                            ; eax = the EFFECTIVE stream end when the stop hit, may be mid-suffix
     add eax, [rel g_stream_window]            ; + W
     sub eax, c_rw_end                         ; - the PLANNED end = the true surviving-window count: the fill-in covered the planned window's head, the accepted suffix only [A, c_pos, rows in [c_pos, planned_end were never written
     jnc .chat_finish_vclamp                   ; clamp low to [0, W]
     xor eax, eax                              ; underflow: the discarded tail covered the whole window
.chat_finish_vclamp:
     cmp eax, [rel g_stream_window]            ; clamp high to W
     jbe .chat_finish_vstore                   ; within W: keep
     mov eax, [rel g_stream_window]            ; cap at W
.chat_finish_vstore:
     mov [rel g_stream_win_valid], eax         ; publish the tracker for the ACTUAL accepted stream, never the value planned for the full suffix
     mov c_rw_tier2, 0                         ; clear the flag
.chat_finish_rw_done:
; Phase 3 timestamp + generated token count, rax/rdx dead, r15d = gen counter
     RDTSC_TO_MEM rdtsc_after_generation       ; statistics: TSC at generation finish
     mov eax, r15d                             ; gen_count register, zero-extends
     mov [rel number_token_generate], rax      ; statistics: generated tokens this round
; Write final sequence position back to ctx->pos, multi-turn support
     mov rax, c_ctx                            ; Load InferenceContext*
     mov ecx, c_pos                            ; Load final position
     mov [rax + CTX_POS], ecx                  ; ctx->pos = final position
     call v7s_stream_remove_flush              ; deliver the turn's remaining evictions, no-op when count == 0
     mov qword [rel g_user_prompt], 0          ; remove the published pointer
     mov rax, c_output                         ; Load output buffer base
     mov ecx, c_out_pos                        ; Load current write position
     mov byte [rax + rcx], 0                   ; output[out_pos] = '\0'
     jmp .chat_done                            ; Jump to normal exit, skip FATAL handlers
.chat_null_output_fatal:
     FATAL "v7s_chat: output buffer is NULL"   ; contract violation: caller must provide output buffer
.chat_output_len_fatal:
     FATAL "v7s_chat: output_len < 1"          ; contract violation: caller must provide output_len >= 1
.chat_null_prompt_fatal:
     FATAL "v7s_chat: user_prompt is NULL"     ; contract violation: prompt required
.chat_kvcache_fatal:
     FATAL "v7s_chat: KV-cache bounds exceeded" ; prompt too long or pos corrupted
.chat_rewrite_history_fatal:
     FATAL "v7s_chat: streaming rewrite is unsupported after a KV-cache load (no recorded token history)"
.chat_rewrite_enc_cap_fatal:
     FATAL "SectorLogo rewrite: replacement exceeds the 16384-token encode budget, shorten the rewritten stream"
.chat_done:
     call v7s_stop_spin                        ; notify pool to enter OS wait; never wait for workers
; Restore the combined rendered + prompt_tokens stack allocation
     add rsp, CHAT_RENDERED_BUF + CHAT_PROMPT_TOKENS_BYTES + BITMAP_BYTES ; Free ~180KB stack, rendered+ptok+bitmap
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; normal exit: restore non-volatile regs and return
