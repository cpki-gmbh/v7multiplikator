; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; stream_cfg.inc, StreamingLLM configuration storage, stateless ring cache .
; Include from v7.asm directly AFTER v7_data.inc, constants must precede the
; forward/chat includes that use them, NASM equ/%define resolve in pass 1 .
; slot = f,pos is a pure function: the only words that ever change are the
; config dwords, API writes, the eviction bookkeeping words, and
; g_stream_slot, one write per token in forward_*, before any dispatch .

; --- eviction callback tuning ------------------------------------------------
; INVARIANT: each forward call under streaming appends exactly ONE eviction
; and chat checks count >= remove_BATCH after every token at all three sites
; ingest loop, decode loop, turn end, so engine-driven flows never exceed
; remove_BATCH. The forward push itself FATALs at remove_LOG_CAP: an undrained
; raw-forward loop, streaming on, no chat session triggers FATAL instead of
; writing past the log into the text buffer.
     remove_BATCH equ 256                      ; flush the pending log every N evictions, callback latency bound
     remove_LOG_CAP equ remove_BATCH           ; pending eviction log capacity == the max reachable count, drained at remove_BATCH
     remove_TEXT_CAP equ 65536                 ; text buffer bytes: remove_BATCH * max vocab piece, 128-byte tokenizer import cap = 32 KB, so one batch always fits

%if remove_LOG_CAP < remove_BATCH              ; the log must hold the max reachable pending count
%error "remove_LOG_CAP smaller than remove_BATCH: the flush cadence would overflow the eviction log" ; assembly-time invariant assert
%endif

     section .bss
     align 16                                  ; ring/log/text want cache-line neighbours, checker: align is exempt from comments
g_stream_enable:
     resd 1                                    ; 0 = legacy path, default, 1 = StreamingLLM ring
g_stream_enable_saved:
     resd 1                                    ; outer session's mode during a nested v7s_generate swap
g_stream_otk_live_saved:
     resd 1                                    ; outer session's token-history validity during a nested v7s_generate swap
g_stream_sinks:
     resd 1                                    ; S: immutable sink rows, >= 1
g_stream_window:
     resd 1                                    ; W: rolling window rows, >= 1 ; fill grade = S + W
g_stream_cap:
     resd 1                                    ; C = S + W, derived at set time; <= seq_len
g_stream_slot:
     resd 1                                    ; slot of the CURRENT token, published by forward_*
g_stream_remove_ids:
     resq 1                                    ; eviction ring base, C dwords, slot-indexed; setter-allocated
g_stream_remove_count:
     resd 1                                    ; pending evicted-token ids in the log, flush drains to 0
g_stream_remove_first_pos:
     resd 1                                    ; absolute position of the first eviction in the pending batch
g_stream_remove_warm:
     resd 1                                    ; warm-up latch: >0 after a KV load, fill ring, suppress logging
g_stream_win_valid:
     resd 1                                    ; valid window rows [S, S+valid : W in steady state, shrunk by rewrites, kernel reader, rewrite + wrapped-forward writers
g_stream_otk_live:
     resd 1                                    ; out_tokens[0, base_pos is this process's live-built session prefix, set at chat entry when the turn starts at pos 0, chat.inc; cleared by the KV load, kv_cache_file.inc, the KV file does not serialize token ids
g_stream_remove_log:
     resd remove_LOG_CAP                       ; pending evicted-token ids, turn-scoped, BSS
g_stream_remove_text:
     resb remove_TEXT_CAP                      ; batch text buffer, raw vocab pieces, NUL-terminated
g_stream_remove_batch_first:
     resd 1                                    ; published: first_pos of the CURRENT batch, split-aware
g_stream_remove_batch_tokens:
     resd 1                                    ; published: eviction count of the CURRENT batch

