; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; stream_evict.inc, StreamingLLM eviction batch flush, SectorLogo hook .
; Include from v7.asm directly AFTER chat.inc, call sites live there .
; Delivers the pending eviction log as text: writes two microdb keys
; "stream_remove_text", "stream_remove_meta" via ht_add, then evaluates the
; Logo script under key "on_stream_evict" if the hook is enabled. The script
; reads the batch with the existing DatabaseGet primitive and does whatever
; it pleases, typically DatabaseSet into long-term-memory keys .
; chat.inc defines VOCAB_PTR_SCALE but %undef's it at its EOF; this file is
; included after chat.inc, so it needs its own copy.

%define VOCAB_PTR_SCALE 8                      ; sizeof,char* for the vocab array, same value chat.inc uses
%define SEF_FIRST_TAIL_OFF 4                   ; first= prefix: byte offset of the final "t=" word, after the "firs" dword
%define SEF_FIRST_LEN 6                        ; bytes in the "first=" prefix, the number overwrites the terminator position

     section .data
sl_remove_text_key:
     db "stream_remove_text", 0                ; DB key: batch text
sl_remove_meta_key:
     db "stream_remove_meta", 0                ; DB key: "first=<pos> tokens=<k>"
sl_meta_tokens_s:
     db " tokens=", 0                          ; separator between the two decimals
sl_remove_print_first:
     db "stream_evict: first=", 0              ; console notice prefix, the numbers follow as decimal

     section .bss
g_stream_remove_meta_buf:
     resb 48                                   ; "first=" ~11 + " tokens=" 8 + 2x10 digits + NUL = 40 <= 48
g_stream_remove_busy:
     resd 1                                    ; reentrancy gate: 1 while the flush runs, a callback that drives generation would append to the log mid-drain and recurse

     section .text
     global v7s_stream_remove_flush            ; export the flush entry, SF_START re-emits it, established pattern
v7s_stream_remove_flush:
     SF_START v7s_stream_remove_flush, rsi, rdi, r12, r13, r14, r15 ; frame and loop registers
     SF_DWORD sef_count                        ; log count snapshot
     SF_COMMIT                                 ; commit the flush frame, locals + reserved shadow
     cmp dword [rel g_stream_remove_busy], 0   ; reentrant flush, a callback drove a generation ?
     jne .sef_reent_fatal                      ; yes: contract violation, the recursion would append to this log mid-drain and never terminate
     mov dword [rel g_stream_remove_busy], 1   ; mark the flush in flight, cleared at the shared exit; inf_cleanup clears it after a caught FATAL
     mov eax, [rel g_stream_remove_count]      ; pending evictions, 0 = nothing to deliver
     test eax, eax                             ; count == 0?
     jz .sef_exit                              ; nothing pending: the common no-op path
     mov dword sef_count, eax                  ; snapshot BEFORE rax is spent on the loads below
; a late snapshot would turn sef_count into the
; vocab pointer's low dword, runaway loop, OOB reads
     mov eax, [rel sl_hook_enabled]            ; sidecar hook attached?
     or eax, [rel sl_evict_enabled]            ; or an on_stream_evict script lives in the DB? the refresh flag re-probes the key every chat; the or already sets ZF from this result
     jnz .sef_go                               ; a consumer listens -> deliver the batch
.sef_drop:
     mov dword [rel g_stream_remove_count], 0  ; no Logo world attached: drop the batch, documented contract
     jmp .sef_exit                             ; drop = reset the log, no interpreter start
.sef_go:
     mov rax, [rel g_inf_ctx_block]            ; rax = ctx, alive: chat is running
     mov rax, [rax + CTX_TOKENIZER]            ; rax = Tokenizer*
     mov r13, [rax + TOK_VOCAB]                ; r13 = vocab, char**
     xor r14d, r14d                            ; r14d = log index i
     xor r15d, r15d                            ; r15d = text bytes used in the batch
.sef_loop:
     cmp r14d, sef_count                       ; i < snapshot count?
     jae .sef_tail                             ; log exhausted -> deliver the final chunk
     lea rdx, [rel g_stream_remove_log]        ; rdx = pending eviction log base
     mov ecx, [rdx + r14*DWORD_SCALE]          ; ecx = evicted token id
     mov rsi, [r13 + rcx*VOCAB_PTR_SCALE]      ; rsi = piece, char*, NUL-terminated
     mov rdi, rsi                              ; rdi = piece for the inline strlen
     xor eax, eax                              ; al = 0, search for NUL
     mov ecx, -1                               ; rcx = max scan count
     repne scasb                               ; scan for NUL
     not ecx                                   ; ~count = scanned bytes incl. NUL
     dec ecx                                   ; ecx = piece length, may be 0
     mov edx, ecx                              ; edx = piece length, survives the append below
; NO overflow check here, by proof: the log holds at most remove_BATCH evictions
; the three flush sites drain at exactly that count and every vocab piece is
; capped by the tokenizer import, max_len = 128 bytes, so a full batch can
; never exceed 256 * 128 = 32 KB against the 64 KB buffer, the append below
; is always in bounds and chunk-split machinery cannot fire.
.sef_append:
     lea rdi, [rel g_stream_remove_text]       ; rdi = batch text buffer base
     add rdi, r15                              ; rdi = write cursor, text + used
     call chat_decode_piece                    ; EDX = length, RSI = source, RDI = output cursor
     mov byte [rdi], 0                         ; terminate at the new end of the batch text
     add r15d, eax                             ; accumulate the exact decoded byte count
     inc r14d                                  ; next log entry
     jmp .sef_loop                             ; continue draining the log
.sef_tail:
; The bounded log always fits in the text buffer and delivers one batch.
     mov eax, [rel g_stream_remove_first_pos]  ; absolute position of the batch's first eviction
     mov [rel g_stream_remove_batch_first], eax ; publish the first position of this flush
     mov [rel g_stream_remove_batch_tokens], r14d ; publish the count already scanned
     lea rdx, [rel g_stream_remove_text]       ; rdx = batch text buffer base
; DB key 1: the batch text
     lea rcx, [rel sl_remove_text_key]         ; arg1 = "stream_remove_text"
     lea rdx, [rel g_stream_remove_text]       ; arg2 = batch text buffer
     mov r8d, r15d                             ; arg3 = byte length, excluding NUL
     xor r9d, r9d                              ; arg4 = persist 0, temp; the Logo side persists what it keeps
     call ht_add                               ; write the batch text into the microdb
; DB key 2: the batch metadata "first=<pos> tokens=<k>"
     lea rdi, [rel g_stream_remove_meta_buf]   ; rdi = metadata builder cursor
     mov dword [rdi], 0x73726966               ; literal ASCII "firs" (little endian)
     mov word [rdi + SEF_FIRST_TAIL_OFF], 0x3D74 ; literal ASCII "t=", no NUL needed, the decimal follows immediately
     add rdi, SEF_FIRST_LEN                    ; exactly six prefix bytes
     mov eax, [rel g_stream_remove_batch_first] ; eax = this chunk's absolute first position
     call lg_num_to_str                        ; append decimal first_pos, rdi = past last digit; leaf, no NUL
     mov rax, [rel sl_meta_tokens_s]           ; the eight literal ASCII bytes of " tokens=", NUL excluded
     mov [rdi], rax                            ; emit the separator as one 64-bit write
     add rdi, 8                                ; point at the next decimal
     mov eax, [rel g_stream_remove_batch_tokens] ; eax = eviction count of this chunk
     call lg_num_to_str                        ; append decimal token count
     mov byte [rdi], 0                         ; NUL-terminate the metadata
     lea rdx, [rel g_stream_remove_meta_buf]   ; rdx = metadata buffer base
     sub rdi, rdx                              ; rdi = exact byte count, 64-bit sub: no 32-bit address truncation
     mov r8d, edi                              ; r8d = exact content length, NUL-exclusive, the DatabaseGet reader appends its own terminator
     lea rcx, [rel sl_remove_meta_key]         ; arg1 = "stream_remove_meta"
     lea rdx, [rel g_stream_remove_meta_buf]   ; arg2 = metadata buffer
     xor r9d, r9d                              ; persist 0, temp
     call ht_add                               ; write the batch metadata into the microdb
; Always-on console notice: an eviction batch is a VISIBLE engine event, not a
; debug artifact, every delivered batch prints its span so a session's proof
; of eviction, "did it happen, where, how much" is right in the window.
; The batch globals were published above and stay valid until the next flush.
     WRITE_STR sl_remove_print_first           ; "stream_evict: first="
     mov r10d, [rel g_stream_remove_batch_first] ; the chunk's absolute first position, loaded after the overwriteing macro
     WRITE_UINT64 r10                          ; decimal first position
     WRITE_STR sl_meta_tokens_s                ; " tokens="
     mov r10d, [rel g_stream_remove_batch_tokens] ; the chunk's eviction count
     WRITE_UINT64 r10                          ; decimal token count
     WRITE_NL                                  ; end the notice line
; Zero the pending count BEFORE the callback runs: text + meta are already in
; the DB, so the delivery is durable, and a callback script that, against
; the deployment contract ran a forward pass would otherwise append at
; log[256], one PAST the log capacity, into the start of the text buffer.
     mov dword [rel g_stream_remove_count], 0  ; exactly-once: the batch leaves the pending log now
; the callback: evaluate the deployment's Logo script, absent key -> -1, harmless
     lea rcx, [rel stream_remove_script_key]   ; arg1 = "on_stream_evict" script key, shared with sectorlogo_data.inc
     call v7s_sectorlogo_eval_script           ; the committed frame provides shadow space
.sef_exit:
     mov dword [rel g_stream_remove_busy], 0   ; clear the reentrancy guard after the callback
     SF_END rsi, rdi, r12, r13, r14, r15       ; restore the committed frame and return
.sef_reent_fatal:
     FATAL "stream_evict: the eviction callback re-entered generation, a callback must not drive forwards"

; %undef cleanup: file-local defines leave scope here, house convention --
%undef VOCAB_PTR_SCALE
