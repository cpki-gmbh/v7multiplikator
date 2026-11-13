; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .text
%define WE2_ESC_BSLASH 1                       ; +1 escape backslash in the 2N+1 lea displacement
%define WE2_QUOTE_BYTE 1                       ; +1 for the quote byte in the 2N+2 byte count
%define WE2_BSLASH_DUP_SCALE 2                 ; SIB scale: each N backslashes duplicates to 2N
%define WE2_ESC_QUOTE_PAIR 2                   ; lone-quote fast path emits exactly this many bytes, backslash + quote
%define WE2_ESC_QUOTE_WORD 0x225C              ; little-endian word: backslash, 0x5C then '"', 0x22
     global win32_escape_inner                 ; export: Windows command-line argument escaper, v7_execute_fmt argv path
win32_escape_inner:
     push rsi                                  ; save non-volatile, src cursor, Win64 ABI
     push rdi                                  ; save non-volatile, dst cursor
     mov rsi, rcx                              ; rsi = source cursor
     mov rdi, rdx                              ; rdi = dest cursor, rdx preserved for length computation at return
     mov r9d, r8d                              ; r9d = capacity remaining, copy from incoming r8d param, all capacity checks use r9d
     xor r10d, r10d                            ; r10d = pending backslash run
.we2_scan:
     mov al, byte [rsi]                        ; current byte; the scanner uses AL only
     cmp al, '\'                               ; backslash starts a run
     je .we2_bs                                ; yes: enter the run handler
     cmp al, '"'                               ; quote needs escaping
     je .we2_quote                             ; yes: escape the quote
     test al, al                               ; NUL terminator
     jz .we2_close                             ; yes: terminate
; ---- plain byte: no run can be pending here, runs flush at .we2_bs exit ----
.we2_plain:
     test r9d, r9d                             ; room for the byte?
     jz .we2_trunc                             ; no: truncate
     stosb                                     ; store AL at [rdi], advance dest, DF=0: Win64 ABI guarantee, leaf never calls/std
     dec r9d                                   ; capacity--
     inc rsi                                   ; advance source
     jmp .we2_scan                             ; next byte
.we2_bs:
     inc r10                                   ; extend backslash run
     inc rsi                                   ; advance source past the backslash
     mov al, byte [rsi]                        ; peek next byte, only AL is tested or copied
     cmp al, '\'                               ; another backslash extends the run?
     je .we2_bs                                ; yes: keep extending, tight self-loop, 1 test per backslash
     test al, al                               ; NUL end-of-string?
     jz .we2_ctx_check                         ; trailing run: the r11d context decides plain N vs doubled 2N
     cmp al, '"'                               ; run ends at a quote?
     je .we2_quote                             ; yes: run passed UNCONSUMED to .we2_quote, never flush first, .we2_quote consumes it and zeroes r10
.we2_flush_plain:
     cmp r9, r10                               ; capacity for the whole pending literal backslash run?
     jb .we2_trunc                             ; no: report -1 without partial run writes
     sub r9, r10                               ; pay once for the run
     mov rcx, r10                              ; number of backslashes
     mov al, '\'                               ; rep stosb source byte
     rep stosb                                 ; emit the literal run as a bulk fill
     xor r10d, r10d                            ; pending run cleared
     jmp .we2_scan                             ; reload the unconsumed byte at [rsi], including NUL
.we2_ctx_check:
     test r11d, r11d                           ; r11d < 0?
     js .we2_flush_plain                       ; yes: no '"' follows this %s, single-width run
; fall through: a '"' follows M format-contributed backslashes, emit 2N - M
.we2_bs_end:
     mov r11d, r11d                            ; zero-extend the proven nonnegative context count
     lea rax, [r10*WE2_BSLASH_DUP_SCALE]       ; full-width doubled run length
     sub rax, r11                              ; subtract the format-contributed backslashes
     js .we2_bs_end_unrepresentable            ; M > 2N: the format's own run cannot coexist with N
     cmp r9, rax                               ; room for the 2N - M run?, both operands nonnegative, so the unsigned compare cannot spuriously jump; a zero-length emit is a no-op below
     jb .we2_trunc                             ; no: truncate cleanly
     sub r9, rax                               ; deduct 2N - M from capacity
     mov rcx, rax                              ; rep stosb count = 2N - M
     mov al, '\'                               ; fill byte
     rep stosb                                 ; emit the run, count 0 writes nothing, rdi unmoved
     xor r10d, r10d                            ; run consumed
     jmp .we2_close                            ; NUL-terminate as usual
.we2_bs_end_unrepresentable:
     FATAL "win32_escape_inner: the format string places more backslashes before its closing quote than this argument can encode"
.we2_quote:
     test r10d, r10d                           ; preceding backslash run?
     jnz .we2_quote_run                        ; yes: fall into the general 2N+1 path
     cmp r9d, WE2_ESC_QUOTE_PAIR               ; room for the 2-byte escape pair?
     jb .we2_trunc                             ; no: truncate cleanly with no partial writes
     sub r9d, WE2_ESC_QUOTE_PAIR               ; deduct the pair from capacity
     mov word [rdi], WE2_ESC_QUOTE_WORD        ; backslash then '"', little-endian, one store
     add rdi, WE2_ESC_QUOTE_PAIR               ; advance dest past the pair
     inc rsi                                   ; advance source past quote
     jmp .we2_scan                             ; next byte
.we2_quote_run:
     lea rcx, [r10*WE2_BSLASH_DUP_SCALE + WE2_ESC_BSLASH] ; full-width escaped backslash count
     xor r10d, r10d                            ; run consumed
     lea rax, [rcx + WE2_QUOTE_BYTE]           ; total bytes needed for the run and quote
     cmp r9, rax                               ; room for escape sequence + quote?
     jb .we2_trunc                             ; no: truncate cleanly with no partial writes
     sub r9, rax                               ; deduct 2N+2 total bytes, backslashes + quote from capacity
     mov al, '\'                               ; al = backslash
     rep stosb                                 ; fill 2N+1 backslashes, DF=0
     mov byte [rdi], '"'                       ; write the escaped quote
     inc rdi                                   ; advance dest past quote
     inc rsi                                   ; advance source past quote
     jmp .we2_scan                             ; next byte
.we2_close:
     test r9d, r9d                             ; room for the NUL?
     jz .we2_trunc                             ; no: truncate
     mov byte [rdi], 0                         ; NUL-terminate
     mov rax, rdi                              ; rax = dest end
     sub rax, rdx                              ; rax = bytes written, excl. NUL, rdx preserved as the base
     jmp .we2_ret                              ; shared restore-and-return, rsi/rdi pushes outstanding
.we2_trunc:
     or rax, -1                                ; rax = -1, 64-bit truncation marker
.we2_ret:
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     ret                                       ; return length, or -1 on truncation
; %undef cleanup: file-local defines leave scope here, house convention --
%undef WE2_ESC_QUOTE_WORD
%undef WE2_ESC_QUOTE_PAIR
%undef WE2_BSLASH_DUP_SCALE
%undef WE2_QUOTE_BYTE
%undef WE2_ESC_BSLASH
