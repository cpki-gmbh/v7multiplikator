; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; --- getchar frame layout, 56 bytes; 2 pushes + 56 = 72; entry 8-72 = 0 mod 16 at call sites ---
%define SL_GI_FRAME 56                         ; frame bytes: API shadow + alignment, stack args staged inside API_CALL
; ============================================================================
; sl_console_getchar:
; ============================================================================
sl_console_getchar:
     push rbx                                  ; save non-volatile. RSP = 0 mod 16
     push rsi                                  ; save non-volatile, rep movsb cursor inside the read paths.RSP = 8 mod 16
     sub rsp, SL_GI_FRAME                      ; API frame, 2 pushes + 56 = 72; entry 8-72 stays 0 mod 16 at call sites
     mov eax, [rel sl_conin_fixed]             ; handle kind, 0 = never probed
     test eax, eax                             ; kind probe done?
     jz sl_gi_lazy_init                        ; no, first call : resolve + probe kind
sl_gi_dispatch:
     mov rbx, [rel sl_hStdIn]                  ; cached handle. Loaded HERE rather than on the fast path only: the lazy fixup arms jump straight to this label, so a load placed above the label would leave that path inheriting rbx from sl_gi_lazy_init by coincidence
     cmp eax, SL_HKIND_CONSOLE                 ; console input?
     je sl_gi_console_poll                     ; yes: PeekConsoleInputA path
     cmp eax, SL_HKIND_PIPE                    ; pipe / redirected?
     je sl_gi_pipe_poll                        ; yes: PeekNamedPipe path
     cmp eax, SL_HKIND_OTHER                   ; regular file, stdin < file ?
     je sl_gi_other_read                       ; yes: a disk ReadFile cannot block
; ---- unclassifiable handle: NIL with no probe. GetFileType said FILE_TYPE_UNKNOWN and
; the "Unknown: NIL, always" policy needs no availability check at all ----
     jmp sl_gi_nil                             ; kind 4, SL_HKIND_NONE : NIL, without a probe

%macro SL_GI_REQ_PIPE 0
     mov eax, dword [rel sl_pipe_avail]        ; bytes PeekNamedPipe just proved available
     cmp eax, SL_GC_BUF_SIZE                   ; more than the buffer holds?
     jbe %%req_ok                              ; no, ask for exactly what is there
     mov eax, SL_GC_BUF_SIZE                   ; yes, clamp to the buffer, the rest stays queued for the next ReadKey
%%req_ok:
     mov dword [rel sl_gc_req], eax            ; publish the request
%endmacro
; SL_GI_CONSOLE_ARGS

%macro SL_GI_CONSOLE_ARGS 0
     mov rcx, rbx                              ; hConsole
     lea rdx, [rel sl_ir_buf]                  ; INPUT_RECORD buffer, the API writes a whole record per nLength element
     mov r8d, 1                                ; nLength = 1 record, MUST be >= 1, array-element count
     lea r9, [rel sl_gc_got]                   ; &events_read, the API requires it
%endmacro

%macro SL_GI_PUBLISH 1
     mov eax, %1                               ; kind, live in eax for the dispatch chain, no memory re-read
     mov [rel sl_conin_fixed], eax             ; same kind, published for the next call's fast path
     jmp sl_gi_dispatch                        ; reuse the steady-state dispatch entry directly
%endmacro
; ============================================================================
; Console path: PeekConsoleInputA + ReadConsoleInputA, never blocks, the
; character is built from the peeked record, no ReadFile on this path
; ============================================================================
sl_gi_console_poll:
; ---- peek at the head of the console input queue ----
     SL_GI_CONSOLE_ARGS                        ; peek: &sl_ir_buf, nLength=1, &sl_gc_got
     CORRECT_RSP                               ; align for PeekConsoleInputA
     API_CALL [__imp_PeekConsoleInputA], rcx, rdx, r8, r9 ; peek at head record
     RESTORE_RSP                               ; undo the probe
     test eax, eax                             ; PeekConsoleInputA success?
     jz sl_gi_nil                              ; no: return NIL
     cmp dword [rel sl_gc_got], 0              ; events_read count: queue empty?, flags straight from memory; the next instruction overwrites eax anyway
     jz sl_gi_nil                              ; yes: return NIL
; ---- examine the head record's EventType ----
     cmp word [rel sl_ir_buf + SL_IR_EVENTTYPE_OFF], SL_KEY_EVENT ; KEY_EVENT?
     jne sl_gi_console_skip                    ; no: skip this record and re-peek
; ---- key event: check bKeyDown ----
     cmp dword [rel sl_ir_buf + SL_IR_KEY_BDOWN_OFF], 0 ; bKeyDown?
     je sl_gi_console_skip                     ; key-up: skip
; ---- key-down: check uChar, the AsciiChar / UnicodeChar union ----
     cmp word [rel sl_ir_buf + SL_IR_KEY_ASCII_OFF], 0 ; uChar != 0? WORD, not byte: AsciiChar is 0 for every non-ASCII key and the character then lives in UnicodeChar at the same offset
     je sl_gi_console_skip                     ; function key / arrow / dead key: skip
     SL_GI_CONSOLE_ARGS                        ; key event accepted: hConsole, &sl_ir_buf, nLength=1, &sl_gc_got
     CORRECT_RSP                               ; align for ReadConsoleInputA
     API_CALL [__imp_ReadConsoleInputA], rcx, rdx, r8, r9 ; remove the accepted record
     RESTORE_RSP                               ; undo the probe
     test eax, eax                             ; drain succeeded?
     jz sl_gi_nil                              ; no: cached console handle is bad, return NIL, same policy as the skip path above, never retry a failing handle: ReadKey is non-blocking by contract
; ---- build the 1-char atom from the PEEKED record: the record was just
; consumed, so nothing is left in the queue's char view, a ReadFile
; here would block until the next keypress and return THAT key ----
     mov byte [rel sl_gc_buf], SL_TYPE_STRING  ; type byte prefix for Intern
     movzx eax, byte [rel sl_ir_buf + SL_IR_KEY_ASCII_OFF] ; the character, uChar low byte of the accepted key-down record; movzx zeroes bits 15:8, so the word store below carries the NUL for free
     mov word [rel sl_gc_buf + SL_ATOM_TEXT_OFF], ax ; the character + NUL, two-byte store
     lea rsi, [rel sl_gc_buf]                  ; rsi = buffer base, type byte + content; Intern reads from type byte
     call Intern                               ; eax = interned one-char string atom
     jmp sl_gi_ret                             ; shared epilogue
sl_gi_console_skip:
; ---- discard the non-key record and re-peek ----
     SL_GI_CONSOLE_ARGS                        ; non-key record: hConsole, &sl_ir_buf, nLength=1, &sl_gc_got
     CORRECT_RSP                               ; align for ReadConsoleInputA
     API_CALL [__imp_ReadConsoleInputA], rcx, rdx, r8, r9 ; consume the one non-key record
     RESTORE_RSP                               ; undo the probe
     test eax, eax                             ; ReadConsoleInputA failed?
     jz sl_gi_nil                              ; yes: the cached console handle is bad, return NIL, do NOT retry a failing handle forever, ReadKey is non-blocking by contract
     jmp sl_gi_console_poll                    ; success: re-peek the next record
; ============================================================================
; Pipe path: PeekNamedPipe, never blocks
; ============================================================================
sl_gi_pipe_poll:
     mov rcx, rbx                              ; arg1: hNamedPipe
     xor edx, edx                              ; arg2: lpBuffer = NULL, peek only
     xor r8d, r8d                              ; arg3: nBufferSize = 0
     xor r9d, r9d                              ; arg4: lpBytesRead = NULL
     lea r10, [rel sl_pipe_avail]              ; arg5: &avail, pipe BSS dword
     CORRECT_RSP                               ; align for PeekNamedPipe
     API_CALL [__imp_PeekNamedPipe], rcx, rdx, r8, r9, r10, 0, 0 ; arg5=&avail, arg6=lpBytesLeftThisMessage=NULL, arg7=lpOverlapped=NULL. The 7th is MANDATORY on x64: the callee reads its 7th stack slot regardless, and with only 6 args APICALL allocates exactly 48 bytes, so that slot stays uninitialized stack and PeekNamedPipe can fail with ERROR_INVALID_PARAMETER
     RESTORE_RSP                               ; undo the probe
     test eax, eax                             ; PeekNamedPipe success?
     jz sl_gi_nil                              ; no, broken pipe / EOF : return NIL
     cmp dword [rel sl_pipe_avail], 0          ; bytes available: data pending?, flags straight from memory; nothing between here and the read reads eax
     jz sl_gi_nil                              ; no: return NIL
; data available: request ONLY the buffered bytes, a full-size request on a
; byte-mode pipe waits for that many bytes, which would park the per-token hook
     SL_GI_REQ_PIPE                            ; publish min,avail, SL_GC_BUF_SIZE
; fall through to ReadFile
; ============================================================================
; Shared ReadFile path, console with real key, pipe with data, other/file
; ============================================================================
sl_gi_other_read:
     mov byte [rel sl_gc_buf], SL_TYPE_STRING  ; type byte prefix for Intern, read at the content offset by Intern
     mov rcx, rbx                              ; hStdIn
     lea rdx, [rel sl_gc_buf + SL_ATOM_TEXT_OFF] ; dest scratch buffer, content after type byte
     mov r8d, dword [rel sl_gc_req]            ; max bytes: full buffer on console/disk, exactly what the pipe probe proved on a pipe
     lea r9, [rel sl_gc_got]                   ; &bytes read
     CORRECT_RSP                               ; align for ReadFile
     API_CALL [__imp_ReadFile], rcx, rdx, r8, r9, 0 ; 5th arg = NULL overlapped
     RESTORE_RSP                               ; undo the probe, writes no register, eax still holds ReadFile's BOOL
     test eax, eax                             ; ReadFile succeeded?
     jz sl_gi_nil                              ; no: lpNumberOfBytesRead is not contractually written on failure, never use it as a write displacement
     mov eax, [rel sl_gc_got]                  ; byte count, 0..64, buffer holds +1 for this NUL
     test eax, eax                             ; EOF, got == 0 ?
     je sl_gi_nil                              ; yes: return NIL
; ---- NUL-terminate and intern the byte sequence as a string atom ----
     lea rsi, [rel sl_gc_buf]                  ; rsi = buffer base, type byte + content; Intern reads from type byte
     mov byte [rsi + rax + SL_ATOM_TEXT_OFF], 0 ; NUL right after the last read byte
     call Intern                               ; eax = interned sequence atom
     jmp sl_gi_ret                             ; shared epilogue, rare path pays the jmp; NIL falls through
; ============================================================================
; Lazy init: resolve cached stdin handle + probe handle kind
; ============================================================================
sl_gi_lazy_init:
; ---- resolve stdin handle, cold: first call only ----
     mov rcx, SL_STD_INPUT_HANDLE              ; GetStdHandle,-10 arg in full-width reg
     CORRECT_RSP                               ; ensure 16-alignment at the API border
     API_CALL [__imp_GetStdHandle], rcx        ; rax = stdin handle
     RESTORE_RSP                               ; undo the alignment probe
     mov rbx, rax                              ; cache it, the return value is deliberately NOT sentinel-gated, see the header : a host with no stdin resolves to SL_HKIND_NONE below and ReadKey returns NIL forever instead of terminating the engine
     mov [rel sl_hStdIn], rbx                  ; sl_hStdIn = handle, a sentineled one is never re-resolved: the kind is cached nonzero, so the lazy fixup runs once per process
; fall through into the kind probe, first call always probes it
; ============================================================================
; Lazy fixup: probe handle kind and apply raw-mode if console
; ============================================================================
; ---- probe handle kind via GetFileType ----
     mov rcx, rbx                              ; hStdIn
     CORRECT_RSP                               ; align for GetFileType
     API_CALL [__imp_GetFileType], rcx         ; rax = file type
     RESTORE_RSP                               ; undo the probe
; classify: 1=disk, 2=char,console, 3=pipe, 0=unknown, disambiguated below
     cmp eax, SL_FILE_TYPE_CHAR                ; console?
     je sl_gi_kind_console                     ; yes
     cmp eax, SL_FILE_TYPE_PIPE                ; pipe?
     je sl_gi_kind_pipe                        ; yes
     cmp eax, SL_FILE_TYPE_DISK                ; regular file?
     je .sl_gi_kind_disk                       ; yes -> the plain ReadFile path
; FILE_TYPE_UNKNOWN, 0 : MSDN's disambiguation is GetLastError == NO_ERROR
; => the handle really is a disk file, GetFileType succeeded and said so .
     CORRECT_RSP                               ; align for GetLastError
     API_CALL [__imp_GetLastError]             ; rax = last error of the GetFileType above
     RESTORE_RSP                               ; undo the probe
     test eax, eax                             ; NO_ERROR => GetFileType meant DISK
     jz .sl_gi_kind_disk                       ; yes -> the plain ReadFile path
     SL_GI_PUBLISH SL_HKIND_NONE               ; unclassifiable: ReadKey returns NIL
.sl_gi_kind_disk:
; regular file: plain ReadFile, Sec: a full-buffer request cannot block
     mov dword [rel sl_gc_req], SL_GC_BUF_SIZE ; disk channel: load the full request in one write
     SL_GI_PUBLISH SL_HKIND_OTHER              ; regular file: dispatch to the plain ReadFile arm
sl_gi_kind_console:
     mov rcx, rbx                              ; hConsole
     lea rdx, [rel sl_conin_mode]              ; &mode scratch for the stdin fixup
     CORRECT_RSP                               ; align for GetConsoleMode
     API_CALL [__imp_GetConsoleMode], rcx, rdx ; read current mode
     RESTORE_RSP                               ; undo the probe
     test eax, eax                             ; GetConsoleMode succeeded => a real console?
     jz .sl_gi_kind_none                       ; no: a FILE_TYPE_CHAR that is not a console, NUL device, serial port, ....Same "no probe exists" case as FILE_TYPE_UNKNOWN above, decline and return NIL, do not park the engine
; clear ENABLE_LINE_INPUT|ECHO_INPUT|WINDOW_INPUT|MOUSE_INPUT, mask ~0x1E
     mov edx, [rel sl_conin_mode]              ; current mode
     and edx, ~ (SL_CONMODE_LINE | SL_CONMODE_ECHO | SL_CONMODE_WINDOW | SL_CONMODE_MOUSE) ; strip line/echo/window/mouse bits
     mov rcx, rbx                              ; hConsole
     CORRECT_RSP                               ; align for the SetConsoleMode raw-mode write
     API_CALL [__imp_SetConsoleMode], rcx, rdx ; raw single-key mode
     RESTORE_RSP                               ; undo the probe
     test eax, eax                             ; SetConsoleMode failed?
     jz .sl_gi_console_fatal                   ; same block: the handle is still line-mode, so ReadFile would block
     mov eax, SL_HKIND_CONSOLE                 ; publish-safe CONSOLE kind for the dispatch chain
     mov [rel sl_conin_fixed], eax             ; record it for the next call's fast path
     mov dword [rel sl_gc_req], SL_GC_BUF_SIZE ; console: raw mode makes a full-buffer request return on the first character
     jmp sl_gi_dispatch                        ; raw mode established: kind already in eax
.sl_gi_kind_none:
; non-console char device, NUL, serial, ... : ReadKey returns NIL
     SL_GI_PUBLISH SL_HKIND_NONE               ; decline, same as the FILE_TYPE_UNKNOWN arm above
.sl_gi_console_fatal:
     FATAL "ReadKey: console mode probe/setup failed for console input" ; refuse to dispatch a potentially blocking console handle
sl_gi_kind_pipe:
; mark as pipe, then dispatch, the pipe arm computes sl_gc_req per call
     SL_GI_PUBLISH SL_HKIND_PIPE               ; pipe: dispatch to the PeekNamedPipe arm
; ============================================================================
; Shared NIL return and epilogue
; ============================================================================
sl_gi_nil:
     xor eax, eax                              ; NIL
sl_gi_ret:
     add rsp, SL_GI_FRAME                      ; restore frame
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     ret                                       ; return NIL, sl_gi_nil or the sequence atom, sl_gi_ret
%undef SL_GI_FRAME
%unmacro SL_GI_REQ_PIPE 0                      ; pipe-arm byte-clamp macro
%unmacro SL_GI_CONSOLE_ARGS 0                  ; console-argument setup macro
%unmacro SL_GI_PUBLISH 1                       ; publish-kind-and-dispatch macro
