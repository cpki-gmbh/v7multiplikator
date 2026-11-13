; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define CSBI_SIZE 32                           ; CONSOLE_SCREEN_BUFFER_INFO: 22 bytes used, padded
%define CSBI_LEFT 10                           ; SMALL_RECT.Left, after dwSize + dwCursor + wAttributes
%define CSBI_RIGHT 14                          ; SMALL_RECT.Right, word, same record
     extern __imp_GetStdHandle                 ; kernel32 stdout-handle query
     extern __imp_GetConsoleScreenBufferInfo   ; kernel32 screen-buffer geometry query
     extern __imp_CreateFileW                  ; kernel32 CONOUT$ opener, fallback when stdout is piped
     extern __imp_CloseHandle                  ; kernel32 handle closer, the fallback handle is owned by this module
     section .rdata
     align 2
conout_path:
     dw 'C', 'O', 'N', 'O', 'U', 'T', '$', 0   ; the console screen buffer's Win32 namespace path
     section .bss
csbi_buf:
     resb CSBI_SIZE                            ; the query target, filled by GetConsoleScreenBufferInfo
     section .text
     global v7_get_console_width               ; export: console column count for the Logo banner logic
v7_get_console_width:
     push rbx                                  ; rbx is callee-saved: save it, the push also re-aligns the stack to 16
     API_CALL [__imp_GetStdHandle], STD_OUTPUT_HANDLE ; rax = stdout handle, NULL without stdout, -1 on error, both fail the query below
     mov rbx, rax                              ; rbx = handle, callee-saved: kernel32 preserves it across the next call
     lea rdx, [rel csbi_buf]                   ; rdx = csbi buffer, the query's out-parameter
     API_CALL [__imp_GetConsoleScreenBufferInfo], rbx, rdx ; fill csbi; eax = 1 on success
     test eax, eax                             ; query failed, stdout piped, no console attached to it ?
     jz .try_conout                            ; yes: a real console window may still exist, open CONOUT$ and retry
.read_width:
     movzx eax, word [rel csbi_buf + CSBI_RIGHT] ; eax = srWindow.Right
     movzx ecx, word [rel csbi_buf + CSBI_LEFT] ; ecx = srWindow.Left
     sub eax, ecx                              ; eax = Right - Left
     inc eax                                   ; eax = visible column count
     pop rbx                                   ; restore the caller's rbx
     ret                                       ; return to caller
.try_conout:
%ifdef V7_LINUX
     lea rcx, [rel conout_path]                ; PIE: address the console path via RIP-relative lea
     API_CALL [__imp_CreateFileW], rcx, 0xC0000000, 3, 0, 3, 0, 0 ; open CONOUT$: GENERIC_READ|GENERIC_WRITE, share read+write, OPEN_EXISTING, no template/flags
%else
     API_CALL [__imp_CreateFileW], conout_path, 0xC0000000, 3, 0, 3, 0, 0 ; open CONOUT$: GENERIC_READ|GENERIC_WRITE, share read+write, OPEN_EXISTING, no template/flags
%endif
     cmp rax, -1                               ; INVALID_HANDLE_VALUE, no console window at all ?
     je .no_console                            ; yes: default width
     mov rbx, rax                              ; rbx = screen-buffer handle
     lea rdx, [rel csbi_buf]                   ; rdx = csbi buffer
     API_CALL [__imp_GetConsoleScreenBufferInfo], rbx, rdx ; retry the geometry query on the real screen buffer
     test eax, eax                             ; succeeded?
     jz .close_fail                            ; no: close the handle, default width
     API_CALL [__imp_CloseHandle], rbx         ; release the CONOUT$ handle, opened by this module, csbi_buf still holds the query result
     jmp .read_width                           ; shared width computation + pop rbx + ret, csbi_buf holds the retry result; CloseHandle only returns BOOL in eax, it never touches csbi_buf
.close_fail:
     API_CALL [__imp_CloseHandle], rbx         ; release the handle even on the failed retry
.no_console:
     mov eax, 80                               ; classic fallback width
     pop rbx                                   ; restore the caller's rbx
     ret                                       ; return to caller
; ---- file-local defines leave scope here, house convention ----
%undef CSBI_SIZE
%undef CSBI_LEFT
%undef CSBI_RIGHT
