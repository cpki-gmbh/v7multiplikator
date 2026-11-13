; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     default rel                               ; use RIP-relative addressing by default
%define DES_EXIT_STDOUT_DEAD 75                ; exit code for TerminateProcess when stdout_handle init fails, distinct from FATAL exit 1
     section .text
     global debug_ensure_std                   ; export function entry
debug_ensure_std:
     cmp qword [stdout_handle], 0              ; already initialized?
     je .des_init                              ; no -> cold init path, taken once per process; hot path falls through
     ret                                       ; hot path: zero taken branches per call
.des_init:
     push rbp                                  ; save caller's frame pointer, non-volatile, preserved across the API call
     sub rsp, 32                               ; shadow space for API_CALL
     CORRECT_RSP                               ; align for the GetStdHandle call, parity is variable at this entry
     API_CALL [__imp_GetStdHandle], STD_OUTPUT_HANDLE ; rax = stdout handle
     RESTORE_RSP                               ; undo the parity correction, no GPR overwrite; the cmp rax, -1 below sets its own flags
     add rsp, 32                               ; undo shadow space
     pop rbp                                   ; restore caller's frame pointer
     cmp rax, -1                               ; INVALID_HANDLE_VALUE?
     je .des_dead                              ; yes, stdout does not exist, never cache sentinel
     test rax, rax                             ; NULL handle, GetStdHandle failure ?
     jz .des_dead                              ; yes, same dead state
     mov [stdout_handle], rax                  ; cache for later use
     ret                                       ; return to caller
.des_dead:
     push rbp                                  ; frame for the terminate call
     sub rsp, 32                               ; shadow space for API_CALL
     mov rcx, -1                               ; hProcess = GetCurrentProcess, pseudo-handle
     mov edx, DES_EXIT_STDOUT_DEAD             ; uExitCode = 75, distinctive code for "stdout dead at ensure_std", differs from FATAL exit 1
     CORRECT_RSP                               ; align for the TerminateProcess call
     API_CALL [__imp_TerminateProcess], rcx, rdx ; terminate immediately, never returns
     RESTORE_RSP                               ; undo the parity correction, never reached, TerminateProcess does not return
fatal_print_msg:
     SF_START fatal_print_msg, rbx, rdi        ; prologue: save non-volatiles rbx, message holder + rdi, STR_LEN overwrites it
     SF_COMMIT                                 ; finalize stack frame layout
     mov rbx, rcx                              ; save string pointer in non-volatile rbx
     call debug_ensure_std                     ; ensure stdout_handle is cached
     STR_LEN rbx, r8d                          ; r8d = string length, overwrites rax, rcx, rdi, rdi saved above
     mov rcx, [stdout_handle]                  ; arg1 = stdout handle
     mov rdx, rbx                              ; arg2 = string buffer pointer
; r8d already = length from STR_LEN, arg3
     xor r9d, r9d                              ; arg4 = NULL, bytes-written count is discarded, WriteFile accepts a NULL out-pointer
     CORRECT_RSP                               ; align for the WriteFile call
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write the message to stdout
     RESTORE_RSP                               ; undo the parity correction, writes no GPR; SF_END below depends on no flags
     SF_END rbx, rdi                           ; epilogue: restore rbx + rdi, return to caller
     global v7s_fatal_error                    ; exported: fatal error handler, never returns
     SF_START v7s_fatal_error                  ; prologue: frame only, rbx never touched here, fatal_print_msg saves its own
     SF_COMMIT                                 ; finalize stack frame layout
     call fatal_print_msg                      ; print the message, shared print half, returns normally
     mov rcx, -1                               ; hProcess = GetCurrentProcess, = pseudo-handle -1
     mov edx, 1                                ; uExitCode = 1, error exit
     API_CALL [__imp_TerminateProcess], rcx, rdx ; terminate, never returns
%ifdef TRACE_ENABLE
%define TW_FRAME_SZ 120                        ; two pushes, 16 + 120 = 136 = 8 mod 16, which keeps the WriteFile call 16-aligned; room for the buffer + value parking above RSP, do NOT shrink below 104, alignment breaks
%define TW_HEX_BUF 64                          ; 18-byte hex buffer start, offset 0..17 relative to TW_HEX_BUF, plus the appended LF at TW_LF_OFF, entirely ABOVE API_CALL's highest write
%define TW_HEX_LEN 18                          ; "0x" + 16 hex digits = 18 characters, the appended LF makes the written length TW_HEX_LEN + 1
%define TW_LF_OFF (TW_HEX_BUF + TW_HEX_LEN)    ; byte offset of the appended LF, buffer end, free space above the value park
%define TW_HEX_DIGIT_OFF 1                     ; digit store base adjustment: rdi = TW_HEX_BUF + this, so [rdi + ecx] covers buffer offsets 2..17 as ecx counts 16..1, the "0x" prefix owns 0 and 1 ; descending slots + LSB-first extraction = ascending, MSB-first read order
%define TW_VALUE_OFF 88                        ; qword value parking above the buffer, 8-aligned, survives debug_ensure_std
trace_write_hex:
     push rdi                                  ; save rdi for string buffer pointer
     push rsi                                  ; save rsi for hex lookup table
     sub rsp, TW_FRAME_SZ                      ; allocate shadow + buffer + alignment
     mov qword [rsp + TW_VALUE_OFF], rcx       ; park value BEFORE the call, debug_ensure_std overwrites volatiles on the slow path
     call debug_ensure_std                     ; ensure stdout_handle is cached
     mov r10, [rsp + TW_VALUE_OFF]             ; r10 = value to convert, r10 is volatile, reload must be after the call
     lea rdi, [rsp + TW_HEX_BUF]               ; rdi = hex output buffer start
     lea rsi, [hex_chars]                      ; rsi = hex digit lookup table
     mov word [rdi], '0' | ('x' << 8)          ; write "0x" prefix as a single word
     lea rdi, [rsp + TW_HEX_BUF + TW_HEX_DIGIT_OFF] ; digit store base: as ecx counts 16..1 the store lands at rdi+16 down to rdi+1, i.e. buffer offsets 17 down to 2, the "0x" prefix owns 0 and 1 ; the DESCENDING ecx pairs with the loop's LSB-first extraction so the string reads MSB-first
     mov ecx, 16                               ; 16 hex nibbles to convert
.tw_hex_loop:
     mov eax, r10d                             ; eax = current low 32 bits, the nibble source MUST be refreshed from r10 each iteration, eax is stale scratch otherwise
     and eax, 15                               ; extract LOWEST nibble, bits 3:0, the 32-bit and zero-extends, so rax is a clean 0..15 lookup index
     mov dl, [rsi + rax]                       ; lookup hex character for nibble value
     mov [rdi + rcx], dl                       ; store nibble in its fixed slot; the loop extracts LSB-first so it lands at buffer[17] down to buffer[2] in ascending read order
     shr r10, 4                                ; shift next nibble, bits 7:4, then 11:8, ... into position 3:0
     dec ecx                                   ; decrement nibble counter
     jnz .tw_hex_loop                          ; loop until all 16 nibbles converted
     mov byte [rsp + TW_LF_OFF], 10            ; append LF: callers get the newline in the same write, the slot is the buffer end, above the value park
     lea rdx, [rsp + TW_HEX_BUF]               ; rdx = complete hex string, "0x" + 16 digits + LF
     mov r8d, TW_HEX_LEN + 1                   ; r8d = 19 bytes to write, 18 chars + the LF
     mov rcx, [stdout_handle]                  ; arg1 = stdout handle
     xor r9d, r9d                              ; arg4 = NULL, bytes-written count is discarded
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write hex string to stdout
     add rsp, TW_FRAME_SZ                      ; deallocate stack frame
     pop rsi                                   ; restore rsi
     pop rdi                                   ; restore rdi
     ret                                       ; return to caller
%endif                                         ; TRACE_ENABLE: trace_write_hex only assembles for trace builds
; %undef cleanup: file-local names leave scope here, house convention --
%undef DES_EXIT_STDOUT_DEAD
%undef TW_FRAME_SZ
%undef TW_HEX_BUF
%undef TW_HEX_LEN
%undef TW_LF_OFF
%undef TW_HEX_DIGIT_OFF
%undef TW_VALUE_OFF
