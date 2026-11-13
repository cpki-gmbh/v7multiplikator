; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ---- layout constants for Win32 structs, Win64 ABI -----------------------
%define SA_SIZE 24                             ; SECURITY_ATTRIBUTES on Win64, 8-byte aligned
%define STARTUPINFOA_SIZE 104                  ; STARTUPINFOA on Win64
%define PROCESS_INFO_SIZE 24                   ; PROCESS_INFORMATION on Win64
%define V7_EXEC_BUF_SIZE 0x100000              ; 1 MiB capture/read buffer. Also backs v7s_readfile, which is how the SectorLogo autoexec file tier loads autoexec.logo, must stay >= SL_SCRIPT_BUF_SIZE or autoexec.logo loads truncated.
%define V7_CMDLINE_SIZE 65536                  ; BSS reservation for g_cmdline only, must stay >= V7_CMDLINE_LAST+1 ; NOT the enforced spawn-arg ceiling
%define V7_CMDLINE_LAST 32766                  ; last line byte, the NUL slot ; 32766 chars + NUL = 32767 <= the 32768-char CreateProcessA limit
%define V7_FMTBUF_SIZE 98304                   ; 96KB formatted-args buffer, v7sprintf is bounded
%define V7_FMTBUF_LAST 98303                   ; last writable byte offset in g_fmtbuf
%define AQ_SEP_AND_QUOTE 2                     ; separator space plus opening quote before the escaped body
%define EF_SHADOW 32                           ; shadow space before stack varargs
%define EF_QWORD 8                             ; bytes per vararg slot
%define EF_STKARG_SCALE 8                      ; SIB scale for stack-arg bytes, equals EF_QWORD
%define EF_ROUND_BIAS 15                       ; frame size round-up bias, folded into the size LEA's displacement
%define STARTF_USESTDHANDLES 0x00000100        ; STARTUPINFO.dwFlags: use hStdXxx handles
%define HANDLE_FLAG_INHERIT 0x00000001         ; SetHandleInformation: inherit flag mask
%define WAIT_INFINITE 0xFFFFFFFF               ; WaitForSingleObject: no timeout
%define FILE_END_MOVE 2                        ; SetFilePointerEx: move relative to EOF
; STARTUPINFOA field offsets, Win64
%define SI_CB 0                                ; DWORD cb
%define SI_DWFLAGS 60                          ; DWORD dwFlags
%define SI_HSTDOUTPUT 88                       ; HANDLE hStdOutput
%define SI_HSTDERROR 96                        ; HANDLE hStdError
; vmovdqu zero-fill offsets for STARTUPINFOA, exactly 104 bytes: six 16-byte fills cover bytes 0..95 and the hStdError store below completes bytes 96..103, nothing is written past the struct
%define SI_ZFILL1 16                           ; zero bytes 16..31
%define SI_ZFILL2 32                           ; zero bytes 32..47
%define SI_ZFILL3 48                           ; zero bytes 48..63
%define SI_ZFILL4 64                           ; zero bytes 64..79
%define SI_ZFILL5 80                           ; zero bytes 80..95
; PROCESS_INFORMATION field offsets
%define PI_HTHREAD 8                           ; HANDLE hThread
; SECURITY_ATTRIBUTES field offsets
%define SA_NLEN 0                              ; DWORD nLength
%define SA_LPSD 8                              ; LPVOID lpSecurityDescriptor
%define SA_BINHERIT 16                         ; BOOL bInheritHandle
; CreateFileA dwDesiredAccess / dwCreationDisposition constants
%define GENERIC_READ 0x80000000                ; read access
%define GENERIC_WRITE 0x40000000               ; write access
%define OPEN_EXISTING 3                        ; open only if file exists
%define CREATE_ALWAYS 2                        ; create new, truncate if exists
%define OPEN_ALWAYS 4                          ; open if exists, create if not
%define FILE_ATTRIBUTE_NORMAL 0x80             ; normal file attributes
; CreateProcessA manual-call stack layout, 10 args: 4 reg + 6 stack
%define CP_STK_SIZE 80                         ; 32 shadow + 48 stack args, 16-aligned
%define CP_STK_ARG5 32                         ; arg5 slot: bInheritHandles
%define CP_STK_ARG6 40                         ; arg6 slot: dwCreationFlags
%define CP_STK_ARG7 48                         ; arg7 slot: lpEnvironment
%define CP_STK_ARG8 56                         ; arg8 slot: lpCurrentDirectory
%define CP_STK_ARG9 64                         ; arg9 slot: lpStartupInfo
%define CP_STK_ARG10 72                        ; arg10 slot: lpProcessInformation
     section .text
     global v7_execute                         ; Export: create a subprocess and capture its stdout
v7_execute:
     SF_START v7_execute, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles, args + loop state; rbx named so ENGINE_GATE_SFV's fixed 7-push dead path pops exactly this list
     SF_COMMIT                                 ; reserve the 32-byte shadow space the six calls below need, without it the callees' arg-spill window [X+8, X+39] lands on the saved r12-r15 pushed above
     ENGINE_GATE_SFV 1                         ; dead-check + landing zone: spawn/capture failures must unwind to the host, not TerminateProcess it
     PUB_CPU_GATE                              ; refuse to run on hardware below the v7 baseline
; save incoming args, rcx/rdx/r8/r9 overwritten by helper calls
     mov r12, rcx                              ; r12 = exe path
     mov r13, rdx                              ; r13 = parameter1
     mov r14, r8                               ; r14 = parameter2
     mov r15, r9                               ; r15 = parameter3
     lea r8, [rel g_cmdline + V7_CMDLINE_LAST] ; r8 = append bound, survives helper calls
     lea rdi, [rel g_cmdline]                  ; rdi = write cursor for command line
     mov byte [rdi], '"'                       ; opening quote for exe path
     inc rdi                                   ; advance past quote
     mov rcx, r12                              ; rcx = exe path
     call exec_append_raw                      ; append exe path, inside quotes
     cmp rdi, r8                               ; buffer full?
     jae .exec_cmdline_overflow                ; yes: limit exceeded, FATAL, no room for closing quote
     mov byte [rdi], '"'                       ; closing quote for exe path
     inc rdi                                   ; advance past quote
     mov rcx, r13                              ; rcx = parameter1
     call .append_quoted_arg                   ; append ' "param1"' or skip if NULL
     mov rcx, r14                              ; rcx = parameter2
     call .append_quoted_arg                   ; append ' "param2"' or skip if NULL
     mov rcx, r15                              ; rcx = parameter3
     call .append_quoted_arg                   ; append ' "param3"' or skip if NULL
     mov byte [rdi], 0                         ; NUL-terminate command line
; ---- spawn g_cmdline, capture stdout into the shared buffer ----------
     call exec_spawn_capture                   ; rax = NUL-terminated capture buffer
     ENGINE_LEAVE                              ; park the landing zone, normal path
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; epilogue: restore non-volatiles, incl. the gate's rbx slot, return buffer ptr
.exec_cmdline_overflow:
     FATAL "v7_execute: command line exceeds the 32768-char CreateProcessA limit" ; capacity: silent truncation would send a wrong query to the child process
.append_quoted_arg:
     test rcx, rcx                             ; an omitted argument emits nothing
     jz .aq_ret                                ; leave the command-line cursor unchanged
     lea rax, [rdi + AQ_SEP_AND_QUOTE]         ; room needed for the separator and opening quote
     cmp rax, r8                               ; still need space for the closing quote
     jae .aq_overflow                          ; fail before building an oversized command
     mov word [rdi], 0x2220                    ; emit space and opening quote
     mov rdx, rax                              ; start of the escaped argument
     sub r8, rax                               ; capacity includes the helper's terminating NUL
     xor r11d, r11d                            ; closing quote follows with zero format backslashes
     sub rsp, 40                               ; helper entry follows a private CALL from the SF frame
     call win32_escape_inner                   ; apply the same CRT escaping as the formatted API
     add rsp, 40                               ; release the outgoing call frame
     test eax, eax                             ; negative means the escaped argument did not fit
     js .aq_overflow                           ; fail before process creation
     lea rdi, [rdx + rax]                      ; helper preserves rdx as the destination base
     mov byte [rdi], '"'                       ; replace the helper's NUL with the closing quote
     inc rdi                                   ; next argument or final command-line NUL
     lea r8, [rel g_cmdline + V7_CMDLINE_LAST] ; restore the fixed bound for subsequent arguments
.aq_ret:
     ret                                       ; return with the advanced command-line cursor
.aq_overflow:
     FATAL "v7_execute: command line exceeds the 32768-char CreateProcessA limit"
; ============================================================================
; exec_append_raw: append NUL-terminated string at rcx to [rdi], advance rdi
; Shared by v7_execute, exe path and v7_execute_fmt, formatted-arg section .
; r8 = append bound = address of the reserved NUL slot, last valid byte .
; An append may fill up TO the bound, cursor + len == r8 leaves the last data
; byte at r8-1 and the terminator lands exactly at r8 ; only a copy reaching
; PAST r8 is a capacity FATAL at the 32768-char CreateProcessA limit.
; Outputs: rdi = write cursor advanced past the copied body, the caller's only
; result ; rsi likewise advanced, but callers reload it
; Overwrites: rax, rcx, rsi, rdi, cursor advance, r8 preserved as the bound
; ============================================================================
exec_append_raw:
     test rcx, rcx                             ; NULL arg?
     jz .ear_ret                               ; yes: skip
     mov rsi, rcx                              ; rsi = source, rep movsb base; STR_LEN leaves rsi untouched
     push rdi                                  ; STR_LEN overwrites rdi, non-volatile for our callers
     STR_LEN rsi, ecx                          ; ecx = source length, 0 ok; overwrites rax/rcx/rdi
     pop rdi                                   ; restore dest cursor
     lea rax, [rdi + rcx]                      ; rax = cursor after copy
     cmp rax, r8                               ; would the copy extend past the reserved NUL slot?
     ja .ear_overflow                          ; yes: FATAL, exact fit at r8 is legal: data ends at r8-1, NUL goes to r8
     rep movsb                                 ; ERMS bulk copy, rcx = len; rsi/rdi advance past the body
.ear_ret:
     ret                                       ; return to caller
.ear_overflow:
     FATAL "v7_execute: command line exceeds the 32768-char CreateProcessA limit" ; capacity: silent truncation would send a wrong query to the child process
exec_spawn_capture:
     SF_START exec_spawn_capture, rbx, rdi, r12 ; prologue: save non-volatiles, rbx = capture buffer base, rdi = drain discard base held across the API_CALLs, r12 = bytes used
     SF_QWORD sf_hread                         ; HANDLE hRead, pipe read end, parent keeps
     SF_QWORD sf_hwrite                        ; HANDLE hWrite, pipe write end, child inherits
     SF_DWORD sf_bytesrd                       ; DWORD bytesRead from ReadFile
     SF_ARRAY sf_sa, SA_SIZE                   ; SECURITY_ATTRIBUTES, inheritable pipe
     SF_ARRAY sf_si, STARTUPINFOA_SIZE         ; STARTUPINFOA, std-handle wiring
     SF_ARRAY sf_pi, PROCESS_INFO_SIZE         ; PROCESS_INFORMATION, child handles
     SF_COMMIT                                 ; finalize frame layout, emit sub rsp
; ---- SECURITY_ATTRIBUTES for CreatePipe, inheritable handles --------
     lea rax, sf_sa                            ; rax = &sa, SF_ARRAY: lea, no brackets
     mov dword [rax+SA_NLEN], SA_SIZE          ; nLength = sizeof struct
     mov qword [rax+SA_LPSD], 0                ; lpSecurityDescriptor = default
     mov dword [rax+SA_BINHERIT], 1            ; bInheritHandle = TRUE
; ---- CreatePipe,&hRead, &hWrite, &sa, 0 ----------------------------
     lea rcx, sf_hread                         ; arg1: &hRead output slot
     lea rdx, sf_hwrite                        ; arg2: &hWrite output slot
     lea r8, sf_sa                             ; arg3: &sa, inheritable handles
     xor r9d, r9d                              ; arg4: nSize = 0, default buffer
     API_CALL [rel __imp_CreatePipe], rcx, rdx, r8, r9 ; create pipe pair
     test eax, eax                             ; TRUE = success?
     jz .fail_pipe                             ; no: return empty buffer
; ---- SetHandleInformation,hRead, INHERIT, 0, read end NOT inherited
     mov rcx, sf_hread                         ; hRead handle, stays in parent
     mov edx, HANDLE_FLAG_INHERIT              ; dwMask = INHERIT flag
     xor r8d, r8d                              ; dwFlags = 0, clear the bit
     API_CALL [rel __imp_SetHandleInformation], rcx, rdx, r8 ; strip INHERIT from read end, a just-created pipe handle is valid and mutable by the API contract
; ---- zero STARTUPINFOA, then fill the required fields -----------------
     lea rax, sf_si                            ; rax = &STARTUPINFOA, persistent store base
     vxorps xmm0, xmm0, xmm0                   ; zero vector, no memory read
     vmovdqu [rax], xmm0                       ; zero bytes 0..15
     vmovdqu [rax + SI_ZFILL1], xmm0           ; zero bytes 16..31
     vmovdqu [rax + SI_ZFILL2], xmm0           ; zero bytes 32..47
     vmovdqu [rax + SI_ZFILL3], xmm0           ; zero bytes 48..63
     vmovdqu [rax + SI_ZFILL4], xmm0           ; zero bytes 64..79
     vmovdqu [rax + SI_ZFILL5], xmm0           ; zero bytes 80..95, hStdInput at offset 80 stays NULL, the child gets no stdin; the hStdError store below completes bytes 96..103
     mov dword [rax+SI_CB], STARTUPINFOA_SIZE  ; cb = struct size for Windows
     mov dword [rax+SI_DWFLAGS], STARTF_USESTDHANDLES ; route std handles below
     mov rdx, sf_hwrite                        ; rdx = pipe write end handle, rax busy as base
     mov [rax+SI_HSTDOUTPUT], rdx              ; child stdout -> pipe
     mov [rax+SI_HSTDERROR], rdx               ; child stderr -> same pipe
     sub rsp, CP_STK_SIZE                      ; 32 shadow + 6 stack args, 16-aligned
     xor ecx, ecx                              ; arg1: lpApplicationName = NULL, cmdline has it
     lea rdx, [rel g_cmdline]                  ; arg2: lpCommandLine, built by caller
     xor r8d, r8d                              ; arg3: lpProcessAttributes = NULL
     xor r9d, r9d                              ; arg4: lpThreadAttributes = NULL
     mov dword [rsp+CP_STK_ARG5], 1            ; arg5: bInheritHandles = TRUE
     mov dword [rsp+CP_STK_ARG6], r9d          ; arg6: dwCreationFlags = 0
     mov qword [rsp+CP_STK_ARG7], r9           ; arg7: lpEnvironment = NULL, inherit parent's
     mov qword [rsp+CP_STK_ARG8], r9           ; arg8: lpCurrentDirectory = NULL, inherit
     lea rax, sf_si                            ; arg9: lpStartupInfo, rbp-based, shift-safe
     mov [rsp+CP_STK_ARG9], rax                ; store arg9
     lea rax, sf_pi                            ; arg10: lpProcessInformation, rbp-based
     mov [rsp+CP_STK_ARG10], rax               ; store arg10
     call [rel __imp_CreateProcessA]           ; nolint: 10 args exceed API_CALL max, direct import call
     add rsp, CP_STK_SIZE                      ; release shadow + stack args
     test eax, eax                             ; TRUE = child spawned?
     jz .fail_createprocess                    ; no: cleanup and return empty
     mov rcx, sf_hwrite                        ; hWrite handle
     API_CALL [rel __imp_CloseHandle], rcx     ; close parent's copy of write end
; ---- read loop: drain pipe into v7_execute_buf -----------------------
     lea rbx, [rel v7_execute_buf]             ; rbx = buffer base, non-volatile
     xor r12d, r12d                            ; r12 = bytes used so far
.read_loop:
     mov rcx, sf_hread                         ; hRead handle, rbp-based, safe
     lea rdx, [rbx+r12]                        ; dst = buf + bytes_used
     mov r8d, V7_EXEC_BUF_SIZE - 1             ; capacity minus NUL slot, 1 MiB - 1 = 1048575
     sub r8d, r12d                             ; remaining =, V7_EXEC_BUF_SIZE - 1 - bytes_used
     jle .read_drain                           ; buffer full, discard excess
     lea r9, sf_bytesrd                        ; &bytesRead output, rbp-based
     API_CALL [rel __imp_ReadFile], rcx, rdx, r8, r9, 0 ; read chunk from pipe
     test eax, eax                             ; success?
     jz .read_done                             ; FALSE = ERROR_BROKEN_PIPE, child done
     mov eax, sf_bytesrd                       ; eax = bytes actually read
     test eax, eax                             ; 0 bytes = EOF?
     jz .read_done                             ; yes: child closed pipe
     add r12d, eax                             ; advance bytes_used
     jmp .read_loop                            ; continue draining
.read_drain:
     lea rdi, [rel g_fmtbuf]                   ; rdi = discard base, NON-VOLATILE: survives API_CALL; saved by SF_START, restored by SF_END
.drain_loop:
     mov rcx, sf_hread                         ; hRead handle, rbp-based SF slot, volatile, must reload
     mov rdx, rdi                              ; dst = discard buffer, hoisted address
     mov r8d, V7_FMTBUF_SIZE                   ; up to 96KB per discard read, immediate, must reload: volatile
     lea r9, sf_bytesrd                        ; &bytesRead output, rbp-based
     API_CALL [rel __imp_ReadFile], rcx, rdx, r8, r9, 0 ; discard chunk
     test eax, eax                             ; success?
     jz .read_done                             ; no: pipe closed or error
     mov eax, sf_bytesrd                       ; eax = bytes discarded
     test eax, eax                             ; 0 = EOF?
     jnz .drain_loop                           ; more data: keep discarding
.read_done:
; ---- close read end of pipe ------------------------------------------
     mov rcx, sf_hread                         ; hRead handle
     API_CALL [rel __imp_CloseHandle], rcx     ; close read end
; ---- wait for child to exit, then reap its handles -------------------
     mov rcx, sf_pi                            ; hProcess, first qword of PROCESS_INFORMATION
     mov edx, WAIT_INFINITE                    ; wait forever
     API_CALL [rel __imp_WaitForSingleObject], rcx, rdx ; block until child exits
     mov rcx, sf_pi                            ; hProcess, first qword of PROCESS_INFORMATION
     API_CALL [rel __imp_CloseHandle], rcx     ; close process handle
     lea rcx, sf_pi                            ; rcx = &PROCESS_INFORMATION
     mov rcx, [rcx+PI_HTHREAD]                 ; hThread, second qword
     API_CALL [rel __imp_CloseHandle], rcx     ; close thread handle
; ---- NUL-terminate and return buffer pointer -------------------------
     lea rax, [rel v7_execute_buf]             ; rax = buffer base
     mov byte [rax+r12], 0                     ; NUL-terminate at bytes_used
     jmp .done                                 ; return buffer pointer
.fail_createprocess:
     mov rcx, sf_hwrite                        ; hWrite handle, cleanup
     API_CALL [rel __imp_CloseHandle], rcx     ; close write end
     mov rcx, sf_hread                         ; hRead handle, cleanup
     API_CALL [rel __imp_CloseHandle], rcx     ; close read end
.fail_pipe:
     lea rax, [rel v7_execute_buf]             ; rax = buffer base
     mov byte [rax], 0                         ; NUL-terminate empty buffer
.done:
     SF_END rbx, rdi, r12                      ; epilogue: restore non-volatiles, return buffer ptr
v7_execute_fmt:
     SF_START v7_execute_fmt, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles, rdi/rsi overwritten by exec_append_raw, rsi by exec_spawn_capture; rbx named so ENGINE_GATE_SFV's fixed 7-push dead path pops exactly this list
     SF_DWORD sf_ef_framesz                    ; vararg call-frame size, survives v7sprintf call: rax is volatile
     SF_COMMIT                                 ; finalize frame layout, emit sub rsp
     ENGINE_GATE_SFV 1                         ; dead-check + landing zone: overflow FATALs below unwind to the host instead of terminating it
     PUB_CPU_GATE                              ; refuse to run on hardware below the v7 baseline
     mov r12, rcx                              ; r12 = exe path
     mov r13, rdx                              ; r13 = format string
     mov r14, r8                               ; r14 = argv array
     mov r15d, r9d                             ; r15d = argc
     test r15d, r15d                           ; argc < 0?
     jge .ef_fmt_argc_ok                       ; no: proceed
     FATAL "v7_execute_fmt: argc is negative"  ; contract violation: frame size, jns -> 0 and copy bound, unsigned jae would disagree and walk off the stack
.ef_fmt_argc_ok:
; r12=exe, r13=fmt, r14=argv, r15d=argc stay live in non-volatiles
; no parks needed, only the frame SIZE must survive v7sprintf: rax dies
     mov eax, r15d                             ; eax = argc
     dec eax                                   ; stack varargs beyond r9?
     jns .ef_has_stack                         ; yes: compute bytes
     xor eax, eax                              ; no: zero stack args
.ef_has_stack:
     lea eax, [rax*EF_STKARG_SCALE + EF_SHADOW + EF_ROUND_BIAS] ; eax = stack-arg bytes + shadow, biased for round-up
     and eax, -16                              ; ...to a 16-byte multiple
     mov sf_ef_framesz, eax                    ; park frame size in an SF slot
     sub rsp, rax                              ; allocate vararg call frame, 16-aligned
; copy argv[1..argc-1] to their stack-arg slots, argv[0] is in r9
     lea rdx, [rsp + EF_SHADOW]                ; rdx = first stack-vararg slot
     mov ecx, 1                                ; ecx = first stack vararg index
.ef_copy_loop:
     cmp ecx, r15d                             ; past last arg?
     jae .ef_copy_done                         ; yes: registers next
     mov rax, [r14 + rcx*EF_QWORD]             ; rax = argv[ecx], r14 = argv base
     mov [rdx], rax                            ; store into the stack-arg slot
     add rdx, EF_QWORD                         ; advance dest cursor one qword
     inc ecx                                   ; next vararg
     jmp .ef_copy_loop                         ; continue copying
.ef_copy_done:
     lea rcx, [rel g_fmtbuf]                   ; arg1: output buffer
     mov edx, V7_FMTBUF_SIZE                   ; arg2: buffer size, zero-extended: 32-bit mov
     mov r8, r13                               ; arg3: format string, was parked in r13
; load r9 from argv only when argc proves the slot exists, r9 = vararg0
     test r15d, r15d                           ; argc > 0?
     jg .ef_has_vararg0                        ; yes: r9 safe to load
     xor r9d, r9d                              ; no vararg0, argc == 0
     jmp .ef_load_ok                           ; go call
.ef_has_vararg0:
     mov r9, [r14]                             ; vararg0 = argv[0]
.ef_load_ok:
     xor eax, eax                              ; AL = 0: no XMM args, Win64 varargs contract
     call v7sprintf                            ; format the raw argument section into g_fmtbuf
     cmp eax, V7_FMTBUF_LAST                   ; v7sprintf returns 98303 iff it hit the output bound, truncated ; empty, 0 and shorter outputs proceed
     jae .ef_fmt_overflow                      ; yes: capacity FATAL; empty, 0-arg spawn is legal and proceeds
     mov ecx, sf_ef_framesz                    ; ecx = parked frame size, 32-bit slot
     add rsp, rcx                              ; release vararg frame, ecx zero-extends; rax not needed
; ---- build g_cmdline: "exe" + ' ' + g_fmtbuf, RAW, no re-quoting ------
     lea r8, [rel g_cmdline + V7_CMDLINE_LAST] ; r8 = append bound
     lea rdi, [rel g_cmdline]                  ; rdi = write cursor
     mov byte [rdi], '"'                       ; opening quote for exe path
     inc rdi                                   ; advance past quote
     mov rcx, r12                              ; rcx = exe path
     call exec_append_raw                      ; append exe path, inside quotes
     cmp rdi, r8                               ; buffer full after exe?
     jae .ef_overflow                          ; yes: limit exceeded, FATAL
     mov byte [rdi], '"'                       ; closing quote for exe path
     inc rdi                                   ; advance past quote
     cmp rdi, r8                               ; buffer full after exe?
     jae .ef_overflow                          ; yes: limit exceeded, FATAL
     mov byte [rdi], ' '                       ; single space separator
     inc rdi                                   ; advance past separator
     lea rcx, [rel g_fmtbuf]                   ; rcx = formatted argument section
     call exec_append_raw                      ; append RAW, format string owns quoting
     mov byte [rdi], 0                         ; NUL-terminate command line
; ---- spawn + capture via the shared core ------------------------------
     call exec_spawn_capture                   ; rax = NUL-terminated capture buffer
     ENGINE_LEAVE                              ; park the landing zone, normal path
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; epilogue: restore non-volatiles, incl. the gate's rbx slot, return buffer ptr
.ef_overflow:
     FATAL "v7_execute_fmt: command line exceeds the 32768-char CreateProcessA limit" ; capacity: silent truncation would send wrong args to the child process
.ef_fmt_overflow:
     FATAL "v7_execute_fmt: formatted arguments exceed 96KB buffer" ; capacity: v7sprintf returned 98303, hit the output bound, truncated
     global v7s_readfile                       ; export: read whole file into shared buffer
v7s_readfile:
     SF_START v7s_readfile, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles, rbx/rsi/rdi live across API calls; rbx named so ENGINE_GATE_SFV's fixed 7-push dead path pops exactly this list
     SF_QWORD sf_rf_handle                     ; HANDLE hFile from CreateFileA
     SF_DWORD sf_rf_bytesread                  ; DWORD bytesRead per ReadFile
     SF_COMMIT                                 ; finalize frame layout, emit sub rsp
     ENGINE_GATE_SFV 1                         ; dead-check + landing zone: .rf_truncated / .rf_probe_error must unwind to the host with a dead flag, not TerminateProcess it
     PUB_CPU_GATE                              ; refuse to run on hardware below the v7 baseline
     mov rbx, rcx                              ; rbx = path, survives API calls; rcx still holds lpFileName for the call below
     API_CALL [rel __imp_CreateFileA], rcx, GENERIC_READ, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0 ; open for read
     cmp rax, -1                               ; INVALID_HANDLE_VALUE?
     je .rf_fail                               ; yes: return empty string
     mov sf_rf_handle, rax                     ; save file handle
; ---- read loop into v7_execute_buf -----------------------------------
     lea rbx, [rel v7_execute_buf]             ; rbx = shared output buffer
     xor r12d, r12d                            ; r12 = bytes buffered so far
.rf_read_loop:
     mov rcx, sf_rf_handle                     ; hFile handle
     lea rdx, [rbx+r12]                        ; lpBuffer = buf + bytes_used
     mov r8d, V7_EXEC_BUF_SIZE - 1             ; capacity minus NUL slot
     sub r8d, r12d                             ; remaining space
     jle .rf_probe                             ; buffer full, verify EOF before finishing, never silently truncate
     lea r9, sf_rf_bytesread                   ; &bytesRead, rbp-based
     API_CALL [rel __imp_ReadFile], rcx, rdx, r8, r9, 0 ; read one chunk
     test eax, eax                             ; success?
     jz .rf_read_error                         ; no: hard error, never return partial contents as success
     mov eax, sf_rf_bytesread                  ; eax = bytes actually read
     test eax, eax                             ; 0 = EOF?
     jz .rf_done                               ; yes: done
     add r12d, eax                             ; advance bytes_used
     jmp .rf_read_loop                         ; continue
.rf_probe:
     mov rcx, sf_rf_handle                     ; hFile handle
     lea rdx, [rbx + r12]                      ; dst = buf + bytes_used, the NUL slot
     mov r8d, 1                                ; one probe byte
     lea r9, sf_rf_bytesread                   ; &bytesRead
     API_CALL [rel __imp_ReadFile], rcx, rdx, r8, r9, 0 ; probe for extra content
     test eax, eax                             ; probe ReadFile failed?
     jz .rf_probe_error                        ; yes: I/O error, distinct from "too big"
     cmp sf_rf_bytesread, 0                    ; file continues past capacity?
     je .rf_done                               ; no: genuine EOF at exact capacity; a nonzero read falls into the FATAL
.rf_truncated:
     FATAL "v7s_readfile: file exceeds the 1 MiB buffer"
.rf_probe_error:
     FATAL "v7s_readfile: probe read failed at the 1 MiB buffer boundary, file size unknown" ; I/O error on a full buffer: content state unknown
.rf_read_error:
     mov rcx, sf_rf_handle                     ; hFile handle, still open on the error path
     API_CALL [rel __imp_CloseHandle], rcx     ; close before returning the empty result
     jmp .rf_fail                              ; empty string per the documented "Empty string on any error" contract
.rf_done:
; ---- close file and return -------------------------------------------
     mov rcx, sf_rf_handle                     ; hFile handle
     API_CALL [rel __imp_CloseHandle], rcx     ; close file
     lea rax, [rel v7_execute_buf]             ; rax = buffer base
     mov byte [rax+r12], 0                     ; NUL-terminate contents
     jmp .rf_ret                               ; return buffer pointer
.rf_fail:
     lea rax, [rel v7_execute_buf]             ; rax = buffer base
     mov byte [rax], 0                         ; empty string on failure
.rf_ret:
     ENGINE_LEAVE                              ; park the landing zone, normal path, every path converges here
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; epilogue: restore non-volatiles, 7, matching SF_START and the gate's dead path, return buffer ptr
     global v7s_writefile                      ; export: write/append string to file
v7s_writefile:
     SF_START v7s_writefile, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles, 7, matching the sibling readfile frame, ENGINE_GATE_SFV's fixed dead path pops exactly this list
     SF_QWORD sf_wf_handle                     ; HANDLE hFile from CreateFileA
     SF_COMMIT                                 ; finalize frame layout, emit sub rsp
     ENGINE_GATE_SFV 1                         ; dead-check + landing zone: sibling readfile treatment, a FATAL unwinds to the host, not TerminateProcess
     PUB_CPU_GATE                              ; refuse to run on hardware below the v7 baseline
     mov r12, rcx                              ; r12 = path, non-volatile
     mov r13, rdx                              ; r13 = text, non-volatile
     mov r14d, r8d                             ; r14d = append flag, non-volatile
; ---- compute string length, inline STR_LEN: no kernel transition ------
     STR_LEN r13, r15d                         ; r15d = text length, 0 is valid, creates/truncates to zero bytes; rdi saved by SF_START covers the macro overwrite
     mov rcx, r12                              ; arg1: lpFileName
     test r14d, r14d                           ; append mode?
     jnz .wf_open_append                       ; yes: OPEN_ALWAYS + seek end
     API_CALL [rel __imp_CreateFileA], rcx, GENERIC_WRITE, 0, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0 ; truncate/create
     jmp .wf_open_done                         ; go check handle
.wf_open_append:
     API_CALL [rel __imp_CreateFileA], rcx, GENERIC_WRITE, 0, 0, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0 ; keep existing content
.wf_open_done:
     cmp rax, -1                               ; INVALID_HANDLE_VALUE?
     je .wf_fail                               ; yes: return failure
     mov sf_wf_handle, rax                     ; save file handle
; ---- seek to end if appending -----------------------------------------
; SetFilePointerEx takes 4 args:, hFile, liDistanceToMove, &newpos, method
     test r14d, r14d                           ; append mode?
     jz .wf_write                              ; no: already at position 0
     mov rcx, sf_wf_handle                     ; arg1: hFile
     xor edx, edx                              ; arg2: liDistanceToMove = 0, 64-bit in rdx
     xor r8d, r8d                              ; arg3: lpNewFilePointer = NULL
     mov r9d, FILE_END_MOVE                    ; arg4: dwMoveMethod = FILE_END
     API_CALL [rel __imp_SetFilePointerEx], rcx, rdx, r8, r9 ; seek to EOF, rax = 1 success, 0 failure
     test eax, eax                             ; seek failed?
     jz .wf_write_fail                         ; yes: append position unknown, close handle, return 0
; ---- write text to file ----------------------------------------------
.wf_write:
     mov rcx, sf_wf_handle                     ; arg1: hFile
     mov rdx, r13                              ; arg2: lpBuffer = text
     mov r8d, r15d                             ; arg3: nNumberOfBytesToWrite = length
     xor r9d, r9d                              ; arg4 = NULL, bytes-written count is discarded
     API_CALL [rel __imp_WriteFile], rcx, rdx, r8, r9, 0 ; write text to file, a local synchronous file completes each write fully
; ---- close file and return --------------------------------------------
     mov rcx, sf_wf_handle                     ; hFile handle
     API_CALL [rel __imp_CloseHandle], rcx     ; flush + close
     mov eax, 1                                ; return 1, success
     jmp .wf_ret                               ; done
.wf_write_fail:
     mov rcx, sf_wf_handle                     ; hFile handle, seek-failure path
     API_CALL [rel __imp_CloseHandle], rcx     ; close before returning failure
     xor eax, eax                              ; return 0, failure
     jmp .wf_ret                               ; return to caller
.wf_fail:
     xor eax, eax                              ; return 0, failure
.wf_ret:
     ENGINE_LEAVE                              ; park the landing zone, normal path, every path converges here
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; epilogue: restore non-volatiles, 7, matching SF_START and the gate's dead path, return success flag
     section .bss
     align 16
v7_execute_buf:
     resb V7_EXEC_BUF_SIZE                     ; shared capture buffer for execute/readfile
g_cmdline:
     resb V7_CMDLINE_SIZE                      ; exe + up to 3 quoted args + NUL
g_fmtbuf:
     resb V7_FMTBUF_SIZE                       ; raw argument section, fmt owns quoting
     section .text
%undef CP_STK_ARG10
%undef CP_STK_ARG9
%undef CP_STK_ARG8
%undef CP_STK_ARG7
%undef CP_STK_ARG6
%undef CP_STK_ARG5
%undef CP_STK_SIZE
%undef EF_QWORD
%undef AQ_SEP_AND_QUOTE
%undef EF_SHADOW
%undef EF_STKARG_SCALE                         ; SIB scale for stack-arg bytes, file-local
%undef EF_ROUND_BIAS                           ; frame size round-up bias, file-local
%undef V7_FMTBUF_LAST
%undef V7_FMTBUF_SIZE
%undef V7_CMDLINE_LAST
%undef V7_CMDLINE_SIZE
%undef V7_EXEC_BUF_SIZE
%undef FILE_END_MOVE
%undef WAIT_INFINITE
%undef HANDLE_FLAG_INHERIT
%undef STARTF_USESTDHANDLES
%undef FILE_ATTRIBUTE_NORMAL
%undef OPEN_ALWAYS
%undef CREATE_ALWAYS
%undef OPEN_EXISTING
%undef GENERIC_WRITE
%undef GENERIC_READ
%undef SA_BINHERIT
%undef SA_LPSD
%undef SA_NLEN
%undef SA_SIZE
%undef PI_HTHREAD
%undef SI_ZFILL5
%undef SI_ZFILL4
%undef SI_ZFILL3
%undef SI_ZFILL2
%undef SI_ZFILL1
%undef SI_HSTDERROR
%undef SI_HSTDOUTPUT
%undef SI_DWFLAGS
%undef SI_CB
%undef PROCESS_INFO_SIZE
%undef STARTUPINFOA_SIZE
