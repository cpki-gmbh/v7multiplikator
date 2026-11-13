; v7 - part of the v7 inference engine
; Copyright (C) 2026 Görg Pflug & CPKI GmbH
; License: GPL v3 (see LICENSE)

; ============================================================================
; dbg_inference.asm -- Standalone ASM harness for v7 inference.
; Loads v7_threading.dll -> v7_init -> v7_initialize_inference -> v7_chat ->
; v7_shutdown -> v7_free_inference (public v7_* API tier, context hidden in BSS)
;
; Usage: dbg_inference.exe <dll_path> <db_path> ["prompt"] [-q8]
;   dll_path -- path to v7_threading.dll (MANDATORY, no default -- Sec.106)
;   db_path  -- path to lz4ht.db (MANDATORY, no default -- Sec.106)
;   "prompt" -- text prompt, quote it when it contains spaces (default: capital of France)
;   -q8      -- Q8 quantized forward pass (default: F32); unquoted standalone
;               token -- a quoted "-q8" is prompt DATA, not the flag
; Argument parsing implements the MSVC C-runtime post-2008 rules (whitespace
; delimiters, quote toggling, 2n/2n+1 backslash-before-quote, "" literal),
; so the harness sees exactly the argv a C program compiled with MSVC sees.
;
; Build:
;   nasm -f win64 -O0 -w-other dbg_inference.asm -o dbg_inference.obj
;   link /entry:WinMain /subsystem:console /nodefaultlib dbg_inference.obj kernel32.lib
; ============================================================================

     %include "v7_core.inc"
     %include "v7_apicall.inc"
     %include "v7_macros.inc"
     %include "fork.inc"                       ; fork_param layout, needed before v7_data.inc's fork_params BSS
     %include "v7_debug.inc"
     %include "v7_data.inc"

; Stack and buffer allocation constants
     %define OUT_BUF_SIZE       32768       ; output buffer size (32KB)
     %define PROBE_PAGE         4096       ; Windows page size for stack probe
     %define DLL_BUF_SIZE       260        ; DLL path buffer (MAX_PATH room)
     %define DB_BUF_SIZE        260        ; DB path buffer (MAX_PATH room)
     %define PROMPT_BUF_SIZE    1024       ; prompt buffer
     %define FLAG_BUF_SIZE      64         ; flag token buffer
     %define PEEK_NEXT          1          ; offset of the byte after the cursor
                                        ;
       MXCSR_RESET  equ 0x1F80           ; default MXCSR: all exceptions masked, round-to-nearest

; ============================================================================
; BSS globals
; ============================================================================
section .bss align=64
    dll_handle:       resq 1          ; v7_threading.dll module handle
    fn_v7_init:       resq 1          ; v7_init function pointer
    fn_v7_initialize: resq 1          ; v7_initialize_inference function pointer
    fn_v7_free:       resq 1          ; v7_free_inference function pointer
    fn_v7_chat:       resq 1          ; v7_chat function pointer
    fn_set_quantize:  resq 1          ; v7_set_quantize function pointer
    fn_set_max_gen:   resq 1          ; v7_set_max_gen function pointer
    fn_set_temp:      resq 1          ; v7_set_sampler_temperature function pointer
    fn_set_topp:      resq 1          ; v7_set_sampler_topp function pointer
    fn_set_seed:      resq 1          ; v7_set_sampler_seed function pointer
    fn_shutdown:      resq 1          ; v7_shutdown function pointer
    fn_is_dead:       resq 1          ; v7_engine_is_dead function pointer
    fn_get_reason:    resq 1          ; v7_get_death_reason function pointer
    dbg_bytes_written: resq 1       ; WriteFile bytes_written output (safe BSS, not below RSP)
    arg_dll_path:     resb DLL_BUF_SIZE  ; parsed DLL path from command line
    arg_db_path:      resb DB_BUF_SIZE   ; parsed DB path from command line
    arg_prompt:       resb PROMPT_BUF_SIZE ; parsed prompt from command line
    arg_flag:         resb FLAG_BUF_SIZE  ; parsed flag token from command line
    q8_mode:          resd 1         ; 0 = F32 forward (default), 1 = Q8 forward
    arg_overflow:     resd 1         ; set when any argument exceeded its buffer

; ============================================================================
; .rdata -- Read-only data
; ============================================================================
section .rdata align=8
mxcsr_reset_val:    dd MXCSR_RESET          ; default MXCSR: all exceptions masked, round-to-nearest
sz_default_prompt:  db "What is the capital of France?", 0  ; default prompt when no arg given
sz_usage_msg:       db "usage: dbg_inference.exe <dll_path> <db_path> [prompt] [-q8]", 10, 0       ; mandatory-path usage (Sec.106: no defaults)
sz_flag_q8:         db "-q8", 0                          ; Q8 flag token text to match
sz_notice_ignore:   db "[notice] ignoring unknown argument", 10, 0
sz_mode_f32:        db "[mode] F32 forward", 10, 0
sz_mode_q8:         db "[mode] Q8 forward", 10, 0
sz_err_arg_long:    db "[FATAL] command line argument exceeds buffer", 10, 0
sz_fn_v7_init:       db "v7_init", 0                     ; function name for GetProcAddress
sz_fn_v7_initialize: db "v7_initialize_inference", 0     ; function name for GetProcAddress
sz_fn_v7_free:       db "v7_free_inference", 0           ; function name for GetProcAddress
sz_fn_v7_chat:       db "v7_chat", 0                     ; function name for GetProcAddress
sz_fn_set_quantize:  db "v7_set_quantize", 0             ; function name for GetProcAddress
sz_fn_set_max_gen:   db "v7_set_max_gen", 0              ; function name for GetProcAddress
sz_fn_set_temp:      db "v7_set_sampler_temperature", 0  ; function name for GetProcAddress
sz_fn_set_topp:      db "v7_set_sampler_topp", 0         ; function name for GetProcAddress
sz_fn_set_seed:      db "v7_set_sampler_seed", 0         ; function name for GetProcAddress
sz_fn_shutdown:      db "v7_shutdown", 0                 ; function name for GetProcAddress
sz_fn_is_dead:       db "v7_engine_is_dead", 0          ; function name for GetProcAddress
sz_fn_get_reason:    db "v7_get_death_reason", 0        ; function name for GetProcAddress
sz_err_no_dll:      db "[FATAL] LoadLibraryA failed", 10, 0       ; error message: DLL not found
sz_err_no_proc:     db "[FATAL] GetProcAddress failed for ", 0       ; error message: proc not found
sz_err_no_db:       db "[FATAL] v7_init failed", 10, 0       ; error: v7_init returned 0
sz_err_no_create:   db "[FATAL] v7_initialize_inference failed", 10, 0   ; error: context init returned 0
sz_newline:         db 10, 0                            ; bare newline string
sz_trace_load:      db "[trace] Loading DLL", 10, 0
sz_trace_procs:     db "[trace] Resolving procs", 10, 0
sz_trace_init:      db "[trace] Calling v7_init", 10, 0
sz_trace_create:    db "[trace] Calling v7_initialize_inference", 10, 0
sz_trace_sampler:   db "[trace] Setting sampler params", 10, 0
sz_trace_chat:      db "[trace] Calling v7_chat", 10, 0
sz_trace_done:      db "[trace] v7_chat returned", 10, 0
sz_trace_died:      db "[fatal] engine died: ", 0       ; death banner (reason text follows)
sz_trace_shutdown:  db "[trace] Calling v7_shutdown", 10, 0
sz_trace_destroy:   db "[trace] Calling v7_free_inference", 10, 0
float_zero:         dd 0.0                             ; 0.0f for greedy temperature
float_topp:         dd 0.9                              ; 0.9f topp default (IEEE 754)

; ============================================================================
; .text -- Code
; ============================================================================
section .text

; ============================================================================
; copy_cstr -- copy a NUL-terminated string from [rsi] to [rdi]
;   Both pointers end one past the copied NUL. No capacity check: only used
;   for short default constants that provably fit their buffers.
; ============================================================================
copy_cstr:
        cld                                    ; force forward copies (movsb reads DF; never trust callees)
.cc_loop:
        movsb                                ; copy one byte, advance both pointers
        cmp byte [rdi - 1], 0               ; copied the NUL terminator?
        jne .cc_loop                         ; not yet: keep copying
        ret

; PA_EMIT byte -- store one byte into [rdi] with capacity check
;   edx counts down the remaining headroom; when exhausted the byte is
;   dropped and the arg_overflow flag records the condition (parsed once
;   after all arguments, then FATALs -- silent truncation is forbidden).
;   The check keeps >= 1 byte of headroom so the terminating NUL always
;   fits inside the buffer (never written one past the end).
%macro PA_EMIT 1
        cmp edx, 1                            ; headroom must keep 1 byte for the NUL
        jbe %%ovf                             ; none: drop the byte, flag overflow
        mov byte [rdi], %1                   ; store the byte
        inc rdi                               ; advance the destination
        dec edx                               ; consume one byte of headroom
        jmp %%done
%%ovf:
        mov dword [rel arg_overflow], 1      ; record: argument exceeded buffer
%%done:
%endmacro

; ============================================================================
; parse_next_arg -- parse ONE command-line argument into a buffer
;   rsi = cursor into the command line (IN/OUT: advanced past the argument)
;   rdi = destination buffer (OUT: NUL-terminated argument)
;   edx = destination capacity in bytes (including room for the NUL)
;   returns eax = 1 argument parsed, eax = 0 end of line (no argument)
;   sets arg_overflow (BSS) when the argument exceeded the capacity
; Implements the MSVC C-runtime post-2008 rules:
;   - space/tab delimit arguments (only outside a quoted block)
;   - " toggles a quoted block; quotes may appear anywhere inside an argument
;   - 2n backslashes + " emit n backslashes and the quote TOGGLES the block
;   - 2n+1 backslashes + " emit n backslashes and a LITERAL " (state unchanged)
;   - a backslash run not followed by " is entirely literal
;   - "" inside a quoted block emits ONE literal " and the block continues
; Register contract: rsi/rdi advance (like movsb); r8d/ecx/r9d/al are scratch.
; ============================================================================
parse_next_arg:
        ; --- skip whitespace before the argument ---
.pna_ws:
        cmp byte [rsi], 32                   ; space?
        je .pna_ws_adv
        cmp byte [rsi], 9                    ; tab?
        je .pna_ws_adv
        jmp .pna_ws_done
.pna_ws_adv:
        inc rsi                              ; step over the whitespace byte
        jmp .pna_ws
.pna_ws_done:
        cmp byte [rsi], 0                    ; end of the command line?
        je .pna_absent                       ; yes: no argument present
        xor r8d, r8d                          ; r8d = inside-quoted-block flag (start outside)
        ; --- main copy loop: one byte per iteration ---
.pna_loop:
        mov al, [rsi]                        ; al = current byte
        test al, al                          ; NUL terminator?
        jz .pna_finish                       ; yes: argument complete
        cmp al, 92                           ; backslash?
        je .pna_bs
        cmp al, 34                           ; double quote?
        je .pna_quote
        test r8d, r8d                        ; inside a quoted block?
        jnz .pna_store                       ; yes: whitespace is data, store raw
        cmp al, 32                           ; space delimiter?
        je .pna_finish
        cmp al, 9                            ; tab delimiter?
        je .pna_finish
.pna_store:
        PA_EMIT al                           ; store the byte (capacity-checked)
        inc rsi                              ; advance the cursor
        jmp .pna_loop
        ; --- backslash run: literal unless immediately followed by " ---
.pna_bs:
        xor ecx, ecx                          ; ecx = backslash run length
.pna_bs_count:
        cmp byte [rsi], 92                   ; another backslash?
        jne .pna_bs_end
        inc ecx
        inc rsi
        jmp .pna_bs_count
.pna_bs_end:
        cmp byte [rsi], 34                   ; run terminated by a quote?
        jne .pna_bs_plain
        ; --- 2n / 2n+1 backslashes followed by a quote ---
        mov r9d, ecx
        shr r9d, 1                            ; r9d = floor(m/2) backslashes to emit
.pna_bs_half:
        test r9d, r9d
        jz .pna_bs_half_done
        PA_EMIT 92                            ; emit one of the m/2 backslashes
        dec r9d
        jmp .pna_bs_half
.pna_bs_half_done:
        test ecx, 1                           ; odd run? (escaped-quote form)
        jz .pna_bs_toggle
        PA_EMIT 34                            ; odd: emit a literal " (state unchanged)
        inc rsi                               ; consume the quote byte
        jmp .pna_loop
.pna_bs_toggle:
        inc rsi                               ; consume the quote byte
        xor r8d, 1                            ; even: the quote toggles the block
        jmp .pna_loop
.pna_bs_plain:
        ; --- run NOT followed by a quote: all m backslashes are literal ---
        mov r9d, ecx                          ; r9d = m literal backslashes
.pna_bs_all:
        test r9d, r9d
        jz .pna_loop
        PA_EMIT 92                            ; emit one literal backslash
        dec r9d
        jmp .pna_bs_all
        ; --- bare double quote (no backslash run in front) ---
.pna_quote:
        test r8d, r8d                         ; inside a quoted block?
        jz .pna_q_open
        cmp byte [rsi + PEEK_NEXT], 34       ; "" inside quotes?
        jne .pna_q_close
        PA_EMIT 34                            ; "" collapses to ONE literal "
        add rsi, 2                            ; consume BOTH quote bytes
        jmp .pna_loop
.pna_q_close:
        xor r8d, 1                            ; leave the quoted block
        inc rsi                               ; consume the closing quote
        jmp .pna_loop
.pna_q_open:
        mov r8d, 1                            ; enter the quoted block
        inc rsi                               ; consume the opening quote
        jmp .pna_loop
.pna_absent:
        xor eax, eax                          ; eax = 0: no argument
        ret
.pna_finish:
        test edx, edx                         ; zero capacity passed? (PA_EMIT always keeps edx >= 1)
        jz .pna_finish_ovf                    ; yes: cap==0 guard -- even the NUL does not fit
        mov byte [rdi], 0                     ; NUL-terminate the argument
        mov eax, 1                            ; eax = 1: argument present
        ret
.pna_finish_ovf:
        mov dword [rel arg_overflow], 1      ; record: argument exceeded buffer
        xor eax, eax                          ; eax = 0: deterministic return (overflow FATALs later)
        ret

; GETARG_REQUIRED buf, cap -- parse the next CLI argument into buf; when
; absent (end of line), print the usage message and exit non-zero (Sec.106:
; mandatory path arguments, no default, no fallback).
%macro GETARG_REQUIRED 2
        lea rdi, [rel %1]                    ; rdi = destination buffer
        mov edx, %2                           ; edx = buffer capacity
        call parse_next_arg                   ; eax = 1 parsed / 0 absent
        test eax, eax
        jnz %%done                            ; argument present: keep it
        jmp wm_usage_fail                     ; absent: usage + non-zero exit (Sec.106)
%%done:
%endmacro

; GETARG_OR_DEFAULT buf, default_str, cap -- parse the next CLI argument into
; buf; when absent (end of line), copy the default string into buf instead.
; The cmdline cursor (rsi) is SAVED across the default copy -- rsi is both the
; parse cursor and copy_cstr's source register, so without the push/pop the
; cursor would be left pointing into .rdata and later parse_next_arg calls
; would "parse" the adjacent default strings as real command-line arguments.
%macro GETARG_OR_DEFAULT 3
        lea rdi, [rel %1]                    ; rdi = destination buffer
        mov edx, %3                           ; edx = buffer capacity
        call parse_next_arg                   ; eax = 1 parsed / 0 absent
        test eax, eax
        jnz %%done                            ; argument present: keep it
        push rsi                              ; save the cmdline cursor (rsi = copy source next)
        lea rsi, [rel %2]                    ; rsi = default string constant
        lea rdi, [rel %1]                    ; rdi = destination buffer
        call copy_cstr                        ; copy default (incl. NUL)
        pop rsi                               ; restore the cmdline cursor
%%done:
%endmacro

; ============================================================================
; WinMain -- Entry point
; RCX = hInstance, RDX = hPrevInstance, R8 = lpCmdLine, R9 = nCmdShow
; ============================================================================
      SF_START WinMain, rbx, rsi, rdi, r12, r13, r14, r15       ; begin WinMain stack frame
     SF_COMMIT                                    ; sub rsp, N (auto-aligned + 32 shadow)

    ; ========================================================================
    ; Phase 0: Initialize CPU state (no CRT -- must do this manually)
    ; ========================================================================
          finit                               ; reset x87 FPU: clear exceptions, mask all, 64-bit precision
          ldmxcsr [rel mxcsr_reset_val]       ; load MXCSR = 0x1F80 (all exceptions masked, round-to-nearest)
          vpxor ymm0, ymm0, ymm0             ; zero YMM0 to avoid dirty-upper-state AVX stall
          vpxor ymm1, ymm1, ymm1             ; zero YMM1
          vpxor ymm2, ymm2, ymm2             ; zero YMM2
          vpxor ymm3, ymm3, ymm3             ; zero YMM3
          vpxor ymm4, ymm4, ymm4             ; zero YMM4
          vpxor ymm5, ymm5, ymm5             ; zero YMM5
          vpxor ymm6, ymm6, ymm6             ; zero YMM6
          vpxor ymm7, ymm7, ymm7             ; zero YMM7
          vpxor ymm8, ymm8, ymm8             ; zero YMM8
          vpxor ymm9, ymm9, ymm9             ; zero YMM9
          vpxor ymm10, ymm10, ymm10          ; zero YMM10
          vpxor ymm11, ymm11, ymm11          ; zero YMM11
          vpxor ymm12, ymm12, ymm12          ; zero YMM12
          vpxor ymm13, ymm13, ymm13          ; zero YMM13
          vpxor ymm14, ymm14, ymm14          ; zero YMM14
          vpxor ymm15, ymm15, ymm15          ; zero YMM15

    ; ========================================================================
    ; Phase 1: Parse command line arguments. R8 at a RAW /entry:WinMain is
    ; NOT lpCmdLine -- the WinMain signature is a CRT-invented convention;
    ; WinMainCRTStartup builds lpCmdLine from GetCommandLineA + argv[0] skip.
    ; We do the same (codebase precedent: the DLL's own DllMain Phase 1). stdout_handle
    ; is NOT needed yet: WRITE_STR lazily initializes it via debug_ensure_std.
    ;   arg1 = DLL path (MANDATORY, no default -- Sec.106)
    ;   arg2 = DB path (MANDATORY, no default -- Sec.106)
    ;   arg3 = prompt (default: "What is the capital of France?")
    ;   arg4+ = flags (-q8 selects the Q8 forward pass; unknown: notice)
    ; ========================================================================
          API_CALL [__imp_GetCommandLineA]      ; rax = full ANSI command line (incl. argv[0])
          mov rsi, rax                          ; rsi = command line cursor (never NULL)
          ; --- skip argv[0]: the program path, possibly quoted ---
          cmp byte [rsi], 34                    ; quoted program path?
          je .wm_skip_quoted
.wm_skip_bare:                                  ; bare path: scan to whitespace/NUL
          cmp byte [rsi], 0                     ; end of string?
          je .wm_cmdline_ok                     ; no arguments at all
          cmp byte [rsi], 32                    ; space?
          je .wm_skip_ws
          cmp byte [rsi], 9                     ; tab?
          je .wm_skip_ws
          inc rsi                               ; advance within the program path
          jmp .wm_skip_bare
.wm_skip_quoted:                                ; quoted path: scan to closing quote
          inc rsi                               ; step over the opening quote
.wm_sq_loop:
          cmp byte [rsi], 0                     ; unterminated quote?
          je .wm_cmdline_ok                     ; accept -- no arguments
          cmp byte [rsi], 34                    ; closing quote?
          je .wm_sq_end
          inc rsi                               ; advance within the quoted path
          jmp .wm_sq_loop
.wm_sq_end:
          inc rsi                               ; step over the closing quote
.wm_skip_ws:                                    ; skip whitespace before the first argument
          cmp byte [rsi], 0                     ; end of command line?
          je .wm_cmdline_ok                     ; no arguments
          cmp byte [rsi], 32                    ; space?
          je .wm_ws_adv
          cmp byte [rsi], 9                     ; tab?
          je .wm_ws_adv
          cmp byte [rsi], 13                    ; CR?
          je .wm_ws_adv
          cmp byte [rsi], 10                    ; LF?
          je .wm_ws_adv
          jmp .wm_cmdline_ok                    ; first real argument starts here
.wm_ws_adv:
          inc rsi                               ; skip the whitespace byte
          jmp .wm_skip_ws
.wm_cmdline_ok:
          GETARG_REQUIRED arg_dll_path, DLL_BUF_SIZE
          GETARG_REQUIRED arg_db_path, DB_BUF_SIZE
          ; --- peek: an unquoted "-q8" token in prompt position is a FLAG ---
          ; (the prompt GETARG below would otherwise consume it as text; a
          ; quoted "-q8" starts with a quote byte and correctly stays data)
          mov rdx, rsi                          ; rdx = peek cursor (rsi stays authoritative)
.wm_peek_ws:
          cmp byte [rdx], 32                    ; space?
          je .wm_peek_adv
          cmp byte [rdx], 9                     ; tab?
          je .wm_peek_adv
          jmp .wm_peek_tok                      ; token start (or end of line)
.wm_peek_adv:
          inc rdx                               ; skip the whitespace byte
          jmp .wm_peek_ws
.wm_peek_tok:
          cmp byte [rdx], 0                     ; end of line? no flag, no prompt
          je .wm_prompt_normal
          cmp byte [rdx], '-'                   ; dash starts the flag?
          jne .wm_prompt_normal
          inc rdx
          cmp byte [rdx], 'q'                   ; 'q'?
          jne .wm_prompt_normal
          inc rdx
          cmp byte [rdx], '8'                   ; '8'?
          jne .wm_prompt_normal
          inc rdx
          movzx eax, byte [rdx]                 ; delimiter after "-q8"?
          test eax, eax                         ; NUL (end of command line)?
          je .wm_prompt_is_flag
          cmp eax, 32                           ; space?
          je .wm_prompt_is_flag
          cmp eax, 9                            ; tab?
          je .wm_prompt_is_flag
          jmp .wm_prompt_normal                 ; "-q8xyz" glued text: prompt data
.wm_prompt_is_flag:
          ; next token is the -q8 flag: keep the prompt at its default and let
          ; the flag scan below consume the token (cursor rsi is untouched)
          push rsi                              ; save cmdline cursor (rsi = copy source next)
          lea rsi, [rel sz_default_prompt]      ; rsi = default prompt
          lea rdi, [rel arg_prompt]             ; rdi = prompt buffer
          call copy_cstr                        ; copy default (incl. NUL)
          pop rsi                               ; restore cmdline cursor
          jmp .wm_scan                          ; flag scan takes it from here
.wm_prompt_normal:
          GETARG_OR_DEFAULT arg_prompt, sz_default_prompt, PROMPT_BUF_SIZE

          mov dword [rel q8_mode], 0          ; default forward mode: F32
          ; --- scan the remaining arguments for flags ---
.wm_scan:
          lea rdi, [rel arg_flag]             ; rdi = flag token buffer
          mov edx, FLAG_BUF_SIZE              ; edx = token capacity
          call parse_next_arg                  ; eax = 1 token / 0 done
          test eax, eax
          jz .wm_scan_done                     ; no more arguments
          push rsi                             ; save cmdline cursor (rsi = compare scratch next)
          lea rsi, [rel sz_flag_q8]            ; rsi = "-q8" reference text
          lea rdi, [rel arg_flag]              ; rdi = parsed token text
.wm_cmp:
          mov cl, [rsi]                        ; cl = reference byte
          mov dl, [rdi]                        ; dl = token byte
          cmp cl, dl                           ; bytes differ?
          jne .wm_cmp_ne                       ; yes: not the -q8 flag
          test cl, cl                          ; both at the NUL terminator?
          jz .wm_cmp_match                     ; full match
          inc rsi                              ; advance reference cursor
          inc rdi                              ; advance token cursor
          jmp .wm_cmp
.wm_cmp_ne:
          pop rsi                              ; restore cmdline cursor
          WRITE_STR sz_notice_ignore           ; visible notice (Sec.33: never silent)
          jmp .wm_scan
.wm_cmp_match:
          pop rsi                              ; restore cmdline cursor
          mov dword [rel q8_mode], 1           ; Q8 quantized forward selected
          jmp .wm_scan                         ; keep scanning further flags
.wm_scan_done:
          cmp dword [rel arg_overflow], 0      ; any argument overran its buffer?
          je .wm_args_ok
          WRITE_STR sz_err_arg_long            ; visible error, then die
          mov ecx, 1                           ; exit code = 1
          API_CALL [__imp_ExitProcess], rcx    ; terminate -- never returns
.wm_args_ok:
          ; --- mode banner: the smoke log must show which forward ran ---          cmp dword [rel q8_mode], 0
          jne .wm_mode_q8
          WRITE_STR sz_mode_f32
          jmp .wm_phase2
.wm_mode_q8:
          WRITE_STR sz_mode_q8
.wm_phase2:

    ; ========================================================================
    ; Phase 2: Load v7_threading.dll (path from arg1 or default)
    ; ========================================================================
          WRITE_STR sz_trace_load
          lea rcx, [rel arg_dll_path]          ; arg1: DLL path (parsed or default)
          API_CALL [__imp_LoadLibraryA], rcx       ; rax = HMODULE (or NULL)

          test rax, rax                           ; load succeeded?
          jnz .dll_loaded                         ; yes, continue
          WRITE_STR sz_err_no_dll                  ; print fatal: DLL load failed
          mov ecx, 1                              ; exit code = 1
          API_CALL [__imp_ExitProcess], rcx        ; terminate -- never returns

.dll_loaded:
          mov [rel dll_handle], rax                 ; save DLL handle

    ; ========================================================================
    ; Phase 3: Get function pointers via GetProcAddress (public v7_* tier)
    ; RESOLVE_FN <name_label>, <fn_slot> -- resolve one export or die
    ; ========================================================================
          WRITE_STR sz_trace_procs

%macro RESOLVE_FN 2
          mov rcx, [rel dll_handle]                ; arg1: module handle
          lea rdx, [rel %1]                      ; arg2: export name string
          API_CALL [__imp_GetProcAddress], rcx, rdx       ; rax = function pointer (or NULL)
          test rax, rax                           ; found?
          jnz %%got                               ; yes, save it
          WRITE_STR sz_err_no_proc                ; print: GetProcAddress failed for
          WRITE_STR %1                            ; print: export name
          WRITE_STR sz_newline                    ; newline
          jmp .do_unload                          ; free DLL then exit
%%got:
          mov [rel %2], rax                       ; save function pointer
%endmacro

          RESOLVE_FN sz_fn_v7_init, fn_v7_init
          RESOLVE_FN sz_fn_v7_initialize, fn_v7_initialize
          RESOLVE_FN sz_fn_v7_free, fn_v7_free
          RESOLVE_FN sz_fn_v7_chat, fn_v7_chat
          RESOLVE_FN sz_fn_set_quantize, fn_set_quantize
          RESOLVE_FN sz_fn_set_max_gen, fn_set_max_gen
          RESOLVE_FN sz_fn_set_temp, fn_set_temp
          RESOLVE_FN sz_fn_set_topp, fn_set_topp
          RESOLVE_FN sz_fn_set_seed, fn_set_seed
          RESOLVE_FN sz_fn_shutdown, fn_shutdown
          RESOLVE_FN sz_fn_is_dead, fn_is_dead
          RESOLVE_FN sz_fn_get_reason, fn_get_reason

    ; ========================================================================
    ; Phase 3.5: Initialize database via v7_init
    ; API_CALL loads the fn ptr from a MEMORY operand after the arg saves --
    ; passing the fn ptr in a register is WRONG (arg saves clobber rax first).
    ; ========================================================================
          WRITE_STR sz_trace_init
          lea rcx, [rel arg_db_path]              ; arg1: parsed DB path (or default)
          API_CALL [rel fn_v7_init], rcx           ; v7_init(db_path) -> rax = int
          test eax, eax                           ; init succeeded? (nonzero = success)
          jnz .db_init_ok                         ; yes, continue
          WRITE_STR sz_err_no_db                  ; print fatal: v7_init failed
          jmp .do_unload                          ; free DLL then exit

.db_init_ok:

    ; ========================================================================
    ; Phase 4: Initialize inference (context hidden in BSS inside the DLL)
    ; ========================================================================
          WRITE_STR sz_trace_create
          API_CALL [rel fn_v7_initialize]         ; v7_initialize_inference() -> rax = int
          test eax, eax                           ; context ready? (0 = failure; 1 = ready)
          jnz .create_ok                          ; yes, continue
          WRITE_STR sz_err_no_create              ; print fatal: inference init failed
          jmp .do_unload                          ; free DLL then exit
.create_ok:

     ; Set sampler + engine parameters via the public setter tier
          WRITE_STR sz_trace_sampler
          movss xmm0, [rel float_zero]             ; temperature = 0.0f (greedy)
          API_CALL [rel fn_set_temp]               ; v7_set_sampler_temperature(0.0f) -- float arg travels in xmm0
          movss xmm0, [rel float_topp]             ; topp = 0.9f
          API_CALL [rel fn_set_topp]               ; v7_set_sampler_topp(0.9f) -- float arg travels in xmm0
          xor ecx, ecx                             ; seed = 0
          API_CALL [rel fn_set_seed], rcx          ; v7_set_sampler_seed(0)
          mov ecx, [rel q8_mode]                   ; quantize mode from the parsed -q8 flag (0=F32, 1=Q8)
          API_CALL [rel fn_set_quantize], rcx      ; v7_set_quantize(mode)
          mov ecx, 200                             ; max generation tokens
          API_CALL [rel fn_set_max_gen], rcx       ; v7_set_max_gen(200)

    ; ========================================================================
    ; Phase 5: Call v7_chat -- allocate 32KB output buffer on stack with probing
    ; v7_chat(char* user_prompt, char* system_prompt, char* output,
    ;         int output_len, int reset)
    ; ========================================================================
          mov r15, rsp                             ; r15 = original RSP (non-volatile, for later restoration)
    ; Stack probe: touch pages BEFORE sub rsp to trigger guard page commitment
          mov rsi, rsp                             ; rsi = current RSP for probe loop
          sub rsi, OUT_BUF_SIZE                    ; rsi = target RSP (bottom of 32KB area)
          lea rdi, [rsp - PROBE_PAGE]              ; rdi = first page to touch (one page below current RSP)
.probe_loop:
          mov dword [rdi], 0                       ; touch page (triggers guard -> commit)
          sub rdi, PROBE_PAGE                      ; advance to next page down
          cmp rdi, rsi                             ; below target RSP?
          jb .probe_done                           ; yes, all pages touched
          jmp .probe_loop                         ; continue probing
.probe_done:
          sub rsp, OUT_BUF_SIZE                    ; allocate 32KB output buffer on stack
          mov r14, rsp                             ; r14 = output buffer pointer (non-volatile)
          ; Zero the output buffer (matching C harness behavior: char output[32768] = {0})
          mov rdi, r14                             ; rdi = output buffer start
          mov ecx, OUT_BUF_SIZE / 8                 ; 4096 qwords
          xor eax, eax                             ; fill value = 0
           rep stosq                                ; zero entire buffer

    ; ========================================================================
    ; Phase 5b: Set up v7_chat arguments and call
    ; v7_chat(char* user_prompt, char* system_prompt, char* output,
    ;         int output_len, int reset)
    ; API_CALL handles shadow space, stack args (5th), and alignment.
    ; ========================================================================
          WRITE_STR sz_trace_chat
          lea rcx, [rel arg_prompt]                ; arg1: parsed prompt (or default)
          xor edx, edx                             ; arg2: system_prompt = NULL
          mov r8, r14                              ; arg3: output buffer
          mov r9d, OUT_BUF_SIZE                    ; arg4: output_len (32KB buffer)
          mov r11d, 1                              ; arg5: reset = 1 (fresh session)
          API_CALL [rel fn_v7_chat], rcx, rdx, r8, r9, r11       ; v7_chat(prompt, NULL, out, 32768, 1)
          WRITE_STR sz_trace_done

          ; --- pseudo-exception check: a caught internal FATAL unwinds to
          ;     here with the engine parked (workers quiesced by the DLL).
          ;     Report the reason and exit 1 -- a caught FATAL must FAIL the
          ;     smoke test visibly, not crash the harness. ---
          mov rax, [rel fn_is_dead]             ; v7_engine_is_dead function pointer
          call rax                              ; eax = 1 when the engine died inside v7_chat
          test eax, eax                         ; engine alive?
          jz .engine_alive                      ; yes -> continue the normal path
          WRITE_STR sz_trace_died               ; no -> print the death banner
          mov rax, [rel fn_get_reason]          ; v7_get_death_reason function pointer
          call rax                              ; rax = reason string pointer (DLL BSS)
          mov r12, rax                          ; keep in a non-volatile (STR_LEN clobbers rax/rcx/rdi)
          STR_LEN r12, r8d                      ; r8d = reason string length
          mov rcx, [rel stdout_handle]          ; arg1: stdout handle (cached by the traces above)
          mov rdx, r12                          ; arg2: reason string
          lea r9, [rel dbg_bytes_written]       ; arg4: bytes_written scratch (BSS, safe)
          API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0    ; print the reason text
          WRITE_STR sz_newline                  ; terminate the reason line
          jmp .do_unload                        ; free DLL then exit with code 1
.engine_alive:

    ; ========================================================================
    ; Phase 6: Print output text to stdout (compute strlen, then WriteFile)
    ; stdout_handle is guaranteed cached: the Phase 2 WRITE_STR ran first
    ; (WRITE_STR lazily initializes it via debug_ensure_std).
    ; ========================================================================
          ; Compute output string length via inline strlen
          lea rdi, [r14]                            ; rdi = output buffer start
          xor ecx, ecx                             ; ecx = length counter = 0
.ststrlen_out:
          cmp ecx, OUT_BUF_SIZE                    ; scanned past the whole buffer already?
          jae .strlen_done                         ; yes: no NUL found -- clamp at buffer size
          cmp byte [rdi + rcx], 0                  ; null terminator?
          je .strlen_done                          ; yes, got length
          inc ecx                                  ; length++
          jmp .ststrlen_out                        ; continue scanning
.strlen_done:
          mov r12d, ecx                            ; r12d = output string length (non-volatile)
          test r12d, r12d                          ; empty output?
          jz .skip_write                           ; nothing to write

            ; Write output buffer to stdout
             mov rcx, [rel stdout_handle]               ; arg1: hFile = stdout handle
              mov rdx, r14                             ; arg2: lpBuffer = output buffer
             mov r8d, r12d                            ; arg3: nNumberOfBytesToWrite = length
              lea r9, [rel dbg_bytes_written]         ; arg4: &bytes_written (BSS global, safe)
              mov qword [rel dbg_bytes_written], 0    ; zero scratch qword
             API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0       ; write output to stdout

.skip_write:
          ; Restore RSP to pre-allocation point
          mov rsp, r15                               ; restore RSP to pre-allocation point

     ; ========================================================================
     ; Phase 7: Shutdown worker threads (MUST run before destroy + FreeLibrary)
     ; ========================================================================
          WRITE_STR sz_trace_shutdown
          API_CALL [rel fn_shutdown]                ; v7_shutdown() -- joins worker threads

    ; ========================================================================
    ; Phase 8: Free inference (parameterless; context lives in BSS inside the DLL)
    ; ========================================================================
          WRITE_STR sz_trace_destroy
          API_CALL [rel fn_v7_free]                 ; v7_free_inference()

    ; ========================================================================
    ; Phase 9: Free DLL and exit
    ; ========================================================================
          mov rcx, [rel dll_handle]                ; arg1: HMODULE
          API_CALL [__imp_FreeLibrary], rcx          ; release DLL
          mov ecx, 0                               ; exit code = 0 (success)
          API_CALL [__imp_ExitProcess], rcx          ; terminate -- never returns

    ; ========================================================================
    ; Error path: free DLL (if loaded) and exit with code 1
    ; ========================================================================
.do_unload:
          mov rcx, [rel dll_handle]                ; arg1: HMODULE
          test rcx, rcx                            ; non-NULL?
          jz .skip_free                            ; NULL, skip FreeLibrary
          API_CALL [__imp_FreeLibrary], rcx          ; release DLL
.skip_free:
          mov ecx, 1                               ; exit code = 1 (error)
          API_CALL [__imp_ExitProcess], rcx          ; terminate -- never returns

; GETARG_REQUIRED miss (dll/db path absent): print usage, exit non-zero.
; Placed at the END of WinMain so the non-local label does not re-anchor the
; .wm_* local labels above (NASM locals bind to the last non-local label).
wm_usage_fail:
          WRITE_STR sz_usage_msg               ; print usage (Sec.106: no default paths)
          mov ecx, 1                           ; exit code = 1
          API_CALL [__imp_ExitProcess], rcx    ; terminate -- never returns

; SF_END emitted but never reached (all paths call ExitProcess)
      SF_END rbx, rsi, rdi, r12, r13, r14, r15       ; unreached: restore non-volatile regs and return
