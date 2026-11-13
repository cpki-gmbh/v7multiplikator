; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define VOCAB_PTR_SCALE 8                      ; sizeof,char* in vocab array
%define HEX_PEEK_OFF_1 0                       ; '0' after '<', the lodsb at .norm_loop already consumed the '<'
%define HEX_PEEK_OFF_2 1                       ; 'x' after '0'
%define HEX_PEEK_OFF_5 4                       ; '>' closing <0xXX>, counted from the post-lodsb cursor
%define HEX_NIBBLE_HI_OFF 2                    ; high nibble char, counted from the post-lodsb cursor
%define HEX_NIBBLE_LO_OFF 3                    ; low nibble char, counted from the post-lodsb cursor
%define UTF8_PEEK_OFF_1 0                      ; 2nd byte of the 3-byte UTF-8, counted from the post-lodsb cursor
%define UTF8_PEEK_OFF_2 1                      ; 3rd byte of the 3-byte UTF-8, counted from the post-lodsb cursor

%macro SL_RESOLVE_STDOUT 2
     cmp qword [rel sl_hStdOut], 0             ; handle cached?
     jne %1                                    ; yes: skip init
     mov ecx, SL_STD_OUTPUT_HANDLE             ; arg1: -11, STD_OUTPUT_HANDLE
     CORRECT_RSP                               ; ABI border: evaluator parity is unknown here
     API_CALL [rel __imp_GetStdHandle]         ; rax = stdout handle
     RESTORE_RSP                               ; drop realignment marker
     cmp rax, -1                               ; INVALID_HANDLE_VALUE?
     je %2                                     ; yes, refuse to cache a sentinel handle
     test rax, rax                             ; NULL handle?
     jz %2                                     ; yes, same sentinel, every byte would vanish
     mov [rel sl_hStdOut], rax                 ; cache it
%1:                                            ; cached-path label: the caller continues here
%endmacro
PutChar:
     push rsi                                  ; preserve the caller's rsi, Win64 non-volatile
     mov byte [rel sl_rw_byte], al             ; stage the byte in BSS first, BSS survives the call, no register staging needed
     lea rsi, [rel sl_rw_byte]                 ; rsi = buffer, PrintRaw's arg2
     mov edx, 1                                ; edx = byte count, PrintRaw's arg3 source
     call PrintRaw                             ; one resolve + WriteFile,1
     pop rsi                                   ; restore non-volatile
     ret                                       ; return to caller
PrintRaw:
     push rbx                                  ; save non-volatile, also stages length: API calls overwrite volatiles
     mov ebx, edx                              ; ebx = byte length, survives GetStdHandle
     SL_RESOLVE_STDOUT .have_handle, .std_fatal ; resolve + cache stdout, macro emits the .have_handle label
     mov rcx, [rel sl_hStdOut]                 ; arg1: hConsoleOutput
     mov rdx, rsi                              ; arg2: lpBuffer, rsi survives API calls, non-volatile
     mov r8d, ebx                              ; arg3: nNumberOfBytesToWrite
     xor r9d, r9d                              ; arg4 = NULL, bytes-written count is discarded, best-effort diagnostic channel
     xor eax, eax                              ; eax = 0, NULL for lpOverlapped
     CORRECT_RSP                               ; ABI border: evaluator parity is unknown here
     API_CALL [rel __imp_WriteFile], rcx, rdx, r8, r9, rax ; write all bytes, 5 args
     RESTORE_RSP                               ; drop realignment marker, rax survives
     pop rbx                                   ; restore non-volatile
     ret                                       ; return to caller
.std_fatal:
     FATAL "SectorLogo: GetStdHandle(STD_OUTPUT_HANDLE) failed, stdout unavailable" ; no console, refuse to cache a sentinel handle
lg_print_value:
     cmp rsp, [rel sl_stk_limit]               ; private-stack watermark: the caller's depth plus a deep print must not exhaust the block
     jb sl_stack_overflow                      ; runaway recursion -> named FATAL, not an AV
     lea rdi, [rel sl_concat_buf]              ; rdi = render destination, lg_value_to_str is rdi-in / rdi-out
     mov rsi, rdi                              ; rsi = text start, lg_value_to_str saves/restores rsi, and PrintRaw needs the start for the length
     call lg_value_to_str                      ; rdi = past last byte, no NUL is written, the length is explicit
     mov rdx, rdi                              ; rdx = end pointer
     sub rdx, rsi                              ; rdx = byte count, no single-instruction form exists: x86 addressing is base + index*scale + disp, so two registers cannot be subtracted in one operand
     jmp PrintRaw                              ; TAIL JUMP: one WriteFile for the whole value; PrintRaw parity-proofs its own API border, CORRECT_RSP, so this frame stays empty
lg_decode_token_real:
     push rbx                                  ; save non-volatile
     push rsi                                  ; save non-volatile
     push rdi                                  ; save non-volatile
     mov rax, [rel sl_hook_ctx]                ; InferenceContext*, non-NULL by caller contract
     mov rax, [rax + SL_CTX_TOKENIZER]         ; Tokenizer*
     mov rbx, [rax + SL_TOK_VOCAB]             ; char** vocab array
     mov esi, edx                              ; zero-extend token id, sampler-valid by contract
     mov rsi, [rbx + rsi*VOCAB_PTR_SCALE]      ; rsi = vocab[token] string
     mov byte [rel sl_tok_buf], SL_TYPE_STRING ; type prefix for Intern
     lea rdi, [rel sl_tok_buf + SL_ATOM_TEXT_OFF] ; rdi = normalization output buffer, content after type byte
     lea r10, [rel sl_tok_buf + SL_TOK_BUF_END_OFF] ; r10 = last writable position
.norm_loop:
     cmp rdi, r10                              ; tok_buf full? r10 = sl_tok_buf + SL_TOK_BUF_END_OFF = the LAST VALID byte, used as the EXCLUSIVE content bound
     jae .tok_buf_full                         ; yes: FATAL, never truncate. A short decode is indistinguishable from a real one, and a truncated token atom is silent data corruption, never clamp, matches lg_ensure_word / sl_str_tokens / lg_value_to_str
     lodsb                                     ; al = *rsi++, ONE load, and the increment is folded in; lodsb leaves RFLAGS untouched so the test below is the iteration's first flag writer
     test al, al                               ; NUL?
     jz .norm_done                             ; yes: done
     cmp al, 0x3C                              ; '<', maybe <0xXX>?
     je .do_hex                                ; rare: byte-fallback check, out of line
     cmp al, 0xE2                              ; first byte of U+2581?
     je .chk_sp                                ; rare: SentencePiece check, out of line
     stosb                                     ; ordinary byte: store AL, advance dest, DF=0; hot fall-through
     jmp .norm_loop                            ; continue normalizing
.chk_sp:
; --- SentencePiece U+2581: E2 96 81 -> space, rsi already past the consumed 0xE2 ---
     cmp byte [rsi + UTF8_PEEK_OFF_1], 0x96    ; second byte?
     jne .copy1                                ; no: just store the 0xE2 we consumed
     cmp byte [rsi + UTF8_PEEK_OFF_2], 0x81    ; third byte?
     jne .copy1                                ; no: just store the 0xE2 we consumed
     mov al, ' '                               ; replace with space
     stosb                                     ; store space, advance dest, DF=0
     add rsi, 2                                ; skip the remaining 2 bytes of the 3-byte sequence
     jmp .norm_loop                            ; continue normalizing
.copy1:
     stosb                                     ; store it verbatim, advance dest, DF=0
     jmp .norm_loop                            ; continue normalizing
.do_hex:
; --- byte-fallback: <0xXX>, rsi already past the consumed '<' ---
     cmp byte [rsi + HEX_PEEK_OFF_1], '0'      ; '0' after '<'?
     jne .copy1                                ; no: just store the '<' we consumed
     cmp byte [rsi + HEX_PEEK_OFF_2], 'x'      ; 'x' after '0'?
     jne .copy1                                ; no: just store
     cmp byte [rsi + HEX_NIBBLE_HI_OFF], 0     ; NUL at the hi-nibble slot, "<0x" ended ? gate BEFORE peeking past skipped bytes
     je .copy1                                 ; yes: do not peek past the terminator
     cmp byte [rsi + HEX_NIBBLE_LO_OFF], 0     ; NUL at the lo-nibble slot, "<0xA" ended ?
     je .copy1                                 ; yes: do not peek past the terminator
     cmp byte [rsi + HEX_PEEK_OFF_5], '>'      ; '>' at the closing slot?, now provably in bounds
     jne .copy1                                ; no: just store the '<' we consumed
     movzx ecx, byte [rsi + HEX_NIBBLE_HI_OFF] ; hi ASCII char, LUT index
     movzx r8d, byte [rsi + HEX_NIBBLE_LO_OFF] ; lo ASCII char, LUT index
     lea r11, [rel hex_val_lut]                ; nibble LUT base
     movzx edx, byte [r11 + rcx]               ; 0 invalid; 1..16 valid
     test edx, edx                             ; nonzero proves valid hex
     jz .copy1                                 ; invalid: AL still holds '<'
     dec edx                                   ; hi nibble 0..15
     movzx eax, byte [r11 + r8]                ; 0 invalid; 1..16 valid
     test eax, eax                             ; nonzero proves valid hex
     jz .hex_bad                               ; invalid: must restore '<' into AL
     dec eax                                   ; lo nibble 0..15
     shl edx, 4                                ; hi nibble << 4
     or al, dl                                 ; al = decoded byte
     stosb                                     ; store decoded byte
     add rsi, 5                                ; skip the remaining <0xXX> bytes
     jmp .norm_loop                            ; continue normalizing
.hex_bad:
     mov al, '<'                               ; restore the '<' the escape started with
     jmp .copy1                                ; store it as ordinary text
.norm_done:
     mov byte [rdi], 0                         ; NUL-terminate output
     lea rsi, [rel sl_tok_buf]                 ; source for Intern
     call Intern                               ; eax = atom offset; falls through into .ret, hot path
.ret:
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     ret                                       ; return to caller
.tok_buf_full:
     FATAL "SectorLogo: decoded token exceeds the 1 MiB token buffer" ; capacity: sl_tok_buf is 1 MiB, NOT the 2 MiB sl_concat_buf the sibling messages name
; --- ASCII-to-hex-nibble lookup table, 256 bytes, .data ---
     section .data
hex_val_lut:
%assign HV_IDX 0                               ; assembly-time LUT index, 0..255
%rep 256
%if (HV_IDX >= '0') && (HV_IDX <= '9')
     db HV_IDX - '0' + 1                       ; '0'-'9' -> nibble 0..9 encoded as 1..10
%elif (HV_IDX >= 'A') && (HV_IDX <= 'F')
     db HV_IDX - 'A' + 11                      ; 'A'-'F' -> nibble 10..15 encoded as 11..16
%elif (HV_IDX >= 'a') && (HV_IDX <= 'f')
     db HV_IDX - 'a' + 11                      ; 'a'-'f' -> nibble 10..15 encoded as 11..16
%else
     db 0                                      ; anything else -> 0, invalid char
%endif
%assign HV_IDX HV_IDX + 1                      ; next LUT index
%endrep
     section .text
%undef VOCAB_PTR_SCALE
%undef HEX_PEEK_OFF_1
%undef HEX_PEEK_OFF_2
%undef HEX_PEEK_OFF_5
%undef HEX_NIBBLE_HI_OFF
%undef HEX_NIBBLE_LO_OFF
%undef UTF8_PEEK_OFF_1
%undef UTF8_PEEK_OFF_2
%undef HV_IDX                                  ; the %rep counter from the hex_val_lut generator above, leaked without this
     global sl_warn_unknown_word               ; called from eval_primary, .atoi_assoc_miss and eval_stmt, .stmt_nodispatch
sl_warn_unknown_word:
     push rsi                                  ; lg_print_value overwrites rsi, its header documents the overwrite ; this wrapper preserves the caller's rsi
     push rcx                                  ; park the position class across the head-classification hardware, free scratch below
     ASSERT_ATOM_NONNIL ebx, .sluw_silent_tagged ; NIL / CONS / NUMBER / FLOAT: never a user typo, no warning
     pop rcx                                   ; restore the position class, the ASSERT overwrites ecx as scratch
     cmp byte [r13 + rax], SL_TYPE_WORD        ; bare word? type byte compared in memory
     jne .sluw_silent                          ; no: STRING/other = legitimate literal data, no warning
     lea r11, [r13 + rax + SL_ATOM_TEXT_OFF]   ; r11 = the atom's text, volatile scratch; rebased below for the cache scan
     movzx r9d, byte [r11]                     ; r9d = first text character
     sub r9d, '0'                              ; unsigned digit test: <10 means '0'..'9'
     cmp r9d, 9                                ; digit-leading?
     jbe .sluw_silent                          ; yes: number-literal class, no warning
; --- diagnostic seen-cache: one indexed bit test and set; collisions suppress warnings ---
     mov r9d, eax                              ; tagged atom offset, rcx is the live position class
     shr r9d, 3                                ; heap atom offsets are 8-byte-granule aligned
     and r9d, 1023                             ; bitmap index 0..1023
     lea r11, [rel sl_warn_seen]               ; 128-byte (1024-bit) bitmap
     bts dword [r11], r9d                      ; test/set indexed bit across bitmap dwords (register-form bit index spans the array)
     jc .sluw_silent                           ; warning was already emitted for this index
; --- print: fixed prose, atom text, class tail + newline ---
     push rcx                                  ; park the position class FIRST, bottom of the pair : PrintRaw stages the stdout handle into rcx and lg_print_value tail-jumps into PrintRaw, the class cannot survive either call in a register
     push rax                                  ; the atom is the CALLER's live value: preserved across the print ops
     lea rsi, [rel sl_warn_pfx]                ; rsi = "SectorLogo: unknown word '"
     mov edx, SL_WARN_PFX_LEN                  ; edx = its length
     call PrintRaw                             ; one WriteFile, stdout console channel
     pop rax                                   ; restore the atom for lg_print_value, the parked class stays under it
     call lg_print_value                       ; print the atom text, eax = tagged atom; rcx overwritten by its PrintRaw tail jump, the class is parked
     pop rcx                                   ; restore the position class for the tail pick below
     lea rsi, [rel sl_warn_tail_expr]          ; rsi = default, expression class tail: "' self-evaluates in expression position\n"
     mov edx, SL_WARN_TAIL_EXPR_LEN            ; edx = its length
     test ecx, ecx                             ; ecx = class, 0 = expression ?
     jnz .sluw_class_stmt                      ; class 1: use the statement tail
     call PrintRaw                             ; one WriteFile, stdout console channel
     jmp .sluw_pop_done                        ; shared epilogue
.sluw_class_stmt:
     lea rsi, [rel sl_warn_tail_stmt]          ; rsi = "' skipped in statement position\n"
     mov edx, SL_WARN_TAIL_STMT_LEN            ; edx = its length
     call PrintRaw                             ; one WriteFile, stdout console channel
.sluw_pop_done:
     pop rsi                                   ; restore non-volatile, the only one this helper saved
     ret                                       ; return: ebx untouched, the atom survives there, Win64 non-volatile ; eax is consumed as lg_print_value's input and holds WriteFile's BOOL on return, both call sites re-derive eax from ebx
.sluw_silent_tagged:
     pop rcx                                   ; the guard edge jumped BEFORE the shared pop: discard its parked class copy
.sluw_silent:
     pop rsi                                   ; restore non-volatile on the silent edges too, every path is symmetric
     ret                                       ; return, atom preserved
; --- the three fixed prose strings, data + lengths, bytes, via the $ delta ---
     section .data                             ; the prose constants: ASCII text, no runtime writes
sl_warn_pfx:
     db "SectorLogo: unknown word '"           ; shared prefix line
     SL_WARN_PFX_LEN equ $ - sl_warn_pfx       ; byte count of the prefix
sl_warn_tail_expr:
     db "' self-evaluates in expression position (a typo? quoting a word puts it there on purpose)", 10 ; expression-position tail, \n = LF
     SL_WARN_TAIL_EXPR_LEN equ $ - sl_warn_tail_expr ; byte count of the expression tail
sl_warn_tail_stmt:
     db "' skipped in statement position (a typo?)", 10 ; statement-position tail, \n = LF
     SL_WARN_TAIL_STMT_LEN equ $ - sl_warn_tail_stmt ; byte count of the statement tail
     section .text                             ; back to code: subsequent labels are functions
