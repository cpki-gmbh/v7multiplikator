; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; GetChar -> al = next char from script buffer, or 0 on EOF
; Reads from sl_src_mem_ptr; advances pointer on non-NUL.
; ============================================================================
GetChar:
     mov rdx, [rel sl_src_mem_ptr]             ; load current source pointer, volatile rdx, not rsi
     movzx eax, byte [rdx]                     ; read next byte
     test al, al                               ; NUL = end of script
     jz .eof                                   ; return 0
     inc rdx                                   ; advance pointer in register
     mov [rel sl_src_mem_ptr], rdx             ; store updated pointer, pure store, no RMW
     ret                                       ; return to caller
.eof:
     ret                                       ; return 0, eax is ALREADY 0, the movzx above zero-extends the NUL byte, and this label is reached only when al == 0

%macro TOK_ADV_LOOKAHEAD 0
     movzx eax, byte [r15]                     ; next byte, r15 = running source cursor
     test al, al                               ; NUL = EOF?
     jz %%no_adv                               ; yes: al = 0, do NOT advance, GetChar semantics
     inc r15                                   ; advance the cursor past the consumed byte
%%no_adv:
%endmacro

%macro GETCHAR_INLINE 0
     movzx eax, byte [r15]                     ; next byte, r15 = running source cursor
     test al, al                               ; NUL = EOF?
     jz %%eof                                  ; yes: leave the cursor put, al is already 0
     inc r15                                   ; advance past the consumed byte
%%eof:
%endmacro
; TOK_STORE_ADV, store current char, al, advance dest, r11, refresh lookahead.
; For word tokens: folds A-Z -> a-z before storing, case-insensitive code .

%macro TOK_STORE_ADV 0
     mov byte [r11], al                        ; store char to tok_buf
     inc r11                                   ; advance dest pointer
     TOK_ADV_LOOKAHEAD                         ; read next char + publish lookahead
%endmacro

%macro TOK_STORE_ADV_FOLD 0
     cmp al, 'A'                               ; below 'A'?
     jb %%no_fold                              ; yes: no fold needed
     cmp al, 'Z'                               ; above 'Z'?
     ja %%no_fold                              ; yes: no fold needed
     add al, 32                                ; fold A-Z -> a-z
%%no_fold:
     mov byte [r11], al                        ; store, possibly folded char
     inc r11                                   ; advance dest pointer
     TOK_ADV_LOOKAHEAD                         ; read next char + publish lookahead
%endmacro
; TOKCLASS_JMP_SCALE, qword stride of the first-char classification jump table.
%define SL_TOKCLASS_JMP_SCALE 8                ; bytes per qword entry, class id -> handler address

%macro OP_SECOND_MATCH 1
     cmp al, %1                                ; second char match?
     je .op_second                             ; yes: two-char operator
%endmacro
GetToken:
     push r15                                  ; save non-volatile, running source cursor for this token; every non-FATAL exit publishes it to sl_src_mem_ptr before the matching pop
     mov r15, [rel sl_src_mem_ptr]             ; r15 = running source cursor for this whole token, the per-character load/store pair in TOK_ADV_LOOKAHEAD and GETCHAR_INLINE becomes a single indexed read
     lea r11, [rel sl_tok_buf + SL_ATOM_TEXT_OFF] ; r11 = content start, offset 1; slot 0 reserved for type byte
     mov byte [rel sl_tok_buf], SL_TYPE_WORD   ; default type byte = WORD, string-literal path overwrites to STRING
     lea r8, [rel sl_tok_buf + SL_TOK_BUF_END_OFF] ; r8 = last writable byte; the jae bound, cmp r11,r8 leaves SIZE-2 for content and SIZE-1 for the NUL at .done
     lea r10, [rel sl_tok_class]               ; one shared table: delimiter class, boundary-only flag, and ASCII alphanumeric flag
     movzx eax, byte [rel sl_g_look]           ; al = current lookahead
     xor r9d, r9d                              ; ws-flag shadow: 0 = no leading whitespace yet, published to BSS at .skip_done, one store per token, not one per ws byte
; --- skip whitespace and comments ---
.skip_ws:
     test al, al                               ; EOF?
     jz .empty                                 ; yes: empty token
     cmp al, ' '                               ; whitespace or below?
     ja .not_ws                                ; no: done skipping
     mov r9b, 1                                ; this token begins after whitespace, register-held; BSS store deferred to .skip_done
     TOK_ADV_LOOKAHEAD                         ; consume char, refresh lookahead
     jmp .skip_ws                              ; keep skipping
; --- skip ';'-to-EOL comments ---
.not_ws:
     cmp al, 0x3B                              ; comment delimiter, ';'
     jne .skip_done                            ; no: done skipping, al > ' ' guaranteed, EOF impossible here
.skip_comment:
     mov r9b, 1                                ; a comment separates tokens exactly like whitespace, set the leading-ws flag: `1;comment<LF>-2` must not glue the sign
     GETCHAR_INLINE                            ; consume comment char
     test al, al                               ; EOF?
     jz .empty                                 ; yes: empty token
     cmp al, 10                                ; newline LF?
     je .comment_end                           ; yes: done with comment
     cmp al, 13                                ; newline CR?
     jne .skip_comment                         ; keep consuming comment; LF/CR ends it
.comment_end:
     TOK_ADV_LOOKAHEAD                         ; consume the newline, refresh lookahead
     jmp .skip_ws                              ; resume whitespace skip
.skip_done:
     mov byte [rel sl_tok_leading_ws], r9b     ; publish the ws flag once, every classification path passes here before .op_sign reads it
; fall through directly to the string-literal check, every predecessor
; guarantees al >= 0x21 here, .skip_ws exits on NUL before .not_ws can see it
; --- double-quoted string literal: "..." with \" and \\ escapes ---
     cmp al, '"'                               ; opening double-quote?
     jne .not_string                           ; no: try other token types
     mov dword [rel sl_tok_type], SL_TOK_STRING ; mark as string literal token
     mov byte [rel sl_tok_buf], SL_TYPE_STRING ; type byte = STRING for this token
.str_loop:
     GETCHAR_INLINE                            ; read next char inside string
     test al, al                               ; EOF inside string?
     jz .str_unterminated                      ; yes: unterminated string
     cmp al, '"'                               ; closing double-quote?
     je .str_close                             ; yes: end of string literal
     cmp al, '\'                               ; backslash escape?
     jne .str_store                            ; no: store char verbatim
     GETCHAR_INLINE                            ; read char after backslash
     test al, al                               ; EOF after backslash?
     jz .str_unterminated                      ; yes: unterminated escape
     cmp al, '"'                               ; escaped quote \"?
     je .str_store                             ; yes: store literal "
     cmp al, '\'                               ; escaped backslash \\?
     je .str_store                             ; yes: store literal backslash
     cmp al, 'n'                               ; \n -> LF, 0x0A ?
     jne .str_store                            ; no: store as-is
     mov al, 10                                ; convert \n to LF char
.str_store:
     cmp r11, r8                               ; tok_buf full?
     jae .str_overflow_fatal                   ; literal exceeds buffer, FATAL, truncated script = corrupted script
     mov byte [r11], al                        ; store char to tok_buf
     inc r11                                   ; advance dest pointer
     jmp .str_loop                             ; keep reading string
.str_close:
     TOK_ADV_LOOKAHEAD                         ; advance past closing "
     mov dword [rel sl_last_tok_operand], SL_TOK_NOT_OPERAND ; string literal is a name, not a numeric operand
     jmp .done                                 ; finish string token
.str_unterminated:
     FATAL "SectorLogo: unterminated string"   ; EOF inside string literal
.str_overflow_fatal:
     FATAL "SectorLogo: string literal exceeds token buffer"
.not_string:
     mov dword [rel sl_tok_type], SL_TOK_ATOM  ; default: bare word token
     movzx ecx, al                             ; first char -> index into the combined class table
     test byte [r10 + rcx], SL_TOK_BOUNDARY_MASK ; token boundary? class 1..11 or boundary-only 0x80, alnum 0x40 is a word char
     jz .accum                                 ; yes, the common class by far: branch straight to word accumulation, never touching the class or jump table
; --- rare: a delimiter or operator, resolve its handler through the class table ---
     movzx edx, byte [r10 + rcx]               ; class id 1..11 from the same table; special classes cannot reach this point
     lea rcx, [rel sl_tokclass_jmp]            ; class -> handler jump table
     jmp [rcx + rdx*SL_TOKCLASS_JMP_SCALE]     ; dispatch to the existing handler labels
.accum:
     cmp r11, r8                               ; tok_buf full?
     jae .atom_overflow_fatal                  ; atom exceeds buffer, FATAL, never silently truncated
     TOK_STORE_ADV_FOLD                        ; store char, folded, advance, refresh lookahead
     movzx ecx, al                             ; char class index
     test byte [r10 + rcx], SL_TOK_BOUNDARY_MASK ; includes EOF, whitespace, comment, quote, and all operator classes
     jz .accum                                 ; not a boundary: keep accumulating, al already holds lookahead
.accum_done:
     mov dl, byte [rel sl_tok_buf + SL_ATOM_TEXT_OFF] ; only DL is consumed before returning
     cmp dl, '0'                               ; below '0'?
     jb .check_signed_numeric                  ; yes: could be -N/+N, sign bytes are below '0', signed check rejects everything else
     cmp dl, '9'                               ; above '9'?
     ja .check_signed_numeric                  ; yes: check for -/+ prefix
     mov dword [rel sl_last_tok_operand], SL_TOK_OPERAND ; digit-start: number operand
     jmp .done                                 ; finish numeric operand token
.check_signed_numeric:
     cmp dl, '-'                               ; minus sign?
     je .check_sign_digit                      ; yes: check digit after
     cmp dl, '+'                               ; plus sign?
     jne .not_numeric_operand                  ; no: keyword token
.check_sign_digit:
     mov dl, byte [rel sl_tok_buf + SL_TOK_BUF_SIGN_PEEK] ; only DL is consumed before returning
     sub dl, '0'                               ; ASCII -> raw digit
     cmp dl, 9                                 ; valid digit 0..9?
     ja .not_numeric_operand                   ; not a digit after sign
     mov dword [rel sl_last_tok_operand], SL_TOK_OPERAND ; -N or +N: numeric operand
     jmp .done                                 ; finish signed-numeric operand token
.not_numeric_operand:
     mov dword [rel sl_last_tok_operand], SL_TOK_NOT_OPERAND ; keyword: not operand
     jmp .done                                 ; finish keyword token
; --- single-char delimiter: store, advance lookahead, return ---
.delim:
     TOK_STORE_ADV                             ; store delimiter char, advance, refresh lookahead
     mov dword [rel sl_last_tok_operand], SL_TOK_NOT_OPERAND ; [ are not operands
     jmp .done                                 ; finish token
.delim_close:
     TOK_STORE_ADV                             ; store delimiter char, advance, refresh lookahead
     mov dword [rel sl_last_tok_operand], SL_TOK_OPERAND ; ] are operand-ends
     jmp .done                                 ; finish token
.delim_list_open:
     TOK_STORE_ADV                             ; store '[', advance, refresh lookahead
     mov dword [rel sl_last_tok_operand], SL_TOK_NOT_OPERAND ; '[' is not an operand
     inc dword [rel sl_list_depth]             ; enter a list literal: elements are standalone from here
     jmp .done                                 ; finish token
.delim_list_close:
     TOK_STORE_ADV                             ; store ']', advance, refresh lookahead
     mov dword [rel sl_last_tok_operand], SL_TOK_OPERAND ; ']' is an operand-end
     dec dword [rel sl_list_depth]             ; leave a list literal
     jns .done                                 ; depth still >= 0: balanced, or nested
     FATAL "SectorLogo: stray ']' with no matching '[' (unbalanced bracket)" ; silent NIL evaluation hides the typo, name it
; --- single-char operator delimiter, not -/+ : store, advance, return ---
.op_delim:
     TOK_STORE_ADV                             ; store operator char, advance, refresh lookahead
     mov dword [rel sl_last_tok_operand], SL_TOK_NOT_OPERAND ; operators are not operands
     jmp .done                                 ; finish token
.op_sign:
     TOK_STORE_ADV                             ; store the sign char, advance, refresh lookahead
     cmp al, '0'                               ; digit after sign?
     jb .op_sign_bare                          ; no: bare -/+ operator
     cmp al, '9'                               ; upper digit bound?
     ja .op_sign_bare                          ; no: bare -/+ operator
     cmp dword [rel sl_list_depth], 0          ; inside a list literal?
     jne .accum                                ; yes: always glue, elements are standalone
     cmp dword [rel sl_last_tok_operand], SL_TOK_NOT_OPERAND ; prev token was not an operand?
     je .accum                                 ; yes: unary position, keyword head
     cmp byte [rel sl_tok_leading_ws], 1       ; sign begins after whitespace?
     je .accum                                 ; yes: spaced prefix argument, glue
.op_sign_bare:
     mov dword [rel sl_last_tok_operand], SL_TOK_NOT_OPERAND ; -/+ is not an operand
     jmp .done                                 ; finish token
; --- '<' may be <, <=, <>, or << ---
.op_lt:
     TOK_STORE_ADV                             ; store '<', advance, refresh lookahead
     OP_SECOND_MATCH '='                       ; <= ?
     OP_SECOND_MATCH '>'                       ; <> ?
     OP_SECOND_MATCH '<'                       ; << ?
     mov dword [rel sl_last_tok_operand], SL_TOK_NOT_OPERAND ; comparison operator
     jmp .done                                 ; just '<'
; --- '>' may be > or >= or >> ---
.op_gt:
     TOK_STORE_ADV                             ; store '>', advance, refresh lookahead
     OP_SECOND_MATCH '='                       ; >= ?
     OP_SECOND_MATCH '>'                       ; >> ?
     mov dword [rel sl_last_tok_operand], SL_TOK_NOT_OPERAND ; comparison operator
     jmp .done                                 ; just '>'
; --- '!' may be != ---
.op_bang:
     TOK_STORE_ADV                             ; store '!', advance, refresh lookahead
     OP_SECOND_MATCH '='                       ; != ?
     mov dword [rel sl_last_tok_operand], SL_TOK_NOT_OPERAND ; '!' or '!=' are not operands
     jmp .done                                 ; just '!', treated as word
; --- '&' may be & or && ---
.op_amp:
     TOK_STORE_ADV                             ; store '&', advance, refresh lookahead
     OP_SECOND_MATCH '&'                       ; && ?
     mov dword [rel sl_last_tok_operand], SL_TOK_NOT_OPERAND ; '&' is not an operand
     jmp .done                                 ; just '&'
; --- '|' may be | or || ---
.op_pipe:
     TOK_STORE_ADV                             ; store '|', advance, refresh lookahead
     OP_SECOND_MATCH '|'                       ; || ?
     mov dword [rel sl_last_tok_operand], SL_TOK_NOT_OPERAND ; '|' is not an operand
     jmp .done                                 ; just '|'
.op_second:
     TOK_STORE_ADV                             ; store second char, advance, refresh lookahead
     mov dword [rel sl_last_tok_operand], SL_TOK_NOT_OPERAND ; two-char operators are not operands
; fall through to .done
.done:
     mov byte [r11], 0                         ; NUL-terminate tok_buf
     mov byte [rel sl_g_look], al              ; publish the lookahead ONCE per token, al is the current lookahead on EVERY path into .done, .accum_done classifies in edx so it no longer overwrites al; see the note at :47
     mov [rel sl_src_mem_ptr], r15             ; publish the cursor, every non-FATAL exit funnels through here or .empty, .op_second falls through
     pop r15                                   ; restore non-volatile
     ret                                       ; return to caller
.atom_overflow_fatal:
     FATAL "SectorLogo: token exceeds buffer"
.empty:
     mov dword [rel sl_tok_type], SL_TOK_EOF   ; mark as EOF token
     mov word [rel sl_tok_buf], 0              ; type byte = 0 + NUL content in one store
     mov byte [rel sl_g_look], 0               ; publish EOF lookahead: al == 0 on both .empty entries, but .skip_comment consumes via GETCHAR_INLINE which does NOT publish, so without this an unterminated trailing comment leaves a STALE non-zero byte and the next GetToken re-enters the lexer on it
     mov [rel sl_src_mem_ptr], r15             ; publish the cursor, second non-FATAL exit
     pop r15                                   ; restore non-volatile
     ret                                       ; return to caller
     section .rdata
     align 8
sl_tok_class:
     times 0x21 db SL_TOK_BOUNDARY_ONLY        ; 0x00-0x20: word boundaries; handled before token dispatch
     db 7                                      ; 0x21 '!' -> .op_bang, maybe !=
     db SL_TOK_BOUNDARY_ONLY                   ; 0x22 '"', boundary, string-literal path
     db 0, 0                                   ; 0x23 '#', 0x24 '$' -> word chars
     db 9                                      ; 0x25 '%' -> .op_delim
     db 10                                     ; 0x26 '&' -> .op_amp, maybe &&
     db 0                                      ; 0x27 '\'' -> word char
     db 1                                      ; 0x28 ',' -> .delim
     db 2                                      ; 0x29 ' ' -> .delim_close
     db 9                                      ; 0x2A '*' -> .op_delim
     db 8                                      ; 0x2B '+' -> .op_sign, unary gluing check
     db 0                                      ; 0x2C ',' -> word char
     db 8                                      ; 0x2D '-' -> .op_sign, unary gluing check
     db 0                                      ; 0x2E '.' -> word char
     db 9                                      ; 0x2F '/' -> .op_delim
     times 10 db SL_TOK_ALNUM_MASK             ; 0x30-0x39 '0'-'9': word chars, valid names
     db 0                                      ; 0x3A ':' word char, not valid in names
     db SL_TOK_BOUNDARY_ONLY                   ; 0x3B ';': comment and token boundary
     db 5                                      ; 0x3C '<' -> .op_lt, maybe <= <> <<
     db 9                                      ; 0x3D '=' -> .op_delim
     db 6                                      ; 0x3E '>' -> .op_gt, maybe >= >>
     db 0, 0                                   ; 0x3F '?', 0x40 '@' -> word chars
     times 26 db SL_TOK_ALNUM_MASK             ; 0x41-0x5A 'A'-'Z' -> word chars, folded at store time
     db 3                                      ; 0x5B '[' -> .delim_list_open, bumps nesting depth
     db 0                                      ; 0x5C '\' -> word char, per the non-boundary list
     db 4                                      ; 0x5D ']' -> .delim_list_close, nesting depth--
     db 9                                      ; 0x5E '^' -> .op_delim
     db 9                                      ; 0x5F '_' -> .op_delim, concat
     db 0                                      ; 0x60 '`' -> word char
     times 26 db SL_TOK_ALNUM_MASK             ; 0x61-0x7A 'a'-'z' -> word chars
     db 0                                      ; 0x7B '{' -> word char
     db 11                                     ; 0x7C '|' -> .op_pipe, maybe ||
     db 0                                      ; 0x7D '}' -> word char
     db 9                                      ; 0x7E '~' -> .op_delim
     times 0x81 db 0                           ; 0x7F-0xFF: high bytes + UTF-8 -> word chars, verbatim, no folding
     align 8
%ifdef V7_LINUX
     section .data.rel.ro                      ; ELF loader relocates lexer pointers before RELRO
%endif
sl_tokclass_jmp:
     dq GetToken.accum                         ; class 0: word accumulation, case-folded
     dq GetToken.delim                         ; class 1: ',' single-char delimiter
     dq GetToken.delim_close                   ; class 2: ' ' operand-end delimiter
     dq GetToken.delim_list_open               ; class 3: '[' list-open, sl_list_depth++
     dq GetToken.delim_list_close              ; class 4: ']' list-close, sl_list_depth--
     dq GetToken.op_lt                         ; class 5: '<' family, < <= <> <<
     dq GetToken.op_gt                         ; class 6: '>' family, > >= >>
     dq GetToken.op_bang                       ; class 7: '!' family, ! !=
     dq GetToken.op_sign                       ; class 8: '-' / '+', unary gluing check
     dq GetToken.op_delim                      ; class 9: * / % _ = ^ ~ single-char operator delimiter
     dq GetToken.op_amp                        ; class 10: '&' family, & &&
     dq GetToken.op_pipe                       ; class 11: '|' family, | ||
%ifdef V7_LINUX
     section .rdata                            ; fixed lexer strings need no relocations
%endif
     align 8
     sl_trace_spaces: times (SL_TRACE_MAX_INDENT * 2) db ' ' ; 80 spaces
sl_trace_expr_prefix:
     db '|', ' ', '=', ' '                     ; trace result prefix, length 4, no NUL, PrintRaw is length-based
     section .text                             ; restore the code section for the parser below
ReadLogoStream:
     inc dword [rel sl_parse_depth]            ; enter one parse level
     cmp dword [rel sl_parse_depth], SL_MAX_PARSE_DEPTH ; past the guardrail?
     ja sl_parse_depth_fatal                   ; yes: named FATAL, not a stack overflow, non-local label: reachable from both guard sites
     cmp rsp, [rel sl_stk_limit]               ; private-stack watermark: one frame per element can outrun even 64 MiB on adversarial input
     jb sl_stack_overflow                      ; runaway recursion -> named FATAL, not an AV
     call GetToken                             ; read first token
     cmp byte [rel sl_tok_buf + SL_ATOM_TEXT_OFF], 0 ; empty content, skip type byte ?
     jne .parse                                ; no: parse normally
     cmp dword [rel sl_tok_type], SL_TOK_STRING ; empty string literal ""?
     jne .empty                                ; no: truly EOF
.parse:
     call ParseLogoObject                      ; eax = parsed object
     push rax                                  ; save parsed object
     call ReadLogoStream                       ; eax = rest of stream
     mov edx, eax                              ; edx = rest, cdr
     pop rax                                   ; rax = object, car
     call Cons                                 ; eax =, object.rest
     dec dword [rel sl_parse_depth]            ; frame done: unwind the guard
     ret                                       ; return to caller
.empty:
     xor eax, eax                              ; return NIL
     dec dword [rel sl_parse_depth]            ; frame done: unwind the guard
     ret                                       ; return to caller
; ============================================================================
; ParseLogoObject -> eax = parsed object, atom or list
; Dispatches on tok_buf: '[' -> ParseList, else -> Intern.
; Preserves all callee-saved registers, rsi is pushed around the Intern call .
; ============================================================================
ParseLogoObject:
     cmp dword [rel sl_tok_type], SL_TOK_STRING ; string literal token?
     je .parse_atom                            ; yes: NEVER a list, intern verbatim, brackets inside strings are data
     cmp byte [rel sl_tok_buf + SL_ATOM_TEXT_OFF], '[' ; list literal, skip type byte ?
     je .parse_list                            ; yes: parse list
.parse_atom:
     push rsi                                  ; save caller's rsi + parity fix
     lea rsi, [rel sl_tok_buf]                 ; source = tok_buf, type byte at [0]
     call Intern                               ; eax = atom offset
     pop rsi                                   ; restore callee-saved rsi
     ret                                       ; return to caller
.parse_list:
     inc dword [rel sl_parse_depth]            ; parser recursion guard, one level per list element + one per nesting
     cmp dword [rel sl_parse_depth], SL_MAX_PARSE_DEPTH ; past the guardrail?
     ja sl_parse_depth_fatal                   ; yes: named FATAL, not a stack overflow
     cmp rsp, [rel sl_stk_limit]               ; private-stack watermark: one frame per element can outrun even 64 MiB on adversarial input
     jb sl_stack_overflow                      ; runaway recursion -> named FATAL, not an AV
     call GetToken                             ; fetch the first token AFTER '[', element or ']', the '[' itself was consumed by the GetToken whose result dispatched here
     cmp dword [rel sl_tok_type], SL_TOK_STRING ; string literal token?
     je .parse_list_elem                       ; yes: NEVER ']', string content is data, same guard as the '[' dispatch
     cmp byte [rel sl_tok_buf + SL_ATOM_TEXT_OFF], ']' ; empty list, skip type byte ?
     je .end_list                              ; yes: return NIL
     cmp byte [rel sl_tok_buf + SL_ATOM_TEXT_OFF], 0 ; empty content, skip type byte ?
     je .unbalanced_list_fatal                 ; truly EOF inside a list: unbalanced '[', FATAL, truncated AST = half a program; content falls through
.parse_list_elem:
     call ParseLogoObject                      ; eax = first element
     push rax                                  ; save first element
     call .parse_list                          ; eax = rest of list
     mov edx, eax                              ; edx = rest, cdr
     pop rax                                   ; rax = element, car
     call Cons                                 ; eax =, element.rest
     dec dword [rel sl_parse_depth]            ; frame done: unwind the guard
     ret                                       ; return to caller
.end_list:
     xor eax, eax                              ; return NIL
     dec dword [rel sl_parse_depth]            ; frame done: unwind the guard
     ret                                       ; return to callers
.unbalanced_list_fatal:
     FATAL "SectorLogo: unexpected end of script inside a list (unbalanced '[')" ; script typo: a partial AST must never execute
sl_parse_depth_fatal:
     FATAL "SectorLogo: parser nesting too deep (script recursion exceeds 262144)" ; stack-overflow guard: ~12.6 MiB worst case at the limit, far under the 64 MiB private stack
; ---- %undef cleanup: file-local names leave scope here, house convention ----
%undef SL_TOKCLASS_JMP_SCALE
%unmacro TOK_ADV_LOOKAHEAD 0                   ; arity MUST match the %macro declaration
%unmacro GETCHAR_INLINE 0                      ; fetch next byte + advance source pointer, stops on NUL
%unmacro TOK_STORE_ADV 0                       ; store al to tok_buf, advance dest, r11, refresh lookahead
%unmacro TOK_STORE_ADV_FOLD 0                  ; TOK_STORE_ADV plus A-Z -> a-z fold of the stored char
%unmacro OP_SECOND_MATCH 1                     ; two-char operator detect: match lookahead, al against the operator's second char
