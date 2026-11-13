; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define SL_DOT_SEP_BYTES 3                     ; lg_value_to_str "." separator width before a dotted, improper-list tail
%define SL_DOT_MID_OFF 1                       ; '.' byte offset inside that separator
%define SL_DOT_END_OFF 2                       ; trailing ' ' byte offset inside that separator
%define SL_NIL_ELEM_BYTES 2                    ; lg_value_to_str NIL-element '[]' pair width inside list rendering
%define SL_NUMBER_MAX_TEXT 10                  ; widest signed 30-bit NUMBER render in decimal bytes, sign included
; ============================================================================
; lg_num_to_str,eax = raw int32, rdi = dest -> rdi = past last digit
; Converts a signed int32 to decimal ASCII at [rdi]. Does NOT NUL-terminate.
; Preserves all non-volatile registers. Overwrites eax/ecx/edx/r8d.
; ============================================================================
%define SL_DIV10_MAGIC 3435973837              ; 0xCCCCCCCD = ceil,2^35/10, exact unsigned /10 reciprocal
%define SL_DIV10_SHIFT 35                      ; n * MAGIC >> SHIFT == n / 10 for every unsigned 32-bit n
lg_num_to_str:
     push rbx                                  ; save non-volatile
     mov ebx, eax                              ; ebx = value to convert
     test ebx, ebx                             ; negative?
     jns .pos                                  ; no: skip sign
     mov byte [rdi], '-'                       ; write minus sign
     inc rdi                                   ; advance dest
     neg ebx                                   ; make magnitude positive
.pos:
; push digits in reverse order
     xor r8d, r8d                              ; digit counter
     mov ecx, SL_DIV10_MAGIC                   ; /10 reciprocal, loop invariant, 32-bit dest zero-extends into rcx
.div_loop:
     mov rax, rbx                              ; rax = zero-extended remaining magnitude, <= 2^31, see .pos
     imul rax, rcx                             ; rax = magnitude * 0xCCCCCCCD, 2-operand: both sources in registers
     shr rax, SL_DIV10_SHIFT                   ; rax = quotient = magnitude / 10, exact for all 32-bit inputs
     imul rdx, rax, 10                         ; rdx = quotient * 10
     sub ebx, edx                              ; ebx = remainder = magnitude - quotient*10 = the digit, 0..9.Two things are load-bearing here: the MINUEND must be the magnitude, subtracting the other way gives 10q-m, which underflows for every magnitude below 10 and renders 'm=1' as '/', and the width must be 32-bit so the result zero-extends into rbx instead of sign-extending
     xchg rax, rbx                             ; rax = digit, rbx = quotient = the remaining magnitude the next test and iteration consume, the loop MUST carry the quotient, not the remainder, or a magnitude below 10 never terminates. xchg does both halves in one instruction
     push rax                                  ; push digit
     inc r8d                                   ; count digits
     test ebx, ebx                             ; the quotient fits in unsigned 32 bits
     jnz .div_loop                             ; yes: keep dividing
.print_loop:
     pop rax                                   ; pop digit
     add al, '0'                               ; convert to ASCII
     mov byte [rdi], al                        ; write digit
     inc rdi                                   ; advance dest
     dec r8d                                   ; more digits?
     jnz .print_loop                           ; yes: keep printing
     pop rbx                                   ; restore non-volatile
     ret                                       ; return to caller
; ============================================================================
; lg_value_to_str,eax = tagged value, rdi = dest -> rdi = past last byte
; Renders any SectorLogo value, atom/NUMBER/FLOAT/CONS/NIL as text at [rdi].
; Does NOT NUL-terminate; NIL renders as the empty string. rsi is saved and
; restored internally; all volatiles are overwritten, r8 is the bounds register =
; sl_concat_buf + SL_CONCAT_BUF_LAST on every path.Every byte is bounds-checked
; against r8 before its store, exhaustion is a capacity FATAL, never a
; truncation and never an overrun.
; ============================================================================
lg_value_to_str:
     push rsi                                  ; save non-volatile
     lea r8, [rel sl_concat_buf + SL_CONCAT_BUF_LAST] ; r8 = buffer end, bounds limit, ALL paths
     test eax, eax                             ; NIL?
     jz .done                                  ; yes: write nothing
     mov ecx, eax                              ; check tag bits
     shr ecx, 30                               ; packed type code: 0=atom, 1=number, 2=cons, 3=float
     jnz .not_atom                             ; not a plain atom
     lea rsi, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rsi = atom string in heap
     vpxor xmm1, xmm1, xmm1                    ; zero vector, NUL detector
     vpcmpeqb xmm0, xmm1, [rsi]                ; NUL lanes -> 0xFF, load folded into the compare
     vptest xmm0, xmm0                         ; NUL within the first 16 bytes?
     jnz .atom_copy                            ; yes: short atom, byte loop only, NO leaf call
     call sl_swar_strlen                       ; ecx = strlen >= 16, guaranteed by the probe, rsi/rdi preserved
     lea rax, [rdi + rcx]                      ; rax = one past last byte to write
     cmp rax, r8                               ; past the bounds limit?
     ja .concat_overflow                       ; yes: FATAL, truncated result atom = silent data corruption
     rep movsb                                 ; ERMS burst copy, rdi lands past the copy
     jmp .done                                 ; done, rdi past last byte
.atom_copy:
     lodsb                                     ; load al from [rsi], rsi++, entry: first byte, :85 lands here
     test al, al                               ; NUL?
     jz .done                                  ; yes: done, rdi past last byte
     cmp rdi, r8                               ; buffer full?
     jae .concat_overflow                      ; yes: FATAL, truncated result atom = silent data corruption
.atom_copy_body:
     stosb                                     ; store the checked byte to [rdi], rdi++
     lodsb                                     ; load next byte
     test al, al                               ; NUL?
     jz .done                                  ; yes: done, rdi past last byte
     cmp rdi, r8                               ; buffer full?
     jb .atom_copy_body                        ; room AND more text, store it, bottom test replaces the unconditional jmp
     jmp .concat_overflow                      ; buffer exhausted mid-atom
.not_atom:
     cmp ecx, 3                                ; FLOAT tag code
     je .is_float_val                          ; yes: format decimal text
     cmp ecx, 1                                ; NUMBER tag code
     jne .cons_val                             ; CONS: write [elem1 elem2 ...] representation
     call NumVal                               ; eax = signed 30-bit NUMBER value
     mov r11d, eax                             ; preserve value across the width slow path

; every valid tagged NUMBER occupies at most 10 decimal bytes, including '-'
     lea rcx, [rdi + SL_NUMBER_MAX_TEXT]       ; one past the widest NUMBER render
     cmp rcx, r8                               ; room for any NUMBER?
     jbe .num_emit                             ; fast path

; rare near-capacity path: exact printed width before storing
     xor r10d, r10d                            ; r10d = string width in bytes
     mov r9d, eax                              ; r9d = digit source
     test r9d, r9d                             ; negative?
     jns .num_positive                         ; no: no sign byte
     inc r10d                                  ; width += 1, minus sign
     neg r9d                                   ; unsigned magnitude, neg wraps INT32_MIN onto itself
     lea rax, [rel pow10_table - SL_POW10_DWORD] ; pre-biased table, sign byte already counted in r10d
     jmp .num_digits                           ; enter the table walk
.num_positive:
     lea rax, [rel pow10_table]                ; rax = &pow10_table, dwords 10^0..10^9
.num_digits:
.num_digits_loop:
     inc r10d                                  ; count one digit, every value has >= 1
     cmp r10d, 10                              ; maximum signed 30-bit output width
     jae .num_digits_done                      ; yes: width fixed
     cmp r9d, [rax + r10*SL_POW10_DWORD]       ; magnitude >= 10^digits so far?
     jae .num_digits_loop                      ; yes: one more digit to count
.num_digits_done:
     lea rcx, [rdi + r10]                      ; rcx = one past last number byte
     cmp rcx, r8                               ; past the bounds limit?
     ja .concat_overflow                       ; exact-length overflow still FATAL

.num_emit:
     mov eax, r11d                             ; same signed value as before
     call lg_num_to_str                        ; existing formatter; advances rdi, does not NUL-terminate
     jmp .done                                 ; done
.is_float_val:
; FLOAT: verify the fixed max text width fits, then format
     lea rcx, [rdi + SL_FLOAT_MAX_TEXT]        ; rcx = one past the widest float text
     cmp rcx, r8                               ; past the bounds limit?
     ja .concat_overflow                       ; yes: buffer full, FATAL, skip would silently drop the float
     call sl_float_fmt                         ; rdi = past last decimal digit
     jmp .done                                 ; done
.cons_val:
; CONS, list : write [elem1 elem2 ...] representation to dest buffer
     push rbx                                  ; save non-volatile, carries list rest
     mov ebx, eax                              ; ebx = current list node
     cmp rdi, r8                               ; room for '['?
     jae .concat_overflow                      ; no: FATAL
     mov byte [rdi], '['                       ; opening bracket
     inc rdi                                   ; advance dest
.cons_body:
     mov eax, ebx                              ; eax = current node
     call PopCons                              ; ebx = element, eax = rest
     xchg eax, ebx                             ; eax = element, ebx = rest
     test eax, eax                             ; element is NIL, empty list ?
     jz .cons_nil_elem                         ; yes: inline '[]', recursion writes nothing, element would vanish
     push r8                                   ; save bounds limit across recursive call
     sub rsp, 16                               ; parity pad: keeps the recursive call Win64-conventional, callee entry 8 mod 16, convention, not a requirement, lg_value_to_str's callee chain reaches no Win32 call
     call lg_value_to_str                      ; rdi = past last byte written
     add rsp, 16                               ; drop the parity pad
     pop r8                                    ; restore bounds limit
     jmp .cons_sep                             ; shared separator check
.cons_nil_elem:
     lea rcx, [rdi + SL_NIL_ELEM_BYTES]        ; one past the '[]' pair
     cmp rcx, r8                               ; room for both bytes?
     ja .concat_overflow                       ; no: FATAL, same policy as every other write
     mov word [rdi], '[]'                      ; NIL element renders as '[]', nested empty lists must stay visible in concat, both brackets in one word store; the pair was bounds-checked above
     add rdi, SL_NIL_ELEM_BYTES                ; dest past the pair
.cons_sep:
     test ebx, ebx                             ; more elements?
     jz .cons_close                            ; no: skip separator
     cmp ebx, SL_CONS_TAG                      ; rest a proper cons cell?
     jb .cons_dotted                           ; no, atom/NUMBER : dotted tail
     cmp ebx, SL_FLOAT_TAG                     ; FLOAT band?
     jae .cons_dotted                          ; yes: dotted tail
     cmp rdi, r8                               ; room for ' '?
     jae .concat_overflow                      ; no: FATAL
     mov byte [rdi], ' '                       ; separator
     inc rdi                                   ; advance dest
     jmp .cons_body                            ; continue
.cons_dotted:
     lea rcx, [rdi + SL_DOT_SEP_BYTES]         ; one past "."
     cmp rcx, r8                               ; room for all 3 bytes?
     ja .concat_overflow                       ; no: FATAL, same policy as every write
     mov byte [rdi], ' '                       ; "." before the dotted tail
     mov byte [rdi + SL_DOT_MID_OFF], '.'      ; the dot marks the improper tail
     mov byte [rdi + SL_DOT_END_OFF], ' '      ; trailing space separates tail text
     add rdi, SL_DOT_SEP_BYTES                 ; dest past the separator
     mov eax, ebx                              ; eax = the tail value, rbx dead after this
     push r8                                   ; same parity: RSP 8 mod 16 here before the push
     sub rsp, 16                               ; same parity pad as .cons_body's above, convention, not a requirement, lg_value_to_str's callee chain reaches no Win32 call
     call lg_value_to_str                      ; rdi = past the tail text
     add rsp, 16                               ; drop the parity pad
     pop r8                                    ; restore bounds limit, rbx rides .cons_close's own pop
.cons_close:
     cmp rdi, r8                               ; room for ']'?
     jae .concat_overflow                      ; no: FATAL
     mov byte [rdi], ']'                       ; closing bracket
     inc rdi                                   ; advance dest
     pop rbx                                   ; restore non-volatile
     jmp .done                                 ; done
.concat_overflow:
     FATAL "SectorLogo: string result exceeds 2 MiB concat buffer" ; capacity: a truncated result is silent data corruption, grow the buffer if a real sidecar needs more
.done:
     pop rsi                                   ; restore non-volatile
     ret                                       ; return to caller
; ============================================================================
; sl_fatal_value,rcx = static message, eax = tagged offender, never returns
; Composes "[FATAL] <message>, got '<offender>'" in sl_concat_buf, then dies via
; handle_fatal. Self-aligns RSP, call sites hold either parity.Overwrites r12
; freely, no return path exists.Offender rendering truncates, never FATALs so
; a diagnostic cannot FATAL recursively.
; ============================================================================
sl_fatal_value:
     and rsp, -16                              ; hard self-align: call sites hold either parity and we never return
     sub rsp, 32                               ; shadow space for the handle_fatal call
     mov r12, rcx                              ; message pointer survives the appends, r12 free: we FATAL
     mov r9d, eax                              ; offender FIRST: slfv_append overwrites al, mov al,[rsi], saving it later zeroed the low byte, '42' printed as '0', atoms decoded as a DIFFERENT earlier atom
     lea rdi, [rel sl_concat_buf]              ; build cursor, message lives inside the concat buffer
     lea r8, [rel sl_concat_buf + SL_CONCAT_BUF_LAST - SLFV_SLACK] ; r8 = exclusive copy bound, 2 bytes of slack reserve the closing quote + NUL, the append can never pass it
     lea rsi, [rel str_slf_prefix]             ; "[FATAL] "
     call slfv_append                          ; append the prefix
     mov rsi, r12                              ; static message text
     call slfv_append                          ; append the caller's message
     test r9d, r9d                             ; NIL offender?
     jz .slfv_nil                              ; yes: literal NIL text
     mov ecx, r9d                              ; isolate the tag bits
     and ecx, SL_TAG_MASK                      ; keep tag bits only
     cmp ecx, SL_CONS_TAG                      ; plain CONS, not FLOAT ?
     jne .slfv_type                            ; scalar: type word goes OUTSIDE the quotes
     lea rsi, [rel str_slf_got]                ; ", got '"
     call slfv_append                          ; append the offender introduction
     lea rsi, [rel str_slf_list]               ; literal "list" for a CONS offender: the list's identity is fully described by the word, and rendering it here risks either a 2 MiB capacity FATAL or a nested FATAL from .cons_val's unguarded recursion, either would mask the real diagnosis, which is why lg_value_to_str itself is never handed a CONS from this path
     call slfv_append                          ; append the literal
     jmp .slfv_close                           ; join the closing quote
.slfv_nil:
     lea rsi, [rel str_slf_got]                ; ", got '"
     call slfv_append                          ; append the offender introduction
     lea rsi, [rel str_slf_nil]                ; "NIL"
     call slfv_append                          ; append the literal
     jmp .slfv_close                           ; join the closing quote
.slfv_type:
     test ecx, ecx                             ; bit31 here = FLOAT, plain CONS took the 'list' exit above
     js .slfv_float                            ; FLOAT: float-typed introduction
     test ecx, SL_NUMBER_TAG                   ; bit30 alone = NUMBER?
     jnz .slfv_number                          ; NUMBER: number-typed introduction
     lea rsi, [rel str_slf_gotw]               ; ", got word '", atom: the type word tells atom '7' from NUMBER 7
     call slfv_append                          ; append the typed introduction
     jmp .slfv_render                          ; join the shared formatter
.slfv_number:
     lea rsi, [rel str_slf_gotn]               ; ", got number '"
     call slfv_append                          ; append the typed introduction
     jmp .slfv_render                          ; join the shared formatter
.slfv_float:
     lea rsi, [rel str_slf_gotf]               ; ", got float '"
     call slfv_append                          ; append the typed introduction
.slfv_render:
     mov eax, r9d                              ; restore tagged offender
     test eax, SL_TAG_MASK                     ; tagged, NUMBER/FLOAT ?
     jz .slfv_atom_esc                         ; no: plain atom -> escaped copy
     call lg_value_to_str                      ; number/float text, digits, control bytes impossible
     lea r8, [rel sl_concat_buf + SL_CONCAT_BUF_LAST - SLFV_SLACK] ; RELOAD the bound: lg_value_to_str overwrites volatile r8
     jmp .slfv_close                           ; join the closing quote
.slfv_atom_esc:
     lea rsi, [r13 + rax + SL_ATOM_TEXT_OFF]   ; atom text, rax zero-extended from eax above
.slfv_esc_loop:
     movzx edx, byte [rsi]                     ; next text byte
     test dl, dl                               ; NUL terminator?
     jz .slfv_close                            ; yes: rendering complete
     cmp dl, ' '                               ; control byte, below space ?
     jb .slfv_esc_ctrl                         ; yes: render as \xNN
     cmp dl, 0x7F                              ; DEL, invisible too ?
     je .slfv_esc_ctrl                         ; yes: render as \xNN
     cmp rdi, r8                               ; room for one verbatim byte?
     jae .slfv_close                           ; no: truncate, slack keeps quote+NUL in bounds
     mov [rdi], dl                             ; store the byte verbatim
     inc rdi                                   ; advance build cursor
     inc rsi                                   ; advance source cursor
     jmp .slfv_esc_loop                        ; next byte
.slfv_esc_ctrl:
     lea rcx, [rdi + SLFV_ESC_BYTES]           ; cursor past a full 4-byte escape
     cmp rcx, r8                               ; complete escape fits below the bound?
     jae .slfv_close                           ; no: truncate rather than emit a partial escape
     mov word [rdi], 0x785C                    ; little endian: backslash then 'x'
     lea rcx, [rel hex_chars]                  ; existing uppercase hex table, v7_data.inc:407
     mov eax, edx                              ; original input byte
     shr eax, 4                                ; high nibble, 0..15
     mov al, [rcx + rax]                       ; table digit
     mov [rdi + SLFV_ESC_HI_OFF], al           ; store the high hex digit
     mov eax, edx                              ; original input byte
     and eax, 15                               ; low nibble
     mov al, [rcx + rax]                       ; table digit
     mov [rdi + SLFV_ESC_LO_OFF], al           ; store the low hex digit
     add rdi, SLFV_ESC_BYTES                   ; cursor past the escape
     inc rsi                                   ; advance source cursor
     jmp .slfv_esc_loop                        ; next byte
.slfv_close:
     lea rsi, [rel str_slf_quote]              ; "'"
     call slfv_append                          ; append the closing quote, bounded: never past the slack
     mov byte [rdi], 0                         ; NUL-terminate the composed diagnostic, always in bounds: rdi <= bound <= last-1
     lea rcx, [rel sl_concat_buf]              ; the composed message, a too-long diagnostic is TRUNCATED here, never an overrun; we still FATAL via handle_fatal, non-recursively
     call handle_fatal                         ; print + catch-and-unwind / TerminateProcess, never returns
slfv_append:
.slfv_a:
     cmp rdi, r8                               ; cursor reached the copy bound?
     jae .slfv_a_done                          ; yes: stop copying, truncated, no overrun
     mov al, [rsi]                             ; load next source byte
     test al, al                               ; end of source?
     jz .slfv_a_done                           ; yes: cursor stays past the last copied byte
     mov [rdi], al                             ; store the byte
     inc rsi                                   ; advance source cursor
     inc rdi                                   ; advance build cursor
     jmp .slfv_a                               ; next source byte
.slfv_a_done:
     ret                                       ; rdi = past the copied text
; ---- %undef cleanup: file-local defines leave scope here, house convention ----
%undef SL_DOT_SEP_BYTES
%undef SL_DOT_MID_OFF
%undef SL_DOT_END_OFF
%undef SL_NIL_ELEM_BYTES
%undef SL_NUMBER_MAX_TEXT
%undef SL_DIV10_MAGIC
%undef SL_DIV10_SHIFT
