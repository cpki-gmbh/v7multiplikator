; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; Stack slot offsets after SF_START prologue, r9 ABI home used for vararg[0]
; Stack varargs start at [rbp+48], 5th overall param .
%define SPR_VARG_BASE 40                       ; base offset for unified vararg fetch, rbp+40 + n*8 covers all
%define SPR_SWAR_WINDOW 8                      ; SWAR window width in bytes, one qword per bulk-copy step
%define BUF_SIZE_MINUS1 1                      ; constant for buf_size-1 lea offset
%define SIZEOF_PTR 8                           ; pointer size in bytes, x64
%define F32_2POW63_BITS 0x5F000000             ; bit pattern of 2^63f, int-conversion validity bound
%define F32_1E6_BITS 0x49742400                ; bit pattern of 1000000.0f, fraction scaling
%define F32_FRAC_DIGITS 6                      ; fractional digits emitted before trimming
%define PREV_CHAR_OFF 1                        ; byte offset back from out cursor, last char
%define DIV10_MAGIC 0xCCCCCCCD                 ; unsigned div-by-10 magic:, n*MAGIC >>35 == n/10 for n < 2^32, Hacker's Delight
%define DIV10_MAGIC64 0xCCCCCCCCCCCCCCCD       ; unsigned div-by-10 magic, 64-bit: mulhi,r8,MAGIC64 >>3 == r8/10 for r8 < 2^63, Hacker's Delight 10-17
%define LEA_SCALE5 4                           ; lea scale in [rax + rax*4]: base + scaled = q*5
; SWAR loop-invariant constants, RIP-relative: replaces per-iteration mov r8, imm64 reloads
     section .rdata
swar_rep1:
     dq 0x0101010101010101                     ; per-byte 1, SWAR haszero borrow seed, sub minuend
swar_hi80:
     dq 0x8080808080808080                     ; per-byte 0x80, SWAR haszero result mask
swar_pctmask:
     dq 0xDADADADADADADADA                     ; per-byte ~0x25, parse detector: zeroes on NUL and '%'
     section .text
; ============================================================================
; v7sprintf
; ============================================================================
     global v7sprintf                          ; export entry point
     SF_START v7sprintf, rbx, rdi, r12, r13, r15 ; save non-volatile regs, rdi: buf_end cache reg below
     SF_ARRAY digit_buf, 32                    ; 32-byte digit buffer for %d
     SF_QWORD sf_buf_end                       ; buffer + buf_size - 1, last writable byte
     SF_QWORD sf_buf_base                      ; saved buffer base, return value computed at .done as r12 - base
     SF_COMMIT                                 ; allocate shadow space, >=32 bytes
     mov r12, rcx                              ; r12 = buffer base, write cursor
     mov sf_buf_base, rcx                      ; park buffer base for the .done return-value computation
     mov edx, edx                              ; zero-extend buf_size into rdx
; compute buf_end = buffer + buf_size - 1, callers guarantee buf_size >= 1
     lea rax, [rcx + rdx - BUF_SIZE_MINUS1]    ; rax = buffer + buf_size - 1
     mov sf_buf_end, rax                       ; park buf_end in frame slot
     mov rdi, rax                              ; rdi = bound cache for every copy loop; written nowhere else in this leaf
     mov r13, r8                               ; r13 = format string, 3rd arg
     xor ebx, ebx                              ; vararg counter starts at 0
; Save r9, vararg[0] in ABI home slot for fetch_next_arg
     mov [rbp + SPR_VARG_BASE], r9             ; vararg[0] saved in r9 ABI home slot

%macro SPR_BOUND_CHECK 0
     cmp r12, sf_buf_end                       ; room for one more char?, direct frame-slot compare
     jae .done                                 ; no -> stop
%endmacro
; ---- helper: fetch current variadic arg into rax and advance the index ----
; Inlines fetch_next_arg: vararg[n] = [rbp + SPR_VARG_BASE + n*SIZEOF_PTR]

%macro SPR_FETCH_NEXT_ARG 0
     mov rax, [rbp + SPR_VARG_BASE + rbx*SIZEOF_PTR] ; rax = vararg value
     inc ebx                                   ; advance vararg index
%endmacro

%macro SPR_FETCH_NEXT_ARG32 0
     mov eax, [rbp + SPR_VARG_BASE + rbx*SIZEOF_PTR] ; eax = vararg int32, zero-extends rax
     inc ebx                                   ; advance vararg index
%endmacro
.parse_swar_top:
     mov rax, r13                              ; rax = format cursor, page-gate probe
     and eax, 0xFFF                            ; offset within the current 4K page
     cmp rax, 0xFF8                            ; would the 8-byte window cross into the next page?
     ja .parse_loop                            ; yes -> byte loop, never read past a page edge
     lea rax, [r12 + SPR_SWAR_WINDOW]          ; rax = output cursor + one full window
     cmp rax, rdi                              ; window end past the last writable byte?
     ja .parse_loop                            ; yes -> byte loop
     mov rcx, [r13]                            ; pristine window copy, kept for the store
     mov rax, rcx                              ; rax = w
     and rax, [rel swar_pctmask]               ; v = w & ~0x25: zero byte <=> NUL or '%', or rare subset byte
     mov rdx, rax                              ; rdx = v
     sub rdx, [rel swar_rep1]                  ; rdx = v - rep,1, borrows into a zero byte saturate it to 0xFF
     andn rdx, rax, rdx                        ; rdx = ~v & (v-rep,1), BMI1; rax is dead after this read-only use
     test rdx, [rel swar_hi80]                 ; any stop byte in the window?
     jnz .parse_loop                           ; yes -> byte loop handles it exactly
     mov [r12], rcx                            ; store the ORIGINAL 8 literal bytes
     add r12, 8                                ; advance output cursor by the window
     add r13, 8                                ; advance format cursor by the window
     jmp .parse_swar_top                       ; next window
.parse_loop:
     mov al, byte [r13]                        ; load format byte, AL-only scanner
     test al, al                               ; end of format string?
     jz .done                                  ; yes -> null-terminate and return
     cmp al, '%'                               ; start of format specifier?
     je .specifier                             ; yes -> parse what follows
; plain character: copy straight to output buffer, if room
     SPR_BOUND_CHECK                           ; room for one more char?
     mov [r12], al                             ; store literal char to buffer
     inc r12                                   ; advance output cursor
     inc r13                                   ; advance format cursor
     jmp .parse_swar_top                       ; continue parsing, restart the SWAR head
.specifier:
     inc r13                                   ; skip past '%'
     mov al, byte [r13]                        ; load specifier byte, only AL used
     test al, al                               ; premature end of string?
     jz .done                                  ; yes -> stop, trailing % with no specifier
     cmp al, 's'                               ; string pointer?
     je .string                                ; yes -> %s handler
     cmp al, 'd'                               ; decimal integer?
     je .decimal                               ; yes -> %d handler
     cmp al, 'f'                               ; float32?
     je .float                                 ; yes -> %f handler
     cmp al, 'c'                               ; single character?
     je .char                                  ; yes -> %c handler
     cmp al, '%'                               ; literal percent sign?
     je .percent                               ; yes -> %% handler
; unknown specifier: emit '%' then the unknown char as-is, if room
     SPR_BOUND_CHECK                           ; room for '%'?
     mov byte [r12], '%'                       ; write leading '%'
     inc r12                                   ; advance output cursor
     SPR_BOUND_CHECK                           ; room for unknown char?
     mov [r12], al                             ; write the specifier char verbatim, al survives the macro
     inc r12                                   ; advance output cursor
     inc r13                                   ; advance past specifier
     jmp .parse_loop                           ; back to main parse
.percent:
     SPR_BOUND_CHECK                           ; room for '%'?
     mov byte [r12], '%'                       ; emit literal '%'
     inc r12                                   ; advance output cursor
     inc r13                                   ; advance past second '%'
     jmp .parse_loop                           ; continue parsing
.char:
     SPR_FETCH_NEXT_ARG                        ; rax = vararg value, promoted to 64-bit
     SPR_BOUND_CHECK                           ; room for one char?
     mov [r12], al                             ; write low byte as output character
     inc r12                                   ; advance output cursor
     inc r13                                   ; advance past 'c' specifier
     jmp .parse_loop                           ; back to main parse
.float:
     SPR_FETCH_NEXT_ARG                        ; rax = vararg; eax = float32 bits
     mov r10d, eax                             ; keep a copy of the raw bits
     shr r10d, 31                              ; r10d = sign bit, 0 or 1
     btr eax, 31                               ; sign-cleared binary32 magnitude
     cmp eax, SL_F32_EXP_MASK                  ; exponent field all ones?
     jae .float_nan_inf                        ; yes -> NaN or Inf class
     cmp eax, F32_2POW63_BITS                  ; |x| >= 2^63, no 64-bit int ?
     jae .float_range_fatal                    ; yes -> out of contract: name it, do NOT fabricate "inf" for a finite value
     movd xmm0, eax                            ; xmm0 = |x| as float, sign-cleared bits
     cvttss2si r8, xmm0                        ; r8 = integer part, truncated
     cvtsi2ss xmm1, r8                         ; xmm1 = float,integer part
     subss xmm0, xmm1                          ; xmm0 = fraction in [0,1
     mov r11d, F32_1E6_BITS                    ; 1000000.0f bit pattern
     movd xmm2, r11d                           ; xmm2 = 1e6, no .rdata constant needed
     mulss xmm0, xmm2                          ; fraction * 1e6
     cvtss2si r9d, xmm0                        ; rounded scaled fraction, MXCSR nearest, always < 2^24 fits 32-bit
     cmp r9d, 1000000                          ; rounding carried to a whole?
     jl .float_frac_ok                         ; no -> 0..999999 is a valid fraction
     xor r9d, r9d                              ; yes -> fraction becomes .000000
     inc r8                                    ; carry one into the integer part
.float_frac_ok:
; zero-collapse: a result of exactly zero prints "0", never "-0"
     test r8, r8                               ; integer part nonzero?
     jnz .float_emit_sign                      ; yes -> full emission
     test r9d, r9d                             ; fraction nonzero?
     jnz .float_emit_sign                      ; yes -> full emission
     SPR_BOUND_CHECK                           ; room for '0'?
     mov byte [r12], '0'                       ; single '0' for any zero magnitude
     inc r12                                   ; advance output cursor
     jmp .float_next                           ; done with this specifier
.float_emit_sign:
     test r10d, r10d                           ; original value negative?
     jz .float_emit_int                        ; no -> skip the minus sign
     SPR_BOUND_CHECK                           ; room for '-'?
     mov byte [r12], '-'                       ; emit '-'
     inc r12                                   ; advance output cursor
.float_emit_int:
; integer part digits into digit_buf, LSD-first, reversed
     lea r15, digit_buf                        ; r15 = digit buffer base
     mov r11, r15                              ; r11 = digit write cursor, same base, register copy
     test r8, r8                               ; integer part zero?
     jnz .float_int_digits                     ; no -> load the reciprocal and extract digits
     mov byte [r11], '0'                       ; single zero digit
     inc r11                                   ; advance digit cursor
     jmp .float_int_done                       ; straight to the shared tail
.float_int_digits:
     mov r10, DIV10_MAGIC64                    ; r10 = 64-bit div-by-10 reciprocal, setup, NOT on the loop back-edge
.float_int_loop:
     mov rax, r8                               ; rax = n, the one-operand `mul` takes its multiplicand from RAX, not rdx
     mul r10                                   ; rdx:rax = rax * DIV10_MAGIC64
     mov rcx, rdx                              ; rcx = high half of the product
     shr rcx, 3                                ; rcx = q = n / 10
     lea rdx, [rcx + rcx*LEA_SCALE5]           ; rdx = q * 5
     add rdx, rdx                              ; rdx = q * 10
     sub r8, rdx                               ; r8 = digit = n - 10q, 0-9
     add r8b, '0'                              ; digit to ASCII
     mov [r11], r8b                            ; store into digit buffer
     inc r11                                   ; advance digit cursor
     mov r8, rcx                               ; n = q for the next iteration
     test rcx, rcx                             ; more integer digits?
     jnz .float_int_loop                       ; yes -> keep extracting
.float_int_done:
     mov r8, r11                               ; r8 = end of integer digits
.float_emit_frac:
; exactly F32_FRAC_DIGITS fraction digits, zero-padded into the buffer,
; also LSD-first, the reversed copy below restores both parts at once.
     mov r10d, DIV10_MAGIC                     ; unsigned div-by-10 magic, r10 dead as sign; dividend < 2^24 qualifies for the 32-bit magic
     mov ecx, F32_FRAC_DIGITS                  ; digit emission counter
; eax = n = scaled fraction, from cvtss2si above; < 2^24 ; n rides in eax, quotient carried by the shr
     mov eax, r9d                              ; one-time: n into the multiply register, zero-extends; imul reads the full rax
.float_frac_loop:
     mov r9d, eax                              ; r9d = n, saved: the imul below overwrites eax with the product
     imul rax, r10                             ; rax = n * magic, high-half setup for the /10
     shr rax, 35                               ; rax = q = n / 10, exact for n < 2^32 ; stays in eax for the next iteration
     lea edx, [rax + rax*LEA_SCALE5]           ; edx = q * 5
     add edx, edx                              ; edx = q * 10
     sub r9d, edx                              ; r9d = digit = n - 10q, LSD-first, 0-9
     add r9d, '0'                              ; digit to ASCII
     mov [r11], r9b                            ; store into digit buffer
     inc r11                                   ; advance digit cursor
     dec ecx                                   ; one fewer digit to emit
     jnz .float_frac_loop                      ; emit all 6, zero-padded
     mov r9, r11                               ; r9 = saved end-of-fraction cursor, reuse: fraction digits done
.float_copy_out:
; integer digits live in [r15, r8, walk MSD-first, BEFORE the dot
     mov r10, r12                              ; r10 = output cursor snapshot, taken AFTER the possible '-' emission, so r12 == r10 <=> this %f wrote nothing at all
     mov r11, r8                               ; r11 = cursor, one past last int digit
; rdi = output bound, cached in the prologue, no per-digit reload
.float_copy_int:
     cmp r11, r15                              ; back at digit buffer base?
     jbe .float_copy_dot                       ; yes -> integer part done, emit the dot
     dec r11                                   ; move to the last-written digit
     mov dl, [r11]                             ; load the digit
     cmp r12, rdi                              ; room in output?
     jae .float_copy_done                      ; no -> stop
     mov [r12], dl                             ; write to output
     inc r12                                   ; advance output cursor
     jmp .float_copy_int                       ; keep copying in reverse
.float_copy_dot:
     cmp r12, rdi                              ; room for '.'?
     jae .float_copy_done                      ; no -> stop
     mov byte [r12], '.'                       ; decimal point between the parts
     inc r12                                   ; advance output cursor
; fraction digits live in [r8, saved end, walk MSD-first, AFTER the dot
     mov r11, r9                               ; r11 = restored end-of-fraction cursor
.float_copy_frac:
     cmp r11, r8                               ; back at the int/frac boundary?
     jbe .float_trim                           ; yes -> go trim trailing zeros
     dec r11                                   ; move to the last-written digit
     mov dl, [r11]                             ; load the fraction digit
     cmp r12, rdi                              ; room in output?
     jae .float_copy_done                      ; no -> stop
     mov [r12], dl                             ; write to output
     inc r12                                   ; advance output cursor
     jmp .float_copy_frac                      ; keep copying in reverse
.float_copy_done:
     cmp r12, r10                              ; THIS %f wrote nothing at all, not even the dot ?
     je .float_next                            ; yes -> skip backward read, a prior specifier's '.' must survive
     cmp byte [r12 - PREV_CHAR_OFF], '.'       ; last char a bare dot?
     jne .float_next                           ; no -> done
     dec r12                                   ; drop the dangling dot
     jmp .float_next                           ; done with this specifier
.float_trim:
; trim trailing '0' digits, then a bare trailing '.', no padding ever
     cmp byte [r12-PREV_CHAR_OFF], '0'         ; last written char a zero?
     jne .float_trim_dot                       ; no -> check for bare dot
     dec r12                                   ; drop the zero
     jmp .float_trim                           ; keep trimming
.float_trim_dot:
     cmp byte [r12-PREV_CHAR_OFF], '.'         ; fraction fully trimmed -> bare dot?
     jne .float_next                           ; no -> done
     dec r12                                   ; drop the dot, "3." -> "3"
.float_next:
     inc r13                                   ; advance past 'f' specifier
     jmp .parse_loop                           ; back to main parse
.float_nan_inf:
     test eax, 0x7FFFFF                        ; mantissa nonzero -> NaN?
     jnz .float_nan                            ; yes -> "nan"
; mantissa zero -> signed infinity, falls through
.float_inf:
     test r10d, r10d                           ; negative infinity?
     jz .float_inf_body                        ; no -> plain "inf"
     SPR_BOUND_CHECK                           ; room for '-'?
     mov byte [r12], '-'                       ; write '-'
     inc r12                                   ; advance output cursor
.float_inf_body:
     SPR_BOUND_CHECK                           ; room for 'i'?
     mov byte [r12], 'i'                       ; write 'i'
     inc r12                                   ; advance output cursor
     SPR_BOUND_CHECK                           ; room for 'n'?
     mov byte [r12], 'n'                       ; write 'n'
     inc r12                                   ; advance output cursor
     SPR_BOUND_CHECK                           ; room for 'f'?
     mov byte [r12], 'f'                       ; write 'f'
     inc r12                                   ; advance output cursor
     jmp .float_next                           ; done with this specifier
.float_nan:
     SPR_BOUND_CHECK                           ; room for 'n'?
     mov byte [r12], 'n'                       ; write 'n'
     inc r12                                   ; advance output cursor
     SPR_BOUND_CHECK                           ; room for 'a'?
     mov byte [r12], 'a'                       ; write 'a'
     inc r12                                   ; advance output cursor
     SPR_BOUND_CHECK                           ; room for 'n'?
     mov byte [r12], 'n'                       ; write 'n'
     inc r12                                   ; advance output cursor
     jmp .float_next                           ; done with this specifier
.float_range_fatal:
     FATAL "v7sprintf: %f magnitude >= 2^63 (finite but the integer part exceeds int64)" ; contract: no caller formats a value this large
.string:
     SPR_FETCH_NEXT_ARG                        ; rax = string pointer from varargs
     mov r15, rax                              ; r15 = source string
     test r15, r15                             ; NULL pointer from caller?
     jz .sprintf_null_fatal                    ; yes: contract violation, no ",null " fallback
.copy_string:
.copy_swar_top:
     mov rax, r15                              ; rax = source cursor, page-gate probe
     and eax, 0xFFF                            ; offset within the current 4K page
     cmp rax, 0xFF8                            ; would the 8-byte window cross a page edge?
     ja .copy_string_loop                      ; yes -> byte loop
     lea rax, [r12 + SPR_SWAR_WINDOW]          ; rax = output cursor + one full window
     cmp rax, rdi                              ; window end past the last writable byte?
     ja .copy_string_loop                      ; yes -> byte loop
     mov rcx, [r15]                            ; pristine window copy, kept for the store
     mov rdx, rcx                              ; rdx = w
     sub rdx, [rel swar_rep1]                  ; rdx = w - rep,1
     andn rdx, rcx, rdx                        ; rdx = ~w & (w-rep,1), BMI1, two instructions saved
     test rdx, [rel swar_hi80]                 ; NUL anywhere in the window?
     jnz .copy_string_loop                     ; yes -> byte loop stops precisely at the NUL
     mov [r12], rcx                            ; store the 8 string bytes
     add r12, 8                                ; advance output cursor by the window
     add r15, 8                                ; advance source cursor by the window
     jmp .copy_swar_top                        ; next window
.copy_string_loop:
     mov al, byte [r15]                        ; next output byte, only AL used
     test al, al                               ; end of string, null terminator ?
     jz .string_done                           ; yes -> done copying
     cmp r12, rdi                              ; room in output?, register compare against the rdi bound cache; rax untouched so al survives for the store
     jae .string_done                          ; no -> stop
     mov [r12], al                             ; write char to output buffer
     inc r12                                   ; advance output cursor
     inc r15                                   ; advance source pointer
     jmp .copy_string_loop                     ; next source character
.string_done:
     inc r13                                   ; advance past 's' specifier
     jmp .parse_loop                           ; back to main parse
.sprintf_null_fatal:
     FATAL "v7sprintf: %s argument is NULL"
.decimal:
     SPR_FETCH_NEXT_ARG32                      ; rax = vararg int32, zero-extended, caller residue above bit 31 discarded
; Use named digit buffer, 32 bytes, plenty for 10 digits + sign
     lea r15, digit_buf                        ; r15 = digit buffer base
     mov r11, r15                              ; r11 = digit write cursor, same base, register copy; grows upward
     mov ecx, eax                              ; ecx = original value, sign test at .decimal_write via bit 31
     neg eax                                   ; eax = -value; SF=1 iff result negative
     cmovs eax, ecx                            ; value was positive -> keep original; else keep |value|
     test eax, eax                             ; value == zero?
     jnz .decimal_loop_setup                   ; no -> extract digits normally
     mov byte [r11], '0'                       ; special case: emit single '0'
     inc r11                                   ; advance cursor past the zero digit
     jmp .decimal_write                        ; go write reversed digits to output
.decimal_loop_setup:
     mov r8d, DIV10_MAGIC                      ; unsigned div-by-10 magic, r8 free in this region; exact for all n < 2^32, INT_MIN's negated dividend is exactly 2^31, inside the valid range
.decimal_loop:
     mov r9d, eax                              ; r9d = n, current dividend; also holds the digit after the subtract
     imul rax, r8                              ; rax = n * magic, high-half setup for the /10
     shr rax, 35                               ; rax = q = n / 10, exact for n < 2^32
     lea edx, [rax + rax*LEA_SCALE5]           ; edx = q * 5
     add edx, edx                              ; edx = q * 10
     sub r9d, edx                              ; r9d = digit = n - 10q, 0-9
     add r9d, '0'                              ; digit to ASCII
     mov [r11], r9b                            ; store digit in temp buffer
     inc r11                                   ; advance digit buffer cursor
     test eax, eax                             ; any more digits left?, quotient carried in eax
     jnz .decimal_loop                         ; yes -> continue extracting
.decimal_write:
     test ecx, ecx                             ; original value negative?, SF = bit 31 of the original
     jns .decimal_reverse                      ; non-negative -> skip minus sign
     mov byte [r11], '-'                       ; append minus sign after digits
     inc r11                                   ; advance past the minus sign
.decimal_reverse:
.reverse_loop_iter:
     dec r11                                   ; move cursor left to last-written char
     mov dl, [r11]                             ; load digit/sign from temp buffer
     cmp r12, rdi                              ; room in output?
     jae .decimal_done                         ; no -> stop
     mov [r12], dl                             ; write to output buffer
     inc r12                                   ; advance output cursor
     cmp r11, r15                              ; reached start of digit buffer?
     ja .reverse_loop_iter                     ; no -> keep copying in reverse
.decimal_done:
     inc r13                                   ; advance past 'd' specifier
     jmp .parse_loop                           ; back to main parse
.done:
     mov byte [r12], 0                         ; null-terminate output buffer
     mov rax, r12                              ; rax = write cursor, one past last char
     sub rax, sf_buf_base                      ; rax = chars written, excl null
     SF_END rbx, rdi, r12, r13, r15            ; restore non-volatiles and return
; %undef cleanup: file-local defines leave scope here, house convention --
%undef SPR_VARG_BASE
%undef SPR_SWAR_WINDOW
%undef BUF_SIZE_MINUS1
%undef SIZEOF_PTR
%undef F32_2POW63_BITS
%undef F32_1E6_BITS
%undef F32_FRAC_DIGITS
%undef PREV_CHAR_OFF
%undef DIV10_MAGIC
%undef DIV10_MAGIC64
%undef LEA_SCALE5
%unmacro SPR_BOUND_CHECK 0                     ; bounds-check macro, call sites above
%unmacro SPR_FETCH_NEXT_ARG 0                  ; vararg-fetch macro, call sites above
%unmacro SPR_FETCH_NEXT_ARG32 0                ; 32-bit vararg-fetch macro, call sites above
