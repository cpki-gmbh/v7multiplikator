; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define SL_CHAR_LO_BIAS 1                      ; gate bias: n-1 moves the low bound to 0 so ONE unsigned compare covers 1..255 and rejects 0, negatives, and >255
%define SL_CHAR_N1_MAX 254                     ; upper bound on, n-1, i.e. the largest accepted n is 255, do NOT raise this: 255 would admit Character 256, whose ax = 0x0100 stores a NUL as the character and interns the empty-string atom
sl_str_int_to_char:
     push rsi                                  ; save non-volatile, Intern requires caller rsi preserved
     call EvalNextArg                          ; eax = the number arg
     mov r10d, eax                             ; preserve the tagged offender: NumVal returns the raw value in eax and never touches r10d, so this survives for the range FATAL
     call NumVal                               ; eax = raw integer value
     lea ecx, [rax - SL_CHAR_LO_BIAS]          ; ecx = n - 1, so ONE unsigned compare below has its low bound at 1
     cmp ecx, SL_CHAR_N1_MAX                   ; outside 1..255?, unsigned: n=0 -> 0xFFFFFFFF, any negative -> large; both rejected
     ja sl_itc_range                           ; yes: FATAL naming the rejected value
     mov byte [rel sl_concat_buf], SL_TYPE_STRING ; type byte for Intern
     mov word [rel sl_concat_buf + SL_ATOM_TEXT_OFF], ax ; the character + NUL, two-byte store
     lea rsi, [rel sl_concat_buf]              ; source for Intern
     call Intern                               ; real call, NOT jmp: a live push is on the stack ; eax = interned 1-char atom
     pop rsi                                   ; restore the caller's rsi
     ret                                       ; return the atom
sl_itc_range:
     FATAL_VALUE "Character requires a NUMBER in the range 1..255", r10d ; name the offending value
; ---- %undef cleanup: file-local defines leave scope here, house convention ----
%undef SL_CHAR_LO_BIAS                         ; n-1 range-gate bias, used only at :12
%undef SL_CHAR_N1_MAX                          ; upper bound on n-1 = largest accepted n, used only at :13
