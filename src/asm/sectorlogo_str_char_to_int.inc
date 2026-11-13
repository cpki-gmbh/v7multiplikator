; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; --- char-to-int frame layout, alignment only ---
%define SL_CTI_FRAME 40                        ; frame bytes only, no pushes : EvalPrimary's dispatch always enters handlers at RSP 8 mod 16; 40 = 8 mod 16 flips that to 0 mod 16 at the EvalNextArg call, the sibling push-frame primitives encode the same invariant
sl_str_char_to_int:
     sub rsp, SL_CTI_FRAME                     ; alignment frame only, rsi/rdi never used; no eval-chain caller holds them live
     call EvalNextArg                          ; eax = the string arg
     mov ecx, eax                              ; rcx = atom offset, zero-extended: the two memory operands below are 64-bit indexed, and EvalNextArg's chain returns eax with no normalization of its own, .infix_done / .add_done are bare pop+ret; test/cmp write no register, so bits 63:32 would survive
; tag check: FATAL if NUMBER/FLOAT/CONS, NIL allowed: heap[1] is 0 from zero-init, so NIL reads as 0
     test eax, SL_TAG_MASK                     ; any tag bits set, CONS/NUMBER/FLOAT ?
     jnz sl_cti_notatom                        ; yes: FATAL
     jrcxz .cti_nil_ok                         ; NIL, offset 0 ? rcx was zero-extended from eax above, fall straight to the text read
     cmp byte [r13 + rcx], SL_TYPE_STRING      ; atom type byte: WORD atoms are rejected here
     jne sl_cti_notatom                        ; not a STRING atom: FATAL naming the class
.cti_nil_ok:
     movzx eax, byte [r13 + rcx + SL_ATOM_TEXT_OFF] ; first character byte, NIL -> heap[1] = 0 from the zero page; SL_ATOM_TEXT_OFF=1 skips the type byte
     add rsp, SL_CTI_FRAME                     ; restore frame
     or eax, SL_NUMBER_TAG                     ; tag inline: movzx already yields 0..255, so MakeNum's payload mask is provably redundant
     ret                                       ; return tagged NUMBER,code, or is last, so the caller sees the tag-OR's ZF, always 0: the OR can never yield zero
sl_cti_notatom:
     FATAL_VALUE "Number requires a string atom argument", eax ; name the offender: renders got word 'hello' / got number '42'. Name the LIVE primitive, atom "number", cheat sheet ", Number \"s\" "
%undef SL_CTI_FRAME
