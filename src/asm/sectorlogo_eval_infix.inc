; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; --- file-local constants, removed at the end of this file ---
%define SL_TRACE_EXPR_RESULT_OFF 8             ; rsp offset of the saved expression result inside EvalNextArg's trace block: the block does `push rax` then `push rbx`, so the result sits one qword above rsp at the restore point

%macro CHAIN_PEEK 1
     mov ecx, r12d                             ; ecx = stream cell, non-NIL: tested by the caller
     btr ecx, SL_CONS_BIT                      ; ecx = raw cons index
     mov %1, [r13 + rcx*CONS_SCALE + CONS_BASE_OFF] ; %1 =, car stream = the head just peeked, inlined Car: no call, no dead band guard
%endmacro
EvalMulChain:
.loop:
     test r12d, r12d                           ; stream is NIL?
     jz .done                                  ; yes: done
     CHAIN_PEEK r11d                           ; r11d = candidate operator; eax keeps the accumulator
     cmp r11d, [rel atom_sl_mul_sym]           ; * ?
     je .mul                                   ; yes: multiply
     cmp r11d, [rel atom_sl_div_sym]           ; / ?
     je .div                                   ; yes: divide
     cmp r11d, [rel atom_sl_mod_sym]           ; % ?
     je .mod                                   ; yes: modulo
     ret                                       ; not an operator: eax already holds the accumulator
.mul:
     INFIX_EVAL_RHS                            ; consume operator, eval rhs, lhs pushed
     pop rdx                                   ; edx = lhs, tagged
     call sl_promote_pair                      ; edx=1: floats | edx=0: ints, eax,ecx
     test edx, edx                             ; float operands?
     jnz .fmul                                 ; yes: SSE float multiply
     cdqe                                      ; rax = lhs, sign-extended, so the multiply below is exact
     movsxd rcx, ecx                           ; rcx = rhs, sign-extended
     imul rax, rcx                             ; rax = exact 64-bit product, both operands are 30-bit payloads, so it fits in 60 bits
     NUM_RANGE_CHECK rax, "Multiply: integer result does not fit the 30-bit payload range" ; out of range? FATAL, a 32-bit-only check would pass 65536*65536 as 0
     call MakeNum                              ; wrap as NUMBER
     jmp .loop                                 ; check for more */%
.fmul:
     vmulss xmm0, xmm0, xmm1                   ; xmm0 = lhs * rhs
     call MakeFloat                            ; wrap as FLOAT
     jmp .loop                                 ; check for more */%
.div:
     INFIX_EVAL_RHS                            ; consume operator, eval rhs, lhs pushed
     pop rdx                                   ; edx = lhs, tagged
     call sl_promote_pair                      ; edx=1: floats | edx=0: ints, eax,ecx
     test edx, edx                             ; float operands?
     jnz .fdiv                                 ; yes: SSE float divide, IEEE inf on /0
     test ecx, ecx                             ; divide by zero?
     jz .div_zero                              ; yes: FATAL
     cdq                                       ; sign-extend eax into edx:eax, |lhs| <= 2^29 and |rhs| >= 1, so idiv cannot trap and the quotient always fits 32 bits
     idiv ecx                                  ; eax = lhs / rhs
     cdqe                                      ; rax = exact quotient
     NUM_RANGE_CHECK rax, "Divide: integer result does not fit the 30-bit payload range" ; lhs=-2^29, rhs=-1 -> +2^29, ONE PAST the payload maximum: MakeNum's mask would turn it back into -2^29 and the script would read the negation of the answer, same wording as the prefix Divide, so both spellings name the identical cause
     call MakeNum                              ; wrap as NUMBER
     jmp .loop                                 ; check for more */%
.fdiv:
     vdivss xmm0, xmm0, xmm1                   ; xmm0 = lhs / rhs
     call MakeFloat                            ; wrap as FLOAT
     jmp .loop                                 ; check for more */%
.div_zero:
     FATAL "SectorLogo: division by zero"      ; divisor was zero: abort
.mod:
     INFIX_EVAL_RHS                            ; consume operator, eval rhs, lhs pushed
     pop rdx                                   ; edx = lhs, tagged
     call sl_promote_pair                      ; edx=1: floats | edx=0: ints, eax,ecx
     test edx, edx                             ; float operands?
     jnz .fmod_                                ; yes: x87 float remainder
     test ecx, ecx                             ; modulo by zero?
     jz .mod_zero                              ; yes: FATAL
     cdq                                       ; sign-extend eax into edx:eax, no INT_MIN/-1 guard: a NUMBER is never 2^31, and the remainder satisfies |edx| < |rhs| <= 2^29 so it is always representable
     idiv ecx                                  ; eax = quotient, edx = remainder
     mov eax, edx                              ; eax = remainder
     call MakeNum                              ; wrap as NUMBER
     jmp .loop                                 ; check for more */%
.fmod_:
     call sl_fmod                              ; xmm0 = fmod,lhs, rhs ; FATAL rhs==0
     call MakeFloat                            ; wrap as FLOAT
     jmp .loop                                 ; check for more */%
.mod_zero:
     FATAL "SectorLogo: modulo by zero"        ; divisor was zero: abort
.done:
     ret                                       ; return to caller
EvalAddChain:
     push rbx                                  ; save non-volatile + align stack
.add_loop:
     test r12d, r12d                           ; stream is NIL?
     jz .add_done                              ; yes: no more +- ops
; peek at next stream element without consuming it
     CHAIN_PEEK r11d                           ; r11d = candidate operator; eax keeps the accumulator
     cmp r11d, [rel atom_sl_plus]              ; + ?
     je .add_iadd                              ; yes: add
     cmp r11d, [rel atom_sl_minus_sym]         ; - ?
     je .add_isub                              ; yes: subtract
; not + or -: done with additive level
     pop rbx                                   ; restore non-volatile
     ret                                       ; return to caller, eax already holds the accumulator
.add_iadd:
     INFIX_ADD_PROMOTED                        ; consume +, eval rhs, promote pair
     test edx, edx                             ; float operands?
     jnz .add_fiadd                            ; yes: SSE float add
     add eax, ecx                              ; eax = lhs + rhs, a sum of two 30-bit values always fits in 32 bits
     cdqe                                      ; rax = exact result
     NUM_RANGE_CHECK rax, "Add: integer result does not fit the 30-bit payload range" ; out of range? FATAL. Without this a positive sum that leaves the payload reads back NEGATIVE: 400000000+400000000 = 800000000 has bit 29 set, so MakeNum masks to 30 bits and NumVal sign-extends to -273741824
     call MakeNum                              ; wrap as NUMBER
     jmp .add_loop                             ; check for more +-
.add_fiadd:
     vaddss xmm0, xmm0, xmm1                   ; xmm0 = lhs + rhs
     call MakeFloat                            ; wrap as FLOAT
     jmp .add_loop                             ; check for more +-
.add_isub:
     INFIX_ADD_PROMOTED                        ; consume -, eval rhs, promote pair
     test edx, edx                             ; float operands?
     jnz .add_fisub                            ; yes: SSE float subtract
     sub eax, ecx                              ; eax = lhs - rhs, a difference of two 30-bit values always fits in 32 bits
     cdqe                                      ; rax = exact result
     NUM_RANGE_CHECK rax, "Subtract: integer result does not fit the 30-bit payload range" ; out of range? FATAL, same masked-then-sign-extended trap as the sum above
     call MakeNum                              ; wrap as NUMBER
     jmp .add_loop                             ; check for more +-
.add_fisub:
     vsubss xmm0, xmm0, xmm1                   ; xmm0 = lhs - rhs
     call MakeFloat                            ; wrap as FLOAT
     jmp .add_loop                             ; check for more +-
.add_done:
     pop rbx                                   ; restore non-volatile
     ret                                       ; return to caller

%macro INFIX_TAGGED_EQUALITY 1
     INFIX_EVAL_RHS                            ; consume infix op, eval rhs, lhs pushed, the pop rcx below consumes it
     call EvalMulChain                         ; fold */% on rhs at higher precedence
     call EvalAddChain                         ; fold +- on rhs, = and <> / != both sit below + -
     pop rcx                                   ; rcx = lhs, tagged
     call Equal                                ; structural compare, eax,ecx -> ZF
     j%1 .infix_true                           ; equal -> TRUE / not-equal -> TRUE
     jmp .infix_false                          ; otherwise NIL
%endmacro
; ============================================================================
EvalNextArg:
     push rbx                                  ; align stack, entry RSP≡8 → push→RSP≡0 + save non-volatile
     call EvalPrimary                          ; eax = primary value
     call EvalMulChain                         ; fold */% at higher precedence first
     call EvalAddChain                         ; fold +- at higher precedence, below mul, above comparison/bitwise/concat
; --- trace: expression result, level >= 2 ---
     cmp dword [rel sl_trace_level], 1         ; level >= 2?
     jg .expr_trace                            ; yes: out-of-line trace block, cold
.no_trace_expr:
.infix_loop:
     test r12d, r12d                           ; stream is NIL?
     jz .infix_done                            ; yes: no more infix ops
; peek at next stream element without consuming it
     CHAIN_PEEK r11d                           ; r11d = candidate infix operator; eax keeps the accumulator
     cmp r11d, [rel sl_dispatch_highwater]     ; >= highwater: tagged value or post-builtin atom?
     jae .infix_no_op                          ; yes: cannot be an infix operator, leave the stream unchanged
     bt dword [rel sl_op_atom_bitmap], r11d    ; infix-operator bit set?, bt mem,reg sets CF
     jnc .infix_no_op                          ; no: a builtin, but not an operator, skip the 15-compare chain
; dispatch on candidate infix operator, comparison/bitwise/logical/concat only;
; multiplicative * / % handled by EvalMulChain; additive + - by EvalAddChain
     cmp r11d, [rel atom_sl_concat]            ; _ ?
     je .iconcat                               ; yes: string concatenation
     cmp r11d, [rel atom_sl_eq_sym]            ; = ?
     je .ieq                                   ; yes: equality
     cmp r11d, [rel atom_sl_neq_sym1]          ; <> ?
     je .ineq                                  ; yes: not equal
     cmp r11d, [rel atom_sl_neq_sym2]          ; != ?
     je .ineq                                  ; yes: not equal, alt
     cmp r11d, [rel atom_sl_gt_sym]            ; > ?
     je .igt                                   ; yes: greater than
     cmp r11d, [rel atom_sl_lt_sym]            ; < ?
     je .ilt                                   ; yes: less than
     cmp r11d, [rel atom_sl_gte_sym]           ; >= ?
     je .igte                                  ; yes: greater or equal
     cmp r11d, [rel atom_sl_lte_sym]           ; <= ?
     je .ilte                                  ; yes: less or equal
     cmp r11d, [rel atom_sl_and_sym]           ; & ?
     je .iband                                 ; yes: bitwise and
     cmp r11d, [rel atom_sl_or_sym]            ; | ?
     je .ibor                                  ; yes: bitwise or
     cmp r11d, [rel atom_sl_xor_sym]           ; ^ ?
     je .ibxor                                 ; yes: bitwise xor
     cmp r11d, [rel atom_sl_shl_sym]           ; << ?
     je .ishl                                  ; yes: shift left
     cmp r11d, [rel atom_sl_shr_sym]           ; >> ?
     je .ishr                                  ; yes: shift right
     cmp r11d, [rel atom_sl_andand]            ; && ?
     je .iandand                               ; yes: logical and
     cmp r11d, [rel atom_sl_oror]              ; || ?
     je .ioror                                 ; yes: logical or
.infix_no_op:
     pop rbx                                   ; restore non-volatile + unalign stack
     ret                                       ; return to caller, eax already holds the accumulator
; --- infix string concatenation: _ ---
.iconcat:
     INFIX_EVAL_RHS                            ; consume _ operator, eval rhs, lhs pushed, the push rax below stacks rhs above it
     call EvalMulChain                         ; fold */% on rhs at higher precedence
     call EvalAddChain                         ; fold +- on rhs at higher precedence, precedence: _ is below + -
     mov edx, [rsp]                            ; Read the left operand while it is still on the stack
     cmp eax, SL_EMPTY_ATOM_OFF                ; NIL and the empty atom render zero bytes
     jbe .concat_reuse                         ; Try reusing the left operand
     cmp edx, SL_EMPTY_ATOM_OFF                ; Does the left operand render zero bytes?
     ja .concat_render                         ; Render both nonempty operands
     mov edx, eax                              ; Try reusing the right operand
.concat_reuse:
     test edx, SL_TAG_MASK                     ; Is the candidate a tagged value?
     jnz .concat_render                        ; Numbers, floats, and lists need formatting
     cmp byte [r13 + rdx + SL_ATOM_TYPE_OFF], SL_TYPE_STRING ; Only an existing STRING has the result's type
     jne .concat_render                        ; Words and NIL still need conversion
     mov eax, edx                              ; Return the already interned string
     pop rdx                                   ; Release the saved left operand
     jmp .infix_loop                           ; Continue evaluating the chain
.concat_render:
     push rax                                  ; save rhs on stack, above lhs
     mov byte [rel sl_concat_buf], SL_TYPE_STRING ; type byte at [0]
     lea rdi, [rel sl_concat_buf + SL_ATOM_TEXT_OFF] ; rdi = concat buffer, content starts at [1]
     mov eax, [rsp + SL_CONCAT_LHS_OFF]        ; eax = lhs, tagged, below rhs
     call lg_value_to_str                      ; rdi = past lhs string, aligned
     mov eax, [rsp]                            ; eax = rhs, tagged, TOS, NOT popped
     call lg_value_to_str                      ; rdi = past rhs string, aligned
     add rsp, 16                               ; discard both slots
     mov byte [rdi], 0                         ; NUL-terminate
     lea rsi, [rel sl_concat_buf]              ; rsi = source for Intern
     call Intern                               ; eax = concatenated atom
     jmp .infix_loop                           ; check for more infix
; --- infix comparison: =, tagged equality, no NumVal ---
.ieq:
     INFIX_TAGGED_EQUALITY e                   ; = : equality is true
.ineq:
     INFIX_TAGGED_EQUALITY ne                  ; <> and != : inequality is true
; --- infix numeric comparison: > < >= <=, int + float paths ---
.igt:
     INFIX_COMPARE g, a                        ; lhs > rhs ?, ja for floats
.ilt:
     INFIX_COMPARE l, b                        ; lhs < rhs ?, jb for floats
.igte:
     INFIX_COMPARE ge, ae                      ; lhs >= rhs ?, jae for floats
.ilte:
     INFIX_COMPARE le, be                      ; lhs <= rhs ?, jbe for floats
; --- infix bitwise: & | ^ << >>, integers only, floats FATAL ---
.iband:
     INFIX_BITWISE_OP and                      ; & : logical AND
.ibor:
     INFIX_BITWISE_OP or                       ; | : logical OR
.ibxor:
     INFIX_BITWISE_OP xor                      ; ^ : logical XOR
.ishl:
     INFIX_BINARY_PROMOTED                     ; eval both operands, int or float
     test edx, edx                             ; float operands?
     jnz .bit_float_fatal                      ; yes: bitwise needs integers
     cdqe                                      ; rax = lhs, sign-extended, so a 64-bit shift cannot hide a wrap past bit 31
     cmp ecx, 31                               ; count outside 0..31?, an explicit check replaces the mask: x86 masks a negative or >= 32 count to 6 bits for the 64-bit shl below, and to 5 for .ishr's 32-bit sar, answering a different question than the script asked
     ja .shift_count_fatal                     ; yes: name it
     shl rax, cl                               ; rax = lhs << count, cl holds the validated count, x86 takes a variable shift count only from cl
     NUM_RANGE_CHECK rax, "ShiftLeft: integer result does not fit the 30-bit payload range" ; out of range? FATAL. Without it 1 << 29 masks to bit 29 and reads back -536870912, and 1 << 31 masks to NUMBER 0
     call MakeNum                              ; wrap as NUMBER
     jmp .infix_loop                           ; check for more infix
.ishr:
     INFIX_BINARY_PROMOTED                     ; eval both operands, int or float
     test edx, edx                             ; float operands?
     jnz .bit_float_fatal                      ; yes: bitwise needs integers
     cmp ecx, 31                               ; count outside 0..31?, see .ishl
     ja .shift_count_fatal                     ; yes: name it
     sar eax, cl                               ; eax = lhs >> count, arithmetic; |result| <= |lhs| so the result is always representable and needs no NUM_RANGE_CHECK
     call MakeNum                              ; wrap as NUMBER
     jmp .infix_loop                           ; check for more infix
; .ishl and .ishr both end in `jmp .infix_loop`, an unconditional transfer, so
.shift_count_fatal:
     FATAL "SectorLogo: shift count must be a NUMBER in the range 0..31" ; a negative or >= 32 count is masked in hardware, 6 bits for .ishl's 64-bit shl, 5 for .ishr's 32-bit sar, so the operator would silently answer a different question, no semicolon inside the string: the comment checker aligns on ; and would splice one in
.bit_float_fatal:
     FATAL "SectorLogo: bitwise operator requires integers (use F2I)" ; float into & | ^ << >>
.iandand:
     INFIX_EVAL_RHS                            ; consume && operator, eval rhs primary, lhs pushed
     call EvalMulChain                         ; fold */% on rhs at higher precedence
     call EvalAddChain                         ; fold +- on rhs at higher precedence
     pop rcx                                   ; ecx = lhs, tagged ; eax = rhs
     SL_FALSY_JUMP ecx, .infix_false           ; lhs falsy: NIL, rhs was still evaluated, no short-circuit
     SL_FALSY_JUMP eax, .infix_false           ; rhs falsy: NIL
     jmp .infix_true                           ; both truthy: TRUE
.ioror:
     INFIX_EVAL_RHS                            ; consume || operator, eval rhs primary, lhs pushed
     call EvalMulChain                         ; fold */% on rhs at higher precedence
     call EvalAddChain                         ; fold +- on rhs at higher precedence
     pop rcx                                   ; ecx = lhs, tagged ; eax = rhs
     SL_FALSY_JUMP ecx, .ioror_rhs             ; lhs falsy: the result is the rhs truthiness
     jmp .infix_true                           ; lhs truthy: TRUE
.ioror_rhs:
     SL_FALSY_JUMP eax, .infix_false           ; rhs falsy too: NIL
.infix_true:
     mov eax, [rel atom_sl_t]                  ; return TRUE atom
     jmp .infix_loop                           ; check for more infix
.infix_false:
     xor eax, eax                              ; return NIL, false
     jmp .infix_loop                           ; check for more infix
.infix_done:
     pop rbx                                   ; restore non-volatile + unalign stack
     ret                                       ; return to caller
.expr_trace:
     push rax                                  ; save result
     push rbx                                  ; save non-volatile, already saved but parity
     call sl_trace_print_indent                ; print indentation spaces
     lea rsi, [rel sl_trace_expr_prefix]       ; "| = " prefix, the sl_trace_print_indent one-WriteFile pattern
     mov edx, 4                                ; prefix byte length
     call PrintRaw                             ; single kernel transition instead of four
     mov eax, [rsp + SL_TRACE_EXPR_RESULT_OFF] ; restore result to eax, one qword up: the block pushed rax, then rbx
     call lg_print_value                       ; print result value
     mov al, 10                                ; newline
     call PutChar                              ; write newline
     pop rbx                                   ; restore non-volatile
     pop rax                                   ; restore result
     jmp .infix_loop                           ; rejoin the infix loop
.infix_chained_bool:
     FATAL "SectorLogo: comparison needs a NUMBER on the left, a chained comparison like 1 < 2 < 3 feeds the previous result (true or NIL) back in. Parenthesise the sub-expressions instead" ; name the actual mistake instead of NumVal's generic operand list, no semicolon inside the string: the comment checker aligns on ; and would splice one in
; ---- %undef cleanup: file-local defines leave scope here, house convention ----
%undef SL_TRACE_EXPR_RESULT_OFF
%unmacro INFIX_TAGGED_EQUALITY 1               ; remove the shared =/!= body, file-local macro leaves scope here
%unmacro CHAIN_PEEK 1                          ; remove the shared chain-loop peek
