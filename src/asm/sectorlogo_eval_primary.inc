; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

EvalPrimary:
     cmp rsp, [rel sl_stk_limit]               ; private-stack watermark: recursion guard
     jb sl_stack_overflow                      ; runaway recursion -> named FATAL, not an AV
     lea eax, [r12 - SL_CONS_TAG]              ; strip the tag AND fold the band check: eax = cell index for CONS, out-of-band for NIL/atom/NUMBER/FLOAT
     cmp eax, SL_FLOAT_TAG - SL_CONS_TAG       ; 0x40000000 = band width: below = cell, at/above = not a list
     jae .pj_pc_nil                            ; yes: both NIL
     mov ebx, [r13 + rax*CONS_SCALE + CONS_BASE_OFF] ; ebx = head atom, address folded; eax index still live
     mov eax, [r13 + rax*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF] ; eax = rest, folded; reads the index before overwriting it
     mov r12d, eax                             ; advance stream
     ASSERT_ATOM_NONNIL rbx, .ret_bare_atom    ; NOT a FATAL: tagged or NIL takes the normal self-evaluating-literal path
     cmp ebx, [rel sl_dispatch_highwater]      ; builtin region?
     jae .expr_fallback                        ; no: ':var' / literal / user-fn path
     lea rax, [rel sl_dispatch_expr_table]     ; table base in a register, RIP+index is illegal x64
     jmp [rax + rbx]                           ; the aligned atom byte offset indexes the expression table
.pj_pc_nil:
     xor eax, eax                              ; rest = NIL
     xor ebx, ebx                              ; head = NIL
     mov r12d, eax                             ; advance stream, a non-cons stream normalizes to NIL
     jmp .ret_bare_atom                        ; returns eax = ebx = 0
.expr_fallback:
     cmp byte [r13 + rbx], SL_TYPE_STRING      ; STRING-typed atom, quoted literal ?
     je .ret_bare_atom                         ; yes: return the STRING literal as-is, ebx is the atom; a bare word falls through
.expr_nodispatch:
; --- expression-position atom: variable deref or a numeric literal ---
     lea rsi, [r13 + rbx + SL_ATOM_TEXT_OFF]   ; rsi = atom string in heap
     movzx edx, byte [rsi]                     ; first text byte, shared by both tests
     cmp dl, ':'                               ; starts with ':'?
     jne .parse_integer                        ; no: candidate numeric literal
; atoms are never stored with a '"' prefix, the lexer strips quotes, so
; there is nothing to strip or test for here
.var_deref:
; strip ':' and look up the name, shared helper; saves/restores rbx
     mov eax, ebx                              ; eax = raw atom, helper contract
     call lg_atom_unprefix                     ; eax = clean variable name
     mov ebx, eax                              ; ebx = clean variable name, raw :name dead in this branch
     mov ecx, [rel sl_local_count]             ; ecx = LOCAL set entry count
     test ecx, ecx                             ; any LOCAL declarations active?
     jz .var_assoc                             ; no: alist path unchanged
     lea rax, [rel sl_local_set]               ; rax = set base, RIP-rel has no index form
     lea edx, [ecx - SL_LOCAL_NEWEST_ADJ]      ; edx = scan index = count-1, newest first
     LOCAL_SET_SCAN .var_local_hit, .var_assoc, ebx ; descending LOCAL-set scan, macro emits .var_local_scan loop
.var_local_hit:
     mov eax, [rax + rdx*SL_LOCAL_SET_SCALE + SL_LOCAL_CELL_OFF] ; eax = cached cell index, mov zero-extends
     mov eax, [r13 + rax*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF] ; eax = binding value
     ret                                       ; return to caller
.var_assoc:
     mov eax, ebx                              ; key = clean name
     mov edx, r14d                             ; edx = env
     call Assoc                                ; eax = value or NIL
     ret                                       ; return to caller
.parse_integer:
     sub dl, SL_ATOM_STARTER_BASE              ; bias so the numeric starters land at 0
     cmp dl, SL_ATOM_DIGIT9_MINUS_PLUS         ; first byte in '+'..'9' ?, 0x2B..0x39: '+' ',' '-' '.' '/' '0'..'9'. ':' biases to 15 and the ja below rejects it right here, and it can never reach this code anyway, because lines 104-105 divert every colon-leading atom to .var_deref. The zero-digit test at .atoi_done is what rejects ',' '.' '/'
     ja .atoi_not_num                          ; no: bare atom / string literal
     xor eax, eax                              ; accumulator = 0
     xor r8d, r8d                              ; r8d = negative flag, 0 = positive
     xor ecx, ecx                              ; ecx = digit counter, 0 digits parsed yet
     cmp dl, SL_ATOM_MINUS_BIAS                ; leading minus?, dl = biased first byte; '-' - '+' == 2
     jne .atoi_check_plus                      ; no: check for leading plus
; check that char after '-' is a digit, else it's just the '-' atom
     movzx edx, byte [rsi + SL_ATOI_PEEK_OFF]  ; char after '-'
     sub dl, '0'                               ; ASCII -> raw digit, invalid wraps >9 unsigned
     cmp dl, 9                                 ; valid digit 0..9?
     ja .atoi_not_num                          ; no: just the '-' atom
     inc rsi                                   ; skip past '-'
     mov r8d, 1                                ; set negative flag
     jmp .atoi_loop                            ; start parsing digits
.atoi_check_plus:
     test dl, dl                               ; leading plus?, dl == 0 is exactly '+' after the bias
     jne .atoi_loop                            ; no: start parsing digits
; check that char after '+' is a digit, else it's just the '+' atom
     movzx edx, byte [rsi + SL_ATOI_PEEK_OFF]  ; char after '+'
     sub dl, '0'                               ; ASCII -> raw digit, invalid wraps >9 unsigned
     cmp dl, 9                                 ; valid digit 0..9?
     ja .atoi_not_num                          ; no: just the '+' atom
     inc rsi                                   ; skip past '+'
.atoi_loop:
     movzx edx, byte [rsi]                     ; load next char
     sub dl, '0'                               ; ASCII -> raw digit, invalid wraps >9
     cmp dl, 9                                 ; valid digit 0..9?
     ja .atoi_done                             ; not a digit: done parsing
     cmp eax, STN_NUM_MUL_LIMIT                ; pre-multiply accumulator at/over the cap?, STN_NUM_MUL_LIMIT = STN_NUM_PAYLOAD_MAX/10 = 53687091, the largest accumulator that can still reach 2^29-1
     jae .atoi_near_limit                      ; yes: digit-specific overflow, STN_NUM_MUL_LIMIT*10 = 536870910, so the last digit decides, definite-overflow boundary in .atoi_near_limit, -2^29 exception in the digit==2 arm
     imul eax, 10                              ; accumulator *= 10, eax <= STN_NUM_MUL_LIMIT - 1 here
.atoi_mul_done:
     add eax, edx                              ; accumulator += digit
     inc rsi                                   ; advance string pointer
     inc ecx                                   ; increment digit count
     jmp .atoi_loop                            ; continue parsing
.atoi_near_limit:
; eax >= STN_NUM_MUL_LIMIT: check whether the specific digit causes overflow
     cmp eax, STN_NUM_MUL_OVER                 ; eax >= STN_NUM_MUL_OVER always overflows, 53687092*10+0 = 536870920 > 2^29-1
     jae .atoi_ofl                             ; definite overflow: promote to float
     cmp edx, 1                                ; digit <= 1?
     jbe .atoi_limit_digit_ok                  ; yes: 536870910 or 536870911, always valid
     test r8d, r8d                             ; negative literal?
     jz .atoi_ofl                              ; no: positive 536870912 exceeds 2^29-1 -> float
     cmp edx, 2                                ; digit == 2, == -2^29 ?
     jne .atoi_ofl                             ; no: digit >= 3 overflows either sign
.atoi_limit_digit_ok:
     imul eax, 10                              ; safe: eax = 536870910
     jmp .atoi_mul_done                        ; add digit -> 536870910 / 536870911 / 536870912, the last only when negative
.atoi_ofl:
; integer overflow, >= 2^29 payload limit : continue in float, the digit
; at rsi is re-read on the first .atof_ofl_digits iteration, acc so far in xmm0
     vcvtsi2sd xmm0, xmm0, eax                 ; xmm0 =, double partial accumulator, pre-overflow value; binary64, digits add below
.atof_ofl_digits:
     movzx edx, byte [rsi]                     ; load next char
     sub dl, '0'                               ; ASCII -> digit
     cmp dl, 9                                 ; valid digit?
     ja .atof_ofl_chk                          ; no: end of integer part
     vcvtsi2sd xmm1, xmm1, edx                 ; xmm1 = binary64 digit
     vfmadd132sd xmm0, xmm1, qword [rel dbl_ten] ; xmm0 = xmm0*dbl_ten + xmm1, rounded once
     inc rsi                                   ; advance string pointer
     inc ecx                                   ; count digit
     jmp .atof_ofl_digits                      ; continue float accumulation
.atof_ofl_chk:
     cmp byte [rsi], 0                         ; clean end of text?
     je .atof_sign                             ; yes: integer-valued float, apply sign and pack
     cmp byte [rsi], '.'                       ; fractional part after the huge integer?
     jne .atoi_not_num                         ; no: trailing garbage -> atom
     movzx edx, byte [rsi + SL_ATOI_PEEK_OFF]  ; first char after '.'
     sub dl, '0'                               ; ASCII -> raw digit, invalid wraps >9 unsigned
     cmp dl, 9                                 ; valid digit 0..9?
     ja .atoi_not_num                          ; no: "N." stays an atom
     inc rsi                                   ; skip past '.'
     jmp .atof_frac_entry                      ; fraction loop, int part already in xmm0
.atoi_done:
     test ecx, ecx                             ; any digits parsed at all?
     jz .atoi_not_num                          ; no digits, NIL/empty : not a number
     cmp byte [rsi], 0                         ; string ended?
     je .atoi_int_finish                       ; yes: pure integer literal
     cmp byte [rsi], '.'                       ; decimal point?
     jne .atoi_not_num                         ; no: not a numeric literal
     cmp byte [rsi + SL_ATOI_PEEK_OFF], '0'    ; first char after '.' below '0'?
     jb .atoi_not_num                          ; yes: "2." stays an atom
     cmp byte [rsi + SL_ATOI_PEEK_OFF], '9'    ; above '9'?
     ja .atoi_not_num                          ; yes: not a float literal
     jmp .atof_parse                           ; yes: FLOAT literal "int.frac"
.atoi_int_finish:
     test r8d, r8d                             ; negative?
     jz .atoi_ret                              ; no: keep as-is
     neg eax                                   ; negate the value
.atoi_ret:
     jmp MakeNum                               ; wrap as NUMBER
; --- FLOAT literal: eax = |int part|, r8d = negative flag, rsi -> '.' ---
.atof_parse:
     inc rsi                                   ; skip past '.'
     vcvtsi2sd xmm0, xmm0, eax                 ; xmm0 = |int part|, binary64: the fraction add below must not double-round through binary32
.atof_frac_entry:
     xor r9d, r9d                              ; r9d = fraction digit accumulator
     xor r10d, r10d                            ; r10d = fraction digit count
.atof_frac_loop:
     movzx edx, byte [rsi]                     ; load next char
     sub dl, '0'                               ; ASCII -> digit
     cmp dl, 9                                 ; valid digit?
     ja .atof_frac_done                        ; no: end of fraction
     cmp r10d, SF_ATOM_DIGIT_MAX               ; over the exact-parse cap?
     jae .atoi_not_num                         ; yes: reject as float literal
     imul r9d, r9d, 10                         ; frac acc *= 10 ...
     add r9d, edx                              ; ... + digit
     inc r10d                                  ; count digit
     inc rsi                                   ; advance text pointer
     jmp .atof_frac_loop                       ; keep parsing
.atof_frac_done:
     cmp byte [rsi], 0                         ; clean end of text?
     jne .atoi_not_num                         ; no: trailing garbage -> atom
     lea rax, [rel pow10_table]                ; table of 10^n, n = 0..9
     mov eax, [rax + r10*SL_POW10_DWORD]       ; eax = 10^count, r10 upper bits zero: xor'd + inc only
     vcvtsi2sd xmm1, xmm1, r9d                 ; xmm1 =, double frac digits int
     vcvtsi2sd xmm2, xmm2, eax                 ; xmm2 =, double 10^count
     vdivsd xmm1, xmm1, xmm2                   ; xmm1 = fraction value, binary64: exact int32 sources, one rounding
     vaddsd xmm0, xmm0, xmm1                   ; xmm0 = int + frac, binary64 accumulation, single narrowing happens at .atof_sign
.atof_sign:
     vcvtsd2ss xmm0, xmm0                      ; narrow the binary64 accumulation to binary32, the SINGLE correctly-rounded step, covers both the .atof_frac_done fall-through and the .atof_ofl_chk jump predecessor
     test r8d, r8d                             ; negative literal?
     jz .atof_pack                             ; no: keep positive
     vxorps xmm0, xmm0, [rel flt_sign_mask]    ; flip sign bit
.atof_pack:
     jmp MakeFloat                             ; wrap as FLOAT, truncates 2 low bits
.atoi_not_num:
     mov ecx, [rel sl_local_count]             ; ecx = LOCAL set entry count
     test ecx, ecx                             ; any LOCAL declarations active?
     jz .atoi_assoc                            ; no: alist path unchanged
     lea rax, [rel sl_local_set]               ; rax = set base, RIP-rel has no index form
     lea edx, [ecx - SL_LOCAL_NEWEST_ADJ]      ; edx = scan index = count-1, newest first
     LOCAL_SET_SCAN .atoi_local_hit, .atoi_assoc, ebx ; descending LOCAL-set scan, macro emits .atoi_local_scan loop
.atoi_local_hit:
     mov eax, [rax + rdx*SL_LOCAL_SET_SCALE + SL_LOCAL_CELL_OFF] ; eax = cached cell index, mov zero-extends
     mov eax, [r13 + rax*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF] ; eax = binding value
     test eax, eax                             ; NIL?
     jz .ret_bare_atom                         ; yes: return bare atom
     jmp .atoi_bound_check                     ; no: run the existing descriptor checks
.atoi_assoc:
     mov eax, ebx                              ; eax = atom to look up
     mov edx, r14d                             ; edx = env
     call Assoc                                ; eax = bound value or NIL
     test eax, eax                             ; NIL?
     jnz .atoi_bound_check                     ; bound: run the descriptor checks, the fall-through below is the miss path
.atoi_assoc_miss:
     cmp dword [rel sl_strict_names], 0        ; strict mode armed?
     jne .atoi_strict_dead                     ; yes: an unbound name in expression position is a bug, not a value
     mov eax, ebx                              ; eax = the dead name atom, ebx is Assoc-preserved, Win64 non-volatile
     xor ecx, ecx                              ; ecx = position class 0, expression/value slot
     call sl_warn_unknown_word                 ; tier-2 warning, name the unknown word to the console
     jmp .ret_bare_atom                        ; no strict mode: documented bare-atom semantics
.atoi_strict_dead:
     FATAL_VALUE "unbound name in expression position", ebx ; names the dead name, ebx = the atom Assoc could not find
.atoi_bound_check:
; check if bound value is a function descriptor, __FN__.desc
     mov ecx, eax                              ; ecx = bound value
     shr ecx, 30                               ; ecx = 2-bit tag code (CONS = 2)
     cmp ecx, 2                                ; exactly CONS, not FLOAT?
     jne .ret_bare_atom                        ; not cons: return bare atom
     mov r15d, eax                             ; r15d = save fn descriptor
     call Car                                  ; eax =, car bound_val
     cmp eax, [rel atom_sl_fn_tag]             ; __FN__ tag?
     jne .expr_check_xfn                       ; no: maybe __XF__, exe function
; invoke user-defined function in expression position
     mov eax, r15d                             ; eax = restored fn descriptor
     call lg_invoke_user_fn                    ; eax = return value
     ret                                       ; return to caller
.expr_check_xfn:
     cmp eax, [rel atom_sl_xfn_tag]            ; __XF__, DEF_EXE_FN tag?
     jne .ret_bare_atom                        ; no: return bare atom
     mov eax, r15d                             ; eax = restored xfn descriptor
     call lg_invoke_exe_fn                     ; eax = interned stdout string
     ret                                       ; return to caller
.ret_bare_atom:
; return the bare atom
     mov eax, ebx                              ; return the atom itself
     ret                                       ; return to caller
; --- parenthesized expression:, expr ---
.eval_paren:
     call EvalNextArg                          ; eax = evaluated expression
     push rax                                  ; save result
; consume closing ' ' if present
     test r12d, r12d                           ; stream not empty?
     jz .paren_done                            ; empty: no ' ' to consume
     mov eax, r12d                             ; eax = stream cell
     btr eax, SL_CONS_BIT                      ; strip tag -> cell index
     lea rcx, [r13 + rax*CONS_SCALE + CONS_BASE_OFF] ; rcx = stream cell address, derived ONCE for both field reads
     mov eax, [rcx]                            ; eax = car = next head
     cmp eax, [rel atom_sl_rparen]             ; is it ' '?
     jne .paren_done                           ; no: leave stream as-is
     mov eax, [rcx + CONS_CDR_OFF]             ; eax = cdr = stream past ' '
     mov r12d, eax                             ; advance stream past ' '
.paren_done:
     pop rax                                   ; restore result
     ret                                       ; return to caller
.eval_rparen_skip:
     xor eax, eax                              ; stray ' ': return NIL
     ret                                       ; return to caller
; --- RUN/EVAL in expression position ---
.p_run_eval:
     call EvalNextArg                          ; eax = block, evaluated
; if block is a cons list, execute it and return the result
     mov ecx, eax                              ; ecx = block
     shr ecx, 30                               ; ecx = 2-bit tag code (CONS = 2)
     cmp ecx, 2                                ; exactly CONS, not FLOAT?
     jne .p_run_check_type                     ; not cons: check for illegal types
     mov dword [rel sl_stmt_fn_retval], 0      ; clear the statement-fn-return channel, a value from THIS block only, stale outer values never leak
     SL_EVAL_BLOCK 1, eax                      ; stream = block, with trace indent
; return the last OUTPUT value, the last statement fn call's return, or NIL
     cmp dword [rel sl_fn_has_returned], 1     ; did OUTPUT execute, bare Output statement in the block ?
     jne .run_ret_fnval                        ; no: fall to the statement-fn channel
     mov eax, [rel sl_fn_return_val]           ; return the OUTPUT value
     mov dword [rel sl_fn_has_returned], 0     ; absorb the signal, expression RUN
     ret                                       ; return the OUTPUT value, fall-through exit, not a branch target
.run_ret_fnval:
     mov eax, [rel sl_stmt_fn_retval]          ; last user/xfn statement-call return, 0 = no fn ran -> NIL : Run [f args] = dynamic apply
     ret                                       ; return to caller
.p_run_check_type:
; not CONS: NIL is allowed, skip silently, NUMBER/FLOAT are contract violations
     test eax, eax                             ; NIL?
     jnz .p_run_fatal                          ; NUMBER/FLOAT: contract violation; NIL falls through to the nil return
.run_ret_nil:
     xor eax, eax                              ; NIL, no block executed, the fn-return channel was never armed
     ret                                       ; return to caller
.p_run_fatal:
     FATAL_VALUE "RUN requires a list or NIL", eax ; name the offending value
.p_ifelse_val:
     call EvalNextArg                          ; eax = condition value
     mov r15d, eax                             ; r15d = cond, evaluator scratch
     mov eax, r12d                             ; eax = argument stream
     call PopCons                              ; ebx = then-block, eax = rest
     push rbx                                  ; save then-block
     mov r12d, eax                             ; stream = rest, eax already holds it, no reload needed
     call PopCons                              ; ebx = else-block, eax = rest
     push rbx                                  ; save else-block
     mov r12d, eax                             ; argument stream advanced past both blocks
     SL_FALSY_JUMP r15d, .iv_else              ; falsy cond: take the else-block
     pop rax                                   ; drop else-block
     pop rbx                                   ; rbx = then-block, chosen
     jmp .iv_run                               ; evaluate the chosen block
.iv_else:
     pop rbx                                   ; rbx = else-block, chosen
     pop rax                                   ; drop then-block
.iv_run:
     test ebx, ebx                             ; empty chosen block?
     jz .iv_nil                                ; yes: NIL, nothing to value, the ONLY legitimate empty result
     mov ecx, ebx                              ; copy for tag isolation
     shr ecx, 30                               ; ecx = 2-bit tag code (CONS = 2)
     cmp ecx, 2                                ; exactly CONS, not FLOAT, not atom?
     jne .iv_fatal                             ; no: a branch must be a list
; peek at the first element to detect statement-only builtins
     mov eax, ebx                              ; eax = block, CONS-ness proven above
     call Car                                  ; eax = first element, peeked, stream not consumed
     push r12                                  ; save the outer argument stream, EvalNextArg below advances r12d
     push rbx                                  ; keep the block for the fallback path
     push rax                                  ; park the peek: r15d is evaluator scratch, not preserved across EvalNextArg, it may evaluate arbitrary expressions, and the peek is popped back into r15d at :326 below
     mov r12d, ebx                             ; stream = chosen block
     call EvalNextArg                          ; eax = first form's expression value
     pop r15                                   ; r15d = peeked first element, RESTORED, it is read below, not live across the call
     pop rbx                                   ; restore chosen block
; detect statement-only builtin: EvalNextArg returned the same atom unchanged
; the peek was popped back into r15d AFTER the call, so it is intact here
     cmp eax, r15d                             ; same atom returned?, EvalNextArg returned peek unchanged
     jne .iv_expr_done                         ; no: expression was evaluated normally
; check if the returned value is a statement-only builtin, not a bare user atom
     test r15d, r15d                           ; NIL?
     jz .iv_expr_done                          ; NIL is not a builtin
     test r15d, SL_TAG_MASK                    ; tag bits, NUMBER/FLOAT/CONS set? TEST writes no register, and the cmp + bt below both still need r15d
     jnz .iv_expr_done                         ; tagged value: not a builtin atom
     cmp r15d, [rel sl_dispatch_highwater]     ; within the builtin region?
     jae .iv_expr_done                         ; no: user atom far past builtins
     bt dword [rel sl_builtin_bitmap], r15d    ; registered builtin?
     jnc .iv_expr_done                         ; no: bare user atom, same value by coincidence
     pop r12                                   ; restore outer argument stream
     mov esi, ebx                              ; stream = ENTIRE chosen block, not the remainder
     jmp .iv_stmt_run                          ; shared statement-eval path
.iv_expr_done:
     test r12d, r12d                           ; first form consumed the whole block?
     jnz .iv_stmt                              ; no: statement semantics for the block
     pop r12                                   ; yes: restore outer stream; eax = the block's value
     ret                                       ; return to caller
.iv_stmt:
     mov esi, r12d                             ; stream = REMAINING forms, past the first form already evaluated by EvalNextArg above
     pop r12                                   ; restore outer argument stream
.iv_stmt_run:
     SL_EVAL_BLOCK 1, esi                      ; with trace indent
     cmp dword [rel sl_fn_has_returned], 1     ; OUTPUT executed inside the block?
     jne .iv_nil                               ; no: no value produced
     mov eax, [rel sl_fn_return_val]           ; the OUTPUT value is the result
     mov dword [rel sl_fn_has_returned], 0     ; absorb the signal, expression context, RUN contract
     ret                                       ; return to caller
.iv_nil:
     xor eax, eax                              ; no OUTPUT: NIL
     ret                                       ; return to caller
.iv_fatal:
     FATAL_VALUE "IFELSE requires a list or NIL for each branch", ebx ; name the offending branch
