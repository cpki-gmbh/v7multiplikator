; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%macro ENV_UPDATE_WALK 1
     mov eax, r14d                             ; eax = env
     test eax, eax                             ; empty env?, entry test PEELED above the loop label, so the body can bottom-test without dereferencing a NIL cell
     jz %%prepend                              ; not found, an empty env is a miss, not a walk
%%walk:
     mov ecx, eax                              ; ecx = tagged env cell
     btr ecx, SL_CONS_BIT                      ; ecx = raw CONS index
     mov edx, [r13 + rcx*CONS_SCALE + CONS_BASE_OFF] ; edx = tagged inner pair, folded load: replaces the lea + mov pair
     btr edx, SL_CONS_BIT                      ; edx = raw CONS index of inner pair, KEPT: %%found's folded store indexes the cell through it
     cmp dword [r13 + rdx*CONS_SCALE + CONS_BASE_OFF], r15d ; car of the inner pair == target name?
     je %%found                                ; yes: update this binding
     mov eax, [r13 + rcx*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF] ; eax = rest of env chain, folded load: replaces the lea + mov pair
     test eax, eax                             ; end of chain?
     jnz %%walk                                ; more cells: continue, CONDITIONAL back edge, no per-cell jmp
%%prepend:
     mov edx, r8d                              ; edx = value
     mov eax, r15d                             ; eax = name
     call Cons                                 ; eax =, name.value binding pair
     mov edx, r14d                             ; edx = old env, cdr
     call Cons                                 ; eax =, ,name.value.env
     mov r14d, eax                             ; update env
     jmp %1                                    ; exit to the caller's continuation
%%found:
     mov [r13 + rdx*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF], r8d ; CDR = new value, folded store: indexes the inner-pair cell through rdx directly
     jmp %1                                    ; exit to the caller's continuation
%endmacro

%macro INFIX_EVAL_RHS 0
     mov edx, eax                              ; save lhs, tagged
     mov eax, r12d                             ; eax = stream
     call Cdr                                  ; skip past operator
     mov r12d, eax                             ; update stream
     push rdx                                  ; save lhs on stack
     call EvalPrimary                          ; eax = rhs primary
%endmacro

%macro INFIX_ADD_PROMOTED 0
     INFIX_EVAL_RHS                            ; consume operator, eval rhs, lhs pushed
     call EvalMulChain                         ; fold */% on rhs at higher precedence
     pop rdx                                   ; edx = lhs, tagged
     call sl_promote_pair                      ; edx=1: floats xmm0/xmm1 | edx=0: ints eax/ecx
%endmacro

%macro INFIX_BINARY_PROMOTED 0
     INFIX_EVAL_RHS                            ; consume operator, eval rhs, lhs pushed
     call EvalMulChain                         ; fold */% on rhs at higher precedence
     call EvalAddChain                         ; fold +- on rhs at higher precedence, precedence: comparisons see +- but not themselves on RHS
     pop rdx                                   ; edx = lhs, tagged
     call sl_promote_pair                      ; edx=1: floats xmm0/xmm1 | edx=0: ints eax/ecx
%endmacro

%macro INFIX_COMPARE 2
     cmp eax, [rel atom_sl_t]                  ; lhs is the TRUE atom?, a comparison result fed back in, SectorLogo comparisons are not transitive
     je .infix_chained_bool                    ; yes: `1 < 2 < 3`, name the mistake
     test eax, eax                             ; lhs is NIL, the false result, there is no `false` atom ?
     jz .infix_chained_bool                    ; yes: same
     INFIX_BINARY_PROMOTED                     ; eval both operands, int or float
     test edx, edx                             ; float operands?
     jnz %%float                               ; yes: vucomiss path
     cmp eax, ecx                              ; lhs ? rhs, ints
     j%1 .infix_true                           ; condition-specific jump
     jmp .infix_false                          ; not true: return NIL
%%float:
     vucomiss xmm0, xmm1                       ; lhs ? rhs, floats
     jp .infix_false                           ; NaN: unordered -> always false
     j%2 .infix_true                           ; condition-specific float jump
     jmp .infix_false                          ; not true: return NIL
%endmacro

%macro PREFIX_RELATIONAL_CMP 2
     NUM_BINARY_PROMOTED                       ; eax=lhs, ecx=rhs, or floats xmm0/xmm1
     test edx, edx                             ; float operands?
     jnz %%f                                   ; yes: vucomiss path
     cmp eax, ecx                              ; lhs ? rhs, ints
     j%1 .ret_true                             ; condition-specific jump
     jmp .ret_false                            ; not true: return NIL
%%f:
     vucomiss xmm0, xmm1                       ; lhs ? rhs, floats
     jp .ret_false                             ; NaN: unordered -> always false
     j%2 .ret_true                             ; condition-specific float jump
     jmp .ret_false                            ; not true: return NIL
%endmacro

%macro INFIX_BITWISE_OP 1
     INFIX_BINARY_PROMOTED                     ; eval both operands, int or float
     test edx, edx                             ; float operands?
     jnz .bit_float_fatal                      ; yes: bitwise needs integers
%1 eax, ecx                                    ; apply the op: eax = lhs <op> rhs
     call MakeNum                              ; wrap as NUMBER
     jmp .infix_loop                           ; check for more infix
%endmacro

%macro PREFIX_BITWISE_OP 1
     NUM_BINARY_PROLOGUE                       ; eax=lhs, ecx=rhs
%1 eax, ecx                                    ; eax = lhs <op> rhs, always representable, see above
     jmp MakeNum                               ; wrap as NUMBER
%endmacro

%macro SL_FLOAT_ZERO_JUMP 2
     cmp %1, SL_FLOAT_ZERO_POS                 ; +0.0 exact encoding?
     je %2                                     ; yes: falsy
     cmp %1, SL_FLOAT_ZERO_NEG                 ; -0.0 exact encoding?
     je %2                                     ; yes: falsy
%endmacro

%macro TAG_IS_JUMP 3
     mov ecx, %1                               ; ecx = copy for the tag test
     and ecx, SL_TAG_MASK                      ; isolate BOTH tag bits, FLOAT shares bit31 with CONS
     cmp ecx, %2                               ; expected tag?
     je %3                                     ; yes: jump
%endmacro

%macro TAG_IS_JUMP_NE 3
     mov ecx, %1                               ; ecx = copy for the tag test
     shr ecx, 30                               ; extract the 2-bit type code (0 atom/NIL, 1 NUMBER, 2 CONS, 3 FLOAT)
     cmp ecx, (%2 >> 30)                       ; expected type code
     jne %3                                    ; no: not the expected tag
%endmacro

%macro SL_FALSY_JUMP 2
     test %1, %1                               ; NIL?
     jz %2                                     ; yes: falsy
     cmp %1, [rel atom_sl_false]               ; FALSE atom?
     je %2                                     ; yes: falsy
     cmp %1, SL_NUMBER_TAG                     ; NUMBER 0?
     je %2                                     ; yes: falsy
     SL_FLOAT_ZERO_JUMP %1, %2                 ; FLOAT +/-0.0: falsy
%endmacro

%macro SL_EVAL_BLOCK 2
     mov esi, %2                               ; stream = block
     mov edx, r14d                             ; env = current env
     mov ecx, [rel sl_local_count]             ; ecx = LOCAL count, 32-bit load zero-extends into rcx, never read past the dword
     push rcx                                  ; park the clean count, popped into rcx at the restore below as sl_tombstone_locals' argument
%if %1 == 1
     inc dword [rel sl_trace_indent]           ; trace indent: entering block
%endif
     call EvalLogoStream                       ; edx = updated env
%if %1 == 1
     dec dword [rel sl_trace_indent]           ; trace indent: leaving block
%endif
     pop rcx                                   ; restore LOCAL count, old value
     mov r14d, edx                             ; update env
     call sl_tombstone_locals                  ; tombstone discarded LOCAL entries + restore count
%endmacro

%macro LOCAL_SET_SCAN 3
%%scan:
     cmp [rax + rdx*SL_LOCAL_SET_SCALE], %3    ; set entry matches var name?
     je %1                                     ; yes: caller-specific hit action
     dec edx                                   ; advance scan, descending ; SF set = exhausted
     jns %%scan                                ; SF clear -> edx >= 0 -> another entry to scan, ONE branch on the loop back edge
     jmp %2                                    ; SF set -> edx == -1 -> past the oldest entry: caller-specific miss action, same flags %2 saw before: jns and jmp write none
%endmacro
; --- prologue: evaluate two numeric args, return in eax/ecx ---

%macro NUM_BINARY_PROLOGUE 0
     call EvalNextArg                          ; eax = first arg, tagged
     call NumVal                               ; eax = raw first arg
     push rax                                  ; save raw lhs
     call EvalNextArg                          ; eax = second arg, tagged
     call NumVal                               ; eax = raw second arg
     mov ecx, eax                              ; ecx = raw rhs
     pop rax                                   ; eax = raw lhs
%endmacro

%macro NUM_RANGE_CHECK 2
     lea rdx, [%1 + (0 - SL_PAYLOAD_MIN)]      ; bias the inclusive signed payload range into unsigned 0..2^30-1
     cmp rdx, SL_PAYLOAD_MASK                  ; does the biased result still fit?
     jbe %%fits                                ; yes: representable as-is
     test %1, %1                               ; out of range: the original sign selects the endpoint
     mov %1, SL_PAYLOAD_MAX                    ; preload the ceiling
     jns %%fits                                ; non-negative: saturate at the ceiling
     mov %1, SL_PAYLOAD_MIN                    ; negative: saturate at the floor
%%fits:
%endmacro
; --- float-aware prologue: tagged pair -> sl_promote_pair ---
; edx=1 + xmm0,lhs /xmm1,rhs floats | edx=0 + eax,lhs /ecx,rhs ints

%macro NUM_BINARY_PROMOTED 0
     call EvalNextArg                          ; eax = first arg, tagged
     push rax                                  ; save lhs tagged
     call EvalNextArg                          ; eax = second arg, tagged
     mov edx, [rsp]                            ; edx = lhs tagged
     call sl_promote_pair                      ; dispatch on operand tags
     add rsp, 8                                ; discard lhs slot
%endmacro
