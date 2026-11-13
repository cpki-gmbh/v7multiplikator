; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; Assoc: EAX = key, EDX = environment, R13 = heap. Returns its value or NIL.
Assoc:
     mov r8d, eax                              ; Keep the key in a volatile register
     lea r9, [r13 + CONS_BASE_OFF]             ; Reuse the cons base across the walk
     test edx, edx                             ; Is the environment empty?
     jz .not_found                             ; Return NIL for an empty environment
.loop:
     lea ecx, [rdx - SL_CONS_TAG]              ; Convert the environment node to a cell index
     mov eax, [r9 + rcx*CONS_SCALE]            ; Load the node's binding pair
     mov edx, [r9 + rcx*CONS_SCALE + CONS_CDR_OFF] ; Fetch the next node while the pair loads
     btr eax, SL_CONS_BIT                      ; Convert the binding pair to a cell index
     cmp [r9 + rax*CONS_SCALE], r8d            ; Does the binding name match the key?
     je .found                                 ; Return the first matching binding
     test edx, edx                             ; Does another node follow?
     jnz .loop                                 ; Continue the walk
.not_found:
     xor eax, eax                              ; No binding matched
     ret                                       ; Return NIL
.found:
     mov eax, [r9 + rax*CONS_SCALE + CONS_CDR_OFF] ; Read the current binding value
     ret                                       ; Return the value
%define SL_LIDX_LIST_OFF 32                    ; rsp offset of the staged list, EvalLogoStream-overwritten
%define SL_LIDX_IDX_OFF 36                     ; rsp offset of the tagged index park, FATAL_VALUE input; parked BEFORE NumVal, which overwrites eax
%define SL_LIDX_FRAME 40                       ; frame bytes: 2 dword slots + alignment
%define SL_AN_X10_SCALE 2                      ; SIB scale: the second lea of the acc*10 fold
sl_lindex:
     push rsi                                  ; save non-volatile
     push rdi                                  ; save non-volatile
     sub rsp, SL_LIDX_FRAME                    ; 2 slots + padding
     call EvalNextArg                          ; eax = list
     mov [rsp + SL_LIDX_LIST_OFF], eax         ; save list to stack slot
     test eax, eax                             ; sets ZF (NIL) and SF (bit31) in one test
     jz sl_lidx_not_list                       ; NIL, empty list
     jns sl_lidx_not_list                      ; bit31 clear: plain atom/NUMBER
     test eax, SL_NUMBER_TAG                   ; bit30 also set = FLOAT?
     jnz sl_lidx_not_list                      ; yes: FLOAT, FATAL
     call EvalNextArg                          ; eax = index, tagged
     mov [rsp + SL_LIDX_IDX_OFF], eax          ; park the TAGGED index, NumVal overwrites eax; the FATAL sites need the tag bits
     call NumVal                               ; eax = raw index
     mov ecx, eax                              ; ecx = remaining steps
     mov eax, [rsp + SL_LIDX_LIST_OFF]         ; eax = list, from stack slot
     test ecx, ecx                             ; negative index?
     js sl_lidx_bad_idx                        ; yes: out of bounds -> FATAL
sl_lidx_walk:
     test eax, eax                             ; list is NIL?
     jz sl_lidx_bad_idx                        ; yes: walked off the end -> FATAL
     test ecx, ecx                             ; reached target?
     jz sl_lidx_ret_car                        ; yes: return car
     call Cdr                                  ; eax = rest of list
     dec ecx                                   ; count down
     jmp sl_lidx_walk                          ; continue walking
sl_lidx_ret_car:
     call Car                                  ; eax = nth element
; coerce numeric ATOMs to NUMBER/FLOAT for consistent output
     test eax, eax                             ; NIL?
     jz sl_lidx_epilog                         ; yes: return NIL
     test eax, SL_TAG_MASK                     ; NUMBER/FLOAT/CONS, any tag bit set ?
     jnz sl_lidx_epilog                        ; NUMBER/FLOAT/CONS: return as-is
     lea rcx, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rcx = string in heap
     mov dl, byte [rcx]                        ; first char
     cmp dl, '+'                               ; leading plus?
     jne .lidx_sign_chk                        ; no: check minus
     inc rcx                                   ; skip '+'
     mov dl, byte [rcx]                        ; char after '+'
     cmp dl, '-'                               ; second sign after '+'?
     je sl_lidx_epilog                         ; yes: "+-N" is not a number, return the atom unchanged
     jmp .lidx_int                             ; digits start here
.lidx_sign_chk:
     cmp dl, '-'                               ; leading minus?
     jne .lidx_int                             ; no: integer digits start here
     inc rcx                                   ; skip '-'
     mov dl, byte [rcx]                        ; char after '-', the coerce filter REQUIRES a digit here, unlike Sort's coerce grammar, an atom like "-.5" stays the raw atom: lindex's stricter contract, pinned by validator test
     cmp dl, '0'                               ; below '0'?
     jb sl_lidx_epilog                         ; "-.5"/"-foo": reject, return the atom
     cmp dl, '9'                               ; above '9'?
     ja sl_lidx_epilog                         ; reject
.lidx_int:
     xor r8d, r8d                              ; total digit counter = 0
.lidx_int_loop:
     cmp dl, '0'                               ; below '0'?
     jb .lidx_dot                              ; yes: integer run ended, maybe the '.'
     cmp dl, '9'                               ; above '9'?
     ja .lidx_dot                              ; yes: integer run ended
     inc r8d                                   ; one more integer digit
     inc rcx                                   ; advance the cursor
     mov dl, byte [rcx]                        ; next char
     jmp .lidx_int_loop                        ; keep consuming digits
.lidx_dot:
     cmp dl, '.'                               ; fraction point?
     jne .lidx_end                             ; no: terminator check
     inc rcx                                   ; skip '.'
     mov dl, byte [rcx]                        ; first fraction char
     cmp dl, '0'                               ; digit?
     jb .lidx_dot_end                          ; no: maybe the trailing-dot form "42."
     cmp dl, '9'                               ; above '9'?
     ja .lidx_dot_end                          ; no digit after '.'
.lidx_frac_loop:
     inc r8d                                   ; fraction digits count toward the total
     inc rcx                                   ; advance the cursor
     mov dl, byte [rcx]                        ; next char
     cmp dl, '0'                               ; still a digit, low bound ?
     jb .lidx_end                              ; no: terminator check
     cmp dl, '9'                               ; still a digit, high bound ?
     jbe .lidx_frac_loop                       ; yes: keep consuming fraction digits
     jmp .lidx_end                             ; non-digit ends the run
.lidx_dot_end:
     test dl, dl                               ; non-digit after '.': only NUL keeps "N." valid
     jnz sl_lidx_epilog                        ; "N.x": reject
; trailing dot + NUL: fall into the end check with dl = 0
.lidx_end:
     test dl, dl                               ; NUL terminator?
     jnz sl_lidx_epilog                        ; trailing garbage: reject, return the atom
     test r8d, r8d                             ; at least one digit total?
     jz sl_lidx_epilog                         ; "" / "+" / "." / "+.": reject
     call sl_sort_coerce_num                   ; eax = NUMBER or FLOAT
     jmp sl_lidx_epilog                        ; shared frame teardown, this jump is the barrier over the two FATAL sites below, they must never fall through
sl_lidx_not_list:
     FATAL_VALUE "Item requires a non-empty list", eax ; NIL, plain atom, or FLOAT given where a list is required
sl_lidx_bad_idx:
     mov eax, [rsp + SL_LIDX_IDX_OFF]          ; eax = the tagged index the script asked for
     FATAL_VALUE "Item index out of bounds", eax ; one site for negative index AND walk-off-the-end, mirroring sl_lset's single .ls_bad_idx
sl_lidx_epilog:
     add rsp, SL_LIDX_FRAME                    ; restore frame
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     ret                                       ; return to caller
     SL_AN_MAX_DIGITS equ 10                   ; int32 cap, allows 10-digit ints like 2147483647
     SL_AN_FOLD_SCALE equ 4                    ; SIB scale of the acc*5 fold, acc + acc*4, then +acc again = *10
     SL_AN_INT32_MIN_ABS equ 0x80000000        ; absolute value of INT32_MIN, 2147483648
; sl_atom_num,eax = tagged PLAIN atom, caller guarantees no tag bits -> eax = parsed
; decimal int32, optional leading '-', max 10 digits, range-checked to INT32, ZF = 1;
; or ZF = 0, not a numeric atom, eax undefined at this exit and ignored by callers .
sl_atom_num:
     lea r11, [r13 + rax + SL_ATOM_TEXT_OFF]   ; r11 = atom text in heap
     xor eax, eax                              ; eax = decimal accumulator
     xor ecx, ecx                              ; ecx = 0 positive / 1 negative
     xor r10d, r10d                            ; r10d = digit count
     cmp byte [r11], '-'                       ; leading minus sign?
     jne .an_loop                              ; no: start at digits
     mov ecx, 1                                ; mark negative form
     inc r11                                   ; skip past '-'
.an_loop:
     movzx edx, byte [r11]                     ; edx = current char
     test dl, dl                               ; NUL terminator?
     jz .an_end                                ; yes: finish parse
     cmp r10d, SL_AN_MAX_DIGITS                ; already at digit cap?
     jae .an_not_num                           ; yes: too large for int32
     sub dl, '0'                               ; ASCII -> digit value
     cmp dl, 9                                 ; valid digit 0..9?
     ja .an_not_num                            ; no: not a numeric atom
     lea rax, [rax + rax*SL_AN_FOLD_SCALE]     ; acc = acc*5, SIB fold, named scale
     lea rax, [rdx + rax*SL_AN_X10_SCALE]      ; acc = acc*10 + digit, one 3-component lea replaces the *2 add pair
     inc r10d                                  ; count this digit
     inc r11                                   ; advance text pointer
     jmp .an_loop                              ; keep parsing
.an_end:
     test r10d, r10d                           ; zero digits, "-" or "" ?
     jz .an_not_num                            ; yes: not numeric
     test ecx, ecx                             ; negative form?
     jz .an_pos_check                          ; no: check positive range
; negative: absolute value must be <= 2^31, 2147483648
     mov r11d, SL_AN_INT32_MIN_ABS             ; r11 = 2^31: the 32-bit destination zero-extends bits 63:32, a 64-bit destination would SIGN-extend 0x80000000 to 0xFFFFFFFF80000000 and make the bound test below unreachable
     cmp rax, r11                              ; abs value > 2^31?
     ja .an_not_num                            ; yes: doesn't fit in int32
     neg eax                                   ; apply minus sign
     jmp .an_ok                                ; success
.an_pos_check:
; positive: value must be <= 2^31-1, 2147483647
     cmp rax, 0x7FFFFFFF                       ; value > 2^31-1?
     ja .an_not_num                            ; yes: doesn't fit in int32
.an_ok:
     xor edx, edx                              ; edx = 0, xor sets ZF=1: numeric success
     ret                                       ; return value + ZF decision
.an_not_num:
     or edx, 1                                 ; force nonzero: ZF=0 unconditionally, not a numeric atom, at this exit the value is undefined and both callers ignore it
     ret                                       ; return ZF=0, value ignored
Equal:
     cmp rsp, [rel sl_stk_limit]               ; private-stack watermark: CAR-recursion depth is unbounded once lists are built programmatically, FPUT nests past any parser cap
     jb sl_stack_overflow                      ; runaway recursion -> named FATAL, not an AV
     push rbp                                  ; parity pad keeping the sl_eq_floatize calls at 0 mod 16, no rbp use, Equal is call-heavy
     mov r8d, eax                              ; r8d = lhs
     mov r9d, ecx                              ; r9d = rhs
     cmp eax, ecx                              ; raw tag equality first, covers ATOM/NUMBER/NIL
     jne .full_dispatch                        ; differ: run the type-dispatched comparison paths
     bt eax, SL_CONS_BIT                       ; bit31 set, FLOAT or CONS ?
     jnc .done                                 ; no: identical ATOM/NUMBER/NIL, ZF still 1 from the raw compare above
     bt eax, SL_NUMBER_BIT                     ; bit30 set as well, FLOAT pair ?
     jnc .done                                 ; no: identical CONS cells, ZF still 1
     jmp .float_path                           ; identical FLOATs: still compare numerically, bit-identical payloads could encode equal values
.full_dispatch:
     cmp r8d, SL_FLOAT_TAG                     ; lhs in the FLOAT band?
     jae .float_path                           ; yes: numeric comparison
.fd_rhs:
     cmp r9d, SL_FLOAT_TAG                     ; rhs in the FLOAT band?
     jae .float_path                           ; yes: numeric comparison
; --- cross-type: lhs NUMBER, rhs plain atom -> parse rhs numerically ---
     mov eax, r8d                              ; eax = lhs for tag test
     shr eax, 30                               ; eax = 2-bit tag code (NUMBER = 1)
     cmp eax, 1                                ; lhs a NUMBER?
     jne .coerce_rhs                           ; no: try the mirrored case
     test r9d, r9d                             ; rhs NIL?
     jz .not_equal                             ; yes: NUMBER != NIL
     test r9d, SL_TAG_MASK                     ; rhs CONS/NUMBER: any tag bit set?
     jnz .not_equal                            ; rhs CONS/NUMBER: no coercion
     mov eax, r9d                              ; eax = rhs atom for parsing
     call sl_atom_num                          ; eax = parsed rhs; ZF=1 if numeric
     jnz .not_equal                            ; not a numeric atom: not equal
     mov r10d, eax                             ; r10d = parsed rhs value, NumVal preserves r10
     mov eax, r8d                              ; eax = lhs NUMBER for unwrap
     call NumVal                               ; eax = lhs int32 value
     mov ecx, r10d                             ; ecx = parsed rhs value
     cmp eax, ecx                              ; numeric equality
     je .done                                  ; equal: ZF set, return
     jmp .not_equal                            ; values differ
.coerce_rhs:
; --- mirrored: rhs NUMBER, lhs plain atom -> parse lhs numerically ---
     mov eax, r9d                              ; eax = rhs for tag test
     shr eax, 30                               ; eax = 2-bit tag code (NUMBER = 1)
     cmp eax, 1                                ; rhs a NUMBER?
     jne .atom_or_cons                         ; no: no coercion possible
     test r8d, r8d                             ; lhs NIL?
     jz .not_equal                             ; yes: NIL != NUMBER
     test r8d, SL_TAG_MASK                     ; lhs CONS/FLOAT: any tag bit set?
     jnz .not_equal                            ; lhs CONS/FLOAT: no coercion
     mov eax, r8d                              ; eax = lhs atom for parsing
     call sl_atom_num                          ; eax = parsed lhs; ZF=1 if numeric
     jnz .not_equal                            ; not a numeric atom: not equal
     mov r10d, eax                             ; r10d = parsed lhs value
     mov eax, r9d                              ; eax = rhs NUMBER for unwrap
     call NumVal                               ; eax = rhs int32 value
     mov ecx, r10d                             ; ecx = parsed lhs value
     cmp eax, ecx                              ; numeric equality
     je .done                                  ; equal: ZF set, return
     jmp .not_equal                            ; values differ
.float_path:
; --- numeric float comparison: FLOAT vs FLOAT/NUMBER/decimal-atom ---
     mov eax, r8d                              ; eax = lhs, read before the call: sl_atom_float overwrites r8d
     mov dword [rel sl_float_tmp], r9d         ; spill rhs tagged, sl_atom_float overwrites r9d too, this replaces the push r9/pop r9 pair and keeps BOTH sl_eq_floatize calls at 0 mod 16, matching the entry pad above
     call sl_eq_floatize                       ; xmm0 = lhs float; ZF=1 comparable
     mov r9d, dword [rel sl_float_tmp]         ; reload rhs, mov writes no flags, so the jnz below still tests the conversion
     jnz .not_equal                            ; CONS or unparseable text: not equal
     vmovss dword [rel sl_float_tmp], xmm0     ; stash lhs float across the second call
     mov eax, r9d                              ; eax = rhs
     call sl_eq_floatize                       ; xmm0 = rhs float; ZF=1 comparable
     jnz .not_equal                            ; CONS or unparseable text: not equal
     vucomiss xmm0, dword [rel sl_float_tmp]   ; bit-level float equality test, memory operand: the stashed lhs is consumed once
     jp .not_equal                             ; NaN operand: unordered, never equal
     je .done                                  ; equal: ZF set, return
     jmp .not_equal                            ; values differ
.atom_text_compare:
     vpxor xmm0, xmm0, xmm0                    ; zero vector, NUL-lane compare
     lea r10, [r13 + r8 + SL_ATOM_TEXT_OFF]    ; lhs text, r10, not rsi: rsi is callee-saved and this function does not save it, see the overwrite list in this header
     lea r11, [r13 + r9 + SL_ATOM_TEXT_OFF]    ; rhs text, r11 is unused everywhere else in Equal
.atom_text_loop:
     vmovdqu xmm1, [r10]                       ; 16 lhs bytes, overread-safe: committed block
     vpcmpeqb xmm3, xmm1, xmm0                 ; lhs NUL lanes; defer rhs load while no NUL
     vptest xmm3, xmm3                         ; any lhs NUL?
     jnz .atom_text_nul                        ; yes: settle both texts here
     vpxor xmm3, xmm1, [r11]                   ; no NUL: whole-window compare, all-equal iff zero
     vptest xmm3, xmm3                         ; any difference?
     jnz .not_equal                            ; a real difference before any terminator
     add r10, 16                               ; whole window equal, slide both
     add r11, 16                               ; rhs window slides with it, lockstep
     jmp .atom_text_loop                       ; next window
.atom_text_nul:
     vpmovmskb r9d, xmm3                       ; lhs NUL mask, still live from the loop's compare
     vpcmpeqb xmm3, xmm1, [r11]                ; equality mask includes the terminator lane; rhs load folded
     vpmovmskb ecx, xmm3                       ; ecx = equality bitmask, lanes 0..15
     tzcnt r9d, r9d                            ; n = first lhs NUL position, 0..15
     inc r9d                                   ; include the terminator lane in the required mask
     not ecx                                   ; mismatch bits
     bzhi ecx, ecx, r9d                        ; mask to the required lanes, ZF=1 iff all matched
     jnz .not_equal                            ; a real difference at or before the terminator
.atom_text_equal:
     jmp .done                                 ; return with ZF from bzhi
.atom_or_cons:
     test r8d, r8d                             ; lhs NIL?
     jz .not_equal                             ; yes: NIL != atom/CONS
     test r9d, r9d                             ; rhs NIL?
     jz .not_equal                             ; yes: atom/CONS != NIL
; both sides non-NIL, non-FLOAT, non-NUMBER, atom or CONS
; check if both are plain atoms, no tag bits -> text compare
     test r8d, SL_TAG_MASK                     ; lhs has tag bits?
     jnz .cons_check                           ; lhs has tag bits: must be CONS
     test r9d, SL_TAG_MASK                     ; rhs has tag bits?
     jz .atom_text_compare                     ; rhs has no tag bits: both plain atoms, compare text content
.cons_check:
     bt r8d, SL_CONS_BIT                       ; lhs a cons cell?
     jnc .not_equal                            ; no: not equal
     bt r9d, SL_CONS_BIT                       ; rhs a cons cell?
     jnc .not_equal                            ; no: not equal
; both are CONS: resolve each cell once, cars recursively
     mov eax, r8d                              ; lhs tagged CONS
     btr eax, SL_CONS_BIT                      ; strip tag -> raw cell index, 32-bit op zero-extends
     lea r8, [r13 + rax*CONS_SCALE + CONS_BASE_OFF] ; lhs cell address
     mov eax, r9d                              ; rhs tagged CONS
     btr eax, SL_CONS_BIT                      ; strip tag -> raw cell index
     lea r9, [r13 + rax*CONS_SCALE + CONS_BASE_OFF] ; rhs cell address
     mov eax, [r8]                             ; lhs car field
     mov ecx, [r9]                             ; rhs car field
     push r8                                   ; save lhs cell across recursive Equal, base 0 -> 8
     push r9                                   ; save rhs cell, 8 -> 0: call runs aligned
     call Equal                                ; recurse on cars
     pop r9                                    ; restore rhs cell
     pop r8                                    ; restore lhs cell
     jne .not_equal                            ; cars differ: not equal
; cars equal: compare cdrs via tail jump, O(1) stack on linear lists
     mov eax, [r8 + CONS_CDR_OFF]              ; lhs cdr field
     mov ecx, [r9 + CONS_CDR_OFF]              ; rhs cdr field
     pop rbp                                   ; drop entry pad: tail-call parity
     jmp Equal                                 ; tail-call: ZF propagates to caller
.not_equal:
     test rsp, rsp                             ; RSP is never 0 mid-execution -> ZF=0 unconditionally, float path overwrites r8d/r9d via sl_atom_float, cmp r8d,r9d was unsound
.done:
     pop rbp                                   ; drop entry pad, does not touch ZF
     ret                                       ; return ZF to caller
; ---- %undef cleanup: file-local defines leave scope here, house convention ----
%undef SL_LIDX_LIST_OFF
%undef SL_LIDX_IDX_OFF
%undef SL_LIDX_FRAME
%undef SL_AN_MAX_DIGITS
%undef SL_AN_FOLD_SCALE
%undef SL_AN_INT32_MIN_ABS
