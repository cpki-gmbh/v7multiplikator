; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; Prefix arithmetic operators, binary
; ============================================================================
.p_sum:
     NUM_BINARY_PROMOTED                       ; eax=lhs, ecx=rhs, or floats xmm0/xmm1
     test edx, edx                             ; float operands?
     jnz .pf_sum_f                             ; yes: SSE float add
     add eax, ecx                              ; eax = lhs + rhs
     cdqe                                      ; rax = exact result, a sum of two 30-bit values always fits in 32 bits
     NUM_RANGE_CHECK rax, "Add: integer result does not fit the 30-bit payload range" ; out of range? FATAL, a wrapped positive sum reads back negative
     jmp MakeNum                               ; wrap as NUMBER
.pf_sum_f:
     vaddss xmm0, xmm0, xmm1                   ; xmm0 = lhs + rhs
     jmp MakeFloat                             ; wrap as FLOAT
.p_diff:
     call EvalNextArg                          ; eax = first arg, tagged
     test r12d, r12d                           ; stream is NIL?
     jz .p_diff_unary                          ; yes: unary negation, no second arg
     mov ecx, r12d                             ; ecx = stream cell, tagged
     btr ecx, SL_CONS_BIT                      ; strip tag -> raw cons index
     mov ebx, [r13 + rcx*CONS_SCALE + CONS_BASE_OFF] ; ebx = car,stream = peeked head
     cmp ebx, [rel sl_dispatch_highwater]      ; tagged head, or a post-builtin, user name?
     jae .p_diff_binary                        ; yes: value-start, names/lparen/strings head calls
     bt dword [rel sl_op_atom_bitmap], ebx     ; operator atom?, bt mem,reg: full-array bit index, MUST stay after the highwater guard, which is the only thing keeping it inside the 1 KiB bitmap
     jc .p_diff_unary                          ; yes: an operator follows the first arg -> unary
     cmp ebx, [rel atom_sl_rparen]             ; ?
     je .p_diff_unary                          ; yes: unary, no second arg
; next head is a value-start: do binary subtraction
.p_diff_binary:
     push rax                                  ; save lhs tagged
     call EvalNextArg                          ; eax = second arg, tagged
     mov edx, [rsp]                            ; edx = lhs tagged
     call sl_promote_pair                      ; dispatch on operand tags
     add rsp, 8                                ; discard lhs slot
     test edx, edx                             ; float operands?
     jnz .pf_diff_f                            ; yes: SSE float subtract
     sub eax, ecx                              ; eax = lhs - rhs
     cdqe                                      ; rax = exact result, a difference of two 30-bit values always fits in 32 bits
     NUM_RANGE_CHECK rax, "Subtract: integer result does not fit the 30-bit payload range" ; out of range? FATAL
     jmp MakeNum                               ; wrap as NUMBER
.p_diff_unary:
; unary negation: negate the single evaluated argument
     mov edx, eax                              ; edx = copy for tag test
     shr edx, 30                               ; edx = 2-bit tag code (NUMBER = 1, FLOAT = 3)
     cmp edx, 1                                ; NUMBER?
     je .p_diff_unary_int                      ; yes: negate integer
     cmp edx, 3                                ; FLOAT?
     je .p_diff_unary_float                    ; yes: negate float
     FATAL_VALUE "Subtract: unary negation requires a number", eax ; eax is still the TAGGED argument, only edx was written above, so the offender renders with its type
.p_diff_unary_int:
     call NumVal                               ; eax = sign-extended raw value, tag stripped
     neg eax                                   ; eax = negated
     cdqe                                      ; rax = exact result, 64-bit
     NUM_RANGE_CHECK rax, "Subtract: negating the most negative NUMBER is not representable" ; -,-2^29 needs +2^29, one past the 30-bit payload maximum, same edge as .p_neg
     jmp MakeNum                               ; mask + tag
.p_diff_unary_float:
; negate float: XOR the sign bit
     call FloatVal                             ; xmm0 = float value, from tagged eax
     vxorps xmm0, xmm0, [rel flt_sign_mask]    ; flip sign bit
     jmp MakeFloat                             ; wrap as FLOAT
.pf_diff_f:
     vsubss xmm0, xmm0, xmm1                   ; xmm0 = lhs - rhs
     jmp MakeFloat                             ; wrap as FLOAT
.p_prod:
     NUM_BINARY_PROMOTED                       ; eax=lhs, ecx=rhs, or floats xmm0/xmm1
     test edx, edx                             ; float operands?
     jnz .pf_prod_f                            ; yes: SSE float multiply
     cdqe                                      ; rax = lhs, sign-extended, so the multiply below is exact
     movsxd rcx, ecx                           ; rcx = rhs, sign-extended
     imul rax, rcx                             ; rax = exact 64-bit product, both operands are 30-bit payloads, so it fits in 60
     NUM_RANGE_CHECK rax, "Multiply: integer result does not fit the 30-bit payload range" ; out of range? FATAL
     jmp MakeNum                               ; wrap as NUMBER
.pf_prod_f:
     vmulss xmm0, xmm0, xmm1                   ; xmm0 = lhs * rhs
     jmp MakeFloat                             ; wrap as FLOAT
.p_quot:
     NUM_BINARY_PROMOTED                       ; eax=lhs, ecx=rhs, or floats xmm0/xmm1
     test edx, edx                             ; float operands?
     jnz .pf_quot_f                            ; yes: SSE float divide
     test ecx, ecx                             ; divide by zero?
     jz .quot_zero                             ; yes: division by zero
     cdq                                       ; sign-extend for idiv
     idiv ecx                                  ; eax = lhs / rhs
     cdqe                                      ; rax = exact quotient, the 30-bit operand bound means the dividend is never INT_MIN, so 32-bit idiv cannot trap
     NUM_RANGE_CHECK rax, "Divide: integer result does not fit the 30-bit payload range" ; lhs=-2^29, rhs=-1 -> +2^29: FATAL, the mask cannot detect it, it truncates
     jmp MakeNum                               ; wrap as NUMBER
.pf_quot_f:
     vdivss xmm0, xmm0, xmm1                   ; xmm0 = lhs / rhs, IEEE inf on /0
     jmp MakeFloat                             ; wrap as FLOAT
.quot_zero:
     FATAL "SectorLogo: division by zero"      ; divisor was zero: abort
.p_rem:
     NUM_BINARY_PROMOTED                       ; eax=lhs, ecx=rhs, or floats xmm0/xmm1
     test edx, edx                             ; float operands?
     jnz .pf_rem_f                             ; yes: x87 float remainder
     test ecx, ecx                             ; divide by zero?
     jz .rem_zero                              ; yes: modulo by zero
     cdq                                       ; sign-extend for idiv
     idiv ecx                                  ; eax = quotient, edx = remainder
     mov eax, edx                              ; return remainder
     jmp MakeNum                               ; wrap as NUMBER
.pf_rem_f:
     call sl_fmod                              ; xmm0 = fmod,lhs, rhs ; FATAL rhs==0
     jmp MakeFloat                             ; wrap as FLOAT
.rem_zero:
     FATAL "SectorLogo: modulo by zero"        ; divisor was zero: abort
; .pf_rem_f ends in `jmp MakeFloat` and .rem_zero ends in a FATAL, both
; out-of-line exits that nothing falls into, so the .shift_count_fatal block
; below is safely fenced between them.
.shift_count_fatal:
     FATAL "SectorLogo: shift count must be a NUMBER in the range 0..31" ; a negative or >= 32 count is masked to 5 bits by x86, so the
; operator would silently answer a different question, no semicolon inside
; the string: the comment checker aligns on ; and would splice one in
.p_pow:
     NUM_BINARY_PROMOTED                       ; eax=base, ecx=exponent, or floats xmm0/xmm1
     test edx, edx                             ; float operands?
     jnz .pf_pow_f                             ; yes: x87 float power
; int path: eax = base, ecx = exponent, raw ints from promote
     test ecx, ecx                             ; exponent <= 0?
     jl .pow_neg                               ; yes: negative exponent -> 0
     je .pow_zero                              ; exponent == 0 -> 1
     mov r8d, eax                              ; r8d = base, mutable
     mov eax, 1                                ; rax = accumulator = 1, lives in RAX: NUM_RANGE_CHECK needs 64 bits, MakeNum needs EAX
.pow_loop:
     movsxd r9, r8d                            ; r9 = base, sign-extended, r8d zero-extends into r8, so it cannot index directly, hoisted above the parity test: r8d does not change until the bottom-of-loop store
     test cl, 1                                ; exponent odd?
     jz .pow_even                              ; no: skip multiply
     imul rax, r9                              ; accumulator *= base, rax persists across iterations, do NOT reload it from eax
     NUM_RANGE_CHECK rax, "Power: integer result does not fit the 30-bit payload range" ; out of range? FATAL
.pow_even:
     shr ecx, 1                                ; exponent >>= 1
     jz .pow_done                              ; no bits left: the final square would have no consumer
     imul r9, r9                               ; base *= base, only while another exponent bit remains
     NUM_RANGE_CHECK r9, "Power: integer result does not fit the 30-bit payload range" ; out of range? FATAL
     mov r8d, r9d                              ; r8d = range-checked square
     jmp .pow_loop                             ; more bits: continue
.pow_done:
     jmp MakeNum                               ; wrap as NUMBER, EAX already holds the accumulator's low half, NUM_RANGE_CHECK at :131/:136 guarantees it fits the payload, and MakeNum reads EAX
.pf_pow_f:
     call sl_fpow                              ; xmm0 = base^exp; negative base handled for integral exponents
     jmp MakeFloat                             ; wrap as FLOAT
.pow_zero:
     mov eax, 1                                ; x^0 = 1
     jmp MakeNum                               ; wrap as NUMBER
.pow_neg:
     cmp eax, 1                                ; 1^,-n = 1, |base| == 1 does not truncate to 0
     je .pow_neg_one                           ; base is 1
     cmp eax, -1                               ; -1 ^,-n =, -1 ^n = +-1
     jne .pow_neg_zero                         ; |base| > 1: truncates to 0
     test cl, 1                                ; n odd?
     jz .pow_neg_one                           ; even n -> 1
     mov eax, -1                               ; odd n -> -1
     jmp MakeNum                               ; wrap as NUMBER
.pow_neg_one:
     mov eax, 1                                ; result 1
     jmp MakeNum                               ; wrap as NUMBER
.pow_neg_zero:
     xor eax, eax                              ; |base| > 1: x^,-n = 0 for integer math
     jmp MakeNum                               ; wrap as NUMBER
.p_sqrt:
     call EvalNextArg                          ; eax = n, tagged
     cmp eax, SL_FLOAT_TAG                     ; FLOAT argument?
     jae .pf_sqrt_f                            ; yes: float path
     call NumVal                               ; eax = n
     test eax, eax                             ; n <= 0?
     jg .sqrt_pos                              ; n > 0: run the Newton loop below
     jz .sqrt_zero                             ; n == 0: sqrt,0 = 0
; not 0. jle would have swept it into .sqrt_zero silently.
     FATAL_VALUE "SquareRoot: argument must be >= 0", eax ; eax is the RAW int32, NumVal, line 155 stripped the tag, so the offender renders as a bare number, not a tagged type
.sqrt_pos:
     vcvtsi2sd xmm0, xmm0, eax                 ; exact int32 -> binary64
     vsqrtsd xmm0, xmm0, xmm0                  ; binary64 square root, correctly rounded
     vcvttsd2si eax, xmm0                      ; exact floor(sqrt(n)) for 0 < n <= 2^29-1, truncation
     jmp MakeNum                               ; wrap as NUMBER
.sqrt_zero:
     xor eax, eax                              ; sqrt,0 = 0, reached only from n == 0 now that n < 0 FATALs
     jmp MakeNum                               ; wrap as NUMBER
.pf_sqrt_f:
     call FloatVal                             ; xmm0 = float value
     vsqrtss xmm0, xmm0, xmm0                  ; xmm0 = sqrt,x, negative input -> NaN
     jmp MakeFloat                             ; wrap as FLOAT
.p_abs:
     call EvalNextArg                          ; eax = value, tagged
     cmp eax, SL_FLOAT_TAG                     ; FLOAT argument?
     jae .pf_abs_f                             ; yes: float path
     call NumVal                               ; eax = raw value
     test eax, eax                             ; negative?
     jns .abs_done                             ; no: keep as-is
     neg eax                                   ; negate
.abs_done:
     cdqe                                      ; rax = exact result. -2^29 negates to +2^29, ONE PAST the payload maximum, MakeNum's mask would turn it back into -2^29, making AbsoluteValue a silent no-op at that one value
     NUM_RANGE_CHECK rax, "AbsoluteValue: |value| does not fit the 30-bit payload range" ; |x| = 2^29 is not representable as a SectorLogo NUMBER, so there is no correct value to return
     jmp MakeNum                               ; wrap as NUMBER
.pf_abs_f:
     call FloatVal                             ; xmm0 = float value
     vandps xmm0, xmm0, [rel flt_abs_mask]     ; xmm0 = |x|, clear sign bit
     jmp MakeFloat                             ; wrap as FLOAT
.p_neg:
     call EvalNextArg                          ; eax = value, tagged
     cmp eax, SL_FLOAT_TAG                     ; FLOAT argument?
     jae .pf_neg_f                             ; yes: float path
     call NumVal                               ; eax = raw value
     neg eax                                   ; unary negate
     cdqe                                      ; rax = exact result. Negating -2^29 needs +2^29, one past the payload maximum, without this check MakeNum masks it back to -2^29 and Negate is a silent no-op there
     NUM_RANGE_CHECK rax, "Negate: negating the most negative NUMBER is not representable" ; the one NUMBER whose negation has no result in the 30-bit payload
     jmp MakeNum                               ; wrap as NUMBER
.pf_neg_f:
     call FloatVal                             ; xmm0 = float value
     vxorps xmm0, xmm0, [rel flt_sign_mask]    ; xmm0 = -x, flip sign bit
     jmp MakeFloat                             ; wrap as FLOAT
.p_max:
     NUM_BINARY_PROMOTED                       ; eax=lhs, ecx=rhs, or floats xmm0/xmm1
     test edx, edx                             ; float operands?
     jnz .pf_max_f                             ; yes: vmaxss
     cmp ecx, eax                              ; rhs > lhs?
     cmovg eax, ecx                            ; yes: return rhs, branchless max
     jmp MakeNum                               ; wrap as NUMBER
.pf_max_f:
     vmaxss xmm0, xmm0, xmm1                   ; xmm0 = max,lhs, rhs
     jmp MakeFloat                             ; wrap as FLOAT
.p_min:
     NUM_BINARY_PROMOTED                       ; eax=lhs, ecx=rhs, or floats xmm0/xmm1
     test edx, edx                             ; float operands?
     jnz .pf_min_f                             ; yes: vminss
     cmp ecx, eax                              ; rhs < lhs?
     cmovl eax, ecx                            ; yes: return rhs, branchless min
     jmp MakeNum                               ; wrap as NUMBER
.pf_min_f:
     vminss xmm0, xmm0, xmm1                   ; xmm0 = min,lhs, rhs
     jmp MakeFloat                             ; wrap as FLOAT
.p_random:
     call EvalNextArg                          ; eax = limit, tagged
     call NumVal                               ; eax = limit, sign-extended raw
     mov ecx, eax                              ; ecx = limit, ZERO-extended, the unsigned divisor `div ecx` needs
     test eax, eax                             ; limit <= 0?, test EAX, still the sign-extended raw: `test ecx,ecx` here can NEVER fire for a negative limit, because ecx is the ZERO-extended copy, Random -5 would divide by 4294967291
     jle .rand_zero                            ; yes: return 0
; AES-NI PRNG: state += Weyl increment, one AES round, use upper 32 bits
     mov rax, [rel sl_rng_state]               ; load current state
     cmp dword [rel sl_real_random], 0         ; REAL-RANDOM enabled?
     jz .rand_no_rdtsc                         ; no: skip RDTSC
     rdtsc                                     ; edx:eax = timestamp counter
     shl rdx, 32                               ; shift high dword into position
     or rax, rdx                               ; combine into 64-bit TSC value
     xor rax, [rel sl_rng_state]               ; XOR TSC with pre-increment state
.rand_no_rdtsc:
     add rax, [rel sl_weyl_inc]                ; state += Weyl increment, BOTH paths; kept after rdtsc which overwrites eax
     mov [rel sl_rng_state], rax               ; store new state
     vmovq xmm0, rax                           ; xmm0 = state
     vpinsrq xmm0, xmm0, [rel sl_aes_block_hi], 1 ; xmm0 = [state | block_hi], imm8=1 selects the HIGH lane
     aesenc xmm0, [rel sl_aes_keys_combined]   ; one AES round for whitening, legacy AES-NI encoding (no VAES needed), 16-byte operand
     vpextrd eax, xmm0, 3                      ; low 32 bits of the selected AES lane
     mul ecx                                   ; edx = floor(draw * limit / 2^32), bounded by [0, limit-1]
     mov eax, edx                              ; eax = random in [0, limit
     jmp MakeNum                               ; wrap as NUMBER
.rand_zero:
     xor eax, eax                              ; limit <= 0, see the jle at :240 -> 0
     jmp MakeNum                               ; wrap as NUMBER
.p_real_random:
     call EvalNextArg                          ; eax = flag, tagged
     mov r10d, eax                             ; park tagged flag, FATAL_VALUE input; NumVal overwrites eax
     call NumVal                               ; eax = raw 0/1
     cmp eax, 1                                ; unsigned <= 1, a negative flags this too ?
     ja .rr_bad                                ; not 0/1: name the offender
     mov ecx, [rel sl_real_random]             ; ecx = old flag, return value, loaded BEFORE the store
     mov [rel sl_real_random], eax             ; 0 or 1: store verbatim
     mov eax, ecx                              ; eax = old flag
     jmp MakeNum                               ; wrap old flag as NUMBER and return
.rr_bad:
     FATAL_VALUE "RealRandom flag must be 0 or 1", r10d ; name the offender
; ============================================================================
; Prefix bitwise operators
; ============================================================================
.p_bitand:
     PREFIX_BITWISE_OP and                     ; & : logical AND
.p_bitor:
     PREFIX_BITWISE_OP or                      ; | : logical OR
.p_bitxor:
     PREFIX_BITWISE_OP xor                     ; ^ : logical XOR
.p_bitnot:
     call EvalNextArg                          ; eax = value, tagged
     call NumVal                               ; eax = raw value
     not eax                                   ; bitwise complement
     jmp MakeNum                               ; wrap as NUMBER
.p_lshift:
     NUM_BINARY_PROLOGUE                       ; eax=lhs, ecx=rhs
     cdqe                                      ; rax = lhs, sign-extended, so a 64-bit shift cannot hide a wrap past bit 31
     cmp ecx, 31                               ; count outside 0..31?, an explicit check replaces the mask: x86 would mask a negative or >= 32 count to 5 bits and answer a different question than the script asked
     ja .shift_count_fatal                     ; yes: name it
     shl rax, cl                               ; rax = lhs << count, cl holds the validated count, x86 takes a variable shift count only from cl
     NUM_RANGE_CHECK rax, "ShiftLeft: integer result does not fit the 30-bit payload range" ; out of range? FATAL
     jmp MakeNum                               ; wrap as NUMBER
.p_rshift:
     NUM_BINARY_PROLOGUE                       ; eax=lhs, ecx=rhs
     cmp ecx, 31                               ; count outside 0..31?, see .p_lshift
     ja .shift_count_fatal                     ; yes: name it
     sar eax, cl                               ; arithmetic shift right; |result| <= |lhs|, so it is always representable
     jmp MakeNum                               ; wrap as NUMBER
; ============================================================================
; Prefix relational / boolean operators
; ============================================================================
.p_equalp:
     call EvalNextArg                          ; eax = lhs, tagged
     push rax                                  ; save lhs
     call EvalNextArg                          ; eax = rhs, tagged
     pop rcx                                   ; rcx = lhs, tagged
     call Equal                                ; structural compare, eax,ecx -> ZF
     je .ret_true                              ; equal: return TRUE
     jmp .ret_false                            ; not equal: return NIL
.p_notequalp:
     call EvalNextArg                          ; eax = lhs, tagged
     push rax                                  ; save lhs
     call EvalNextArg                          ; eax = rhs, tagged
     pop rcx                                   ; rcx = lhs, tagged
     call Equal                                ; structural compare, eax,ecx -> ZF
     jne .ret_true                             ; not equal = true
     jmp .ret_false                            ; equal: return NIL
.p_greaterp:
     PREFIX_RELATIONAL_CMP g, a                ; lhs > rhs ?, jg for ints, ja for floats
.p_lessp:
     PREFIX_RELATIONAL_CMP l, b                ; lhs < rhs ?, jl for ints, jb for floats
.p_gte:
     PREFIX_RELATIONAL_CMP ge, ae              ; lhs >= rhs ?, jge for ints, jae for floats
.p_lte:
     PREFIX_RELATIONAL_CMP le, be              ; lhs <= rhs ?, jle for ints, jbe for floats
; --- && / || have NO prefix form: infix only, dispatch_list has no expression
; entries for their atoms, so a prefix spelling self-evaluates as a dead name ---
.p_bang:
     call EvalPrimary                          ; eax = operand: ONE primary, so `! a && b` is, !a && b like C
     SL_FALSY_JUMP eax, .ret_true              ; falsy operand: TRUE atom
     jmp .ret_false                            ; truthy operand: NIL
.p_zerop:
     call EvalNextArg                          ; eax = value, tagged
     cmp eax, SL_FLOAT_TAG                     ; FLOAT argument?
     jae .pf_zerop_f                           ; yes: float path
     call NumVal                               ; eax = raw value
     test eax, eax                             ; zero?
     jz .ret_true                              ; yes: return TRUE
     jmp .ret_false                            ; no: return NIL
.pf_zerop_f:
     SL_FLOAT_ZERO_JUMP eax, .ret_true         ; +/-0.0: return TRUE
     jmp .ret_false                            ; any other float: return NIL
.p_numberp:
     call EvalNextArg                          ; eax = tagged value
     TAG_IS_JUMP eax, SL_NUMBER_TAG, .ret_true ; NUMBER tag?
     jmp .ret_false                            ; no: return NIL
; --- LIST? predicate: bit31 set AND bit30 clear = cons cell, not FLOAT ---
.p_listp:
     call EvalNextArg                          ; eax = tagged value
     TAG_IS_JUMP eax, SL_CONS_TAG, .ret_true   ; CONS tag, bit31=1, bit30=0 ?
     jmp .ret_false                            ; no: return NIL
.p_isstringatom:
     call EvalNextArg                          ; eax = tagged value
     TAG_GUARD_ATOM .ret_false, .ret_false     ; NIL or CONS/NUMBER/FLOAT: not a string atom
     cmp byte [r13 + rax], SL_TYPE_STRING      ; atom type byte = STRING?
     je .ret_true                              ; yes: quoted-string atom
     jmp .ret_false                            ; no, WORD atom : return NIL
.p_iswordatom:
     call EvalNextArg                          ; eax = tagged value
     TAG_GUARD_ATOM .ret_false, .ret_false     ; NIL or CONS/NUMBER/FLOAT: not a word atom
     cmp byte [r13 + rax], SL_TYPE_WORD        ; atom type byte = WORD?
     je .ret_true                              ; yes: bare-word atom
     jmp .ret_false                            ; no, STRING atom : return NIL
; ============================================================================
; Float primitives, I2F F2I FLOOR FROUND FLOATP FLOG2 FEXP2 FLN FMOD
; ============================================================================
.p_i2f:
     call EvalNextArg                          ; eax = NUMBER, or FLOAT passthrough
     call sl_to_float                          ; xmm0 = float, atoms FATAL
     jmp MakeFloat                             ; wrap as FLOAT
.pf_f2i_bad:
     FATAL_VALUE "AsInteger: value is NaN, no clamped value exists", r10d ; NaN has no nearest value: name the offender instead of a silent mask
.p_f2i:
     call EvalNextArg                          ; eax = FLOAT, or NUMBER converts too
     mov r10d, eax                             ; park tagged arg, FATAL_VALUE input; sl_to_float overwrites eax
     call sl_to_float                          ; xmm0 = float
; OWNER-DIRECTED EXCEPTION to the range-FATAL law: a finite float whose
; truncated value lands outside the 30-bit NUMBER payload SATURATES to the
; nearest representable NUMBER instead of FATALing, conversion consumers
; the tuner's distance math multiply large ratios, a saturation value keeps
; the computation alive, and the caller's own legality window bounds it.
; NaN keeps the loud FATAL above: it has no nearest value.
     vcomiss xmm0, xmm0                        ; NaN?, a value compares unordered to itself -> PF set
     jp .pf_f2i_bad                            ; NaN: name the offender
     vmaxss xmm0, xmm0, [rel sl_f2i_min_f]     ; saturate below: anything under -2^29 becomes exactly the floor
     vminss xmm0, xmm0, [rel sl_f2i_cap_f]     ; saturate above: anything at or past 2^29 becomes exactly the boundary
     vcvttss2si eax, xmm0                      ; exact trunc toward 0 of the clamped value, worst case 2^29 itself
     mov edx, SL_PAYLOAD_MAX                   ; edx = +2^29-1, the boundary above converts one past the payload max
     cmp eax, edx                              ; did the ceiling clamp land on the boundary?
     cmovg eax, edx                            ; saturate that single value down to the payload max
     jmp MakeNum                               ; wrap as NUMBER
.p_floor:
     call EvalNextArg                          ; eax = FLOAT or NUMBER
     call sl_to_float                          ; xmm0 = float
     vroundss xmm0, xmm0, xmm0, 0x09           ; round toward -inf, mode 1 + no-exc bit
     jmp MakeFloat                             ; wrap as FLOAT
.p_fround:
     call EvalNextArg                          ; eax = FLOAT or NUMBER
     call sl_to_float                          ; xmm0 = float
     vroundss xmm0, xmm0, xmm0, 0x08           ; round half-to-even, mode 0 + no-exc bit
     jmp MakeFloat                             ; wrap as FLOAT
.p_floatp:
     call EvalNextArg                          ; eax = tagged value
     cmp eax, SL_FLOAT_TAG                     ; FLOAT tag?
     jae .ret_true                             ; yes: true
     jmp .ret_false                            ; no: return NIL
.p_flog2:
     call EvalNextArg                          ; eax = x, positive finite
     call sl_to_float                          ; xmm0 = float
     call sl_flog2                             ; xmm0 = log2,x ; FATAL x <= 0
     jmp MakeFloat                             ; wrap as FLOAT
.p_fexp2:
     call EvalNextArg                          ; eax = x, any finite
     call sl_to_float                          ; xmm0 = float
     call sl_fexp2                             ; xmm0 = 2^x
     jmp MakeFloat                             ; wrap as FLOAT
.p_fln:
     call EvalNextArg                          ; eax = x, positive finite
     call sl_to_float                          ; xmm0 = float
     call sl_fln                               ; xmm0 = ln,x, single binary32 rounding; FATAL x <= 0
     jmp MakeFloat                             ; wrap as FLOAT
.p_fmod:
     NUM_BINARY_PROMOTED                       ; eax=lhs, ecx=rhs, or floats xmm0/xmm1
     test edx, edx                             ; float operands?
     jnz .pf_fmod_f                            ; yes: x87 remainder directly
     test ecx, ecx                             ; divisor zero?, idiv would raise #DE
     jz .pfm_int_zero                          ; yes: hand sl_fmod the +0.0 it FATALs on
     cdq                                       ; sign-extend lhs for idiv
     idiv ecx                                  ; eax = lhs / rhs, edx = lhs % rhs, exact; 30-bit operands cannot overflow or trap
     vcvtsi2ss xmm0, xmm0, edx                 ; xmm0 =, float, lhs % rhs, the single rounding is on the remainder, the smallest quantity involved
     jmp MakeFloat                             ; wrap as FLOAT
.pfm_int_zero:
     vxorps xmm0, xmm0, xmm0                   ; dividend is irrelevant, the handler FATALs before reading it
     vxorps xmm1, xmm1, xmm1                   ; +0.0 divisor -> sl_fmod's .sfm_zero_div raises the same FATAL as
.pf_fmod_f:
     call sl_fmod                              ; xmm0 = fmod,lhs, rhs ; FATAL rhs==0
     jmp MakeFloat                             ; wrap as FLOAT
; --- shared boolean return points ---
.ret_true:
     mov eax, [rel atom_sl_t]                  ; return TRUE atom
     ret                                       ; return to caller
.ret_false:
     xor eax, eax                              ; return NIL, false
     ret                                       ; return to caller
; ============================================================================
; List primitives
; ============================================================================
.p_first:
     call EvalNextArg                          ; eax = list value
     test eax, eax                             ; empty list?
     jz .ret_false                             ; yes: FIRST of [] is NIL, legitimate empty result
     TAG_IS_JUMP_NE eax, SL_CONS_TAG, .pf_badtype ; non-list: reject, plain atom or NUMBER/FLOAT
     call Car                                  ; eax =, car list = first element
     ret                                       ; return to caller
.pf_badtype:
     FATAL_VALUE "FIRST requires a list", eax  ; name the offending value
.p_butfirst:
     call EvalNextArg                          ; eax = list value
     test eax, eax                             ; empty list?
     jz .ret_false                             ; yes: BUTFIRST of [] is NIL, legitimate empty result
     TAG_IS_JUMP_NE eax, SL_CONS_TAG, .pbf_badtype ; non-list: reject, plain atom or NUMBER/FLOAT
     call Cdr                                  ; eax =, cdr list = rest
     ret                                       ; return to caller
.pbf_badtype:
     FATAL_VALUE "BUTFIRST requires a list", eax ; name the offending value
.p_fput:
     call EvalNextArg                          ; eax = item to prepend
     push rax                                  ; save item
     call EvalNextArg                          ; eax = list
     mov edx, eax                              ; edx = list, cdr
     pop rax                                   ; eax = item, car
     jmp Cons                                  ; eax =, item.list
; ============================================================================
; String operation primitives
; ============================================================================
.p_substr:
     jmp sl_str_substr                         ; Substring
.p_find:
     jmp sl_str_find                           ; Search
.p_strlower:
     jmp sl_str_lower                          ; Lowercase
.p_strupper:
     jmp sl_str_upper                          ; Uppercase
.p_strreplace:
     jmp sl_str_replace                        ; Replace
.p_strtonum:
     jmp sl_str_to_num                         ; WordToNumber
.p_strlen:
     jmp sl_str_strlen                         ; TextLength
.p_explode:
     jmp sl_str_explode                        ; Explode
.p_str_tokens:
     jmp sl_str_tokens                         ; Words
.p_split_ngrams:
     jmp sl_split_ngrams                       ; NGrams
.p_lindex:
     jmp sl_lindex                             ; Item
.p_sort:
     jmp sl_sort                               ; Sort
.p_reverse:
     jmp sl_reverse                            ; Reverse
.p_hash_float:
     jmp sl_hash_float                         ; HashNumber
.p_length:
     jmp sl_length                             ; Count
.p_lset:
     jmp sl_lset                               ; SetItem
.p_setcar:
     jmp sl_setcar                             ; SetCar
.p_setcdr:
     jmp sl_setcdr                             ; SetCdr
.p_samevalue:
     jmp sl_samevalue                          ; SameValue
.p_asword:
     jmp sl_asword                             ; AsWord
.p_lfind:
     jmp sl_lfind                              ; FindItem
.p_lfindge:
     jmp sl_lfindge                            ; FindFirstAtLeast
.p_lfindle:
     jmp sl_lfindle                            ; FindFirstAtMost
.p_int_to_char:
     jmp sl_str_int_to_char                    ; IntToChar
.p_char_to_int:
     jmp sl_str_char_to_int                    ; CharToInt
.p_do_vq:
     jmp sl_do_vq                              ; NearestMatch
; ============================================================================
; FFI, C-CALL primitive
; ============================================================================
.p_c_call:
     jmp sl_ffi_invoke                         ; CallLibrary
; ============================================================================
; Console primitives, expression position
; ============================================================================
.p_getchar:
     jmp sl_console_getchar                    ; ReadChar
.p_console_24bit:
     jmp sl_console_24bit                      ; Console-24bit
; --- ENABLETRACE level, expression -> returns old level ---
.p_enabletrace:
     call EvalNextArg                          ; eax = trace level, tagged
     mov r10d, eax                             ; park tagged level, FATAL_VALUE input; NumVal overwrites eax
     call NumVal                               ; eax = raw int
     mov ecx, [rel sl_trace_level]             ; ecx = old level, return value, loaded BEFORE the range check
     cmp eax, SL_TRACE_MAX_LEVEL               ; unsigned compare: also catches every negative, its unsigned image is >= 0x80000000 > MAX_LEVEL, one test covers both bounds
     ja .pet_bad_level                         ; out of range: name the offender
     mov [rel sl_trace_level], eax             ; in range: set new trace level verbatim
     mov eax, ecx                              ; eax = old level, return as NUMBER
     jmp MakeNum                               ; wrap old level as NUMBER and return
.pet_bad_level:
     FATAL_VALUE "EnableTrace level must be 0..3", r10d ; name the offender
; --- ENABLESTRICTNAMES flag, expression -> returns old flag ---
.p_enablestrictnames:
     call EvalNextArg                          ; eax = flag, tagged
     mov r10d, eax                             ; park tagged flag, FATAL_VALUE input; NumVal overwrites eax
     call NumVal                               ; eax = raw 0/1
     cmp eax, 1                                ; unsigned <= 1, a negative flags this too ?
     ja .pesn_bad                              ; no: FATAL
     mov ecx, [rel sl_strict_names]            ; ecx = old flag, return value, loaded BEFORE the store
     mov [rel sl_strict_names], eax            ; 0 or 1: store verbatim
     mov eax, ecx                              ; eax = old flag
     jmp MakeNum                               ; wrap old flag as NUMBER and return
.pesn_bad:
     FATAL_VALUE "EnableStrictNames flag must be 0 or 1", r10d ; name the offender
; ============================================================================
; V7 host primitives, expression position
; ============================================================================
.p_v7_encode_word:
     call EvalNextArg                          ; eax = word atom
     jmp sl_v7_encode_word                     ; delegate to v7prim
.p_v7_db_get:
     call EvalNextArg                          ; eax = key atom
     jmp sl_v7_db_get                          ; delegate to v7prim
.p_v7_get_token:
     call EvalNextArg                          ; eax = position, tagged number
     jmp sl_v7_get_token                       ; delegate to v7prim
.p_v7_pos:
     jmp sl_v7_pos                             ; delegate to v7prim
.p_v7_get_token_pos:
     jmp sl_v7_get_token_pos                   ; delegate to v7prim, query-local token-stream position
.p_v7_vocab_size:
     jmp sl_v7_vocab_size                      ; delegate to v7prim
.p_v7_dim:
     jmp sl_v7_dim                             ; delegate to v7prim
.p_v7_layers:
     jmp sl_v7_layers                          ; delegate to v7prim
.p_v7_eos:
     jmp sl_v7_eos                             ; delegate to v7prim
.p_v7_stream:
     jmp sl_v7_stream                          ; delegate to v7prim
.p_v7_str_len:
     jmp sl_v7_str_len                         ; delegate to v7prim
; PREFIX_BITWISE_OP follows the same rule for the same reason: defined in the macro
; home, expanded only by the three prefix bitwise handlers above.
%unmacro PREFIX_BITWISE_OP 1                   ; remove: the three prefix bitwise expansions are above, so no later include can use it
%unmacro NUM_BINARY_PROMOTED 0                 ; remove: the prefix arithmetic operators above are its only expansions
%unmacro NUM_BINARY_PROLOGUE 0                 ; remove: .p_lshift / .p_rshift above are its only expansions
%unmacro TAG_IS_JUMP 3                         ; remove: the tag-dispatch test sites above are its only expansions
%unmacro PREFIX_RELATIONAL_CMP 2               ; remove: the four relational operators above are its only expansions
%unmacro SL_FLOAT_ZERO_JUMP 2                  ; remove: .pf_zerop_f above is its only expansion
%unmacro SL_FALSY_JUMP 2                       ; remove: .p_bang above is its last expansion
%unmacro NUM_RANGE_CHECK 2                     ; remove: the int-arithmetic range guards above are its only expansions
%unmacro INFIX_BITWISE_OP 1                    ; remove: the infix bitwise operators above are its only expansions
%unmacro INFIX_COMPARE 2                       ; remove: the infix relational operators above are its only expansions
%unmacro INFIX_BINARY_PROMOTED 0               ; remove: the infix arithmetic operators above are its only expansions
%unmacro INFIX_ADD_PROMOTED 0                  ; remove: infix + is its only expansion
%unmacro INFIX_EVAL_RHS 0                      ; remove: the infix chains above are its only expansions
%unmacro ENV_UPDATE_WALK 1                     ; remove: the assignment operators above are its only expansions
%unmacro LOCAL_SET_SCAN 3                      ; remove: Make/LocalSet above are its only expansions
%unmacro SL_EVAL_BLOCK 2                       ; remove: Run/IfElse above are its only expansions
%unmacro TAG_IS_JUMP_NE 3                      ; remove: .p_first / .p_butfirst above are its last expansions
%unmacro TAG_GUARD_ATOM 2                      ; remove: the atom guards above are its only expansions
