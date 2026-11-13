; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define SAF_PEEK_OFF 1                         ; char offset past decimal point
%define SFM_SLOT_B 0                           ; x87 bridge slot: divisor
%define SFM_SLOT_A 8                           ; x87 bridge slot: dividend
%define SF_WM_PLUS1 1                          ; watermark = digits + 1, lea
%define SAF_DIG_X5 4                           ; decimal-accumulator x5 fold scale, lea reg,[reg+reg*SAF_DIG_X5] = reg*5; the paired lea [rdx+reg*SAF_DIG_X10] doubles it to x10
%define SAF_DIG_X10 2                          ; SIB scale of the acc*10 second lea
%define SFM_C2 4                               ; x87 status word C2, "FPREM reduction incomplete" = status bit 10,
MakeFloat:
     vmovd eax, xmm0                           ; move binary32 bits into GPR
     shr eax, SL_FLOAT_SHIFT                   ; drop 2 low mantissa bits -> 30-bit payload
     or eax, SL_FLOAT_TAG                      ; set bits31:30 both-one FLOAT tag
     ret                                       ; return tagged FLOAT in eax
FloatVal:
     shl eax, SL_FLOAT_SHIFT                   ; realign payload to binary32 positions, low 2 bits zero; the shift itself discards the 2 tag bits, no pre-mask needed
     vmovd xmm0, eax                           ; xmm0 = float value
     ret                                       ; return float in xmm0
sl_to_float:
     cmp eax, SL_FLOAT_TAG                     ; bits 31:30 both set = FLOAT?, SL_TAG_MASK == SL_FLOAT_TAG, so, v & mask == mask <=> v >=u mask, the same identity Car uses; no mask, no temporary
     jae FloatVal                              ; yes: unpack, identical body, tail-branch to the shared routine
     call NumVal                               ; eax = raw int32, FATALs on non-numbers
     vxorps xmm0, xmm0, xmm0                   ; break the false dependency on the vcvtsi2ss destination, xmm0 is also its first source; VEX vcvtsi2ss ZEROES dest[127:32], so this is not a correctness requirement
     vcvtsi2ss xmm0, xmm0, eax                 ; xmm0 =, float int32
     ret                                       ; return float in xmm0
sl_promote_pair:
     mov r9d, eax                              ; r9d = rhs tagged
     mov r8d, edx                              ; r8d = lhs tagged
     cmp r8d, SL_FLOAT_TAG                     ; lhs >= FLOAT tag => bits 31:30 both set?, no mask, no ecx temp, same identity as sl_to_float
     jae .pp_float                             ; yes: float path
     cmp r9d, SL_FLOAT_TAG                     ; rhs likewise
     jae .pp_float                             ; yes: float path
; --- integer path: unwrap both via NumVal ---
     mov eax, r8d                              ; eax = lhs tagged
     call NumVal                               ; eax = lhs int32
     mov r10d, eax                             ; r10d = lhs int, volatile save
     mov eax, r9d                              ; eax = rhs tagged
     call NumVal                               ; eax = rhs int32
     mov ecx, eax                              ; ecx = rhs int
     mov eax, r10d                             ; eax = lhs int
     xor edx, edx                              ; edx = kind 0, integer operands
     ret                                       ; return kind + raw ints
.pp_float:
     mov eax, r9d                              ; eax = rhs tagged
     call sl_to_float                          ; xmm0 = rhs float
     vmovaps xmm1, xmm0                        ; xmm1 = rhs float, contract slot
     mov eax, r8d                              ; eax = lhs tagged
     call sl_to_float                          ; xmm0 = lhs float, already in the contract slot
     mov edx, 1                                ; kind 1, float operands in edx for callers
     ret                                       ; return kind + floats
sl_atom_float:
     lea rdi, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rdi = atom text in heap
     xor r8d, r8d                              ; r8d = integer-part accumulator
     xor r9d, r9d                              ; r9d = fraction accumulator
     xor r10d, r10d                            ; r10d = int digit count
     xor r11d, r11d                            ; r11d = frac digit count
     xor ecx, ecx                              ; ecx = negative flag
     cmp byte [rdi], '-'                       ; leading minus?
     jne .saf_int_loop                         ; no: parse digits
     mov ecx, 1                                ; mark negative form
     inc rdi                                   ; skip past '-'
.saf_int_loop:
     movzx edx, byte [rdi]                     ; load next char
     sub dl, '0'                               ; ASCII -> digit value
     cmp dl, 9                                 ; valid digit?
     ja .saf_int_done                          ; no: end of integer part
     cmp r10d, SF_ATOM_DIGIT_MAX               ; over the int-digit cap?
     jae .saf_not_float                        ; yes: the cap is int32 accumulator safety, 999999999 < INT32_MAX, so r8d cannot wrap, it is NOT an exactness guarantee, for the same reason as the fraction cap below
     lea eax, [r8 + r8*SAF_DIG_X5]             ; int acc *= 5, 1-cycle lea; eax free, last read of the incoming atom offset was :148
     lea r8d, [rdx + rax*SAF_DIG_X10]          ; ... *2 then + digit => 10*acc + digit, one 3-component lea replaces the two 1-cycle leas plus the add
     inc r10d                                  ; count this int digit
     inc rdi                                   ; advance text pointer
     jmp .saf_int_loop                         ; keep parsing
.saf_int_done:
     test r10d, r10d                           ; zero integer digits?
     jz .saf_not_float                         ; yes: "" or "-" alone: not a float
     cmp byte [rdi], '.'                       ; decimal point?
     jne .saf_check_end                        ; no: integer-only text
; '.' must be followed by at least one digit
     cmp byte [rdi + SAF_PEEK_OFF], '0'        ; below '0'?
     jb .saf_not_float                         ; yes: "1." is not a float
     cmp byte [rdi + SAF_PEEK_OFF], '9'        ; above '9'?
     ja .saf_not_float                         ; yes: reject
     inc rdi                                   ; skip past '.'
.saf_frac_loop:
     movzx edx, byte [rdi]                     ; load next char
     sub dl, '0'                               ; ASCII -> digit value
     cmp dl, 9                                 ; valid digit?
     ja .saf_check_end                         ; no: end of fraction
     cmp r11d, SF_ATOM_DIGIT_MAX               ; over the frac-digit cap?
     jae .saf_not_float                        ; yes: the cap is int32 accumulator safety, 999999999 < INT32_MAX, so r9d cannot wrap below .
     lea eax, [r9 + r9*SAF_DIG_X5]             ; frac acc *= 5, 1-cycle lea; eax free, the incoming atom offset was consumed by the lea at :148
     lea r9d, [rdx + rax*SAF_DIG_X10]          ; ... *2 then + digit => 10*acc + digit, one 3-component lea replaces the two 1-cycle leas plus the add
     inc r11d                                  ; count this frac digit
     inc rdi                                   ; advance text pointer
     jmp .saf_frac_loop                        ; keep parsing
.saf_check_end:
     cmp byte [rdi], 0                         ; clean end of text?
     jne .saf_not_float                        ; no: trailing garbage rejects
     vcvtsi2sd xmm0, xmm0, r8d                 ; xmm0 =, double int part
     test r11d, r11d                           ; any fraction digits?
     jz .saf_narrow                            ; no: integer value only
     lea rax, [rel pow10_table]                ; rax = &power-of-10 table, shared with atof
     mov eax, [rax + r11*SL_POW10_DWORD]       ; eax = 10^frac_digits, r11d <= 9, single load vs up to 9 imul
     vcvtsi2sd xmm1, xmm1, r9d                 ; xmm1 =, double frac_digits_int
     vcvtsi2sd xmm2, xmm2, eax                 ; xmm2 =, double 10^count
     vdivsd xmm1, xmm1, xmm2                   ; xmm1 = fraction value, exact int32 sources, one rounding
     vaddsd xmm0, xmm0, xmm1                   ; xmm0 = int + frac, binary64 accumulation
.saf_narrow:
     vcvtsd2ss xmm0, xmm0                      ; THE single correctly-rounded narrowing, covers both predecessors: the jz above and the vaddsd fall-through
     test ecx, ecx                             ; negative form?
     jz .saf_trunc                             ; no: keep positive
     vxorps xmm0, xmm0, [rel flt_sign_mask]    ; flip sign bit
.saf_trunc:
; truncate to the 30-bit tag precision, matches literal packing
     vmovd eax, xmm0                           ; bits to GPR
     and eax, SL_F32_LOW_CLEAR                 ; clear 2 low mantissa bits
     vmovd xmm0, eax                           ; back to xmm
     xor edx, edx                              ; edx = 0, xor sets ZF=1: parse success
     ret                                       ; return float + success flag
.saf_not_float:
     xor edx, edx                              ; edx = 0, xor WRITES ZF, a `mov edx,1` here is a 5-byte no-op for flags and leaves the ZF=1 of the rejecting test/jz above, which every caller reads with a bare `jnz`
     inc edx                                   ; edx = 1, ZF=0: parse failure, inc writes ZF; this pair is the only single-flag-setting way to return edx=1 AND ZF=0
     ret                                       ; return failure flag only
sl_eq_floatize:
     test eax, eax                             ; bit31 set, CONS or FLOAT ?
     js .sef_tag31                             ; yes: distinguish below
     test eax, SL_NUMBER_TAG                   ; bit30 set, NUMBER ?
     jz .sef_atom                              ; neither: plain atom, NIL filtered inside
     call NumVal                               ; eax = raw int32, leaf; FATAL parity-proofed
     vxorps xmm0, xmm0, xmm0                   ; break the false dependency on the vcvtsi2ss destination, VEX vcvtsi2ss zeroes dest[127:32], so this is not a correctness requirement
     vcvtsi2ss xmm0, xmm0, eax                 ; xmm0 =, float int32
     jmp .sef_yes                              ; report success
.sef_tag31:
     test eax, SL_NUMBER_TAG                   ; bit30 also set = FLOAT?
     jz .sef_no                                ; bit31 only: CONS, not comparable
     shl eax, SL_FLOAT_SHIFT                   ; realign payload to binary32, the shift discards the 2 tag bits, no pre-mask needed
     vmovd xmm0, eax                           ; xmm0 = float value
     jmp .sef_yes                              ; report success
.sef_atom:
     test eax, eax                             ; NIL, tag 0 + offset 0 ?
     jnz sl_atom_float                         ; tail-call: ZF + xmm0 from text parse, pure leaf; NIL falls through
.sef_no:
     xor edx, edx                              ; edx = 0, xor WRITES ZF, see .saf_not_float: a flagless `mov edx,1` leaves the ZF=1 of the test/jz that branched here, and the caller's bare `jnz` then reads failure as success
     inc edx                                   ; edx = 1, ZF=0: not comparable, inc writes ZF
     ret                                       ; return failure flag
.sef_yes:
     xor edx, edx                              ; edx = 0, xor sets ZF=1: success
     ret                                       ; return float + success flag
sl_flog2:
     vmovd eax, xmm0                           ; eax = float bits; the valid log domain is exactly the unsigned range [1, SL_F32_MAX_FINITE], smallest subnormal .. largest finite
     dec eax                                   ; fold the lower bound into the upper one so ONE unsigned compare tests both ends; the cmp below sets its own flags
     cmp eax, SL_F32_MAX_FINITE - 1            ; accept iff eax-1 <=u 0x7F7FFFFE, i.e. exactly [1, SL_F32_MAX_FINITE]. Rejected below that: +0.0, bits 0 wraps to 0xFFFFFFFF and compares above plus all negatives, incl. -0.0, +Inf and every NaN; the accepted low end includes the subnormal bits range, which is unreachable in practice, subnormal-tagged FLOATs cannot be produced under the engine's SSE FTZ/DAZ regime, and a hypothetical arrival would simply get x87's true log2 of the stored bits, x87 does not honor MXCSR FTZ/DAZ
     ja .sfl_fatal                             ; yes: log2,0 / undefined
     sub rsp, 16                               ; x87 bridge slot, 16 bytes; only 4-byte vmovss/fld/fstp touch it, so there is no alignment requirement, and the incoming parity varies, see sl_fpow's note
     vmovss dword [rsp], xmm0                  ; spill SSE value to memory
     fld1                                      ; st0 = 1.0, the y in y*log2,x
     fld dword [rsp]                           ; st0 = x, st1 = 1.0
     fyl2x                                     ; st0 = 1.0 * log2,x, pop -> single value
     fstp dword [rsp]                          ; store result, pop -> stack empty
     vmovss xmm0, dword [rsp]                  ; xmm0 = log2,x
     add rsp, 16                               ; release bridge slot
     ret                                       ; return log2 in xmm0
.sfl_fatal:
     FATAL "SectorLogo: FLOG2 requires a positive finite argument" ; domain violation
sl_fln:
     vmovd eax, xmm0                           ; eax = float bits; the valid log domain is exactly the unsigned range [1, SL_F32_MAX_FINITE], smallest subnormal .. largest finite
     dec eax                                   ; fold the lower bound into the upper one so ONE unsigned compare tests both ends; the cmp below sets its own flags
     cmp eax, SL_F32_MAX_FINITE - 1            ; accept iff eax-1 <=u 0x7F7FFFFE, i.e. exactly [1, SL_F32_MAX_FINITE]. Rejected below that: +0.0, bits 0 wraps to 0xFFFFFFFF and compares above plus all negatives, incl. -0.0, +Inf and every NaN; the accepted low end includes the subnormal bits range, which is unreachable in practice, subnormal-tagged FLOATs cannot be produced under the engine's SSE FTZ/DAZ regime, and a hypothetical arrival would simply get x87's true log2 of the stored bits, x87 does not honor MXCSR FTZ/DAZ
     ja .sfln_fatal                            ; yes: ln,0 undefined
     sub rsp, 16                               ; x87 bridge slot, 16 bytes; only 4-byte vmovss/fld/fstp touch it, so there is no alignment requirement, and the incoming parity varies, see sl_fpow's note
     vmovss dword [rsp], xmm0                  ; spill SSE value to memory
     fldln2                                    ; st0 = ln,2, the y in y*log2,x
     fld dword [rsp]                           ; st0 = x, st1 = ln,2
     fyl2x                                     ; st0 = ln,2 * log2,x = ln,x, pop -> single value
     fstp dword [rsp]                          ; store result, pop -> stack empty
     vmovss xmm0, dword [rsp]                  ; xmm0 = ln,x
     add rsp, 16                               ; release bridge slot
     ret                                       ; return ln in xmm0
.sfln_fatal:
     FATAL "SectorLogo: FLN requires a positive finite argument" ; domain violation
sl_fexp2:
     vmovd eax, xmm0                           ; eax = x bit pattern
     not eax                                   ; invert so a cleared required bit shows as set
     test eax, SL_F32_EXP_MASK                 ; all-ones exponent, +-inf or NaN ?
     je .fx2_special                           ; yes: x - round,x = inf-inf = NaN, short-circuit instead
     sub rsp, 16                               ; x87 bridge slots, 16 bytes; only 4-byte vmovss/fld/fstp touch them, so there is no alignment requirement, and the incoming parity varies, see sl_fpow's note
     vmovss dword [rsp], xmm0                  ; spill SSE value to memory
     fld dword [rsp]                           ; st0 = x
     fld st0                                   ; st0 = x, st1 = x
     frndint                                   ; st0 = n = round,x, st1 = x
     fsub st1, st0                             ; st1 = frac = x - n, st0 = n
     fxch st1                                  ; st0 = frac, st1 = n
     f2xm1                                     ; st0 = 2^frac - 1, st1 = n
     fld1                                      ; st0 = 1, st1 = 2^frac-1, st2 = n
     faddp st1, st0                            ; st0 = 2^frac, st1 = n
     fscale                                    ; st0 = 2^frac * 2^n = 2^x, st1 = n
     fstp dword [rsp]                          ; store result, pop -> st0 = n
     fstp st0                                  ; pop leftover n -> stack empty
     vmovss xmm0, dword [rsp]                  ; xmm0 = 2^x
     add rsp, 16                               ; release bridge slots
     ret                                       ; return 2^x in xmm0
.fx2_special:
     vmovd eax, xmm0                           ; reload full bits, sign + mantissa
     test eax, SL_F32_MAN_MASK                 ; mantissa nonzero, NaN ?
     jz .fx2_inf                               ; no: true infinity
     ret                                       ; NaN: propagate unchanged, xmm0 already holds it
.fx2_inf:
     test eax, eax                             ; sign bit set, -inf ?
     jns .fx2_ret                              ; no, +inf : 2^inf = +inf, xmm0 already holds it
     vxorps xmm0, xmm0, xmm0                   ; 2^-inf = +0.0
.fx2_ret:
     ret                                       ; return special-case result
sl_fmod:
     vmovd eax, xmm1                           ; eax = divisor bits
     and eax, SL_F32_NOT_SIGN_MASK             ; |b| bits, btr NOT usable: ZF = old bit, misses +0.0
     jz .sfm_zero_div                          ; zero divisor: contract violation
; ---- special-value classification before any x87 load ----
     vmovd r8d, xmm0                           ; r8d = dividend bits
     not r8d                                   ; invert so a cleared required bit shows as set
     test r8d, SL_F32_EXP_MASK                 ; all ones, inf or NaN ?
     je .sfm_nan                               ; yes: fmod,inf|NaN, b = NaN
     cmp eax, SL_F32_EXP_MASK                  ; |b|, sign already cleared above vs the +Inf pattern: NaN magnitudes sort strictly above it, finite below
     ja .sfm_nan                               ; above => exponent all ones + nonzero mantissa => NaN divisor
     je .sfm_inf_divisor                       ; equal => +Inf divisor, mantissa is provably zero
     sub rsp, 16                               ; x87 bridge slots, 16 bytes; only 4-byte vmovss/fld/fstp touch them, so there is no alignment requirement, and the incoming parity varies, see sl_fpow's note
     vmovss dword [rsp + SFM_SLOT_B], xmm1     ; spill divisor
     vmovss dword [rsp + SFM_SLOT_A], xmm0     ; spill dividend
     fld dword [rsp + SFM_SLOT_B]              ; st0 = b
     fld dword [rsp + SFM_SLOT_A]              ; st0 = a, st1 = b
.sfm_loop:
     fprem                                     ; st0 = partial remainder; C2 set = reduction incomplete, keep going
     fstsw ax                                  ; ax = x87 status word, condition codes C0-C3 in AH, exception flags in AL
     test ah, SFM_C2                           ; C2 set?, C2 = status bit 10 = AH bit 2. Testing AL bit 2 would read ZE,
; which FPREM never sets, the loop would run exactly once and return an
; unscaled partial remainder for any quotient needing more than one reduction
     jnz .sfm_loop                             ; yes: reduction incomplete, keep going
     fstp dword [rsp + SFM_SLOT_A]             ; store final remainder, pop -> st0 = b
     fstp st0                                  ; pop b -> stack empty
     vmovss xmm0, dword [rsp + SFM_SLOT_A]     ; xmm0 = fmod,a, b
     add rsp, 16                               ; release bridge slots
     ret                                       ; return remainder in xmm0
.sfm_inf_divisor:
     ret                                       ; +Inf divisor, finite dividend -> remainder = dividend, xmm0 unchanged
.sfm_nan:
     mov eax, SL_F32_QNAN_BITS                 ; canonical quiet NaN 0x7FC00000
     vmovd xmm0, eax                           ; xmm0 = NaN
     ret                                       ; before any x87 load, so the x87 stack stays balanced
.sfm_zero_div:
     FATAL "SectorLogo: FMOD division by zero" ; zero divisor: abort
sl_fpow:
; zero-base fast path: 0^0 = 1, 0^e = 0 for any e != 0
     vxorps xmm2, xmm2, xmm2                   ; xmm2 = 0.0
     vucomiss xmm0, xmm2                       ; base == 0?
     jne .fpow_nonzero                         ; no: normal path
     jp .fpow_nonzero                          ; unordered, NaN base : treat as nonzero
; base is 0: exponent decides
     vucomiss xmm1, xmm2                       ; exponent vs 0.0?
     jp .fpow_nan                              ; unordered, NaN exponent : IEEE 754-2019 9.2.1 / C99 F.10.4.4, pow,x,NaN
     je .fpow_one                              ; yes: 0^+-0 = 1, shared site with inf^0; C99 F.10.4.4
     jb .fpow_zero_e_chk                       ; yes: e < 0, -inf -> +inf, below, every other negative -> +0
; e > 0, finite and non-zero, jp at :432, je at :436 and jb at :437 consumed every other case.IEEE 754-2019 9.2.1 gives pow,-0, y>0 = -0 ONLY for an ODD INTEGER y: an even integer, a non-integer exponent, and every |e| >= 2^24, all of which are even integers, so a sign flip can never be needed all answer +0.
     vmovd eax, xmm1                           ; eax = exponent bits, e > 0, so the sign bit is already clear and no mask is needed
     cmp eax, FPOW_BIG_E_BITS                  ; |e| >= 2^24?
     jae .fpow_zero_e                          ; yes: an EVEN integer -> +0
     vcvttss2si r9d, xmm1                      ; r9d = trunc,e, |e| < 2^24, so vcvttss2si cannot saturate
     vroundss xmm3, xmm1, xmm1, 3              ; float trunc(exponent) directly
     vucomiss xmm3, xmm1                       ; exponent integral?
     jne .fpow_zero_e                          ; no: +0, always ordered, e is finite and positive here, so no jp is needed
     test r9b, 1                               ; odd exponent?
     jz .fpow_zero_e                           ; even: +0
     ret                                       ; odd integral e > 0: the base keeps its own sign
.fpow_zero_e_chk:
     vmovd eax, xmm1                           ; exponent bits, sign INCLUDED
     cmp eax, FPOW_INF_BITS | SL_F32_SIGN_MASK ; exactly -inf?
     jne .fpow_zero_e_finite                   ; no: finite negative e, classify below
     and eax, FPOW_INF_BITS                    ; +inf pattern, sign cleared
     vmovd xmm0, eax                           ; 0^-inf = +inf, unchanged behavior
     ret                                       ; return to caller
.fpow_zero_e_finite:                           ; C99 F.10.4.4: pow,+-0, y<0 = +-HUGE_VAL
     vmovd r9d, xmm1                           ; exponent bits
     btr r9d, 31                               ; strip only the IEEE-754 sign bit
     cmp r9d, FPOW_BIG_E_BITS                  ; |e| >= 2^24?
     jae .fpow_zero_e_pinf                     ; yes: an EVEN integer -> +inf
     vcvttss2si r9d, xmm1                      ; trunc,e, |e| < 2^24 -> no saturation
     vroundss xmm3, xmm1, xmm1, 3              ; float trunc(exponent) directly
     vucomiss xmm3, xmm1                       ; integral?, finite non-zero -> ordered, no jp needed
     jne .fpow_zero_e_pinf                     ; non-integral -> +inf
     test r9b, 1                               ; odd integer?, parity of the two's-complement trunc == parity of |e|
     jz .fpow_zero_e_pinf                      ; even -> +inf
     vmovd eax, xmm0                           ; base bits
     test eax, eax                             ; base is -0.0?
     jns .fpow_zero_e_pinf                     ; +0 base -> +inf
     mov eax, FPOW_INF_BITS | SL_F32_SIGN_MASK ; -0 base, odd y<0 -> -inf
     vmovd xmm0, eax                           ; return -inf
     ret                                       ; return to caller
.fpow_zero_e_pinf:
     mov eax, FPOW_INF_BITS                    ; +HUGE_VAL
     vmovd xmm0, eax                           ; return +inf
     ret                                       ; return to caller
.fpow_zero_e:
     vxorps xmm0, xmm0, xmm0                   ; return 0.0
     ret                                       ; return to caller
.fpow_nonzero:
     vmovd r11d, xmm0                          ; r11d = full base bit pattern, kept for the special-base test and the infinity sign
     mov eax, r11d                             ; eax = base bits
     not eax                                   ; invert so a cleared required bit shows as set
     test eax, SL_F32_EXP_MASK                 ; all-ones exponent, +-Inf or NaN ?
     je .fpow_special_base                     ; yes: answer from the IEEE table, sl_flog2 would FATAL
     test r11d, r11d                           ; sign bit set, negative base ? r11d is read again at :560 and :571, so test it in place
     jns .fpow_pos                             ; no: positive base, existing log2/exp2 path
; --- negative base: only integral exponents have a real result ---
     vmovd eax, xmm1                           ; exponent bits for magnitude classification
     btr eax, 31                               ; strip only the IEEE-754 sign bit
     cmp eax, FPOW_BIG_E_BITS                  ; finite magnitude >= 2^24?
     jae .fpow_big_e                           ; yes: every binary32 there is an EVEN integer, no int32 round-trip, it saturates
     vcvttss2si r9d, xmm1                      ; r9d = trunc,e
     vroundss xmm2, xmm1, xmm1, 3              ; float trunc(exponent) directly
     vucomiss xmm2, xmm1                       ; exponent integral?
     jp .fpow_neg_fatal                        ; unordered, NaN exponent : no real result -> FATAL, ZF=1 on unordered, so jne alone cannot catch it
     jne .fpow_neg_fatal                       ; not integral: no real result -> FATAL
     jmp .fpow_e_ready                         ; small integral exponent: magnitude path with r9d = trunc,e
.fpow_big_e:
     cmp eax, FPOW_INF_BITS                    ; |e| bits vs the +Inf pattern, sign already stripped at :469
     ja .fpow_nan                              ; above it: NaN exponent, pow,x,NaN = NaN for any finite nonzero x
     jne .fpow_big_e_mag                       ; magnitude path with the REAL float e, r9d never receives the, saturating truncation; equal falls to the inf-exponent case
.fpow_neg_base_inf_e:
     vmovd eax, xmm1                           ; exponent bits, sign INCLUDED
     test eax, eax                             ; exponent is -Inf?
     js .fpow_nb_neg_inf                       ; yes: mirrored answers
     vandps xmm2, xmm0, [rel flt_abs_mask]     ; xmm2 = |b|
     vucomiss xmm2, [rel flt_one]              ; |b| vs 1.0
     jb .fpow_zero_e                           ; |b| < 1: pow,x, +Inf = +0
     je .fpow_one_e                            ; |b| == 1 and b < 0, so b == -1: pow,-1, +Inf = 1
     mov eax, FPOW_INF_BITS                    ; |b| > 1: pow,x, +Inf = +Inf
     vmovd xmm0, eax                           ; return +inf
     ret                                       ; not-equal base magnitude decides; no x87 load, so the x87 stack stays balanced
.fpow_nb_neg_inf:                              ; e = -Inf: mirror row, C99 F.10.4.4
     vandps xmm2, xmm0, [rel flt_abs_mask]     ; xmm2 = |b|
     vucomiss xmm2, [rel flt_one]              ; |b| vs 1.0
     jb .fpow_nb_neg_inf_big                   ; |b| < 1: pow,x, -Inf = +Inf
     je .fpow_one_e                            ; b == -1: pow,-1, -Inf = 1
     jmp .fpow_zero_e                          ; |b| > 1: pow,x, -Inf = +0
.fpow_nb_neg_inf_big:
     mov eax, FPOW_INF_BITS                    ; +HUGE_VAL
     vmovd xmm0, eax                           ; return +inf
     ret                                       ; return to caller
.fpow_one_e:
     vmovss xmm0, dword [rel flt_one]          ; pow,-1, +-Inf = 1.0
     ret                                       ; return to caller
.fpow_e_ready:
     push r9                                   ; preserve integer parity and stack alignment
     vandps xmm0, xmm0, [rel flt_abs_mask]     ; xmm0 = |b|
     call sl_flog2                             ; xmm0 = log2,|b|
     vmulss xmm0, xmm0, xmm1                   ; exponent already in xmm1, preserved by sl_flog2, equal to trunc(e) here
     call sl_fexp2                             ; xmm0 = |b|^e
     pop r9                                    ; restore trunc,e
     test r9b, 1                               ; odd exponent?, ZF=1 iff bit0 clear = even exponent
     jz .fpow_ret                              ; even: result correct as-is
     vxorps xmm0, xmm0, [rel flt_sign_mask]    ; odd: flip result sign
.fpow_ret:
     ret                                       ; return |b|^e, sign-adjusted
.fpow_big_e_mag:
     vandps xmm0, xmm0, [rel flt_abs_mask]     ; xmm0 = |b|, huge-|e| negative base: parity is even, no flip needed
     call sl_flog2                             ; xmm0 = log2,|b| ; xmm1 = e survives, leaf that touches only eax/xmm0/x87
     vmulss xmm0, xmm0, xmm1                   ; xmm0 = e * log2|b|, the REAL exponent, an r9d=0 placeholder here would answer 1.0 for every base
     jmp sl_fexp2                              ; tail: 2^,e*log2|b| = |b|^e; |b|>1 overflows to +inf, |b|<1 underflows to +0.0, both correct for an even exponent
.fpow_pos:
     vandps xmm2, xmm0, [rel flt_abs_mask]     ; xmm2 = |b|, positive arm, so == b
     vucomiss xmm2, dword [rel flt_one]        ; b == 1.0?
     je .fpow_one                              ; yes: 1^y = 1 for any y
     call sl_flog2                             ; xmm0 = log2,base
     vmulss xmm0, xmm0, xmm1                   ; xmm0 = e * log2,base, xmm1 survives sl_flog2, a leaf that touches only xmm0/eax/x87
     jmp sl_fexp2                              ; tail: xmm0 = 2^,e*log2 base = base^e
.fpow_special_base:
     vxorps xmm2, xmm2, xmm2                   ; xmm2 = 0.0
     vucomiss xmm1, xmm2                       ; exponent == 0?
     jp .fpow_nan                              ; unordered, NaN exponent : no
     je .fpow_one                              ; yes: x^0 = 1.0, shared site with inf^0
     test r11d, SL_F32_MAN_MASK                ; mantissa nonzero => NaN, both infinities have a zero mantissa
     jnz .fpow_ret                             ; yes: NaN^e = NaN for e != 0, IEEE propagates the payload
     vucomiss xmm1, xmm2                       ; e < 0?
     jb .fpow_zero_e                           ; yes: inf^-n = 0.0
     test r11d, r11d                           ; base is -Inf?
     jns .fpow_ret                             ; no, +Inf : +Inf^e = +Inf for e > 0, xmm0 already holds it
; -Inf base, e > 0: an even/odd integral exponent is the only remaining defined case
     vmovd eax, xmm1                           ; exponent bits
     btr eax, 31                               ; strip only the IEEE-754 sign bit
     cmp eax, FPOW_BIG_E_BITS                  ; finite magnitude >= 2^24?
     jae .fpow_inf_even                        ; yes: every such binary32 is an EVEN integer
     vcvttss2si r9d, xmm1                      ; r9d = trunc,e
     vroundss xmm3, xmm1, xmm1, 3              ; float trunc(exponent) directly
     vucomiss xmm3, xmm1                       ; exponent integral?
     jp .fpow_neg_fatal                        ; unordered: no real result -> FATAL, mirrors the finite negative path
     jne .fpow_neg_fatal                       ; not integral: no real result -> FATAL, e.g., -Inf ^0.5
     test r9b, 1                               ; odd exponent?
     jnz .fpow_inf_odd                         ; odd: -Inf; even falls through to the +Inf case
.fpow_inf_even:
; Only -Inf reaches here, the +Inf early-out above already returned, so the
; sign flip is unconditional:, -Inf ^even = +Inf.
     vxorps xmm0, xmm0, [rel flt_sign_mask]    ; -Inf -> +Inf
     ret                                       ; even integral exponent: the result is +Inf
.fpow_inf_odd:
     ret                                       ; odd integral exponent:, -Inf ^odd = -Inf, which xmm0 already holds, only -Inf reaches here
.fpow_one:
     vmovss xmm0, dword [rel flt_one]          ; the "answer is 1.0" leaf, shared by 0^+-0, x^+-0, any x, inf^0, pow,+-1, NaN, pow,-1, +-Inf and pow,1, y, any y
     ret                                       ; return 1.0
.fpow_nan:
     vandps xmm2, xmm0, [rel flt_abs_mask]     ; xmm2 = |base|
     vucomiss xmm2, dword [rel flt_one]        ; |base| == 1?, one absolute test covers both +1 and -1
     jp .fpow_nan_emit                         ; unordered: base is NaN -> NaN^NaN = NaN
     je .fpow_one                              ; yes: pow,+-1, NaN = 1, shared site
.fpow_nan_emit:
     mov eax, SL_F32_QNAN_BITS                 ; canonical quiet NaN, the same constant .sfm_nan uses
     vmovd xmm0, eax                           ; xmm0 = NaN
     ret                                       ; no x87 load on this path, so the x87 stack stays balanced
.fpow_neg_fatal:
     FATAL "SectorLogo: POWER with negative base needs an integral exponent" ; no real result, e.g., -2 ^0.5

%macro FMT_FRAC_DIGITS 3
%%loop:
     vmulss xmm1, xmm1, [rel flt_ten]          ; frac *= 10
     vcvttss2si eax, xmm1                      ; eax = next digit, 0..9 (ASCII below); frac < 1 keeps the product < 10, so no eax==10 clamp is needed: the residual is fl,fl,frac*10 -trunc, always in [0,1, worst entry 1-2^-24 yields 9.9999990463 -> 9
     vroundss xmm2, xmm1, xmm1, 3              ; digit as exact binary32 from the fractional value
     vsubss xmm1, xmm1, xmm2                   ; frac -= digit
     add al, '0'                               ; digit -> ASCII
     mov byte [rdi], al                        ; store digit
     inc rdi                                   ; advance dest
     cmp al, '0'                               ; nonzero digit?
     je %%next                                 ; no: trailing-zero candidate
     lea r10d, [r9 + SF_WM_PLUS1]              ; watermark = digits so far + 1
%%next:
     inc r9d                                   ; next digit position
     cmp r9d, %1                               ; all digits produced?
     jb %%loop                                 ; no: continue
     test r10d, r10d                           ; any nonzero fraction digit?
     jnz %2                                    ; yes: caller's cut-at-watermark path
     jmp %3                                    ; no: caller's zero fallback
%endmacro
sl_float_fmt:
     push rbx                                  ; save non-volatile: the exponent accumulator, and lg_num_to_str's ebx live in rbx below.
     shl eax, SL_FLOAT_SHIFT                   ; realign payload to binary32 bit positions, shift discards the tag bits, no pre-mask needed
     vmovd xmm0, eax                           ; xmm0 = float value
     mov r11d, eax                             ; r11d = raw binary32 bits, eax copy, skips the SIMD round-trip
; --- specials: exponent field all ones means inf or nan ---
     not eax                                   ; invert so a cleared required bit shows as set
     test eax, SL_F32_EXP_MASK                 ; exponent saturated?
     jne .sff_signed                           ; no: finite number
     test r11d, SL_F32_MAN_MASK                ; mantissa nonzero?
     jz .sff_inf                               ; no: infinity; NaN falls through
.sff_nan:
     mov dword [rdi], 'nan'                    ; write "nan", high NUL byte is harmless
     add rdi, 3                                ; advance past 3 chars
     jmp .sff_done                             ; finished
.sff_inf:
     test r11d, r11d                           ; negative infinity?
     jns .sff_inf_body                         ; no: plain "inf"
     mov byte [rdi], '-'                       ; leading minus sign
     inc rdi                                   ; advance dest
.sff_inf_body:
     mov dword [rdi], 'inf'                    ; write "inf"
     add rdi, 3                                ; advance past 3 chars
     jmp .sff_done                             ; finished
.sff_signed:
; --- sign: emit '-' and work on |x| ---
     test r11d, r11d                           ; negative?
     jns .sff_zero                             ; no: keep value
     mov byte [rdi], '-'                       ; leading minus sign
     inc rdi                                   ; advance dest
     vandps xmm0, xmm0, [rel flt_abs_mask]     ; xmm0 = |x|, clear sign bit
.sff_zero:
; --- exact zero prints "0.0" ---
     test r11d, SL_F32_NOT_SIGN_MASK           ; any value bit set?, signless zero test, the magnitude mask; using SL_F32_SIGN_MASK here would invert it
     jnz .sff_range                            ; nonzero: continue
     mov dword [rdi], '0.0'                    ; write "0.0", + NUL byte
     add rdi, 3                                ; advance past 3 chars
     jmp .sff_done                             ; finished
.sff_range:
; --- notation dispatch ---
     vucomiss xmm0, dword [rel flt_1e9_f]      ; |x| >= 1e9?
     jae .sff_exp                              ; yes: exponent notation
     vucomiss xmm0, dword [rel flt_1e_3_f]     ; |x| < 1e-3?
     jb .sff_exp                               ; yes: exponent notation
; ==================== FIXED NOTATION ====================
     vcvttss2si eax, xmm0                      ; eax = integer part, < 1e9
     vroundss xmm1, xmm0, xmm0, 3              ; integer part via truncation
     vsubss xmm1, xmm0, xmm1                   ; xmm1 = fraction in [0,1
     call lg_num_to_str                        ; rdi = past integer digits
     mov byte [rdi], '.'                       ; decimal point
     inc rdi                                   ; advance dest
     xor r9d, r9d                              ; produced-digit counter
     xor r10d, r10d                            ; significant, nonzero digit watermark
     FMT_FRAC_DIGITS SF_FRAC_DIGITS, .sff_frac_cut, .sff_frac_zero ; fixed-notation fraction digits, 6
.sff_frac_zero:
     mov r10d, 1                               ; no: keep a single "0" digit
.sff_frac_cut:
     mov eax, SF_FRAC_DIGITS                   ; total produced
     sub eax, r10d                             ; trailing zeros to remove
     sub rdi, rax                              ; rewind dest past stripped zeros
     jmp .sff_done                             ; finished
; ==================== EXPONENT NOTATION ====================
.sff_exp:
     xor ebx, ebx                              ; ebx = decimal exponent accumulator
; down loop: while, x >= 10 { x /= 10; exp++ }, bottom-tested do-while
; with a one-time entry pre-check, every |x| < 10 float skips the body
     vucomiss xmm0, [rel flt_ten]              ; x >= 10?
     jb .sff_exp_up                            ; no: check the low side
.sff_exp_down:
     vdivss xmm0, xmm0, [rel flt_ten]          ; x /= 10
     inc ebx                                   ; exponent++
     vucomiss xmm0, [rel flt_ten]              ; still >= 10?, bottom test, no back-edge jmp
     jae .sff_exp_down                         ; yes: keep scaling down
.sff_exp_up:
; up loop: while, x < 1 { x *= 10; FTZ-flush test; exp-- }, bottom-tested
; do-while with entry pre-check, NaN never reaches here: .sff_nan handled it
     vucomiss xmm0, [rel flt_one]              ; x < 1?
     jae .sff_digits                           ; no: x in [1,10 : emit digits
.sff_exp_up_loop:
     vmulss xmm0, xmm0, [rel flt_ten]          ; x *= 10
     vmovd eax, xmm0                           ; raw bits for FTZ-flush test, eax dead here: next read is a write below; eax is in the documented overwrite list
     test eax, SL_F32_NOT_SIGN_MASK            ; flushed to 0.0 by FTZ?, raw-bit test catches both zero signs
     jz .sff_flushed_zero                      ; yes: below the thread's arithmetic floor, emit "0.0"
     dec ebx                                   ; exponent--
     vucomiss xmm0, [rel flt_one]              ; still < 1?, bottom test
     jb .sff_exp_up_loop                       ; yes: keep scaling up
     jmp .sff_digits                           ; no: x reached [1,10, emit digits. MANDATORY: the loop's success exit falls out of
; the body, so without this jump it lands in the FTZ-flush block and every
; successfully scaled value would print as "0.0"
.sff_flushed_zero:
     mov dword [rdi], '0.0'                    ; write "0.0", + NUL byte, matches the thread's FTZ/DAZ arithmetic semantics
     add rdi, 3                                ; advance past 3 chars
     jmp .sff_done                             ; finished
.sff_digits:
     vcvttss2si r8d, xmm0                      ; r8d = leading digit, 1..9 (ASCII below)
     vroundss xmm1, xmm0, xmm0, 3              ; leading decimal digit in FP32
     vsubss xmm1, xmm0, xmm1                   ; frac = x - digit in [0,1
     lea eax, [r8 + '0']                       ; leading digit -> ASCII
     mov byte [rdi], al                        ; store leading digit
     inc rdi                                   ; advance dest
     mov byte [rdi], '.'                       ; decimal point
     inc rdi                                   ; advance dest
     xor r9d, r9d                              ; produced-digit counter
     xor r10d, r10d                            ; significant digit watermark
     FMT_FRAC_DIGITS SF_EDIG_DIGITS, .sff_edig_cut, .sff_edig_zero ; exponent-notation fraction digits, 5
.sff_edig_zero:
     mov eax, SF_EDIG_DIGITS + 1               ; no: drop digits AND the '.'
     sub rdi, rax                              ; rewind dest, 5 digits + point
     jmp .sff_esign                            ; emit exponent
.sff_edig_cut:
     mov eax, SF_EDIG_DIGITS                   ; total produced
     sub eax, r10d                             ; trailing zeros to remove
     sub rdi, rax                              ; rewind dest past stripped zeros
.sff_esign:
     mov byte [rdi], 'e'                       ; exponent marker
     inc rdi                                   ; advance dest
     test ebx, ebx                             ; negative exponent?
     jns .sff_epos                             ; no: plain digits
     mov byte [rdi], '-'                       ; exponent minus sign
     inc rdi                                   ; advance dest
     neg ebx                                   ; exponent magnitude
.sff_epos:
     mov eax, ebx                              ; eax = exponent for the converter
     call lg_num_to_str                        ; decimal exponent digits, fall-through: .sff_done is next
.sff_done:
     pop rbx                                   ; restore non-volatile
     ret                                       ; return rdi past last byte
%undef SAF_PEEK_OFF
%undef SAF_DIG_X5
%undef SFM_SLOT_B
%undef SFM_SLOT_A
%undef SF_WM_PLUS1
%undef SFM_C2
%unmacro FMT_FRAC_DIGITS 3                     ; defined and used only in this file; leaving it bound would let a later include redefine its meaning with no diagnostic
