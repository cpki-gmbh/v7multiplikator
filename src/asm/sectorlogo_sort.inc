; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; --- file-local frame/slot constants, comment checker: named offsets only ---
%define SS_LIST 32                             ; sl_sort: staged list, rsp slot
%define SS_FRAME 48                            ; sl_sort: frame bytes, home area + list slot + padding
%define SV_LHS_OFF 32                          ; sl_samevalue: parked lhs, above the 32-byte callee home area
%define SV_FRAME 40                            ; sl_samevalue / sl_asword: frame bytes, home area + parity
%define RS_VAL_OFF 0                           ; record dword 0: tagged value
%define RS_KEY_OFF 8                           ; record qword at +8: monotone binary64 sort key
%define RS_QW_SCALE 8                          ; address scale for the DOUBLED element index, 2*idx * 8 = 16-byte record stride
%define RS_INDEX_LIMIT (SL_SORT_MAX_ELEMS * 2) ; the walk counts DOUBLED indices, record stride is 16 = 2*8, and x86 scales max at 8 -> the capacity guard is the doubled limit
%define RS_REC_STRIDE 16                       ; bytes per sort record, 2 x RS_QW_SCALE for the insertion shift
%define RS_INSERT_MAX 64                       ; N at or below this uses insertion, beats 8 radix passes, above uses radix
%define RS_HIST_SCALE 4                        ; bytes per histogram dword
%define RS_HIST_BUCKETS 256                    ; histogram entries per pass
%define RS_NUM_PASSES 8                        ; 8 key bytes x 8 bits = 64-bit key
%define HF_FRAME 32                            ; sl_hash_float: frame bytes, home area + padding
%define HF_WINDOW 16                           ; sl_hash_float: byte window scanned per NUL probe
%define HF_QWORD 8                             ; sl_hash_float: CRC32C fold unit and its in-window offset
%define LS_LIST_OFF 32                         ; sl_lset: remaining list cursor, GC root
%define LS_IDX_OFF 40                          ; sl_lset: remaining index, raw int32
%define LS_VAL_OFF 48                          ; sl_lset: replacement value, tagged
%define LS_HEAD_OFF 56                         ; sl_lset: result head, GC root
%define LS_TAIL_OFF 64                         ; sl_lset: result tail cell, GC root
%define LS_FOUND_OFF 72                        ; sl_lset: replacement-done flag, raw int32
%define LS_FRAME 80                            ; sl_lset: frame bytes, home area + 5 slots + padding
%define SC_CELL_OFF 32                         ; sl_setcar/sl_setcdr: target cell, tagged, GC root
%define SC_FIELD_OFF 36                        ; sl_setcar/sl_setcdr: field offset within the cell, 0 = car, CONS_CDR_OFF = cdr, packed with SC_CELL_OFF in the same qword above the home area
%define SC_FRAME 40                            ; sl_setcar/sl_setcdr: frame bytes
%define LF_LIST_OFF 32                         ; sl_lfind: remaining list cursor, GC root
%define LF_KEY_OFF 40                          ; sl_lfind: search key, tagged NUMBER/FLOAT
%define LF_POS_OFF 48                          ; sl_lfind: scan position counter, raw int32
%define LF_START_OFF 56                        ; sl_lfind: startpos, raw int32
%define LF_MODE_OFF 64                         ; sl_lfind: comparison mode, eq/ge/le
%define LF_LAST_OFF 72                         ; sl_lfind: LE mode last-match position, -1 = none
%define LF_FRAME 80                            ; sl_lfind: frame bytes, home area + 6 slots + padding
%define LF_MODE_EQ 0                           ; LFIND: exact match
%define LF_MODE_GE 1                           ; LFINDGE: first element >= key
%define LF_MODE_LE 2                           ; LFINDLE: last element <= key
%define LG_FRAME 8                             ; sl_length alignment pad bytes, the 32-byte Win64 home area is added as a separate +32 at the sub
sl_reverse:
     push rbx                                  ; save non-volatile, accumulator lives here
     push rsi                                  ; save non-volatile, cursor lives here
     sub rsp, 40                               ; home area + padding, keeps EvalNextArg 0 mod 16: 2 pushes + 40
     call EvalNextArg                          ; eax = evaluated list argument
     test eax, eax                             ; NIL, empty list ?
     jz .rv_done_nil                           ; yes: return NIL
     lea ecx, [rax - SL_CONS_TAG]              ; CONS-band probe
     cmp ecx, SL_FLOAT_TAG - SL_CONS_TAG       ; above the band -> not a cons
     jae .rv_not_list                          ; FLOAT/atom/NUMBER: not a list -> FATAL
     xor ebx, ebx                              ; accumulator = NIL, register-held, GC-visible via GC spill
     mov esi, eax                              ; cursor = list head, esi write zero-extends rsi
.rv_loop:
     lea r8d, [rsi - SL_CONS_TAG]              ; CONS-band probe
     cmp r8d, SL_FLOAT_TAG - SL_CONS_TAG       ; above the band -> not a cons
     jae .rv_tail                              ; not a genuine CONS: cdr = NIL, elem = NIL
     btr esi, SL_CONS_BIT                      ; clear tag bit 31; 32-bit result zero-extends into RSI
     mov r9d, [r13 + rsi*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF] ; r9d = cdr field, raw
     mov edx, [r13 + rsi*CONS_SCALE + CONS_BASE_OFF] ; edx = car field, raw; cell offset 0
.rv_have_elem:
     mov esi, r9d                              ; cursor = rest, NIL for terminal append
     mov eax, edx                              ; eax = element, Cons car arg
     mov edx, ebx                              ; edx = accumulator, Cons cdr arg
     call Cons                                 ; may GC: rax/rdx pushed by .cons_gc; rbx/rsi spilled by GC_Collect
     mov ebx, eax                              ; accumulator = new head
     jmp .rv_loop                              ; re-classify at the head: a NIL cursor fails the CONS compare and falls to .rv_tail -> .rv_done
.rv_tail:
; non-CONS tail: end of list, matches sl_sort's .s_walk classification and sl_length's walk classification, both classify on a copy with one CONS compare and stop there, do NOT append a NIL element
.rv_done:
     mov eax, ebx                              ; return accumulator
.rv_epilogue:
     add rsp, 40                               ; drop frame + parity pad
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     ret                                       ; return to caller
.rv_done_nil:
     xor eax, eax                              ; NIL input: return NIL
     jmp .rv_epilogue                          ; shared stack teardown
.rv_not_list:
     FATAL_VALUE "REVERSE requires a list", eax ; atom/NUMBER/FLOAT argument, named
sl_sort_coerce_num:
     test eax, SL_NUMBER_TAG                   ; NUMBER, bit30 or FLOAT, bits31+30 ?
     jnz .scn_done                             ; yes: return as-is, both hot cases, one test
     test eax, eax                             ; NIL?
     jz .scn_fatal                             ; yes: FATAL
     test eax, SL_CONS_TAG                     ; CONS, bit31 only, bit30 clear ?
     jnz .scn_fatal                            ; yes: FATAL
; ATOM: parse string at [r13+rax] as integer or float
     push rbx                                  ; save non-volatile
     push rsi                                  ; save non-volatile, ATOM parse path uses rsi as text cursor
     sub rsp, 40                               ; shadow + alignment, 2 pushes + 40 -> calls at 0 mod 16
     lea rsi, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rsi = string bytes in heap
     mov r9d, eax                              ; park the tagged atom, offender for the FATAL paths below
     cmp byte [rsi], 0                         ; empty string?
     je .scn_not_num                           ; yes: not a number
     xor eax, eax                              ; accumulator = 0
     xor r8d, r8d                              ; negative flag = 0
     xor ebx, ebx                              ; digit counter = 0
     cmp byte [rsi], '-'                       ; leading minus?
     je .scn_minus                             ; yes: parse negative
     cmp byte [rsi], '+'                       ; leading plus?
     jne .scn_digits                           ; no: start digit loop
     inc rsi                                   ; skip past '+'
     jmp .scn_digits                           ; start digit loop
.scn_minus:
     inc rsi                                   ; skip past '-'
     mov r8d, 1                                ; set negative flag
.scn_digits:
     movzx edx, byte [rsi]                     ; load next char
     sub dl, '0'                               ; ASCII -> digit
     cmp dl, 9                                 ; valid digit?
     ja .scn_digit_done                        ; no: end of integer part
     cmp eax, STN_NUM_MUL_LIMIT                ; acc*10+9 would exceed the NUMBER payload limit 2^29-1, STN_NUM_MUL_LIMIT = STN_NUM_PAYLOAD_MAX/10 ?
     jae .scn_boundary                         ; at/over the boundary: the digit ALREADY parsed in edx decides
     imul eax, eax, 10                         ; accumulator *= 10, eax <= STN_NUM_MUL_LIMIT - 1 here
     add eax, edx                              ; accumulator += digit, <= STN_NUM_MUL_LIMIT*10 - 1 < 2^29-1: MakeNum/NumVal round-trip exactly, never wraps
     inc rsi                                   ; advance string pointer
     inc ebx                                   ; count digits
     jmp .scn_digits                           ; continue parsing
.scn_boundary:
     cmp eax, STN_NUM_MUL_LIMIT                ; exactly at the boundary?
     jne .scn_int_overflow                     ; above: no single digit can fit -> float
     cmp edx, 1                                ; d <= 1 -> magnitude 536870910 or 536870911: always valid
     jbe .scn_boundary_acc                     ; yes: magnitude stays inside the payload
     cmp edx, 2                                ; d == 2 -> magnitude 536870912 = 2^29: valid ONLY as -2^29
     jne .scn_int_overflow                     ; d >= 3: overflow -> float
     test r8d, r8d                             ; negative flag set?
     jz .scn_int_overflow                      ; positive +2^29 exceeds 2^29-1 -> float
.scn_boundary_acc:
     imul eax, eax, 10                         ; 536870910, fits the payload; no wrap
     add eax, edx                              ; + d -> 536870910 / 536870911 / 536870912
     inc rsi                                   ; advance string pointer
     inc ebx                                   ; count digits
     jmp .scn_digits                           ; continue parsing, acc is now back inside the payload
.scn_int_overflow:
     mov r11, 922337203685477579               ; r11 =, 2^63-1-9 /10: last exact accumulator, imm64 cannot encode in cmp r64,imm32, nasm truncates
.scn_ofl64:
     cmp rax, r11                              ; rax*10+9 would exceed 2^63-1?
     ja .scn_ofl_float                         ; yes: degrade to per-digit float chain
     movzx edx, byte [rsi]                     ; load current digit
     sub dl, '0'                               ; ASCII -> digit value
     cmp dl, 9                                 ; valid digit?, garbage terminator -> exit
     ja .scn_ofl64_done                        ; no: ONE conversion, then terminator checks
     imul rax, rax, 10                         ; rax *= 10, imm form, legacy encoding, no APX
     add rax, rdx                              ; rax += digit, still exact; rdx = [0,9], upper bits zero
     inc rsi                                   ; advance string pointer
     inc ebx                                   ; count digits
     jmp .scn_ofl64                            ; keep accumulating exactly
.scn_ofl64_done:
     cmp byte [rsi], '.'                       ; fraction needs the FP64 integer part
     je .scn_ofl64_fraction                    ; yes: keep binary64, the fraction parser starts from it
     cmp byte [rsi], 0                         ; accept only clean integer text
     jne .scn_not_num                          ; trailing garbage -> reject, same as the binary64 path
     vcvtsi2ss xmm0, xmm0, rax                 ; int64 -> FP32 directly, single rounding, no double rounding
     jmp .scn_float_sign                       ; sign + MakeFloat tail, unchanged
.scn_ofl64_fraction:
     vcvtsi2sd xmm0, xmm0, rax                 ; fraction path: exact binary64 integer part, unchanged
     jmp .scn_ofl_frac_64                      ; expects rsi on '.', advances it itself
.scn_ofl_float:
     vcvtsi2sd xmm0, xmm0, rax                 ; xmm0 = EXACT binary64 of the 63-bit int part, binary64 all the way down, so no per-digit binary32 rounding and no double rounding at the end
.scn_ofl_digits:
     movzx edx, byte [rsi]                     ; load next char
     sub dl, '0'                               ; ASCII -> digit
     cmp dl, 9                                 ; valid digit?
     ja .scn_ofl_done                          ; no: end of integer part
     vcvtsi2sd xmm1, xmm1, edx                 ; xmm1 = binary64 digit
     vfmadd132sd xmm0, xmm1, qword [rel dbl_ten] ; xmm0 = xmm0*dbl_ten + xmm1, rounded once
     inc rsi                                   ; advance string pointer
     inc ebx                                   ; count digits
     jmp .scn_ofl_digits                       ; continue float accumulation
.scn_ofl_done:
     cmp byte [rsi], '.'                       ; decimal point?
     je .scn_ofl_frac_64                       ; yes: parse fraction, xmm0 is exact binary64
     cmp byte [rsi], 0                         ; end of string?
     jne .scn_not_num                          ; no: trailing garbage
     vcvtsd2ss xmm0, xmm0, xmm0                ; pure integer tail: single correct rounding binary64 -> binary32, MakeFloat packs the low 32 bits
     jmp .scn_float_sign                       ; apply sign and return FLOAT
.scn_ofl_frac_64:
     inc rsi                                   ; skip past '.'
     movzx edx, byte [rsi]                     ; peek first fraction digit
     sub dl, '0'                               ; ASCII -> digit
     cmp dl, 9                                 ; valid digit?
     jbe .scn_frac_scan                        ; yes: shared backward binary64 fraction parser, xmm0 already exact binary64, no widening instruction, no double rounding
.scn_ofl_nodigit_narrow:
     vcvtsd2ss xmm0, xmm0, xmm0                ; binary64 -> binary32, one correct rounding ; .scn_ofl_nodigit hands xmm0 to MakeFloat, which takes the low 32 bits
.scn_ofl_nodigit:
; cold: first char after '.' is not a digit, only a clean NUL keeps
; "N." valid, trailing-dot float ; any other text is trailing garbage.
     cmp byte [rsi], 0                         ; end of string?
     jne .scn_not_num                          ; no: garbage after the dot -> reject
     jmp .scn_float_sign                       ; yes: "N." -> float of the int part
.scn_digit_done:
     cmp byte [rsi], '.'                       ; decimal point, even with 0 integer digits ?
     je .scn_check_frac                        ; yes: allow ".5" style decimals
     test ebx, ebx                             ; any digits at all?
     jz .scn_not_num                           ; no: not a number
     cmp byte [rsi], 0                         ; string ended?
     je .scn_int_finish                        ; yes: pure integer
     jmp .scn_not_num                          ; trailing garbage, '.' was already excluded by the branch above
.scn_check_frac:
     movzx edx, byte [rsi + SL_ATOI_PEEK_OFF]  ; peek char after '.'
     cmp dl, '0'                               ; below '0'?
     jb .scn_trailing_dot                      ; maybe "42." — check below
     cmp dl, '9'                               ; above '9'?
     ja .scn_trailing_dot                      ; no digit after '.' -> maybe "42."
     inc rsi                                   ; skip past '.'
     vcvtsi2sd xmm0, xmm0, eax                 ; xmm0 = integer part, exact binary64, 0 for ".5"
.scn_frac_scan:
     xor r10d, r10d                            ; frac digit accumulator, forward scheme, r10/r11 dead here: the int ladders consumed them
     xor r11d, r11d                            ; frac digit count
.scn_frac_scan_loop:
     movzx edx, byte [rsi]                     ; load next char
     sub dl, '0'                               ; ASCII -> digit
     cmp dl, 9                                 ; valid digit?
     ja .scn_frac_scan_done                    ; no: end of fraction
     cmp r11d, SF_ATOM_DIGIT_MAX               ; over the exact-parse cap?, the literal parser's own contract, SF_ATOM_DIGIT_MAX = 9
     jae .scn_not_num                          ; yes: reject, same domain rule as literals
     imul r10d, r10d, 10                       ; frac acc *= 10 ...
     add r10d, edx                             ; ... += digit
     inc r11d                                  ; count++
     inc rsi                                   ; advance
     jmp .scn_frac_scan_loop                   ; keep parsing
.scn_frac_scan_done:
     cmp byte [rsi], 0                         ; clean end of text?
     jne .scn_not_num                          ; no: trailing garbage
     vcvtsi2sd xmm1, xmm1, r10d                ; binary64 frac_int, exact
     lea rax, [rel pow10_table]                ; RIP+index is illegal, base into a register
     vcvtsi2sd xmm2, xmm2, dword [rax + r11*SL_POW10_DWORD] ; binary64 10^count, direct memory source
     vdivsd xmm1, xmm1, xmm2                   ; frac = frac_int / 10^count, ONE rounding
     vaddsd xmm0, xmm0, xmm1                   ; value = int + frac, binary64
     vcvtsd2ss xmm0, xmm0, xmm0                ; the single final narrowing
     jmp .scn_float_sign                       ; apply sign and pack as FLOAT
.scn_trailing_dot:
; "." with no fraction digit after it, e.g. "42." or just "."
     test ebx, ebx                             ; any integer digits before the dot?
     jz .scn_not_num                           ; no: bare "." is not a number
     cmp byte [rsi + SL_ATOI_PEEK_OFF], 0      ; NUL after '.'?
     jne .scn_not_num                          ; no: trailing garbage after '.'
     vcvtsi2ss xmm0, xmm0, eax                 ; xmm0 = integer part as float
; fall through to .scn_float_sign: "42." -> FLOAT,42.0
.scn_float_sign:
     test r8d, r8d                             ; negative?
     jz .scn_float_ret                         ; no: keep positive
     vxorps xmm0, xmm0, [rel flt_sign_mask]    ; flip sign bit
.scn_float_ret:
     call MakeFloat                            ; wrap as FLOAT
     add rsp, 40                               ; drop shadow + alignment
     pop rsi                                   ; restore non-volatile, ATOM parse path used it
     pop rbx                                   ; restore non-volatile
     ret                                       ; return FLOAT
.scn_int_finish:
     test r8d, r8d                             ; negative?
     jz .scn_int_ret                           ; no: keep as-is
     neg eax                                   ; negate integer result
.scn_int_ret:
     call MakeNum                              ; wrap as NUMBER
     add rsp, 40                               ; drop shadow + alignment
     pop rsi                                   ; restore non-volatile, ATOM parse path used it
     pop rbx                                   ; restore non-volatile
     ret                                       ; return NUMBER
.scn_not_num:
     add rsp, 40                               ; drop shadow + alignment, pushes below are undone first
     pop rsi                                   ; restore non-volatile, ATOM parse path used it
     pop rbx                                   ; restore non-volatile
     FATAL_VALUE "a numeric value was required", r9d ; names the atom text that failed to parse, operation-neutral: shared by Sort/LFIND/NearestMatch
.scn_fatal:
     test eax, eax                             ; NIL, tagged 0 ?
     jz .scn_fatal_nil                         ; yes: report NIL
     FATAL "SectorLogo: a numeric value was required (got a list)" ; bits31:30 == 01 is the only remaining case, a list reached a scalar site
.scn_fatal_nil:
     FATAL "SectorLogo: a numeric value was required (got NIL)" ; NIL input, upstream produced nothing
.scn_done:
     ret                                       ; NUMBER/FLOAT: return as-is
sl_sort:
     push rbx                                  ; save non-volatile, aligns RSP to 0
     push rsi                                  ; save non-volatile, scatter + walk-buffer base
     push rdi                                  ; save non-volatile, radix scatter bound holder
     sub rsp, SS_FRAME                         ; 32-byte Win64 home area + the 4-byte SS_LIST slot at offset 32 + unused tail, 3 pushes 24 + 48 = 72 = 8 mod 16
     call EvalNextArg                          ; eax = list, advances r12d, keep it!
; ---------- phase 1: argument + type check ----------
     test eax, eax                             ; NIL, empty list ?
     jz .s_done                                ; yes: return NIL
     lea ecx, [rax - SL_CONS_TAG]              ; CONS-band probe
     cmp ecx, SL_FLOAT_TAG - SL_CONS_TAG       ; above the band -> not a cons
     jae .s_not_list                           ; FLOAT/atom/NUMBER: not a list -> FATAL
     mov dword [rsp + SS_LIST], eax            ; GC-root the original list
     mov edi, eax                              ; walk cursor = list head, register-held
     xor ebx, ebx                              ; ebx = doubled element index, non-volatile across calls
     lea rsi, [rel sl_sort_qw_a]               ; rsi = record buffer base held for the whole walk, survives Car/Cdr/coerce/sl_to_float
.s_walk:
; cursor classification: a NIL cursor is NOT a CONS cell either, so this one
; test covers both "proper list ended" and "improper-list tail", NUMBER/FLOAT/ATOM
     lea ecx, [rdi - SL_CONS_TAG]              ; CONS-band probe
     cmp ecx, SL_FLOAT_TAG - SL_CONS_TAG       ; above the band -> not a cons
     jae .s_walk_done                          ; no, NIL / NUMBER / FLOAT / ATOM : end of list
     cmp ebx, RS_INDEX_LIMIT                   ; doubled index >= 2 * capacity?, buffer full BEFORE the store?
     jae .s_overflow                           ; yes: FATAL before touching past-the-end memory
     mov r8d, edi                              ; r8d = cursor copy, the loop-top CONS test, .s_walk already proved it
     btr r8d, SL_CONS_BIT                      ; r8d = raw cons cell index, btr r32 zero-extends into r8
     mov eax, [r13 + r8*CONS_SCALE + CONS_BASE_OFF] ; eax = element, inlined Car: btr + one SIB load, no call, no dead band guard
     test eax, eax                             ; NIL element?
     jz .s_bad_elem                            ; yes: FATAL
; --- classify element: CONS sub-list or bare value? ---
     lea ecx, [rax - SL_CONS_TAG]              ; CONS-band probe
     cmp ecx, SL_FLOAT_TAG - SL_CONS_TAG       ; above the band -> not a cons
     jb .s_cons_elem                           ; yes: key = Car,element
; bare value: coerce ATOMs to NUMBER/FLOAT, then use as both key and value
     call sl_sort_coerce_num                   ; eax = NUMBER or FLOAT, ATOMs parsed
     mov dword [rsi + rbx*RS_QW_SCALE + RS_VAL_OFF], eax ; value straight into the record, no slot: nothing in the walk can allocate, see the phase-2 note, and the record buffer is GC-inert
     jmp .s_key_to_float                       ; convert key to float
.s_cons_elem:
; sub-list: key = Car,element, value = the sub-list
     mov dword [rsi + rbx*RS_QW_SCALE + RS_VAL_OFF], eax ; sub-list value straight into the record, no slot, see above
     mov r8d, eax                              ; r8d = element copy for the tag strip
     btr r8d, SL_CONS_BIT                      ; r8d = raw cons cell index of the SUB-LIST
     mov eax, [r13 + r8*CONS_SCALE + CONS_BASE_OFF] ; eax = first sub-element, the key, inlined Car, CONS proven at 477
     test eax, eax                             ; key NIL?
     jz .s_bad_key                             ; yes: FATAL
     call sl_sort_coerce_num                   ; eax = NUMBER or FLOAT key
.s_key_to_float:
     cmp eax, SL_FLOAT_TAG                     ; FLOAT key?
     jae .s_key_float                          ; yes: widen the binary32 payload
     call NumVal                               ; eax = int32 payload, NUMBER; exact
     cdqe                                      ; sign-extend into 64 bits
     vcvtsi2sd xmm0, xmm0, rax                 ; xmm0 = exact binary64 value
     jmp .s_key_monotone                       ; to the qword transform
.s_key_float:
     shl eax, SL_FLOAT_SHIFT                   ; the shift discards tag bits 30 and 31
     test eax, SL_F32_NOT_SIGN_MASK            ; sign bit stripped: exponent+mantissa, eax must survive for vmovd
     jnz .s_key_widen                          ; nonzero: real value, Inf, or NaN, keep bits, the monotone map orders NaNs by RAW BITS: +NaN keys above +Inf, -NaN below -Inf, pinned by the validator's "Sort negative NaN key below -inf" test
     xor eax, eax                              ; +-0.0: normalize to +0.0 bits
.s_key_widen:
     vmovd xmm0, eax                           ; xmm0 = binary32 bits
     vcvtss2sd xmm0, xmm0, xmm0                ; exact widening to binary64
.s_key_monotone:
     vmovq rcx, xmm0                           ; rcx = binary64 key bits
     mov rax, rcx                              ; copy for the sign mask
     sar rax, 63                               ; positive: 0, negative: -1
     bts rax, 63                               ; positive: topbit, negative: all-ones
     xor rcx, rax                              ; monotone unsigned 64-bit ordering key
     mov qword [rsi + rbx*RS_QW_SCALE + RS_KEY_OFF], rcx ; binary64 monotone key
     add ebx, 2                                ; next doubled index, capacity checked at loop top
; advance to next element
     mov r8d, edi                              ; r8d = cursor copy for the tag strip, r8 was overwritten by the coerce/NumVal calls above
     btr r8d, SL_CONS_BIT                      ; r8d = raw cons cell index
     mov edi, [r13 + r8*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF] ; edi = rest of list, inlined Cdr, CONS proven at the loop top; the 32-bit load zero-extends the tagged cdr into rdi
     jmp .s_walk                               ; continue walking
.s_walk_done:
     shr ebx, 1                                ; doubled index -> element count N
     mov [rel sl_sort_n], ebx                  ; store element count N
     call sl_sort_records                      ; rbx = sorted record buffer, tagged values ride along untouched
     xor edi, edi                              ; accumulator = NIL, register-held; GC-safe: GC_Collect spills rdi into the scanned range
     mov esi, [rel sl_sort_n]                  ; index = N, esi write zero-extends rsi for 64-bit addressing
.s_rebuild_loop:
     dec esi                                   ; index--, element N-1 down to 0
     lea rdx, [rsi + rsi]                      ; rdx = 2 * element index, 16-byte stride: 2*idx * 8
     mov eax, dword [rbx + rdx*RS_QW_SCALE + RS_VAL_OFF] ; tagged value from record
     mov edx, edi                              ; cdr = accumulator, Cons arg 2
     call Cons                                 ; may GC: rax/rdx pushed by .cons_gc; rdi/rsi spilled by GC_Collect
     mov edi, eax                              ; accumulator = new head
     test esi, esi                             ; element 0 done?, re-test: call Cons overwrites flags
     jnz .s_rebuild_loop                       ; no: keep rebuilding
.s_rebuild_done:
     mov eax, edi                              ; return the sorted list head
.s_done:
     add rsp, SS_FRAME                         ; drop the GC-root slot + home area + unused tail
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     ret                                       ; return to caller
.s_not_list:
     FATAL "SectorLogo: SORT requires a list of numbers or sub-lists" ; atom/NUMBER argument
.s_bad_elem:
     FATAL "SectorLogo: SORT elements must be numbers or sub-lists" ; bare atom element
.s_bad_key:
     FATAL "SectorLogo: SORT sub-list first element must be numeric" ; non-numeric key
.s_overflow:
     FATAL "SectorLogo: SORT element limit exceeded" ; over SL_SORT_MAX_ELEMS
; sl_sort_records -- pure record sorter shared by SORT and FULL NearestMatch.
; Sorts the sl_sort_n 16-byte records in sl_sort_qw_a in place by their
; monotone binary64 keys (record +8); the tagged values at +0 ride along.
; IN: sl_sort_n >= 1, records populated. OUT: rbx = sorted record buffer
; (sl_sort_qw_a or sl_concat_buf). Clobbers volatile registers and rbx;
; preserves rsi/rdi. No Lisp cells touched, no allocation, no GC.
sl_sort_records:
     push rsi                                  ; save non-volatile, insertion buffer base
     push rdi                                  ; save non-volatile, radix scatter bound holder
     sub rsp, 40                               ; home area + parity, no calls inside the sorter
     cmp dword [rel sl_sort_n], 1              ; 1 element?
     je .sr_single                             ; yes: identity, buffer already sorted
     cmp dword [rel sl_sort_n], RS_INSERT_MAX  ; small N? insertion beats 8 radix passes, the prefix term alone is ~12k instructions; a shift iteration is 8, so worst-case insertion is ~4N^2 vs radix's 12664 + 104N, crossover at N ~= 71, and RS_INSERT_MAX sits just under it
     jbe .sr_insertion                         ; yes: stable insertion on the records
; ---------- 8 x LSD radix passes, NO calls, volatiles free ----------
.sr_radix_init:
     lea rbx, [rel sl_sort_qw_a]               ; source records
     lea r11, [rel sl_concat_buf]              ; destination records (unchanged spelling from the original line)
     mov r9d, 3                                ; bytes 0..2 are 0 for nonnegative keys and FF for negative keys; sign is resolved in higher passes
.sr_pass:
; --- zero the 1024-byte histogram with one bulk string operation ---
     lea r10, [rel sl_concat_buf + SL_SORT_MAX_ELEMS * SL_RECORD_QWORDS * RS_QW_SCALE] ; histogram base, disjoint from radix records
     mov rdi, r10                              ; rdi is dead until .sr_scatter, where edi is reloaded
     xor eax, eax                              ; fill byte 0
     mov ecx, RS_HIST_BUCKETS * RS_HIST_SCALE  ; 256 buckets * 4 bytes = 1024 bytes
     rep stosb                                 ; clear histogram, DF=0 per Win64
     lea rcx, [rbx + r9]                       ; rcx = record base + key byte index, byte-position load
; --- count keys into buckets, N >= 2 guaranteed; descending byte offset, histogram is commutative ---
     mov edx, [rel sl_sort_n]                  ; 1 <= N <= 65536 (the N == 1 case took .sr_single)
     dec edx                                   ; last record index N-1
     shl edx, 4                                ; byte offset (N-1)*16, fits in 32 bits
.sr_hist_count:
     movzx eax, byte [rcx + rdx + RS_KEY_OFF]  ; key byte at pass position = bucket, byte-position load, same address the indexed form computed
     add dword [r10 + rax*RS_HIST_SCALE], RS_REC_STRIDE ; hist[bucket] += 16, byte counts, not element counts, offsets feed the scatter directly
     sub rdx, RS_REC_STRIDE                    ; previous record, 16 bytes back
     jns .sr_hist_count                        ; processed all N keys
; An all-equal radix digit cannot alter ordering: omit prefix and O(N) stable scatter.
     movzx eax, byte [rbx + r9 + RS_KEY_OFF]   ; first record's digit
     mov eax, [r10 + rax*RS_HIST_SCALE]        ; histogram byte count for this digit
     mov edx, [rel sl_sort_n]                  ; total record count N
     shl edx, 4                                ; total bytes = N * record stride
     cmp eax, edx                              ; all N records in the same bucket?
     je .sr_pass_uniform                       ; yes, keep the current source buffer
.sr_prefix:
     xor edx, edx                              ; edx = running sum, start BYTE offsets
     xor ecx, ecx                              ; ecx = bucket index
.sr_prefix_loop:
%assign sl_pfx 0                               ; byte offset of this unrolled bucket within the group of four
%rep 4
     mov eax, dword [r10 + rcx*RS_HIST_SCALE + sl_pfx] ; eax = bucket count, bytes
     mov dword [r10 + rcx*RS_HIST_SCALE + sl_pfx], edx ; replace with start byte offset
     add edx, eax                              ; running sum += count, bytes
%assign sl_pfx sl_pfx+4                        ; next bucket in the unrolled group, +4 bytes
%endrep
     add ecx, 4                                ; four buckets per iteration
     cmp ecx, RS_HIST_BUCKETS                  ; all buckets processed?, 256 is 4*64, no remainder
     jb .sr_prefix_loop                        ; no: continue
     lea rcx, [rbx + r9]                       ; rcx = record base + key byte index, byte-position load
; --- stable scatter of whole 16-byte records src -> dst, N >= 2, do-while ---
     mov edi, [rel sl_sort_n]                  ; edi = element count N hoisted, rdi saved by prologue; no calls in radix
     shl edi, 4                                ; bound N*16 fits in 32 bits, zero-extends to rdi
     xor r8d, r8d                              ; r8 = src byte offset
.sr_scatter:
     vmovdqu xmm3, [rbx + r8]                  ; xmm3 = full 16-byte record, value | key
     movzx edx, byte [rcx + r8 + RS_KEY_OFF]   ; edx = bucket, byte load, off the record-load chain
     mov esi, dword [r10 + rdx*RS_HIST_SCALE]  ; esi = this bucket's write offset
     add dword [r10 + rdx*RS_HIST_SCALE], RS_REC_STRIDE ; next write goes one record later, stability
     vmovdqu [r11 + rsi], xmm3                 ; dst[offset] = record, 16-byte move
     add r8, RS_REC_STRIDE                     ; next src record, 16 bytes
     cmp r8, rdi                               ; past all elements?, byte bound
     jb .sr_scatter                            ; no: keep scattering
.sr_pass_end:
     xchg rbx, r11                             ; swap buffers only after an actual scatter
.sr_pass_uniform:
     inc r9d                                   ; next key byte position
     cmp r9d, RS_NUM_PASSES                    ; all 8 passes done?
     jb .sr_pass                               ; no: run the next pass
     jmp .sr_ret                               ; rbx already names the sorted buffer
.sr_insertion:
     lea rsi, [rel sl_sort_qw_a]               ; rsi = record buffer base
     mov ecx, 1                                ; prefix [0,1 is sorted; insert elements 1..N-1
     mov ebx, [rel sl_sort_n]                  ; ebx = N, the insertion loop bound
.sr_ins_outer:
     lea r9, [rsi + rcx*RS_QW_SCALE]           ; r9 = buf + idx*8, composed with idx*8 below = 16-byte stride
     vmovdqu xmm3, [r9 + rcx*RS_QW_SCALE]      ; record being inserted, xmm3 rides the whole inner loop
     mov rdx, [r9 + rcx*RS_QW_SCALE + RS_KEY_OFF] ; its monotone key, unsigned compare domain
     mov r8d, ecx                              ; gap index walks down from idx, always >= 1 here: ecx starts at 1 and only increments, so no zero test is needed on entry
.sr_ins_inner:
     lea r9, [rsi + r8*RS_QW_SCALE]            ; r9 = buf + gap*8, base for both predecessor and gap slots
     cmp qword [r9 + r8*RS_QW_SCALE - RS_REC_STRIDE + RS_KEY_OFF], rdx ; predecessor <= key?, unsigned: same order as the radix
     jbe .sr_ins_place                         ; yes: drop the record into the gap, equal does NOT shift -> stable
     vmovdqu xmm4, [r9 + r8*RS_QW_SCALE - RS_REC_STRIDE] ; predecessor record, whole 16 bytes
     vmovdqu [r9 + r8*RS_QW_SCALE], xmm4       ; shift it into the gap
     dec r8d                                   ; gap moves down; r8d == 0 falls through into .sr_ins_place below
     jnz .sr_ins_inner                         ; gap still above the front? keep shifting
.sr_ins_place:
     lea r9, [rsi + r8*RS_QW_SCALE]            ; r9 = buf + gap*8
     vmovdqu [r9 + r8*RS_QW_SCALE], xmm3       ; drop the inserted record into its gap
     inc ecx                                   ; next element
     cmp ecx, ebx                              ; all N placed?, ebx = N
     jb .sr_ins_outer                          ; no: insert the next one
     lea rbx, [rel sl_sort_qw_a]               ; rbx = ordered buffer, the caller reads rbx
.sr_ret:
     add rsp, 40                               ; drop home area + parity
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     ret                                       ; return, rbx = sorted record buffer
.sr_single:
     lea rbx, [rel sl_sort_qw_a]               ; identity: one record, already sorted
     jmp .sr_ret                               ; shared teardown
sl_hash_float:
     push rsi                                  ; save non-volatile, string pointer
     sub rsp, HF_FRAME                         ; home area + padding, keeps EvalNextArg 0 mod 16
     call EvalNextArg                          ; eax = evaluated argument
     test eax, eax                             ; NIL?
     jz .hf_not_string                         ; yes: FATAL
     test eax, SL_TAG_MASK                     ; CONS/NUMBER/FLOAT: any tag bit set ?
     jnz .hf_not_string                        ; yes: not a plain atom -> FATAL
     lea rsi, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rsi = string bytes, heap atom area
     xor eax, eax                              ; eax = CRC32C accumulator = 0
     vpxor xmm0, xmm0, xmm0                    ; zero vector for 16-byte NUL detector
.hf_find_len:
     vpcmpeqb xmm1, xmm0, [rsi]                ; detect terminator in the current 16-byte window
     vpmovmskb edx, xmm1                       ; mask of NUL positions
     test edx, edx                             ; any NUL in this window?
     jnz .hf_last                              ; yes: final partial window
     crc32 rax, qword [rsi]                    ; no NUL: fold the first eight bytes immediately
     crc32 rax, qword [rsi + HF_QWORD]         ; fold the following eight bytes
     add rsi, HF_WINDOW                        ; advance one full window
     jmp .hf_find_len                          ; scan the next window
.hf_last:
     tzcnt ecx, edx                            ; remaining bytes before NUL, 0..15
     cmp ecx, HF_QWORD                         ; at least eight bytes before the NUL?
     jb .hf_crc_tail_init                      ; no: byte tail only
     crc32 rax, qword [rsi]                    ; consume the first eight bytes when possible
     add rsi, HF_QWORD                         ; advance past the folded qword
     sub ecx, HF_QWORD                         ; eight fewer bytes remain
.hf_crc_tail_init:
     test ecx, ecx                             ; any tail bytes left?
     jz .hf_crc_done                           ; no: CRC complete
.hf_crc_tail:
     crc32 eax, byte [rsi]                     ; 0..7 remaining bytes
     inc rsi                                   ; advance one byte
     dec ecx                                   ; one fewer tail byte
     jnz .hf_crc_tail                          ; continue the byte tail
.hf_crc_done:
; convert uint32 CRC to float in [0,1 : use top 24 bits so binary32 cannot round to 1.0
     shr eax, 8                                ; 24-bit integer [0, 16777215]
     vcvtsi2ss xmm0, xmm0, eax                 ; xmm0 =, float 24-bit value, exact
     vmulss xmm0, xmm0, [rel flt_2pow_neg24]   ; xmm0 = value / 2^24 = hash in [0, 1
     call MakeFloat                            ; eax = tagged FLOAT
     add rsp, HF_FRAME                         ; drop frame
     pop rsi                                   ; restore non-volatile
     ret                                       ; return to caller
.hf_not_string:
     FATAL "SectorLogo: HASH-FLOAT requires an atom (string or word)" ; non-atom argument, only NIL/CONS/NUMBER/FLOAT are rejected above, a bare WORD atom is hashed too
sl_length:
     sub rsp, LG_FRAME + 32                    ; 32-byte Win64 home area + alignment pad, 8 + 32 = 40 = 8 mod 16 -> call sites 0 mod 16; body writes only eax/ecx/r8d, all volatile, no non-volatiles to save
     call EvalNextArg                          ; eax = list argument
     test eax, eax                             ; NIL, empty list ?
     jz .lg_nil                                ; yes: length = 0
     test eax, SL_CONS_TAG                     ; bit31 set, CONS or FLOAT ?
     jz .lg_not_list                           ; no: not a list -> FATAL
     test eax, SL_NUMBER_TAG                   ; bit30 also set = FLOAT?
     jnz .lg_not_list                          ; yes: not a list -> FATAL
     xor ecx, ecx                              ; ecx = count = 0
.lg_walk:
; single-compare classification on a copy: exactly bit31-set, CONS passes
; atom/NUMBER/FLOAT improper tails all fail one compare, house idiom
     lea r8d, [rax - SL_CONS_TAG]              ; CONS-band probe
     cmp r8d, SL_FLOAT_TAG - SL_CONS_TAG       ; above the band -> not a cons
     jae .lg_done                              ; not a genuine CONS cell
     btr eax, SL_CONS_BIT                      ; clear tag bit -> cons cell index, inlined Cdr leaf walk
     mov eax, [r13 + rax*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF] ; eax = cdr field, no call overhead
     inc ecx                                   ; count++
     jmp .lg_walk                              ; re-classify at the head: a NIL cursor fails the CONS compare exactly like an improper tail
.lg_done:
     mov eax, ecx                              ; eax = raw count
     call MakeNum                              ; eax = tagged NUMBER
     add rsp, LG_FRAME + 32                    ; drop frame, rsi/rdi never written, nothing to pop
     ret                                       ; return to caller
.lg_nil:
     xor ecx, ecx                              ; count = 0
     jmp .lg_done                              ; shared epilogue, MakeNum,0 = NUMBER 0
.lg_not_list:
     FATAL "SectorLogo: LENGTH requires a list" ; atom/NUMBER/FLOAT argument
sl_lset:
     push rbx                                  ; save non-volatile
     push rsi                                  ; save non-volatile
     push rdi                                  ; save non-volatile
     sub rsp, LS_FRAME                         ; 6 slots at offset 32..72 above the 32-byte Win64 home area, call sites stay aligned
; --- evaluate the 3 arguments into stack slots, EvalNextArg overwrites volatiles ---
     call EvalNextArg                          ; eax = list
     test eax, eax                             ; NIL, empty list ?
     jz .ls_not_list                           ; yes: FATAL, nothing to index
     test eax, SL_CONS_TAG                     ; bit31 set, CONS or FLOAT ?
     jz .ls_not_list                           ; no: FATAL
     test eax, SL_NUMBER_TAG                   ; bit30 also set = FLOAT?
     jnz .ls_not_list                          ; yes: FATAL
     mov dword [rsp + LS_LIST_OFF], eax        ; save original list cursor, GC root
     call EvalNextArg                          ; eax = index argument
     mov ecx, eax                              ; ecx = copy for tag isolation
     shr ecx, 30                               ; ecx = 2-bit tag code (NUMBER = 1, FLOAT = 3)
     cmp ecx, 1                                ; NUMBER?
     je .ls_idx_num                            ; yes: extract integer
     cmp ecx, 3                                ; FLOAT?
     jne .ls_bad_idx_type                      ; no: FATAL, ATOM/CONS/NIL
     call FloatVal                             ; xmm0 = float value
     vcvttss2si eax, xmm0                      ; truncate float to int32
     jmp .ls_idx_check                         ; check range
.ls_idx_num:
     call NumVal                               ; eax = raw int32 index
.ls_idx_check:
     test eax, eax                             ; negative index?
     js .ls_bad_idx                            ; yes: FATAL
     mov dword [rsp + LS_IDX_OFF], eax         ; save remaining index
     call EvalNextArg                          ; eax = replacement value, any tag
     mov dword [rsp + LS_VAL_OFF], eax         ; save replacement value
     xor eax, eax                              ; zero source for the three result slots
     mov dword [rsp + LS_FOUND_OFF], eax       ; replacement not yet done
     mov dword [rsp + LS_HEAD_OFF], eax        ; result head = NIL
     mov dword [rsp + LS_TAIL_OFF], eax        ; result tail = NIL
.ls_walk:
     mov eax, [rsp + LS_LIST_OFF]              ; eax = original cursor
     lea r8d, [rax - SL_CONS_TAG]              ; CONS-band probe
     cmp r8d, SL_FLOAT_TAG - SL_CONS_TAG       ; above the band -> not a cons
     jae .ls_walk_done                         ; no, NIL / NUMBER / FLOAT / ATOM : end of list, matches sl_sort's .s_walk classification, sl_length's .lg_walk classification and sl_reverse's .rv_loop classification, all classify on a copy with one CONS compare and stop there
     call PopCons                              ; fused: eax = successor, ebx = current element, one traversal per cell
     mov [rsp + LS_LIST_OFF], eax              ; root the successor BEFORE Cons can GC, frame slots are roots
     mov eax, ebx                              ; current element, subject to substitution
     cmp dword [rsp + LS_IDX_OFF], 0           ; at target index?
     jne .ls_use_orig                          ; no: keep the original element
     mov eax, [rsp + LS_VAL_OFF]               ; substitution
     mov dword [rsp + LS_FOUND_OFF], 1         ; mark replacement done
     mov dword [rsp + LS_IDX_OFF], -1          ; invalidate index, prevent re-replace
     jmp .ls_append                            ; go append it
.ls_use_orig:
     dec dword [rsp + LS_IDX_OFF]              ; index--, count toward target
.ls_append:
     xor edx, edx                              ; cdr = NIL, fresh tail cell
     call Cons                                 ; eax = fresh, element.NIL
     cmp dword [rsp + LS_HEAD_OFF], 0          ; first cell of the result?
     je .ls_first_cell                         ; yes: becomes the head
; link: overwrite the OWNED tail cell's CDR to point at the new cell
     mov ecx, [rsp + LS_TAIL_OFF]              ; ecx = tagged tail cell
     btr ecx, SL_CONS_BIT                      ; tail came from Cons, so bit30 was already clear
     mov dword [r13 + rcx*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF], eax ; tail.cdr = new cell, SIB replaces imul+add+mov; CONS_SCALE/CONS_BASE_OFF = the file's own names for 8/64MiB
     jmp .ls_update_tail                       ; continue
.ls_first_cell:
     mov dword [rsp + LS_HEAD_OFF], eax        ; head = first cell
.ls_update_tail:
     mov dword [rsp + LS_TAIL_OFF], eax        ; tail = new cell, always
     jmp .ls_walk                              ; continue walking
.ls_walk_done:
     cmp dword [rsp + LS_FOUND_OFF], 0         ; was the replacement done?
     je .ls_bad_idx                            ; no: index out of bounds -> FATAL
     mov eax, [rsp + LS_HEAD_OFF]              ; return the new list head
     add rsp, LS_FRAME                         ; drop frame
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     ret                                       ; return to caller
.ls_not_list:
     FATAL "SectorLogo: LSET requires a list"  ; non-list first argument
.ls_bad_idx:
     FATAL "SectorLogo: LSET index out of bounds" ; negative or past-the-end index
.ls_bad_idx_type:
     FATAL "SectorLogo: LSET index must be numeric" ; ATOM/CONS index argument
sl_setcar:
     xor edx, edx                              ; field = car, cell offset 0
     jmp sl_set_impl                           ; shared destructive-write body
sl_setcdr:
     mov edx, CONS_CDR_OFF                     ; field = cdr, cell offset 4
; fall through into the shared body
sl_set_impl:
     sub rsp, SC_FRAME                         ; 2 slots at offsets 32/36 above the 32-byte home area, calls stay aligned
     mov dword [rsp + SC_FIELD_OFF], edx       ; stage field offset BEFORE the first eval, edx is volatile
     call EvalNextArg                          ; eax = cell argument
     test eax, eax                             ; NIL, empty list ?
     jz .sc_not_cons                           ; yes: no cell to mutate -> FATAL
     lea ecx, [rax - SL_CONS_TAG]              ; CONS-band probe
     cmp ecx, SL_FLOAT_TAG - SL_CONS_TAG       ; above the band -> not a cons
     jae .sc_not_cons                          ; FLOAT/atom/NUMBER: not a cons cell -> FATAL
     mov dword [rsp + SC_CELL_OFF], eax        ; stage cell, GC root across the arg-2 evaluation
     call EvalNextArg                          ; eax = replacement value, any tag, stored verbatim
     mov ecx, [rsp + SC_CELL_OFF]              ; ecx = tagged cell
     btr ecx, SL_CONS_BIT                      ; bit30 was rejected by the prior tags checks
     mov edx, [rsp + SC_FIELD_OFF]             ; edx = 0, car slot or CONS_CDR_OFF, cdr slot
     lea rcx, [r13 + rcx*CONS_SCALE + CONS_BASE_OFF] ; rcx = cons cell address, base + index*scale + disp
     mov dword [rcx + rdx], eax                ; cell.field = value, SIB cannot hold a second index reg, address precomputed above
     xor eax, eax                              ; return NIL, Scheme unspecified value
     add rsp, SC_FRAME                         ; drop frame
     ret                                       ; return to caller
.sc_not_cons:
     cmp dword [rsp + SC_FIELD_OFF], CONS_CDR_OFF ; which primitive fired?
     jne .sc_not_cons_car                      ; car slot: SetCar message
     FATAL_VALUE "SetCdr requires a pair (a non-empty list cell) as its first argument", eax ; names type + value of the offender
.sc_not_cons_car:
     FATAL_VALUE "SetCar requires a pair (a non-empty list cell) as its first argument", eax ; names type + value of the offender
sl_samevalue:
     sub rsp, SV_FRAME                         ; align: entry 8 mod 16 -> 0 at both calls + home area
     call EvalNextArg                          ; eax = lhs, any tagged value
     mov [rsp + SV_LHS_OFF], rax               ; park lhs ABOVE the 32-byte shadow area, callee-safe slot
     call EvalNextArg                          ; eax = rhs, any tagged value
     cmp eax, dword [rsp + SV_LHS_OFF]         ; identical tagged word = same object?
     jne .sv_not_same                          ; no: different values
     mov eax, [rel atom_sl_t]                  ; yes: return the TRUE atom
     add rsp, SV_FRAME                         ; drop frame
     ret                                       ; return to caller
.sv_not_same:
     xor eax, eax                              ; return NIL, different values
     add rsp, SV_FRAME                         ; drop frame
     ret                                       ; return to caller
sl_asword:
     sub rsp, SV_FRAME                         ; align: entry 8 mod 16 -> 0 at both calls
     call EvalNextArg                          ; eax = argument atom
     ASSERT_ATOM_NONNIL rax, .asw_notatom      ; FATAL naming the offender: ASWORD needs a plain atom
     call lg_ensure_word                       ; STRING -> folded WORD re-intern; WORD -> identity return
     add rsp, SV_FRAME                         ; drop frame
     ret                                       ; eax = WORD-spelling atom
.asw_notatom:
     FATAL_VALUE "ASWORD requires an atom (string or word)", eax ; NIL / tagged value given to ASWORD
sl_lfind:
     mov ecx, LF_MODE_EQ                       ; mode = exact match
     jmp sl_lfind_impl                         ; shared implementation
sl_lfindge:
     mov ecx, LF_MODE_GE                       ; mode = greater-or-equal
     jmp sl_lfind_impl                         ; shared implementation
sl_lfindle:
     mov ecx, LF_MODE_LE                       ; mode = lesser-or-equal
; fall through to sl_lfind_impl
sl_lfind_impl:
     push rbx                                  ; save non-volatile, extracted key
     push rsi                                  ; save non-volatile
     push rdi                                  ; save non-volatile
     sub rsp, LF_FRAME                         ; 6 slots at offset 32..72 above the 32-byte Win64 home area, call sites stay aligned
     mov dword [rsp + LF_MODE_OFF], ecx        ; save comparison mode
; --- evaluate 3 arguments: list, startpos, key ---
     call EvalNextArg                          ; eax = list
     test eax, eax                             ; NIL, empty list ?
     jz .lf_empty_list                         ; yes: store NIL and STILL consume startpos+key, an early return would desync the evaluator stream
     lea ecx, [rax - SL_CONS_TAG]              ; CONS-band probe
     cmp ecx, SL_FLOAT_TAG - SL_CONS_TAG       ; above the band -> not a cons
     jae .lf_not_list                          ; FLOAT/atom/NUMBER: not a list -> FATAL
     mov dword [rsp + LF_LIST_OFF], eax        ; save list cursor, GC root
     jmp .lf_eval_start                        ; nonempty: continue with startpos
.lf_empty_list:
     mov dword [rsp + LF_LIST_OFF], 0          ; cursor = NIL: the skip/scan loops terminate on it -> not-found naturally
.lf_eval_start:
     call EvalNextArg                          ; eax = startpos argument
     mov ecx, eax                              ; ecx = copy for tag isolation
     and ecx, SL_TAG_MASK                      ; isolate tag bits
     cmp ecx, SL_NUMBER_TAG                    ; NUMBER?
     je .lf_start_num                          ; yes: extract integer
     cmp ecx, SL_FLOAT_TAG                     ; FLOAT?
     jne .lf_bad_start_type                    ; no: FATAL, ATOM/CONS/NIL
     call FloatVal                             ; xmm0 = float value
     vcvttss2si eax, xmm0                      ; truncate float to int32
     jmp .lf_start_check                       ; check range
.lf_start_num:
     call NumVal                               ; eax = raw int32 startpos
.lf_start_check:
     test eax, eax                             ; negative startpos?
     js .lf_bad_start                          ; yes: FATAL
     mov dword [rsp + LF_START_OFF], eax       ; save startpos
     call EvalNextArg                          ; eax = search key
     test eax, eax                             ; NIL key?
     jz .lf_bad_key                            ; yes: FATAL
     mov ecx, eax                              ; ecx = key copy for tag isolation
     shr ecx, 30                               ; ecx = 2-bit tag code (atom/NIL=0, NUMBER=1, CONS=2, FLOAT=3)
     cmp ecx, 2                                ; genuine CONS?
     je .lf_bad_key                            ; yes: FATAL, list is not a key
     cmp ecx, 1                                ; NUMBER tag?
     je .lf_key_ok                             ; yes: valid key
     cmp ecx, 3                                ; FLOAT tag?
     jne .lf_bad_key                           ; no: bare atom -> FATAL
.lf_key_ok:
     mov dword [rsp + LF_KEY_OFF], eax         ; save search key
; --- skip to startpos ---
     mov ecx, [rsp + LF_START_OFF]             ; ecx = steps to skip
     mov eax, [rsp + LF_LIST_OFF]              ; eax = cursor
.lf_skip:
     test ecx, ecx                             ; done skipping?
     jz .lf_scan_init                          ; yes: begin scanning
     test eax, eax                             ; list ended before startpos?
     jz .lf_not_found                          ; yes: return NIL
     call Cdr                                  ; eax = rest, leaf, no GC
     dec ecx                                   ; one step skipped
     jmp .lf_skip                              ; continue skipping
.lf_scan_init:
     mov dword [rsp + LF_POS_OFF], 0           ; scan-relative position = 0
     mov dword [rsp + LF_LAST_OFF], -1         ; LE mode: no match remembered yet
.lf_scan_loop:
     test eax, eax                             ; cursor NIL?
     jz .lf_scan_end                           ; exhausted, mode dispatch
     call PopCons                              ; EBX = element, EAX = successor
     mov dword [rsp + LF_LIST_OFF], eax        ; cache successor in the existing root slot
     mov eax, ebx                              ; process the element as before
; --- extract the comparison key from the element, SORT convention ---
     test eax, eax                             ; NIL element?
     jz .lf_advance                            ; yes: skip NIL and continue scanning
     test eax, SL_CONS_TAG                     ; bit31 set, CONS or FLOAT ?
     jz .lf_bare_key                           ; no: bare NUMBER
     test eax, SL_NUMBER_TAG                   ; bit30 also set = FLOAT?
     jnz .lf_bare_key                          ; yes: FLOAT is bare
; CONS sub-list: key = Car,element, then the SAME coercion the bare path runs
     call Car                                  ; eax = first sub-element (key candidate)
.lf_bare_key:
; bare NUMBER/FLOAT/ATOM or SUB-LIST's first element: coerce ATOMs to numeric, use as key
     call sl_sort_coerce_num                   ; eax = NUMBER or FLOAT, ATOMs parsed
     mov ebx, eax                              ; preserve coerced key, survives promote
.lf_compare:
     mov edx, [rsp + LF_KEY_OFF]               ; edx = search key, promote lhs
     mov eax, ebx                              ; eax = extracted key, promote rhs
     call sl_promote_pair                      ; edx=1: xmm0/xmm1 floats; edx=0: eax/ecx ints
     test edx, edx                             ; float comparison?
     jnz .lf_cmp_float                         ; yes: SSE compare path
; --- integer path: eax = search key raw, ecx = extracted key raw ---
     mov edx, [rsp + LF_MODE_OFF]              ; edx = comparison mode
     cmp edx, LF_MODE_EQ                       ; exact?
     je .lf_int_eq                             ; yes
     cmp edx, LF_MODE_GE                       ; greater-or-equal?
     je .lf_int_ge                             ; yes
     cmp ecx, eax                              ; LE: extracted <= search?
     jle .lf_le_match                          ; yes: remember, keep scanning
     jmp .lf_advance                           ; no: next element
.lf_int_eq:
     cmp ecx, eax                              ; extracted == search?
     je .lf_found                              ; yes: match
     jmp .lf_advance                           ; no: next element
.lf_int_ge:
     cmp ecx, eax                              ; extracted >= search?
     jge .lf_found                             ; yes: match
     jmp .lf_advance                           ; no: next element
.lf_cmp_float:
; --- float path: xmm0 = search key, xmm1 = extracted key ---
     mov edx, [rsp + LF_MODE_OFF]              ; edx = comparison mode
     cmp edx, LF_MODE_EQ                       ; exact?
     je .lf_flt_eq                             ; yes
     cmp edx, LF_MODE_GE                       ; greater-or-equal?
     je .lf_flt_ge                             ; yes
     vucomiss xmm1, xmm0                       ; extracted vs search, LE mode
     jp .lf_advance                            ; NaN: unordered, never matches
     jbe .lf_le_match                          ; extracted <= search: remember
     jmp .lf_advance                           ; no: next element
.lf_flt_eq:
     vucomiss xmm1, xmm0                       ; extracted vs search, EQ mode
     jp .lf_advance                            ; NaN: unordered, never equal
     je .lf_found                              ; equal: match
     jmp .lf_advance                           ; no: next element
.lf_flt_ge:
     vucomiss xmm1, xmm0                       ; extracted vs search, GE mode
     jp .lf_advance                            ; NaN: unordered, never matches
     jae .lf_found                             ; extracted >= search: match
; miss: fall through into .lf_advance
.lf_advance:
     mov eax, [rsp + LF_LIST_OFF]              ; successor fetched at loop entry
     inc dword [rsp + LF_POS_OFF]              ; scan position++
     jmp .lf_scan_loop                         ; continue scanning
.lf_le_match:
; LE mode hit: remember position as the LATEST <= key, then keep scanning
     mov eax, [rsp + LF_POS_OFF]               ; current scan position
     mov [rsp + LF_LAST_OFF], eax              ; overwrite earlier match
     jmp .lf_advance                           ; continue scanning
.lf_scan_end:
; list exhausted: EQ/GE return NIL; LE returns the remembered last match
     cmp dword [rsp + LF_MODE_OFF], LF_MODE_LE ; lesser-or-equal scan?
     jne .lf_not_found                         ; EQ/GE: exhausted -> NIL
     cmp dword [rsp + LF_LAST_OFF], -1         ; any <= key element seen?
     je .lf_not_found                          ; no: NIL
     mov eax, [rsp + LF_LAST_OFF]              ; eax = last-match position
     add eax, [rsp + LF_START_OFF]             ; absolute position, + startpos
     jmp .lf_ret_num                           ; shared MakeNum + teardown
.lf_found:
     mov eax, [rsp + LF_POS_OFF]               ; scan-relative position
     add eax, [rsp + LF_START_OFF]             ; absolute position, + startpos
     jmp .lf_ret_num                           ; shared MakeNum + teardown
.lf_not_found:
     xor eax, eax                              ; return NIL
.lf_exit:
     add rsp, LF_FRAME                         ; drop frame
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     ret                                       ; return to caller
.lf_ret_num:
     call MakeNum                              ; eax = tagged NUMBER, frame still live, same as before
     jmp .lf_exit                              ; shared teardown
.lf_not_list:
     FATAL "SectorLogo: LFIND requires a list" ; non-list first argument
.lf_bad_start:
     FATAL "SectorLogo: LFIND startpos must be non-negative" ; negative startpos
.lf_bad_start_type:
     FATAL "SectorLogo: LFIND startpos must be numeric" ; ATOM/CONS startpos argument
.lf_bad_key:
     FATAL "SectorLogo: LFIND key must be NUMBER or FLOAT" ; atom/CONS search key
; ---- %undef cleanup: file-local defines leave scope here, house convention ----
%undef LG_FRAME
%undef SV_LHS_OFF
%undef SV_FRAME
%undef LF_MODE_LE
%undef LF_MODE_GE
%undef LF_MODE_EQ
%undef LF_FRAME
%undef LF_LAST_OFF
%undef LF_MODE_OFF
%undef LF_START_OFF
%undef LF_POS_OFF
%undef LF_KEY_OFF
%undef LF_LIST_OFF
%undef SC_FRAME
%undef SC_FIELD_OFF
%undef SC_CELL_OFF
%undef LS_FRAME
%undef LS_LIST_OFF
%undef LS_IDX_OFF
%undef LS_VAL_OFF
%undef LS_HEAD_OFF
%undef LS_TAIL_OFF
%undef LS_FOUND_OFF
%undef RS_VAL_OFF
%undef RS_KEY_OFF
%undef RS_QW_SCALE
%undef RS_INDEX_LIMIT
%undef RS_REC_STRIDE
%undef RS_INSERT_MAX
%undef RS_HIST_SCALE
%undef RS_HIST_BUCKETS
%undef RS_NUM_PASSES
%undef HF_FRAME
%undef HF_WINDOW
%undef HF_QWORD
%undef SS_LIST
%undef SS_FRAME
