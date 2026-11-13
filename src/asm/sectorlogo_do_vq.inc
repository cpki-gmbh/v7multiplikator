; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; --- file-local frame/slot constants, comment checker: named offsets only ---
%define VQ_VEC_OFF 32                          ; staged vector list, GC root
%define VQ_CB_OFF 40                           ; codebook cursor, GC root
%define VQ_ROW_OFF 48                          ; row cursor, GC root
%define VQ_KEY_OFF 56                          ; coerced row key, GC root
%define VQ_PAIR_OFF 64                         ; staged dist FLOAT / partial pair, GC root
%define VQ_RCNT_OFF 72                         ; full-mode record count, raw dword
%define VQ_BESTK_OFF 88                        ; best key, GC root
%define VQ_BESTD_OFF 96                        ; best dist tagged FLOAT, GC root
%define VQ_N_OFF 100                           ; vector length n, raw dword
%define VQ_BESTF_OFF 104                       ; best distance binary32 bits, raw dword, exact, never 30-bit packed
%define VQ_FULL_OFF 112                        ; full flag, raw dword: 0=non-full, 1=full; ebx overwritten by PopCons
%define VQ_MAINEND_OFF 116                     ; n & ~7 * 4, main-loop byte bound, staged once per call
%define VQ_TAILEND_OFF 120                     ; n * 4, tail END byte offset, staged once per call, equals r10 +, n&7 *4 on BOTH entry paths
%define VQ_RECPTR_OFF 124                      ; full-mode sort record write cursor, qword, advances 16 per row
%define VQ_VECBUF_OFF 144                      ; stack-staged vector floats, SL_VQ_MAX_ELEMS dwords, re-entrancy safe.Layout: 0..31 Win64 home area, 32..123 the named dword slots, 124..131 the record cursor, 132..143 pad, 144.. the vector buffer. 144 is 0 mod 16, which is what the vmovss staging in .vq_stage_vec needs; the call-site alignment is restored by the +8 in VQ_FRAME, not by this pad
%define VQ_FRAME (144 + SL_VQ_MAX_ELEMS * 4 + 8) ; total stack frame: staging area + vector buffer + the +8 alignment pad for the 4 pushes
%define VQ_FLOAT_SCALE 4                       ; bytes per staged binary32 in the float buffers
%define VQ_REC_VAL_OFF 0                       ; sort record dword 0: tagged key, output value, sl_sort_records layout
%define VQ_REC_DIST_OFF 4                      ; sort record dword 4: tagged distance FLOAT
%define VQ_REC_KEY_OFF 8                       ; sort record qword at +8: monotone binary64 sort key
%define VQ_REC_STRIDE 16                       ; bytes per sort record, sl_sort_qw_a layout
%define VQ_REC_SCALE 8                         ; SIB scale for the DOUBLED element index, 2*idx * 8 = 16-byte record stride
sl_do_vq:
     push rbx                                  ; preserve non-volatile for the caller, PopCons writes ebx without saving it, which is why the full flag lives in the VQ_FULL_OFF stack slot, not here
     push rsi                                  ; save non-volatile, buffer base staging
     push rdi                                  ; save non-volatile, alignment
     push r15                                  ; save non-volatile, row-component index; the evaluator overwrites r15d, so the pair is load-bearing, VQ_FRAME carries the matching +8 alignment pad
     sub rsp, VQ_FRAME                         ; named scratch slots + stack vector buffer, call sites stay aligned
; ---------- evaluate arg 1: the full flag ----------
     call EvalNextArg                          ; eax = full flag, bare atom self-evaluates
     test eax, SL_TAG_MASK                     ; tag bits set, NUMBER/FLOAT/CONS ?
     jnz .vq_bad_flag                          ; yes: FATAL, flags-only test: eax survives for the atom compares
     cmp eax, [rel atom_sl_t]                  ; interned TRUE?
     je .vq_flag_full                          ; yes
; The language's false IS NIL: every equality/relational primitive returns the
; TRUE atom or NIL, infix =, EqualP, ....A bare `False` word self-evaluates
; to the interned FALSE word atom. All three spellings select non-full mode.
     test eax, eax                             ; NIL, the language's false ?
     jz .vq_flag_nonfull                       ; yes
     cmp eax, [rel atom_sl_false]              ; interned FALSE word?
     jne .vq_bad_flag                          ; any other atom: FATAL
.vq_flag_nonfull:
     mov dword [rsp + VQ_FULL_OFF], 0          ; non-full mode, stack slot: survives PopCons overwriteing ebx
     jmp .vq_eval_vec                          ; continue with arg 2, shared path
.vq_flag_full:
     mov dword [rsp + VQ_FULL_OFF], 1          ; full mode, stack slot: survives PopCons overwriteing ebx
     lea rax, [rel sl_sort_qw_a]               ; sort record buffer base, BSS, sl_sort_records layout
     mov [rsp + VQ_RECPTR_OFF], rax            ; park the write cursor for the whole FULL scan
     mov dword [rsp + VQ_RCNT_OFF], 0          ; no records staged yet
.vq_eval_vec:
; ---------- evaluate arg 2: the vector; stage floats ----------
     call EvalNextArg                          ; eax = vector list
     test eax, eax                             ; NIL?
     jz .vq_bad_vec                            ; yes: FATAL
     TAG_IS_JUMP_NE eax, SL_CONS_TAG, .vq_bad_vec ; vector must be a CONS
     mov dword [rsp + VQ_VEC_OFF], eax         ; GC-root vector cursor
     xor edi, edi                              ; rdi = component index i, REGISTER, not a stack slot.rdi is VOLATILE: PopCons, writes ebx/eax only, sl_sort_coerce_num, 0 rdi writes across its whole body and sl_to_float -> NumVal/FloatVal, write eax/xmm0 only happen to leave it untouched, observed property, NOT an ABI guarantee. A future edit to any of them that writes rdi corrupts the staged vector at :67 and then trips .vq_overflow; re-stage i in a frame slot if that happens. rdi is not given the row-buffer role until :87, revisit if that ever moves earlier
     lea rsi, [rsp + VQ_VECBUF_OFF]            ; rsi = stack staging buffer base, hoisted: callees in the loop preserve rsi
.vq_stage_vec:
     mov eax, [rsp + VQ_VEC_OFF]               ; eax = vector cursor
     test eax, eax                             ; end of vector?
     jz .vq_vec_staged                         ; yes: staging done
     cmp edi, SL_VQ_MAX_ELEMS                  ; past the staging buffer?
     jae .vq_overflow                          ; yes: FATAL
; eax still holds the vector cursor, no reload needed
     TAG_IS_JUMP_NE eax, SL_CONS_TAG, .vq_bad_vec ; staging cursor must be a CONS
     call PopCons                              ; ebx = component, eax = rest
     mov dword [rsp + VQ_VEC_OFF], eax         ; cursor = rest
     mov eax, ebx                              ; element
     test eax, eax                             ; NIL?
     jz .vq_bad_vec                            ; yes: FATAL
     call sl_sort_coerce_num                   ; eax = NUMBER/FLOAT, ATOMs parsed
     call sl_to_float                          ; xmm0 = component float
     vmovss dword [rsi + rdi*VQ_FLOAT_SCALE], xmm0 ; vec_buf[i] = component, rdi holds i, per the register contract at :50
     inc edi                                   ; i++
     jmp .vq_stage_vec                         ; continue staging vector components
.vq_vec_staged:
     mov eax, edi                              ; eax = n, n >= 1 proven: a NIL vector FATALs at :46-47 and the CONS gate at :48 precedes staging, so the loop always stages at least one component before its cursor can hit NIL
     mov dword [rsp + VQ_N_OFF], eax           ; n = component count
; ---- stage the two n-derived byte bounds, loop-invariant per call ----
     lea r11d, [rax*VQ_FLOAT_SCALE]            ; r11d = n * 4 = total byte extent, n <= SL_VQ_MAX_ELEMS = 256, so no overflow
     mov dword [rsp + VQ_TAILEND_OFF], r11d    ; tail END offset, same value on both entry paths
     and r11d, ~31                             ; n*4 & ~31 ==, n & ~7 * 4, one instruction instead of and + lea: n's low 3 bits sit in bits 2..4 of n*4
     mov dword [rsp + VQ_MAINEND_OFF], r11d    ; main-loop byte bound
; ---------- evaluate arg 3: the codebook ----------
     call EvalNextArg                          ; eax = codebook list
     test eax, eax                             ; NIL?
     jz .vq_bad_cb                             ; yes: FATAL
     TAG_IS_JUMP_NE eax, SL_CONS_TAG, .vq_bad_cb ; codebook must be a CONS
     mov dword [rsp + VQ_CB_OFF], eax          ; GC-root codebook cursor
     xor eax, eax                              ; zero source for the result slot
     mov dword [rsp + VQ_BESTD_OFF], eax       ; best dist = NIL, no row yet
     lea rdi, [rel sl_vq_row_buf]              ; rdi = row buffer base, BSS; loop-invariant, row staging never re-enters the evaluator
     lea rsi, [rsp + VQ_VECBUF_OFF]            ; rsi = STACK vector buffer base, survives nested NearestMatch during codebook eval; no writes to rsi between here and the dist loop
; ---------- walk the codebook rows ----------
.vq_row_loop:
     mov eax, [rsp + VQ_CB_OFF]                ; eax = codebook cursor
     TAG_IS_JUMP_NE eax, SL_CONS_TAG, .vq_bad_cb ; codebook row slot must be a CONS, improper tail: FATAL, matches vector/row handling
     btr eax, SL_CONS_BIT                      ; strip tag -> cell index, CONS band proven by the TAG_IS_JUMP_NE above, inline Car body, guard compares redundant
     mov eax, [r13 + rax*CONS_SCALE + CONS_BASE_OFF] ; eax = row, car field
     TAG_IS_JUMP_NE eax, SL_CONS_TAG, .vq_row_notlist ; row must be a CONS
     mov dword [rsp + VQ_ROW_OFF], eax         ; GC-root row cursor
; ---- extract + coerce the row key ----
     btr eax, SL_CONS_BIT                      ; strip tag -> cell index, row's CONS proven by the TAG_IS_JUMP_NE above, inline Car body, guard compares redundant
     mov eax, [r13 + rax*CONS_SCALE + CONS_BASE_OFF] ; eax = key, car field
     test eax, eax                             ; NIL key?
     jz .vq_bad_row                            ; yes: FATAL
     call sl_sort_coerce_num                   ; eax = NUMBER/FLOAT, ATOMs parsed; malformed input FATALs inside, only successful exits return
     mov dword [rsp + VQ_KEY_OFF], eax         ; GC-root coerced key
; ---- stage the row floats into sl_vq_row_buf ----
     mov eax, [rsp + VQ_ROW_OFF]               ; eax = row
     btr eax, SL_CONS_BIT                      ; strip tag -> cell index, row's CONS proven at the TAG_IS_JUMP_NE above; slot unmodified since
     mov eax, [r13 + rax*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF] ; eax = row past the key, cdr field
     mov dword [rsp + VQ_ROW_OFF], eax         ; row cursor = past key
     xor r15d, r15d                            ; i = 0, REGISTER: non-volatile, survives PopCons/sl_sort_coerce_num/sl_to_float; removes a store-to-load-forwarded slot RMW
.vq_stage_row:
     mov eax, [rsp + VQ_ROW_OFF]               ; eax = row cursor
     test eax, eax                             ; row ended early?
     jz .vq_bad_rowlen                         ; yes: row length != n -> FATAL
     TAG_IS_JUMP_NE eax, SL_CONS_TAG, .vq_bad_rowlen ; staging cursor must be a CONS, improper tail: treat as row end
     call PopCons                              ; ebx = component, eax = rest
     mov dword [rsp + VQ_ROW_OFF], eax         ; cursor = rest
     mov eax, ebx                              ; element
     test eax, eax                             ; NIL component?
     jz .vq_bad_comp                           ; yes: FATAL
     call sl_sort_coerce_num                   ; eax = NUMBER/FLOAT, ATOMs parsed
     call sl_to_float                          ; xmm0 = row float
     vmovss dword [rdi + r15*VQ_FLOAT_SCALE], xmm0 ; row_buf[i] = component, rdi = row base hoisted at :87, r15 = i
     inc r15d                                  ; next component index
     cmp r15d, [rsp + VQ_N_OFF]                ; staged every vector dimension?
     jb .vq_stage_row                          ; continue until n elements are staged
.vq_row_lenchk:
     mov eax, [rsp + VQ_ROW_OFF]               ; leftover cursor, tagged offender for the FATAL
     test eax, eax                             ; leftover cells after n components?
     jne .vq_bad_rowlen                        ; yes: FATAL, length mismatch
; ---------- distance: 8-wide packed main loop + scalar tail ----------
.vq_dist_loop:
     mov r11d, [rsp + VQ_MAINEND_OFF]          ; r11d = main-loop byte bound, n & ~7 * 4, staged once per call
     xor r10d, r10d                            ; r10d = byte offset i = 0
     cmp r10d, r11d                            ; any full 8-float blocks?, n < 8 palette case; rsi already = vec_buf base
     jae .vq_dist_no_vec                       ; no: skip the 6-instruction hsum of an all-zero accumulator
     vpxor ymm2, ymm2, ymm2                    ; ymm2 = 8-lane accumulator = 0.0, packed path ONLY, the jae above skips it, and above the .vq_dist_main label so the loop's back-edge cannot re-zero it
     jmp .vq_dist_main                         ; yes: r10 = 0 < r11, so the loop's bottom test is redundant on entry
.vq_dist_no_vec:
; n < 8, the common palette path : the main loop would have accumulated nothing, so the
; vector sum is exactly 0.0. Produce it directly instead of reducing a zeroed ymm2.
     vxorps xmm0, xmm0, xmm0                   ; xmm0 = 0.0 == hsum,ymm2 = 0
     jmp .vq_dist_tail_entry                   ; go directly to scalar tail setup, r10d is already 0: the only write above is the xor r10d,r10d, and this branch is taken exactly when n < 8
.vq_dist_main:
     vmovups ymm0, [rsi + r10]                 ; 8 vector floats
     vsubps ymm0, ymm0, [rdi + r10]            ; diff = v - c, memory operand, one 256-bit move gone
     vfmadd231ps ymm2, ymm0, ymm0              ; accumulator += diff * diff, one rounding, strictly less error than mul+add
     add r10d, 32                              ; next 8 floats, 32 bytes
     cmp r10d, r11d                            ; more full blocks?
     jb .vq_dist_main                          ; yes: next iteration
.vq_dist_hsum:
     vextractf128 xmm0, ymm2, 1                ; xmm0 = high 128 bits
     vaddps xmm0, xmm0, xmm2                   ; xmm0 = low+high, 4 lanes
     vmovhlps xmm1, xmm0, xmm0                 ; xmm1 = lanes 2:3 in low half, high half from xmm0, no dependence on stale xmm1
     vaddps xmm0, xmm0, xmm1                   ; xmm0 = lane0+lane2, lane1+lane3
     vmovshdup xmm1, xmm0                      ; xmm1 = lane1 duplicated
     vaddss xmm0, xmm0, xmm1                   ; xmm0 = scalar dist, main part
; ---- scalar tail over the n&7 remainder, bottom-tested; entry guard handles zero-trip ----
.vq_dist_tail_entry:
     mov r11d, [rsp + VQ_TAILEND_OFF]          ; r11d = tail END byte offset = n * 4, staged once; equals r10 +, n&7 *4 on both entry paths
     cmp r10d, r11d                            ; zero-trip guard: n&7 == 0 -> straight to scoring
     jae .vq_row_scored                        ; yes: dist complete
.vq_dist_tail:
     vmovss xmm1, [rsi + r10]                  ; v[i], scalar tail
     vsubss xmm1, xmm1, [rdi + r10]            ; c[i] consumed once, memory operand, no register staging
     vfmadd231ss xmm0, xmm1, xmm1              ; dist += diff * diff, one rounding, strictly less error
     add r10d, 4                               ; next float, register add
     cmp r10d, r11d                            ; tail done?, bottom test, no loop-closing jmp
     jb .vq_dist_tail                          ; continue, ≤ 7 iterations
.vq_row_scored:
; xmm0 = dist float; wrap as a tagged FLOAT, inline MakeFloat: per-row pack
     vmovd eax, xmm0                           ; dist binary32 bits -> GPR
     shr eax, SL_FLOAT_SHIFT                   ; drop 2 low mantissa bits -> 30-bit payload
     or eax, SL_FLOAT_TAG                      ; set FLOAT tag
     mov dword [rsp + VQ_PAIR_OFF], eax        ; stage dist, GC root across Cons
; ---- full mode: write a sort record (key, dist FLOAT, monotone key) ----
     cmp dword [rsp + VQ_FULL_OFF], 0          ; full mode?, stack slot: ebx overwritten by PopCons
     jz .vq_nonfull_cmp                        ; no: min-tracking path
     mov ecx, [rsp + VQ_RCNT_OFF]              ; ecx = records staged so far
     cmp ecx, SL_SORT_MAX_ELEMS                ; record buffer capacity reached?
     jae .vq_sort_overflow                     ; yes: FATAL, same overflow policy as SORT
     mov r8, [rsp + VQ_RECPTR_OFF]             ; r8 = this row's 16-byte record slot
     mov eax, [rsp + VQ_KEY_OFF]               ; eax = tagged key, the output value
     mov dword [r8 + VQ_REC_VAL_OFF], eax      ; record.key
     mov eax, [rsp + VQ_PAIR_OFF]              ; eax = tagged distance FLOAT, packed above
     mov dword [r8 + VQ_REC_DIST_OFF], eax     ; record.dist
     shl eax, SL_FLOAT_SHIFT                   ; reconstruct FLOAT bits, tag bits 31:30 fall off
     test eax, SL_F32_NOT_SIGN_MASK            ; same signed-zero normalization as SORT
     jnz .vq_key_widen                         ; real value, Inf or NaN: keep the bits
     xor eax, eax                              ; +or-0.0: normalize to +0.0 bits
.vq_key_widen:
     vmovd xmm0, eax                           ; xmm0 dead after the FP32 pack: the dist loop's consumer is done
     vcvtss2sd xmm0, xmm0, xmm0                ; exact widening to binary64
     vmovq rcx, xmm0                           ; rcx = binary64 key bits
     mov rax, rcx                              ; copy for the sign mask
     sar rax, 63                               ; positive: 0, negative: -1
     bts rax, 63                               ; positive: topbit, negative: all-ones
     xor rcx, rax                              ; monotone unsigned 64-bit ordering key
     mov qword [r8 + VQ_REC_KEY_OFF], rcx      ; record.sort key
     add qword [rsp + VQ_RECPTR_OFF], VQ_REC_STRIDE ; next row's record slot
     inc dword [rsp + VQ_RCNT_OFF]             ; one more record staged
     jmp .vq_row_advance                       ; next codebook row
.vq_nonfull_cmp:
; ---- non-full: keep the least dist; first-encountered wins ties ----
     cmp dword [rsp + VQ_BESTD_OFF], 0         ; no row seen yet?
     je .vq_nonfull_first                      ; yes: this row wins by default
     vmovd xmm4, dword [rsp + VQ_BESTF_OFF]    ; xmm4 = exact best distance
     vucomiss xmm0, xmm4                       ; current exact dist vs best exact dist
     jnp .vq_ordered                           ; ordered, neither NaN : normal compare below
; ---- unordered: at least one operand is NaN, find out which ----
     vucomiss xmm4, xmm4                       ; is BEST the NaN?
     jnp .vq_row_advance                       ; best finite: CURRENT is the NaN, keep best
     vucomiss xmm0, xmm0                       ; is current NaN too?
     jnp .vq_nonfull_first                     ; best NaN, current finite: replace, first finite wins
     jmp .vq_row_advance                       ; both NaN: keep the first-encountered NaN as best
.vq_ordered:
     jae .vq_row_advance                       ; current >= best: keep best
; fall through: current < best, it wins
.vq_nonfull_first:
     vmovd dword [rsp + VQ_BESTF_OFF], xmm0    ; best raw binary32 = current exact dist, init before any compare reads it
     mov eax, [rsp + VQ_PAIR_OFF]              ; eax = dist FLOAT
     mov dword [rsp + VQ_BESTD_OFF], eax       ; best dist = current
     mov eax, [rsp + VQ_KEY_OFF]               ; eax = key
     mov dword [rsp + VQ_BESTK_OFF], eax       ; best key = current
; fall through to advance
.vq_row_advance:
     mov eax, [rsp + VQ_CB_OFF]                ; eax = codebook cursor
     call Cdr                                  ; eax = next row, leaf
     mov dword [rsp + VQ_CB_OFF], eax          ; cursor = next
     test eax, eax                             ; NIL marks the end of the codebook
     jnz .vq_row_loop                          ; another row exists
; ---------- full mode: sort + rebuild ----------
.vq_rows_done:
     cmp dword [rsp + VQ_FULL_OFF], 0          ; full mode?, stack slot: ebx overwritten by PopCons
     jz .vq_nonfull_rebuild                    ; no: rebuild winner only
     mov eax, [rsp + VQ_RCNT_OFF]              ; eax = record count from the scan
     mov [rel sl_sort_n], eax                  ; publish for the shared record sorter
     call sl_sort_records                      ; rbx = sorted record buffer, ascending by the packed dist, stable on equal dists
; Rebuild the output list head-to-tail from the sorted records: for each
; record allocate (dist . NIL), then (key . inner), then prepend. Walking
; records N-1..0 under prepend yields ascending order; equal distances keep
; first-encounter order because the record sort is stable.
     mov esi, [rel sl_sort_n]                  ; i = N, esi write zero-extends for the doubled addressing
     xor edi, edi                              ; accumulator = NIL, register-held, GC-safe: GC_Collect spills rdi
.vq_full_rebuild_loop:
     dec esi                                   ; i--, element N-1 down to 0
     lea rdx, [rsi + rsi]                      ; rdx = doubled element index, 16-byte record stride
     mov eax, dword [rbx + rdx*VQ_REC_SCALE + VQ_REC_DIST_OFF] ; tagged dist, scalar, GC-inert record buffer
     xor edx, edx                              ; cdr = NIL
     call Cons                                 ; eax =, dist.NIL inner cell
     mov dword [rsp + VQ_PAIR_OFF], eax        ; GC-root the inner cell
     lea rdx, [rsi + rsi]                      ; doubled index again, Cons overwrote rdx
     mov eax, dword [rbx + rdx*VQ_REC_SCALE + VQ_REC_VAL_OFF] ; tagged key
     mov edx, [rsp + VQ_PAIR_OFF]              ; edx = rooted inner cell
     call Cons                                 ; eax =, key.inner pair
     mov dword [rsp + VQ_KEY_OFF], eax         ; GC-root the pair
     mov edx, edi                              ; cdr = current accumulator
     mov eax, [rsp + VQ_KEY_OFF]               ; car = rooted pair
     call Cons                                 ; eax = new accumulator head
     mov edi, eax                              ; accumulator = new head
     test esi, esi                             ; element 0 done?, re-test: call Cons overwrites flags
     jnz .vq_full_rebuild_loop                 ; no: keep rebuilding
.vq_full_done:
     mov eax, edi                              ; the sorted [key dist] list head
     jmp .vq_done                              ; return it
; ---------- non-full: build [key dist] for the winner ----------
.vq_nonfull_rebuild:
     mov eax, [rsp + VQ_BESTD_OFF]             ; eax = best dist FLOAT
     xor edx, edx                              ; cdr = NIL
     call Cons                                 ; eax =, best_dist.NIL
     mov dword [rsp + VQ_PAIR_OFF], eax        ; GC-root the inner cell, edx is NOT GC-scannable
     mov edx, [rsp + VQ_PAIR_OFF]              ; edx = inner cell, reload from the slot
     mov eax, [rsp + VQ_BESTK_OFF]             ; eax = best key
     call Cons                                 ; eax = [key dist] winner pair
; fall through to return
.vq_done:
     vzeroupper                                ; clean YMM upper halves, ymm2 dirtied by the dist loop before returning to the evaluator
     add rsp, VQ_FRAME                         ; drop frame
     pop r15                                   ; restore non-volatile, row-component index, last push, first pop
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     ret                                       ; return to caller
; ---------- FATAL exits ----------
.vq_overflow:
     FATAL "SectorLogo: NearestMatch vector exceeds staging buffer" ; n > SL_VQ_MAX_ELEMS
.vq_sort_overflow:
     FATAL "SectorLogo: NearestMatch result limit exceeded" ; codebook rows > SL_SORT_MAX_ELEMS in full mode
.vq_bad_flag:
     FATAL "SectorLogo: NearestMatch full flag must be true, false or NIL (got a tagged value or a foreign atom)" ; non-atom / numeric / cons flag value
.vq_bad_vec:
     FATAL_VALUE "NearestMatch vector requires numbers", eax ; non-list / empty / non-numeric vector
.vq_bad_cb:
     FATAL_VALUE "NearestMatch codebook requires rows", eax ; non-list / empty codebook
.vq_row_notlist:
     FATAL_VALUE "NearestMatch codebook rows must be lists", eax ; a codebook row that is not a list
.vq_bad_row:
     FATAL "SectorLogo: NearestMatch codebook row has no components" ; NIL key, the row IS a list here; the key slot is empty
.vq_bad_rowlen:
     FATAL_VALUE "NearestMatch codebook row length != vector length", eax ; short row / leftover tail
.vq_bad_comp:
     FATAL "SectorLogo: NearestMatch codebook row requires numbers" ; NIL/non-numeric row component
; %undef cleanup: file-local defines leave scope here, house convention --
%undef VQ_FLOAT_SCALE
%undef VQ_FRAME
%undef VQ_VECBUF_OFF
%undef VQ_TAILEND_OFF
%undef VQ_MAINEND_OFF
%undef VQ_FULL_OFF
%undef VQ_BESTF_OFF
%undef VQ_N_OFF
%undef VQ_BESTD_OFF
%undef VQ_BESTK_OFF
%undef VQ_RCNT_OFF
%undef VQ_RECPTR_OFF
%undef VQ_REC_VAL_OFF
%undef VQ_REC_DIST_OFF
%undef VQ_REC_KEY_OFF
%undef VQ_REC_STRIDE
%undef VQ_REC_SCALE
%undef VQ_PAIR_OFF
%undef VQ_KEY_OFF
%undef VQ_ROW_OFF
%undef VQ_CB_OFF
%undef VQ_VEC_OFF
