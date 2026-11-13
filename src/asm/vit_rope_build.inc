; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define VRB_INVF_MAX 32                        ; max rope_qtr entries, head_dim <= 128, capacity is data-driven; rope_qtr within it is the import's head_dim contract, no engine guard
%define VRB_X87_I 0                            ; x87 temp frame slot: exponent index i, int32 for fild
%define VRB_X87_QTR 4                          ; x87 temp frame slot: runtime rope_qtr, int32 for fild
%define VRB_ROW_SLOT 0                         ; x87 temp frame slot: exact patch row, int32 for fimul
%define VRB_COL_SLOT 4                         ; x87 temp frame slot: exact patch col, int32 for fimul
%define VRB_COL_SCALE 2                        ; SIB scale: bcol * 2 row/col fold
%define VRB_BLOCK_RECORDS 4                    ; records per 2x2 patch block, the row-anchor distance scale
%define VRB_QWORD_SCALE 8                      ; sizeof,qword, inv_freq cache stride (binary64 entries)
     section .bss
g_vrb_invf:
     resq VRB_INVF_MAX                         ; 32 x float64, retain x87 precision
g_vrb_theta_bits:
     resd 1                                    ; rope_theta bits, memoized; F == 0 = never computed
g_vrb_theta_qtr:
     resd 1                                    ; qtr the inv_freq cache was built with, key component 2, values depend on qtr
g_cached_rb_gh:
     resd 1                                    ; cached grid_h for resolution memoization, 0 = never computed
g_cached_rb_gw:
     resd 1                                    ; cached grid_w
g_vrb_last_cos:
     resq 1                                    ; cos buffer the cached tables were built into, identity half of the cache key
g_vrb_last_sin:
     resq 1                                    ; sin buffer the cached tables were built into
     section .text
     SF_START vit_rope_build, rbx, rdi, rsi, r12, r13, r14, r15 ; prologue: save all non-volatiles written, r13d/ebx/edi are hoisted for the patch loop
     SF_QWORD vrb_cos                          ; cos table base
     SF_QWORD vrb_sin                          ; sin table base
     SF_DWORD vrb_gh                           ; grid height
     SF_DWORD vrb_gw                           ; grid width
     SF_QWORD vrb_n                            ; total patch count n = gh * gw
     SF_DWORD vrb_in_row                       ; cascade: in-sub-patch row toggle, 0/1
     SF_DWORD vrb_bcol                         ; cascade: block column counter
     SF_DWORD vrb_brow                         ; cascade: block row counter
     SF_COMMIT                                 ; finalize stack frame layout
     cmp r8d, [rel g_cached_rb_gh]             ; same grid_h as last time?
     jne .vrb_compute                          ; no, must recompute
     cmp r9d, [rel g_cached_rb_gw]             ; same grid_w?
     jne .vrb_compute                          ; no, must recompute
     cmp rcx, [rel g_vrb_last_cos]             ; same cos buffer as last time?
     jne .vrb_compute                          ; no, new context, fresh buffer: rebuild
     cmp rdx, [rel g_vrb_last_sin]             ; same sin buffer?
     jne .vrb_compute                          ; no, rebuild
     mov eax, dword [rel g_vit_cfg + VCFG_ROPE_THETA_LOG2] ; this model's theta bits
     cmp eax, [rel g_vrb_theta_bits]           ; tables built with the SAME theta?
     jne .vrb_compute                          ; no, model switch: rebuild, tables depend on theta
     mov eax, dword [rel g_vit_cfg + VCFG_ROPE_QTR] ; this model's qtr, tables depend on it via inv_freq AND half
     cmp eax, [rel g_vrb_theta_qtr]            ; built with the SAME qtr?
     je .vrb_done                              ; cache hit, tables already in THIS buffer, just return; a different qtr falls into the rebuild
.vrb_compute:
     mov vrb_cos, rcx                          ; save cos base
     mov vrb_sin, rdx                          ; save sin base
     mov vrb_gh, r8d                           ; save gh
     mov vrb_gw, r9d                           ; save gw
     mov eax, r8d                              ; gh, a 32-bit store zero-extends into rax, so this IS the zero-extension
     mov ecx, r9d                              ; ecx = gw, the 64-bit imul below needs a clean upper half
     imul rax, rcx                             ; n = gh * gw, 64-bit product
     mov vrb_n, rax                            ; save n, full 64-bit product
     mov edi, r9d                              ; block-column count before halving
     shr edi, 1                                ; gw / 2, preserved across the builder
     sub rsp, 16                               ; x87 temp frame: 16 bytes of scratch below every SF slot, SF_COMMIT already leaves rsp at rbp-144, so no slot alias is possible here. Holds the exact integer row/col for the patch loop and, on the inv_freq miss path, the two fild operands VRB_X87_I / VRB_X87_QTR. Those two loops share this frame and are mutually exclusive: the inv_freq loop is skipped on a cache hit and the patch loop is reachable only after it. Freed once at .vrb_frames_out.
; Initialize cascade counters, replace per-patch div with simple toggles .
; in_col lives in r11d and is zeroed in the .vrb_invf_done prologue.
     mov dword vrb_in_row, 0                   ; inner row counter
     mov dword vrb_bcol, 0                     ; block column counter
     mov dword vrb_brow, 0                     ; block row counter
     mov eax, dword [rel g_vit_cfg + VCFG_ROPE_THETA_LOG2] ; this model's theta bits
     lea rsi, [rel g_vrb_invf]                 ; rsi = &inv_freq[0], BSS cache, loop-invariant
; rope_qtr is a runtime, DB-owned value, the *4 byte scale is folded with the sign flip below, never a compile-time byte count
     mov r10d, dword [rel g_vit_cfg + VCFG_ROPE_QTR] ; runtime split point, zero-extends into r10
     mov edx, r10d                             ; qtr for the cache key and rebuild
     imul r10, -8                              ; -(qtr * sizeof(double))
     add r10, rsi                              ; r10 = &invf[0] - qtr*8; [r10+i*8] == invf[i-qtr]
     cmp eax, [rel g_vrb_theta_bits]           ; cache built with the same theta?
     jne .vrb_qtr_key_miss                     ; no, rebuild, capacity FATAL re-runs too
     cmp edx, [rel g_vrb_theta_qtr]            ; cache built with the same qtr?
     je .vrb_invf_done                         ; yes, reuse cached table
.vrb_qtr_key_miss:
     mov [rel g_vrb_theta_bits], eax           ; publish cache key before building, rope_qtr <= VRB_INVF_MAX is the import's head_dim contract
     mov [rel g_vrb_theta_qtr], edx            ; publish cache key 2, qtr, the cached values depend on it, not on theta alone
; Loop body is pure x87, no hoisted xmm constants. The hoisted r11d is the loop bound AND the fild source for rope_qtr.
     mov r11d, edx                             ; hoisted loop bound, reg-reg copy; r11 free file-wide, no use between entry and the patch loop
     xor ecx, ecx                              ; i = 0, register counter, no stack traffic
     mov [rsp + VRB_X87_QTR], r11d             ; rope_qtr as int32 for fild, loop-invariant, stored once
.vrb_invf_loop:
; rope_qtr >= 4 by the model-import contract, so the first iteration executes directly
     mov [rsp + VRB_X87_I], ecx                ; i as int32 for fild
     fild dword [rsp + VRB_X87_I]              ; st0 =, extended i
     fild dword [rsp + VRB_X87_QTR]            ; st0 = rope_qtr, stored once above the loop, st1 = i
     fdivp st1, st0                            ; st0 = i/rope_qtr in extended, exact for power-of-two qtr, correctly rounded beyond
     fchs                                      ; st0 = -,i/rope_qtr, exact; replaces the vxorps sign-mask trick
     fld dword [rel g_vit_cfg + VCFG_ROPE_THETA_LOG2] ; st0 = log2,theta, f32 bits widened exactly, st1 = -,i/rope_qtr
     fmulp st1, st0                            ; st0 = x, net: push 3, pop 2, x alone on the stack, which is what the fld st0 below expects
; Compute 2^x via x87 f2xm1+fscale. f2xm1 only works on [-1,1], so split x=n+f.
; Stack management is critical: fscale uses st,1 as the exponent, so st,1 MUST hold round,x .
     fld st0                                   ; st0 = x, copy, st1 = x
     frndint                                   ; st0 = round,x = n, st1 = x
     fsub st1, st0                             ; st1 = frac = x - n, st0 = n
     fxch st1                                  ; st0 = frac, st1 = n
     f2xm1                                     ; st0 = 2^frac - 1, st1 = n
     fld1                                      ; st0 = 1, st1 = 2^frac-1, st2 = n
     faddp st1, st0                            ; st0 = 2^frac, st1 = n, faddp: st1+=st0, pop
     fscale                                    ; st0 = 2^frac * 2^n = 2^x, st1=n is the exponent AND REMAINS ON THE STACK, FSCALE does not pop, despite the SDM entry's "Examine Scale and Pop" title; the fstp below stores st0 and the fstp after it drops n
     fstp qword [rsi + rcx*VRB_QWORD_SCALE]    ; inv_freq[i] = 2^(-i/qtr) * log2(theta), binary64
     fstp st0                                  ; pop leftover exponent n, fscale does NOT pop, x87 stack balanced
     inc ecx                                   ; i++
     cmp ecx, r11d                             ; loop bound checked at the bottom
     jl .vrb_invf_loop                         ; more positions: next i
.vrb_invf_done:
; Hoist loop-invariant VCFG values into the registers freed by the Q-preload:
; these would otherwise be RIP-relative loads on every inner iteration
     mov r13d, dword [rel g_vit_cfg + VCFG_ROPE_HALF] ; rope_half, 32 : inner loop bound + dup offset, multiple of 8 is the import's head_dim contract
     mov ebx, dword [rel g_vit_cfg + VCFG_ROPE_QTR] ; rope_qtr, 16 : row/col angle split
     xor r12d, r12d                            ; p = 0, patch counter in non-volatile r12, 64-bit to match vrb_n
     mov r14, vrb_cos                          ; r14 = running cos pointer
     mov r15, vrb_sin                          ; r15 = running sin pointer
     mov r9, qword [rel g_vit_cfg + VCFG_ROPE_HALF_F4] ; compact row stride: only rope_half floats are consumed
     xor r11d, r11d                            ; in_col cascade toggle, 0/1, register, not memory RMW
; x87 temp frame already allocated at invf loop entry, merged
.vrb_patch_loop:
; The patch walk writes the first occurrence of each row and column as
; an anchor in the output table. Later patches copy their exact FP32 bits.

%macro VRB_INNER_BODY 2                        ; %1 = inv_freq base; %2 = row/col integer scratch offset
     fld qword [%1 + rcx*VRB_QWORD_SCALE]      ; cache frequency (binary64) -> x87 extended
     fimul dword [rsp + %2]                    ; multiply by exact signed int32 coordinate
     fsincos                                   ; cos in st0, sin in st1
     fstp dword [r14 + rcx*VK_F4]              ; FP32 cos, output format unchanged
     fstp dword [r15 + rcx*VK_F4]              ; FP32 sin
%endmacro
; r9 = compact record stride. For row reuse, walk back 4*bcol+in_col
; records to the first patch of this row in the current row band.
     mov eax, vrb_bcol                         ; block column index
     lea eax, [r11 + rax*VRB_BLOCK_RECORDS]    ; distance to the row anchor in records
     test eax, eax                             ; is this patch the row anchor?
     jz .vrb_calc_row                          ; calculate this row once
     imul rax, r9                              ; byte distance to the anchor
     neg rax                                   ; signed displacement backwards
     xor ecx, ecx                              ; row component starts at zero
     mov edx, ebx                              ; row component ends at rope_qtr
     call .vrb_copy_component                  ; read the exact FP32 bits from the row anchor
     jmp .vrb_row_done                         ; proceed to the column component
.vrb_calc_row:
     mov eax, vrb_brow                         ; row-band index
     shl eax, 1                                ; two patch rows per band
     add eax, vrb_in_row                       ; exact row coordinate
     mov [rsp + VRB_ROW_SLOT], eax             ; x87 integer operand
     xor ecx, ecx                              ; start of the row component
.vrb_row_loop:
     VRB_INNER_BODY rsi, VRB_ROW_SLOT          ; invf[j] * exact row
     inc ecx                                   ; i++
     cmp ecx, ebx                              ; i < rope_qtr?, bound hoisted out of the patch loop
     jl .vrb_row_loop                          ; next half-dim; trip count and divisibility are the import's head_dim contract
.vrb_row_done:
; Column anchor is the same column in the first row band with in_row=0.
; Backwards distance in records = 2*(brow*gw + in_row).
     mov eax, vrb_brow                         ; row-band index
     mov edx, vrb_gw                           ; patches across a row
     imul rax, rdx                             ; number of columns across preceding bands
     mov edx, vrb_in_row                       ; second row inside a block adds two records
     add rax, rdx                              ; half the distance in records
     jz .vrb_calc_col                          ; calculate this column once
     add rax, rax                              ; full backwards distance in records
     imul rax, r9                              ; convert to bytes
     neg rax                                   ; backwards displacement
     mov ecx, ebx                              ; column component starts at rope_qtr
     mov edx, r13d                             ; column component ends at rope_half
     call .vrb_copy_component                  ; read the exact FP32 bits from the column anchor
     jmp .vrb_col_done                         ; proceed to the next patch
.vrb_calc_col:
     mov eax, vrb_bcol                         ; block column index
     lea eax, [r11 + rax*VRB_COL_SCALE]        ; exact column coordinate
     mov [rsp + VRB_COL_SLOT], eax             ; x87 integer operand
     mov ecx, ebx                              ; column component starts at rope_qtr
.vrb_col_loop:
     VRB_INNER_BODY r10, VRB_COL_SLOT          ; invf[j-qtr] * exact col
     inc ecx                                   ; i++
     cmp ecx, r13d                             ; i < rope_half?, bound hoisted out of the patch loop
     jl .vrb_col_loop                          ; next half-dim, then advance the compact table row
.vrb_col_done:
; No second-half duplication: vit_rope_apply uses only the first rope_half floats.
     add r14, r9                               ; cos pointer += rope_half*4 bytes
     add r15, r9                               ; sin pointer += rope_half*4 bytes
     inc r12                                   ; p++, non-volatile 64-bit counter
; Cascade advance: in_col fastest, then in_row, then bcol, then brow
     xor r11d, 1                               ; toggle in_col, register; ZF set when it wrapped 1->0
     jnz .vrb_casc_done                        ; 0->1: only in_col changed
     xor dword vrb_in_row, 1                   ; in_col wrapped: toggle in_row, ZF on wrap
     jnz .vrb_casc_done                        ; 0->1: nothing else changes
     inc dword vrb_bcol                        ; in_row wrapped: advance bcol
     cmp dword vrb_bcol, edi                   ; bcol < blocks_w?, blocks_w hoisted into edi at :116
     jb .vrb_casc_done                         ; yes, done
     mov dword vrb_bcol, 0                     ; bcol wrapped: reset
     inc dword vrb_brow                        ; advance brow
.vrb_casc_done:
     cmp r12, vrb_n                            ; bottom-test, eliminate the loop-head branch
     jl .vrb_patch_loop                        ; more patches: next
.vrb_frames_out:
     add rsp, 16                               ; free the merged x87 temp frame, sub'd at invf loop entry
; publish resolution-memo cache keys AFTER successful computation, not before
     mov eax, vrb_gh                           ; load grid_h from stack slot
     mov [rel g_cached_rb_gh], eax             ; cache grid_h
     mov eax, vrb_gw                           ; load grid_w from stack slot
     mov [rel g_cached_rb_gw], eax             ; cache grid_w
     mov rax, vrb_cos                          ; load cos pointer from stack slot
     mov [rel g_vrb_last_cos], rax             ; cache cos buffer identity
     mov rax, vrb_sin                          ; load sin pointer from stack slot
     mov [rel g_vrb_last_sin], rax             ; cache sin buffer identity
.vrb_done:
     SF_END rbx, rdi, rsi, r12, r13, r14, r15  ; restore non-volatiles, all written in body and return
.vrb_copy_component:
; Private stack-free leaf, no outbound calls. RAX = negative byte distance
; to the anchor. ECX/EDX = first and one-past-last float index. Both bounds
; are multiples of four under the import's head_dim contract.
     lea r8, [r14 + rax]                       ; anchor cosine base
     add rax, r15                              ; anchor sine base
.vrb_copy_next:
     vmovups xmm0, [r8 + rcx*VK_F4]            ; copy four exact cosine values
     vmovups [r14 + rcx*VK_F4], xmm0           ; write current cosine component
     vmovups xmm0, [rax + rcx*VK_F4]           ; copy four exact sine values
     vmovups [r15 + rcx*VK_F4], xmm0           ; write current sine component
     add ecx, 4                                ; next group of four
     cmp ecx, edx                              ; reached the exclusive component bound?
     jb .vrb_copy_next                         ; copy the rest
     ret                                       ; current patch component is ready
%unmacro VRB_INNER_BODY 2                      ; file-local macro, removed so a later include cannot redefine it silently
%undef VRB_INVF_MAX                            ; file-local constant, used only above, same reason
%undef VRB_X87_QTR                             ; x87 frame slot offset, file-local
%undef VRB_X87_I                               ; x87 frame slot offset, file-local
%undef VRB_ROW_SLOT                            ; x87 frame slot offset, file-local
%undef VRB_COL_SLOT                            ; x87 frame slot offset, file-local
%undef VRB_QWORD_SCALE                         ; inv_freq cache stride, file-local
; SF slot names of this file's one function, each carries the _raw alias too
%undef vrb_cos
%undef vrb_cos_raw
%undef vrb_sin
%undef vrb_sin_raw
%undef vrb_gh
%undef vrb_gh_raw
%undef vrb_gw
%undef vrb_gw_raw
%undef vrb_n
%undef vrb_n_raw
%undef vrb_in_row
%undef vrb_in_row_raw
%undef vrb_bcol
%undef vrb_bcol_raw
%undef vrb_brow
%undef vrb_brow_raw
