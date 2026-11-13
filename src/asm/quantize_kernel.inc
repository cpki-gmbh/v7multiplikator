; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; Constants, loaded from .rodata at kernel entry
%define QK_FLOAT4_SCALE 4                      ; scale factor for float pointer arithmetic, sizeof,float
%define QK_GS_BYTES 256                        ; GS * sizeof,float = 64 * 4 = 256 bytes per group
; Stack layout offsets for v7s_quantize_chunk, at ENTRY, before the sub rsp,48 --
; retaddr 8 + 4 register-arg homes 32 = 40, which is QK_GROUP_START_OFF below
%define QK_GROUP_START_OFF 40                  ; offset of group_start arg on stack, 5th arg
%define QK_XMM13_SAVE_OFF 0                    ; xmm13 low-128 save slot, at RSP after sub rsp,48, Win64 only preserves XMM lows
%define QK_XMM14_SAVE_OFF 16                   ; xmm14 low-128 save slot
%define QK_XMM6_SAVE_OFF 32                    ; xmm6 low-128 save slot
; Float buffer offsets, bytes for 8 YMM loads per GS group, 8 floats x 32 bytes per YMM
%define QK_FLT_1 32                            ; byte offset for YMM float block 1, elements 8..15
%define QK_FLT_2 64                            ; byte offset for YMM float block 2, elements 16..23
%define QK_FLT_3 96                            ; byte offset for YMM float block 3, elements 24..31
%define QK_FLT_4 128                           ; byte offset for YMM float block 4, elements 32..39
%define QK_FLT_5 160                           ; byte offset for YMM float block 5, elements 40..47
%define QK_FLT_6 192                           ; byte offset for YMM float block 6, elements 48..55
%define QK_FLT_7 224                           ; byte offset for YMM float block 7, elements 56..63
; Int8 buffer offset for the second 32-float store per GS group
%define QK_INT8_2 32                           ; byte offset for the second 32-wide int8 store, elements 32..63
     section .rdata align=32
     align 32
qk_abs_mask:
     dd 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF ; absolute value mask, clear sign bit
     dd 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF ; high 128 bits of mask
qk_f127:
     dd 127.0                                  ; only scalar reads and broadcasts use this constant
qk_fmin127:
     dd -127.0                                 ; scalar tail lower clamp
     align 32
     qk_min127_b8: times 32 db -127            ; 32 int8 lanes of -127, 0x81 each, widened for the ymm vpmaxsb, the -127 floor AFTER vpacksswb
     align 32
qk_perm_0415:
     dd 0, 4, 1, 5, 2, 6, 3, 7                 ; vpermd lane fix-up: unscramble the 4-vector double pack into linear byte order
qk_fone:
     dd 1.0                                    ; 1.0f, kt_quantize_c_ref's exact reciprocal numerator; vdivss reads the low dword only, so one dd is enough
scalar_epsilon:
     dd 1.4928778e-36                          ; 127 x 2^-126: the group-max floor chosen so scale = eps/127 lands EXACTLY on the smallest normal float, never "correct" this to plain 2^-126, that would make the all-zero-group scale a subnormal and break the DAZ-invariance argument below
     section .text
     align 64
     global v7s_quantize_chunk                 ; export quantize kernel function
v7s_quantize_chunk:
     mov r10d, [rsp + QK_GROUP_START_OFF]      ; r10d = group_start, read BEFORE stack alloc
     sub rsp, 48                               ; allocate stack for xmm13/xmm14/xmm6 low-128 saves, Win64 preserves only the low halves
     vmovdqu [rsp + QK_XMM13_SAVE_OFF], xmm13  ; save xmm13 low 128 bits, callee-saved half
     vmovdqu [rsp + QK_XMM14_SAVE_OFF], xmm14  ; save xmm14 low 128 bits
     vmovdqu [rsp + QK_XMM6_SAVE_OFF], xmm6    ; save xmm6 low 128 bits, callee-saved half
     vmovdqa ymm6, [qk_perm_0415]              ; ymm6 = lane fix-up index, preloaded once, reused by every QUANTIZE_32 call
     lea r8, [r8 + r10*QK_FLOAT4_SCALE]        ; r8 = out_s + group_start*4, point at first scale slot
; Load constants, qk_f127 into volatile ymm5; abs mask into callee-saved ymm13;
; qk_min127_b8 into ymm14, preloaded once, reused across all groups
     vbroadcastss ymm5, [qk_f127]              ; ymm5 = 127.0f, all lanes, stable across loop, never overwritten
     vmovdqa ymm13, [qk_abs_mask]              ; ymm13 = abs mask, callee-saved, preloaded once, no per-group reload
     vmovdqa ymm14, [qk_min127_b8]             ; ymm14 = 32 int8 lanes of -127, preloaded clamp floor, reused by all QUANTIZE_32 calls
; Pre-loop guard: a full group must remain before the body runs, the first iteration is guarded here, subsequent ones by the bottom test at the loop tail
     cmp r9d, GS                               ; remaining >= 64?
     jb .qk_tail                               ; no, handle tail remaining
.qk_group_loop:
     vandps ymm0, ymm13, [rdx]                 ; ymm0 = |x[0..7]|
     vandps ymm1, ymm13, [rdx + QK_FLT_1]      ; ymm1 = |x[8..15]|
     vmaxps ymm0, ymm0, ymm1                   ; ymm0 = max,0..15
     vandps ymm1, ymm13, [rdx + QK_FLT_2]      ; ymm1 = |x[16..23]|
     vandps ymm2, ymm13, [rdx + QK_FLT_3]      ; ymm2 = |x[24..31]|
     vmaxps ymm1, ymm1, ymm2                   ; ymm1 = max,16..31
     vmaxps ymm0, ymm0, ymm1                   ; ymm0 = max,0..31
     vandps ymm1, ymm13, [rdx + QK_FLT_4]      ; ymm1 = |x[32..39]|
     vandps ymm2, ymm13, [rdx + QK_FLT_5]      ; ymm2 = |x[40..47]|
     vmaxps ymm1, ymm1, ymm2                   ; ymm1 = max,32..47
     vandps ymm2, ymm13, [rdx + QK_FLT_6]      ; ymm2 = |x[48..55]|
     vandps ymm3, ymm13, [rdx + QK_FLT_7]      ; ymm3 = |x[56..63]|
     vmaxps ymm2, ymm2, ymm3                   ; ymm2 = max,48..63
     vmaxps ymm1, ymm1, ymm2                   ; ymm1 = max,32..63
; Level 3: final combine -> ymm3
     vmaxps ymm3, ymm0, ymm1                   ; ymm3 lane i = max of lane i's congruent class {i,8+i,...,56+i}, 8 of the 64 elements, lanes differ
     vextractf128 xmm0, ymm3, 1                ; xmm0 = upper 4 class maxes
     vmaxps xmm3, xmm3, xmm0                   ; max across lower+upper
     vmovhlps xmm0, xmm3, xmm3                 ; xmm0 = [m2, m3, m2, m3], hi half into lo, no immediate
     vmaxps xmm3, xmm3, xmm0                   ; max of pairs -> [n0, n1, ...]
     vmovshdup xmm0, xmm3                      ; xmm0 = [n1, n1, ...], lane 0 = n1, no immediate
     vmaxps xmm3, xmm3, xmm0                   ; scalar wmax in xmm3[0]
     vmaxss xmm3, xmm3, [scalar_epsilon]       ; xmm3 = max,wmax, eps, DAZ-invariant subnormal guard
     vmovss xmm2, [qk_fone]                    ; reciprocal numerator into dead xmm2, issues before the slow divide
     vdivss xmm0, xmm3, xmm5                   ; xmm0 = scale = wmax / 127.0f, unchanged
     vmovss [r8], xmm0                         ; store the same rounded scale
     vdivss xmm2, xmm2, xmm0                   ; same exact reciprocal division, unchanged
     vbroadcastss ymm2, xmm2                   ; ymm2 = reciprocal scale, broadcast to all lanes

%macro QUANTIZE_32 5
     vmulps ymm0, ymm2, [rdx + %1]             ; x[ 0.. 7] * recip
     vmulps ymm1, ymm2, [rdx + %2]             ; x[ 8..15] * recip
     vcvtps2dq ymm0, ymm0                      ; round+convert to int32
     vmulps ymm3, ymm2, [rdx + %3]             ; x[16..23] * recip
     vcvtps2dq ymm1, ymm1                      ; round+convert to int32
     vmulps ymm4, ymm2, [rdx + %4]             ; x[24..31] * recip
     vcvtps2dq ymm3, ymm3                      ; round+convert to int32
     vcvtps2dq ymm4, ymm4                      ; round+convert to int32
     vpackssdw ymm0, ymm0, ymm1                ; 16 int32 -> 16 int16, saturating
     vpackssdw ymm3, ymm3, ymm4                ; next 16 int32 -> 16 int16
     vpacksswb ymm0, ymm0, ymm3                ; 32 int16 -> 32 int8, saturating
     vpmaxsb ymm0, ymm0, ymm14                 ; clamp all 32 byte lanes >= -127 in one op
     vpermd ymm0, ymm6, ymm0                   ; restore linear byte order, permutation vector preloaded once
     vmovdqu [rcx + %5], ymm0                  ; store 32 int8 values
%endmacro
; Clamp constant from preloaded ymm14, no memory traffic per call
; --- Iter 0: floats 0..31 -> int8 0..31 ---
     QUANTIZE_32 0, QK_FLT_1, QK_FLT_2, QK_FLT_3, 0 ; quantize first 32 floats, store at offset 0
; --- Iter 1: floats 32..63 -> int8 32..63 ---
     QUANTIZE_32 QK_FLT_4, QK_FLT_5, QK_FLT_6, QK_FLT_7, QK_INT8_2 ; quantize floats 32..63, store at offset 32
; Advance pointers for next group
     add rcx, GS                               ; out_q += 64
     add rdx, QK_GS_BYTES                      ; in_x += 256 bytes, 64 floats
     sub r9d, GS                               ; n_slice -= 64
     add r8, QK_FLOAT4_SCALE                   ; out_s += 4 bytes, next scale slot
     cmp r9d, GS                               ; another full group left?
     jae .qk_group_loop                        ; yes, next group, bottom-tested, no tail jmp
.qk_tail:
     test r9d, r9d                             ; any tail?
     jz .qk_done                               ; no, done
; Handle partial group as scalar loop for correctness
     vxorps xmm3, xmm3, xmm3                   ; wmax = 0
     xor r11d, r11d                            ; i = 0, use r11d, volatile reg, preserves rcx as out_q; r9d>0 guaranteed, the jz above exited the 0 case, so no entry guard needed
.qk_tail_max_loop:
     vmovss xmm0, [rdx + r11*QK_FLOAT4_SCALE]  ; load single float x[i], scalar, no OOB
     vandps xmm0, xmm0, xmm13                  ; |x[i]| with scalar abs mask, low lane of persistent ymm13
     vmaxss xmm3, xmm0, xmm3                   ; update wmax. INCOMING is src1: VMAXSS returns src2 on either NaN, so a NaN element discards itself, vandps cannot manufacture one and the accumulator stays non-NaN. This operand order is the OPPOSITE of the pass-1 vmaxps tree, there a NaN reaching src2 of the final combine sentinels wmax while one discarded as src1 does not, MAXPS is position-dependent, i.e. the vector path has no NaN discipline; deliberately so here: with the accumulator in src1 a NaN in the LAST element would survive to the DAZ guard below, map wmax to scalar_epsilon, and rewrite the scale of the whole tail group
     inc r11d                                  ; i++
     cmp r11d, r9d                             ; i < n_remaining?
     jb .qk_tail_max_loop                      ; continue max-find loop
.qk_tail_max_done:
; xmm3[0] = wmax for tail group
; DAZ-invariant subnormal guard first, see main-path comment above
     vmaxss xmm3, xmm3, [scalar_epsilon]       ; xmm3 = max,wmax, eps, DAZ-invariant subnormal guard
     vmovss xmm2, [qk_fone]                    ; independent numerator load issues before the divide
; Compute scale = wmax / 127.0f via single correctly-rounded division, matches C ref
     vdivss xmm0, xmm3, xmm5                   ; xmm0 = tail scale = wmax / 127.0f, unchanged
     vmovss [r8], xmm0                         ; store the same exact tail scale
     vdivss xmm3, xmm2, xmm0                   ; same tail reciprocal division
; Quantize tail elements with scalar loop
     xor r11d, r11d                            ; i = 0, use r11d, volatile reg; r9d>0 guaranteed here too
.qk_tail_quant_loop:
     vmovss xmm0, [rdx + r11*QK_FLOAT4_SCALE]  ; load float
     vmulss xmm0, xmm0, xmm3                   ; x[i] * recip, faster and numerically consistent with main path
     vmaxss xmm0, xmm0, [qk_fmin127]           ; clamp float to >= -127.0 FIRST, memory operand: qk_fmin127 is needed only here, so a callee-saved preload costs every call to serve one; NaN -> src2 = -127.0, matching the main path's INT_MIN->-127
     vminss xmm0, xmm0, xmm5                   ; then clamp to <= 127.0, xmm5 low lane = 127.0f; min,-127.0, 127.0 = -127.0 ==> NaN quantizes to -127
     vcvtss2si eax, xmm0                       ; round+convert to int32, guaranteed in [-127,127] by float clamps
     mov byte [rcx + r11], al                  ; store int8 at out_q[i]
     inc r11d                                  ; advance loop counter
     cmp r11d, r9d                             ; i < n_remaining?
     jb .qk_tail_quant_loop                    ; continue tail quant loop
.qk_done:
     vmovdqu xmm13, [rsp + QK_XMM13_SAVE_OFF]  ; restore xmm13 low 128 bits
     vmovdqu xmm14, [rsp + QK_XMM14_SAVE_OFF]  ; restore xmm14 low 128 bits
     vmovdqu xmm6, [rsp + QK_XMM6_SAVE_OFF]    ; restore xmm6 low 128 bits
     vzeroupper                                ; clear dirty YMM upper bits AFTER restores, effective, nothing re-dirties them
     add rsp, 48                               ; free stack
     ret                                       ; return to caller
; %undef cleanup: file-local defines leave scope here, house convention --
%unmacro QUANTIZE_32 5                         ; file-local quantize macro, all 2 call sites are above
%undef QK_INT8_2                               ; byte offset for XMM int8 block 2, elements 32..47
%undef QK_FLT_7                                ; byte offset for YMM float block 7, elements 56..63
%undef QK_FLT_6                                ; byte offset for YMM float block 6, elements 48..55
%undef QK_FLT_5                                ; byte offset for YMM float block 5, elements 40..47
%undef QK_FLT_4                                ; byte offset for YMM float block 4, elements 32..39
%undef QK_FLT_3                                ; byte offset for YMM float block 3, elements 24..31
%undef QK_FLT_2                                ; byte offset for YMM float block 2, elements 16..23
%undef QK_FLT_1                                ; byte offset for YMM float block 1, elements 8..15
%undef QK_XMM14_SAVE_OFF                       ; xmm14 low-128 save slot
%undef QK_XMM6_SAVE_OFF                        ; xmm6 low-128 save slot
%undef QK_XMM13_SAVE_OFF                       ; xmm13 low-128 save slot
%undef QK_GROUP_START_OFF                      ; offset of group_start arg on stack, 5th arg
%undef QK_GS_BYTES                             ; GS * sizeof,float = 64 * 4 = 256 bytes per group
%undef QK_FLOAT4_SCALE                         ; scale factor for float pointer arithmetic, sizeof,float
