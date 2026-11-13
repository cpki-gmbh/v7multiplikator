; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; Numeric constants for struct offsets and array sizes, NUMOFFSET compliance
%define F32K_ACT_OFF0 0                        ; Activation byte offset for vec0, act[0..7] within block
%define F32K_ACT_OFF8 32                       ; Activation byte offset for vec1, act[8..15]
%define F32K_ACT_OFF16 64                      ; Activation byte offset for vec2, act[16..23]
%define F32K_ACT_OFF24 96                      ; Activation byte offset for vec3, act[24..31]
%define F32K_WT_OFF0 0                         ; Weight byte offset for vec0, wt[0..7] within 32-int8 block
%define F32K_WT_OFF8 8                         ; Weight byte offset for vec1, wt[8..15]
%define F32K_WT_OFF16 16                       ; Weight byte offset for vec2, wt[16..23]
%define F32K_WT_OFF24 24                       ; Weight byte offset for vec3, wt[24..31]
%define F32K_FLOAT_BYTES 4                     ; Bytes per float element, SIB scale, offset conversions
%define F32K_ACT_BLOCK1 128                    ; Activation block 1 byte offset, 128..255
%define F32K_ACT_BLOCK2 256                    ; Activation block 2 byte offset, 256..383
%define F32K_ACT_BLOCK3 384                    ; Activation block 3 byte offset, 384..511
%define F32K_PREFETCH_ACT_DIST 1280            ; Activation prefetch distance, bytes
%define F32K_PREFETCH_WT_DIST 320              ; Weight prefetch distance, bytes
%define F32K_ROWS_PER_ITER 2                   ; Number of rows advanced per paired-row iteration

%macro FMA_32BLK 7
     vpmovsxbd ymm10, [%2 + rcx + F32K_WT_OFF0 + (%7 / F32K_FLOAT_BYTES)] ; expand wt[0..7]
     vpmovsxbd ymm11, [%2 + rcx + F32K_WT_OFF8 + (%7 / F32K_FLOAT_BYTES)] ; expand wt[8..15], issue early
     vcvtdq2ps ymm10, ymm10                    ; convert first 8 weights to float
     vcvtdq2ps ymm11, ymm11                    ; convert next 8 independently
     vmulps ymm10, ymm10, %1                   ; same dequantization product for 0..7
     vmulps ymm11, ymm11, %1                   ; independent product, issued before either FMA
     vfmadd231ps %3, ymm12, ymm10              ; same FMA for accumulator 0
     vfmadd231ps %4, ymm13, ymm11              ; same FMA for accumulator 1
     vpmovsxbd ymm10, [%2 + rcx + F32K_WT_OFF16 + (%7 / F32K_FLOAT_BYTES)] ; expand wt[16..23]
     vpmovsxbd ymm11, [%2 + rcx + F32K_WT_OFF24 + (%7 / F32K_FLOAT_BYTES)] ; expand wt[24..31], issue early
     vcvtdq2ps ymm10, ymm10                    ; convert first 8 weights to float
     vcvtdq2ps ymm11, ymm11                    ; convert next 8 independently
     vmulps ymm10, ymm10, %1                   ; same dequantization product for 16..23
     vmulps ymm11, ymm11, %1                   ; independent product, issued before either FMA
     vfmadd231ps %5, ymm14, ymm10              ; same FMA for accumulator 2
     vfmadd231ps %6, ymm15, ymm11              ; same FMA for accumulator 3
%endmacro

%macro F32K_PAIR_8 4                           ; weight byte offset, activation ymm, accum0, accum1
     vpmovsxbd ymm10, [r10 + rcx + %1]         ; row0 wt[0..7]
     vpmovsxbd ymm11, [rsi + rcx + %1]         ; row1 wt[0..7]
     vcvtdq2ps ymm10, ymm10                    ; convert row0 weights to float
     vcvtdq2ps ymm11, ymm11                    ; convert row1 weights to float
     vmulps ymm10, ymm10, ymm8                 ; row0 scale
     vmulps ymm11, ymm11, ymm9                 ; row1 scale
     vfmadd231ps %3, %2, ymm10                 ; row0 accumulator FMA
     vfmadd231ps %4, %2, ymm11                 ; row1 accumulator FMA
%endmacro

%macro F32K_PAIR_32 1                          ; activation offset in FLOAT bytes
     F32K_PAIR_8 F32K_WT_OFF0 + (%1 / F32K_FLOAT_BYTES), ymm12, ymm0, ymm4 ; lanes 0..7, rows interleaved
     F32K_PAIR_8 F32K_WT_OFF8 + (%1 / F32K_FLOAT_BYTES), ymm13, ymm1, ymm5 ; lanes 8..15, rows interleaved
     F32K_PAIR_8 F32K_WT_OFF16 + (%1 / F32K_FLOAT_BYTES), ymm14, ymm2, ymm6 ; lanes 16..23, rows interleaved
     F32K_PAIR_8 F32K_WT_OFF24 + (%1 / F32K_FLOAT_BYTES), ymm15, ymm3, ymm7 ; lanes 24..31, rows interleaved
%endmacro

%macro ACT_QUAD 1
     vmovups ymm12, [r13 + rcx*F32K_FLOAT_BYTES + %1 + F32K_ACT_OFF0] ; Load raw act[0..7], unscaled
     vmovups ymm13, [r13 + rcx*F32K_FLOAT_BYTES + %1 + F32K_ACT_OFF8] ; Load raw act[8..15], unscaled
     vmovups ymm14, [r13 + rcx*F32K_FLOAT_BYTES + %1 + F32K_ACT_OFF16] ; Load raw act[16..23], unscaled
     vmovups ymm15, [r13 + rcx*F32K_FLOAT_BYTES + %1 + F32K_ACT_OFF24] ; Load raw act[24..31], unscaled
%endmacro
%define F32K_SCALES_OFF 8                      ; tensor scales field, QuantizedTensor.s
%define F32K_FEAT_DIM_OFF 48                   ; Feature dimension d, 5th stack arg
%define F32K_NEXT_SCALE 4                      ; float offset to next group scale, 4 bytes
%define F32K_128_SHIFT 7                       ; log2,128 : 128-element blocks per main-loop iteration
     section .text
     global v7s_f32_chunk                      ; Export f32 chunk kernel
v7s_f32_chunk:
     SF_START v7s_f32_chunk, rbx, rsi, rdi, r12, r13 ; Begin f32 chunk stack frame
     SF_XMM f32k_xmm6                          ; xmm6 spill slot, upper 128 bits are volatile per Win64 ABI
     SF_XMM f32k_xmm7                          ; xmm7 spill slot
     SF_XMM f32k_xmm8                          ; xmm8 spill slot
     SF_XMM f32k_xmm9                          ; xmm9 spill slot
     SF_XMM f32k_xmm10                         ; xmm10 spill slot
     SF_XMM f32k_xmm11                         ; xmm11 spill slot
     SF_XMM f32k_xmm12                         ; xmm12 spill slot
     SF_XMM f32k_xmm13                         ; xmm13 spill slot
     SF_XMM f32k_xmm14                         ; xmm14 spill slot
     SF_XMM f32k_xmm15                         ; xmm15 spill slot
     SF_COMMIT                                 ; Commit f32 chunk stack frame
     vmovdqu f32k_xmm6, xmm6                   ; Save callee-saved xmm6, lower 128 bits only
     vmovdqu f32k_xmm7, xmm7                   ; Save callee-saved xmm7
     vmovdqu f32k_xmm8, xmm8                   ; Save callee-saved xmm8
     vmovdqu f32k_xmm9, xmm9                   ; Save callee-saved xmm9
     vmovdqu f32k_xmm10, xmm10                 ; Save callee-saved xmm10
     vmovdqu f32k_xmm11, xmm11                 ; Save callee-saved xmm11
     vmovdqu f32k_xmm12, xmm12                 ; Save callee-saved xmm12
     vmovdqu f32k_xmm13, xmm13                 ; Save callee-saved xmm13
     vmovdqu f32k_xmm14, xmm14                 ; Save callee-saved xmm14
     vmovdqu f32k_xmm15, xmm15                 ; Save callee-saved xmm15
; --- Load arguments ---
     mov r12, rcx                              ; r12 = output pointer xout
     mov r13, [rdx]                            ; r13 = activations base pointer, x
     mov r10, [r8]                             ; r10 = wq, int8 weight base; row0 weight ptr from here
     mov r11, [r8+F32K_SCALES_OFF]             ; r11 = ws, float scale base; row0 scale ptr from here
     mov r8d, dword [rbp + F32K_FEAT_DIM_OFF]  ; r8d = d, feature dimension, from stack
     dec r8d                                   ; r8d = d-1: pair check "row+1 >= d" == "row >= d-1"; last check "row >= d" == "row > d-1"
; JOB_LOAD_INPUT_ELEMENTS, 32-bit writes zero bits 63:32 architecturally
; --- 64-aligned vector limit: n is 64-aligned by the init contract, so r9d IS the limit ---
; --- Scale row stride:, n/64 *4 = n/16, since n%64==0 by the init contract ---
; Initialize running weight and scale pointers for row 0
     mov edx, r9d                              ; edx = n, rdx dead: incoming rdx consumed at the activation load
     shr edx, 4                                ; edx = scale row stride = n/16 bytes, u32 exact: n is u32 and n%64==0
     xor ebx, ebx                              ; ebx = row index, start at 0
; ============================================================================
; Paired outer loop: 2 rows at a time
; ============================================================================
     align 32
.outer_f32_paired:
     cmp ebx, r8d                              ; row >= d-1?, no second row, r8d pre-decremented
     jae .outer_f32_last                       ; If no second row, handle single row
; Row1 weight pointer: row0 + n, byte offset for int8 weights
     lea rsi, [r10 + r9]                       ; rsi = row1 weight ptr = row0 + n
; Row1 scale pointer: row0 scale ptr + stride, r11 carried in incrementally from the
; previous pair / the row-0 init, no per-pair multiply is needed; rdx = stride
     lea rdi, [r11 + rdx]                      ; rdi = row1 scale ptr = row0 + stride
     vzeroall                                  ; Zero row0/row1 accumulators ymm0-7 in 1 instr, ymm8-15 fully rewritten before any read
     xor ecx, ecx                              ; ecx = element offset, start at 0
     mov eax, r9d                              ; eax = n, row width in elements
     shr eax, F32K_128_SHIFT                   ; eax = floor,n/128 = 128-block count; an n%128==64 remainder goes to .tail64_paired
     jz .tail_or_done_paired                   ; n < 128: zero blocks, go straight to the tail/done check
; ============================================================================
; 128-element main loop: 2 groups x 64 elements, 2 rows each
; ============================================================================
     align 64                                  ; align the hot loop HEAD, back-edge target ; body entered by fall-through, count > 0 proven above
.loop128_paired_body:
     ds prefetcht1 [r13 + rcx*F32K_FLOAT_BYTES + F32K_PREFETCH_ACT_DIST] ; Prefetch activations ahead 1280 bytes
     PREFETCH_SLOT 8                           ; Slot 8: f32 activation prefetch
; ===== GROUP 0: elements ecx..ecx+63 =====
     vbroadcastss ymm8, [r11]                  ; ymm8 = row0 scale s0, broadcast
     vbroadcastss ymm9, [rdi]                  ; ymm9 = row1 scale s0, broadcast, rdi live
; --- Group 0, sub-block a: elements ecx..ecx+31, load act ONCE for both rows ---
     ACT_QUAD 0                                ; raw act[ecx+0..31], shared by both rows
     ds prefetcht1 [r10 + rcx + F32K_PREFETCH_WT_DIST] ; Prefetch row0 weights ahead 320 bytes
     PREFETCH_SLOT 9                           ; Slot 9: f32 weight prefetch
     F32K_PAIR_8 F32K_WT_OFF0 + (0 / F32K_FLOAT_BYTES), ymm12, ymm0, ymm4 ; lanes 0..7, rows interleaved
     F32K_PAIR_8 F32K_WT_OFF8 + (0 / F32K_FLOAT_BYTES), ymm13, ymm1, ymm5 ; lanes 8..15, rows interleaved
     ds prefetcht1 [rsi + rcx + F32K_PREFETCH_WT_DIST] ; Prefetch row1 weights ahead 320 bytes, issued halfway into the first paired dequant block
     PREFETCH_SLOT 10                          ; Slot 10: f32 row1 weight prefetch, patchable dist/type
     F32K_PAIR_8 F32K_WT_OFF16 + (0 / F32K_FLOAT_BYTES), ymm14, ymm2, ymm6 ; lanes 16..23, rows interleaved
     F32K_PAIR_8 F32K_WT_OFF24 + (0 / F32K_FLOAT_BYTES), ymm15, ymm3, ymm7 ; lanes 24..31, rows interleaved
; --- Group 0, sub-block b: elements ecx+32..ecx+63, load act ONCE for both rows ---
     ACT_QUAD F32K_ACT_BLOCK1                  ; raw act[ecx+32..63], shared by both rows
     F32K_PAIR_32 F32K_ACT_BLOCK1              ; Row0+Row1 offset 128, interleaved dequant
; ===== GROUP 1: elements ecx+64..ecx+127 =====
     vbroadcastss ymm8, [r11 + F32K_NEXT_SCALE] ; ymm8 = row0 scale s1, next group
     vbroadcastss ymm9, [rdi + F32K_NEXT_SCALE] ; ymm9 = row1 scale s1, next group, rdi live
; --- Group 1, sub-block a: elements ecx+64..ecx+95, load act ONCE for both rows ---
     ACT_QUAD F32K_ACT_BLOCK2                  ; raw act[ecx+64..95], shared by both rows
     F32K_PAIR_32 F32K_ACT_BLOCK2              ; Row0+Row1 offset 256, interleaved dequant
; --- Group 1, sub-block b: elements ecx+96..ecx+127, load act ONCE for both rows ---
     ACT_QUAD F32K_ACT_BLOCK3                  ; raw act[ecx+96..127], shared by both rows
     F32K_PAIR_32 F32K_ACT_BLOCK3              ; Row0+Row1 offset 384, interleaved dequant
; --- Advance pointers ---
     add r11, F32K_NEXT_SCALE*2                ; Advance row0 scale ptr by 2 groups, 8 bytes
     add rdi, F32K_NEXT_SCALE*2                ; Advance row1 scale ptr by 2 groups, 8 bytes
     sub ecx, -128                             ; Advance element offset by 128, imm8 encoding of the same +128
     dec eax                                   ; one fewer 128-block, eax held the exact block count
     jnz .loop128_paired_body                  ; more 128-blocks, next iteration
; ============================================================================
; Tail: single 64-block, remaining is exactly 64 since n % 64 == 0
; ============================================================================
     align 16
.tail_or_done_paired:
     cmp ecx, r9d                              ; ecx >= n?, loop termination check
     jb .tail64_paired                         ; No: remaining is exactly 64 -> out-of-line tail, SmolLM2 576/960 shapes
; ============================================================================
; Reduce and store paired rows, hot fall-through: all n%128==0 shapes
; ============================================================================
.reduce_paired:
     vaddps ymm0, ymm0, ymm1                   ; row0: acc0+acc1
     vaddps ymm4, ymm4, ymm5                   ; row1: acc0+acc1 (independent)
     vaddps ymm2, ymm2, ymm3                   ; row0: acc2+acc3
     vaddps ymm6, ymm6, ymm7                   ; row1: acc2+acc3 (independent)
     vaddps ymm0, ymm0, ymm2                   ; row0: (0+1)+(2+3)
     vaddps ymm4, ymm4, ymm6                   ; row1: same tree
     vextractf128 xmm12, ymm0, 1               ; row0: high 4 floats
     vextractf128 xmm13, ymm4, 1               ; row1: high 4 floats
     vaddps xmm0, xmm0, xmm12                  ; row0: low + high
     vaddps xmm4, xmm4, xmm13                  ; row1: low + high
     vmovshdup xmm12, xmm0                     ; row0: odd lanes
     vmovshdup xmm13, xmm4                     ; row1: odd lanes
     vaddps xmm0, xmm0, xmm12                  ; row0: pair reduction
     vaddps xmm4, xmm4, xmm13                  ; row1: pair reduction
     vmovhlps xmm12, xmm0, xmm0                ; row0: high qword
     vmovhlps xmm13, xmm4, xmm4                ; row1: high qword
     vaddss xmm0, xmm0, xmm12                  ; row0: scalar sum
     vaddss xmm4, xmm4, xmm13                  ; row1: scalar sum
     vunpcklps xmm0, xmm0, xmm4                ; pack [row0, row1] into the low 64 bits, lane 0 of each scalar
     vmovq [r12 + rbx*F32K_FLOAT_BYTES], xmm0  ; store rows 0+1 as one 64-bit write, same bytes at the same addresses as two vmovss stores
     add ebx, 2                                ; Advance row index by 2
; Next row0 weight ptr = current row0 + 2*n bytes, skip both processed rows
     lea r10, [r10 + r9*F32K_ROWS_PER_ITER]    ; r10 = next row0 weight ptr, skip 2 rows
     add r11, rdx                              ; scale pointers advance the second stride: r11 = row k+2 start
; loop+tail already delivered exactly 1 stride for row k
     jmp .outer_f32_paired                     ; Next pair of rows, r11 carried; rdi re-derived there
.tail64_paired:
     vbroadcastss ymm8, [r11]                  ; ymm8 = row0 scale broadcast
     vbroadcastss ymm9, [rdi]                  ; ymm9 = row1 scale broadcast, rdi live
; Sub-block a: elements offset 0..31, load act ONCE for both rows
     ACT_QUAD 0                                ; raw act[ecx+0..31], tail sub-block a
     F32K_PAIR_32 0                            ; Row0+Row1 offset 0, interleaved dequant
; Sub-block b: elements offset 32..63, load act ONCE for both rows
     ACT_QUAD F32K_ACT_BLOCK1                  ; raw act[ecx+32..63], tail sub-block b
     F32K_PAIR_32 F32K_ACT_BLOCK1              ; Row0+Row1 offset 128, interleaved dequant
     add r11, F32K_NEXT_SCALE                  ; count the tail's consumed group, 64 elements = 1 scale
     add ecx, 64                               ; Advance element offset by 64, ecx == n now
     jmp .reduce_paired                        ; rejoin the reduce, backward: the reduce must precede the out-of-line tail so the hot n%128==0 path falls through it
; ============================================================================
; Single-row fallback for odd d
; ============================================================================
     align 16
.outer_f32_last:
     cmp ebx, r8d                              ; All rows done?, r8d = d-1: done when row > d-1
     ja .done_f32                              ; If yes, exit, even d passes through here
     vzeroall                                  ; Zero row accumulator ymm0-3 in 1 instr, single-row path: ymm4-15 dead before any read
     xor ecx, ecx                              ; ecx = element offset, start at 0
.sfl_loop:
; n is a positive multiple of 64 on this path; process one full scale group per iteration
     vbroadcastss ymm8, [r11]                  ; one weight scale for these 64 elements
     ACT_QUAD 0                                ; raw activations 0..31 of this group
     FMA_32BLK ymm8, r10, ymm0, ymm1, ymm2, ymm3, 0 ; first 32
     ACT_QUAD F32K_ACT_BLOCK1                  ; raw activations 32..63 of same group
     FMA_32BLK ymm8, r10, ymm0, ymm1, ymm2, ymm3, F32K_ACT_BLOCK1 ; second 32
     add ecx, 64                               ; advance one complete weight-scale group
     add r11, F32K_NEXT_SCALE                  ; advance to next group scale
     cmp ecx, r9d                              ; n is a multiple of 64, no partial tail
     jb .sfl_loop                              ; repeat remaining groups
.sfl_store:
     vaddps ymm0, ymm0, ymm1                   ; Combine accumulators 0+1
     vaddps ymm2, ymm2, ymm3                   ; Combine accumulators 2+3
     vaddps ymm0, ymm0, ymm2                   ; Final sum
     REDUCE_YMM_TO_SCALAR_FAST xmm0, ymm0, xmm12, xmm13 ; Reduce row vector to scalar
     vmovss [r12 + rbx*F32K_FLOAT_BYTES], xmm0 ; Store row dot product
     inc ebx                                   ; Next row index, = d: ebx sequence 0,2,4..d-1 then d-1+1
     jmp .outer_f32_last                       ; -> loop-top check exits, row > d-1 ; odd d runs this block exactly once
; ============================================================================
; Epilogue
; ============================================================================
.done_f32:
     vmovdqu xmm6, f32k_xmm6                   ; Restore callee-saved xmm6, lower 128 bits
     vmovdqu xmm7, f32k_xmm7                   ; Restore callee-saved xmm7
     vmovdqu xmm8, f32k_xmm8                   ; Restore callee-saved xmm8
     vmovdqu xmm9, f32k_xmm9                   ; Restore callee-saved xmm9
     vmovdqu xmm10, f32k_xmm10                 ; Restore callee-saved xmm10
     vmovdqu xmm11, f32k_xmm11                 ; Restore callee-saved xmm11
     vmovdqu xmm12, f32k_xmm12                 ; Restore callee-saved xmm12
     vmovdqu xmm13, f32k_xmm13                 ; Restore callee-saved xmm13
     vmovdqu xmm14, f32k_xmm14                 ; Restore callee-saved xmm14
     vmovdqu xmm15, f32k_xmm15                 ; Restore callee-saved xmm15
     vzeroupper                                ; Clear YMM upper halves, volatile bits, zeroed after xmm restore
     SF_END rbx, rsi, rdi, r12, r13            ; Restore callee-saved regs and return
; --- Cleanup: F32K_* defines, FMA_32BLK and ACT_QUAD are file-local, no other file references them ---
%unmacro FMA_32BLK 7                           ; undefine the file-local 7-arg FMA block macro
%unmacro F32K_PAIR_8 4                         ; undefine the file-local interleaved pair macro
%unmacro F32K_PAIR_32 1                        ; undefine the file-local 32-lane pair macro
%unmacro ACT_QUAD 1                            ; undefine the file-local 1-arg activation-quad macro
%undef F32K_ACT_OFF0
%undef F32K_ACT_OFF8
%undef F32K_ACT_OFF16
%undef F32K_ACT_OFF24
%undef F32K_WT_OFF0
%undef F32K_WT_OFF8
%undef F32K_WT_OFF16
%undef F32K_WT_OFF24
%undef F32K_FLOAT_BYTES
%undef F32K_ACT_BLOCK1
%undef F32K_ACT_BLOCK2
%undef F32K_ACT_BLOCK3
%undef F32K_PREFETCH_ACT_DIST
%undef F32K_PREFETCH_WT_DIST
%undef F32K_ROWS_PER_ITER
%undef F32K_SCALES_OFF
%undef F32K_FEAT_DIM_OFF
%undef F32K_NEXT_SCALE
%undef F32K_128_SHIFT
