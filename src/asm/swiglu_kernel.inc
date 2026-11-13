; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .rdata align=32
     align 16
swk_sign_mask:
     dd 0x80000000, 0, 0, 0                    ; sign-bit mask for float negation, loaded via 128-bit vxorps
     align 32
     swk_sign_mask_ymm: times 8 dd 0x80000000  ; YMM sign-bit mask for 8-wide float negation
     section .text
; Named constants for stack frame and immediate offsets
%define SWK_FEAT_DIM 48                        ; Feature dimension d, 5th stack arg
%define SWK_COL_OFF 56                         ; Column offset for weight table, 6th stack arg
%define SWK_B256 256                           ; 256-element chunk size for main loop
%define SWK_B64 64                             ; 64-element tail chunk
%define SWK_B32 32                             ; 32-element tail chunk
%define SWK_B16 16                             ; 16-element tail chunk
%define SWK_B8 8                               ; 8-element tail chunk
%define SWK_SCALE_STR 4                        ; Scale stride, 4 bytes per float scale
%define SWK_4GRP_SOFF 16                       ; 4 scale groups = 16 bytes, 256-loop advance
%define SWK_FLOAT4_SCALE 4                     ; sizeof,float for SIB scale in activation addressing
%define SWK_W8_OFF 8                           ; 8 int8 weights = 8 bytes offset to next weight block
%define SWK_W16_OFF 16                         ; 16 int8 weights = 16 bytes offset
%define SWK_W24_OFF 24                         ; 24 int8 weights = 24 bytes offset
%define SWK_A32_OFF 32                         ; 8 floats = 32 bytes offset to next activation block
%define SWK_A64_OFF 64                         ; 16 floats = 64 bytes offset
%define SWK_A96_OFF 96                         ; 24 floats = 96 bytes offset
%define SWK_PF6_OFF 384                        ; prefetch distance: 6 groups * 64 bytes ahead
%define SWK_PF7_OFF 448                        ; prefetch distance: 7 groups * 64 bytes ahead
%define SWK_GRP2_SOFF 2                        ; scale offset for group 2, 2 scales from base
%define SWK_GRP3_SOFF 3                        ; scale offset for group 3, 3 scales from base

%macro DUAL_8BLOCK 6
; Step A: Load and convert Row0 weights + activation
     vpmovsxbd ymm10, [r10 + rcx + %1]         ; load+extend 8 gate_r0 weights
     vpmovsxbd ymm11, [r11 + rcx + %1]         ; load+extend 8 up_r0 weights
     vmovups ymm4, [rdi + rcx*SWK_FLOAT4_SCALE + %2] ; load 8 activations, independent stream
; Step B: Issue row1 loads while row0 converts, port interleaving
     vpmovsxbd ymm12, [r15 + rcx + %1]         ; load+extend 8 gate_r1 weights
     vpmovsxbd ymm13, [rbx + rcx + %1]         ; load+extend 8 up_r1 weights
     vcvtdq2ps ymm10, ymm10                    ; gate_r0 int32 -> float
     vmulps ymm10, ymm10, %3                   ; start gate_r0 scale chain
     vcvtdq2ps ymm11, ymm11                    ; up_r0 int32 -> float
     vmulps ymm11, ymm11, %4                   ; start up_r0 scale chain
     vcvtdq2ps ymm12, ymm12                    ; gate_r1 int32 -> float
     vmulps ymm12, ymm12, %5                   ; start gate_r1 scale chain
     vcvtdq2ps ymm13, ymm13                    ; up_r1 int32 -> float
     vmulps ymm13, ymm13, %6                   ; start up_r1 scale chain
     vfmadd231ps ymm0, ymm4, ymm10             ; gate_r0 accumulator
     vfmadd231ps ymm1, ymm4, ymm11             ; up_r0 accumulator
     vfmadd231ps ymm2, ymm4, ymm12             ; gate_r1 accumulator
     vfmadd231ps ymm3, ymm4, ymm13             ; up_r1 accumulator
%endmacro

%macro DUAL_FUSED_32 6
     DUAL_8BLOCK %1, %2, %3, %4, %5, %6        ; block 0: offsets %1, %2
     DUAL_8BLOCK %1+SWK_W8_OFF, %2+SWK_A32_OFF, %3, %4, %5, %6 ; block 1: weight +8 bytes, activation +32 bytes
     DUAL_8BLOCK %1+SWK_W16_OFF, %2+SWK_A64_OFF, %3, %4, %5, %6 ; block 2: weight +16 bytes, activation +64 bytes
     DUAL_8BLOCK %1+SWK_W24_OFF, %2+SWK_A96_OFF, %3, %4, %5, %6 ; block 3: weight +24 bytes, activation +96 bytes
%endmacro

%macro F32Q8_FUSED_8 6
     vpmovsxbd ymm10, [r10 + rcx + %1]         ; load+extend 8 gate weights
     vpmovsxbd ymm11, [r11 + rcx + %1]         ; load+extend 8 up weights
     vmovups ymm12, [rdi + rcx*SWK_FLOAT4_SCALE + %2] ; load 8 activations, independent stream
     vcvtdq2ps ymm10, ymm10                    ; gate weights to float
     vcvtdq2ps ymm11, ymm11                    ; up weights to float
     vmulps ymm10, ymm10, %3                   ; scale gate weights by gate_scale
     vmulps ymm11, ymm11, %4                   ; start both independent scale chains
     vfmadd231ps %5, ymm12, ymm10              ; gate acc += act * scaled_gate
     vfmadd231ps %6, ymm12, ymm11              ; up acc += act * scaled_up
%endmacro

%macro F32Q8_FUSED_32 4
     F32Q8_FUSED_8 %1, %2, %3, %4, ymm0, ymm4  ; gate/up acc0
     F32Q8_FUSED_8 %1+SWK_W8_OFF, %2+SWK_A32_OFF, %3, %4, ymm1, ymm5 ; gate/up acc1
     F32Q8_FUSED_8 %1+SWK_W16_OFF, %2+SWK_A64_OFF, %3, %4, ymm2, ymm6 ; gate/up acc2
     F32Q8_FUSED_8 %1+SWK_W24_OFF, %2+SWK_A96_OFF, %3, %4, ymm3, ymm7 ; gate/up acc3
%endmacro
     global v7s_swiglu_chunk                   ; exported: fused SwiGLU kernel
v7s_swiglu_chunk:
     SF_START swk_kernel, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles, reserve frame
     SF_XMM swk_xmm6                           ; xmm6 low half spill slot, Win64 preserves only the low 128 bits
     SF_XMM swk_xmm7                           ; xmm7 low half spill slot
     SF_XMM swk_xmm8                           ; xmm8 low half spill slot
     SF_XMM swk_xmm9                           ; xmm9 low half spill slot
     SF_XMM swk_xmm10                          ; xmm10 low half spill slot
     SF_XMM swk_xmm11                          ; xmm11 low half spill slot
     SF_XMM swk_xmm12                          ; xmm12 low half spill slot
     SF_XMM swk_xmm13                          ; xmm13 low half spill slot; DUAL_8BLOCK writes ymm13, so the low half must be preserved
     SF_QWORD swk_saved_r15                    ; gate ws base save slot, dual-row inner loop
     SF_DWORD swk_saved_ebx                    ; row index save slot, dual-row inner loop
     SF_QWORD swk_up_ws_base                   ; cached up scales base, avoid per-row global reload
     SF_QWORD swk_up_col_bias                  ; hoisted col_off * n, call-invariant up-row displacement
     SF_COMMIT                                 ; finalize stack frame layout
     vmovdqu swk_xmm6, xmm6                    ; save callee-saved xmm6 low half
     vmovdqu swk_xmm7, xmm7                    ; save callee-saved xmm7 low half
     vmovdqu swk_xmm8, xmm8                    ; save callee-saved xmm8 low half
     vmovdqu swk_xmm9, xmm9                    ; save callee-saved xmm9 low half
     vmovdqu swk_xmm10, xmm10                  ; save callee-saved xmm10 low half
     vmovdqu swk_xmm11, xmm11                  ; save callee-saved xmm11 low half
     vmovdqu swk_xmm12, xmm12                  ; save callee-saved xmm12 low half
     vmovdqu swk_xmm13, xmm13                  ; save callee-saved xmm13 low half; DUAL_8BLOCK writes ymm13, so it must be preserved
; --- Load arguments ---
     mov r12, rcx                              ; r12 = xout, output pointer, already offset
     mov r13, [rdx]                            ; r13 = float* x, input activations
     mov r14, [r8]                             ; r14 = gate wq, already offset int8_t*
     mov r15, [r8 + TENSOR_S_OFF]              ; r15 = gate ws, already offset float*
     mov rsi, [rel swiglu_up_weight_q]         ; rsi = up weight int8_t*, full base; set by v7d_swiglu immediately before dispatch, never NULL
     mov rax, [rel swiglu_up_weight_s]         ; rax = up weight scales, full base
     mov swk_up_ws_base, rax                   ; cache up scales base for per-row reuse
     mov edx, r9d                              ; edx = n, n/16 scale stride derivation below
     mov rdi, r13                              ; rdi = activation base, preserved across rows
     mov r13, rdx                              ; r13 = n, upstream pointer derivation below
     shr r13, 4                                ; r13 = n/16 = scale_stride for dual-row, 64-bit shift
     mov eax, dword [rbp + SWK_COL_OFF]        ; column offset for the up matrix, call-invariant
     imul rax, rdx                             ; rax = col_off * n
     mov swk_up_col_bias, rax                  ; park the displacement for every row setup below
; ===== Row loop, dual-row main path, single-row fallback for odd rows =====
     xor ebx, ebx                              ; row index = 0
     align 32
.outer_swk:
     mov eax, [rbp + SWK_FEAT_DIM]             ; eax = d, single stack load per row iteration
     cmp ebx, eax                              ; row >= d?
     jge .done_swk                             ; all rows done
; Check if 2 or more rows remain, dual path is the hot case: d/2-1 of d/2
; outer iterations, it FALLS THROUGH; only the final odd row jumps out
     dec eax                                   ; eax = d - 1
     cmp ebx, eax                              ; row >= d-1?, equivalent to row+1 >= d, one fewer stack read
     jge .single_row_swk                       ; no -> last odd row, single-row fallback
; SWK_SETUP_ROW_PTRS -- the shared row0 setup: gate/up weight row pointers and
; scale offsets for the row in ebx. IN: ebx = row, rdx = n, r15 = gate ws base,
; r14 = gate wq base, rsi = up wq base. OUT: r9 = gate ws walking ptr,
; r10 = gate wq row ptr, r8 = up ws row base, r11 = up wq row ptr.
; Clobbers rax.

%macro SWK_SETUP_ROW_PTRS 0
     mov r10d, ebx                             ; row index
     imul r10, rdx                             ; r10 = row * n
     mov rax, r10                              ; copy for scale offset
     shr rax, 4                                ; /16 -> scale byte offset = row * n/16
     lea r9, [r15 + rax]                       ; r9 = gate ws row0 base, walking
     mov r11, swk_up_col_bias                  ; hoisted col_off * n
     add r11, r10                              ; r11 = (col_off + row) * n
     add r10, r14                              ; r10 = gate wq row0 ptr, moved below the r11 use
     mov r8, r11                               ; copy for up scale index
     shr r8, 4                                 ; /16 -> up scale byte offset
     add r8, swk_up_ws_base                    ; r8 = up ws row0 base
     add r11, rsi                              ; r11 = up wq row0 ptr
%endmacro
; ===== Dual-row path, process 2 rows per iteration =====
     align 32
.dual_row_swk:
; Save registers that will be repurposed during the inner loop
     mov swk_saved_r15, r15                    ; save gate ws base for restore after inner loop
     mov swk_saved_ebx, ebx                    ; save current row index
; Row0 setup: gate weight row ptr + scale base --
     SWK_SETUP_ROW_PTRS                        ; r9/r10/r8/r11 = row0's four pointers
; Row1 base computation, row1 = row0 + n in weight space --
     lea r15, [r10 + rdx]                      ; r15 = gate_r1 weight base, r10 + n
     lea rbx, [r11 + rdx]                      ; rbx = up_r1 weight base, r11 + n
; Zero 4 vector accumulators, full 256-bit, one per dot product
     vxorps ymm0, ymm0, ymm0                   ; gate_r0 accumulator, 256-bit: the FMAs and HADD_YMM both consume all 8 lanes
     vxorps ymm1, ymm1, ymm1                   ; up_r0 accumulator, 256-bit, as above
     vxorps ymm2, ymm2, ymm2                   ; gate_r1 accumulator, 256-bit, as above
     vxorps ymm3, ymm3, ymm3                   ; up_r1 accumulator, 256-bit, as above
     xor ecx, ecx                              ; byte offset = 0
; ===== Dual-row 256-element fused inner loop, bottom-tested; eax holds the loop limit =====
     align 64
     lea eax, [rdx - SWK_B256]                 ; loop limit = n - 256, valid: n >= 256 on this path
     cmp edx, SWK_B256                         ; fewer than 256 elements?
     jb .dual_tail64                           ; yes -> straight to the cascade
.dual_loop256:
     ds prefetcht0 [r10 + rcx + SWK_PF6_OFF]   ; prefetch gate_r0 6 groups ahead, slot 19
     PREFETCH_SLOT 19                          ; Slot 19: swiglu dual gate_r0 pf6
; Group 0: elements 0-63, scales at [r9], [r8]
     vbroadcastss ymm8, [r9]                   ; gate_s0 group 0
     vbroadcastss ymm9, [r8]                   ; up_s0 group 0
     vbroadcastss ymm5, [r9 + r13]             ; gate_s1 group 0, row1 = row0 + scale_stride
     vbroadcastss ymm6, [r8 + r13]             ; up_s1 group 0
     DUAL_FUSED_32 0, 0, ymm8, ymm9, ymm5, ymm6 ; group 0: first 32 elements
     DUAL_FUSED_32 32, 128, ymm8, ymm9, ymm5, ymm6 ; group 0: second 32 elements
     ds prefetcht0 [r11 + rcx + SWK_PF6_OFF]   ; prefetch up_r0 6 groups ahead, slot 20
     PREFETCH_SLOT 20                          ; Slot 20: swiglu dual up_r0 pf6
; Group 1: elements 64-127
     vbroadcastss ymm8, [r9 + SWK_SCALE_STR]   ; gate_s0 group 1
     vbroadcastss ymm9, [r8 + SWK_SCALE_STR]   ; up_s0 group 1
     vbroadcastss ymm5, [r9 + SWK_SCALE_STR + r13] ; gate_s1 group 1
     vbroadcastss ymm6, [r8 + SWK_SCALE_STR + r13] ; up_s1 group 1
     DUAL_FUSED_32 64, 256, ymm8, ymm9, ymm5, ymm6 ; group 1: first 32 elements
     DUAL_FUSED_32 96, 384, ymm8, ymm9, ymm5, ymm6 ; group 1: second 32 elements
     ds prefetcht0 [r15 + rcx + SWK_PF6_OFF]   ; prefetch gate_r1 6 groups ahead, slot 21
     PREFETCH_SLOT 21                          ; Slot 21: swiglu dual gate_r1 pf6
; Group 2: elements 128-191
     vbroadcastss ymm8, [r9 + SWK_GRP2_SOFF*SWK_SCALE_STR] ; gate_s0 group 2
     vbroadcastss ymm9, [r8 + SWK_GRP2_SOFF*SWK_SCALE_STR] ; up_s0 group 2
     vbroadcastss ymm5, [r9 + SWK_GRP2_SOFF*SWK_SCALE_STR + r13] ; gate_s1 group 2
     vbroadcastss ymm6, [r8 + SWK_GRP2_SOFF*SWK_SCALE_STR + r13] ; up_s1 group 2
     DUAL_FUSED_32 128, 512, ymm8, ymm9, ymm5, ymm6 ; group 2: first 32 elements
     DUAL_FUSED_32 160, 640, ymm8, ymm9, ymm5, ymm6 ; group 2: second 32 elements
     ds prefetcht0 [rbx + rcx + SWK_PF6_OFF]   ; prefetch up_r1 6 groups ahead, slot 22
     PREFETCH_SLOT 22                          ; Slot 22: swiglu dual up_r1 pf6
; Group 3: elements 192-255
     vbroadcastss ymm8, [r9 + SWK_GRP3_SOFF*SWK_SCALE_STR] ; gate_s0 group 3
     vbroadcastss ymm9, [r8 + SWK_GRP3_SOFF*SWK_SCALE_STR] ; up_s0 group 3
     vbroadcastss ymm5, [r9 + SWK_GRP3_SOFF*SWK_SCALE_STR + r13] ; gate_s1 group 3
     vbroadcastss ymm6, [r8 + SWK_GRP3_SOFF*SWK_SCALE_STR + r13] ; up_s1 group 3
     DUAL_FUSED_32 192, 768, ymm8, ymm9, ymm5, ymm6 ; group 3: first 32 elements
     DUAL_FUSED_32 224, 896, ymm8, ymm9, ymm5, ymm6 ; group 3: second 32 elements
     add r9, SWK_4GRP_SOFF                     ; advance gate scale by 4 groups
     add r8, SWK_4GRP_SOFF                     ; advance up scale by 4 groups
     add ecx, SWK_B256                         ; advance byte offset by 256
     cmp ecx, eax                              ; another 256 fits?, i <= n-256
     jbe .dual_loop256                         ; loop, fall through to .dual_tail64 when done
; ===== Dual-row granular SIMD tails =====
     align 16
.dual_tail64:
     cmp ecx, edx                              ; all elements already done?, n%256==0
     jae .dual_reduce                          ; yes -> skip the whole cascade dead-walk
     ds prefetcht0 [r10 + rcx + SWK_PF6_OFF]   ; prefetch gate_r0 ahead, slot 23
     PREFETCH_SLOT 23                          ; Slot 23: swiglu dual tail gate_r0 pf6
     vbroadcastss ymm8, [r9]                   ; gate_s0
     vbroadcastss ymm9, [r8]                   ; up_s0
     ds prefetcht0 [r15 + rcx + SWK_PF6_OFF]   ; prefetch gate_r1 ahead, slot 24
     PREFETCH_SLOT 24                          ; Slot 24: swiglu dual tail gate_r1 pf6
     vbroadcastss ymm5, [r9 + r13]             ; gate_s1
     vbroadcastss ymm6, [r8 + r13]             ; up_s1
     ds prefetcht0 [r11 + rcx + SWK_PF6_OFF]   ; prefetch up_r0 ahead, slot 25
     PREFETCH_SLOT 25                          ; Slot 25: swiglu dual tail up_r0 pf6
     DUAL_FUSED_32 0, 0, ymm8, ymm9, ymm5, ymm6 ; 64-element tail: first 32 elements
     ds prefetcht0 [rbx + rcx + SWK_PF6_OFF]   ; prefetch up_r1 ahead, slot 26
     PREFETCH_SLOT 26                          ; Slot 26: swiglu dual tail up_r1 pf6
     DUAL_FUSED_32 32, 128, ymm8, ymm9, ymm5, ymm6 ; 64-element tail: second 32 elements
     add r9, SWK_SCALE_STR                     ; advance gate scale by 1 group
     add r8, SWK_SCALE_STR                     ; advance up scale by 1 group
     add ecx, SWK_B64                          ; advance byte offset by 64
.dual_tail32:
     cmp ecx, edx                              ; all elements already done?, n%256==64 after tail64
     jae .dual_reduce                          ; yes -> skip the remaining cascade dead-walk
     vbroadcastss ymm8, [r9]                   ; gate_s0
     vbroadcastss ymm9, [r8]                   ; up_s0
     vbroadcastss ymm5, [r9 + r13]             ; gate_s1
     vbroadcastss ymm6, [r8 + r13]             ; up_s1
     DUAL_FUSED_32 0, 0, ymm8, ymm9, ymm5, ymm6 ; 32-element tail: 32 elements dual-row
     add ecx, SWK_B32                          ; advance byte offset by 32
.dual_tail16:
; Gate/up scale registers are still live from .dual_tail32's reload above: DUAL_FUSED_32 and DUAL_8BLOCK only read ymm8/9/5/6.
     DUAL_8BLOCK 0, 0, ymm8, ymm9, ymm5, ymm6  ; 16-element tail: first 8 elements dual-row
     DUAL_8BLOCK 8, 32, ymm8, ymm9, ymm5, ymm6 ; 16-element tail: second 8 elements dual-row
     add ecx, SWK_B16                          ; advance byte offset by 16
.dual_tail8:
     DUAL_8BLOCK 0, 0, ymm8, ymm9, ymm5, ymm6  ; 8-element tail: 8 elements dual-row
     add ecx, SWK_B8                           ; advance byte offset by 8
     cmp ecx, edx                              ; all elements done?
     jae .dual_reduce                          ; yes -> reduce
     test cl, 63                               ; crossed 64-element boundary?
     jnz .dual_tail8                           ; no -> loop back, same scale
     add r9, SWK_SCALE_STR                     ; advance gate scale by one group
     add r8, SWK_SCALE_STR                     ; advance up scale by one group
     vbroadcastss ymm8, [r9]                   ; reload gate_s0 ONLY at the group boundary, the same-scale back-edge reuses registers
     vbroadcastss ymm9, [r8]                   ; reload up_s0
     vbroadcastss ymm5, [r9 + r13]             ; reload gate_s1
     vbroadcastss ymm6, [r8 + r13]             ; reload up_s1
     jmp .dual_tail8                           ; loop back with updated scales
; ===== Dual-row reduction + 2-wide vectorized sigmoid fusion =====
; Reduce ALL four accumulators to scalars, then process both rows'
; sigmoid evaluations in a single 2-wide SIMD pass.
     align 16
.dual_reduce:
     mov r15, swk_saved_r15                    ; restore gate ws base
     mov ebx, swk_saved_ebx                    ; restore current row index
; Four identical per-accumulator trees, interleaved for four independent chains.
     HADD4_YMM                                 ; gate_r0, up_r0, gate_r1, up_r1
; Pack both gates into xmm6, both ups into xmm7 for 2-wide sigmoid
     vinsertps xmm6, xmm0, xmm2, 0x1C          ; xmm6 = [gate_r0, gate_r1, 0, 0], imm bits 3-2 zero lanes 2/3 so the packed sigmoid runs on defined values
     vinsertps xmm7, xmm1, xmm3, 0x1C          ; xmm7 = [up_r0, up_r1, 0, 0]
     vmovaps xmm0, xmm6                        ; xmm0 = packed gates for sigmoid input
; 2-wide vectorized sigmoid on both gates in parallel
     EVAL_SIGMOID x, ps, [rel swk_sign_mask_ymm] ; xmm0 = [sig,gate_r0, sig,gate_r1, ...]
; Fuse both rows: result = gate * sigmoid,gate * up
     vmulps xmm0, xmm0, xmm6                   ; gate_r0/r1 * sigmoid,gate_r0/r1
     vmulps xmm0, xmm0, xmm7                   ; result *= up_r0/r1, final SwiGLU fusion
; Store both results as a single 64-bit write
     vmovq [r12 + rbx*SWK_FLOAT4_SCALE], xmm0  ; store both fused results as packed 64-bit
     add ebx, 2                                ; advance row index by 2, dual-row
     jmp .outer_swk                            ; continue to next work group
; ===== Single-row path, last odd row; entered ONLY via the outer-loop jge =====
.single_row_swk:
     SWK_SETUP_ROW_PTRS                        ; r9/r10/r8/r11 = the row's four pointers
     vxorps ymm0, ymm0, ymm0                   ; gate acc0 = 0, 256-bit: FMAs accumulate and HADD_YMM reduces all 8 lanes
     vxorps ymm1, ymm1, ymm1                   ; gate acc1 = 0, 256-bit, as above
     vxorps ymm2, ymm2, ymm2                   ; gate acc2 = 0, 256-bit, as above
     vxorps ymm3, ymm3, ymm3                   ; gate acc3 = 0, 256-bit, as above
     vxorps ymm4, ymm4, ymm4                   ; up acc0 = 0, 256-bit, as above
     vxorps ymm5, ymm5, ymm5                   ; up acc1 = 0, 256-bit, as above
     vxorps ymm6, ymm6, ymm6                   ; up acc2 = 0, 256-bit, as above
     vxorps ymm7, ymm7, ymm7                   ; up acc3 = 0, 256-bit, as above
     xor ecx, ecx                              ; byte offset = 0
     align 64
     lea eax, [rdx - SWK_B256]                 ; loop limit = n - 256, valid: n >= 256 on this path
     cmp edx, SWK_B256                         ; fewer than 256 elements?
     jb .single_tail64                         ; yes -> straight to the cascade
.single_loop256:
     ds prefetcht0 [r10 + rcx + SWK_PF6_OFF]   ; prefetch gate weights 6 groups ahead, slot 13
     PREFETCH_SLOT 13                          ; Slot 13: swiglu gate weights pf6
; Group 0: elements 0-63
     vbroadcastss ymm8, [r9]                   ; gate scale group 0
     vbroadcastss ymm9, [r8]                   ; up scale group 0
     F32Q8_FUSED_32 0, 0, ymm8, ymm9           ; group 0: first 32 elements, gate+up fused
     F32Q8_FUSED_32 32, 128, ymm8, ymm9        ; group 0: second 32 elements, gate+up fused
     ds prefetcht0 [r11 + rcx + SWK_PF6_OFF]   ; prefetch up weights 6 groups ahead, slot 14
     PREFETCH_SLOT 14                          ; Slot 14: swiglu up weights pf6
; Group 1: elements 64-127
     vbroadcastss ymm8, [r9 + SWK_SCALE_STR]   ; gate scale group 1
     vbroadcastss ymm9, [r8 + SWK_SCALE_STR]   ; up scale group 1
     F32Q8_FUSED_32 64, 256, ymm8, ymm9        ; group 1: first 32 elements, gate+up fused
     F32Q8_FUSED_32 96, 384, ymm8, ymm9        ; group 1: second 32 elements, gate+up fused
     ds prefetcht0 [r10 + rcx + SWK_PF7_OFF]   ; prefetch gate weights 7 groups ahead, slot 15
     PREFETCH_SLOT 15                          ; Slot 15: swiglu gate weights pf7
; Group 2: elements 128-191
     vbroadcastss ymm8, [r9 + SWK_GRP2_SOFF*SWK_SCALE_STR] ; gate scale group 2
     vbroadcastss ymm9, [r8 + SWK_GRP2_SOFF*SWK_SCALE_STR] ; up scale group 2
     F32Q8_FUSED_32 128, 512, ymm8, ymm9       ; group 2: first 32 elements, gate+up fused
     F32Q8_FUSED_32 160, 640, ymm8, ymm9       ; group 2: second 32 elements, gate+up fused
     ds prefetcht0 [r11 + rcx + SWK_PF7_OFF]   ; prefetch up weights 7 groups ahead, slot 16
     PREFETCH_SLOT 16                          ; Slot 16: swiglu up weights pf7
; Group 3: elements 192-255
     vbroadcastss ymm8, [r9 + SWK_GRP3_SOFF*SWK_SCALE_STR] ; gate scale group 3
     vbroadcastss ymm9, [r8 + SWK_GRP3_SOFF*SWK_SCALE_STR] ; up scale group 3
     F32Q8_FUSED_32 192, 768, ymm8, ymm9       ; group 3: first 32 elements, gate+up fused
     F32Q8_FUSED_32 224, 896, ymm8, ymm9       ; group 3: second 32 elements, gate+up fused
     add r9, SWK_4GRP_SOFF                     ; advance gate scale by 4 groups
     add r8, SWK_4GRP_SOFF                     ; advance up scale by 4 groups
     add ecx, SWK_B256                         ; advance byte offset by 256
     cmp ecx, eax                              ; another 256 fits?, i <= n-256
     jbe .single_loop256                       ; loop, fall through to .single_tail64 when done
.single_tail64:
     cmp ecx, edx                              ; all elements already done?, n%256==0
     jae .single_reduce                        ; yes -> skip the whole cascade dead-walk
; n%64==0, init-validated guarantees the 64-element chunk always fits here
     ds prefetcht0 [r10 + rcx + SWK_PF6_OFF]   ; prefetch gate weights ahead, slot 17
     PREFETCH_SLOT 17                          ; Slot 17: swiglu gate weights tail pf6
     vbroadcastss ymm8, [r9]                   ; gate scale
     vbroadcastss ymm9, [r8]                   ; up scale
     ds prefetcht0 [r11 + rcx + SWK_PF6_OFF]   ; prefetch up weights ahead, slot 18
     PREFETCH_SLOT 18                          ; Slot 18: swiglu up weights tail pf6
     F32Q8_FUSED_32 0, 0, ymm8, ymm9           ; 64-element tail: first 32 elements, gate+up fused
     F32Q8_FUSED_32 32, 128, ymm8, ymm9        ; 64-element tail: second 32 elements, gate+up fused
     add r9, SWK_SCALE_STR                     ; advance gate scale by 1 group
     add r8, SWK_SCALE_STR                     ; advance up scale by 1 group
     add ecx, SWK_B64                          ; advance byte offset by 64
.single_tail32:
     cmp ecx, edx                              ; all elements already done?, n%256==64 after tail64
     jae .single_reduce                        ; yes -> skip the remaining cascade dead-walk
     vbroadcastss ymm8, [r9]                   ; gate scale
     vbroadcastss ymm9, [r8]                   ; up scale
     F32Q8_FUSED_32 0, 0, ymm8, ymm9           ; 32-element tail: 32 elements, gate+up fused
     add ecx, SWK_B32                          ; advance byte offset by 32
.single_tail16:
; Gate/up scale registers are still live from .single_tail32's reload above: F32Q8_FUSED_32 only reads ymm8/9.
     F32Q8_FUSED_8 0, 0, ymm8, ymm9, ymm0, ymm4 ; 16-tail: first 8 elements
     F32Q8_FUSED_8 SWK_W8_OFF, SWK_A32_OFF, ymm8, ymm9, ymm1, ymm5 ; 16-tail: next 8 elements
     add ecx, SWK_B16                          ; advance byte offset by 16
.single_tail8:
     F32Q8_FUSED_8 0, 0, ymm8, ymm9, ymm0, ymm4 ; 8-tail: the last 8 elements
     add ecx, SWK_B8                           ; advance byte offset by 8
     cmp ecx, edx                              ; all elements done?
     jae .single_reduce                        ; yes -> reduce
     test cl, 63                               ; crossed 64-element boundary?
     jnz .single_tail8                         ; no -> loop back, same scale
     add r9, SWK_SCALE_STR                     ; advance gate scale by one group
     add r8, SWK_SCALE_STR                     ; advance up scale by one group
     vbroadcastss ymm8, [r9]                   ; reload gate scale ONLY at the group boundary, the same-scale back-edge reuses registers
     vbroadcastss ymm9, [r8]                   ; reload up scale
     jmp .single_tail8                         ; loop back with updated scales
; ===== Reduction + sigmoid + fusion, single-row =====
.single_reduce:
; Reduce up, ymm4-ymm7 -> xmm7, directly into non-volatile
     vaddps ymm4, ymm4, ymm5                   ; up: acc0+acc1
     vaddps ymm6, ymm6, ymm7                   ; up: acc2+acc3
     vaddps ymm4, ymm4, ymm6                   ; up: combined
     HADD_YMM xmm4, ymm4, xmm5                 ; Reduce up to scalar in xmm4, volatile, so the low-half extract is a self-copy
     vmovaps xmm7, xmm4                        ; park the up scalar outside EVAL_SIGMOID's xmm1-xmm5 overwrite set, VEX.128 register move, no legacy-SSE state
; Reduce gate, ymm0-ymm3 -> xmm6, directly into non-volatile
     vaddps ymm0, ymm0, ymm1                   ; gate: acc0+acc1
     vaddps ymm2, ymm2, ymm3                   ; gate: acc2+acc3
     vaddps ymm0, ymm0, ymm2                   ; gate: combined
     HADD_YMM xmm0, ymm0, xmm2                 ; Reduce gate to scalar in xmm0, volatile, so the low-half extract is a self-copy
     vmovaps xmm6, xmm0                        ; park the gate scalar outside EVAL_SIGMOID's xmm1-xmm5 overwrite set, VEX.128 register move, no legacy-SSE state
; Fuse: result = gate * sigmoid,gate * up
     vmovss xmm0, xmm6                         ; xmm0 = gate scalar
     EVAL_SIGMOID x, ss, [rel swk_sign_mask]   ; xmm0 = sigmoid,gate
     vmulss xmm0, xmm0, xmm6                   ; xmm0 = sigmoid,gate * gate
     vmulss xmm0, xmm0, xmm7                   ; xmm0 = gate * sigmoid,gate * up
     vmovss [r12 + rbx*SWK_FLOAT4_SCALE], xmm0 ; xout[row] = SwiGLU result
     inc ebx                                   ; advance row index
     jmp .outer_swk                            ; next row
; ===== Epilogue =====
.done_swk:
     CLEANUP_YMM                               ; vzeroupper BEFORE the restores: ABI volatility of the upper YMM halves means only the low 128 bits need to survive
     vmovdqu xmm13, swk_xmm13                  ; restore callee-saved xmm13 low half
     vmovdqu xmm12, swk_xmm12                  ; restore callee-saved xmm12 low half
     vmovdqu xmm11, swk_xmm11                  ; restore callee-saved xmm11 low half
     vmovdqu xmm10, swk_xmm10                  ; restore callee-saved xmm10 low half
     vmovdqu xmm9, swk_xmm9                    ; restore callee-saved xmm9 low half
     vmovdqu xmm8, swk_xmm8                    ; restore callee-saved xmm8 low half
     vmovdqu xmm7, swk_xmm7                    ; restore callee-saved xmm7 low half
     vmovdqu xmm6, swk_xmm6                    ; restore callee-saved xmm6 low half
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; epilogue: restore non-volatiles, return
%unmacro F32Q8_FUSED_32 4                      ; remove the 32-element fused macro, the trailing count must match its %macro
%unmacro F32Q8_FUSED_8 6                       ; remove the 8-element fused leaf macro that F32Q8_FUSED_32 expands
%unmacro DUAL_FUSED_32 6                       ; remove the dual-row 32-element fused macro
%unmacro DUAL_8BLOCK 6                         ; remove the dual-row 8-block macro
%undef SWK_GRP3_SOFF
%undef SWK_GRP2_SOFF
%undef SWK_PF7_OFF
%undef SWK_PF6_OFF
%undef SWK_A96_OFF
%undef SWK_A64_OFF
%undef SWK_A32_OFF
%undef SWK_W24_OFF
%undef SWK_W16_OFF
%undef SWK_W8_OFF
%undef SWK_FLOAT4_SCALE
%undef SWK_4GRP_SOFF
%undef SWK_SCALE_STR
%undef SWK_B8
%undef SWK_B16
%undef SWK_B32
%undef SWK_B64
%undef SWK_B256
%undef SWK_COL_OFF
%undef SWK_FEAT_DIM
%undef swk_xmm13
%undef swk_xmm12
%undef swk_xmm11
%undef swk_xmm10
%undef swk_xmm9
%undef swk_xmm8
%undef swk_xmm7
%undef swk_xmm6
%undef swk_up_ws_base_raw
%undef swk_up_ws_base
%undef swk_up_col_bias_raw
%undef swk_up_col_bias
%undef swk_saved_ebx_raw
%undef swk_saved_ebx
%undef swk_saved_r15_raw
%undef swk_saved_r15
