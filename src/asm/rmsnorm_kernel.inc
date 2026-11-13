; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define RN_4X_STRIDE 128                       ; byte stride per 4-block iteration, both loops
%define RN_4X_STRIDE_NEG -RN_4X_STRIDE         ; negated stride for the [reg - N] addressing form, RAO disp8 range
%define RN_BLK1 32                             ; byte offset to block 1
%define RN_BLK2 64                             ; byte offset to block 2
%define RN_BLK3 96                             ; byte offset to block 3
%define RN_ELEMS_PER_BLOCK 8                   ; elements per 32-byte AVX block, restores the original element count from the block count
     section .text
     SF_START v7s_rmsnorm                      ; declare stack frame, emits the global + entry label ; no register list, zero callee-saved GPR usage, no calls
     SF_COMMIT                                 ; frame = shadow only
     mov rax, rdx                              ; rax = x walker, advanced per block
; size / 8 = AVX block count. BOTH vector loops are do-while 4-block groups with
; no leftover handling: the import contract is size % 32 == 0, block count % 4 == 0 .
; Every supported model dim, head_dim 64/128 x heads products: 576..4096 satisfies it.
; A size that is a multiple of 8 but not 32 would silently drop leftover blocks and
; wrap the group counter, do not call this kernel with such a size.
     shr r9d, 3                                ; r9d = size / 8, AVX block count
; ===== Phase 1: sum-of-squares with four independent accumulators, 4-way unrolled =====
     vxorps ymm0, ymm0, ymm0                   ; sumsq accumulator A, blocks 0,4,8,...
     vxorps ymm1, ymm1, ymm1                   ; sumsq accumulator B, blocks 1,5,9,...
     vxorps ymm2, ymm2, ymm2                   ; sumsq accumulator C, blocks 2,6,10,...
     vxorps ymm3, ymm3, ymm3                   ; sumsq accumulator D, blocks 3,7,11,...
     mov r10d, r9d                             ; r10d = total AVX blocks
     shr r10d, 2                               ; r10d = 4-block group count for Phase 1
     align 16
.rn_p1_vec_loop:
     vmovups ymm4, [rax]                       ; x[block0]
     vmovups ymm5, [rax+RN_BLK1]               ; x[block1]
     vfmadd231ps ymm0, ymm4, ymm4              ; sumsq_A += x[block0]^2
     vfmadd231ps ymm1, ymm5, ymm5              ; sumsq_B += x[block1]^2
     vmovups ymm4, [rax+RN_BLK2]               ; x[block2]
     vmovups ymm5, [rax+RN_BLK3]               ; x[block3]
     vfmadd231ps ymm2, ymm4, ymm4              ; sumsq_C += x[block2]^2
     vfmadd231ps ymm3, ymm5, ymm5              ; sumsq_D += x[block3]^2
     sub rax, RN_4X_STRIDE_NEG                 ; advance x walker FORWARD by 4 blocks, sub -128 is the 4-byte sign-extended imm8 form, against 6 for 'add rax, 128'
     dec r10d                                  ; one fewer group
     jne .rn_p1_vec_loop                       ; continue 4-way loop
.rn_p1_reduce:
; Pairwise merge of the four accumulators, shallower tree than the dual merge
     vaddps ymm0, ymm0, ymm1                   ; merge A+B
     vaddps ymm2, ymm2, ymm3                   ; merge C+D
     vaddps ymm0, ymm0, ymm2                   ; merge the pairs
     HADD_YMM xmm0, ymm0, xmm1                 ; Reduce to scalar
     lea r10d, [r9d*RN_ELEMS_PER_BLOCK]        ; r10d = blocks * 8 = original size, element count
     vcvtsi2ss xmm5, xmm5, r10d                ; xmm5 =, float size, VEX-encoded, avoids AVX-SSE transition
     COMPUTE_RMS_SCALE xmm0, xmm5, xmm0        ; mean_sq = sumsq/size, + eps, sqrt, reciprocal, expands to the same five instructions; %2 is an XMM REGISTER, required because xmm5 is produced by the vcvtsi2ss above and consumed immediately, the macro header documents both the memory and register forms
     vbroadcastss ymm0, xmm0                   ; ymm0 = scale, broadcast to all lanes
     xor eax, eax                              ; rax = byte offset 0
     mov r10d, r9d                             ; r10d = total AVX blocks, r9d kept as block count since the shr
     shr r10d, 2                               ; r10d = 4-block group count for Phase 2
     align 16
.rn_p2_vec_loop:
; 4-way interleaved: issue all scale muls first, then weight muls, then
; stores, 4 independent chains hide 4-cycle vmulps latency
     vmulps ymm2, ymm0, [rdx + rax]            ; ymm2 = scale * x[block0], folded load
     vmulps ymm4, ymm0, [rdx + rax + RN_BLK1]  ; ymm4 = scale * x[block1], folded load
     vmulps ymm3, ymm0, [rdx + rax + RN_BLK2]  ; ymm3 = scale * x[block2], folded load
     vmulps ymm5, ymm0, [rdx + rax + RN_BLK3]  ; ymm5 = scale * x[block3], folded load
     vmulps ymm2, ymm2, [r8 + rax]             ; ymm2 =, scale*x * weight[block0], C ref order
     vmulps ymm4, ymm4, [r8 + rax + RN_BLK1]   ; ymm4 =, scale*x * weight[block1]
     vmulps ymm3, ymm3, [r8 + rax + RN_BLK2]   ; ymm3 =, scale*x * weight[block2]
     vmulps ymm5, ymm5, [r8 + rax + RN_BLK3]   ; ymm5 =, scale*x * weight[block3]
     vmovups [rcx + rax], ymm2                 ; o[block0] = result
     vmovups [rcx + rax + RN_BLK1], ymm4       ; o[block1] = result
     vmovups [rcx + rax + RN_BLK2], ymm3       ; o[block2] = result
     vmovups [rcx + rax + RN_BLK3], ymm5       ; o[block3] = result
     sub rax, RN_4X_STRIDE_NEG                 ; advance offset FORWARD by 4 blocks, bases stay fixed, 2 fewer adds
     dec r10d                                  ; one fewer group
     jne .rn_p2_vec_loop                       ; continue 4-way loop
.rn_done:
     CLEANUP_YMM                               ; vzeroupper for AVX ABI compliance
     SF_END                                    ; release stack frame, no callee-saved GPRs to restore
; %undef cleanup: file-local defines leave scope here, house convention --
%undef RN_ELEMS_PER_BLOCK                      ; file-local elements per AVX block
%undef RN_BLK3                                 ; file-local block-3 byte offset
%undef RN_BLK2                                 ; file-local block-2 byte offset
%undef RN_BLK1                                 ; file-local block-1 byte offset
%undef RN_4X_STRIDE_NEG                        ; file-local negated 4-block stride, sub-imm8 encoding trick
%undef RN_4X_STRIDE                            ; file-local 4-block byte stride
