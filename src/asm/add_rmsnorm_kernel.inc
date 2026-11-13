; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define ARN_YMM_BLOCK 32                       ; 32-byte stride between consecutive YMM blocks
%define ARN_4X_STRIDE 128                      ; 4 * ARN_YMM_BLOCK, stride per 4-way iteration
%define ARN_4X_STRIDE_NEG -ARN_4X_STRIDE       ; negated stride: 'sub rax, ARN_4X_STRIDE_NEG' is a forward advance of 128 bytes written in 4 bytes, REX.W + 83 /5 + sign-extended imm8; -128 is inside the signed imm8 range, so NASM narrows to the short form against 6 for 'add rax, 128', the 48 05 imm32 opcode; +128 exceeds the signed imm8 range and 7 for 'lea rax, [rax+128]'. The negated form is a size trick only, the advance is forward, so do not spell it 'lea rax, [rax + ARN_4X_STRIDE_NEG]', which would walk the output backwards
%define ARN_BLK1 32                            ; offset to block 1, 1 * ARN_YMM_BLOCK
%define ARN_BLK2 64                            ; offset to block 2, 2 * ARN_YMM_BLOCK
%define ARN_BLK3 96                            ; offset to block 3, 3 * ARN_YMM_BLOCK
%define ARN_FEAT_DIM 48                        ; Feature dimension, 5th stack arg
%define ARN_XMM6_SAVE 0                        ; xmm6 save slot, lower 16 bytes of the 32-byte frame
%define ARN_XMM7_SAVE 16                       ; xmm7 save slot, upper 16 bytes of the 32-byte frame
     section .text
     SF_START v7s_fused_add_rmsnorm            ; declare stack frame, emits the global + entry label
     SF_COMMIT                                 ; commit stack frame, allocate shadow
; Save xmm6/xmm7: the 4-way loop uses ymm4-7 for block temps; on Win64
; ymm6-7's lower 128 bits ARE xmm6-7, callee-saved, and vzeroupper only
; clears the upper halves. Without this save, every C++ caller that
; keeps a double in xmm6/xmm7 gets it destroyed, the compiler's own
; /arch:AVX2 codegen does this constantly, the loadtest's t0 was
; garbled every run .
     vmovups [rsp + ARN_XMM6_SAVE], xmm6       ; save callee-saved xmm6, ymm6 lower half
     vmovups [rsp + ARN_XMM7_SAVE], xmm7       ; save callee-saved xmm7, ymm7 lower half
     mov r10, rdx                              ; r10 = residual base, Pass 1 advances rdx; Pass 2 addresses via [r10+rax]
     mov r11, r9                               ; r11 = weight base, untouched by Pass 1; Pass 2 addresses via [r11+rax]
     mov eax, [rbp + ARN_FEAT_DIM]             ; eax = dim from 5th stack arg, multiple of 8 is the import's contract for the block split
     TRACE_REG "arn_dim", rax                  ; debug: dim as read from the stack arg
; Split dim into AVX blocks, no scalar tail: the import guarantees dim % 8 == 0
     shr eax, 3                                ; eax = dim / 8, block count, preserved
     vxorps ymm0, ymm0, ymm0                   ; sumsq accumulator A, blocks 0,4,8,...
     vxorps ymm1, ymm1, ymm1                   ; sumsq accumulator B, blocks 1,5,9,...
     vxorps ymm2, ymm2, ymm2                   ; sumsq accumulator C, blocks 2,6,10,...
     vxorps ymm3, ymm3, ymm3                   ; sumsq accumulator D, blocks 3,7,11,...
; eax = total AVX blocks, preserved for Pass 2 at :67, dim % 8 == 0 is the import
; contract at :17; dim == 0 gives 0 blocks and falls straight through both zero exits
     mov r9d, eax                              ; r9d = blocks that do not fill a 4-way group
     and r9d, 3                                ; blocks % 4, 0 for every supported model
     jnz .arn_p1_left_loop                     ; rare: leftovers -> out-of-line loop below
.arn_p1_groups:
     mov r9d, eax                              ; r9d = total AVX blocks
     shr r9d, 2                                ; r9d = groups of 4 blocks, 128 bytes per iteration
     jz .arn_p1_reduce                         ; fewer than 4 blocks -> reduce directly
     align 16                                  ; align vector loop for decode efficiency
.arn_p1_vec_loop:
; All four loads, then all four adds, then the four sumsq FMAs, then the
; in-place residual stores -- the FMAs read registers only, so they run
; before the stores retire
     vmovups ymm4, [rdx]                       ; ymm4 = residual[block0]
     vmovups ymm5, [rdx+ARN_BLK1]              ; ymm5 = residual[block1]
     vmovups ymm6, [rdx+ARN_BLK2]              ; ymm6 = residual[block2]
     vmovups ymm7, [rdx+ARN_BLK3]              ; ymm7 = residual[block3]
     vaddps ymm4, ymm4, [r8]                   ; ymm4 = residual + add, block 0
     vaddps ymm5, ymm5, [r8+ARN_BLK1]          ; block 1
     vaddps ymm6, ymm6, [r8+ARN_BLK2]          ; block 2
     vaddps ymm7, ymm7, [r8+ARN_BLK3]          ; block 3
     vfmadd231ps ymm0, ymm4, ymm4              ; sumsq_A += sum^2, block 0
     vfmadd231ps ymm1, ymm5, ymm5              ; sumsq_B += sum^2, block 1
     vfmadd231ps ymm2, ymm6, ymm6              ; sumsq_C += sum^2, block 2
     vfmadd231ps ymm3, ymm7, ymm7              ; sumsq_D += sum^2, block 3
     vmovups [rdx], ymm4                       ; residual[block0] = sum, in-place
     vmovups [rdx+ARN_BLK1], ymm5              ; block 1
     vmovups [rdx+ARN_BLK2], ymm6              ; block 2
     vmovups [rdx+ARN_BLK3], ymm7              ; block 3
     sub rdx, ARN_4X_STRIDE_NEG                ; advance residual by 4 blocks, FORWARD: sub -128 == add 128 in 4 bytes instead of 7, see the encoding note at :7
     sub r8, ARN_4X_STRIDE_NEG                 ; advance add by 4 blocks, same encoding
     dec r9d                                   ; one fewer group remaining
     jne .arn_p1_vec_loop                      ; continue 4-way loop
.arn_p1_reduce:
     vaddps ymm0, ymm0, ymm1                   ; merge accumulators A + B
     vaddps ymm2, ymm2, ymm3                   ; merge accumulators C + D
     vaddps ymm0, ymm0, ymm2                   ; A+B +, C+D, same tree as RMSNORM_ROPE_BODY
     HADD_YMM xmm0, ymm0, xmm1                 ; Reduce to scalar
     mov r9d, [rbp + ARN_FEAT_DIM]             ; r9d = dim, re-read: eax is the block count now
     vcvtsi2ss xmm5, xmm5, r9d                 ; xmm5 =, float dim
     COMPUTE_RMS_SCALE xmm0, xmm5, xmm0        ; RMS scale 1/sqrt,sumsq/dim + eps, value-identical, each op exact IEEE-754.%2 is an XMM REGISTER, the macro header documents both the memory and the register form, required: Pass 1's block temps are ymm4-ymm7, so xmm5 cannot be preserved across the loop and, float dim cannot be spilled and re-read after the reduce
     vbroadcastss ymm0, xmm0                   ; ymm0 = scale, broadcast to all lanes
; ===== Pass 2: o = weight * residual * scale, 4-way unrolled =====
     mov r9d, eax                              ; r9d = total AVX blocks, dim % 8 == 0 is the import contract at :17; dim == 0 gives 0 blocks and .arn_p2_tail takes its zero exit
     xor edx, edx                              ; rdx = shared byte offset; zeroed BEFORE the flag-setting shr so the jz below reads shr's ZF
     and r9d, 3                                ; r9d = blocks, 0-3 that do not fill a 4-way group. Covers ODD counts directly, dim=8 -> 1, dim=24 -> 3, so no odd-block prologue is needed
     shr eax, 2                                ; eax = groups of 4 blocks, 128 bytes per iteration
     jz .arn_p2_tail                           ; fewer than 4 blocks -> tail path
     align 16                                  ; align vector loop for decode efficiency
.arn_p2_vec_loop:
; 4-way interleaved: issue all scale muls first, then weight muls, hide 4-cycle vmulps latency
     vmulps ymm2, ymm0, [r10+rdx]              ; residual * scale, block 0
     vmulps ymm3, ymm0, [r10+rdx+ARN_BLK1]     ; residual * scale, block 1
     vmulps ymm4, ymm0, [r10+rdx+ARN_BLK2]     ; residual * scale, block 2
     vmulps ymm5, ymm0, [r10+rdx+ARN_BLK3]     ; residual * scale, block 3
     vmulps ymm2, ymm2, [r11+rdx]              ; * weight, block 0
     vmulps ymm3, ymm3, [r11+rdx+ARN_BLK1]     ; * weight, block 1
     vmulps ymm4, ymm4, [r11+rdx+ARN_BLK2]     ; * weight, block 2
     vmulps ymm5, ymm5, [r11+rdx+ARN_BLK3]     ; * weight, block 3
     vmovups [rcx+rdx], ymm2                   ; o[block0]
     vmovups [rcx+rdx+ARN_BLK1], ymm3          ; o[block1]
     vmovups [rcx+rdx+ARN_BLK2], ymm4          ; o[block2]
     vmovups [rcx+rdx+ARN_BLK3], ymm5          ; o[block3]
     sub rdx, ARN_4X_STRIDE_NEG                ; advance the shared byte offset FORWARD by 4 blocks, one update for all 3 arrays; 'sub rdx, -128' is the 4-byte sign-extended imm8 form, against 7 for 'add rdx, 128' and 7 for 'lea rdx, [rdx+128]'
     dec eax                                   ; one fewer group remaining
     jne .arn_p2_vec_loop                      ; continue 4-way loop
.arn_p2_tail:
     test r9d, r9d                             ; any remaining blocks?
     jnz .arn_p2_tail_loop                     ; rare: leftovers -> out-of-line loop below
.arn_done:
     CLEANUP_YMM                               ; vzeroupper for AVX ABI compliance
     vmovups xmm6, [rsp + ARN_XMM6_SAVE]       ; restore callee-saved xmm6, was in ymm6's lower half
     vmovups xmm7, [rsp + ARN_XMM7_SAVE]       ; restore callee-saved xmm7, was in ymm7's lower half
     SF_END                                    ; release stack frame and restore saved regs
; ---- out-of-line rare paths, contract: blocks % 4 == 0 for every supported
; ---- model, so these loops exist for the general contract only ----
.arn_p1_left_loop:                             ; out-of-line: blocks that do not fill a 4-way group, contract path, no supported model hits it
     vmovups ymm4, [rdx]                       ; ymm4 = residual[block]
     vaddps ymm4, ymm4, [r8]                   ; ymm4 = residual + add, folded load
     vmovups [rdx], ymm4                       ; residual[block] = sum, in-place
     vfmadd231ps ymm0, ymm4, ymm4              ; sumsq_A += sum^2
     add rdx, ARN_YMM_BLOCK                    ; advance residual by 1 block
     add r8, ARN_YMM_BLOCK                     ; advance add by 1 block
     dec r9d                                   ; one fewer leftover block
     jnz .arn_p1_left_loop                     ; continue, entered with r9d >= 1
     jmp .arn_p1_groups                        ; leftovers done -> join the groups loop, rdx/r8 advanced past them
.arn_p2_tail_loop:                             ; out-of-line: blocks that do not fill a 4-way group, contract path, no supported model hits it
     vmulps ymm2, ymm0, [r10+rdx]              ; residual * scale
     vmulps ymm2, ymm2, [r11+rdx]              ; * weight
     vmovups [rcx+rdx], ymm2                   ; store
     add rdx, ARN_YMM_BLOCK                    ; advance the shared byte offset by 1 block
     dec r9d                                   ; one fewer block
     jnz .arn_p2_tail_loop                     ; continue
     jmp .arn_done                             ; all leftover blocks done -> epilogue
; ---- %undef cleanup: keep the preprocessor namespace clean for later includes ----
%undef ARN_FEAT_DIM
%undef ARN_XMM6_SAVE
%undef ARN_XMM7_SAVE
%undef ARN_YMM_BLOCK
%undef ARN_4X_STRIDE_NEG
%undef ARN_4X_STRIDE
%undef ARN_BLK3
%undef ARN_BLK2
%undef ARN_BLK1
