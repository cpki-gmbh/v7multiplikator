; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; F16C KV cache pack helper. Included by attention_kernel.inc (part of v7_kernels).
; Win64: RCX=uint16_t* dest, RDX=float* source, R8D=count, positive multiple of 8.
; Rounding: nearest-even independently of MXCSR. Volatile regs only; no dispatch.
; Full groups convert to registers first, then issue separate 16-byte stores:
; the core schedules conversions and stores independently, measured 28-49%
; faster on the leaf across sizes.
%define KVF_SRC_BLOCK_BYTES 32                 ; bytes in one 8-element FP32 source block
%define KVF_SRC_BLOCK2_BYTES 64                ; two source blocks, second 8-float displacement
%define KVF_SRC_BLOCK3_BYTES 96                ; three source blocks, fourth 8-float displacement
%define KVF_DST_BLOCK_BYTES 16                 ; bytes in one 8-element FP16 destination block
%define KVF_DST_BLOCK2_BYTES 32                ; two destination blocks
%define KVF_DST_BLOCK3_BYTES 48                ; three destination blocks
     section .text
v7s_kv_pack_f16:
     cmp r8d, 32                               ; at least one full 32-float group?
     jb .pack_small                            ; no: only the 8/16/24-float remainders
.pack_thirtytwo:
     vmovups ymm0, [rdx]                       ; source floats 0..7 of this group
     vmovups ymm1, [rdx + KVF_SRC_BLOCK_BYTES] ; source floats 8..15
     vmovups ymm2, [rdx + KVF_SRC_BLOCK2_BYTES] ; source floats 16..23
     vmovups ymm3, [rdx + KVF_SRC_BLOCK3_BYTES] ; source floats 24..31
     vcvtps2ph xmm0, ymm0, 0                   ; register-destination conversion, nearest-even
     vcvtps2ph xmm1, ymm1, 0                   ; same rounding immediate as the memory-destination form
     vcvtps2ph xmm2, ymm2, 0                   ; independent of MXCSR, per the pack contract
     vcvtps2ph xmm3, ymm3, 0                   ; the core can overlap these with the stores below
     vmovups [rcx], xmm0                       ; separate 16-byte stores
     vmovups [rcx + KVF_DST_BLOCK_BYTES], xmm1 ; FP16 dest 8..15
     vmovups [rcx + KVF_DST_BLOCK2_BYTES], xmm2 ; FP16 dest 16..23
     vmovups [rcx + KVF_DST_BLOCK3_BYTES], xmm3 ; FP16 dest 24..31
     sub rdx, -(4 * KVF_SRC_BLOCK_BYTES)       ; advance 128 bytes using a signed imm8
     add rcx, 4 * KVF_DST_BLOCK_BYTES          ; advance the destination one full group
     sub r8d, 32                               ; 32 fewer floats to pack
     cmp r8d, 32                               ; another full group fits?
     jae .pack_thirtytwo                       ; yes: next group
.pack_small:
     test r8d, r8d                             ; a multiple of 32 needs no tail
     jz .pack_done                             ; yes: all packed
     cmp r8d, 16                               ; 8, 16 or 24 floats remain
     jb .pack_eight                            ; 8 floats: single-block tail
.pack_sixteen:
     vmovups ymm0, [rdx]                       ; source floats 0..7 of the pair
     vmovups ymm2, [rdx + KVF_SRC_BLOCK_BYTES] ; source floats 8..15
     vcvtps2ph oword [rcx], ymm0, 0            ; short remainders keep the memory-destination form
     vcvtps2ph oword [rcx + KVF_DST_BLOCK_BYTES], ymm2, 0 ; FP16 dest 8..15
     sub r8d, 16                               ; one pair done
     jz .pack_done                             ; done (sub set ZF), otherwise 8-float tail
     add rdx, 2 * KVF_SRC_BLOCK_BYTES          ; advance the source one pair
     add rcx, 2 * KVF_DST_BLOCK_BYTES          ; advance the destination one pair
.pack_eight:
     vmovups ymm0, [rdx]                       ; final 8 FP32 source elements
     vcvtps2ph oword [rcx], ymm0, 0            ; final 8 FP16 store
.pack_done:
     ret                                       ; return, dest fully written
%undef KVF_SRC_BLOCK_BYTES
%undef KVF_SRC_BLOCK2_BYTES
%undef KVF_SRC_BLOCK3_BYTES
%undef KVF_DST_BLOCK_BYTES
%undef KVF_DST_BLOCK2_BYTES
%undef KVF_DST_BLOCK3_BYTES
