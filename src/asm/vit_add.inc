; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define VA_YMM32 32                            ; byte offset of YMM block 1 within the group, floats 8..15
%define VA_YMM64 64                            ; byte offset of YMM block 2 within the group, floats 16..23
%define VA_YMM96 96                            ; byte offset of YMM block 3 within the group, floats 24..31
%define VA_LOG2_GROUP 5                        ; log2,32, group count = count / 32
%define VA_UNROLL_STRIDE 128                   ; byte stride from a 32-float group to the next, == 32 floats ; the pair loop's 64-float advance is this times 2

; VA_ADD32 -- one 32-float add group at the given displacement; the identical
; body of the pair loop's two halves and of the odd-group tail.

%macro VA_ADD32 1                              ; %1 = displacement of this 32-float group
     vmovups ymm0, [rdx + %1]                  ; load source floats 0..7
     vmovups ymm1, [rdx + %1 + VA_YMM32]       ; load source floats 8..15
     vmovups ymm2, [rdx + %1 + VA_YMM64]       ; load source floats 16..23
     vmovups ymm3, [rdx + %1 + VA_YMM96]       ; load source floats 24..31
     vaddps ymm0, ymm0, [rcx + %1]             ; add destination floats 0..7
     vaddps ymm1, ymm1, [rcx + %1 + VA_YMM32]  ; add destination floats 8..15
     vaddps ymm2, ymm2, [rcx + %1 + VA_YMM64]  ; add destination floats 16..23
     vaddps ymm3, ymm3, [rcx + %1 + VA_YMM96]  ; add destination floats 24..31
     vmovups [rcx + %1], ymm0                  ; write floats 0..7
     vmovups [rcx + %1 + VA_YMM32], ymm1       ; write floats 8..15
     vmovups [rcx + %1 + VA_YMM64], ymm2       ; write floats 16..23
     vmovups [rcx + %1 + VA_YMM96], ymm3       ; write floats 24..31
%endmacro

     global vit_add                            ; export: elementwise DeepStack feature add, dst += src
vit_add:
     shr r8d, VA_LOG2_GROUP                    ; r8d = group count = count / 32, count must be a whole number of 32-float groups; the 32 stays the group width: vis_hidden has no % 64 init validation, so a 64-wide guard would FATAL a future VLM the current code handles
     mov r9d, r8d                              ; r9d = raw group count, bit 0 = odd group left over, folded at the test below
     shr r8d, 1                                ; r8d = 64-float pair count, sets ZF for the next test
     jz .va_single_group                       ; no, process the one remaining 32-float group directly
     align 16                                  ; align loop entry for decode efficiency
.va_body64:
     VA_ADD32 0                                ; floats 0..31
     VA_ADD32 VA_UNROLL_STRIDE                 ; floats 32..63
     add rcx, VA_UNROLL_STRIDE * 2             ; dst += 64 floats, imm32: at a 256-byte stride -256 does not fit imm8 either, so the negative form buys nothing here
     add rdx, VA_UNROLL_STRIDE * 2             ; src += 64 floats, same
     dec r8d                                   ; one fewer pair
     jnz .va_body64                            ; continue 64-float loop, head-to-tail
     test r9b, 1                               ; bit 0 of the raw group count = odd 32-float group remaining?
     jz .va_done                               ; no: all groups processed
.va_single_group:
     VA_ADD32 0                                ; the one remaining 32-float group
; fall through to .va_done, CLEANUP_YMM + ret
.va_done:
     CLEANUP_YMM                               ; clean AVX state for ABI
     ret                                       ; return to caller, volatile regs only, no restore needed
; ---- %undef cleanup: keep the preprocessor namespace clean for later includes ----
%undef VA_UNROLL_STRIDE                        ; generic name, a later kernel would plausibly reuse it
%undef VA_LOG2_GROUP
%undef VA_YMM96
%undef VA_YMM64
%undef VA_YMM32
