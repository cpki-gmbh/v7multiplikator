; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define AK_UNROLL_STRIDE 128                   ; 4x YMM block stride, 32 floats
%define AK_BLK1_OFF 32                         ; byte offset to block1 within unroll group
%define AK_BLK2_OFF 64                         ; byte offset to block2 within unroll group
%define AK_BLK3_OFF 96                         ; byte offset to block3 within unroll group
%define AK_LOG2_GROUP 5                        ; log2,32 floats per unroll group for shr n/32

%macro AK_ADD_GROUP 1
     vmovups ymm0, [rcx+%1]                    ; x[block0]
     vmovups ymm1, [rcx+%1+AK_BLK1_OFF]        ; x[block1]
     vmovups ymm2, [rcx+%1+AK_BLK2_OFF]        ; x[block2]
     vmovups ymm3, [rcx+%1+AK_BLK3_OFF]        ; x[block3]
     vaddps ymm0, ymm0, [rdx+%1]               ; x[block0] + y[block0]
     vaddps ymm1, ymm1, [rdx+%1+AK_BLK1_OFF]   ; x[block1] + y[block1]
     vaddps ymm2, ymm2, [rdx+%1+AK_BLK2_OFF]   ; x[block2] + y[block2]
     vaddps ymm3, ymm3, [rdx+%1+AK_BLK3_OFF]   ; x[block3] + y[block3]
     vmovups [rcx+%1], ymm0                    ; store block0
     vmovups [rcx+%1+AK_BLK1_OFF], ymm1        ; store block1
     vmovups [rcx+%1+AK_BLK2_OFF], ymm2        ; store block2
     vmovups [rcx+%1+AK_BLK3_OFF], ymm3        ; store block3
%endmacro
     section .text
     global v7s_add_inplace                    ; export entry point
v7s_add_inplace:
; 4x unrolled vector loop: 32 floats, 128 bytes per group, two groups
; 64 floats per iteration of the pair loop
     shr r8d, AK_LOG2_GROUP                    ; r8d = n / 32, groups of 4 blocks still to process
     cmp r8d, 2                                ; fewer than two groups left?
     jb .ak_single_group                       ; yes: process the lone trailing group directly, r8d == 0 means n < 32, outside the contract
     align 16                                  ; align loop entry for decode efficiency
.ak_dbl_vec_loop:
     AK_ADD_GROUP 0                            ; x[0:32] += y[0:32]
; second 32-float group, addressed at +AK_UNROLL_STRIDE; rcx/rdx advance by two groups at the loop tail
     AK_ADD_GROUP AK_UNROLL_STRIDE             ; x[32:64] += y[32:64], same macro, displaced one group
     add rcx, AK_UNROLL_STRIDE*2               ; advance x by 2 groups, 256 bytes
     add rdx, AK_UNROLL_STRIDE*2               ; advance y by 2 groups, 256 bytes
     sub r8d, 2                                ; two groups consumed, r8d is now the REMAINING group count
     cmp r8d, 2                                ; at least two groups still left?, a 'sub' sets CF only on BORROW, so r8d-2 == 0 is a no-borrow CF=0 and a 'jae' straight off the sub would branch back on an EMPTY remainder and run the loop off the end of the buffer, the remainder must be re-tested against 2
     jae .ak_dbl_vec_loop                      ; yes: continue 64-float loop
     test r8d, r8d                             ; exactly one group left?
     jz .ak_done                               ; no: all done
.ak_single_group:
     AK_ADD_GROUP 0                            ; the lone trailing group, rcx/rdx already point at it
.ak_done:
     CLEANUP_YMM                               ; vzeroupper for AVX ABI compliance
     ret                                       ; return to caller, volatile regs only, no restore needed
; ---- %undef cleanup: keep the preprocessor namespace clean for later includes ----
%unmacro AK_ADD_GROUP 1                        ; remove: the three call sites above are its only expansions
%undef AK_LOG2_GROUP                           ; generic name, a later kernel would plausibly reuse it
%undef AK_BLK3_OFF
%undef AK_BLK2_OFF
%undef AK_BLK1_OFF
%undef AK_UNROLL_STRIDE
