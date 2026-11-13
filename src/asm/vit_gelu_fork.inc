; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; vit_gelu_fork.inc -- Forked ViT GELU: two independent contiguous ranges,
; original vit_gelu math unchanged. Call while v7s_set_spin(1) has declared the
; active inference phase; there is no asleep-worker or workerless bypass.

     section .text
; ============================================================================
; vit_gelu_fork -- split an in-place GELU between the caller and worker 0
; RCX = float buffer, EDX = total float count
; ============================================================================
     SF_START vit_gelu_fork                    ; prologue: frame only, no saved registers
     SF_COMMIT                                 ; finalize stack frame layout
     cmp edx, 4096                             ; below this, one core normally wins
     jb .direct                                ; keep the original direct call
; Split at a multiple of 32 floats to preserve the existing vector-tail behavior.
     FORK 0                                    ; publish RCX/EDX to worker 0, parent ZF=1, child ZF=0
     jz .main                                  ; parent continues with the lower range
; W0 branch: second range. RCX=buffer, EDX=total count, transferred by FORK.
     mov r9d, edx                              ; r9d = unsigned float count
     and r9d, -64                              ; discard the low 6 bits
     shr r9d, 1                                ; half = floor(count/64)*32 floats
     lea rcx, [rcx + r9*VK_F4]                 ; rcx = buffer + half floats, byte address
     sub edx, r9d                              ; edx = upper range count, keeps the 32-float alignment
     call vit_gelu                             ; GELU over the upper range, unchanged math
     CONTINUE_DISPATCH 0                       ; return worker 0 to its spin loop, never falls through
.main:
     and edx, -64                              ; round down to 64 floats
     shr edx, 1                                ; half, multiple of 32 floats
     call vit_gelu                             ; caller processes the first range
     FORK_WAIT 0                               ; join worker 0 before the next vision matmul
     SF_END                                    ; restore frame and return
.direct:
     call vit_gelu                             ; original direct GELU call, unchanged behavior
     SF_END                                    ; restore frame and return
