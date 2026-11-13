; v7 - part of the v7 inference engine
; Copyright (C) 2026 Görg Pflug & CPKI GmbH
; License: GPL v3 (see LICENSE)

; sample comments lifted from the real src/asm tree -- the shapes the
; reformulator will face in production. Code side is dummy but plausible.
;
; full-line kernel comments
; the fused Q8 matmul processes 4 rows per group -- one vpmaddubsw pair per lane
; rope tables are built once at init -- positions advance by half the head dim
; the swiglu gate multiplies the activation by its sigmoid -- gate and up rows share one prefetch
; workers claim jobs with lock xadd -- the cursor carries the generation tag in bits 31:16
; quantize wraps every 64 floats into one group -- the scale is the reciprocal of the max abs value
;
; inline comments on plausible code
     mov eax, 12                                ; layer count (24 layers -- two passes each)
     imul rbx, 4                                ; stride in dwords -- one row of the param set
     vfmadd231ps ymm0, ymm1, ymm2               ; accumulate row sums -- xmm6 holds the scale
     vpbroadcastd ymm3, dword [rsi + 4]         ; broadcast the group scale -- uniform across 8 lanes
     lock xadd dword [rel next_job], edx        ; claim the next chunk -- the tag rides in edx
     cmp r8d, 64                                ; chunk width floor (4 rows -- never below the kernel block)
