; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .text
%define RO_FLOAT4_SCALE 4                      ; sizeof,float, scale for pair-indexed cos/sin/head addressing
%define RO_HALF_SCALE 2                        ; second-half offset scale: head_dim*2 bytes ==, head_dim/2 *4 bytes below the head base
%define RO_BLK_BYTES 32                        ; byte offset of the second 8-pair SIMD block inside one 16-pair iteration
%define RO_PAIRS_PER_ITER 16                   ; pairs consumed per loop iteration, 2 x the 8-pair SIMD width
%define RO_PAIRS_SHIFT 5                       ; log2,2 * RO_PAIRS_PER_ITER : head_dim/32 block count, 16 pairs x lower+upper float each
     SF_START v7s_rope_only_head               ; prologue: no non-volatile saves, leaf kernel, no calls, all 6 live loop values fit the 7 volatile GPRs, rax unused ; frame name == entry symbol, so no second global is emitted and .ro_loop/.roh_done anchor to v7s_rope_only_head
     SF_COMMIT                                 ; finalize stack frame layout
     cmp dword [rmsnorm_pos], 0                ; pos == 0?, folded load+test into single memory cmp
     je .roh_done                              ; yes, no rotation needed, cos=1, sin=0 => identity
; Load globals into volatile registers, leaf function: no calls, globals stable
     mov r9d, [rmsnorm_head_dim]               ; r9d = head dimension, volatile home, no non-volatile save
     mov r10, [rmsnorm_cos_table]              ; r10 = cos table pos-offset pointer, volatile home, no non-volatile save
     mov r11, [rmsnorm_sin_table]              ; r11 = sin table pos-offset pointer, volatile home, no non-volatile save
     imul rdx, r9                              ; element offset = head_idx * head_dim
     lea rdx, [rcx + rdx*RO_FLOAT4_SCALE]      ; rdx = &buf[head_idx * head_dim], buf consumed from rcx, the pair counter lives here afterwards
; head_dim stays in r9d: the second-half lea below derives the offset with SIB
; scale 2, head_dim*2 bytes, so no `shr` to half is needed
     lea r8, [rdx + r9*RO_HALF_SCALE]          ; r8 = second half of head, r9 zero-extended by the mov r9d at :16
     xor ecx, ecx                              ; pair counter
     shr r9d, RO_PAIRS_SHIFT                   ; r9d = 16-pair iteration count for this head
     align 32                                  ; align the hot back-edge target, entered once per 16-pair block
.ro_loop:
     vmovups ymm4, [r10 + rcx*RO_FLOAT4_SCALE] ; load cos values, early: overlaps data loads
     vmovups ymm5, [r11 + rcx*RO_FLOAT4_SCALE] ; load sin values, early: overlaps data loads
     vmovups ymm0, [rdx + rcx*RO_FLOAT4_SCALE] ; load lower half, pairs j..j+7
     vmovups ymm1, [r8 + rcx*RO_FLOAT4_SCALE]  ; load upper half, pairs j..j+7
     vmulps ymm2, ymm0, ymm4                   ; ymm2 = lower * cos, volatile, ymm6/7 are Win64 non-volatile, never overwritten here
     vmulps ymm3, ymm0, ymm5                   ; ymm3 = lower * sin, independent of the first FMA below
     vfnmadd231ps ymm2, ymm1, ymm5             ; ymm2 = lower*cos - upper*sin
     vfmadd231ps ymm3, ymm1, ymm4              ; ymm3 = lower*sin + upper*cos
     vmovups ymm0, [rdx + rcx*RO_FLOAT4_SCALE + RO_BLK_BYTES] ; lower half, pairs j+8..j+15
     vmovups ymm1, [r8 + rcx*RO_FLOAT4_SCALE + RO_BLK_BYTES] ; upper half, pairs j+8..j+15
     vmovups ymm4, [r10 + rcx*RO_FLOAT4_SCALE + RO_BLK_BYTES] ; cos values, pairs j+8..j+15, issued before the stores
     vmovups ymm5, [r11 + rcx*RO_FLOAT4_SCALE + RO_BLK_BYTES] ; sin values, pairs j+8..j+15
     vmovups [rdx + rcx*RO_FLOAT4_SCALE], ymm2 ; store rotated lower
     vmovups [r8 + rcx*RO_FLOAT4_SCALE], ymm3  ; store rotated upper
     vmulps ymm2, ymm0, ymm4                   ; ymm2 = lower * cos
     vmulps ymm3, ymm0, ymm5                   ; ymm3 = lower * sin, independent of the first FMA below
     vfnmadd231ps ymm2, ymm1, ymm5             ; ymm2 = lower*cos - upper*sin
     vfmadd231ps ymm3, ymm1, ymm4              ; ymm3 = lower*sin + upper*cos
     vmovups [rdx + rcx*RO_FLOAT4_SCALE + RO_BLK_BYTES], ymm2 ; store rotated lower
     vmovups [r8 + rcx*RO_FLOAT4_SCALE + RO_BLK_BYTES], ymm3 ; store rotated upper
     add ecx, RO_PAIRS_PER_ITER                ; advance one 16-pair block
     dec r9d                                   ; one block done, independent chain, removes the serial cmp-on-ecx recurrence
     jnz .ro_loop                              ; more blocks, continue, bottom-tested
.roh_done:
     CLEANUP_YMM                               ; clear YMM upper halves for ABI
     SF_END                                    ; restore frame pointer and return
; ---- %undef cleanup: keep the preprocessor namespace clean for later includes ----
%undef RO_BLK_BYTES
%undef RO_PAIRS_PER_ITER
%undef RO_HALF_SCALE
%undef RO_PAIRS_SHIFT
%undef RO_FLOAT4_SCALE
