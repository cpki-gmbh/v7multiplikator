; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%include "vit_structs.inc"
%include "inf_structs.inc"
%define VLC_TAIL_THRESH4 31                    ; slow path: need d + 31 < K to safely process 32 floats, 4 accumulators
%define VLC_16F_BYTES 64                       ; 16 floats * 4 bytes, offset for d+16 in 4-accumulator K-loop
%define VLC_24F_BYTES 96                       ; 24 floats * 4 bytes, offset for d+24 in 4-accumulator K-loop
%define VLC_SCALE2 2                           ; element scale 2 within a 4-column block, bias/output
%define VLC_SCALE3 3                           ; element scale 3 within a 4-column block, bias/output
%define VLC_WR_SCALE2 2                        ; weight-row scale 2, 2 * K*4 bytes within a block
%define VLC_WR_SCALE4 4                        ; weight-row scale 4, next 4-column block, 4 * K*4 bytes
%define VLC_PF_DIST 64                         ; P5: prefetch distance in bytes, one cache line ahead

%macro VLC_K_LOOP_BODY 1
%1:                                            ; 32-wide K-loop head, site label %1
     vmovups ymm1, [r12 + rcx*VK_F4]           ; load 8 activation floats, block 0
     vmovups ymm2, [r12 + rcx*VK_F4 + VK_8F_BYTES] ; stage activation block 1
     vfmadd231ps ymm3, ymm1, [rsi + rcx*VK_F4] ; acc0 += act * wt, weight operand folded into the FMA
     vfmadd231ps ymm4, ymm2, [rsi + rcx*VK_F4 + VK_8F_BYTES] ; acc1 += act * wt
     vmovups ymm1, [r12 + rcx*VK_F4 + VLC_16F_BYTES] ; load activation block 2
     vmovups ymm2, [r12 + rcx*VK_F4 + VLC_24F_BYTES] ; stage activation block 3
     vfmadd231ps ymm5, ymm1, [rsi + rcx*VK_F4 + VLC_16F_BYTES] ; acc2 += act * wt
     vfmadd231ps ymm6, ymm2, [rsi + rcx*VK_F4 + VLC_24F_BYTES] ; acc3 += act * wt
     add ecx, 32                               ; d += 32, processed four 8-float blocks
     cmp ecx, edx                              ; d < K-31?, precomputed bound <=> d+32 <= K
     jl %1                                     ; yes, continue 32-wide loop
%endmacro

%macro VLC_K_TAIL_BODY 2
%1:                                            ; scalar remainder head, site label %1
     cmp ecx, r15d                             ; d >= K?
     jge %2                                    ; yes, remainder complete
%%tail_work:
     vmovss xmm1, [r12 + rcx*VK_F4]            ; load 1 act float
     vfmadd231ss xmm7, xmm1, [rsi + rcx*VK_F4] ; accumulate act * wt, weight operand folded into the FMA
     inc ecx                                   ; advance d
     cmp ecx, r15d                             ; next element still within K?
     jl %%tail_work                            ; bottom test avoids an unconditional taken jump
%2:                                            ; remainder exit, site label %2
%endmacro

%macro VLC_REDUCE4 0
     vaddps ymm3, ymm3, ymm4                   ; acc01 = acc0 + acc1
     vaddps ymm5, ymm5, ymm6                   ; acc23 = acc2 + acc3
     vaddps ymm3, ymm3, ymm5                   ; acc0123 = acc01 + acc23 -> ymm3
     vextractf128 xmm2, ymm3, 1                ; extract upper 128 bits to xmm2
     vaddps xmm3, xmm3, xmm2                   ; add upper + lower 128 -> 4 lanes remain
     vmovhlps xmm2, xmm3, xmm3                 ; hi 64 -> lo 64, [2,3,2,3] like 0xEE
     vaddps xmm3, xmm3, xmm2                   ; add -> 2 lanes remain, p0+p2, p1+p3
     vmovshdup xmm2, xmm3                      ; lane 1 -> lane 0 in lane0, [1,1,3,3], only lane0 consumed
     vaddss xmm3, xmm3, xmm2                   ; xmm3[0] =, p0+p2 +,p1+p3 = full 8-wide sum
     vaddss xmm0, xmm0, xmm3                   ; acc += full dot product, xmm3 lane 0
     vaddss xmm0, xmm0, xmm7                   ; acc += remainder dot product, 0 if K%32==0
%endmacro

%macro VLC4_REDUCE_ALL 1                       ; %1 = 0: bias seed in xmm10..xmm13; 1: no bias
     vaddps ymm0, ymm0, ymm1                   ; col j+0: even + odd accumulators
     vaddps ymm2, ymm2, ymm3                   ; col j+1
     vaddps ymm4, ymm4, ymm5                   ; col j+2
     vaddps ymm6, ymm6, ymm7                   ; col j+3
     vextractf128 xmm1, ymm0, 1                ; col j+0: upper 128
     vextractf128 xmm3, ymm2, 1                ; col j+1
     vextractf128 xmm5, ymm4, 1                ; col j+2
     vextractf128 xmm7, ymm6, 1                ; col j+3
     vaddps xmm0, xmm0, xmm1                   ; col j+0: merge 8 -> 4
     vaddps xmm2, xmm2, xmm3                   ; col j+1
     vaddps xmm4, xmm4, xmm5                   ; col j+2
     vaddps xmm6, xmm6, xmm7                   ; col j+3
     vmovhlps xmm1, xmm0, xmm0                 ; col j+0: high 64 -> low
     vmovhlps xmm3, xmm2, xmm2                 ; col j+1
     vmovhlps xmm5, xmm4, xmm4                 ; col j+2
     vmovhlps xmm7, xmm6, xmm6                 ; col j+3
     vaddps xmm0, xmm0, xmm1                   ; col j+0: merge 4 -> 2
     vaddps xmm2, xmm2, xmm3                   ; col j+1
     vaddps xmm4, xmm4, xmm5                   ; col j+2
     vaddps xmm6, xmm6, xmm7                   ; col j+3
     vmovshdup xmm1, xmm0                      ; col j+0: lane 1 -> lane 0
     vmovshdup xmm3, xmm2                      ; col j+1
     vmovshdup xmm5, xmm4                      ; col j+2
     vmovshdup xmm7, xmm6                      ; col j+3
     vaddss xmm0, xmm0, xmm1                   ; col j+0: full 8-wide sum
     vaddss xmm2, xmm2, xmm3                   ; col j+1
     vaddss xmm4, xmm4, xmm5                   ; col j+2
     vaddss xmm6, xmm6, xmm7                   ; col j+3
%if %1 = 0
     vaddss xmm10, xmm10, xmm0                 ; bias seed + full dot, col j+0
     vaddss xmm11, xmm11, xmm2                 ; col j+1
     vaddss xmm12, xmm12, xmm4                 ; col j+2
     vaddss xmm13, xmm13, xmm6                 ; col j+3
%else
     vmovaps xmm10, xmm0                       ; no-bias: move full dot directly, col j+0
     vmovaps xmm11, xmm2                       ; col j+1
     vmovaps xmm12, xmm4                       ; col j+2
     vmovaps xmm13, xmm6                       ; col j+3
%endif
%endmacro
     section .rdata align=32
vk_one_f:
     dd 1.0                                    ; 1.0 for softmax 1/sum full-precision division
vk_neg_inf:
     dd 0xFF800000                             ; -inf IEEE-754 bits, softmax max init
     section .text
     global v7s_vit_linear_chunk               ; export ViT linear chunk kernel symbol
v7s_vit_linear_chunk:
     SF_START v7s_vit_linear_chunk, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles
     SF_DWORD vlc_n                            ; n_rows
     SF_DWORD vlc_cstart                       ; col_start
     SF_DWORD vlc_cd                           ; chunk_d
     SF_QWORD vlc_out                          ; output base pointer
     SF_QWORD vlc_in                           ; input base pointer
     SF_QWORD vlc_w                            ; weight base pointer
     SF_QWORD vlc_bias                         ; bias base, or NULL
     SF_XMM vlc_xmm6                           ; xmm6 low half spill slot, slow acc3 / fast j+3 even
     SF_XMM vlc_xmm7                           ; xmm7 low half spill slot, slow K-remainder acc / fast j+3 odd
     SF_XMM vlc_xmm10                          ; xmm10 low half spill slot, fast-path bias seed j+0
     SF_XMM vlc_xmm11                          ; xmm11 low half spill slot, fast-path bias seed j+1
     SF_XMM vlc_xmm12                          ; xmm12 low half spill slot, fast-path bias seed j+2
     SF_XMM vlc_xmm13                          ; xmm13 low half spill slot, fast-path bias seed j+3
     SF_XMM vlc_xmm14                          ; xmm14 low half spill slot, fast-path shared activation lo
     SF_XMM vlc_xmm15                          ; xmm15 low half spill slot, fast-path shared activation hi
     SF_COMMIT                                 ; commit stack frame, allocate shadow space
     vmovdqu vlc_xmm6, xmm6                    ; save xmm6 low half, slow acc3 / fast j+3 even, NON-VOLATILE per Win64
     vmovdqu vlc_xmm7, xmm7                    ; save xmm7 low half, slow K-remainder acc / fast j+3 odd, NON-VOLATILE per Win64
     vmovdqu vlc_xmm10, xmm10                  ; save xmm10 low half, fast-path bias seed j+0
     vmovdqu vlc_xmm11, xmm11                  ; save xmm11 low half, fast-path bias seed j+1
     vmovdqu vlc_xmm12, xmm12                  ; save xmm12 low half, fast-path bias seed j+2
     vmovdqu vlc_xmm13, xmm13                  ; save xmm13 low half, fast-path bias seed j+3
     vmovdqu vlc_xmm14, xmm14                  ; save xmm14 low half, fast-path shared activation lo
     vmovdqu vlc_xmm15, xmm15                  ; save xmm15 low half, fast-path shared activation hi
     mov vlc_out, rcx                          ; save output base
     mov vlc_in, rdx                           ; save input base
     mov vlc_w, r8                             ; save weight base
     mov vlc_bias, r9                          ; save bias base
     mov eax, [rbp + VK_ARG5]                  ; load n_rows
     mov vlc_n, eax                            ; save n_rows
     mov eax, [rbp + VK_ARG6]                  ; load col_start
     mov vlc_cstart, eax                       ; save col_start
     mov eax, [rbp + VK_ARG7]                  ; load chunk_d
     mov vlc_cd, eax                           ; save chunk_d
     mov r8d, [rel g_vit_m]                    ; r8d = M
     shl r8d, 2                                ; r8d = M*4, the output-row stride. This value is unchanged during the K loop.
     mov r15d, [rel g_vit_k]                   ; r15d = K, zero-extends into r15; register, constant across all loops
     lea r13, [r15*VK_F4]                      ; r13 = K*4, input row + weight row byte stride, 64-bit stride addend
     mov eax, vlc_cstart                       ; col_start, zero-extends into rax
     imul rax, r15                             ; rax = col_start * K, element offset into the weight rows
     shl rax, 2                                ; byte offset, 64-bit
     add rax, vlc_w                            ; rax = &w[col_start][0]
     mov vlc_w, rax                            ; vlc_w = chunk weight base, constant across rows
     mov eax, vlc_cstart                       ; col_start, compute col_start*4 ONCE for bias + output paths
     shl eax, 2                                ; eax = col_start * sizeof,float, reused below
     mov rdi, vlc_bias                         ; load bias base, or NULL
     test rdi, rdi                             ; bias NULL?
     jz .vlc_no_bias_init                      ; keep NULL
     add rdi, rax                              ; rdi = &bias[col_start], reuse col_start*4 from eax
.vlc_no_bias_init:
     mov vlc_bias, rdi                         ; vlc_bias = chunk bias base, constant across rows
; Initialize RUNNING row pointers, advanced by stride at row-loop bottom
     mov r12, vlc_in                           ; r12 = &in[0][0], running input row ptr
     mov r14, vlc_out                          ; r14 = &out[0][0], output base
     add r14, rax                              ; r14 = &out[0][col_start], reuse col_start*4
     xor r11d, r11d                            ; r11d = r = 0, row counter in volatile register
     mov r10d, vlc_cd                          ; r10d = chunk_d, register compare target
     mov r9d, vlc_n                            ; r9d = n_rows, register compare target
     test r15b, 15                             ; K a multiple of 16?
     jnz .vlc_slow_path                        ; no, original loops, K-tail required
     test r10b, 3                              ; chunk_d a multiple of 4?
     jnz .vlc_slow_path                        ; no, original loops, ragged block
     sub r15d, 16                              ; K-16: fast-path loop bound, d <= K-16 <=> d+16 <= K ; slow path branched away earlier, its K untouched
     test rdi, rdi                             ; bias pointer NULL?, fast-path variant split
     jz .vlc4_row_loop_no_bias                 ; NULL, fast no-bias variant
     jmp .vlc4_row_loop_bias                   ; non-NULL, fast bias variant
.vlc_slow_path:
     lea edx, [r15d - VLC_TAIL_THRESH4]        ; edx = K-31, the VLC_K_LOOP_BODY bound: d < K-31 <=> d+31 < K
     test rdi, rdi                             ; bias pointer NULL?, checked ONCE per call
     jnz .vlc_sp_bias                          ; bias slow path, residual split applied inside
; no-bias slow path: a fused residual is only ever set together with a non-NULL bias, so the no-bias arm is always residual-free
     cmp qword [rel g_vit_residual], 0         ; fused residual add enabled?
     je .vlc_row_loop_no_bias                  ; no: no-bias, residual-free variant
.vlc_sp_bias:
     cmp qword [rel g_vit_residual], 0         ; fused residual add enabled?
     je .vlc_row_loop_bias                     ; no, bias, no-residual variant; residual falls through
.vlc_row_loop_bias_resid:
     mov rsi, vlc_w                            ; rsi = &w[col_start][0], reset weight ptr per row
     mov rdi, vlc_bias                         ; rdi = &bias[col_start], reset bias ptr per row
     xor ebx, ebx                              ; ebx = j = 0, column index, ebx, not edi which aliases rdi=bias ptr
.vlc_col_loop_bias_resid:
; Initialize accumulator with bias, bias guaranteed non-NULL in this variant
     vmovss xmm0, [rdi]                        ; acc = bias[col_start + j]
; K-loop: 4 independent accumulators x 8 floats = 32/iteration
     vxorps ymm3, ymm3, ymm3                   ; acc0 = 0, d=0..7
     vxorps ymm4, ymm4, ymm4                   ; acc1 = 0, d=8..15
     vxorps ymm5, ymm5, ymm5                   ; acc2 = 0, d=16..23
     vxorps ymm6, ymm6, ymm6                   ; acc3 = 0, d=24..31
     xor ecx, ecx                              ; ecx = d = 0, K-loop index, volatile, safe inside K-loop only
     VLC_K_LOOP_BODY .vlc_k_loop_bias          ; 32-wide K loop
; Scalar remainder: process remaining K%32 elements into xmm7, separate from ymm3-6
     vxorps xmm7, xmm7, xmm7                   ; remainder accumulator = 0
     VLC_K_TAIL_BODY .vlc_k_tail_bias, .vlc_k_tail_bias_done ; bias variant: scalar K%32 remainder
; Combine 4 accumulators and reduce to scalar via shuffle tree
     VLC_REDUCE4                               ; acc0-3 -> scalar, added into the bias seed, xmm0
; Store result, resid variant: g_vit_residual != 0 for the whole call
     vaddss xmm0, xmm0, [r14 + rbx*VK_F4]      ; acc += residual[r][col_start+j], read-modify-write on out
     vmovss [r14 + rbx*VK_F4], xmm0            ; out[r][col_start+j] = acc
; Advance pointers for next column, no imul, just add precomputed stride
     add rsi, r13                              ; rsi += K*4, next weight row
     add rdi, VK_F4                            ; rdi += 4, next bias entry, no NULL check, variant guarantees bias
     inc ebx                                   ; j++
     cmp ebx, r10d                             ; j < chunk_d?, register compare
     jl .vlc_col_loop_bias_resid               ; next column
; Advance RUNNING row pointers by precomputed strides, no imul per row
     add r12, r13                              ; r12 = &in[r+1][0], input row stride K*4
     add r14, r8                               ; r14 = &out[r+1][col_start], output row stride M*4
     inc r11d                                  ; r11d = r++
     cmp r11d, r9d                             ; r < n_rows?, register compare
     jl .vlc_row_loop_bias_resid               ; next row
     jmp .vlc_done                             ; all rows done, bias+resid variant
.vlc_row_loop_bias:
     mov rsi, vlc_w                            ; rsi = &w[col_start][0], reset weight ptr per row
     mov rdi, vlc_bias                         ; rdi = &bias[col_start], reset bias ptr per row
     xor ebx, ebx                              ; ebx = j = 0, column index, ebx, not edi which aliases rdi=bias ptr
.vlc_col_loop_bias:
; Initialize accumulator with bias, bias guaranteed non-NULL in this variant
     vmovss xmm0, [rdi]                        ; acc = bias[col_start + j]
; K-loop: 4 independent accumulators x 8 floats = 32/iteration
     vxorps ymm3, ymm3, ymm3                   ; acc0 = 0, d=0..7
     vxorps ymm4, ymm4, ymm4                   ; acc1 = 0, d=8..15
     vxorps ymm5, ymm5, ymm5                   ; acc2 = 0, d=16..23
     vxorps ymm6, ymm6, ymm6                   ; acc3 = 0, d=24..31
     xor ecx, ecx                              ; ecx = d = 0, K-loop index, volatile, safe inside K-loop only
     VLC_K_LOOP_BODY .vlc_k_loop_bnr           ; 32-wide K loop
; Scalar remainder: process remaining K%32 elements into xmm7, separate from ymm3-6
     vxorps xmm7, xmm7, xmm7                   ; remainder accumulator = 0
     VLC_K_TAIL_BODY .vlc_k_tail_bnr, .vlc_k_tail_bnr_done ; bnr variant: scalar K%32 remainder
; Combine 4 accumulators and reduce to scalar via shuffle tree
     VLC_REDUCE4                               ; acc0-3 -> scalar, added into the bias seed, xmm0
; Store result, no-resid variant: check hoisted per call, out fully overwritten, no read
     vmovss [r14 + rbx*VK_F4], xmm0            ; out[r][col_start+j] = acc
; Advance pointers for next column, no imul, just add precomputed stride
     add rsi, r13                              ; rsi += K*4, next weight row
     add rdi, VK_F4                            ; rdi += 4, next bias entry, no NULL check, variant guarantees bias
     inc ebx                                   ; j++
     cmp ebx, r10d                             ; j < chunk_d?, register compare
     jl .vlc_col_loop_bias                     ; next column
; Advance RUNNING row pointers by precomputed strides, no imul per row
     add r12, r13                              ; r12 = &in[r+1][0], input row stride K*4
     add r14, r8                               ; r14 = &out[r+1][col_start], output row stride M*4
     inc r11d                                  ; r11d = r++
     cmp r11d, r9d                             ; r < n_rows?, register compare
     jl .vlc_row_loop_bias                     ; next row
     jmp .vlc_done                             ; all rows done, bias, no-resid variant
.vlc_row_loop_no_bias:
     mov rsi, vlc_w                            ; rsi = &w[col_start][0], reset weight ptr per row
     xor ebx, ebx                              ; ebx = j = 0, column index
.vlc_col_loop_no_bias:
; Initialize accumulator with zero, no bias in this variant
     vxorps xmm0, xmm0, xmm0                   ; acc = 0
; K-loop: 4 independent accumulators x 8 floats = 32/iteration
     vxorps ymm3, ymm3, ymm3                   ; acc0 = 0, d=0..7
     vxorps ymm4, ymm4, ymm4                   ; acc1 = 0, d=8..15
     vxorps ymm5, ymm5, ymm5                   ; acc2 = 0, d=16..23
     vxorps ymm6, ymm6, ymm6                   ; acc3 = 0, d=24..31
     xor ecx, ecx                              ; ecx = d = 0, K-loop index, volatile, safe inside K-loop only
     VLC_K_LOOP_BODY .vlc_k_loop_no_bias       ; 32-wide K loop
; Scalar remainder: process remaining K%32 elements into xmm7, separate from ymm3-6
     vxorps xmm7, xmm7, xmm7                   ; remainder accumulator = 0
     VLC_K_TAIL_BODY .vlc_k_tail_nb, .vlc_k_tail_nb_done ; nb variant: scalar K%32 remainder
; Combine 4 accumulators and reduce to scalar via shuffle tree
     VLC_REDUCE4                               ; acc0-3 -> scalar, added into the zero seed, xmm0
; Store result, no-resid variant: check hoisted per call, out fully overwritten, no read
     vmovss [r14 + rbx*VK_F4], xmm0            ; out[r][col_start+j] = acc
; Advance pointers for next column, no imul, just add precomputed stride
     add rsi, r13                              ; rsi += K*4, next weight row
     inc ebx                                   ; j++
     cmp ebx, r10d                             ; j < chunk_d?, register compare
     jl .vlc_col_loop_no_bias                  ; next column
; Advance RUNNING row pointers by precomputed strides, no imul per row
     add r12, r13                              ; r12 = &in[r+1][0], input row stride K*4
     add r14, r8                               ; r14 = &out[r+1][col_start], output row stride M*4
     inc r11d                                  ; r11d = r++
     cmp r11d, r9d                             ; r < n_rows?, register compare
     jl .vlc_row_loop_no_bias                  ; next row
     jmp .vlc_done                             ; all rows done, slow no-bias, must jump: the
; P0 fast path follows; falling through would enter
; it with a stale/NULL bias pointer, validator crash
.vlc4_row_loop_bias:
     mov rsi, vlc_w                            ; rsi = &w[col_start + j][0], weight row of column j+0
     mov rdi, vlc_bias                         ; rdi = &bias[col_start + j]
     xor ebx, ebx                              ; ebx = j = 0, 4-column block start
; VLC4_ZERO_AND_ROW_PTRS -- zero the eight column accumulators and derive the
; four weight-row bases of a 4-column block. IN: rsi = &w[j][0], r13 = weight
; row stride. OUT: ymm0-7 = 0, rdx/rax/r9 = &w[j+1..3][0], ecx = 0.

%macro VLC4_ZERO_AND_ROW_PTRS 0
     vxorps ymm0, ymm0, ymm0                   ; acc col j+0 even, d=0,16,...
     vxorps ymm1, ymm1, ymm1                   ; acc col j+0 odd, d=8,24,...
     vxorps ymm2, ymm2, ymm2                   ; acc col j+1 even
     vxorps ymm3, ymm3, ymm3                   ; acc col j+1 odd
     vxorps ymm4, ymm4, ymm4                   ; acc col j+2 even
     vxorps ymm5, ymm5, ymm5                   ; acc col j+2 odd
     vxorps ymm6, ymm6, ymm6                   ; acc col j+3 even
     vxorps ymm7, ymm7, ymm7                   ; acc col j+3 odd
     lea rdx, [rsi + r13]                      ; rdx = &w[j+1][0], weight rows are K*4 apart
     lea rax, [rsi + r13*VLC_WR_SCALE2]        ; rax = &w[j+2][0]
     lea r9, [rax + r13]                       ; r9 = &w[j+3][0], rsi + 3*K*4, must be rsi-based: rdx+r13 is only 2 rows!
     xor ecx, ecx                              ; ecx = d = 0, K-loop index
%endmacro
; VLC4_FMA_STEP -- one 16-wide K iteration of the 4-column accumulation.
; IN/OUT: rsi/rdx/rax/r9 = weight row bases, r12 = activation row,
; rcx = K offset, r15d = vectorized K bound, ymm0-7 = accumulators.

%macro VLC4_FMA_STEP 0
%%k_loop:
; P5: prefetch next iteration's weight + activation cache lines, hint only --
; never faults, no precision impact.Distance 64B = one cache line ahead.
     ds prefetcht0 [rsi + rcx*VK_F4 + VLC_PF_DIST] ; w[j+0] next line
     vmovups ymm14, [r12 + rcx*VK_F4]          ; act in[r][d..d+7], shared by all 4 columns
     vmovups ymm15, [r12 + rcx*VK_F4 + VK_8F_BYTES] ; act in[r][d+8..d+15]
     ds prefetcht0 [r12 + rcx*VK_F4 + VLC_PF_DIST] ; act in[r] next line
     vfmadd231ps ymm0, ymm14, [rsi + rcx*VK_F4] ; col j+0 even += act * w[j+0][d..d+7]
     vfmadd231ps ymm1, ymm15, [rsi + rcx*VK_F4 + VK_8F_BYTES] ; col j+0 odd += act * w[j+0][d+8..d+15]
     vfmadd231ps ymm2, ymm14, [rdx + rcx*VK_F4] ; col j+1 even += act * w[j+1][d..d+7]
     vfmadd231ps ymm3, ymm15, [rdx + rcx*VK_F4 + VK_8F_BYTES] ; col j+1 odd += act * w[j+1][d+8..d+15]
     ds prefetcht0 [rax + rcx*VK_F4 + VLC_PF_DIST] ; w[j+2] next line
     vfmadd231ps ymm4, ymm14, [rax + rcx*VK_F4] ; col j+2 even += act * w[j+2][d..d+7]
     vfmadd231ps ymm5, ymm15, [rax + rcx*VK_F4 + VK_8F_BYTES] ; col j+2 odd += act * w[j+2][d+8..d+15]
     vfmadd231ps ymm6, ymm14, [r9 + rcx*VK_F4] ; col j+3 even += act * w[j+3][d..d+7]
     vfmadd231ps ymm7, ymm15, [r9 + rcx*VK_F4 + VK_8F_BYTES] ; col j+3 odd += act * w[j+3][d+8..d+15]
     add ecx, 16                               ; d += 16
     cmp ecx, r15d                             ; d <= K-16?, r15d pre-reduced; <=> d+16 <= K
     jle %%k_loop                              ; continue 16-wide loop
%endmacro
.vlc4_col_loop_bias:
     vmovss xmm10, [rdi]                       ; bias[j+0], scalar result seed, low lane
     vmovss xmm11, [rdi + VK_F4]               ; bias[j+1]
     vmovss xmm12, [rdi + VK_F4*VLC_SCALE2]    ; bias[j+2]
     vmovss xmm13, [rdi + VK_F4*VLC_SCALE3]    ; bias[j+3]
     VLC4_ZERO_AND_ROW_PTRS                    ; zero accumulators, derive the four weight rows
.vlc4_k_loop_bias:
     VLC4_FMA_STEP                             ; one 16-wide K iteration
     VLC4_REDUCE_ALL 0                         ; interleaved reduce of all four columns, bias seed
; All four residual READS of out[...] precede all four stores.
     cmp qword [rel g_vit_residual], 0         ; fused residual add enabled?
     je .vlc4_bias_resid_skip                  ; no, skip residual adds for all 4 columns
     vaddss xmm10, xmm10, [r14 + rbx*VK_F4]    ; col j+0 += residual
     vaddss xmm11, xmm11, [r14 + rbx*VK_F4 + VK_F4] ; col j+1 += residual
     vaddss xmm12, xmm12, [r14 + rbx*VK_F4 + VK_F4*VLC_SCALE2] ; col j+2 += residual
     vaddss xmm13, xmm13, [r14 + rbx*VK_F4 + VK_F4*VLC_SCALE3] ; col j+3 += residual
.vlc4_bias_resid_skip:
     vmovss [r14 + rbx*VK_F4], xmm10           ; out[r][col_start + j+0] = acc
     vmovss [r14 + rbx*VK_F4 + VK_F4], xmm11   ; out[r][col_start + j+1] = acc
     vmovss [r14 + rbx*VK_F4 + VK_F4*VLC_SCALE2], xmm12 ; out[r][col_start + j+2] = acc
     vmovss [r14 + rbx*VK_F4 + VK_F4*VLC_SCALE3], xmm13 ; out[r][col_start + j+3] = acc
; Advance to the next 4-column block
     lea rsi, [rsi + r13*VLC_WR_SCALE4]        ; rsi = &w[j+4][0], four weight rows down
     add rdi, VK_F4*4                          ; rdi = &bias[j+4]
     add ebx, 4                                ; j += 4
     cmp ebx, r10d                             ; j < chunk_d?
     jl .vlc4_col_loop_bias                    ; next 4-column block
.vlc4_row_bottom_bias:
     add r12, r13                              ; r12 = &in[r+1][0]
     add r14, r8                               ; r14 = &out[r+1][col_start]
     inc r11d                                  ; r11d = r++
     mov r9d, vlc_n                            ; reload n_rows, register compare target
     cmp r11d, r9d                             ; r < n_rows?
     jl .vlc4_row_loop_bias                    ; next row
     jmp .vlc_done                             ; all rows done, fast bias variant
.vlc4_row_loop_no_bias:
     mov rsi, vlc_w                            ; rsi = &w[col_start + j][0], weight row of column j+0
     xor ebx, ebx                              ; ebx = j = 0, 4-column block start
.vlc4_col_loop_no_bias:
; No top guard, same proof as the bias variant : every entry has
; ebx <= chunk_d-4; the bottom jl is the sole loop exit.
     VLC4_ZERO_AND_ROW_PTRS                    ; zero accumulators, derive the four weight rows
.vlc4_k_loop_no_bias:
     VLC4_FMA_STEP                             ; one 16-wide K iteration
     VLC4_REDUCE_ALL 1                         ; interleaved reduce of all four columns, no bias
     cmp qword [rel g_vit_residual], 0         ; fused residual add enabled?
     je .vlc4_nb_resid_skip                    ; no, skip residual adds for all 4 columns
     vaddss xmm10, xmm10, [r14 + rbx*VK_F4]    ; col j+0 += residual
     vaddss xmm11, xmm11, [r14 + rbx*VK_F4 + VK_F4] ; col j+1 += residual
     vaddss xmm12, xmm12, [r14 + rbx*VK_F4 + VK_F4*VLC_SCALE2] ; col j+2 += residual
     vaddss xmm13, xmm13, [r14 + rbx*VK_F4 + VK_F4*VLC_SCALE3] ; col j+3 += residual
.vlc4_nb_resid_skip:
     vmovss [r14 + rbx*VK_F4], xmm10           ; out[r][col_start + j+0] = acc
     vmovss [r14 + rbx*VK_F4 + VK_F4], xmm11   ; out[r][col_start + j+1] = acc
     vmovss [r14 + rbx*VK_F4 + VK_F4*VLC_SCALE2], xmm12 ; out[r][col_start + j+2] = acc
     vmovss [r14 + rbx*VK_F4 + VK_F4*VLC_SCALE3], xmm13 ; out[r][col_start + j+3] = acc
     lea rsi, [rsi + r13*VLC_WR_SCALE4]        ; rsi = &w[j+4][0]
     add ebx, 4                                ; j += 4
     cmp ebx, r10d                             ; j < chunk_d?
     jl .vlc4_col_loop_no_bias                 ; next 4-column block
.vlc4_row_bottom_no_bias:
     add r12, r13                              ; r12 = &in[r+1][0]
     add r14, r8                               ; r14 = &out[r+1][col_start]
     inc r11d                                  ; r11d = r++
     mov r9d, vlc_n                            ; reload n_rows, register compare target
     cmp r11d, r9d                             ; r < n_rows?
     jl .vlc4_row_loop_no_bias                 ; next row
.vlc_done:
     vzeroupper                                ; clean AVX state before restoring callee-saved XMM lows
     vmovdqu xmm10, vlc_xmm10                  ; restore xmm10 low half, fast-path bias seed j+0
     vmovdqu xmm11, vlc_xmm11                  ; restore xmm11 low half, fast-path bias seed j+1
     vmovdqu xmm12, vlc_xmm12                  ; restore xmm12 low half, fast-path bias seed j+2
     vmovdqu xmm13, vlc_xmm13                  ; restore xmm13 low half, fast-path bias seed j+3
     vmovdqu xmm14, vlc_xmm14                  ; restore xmm14 low half, fast-path shared activation lo
     vmovdqu xmm15, vlc_xmm15                  ; restore xmm15 low half, fast-path shared activation hi
     vmovdqu xmm6, vlc_xmm6                    ; restore xmm6 low half, ABI: non-volatile; disjoint from 10-15, order irrelevant
     vmovdqu xmm7, vlc_xmm7                    ; restore xmm7 low half, ABI: non-volatile
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore non-volatiles and return
     global v7s_vit_attention_chunk            ; export unified ViT attention kernel
; VAC_REDUCE_SCORE -- shared score reduce of both attention variants: sum the
; four partial accumulators to one scalar, scale it, store scores[j], and
; update the running max. IN: ymm0-3 = dot partials, vac_scale, rdi/rbx =
; scores cursor, xmm4 = running max. Clobbers xmm0/xmm1.

%macro VAC_REDUCE_SCORE 0
     vaddps ymm0, ymm0, ymm1                   ; acc0 + acc1
     vaddps ymm2, ymm2, ymm3                   ; acc2 + acc3
     vaddps ymm0, ymm0, ymm2                   ; full head-wide sum
     vextractf128 xmm1, ymm0, 1                ; extract upper 128
     vaddps xmm0, xmm0, xmm1                   ; add upper + lower -> 4 lanes
     vmovhlps xmm1, xmm0, xmm0                 ; hi 64 -> lo 64
     vaddps xmm0, xmm0, xmm1                   ; add -> 2 lanes, p0+p2, p1+p3
     vmovshdup xmm1, xmm0                      ; lane 1 -> lane 0, p1+p3
     vaddss xmm0, xmm0, xmm1                   ; xmm0[0] = full head-wide sum
     vmulss xmm0, xmm0, vac_scale              ; score = dot * 1/sqrt(head_dim) from VCFG
     vmovss [rdi + rbx*VK_F4], xmm0            ; scores[j] = score, indexed by j, ebx is j, zeroed at the loop head; rdi stays vac_scratch for the whole pass
     vmaxss xmm4, xmm4, xmm0                   ; xmm4 = max,score, running_max, reg->reg, 1c latency
%endmacro
v7s_vit_attention_chunk:
     SF_START v7s_vit_attention_chunk, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles
     SF_XMM vac_xmm6                           ; xmm6 low half spill slot, Q register
     SF_XMM vac_xmm7                           ; xmm7 low half spill slot, Q register
     SF_XMM vac_xmm8                           ; xmm8 low half spill slot, Q register
     SF_XMM vac_xmm9                           ; xmm9 low half spill slot, 72-path vmin/max + weight broadcast
     SF_XMM vac_xmm10                          ; xmm10 low half spill slot, softmax parallel sum accumulator
     SF_XMM vac_xmm11                          ; xmm11 low half spill slot, Q register
     SF_XMM vac_xmm12                          ; xmm12 low half spill slot, Q register
     SF_XMM vac_xmm13                          ; xmm13 low half spill slot, Q register
     SF_XMM vac_xmm14                          ; xmm14 low half spill slot, 72-path Q remainder hoist
     SF_DWORD vac_h                            ; head index
     SF_QWORD vac_attn                         ; attn output base
     SF_QWORD vac_qkv                          ; qkv buffer base
     SF_QWORD vac_scratch                      ; scores scratch
     SF_DWORD vac_max                          ; softmax max, float stored as dword
     SF_DWORD vac_invsum                       ; 1/softmax sum, refined
     SF_DWORD vac_scale                        ; attn scale, IEEE-754 float bits, precomputed in C
     SF_COMMIT                                 ; commit stack frame, allocate shadow space
     vmovdqu vac_xmm6, xmm6                    ; save xmm6 low half
     vmovdqu vac_xmm7, xmm7                    ; save xmm7 low half
     vmovdqu vac_xmm8, xmm8                    ; save xmm8 low half
     vmovdqu vac_xmm9, xmm9                    ; save xmm9 low half, used by 72-path's vmin/max and weight broadcast
     vmovdqu vac_xmm10, xmm10                  ; save xmm10 low half, used as softmax parallel sum accumulator in both paths
     vmovdqu vac_xmm11, xmm11                  ; save xmm11 low half, Q register
     vmovdqu vac_xmm12, xmm12                  ; save xmm12 low half, Q register
     vmovdqu vac_xmm13, xmm13                  ; save xmm13 low half, Q register
     vmovdqu vac_xmm14, xmm14                  ; save xmm14 low half, 72-path hoisted Q remainder, NON-VOLATILE per Win64
     mov vac_attn, rcx                         ; attn output base
     mov vac_qkv, rdx                          ; qkv buffer base
     mov vac_scratch, r8                       ; scores scratch
     mov r10d, r9d                             ; r10d = n, register compare target; remains available across s_exp256_soft: pure AVX, no GPR overwrite
     mov r11, qword [rel g_vit_cfg + VCFG_QKV_STRIDE] ; r11 = qkv_dim*4, cached K/V/Q row stride; remains available across s_exp256_soft: no GPR overwrites
     mov eax, [rbp + VK_ARG5]                  ; head index
     mov vac_h, eax                            ; save h
; Cache the attention scale, precomputed in C, stored as IEEE bits in VCFG_ATTN_SCALE
     mov eax, dword [rel g_vit_cfg + VCFG_ATTN_SCALE] ; attn scale IEEE bits, 0x3E000000 for 0.125, head_dim 64 ; 0x3DF15BEF for head_dim 72, single source of truth, VCFG_ATTN_SCALE
     mov vac_scale, eax                        ; save scale, xmm read via vmovss later
; Branch on head_dim ONCE per call, not per j
     mov eax, dword [rel g_vit_cfg + VCFG_HEAD_DIM] ; head_dim, 64 or 72
     cmp eax, 64                               ; head_dim == 64?
     je .vac_path64                            ; yes, run the original Qwen3-VL 8-block path
     cmp eax, 72                               ; head_dim == 72?
     je .vac_path72                            ; yes, run the LLaVA/SigLIP 9-block path
; neither: the builder writes only 64/72, the fall-through, past the softmax macro
; definition, which emits no bytes lands on .vac_path64, acceptable for unreachable input

%macro VLC_EXP256_SOFT_INLINE 0
; Exact inline of s_exp256_soft: ymm0 = exp(ymm0), needs ymm1=C6 and ymm5=1.0 live.
; Clobbers ymm0/ymm2/ymm3/ymm4 only, reads ymm6/ymm8/ymm14 = log2e/ln2hi/ln2lo held in registers
; by .vac_softmax (dead Q block registers since pass 1, re-zeroed in pass 3) and ymm1/ymm5.
     vmaxps ymm0, ymm0, [rel s_negclamp_v]     ; clamp lower, underflow guard
     vmulps ymm3, ymm0, ymm6                   ; x * log2,e
     vroundps ymm3, ymm3, 0                    ; n = round,x/ln2
     vcvttps2dq ymm2, ymm3                     ; n as int32
     vfnmadd231ps ymm0, ymm3, ymm8             ; x - n*ln2_hi, single rounding
     vfnmadd231ps ymm0, ymm3, ymm14            ; f, single rounding
     vmovaps ymm4, ymm1                        ; polynomial seed = C6
     vpaddd ymm2, ymm2, [rel swk_bias_127]     ; bias n to the float exponent field
     vfmadd213ps ymm4, ymm0, [rel s_c5_v]      ; f*prev + c5
     vpslld ymm2, ymm2, 23                     ; construct the exponent field
     vfmadd213ps ymm4, ymm0, [rel s_c4_v]      ; f*prev + c4
     vfmadd213ps ymm4, ymm0, [rel s_c3_v]      ; f*prev + c3
     vfmadd213ps ymm4, ymm0, [rel s_half_v]    ; f*prev + 0.5, c2
     vfmadd213ps ymm4, ymm0, ymm5              ; f*prev + 1.0, c1
     vfmadd213ps ymm4, ymm0, ymm5              ; f*prev + 1.0, c0
     vmulps ymm0, ymm4, ymm2                   ; exp,x = poly * 2^n
%endmacro

%macro VLC_EXP256_SOFT_PAIR 0
; Two-lane form of the same 17-instruction polynomial: ymm0 = exp(ymm0), ymm7 = exp(ymm7).
; Second chain scratch: ymm11/ymm12/ymm13 (low halves saved in the prologue).
; Reads ymm6/ymm8/ymm14 = log2e/ln2hi/ln2lo held in registers by .vac_softmax
; (dead Q block registers since pass 1, re-zeroed in pass 3).
     vmaxps ymm0, ymm0, [rel s_negclamp_v]     ; chain A clamp lower, underflow guard
     vmaxps ymm7, ymm7, [rel s_negclamp_v]     ; chain B clamp lower, underflow guard
     vmulps ymm3, ymm0, ymm6                   ; A: x * log2,e
     vmulps ymm12, ymm7, ymm6                  ; B: x * log2,e
     vroundps ymm3, ymm3, 0                    ; A: n = round,x/ln2
     vroundps ymm12, ymm12, 0                  ; B: n = round,x/ln2
     vcvttps2dq ymm2, ymm3                     ; A: n as int32
     vcvttps2dq ymm11, ymm12                   ; B: n as int32
     vfnmadd231ps ymm0, ymm3, ymm8             ; A: x - n*ln2_hi, single rounding
     vfnmadd231ps ymm7, ymm12, ymm8            ; B: x - n*ln2_hi, single rounding
     vfnmadd231ps ymm0, ymm3, ymm14            ; A: f, single rounding
     vfnmadd231ps ymm7, ymm12, ymm14           ; B: f, single rounding
     vmovaps ymm4, ymm1                        ; A: polynomial seed = C6
     vmovaps ymm13, ymm1                       ; B: polynomial seed = C6
     vpaddd ymm2, ymm2, [rel swk_bias_127]     ; A: bias n to the float exponent field
     vpaddd ymm11, ymm11, [rel swk_bias_127]   ; B: bias n to the float exponent field
     vfmadd213ps ymm4, ymm0, [rel s_c5_v]      ; A: f*prev + c5
     vfmadd213ps ymm13, ymm7, [rel s_c5_v]     ; B: f*prev + c5
     vpslld ymm2, ymm2, 23                     ; A: construct the exponent field
     vpslld ymm11, ymm11, 23                   ; B: construct the exponent field
     vfmadd213ps ymm4, ymm0, [rel s_c4_v]      ; A: f*prev + c4
     vfmadd213ps ymm13, ymm7, [rel s_c4_v]     ; B: f*prev + c4
     vfmadd213ps ymm4, ymm0, [rel s_c3_v]      ; A: f*prev + c3
     vfmadd213ps ymm13, ymm7, [rel s_c3_v]     ; B: f*prev + c3
     vfmadd213ps ymm4, ymm0, [rel s_half_v]    ; A: f*prev + 0.5, c2
     vfmadd213ps ymm13, ymm7, [rel s_half_v]   ; B: f*prev + 0.5, c2
     vfmadd213ps ymm4, ymm0, ymm5              ; A: f*prev + 1.0, c1
     vfmadd213ps ymm13, ymm7, ymm5             ; B: f*prev + 1.0, c1
     vfmadd213ps ymm4, ymm0, ymm5              ; A: f*prev + 1.0, c0
     vfmadd213ps ymm13, ymm7, ymm5             ; B: f*prev + 1.0, c0
     vmulps ymm0, ymm4, ymm2                   ; A: exp,x = poly * 2^n
     vmulps ymm7, ymm13, ymm11                 ; B: exp,x = poly * 2^n
%endmacro

.vac_path64:
; Per-head base offsets within qkv, head stride = head_dim*4 bytes; data-driven via VCFG_HEAD_DIM_F4, NOT a compile-time constant .
; Q at qkv + h*head_dim*4, K at qkv + hidden*4 + h*head_dim*4, V at qkv + 2*hidden*4 + h*head_dim*4
     mov eax, vac_h                            ; head index
     mov ecx, dword [rel g_vit_cfg + VCFG_HEAD_DIM_F4] ; ecx = head_dim * 4, head byte stride, DB-precomputed
     imul eax, ecx                             ; eax = h * head_dim * 4, head byte offset, data-driven
     mov r9, rax                               ; r9 = running attn byte offset = h*head_dim*4, init once
     mov r12, vac_qkv                          ; qkv base
     add r12, rax                              ; r12 = Q base for head h, constant
     mov r13, r12                              ; r13 = qkv + head_off, K base is two strides above the Q base
     add r13, qword [rel g_vit_cfg + VCFG_HIDDEN_F4] ; r13 = K base = qkv + head_off + hidden*4
     mov r14, r13                              ; r14 = copy of K base
     add r14, qword [rel g_vit_cfg + VCFG_HIDDEN_F4] ; r14 = V base = K base + one hidden*4 stride
     xor r15d, r15d                            ; r15d = i = 0, query position, REGISTER
     mov rsi, r12                              ; rsi = &Q[0][h*64], RUNNING Q_i pointer
.vac_i_loop:
; rsi already = &Q[i][h*64], advanced by qkv_dim*4 at loop bottom
     mov rdi, vac_scratch                      ; rdi = &scores[0]
; preload the 8 Q_i blocks into ymm6..ymm13 ONCE per i, they are
; invariant across the entire j loop below, loaded once per i .
     vmovups ymm6, [rsi + VK_QK_BLK0]          ; Q[0..7]
     vmovups ymm7, [rsi + VK_QK_BLK1]          ; Q[8..15]
     vmovups ymm8, [rsi + VK_QK_BLK2]          ; Q[16..23]
     vmovups ymm9, [rsi + VK_QK_BLK3]          ; Q[24..31]
     vmovups ymm10, [rsi + VK_QK_BLK4]         ; Q[32..39]
     vmovups ymm11, [rsi + VK_QK_BLK5]         ; Q[40..47]
     vmovups ymm12, [rsi + VK_QK_BLK6]         ; Q[48..55]
     vmovups ymm13, [rsi + VK_QK_BLK7]         ; Q[56..63]
; --- Pass 1: compute scores[j] = dot,Q_i, K_j * scale, track running max ---
; K pointer starts at K_base, incremented by qkv_dim*4, VCFG_QKV_STRIDE per key
     mov r8, r13                               ; r8 = &K[0][h*64], r13 = constant K base
     xor ebx, ebx                              ; ebx = j = 0, key position, REGISTER
     vmovss xmm4, [rel vk_neg_inf]             ; xmm4 = running max = -inf, xmm4, NOT xmm9: VEX writes zero-extend into ymm9, the Q block 3
.vac_j_score:
     vxorps ymm0, ymm0, ymm0                   ; acc0 = 0
     vxorps ymm1, ymm1, ymm1                   ; acc1 = 0
     vxorps ymm2, ymm2, ymm2                   ; acc2 = 0
     vxorps ymm3, ymm3, ymm3                   ; acc3 = 0
     ds prefetcht0 [r8 + r11]                  ; prefetch the next K block, mandatory ds prefix keeps instruction lengths uniform for the SMC patcher
     vfmadd231ps ymm0, ymm6, [r8 + VK_QK_BLK0] ; acc0 += Q[0..7] * K[0..7]
     vfmadd231ps ymm1, ymm7, [r8 + VK_QK_BLK1] ; acc1 += Q[8..15] * K[8..15]
     vfmadd231ps ymm2, ymm8, [r8 + VK_QK_BLK2] ; acc2 += Q[16..23] * K[16..23]
     vfmadd231ps ymm3, ymm9, [r8 + VK_QK_BLK3] ; acc3 += Q[24..31] * K[24..31]
     vfmadd231ps ymm0, ymm10, [r8 + VK_QK_BLK4] ; acc0 += Q[32..39] * K[32..39]
     vfmadd231ps ymm1, ymm11, [r8 + VK_QK_BLK5] ; acc1 += Q[40..47] * K[40..47]
     vfmadd231ps ymm2, ymm12, [r8 + VK_QK_BLK6] ; acc2 += Q[48..55] * K[48..55]
     vfmadd231ps ymm3, ymm13, [r8 + VK_QK_BLK7] ; acc3 += Q[56..63] * K[56..63]
; Reduce to scalar via shuffle tree, combine + reduce, no vhaddps
     VAC_REDUCE_SCORE                          ; scores[j] = scaled dot, running max updated
     add r8, r11                               ; r8 += qkv_dim*4, next K row, cached stride
     inc ebx                                   ; j++
     cmp ebx, r10d                             ; j < n?, register compare
     jl .vac_j_score                           ; next key
     vmovss vac_max, xmm4                      ; store running max once for pass 2's broadcast + scalar remainder
; --- Pass 2: softmax, exp,score - max, compute sum, SIMD via s_exp256_soft ---
     call .vac_softmax                         ; shared softmax over this head
; --- Pass 3: value aggregation, attn_out[i][h*64..] = sum_j w[j] * V[j][h*64..] ---
; The cumulative attention byte offset is retained in r9 for each query position.
; Zero 8 accumulators, 64 floats = 8 x 8-float vectors = head_dim=64
     vxorps ymm0, ymm0, ymm0                   ; acc[0..7]
     vxorps ymm1, ymm1, ymm1                   ; acc[8..15]
     vxorps ymm2, ymm2, ymm2                   ; acc[16..23]
     vxorps ymm3, ymm3, ymm3                   ; acc[24..31]
     vxorps ymm4, ymm4, ymm4                   ; acc[32..39]
     vxorps ymm5, ymm5, ymm5                   ; acc[40..47]
     vxorps ymm6, ymm6, ymm6                   ; acc[48..55]
     vxorps ymm7, ymm7, ymm7                   ; acc[56..63]
     mov rdi, vac_scratch                      ; rdi = &scores[0]
     mov r8, r14                               ; r8 = &V[0][h*64], r14 = constant V base
     xor ebx, ebx                              ; ebx = j = 0
.vac_value:
; Load softmax weight and broadcast to all 8 YMM lanes
     vbroadcastss ymm8, [rdi + rbx*VK_F4]      ; ymm8 = broadcast weight[j], indexed by j, ebx is j, zeroed at the loop head
; Accumulate 8 FMA operations, 64 floats = full head_dim, using named byte offsets
     vfmadd231ps ymm0, ymm8, [r8 + VK_VOFF_0]  ; acc[0..7] += w * V[j][0..7]
     vfmadd231ps ymm1, ymm8, [r8 + VK_VOFF_1]  ; acc[8..15] += w * V[j][8..15]
     vfmadd231ps ymm2, ymm8, [r8 + VK_VOFF_2]  ; acc[16..23] += w * V[j][16..23]
     vfmadd231ps ymm3, ymm8, [r8 + VK_VOFF_3]  ; acc[24..31] += w * V[j][24..31]
     vfmadd231ps ymm4, ymm8, [r8 + VK_VOFF_4]  ; acc[32..39] += w * V[j][32..39]
     vfmadd231ps ymm5, ymm8, [r8 + VK_VOFF_5]  ; acc[40..47] += w * V[j][40..47]
     vfmadd231ps ymm6, ymm8, [r8 + VK_VOFF_6]  ; acc[48..55] += w * V[j][48..55]
     vfmadd231ps ymm7, ymm8, [r8 + VK_VOFF_7]  ; acc[56..63] += w * V[j][56..63]
     add r8, r11                               ; r8 += qkv_dim*4, next V row, cached stride
     inc ebx                                   ; j++
     cmp ebx, r10d                             ; j < n?, register compare
     jl .vac_value                             ; next value
     vbroadcastss ymm8, vac_invsum             ; ymm8 = broadcast refined inv_sum
     mov rax, vac_attn                         ; rax = attn base
     add rax, r9                               ; rax = &attn[i][h*64], offset lives in r9, no imul
     vmulps ymm0, ymm0, ymm8                   ; normalize acc[0..7]
     vmulps ymm1, ymm1, ymm8                   ; normalize acc[8..15]
     vmovups [rax + VK_VOFF_0], ymm0           ; store attn[i][h*64 + 0..7]
     vmovups [rax + VK_VOFF_1], ymm1           ; store attn[i][h*64 + 8..15]
     vmulps ymm2, ymm2, ymm8                   ; normalize acc[16..23]
     vmulps ymm3, ymm3, ymm8                   ; normalize acc[24..31]
     vmovups [rax + VK_VOFF_2], ymm2           ; store attn[i][h*64 + 16..23]
     vmovups [rax + VK_VOFF_3], ymm3           ; store attn[i][h*64 + 24..31]
     vmulps ymm4, ymm4, ymm8                   ; normalize acc[32..39]
     vmulps ymm5, ymm5, ymm8                   ; normalize acc[40..47]
     vmovups [rax + VK_VOFF_4], ymm4           ; store attn[i][h*64 + 32..39]
     vmovups [rax + VK_VOFF_5], ymm5           ; store attn[i][h*64 + 40..47]
     vmulps ymm6, ymm6, ymm8                   ; normalize acc[48..55]
     vmulps ymm7, ymm7, ymm8                   ; normalize acc[56..63]
     vmovups [rax + VK_VOFF_6], ymm6           ; store attn[i][h*64 + 48..55]
     vmovups [rax + VK_VOFF_7], ymm7           ; store attn[i][h*64 + 56..63]
     add rsi, r11                              ; rsi = &Q[i+1][h*64], running Q_i ptr, cached stride
     add r9, qword [rel g_vit_cfg + VCFG_HIDDEN_F4] ; r9 = attn byte offset for i+1, running, no imul
     inc r15d                                  ; i++
     cmp r15d, r10d                            ; i < n?, register compare
     jl .vac_i_loop                            ; next query
     jmp .vac_done                             ; 64-path done, jump over the 72-path
.vac_path72:
; Per-head base offsets within qkv, head stride = head_dim*4 bytes; data-driven via VCFG_HEAD_DIM_F4, NOT a compile-time constant .
; Q at qkv + h*head_dim*4, K at qkv + hidden*4 + h*head_dim*4, V at qkv + 2*hidden*4 + h*head_dim*4
     mov eax, vac_h                            ; head index
     mov ecx, dword [rel g_vit_cfg + VCFG_HEAD_DIM_F4] ; ecx = head_dim * 4, head byte stride, DB-precomputed
     imul eax, ecx                             ; eax = h * head_dim * 4, head byte offset, data-driven
     mov r12, vac_qkv                          ; qkv base
     add r12, rax                              ; r12 = Q base for head h, constant
     mov r13, r12                              ; r13 = qkv + head_off, K base is two strides above the Q base
     add r13, qword [rel g_vit_cfg + VCFG_HIDDEN_F4] ; r13 = K base = qkv + head_off + hidden*4
     mov r14, r13                              ; r14 = copy of K base
     add r14, qword [rel g_vit_cfg + VCFG_HIDDEN_F4] ; r14 = V base = K base + one hidden*4 stride
     mov r9, rax                               ; r9 = running attn byte offset = h*head_dim*4, init once
     xor r15d, r15d                            ; r15d = i = 0, query position, REGISTER
     mov rsi, r12                              ; rsi = &Q[0][h*head_dim], RUNNING Q_i pointer
.vac72_i_loop:
; rsi already = &Q[i][h*head_dim], advanced by qkv_dim*4 at loop bottom
     mov rdi, vac_scratch                      ; rdi = &scores[0]
     vmovups ymm6, [rsi + VK_QK_BLK0]          ; Q[0..7]
     vmovups ymm7, [rsi + VK_QK_BLK1]          ; Q[8..15]
     vmovups ymm8, [rsi + VK_QK_BLK2]          ; Q[16..23]
     vmovups ymm9, [rsi + VK_QK_BLK3]          ; Q[24..31]
     vmovups ymm10, [rsi + VK_QK_BLK4]         ; Q[32..39]
     vmovups ymm11, [rsi + VK_QK_BLK5]         ; Q[40..47]
     vmovups ymm12, [rsi + VK_QK_BLK6]         ; Q[48..55]
     vmovups ymm13, [rsi + VK_QK_BLK7]         ; Q[56..63]
     vmovups ymm14, [rsi + VK_QK_BLK8]         ; Q[64..71], 9th block, invariant across the j loop, same argument as blocks 0..7
; --- Pass 1: compute scores[j] = dot,Q_i, K_j * scale, track running max ---
     mov r8, r13                               ; r8 = &K[0][h*head_dim], r13 = constant K base
     xor ebx, ebx                              ; ebx = j = 0, key position, REGISTER
     vmovss xmm4, [rel vk_neg_inf]             ; xmm4 = running max = -inf, xmm4, NOT xmm9: VEX writes zero-extend into ymm9, the Q block 3
.vac72_j_score:
     vxorps ymm0, ymm0, ymm0                   ; acc0 = 0
     vxorps ymm1, ymm1, ymm1                   ; acc1 = 0
     vxorps ymm2, ymm2, ymm2                   ; acc2 = 0
     vxorps ymm3, ymm3, ymm3                   ; acc3 = 0
     ds prefetcht0 [r8 + r11]                  ; prefetch the next K block, mandatory ds prefix keeps instruction lengths uniform for the SMC patcher
     vfmadd231ps ymm0, ymm6, [r8 + VK_QK_BLK0] ; acc0 += Q[0..7] * K[0..7]
     vfmadd231ps ymm1, ymm7, [r8 + VK_QK_BLK1] ; acc1 += Q[8..15] * K[8..15]
     vfmadd231ps ymm2, ymm8, [r8 + VK_QK_BLK2] ; acc2 += Q[16..23] * K[16..23]
     vfmadd231ps ymm3, ymm9, [r8 + VK_QK_BLK3] ; acc3 += Q[24..31] * K[24..31]
     vfmadd231ps ymm0, ymm10, [r8 + VK_QK_BLK4] ; acc0 += Q[32..39] * K[32..39]
     vfmadd231ps ymm1, ymm11, [r8 + VK_QK_BLK5] ; acc1 += Q[40..47] * K[40..47]
     vfmadd231ps ymm2, ymm12, [r8 + VK_QK_BLK6] ; acc2 += Q[48..55] * K[48..55]
     vfmadd231ps ymm3, ymm13, [r8 + VK_QK_BLK7] ; acc3 += Q[56..63] * K[56..63]
     vfmadd231ps ymm0, ymm14, [r8 + VK_QK_BLK8] ; acc0 += Q[64..71] * K[64..71]
     VAC_REDUCE_SCORE                          ; scores[j] = scaled dot, running max updated
     add r8, r11                               ; r8 += qkv_dim*4, next K row, cached stride
     inc ebx                                   ; j++
     cmp ebx, r10d                             ; j < n?, register compare
     jl .vac72_j_score                         ; next key
     vmovss vac_max, xmm4                      ; store running max once for pass 2's broadcast + scalar remainder
     call .vac_softmax                         ; shared softmax over this head
; --- Pass 3: value aggregation, attn_out[i][h*head_dim..] = sum_j w[j] * V[j][h*head_dim..] ---
; The cumulative attention byte offset is retained in r9 for each query position.
; Zero 9 accumulators, 72 floats = 9 x 8-float YMM vectors = head_dim=72
     vxorps ymm0, ymm0, ymm0                   ; acc[0..7]
     vxorps ymm1, ymm1, ymm1                   ; acc[8..15]
     vxorps ymm2, ymm2, ymm2                   ; acc[16..23]
     vxorps ymm3, ymm3, ymm3                   ; acc[24..31]
     vxorps ymm4, ymm4, ymm4                   ; acc[32..39]
     vxorps ymm5, ymm5, ymm5                   ; acc[40..47]
     vxorps ymm6, ymm6, ymm6                   ; acc[48..55]
     vxorps ymm7, ymm7, ymm7                   ; acc[56..63]
     vxorps ymm8, ymm8, ymm8                   ; acc[64..71], 9th accumulator for head_dim=72
     mov rdi, vac_scratch                      ; rdi = &scores[0]
     mov r8, r14                               ; r8 = &V[0][h*head_dim], r14 = constant V base
     xor ebx, ebx                              ; ebx = j = 0
.vac72_value:
; Load softmax weight and broadcast to all 8 YMM lanes
     vbroadcastss ymm9, [rdi + rbx*VK_F4]      ; ymm9 = broadcast weight[j], indexed by j, ebx is j, zeroed at the loop head
; Accumulate 9 FMA operations, 72 floats = full head_dim=72
     vfmadd231ps ymm0, ymm9, [r8 + VK_VOFF_0]  ; acc[0..7] += w * V[j][0..7]
     vfmadd231ps ymm1, ymm9, [r8 + VK_VOFF_1]  ; acc[8..15] += w * V[j][8..15]
     vfmadd231ps ymm2, ymm9, [r8 + VK_VOFF_2]  ; acc[16..23] += w * V[j][16..23]
     vfmadd231ps ymm3, ymm9, [r8 + VK_VOFF_3]  ; acc[24..31] += w * V[j][24..31]
     vfmadd231ps ymm4, ymm9, [r8 + VK_VOFF_4]  ; acc[32..39] += w * V[j][32..39]
     vfmadd231ps ymm5, ymm9, [r8 + VK_VOFF_5]  ; acc[40..47] += w * V[j][40..47]
     vfmadd231ps ymm6, ymm9, [r8 + VK_VOFF_6]  ; acc[48..55] += w * V[j][48..55]
     vfmadd231ps ymm7, ymm9, [r8 + VK_VOFF_7]  ; acc[56..63] += w * V[j][56..63]
     vfmadd231ps ymm8, ymm9, [r8 + VK_VOFF_8]  ; acc[64..71] += w * V[j][64..71]
     add r8, r11                               ; r8 += qkv_dim*4, next V row, cached stride
     inc ebx                                   ; j++
     cmp ebx, r10d                             ; j < n?, register compare
     jl .vac72_value                           ; next value
; Normalize all 9 accumulators by inv_sum, interleaved with stores so the
; 5-cycle multiply latency overlaps store-port execution, mirrors the 64-path .
     vbroadcastss ymm9, vac_invsum             ; ymm9 = broadcast inv_sum
     mov rax, vac_attn                         ; rax = attn base
     add rax, r9                               ; rax = &attn[i][h*head_dim], offset from running r9
     vmulps ymm0, ymm0, ymm9                   ; normalize acc[0..7]
     vmulps ymm1, ymm1, ymm9                   ; normalize acc[8..15]
     vmovups [rax + VK_VOFF_0], ymm0           ; store attn[i][h*head_dim + 0..7]
     vmovups [rax + VK_VOFF_1], ymm1           ; store attn[i][h*head_dim + 8..15]
     vmulps ymm2, ymm2, ymm9                   ; normalize acc[16..23]
     vmulps ymm3, ymm3, ymm9                   ; normalize acc[24..31]
     vmovups [rax + VK_VOFF_2], ymm2           ; store attn[i][h*head_dim + 16..23]
     vmovups [rax + VK_VOFF_3], ymm3           ; store attn[i][h*head_dim + 24..31]
     vmulps ymm4, ymm4, ymm9                   ; normalize acc[32..39]
     vmulps ymm5, ymm5, ymm9                   ; normalize acc[40..47]
     vmovups [rax + VK_VOFF_4], ymm4           ; store attn[i][h*head_dim + 32..39]
     vmovups [rax + VK_VOFF_5], ymm5           ; store attn[i][h*head_dim + 40..47]
     vmulps ymm6, ymm6, ymm9                   ; normalize acc[48..55]
     vmulps ymm7, ymm7, ymm9                   ; normalize acc[56..63]
     vmovups [rax + VK_VOFF_6], ymm6           ; store attn[i][h*head_dim + 48..55]
     vmovups [rax + VK_VOFF_7], ymm7           ; store attn[i][h*head_dim + 56..63]
     vmulps ymm8, ymm8, ymm9                   ; normalize acc[64..71]
     vmovups [rax + VK_VOFF_8], ymm8           ; store attn[i][h*head_dim + 64..71]
     add rsi, r11                              ; rsi = &Q[i+1][h*head_dim], running Q_i ptr, cached stride
     add r9, qword [rel g_vit_cfg + VCFG_HIDDEN_F4] ; r9 = attn byte offset for i+1, running, +hidden*4
     inc r15d                                  ; i++
     cmp r15d, r10d                            ; i < n?, register compare
     jl .vac72_i_loop                          ; next query
.vac_done:
     vzeroupper                                ; clean AVX state before restoring callee-saved YMMs
     vmovdqu xmm14, vac_xmm14                  ; restore xmm14 low half, 72-path Q remainder
     vmovdqu xmm13, vac_xmm13                  ; restore xmm13 low half, Q register
     vmovdqu xmm12, vac_xmm12                  ; restore xmm12 low half, Q register
     vmovdqu xmm11, vac_xmm11                  ; restore xmm11 low half, Q register
     vmovdqu xmm10, vac_xmm10                  ; restore xmm10 low half, softmax parallel sum
     vmovdqu xmm9, vac_xmm9                    ; restore xmm9 low half, 72-path vmin/max + weight broadcast
     vmovdqu xmm8, vac_xmm8                    ; restore xmm8 low half, Q register
     vmovdqu xmm7, vac_xmm7                    ; restore xmm7 low half, Q register
     vmovdqu xmm6, vac_xmm6                    ; restore xmm6 low half, Q register
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore non-volatiles and return
.vac_softmax:
; Private leaf in the attention frame. RDI = start of this query's scores.
     vxorps xmm7, xmm7, xmm7                   ; sum = 0
     vmovups ymm5, [rel s_one_v]               ; exp polynomial unit constant
     vmovups ymm1, [rel s_c6_v]                ; highest-order coefficient
     vmovups ymm6, [rel s_log2e_v]             ; convert to base 2
     vmovups ymm8, [rel s_ln2hi_v]             ; range reduction high term
     vmovups ymm14, [rel s_ln2lo_v]            ; range reduction low term
     vbroadcastss ymm9, vac_max                ; score max, same for all lanes
     mov eax, r10d                             ; number of scores in this head
     shr eax, 3                                ; count of full score vectors
     jz .vac_sm_tail                           ; short head has only a partial vector
     vxorps ymm10, ymm10, ymm10                ; ymm10 = parallel sum accumulator, remains available across exp
.vac_sm_vec:                                   ; vector loop: two 8-score blocks per iteration, rdi advances by VK_8F_BYTES
     cmp eax, 2                                ; two full blocks available?
     jb .vac_sm_vec_one                        ; no, single-block path
.vac_sm_vec_two:
     vmovups ymm0, [rdi]                       ; block A scores
     vmovups ymm7, [rdi + VK_8F_BYTES]         ; block B scores
     vsubps ymm0, ymm0, ymm9                   ; A - max
     vsubps ymm7, ymm7, ymm9                   ; B - max
     VLC_EXP256_SOFT_PAIR                      ; ymm0 = exp(A), ymm7 = exp(B), interleaved
     vmovups [rdi], ymm0                       ; store A back 8 normalized weights, in place
     vmovups [rdi + VK_8F_BYTES], ymm7         ; store B back 8 normalized weights, in place
     vaddps ymm10, ymm10, ymm0                 ; parallel sum += A weights, original token order
     vaddps ymm10, ymm10, ymm7                 ; parallel sum += B weights, original token order
     add rdi, VK_8F_BYTES*2                    ; advance two blocks
     sub eax, 2                                ; two groups done
     cmp eax, 2                                ; another full pair?
     jae .vac_sm_vec_two                       ; yes: bottom-tested loop
.vac_sm_vec_one:
     test eax, eax                             ; odd trailing block?
     jz .vac_sm_vec_finished                   ; no, all covered
     vmovups ymm0, [rdi]                       ; load 8 scores
     vsubps ymm0, ymm0, ymm9                   ; score - max, 8-wide
     VLC_EXP256_SOFT_INLINE                    ; ymm0 = exp,score - max, polynomial + 2^n
     vmovups [rdi], ymm0                       ; store back 8 normalized weights, in place
     vaddps ymm10, ymm10, ymm0                 ; parallel sum accumulator += weights
     add rdi, VK_8F_BYTES                      ; advance 8 floats
.vac_sm_vec_finished:
; Reduce 8-lane sum to scalar via shuffle tree, no vhaddps
     vextractf128 xmm7, ymm10, 1               ; xmm7 = upper 4 lanes
     vaddps xmm7, xmm7, xmm10                  ; xmm7 = low + high -> 4 lanes, xmm10 = low 128
     vmovhlps xmm2, xmm7, xmm7                 ; hi 64 -> lo 64
     vaddps xmm7, xmm7, xmm2                   ; add -> 2 lanes
     vmovshdup xmm2, xmm7                      ; lane 1 -> lane 0
     vaddss xmm7, xmm7, xmm2                   ; xmm7 = scalar vector sum
.vac_sm_tail:
     mov ebx, r10d                             ; total score count
     and ebx, 7                                ; valid lanes in the partial vector
     jz .vac_sm_done                           ; no partial vector
     vmovd xmm10, ebx                          ; exclusive valid lane bound
     vpbroadcastd ymm10, xmm10                 ; compare the bound in every lane
     vpmovzxbd ymm2, [rel AESH_BYTE_INDEX]     ; existing byte indices 0..7
     vpcmpgtd ymm10, ymm10, ymm2               ; sign bit set only for valid lanes
     vmaskmovps ymm0, ymm10, [rdi]             ; load only valid scores, even at a page edge
     vblendvps ymm0, ymm9, ymm0, ymm10         ; inactive lanes get the max before subtraction
     vsubps ymm0, ymm0, ymm9                   ; original score-minus-max
     VLC_EXP256_SOFT_INLINE                    ; identical per-lane FP32 polynomial
     vmaskmovps [rdi], ymm10, ymm0             ; write only valid probabilities
.vac_sm_tail_sum:
     vaddss xmm7, xmm7, [rdi]                  ; preserve the original left-to-right tail sum
     add rdi, VK_F4                            ; next valid probability
     dec ebx                                   ; one fewer tail lane
     jnz .vac_sm_tail_sum                      ; no horizontal reordering in this tail
.vac_sm_done:                                  ; all weights normalized in place, xmm7 = full sum
; --- Compute 1/sum via full-precision vdivss, matches PyTorch float32 ---
     vmovss xmm1, [rel vk_one_f]               ; xmm1 = 1.0
     vdivss xmm7, xmm1, xmm7                   ; xmm7 = 1.0 / sum, full float32 precision
     vmovss vac_invsum, xmm7                   ; save inv_sum for value normalization
     ret                                       ; vac_invsum and scratch probabilities are ready
%unmacro VLC_EXP256_SOFT_INLINE 0              ; both softmax paths expanded; the leaf above was its last user
; ---- %unmacro teardown: the three slow-path macros leave scope here ----
%unmacro VLC_K_LOOP_BODY 1                     ; 32-wide K loop body, four slow-path variants above
%unmacro VLC_K_TAIL_BODY 2                     ; scalar K%32 remainder, loop label + exit label args
%unmacro VLC_REDUCE4 0                         ; acc0-3 shuffle-reduce into the xmm0 seed
%unmacro VLC4_REDUCE_ALL 1                     ; the interleaved four-column fast-path reduce, with optional no-bias mode
%unmacro VLC_EXP256_SOFT_PAIR 0                ; two-lane exp form, consumed only inside .vac_softmax above
; ---- %undef teardown: file-local constants leave scope here ----
%undef VLC_TAIL_THRESH4                        ; 31: slow-path 32-wide K-loop bound
%undef VLC_16F_BYTES                           ; 64: d+16 byte offset in the 4-accumulator K loop
%undef VLC_24F_BYTES                           ; 96: d+24 byte offset in the 4-accumulator K loop
%undef VLC_SCALE2                              ; 2: element scale 2 within a 4-column block
%undef VLC_SCALE3                              ; 3: element scale 3 within a 4-column block
%undef VLC_WR_SCALE2                           ; 2: weight-row scale 2 within a block
%undef VLC_WR_SCALE4                           ; 4: weight-row scale 4, next 4-column block
%undef VLC_PF_DIST                             ; 64: prefetch distance in bytes, one cache line ahead
; ---- SF slot retirement: the vlc_*/vac_* frame slots must not leak into the
; rest of the translation unit, NASM re-defines textually with no diagnostic .
%undef vlc_n
%undef vlc_n_raw
%undef vlc_cstart
%undef vlc_cstart_raw
%undef vlc_cd
%undef vlc_cd_raw
%undef vlc_out
%undef vlc_out_raw
%undef vlc_in
%undef vlc_in_raw
%undef vlc_w
%undef vlc_w_raw
%undef vlc_bias
%undef vlc_bias_raw
%undef vlc_xmm6
%undef vlc_xmm7
%undef vlc_xmm10
%undef vlc_xmm11
%undef vlc_xmm12
%undef vlc_xmm13
%undef vlc_xmm14
%undef vlc_xmm15
%undef vac_xmm6
%undef vac_xmm7
%undef vac_xmm8
%undef vac_xmm9
%undef vac_xmm10
%undef vac_xmm11
%undef vac_xmm12
%undef vac_xmm13
%undef vac_xmm14
%undef vac_h
%undef vac_h_raw
%undef vac_attn
%undef vac_attn_raw
%undef vac_qkv
%undef vac_qkv_raw
%undef vac_scratch
%undef vac_scratch_raw
%undef vac_max
%undef vac_max_raw
%undef vac_invsum
%undef vac_invsum_raw
%undef vac_scale
%undef vac_scale_raw

