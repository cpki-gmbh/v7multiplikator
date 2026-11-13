; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; q8_batch4_kernel.inc -- four-token Q8 batch GEMV leaf. Computes four
; independent Q8 GEMVs against one shared Q8 weight matrix, loading each
; 64-byte weight group once for all four tokens (one output row at a time,
; four independent accumulators; exact arithmetic sequence identical to
; q8_kernel.inc's paired-row dot chain). Invoked by the existing row-sliced
; dispatcher in utility_dispatch.inc from the Q8 prefill path.

; Argument block offsets; MUST stay in sync with src/q8_batch4.h
%define Q8B_OUT 0                              ; qword[4], four output row array pointers
%define Q8B_XQ 32                              ; qword[4], four quantized input pointers
%define Q8B_XS 64                              ; qword[4], four activation scale array pointers
%define Q8B_WQ 96                              ; qword, contiguous row-major Q8 weight bytes
%define Q8B_WS 104                             ; qword, weight scales, one float per 64 values
%define Q8B_N 112                              ; dword, elements per input, multiple of 64
%define Q8B_D 116                              ; dword, output row count
%define Q8B_OUT1 8                             ; qword slot of output pointer one, Q8B_OUT + one pointer stride
%define Q8B_OUT2 16                            ; qword slot of output pointer two
%define Q8B_OUT3 24                            ; qword slot of output pointer three
%define Q8B_TOKEN_STRIDE 8                     ; byte stride between the per-token input/scale pointer slots
%define Q8B_TOK0_OFF 0                         ; XQ/XS slot 0 byte offset
%define Q8B_TOK1_OFF Q8B_TOKEN_STRIDE          ; XQ/XS slot 1 byte offset
%define Q8B_TOK2_OFF Q8B_TOKEN_STRIDE*2        ; XQ/XS slot 2 byte offset
%define Q8B_TOK3_OFF Q8B_TOKEN_STRIDE*3        ; XQ/XS slot 3 byte offset
%define Q8B_PACK_EXPANSION 2                   ; Signed and magnitude bytes per original byte
%define Q8B_PACK_UPPER_SIGN 64                 ; Upper signed vector offset in a packed group
%define Q8B_PACK_UPPER_ABS 96                  ; Upper magnitude vector offset in a packed group
%define Q8B_HALF 32                            ; byte offset of the upper 32-byte half of a 64-byte group
%define Q8B_FLOAT_STRIDE 4                     ; byte stride of one float in a scale or output array

; Parameters: two token activation pointers, their precomputed scale vectors,
; and their accumulators. Interleaves the two tokens' integer dot chains over
; the shared weight bytes (ymm4/ymm5 already loaded) so the load/sign/madd
; latency of one chain hides behind the other. Per token the integer sequence
; is identical to the former single-token leaf, so results are bit-identical.

%macro Q8B_DOT_PAIR 7                          ; %1/%2 token ptrs, %3/%4 scales, %5/%6 accumulators
%if %7
     vmovdqu ymm8, [%1 + rcx*Q8B_PACK_EXPANSION + Q8B_HALF] ; A precomputed lower magnitudes
     vmovdqu ymm9, [%2 + rcx*Q8B_PACK_EXPANSION + Q8B_HALF] ; B precomputed lower magnitudes
     vpsignb ymm10, ymm4, [%1 + rcx*Q8B_PACK_EXPANSION] ; A lower weights with the activation signs
     vpsignb ymm11, ymm4, [%2 + rcx*Q8B_PACK_EXPANSION] ; B lower weights with the activation signs
%else
     vmovdqu ymm6, [%1 + rcx]                  ; token A lower 32 activation bytes
     vmovdqu ymm7, [%2 + rcx]                  ; token B lower 32
     vpabsb ymm8, ymm6                         ; A unsigned magnitudes
     vpabsb ymm9, ymm7                         ; B unsigned magnitudes
     vpsignb ymm10, ymm4, ymm6                 ; A sign-corrected lower weights
     vpsignb ymm11, ymm4, ymm7                 ; B sign-corrected lower weights
%endif
     vpmaddubsw ymm10, ymm8, ymm10             ; A pairwise madd lower
     vpmaddubsw ymm11, ymm9, ymm11             ; B pairwise madd lower
     vpmaddwd ymm10, ymm10, ymm13              ; A horizontal pair-add low
     vpmaddwd ymm11, ymm11, ymm13              ; B horizontal pair-add low
%if %7
     vmovdqu ymm8, [%1 + rcx*Q8B_PACK_EXPANSION + Q8B_PACK_UPPER_ABS] ; A precomputed upper magnitudes
     vmovdqu ymm9, [%2 + rcx*Q8B_PACK_EXPANSION + Q8B_PACK_UPPER_ABS] ; B precomputed upper magnitudes
     vpsignb ymm6, ymm5, [%1 + rcx*Q8B_PACK_EXPANSION + Q8B_PACK_UPPER_SIGN] ; A upper weights with the activation signs
     vpsignb ymm7, ymm5, [%2 + rcx*Q8B_PACK_EXPANSION + Q8B_PACK_UPPER_SIGN] ; B upper weights with the activation signs
%else
     vmovdqu ymm6, [%1 + rcx + Q8B_HALF]       ; token A upper 32
     vmovdqu ymm7, [%2 + rcx + Q8B_HALF]       ; token B upper 32
     vpabsb ymm8, ymm6                         ; A upper unsigned magnitudes
     vpabsb ymm9, ymm7                         ; B upper unsigned magnitudes
     vpsignb ymm6, ymm5, ymm6                  ; A upper sign-corrected
     vpsignb ymm7, ymm5, ymm7                  ; B upper sign-corrected
%endif
     vpmaddubsw ymm6, ymm8, ymm6               ; A pairwise madd upper
     vpmaddubsw ymm7, ymm9, ymm7               ; B pairwise madd upper
     vpmaddwd ymm6, ymm6, ymm13                ; A horizontal pair-add high
     vpmaddwd ymm7, ymm7, ymm13                ; B horizontal pair-add high
     vpaddd ymm10, ymm10, ymm6                 ; A combine low+high
     vpaddd ymm11, ymm11, ymm7                 ; B combine low+high
     vcvtdq2ps ymm10, ymm10                    ; A dot to float
     vcvtdq2ps ymm11, ymm11                    ; B dot to float
     vfmadd231ps %5, ymm10, %3                 ; A accumulate
     vfmadd231ps %6, ymm11, %4                 ; B accumulate
%endmacro

     section .text
%macro Q8B_KERNEL 2                            ; Leaf name and packed-input layout flag
     SF_START %1, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: emits global+label, saves the non-volatiles used
     SF_XMM q8b_xmm6                           ; xmm6 low half spill slot, Win64 preserves only the low 128 bits
     SF_XMM q8b_xmm7                           ; xmm7 low half spill slot
     SF_XMM q8b_xmm8                           ; xmm8 low half spill slot
     SF_XMM q8b_xmm9                           ; xmm9 low half spill slot
     SF_XMM q8b_xmm10                          ; xmm10 low half spill slot
     SF_XMM q8b_xmm11                          ; xmm11 low half spill slot
     SF_XMM q8b_xmm13                          ; xmm13 low half spill slot, ones vector
     SF_XMM q8b_xmm14                          ; xmm14 low half spill slot, weight-group scale
     SF_XMM q8b_xmm15                          ; xmm15 low half spill slot, combined scale
     SF_XMM q8b_xmm12                          ; xmm12 low half spill slot, scale staging
     SF_COMMIT                                 ; finalize stack frame layout
     vmovdqu q8b_xmm6, xmm6                    ; save callee-saved xmm6 low half
     vmovdqu q8b_xmm7, xmm7                    ; save callee-saved xmm7 low half
     vmovdqu q8b_xmm8, xmm8                    ; save callee-saved xmm8 low half
     vmovdqu q8b_xmm9, xmm9                    ; save callee-saved xmm9 low half
     vmovdqu q8b_xmm10, xmm10                  ; save callee-saved xmm10 low half
     vmovdqu q8b_xmm11, xmm11                  ; save callee-saved xmm11 low half
     vmovdqu q8b_xmm13, xmm13                  ; save callee-saved xmm13 low half
     vmovdqu q8b_xmm14, xmm14                  ; save callee-saved xmm14 low half
     vmovdqu q8b_xmm15, xmm15                  ; save callee-saved xmm15 low half
     vmovdqu q8b_xmm12, xmm12                  ; save callee-saved xmm12 low half
     mov rbx, rcx                              ; retain the argument block throughout the call
     mov r8, [rbx + Q8B_XQ + Q8B_TOK0_OFF]     ; token 0 activation base, invariant across rows/groups
     mov r9, [rbx + Q8B_XQ + Q8B_TOK1_OFF]     ; token 1 activation base
     mov r10, [rbx + Q8B_XQ + Q8B_TOK2_OFF]    ; token 2 activation base
     mov r11, [rbx + Q8B_XQ + Q8B_TOK3_OFF]    ; token 3 activation base
     mov rax, [rbx + Q8B_XS + Q8B_TOK0_OFF]    ; token 0 activation-scale base
     mov rdx, [rbx + Q8B_XS + Q8B_TOK1_OFF]    ; token 1 activation-scale base
     mov r13, [rbx + Q8B_XS + Q8B_TOK2_OFF]    ; token 2 activation-scale base
     mov r14, [rbx + Q8B_XS + Q8B_TOK3_OFF]    ; token 3 activation-scale base
     vpcmpeqd ymm13, ymm13, ymm13              ; construct a vector of all one bits
     vpabsw ymm13, ymm13                       ; convert the ones into 16 signed word values of one
     xor r12d, r12d                            ; begin with output row zero
     mov rsi, [rbx + Q8B_WQ]                   ; weight pointer for row 0 of THIS job
     mov rdi, [rbx + Q8B_WS]                   ; scale pointer for row 0 of THIS job
.row_loop:
     cmp r12d, [rbx + Q8B_D]                   ; every output row processed?
     jae .done                                 ; yes, exit after the last row
     vxorps ymm0, ymm0, ymm0                   ; clear the accumulator for token zero
     vxorps ymm1, ymm1, ymm1                   ; clear the accumulator for token one
     vxorps ymm2, ymm2, ymm2                   ; clear the accumulator for token two
     vxorps ymm3, ymm3, ymm3                   ; clear the accumulator for token three
     xor ecx, ecx                              ; byte offset within the input and weight rows
     xor r15d, r15d                            ; scale-group index
.group_loop:
     vbroadcastss ymm14, [rdi + r15*Q8B_FLOAT_STRIDE] ; issue the independent weight-group scale load first
     vmovdqu ymm4, [rsi + rcx]                 ; load the lower 32 weight bytes once for all four tokens
     vmovdqu ymm5, [rsi + rcx + Q8B_HALF]      ; load the upper 32 weight bytes once
; token0 and token1 scales, both ready before either dot
     vbroadcastss ymm15, [rax + r15*Q8B_FLOAT_STRIDE] ; token0 activation scale
     vbroadcastss ymm12, [rdx + r15*Q8B_FLOAT_STRIDE] ; token1 activation scale
     vmulps ymm15, ymm15, ymm14                ; token0 combined scale, ready before its dot
     vmulps ymm12, ymm12, ymm14                ; token1 combined scale
     Q8B_DOT_PAIR r8, r9, ymm15, ymm12, ymm0, ymm1, %2 ; tokens 0+1, interleaved integer dot chains
; token2 and token3: same schedule
     vbroadcastss ymm15, [r13 + r15*Q8B_FLOAT_STRIDE] ; token2 activation scale
     vbroadcastss ymm12, [r14 + r15*Q8B_FLOAT_STRIDE] ; token3 activation scale
     vmulps ymm15, ymm15, ymm14                ; token2 combined scale
     vmulps ymm12, ymm12, ymm14                ; token3 combined scale
     Q8B_DOT_PAIR r10, r11, ymm15, ymm12, ymm2, ymm3, %2 ; tokens 2+3, interleaved integer dot chains
     add ecx, 64                               ; advance to the next 64-byte input group
     inc r15d                                  ; advance to the next scale group
     cmp ecx, [rbx + Q8B_N]                    ; compare with the full input dimension
     jb .group_loop                            ; repeat while another group remains
     HADD4_YMM                                 ; four independent reductions, identical lane-wise addition tree
     mov rcx, [rbx + Q8B_OUT]                  ; output pointer zero, rcx dead after the group loop
     vmovss [rcx + r12*Q8B_FLOAT_STRIDE], xmm0 ; write this row for token zero
     mov rcx, [rbx + Q8B_OUT1]                 ; output pointer one
     vmovss [rcx + r12*Q8B_FLOAT_STRIDE], xmm1 ; write this row for token one
     mov rcx, [rbx + Q8B_OUT2]                 ; output pointer two
     vmovss [rcx + r12*Q8B_FLOAT_STRIDE], xmm2 ; write this row for token two
     mov rcx, [rbx + Q8B_OUT3]                 ; output pointer three
     vmovss [rcx + r12*Q8B_FLOAT_STRIDE], xmm3 ; write this row for token three
     inc r12d                                  ; advance to the next weight/output row
     mov ecx, [rbx + Q8B_N]                    ; Q8 bytes/row; ecx is dead after the stores
     add rsi, rcx                              ; next weight row
     shr ecx, 4                                ; n / 16 bytes per scale row: (n/64)*4
     add rdi, rcx                              ; next scale row
     jmp .row_loop                             ; continue until every row is processed
.done:
     CLEANUP_YMM                               ; clear YMM upper halves before restoring saved XMM registers
     vmovdqu xmm6, q8b_xmm6                    ; restore callee-saved xmm6 low half
     vmovdqu xmm7, q8b_xmm7                    ; restore callee-saved xmm7 low half
     vmovdqu xmm8, q8b_xmm8                    ; restore callee-saved xmm8 low half
     vmovdqu xmm9, q8b_xmm9                    ; restore callee-saved xmm9 low half
     vmovdqu xmm10, q8b_xmm10                  ; restore callee-saved xmm10 low half
     vmovdqu xmm11, q8b_xmm11                  ; restore callee-saved xmm11 low half
     vmovdqu xmm13, q8b_xmm13                  ; restore callee-saved xmm13 low half
     vmovdqu xmm14, q8b_xmm14                  ; restore callee-saved xmm14 low half
     vmovdqu xmm15, q8b_xmm15                  ; restore callee-saved xmm15 low half
     vmovdqu xmm12, q8b_xmm12                  ; restore callee-saved xmm12 low half
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore the saved GPRs and return

%undef q8b_xmm6
%undef q8b_xmm7
%undef q8b_xmm8
%undef q8b_xmm9
%undef q8b_xmm10
%undef q8b_xmm11
%undef q8b_xmm12
%undef q8b_xmm13
%undef q8b_xmm14
%undef q8b_xmm15
%endmacro
     Q8B_KERNEL v7s_q8_batch4, 0               ; Original four-token kernel
     Q8B_KERNEL v7s_q8_batch4_packed, 1        ; Panel kernel reads interleaved signs and magnitudes
%unmacro Q8B_KERNEL 2                          ; End the shared leaf definition
%undef Q8B_OUT
%undef Q8B_XQ
%undef Q8B_XS
%undef Q8B_WQ
%undef Q8B_WS
%undef Q8B_N
%undef Q8B_D
%undef Q8B_OUT1
%undef Q8B_OUT2
%undef Q8B_OUT3
%undef Q8B_TOKEN_STRIDE
%undef Q8B_TOK0_OFF
%undef Q8B_TOK1_OFF
%undef Q8B_TOK2_OFF
%undef Q8B_TOK3_OFF
%undef Q8B_HALF
%undef Q8B_FLOAT_STRIDE
%unmacro Q8B_DOT_PAIR 7                        ; release the paired-token leaf macro, group-local to this file
