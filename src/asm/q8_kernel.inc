; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define SF_ARG_D_OFF 48                        ; 5th stack arg d, frame offset 48; rbp is set at entry, before the non-volatile pushes
%define Q8K_FLOAT4_SCALE 4                     ; scale factor for float pointer arithmetic, sizeof,float
%define Q8K_32B_OFF 32                         ; 32-byte offset for high half of 64-byte loads
%define Q8K_PREFETCH_DIST 256                  ; prefetch distance in bytes for weight rows
%define Q8K_ROW2_OFF 8                         ; byte offset of row 2 within the output row, 2 floats past row 0, the base for the SECOND packed store: rows 0+1 already went out at +0, and rows 2+3 travel together in xmm2's low 64 bits, so this displacement lands row 2 and needs no further one for row 3
%define Q8K_SMC_DISP32_PLACEHOLDER 0x12345678  ; oversized placeholder value: forces NASM to emit disp32 for the SMC-patched scale displacements

; Two weight rows' integer dot chains interleaved over the shared activation
; vectors (ymm4/ymm5 abs input, ymm10/ymm11 sign input, ymm9 word-ones, ymm8
; activation scale). Per row the arithmetic sequence is identical: sign-correct
; weights, unsigned-abs madd, word pair-add, low+high combine, int32-to-float
; convert, combined-scale multiply, correctly-rounded FMA.
; The patchable prefetches keep their original per-row targets and slots and
; are spread across the chain, the own-row prefetch after the madd wave and
; the next-row prefetch after each combine, so consecutive prefetches never
; sit closer than five instructions apart.

%macro Q8K_PAIR_DOT 11                         ; %1/%2 weight ptrs, %3/%4 scale row numbers, %5/%6 accs, %7 ownA, %8 nextA, %9 ownB, %10 nextB ptr, %11 nextB slot
     vmovdqu ymm6, [%1 + rcx]                  ; row A weight low 32B
     vmovdqu ymm7, [%1 + rcx + V7M_Q8_HI_OFF]  ; row A weight high 32B
     ds prefetchnta [%1 + rcx + V7M_Q8_PF_OWN] ; row A own-row prefetch, patchable
     PREFETCH_SLOT %7                          ; mark patchable slot for own +288
     vmovdqu ymm12, [%2 + rcx]                 ; row B weight low 32B
     vmovdqu ymm13, [%2 + rcx + V7M_Q8_HI_OFF] ; row B weight high 32B
     vpsignb ymm6, ymm6, ymm10                 ; row A sign correction low
     vpsignb ymm12, ymm12, ymm10               ; row B sign correction low
     vpsignb ymm7, ymm7, ymm11                 ; row A sign correction high
     vpsignb ymm13, ymm13, ymm11               ; row B sign correction high
     vpmaddubsw ymm6, ymm4, ymm6               ; row A pairwise madd low
     vpmaddubsw ymm12, ymm4, ymm12             ; row B pairwise madd low
     vpmaddubsw ymm7, ymm5, ymm7               ; row A pairwise madd high
     vpmaddubsw ymm13, ymm5, ymm13             ; row B pairwise madd high
     ds prefetchnta [%2 + rcx + V7M_Q8_PF_OWN] ; row B own-row prefetch, patchable
     PREFETCH_SLOT %9                          ; mark patchable slot for own +288
     vpmaddwd ymm6, ymm6, ymm9                 ; row A pair-add low
     vpmaddwd ymm12, ymm12, ymm9               ; row B pair-add low
     vpmaddwd ymm7, ymm7, ymm9                 ; row A pair-add high
     vpmaddwd ymm13, ymm13, ymm9               ; row B pair-add high
     vpaddd ymm6, ymm6, ymm7                   ; row A combine low+high
     ds prefetchnta [%2 + rcx + V7M_Q8_PF_NEXT] ; row A next-row prefetch (row B), patchable
     PREFETCH_SLOT %8                          ; mark patchable slot for next +256
; Scale base is r15, with row 1..3 byte offsets patched once per shared n.
; The 0x12345678 placeholder FORCES disp32 encoding. Each label lands
; exactly four bytes after the displacement; SMC only replaces that dword.
; Both broadcast blocks issue right after their row's combine, so the
; independent scale loads overlap the dot-to-float conversions.
%if %3 = 0
     vbroadcastss ymm7, [r15 + r9*Q8K_FLOAT4_SCALE] ; row 0 scale, no displacement
%elif %3 = 2
     vbroadcastss ymm7, [r15 + r9*Q8K_FLOAT4_SCALE + Q8K_SMC_DISP32_PLACEHOLDER] ; row 2 scale, patched disp32
.scale2_disp_end:                              ; disp32 immediately precedes this label
%else
%error Q8K_PAIR_DOT row A must be 0 or 2       ; row A of the pair is fixed to rows 0/2
%endif
     vpaddd ymm12, ymm12, ymm13                ; row B combine low+high
%if %4 = 1
     vbroadcastss ymm13, [r15 + r9*Q8K_FLOAT4_SCALE + Q8K_SMC_DISP32_PLACEHOLDER] ; row 1 scale, patched disp32
.scale1_disp_end:                              ; disp32 immediately precedes this label
%elif %4 = 3
     vbroadcastss ymm13, [r15 + r9*Q8K_FLOAT4_SCALE + Q8K_SMC_DISP32_PLACEHOLDER] ; row 3 scale, patched disp32
.scale3_disp_end:                              ; disp32 immediately precedes this label
%else
%error Q8K_PAIR_DOT row B must be 1 or 3       ; row B of the pair is fixed to rows 1/3
%endif
     vcvtdq2ps ymm6, ymm6                      ; row A dot to float
     vcvtdq2ps ymm12, ymm12                    ; row B dot to float, issued with row A
     vmulps ymm7, ymm7, ymm8                   ; row A combined scale
     vmulps ymm13, ymm13, ymm8                 ; row B combined scale
     vfmadd231ps %5, ymm6, ymm7                ; row A accumulator FMA
%if %11 >= 0
     ds prefetchnta [%10 + rcx + V7M_Q8_PF_NEXT] ; row B next-row prefetch, patchable
     PREFETCH_SLOT %11                         ; mark patchable slot for next +256
%endif
     vfmadd231ps %6, ymm12, ymm13              ; row B accumulator FMA
%endmacro

     section .text
     global v7s_q8_chunk                       ; export Q8 chunk kernel
v7s_q8_chunk:
     SF_START v7s_q8_chunk, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue, save non-volatiles
     SF_XMM q8k_xmm6                           ; xmm6 spill slot, upper 128 bits are volatile per Win64 ABI
     SF_XMM q8k_xmm7                           ; xmm7 spill slot
     SF_XMM q8k_xmm8                           ; xmm8 spill slot
     SF_XMM q8k_xmm9                           ; xmm9 spill slot
     SF_XMM q8k_xmm10                          ; xmm10 spill slot
     SF_XMM q8k_xmm11                          ; xmm11 spill slot
     SF_XMM q8k_xmm12                          ; xmm12 spill slot, row-B pair temp
     SF_XMM q8k_xmm13                          ; xmm13 spill slot, row-B pair temp
     SF_COMMIT                                 ; commit stack frame locals
; Save callee-saved YMM registers, must be after SF_COMMIT, stack frame is allocated
     vmovdqu q8k_xmm6, xmm6                    ; save callee-saved xmm6, lower 128 bits only
     vmovdqu q8k_xmm7, xmm7                    ; save callee-saved xmm7
     vmovdqu q8k_xmm8, xmm8                    ; save callee-saved xmm8
     vmovdqu q8k_xmm9, xmm9                    ; save callee-saved xmm9
     vmovdqu q8k_xmm10, xmm10                  ; save callee-saved xmm10
     vmovdqu q8k_xmm11, xmm11                  ; save callee-saved xmm11
     vmovdqu q8k_xmm12, xmm12                  ; save callee-saved xmm12
     vmovdqu q8k_xmm13, xmm13                  ; save callee-saved xmm13
; Generate ones vector for vpmaddwd reduction inline, avoid memory load
     vpcmpeqd ymm9, ymm9, ymm9                 ; ymm9 = all-ones, 0xFFFFFFFF per 32-bit lane
     vpabsw ymm9, ymm9                         ; ymm9 = abs,-1 = 1 per 16-bit lane, ones_vec
; All per-job values remain register-resident, including d: jobs in a
; split/taper dispatch can have DIFFERENT row counts, so d is NEVER an SMC
; immediate.
; rdi/r15 are advancing row-0 weight/scale pointers; r12=n, r13=d, r8=n/16.
     mov r14, rcx                              ; output base (constant)
     mov r12d, r9d                             ; input dimension n (constant within this job)
     mov r13d, dword [rbp + SF_ARG_D_OFF]      ; this job's variable d (preserve in r13)
     mov rbx, [rdx]                            ; quantized input data, loop invariant
     mov r11, [rdx + TENSOR_S_OFF]             ; input scale base, loop invariant
     mov rdi, [r8]                             ; initial row-0 weight pointer, advances by n/row
     mov r15, [r8 + TENSOR_S_OFF]              ; initial row-0 scale pointer, advances by n/16/row
     mov r8d, r12d                             ; copy n for scale-row stride
     shr r8d, 4                                ; scale-row stride in bytes, n/16
     xor r10d, r10d                            ; output row index, starts at zero
.outer_loop:
     mov eax, r13d                             ; job-local output row count d
     sub eax, r10d                             ; unsigned remaining rows, d-row_index
     jz .done                                  ; exact end of job
     cmp eax, 4                                ; four output rows available?
     jl .tail_rows                             ; one to three remaining
; Weight row pointers advance across iterations.
; rdi and r15 always point at the first unprocessed output row.
     lea rsi, [rdi + r12]                      ; weight row 1
     lea rdx, [rsi + r12]                      ; weight row 2
     lea rax, [rdx + r12]                      ; weight row 3
; Zero out 4 row accumulators, ymm0-ymm3
     vxorps ymm0, ymm0, ymm0                   ; row 0 accumulator = 0
     vxorps ymm1, ymm1, ymm1                   ; row 1 accumulator = 0
     vxorps ymm2, ymm2, ymm2                   ; row 2 accumulator = 0
     vxorps ymm3, ymm3, ymm3                   ; row 3 accumulator = 0
     xor ecx, ecx                              ; ecx = byte offset into activation/weight vectors
; Weight rows already live in rdi/rsi/rdx/rax, chain above ; rbx holds x
; activation pointer, since entry ; r11 holds input scales, since entry .
     xor r9d, r9d                              ; r9d = scale group index
     align 32                                  ; align hot inner loop entry to 32 bytes, A9a
.k_loop:
; Load 64 bytes of input activations, int8 and compute sign vectors
; Broadcast activation scale for this 64-byte group as early as possible
     vbroadcastss ymm8, [r11 + r9*Q8K_FLOAT4_SCALE] ; s_act = input_scales[scale_idx]
     vmovdqu ymm10, [rbx + rcx]                ; load low 32B of activations
     vmovdqu ymm11, [rbx + rcx + Q8K_32B_OFF]  ; load high 32B of activations
     vpabsb ymm4, ymm10                        ; absolute value of activations low
     vpabsb ymm5, ymm11                        ; absolute value of activations high
; Row 0: initial +256 prefetch, only row 0 has this, no prior row to prefetch for it
     ds prefetchnta [rdi + rcx + Q8K_PREFETCH_DIST] ; prefetch weight row 0 ahead
     PREFETCH_SLOT 0                           ; mark patchable slot 0 for SMC
; 4-row unrolled madd block, rows 0+1 and 2+3 interleaved pair-wise
     Q8K_PAIR_DOT rdi, rsi, 0, 1, ymm0, ymm1, 1, 2, 3, rdx, 4 ; rows 0+1, row-scale disps 0/S
     Q8K_PAIR_DOT rdx, rax, 2, 3, ymm2, ymm3, 5, 6, 7, 0, -1 ; rows 2+3, row-scale disps 2S/3S
     add ecx, 64                               ; advance byte offset by 64, one cache line
     inc r9d                                   ; advance scale group index by 1
     cmp ecx, r12d                             ; offset < n? r12=n, live and unused inside the k_loop
     jne .k_loop                               ; more k to process -> continue loop
     align 32                                  ; Align store results entry to 32 bytes
.store_results:
; Reduce each row accumulator to scalar, one shared 4-chain reduction sequence
     HADD4_YMM                                 ; four independent reductions, identical lane-wise addition tree
     vunpcklps xmm0, xmm0, xmm1                ; pack [row0, row1] low 64 bits, uses low 32-bit lanes only
     vunpcklps xmm2, xmm2, xmm3                ; pack [row2, row3] low 64 bits, uses low 32-bit lanes only
     vmovq [r14 + r10*Q8K_FLOAT4_SCALE], xmm0  ; store rows 0+1 as one 64-bit write, halves store-port pressure
     vmovq [r14 + r10*Q8K_FLOAT4_SCALE + Q8K_ROW2_OFF], xmm2 ; store rows 2+3 as one 64-bit write
     lea rdi, [rax + r12]                      ; advance row-0 weight pointer by four n-byte rows
     lea r15, [r15 + r8*Q8K_FLOAT4_SCALE]      ; advance row-0 scale pointer by four stride-byte rows
     add r10d, 4                               ; advance output row index by four
     jmp .outer_loop                           ; process next 4 rows
.tail_rows:
; rdi/r15 already point to this row: no base reload or row-index multiply.
     vxorps ymm0, ymm0, ymm0                   ; all 8 accumulator lanes = 0, ymm0 is row 0's accumulator in the 4-row path, so none of the lanes may start as leftover garbage, vfmadd231ps below reads all 8
     xor ecx, ecx                              ; byte offset = 0
     xor r9d, r9d                              ; tail scale group index = 0; r8 remains row stride
.tail_loop:
     vbroadcastss ymm8, [r11 + r9*Q8K_FLOAT4_SCALE] ; input scale for this group
     vmovdqu ymm1, [rbx + rcx]                 ; activation low
     vmovdqu ymm2, [rbx + rcx + Q8K_32B_OFF]   ; activation high
     vmovdqu ymm6, [rdi + rcx]                 ; weight low from advancing row pointer
     vmovdqu ymm7, [rdi + rcx + Q8K_32B_OFF]   ; weight high
     vpabsb ymm3, ymm1                         ; activation abs low
     vpabsb ymm4, ymm2                         ; activation abs high
     vpsignb ymm6, ymm6, ymm1                  ; corrected weight low
     vpsignb ymm7, ymm7, ymm2                  ; corrected weight high
     vpmaddubsw ymm6, ymm3, ymm6               ; pairwise madd, low
     vpmaddubsw ymm7, ymm4, ymm7               ; pairwise madd, high
     vpmaddwd ymm6, ymm6, ymm9                 ; horizontal pair-add, low
     vpmaddwd ymm7, ymm7, ymm9                 ; horizontal pair-add, high
     vpaddd ymm6, ymm6, ymm7                   ; combine low+high dot products, ymm7 dies here
     vbroadcastss ymm7, [r15 + r9*Q8K_FLOAT4_SCALE] ; current row scale, no patched displacement
     vmulps ymm7, ymm7, ymm8                   ; compute the combined scale while the dot converts
     vcvtdq2ps ymm6, ymm6                      ; independent int32 dot -> float
     vfmadd231ps ymm0, ymm6, ymm7              ; same FMA operand order
     add ecx, 64                               ; advance to next 64-byte group
     inc r9d                                   ; advance to next scale group
     cmp ecx, r12d                             ; offset < n?
     jb .tail_loop                             ; yes -> continue, bottom-tested, no tail jmp
; loop exit is the fall-through of the `jb` above, there is no separate
; store label, nothing branches here
     HADD_YMM xmm0, ymm0, xmm4                 ; reduce the 8 lanes exactly as .store_results does per row, xmm4 is dead scratch by here
     vmovss [r14 + r10*Q8K_FLOAT4_SCALE], xmm0 ; store dot product result
     add rdi, r12                              ; advance weight pointer to next row
     add r15, r8                               ; advance scale pointer by this job's row stride
     inc r10d                                  ; advance to next row
     jmp .outer_loop                           ; continue outer loop
.done:
     CLEANUP_YMM                               ; vzeroupper BEFORE restoring callee-saved XMMs, ABI compliance
     vmovdqu xmm6, q8k_xmm6                    ; restore callee-saved xmm6, lower 128 bits
     vmovdqu xmm7, q8k_xmm7                    ; restore callee-saved xmm7
     vmovdqu xmm8, q8k_xmm8                    ; restore callee-saved xmm8
     vmovdqu xmm9, q8k_xmm9                    ; restore callee-saved xmm9
     vmovdqu xmm10, q8k_xmm10                  ; restore callee-saved xmm10
     vmovdqu xmm11, q8k_xmm11                  ; restore callee-saved xmm11
     vmovdqu xmm12, q8k_xmm12                  ; restore callee-saved xmm12
     vmovdqu xmm13, q8k_xmm13                  ; restore callee-saved xmm13
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; epilogue, restore non-volatiles and return

%unmacro Q8K_PAIR_DOT 11                       ; release the paired-row leaf macro, group-local to this file
