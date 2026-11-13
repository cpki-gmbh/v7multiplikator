; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define SWKQ_ROW1 1                            ; row+1 for dual-row check
%define SWKQ_PAIR_ROWS 2                       ; rows per dual-loop iteration: pointer advance stride
%define SWKQ_SCALE_SHIFT 4                     ; weight bytes -> scale bytes, n/16 per row
%define SWKQ_FEAT_DIM_OFF 48                   ; Feature dimension d, 5th stack arg
%define SWKQ_COL_OFF 56                        ; Column offset, 6th stack arg
%define SWKQ_SCALES_OFF 8                      ; tensor scales field offset, QuantizedTensor.s
%define SWKQ_PF6_OFF 384                       ; prefetch distance: 6 groups * 64 bytes ahead
%define SWKQ_PF7_OFF 448                       ; prefetch distance: 7 groups * 64 bytes ahead
%define SWKQ_YMM32_OFF 32                      ; 32 bytes = high half of 64-byte YMM load
%define SWKQ_FLOAT4_SCALE 4                    ; sizeof,float for SIB scale in addressing

%macro SWKQ_DOT_32 3
; Weight contract: the exporter clamps every Q8 weight byte to [-127, 127]
; quantize_q80, so vpsignb's -128 case never occurs here: vpsignb cannot
; negate -128 in an int8 lane, it would stay -128 instead of becoming +128 .
; With weights in [-127, 127] the unsigned x signed pair product below is
; exact and cannot saturate: the largest two-product magnitude is
; 2*128*127 = 32512 < 2^15, activations may reach -128; vpabsb turns them
; into the unsigned 128, which the maddubsw unsigned operand accepts .
     vmovdqu %1, [%3 + rcx]                    ; low 32B weights
     vmovdqu %2, [%3 + rcx + SWKQ_YMM32_OFF]   ; high 32B weights, independent of the low sign conversion
     vpsignb %1, %1, ymm10                     ; apply sign from activations, low
     vpsignb %2, %2, ymm11                     ; apply sign from activations, high
     vpmaddubsw %1, ymm4, %1                   ; low: unsigned*signed byte -> word
     vpmaddubsw %2, ymm5, %2                   ; high: unsigned*signed byte -> word
     vpmaddwd %1, %1, ymm9                     ; word -> dword, low half
     vpmaddwd %2, %2, ymm9                     ; word -> dword, high half
     vpaddd %1, %1, %2                         ; 8 int32 dot products
     vcvtdq2ps %1, %1                          ; 8 float partial sums
%endmacro

%macro SWKQ_ROW0_SETUP 0
     mov r10d, ebx                             ; row index
     imul r10, swkq_n                          ; gate row byte offset = row * n
     mov rax, r10                              ; copy row*n for scale offset
     shr rax, 4                                ; /16 -> byte offset for scale ptr
     lea r9, [r15 + rax]                       ; r9 = gate ws row0 base, walking
     mov r11, swkq_up_row_bias                 ; hoisted col_off * n
     add r11, r10                              ; r11 = (col_off + row) * n
     add r10, r14                              ; r10 = gate wq row0 ptr, relocated below the r11 use
     mov rax, r11                              ; copy for scale offset
     shr rax, 4                                ; /16 -> byte offset for scale ptr
     mov r8, swkq_up_ws                        ; up scales base
     add r8, rax                               ; r8 = up ws row0 base, walking
     add r11, swkq_up_wq                       ; r11 = up wq row0 ptr, rsi is the row1 pointer in the dual path
%endmacro
     section .text
     SF_START v7s_swiglu_q8_chunk, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: global+label+frame, SF_START emits the export itself
     SF_XMM swkq_xmm6                          ; callee-saved xmm6 low half spill slot, Win64 preserves only the low 128 bits
     SF_XMM swkq_xmm7                          ; callee-saved xmm7 low half spill slot
     SF_XMM swkq_xmm9                          ; callee-saved xmm9 low half spill slot, volatile ymm8 is not saved
     SF_XMM swkq_xmm10                         ; callee-saved xmm10 low half spill slot, written 256-bit by the madd blocks
     SF_XMM swkq_xmm11                         ; callee-saved xmm11 low half spill slot, written 256-bit by the madd blocks
     SF_XMM swkq_xmm12                         ; callee-saved xmm12 low half spill slot
     SF_XMM swkq_xmm13                         ; callee-saved xmm13 low half spill slot
     SF_XMM swkq_xmm14                         ; callee-saved xmm14 low half spill slot
     SF_XMM swkq_xmm15                         ; callee-saved xmm15 low half spill slot
     SF_QWORD swkq_n                           ; input_elements, n, qword for imul
     SF_DWORD swkq_d                           ; feature_dimension, chunk dim
     SF_DWORD swkq_col_off                     ; column_offset for up weight base offset
     SF_QWORD swkq_up_ws                       ; up weight scales base, float*, full tensor
     SF_QWORD swkq_up_wq                       ; up weight quantized base, int8_t*, full tensor, captured once at entry alongside the scales
     SF_QWORD swkq_in_s                        ; input scales base, float*
     SF_QWORD swkq_in_q                        ; input int8_t* xq base, activation
     SF_QWORD swkq_saved_r12                   ; save xout during dual-row inner loop
     SF_DWORD swkq_saved_ebx                   ; save row index during dual-row inner loop
     SF_QWORD swkq_up_row_bias                 ; hoisted col_off * n, call-invariant up-row displacement
     SF_COMMIT                                 ; finalize stack frame layout
     vmovdqu swkq_xmm6, xmm6                   ; save callee-saved xmm6 low half
     vmovdqu swkq_xmm7, xmm7                   ; save callee-saved xmm7 low half
     vmovdqu swkq_xmm9, xmm9                   ; save callee-saved xmm9 low half
     vmovdqu swkq_xmm10, xmm10                 ; save callee-saved xmm10 low half, written 256-bit by the madd blocks
     vmovdqu swkq_xmm11, xmm11                 ; save callee-saved xmm11 low half, written 256-bit by the madd blocks
     vmovdqu swkq_xmm12, xmm12                 ; save callee-saved xmm12 low half
     vmovdqu swkq_xmm13, xmm13                 ; save callee-saved xmm13 low half
     vmovdqu swkq_xmm14, xmm14                 ; save callee-saved xmm14 low half
     vmovdqu swkq_xmm15, xmm15                 ; save callee-saved xmm15 low half
; Generate ones_vec inline at entry, avoid memory load, never overwritten
     vpcmpeqd ymm9, ymm9, ymm9                 ; ymm9 = all-ones, 0xFFFFFFFF per 32-bit lane
     vpabsw ymm9, ymm9                         ; ymm9 = abs,-1 = 1 per 16-bit lane, ones_vec
; --- Load arguments ---
     mov r12, rcx                              ; r12 = xout, output pointer, already offset
     mov swkq_saved_r12, r12                   ; save xout in stack slot for dual-row reuse
     mov r13, [rdx]                            ; r13 = input int8_t* xq, activation quantized, loaded directly
     mov swkq_in_q, r13                        ; activation base for the single-row path, r13 is reused inside the dual path
     mov rax, [rdx+SWKQ_SCALES_OFF]            ; rax = input float* xs, activation scales
     mov swkq_in_s, rax                        ; save input scales base
; Gate weight with its own scales
     mov r14, [r8]                             ; r14 = gate wq, int8_t* with its offset already applied, loaded directly
     mov r15, [r8+SWKQ_SCALES_OFF]             ; r15 = gate ws, float* with its offset already applied
     mov rax, [rel swiglu_up_weight_s]         ; rax = up weight scales, full base
     mov swkq_up_ws, rax                       ; save up scales base for per-row reuse
     mov rax, [rel swiglu_up_weight_q]         ; rax = up weight quantized data, full base; captured once so every row pairs weights and scales from the same tensors
     mov swkq_up_wq, rax                       ; save up wq base for per-row reuse
     mov eax, r9d                              ; zero-extend n to 64-bit, upper bits undefined per ABI
     mov swkq_n, rax                           ; save n for 64-bit imul, cleanly zero-extended
     mov eax, [rbp+SWKQ_FEAT_DIM_OFF]          ; d from stack arg 5
     mov swkq_d, eax                           ; save feature dimension
     mov eax, [rbp+SWKQ_COL_OFF]               ; column_offset from stack arg 6
     mov swkq_col_off, eax                     ; save column offset
     mov eax, swkq_col_off                     ; column offset, call-invariant
     imul rax, swkq_n                          ; rax = col_off * n
     mov swkq_up_row_bias, rax                 ; park the displacement for every SWKQ_ROW0_SETUP expansion
; ===== Row loop, dual-row main path, single-row fallback for odd rows =====
     xor ebx, ebx                              ; row index = 0
; Row0 setup runs ONCE: r10/r11/r9/r8 carry the current row's bases across
; every outer iteration and are advanced by two rows after each dual store.
     SWKQ_ROW0_SETUP                           ; r9/r8 = scale bases, r10/r11 = weight row ptrs
     align 32
.outer_swkq:
     cmp ebx, swkq_d                           ; row >= d?
     jge .done_swkq                            ; all rows done
; Check if 2 or more rows remain, dual path is the hot case: d/2-1 of d/2
; outer iterations, it FALLS THROUGH; only the final odd row jumps out
     lea eax, [ebx + SWKQ_ROW1]                ; eax = row + 1
     cmp eax, swkq_d                           ; row+1 < d?
     jge .single_row_swkq                      ; no -> last odd row, single-row fallback
; ===== Dual-row path, process 2 rows per iteration =====
     align 32
.dual_row_swkq:
; Save registers that will be repurposed during the inner loop
     mov swkq_saved_ebx, ebx                   ; save current row index
; Row1 base computation, row1 = row0 + n in weight space; the row0 bases in
; r10/r11/r9/r8 were set up at entry and are carried across iterations --
     mov rdx, swkq_n                           ; rdx = n
     lea rdi, [r10 + rdx]                      ; rdi = gate wq row1 base, repurpose rdi as temp
     lea rsi, [r11 + rdx]                      ; rsi = up wq row1 base, r15 keeps the gate ws base for the whole call
     shr edx, 4                                ; edx = scale_stride = n/16 for row1 scale offset
; Activation base in r12, xout saved in swkq_saved_r12 at function entry
     mov r12, swkq_in_q                        ; r12 = activation base, overwrite xout temporarily
; Fold the row stride into two invariant row1 scale bases, then rdx becomes the one
; walking byte offset shared by all five scale streams, n/64 groups per row
     lea rbx, [r9 + rdx]                       ; rbx = gate ws row1 base, invariant across groups
     lea r13, [r8 + rdx]                       ; r13 = up ws row1 base, invariant across groups
     xor edx, edx                              ; rdx = walking scale byte offset
; Zero 4 vector accumulators
     vxorps ymm0, ymm0, ymm0                   ; gate_r0 accumulator
     vxorps ymm1, ymm1, ymm1                   ; up_r0 accumulator
     vxorps ymm2, ymm2, ymm2                   ; gate_r1 accumulator
     vxorps ymm3, ymm3, ymm3                   ; up_r1 accumulator
     xor ecx, ecx                              ; byte offset = 0
     mov rax, swkq_in_s                        ; rax = input scales base, invariant
; ===== Dual-row fused inner loop, 4 dot products per group =====
     align 64
.dual_q8_loop:
; rcx = 0 at entry, swkq_n >= 64, n%64==0 FATAL-guarded : first top-compare
; would always fall through, bottom-test runs identical iterations
; Prefetch both rows' weights, 8 prefetches, slots 31-38, spread across the chain
     ds prefetcht0 [r10 + rcx + SWKQ_PF6_OFF]  ; prefetch gate row0 6 groups ahead, slot 31
     PREFETCH_SLOT 31                          ; Slot 31: swiglu_q8 dual gate row0 pf6
; Activation load, shared across both rows --
     vbroadcastss ymm14, [rax + rdx]           ; input scale
     vmovdqu ymm10, [r12 + rcx]                ; low 32B activations, early
     vmovdqu ymm11, [r12 + rcx + SWKQ_YMM32_OFF] ; high 32B activations, early
     ds prefetcht0 [rdi + rcx + SWKQ_PF6_OFF]  ; prefetch gate row1 6 groups ahead, slot 32
     PREFETCH_SLOT 32                          ; Slot 32: swiglu_q8 dual gate row1 pf6
     vbroadcastss ymm13, [r9 + rdx]            ; gate row0 scale
     vbroadcastss ymm15, [r8 + rdx]            ; up row0 scale
     vpabsb ymm4, ymm10                        ; unsigned low activations
     vpabsb ymm5, ymm11                        ; unsigned high activations
     vmulps ymm13, ymm13, ymm14                ; combined gate row0 scale
     vmulps ymm15, ymm15, ymm14                ; combined up row0 scale
     ds prefetcht0 [r11 + rcx + SWKQ_PF6_OFF]  ; prefetch up row0 6 groups ahead, slot 33
     PREFETCH_SLOT 33                          ; Slot 33: swiglu_q8 dual up row0 pf6
     vbroadcastss ymm8, [rbx + rdx]            ; gate row1 scale, independent of the dots
     vmulps ymm8, ymm8, ymm14                  ; combined gate row1 scale
     SWKQ_DOT_32 ymm6, ymm7, r10               ; gate row0 dot
     ds prefetcht0 [rsi + rcx + SWKQ_PF6_OFF]  ; prefetch up row1 6 groups ahead, slot 34
     PREFETCH_SLOT 34                          ; Slot 34: swiglu_q8 dual up row1 pf6
     vfmadd231ps ymm0, ymm6, ymm13             ; gate_r0_acc += dot * combined gate scale
     SWKQ_DOT_32 ymm12, ymm13, r11             ; up row0 dot
     ds prefetcht0 [r10 + rcx + SWKQ_PF7_OFF]  ; prefetch gate row0 7 groups ahead, slot 35
     PREFETCH_SLOT 35                          ; Slot 35: swiglu_q8 dual gate row0 pf7
     vfmadd231ps ymm1, ymm12, ymm15            ; up_r0_acc += dot * combined up scale
     vbroadcastss ymm15, [r13 + rdx]           ; row1 up scale
     vmulps ymm15, ymm15, ymm14                ; combined up row1 scale
     ds prefetcht0 [rdi + rcx + SWKQ_PF7_OFF]  ; prefetch gate row1 7 groups ahead, slot 36
     PREFETCH_SLOT 36                          ; Slot 36: swiglu_q8 dual gate row1 pf7
     SWKQ_DOT_32 ymm6, ymm7, rdi               ; row1 gate dot
     ds prefetcht0 [r11 + rcx + SWKQ_PF7_OFF]  ; prefetch up row0 7 groups ahead, slot 37
     PREFETCH_SLOT 37                          ; Slot 37: swiglu_q8 dual up row0 pf7
     vfmadd231ps ymm2, ymm6, ymm8              ; gate_r1_acc += dot * combined gate scale
     SWKQ_DOT_32 ymm12, ymm13, rsi             ; row1 up dot
     ds prefetcht0 [rsi + rcx + SWKQ_PF7_OFF]  ; prefetch up row1 7 groups ahead, slot 38
     PREFETCH_SLOT 38                          ; Slot 38: swiglu_q8 dual up row1 pf7
     vfmadd231ps ymm3, ymm12, ymm15            ; up_r1_acc += dot * combined up scale
; Advance the byte offset and the ONE walking scale index --
     add rcx, 64                               ; advance to next 64-byte group
     add rdx, 4                                ; advance the shared scale byte offset, all five streams
     cmp rcx, swkq_n                           ; all groups done?, swkq_n is L1-resident; frees rbx for a scale base
     jb .dual_q8_loop                          ; no, next group, fall through to reduce when done
     align 16
.dual_q8_reduce:
; Restore saved registers for xout and the row index
     mov r12, swkq_saved_r12                   ; restore xout from function-entry save
     mov ebx, swkq_saved_ebx                   ; restore current row index
; Four identical per-accumulator trees, interleaved for four independent chains.
     HADD4_YMM                                 ; gate_r0, up_r0, gate_r1, up_r1
; Fuse both rows: packed sigmoid on [gate_r0, gate_r1]
     vinsertps xmm6, xmm0, xmm2, 0x1C          ; xmm6 = [gate_r0, gate_r1, 0, 0], imm bits 3-2 zero lanes 2/3 so the packed sigmoid runs on defined values
     vinsertps xmm7, xmm1, xmm3, 0x1C          ; xmm7 = [up_r0, up_r1, 0, 0]
     vmovaps xmm0, xmm6                        ; packed gates for sigmoid
     EVAL_SIGMOID x, ps, [rel swk_sign_mask_ymm] ; packed 2-wide sigmoid
     vmulps xmm0, xmm0, xmm6                   ; packed sig*gate
     vmulps xmm0, xmm0, xmm7                   ; packed sig*gate*up
     vmovq [r12 + rbx*SWKQ_FLOAT4_SCALE], xmm0 ; 64-bit store both results
     mov rax, swkq_n                           ; n for the two-row pointer advance
     lea r10, [r10 + rax*SWKQ_PAIR_ROWS]       ; gate Q8: +2*n bytes, next dual pair
     lea r11, [r11 + rax*SWKQ_PAIR_ROWS]       ; up Q8: +2*n bytes
     shr rax, SWKQ_SCALE_SHIFT                 ; scale stride = n/16 bytes
     lea r9, [r9 + rax*SWKQ_PAIR_ROWS]         ; gate scales: +2*(n/16) bytes
     lea r8, [r8 + rax*SWKQ_PAIR_ROWS]         ; up scales: +2*(n/16) bytes
     add ebx, 2                                ; advance row index by 2
     jmp .outer_swkq                           ; next iteration
; ===== Single-row path, last odd row; entered ONLY via the outer-loop jge =====
.single_row_swkq:
     mov rdi, swkq_in_q                        ; rdi = activation quantized data base; the carried
; r10/r11/r9/r8 already point at this final odd row
     mov rax, swkq_in_s                        ; rax = input scales base, invariant
     vxorps ymm0, ymm0, ymm0                   ; gate float vector acc = 0
     vxorps ymm1, ymm1, ymm1                   ; up float vector acc = 0
     xor ecx, ecx                              ; byte offset = 0
     mov rdx, swkq_n                           ; rdx = n loop bound, register-held: rdx dead in the single path, saves a load per group
     xor r13d, r13d                            ; walking scale byte offset, zeroes the full 64-bit register
     align 64
.single_q8_loop:
; rcx = 0 at entry, rdx = n >= 64, n%64==0 FATAL-guarded : first top-compare
; would always fall through, bottom-test runs identical iterations
     ds prefetcht0 [r10 + rcx + SWKQ_PF6_OFF]  ; prefetch gate weights 6 groups ahead, slot 27
     PREFETCH_SLOT 27                          ; Slot 27: swiglu_q8 gate weights pf6
     vbroadcastss ymm14, [rax + r13]           ; input scale
     vbroadcastss ymm15, [r8 + r13]            ; up scale
     vmulps ymm15, ymm15, ymm14                ; combined up scale
     vbroadcastss ymm8, [r9 + r13]             ; gate scale, independent of the dots
     vmulps ymm8, ymm8, ymm14                  ; combined gate scale
     vmovdqu ymm10, [rdi + rcx]                ; low 32B activations
     vmovdqu ymm11, [rdi + rcx + SWKQ_YMM32_OFF] ; high 32B activations
     ds prefetcht0 [r11 + rcx + SWKQ_PF6_OFF]  ; prefetch up weights 6 groups ahead, slot 28
     PREFETCH_SLOT 28                          ; Slot 28: swiglu_q8 up weights pf6
     vpabsb ymm4, ymm10                        ; unsigned low activations
     vpabsb ymm5, ymm11                        ; unsigned high activations
     vmovdqu ymm6, [r10 + rcx]                 ; gate low
     vmovdqu ymm12, [r11 + rcx]                ; up low
     vmovdqu ymm7, [r10 + rcx + SWKQ_YMM32_OFF] ; gate high
     vmovdqu ymm13, [r11 + rcx + SWKQ_YMM32_OFF] ; up high
     vpsignb ymm6, ymm6, ymm10                 ; gate sign from activations, low
     vpsignb ymm12, ymm12, ymm10               ; up sign from activations, low
     vpsignb ymm7, ymm7, ymm11                 ; gate sign from activations, high
     vpsignb ymm13, ymm13, ymm11               ; up sign from activations, high
     ds prefetcht0 [r10 + rcx + SWKQ_PF7_OFF]  ; prefetch gate weights 7 groups ahead, slot 29
     PREFETCH_SLOT 29                          ; Slot 29: swiglu_q8 gate weights pf7
     vpmaddubsw ymm6, ymm4, ymm6               ; gate low: unsigned*signed byte -> word
     vpmaddubsw ymm12, ymm4, ymm12             ; up low: unsigned*signed byte -> word
     vpmaddubsw ymm7, ymm5, ymm7               ; gate high: unsigned*signed byte -> word
     vpmaddubsw ymm13, ymm5, ymm13             ; up high: unsigned*signed byte -> word
     vpmaddwd ymm6, ymm6, ymm9                 ; gate low: word -> dword
     vpmaddwd ymm12, ymm12, ymm9               ; up low: word -> dword
     vpmaddwd ymm7, ymm7, ymm9                 ; gate high: word -> dword
     vpmaddwd ymm13, ymm13, ymm9               ; up high: word -> dword
     ds prefetcht0 [r11 + rcx + SWKQ_PF7_OFF]  ; prefetch up weights 7 groups ahead, slot 30
     PREFETCH_SLOT 30                          ; Slot 30: swiglu_q8 up weights pf7
     vpaddd ymm6, ymm6, ymm7                   ; gate 8 int32 dot products
     vpaddd ymm12, ymm12, ymm13                ; up 8 int32 dot products
     vcvtdq2ps ymm6, ymm6                      ; gate 8 float partial sums
     vcvtdq2ps ymm12, ymm12                    ; up 8 float partial sums
     vfmadd231ps ymm0, ymm6, ymm8              ; gate_acc += gate_dot * combined_gate_scale
     vfmadd231ps ymm1, ymm12, ymm15            ; up_acc += up_dot * combined_up_scale
     add rcx, 64                               ; advance to next 64-byte group
     add r13, 4                                ; advance the shared scale byte offset, all three streams
     cmp rcx, rdx                              ; all groups done?, register compare, no memory operand
     jb .single_q8_loop                        ; no, next group, fall through to reduce when done
.single_q8_reduce:
     HADD_YMM xmm0, ymm0, xmm2                 ; Reduce gate row0 to scalar
     HADD_YMM xmm1, ymm1, xmm2                 ; Reduce up row0 to scalar
     vmovaps xmm7, xmm0                        ; park the gate value, VEX.128 register move, no legacy-SSE state; EVAL_SIGMOID overwrites xmm0-5 + eax in ss mode
     vmovaps xmm6, xmm1                        ; park the up value
     EVAL_SIGMOID x, ss, [rel swk_sign_mask]   ; xmm0 = sigmoid,gate
     vmulss xmm0, xmm0, xmm7                   ; xmm0 = sigmoid,gate * gate
     vmulss xmm0, xmm0, xmm6                   ; xmm0 = gate * sigmoid,gate * up
     vmovss [r12 + rbx*SWKQ_FLOAT4_SCALE], xmm0 ; xout[row] = SwiGLU result
     inc ebx                                   ; advance row index
     jmp .outer_swkq                           ; next row
; ===== Epilogue =====
.done_swkq:
     CLEANUP_YMM                               ; vzeroupper BEFORE the restores: ABI volatility of the upper YMM halves means only the low 128 bits need to survive
     vmovdqu xmm15, swkq_xmm15                 ; restore callee-saved xmm15 low half
     vmovdqu xmm14, swkq_xmm14                 ; restore callee-saved xmm14 low half
     vmovdqu xmm13, swkq_xmm13                 ; restore callee-saved xmm13 low half
     vmovdqu xmm12, swkq_xmm12                 ; restore callee-saved xmm12 low half
     vmovdqu xmm11, swkq_xmm11                 ; restore callee-saved xmm11 low half
     vmovdqu xmm10, swkq_xmm10                 ; restore callee-saved xmm10 low half
     vmovdqu xmm9, swkq_xmm9                   ; restore callee-saved xmm9 low half
     vmovdqu xmm7, swkq_xmm7                   ; restore callee-saved xmm7 low half
     vmovdqu xmm6, swkq_xmm6                   ; restore callee-saved xmm6 low half
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; epilogue: restore non-volatiles, return
%unmacro SWKQ_ROW0_SETUP 0                     ; remove: the .dual_row_swkq and .single_row_swkq expansions above are its only uses
%unmacro SWKQ_DOT_32 3                         ; remove: the six dual/single-path expansions above are its only uses
%undef SWKQ_FLOAT4_SCALE                       ; file-local float SIB scale
%undef SWKQ_YMM32_OFF                          ; file-local high-half load offset
%undef SWKQ_PF7_OFF                            ; file-local 7-group prefetch distance
%undef SWKQ_PF6_OFF                            ; file-local 6-group prefetch distance
%undef SWKQ_SCALES_OFF                         ; file-local QuantizedTensor.s field offset
%undef SWKQ_COL_OFF                            ; file-local 6th stack-arg offset
%undef SWKQ_FEAT_DIM_OFF                       ; file-local 5th stack-arg offset
%undef SWKQ_ROW1                               ; file-local dual-row check increment
; ---- SF slot cleanup: the 17 swkq_* frame slots must not leak, NASM re-defines textually ----
%undef swkq_saved_ebx
%undef swkq_saved_ebx_raw
%undef swkq_saved_r12
%undef swkq_saved_r12_raw
%undef swkq_in_q
%undef swkq_in_q_raw
%undef swkq_in_s
%undef swkq_in_s_raw
%undef swkq_up_wq
%undef swkq_up_wq_raw
%undef swkq_up_ws
%undef swkq_up_ws_raw
%undef swkq_n
%undef swkq_n_raw
%undef swkq_d
%undef swkq_d_raw
%undef swkq_col_off
%undef swkq_col_off_raw
%undef swkq_up_row_bias
%undef swkq_up_row_bias_raw
%undef swkq_xmm6
%undef swkq_xmm7
%undef swkq_xmm9
%undef swkq_xmm10
%undef swkq_xmm11
%undef swkq_xmm12
%undef swkq_xmm13
%undef swkq_xmm14
%undef swkq_xmm15
