; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; Stochastic sampler: three independent 2-core phases. This include is placed
; immediately after v7s_sample and BEFORE the S_EXP256_SOFT_PAIR unmacro.
; The first two phases yield into a tight helper spin; the third rejoins pool.
; No worker branch accesses the caller's frame. Only RAX/RBX/RCX/RDX transfer.

; Numerical note: per-element scale/exp/divide use the baseline instructions.
; Splitting the exp accumulation changes the last few sum bits (expected).

     section .text

%define STOCH_SPLIT_EXP_OFF 4                  ; byte offset of the exponential-stage fraction inside stoch_split_percentage
%define STOCH_SPLIT_NORM_OFF 8                 ; byte offset of the normalization-stage fraction inside stoch_split_percentage
%define STOCH_MIN_RANGE 32                     ; minimum float count each side of a stochastic split

; ymm0, ymm7: 8 input x-max values each; outputs overwrite the same registers.
; Live across expansion: ymm1=C6, ymm5=1.0, ymm6=global max, ymm12-ymm15=partial sums.
; Scratch A: ymm2,ymm3,ymm4. Scratch B: ymm8,ymm9,ymm10.
; With %1=1, load blocks C/D into ymm0/ymm7 and return A/B in ymm3/ymm8.
; Exact per-lane instruction sequence matches s_exp256_soft, two interleaved chains.

%macro S_EXP256_SOFT_PAIR 1
     vmaxps ymm0, ymm0, [rel s_negclamp_v]     ; chain A: clamp input at -87.3365, exp domain floor
     vmaxps ymm7, ymm7, [rel s_negclamp_v]     ; chain B: clamp input at -87.3365, exp domain floor
     vmulps ymm3, ymm0, [rel s_log2e_v]        ; chain A: x * log2(e), argument reduction base
     vmulps ymm8, ymm7, [rel s_log2e_v]        ; chain B: x * log2(e), argument reduction base
     vroundps ymm3, ymm3, 0                    ; chain A: n = round-to-nearest of x*log2(e)
     vroundps ymm8, ymm8, 0                    ; chain B: n = round-to-nearest of x*log2(e)
     vcvttps2dq ymm2, ymm3                     ; chain A: integer exponent n for the 2^n construction
     vcvttps2dq ymm9, ymm8                     ; chain B: integer exponent n for the 2^n construction
     vfnmadd231ps ymm0, ymm3, [rel s_ln2hi_v]  ; chain A: r -= n*ln2hi, high precision part
     vfnmadd231ps ymm7, ymm8, [rel s_ln2hi_v]  ; chain B: r -= n*ln2hi, high precision part
     vfnmadd231ps ymm0, ymm3, [rel s_ln2lo_v]  ; chain A: r -= n*ln2lo, low precision part
     vfnmadd231ps ymm7, ymm8, [rel s_ln2lo_v]  ; chain B: r -= n*ln2lo, low precision part
     vmovaps ymm4, ymm1                        ; chain A: polynomial accumulator = C6
     vmovaps ymm10, ymm1                       ; chain B: polynomial accumulator = C6
     vpaddd ymm2, ymm2, [rel swk_bias_127]     ; chain A: bias n into the float exponent field
     vpaddd ymm9, ymm9, [rel swk_bias_127]     ; chain B: bias n into the float exponent field
     vfmadd213ps ymm4, ymm0, [rel s_c5_v]      ; chain A: acc = acc*r + c5
     vfmadd213ps ymm10, ymm7, [rel s_c5_v]     ; chain B: acc = acc*r + c5
     vpslld ymm2, ymm2, 23                     ; chain A: 2^n bit pattern in the mantissa field
     vpslld ymm9, ymm9, 23                     ; chain B: 2^n bit pattern in the mantissa field
     vfmadd213ps ymm4, ymm0, [rel s_c4_v]      ; chain A: acc = acc*r + c4
     vfmadd213ps ymm10, ymm7, [rel s_c4_v]     ; chain B: acc = acc*r + c4
     vfmadd213ps ymm4, ymm0, [rel s_c3_v]      ; chain A: acc = acc*r + c3
     vfmadd213ps ymm10, ymm7, [rel s_c3_v]     ; chain B: acc = acc*r + c3
     vfmadd213ps ymm4, ymm0, [rel s_half_v]    ; chain A: acc = acc*r + 0.5
     vfmadd213ps ymm10, ymm7, [rel s_half_v]   ; chain B: acc = acc*r + 0.5
     vfmadd213ps ymm4, ymm0, ymm5              ; chain A: acc = acc*r + 1.0
     vfmadd213ps ymm10, ymm7, ymm5             ; chain B: acc = acc*r + 1.0
     vfmadd213ps ymm4, ymm0, ymm5              ; chain A: acc = acc*r + 1.0, degree-7 tail
     vfmadd213ps ymm10, ymm7, ymm5             ; chain B: acc = acc*r + 1.0, degree-7 tail
%if %1
     vmovups ymm0, [rcx + r10*S_FLOAT4_SCALE + S_VEC_C] ; next group's block 2 loads during A/B exp
     vmovups ymm7, [rcx + r10*S_FLOAT4_SCALE + S_VEC_D] ; next group's block 3
     vmulps ymm3, ymm4, ymm2                   ; chain A: exp(x) = poly * 2^n, into the dead scratch pair
     vmulps ymm8, ymm10, ymm9                  ; chain B: exp(x) = poly * 2^n
%else
     vmulps ymm0, ymm4, ymm2                   ; chain A: exp(x) = poly * 2^n
     vmulps ymm7, ymm10, ymm9                  ; chain B: exp(x) = poly * 2^n
%endif
%endmacro

%macro STOCH_BOUND 1
     mov eax, r12d                             ; vocab size
     imul eax, dword [rel stoch_split_percentage + %1] ; stage fraction
     shr eax, 8                                ; p/256 fixed point
     and eax, -32                              ; align to four vectors
     mov edx, 32                               ; lower bound for each thread
     cmp eax, edx                              ; caller has at least 32?
     cmovb eax, edx                            ; clamp low
     lea edx, [r12d-STOCH_MIN_RANGE]           ; reserve at least 32 for helper
     and edx, -32                              ; align upper limit
     cmp eax, edx                              ; caller exceeds upper bound?
     cmova eax, edx                            ; clamp high
%endmacro

; RCX raw logits, EDX vocab count, R8D temperature bits, R9 scratch pointer.
     SF_START s_fork_stochastic_pipeline, rbx, rsi, rdi, r12, r13, r14 ; save six nonvolatile GPRs
     SF_DWORD st_main_result                   ; main phase max/sum, read after helper completes
     SF_DWORD st_scalar_bits                   ; merged maximum, then merged sum
     SF_COMMIT                                 ; Win64 shadow space for range calls
     mov rsi, rcx                              ; keep input logits across all calls
     mov rdi, r9                               ; keep scratch base across all calls
     mov r12d, edx                             ; total vocab length
     mov r13d, r8d                             ; temperature bits, unchanged

; ---- Phase 0: scale/copy logits and find two local maxima ----
     STOCH_BOUND 0                             ; aligned, nonempty caller/helper ranges
     mov r14d, eax                             ; save boundary for the main range
     lea rax, [rsi + r14*S_FLOAT4_SCALE]       ; helper source starts at boundary
     lea rbx, [rdi + r14*S_FLOAT4_SCALE]       ; helper output starts at boundary
     mov ecx, r12d                             ; full size
     sub ecx, r14d                             ; helper count
     mov edx, r13d                             ; temperature bits
     FORK 0                                    ; publish helper range
     jz .st_temp_parent                        ; ZF=1: original thread
; W0: convert handoff registers to ordinary Win64 function arguments.
     mov r8d, ecx                              ; arg3 count
     mov r9d, edx                              ; arg4 temperature bits
     mov rcx, rax                              ; arg1 logits pointer
     mov rdx, rbx                              ; arg2 scratch pointer
     call s_stoch_temp_range                   ; returns max float bits in EAX
     CONTINUE_FORK 0                           ; publish phase result and await next FORK, no pool transition
.st_temp_parent:
     mov rcx, rsi                              ; arg1 original logits
     mov rdx, rdi                              ; arg2 scratch base
     mov r8d, r14d                             ; arg3 caller count
     mov r9d, r13d                             ; arg4 temperature bits
     call s_stoch_temp_range                   ; scale/copy caller partition
     mov st_main_result, eax                   ; save main max before checking helper
     cmp qword [rel fork_params + fork_param.targetaddr], 0 ; did W0 finish first?
     jne .st_temp_parent_first                 ; no, increase caller workload next time
     cmp dword [rel stoch_split_percentage], 1 ; can decrease p?
     jbe .st_temp_join                         ; already at lower bound
     dec dword [rel stoch_split_percentage]    ; worker won -> give main less next time
     jmp .st_temp_join                         ; join with the worker
.st_temp_parent_first:
     cmp dword [rel stoch_split_percentage], 255 ; can increase p?
     jae .st_temp_join                         ; already at upper bound
     inc dword [rel stoch_split_percentage]    ; main won -> give main more next time
.st_temp_join:
     FORK_WAIT 0                               ; phase-0 result is ready; W0 is parked for next continuation
     vmovd xmm0, dword [rel fork_params + fork_param.rax] ; helper maximum bits as an exact FP32 value
     vmaxss xmm0, xmm0, st_main_result         ; combine helper and main maxima, memory operand is source 2
     vmovd st_scalar_bits, xmm0                ; broadcast input to exp phases

; ---- Phase 1: exponentials and two independent partial sums ----
     STOCH_BOUND 4                             ; aligned, nonempty caller/helper ranges
     mov r14d, eax                             ; main range size
     lea rax, [rdi + r14*S_FLOAT4_SCALE]       ; helper scratch start
     mov ebx, r12d                             ; full length
     sub ebx, r14d                             ; helper length
     mov ecx, st_scalar_bits                   ; max bits, for both threads
     xor edx, edx                              ; spare register in fork handoff
     FORK 0                                    ; resume W0 immediately without pool re-entry
     jz .st_exp_parent                         ; parent computes prefix
     mov r8d, ecx                              ; arg3 maximum bits
     mov rcx, rax                              ; arg1 scratch start
     mov edx, ebx                              ; arg2 length
     call s_stoch_exp_range                    ; returns local sum bits in EAX
     CONTINUE_FORK 0                           ; keep helper frame and await normalization phase
.st_exp_parent:
     mov rcx, rdi                              ; arg1 scratch base
     mov edx, r14d                             ; arg2 main count
     mov r8d, st_scalar_bits                   ; arg3 maximum bits
     call s_stoch_exp_range                    ; main partial sum
     mov st_main_result, eax                   ; preserve sum bits
     cmp qword [rel fork_params + fork_param.targetaddr], 0 ; worker completed already?
     jne .st_exp_parent_first                  ; no
     cmp dword [rel stoch_split_percentage + STOCH_SPLIT_EXP_OFF], 1 ; lower clamp
     jbe .st_exp_join                          ; lower limit reached
     dec dword [rel stoch_split_percentage + STOCH_SPLIT_EXP_OFF] ; worker won -> main share shrinks
     jmp .st_exp_join                          ; join
.st_exp_parent_first:
     cmp dword [rel stoch_split_percentage + STOCH_SPLIT_EXP_OFF], 255 ; upper clamp
     jae .st_exp_join                          ; upper limit reached
     inc dword [rel stoch_split_percentage + STOCH_SPLIT_EXP_OFF] ; main won -> caller share grows
.st_exp_join:
     FORK_WAIT 0                               ; both sums ready; W0 remains in its fast helper spin
     vmovd xmm0, dword [rel fork_params + fork_param.rax] ; worker's partial sum bits
     vaddss xmm0, xmm0, st_main_result         ; combine two sums, memory operand is source 2
     vmovd st_scalar_bits, xmm0                ; normalization divisor

; ---- Phase 2: independent probability normalization ----
     STOCH_BOUND 8                             ; aligned, nonempty caller/helper ranges
     mov r14d, eax                             ; caller count
     lea rax, [rdi + r14*S_FLOAT4_SCALE]       ; helper scratch range start
     mov ebx, r12d                             ; vocabulary size
     sub ebx, r14d                             ; helper length
     mov ecx, st_scalar_bits                   ; shared scalar sum bits
     xor edx, edx                              ; fourth transferred register unused
     FORK 0                                    ; resume W0 for final phase, then pool rejoin
     jz .st_norm_parent                        ; parent computes prefix
     mov r8d, ecx                              ; arg3 sum bits
     mov rcx, rax                              ; arg1 scratch range
     mov edx, ebx                              ; arg2 length
     call s_stoch_norm_range                   ; normalize helper slice
     CONTINUE_DISPATCH 0                       ; W0 resumes ordinary dispatcher
.st_norm_parent:
     mov rcx, rdi                              ; arg1 scratch base
     mov edx, r14d                             ; arg2 caller count
     mov r8d, st_scalar_bits                   ; arg3 sum bits
     call s_stoch_norm_range                   ; normalize caller slice
     cmp qword [rel fork_params + fork_param.targetaddr], 0 ; helper already complete?
     jne .st_norm_parent_first                 ; no
     cmp dword [rel stoch_split_percentage + STOCH_SPLIT_NORM_OFF], 1 ; minimum split
     jbe .st_norm_join                         ; no decrease
     dec dword [rel stoch_split_percentage + STOCH_SPLIT_NORM_OFF] ; worker won -> main share shrinks
     jmp .st_norm_join                         ; join
.st_norm_parent_first:
     cmp dword [rel stoch_split_percentage + STOCH_SPLIT_NORM_OFF], 255 ; maximum split
     jae .st_norm_join                         ; no increase
     inc dword [rel stoch_split_percentage + STOCH_SPLIT_NORM_OFF] ; main won -> main share grows
.st_norm_join:
     FORK_WAIT 0                               ; all probabilities ready before RNG and top-p
     SF_END rbx, rsi, rdi, r12, r13, r14       ; restore frame and return

; RCX source, RDX dest, R8D count (multiple of 32 for caller), R9D temp bits.
; Returns EAX=maximum FP32 bits. Nonvolatile GPRs/XMM registers untouched.
     SF_START s_stoch_temp_range               ; create Win64 frame
     SF_COMMIT                                 ; call shadow for s_hmax256_ps
     vmovd xmm2, r9d                           ; scalar temperature bits
     vbroadcastss ymm2, xmm2                   ; 8 independent temperature divisors
     vbroadcastss ymm3, [rel scalar_neg_flt_max] ; baseline maximum initialization
     xor r10d, r10d                            ; element index
     mov r11d, r8d                             ; four-vector bound
     and r11d, -32                             ; floor(count/32)*32
     cmp r9d, 0x3f800000                       ; temperature == 1.0?
     je .st_temp_copy_quad                     ; use direct copy path
.st_temp_div_quad:
     cmp r10d, r11d                            ; 32-float block remains?
     jae .st_temp_div_vec8_setup               ; no
     vmovups ymm0, [rcx + r10*S_FLOAT4_SCALE]  ; vector 0
     vmovups ymm1, [rcx + r10*S_FLOAT4_SCALE + S_VEC_B] ; vector 1
     vmovups ymm4, [rcx + r10*S_FLOAT4_SCALE + S_VEC_C] ; vector 2
     vmovups ymm5, [rcx + r10*S_FLOAT4_SCALE + S_VEC_D] ; vector 3
     vdivps ymm0, ymm0, ymm2                   ; divide vector 0
     vdivps ymm1, ymm1, ymm2                   ; divide vector 1
     vdivps ymm4, ymm4, ymm2                   ; divide vector 2
     vdivps ymm5, ymm5, ymm2                   ; divide vector 3
     vmovups [rdx + r10*S_FLOAT4_SCALE], ymm0  ; output vector 0
     vmovups [rdx + r10*S_FLOAT4_SCALE + S_VEC_B], ymm1 ; output vector 1
     vmovups [rdx + r10*S_FLOAT4_SCALE + S_VEC_C], ymm4 ; output vector 2
     vmovups [rdx + r10*S_FLOAT4_SCALE + S_VEC_D], ymm5 ; output vector 3
     vmaxps ymm3, ymm3, ymm0                   ; update max from vector 0
     vmaxps ymm3, ymm3, ymm1                   ; update max from vector 1
     vmaxps ymm3, ymm3, ymm4                   ; update max from vector 2
     vmaxps ymm3, ymm3, ymm5                   ; update max from vector 3
     add r10d, 32                              ; advance 32 floats
     jmp .st_temp_div_quad                     ; continue
.st_temp_div_vec8_setup:
     mov r11d, r8d                             ; 8-vector bound
     and r11d, -8                              ; floor(count/8)*8
.st_temp_div_vec:
     cmp r10d, r11d                            ; 8-float residue?
     jae .st_temp_hmax                         ; no
     vmovups ymm0, [rcx + r10*S_FLOAT4_SCALE]  ; logits
     vdivps ymm0, ymm0, ymm2                   ; divide correctly
     vmovups [rdx + r10*S_FLOAT4_SCALE], ymm0  ; write scratch
     vmaxps ymm3, ymm3, ymm0                   ; local max
     add r10d, 8                               ; next vector
     jmp .st_temp_div_vec                      ; continue
.st_temp_copy_quad:
     cmp r10d, r11d                            ; 32 floats remain?
     jae .st_temp_copy_vec8_setup              ; no
     vmovups ymm0, [rcx + r10*S_FLOAT4_SCALE]  ; vector 0
     vmovups ymm1, [rcx + r10*S_FLOAT4_SCALE + S_VEC_B] ; vector 1
     vmovups ymm4, [rcx + r10*S_FLOAT4_SCALE + S_VEC_C] ; vector 2
     vmovups ymm5, [rcx + r10*S_FLOAT4_SCALE + S_VEC_D] ; vector 3
     vmovups [rdx + r10*S_FLOAT4_SCALE], ymm0  ; copy 0
     vmovups [rdx + r10*S_FLOAT4_SCALE + S_VEC_B], ymm1 ; copy 1
     vmovups [rdx + r10*S_FLOAT4_SCALE + S_VEC_C], ymm4 ; copy 2
     vmovups [rdx + r10*S_FLOAT4_SCALE + S_VEC_D], ymm5 ; copy 3
     vmaxps ymm0, ymm0, ymm1                   ; max of vectors 0/1, independent of the 2/3 max below
     vmaxps ymm4, ymm4, ymm5                   ; max of vectors 2/3
     vmaxps ymm0, ymm0, ymm4                   ; max of all four vectors, balanced tree beats the serial chain
     vmaxps ymm3, ymm3, ymm0                   ; update the running maximum once
     add r10d, 32                              ; advance 32
     jmp .st_temp_copy_quad                    ; continue
.st_temp_copy_vec8_setup:
     mov r11d, r8d                             ; vector boundary
     and r11d, -8                              ; floor(count/8)*8
.st_temp_copy_vec:
     cmp r10d, r11d                            ; eight floats remain?
     jae .st_temp_hmax                         ; no
     vmovups ymm0, [rcx + r10*S_FLOAT4_SCALE]  ; raw logits
     vmovups [rdx + r10*S_FLOAT4_SCALE], ymm0  ; identity temperature
     vmaxps ymm3, ymm3, ymm0                   ; update local max
     add r10d, 8                               ; next vector
     jmp .st_temp_copy_vec                     ; continue
.st_temp_hmax:
     vmovaps ymm0, ymm3                        ; reduce max vector
     call s_hmax256_ps                         ; XMM0=local maximum
     cmp r10d, r8d                             ; scalar residue?
     jae .st_temp_result                       ; no residue
     cmp r9d, 0x3f800000                       ; identity temperature?
     je .st_temp_copy_tail                     ; no scalar divide
.st_temp_div_tail:
     vmovss xmm1, [rcx + r10*S_FLOAT4_SCALE]   ; raw scalar
     vdivss xmm1, xmm1, xmm2                   ; correctly rounded scalar divide
     vmovss [rdx + r10*S_FLOAT4_SCALE], xmm1   ; store scratch
     vmaxss xmm0, xmm1, xmm0                   ; update local max
     inc r10d                                  ; next scalar
     cmp r10d, r8d                             ; more?
     jb .st_temp_div_tail                      ; repeat
     jmp .st_temp_result                       ; done
.st_temp_copy_tail:
     vmovss xmm1, [rcx + r10*S_FLOAT4_SCALE]   ; raw scalar
     vmovss [rdx + r10*S_FLOAT4_SCALE], xmm1   ; store identity value
     vmaxss xmm0, xmm1, xmm0                   ; update local max
     inc r10d                                  ; next scalar
     cmp r10d, r8d                             ; more?
     jb .st_temp_copy_tail                     ; repeat
.st_temp_result:
     vmovd eax, xmm0                           ; return max bits via RAX
     SF_END                                    ; restore and return

; RCX scratch start, EDX element count, R8D max bits.
; Returns EAX=partial sum bits. This path uses the baseline interleaved exp
; polynomial and four YMM sums, with a separate sum tree per range.
     SF_START s_stoch_exp_range                ; create SIMD-preserving Win64 frame
     SF_XMM st_xmm6                            ; Win64 low 128 bits preserved
     SF_XMM st_xmm7                            ; polynomial pair temporary
     SF_XMM st_xmm8                            ; polynomial pair temporary
     SF_XMM st_xmm9                            ; polynomial pair temporary
     SF_XMM st_xmm10                           ; polynomial pair temporary
     SF_XMM st_xmm12                           ; accumulator 0
     SF_XMM st_xmm13                           ; accumulator 1
     SF_XMM st_xmm14                           ; accumulator 2
     SF_XMM st_xmm15                           ; accumulator 3
     SF_COMMIT                                 ; save XMM state before overwriting
     vmovdqu st_xmm6, xmm6                     ; preserve callee-saved XMM6
     vmovdqu st_xmm7, xmm7                     ; preserve callee-saved XMM7
     vmovdqu st_xmm8, xmm8                     ; preserve callee-saved XMM8
     vmovdqu st_xmm9, xmm9                     ; preserve callee-saved XMM9
     vmovdqu st_xmm10, xmm10                   ; preserve callee-saved XMM10
     vmovdqu st_xmm12, xmm12                   ; preserve callee-saved XMM12
     vmovdqu st_xmm13, xmm13                   ; preserve callee-saved XMM13
     vmovdqu st_xmm14, xmm14                   ; preserve callee-saved XMM14
     vmovdqu st_xmm15, xmm15                   ; preserve callee-saved XMM15
     vmovd xmm6, r8d                           ; max float bits
     vbroadcastss ymm6, xmm6                   ; max broadcast to all lanes
     vmovups ymm1, [rel s_c6_v]                ; polynomial first coefficient
     vmovups ymm5, [rel s_one_v]               ; polynomial unity coefficient
     vxorps ymm12, ymm12, ymm12                ; local accumulator 0
     vxorps ymm13, ymm13, ymm13                ; local accumulator 1
     vxorps ymm14, ymm14, ymm14                ; local accumulator 2
     vxorps ymm15, ymm15, ymm15                ; local accumulator 3
     xor r10d, r10d                            ; current element
     mov r9d, edx                              ; total vectorized 32-float prefix
     and r9d, -32                              ; floor(count / 32) * 32
.st_exp_quad:
     cmp r10d, r9d                             ; four-vector group available?
     jae .st_exp_rem_setup                     ; do 8-lane residue
     vmovups ymm0, [rcx + r10*S_FLOAT4_SCALE]  ; block 0
     vmovups ymm7, [rcx + r10*S_FLOAT4_SCALE + S_VEC_B] ; block 1
     vsubps ymm0, ymm0, ymm6                   ; subtract common maximum
     vsubps ymm7, ymm7, ymm6                   ; subtract common maximum
     S_EXP256_SOFT_PAIR 1                      ; A/B exp; C/D loads overlap the Horner tail
     vmovups [rcx + r10*S_FLOAT4_SCALE], ymm3  ; block 0 output
     vmovups [rcx + r10*S_FLOAT4_SCALE + S_VEC_B], ymm8 ; block 1 output
     vaddps ymm12, ymm12, ymm3                 ; local sum acc0
     vaddps ymm13, ymm13, ymm8                 ; local sum acc1
     vsubps ymm0, ymm0, ymm6                   ; subtract max, block 2
     vsubps ymm7, ymm7, ymm6                   ; subtract max, block 3
     S_EXP256_SOFT_PAIR 0                      ; identical polynomial for block 2/3
     vmovups [rcx + r10*S_FLOAT4_SCALE + S_VEC_C], ymm0 ; block 2 output
     vmovups [rcx + r10*S_FLOAT4_SCALE + S_VEC_D], ymm7 ; block 3 output
     vaddps ymm14, ymm14, ymm0                 ; local sum acc2
     vaddps ymm15, ymm15, ymm7                 ; local sum acc3
     add r10d, 32                              ; next four-vector group
     jmp .st_exp_quad                          ; next 32 floats
.st_exp_rem_setup:
     mov r9d, edx                              ; total 8-float vector range
     and r9d, -8                               ; floor(count / 8) * 8
.st_exp_rem:
     cmp r10d, r9d                             ; 8-float residue?
     jae .st_exp_reduce                        ; none
     vmovups ymm0, [rcx + r10*S_FLOAT4_SCALE]  ; next 8 scaled logits
     vsubps ymm0, ymm0, ymm6                   ; x-max
     call s_exp256_soft                        ; baseline 8-wide exp calculation
     vmovups [rcx + r10*S_FLOAT4_SCALE], ymm0  ; store exp
     vaddps ymm12, ymm12, ymm0                 ; remainder contributes to accumulator 0
     add r10d, 8                               ; next block
     jmp .st_exp_rem                           ; continue
.st_exp_reduce:
     vaddps ymm12, ymm12, ymm13                ; original four-way tree level 1
     vaddps ymm14, ymm14, ymm15                ; original tree level 1
     vaddps ymm12, ymm12, ymm14                ; original tree level 2
     vmovaps ymm0, ymm12                       ; total vector sum
     call s_hsum256_ps                         ; return scalar sum in XMM0
     vmovaps xmm4, xmm0                        ; carry sum through scalar remainder
.st_exp_scalar:
     cmp r10d, edx                             ; scalar residue?
     jae .st_exp_done                          ; no
     vmovss xmm1, [rcx + r10*S_FLOAT4_SCALE]   ; scaled logit
     vsubss xmm1, xmm1, xmm6                   ; x - max
     vmaxss xmm1, xmm1, [rel s_exp_neg_clamp]  ; clamp lower like original
     vmulss xmm2, xmm1, [rel scalar_log2e]     ; base-2 range reduction
     vroundss xmm2, xmm2, xmm2, 0              ; rounded exponent
     vcvttss2si eax, xmm2                      ; integer exponent
     vfnmadd231ss xmm1, xmm2, [rel s_exp_ln2_hi] ; f = x - n * ln2hi
     vfnmadd231ss xmm1, xmm2, [rel s_exp_ln2_lo] ; f -= n * ln2lo
     vmovss xmm3, [rel s_exp_c6]               ; polynomial c6
     add eax, 127                              ; biased exponent
     vfmadd213ss xmm3, xmm1, [rel s_exp_c5]    ; polynomial c5
     shl eax, 23                               ; float exponent bits
     vfmadd213ss xmm3, xmm1, [rel s_exp_c4]    ; polynomial c4
     vmovd xmm2, eax                           ; exponent as FP32 bits
     vfmadd213ss xmm3, xmm1, [rel s_exp_c3]    ; polynomial c3
     vfmadd213ss xmm3, xmm1, [rel rpe_half]    ; polynomial c2
     vfmadd213ss xmm3, xmm1, [rel rpe_one]     ; polynomial c1
     vfmadd213ss xmm3, xmm1, [rel rpe_one]     ; polynomial c0
     vmulss xmm1, xmm3, xmm2                   ; final exp approximation
     vmovss [rcx + r10*S_FLOAT4_SCALE], xmm1   ; write exp to scratch
     vaddss xmm4, xmm4, xmm1                   ; include scalar into sum
     inc r10d                                  ; next element
     jmp .st_exp_scalar                        ; continue
.st_exp_done:
     vmovd eax, xmm4                           ; return local sum as 32-bit bits
     vmovdqu xmm6, st_xmm6                     ; restore low XMM6
     vmovdqu xmm7, st_xmm7                     ; restore low XMM7
     vmovdqu xmm8, st_xmm8                     ; restore low XMM8
     vmovdqu xmm9, st_xmm9                     ; restore low XMM9
     vmovdqu xmm10, st_xmm10                   ; restore low XMM10
     vmovdqu xmm12, st_xmm12                   ; restore low XMM12
     vmovdqu xmm13, st_xmm13                   ; restore low XMM13
     vmovdqu xmm14, st_xmm14                   ; restore low XMM14
     vmovdqu xmm15, st_xmm15                   ; restore low XMM15
     SF_END                                    ; restore frame and return
%undef st_xmm6
%undef st_xmm7
%undef st_xmm8
%undef st_xmm9
%undef st_xmm10
%undef st_xmm12
%undef st_xmm13
%undef st_xmm14
%undef st_xmm15

; RCX=contiguous scratch range, EDX=count, R8D=sum FP32 bits.
; Correctly rounded AVX2 divisions, same element-wise values as the baseline
; for a given global sum.
     SF_START s_stoch_norm_range               ; create Win64 frame
     SF_COMMIT                                 ; standard Win64 stack frame
     vmovd xmm1, r8d                           ; sum FP32 bits
     vbroadcastss ymm1, xmm1                   ; scalar divisor for all lanes
     xor r10d, r10d                            ; index
     mov r9d, edx                              ; 32-float vector bound
     and r9d, -32                              ; floor(count/32)*32
.st_norm_quad:
     cmp r10d, r9d                             ; 32 available?
     jae .st_norm_tailvec_setup                ; less than 32
     vmovups ymm0, [rcx + r10*S_FLOAT4_SCALE]  ; vec0
     vmovups ymm2, [rcx + r10*S_FLOAT4_SCALE + S_VEC_B] ; vec1
     vmovups ymm3, [rcx + r10*S_FLOAT4_SCALE + S_VEC_C] ; vec2
     vmovups ymm4, [rcx + r10*S_FLOAT4_SCALE + S_VEC_D] ; vec3
     vdivps ymm0, ymm0, ymm1                   ; same rounded division as original
     vdivps ymm2, ymm2, ymm1                   ; same rounded division as original
     vdivps ymm3, ymm3, ymm1                   ; same rounded division as original
     vdivps ymm4, ymm4, ymm1                   ; same rounded division as original
     vmovups [rcx + r10*S_FLOAT4_SCALE], ymm0  ; store vec0
     vmovups [rcx + r10*S_FLOAT4_SCALE + S_VEC_B], ymm2 ; store vec1
     vmovups [rcx + r10*S_FLOAT4_SCALE + S_VEC_C], ymm3 ; store vec2
     vmovups [rcx + r10*S_FLOAT4_SCALE + S_VEC_D], ymm4 ; store vec3
     add r10d, 32                              ; advance 32
     jmp .st_norm_quad                         ; continue
.st_norm_tailvec_setup:
     mov r9d, edx                              ; vector bound
     and r9d, -8                               ; floor(count/8)*8
.st_norm_vec:
     cmp r10d, r9d                             ; >= 8 residue?
     jae .st_norm_scalar                       ; none
     vmovups ymm0, [rcx + r10*S_FLOAT4_SCALE]  ; next eight
     vdivps ymm0, ymm0, ymm1                   ; correct divide
     vmovups [rcx + r10*S_FLOAT4_SCALE], ymm0  ; store eight
     add r10d, 8                               ; advance
     jmp .st_norm_vec                          ; continue
.st_norm_scalar:
     cmp r10d, edx                             ; scalar tail?
     jae .st_norm_done                         ; no
     vmovss xmm0, [rcx + r10*S_FLOAT4_SCALE]   ; scalar value
     vdivss xmm0, xmm0, xmm1                   ; scalar divide
     vmovss [rcx + r10*S_FLOAT4_SCALE], xmm0   ; store
     inc r10d                                  ; advance
     jmp .st_norm_scalar                       ; next
.st_norm_done:
     SF_END                                    ; return
%undef st_main_result
%undef st_main_result_raw
%undef st_scalar_bits
%undef st_scalar_bits_raw

%unmacro STOCH_BOUND 1                         ; local split helper
%undef STOCH_SPLIT_EXP_OFF
%undef STOCH_SPLIT_NORM_OFF
%undef STOCH_MIN_RANGE
