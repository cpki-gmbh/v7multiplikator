; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define SF_TOKEN_VEC_APPEND -1                 ; token vector offset for last element, n-1
%define S_MAX_VOCAB 262144                     ; scratch table capacity: 256k vocab entries, headroom for models beyond Qwen3's 151936
     section .bss align=64
; --- Permanent normalized-probability working table, 256k vocab entries ---
; The sampler is NON-DESTRUCTIVE since this table landed: temperature scaling,
; exp and normalization all read the caller's RAW logits, rsi and write the
; transformed values HERE, so the caller's array keeps the model's raw values
; forever. That makes v7_get_token_probs correct at any call time and lets a
; SectorLogo Reject re-sample the untouched raw logits directly, no
; per-token copies anywhere. ONE table suffices even for nested v7_generate:
; sampling is strictly sequential and the table carries no cross-call state
; every sample fully rewrites it from the raw logits before reading .
     alignb 64
s_probs_scratch:
     resb S_MAX_VOCAB * 4                      ; one float per vocab entry
     section .rdata align=32
; --- Exp constants, match the C reference exactly, tools/v_sampler_validate.cpp ---
     align 4
s_exp_neg_clamp:
     dd -87.3365                               ; -126*ln,2 : keeps n >= -126 so biased exp >= 1, normal range, prevents subnormal flush to 0
s_exp_ln2_hi:
     dd 6.93145751953125e-1                    ; high bits of ln,2 for range reduction
s_exp_ln2_lo:
     dd 1.42860677302611e-6                    ; low bits of ln,2 for precision
s_exp_c6:
     dd 0x3AB5C2F0                             ; degree-6 fitted c6
s_exp_c5:
     dd 0x3C0917E1                             ; c5
s_exp_c4:
     dd 0x3D2AAB3A                             ; c4
s_exp_c3:
     dd 0x3E2AAA52                             ; c3
; --- RNG constants ---
     align 8
s_weyl_inc:
     dq 0x9e3779b97f4a7c15                     ; Weyl increment constant
s_aes_block_hi:
     dq 0xbf58476d1ce4e5b9                     ; high qword of AES plaintext block, SplitMix64 constant; state goes low
     align 16                                  ; must stay 16: legacy aesenc memory operand faults unaligned
s_aes_keys_combined:
     dq 0x1bf95cfc92db11eb                     ; LOW qword of AES key, validator e0
     dq 0x94d049bb133111eb                     ; HIGH qword of AES key, validator e1
; --- Float constants, unique ---
     align 4
s_zero_f:
     dd 0.0                                    ; float zero constant
s_neg_inf_f:
     dd 0xFF800000                             ; -inf, argmax top-2 dump: floor for the second-best scan
; --- argmax top-2 dump strings, debug tooling, gated by enable_token_dump ---
sdbg_pre:
     db "[samp] tok=", 0                       ; dump prefix: winning token index label
sdbg_bits:
     db " bits=", 0                            ; label before a decimal logit bit pattern
sdbg_second:
     db " 2nd=", 0                             ; label before the runner-up token index
sdbg_hash:
     db " h=", 0                               ; label before the hidden-state CRC, forward_q8 step 11
sdbg_l0:
     db " l0=", 0                              ; label before the layer-0 output CRC, forward_q8 layer loop
sdbg_nl:
     db " ", 0                                 ; trailing space, WRITE_STR_NL appends the CRLF
; --- RNG mask constants, s_random_f32 mantissa-direct method ---
     align 16
s_mantissa_mask:
     dq 0x00000000007FFFFF, 0                  ; keep low 23 bits for mantissa, upper lanes unused
s_exp127_mask:
     dq 0x000000003F800000, 0                  ; set exponent=127 -> [1.0, 2.0, upper lanes unused
     align 32
     s_expclamp_v: times 8 dd 88.376           ; exp overflow clamp, ln,FLT_MAX = 88.7228, minus poly headroom
     s_negclamp_v: times 8 dd -87.3365         ; -126*ln,2 lower clamp, softmax input x-max <= 0: upper clamp provably dead
     s_log2e_v: times 8 dd 1.4426950408889634  ; log2,e
     s_ln2hi_v: times 8 dd 6.93145751953125e-1 ; high bits of ln,2
     s_ln2lo_v: times 8 dd 1.42860677302611e-6 ; low bits of ln,2
     s_c6_v: times 8 dd 0x3AB5C2F0             ; fitted c6, == s_exp_c6
     s_c5_v: times 8 dd 0x3C0917E1             ; fitted c5, == s_exp_c5
     s_c4_v: times 8 dd 0x3D2AAB3A             ; fitted c4, == s_exp_c4
     s_c3_v: times 8 dd 0x3E2AAA52             ; fitted c3, == s_exp_c3
     s_half_v: times 8 dd 0.5                  ; c2
     s_one_v equ ymm_one                       ; c1, c0; identical 8x1.0f vector shared with math_data.inc
     section .text
s_random_f32:
     mov rax, [rcx]                            ; rax = current state
     add rax, [rel s_weyl_inc]                 ; state += Weyl increment
     mov [rcx], rax                            ; store back
     vmovq xmm0, rax                           ; xmm0 = state, zeroes upper 64 bits so the legacy aesenc below never hits a dirty-upper stall
     vpinsrq xmm0, xmm0, [rel s_aes_block_hi], 1 ; xmm0 = [block_hi | state], saves vmovq+punpcklqdq
     aesenc xmm0, [rel s_aes_keys_combined]    ; one AES round for whitening, legacy AES-NI encoding (no VAES needed), memory operand, avoids vmovdqu xmm1
     vpand xmm0, xmm0, [rel s_mantissa_mask]   ; keep the 23 mantissa bits of the whitened state
     vpor xmm0, xmm0, [rel s_exp127_mask]      ; force exponent 127 -> value in [1.0, 2.0
     vsubss xmm0, xmm0, [rel rpe_one]          ; subtract 1.0 -> value in [0.0, 1.0
     ret                                       ; return xmm0 = float in [0, 1
; Internal helper register contract: callers provide ymm1 = s_c6_v and ymm5 = s_one_v.
; The complete call-site set is sampler.inc and VLC_SOFTMAX_PASS in vit_kernel.inc.
s_exp256_soft:
     vmaxps ymm0, ymm0, [rel s_negclamp_v]     ; clamp lower only, upper dead: x-max <= 0
     vmulps ymm3, ymm0, [rel s_log2e_v]        ; x * log2,e
     vroundps ymm3, ymm3, 0                    ; n = round,x/ln2
     vcvttps2dq ymm2, ymm3                     ; n as int32
     vfnmadd231ps ymm0, ymm3, [rel s_ln2hi_v]  ; x - n*ln2_hi, single rounding
     vfnmadd231ps ymm0, ymm3, [rel s_ln2lo_v]  ; f, single rounding
     vmovaps ymm4, ymm1                        ; c6 seed from caller-held ymm1
     vpaddd ymm2, ymm2, [rel swk_bias_127]     ; bias n to the float exponent field
     vfmadd213ps ymm4, ymm0, [rel s_c5_v]      ; f*c6 + c5
     vpslld ymm2, ymm2, 23                     ; construct the exponent field
     vfmadd213ps ymm4, ymm0, [rel s_c4_v]      ; f*prev + c4
     vfmadd213ps ymm4, ymm0, [rel s_c3_v]      ; f*prev + c3
     vfmadd213ps ymm4, ymm0, [rel s_half_v]    ; f*prev + 0.5, c2
     vfmadd213ps ymm4, ymm0, ymm5              ; f*prev + 1.0, c1, ymm5 = +1.0f constant preloaded by callers
     vfmadd213ps ymm4, ymm0, ymm5              ; f*prev + 1.0, c0
     vmulps ymm0, ymm4, ymm2                   ; exp,x = poly * 2^n
     ret                                       ; return ymm0 = exp,x
s_hsum256_ps:
     vextractf128 xmm1, ymm0, 1                ; xmm1 = high 4 floats
     vaddps xmm0, xmm0, xmm1                   ; xmm0 = low + high, 4 partials
     vmovshdup xmm1, xmm0                      ; [e1, e1, e3, e3]
     vaddps xmm0, xmm0, xmm1                   ; [e0+e1, -, e2+e3, -]
     vmovhlps xmm1, xmm0, xmm0                 ; [e2+e3, ...]
     vaddss xmm0, xmm0, xmm1                   ; scalar sum
     ret                                       ; return scalar sum in xmm0
s_hmax256_ps:
     vextractf128 xmm1, ymm0, 1                ; xmm1 = high 4 floats [e4..e7]
     vmaxps xmm0, xmm0, xmm1                   ; xmm0 = [m0..m3], mj = max(ej, ej+4)
     vmovhlps xmm1, xmm0, xmm0                 ; xmm1 = [m2, m3, -, -]
     vmaxps xmm0, xmm0, xmm1                   ; xmm0 = [n0, n1, -, -], n0 = max(m0,m2), n1 = max(m1,m3)
     vmovshdup xmm1, xmm0                      ; xmm1 = [n1, n1, -, -]
     vmaxss xmm0, xmm0, xmm1                   ; final max = max(n0, n1)
     ret                                       ; return scalar max value in xmm0
%define TOPK_LIMIT 256                         ; keep top K probabilities for nucleus extraction
%define SIFT_RIGHT_STEP 1                      ; right child = left child + 1
%define SIFT_CHILD_SCALE 2                     ; left child index = 2 * parent + 1
%define PI_STRIDE 8                            ; ProbIndex struct size: {float prob; int index} = 8 bytes
%define PI_INDEX_OFF 4                         ; offset of index field within ProbIndex struct
%define S_FLOAT4_SCALE 4                       ; sizeof,float for SIB scale in logit addressing
%define STP_VEC_SPAN 8                         ; probabilities per vector group in the cutoff scan
%define STP_GROUP_LAST 7                       ; last index of a full vector group, span minus one
%define S_VEC_B 32                             ; byte offset of accumulator block B within a 32-float quad
%define S_VEC_C 64                             ; byte offset of accumulator block C within a 32-float quad
%define S_VEC_D 96                             ; byte offset of accumulator block D within a 32-float quad
%define SIFT_END_STEP 1                        ; extract phase: sift_down end = heap_end - 1
%define NUCLEUS_WALK_STEP 1                    ; nucleus walk starts at n0 - 1

%macro SIFT_DOWN 2
     mov r9d, edx                              ; r9 = current hole, starts at root, 64-bit
     mov rdx, [rcx + r9*PI_STRIDE]             ; rdx = carried original root record, prob+index together
     vmovd xmm2, edx                           ; xmm2 = carried root probability, exact original low 32 bits
%%sift_loop:
     lea r10, [r9*SIFT_CHILD_SCALE + SIFT_RIGHT_STEP] ; r10 = left child = 2*hole + 1
     cmp r10d, r8d                             ; child > end?
     ja %%sift_done                            ; done if child past end
     lea r11, [r10+SIFT_RIGHT_STEP]            ; r11 = right child index, 64-bit
     cmp r11d, r8d                             ; right child > end?
     ja %%sift_use_child                       ; only left child exists
     vmovss xmm1, [rcx + r11*PI_STRIDE]        ; heap[right].prob
     vucomiss xmm1, dword [rcx + r10*PI_STRIDE] ; compare right vs left
%1 %%sift_use_child                            ; child-select: max=jbe, left larger, min=jae, left smaller
     inc r10                                   ; child = right child, 64-bit
%%sift_use_child:
     vucomiss xmm2, dword [rcx + r10*PI_STRIDE] ; compare carried root prob vs child
%2 %%sift_done                                 ; root-test: max=jae, heap ok, min=jbe, heap ok
     mov rax, [rcx + r10*PI_STRIDE]            ; rax = child record, prob+index together
     mov [rcx + r9*PI_STRIDE], rax             ; promote child into the parent hole, ONE store
     mov r9, r10                               ; hole moves to the child's old slot
     jmp %%sift_loop                           ; continue sifting down
%%sift_done:
     mov [rcx + r9*PI_STRIDE], rdx             ; place the carried original root record ONCE at the final hole
     ret                                       ; return, heap property restored
%endmacro
s_sift_down:
     SIFT_DOWN jbe, jae                        ; max-heap: left larger -> jbe, root >= child -> jae
s_sift_down_min:
     SIFT_DOWN jae, jbe                        ; min-heap: left smaller -> jae, root <= child -> jbe
%define HEAP_PARENT_STEP 2                     ; first non-leaf parent =, n0 - 2 / 2
%define HEAP_LAST_OFFSET 1                     ; end index = n0 - 1

%macro BUILD_HEAP 1
     push r12                                  ; save non-volatile r12, n0
     push r13                                  ; save non-volatile r13, base
     push r14                                  ; save non-volatile r14, i
     mov r12d, edx                             ; r12d = n0, persistent across calls
     mov r13, rcx                              ; r13 = probindex base, preserved across calls
     cmp r12d, 2                               ; n0 < 2?
     jb %%bmh_done                             ; nothing to heapify
     lea r14d, [r12d - HEAP_PARENT_STEP]       ; r14d = n0 - 2, parent =, n-2 /2
     sar r14d, 1                               ; r14d =, n0-2 /2 = first non-leaf
     sub rsp, 32                               ; allocate shadow space ONCE for sift, Win64 ABI
     lea r8d, [r12d - HEAP_LAST_OFFSET]        ; r8d = n0 - 1 = end index, hoisted out of the loop; held across call %1 on every iteration, which is only correct because SIFT_DOWN never WRITES the volatile r8, it only reads it as the end bound
%%bmh_loop:
     mov edx, r14d                             ; start = i
     mov rcx, r13                              ; base = probindex, from non-volatile
     call %1                                   ; sift_down,base, i, end
     sub r14d, 1                               ; i--, sub+jns macro-fuses on Zen 4
     jns %%bmh_loop                            ; yes, continue heaping, signed compare
     add rsp, 32                               ; deallocate shadow space
%%bmh_done:
     pop r14                                   ; restore non-volatile r14
     pop r13                                   ; restore non-volatile r13
     pop r12                                   ; restore non-volatile r12
     ret                                       ; return, heap built
%endmacro
s_build_max_heap:
     BUILD_HEAP s_sift_down                    ; max-heap builder using max-heap sift-down
s_build_min_heap:
     BUILD_HEAP s_sift_down_min                ; min-heap builder using min-heap sift-down
%define STACK_ARG5_OFFSET 48                   ; rng_state*, 5th stack arg past SF_START frame, retaddr + shadow
%define STACK_ARG6_OFFSET 56                   ; probindex*, 6th stack arg past SF_START frame
%define STOCH_MIN_VOCAB 64                     ; forked paths need two ranges of STOCH_MIN_RANGE each

; ---------------------------------------------------------------------------
; Adaptive greedy fork scan helpers. Independent ranges, no shared scratch.
; s_greedy_max_range: RCX = float* range, EDX = positive count, EAX = max bits.
; s_greedy_find_range: RCX = float* range, EDX = count, R8D = max bits.
; EAX = lowest matching LOCAL index, 0xffffffff when none.
; The helper runs on W0's independent stack; all live arguments fit in the
; four FORK registers (RAX carries the max bits for the second pass).
; ---------------------------------------------------------------------------
s_greedy_max_range:
     vbroadcastss ymm0, [rcx]                  ; seed all accumulators with first value, same as original argmax
     vmovaps ymm1, ymm0                        ; second max accumulator
     vmovaps ymm2, ymm0                        ; third max accumulator
     vmovaps ymm3, ymm0                        ; fourth max accumulator
     xor r9d, r9d                              ; element index = 0
     mov r10d, edx                             ; 32-wide vector bound
     and r10d, -32                             ; floor count to multiple of 32
     jz .sgr_max_vec8                          ; fewer than 32 floats
.sgr_max_vec32_loop:
     vmaxps ymm0, ymm0, [rcx + r9*S_FLOAT4_SCALE] ; group A of 8 floats
     vmaxps ymm1, ymm1, [rcx + r9*S_FLOAT4_SCALE + S_VEC_B] ; group B of 8 floats
     vmaxps ymm2, ymm2, [rcx + r9*S_FLOAT4_SCALE + S_VEC_C] ; group C of 8 floats
     vmaxps ymm3, ymm3, [rcx + r9*S_FLOAT4_SCALE + S_VEC_D] ; group D of 8 floats
     add r9d, 32                               ; advance four vector lanes
     cmp r9d, r10d                             ; 32-wide work remaining?
     jb .sgr_max_vec32_loop                    ; yes
     vmaxps ymm0, ymm0, ymm1                   ; merge A and B
     vmaxps ymm2, ymm2, ymm3                   ; merge C and D
     vmaxps ymm0, ymm0, ymm2                   ; merge 4-way maximum
.sgr_max_vec8:
     mov r10d, edx                             ; scalar count for the 8-wide tail
     and r10d, -8                              ; floor count to 8
     cmp r9d, r10d                             ; any complete vector left?
     jae .sgr_max_reduce                       ; no
.sgr_max_vec8_loop:
     vmaxps ymm0, ymm0, [rcx + r9*S_FLOAT4_SCALE] ; incorporate next eight floats
     add r9d, 8                                ; advance input cursor
     cmp r9d, r10d                             ; vector work remains?
     jb .sgr_max_vec8_loop                     ; yes
.sgr_max_reduce:
     vextractf128 xmm1, ymm0, 1                ; upper four lanes
     vmaxps xmm0, xmm0, xmm1                   ; four pairwise maxima
     vmovhlps xmm1, xmm0, xmm0                 ; lanes 2,3
     vmaxps xmm0, xmm0, xmm1                   ; two pairwise maxima
     vmovshdup xmm1, xmm0                      ; replicate odd lane
     vmaxss xmm0, xmm0, xmm1                   ; final scalar max
     cmp r9d, edx                              ; any scalar tail?
     jae .sgr_max_ret                          ; no
.sgr_max_scalar:
     vmovss xmm1, [rcx + r9*S_FLOAT4_SCALE]    ; scalar input
     vmaxss xmm0, xmm1, xmm0                   ; same operand order as original scalar tail
     inc r9d                                   ; next scalar
     cmp r9d, edx                              ; more elements?
     jb .sgr_max_scalar                        ; yes
.sgr_max_ret:
     vmovd eax, xmm0                           ; exact max float bits in return register
     ret                                       ; leaf helper, no nonvolatile registers modified

s_greedy_find_range:
     vmovd xmm0, r8d                           ; scalar global maximum, exact bits
     vbroadcastss ymm0, xmm0                   ; broadcast max for ordered vector comparisons
     xor r9d, r9d                              ; local index = 0
     mov r10d, edx                             ; original element count
     and r10d, -8                              ; floor to complete 8-float vectors
     jz .sgr_find_scalar                       ; short range without a vector
.sgr_find_vec:
     vcmpps ymm2, ymm0, [rcx + r9*S_FLOAT4_SCALE], 0x02 ; ordered max <= logit, identical to existing pass-2 predicate
     vptest ymm2, ymm2                         ; any matching lane?
     jnz .sgr_find_hit                         ; yes: locate earliest match in vector
     add r9d, 8                                ; skip this vector
     cmp r9d, r10d                             ; more full vectors?
     jb .sgr_find_vec                          ; yes
.sgr_find_scalar:
     cmp r9d, edx                              ; reached end of range?
     jae .sgr_find_none                        ; no match in this partition
     vmovss xmm1, [rcx + r9*S_FLOAT4_SCALE]    ; examine one scalar
     vucomiss xmm1, xmm0                       ; compare current logit with global max
     jae .sgr_find_ret                         ; first ordered matching scalar
     inc r9d                                   ; advance local index
     jmp .sgr_find_scalar                      ; continue until match or end
.sgr_find_hit:
     vmovmskps eax, ymm2                       ; bit mask of matching lanes
     tzcnt eax, eax                            ; lowest matching lane (0..7)
     add r9d, eax                              ; local index of first vector match
.sgr_find_ret:
     mov eax, r9d                              ; return local index
     ret                                       ; leaf helper
.sgr_find_none:
     mov eax, -1                               ; no matching value in this range
     ret                                       ; caller merges two sentinel-capable indices

s_greedy_best_range:
; RCX = float* partition, EDX = count (>0).
; EAX = local maximum FP32 bits; EDX = earliest local index, or -1.
; Caller and W0 each invoke this once; no shared state.
     push rbx                                  ; preserve non-volatile, local max bits home
     push rsi                                  ; preserve non-volatile, partition base home
     push rdi                                  ; preserve non-volatile, partition count home
     sub rsp, 32                               ; Win64 call shadow; RSP remains 16-aligned
     mov rsi, rcx                              ; preserve partition base
     mov edi, edx                              ; preserve partition count
     call s_greedy_max_range                   ; scan the partition once for its maximum
     mov ebx, eax                              ; preserve local FP32 maximum bits
     mov rcx, rsi                              ; restore partition base argument
     mov edx, edi                              ; restore partition count argument
     mov r8d, ebx                              ; use the LOCAL max: same ordered >= predicate
     call s_greedy_find_range                  ; earliest local index of the local maximum
     mov edx, eax                              ; edx = first local index, returned in the second result register
     mov eax, ebx                              ; eax = local max FP32 bits, returned in the first result register
     add rsp, 32                               ; release call shadow
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     ret                                       ; both results transferred in the fork's four registers

     global v7s_sample                         ; exported: main sampling entry point
v7s_sample:
     SF_START v7s_sample, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles, rdi = the scratch-table base, live across the whole non-greedy path
     SF_QWORD s_logits                         ; logits float*
     SF_DWORD s_vocab_size                     ; vocab_size
     SF_DWORD s_temp_bits                      ; temperature float bits
     SF_DWORD s_topp_bits                      ; topp float bits
     SF_QWORD s_rng_state                      ; rng_state uint64_t*
     SF_QWORD s_probindex                      ; probindex ProbIndex*
     SF_DWORD s_max_val_bits                   ; max_val float bits
     SF_DWORD s_coin                           ; coin float bits
     SF_DWORD s_greedy_index                   ; first-index result from caller, before joining W0
     SF_DWORD s_entered_idle                   ; pool activity state at entry: 1 when this call must return the pool to idle
     SF_XMM s_xmm6                             ; declare xmm6 low half spill slot, only callee-saved xmm used in body
     SF_XMM s_xmm7                             ; declare xmm7 low half spill slot
     SF_XMM s_xmm8                             ; declare xmm8 low half spill slot
     SF_XMM s_xmm9                             ; declare xmm9 low half spill slot
     SF_XMM s_xmm10                            ; declare xmm10 low half spill slot
     SF_XMM s_xmm12                            ; declare xmm12 low half spill slot, softmax accumulators
     SF_XMM s_xmm13                            ; declare xmm13 low half spill slot
     SF_XMM s_xmm14                            ; declare xmm14 low half spill slot
     SF_XMM s_xmm15                            ; declare xmm15 low half spill slot
     SF_COMMIT                                 ; finalize stack frame layout
     mov s_logits, rcx                         ; save logits pointer
     vmovdqu s_xmm6, xmm6                      ; save callee-saved xmm6 low half
     vmovdqu s_xmm7, xmm7                      ; save callee-saved xmm7 low half
     vmovdqu s_xmm8, xmm8                      ; save callee-saved xmm8 low half
     vmovdqu s_xmm9, xmm9                      ; save callee-saved xmm9 low half
     vmovdqu s_xmm10, xmm10                    ; save callee-saved xmm10 low half
     vmovdqu s_xmm12, xmm12                    ; save callee-saved xmm12 low half, tail accumulator uses volatile xmm4, xmm7-11 stay untouched
     vmovdqu s_xmm13, xmm13                    ; save callee-saved xmm13 low half
     vmovdqu s_xmm14, xmm14                    ; save callee-saved xmm14 low half
     vmovdqu s_xmm15, xmm15                    ; save callee-saved xmm15 low half
     mov s_vocab_size, edx                     ; save vocab size
     vmovd s_temp_bits, xmm2                   ; save temperature float bits
     vmovd s_topp_bits, xmm3                   ; save topp float bits
     mov rax, [rbp + STACK_ARG5_OFFSET]        ; rbp+48 = rng_state*, stack arg5
     mov s_rng_state, rax                      ; save rng_state pointer
     mov rax, [rbp + STACK_ARG6_OFFSET]        ; rbp+56 = probindex*, stack arg6
     mov s_probindex, rax                      ; save probindex pointer
; --- Guards: after the arg saves, because initialize_once clobbers every
; volatile register. A direct v7s_sample call cannot assume a previous chat
; left workers spinning: bring the pool up lazily like every dispatched entry
; does and declare the active phase -- v7s_set_spin(1) also wakes registered
; sleepers, so the fork below can never publish to a sleeping W0 (worst case
; is the 10 ms WaitOnAddress timeout, never a missed fork). When the pool was
; idle at entry the epilogue returns it to the OS wait; a call made inside an
; inference-active phase leaves the activity state untouched for the caller.
; Vocabulary must fit the fork geometry.
; Vocabulary must fit the fork geometry: rejected BEFORE any pool wake, so an
; invalid direct call never starts or spins the workers.
     cmp dword s_vocab_size, STOCH_MIN_VOCAB   ; forked paths need two >= 32-element ranges
     jb .s_vocab_small_fatal                   ; smaller vocab cannot split: refuse instead of scanning out of bounds
     mov eax, dword [rel always_spin]          ; pool activity flag before this call wakes the workers
     xor eax, 1                                ; invert: 1 means the pool was idle at entry and must be returned to idle
     mov s_entered_idle, eax                   ; remember whether this call must return the pool to idle
     call initialize_once                      ; CAS-gated pool init, no-op when the pool is already up
     API_CALL v7s_set_spin, 1                  ; declare spin-active AND wake registered sleepers, centralized in the export
; --- Greedy: temperature <= 0?, handles +0.0, -0.0, NaN ---
     vmovd xmm0, s_temp_bits                   ; reload temp from stack
     vucomiss xmm0, [rel s_zero_f]             ; compare temp with 0.0
     ja .s_temp_nonzero                        ; temp > 0, ordered, above, do temperature scaling
; ==== ARGMAX ====, reads the caller's RAW logits, never written, greedy is non-destructive
     mov rsi, s_logits                         ; rsi = logits base pointer
     mov r12d, s_vocab_size                    ; r12d = vocab_size
; One independently tuned split: max and first-index are computed together per
; partition, W0 returns BOTH its local maximum and its earliest matching index.
     mov eax, r12d                             ; total vocabulary length
     imul eax, dword [rel split_percentage]    ; multiply by fixed-point main-thread fraction
     shr eax, 8                                ; divide by 256, floor to main range size
     jnz .s_greedy_split_nonzero               ; tiny vocab x low fraction can floor to zero
     mov eax, 1                                ; caller keeps one element; worker keeps the rest
.s_greedy_split_nonzero:                       ; caller count stays <= vocab-1: the fraction
     mov r13d, eax                             ; preserve range split across the helper call
     lea rcx, [rsi + r13*S_FLOAT4_SCALE]       ; worker input begins at split
     mov edx, r12d                             ; vocabulary length
     sub edx, r13d                             ; worker length = vocab - caller length
     FORK 0                                    ; publish four GPRs + worker continuation to W0
     jz .s_fork_best_main                      ; main receives ZF=1; worker receives ZF=0
     call s_greedy_best_range                  ; W0: EAX = worker local max bits, EDX = worker local index
     CONTINUE_DISPATCH 0                       ; only fork phase: publish results, return to the worker pool
.s_fork_best_main:
     mov rcx, rsi                              ; caller input starts at first logit
     mov edx, r13d                             ; caller range size
     call s_greedy_best_range                  ; caller: EAX = caller local max bits, EDX = caller local index
     mov s_max_val_bits, eax                   ; caller local maximum bits
     mov s_greedy_index, edx                   ; caller's earliest matching local index
; Snapshot completion BEFORE waiting, then nudge the caller fraction by one unit.
     cmp qword [rel fork_params + fork_param.targetaddr], 0 ; has W0 finished its half?
     jne .s_fork_best_main_faster              ; no: caller finished first, enlarge caller's range
     cmp dword [rel split_percentage], 1       ; worker finished first: can caller fraction decrease?
     jbe .s_fork_best_split_done               ; already at minimum
     dec dword [rel split_percentage]          ; worker won, reduce the caller's work next time
     jmp .s_fork_best_split_done               ; join W0
.s_fork_best_main_faster:
     cmp dword [rel split_percentage], 255     ; caller finished first: can main share grow?
     jae .s_fork_best_split_done               ; already at maximum
     inc dword [rel split_percentage]          ; send more vocabulary to caller next time
.s_fork_best_split_done:
     FORK_WAIT 0                               ; result ready, W0 has returned to the regular dispatcher
; W0's result slots were published before targetaddr became zero.
     mov edx, [rel fork_params + fork_param.rdx] ; worker local index or sentinel
     vmovd xmm0, s_max_val_bits                ; caller local max as an exact FP32 value
     vmovd xmm1, dword [rel fork_params + fork_param.rax] ; worker local max as an exact FP32 value
     vmaxss xmm2, xmm0, xmm1                   ; merge both maxima without numerical summation, kept live for the dump
; The low partition wins every numeric tie, as in the original earliest-index scan.
     vucomiss xmm0, xmm2                       ; caller max against the global max
     jp .s_greedy_try_worker                   ; unordered: caller cannot win
     je .s_greedy_take_main                    ; caller holds the global max: its match is strictly earlier
.s_greedy_try_worker:
     vucomiss xmm1, xmm2                       ; worker max against the global max
     jp .s_argmax_drift                        ; unordered AND caller lost: no partition owns the max
     jne .s_argmax_drift                       ; worker does not hold it either: same FATAL as the sequential scan
     mov ebx, edx                              ; ebx = worker's earliest LOCAL index
     cmp ebx, -1                               ; worker found a matching element?
     je .s_argmax_drift                        ; local max exists but no ordered match: drift, same FATAL
     add ebx, r13d                             ; convert W0 local index to global vocabulary index
     jmp .s_greedy_best_ready                  ; worker's match wins
.s_greedy_take_main:
     mov ebx, s_greedy_index                   ; caller's earliest matching index
     cmp ebx, -1                               ; caller found a matching element?
     je .s_greedy_try_worker                   ; no: fall through to the worker's match
.s_greedy_best_ready:
     jmp .s_argmax_found                       ; follow the original result/dump/cleanup path
.s_argmax_drift:
     FATAL "sampler: argmax pass-2 found no match for pass-1 max (NaN/logits drift?)"
.s_argmax_found:
     mov eax, ebx                              ; eax = sampled token index
     cmp dword [rel enable_token_dump], 0      ; logits top-2 dump requested, debug flag ?
     je .s_cleanup                             ; no: shared epilogue; requested falls into the dump
.s_argmax_dump:
     vbroadcastss ymm0, xmm2                   ; global maximum, still live from the merge, for the optional top-2 dump
     vbroadcastss ymm3, [rel s_neg_inf_f]      ; ymm3 = second_max = -inf, all lanes
     xor r8d, r8d                              ; r8d = i = 0
     mov ecx, r12d                             ; ecx = vocab_size
     and ecx, -8                               ; ecx = vector trip count
     jz .s_dump_tail1                          ; < 8 elements, scalar only
.s_dump_vec:
     vmovups ymm1, [rsi + r8*S_FLOAT4_SCALE]   ; load 8 logits, dword lanes
     vcmpps ymm2, ymm1, ymm0, 0x11             ; mask: logit < max_val, LT_OQ = 0x11, predicate in imm8[4:0]; quiet, ordered, NaN prob is not a candidate
     vblendvps ymm4, ymm3, ymm1, ymm2          ; candidates strictly below max, else -inf
     vmaxps ymm3, ymm3, ymm4                   ; second_max = max,second_max, candidates
     add r8, 8                                 ; i += 8
     cmp r8, rcx                               ; i < vector trip count?
     jb .s_dump_vec                            ; yes: continue
.s_dump_tail1:
     cmp r8d, r12d                             ; scalar tail elements left?
     jae .s_dump_reduce                        ; no: reduce
.s_dump_tail1_loop:
     vmovss xmm1, [rsi + r8*S_FLOAT4_SCALE]    ; load scalar logit
     vucomiss xmm1, xmm0                       ; strictly below max_val, ordered ?
     jae .s_dump_tail1_skip                    ; no, >= max or NaN : not a candidate
     vucomiss xmm1, xmm3                       ; above current second_max?
     jbe .s_dump_tail1_skip                    ; no: skip
     vmovss xmm3, xmm1                         ; second_max = candidate
.s_dump_tail1_skip:
     inc r8                                    ; i++
     cmp r8d, r12d                             ; i < vocab_size?
     jb .s_dump_tail1_loop                     ; yes: continue
.s_dump_reduce:
     vmovaps ymm0, ymm3                        ; second_max lanes into the hmax helper input, max_val dead: pass-2 complete
     call s_hmax256_ps                         ; xmm0 = scalar second_max
; park the dump values in dead SF slots, temperature/topp/vocab slots are never read again
     vmovss xmm1, [rsi + rbx*S_FLOAT4_SCALE]   ; reload the winning logit, rsi untouched by the leaf helper
     vmovd s_temp_bits, xmm1                   ; dead slot = winner logit bits
     vmovd s_topp_bits, xmm0                   ; dead slot = runner-up logit bits
     vmovd r9d, xmm0                           ; r9d = runner-up bits for the index scan
     xor r8d, r8d                              ; i = 0
.s_dump_idx_loop:
     cmp dword [rsi + r8*S_FLOAT4_SCALE], r9d  ; logits[i] == runner-up bits?
     je .s_dump_idx_done                       ; yes: runner-up index = i
     inc r8                                    ; i++
     cmp r8d, r12d                             ; i < vocab_size?
     jb .s_dump_idx_loop                       ; yes: continue
     xor r8d, r8d                              ; degenerate all-equal list: report index 0
.s_dump_idx_done:
     mov s_vocab_size, r8d                     ; dead slot = runner-up index
; emit: "[samp] tok=<idx> bits=<bits> 2nd=<idx> bits=<bits> \r\n"
     WRITE_STR sdbg_pre                        ; "[samp] tok="
     WRITE_UINT64 rbx                          ; winner index, clean 64-bit since pass-2 zeroed ebx
     WRITE_STR sdbg_bits                       ; " bits="
     mov ecx, s_temp_bits                      ; winner logit bits, dword slot load zero-extends
     WRITE_UINT64 rcx                          ; decimal bit pattern of the winning logit
     WRITE_STR sdbg_second                     ; " 2nd="
     mov ecx, s_vocab_size                     ; runner-up index, stashed dead slot
     WRITE_UINT64 rcx                          ; runner-up token index
     WRITE_STR sdbg_bits                       ; " bits="
     mov ecx, s_topp_bits                      ; runner-up logit bits, stashed dead slot
     WRITE_UINT64 rcx                          ; decimal bit pattern of the runner-up logit
     WRITE_STR sdbg_hash                       ; " h="
     mov ecx, [rel dbg_hidden_crc]             ; hidden-state CRC32C, published by forward_q8 step 11
     WRITE_UINT64 rcx                          ; decimal CRC, classifier-input fingerprint
     WRITE_STR sdbg_l0                         ; " l0="
     mov ecx, [rel dbg_layer0_crc]             ; layer-0 output CRC32C, published in the layer loop
     WRITE_UINT64 rcx                          ; decimal CRC, layer-0 fingerprint
     WRITE_STR_NL sdbg_nl                      ; " " + CRLF terminator
     mov eax, ebx                              ; eax = sampled token index, WRITE_* overwritten it
     jmp .s_cleanup                            ; shared epilogue
.s_temp_nonzero:
     mov rcx, s_logits                         ; non-destructive input logits
     mov edx, s_vocab_size                     ; vocabulary length
     mov r8d, s_temp_bits                      ; temperature bits
     lea r9, [rel s_probs_scratch]             ; normalized output
     call s_fork_stochastic_pipeline           ; W0 remains in helper mode between first two phases
     lea rdi, [rel s_probs_scratch]            ; output for nucleus sampling
     mov r12d, s_vocab_size                    ; count for nucleus sampling
.s_coin_flip:
     mov rcx, s_rng_state                      ; rcx = rng_state* for random_f32
     call s_random_f32                         ; xmm0 = coin in [0, 1
     vmovss xmm2, xmm0                         ; xmm2 = coin
     vmovd s_coin, xmm2                        ; save coin to stack slot, non-volatile
; ==== TOP-P CHECK ====
     vmovd xmm0, s_topp_bits                   ; xmm0 = topp from dword slot
     vxorps xmm1, xmm1, xmm1                   ; zero for topp <= 0 comparison
     vucomiss xmm0, xmm1                       ; topp <= 0?
     jbe .s_do_mult                            ; disabled, use multinomial, matches runq.c clamp
     vucomiss xmm0, [rel rpe_one]              ; topp >= 1.0?
     jb .s_do_topp                             ; 0 < topp < 1: do top-p, the DEFAULT sampling path, topp defaults to 0.9, so every non-greedy token extracts a nucleus here
; topp >= 1.0 or topp <= 0: fall through to .s_do_mult, common: topp disabled
; ==== MULTINOMIAL CDF WALK ====
.s_do_mult:
; rdi, scratch probs /r12 still live from .s_temp_nonzero, no intervening writer, pure-XMM leaf calls only
     vxorps xmm1, xmm1, xmm1                   ; cdf = 0, xmm2 still holds coin from .s_coin_flip
     xor ebx, ebx                              ; i = 0
.s_mult_loop:
     vaddss xmm1, xmm1, [rdi + rbx*S_FLOAT4_SCALE] ; CDF += prob[i], memory operand
     vucomiss xmm2, xmm1                       ; coin < cdf?
     jb .s_mult_found                          ; yes, sampled token
     inc ebx                                   ; i++
     cmp ebx, r12d                             ; i < vocab_size?
     jb .s_mult_loop                           ; yes, continue, unsigned compare
     lea eax, [r12d+SF_TOKEN_VEC_APPEND]       ; fallback: return n-1, last token
     jmp .s_cleanup                            ; shared epilogue
.s_mult_found:
     mov eax, ebx                              ; return sampled token index
     jmp .s_cleanup                            ; shared epilogue
; ==== TOP-P, NUCLEUS SAMPLING ====
; no vocab_size <= 1 special case: no supported model has a 1-token vocab, and the general path handles it correctly
.s_do_topp:
.s_topp_do_cutoff:
     mov r13, s_probindex                      ; min-heap record base
     xor ebx, ebx                              ; next vocabulary index
     mov r14d, TOPK_LIMIT                      ; cap of the initial heap
     cmp r12d, r14d                            ; can the vocabulary fill the cap?
     cmovb r14d, r12d                          ; clamp to the vocabulary size
     sub rsp, 32                               ; shadow space for heap calls
.s_topp_seed:
     vmovss xmm1, [rdi + rbx*S_FLOAT4_SCALE]   ; seed in original vocabulary order
     vmovss [r13 + rbx*PI_STRIDE], xmm1        ; copy the probability bits
     mov [r13 + rbx*PI_STRIDE + PI_INDEX_OFF], ebx ; pair it with the token index
     inc ebx                                   ; one record seeded
     cmp ebx, r14d                             ; all seed entries stored?
     jb .s_topp_seed                           ; continue seeding
     cmp r14d, TOPK_LIMIT                      ; build the min-heap only if the cap is reached
     jb .s_topp_cut_done                       ; the final max-heap receives the seed order
     mov rcx, r13                              ; min-heap record base
     mov edx, r14d                             ; all TOPK_LIMIT entries are live
     call s_build_min_heap                     ; establish the replacement threshold
.s_topp_next_group:
     lea eax, [rbx + STP_GROUP_LAST]           ; last index in a full vector
     cmp eax, r12d                             ; do eight probability values fit?
     jae .s_topp_tail                          ; handle the remainder without overreading
     vbroadcastss ymm3, [r13]                  ; root at the start of this group
     vcmpps ymm5, ymm3, [rdi + rbx*S_FLOAT4_SCALE], 0x11 ; ordered root < p
     vmovmskps r15d, ymm5                      ; one bit per eligible lane
     add ebx, 8                                ; exclusive group end, preserved by sift
     test r15d, r15d                           ; any lane can improve the heap?
     jz .s_topp_next_group                     ; no eligible lane
.s_topp_group_hit:
     tzcnt r15d, r15d                          ; first lane that can improve the heap
     lea r15d, [rbx + r15 - STP_VEC_SPAN]      ; absolute vocabulary index
.s_topp_scan:
     vmovss xmm1, [rdi + r15*S_FLOAT4_SCALE]   ; probability in original scan order
     vucomiss xmm1, dword [r13]                ; compare against the live root
     jbe .s_topp_scan_next                     ; preserve the old entry on a tie or NaN
     vmovss [r13], xmm1                        ; replace the root probability
     mov [r13 + PI_INDEX_OFF], r15d            ; replace its token index
     mov rcx, r13                              ; min-heap base
     xor edx, edx                              ; start sifting at root
     mov r8d, TOPK_LIMIT - 1                   ; inclusive heap end
     call s_sift_down_min                      ; restore the original heap ordering
.s_topp_scan_next:
     inc r15d                                  ; next lane, with a possibly changed root
     cmp r15d, ebx                             ; any lanes left in this group or tail?
     jb .s_topp_scan                           ; visit the remainder in original order
     cmp ebx, r12d                             ; is there more of the vocabulary?
     jb .s_topp_next_group                     ; start the next group or the tail
     jmp .s_topp_cut_done                      ; all candidates considered
.s_topp_tail:
     mov r15d, ebx                             ; first unexamined index
     mov ebx, r12d                             ; exact exclusive end of the tail
     cmp r15d, ebx                             ; is there a scalar remainder?
     jb .s_topp_scan                           ; reuse the live-root scan, no overread
.s_topp_cut_done:
; the fill phase stores unconditionally, so n0 >= 1 is structural, vocab_size >= 1
     mov rcx, r13                              ; base = probindex
     mov edx, r14d                             ; n0
     call s_build_max_heap                     ; build max-heap from probindex[0..n0-1], persistent shadow from the cutoff entry stands
; --- Extract nucleus from max-heap ---
     vxorps xmm6, xmm6, xmm6                   ; xmm6 = cumulative_prob = 0, callee-saved, survives calls
     vmovd xmm3, s_topp_bits                   ; xmm3 = topp, volatile; survives heap calls, s_sift_down/s_build_max_heap overwrite xmm0-xmm2 only: xmm2 is the sift's own root-prob carrier, never free across a heap call; xmm3 is outside that set
     mov r15d, r14d                            ; r15d = heap_end = n0, non-volatile
.s_topp_extract:
     test r15d, r15d                           ; heap_end <= 0?
     jbe .s_topp_extract_done                  ; done if no elements left
     vaddss xmm6, xmm6, [r13]                  ; top-p extraction sum += heap root, memory operand
     dec r15d                                  ; heap_end--, 32-bit op zero-extends to r15
; Swap heap[0] <-> heap[heap_end]
     mov rdx, [r13]                            ; rdx = heap[0] entry, 8 bytes: prob+index
     mov rax, [r13 + r15*PI_STRIDE]            ; rax = heap[heap_end] entry
     mov [r13 + r15*PI_STRIDE], rdx            ; heap[heap_end] = old heap[0]
     mov [r13], rax                            ; heap[0] = old heap[heap_end]
     vucomiss xmm6, xmm3                       ; cumulative_prob >= topp already? the nucleus is the SMALLEST prefix with cum >= topp, on equality the prefix already covers the requested mass and another token would widen it past topp
     jae .s_topp_extract_done                  ; nucleus complete: the retained heap prefix is never read again, skip the repair
     test r15d, r15d                           ; elements remain for further extraction?
     jbe .s_topp_extract_done                  ; heap empty, nucleus cannot grow, done
     mov rcx, r13                              ; base = probindex
     xor edx, edx                              ; start = 0
     lea r8d, [r15d - SIFT_END_STEP]           ; end = heap_end - 1 for sift_down
     call s_sift_down                          ; repair needed only because another extraction follows
     jmp .s_topp_extract                       ; next extraction
; cum >= topp, fall through: nucleus extracted
.s_topp_extract_done:
     vucomiss xmm6, xmm3                       ; does extracted mass cover topp?
     jae .s_topp_nucleus_ok                    ; yes: selected sufficient probability mass
     cmp r14d, r12d                            ; did we already extract the FULL vocabulary?, n0 == vocab_size: the wide fallback ran, or the caller's vocab fits the heap
     je .s_topp_nucleus_ok                     ; yes: every token is in the nucleus, r15d = 0, the walk covers the full vocab, the float32 cumulative can settle below topp even over a correct normalization, and rebuilding the same full heap would loop forever
; --- Wide-nucleus fallback: TOPK_LIMIT entries did not cover topp ---
; A uniform-ish distribution over a large vocab makes the true nucleus span more than 256
; tokens; the retained 256 largest carry less than topp, so this top-p request was silently
; capped at top-k. The probindex workspace is vocab-sized, inf_tokenizer reserves 8 bytes
; per entry, so the exact selection is a full-vocab rebuild with the same heap primitives --
; no new allocation, and the normalized probs are still live at rdi, the scratch table .
     xor ebx, ebx                              ; i = n0 = 0; one entry stored per vocabulary index
.s_topp_wide_fill:
     vmovss xmm1, [rdi + rbx*S_FLOAT4_SCALE]   ; normalized prob of vocabulary entry i
     vmovss [r13 + rbx*PI_STRIDE], xmm1        ; probindex[i].prob = prob
     mov [r13 + rbx*PI_STRIDE + PI_INDEX_OFF], ebx ; probindex[i].index = i
     inc ebx                                   ; one loop counter, both entry index and size
     cmp ebx, r12d                             ; i < vocab_size?
     jb .s_topp_wide_fill                      ; next entry
     mov r14d, ebx                             ; restore persistent n0 = vocab_size for heap extraction
     mov rcx, r13                              ; base = probindex
     mov edx, r14d                             ; n0 = vocab_size
     call s_build_max_heap                     ; max-heap over the full vocab, same primitive as the fast path
     vxorps xmm6, xmm6, xmm6                   ; cumulative_prob = 0, restart the extraction
     mov r15d, r14d                            ; heap_end = vocab_size
     jmp .s_topp_extract                       ; re-run the extraction: terminates on cum >= topp, full mass ~1 > any topp < 1
.s_topp_nucleus_ok:
; Nucleus = probindex[heap_end .. n0-1], prob-descending, highest at n0-1
     add rsp, 32                               ; release persistent shadow once, build + extract phases
; --- Sample from nucleus, walk from high prob to low prob ---
; n0 rides in r14d, non-volatile: BUILD_HEAP/SIFT_DOWN preserve it and
; heap_end in r15d, live since the extract phase, no frame slots needed
     vmovd xmm2, s_coin                        ; reload coin from stack, volatile
     vmulss xmm4, xmm2, xmm6                   ; xmm4 = r = coin * cumulative_prob
     vxorps xmm1, xmm1, xmm1                   ; xmm1 = cdf = 0
     lea ebx, [r14d - NUCLEUS_WALK_STEP]       ; ebx = j = n0-1, start at highest prob; ebx's upper half is zero, 32-bit lea/dec maintain it
     jmp .s_topp_sample_chk                    ; enter at the bottom test, loop inversion
.s_topp_sample:
     vaddss xmm1, xmm1, [r13 + rbx*PI_STRIDE]  ; nucleus CDF += prob, memory operand
     vucomiss xmm4, xmm1                       ; r < cdf?
     jb .s_topp_sample_found                   ; found sampled token
     dec ebx                                   ; j--
.s_topp_sample_chk:
     cmp ebx, r15d                             ; j < nucleus_start?, r15d = heap_end, live since the extract phase; signed compare preserved
     jge .s_topp_sample                        ; yes, continue walking nucleus
; fall through to .s_topp_sample_fallback
.s_topp_sample_fallback:
; Fallback: return lowest-probability nucleus element at heap_end
; CDF walk fell through due to FP rounding; matches C reference pi[last].index
     mov eax, [r13 + r15*PI_STRIDE + PI_INDEX_OFF] ; return probindex[heap_end].index, r15d = heap_end
     jmp .s_cleanup                            ; shared epilogue
.s_topp_sample_found:
     mov eax, [r13 + rbx*PI_STRIDE + PI_INDEX_OFF] ; return probindex[j].index, rbx = j
.s_cleanup:
     cmp dword s_entered_idle, 0               ; did this call enter with an idle pool?
     je .s_cleanup_keep_active                 ; no: an inference-active caller owns the activity state
     call v7s_stop_spin                        ; entered idle: return the pool to the OS wait between calls, this one-store leaf preserves EAX
.s_cleanup_keep_active:
     CLEANUP_YMM                               ; clear YMM upper halves before restoring callee-saved regs
     vmovdqu xmm6, s_xmm6                      ; restore callee-saved xmm6 low half
     vmovdqu xmm7, s_xmm7                      ; restore callee-saved xmm7 low half
     vmovdqu xmm8, s_xmm8                      ; restore callee-saved xmm8 low half
     vmovdqu xmm9, s_xmm9                      ; restore callee-saved xmm9 low half
     vmovdqu xmm10, s_xmm10                    ; restore callee-saved xmm10 low half
     vmovdqu xmm15, s_xmm15                    ; restore callee-saved xmm15 low half
     vmovdqu xmm14, s_xmm14                    ; restore callee-saved xmm14 low half
     vmovdqu xmm13, s_xmm13                    ; restore callee-saved xmm13 low half
     vmovdqu xmm12, s_xmm12                    ; restore callee-saved xmm12 low half
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore non-volatiles and return
.s_vocab_small_fatal:
     FATAL "sampler fork requires vocabulary >= 64 (two 32-element helper/caller ranges), smaller vocab cannot be split"
%include "sampler_stochastic_fork.inc"         ; three-phase adaptive stochastic pipeline, no pool transition between phases
%unmacro SIFT_DOWN 2                           ; release the max-heap sift-down macro, generic name, declared %macro SIFT_DOWN 2
%unmacro BUILD_HEAP 1                          ; release the in-place heap builder, generic name, declared %macro BUILD_HEAP 1
%unmacro S_EXP256_SOFT_PAIR 1                  ; release the interleaved softmax exp pair helper
%undef SF_TOKEN_VEC_APPEND
%undef TOPK_LIMIT
%undef SIFT_RIGHT_STEP
%undef SIFT_CHILD_SCALE
%undef PI_STRIDE
%undef PI_INDEX_OFF
%undef S_FLOAT4_SCALE
%undef S_VEC_B
%undef S_VEC_C
%undef S_VEC_D
%undef HEAP_PARENT_STEP
%undef HEAP_LAST_OFFSET
%undef STACK_ARG5_OFFSET
%undef STACK_ARG6_OFFSET
%undef STOCH_MIN_VOCAB
%undef SIFT_END_STEP
%undef NUCLEUS_WALK_STEP
%undef s_logits
%undef s_logits_raw
%undef s_vocab_size
%undef s_vocab_size_raw
%undef s_temp_bits
%undef s_temp_bits_raw
%undef s_topp_bits
%undef s_topp_bits_raw
%undef s_rng_state
%undef s_rng_state_raw
%undef s_probindex
%undef s_probindex_raw
%undef s_max_val_bits
%undef s_max_val_bits_raw
%undef s_coin
%undef s_coin_raw
%undef s_xmm6
%undef s_xmm7
%undef s_xmm8
%undef s_xmm9
%undef s_xmm10
%undef s_xmm12
%undef s_xmm13
%undef s_xmm14
%undef s_xmm15
