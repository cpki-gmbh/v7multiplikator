; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; Struct offsets
; ============================================================================
; reads the same two fields, S_XB, S_Q and must use the canonical names too.
; loff, kv_dim, kv_mul are 5th/6th/7th stack args, after shadow space + return address
%define ARG5_LOFF_OFFSET 48                    ; rbp+48 = loff, uint64_t, 5th stack arg
%define ARG6_KV_DIM_OFFSET 56                  ; rbp+56 = kv_dim, int, 6th stack arg
%define ARG7_KV_MUL_OFFSET 64                  ; rbp+64 = kv_mul, int, 7th stack arg
; Named constants for NUMOFFSET compliance, boundary check and stride offsets
%define HD_WSUM_LIMIT -32                      ; head_dim - 32 for wsum SIMD limit check
%define HD_DOT_LIMIT -64                       ; head_dim - 64 for dot SIMD limit check
%define SIMD_CHUNK_FLOATS 32                   ; floats per SIMD chunk, the 32-float stride every inner loop steps by
%define TRIP_COUNT_BASE 1                      ; countdown base: t runs 0..pos INCLUSIVE, so the trip count is pos + this
%define KV_HALF_SCALE 2                        ; packed FP16 K/V byte scale
%define ATTN_SCALE_TABLE_SCALE 4               ; sizeof,float: dword entries in attn_inv_sqrt_by32
%define FLOAT_SCALE 4                          ; bytes per float, index scale for rcx*N
%define Q_8FLT_STRIDE 32                       ; 32 bytes = one 8-float YMM block offset
%define Q_16FLT_STRIDE 64                      ; 64 bytes = two 8-float YMM blocks offset
%define Q_24FLT_STRIDE 96                      ; 96 bytes = three 8-float YMM blocks offset
%define Q_32FLT_STRIDE 128                     ; 128 bytes = four 8-float YMM blocks offset
%define Q_40FLT_STRIDE 160                     ; 160 bytes = five 8-float YMM blocks offset
%define Q_48FLT_STRIDE 192                     ; 192 bytes = six 8-float YMM blocks offset
%define Q_56FLT_STRIDE 224                     ; 224 bytes = seven 8-float YMM blocks offset
%define K_8FLT_STRIDE 16                       ; 16 bytes = one 8-half FP16 block offset, half of Q_8FLT_STRIDE
%define K_16FLT_STRIDE 32                      ; 32 bytes = two 8-half FP16 blocks, half of Q_16FLT_STRIDE
%define K_24FLT_STRIDE 48                      ; 48 bytes = three 8-half FP16 blocks, half of Q_24FLT_STRIDE
%define K_32FLT_STRIDE 64                      ; 64 bytes = four 8-half FP16 blocks, half of Q_32FLT_STRIDE
%define K_40FLT_STRIDE 80                      ; 80 bytes = five 8-half FP16 blocks, half of Q_40FLT_STRIDE
%define K_48FLT_STRIDE 96                      ; 96 bytes = six 8-half FP16 blocks, half of Q_48FLT_STRIDE
%define K_56FLT_STRIDE 112                     ; 112 bytes = seven 8-half FP16 blocks, half of Q_56FLT_STRIDE
     section .text

%macro POLY_2X 0
     vmovss xmm4, [rel attn_c5]                ; C5 coefficient is loaded
     vfmadd213ss xmm4, xmm3, [rel attn_c4]     ; xmm4 = fpart*C5 + C4
     vfmadd213ss xmm4, xmm3, [rel attn_c3]     ; xmm4 = fpart*prev + C3
     vfmadd213ss xmm4, xmm3, [rel attn_c2]     ; xmm4 = fpart*prev + C2
     vfmadd213ss xmm4, xmm3, [rel attn_c1]     ; xmm4 = fpart*prev + C1
     vfmadd213ss xmm4, xmm3, [rel rpe_one]     ; xmm4 = fpart*prev + 1.0
%endmacro

%macro INLINE_EXP 0
     vmaxss xmm0, xmm0, [rel scalar_m80]       ; Clamp lower: x >= -80, prevent underflow; input <= 0 at both sites
     vmulss xmm1, xmm0, [rel scalar_log2e]     ; xmm1 = approx,x/ln2
     vroundss xmm2, xmm1, xmm1, 0              ; xmm2 = ipart = round-to-nearest,x/ln2
     vfnmadd231ss xmm0, xmm2, [rel s_exp_ln2_hi] ; xmm0 = x - ipart*ln2_hi, fused, single rounding
     vfnmadd231ss xmm0, xmm2, [rel s_exp_ln2_lo] ; xmm0 = r = x - ipart*ln2, precise residual
     vmulss xmm3, xmm0, [rel scalar_log2e]     ; xmm3 = fpart = r * log2,e in [-0.5, +0.5]
     vcvttps2dq xmm5, xmm2                     ; xmm5 lane0 = n, xmm2 floor-valued: trunc exact; vcvttps2dq zeroes xmm5 upper lanes
     vpaddd xmm5, xmm5, [rel swk_bias_127]     ; lane0 = n + 127, integer add, reads low 128 bits of the table
     vpslld xmm5, xmm5, 23                     ; lane0 bits =, n+127 <<23 = 2^ipart
     POLY_2X                                   ; xmm4 = poly,fpart
     vmulss xmm0, xmm5, xmm4                   ; exp,x = 2^ipart * poly,fpart, vmulss reads lane 0 only
%endmacro
; ============================================================================

%macro FOR_EACH_ZERO 1
     lea ecx, [r14 + HD_WSUM_LIMIT]            ; ecx = head_dim - 32, float index, last chunk first; r14d = head_dim
     vxorps ymm0, ymm0, ymm0                   ; Zero ymm0, hoisted outside loop
     align 32                                  ; 32-byte alignment for SIMD loop
%%_lp_%1:
     vmovups [rdi + rcx*FLOAT_SCALE], ymm0     ; zeros are stored at [i:i+7]
     vmovups [rdi + rcx*FLOAT_SCALE + Q_8FLT_STRIDE], ymm0 ; zeros are stored at [i+8:i+15]
     vmovups [rdi + rcx*FLOAT_SCALE + Q_16FLT_STRIDE], ymm0 ; zeros are stored at [i+16:i+23]
     vmovups [rdi + rcx*FLOAT_SCALE + Q_24FLT_STRIDE], ymm0 ; zeros are stored at [i+24:i+31]
     sub ecx, SIMD_CHUNK_FLOATS                ; Step down one 32-float chunk, negative once the whole buffer is covered
     jns %%_lp_%1                              ; Continue while the index is still >= 0, head_dim % 32 == 0, so it lands on exactly -32
%endmacro
; ============================================================================

%macro FOR_EACH_NORM 1
     lea ecx, [r14 + HD_WSUM_LIMIT]            ; ecx = head_dim - 32, float index, last chunk first; r14d = head_dim
     align 32                                  ; 32-byte alignment for normalize SIMD loop
%%_lp_%1:
     vmovups ymm0, [rdi + rcx*FLOAT_SCALE]     ; [i:i+7] is loaded
     vmovups ymm1, [rdi + rcx*FLOAT_SCALE + Q_8FLT_STRIDE] ; [i+8:i+15] is loaded
     vmovups ymm2, [rdi + rcx*FLOAT_SCALE + Q_16FLT_STRIDE] ; [i+16:i+23] is loaded
     vmovups ymm3, [rdi + rcx*FLOAT_SCALE + Q_24FLT_STRIDE] ; [i+24:i+31] is loaded
     vdivps ymm0, ymm0, ymm5                   ; [i:i+7] is divided by d0, correctly rounded division
     vdivps ymm1, ymm1, ymm5                   ; [i+8:i+15] is divided by d0
     vdivps ymm2, ymm2, ymm5                   ; [i+16:i+23] is divided by d0
     vdivps ymm3, ymm3, ymm5                   ; [i+24:i+31] is divided by d0
     vmovups [rdi + rcx*FLOAT_SCALE], ymm0     ; normalized value is stored
     vmovups [rdi + rcx*FLOAT_SCALE + Q_8FLT_STRIDE], ymm1 ; normalized value is stored
     vmovups [rdi + rcx*FLOAT_SCALE + Q_16FLT_STRIDE], ymm2 ; normalized value is stored
     vmovups [rdi + rcx*FLOAT_SCALE + Q_24FLT_STRIDE], ymm3 ; normalized value is stored
     sub ecx, SIMD_CHUNK_FLOATS                ; Step down one 32-float chunk, negative once the whole buffer is covered
     jns %%_lp_%1                              ; Loop while the index is still >= 0, head_dim % 32 == 0, so it lands on exactly -32
%endmacro
; ============================================================================

%macro SINGLE_HEAD_DOT 1
     vxorps ymm0, ymm0, ymm0                   ; Zero acc0, lane0, floats 0-7, 32-39
     vxorps ymm1, ymm1, ymm1                   ; Zero acc1, lane1, floats 8-15, 40-47
     vxorps ymm2, ymm2, ymm2                   ; Zero acc2, lane2, floats 16-23, 48-55
     vxorps ymm3, ymm3, ymm3                   ; Zero acc3, lane3, floats 24-31, 56-63
     xor ecx, ecx                              ; ecx = 0, float index
     STEP_DOT_64_REG                           ; First 64 floats via ymm8-ymm13
     ds prefetcht0 [r10 + rsi]                 ; value row of next timestep, overlaps the remaining K dot + softmax + V aggregation, 6-byte type-only form per the attention slots
     PREFETCH_SLOT 40                          ; mark patchable slot 40 for SMC
     mov ecx, 64                               ; Advance index past register-hoisted chunk
     cmp ecx, r9d                              ; i > head_dim - 64?, loop bound check
     jg %%_dot_rem_%1                          ; yes: straight to the remainder, fall-through enters the loop.jg is the exact signed complement of the jle back edge below, entry test and back edge must share one comparison width, r9d >= 0 for every reachable head_dim >= 64, so signed == unsigned here
     align 32                                  ; 32-byte alignment for dot SIMD loop
%%_dot_lp_%1:
     STEP_DOT_64 ymm0, ymm1, ymm2, ymm3, r11   ; accumulate 64 floats into dot-product lanes
     add ecx, 64                               ; Advance by 64 floats
     cmp ecx, r9d                              ; another 64-float chunk fits?
     jle %%_dot_lp_%1                          ; bottom-test: loop without a jmp back
%%_dot_rem_%1:
     cmp ecx, r14d                             ; remainder == 0?, head_dim%64==0
     jae %%_dot_reduce_%1                      ; yes: reduce now
     vcvtph2ps ymm4, oword [r11 + rcx*KV_HALF_SCALE] ; expand K[i..i+7]
     vcvtph2ps ymm5, oword [r11 + rcx*KV_HALF_SCALE + K_8FLT_STRIDE] ; expand K[i+8..i+15]
     vfmadd231ps ymm0, ymm4, [r13 + rcx*FLOAT_SCALE] ; acc_lane0 += k[i..i+7] * q[i..i+7]
     vfmadd231ps ymm1, ymm5, [r13 + rcx*FLOAT_SCALE + Q_8FLT_STRIDE] ; acc_lane1 += k[i+8..i+15] * q[i+8..i+15]
     add ecx, 16                               ; advance two 8-float blocks
     vcvtph2ps ymm4, oword [r11 + rcx*KV_HALF_SCALE] ; expand K[i+16..i+23]
     vcvtph2ps ymm5, oword [r11 + rcx*KV_HALF_SCALE + K_8FLT_STRIDE] ; expand K[i+24..i+31]
     vfmadd231ps ymm0, ymm4, [r13 + rcx*FLOAT_SCALE] ; acc_lane0 += k[i+16..i+23] * q[i+16..i+23]
     vfmadd231ps ymm1, ymm5, [r13 + rcx*FLOAT_SCALE + Q_8FLT_STRIDE] ; acc_lane1 += k[i+24..i+31] * q[i+24..i+31]
%%_dot_reduce_%1:
     vaddps ymm0, ymm0, ymm1                   ; lane0 + lane1 = upper half
     vaddps ymm2, ymm2, ymm3                   ; lane2 + lane3 = lower half
     vaddps ymm0, ymm0, ymm2                   ; Combine both halves
     REDUCE_YMM_TO_SCALAR_FAST xmm0, ymm0, xmm1, xmm2 ; Reduce 8 floats to scalar via fast extract+shuffle
     vmulss xmm0, xmm0, xmm6                   ; score *= scale, hoisted in xmm6, no load-port uop per timestep
%endmacro
; ============================================================================

; SINGLE_HEAD_WSUM id, old_scale_src  -- cold path only, new_scale == 1.0

%macro SINGLE_HEAD_WSUM 2
     vbroadcastss ymm4, %2                     ; old_scale = exp(previous_max - new_max)
     lea ecx, [r14 + HD_WSUM_LIMIT]            ; ecx = head_dim - 32, float index, last chunk first
     align 32                                  ; 32-byte alignment for weighted-sum SIMD loop
%%_wlp_%1:
     vcvtph2ps ymm7, oword [r10 + rcx*KV_HALF_SCALE] ; expand packed V, stream A, issued before the xb loads
     vcvtph2ps ymm5, oword [r10 + rcx*KV_HALF_SCALE + K_8FLT_STRIDE] ; stream B
     vmovups ymm0, [rdi + rcx*FLOAT_SCALE]     ; xb[i:i+7]
     vmovups ymm1, [rdi + rcx*FLOAT_SCALE + Q_8FLT_STRIDE] ; xb[i+8:i+15]
     vmovups ymm2, [rdi + rcx*FLOAT_SCALE + Q_16FLT_STRIDE] ; xb[i+16:i+23]
     vmovups ymm3, [rdi + rcx*FLOAT_SCALE + Q_24FLT_STRIDE] ; xb[i+24:i+31]
     vfmadd213ps ymm0, ymm4, ymm7              ; xb = fma(xb, old_scale, V), ONE rounding
     vfmadd213ps ymm1, ymm4, ymm5              ; xb = fma(xb, old_scale, V)
     vcvtph2ps ymm7, oword [r10 + rcx*KV_HALF_SCALE + K_16FLT_STRIDE] ; stream A, second pair
     vcvtph2ps ymm5, oword [r10 + rcx*KV_HALF_SCALE + K_24FLT_STRIDE] ; stream B, second pair
     vfmadd213ps ymm2, ymm4, ymm7              ; xb = fma(xb, old_scale, V)
     vfmadd213ps ymm3, ymm4, ymm5              ; xb = fma(xb, old_scale, V)
     vmovups [rdi + rcx*FLOAT_SCALE], ymm0     ; store updated xb
     vmovups [rdi + rcx*FLOAT_SCALE + Q_8FLT_STRIDE], ymm1 ; store updated xb
     vmovups [rdi + rcx*FLOAT_SCALE + Q_16FLT_STRIDE], ymm2 ; store updated xb
     vmovups [rdi + rcx*FLOAT_SCALE + Q_24FLT_STRIDE], ymm3 ; store updated xb
     sub ecx, SIMD_CHUNK_FLOATS                ; step down one 32-float chunk
     jns %%_wlp_%1                             ; continue while the index is still >= 0
; head_dim % 32 == 0, init check -> ecx == -32 here, no remainder, loop needs no tail
%endmacro

%macro SINGLE_HEAD_WSUM_HOT 2
     vbroadcastss ymm5, %2                     ; ymm5 = new_scale broadcast
     lea ecx, [r14 + HD_WSUM_LIMIT]            ; ecx = head_dim - 32, float index, last chunk first; r14d = head_dim
     align 32                                  ; 32-byte alignment for hot weighted-sum SIMD loop
%%_wlp_%1:
     vcvtph2ps ymm4, oword [r10 + rcx*KV_HALF_SCALE] ; expand packed V floats 0-7
     vcvtph2ps ymm7, oword [r10 + rcx*KV_HALF_SCALE + K_8FLT_STRIDE] ; expand packed V floats 8-15
     vcvtph2ps ymm2, oword [r10 + rcx*KV_HALF_SCALE + K_16FLT_STRIDE] ; expand packed V floats 16-23
     vcvtph2ps ymm3, oword [r10 + rcx*KV_HALF_SCALE + K_24FLT_STRIDE] ; expand packed V floats 24-31
     vmovups ymm0, [rdi + rcx*FLOAT_SCALE]     ; xb[i:i+7] is loaded
     vmovups ymm1, [rdi + rcx*FLOAT_SCALE + Q_8FLT_STRIDE] ; xb[i+8:i+15] is loaded
     vfmadd231ps ymm0, ymm5, ymm4              ; xb += new_scale * v at [i:i+7]
     vfmadd231ps ymm1, ymm5, ymm7              ; xb += new_scale * v at [i+8:i+15]
     vmovups [rdi + rcx*FLOAT_SCALE], ymm0     ; xb[i:i+7] is stored, retires ymm0 for reuse
     vmovups [rdi + rcx*FLOAT_SCALE + Q_8FLT_STRIDE], ymm1 ; xb[i+8:i+15] is stored, retires ymm1 for reuse
     vmovups ymm0, [rdi + rcx*FLOAT_SCALE + Q_16FLT_STRIDE] ; xb[i+16:i+23] is loaded, distinct from the stored floats
     vmovups ymm1, [rdi + rcx*FLOAT_SCALE + Q_24FLT_STRIDE] ; xb[i+24:i+31] is loaded, distinct from the stored floats
     vfmadd231ps ymm0, ymm5, ymm2              ; xb += new_scale * v at [i+16:i+23]
     vfmadd231ps ymm1, ymm5, ymm3              ; xb += new_scale * v at [i+24:i+31]
     vmovups [rdi + rcx*FLOAT_SCALE + Q_16FLT_STRIDE], ymm0 ; xb[i+16:i+23] is stored
     vmovups [rdi + rcx*FLOAT_SCALE + Q_24FLT_STRIDE], ymm1 ; xb[i+24:i+31] is stored
     sub ecx, SIMD_CHUNK_FLOATS                ; Step down one 32-float chunk, negative once the whole buffer is covered
     jns %%_wlp_%1                             ; Continue SIMD loop while the index is still >= 0
; head_dim % 32 == 0, init check -> ecx == -32 here, no remainder, loop needs no tail
%endmacro

%macro NORMALIZE_HEAD 2
     vbroadcastss ymm5, %2                     ; ymm5 = d0 broadcast, 8 lanes, FOR_EACH_NORM divides by it
     FOR_EACH_NORM %1                          ; Normalize entire xb buffer by d0, FOR_EACH_NORM's own %%_lp_%1 exits on ecx < 0, so control returns here
%endmacro
; ============================================================================

%macro STEP_DOT_64 5
; Pair 1: expand both K halves first, then update accumulators 0/1
     vcvtph2ps ymm4, oword [%5 + rcx*KV_HALF_SCALE] ; K[0:7]
     vcvtph2ps ymm5, oword [%5 + rcx*KV_HALF_SCALE + K_8FLT_STRIDE] ; K[8:15]
     vfmadd231ps %1, ymm4, [r13 + rcx*FLOAT_SCALE] ; acc0 += k[0:7] * q[0:7]
     vfmadd231ps %2, ymm5, [r13 + rcx*FLOAT_SCALE + Q_8FLT_STRIDE] ; acc1 += k[8:15] * q[8:15]
; Pair 2: expand K16/K24, then update accumulators 2/3
     vcvtph2ps ymm4, oword [%5 + rcx*KV_HALF_SCALE + K_16FLT_STRIDE] ; K[16:23]
     vcvtph2ps ymm5, oword [%5 + rcx*KV_HALF_SCALE + K_24FLT_STRIDE] ; K[24:31]
     vfmadd231ps %3, ymm4, [r13 + rcx*FLOAT_SCALE + Q_16FLT_STRIDE] ; acc2 += k[16:23] * q[16:23]
     vfmadd231ps %4, ymm5, [r13 + rcx*FLOAT_SCALE + Q_24FLT_STRIDE] ; acc3 += k[24:31] * q[24:31]
; Pair 3: expand K32/K40, then update accumulators 0/1
     vcvtph2ps ymm4, oword [%5 + rcx*KV_HALF_SCALE + K_32FLT_STRIDE] ; K[32:39]
     vcvtph2ps ymm5, oword [%5 + rcx*KV_HALF_SCALE + K_40FLT_STRIDE] ; K[40:47]
     vfmadd231ps %1, ymm4, [r13 + rcx*FLOAT_SCALE + Q_32FLT_STRIDE] ; acc0 += k[32:39] * q[32:39]
     vfmadd231ps %2, ymm5, [r13 + rcx*FLOAT_SCALE + Q_40FLT_STRIDE] ; acc1 += k[40:47] * q[40:47]
; Pair 4: expand K48/K56, then update accumulators 2/3
     vcvtph2ps ymm4, oword [%5 + rcx*KV_HALF_SCALE + K_48FLT_STRIDE] ; K[48:55]
     vcvtph2ps ymm5, oword [%5 + rcx*KV_HALF_SCALE + K_56FLT_STRIDE] ; K[56:63]
     vfmadd231ps %3, ymm4, [r13 + rcx*FLOAT_SCALE + Q_48FLT_STRIDE] ; acc2 += k[48:55] * q[48:55]
     vfmadd231ps %4, ymm5, [r13 + rcx*FLOAT_SCALE + Q_56FLT_STRIDE] ; acc3 += k[56:63] * q[56:63]
%endmacro

%macro STEP_DOT_64_REG 0
     vcvtph2ps ymm4, oword [r11 + rcx*KV_HALF_SCALE] ; expand packed K into FP32, pairs 0-7
     vcvtph2ps ymm5, oword [r11 + rcx*KV_HALF_SCALE + K_8FLT_STRIDE] ; expand packed K into FP32, pairs 8-15
     vfmadd231ps ymm0, ymm8, ymm4              ; acc0 += q[0:7] * k[0:7]
     vfmadd231ps ymm1, ymm9, ymm5              ; acc1 += q[8:15] * k[8:15]
     vcvtph2ps ymm4, oword [r11 + rcx*KV_HALF_SCALE + K_16FLT_STRIDE] ; expand packed K into FP32, pairs 16-23
     vcvtph2ps ymm5, oword [r11 + rcx*KV_HALF_SCALE + K_24FLT_STRIDE] ; expand packed K into FP32, pairs 24-31
     vfmadd231ps ymm2, ymm10, ymm4             ; acc2 += q[16:23] * k[16:23]
     vfmadd231ps ymm3, ymm11, ymm5             ; acc3 += q[24:31] * k[24:31]
     vcvtph2ps ymm4, oword [r11 + rcx*KV_HALF_SCALE + K_32FLT_STRIDE] ; expand packed K into FP32, pairs 32-39
     vcvtph2ps ymm5, oword [r11 + rcx*KV_HALF_SCALE + K_40FLT_STRIDE] ; expand packed K into FP32, pairs 40-47
     vfmadd231ps ymm0, ymm12, ymm4             ; acc0 += q[32:39] * k[32:39]
     vfmadd231ps ymm1, ymm13, ymm5             ; acc1 += q[40:47] * k[40:47]
     vcvtph2ps ymm4, oword [r11 + rcx*KV_HALF_SCALE + K_48FLT_STRIDE] ; expand packed K, pairs 48-55
     vcvtph2ps ymm5, oword [r11 + rcx*KV_HALF_SCALE + K_56FLT_STRIDE] ; expand packed K, pairs 56-63
     vfmadd231ps ymm2, ymm4, [r13 + rcx*FLOAT_SCALE + Q_48FLT_STRIDE] ; acc2 += q_mem[48:55] * k[48:55]
     vfmadd231ps ymm3, ymm5, [r13 + rcx*FLOAT_SCALE + Q_56FLT_STRIDE] ; acc3 += q_mem[56:63] * k[56:63]
%endmacro
; ============================================================================
; STREAM_ROTATE_Q, delta-rotate the r13 head slice in place, half-split RoPE,
; house algebra: lower' = lower*cos - upper*sin, upper' = lower*sin + upper*cos
; identical to rope_only_kernel.inc / fused_f32_kernel.inc, indexed by
; DELTA row. Input eax = delta, >= 1.Overwrites rax, rcx, rdx, r10, r11,
; ymm0-5, K/V bases and the hoisted q are, re loaded after every rotation;
; r8, r9, r12-r15, rbx, rsi, rdi, xmm6/7/13/14/15, ymm8-13 preserved.Only
; the DELTA is ever indexed: delta <= W < C <= seq_len, so the 131072-row
; fused tables always cover it, virtual positions, h+W are never looked up.
; One ymm block = 8 pairs per iteration, advance 8. Requires half % 8 == 0, i.e.
; head_dim % 16 == 0, implied by the head_dim % 32 == 0 init contract.
; All table/slice indexing uses FLOAT_SCALE, bytes per float, same file per
; the NUMOFFSET house rule.
; ============================================================================

%macro STREAM_ROTATE_Q 0
     mov r10d, r14d                            ; r10d = head_dim
     mov rcx, [rel fused_rope_cos]             ; rcx = fused cos table base
     mov rdx, [rel fused_rope_sin]             ; rdx = fused sin table base
     shr r10d, 1                               ; r10d = half = head_dim / 2
     imul eax, r10d                            ; eax = delta * half, float offset
     lea rcx, [rcx + rax*FLOAT_SCALE]          ; cos row, unscaled float index * 4
     lea rdx, [rdx + rax*FLOAT_SCALE]          ; sin row, same float index * 4
     lea rax, [r13 + r10*FLOAT_SCALE]          ; rax = upper-half base, r10 = half
     xor r11d, r11d                            ; r11d = pair index, float index into the lower half
%%srq_lp:
     vmovups ymm4, [rcx + r11*FLOAT_SCALE]     ; cos block, 8 floats
     vmovups ymm5, [rdx + r11*FLOAT_SCALE]     ; sin block, 8 floats
     vmovups ymm0, [r13 + r11*FLOAT_SCALE]     ; lower halves, pairs j..j+7
     vmovups ymm1, [rax + r11*FLOAT_SCALE]     ; upper halves
     vmulps ymm2, ymm0, ymm4                   ; lower * cos
     vmulps ymm3, ymm0, ymm5                   ; lower * sin, independent of the first product
     vfnmadd231ps ymm2, ymm1, ymm5             ; lower*cos - upper*sin, house convention
     vfmadd231ps ymm3, ymm1, ymm4              ; lower*sin + upper*cos, house convention
     vmovups [r13 + r11*FLOAT_SCALE], ymm2     ; store rotated lower
     vmovups [rax + r11*FLOAT_SCALE], ymm3     ; store rotated upper
     add r11, 8                                ; 8 pairs per iteration, one ymm block per half
     cmp r11, r10                              ; covered all half pairs?
     jb %%srq_lp                               ; head_dim = 64: 4 iterations cover all 32 pairs
%endmacro

     align 64                                  ; 64-byte alignment for hot function entry
     global v7s_attention_inner                ; export single-head attention kernel
v7s_attention_inner:
     SF_START v7s_attention_inner, rbx, rsi, rdi, r12, r13, r14, r15 ; declare stack frame
; ---- Non-volatile XMM saves, lower 128b of ymm8-ymm12, upper 128b volatile per Win64 ABI ----
     SF_XMM stk_ymm8                           ; ymm8 save area, q[0:7]
     SF_XMM stk_ymm9                           ; ymm9 save area, q[8:15]
     SF_XMM stk_ymm10                          ; ymm10 save area, q[16:23]
     SF_XMM stk_ymm11                          ; ymm11 save area, q[24:31]
     SF_XMM stk_ymm12                          ; ymm12 save area, q[32:39]
; ---- XMM saves ----
     SF_XMM stk_xmm6                           ; xmm6 save area, vmovdqu access, slot is 8-mod-16 in this odd-reg frame; do NOT switch to aligned moves, hoisted scale
     SF_XMM stk_xmm13                          ; xmm13 save area, vmovdqu access, slot is 8-mod-16 in this odd-reg frame; do NOT switch to aligned moves, q[40:47]
     SF_XMM stk_xmm14                          ; xmm14 save area, vmovdqu access, slot is 8-mod-16 in this odd-reg frame; do NOT switch to aligned moves
     SF_XMM stk_xmm15                          ; xmm15 save area, vmovdqu access, slot is 8-mod-16 in this odd-reg frame; do NOT switch to aligned moves
     SF_XMM stk_xmm7                           ; xmm7 save area, vmovdqu access, slot is 8-mod-16 in this odd-reg frame; do NOT switch to aligned moves, s parking on cold path
     SF_DWORD sm_h                             ; StreamingLLM self slot h, mode select; read by the fresh/sinks/stale phase loaders
     SF_DWORD sm_fresh                         ; clamped fresh trip = min,h - S + 1, valid, mode select computes once per head call
     SF_DWORD sm_stale                         ; clamped stale trip = valid - fresh', 0 empties the stale phase
     SF_COMMIT                                 ; commit stack frame, allocate shadow
; ---- Save non-volatile YMM/XMM, must be after SF_COMMIT ----
     vmovdqu stk_ymm8, xmm8                    ; Save non-volatile xmm8, lower 128b of hoisted q[0:7]
     vmovdqu stk_ymm9, xmm9                    ; Save non-volatile xmm9, lower 128b of hoisted q[8:15]
     vmovdqu stk_ymm10, xmm10                  ; Save non-volatile xmm10, lower 128b of hoisted q[16:23]
     vmovdqu stk_ymm11, xmm11                  ; Save non-volatile xmm11, lower 128b of hoisted q[24:31]
     vmovdqu stk_ymm12, xmm12                  ; Save non-volatile xmm12, lower 128b of hoisted q[32:39]
     vmovdqu stk_xmm6, xmm6                    ; Save non-volatile xmm6, scale hoist register
     vmovdqu stk_xmm13, xmm13                  ; Save non-volatile xmm13
     vmovdqu stk_xmm14, xmm14                  ; Save non-volatile xmm14
     vmovdqu stk_xmm15, xmm15                  ; Save non-volatile xmm15
     vmovdqu stk_xmm7, xmm7                    ; Save non-volatile xmm7, cold-path s parking
; === Parameters ===
     mov r12, rcx                              ; r12 = RunState* s, non-volatile save
     mov ebx, r9d                              ; park pos in non-volatile rbx, r9d = pos; rbx saved by SF_START
     mov ecx, [rbp + ARG7_KV_MUL_OFFSET]       ; ecx = kv_mul, 7th stack arg, kept live in ecx for the div below; >= 1 by GQA contract
; === head_dim ===
     mov r14d, [r8 + CFG_HEAD_DIM]             ; r14d = head_dim, rot per head; the pass applies exactly head_dim*2 rotations per position
; === Hoist SIMD threshold, r8 = Config* no longer needed after head_dim load ===
     lea r9d, [r14 + HD_DOT_LIMIT]             ; r9d = head_dim - 64, dot SIMD limit, 64-float stride
     mov eax, r14d                             ; eax = head_dim
     cmp eax, 64                               ; inside the precomputed table range?
     jb .attn_scale_direct                     ; below 64: fallback
     cmp eax, 1024                             ; above 1024: fallback
     ja .attn_scale_direct                     ; beyond the table: fallback
     test eax, 31                              ; table covers multiples of 32 only
     jnz .attn_scale_direct                    ; odd size: fallback
     shr eax, 5                                ; eax = table index
     lea r10, [rel attn_inv_sqrt_by32]         ; r10 = table base, init-once in v7_init
     vmovss xmm6, [r10 + rax*ATTN_SCALE_TABLE_SCALE] ; xmm6 = 1/sqrt(head_dim), computed under the init MXCSR
     jmp .attn_scale_ready                     ; join the hoist path, scale in xmm6
.attn_scale_direct:
     vcvtsi2ss xmm0, xmm0, r14d                ; scalar float head_dim; only lane 0 is used before xmm0 is overwritten
     vsqrtss xmm0, xmm0, xmm0                  ; xmm0 = precise sqrt,head_dim
     vmovss xmm1, [rel rpe_one]                ; xmm1 = 1.0f, staged: VDIVSS requires src1 in VEX.vvvv as a register, so the constant cannot fold into a memory operand
     vdivss xmm6, xmm1, xmm0                   ; xmm6 = exactly 1.0 / sqrt,head_dim, hoisted for the per-timestep dot; non-volatile, SINGLE_HEAD_DOT preserves no volatile XMM
.attn_scale_ready:
; === h * head_dim * 4, computed once, reused for q_ptr and xb_ptr ===
     mov eax, edx                              ; eax = h, head index; edx untouched since entry
     imul rax, r14                             ; rax = h * head_dim, float index
     shl rax, 2                                ; rax =, h * head_dim * 4, byte offset
; === q_ptr = s->q + byte_offset ===
     mov r13, [r12 + S_Q]                      ; r13 = s->q
     add r13, rax                              ; r13 = q_ptr, non-volatile, persistent across timesteps
; === xb_ptr = s->xb + byte_offset, shares the one offset computation in rax ===
     mov rdi, [r12 + S_XB]                     ; rdi = s->xb
     add rdi, rax                              ; rdi = xb_ptr, output buffer for this head, pinned across hot loop
; === static_offset_bytes =, loff +, h/kv_mul *head_dim * 4 ===
     mov eax, edx                              ; eax = h, head index for group/offset computation
     blsr r10d, ecx                            ; r10d = kv_mul & (kv_mul-1), ZF set iff kv_mul is a power of two
     jnz .kv_generic_div                       ; no: keep the exact generic divide
     bsf ecx, ecx                              ; ecx = log2(kv_mul), well-defined for kv_mul >= 1
     shr eax, cl                               ; eax = h / kv_mul, exact for powers of two
     jmp .kv_quotient_done                     ; skip the generic divide, quotient ready
.kv_generic_div:
     xor edx, edx                              ; Zero edx for div, high 32 bits of dividend
     div ecx                                   ; eax = h / kv_mul, exact for every kv_mul
.kv_quotient_done:
     imul rax, r14                             ; rax =, h/kv_mul *head_dim, float count, group-head offset
     add rax, [rbp + ARG5_LOFF_OFFSET]         ; rax = loff +, h/kv_mul *head_dim, 5th stack arg, loff in float-element units
     shl rax, 1                                ; packed FP16 K/V offset, two bytes per element
     mov r15, [r12 + S_VALUE_CACHE]            ; value plane before replacing the state pointer
     add r15, rax                              ; value base for this layer and KV head
     add rax, [r12 + S_KEY_CACHE]              ; key base for this layer and KV head
     mov r12, rax                              ; retain the key base across all attention phases
; === kv_dim_bytes ===
     mov eax, [rbp + ARG6_KV_DIM_OFFSET]       ; eax = kv_dim, 6th stack arg, int; 32-bit load zero-extends
     lea rsi, [rax + rax]                      ; rsi = kv_dim_bytes, two bytes per element, stride per timestep
; === Zero xb buffer, required for online softmax accumulation ===
     FOR_EACH_ZERO _sh                         ; zero xb buffer for online softmax accumulation
; === StreamingLLM mode select, early exits land BEFORE the softmax init ===
; phases: 0 = fresh [S..h], q @ h, 1 = sinks [0..S-1], q @ C-1,
; 2 = stale [h+1..C-1], q @ h+W, 3 = legacy single pass [0..pos]
     mov r8d, 3                                ; default: legacy single pass, r8 free since the Config* load above
     cmp dword [rel g_stream_enable], 0        ; streaming enabled?
     je .t_init_done                           ; legacy: fall into the softmax init, never jump past it
     cmp ebx, [rel g_stream_cap]               ; pos < C, linear fill ? UNSIGNED: positions are 32-bit absolute counters, the resolvers use jb too, a signed jl would misread pos >= 2^31 as negative and pick the legacy pass
     jb .t_init_done                           ; linear fill behaves exactly legacy
     mov edx, [rel g_stream_slot]              ; edx = h, published once per token by forward_*
     mov sm_h, edx                             ; park self slot in the frame
     mov eax, [rel g_stream_win_valid]         ; eax = valid window rows, rollback-aware; W in the canonical case
     sub edx, [rel g_stream_sinks]             ; edx = h - S
     inc edx                                   ; edx = fresh_trip = h - S + 1
     cmp edx, eax                              ; clamp the fresh trip to the valid count
     jbe .t_fresh_ok                           ; canonical case, valid = W >= fresh_trip : unclamped
     mov edx, eax                              ; fresh' = min,fresh_trip, valid, attend only true rows
.t_fresh_ok:
     mov sm_fresh, edx                         ; park for the phase-0 loader
     sub eax, edx                              ; eax = stale' = valid - fresh', >= 0 by the clamp
     mov sm_stale, eax                         ; park for the stale loader
     xor r8d, r8d                              ; phase 0 = fresh segment
.t_init_done:
; === Online softmax state init, single precision; every entry path passes
; === through exactly once; phase-to-phase advances land AFTER it ===
     vmovss xmm14, [rel scalar_neg_flt_max]    ; xmm14 = m0 = -FLT_MAX, running max, float
     vxorps xmm15, xmm15, xmm15                ; xmm15 = d0 = 0.0f, running sum, float
; === phase loader: K/V base + trip count per phase, m0/d0 persist across ===
.t_phase_load:
     cmp r8d, 3                                ; legacy / linear-fill phase?
     jne .t_stream_phase                       ; no -> streaming phases below
; --- legacy / linear-fill pass: slots [0..pos], q already rotated to pos ----
     mov r11, r12                              ; start at this head's key base
     mov r10, r15                              ; start at this head's value base
     lea edx, [rbx + TRIP_COUNT_BASE]          ; edx = pos + 1, countdown trip count: t = 0..pos inclusive; rbx = pos has no further role
     jmp .t_hoist                              ; K/V base + trip loaded -> hoist q and enter the loop
.t_stream_phase:
     test r8d, r8d                             ; phase 0, fresh ?
     jne .t_phase_sinks                        ; no -> sinks next
     mov edx, sm_fresh                         ; edx = fresh', clamped trip
     test edx, edx                             ; zero valid fresh rows, deep tier-2 rollback shaved the tracker to sinks-only ?
     jz .t_phase0_empty                        ; yes: skip the fresh phase entirely, a 0 trip would underflow the countdown dec and walk 2^32 rows
; --- phase 0: fresh segment [S..h]; q needs NO rotation, it sits at h ------
     mov r11, r12                              ; start at this head's key base
     mov r10, r15                              ; start at this head's value base
     mov ecx, sm_h                             ; ecx = h
     sub ecx, edx                              ; h - fresh'
     inc ecx                                   ; ecx = first fresh slot = h - fresh' + 1, == S unclamped
     imul rcx, rsi                             ; first slot * kv_dim_bytes
     add r11, rcx                              ; key base at the first valid fresh slot
     add r10, rcx                              ; value base at the same slot
     jmp .t_hoist                              ; K/V base + trip loaded -> hoist q and enter the loop
.t_phase0_empty:
     inc r8d                                   ; no fresh rows: phase 0 -> 1, the sinks loader runs next
     jmp .t_phase_load                         ; straight to the sinks phase
.t_phase_sinks:
     cmp r8d, 1                                ; phase 1, sinks ?
     jne .t_phase_stale                        ; no -> stale next
; --- phase 1: sinks [0..S-1]; delta-rotate q from h to C-1 ------------------
     mov eax, [rel g_stream_cap]               ; eax = C
     dec eax                                   ; eax = C - 1
     sub eax, sm_h                             ; eax = delta1 = C - 1 - h, 0 when h == C-1, sets ZF for the next test
     jz .t_sinks_norot                         ; skip the rotation entirely
     STREAM_ROTATE_Q                           ; q: h -> C-1, overwrites rax/rcx/rdx/r10/r11
.t_sinks_norot:
     mov r11, r12                              ; start at this head's key base
     mov r10, r15                              ; start at this head's value base
     mov edx, [rel g_stream_sinks]             ; trip = S, >= 1, API-validated
     jmp .t_hoist                              ; K/V base + trip loaded -> hoist q and enter the loop
.t_phase_stale:
     cmp dword sm_stale, 0                     ; empty stale segment?, stale' is read only for the zero test here
     jnz .t_stale_go                           ; stale segment non-empty -> rotate + load
     inc r8d                                   ; empty stale segment -> phase 2 -> 3, r8d now 3: no phase pending
     jmp .t_done                               ; straight out: nothing left to attend, m0/d0 carried, normalize runs
.t_stale_go:
     mov eax, sm_h                             ; eax = h, never overwritten: the trip lives in sm_stale
     inc eax                                   ; eax = h + 1
     sub eax, [rel g_stream_sinks]             ; eax = delta2 = h - S + 1, >= 1, always rotates
     STREAM_ROTATE_Q                           ; q: C-1 -> C-1+,h-S+1 = h+W, overwrites rax/rcx/rdx/r10/r11
     mov edx, sm_stale                         ; edx = stale', reloaded AFTER the macro, ecx was overwritten
     mov r11, r12                              ; start at this head's key base
     mov r10, r15                              ; start at this head's value base
     mov eax, [rel g_stream_cap]               ; eax = C
     sub eax, edx                              ; eax = C - stale' = first stale slot: identity h+1 in the canonical case, and the ring TOP for a shaved tracker, the correct new-timeline rows sit at [C-stale', C-1], never the old-timeline slots [h+1, h+stale']
     imul rax, rsi                             ; first stale slot * kv_dim_bytes
     add r11, rax                              ; key base at slot C - stale'
     add r10, rax                              ; value base at slot C - stale'
.t_hoist:
     vmovups ymm8, [r13]                       ; ymm8 = q[0:7]
     vmovups ymm9, [r13 + Q_8FLT_STRIDE]       ; ymm9 = q[8:15]
     vmovups ymm10, [r13 + Q_16FLT_STRIDE]     ; ymm10 = q[16:23]
     vmovups ymm11, [r13 + Q_24FLT_STRIDE]     ; ymm11 = q[24:31]
     vmovups ymm12, [r13 + Q_32FLT_STRIDE]     ; ymm12 = q[32:39]
     vmovups ymm13, [r13 + Q_40FLT_STRIDE]     ; ymm13 = q[40:47], 8 floats, full 256-bit load
     align 32                                  ; 32-byte alignment for hot timestep loop
.t_loop:
; Prefetch next key/val row, one stride ahead
     ds prefetcht0 [r11 + rsi]                 ; Prefetch key row for t+1, slot 39, type-patchable only
     PREFETCH_SLOT 39                          ; Slot 39: attention key row, t+1
     SINGLE_HEAD_DOT _sh                       ; compute score = q?k[t] * scale, includes the moved slot-40 value prefetch after the first dot block
; === Branch on s > m0, cold path <5% using xmm0 directly ===
     vucomiss xmm0, xmm14                      ; compare s with m0, quiet: no #IA on NaN, same branch semantics
     ja .s_gt_m0                               ; cold path, <5% : s > m0, out-of-line block below the loop
; === HOT PATH: s <= m0, compute new_scale = exp,s-m0, old_scale = 1.0 implicit ===
; falls straight through into .t_advance, zero jumps on the >=95% path
.s_le_m0:
     vsubss xmm0, xmm0, xmm14                  ; xmm0 = s - m0, delta <= 0
     INLINE_EXP                                ; xmm0 = exp_delta = new_scale
     vaddss xmm15, xmm15, xmm0                 ; d0 += new_scale
     SINGLE_HEAD_WSUM_HOT _sh, xmm0            ; xb += new_scale * v, pass register directly
.t_advance:
     add r11, rsi                              ; Advance key pointer by kv_dim_bytes
     add r10, rsi                              ; Advance value pointer by kv_dim_bytes
     dec edx                                   ; one timestep consumed
     jne .t_loop                               ; more timesteps remaining?, countdown reaches 0 only after the last one
     inc r8d                                   ; phase advance: 0->1->2 loop back; 3+ falls through
     cmp r8d, 3                                ; another streaming phase pending?
     jb .t_phase_load                          ; next phase loader, AFTER the init, m0/d0 persist
; fall through to .t_done, zero unconditional jumps on the hot exit path
.t_done:
; === FINAL NORMALIZE: xb /= d0, d0 is already float ===
     NORMALIZE_HEAD _sh, xmm15                 ; xb /= d0, finalize online softmax
; === Epilogue ===
     vmovdqu xmm8, stk_ymm8                    ; Restore non-volatile xmm8, lower 128b of hoisted q[0:7]
     vmovdqu xmm9, stk_ymm9                    ; Restore non-volatile xmm9, lower 128b of hoisted q[8:15]
     vmovdqu xmm10, stk_ymm10                  ; Restore non-volatile xmm10, lower 128b of hoisted q[16:23]
     vmovdqu xmm11, stk_ymm11                  ; Restore non-volatile xmm11, lower 128b of hoisted q[24:31]
     vmovdqu xmm12, stk_ymm12                  ; Restore non-volatile xmm12, lower 128b of hoisted q[32:39]
     vmovdqu xmm6, stk_xmm6                    ; Restore non-volatile xmm6, scale hoist register
     vmovdqu xmm13, stk_xmm13                  ; Restore non-volatile xmm13
     vmovdqu xmm14, stk_xmm14                  ; Restore non-volatile xmm14
     vmovdqu xmm15, stk_xmm15                  ; Restore non-volatile xmm15
     vmovdqu xmm7, stk_xmm7                    ; Restore non-volatile xmm7
     CLEANUP_YMM                               ; vzeroupper: clear upper 128 bits of YMM to avoid SSE-AVX penalty
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore saved regs and release frame
; === COLD PATH: s > m0, compute old_scale = exp,m0-s, new_scale = 1.0 ===
; out of line: sits after the epilogue ret; distance is irrelevant at <5% taken
.s_gt_m0:
     vmovss xmm7, xmm0                         ; park s in xmm7, register, no store-forward round trip
     vsubss xmm0, xmm14, xmm0                  ; xmm0 = m0 - s, delta <= 0
     INLINE_EXP                                ; xmm0 = exp_delta = old_scale
     vfmadd213ss xmm15, xmm0, [rel rpe_one]    ; d0 = old_scale*d0 + 1.0, fused: single rounding vs mul+add double; 213 form takes the memory addend as src3
     vmovss xmm14, xmm7                        ; m0 = s, reload parked value from register
     SINGLE_HEAD_WSUM _sh, xmm0                ; fused old*xb + V
     jmp .t_advance                            ; rejoin the advance block
%undef ARG5_LOFF_OFFSET                        ; file-local rbp offset of the 5th stack arg, loff
%undef ARG6_KV_DIM_OFFSET                      ; file-local rbp offset of the 6th stack arg, kv_dim
%undef ARG7_KV_MUL_OFFSET                      ; file-local rbp offset of the 7th stack arg, kv_mul
%undef HD_WSUM_LIMIT                           ; file-local head_dim-32 wsum/norm SIMD limit
%undef HD_DOT_LIMIT                            ; file-local head_dim-64 dot SIMD limit
%undef SIMD_CHUNK_FLOATS                       ; file-local floats per SIMD chunk, loop stride + trip-count fold
%undef TRIP_COUNT_BASE                         ; file-local countdown base, trip count = pos + 1
%undef KV_HALF_SCALE
%undef FLOAT_SCALE                             ; file-local bytes-per-float index scale
%undef Q_40FLT_STRIDE                          ; file-local 5x8-float block offset, STEP_DOT_64 only
%undef Q_48FLT_STRIDE                          ; file-local 6x8-float block offset, STEP_DOT_64 only
%undef Q_56FLT_STRIDE                          ; file-local 7x8-float block offset, STEP_DOT_64 only
%undef K_8FLT_STRIDE                           ; file-local 1x8-half FP16 block offset
%undef K_16FLT_STRIDE                          ; file-local 2x8-half FP16 block offset
%undef K_24FLT_STRIDE                          ; file-local 3x8-half FP16 block offset
%undef K_32FLT_STRIDE                          ; file-local 4x8-half FP16 block offset
%undef K_40FLT_STRIDE                          ; file-local 5x8-half FP16 block offset
%undef K_48FLT_STRIDE                          ; file-local 6x8-half FP16 block offset
%undef K_56FLT_STRIDE                          ; file-local 7x8-half FP16 block offset
; Standalone Win64 packer follows the SMC attention kernel body.
%include "kv_f16c.inc"
