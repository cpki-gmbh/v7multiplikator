; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     global vit_pos_embed                      ; export vit_pos_embed entry point
%define VPE_YMM32 32                           ; byte offset for second YMM group in interleaved pos_embed interp
%define VPE_TAP_NEXT 1                         ; h1/w1 candidate displacement: the neighbouring tap is current tap + 1
%define VPE_DENOM_BIAS 1                       ; grid-denominator bias: gh-1 / gw-1
     section .bss
g_cached_pe_gh:
     resd 1                                    ; cached grid_h, 0 = never computed
g_cached_pe_gw:
     resd 1                                    ; cached grid_w
g_cached_pe_tbl:
     resq 1                                    ; cached table base, output depends on learned table
g_cached_pe_pos:
     resq 1                                    ; cached destination buffer identity, fresh alloc = recompute
     section .text
vit_pos_embed:
     SF_START vit_pos_embed, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles, reserve frame
     SF_QWORD vpe_n                            ; total patch count n = gh * gw, full 64-bit
     SF_DWORD vpe_gw_max                       ; exclusive bound for 2*block_col, the cascade wraps at gw
     SF_QWORD vpe_table_row_bytes              ; hoisted grid_side * hidden*4, one table row in BYTES, per-patch vertical tap AND mask
     SF_XMM vpe_xmm7                           ; callee-saved xmm7 spill, low 128, the 1.0 broadcast below writes the full ymm7
     SF_XMM vpe_xmm8                           ; spill slot: xmm8 low half, grid_side-1 scalar
     SF_XMM vpe_xmm9                           ; spill slot: xmm9 low half, gh-1 denominator
     SF_XMM vpe_xmm10                          ; spill slot: xmm10 low half, gw-1 denominator
     SF_COMMIT                                 ; finalize stack frame layout
     vmovdqu vpe_xmm7, xmm7                    ; save callee-saved xmm7, low half, restored at the epilogue
     vmovdqu vpe_xmm8, xmm8                    ; save XMM8, Win64 non-volatile low half
     vmovdqu vpe_xmm9, xmm9                    ; save XMM9
     vmovdqu vpe_xmm10, xmm10                  ; save XMM10
     cmp r8d, [rel g_cached_pe_gh]             ; same grid_h as last time?
     jne .vpe_compute                          ; no — must recompute
     cmp r9d, [rel g_cached_pe_gw]             ; same grid_w?
     jne .vpe_compute                          ; no — must recompute
     cmp rdx, [rel g_cached_pe_tbl]            ; same learned table as last time?
     jne .vpe_compute                          ; no — model switch: recompute
     cmp rcx, [rel g_cached_pe_pos]            ; same destination buffer, fresh alloc = recompute ?
     je .vpe_done                              ; cache hit — output buffer already has correct values
.vpe_compute:
     mov rbx, rdx                              ; table base cached in GPR, non-volatile, hoisted
     mov eax, r8d                              ; gh, zero-extends to rax
     imul rax, r9                              ; n = gh * gw, 64-bit
     mov vpe_n, rax                            ; save n, full 64-bit product, a dword store would truncate a >4G patch count
     mov vpe_gw_max, r9d                       ; exclusive bound for 2*block_col, the cascade needs >= 1 block column; grid geometry is the DB builder's isqrt contract: even, >= 2
     lea eax, [r8 - VPE_DENOM_BIAS]            ; gh-1, exact integer denominator
     vcvtsi2ss xmm9, xmm1, eax                 ; init saved nonvolatile xmm9 low lane directly, no frame temp
     lea eax, [r9 - VPE_DENOM_BIAS]            ; gw-1, exact integer denominator (gw already >= 2 via the blocks_w gate above)
     vcvtsi2ss xmm10, xmm1, eax                ; init saved nonvolatile xmm10 low lane directly
     vunpcklps xmm9, xmm9, xmm10               ; {gh-1, gw-1, ...}, loop-invariant denominator pair
     vshufps xmm9, xmm9, xmm9, 0x44            ; {gh-1, gw-1, gh-1, gw-1}, exact divisor lanes, built once per call
     vmovss xmm8, [rel g_vit_cfg + VCFG_GRID_MINUS1_F] ; grid_side-1, per-call invariant
     xor r15d, r15d                            ; p = 0, patch counter in non-volatile r15, 64-bit to match vpe_n
     mov r14, rcx                              ; running output ptr = pos base, in non-volatile r14
; Cascade counters live in r9d, 2*block_col and r10d, 2*block_row, pre-doubled
; registers; the loop-bottom advance mirrors the patch producer's own order
; Cache model constants in volatile regs, leaf function, r8-r11 free to use
     mov r11d, dword [rel g_vit_cfg + VCFG_HIDDEN] ; r11d = hidden, interp row width; multiple of 16 is the import's contract for the 2x8-wide interp groups
     mov [rel g_cached_pe_gh], r8d             ; publish cache key, grid_h, r8d still holds arg3 here
     mov [rel g_cached_pe_gw], r9d             ; publish cache key, grid_w, r9d still holds arg4 here
     mov [rel g_cached_pe_tbl], rdx            ; publish table identity, arg2 at entry
     mov [rel g_cached_pe_pos], rcx            ; publish buffer identity, arg1 at entry
     mov r8, qword [rel g_vit_cfg + VCFG_HIDDEN_F4] ; r8 = hidden * 4 bytes, DB-owned, folded load
     mov rax, [rel g_vit_cfg + VCFG_GRID_SIDE] ; grid_side, qword DB field
     imul rax, r8                              ; grid_side * hidden*4 = one table row in BYTES
     mov vpe_table_row_bytes, rax              ; park the row stride for the per-patch AND
     vbroadcastss ymm7, [rel vit_gelu_one]     ; ymm7 = 1.0, loop-invariant hoist: leaf fn, no calls, survives all patches; legitimate only because ymm7 is callee-saved and the SF_XMM spill in the prologue + the restore at .vpe_done preserve it
     xor r9d, r9d                              ; 2*block_col = 0, register, not a frame slot
     xor r10d, r10d                            ; 2*block_row = 0
.vpe_patch_loop:
     mov eax, r10d                             ; 2*block_row, pre-doubled in the register
     bt r15, 1                                 ; CF =, p >> 1 & 1 = in_row, p =, block_row*blocks_w + block_col *4 + in_row*2 + in_col, mod-4 bits are exact
     adc eax, 0                                ; row = 2*block_row + in_row
     mov edx, r9d                              ; 2*block_col, pre-doubled in the register
     bt r15, 0                                 ; CF = p & 1 = in_col
     adc edx, 0                                ; col = 2*block_col + in_col
; Both coordinate divides are independent: issue both early so the row
; chain does not serialize the column chain
     vcvtsi2ss xmm0, xmm7, eax                 ; float row, eax still holds row from the cascade combine, no stack round-trip
     vmulss xmm0, xmm0, xmm8                   ; row * cached (side-1), same multiply
     vcvtsi2ss xmm3, xmm7, edx                 ; float col
     vmulss xmm3, xmm3, xmm8                   ; col * cached (side-1)
     vinsertps xmm2, xmm0, xmm3, 0x04          ; {row, col, ...}, col into lane 1
     vdivps xmm2, xmm2, xmm9                   ; both divides, one instruction, same FP32 rounding, divisor hoisted out of the loop
     vmovaps xmm0, xmm2                        ; row quotient = lane 0
     vpermilps xmm3, xmm2, 0x55                ; col quotient = lane 1
     vroundss xmm2, xmm0, xmm0, 1              ; xmm2 = floor(hsrc)
     vroundss xmm1, xmm3, xmm3, 1              ; xmm1 = floor(wsrc)
     vsubss xmm4, xmm0, xmm2                   ; xmm4 = hfrac
     vsubss xmm3, xmm3, xmm1                   ; xmm3 = wfrac
     vcvttss2si ecx, xmm2                      ; ecx = h0
     vcvttss2si edi, xmm1                      ; edi = w0
     lea rax, [rcx + VPE_TAP_NEXT]             ; h0 + 1
     cmp eax, [rel g_vit_cfg + VCFG_GRID_MINUS1_I] ; > side-1?
     jle .vpe_h1_ok                            ; no, use h0+1
     mov eax, [rel g_vit_cfg + VCFG_GRID_MINUS1_I] ; clamp to side-1
.vpe_h1_ok:
     mov r12, rax                              ; r12 = h1
     lea rax, [rdi + VPE_TAP_NEXT]             ; w0 + 1
     cmp eax, [rel g_vit_cfg + VCFG_GRID_MINUS1_I] ; > side-1?
     jle .vpe_w1_ok                            ; no
     mov eax, [rel g_vit_cfg + VCFG_GRID_MINUS1_I] ; clamp
.vpe_w1_ok:
     mov r13, rax                              ; r13 = w1
     mov rax, rcx                              ; h0
     imul rax, [rel g_vit_cfg + VCFG_GRID_SIDE] ; h0 * side, 64-bit
     add rax, rdi                              ; + w0, 64-bit add, a 32-bit add would keep only the low 32 bits of the product
     imul rax, r11                             ; h0*side + w0 * hidden = element offset
     lea rsi, [rbx + rax*VK_F4]                ; rsi = t00 = table base + element_count * sizeof,float
; t01 = t00 +, w1-w0 *hidden, t10 = t00 +, h1-h0 *grid_side*hidden, t11 = t10 +, w1-w0 *hidden
; w1-w0 and h1-h0 are clamped to {0,1}: the offset is a 0 / all-ones mask ANDed
; with the byte stride, replacing three imuls with ands
     mov rax, rdi                              ; w0
     sub rax, r13                              ; w0-w1 = 0 or -1, r13 = w1 still live here
     and rax, r8                               ; 0 or hidden*4 bytes, col stride
     mov rdx, rcx                              ; h0
     sub rdx, r12                              ; h0-h1 = 0 or -1, r12 = h1 still live here
     and rdx, vpe_table_row_bytes              ; 0 or the exact row stride in BYTES
     lea rdi, [rsi + rax]                      ; rdi = t01 = t00 + col_stride
     lea r12, [rsi + rdx]                      ; r12 = t10 = t00 + row_stride
     lea r13, [r12 + rax]                      ; r13 = t11 = t10 + col_stride
     vbroadcastss ymm2, xmm4                   ; ymm2 = hfrac, register broadcast, AVX2
     vbroadcastss ymm3, xmm3                   ; ymm3 = wfrac, register broadcast, AVX2
     vsubps ymm4, ymm7, ymm2                   ; ymm4 = 1 - hfrac, 1.0 hoisted into ymm7
     vsubps ymm5, ymm7, ymm3                   ; ymm5 = 1 - wfrac
     vmulps ymm1, ymm4, ymm3                   ; ymm1 = w01 =, 1-hf *wf, wfrac still live
     vmulps ymm3, ymm2, ymm3                   ; ymm3 = w11 = hf*wf, wfrac dead
     vmulps ymm2, ymm2, ymm5                   ; ymm2 = w10 = hf*,1-wf, hfrac dead
     vmulps ymm0, ymm4, ymm5                   ; ymm0 = w00 =, 1-hf *,1-wf
; ymm0/ymm1/ymm2/ymm3 = w00/w01/w10/w11, exactly what the interp loop
; expects; ymm4/ymm5 are dead, overwritten as temps below
; rsi=t00, rdi=t01, r12=t10, r13=t11 are already set from the pointer computation above
; r14 already holds the running output pointer for this patch, persistent, non-volatile
     xor ecx, ecx                              ; d = 0, 64-bit register counter, matches the 64-bit hidden bound below
     align 32                                  ; hot loop head, back-edge target ; 13-instruction body entered by fall-through
.vpe_interp_loop2:
     vmulps ymm4, ymm0, [rsi + rcx*VK_F4]      ; w00 * t00[d..d+7]
     vmulps ymm5, ymm0, [rsi + rcx*VK_F4 + VPE_YMM32] ; w00 * t00+8, both independent chains start together
     vfmadd231ps ymm4, ymm1, [rdi + rcx*VK_F4] ; += w01 * t01
     vfmadd231ps ymm5, ymm1, [rdi + rcx*VK_F4 + VPE_YMM32] ; += w01 * t01+8
     vfmadd231ps ymm4, ymm2, [r12 + rcx*VK_F4] ; += w10 * t10
     vfmadd231ps ymm5, ymm2, [r12 + rcx*VK_F4 + VPE_YMM32] ; += w10 * t10+8
     vfmadd231ps ymm4, ymm3, [r13 + rcx*VK_F4] ; += w11 * t11
     vfmadd231ps ymm5, ymm3, [r13 + rcx*VK_F4 + VPE_YMM32] ; += w11 * t11+8
     vmovups [r14 + rcx*VK_F4], ymm4           ; store pos[d..d+7]
     vmovups [r14 + rcx*VK_F4 + VPE_YMM32], ymm5 ; store pos+8
     add rcx, 16                               ; d += 16
     cmp rcx, r11                              ; d < hidden?, 64-bit cached bound
     jl .vpe_interp_loop2                      ; continue
; Advance output pointer
     add r14, r8                               ; pos += hidden * 4 bytes, cached, persistent in r14
     inc r15                                   ; p++, non-volatile 64-bit counter
; Cascade advance: in_col fastest, then in_row, then bcol, then brow
; mirrors the producer's p decode: p%2,, p/2 %2,, p/4 %bw, p/,4*bw
     test r15b, 3                              ; p+1 % 4 == 0?, r15 was already incremented above; the low byte carries bits 1:0 exactly for ANY r15 because 4 divides 256, no bound on p is needed
     jne .vpe_casc_done                        ; no, 2*block_col / 2*block_row unchanged
     add r9d, 2                                ; 2*block_col += 2
     cmp r9d, vpe_gw_max                       ; reached the exclusive bound, gw ?
     jb .vpe_casc_done                         ; no, still inside this block row
     xor r9d, r9d                              ; wrap 2*block_col to 0
     add r10d, 2                               ; 2*block_row += 2, no wrap: p < n bounds it
.vpe_casc_done:
     cmp r15, vpe_n                            ; more patches?, loop-bottom test
     jb .vpe_patch_loop                        ; yes
.vpe_done:
     vzeroupper                                ; clean AVX state
     vmovdqu xmm7, vpe_xmm7                    ; restore callee-saved xmm7, low half only
     vmovdqu xmm8, vpe_xmm8                    ; restore XMM8
     vmovdqu xmm9, vpe_xmm9                    ; restore XMM9
     vmovdqu xmm10, vpe_xmm10                  ; restore XMM10
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore non-volatiles, return
; %undef cleanup: file-local defines leave scope here, house convention --
%undef VPE_YMM32
%undef VPE_TAP_NEXT
