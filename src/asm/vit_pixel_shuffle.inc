; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%include "vit_structs.inc"
%include "inf_structs.inc"
%define VPS_CALL7_FRAME 64                     ; 32 shadow + 24 stack args + 8 pad = 64, 16-aligned
%define VPS_GRID_BIAS 1                        ; grid_side - 1: the row-correction factor strips minus one
%define VPS_CALL_ARG5 32                       ; byte offset of 5th arg from rsp in call frame
%define VPS_CALL_ARG6 40                       ; byte offset of 6th arg from rsp
%define VPS_CALL_ARG7 48                       ; byte offset of 7th arg from rsp
     global vit_pixel_shuffle                  ; export: SmolVLM pixel-shuffle projector, phase 1 gather + phase 2 Linear
vit_pixel_shuffle:
     SF_START vit_pixel_shuffle, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles, reserve frame
     SF_QWORD vps_out                          ; final output buffer, caller-provided, not freed here
     SF_QWORD vps_proj_w                       ; Linear weight base, proj_blob + 8-byte header
     SF_COMMIT                                 ; finalize stack frame layout
     mov r12, rdx                              ; r12 = vit_out base, constant across all tokens; pipeline-allocated, never NULL
     mov vps_out, rcx                          ; save output base, pre-allocated by caller
; r8, proj_blob : compute Linear weight base now, skip 8-byte VITP header
     lea rax, [r8 + VIT_PROJ_HDR_BYTES]        ; rax = proj_blob + 8 = &W[0][0]
     mov vps_proj_w, rax                       ; save Linear weight base for Phase 2
; The ViT forward has finished: its persistent packed scratch is available.
; PS_N_TOKENS*PS_FLAT_DIM = PS_GRID_SIDE^2*HIDDEN <= NUM_POS*HIDDEN.
     mov r13, [rel g_vit_ptr_packed]           ; reuse allocated scratch, no per-image heap allocation
; === Cache remaining gather-loop constants in non-volatile registers ===
; NOTE: r12 holds the input base; the scratch pointer load does not alter it.
     mov r14, qword [rel g_vit_cfg + VCFG_PS_GRID_SIDE] ; r14 = grid_side = 32, patches per side
     mov r15, qword [rel g_vit_cfg + VCFG_PS_OUT_W] ; r15 = out_w = grid_side / F, precomputed by the DB builder
; no tok register: rdi is seeded once before .vps_oh_loop and advances by
; flat_dim*4 per token, which embodies tok = oh*out_w+ow implicitly
; r9 is a setup temp: holds vision_hidden for the strip/gap/CORR_ROW derivation,
; then becomes CORR_ROW below. CORR_ROW survives the gather loops -- rep movsb
; clobbers only rcx/rsi/rdi.
     mov r9, qword [rel g_vit_cfg + VCFG_HIDDEN] ; r9 = 768, vision_hidden, 64-bit
     mov r8, qword [rel g_vit_cfg + VCFG_PS_FACTOR] ; r8 = F, the per-token strip count, hoisted for the whole gather (strip/gap/CORR_ROW math)
; Strip-copy constants, hoisted once for the whole gather, both survive
; the token/strip loops: nothing in them writes rax/rdx :
     mov rax, r9                               ; rax = vision_hidden
     imul rax, r8                              ; rax = hidden * F = one strip's floats
     shl rax, 2                                ; rax = strip byte count, ERMS rep movsb count
     mov rdx, r14                              ; rdx = grid_side
     sub rdx, r8                               ; rdx = grid_side - F, patch gap between strips
     imul rdx, r9                              ; rdx = gap * hidden = gap floats per strip row
     shl rdx, 2                                ; rdx = gap byte advance, added after each strip copy
; Per-token source-base carry: rsi advances F*grid_side*hidden*4 bytes per
; token, F strips, each strip+gap = grid_side*hidden*4, so the next token's
; strip-0 source is rsi minus one correction constant, no per-token recompute.
     mov rsi, r12                              ; strip-0 source of token, 0,0 = vit_out + 0, p0 = 0
     lea r9, [r14 - VPS_GRID_BIAS]             ; grid_side - 1, r9's old vision_hidden is dead since the gap product at :43
     imul r9, rax                              ; CORR_ROW = (grid_side-1) * strip bytes, same low-64 product, ow -> ow+1 within a row
     mov r12, r15                              ; r12 = out_w
     sub r12, r14                              ; out_w - grid_side
     imul r12, rax                             ; CORR_WRAP_DELTA = (out_w-grid_side)*F*hidden*4, applied on top of CORR_ROW
     xor ebx, ebx                              ; oh = 0
     mov rdi, r13                              ; rdi = destination cursor, seeded ONCE for the whole gather: each token advances it by F strips x, F*hidden*4 bytes = flat_dim*4, and tok = oh*out_w+ow advances by exactly 1 per token, including across the ow wrap, where tok goes out_w-1 -> out_w, so rdi is always r13 + tok*flat_dim*4
.vps_oh_loop:
     xor r14d, r14d                            ; ow = 0 (r14 free: grid_side was consumed by the CORR_ROW product above)
.vps_ow_loop:
     mov r10, r8                               ; strip countdown seed = F
.vps_ih_loop:
     mov rcx, rax                              ; rcx = strip byte count, 64-bit, ERMS rep movsb
     rep movsb                                 ; copy the whole strip, rsi/rdi auto-advance
     add rsi, rdx                              ; skip, grid-F *hidden*4 gap to the next ih strip, dead after last strip, harmless
     dec r10                                   ; one strip less to go
     jnz .vps_ih_loop                          ; next strip
; ow/oh cascade advance, bottom tests
     inc r14d                                  ; ow++
     sub rsi, r9                               ; CORR_ROW on every token, including the last column
     cmp r14d, r15d                            ; ow < out_w?
     jl .vps_ow_loop                           ; hot path: one backward branch, no forward+jump pair
.vps_row_wrap:
     sub rsi, r12                              ; extra wrap correction restores the original next-row source
     inc ebx                                   ; oh++
     cmp rbx, r15                              ; oh < out_h?, qword compare against out_w (out_h == out_w, square grid), once per oh, cold vs the ih loop
     jl .vps_oh_loop                           ; next row
; fall through: all tokens gathered
.vps_gather_done:
     vzeroupper                                ; clear the upper YMM halves before the call
     sub rsp, VPS_CALL7_FRAME                  ; 7-arg call frame, 64 bytes: 32 shadow + 24 stack + 8 pad
     mov rcx, vps_out                          ; arg1: out, final output, caller-provided
     mov rdx, r13                              ; arg2: in = scratch, gathered pixel-unshuffle data
     mov r8, vps_proj_w                        ; arg3: w = Linear weight [text_hidden, flat_dim]
     xor r9d, r9d                              ; arg4: bias = NULL, SmolVLM projector has NO bias
     mov eax, dword [rel g_vit_cfg + VCFG_PS_N_TOKENS] ; arg5: n_rows = n_tokens (callee reads dwords via VK_ARG5)
     mov dword [rsp + VPS_CALL_ARG5], eax      ; int API
     mov eax, dword [rel g_vit_cfg + VCFG_PROJ_OUT_DIM] ; arg6: M = text_hidden (callee reads dwords via VK_ARG6)
     mov dword [rsp + VPS_CALL_ARG6], eax      ; int API, output columns
     mov eax, dword [rel g_vit_cfg + VCFG_PS_FLAT_DIM] ; arg7: K = flat_dim (callee reads dwords via VK_ARG7)
     mov dword [rsp + VPS_CALL_ARG7], eax      ; int API, inner dimension
     call v7d_vit_linear                       ; out = W @ scratch, dispatched across worker pool
     add rsp, VPS_CALL7_FRAME                  ; restore stack after 7-arg call
; Persistent ViT scratch belongs to g_vit_scratch_block; freed by inference cleanup.
.vps_done:
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore non-volatiles, release frame, return
%undef VPS_CALL_ARG7                           ; file-local 7th stack-arg offset
%undef VPS_CALL_ARG6                           ; file-local 6th stack-arg offset
%undef VPS_CALL_ARG5                           ; file-local 5th stack-arg offset
%undef VPS_CALL7_FRAME                         ; file-local 7-arg call frame size
