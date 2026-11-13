; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%include "vit_structs.inc"
%include "inf_structs.inc"
%define BC_RGB_CHANS 3                         ; bytes per pixel, packed RGB
%define BC_SARG5_OFF 48                        ; 5th arg at rbp+48 after SF_START
%define BC_SARG6_OFF 56                        ; 6th arg at rbp+56 after SF_START
%define BC_DW3 2                               ; LEA scale for *3, reg + reg*2
%define BC_FLOAT_SCALE 4                       ; sizeof,float for indexed weight loads
%define BC_QWORD_SCALE 8                       ; sizeof,qword for indexed rb[] loads, 64-bit row bases
%define BC_CHAN_B 2                            ; B channel byte offset within pixel
%define BC_TAP_COUNT 4                         ; y taps per pixel, j=0..3 : wy[j] dwords + rb[j] qwords
%define BC_TAP_DELTA_0 -1                      ; y-tap j=0 offset from y0, y0-1
%define BC_CLAMP_MINUS1 1                      ; subtract-one fold for hoisted clamp bounds, LEA displacements must be named
     section .rdata align=32
bc_half:
     times 4 dd 0.5                            ; 0.5f, half-pixel centering + rounding
bc_255:
     times 4 dd 255.0                          ; 255.0f, upper clamp
bc_poly_a:
     dd -0.5, 1.5, -1.5, 0.5                   ; Catmull-Rom weights, first Horner coefficient per tap
bc_poly_b:
     dd 1.0, -2.5, 2.0, -0.5                   ; second coefficient per tap
bc_poly_c:
     dd -0.5, 0.0, 0.5, 0.0                    ; third coefficient per tap
bc_poly_d:
     dd 0.0, 1.0, 0.0, 0.0                     ; constant term, tap 1 passes through at f=0
bc_x_tap_delta:
     dd -1, 0, 1, 2                            ; x-tap deltas from x0, BC_TAP_DELTA_0..3 as a dword vector

%macro BC_POLY4 2                              ; %1: fractional XMM; %2: contiguous w0..w3 frame slot (raw alias)
     vmovups xmm1, [rel bc_poly_a]             ; independent Horner seed load first
     vbroadcastss xmm0, %1                     ; all four fractional coordinates
     vfmadd213ps xmm1, xmm0, [rel bc_poly_b]   ; A*x + B
     vfmadd213ps xmm1, xmm0, [rel bc_poly_c]   ; (A*x + B)*x + C
     vfmadd213ps xmm0, xmm1, [rel bc_poly_d]   ; ((A*x + B)*x + C)*x + D
     vmovups %2, xmm0                          ; contiguous weights w0..w3
%endmacro

%macro BC_RGB_TAP 3                            ; clamped x-offset stack slot, tap lane selection imm8, accumulator XMM
     mov edx, %1                               ; byte offset (zero-extends rdx)
     vpermilps xmm0, xmm5, %2                  ; replicate one of four independently rounded tap products
     movzx eax, word [r9 + rdx]                ; R | (G << 8)
     vmovd xmm1, eax                           ; bytes [R,G,0,0]
     vpinsrb xmm1, xmm1, byte [r9 + rdx + BC_CHAN_B], 2 ; insert B into lane 2
     vpmovzxbd xmm1, xmm1                      ; int32 lanes [R,G,B,0]
     vcvtdq2ps xmm1, xmm1                      ; convert 0..255 exactly to FP32
     vfmadd231ps %3, xmm0, xmm1                ; one independent FMA per R/G/B lane into this tap's accumulator
%endmacro
     section .text
     global v7s_bicubic_resize                 ; export the bicubic resize entry
v7s_bicubic_resize:
     xor r10d, r10d                            ; public entry writes contiguous rows
     SF_START v7s_bicubic_resize_strided, rbx, rsi, rdi, r12, r13, r14, r15 ; save non-volatiles
     SF_DWORD bc_wy3                           ; y weight for tap +2, DECLARED FIRST = highest address
     SF_DWORD bc_wy2                           ; y weight for tap +1
     SF_DWORD bc_wy1                           ; y weight for tap 0
     SF_DWORD bc_wy0                           ; y weight for tap -1, DECLARED LAST = lowest address = wy[] array base
     SF_QWORD bc_rb3                           ; y row byte base for tap +2
     SF_QWORD bc_rb2                           ; y row byte base for tap +1
     SF_QWORD bc_rb1                           ; y row byte base for tap 0
     SF_QWORD bc_rb0                           ; y row byte base for tap -1, DECLARED LAST = lowest address = rb[] array base
     SF_DWORD bc_xo3                           ; clamped x byte offset for tap +2, DECLARED FIRST = highest address
     SF_DWORD bc_xo2                           ; clamped x byte offset for tap +1
     SF_DWORD bc_xo1                           ; clamped x byte offset for tap 0
     SF_DWORD bc_xo0                           ; clamped x byte offset for tap -1, DECLARED LAST = lowest address = xo[] array base
     SF_DWORD bc_y_scale                       ; src_h / dst_h, float
     SF_DWORD bc_clamp_y                       ; src_h - 1, hoisted y clamp bound
     SF_DWORD bc_row_skip                      ; extra destination bytes between fitted rows
     SF_DWORD bc_dst_h                         ; destination height
     SF_XMM bc_xmm6                            ; xmm6 low spill, positive-lobe x-tap accumulator
     SF_XMM bc_xmm7                            ; xmm7 low spill, positive-lobe x-tap accumulator
     SF_XMM bc_xmm8                            ; xmm8 low spill, horizontal weights held across all four y taps
     SF_COMMIT                                 ; commit stack frame layout
     mov bc_row_skip, r10d                     ; retain the private row skip through filter setup
     vmovdqu bc_xmm6, xmm6                     ; save callee-saved xmm6 low
     vmovdqu bc_xmm7, xmm7                     ; save callee-saved xmm7 low
     vmovdqu bc_xmm8, xmm8                     ; save callee-saved xmm8 low
; Exact identity resize: avoid weight generation and per-pixel 16 taps.
     cmp edx, dword [rbp + BC_SARG5_OFF]       ; src_w == dst_w?
     jne .bc_nonidentity                       ; no: full filter path
     cmp r8d, dword [rbp + BC_SARG6_OFF]       ; src_h == dst_h?
     jne .bc_nonidentity                       ; no: full filter path
     test edx, edx                             ; positive dimensions required for the exactness bound
     jle .bc_nonidentity                       ; degenerate width: full path
     test r8d, r8d                             ; degenerate height: full path
     jle .bc_nonidentity                       ; full path
     cmp edx, 0x007FFFFF                       ; width within the exact half-pixel range?
     ja .bc_nonidentity                        ; no: full filter path
     cmp r8d, 0x007FFFFF                       ; height within the exact half-pixel range?
     ja .bc_nonidentity                        ; no: full filter path
     mov rsi, rcx                              ; rsi = src base for the byte copy
     mov rdi, r9                               ; rdi = dst base
     mov eax, edx                              ; eax = src_w, zero-extends
     test r10d, r10d                           ; does identity output have padding?
     jnz .bc_identity_rows                     ; preserve padding between copies
     imul rax, r8                              ; rax = src_w * src_h, pixels
     lea rcx, [rax + rax*BC_DW3]               ; rcx = src_w * src_h * 3 bytes
     rep movsb                                 ; identity copy, DF=0 from the Win64 entry ABI
     jmp .bc_dy_done                           ; original vzeroupper + SF_END
.bc_identity_rows:
     lea rax, [rax + rax*BC_DW3]               ; fitted row size in bytes
.bc_identity_row:
     mov rcx, rax                              ; bytes to copy from the contiguous input row
     rep movsb                                 ; advance both cursors by the fitted row size
     add rdi, r10                              ; skip the canvas padding
     dec r8d                                   ; one fewer source row
     jnz .bc_identity_row                      ; copy the next source row
     jmp .bc_dy_done                           ; restore the frame
.bc_nonidentity:
; --- Save register args ---
     mov r12, rcx                              ; r12 = src base, constant
     mov r14d, edx                             ; r14d = src_w, constant, in non-volatile reg
     mov r15d, r8d                             ; r15d = src_h, constant, in non-volatile reg
     mov r13, r9                               ; r13 = dst cursor, walked by add BC_RGB_CHANS per pixel; dx loop lands exactly on the next row start
; --- Load stack args, 5th at rbp+48, 6th at rbp+56 ---
     mov eax, dword [rbp + BC_SARG5_OFF]       ; dst_w
     mov ebx, eax                              ; rbx = dst_w, constant, in non-volatile
     mov eax, dword [rbp + BC_SARG6_OFF]       ; dst_h
     mov bc_dst_h, eax                         ; save dst_h to stack slot
; --- Precompute scale factors, stored in stack slots, not XMM ---
     vcvtsi2ss xmm0, xmm0, r14d                ; float src_w
     vcvtsi2ss xmm1, xmm1, ebx                 ; float dst_w
     vdivss xmm0, xmm0, xmm1                   ; x_scale = src_w / dst_w
     vmovaps xmm4, xmm0                        ; retain x_scale in volatile XMM4 for the whole pixel loop
     vcvtsi2ss xmm0, xmm0, r15d                ; float src_h
     vcvtsi2ss xmm1, xmm1, eax                 ; float dst_h
     vdivss xmm0, xmm0, xmm1                   ; y_scale = src_h / dst_h
     vmovss bc_y_scale, xmm0                   ; save to stack
     lea eax, [r15 - BC_CLAMP_MINUS1]          ; eax = src_h - 1, same fold, from r15 BEFORE it is repurposed below
     mov bc_clamp_y, eax                       ; hoisted y clamp bound
     lea r15d, [r14 - BC_CLAMP_MINUS1]         ; r15d = src_w - 1, x clamp bound, hot-loop register; r15's old job, src_h, is consumed above and never read again
; the clamp bounds are recomputed per tap, not hoisted as floats, because
; PIL floors the raw coordinate and clamps only taps
     lea r14, [r14 + r14*BC_DW3]               ; source row stride in bytes
     lea r10, bc_wy0_raw                       ; r10 = &wy[0], frame-fixed base for indexed tap loads
     lea rcx, bc_rb0_raw                       ; rcx = &rb[0], frame-fixed base for indexed row-byte-base loads
     xor edi, edi                              ; dy = 0
     cmp edi, bc_dst_h                         ; dy < dst_h?
     jae .bc_dy_done                           ; no, empty output, done
     test ebx, ebx                             ; dst_w == 0?, hoisted: call-constant, would otherwise be re-tested every row
     jz .bc_dy_done                            ; no pixels in ANY row, done
.bc_dy_loop:
     vcvtsi2ss xmm0, xmm0, edi                 ; float dy
     vaddss xmm0, xmm0, [rel bc_half]          ; dy + 0.5
     vmovss xmm3, bc_y_scale                   ; row-local y_scale
     vfmsub213ss xmm0, xmm3, [rel bc_half]     ; sy = (dy+0.5)*y_scale - 0.5, fused
     vroundss xmm1, xmm0, xmm0, 1              ; floor,sy, imm 1 = round toward -inf, truncation would be WRONG for negative sy ; xmm1 doubles as, float y0
     vcvttss2si eax, xmm1                      ; y0 = floor,sy, can be -1: the tap clamp below handles it exactly like PIL
     vsubss xmm0, xmm0, xmm1                   ; xmm0 = fy = sy - floor,sy in [0,1 ALWAYS, floor semantics guarantee it for negative sy too
     BC_POLY4 xmm0, bc_wy0_raw                 ; generate all four row weights at once
     lea esi, [rax + BC_TAP_DELTA_0]           ; first source row before clamping
     xor r11d, r11d                            ; row tap index
     xor r9d, r9d                              ; lower clamp bound
.bc_row_bases:
     mov eax, esi                              ; current unclamped source row
     test eax, eax                             ; below the first row?
     cmovs eax, r9d                            ; clamp to row zero
     cmp eax, bc_clamp_y                       ; above the final row?
     cmovg eax, bc_clamp_y                     ; clamp to the last row
     imul rax, r14                             ; clamped source row byte offset
     add rax, r12                              ; absolute source row pointer
     mov [rcx + r11*BC_QWORD_SCALE], rax       ; keep the absolute tap base for every pixel
     inc esi                                   ; next unclamped row
     inc r11d                                  ; next row tap
     cmp r11d, BC_TAP_COUNT                    ; all tap bases ready?
     jb .bc_row_bases                          ; compute the next one
     xor esi, esi                              ; dx = 0, dst_w > 0 guaranteed by the hoisted test
.bc_dx_loop:
     vcvtsi2ss xmm0, xmm0, esi                 ; float dx
     vaddss xmm0, xmm0, [rel bc_half]          ; dx + 0.5
     vfmsub213ss xmm0, xmm4, [rel bc_half]     ; sx = (dx+0.5)*x_scale - 0.5, fused
     vroundss xmm1, xmm0, xmm0, 1              ; floor,sx, imm 1 = round toward -inf; xmm1 doubles as, float x0
     vcvttss2si eax, xmm1                      ; x0 = floor,sx, can be -1: the SIMD clamp handles it exactly like PIL, x0 stays in eax through the phase
     vmovd xmm2, eax                           ; x0 into lane 0
     vpbroadcastd xmm2, xmm2                   ; [x0,x0,x0,x0]
     vsubss xmm0, xmm0, xmm1                   ; xmm0 = fx = sx - floor,sx in [0,1 ALWAYS
     vpaddd xmm2, xmm2, [rel bc_x_tap_delta]   ; x0 + {-1,0,1,2}, same 32-bit wrap as the four lea
     vmovups xmm1, [rel bc_poly_a]             ; BC_POLY4 body: independent Horner seed load first
     vpxor xmm3, xmm3, xmm3                    ; zero clamp bound
     vbroadcastss xmm8, xmm0                   ; broadcast the fractional x coordinate into all four lanes
     vpmaxsd xmm2, xmm2, xmm3                  ; clamp lower to 0, signed
     vfmadd213ps xmm1, xmm8, [rel bc_poly_b]   ; Horner stage 1
     vmovd xmm3, r15d                          ; src_w-1, the original clamp high bound register
     vpbroadcastd xmm3, xmm3                   ; [src_w-1] x4
     vfmadd213ps xmm1, xmm8, [rel bc_poly_c]   ; Horner stage 2
     vpminsd xmm2, xmm2, xmm3                  ; clamp upper to src_w-1, signed
     vpslld xmm3, xmm2, 1                      ; 2*clamped_x
     vfmadd213ps xmm8, xmm1, [rel bc_poly_d]   ; horizontal weights for all four x taps, kept live across the y taps
     vpaddd xmm2, xmm2, xmm3                   ; 3*clamped_x, one store below replaces four
     vmovdqu bc_xo0_raw, xmm2                  ; store xo[0..3], four consecutive dword slots in one 16-byte store
; --- Zero four per-tap x accumulators, R/G/B lanes packed ---
     vxorps xmm2, xmm2, xmm2                   ; x tap 0, small negative lobe
     vxorps xmm3, xmm3, xmm3                   ; x tap 3, small negative lobe
     vxorps xmm6, xmm6, xmm6                   ; x tap 2, positive lobe
     vxorps xmm7, xmm7, xmm7                   ; x tap 1, positive lobe
     xor r11d, r11d                            ; j = 0, y tap index
.bc_tap_y:
     vbroadcastss xmm5, [r10 + r11*BC_FLOAT_SCALE] ; duplicate scalar wy[j] into four lanes
     vmulps xmm5, xmm5, xmm8                   ; four wx[i] * wy[j] products, one correctly-rounded FP32 multiply per lane
     mov r9, [rcx + r11*BC_QWORD_SCALE]        ; absolute source row base
     BC_RGB_TAP bc_xo0, 0x00, xmm2             ; x tap i=0, wx[0] lane, negative lobe
     BC_RGB_TAP bc_xo3, 0xFF, xmm3             ; x tap i=3, wx[3] lane, negative lobe
     BC_RGB_TAP bc_xo2, 0xAA, xmm6             ; x tap i=2, wx[2] lane, positive lobe
     BC_RGB_TAP bc_xo1, 0x55, xmm7             ; x tap i=1, wx[1] lane, positive lobe
     inc r11d                                  ; next y tap
     cmp r11d, BC_TAP_COUNT                    ; all 4 y taps done?
     jl .bc_tap_y                              ; no: next y tap, bottom-tested
.bc_tap_y_done:
     vaddps xmm2, xmm2, xmm3                   ; small / negative taps 0+3
     vaddps xmm6, xmm6, xmm7                   ; positive taps 2+1
     vaddps xmm2, xmm2, xmm6                   ; final RGB sum
; --- Round, clamp, and store 3 channel bytes, packed lanes in xmm2 ---
     vxorps xmm3, xmm3, xmm3                   ; +0.0f in every lane
     vmaxps xmm2, xmm2, xmm3                   ; clamp each RGB lane to 0
     vminps xmm2, xmm2, [rel bc_255]           ; clamp to 255
     vaddps xmm2, xmm2, [rel bc_half]          ; +0.5, same per-channel order
     vcvttps2dq xmm2, xmm2                     ; truncate to int32
     vpackssdw xmm2, xmm2, xmm2                ; int32 -> int16, values already 0..255 so identity
     vpackuswb xmm2, xmm2, xmm2                ; int16 -> uint8
     vpextrw word [r13], xmm2, 0               ; write R,G directly from the packed result
     vpextrb byte [r13 + BC_CHAN_B], xmm2, 2   ; write B, lane 2 only
; --- Advance dst cursor ---
     add r13, BC_RGB_CHANS                     ; dst += 3
     inc esi                                   ; dx++
     cmp esi, ebx                              ; dx < dst_w?
     jb .bc_dx_loop                            ; next pixel in row, do-while: bottom test
.bc_dx_done:
     mov eax, bc_row_skip                      ; extra bytes beyond the filtered row
     add r13, rax                              ; reach the next fitted row start
     inc edi                                   ; dy++
     cmp edi, bc_dst_h                         ; dy < dst_h?
     jb .bc_dy_loop                            ; next output row, do-while: bottom test
.bc_dy_done:
     vmovdqu xmm6, bc_xmm6                     ; restore callee-saved xmm6 low
     vmovdqu xmm7, bc_xmm7                     ; restore callee-saved xmm7 low
     vmovdqu xmm8, bc_xmm8                     ; restore callee-saved xmm8 low
     vzeroupper                                ; clean AVX state before return
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore non-volatiles and return
%undef bc_dst_h
%undef bc_dst_h_raw
%undef bc_clamp_y
%undef bc_clamp_y_raw
%undef bc_y_scale
%undef bc_y_scale_raw
%undef bc_row_skip
%undef bc_row_skip_raw
%undef bc_xo0
%undef bc_xo1
%undef bc_xo2
%undef bc_xo3
%undef bc_xo0_raw
%undef bc_xo1_raw
%undef bc_xo2_raw
%undef bc_xo3_raw
%undef bc_rb0
%undef bc_rb1
%undef bc_rb2
%undef bc_rb3
%undef bc_rb0_raw
%undef bc_rb1_raw
%undef bc_rb2_raw
%undef bc_rb3_raw
%undef bc_wy0
%undef bc_wy1
%undef bc_wy2
%undef bc_wy3
%undef bc_wy0_raw
%undef bc_wy1_raw
%undef bc_wy2_raw
%undef bc_wy3_raw
%unmacro BC_POLY4 2                            ; remove the packed 4-tap weight macro, file-local
%unmacro BC_CLAMP_X_OFFSET 3                   ; remove the x-offset clamp macro, file-local
%unmacro BC_RGB_TAP 3                          ; remove the packed-RGB tap macro, file-local
%undef bc_xmm6
%undef bc_xmm7
%undef bc_xmm8
%undef BC_RGB_CHANS
%undef BC_SARG5_OFF
%undef BC_SARG6_OFF
%undef BC_DW3
%undef BC_FLOAT_SCALE
%undef BC_QWORD_SCALE
%undef BC_CHAN_B
%undef BC_TAP_COUNT
%undef BC_CLAMP_MINUS1
%undef BC_TAP_DELTA_0
