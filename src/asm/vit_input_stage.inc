; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%include "vit_structs.inc"
; Local constants for the nearest-neighbor sampler. The raw-RGB layout names
%define VIT_X3_SCALE 2                         ; LEA scale factor for x3, reg + reg*2 = 3*reg
%define VIT_ROW_BIAS 1                         ; last-row index bias, src_h - 1
%define VIT_XY_H_OFF 8                         ; target_h u64 offset inside __sys_vision_xy_res
%define VIT_SLACK_BYTES 16                     ; alloc slack: absorbs the final row's last-pixel dword-store spill
%define VIT_DDA_FRAC_BITS 16                   ; 16.16 fixed-point fractional bits: the DDA accumulator scale and the extract width at every use site

%macro VIT_SX3 0
     mov eax, r10d                             ; accum_x, register, not stack load
     shr eax, VIT_DDA_FRAC_BITS                ; sx = integer part, zero-extends rax
     lea rax, [rax + rax * VIT_X3_SCALE]       ; sx * 3, byte offset within source row, 64-bit
%endmacro
     section .rdata align=32
sz_sys_vit_xy_res:
     db "__sys_vision_xy_res", 0               ; microdb sys key: [u64 target_w, u64 target_h]
     section .bss
g_vis_target_w:
     resq 1                                    ; cached target width, 0 = not yet loaded, triggers lazy init
g_vis_target_h:
     resq 1                                    ; cached target height, 0 = not yet loaded, triggers lazy init
g_vis_scaled_buf:
     resq 1                                    ; BSS-owned rescaled RGB buffer, NULL until first mismatched-size call
     section .text
vit_input_fn:
     SF_START vit_input_fn, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles
     SF_QWORD vis_src                          ; const uint8_t* src_rgb, input
     SF_COMMIT                                 ; finalize stack frame layout
; --- Save input arguments ---
     mov vis_src, rcx                          ; save src_rgb pointer
     mov r14d, edx                             ; pin src_w in r14d
     mov r15d, r8d                             ; pin src_h in r15d
     TRACE_STR "vit_input_fn: entry"           ; debug: enter rescale pre-stage
     TRACE_REG "src", rcx                      ; debug: source RGB pointer
     TRACE_REG "sw", rdx                       ; debug: source width
     TRACE_REG "sh", r8                        ; debug: source height
     cmp qword [rel g_vis_target_w], 0         ; target resolution cached?
     jne .vis_target_ready                     ; yes, use cached BSS values
; First call: read "__sys_vision_xy_res" from microdb, FATAL on miss
     lea rcx, [rel sz_sys_vit_xy_res]          ; key = "__sys_vision_xy_res"
     xor edx, edx                              ; out_len = NULL
     call ht_find_or_fatal                     ; rax = data ptr, FATAL with key name on miss
; Blob layout: [u64 target_w, u64 target_h], two 8-byte little-endian integers
     mov rcx, [rax]                            ; target_w, first u64
     mov rdx, [rax + VIT_XY_H_OFF]             ; target_h, second u64
     mov [rel g_vis_target_w], rcx             ; cache target_w in BSS, persists across calls
     mov [rel g_vis_target_h], rdx             ; cache target_h in BSS
.vis_target_ready:
     mov ebx, [rel g_vis_target_w]             ; ebx = target width, pinned non-volatile: fast-path compare + DDA divisor + dx countdown are all 32-bit
     TRACE_REG "tw", rbx                       ; debug: target width, value live here
     mov edi, [rel g_vis_target_h]             ; pin target_h in edi, DWORD for the same reason: the fast-path compare, the DDA divisor and the dy countdown are all 32-bit
     TRACE_REG "th", rdi                       ; debug: target height, value live here
; --- Fast path: source already at target resolution, zero-copy ---
     cmp r14d, ebx                             ; src_w == target_w?
     jne .vis_needs_scale                      ; no, must rescale
     cmp r15d, edi                             ; src_h == target_h?
     jne .vis_needs_scale                      ; no, must rescale
; Source matches target: return src pointer directly, no copy, no alloc
     TRACE_STR "vit_input_fn: fast path (no rescale)" ; debug: src already target-sized
     mov rax, vis_src                          ; rax = original src_rgb
     jmp .vis_done                             ; skip allocation + scaling
.vis_needs_scale:
     TRACE_STR "vit_input_fn: needs_scale"     ; debug: src larger/smaller than target
; Dimensions are validated at the public border, public_api.inc PUB_VALIDATE_IMAGE:
; positive, w*h <= INT32_MAX, and per-dimension < 65536 for the 16.16 DDA below .
; This function is an internal consumer and trusts its caller.
; The cached target dimensions remain fixed for the lifetime of the buffer.
     cmp qword [rel g_vis_scaled_buf], 0       ; allocate only on the first size mismatch
     jne .vis_buf_ok                           ; reuse the buffer for this model
     mov eax, ebx                              ; target width
     imul rax, rdi                             ; target pixel count
     lea rcx, [rax + rax * VIT_X3_SCALE + VIT_SLACK_BYTES] ; RGB bytes plus final store slack
     call mdb_alloc_mem                        ; allocate the target-sized buffer, FATALs on failure
     mov [rel g_vis_scaled_buf], rax           ; publish the owning pointer
.vis_buf_ok:
     TRACE_REG "dst_buf", [rel g_vis_scaled_buf] ; debug: scaled RGB buffer pointer, read from BSS, a load into rax here would be dead, r13 gets the live copy at :192
     mov r12, vis_src                          ; r12 = src_rgb base, constant across loops
     mov r13, [rel g_vis_scaled_buf]           ; r13 = dst running pointer, BSS read; the only live load of this value, rax does not carry it on the :148 reuse path
; r14d = src_w / r15d = src_h / ebx = target_w: already pinned since entry, no reloads
; P2: Precompute DDA steps, 2 divisions outside the loops instead of 2 per pixel
; step_x =, src_w << 16 / target_w
     mov eax, r14d                             ; src_w
     shl eax, VIT_DDA_FRAC_BITS                ; src_w << 16, 32-bit shift: src_w < 2^16 per the guard above, result zero-extends
     xor edx, edx                              ; zero upper dividend
     div ebx                                   ; eax =, src_w << 16 / target_w = step_x, 32-bit form: dividend < 2^32, divisor zero-extended, saves REX.W
     mov r11d, eax                             ; r11d = step_x, cached in register, loop-invariant
; step_y =, src_h << 16 / target_h
     mov eax, r15d                             ; src_h
     shl eax, VIT_DDA_FRAC_BITS                ; src_h << 16, 32-bit shift: src_h < 2^16 per the guard above, result zero-extends
     xor edx, edx                              ; zero upper dividend
     div edi                                   ; eax =, src_h << 16 / target_h = step_y, divisor already in edi
     mov ecx, eax                              ; ecx = step_y, pinned in register across the loops
     xor edx, edx                              ; edx = accum_y = 0, 16.16 DDA y-accumulator, register-held
; edi = dy countdown = target_h, pinned since entry, never read as a value
.vis_dy_loop:
; P2: sy = accum_y >> 16, integer source row; do-while: target_h >= 1, a zero target faults at the div above
     mov eax, edx                              ; accum_y, register-held
     shr eax, VIT_DDA_FRAC_BITS                ; sy = integer part, 32-bit write zero-extends rax, full 64-bit sy
     mov esi, eax                              ; esi = sy, MUST be parked here: the imul below overwrites eax
     imul rax, r14                             ; rax = sy * src_w, row offset in pixels
     lea r9, [rax + rax * VIT_X3_SCALE]        ; sy * src_w * 3 = source row byte offset, 64-bit
     add r9, r12                               ; r9 = src base + row offset = source row base pointer, offset folded, inner loop drops an add
     xor r10d, r10d                            ; r10d = accum_x = 0, register, not stack store
     lea eax, [r15 - VIT_ROW_BIAS]             ; eax = src_h - 1, last source row index
     sub eax, esi                              ; eax =, src_h-1 - sy; zero iff this is the FINAL SOURCE ROW
     jz .vis_last_row_bytes                    ; yes, byte loop for the whole row
     mov esi, ebx                              ; esi = pair countdown = target_w/2, never read as a value
     shr esi, 1                                ; esi = number of 2-pixel pairs
     jz .vis_dx_odd                            ; target_w <= 1: only the odd tail runs
.vis_dx_loop:
     VIT_SX3                                   ; pixel A: source byte offset
     mov r8d, dword [r9 + rax]                 ; load R+G+B + next-R, spill byte, overwritten below
     mov dword [r13], r8d                      ; store R+G+B + spill byte, forward-overwritten
     add r10d, r11d                            ; accum_x += step_x, register add, not stack RMW
; pixel B of the pair, consumes the NEXT accum value; same dword method
     VIT_SX3                                   ; pixel B: source byte offset
     mov r8d, dword [r9 + rax]                 ; load R+G+B, +spill: next-row R for row-last B, or dead slack
     mov dword [r13 + VIT_RAW_RGB_CHANS], r8d  ; store B at dst+3, pair-fused
     add r10d, r11d                            ; accum_x += step_x
     add r13, 2 * VIT_RAW_RGB_CHANS            ; advance dst by both pixels in one add
     dec esi                                   ; pair countdown, 2 bytes, sets ZF
     jnz .vis_dx_loop                          ; next pixel pair
.vis_dx_odd:
     test bl, 1                                ; target_w odd?, bl = low byte of ebx = target_w; test writes no register
     jz .vis_dx_done                           ; even, all pixels already emitted by the pairs
     VIT_SX3                                   ; odd tail: source byte offset
     mov r8d, dword [r9 + rax]                 ; load R+G+B, +1 byte into the next source row, in bounds, see above
     mov dword [r13], r8d                      ; store R+G+B, +1 byte past the pixel, overwritten or slack
     add r13, VIT_RAW_RGB_CHANS                ; advance dst pointer by 3 bytes
     jmp .vis_dx_done                          ; row complete, r10d is reset at the next row head, :126, so no accum advance is needed here
.vis_last_row_bytes:
     mov esi, ebx                              ; esi = last-row pixel countdown = target_w
.vlr_pixel:
     VIT_SX3                                   ; last source row: source byte offset
     mov r8w, word [r9 + rax]                  ; load src R+G, 16-bit, never crosses the buffer end
     mov [r13], r8w                            ; store R+G to dst
     mov r8b, byte [r9 + rax + VIT_RAW_B_OFF]  ; load src B byte, last byte of the buffer, in bounds
     mov [r13 + VIT_RAW_B_OFF], r8b            ; store B to dst
     add r13, VIT_RAW_RGB_CHANS                ; advance dst pointer by 3 bytes
     add r10d, r11d                            ; accum_x += step_x
     dec esi                                   ; pixel countdown
     jnz .vlr_pixel                            ; next pixel
.vis_dx_done:
     add edx, ecx                              ; accum_y += step_y, reg-reg add, ecx holds the DDA y-step
     dec edi                                   ; dy countdown, 1 instruction vs inc+cmp+slot-load
     jnz .vis_dy_loop                          ; next destination row
.vis_scale_done:
; Return the scaled buffer pointer, NOT the source pointer
     mov rax, [rel g_vis_scaled_buf]           ; rax = scaled RGB buffer, BSS read
.vis_done:
     TRACE_STR "vit_input_fn: done"            ; debug: rescale completed
     TRACE_REG "ret", rax                      ; debug: returned RGB buffer pointer
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore non-volatiles, release frame, ret
; ---- %undef cleanup: keep the preprocessor namespace clean for later includes ----
%undef vis_src
%undef vis_src_raw
%unmacro VIT_SX3 0                             ; scale*3 helper macro, uses above
%undef VIT_DDA_FRAC_BITS
%undef VIT_X3_SCALE
%undef VIT_XY_H_OFF
%undef VIT_SLACK_BYTES
