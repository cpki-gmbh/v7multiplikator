; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; Named offset defines for struct field access
; ============================================================================
%include "inf_structs.inc"
; ============================================================================
; DLL-internal globals for rope table pointers, set via set_rope_tables
; ============================================================================
     section .text
; RCX: output FP32 row, EDX: token ID, R8D: dim, R9: q_tokens descriptor.
; The model dimension is a positive multiple of GS (64).
%define EMB_ROW_F4 4                           ; scale and output element size in bytes
v7s_embedding_row:
     ldmxcsr [rel g_inf_embed_mxcsr]           ; use the eager converter's original rounding and denormal mode
     imul rdx, r8                              ; first int8 element for this token
     mov r10, [r9 + QT_Q]                      ; quantized embedding base
     add r10, rdx                              ; quantized row start
     shr rdx, 6                                ; first scale group in the table
     mov r11, [r9 + QT_S]                      ; scale table base
     lea r11, [r11 + rdx*EMB_ROW_F4]           ; this row's first scale
     xor eax, eax                              ; element offset within the row
.row_group:
     vbroadcastss ymm1, [r11]                  ; one scale for the next 64 weights
%assign EMB_ROW_ELEM 0                         ; first vector within this group
%rep GS / 8
     vpmovsxbd ymm0, [r10 + rax + EMB_ROW_ELEM] ; eight signed weights to int32
     vcvtdq2ps ymm0, ymm0                      ; convert the integers exactly to FP32
     vmulps ymm0, ymm0, ymm1                   ; round weight * scale as in the eager table
     vmovups [rcx + rax*EMB_ROW_F4 + EMB_ROW_ELEM*EMB_ROW_F4], ymm0 ; write this output vector
%assign EMB_ROW_ELEM EMB_ROW_ELEM + 8          ; advance by one vector of weights
%endrep
%undef EMB_ROW_ELEM
     add eax, GS                               ; advance the output and input offsets
     add r11, 4                                ; advance to the next scale
     cmp eax, r8d                              ; reached the end of the row?
     jb .row_group                             ; dequantize the next group
     SET_FTZ_DAZ                               ; match the caller mode after the former dispatched copy
     vzeroupper                                ; clear the upper vector state
     ret                                       ; the row is ready
%undef EMB_ROW_F4
     global set_rope_tables                    ; Internal symbol, global for linker; NOT in v7_threading.def
set_rope_tables:
     mov [rel fused_rope_cos], rcx             ; Store cosine lookup table pointer, shared by all kernels
     mov [rel fused_rope_sin], rdx             ; Store sine lookup table pointer, shared by all kernels
     mov [rel rope_max_dim], r8d               ; elements per table position = head_dim/2, Q8 path pre-offsets by pos * this
     ret                                       ; Return to caller

; V7_STREAM_RECORD_SLOT -- StreamingLLM cache-slot resolution + eviction
; bookkeeping + ring write + slot publish, the shared one-token body of the
; F32 and Q8 forwards. %1 = position frame slot, %2 = token frame slot,
; %3 = the file's existing log-overflow FATAL label. Uses eax/ecx/edx/r10/r11
; and preserves flags as the inline originals did.
%macro V7_STREAM_RECORD_SLOT 3
     mov eax, %1                               ; start with linear position
     cmp dword [rel g_stream_enable], 0        ; streaming enabled?
     je %%publish                              ; disabled: publish raw position
     cmp eax, [rel g_stream_cap]               ; pos < C, linear fill ?
     mov r10, [rel g_stream_remove_ids]        ; r10 = ring base, flag-preserving between cmp and jb; both fill paths need it
     jb %%fill                                 ; linear fill: slot = pos, ring write only, no eviction
     mov r11d, [rel g_stream_sinks]            ; keep S through quotient and slot reconstruction
     sub eax, r11d                             ; wrapped position minus S
     mov ecx, [rel g_stream_window]            ; keep W for the valid-row check and the batch opener
     xor edx, edx                              ; zero the high half of the dividend
     div ecx                                   ; EDX = (pos - S) mod W
     lea eax, [rdx + r11]                      ; ring slot = S + remainder
     mov edx, [rel g_stream_win_valid]         ; valid window rows, including rewrites
     cmp edx, ecx                              ; has the window already been filled?
     jae %%valid_full                          ; avoid a redundant store in steady state
     inc edx                                   ; validate one more fresh row
     mov [rel g_stream_win_valid], edx         ; publish before attention runs
%%valid_full:
     cmp dword [rel g_stream_remove_warm], 0   ; post-load warm-up active?
     je %%remove_track                         ; warm latch zero -> log evictions normally
     dec dword [rel g_stream_remove_warm]      ; warm-up: fill the ring, suppress eviction logging
     jmp %%fill                                ; warm tokens only rewrite the ring
%%remove_track:
; The ring is SLOT-indexed, slot == pos below C, so one index serves both
; regimes.A wrapped write at slot h evicts the ring's current occupant of
; h, the token inserted at pos - W, provably tracked.The evicted token id is read
; BEFORE the overwrite.
     mov edx, [rel g_stream_remove_count]      ; pending log count
     cmp edx, remove_LOG_CAP                   ; refuse an undrained overflow
     jae %3                                    ; preserve the original fatal path
     test edx, edx                             ; does this token open the batch?
     jnz %%remove_push                         ; subsequent entries need only the previous token ID
     mov r11d, %1                              ; recover the absolute position for this batch opener
     sub r11d, ecx                             ; position of the removed token = pos - W
     mov [rel g_stream_remove_first_pos], r11d ; publish the first position exactly once
%%remove_push:
     mov r11d, [r10 + rax*DWORD_SCALE]         ; read the previous occupant before replacing it
     lea rcx, [rel g_stream_remove_log]        ; rcx = pending log base
     mov [rcx + rdx*DWORD_SCALE], r11d         ; log[count] = evicted token id
     inc edx                                   ; count++
     mov [rel g_stream_remove_count], edx      ; publish the new pending count
%%fill:
     mov ecx, %2                               ; ecx = this token's id
     mov [r10 + rax*DWORD_SCALE], ecx          ; ring[slot] = this token, it becomes the future eviction
%%publish:
     mov [rel g_stream_slot], eax              ; publish, single-threaded here; pool threads read after the dispatch barrier
%endmacro

%include "forward_q8.inc"
%include "forward_f32.inc"
%include "prefill_f32.inc"
%unmacro V7_STREAM_RECORD_SLOT 3               ; both forwards expanded, file-local macro leaves scope
