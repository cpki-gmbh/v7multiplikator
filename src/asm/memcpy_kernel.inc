; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; Local constants for overlap method
; ============================================================================
%define OFFSET_NEG_32 -32                      ; Negative offset for 32-byte overlap
%define OFFSET_NEG_16 -16                      ; Negative offset for 16-byte overlap
%define OFFSET_NEG_8 -8                        ; Negative offset for 8-byte overlap
%define OFFSET_NEG_4 -4                        ; Negative offset for 4-byte overlap
%define OFFSET_NEG_2 -2                        ; Negative offset for 2-byte overlap

%macro MEMCPY_TINY 3
     cmp r11d, 32                              ; == 32 B exactly?
     je %%tiny_32_exact                        ; yes, single full-width load/store, the two-ended path would copy the same 32 B twice
     jb %%tiny_16                              ; < 32 B, try 16 B tier
     vmovdqu ymm0, [%3]                        ; load first 32 B
     vmovdqu ymm1, [%3 + r11 + OFFSET_NEG_32]  ; load last 32 B, overlaps middle
     vmovdqu [%2], ymm0                        ; write first 32 B
     vmovdqu [%2 + r11 + OFFSET_NEG_32], ymm1  ; write last 32 B
     vzeroupper                                ; clean dirty ymm0/1 upper bits, only tier touching ymm
     jmp %1                                    ; all done
%%tiny_32_exact:
     vmovdqu ymm0, [%3]                        ; load the exact 32 B
     vmovdqu [%2], ymm0                        ; store them once, 256-bit VEX form, consistent with the 32-byte tier above, both vmovdqu and vmovups are unaligned-safe, so this is encoding consistency, not alignment; the vzeroupper below is what this tier's 256-bit access costs
     vzeroupper                                ; clean dirty ymm0 upper bits
     jmp %1                                    ; all done
%%tiny_16:
     cmp r11d, 16                              ; >= 16 B?
     jb %%tiny_8                               ; no, try 8 B tier
     je %%tiny_16_exact                        ; exactly 16 B: skip the duplicated end load/store
     vmovups xmm0, [%3]                        ; load first 16 B, VEX.128: no AVX-SSE transition when uppers are dirty
     vmovups xmm1, [%3 + r11 + OFFSET_NEG_16]  ; load last 16 B, overlaps middle
     vmovups [%2], xmm0                        ; write first 16 B
     vmovups [%2 + r11 + OFFSET_NEG_16], xmm1  ; write last 16 B
     jmp %1                                    ; all done
%%tiny_16_exact:
     vmovups xmm0, [%3]                        ; exact 16 B, no overlap load needed
     vmovups [%2], xmm0                        ; write the 16 bytes once
     jmp %1                                    ; all done
%%tiny_8:
     cmp r11d, 8                               ; >= 8 B?
     jb %%tiny_4                               ; no, try 4 B tier
     je %%tiny_8_exact                         ; exactly 8 B: avoid the duplicated overlapping qword copy
     mov r10, [%3]                             ; load first 8 B
     mov r8, [%3 + r11 + OFFSET_NEG_8]         ; load last 8 B
     mov [%2], r10                             ; write first 8 B
     mov [%2 + r11 + OFFSET_NEG_8], r8         ; write last 8 B
     jmp %1                                    ; all done
%%tiny_8_exact:
     mov r10, [%3]                             ; exact 8 B
     mov [%2], r10                             ; write the 8 bytes once
     jmp %1                                    ; all done
%%tiny_4:
     cmp r11d, 4                               ; >= 4 B?
     jb %%tiny_2                               ; no, try 2 B tier
     mov r8d, [%3]                             ; load first 4 B
     mov r9d, [%3 + r11 + OFFSET_NEG_4]        ; load last 4 B
     mov [%2], r8d                             ; write first 4 B
     mov [%2 + r11 + OFFSET_NEG_4], r9d        ; write last 4 B
     jmp %1                                    ; all done
%%tiny_2:
     cmp r11d, 2                               ; >= 2 B?
     jb %%tiny_1                               ; no, 1 B tier
     movzx r8d, word [%3]                      ; load first 2 B
     movzx r9d, word [%3 + r11 + OFFSET_NEG_2] ; load last 2 B
     mov [%2], r8w                             ; write first 2 B
     mov [%2 + r11 + OFFSET_NEG_2], r9w        ; write last 2 B
     jmp %1                                    ; all done
%%tiny_1:
     movzx r8d, byte [%3]                      ; load the single byte, full 32-bit write, no partial-reg dependency
     mov [%2], r8b                             ; store it
     jmp %1                                    ; done
%endmacro
     section .text                             ; self-declare: the aligned kernel entry below must land in .text regardless of the predecessor file's trailing section
     align 64
     global v7s_memcpy_chunk                   ; Export ERMS memcpy kernel
v7s_memcpy_chunk:
     mov rax, rcx                              ; rax = original dst pointer, return value
     cmp r8d, 64                               ; n < 64?
     jb .tiny_erms                             ; yes, overlap method, no rdi/rsi setup needed there: the tiers never touch them
; Align dst to 16 bytes: Zen 3 ERMS penalty avoidance.
; Unaligned rep movsb can be 20x slower on Zen 3/Zen 4.
     push rdi                                  ; save rdi, non-volatile, overwritten by rep movsb
     push rsi                                  ; save rsi, non-volatile, overwritten by rep movsb
     mov rdi, rcx                              ; rdi = dst, rep movsb destination
     mov rsi, rdx                              ; rsi = src, rep movsb source
     ALIGN_DESTINATION_POINTER edi, r8d, 16    ; Align dst to 16 bytes for ERMS, count stays in r8d, no staging hop
     mov ecx, r8d                              ; rcx = remaining byte count, 32-bit mov zero-extends
     rep movsb                                 ; copy remaining bytes via ERMS, dst now 16-aligned
     pop rsi                                   ; restore rsi
     pop rdi                                   ; restore rdi
     ret                                       ; return to caller, rax = original dst
; Overlap method for n < 64 bytes, branchless per tier
; 8-byte scratch uses volatile r10: r11d is reserved for the remaining count, and the non-volatiles are untouched
.tiny_erms:
     mov r11d, r8d                             ; r11d = byte_count, moved from entry: the tiny tiers own r11d
     MEMCPY_TINY .tiny_done, rcx, rdx          ; Expand tiny copy macro on the caller's arg registers, rdi/rsi never touched
.tiny_done:
     ret                                       ; return to caller, rax = original dst
; ---- %undef cleanup: keep the preprocessor namespace clean for later includes ----
%undef OFFSET_NEG_32
%undef OFFSET_NEG_16
%undef OFFSET_NEG_8
%undef OFFSET_NEG_4
%undef OFFSET_NEG_2
%unmacro MEMCPY_TINY 3                         ; arity MUST match the %macro declaration; defined and expanded only in this file
