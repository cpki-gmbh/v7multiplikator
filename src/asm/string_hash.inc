; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define SH_LAST_WINDOW 0xFF0                   ; last in-page cursor value whose 16-byte window still fits, 0xFF0 + 15 = 0xFFF
%define SH_WINDOW_BYTES 16                     ; AVX2 window width, bytes per iteration
%define SH_FOLD_OFF8 8                         ; crc32 fold lane 2: byte offset of the window's 2nd qword
     section .text
StringHash:
     mov rdi, rsi                              ; walk pointer, rsi preserved
     xor eax, eax                              ; CRC32 accumulator = 0
     vpxor xmm0, xmm0, xmm0                    ; zero vector for NUL detection, VEX: no SSE transition penalty
     mov r8, rdi                               ; original cursor
     or r8, (SH_LAST_WINDOW | (SH_WINDOW_BYTES - 1)) ; last byte in this page
     and r8, -SH_WINDOW_BYTES                  ; last safe unaligned-vector start
     cmp rdi, r8                               ; first window page-safe?, a source may start within 16 bytes of a page end
     ja .sh_edge_walk                          ; no, byte walk, re-gates per page ; yes, falls through into .sh_win
.sh_win:
     vpcmpeqb xmm1, xmm0, [rdi]                ; NUL lanes; the 16-byte load is FOLDED into the compare, byte-for-byte the same access the separate vmovdqu made
     vptest xmm1, xmm1                         ; ZF=1 = AND all-0 = no NUL lane, yes/no is all the tail scan needs, it stops on the first NUL, so no pmovmskb/tzcnt position extraction is needed
     jnz .sh_nul_scan                          ; NUL in window -> tight scan, window is page-safe; see below
     crc32 rax, qword [rdi]                    ; fold 8 bytes, one chained step instead of two dword steps
     crc32 rax, qword [rdi + SH_FOLD_OFF8]     ; fold next 8, 16 bytes hashed per window, bit-identical
     add rdi, SH_WINDOW_BYTES                  ; advance cursor, klen derived from rdi-rsi at normalize, no per-window counter
     cmp rdi, r8                               ; NEXT window still page-safe?, the gate sits at the BOTTOM: the body falls into its own test, so the loop needs no back-edge jmp
     jbe .sh_win                               ; yes, next 16 bytes fit
     test dil, SH_WINDOW_BYTES - 1             ; already at the next aligned page start?
     jz .sh_next_page                          ; rebuild the vector bound
     jmp .sh_edge_walk                         ; consume the last bytes of the current page
.sh_nul_scan:
     vpmovmskb r8d, xmm1                       ; NUL lanes in the safe window
     tzcnt r8d, r8d                            ; first NUL lane
     inc r8d                                   ; number of bytes to hash, including NUL
     cmp r8d, 8                                ; a complete qword fits?
     jb .sh_nul_small                          ; finish the 1..7 byte remainder
.sh_nul_qword:
     crc32 rax, qword [rdi]                    ; fold eight bytes in their original order
     add rdi, 8                                ; advance the cursor
     sub r8d, 8                                ; reduce the byte count
     cmp r8d, 8                                ; a second complete qword fits only at length 16
     jae .sh_nul_qword                         ; fold the second qword
.sh_nul_small:
     test r8b, 4                               ; a dword in the binary decomposition?
     jz .sh_nul_word                           ; no
     crc32 eax, dword [rdi]                    ; fold four bytes
     add rdi, 4                                ; advance the cursor
.sh_nul_word:
     test r8b, 2                               ; a word in the decomposition?
     jz .sh_nul_byte                           ; no
     crc32 eax, word [rdi]                     ; fold two bytes
     add rdi, 2                                ; advance the cursor
.sh_nul_byte:
     test r8b, 1                               ; one byte still to hash?
     jz .sh_normalize                          ; cursor is already after the NUL
     crc32 eax, byte [rdi]                     ; fold the final byte
     inc rdi                                   ; leave cursor after the NUL
.sh_normalize:
     mov rbx, rdi                              ; klen = final cursor - original source, includes the NUL; same value the per-byte counter produced
     sub rbx, rsi                              ; rbx = klen
     mov edi, eax                              ; hash -> edi, output register, MUST move: eax holds the CRC accumulator, edi the returned hash
     test edi, edi                             ; zero hash?, empty-slot sentinel
     jz .sh_zero_hash                          ; rare: 0 -> 1, forward branch; common path falls through
     cmp edi, -1                               ; tombstone sentinel value?, imm8 sign-extends to 0xFFFFFFFF
     je .sh_tomb_hash                          ; rare: 0xFFFFFFFF -> 0xFFFFFFFE, forward branch
     ret                                       ; common: already normalized, zero taken branches
.sh_zero_hash:
     inc edi                                   ; 0 -> 1
     ret                                       ; edi = hash, ebx = klen
.sh_tomb_hash:
     dec edi                                   ; 0xFFFFFFFF -> 0xFFFFFFFE
     ret                                       ; edi = hash, ebx = klen
.sh_edge_walk:
     crc32 eax, byte [rdi]                     ; fold the next byte, r/m8 form
     cmp byte [rdi], 0                         ; NUL terminator reached?
     je .sh_edge_nul                           ; yes, rdi still points ON the NUL; advance and finish below
     inc rdi                                   ; advance walk pointer, non-NUL byte
     test dil, SH_WINDOW_BYTES - 1             ; reached the next page start?
     jnz .sh_edge_walk                         ; still inside the page-edge suffix
.sh_next_page:
     lea r8, [rdi + SH_LAST_WINDOW]            ; next page's last safe vector start
     jmp .sh_win                               ; the cursor is at that page's first byte
.sh_edge_nul:
     inc rdi                                   ; advance past the NUL, klen includes the NUL, .sh_normalize derives it from rdi - rsi
     jmp .sh_normalize                         ; cold path only
; ---- %undef cleanup: file-local defines leave scope here, house convention ----
%undef SH_FOLD_OFF8
%undef SH_WINDOW_BYTES
%undef SH_LAST_WINDOW
