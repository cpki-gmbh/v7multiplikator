; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; The AES rounds here use the LEGACY SSE encoding (aesenc, AES-NI since
; Westmere 2010), not the VEX encoding (vaesenc, which needs VAES): every
; site is the destructive form dst == first source, so both encodings are
; the same operation with the same result bits -- the legacy spelling runs
; on AES-NI-only CPUs that VAES would fatally gate.

%define AESH_BLOCK 16                          ; absorb block size, bytes per loop iteration
%define AESH_MASK_BYTES 16                     ; absorb block width, the tail keeps exactly r8d + AESH_MASK_BYTES real bytes
     section .rdata
     align 16
AESH_BYTE_INDEX:
     db 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ; granule index 0..15, the tail mask is built by comparing these against k, the real-byte count
     section .text
aes_string_hash:
     mov r9, rsi                               ; r9 = source cursor, rsi itself is never touched
     vpxor xmm1, xmm1, xmm1                    ; b = rolling AES state = 0
     vpxor xmm2, xmm2, xmm2                    ; c = XOR accumulator = 0
.block:
     vmovdqu xmm0, [r9]                        ; a = 16 bytes at pos, final block may overread past the NUL: committed memory, by design
     add r9, AESH_BLOCK                        ; pos += 16
     sub r8d, AESH_BLOCK                       ; remaining -= 16, CF set iff the block ran past the key end
     jc .tail                                  ; partial block: mask the overread lanes before absorbing
     aesenc xmm1, xmm0                         ; all 16 lanes carry real key bytes, b = AESRound,state=b, roundkey=a
     vpxor xmm2, xmm2, xmm1                    ; c ^= b
     ja .block                                 ; remaining > 0: another block, flags are active from the sub, aesenc/vpxor write no flags
.finish:
     aesenc xmm2, xmm1                         ; FINALIZE round 1: mixes columns across the halves so the low-64 extraction depends on all 128 absorbed bits
     aesenc xmm2, xmm1                         ; FINALIZE round 2: completes the mixing, both rounds together make every output bit depend on all 128 absorbed bits
     vmovq rax, xmm2                           ; hash = low64,c, now carrying all 128 absorbed bits
     test rax, rax                             ; zero hash?, would false-match an empty cache slot
     jnz .done                                 ; common path: already nonzero
     inc eax                                   ; map 0 -> 1
.done:
     ret                                       ; rax = normalized u64 hash
.tail:
     lea eax, [r8 + AESH_MASK_BYTES]           ; eax = k = r8d(remaining - 16) + 16 = real bytes in the final block, 1..15
     vmovd xmm3, eax                           ; xmm3 = k in the low byte
     vpbroadcastb xmm3, xmm3                   ; all 16 lanes = k
     vpcmpgtb xmm3, xmm3, [rel AESH_BYTE_INDEX] ; byte i = 0xFF iff i < k: keep exactly the k real bytes, zero the overread tail
     vpand xmm0, xmm0, xmm3                    ; zero the overread tail lanes
     aesenc xmm1, xmm0                         ; b = AESRound,b, masked a
     vpxor xmm2, xmm2, xmm1                    ; c ^= b
     jmp .finish                               ; same finalization as full blocks
; %undef cleanup: file-local defines leave scope here, house convention --
%undef AESH_BLOCK
%undef AESH_MASK_BYTES
