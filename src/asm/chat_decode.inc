; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; chat_decode.inc, decode/penalty macros and their constants .
; Macros plus the shared penalty leaf; the %undef/%unmacro teardown stays centralized at chat.inc's tail.

; Presence penalty bitmap: 163840 bits, covers all vocab sizes up to ~152k
; Bitmap organized as dword array, one bit per token
%define BITMAP_DWORDS 5120                     ; 163840 / 32 = 5120 dwords
%define BITMAP_BYTES 20480                     ; 5120 * 4 = 20480 bytes
%define PP_QWORD_SCALE 8                       ; sizeof,qword for bitmap block addressing
%define PP_BLOCK_QWORDS 4                      ; qwords per 256-bit presence block
%define VOCAB_PTR_SCALE 8                      ; sizeof,char* for vocab array index scale
; Sentence-piece boundary, U+2581 = E2 96 81, decode-time SP-to-space constants
%define SP_BOUNDARY_SIZE 3                     ; bytes in SP boundary, 3 UTF-8 bytes
%define SP_BYTE_MID 1                          ; offset to 2nd byte of 3-byte boundary
; Byte-fallback token pattern <0xXX>, decode-time hex-to-byte constants
%define CH_HEX_OFF_0 1                         ; offset of '0' in <0xXX>
%define CH_HEX_OFF_HI 3                        ; offset of hi hex digit in <0xXX>
%define CH_HEX_OFF_LO 4                        ; offset of lo hex digit in <0xXX>
%define CH_HEX_OFF_GT 5                        ; offset of '>' in <0xXX>
%define CH_HEX_PATTERN_LEN 6                   ; total length of <0xXX> pattern
; Rewrite token buffer indexing scale, for sl_rewrite_tok_buf dword array access
%define RW_DWORD_SCALE 4                       ; sizeof,dword for rewrite token buffer indexing
%define SL_REWRITE_MAX_TOKENS 16384            ; max re-encoded replacement tokens per rewrite

%macro APPLY_PRESENCE_PENALTY 1
     mov ecx, c_presence_penalty_bits          ; FP32 bits
     test ecx, ecx                             ; +0.0?
     jz %1                                     ; skip
     cmp ecx, 0x80000000                       ; -0.0?
     je %1                                     ; skip
     mov rax, c_bitmap_base                    ; bitmap base
     mov rdx, c_logits                         ; logits base
     call chat_apply_presence_bits             ; private leaf, uses no home space
%endmacro

     section .text
chat_apply_presence_bits:
     vmovd xmm5, ecx                           ; raw penalty
     vxorps xmm0, xmm0, xmm0                   ; same positive-zero operand
     vsubss xmm5, xmm0, xmm5                   ; exact original negation
     xor r8d, r8d                              ; qword index in the bitmap
.pp_block:
     vmovdqu ymm1, [rax + r8*PP_QWORD_SCALE]   ; four complete qwords, 20 KiB is a multiple of 32
     lea r9d, [r8 + PP_BLOCK_QWORDS]           ; exclusive qword bound of this block
     vptest ymm1, ymm1                         ; any tokens present in these 256 bits?
     jz .pp_skip                               ; skip all four empty qwords
.pp_qword:
     mov r10, [rax + r8*PP_QWORD_SCALE]        ; pending token bits in this word
     test r10, r10                             ; any present?
     jz .pp_next                               ; no work for this word
.pp_bit:
     tzcnt r11, r10                            ; next set bit in ascending token order
     blsr r10, r10                             ; clear the processed bit
     mov ecx, r8d                              ; qword index
     shl ecx, 6                                ; base token ID
     lea ecx, [rcx + r11]                      ; absolute token ID
     vaddss xmm0, xmm5, [rdx + rcx*DWORD_SCALE] ; exact same FP32 operand order
     vmovss [rdx + rcx*DWORD_SCALE], xmm0      ; publish the adjusted logit
     test r10, r10                             ; more set bits in this qword?
     jnz .pp_bit                               ; process the next present token
.pp_next:
     inc r8d                                   ; next qword
     cmp r8d, r9d                              ; this 256-bit block finished?
     jb .pp_qword                              ; examine its next word
.pp_skip:
     mov r8d, r9d                              ; advance to the next block
     cmp r8d, BITMAP_BYTES / 8                 ; reached the bitmap end?
     jb .pp_block                              ; examine another complete block
     vzeroupper                                ; clear the upper vector state before returning
     ret                                       ; logits are adjusted
; Private leaf: RSI = piece, EDX = byte count, RDI = destination.
; Returns EAX = decoded byte count and RDI = destination end.
chat_decode_piece:
     xor r8d, r8d                              ; source index
     mov r10, rdi                              ; destination start
.decode_loop:
     cmp r8d, edx                              ; at the end of the source?
     jae .decode_done                          ; return the exact number written
     mov al, [rsi + r8]                        ; ordinary bytes are emitted unchanged
     cmp al, '<'                               ; possible byte escape?
     je .decode_hex                            ; validate its full shape out of line
     cmp al, 0xE2                              ; possible SentencePiece boundary?
     je .decode_sp                             ; validate both remaining marker bytes
.decode_copy_byte:
     stosb                                     ; emit one byte, with DF clear
     inc r8d                                   ; consume one ordinary source byte
     jmp .decode_loop                          ; continue decoding
.decode_hex:
     lea ecx, [r8 + CH_HEX_PATTERN_LEN]        ; exclusive end of the six-byte escape
     cmp ecx, edx                              ; inside the input extent?
     ja .decode_copy_byte                      ; truncated escape remains literal
     cmp word [rsi + r8 + CH_HEX_OFF_0], 0x7830 ; require '0x'
     jne .decode_copy_byte                     ; unrecognized shape remains literal
     cmp byte [rsi + r8 + CH_HEX_OFF_GT], '>'  ; require closing bracket
     jne .decode_copy_byte                     ; unrecognized shape remains literal
     lea r11, [rel hex_val_lut]                ; nibble plus one, zero is invalid
     movzx ecx, byte [rsi + r8 + CH_HEX_OFF_HI] ; load the high digit
     movzx r9d, byte [r11 + rcx]               ; decode the high nibble
     test r9d, r9d                             ; reject invalid high digit
     jz .decode_copy_byte                      ; retain the original '<' in AL
     movzx ecx, byte [rsi + r8 + CH_HEX_OFF_LO] ; load the low digit
     movzx ecx, byte [r11 + rcx]               ; decode the low nibble
     test ecx, ecx                             ; reject invalid low digit
     jz .decode_copy_byte                      ; retain the original '<' in AL
     dec r9d                                   ; remove the high nibble bias
     dec ecx                                   ; remove the low nibble bias
     shl r9d, 4                                ; position the high nibble
     or r9b, cl                                ; combine the two nibbles
     mov al, r9b                               ; byte to emit, including an escaped zero
     add r8d, CH_HEX_PATTERN_LEN - 1           ; common emit consumes the final byte
     jmp .decode_copy_byte                     ; emit and advance
.decode_sp:
     lea ecx, [r8 + SP_BOUNDARY_SIZE]          ; exclusive end of the three-byte marker
     cmp ecx, edx                              ; inside the input extent?
     ja .decode_copy_byte                      ; truncated marker remains literal
     cmp word [rsi + r8 + SP_BYTE_MID], 0x8196 ; require the full marker
     jne .decode_copy_byte                     ; partial match remains literal
     mov al, ' '                               ; emit one ASCII space
     add r8d, SP_BOUNDARY_SIZE - 1             ; common emit consumes the final byte
     jmp .decode_copy_byte                     ; emit and advance
.decode_done:
     mov eax, edi                              ; low dword of the output end
     sub eax, r10d                             ; byte count fits in a dword
     ret                                       ; RDI still points at the output end

%macro CHAT_DECODE_PIECE 0
     mov edx, r14d                             ; byte count for the shared decoder
     call chat_decode_piece                    ; normalize the token piece
     mov r13d, eax                             ; existing chat decoded-length result
     mov edi, eax                              ; retain the original macro's RDI result
%endmacro

chat_decode_len:
     xor r9d, r9d                              ; r9d = decoded byte count = 0
     xor r8d, r8d                              ; r8d = src_idx = 0
.dlen_loop:
     cmp r8d, r14d                             ; src_idx < slen?
     jge .dlen_done                            ; no -> done
     movzx eax, byte [rsi + r8]                ; al = piece[src_idx]
     cmp al, 0x3C                              ; '<', potential <0xXX> token?
     jne .dlen_check_sp                        ; not '<', check SP boundary
     lea ecx, [r8 + CH_HEX_PATTERN_LEN]        ; exclusive end of the escape or marker
     cmp ecx, r14d                             ; does it fit in the source?
     ja .dlen_copy_byte                        ; count a truncated prefix literally
     cmp word [rsi + r8 + CH_HEX_OFF_0], 0x7830 ; '0','x' little-endian at +1/+2
     jne .dlen_copy_byte                       ; not <0xXX>, counts as-is
     cmp byte [rsi + r8 + CH_HEX_OFF_GT], 0x3E ; '>' at offset +5?
     jne .dlen_copy_byte                       ; not <0xXX>, counts as-is
     movzx ecx, byte [rsi + r8 + CH_HEX_OFF_HI] ; ecx = hi hex char, LUT index
     movzx edx, byte [rsi + r8 + CH_HEX_OFF_LO] ; edx = lo hex char, LUT index
     lea r11, [rel hex_val_lut]                ; r11 = nibble decode table, nibble+1 encoding, 0 invalid
     cmp byte [r11 + rcx], 0                   ; zero invalid; nonzero = valid hex
     je .dlen_copy_byte                        ; invalid hi: counts as ordinary text
     cmp byte [r11 + rdx], 0                   ; zero invalid; nonzero = valid hex
     je .dlen_copy_byte                        ; invalid lo: counts as ordinary text
     add r8d, CH_HEX_PATTERN_LEN               ; src_idx += 6, skip <0xXX>
     inc r9d                                   ; decoded length += 1
     jmp .dlen_loop                            ; continue scan
.dlen_check_sp:
     cmp al, 0xE2                              ; first byte of SP boundary?
     jne .dlen_copy_byte                       ; no, counts as-is
     lea ecx, [r8 + SP_BOUNDARY_SIZE]          ; exclusive end of the escape or marker
     cmp ecx, r14d                             ; does it fit in the source?
     ja .dlen_copy_byte                        ; count a truncated prefix literally
     cmp word [rsi + r8 + SP_BYTE_MID], 0x8196 ; bytes 96 81 little-endian at +1/+2
     jne .dlen_copy_byte                       ; no, counts as-is
     add r8d, SP_BOUNDARY_SIZE                 ; src_idx += 3, skip SP boundary
     inc r9d                                   ; decoded length += 1
     jmp .dlen_loop                            ; continue scan
.dlen_copy_byte:
     inc r8d                                   ; src_idx += 1
     inc r9d                                   ; decoded length += 1
     jmp .dlen_loop                            ; continue scan
.dlen_done:
     ret                                       ; R9D is the decoded length

%macro CHAT_DECODE_LEN 0
     call chat_decode_len                      ; count without writing to the caller's output
%endmacro
