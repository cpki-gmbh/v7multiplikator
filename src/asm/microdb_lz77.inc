; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define LZ77_PTR_SCALE 2                       ; 16-bit modulo-address tags, two bytes per hash slot
%define LZ77_TOKEN_BYTES 1                     ; bytes of the leading short-input token
%define LZ77_VECTOR_SAFE 15                    ; 16-byte load needs 16 remaining bytes
; NOTE: callers invoke this with their LZ77 hash table adjacent to RSP, this helper must stay stack-free
; rcx/rdi/flags only ; a push or call here would scratch the caller's shadow space, which overlaps hash slots 0-2.
emit_varlen_func:
     cmp rcx, 255                              ; more than one marker byte needed?
     jb .done                                  ; remainder only, skip loop body
.loop:
     mov byte [rdi], 255                       ; write 255 marker, direct store beats microcoded stosb
     inc rdi                                   ; advance output pointer
     sub rcx, 255                              ; reduce remaining count, 64-bit
     cmp rcx, 255                              ; still more markers?
     jae .loop                                 ; bottom-tested back-edge, cmp/jae macro-fuse on Zen 4, no jmp
.done:
     mov byte [rdi], cl                        ; write remainder, rcx < 255 here, fits in low byte
     inc rdi                                   ; advance output pointer
     ret                                       ; back to caller
mdb_compress:
; Inputs below 15 fit a single literal token without a length-extension byte;
; bypass the costly hash-table setup even if a tiny match would be possible.
     cmp rdx, 15                               ; literal count fits the 4-bit token field?
     jae .c_long_input                         ; no: full prologue and compressor
     push rsi                                  ; save the two string registers this path uses
     push rdi                                  ; save destination cursor, rep movsb advances it
     mov rsi, rcx                              ; rsi = src
     mov rdi, r8                               ; rdi = dst
     mov eax, edx                              ; eax = length 0..14
     shl eax, 4                                ; single final token, high nibble = length, low nibble 0
     mov byte [rdi], al                        ; emit the token
     inc rdi                                   ; advance past the token
     mov ecx, edx                              ; literal byte count, 0..14
     rep movsb                                 ; copy the literals, DF=0 from the entry ABI
     lea rax, [rdx + LZ77_TOKEN_BYTES]         ; one token plus rdx input bytes
     pop rdi                                   ; restore the string registers
     pop rsi                                   ; restore source cursor
     ret                                       ; done, no frame was ever built
.c_long_input:
     push rbp                                  ; non-volatile, holds the nibble-clamp constant 15 for the whole call; completes the 8-push parity
     push rbx                                  ; save non-volatile index
     push rsi                                  ; save non-volatile source
     push rdi                                  ; save non-volatile destination
     push r12                                  ; save non-volatile
     push r13                                  ; save non-volatile, caches min-match threshold
     push r14                                  ; save non-volatile, dst base
     push r15                                  ; save non-volatile
     mov ebp, 15                               ; nibble-clamp constant, rbp is otherwise dead for the whole function, set before every path into .c_final
     mov r15d, (LZ77_HT_SLOTS * 2) / 4096      ; 4 stack pages for the 16 KiB tag table, frame and probe stay synchronized
.probe:
     sub rsp, 4096                             ; move down one page
     test [rsp], rsp                           ; touch page, guard page trigger
     dec r15d                                  ; one fewer page
     jnz .probe                                ; continue until all pages probed
     sub rsp, 24                               ; reach LZ77_COMPRESS_FRAME total, post-prologue RSP 0 mod 16, calls aligned
     test [rsp - LZ77_SAVED_RDI_OFF], rsp      ; commit the page holding the pad and the push rdi slot, the lowest byte the frame uses
     lea r12, [rcx + rdx]                      ; src end = base + length, 64-bit, from the incoming src pointer, rcx is untouched by the prologue
     lea r9, [r12 - LZ77_VECTOR_SAFE]          ; vector-safe match bound = src_end - 15, loop-invariant: r12 never modified, r9 untouched elsewhere, hoisted out of the per-match path
     mov rbx, rcx                              ; current position, ip
     mov rsi, rbx                              ; literal start follows position
     mov r14, r8                               ; save dst base in non-volatile r14
     mov rdi, r8                               ; dst position, op
; The entry short-input gate already established rdx >= LZ77_MIN_MATCH_END
; Hash table on stack, 16 KiB = 8K words; unsigned 16-bit modulo distances
     lea r15, [rsp + LZ77_HT_BASE_OFF]         ; hash table base
; Seed every slot with the input start tag (a valid candidate, unlike zero which can alias a real low-16 address)
     push rdi                                  ; save dst pointer
     mov rdi, r15                              ; point rdi at hash table
     mov ecx, LZ77_HT_SLOTS                    ; 8K 16-bit tags
     mov eax, ebx                              ; input start address, only AX is stored
     rep stosw                                 ; seed 16 KiB with the start tag
     pop rdi                                   ; restore dst pointer
     lea r13, [r12 - LZ77_MIN_MATCH_END]       ; cache threshold in r13, constant for entire function
     cmp rbx, r13                              ; enough room for a match?
     jae .c_final                              ; no, emit all remaining as literals
.c_loop:
     mov ecx, [rbx]                            ; cache current dword once, hash input + match compare source
     imul eax, ecx, 0x9E3779B9                 ; multiply by the sqrt2 hash constant, register form, legacy-encoded
     shr eax, 19                               ; top 13 bits index the 8192-slot hash table
     movzx r8d, word [r15 + rax*LZ77_PTR_SCALE] ; previous low-16 tag, pre-seeded to the source start
     mov word [r15 + rax*LZ77_PTR_SCALE], bx   ; publish current low-16 tag
     movzx r10d, bx                            ; zero r10 upper bits before the modular subtraction
     sub r10w, r8w                             ; modulo-65536 distance, upper 16 bits stay zero
     jz .c_adv                                 ; 0/65536 distance is not encodable
     mov r8, r10                               ; keep the positive distance for the encoded offset
     neg r8                                    ; negative distance indexes the matching bytes from rbx
     cmp ecx, [rbx + r8]                       ; verify the candidate's first 4 bytes before extending
     jne .c_adv                                ; no, skip
     mov r11, rbx                              ; literal length = position - literal start
     sub r11, rsi                              ; bytes since last match/literal
     mov rcx, rbx                              ; preserve the match start, r10 keeps the verified distance, not a push: RSP stays aligned for the emit_varlen_func calls below
     add rbx, LZ77_MIN_MATCH_LEN               ; the first four bytes already match
; Fast path: compare 16 bytes at a time while both loads stay inside the input
; bound hoisted in r9 at function entry, r9 has no other writer in mdb_compress
     cmp rbx, r9                               ; room for two 16-byte loads?
     jae .ext_tail                             ; no, fall back to byte compare
.extend_vec:
     vmovdqu xmm0, [rbx]                       ; 16 current bytes
     vpcmpeqb xmm0, xmm0, [rbx + r8]           ; compare the corresponding earlier bytes
     vpmovmskb eax, xmm0                       ; one bit per matching byte
     xor eax, 0xFFFF                           ; differing byte positions
     jnz .ext_diff                             ; difference found, locate it
     add rbx, LZ77_VECTOR_SAFE + 1             ; advance the cursor a full vector
     cmp rbx, r9                               ; room for another full vector?
     jb .extend_vec                            ; yes, continue vector loop
.ext_tail:
     cmp rbx, r12                              ; safe to read one more byte?, compare against src_end directly
     jae .ext_done                             ; no, stop
     mov al, byte [rbx]                        ; load current byte, mov r8,m8 = 3 bytes vs 5 for movzx; only al is consumed by the cmp, eax upper bits dead until rewritten
     cmp al, [rbx + r8]                        ; earlier byte at the fixed match distance
     jne .ext_done                             ; no, extension complete
     inc rbx                                   ; advance position
     jmp .ext_tail                             ; continue byte compare
.ext_diff:
     bsf eax, eax                              ; find first differing byte within the vector
     add rbx, rax                              ; advance position to diff
.ext_done:
; match-4 length = rbx - match_start - 4
     lea rax, [rbx - LZ77_MIN_MATCH_LEN]       ; match end - 4-byte minimum, disp8 folds the constant; base rbx, no index
     sub rax, rcx                              ; subtract match start = match-4 length, 64-bit
; emit token with branchless CMOV nibble clamping
     mov rdx, r11                              ; literal length, 64-bit, r11d form truncates >4GB runs, §45
     cmp rdx, rbp                              ; exceeds nibble max?, 64-bit compare
     cmovae edx, ebp                           ; clamp high nibble to 15
     shl edx, 4                                ; shift to token high nibble
     mov r8, rax                               ; match-4 length, 64-bit, r8d form truncates >4GB matches, §45
     cmp r8, rbp                               ; exceeds nibble max?, 64-bit compare
     cmovae r8d, ebp                           ; clamp low nibble to 15
     or edx, r8d                               ; combine nibbles into token byte
     mov [rdi], dl                             ; emit token byte
     inc rdi                                   ; advance past token
     cmp r11, rbp                              ; extra literal length needed?, 64-bit
     jb .no_ll_extra                           ; no
     lea rcx, [r11 - LZ77_NIBBLE_MAX]          ; rcx = total literal len - 15, disp8 folds the subtract; r11 is read-only here
; RSP%16=0 here, post-prologue, no pushes live : aligned for CALL
     call emit_varlen_func                     ; emit variable-length literal size
.no_ll_extra:
     mov rcx, r11                              ; literal byte count for rep, 64-bit
     rep movsb                                 ; copy literal bytes from rsi to rdi
     mov [rdi], r10w                           ; emit 16-bit offset
     add rdi, 2                                ; advance past offset bytes
     cmp rax, rbp                              ; extra match length needed?, unclamped match-4 len still in rax, emit_varlen_func touches only rcx/rdi
     jb .no_ml_extra                           ; no
     lea rcx, [rax - LZ77_NIBBLE_MAX]          ; rcx = total match len - 15, disp8 folds the subtract; rax is read-only here
     call emit_varlen_func                     ; emit variable-length match size, RSP still 16-aligned: no pushes live
.no_ml_extra:
     mov rsi, rbx                              ; new literal start = current position
     cmp rbx, r13                              ; still room for another match?
     jb .c_loop                                ; yes, continue compressing
; no -> fall through into .c_final
.c_final:
     mov r11, r12                              ; end of source
     sub r11, rsi                              ; remaining literals = end - literal start
     mov eax, 15                               ; default high nibble = 15
     cmp r11, rax                              ; 64-bit compare, r11 is a byte count, not a nibble
     cmovb eax, r11d                           ; no: use actual length, branchless, 32-bit fine for nibble
     shl eax, 4                                ; shift to token high nibble
     mov [rdi], al                             ; emit final token, no match
     inc rdi                                   ; advance past token
     cmp r11, rbp                              ; 64-bit compare, r11 is a byte count, not a nibble
     jb .lit_copy                              ; no
     lea rcx, [r11 - LZ77_NIBBLE_MAX]          ; rcx = total literal len - 15, disp8 folds the subtract
; RSP%16=0 here, post-prologue 0, depth 0 : aligned for CALL
     call emit_varlen_func                     ; emit variable-length literal size
.lit_copy:
     mov rcx, r11                              ; literal byte count for rep, full 64-bit count
     rep movsb                                 ; copy remaining literals to output
     mov rax, rdi                              ; current output position, 64-bit
     sub rax, r14                              ; minus base = compressed size, r14 = dst base
     jmp .c_done                               ; epilogue, out of line below
.c_adv:
     inc rbx                                   ; skip one byte, no match found
     cmp rbx, r13                              ; enough room for a match now?
     jb .c_loop                                ; yes, try again at next position
     jmp .c_final                              ; no, backward jump, emit remaining as literals
.c_done:
     add rsp, LZ77_COMPRESS_FRAME              ; undo hash table + alignment pad
     pop r15                                   ; restore non-volatile
     pop r14                                   ; restore non-volatile
     pop r13                                   ; restore non-volatile, ABI compliance
     pop r12                                   ; restore non-volatile
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     pop rbp                                   ; restore non-volatile
     ret                                       ; back to caller
mdb_decompress:
     push rbp                                  ; save non-volatile, dst base
     push rsi                                  ; save non-volatile
     push rdi                                  ; save non-volatile
     push rbx                                  ; save non-volatile
     push r12                                  ; save non-volatile, for src_end
     mov rsi, rcx                              ; src pointer
     lea r12, [rcx + rdx]                      ; r12 = src end, 64-bit
     mov rdi, r8                               ; dst pointer
     mov rbp, r8                               ; save dst base in non-volatile rbp
     lea r9, [r8 + r9]                         ; r9 = dst end = dst base + max_dst, the caller's output limit, enforced at every write below
.d_loop:
     cmp rsi, r12                              ; reached end of input?
     jae .d_done                               ; yes, decompression done
     movzx r10d, byte [rsi]                    ; load token byte, zero-extends to full r10
     inc rsi                                   ; advance past token
     mov ebx, r10d                             ; token copy for match base
     and ebx, 0Fh                              ; extract low nibble = match base
     shr r10d, 4                               ; extract high nibble = literal length, 32-bit op zero-extends r10
; --- literal length extension ---
     cmp r10d, 15                              ; literal nibble, SHR R10D,4 zero-extended r10
     jb .lit_ready                             ; no
     call .read_ext                            ; shared extension reader
.lit_ready:
     lea rax, [rsi + r10]                      ; literal run end in the source
     cmp rax, r12                              ; run ends inside the source buffer?
     ja .d_corrupt                             ; no, truncated or corrupt stream, refuse
     lea rax, [rdi + r10]                      ; literal run end in the destination
     cmp rax, r9                               ; run ends inside max_dst?
     ja .d_corrupt                             ; no, would overflow the output buffer, refuse
     mov rcx, r10                              ; literal byte count
     rep movsb                                 ; copy literal bytes
     cmp rsi, r12                              ; reached end of input?
     jae .d_done                               ; stream end, the low nibble is 0 on our compressor's final token
.need_match:
; --- offset ---
     lea rax, [rsi + LZ77_OFFSET_BYTES]        ; offset field end in the source
     cmp rax, r12                              ; full offset field inside the source buffer?
     ja .d_corrupt                             ; no, truncated stream, refuse
     movzx r11d, word [rsi]                    ; load 16-bit offset, zero-extends to r11
     add rsi, LZ77_OFFSET_BYTES                ; advance past offset
     test r11d, r11d                           ; 16-bit offset was zero-extended, 32-bit test suffices
     jz .d_corrupt                             ; zero offset, corrupt stream, refuse
     mov rax, rdi                              ; dst cursor
     sub rax, rbp                              ; minus dst base = bytes written so far
     cmp r11, rax                              ; offset may reach back to the first output byte, == written is legal
     ja .d_corrupt                             ; offset beyond the written output, corrupt stream, refuse
; --- match length extension ---
     mov r10, rbx                              ; reload match base, 64-bit for >4GB lengths
     cmp r10, 15                               ; extra match length bytes?, 64-bit
     jb .match_ready                           ; no
     call .read_ext                            ; shared extension reader
.match_ready:
     add r10, 4                                ; minimum match is 4 bytes, 64-bit
     lea rax, [rdi + r10]                      ; match copy end in the destination
     cmp rax, r9                               ; copy ends inside max_dst?
     ja .d_corrupt                             ; no, would overflow the output buffer, refuse
; An overlapping match repeats the last offset bytes. Copy only from
; the prefix that is already available, growing that prefix after each copy.
     mov rdx, rsi                              ; preserve the compressed input cursor
     mov r8, rdi                               ; destination at the start of this match
     sub r8, r11                               ; start of the available periodic prefix
     cmp r11, 1                                ; a one-byte period is a fill
     jne .copy_test                            ; other periods use the common block copy
     mov al, [r8]                              ; repeated byte
     mov rcx, r10                              ; match length
     rep stosb                                 ; fill the entire match
     jmp .copy_done                            ; recover the compressed input cursor
.copy_grow:
     mov rsi, r8                               ; begin at the start of the repeating prefix
     mov rcx, r11                              ; copy the complete available prefix
     rep movsb                                 ; source ends exactly at the old destination
     sub r10, r11                              ; remaining output bytes
     add r11, r11                              ; twice as many prefix bytes are now available
.copy_test:
     cmp r10, r11                              ; does the rest fit in the available prefix?
     ja .copy_grow                             ; expand the prefix once more
     mov rsi, r8                               ; start of the available prefix
     mov rcx, r10                              ; final nonoverlapping copy length
     rep movsb                                 ; copy the remaining bytes
.copy_done:
     mov rsi, rdx                              ; restore the compressed input cursor
     jmp .d_loop                               ; decode the next token
.read_ext:
     cmp rsi, r12                              ; extension byte still inside the source buffer?
     jae .d_corrupt                            ; no, truncated stream or unterminated 0xFF run, refuse
     movzx eax, byte [rsi]                     ; read extra length byte, zero-extends
     inc rsi                                   ; advance past it
     add r10, rax                              ; accumulate into length, 64-bit for >4GB
     cmp al, 255                               ; more extra bytes follow?
     je .read_ext                              ; yes, continue reading
     ret                                       ; back to caller
.d_done:
     mov rax, rdi                              ; current output position, 64-bit
     sub rax, rbp                              ; minus base = decompressed size, rbp = dst base
.d_end:
     pop r12                                   ; restore non-volatile, src_end, the 5 pushes keep rsp 16-aligned at the .read_ext call sites
     pop rbx                                   ; restore non-volatile
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbp                                   ; restore non-volatile
     ret                                       ; back to caller
.d_corrupt:
     FATAL "mdb_decompress: corrupt LZ77 stream, offset or length outside source or destination bounds"
%undef LZ77_VECTOR_SAFE
%undef LZ77_TOKEN_BYTES
%undef LZ77_PTR_SCALE
