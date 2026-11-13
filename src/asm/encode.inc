; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; All Tokenizer field offsets come from the shared TOK_* defines in
; inf_structs.inc, so a layout change lands in one place.
%include "inf_structs.inc"
; Float constant -1e10f as int32 bit pattern for best_score initialization
; Verified: -1e10f hex = 0xD01502F9
%define ENC_NEG_1E10 0xD01502F9                ; float -1e10 as int32 bit pattern for best_score init
%define MAX_SPECIAL_TOKEN_LEN 64               ; max special-token text length, scan bound
%define ENC_SPECIAL_BUF (MAX_SPECIAL_TOKEN_LEN + 4) ; special buffer: 64 chars + null + safety pad
; Numeric constants for struct offsets and array sizes, NUMOFFSET compliance
%define PTR_SCALE 8                            ; sizeof,pointer for char** vocab array indexing
%define ARG5_OFF 48                            ; 5th stack arg, max_tokens at rbp+48
%define STRBUF_OVRHD 8                         ; str_buffer headroom beyond 2*max_token_length, null + safety
%define SKIP1_TOKEN 4                          ; skip one int32 in token array
%define SKIP2_TOKEN 8                          ; skip two int32s in token array
%define DECODE_OFF_1 1                         ; byte offset 1 in vocab string, second char: '0'
%define DECODE_OFF_2 2                         ; byte offset 2 in vocab string, third char: 'x'
%define DECODE_OFF_3 3                         ; byte offset 3 in vocab string, high hex nibble
%define DECODE_OFF_4 4                         ; byte offset 4 in vocab string, low hex nibble
%define DECODE_OFF_5 5                         ; byte offset 5 in vocab string, closing '>'
%define DECODE_OFF_NULL 6                      ; byte offset 6: null terminator after closing '>'
%define SP_UTF8_LO 0x8196E2                    ; _ U+2581 as little-endian int32, E2 96 81 00
%define ENC_SPECIAL_PAST_GT 1                  ; +1 advance past the closing '>' when skipping a special token
%define ENC_CONT_MASK 0xC0                     ; top-2-bit mask: a UTF-8 continuation byte is 0b10xxxxxx
%define ENC_CONT_TAG 0x80                      ; continuation-byte class value after ENC_CONT_MASK
%define ENC_CODEPOINT_MAX 4                    ; longest UTF-8 codepoint, cursor cap from the reference
%define ENC_KEY_SCRATCH 8                      ; offset of the single-byte scratch key past str_buffer's codepoint
%define ENC_PEEK_NEXT 1                        ; offset of the next input byte for the codepoint-completeness peek
%define DN_SWAR_REP1 0x0101010101010101        ; rep,0x01 per byte lane, SWAR zero-detect subtrahend
%define DN_SWAR_FLAG80 0x8080808080808080      ; per-lane 0x80 flag mask, only these bits indicate a zero byte
%define DN_PAGE_MASK 0xFFF                     ; 4KB page offset mask for the source page-gate probe
%define DN_PAGE_EDGE 0xFF8                     ; last in-page source offset where an 8-byte read stays in-page
%define DN_QWORD_BYTES 8                       ; SWAR window width, bytes per load/store
%define DN_NUL_SLOT 1                          ; one slot reserved for the final NUL, dest bound = base + size - 1
%define ENC_STACK_BUF 4096                     ; str_buffer stack-buffer size, must match the SF_ARRAY below; sizes <= this use the stack path
%define EDGE_STRIDE 8                          ; sizeof,edge record: merged id dword + score bits dword
%define EDGE_SCORE_OFF 4                       ; byte offset of score bits inside one edge record
%define EDGE_SKIP1 8                           ; byte advance of one edge record in the shift sources/dests
%define EDGE_SKIP2 16                          ; byte advance of two edge records, the dropped-pair window

%macro ENC_APPEND_TOKEN 0
     mov ecx, enc_n_tokens                     ; ecx = current token count
     cmp ecx, ebx                              ; n_tokens < max_tokens? SIGNED: ebx is zero-extended by its 32-bit load, mov ebx, [rbp + ARG5_OFF], so an unsigned compare would read a negative max_tokens as ~4G and never bound the append
     jge .main_done                            ; When the token output buffer is full, encoding stops and returns the emitted prefix.
     mov [r14 + rcx*DWORD_SCALE], eax          ; tokens[n_tokens] = token id, fall-through: room for one more
     inc enc_n_tokens                          ; n_tokens++
%endmacro
     global encode                             ; export encode entry point
; ============================================================================
encode:
; ============================================================================
     SF_START encode, rbx, rsi, rdi, r12, r13, r14, r15 ; begin encode stack frame
     SF_ARRAY enc_str_stack, ENC_STACK_BUF     ; 4KB stack buffer, used when str_buf_size <= ENC_STACK_BUF
     SF_ARRAY enc_special, ENC_SPECIAL_BUF     ; special_token scratch: 64 copied bytes + NUL at index 64 = 65 used of ENC_SPECIAL_BUF, 68 ; the scan bound at :230 caps the copy at 64 bytes
     SF_DWORD enc_n_tokens                     ; running token count
     SF_DWORD enc_loop_limit                   ; n_tokens - 1 cached once per merge iteration
     SF_DWORD enc_best_idx                     ; best merge index
     SF_QWORD enc_str_buf_size                 ; byte count for str_buffer, 2*max_token_len + overhead
     SF_DWORD enc_special_end                  ; special-token end position, saved across ht_find
     SF_QWORD enc_str_dyn                      ; 0=stack buffer, non-0=mdb_alloc_mem pointer
     SF_QWORD enc_buf_end                      ; str_buffer bound: base + size - 1, NUL slot reserved, SWAR dest bound
     SF_QWORD enc_nptr                         ; parked n_tokens* output pointer, r15 repurposed as merge_scores base in Phase 2
     SF_DWORD enc_bf_i                         ; byte-fallback loop index, preserved across the ht_find inside the loop
     SF_QWORD enc_edges                        ; edge-record cache base, (n_tokens-1) x 8-byte (id, score) records, 0 = no cache
     SF_COMMIT                                 ; Commit encode stack frame
     mov qword enc_str_dyn, 0                  ; initialize to 0, stack buffer mode, no free needed
     mov qword enc_edges, 0                    ; initialize to 0, no edge cache until the merge phase allocates one
     mov r12, rcx                              ; r12 = Tokenizer* t
     mov r13, rdx                              ; r13 = text input cursor
     mov r14, r8                               ; r14 = tokens[] output array
     mov r15, r9                               ; r15 = n_tokens* output pointer
     mov enc_nptr, r9                          ; park n_tokens* in frame slot, r15 becomes merge_scores base in Phase 2
     xor r15d, r15d                            ; str_len = 0, UTF-8 codepoint cursor; r15's n_tokens* value is parked above
     mov ebx, [rbp + ARG5_OFF]                 ; ebx = max_tokens, 5th stack arg; SIGNED compare at :70, a value below 1, positive or negative, yields *n_tokens = 0, matching runq.c encode, 's early return
     mov eax, [r12 + TOK_MAX_TOKEN_LEN]        ; eax = t->max_token_length, 32-bit load zero-extends into rax
     lea rax, [rax + rax + STRBUF_OVRHD]       ; str_buf_size = 2*max_token_length + overhead, 2 bytes/char worst case + slack
     mov enc_str_buf_size, rax                 ; save for allocation, qword slot
     cmp rax, ENC_STACK_BUF                    ; fits in stack buffer?
     jbe .use_stack_buf                        ; yes, use stack
     mov rcx, rax                              ; arg1: byte count, rax still holds the size, cmp/jbe above don't overwrite it
     call mdb_alloc_mem                        ; allocate str_buffer, oversized token, stack buffer too small
     mov enc_str_dyn, rax                      ; save for cleanup, non-zero = allocated
     mov rsi, rax                              ; rsi = str_buffer, heap pointer
     jmp .buf_ready                            ; skip stack path
.use_stack_buf:
     lea rsi, enc_str_stack                    ; rsi = str_buffer, stack buffer
.buf_ready:
     mov rax, enc_str_buf_size                 ; rax = str_buf_size, preserved across both alloc paths
     lea rax, [rsi + rax - DN_NUL_SLOT]        ; rax = base + size - 1, bound, NUL slot reserved
     mov enc_buf_end, rax                      ; park bound in frame slot, stable across calls
; Initialize token count
     mov enc_n_tokens, 0                       ; n_tokens = 0, single store, eax unused until the movzx
     mov edi, [r12 + TOK_IS_SP]                ; cache SP flag in rdi, loop-invariant through Phase 1;
; reloaded at the special-token rejoin, rep movsb overwrites rdi
     test edi, edi                             ; is SentencePiece?
     je .main_loop                             ; no, skip prefix
     cmp byte [r13], 0                         ; empty text?
     je .main_loop                             ; yes: no dummy prefix, reference checks text != ""
; Store _ UTF-8 bytes, E2 96 81 + null in str_buffer for lookup
     mov dword [rsi], SP_UTF8_LO               ; E2 96 81 00, _ + null terminator
     mov rcx, rsi                              ; rcx = key string
     xor edx, edx                              ; rdx = NULL, no length needed
     call ht_find_or_fatal                     ; rax = data ptr, mandatory for SP models: silent skip would mis-tokenize
     mov eax, [rax]                            ; eax = _ token id
     ENC_APPEND_TOKEN                          ; append _ prefix token, check store + count bump
.main_loop:
     movzx eax, byte [r13]                     ; al = *c, current byte
     test al, al                               ; null terminator, end of input ?
     jz .main_done                             ; yes, exit main loop
     cmp al, 0x80                              ; ASCII fast path: a byte < 0x80 is neither continuation nor multi-byte leading
     jb .new_codepoint                         ; ASCII: reset + space check, then store
     mov r8d, eax                              ; copy the byte for the class test, only 0x80..0xFF reaches here
     and r8d, ENC_CONT_MASK                    ; keep the top 2 bits
     cmp r8d, ENC_CONT_TAG                     ; continuation byte, 0b10xxxxxx ?
     je .append                                ; yes: keep accumulating, no cursor reset
.new_codepoint:                                ; a UTF-8 leading byte (0xC0..0xFF) falls through to the reset
     xor r15d, r15d                            ; a UTF-8 leading byte or ASCII byte starts a new code point
     cmp al, 0x20                              ; is it a space?
     je .sp_space_check                        ; yes: maybe SentencePiece rewrites it to U+2581
.append:
.store_byte:
     mov [rsi + r15], ax                       ; byte + NUL terminator in one 16-bit store, ah = 0 from the movzx above
     inc r15d                                  ; str_len++
     cmp al, '<'                               ; special token opening bracket?
     je .special_check                         ; yes: scan forward for a closing '>'
.single_char:
     movzx ecx, byte [r13 + ENC_PEEK_NEXT]     ; safe: c[0] != 0, so c[1] is at worst the NUL
     and ecx, ENC_CONT_MASK                    ; keep the top 2 bits of the peeked byte
     cmp ecx, ENC_CONT_TAG                     ; next byte a continuation byte?
     jne .lookup                               ; no: the codepoint is complete
     cmp r15d, ENC_CODEPOINT_MAX               ; str_len < 4 cap, reference: "avoid overrunning str_buffer"
     jl .next_char                             ; incomplete: keep accumulating, no lookup yet
.lookup:
     mov rcx, rsi                              ; rcx = key, accumulated codepoint string
     xor edx, edx                              ; rdx = NULL, no length needed
     call ht_find                              ; rax = data ptr or NULL
     test rax, rax                             ; found in vocab?
     jz .byte_fallback                         ; miss: encode the accumulated bytes through the aliases
     mov eax, [rax]                            ; eax = token id
.add_single_token:
; Add codepoint token
     ENC_APPEND_TOKEN                          ; append the codepoint token, check store + count bump
     xor r15d, r15d                            ; After the code point is complete, reset its byte count.
.next_char:
     inc r13                                   ; c++, advance to next input byte
     jmp .main_loop                            ; continue Phase 1
.sp_space_check:
; SentencePiece: convert space, 0x20 to _, U+2581 = E2 96 81 before lookup
     test edi, edi                             ; SentencePiece model?, cached flag, no memory operand
     je .store_byte                            ; no: fall back to storing the raw space byte
; Store _ UTF-8 bytes + null in str_buffer, then lookup
     mov dword [rsi], SP_UTF8_LO               ; E2 96 81 00, _ + null terminator
     mov r15d, 3                               ; str_len = 3 bytes
     jmp .lookup                               ; complete codepoint: skip the peek and the '<' check
.byte_fallback:
     mov enc_bf_i, 0                           ; i = 0
.bf_loop:
     mov r9d, enc_bf_i                         ; i, reload: ht_find overwrites r9
     cmp r9d, r15d                             ; all bytes encoded?
     jge .bf_done                              ; yes
     movzx eax, byte [rsi + r9]                ; str_buffer[i]
     mov word [rsi + r15 + ENC_KEY_SCRATCH], ax ; scratch 2-byte key, byte + NUL past the accumulated
; codepoint: str_len <= 4 and str_buf_size >= 2*mtl+8,
; so the scratch key is always within the buffer's headroom
     lea rcx, [rsi + r15 + ENC_KEY_SCRATCH]    ; rcx = single-byte key
     xor edx, edx                              ; rdx = NULL, no length needed
     call ht_find_or_fatal                     ; hits the byte-fallback alias, or a real 1-byte vocab key; miss = corrupt DB, the helper names the byte
     mov eax, [rax]                            ; eax = byte token id
     ENC_APPEND_TOKEN                          ; append, overwrites ecx/flags only; ebx/r14 active
     inc enc_bf_i                              ; i++, frame slot, preserved across the next ht_find
     jmp .bf_loop                              ; next byte of the codepoint
.bf_done:
     xor r15d, r15d                            ; end of codepoint
     jmp .next_char                            ; continue Phase 1
.special_check:
; Scan forward for '>', max 64 bytes
     mov ecx, 1                                ; k = 1: skip the known '<' at k=0, scan starts at the byte AFTER it
.scan_special:
     movzx eax, byte [r13 + rcx]               ; al = c[k], single load, zero-extended
     test al, al                               ; hit end of input before '>'?
     jz .single_char                           ; yes, not a special token
     cmp al, '>'                               ; found closing bracket?
     je .found_close                           ; yes, check if it's a known token
     inc ecx                                   ; advance offset
     cmp ecx, MAX_SPECIAL_TOKEN_LEN            ; exceeded max special token length?
     jl .scan_special                          ; no, continue scanning
     jmp .single_char                          ; too many bytes without '>' -> not special
.found_close:
; ecx = end_of_token_pos, offset of '>' from c[0]
; Copy c[0..end_of_token_pos] to special_token buffer
     mov enc_special_end, ecx                  ; enc_special_end = end_of_token_pos, safe across ht_find, non-volatile slot
     inc ecx                                   ; ecx = bytes to copy, including '>'
     lea rdi, enc_special                      ; rdi = dest, special_token
     push rsi                                  ; save str_buffer pointer
     mov rsi, r13                              ; rsi = source, text cursor
     rep movsb                                 ; copy ecx bytes to special_token
     mov byte [rdi], 0                         ; null terminate
     pop rsi                                   ; restore str_buffer pointer
; Look up special_token in hash table, ht_find
     lea rcx, enc_special                      ; rcx = key, special_token
     xor edx, edx                              ; rdx = NULL, no length needed
     call ht_find                              ; rax = data ptr or NULL
     test rax, rax                             ; token found in vocab?
     jz .special_miss                          ; not found, fall through to single-char lookup, reload SP flag first
; Special token found! Add to tokens array
     mov eax, [rax]                            ; eax = token id from hash table
     ENC_APPEND_TOKEN                          ; append special token, check store + count bump
; Advance cursor past the special token and continue, inlined .next_char:
; the +1 for the loop's c++ is folded into end_of_token_pos+1
     mov r9d, enc_special_end                  ; r9d = end_of_token_pos, restore from non-volatile slot, zero-extends to r9
     lea r13, [r13 + r9 + ENC_SPECIAL_PAST_GT] ; c += end_of_token_pos + 1, past '>', the loop's inc is folded into the advance
     xor r15d, r15d                            ; the special token consumed the codepoint: reset str_len, reference resets after every emitted token
     mov edi, [r12 + TOK_IS_SP]                ; reload SP flag, rep movsb above overwritten rdi
     jmp .main_loop                            ; continue Phase 1
.special_miss:
; not a known special token: the rep movsb above overwritten rdi, reload the
; cached SP flag before rejoining the Phase-1 flow at the single-char lookup
     mov edi, [r12 + TOK_IS_SP]                ; restore cached SentencePiece flag
     jmp .single_char                          ; fall through to single-char lookup
.main_done:
     mov rbx, [r12]                            ; rbx = t->vocab, parked for the merge phase, max_tokens is unused after Phase 1; non-volatile rbx preserved across .decode_vocab AND ht_find
     mov r15, [r12 + TOK_MERGE_SCORES]         ; r15 = merge_scores base for the whole merge phase, loop-invariant; n_tokens* parked in enc_nptr
     cmp enc_n_tokens, 2                       ; at least one adjacent pair to cache?
     jl .merge_done                            ; 0 or 1 tokens, no merging possible and no edge cache
; Build the edge cache: one (merged id, score bits) record per adjacent pair,
; so each merge pass scans records instead of re-decoding and re-hashing every
; pair. Only the two edges touched by a merge are ever recomputed.
     mov eax, enc_n_tokens                     ; eax = N
     dec eax                                   ; eax = N-1 edge records
     mov enc_loop_limit, eax                   ; fill bound = N-1, the scan reuses the same slot each pass
     shl rax, 3                                ; * EDGE_STRIDE, byte count for the record block
     mov rcx, rax                              ; arg1: byte count
     call mdb_alloc_mem                        ; allocate the record block, failure FATALs inside
     mov enc_edges, rax                        ; park the record base for the whole merge phase
     xor r13d, r13d                            ; edge index = 0, non-volatile, preserved across .edge_lookup
.fill_edges:
     call .edge_refresh                        ; build and store this cached pair
     inc r13d                                  ; next edge
     cmp r13d, enc_loop_limit                  ; all N-1 initial pairs cached?
     jl .fill_edges                            ; next
.merge_loop:
     mov eax, enc_n_tokens                     ; current token count
     dec eax                                   ; number of live edges
     jle .merge_done                           ; no adjacent pair remains
     mov ecx, ENC_NEG_1E10                     ; exact baseline score bits
     vmovd xmm1, ecx                           ; keep the running score in a register
     mov r10d, -1                              ; no edge beats the baseline yet
     mov rdi, enc_edges                        ; base of the edge records
     xor r13d, r13d                            ; scan in left-to-right order
.inner_loop:
     vmovss xmm0, [rdi + r13*EDGE_STRIDE + EDGE_SCORE_OFF] ; cached score, miss carries the baseline
     vcomiss xmm0, xmm1                        ; keep the original strict ordered comparison
     jbe .next_edge                            ; keep the earlier edge on equal scores, ignore NaNs
     vmovaps xmm1, xmm0                        ; remember the new maximum
     mov r10d, r13d                            ; record its edge index
.next_edge:
     inc r13d                                  ; next edge index
     cmp r13d, eax                             ; all live edges examined?
     jl .inner_loop                            ; no, keep scanning
     mov eax, r10d                             ; winning edge index, if any
     test eax, eax                             ; did any edge win?
     js .merge_done                            ; no, tokenization is complete
     mov enc_best_idx, eax                     ; preserve the index across edge refresh
     mov ecx, [rdi + rax*EDGE_STRIDE]          ; read only the winning merged token ID
     mov [r14 + rax*DWORD_SCALE], ecx          ; replace the left token
; Shift remaining tokens left by 1
     mov ecx, enc_n_tokens                     ; ecx = n_tokens
     sub ecx, eax                              ; ecx = n_tokens - best_idx
     sub ecx, 2                                ; ecx = count of elements to shift
     jle .shift_done                           ; nothing to shift, best_idx was last pair
     push rsi                                  ; save str_buffer pointer
     lea rsi, [r14 + rax*DWORD_SCALE + SKIP2_TOKEN] ; rsi = src = &tokens at best_idx+2
     lea rdi, [r14 + rax*DWORD_SCALE + SKIP1_TOKEN] ; rdi = dst = &tokens at best_idx+1
     rep movsd                                 ; shift dwords left by one position
     pop rsi                                   ; restore str_buffer pointer
.shift_done:
; Edge records: pairs (i,i+1) and (i+1,i+2) changed, drop records i and i+1;
; surviving record k >= i+2 (old pair (k,k+1)) moves down to slot k-1, its pair
; is unchanged by the shift. Forward copy, dest below source.
     mov ecx, enc_n_tokens                     ; ecx = old n_tokens, not yet decremented
     sub ecx, eax                              ; ecx = n_tokens - best_idx
     sub ecx, 3                                ; ecx = records to shift down, negative/zero when best_idx was at the end
     jle .edge_shift_done                      ; nothing to shift
     push rsi                                  ; save str_buffer pointer
     mov rdi, enc_edges                        ; rdi = record base, reloaded after the token shift overwrote it
     lea rsi, [rdi + rax*EDGE_STRIDE + EDGE_SKIP2] ; rsi = src = &record[best_idx+2]
     lea rdi, [rdi + rax*EDGE_STRIDE + EDGE_SKIP1] ; rdi = dst = &record[best_idx+1]
     rep movsq                                 ; shift whole 8-byte records left by one slot
     pop rsi                                   ; restore str_buffer pointer
.edge_shift_done:
     dec enc_n_tokens                          ; n_tokens--, one fewer token
; Recompute only the two edges a merge can change: (i-1) = (prev, merged) and
; (i) = (merged, next), both exist only within the shrunk record range.
     test eax, eax                             ; best_idx == 0?
     jz .edge_recomp_right                     ; no left neighbor
     mov r13d, eax                             ; r13d = best_idx
     dec r13d                                  ; r13d = edge index best_idx-1
     call .edge_refresh                        ; recompute and store the changed edge
.edge_recomp_right:
     mov eax, enc_best_idx                     ; reload best_idx, the left lookup's return overwrote eax
     mov ecx, enc_n_tokens                     ; ecx = new n_tokens
     dec ecx                                   ; ecx = new record count limit
     cmp eax, ecx                              ; best_idx < new n_tokens - 1?
     jge .edge_recomp_done                     ; right neighbor out of range
     mov r13d, eax                             ; r13d = edge index best_idx
     call .edge_refresh                        ; recompute and store the changed edge
.edge_recomp_done:
     jmp .merge_loop                           ; continue with remaining tokens
.merge_done:
     mov eax, enc_n_tokens                     ; eax = final token count
     mov rcx, enc_nptr                         ; rcx = n_tokens*, parked frame slot, r15 holds merge_scores base now
     mov [rcx], eax                            ; *n_tokens = n_tokens
; CLEANUP_YMM before any API call, mdb_free_mem path may use AVX internally
     CLEANUP_YMM                               ; Clear YMM before API call
; Free the edge-record cache if it was allocated
     cmp enc_edges, 0                          ; was the edge cache allocated?
     je .no_edge_free                          ; 0/1-token input never allocated one
     mov rcx, enc_edges                        ; arg1: record block pointer
     call mdb_free_mem                         ; free the record block, non-NULL
.no_edge_free:
; Free allocated buffer if needed
     cmp enc_str_dyn, 0                        ; was the buffer allocated dynamically?
     je .return                                ; no, skip free
     mov rcx, enc_str_dyn                      ; arg1: buffer pointer
     call mdb_free_mem                         ; free the buffer, non-NULL: heap mode selected above
.return:
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore regs and return
.edge_refresh:
; Extract pair (tokens[r13], tokens[r13+1]) into str_buffer, hash-lookup the
; merged token, then write its cached record. IN: r13d = edge index,
; r14 = tokens, rbx = vocab, rsi = str_buffer, r15 = merge_scores.
; Writes the ID and score into the record. Clobbers rax/rcx/rdx/r8/r9/rdi/r10/r11/xmm0.
     mov ecx, [r14 + r13*DWORD_SCALE]          ; ecx = tokens[edge]
     mov r9, [rbx + rcx*PTR_SCALE]             ; r9 = vocab[tokens[edge]] string pointer, rbx = parked vocab base
     mov rdi, rsi                              ; rdi = dest, str_buffer
     call .decode_vocab                        ; copy with <0xNN> decoding; overwrites rax,rcx,rdx,r8,r9,rdi,r10,r11, NOT rbx, vocab cache preserved ; rsi preserved, borrowed and restored inside
     mov ecx, [r14 + r13*DWORD_SCALE + SKIP1_TOKEN] ; ecx = tokens[edge+1], r13 non-volatile, preserved across decode_vocab
     mov r9, [rbx + rcx*PTR_SCALE]             ; r9 = vocab string pointer for next token, rbx = parked vocab base
     call .decode_vocab                        ; append with <0xNN> decoding; overwrites rax,rcx,rdx,r8,r9,rdi,r10,r11, NOT rbx, vocab cache preserved ; rsi preserved, borrowed and restored inside
     mov byte [rdi], 0                         ; null-terminate for ht_find lookup
     mov rcx, rsi                              ; rcx = key, concatenated string
     xor edx, edx                              ; rdx = NULL, no length needed
     call ht_find                              ; rax = data ptr or NULL, overwrites volatiles only
     test rax, rax                             ; merge token exists?
     jz .edge_invalid                          ; not found: sentinel record
     mov eax, [rax]                            ; eax = merge token id
     vmovss xmm0, [r15 + rax*DWORD_SCALE]      ; xmm0 = merge_scores[id], r15 = parked merge_scores base
.edge_store:
     mov rdx, enc_edges                        ; edge record base
     mov [rdx + r13*EDGE_STRIDE], eax          ; store the merged ID or miss sentinel
     vmovss [rdx + r13*EDGE_STRIDE + EDGE_SCORE_OFF], xmm0 ; exact score bits
     ret                                       ; cached edge is ready
.edge_invalid:
     mov eax, -1                               ; no merged token for this pair
     mov edx, ENC_NEG_1E10                     ; a miss can never beat the scan baseline
     vmovd xmm0, edx                           ; exact sentinel bits
     jmp .edge_store                           ; use the same record writer
.decode_vocab:
     cmp byte [r9], '<'                        ; first char must be '<'
     jne .decode_normal                        ; not byte-fallback, copy normally
     cmp byte [r9 + DECODE_OFF_1], '0'         ; second char must be '0', a NUL here fails too
     jne .decode_normal                        ; not byte-fallback, copy normally
     cmp byte [r9 + DECODE_OFF_2], 'x'         ; third char must be 'x'
     jne .decode_normal                        ; not byte-fallback, copy normally
     cmp byte [r9 + DECODE_OFF_3], 0           ; probe: string >= 4 chars?, else +5 read could leave the allocation
     je .decode_normal                         ; shorter than a byte-fallback token, literal text
     cmp byte [r9 + DECODE_OFF_4], 0           ; probe: string >= 5 chars?, else +5 read could leave the allocation
     je .decode_normal                         ; shorter than a byte-fallback token, literal text
     cmp byte [r9 + DECODE_OFF_5], '>'         ; sixth char must be '>'
     jne .decode_normal                        ; not byte-fallback, copy normally
     cmp byte [r9 + DECODE_OFF_NULL], 0        ; ensure exactly 6 bytes, null at offset 6
     jne .decode_normal                        ; extra chars after '>', not a byte-fallback token
     lea r11, [rel hex_val_lut]                ; existing 256-byte shared table
     movzx ecx, byte [r9 + DECODE_OFF_3]       ; high ASCII nibble
     movzx edx, byte [r11 + rcx]               ; 0 invalid, 1..16 valid
     test edx, edx                             ; nonzero proves valid hex
     jz .decode_normal                         ; invalid hi hex -> literal text
     dec edx                                   ; high nibble 0..15
     shl edx, 4                                ; high nibble to the high bits
     movzx ecx, byte [r9 + DECODE_OFF_4]       ; low ASCII nibble
     movzx eax, byte [r11 + rcx]               ; 0 invalid, 1..16 valid
     test eax, eax                             ; nonzero proves valid hex
     jz .decode_normal                         ; invalid lo hex -> literal text
     dec eax                                   ; low nibble 0..15
     or eax, edx                               ; combine nibbles -> decoded byte
     stosb                                     ; write single byte to dest, advance rdi
     ret                                       ; return to caller
.decode_normal:
     push rsi                                  ; save str_buffer base, active in caller
     mov rsi, r9                               ; rsi = source cursor
     mov r10, DN_SWAR_REP1                     ; rep,1 per byte lane, hoisted
     mov r11, DN_SWAR_FLAG80                   ; per-lane 0x80 flag mask, hoisted
.dn_gate:
     mov rax, rsi                              ; page-gate probe on the source cursor
     and rax, DN_PAGE_MASK                     ; in-page offset of the next 8-byte read
     cmp rax, DN_PAGE_EDGE                     ; would the read cross a page edge?
     ja .dn_byte                               ; page-edge crossing -> fall to the safe byte loop
     mov rax, enc_buf_end                      ; dest bound, frame slot, stable across the call
     sub rax, rdi                              ; room from cursor to bound
     cmp rax, DN_QWORD_BYTES                   ; room for the full 8-byte store?
     jb .dn_byte                               ; no -> byte loop
     mov rcx, [rsi]                            ; w = 8 source bytes
     mov rdx, rcx                              ; rdx = copy of w, becomes w - rep,1
     sub rdx, r10                              ; w - rep,1, hoisted constant
     andn rdx, rcx, rdx                        ; (~w) & (w-rep,1): raw zero-detect bits; BMI1, 3 bytes shorter than not+and
     and rdx, r11                              ; keep ONLY the flag bits, and sets ZF: no NUL?
     jz .dn_store8                             ; no NUL in window -> full advance
     tzcnt rdx, rdx                            ; first flag bit, 8k+7, mask is flag-only, index is exact; BMI1 native on Zen 4
     shr rdx, 3                                ; rdx = k = NUL offset in window
     mov [rdi], rcx                            ; store all 8 bytes, bytes past NUL are scratch
     add rdi, rdx                              ; rdi points AT the stored NUL, byte-loop contract
     jmp .dn_done                              ; piece complete, rdi at NUL, contract
.dn_store8:
     mov [rdi], rcx                            ; store the 8 non-NUL bytes
     add rdi, DN_QWORD_BYTES                   ; advance dest cursor
     add rsi, DN_QWORD_BYTES                   ; advance source cursor
     jmp .dn_gate                              ; next window
.dn_byte:
     lodsb                                     ; al = [rsi++], load + advance in one instruction
     stosb                                     ; [rdi++] = al, store + advance in one instruction
     test al, al                               ; null terminator?
     jnz .dn_byte                              ; continue until null
     dec rdi                                   ; back up over null, next string overwrites it
.dn_done:
     pop rsi                                   ; restore str_buffer base
     ret                                       ; return to caller
; ============================================================================
%undef PTR_SCALE
%undef ARG5_OFF
%undef STRBUF_OVRHD
%undef SKIP1_TOKEN
%undef SKIP2_TOKEN
%undef DECODE_OFF_1
%undef DECODE_OFF_2
%undef DECODE_OFF_3
%undef DECODE_OFF_4
%undef DECODE_OFF_5
%undef DECODE_OFF_NULL
%undef SP_UTF8_LO
%undef ENC_SPECIAL_PAST_GT
%undef DN_SWAR_REP1
%undef DN_SWAR_FLAG80
%undef DN_PAGE_MASK
%undef DN_PAGE_EDGE
%undef DN_QWORD_BYTES
%undef DN_NUL_SLOT
%undef ENC_SPECIAL_BUF
%undef MAX_SPECIAL_TOKEN_LEN
%undef ENC_CONT_MASK
%undef ENC_CONT_TAG
%undef ENC_CODEPOINT_MAX
%undef ENC_KEY_SCRATCH
%undef ENC_PEEK_NEXT
%undef ENC_NEG_1E10
%undef ENC_STACK_BUF
; SF frame slots leave scope with the file, house convention; the typed names
; carry a _raw alias each, see v7_apicall.inc, while the two SF_ARRAY
; slots have only the plain name
%undef enc_bf_i
%undef enc_bf_i_raw
%undef enc_nptr
%undef enc_nptr_raw
%undef enc_buf_end
%undef enc_buf_end_raw
%undef enc_str_dyn
%undef enc_str_dyn_raw
%undef enc_special_end
%undef enc_special_end_raw
%undef enc_str_buf_size
%undef enc_str_buf_size_raw
%undef enc_best_idx
%undef enc_best_idx_raw
%undef enc_loop_limit
%undef enc_loop_limit_raw
%undef enc_n_tokens
%undef enc_n_tokens_raw
%undef enc_special
%undef enc_str_stack
