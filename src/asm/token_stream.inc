; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define ST_DWORD_SCALE 4                       ; sizeof,dword for array indexing
%define ST_IDX_BIAS 1                          ; index bias: last byte position of a length-1 run
%if (SL_STREAM_BUF_SIZE & 31) != 0
%error Rewrite buffers must hold whole 32-byte blocks ; the vector scan reads 32-byte blocks, a partial block would read past the allocation
%endif
%if (sl_stream_tok_count - sl_stream_str_len) != 4
%error "token_stream: sl_stream_str_len and sl_stream_tok_count are no longer adjacent dwords, v7s_stream_init_turn's single qword zeroing store must become two stores" ; assembly-time guard: the qword store below zeroes BOTH counters, a field inserted between them must turn it back into two stores
%endif
v7s_stream_init_turn:
     mov [rel sl_stream_base_pos], ecx         ; record base position
     mov qword [rel sl_stream_str_len], 0      ; zero BOTH counters in one store, str_len + tok_count are adjacent dwords, guarded above
     ret                                       ; return to caller
v7s_stream_append_token:
; Use volatile registers only, leaf function, no API calls, eliminates 6 push/pop
     mov eax, [rel sl_stream_tok_count]        ; eax = current token index
     cmp eax, SL_STREAM_TOK_MAX                ; the two per-token arrays are indexed by tok_count and NOTHING else bounds them
     jae .append_tok_full                      ; yes: FATAL, this is a SEPARATE limit from the byte check below, and neither implies the other
; Record token ID in history array
     lea r10, [rel sl_stream_tok_history]      ; r10 = history base
     mov dword [r10 + rax*ST_DWORD_SCALE], ecx ; history[count] = token_id, ecx = arg1, still intact
; Record start byte offset in offset array
     lea r10, [rel sl_stream_tok_offset]       ; r10 = offset base
     mov ecx, [rel sl_stream_str_len]          ; ecx = current str_len, byte offset
     mov dword [r10 + rax*ST_DWORD_SCALE], ecx ; offset[count] = str_len
     lea r9d, [rcx + r8]                       ; r9d = new str_len, str_len + piece_len
     cmp r9d, SL_STREAM_BUF_SIZE               ; the NUL at sl_stream_str_buf[str_len] must stay inside the 1 MiB buffer
     jae .append_full                          ; no: FATAL, never silently truncate the stream
     mov [rel sl_stream_str_len], r9d          ; publish new_str_len now: r9d becomes the copy index below and nothing reads the global until .append_done
     mov r11d, eax                             ; r11d = token index, preserves across loop
     lea rax, [rel sl_stream_str_buf]          ; rax = str_buf dest base
     add rax, rcx                              ; rax = dest = str_buf + str_len
     test r8d, r8d                             ; piece_len == 0?
     jz .append_done                           ; yes: skip copy loop
     lea r9d, [r8 - ST_IDX_BIAS]               ; r9d = final byte index, piece_len >= 1 proved by the guard, so no underflow
.append_copy:
     mov cl, byte [rdx + r9]                   ; load piece[i] without clobbering rax destination pointer
; r11d = token index and rdx = source pointer remain live. r8d is dead after loop setup.
     mov byte [rax + r9], cl                   ; str_buf[str_len + i] = piece[i]
     dec r9d                                   ; i--, 0 -> -1 terminates via SF below
     jns .append_copy                          ; more to copy, dec+jns macro-fuse: 2 loop-control uops
.append_done:
; str_len was published above from r9d, str_len + piece_len ; zero-length
; pieces published the unchanged length. r11d = the token index.
     inc r11d                                  ; r11d = new token count, register ALU, breaks the RMW store-forwarding chain
     mov [rel sl_stream_tok_count], r11d       ; publish new token count
     ret                                       ; return to caller
.append_full:
     FATAL "SectorLogo: generated stream exceeded the 1 MiB stream buffer"
.append_tok_full:
     FATAL "SectorLogo: generated stream exceeded the 256K token history buffer"
v7s_stream_compute_rewrite_diff:
     mov r8d, [rel sl_stream_str_len]          ; old length
     mov r9d, [rel sl_stream_str2_len]         ; new length
     cmp r9d, r8d                              ; compare the two logical lengths
     cmova r9d, r8d                            ; r9d = minimum length
     xor ecx, ecx                              ; byte offset in both fixed-size buffers
     lea r10, [rel sl_stream_str_buf]          ; original bytes
     lea r11, [rel sl_stream_str_buf2]         ; rewritten bytes
     test r9d, r9d                             ; empty common extent?
     jz .lcp_min_end                           ; no vector load needed
.lcp_vec:
     vmovdqu ymm0, [r10 + rcx]                 ; 32 bytes from the old buffer
     vpcmpeqb ymm0, ymm0, [r11 + rcx]          ; set one mask bit for each equal byte
     vpmovmskb eax, ymm0                       ; equality mask
     inc eax                                   ; carry through the leading equal run clears it and sets the first mismatch bit, zero only when all equal
     jnz .lcp_vec_diff                         ; locate the first difference
     add ecx, 32                               ; advance one vector
     cmp ecx, r9d                              ; another vector start below the logical bound?
     jb .lcp_vec                               ; yes, read the next allocated block
     mov ecx, r9d                              ; the common prefix ends at the shorter length
     jmp .lcp_min_end                          ; join at the logical boundary
.lcp_vec_diff:
     tzcnt eax, eax                            ; byte offset within this vector
     add ecx, eax                              ; candidate mismatch position
     cmp ecx, r9d                              ; ignore differences outside the logical extent
     cmova ecx, r9d                            ; clamp to the shorter length
.lcp_min_end:
     vzeroupper                                ; leave upper vector state clean
; ecx == min_len: append-case iff old is the exhausted one
     cmp ecx, r8d                              ; LCP >= old length?
     jae .diverge_at_end                       ; yes, new is pure append, or equal
.lcp_middle:
; ecx < old_len: find the containing token via sorted start offsets
     mov r8d, ecx                              ; first differing byte, x < old string length
     lea r10, [rel sl_stream_tok_offset]       ; sorted token-start offsets, including duplicate offsets for empty pieces
     xor ecx, ecx                              ; lo = 0
     mov r9d, [rel sl_stream_tok_count]        ; hi = number of old tokens
.find_token:
     cmp ecx, r9d                              ; upper_bound(offset[], x)
     jae .found_token                          ; lo == hi: search complete
     lea eax, [rcx + r9]                       ; lo + hi
     shr eax, 1                                ; midpoint, no overflow (max 256K entries)
     cmp dword [r10 + rax*ST_DWORD_SCALE], r8d ; offset[mid] > x?
     ja .search_left                           ; yes: move hi left
     lea ecx, [rax + ST_IDX_BIAS]              ; lo = mid + 1, one address-gen op instead of copy+inc
     jmp .find_token                           ; continue search
.search_left:
     mov r9d, eax                              ; hi = mid
     jmp .find_token                           ; continue search
.found_token:
     dec ecx                                   ; rightmost token start <= mismatch byte
     mov [rel sl_stream_tok_count], ecx        ; rewind token count
     mov r8d, [r10 + rcx*ST_DWORD_SCALE]       ; corresponding byte offset
     mov [rel sl_stream_str_len], r8d          ; str_len = byte offset
.diverge_at_end:
; Middle path: rewound tok_count == T_diverge. Append path: current tok_count == T_diverge.
; Either way this single load produces the return value.
     mov r11d, [rel sl_stream_tok_count]       ; r11d = T_diverge, r11d is dead here: last read was the LCP tail compare
; fall through to .suffix_calc, str_len already correct on both paths
.suffix_calc:
; After rewind, or skip, sl_stream_str_len = byte offset of T_diverge
; The suffix in str_buf2 starts at the same byte offset
     mov eax, [rel sl_stream_str_len]          ; eax = byte offset of T_diverge
     lea rdx, [rel sl_stream_str_buf2]         ; rdx = buf2 base
     add rdx, rax                              ; rdx = buf2 + byte_offset, suffix ptr
     mov eax, r11d                             ; eax = T_diverge, return value
     ret                                       ; return to caller
%undef ST_DWORD_SCALE
%undef ST_IDX_BIAS
