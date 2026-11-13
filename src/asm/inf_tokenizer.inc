; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; Local constants for offset arithmetic
%define PTR_SCALE 8                            ; Bytes per pointer, x64
%define FK_FLOAT_BYTES 4                       ; Bytes per float element
%define TOK_DW3_SCALE 2                        ; LEA scale for *3, reg + reg*2
%define STOP_HDR_SIZE 4                        ; Bytes of count dword before stop ID array
%define PER_VOCAB_BYTES 16                     ; ptr_array,8 + probindex,8; merge_scores alias read-only MicroDB
%define VOCAB_FAST_STORE_BYTES 16              ; bytes written by the vocab-copy fast path, one unaligned 16-byte store
%define TOK_SEQ_QWORD_SCALE 8                  ; SIB scale: seq_len * 8 bytes of token buffers
%define SP_LEAD_BYTE 0xE2                      ; byte 0 of the U+2581 pattern (E2 96 81), necessary condition, rejects on one byte before the 3-byte compare
%define SP_TRAIL_WORD_OFF 1                    ; byte offset to the U+2581 trailing two bytes
%define SP_TRAIL_WORD 0x8196                   ; bytes 1..2 of U+2581, little-endian word

%macro COPY_TEMPLATE 2
     lea rcx, [rel %1]                         ; rcx = template string key
     lea rdx, inf_db_len                       ; out_len = &inf_db_len, slot name IS the [rbp-N] operand
     call ht_find                              ; Find template entry, may be absent
     test rax, rax                             ; Check if entry exists
     jz %%done                                 ; Skip if missing, optional
     mov rcx, inf_db_len                       ; rcx = full 64-bit entry length, includes the NUL
     dec rcx                                   ; rcx = string byte count, NUL excluded; the copy count
     mov rsi, rax                              ; rsi = source string
     lea rdi, %2                               ; rdi = destination buffer, any addressing
     rep movsb                                 ; Copy slen bytes; rsi += slen, rdi += slen
     mov byte [rdi], 0                         ; Null-terminate immediately after copy
%%done:
%endmacro
; === Step 14: Lock microdb to readonly ===
     mov ecx, 1                                ; arg1: readonly = true
     call db_set_readonly                      ; Prevent accidental microdb writes
; === Step 15b: Get tokenizer header from microdb ===
     lea rcx, [rel inf_key_tokenizer_hdr]      ; rcx = "__sys_tokenizer_hdr"
     lea rdx, inf_db_len                       ; out_len = &inf_db_len, slot name IS the [rbp-N] operand
     call ht_find_or_fatal                     ; Find tokenizer_hdr entry, FATAL with key name on miss
     mov ecx, [rax]                            ; +0: max_token_length
     mov inf_hdr_max_tok_len, ecx              ; Cache max_token_length
     mov ecx, [rax + HDR_BOS_ID]               ; +4: bos_token_id
     mov inf_hdr_bos_id, ecx                   ; Cache bos_token_id
     mov ecx, [rax + HDR_EOS_ID]               ; +8: eos_token_id
     mov inf_hdr_eos_id, ecx                   ; Cache eos_token_id
     lea rcx, [rel inf_key_stop_token_ids]     ; rcx = "__sys_stop_token_ids"
     lea rdx, inf_db_len                       ; out_len = &inf_db_len, slot name IS the [rbp-N] operand
     call ht_find_or_fatal                     ; Find stop_token_ids entry, FATAL with key name on miss
     mov ecx, [rax]                            ; ecx = count of stop token IDs, entry layout is the import's contract
     mov [rel g_stop_id_count], ecx            ; Set global count
     lea rsi, [rax + STOP_HDR_SIZE]            ; rsi = first stop ID in array
     lea rdi, [rel g_stop_ids]                 ; rdi = global stop IDs array
     rep movsd                                 ; Copy stop ID dwords to globals, ecx = count, no byte-count shift needed
; === Step 16: Scan vocab_bulk to compute total string bytes ===
     lea rcx, [rel inf_key_vocab_bulk]         ; rcx = "__sys_vocab_bulk"
     lea rdx, inf_db_len                       ; out_len = &inf_db_len, slot name IS the [rbp-N] operand
     call ht_find_or_fatal                     ; Find vocab_bulk entry, FATAL with key name on miss
     mov rcx, inf_db_len                       ; rcx = full 64-bit blob byte length
     lea r10, [rax + rcx]                      ; r10 = blob end
     mov inf_vocab_bulk_ptr, rax               ; cache the blob base for pass 2
     mov inf_vocab_bulk_end, r10               ; cache the blob end for pass 2
     mov ecx, [rax]                            ; First dword = entry count
     mov inf_vocab_bulk_count, ecx             ; Save entry count
; Two-pass: first scan replaced by a constant-time size formula
; Blob = 4-byte count + count*(4-byte slen + slen data bytes), exactly packed
; by the importer. Destination adds one NUL per entry, so the exact flat
; storage is blob_len - 4 - 3*count = sum(slen+1).
     lea r15, [rcx + rcx*TOK_DW3_SCALE + VB_HEADER_SIZE] ; 3*count + 4, ecx holds the entry count
     mov rcx, inf_db_len                       ; full blob length
     sub rcx, r15                              ; exact flat storage including one NUL per entry
     mov inf_str_bytes, rcx                    ; save exact byte count
     mov eax, inf_vocab_size                   ; eax = vocab_size
     imul rcx, rax, PER_VOCAB_BYTES            ; rcx = vocab_size * 16, ptr_array plus probindex
     add rcx, inf_str_bytes                    ; rcx += str_buffer, 64-bit, straight from the qword slot, rax is unused here, next line rewrites it
     mov eax, inf_seq_len                      ; eax = seq_len, zero-extends to rax
     lea rcx, [rcx + rax*TOK_SEQ_QWORD_SCALE + CTX_HEADER_SIZE + TOK_SIZE + SAM_SIZE] ; rcx = total block size, seq_len*8 + all fixed headers
     call mdb_alloc_mem                        ; rax = single ctx block, header + tokenizer + sampler + scores + vocab + strings + token buffers
     mov inf_ctx_block, rax                    ; Cache pointer in SF frame slot
     mov [g_inf_ctx_block], rax                ; Track in BSS for cleanup
; === Step 18: Build Tokenizer within InferenceContext block ===
; Compute sub-block addresses
     mov r15, rax                              ; allocation result remains available
     lea r12, [r15 + CTX_HEADER_SIZE]          ; r12 = Tokenizer*, within block
     mov inf_tokenizer, r12                    ; Save tokenizer ptr
     lea r13, [r15 + CTX_HEADER_SIZE + TOK_SIZE] ; r13 = Sampler*, within block
     mov inf_sampler, r13                      ; Save sampler ptr
; Tokenizer merge scores are immutable and MicroDB storage outlives the
; context, the same direct-pool-alias contract as the model weights.
     lea rcx, [rel inf_key_merge_scores]       ; rcx = "__sys_merge_scores"
     lea rdx, inf_db_len                       ; out_len = &inf_db_len, slot name IS the [rbp-N] operand
     call ht_find_or_fatal                     ; pointer to the read-only merge_scores array
     mov [r12 + TOK_MERGE_SCORES], rax         ; zero-copy alias, valid until the model DB unloads
; Set tokenizer metadata from cached header
     mov eax, inf_vocab_size                   ; Load vocab_size for metadata set
     mov [r12 + TOK_VOCAB_SIZE], eax           ; tok->vocab_size = vocab_size
     mov eax, inf_hdr_max_tok_len              ; Load max_token_length for metadata set
     mov [r12 + TOK_MAX_TOKEN_LEN], eax        ; tok->max_token_length
     mov eax, inf_hdr_bos_id                   ; Load bos_token_id for metadata set
     mov [r12 + TOK_BOS_ID], eax               ; tok->bos_token_id
     mov eax, inf_hdr_eos_id                   ; Load eos_token_id for metadata set
     mov [r12 + TOK_EOS_ID], eax               ; tok->eos_token_id
; Vocab ptr array begins immediately after Sampler (no merge-score block)
     mov ecx, inf_vocab_size                   ; restore size for the ptr-array end calculation below
     lea r10, [r13 + SAM_SIZE]                 ; r10 = ptr_array dest
     mov [r12 + TOK_VOCAB], r10                ; tok->vocab = ptr_array
; Flat string buffer dest = ptr_array end
     lea rdi, [r10 + rcx*PTR_SCALE]            ; rdi = str_buffer dest
; Second pass over vocab_bulk: copy strings, build ptr array
     mov r10, inf_vocab_bulk_end               ; blob end, bounds pass 2; byte-identical to pass 1 because both walks run back-to-back on the single init thread, the readonly flag gates db_save only, NOT in-memory ht_add
     mov rax, inf_vocab_bulk_ptr               ; blob base
     lea rsi, [rax + VB_HEADER_SIZE]           ; rsi = p = past count header
     mov r9d, inf_vocab_bulk_count             ; r9d = remaining entries
     mov rbx, [r12 + TOK_VOCAB]                ; rbx = ptr_array write ptr, reload, r10 volatile
     mov rax, inf_str_bytes                    ; rax = flat-string byte count, for the dest-end check below
     lea r11, [rdi + rax - VOCAB_FAST_STORE_BYTES] ; r11 = last legal 16-byte store address in str_buffer, dest-end check
     xor r8d, r8d                              ; r8d = is_sp flag cache, r8 unused in this file; mirrors TOK_IS_SP
     test r9d, r9d                             ; count == 0?
     jz .vocab_copy_done                       ; Skip if empty
.vocab_copy_loop:
     mov r14d, [rsi]                           ; r14d = slen, preserved through rep movsb
     mov [rbx], rdi                            ; vocab[i] = current str_buffer pos
     lea rsi, [rsi + VB_ENTRY_HEADER]          ; rsi = string data
     test r8d, r8d                             ; already marked?, register mirror of TOK_IS_SP, no L1 load per entry
     jnz .vocab_copy_sp_done                   ; yes, nothing left to test
     cmp r14d, 3                               ; slen >= 3? A 3-byte pattern cannot fit a shorter string, and WITHOUT this the dword read below would test the NEXT entry's slen dword, which can carry 0x96/0x81 in its low bytes, that is a wrong answer, not just an over-read
     jb .vocab_copy_sp_done                    ; no, 0/1/2-byte string, cannot be U+2581
     cmp byte [rsi], SP_LEAD_BYTE              ; first byte is 0xE2?, necessary condition of U+2581, one byte load instead of the masked dword test on the overwhelming majority of entries
     jne .vocab_copy_sp_done                   ; no, the masked 3-byte compare could not have matched either
     cmp word [rsi + SP_TRAIL_WORD_OFF], SP_TRAIL_WORD ; bytes 1..2 = 96 81
     jne .vocab_copy_sp_done                   ; 0xE2 lead byte but not the pattern
     mov r8d, 1                                ; set the register mirror, memory store below stays once-per-call
     mov dword [r12 + TOK_IS_SP], r8d          ; found _, mark SentencePiece
.vocab_copy_sp_done:
     cmp r14d, VOCAB_FAST_STORE_BYTES          ; slen > 16?, short-string fast path below
     ja .vocab_copy_bytes                      ; yes -> bulk copy
     cmp rdi, r11                              ; store would start past the last legal 16-byte slot?
     ja .vocab_copy_bytes                      ; yes -> byte-exact copy
     vmovups xmm0, [rsi]                       ; load 16 bytes, the loaded pool always carries the db_load 1MB committed reserve above KP_USED, microdb_persist pre-alloc, and any post-load keys sit above the blob, so the <=16-byte read past the string end always stays inside committed pool memory
     vmovups [rdi], xmm0                       ; store 16 bytes, dest exact: rdi <= r11 from the check above, so the store ends at or before the last valid byte; bytes past slen are garbage, overwritten or never read
     add rdi, r14                              ; advance dest by actual slen
     add rsi, r14                              ; advance source past string bytes
     jmp .vocab_null_term                      ; short 16-byte path pays the jmp
.vocab_copy_bytes:
     mov ecx, r14d                             ; slen bytes exactly, no over-read or over-store possible
     rep movsb                                 ; copies exactly slen bytes; r14, slen preserved across rep movsb
.vocab_null_term:
     mov byte [rdi], 0                         ; Null-terminate
     inc rdi                                   ; Advance past null byte
     add rbx, PTR_SCALE                        ; Advance ptr array to next entry
     dec r9d                                   ; --remaining
     jnz .vocab_copy_loop                      ; Loop while entries remain
.vocab_copy_done:
     mov r10, [r12 + TOK_VOCAB]                ; Reload ptr array base, r10 held the blob end, a different address, so the reload is required
.copy_template_start:
; Prompt template, MANDATORY, a DB without it cannot render chat
     lea rcx, [rel inf_key_template]           ; rcx = "__sys_template", mandatory key
     lea rdx, inf_db_len                       ; out_len = &inf_db_len, slot name IS the [rbp-N] operand
     call ht_find_or_fatal                     ; rax = template blob, FATAL with the key name on miss
     mov rcx, inf_db_len                       ; rcx = full entry length, includes the NUL
     dec rcx                                   ; rcx = string byte count, the copy count
     mov rsi, rax                              ; rsi = source string
     lea rdi, [r12 + TOK_PROMPT_TEMPLATE]      ; rdi = destination buffer
     rep movsb                                 ; Copy slen bytes
     mov byte [rdi], 0                         ; Null-terminate
; System prompt template, MANDATORY, same contract
     lea rcx, [rel inf_key_template_sys]       ; rcx = "__sys_template_sys", mandatory key
     lea rdx, inf_db_len                       ; out_len = &inf_db_len
     call ht_find_or_fatal                     ; rax = template blob, FATAL with the key name on miss
     mov rcx, inf_db_len                       ; rcx = full entry length, includes the NUL
     dec rcx                                   ; rcx = string byte count, the copy count
     mov rsi, rax                              ; rsi = source string
     lea rdi, [r12 + TOK_SYSTEM_TEMPLATE]      ; rdi = destination buffer
     rep movsb                                 ; Copy slen bytes
     mov byte [rdi], 0                         ; Null-terminate
     mov dword [rel g_img_template_valid], 0   ; reset valid flag, stale from a previous context
     mov byte [rel g_img_template], 0          ; reset first template byte, marks buffer empty
     COPY_TEMPLATE inf_key_template_img, [rel g_img_template] ; Image-turn template from microdb, optional
     cmp byte [rel g_img_template], 0          ; buffer still empty, key absent ?
     je .img_tpl_absent                        ; yes, keep valid=0, fall back to base template
     mov dword [rel g_img_template_valid], 1   ; mark BSS image template as loaded
.img_tpl_absent:
     mov byte [rel g_think_template], 0        ; reset, stale from a previous context
     mov byte [rel g_think_sys_template], 0    ; reset, stale from a previous context
     COPY_TEMPLATE inf_key_template_think, [rel g_think_template] ; Thinking user-turn template from microdb, optional
     COPY_TEMPLATE inf_key_template_sys_think, [rel g_think_sys_template] ; Thinking system+user template from microdb, optional
     mov byte [rel g_multi_template], 0        ; reset, stale from a previous context
     mov byte [rel g_multi_think_template], 0  ; reset, stale from a previous context
     COPY_TEMPLATE inf_key_template_multi, [rel g_multi_template] ; Inter-turn template from microdb, optional
     COPY_TEMPLATE inf_key_template_multi_think, [rel g_multi_think_template] ; Inter-turn thinking template from microdb, optional
     lea rax, [r12 + TOK_PROMPT_TEMPLATE]      ; rax = &tok->prompt_template, base fallback
     mov [rel g_tpl_t1_base], rax              ; cache for the selector + init-order check
     lea rax, [r12 + TOK_SYSTEM_TEMPLATE]      ; rax = &tok->system_template, base fallback
     mov [rel g_tpl_sys_base], rax             ; cache for the selector
     lea rax, [rel g_multi_template]           ; rax = &g_multi_template buffer, @multi fallback
     mov [rel g_tpl_multi_base], rax           ; cache for the selector, macro derefs POINTER vars
     mov ecx, 1                                ; engine default: thinking ON, per-context reset
     call v7s_enable_thinking                  ; arm the engine default, ON, templates already carry the open tag
; === Step 19: Build Sampler within block ===
; Probindex dest = after str_buffer. Recompute from known offsets.
     mov eax, inf_vocab_size                   ; eax = vocab_size
     mov [r13 + SAM_VOCAB_SIZE], eax           ; sampler->vocab_size = vocab_size
     mov rbx, [r12 + TOK_VOCAB]                ; rbx = ptr_array base, reload from tok
     lea r15, [rbx + rax*PTR_SCALE]            ; r15 = ptr_array end = str_buffer start
     add r15, inf_str_bytes                    ; r15 += str_buffer = probindex start, 64-bit, straight from the qword slot
     mov [r13 + SAM_PROBINDEX], r15            ; sampler->probindex = probindex start
; sampler fields stay at the allocator's zero: temperature 0.0 = greedy, argmax default,
; === Step 20: Fill InferenceContext header and return ===
     mov rax, inf_ctx_block                    ; rax = ctx block base
     mov rcx, inf_transformer                  ; Load transformer pointer for ctx header
     mov [rax + CTX_TRANSFORMER], rcx          ; ctx->transformer = transformer
     mov [rax + CTX_TOKENIZER], r12            ; ctx->tokenizer = tokenizer
     mov [rax + CTX_SAMPLER], r13              ; ctx->sampler = sampler
; Set prompt_tokens and out_tokens, after probindex_end
     mov r9d, inf_vocab_size                   ; r9d = vocab_size
     lea r10, [r15 + r9*PTR_SCALE]             ; r10 = probindex_end = prompt_tokens start, r15 still holds the probindex base computed in Step 19 and never rewritten since
     mov r9d, inf_seq_len                      ; r9d = seq_len
     lea r11, [r10 + r9*FK_FLOAT_BYTES]        ; r11 = out_tokens start
     mov [rax + CTX_PROMPT_TOKENS], r10        ; ctx->prompt_tokens = prompt_tokens buffer
     mov [rax + CTX_OUT_TOKENS], r11           ; ctx->out_tokens = out_tokens buffer
     mov [rax + CTX_MAX_PROMPT], r9d           ; ctx->max_prompt = seq_len
     mov [rax + CTX_MAX_GEN], r9d              ; ctx->max_gen = seq_len
     mov dword [rax + CTX_QUANTIZE], 1         ; ctx->quantize = 1, Q8 quantized activation mode; CTX_POS stays at the allocator's zero, a fresh context starts at position 0
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; Epilogue: restore non-volatiles; rax = ctx from the 'mov rax, inf_ctx_block' in Step 20, the COPY_TEMPLATE ht_find calls above leave an entry pointer in rax, so only that load produces the return value, nothing between it and here rewrites rax
; ---- %undef cleanup: file-local defines and macro leave scope here, house convention ----
%unmacro COPY_TEMPLATE 2                       ; remove the file-local macro after its five optional call sites above, file scope ends here
%undef SP_LEAD_BYTE
%undef SP_TRAIL_WORD_OFF
%undef SP_TRAIL_WORD
%undef TOK_SEQ_QWORD_SCALE
%undef VOCAB_FAST_STORE_BYTES
%undef PER_VOCAB_BYTES
%undef STOP_HDR_SIZE
%undef TOK_DW3_SCALE
%undef FK_FLOAT_BYTES
%undef PTR_SCALE
