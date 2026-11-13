; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define SL_V7_ENCODE_ARG5_RSP_OFF 32           ; OFFSET FROM RSP, not rbp+48: after `and rsp,-16` / `sub rsp,48`, rsp = aligned-48, and encode's own arg5 sits at encode_entry_rsp+40 = aligned-16 = rsp+32
%define TOKEN_ID_SCALE 4                       ; dword scale for token id arrays
%define SL_ENCODE_SPARE_TOKENS 2               ; max_tokens headroom over word_len: 1 SP prefix token + 1 spare
; ============================================================================
; SL_CONS_ADDR dest64, src32
; Strip-cons-tag + cell-address idiom shared by the set_token env walks:
; src32 holds a TAGGED cons ref; dest gets the cell's address in the heap.
; Overwrites: dest32, the low half of dest.r13/CONS_SCALE/CONS_BASE_OFF are
; the evaluator's heap base and cell geometry, live across the walks.
; ============================================================================

%macro SL_CONS_ADDR 2
     mov %1d, %2                               ; copy the tagged cons ref, 32-bit: upper dest bits must end zero for the SIB
     btr %1d, SL_CONS_BIT                      ; strip the cons tag bit -> raw cell index
     lea %1, [r13 + %1*CONS_SCALE + CONS_BASE_OFF] ; the cell's address
%endmacro
sl_v7_encode_word:
     push rsi                                  ; save non-volatile: the lea below writes rsi, the atom text pointer ; both callees preserve it across the call itself, so this pair is entry/exit only
; tag guard: argument must be an atom, string for string lookup
     TAG_GUARD_ATOM sl_v7_enc_no_word, sl_v7_enc_not_atom ; must be a plain non-NIL atom for the string lookup, flags-only, rax survives for the lea
     lea rsi, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rsi = word's string text, no '"' prefix
     mov rax, [rel sl_hook_ctx]                ; InferenceContext*
     test rax, rax                             ; null = standalone interp?
     jz sl_v7_enc_no_ctx                       ; yes: FATAL
     mov rax, [rax + SL_CTX_TOKENIZER]         ; Tokenizer*
; max_tokens = word_len + 2, Phase 1 appends <= len bytes + 1 SP prefix token;
     call sl_swar_strlen                       ; ecx = word byte length, leaf; rsi/rax preserved
     lea r11d, [ecx + SL_ENCODE_SPARE_TOKENS]  ; r11d = max_tokens = len + 2, r11 is untouched by the arg setup below; rdx is not, it carries arg2, the text pointer
     cmp r11d, SL_CONCAT_BUF_SIZE              ; does sl_lg_encode_scratch, resd SL_CONCAT_BUF_SIZE = 2097152 ints hold that many? MANDATORY: max_tokens is derived from the INPUT length, and the worst input is not a GetStream word, capped at SL_STREAM_BUF_SIZE = 1 MiB, so len+2 <= 1048578 but a `_` concatenation atom, which can reach SL_CONCAT_BUF_LAST = 2097151 bytes, len+2 = 2097153 then EXCEEDS the 2097152-int buffer by one int. encode, treats max_tokens as a hard bound and writes that many ints into arg3
     ja sl_v7_enc_toolong                      ; no: FATAL, never clamp, a clamped token count silently truncates the encoding
     push r12                                  ; save the evaluator's stream cursor across the rsp borrow
     mov r12, rsp                              ; park the pre-alignment rsp, encode preserves r12, SF_START/SF_END list
     and rsp, -16                              ; force 0 mod 16 for the call, NOT a no-op: entry parity is variable, so the mask does real work on the 0-mod-16 entry paths; the r12 park restores the caller's parity either way
     sub rsp, 48                               ; 32 shadow + arg5 slot + pad, aligned base keeps 0 mod 16.RELEASED BY `mov rsp, r12` just below the call, NOT by an `add`, that restore is unconditional, so a paired `add rsp, 48` here would be a dead instruction
     mov rcx, rax                              ; arg1: Tokenizer*, rax survived the strlen leaf
     mov rdx, rsi                              ; arg2: char* text
     lea r8, [rel sl_lg_encode_scratch]        ; arg3: tokens[] output buffer
     lea r9, [rel sl_lg_encode_n]              ; arg4: &n_tokens
     mov dword [rsp + SL_V7_ENCODE_ARG5_RSP_OFF], r11d ; arg5: max_tokens = len + 2, r11d survived the arg setup; the slot is relative to the ALIGNED rsp and lands on encode's own arg5 read, ARG5_OFF 48 off encode's rbp
     call encode                               ; BPE/SP encode -> tokens[] in sl_lg_encode_scratch, eax = token count
     mov rsp, r12                              ; restore the PRE-ALIGNMENT rsp, the and-mask is destructive, the caller's parity lives only in r12
     pop r12                                   ; restore the evaluator's stream cursor, the push above
     pop rsi                                   ; restore the caller's rsi, the new push above
     mov rax, [rel sl_hook_ctx]                ; InferenceContext*, non-NULL: sl_v7_enc_no_ctx guard already ran
     mov rax, [rax + SL_CTX_TOKENIZER]         ; Tokenizer*
     mov ecx, [rax + TOK_IS_SP]                ; ecx = 0, BPE or 1, SP : doubles as the tokens[] index
     cmp dword [rel sl_lg_encode_n], ecx       ; enough tokens for a content token?, BPE needs n>=1, SP needs n>=2
     jbe sl_v7_enc_none                        ; no: return NIL
     lea r11, [rel sl_lg_encode_scratch]       ; r11 = tokens[] base, RIP+index is illegal x64, rel would silently encode absolute ADDR32
     mov eax, [r11 + rcx*TOKEN_ID_SCALE]       ; SP -> tokens[1], first content, BPE -> tokens[0]
     jmp MakeNum                               ; tail call: wrap as NUMBER
sl_v7_enc_none:
     xor eax, eax                              ; return NIL, NO pop here: the two pops above already consumed the function's only two pushes, r12, rsi, so another pop would load this frame's own return address, overwriting the saved registers and advance rsp into the caller's frame before the ret
     ret                                       ; return to caller
sl_v7_enc_no_word:
     FATAL "SectorLogo: EncodeWord requires a word argument (got NIL)" ; NIL word argument, distinct from a missing hook context
sl_v7_enc_no_ctx:
     FATAL "SectorLogo: EncodeWord requires host hook context" ; null ctx guard
sl_v7_enc_toolong:
     FATAL "SectorLogo: EncodeWord input is longer than the token scratch buffer" ; max_tokens = len+2 exceeds the output buffer; FATAL rather than clamp, because a clamped bound silently truncates the encoding
sl_v7_enc_not_atom:
     FATAL_VALUE "EncodeWord requires a word or string argument", eax ; name the offending value
sl_v7_db_get:
     push rsi                                  ; save non-volatile, rep movsb overwrites it
     push rdi                                  ; save non-volatile, rep movsb overwrites it
; tag guard: key must be an atom, string for ht_find lookup
     TAG_GUARD_ATOM sl_v7_dbget_nil_early, sl_v7_dbget_not_atom ; NIL returns early, only rsi/rdi pushed so far, 2-pop exit ; a non-atom key is a FATAL
; atom text is stored quote-free, the lexer strips '"' bytes
     lea rsi, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rsi = key string text in heap
; r12/r14 need no save: ht_find/Intern preserve them via their own prologues
     sub rsp, 40                               ; home area + padding; entry parity is variable, ht_find below is aligned by CORRECT_RSP, the Intern call relies on Intern's parity-tolerant, zero-Win32 design
; ht_find,rcx=key_string, rdx=&out_len -> rax=data_ptr or NULL
     mov rcx, rsi                              ; arg1: key string
     lea rdx, [rel sl_lg_db_len]               ; arg2: &out_len
     CORRECT_RSP                               ; align for the ht_find call below, its shadow is reserved under the flag
     sub rsp, 32                               ; reserve the callee's shadow AFTER the flag: space above CORRECT_RSP's push is what the callee may write, first shadow qword
     call ht_find                              ; rax = data pointer or NULL
     add rsp, 32                               ; release the shadow reservation, flag back at [rsp]
     RESTORE_RSP                               ; undo the parity correction
     test rax, rax                             ; NULL = not found?
     jz sl_v7_dbget_nil                        ; yes: return NIL
; copy result to tok_buf and intern it
     mov rsi, rax                              ; rsi = data pointer
     mov rcx, [rel sl_lg_db_len]               ; FULL 64-bit data length: the slot is resq and ht_find writes the whole E_DLEN qword
     cmp rcx, SL_TOK_BUF_NUL_OFF               ; fits in tok_buf?, SL_TOK_BUF_NUL_OFF is the content bound: -1 type byte, -1 NUL; 64-bit compare rejects any length with bits above 31 set
     jbe sl_v7_dbget_len_ok                    ; yes: use as-is
     mov eax, ecx                              ; narrow ONCE, explicitly, rcx may hold bits above 31, the diagnostic is 32-bit
     bts eax, 30                               ; mark value as NUMBER for FATAL_VALUE
     FATAL_VALUE "DatabaseGet value exceeds token buffer (stored data corrupt)", eax ; capacity: never clamp a stored blob
sl_v7_dbget_len_ok:
     mov byte [rel sl_tok_buf], SL_TYPE_STRING ; type prefix for Intern
     lea rdi, [rel sl_tok_buf + SL_ATOM_TEXT_OFF] ; dest = tok_buf content area
     rep movsb                                 ; ERMS forward copy: ~16 bytes/cycle, rcx=byte count, rsi/rdi already set; DF=0 per the file header
     mov byte [rdi], 0                         ; NUL-terminate, rdi = one past last copied byte
     lea rsi, [rel sl_tok_buf]                 ; source for Intern, starts at type byte
     call Intern                               ; eax = atom for the stored value
     add rsp, 40                               ; free shadow AFTER the call, 32 shadow + 8 align
     pop rdi                                   ; restore caller's non-volatile rdi
     pop rsi                                   ; restore caller's non-volatile rsi
     ret                                       ; return to caller
sl_v7_dbget_nil:
     add rsp, 40                               ; free shadow, entered post-ht_find: 2 pushes outstanding
; fall through into sl_v7_dbget_nil_early, shared pop+return tail
sl_v7_dbget_nil_early:
     pop rdi                                   ; restore caller's rdi
     pop rsi                                   ; restore caller's rsi
     xor eax, eax                              ; return NIL
     ret                                       ; return to caller
sl_v7_dbget_not_atom:
     FATAL_VALUE "DatabaseGet requires a word or string key", eax ; name the offending key
sl_v7_db_put:
     push rsi                                  ; save non-volatile, key ptr reuses rsi below
     push rdi                                  ; save non-volatile, value ptr reuses rdi below
; tag guard: key must be an atom, string for ht_add lookup
     TAG_GUARD_ATOM sl_v7_dbput_key_fatal, sl_v7_dbput_key_fatal ; a NIL and a non-atom key share one handler, cannot store with either
; key is a valid atom
     lea rsi, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rsi = key string, no '"' prefix
     push rax                                  ; save the TAGGED key atom, a raw pointer is invisible to the GC's conservative stack scan, only tagged values get marked; EvalNextArg's value side effects can allocate atoms and a collect could reclaim a key that exists nowhere else but this slot
     call EvalNextArg                          ; eax = value, tagged
     pop r10                                   ; the tagged key atom back in r10, the collect under EvalNextArg marked it through this very stack slot, NEVER pop into rax: the value's tagged result lives there
     lea rsi, [r13 + r10 + SL_ATOM_TEXT_OFF]   ; RECOMPUTE the key pointer AFTER the call surround, never trust a raw pointer across a callee
; check value type: NIL, NUMBER, atom, or CONS
     TAG_GUARD_ATOM sl_v7_dbput_val_fatal, .dbput_tagged ; cannot store NIL; NUMBER/FLOAT/CONS route out of line
; --- plain ATOM, tag=0 : common RAG value type, falls through ---
     lea rdi, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rdi = value string, no '"' prefix
; length via the AVX2 leaf; rsi must still hold the KEY for ht_add, and the
; leaf reads rsi, so park it across the call, net-zero parity
     push rsi                                  ; save key pointer
     mov rsi, rdi                              ; rsi = value string for the leaf
     call sl_swar_strlen                       ; ecx = strlen, NUL excluded
     pop rsi                                   ; restore key pointer
     mov rdx, rdi                              ; arg2: data pointer, AFTER the call, rdx is volatile
     mov r8d, ecx                              ; arg3: data length
     jmp sl_v7_dbput_call_add                  ; shared ht_add tail
.dbput_tagged:
     mov ecx, eax                              ; ecx = value for tag check
     shr ecx, 30                               ; ecx = 2-bit tag code (NUMBER = 1, FLOAT = 3)
     cmp ecx, 1                                ; NUMBER?
     je sl_v7_dbput_num_val                    ; yes: convert to string
     cmp ecx, 3                                ; FLOAT?
     je sl_v7_dbput_flt_val                    ; yes: convert decimal text
     jmp sl_v7_dbput_val_fatal                 ; CONS: cannot store
sl_v7_dbput_num_val:
; convert NUMBER to decimal string in sl_tok_buf via the shared helper
     call NumVal                               ; eax = raw integer, leaf, any RSP parity safe
     lea rdi, [rel sl_tok_buf]                 ; dest = tok_buf
     call lg_num_to_str                        ; rdi = past last digit, leaf, preserves rsi, key pointer survives
     mov byte [rdi], 0                         ; NUL-terminate
     lea rax, [rel sl_tok_buf]                 ; rax = buf base
     mov rdx, rax                              ; arg2: data pointer
     sub rdi, rax                              ; rdi = length, past-last-digit minus base, the formatter already returned it
     mov r8d, edi                              ; arg3: data length
     jmp sl_v7_dbput_call_add                  ; shared ht_add tail
sl_v7_dbput_flt_val:
; convert FLOAT to decimal string in sl_tok_buf, sl_float_fmt preserves rsi; only callee is the leaf
     lea rdi, [rel sl_tok_buf]                 ; dest = tok_buf
     call sl_float_fmt                         ; rdi = past last decimal digit
     mov byte [rdi], 0                         ; NUL-terminate
     lea rax, [rel sl_tok_buf]                 ; rax = buf base
     mov rdx, rax                              ; arg2: data pointer
     sub rdi, rax                              ; rdi = length, past-last-digit minus base
     mov r8d, edi                              ; arg3: data length
; fall through into the shared ht_add tail
sl_v7_dbput_call_add:
     sub rsp, 40                               ; home area + padding, 2 pushes + 40 -> calls at 0 mod 16
     mov rcx, rsi                              ; arg1: key string, parked across EvalNextArg / strlen
     xor r9d, r9d                              ; arg4: persist = 0, temp
     CORRECT_RSP                               ; align for the ht_add call below, its shadow is reserved under the flag
     sub rsp, 32                               ; reserve the callee's shadow AFTER the flag: space above CORRECT_RSP's push is what the callee may write, first shadow qword
     call ht_add                               ; eax = slot or 0
     add rsp, 32                               ; release the shadow reservation, flag back at [rsp]
     RESTORE_RSP                               ; undo the parity correction
     add rsp, 40                               ; restore stack, 32 shadow + 8 align
     mov eax, [rel atom_sl_t]                  ; return TRUE, ht_add's error modes FATAL internally
     pop rdi                                   ; restore caller's non-volatile rdi
     pop rsi                                   ; restore caller's non-volatile rsi
     ret                                       ; return to caller
sl_v7_dbput_key_fatal:
     FATAL_VALUE "DatabaseSet requires a word or string key", eax ; name the offending key
sl_v7_dbput_val_fatal:
     FATAL_VALUE "DatabaseSet value must be a string, word, number, or float", eax ; name the offending value
sl_v7_get_token:
     call NumVal                               ; eax = raw position
     cmp eax, [rel sl_hook_pos]                ; unsigned: a negative index reads as ~4e9 -> always OOB
     ja sl_v7_gettoken_oob                     ; yes: out of bounds
     mov rdx, [rel sl_hook_tokens_ptr]         ; &token_history[0], non-NULL is also the "ctx is present" precondition for the load below
     test rdx, rdx                             ; null = no token history?
     jz sl_v7_gettoken_oob                     ; yes: return NIL
     mov r8, [rel sl_hook_ctx]                 ; ctx present: the tokens_ptr NULL check above is its proxy, tokens_ptr is read out of the ctx, so non-NULL implies non-NULL ctx
     mov r8, [r8 + SL_CTX_TRANSFORMER]         ; Transformer* == &Config, Config is the struct's first member, 64 bytes
     cmp eax, [r8 + SL_CFG_SEQ_LEN]            ; position within the model's KV length?
     jae sl_v7_gettoken_oob                    ; yes: out of bounds, return NIL
     mov eax, dword [rdx + rax*TOKEN_ID_SCALE] ; token_history[n], rax already zero-extended by NumVal's 32-bit ops; load feeds MakeNum directly
     jmp MakeNum                               ; tail call: wrap as NUMBER
sl_v7_gettoken_oob:
     xor eax, eax                              ; return NIL for out-of-bounds
     ret                                       ; return to caller
sl_v7_pos:
     mov eax, [rel sl_hook_pos]                ; current position
     jmp MakeNum                               ; tail call: wrap as NUMBER
sl_v7_get_token_pos:
     mov eax, [rel sl_hook_pos]                ; current KV position, c_pos as parked by THIS hook call
     mov ecx, [rel g_query_base_pos]           ; this chat call's entry position, the query base
     sub eax, ecx                              ; query-local token-stream position
     jmp MakeNum                               ; tail call: wrap as NUMBER

%macro SL_V7_CFG_NUM 1
     mov rax, [rel sl_hook_ctx]                ; InferenceContext*
     test rax, rax                             ; null = standalone interp?
     jz sl_v7_no_ctx                           ; yes: FATAL
     mov rax, [rax + SL_CTX_TRANSFORMER]       ; Transformer*
     mov eax, [rax + %1]                       ; config field
     jmp MakeNum                               ; tail call: wrap as NUMBER
%endmacro
sl_v7_vocab_size:
     SL_V7_CFG_NUM SL_CFG_VOCAB_SIZE           ; vocab_size
sl_v7_dim:
     SL_V7_CFG_NUM SL_CFG_DIM                  ; dim
sl_v7_layers:
     SL_V7_CFG_NUM SL_CFG_N_LAYERS             ; n_layers
sl_v7_eos:
     mov eax, [rel g_chat_eos_token]           ; eos id from the chat stop check
     jmp MakeNum                               ; tail call: wrap as NUMBER

%macro V7_PCT_STMT 1
     call EvalNextArg                          ; eax = percentage, tagged
     call NumVal                               ; eax = raw int
     vcvtsi2ss xmm0, xmm0, eax                 ; xmm0 =, float percentage, MUST run before the pointer load below, whose 64-bit write destroys eax
     vdivss xmm0, xmm0, dword [rel sl_v7_100f] ; xmm0 = percentage / 100.0, folded memory operand, no tmp reg
     mov rax, [rel %1]                         ; rax = &target float, hook global; NULL = standalone interp
     test rax, rax                             ; null = standalone interp?
     jz %%pct_done                             ; yes: skip the write, the conversions above are pure, their result is discarded with xmm0
     vmovss dword [rax], xmm0                  ; *target = result, VEX-encoded store, low 32 bits identical
%%pct_done:
     ret                                       ; return to caller
%endmacro
sl_v7_temp_stmt:
     V7_PCT_STMT sl_hook_temp_ptr              ; write percentage/100 into &temperature
sl_v7_topp_stmt:
     V7_PCT_STMT sl_hook_topp_ptr              ; write percentage/100 into &topp
; GetStream: returns the current output stream as a string atom. RETAINS a fresh
; interned snapshot of the whole stream on EVERY call, one call per turn is cheap,
; one call per token multiplies the copy+intern cost by the stream length. Prefer
; caching the result in a script variable and re-reading only after a rewrite.
sl_v7_stream:
     push rsi                                  ; save non-volatile
     push rdi                                  ; save non-volatile
     push rbp                                  ; pad; entry parity is variable, the Intern call below may run one slot off 0 mod 16 on the 0-mod-16 entry path, harmless: Intern is parity-tolerant, zero Win32 calls, FATAL macro self-aligns
     sub rsp, 32                               ; home area for the Intern call below
; NUL-terminate the stream buffer at current length
     mov ecx, [rel sl_stream_str_len]          ; ecx = current stream length
     lea rsi, [rel sl_stream_str_buf]          ; rsi = stream buffer base
     mov byte [rsi + rcx], 0                   ; NUL-terminate in-place
     mov byte [rel sl_concat_buf], SL_TYPE_STRING ; type prefix
     lea rdi, [rel sl_concat_buf + SL_ATOM_TEXT_OFF] ; dest for content
     inc ecx                                   ; +1 to copy NUL too
     rep movsb                                 ; ERMS forward copy: ~16+ bytes/cycle, content + NUL; DF=0 per the file header, no call intervenes anyway
     lea rsi, [rel sl_concat_buf]              ; source for Intern, type byte first
     call Intern                               ; eax = stream atom
     add rsp, 32                               ; drop the home area
     pop rbp                                   ; restore non-volatile
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     ret                                       ; return to caller
sl_v7_str_len:
     mov eax, [rel sl_stream_str_len]          ; eax = stream length
     jmp MakeNum                               ; tail call: wrap as NUMBER
sl_v7_set_stream:
     push rsi                                  ; save non-volatile
     push rdi                                  ; save non-volatile
; entry parity is variable, see the header note above sl_v7_encode_word, EvalNextArg/sl_swar_strlen are parity-tolerant; no CORRECT_RSP needed
     sub rsp, 40                               ; home area + parity pad, 2 pushes + 40 -> calls at 0 mod 16 on the 8-mod-16 entry
     call EvalNextArg                          ; eax = string atom
; tag check: must be atom, or NIL for empty
     TAG_GUARD_ATOM .ss_empty, .ss_notatom     ; NIL is an empty replacement; CONS/NUMBER/FLOAT is a type error
; get string pointer from heap
     lea rsi, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rsi = string in heap, no '"' prefix
     call sl_swar_strlen                       ; ecx = len, NUL excluded; rax dead here
     cmp ecx, SL_STREAM_BUF_SIZE - 1           ; len bytes + NUL must fit SIZE?
     ja .ss_fatal                              ; oversized, FATAL, never silently truncate
     mov [rel sl_stream_str2_len], ecx         ; store rewritten length
     inc ecx                                   ; len + 1: copy the NUL as well
     lea rdi, [rel sl_stream_str_buf2]         ; dest for the copy
     rep movsb                                 ; ERMS forward copy: ~16+ bytes/cycle incl the NUL, DF=0 per the file header; EvalNextArg/sl_swar_strlen do not leak it
; fall through: flags-only epilogue, stores set below
.ss_done_flags:
     mov dword [rel sl_action_code], SL_ACTION_REWRITE ; arm the rewrite action, chat loop re-encodes the replacement stream
     add rsp, 40                               ; drop the home area + parity pad, the .ss_empty path jumps into this shared epilogue, so one add covers both
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     ret                                       ; return to caller
.ss_empty:
; empty replacement string: buf2[0]=0, len=0, then the shared flag epilogue
     mov byte [rel sl_stream_str_buf2], 0      ; empty string: NUL at [0] plus a zero length, the stale bytes past [0] from an earlier rewrite are unreachable, because every consumer copies exactly sl_stream_str2_len bytes
     mov dword [rel sl_stream_str2_len], 0     ; length = 0
     jmp .ss_done_flags                        ; shared epilogue
.ss_notatom:
     FATAL_VALUE "SetStream requires a string or word atom argument", eax ; name the offending value
.ss_fatal:
     bts ecx, 30                               ; mark value as NUMBER for FATAL_VALUE
     FATAL_VALUE "SetStream replacement exceeds stream buffer", ecx ; capacity: never truncate a rewrite
sl_v7_set_token:
     call EvalNextArg                          ; eax = tagged argument
     call NumVal                               ; eax = raw token id, FATALs on a non-NUMBER with its own message
     mov [rel sl_settok_id], eax               ; park: the replacement id survives every state-switch call below
     mov r9, [rel sl_hook_out_tok_ptr]         ; &out_token, NULL outside a running generation
     test r9, r9                               ; null = standalone interp, no hook armed ?
     jz sl_v7_SetOutputToken_noctx             ; standalone interp: no engine token slot to write
     mov r11, [rel sl_hook_ctx]                ; InferenceContext*
     mov r11, [r11 + SL_CTX_TRANSFORMER]       ; Transformer* == &Config
     cmp eax, [r11 + SL_CFG_VOCAB_SIZE]        ; id >= vocab_size, or negative ?
     jae .stk_badid                            ; out of range: FATAL, never let a garbage id into the engine slots
     mov [r9], eax                             ; write the replacement id into the engine's out_token slot
     mov r10, [rel sl_hook_tokens_ptr]         ; out_tokens history base, set from ctx at on_token entry, non-NULL together with out_tok_ptr
     mov ecx, [rel sl_hook_pos]                ; current absolute position
     mov r11, [rel sl_hook_ctx]                ; InferenceContext*
     mov r11, [r11 + SL_CTX_TRANSFORMER]       ; Transformer* == &Config, Config is the struct's first member
     cmp ecx, [r11 + SL_CFG_SEQ_LEN]           ; pos >= seq_len? out_tokens holds exactly seq_len ints, same bounds contract as the engine's own fill guard
     jae .stk_env_track                        ; slot out of bounds: the override stands, there is no history slot to record
     mov [r10 + rcx*TOKEN_ID_SCALE], eax       ; out_tokens[pos] = replacement id
     jmp .stk_env_track                        ; the engine slots are set: on to the script-side state switch
.stk_badid:
     mov eax, [rel sl_settok_id]               ; the offending id, for the FATAL text
     bts eax, 30                               ; mark value as NUMBER for FATAL_VALUE
     FATAL_VALUE "SetOutputToken id is outside the model vocabulary", eax ; names the offender
.stk_env_track:
     mov [rel sl_hook_tok_in], eax             ; publish the replacement id
     call MakeNum                              ; tagged NUMBER, leaf: r14 survives
     mov r9d, eax                              ; pending :TokenNumber value (0 = already written)
     mov dword [rel sl_settok_cursor], 0       ; no :piece entry found yet, per-invocation parking written before read
     mov r8d, r14d                             ; scan the current environment exactly once
.tstk_env_walk:
     test r8d, r8d                             ; cons cell?, atoms/numbers/NIL end the chain
     jns .tstk_env_done                        ; non-cons tail: scan complete, decode what was found
     SL_CONS_ADDR r11, r8d                     ; the cell's address
     mov eax, [r11]                            ; the cell's car = the KEY.VALUE pair
     SL_CONS_ADDR r10, eax                     ; the pair cell's address
     mov eax, [r10]                            ; the pair's car = the KEY atom, tagged
     cmp eax, dword [rel atom_sl_lg_tok]       ; the :TokenNumber pair?
     jne .tstk_env_check_piece                 ; no: this cell is not the TokenNumber binding
     test r9d, r9d                             ; first-match policy: skip later duplicates
     jz .tstk_env_next                         ; already written: skip this duplicate
     mov [r10 + CONS_CDR_OFF], r9d             ; pair cdr := the parked tagged NUMBER, no allocation
     xor r9d, r9d                              ; mark done
     cmp dword [rel sl_settok_cursor], 0       ; both entries found -> stop
     jnz .tstk_env_done                        ; :piece already found: scan complete
     jmp .tstk_env_next                        ; :piece still pending: keep scanning
.tstk_env_check_piece:
     cmp eax, dword [rel atom_sl_lg_piece]     ; the :piece pair?
     jne .tstk_env_next                        ; no: neither entry, step to the next cell
     cmp dword [rel sl_settok_cursor], 0       ; keep the FIRST :piece on duplicates
     jne .tstk_env_next                        ; a :piece is already parked: keep the first
     mov [rel sl_settok_cursor], r8d           ; root the matching env cell for the post-walk decode
     test r9d, r9d                             ; :TokenNumber still pending?
     jz .tstk_env_done                         ; both found -> decode now
.tstk_env_next:
     mov r8d, [r11 + CONS_CDR_OFF]             ; cursor = next chain node
     jmp .tstk_env_walk                        ; continue the single environment scan
.tstk_env_done:
     mov ecx, [rel sl_settok_cursor]           ; ecx = parked :piece env cell, 0 = none found
     test ecx, ecx                             ; no :piece pair (standalone path)?
     jz .stk_done                              ; nothing to switch: done
     mov edx, [rel sl_settok_id]               ; arg1: the replacement token id
     CORRECT_RSP                               ; align for the decode call, paired on every path
     sub rsp, 32                               ; home area ABOVE the flag, house pattern
     call lg_decode_token_real                 ; eax = the replacement's interned piece atom
     add rsp, 32                               ; drop the home area, flag is back at [rsp]
     RESTORE_RSP                               ; undo the parity correction, paired on every path
     mov ecx, [rel sl_settok_cursor]           ; re-derive after the decoder may have GC'd/interned
     SL_CONS_ADDR r11, ecx                     ; the env cell's address
     SL_CONS_ADDR r10, dword [r11]             ; the cell's car = the PIECE.value pair cell
     mov [r10 + CONS_CDR_OFF], eax             ; pair cdr := the switched piece
.stk_done:
     ret                                       ; ACCEPT: the engine decodes the REPLACEMENT id exactly as if it had sampled it
sl_v7_SetOutputToken_noctx:
     FATAL "SectorLogo: SetOutputToken requires the token hook (no engine generation is running)" ; names the missing context
sl_v7_reject:
     mov dword [rel sl_action_code], SL_ACTION_RESAMPLE ; set action code
     ret                                       ; return to caller
sl_v7_halt:
     mov dword [rel sl_action_code], SL_ACTION_EOS ; set action code
     ret                                       ; return to caller
sl_v7_no_ctx:
     FATAL "SectorLogo: Primitive requires host hook context" ; null ctx guard
%unmacro SL_V7_CFG_NUM 1                       ; release: a later %macro redefinition would be a build break, not a silent collision
%unmacro V7_PCT_STMT 1                         ; release: generic-enough name that a collision is plausible
%unmacro SL_CONS_ADDR 2                        ; release: file-local strip+lea idiom macro, the piece-walk sites above are its only users
%undef SL_V7_ENCODE_ARG5_RSP_OFF               ; release: file-local, used only in sl_v7_encode_word's arg5 store
%undef TOKEN_ID_SCALE                          ; release: file-local, used only in sl_v7_encode_word's token load and sl_v7_get_token's history read ; unprefixed and generic, so a leaked definition would be picked up silently
%undef SL_ENCODE_SPARE_TOKENS                  ; release: file-local, used only in sl_v7_encode_word's max_tokens lea
