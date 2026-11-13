; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; chat_rewrite.inc, rewrite machinery: remove flush, LCP diff, bitmap repair, mute+tier dispatch, tier-2 fill-in, re-encode, re-prefill, rewrite sample, zero path .
; ANCHOR RULE: local labels only, the non-local label anchor stays v7s_chat, SF_START in chat.inc .
; Entered from the gen loop's hook dispatch, SL_ACTION_REWRITE ; ends jmp .chat_rewrite_sample -> .chat_gen_loop.

.chat_hook_rewrite:
     call v7s_stream_remove_flush              ; deliver the pending evicted batch BEFORE any rewrite mutation: the flush's meta reports evicted tokens at first_pos + index, stream_evict.inc, a mapping that is only truthful within one contiguous position timeline, a rewrite moves that timeline, so evictions pushed after the rewrite would merge into the stale batch with wrong reported positions; count == 0 is the documented no-op path, and the standing-batch edge, streaming disabled at pos <= C with evictions pending is delivered here too, its positions are still truthful; the flush preserves rbx/rsi/rdi/r12-r15, its SF frame and overwrites only rax/rcx/rdx/r8-r11, all dead at hook entry, eax = action code, consumed by the dispatch cmp eax, 3 ; the Logo eval inside the flush is sequential, the on_token script has already returned and may legally SetStream, the rewrite below then targets the newest str_buf2, the documented contract
; 1. Save old token count, before rewind for bitmap cleanup
     mov r12d, [rel sl_stream_tok_count]       ; r12d = old tok_count, save before rewind
; 2. Compute LCP divergence and truncate tri-array
     call v7s_stream_compute_rewrite_diff      ; eax = T_diverge, rdx = suffix_str
     mov r11, rdx                              ; r11 = save suffix_str, rdx overwritten by btr loop
; 3. Clear presence-penalty bitmap bits for discarded tokens
; Discarded tokens = history[T_diverge .. old_tok_count-1]
     cmp dword c_presence_penalty_bits, 0      ; penalty == +0.0f, disabled?
     jz .chat_remark_done                      ; disabled: neither clearing nor re-marking
     mov r8d, eax                              ; r8d = T_diverge
     lea r9, [rel sl_stream_tok_history]       ; r9 = token history base
     mov r10, c_bitmap_base                    ; r10 = bitmap base
     cmp r8d, r12d                             ; T_diverge >= old token count?
     jae .chat_remark_done                     ; pure append: discarded suffix empty, retained bits already set
.chat_rewrite_btr_loop:
     mov edx, [r9 + r8*RW_DWORD_SCALE]         ; edx = discarded token ID
     btr dword [r10], edx                      ; clear the discarded token's presence bit
     inc r8d                                   ; next discarded token
     cmp r8d, r12d                             ; index >= old count?
     jb .chat_rewrite_btr_loop                 ; continue
.chat_rewrite_btr_skip:
     xor r8d, r8d                              ; at least one old token discarded -> re-mark kept prefix
     lea r9, [rel sl_stream_tok_history]       ; r9 = token history base
     mov r10, c_bitmap_base                    ; r10 = bitmap base
     cmp r8d, eax                              ; index >= T_diverge?
     jae .chat_remark_done                     ; full-rewrite / empty prefix: nothing to re-mark
.chat_remark_loop:
     mov edx, [r9 + r8*RW_DWORD_SCALE]         ; edx = kept token ID
     bts dword [r10], edx                      ; set the kept token's presence bit
     inc r8d                                   ; next kept token
     cmp r8d, eax                              ; index >= T_diverge?
     jb .chat_remark_loop                      ; continue
.chat_remark_done:
; === Tier dispatch: tier 2, streaming on AND pos_old > C mutes + fills in; tier 1 stays unmuted ===
; g_stream_cap is only meaningful while g_stream_enable == 1, the same contract
; every forward's slot resolver follows : with streaming disabled, never
; enabled, BSS zero or disabled-after-enable, geometry retained for the
; disable query contract, the C compare would otherwise route a plain
; legacy rewrite into the tier-2 arm with a zero/stale window. eax, T_diverge
; and r11, suffix pointer are untouched here and stay live for the shared body.
; === Rewrite mute, tier 2 only + tier dispatch ===
; Tier 2's fill-in/rebuild rewrites ring slots at wrapped positions without
; draining the pending eviction log, and none of those rows are evictions of
; the live timeline, a rollback discards them, a rebuild rewrites the same
; content.Unmuted, a long rebuild overflows the pending log, the resolvers'
; FATAL or pollutes the delivered batch with rollback-timeline positions, so
; tier 2 parks the warm latch at 0x7FFFFFFF and restores it exactly at
; .chat_rewrite_sample / .chat_finish, c_rw_muted gates both: tier-1 mid-suffix
; stops bypass the tier-2 flag below .
; Tier 1 does NOT mute: below C nothing logs anyway, linear fills never remove,
; and when the replacement crosses C the writes at [C, end remove REAL kept
; rows exactly like a natural crossing, those must reach the callback, and
; the re-prefill loop's remove_BATCH drain keeps the log from overflowing.
     mov ecx, c_pos                            ; ecx = pos_old, c_pos is still the pre-rewrite cursor
     cmp dword [rel g_stream_enable], 0        ; streaming session?, gate FIRST: cap is meaningless when disabled
     je .rw_tier1_unmuted                      ; no: legacy rollback body below
     cmp ecx, [rel g_stream_cap]               ; evictions have begun, wrapped session ?
     jbe .rw_tier1_unmuted                     ; no: tier 1, logging stays live
     cmp dword [rel g_stream_otk_live], 0      ; is out_tokens[0..min,pos,seq_len a real id history?, validated BEFORE any tier-2 state is touched, a FATAL here must not leave flag/latch half-applied
     je .chat_rewrite_history_fatal            ; no, session resumed from a KV file : refuse, garbage ids would OOB the embedding table
     mov r9d, [rel g_stream_remove_warm]       ; save the post-load warm latch via r9d, DEAD here, eax holds T_diverge and stays live for the shared body below
     mov c_rw_latch, r9d                       ; park for the exact restore at .chat_rewrite_sample
     mov dword [rel g_stream_remove_warm], 0x7FFFFFFF ; mute eviction logging for every rebuild write, wrap-safe for ~2G tokens
     mov c_rw_muted, 1                         ; mark the latch parked, the finish tests this, not the tier-2 flag
     mov c_rw_pos_old, ecx                     ; park pos_old for the tracker-store math
     mov c_rw_tier2, 1                         ; mark the turn tier-2, zero/sample/finish unwinds read it
     jmp .rw_tier1                             ; shared rollback + re-prefill body
.rw_tier1_unmuted:
     mov c_rw_muted, 0                         ; nothing parked: the sample/finish gates must read a defined 0
.rw_tier1:
; 4. Sync c_pos to rollback position: c_pos = base_pos + T_diverge
; Also fix gen_count: r15d = T_diverge, only valid tokens count from this turn
     mov r12d, eax                             ; r12d = T_diverge, loop counter & gen_count
     mov ecx, [rel sl_stream_base_pos]         ; ecx = base position at turn start
     add ecx, eax                              ; ecx = base + T_diverge
     mov c_pos, ecx                            ; c_pos = new position, KV cache cursor
; r15d = gen_count: after rewind, only T_diverge tokens are valid
; gen_count at turn start was 0, so r15d = T_diverge
     mov r15d, r12d                            ; r15d = T_diverge, gen_count corrected
; 5. Sync c_out_pos and truncate caller output buffer
     mov eax, [rel sl_stream_str_len]          ; eax = truncated stream byte length
     cmp eax, c_output_len                     ; rewritten stream fits the caller buffer?, NUL writes at [len]
     jae .chat_rewrite_overflow_fatal          ; rewritten stream would truncate, FATAL rather than corrupt the caller buffer
     mov c_out_pos, eax                        ; c_out_pos = truncated length
     mov rcx, rax                              ; rcx = truncated length, for null-term
     mov rax, c_output                         ; rax = output buffer base
     mov byte [rax + rcx], 0                   ; output[truncated_len] = NUL
; 6. Re-encode replacement suffix via encode,
; rdx was overwritten by the btr bitmap loop, restore from r11 save
     mov rdx, r11                              ; rdx = suffix string, arg2, restored
     mov rax, c_ctx                            ; InferenceContext*
     mov rcx, [rax + CTX_TOKENIZER]            ; rcx = Tokenizer*, arg1
; rdx already = suffix string, arg2
     lea r8, [rel sl_rewrite_tok_buf]          ; arg3: tokens[] output buffer
     lea r9, [rel sl_rewrite_count]            ; arg4: &n_tokens output
     sub rsp, CHAT_TEMP_FRAME_2                ; encode, takes 5 args, so only CHAT_TEMP_ARG5 is staged; same 48-byte frame as the 2-arg form
     mov qword [rsp + CHAT_TEMP_ARG5], SL_REWRITE_MAX_TOKENS ; arg5: max_tokens = 16K, imm32 MOV r/m64 form
     call encode                               ; encode suffix -> token IDs
     add rsp, CHAT_TEMP_FRAME_2                ; restore stack
     cmp dword [rel sl_rewrite_count], SL_REWRITE_MAX_TOKENS ; encode hit its 16K budget?, the encoder truncates silently, the only way to detect from here
     je .chat_rewrite_enc_cap_fatal            ; a replacement at the cap is treated as truncated: never silently accept a partial rewrite
; === Tier 2 delta fill-in: re-forward exactly the window rows the rewrite broke ==
; Geometry, all absolute : A = base_pos + T_diverge, r12d, live since the step-4
; store, end = A + n. The new window is [end - W, end.Its rows are:
; re-forwarded [max,0, end-W, min,A, pos_old-W, evicted-or-discarded AND
; pre-divergence; resident pre-divergence [max,end-W, pos_old-W, A, never
; touched, recomputing them under a mis-dated stale segment would degrade true
; rows ; replacement [A, end, the existing loop below. Rows [end, pos_old
; map to the same slots as the re-forwarded head, so nothing is left stale.
     cmp c_rw_tier2, 0                         ; tier 2?
     je .rw_fill_done                          ; no: tier 1 has no fill-in
     mov eax, [rel sl_stream_base_pos]         ; eax = base_pos
     add eax, r12d                             ; eax = A = base_pos + T_diverge
     mov r10d, eax                             ; r10d = A, parked; survives the forward calls
     add eax, [rel sl_rewrite_count]           ; eax = end = A + n
     mov c_rw_end, eax                         ; park the PLANNED end: the finish unwind recomputes the tracker from the ACTUAL stop cursor against this, an early max_gen/output stop must not publish full-suffix geometry
     mov edx, [rel g_stream_window]            ; edx = W
     mov r8d, eax                              ; r8d = end
     sub r8d, edx                              ; r8d = end - W, CF set when end < W
     jnc .rw_lo_ok                             ; window head inside the stream
     xor r8d, r8d                              ; end < W: head clamps to 0, degenerate short stream
.rw_lo_ok:
     mov r9d, c_rw_pos_old                     ; r9d = pos_old
     sub r9d, edx                              ; r9d = pos_old - W, no underflow: pos_old > C = S + W > W
     cmp r9d, r10d                             ; resident boundary vs divergence
     jbe .rw_hi_ok                             ; pos_old - W <= A: stop at the resident boundary
     mov r9d, r10d                             ; deeper divergence: stop at A, the suffix loop owns [A, end
.rw_hi_ok:
     mov ecx, edx                              ; ecx = W, default tracker value: the whole window is restored
     cmp r9d, c_seq_len                        ; does the fill-in need ids at or above the recorded-history bound?
     jbe .rw_store_full                        ; no: every id is live, the full window becomes valid
     xor r9d, r9d                              ; shaved: ids above seq_len were never recorded, otk fills stop at seq_len, skip the fill-in entirely; the adds below set CF for the jnc
     mov ecx, r10d                             ; shaved tracker: end + W - pos_old =, A + n + W - pos_old, the resident-and-true count [pos_old - W, end
     add ecx, [rel sl_rewrite_count]           ; + n
     add ecx, edx                              ; + W
     sub ecx, c_rw_pos_old                     ; - pos_old, CF set when the discarded tail exceeds W
     jnc .rw_shaved_clamp                      ; clamp to [0, W]
     xor ecx, ecx                              ; tail >= W: nothing in the window survives, attend sinks only
.rw_shaved_clamp:
     cmp ecx, edx                              ; clamp high side to W
     jbe .rw_store_shaved                      ; within W: keep
     mov ecx, edx                              ; cap at W
.rw_store_shaved:
     mov c_rw_valid_store, ecx                 ; shaved value: the fill-in was skipped, ids above seq_len were never recorded, only [pos_old-W, end is true
     jmp .rw_fill_range                        ; prefix empty, r9d = 0 <= r8d : straight to the cursor restore
.rw_store_full:
     mov c_rw_valid_store, ecx                 ; full restore: the fill-in re-forwarded the whole window head, every window row is true
.rw_fill_range:
     mov r13d, r9d                             ; park prefix_hi, r13 is dead until the suffix loop re-purposes it
     mov r14d, r10d                            ; park A, r14 is dead until the suffix loop's max_gen reload
     cmp r13d, r8d                             ; prefix_hi <= prefix_lo, empty range: grow/equal or n >= W ?
     jbe .rw_fill_noloop                       ; yes: nothing to re-forward
     mov c_pos, r8d                            ; cursor = prefix_lo, the walk starts at the window head
.rw_fill_loop:
     mov r11, c_out_tokens                     ; r11 = out_tokens base, reloaded per token, the forward calls overwrite it
     mov eax, c_pos                            ; eax = p
     mov edx, [r11 + rax*DWORD_SCALE]          ; edx = out_tokens[p], id history: live range proven by the seq_len shave
     mov r8d, eax                              ; r8d = p, forward arg3
     mov rcx, c_transformer                    ; arg1: Transformer*
     mov r9d, 1                                ; muted prefix replay only rebuilds the KV rows
     sub rsp, 32                               ; common Win64 shadow space
     cmp dword c_quantize, 0                   ; select the original precision path
     jz .rw_fill_f32                           ; F32 replay
     call forward_q8_mode                      ; rebuild Q8 KV without final projection
     jmp .rw_fill_adv                          ; common frame release
.rw_fill_f32:
     call forward_f32_mode                     ; rebuild F32 KV without final projection
.rw_fill_adv:
     add rsp, 32                               ; release common Win64 shadow space
     inc c_pos                                 ; p++, no r15d/g_live_gen_count/bts/append work here: kept-equivalent tokens
     mov eax, c_pos                            ; eax = p + 1
     cmp eax, r13d                             ; walked all of [prefix_lo, prefix_hi ?
     jb .rw_fill_loop                          ; no: next window row
.rw_fill_noloop:
     mov c_pos, r14d                           ; cursor = A, jump the resident kept rows [prefix_hi, A ; matches the step-4 store on tier 1
.rw_fill_done:
; 7. Prefill re-encoded tokens into KV cache & update tri-array
     cmp dword [rel sl_rewrite_count], 0       ; zero tokens?, counter read only for the zero test
     jz .chat_rewrite_zero                     ; yes: re-forward last token, then sample
     xor r12d, r12d                            ; r12d = loop index, NOT ebx, rbx=vocab!
.chat_rewrite_prefill_loop:
     lea r10, [rel sl_rewrite_tok_buf]         ; r10 = rewrite token buffer base
     mov edx, [r10 + r12*RW_DWORD_SCALE]       ; edx = token ID
     mov c_next_token, edx                     ; store for output + tri-array
     mov rsi, [rbx + rdx*VOCAB_PTR_SCALE]      ; rsi = piece text, rbx = tok->vocab, parked at :695
; Piece length via inline strlen, repne scasb overwrites rcx/rdi/rflags
     mov rdi, rsi                              ; rdi = piece for strlen
     xor eax, eax                              ; al = 0, search for NUL
     mov ecx, -1                               ; rcx = max scan count
     repne scasb                               ; scan for NUL byte
     not ecx                                   ; ecx = length including NUL
     dec ecx                                   ; ecx = piece_len, excluding NUL
     mov r14d, ecx                             ; r14d = slen, source length for macro
; Output buffer bounds check, same retest structure as the main loop
     mov eax, c_out_pos                        ; eax = current write position
     add eax, r14d                             ; eax = out_pos + raw slen, decoded <= raw
     cmp eax, c_output_len                     ; raw length fits?
     jb .chat_rw_decode_fits                   ; yes -> decoded cannot overflow either
; Raw does NOT fit, retest with the decoded length, decode shrinks
; <0xXX> 6->1 and SP boundaries 3->1: rsi/r14d already hold the scan inputs
     CHAT_DECODE_LEN                           ; r9d = decoded byte count, write-free scan
     mov eax, c_out_pos                        ; reload out_pos, scan overwritten eax
     add eax, r9d                              ; eax = out_pos + decoded length
     cmp eax, c_output_len                     ; decoded length fits?
     jae .chat_finish                          ; no even decoded: truncate for real
.chat_rw_decode_fits:
; Normalize piece to output buffer via decode macro
     mov eax, c_out_pos                        ; eax = current write position
     mov rdi, c_output                         ; rdi = output buffer base
     add rdi, rax                              ; rdi = output + out_pos, destination
     CHAT_DECODE_PIECE                         ; shared decoder returns the byte count in r13d
; Update output position and NUL-terminate
     add c_out_pos, r13d                       ; out_pos += decoded length
     mov eax, c_out_pos                        ; eax = new out_pos
     mov rdx, c_output                         ; rdx = output base
     mov byte [rdx + rax], 0                   ; NUL-terminate output
; Append decoded text to tri-array stream, same as main loop
     mov r8d, r13d                             ; r8d = decoded piece length
     sub eax, r8d                              ; eax = start of decoded text
     mov rdx, c_output                         ; rdx = output buffer base
     add rdx, rax                              ; rdx = &decoded_text in output buffer
     mov ecx, c_next_token                     ; ecx = token ID
     call v7s_stream_append_token              ; append to tri-array stream
; Mark replacement token in presence_penalty bitmap, same as main loop
     cmp dword c_presence_penalty_bits, 0      ; penalty == +0.0f, disabled?
     jz .chat_rewrite_bts_done                 ; skip: bitmap not in use
     mov rax, c_bitmap_base                    ; rax = bitmap base
     mov ecx, c_next_token                     ; ecx = token ID
     bts dword [rax], ecx                      ; hardware handles offset+bit automatically
.chat_rewrite_bts_done:
; Eviction drain, tier-1 crossing C logs real evictions, same cadence as the
; main decode loop; tier 2 runs muted so this never fires there
     cmp dword [rel g_stream_remove_count], remove_BATCH ; pending evictions at the drain threshold?
     jb .chat_rewrite_remove_ok                ; below -> no flush needed
     call v7s_stream_remove_flush              ; drain ~every remove_BATCH evictions, preserves rbx/rsi/rdi/r12-r15; overwrites volatiles only, all loop state lives in non-volatiles or frame slots
.chat_rewrite_remove_ok:
; KV-cache bounds: streaming wraps, the resolver owns the ring, so the
; legacy linear stop is wrong there, only the out_tokens store is
; seq_len-bounded, the id history stops at seq_len, like every fill
     mov r8d, c_pos                            ; current token position
     mov edx, c_next_token                     ; replacement token ID
     cmp r8d, c_seq_len                        ; absolute history uses this bound in either mode
     jb .chat_rewrite_store_id                 ; within the history array
     cmp dword [rel g_stream_enable], 0        ; may streaming continue past the history array?
     je .chat_finish                           ; linear KV is full
     jmp .chat_rewrite_stored                  ; forward without a history slot
.chat_rewrite_store_id:
     mov rax, c_out_tokens                     ; history base
     mov [rax + r8*DWORD_SCALE], edx           ; store the replacement token ID
.chat_rewrite_stored:
; Forward pass at c_pos, respecting quantization mode
     mov rcx, c_transformer                    ; Transformer*
     sub rsp, 32                               ; common Win64 shadow space
     cmp dword c_quantize, 0                   ; F32 mode?
     je .chat_rewrite_f32                      ; choose F32
     call forward_q8                           ; compute the replacement token's logits
     jmp .chat_rewrite_fwd_done                ; common frame release
.chat_rewrite_f32:
     call forward_f32                          ; compute the replacement token's logits
.chat_rewrite_fwd_done:
     add rsp, 32                               ; release common Win64 shadow space
     mov c_logits, rax                         ; cache logits pointer
     inc c_pos                                 ; advance KV cache position
     inc r15d                                  ; r15d++, rewrite-loop max_gen accounting keeps its own counter
     inc dword [rel g_live_gen_count]          ; every token the rewrite re-prefills COUNTS toward the live per-turn counter: GenCount must pass 0 exactly once per chat call, at the first hook call and then grow with every token the round processes, a single-INC-only-main-loop design made the first post-rewrite hook call see 0 again, which broke turn-start detection mid-round
; max_gen check inside rewrite prefill loop, same as main loop
; r14d is slen, piece length, NOT max_gen, load from stack slot
     mov eax, c_max_gen                        ; eax = max_gen, from stack frame
     test eax, eax                             ; max_gen == 0, unlimited ?
     jz .chat_rewrite_skip_maxgen              ; yes: skip check
     cmp r15d, eax                             ; gen_count >= max_gen?
     jae .chat_finish                          ; yes: stop generating, share finish path
.chat_rewrite_skip_maxgen:
     inc r12d                                  ; next replacement token, r12d, NOT ebx
     cmp r12d, dword [rel sl_rewrite_count]    ; all replacement tokens processed?
     jl .chat_rewrite_prefill_loop             ; no: continue prefill
; Sample next token from replacement logits and resume generation
.chat_rewrite_sample:
     cmp c_rw_muted, 0                         ; did this rewrite park the warm latch, tier 2 only ?
     je .rw_sample_pp                          ; no: the latch was never touched, nothing to restore
     mov eax, c_rw_latch                       ; the saved warm-latch value, tier-2 entry parked it
     mov [rel g_stream_remove_warm], eax       ; restore: eviction logging resumes on the next real wrapped write
     mov c_rw_muted, 0                         ; the latch is back in the live set
     cmp c_rw_tier2, 0                         ; tier 2?
     je .rw_sample_pp                          ; no: linear regime, the tracker was never touched
     mov c_rw_tier2, 0                         ; rewrite unwound: clear the flag so a later finish/second rewrite starts clean, the mid-suffix .chat_finish stops jump here DIRECTLY from the prefill loop, so they still see the flag and re-derive the tracker from the actual stop cursor
     cmp dword [rel sl_rewrite_count], 0       ; zero-token replacement?
     je .rw_sample_pp                          ; yes: .chat_rewrite_zero already published the tracker value
     mov eax, c_rw_valid_store                 ; eax = W, full restore or the shaved clamp
     mov [rel g_stream_win_valid], eax         ; publish: the window is canonical for the resumed generation
.rw_sample_pp:
     APPLY_PRESENCE_PENALTY .chat_skip_pp_rewrite ; apply penalty to rewrite logits
.chat_skip_pp_rewrite:
     CHAT_SAMPLE_TO_NEXT                       ; resample after the rewrite restored the window
     jmp .chat_gen_loop                        ; resume normal generation
.chat_rewrite_overflow_fatal:
     FATAL "SectorLogo rewrite: rewritten stream exceeds the caller output buffer"
.chat_rewrite_zero:
     cmp c_rw_tier2, 0                         ; tier-2 zero rewrite, wrapped rollback ?
     je .chat_rewrite_zero_count               ; no: linear regime, the tracker is not consulted below C
     mov eax, c_rw_valid_store                 ; eax = W, fill-in restored the whole window head or the shaved clamp, fill-in skipped
     mov [rel g_stream_win_valid], eax         ; publish BEFORE the refresh forward, its attention reads the tracker
.chat_rewrite_zero_count:
     mov eax, [rel sl_stream_tok_count]        ; eax = tok_count, after rewind
     test eax, eax                             ; any generated tokens kept?
     jz .chat_rewrite_zero_empty               ; no: entire stream erased, forward the last PROMPT token instead
     dec eax                                   ; eax = last valid token index
     lea r10, [rel sl_stream_tok_history]      ; r10 = history base
     mov edx, [r10 + rax*RW_DWORD_SCALE]       ; edx = last valid token ID, arg2
     jmp .chat_rewrite_zero_fwd                ; shared forward setup below
.chat_rewrite_zero_empty:
     mov eax, c_num_prompt_tokens              ; eax = prompt token count, the prompt FATALs above seq_len, so every index is in bounds
     test eax, eax                             ; empty prompt, nothing encoded ?
     jz .chat_finish                           ; nothing to re-forward: end the turn cleanly, SetStream "" on an empty session
     mov r10, c_prompt_tokens                  ; r10 = prompt token buffer, absolute positions [0, num
     dec eax                                   ; eax = last prompt token index
     mov edx, [r10 + rax*DWORD_SCALE]          ; edx = last prompt token ID, arg2, never index out_tokens by an unbounded absolute position, streaming base_pos can exceed seq_len
.chat_rewrite_zero_fwd:
     mov r8d, c_pos                            ; r8d = c_pos
     dec r8d                                   ; r8d = c_pos - 1, position of last valid token
     cmp dword c_quantize, 0                   ; F32 mode?, flag read only for the zero test
     jz .chat_rewrite_zero_f32                 ; yes: F32 forward
     mov rcx, c_transformer                    ; rcx = Transformer*, arg1
     sub rsp, 32                               ; shadow space
     call forward_q8                           ; logits in RAX
     add rsp, 32                               ; restore
     jmp .chat_rewrite_zero_done               ; skip F32 path
.chat_rewrite_zero_f32:
     mov rcx, c_transformer                    ; rcx = Transformer*, arg1
     sub rsp, 32                               ; shadow space
     call forward_f32                          ; logits in RAX
     add rsp, 32                               ; restore
.chat_rewrite_zero_done:
     mov c_logits, rax                         ; cache fresh logits
     jmp .chat_rewrite_sample                  ; sample from fresh logits
