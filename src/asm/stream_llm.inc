; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; stream_llm.inc, StreamingLLM configuration setter, v7_set_streaming .
; Include from v7.asm directly AFTER mrope.inc, all deps live there.
; Reads stream_cfg.inc storage via [rel ...].

     section .bss
ss_cand_s:
     resd 1                                    ; candidate S scratch, published to g_stream_sinks only after every check passed
ss_cand_w:
     resd 1                                    ; candidate W scratch, published to g_stream_window together with S and cap

; SS_RAW_CALL fn, call a Win64-ABI callee from this raw, frameless, rsp=8 mod 16
; entry: 32-byte shadow + 8-byte parity realignment. Argument registers must be fully
; loaded BEFORE the macro, the macro owns the region .

%macro SS_RAW_CALL 1
     sub rsp, 40                               ; 32 shadow + 8 parity: raw entry sits at rsp=8, mod 16
     call %1                                   ; the ABI callee, overwrites per its own contract
     add rsp, 40                               ; unwind the region, rsp back to entry parity
%endmacro

     section .text
; ============================================================================
; int v7_set_streaming,int enable, int num_sinks, int window_size
; rcx = enable, 0/1, edx = num_sinks, r8d = window_size; returns 0 / -1.
; Fill grade: streaming/eviction starts when C = S + W rows are
; full, pass window_size = fill_grade - num_sinks, C <= seq_len enforced .
; Gate/leave pairing is MANDATORY: ENGINE_LEAVE without a prior gate
; underflows engine_call_depth and leaks the zone claim, every later public
; call would FATAL "concurrent" ; the FATAL paths below also rely on the
; gate-parked engine_saved_rsp for the catch-and-unwind landing.
; Validation: S >= 1, sinks are the paper's whole point, W >= 1, the
; resolver divides by W, enable in {0,1}, C = S + W must not overflow
; int32, add/jc, lea writes no flags and must fit the allocated cache.
; The S >= 1 test only rejects S == 0 explicitly, but the gate is
; transitively watertight: any ACCEPTED, S, W pair satisfies S + W <=
; seq_len < 2^31 with no carry, the add/jc + unsigned cmp-ja pair, so a
; negative S, bit 31 set can never reach the accepted state, no
; dedicated sign check is needed.
; enable=1 FATALs while ctx->pos > 0, a state contract violation, not a
; -1 arg rejection : rows written under a previous
; S/W/C, or a resumed session would not match the new mapping. Disable is
; legal while the cache layout is still linear, pos <= C ; a WRAPPED session
; pos > C refuses with -1, legacy would read the slot-ordered ring rows
; as position-ordered. Disable stores the flag; the ring stays allocated
; reused by the next enable, freed at destroy and the pending eviction
; count stays standing.
; ============================================================================
     global v7_set_streaming                   ; export the streaming setter
v7_set_streaming:
     ENGINE_GATE_RAW 1                         ; claim the engine zone, raw SF-less entry; dead engine returns 0
     PUB_CPU_GATE                              ; refuse hardware below the v7 baseline, public-entry symmetry
     cmp qword [rel g_inf_ctx_block], 0        ; inference context alive?, kv_cache_file.inc guard style
     je .ss_not_init                           ; no -> FATAL, contract violation; the dead-flag path above is for post-FATAL calls
     cmp ecx, 1                                ; enable must be exactly 0 or 1
     ja .ss_bad                                ; anything else -> reject
     jc .ss_disable                            ; enable=0: CF=1 iff ecx==0 after the cmp above, the ja already excluded ecx>1
     test edx, edx                             ; S >= 1
     jz .ss_bad                                ; no sinks -> reject
     test r8d, r8d                             ; W >= 1, resolver divides by W
     jz .ss_bad                                ; zero window -> reject
     mov [rel ss_cand_s], edx                  ; park the CANDIDATE S: the live config must stay
     mov [rel ss_cand_w], r8d                  ; intact until every check passed, a rejecting
; path would otherwise leave a mixed S/W/cap triangle
; --- M-RoPE refusal: VLM wrap semantics are NOT implemented, slot-rope vs
; --- delta-rope rows would mix in the first window after the wrap boundary.
     lea rcx, [rel sz_sys_mrope_bound]         ; key = "__sys_mrope_bound", shared with mrope.inc
     xor edx, edx                              ; out_len = NULL, mrope.inc:139 call shape
     SS_RAW_CALL ht_find                       ; rax = bound ptr or NULL, OVERWRITES all volatiles
     test rax, rax                             ; key absent -> plain 1D rope model -> allow
     jz .ss_mrope_ok                           ; absent -> plain 1D rope model -> allow
     cmp dword [rax], 0                        ; key present: bound == 0 explicitly disables M-RoPE
     jne .ss_bad                               ; nonzero bound -> M-RoPE model -> reject
.ss_mrope_ok:
     mov rax, [rel g_inf_ctx_block]            ; rax = InferenceContext*, reload: the call ate rax
     cmp dword [rax + CTX_POS], 0              ; pos > 0 means a session is live, the count is read only for the zero test here
     jnz .ss_midstream                         ; refuse to re-map a live cache
     SS_RAW_CALL v7s_stream_remove_flush       ; deliver a standing batch BEFORE the reconfig drops it: the count=0 below is a deliberate fresh-session drop, but the batch's first= positions are truthful for the OLD timeline only, deliver them while that timeline is still the live one, no-op at count 0
     mov rax, [rel g_inf_ctx_block]            ; reload the ctx: the flush call ate rax, volatile
     mov r10, [rax + CTX_TRANSFORMER]          ; capacity must fit the allocated cache
     mov r10d, [r10 + CFG_SEQ_LEN]             ; r10d = seq_len, cache rows per layer
     mov r9d, [rel ss_cand_s]                  ; r9d = candidate S
     add r9d, [rel ss_cand_w]                  ; r9d = C = S + W, add sets CF on int32 overflow
     jc .ss_bad                                ; S + W wrapped 32 bits -> reject
     cmp r9d, r10d                             ; C <= seq_len?
     ja .ss_bad                                ; C > seq_len -> ring does not fit the cache
; --- every check passed: publish the candidate config atomically, all three
; --- words together, so no rejecting path can leave a mixed triangle
     mov eax, [rel ss_cand_s]                  ; candidate S, validated above
     mov [rel g_stream_sinks], eax             ; store S
     mov eax, [rel ss_cand_w]                  ; candidate W
     mov [rel g_stream_window], eax            ; store W
     mov [rel g_stream_cap], r9d               ; store C = S + W, r9d survives the cmp
     mov [rel g_stream_win_valid], eax         ; tracker starts full: unread below C, exact at the first wrapped read, eax = W from the store above
     mov dword [rel g_stream_otk_live], 0      ; no recorded history until this session's prompt fill publishes it
     mov dword [rel g_stream_remove_warm], 0   ; fresh enable: tracking warm from token 0, a KV load sets it again
     mov rax, [rel g_stream_remove_ids]        ; ring from a previous session?
     test rax, rax                             ; NULL = first enable, skip the free
     jz .ss_ring_alloc                         ; nothing to free -> straight to allocation
     mov rcx, rax                              ; arg = old ring base, loaded above; rcx not overwritten between
     SS_RAW_CALL mdb_free_mem                  ; free a previous session's ring, re-enable path, OVERWRITES r9
     mov qword [rel g_stream_remove_ids], 0    ; zero-pin, house free pattern
.ss_ring_alloc:
     mov ecx, [rel g_stream_cap]               ; reload C from BSS, r9d is dead after mdb_free_mem
     shl rcx, 2                                ; rcx = C * 4 bytes, one dword per cache slot
     SS_RAW_CALL mdb_alloc_mem                 ; rax = ring base, fbm fallback FATALs internally on true OOM
     mov [rel g_stream_remove_ids], rax        ; publish the new ring base BEFORE the mode goes live
     mov dword [rel g_stream_remove_count], 0  ; pending log empty at, re enable
     mov dword [rel g_stream_enable], 1        ; enable LAST, value necessarily 1 on this path, ecx was consumed as the allocator's size argument : the ring is allocated, the config stored, no consumer can ever see enable=1 with ring=NULL
     xor eax, eax                              ; 0 = accepted
     ENGINE_LEAVE                              ; release the zone, pairs with ENGINE_GATE_RAW above
     ret                                       ; return 0 to the host
.ss_disable:
     cmp dword [rel g_stream_enable], 0        ; was streaming even on?
     je .ss_disable_store                      ; no -> plain no-op flag store
     mov rax, [rel g_inf_ctx_block]            ; rax = ctx, alive per the guard above
     mov r9d, [rax + CTX_POS]                  ; r9d = session position
     cmp r9d, [rel g_stream_cap]               ; pos > C: the cache holds SLOT-ordered ring rows --
     ja .ss_bad                                ; legacy reads [0..pos as position-ordered -> refuse, -1
.ss_disable_store:
     mov [rel g_stream_enable], ecx            ; store 0; the linear layout below C stays position-ordered.
; ring + pending count stay allocated/standing, the next enable
; flushes the batch, frees this ring and allocates a fresh one; destroy frees it
     xor eax, eax                              ; 0 = accepted, legal at any pos <= C
     ENGINE_LEAVE                              ; release the zone
     ret                                       ; return 0 to the host
.ss_bad:
     mov eax, -1                               ; -1 = rejected, bad args, overflow, or capacity
     ENGINE_LEAVE                              ; release the zone
     ret                                       ; return -1 to the host
.ss_midstream:
     FATAL "public API: v7_set_streaming cannot re-map a session in progress (ctx->pos > 0)"
.ss_not_init:
     FATAL "public API: v7_set_streaming called before v7_initialize_inference"

; %unmacro cleanup: file-local names leave scope here, house convention --
%unmacro SS_RAW_CALL 1                         ; file-local parity-call macro, all four invocations are above
