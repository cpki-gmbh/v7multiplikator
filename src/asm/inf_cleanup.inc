; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define G_VIT_PTR_COUNT 11                     ; qwords in the g_vit pointer block freed by INVALIDATE_VISION_STATE

%macro FREE_BSS 1
     cmp qword [%1], 0                         ; owned block ever allocated?
     je %%skip                                 ; no, partial init / prior free, nothing to release
     mov rcx, [%1]                             ; arg1: owned block base, non-NULL by the gate
     call mdb_free_mem                         ; release, VirtualFree or fbm unmap, transparent ; FATALs on NULL
     mov qword [%1], 0                         ; zero-pin: idempotent, no double-free
%%skip:
%endmacro
inf_cleanup_alloc:
     SF_START inf_cleanup_alloc, rdi           ; Prologue, save the 1 non-volatile actually used, rdi=rep stosq dest; rbx/rsi/r12-r15 never touched
     SF_COMMIT                                 ; Commit stack frame
     call cycle_tuner_save_blocks              ; cycle tuner: save every parameter block to v7_tune_param_<n>_<d>, then free + zero-pin
     call save_sampler_percentages             ; Persist the greedy fork ratio beside the 64 KiB tuning block files
     call save_stoch_percentages               ; Persist three stochastic adaptive splits in separate 12-byte file
; Free the 7 owned inference blocks, pointer-gated: NULL = never allocated
; under a partial init, or zero-pinned by a prior destroy
     FREE_BSS g_inf_ctx_block                  ; InferenceContext, tokenizer+sampler+header
     FREE_BSS g_inf_qkv_desc                   ; fused QKV descriptor block, single resource holder
     FREE_BSS g_inf_qkv_data                   ; fused QKV weight data buffer
     FREE_BSS g_inf_qt_desc                    ; QuantizedTensor descriptor array
     FREE_BSS g_inf_key_cache                  ; release both packed KV planes through their base
     mov qword [rel g_inf_kv_scratch], 0       ; clear the interior RunState scratch view
     FREE_BSS g_sys_prompt_owned               ; owned opener system prompt, freed+pointer-cleared; the save slot zeroes in chat_session reset
     FREE_BSS g_generate_kv_owner              ; merged KV block leaked by a caught FATAL, gated free + zero-pin, same as every owned block
     FREE_BSS g_inf_rs_block                   ; RunState working-buffer block, KV caches excluded, own blocks
     FREE_BSS g_inf_transformer                ; Transformer struct
     FREE_BSS g_ar_resized_buf                 ; anyres: resized canvas leaked by a caught FATAL mid-encode, success path zero-pins it
     FREE_BSS g_ar_tile_rgb                    ; anyres: per-tile RGB buffer, same caught-FATAL class
     FREE_BSS g_ar_tile_res                    ; anyres: per-tile planar float buffer
     FREE_BSS g_ar_tile_patches                ; anyres: per-tile patch features buffer
     FREE_BSS g_ar_tile_vit_out                ; anyres: per-tile ViT output buffer
     FREE_BSS g_ar_proj_inter                  ; anyres: projector FC1 intermediate buffer
     FREE_BSS g_ar_all_features                ; anyres: all-tiles feature array
     FREE_BSS g_ar_final_emb                   ; anyres: final embedding buffer, transferred to g_vision_embeddings on success, free here only fires on a mid-turn FATAL
     mov qword [g_qkv_bias], 0                 ; detach QKV bias view, points into DB blob; never freed
     FREE_BSS g_vit_scratch_block              ; ViT scratch block, NULL = text-only model
; Zero all ViT pointer views into the freed block, 11 contiguous qwords
     lea rdi, [g_vit_ptr_hidden]               ; rdi = start of the contiguous g_vit_ptr_* block
     xor eax, eax                              ; rax = fill value, zero
     mov ecx, G_VIT_PTR_COUNT                  ; rcx = 11 qwords to clear, g_vit_ptr_* block
     rep stosq                                 ; zero all ViT scratch globals
     INVALIDATE_VISION_STATE                   ; free the vision embedding pool + DeepStack buffers + zero all vision state, gates inside the macro
     mov qword [rel g_vis_target_w], 0         ; force lazy re-init from the next model's DB
     mov qword [rel g_vis_target_h], 0         ; 0 = not loaded sentinel
     FREE_BSS g_vis_scaled_buf                 ; release the model's fixed-size RGB buffer
     call v7s_mrope_restore                    ; restore base tables before freeing either allocation
     cmp dword [fused_rope_owned], 0           ; tables internally owned?
     jz .skip_rope_free                        ; no, external pointers, never free
     mov rcx, [fused_rope_cos]                 ; arg1: cos table, owned, non-NULL
     call mdb_free_mem                         ; free the cos table, FATALs internally on an invalid pointer
     mov rcx, [fused_rope_sin]                 ; arg1: sin table, owned, non-NULL
     call mdb_free_mem                         ; free the sin table, FATALs internally on an invalid pointer; a failed free terminates inside, so no zero-pin between the frees is reachable
     mov qword [fused_rope_cos], 0             ; zero-pin cos
     mov qword [fused_rope_sin], 0             ; zero-pin sin
     mov dword [fused_rope_owned], 0           ; clear owned flag, tables gone, prevents NULL+owned FATAL on next create
.skip_rope_free:
     FREE_BSS g_mrope_scratch_cos              ; release both scratch planes via their owning base
     mov qword [rel g_mrope_scratch_sin], 0    ; clear the interior pointer
     mov qword [rel g_mrope_scratch_bytes], 0  ; invalidate the scratch capacity
     FREE_BSS g_stream_remove_ids              ; StreamingLLM eviction ring, NULL = never allocated
     xor eax, eax                              ; RAX = 0: shared zero source for the BSS resets below
     mov [rel g_stream_enable], eax            ; a re-initialized engine starts legacy
     mov [rel g_stream_enable_saved], eax      ; the generate swap slot resets too
     mov [rel g_stream_sinks], eax             ; Session cleanup invalidates the ring geometry; the setter recomputes S/W/C.
     mov [rel g_stream_window], eax            ; Session cleanup invalidates the ring geometry
     mov [rel g_stream_cap], eax               ; Session cleanup invalidates the ring geometry
     mov [rel g_stream_remove_warm], eax       ; clear the post-load warm-up latch
     mov [rel g_stream_remove_count], eax      ; pending evictions are session state, drop
     mov [rel g_stream_remove_busy], eax       ; a caught FATAL inside a flush must not wedge the reentrancy gate, a re-initialized engine flushes again
     mov [rel g_stream_win_valid], eax         ; the tracker republishes at enable/load/reset
     mov [rel g_stream_otk_live], eax          ; the id history republishes at the chat fill
     mov [rel g_user_prompt], rax              ; remove a dangling caller-pointer publish, chat_finish only runs on the success path
     mov [rel g_tpl_t1_base], rax              ; restore the "init has not run" guard: after destroy the DB blob is gone, so v7s_enable_thinking must FATAL again instead of folding stale pointers
     mov [rel g_mrope_run], eax                ; reset running M-RoPE counter
     mov [rel g_mrope_delta], eax              ; reset generation rope delta
     mov [rel g_mrope_bound], eax              ; reset stale M-RoPE bound, VLM -> text switch: token-id-0 scan + grid math on a text model
     mov [rel g_mrope_bound_loaded], eax       ; force the bound lookup from the NEXT model's DB on first prepare
     mov [rel g_raw_generation], eax           ; clear stale raw-mode flag, public v7_generate FATAL-leak: pseudo-exception unwind skips its per-call clear
     mov [rel g_cached_pe_gh], eax             ; pos-embed cache: grid_h
     mov [rel g_cached_pe_gw], eax             ; pos-embed cache: grid_w
     mov [rel g_cached_pe_tbl], rax            ; pos-embed cache: table identity
     mov [rel g_cached_pe_pos], rax            ; pos-embed cache: buffer identity
     mov [rel g_cached_rb_gh], eax             ; rope-build cache: grid_h
     mov [rel g_cached_rb_gw], eax             ; rope-build cache: grid_w
     mov [rel g_vrb_last_cos], rax             ; rope-build cache: cos buffer identity
     mov [rel g_vrb_last_sin], rax             ; rope-build cache: sin buffer identity
     mov [rel g_vrb_theta_bits], eax           ; inv_freq cache: force theta-keyed rebuild, stale-safe
     SF_END rdi                                ; Restore stack frame and return
     global v7s_inference_destroy              ; Export destroy entry point
v7s_inference_destroy:
     SF_START v7s_inference_destroy            ; no non-volatile is written in this body, only ecx + memory operands, so the frame saves none
     SF_COMMIT                                 ; Commit stack frame
; === Step 0: the lifetime tuner proof, printed BEFORE any teardown ===
     call cycle_tuner_print_totals             ; "cycle tuner: N episode(s), M promotion(s)"
; === Step 1: Shut down pool thread pool BEFORE freeing any memory ===
     call v7s_shutdown                         ; idempotent: polls workers_alive to observed zero
; never races an executing pool thread, teardown race
; === Step 1: Free all owned inference blocks ===
     call inf_cleanup_alloc                    ; Free all BSS-tracked allocations
; === Step 2: Destroy microdb hash table ===
     call ht_destroy                           ; Free microdb heap memory
; === Step 3: Re-init microdb for next use ===
     call ht_init                              ; Reset hash table for future inference_create
; Don't db_load, the caller will do that before next inference_create
     mov dword [rel v7_engine_dead], 0         ; engine active again from the API contract's view
     mov byte [rel engine_death_reason], 0     ; reason string back to ""
     mov dword [rel engine_call_depth], 0      ; landing zone parked, no active zone
     mov dword [rel engine_gate_claim], 0      ; release the atomic zone claim, next gated call may re-claim
     FREE_BSS sl_lg_heap_base                  ; SectorLogo heap: freed with the engine, lg_init allocates a fresh arena; leaving it would leak one heap per destroy/reinit cycle
     mov dword [rel sl_lg_initialized], 0      ; reset the Logo init latch: a caught FATAL leaves the world half-mutated, and without this the next hook/interp would skip lg_init and dispatch against it, same reset as the v7_free_inference path
     SF_END                                    ; Return to caller, all resources freed
; %unmacro / %undef cleanup: file-local names leave scope here, house convention --
; FREE_BSS first, its %unmacro arity is the macro's parameter count, 1, then the %define.
%unmacro FREE_BSS 1                            ; release the per-pointer free macro while its %define still exists
%undef G_VIT_PTR_COUNT

