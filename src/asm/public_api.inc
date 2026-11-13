; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define PUB_RBP_ARG5 48                        ; 5th incoming arg via rbp after the frame
%define PUB_RBP_ARG6 56                        ; 6th incoming arg via rbp after the frame
%define PUB_RBP_ARG7 64                        ; 7th incoming arg via rbp after the frame
%define PUB_RBP_ARG8 72                        ; 8th incoming arg via rbp after the frame
%define PUB_F_CHAT 48                          ; v7s_chat outgoing: 32 shadow + 2 stack args, len, reset = 48, already 16-aligned
%define PUB_F_ENC 40                           ; raw-entry outgoing: 32 shadow + 1 stack arg = 40, already 16-aligned, no pad
%define PUB_F_LEAF 40                          ; raw-entry outgoing: 32 shadow + 8 pad to 16-alignment = 40
%define PUB_F_IMG 80                           ; v7s_chat_image_mem outgoing: 32 shadow + 5 stack args = 72, + 8 pad to 16-alignment = 80
%define PUB_F_GIMG 64                          ; v7s_generate_image_mem outgoing: 32 shadow + 4 stack args = 64, already 16-aligned
%define PUB_OUT_ARG5 32                        ; outgoing 5th arg slot, first past shadow
%define PUB_OUT_ARG6 40                        ; outgoing 6th arg slot
%define PUB_OUT_ARG7 48                        ; outgoing 7th arg slot
%define PUB_OUT_ARG8 56                        ; outgoing 8th arg slot
%define PUB_OUT_ARG9 64                        ; outgoing 9th arg slot

%macro PUB_SARG_TO_OUT 3
     mov %1, [rbp + PUB_RBP_ARG%2]             ; incoming arg %2 via the stable rbp slot
     mov [rsp + PUB_OUT_ARG%3], %1             ; internal arg %3
%endmacro

%macro PUB_RAW_CALL 2
     ENGINE_GATE_RAW %1                        ; dead-check + landing zone, raw entry: rcx/rdx/r8/r9 pass through
     PUB_CPU_GATE                              ; refuse to run on hardware below the v7 baseline
     sub rsp, PUB_F_LEAF                       ; raw-entry outgoing frame: 32 shadow + align pad
     call %2                                   ; internal callee, args already in the public order
     add rsp, PUB_F_LEAF                       ; restore stack
     ENGINE_LEAVE                              ; park the landing zone, normal path
     ret                                       ; return the callee's result in rax/eax
%endmacro
     section .text
     global v7_init                            ; export public wrapper
v7_init:
     PUB_RAW_CALL 1, V7_init                   ; rcx = db_path passes through, the gate writes rax and flags only
     global v7_db_load                         ; export public wrapper
v7_db_load:
     PUB_RAW_CALL 1, db_load                   ; rcx = path passes through, db_load/db_save use the caller's filename and fall back to g_db_filename only when it is NULL
     global v7_db_save                         ; export public wrapper
v7_db_save:
     PUB_RAW_CALL 1, db_save                   ; rcx = path passes through, same NULL-fallback contract as db_load
     global v7_initialize_inference            ; export public wrapper
v7_initialize_inference:
     ENGINE_GATE_RAW 1                         ; dead-check + landing zone, pristine raw entry
     PUB_CPU_GATE                              ; refuse to run on hardware below the v7 baseline
     cmp qword [rel g_inf_ctx_block], 0        ; context already alive?
     jne .v7ii_leave                           ; yes, idempotent success, skip create
     sub rsp, PUB_F_LEAF                       ; raw-entry frame: 32 shadow + align, call below
     call v7s_inference_create                 ; no, create, FATALs on any failure; stores g_inf_ctx_block itself
     add rsp, PUB_F_LEAF                       ; restore stack
.v7ii_leave:
     ENGINE_LEAVE                              ; park the landing zone, both paths converge here
     mov eax, 1                                ; return true, ready
     ret                                       ; no non-volatiles touched
     global v7_free_inference                  ; export public wrapper
v7_free_inference:
     jmp v7s_inference_destroy                 ; pure alias, v7_shutdown pattern : parameterless void callee, own SF frame; the callee's return path IS the public return
     global v7_chat                            ; export public wrapper
v7_chat:
     SF_START v7_chat                          ; frame, defines global+label; no non-volatiles overwritten
     SF_COMMIT                                 ; finalize frame
     ENGINE_GATE_SFR 0                         ; dead-check + landing zone, gate preserves rcx/rdx/r8/r9
     PUB_CPU_GATE                              ; refuse to run on hardware below the v7 baseline
     test rcx, rcx                             ; user prompt, crash-capable caller mistake: a NULL prompt strlen's inside the engine
     jz .v7c_null_prompt                       ; -> FATAL
     sub rsp, PUB_F_CHAT                       ; outgoing frame: shadow + 2 stack args
     PUB_SARG_TO_OUT eax, 5, 6                 ; internal arg6: reset
     mov [rsp + PUB_OUT_ARG5], r9d             ; internal arg5: output_len, direct from public arg4
     mov r9, r8                                ; internal arg4: output buffer, public arg3
     mov r8, rdx                               ; internal arg3: system prompt, public arg2
     mov rdx, rcx                              ; internal arg2: user prompt, public arg1
     mov rcx, [rel g_inf_ctx_block]            ; init-order guard + internal arg1 in ONE load
     test rcx, rcx                             ; 0 = no context
     jz .v7c_not_init                          ; -> FATAL
     call v7s_chat                             ; run the chat loop, internal, unchanged
     add rsp, PUB_F_CHAT                       ; restore outgoing frame
     ENGINE_LEAVE                              ; park the landing zone, normal path
     SF_END                                    ; epilogue, no non-volatiles
.v7c_not_init:
     FATAL "public API: v7_chat called before v7_initialize_inference" ; contract violation, FATAL
.v7c_null_prompt:
     FATAL "public API: the user prompt must be non-NULL (an empty string is legal)" ; crash-capable caller mistake
     global v7_generate                        ; export public wrapper
v7_generate:
     SF_START v7_generate                      ; frame, defines global+label; no non-volatiles overwritten
     SF_COMMIT                                 ; finalize frame
     ENGINE_GATE_SFR 0                         ; dead-check + landing zone, gate preserves rcx/rdx/r8/r9
     PUB_CPU_GATE                              ; refuse to run on hardware below the v7 baseline
     test rcx, rcx                             ; prompt, crash-capable caller mistake: NULL strlen's inside the engine
     jz .v7g_null_prompt                       ; -> FATAL
     mov rax, r9                               ; park the raw-mode flag, r9d is overwritten by the arg shuffle below
     mov r9d, r8d                              ; internal arg4: output_len, public arg3
     mov r8, rdx                               ; internal arg3: output buffer, public arg2
     mov rdx, rcx                              ; internal arg2: prompt, public arg1
     mov rcx, [rel g_inf_ctx_block]            ; init-order guard + internal arg1 in ONE load
     test rcx, rcx                             ; 0 = no context
     jz .v7g_not_init                          ; -> FATAL, unwind skips the publish, the flag cannot leak into a later session
     mov [rel g_raw_generation], eax           ; publish mode only after the guard passes, no calls between park and publish
     call v7s_generate                         ; single-turn generate, separate file-backed KV; SF_COMMIT already provides shadow
     mov dword [rel g_raw_generation], 0       ; clear for normal paths, FATAL unwind leaks past, inf_cleanup resets at destroy
     ENGINE_LEAVE                              ; park the landing zone, normal path
     SF_END                                    ; epilogue
.v7g_not_init:
     FATAL "public API: v7_generate called before v7_initialize_inference" ; contract violation
.v7g_null_prompt:
     FATAL "public API: the prompt must be non-NULL (an empty string is legal)" ; crash-capable caller mistake
; --- Border validation for the two image entries, the public tier is the
; outermost layer, checks are sanctioned HERE and only here : both
; dimensions positive and below 65536, the vit DDA runs 16.16 fixed point --
; a dimension of 65536+ wraps the accumulators; this is the only guard for
; that contract since the old vit_input_stage FATALs moved here, the w*h
; pixel count inside 32 bits, and the pixel buffer non-NULL. A negative or
; 32-bit-wrapped dimension makes the vit stage allocate from a wrapped count
; and read the FULL w*h, an OOB read; that is the crash class this check
; exists for. Sizes that are not patch multiples are legal: the vit grid
; floors the division and crops to the patch lattice, every read stays
; inside w*h by construction.Overwrites only rax/r11, rcx/rdx/r8/r9, the
; incoming argument registers survive.
; %1 = img_w, 32-bit register, %2 = img_h, 32-bit register or rbp slot,
; %3 = raw_rgb pointer register, r8 for v7_chat_image, rdx for v7_generate_image

%macro PUB_VALIDATE_IMAGE 3
     test %1, %1                               ; img_w sign
     jle %%bad_w                               ; <= 0 is not an image
     cmp %1, PUB_IMG_MAXDIM                    ; img_w < 65536?, 16.16 DDA wrap limit
     jae %%bad_dim                             ; 65536+ would wrap the vit DDA accumulators
     mov eax, %2                               ; img_h
     test eax, eax                             ; img_h sign
     jle %%bad_h                               ; <= 0 is not an image
     cmp eax, PUB_IMG_MAXDIM                   ; img_h < 65536?
     jae %%bad_dim                             ; 65536+ would wrap the vit DDA accumulators
     mov r11d, eax                             ; h, already validated nonnegative in eax
     mov eax, %1                               ; w, zero-extended (see note)
     imul rax, r11                             ; w*h in 64 bits, the 32-bit wrap cannot hide
     cmp rax, PUB_IMG_MAXPX                    ; pixel count must fit int32, the vit stage reads it as a dword
     ja %%bad_px                               ; wrapped pixel count would OOB the vit stage
     test %3, %3                               ; raw RGB buffer
     jz %%bad_rgb                              ; a NULL pixel buffer cannot be dereferenced
     jmp %%ok                                  ; all contracts hold, skip the FATAL blocks, fall-through into them would terminate every valid image
%%bad_w:
     FATAL "public API: img_w must be positive"
%%bad_h:
     FATAL "public API: img_h must be positive"
%%bad_dim:
     FATAL "public API: img_w and img_h must be below 65536 (16.16 fixed-point limit)"
%%bad_px:
     FATAL "public API: img_w * img_h exceeds the 32-bit pixel count"
%%bad_rgb:
     FATAL "public API: the image entries require a non-NULL raw_rgb buffer"
%%ok:
%endmacro
%define PUB_IMG_MAXPX 0x7FFFFFFF               ; largest pixel count the dword-sized vit stage can hold
%define PUB_IMG_MAXDIM 65536                   ; per-dimension limit: the vit DDA runs 16.16 fixed point, a dimension of 65536+ wraps the accumulators; 65535 itself is legal
     global v7_chat_image                      ; export public wrapper
v7_chat_image:
     SF_START v7_chat_image                    ; frame, defines global+label; no non-volatiles overwritten
     SF_COMMIT                                 ; finalize frame, no locals, direct shuffle, v7_chat pattern
     ENGINE_GATE_SFR 1                         ; dead-check + landing zone, int return; 0 = check is_dead
     PUB_CPU_GATE                              ; refuse to run on hardware below the v7 baseline
     PUB_VALIDATE_IMAGE r9d, dword [rbp + PUB_RBP_ARG5], r8 ; img_w / img_h / raw_rgb border contract
     test rcx, rcx                             ; user prompt, rcx still public arg1, the image macro overwrites only rax/r11
     jz .v7ci_null_prompt                      ; -> FATAL
     sub rsp, PUB_F_IMG                        ; outgoing frame FIRST, shadow + 5 stack args
     mov [rsp + PUB_OUT_ARG5], r9d             ; internal arg5: img_w, direct from public arg4
     PUB_SARG_TO_OUT eax, 5, 6                 ; internal arg6: img_h
     PUB_SARG_TO_OUT rax, 6, 7                 ; internal arg7: output buffer, pointer, 64-bit slot
     PUB_SARG_TO_OUT eax, 7, 8                 ; internal arg8: output_len
     PUB_SARG_TO_OUT eax, 8, 9                 ; internal arg9: reset
     mov r9, r8                                ; internal arg4: raw RGB, public arg3
     mov r8, rdx                               ; internal arg3: system prompt, public arg2
     mov rdx, rcx                              ; internal arg2: user prompt, public arg1
     mov rcx, [rel g_inf_ctx_block]            ; init-order guard + internal arg1 in ONE load
     test rcx, rcx                             ; 0 = no context
     jz .v7ci_not_init                         ; -> FATAL
     call v7s_chat_image_mem                   ; full vision pipeline + chat turn
     add rsp, PUB_F_IMG                        ; restore outgoing frame
     xor eax, eax                              ; return 0, v7s_chat_image_mem forwards v7s_chat's void residue, so the DEFINING write for this return lives here, in the wrapper
     ENGINE_LEAVE                              ; park the landing zone, normal path
     SF_END                                    ; epilogue, status = 0, per the header note
.v7ci_not_init:
     FATAL "public API: v7_chat_image called before v7_initialize_inference" ; contract violation
.v7ci_null_prompt:
     FATAL "public API: the user prompt must be non-NULL (an empty string is legal)" ; crash-capable caller mistake
     global v7_generate_image                  ; export public wrapper
v7_generate_image:
     SF_START v7_generate_image                ; frame, defines global+label; no non-volatiles overwritten
     SF_COMMIT                                 ; finalize frame, no locals, direct shuffle, v7_chat pattern
     ENGINE_GATE_SFR 0                         ; dead-check + landing zone, this wrapper saves no nonvolatiles
     PUB_CPU_GATE                              ; refuse to run on hardware below the v7 baseline
     PUB_VALIDATE_IMAGE r8d, r9d, rdx          ; img_w / img_h / raw_rgb border contract
     test rcx, rcx                             ; prompt, rcx still public arg1, the image macro overwrites only rax/r11
     jz .v7gi_null_prompt                      ; -> FATAL
     sub rsp, PUB_F_GIMG                       ; outgoing frame FIRST, shadow + 4 stack args
     mov [rsp + PUB_OUT_ARG5], r8d             ; internal arg5: img_w, direct from public arg3
     mov [rsp + PUB_OUT_ARG6], r9d             ; internal arg6: img_h, direct from public arg4
     PUB_SARG_TO_OUT rax, 5, 7                 ; internal arg7: output, pointer, 64-bit slot
     PUB_SARG_TO_OUT eax, 6, 8                 ; internal arg8: output_len
     mov r9, rdx                               ; internal arg4: raw RGB, public arg2
     xor r8d, r8d                              ; internal arg3: system_prompt = NULL, generate path FATALs on non-NULL
     mov rdx, rcx                              ; internal arg2: user prompt, public arg1
     mov rcx, [rel g_inf_ctx_block]            ; init-order guard + internal arg1 in ONE load
     test rcx, rcx                             ; 0 = no context
     jz .v7gi_not_init                         ; -> FATAL
     call v7s_generate_image_mem               ; full vision pipeline + single-turn generate
     add rsp, PUB_F_GIMG                       ; restore outgoing frame
     ENGINE_LEAVE                              ; park the landing zone, normal path
     SF_END                                    ; epilogue
.v7gi_not_init:
     FATAL "public API: v7_generate_image called before v7_initialize_inference" ; contract violation
.v7gi_null_prompt:
     FATAL "public API: the prompt must be non-NULL (an empty string is legal)" ; crash-capable caller mistake
; --- Common body: load the ctx singleton into rax, FATAL when not init ---
; Single BSS load; rax stays live as InferenceContext* for the caller's use.

%macro PUB_LOAD_CTX 1
     ENGINE_GATE_RAW 1                         ; dead-check + landing zone, raw entry: every wrapper using this macro is SF-less.Runs BEFORE the ctx load, so the rax the gate overwrites is re-established below
     PUB_CPU_GATE                              ; refuse to run on hardware below the v7 baseline
     mov rax, [rel g_inf_ctx_block]            ; context alive?, one load, rax = ctx stays live
     test rax, rax                             ; 0 = not initialized
     jnz %%done                                ; alive -> continue
     FATAL %1                                  ; contract violation, FATAL, the gate above catches it: the host is informed, not terminated
%%done:
%endmacro
; --- Common body: load ctx -> sampler pointer into rax, FATAL when not init ---

%macro PUB_SAMPLER_CTX 0
     PUB_LOAD_CTX "public API: sampler setter called before v7_initialize_inference" ; guard + rax = ctx
     mov rax, [rax + CTX_SAMPLER]              ; rax = Sampler*
%endmacro
; void v7_set_sampler_temperature,float t, float arg arrives in xmm0
     global v7_set_sampler_temperature         ; export public wrapper
v7_set_sampler_temperature:
     PUB_SAMPLER_CTX                           ; rax = sampler, FATALs when uninitialized
; SANCTIONED DATA-FIXING BOUNDARY, owner directive : the sampler kernel takes
; t == 0, greedy or t >= 1e-6 by contract, with no downstream floor. The
; PUBLIC API SETTER is the one allowed correction point: 0 < t < 1e-6 is
; fixed up to 1e-6 here, silently. t == 0 stores verbatim, greedy .
     vpxor xmm1, xmm1, xmm1                    ; xmm1 = 0.0
     vucomiss xmm0, xmm1                       ; t vs 0.0, unordered sets PF
     jp .vst_fix                               ; NaN: unordered, vmaxss below maps NaN to the epsilon, NaN -> src2
     je .vst_store                             ; t == 0: greedy, store verbatim
.vst_fix:                                      ; ordered and nonzero, positive or negative : raise sub-epsilon/negative to the 1e-6 contract
     vucomiss xmm0, [rel pub_float_inf]        ; t vs +infinity
     jp .vst_fix_raise                         ; unordered, NaN is NOT +inf, keep the silent fix-up contract
     je .vst_inf_fatal                         ; ordered equal to +inf: every logit would divide to zero, refuse
.vst_fix_raise:
     vmaxss xmm0, xmm0, [rel rpe_epsilon]      ; raise sub-epsilon, or NaN t to the 1e-6 contract
.vst_store:
     vmovss [rax + SAM_TEMPERATURE], xmm0      ; store temperature
     ENGINE_LEAVE                              ; park the landing zone, normal path
     ret                                       ; plain return
.vst_inf_fatal:
     FATAL "public API: v7_set_sampler_temperature requires a finite value" ; +inf is a caller bug
; void v7_set_sampler_topp,float p
     global v7_set_sampler_topp                ; export public wrapper
v7_set_sampler_topp:
     PUB_SAMPLER_CTX                           ; rax = sampler
     vmovss [rax + SAM_TOPP], xmm0             ; store top-p
     ENGINE_LEAVE                              ; park the landing zone, normal path
     ret                                       ; plain return
; void v7_set_sampler_presence_penalty,float p
     global v7_set_sampler_presence_penalty    ; export public wrapper
v7_set_sampler_presence_penalty:
     PUB_SAMPLER_CTX                           ; rax = sampler
     vucomiss xmm0, xmm0                       ; NaN test: x != x iff NaN, sets PF
     jp .v7ssp_nan                             ; NaN logits sentinel every later sample, reject
     vandps xmm1, xmm0, [rel pub_float_absmask] ; xmm1 = |p|, sign bit cleared
     vucomiss xmm1, [rel pub_pen_limit]        ; |p| vs 2.0
     ja .v7ssp_range                           ; beyond [-2, 2], inf included would drown the logits, refuse
     vmovss [rax + SAM_PRESENCE_PENALTY], xmm0 ; store presence penalty
     ENGINE_LEAVE                              ; park the landing zone, normal path
     ret                                       ; plain return
.v7ssp_nan:
     FATAL "public API: v7_set_sampler_presence_penalty requires a finite value" ; non-finite is a caller bug
.v7ssp_range:
     FATAL "public API: v7_set_sampler_presence_penalty requires a value between -2 and 2" ; magnitude contract
; void v7_set_sampler_topk,int k, only 0 is legal: v7s_sample has NO top-k parameter
     global v7_set_sampler_topk                ; export public wrapper
v7_set_sampler_topk:
     PUB_SAMPLER_CTX                           ; rax = sampler, the gate catches the FATAL below: the host is informed, not terminated
     test ecx, ecx                             ; top-k is not in the v7s_sample ABI; only 0, disabled is legal
     jz .v7stk_store                           ; yes: k == 0, disabled is the one legal value
     FATAL "public API: v7_set_sampler_topk requires 0 (top-k is not supported by v7s_sample)"
.v7stk_store:
     mov [rax + SAM_TOPK], ecx                 ; store top-k, always 0 here
     ENGINE_LEAVE                              ; park the landing zone, normal path
     ret                                       ; plain return
; void v7_set_sampler_seed,unsigned __int64 seed
     global v7_set_sampler_seed                ; export public wrapper
v7_set_sampler_seed:
     PUB_SAMPLER_CTX                           ; rax = sampler
     mov [rax + SAM_RNG_STATE], rcx            ; store PRNG state, qword
     ENGINE_LEAVE                              ; park the landing zone, normal path
     ret                                       ; plain return
     global v7_set_quantize                    ; export public wrapper
v7_set_quantize:
     PUB_LOAD_CTX "public API: v7_set_quantize called before v7_initialize_inference" ; guard + rax = ctx
     cmp ecx, 1                                ; The engine supports only mode 0, F32 and mode 1, Q8 .
     ja .v7sq_bad_mode                         ; unsigned compare: negatives wrap huge and land here with >= 2
     mov [rax + CTX_QUANTIZE], ecx             ; store quantize mode
     ENGINE_LEAVE                              ; park the landing zone, normal path
     ret                                       ; plain return
.v7sq_bad_mode:
     FATAL "public API: v7_set_quantize requires 0 (F32) or 1 (Q8)" ; unknown mode is a caller bug
; int v7_get_quantize,void, returns the active forward mode: 1 = Q8 matmul, 0 = F32
; Bare query, no gate: it only reads the context dword, it cannot crash, and
; it must stay callable from inside the Logo world mid-dispatch (the tuner's
; owned-slot resolution) -- a landing-zone install there would clobber the
; still-active outer gate, exactly what killed v_stream_validate's rewrite turn.
     global v7_get_quantize                    ; export public wrapper
v7_get_quantize:
     mov rax, [rel g_inf_ctx_block]            ; context singleton
     test rax, rax                             ; not initialized -> mode 0 is the honest answer
     jz .v7gq_default                          ; no context: return the default mode
     mov eax, [rax + CTX_QUANTIZE]             ; read the active forward mode
     ret                                       ; eax = 0 (F32) or 1 (Q8)
.v7gq_default:
     xor eax, eax                              ; uninitialized: F32 is the engine default mode
     ret                                       ; eax = 0, the query never fails
; void v7_set_max_gen,int cap, replaces raw ctx->max_gen writes
     global v7_set_max_gen                     ; export public wrapper
v7_set_max_gen:
     PUB_LOAD_CTX "public API: v7_set_max_gen called before v7_initialize_inference" ; guard + rax = ctx
     test ecx, ecx                             ; cap sign bit
     js .v7smg_negative                        ; the gen loop compares unsigned, a negative cap would read as unlimited
     mov [rax + CTX_MAX_GEN], ecx              ; store generation cap
     ENGINE_LEAVE                              ; park the landing zone, normal path
     ret                                       ; plain return
.v7smg_negative:
     FATAL "public API: v7_set_max_gen requires a non-negative cap (0 = unlimited)" ; negative cap is a caller bug
; ============================================================================
; Position access
; ============================================================================
; int v7_get_position,void, replaces ctx->pos reads
     global v7_get_position                    ; export public wrapper
v7_get_position:
     PUB_LOAD_CTX "public API: v7_get_position called before v7_initialize_inference" ; guard + rax = ctx
     mov eax, [rax + CTX_POS]                  ; read current KV position
     ENGINE_LEAVE                              ; park the landing zone, normal path
     ret                                       ; plain return
; void v7_set_position,int pos, rewind-only; streaming rewind preserves only resident KV rows
     global v7_set_position                    ; export public wrapper
v7_set_position:
     PUB_LOAD_CTX "public API: v7_set_position called before v7_initialize_inference" ; guard + rax = ctx
     test ecx, ecx                             ; negative target position?
     js .v7sp_negative                         ; reject negative cache indices
     cmp ecx, [rax + CTX_POS]                  ; target > current position?
     jg .v7sp_forward                          ; reject uncomputed positions
     cmp dword [rel g_stream_enable], 0        ; StreamingLLM enabled?
     je .v7sp_store                            ; legacy linear KV, no ring state
     mov edx, [rax + CTX_POS]                  ; edx = original position P
     cmp ecx, edx                              ; target == P?
     je .v7sp_store                            ; no-op keeps tracker and warm latch unchanged
     cmp edx, [rel g_stream_cap]               ; has a wrapped write happened, P > C?
     jbe .v7sp_store                           ; no, original layout is still linear
     cmp ecx, [rel g_stream_cap]               ; target T <= C?
     jbe .v7sp_rebuild_linear                  ; yes, restore overwritten linear rows from live IDs so the physical layout is position-ordered again
; Wrapped rewind, T > C. Only the suffix of the current resident window
; preceding T is still available. After P-T discarded writes, the retained
; count becomes max(0, old_valid-(P-T)); attention already respects this
; tracker, and every forward repopulates one row until the window is full.
     sub edx, ecx                              ; edx = discarded token count D = P - T
     mov r8d, [rel g_stream_win_valid]         ; previous number of valid window rows
     sub r8d, edx                              ; remove the rows at/above T
     jnc .v7sp_valid_ready                     ; nonnegative count survives
     xor r8d, r8d                              ; discarded suffix consumed every valid row
.v7sp_valid_ready:
     mov [rel g_stream_win_valid], r8d         ; attention must not see the overwritten prefix
; Eviction IDs live in the same slots. Until those slots have been written
; by the new timeline, suppress eviction reports: they name discarded rows.
; Keep any existing post-load warm-up; a rewind moves its end by D writes.
     mov r9d, [rel g_stream_window]            ; W = rolling window capacity
     mov r10d, [rel g_stream_remove_warm]      ; remaining untrusted eviction-ID writes
     cmp r10d, r9d                             ; special long mute from an in-chat rewrite?
     ja .v7sp_store                            ; leave that mute intact, only shrink validity
     mov r11d, r9d                             ; r11d = W
     sub r11d, r10d                            ; remaining room in the warm latch
     cmp edx, r11d                             ; D >= remaining room?
     jae .v7sp_warm_full                       ; yes, full W writes must warm the ring
     add r10d, edx                             ; extend the previous warm-up by D
     jmp .v7sp_warm_floor                      ; now enforce the validity deficit
.v7sp_warm_full:
     mov r10d, r9d                             ; clamp warm-up at W, no more is needed
.v7sp_warm_floor:
     sub r9d, r8d                              ; W - new_valid = untrusted slots at target
     cmp r10d, r9d                             ; old warm-up large enough for those slots?
     cmovb r10d, r9d                           ; suppress all non-real eviction reports
     mov [rel g_stream_remove_warm], r10d      ; forward decrements this once per wrapped write
     jmp .v7sp_store                           ; cursor is derived by forward from its pos
; Rewind across C: before C, attention uses the linear [0..pos] layout.
; A wrapped session has overwritten slots in [S,C), so a tracker change
; cannot repair this case. Re-forward live token IDs [S,T) to restore KV.
; Sinks [0,S) are immutable. All forwarded positions are < C <= seq_len.
.v7sp_rebuild_linear:
     cmp dword [rel g_stream_otk_live], 0      ; are recorded token IDs [0,C) available?
     je .v7sp_missing_history                  ; after KV load they are not: fail rather than invent KV
     sub rsp, 72                               ; Win64 32-byte shadow, 24-byte locals, alignment
%define V7SP_CTX_QWORD 32                      ; local ctx pointer across forward calls
%define V7SP_TARGET_DWORD 40                   ; local rewind target T
%define V7SP_POS_DWORD 48                      ; local loop position p
%define V7SP_TOKEN_SCALE 4                     ; bytes per out_tokens entry, int32 IDs
     mov [rsp + V7SP_CTX_QWORD], rax           ; local ctx pointer across forward calls
     mov [rsp + V7SP_TARGET_DWORD], ecx        ; local rewind target T
     mov eax, [rel g_stream_sinks]             ; first mutable KV row = S
     mov [rsp + V7SP_POS_DWORD], eax           ; local loop position p
.v7sp_rebuild_loop:
     mov r8d, [rsp + V7SP_POS_DWORD]           ; p, third argument to forward
     cmp r8d, [rsp + V7SP_TARGET_DWORD]        ; p reached target T?
     jae .v7sp_rebuild_done                    ; [S,T) has been reconstructed
     mov rax, [rsp + V7SP_CTX_QWORD]           ; context reload after any forward call
     mov rdx, [rax + CTX_OUT_TOKENS]           ; token ID history (valid for p < C)
     mov edx, [rdx + r8*V7SP_TOKEN_SCALE]      ; token at absolute position p
     mov rcx, [rax + CTX_TRANSFORMER]          ; Transformer* = first forward argument
     cmp dword [rax + CTX_QUANTIZE], 0         ; current forward mode?
     jne .v7sp_rebuild_q8                      ; Q8 mode
     call forward_f32                          ; refill linear KV row p in FP32 mode
     jmp .v7sp_rebuild_advance                 ; shared increment
.v7sp_rebuild_q8:
     call forward_q8                           ; refill linear KV row p in Q8 mode
.v7sp_rebuild_advance:
     inc dword [rsp + V7SP_POS_DWORD]          ; p++
     jmp .v7sp_rebuild_loop                    ; repeat until all original prefix rows exist
.v7sp_rebuild_done:
     mov eax, [rel g_stream_window]            ; W: all linear window rows are now reconstructable
     mov [rel g_stream_win_valid], eax         ; first wrapped forward will see a complete window
     cmp dword [rel g_stream_remove_warm], eax ; special in-chat mute still active?
     ja .v7sp_rebuild_keep_warm                ; keep an intentional mute marker
     mov dword [rel g_stream_remove_warm], 0   ; ID ring becomes complete as [T,C) refills
.v7sp_rebuild_keep_warm:
     mov rax, [rsp + V7SP_CTX_QWORD]           ; reload context for the public position store
     mov ecx, [rsp + V7SP_TARGET_DWORD]        ; reload requested position T
     mov [rax + CTX_POS], ecx                  ; expose only the rebuilt rewind position
     add rsp, 72                               ; restore raw-entry stack pointer
%undef V7SP_CTX_QWORD
%undef V7SP_TARGET_DWORD
%undef V7SP_POS_DWORD
%undef V7SP_TOKEN_SCALE
     ENGINE_LEAVE                              ; release public API landing zone
     ret                                       ; success
.v7sp_store:
     mov [rax + CTX_POS], ecx                  ; store rewound position, slot is derived per forward
     ENGINE_LEAVE                              ; release landing zone
     ret                                       ; success
.v7sp_missing_history:
     FATAL "public API: wrapped rewind below C requires in-process token history (KV load has no token IDs)"
.v7sp_forward:
     FATAL "public API: v7_set_position cannot wind forward (unwritten KV slots)"
.v7sp_negative:
     FATAL "public API: v7_set_position requires a non-negative position"
; int v7_get_think_mode_active,void, 1 when thinking templates are selected
     global v7_get_think_mode_active           ; export public wrapper
v7_get_think_mode_active:
     PUB_LOAD_CTX "public API: v7_get_think_mode_active called before v7_initialize_inference" ; guard + rax = ctx
     movzx eax, byte [rel g_thinking_active]   ; eax = thinking-ON flag, set by v7_enable_thinking
     ENGINE_LEAVE                              ; park the landing zone, normal path
     ret                                       ; plain return
; int v7_get_gen_count,void, tokens generated so far this chat turn, live
     global v7_get_gen_count                   ; export public wrapper
v7_get_gen_count:
     PUB_LOAD_CTX "public API: v7_get_gen_count called before v7_initialize_inference" ; guard + rax = ctx
     mov eax, [rel g_live_gen_count]           ; eax = live per-turn generated-token count
     ENGINE_LEAVE                              ; park the landing zone, normal path
     ret                                       ; plain return
     global v7_enable_thinking                 ; export public wrapper
v7_enable_thinking:
     PUB_RAW_CALL 0, v7s_enable_thinking       ; ecx = on passes through; v7s_enable_thinking FATALs on a pre-create call
; void v7_set_worker_priority,int priority, switches all worker threads, parks the request when called before pool creation
; Gated: set_worker_priority_all FATALs when SetThreadPriority rejects a handle or the value, the host must be informed, not terminated.
     global v7_set_worker_priority             ; export public wrapper
v7_set_worker_priority:
     TRACE_STR "swpa: public entry (host called the setter)" ; debug: identifies the caller path, the env/apply-pending callers trace their own names
     PUB_RAW_CALL 0, set_worker_priority_all   ; ecx = priority passes through, void return: the v7_enable_thinking/v7_set_sidecar pattern
; int v7_db_add,const char* key, void* data, int len, unsigned int persist
; Gated: ht_add FATALs on a NULL key or an uninitialized table.
     global v7_db_add                          ; export public wrapper
v7_db_add:
     ENGINE_GATE_RAW 1                         ; dead-check + landing zone, rcx/rdx/r8/r9 pass through
     PUB_CPU_GATE                              ; refuse to run on hardware below the v7 baseline
     test r8d, r8d                             ; len, int sign bit
     js .v7da_bad_len                          ; the zero-extension below would hand ht_add a ~4GB length
     test rdx, rdx                             ; data pointer: NULL is legal only with len == 0
     jnz .v7da_args_ok                         ; non-NULL data, the normal path
     test r8d, r8d                             ; NULL data is acceptable only for a zero-length entry
     jnz .v7da_bad_data                        ; NULL data with bytes to copy, caller bug
.v7da_args_ok:
     sub rsp, PUB_F_LEAF                       ; raw-entry outgoing frame: 4 register args, no stack args
     mov r8d, r8d                              ; zero-extend the 32-bit len arg into the 64-bit register ht_add reads
     call ht_add                               ; internal add, same arg order, no shuffle needed
     add rsp, PUB_F_LEAF                       ; restore stack
     ENGINE_LEAVE                              ; park the landing zone, normal path
     ret                                       ; return ht_add's status, 0 on the dead path
.v7da_bad_len:
     FATAL "public API: v7_db_add requires a non-negative len" ; negative length is a caller bug
.v7da_bad_data:
     FATAL "public API: v7_db_add requires non-NULL data when len > 0" ; NULL data is a caller bug
; void* v7_db_find,const char* key, unsigned __int64* out_len
; Gated: ht_find FATALs on a NULL key or an uninitialized table.
     global v7_db_find                         ; export public wrapper
v7_db_find:
     PUB_RAW_CALL 1, ht_find                   ; rcx/rdx pass through; returns the blob pointer, 0 on the dead path
; unsigned __int64* v7_get_generation_statistic,void
     global v7_get_generation_statistic        ; export public wrapper
v7_get_generation_statistic:
     jmp v7s_get_generation_statistic          ; pure alias
; void v7_shutdown,void
     global v7_shutdown                        ; export public wrapper
v7_shutdown:
     jmp v7s_shutdown                          ; pure alias
     global v7_encode                          ; export public wrapper
v7_encode:
     PUB_LOAD_CTX "public API: v7_encode called before v7_initialize_inference" ; guard + rax = ctx, stays live through the shuffle
     test rcx, rcx                             ; text pointer
     jz .v7enc_null                            ; the tokenizer dereferences all three, a NULL is a caller bug
     test rdx, rdx                             ; tokens output array
     jz .v7enc_null                            ; the tokenizer writes through it
     test r8, r8                               ; n_tokens out-pointer
     jz .v7enc_null                            ; the tokenizer writes the count through it
     sub rsp, PUB_F_ENC                        ; outgoing frame: shadow + 1 stack arg + pad
     mov [rsp + PUB_OUT_ARG5], r9d             ; internal arg5: max_tokens, int, 32-bit store matches the signature; both widths encode to the same 5 bytes, REX.W being one bit of the single REX byte
     mov r9, r8                                ; internal arg4: n_tokens, public arg3
     mov r8, rdx                               ; internal arg3: tokens, public arg2
     mov rdx, rcx                              ; internal arg2: text, public arg1, read before rcx overwrite
     mov rcx, [rax + CTX_TOKENIZER]            ; internal arg1: tokenizer from live rax = ctx
     call encode                               ; BPE-encode the text
     add rsp, PUB_F_ENC                        ; restore outgoing frame
     ENGINE_LEAVE                              ; park the landing zone, normal path
     ret                                       ; plain return
.v7enc_null:
     FATAL "public API: v7_encode requires non-NULL text, tokens and n_tokens" ; pointer contract
     global v7_get_internal_inference_ctx      ; export public wrapper
v7_get_internal_inference_ctx:
     mov rax, [rel g_inf_ctx_block]            ; the singleton pointer itself, may be 0
     ret                                       ; plain return
     global v7_get_token_probs                 ; export public wrapper
%define V7GTP_F4 4                             ; sizeof,float, scale for id-indexed logits/probs addressing
v7_get_token_probs:
     PUB_LOAD_CTX "public API: v7_get_token_probs called before v7_initialize_inference" ; gate + guard + rax = ctx
     test edx, edx                             ; n == 0 -> nothing to normalize
     jz .v7gtp_bad_n                           ; contract: 1..256 ids
     cmp edx, 256                              ; cap at the wide-label ceiling
     ja .v7gtp_bad_n                           ; contract: 1..256 ids
     mov r9, [rax + CTX_TRANSFORMER]           ; r9 = Transformer*, ctx header offset CTX_TRANSFORMER = 0
     mov r10d, [r9 + CFG_VOCAB_SIZE]           ; r10d = vocab_size, embedded Config dword field, offset 28
     mov r9, [r9 + TRANSFORMER_STATE + S_LOGITS] ; r9 = logits base, RunState is INLINE EMBEDDED in the Transformer at TRANSFORMER_STATE=184; S_LOGITS=96 is its logits-pointer slot
; --- pass 1: max over logits[id], also verifies every id in range ---
     mov eax, [rcx]                            ; id of element 0
     cmp eax, r10d                             ; unsigned compare: negative ids wrap huge and are caught here
     jae .v7gtp_bad_id                         ; id outside vocab range
     vmovss xmm0, [r9 + rax*V7GTP_F4]          ; max = logits[id0]
     mov r11d, 1                               ; i = 1
     cmp r11d, edx                             ; n >= 2?, 1 == n means element 0 was the only candidate
     jae .v7gtp_exp_pass                       ; single candidate, pass 1 done
.v7gtp_max_loop:
     mov eax, [rcx + r11*V7GTP_F4]             ; id of element i
     cmp eax, r10d                             ; in-vocab check, pass 2 relies on it
     jae .v7gtp_bad_id                         ; id outside vocab range
     vmaxss xmm0, xmm0, [r9 + rax*V7GTP_F4]    ; max = max,max, logits[id]
     inc r11d                                  ; i++
     cmp r11d, edx                             ; i < n?
     jb .v7gtp_max_loop                        ; next candidate max
; --- pass 2: out[i] = exp,logit - max, sum accumulated in xmm1 ---
.v7gtp_exp_pass:
     vxorps xmm1, xmm1, xmm1                   ; sum = 0
     xor r11d, r11d                            ; i = 0
.v7gtp_exp_loop:
     mov eax, [rcx + r11*V7GTP_F4]             ; id of element i, range-verified in pass 1
     vmovss xmm2, [r9 + rax*V7GTP_F4]          ; x = logits[id]
     vsubss xmm2, xmm2, xmm0                   ; x - max
; inline scalar Cody-Waite + Horner exp, instruction-for-instruction
     vmaxss xmm2, xmm2, [rel s_exp_neg_clamp]  ; clamp lower, underflow guard, same bound as the sampler path
     vmulss xmm3, xmm2, [rel scalar_log2e]     ; x * log2,e
     vroundss xmm3, xmm3, xmm3, 0              ; n = round,x/ln2
     vcvttss2si eax, xmm3                      ; eax = n as int32, id no longer needed
     vfnmadd231ss xmm2, xmm3, [rel s_exp_ln2_hi] ; x - n*ln2_hi, single rounding
     vfnmadd231ss xmm2, xmm3, [rel s_exp_ln2_lo] ; f, single rounding
     vmovss xmm4, [rel s_exp_c6]               ; c6
     vfmadd213ss xmm4, xmm2, [rel s_exp_c5]    ; f*c6 + c5
     vfmadd213ss xmm4, xmm2, [rel s_exp_c4]    ; f*prev + c4
     vfmadd213ss xmm4, xmm2, [rel s_exp_c3]    ; f*prev + c3
     vfmadd213ss xmm4, xmm2, [rel rpe_half]    ; f*prev + 0.5, c2
     vfmadd213ss xmm4, xmm2, [rel rpe_one]     ; f*prev + 1.0, c1
     vfmadd213ss xmm4, xmm2, [rel rpe_one]     ; f*prev + 1.0, c0
     add eax, 127                              ; bias the exponent: 127 = float exponent bias
     shl eax, 23                               ; shift into exponent field
     vmovd xmm3, eax                           ; 2^n
     vmulss xmm2, xmm4, xmm3                   ; exp,x - max = poly * 2^n
     vmovss [r8 + r11*V7GTP_F4], xmm2          ; out[i] = exp,x - max
     vaddss xmm1, xmm1, xmm2                   ; sum += exp
     inc r11d                                  ; i++
     cmp r11d, edx                             ; i < n?
     jb .v7gtp_exp_loop                        ; next exp term
; --- NaN guard, then pass 3: out[i] /= sum ---
.v7gtp_nan_check:
     vucomiss xmm1, xmm1                       ; NaN test: x != x iff NaN, sets PF
     jp .v7gtp_nan_sum                         ; NaN -> FATAL
     xor r11d, r11d                            ; i = 0
     mov eax, edx                              ; eax = candidate count n
     and eax, -8                               ; eax = largest multiple of 8 <= n
     jz .v7gtp_div_loop                        ; n < 8: scalar loop handles every element
     vbroadcastss ymm3, xmm1                   ; same scalar denominator in every lane
.v7gtp_div_vec:
     vmovups ymm2, [r8 + r11*V7GTP_F4]         ; out[i..i+7]
     vdivps ymm2, ymm2, ymm3                   ; / sum, 8 correctly rounded quotients, bit-identical to vdivss per lane
     vmovups [r8 + r11*V7GTP_F4], ymm2         ; store 8 probabilities
     add r11d, 8                               ; advance 8 elements
     cmp r11d, eax                             ; vector prefix done?
     jb .v7gtp_div_vec                         ; more full vectors
.v7gtp_div_loop:
     cmp r11d, edx                             ; tail elements left?
     jae .v7gtp_done                           ; no, all n normalized
     vmovss xmm2, [r8 + r11*V7GTP_F4]          ; out[i]
     vdivss xmm2, xmm2, xmm1                   ; / sum, exact IEEE divide, precision-preserving
     vmovss [r8 + r11*V7GTP_F4], xmm2          ; out[i] = probability
     inc r11d                                  ; i++
     jmp .v7gtp_div_loop                       ; next tail probability
.v7gtp_done:
     vzeroupper                                ; clear upper YMM halves, the vector path dirtied them
     ENGINE_LEAVE                              ; park the landing zone, normal path
     ret                                       ; plain return
.v7gtp_bad_n:
     FATAL "public API: v7_get_token_probs requires 1..256 token ids" ; border check, outermost layer
.v7gtp_bad_id:
     FATAL "public API: v7_get_token_probs token id outside vocab range" ; OOB read guard
.v7gtp_nan_sum:
     FATAL "public API: v7_get_token_probs softmax sum is NaN (logits drift)" ; same drift class the sampler FATALs on
; ---- %undef cleanup: keep the preprocessor namespace clean for later includes ----
%unmacro PUB_SAMPLER_CTX 0                     ; sampler-context loader macro, call sites above
%unmacro PUB_LOAD_CTX 1                        ; context-loader macro, call sites above
%unmacro PUB_CPU_GATE 0                        ; last PUB_CPU_GATE call site was above, remove the macro
%unmacro PUB_SARG_TO_OUT 3                     ; incoming-to-outgoing stack-arg shuffle macro, call sites above
%unmacro PUB_RAW_CALL 2                        ; raw-entry gate-and-call wrapper, six call sites above
%unmacro PUB_VALIDATE_IMAGE 3                  ; image-entry border validation macro, two call sites above
%undef PUB_IMG_MAXPX
%undef PUB_IMG_MAXDIM
%undef PUB_RBP_ARG5
%undef PUB_RBP_ARG6
%undef PUB_RBP_ARG7
%undef PUB_RBP_ARG8
%undef PUB_F_CHAT
%undef PUB_F_ENC
%undef PUB_F_LEAF
%undef PUB_F_IMG
%undef PUB_F_GIMG
%undef PUB_OUT_ARG5
%undef PUB_OUT_ARG6
%undef PUB_OUT_ARG7
%undef PUB_OUT_ARG8
%undef PUB_OUT_ARG9
%undef V7GTP_F4
