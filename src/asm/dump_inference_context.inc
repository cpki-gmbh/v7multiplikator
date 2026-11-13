; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%include "inf_structs.inc"
     section .data
     global enable_dump                        ; export flag so tools can toggle the dump, mirrors enable_token_dump
enable_dump:
     dd 0                                      ; runtime flag: 0=disabled, nonzero=enabled

%macro DUMP_PTR 3
     WRITE_STR %1                              ; label string, includes the trailing '='
     mov rbx, [%2 + %3]                        ; load pointer field directly into non-volatile rbx
     WRITE_HEX64 rbx                           ; print as 16-hex-digit value
     WRITE_NL                                  ; newline
%endmacro

%macro DUMP_DWORD 3
     WRITE_STR %1                              ; label string, includes the trailing '='
     mov ebx, [%2 + %3]                        ; load dword directly into rbx, zero-extends
     WRITE_HEX64 rbx                           ; print as 16-hex-digit value
     WRITE_NL                                  ; newline
%endmacro

%macro DUMP_PEEK8 3
     WRITE_STR %1                              ; label string, includes the trailing '='
     mov rax, [%2 + %3]                        ; rax = buffer pointer to peek through
     test rax, rax                             ; NULL check
     jz %%peek_null                            ; NULL -> print the marker instead of data
     mov rbx, [rax]                            ; first 8 bytes into non-volatile rbx
     WRITE_HEX64 rbx                           ; print as 16 hex digits
     jmp %%peek_done                           ; data printed, skip the marker
%%peek_null:
     WRITE_STR dc_null                         ; buffer was NULL, print the marker
%%peek_done:
     WRITE_NL                                  ; newline
%endmacro
     section .text
; ============================================================================
; dump_inference_context

; Non-volatile register usage, all saved by SF_START, restored by SF_END :
; r12 = InferenceContext*
; r13 = Config*, embedded at Transformer+0
; r14 = RunState*
; r15 = Tokenizer*, then Sampler*, re-derived from r12 after the Tokenizer
; section, whose dumps are the last users of r15
; rbx is never a base: it is the value register of every DUMP_*
; rsi/rdi not saved: WRITE_HEX64 and WRITE_STR push/restore them internally
; ============================================================================
     global dump_inference_context             ; entry label, SF_START emits its own frame name
dump_inference_context:
     cmp dword [rel enable_dump], 0            ; check runtime flag BEFORE frame setup
     jnz .dc_enabled                           ; nonzero -> take the dump
     ret                                       ; disabled, normal state -> zero frame overhead, zero taken branches
.dc_enabled:
     SF_START dc_frame, rbx, r12, r13, r14, r15 ; save non-volatiles, set frame pointer, frame name differs
     SF_COMMIT                                 ; allocate stack locals + shadow space
; Cache sub-struct pointers in non-volatile registers
     mov r12, rcx                              ; r12 = InferenceContext*
     mov r13, [rcx + CTX_TRANSFORMER]          ; r13 = Transformer* = Config*, embedded at offset 0
     lea r14, [r13 + TRANSFORMER_STATE]        ; r14 = &RunState, embedded at offset 184
     mov r15, [rcx + CTX_TOKENIZER]            ; r15 = Tokenizer*
     WRITE_STR_NL dc_hdr                       ; print header + newline
     DUMP_PTR dc_lbl_tf_ptr, r12, CTX_TRANSFORMER ; ctx->transformer
     DUMP_PTR dc_lbl_tk_ptr, r12, CTX_TOKENIZER ; ctx->tokenizer
     DUMP_PTR dc_lbl_sp_ptr, r12, CTX_SAMPLER  ; ctx->sampler
     DUMP_PTR dc_lbl_pt_ptr, r12, CTX_PROMPT_TOKENS ; ctx->prompt_tokens
     DUMP_PTR dc_lbl_ot_ptr, r12, CTX_OUT_TOKENS ; ctx->out_tokens
     DUMP_DWORD dc_lbl_maxp, r12, CTX_MAX_PROMPT ; ctx->max_prompt
     DUMP_DWORD dc_lbl_maxg, r12, CTX_MAX_GEN  ; ctx->max_gen
     DUMP_DWORD dc_lbl_pos, r12, CTX_POS       ; ctx->pos
     DUMP_DWORD dc_lbl_qmode, r12, CTX_QUANTIZE ; ctx->quantize, Q8 mode flag
     WRITE_STR_NL dc_sec_cfg                   ; section header + newline
     DUMP_DWORD dc_cf_dim, r13, CFG_DIM        ; config->dim
     DUMP_DWORD dc_cf_hid, r13, CFG_HIDDEN_DIM ; config->hidden_dim
     DUMP_DWORD dc_cf_nl, r13, CFG_N_LAYERS    ; config->n_layers
     DUMP_DWORD dc_cf_nh, r13, CFG_N_HEADS     ; config->n_heads
     DUMP_DWORD dc_cf_nkv, r13, CFG_N_KV_HEADS ; config->n_kv_heads
     DUMP_DWORD dc_cf_vs, r13, CFG_VOCAB_SIZE  ; config->vocab_size
     DUMP_DWORD dc_cf_sl, r13, CFG_SEQ_LEN     ; config->seq_len
     DUMP_DWORD dc_cf_hd, r13, CFG_HEAD_DIM    ; config->head_dim
     DUMP_DWORD dc_cf_qk, r13, CFG_HAS_QK_NORM ; config->has_qk_norm
     DUMP_DWORD dc_cf_rb, r13, CFG_ROPE_BASE   ; config->rope_base, integer theta, e.g. 1000000, float copy is __sys_rope_base
     DUMP_DWORD dc_cf_sc, r13, CFG_SHARED_CLASSIFIER ; config->shared_classifier, classifier == embedding?
     DUMP_DWORD dc_cf_gs, r13, CFG_GROUP_SIZE  ; config->group_size, Q8 quantization GS
     WRITE_STR_NL dc_sec_rs                    ; section header + newline
     DUMP_PTR dc_rs_x, r14, S_X                ; rs->x
     DUMP_PTR dc_rs_xb, r14, S_XB              ; rs->xb
     DUMP_PTR dc_rs_hb, r14, S_HB              ; rs->hb
     DUMP_PTR dc_rs_hb2, r14, S_HB2            ; rs->hb2
     DUMP_PTR dc_rs_xqq, r14, S_XQ_Q           ; rs->xq.q
     DUMP_PTR dc_rs_xqs, r14, S_XQ_S           ; rs->xq.s
     DUMP_PTR dc_rs_hqq, r14, S_HQ_Q           ; rs->hq.q
     DUMP_PTR dc_rs_hqs, r14, S_HQ_S           ; rs->hq.s
     DUMP_PTR dc_rs_q, r14, S_Q                ; rs->q
     DUMP_PTR dc_rs_k, r14, S_K                ; rs->k
     DUMP_PTR dc_rs_v, r14, S_V                ; rs->v
     DUMP_PTR dc_rs_att, r14, S_ATT            ; rs->att
     DUMP_PTR dc_rs_logits, r14, S_LOGITS      ; rs->logits
; KV cache pointers, the state a KV-isolation failure shows up in
     WRITE_STR_NL dc_sec_kv                    ; KV cache section header + newline
     DUMP_PTR dc_rs_kc, r14, S_KEY_CACHE       ; rs->key_cache
     DUMP_PTR dc_rs_vc, r14, S_VALUE_CACHE     ; rs->value_cache
     DUMP_PTR dc_rs_xs, r14, S_X_STORAGE       ; rs->x_storage
     DUMP_PTR dc_rs_qs, r14, S_Q_STORAGE       ; rs->q_storage
     DUMP_PTR dc_rs_ao, r14, S_ATT_OUT         ; rs->att_out
     WRITE_STR_NL dc_sec_kvdata                ; data peek header + newline
     DUMP_PEEK8 dc_kc_dq0, r14, S_KEY_CACHE    ; key_cache[0:8], detect data corruption
     DUMP_PEEK8 dc_vc_dq0, r14, S_VALUE_CACHE  ; value_cache[0:8], detect data corruption
     WRITE_STR_NL dc_sec_tok                   ; tokenizer section header + newline
     DUMP_PTR dc_tk_vocab, r15, TOK_VOCAB      ; tok->vocab ptr array
     DUMP_PTR dc_tk_merge, r15, TOK_MERGE_SCORES ; tok->merge_scores
     DUMP_DWORD dc_tk_vs, r15, TOK_VOCAB_SIZE  ; tok->vocab_size
     DUMP_DWORD dc_tk_mtl, r15, TOK_MAX_TOKEN_LEN ; tok->max_token_length
     DUMP_DWORD dc_tk_bos, r15, TOK_BOS_ID     ; tok->bos_token_id
     DUMP_DWORD dc_tk_eos, r15, TOK_EOS_ID     ; tok->eos_token_id
     DUMP_DWORD dc_tk_sp, r15, TOK_IS_SP       ; tok->is_sentencepiece
     WRITE_STR_NL dc_sec_sam                   ; sampler section header + newline
     mov r15, [r12 + CTX_SAMPLER]              ; r15 = Sampler*, the Tokenizer section is the last user of r15, so re-derive the base here from r12, rbx cannot carry it: every DUMP_* overwrites rbx with the field value
     DUMP_DWORD dc_sp_vs, r15, SAM_VOCAB_SIZE  ; sam->vocab_size
     DUMP_PTR dc_sp_pi, r15, SAM_PROBINDEX     ; sam->probindex
     DUMP_DWORD dc_sp_temp, r15, SAM_TEMPERATURE ; sam->temperature, f32 bits
     DUMP_DWORD dc_sp_top, r15, SAM_TOPP       ; sam->topp, f32 bits
     DUMP_DWORD dc_sp_pp, r15, SAM_PRESENCE_PENALTY ; sam->presence_penalty, f32 bits
     DUMP_DWORD dc_sp_tk, r15, SAM_TOPK        ; sam->topk
     DUMP_PTR dc_sp_rng, r15, SAM_RNG_STATE    ; sam->rng_state, qword
     WRITE_STR_NL dc_end                       ; closing separator + newline
; Return
     SF_END rbx, r12, r13, r14, r15            ; restore non-volatiles and return
     section .data
dc_hdr:
     db ".......... InferenceContext Dump ..........", 0 ; header
dc_end:
     db ".......... End Dump ..........", 0    ; footer
dc_sec_cfg:
     db "--- Config ---", 0                    ; config section
dc_sec_rs:
     db "--- RunState ---", 0                  ; runstate section
dc_sec_kv:
     db "--- KV Cache (critical) ---", 0       ; kv section
dc_sec_kvdata:
     db "--- KV Cache Data Peek ---", 0        ; data peek section
dc_sec_tok:
     db "--- Tokenizer ---", 0                 ; tokenizer section
dc_sec_sam:
     db "--- Sampler ---", 0                   ; sampler section
; InferenceContext fields
dc_lbl_tf_ptr:
     db "  ctx->transformer=", 0               ; transformer ptr
dc_lbl_tk_ptr:
     db "  ctx->tokenizer=", 0                 ; tokenizer ptr
dc_lbl_sp_ptr:
     db "  ctx->sampler=", 0                   ; sampler ptr
dc_lbl_pt_ptr:
     db "  ctx->prompt_tokens=", 0             ; prompt tokens ptr
dc_lbl_ot_ptr:
     db "  ctx->out_tokens=", 0                ; out tokens ptr
dc_lbl_maxp:
     db "  ctx->max_prompt=", 0                ; max prompt
dc_lbl_maxg:
     db "  ctx->max_gen=", 0                   ; max gen
dc_lbl_pos:
     db "  ctx->pos=", 0                       ; position
dc_lbl_qmode:
     db "  ctx->quantize=", 0                  ; quantize mode
; Config fields
dc_cf_dim:
     db "  config->dim=", 0                    ; dim
dc_cf_hid:
     db "  config->hidden_dim=", 0             ; hidden dim
dc_cf_nl:
     db "  config->n_layers=", 0               ; n layers
dc_cf_nh:
     db "  config->n_heads=", 0                ; n heads
dc_cf_nkv:
     db "  config->n_kv_heads=", 0             ; kv heads
dc_cf_vs:
     db "  config->vocab_size=", 0             ; vocab size
dc_cf_sl:
     db "  config->seq_len=", 0                ; seq len
dc_cf_hd:
     db "  config->head_dim=", 0               ; head dim
dc_cf_qk:
     db "  config->has_qk_norm=", 0            ; has qk norm
dc_cf_rb:
     db "  config->rope_base=", 0              ; rope base
dc_cf_sc:
     db "  config->shared_classifier=", 0      ; shared classifier flag
dc_cf_gs:
     db "  config->group_size=", 0             ; quantization group size
; RunState fields
dc_rs_x:
     db "  rs->x=", 0                          ; x buffer
dc_rs_xb:
     db "  rs->xb=", 0                         ; xb buffer
dc_rs_hb:
     db "  rs->hb=", 0                         ; hb buffer
dc_rs_hb2:
     db "  rs->hb2=", 0                        ; hb2 buffer
dc_rs_xqq:
     db "  rs->xq.q=", 0                       ; xq.q buffer
dc_rs_xqs:
     db "  rs->xq.s=", 0                       ; xq.s buffer
dc_rs_hqq:
     db "  rs->hq.q=", 0                       ; hq.q buffer
dc_rs_hqs:
     db "  rs->hq.s=", 0                       ; hq.s buffer
dc_rs_q:
     db "  rs->q=", 0                          ; q buffer
dc_rs_k:
     db "  rs->k=", 0                          ; k buffer
dc_rs_v:
     db "  rs->v=", 0                          ; v buffer
dc_rs_att:
     db "  rs->att=", 0                        ; att buffer
dc_rs_logits:
     db "  rs->logits=", 0                     ; logits buffer
dc_rs_kc:
     db "  rs->key_cache=", 0                  ; key cache
dc_rs_vc:
     db "  rs->value_cache=", 0                ; value cache
dc_rs_xs:
     db "  rs->x_storage=", 0                  ; x storage
dc_rs_qs:
     db "  rs->q_storage=", 0                  ; q storage
dc_rs_ao:
     db "  rs->att_out=", 0                    ; att out
; KV data peek
dc_kc_dq0:
     db "  key_cache[0:8]=", 0                 ; key cache peek
dc_vc_dq0:
     db "  value_cache[0:8]=", 0               ; value cache peek
dc_null:
     db "  (NULL)", 0                          ; null marker
; Tokenizer fields
dc_tk_vocab:
     db "  tok->vocab=", 0                     ; vocab ptr
dc_tk_merge:
     db "  tok->merge_scores=", 0              ; merge scores ptr
dc_tk_vs:
     db "  tok->vocab_size=", 0                ; vocab size
dc_tk_mtl:
     db "  tok->max_token_length=", 0          ; max token len
dc_tk_bos:
     db "  tok->bos_token_id=", 0              ; bos token id
dc_tk_eos:
     db "  tok->eos_token_id=", 0              ; eos token id
dc_tk_sp:
     db "  tok->is_sentencepiece=", 0          ; is sentencepiece
; Sampler fields
dc_sp_vs:
     db "  sam->vocab_size=", 0                ; vocab size
dc_sp_pi:
     db "  sam->probindex=", 0                 ; probindex ptr
dc_sp_temp:
     db "  sam->temperature (f32 bits)=", 0    ; temperature
dc_sp_top:
     db "  sam->topp (f32 bits)=", 0           ; topp
dc_sp_pp:
     db "  sam->presence_penalty (f32 bits)=", 0 ; presence penalty
dc_sp_tk:
     db "  sam->topk=", 0                      ; topk
dc_sp_rng:
     db "  sam->rng_state=", 0                 ; rng state
; ---- %unmacro cleanup: file-local macros leave scope here, house convention ----
%unmacro DUMP_PTR 3                            ; pointer-field dump, label + 16-hex + newline
%unmacro DUMP_DWORD 3                          ; dword-field dump, zero-extended to 64 bits
%unmacro DUMP_PEEK8 3                          ; 8-byte peek through a buffer pointer with NULL marker
     section .text                             ; restore .text, an included file must leave the section it borrowed, house convention
