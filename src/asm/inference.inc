; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%include "inf_structs.inc"
%include "inf_data.inc"
     section .text
     global v7s_inference_create               ; Export inference_create entry point
v7s_inference_create:
     SF_START v7s_inference_create, rbx, rsi, rdi, r12, r13, r14, r15 ; Prologue, save non-volatiles
     SF_PTR inf_cfg_ptr                        ; Config* from microdb
     SF_PTR inf_mw_ptr                         ; model_weights* from microdb
     SF_PTR inf_wm_ptr                         ; weight_meta* from microdb
     SF_PTR inf_ml_ptr                         ; MemLayout* from microdb
     SF_PTR inf_transformer                    ; allocated Transformer struct
     SF_PTR inf_rs_block                       ; RunState working-buffer block, KV caches excluded
     SF_PTR inf_key_cache                      ; key cache block, mdb_alloc_mem, Step 10
     SF_PTR inf_value_cache                    ; value cache block, mdb_alloc_mem, Step 10
     SF_DWORD inf_vocab_size                   ; config values, cached, n_layers, dim, etc.
     SF_DWORD inf_n_layers                     ; n_layers from config
     SF_DWORD inf_dim                          ; dim from config
     SF_DWORD inf_hidden_dim                   ; hidden_dim from config
     SF_DWORD inf_head_dim                     ; head_dim from config
     SF_DWORD inf_n_heads                      ; n_heads from config
     SF_DWORD inf_n_kv_heads                   ; n_kv_heads from config
     SF_DWORD inf_seq_len                      ; seq_len from config
     SF_DWORD inf_group_size                   ; group_size from config
     SF_PTR inf_qt_desc                        ; QuantizedTensor descriptor block
     SF_PTR inf_qkv_desc                       ; fused QKV weight descriptor block
     SF_PTR inf_qkv_data                       ; fused QKV data buffer, wq|wk|wv + scales
     SF_DWORD inf_shared_cls                   ; shared_classifier flag
     SF_PTR inf_ctx_block                      ; InferenceContext block, mdb_alloc_mem, Step 20
     SF_PTR inf_tokenizer                      ; ptr to Tokenizer within ctx block
     SF_PTR inf_sampler                        ; ptr to Sampler within ctx block
     SF_DWORD inf_vocab_bulk_count             ; entry count from vocab_bulk
     SF_QWORD inf_str_bytes                    ; total flat vocab-string byte count, from the vocab_bulk scan
     SF_DWORD inf_hdr_max_tok_len              ; from tokenizer_hdr
     SF_DWORD inf_hdr_bos_id                   ; from tokenizer_hdr
     SF_DWORD inf_hdr_eos_id                   ; from tokenizer_hdr
     SF_QWORD inf_db_len                       ; byte length of the most recent ht_find result, scratch, reused per lookup
     SF_QWORD inf_vocab_bulk_ptr               ; vocab_bulk blob base, captured in pass 1, the DB is read-only-locked, so pass 2 cannot get a different pointer
     SF_QWORD inf_vocab_bulk_end               ; vocab_bulk blob end, captured in pass 1, inf_db_len is overwritten by the merge_scores lookup, so the length must be carried forward
     SF_QWORD inf_vit_push_cos                 ; ViT scratch carve: aligned cos/sin size
     SF_COMMIT                                 ; Commit stack frame
     call load_sampler_percentages             ; Reload two adaptive greedy split fractions, default 128 on missing/short file
     call load_stoch_percentages               ; Read independent stochastic splits or initialize to 128
%include "inf_microdb.inc"
%include "inf_weights.inc"
%include "inf_state.inc"
%include "inf_tokenizer.inc"
%include "inf_cleanup.inc"
%define V7_BACKSTEP_OFF 1                      ; [rsi + ecx - N] is source byte ecx-1 in the length scan
%define V7_DBFILENAME_LAST 511                 ; last writable index of g_db_filename, resb 512 AND the longest storable string, the NUL occupies that same index
     global V7_init                            ; export V7_init for DLL callers
V7_init:
     SF_START V7_init, rsi, rdi                ; save rsi/rdi, movsb uses both ; rbx unused, eax carries the result to .v7i_done directly
     SF_COMMIT                                 ; finalize stack frame layout
; Save filename to g_db_filename
     test rcx, rcx                             ; check if filename is NULL
     jz .empty_db                              ; NULL -> skip copy, use empty DB
     lea rdi, [rel g_db_filename]              ; dst = BSS buffer for filename
     mov rsi, rcx                              ; src = caller-provided filename
     mov ecx, 1                                ; ecx is 1-based: byte i-1 lives at [rsi + rcx - V7_BACKSTEP_OFF], 64-bit rcx because rep movsb below consumes the full 64-bit count; a 32-bit index register encodes here too, zero-extended
.v7i_len:
     cmp byte [rsi + rcx - V7_BACKSTEP_OFF], 0 ; end of string?
     jz .v7i_len_done                          ; yes, length measured
     cmp ecx, V7_DBFILENAME_LAST               ; strlen > 511, i.e. longer than the buffer can hold?
     ja .v7i_too_long                          ; yes, reject BEFORE any write
     inc ecx                                   ; length++
     jmp .v7i_len                              ; keep measuring
.v7i_too_long:
     FATAL "V7_init: database filename exceeds 511 chars"
.v7i_len_done:
     rep movsb                                 ; copy the DB filename into g_db_filename, rcx = measured length
     call ht_destroy                           ; drop the previously loaded table
     call ht_init                              ; fresh empty table
     xor ecx, ecx                              ; ecx = flag 0, readwrite
     call db_set_readonly                      ; clear the lock from the previous session
; db_load,g_db_filename
     lea rcx, [rel g_db_filename]              ; rcx = filename for db_load
     call db_load                              ; load entries from file
     test eax, eax                             ; db_load nonzero = success, eax is db_load's result: the reset call below clobbers it, so the test runs first
     jz .v7i_no_db                             ; zero -> no entries were loaded, FATAL below
     call cycle_tuner_init                     ; read the tuner DB switches, reset the per-shape state, eax/ecx clobbered, success already latched by the branch above
     mov eax, 1                                ; success return, same value db_load handed back, the call clobbered rax
     jmp .v7i_done                             ; loaded success
.v7i_no_db:
     FATAL "V7_init: database file does not exist, no entries were loaded"
.empty_db:
     mov byte [rel g_db_filename], 0           ; ensure g_db_filename is empty
     call ht_destroy                           ; drop any previously loaded table
     call ht_init                              ; fresh empty table
     call cycle_tuner_init                     ; the cycle tuner resets with the empty engine too
     xor ecx, ecx                              ; ecx = flag 0, readwrite
     call db_set_readonly                      ; clear any readonly set by the previous loaded session
     mov eax, 1                                ; success, empty DB is valid
.v7i_done:
     SF_END rsi, rdi                           ; tear down stack frame, restore rsi/rdi
; ============================================================================
; Clean up SF_* slot names from v7s_inference_create, both forms: NAME and NAME_raw
; ============================================================================
%undef inf_cfg_ptr
%undef inf_cfg_ptr_raw
%undef inf_mw_ptr
%undef inf_mw_ptr_raw
%undef inf_wm_ptr
%undef inf_wm_ptr_raw
%undef inf_ml_ptr
%undef inf_ml_ptr_raw
%undef inf_transformer
%undef inf_transformer_raw
%undef inf_rs_block
%undef inf_rs_block_raw
%undef inf_key_cache
%undef inf_key_cache_raw
%undef inf_value_cache
%undef inf_value_cache_raw
%undef inf_vocab_size
%undef inf_vocab_size_raw
%undef inf_n_layers
%undef inf_n_layers_raw
%undef inf_dim
%undef inf_dim_raw
%undef inf_hidden_dim
%undef inf_hidden_dim_raw
%undef inf_head_dim
%undef inf_head_dim_raw
%undef inf_n_heads
%undef inf_n_heads_raw
%undef inf_n_kv_heads
%undef inf_n_kv_heads_raw
%undef inf_seq_len
%undef inf_seq_len_raw
%undef inf_group_size
%undef inf_group_size_raw
%undef inf_qt_desc
%undef inf_qt_desc_raw
%undef inf_qkv_desc
%undef inf_qkv_desc_raw
%undef inf_qkv_data
%undef inf_qkv_data_raw
%undef inf_shared_cls
%undef inf_shared_cls_raw
%undef inf_ctx_block
%undef inf_ctx_block_raw
%undef inf_tokenizer
%undef inf_tokenizer_raw
%undef inf_sampler
%undef inf_sampler_raw
%undef inf_vocab_bulk_count
%undef inf_vocab_bulk_count_raw
%undef inf_str_bytes
%undef inf_str_bytes_raw
%undef inf_hdr_max_tok_len
%undef inf_hdr_max_tok_len_raw
%undef inf_hdr_bos_id
%undef inf_hdr_bos_id_raw
%undef inf_hdr_eos_id
%undef inf_hdr_eos_id_raw
%undef inf_db_len
%undef inf_db_len_raw
%undef inf_vocab_bulk_ptr
%undef inf_vocab_bulk_ptr_raw
%undef inf_vocab_bulk_end
%undef inf_vocab_bulk_end_raw
%undef inf_vit_push_cos
%undef inf_vit_push_cos_raw
; ---- %undef cleanup: file-local constants ----
%undef V7_BACKSTEP_OFF
%undef V7_DBFILENAME_LAST
