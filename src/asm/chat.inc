; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; chat.inc, v7s_chat orchestrator. The body lives in fragment files included
; below at their exact emission positions: chat_session -> chat_prompt ->
; chat_prefill -> chat_genloop -> chat_rewrite -> chat_finish. Emission order
; equals execution order; every fragment carries only local labels so the
; non-local label anchor stays v7s_chat, SF_START here.The %undef/%unmacro
; teardown for ALL fragment-owned names is centralized at this file's tail.

; Struct offsets all come from inf_structs.inc's CTX_*/CFG_*/TRANSFORMER_*/TOK_*/SAM_* name
; families, so a struct-layout change lands in exactly one place.
%include "inf_structs.inc"
; Stack argument offsets, 5th/6th args relative to rbp after SF_START
%define CHAT_SARG_OUTPUT_LEN 48                ; int output_len, 5th stack arg
%define CHAT_SARG_RESET 56                     ; int reset, 6th stack arg: !=0 resets KV cache + pos
; Rendered prompt byte buffer size, matches PROMPT_BUFFER_SIZE = 32768
%define CHAT_RENDERED_BUF 32768                ; 32KB prompt render buffer
; Maximum prompt token count, matches prompt_tokens array capacity
%define CHAT_MAX_PROMPT_TOKENS 32768           ; max tokens after encode
; Stack probe page size, Windows 4KB page for guard page committing
%define CHAT_PROBE_PAGE 4096                   ; Windows 4KB guard page size
; Prompt tokens heap allocation size in bytes, 32768 tokens * 4 bytes/int
%define CHAT_PROMPT_TOKENS_BYTES (CHAT_MAX_PROMPT_TOKENS * 4) ; bytes for prompt token array
; Temp frame offsets for stack args, relative to temp RSP after sub
%define CHAT_TEMP_ARG5 32                      ; 5th arg slot, caller shadow + 0
%define CHAT_TEMP_ARG6 40                      ; 6th arg slot, caller shadow + 8
; Temp frame sizes, must be 16-aligned to keep RSP 16-aligned before CALL
%define CHAT_TEMP_FRAME_1 48                   ; shadow, 32 + 1 arg, 8 + align, 8
%define CHAT_TEMP_FRAME_2 48                   ; shadow, 32 + 2 args, 16
; System-prompt ownership: extra byte for the NUL terminator on the owned copy
%define SYSPROMPT_NUL_EXTRA 1                  ; the +1 in the mdb_alloc_mem size, NUL terminator

%macro RDTSC_TO_MEM 1
     lfence                                    ; serialize prior instructions before reading TSC
     rdtsc                                     ; edx:eax = 64-bit timestamp counter
     shl rdx, 32                               ; shift high dword into position
     or rax, rdx                               ; rax = full 64-bit timestamp
     mov [rel %1], rax                         ; store timestamp into statistics variable
%endmacro
; CHAT_SAMPLE_TO_NEXT -- the shared sampling ABI: next = v7s_sample(logits,
; vocab_size, temperature, topp, &rng_state, probindex), result into
; c_next_token. Uses every frame slot below, clobbers volatile regs only.
%macro CHAT_SAMPLE_TO_NEXT 0
     mov rcx, c_logits                         ; argument 1: logits
     mov edx, c_vocab_size                     ; argument 2: vocabulary size
     vmovd xmm2, c_temperature_bits            ; argument 3: temperature
     vmovd xmm3, c_topp_bits                   ; argument 4: top-p
     sub rsp, CHAT_TEMP_FRAME_2                ; Win64 shadow + stack arguments
     mov rax, c_rng_state_ptr                  ; argument 5: RNG pointer
     mov [rsp + CHAT_TEMP_ARG5], rax           ; store RNG pointer
     mov rax, c_probindex_ptr                  ; argument 6: probindex pointer
     mov [rsp + CHAT_TEMP_ARG6], rax           ; store probindex pointer
     call v7s_sample                           ; eax = sampled token
     add rsp, CHAT_TEMP_FRAME_2                ; restore aligned stack
     mov c_next_token, eax                     ; publish the next token
%endmacro
; Decode/penalty macros, constants, and shared penalty leaf
%include "chat_decode.inc"
     section .text
     global v7s_get_generation_statistic       ; export statistics pointer getter
v7s_get_generation_statistic:
     lea rax, [rel rdtsc_chat_start]           ; address of first statistics variable
     ret                                       ; return pointer in rax
     global v7s_get_user_prompt                ; export for C-CALL access
v7s_get_user_prompt:
     mov rax, [rel g_user_prompt]              ; char* published by v7s_chat
     ret                                       ; return pointer, or NULL in rax
; SELECT_TPL + v7s_enable_thinking, emitted here: between the getters and v7s_chat
%include "chat_template.inc"
     global v7s_chat                           ; export v7s_chat entry point
v7s_chat:
     SF_START v7s_chat, rbx, rsi, rdi, r12, r13, r14, r15 ; begin v7s_chat stack frame
     SF_PTR c_ctx                              ; InferenceContext* ctx, rcx
     SF_PTR c_user_prompt                      ; char* user_prompt, rdx
     SF_PTR c_system_prompt                    ; char* system_prompt, r8
     SF_PTR c_output                           ; char* output, r9
     SF_DWORD c_output_len                     ; int output_len, rbp+48
     SF_DWORD c_reset                          ; int reset, rbp+56, !=0 -> ctx->pos := 0
     SF_DWORD c_num_prompt_tokens              ; int n_tokens from encode
     SF_PTR c_prompt_tokens                    ; int* heap buffer for token IDs
     SF_DWORD c_pos                            ; current sequence position
     SF_DWORD c_out_pos                        ; current write offset into output
     SF_DWORD c_seq_len                        ; config->seq_len, KV-cache capacity
     SF_DWORD c_max_gen                        ; max generated tokens, ctx->max_gen
     SF_DWORD c_eos_token_id                   ; tokenizer->eos_token_id
     SF_DWORD c_stop_count                     ; stop-ID count, chat-invariant, cached for per-token scan
     SF_DWORD c_raw_mode                       ; raw-generation flag cached at entry, 1 = no template, EOS-only stop
     SF_DWORD c_quantize                       ; ctx->quantize cached at entry, 0=F32, 1=Q8; loop-invariant
     SF_DWORD c_vocab_size                     ; sampler->vocab_size
     SF_DWORD c_temperature_bits               ; sampler->temperature as float bits
     SF_DWORD c_topp_bits                      ; sampler->topp as float bits
     SF_QWORD c_rng_state_ptr                  ; &sampler->rng_state
     SF_QWORD c_probindex_ptr                  ; sampler->probindex
     SF_PTR c_logits                           ; float* logits from forward/prefill
     SF_PTR c_out_tokens                       ; int* ctx->out_tokens, token history fill for V7-GET-TOKEN
     SF_DWORD c_next_token                     ; sampled or prompt token for next iter
     SF_QWORD c_bitmap_base                    ; bitmap base pointer, stack
     SF_DWORD c_presence_penalty_bits          ; presence_penalty float bits
     SF_PTR c_transformer                      ; ctx->transformer cached at entry, loop-invariant; arg1 of every forward/prefill call
     SF_DWORD c_rw_pos_old                     ; tier 2: pre-rewrite cursor, tracker-store math
     SF_DWORD c_rw_tier2                       ; tier 2 flag: 1 while a wrapped-session rewrite is unwinding
     SF_DWORD c_rw_latch                       ; rewrite: saved g_stream_remove_warm, BOTH tiers mute at entry, exact restore at the sample/finish
     SF_DWORD c_rw_muted                       ; rewrite: 1 while the warm latch is parked in c_rw_latch, the finish tests this, tier-1 mid-suffix stops bypass the tier-2 flag
     SF_DWORD c_rw_valid_store                 ; tier 2: tracker value to publish at the sample, W, or the shaved clamp.SEMANTICS: counts rows strictly BELOW the cursor, the cursor's own slot is attended in addition, the wrapped forward writes + attends it ; do not "fix" a perceived off-by-one here
     SF_DWORD c_rw_end                         ; tier 2: the PLANNED replacement end, A + n, the finish unwind derives the true valid count from the ACTUAL stop cursor against it
     SF_COMMIT                                 ; commit stack frame slot names
; --- body fragments, emission order = execution order ---
%include "chat_session.inc"
%include "chat_prompt.inc"
%include "chat_prefill.inc"
%include "chat_genloop.inc"
%include "chat_rewrite.inc"
%include "chat_finish.inc"
%unmacro CHAT_SAMPLE_TO_NEXT 0                 ; all four call sites expanded, file-local macro leaves scope
; ============================================================================
; Clean up SF_* slot names
%undef c_ctx
%undef c_ctx_raw
%undef c_user_prompt
%undef c_user_prompt_raw
%undef c_system_prompt
%undef c_system_prompt_raw
%undef c_output
%undef c_output_raw
%undef c_output_len
%undef c_output_len_raw
%undef c_reset
%undef c_reset_raw
%undef c_num_prompt_tokens
%undef c_num_prompt_tokens_raw
%undef c_prompt_tokens
%undef c_prompt_tokens_raw
%undef c_pos
%undef c_pos_raw
%undef c_out_pos
%undef c_out_pos_raw
%undef c_eos_token_id
%undef c_eos_token_id_raw
%undef c_vocab_size
%undef c_vocab_size_raw
%undef c_temperature_bits
%undef c_temperature_bits_raw
%undef c_topp_bits
%undef c_topp_bits_raw
%undef c_rng_state_ptr
%undef c_rng_state_ptr_raw
%undef c_probindex_ptr
%undef c_probindex_ptr_raw
%undef c_logits
%undef c_logits_raw
%undef c_next_token
%undef c_next_token_raw
%undef c_seq_len
%undef c_seq_len_raw
%undef c_max_gen
%undef c_max_gen_raw
%undef c_bitmap_base
%undef c_bitmap_base_raw
%undef c_presence_penalty_bits
%undef c_presence_penalty_bits_raw
%undef c_quantize
%undef c_quantize_raw
%undef c_transformer
%undef c_transformer_raw
%undef c_out_tokens
%undef c_out_tokens_raw
%undef c_stop_count
%undef c_stop_count_raw
%undef c_raw_mode
%undef c_raw_mode_raw
%undef c_rw_pos_old                            ; tier-2 fill-in frame slot, typed alias
%undef c_rw_pos_old_raw                        ; untyped alias generated by SF_DWORD
%undef c_rw_tier2                              ; tier-2 fill-in frame slot, typed alias
%undef c_rw_tier2_raw                          ; untyped alias generated by SF_DWORD
%undef c_rw_latch                              ; rewrite frame slot, typed alias
%undef c_rw_latch_raw                          ; untyped alias generated by SF_DWORD
%undef c_rw_muted                              ; rewrite frame slot, typed alias
%undef c_rw_muted_raw                          ; untyped alias generated by SF_DWORD
%undef c_rw_valid_store                        ; tier-2 fill-in frame slot, typed alias
%undef c_rw_valid_store_raw                    ; untyped alias generated by SF_DWORD
%undef c_rw_end                                ; tier-2 fill-in frame slot, typed alias
%undef c_rw_end_raw                            ; untyped alias generated by SF_DWORD
; the inf_structs CTX_*/CFG_*/SAM_*/TOK_* offset names are shared across files, so they are NOT undef'd here; file-local names below are
%undef CHAT_SARG_OUTPUT_LEN
%undef CHAT_SARG_RESET
%undef CHAT_RENDERED_BUF
%undef CHAT_MAX_PROMPT_TOKENS
%undef CHAT_PROMPT_TOKENS_BYTES
%undef CHAT_TEMP_ARG5
%undef CHAT_TEMP_ARG6
%undef CHAT_TEMP_FRAME_1
%undef CHAT_TEMP_FRAME_2
%undef CHAT_PROBE_PAGE
%undef BITMAP_DWORDS
%undef BITMAP_BYTES
%undef SL_REWRITE_MAX_TOKENS                   ; release: file-local, no expansion outside this file
%undef RW_DWORD_SCALE                          ; release: file-local, no expansion outside this file
%undef CH_HEX_PATTERN_LEN                      ; release: file-local, no expansion outside this file
%undef CH_HEX_OFF_GT                           ; release: file-local, no expansion outside this file
%undef CH_HEX_OFF_LO                           ; release: file-local, no expansion outside this file
%undef CH_HEX_OFF_HI                           ; release: file-local, no expansion outside this file
%undef CH_HEX_OFF_0                            ; release: file-local, no expansion outside this file
%undef SP_BYTE_MID                             ; release: file-local, no expansion outside this file
%undef SP_BOUNDARY_SIZE                        ; release: file-local, no expansion outside this file
%undef VOCAB_PTR_SCALE
%unmacro CHAT_DECODE_LEN 0                     ; release: file-local, no expansion outside this file
%unmacro CHAT_DECODE_PIECE 0                   ; release: file-local, no expansion outside this file
%unmacro APPLY_PRESENCE_PENALTY 1              ; release: file-local, no expansion outside this file
