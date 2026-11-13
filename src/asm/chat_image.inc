; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; Named offset constants, checker requires named %define for all stack/array offsets
%define CIMG_SARG5 48                          ; 5th stack arg, output_len / ds2 at rbp+48 after SF_START
%define CIMG_TEMP_ARG5 32                      ; caller temp-frame slot for callee's 5th arg
%define CIMG_TEMP_ARG6 40                      ; caller temp-frame slot for callee's 6th arg
%define CIMG_TEMP_FRAME 48                     ; temp frame size: 32 shadow + 16 stack args, 16-aligned
; Constants for v7s_chat_image_mem, 9-arg ABI: 4 regs + 5 stack args
%define CIM_SARG5 48                           ; 5th stack arg, img_w at rbp+48 after SF_START
%define CIM_SARG6 56                           ; 6th stack arg, img_h at rbp+56 after SF_START
%define CIM_SARG7 64                           ; 7th stack arg, output at rbp+64 after SF_START
%define CIM_SARG8 72                           ; 8th stack arg, output_len at rbp+72 after SF_START
%define CIM_SARG9 80                           ; 9th stack arg, reset at rbp+80 after SF_START
%define CIM_CALL8_FRAME 64                     ; 8-arg ViT call frame: 32 shadow + 4 stack args * 8 = 64
%define CIM_CALL_ARG5 32                       ; byte offset of 5th arg from rsp in 8-arg call frame
%define CIM_CALL_ARG6 40                       ; byte offset of 6th arg from rsp in 8-arg call frame
%define CIM_CALL_ARG7 48                       ; byte offset of 7th arg from rsp in 8-arg call frame
%define CIM_CALL_ARG8 56                       ; byte offset of 8th arg from rsp in 8-arg call frame
; Vision prompt buffer constants, per-wrapper 256KB STACK allocation, see rationale below
%define CIMG_PROMPT_BUF_BYTES 262144           ; size of the vision marker prompt buffer, 256KB
%define CIMG_PROBE_PAGE 4096                   ; stack check-page probe stride, one 4KB page
%if (CIMG_PROMPT_BUF_BYTES <= 0) || ((CIMG_PROMPT_BUF_BYTES / CIMG_PROBE_PAGE) * CIMG_PROBE_PAGE != CIMG_PROMPT_BUF_BYTES)
%error CIMG_PROMPT_BUF_BYTES must be a positive multiple of CIMG_PROBE_PAGE (the .probe_loop stride) ; assembly-time check: catches the silent write-below-RSP mistake at build time
%endif
%define CIMG_MARKER_LIMIT 245760               ; max marker bytes, 256KB - 16KB text reserve; EMIT check
%define CIMG_VEC_MARKER_MAX 16                 ; max bytes per one <0xNN>-style marker escape, longest form

%macro INVALIDATE_VISION_STATE 0
     cmp dword [rel g_vision_emb_count], 0     ; vision pool installed?
     je %%no_pool                              ; no, nothing was allocated, no frees
     mov rcx, [rel g_vision_embeddings]        ; old out_main, non-NULL by the pool invariant
     call mdb_free_mem                         ; free previous image embedding buffer
     cmp dword [rel g_cfg_has_deepstack], 0    ; deepstack buffers present?
     je %%no_ds                                ; no, ds slots were never filled
     mov rcx, [rel g_deepstack_features + VDS_SLOT0] ; old ds0, non-NULL by contract
     call mdb_free_mem                         ; free previous ds0 buffer
     mov rcx, [rel g_deepstack_features + VDS_SLOT1] ; old ds1, non-NULL by contract
     call mdb_free_mem                         ; free previous ds1 buffer
     mov rcx, [rel g_deepstack_features + VDS_SLOT2] ; old ds2, non-NULL by contract
     call mdb_free_mem                         ; free previous ds2 buffer
%%no_ds:
%%no_pool:
     xor eax, eax                              ; zero eax for the invalidation writes
     mov [rel g_vision_embeddings], rax        ; NULL the vision embedding pool pointer
     mov [rel g_vision_emb_count], eax         ; zero the pending vision token count
     mov [rel g_vision_emb_consumed], eax      ; zero the consumed vision token counter
     mov [rel g_deepstack_features + VDS_SLOT0], rax ; NULL DeepStack feature slot 0
     mov [rel g_deepstack_features + VDS_SLOT1], rax ; NULL DeepStack feature slot 1
     mov [rel g_deepstack_features + VDS_SLOT2], rax ; NULL DeepStack feature slot 2
     mov [rel g_cfg_has_deepstack], eax        ; clear the DeepStack feature-injection flag
     mov [rel g_image_token_id], eax           ; deactivate the vision override, pool is gone, flag must not stay set
%endmacro

%macro PROBE_VISION_PROMPT_BUF 1
     mov r14, rsp                              ; temporary top bound for the probe loop
     sub rsp, CIMG_PROMPT_BUF_BYTES            ; reserve 256KB for the marker prompt buffer
     lea rcx, [r14 - CIMG_PROBE_PAGE]          ; start probing at highest new page, below old RSP
%%probe_loop:
     mov dword [rcx], 0                        ; write-touch this page to commit it
     sub rcx, CIMG_PROBE_PAGE                  ; step down to the next lower 4KB page
     cmp rcx, rsp                              ; reached or passed the new RSP?
     jae %%probe_loop                          ; keep probing while still above new RSP, invariant: see the guard above
     mov %1, rsp                               ; cache buffer base, lowest address in frame slot
%endmacro

%macro EMIT_IMAGE_PADS 1
     mov edx, %1                               ; edx = N, use EDX: rep movsb overwrites ECX; >= 1 by construction, a successful encode always emits pads
     mov r9d, [rel g_vis_prompt_marker_len]    ; r9d = marker body length, loaded once for limit check + loop
     mov rax, r9                               ; rax = marker length, 64-bit operand for the budget imul below
     imul rax, rdx                             ; total bytes the N markers would occupy, 64-bit product, cannot wrap
     cmp rax, CIMG_MARKER_LIMIT                ; beyond the buffer's safe marker budget?
     jae %%limit_fatal                         ; Reject images whose temporary buffer would exceed the available stack allocation.
     mov r8, [rel g_vis_prompt_marker]         ; r8 = marker string pointer, loop-invariant, hoisted out of pad_loop
%%pad_first:                                   ; within budget, straight-line fall-through into the first marker copy
     mov rsi, r8                               ; rsi = marker string, from hoisted register
     mov ecx, r9d                              ; ecx = marker body length, from hoisted register
     rep movsb                                 ; copy the FIRST marker body exactly, rdi advances by body length
     dec edx                                   ; decrement remaining pad count
     jz %%pads_done                            ; single marker: done, nothing to broadcast
     cmp r9d, CIMG_VEC_MARKER_MAX              ; exact 16-byte marker body?, the splat's store width must equal the add-rdi stride, or the bytes between consecutive stores are left unwritten
     jne %%pad_rep_loop                        ; no: exact rep movsb copies, correct for any length
     vmovdqu xmm0, [r8]                        ; xmm0 = first 16 marker bytes, one load; source is stored in the committed DB pool
%%pad_vec_loop:
     vmovdqu [rdi], xmm0                       ; splat 16 marker bytes at the next pad slot, only reached when r9 == 16, so store width == stride and no gap is left unwritten; 16KB reserve above the budget keeps the last store in-page
     add rdi, r9                               ; rdi advances by the 64-bit marker body length
     dec edx                                   ; decrement remaining pad count
     jnz %%pad_vec_loop                        ; emit next marker, do-while, no rep-movsb startup per marker
     jmp %%pads_done                           ; all markers emitted via the vector path
%%pad_rep_loop:
     mov rsi, r8                               ; rsi = marker string, from hoisted register
     mov ecx, r9d                              ; ecx = marker body length, from hoisted register
     rep movsb                                 ; copy one marker body, rdi advances by body length
     dec edx                                   ; decrement remaining pad count
     jnz %%pad_rep_loop                        ; emit next marker, do-while, no top test
%%pads_done:
     jmp %%pads_end                            ; skip the overflow FATAL on the normal path
%%limit_fatal:
     FATAL "vision prompt overflow: image yields more vision tokens than the 256KB marker buffer holds (downscale the image)" ; outermost-API boundary check, user-reachable input size
%%pads_end:
%endmacro

%macro VISION_ENCODE_AND_BUILD 2
; v7s_vit_encode_callback,raw_rgb, img_w, img_h, &emb, &cnt, &ds0, &ds1, &ds2
; 8-arg call: 4 in regs, rcx,rdx,r8,r9, 4 on stack
     sub rsp, CIM_CALL8_FRAME                  ; allocate 8-arg call frame, 32 shadow + 4 stack args
; Vision phase timestamp: before ViT encode, rax/rdx unused, args not loaded yet
     RDTSC_TO_MEM rdtsc_vit_encode_start       ; statistics: TSC before vision encoder call
     mov rcx, %1raw                            ; arg1: raw_rgb pointer, caller-owned
     mov edx, %1img_w                          ; arg2: image width
     mov r8d, %1img_h                          ; arg3: image height
     lea r9, %1emb_ptr                         ; arg4: &emb_ptr
     lea rax, %1emb_cnt                        ; load &emb_cnt
     mov [rsp + CIM_CALL_ARG5], rax            ; arg5: &emb_cnt
     lea rax, %1ds0_ptr                        ; load &ds0_ptr
     mov [rsp + CIM_CALL_ARG6], rax            ; arg6: &ds0_ptr
     lea rax, %1ds1_ptr                        ; load &ds1_ptr
     mov [rsp + CIM_CALL_ARG7], rax            ; arg7: &ds1_ptr
     lea rax, %1ds2_ptr                        ; load &ds2_ptr
     mov [rsp + CIM_CALL_ARG8], rax            ; arg8: &ds2_ptr
     call v7s_vit_encode_callback              ; run slim ViT encoder, compute-only
     add rsp, CIM_CALL8_FRAME                  ; restore stack after 8-arg call
     RDTSC_TO_MEM rdtsc_vit_encode_end         ; statistics: TSC after vision encoder call
; If encoder succeeded, set vision embeddings in engine
     TRACE_STR %2                              ; diagnostic: about to set globals
     mov rcx, %1emb_ptr                        ; arg1: embeddings pointer
     mov edx, %1emb_cnt                        ; arg2: embedding count
     mov r8, %1ds0_ptr                         ; arg3: ds0 pointer
     mov r9, %1ds1_ptr                         ; arg4: ds1 pointer
     sub rsp, CIMG_TEMP_FRAME                  ; allocate shadow space + 5th arg slot
     mov r10, %1ds2_ptr                        ; load ds2 pointer
     mov [rsp + CIMG_TEMP_ARG5], r10           ; arg5: ds2 pointer, caller rsp+32 -> callee rbp+48
     call v7s_set_vision_embeddings            ; set global vision embeddings and DeepStack features
     add rsp, CIMG_TEMP_FRAME                  ; restore stack pointer after function call
     mov rdi, %1vision_prompt_base             ; destination cursor for marker concatenation
     mov rsi, [rel g_vis_prompt_prefix]        ; source: prefix string, may be empty for LLaVA
     copy_vision_token                         ; append prefix body, rdi left past its NUL
     EMIT_IMAGE_PADS %1emb_cnt                 ; append N marker bodies, data-driven: <|image_pad|> or <image>
     mov rsi, [rel g_vis_prompt_suffix]        ; source: suffix string, "<|vision_end|>" or "\n"
     copy_vision_token                         ; append suffix body, rdi past its NUL
     mov r8, rdi                               ; r8 = build cursor after suffix, saved first, scasb overwrites rdi
     mov rdi, %1prompt                         ; rdi = scan cursor, repne scasb reads [rdi]
     xor eax, eax                              ; al = 0, search for the NUL byte
     or rcx, -1                                ; rcx = all-ones scan bound, 3-byte imm8 form
     repne scasb                               ; advance rdi to the NUL, rcx counts down
     not rcx                                   ; rcx = strlen + 1, bytes scanned incl. NUL
     dec rcx                                   ; rcx = prompt byte length
     mov r10, rcx                              ; save strlen in r10, volatile, unused after arg staging; preserved across the bound check
     lea rax, [r8 + rcx]                       ; rax = cursor + prompt_len, the NUL's target address
     mov rdi, r8                               ; restore build cursor, final append writes via rdi
     mov rcx, %1vision_prompt_base             ; buffer base
     add rcx, CIMG_PROMPT_BUF_BYTES - 1        ; rcx = last writable byte, the NUL may sit on it
     mov rsi, %1prompt                         ; source: raw user prompt string
     cmp rax, rcx                              ; would the NUL pass the last writable byte?
     jbe %%copy_full                           ; no: the full user prompt fits
     sub rcx, r8                               ; yes: chat-border correction, rcx = writable bytes incl. the NUL slot
     jnc %%copy                                ; at least one body byte fits: copy the truncated prefix
     xor ecx, ecx                              ; cursor at the last byte: zero-length user text
     jmp %%copy                                ; zero-room case joins the copy tail
%%copy_full:
     mov rcx, r10                              ; rcx = full prompt body length, saved above
%%copy:
     rep movsb                                 ; bulk copy prompt body, ERMSB ~16 bytes/cycle
     mov byte [rdi], 0                         ; single terminator, at most the last writable byte
%endmacro

%macro copy_vision_token 0
     mov r8, rdi                               ; r8 = destination cursor, scasb needs rdi as its scan register
     mov rdi, rsi                              ; rdi = source string, probe length
     xor eax, eax                              ; al = 0, scasb compares AL against [rdi]
     or rcx, -1                                ; all-ones scan bound, a missing terminator runs rcx to 0 and stops
     repne scasb                               ; rdi -> one past the NUL; rcx = 2^64-1-N
     not rcx                                   ; rcx = N = bytes consumed incl. the NUL
     mov rsi, rdi                              ; source cursor currently one past the NUL
     sub rsi, rcx                              ; rewind rsi to the body start, N includes the NUL, so this lands exactly on it
     dec rcx                                   ; rcx = marker body length, NUL excluded, AFTER the rewind: count and rewind need different values
     mov rdi, r8                               ; restore the destination cursor
     rep movsb                                 ; ERMS copy of the body only; rdi ends on the NUL slot
%endmacro
     global v7s_set_vision_embeddings          ; export v7s_set_vision_embeddings entry point
v7s_set_vision_embeddings:
     SF_START v7s_set_vision_embeddings        ; begin stack frame, no non-volatile regs to save
     SF_COMMIT                                 ; commit shadow space for Win64 ABI
     mov [rel g_vision_embeddings], rcx        ; store main embedding base pointer, arg1
     mov [rel g_vision_emb_count], edx         ; store count of vision tokens, arg2
     mov dword [rel g_vision_emb_consumed], 0  ; reset consumed vision token counter to 0
     mov [rel g_deepstack_features + VDS_SLOT0], r8 ; store ds0 pointer, arg3 in slot 0
     mov [rel g_deepstack_features + VDS_SLOT1], r9 ; store ds1 pointer, arg4 in slot 1
     mov rax, [rbp + CIMG_SARG5]               ; load ds2 pointer, arg5, 5th stack arg at rbp+48
     mov [rel g_deepstack_features + VDS_SLOT2], rax ; store ds2 pointer in slot 2
     xor ecx, ecx                              ; ecx = non-NULL DeepStack pointer count, 0..3
     neg r8                                    ; CF = 1 iff ds0 != 0, NEG sets CF on a non-zero operand
     adc ecx, 0                                ; ecx += CF, r8 is unused: slot 0 already published
     neg r9                                    ; CF = 1 iff ds1 != 0
     adc ecx, 0                                ; ecx += CF, r9 is unused: slot 1 already published
     neg rax                                   ; CF = 1 iff ds2 != 0
     adc ecx, 0                                ; ecx += CF, rax is unused: slot 2 already published
     test ecx, ecx                             ; all three NULL?
     je .ds_off                                ; yes: DeepStack disabled
     cmp ecx, 3                                ; all three present?
     je .ds_on                                 ; yes: DeepStack enabled
     FATAL "v7s_set_vision_embeddings: partial DeepStack set (pass all three pointers or none)" ; mixed combination, caller bug
.ds_on:
     mov eax, 1                                ; has_deepstack = 1
     jmp .ds_store                             ; publish the flag
.ds_off:
     xor eax, eax                              ; has_deepstack = 0
.ds_store:
     mov [rel g_cfg_has_deepstack], eax        ; store validated has_deepstack flag
     mov eax, [rel g_model_image_token_id]     ; load the model's persistent image token id
     mov [rel g_image_token_id], eax           ; re-apply the vision override, fresh pool is now installed
     SF_END                                    ; restore frame and return to caller
     global v7s_chat_image_mem                 ; export v7s_chat_image_mem entry point
v7s_chat_image_mem:
     SF_START v7s_chat_image_mem, rbx, rsi, rdi, r12, r13, r14, r15 ; begin stack frame
     SF_PTR cim_ctx                            ; InferenceContext* ctx, rcx
     SF_PTR cim_prompt                         ; const char* user_prompt, rdx
     SF_PTR cim_system                         ; const char* system_prompt, r8, optional/NULL
     SF_PTR cim_raw                            ; const uint8_t* raw_rgb, r9, caller-owned
     SF_DWORD cim_img_w                        ; int img_w, 5th stack arg
     SF_DWORD cim_img_h                        ; int img_h, 6th stack arg
     SF_PTR cim_out                            ; char* output, 7th stack arg
     SF_DWORD cim_out_len                      ; int output_len, 8th stack arg
     SF_DWORD cim_reset                        ; int reset, 9th stack arg
     SF_PTR cim_emb_ptr                        ; out pointer for embeddings
     SF_DWORD cim_emb_cnt                      ; out dword for embedding count
     SF_PTR cim_ds0_ptr                        ; out pointer for DeepStack 0
     SF_PTR cim_ds1_ptr                        ; out pointer for DeepStack 1
     SF_PTR cim_ds2_ptr                        ; out pointer for DeepStack 2
     SF_PTR cim_vision_prompt_base             ; base address of the 256KB stack prompt buffer
     SF_COMMIT                                 ; commit stack frame slots
; save register parameters to stack frame before the stack probe
; loop, which uses rcx as scratch, overwrites the ctx argument in rcx .
     TRACE_STR "chat_image_mem: entry"         ; diagnostic: confirm entry reached
     mov cim_ctx, rcx                          ; save ctx pointer to stack frame
     mov cim_prompt, rdx                       ; save prompt pointer to stack frame
     mov cim_system, r8                        ; save system_prompt pointer to stack frame
     mov cim_raw, r9                           ; save raw_rgb pointer to stack frame
     PROBE_VISION_PROMPT_BUF cim_vision_prompt_base ; reserve+probe 256KB, save base in frame
; Load remaining stack args, img_w, img_h, output, output_len, reset
     mov eax, [rbp + CIM_SARG5]                ; load 5th stack arg, img_w
     mov cim_img_w, eax                        ; save img_w to frame local
     mov eax, [rbp + CIM_SARG6]                ; load 6th stack arg, img_h
     mov cim_img_h, eax                        ; save img_h to frame local
     mov rax, [rbp + CIM_SARG7]                ; load 7th stack arg, output
     mov cim_out, rax                          ; save output pointer to frame local
     mov eax, [rbp + CIM_SARG8]                ; load 8th stack arg, output_len
     mov cim_out_len, eax                      ; save output_len to frame local
     mov eax, [rbp + CIM_SARG9]                ; load 9th stack arg, reset
     mov cim_reset, eax                        ; save reset flag to frame local
; Invalidate ALL vision-embedding state for a fresh session, same as chat_image
     INVALIDATE_VISION_STATE                   ; zero all vision state
; ViT encode + embeddings + marker prompt build + bound check + append
     VISION_ENCODE_AND_BUILD cim_, "chat_image_mem: set_vision_embeddings" ; shared body, see the macro header
     mov dword [rel g_img_turn], 1             ; image turn: v7s_chat selects __sys_template@img, cleared after render
     mov eax, cim_emb_cnt                      ; load encoded embedding count for tracing
     TRACE_REG "chat_image_mem: emb_cnt", rax  ; print number of image_pad tokens emitted
; Now delegate to v7s_chat to run the generation pipeline
     TRACE_STR "chat_image_mem: calling v7s_chat" ; diagnostic: delegating to text generation
; Signature: v7s_chat,ctx, prompt, system_prompt, output, output_len, reset
     mov rcx, cim_ctx                          ; arg1: InferenceContext* ctx
     mov rdx, cim_vision_prompt_base           ; arg2: built vision marker prompt, always built on this path
     mov r8, cim_system                        ; arg3: system_prompt, forwarded; NULL = no-system render
     mov r9, cim_out                           ; arg4: char* output
     sub rsp, CIMG_TEMP_FRAME                  ; allocate shadow space, 32 + 2 stack args, 16
     mov eax, cim_out_len                      ; load output_len
     mov [rsp + CIMG_TEMP_ARG5], rax           ; arg5: output_len, 5th arg
     mov eax, cim_reset                        ; load reset flag, 0=continue, !=0=fresh session
     mov [rsp + CIMG_TEMP_ARG6], rax           ; arg6: reset, 6th arg, forwarded from caller
     call v7s_chat                             ; execute ASM chat loop
     add rsp, CIMG_TEMP_FRAME                  ; restore stack pointer after v7s_chat call
; Mirror this image turn's token counts into the vision statistics slots
; v7s_chat wrote the main counters; r10 is volatile/dead, eax status preserved
     mov r10, [rel number_token_prefill]       ; image-turn prefill token count
     mov [rel number_token_prefill_vision], r10 ; statistics: vision prefill tokens
     mov r10, [rel number_token_generate]      ; image-turn generated token count
     mov [rel number_token_generate_vision], r10 ; statistics: vision generate tokens
     mov dword [rel g_image_token_id], 0       ; override inactive until a fresh pool is installed
; no return value: v7s_chat is void, so eax carries generation-loop residue here.
     add rsp, CIMG_PROMPT_BUF_BYTES            ; free the 256KB stack prompt buffer, SF_END won't
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore registers and return to caller
     global v7s_generate_image_mem             ; export v7s_generate_image_mem entry point
v7s_generate_image_mem:
     SF_START v7s_generate_image_mem, rbx, rsi, rdi, r12, r13, r14, r15 ; begin stack frame
     SF_PTR cgim_ctx                           ; InferenceContext* ctx, rcx
     SF_PTR cgim_prompt                        ; const char* user_prompt, rdx
     SF_PTR cgim_system                        ; const char* system_prompt, r8, must be NULL
     SF_PTR cgim_raw                           ; const uint8_t* raw_rgb, r9, caller-owned
     SF_DWORD cgim_img_w                       ; int img_w, 5th stack arg
     SF_DWORD cgim_img_h                       ; int img_h, 6th stack arg
     SF_PTR cgim_out                           ; char* output, 7th stack arg
     SF_DWORD cgim_out_len                     ; int output_len, 8th stack arg
     SF_PTR cgim_emb_ptr                       ; out pointer for embeddings
     SF_DWORD cgim_emb_cnt                     ; out dword for embedding count
     SF_PTR cgim_ds0_ptr                       ; out pointer for DeepStack 0
     SF_PTR cgim_ds1_ptr                       ; out pointer for DeepStack 1
     SF_PTR cgim_ds2_ptr                       ; out pointer for DeepStack 2
     SF_PTR cgim_vision_prompt_base            ; base address of the 256KB stack prompt buffer
     SF_COMMIT                                 ; commit stack frame slots
; save register parameters to stack frame before the stack probe
; loop, which uses rcx as scratch, overwrites the ctx argument in rcx .
     TRACE_STR "generate_image_mem: entry"     ; diagnostic: confirm entry reached
     mov cgim_ctx, rcx                         ; save ctx pointer to stack frame
     mov cgim_prompt, rdx                      ; save prompt pointer to stack frame
     mov cgim_system, r8                       ; save system_prompt pointer to stack frame
     mov cgim_raw, r9                          ; save raw_rgb pointer to stack frame
     test r8, r8                               ; system_prompt != NULL, contract violation ?
     jz .cgim_system_ok                        ; NULL, continue
     FATAL "generate_image_mem: system_prompt must be NULL (v7s_generate is template-free)" ; Reject silently-dropped caller intent
.cgim_system_ok:
     PROBE_VISION_PROMPT_BUF cgim_vision_prompt_base ; reserve+probe 256KB, save base in frame
; Load remaining stack args, img_w, img_h, output, output_len
     mov eax, [rbp + CIM_SARG5]                ; load 5th stack arg, img_w
     mov cgim_img_w, eax                       ; save img_w to frame local
     mov eax, [rbp + CIM_SARG6]                ; load 6th stack arg, img_h
     mov cgim_img_h, eax                       ; save img_h to frame local
     mov rax, [rbp + CIM_SARG7]                ; load 7th stack arg, output
     mov cgim_out, rax                         ; save output pointer to frame local
     mov eax, [rbp + CIM_SARG8]                ; load 8th stack arg, output_len
     mov cgim_out_len, eax                     ; save output_len to frame local
; Invalidate ALL vision-embedding state for a fresh session, same as chat_image_mem
     INVALIDATE_VISION_STATE                   ; zero all vision state
; call the slim compute-only ViT encoder directly, no file I/O, no callback indirection
     TRACE_STR "generate_image_mem: call vit_encode" ; diagnostic: about to run the shared encode+build body
; ViT encode + embeddings + marker prompt build + bound check + append
     VISION_ENCODE_AND_BUILD cgim_, "generate_image_mem: set_vision_embeddings" ; shared body, see the macro header
     TRACE_STR "generate_image_mem: body done" ; diagnostic: shared body finished, encode + embeddings + prompt built
     mov eax, cgim_emb_cnt                     ; load encoded embedding count for tracing
     TRACE_REG "generate_image_mem: emb_cnt", rax ; print number of image_pad tokens emitted
; Now delegate to v7s_generate for the single-turn separate-KV session
     TRACE_STR "generate_image_mem: calling v7s_generate" ; diagnostic: delegating to generate
; Signature: v7s_generate,ctx, prompt, output, output_len
     mov rcx, cgim_ctx                         ; arg1: InferenceContext* ctx
     mov rdx, cgim_vision_prompt_base          ; arg2: built vision marker prompt, always built on this path
     mov r8, cgim_out                          ; arg3: char* output
     mov r9d, cgim_out_len                     ; arg4: output_len, int, zero-extends
     sub rsp, CIMG_TEMP_FRAME                  ; shadow space for the direct call, matches the v7s_chat pattern
     call v7s_generate                         ; single-turn generate w/ separate KV, direct call: no API_CALL macro overhead
     add rsp, CIMG_TEMP_FRAME                  ; restore stack pointer after v7s_generate
     mov r10, [rel number_token_prefill]       ; image-turn prefill token count
     mov [rel number_token_prefill_vision], r10 ; statistics: vision prefill tokens
     mov r10, [rel number_token_generate]      ; image-turn generated token count
     mov [rel number_token_generate_vision], r10 ; statistics: vision generate tokens
; clear the image-token override: the pool is exhausted, so leaving it
; set would let a later text turn take the step-0 override against it
     mov dword [rel g_image_token_id], 0       ; override inactive until a fresh pool is installed
     add rsp, CIMG_PROMPT_BUF_BYTES            ; free the 256KB stack prompt buffer, SF_END won't
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore registers and return to caller
; Clean up frame local names, v7s_generate_image_mem
%undef cgim_ctx
%undef cgim_ctx_raw
%undef cgim_prompt
%undef cgim_prompt_raw
%undef cgim_system
%undef cgim_system_raw
%undef cgim_raw
%undef cgim_raw_raw
%undef cgim_img_w
%undef cgim_img_w_raw
%undef cgim_img_h
%undef cgim_img_h_raw
%undef cgim_out
%undef cgim_out_raw
%undef cgim_out_len
%undef cgim_out_len_raw
%undef cgim_emb_ptr
%undef cgim_emb_ptr_raw
%undef cgim_emb_cnt
%undef cgim_emb_cnt_raw
%undef cgim_ds0_ptr
%undef cgim_ds0_ptr_raw
%undef cgim_ds1_ptr
%undef cgim_ds1_ptr_raw
%undef cgim_ds2_ptr
%undef cgim_ds2_ptr_raw
%undef cgim_vision_prompt_base
%undef cgim_vision_prompt_base_raw
; Clean up frame local names
%undef cim_ctx
%undef cim_ctx_raw
%undef cim_prompt
%undef cim_prompt_raw
%undef cim_system
%undef cim_system_raw
%undef cim_raw
%undef cim_raw_raw
%undef cim_img_w
%undef cim_img_w_raw
%undef cim_img_h
%undef cim_img_h_raw
%undef cim_out
%undef cim_out_raw
%undef cim_out_len
%undef cim_out_len_raw
%undef cim_reset
%undef cim_reset_raw
%undef cim_emb_ptr
%undef cim_emb_ptr_raw
%undef cim_emb_cnt
%undef cim_emb_cnt_raw
%undef cim_ds0_ptr
%undef cim_ds0_ptr_raw
%undef cim_ds1_ptr
%undef cim_ds1_ptr_raw
%undef cim_ds2_ptr
%undef cim_ds2_ptr_raw
%undef cim_vision_prompt_base
%undef cim_vision_prompt_base_raw
%unmacro EMIT_IMAGE_PADS 1                     ; remove: splat the image_pad marker run, arity 1 = pad count
%unmacro PROBE_VISION_PROMPT_BUF 1             ; remove: commit the probed stack pages, arity 1 = base slot
%unmacro VISION_ENCODE_AND_BUILD 2             ; remove: shared encode+build body, arity 2 = prefix, trace
%unmacro copy_vision_token 0                   ; remove: append one marker BODY, arity 0, result in rdi, overwrites rsi
%undef CIMG_SARG5
%undef CIMG_TEMP_ARG5
%undef CIMG_TEMP_ARG6
%undef CIMG_TEMP_FRAME
%undef CIM_SARG5
%undef CIM_SARG6
%undef CIM_SARG7
%undef CIM_SARG8
%undef CIM_SARG9
%undef CIM_CALL8_FRAME
%undef CIM_CALL_ARG5
%undef CIM_CALL_ARG6
%undef CIM_CALL_ARG7
%undef CIM_CALL_ARG8
%undef CIMG_PROMPT_BUF_BYTES
%undef CIMG_PROBE_PAGE
%undef CIMG_MARKER_LIMIT
%undef CIMG_VEC_MARKER_MAX
