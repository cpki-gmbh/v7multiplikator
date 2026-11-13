; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; chat_prompt.inc, prompt render, encode, stats, token dump .
; ANCHOR RULE: local labels only, the non-local label anchor stays v7s_chat, SF_START in chat.inc .
; Falls through from chat_session.inc; falls through into chat_prefill.inc.

; Validate user_prompt is non-NULL before template rendering
     cmp c_user_prompt, 0                      ; user_prompt == NULL?
     je .chat_null_prompt_fatal                ; FATAL: NULL user prompt
     mov rax, c_user_prompt                    ; validated char* user_prompt
     mov [rel g_user_prompt], rax              ; publish for v7s_get_user_prompt
     cmp c_raw_mode, 0                         ; raw text continuation requested?
     je .chat_templated                        ; no, normal template selection below
     mov rdi, r14                              ; dest = rendered buffer base
     mov rsi, c_user_prompt                    ; src = raw user prompt
     mov ecx, CHAT_RENDERED_BUF - 1            ; byte budget, leave room for the NUL
.chat_raw_copy_loop:
     lodsb                                     ; load prompt byte, advances rsi
     test al, al                               ; NUL terminator reached?
     jz .chat_raw_copied                       ; yes, done
     stosb                                     ; store to rendered buffer, advances rdi
     dec ecx                                   ; byte budget--
     jnz .chat_raw_copy_loop                   ; copy until NUL or budget exhausted
.chat_raw_copied:
     mov byte [rdi], 0                         ; NUL-terminate, always in bounds: budget reserved 1
     jmp .chat_rendered                        ; straight to encode, bypasses v7sprintf entirely
.chat_templated:
; Template rule: session position determines FIRST. The system block is part of
; a turn-1 prefix that is ALREADY IN THE KV, a continuation never re-renders
; it, the inter-turn template is the only legal mid-session shape. A non-NULL
; system string on a continuation must byte-match the opening one, the caller
; passes reset=1 to rebuild the KV otherwise ; NULL on a continuation means
; the session opened without a system block.
     cmp dword c_pos, 0                        ; pos <= 0 = turn 1?, read only for the sign test
     jle .chat_sys_turn1                       ; turn 1: fresh system+user or user-only prefix
     mov rsi, c_system_prompt                  ; continuation system string, NULL allowed
     test rsi, rsi                             ; NULL system on a continuation?
     jz .chat_no_system                        ; yes, the session never opened with one, plain turn
     mov rdi, [rel g_sys_prompt_save]          ; rdi = the session's OWNED copy of the opener system string, engine-allocated at turn 1, the caller's original buffer was borrowed only for that call
     test rdi, rdi                             ; opener had no system block?
     jz .chat_sys_changed_fatal                ; yes, supplying one now needs a KV rebuild
     xor eax, eax                              ; single-pass exact comparison, including NUL
.chat_sys_compare:
     mov dl, [rsi + rax]                       ; caller byte
     cmp dl, [rdi + rax]                       ; owned opener byte, identical including NUL?
     jne .chat_sys_changed_fatal               ; mismatch or different length
     inc rax                                   ; advance only when the bytes match
     test dl, dl                               ; matched NUL on both strings?
     jnz .chat_sys_compare                     ; not finished
.chat_sys_same:
     mov r11, [rel g_tpl_multi]                ; opener's system prefix already stored in the KV
     jmp .chat_tpl_t1                          ; shared image override + prompt-only render
.chat_sys_changed_fatal:
     FATAL "chat: system prompt on a continuation call differs from the session opener, pass reset=1 to rebuild the KV history"
.chat_sys_turn1:
     mov rsi, c_system_prompt                  ; this turn's system string, NULL -> user-only; BORROWED, valid only for this call
     test rsi, rsi                             ; system prompt given?
     jz .chat_no_system                        ; user-only turn-1 template
; Own the opener bytes for the session validity interval: the continuation byte-match
; reads g_sys_prompt_save AFTER the caller's buffer may have been freed, so
; the session keeps a private copy, allocated once per opener turn .
     mov rcx, [rel g_sys_prompt_owned]         ; a previous session's owned copy?
     test rcx, rcx                             ; flags for the pointer, NOT the stale ZF from the system-prompt test: mov does not set flags
     jz .chat_sys_own_alloc                    ; none: clean slate, mdb_free_mem FATALs on NULL
     mov qword [rel g_sys_prompt_owned], 0     ; zero-pin BEFORE the free, idempotent, no double-free; explicit qword, a bare store assembles as BYTE and leaves the pointer non-null
     call mdb_free_mem                         ; release the old session's copy
.chat_sys_own_alloc:
     xor r8d, r8d                              ; byte index / length accumulation
.chat_sys_t1_len_loop:
     cmp byte [rsi + r8], 0                    ; NUL ends the opener system string
     je .chat_sys_t1_len_done                  ; length complete
     inc r8                                    ; one more byte
     jmp .chat_sys_t1_len_loop                 ; walk the opener string to its NUL
.chat_sys_t1_len_done:
     mov [rel g_sys_prompt_len], r8d           ; preserve the measured byte length across the allocator call, r8 is volatile
     lea rcx, [r8 + SYSPROMPT_NUL_EXTRA]       ; rcx = bytes including the NUL, r8 zero-extends
     call mdb_alloc_mem                        ; rax = owned block, FATALs on OOM
     mov [rel g_sys_prompt_owned], rax         ; track the block for reset free + destroy cleanup
     mov rdi, rax                              ; rdi = owned block dest
     mov rsi, c_system_prompt                  ; rsi = caller bytes, BORROWED; the alloc call may overwrite rsi (volatile), this is the canonical reload
     mov ecx, [rel g_sys_prompt_len]           ; exact measured bytes, excludes NUL
     inc ecx                                   ; include the terminator
     rep movsb                                 ; bulk-copy opener once; DF=0 on Win64 ABI entry
     mov rax, [rel g_sys_prompt_owned]         ; base; rep movsb advanced rdi
     mov [rel g_sys_prompt_save], rax          ; continuation compare reads the OWNED copy
     mov rsi, c_system_prompt                  ; reload the caller pointer for the sprintf vararg below
; System template: v7sprintf,rendered, buf_size, template, system_prompt, user_prompt
; v7sprintf ABI: rcx=buffer, edx=buf_size, r8=format, r9+=varargs
     mov r11, [rel g_tpl_sys]                  ; effective system+user template, think pre-folded by template index
     sub rsp, CHAT_TEMP_FRAME_1                ; shadow, 32 + 1 stack arg, 8 + align, 8 = 48
     mov rcx, r14                              ; arg1: rendered buffer
     mov edx, CHAT_RENDERED_BUF                ; arg2: buffer size, 32KB, zero-extends to rdx
     mov r8, r11                               ; arg3: template format string
     mov r9, rsi                               ; arg4: system prompt, vararg[0]
     mov rax, c_user_prompt                    ; arg5: user prompt, vararg[1], on stack
     mov [rsp + CHAT_TEMP_ARG5], rax           ; store 5th arg past shadow, no alias with r14
     xor eax, eax                              ; 0 XMM regs for variadic ABI
     call v7sprintf                            ; render full template
     add rsp, CHAT_TEMP_FRAME_1                ; restore 48 bytes
     jmp .chat_rendered                        ; Skip no-system path
.chat_no_system:
; Simple template: v7sprintf,rendered, buf_size, template, user_prompt
; v7sprintf ABI: rcx=buffer, edx=buf_size, r8=format, r9+=varargs
     cmp dword c_pos, 0                        ; pos <= 0 = a fresh, reset session?, read only for the sign test
     jg .chat_no_sys_keep                      ; continuation: the saved opener stays valid
     mov rcx, [rel g_sys_prompt_owned]         ; fresh session without a system block: a stale owned opener?
     test rcx, rcx                             ; flags for the pointer, NOT the stale ZF from the c_pos test: mov does not set flags
     jz .chat_no_sys_cleared                   ; none: nothing to release
     mov qword [rel g_sys_prompt_owned], 0     ; zero-pin BEFORE the free, idempotent, no double-free; explicit qword, a bare store assembles as BYTE and leaves the pointer non-null
     call mdb_free_mem                         ; release the stale copy
.chat_no_sys_cleared:
     mov qword [rel g_sys_prompt_save], 0      ; fresh session: clear a stale opener pointer, the owned copy is freed above
.chat_no_sys_keep:
     mov r11, [rel g_tpl_t1]                   ; effective turn-1 template, think pre-folded by template index
     cmp c_pos, 0                              ; continuation turn?, pos resolved above
     jle .chat_tpl_t1                          ; turn 1, t1 stands
     mov r11, [rel g_tpl_multi]                ; effective turn-2+ template, multi+think pre-processed
.chat_tpl_t1:
     cmp dword [rel g_img_template_valid], 0   ; image-turn template stored at init?
     je .chat_tpl_selected                     ; no, selected template stands
     cmp dword [rel g_img_turn], 0             ; this call an image turn?
     je .chat_tpl_selected                     ; no, selected template stands
     lea r11, [rel g_img_template]             ; condition met, swap in image-turn template, most specific: img > multi > think
.chat_tpl_selected:
     sub rsp, 32                               ; shadow space for v7sprintf
     mov rcx, r14                              ; arg1: rendered buffer
     mov edx, CHAT_RENDERED_BUF                ; arg2: buffer size, 32KB, zero-extends to rdx
     mov r8, r11                               ; arg3: template format string
     mov r9, c_user_prompt                     ; arg4: user prompt, vararg[0]
     xor eax, eax                              ; 0 XMM regs for variadic ABI
     call v7sprintf                            ; render prompt-only template
     add rsp, 32                               ; restore shadow space
.chat_rendered:
; End of image turn: clear the flag so subsequent text turns use the base
; template even inside the same session, set again by the next image call .
     mov dword [rel g_img_turn], 0             ; self-clearing after template selection
; prompt_tokens already allocated on stack, after rendered buffer, no allocator call needed
     mov rcx, c_ctx                            ; InferenceContext*, temp for tokenizer load
     mov rcx, [rcx + CTX_TOKENIZER]            ; arg1: Tokenizer* from ctx
     mov r8, c_prompt_tokens                   ; arg3: prompt_tokens output buffer
     lea r9, c_num_prompt_tokens               ; arg4: &n_tokens, SF local via rbp
     sub rsp, CHAT_TEMP_FRAME_1                ; shadow, 32 + 1 arg + align
     mov rdx, r14                              ; arg2: rendered string, cached base pointer
     mov eax, CHAT_MAX_PROMPT_TOKENS           ; eax = 32768, zero-extends to rax
     mov [rsp + CHAT_TEMP_ARG5], rax           ; arg5: max_tokens for encode,, full 64-bit slot
     call encode                               ; BPE-encode rendered prompt
     add rsp, CHAT_TEMP_FRAME_1                ; Restore RSP
; Phase 1 timestamp + prefill token count, rax/rdx unused across the probe
     RDTSC_TO_MEM rdtsc_after_tokenization     ; statistics: TSC after prompt encode
     mov eax, c_num_prompt_tokens              ; load prompt token count, zero-extends
     mov [rel number_token_prefill], rax       ; statistics: prompt tokens this round
     cmp dword [rel enable_token_dump], 0      ; token dump enabled?
     je .chat_prompt_dump_skip                 ; no, skip entirely, jump PAST add rsp, 32
; Save num_prompt_tokens in non-volatile r12d, avoids push which misaligns stack
     mov r12d, eax                             ; r12d = num_prompt_tokens, free until gen loop
     sub rsp, 32                               ; shadow space for dump_token_stream calls
     mov rsi, c_prompt_tokens                  ; rsi = prompt_tokens array
     xor edi, edi                              ; edi = 0, loop index
     mov r15d, c_pos                           ; r15d = start position, non-volatile, preserved across calls
.chat_prompt_dump_loop:
     cmp edi, r12d                             ; i < num_prompt_tokens, saved in r12d, loop bound
     jae .chat_prompt_dump_done                ; no, done
     mov rax, c_ctx                            ; rax = InferenceContext*
     mov rcx, [rax + CTX_TOKENIZER]            ; arg1: Tokenizer* directly
     mov edx, [rsi + rdi*DWORD_SCALE]          ; arg2: token_id directly into edx
     mov r8d, r15d                             ; arg3: position, from non-volatile, not overwritten
     call dump_token_stream                    ; dump prompt token to stdout
     inc r15d                                  ; position++
     inc edi                                   ; i is incremented
     jmp .chat_prompt_dump_loop                ; continue
.chat_prompt_dump_done:
     add rsp, 32                               ; restore shadow space
.chat_prompt_dump_skip:
     cmp dword c_num_prompt_tokens, 0          ; zero encoded tokens, raw-mode empty prompt: chat-border correction, read only for the zero test; prefill's Phase-2 countdown is a do-while and would spin ~2^32 trips on an empty batch
     jz .chat_finish                           ; stable empty turn: output stays "", KV cache untouched, pos unchanged
