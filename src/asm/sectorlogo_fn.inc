; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define LG_FN_SAVED_HAS_RET 32                 ; dword: saved has_returned, exclusive slot
%define LG_FN_SAVED_RET_VAL 40                 ; dword: saved return_val, exclusive slot
%define LG_FN_STACK_BODY 48                    ; dword: function body cons cell
%define LG_FN_STACK_PARAMS 56                  ; dword: params list cursor
%define LG_FN_SAVED_LOCAL_COUNT 72             ; dword: saved sl_local_count, scope discard
%define LG_FN_SAVED_PARAMS 76                  ; dword: original params list, for bind pass
%define LG_FN_ARG_COUNT 80                     ; dword: arg index counter
%define LG_FN_ARG_VALS_OFF 84                  ; dword[]: pre-evaluated arg values, up to 16
%define LG_FN_ARG_VAL_SCALE 4                  ; bytes per arg value entry
%define LG_FN_MAX_ARGS 16                      ; max supported function args
; EvalPrimary overwrites r15d, must use stack
; ============================================================================
; lg_clean_param_list,eax = params cons list -> eax = cleaned params list
; Strips leading ':' from parameter atom names, Logo convention: :X -> X .
; If the first char of a param's string is ':', interns the rest as the
; clean name. Recursively processes the entire list, building a new list.
; ============================================================================
lg_clean_param_list:
     push rbx                                  ; save non-volatile
     push r12                                  ; save non-volatile
     cmp rsp, [rel sl_stk_limit]               ; guard the per-param recursion like the file's other entries: Define's param list is evaluated data, data-as-code, so no parser cap applies and the walk compounds with caller depth
     jb sl_stack_overflow                      ; runaway param list -> named FATAL, not a guard-page AV
     mov r12d, eax                             ; r12d = original params
     cmp eax, SL_CONS_TAG                      ; below CONS tag range, unsigned ?
     jb .nil_or_atom                           ; yes: NIL or plain atom
     cmp eax, SL_FLOAT_TAG                     ; in the FLOAT band, bits 31:30 both set ?
     jae .nil_or_atom                          ; yes: a FLOAT is not a param list either, Car returns NIL for it and would report it as a bad element instead of a bad list
; process, car params, strip ':' prefix if present
; eax still holds the original params, cmp/jb only read it, no reload needed
     call Car                                  ; eax = first param atom
     mov ebx, eax                              ; ebx = raw param atom
; tag guard: param name must be a plain atom, not CONS or NUMBER
     test ebx, ebx                             ; NIL?
     jz .param_notatom                         ; yes: FATAL
     js .param_notatom                         ; bit31 set = CONS: FATAL
     test ebx, SL_NUMBER_TAG                   ; NUMBER?
     jnz .param_notatom                        ; yes: FATAL
     call lg_atom_unprefix                     ; strip ':' prefix if present, shared helper; tag guards above satisfy its plain-atom contract
     call lg_atom_lowercase                    ; fold A-Z -> a-z: string-literal params keep case, but body :VAR lookups are lexer-folded, without this an uppercase param binds under a name the body can never read
     mov ebx, eax                              ; ebx = clean param atom
; recurse on, cdr params
     mov eax, r12d                             ; eax = original params
     call Cdr                                  ; eax = rest of params
     call lg_clean_param_list                  ; eax = cleaned rest
     mov edx, eax                              ; edx = cleaned rest, cdr
     mov eax, ebx                              ; eax = clean param, car
     call Cons                                 ; eax =, clean_param.cleaned_rest
     jmp .done                                 ; return cleaned list
.nil_or_atom:
     test eax, eax                             ; NIL?
     jz .done                                  ; NIL: eax == 0 is the return value
     FATAL_VALUE "function parameter list must be a list", eax ; atom is not a valid param list, eax holds the offender
.done:
     pop r12                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     ret                                       ; return to caller
.param_notatom:
     FATAL_VALUE "function parameter must be a name", ebx ; CONS/NUMBER/NIL in param list, ebx holds the offender
lg_invoke_user_fn:
; --- recursion depth guardrail ---
     cmp dword [rel sl_call_depth], SL_MAX_CALL_DEPTH ; exceeded?
     jge .depth_overflow                       ; yes: FATAL
     cmp rsp, [rel sl_stk_limit]               ; private-stack watermark: 65536 x ~350-700 B is the budgeted worst case, the block holds it with headroom
     jb sl_stack_overflow                      ; runaway recursion -> named FATAL, not an AV
     inc dword [rel sl_call_depth]             ; increment depth
     inc dword [rel sl_trace_indent]           ; trace indent: entering function
; --- extract params and body from descriptor ---
     push rbx                                  ; save non-volatile
     push r14                                  ; save caller's env, bindings must not leak
     push r15                                  ; save non-volatile, evaluator scratch across the Cdr/Car calls below
     mov ecx, [rel sl_fn_has_returned]         ; save caller's return flag
     mov edx, [rel sl_fn_return_val]           ; save caller's return value
     sub rsp, 152                              ; frame: saved flags + params/body cursors + arg slot array, LG_FN_* offsets above
     mov [rsp + LG_FN_SAVED_HAS_RET], ecx      ; store saved has_returned, exclusive slot
     mov [rsp + LG_FN_SAVED_RET_VAL], edx      ; store saved return_val, exclusive slot
     call Cdr                                  ; descriptor -> inner (params . body)
     call PopCons                              ; EBX = params, EAX = body
     mov [rsp + LG_FN_STACK_PARAMS], ebx       ; store params cursor
     mov [rsp + LG_FN_SAVED_PARAMS], ebx       ; save original params for bind pass
     mov [rsp + LG_FN_STACK_BODY], eax         ; store body
; --- reset return state for this call ---
     mov qword [rel sl_fn_has_returned], 0     ; clear return flag + return value, adjacent dwords
     mov dword [rsp + LG_FN_ARG_COUNT], 0      ; arg count = 0
     mov eax, [rsp + LG_FN_STACK_PARAMS]       ; eax = params cursor, entry load only; the loop tail re-enters with Cdr's result, so the loop top does not reload it
     test eax, eax                             ; any params at all?
     jz .eval_args_done                        ; no: zero-arg function, nothing to evaluate
.eval_args_loop:
     mov ebx, [rsp + LG_FN_ARG_COUNT]          ; ebx = current arg index, its durable home is this frame slot, the evals below overwrite ebx and it is reloaded from the slot after them
     cmp ebx, LG_FN_MAX_ARGS                   ; exceeded 16-arg limit?
     jge .fn_err_max_args                      ; Reject argument counts that would overwrite the stack argument area.
     call EvalPrimary                          ; eax = primary value
     call EvalMulChain                         ; fold */% at higher precedence
     call EvalAddChain                         ; fold +- at higher precedence
     mov ebx, [rsp + LG_FN_ARG_COUNT]          ; reload arg index, the evals overwrite ebx; the frame slot is untouched above offset 31, callees can only scratch [rsp, rsp+32 ; lg_invoke_exe_fn reloads its params cursor the same way
     mov [rsp + LG_FN_ARG_VALS_OFF + rbx*LG_FN_ARG_VAL_SCALE], eax ; store arg value
     inc dword [rsp + LG_FN_ARG_COUNT]         ; arg count++
; advance param cursor
     mov eax, [rsp + LG_FN_STACK_PARAMS]       ; reload cursor, the three eval calls overwritten eax
     call Cdr                                  ; eax = rest of params
     mov [rsp + LG_FN_STACK_PARAMS], eax       ; advance param cursor
     test eax, eax                             ; more params?
     jnz .eval_args_loop                       ; yes: evaluate the next argument, bottom-tested, no per-argument jmp
.eval_args_done:
     mov ecx, [rel sl_local_count]             ; capture count AFTER arg eval, includes caller LOCALs
     mov [rsp + LG_FN_SAVED_LOCAL_COUNT], ecx  ; save the post-eval count
; Second pass: bind param_name -> arg_value, reading args from stack area
; Reset param cursor to the beginning of the param list
     mov eax, [rsp + LG_FN_SAVED_PARAMS]       ; eax = original params list
     mov [rsp + LG_FN_STACK_PARAMS], eax       ; restore param cursor
     mov dword [rsp + LG_FN_ARG_COUNT], 0      ; reset arg index
     test eax, eax                             ; any params to bind?
     jz .bind_done                             ; no: done binding
.bind_loop:
     call PopCons                              ; fused: ebx = param_name, eax = rest of params
     mov [rsp + LG_FN_STACK_PARAMS], eax       ; advance parameter cursor
; read pre-evaluated arg value from stack area
     mov ecx, [rsp + LG_FN_ARG_COUNT]          ; ecx = current arg index
     mov edx, [rsp + LG_FN_ARG_VALS_OFF + rcx*LG_FN_ARG_VAL_SCALE] ; edx = arg value
     inc dword [rsp + LG_FN_ARG_COUNT]         ; advance arg index
; bind param_name -> arg_value in local env
     mov eax, ebx                              ; eax = param_name, from register park
     call Cons                                 ; eax =, param_name.arg_value pair, TAGGED
     mov ecx, [rel sl_local_count]             ; ecx = current entry count, Cons overwritten ecx, reload mandatory
     cmp ecx, SL_LOCAL_SET_MAX                 ; over the limit?
     jae .fn_err_local_overflow                ; yes: FATAL
     lea r9, [rel sl_local_set]                ; r9 = set base, RIP-rel has no index form
     mov [r9 + rcx*SL_LOCAL_SET_SCALE], ebx    ; set[count].var = param name
     btr eax, SL_CONS_BIT                      ; Cons returns a bit31-tagged cell with bit30 clear
     mov [r9 + rcx*SL_LOCAL_SET_SCALE + SL_LOCAL_CELL_OFF], eax ; persist raw cell index
     bts eax, SL_CONS_BIT                      ; restore CONS tag before building environment
     inc dword [rel sl_local_count]            ; count++, entry visible NOW
     mov edx, r14d                             ; edx = current env, cdr
     call Cons                                 ; eax =, ,param_name.arg_value.env
     mov r14d, eax                             ; update env
     mov eax, [rsp + LG_FN_STACK_PARAMS]       ; reload cursor for the next test, REQUIRED: Cons overwritten eax; carrying it in a register across this binding body would cost a callee-saved register that is not available here
     test eax, eax                             ; more params?
     jnz .bind_loop                            ; yes: bind the next param, bottom-tested, no per-param jmp
.bind_done:
; --- execute body stream ---
; Body is on the stack, not in a volatile register
     mov esi, [rsp + LG_FN_STACK_BODY]         ; stream = body, from stack slot
     mov edx, r14d                             ; env = local env
     call EvalLogoStream                       ; edx = updated env, DISCARDED: r14 is restored from the caller's value at the epilogue, so the body's env never escapes this function
     cmp dword [rel sl_stop_requested], 0      ; STOP requested in this body?
     je .no_stop_pending                       ; no: normal return path
     mov dword [rel sl_stop_requested], 0      ; yes: consume, fn boundary caught it
     xor eax, eax                              ; return value = NIL, STOP outputs nothing
     jmp .fn_return_with                       ; skip the return-val extraction
.no_stop_pending:
     cmp dword [rel sl_continue_requested], 0  ; CONTINUE requested in this body?
     jz .no_continue_pending                   ; no: normal return path
     mov dword [rel sl_continue_requested], 0  ; yes: consume, fn boundary caught it
     xor eax, eax                              ; return value = NIL, CONTINUE outputs nothing
     jmp .fn_return_with                       ; skip the return-val extraction
.no_continue_pending:
; --- extract return value ---
     mov eax, [rel sl_fn_return_val]           ; eax = return value
.fn_return_with:
; --- restore caller's return state ---
     mov edx, [rsp + LG_FN_SAVED_RET_VAL]      ; restore caller's return_val
     mov ecx, [rsp + LG_FN_SAVED_HAS_RET]      ; restore caller's has_returned
     mov [rel sl_fn_return_val], edx           ; restore
     mov [rel sl_fn_has_returned], ecx         ; restore
     mov ecx, [rsp + LG_FN_SAVED_LOCAL_COUNT]  ; restore caller's LOCAL count
     mov [rel sl_local_count], ecx             ; discard this function's LOCAL declarations
; --- decrement recursion depth ---
     dec dword [rel sl_call_depth]             ; decrement depth
     dec dword [rel sl_trace_indent]           ; trace indent: leaving function
     add rsp, 152                              ; restore shadow + stack slots; 24+152=176 ≡ 0 mod 16
     pop r15                                   ; restore non-volatile
     pop r14                                   ; restore caller's env
     pop rbx                                   ; restore non-volatile
     ret                                       ; return to caller
.depth_overflow:
     mov eax, [rel sl_call_depth]              ; depth at abort, raw count, not yet tagged
     bts eax, 30                               ; set the NUMBER tag bit for the fatal-value formatter
     FATAL_VALUE "recursion depth exceeded (infinite recursion? abort depth shown)", eax ; guardrail trip names the depth
.fn_err_max_args:
     FATAL "SectorLogo: user function exceeds 16 arguments" ; Validate the maximum stack argument count.
.fn_err_local_overflow:
     FATAL_VALUE "parameter binding exceeded the LOCAL set (scope stack full, or leak/deep recursion?)", ebx ; scope stack exhausted, names the param being bound
lg_atom_unprefix:
     push rbx                                  ; save non-volatile, holds the atom
     push rsi                                  ; save non-volatile, overwritten by Intern
     push rdi                                  ; save non-volatile, dest buffer cursor
     mov ebx, eax                              ; ebx = original atom
     movzx ecx, byte [r13 + rbx]               ; ecx = type byte from the original atom, .unp_check_string_type's SL_TYPE_STRING compare is its only consumer
     lea rsi, [r13 + rbx + SL_ATOM_TEXT_OFF]   ; rsi = atom string in heap
     cmp byte [rsi], ':'                       ; first char a colon?
     jne .unp_check_string_type                ; no: check if STRING type needs WORD conversion
     inc rsi                                   ; skip past ':'
; :-prefixed atoms always force WORD type, variable names must be WORD,
; regardless of whether the source was a string literal or a bare word
     mov byte [rel sl_tok_buf], SL_TYPE_WORD   ; force WORD type
     jmp .unp_copy_to_buf                      ; copy content, minus ':' to tok_buf for re-intern
.unp_check_string_type:
     cmp cl, SL_TYPE_STRING                    ; atom is STRING type?
     jne .unp_done                             ; no: already WORD, return as-is
; STRING -> WORD conversion: overwrite the type byte before copying content
     mov byte [rel sl_tok_buf], SL_TYPE_WORD   ; force WORD type for variable names
.unp_copy_to_buf:
; rsi = first content char, past ':' for colon-strip, or at atom text for STRING-to-WORD
     lea rdi, [rel sl_tok_buf + SL_ATOM_TEXT_OFF] ; dest = tok_buf content area
     lea r8, [rel sl_tok_buf + SL_TOK_BUF_END_OFF] ; r8 = last writable byte; the jae bound leaves SIZE-2 for content and SIZE-1 for the NUL at .unp_copy_done
.unp_copy:
     mov al, byte [rsi]                        ; load char
     test al, al                               ; NUL?
     jz .unp_copy_done                         ; yes: done
     cmp rdi, r8                               ; tok_buf full?
     jae .unp_overflow                         ; yes: name too long for the cleaner buffer
     mov byte [rdi], al                        ; store char
     inc rsi                                   ; advance source
     inc rdi                                   ; advance dest
     jmp .unp_copy                             ; continue
.unp_copy_done:
     mov byte [rdi], 0                         ; NUL-terminate
     lea rsi, [rel sl_tok_buf]                 ; source for Intern
     call Intern                               ; eax = clean atom
     mov ebx, eax                              ; ebx = clean atom
.unp_done:
     mov eax, ebx                              ; return, possibly original atom
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     ret                                       ; return to caller
.unp_overflow:
     FATAL "SectorLogo: atom name exceeds token buffer (colon strip)" ; 1 MiB atom through the cleaner: abort, never truncate
lg_atom_lowercase:
     push rbx                                  ; save non-volatile
     push rsi                                  ; save non-volatile, overwritten by Intern
     push rdi                                  ; save non-volatile, dest buffer cursor
     mov ebx, eax                              ; ebx = original atom
     mov byte [rel sl_tok_buf], SL_TYPE_WORD   ; tok_buf type = WORD, names are always words: "X" and :X both resolve to WORD "x"
     lea rsi, [r13 + rbx + SL_ATOM_TEXT_OFF]   ; rsi = atom string in heap
     lea rdi, [rel sl_tok_buf + SL_ATOM_TEXT_OFF] ; dest = tok_buf content area, lowercase
     lea r8, [rel sl_tok_buf + SL_TOK_BUF_END_OFF] ; r8 = last writable byte; the jae bound leaves SIZE-2 for content and SIZE-1 for the NUL at .lc_copy_done
.lc_copy:
     mov al, byte [rsi]                        ; load char
     test al, al                               ; NUL?
     jz .lc_copy_done                          ; yes: done
     cmp al, 'A'                               ; below 'A'?
     jb .lc_store                              ; yes: not uppercase
     cmp al, 'Z'                               ; above 'Z'?
     ja .lc_store                              ; yes: not uppercase
     add al, 32                                ; fold A-Z -> a-z
.lc_store:
     cmp rdi, r8                               ; tok_buf full?
     jae .lc_overflow                          ; yes: name too long for the cleaner buffer
     mov byte [rdi], al                        ; store char, possibly folded
     inc rsi                                   ; advance source
     inc rdi                                   ; advance dest
     jmp .lc_copy                              ; continue
.lc_copy_done:
     mov byte [rdi], 0                         ; NUL-terminate
     lea rsi, [rel sl_tok_buf]                 ; source for Intern
     call Intern                               ; eax = lowercase-interned atom
     mov ebx, eax                              ; ebx = result
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     ret                                       ; return to caller
.lc_overflow:
     FATAL "SectorLogo: atom name exceeds token buffer (case fold)" ; 1 MiB atom through the cleaner: abort, never truncate
%define LG_XFN_EXE_OFF 32                      ; qword, exe C-string pointer, shifted past outbound shadow space
%define LG_XFN_FMT_OFF 40                      ; qword, format C-string pointer
%define LG_XFN_ARGC_OFF 48                     ; dword, evaluated argument count
%define LG_XFN_PARAMS_OFF 52                   ; dword, remaining params list cursor
%define LG_XFN_ARGV_OFF 56                     ; 8 qwords, 64 bytes, vararg slots
%define LG_XFN_SLOTS 104                       ; data-slot area in bytes, occupying frame offsets 32..135 and FULLY used, top slot is LG_XFN_ESC_SAVED at 128..135
%define LG_XFN_SHADOW 32                       ; outbound-call shadow space at frame offsets 0..31, BELOW the data slots, so the total frame is LG_XFN_SLOTS + LG_XFN_SHADOW
%define LG_XFN_QWORD 8                         ; bytes per vararg slot
%define SL_XFN_MAX_ARGS 8                      ; max arguments per exe function call
%define LG_XFN_ESC_CURSOR 120                  ; qword, escape-arena cursor, next free byte in sl_xfn_escape_buf ; MUST sit PAST argv[7], 56 + 8*8 = 120 or the 8th string argument's argv store, argv[7], the last slot overwrites it
%define LG_XFN_ESC_SAVED 128                   ; qword, saved parent cursor, reentrancy guard ; must sit past the cursor slot, and past argv[7]
%define LG_XFN_ESC_NUL_ADV 1                   ; cursor advance past the NUL terminator byte
%define SQF_SPEC_OFF 1                         ; fmt byte offset of the spec char after '%'
%define SQF_QUOTE_OFF 2                        ; fmt byte offset of the char after the spec, '"' check

%macro XFN_PROBE 0
     mov r11d, eax                             ; park the tagged value, the probe writes eax and never touches r11
     mov rcx, [rsp + LG_XFN_FMT_OFF]           ; fmt C string
     mov edx, [rsp + LG_XFN_ARGC_OFF]          ; this arg's 0-based conversion ordinal
     call sl_xfn_quote_follows                 ; eax = M, >=0 / -1, raw / -2, not %s
     xchg r11d, eax                            ; eax = tagged value, r11d = decision
%endmacro
sl_xfn_quote_follows:
     xor r8d, r8d                              ; r8d = conversion index, 0-based
.sqf_scan:
     mov al, byte [rcx]                        ; current format byte, only AL is consumed
     test al, al                               ; NUL: end of fmt?
     jz .sqf_no                                ; ordinal-th conversion never found
     cmp al, '%'                               ; conversion introducer?
     jne .sqf_next                             ; no: plain fmt byte, advance one
     mov r9b, byte [rcx + SQF_SPEC_OFF]        ; spec byte after percent, only R9B used
     test r9b, r9b                             ; NUL right after '%', format ends mid-conversion ?
     jz .sqf_no                                ; yes: no spec byte, stop here. Without this the two add-rcx,2 steps below would land ONE BYTE PAST the terminator and the walk would continue through the neighbouring heap until the next zero byte
     cmp r9b, '%'                              ; '%%' escaped literal percent?
     je .sqf_next2                             ; yes: not a conversion, skip both bytes
     cmp r9b, 's'                              ; conversion test FIRST: an unsupported spec, e.g. %i at
     je .sqf_is_conv                           ; the ordinal position is NOT ours to claim, it advances
     cmp r9b, 'd'                              ; neither the walk's ordinal nor rcx here; testing the
     je .sqf_is_conv                           ; ordinal before membership handed %i the next %s's slot
     cmp r9b, 'f'                              ; the ordinal desync this block exists to prevent
     je .sqf_is_conv                           ; %f is a vararg-consuming conversion
     cmp r9b, 'c'                              ; %c
     jne .sqf_next2                            ; any other byte after '%': not a conversion, skip both; %c falls through to the conversion case
.sqf_is_conv:
     cmp r8d, edx                              ; a real conversion, the ordinal-th one?
     jne .sqf_conv_skip                        ; no: count it, r8d++ / rcx+=2 and keep walking, must mirror .scc_count in sl_xfn_count_conversions; ours falls through to the classify
.sqf_classify:
     cmp r9b, 's'                              ; string conversion?
     jne .sqf_nots                             ; no: the caller must not pass a char* into this slot, report -2, not "raw context"
     xor r9d, r9d                              ; r9d = M = format backslash run after the spec, r9 held the spec byte, already consumed by the 's' test above
.sqf_bsrun:
     cmp byte [rcx + SQF_QUOTE_OFF + r9], '\'  ; another format backslash?
     jne .sqf_bsrun_done                       ; no: run of M ends here, the format's NUL ends it too, since NUL is neither '\' nor '"'
     inc r9d                                   ; M++
     jmp .sqf_bsrun                            ; keep counting, bounded by the format's NUL, no explicit bound needed, it is a NUL-terminated C string by contract
.sqf_bsrun_done:
     cmp byte [rcx + SQF_QUOTE_OFF + r9], '"'  ; is the '"' right after the M backslashes?
     jne .sqf_no                               ; no: nothing will close a section here, raw context
     mov eax, r9d                              ; yes: return M, a quote follows M format-contributed backslashes
     ret                                       ; eax = M >= 0
.sqf_conv_skip:
     inc r8d                                   ; one more conversion seen
     add rcx, 2                                ; advance past '%' + spec byte
     jmp .sqf_scan                             ; resume the fmt walk
.sqf_next2:
     add rcx, 2                                ; advance 2 bytes WITHOUT advancing the ordinal, '%%' escape or a spec v7sprintf does not consume
     jmp .sqf_scan                             ; resume the fmt walk
.sqf_next:
     inc rcx                                   ; advance past the plain byte
     jmp .sqf_scan                             ; resume the fmt walk
.sqf_no:
     mov eax, -1                               ; raw context: trailing runs stay single-width
     ret                                       ; eax = -1
.sqf_nots:
     mov eax, -2                               ; the ordinal-th conversion is not %s, the caller must not pass a char* here
     ret                                       ; eax = -2
sl_xfn_count_conversions:
     xor r8d, r8d                              ; r8d = conversion count
.scc_scan:
     mov al, byte [rcx]                        ; current format byte, only AL is consumed
     test al, al                               ; NUL: end of fmt?
     jz .scc_done                              ; yes: done
     cmp al, '%'                               ; conversion introducer?
     jne .scc_next                             ; no: plain fmt byte
     mov al, byte [rcx + SQF_SPEC_OFF]         ; spec byte, only AL used
     test al, al                               ; NUL right after '%'?
     jz .scc_done                              ; yes: format ends mid-conversion, stop here, never step past the terminator
     cmp al, '%'                               ; '%%' escaped literal percent?
     je .scc_adv2                              ; yes: not a conversion, skip both bytes
     cmp al, 's'                               ; %s conversion?
     je .scc_count                             ; yes: one conversion
     cmp al, 'd'                               ; %d conversion?
     je .scc_count                             ; yes: one conversion
     cmp al, 'f'                               ; %f conversion?
     je .scc_count                             ; yes: one conversion
     cmp al, 'c'                               ; %c conversion?
     jne .scc_adv2                             ; no: not a conversion, skip both bytes
.scc_count:
     inc r8d                                   ; one vararg-consuming conversion
.scc_adv2:
     add rcx, 2                                ; advance past '%' + spec byte
     jmp .scc_scan                             ; resume the fmt walk
.scc_next:
     inc rcx                                   ; advance past the plain byte
     jmp .scc_scan                             ; resume the fmt walk
.scc_done:
     mov eax, r8d                              ; eax = conversion count
     ret                                       ; eax = count
lg_invoke_exe_fn:
; --- recursion depth guardrail, shared with user functions ---
     cmp dword [rel sl_call_depth], SL_MAX_CALL_DEPTH ; exceeded?
     jge .xfn_depth_overflow                   ; yes: FATAL
     cmp rsp, [rel sl_stk_limit]               ; private-stack watermark: DEF_EXE_FN frames are larger than user-fn frames, same budget, same trip
     jb sl_stack_overflow                      ; runaway recursion -> named FATAL, not an AV
     inc dword [rel sl_call_depth]             ; increment depth
     push rbx                                  ; save non-volatile, desc scratch
     push rsi                                  ; save non-volatile, the capture copy loop advances it; Intern overwrites it
     push r14                                  ; save caller's env, arg-eval bindings must not leak
     push rdi                                  ; save non-volatile, capture-copy dest cursor; Intern overwrites it
     push r15                                  ; save non-volatile, desc scratch
     sub rsp, LG_XFN_SLOTS + LG_XFN_SHADOW     ; 136 bytes total, 32 shadow + 104 data ; 5 pushes + 136 = 176: 16-NEUTRAL delta, preserves the inherited entry parity, neither 104 nor 136 is itself 16-aligned, only the TOTAL is, so the v7_execute_fmt border still needs CORRECT_RSP
     mov r15d, eax                             ; r15d = xfn_desc
     call Cdr                                  ; descriptor tail = params . exe.fmt
     call PopCons                              ; fused: ebx = params, eax = exe.fmt
     mov [rsp + LG_XFN_PARAMS_OFF], ebx        ; park params cursor
     call PopCons                              ; fused: ebx = exe atom, eax = fmt atom
     mov r15d, eax                             ; keep fmt until exe pointer is stored
     mov eax, ebx                              ; exe atom
     and eax, SL_PAYLOAD_MASK                  ; strip tag bits -> heap offset
     lea rcx, [r13 + rax + SL_ATOM_TEXT_OFF]   ; exe C string, skip type byte
     mov [rsp + LG_XFN_EXE_OFF], rcx           ; park exe pointer
     mov eax, r15d                             ; fmt atom, already unpacked
     and eax, SL_PAYLOAD_MASK                  ; strip tag bits -> heap offset
     lea rcx, [r13 + rax + SL_ATOM_TEXT_OFF]   ; fmt C string, skip type byte
     mov [rsp + LG_XFN_FMT_OFF], rcx           ; park fmt pointer, rcx stays the live arg for the call below
     call sl_xfn_count_conversions             ; eax = conversions in the format
     mov ecx, eax                              ; ecx = conversion count
     xor edx, edx                              ; edx = declared param count
     mov eax, [rsp + LG_XFN_PARAMS_OFF]        ; eax = params cursor, parked earlier, not yet advanced
.scp_count:
     test eax, eax                             ; params exhausted?
     jz .scp_arity                             ; yes: compare and FATAL on mismatch
     call Cdr                                  ; eax = rest of params
     inc edx                                   ; one declared parameter
     jmp .scp_count                            ; keep counting
.scp_arity:
     cmp ecx, edx                              ; conversions == declared params?
     jne .xfn_err_arity                        ; no: FATAL, name the invariant
; --- evaluate EXACTLY len,params arguments from the caller's stream --
     mov dword [rsp + LG_XFN_ARGC_OFF], 0      ; arg count = 0
     mov rdx, [rel sl_xfn_arena_cursor]        ; rdx = current global arena cursor
     mov [rsp + LG_XFN_ESC_SAVED], rdx         ; Save the invoking frame cursor on this frame
     mov [rsp + LG_XFN_ESC_CURSOR], rdx        ; this call's cursor starts where parent left off
; --- entry test PEELED above the loop label, zero-arg case : the loop no
; longer tests for exhaustion at the top. ---
     cmp dword [rsp + LG_XFN_PARAMS_OFF], 0    ; params exhausted?
     jz .xfn_args_done                         ; yes: argument list complete
.xfn_arg_loop:
     cmp dword [rsp + LG_XFN_ARGC_OFF], SL_XFN_MAX_ARGS ; more than 8 params declared?
     jge .xfn_err_max_args                     ; yes: FATAL, silent truncation desyncs fmt
     call EvalPrimary                          ; eax = primary value
     call EvalMulChain                         ; fold */% at higher precedence
     call EvalAddChain                         ; fold +- at higher precedence
; --- convert the tagged value into a raw v7sprintf vararg slot --------
     test eax, eax                             ; NIL, 0 ?
     jz .xfn_nil_arg                           ; yes -> empty-string pointer
     cmp eax, SL_FLOAT_TAG                     ; bits31:30 both set = FLOAT?
     jae .xfn_float_arg                        ; yes -> binary32 bits
     test eax, SL_CONS_TAG                     ; bit31 alone = CONS?
     jnz .xfn_err_type                         ; yes: lists are not argv values
     test eax, SL_NUMBER_TAG                   ; bit30 = NUMBER?
     jnz .xfn_int_arg                          ; yes -> sign-extended int
     XFN_PROBE                                 ; tagged value parked/restored, probe decision in r11d
     cmp r11d, -2                              ; probe said this conversion is not %s
     je .xfn_err_slot                          ; yes: a string slot demands a char*, abort naming it
     lea rcx, [r13 + rax + SL_ATOM_TEXT_OFF]   ; atom content, tag bits proven clear
     mov rdx, [rsp + LG_XFN_ESC_CURSOR]        ; dst = arena cursor, per-arg region
     lea r8, [rel sl_xfn_escape_buf + SL_XFN_ESCAPE_SIZE] ; r8 = arena end, one past the last writable byte
     sub r8, rdx                               ; r8 = capacity remaining for THIS arg
     jbe .xfn_err_escape                       ; no capacity left, cursor >= end : FATAL
     call win32_escape_inner                   ; eax = escaped length, or -1 ; r11d = XFN_PROBE's M context is a LIVE INPUT, do not overwrite r11 between the probe and this call
     test eax, eax                             ; truncation / arena exhausted?
     js .xfn_err_escape                        ; yes: FATAL
     mov rdx, [rsp + LG_XFN_ESC_CURSOR]        ; rdx = escaped string start, old cursor
     lea rcx, [rdx + rax + LG_XFN_ESC_NUL_ADV] ; rcx = new cursor, past NUL; rcx overwritten by the call anyway
     mov [rsp + LG_XFN_ESC_CURSOR], rcx        ; publish next free byte, stack frame
     mov [rel sl_xfn_arena_cursor], rcx        ; publish next free byte, BSS global for reentrancy
     mov rax, rdx                              ; rax = escaped string POINTER, this is what .xfn_store_arg stores
     jmp .xfn_store_arg                        ; store the slot
.xfn_nil_arg:
; NIL -> empty string: use the dedicated empty-string atom at SL_EMPTY_ATOM_OFF
     XFN_PROBE                                 ; tagged value parked/restored, probe decision in r11d
     cmp r11d, -2                              ; is the format's specifier numeric, %d/%f/%c ?
     je .xfn_err_slot                          ; yes: a NIL becomes a char*, empty string, which a numeric slot would read as an int32/float, NIL is %s-only, abort naming it
     lea rax, [r13 + SL_EMPTY_ATOM_OFF + SL_ATOM_TEXT_OFF] ; pointer to empty-string content, NUL terminator
     jmp .xfn_store_arg                        ; store the slot
.xfn_int_arg:
     XFN_PROBE                                 ; tagged value parked/restored, probe decision in r11d
     cmp r11d, -2                              ; is the format's specifier numeric, %d/%f/%c ?
     jne .xfn_err_slot                         ; no, it IS a %s : an integer in a string slot is dereferenced as a char*, abort naming it
     call NumVal                               ; eax = raw int32
     cdqe                                      ; sign-extend to 64 bits
     jmp .xfn_store_arg                        ; store the slot
.xfn_float_arg:
     XFN_PROBE                                 ; tagged value parked/restored, probe decision in r11d
     cmp r11d, -2                              ; is the format's specifier numeric, %d/%f/%c ?
     jne .xfn_err_slot                         ; no, it IS a %s : float bits in a string slot are dereferenced as a char* by v7sprintf, abort naming it
     shl eax, SL_FLOAT_SHIFT                   ; 32-bit left shift discards the two high tag bits
.xfn_store_arg:
     mov ecx, [rsp + LG_XFN_ARGC_OFF]          ; ecx = arg index
     mov [rsp + LG_XFN_ARGV_OFF + rcx*LG_XFN_QWORD], rax ; store the vararg slot
     inc dword [rsp + LG_XFN_ARGC_OFF]         ; count++
     mov eax, [rsp + LG_XFN_PARAMS_OFF]        ; eax = params cursor
     call Cdr                                  ; eax = rest of params
     mov [rsp + LG_XFN_PARAMS_OFF], eax        ; advance params cursor, leaves the advanced cursor in eax, no tail reload needed
     test eax, eax                             ; more params?, eax already holds the cursor
     jnz .xfn_arg_loop                         ; yes: evaluate the next argument, bottom-tested, no per-arg jmp
.xfn_args_done:
     mov rcx, [rsp + LG_XFN_EXE_OFF]           ; arg1: exe path, load BEFORE pushes
     mov rdx, [rsp + LG_XFN_FMT_OFF]           ; arg2: format string, load BEFORE pushes
     lea r8, [rsp + LG_XFN_ARGV_OFF]           ; arg3: argv array, pre-correction rsp
     mov r9d, [rsp + LG_XFN_ARGC_OFF]          ; arg4: argc, load BEFORE pushes
     CORRECT_RSP                               ; realign RSP for the outbound call
     sub rsp, 32                               ; reserve v7_execute_fmt's shadow AFTER the flag: space above CORRECT_RSP's push is what the callee may write
     call v7_execute_fmt                       ; rax = captured stdout+stderr C string
     add rsp, 32                               ; release the shadow reservation, flag back at [rsp]
     RESTORE_RSP                               ; drop the correction marker, flag is at [rsp] here
; --- intern the capture as the return value ---------------------------
; copy C string to concat_buf with type byte prefix for Intern
     test rax, rax                             ; NULL return from v7_execute_fmt?
     jz .xfn_empty_capture                     ; yes: return empty string
     mov byte [rel sl_concat_buf], SL_TYPE_STRING ; type prefix
     lea rdi, [rel sl_concat_buf + SL_ATOM_TEXT_OFF] ; dest for content
     mov rsi, rax                              ; source = C string from v7_execute_fmt
     lea r8, [rel sl_concat_buf + SL_CONCAT_BUF_LAST] ; last writable offset
.xfn_copy_loop:
     lodsb                                     ; al = [rsi++]
     test al, al                               ; NUL terminator?
     jz .xfn_copy_nul                          ; yes: store it outside the bound
     cmp rdi, r8                               ; room for this content byte?, the jae bound leaves SIZE-2 for content and SIZE-1 for the NUL at .xfn_copy_nul, same convention as .unp_copy
     jae .xfn_toolong                          ; no: FATAL before any store
     stosb                                     ; [rdi++] = al, in bounds: rdi < r8 proven above
     jmp .xfn_copy_loop                        ; keep copying
.xfn_copy_nul:
     mov byte [rdi], 0                         ; NUL-terminate, rdi <= r8 by the content bound
.xfn_copy_done:
     lea rsi, [rel sl_concat_buf]              ; source for Intern, type byte first
     call Intern                               ; eax = interned string atom
     jmp .xfn_epilogue                         ; skip empty-capture path
.xfn_toolong:
     FATAL "SectorLogo: DefineCommand capture too long" ; exceeds the 2 MiB concat buffer, content bytes are checked BEFORE every store, so no partial overrun is possible
.xfn_empty_capture:
; v7_execute_fmt returned NULL, intern the empty string
     lea rsi, [rel sl_concat_buf]              ; source for Intern
     mov byte [rel sl_concat_buf], SL_TYPE_STRING ; type prefix
     mov byte [rel sl_concat_buf + SL_ATOM_TEXT_OFF], 0 ; NUL content
     call Intern                               ; eax = empty-string atom
.xfn_epilogue:
     mov rcx, [rsp + LG_XFN_ESC_SAVED]         ; rcx = parent's arena cursor
     mov [rel sl_xfn_arena_cursor], rcx        ; restore global cursor, eax untouched
     add rsp, LG_XFN_SLOTS + LG_XFN_SHADOW     ; release frame, 5 pushes + 136 = 176: 16-NEUTRAL delta, matches the sub, preserves the inherited entry parity
     pop r15                                   ; restore non-volatile
     pop rdi                                   ; restore non-volatile
     pop r14                                   ; restore caller's env
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     dec dword [rel sl_call_depth]             ; decrement depth
     ret                                       ; eax = return atom
.xfn_err_type:
     FATAL_VALUE "DefineCommand argument must be atom, number, or float", eax ; cons list into argv: abort naming it
.xfn_err_slot:
     FATAL_VALUE "DefineCommand format slot does not accept this argument type (a string or NIL argument needs a %s slot, a number needs %d or %c, a float needs %f)", eax ; the value type is legal but the fmt's slot is not, name the slot conflict
.xfn_err_escape:
     FATAL "SectorLogo: DefineCommand argument escape overflow" ; 2 MiB scratch too small for a 1 MiB string: abort
.xfn_err_max_args:
     FATAL "SectorLogo: DefineCommand exceeds 8 arguments" ; more params than argv slots: abort
.xfn_err_arity:
     FATAL "SectorLogo: DefineCommand format has a different number of %-conversions than the function declares parameters (surplus conversions would spawn the child with uninitialised stack bytes as arguments)"
.xfn_depth_overflow:
     mov eax, [rel sl_call_depth]              ; depth at abort, raw count, not yet tagged
     bts eax, 30                               ; set the NUMBER tag bit for the fatal-value formatter
     FATAL_VALUE "recursion depth exceeded (infinite recursion? abort depth shown)", eax ; guardrail trip names the depth
%unmacro XFN_PROBE 0                           ; file-local macro, removed so a later include cannot redefine it silently
%undef SQF_QUOTE_OFF
%undef SQF_SPEC_OFF
%undef SL_XFN_MAX_ARGS
%undef LG_XFN_ESC_NUL_ADV
%undef LG_XFN_ESC_SAVED
%undef LG_XFN_ESC_CURSOR
%undef LG_XFN_QWORD
%undef LG_XFN_SHADOW
%undef LG_XFN_SLOTS
%undef LG_XFN_ARGV_OFF
%undef LG_XFN_PARAMS_OFF
%undef LG_XFN_ARGC_OFF
%undef LG_XFN_FMT_OFF
%undef LG_XFN_EXE_OFF
%undef LG_FN_MAX_ARGS
%undef LG_FN_ARG_VAL_SCALE
%undef LG_FN_ARG_VALS_OFF
%undef LG_FN_ARG_COUNT
%undef LG_FN_SAVED_PARAMS
%undef LG_FN_SAVED_LOCAL_COUNT
%undef LG_FN_STACK_PARAMS
%undef LG_FN_STACK_BODY
%undef LG_FN_SAVED_RET_VAL
%undef LG_FN_SAVED_HAS_RET
