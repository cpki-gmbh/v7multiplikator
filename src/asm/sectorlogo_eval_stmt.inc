; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; STMT_DELEGATE callee, a statement handler that delegates to callee and
; continues the stream; the callee's return value is discarded.

%macro STMT_DELEGATE 1
     call %1                                   ; invoke the delegated builtin, evaluates its args; return value discarded
     jmp .stream_loop                          ; continue statement stream
%endmacro

%macro NAME_CLEAN_FOLD 1
     ASSERT_ATOM_NONNIL eax, %1                ; FATAL: name must be a plain non-NIL atom
     call lg_atom_unprefix                     ; eax = clean name atom, no ':' prefix returns as-is
     call lg_atom_lowercase                    ; eax = lowercase-folded atom
     call .lg_validate_name                    ; FATAL if the name contains non-alphanumeric chars
%endmacro

%macro DEF_NAME_FETCH 1
     mov eax, r12d                             ; eax = stream
     call PopCons                              ; ebx = raw name atom, eax = rest
     mov r12d, eax                             ; advance stream past name
     mov r15d, ebx                             ; r15d = raw name, published BEFORE the guard, because .defn_notatom / .xdefn_name_notatom FATAL_VALUE reads r15d and must name the OFFENDING value, not a stale register
     mov eax, ebx                              ; eax = raw name atom, NAME_CLEAN_FOLD contract
     NAME_CLEAN_FOLD %1                        ; FATAL on tagged/NIL/bad chars; eax = clean folded name
     mov r15d, eax                             ; r15d = clean folded name, DEF_BIND_TAIL's env search key
%endmacro

%macro DEF_PARAMS_PARK 0
     mov eax, r12d                             ; eax = stream
     call PopCons                              ; ebx = raw params, eax = rest
     mov r12d, eax                             ; advance stream past params
     mov eax, ebx                              ; eax = raw params
     call lg_clean_param_list                  ; eax = cleaned params
     push rax                                  ; park cleaned params, PopCons overwrites ebx below
     push rax                                  ; shadow copy: one push parks the value, its twin balances PopCons's pop so the value survives to the bind
%endmacro

%macro DEF_BIND_TAIL 0
     mov r8d, eax                              ; r8d = NEW descriptor, consumed at the insert, before any Cons call
     push rax                                  ; park the sentinel's tagged cell, 0 = not passed yet
     push rax                                  ; shadow copy: one push parks the sentinel, its twin balances PopCons's pop so the sentinel survives to the bind
     mov dword [rsp], 0                        ; sentinel park = not seen yet
     mov ebx, [rel sl_lg_base_env]             ; ebx = base chain head, tagged; may be the sentinel or, autoexec-makes.sentinel
%%bind_walk:
     test ebx, ebx                             ; end of chain?
     jz %%bind_insert                          ; yes: name is bound nowhere, insert at the sentinel
     mov ecx, ebx                              ; ecx = tagged chain cell
     btr ecx, SL_CONS_BIT                      ; ecx = raw cell index
     lea rcx, [r13 + rcx*CONS_SCALE + CONS_BASE_OFF] ; rcx = chain cell address
     mov edx, [rcx]                            ; edx = tagged inner pair, name.old_value
     btr edx, SL_CONS_BIT                      ; edx = raw inner-pair index
     mov r9d, [r13 + rdx*CONS_SCALE + CONS_BASE_OFF] ; r9d = car,inner pair = bound name atom
     test r9d, r9d                             ; NIL name = the sentinel cell itself, its inner pair is, NIL.NIL ?
     jnz %%bind_cmp                            ; real name: compare it
     mov [rsp], ebx                            ; remember the sentinel's tagged cell, the insert point
     jmp %%bind_next                           ; keep walking: the name may still sit in the stdlib chain below
%%bind_cmp:
     cmp r9d, r15d                             ; this cell binds the target name?
     je %%bind_found                           ; yes: update in place, nearest binding wins, so a makes-region hit shadows the stdlib one
%%bind_next:
     mov ebx, [rcx + CONS_CDR_OFF]             ; ebx = rest of the chain
     jmp %%bind_walk                           ; continue, conditional back edge via the test above
%%bind_insert:
     mov edx, r8d                              ; edx = descriptor
     mov eax, r15d                             ; eax = clean name atom, Cons preserves r15d
     call Cons                                 ; eax =, name.descriptor binding pair
     mov ebx, eax                              ; rbx = binding cell, parked; Cons preserves rbx
     mov ecx, [rsp]                            ; ecx = tagged sentinel cell, parked when the walk passed it, NOT the walk-end cell: the old chain lives in the SENTINEL's cdr
     btr ecx, SL_CONS_BIT                      ; ecx = raw sentinel index
     lea rcx, [r13 + rcx*CONS_SCALE + CONS_BASE_OFF] ; rcx = sentinel cell address
     mov edx, [rcx + CONS_CDR_OFF]             ; edx = the sentinel's CURRENT chain, the stdlib head, must be preserved below the new binding
     mov eax, ebx                              ; eax = binding cell, outer car
     call Cons                                 ; eax =, binding.old_sentinel_chain, new chain head
     mov ecx, [rsp]                            ; ecx = tagged sentinel, reloaded, Cons overwritten rcx
     btr ecx, SL_CONS_BIT                      ; ecx = raw sentinel index
     lea rcx, [r13 + rcx*CONS_SCALE + CONS_BASE_OFF] ; rcx = sentinel cell address again
     mov [rcx + CONS_CDR_OFF], eax             ; sentinel cdr = new chain: env chains captured BEFORE this Define terminate IN the sentinel and see the binding through it
     add rsp, 16                               ; drop the park pair
     jmp .stream_loop                          ; continue stream evaluation
%%bind_found:
     lea r9, [r13 + rdx*CONS_SCALE + CONS_BASE_OFF] ; r9 = inner pair cell address, rdx kept raw by the walk body
     mov [r9 + CONS_CDR_OFF], r8d              ; CDR = new descriptor, update in place: repeated Define does not grow the chain
     add rsp, 16                               ; drop the park pair
     jmp .stream_loop                          ; continue stream evaluation
%endmacro
; --- CallLibrary, statement position: invoke FFI, discard return value ---
.exec_c_call_st:
     STMT_DELEGATE sl_ffi_invoke               ; invoke CallLibrary, evaluates args + calls C; return value discarded
.exec_setcar_st:
     STMT_DELEGATE sl_setcar                   ; destructive car-field write, returns NIL, discarded
.exec_setcdr_st:
     STMT_DELEGATE sl_setcdr                   ; destructive cdr-field write, returns NIL, discarded
.exec_samevalue_st:
     STMT_DELEGATE sl_samevalue                ; identity compare, returns TRUE/NIL, discarded
; --- AsWord, statement position: conversion, result discarded ---
; Both dispatch positions registered, for the same reason as SameValue above.
.exec_asword_st:
     STMT_DELEGATE sl_asword                   ; STRING -> folded WORD conversion, result discarded
.lg_validate_name:
     lea r10, [r13 + rax + SL_ATOM_TEXT_OFF]   ; r10 = atom string bytes, VOLATILE scratch, rsi/r11 are Win64 non-volatile and must not be written here
     lea r9, [rel sl_tok_class]                ; shared classification table, VOLATILE; bit6 means ASCII alphanumeric
.vname_loop:
     movzx ecx, byte [r10]                     ; cl = current character
     test cl, cl                               ; NUL terminator?
     jz .vname_ok                              ; yes: all chars valid
     test byte [r9 + rcx], SL_TOK_ALNUM_MASK   ; alphanumeric flag set?
     jz .vname_bad                             ; no: invalid name character
     inc r10                                   ; advance to next character
     jmp .vname_loop                           ; continue
.vname_ok:
     ret                                       ; all chars valid
.vname_bad:
     FATAL_VALUE "names may only contain alphanumeric characters (A-Za-z0-9)", eax ; non-alphanumeric char: name the offender
; --- MAKE "var expr ---
.exec_make:
; get variable name from stream, fused Car+Cdr
     mov eax, r12d                             ; eax = stream
     call PopCons                              ; ebx = raw variable name atom, eax = rest
     mov r15d, ebx                             ; r15d = raw var name
     mov r12d, eax                             ; advance stream
     mov eax, r15d                             ; eax = raw var name atom, macro contract
     NAME_CLEAN_FOLD .make_notatom             ; FATAL on tagged/NIL; strip ':'; lowercase; reject bad chars
     mov r15d, eax                             ; r15d = clean folded var name
.make_eval_val:
; evaluate the value expression
     mov ebx, r15d                             ; park var name, EvalNextArg overwrites ebx
     call EvalNextArg                          ; eax = evaluated value
     mov r9d, eax                              ; r9d = value
     mov r15d, ebx                             ; restore var name
     mov ecx, [rel sl_local_count]             ; ecx = LOCAL set entry count
     test ecx, ecx                             ; any LOCAL declarations active?
     jz .make_env_walk                         ; no LOCAL: walk env chain for existing binding
     lea rax, [rel sl_local_set]               ; rax = set base, RIP-rel has no index form
     lea edx, [ecx - SL_LOCAL_NEWEST_ADJ]      ; edx = scan index = count-1, newest first
     LOCAL_SET_SCAN .make_local_found, .make_env_walk, r15d ; descending LOCAL-set scan; miss -> env walk
.make_env_walk:
; walk env chain to find existing binding for r15d before prepending
; this prevents MAKE inside a function from shadowing outer-scope variables
     mov r8d, r9d                              ; r8d = new value, unify register with DEFN/XDEFN walk
     ENV_UPDATE_WALK .stream_loop              ; walk env, find, name.old, set CDR to r8d, both exits jump .stream_loop
.make_local_found:
     mov eax, [rax + rdx*SL_LOCAL_SET_SCALE + SL_LOCAL_CELL_OFF] ; eax = cached cell index, mov zero-extends into rax
     mov dword [r13 + rax*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF], r9d ; binding value = new value
     jmp .stream_loop                          ; continue stream evaluation
.make_notatom:
     FATAL_VALUE "MAKE requires a string atom as variable name", r15d ; names the offending var name
.defn_notatom:
     FATAL_VALUE "Define requires a string atom as function name", r15d ; names the offending function name
.exec_local:
     mov eax, r12d                             ; eax = stream
     call PopCons                              ; ebx = raw name atom, eax = rest, no evaluation, builtins like "length" must NOT dispatch
     mov r12d, eax                             ; advance stream
     mov eax, ebx                              ; eax = raw name atom
.local_have_name:
     NAME_CLEAN_FOLD .local_notatom            ; FATAL on tagged/NIL; strip ':'; lowercase; reject bad chars, eax = folded name for the store below
; record in the LOCAL set, BSS array of [var atom][binding cell index] pairs
     mov ecx, [rel sl_local_count]             ; ecx = current entry count
     cmp ecx, SL_LOCAL_SET_MAX                 ; over the limit?
     jae .local_overflow                       ; yes: FATAL
     lea r11, [rel sl_local_set]               ; r11 = set base, RIP-rel has no index form
     mov [r11 + rcx*SL_LOCAL_SET_SCALE], eax   ; set[count].var = clean name atom
; prepend initial, var.NIL binding; cache the pair's cell index in the entry
     push rcx                                  ; save entry index across Cons, Cons overwrites ecx
     push r11                                  ; save set base across Cons, Cons overwrites r10/r11 + parity pad, 2 pushes = 16B
     xor edx, edx                              ; edx = initial value = NIL, xor: 2 bytes vs 5 for mov edx,0
     call Cons                                 ; eax =, var.NIL pair, TAGGED
     pop r11                                   ; restore set base
     pop rcx                                   ; restore entry index
     btr eax, SL_CONS_BIT                      ; strip the known CONS tag only
     mov [r11 + rcx*SL_LOCAL_SET_SCALE + SL_LOCAL_CELL_OFF], eax ; save raw cell index
     bts eax, SL_CONS_BIT                      ; retag before the environment Cons call
     inc dword [rel sl_local_count]            ; count++, entry visible NOW, both dwords committed
     mov edx, r14d                             ; edx = old env
     call Cons                                 ; eax =, ,var.NIL.env
     mov r14d, eax                             ; update env
     mov eax, r12d                             ; eax = stream, peek source
     call Car                                  ; eax = raw next form
     test eax, eax                             ; NIL -> name list exhausted
     jz .stream_loop                           ; done: continue stream evaluation
     test eax, SL_TAG_MASK                     ; isolate tag bits, clear = plain atom, eax itself must survive
     jnz .stream_loop                          ; NUMBER/FLOAT/CONS follows -> name list done
     cmp eax, [rel sl_dispatch_highwater]      ; within the builtin region?
     jae .local_peek_colon                     ; no: user atom far past builtins -> a name
     bt dword [rel sl_builtin_bitmap], eax     ; TRUE registered builtin keyword?, bit index = offset
     jc .stream_loop                           ; yes: keyword, make/output/ifelse/... -> done
     bt dword [rel sl_op_atom_bitmap], eax     ; infix OPERATOR atom, _ + - * / = ... ?, full-array bit index; eax < highwater <= dispatch max here
     jc .stream_loop                           ; yes: an expression follows, Local "x" _ 5 -> x is "_"-concatenated -> name list done, never a name
.local_peek_colon:
     mov r10b, byte [r13 + rax + SL_ATOM_TEXT_OFF] ; prefix byte; the next CMP reads R10B only
     cmp r10b, ':'                             ; ':'-prefixed variable reference?
     je .stream_loop                           ; yes: a deref form, never a name -> done
     mov edx, r14d                             ; edx = current env, eax still = candidate atom
     call Assoc                                ; eax = bound value or NIL
     test eax, eax                             ; bound at all?
     jz .local_peek_name                       ; no: plain name
     mov ecx, eax                              ; ecx = copy for tag isolation
     shr ecx, 30                               ; ecx = 2-bit tag code (CONS = 2)
     cmp ecx, 2                                ; exactly CONS, not FLOAT?
     jne .local_peek_name                      ; no: scalar binding, still a name
     call Car                                  ; eax = car of the binding, eax = the pair Assoc returned
     cmp eax, [rel atom_sl_fn_tag]             ; __FN__, user function ?
     je .stream_loop                           ; yes: call head follows -> name list done
     cmp eax, [rel atom_sl_xfn_tag]            ; __XF__, exe function ?
     je .stream_loop                           ; yes: call head follows -> name list done
.local_peek_name:
; consume the raw name form WITHOUT evaluation: a raw atom IS its own name
     mov eax, r12d                             ; eax = stream
     call PopCons                              ; ebx = name atom, eax = rest of stream
     mov r12d, eax                             ; advance stream past the name
     mov eax, ebx                              ; eax = raw name atom
     jmp .local_have_name                      ; register it, then peek for yet another name
.local_notatom:
     FATAL_VALUE "LOCAL requires a string atom variable name", eax ; names the offending var name
.local_overflow:
     FATAL_VALUE "LOCAL declaration limit exceeded (scope stack full, or leak/deep recursion?)", eax ; scope stack exhausted, names the var being declared
; --- PRINTRAW expr, PRINT is a Logo stdlib fn; it adds the CRLF ---
.exec_printraw:
     call EvalNextArg                          ; eax = evaluated value
     call lg_print_value                       ; print the value, no CRLF
     jmp .stream_loop                          ; continue stream evaluation
; --- REPEAT [count] [ body ], count optional: missing count = endless loop ---
.exec_repeat:
; peek at the next form WITHOUT consuming it: a CONS head IS the body
; block -> no count form -> endless mode, REPEAT [ body ]
     mov eax, r12d                             ; eax = stream, peek source
     call Car                                  ; eax = next form, tagged
     mov ecx, eax                              ; ecx = copy for tag isolation
     shr ecx, 30                               ; ecx = 2-bit tag code (CONS = 2)
     cmp ecx, 2                                ; exactly CONS, list literal?
     je .repeat_endless                        ; yes: endless loop, body is the next form
     call EvalNextArg                          ; no: evaluate the count expression
     call NumVal                               ; eax = raw count, FATALs on FLOAT/atom/NIL
     movsxd r15, eax                           ; r15 = 64-bit trip counter, MUST sign-extend: NumVal writes only eax, and a zero-extended negative count would run ~2^64 trips instead of zero
     jmp .repeat_body_fetch                    ; both modes converge on the body fetch
.repeat_endless:
     mov r15, 0x7FFFFFFFFFFFFFFF               ; endless sentinel = +2^63-1: POSITIVE so the entry test admits it, and never reaches 0 so the back edge never counts out
.repeat_body_fetch:
; get body block from stream, fused Car+Cdr; body parks on the stack
     mov eax, r12d                             ; eax = stream
     call PopCons                              ; ebx = body block, eax = rest
     push rbx                                  ; save body on stack, not volatile r9d!
     mov r12d, eax                             ; advance stream past body
; entry test: the endless sentinel is POSITIVE so it enters; counted exits when count <= 0
     test r15, r15                             ; count <= 0?, REPEAT 0/-n = zero trips
     jle .repeat_done                          ; yes: skip body entirely
     mov qword [rel sl_stop_requested], 0      ; fresh loop: clear stop+continue, one qword store covers both contiguous dwords
.repeat_body:
     SL_EVAL_BLOCK 1, ebx                      ; stream = ebx, ebx still holds the body on every entry: EvalLogoStream pushes/pops rbx and sl_tombstone_locals never writes it, so it survives the macro on the back edge too, no reload needed; r15 likewise
; check for Exit/Output, propagate upward through all blocks
     cmp dword [rel sl_fn_has_returned], 1     ; Exit or Output called?
     je .repeat_early_exit                     ; yes: propagate upward
     mov rax, [rel sl_stop_requested]          ; 8-byte: stop |, continue << 32
     mov ecx, [rel sl_terminate_requested]     ; ecx is 32-bit -> m32, reads EXACTLY 4 bytes
     or rax, rcx                               ; ZF = any of stop/continue/terminate set
     jz .repeat_backedge                       ; none set: straight to the back edge
; out of line: at least one of stop / continue / terminate is pending
.repeat_flag_hit:
     cmp dword [rel sl_stop_requested], 1      ; STOP called in body?
     je .repeat_stop                           ; yes: break this loop
     cmp dword [rel sl_continue_requested], 1  ; CONTINUE called in body?
     je .repeat_continue                       ; yes: next iteration
     jmp .repeat_early_exit                    ; only TERMINATE remains, the fn boundary restored has_returned, so only the dedicated flag survives
.repeat_backedge:
; back edge: one trip consumed per pass; the endless sentinel decrements harmlessly and never reaches 0
     dec r15                                   ; counted: consume one iteration
     jg .repeat_body                           ; trips remain: loop, signed, the endless sentinel stays positive forever
; counted counter exhausted: fall through to done, zero jumps on the common exit
.repeat_done:
     add rsp, 8                                ; discard saved body from stack
     jmp .stream_loop                          ; continue stream evaluation
.repeat_stop:
     mov dword [rel sl_stop_requested], 0      ; clear: we caught the STOP
     jmp .repeat_done                          ; break out of this loop
.repeat_continue:
     mov dword [rel sl_continue_requested], 0  ; clear: we caught the CONTINUE
     jmp .repeat_backedge                      ; next iteration, consumes a trip when counted
.repeat_early_exit:
     add rsp, 8                                ; discard saved body from stack
     jmp .stream_done                          ; unwind to stream_done
; --- IF cond [then-block] ---
.exec_if:
     call EvalNextArg                          ; eax = condition value
     mov r15d, eax                             ; r15d = condition, nonzero = true
; get then-block from stream, fused Car+Cdr
     mov eax, r12d                             ; eax = stream
     call PopCons                              ; ebx = then-block, eax = rest
     mov r12d, eax                             ; advance stream past then-block
     SL_FALSY_JUMP r15d, .if_skip              ; falsy, NIL/0/0.0/FALSE : skip then-block
; execute then-block straight from ebx, PopCons's car output, safe: SL_FALSY_JUMP
; writes flags only and rbx is re-derived by the dispatch core each iteration
     SL_EVAL_BLOCK 0, ebx                      ; no trace indent inside a bare IF body, ebx: the macro's mov esi,%2 needs a 32-bit source
.if_skip:
     jmp .stream_loop                          ; continue stream evaluation
; --- IFELSE cond [then-block] [else-block] ---
.exec_ifelse:
     call EvalNextArg                          ; eax = condition value
     mov r15d, eax                             ; r15d = condition
; get then-block from stream, fused Car+Cdr
     mov eax, r12d                             ; eax = stream
     call PopCons                              ; ebx = then-block, eax = rest, contains else-block
     push rbx                                  ; save then-block on stack, not volatile r9d!
; get else-block from stream, fused Car+Cdr on rest, eax already holds
; `rest` from the PopCons above, which returns the cdr in eax
     call PopCons                              ; ebx = else-block, eax = advance
     mov r9d, ebx                              ; r9d = else-block, PARKED IN A REGISTER, not a stack slot:
     mov r12d, eax                             ; advance stream past else-block
; branch on condition
     SL_FALSY_JUMP r15d, .ifelse_false         ; falsy: execute else-block
; execute then-block, else-block in r9d is discarded, no pop needed
     SL_EVAL_BLOCK 0, [rsp]                    ; no trace indent inside an IF/ELSE body
     add rsp, 8                                ; discard saved then-block
     jmp .stream_loop                          ; continue stream evaluation
.ifelse_false:
     add rsp, 8                                ; discard saved then-block
     SL_EVAL_BLOCK 0, r9d                      ; stream = else-block, parked in r9d, not a stack slot
     jmp .stream_loop                          ; continue stream evaluation
; --- STOP, break current REPEAT loop ---
.exec_stop:
     mov dword [rel sl_stop_requested], 1      ; request STOP, REPEAT will catch it
     jmp .stream_done                          ; stop evaluating this stream
; --- CONTINUE, skip to next REPEAT iteration ---
.exec_continue:
     mov dword [rel sl_continue_requested], 1  ; request CONTINUE, REPEAT will catch it
     jmp .stream_done                          ; stop evaluating this stream
; --- Exit, exit entire program ---
.exec_terminate:
     mov dword [rel sl_fn_return_val], 0       ; clear any pending OUTPUT value, Exit discards it
     mov dword [rel sl_fn_has_returned], 1     ; flag Exit to outer blocks, fast unwind of THIS stream
     mov dword [rel sl_terminate_requested], 1 ; dedicated script-wide flag: fn boundaries restore has_returned, but NEVER this, Exit must exit everything
     jmp .stream_done                          ; stop evaluating this stream
; --- Run block ---
.exec_run:
     call EvalNextArg                          ; eax = block, evaluated
; if the block is a cons list, execute it as a stream
     mov ecx, eax                              ; copy for tag isolation
     shr ecx, 30                               ; ecx = 2-bit tag code (CONS = 2)
     cmp ecx, 2                                ; exactly CONS, not FLOAT?
     je .exec_run_cons                         ; yes: execute as stream
     test eax, eax                             ; NIL?
     jz .stream_loop                           ; NIL: skip silently
     jmp .exec_run_fatal                       ; NUMBER/FLOAT: contract violation
.exec_run_cons:
     SL_EVAL_BLOCK 1, eax                      ; stream = block, with trace indent
     jmp .stream_loop                          ; continue stream evaluation
.exec_run_fatal:
     FATAL_VALUE "RUN requires a list or NIL", eax ; NUMBER/FLOAT block: names the offender
; --- Define "name [params] [body] ---
.exec_def_fn:
     DEF_NAME_FETCH .defn_notatom              ; r15d = clean folded name, FATAL on tagged/NIL/bad chars
     DEF_PARAMS_PARK                           ; params cleaned + parked on the stack
; get body block from stream, fused Car+Cdr
     mov eax, r12d                             ; eax = stream
     call PopCons                              ; ebx = body block, eax = rest
     mov r12d, eax                             ; advance stream past body
; construct descriptor:, __FN__ ., params.body
     pop rax                                   ; discard park pad, top copy
     pop rax                                   ; eax = cleaned params, car, from the stack park, popped straight into the car slot
     mov edx, ebx                              ; edx = body, cdr, ebx is untouched by the two pops above
     call Cons                                 ; eax =, params.body
     mov edx, eax                              ; edx =, params.body = inner
     mov eax, [rel atom_sl_fn_tag]             ; eax = __FN__ atom, car
     call Cons                                 ; eax =, __FN__ ., params.body
     DEF_BIND_TAIL                             ; bind, name.descriptor in env, update in place if bound
; --- Output expr ---
.exec_output:
     call EvalNextArg                          ; eax = return value
     mov dword [rel sl_fn_return_val], eax     ; store return value
     mov dword [rel sl_fn_has_returned], 1     ; set early-return flag
     jmp .stream_done                          ; stop evaluating this stream
.exec_def_exe_fn:
     DEF_NAME_FETCH .xdefn_name_notatom        ; r15d = clean folded name, FATAL on tagged/NIL/bad chars
     DEF_PARAMS_PARK                           ; params cleaned + parked on the stack, documents call arity
; exe path atom, must be a plain string atom
     mov eax, r12d                             ; eax = stream
     call PopCons                              ; ebx = exe atom, eax = rest
     mov ebp, ebx                              ; ebp = exe atom, non-volatile scratch
     mov r12d, eax                             ; advance stream past exe
     ASSERT_ATOM_NONNIL rbp, .xdefn_exe_notatom ; FATAL: exe must be a plain atom, ebp zero-extended by the 32-bit mov
; format string atom, must be a plain string atom ; held in edx
; PopCons writes eax+ebx only, edx survives
     mov eax, r12d                             ; eax = stream
     call PopCons                              ; ebx = fmt atom, eax = rest
     mov r12d, eax                             ; advance stream past fmt
     xchg edx, ebx                             ; edx = fmt atom, the car PopCons returned, ebx = junk
     ASSERT_ATOM_NONNIL rdx, .xdefn_fmt_notatom ; FATAL: fmt must be a plain atom, edx zero-extended by the 32-bit xchg
; construct descriptor:, __XF__ ., params ., exe.fmt
     mov eax, ebp                              ; eax = exe atom, car
     call Cons                                 ; eax =, exe.fmt
     mov edx, eax                              ; edx =, exe.fmt
     pop rax                                   ; discard park pad, top copy
     pop rax                                   ; eax = cleaned params, car, from the stack park, popped straight into the car slot
     call Cons                                 ; eax =, params ., exe.fmt
     mov edx, eax                              ; edx = inner pair, cdr
     mov eax, [rel atom_sl_xfn_tag]            ; eax = __XF__ tag atom, car
     call Cons                                 ; eax =, __XF__.inner
     DEF_BIND_TAIL                             ; bind, name.descriptor in env, update in place if bound
.xdefn_name_notatom:
     FATAL_VALUE "DefineCommand requires a string atom as function name", r15d ; name the offending field and value
.xdefn_exe_notatom:
     FATAL_VALUE "DefineCommand requires a string atom as exe path", ebp ; ditto
.xdefn_fmt_notatom:
     FATAL_VALUE "DefineCommand requires a string atom as format string", edx ; ditto
; --- SetTemperature expr, statement ---
.exec_v7_temp:
     STMT_DELEGATE sl_v7_temp_stmt             ; delegate to v7prim
; --- SetNucleus expr, statement ---
.exec_v7_topp:
     STMT_DELEGATE sl_v7_topp_stmt             ; delegate to v7prim
; --- DatabaseSet key value, statement ---
.exec_v7_db_put:
     call EvalNextArg                          ; eax = key atom
     STMT_DELEGATE sl_v7_db_put                ; delegate to v7prim, it evaluates the value arg itself, so only the key is fetched above
; --- SetStream "new_text", statement ---
.exec_v7_set_stream:
     STMT_DELEGATE sl_v7_set_stream            ; delegate to v7prim
; --- SetOutputToken id, statement ---
.exec_v7_set_token:
     STMT_DELEGATE sl_v7_set_token             ; delegate to v7prim
; --- Reject, statement ---
.exec_v7_reject:
     STMT_DELEGATE sl_v7_reject                ; delegate to v7prim
; --- Halt, statement ---
.exec_v7_halt:
     STMT_DELEGATE sl_v7_halt                  ; delegate to v7prim
; --- ENABLETRACE level, statement ---
.exec_enabletrace:
     call EvalNextArg                          ; eax = trace level, tagged
     mov r10d, eax                             ; park tagged level, FATAL_VALUE input; NumVal overwrites eax
     call NumVal                               ; eax = raw int
     cmp eax, SL_TRACE_MAX_LEVEL               ; unsigned compare: also catches every negative, its unsigned image is >= 0x80000000 > MAX_LEVEL
     ja .et_bad_level                          ; out of range: name the offender
     mov [rel sl_trace_level], eax             ; in range: set verbatim
     jmp .stream_loop                          ; continue stream evaluation
.et_bad_level:
     FATAL_VALUE "EnableTrace level must be 0..3", r10d ; name the offender: renders got number '99'
; --- ENABLESTRICTNAMES flag, statement ---
.exec_enablestrictnames:
     call EvalNextArg                          ; eax = flag, tagged
     mov r10d, eax                             ; park tagged flag, FATAL_VALUE input; NumVal overwrites eax
     call NumVal                               ; eax = raw 0/1
     cmp eax, 1                                ; unsigned <= 1, a negative flags this too ?
     ja .esn_bad                               ; not 0/1: name the offender
     mov [rel sl_strict_names], eax            ; 0 or 1: store verbatim
     jmp .stream_loop                          ; continue stream evaluation
.esn_bad:
     FATAL_VALUE "EnableStrictNames flag must be 0 or 1", r10d ; name the offender
; --- REALRANDOM flag, statement ---
.exec_realrandom:
     call EvalNextArg                          ; eax = flag, tagged
     mov r10d, eax                             ; park tagged flag, FATAL_VALUE input; NumVal overwrites eax
     call NumVal                               ; eax = raw 0/1
     cmp eax, 1                                ; unsigned <= 1, a negative flags this too ?
     ja .rr_bad                                ; not 0/1: name the offender
     mov [rel sl_real_random], eax             ; 0 or 1: store verbatim, Random mixes RDTSC when 1
     jmp .stream_loop                          ; continue stream evaluation
.rr_bad:
     FATAL_VALUE "RealRandom flag must be 0 or 1", r10d ; name the offender
%unmacro STMT_DELEGATE 1                       ; remove: the twelve .exec_* delegation sites above are its only expansions
%unmacro NAME_CLEAN_FOLD 1                     ; remove: DEF_NAME_FETCH / .exec_make / .exec_local are its only expansions
%unmacro DEF_NAME_FETCH 1                      ; remove: .exec_def_fn / .exec_def_exe_fn are its only expansions
%unmacro DEF_PARAMS_PARK 0                     ; remove: .exec_def_fn / .exec_def_exe_fn are its only expansions
%unmacro DEF_BIND_TAIL 0                       ; remove: .exec_def_fn / .exec_def_exe_fn are its only expansions
