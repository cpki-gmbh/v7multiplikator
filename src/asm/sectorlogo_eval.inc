; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

sl_tombstone_locals:
     push r9                                   ; save Logo-world scratch r9, Win64-VOLATILE, but callers hold live values per the internal convention
     push r8                                   ; save Logo-world scratch r8, Win64-VOLATILE; holds the set base below
     push rcx                                  ; park old_local_count, the walk below overwrites ecx
     mov r9d, [rel sl_local_count]             ; r9d = current count, end of range
     cmp ecx, r9d                              ; old >= current?, no new entries
     jae .tsl_done                             ; no entries to tombstone: skip
     lea r8, [rel sl_local_set]                ; r8 = set base
.tsl_loop:
     mov eax, [r8 + rcx*SL_LOCAL_SET_SCALE + SL_LOCAL_CELL_OFF] ; eax = cell index of this LOCAL entry
; set car of the binding cell to NIL, tombstone : Assoc never matches NIL
     mov dword [r13 + rax*CONS_SCALE + CONS_BASE_OFF], 0 ; car,cell = NIL
     inc ecx                                   ; next entry, only the scratch copy is overwritten
     cmp ecx, r9d                              ; past the end?
     jb .tsl_loop                              ; no: continue
.tsl_done:
     pop rcx                                   ; ecx = old_local_count, what the caller parked
     mov [rel sl_local_count], ecx             ; restore count to old value
     pop r8                                    ; restore Logo-world scratch r8
     pop r9                                    ; restore Logo-world scratch r9
     ret                                       ; return to caller

EvalLogoStream:
     push rbx                                  ; save non-volatile
     push r12                                  ; save non-volatile, stream
     push r14                                  ; save non-volatile, env
     push r15                                  ; save non-volatile, scratch
     push rbp                                  ; save non-volatile rbp, also completes 16-byte alignment: 5 pushes + retaddr = 48
     mov r12d, esi                             ; r12d = stream, cons list
     mov r14d, edx                             ; r14d = env, alist
; r13 is already the pinned heap base, NEVER touch it here
     cmp dword [rel sl_call_depth], 0          ; are we at the top level?
     jne .stream_loop                          ; no: inside a fn call, don't clear
     mov qword [rel sl_stop_requested], 0      ; top level: clear stop+continue flags, one qword store covers both contiguous dwords
.stream_loop:
     test r12d, r12d                           ; stream is NIL?
     jz .stream_done                           ; yes: done evaluating
     mov rax, [rel sl_stop_requested]          ; stop | continue<<32
     or rax, [rel sl_terminate_requested]      ; | terminate | fn_has_returned<<32, contiguous dword pairs, one qword load covers both
     jnz .stream_done                          ; any flag nonzero -> unwind immediately, ZF from here decides ; all zero -> keep evaluating
; extract current form head, fused Car+Cdr
     mov eax, r12d                             ; eax = stream
     call PopCons                              ; ebx = form head, eax = rest
     mov r12d, eax                             ; r12d = rest of stream
; --- trace: statement dispatch, level >= 1 ---
     cmp dword [rel sl_trace_level], 0         ; tracing on?
     jne .stmt_trace                           ; yes: out-of-line trace block, cold
.no_trace_stmt:
     cmp ebx, [rel sl_dispatch_highwater]      ; builtin region?
     jae .stmt_fallback                        ; no: user fn / bare atom / variable path
.stmt_dispatch_hit:
     lea rax, [rel sl_dispatch_stmt_table]     ; table base in a register, RIP+index is illegal x64
     jmp [rax + rbx]                           ; the aligned atom byte offset indexes the statement table
.stmt_trace:
     push rbx                                  ; save form head
     push r12                                  ; save stream
     call sl_trace_print_indent                ; print indentation spaces
     mov al, '|'                               ; trace line prefix
     call PutChar                              ; write '|'
     mov al, ' '                               ; space after pipe
     call PutChar                              ; write space
     mov eax, ebx                              ; eax = form head atom
     call lg_print_value                       ; print atom name
     mov al, 10                                ; newline
     call PutChar                              ; write newline
     pop r12                                   ; restore stream
     pop rbx                                   ; restore form head
     cmp ebx, [rel sl_dispatch_highwater]      ; rejoin: builtin region?
     jb .stmt_dispatch_hit                     ; yes: dispatch, unsigned complement of the old jae, misses fall through to the fallback
.stmt_fallback:
     test ebx, ebx                             ; NIL statement head, bare [] element ? inert data, not a statement
     jz .stream_loop                           ; Skip: a NIL key can match an invalidated LOCAL pair in `Assoc` and restore a discarded binding
     mov eax, ebx                              ; eax = atom to look up
     call lg_ensure_word                       ; convert STRING->WORD if needed
     cmp eax, ebx                              ; did lg_ensure_word change the atom?, ebx is non-volatile and lg_ensure_word preserves it, so it still holds the original
     je .stmt_nodispatch                       ; unchanged: already a WORD, do NOT re-dispatch, fall into the Assoc lookup
; re-check dispatch range after conversion, STRING "repeat" -> WORD "repeat"
     cmp eax, [rel sl_dispatch_highwater]      ; WORD version is a builtin?
     jae .stmt_word_not_builtin                ; no: not a builtin, try the user-fn lookup
     mov ebx, eax                              ; ebx = WORD builtin atom for dispatch
     jmp .stmt_dispatch_hit                    ; dispatch on the WORD atom
.stmt_word_not_builtin:
     mov ebx, eax                              ; ebx = folded WORD atom
.stmt_nodispatch:
     mov edx, r14d                             ; edx = env
     call Assoc                                ; eax = bound value or NIL, key already in eax
     test eax, eax                             ; NIL = not bound?
     jz .stmt_warn_miss                        ; not bound: warn once per word, then skip, the warning below; the bare-atom skip follows
; check if it's a cons with __FN__ tag
     mov ecx, eax                              ; ecx = bound value
     shr ecx, 30                               ; ecx = 2-bit tag code (CONS = 2)
     cmp ecx, 2                                ; exactly CONS, not FLOAT or NUMBER?
     jne .stream_loop                          ; not a cons: skip
     mov r15d, eax                             ; r15d = save fn descriptor
     call Car                                  ; eax =, car bound_val
     cmp eax, [rel atom_sl_fn_tag]             ; is it __FN__?
     jne .stmt_check_xfn                       ; no: maybe __XF__, exe function
; invoke user-defined function as statement, capture return for expression-RUN
     mov eax, r15d                             ; eax = restored fn descriptor
     call lg_invoke_user_fn                    ; invoke
     mov [rel sl_stmt_fn_retval], eax          ; park return value for expression-RUN, OUTPUT flag is consumed at the fn boundary
     jmp .stream_loop                          ; continue
.stmt_check_xfn:
     cmp eax, [rel atom_sl_xfn_tag]            ; is it __XF__, DEF_EXE_FN ?
     jne .stream_loop                          ; no: not a function: skip
     mov eax, r15d                             ; eax = restored xfn descriptor
     call lg_invoke_exe_fn                     ; invoke exe function
     mov [rel sl_stmt_fn_retval], eax          ; publish the return value, same expression-RUN apply channel as user fns
     jmp .stream_loop                          ; continue
.stmt_warn_miss:
; the statement head is an unbound atom: the documented bare-atom skip is
; about to happen, so warn ONCE per word first, typo trap, e.g. `pritn`
     mov eax, ebx                              ; eax = the head atom, ebx unchanged by Assoc, Win64 non-volatile
     mov ecx, 1                                ; ecx = position class 1, statement
     call sl_warn_unknown_word                 ; one-time-per-word warning, the bare-atom skip at the jmp below still runs
     jmp .stream_loop                          ; continue stream evaluation, the bare-atom skip
; ---- SPLICE: statement handlers, locals of the EvalLogoStream namespace ----
%include "sectorlogo_eval_stmt.inc"
.stream_done:
     mov edx, r14d                             ; return updated env in edx
     pop rbp                                   ; restore non-volatile rbp, saved at entry, used as scratch
     pop r15                                   ; restore non-volatile
     pop r14                                   ; restore non-volatile
     pop r12                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     ret                                       ; return to caller
%include "sectorlogo_eval_infix.inc"
%include "sectorlogo_eval_primary.inc"
%include "sectorlogo_eval_ops.inc"
%include "sectorlogo_eval_dispatch.inc"
