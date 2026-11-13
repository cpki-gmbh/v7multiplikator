; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; attention_fork.inc -- Reentrant single-head attention range runner, used only
; between full-pool dispatches. Argument RCX = immutable descriptor on the
; parent stack; EDX = 0 (first half of heads), 1 (second half).

     struc attention_fork_args                 ; fork range-runner descriptor layout, 48 bytes
.state:
     resq 1                                    ; RunState* s
.config:
     resq 1                                    ; Config* p
.loff:
     resq 1                                    ; layer KV-cache offset, uint64
.pos:
     resd 1                                    ; sequence position
.kv_dim:
     resd 1                                    ; KV cache dimension
.kv_mul:
     resd 1                                    ; query heads per KV head, GQA
.heads:
     resd 1                                    ; total query head count
.half:
     resd 1                                    ; split point, floor(heads/2)
.padding:
     resd 1                                    ; pad to 48 bytes
     endstruc                                  ; defines attention_fork_args_size
%if attention_fork_args_size != 48
%error attention_fork_args must be 48 bytes    ; layout guard: descriptor slots are addressed by these offsets
%endif

     section .text
; ============================================================================
; v7s_attention_fork_range -- run v7s_attention_inner over a disjoint head range
; RCX = attention_fork_args* descriptor, EDX = half selector (0/1)
; ============================================================================
     SF_START v7s_attention_fork_range, rbx, r12, r13 ; preserve the range state
     SF_ARRAY af_stack_args, 24                ; reserves frame space so stack-arg homes 5..7 sit inside the frame beyond Win64 shadow space
     SF_COMMIT                                 ; finalize stack frame layout
     mov r12, rcx                              ; r12 = shared read-only descriptor
     mov eax, [r12 + attention_fork_args.half] ; split point used only during setup
     test edx, edx                             ; half selector == 0?
     jnz .second                               ; no, run the upper range
     xor ebx, ebx                              ; ebx = first head, lower range starts at 0
     mov r13d, eax                             ; r13d = upper bound = split point
     jmp .head_check                           ; enter the loop
.second:
     mov ebx, eax                              ; ebx = first head of the upper range
     mov r13d, [r12 + attention_fork_args.heads] ; r13d = upper bound = total heads
.head_check:
     cmp ebx, r13d                             ; range empty?
     jae .done                                 ; nothing to do for this half
; The three stack arguments are loop-invariant: v7s_attention_inner only reads them.
     mov rax, [r12 + attention_fork_args.loff] ; rax = layer KV offset, stack arg 5
     mov [rsp + APICALL_ARG5], rax             ; arg5 = loff, written once per range
     mov eax, [r12 + attention_fork_args.kv_dim] ; eax = kv_dim, stack arg 6
     mov [rsp + APICALL_ARG6], rax             ; arg6 = kv_dim, written once per range
     mov eax, [r12 + attention_fork_args.kv_mul] ; eax = kv_mul, stack arg 7
     mov [rsp + APICALL_ARG7], rax             ; arg7 = kv_mul, written once per range
.head_loop:
     mov rcx, [r12 + attention_fork_args.state] ; arg1 = RunState* s
     mov edx, ebx                              ; arg2 = query head index
     mov r8, [r12 + attention_fork_args.config] ; arg3 = Config* p
     mov r9d, [r12 + attention_fork_args.pos]  ; arg4 = sequence position
     call v7s_attention_inner                  ; identical single-head kernel as process_job
     inc ebx                                   ; head index++
     cmp ebx, r13d                             ; range finished?
     jb .head_loop                             ; next head
.done:
     SF_END rbx, r12, r13                      ; restore the caller state
; ---- %undef cleanup: keep the preprocessor namespace clean for later includes ----
%undef af_stack_args                           ; SF_ARRAY slot alias
