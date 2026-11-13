; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; fork.inc -- Recursive fork/join over the existing spinning worker pool.
; Internal NASM macros only. No additional threads, queue, export, or OS call.

; The parent publishes RAX/RBX/RCX/RDX and a worker-only continuation address.
; Both threads reach the same code site with different ZF values:
; ZF=1 on the parent, ZF=0 on the helper.
; A child may execute CONTINUE_FORK N for intermediate phases and must end
; its last phase with CONTINUE_DISPATCH N. Join after the final phase before
; any full-pool DISPATCH_EXECUTE.

; CONTRACT:
; - FORK N is used only when worker N exists and is in the pause-spin path.
; - Exactly one parent owns each worker N at a time; slots may not overlap.
; - Forked branch uses its own worker stack; caller stack locals are NOT copied.
; - Only RAX/RBX/RCX/RDX are transferred, R11 is scratch at FORK.
; - Forked branch keeps RSP balanced; intermediate CONTINUE_FORK N keeps
; the original worker stack frame, final CONTINUE_DISPATCH N restores pool.
; - No live flags, SIMD state, or other GPR state is transferred.
; - Child branches must respect Win64 ABI callee-saved XMM6-XMM15.
; - All descendants must complete before their parent publishes completion.

%ifndef V7_FORK_INC
%define V7_FORK_INC 1                          ; include guard, file scanned once per translation unit

     struc fork_param                          ; per-worker handoff slot layout, one 64-byte cache line
.rax:
     resq 1                                    ; input/output RAX
.rbx:
     resq 1                                    ; input/output RBX
.rcx:
     resq 1                                    ; input/output RCX
.rdx:
     resq 1                                    ; input/output RDX
.targetaddr:
     resq 1                                    ; nonzero=active, zero=completed/available
.padding:
     resb 24                                   ; one independent 64-byte cache line
     endstruc                                  ; defines fork_param_size

%if fork_param_size != 64
%error fork_param must occupy exactly one 64-byte cache line ; layout guard: slots are addressed by cache line
%endif

; FORK N -- N is a compile-time worker ID, 0..MAX_WORKERS-1.
; Immediately follow this macro with JZ parent_label or JNZ child_label.
; Main/caller continues with ZF=1; worker N starts with ZF=0.
; Preserves passed GPR values on caller, R11 included (restored after the flag setup).

%macro FORK 1
%if %1 < 0
%error FORK: worker ID must be nonnegative     ; N is a compile-time pool index, negative IDs have no slot
%endif
%if %1 >= MAX_WORKERS
%error FORK: worker ID exceeds MAX_WORKERS     ; N must stay inside the fork_params slot array
%endif
     mov qword [rel fork_params + %1*fork_param_size + fork_param.rax], rax ; Copy RAX to the target worker slot
     mov qword [rel fork_params + %1*fork_param_size + fork_param.rbx], rbx ; Copy RBX to the target worker slot
     mov qword [rel fork_params + %1*fork_param_size + fork_param.rcx], rcx ; Copy RCX to the target worker slot
     mov qword [rel fork_params + %1*fork_param_size + fork_param.rdx], rdx ; Copy RDX to the target worker slot
     push r11                                  ; caller's R11 is volatile but preserved here for cleanliness
     lea r11, [rel %%child]                    ; Take the helper-only continuation address
     mov qword [rel fork_params + %1*fork_param_size + fork_param.targetaddr], r11 ; Publish continuation LAST
     xor r11d, r11d                            ; Original thread enters the common site with ZF=1, R11 is contract scratch
     pop r11                                   ; restore the pushed caller R11 after the flag setup
     jmp %%common                              ; Skip the worker-only flag instruction
%%child:
     test rsp, rsp                             ; Worker has a nonzero stack: enter with ZF=0
%%common:
%endmacro

; CONTINUE_DISPATCH N -- ONLY on worker N's forked branch at balanced RSP.
; Save the four result GPRs; shared targetaddr is cleared by the worker
; epilogue only AFTER its normal dispatcher state and stack are restored.

%macro CONTINUE_DISPATCH 1
%if %1 < 0
%error CONTINUE_DISPATCH: worker ID must be nonnegative ; N is a compile-time pool index, negative IDs have no slot
%endif
%if %1 >= MAX_WORKERS
%error CONTINUE_DISPATCH: worker ID exceeds MAX_WORKERS ; N must stay inside the fork_params slot array
%endif
     mov qword [rel fork_params + %1*fork_param_size + fork_param.rax], rax ; Return RAX in the worker slot
     mov qword [rel fork_params + %1*fork_param_size + fork_param.rbx], rbx ; Return RBX in the worker slot
     mov qword [rel fork_params + %1*fork_param_size + fork_param.rcx], rcx ; Return RCX in the worker slot
     mov qword [rel fork_params + %1*fork_param_size + fork_param.rdx], rdx ; Return RDX in the worker slot
     jmp worker_proc.fork_return_worker        ; Restore worker stack/dispatcher state, then publish completion
%endmacro

; CONTINUE_FORK N -- intermediate worker phase, keep the original fork frame.
; The worker publishes the four results and clears targetaddr without restoring
; pool registers or returning to the worker scheduler. It spins without PAUSE
; until the owner publishes another FORK N, then restores only four arguments
; and branches straight to the new child continuation. Exactly one owner.
; The parent must observe targetaddr == 0 before reading results or reusing slot.
; The last worker phase MUST use CONTINUE_DISPATCH N, then FORK_WAIT N,
; before a normal full-pool dispatch can begin.

%macro CONTINUE_FORK 1
%if %1 < 0
%error CONTINUE_FORK: worker ID must be nonnegative ; must address a pool worker
%endif
%if %1 >= MAX_WORKERS
%error CONTINUE_FORK: worker ID exceeds MAX_WORKERS ; stay inside fork_params
%endif
     mov qword [rel fork_params + %1*fork_param_size + fork_param.rax], rax ; Publish phase RAX
     mov qword [rel fork_params + %1*fork_param_size + fork_param.rbx], rbx ; Publish phase RBX
     mov qword [rel fork_params + %1*fork_param_size + fork_param.rcx], rcx ; Publish phase RCX
     mov qword [rel fork_params + %1*fork_param_size + fork_param.rdx], rdx ; Publish phase RDX
     mov qword [rel fork_params + %1*fork_param_size + fork_param.targetaddr], 0 ; Phase complete; worker stays in helper frame
%%next:
     cmp qword [rel fork_params + %1*fork_param_size + fork_param.targetaddr], 0 ; New continuation published?
     je %%next                                 ; Dedicated hot spin: deliberately NO PAUSE, pool logic or OS calls
     mov r10, qword [rel fork_params + %1*fork_param_size + fork_param.targetaddr] ; Capture next child continuation
     mov rax, qword [rel fork_params + %1*fork_param_size + fork_param.rax] ; Read next RAX argument
     mov rbx, qword [rel fork_params + %1*fork_param_size + fork_param.rbx] ; Read next RBX argument
     mov rcx, qword [rel fork_params + %1*fork_param_size + fork_param.rcx] ; Read next RCX argument
     mov rdx, qword [rel fork_params + %1*fork_param_size + fork_param.rdx] ; Read next RDX argument
     jmp r10                                   ; Execute the next worker-only FORK continuation
%endmacro

; FORK_WAIT N -- wait for targetaddr == 0 (intermediate yield OR final pool return).
; Clobbers only flags; does not read result registers. The parent MUST publish
; another FORK after an intermediate CONTINUE_FORK, or finish with a final
; CONTINUE_DISPATCH phase before any full-pool dispatch or shutdown.

%macro FORK_WAIT 1
%if %1 < 0
%error FORK_WAIT: worker ID must be nonnegative ; N is a compile-time pool index, negative IDs have no slot
%endif
%if %1 >= MAX_WORKERS
%error FORK_WAIT: worker ID exceeds MAX_WORKERS ; N must stay inside the fork_params slot array
%endif
%%wait:
     pause                                     ; Short cross-core polling hint
     cmp qword [rel fork_params + %1*fork_param_size + fork_param.targetaddr], 0 ; Worker has published completion?
     jne %%wait                                ; Keep waiting while its target is nonzero
%endmacro

%endif                                         ; V7_FORK_INC
