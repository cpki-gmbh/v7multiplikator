; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .rdata
kp_env_name:
     db "v7_override_prio", 0                  ; env var checked once at init, Win32 env names are case-insensitive
kp_msg_env_prio:
     db "Worker priority override set to ", 0  ; same visible line the -prio flag path prints, the env override must be equally observable
     section .bss
     align 8
kp_env_value:
     resb 64                                   ; raw env value bytes, GetEnvironmentVariableA target; init-path only, single-threaded here
kp_pending_prio:
     dd ?                                      ; deferred request value: a v7_set_worker_priority call that arrived before pool creation
kp_pending_flag:
     dd ?                                      ; 1 = kp_pending_prio holds an unapplied request, consumed at pool creation or by the next direct apply
kp_env_seen:
     dd ?                                      ; 1 = the v7_override_prio env var was consulted during this pool creation, read by the threading report
     section .text
; v7_set_worker_priority is a PUBLIC entry: its gated wrapper lives in
; public_api.inc, PUB_RAW_CALL 0 -> set_worker_priority_all, per the
; v7_enable_thinking / v7_set_sidecar pattern. This file keeps the core.
set_worker_priority_all:
     SF_START set_worker_priority_all, rbx, r12, r13 ; frame; rbx = worker index, r12d = priority, r13 = threads base
     SF_COMMIT                                 ; commit stack frame
     mov r12d, ecx                             ; r12d = requested priority, survives API_CALL overwrites
     TRACE_STR "swpa: enter"                   ; debug: the priority setter core ran, caller identified by the caller's own trace
     TRACE_REG "swpa_prio", r12                ; debug: requested priority value
     mov eax, [rel num_workers]                ; eax = worker count, 0 = the deferred path below
     TRACE_REG "swpa_num_workers", rax         ; debug: pool existence at setter entry
     test eax, eax                             ; worker pool exists?, uses the load above, eax survives the TRACE_REG in debug builds, and no call intervenes
     je .swpa_deferred                         ; no, rare: park the request; apply_pending picks it up after pool creation
     lea r13, [rel threads]                    ; r13 = worker-handle array base, loop-invariant
     xor ebx, ebx                              ; rbx = worker index = 0
.swpa_loop:
     mov rcx, [r13 + rbx*QWORD_SCALE]          ; rcx = worker i's thread handle
     mov edx, r12d                             ; edx = priority
     TRACE_REG "swpa_handle", rcx              ; debug: the handle SetThreadPriority is about to receive
     API_CALL [__imp_SetThreadPriority], rcx, rdx ; apply to worker i, overwrites rax,rcx,rdx,r8-r11
     TRACE_REG "swpa_stp_ret", rax             ; debug: SetThreadPriority return, 0 = rejected
     test eax, eax                             ; BOOL success?
     jz .swpa_fatal                            ; no: partially-applied pool, FATAL
     inc ebx                                   ; next worker
     cmp ebx, [rel num_workers]                ; all workers done?
     jl .swpa_loop                             ; not yet, continue pool walk, bottom-tested: entry above proved num_workers >= 1, so the first trip is guaranteed
; Apply the same priority to the pinned caller through its retained handle, so the
; whole participant set (workers + caller) runs at one priority.
     mov rcx, [rel caller_thread_handle]       ; real caller thread handle, captured at pool creation
     mov edx, r12d                             ; edx = priority
     API_CALL [__imp_SetThreadPriority], rcx, rdx ; apply to the caller
     test eax, eax                             ; BOOL success?
     jz .swpa_caller_fatal                     ; no: caller left at the old priority, FATAL
.swpa_done:
     mov [rel scheduler_effective_priority], r12d ; save effective priority only after successful application, the per-shape worker_priority slots are config the dispatcher overwrites and never scheduler truth
     SF_END rbx, r12, r13                      ; restore non-volatiles, return, SF_END emits the ret, the fatal block below is only reachable via jz
.swpa_caller_fatal:
     FATAL "v7_set_worker_priority: SetThreadPriority failed for the calling thread, workers were switched but the caller keeps the old priority" ; SetThreadPriority rejected the caller handle or the value
.swpa_fatal:
     API_CALL [__imp_GetLastError]             ; eax = SetThreadPriority's failure code, MUST be the first call: anything else resets the per-thread error
     mov r14d, eax                             ; park the code, the traces below call Win32 and would reset it
     mov rcx, [r13 + rbx*QWORD_SCALE]          ; rcx = the rejected handle, reload, rcx was overwritten by the failed call
     TRACE_REG "swpa_bad_handle", rcx          ; debug: the handle SetThreadPriority rejected
     TRACE_REG "swpa_GetLastError", r14        ; debug: the Win32 failure code behind the rejection
     FATAL "v7_set_worker_priority: SetThreadPriority failed, workers before the failing one are already switched, the rest keep the old priority" ; SetThreadPriority rejected a handle or the value; the pool is left partially updated
; ---- out of line: the rare pre-pool defer path, nothing falls into it ----
.swpa_deferred:
     TRACE_STR "swpa: pool absent, request parked" ; debug: the deferred path was taken
     mov dword [rel kp_pending_prio], r12d     ; no pool yet, remember the requested priority
     mov dword [rel kp_pending_flag], 1        ; flag it; apply_pending picks it up right after pool creation
     SF_END rbx, r12, r13                      ; restore non-volatiles, return, nothing to apply yet
worker_priority_check_env_override:
     SF_START worker_priority_check_env_override, rsi ; frame; rsi = parse cursor into kp_env_value
     SF_COMMIT                                 ; commit stack frame
     lea rcx, [rel kp_env_name]                ; arg1 = variable name "v7_override_prio"
     lea rdx, [rel kp_env_value]               ; arg2 = value buffer, 64 BSS bytes
     mov r8d, 64                               ; arg3 = buffer capacity in chars
     API_CALL [__imp_GetEnvironmentVariableA], rcx, rdx, r8 ; read env var, returns chars copied excl. NUL; 0 = unset or empty
     test eax, eax                             ; variable found and non-empty?
     jz .kpce_ret                              ; no, no-op: the INI priority stands
     mov dword [rel kp_env_seen], 1            ; report flag: the env override was consulted this pool creation
     TRACE_REG "kpce_env_len", rax             ; debug: the override value's char count
     cmp eax, 64                               ; value fit the buffer completely?
     jae .kpce_malformed                       ; no, oversize garbage, never apply a truncated value
     lea rsi, [rel kp_env_value]               ; rsi = value string cursor
     xor r10d, r10d                            ; r10d = accumulator = 0
     xor r11d, r11d                            ; r11d = negative flag, 0 = positive
     xor r9d, r9d                              ; r9d = consumed digit count, a bare "-" has none -> malformed
     cmp byte [rsi], '-'                       ; leading minus?
     jne .kpce_digits                          ; no, plain digits
     mov r11d, 1                               ; remember negation
     inc rsi                                   ; skip the sign
.kpce_digits:
     movzx eax, byte [rsi]                     ; load next character
     sub al, '0'                               ; ASCII -> digit
     cmp al, 9                                 ; valid decimal digit?
     ja .kpce_digits_done                      ; no, end of the numeric run
     inc r9d                                   ; one more valid digit consumed
     imul r10d, 10                             ; accumulator *= 10, shift to the next decimal place
     jo .kpce_malformed                        ; signed overflow, not a representable priority
     add r10d, eax                             ; accumulator += digit, eax is 0..9 from the movzx/sub above
     jo .kpce_malformed                        ; overflow on the add as well
     inc rsi                                   ; next character
     jmp .kpce_digits                          ; continue parsing
.kpce_digits_done:
     cmp byte [rsi], 0                         ; cursor must sit on the NUL terminator
     jne .kpce_malformed                       ; non-numeric remainder, "12x", " 15", "--5", reject
     test r9d, r9d                             ; at least one digit in the run?
     jz .kpce_malformed                        ; no: a bare "-" is not an integer, the same FATAL as every other non-numeric value
     test r11d, r11d                           ; negative sign seen?
     jz .kpce_apply                            ; no, keep magnitude
     neg r10d                                  ; yes, apply it, 0x80000000 edge cannot occur: digits ran through imul/add JO checks
.kpce_apply:
     mov ecx, r10d                             ; ecx = parsed priority, core ABI arg1
     TRACE_REG "kpce_parsed_prio", rcx         ; debug: the env override value being applied
     call set_worker_priority_all              ; apply through the SAME core the public entry uses
     WRITE_STR kp_msg_env_prio                 ; "Worker priority override set to "
     WRITE_STR kp_env_value                    ; the raw env value text, NUL-terminated, the same wording the flag path prints
     WRITE_NL                                  ; end the line
.kpce_ret:
     SF_END rsi                                ; restore non-volatiles, return, init continues
.kpce_malformed:
     FATAL "v7_override_prio env var malformed, expected a Win32 thread-priority integer (e.g. -15 .. 15), got an empty/oversize/non-numeric value"
worker_priority_apply_pending:
     SF_START worker_priority_apply_pending    ; frame, no non-volatiles used
     SF_COMMIT                                 ; commit stack frame
     TRACE_STR "kpap: enter"                   ; debug: pool creation asks for a parked pre-pool request
     mov eax, [rel kp_pending_flag]            ; eax = 1 when a pre-pool request is parked
     TRACE_REG "kpap_flag", rax                ; debug: the parked-request flag
     test eax, eax                             ; did a setter call arrive before pool creation?, uses the load above
     jz .kpap_ret                              ; no, nothing deferred
     mov dword [rel kp_pending_flag], 0        ; consume the request, one-shot
     mov ecx, [rel kp_pending_prio]            ; ecx = deferred priority, core ABI arg1
     TRACE_REG "kpap_deferred_prio", rcx       ; debug: the parked value being applied now
     call set_worker_priority_all              ; apply to the just-created pool, pool exists here -> direct path; the core no longer touches the flag, the consume above IS the one-shot
.kpap_ret:
     SF_END                                    ; restore frame, return, init continues
