; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .text
%define ITOA_BUF_PAST_END 11                   ; [itoa_buf + ITOA_BUF_PAST_END] = one past last digit in buffer
%define ITOA_DEC_TEN 10                        ; decimal multiplier for quotient*10 remainder math
; v7s_itoa, render u32 value as decimal ASCII at rdi, NUL-terminated.
; in: eax = value, MUST arrive zero-extended: the 64-bit imul below consumes
; the upper half of rax, load it with a 32-bit mov, never mov rax, reg64
; rdi = destination buffer, needs 12 bytes
; out: rdi = advanced past the NUL, rdi is NOT preserved, callers chain on it
; overwrites: rax, rcx, rdx, r9, flags, rsi/rbx saved by the frame
     SF_START v7s_itoa, rsi, rbx               ; prologue: save non-volatile rsi, rep movsb overwrites and rbx
     SF_ARRAY itoa_buf, 16                     ; scratch buffer for backward digit write, 11 bytes used + pad
%assign ITOA_RBP_DISP __sf_arr_end             ; capture the end-anchored array displacement at declaration time
     SF_COMMIT                                 ; finalize stack frame layout
     lea rbx, [rbp - ITOA_RBP_DISP + ITOA_BUF_PAST_END] ; rbx = one past buffer end; digits are written BACKWARD from here, dec rbx before each store
     mov rsi, rbx                              ; cache far end in rsi, digit-count anchor; survives the loop
     mov ecx, 0xCCCCCCCD                       ; magic reciprocal of 10, hoisted before loop; 32-bit write zero-extends rcx
v7s_itoa_loop:
     mov edx, eax                              ; save original value
     imul rax, rcx                             ; rax = value * magic, full product fits 64 bits; quotient carried in eax after the shr
     shr rax, 35                               ; rax = quotient, value / 10
     imul r9d, eax, ITOA_DEC_TEN               ; r9d = quotient * 10, for the remainder below
     sub edx, r9d                              ; edx = remainder, digit
     add dl, '0'                               ; convert digit to ASCII
     dec rbx                                   ; move backward in buffer
     mov [rbx], dl                             ; store digit
     test eax, eax                             ; any more digits?, eax = carried quotient
     jnz v7s_itoa_loop                         ; yes, continue
; Copy digits from rbx to rdi, forward direction
     mov rcx, rsi                              ; rcx = base+11, cached in rsi before loop
     sub rcx, rbx                              ; rcx = digit count, base+11 - first digit
     mov rsi, rbx                              ; source = first digit position
     MEMORY_COPY_BYTE                          ; copy all digits to destination
     mov byte [rdi], 0                         ; null terminate after digits
     inc rdi                                   ; advance past null
     SF_END rsi, rbx                           ; epilogue: restore rsi, rbx, return rdi past null
; ============================================================================
; Debug / diagnostic exports, v7s_get_num_workers, get_config_was_fixed, unpin_calling_thread, v7s_set_spin
; ============================================================================
     EXPORT_DWORD_GETTER v7s_get_num_workers, num_workers ; export: returns current num_workers setting
     EXPORT_QWORD_GETTER get_config_was_fixed, config_was_fixed ; export: returns the u64 config-degeneration bitmap
     SF_START unpin_calling_thread             ; prologue: no non-volatiles needed
     SF_COMMIT                                 ; finalize stack frame layout
     cmp dword [rel pinned_flag], 0            ; is any caller affinity binding live?
     je .unpin_done                            ; no, nothing to restore
     mov rcx, [rel caller_thread_handle]       ; stable real handle saved by DuplicateHandle
     xor edx, edx                              ; 0-ms wait, inspect terminated state without blocking
     API_CALL [__imp_WaitForSingleObject], rcx, rdx ; WAIT_OBJECT_0 = ended, WAIT_TIMEOUT = running
     test eax, eax                             ; WAIT_OBJECT_0 is 0
     jz .close_only                            ; terminated host thread: cannot restore its settings
     cmp eax, 0x102                            ; WAIT_TIMEOUT, still running?
     jne .unpin_fatal                          ; WAIT_FAILED or unexpected result
     mov rcx, [rel caller_thread_handle]       ; existing caller handle
     mov edx, [rel caller_saved_priority]      ; original priority before V7 changed it
     API_CALL [__imp_SetThreadPriority], rcx, rdx ; restore original priority
     test eax, eax                             ; success?
     jz .unpin_fatal                           ; no
     mov rcx, [rel caller_thread_handle]       ; existing caller handle
     mov rdx, [rel saved_affinity]             ; original 64-bit affinity mask
     API_CALL [__imp_SetThreadAffinityMask], rcx, rdx ; restore original affinity
     test rax, rax                             ; a full 64-bit success result
     jz .unpin_fatal                           ; no
.close_only:
     mov rcx, [rel caller_thread_handle]       ; terminated or restored thread
     API_CALL [__imp_CloseHandle], rcx         ; close duplicated handle, it is no longer needed
     test eax, eax                             ; close succeeded?
     jz .unpin_fatal                           ; report invalid handle instead of discarding it
     mov qword [rel caller_thread_handle], 0   ; clear old handle
     mov dword [rel pinned_flag], 0            ; no active caller binding
     mov dword [rel caller_bound_tid], 0       ; invalidate fast-path identity cache
.unpin_done:
     SF_END                                    ; epilogue: return
.unpin_fatal:
     FATAL "Cannot restore or release the previous dispatch caller thread." ; visible death, never a silent wrong-core pin
; Bind the active dispatching thread to caller CPU 0.
; Called only on an OS thread-ID mismatch. All queue dispatches remain serialized.
     SF_START scheduler_bind_current_caller, rbx ; prologue: rbx carries the new thread ID across the API calls
     SF_COMMIT                                 ; stack frame and Win64 call alignment
     V7_THREAD_ID ebx, rbx                     ; save new thread identity across Windows API calls
%ifdef CPU0_DIAG
     mov eax, [rel caller_bound_tid]           ; eax = thread ID currently pinned as the CPU-0 caller
     TRACE_REG "cpu0diag_prev_bound_tid", rax  ; diag: the binding in effect before this handoff
     TRACE_REG "cpu0diag_new_tid", rbx         ; diag: the dispatching thread requesting the bind
%endif
     cmp ebx, [rel caller_bound_tid]           ; unnecessary repeat bind?
     je .bind_done                             ; do nothing
     call unpin_calling_thread                 ; restore and close the old binding, if present
     lea r9, [rel caller_thread_handle]        ; destination for a persistent thread handle
     API_CALL [__imp_DuplicateHandle], -1, -2, -1, r9, 0, 0, 2 ; duplicate THIS thread's pseudo-handle
     test eax, eax                             ; duplicate succeeded?
     jz .bind_handle_fatal                     ; cannot retain caller handle
     mov rcx, [rel caller_thread_handle]       ; handle of this executing thread
     API_CALL [__imp_GetThreadPriority], rcx   ; preserve its incoming priority for later restoration
     cmp eax, 0x7fffffff                       ; THREAD_PRIORITY_ERROR_RETURN?
     je .bind_priority_read_fatal              ; API failed
     mov [rel caller_saved_priority], eax      ; save original priority
     mov rcx, [rel caller_thread_handle]       ; selected caller thread
     mov edx, 1                                ; logical CPU 0, same group-0 assumption as normal scheduler
     API_CALL [__imp_SetThreadAffinityMask], rcx, rdx ; pin the active caller to CPU 0
     test rax, rax                             ; nonzero previous affinity means success
     jz .bind_affinity_fatal                   ; failed: do not publish bound thread ID
     mov [rel saved_affinity], rax             ; preserve original mask for shutdown/handoff
     mov dword [rel pinned_flag], 1            ; unpin can now restore this binding
%ifdef CPU0_DIAG
     TRACE_REG "cpu0diag_bound_tid", rbx       ; diag: the newly pinned caller thread ID
%endif
     mov rcx, [rel caller_thread_handle]       ; selected caller thread
     mov edx, [rel scheduler_effective_priority] ; current effective priority, including overrides, independent of the per-shape config slots
     API_CALL [__imp_SetThreadPriority], rcx, rdx ; give caller the same priority as the pool
     test eax, eax                             ; success?
     jz .bind_priority_set_fatal               ; failure must be reported
     mov [rel caller_bound_tid], ebx           ; publish ID LAST, once binding is complete
.bind_done:
     SF_END rbx                                ; restore nonvolatile register and return
.bind_handle_fatal:
     FATAL "Cannot duplicate the dispatch caller thread handle." ; the caller thread cannot be retained for later restore
.bind_priority_read_fatal:
     FATAL "Cannot read the dispatch caller thread priority." ; GetThreadPriority returned the error sentinel
.bind_affinity_fatal:
     FATAL "Cannot pin the dispatch caller to logical processor zero." ; SetThreadAffinityMask refused the CPU-0 mask
.bind_priority_set_fatal:
     FATAL "Cannot set the dispatch caller thread priority." ; SetThreadPriority rejected the pinned caller
%ifdef CPU0_DIAG
; One-time CPU0_DIAG session dump: caller identity, binding state and caller-job
; counters, printed after the first completed dispatch of a session.
     SF_START cpu0_diag_dump, rbx, r12, r13, r14 ; prologue: rbx = worker index, r12 = value holder, r13 = worker count, r14 = array base
     SF_COMMIT                                 ; finalize stack frame layout
     WRITE_STR lbl_cpu0_num_workers            ; print "cpu0_diag: num_workers: "
     mov r13d, [num_workers]                   ; r13d = active worker count, loop bound below
     mov r12d, r13d                            ; r12d = count, 32-bit move zero-extends into r12
     WRITE_UINT64 r12                          ; print worker count
     WRITE_NL                                  ; end the line
     WRITE_STR lbl_cpu0_worker_aff             ; print "cpu0_diag: worker_aff: "
     test r13d, r13d                           ; any workers?
     jz .diag_aff_done                         ; no, skip the mask walk
     xor ebx, ebx                              ; rbx = worker index
     lea r14, [rel worker_aff]                 ; r14 = worker affinity array base, non-volatile, survives the WRITE_* calls
.diag_aff_loop:
     mov r12, [r14 + rbx*QWORD_SCALE]          ; r12 = worker_aff[i], pre-loaded into non-volatile
     WRITE_HEX64 r12                           ; print the mask in hex
     WRITE_CHR ' '                             ; space separator between masks
     inc rbx                                   ; next worker
     cmp ebx, r13d                             ; all active workers printed?
     jl .diag_aff_loop                         ; continue the walk
.diag_aff_done:
     WRITE_NL                                  ; end the affinity line
     WRITE_STR lbl_cpu0_saved_affinity         ; print "cpu0_diag: saved_affinity: "
     mov r12, [rel saved_affinity]             ; r12 = previous affinity mask of the bound caller
     WRITE_HEX64 r12                           ; print the mask in hex
     WRITE_NL                                  ; end the line
     WRITE_STR lbl_cpu0_dispatch_tid           ; print "cpu0_diag: dispatch_tid: "
     V7_THREAD_ID eax, rax                     ; eax = OS thread ID of this dispatching caller
     mov r12, rax                              ; r12 = zero-extended TID for the print
     WRITE_UINT64 r12                          ; print the dispatch thread ID
     WRITE_NL                                  ; end the line
     WRITE_STR lbl_cpu0_bound_tid              ; print "cpu0_diag: caller_bound_tid: "
     mov eax, [rel caller_bound_tid]           ; eax = thread ID pinned as the CPU-0 caller
     mov r12, rax                              ; r12 = zero-extended TID for the print
     WRITE_UINT64 r12                          ; print the bound thread ID
     WRITE_NL                                  ; end the line
     WRITE_STR lbl_cpu0_caller_jobs            ; print "cpu0_diag: caller_jobs_done: "
     mov r12, [rel caller_jobs_done]           ; r12 = jobs the caller itself completed since the last matmul reset
     WRITE_UINT64 r12                          ; print the caller-job total
     WRITE_NL                                  ; end the line
     SF_END rbx, r12, r13, r14                 ; restore non-volatiles and return
%endif
     global v7s_set_spin                       ; export: declare the inference activity phase
v7s_set_spin:
     mov dword [always_spin], ecx              ; private activity flag: 1 keeps workers pause-spinning and fork-receptive, 0 returns them to the OS wait
     test ecx, ecx                             ; only the spin-active declaration must wake anyone
     jz .set_spin_done                         ; flag 0 just arms the idle transition workers see on their own
     cmp dword [workers_may_sleep], 0          ; anyone registered in the OS wait?
     je .set_spin_done                         ; no: the pool already pause-spins, nothing to wake
     lea rcx, [rel gen_counter]                ; the address registered sleepers wait on
     API_CALL [__imp_WakeByAddressAll], rcx    ; wake them; the recheck protocol sends them to the spin loop before any fork
.set_spin_done:
     ret                                       ; no mode semantics: forks run whenever inference is active, the idle wait style is not configurable
; Internal end-of-chat notification. Workers observe the store in their spin loop.
v7s_stop_spin:
     mov dword [rel always_spin], 0            ; notify idle workers to enter WaitOnAddress, no acknowledgment wait
     ret                                       ; caller does not wait for the transition
; ---- %undef cleanup: file-local defines leave scope here, house convention ----
%undef ITOA_BUF_PAST_END                       ; backward-digit far-end offset, used only at :12
%undef ITOA_RBP_DISP                           ; captured array displacement, site expanded
%undef ITOA_DEC_TEN                            ; decimal multiplier, used only at :20
