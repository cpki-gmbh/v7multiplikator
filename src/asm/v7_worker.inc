; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .text
     SF_START worker_proc, rbx, rsi, rdi, r12, r13, r14, r15 ; function prologue: save non-volatiles
     SF_DWORD wp_count                         ; local: remaining job count for this worker batch
     SF_DWORD wp_captured                      ; local: captured expected value for WaitOnAddress
%ifdef PROFILE_MODE
     SF_QWORD wp_fa                            ; local: fetch-add count, profiling builds only
%endif
     SF_QWORD wp_fork_ptr                      ; this worker's 64-byte fork_param slot, computed once
     SF_COMMIT                                 ; commit stack frame for worker_proc
     mov r12d, ecx                             ; r12d = id, worker index, 0-based
     lock inc dword [workers_alive]            ; Mark this pool thread as active; shutdown waits for the matching decrement
     xor r13d, r13d                            ; my_gen = 0, first dispatch will be gen=1
     SET_FTZ_DAZ                               ; worker MXCSR == caller MXCSR by construction, not by assuming every thread starts at the CRT default
%ifdef ZYDIS_TRACE
     INT3_DBG_STR "worker: started"            ; debug: worker thread entered main loop
     INT3_DBG_REG 12                           ; R12 = id
%endif
; Cache jobs base for the lifetime of this worker
     JOB_BASE                                  ; rsi = &jobs_arr[0], the index is known zero
     mov r15, rsi                              ; r15 = jobs array base pointer, cached for this worker's lifetime
; One-time startup line: prove which thread started and where it landed
     lea r14, [rel worker_aff]                 ; r14 = affinity array base
     mov r14, [r14 + r12*QWORD_SCALE]          ; r14 = this worker's affinity mask, r12 = id, live
     V7_THREAD_ID edi, rdi                     ; edi = OS thread id of this new worker, dword load zero-extends
     API_CALL [__imp_GetCurrentProcessorNumber] ; eax = logical processor this thread is on right now
     mov ebx, eax                              ; rbx = observed logical processor
     WRITE_STR lbl_thr_w_started               ; "worker W"
     WRITE_UINT64 r12                          ; id
     WRITE_STR lbl_thr_w_mid                   ; " started: tid="
     WRITE_UINT64 rdi                          ; OS thread id
     WRITE_STR lbl_thr_w_cpu                   ; " on logical processor "
     WRITE_UINT64 rbx                          ; observed logical processor
     WRITE_STR lbl_thr_w_aff                   ; ", aff=0x"
     WRITE_HEX64 r14                           ; the mask it was created with
     WRITE_NL                                  ; end the line
; Compute this worker's independent register-handoff slot once at startup.
     mov r11, r12                              ; r11 = 0-based worker number
     shl r11, 6                                ; r11 = worker number * 64-byte slot
     lea rax, [rel fork_params]                ; rax = first helper register slot
     add rax, r11                              ; rax = this worker's register slot
     mov wp_fork_ptr, rax                      ; Cache its slot pointer in the worker's own frame
.worker_outer:
     xor ebx, ebx                              ; spins = 0, initialize for both paths
     cmp dword [always_spin], 0                ; this worker's direct observation of the mode
     je .go_to_sleep                           ; sleep mode: register and wait; spin mode falls through
; ----- Pure pause-spin path: no Windows API calls, never sleeps -----
.spin_gen:
     pause                                     ; SMT-friendly spin hint
     mov eax, [gen_counter]                    ; load gen_counter, acquire semantics
     cmp eax, r13d                             ; gen != my_gen?
     jne .gen_changed                          ; yes, new dispatch available
     cmp dword [shutdown_flag], 0              ; worker requested to stop?
     jne .worker_exit                          ; yes, exit thread
     mov r11, wp_fork_ptr                      ; Find this worker's independent fork slot
     cmp qword [r11 + fork_param.targetaddr], 0 ; A parent published a helper continuation?
     jne .fork_enter                           ; Yes, enter the helper branch
     cmp dword [always_spin], 0                ; chat ended: leave the pure spin loop without a dispatch
     je .go_to_sleep                           ; worker registers and enters existing OS wait path
%ifdef PROFILE_MODE
     inc ebx                                   ; spins++, profiling builds only
%endif
     jmp .spin_gen                             ; keep spinning
; ----- Helper execution: save normal pool state once per fork -----
.fork_enter:
     push rbx                                  ; Keep spin count / job index for normal worker service
     push rsi                                  ; Preserve the normal job pointer
     push rdi                                  ; Preserve batch size and other worker-owned state
     push rbp                                  ; Preserve the worker's stack-frame base
     push r12                                  ; Preserve worker index
     push r13                                  ; Preserve worker generation
     push r14                                  ; Preserve remaining worker-owned state
     push r15                                  ; Preserve the cached jobs array pointer
     sub rsp, 32                               ; Keep Win64 call shadow space below the saved registers
     mov r10, [r11 + fork_param.targetaddr]    ; Capture the helper's published continuation
     mov rax, [r11 + fork_param.rax]           ; Transfer caller RAX
     mov rbx, [r11 + fork_param.rbx]           ; Transfer caller RBX
     mov rcx, [r11 + fork_param.rcx]           ; Transfer caller RCX
     mov rdx, [r11 + fork_param.rdx]           ; Transfer caller RDX
     jmp r10                                   ; Enter the FORK macro's worker-only label
; Reached only through CONTINUE_DISPATCH, with exactly the original helper RSP.
.fork_return_worker:
     add rsp, 32                               ; Release the helper's Win64 shadow space
     pop r15                                   ; Recover jobs array pointer
     pop r14                                   ; Recover normal worker state
     pop r13                                   ; Recover the observed generation
     pop r12                                   ; Recover worker ID
     pop rbp                                   ; Recover worker frame base for wp_fork_ptr
     pop rdi                                   ; Recover worker batch state
     pop rsi                                   ; Recover normal job pointer
     pop rbx                                   ; Recover worker spin/job index
     mov r11, wp_fork_ptr                      ; Recover the worker's fork slot after frame restoration
     mov qword [r11 + fork_param.targetaddr], 0 ; Publish done LAST: worker state is restored
     jmp .worker_outer                         ; Rejoin the ordinary dispatch/sleep decision loop
; ----- Sleep path: register in the wait-capable count, then WaitOnAddress -----
.go_to_sleep:
     xor ebx, ebx                              ; spins = 0, sleep path never spins; .gen_changed's wake_spins += rbx is a no-op add from here, only .spin_gen records nonzero
     lock inc dword [workers_may_sleep]        ; register BEFORE rechecking the generation
     mov eax, [gen_counter]                    ; mandatory recheck AFTER registration
     cmp eax, r13d                             ; gen already changed since last check?
     jne .sleep_gen_changed                    ; yes, handle it now, unregistering first
     cmp dword [shutdown_flag], 0              ; worker requested to stop?
     jne .sleep_shutdown                       ; yes, exit, unregistering first
; Record sleep
     mov wp_captured, eax                      ; SAVE captured gen BEFORE overwriteing eax
%ifdef PROFILE_MODE
     mov eax, r12d                             ; id
     lea r11, [rel worker_sleeps]              ; base of per-worker sleep counters
     shl rax, 6                                ; rax = id * 64, cache-line stride
     inc qword [r11 + rax]                     ; worker_sleeps[id]++, PROFILE
%endif
     lea rcx, [gen_counter]                    ; lpAddress = &gen_counter
     lea rdx, wp_captured_raw                  ; lpCompareAddress = my captured gen value
     mov r8d, 4                                ; AddressSize = 4, gen_counter is a dword
     mov r9d, 10                               ; dwMilliseconds = 10ms, safety-net timeout; publish-wake handles the normal case
     API_CALL [__imp_WaitOnAddress], rcx, rdx, r8, r9 ; 4-parameter API: address, compare, size, timeout
     lock dec dword [workers_may_sleep]        ; normal wake, timeout or spurious wake: unregister
     jmp .worker_outer                         ; re-evaluate spin mode and generation
.sleep_gen_changed:
     lock dec dword [workers_may_sleep]        ; unregister before handling the generation change
     jmp .gen_changed                          ; eax still contains the new generation
.sleep_shutdown:
     lock dec dword [workers_may_sleep]        ; unregister before exit
     jmp .worker_exit                          ; leave the worker loop on the shutdown signal
.gen_changed:
; Save cur_gen before overwriteing eax
     mov r13d, eax                             ; my_gen = cur_gen
     lfence                                    ; serialize: prevent speculative job reads
%ifdef ZYDIS_TRACE
     INT3_DBG_STR "worker: gen_changed"        ; debug: gen_counter changed, worker waking
     INT3_DBG_REG 13                           ; R13 = my_gen, new
%endif
; Record wake spins
%ifdef PROFILE_MODE
     mov eax, r12d                             ; id
     lea r11, [rel worker_wake_spins]          ; base of per-worker wake-spin counters
     shl rax, 6                                ; rax = id * 64, cache-line stride
     add qword [r11 + rax], rbx                ; worker_wake_spins[id] += spins, PROFILE
%endif
; Check shutdown after wake
     cmp dword [rel shutdown_flag], 0          ; check for shutdown signal
     jne .worker_exit                          ; yes - exit thread
; ------- Phase 2: Claim and process jobs via fetch_add -------
; C++ ref: int nj = num_jobs; int batch = ...; for,;; { start = next_job.fetch_add,batch ; ... }
     mov r14d, [num_jobs]                      ; initial nj for batch selection, tagged value
     movzx r14d, r14w                          ; r14d = job count, generation tag stripped for batch selection
%ifdef ZYDIS_TRACE
     INT3_DBG_STR "worker: steal_phase"        ; debug: entering job steal phase
     INT3_DBG_REG 14                           ; R14 = num_jobs
%endif
     SELECT_BATCH r14d, edi                    ; select batch size based on num_jobs
     cmp byte [rel prefill_unit_claim], 0      ; Published before the generation increment
     je .batch_ready                           ; Ordinary jobs use the selected tuned batch
     mov edi, 1                                ; Independent panel jobs are already substantial
.batch_ready:
; batch in edi, nj in r14d - no stack saves needed, both non-volatile
     mov wp_count, 0                           ; count = 0
%ifdef PROFILE_MODE
     mov wp_fa, 0                              ; fa = 0, profiling builds only
%endif
.steal_loop:
     mov r14d, [num_jobs]                      ; tagged nj, reloaded each iteration
     mov ecx, r14d                             ; copy for tag extraction
     shr ecx, JOB_TAG_SHIFT                    ; ecx = num_jobs generation tag, 16-bit field
     cmp cx, r13w                              ; num_jobs belongs to my generation?, both tags are the low 16 bits
     jne .steal_done                           ; no, mid-publish or stale: resync via the gen recheck below
     movzx r14d, r14w                          ; r14d = nj count of my generation
     mov eax, [next_job]                       ; non-atomic peek at the tagged cursor, avoids cacheline bounce at the tail
     mov ecx, eax                              ; copy raw peek
     shr ecx, JOB_TAG_SHIFT                    ; ecx = cursor generation tag, 16-bit field
     cmp cx, r13w                              ; cursor belongs to my generation? r13w holds my tag, nothing between writes it
     jne .steal_done                           ; no, a publish just happened: resync with the fresh generation
     movzx eax, ax                             ; eax = local cursor position
     cmp eax, r14d                             ; local cursor >= nj, every job already claimed ?
     jae .steal_done                           ; yes, skip the lock xadd entirely, no work remains
     cmp dword [gen_counter], r13d             ; final gen check: gen changed since we captured my_gen?
     jne .steal_done                           ; yes, resync with the fresh generation
; next_job.fetch_add,batch, acq_rel -> tagged old value
     mov eax, edi                              ; batch for fetch_add
     lock xadd dword [next_job], eax           ; eax = tagged old cursor value
     mov ecx, eax                              ; copy raw claim
     shr ecx, JOB_TAG_SHIFT                    ; ecx = claim_gen, generation whose cursor the xadd advanced
     movzx eax, ax                             ; eax = start, local index within the claim's generation
     mov edx, [num_jobs]                       ; tagged nj reload, TSO: >= the cursor-tag store the xadd observed
     mov r8d, edx                              ; copy for tag extraction
     shr r8d, JOB_TAG_SHIFT                    ; r8d = generation of the reloaded num_jobs
     cmp r8d, ecx                              ; num_jobs tag matches the claim's generation?
     jne .steal_loop                           ; no, a later generation already completed: the claim is provably pure overshoot, re-loop
     movzx edx, dx                             ; edx = exact job-count bound of the claim's generation
; end = min,start + batch, bound, clamp at the generation boundary
     lea ebx, [eax + edi]                      ; ebx = start + batch
     cmp ebx, edx                              ; batch end beyond the generation's job count?
     cmova ebx, edx                            ; yes, clamp: never spill into another generation's jobs
     cmp eax, ebx                              ; start >= end?, pure overshoot: zero real jobs in this claim
     jae .steal_loop                           ; yes, nothing to process, re-loop, peek or gen check exits
%ifdef PROFILE_MODE
     inc wp_fa                                 ; fa++ only on successful steal, profiling builds only
%endif
     sub ebx, eax                              ; batch_count = end - start, strictly positive after the clamp
     add wp_count, ebx                         ; charge the whole claimed batch once; the lock sub at steal_done publishes it only after the batch has run
     JOB_ADDR r15, rax                         ; rsi = r15 + start * JOB_SIZE, never crosses a job boundary
     align 32                                  ; Align the common job loop
.job_loop:
     call process_job                          ; process one job; preserves rsi and nonvolatile ebx
     JOB_NEXT rsi                              ; advance to next 64-byte job
     dec ebx                                   ; one fewer job remains in the claimed batch
     jnz .job_loop                             ; process until the strictly-positive batch count reaches zero
     jmp .steal_loop                           ; batch complete: attempt another claim
.steal_done:
%ifdef ZYDIS_TRACE
     INT3_DBG_STR "worker: steal_done"         ; debug: job steal phase complete
%endif
%ifdef PROFILE_MODE
; Record instrumentation, PROFILE
     mov eax, r12d                             ; id
     lea r11, [rel worker_fetch_adds]          ; base of per-worker fetch_add counters
     shl rax, 6                                ; rax = id * 64, cache-line stride
     mov rsi, wp_fa                            ; fa, from stack slot
     add qword [r11 + rax], rsi                ; worker_fetch_adds[id] += fa
; Completed-job totals, the dump derives total_jobs / workers_active from these
     mov esi, wp_count                         ; jobs completed this dispatch, dword load zero-extends
     lea r11, [rel worker_jobs_done]           ; base of per-worker completed-job totals
     add rsi, [r11 + rax]                      ; new total = old total + this dispatch's count, rax = id*64 still live
     mov [r11 + rax], rsi                      ; worker_jobs_done[id] += wp_count, PROFILE
%endif
; Batch decrement remaining_jobs by wp_count, single lock sub instead of per-job
     mov eax, wp_count                         ; load batch job count
     test eax, eax                             ; count == 0?
     jz .sd_skip_dec                           ; skip if zero
     lock sub dword [remaining_jobs], eax      ; batch decrement
.sd_skip_dec:
     mov eax, [gen_counter]                    ; load current gen_counter
     xor ebx, ebx                              ; ebx = 0 spins, came from processing, not a wait, keep PROFILE wake_spins honest
     cmp eax, r13d                             ; gen changed since we captured my_gen?
     jne .gen_changed                          ; yes, new dispatch available, re-steal immediately
     jmp .worker_outer                         ; no change, go back to waiting
.worker_exit:
     xor eax, eax                              ; return 0, success
     vzeroupper                                ; clear dirty YMM upper bits from kernel
     lock dec dword [workers_alive]            ; paired with the inc at entry; only SF_END's register pops + ret follow, so shutdown reaching 0 proves no memory writes or calls remain, the dec lands before RtlExitUserThread takes the loader lock, so a DllMain-side poll cannot deadlock
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore non-volatiles and return
; ============================================================================
