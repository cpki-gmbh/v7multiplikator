; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%macro DISPATCH_EXECUTE 3
; ====== Save job count to non-volatile register BEFORE any overwrites ======
     mov r15d, %1                              ; r15d = total job count, non-volatile, preserved across API calls
     JOB_BASE                                  ; Inspect the published job family
     JOB_LOAD_MODE eax                         ; All jobs in this dispatch use one mode
     sub eax, MATMUL_Q8_BATCH4                 ; Three prefill panel job families are contiguous
     cmp eax, MATMUL_QKV_POST - MATMUL_Q8_BATCH4 ; Test the contiguous prefill job range
     setbe byte [rel prefill_unit_claim]       ; Panels claim one substantial job at a time
     mov dword [remaining_jobs], r15d          ; remaining_jobs = count
     mov eax, [gen_counter]                    ; current generation, no other writer can interleave here
     inc eax                                   ; new generation tag, equals gen_counter after the lock inc below
     shl eax, JOB_TAG_SHIFT                    ; eax = tag << 16 with local cursor 0
     lea edx, [rax + r15]                      ; edx = tag << 16 | count, disjoint bit fields, one instruction
     or eax, 1                                 ; reserve job 0 for the caller: the cursor is published at 1, so no worker can ever claim index 0
     mov dword [num_jobs], edx                 ; publish tagged num_jobs first, claim validation reads it after the cursor tag
     mov dword [next_job], eax                 ; publish tagged cursor, local cursor starts at 1 with job 0 reserved for the caller
     lock inc dword [gen_counter]              ; atomic gen_counter++ with full barrier, signal LAST
     cmp dword [workers_may_sleep], 0          ; any worker registered in the wait-capable path?
     je %%de_publish_wake_done                 ; no: spinning workers see gen_counter directly
     lea rcx, [rel gen_counter]                ; yes, signal waiters so they see new gen
     API_CALL [__imp_WakeByAddressAll], rcx    ; publish-signal: wake registered sleepers
%%de_publish_wake_done:
; ====== Caller's reserved job 0, then the reassign-work loop ======
; The caller thread participates in work reassignment alongside pool threads,
; and job 0 is reserved for it: the cursor was published at 1, so the caller
; always processes at least one job no matter how fast the workers claim.
     mov dword %2, 1                           ; caller_count starts at 1: job 0 belongs to the caller
     JOB_BASE                                  ; rsi = &jobs_arr[0], the reserved job index is known zero
     call process_job                          ; process the reserved job, preserves rsi
; CALLER_SELECT_BATCH picks batch size from INI thresholds.
; r15d already holds total job count, set at macro entry, preserved across API calls .
     CALLER_SELECT_BATCH r15d, r14d            ; r14d = caller_batch, non-volatile
     cmp byte [rel prefill_unit_claim], 0      ; Keep shape tuning for ordinary dispatches
     je %%de_batch_ready                       ; Ordinary dispatches retain their tuned batch
     mov r14d, 1                               ; Avoid reserving an entire panel on one core
%%de_batch_ready:
%%de_caller_steal:
     movzx eax, word [next_job]                ; one load/zero-extend of this generation's cursor
     cmp eax, r15d                             ; have all jobs already been claimed?
     jae %%de_caller_steal_done                ; yes: avoid a redundant locked RMW
     mov eax, r14d                             ; batch for fetch_add
     lock xadd dword [next_job], eax           ; eax = old next_job, tagged; only our own tag can appear, no foreign publishes
     movzx eax, ax                             ; eax = local start index, tag stripped, bits 16..31 zeroed
     cmp eax, r15d                             ; start >= total_jobs?
     jae %%de_caller_steal_done                ; no more work
     lea edx, [eax + r14d]                     ; edx = start + batch, fused arithmetic
     cmp edx, r15d                             ; end > total_jobs?
     cmova edx, r15d                           ; clamp to total_jobs
     sub edx, eax                              ; edx = end - start = batch_count, always >= 1
     add dword %2, edx                         ; caller_count += batch_count once per claim
     mov ebx, edx                              ; ebx = jobs remaining in this batch, nonvolatile across process_job
     JOB_ADDR_X rsi, rax                       ; rsi = &jobs_arr[start], rax overwritten, ebx preserved
%%de_caller_job_loop:
     call process_job                          ; process one job; preserves rsi and nonvolatile ebx
     JOB_NEXT rsi                              ; advance to next 64-byte job
     dec ebx                                   ; one fewer job remains in the claimed batch
     jnz %%de_caller_job_loop                  ; repeat until the count reaches zero; no per-job end load
     jmp %%de_caller_steal                     ; batch complete: claim another batch
%%de_caller_steal_done:
%ifdef PROFILE_MODE
     mov eax, dword %2                         ; jobs actually processed by this dispatching caller
     add qword [rel caller_jobs_done], rax     ; make caller_jobs_done reflect completed caller jobs
%else
%ifdef CPU0_DIAG
     mov eax, dword %2                         ; jobs actually processed by this dispatching caller
     add qword [rel caller_jobs_done], rax     ; CPU0_DIAG-only builds need the same counter the profile dump reads
%endif
%endif
; Batch decrement remaining_jobs by caller_count, at least 1: job 0 is reserved
; for the caller, so caller_count can never be zero here
     mov eax, dword %2                         ; load caller job count
     lock sub dword [remaining_jobs], eax      ; dec remaining by caller count
%%de_wait:
     cmp dword [remaining_jobs], 0             ; remaining_jobs vs zero?
     jne %%de_wait_spin                        ; positive, jobs still outstanding
     jmp %%de_all_done                         ; exactly zero, all jobs done. REQUIRED EDGE, not a layout jump: the zero case arrives by falling off the jne above, so without this jmp it would fall into %%de_wait_spin and spin forever on a dispatch that is already complete
%%de_wait_spin:
     pause                                     ; SMT-friendly spin hint
     jmp %%de_wait                             ; re-check
; ====== Dispatch complete ======
%%de_all_done:
%ifdef CPU0_DIAG
     cmp dword [cpu0_diag_pending], 0          ; one-time session dump still owed?
     je %%de_diag_done                         ; no, straight to the dispatcher's own epilogue
     mov dword [cpu0_diag_pending], 0          ; consume the one-shot, this session reports once
     call cpu0_diag_dump                       ; print identity + affinity + caller-job counters, after work
%%de_diag_done:
%endif
%endmacro
; ============================================================================
; JOB QUEUE ACCESS MACROS

; ALL reads and writes to jobs_arr[] must go through these macros.
; No raw arithmetic, shl reg,6 / lea reg,[rel jobs_arr] / add reg,reg allowed.

; Write macros take only the source value; the job pointer is always rsi.

; Read/LEA macros read from rsi, the current job pointer, set by DISPATCH_EXECUTE
; or the pool thread reassignment loop .

; ============================================================================

%macro JOB_ADDR 1-2
%if %0 = 1
     mov r11, %1                               ; copy index to scratch, 64-bit
     shl r11, JOB_SIZE_SHL                     ; index * JOB_SIZE
     lea rsi, [rel jobs_arr]                   ; rsi = jobs_arr base
     add rsi, r11                              ; rsi = &jobs_arr[index]
%elif %0 = 2
     mov rsi, %2                               ; copy index to rsi
     shl rsi, JOB_SIZE_SHL                     ; rsi = index * JOB_SIZE
     add rsi, %1                               ; rsi = base + index*JOB_SIZE
%endif
%endmacro

%macro JOB_ADDR_X 2
     shl %2, JOB_SIZE_SHL                      ; %2 = index * JOB_SIZE
     lea %1, [rel jobs_arr]                    ; %1 = jobs_arr base
     add %1, %2                                ; %1 = &jobs_arr[index]
%endmacro

%macro JOB_BASE 0
     lea rsi, [rel jobs_arr]                   ; rsi = jobs_arr base, = &jobs_arr[0]
%endmacro

%macro JOB_NEXT 1
     add %1, JOB_SIZE                          ; reg += 64: next 64-byte job entry
%endmacro

%macro JOB_SET_OUTPUT 1
     mov qword [rsi + JOB_OUTPUT], %1          ; set output pointer in job descriptor
%endmacro

%macro JOB_SET_INPUT_Q 1
     mov qword [rsi + JOB_INPUT_Q], %1         ; set input Q pointer in job descriptor
%endmacro

%macro JOB_SET_INPUT_S 1
     mov qword [rsi + JOB_INPUT_S], %1         ; set input S pointer in job descriptor
%endmacro

%macro JOB_SET_WEIGHT_Q 1
     mov qword [rsi + JOB_WEIGHT_Q], %1        ; set weight Q pointer in job descriptor
%endmacro

%macro JOB_SET_WEIGHT_S 1
     mov qword [rsi + JOB_WEIGHT_S], %1        ; set weight S pointer in job descriptor
%endmacro

%macro JOB_SET_INPUT_ELEMENTS 1
     mov dword [rsi + JOB_INPUT_ELEMENTS], %1  ; set input elements count in job descriptor
%endmacro

%macro JOB_SET_COLUMN_OFFSET 1
     mov dword [rsi + JOB_COLUMN_OFFSET], %1   ; set column offset in job descriptor
%endmacro

%macro JOB_SET_FEATURE_DIMENSION 1
     mov dword [rsi + JOB_FEATURE_DIMENSION], %1 ; set feature dimension in job descriptor
%endmacro

%macro JOB_SET_MODE 1
     mov dword [rsi + JOB_MODE], %1            ; set matmul mode in job descriptor
%endmacro

%macro JOB_SET_SCALE 1
     mov dword [rsi + JOB_SCALE], %1           ; set scale in job descriptor
%endmacro
; ------ Read macros, load from rsi job pointer ------

%macro JOB_LOAD_OUTPUT 1
     mov %1, [rsi + JOB_OUTPUT]                ; load output pointer from job descriptor
%endmacro

%macro JOB_LOAD_INPUT_Q 1
     mov %1, [rsi + JOB_INPUT_Q]               ; load input Q pointer from job descriptor
%endmacro

%macro JOB_LOAD_INPUT_S 1
     mov %1, [rsi + JOB_INPUT_S]               ; load input S pointer from job descriptor
%endmacro

%macro JOB_LOAD_WEIGHT_Q 1
     mov %1, [rsi + JOB_WEIGHT_Q]              ; load weight Q pointer from job descriptor
%endmacro

%macro JOB_LOAD_INPUT_ELEMENTS 1
     mov %1, [rsi + JOB_INPUT_ELEMENTS]        ; load input elements count from job descriptor
%endmacro

%macro JOB_LOAD_COLUMN_OFFSET 1
     mov %1, [rsi + JOB_COLUMN_OFFSET]         ; load column offset from job descriptor
%endmacro

%macro JOB_LOAD_FEATURE_DIMENSION 1
     mov %1, [rsi + JOB_FEATURE_DIMENSION]     ; load feature dimension from job descriptor
%endmacro

%macro JOB_LOAD_MODE 1
     mov %1, [rsi + JOB_MODE]                  ; load matmul mode from job descriptor
%endmacro

%macro JOB_LOAD_SCALE 1
     mov %1, [rsi + JOB_SCALE]                 ; load scale from job descriptor
%endmacro
; ------ LEA macros, address of field within job, for zero-copy tensor method ------

%macro JOB_LEA_INPUT_Q 1
     lea %1, [rsi + JOB_INPUT_Q]               ; get address of input Q field in job descriptor
%endmacro

%macro JOB_LEA_WEIGHT_Q 1
     lea %1, [rsi + JOB_WEIGHT_Q]              ; get address of weight Q field in job descriptor
%endmacro
