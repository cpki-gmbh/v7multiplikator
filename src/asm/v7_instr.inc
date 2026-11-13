; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .text
     default rel                               ; use RIP-relative addressing by default for this module
     STK_ARG5 equ 32                           ; 5th Win64 stack argument slot
     STK_ARG6 equ 40                           ; 6th Win64 stack argument slot
     STK_ARG7 equ 48                           ; 7th Win64 stack argument slot
     DEC_BUF_LAST_OFF equ 20                   ; Last byte offset in 21-byte decimal buffer

; Writer contracts:
; - uint64_to_str is a naked leaf: no SF_START frame, zero calls, and it writes no
; non-volatile register. The WRITE_UINT64 macro that calls it owns the shadow space,
; the alignment and the non-volatile preservation.
; - v7s_shutdown bumps gen_counter atomically, lock inc dword, never
; with a non-atomic read-modify-write.
; - v7s_shutdown leaves shutdown_flag SET on exit so any still-running
; workers can see it; the reset happens via the init_once_flag clear, which
; forces re-initialization on next call.
; ============================================================================
uint64_to_str:
     lea r10, [rdx + DEC_BUF_LAST_OFF]         ; r10 = write cursor, end of buffer - 1
     mov byte [r10], 0                         ; null terminate
     mov rax, rcx                              ; rax = value to convert
     mov r9, 0xCCCCCCCCCCCCCCCD                ; r9 = magic multiplicative inverse of 10
.next_digit:
     dec r10                                   ; move cursor left
     mov r8, rax                               ; r8 = copy of current quotient, for remainder
     mul r9                                    ; rdx = high 64 bits of, rax * magic
     shr rdx, 3                                ; rdx = quotient = value / 10, via magic inverse
     mov rax, rdx                              ; rax = new quotient for next iteration
     imul rdx, rdx, 10                         ; rdx = quotient * 10, 64-bit: exact, no truncation caveat, the quotient is at most 2^64/10 so the product cannot overflow
     sub r8, rdx                               ; r8 = original_value -, quotient*10 = remainder, exact, 0..9
     add r8b, '0'                              ; convert remainder to ASCII digit
     mov [r10], r8b                            ; store digit
     test rax, rax                             ; quotient == 0?, FULL 64-bit test: mov rax, rdx above does NOT zero-extend, a quotient >= 2^32 with a zero low dword, e.g. 42949672965, truncated the output
     jnz .next_digit                           ; more digits to convert
     mov rax, r10                              ; return pointer to first digit
     ret                                       ; return to caller
%ifdef PROFILE_MODE
     SF_START v7s_dump_instrumentation, rbx, rsi, rdi, r12, r13, r14, r15 ; begin stack frame for instrumentation dump
     SF_COMMIT                                 ; commit stack frame for instrumentation dump
     call debug_ensure_std                     ; ensure stdout_handle is initialized
; Separator
     WRITE_STR lbl_sep                         ; print separator line
     WRITE_NL                                  ; print newline after separator
     WRITE_STR lbl_num_workers                 ; print "num_workers:" label
     mov r12d, [num_workers]                   ; pre-load dword, zero-extends to r12
     WRITE_UINT64 r12                          ; print num_workers value
     WRITE_NL                                  ; print newline after num_workers
     WRITE_STR lbl_gen_counter                 ; print "gen_counter:" label
     mov r12d, [gen_counter]                   ; pre-load dword, zero-extends to r12
     WRITE_UINT64 r12                          ; print gen_counter value
     WRITE_NL                                  ; print newline after gen_counter
     WRITE_STR lbl_remaining_jobs              ; print "remaining_jobs:" label
     mov r12d, [remaining_jobs]                ; pre-load dword, zero-extends to r12
     WRITE_UINT64 r12                          ; print remaining_jobs value
     WRITE_NL                                  ; print newline after remaining_jobs
     WRITE_STR lbl_next_job                    ; print "next_job:" label
     movzx r12d, word [next_job]               ; strip the generation tag, display the local cursor position only, zero-extension of the low half
     WRITE_UINT64 r12                          ; print next_job local cursor
     WRITE_NL                                  ; print newline after next_job
; Per-worker affinity masks
     WRITE_STR lbl_worker_aff                  ; print "worker_affinity:" label
     mov r14d, [num_workers]                   ; active worker count, loop bound, r14: not touched until the totals xor below
     xor r15d, r15d                            ; i = 0
     test r14d, r14d                           ; pre-test: a label at the TOP of the body runs it once before any cmp, so with num_workers == 0 this emitted a phantom affinity mask
     jz .aff_done                              ; no workers: no iterations, matches .sum_loop's own pre-test below
.aff_loop:
     lea r11, [rel worker_aff]                 ; base of affinity array
     mov r12, [r11 + r15*QWORD_SCALE]          ; pre-load into non-volatile r12, survives WRITE_UINT64's debug_ensure_std call
     WRITE_UINT64 r12                          ; pass via register, safe, r12 is non-volatile
     WRITE_CHR ' '                             ; print space between affinity masks
     inc r15d                                  ; i++
     cmp r15d, r14d                            ; i < num_workers?, active slots only
     jl .aff_loop                              ; continue loop
.aff_done:
     WRITE_NL                                  ; print newline after affinity section
; Per-worker detail line: W<id>: jobs=N fa=N spins=N sleeps=N aff=0xABC
     mov r14d, [num_workers]                   ; active worker count, loop bound in r14: ebx holds wake_spins[i] inside
     xor r15d, r15d                            ; i = 0
     test r14d, r14d                           ; pre-test, for the same reason as .aff_loop: without it a zero-worker dump printed a complete W0: line for a worker that does not exist, contradicting the workers_active: 0 that .sum_loop reports
     jz .detail_done                           ; no workers: no iterations
.detail_loop:
; Print "W" prefix and worker id
     WRITE_CHR 'W'                             ; print "W" prefix for worker detail line
     WRITE_UINT64 r15                          ; print worker id, r15 zero-extended by the 32-bit loop counter ops
     WRITE_CHR ':'                             ; print colon separator after worker id
; jobs_done for this worker
     mov rax, r15                              ; i, zero-extended from r15d loop counter
     shl rax, 6                                ; i * 64, cache-line stride
     lea rcx, [rel worker_jobs_done]           ; rcx = base of worker_jobs_done, RIP-relative
     mov r12, [rcx + rax]                      ; r12 = jobs_done[i], non-volatile
     lea rcx, [rel worker_fetch_adds]          ; rcx = base of worker_fetch_adds
     mov r13, [rcx + rax]                      ; r13 = fetch_adds[i], non-volatile
     lea rcx, [rel worker_wake_spins]          ; rcx = base of worker_wake_spins
     mov rbx, [rcx + rax]                      ; rbx = wake_spins[i], non-volatile
     lea rcx, [rel worker_sleeps]              ; rcx = base of worker_sleeps
     mov rsi, [rcx + rax]                      ; rsi = sleeps[i], non-volatile
     WRITE_UINT64 r12                          ; jobs_done, safe, non-volatile
     WRITE_CHR ' '                             ; print space before fetch_adds value
     WRITE_UINT64 r13                          ; fetch_adds, safe, non-volatile
     WRITE_CHR ' '                             ; print space before wake_spins value
     WRITE_UINT64 rbx                          ; wake_spins, safe, non-volatile
     WRITE_CHR ' '                             ; print space before sleeps value
     WRITE_UINT64 rsi                          ; sleeps, safe, non-volatile
; newline
     WRITE_NL                                  ; print newline after worker detail line
     inc r15d                                  ; i++
     cmp r15d, r14d                            ; i < num_workers?, active slots only
     jl .detail_loop                           ; continue loop
.detail_done:
; Per-worker sticky peak of jobs_done: the high-water mark over every period of
; this session, so a worker that got jobs in ANY dispatch type shows nonzero
     WRITE_STR lbl_peak                        ; print "worker_jobs_peak:" label
     mov r14d, [num_workers]                   ; active worker count, loop bound
     xor r15d, r15d                            ; i = 0
     test r14d, r14d                           ; pre-test, same reason as .aff_loop
     jz .peak_done                             ; no workers: no iterations
.peak_loop:
     lea r11, [rel worker_jobs_peak]           ; base of the peak array, reloaded per iteration, WRITE_* clobbers r11
     mov rax, r15                              ; i, zero-extended from r15d loop counter
     shl rax, 6                                ; rax = i * 64, cache-line stride, the merge stores at the same stride
     mov r12, [r11 + rax]                      ; pre-load into non-volatile r12, survives the WRITE_* calls
     WRITE_UINT64 r12                          ; print the peak value
     WRITE_CHR ' '                             ; space separator between peaks
     inc r15d                                  ; i++
     cmp r15d, r14d                            ; i < num_workers?, active slots only
     jl .peak_loop                             ; continue loop
.peak_done:
     WRITE_NL                                  ; end the peak line
; Compute totals and workers_active
     xor r12d, r12d                            ; total_worker jobs
     xor r13d, r13d                            ; total_fetch_adds
     xor r14d, r14d                            ; total_sleeps
     xor ebx, ebx                              ; workers_active
     cmp dword [num_workers], 0                ; any workers?
     jle .sum_done                             ; no, skip summation
     xor r15d, r15d                            ; i
     lea rdi, [rel worker_jobs_done]           ; base of per-worker jobs_done
     lea r9, [rel worker_fetch_adds]           ; base of per-worker fetch_adds
     lea r10, [rel worker_sleeps]              ; base of per-worker sleeps
.sum_loop:
     mov rax, r15                              ; i, zero-extended from r15d loop counter
     shl rax, 6                                ; rax = i * 64, cache-line stride
     mov rcx, [rdi + rax]                      ; worker_jobs_done[i]
     add r12, rcx                              ; total_jobs += worker_jobs_done[i]
     test rcx, rcx                             ; worker_jobs_done[i] == 0?, replaces slow jrcxz
     jz .sum_skip_active                       ; skip active count if zero
     inc ebx                                   ; workers_active++
.sum_skip_active:
     add r13, [r9 + rax]                       ; total_fetch_adds += worker_fetch_adds[i]
     add r14, [r10 + rax]                      ; total_sleeps += worker_sleeps[i]
     inc r15d                                  ; i++
     cmp r15d, [num_workers]                   ; i < num_workers?, same bound as .aff_loop/.detail_loop, the per-dispatch reset only zeroes i < num_workers, so slots above that keep stale counts
     jl .sum_loop                              ; next worker
.sum_done:
     add r12, [caller_jobs_done]               ; total_jobs += caller_jobs_done
; Print summary lines
     WRITE_STR lbl_sep                         ; print separator before summary
     WRITE_NL                                  ; print newline after separator
     WRITE_STR lbl_total_jobs                  ; print "total_jobs:" label
     WRITE_UINT64 r12                          ; print total jobs value
     WRITE_NL                                  ; print newline after total_jobs
     WRITE_STR lbl_caller                      ; print "caller_jobs_done:" label
     WRITE_UINT64 [caller_jobs_done]           ; print caller jobs value
     WRITE_NL                                  ; print newline after caller_jobs_done
     WRITE_STR lbl_caller_total                ; print "caller_total:" label
     WRITE_UINT64 [caller_jobs_total]          ; sticky caller total across all dispatch types and periods
     WRITE_NL                                  ; print newline after caller total
     WRITE_STR lbl_workers_active              ; print "workers_active:" label
     WRITE_UINT64 rbx                          ; print workers active count
     WRITE_NL                                  ; newline after workers_active
     WRITE_STR lbl_total_sleeps                ; print "total_sleeps:" label
     WRITE_UINT64 r14                          ; print total sleeps value
     WRITE_NL                                  ; newline after total_sleeps
     WRITE_STR lbl_caller_spins                ; print "caller_spins_wait:" label
     WRITE_UINT64 [caller_spins_wait]          ; print caller spins value
     WRITE_NL                                  ; newline after caller_spins
     WRITE_STR lbl_total_fetch_adds            ; print "total_fetch_adds:" label
     WRITE_UINT64 r13                          ; print total fetch-adds value
     WRITE_NL                                  ; newline after total_fetch_adds
; Debug: core detection diagnostics, dword BSS symbols pre-loaded into
; non-volatile r12, which survives WRITE_UINT64's internal calls
     WRITE_STR lbl_dbg_glpiex_len              ; print "dbg_glpiex_len:" label
     mov r12d, [debug_glpiex_length]           ; pre-load dword, zero-extends to r12
     WRITE_UINT64 r12                          ; print glpiex length value
     WRITE_NL                                  ; newline after glpiex_len
     WRITE_STR lbl_dbg_core_map_count          ; print "dbg_core_map_count:" label
     mov r12d, [debug_core_map_count]          ; pre-load dword, zero-extends to r12
     WRITE_UINT64 r12                          ; print core map count value
     WRITE_NL                                  ; newline after core_map_count
     WRITE_STR lbl_dbg_use_phys_cores          ; print "dbg_use_phys_cores:" label
     mov r12d, [debug_use_phys_cores]          ; pre-load dword, zero-extends to r12
     WRITE_UINT64 r12                          ; print use_phys_cores value
     WRITE_NL                                  ; newline after use_phys_cores
     WRITE_STR lbl_dbg_cores_subtract          ; print "dbg_cores_subtract:" label
     mov r12d, [debug_cores_subtract]          ; pre-load dword, zero-extends to r12
     WRITE_UINT64 r12                          ; print cores_subtract value
     WRITE_NL                                  ; newline after cores_subtract
     WRITE_STR lbl_dbg_r14_workers             ; print "dbg_r14_workers:" label
     mov r12d, [debug_worker_count_raw]        ; pre-load dword, zero-extends to r12
     WRITE_UINT64 r12                          ; print worker count raw value
     WRITE_NL                                  ; newline after worker_count_raw
     WRITE_STR lbl_dbg_r13_total_cores         ; print "dbg_r13_total_cores:" label
     mov r12d, [debug_r13_at_workers]          ; pre-load dword, zero-extends to r12
     WRITE_UINT64 r12                          ; print r13 at workers value
     WRITE_NL                                  ; newline after r13_total_cores
     WRITE_STR lbl_dbg_num_workers_calc        ; print "dbg_num_workers_calc:" label
     mov r12d, [debug_num_workers_computed]    ; pre-load dword, zero-extends to r12
     WRITE_UINT64 r12                          ; print num workers computed value
     WRITE_NL                                  ; newline after num_workers_calc
; Dump param_set_0, 24 dwords
     WRITE_STR lbl_dbg_param_set_0             ; print "dbg_param_set_0:" label
     lea r14, [rel debug_param_set_0]          ; base address for indexed access
     xor r15d, r15d                            ; i = 0
.dbg_ps_loop:
     mov r12d, [r14 + r15*DWORD_SCALE]         ; pre-load dword into r12, zero-extended, survives WRITE_UINT64 call
     WRITE_UINT64 r12                          ; print param_set_0[i] value
     WRITE_CHR ' '                             ; print space separator
     inc r15d                                  ; i++
     cmp r15d, PARAM_SET_DWORDS                ; i < 24?, bottom-test: no per-iteration jmp; PARAM_SET_DWORDS is a non-zero constant so the do-while runs at least one pass
     jl .dbg_ps_loop                           ; next element
.dbg_ps_done:
     WRITE_NL                                  ; newline after param_set_0 dump
; DUMP_ARRAY for each per-worker instrumentation array
     DUMP_ARRAY fmt_instr_worker_jobs, worker_jobs_done ; dump worker jobs done array
     DUMP_ARRAY fmt_instr_worker_fa, worker_fetch_adds ; dump worker fetch-adds array
     DUMP_ARRAY fmt_instr_worker_spins, worker_wake_spins ; dump worker wake spins array
     DUMP_ARRAY fmt_instr_worker_sleeps, worker_sleeps ; dump worker sleeps array
; Final separator
     WRITE_STR lbl_sep                         ; print final separator line
     WRITE_NL                                  ; newline after separator
     vzeroupper                                ; clear dirty YMM upper bits
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore non-volatiles and return
%endif
; ============================================================================
; EXPORT: v7s_shutdown

; Stack frame, rbp-relative :
; rbp+8 = return address
; rbp+0 = saved rbp
; rbp-8 = saved r12
; rbp-16 = saved r13
; rbp-24 = saved r15

; Register usage:
; r12 = threads base pointer
; r13 = num_workers captured at close-loop entry, before the reset zeroes it
; r15 = loop counter
; ============================================================================
     SF_START v7s_shutdown, r12, r13, r15      ; function prologue: save non-volatiles
     SF_COMMIT                                 ; commit stack frame for v7s_shutdown
%ifdef PROFILE_MODE
     cmp dword [num_workers], 0                ; pool still up? later idempotent calls and the DllMain path must not re-dump
     je .sd_dump_done                          ; no, already shut down, skip the diagnostic
     call v7s_dump_instrumentation             ; diagnostic builds: print the last dispatch's per-worker/caller distribution while the pool state is still live
.sd_dump_done:
%endif
; Set shutdown flag, workers check this at .worker_outer loop entry
     mov dword [shutdown_flag], 1              ; arm the exit flag, every worker observes it on its next outer-loop pass
; Atomically increment gen_counter and wake workers twice, caller sleeps 1s after this
; Using lock inc to avoid read-modify-write race with live worker threads
     lock inc dword [gen_counter]              ; atomic increment, release to workers
     lea rcx, [rel gen_counter]                ; RIP-relative for ASLR-safe address load
     API_CALL [__imp_WakeByAddressAll], rcx    ; wake all sleeping workers for shutdown, first wake
     lea rcx, [rel gen_counter]                ; reload the address, API_CALL overwritten rcx
     API_CALL [__imp_WakeByAddressAll], rcx    ; wake again to ensure workers see gen_counter change, second wake
; Wait until every worker has DECREMENTED workers_alive to 0: each worker's dec
; is its last DLL action, before RtlExitUserThread, so zero means no thread can
; execute DLL code any more. Sleep+poll only, waiting on thread handles here
; would deadlock under the loader lock when called from DllMain,DETACH, because
; a worker's exit path must acquire that lock. Budget 2s, 2x the old blind 1s
; sleep ; on timeout proceed exactly as before.
     mov r15d, 2000                            ; poll budget in 1ms ticks
.shutdown_wait_alive:
     cmp dword [workers_alive], 0              ; all workers past their last DLL action?
     jz .wait_alive_done                       ; yes, observed termination
     mov ecx, 1                                ; Sleep,1ms : releases the core, holds no lock
     API_CALL [__imp_Sleep], rcx               ; overwrites volatile regs only, rcx reloaded each pass
     dec r15d                                  ; one tick consumed
     jnz .shutdown_wait_alive                  ; keep polling until the budget runs out
.wait_alive_done:
; Close worker thread handles: leaving them open would leak one full set per
; shutdown->re-init cycle. CloseHandle does NOT terminate the threads, the OS keeps running threads alive ; workers
; still exit via shutdown_flag and the poll above OBSERVED their termination.
     mov r13d, [num_workers]                   ; capture count BEFORE the reset below zeroes it
     lea r12, [rel threads]                    ; r12 = threads array base, safe to compute at count 0, nothing dereferences it
     xor r15d, r15d                            ; i = 0
.close_handles:
     cmp r15d, r13d                            ; i < num_workers?, also the num_workers == 0 case: 0 >= 0 exits here, no separate pre-test needed
     jge .handles_done                         ; no, done
     mov rcx, [r12 + r15*QWORD_SCALE]          ; handle = threads[i]
     test rcx, rcx                             ; slot populated?
     jz .close_next                            ; partial-init slot, post-FATAL destroy -> skip
     API_CALL [__imp_CloseHandle], rcx         ; close this worker handle
     mov qword [r12 + r15*QWORD_SCALE], 0      ; zero slot: fresh state for re-init
.close_next:
     inc r15d                                  ; i++
     jmp .close_handles                        ; next handle
.handles_done:
; Reset state for potential re-initialization
     xor eax, eax                              ; zero source for the four re-init stores
     mov dword [init_ok], eax                  ; clear init success flag, pure write avoids RMW stall
     sfence                                    ; ensure init_ok=0 is globally visible before clearing init_once_flag
     mov dword [init_once_flag], eax           ; allow re-initialization, ordered after init_ok clear, pure write avoids RMW stall
     mov dword [num_workers], eax              ; reset worker count, pure write avoids RMW stall
     mov dword [always_spin], eax              ; new workers must start idle, not inherit a stale active phase
     call unpin_calling_thread                 ; restore caller thread affinity, pinned during init_affinity
     vzeroupper                                ; clear dirty YMM upper bits
     SF_END r12, r13, r15                      ; restore non-volatiles and return
process_job:
     INT3_DBG_STR "pj:entry"                   ; debug: entering process_job
     sub rsp, 56                               ; 32 shadow + stack args 5/6/7 at +32/+40/+48 = exactly 56, NO padding: entry RSP is 8, mod 16 and 56 is 8, mod 16, so RSP lands on 0, mod 16 for every call below. Do NOT round this to 64 or trim it to 48, both are 0, mod 16 and would misalign every kernel call
; Point directly into job struct via JOB_LOAD_* / JOB_LEA_* macros
     JOB_LOAD_OUTPUT rcx                       ; xout, float* result pointer
     JOB_LEA_INPUT_Q rdx                       ; &input_tensor, zero-copy: directly in job struct
     JOB_LEA_WEIGHT_Q r8                       ; &weight_tensor, zero-copy: directly in job struct
     JOB_LOAD_INPUT_ELEMENTS r9d               ; n, number of input elements
     JOB_LOAD_FEATURE_DIMENSION eax            ; d, feature dimension, zero-extended to rax
     mov [rsp + STK_ARG5], rax                 ; 5th arg = d
     JOB_LOAD_COLUMN_OFFSET eax                ; column offset, zero-extended to rax
     mov [rsp + STK_ARG6], rax                 ; 6th arg = column_offset, for fused kernels
     JOB_LOAD_MODE eax                         ; load job mode once
     cmp eax, MATMUL_Q8                        ; Q8 mode, quantized activations ?
     je .call_q8                               ; yes -> call Q8 kernel, hottest path
     cmp eax, MATMUL_F32                       ; F32 mode?
     je .call_f32                              ; yes -> call F32 kernel, 2nd hottest
     cmp eax, MATMUL_SWIGLU_Q8                 ; Q8 SwiGLU fused mode?
     je .call_swiglu_q8                        ; yes -> call Q8 SwiGLU kernel
     cmp eax, MATMUL_SWIGLU                    ; SwiGLU fused mode?
     je .call_swiglu                           ; yes -> call SwiGLU kernel, gate+up+SiLU
     cmp eax, MATMUL_RMSNORM_ROPE              ; RMSNorm+RoPE per-head mode?
     je .call_rmsnorm_rope                     ; yes -> call per-head kernel
     cmp eax, MATMUL_ATTENTION                 ; Attention mode, single-head ?
     je .call_attention                        ; yes -> call v7s_attention_inner kernel
     cmp eax, MATMUL_FUSED_F32                 ; Fused F32 mode?
     je .call_fused_f32                        ; yes -> call fused F32 kernel, matmul + rmsnorm + rope
     cmp eax, MATMUL_QUANTIZE                  ; Quantize mode?
     je .call_quantize                         ; yes -> call quantize kernel
     cmp eax, MATMUL_ROPE_ONLY                 ; RoPE-only mode, no RMSNorm ?
     je .call_rope_only                        ; yes -> call per-head RoPE-only kernel
     cmp eax, MATMUL_MEMCPY                    ; Memcpy mode?
     je .call_memcpy                           ; yes -> call memcpy kernel, rep movsb
     cmp eax, MATMUL_Q8_BATCH4                 ; Experimental four-token Q8 batch mode?
     je .call_q8_batch4                        ; yes -> call the batch kernel on this job's row range
     cmp eax, MATMUL_VIT_LINEAR                ; ViT pure F32 linear?
     je .call_vit_linear                       ; yes -> call ViT linear kernel
     cmp eax, MATMUL_VIT_ATTENTION             ; ViT bidirectional attention?
     je .call_vit_attention                    ; yes -> call ViT attention kernel
     cmp eax, MATMUL_ATTN_TOKEN                ; Prefill attention token?
     je .call_attention_token                  ; Process this token's independent head range
     cmp eax, MATMUL_QKV_POST                  ; QKV prefill postprocessing token?
     je .call_qkv_post                         ; Apply bias, RoPE, and cache packing
; every mode above is written only by the engine's own dispatchers, the
; compare chain covers all of them, and a non-job value here is outside the
; contract, undefined behaviour, no default handler
.call_f32:
     call v7s_f32_chunk                        ; F32 kernel, float activations, quantized weights
     jmp .pj_exit                              ; shared exit, saves duplicate add rsp,ret vs inline
.call_q8:
     call v7s_q8_chunk                         ; Q8 kernel, quantized activations, quantized weights
     jmp .pj_exit                              ; shared exit, MANDATORY: fall-through would enter .call_fused_f32 with Q8 args
.call_fused_f32:
     INT3_DBG_STR "pj:call_fused"              ; debug: about to call fused F32 kernel
     call v7s_fused_f32_chunk                  ; Fused F32 kernel, matmul + rmsnorm + rope
     INT3_DBG_STR "pj:fused_ret"               ; debug: returned from fused F32 kernel
     jmp .pj_exit                              ; goto shared epilogue
.call_swiglu:
     call v7s_swiglu_chunk                     ; SwiGLU kernel, gate+up matmul + SiLU fusion
     jmp .pj_exit                              ; goto shared epilogue
.call_swiglu_q8:
     call v7s_swiglu_q8_chunk                  ; Q8 SwiGLU kernel, Q8xQ8 gate+up + SiLU
     jmp .pj_exit                              ; goto shared epilogue
.call_memcpy:
; Memcpy kernel: rcx=dst, rdx=src, r8d=byte_count, 3-arg Win64 ABI
; Always uses ERMS rep movsb, v7s_memcpy_chunk ??" single kernel path
     JOB_LOAD_INPUT_Q rdx                      ; arg2: src pointer
     mov r8d, r9d                              ; arg3: byte_count for this slice, r9d already holds JOB_INPUT_ELEMENTS from the prologue load, nothing on this path wrote it
     call v7s_memcpy_chunk                     ; ERMS kernel, only path
     jmp .pj_exit                              ; goto shared epilogue
.call_q8_batch4:
; Four-token Q8 batch kernel on this job's contiguous output-row range.
; JOB_OUTPUT holds the FIRST row index of the range (an integer, not a
; pointer), JOB_INPUT_ELEMENTS holds the row count, and q8b_args_ptr (BSS,
; published by v7d_q8_batch4 before DISPATCH_EXECUTE) points at the whole
; matrix's v7_q8_batch4_args block: out[4], xq[4], xs[4], wq, ws, n, d.
; Every worker sees all four input vectors and streams its own disjoint
; weight-row range once -- the whole point of the mode.
; The kernel takes rcx = pointer to a 120-byte args block, so build a
; job-local SLICE block on this stack frame: copy the whole block, then
; advance the pointers to the row range and shrink d to the row count.
%define Q8B_SLICE_OUT0 0                       ; slice block offset, output pointer token zero
%define Q8B_SLICE_OUT1 8                       ; slice block offset, output pointer token one
%define Q8B_SLICE_OUT2 16                      ; slice block offset, output pointer token two
%define Q8B_SLICE_OUT3 24                      ; slice block offset, output pointer token three
%define Q8B_SLICE_WQ 96                        ; slice block offset, weight byte-array base
%define Q8B_SLICE_WS 104                       ; slice block offset, weight scale-array base
%define Q8B_SLICE_N 112                        ; slice block offset, elements per input
%define Q8B_SLICE_D 116                        ; slice block offset, row count of this slice
%define Q8B_SLICE_FRAME 128                    ; stack frame for the slice block, 120 rounded to 16
%define Q8B_SLICE_NEXT Q8_BATCH_ARGS_BYTES     ; Next descriptor pointer in the slice padding
%define Q8B_COPY_QWORDS 15                     ; 15 qwords = 120 bytes, the whole args block
%define Q8B_SLICE_FLOAT 4                      ; byte stride of one float in an output array
%define Q8B_QWORD 8                            ; byte stride of one qword slot in the args block
%define Q8B_SHADOW 32                          ; Win64 caller shadow space for the kernel call, kept separate from the slice block
     add rsp, -Q8B_SLICE_FRAME                 ; slice args block, imm8 encoding of sub rsp, 128; post-sub rsp stays 0 mod 16, kernel call ABI-correct
     mov r11, [rel q8b_args_ptr]               ; r11 = whole-matrix args block, valid for the whole blocking dispatch
.q8b_tile:
     lea rax, [r11 + Q8_BATCH_ARGS_BYTES]      ; Locate the next tile descriptor
     mov [rsp + Q8B_SLICE_NEXT], rax           ; Preserve the descriptor cursor across the leaf
     xor r10d, r10d                            ; qword index 0..14, counted loop: every slot including out[0] must be copied, a dec-first do-while would skip slot 0 and leave stack garbage in out[0]
.q8b_copy_loop:
     mov rax, [r11 + r10*Q8B_QWORD]            ; one qword of the whole args block, 8-byte slot stride
     mov [rsp + r10*Q8B_QWORD], rax            ; copy into the job-local slice block
     inc r10d                                  ; next qword
     cmp r10d, Q8B_COPY_QWORDS                 ; all 15 qwords copied?
     jb .q8b_copy_loop                         ; continue until the whole block is copied
     JOB_LOAD_OUTPUT rax                       ; rax = first row index of this job's range
     JOB_LOAD_INPUT_ELEMENTS r9d               ; Reload the row count after the preceding tile
     mov [rsp + Q8B_SLICE_D], r9d              ; slice d = row count, r9d already holds JOB_INPUT_ELEMENTS from the prologue load, nothing on this path wrote it
     mov rdx, [rsp + Q8B_SLICE_OUT0]           ; out[0]
     lea rdx, [rdx + rax*Q8B_SLICE_FLOAT]      ; out[0] + first floats
     mov [rsp + Q8B_SLICE_OUT0], rdx           ; store the advanced output pointer
     mov rdx, [rsp + Q8B_SLICE_OUT1]           ; out[1]
     lea rdx, [rdx + rax*Q8B_SLICE_FLOAT]      ; out[1] + first floats
     mov [rsp + Q8B_SLICE_OUT1], rdx           ; store the advanced output pointer
     mov rdx, [rsp + Q8B_SLICE_OUT2]           ; out[2]
     lea rdx, [rdx + rax*Q8B_SLICE_FLOAT]      ; out[2] + first floats
     mov [rsp + Q8B_SLICE_OUT2], rdx           ; store the advanced output pointer
     mov rdx, [rsp + Q8B_SLICE_OUT3]           ; out[3]
     lea rdx, [rdx + rax*Q8B_SLICE_FLOAT]      ; out[3] + first floats
     mov [rsp + Q8B_SLICE_OUT3], rdx           ; store the advanced output pointer
     mov rdx, [rsp + Q8B_SLICE_WQ]             ; wq base
     mov r8d, [rsp + Q8B_SLICE_N]              ; n, elements per input, multiple of 64
     imul r8, rax                              ; r8 = first * n, weight bytes to skip
     add rdx, r8                               ; wq + first*n = this range's first weight row
     mov [rsp + Q8B_SLICE_WQ], rdx             ; store the advanced weight pointer
     mov rdx, [rsp + Q8B_SLICE_WS]             ; ws base
     shr r8, 6                                 ; (first*n)/64 = first * (n/64) weight scales to skip, exact: n is a multiple of 64
     lea rdx, [rdx + r8*Q8B_SLICE_FLOAT]       ; ws + first*(n/64) floats
     mov [rsp + Q8B_SLICE_WS], rdx             ; store the advanced scale pointer
     mov rcx, rsp                              ; arg1 = slice args block, job-local, valid until the kernel returns
     sub rsp, Q8B_SHADOW                       ; reserve the kernel's shadow space below the slice block, so a shadow write can never touch out[0..3]
     cmp dword [rel q8b_input_packed], 0       ; Are magnitudes precomputed for this panel?
     jne .q8b_packed                           ; Select the packed-input leaf
     call v7s_q8_batch4                        ; Ordinary signed input rows
     jmp .q8b_join                             ; Finish the tile
.q8b_packed:
     call v7s_q8_batch4_packed                 ; Reuse the precomputed activation magnitudes
.q8b_join:
     add rsp, Q8B_SHADOW                       ; Release the kernel shadow space
     mov r11, [rsp + Q8B_SLICE_NEXT]           ; Retrieve the next tile descriptor
     cmp r11, [rel q8b_args_end]               ; Are there more tiles for this weight slice?
     jb .q8b_tile                              ; Reuse the same weights while they are in cache
     add rsp, Q8B_SLICE_FRAME                  ; Release the local slice block
     jmp .pj_exit                              ; goto shared epilogue
%undef Q8B_SLICE_OUT0
%undef Q8B_SLICE_OUT1
%undef Q8B_SLICE_OUT2
%undef Q8B_SLICE_OUT3
%undef Q8B_SLICE_WQ
%undef Q8B_SLICE_WS
%undef Q8B_SLICE_N
%undef Q8B_SLICE_D
%undef Q8B_SLICE_FRAME
%undef Q8B_SLICE_NEXT
%undef Q8B_COPY_QWORDS
%undef Q8B_SLICE_FLOAT
%undef Q8B_QWORD
.call_quantize:
     JOB_LOAD_INPUT_Q rdx                      ; arg2: in_x, float* input
     JOB_LOAD_INPUT_S r8                       ; arg3: out_s, float* scales base
     call v7s_quantize_chunk                   ; execute quantize kernel for this slice
     jmp .pj_exit                              ; goto shared epilogue
.call_attention:
     mov edx, r9d                              ; arg2: h, head index, r9d already holds JOB_INPUT_ELEMENTS from the prologue load, nothing on this path wrote it
     JOB_LOAD_INPUT_Q r8                       ; arg3: p, Config*
     JOB_LOAD_FEATURE_DIMENSION r9d            ; arg4: pos
     JOB_LOAD_INPUT_S rax                      ; loff, uint64_t
     mov [rsp + STK_ARG5], rax                 ; arg5: loff
     JOB_LOAD_SCALE eax                        ; kv_mul
     mov [rsp + STK_ARG7], rax                 ; arg7: kv_mul
     INT3_DBG_STR "attn:"                      ; debug: entering attention kernel
     INT3_DBG_REG 1                            ; dump RCX = s
     INT3_DBG_REG 2                            ; dump RDX = h
     INT3_DBG_REG 8                            ; dump R8 = p
     INT3_DBG_REG 9                            ; dump R9D = pos
     call v7s_attention_inner                  ; call fused attention kernel directly, compiled in
     jmp .pj_exit                              ; goto shared epilogue
.call_attention_token:
     mov edx, 1                                ; Select the entire range, whose split point is zero
     call v7s_attention_fork_range             ; rcx points to the immutable token descriptor
     jmp .pj_exit                              ; Return through the common worker epilogue
.call_qkv_post:
     call v7s_pq4_fork_member                  ; rcx points to the private token descriptor
     jmp .pj_exit                              ; Return through the common worker epilogue
.call_rmsnorm_rope:
     JOB_LOAD_COLUMN_OFFSET edx                ; arg2: head_idx, int
     call v7s_rmsnorm_rope_head                ; call per-head RMSNorm+RoPE kernel
     jmp .pj_exit                              ; goto shared epilogue
.call_rope_only:
     JOB_LOAD_COLUMN_OFFSET edx                ; arg2: head_idx, int
     call v7s_rope_only_head                   ; call per-head RoPE-only kernel, no RMSNorm
     jmp .pj_exit                              ; goto shared epilogue
.call_vit_linear:
     JOB_LOAD_INPUT_Q rdx                      ; arg2: in float*
     JOB_LOAD_WEIGHT_Q r8                      ; arg3: w float*
     mov eax, r9d                              ; n_rows, r9d already holds JOB_INPUT_ELEMENTS from the prologue load, nothing on this path wrote it before the r9 load below
     mov [rsp + STK_ARG5], rax                 ; arg5: n_rows, override
     JOB_LOAD_INPUT_S r9                       ; arg4: bias float*, or NULL, r9 free again after the n_rows store
; STK_ARG6 already = column_offset = col_start
     JOB_LOAD_FEATURE_DIMENSION eax            ; chunk_d
     mov [rsp + STK_ARG7], rax                 ; arg7: chunk_d
     call v7s_vit_linear_chunk                 ; execute dispatched F32 linear matmul
     jmp .pj_exit                              ; goto shared epilogue
.call_vit_attention:
     JOB_LOAD_INPUT_Q rdx                      ; arg2: qkv float*
     JOB_LOAD_INPUT_S r8                       ; arg3: scores float*
     JOB_LOAD_COLUMN_OFFSET eax                ; h, head index
     mov [rsp + STK_ARG5], rax                 ; arg5: h
     call v7s_vit_attention_chunk              ; execute per-head bidirectional attention
.pj_exit:
     add rsp, 56                               ; tear down stack frame
; ---- %undef cleanup: file-local names leave scope here ----
%undef STK_ARG7                                ; 7th Win64 stack argument slot
%undef STK_ARG6                                ; 6th Win64 stack argument slot
%undef STK_ARG5                                ; 5th Win64 stack argument slot
%undef DEC_BUF_LAST_OFF                        ; last byte offset in the 21-byte decimal buffer
     ret                                       ; return to caller, steal loop
