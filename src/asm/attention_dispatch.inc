; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .text
; Caller stack-arg homes, Win64 SF frame: incoming arg 5 sits at rbp+48 for the
; two dispatch entries below. S_XB / S_Q are RunState field offsets from
; inf_structs.inc, the two fields v7d_attention_prefill repoints per token and
; restores at .ap_restore.
%define ATTN_ARG_KV_DIM 48                     ; rbp+48 = kv_dim, 5th caller arg
%define ATTN_ARG_KV_MUL 56                     ; rbp+56 = kv_mul, 6th caller arg
; v7d_attention_prefill, caller stack args, rbp frame
%define PREFILL_ARG_START_POS 48               ; rbp+48 = start_pos, 5th caller arg
%define PREFILL_ARG_N_TOKENS 56                ; rbp+56 = n_tokens, 6th caller arg
%define PREFILL_ARG_LOFF 64                    ; rbp+64 = loff, uint64_t, 7th caller arg
%define PREFILL_ARG_KV_DIM 72                  ; rbp+72 = kv_dim, 8th caller arg
%define PREFILL_ARG_KV_MUL 80                  ; rbp+80 = kv_mul, 9th caller arg
     global v7d_attention                      ; export attention dispatch entry point
     SF_START v7d_attention, rbx, rsi, rdi, r12, r13, r14, r15 ; declare stack frame, r14/r15 REQUIRED: DISPATCH_EXECUTE uses them internally
     SF_DWORD am_caller_count                  ; caller job count
     SF_DWORD am_caller_end                    ; end bound for caller job loop
     SF_ARRAY am_fork_desc, attention_fork_args_size ; stack-local descriptor, read-only during FORK
     SF_COMMIT                                 ; commit stack frame, allocate shadow
; Save entry args into non-volatile registers, preserved across initialize_once
; Budget: rdi=s, r13=p, r15d=pos, r14=loff, r12d=n_heads, set below
     mov rdi, rcx                              ; s, RunState*
     mov r13, rdx                              ; p, Config*
     mov r15d, r8d                             ; pos, int
     mov r14, r9                               ; loff, uint64_t
; kv_dim / kv_mul stay in the caller stack-arg homes, read at fill setup
; Init thread pool, runs once via CAS check
     call initialize_once                      ; ensures attention DLL is loaded
     SET_FTZ_DAZ                               ; set flush-to-zero/denormals-are-zero on caller thread
; Load n_heads from Config* p
     mov r12d, [r13 + CFG_N_HEADS]             ; r12d = n_heads, fill-loop compare bound, non-volatile
; Short sequences: compare single-core, two-core fork, and normal full-pool dispatch.
; The fork has no disable switch and no asleep-worker bypass: the forward
; entry's v7s_set_spin(1) guarantees a fork-receptive pool for the whole call.
     cmp r15d, 16                              ; positions 0..15 retain caller-only execution
     jb .attention_direct                      ; keep the original direct path
     cmp r15d, 96                              ; 16..96 use the fork, when spinning
     ja .attention_regular_cutoff              ; outside the trial window: original policy
     cmp r12d, 2                               ; split requires at least two independent heads
     jb .attention_direct                      ; single head: caller-only
     jmp .attention_fork                       ; enter the fork
.attention_regular_cutoff:
; ====== One job per head, 0..n_heads-1, fanned out across the pool thread pool ======
; Each job runs the single-head kernel, v7s_attention_inner for its own head index.
     xor ebx, ebx                              ; h = 0, the head loop counter
     JOB_BASE                                  ; rsi = &jobs_arr[0], the index is known zero
; Hoist loop-invariant job fill constants into volatile regs, saves 4 movs per head
     mov r8d, dword [rbp + ATTN_ARG_KV_DIM]    ; kv_dim, caller stack-arg home, stable
     mov r9d, r15d                             ; pos, from non-volatile
     mov r10d, dword [rbp + ATTN_ARG_KV_MUL]   ; kv_mul, caller stack-arg home, stable
     mov r11d, MATMUL_ATTENTION                ; mode, AFTER JOB_ADDR, the 1-arg form uses r11 as scratch
     align 32
.fill_loop:
; Fill job fields via JOB_SET_* macros, rsi already points to current job
; Do-while form: n_heads >= 1 for every validated model, init bound, see above
     JOB_SET_OUTPUT rdi                        ; JOB_OUTPUT = RunState* s, cached in rdi
     JOB_SET_INPUT_Q r13                       ; JOB_INPUT_Q = Config* p, cached in r13
     JOB_SET_INPUT_S r14                       ; JOB_INPUT_S = loff, cached in r14
     JOB_SET_INPUT_ELEMENTS ebx                ; JOB_INPUT_ELEMENTS = h, head index
     JOB_SET_COLUMN_OFFSET r8d                 ; JOB_COLUMN_OFFSET = kv_dim, hoisted in r8d
     JOB_SET_FEATURE_DIMENSION r9d             ; JOB_FEATURE_DIMENSION = pos, hoisted in r9d
     JOB_SET_SCALE r10d                        ; JOB_SCALE = kv_mul, hoisted in r10d
     JOB_SET_MODE r11d                         ; JOB_MODE = MATMUL_ATTENTION, hoisted in r11d
     JOB_NEXT rsi                              ; advance to next job, the one permitted spelling
     inc rbx                                   ; h++
     cmp ebx, r12d                             ; h < n_heads?, bound hoisted into r12d
     jl .fill_loop                             ; next head, not taken on last iteration
     DISPATCH_EXECUTE r12d, am_caller_count, am_caller_end ; publish n_heads jobs, count direct from non-volatile r12d, reassign work + wait
     jmp .attention_done                       ; skip the direct head loop
.attention_fork:
     lea r11, am_fork_desc                     ; put all six attention arguments in a read-only block
     mov [r11 + attention_fork_args.state], rdi ; RunState* s
     mov [r11 + attention_fork_args.config], r13 ; Config* p
     mov [r11 + attention_fork_args.loff], r14 ; layer KV offset
     mov [r11 + attention_fork_args.pos], r15d ; sequence position
     mov eax, dword [rbp + ATTN_ARG_KV_DIM]    ; load kv_dim, caller stack-arg home
     mov [r11 + attention_fork_args.kv_dim], eax ; store kv_dim
     mov eax, dword [rbp + ATTN_ARG_KV_MUL]    ; load kv_mul, caller stack-arg home
     mov [r11 + attention_fork_args.kv_mul], eax ; store kv_mul
     mov [r11 + attention_fork_args.heads], r12d ; total head count
     mov eax, r12d                             ; eax = n_heads for the split
     shr eax, 1                                ; eax = floor(n_heads/2)
     mov [r11 + attention_fork_args.half], eax ; store the split point
     lea rcx, am_fork_desc                     ; rcx = descriptor
     xor edx, edx                              ; worker 0 receives first-half selector
     FORK 0                                    ; parent ZF=1, worker 0 ZF=0
     jz .attention_fork_parent                 ; parent continues with the second half
; W0: first half of heads, no ordinary dispatcher while active.
     call v7s_attention_fork_range             ; heads [0, half) on worker 0
     CONTINUE_DISPATCH 0                       ; return worker 0 to its spin loop
.attention_fork_parent:
     lea rcx, am_fork_desc                     ; rcx = descriptor
     mov edx, 1                                ; caller: second half
     call v7s_attention_fork_range             ; heads [half, heads) on the caller
     FORK_WAIT 0                               ; join worker 0 before the next regular dispatch
     jmp .attention_done                       ; both halves done
.attention_direct:
     mov [rsp + APICALL_ARG5], r14             ; arg5 = loff, constant across all query heads
     mov eax, dword [rbp + ATTN_ARG_KV_DIM]    ; load kv_dim, head-loop invariant
     mov [rsp + APICALL_ARG6], rax             ; arg6 = kv_dim
     mov eax, dword [rbp + ATTN_ARG_KV_MUL]    ; load kv_mul, head-loop invariant
     mov [rsp + APICALL_ARG7], rax             ; arg7 = kv_mul
     xor ebx, ebx                              ; h = 0
.attention_direct_head:
     mov rcx, rdi                              ; arg1 = RunState* s
     mov edx, ebx                              ; arg2 = query head index
     mov r8, r13                               ; arg3 = Config* p
     mov r9d, r15d                             ; arg4 = pos
     call v7s_attention_inner                  ; identical single-head kernel as process_job
     inc ebx                                   ; h++
     cmp ebx, r12d                             ; all heads processed?
     jb .attention_direct_head                 ; no, process next head
.attention_done:
     vzeroupper                                ; clear dirty YMM upper bits
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore saved regs and release frame, r15 mandatory, DISPATCH_EXECUTE overwrites it
     global v7d_attention_prefill              ; export prefill attention dispatch
     SF_START v7d_attention_prefill, rbx, rsi, rdi, r12, r13, r14, r15 ; declare stack frame
     SF_QWORD ap_s                             ; RunState* s
     SF_QWORD ap_p                             ; Config* p
     SF_QWORD ap_q_storage                     ; float* q_storage
     SF_QWORD ap_att_out                       ; float* att_out_storage
     SF_DWORD ap_start_pos                     ; int start_pos
     SF_DWORD ap_n_tokens                      ; int n_tokens
     SF_QWORD ap_loff                          ; uint64_t loff
     SF_DWORD ap_kv_dim                        ; int kv_dim
     SF_DWORD ap_kv_mul                        ; int kv_mul
     SF_QWORD ap_orig_xb                       ; original s->xb, restored after the token loop
     SF_QWORD ap_orig_q                        ; original s->q, restored after the token loop
     SF_COMMIT                                 ; commit stack frame, allocate shadow
; Save entry args
     mov ap_s, rcx                             ; s, RunState*
     mov ap_p, rdx                             ; p, Config*
     mov ap_q_storage, r8                      ; q_storage, float*
     mov ap_att_out, r9                        ; att_out_storage, float*
; Load stack args 5-9
     mov eax, [rbp + PREFILL_ARG_START_POS]    ; start_pos from stack arg 5
     mov ap_start_pos, eax                     ; save start position for token offset computation
     mov eax, [rbp + PREFILL_ARG_N_TOKENS]     ; n_tokens from stack arg 6
     mov ap_n_tokens, eax                      ; save token count for loop bound
     mov rax, [rbp + PREFILL_ARG_LOFF]         ; loff from stack arg 7
     mov ap_loff, rax                          ; save layer offset for KV cache addressing
     mov eax, [rbp + PREFILL_ARG_KV_DIM]       ; kv_dim from stack arg 8
     mov ap_kv_dim, eax                        ; save kv_dim for attention kernel
     mov eax, [rbp + PREFILL_ARG_KV_MUL]       ; kv_mul from stack arg 9
     mov ap_kv_mul, eax                        ; save kv_mul for GQA head mapping
; Save original s->xb and s->q to frame slots, restored after the token loop
     mov rax, ap_s                             ; rax = RunState* s
     mov rcx, [rax + S_XB]                     ; rcx = original s->xb
     mov ap_orig_xb, rcx                       ; save to frame slot
     mov rcx, [rax + S_Q]                      ; rcx = original s->q
     mov ap_orig_q, rcx                        ; save to frame slot
; Cache loop-invariants in free non-volatiles, preserved across v7d_attention calls
; Budget: r12=s, rdi=q_storage, rbx=p, rsi=running att_out, r13d=t, r14=offset, r15=stride
     mov r12, ap_s                             ; r12 = RunState* s
     mov rdi, ap_q_storage                     ; rdi = q_storage base
     mov rbx, ap_p                             ; rbx = Config* p
     mov rsi, ap_att_out                       ; rsi = running att_out pointer
     mov eax, [rbx + CFG_N_HEADS]              ; eax = p->n_heads, zero-extends to rax
     mov ecx, [rbx + CFG_HEAD_DIM]             ; ecx = p->head_dim, zero-extends to rcx
     imul rax, rcx                             ; rax = n_heads*head_dim, all_heads_dim, 64-bit product
     shl rax, 2                                ; byte stride per token
     mov r15, rax                              ; r15 = stride, full 64-bit
     xor r14d, r14d                            ; r14 = running byte offset, token t
; Token loop: for t = 0 to n_tokens-1, do-while form: n_tokens >= 1 for every prefill call, the prompt always carries at least one token; a zero-token prefill is out of contract
     xor r13d, r13d                            ; t = 0, in non-volatile r13d, ABI-preserved by SF_START
     sub rsp, 48                               ; allocate shadow + stack args for v7d_attention outside loop, hoisted
; Hoist loop-invariant stack args, kv_dim, kv_mul out of the token loop
     mov eax, ap_kv_dim                        ; load kv_dim
     mov [rsp + APICALL_ARG5], rax             ; arg5: kv_dim
     mov eax, ap_kv_mul                        ; load kv_mul
     mov [rsp + APICALL_ARG6], rax             ; arg6: kv_mul
.ap_token_loop:
; Repoint s->q = q_storage + running offset
     lea rcx, [rdi + r14]                      ; rcx = q_storage + byte_offset
     mov [r12 + S_Q], rcx                      ; s->q = this token's Q vector
; Repoint s->xb = running att_out pointer, rsi
     mov [r12 + S_XB], rsi                     ; s->xb = this token's attention output slot
; Call v7d_attention,s, p, start_pos + t, loff, kv_dim, kv_mul, stack args pre-staged
     mov rcx, r12                              ; arg1: s
     mov rdx, rbx                              ; arg2: p, hoisted into a non-volatile before the loop
     mov r8d, ap_start_pos                     ; arg3 base position
     add r8d, r13d                             ; pos = start_pos + t
     mov r9, ap_loff                           ; arg4: loff
     call v7d_attention                        ; dispatch per-head attention for this token
; Advance to next token
     inc r13d                                  ; t++
     add r14, r15                              ; byte offset += stride
     add rsi, r15                              ; att_out pointer += stride, running
     cmp r13d, ap_n_tokens                     ; t < n_tokens?
     jb .ap_token_loop                         ; next token
.ap_restore:
     add rsp, 48                               ; tear down shadow + stack args allocation
     mov rcx, ap_orig_xb                       ; rcx = original s->xb
     mov [r12 + S_XB], rcx                     ; s->xb restored
     mov rcx, ap_orig_q                        ; rcx = original s->q
     mov [r12 + S_Q], rcx                      ; s->q restored
     vzeroupper                                ; clear dirty YMM upper bits before return
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore saved regs and release frame
; Q8 prefill has already populated every K/V row in this causal span. Each
; token job reads that cache and writes its own attention output.
%define AP_TOKEN_JOBS 16                       ; Bounded stack panel of independent token jobs
%define AP_STATE_BYTES (S_VALUE_CACHE + 8)     ; RunState prefix through its last attention field
%define AP_TOKEN_BYTES (AP_STATE_BYTES + attention_fork_args_size) ; State and immutable arguments
     SF_START v7d_q8_attention_panel, rbx, rsi, rdi, r12, r13, r14, r15 ; Preserve dispatcher registers
     SF_QWORD aq_state                         ; Original RunState
     SF_QWORD aq_config                        ; Model configuration
     SF_QWORD aq_q                             ; Next query row to dispatch
     SF_QWORD aq_out                           ; Next output row to dispatch
     SF_QWORD aq_stride                        ; Bytes per query or output row
     SF_QWORD aq_loff                          ; Layer offset in the KV cache
     SF_DWORD aq_pos                           ; Absolute position of the next query
     SF_DWORD aq_remaining                     ; Number of tokens not yet dispatched
     SF_DWORD aq_kv_dim                        ; KV width
     SF_DWORD aq_kv_mul                        ; Query heads per KV head
     SF_DWORD aq_jobs                          ; Job count of this panel
     SF_DWORD aq_caller_count                  ; Caller job count
     SF_DWORD aq_caller_end                    ; Caller steal loop bound
     SF_ARRAY aq_desc, AP_TOKEN_JOBS * AP_TOKEN_BYTES ; Private state and descriptor for each job
     SF_COMMIT                                 ; Allocate fixed Win64 frame and shadow space
     mov aq_state, rcx                         ; Record the source state
     mov aq_config, rdx                        ; Record the model configuration
     mov aq_q, r8                              ; Record the first query row
     mov aq_out, r9                            ; Record the first output row
     mov eax, [rbp + PREFILL_ARG_START_POS]    ; Read the first absolute position
     mov aq_pos, eax                           ; Initialize the position cursor
     mov eax, [rbp + PREFILL_ARG_N_TOKENS]     ; Read the prompt token count
     mov aq_remaining, eax                     ; Initialize the token countdown
     mov rax, [rbp + PREFILL_ARG_LOFF]         ; Read the layer's KV offset
     mov aq_loff, rax                          ; Save the layer offset
     mov eax, [rbp + PREFILL_ARG_KV_DIM]       ; Read the KV width
     mov aq_kv_dim, eax                        ; Save the KV width
     mov eax, [rbp + PREFILL_ARG_KV_MUL]       ; Read the group size
     mov aq_kv_mul, eax                        ; Save the group size
     mov eax, [rdx + CFG_N_HEADS]              ; Query heads per row
     mov ecx, [rdx + CFG_HEAD_DIM]             ; Elements per query head
     imul rax, rcx                             ; Elements per query row
     shl rax, 2                                ; Convert the row stride to FP32 bytes
     mov aq_stride, rax                        ; Save the query and output stride
     SET_FTZ_DAZ                               ; Match the caller's regular attention dispatch
.panel:
     mov eax, aq_remaining                     ; Remaining tokens
     mov ecx, AP_TOKEN_JOBS                    ; Maximum jobs in this panel
     cmp eax, ecx                              ; Can every remaining token fit?
     cmovbe ecx, eax                           ; Limit the final panel to the remainder
     mov aq_jobs, ecx                          ; Save the number of jobs to dispatch
     mov r12d, ecx                             ; Number of descriptors left to initialize
     lea rdi, aq_desc                          ; First token's private RunState prefix
     JOB_BASE                                  ; First worker job slot
.fill:
     lea rcx, [rdi + AP_STATE_BYTES]           ; Arguments immediately follow the state prefix
     mov rax, aq_state                         ; Read the shared source RunState
     mov r8, [rax + S_KEY_CACHE]               ; Cache pointer is shared and read-only
     mov [rdi + S_KEY_CACHE], r8               ; Set private state's key cache pointer
     mov r8, [rax + S_VALUE_CACHE]             ; Value cache is shared and read-only
     mov [rdi + S_VALUE_CACHE], r8             ; Set private state's value cache pointer
     mov r8, aq_q                              ; Read the current token's query
     mov [rdi + S_Q], r8                       ; Set private state's query pointer
     mov r8, aq_out                            ; Read the current token's output slot
     mov [rdi + S_XB], r8                      ; Set private state's output pointer
     mov [rcx + attention_fork_args.state], rdi ; Use this job's private state
     mov rax, aq_config                        ; Read the immutable model configuration
     mov [rcx + attention_fork_args.config], rax ; Set configuration pointer
     mov r8d, [rax + CFG_N_HEADS]              ; This job owns every query head
     mov [rcx + attention_fork_args.heads], r8d ; Set the exclusive head bound
     mov dword [rcx + attention_fork_args.half], 0 ; Upper range starts at head zero
     mov rax, aq_loff                          ; Read the layer's cache offset
     mov [rcx + attention_fork_args.loff], rax ; Set the layer offset
     mov eax, aq_pos                           ; Read this query's absolute position
     mov [rcx + attention_fork_args.pos], eax  ; Keep the causal bound for this token
     mov eax, aq_kv_dim                        ; Read the KV row width
     mov [rcx + attention_fork_args.kv_dim], eax ; Set the KV row width
     mov eax, aq_kv_mul                        ; Read the query head group size
     mov [rcx + attention_fork_args.kv_mul], eax ; Set the query head group size
     JOB_SET_OUTPUT rcx                        ; Publish the private token descriptor
     mov eax, MATMUL_ATTN_TOKEN                ; Select the token-range attention kernel
     JOB_SET_MODE eax                          ; Stage the job mode
     JOB_NEXT rsi                              ; Advance to the next job slot
     add rdi, AP_TOKEN_BYTES                   ; Advance to the next descriptor
     mov rax, aq_stride                        ; Read the common row stride
     add aq_q, rax                             ; Advance the query pointer
     add aq_out, rax                           ; Advance the output pointer
     inc dword aq_pos                          ; Advance the absolute position
     dec r12d                                  ; Count down the descriptors
     jnz .fill                                 ; Prepare the next job
     mov eax, aq_jobs                          ; Load this panel's job count
     DISPATCH_EXECUTE eax, aq_caller_count, aq_caller_end ; Wait for the panel before reusing its state
     mov eax, aq_jobs                          ; Count the completed tokens
     sub aq_remaining, eax                     ; Remove the completed panel
     jnz .panel                                ; Dispatch another panel if needed
     vzeroupper                                ; Clean upper lanes for the caller
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; Restore the caller's preserved registers
%undef AP_TOKEN_JOBS
%undef AP_STATE_BYTES
%undef AP_TOKEN_BYTES
%undef aq_state
%undef aq_state_raw
%undef aq_config
%undef aq_config_raw
%undef aq_q
%undef aq_q_raw
%undef aq_out
%undef aq_out_raw
%undef aq_stride
%undef aq_stride_raw
%undef aq_loff
%undef aq_loff_raw
%undef aq_pos
%undef aq_pos_raw
%undef aq_remaining
%undef aq_remaining_raw
%undef aq_kv_dim
%undef aq_kv_dim_raw
%undef aq_kv_mul
%undef aq_kv_mul_raw
%undef aq_jobs
%undef aq_jobs_raw
%undef aq_caller_count
%undef aq_caller_count_raw
%undef aq_caller_end
%undef aq_caller_end_raw
%undef aq_desc
; %undef cleanup: file-local defines leave scope here --
%undef ATTN_ARG_KV_DIM                         ; caller stack arg 5, kv_dim
%undef ATTN_ARG_KV_MUL                         ; caller stack arg 6, kv_mul
%undef PREFILL_ARG_START_POS                   ; caller stack arg 5, start_pos
%undef PREFILL_ARG_N_TOKENS                    ; caller stack arg 6, n_tokens
%undef PREFILL_ARG_LOFF                        ; caller stack arg 7, loff
%undef PREFILL_ARG_KV_DIM                      ; caller stack arg 8, kv_dim
%undef PREFILL_ARG_KV_MUL                      ; caller stack arg 9, kv_mul
