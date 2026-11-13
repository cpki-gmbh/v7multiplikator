; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; prefill_q8_fork.inc -- batch-4 QKV Stage-C postprocessing, one independent
; descriptor per tile member. Requires qk_fork_kernel.inc and the FORK macro.
; This kernel does not use the shared rmsnorm_* BSS parameter globals.

; The four packed QKV outputs remain available until all forked members join.
; Members 0..2 process K/V in their own packed block. Member 3 copies K/V
; into the existing shared FP32 K/V scratch, preserving the original s->k/v
; values and pointers for callers after this tile.

     struc pq4_fork_args                       ; per-member descriptor, one per tile token
.out:
     resq 1                                    ; packed QKV output for this token
.q_dst:
     resq 1                                    ; persistent float Q row
.key_cache:
     resq 1                                    ; packed FP16 key cache destination row
.value_cache:
     resq 1                                    ; packed FP16 value cache destination row
.q_bias:
     resq 1                                    ; per-layer Q bias
.k_bias:
     resq 1                                    ; per-layer K bias
.v_bias:
     resq 1                                    ; per-layer V bias
.q_weight:
     resq 1                                    ; per-layer Q head norm weights
.k_weight:
     resq 1                                    ; per-layer K head norm weights
.cos:
     resq 1                                    ; RoPE cos row, position-adjusted
.sin:
     resq 1                                    ; RoPE sin row, position-adjusted
.all_heads:
     resd 1                                    ; Q float count
.kv_dim:
     resd 1                                    ; K and V float count each
.head_dim:
     resd 1                                    ; per-head dimension
.n_heads:
     resd 1                                    ; number of Q heads
.n_kv_heads:
     resd 1                                    ; number of K heads
.pos:
     resd 1                                    ; M-RoPE-adjusted position
.bias_active:
     resd 1                                    ; no-op when model has no QKV bias
.last_member:
     resd 1                                    ; member 3 writes final K/V staging row
.padding:
     resq 1                                    ; 128-byte stride for simple indexing
     endstruc                                  ; defines pq4_fork_args_size
%if pq4_fork_args_size != 128
%error pq4_fork_args must remain 128 bytes     ; layout guard: descriptor stride is shl 7 at the call site
%endif

%define PQ4F_FLOAT_BYTES 4                     ; sizeof float, SIB scale for packed-block addressing

; v7s_pq4_fork_member(pq4_fork_args*), Win64 RCX.
; Every invocation owns one Q row, one QKV scratch row and one KV cache row.
; Only the last member accesses the shared FP32 K/V scratch, on the caller.
     section .text
     SF_START v7s_pq4_fork_member, rbx, rsi, rdi, r12, r13 ; prologue: save the 5 non-volatiles the body writes
     SF_ARRAY pq4f_q_head, qk_fork_args_size   ; thread-private Q head descriptor
     SF_ARRAY pq4f_k_head, qk_fork_args_size   ; thread-private K head descriptor
     SF_COMMIT                                 ; finalize stack frame layout
     mov rbx, rcx                              ; immutable per-member descriptor
; Scatter Q into persistent q_storage. RSI then points directly to K input.
     mov rsi, [rbx + pq4_fork_args.out]        ; rsi = packed QKV block, Q floats first
     mov rdi, [rbx + pq4_fork_args.q_dst]      ; rdi = persistent Q row destination
     mov ecx, [rbx + pq4_fork_args.all_heads]  ; ecx = all_heads_dim floats
     shl rcx, 2                                ; rcx = Q byte count
     rep movsb                                 ; Q block copied, rsi now at K
     mov r12, rsi                              ; K FP32 source in packed block
     mov eax, [rbx + pq4_fork_args.kv_dim]     ; eax = kv_dim floats
     lea r13, [r12 + rax*PQ4F_FLOAT_BYTES]     ; V FP32 source in packed block
     cmp dword [rbx + pq4_fork_args.last_member], 0 ; caller's token 3?
     je .private_kv                            ; members 0..2 keep their private K/V
; Last member: preserve the existing S_K/S_V FP32 state in g_inf_kv_scratch.
; One 2*kv_dim copy replaces the old per-member copies into shared scratch.
     mov rdi, [rel g_inf_kv_scratch]           ; rdi = shared K/V FP32 staging base
     mov ecx, [rbx + pq4_fork_args.kv_dim]     ; ecx = kv_dim
     shl rcx, 3                                ; two * kv_dim * sizeof(float)
     rep movsb                                 ; source is already at the K block
     mov r12, [rel g_inf_kv_scratch]           ; r12 = K row now lives in shared scratch
     mov eax, [rbx + pq4_fork_args.kv_dim]     ; eax = kv_dim
     lea r13, [r12 + rax*PQ4F_FLOAT_BYTES]     ; r13 = V row in shared scratch
.private_kv:
; Q bias, identical PQ_BIAS_ADD AVX2 arithmetic to the serial stage-C path.
     cmp dword [rbx + pq4_fork_args.bias_active], 0 ; bias block contains real data?
     je .q_bias_done                           ; no, skip the add loop entirely
     mov rdx, [rbx + pq4_fork_args.q_dst]      ; rdx = Q row destination
     mov rax, [rbx + pq4_fork_args.q_bias]     ; rax = Q bias base
     mov ecx, [rbx + pq4_fork_args.all_heads]  ; ecx = all_heads_dim, loop count
     PQ_BIAS_ADD                               ; 32-float bias add loop
     vzeroupper                                ; clean AVX state before the call below
.q_bias_done:
     lea r10, pq4f_q_head                      ; r10 = Q head descriptor base
     mov rax, [rbx + pq4_fork_args.q_dst]      ; rax = normed+biased Q row
     mov [r10 + qk_fork_args.buf], rax         ; head buffer base
     mov rax, [rbx + pq4_fork_args.q_weight]   ; rax = q_norm[layer]
     mov [r10 + qk_fork_args.weight], rax      ; per-layer norm weight base
     mov rax, [rbx + pq4_fork_args.cos]        ; rax = position-offset cos row
     mov [r10 + qk_fork_args.cos], rax         ; cos table pointer
     mov rax, [rbx + pq4_fork_args.sin]        ; rax = position-offset sin row
     mov [r10 + qk_fork_args.sin], rax         ; sin table pointer
     mov eax, [rbx + pq4_fork_args.head_dim]   ; eax = head_dim
     mov [r10 + qk_fork_args.head_dim], eax    ; dimension per head
     mov eax, [rbx + pq4_fork_args.n_heads]    ; eax = n_heads
     mov [r10 + qk_fork_args.heads], eax       ; head count
     mov eax, [rbx + pq4_fork_args.pos]        ; eax = RoPE position
     mov [r10 + qk_fork_args.pos], eax         ; sequence position
     lea rcx, pq4f_q_head                      ; rcx = Q descriptor, arg1
     call v7s_qk_fork_heads                    ; Q per-head norm + RoPE without shared BSS
; K bias and norm+RoPE, disjoint from other tile members.
     cmp dword [rbx + pq4_fork_args.bias_active], 0 ; bias block contains real data?
     je .k_bias_done                           ; no, skip the add loop entirely
     mov rdx, r12                              ; rdx = K row, private or shared scratch
     mov rax, [rbx + pq4_fork_args.k_bias]     ; rax = K bias base
     mov ecx, [rbx + pq4_fork_args.kv_dim]     ; ecx = kv_dim, loop count
     PQ_BIAS_ADD                               ; 32-float bias add loop
     vzeroupper                                ; clean AVX state before the call below
.k_bias_done:
     lea r10, pq4f_k_head                      ; r10 = K head descriptor base
     mov [r10 + qk_fork_args.buf], r12         ; K head buffer base
     mov rax, [rbx + pq4_fork_args.k_weight]   ; rax = k_norm[layer]
     mov [r10 + qk_fork_args.weight], rax      ; per-layer norm weight base
     mov rax, [rbx + pq4_fork_args.cos]        ; rax = position-offset cos row
     mov [r10 + qk_fork_args.cos], rax         ; cos table pointer
     mov rax, [rbx + pq4_fork_args.sin]        ; rax = position-offset sin row
     mov [r10 + qk_fork_args.sin], rax         ; sin table pointer
     mov eax, [rbx + pq4_fork_args.head_dim]   ; eax = head_dim
     mov [r10 + qk_fork_args.head_dim], eax    ; dimension per head
     mov eax, [rbx + pq4_fork_args.n_kv_heads] ; eax = n_kv_heads
     mov [r10 + qk_fork_args.heads], eax       ; head count
     mov eax, [rbx + pq4_fork_args.pos]        ; eax = RoPE position
     mov [r10 + qk_fork_args.pos], eax         ; sequence position
     lea rcx, pq4f_k_head                      ; rcx = K descriptor, arg1
     call v7s_qk_fork_heads                    ; K per-head norm + RoPE
; V bias, no RoPE.
     cmp dword [rbx + pq4_fork_args.bias_active], 0 ; bias block contains real data?
     je .v_bias_done                           ; no, skip the add loop entirely
     mov rdx, r13                              ; rdx = V row, private or shared scratch
     mov rax, [rbx + pq4_fork_args.v_bias]     ; rax = V bias base
     mov ecx, [rbx + pq4_fork_args.kv_dim]     ; ecx = kv_dim, loop count
     PQ_BIAS_ADD                               ; 32-float bias add loop
     vzeroupper                                ; clean AVX state before the call below
.v_bias_done:
; Pack independent K/V FP16 cache rows. Each member has its own destinations.
     mov rcx, [rbx + pq4_fork_args.key_cache]  ; rcx = this member's key cache row
     mov rdx, r12                              ; rdx = K FP32 source
     mov r8d, [rbx + pq4_fork_args.kv_dim]     ; r8d = kv_dim elements
     call v7s_kv_pack_f16                      ; pack K row FP32 -> FP16 cache
     mov rcx, [rbx + pq4_fork_args.value_cache] ; rcx = this member's value cache row
     mov rdx, r13                              ; rdx = V FP32 source
     mov r8d, [rbx + pq4_fork_args.kv_dim]     ; r8d = kv_dim elements
     call v7s_kv_pack_f16                      ; pack V row FP32 -> FP16 cache
     CLEANUP_YMM                               ; vzeroupper before the epilogue
     SF_END rbx, rsi, rdi, r12, r13            ; restore non-volatiles and return

%undef pq4f_q_head
%undef pq4f_k_head
%undef PQ4F_FLOAT_BYTES

; Publish one independent QKV postprocessing job for each prepared token.
     SF_START v7d_qkv_post_panel, rbx, rsi, rdi, r12, r13, r14, r15 ; Preserve dispatcher registers
     SF_DWORD qp_caller_count                  ; Jobs handled by the caller
     SF_DWORD qp_caller_end                    ; Caller steal loop bound
     SF_COMMIT                                 ; Allocate the fixed frame
     mov r12d, edx                             ; Preserve the job count
     mov r9, rcx                               ; First token descriptor
     mov r10d, edx                             ; Number of jobs left to stage
     JOB_BASE                                  ; First worker job slot
     mov eax, MATMUL_QKV_POST                  ; Shared mode for this panel
.fill:
     JOB_SET_OUTPUT r9                         ; Stage this token's descriptor
     JOB_SET_MODE eax                          ; Stage the worker mode
     JOB_NEXT rsi                              ; Advance the job array cursor
     add r9, pq4_fork_args_size                ; Advance to the next token descriptor
     dec r10d                                  ; Count the staged job
     jnz .fill                                 ; Fill another job if needed
     DISPATCH_EXECUTE r12d, qp_caller_count, qp_caller_end ; Join before returning the workspace
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; Restore the caller's registers
%undef qp_caller_count
%undef qp_caller_count_raw
%undef qp_caller_end
%undef qp_caller_end_raw
