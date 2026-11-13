; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define MROPE_F4 4                             ; sizeof,float for table addressing
%define MROPE_PAIR_PLUS1 1                     ; pair index delta j+1, h slot inside a, t,h,w group
%define MROPE_PAIR_PLUS2 2                     ; pair index delta j+2, w slot inside a, t,h,w group
%define MROPE_GROUP_NEXT 3                     ; pair index delta j+3, start of the next, t,h,w group
     section .bss
g_mrope_bound:
     resd 1                                    ; interleaved pair bound, 0 = no M-RoPE
g_mrope_bound_loaded:
     resd 1                                    ; 1 = g_mrope_bound fetched from microdb
g_mrope_delta:
     resd 1                                    ; seq-index correction for generation rope positions
g_mrope_run:
     resd 1                                    ; running M-RoPE component counter, persists across turns
g_mrope_scratch_cos:
     resq 1                                    ; scratch cos plane, OWNING allocation for both planes, sin is the interior half
g_mrope_scratch_bytes:
     resq 1                                    ; byte capacity of one scratch plane
g_mrope_scratch_sin:
     resq 1                                    ; scratch sin table, interior pointer, never freed directly
g_mrope_saved_cos:
     resq 1                                    ; base cos pointer saved during a scratch swap, 0 = not swapped
g_mrope_saved_sin:
     resq 1                                    ; base sin pointer saved during a scratch swap
     section .rdata align=8
sz_sys_mrope_bound:
     db "__sys_mrope_bound", 0                 ; microdb key holding the u64 bound
     section .text
mrope_merged_row:
     SF_START mmr_frame, rbx, r13, r14         ; save non-volatiles: rbx = group limit, r13 = h base, r14 = w base, per-component row offsets
     mov r10d, dword [rel g_mrope_bound]       ; bound, loop-invariant, hoisted; still read by .mmr_partial below
     xor ecx, ecx                              ; j = 0, rotary pair index
     cmp r10d, MROPE_GROUP_NEXT                ; bound < 3? COMPARE, not a sub: r10d, bound is live in .mmr_partial's phase tests below, a sub would destroy it
     jb .mmr_partial                           ; bound < 3, no full group fits, at most 2 scalar remainder pairs
     lea ebx, [r10 - MROPE_GROUP_NEXT]         ; ebx = bound-3 = last j that starts a full group, tested by cmp ecx,ebx
.mmr_group:
; --- pair j: t component ---
     lea rax, [r12 + rcx]                      ; rax = t_base + j, element offset
     mov edx, dword [rsi + rax*MROPE_F4]       ; load exact cos float bits
     mov eax, dword [rdi + rax*MROPE_F4]       ; load sin bits BEFORE any destination store, address read is pre-write
     mov dword [r8 + rcx*MROPE_F4], edx        ; identical 32-bit cos value
     mov dword [r9 + rcx*MROPE_F4], eax        ; identical 32-bit sin value
; --- pair j+1: h component, pair delta folded into the disp8 addend, no edx temp ---
     lea rax, [r13 + rcx + MROPE_PAIR_PLUS1]   ; rax = h_base +, j+1 : the pair delta folds into the disp8 addend
     mov edx, dword [rsi + rax*MROPE_F4]       ; load exact cos float bits
     mov eax, dword [rdi + rax*MROPE_F4]       ; load sin bits BEFORE any destination store, address read is pre-write
     mov dword [r8 + rcx*MROPE_F4 + MROPE_PAIR_PLUS1*MROPE_F4], edx ; identical 32-bit cos value
     mov dword [r9 + rcx*MROPE_F4 + MROPE_PAIR_PLUS1*MROPE_F4], eax ; identical 32-bit sin value
; --- pair j+2: w component, pair delta folded into the disp8 addend, no edx temp ---
     lea rax, [r14 + rcx + MROPE_PAIR_PLUS2]   ; rax = w_base +, j+2 : the pair delta folds into the disp8 addend
     mov edx, dword [rsi + rax*MROPE_F4]       ; load exact cos float bits
     mov eax, dword [rdi + rax*MROPE_F4]       ; load sin bits BEFORE any destination store, address read is pre-write
     mov dword [r8 + rcx*MROPE_F4 + MROPE_PAIR_PLUS2*MROPE_F4], edx ; identical 32-bit cos value
     mov dword [r9 + rcx*MROPE_F4 + MROPE_PAIR_PLUS2*MROPE_F4], eax ; identical 32-bit sin value
     add ecx, MROPE_GROUP_NEXT                 ; j += 3, full group done
     cmp ecx, ebx                              ; another full group fits in [0, bound ?
     jbe .mmr_group                            ; yes, next, t,h,w group
.mmr_partial:
; remainder pairs inside [0, bound : phases continue t, h from group end
     cmp ecx, r10d                             ; j < bound, phase 0 = t ?
     jge .mmr_tail                             ; no, interleaved region complete
     lea rax, [r12 + rcx]                      ; rax = t_base + j, element offset
     mov edx, dword [rsi + rax*MROPE_F4]       ; load exact cos float bits
     mov eax, dword [rdi + rax*MROPE_F4]       ; load sin bits BEFORE any destination store, address read is pre-write
     mov dword [r8 + rcx*MROPE_F4], edx        ; identical 32-bit cos value
     mov dword [r9 + rcx*MROPE_F4], eax        ; identical 32-bit sin value
     inc ecx                                   ; j++
     cmp ecx, r10d                             ; j < bound, phase 1 = h ?
     jge .mmr_tail                             ; no, interleaved region complete
     lea rax, [r13 + rcx]                      ; rax = h_base + j, element offset
     mov edx, dword [rsi + rax*MROPE_F4]       ; load exact cos float bits
     mov eax, dword [rdi + rax*MROPE_F4]       ; load sin bits BEFORE any destination store, address read is pre-write
     mov dword [r8 + rcx*MROPE_F4], edx        ; identical 32-bit cos value
     mov dword [r9 + rcx*MROPE_F4], eax        ; identical 32-bit sin value
     inc ecx                                   ; j++, j == bound now at most
.mmr_tail:
     cmp ecx, r11d                             ; j >= half, row complete ?
     jge .mmr_done                             ; yes, done, before the tail setup: no walk emitted
     mov r10d, ecx                             ; r10 = j, walk index, 64-bit; j >= 0 so the 32-bit move zero-extends, identical to movsxd
     sub r10, r11                              ; r10 = j - half: walks -count .. -1, MUST stay 64-bit, see the note above
     lea rax, [r12 + r11]                      ; t_base + half, t-row end element index
     lea rdx, [rsi + rax*MROPE_F4]             ; cos src end pointer
     lea r13, [rdi + rax*MROPE_F4]             ; sin src end pointer, r13 is scratch; the frame restores the caller's h_base
     lea rbx, [r8 + r11*MROPE_F4]              ; cos dst end pointer
     lea r14, [r9 + r11*MROPE_F4]              ; sin dst end pointer
.mmr_tail_loop:
     mov eax, [rdx + r10*MROPE_F4]             ; cos tail dword
     mov ecx, [r13 + r10*MROPE_F4]             ; sin tail dword
     mov [rbx + r10*MROPE_F4], eax             ; store cos
     mov [r14 + r10*MROPE_F4], ecx             ; store sin
     inc r10                                   ; next, less negative, 64-bit inc keeps the sign bits
     jne .mmr_tail_loop                        ; until index reaches exactly 0
.mmr_done:
     SF_END rbx, r13, r14                      ; row complete; restore non-volatiles, frame pointer, return
     global v7s_mrope_prepare                  ; export for C diagnostics/tools
v7s_mrope_prepare:
     SF_START v7s_mrope_prepare, rbx, rsi, rdi, r12, r13, r14, r15 ; prologue: save non-volatiles
     SF_QWORD mp_tokens                        ; const int* tokens
     SF_DWORD mp_n                             ; token count this call
     SF_DWORD mp_start                         ; absolute seq index of tokens[0]
     SF_DWORD mp_i                             ; scan index within [0, n
     SF_DWORD mp_run                           ; running component counter, local copy
     SF_DWORD mp_half                          ; rope table row width, head_dim/2
     SF_DWORD mp_row_bytes                     ; half * 4
     SF_DWORD mp_rows_total                    ; start_pos + n
     SF_DWORD mp_len                           ; image-run length L
     SF_DWORD mp_grid                          ; merged grid side G = isqrt,L
     SF_DWORD mp_hh                            ; row index within an image run, k/G
     SF_QWORD mp_base_cos                      ; base cos table, saved before swap
     SF_QWORD mp_base_sin                      ; base sin table, saved before swap
     SF_COMMIT                                 ; finalize stack frame layout
     mov mp_tokens, rcx                        ; tokens array
     mov mp_n, edx                             ; n
     mov mp_start, r8d                         ; start_pos
; --- Exported-boundary validation, BEFORE any state access or allocation:
; an invalid call must FATAL without mutating g_mrope_* or the scratch ---
     test edx, edx                             ; n_tokens <= 0?
     jle .mp_arg_fatal                         ; yes, invalid call
     test rcx, rcx                             ; tokens == NULL with n > 0?
     jz .mp_arg_fatal                          ; yes, invalid call
     test r8d, r8d                             ; start_pos < 0?
     js .mp_arg_fatal                          ; yes, invalid call
     mov ebx, r8d                              ; validated nonnegative start position
     add ebx, edx                              ; start + count, at most 2*INT_MAX
     cmp ebx, ROPE_INTERNAL_MAX_SEQ_LEN        ; row range within the base table?
     ja .mp_cap_fatal                          ; no, beyond the flat RoPE table capacity
; --- Lazy-load the M-RoPE bound from microdb, first call only ---
     cmp dword [rel g_mrope_bound_loaded], 0   ; bound fetched yet?
     jne .mp_bound_ready                       ; yes, skip lookup
     lea rcx, [rel sz_sys_mrope_bound]         ; key = "__sys_mrope_bound"
     xor edx, edx                              ; out_len = NULL, value is a fixed u64
     call ht_find                              ; rax = data ptr or NULL on miss
     test rax, rax                             ; key present?
     jz .mp_no_mrope                           ; no, model predates the key: plain 1D rope
     mov ecx, dword [rax]                      ; low dword of the u64 value, bound < 2^31
     mov dword [rel g_mrope_bound], ecx        ; store bound
     mov dword [rel g_mrope_bound_loaded], 1   ; mark loaded
     test ecx, ecx                             ; bound == 0, explicitly disabled ?
     jz .mp_ret                                ; yes, plain 1D rope for this model
     jmp .mp_bound_ready                       ; LOAD-BEARING: fall-through would enter .mp_no_mrope -> .mp_ret, skips prepare on first call!
.mp_no_mrope:
     mov dword [rel g_mrope_bound_loaded], 1   ; mark loaded, missing key = bound 0
.mp_ret:
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; early exit: nothing to do
.mp_bound_ready:
     cmp dword [rel g_mrope_bound], 0          ; M-RoPE disabled for this model?
     je .mp_ret                                ; yes, plain 1D rope, no side effects
; --- Rope geometry from the global table state ---
     mov eax, dword [rel rope_max_dim]         ; row width = head_dim/2
     test eax, eax                             ; rope tables initialized?
     jz .mp_rope_fatal                         ; no, contract violation
     mov mp_half, eax                          ; save half
     shl eax, 2                                ; half * 4 bytes per row. MROPE_F4 is a power of two, so a shift, asm_trick.md --
     mov mp_row_bytes, eax                     ; save row_bytes
     mov r15d, eax                             ; park row bytes in non-volatile r15, survives the scratch allocs below
     mov rax, qword [rel fused_rope_cos]       ; base cos pointer
     test rax, rax                             ; table built?
     jz .mp_rope_fatal                         ; no, contract violation
     mov mp_base_cos, rax                      ; save base cos
     mov rax, qword [rel fused_rope_sin]       ; base sin pointer
     mov mp_base_sin, rax                      ; save base sin
; --- Fresh session: reset the running counter ---
     cmp mp_start, 0                           ; first turn of a session?
     jne .mp_run_kept                          ; no, keep persisted run
     mov qword [rel g_mrope_delta], 0          ; delta + run cleared together, plain rope, fresh positions
.mp_run_kept:
     mov eax, mp_start                         ; start_pos
     add eax, dword [rel g_mrope_delta]        ; + current generation delta
     mov mp_run, eax                           ; into local copy
; --- Pre-scan: any image tokens in this range? ---
     mov edx, dword [rel g_image_token_id]     ; image token ID
     test edx, edx                             ; image_token_id == 0, text-only model ?
     jz .mp_scan_done                          ; yes, no image tokens possible, skip scan
     mov rax, mp_tokens                        ; tokens base, hoisted out of loop
     xor ecx, ecx                              ; i = 0
.mp_scan_head:
     cmp dword [rax + rcx*MROPE_F4], edx       ; tokens[i] == image-pad? (n > 0 validated, first probe safe)
     je .mp_need_scratch                       ; image pad found, full scratch build required
     inc ecx                                   ; i++
     cmp ecx, mp_n                             ; reached the end?
     jl .mp_scan_head                          ; scan the next token
.mp_scan_done:
     mov eax, mp_run                           ; corrected local start, re-based at .mp_run_kept
     add eax, mp_n                             ; run + n, register arithmetic
     mov dword [rel g_mrope_run], eax          ; store updated run
     sub eax, mp_start                         ; delta = run - start_pos ...
     sub eax, mp_n                             ; ... - n, = run -, start+n
     mov dword [rel g_mrope_delta], eax        ; store delta for generation
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; done, no table swap
.mp_need_scratch:
     mov mp_rows_total, ebx                    ; rows_total, bounded by the ROPE_INTERNAL_MAX_SEQ_LEN check at entry; read by the delta math below
     mov r14d, ebx                             ; absolute row count through the end of the prompt
     imul r14, r15                             ; requested bytes per plane
     cmp r14, [rel g_mrope_scratch_bytes]      ; do both existing planes fit?
     jbe .mp_buf_ok                            ; reuse the high-water allocation
     mov rcx, [rel g_mrope_scratch_cos]        ; old owning base
     test rcx, rcx                             ; is there an allocation to replace?
     jz .mp_buf_allocate                       ; first image prompt
     mov qword [rel g_mrope_scratch_cos], 0    ; detach the old owner before freeing it
     mov qword [rel g_mrope_scratch_bytes], 0  ; invalidate the old capacity
     call mdb_free_mem                         ; merged rows are regenerated, so no copy is needed
.mp_buf_allocate:
     lea rcx, [r14 + r14]                      ; two planes of equal capacity
     call mdb_alloc_mem                        ; allocate the needed absolute row range
     lea rdx, [rax + r14]                      ; sin starts immediately after cos
     mov [rel g_mrope_scratch_sin], rdx        ; publish the interior pointer
     mov [rel g_mrope_scratch_cos], rax        ; publish the owning base
     mov [rel g_mrope_scratch_bytes], r14      ; publish the new usable capacity
.mp_buf_ok:
; --- Main scan: build merged rows for [start_pos, start_pos+n ---
     mov dword mp_i, 0                         ; i = 0
.mp_loop:
     mov eax, mp_i                             ; i
     cmp eax, mp_n                             ; i < n?
     jge .mp_loop_done                         ; no, scan complete
     mov rax, mp_tokens                        ; tokens base
     mov ecx, mp_i                             ; i
     mov edx, dword [rax + rcx*MROPE_F4]       ; edx = tokens[i]
     cmp edx, dword [rel g_image_token_id]     ; image-pad token?
     je .mp_image_run                          ; yes, handle the whole run
; --- Text token: scratch row = base row [run], plain copy, both tables
; Compute dst/src byte offsets once; r8/r9 survive rep movsb
     mov eax, mp_start                         ; absolute row index = start + i
     add eax, mp_i                             ; + i, row index fits 32 bits
     mov r8d, eax                              ; zero-extend row index to 64-bit
     imul r8, r15                              ; dst byte offset, 64-bit product
     mov eax, mp_run                           ; src row = run
     mov r9d, eax                              ; zero-extend src row to 64-bit
     imul r9, r15                              ; src byte offset, 64-bit product
; Cos row copy
     mov rsi, mp_base_cos                      ; src base cos
     add rsi, r9                               ; rsi = src row
     mov rdi, qword [rel g_mrope_scratch_cos]  ; dst scratch cos
     add rdi, r8                               ; rdi = dst row
     mov ecx, mp_row_bytes                     ; copy one full row
     rep movsb                                 ; cos row copy
; Sin row copy, reuse saved offsets
     mov rsi, mp_base_sin                      ; src base sin
     add rsi, r9                               ; rsi = src row
     mov rdi, qword [rel g_mrope_scratch_sin]  ; dst scratch sin
     add rdi, r8                               ; rdi = dst row
     mov ecx, mp_row_bytes                     ; one row
     rep movsb                                 ; sin row copy
     inc dword mp_run                          ; run += 1, text advances counter
     inc dword mp_i                            ; i += 1
     jmp .mp_loop                              ; next token
.mp_image_run:
; The .mp_loop dispatch already established tokens[mp_i] is image-pad
     mov eax, mp_i                             ; current image run starts here
     mov rdx, mp_tokens                        ; stable base for the run scan
.mp_measure:
     inc eax                                   ; consume one known image-pad token
     cmp eax, mp_n                             ; reached n?
     jae .mp_measure_done                      ; yes: run ends at the prompt end
     mov ecx, dword [rdx + rax*MROPE_F4]       ; inspect the following token
     cmp ecx, dword [rel g_image_token_id]     ; still image-pad?
     je .mp_measure                            ; another image-pad, consume it
.mp_measure_done:
     sub eax, mp_i                             ; L = j - i
     mov mp_len, eax                           ; save L
; isqrt,L : grid side with grid*grid == L, FATAL otherwise
     vcvtsi2sd xmm0, xmm0, eax                 ; exact binary64 representation
     vsqrtsd xmm0, xmm0, xmm0                  ; sqrt(L)
     vroundsd xmm0, xmm0, xmm0, 2              ; ceil(sqrt(L))
     vcvttsd2si ecx, xmm0                      ; integer grid side G
     mov mp_grid, ecx                          ; G; the builder emits grid*grid embeddings, so the candidate IS exact
; Emit the L merged rows: k-th row gathers t=run, h=run+k/G, w=run+k%G
; Hoist loop-invariant setup, all survive mrope_merged_row call
     mov r11d, mp_half                         ; r11d = half, lives across the loops, mrope_merged_row's documented contract preserves it
     mov r12d, mp_run                          ; r12d = run, zero-extends
     imul r12, r11                             ; r12 = t_base = run * half, element offset, constant over run
     mov rsi, mp_base_cos                      ; rsi = base cos, survives call
     mov rdi, mp_base_sin                      ; rdi = base sin, survives call
; r15 = row_bytes, hoisted at top, dst advance stride each iter
     mov eax, mp_start                         ; initial dst row =, start + i * row_bytes
     add eax, mp_i                             ; + i, row index fits 32 bits, zero-extends to rax
     imul rax, r15                             ; 64-bit byte offset
     mov r8, qword [rel g_mrope_scratch_cos]   ; cos scratch base
     add r8, rax                               ; r8 = first cos dst row
     mov r9, qword [rel g_mrope_scratch_sin]   ; sin scratch base
     add r9, rax                               ; r9 = first sin dst row
     mov dword mp_hh, 0                        ; hh = 0, row index k/G
     mov r13d, mp_run                          ; initial h=run, zero-extends
     imul r13, r11                             ; h_base = run*half, once
.mp_row_outer:
     xor ebx, ebx                              ; column = 0, held live in a callee-saved register
     mov r14d, mp_run                          ; initial w=run, zero-extends
     imul r14, r11                             ; w_base = run*half, once per row band
.mp_row_inner:
     call mrope_merged_row                     ; preserves rbx via its own SF_START/SF_END
     add r8, r15                               ; advance cos dst to next row
     add r9, r15                               ; advance sin dst to next row
     add r14, r11                              ; w_base += half for next column
     inc ebx                                   ; column++
     cmp ebx, mp_grid                          ; column < G?
     jl .mp_row_inner                          ; yes, next column
     inc dword mp_hh                           ; hh++
     add r13, r11                              ; h_base += half for next row band
     mov eax, mp_hh                            ; load row index for compare
     cmp eax, mp_grid                          ; hh < G?
     jl .mp_row_outer                          ; yes, next row band
     mov eax, mp_grid                          ; after an image run the counter
     add dword mp_run, eax                     ; advances by G, HF semantics
     mov eax, mp_len                           ; L
     add mp_i, eax                             ; i += L, whole run consumed
     jmp .mp_loop                              ; continue scan
.mp_loop_done:
; --- Finalize: delta, table swap ---
     mov eax, mp_run                           ; run, max component + 1
     mov dword [rel g_mrope_run], eax          ; persist across turns
     sub eax, mp_rows_total                    ; delta = run -, start + n
     mov dword [rel g_mrope_delta], eax        ; store for generation
     mov rax, qword [rel fused_rope_cos]       ; current cos, base
     mov qword [rel g_mrope_saved_cos], rax    ; save for restore
     mov rax, qword [rel fused_rope_sin]       ; current sin, base
     mov qword [rel g_mrope_saved_sin], rax    ; save for restore
     mov rax, qword [rel g_mrope_scratch_cos]  ; scratch cos
     mov qword [rel fused_rope_cos], rax       ; swap in
     mov rax, qword [rel g_mrope_scratch_sin]  ; scratch sin
     mov qword [rel fused_rope_sin], rax       ; swap in
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; done, tables swapped
.mp_rope_fatal:
     FATAL "mrope: rope tables not initialized before prepare" ; contract: init order
.mp_arg_fatal:
     FATAL "mrope: invalid prepare arguments (n_tokens <= 0, tokens NULL, or negative start_pos)" ; exported-boundary guard: FATAL before any state mutation
.mp_cap_fatal:
     FATAL "mrope: start_pos + n_tokens exceeds the RoPE table capacity" ; exported-boundary guard: FATAL before the scratch allocation
     global v7s_mrope_restore                  ; export for C diagnostics/tools
v7s_mrope_restore:
     mov rax, qword [rel g_mrope_saved_cos]    ; saved base cos, 0 = not swapped
     test rax, rax                             ; swap active?
     jz .mr_done                               ; no, nothing to restore
     mov qword [rel fused_rope_cos], rax       ; restore base cos
     mov rax, qword [rel g_mrope_saved_sin]    ; saved base sin
     mov qword [rel fused_rope_sin], rax       ; restore base sin
     xor eax, eax                              ; 0
     mov qword [rel g_mrope_saved_cos], rax    ; clear swap marker
     mov qword [rel g_mrope_saved_sin], rax    ; clear swap marker
.mr_done:
     ret                                       ; leaf return, no frame needed
; ---- %undef cleanup: NASM re-defines textually, last one wins, with no diagnostic ----
%undef MROPE_GROUP_NEXT
%undef MROPE_PAIR_PLUS2
%undef MROPE_PAIR_PLUS1
%undef MROPE_F4                                ; bare "sizeof,float " name, a later file could plausibly reuse it
; ---- %undef cleanup: SF_* slot names, both forms: NAME and NAME_raw ----
%undef mp_tokens
%undef mp_tokens_raw
%undef mp_n
%undef mp_n_raw
%undef mp_start
%undef mp_start_raw
%undef mp_i
%undef mp_i_raw
%undef mp_run
%undef mp_run_raw
%undef mp_half
%undef mp_half_raw
%undef mp_row_bytes
%undef mp_row_bytes_raw
%undef mp_rows_total
%undef mp_rows_total_raw
%undef mp_len
%undef mp_len_raw
%undef mp_grid
%undef mp_grid_raw
%undef mp_hh
%undef mp_hh_raw
%undef mp_base_cos
%undef mp_base_cos_raw
%undef mp_base_sin
%undef mp_base_sin_raw
