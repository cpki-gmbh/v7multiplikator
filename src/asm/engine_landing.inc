; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; --- Zone constants, all memory offsets are %define'd, never raw numbers ---
%define TEB_THREAD_ID 0x48                     ; x64 TEB ClientId.UniqueThread, LOW dword. NT_TIB is 0x38 bytes, seven pointers 0x00-0x37: ExceptionList/StackBase/StackLimit/SubSystemTib/FiberData/ArbitraryUserPointer/Self, so 0x34 is Self's HIGH dword, not a field; ThreadId/ThreadPriority are x86-only NT_TIB tail fields, EnvironmentPointer@0x38, CLIENT_ID@0x40, UniqueProcess@0x40, UniqueThread@0x48. Read as a dword: a user-mode TID is small so its LOW dword is always nonzero, which is REQUIRED because 0 is the "idle"/"unowned" sentinel in both sl_stk_owner_tid and engine_thread_id. The two adjacent offsets fail in OPPOSITE directions, so neither is a safe substitute: 0x44 is UniqueProcess's HIGH dword, 0 on every real thread, silently disabling every tripwire, and 0x40 is UniqueProcess's LOW dword, the PROCESS id, IDENTICAL on every thread, so a worker-thread FATAL would pass the "do I own the zone?" test and unwind onto the host's stack. Measured: gs:[0x40] = the PID, gs:[0x48] == GetCurrentThreadId,
%define DEATH_REASON_MAX 512                   ; total reason buffer size in bytes
%define REASON_CAP 511                         ; max copied message bytes, NUL takes the last slot
%define WORKER_JOIN_SLEEP_MS 1000              ; worker join wait, mirrors the DllMain DETACH sequence
%define ENG_RETADDR_SLOT 8                     ; bytes above saved rbp where the return address sits
%define ENG_XMM6_OFF 0                         ; xmm6 save slot offset inside engine_saved_xmm
%define ENG_XMM7_OFF 16                        ; xmm7 save slot offset
%define ENG_XMM8_OFF 32                        ; xmm8 save slot offset
%define ENG_XMM9_OFF 48                        ; xmm9 save slot offset
%define ENG_XMM10_OFF 64                       ; xmm10 save slot offset
%define ENG_XMM11_OFF 80                       ; xmm11 save slot offset
%define ENG_XMM12_OFF 96                       ; xmm12 save slot offset
%define ENG_XMM13_OFF 112                      ; xmm13 save slot offset
%define ENG_XMM14_OFF 128                      ; xmm14 save slot offset
%define ENG_XMM15_OFF 144                      ; xmm15 save slot offset
%define ENGINE_AVX_STATE_PROVEN 1              ; zone snapshot proved to cover the full AVX state, xmm6-15 lows on every entry
     section .bss align=64
     alignb 64                                 ; zone start 64-aligned regardless of other .bss users
engine_call_depth:
     resd 1                                    ; owner's nesting count, 0 = no live zone, >0 = depth
engine_gate_claim:
     resd 1                                    ; ATOMIC single-bit zone reservation: bit0 set = a thread owns/has reserved the zone, lock bts acquire, plain store release
engine_thread_id:
     resd 1                                    ; thread id that owns the zone, outermost entry
engine_saved_rsp:
     resq 1                                    ; caller RSP pointing AT the return-address slot
engine_saved_rbp:
     resq 1                                    ; caller RBP
engine_saved_rbx:
     resq 1                                    ; caller rbx, callee-saved, zone must restore it
engine_saved_rsi:
     resq 1                                    ; caller rsi
engine_saved_rdi:
     resq 1                                    ; caller rdi
engine_saved_r12:
     resq 1                                    ; caller r12
engine_saved_r13:
     resq 1                                    ; caller r13
engine_saved_r14:
     resq 1                                    ; caller r14
engine_saved_r15:
     resq 1                                    ; caller r15
     alignb 16                                 ; MANDATORY before the xmm block: movaps requires 16-byte slots. The alignment is an explicit directive on purpose, deriving it from the byte count of the scalar fields above is exactly what breaks the moment a field is added or resized
engine_saved_xmm:
     resb 160                                  ; xmm6-15 low 128 bits each, 16-aligned by the alignb above
v7_engine_dead:
     resd 1                                    ; 0 = operational; 1 = fatal engine state.
engine_death_reason:
     resb DEATH_REASON_MAX                     ; NUL-terminated copy of the printed FATAL text
     section .text

%macro ENGINE_ZONE_SAVE_REGS 0
     mov [rel engine_saved_rbx], rbx           ; snapshot caller rbx
     mov [rel engine_saved_rsi], rsi           ; snapshot caller rsi
     mov [rel engine_saved_rdi], rdi           ; snapshot caller rdi
     mov [rel engine_saved_r12], r12           ; snapshot caller r12
     mov [rel engine_saved_r13], r13           ; snapshot caller r13
     mov [rel engine_saved_r14], r14           ; snapshot caller r14
     mov [rel engine_saved_r15], r15           ; snapshot caller r15
     movaps [rel engine_saved_xmm + ENG_XMM6_OFF], xmm6 ; snapshot xmm6 low 128 bits, 16-aligned slot
     movaps [rel engine_saved_xmm + ENG_XMM7_OFF], xmm7 ; snapshot xmm7 low 128 bits
     movaps [rel engine_saved_xmm + ENG_XMM8_OFF], xmm8 ; snapshot xmm8 low 128 bits
     movaps [rel engine_saved_xmm + ENG_XMM9_OFF], xmm9 ; snapshot xmm9 low 128 bits
     movaps [rel engine_saved_xmm + ENG_XMM10_OFF], xmm10 ; snapshot xmm10 low 128 bits
     movaps [rel engine_saved_xmm + ENG_XMM11_OFF], xmm11 ; snapshot xmm11 low 128 bits
     movaps [rel engine_saved_xmm + ENG_XMM12_OFF], xmm12 ; snapshot xmm12 low 128 bits
     movaps [rel engine_saved_xmm + ENG_XMM13_OFF], xmm13 ; snapshot xmm13 low 128 bits
     movaps [rel engine_saved_xmm + ENG_XMM14_OFF], xmm14 ; snapshot xmm14 low 128 bits
     movaps [rel engine_saved_xmm + ENG_XMM15_OFF], xmm15 ; snapshot xmm15 low 128 bits
%endmacro

%macro ENGINE_GATE 2
     jmp %%entry                               ; entry hop over the cold blocks, they sit physically first
%%dead:
%if %1                                         ; integer-returning entry?
     xor eax, eax                              ; dead-path return value: always 0
%endif
%if %2 = 0                                     ; RAW: no prologue ran, RSP still at the return address
     ret                                       ; raw entry: RSP still at the return address
%elif %2 = 1                                   ; SFR: SF_START pushed no non-volatiles, discard the frame directly
     mov rsp, rbp                              ; discard the SF frame, no pushed regs to pop
     pop rbp                                   ; restore the caller frame pointer
     ret                                       ; return to the C caller
%else                                          ; SFV: SF_START pushed 7 non-volatiles, full epilogue so they are restored
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; full epilogue: pop the 7 pushed regs, unwind frame, ret
%endif
%%concurrent:
     FATAL "engine landing: concurrent public API calls from two host threads" ; FATAL, never returns
%%entry:                                       ; hot path starts here, entry hop lands
     cmp dword [rel v7_engine_dead], 0         ; engine already dead?
     jne %%dead                                ; yes -> dead-path return
     lock bts dword [rel engine_gate_claim], 0 ; ATOMIC zone reservation, CF=1: another entry owns/held it
     jnc %%outermost                           ; claim won: publish the zone, death rechecked inside
     cmp dword [rel v7_engine_dead], 0         ; claim lost AND the owner died mid-unwind?, TOCTOU: dead==0 was read above, the unwinder set it and released the claim before the bts landed
     jne %%dead                                ; yes -> dead return; no depth change, this entry never incremented
     V7_THREAD_ID eax, rax                     ; eax = thread id, TEB ClientId.UniqueThread on win64
     cmp eax, [rel engine_thread_id]           ; zone owner is this thread?
     jne %%concurrent                          ; no -> single-caller contract violation
     inc dword [rel engine_call_depth]         ; nested entry: keep the OUTER zone
     jmp %%done                                ; body continues under the outer zone, also the fall-through barrier for %%claim_dead below
%%claim_dead:                                  ; won the claim of a dead engine, race window only
     mov dword [rel engine_gate_claim], 0      ; drop the stale claim BEFORE returning, leaving it set would FATAL every later caller as "concurrent"
     jmp %%dead                                ; dead-path return; the zone was never published
%%outermost:                                   ; claim won, atomic claim held from the bts above
     cmp dword [rel v7_engine_dead], 0         ; a FATAL unwinder ran between the entry check and the bts?, a claim held by the DEAD owner was released, entering the body would run a quiesced engine
     jne %%claim_dead                          ; yes -> release the stale claim, dead return
     V7_THREAD_ID eax, rax                     ; eax = thread id, TEB ClientId.UniqueThread on win64
     mov [rel engine_thread_id], eax           ; zone owner = this thread
%if %2 = 0                                     ; RAW: rsp/rbp are already the caller's
     mov [rel engine_saved_rsp], rsp           ; RSP points AT the return address here
     mov [rel engine_saved_rbp], rbp           ; caller frame pointer, untouched so far
%else                                          ; SFR/SFV: derive them from the SF frame pointer
     lea rax, [rbp + ENG_RETADDR_SLOT]         ; rax = caller RSP at entry, return-address slot
     mov [rel engine_saved_rsp], rax           ; unwind target: RSP pointing AT the return address
     mov rax, [rbp]                            ; rax = caller's RBP, saved by the SF prologue
     mov [rel engine_saved_rbp], rax           ; caller frame pointer for the unwind
%endif
     ENGINE_ZONE_SAVE_REGS                     ; snapshot the callee-saved register file
     mov dword [rel engine_call_depth], 1      ; outermost: depth 1 = live zone
; fall through into %%done, no unconditional jump on the hot path
%%done:                                        ; body continues here, fall-through
%endmacro
; thin, self-documenting wrappers so call sites keep reading by intent

%macro ENGINE_GATE_RAW 1
     ENGINE_GATE %1, 0                         ; hand-labelled raw entry: no prologue ran
%endmacro

%macro ENGINE_GATE_SFR 1
     ENGINE_GATE %1, 1                         ; SF-framed wrapper, no pushed non-volatiles
%endmacro

%macro ENGINE_GATE_SFV 1
     ENGINE_GATE %1, 2                         ; SF-framed wrapper that already pushed its non-volatiles
%endmacro

%macro ENGINE_LEAVE 0
     dec dword [rel engine_call_depth]         ; nesting protocol: 0 = zone parked
     jnz %%leave_nested                        ; still inside an outer call: the claim stays held
     mov dword [rel engine_thread_id], 0       ; drop the owner id BEFORE releasing the claim: a stale id would let a second host thread match the lost-claim thread check below, ENGINE_GATE line 139 and enter the body concurrently with the new owner
     mov dword [rel engine_gate_claim], 0      ; outermost exit: release the atomic claim, owner is the only 0-writer; TSO-ordered after the dec
%%leave_nested:
%endmacro
     global handle_fatal                       ; internal symbol, FATAL macro calls it
handle_fatal:
     SF_START handle_fatal, rbx, rsi, rdi      ; frame + non-volatile scratch for the copy, r12-r15 unused here
     SF_COMMIT                                 ; finalize stack frame layout
; Prevent a second unwind when the engine is already in a fatal state.
     cmp dword [rel v7_engine_dead], 0         ; engine already dead from a caught FATAL?
     jne .hf_old_path                          ; yes -> old terminal path, never re-unwind
; --- step 2: zone validity + thread ownership, worker policy ---
     cmp dword [rel engine_call_depth], 0      ; any live landing zone?
     je .hf_old_path                           ; no, worker/DllMain/pre-gate -> old path
     V7_THREAD_ID eax, rax                     ; eax = thread id, TEB ClientId on win64
     cmp eax, [rel engine_thread_id]           ; does this thread own the zone?
     jne .hf_old_path                          ; no, worker thread -> old path, by directive
; --- step 3: record the death BEFORE anything else can go wrong ---
     mov rbx, rcx                              ; rbx = message pointer, STR_LEN overwrites rcx
     STR_LEN rbx, r8d                          ; ecx = message byte length, edx keeps the string data, unused here
     cmp ecx, REASON_CAP                       ; message longer than the reason buffer?, ecx = same length, saves a mov
     jbe .hf_len_ok                            ; no -> copy it whole
     mov ecx, REASON_CAP                       ; yes -> cap at 511 bytes, rep count register directly
.hf_len_ok:
     lea rdi, [rel engine_death_reason]        ; rdi = reason buffer destination
     mov rsi, rbx                              ; rsi = message source, lea/mov do not touch ecx
     rep movsb                                 ; copy the FATAL message into the reason buffer, length already capped
     mov byte [rdi], 0                         ; NUL-terminate the reason copy
     mov dword [rel v7_engine_dead], 1         ; park the engine: every gate now early-returns
     xor edx, edx                              ; zero source for the three ownership fields
     mov dword [rel engine_call_depth], edx    ; zone parked: depth 0 = no live zone
     mov dword [rel engine_thread_id], edx     ; drop the owner id BEFORE the release: a stale id would let another host thread pass ENGINE_GATE's lost-claim thread check and enter a quiesced engine
     mov dword [rel engine_gate_claim], edx    ; release the atomic claim, a post-free gated call may re-claim
; --- step 5: keep the fatal print, no debugging output lost ---
     mov rcx, rbx                              ; rcx = message pointer again
     call fatal_print_msg                      ; print to stdout, identical to v7s_fatal_error
; --- step 6: quiesce the worker pool, safe FreeLibrary prerequisite,
; the proven DllMain DETACH sequence, R3 ---
     cmp dword [rel init_ok], 0                ; worker pool ever fully initialized?
     je .hf_pool_partial                       ; no, but workers created before the FATAL may already spin
     call v7s_shutdown                         ; shutdown flag + wake + join workers
     mov ecx, WORKER_JOIN_SLEEP_MS             ; 1000 ms join wait, mirrors DllMain exactly
     API_CALL [__imp_Sleep], rcx               ; give workers time to exit .worker_outer
     jmp .hf_pool_done                         ; full pool handled by the shutdown sequence
.hf_pool_partial:
     cmp dword [rel workers_alive], 0          ; any worker started before the init FATAL?
     je .hf_pool_done                          ; no, nothing spinning to stop
     mov dword [rel shutdown_flag], 1          ; started workers poll this every spin iteration and exit on their own, init workers always run always_spin=1
.hf_pool_done:
     mov qword [rel g_user_prompt], 0          ; NULL outside a live turn, same contract as .chat_finish
     mov dword [rel g_img_turn], 0             ; no live image turn after the unwind
     xor eax, eax                              ; dead-path return value: always 0
     mov rbx, [rel engine_saved_rbx]           ; restore caller rbx
     mov rsi, [rel engine_saved_rsi]           ; restore caller rsi
     mov rdi, [rel engine_saved_rdi]           ; restore caller rdi
     mov r12, [rel engine_saved_r12]           ; restore caller r12
     mov r13, [rel engine_saved_r13]           ; restore caller r13
     mov r14, [rel engine_saved_r14]           ; restore caller r14
     mov r15, [rel engine_saved_r15]           ; restore caller r15
     movaps xmm6, [rel engine_saved_xmm + ENG_XMM6_OFF] ; restore xmm6 low half, 16-aligned slot
     movaps xmm7, [rel engine_saved_xmm + ENG_XMM7_OFF] ; restore xmm7 low half
     movaps xmm8, [rel engine_saved_xmm + ENG_XMM8_OFF] ; restore xmm8 low half
     movaps xmm9, [rel engine_saved_xmm + ENG_XMM9_OFF] ; restore xmm9 low half
     movaps xmm10, [rel engine_saved_xmm + ENG_XMM10_OFF] ; restore xmm10 low half
     movaps xmm11, [rel engine_saved_xmm + ENG_XMM11_OFF] ; restore xmm11 low half
     movaps xmm12, [rel engine_saved_xmm + ENG_XMM12_OFF] ; restore xmm12 low half
     movaps xmm13, [rel engine_saved_xmm + ENG_XMM13_OFF] ; restore xmm13 low half
     movaps xmm14, [rel engine_saved_xmm + ENG_XMM14_OFF] ; restore xmm14 low half
     movaps xmm15, [rel engine_saved_xmm + ENG_XMM15_OFF] ; restore xmm15 low half
     mov rsp, [rel engine_saved_rsp]           ; RSP now points at the return address
     mov rbp, [rel engine_saved_rbp]           ; caller frame pointer restored
     cmp dword [rel cpu_features_ok], ENGINE_AVX_STATE_PROVEN ; did the v7s_cpu_gate probe prove AVX + OS-enabled YMM state?
     jne .hf_ret                               ; no, vzeroupper would #UD, and the upper halves were never dirtied
     vzeroupper                                ; clean YMM upper halves before returning to C
.hf_ret:
     ret                                       ; pops the ORIGINAL return address -> C caller resumes
.hf_old_path:
     call v7s_fatal_error                      ; print + TerminateProcess,1, never returns
     global v7_engine_is_dead                  ; export public query
v7_engine_is_dead:
     mov eax, [rel v7_engine_dead]             ; return the death flag, 0 or 1
     ret                                       ; plain leaf return, no frame needed
     global v7_get_death_reason                ; export public query
v7_get_death_reason:
     lea rax, [rel engine_death_reason]        ; return pointer to the reason buffer
     ret                                       ; plain leaf return, no frame needed
