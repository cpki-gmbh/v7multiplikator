; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ALIGN4 reg, round 32-bit register up to multiple of 4
; Uses add+and: LEA cannot bitwise-AND, masking low bits requires and

%macro ALIGN4 1
     add %1, 3                                 ; round up by 3
     and %1, -4                                ; mask to multiple of 4
%endmacro
; Initial key+value pool size, 128KB ; pool grows geometrically, doubling via KP_ALLOC_INLINE
%define KP_GROW 131072                         ; Initial key+value pool size, 128KB
; Headroom added on top of the exact db_load pool size so that V7-DB-PUT writes made
; during a live session fit WITHOUT relocating the pool. Model-weight pointers are
; derived from pool offsets, so a post-init relocation would dangle every one of them.
%define KP_LIVE_RESERVE 1048576                ; 1MB of free key+value pool reserved at load time

%macro KP_ALLOC_INLINE 0
     mov rdx, rcx                              ; save size; rdx becomes new_used below
     mov rax, [rel g_ht + KP_USED]             ; rax = old used offset = the offset to return
     add rdx, rax                              ; rdx = new used = old+size, 64-bit wrap is physically impossible: the previous allocation would have failed first
     cmp rdx, [rel g_ht + KP_CAP]              ; fits in the current pool capacity?
     jbe %%_kp_fast                            ; common case: fits
     call kp_grow_alloc                        ; rare: grow once in the shared helper
     jmp %%_kpdone                             ; helper already published the used size
%%_kp_fast:                                    ; fast path lands here, jbe target ; rax already holds the offset to return
     mov [rel g_ht + KP_USED], rdx             ; publish new used: the cmp above already proved the pool can hold it
%%_kpdone:
%endmacro
; kp_grow_alloc -- private slow path of the pool allocator, shared by every
; KP_ALLOC_INLINE expansion. IN: RAX = old used, RDX = required used.
; OUT: RAX = allocated offset, KP_USED published. Same nonvolatile contract.
     section .text
kp_grow_alloc:
     cmp qword [g_inf_transformer], 0          ; live context holding pool-derived pointers?
     jne .kp_relocate_fatal                    ; yes: never move the pool out from under them
     sub rsp, 40                               ; Win64 shadow space and alignment for the two allocator calls
     push rbx                                  ; preserve caller hash length
     push rsi                                  ; save non-volatile
     push rdi                                  ; save non-volatile
     push r12                                  ; save non-volatile
     push r13                                  ; save non-volatile
     push r14                                  ; save non-volatile
     mov r12, rax                              ; park old used, rax = the offset to return, overwritten by the calls below
     mov r13, rdx                              ; r13 = new used = old+size
     mov rbx, [rel g_ht + KP_CAP]              ; rbx = current capacity for the doubling below
     add rbx, rbx                              ; rbx = geometric candidate, doubling, O,N total copies vs O,N^2 ; 64-bit wrap needs an 8-exabyte pool, impossible
     cmp rbx, r13                              ; doubled candidate >= required new used?
     cmovb rbx, r13                            ; no: large key, use exact required size as new capacity, branchless
     mov rcx, rbx                              ; arg1: new pool byte size
     call mdb_alloc_mem                        ; allocate new pool, rax = new base, always valid: allocator FATALs on OOM
     mov r14, rax                              ; r14 = new pool base, stable across rep movsb + mdb_free_mem
     mov rdi, rax                              ; rdi = dst for rep movsb, will advance
     mov rsi, [rel g_ht + KP_BASE]             ; rsi = old pool base, rep movsb advances it, reload below before free
     mov rcx, r12                              ; rcx = bytes to copy from old pool, r12 = old used, survives the call
     rep movsb                                 ; copy old pool data to new, advances rsi/rdi, r14 untouched
     mov rcx, [rel g_ht + KP_BASE]             ; rcx = old pool BASE, reload: rsi advanced past it
     call mdb_free_mem                         ; free old pool, overwrites rax, rcx, rdx, r8-r11; r14 still has base
     mov [rel g_ht + KP_BASE], r14             ; store new pool base from saved r14
     mov [rel g_ht + KP_CAP], rbx              ; store new capacity
     mov [rel g_ht + KP_USED], r13             ; publish new used, r13 = old+size from above, NOW that KP_CAP is known to hold it
     mov rax, r12                              ; restore the returned offset, the allocated slice never moved, it lives past old KP_USED
     pop r14                                   ; restore non-volatile
     pop r13                                   ; restore non-volatile
     pop r12                                   ; restore non-volatile
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     add rsp, 40                               ; release helper shadow space and alignment
     ret                                       ; return the pool offset in rax
.kp_relocate_fatal:
     FATAL "microdb: key/value pool cannot grow while a live inference context holds model-weight pointers (the sidecar write exceeds the reserved pool headroom)"
