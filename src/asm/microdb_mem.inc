; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

     section .text
mdb_alloc_mem:
     push rcx                                  ; park size for the OOM fallback path, the push also supplies the 8-byte parity fix, entry 8 mod 16 -> 0 mod 16 at every call below
; VirtualAlloc,NULL, size, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE
     mov rdx, rcx                              ; rdx = size, arg2 source, C-declaration position
     xor ecx, ecx                              ; rcx = lpAddress = NULL, identity-skip arg: API_CALL emits no save/reload round-trip for it, 2-byte xor replaces the mov rax,0 + store + reload triplet
     mov r8d, MEM_COMMIT | MEM_RESERVE         ; flAllocationType
     mov r9d, PAGE_READWRITE                   ; flProtect
     API_CALL [__imp_VirtualAlloc], rcx, rdx, r8, r9 ; all 4 args identity-skip: macro body is sub/call/add only
     test rax, rax                             ; VirtualAlloc succeeded?
     jz .mam_oom                               ; no, file-backed fallback, rare path, out of line
.mam_done:
     pop rcx                                   ; drop the parked size, restores entry parity ; rcx is dead on both paths into .mam_done
     ret                                       ; rax = pointer, both paths converge here
.mam_oom:
; --- OOM fallback, user directive : get file-backed memory instead ---
; Commit charge exhausted -> a temp-file mapping needs ~zero commit.
; returns a fresh zero-filled view, preserving this function's contract.
     mov rcx, [rsp]                            ; restore the parked size, rcx was zeroed for VirtualAlloc above
     call fbm_alloc                            ; rax = file-backed pointer, 0 mod 16 site via the pad above
     jmp .mam_done                             ; converge on the shared pop/ret, rax carries the fbm pointer; rcx stays dead
mdb_free_mem:
     test rcx, rcx                             ; NULL pointer = allocation failed?, tested before the frame: the test needs no frame, and .null_fatal below is reached at entry parity, FATAL realigns itself before calling handle_fatal, so it is legal from either parity
     jz .null_fatal                            ; yes, FATAL, never silently skip, the single choke point: a NULL free is a bug, never a no-op
     push rcx                                  ; park the pointer for the VirtualFree path, fbm_free does NOT preserve rcx, it hands rcx to UnmapViewOfFile ; the push also supplies the 8-byte parity fix, entry 8 mod 16 -> 0 mod 16 at every call below
     call fbm_free                             ; eax = 1: was fbm's, now freed; eax = 0: not fbm's, 0 mod 16 site via the pad
     test eax, eax                             ; did fbm_free handle it?
     jnz .mfm_done                             ; yes, file-backed block unmapped + handles closed, nothing left to do
     mov rcx, [rsp]                            ; no, restore the parked pointer for the normal path
     xor edx, edx                              ; dwSize = 0, required for MEM_RELEASE
     mov r8d, MEM_RELEASE                      ; dwFreeType
     API_CALL [__imp_VirtualFree], rcx, rdx, r8 ; VirtualFree with MEM_RELEASE, 0 mod 16 site via the pad
.mfm_done:
     pop rcx                                   ; drop the parity-pad push, rcx is dead on return
     ret                                       ; back to caller
.null_fatal:
     FATAL "mdb_free_mem: NULL pointer, allocation failed or never happened" ; bug-hiding free refused
