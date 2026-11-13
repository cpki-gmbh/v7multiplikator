; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define EVICT_KEY_TAIL_OFF 8                   ; byte offset of the second "on_stream_evict" name half in the key bytes

ht_init:
     sub rsp, 40                               ; shadow space + alignment, entry RSP=8mod16; sub 40 -> 0mod16; body touches no non-volatiles, no pushes needed
     lea rcx, [rel g_ht]                       ; table state base
     mov dword [rcx + HT_CAP], MDB_INIT_CAP    ; store initial capacity
     mov dword [rcx + HT_MASK], MDB_INIT_CAP - 1 ; store initial mask
     xor eax, eax                              ; zero source for the following init stores
     mov dword [rcx + HT_COUNT], eax           ; start with 0 entries, pure write avoids RMW stall
     mov qword [rcx + HT_TABLE], rax           ; clear table pointer, fail-safe, pure write avoids RMW stall
     mov qword [rcx + KP_BASE], rax            ; clear pool pointer, pure write avoids RMW stall
     mov qword [rcx + KP_USED], 4              ; offset-4 sentinel, never handed out; keeps offset 0 unusable so a NULL-ish offset can't alias entry data
     mov qword [rcx + KP_CAP], rax             ; pool capacity 0 until ht_init allocates below
; Allocate entry table
     mov ecx, MDB_INIT_CAP * ENTRY_SZ          ; size = 512 * 40 = 20480
     call mdb_alloc_mem                        ; allocate zeroed Entry array
     lea rcx, [rel g_ht]                       ; reload base
     mov [rcx + HT_TABLE], rax                 ; store table pointer
; Allocate initial key pool, 128KB
     mov ecx, KP_GROW                          ; 128KB initial pool
     call mdb_alloc_mem                        ; allocate pool, zeroed by the kernel
     lea rcx, [rel g_ht]                       ; reload base
     mov [rcx + KP_BASE], rax                  ; store pool base
     mov qword [rcx + KP_CAP], KP_GROW         ; pool capacity = 128KB
; KP_USED set to 4, sentinel above
     mov eax, 1                                ; return success
     add rsp, 40                               ; reclaim shadow space + alignment
     ret                                       ; back to caller
ht_destroy:
     sub rsp, 40                               ; 40 = 0 pushes + 32 shadow + 8 pad -> 0mod16, body touches no non-volatiles, no pushes needed
     lea rax, [rel g_ht]                       ; table state base
     mov rcx, [rax + HT_TABLE]                 ; rcx = entry table pointer to free
     call mdb_free_mem                         ; free,table
     lea rax, [rel g_ht]                       ; reload base, volatile after call
     mov qword [rax + HT_TABLE], 0             ; null out table pointer, pure write avoids RMW stall
     mov dword [rax + HT_COUNT], 0             ; reset count, pure write avoids RMW stall
     mov qword [rax + HT_CAP], 0               ; reset capacity+mask in one wide pure write, avoids RMW stall
; Free the entire key+value pool in one shot, no per-entry free needed
     mov rcx, [rax + KP_BASE]                  ; pool base pointer, non-NULL: ht_init allocated it
     call mdb_free_mem                         ; free entire pool
     lea rax, [rel g_ht]                       ; reload base, volatile
     mov qword [rax + KP_BASE], 0              ; null out pool pointer, pure write avoids RMW stall
     mov qword [rax + KP_USED], 4              ; restore the offset-4 sentinel, matches ht_init state
     mov qword [rax + KP_CAP], 0               ; pool gone
     add rsp, 40                               ; reclaim shadow space
     ret                                       ; back to caller
ht_grow:
     push rbp                                  ; save non-volatile base pointer
     push rbx                                  ; save non-volatile
     push rsi                                  ; save non-volatile
     push r12                                  ; save non-volatile
     push r13                                  ; save non-volatile
     push r14                                  ; save non-volatile
     push r15                                  ; save non-volatile
     sub rsp, 32                               ; shadow space for mdb_alloc_mem/mdb_free_mem, 7 pushes = 56; 56+32=88 -> 0mod16 after entry 8mod16
     lea r12, [rel g_ht]                       ; r12 = g_ht base
     mov ebx, [r12 + HT_CAP]                   ; ebx = old capacity
     add rbx, rbx                              ; new capacity = old * 2
     lea r13d, [ebx - MASK_SUB_ONE]            ; r13d = new mask, wraparound mask fits 32-bit, cap is power of 2
     mov ecx, ebx                              ; new capacity
     imul rcx, rcx, ENTRY_SZ                   ; byte size of the new table
     call mdb_alloc_mem                        ; allocate zeroed new table
     mov r14, rax                              ; r14 = new table, zeroed
     mov r15, [r12 + HT_TABLE]                 ; r15 = old table
     mov ebp, [r12 + HT_CAP]                   ; ebp = old capacity, down-counter, dead after the loop; rbp saved by prologue
     mov rsi, r15                              ; rsi = &old_table[0], advanced incrementally, no per-slot address recompute
     align 16
.g_loop:
     mov eax, [rsi + E_HASH]                   ; load hash
     test eax, eax                             ; empty slot?
     jz .g_next                                ; yes, skip
     mov ecx, eax                              ; ecx = hash for the re-index below
     and ecx, r13d                             ; ecx = new_idx
     imul rax, rcx, ENTRY_SZ                   ; rax = new_idx * ENTRY_SZ, one-time probe-slot setup, 1 uop vs lea+shl
     lea r10, [r14 + rax]                      ; r10 = &new_table[new_idx], r10 unused elsewhere in ht_grow
.g_probe:
     cmp dword [r10], 0                        ; E_HASH == 0, empty ?
     je .g_place                               ; yes, use this slot
     add r10, ENTRY_SZ                         ; advance probe slot pointer, incremental
     inc ecx                                   ; linear probe
     cmp ecx, r13d                             ; idx <= mask, no wrap needed ?
     jbe .g_probe                              ; yes: keep looking
     xor ecx, ecx                              ; wrap, rare : idx = 0
     mov r10, r14                              ; probe slot = new table head
     jmp .g_probe                              ; keep looking
.g_place:
; Copy full 40-byte entry: 256-bit vector copy, bytes 0..31 + qword store, bytes 32..39
; Direct [r10] stores, r10 is the live slot pointer, no rdi staging copy needed
     vmovdqu ymm0, [rsi]                       ; load entry bytes 0..31, E_HASH+pad, E_DLEN, E_KLEN, E_FLAG, E_KEY
     vmovdqu [r10], ymm0                       ; store entry bytes 0..31
     mov rax, [rsi + E_DATA]                   ; load E_DATA qword, bytes 32..39
     mov [r10 + E_DATA], rax                   ; store E_DATA qword, bytes 32..39
.g_next:
     add rsi, ENTRY_SZ                         ; advance old-table slot pointer, incremental
     dec ebp                                   ; one fewer old slot to process, down-counter
     jnz .g_loop                               ; more old entries?, old cap >= MDB_INIT_CAP > 0, so the loop always runs
.g_done:
     vzeroupper                                ; clean dirty ymm0 upper bits BEFORE the free call, mdb_free_mem -> VirtualFree is SSE-capable; AVX-SSE transition rule
     mov rcx, r15                              ; old table pointer
     call mdb_free_mem                         ; free,old_table
     mov [r12 + HT_TABLE], r14                 ; store new table pointer
     mov [r12 + HT_CAP], ebx                   ; store new capacity
     mov [r12 + HT_MASK], r13d                 ; store new mask
     add rsp, 32                               ; reclaim shadow space
     pop r15                                   ; restore non-volatile
     pop r14                                   ; restore non-volatile
     pop r13                                   ; restore non-volatile
     pop r12                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     pop rbp                                   ; restore frame pointer
     ret                                       ; back to caller
     SF_START ht_add, rbx, rsi, rdi, r12, r13, r14, r15 ; Prologue, save non-volatiles
     SF_DWORD saved_idx                        ; saved probe index across calls
     SF_DWORD saved_pklen                      ; padded key length, cached once, saves recompute at 4 sites
     SF_DWORD saved_hash                       ; hash parked across the ht_grow call, edi is volatile
     SF_QWORD saved_slot                       ; saved slot pointer across calls
     SF_COMMIT                                 ; commit stack frame locals
     mov r12, rcx                              ; r12 = key pointer, non-volatile; keys are literals/atoms from the engine and documented-lifecycle hosts, a NULL key is outside the contract
     mov r13, rdx                              ; r13 = data pointer, non-volatile
     mov r14, r8                               ; r14 = data_len, non-volatile, 64-bit
     mov r15d, r9d                             ; r15d = persist flag, non-volatile, never overwritten
     mov rsi, r12                              ; rsi = key pointer, StringHash takes the key in rsi
     call StringHash                           ; edi = hash, normalized, ebx = klen = strlen + 1
; padded klen
     mov r8d, ebx                              ; r8d = klen
     ALIGN4 r8d                                ; r8d = pklen
     mov saved_pklen, r8d                      ; cache pklen in a frame slot, r8 is volatile and must be reloaded after every call
; Load table state
     lea r10, [rel g_ht]                       ; r10 = g_ht base
     mov r11, [r10 + HT_TABLE]                 ; r11 = table pointer, ht_init ran before any add: DllMain-ATTACH db_load, or the V7-DB-PUT lifecycle
     mov r9d, [r10 + HT_MASK]                  ; r9d = mask
; Initial probe: idx + one-time slot pointer, r9d already holds mask
     mov eax, edi                              ; eax = hash
     and eax, r9d                              ; idx = hash & mask
     imul rdx, rax, ENTRY_SZ                   ; rdx = idx * ENTRY_SZ, one-time slot setup, 1 uop vs lea+shl
     lea rsi, [r11 + rdx]                      ; rsi = &table[idx], advanced incrementally below
     align 16
.probe_loop:
     mov edx, [rsi + E_HASH]                   ; load stored hash
     test edx, edx                             ; empty slot?
     jz .empty_found                           ; yes, use this slot
     cmp edx, edi                              ; hash matches?
     jne .probe_next                           ; no
     cmp [rsi + E_KLEN], r8d                   ; pklen matches?
     jne .probe_next                           ; no
; Compare keys, 8-byte qword fast path + byte tail
     mov saved_idx, eax                        ; save idx
     mov rdx, [rsi + E_KEY]                    ; stored key pool offset
     add rdx, [r10 + KP_BASE]                  ; rdx = actual key pointer in pool
     mov rcx, r12                              ; rcx = caller's key
     mov r8d, ebx                              ; r8d = klen
     cmp r8d, 8                                ; klen >= 8 for qword fast path?
     jb .key_cmp_tail                          ; no, byte-by-byte for short keys
.key_cmp_qw:
     mov rax, [rdx]                            ; load 8 entry key bytes
     cmp rax, [rcx]                            ; compare with 8 caller key bytes
     jne .key_cmp_fail                         ; mismatch, keep probing
     add rdx, 8                                ; advance entry pointer by 8
     add rcx, 8                                ; advance caller pointer by 8
     sub r8d, 8                                ; 8 fewer bytes to compare
     cmp r8d, 8                                ; still >= 8 bytes remaining?
     jae .key_cmp_qw                           ; yes, continue qword compare, cmp on POST-sub value, sub's CF tests pre-sub, provably always clear here
.key_cmp_tail:
     test r8d, r8d                             ; any bytes left?
     jz .found                                 ; no, full match
     mov al, [rdx]                             ; load entry key byte
     cmp al, [rcx]                             ; compare with caller key byte
     jne .key_cmp_fail                         ; mismatch, keep probing
     inc rdx                                   ; advance entry key pointer
     inc rcx                                   ; advance caller key pointer
     dec r8d                                   ; one fewer byte
     jmp .key_cmp_tail                         ; continue byte compare
.key_cmp_fail:
     mov eax, saved_idx                        ; restore idx
     mov r8d, saved_pklen                      ; reload cached pklen, ebx=klen unchanged
.probe_next:
     add rsi, ENTRY_SZ                         ; advance slot pointer, incremental, no address recompute
     inc eax                                   ; idx + 1
     cmp eax, r9d                              ; idx <= mask, no wrap needed ?
     jbe .probe_loop                           ; yes: continue directly
     xor eax, eax                              ; wrap, rare : idx = 0
     mov rsi, r11                              ; slot = table head
     jmp .probe_loop                           ; continue probing
.empty_found:
     mov saved_slot, rsi                       ; save slot pointer, a tombstone can never exist: no delete path, one E_HASH writer, and StringHash normalises 0xFFFFFFFF away
; Check load factor, 64-bit arithmetic, prevents 32-bit overflow once count exceeds ~214M
; r10 still holds g_ht base from the setup, loop-invariant, no calls since
     mov ecx, [r10 + HT_COUNT]                 ; ecx = current count, 32-bit load, zero-extends to rcx
     inc rcx                                   ; rcx = count + 1, projected, 64-bit safe
     mov edx, [r10 + HT_CAP]                   ; edx = capacity, 32-bit load, zero-extends to rdx
     imul rcx, MDB_GROW_DEN                    ; count+1 * 10, 64-bit multiply
     imul rdx, MDB_GROW_NUM                    ; cap * 7, 64-bit multiply
     cmp rcx, rdx                              ; exceeds 70% threshold?
     jb .do_insert                             ; no, insert directly
     mov saved_hash, edi                       ; edi is volatile: park the hash before the call, restored below
     call ht_grow                              ; double capacity and rehash, loads its own g_ht copy
     lea r10, [rel g_ht]                       ; reload g_ht base, overwritten by ht_grow's internals
     mov r11, [r10 + HT_TABLE]                 ; new table pointer
     mov r9d, [r10 + HT_MASK]                  ; new mask
     mov r8d, saved_pklen                      ; reload cached pklen, klen unchanged by ht_grow
     mov edi, saved_hash                       ; restore the hash: ht_grow may overwrite edi, and .probe_loop plus the E_HASH store both read it after this point
     mov eax, edi                              ; hash
     and eax, r9d                              ; idx = hash & new_mask
     imul rdx, rax, ENTRY_SZ                   ; rdx = idx * ENTRY_SZ, one-time slot setup, 1 uop vs lea+shl
     lea rsi, [r11 + rdx]                      ; rsi = &table[idx], restart incremental probe
     jmp .probe_loop                           ; restart probing
.do_insert:
     mov rsi, saved_slot                       ; restore slot pointer
     mov ecx, r8d                              ; ecx = pklen bytes
     KP_ALLOC_INLINE                           ; rax = pool offset for the key, grows the pool if needed
; Write new-entry fields, hash, key length, key offset
     mov [rsi + E_HASH], edi                   ; store hash, marks slot occupied
     mov r8d, saved_pklen                      ; reload cached pklen
     mov [rsi + E_KLEN], r8d                   ; store padded key length
     mov [rsi + E_KEY], rax                    ; store key pool offset, rax = KP_ALLOC_INLINE's return, NO call between the macro and this store
; no E_FLAG store: a fresh slot's flags are already 0, zero-filled table, no
; tombstones and .store_data below writes the real persist flag unconditionally
; Copy key bytes to pool
     lea r10, [rel g_ht]                       ; reload g_ht base
     add rax, [r10 + KP_BASE]                  ; rax = actual key pointer in pool, rax still holds KP_ALLOC_INLINE's return, NO call since
     mov saved_slot, rsi                       ; save slot pointer
     mov rdi, rax                              ; rdi = dst = key in pool
     mov rsi, r12                              ; rsi = src = caller's key
     mov ecx, ebx                              ; ecx = klen
     rep movsb                                 ; copy key bytes to pool
; Zero-pad key buffer to pklen
     mov ecx, ebx                              ; klen
     mov eax, saved_pklen                      ; cached aligned pklen
     sub eax, ecx                              ; pad_bytes = pklen - klen
     jz .pad_done                              ; no padding needed
     mov ecx, eax                              ; pad byte count
     xor eax, eax                              ; zero byte
     rep stosb                                 ; write padding zeros
.pad_done:
     mov rsi, saved_slot                       ; restore slot pointer
     inc dword [r10 + HT_COUNT]                ; count++, new entry
     jmp .need_alloc                           ; a new entry needs its first value allocation
.found:
; Only an existing slot can reuse its value allocation.
     cmp qword [rsi + E_DLEN], r14             ; old_len >= new_len?
     jb .need_alloc                            ; no, old allocation too small, must allocate new
     mov r12, [rsi + E_DATA]                   ; reuse existing pool offset, no leak, no new allocation
     jmp .have_offset                          ; skip to data copy
.need_alloc:
     mov rcx, r14                              ; rcx = data_len bytes
     KP_ALLOC_INLINE                           ; rax = pool offset for the data, grows the pool if needed
     mov r12, rax                              ; r12 = data pool offset
.have_offset:
; Copy data bytes to pool
     lea r10, [rel g_ht]                       ; reload g_ht base
     mov rdi, [r10 + KP_BASE]                  ; pool base
     add rdi, r12                              ; rdi = dst = data in pool
     mov saved_slot, rsi                       ; save slot pointer
     mov rsi, r13                              ; rsi = src = caller data
     mov rcx, r14                              ; rcx = data_len, 64-bit
     rep movsb                                 ; copy data bytes to pool
     mov rsi, saved_slot                       ; restore slot pointer
.store_data:
; Store data pool offset and length
     mov [rsi + E_DATA], r12                   ; entry.data = pool offset
     mov [rsi + E_DLEN], r14                   ; data length, qword
     and r15d, 1                               ; isolate caller's persist flag
     mov [rsi + E_FLAG], r15d                  ; store flags once, plain store, zero RMW dependency
; Key-specific cache invalidation: a replacement of the "on_stream_evict"
; script must retire the cached eviction-callback program even when the new
; text has the SAME length as the old one (the length-only cache check cannot
; see it). Every writer converges here, the compare is exact (16 bytes, one
; entry), and eviction/metadata writes carry different keys, so the normal
; per-token callback keeps its O(1) cached-program path.
     mov rax, [rsi + E_KEY]                    ; rax = pool offset of this entry's key (qword: pool offsets are 64-bit)
     lea r10, [rel g_ht]                       ; g_ht base for the pool pointer
     add rax, [r10 + KP_BASE]                  ; rax = the entry's key bytes in the pool
     mov rcx, 0x61657274735F6E6F               ; "on_strea"
     cmp [rax], rcx                            ; first name half matches?
     jne .store_no_invalidate                  ; different key: keep the cache
     mov rcx, 0x0074636976655F6D               ; "m_evict" + NUL
     cmp [rax + EVICT_KEY_TAIL_OFF], rcx       ; second name half matches?
     jne .store_no_invalidate                  ; different key: keep the cache
     mov qword [rel stream_remove_prog], 0     ; drop the cached program: the next callback reparses
.store_no_invalidate:
     mov eax, 1                                ; return success
.ret:
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; Epilogue, restore non-volatiles
     SF_START ht_find, rbx, rsi, rdi, r12, r13, r14 ; Prologue, save non-volatiles
     SF_DWORD hf_saved_idx                     ; saved probe index across key_cmp, al overwritten by mov
     SF_COMMIT                                 ; Commit stack frame
     mov r12, rcx                              ; r12 = key pointer, keys are literals/atoms from the engine and documented-lifecycle hosts
     mov rsi, r12                              ; rsi = key pointer, StringHash takes the key in rsi
     call StringHash                           ; edi = hash, normalized, ebx = klen = strlen + 1
     mov r14d, ebx                             ; r14d = klen, ht_find's plan register
     mov rbx, rdx                              ; rbx = out_len pointer, moved after the call, ebx was overwritten
; padded klen
     mov r13d, r14d                            ; r13d = klen
     ALIGN4 r13d                               ; r13d = pklen
; Load table state
     lea rdx, [rel g_ht]                       ; rdx = g_ht base, rdx free: out_len saved to rbx above
     mov r10, [rdx + HT_TABLE]                 ; r10 = table pointer, ht_init ran before any find: DllMain-ATTACH db_load, or the V7-DB-GET lifecycle
; Initial probe: cache mask in r9d, read by the wrap test at the bottom of the probe loop + one-time slot pointer
     mov r9d, [rdx + HT_MASK]                  ; r9d = mask, cached, no RIP-relative reload per probe step
     mov eax, edi                              ; hash
     and eax, r9d                              ; idx = hash & mask
     imul rcx, rax, ENTRY_SZ                   ; rcx = idx * ENTRY_SZ, one-time slot setup, 1 uop vs lea+shl
     lea rsi, [r10 + rcx]                      ; rsi = &table[idx], advanced incrementally below
     align 16
.flp:
     mov ecx, [rsi + E_HASH]                   ; load stored hash
     test ecx, ecx                             ; empty slot?
     jz .nf                                    ; empty = key absent, terminate probe
     cmp ecx, edi                              ; hash matches?
     jne .fnext                                ; no, eax still holds idx, nothing above overwritten it
     cmp [rsi + E_KLEN], r13d                  ; pklen matches?
     jne .fnext                                ; no, eax still holds idx
     mov hf_saved_idx, eax                     ; save idx ONLY on the key-compare path, mirrors ht_add .key_cmp_fail pattern
; Compare keys using 8-byte qword fast path + byte tail
     mov rcx, [rsi + E_KEY]                    ; stored key pool offset
     add rcx, [rel g_ht + KP_BASE]             ; rcx = actual key pointer in pool
     mov rdx, r12                              ; rdx = caller's key
     mov r8d, r14d                             ; r8d = klen
     cmp r8d, 8                                ; klen >= 8 for qword fast path?
     jb .fkey_cmp_tail                         ; no, byte-by-byte for short keys
.fkey_cmp_qw:
     mov rax, [rcx]                            ; load 8 entry key bytes
     cmp rax, [rdx]                            ; compare with 8 caller key bytes
     jne .fkey_fail                            ; mismatch, keep probing, eax overwritten, restore from slot
     add rcx, 8                                ; advance entry pointer by 8
     add rdx, 8                                ; advance caller pointer by 8
     sub r8d, 8                                ; 8 fewer bytes to compare
     cmp r8d, 8                                ; still >= 8 bytes remaining?
     jae .fkey_cmp_qw                          ; yes, continue qword compare, cmp on POST-sub value, sub's CF tests pre-sub, provably always clear here
.fkey_cmp_tail:
     test r8d, r8d                             ; any bytes left?
     jz .fd                                    ; no, full match
     mov al, [rcx]                             ; load entry key byte, overwrites low byte of eax
     cmp al, [rdx]                             ; compare with caller key byte
     jne .fkey_fail                            ; mismatch, keep probing, eax overwritten, restore from slot
     inc rcx                                   ; advance entry key pointer
     inc rdx                                   ; advance caller key pointer
     dec r8d                                   ; one fewer byte
     jmp .fkey_cmp_tail                        ; continue byte compare, exits via jz .fd / jne .fkey_fail above
.fkey_fail:
     mov eax, hf_saved_idx                     ; restore idx, key compare overwritten eax; rsi preserved by compare
.fnext:
     add rsi, ENTRY_SZ                         ; advance slot pointer, incremental, no address recompute
     inc eax                                   ; idx + 1
     cmp eax, r9d                              ; idx <= mask, no wrap needed ?
     jbe .flp                                  ; yes: continue directly
     xor eax, eax                              ; wrap, rare : idx = 0
     mov rsi, r10                              ; slot = table head
     jmp .flp                                  ; keep probing
.fd:
     test rbx, rbx                             ; out_len pointer provided?
     jz .fd2                                   ; no, skip write
     mov rax, [rsi + E_DLEN]                   ; load full 64-bit entry data length
     mov [rbx], rax                            ; store full 64-bit length to *out_len
.fd2:
     mov rax, [rsi + E_DATA]                   ; data pool offset
     add rax, [rel g_ht + KP_BASE]             ; rax = data pointer, base + offset
     SF_END rbx, rsi, rdi, r12, r13, r14       ; Epilogue, restore non-volatiles
.nf:
     xor eax, eax                              ; return NULL, not found
     SF_END rbx, rsi, rdi, r12, r13, r14       ; Epilogue, restore non-volatiles
ht_find_or_fatal:
     push rbx                                  ; save non-volatile, holds key ptr across all calls
     push rdi                                  ; save non-volatile, STR_LEN overwrites rdi
     sub rsp, 40                               ; 32 shadow + 8 alignment pad, entry 8 mod 16, two pushes -> 8, 40 keeps call sites at 0 mod 16
     mov rbx, rcx                              ; rbx = key pointer, survives ht_find, debug_ensure_std, API_CALL
; rcx, key and rdx, out_len already set, pass through to ht_find
     call ht_find                              ; rax = data ptr or NULL
     test rax, rax                             ; found?
     jnz .hff_found                            ; yes, return data ptr directly
; --- Key not found: print key name to stdout, then terminate ---
     call debug_ensure_std                     ; ensure stdout_handle is cached, overwrites rax,rcx,rdx,r8-r11
; Write diagnostic prefix: "[FATAL] ht_find_or_fatal: key not found: "
     lea rdx, [hff_prefix]                     ; rdx = prefix string
     STR_LEN rdx, r8d                          ; r8d = prefix length, overwrites rax, rcx, rdi
; No "empty prefix?" test: hff_prefix is a fixed literal, see its db
; line below, so its length is a compile-time constant and can never be 0.
     mov rcx, [stdout_handle]                  ; arg1 = stdout handle, reload, STR_LEN overwritten rcx
     xor r9d, r9d                              ; arg4 = NULL: the bytes-written out-pointer is OPTIONAL and discarded here
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write prefix string to stdout
; Write the actual key name that failed lookup
     mov rdx, rbx                              ; rdx = key string, rbx survived all overwrites
     STR_LEN rdx, r8d                          ; r8d = key length, overwrites rax, rcx, rdi
     test r8d, r8d                             ; empty key, nothing to name ?
     jz .hff_nl                                ; skip WriteFile, go straight to the newline
     mov rcx, [stdout_handle]                  ; arg1 = stdout handle, reload, STR_LEN overwritten rcx
     xor r9d, r9d                              ; arg4 = NULL: optional out-pointer discarded
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write key name string to stdout
.hff_nl:
; Write trailing newline for clean stdout output
     lea rdx, [fmt_instr_nl]                   ; rdx = "\n", 1-byte string
     mov rcx, [stdout_handle]                  ; arg1 = stdout handle
     mov r8d, 1                                ; arg3 = 1 byte
     xor r9d, r9d                              ; arg4 = NULL: optional out-pointer discarded
     API_CALL [__imp_WriteFile], rcx, rdx, r8, r9, 0 ; write newline character to stdout
; Terminate, diagnostic already printed above
     FATAL "ht_find_or_fatal"                  ; prints "[FATAL] ht_find_or_fatal" + TerminateProcess
.hff_found:
     add rsp, 40                               ; reclaim shadow + alignment pad
     pop rdi                                   ; restore non-volatile rdi
     pop rbx                                   ; restore non-volatile rbx
     ret                                       ; return with rax = data pointer from ht_find
; --- Diagnostic prefix string, embedded in .text, 8-byte aligned for LEA ---
     align 8
hff_prefix:
     db "[FATAL] ht_find_or_fatal: key not found: ", 0 ; diagnostic prefix for stdout output
