; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define SL_SLEN_WIN_BYTES 16                   ; AVX2 window width, bytes per iteration
%define SL_QWORD_SCALE 8                       ; sizeof,qword, START bitmap word stride
%define SL_TERMINATOR_OFF -1                   ; NUL offset from the exclusive source end
%if SL_CONCAT_BUF_SIZE > SL_CONS_CELL_COUNT * SL_CONS_CELL_BYTES
%error Intern candidate probe must fit within the committed heap ; Keep the maximum terminator offset inside the heap
%endif
%define SL_TAIL_DWORD_OFF -4                   ; tail dword starts four bytes before the exclusive end
%define SL_TAIL_WORD_OFF -2                    ; tail word starts two bytes before the exclusive end
sl_swar_strlen:
     vpxor xmm0, xmm0, xmm0                    ; zero vector for NUL detection
     xor ecx, ecx                              ; byte count accumulator
.sl_slen_window:
     vpcmpeqb xmm1, xmm0, [rsi + rcx]          ; NUL bytes -> 0xFF lanes, load folded into the compare
     vptest xmm1, xmm1                         ; any NUL lane this window?
     jz .sl_slen_next                          ; no: advance a full 16 bytes
     vpmovmskb edx, xmm1                       ; NUL position bitmask
     tzcnt edx, edx                            ; edx = NUL offset within the window
     add ecx, edx                              ; ecx = total length, windows*16 + offset
     ret                                       ; return strlen in ecx
.sl_slen_next:
     add ecx, SL_SLEN_WIN_BYTES                ; 16 more bytes counted
     jmp .sl_slen_window                       ; next window
Intern:
     push rbx                                  ; save non-volatile, 3 pushes: Win64 entry 8 mod 16 -> 0 mod 16
     push rsi                                  ; save non-volatile, hashes preserve it; movsb advances it
     push rdi                                  ; save non-volatile, overwritten by string ops
     call StringHash                           ; edi = h1, crc32, normalized, ebx = klen = strlen + 1
; --- slot cache: probed FIRST, CRC-filtered then byte-verified hit; the bloom gates only run
; on a slot miss. A slot hit implies the bits are SET: the atom was interned
; before and the filter is never cleared, so the hit path skips them entirely ---
     mov eax, edi                              ; Use the primary hash to select a set
     and eax, SL_INTERN_SETS - 1               ; Isolate the set index
     shl eax, SL_INTERN_SET_SHL                ; Scale to the set's byte offset
     lea r11, [rel sl_intern_cache]            ; Load the cache base
     cmp dword [r11 + rax], edi                ; Is the first way a hash match?
     jne .check_other_ways                     ; Check the other ways on a miss
     mov edx, [r11 + rax + SL_INTERN_ATOMS_OFF] ; Load the first way's atom offset
     add rdx, r13                              ; Resolve the candidate address
     call sl_intern_key_equal                  ; Verify the first way without vector setup
     je .slot_hit                              ; Return the verified atom
.check_other_ways:
     vmovd xmm2, edi                           ; Move the search hash to a vector lane
     vpshufd xmm2, xmm2, 0                     ; Broadcast the hash across the four ways
     vpcmpeqd xmm2, xmm2, [r11 + rax]          ; Test all four stored hashes at once
     vmovmskps ecx, xmm2                       ; Extract one bit per matching way
     and ecx, 0b1110                           ; Exclude the first way already checked
     jz .bloom_gate                            ; No match in the cache
.cache_candidate:
     tzcnt edx, ecx                            ; Find the first remaining matching way
     lea rax, [rax + rdx*SL_INTERN_LANE_BYTES] ; Select the matching lane within the set
     mov edx, [r11 + rax + SL_INTERN_ATOMS_OFF] ; Load the candidate atom offset
     add rdx, r13                              ; Resolve the candidate address
     call sl_intern_key_equal                  ; Check exact equality after the hash filter
     je .slot_hit                              ; Return the verified atom
     and eax, -SL_INTERN_SET_BYTES             ; Recover the set offset after a hash collision
     blsr ecx, ecx                             ; Discard the way just checked
     jnz .cache_candidate                      ; Check other ways with the same hash
     jmp .bloom_gate                           ; No cached atom matched exactly
.slot_hit:
     mov eax, [r11 + rax + SL_INTERN_ATOMS_OFF] ; Return the atom offset from the verified way
     jmp .ret                                  ; Its Bloom bits and cache entry are already set
.slot_filler:
; eax = atom offset, append ; edi = h1; r10 = h2, live on every path here
     mov edx, edi                              ; edx = h1
     and edx, SL_INTERN_FILTER_MASK            ; bloom bit index
     bts dword [rel sl_intern_filter], edx     ; set bloom bit 1, h1
     mov rdx, r10                              ; rdx = h2
     and rdx, SL_INTERN_FILTER_MASK            ; bloom bit index
     bts qword [rel sl_intern_filter], rdx     ; set bloom bit 2, h2
.slot_store:
     mov edx, edi                              ; Use the same hash to select the set
     and edx, SL_INTERN_SETS - 1               ; Isolate the set index
     shl edx, SL_INTERN_SET_SHL                ; Scale to the set's byte offset
     lea r11, [rel sl_intern_cache]            ; Load the cache base
     vpxor xmm2, xmm2, xmm2                    ; Empty hash lanes contain zero
     vpcmpeqd xmm2, xmm2, [r11 + rdx]          ; Test the four ways for an empty slot
     vmovmskps ecx, xmm2                       ; Extract one bit per empty way
     tzcnt ecx, ecx                            ; Pick the first empty way, CF marks a full set
     jnc .cache_write                          ; Use the free way
     mov ecx, edi                              ; Use high hash bits to choose a way in a full set
     shr ecx, SL_INTERN_VICTIM_SHIFT           ; Select one of four ways independent of the set index
.cache_write:
     lea rdx, [rdx + rcx*SL_INTERN_LANE_BYTES] ; Select the way within the set
     mov [r11 + rdx], edi                      ; Store the primary hash
     mov [r11 + rdx + SL_INTERN_ATOMS_OFF], eax ; Store the atom offset in the parallel vector
     jmp .ret                                  ; eax = atom offset
; --- bloom gate: reached only on a slot miss, both bits clear = never interned ---
.bloom_gate:
     mov r8d, ebx                              ; r8d = klen for the secondary AES hash
     call aes_string_hash                      ; rax = h2, u64, computed once per miss
     mov r10, rax                              ; h2 for the bloom bit and the new-atom insertion
     mov eax, edi                              ; eax = h1
     and eax, SL_INTERN_FILTER_MASK            ; bit index within the 2^22-bit field
     bt dword [rel sl_intern_filter], eax      ; h1 bit set?
     jnc .append_new                           ; clear = NEVER interned: skip scan, append now
     mov rax, r10                              ; rax = h2
     and rax, SL_INTERN_FILTER_MASK            ; bit index within the 2^22-bit field, a qword bit base divides the index by 64
     bt qword [rel sl_intern_filter], rax      ; h2 bit set?
     jnc .append_new                           ; clear = NEVER interned: append now
; --- fallback: linear scan, candidate list = the START bit index ---
.linear_scan:
     push r14                                  ; save non-volatile, r14 = loop-invariant START bitmap base
     push rdi                                  ; save h1, scan + compare overwrite rdi
     push r15                                  ; pending start-bit mask, no calls inside scan
     mov r11d, [rel sl_atom_free]              ; granule bound: loop-invariant for the whole scan, no calls inside, so no collect can move the cursor mid-walk
     shr r11d, 3                               ; bound in granules, ecx counts granules
     xor ecx, ecx                              ; scan cursor = granule 0, NIL and "" are both indexed
     lea r14, [rel sl_atom_start_bits]         ; r14 = START index base, loop-invariant
.scan:
     cmp ecx, r11d                             ; past all interned atoms?
     jae .not_found                            ; yes: allocate new, unsigned offset
     mov eax, ecx                              ; ecx is 64-granule-aligned here
     shr eax, 6                                ; word index = granule / 64
     mov r15, qword [r14 + rax*SL_QWORD_SCALE] ; pending START bits, retained across comparisons
.scan_bit:
     tzcnt rdx, r15                            ; low set bit; CF=1 when the word is exhausted
     jc .scan_skip64                           ; zero word: advance by a whole word
     lea ecx, [rcx + rdx]                      ; absolute candidate granule (word base + bit)
     cmp ecx, r11d                             ; this last word may extend beyond the heap cursor
     jae .not_found                            ; candidate beyond atom_free is not allocated
; --- candidate atom ---
     shl ecx, 3                                ; ecx = candidate byte offset, the .match tail returns it
     lea rdx, [r13 + rcx]                      ; candidate atom in the heap
     mov al, [rdx]                             ; reject a different type before calling the comparer
     cmp al, [rsi]                             ; source type byte
     jne .scan_miss                            ; different type
     call sl_intern_key_equal                  ; ZF = equality through the source NUL
     je .match                                 ; found the existing atom
.scan_miss:
     shr ecx, 3                                ; candidate byte offset -> granule
     and ecx, -64                              ; restore block base (64 granules)
     blsr r15, r15                             ; discard the tested lowest START bit
     jmp .scan_bit                             ; next set bit, no bitmap reload
.scan_skip64:
     add ecx, 64                               ; advance one complete 64-granule block
     jmp .scan                                 ; load the next bitmap word
.match:
     mov eax, ecx                              ; eax = matched atom byte offset, ecx = candidate offset at .match
     pop r15                                   ; restore pending start-bit mask
     pop rdi                                   ; restore h1
     pop r14                                   ; restore non-volatile, reverse push order
     jmp .slot_store                           ; slot refresh only: every atom the scan can find was interned through .slot_filler, so both bloom bits are already SET and re-setting them is a no-op
.not_found:
     pop r15                                   ; restore pending start-bit mask
     pop rdi                                   ; restore h1 for filter/slot writeback
     pop r14                                   ; restore non-volatile, reverse push order
.append_new:
     mov edx, edi                              ; park h1 in edx, survives the movsb; NOT the collect, this instruction re-parks from edi on every .append_new entry, including the post-collect retries
     mov r9, rbx                               ; r9 = klen
     SL_ALIGN8 r9                              ; r9 = need = align8,klen, C5: align the INCREMENT
     lea r8, [rel sl_atom_freelist]            ; address of the link to the current block, head variable acts as the first link
.alloc_ff_loop:
     mov r11d, [r8]                            ; read current block offset through that link
     test r11d, r11d                           ; end of the free list?
     jz .alloc_bump                            ; no fitting block
     cmp dword [r13 + r11 + SL_ATOM_FB_LEN], r9d ; does this block fit?
     jae .alloc_ff_take                        ; keep the first fit
     lea r8, [r13 + r11 + SL_ATOM_FB_LINK]     ; follow its next-link address
     jmp .alloc_ff_loop                        ; inspect the following block
.alloc_ff_take:
     mov eax, r11d                             ; selected block offset
     mov r11d, [r13 + r11 + SL_ATOM_FB_LINK]   ; selected block's successor
     mov [r8], r11d                            ; unlink through the same link, including at the head
     mov r11d, [rel sl_atom_freelist]          ; r11d = current head, for the remainder's link
     mov ecx, [r13 + rax + SL_ATOM_FB_LEN]     ; ecx = block length, >= need, from the fit test
     sub ecx, r9d                              ; ecx = remainder bytes
     cmp ecx, SL_ATOM_FB_HDR                   ; remainder big enough for a header?
     jb .alloc_ff_took                         ; no: consume the whole block, the tail waste stays in the block
     lea r8, [rax + r9]                        ; r8 = remainder offset, off + need
     mov [r13 + r8 + SL_ATOM_FB_LEN], ecx      ; remainder [len]
     mov [r13 + r8 + SL_ATOM_FB_LINK], r11d    ; remainder [next] = current head
     mov [rel sl_atom_freelist], r8d           ; remainder becomes the head
.alloc_ff_took:
     jmp .alloc_store                          ; the atom bytes go in at eax
.alloc_bump:
     mov eax, [rel sl_atom_free]               ; eax = cursor offset, 8-aligned by induction, C5
     lea rcx, [rax + rbx]                      ; rcx = cursor + klen, the soft-limit test uses the tight end
     cmp rcx, SL_ATOM_GC_SOFT_OFF              ; past the soft limit?
     jbe .alloc_hard                           ; no: take the hard bounds; past the limit falls into the slow collect
.alloc_slow:
     cmp eax, [rel sl_atom_gc_restart_at]      ; restart point reached?
     jb .alloc_hard                            ; no: cursor is below the published restart point, a prior collect throttles the next one; restart_at == 0 never takes this branch, a fresh heap falls through and collects on first soft crossing
     mov dword [rel sl_gc_in_retry], 0         ; a fresh collect is staged for THIS request, C12
     lea rdx, [rax + SL_ATOM_GC_RESTART_BYTES] ; rdx = restart point
     mov [rel sl_atom_gc_restart_at], edx      ; publish the restart point BEFORE the collect, E12: the collect itself can move the cursor
     mov dword [rel sl_gc_want_atoms], 1       ; gate the atom half of the collect, B10
     push r10                                  ; h2 is volatile-overwritten by GC_Collect, B3
     sub rsp, 8                                ; parity pad: 4 "pushes" = 8 mod 16, pad restores 0, D17
     call GC_Collect                           ; mark + sweep, cons half always, atom half gated
     add rsp, 8                                ; drop the parity pad
     pop r10                                   ; recover h2
     mov dword [rel sl_gc_want_atoms], 0       ; collect done: the atom half is off again
     jmp .append_new                           ; retry from step 1, the sweep may have linked a fitting block; the entry's own store re-parks h1 from edi
.alloc_hard:
     cmp rcx, SL_ATOM_LIMIT                    ; would the copy run past the usable atom area? The guard pad remains reserved.
     ja .atom_full                             ; yes: terminal, the collect retry below is bounded
     lea r11, [rax + r9]                       ; r11 = next cursor, aligned increment
     mov [rel sl_atom_free], r11d              ; publish the advanced cursor
.alloc_store:
     mov dword [rel sl_gc_in_retry], 0         ; a successful append re-enables the one-shot collect, a spent .atom_full retry must not leave the flag set for the NEXT hard-arena trip
     mov ecx, ebx                              ; rep count = klen including the NUL, zero-extends into rcx
     lea rdi, [r13 + rax]                      ; rdi = destination in atom area, write from type byte
     rep movsb                                 ; copy string + NUL, rdi/rsi/rcx advance; eax/edx/r10 survive
     mov edi, edx                              ; restore h1 from the park
     mov ecx, eax                              ; ecx = a copy of the atom offset, dword: the arena is < 4 GiB, and this NASM rejects mixed-width moves
     SL_ATOM_BIT sl_atom_start_bits, bts       ; set the START bit for the new atom, overwrites eax/edx/r9
     mov eax, ecx                              ; eax = the atom offset again
     jmp .slot_filler                          ; returns eax as the new atom's offset
.ret:
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     ret                                       ; return atom offset in eax
.atom_full:
; out of line: the collect retry is the rare path
     cmp dword [rel sl_gc_in_retry], 0         ; already collected for THIS request?
     jnz .atom_full_fatal                      ; yes: a second collect still did not fit -> terminal
     mov dword [rel sl_gc_in_retry], 1         ; this request has now spent its collect
     mov dword [rel sl_gc_want_atoms], 1       ; gate the atom half of the collect, B10
     push r10                                  ; h2 is volatile-overwritten by GC_Collect, B3
     sub rsp, 8                                ; parity pad, D17
     call GC_Collect                           ; UNCONDITIONAL collect: the area bound is absolute, and it may run with the cursor far past the soft limit
     add rsp, 8                                ; drop the parity pad
     pop r10                                   ; recover h2
     mov dword [rel sl_gc_want_atoms], 0       ; collect done
     mov r8d, [rel sl_atom_free]               ; r8d = the post-collect cursor, possibly rewound
     add r8d, SL_ATOM_GC_RESTART_BYTES         ; r8d = the new candidate restart point
     cmp r8d, [rel sl_atom_gc_restart_at]      ; above the previously published point?
     jbe .atom_full_restart_set                ; no: the new point is the min, take it
     mov r8d, [rel sl_atom_gc_restart_at]      ; yes: keep the old, lower point
.atom_full_restart_set:
     mov [rel sl_atom_gc_restart_at], r8d      ; publish min,old, cursor + RESTART
     jmp .append_new                           ; retry: the sweep may have linked a fitting block, C12
.atom_full_fatal:
     FATAL "SectorLogo: Atom area overflow (GC exhausted)" ; no space above the soft limit even after a collect
sl_intern_key_equal:
; Private leaf. RSI = source, RDX = candidate, EBX = source size including NUL.
; StringHash establishes the source extent. Intern inputs are bounded by the
; concat buffer. The committed cons area beyond the atom area covers a probe
; at that offset from any live atom. A zero there still needs the full compare.
; Preserves RAX/RCX/RDI; returns equality in ZF.
     push rax                                  ; preserve slot offset and scan scratch
     cmp byte [rdx + rbx + SL_TERMINATOR_OFF], 0 ; Could the candidate end at the source's NUL?
     jne .result_neq                           ; Reject a mismatch before comparing the shared prefix
     mov r8, rsi                               ; source cursor
     mov r9d, ebx                              ; bytes still to check
.vec:
     cmp r9d, 32                               ; a complete vector remains?
     jb .qword                                 ; use smaller exact reads
     vmovdqu ymm1, [rdx]                       ; candidate bytes
     vpxor ymm1, ymm1, [r8]                    ; source bytes, all lie inside klen
     vptest ymm1, ymm1                         ; did any byte differ?
     jnz .result_neq                           ; differ: return ZF=0
     add rdx, 32                               ; advance candidate
     add r8, 32                                ; advance source
     sub r9d, 32                               ; remaining bytes
     jmp .vec                                  ; continue full blocks
.qword:
     cmp r9d, 8                                ; a full qword remains?
     jb .short                                 ; finish the 0..7 byte tail
     mov rax, [rdx]                            ; candidate qword
     cmp rax, [r8]                             ; source qword
     jne .result_neq                           ; differ: return ZF=0
     add rdx, 8                                ; advance candidate
     add r8, 8                                 ; advance source
     sub r9d, 8                                ; count the bytes consumed
     jmp .qword                                ; compare another full qword if possible
.short:
     cmp r9d, 4                                ; at least one dword?
     jb .short2                                ; check 0..3 bytes
     mov eax, [rdx]                            ; candidate head dword
     cmp eax, [r8]                             ; source head dword
     jne .result_neq                           ; differ: return ZF=0
     mov eax, [rdx + r9 + SL_TAIL_DWORD_OFF]   ; candidate tail dword
     cmp eax, [r8 + r9 + SL_TAIL_DWORD_OFF]    ; source tail covers every remaining byte
     jne .result_neq                           ; differ: return ZF=0
     jmp .result_eq                            ; equal: ZF=1 from the compare
.short2:
     cmp r9d, 2                                ; two or three bytes?
     jb .short1                                ; finish zero or one byte
     mov ax, [rdx]                             ; candidate head word
     cmp ax, [r8]                              ; source head word
     jne .result_neq                           ; differ: return ZF=0
     mov ax, [rdx + r9 + SL_TAIL_WORD_OFF]     ; candidate tail word
     cmp ax, [r8 + r9 + SL_TAIL_WORD_OFF]      ; source tail word
     jne .result_neq                           ; differ: return ZF=0
     jmp .result_eq                            ; equal: ZF=1 from the compare
.short1:
     test r9d, r9d                             ; empty tail?
     jz .result_eq                             ; yes: ZF=1 from the test
     mov al, [rdx]                             ; last candidate byte
     cmp al, [r8]                              ; last source byte
     jne .result_neq                           ; differ: return ZF=0
.result_eq:
     vzeroupper                                ; clear upper YMM halves without changing flags
     pop rax                                   ; restore caller state without changing flags
     ret                                       ; return ZF=1, equal through the source NUL
.result_neq:
     vzeroupper                                ; clear upper YMM halves without changing flags
     pop rax                                   ; restore caller state without changing flags
     ret                                       ; return ZF=0, reached only from a taken jne
; ---- %undef cleanup: file-local defines leave scope here, house convention ----
%undef SL_SLEN_WIN_BYTES
%undef SL_QWORD_SCALE
%undef SL_TERMINATOR_OFF
