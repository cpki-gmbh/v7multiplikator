; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define SL_GRANULE_SCALE 8                     ; SIB scale: atom granule -> byte offset

%macro GC_MARK_LOOP 4
%2:                                            ; entry label
     cmp rsp, [rel sl_stk_limit]               ; private-stack watermark: car-chains can nest arbitrarily deep, no parser cap bounds built-at-runtime trees
     jb sl_stack_overflow                      ; runaway recursion -> named FATAL, not an AV
%3:                                            ; loop label, the cdr tail-call target
     test eax, eax                             ; SF = bit31: clear = NIL,0 /atom/NUMBER -> GC-inert, one test, one branch
     jns %4                                    ; skip marking; FLOAT, bit31 set falls into the range rejection below unchanged
; strip cons tag to get cell index
     mov ecx, eax                              ; ecx = tagged value
     btr ecx, SL_CONS_BIT                      ; strip tag bit -> raw cell index, 4B vs 6B and
     cmp ecx, [rel sl_cons_free]               ; index within allocated range?
     jae %4                                    ; no: false positive, skip
; check/set mark bit in bitmap
     bts dword [rel sl_gc_bitmap], ecx         ; set mark bit of cell ecx; CF=1 if already marked, memory-operand bts selects dword+bit from the full index
     jc %4                                     ; already marked: skip, prevent infinite loop
; mark CAR recursively, push, recurse, pop
     lea r10, [r13 + rcx*CONS_SCALE + CONS_BASE_OFF] ; address of cons cell
     push r10                                  ; save cell address across CAR mark
     mov eax, [r10]                            ; eax = car field
     call %2                                   ; recursively mark car subtree, SAME instantiation, so the car walks with the arm too
     pop r10                                   ; restore cell address
; mark CDR via tail-call, jmp, not call for O,1 stack depth
     mov eax, [r10 + CONS_CDR_OFF]             ; eax = cdr field
     jmp %3                                    ; tail-call: mark cdr without growing stack
%4:                                            ; done label: NIL/atom/NUMBER/FLOAT/false positive
%if %1 == 1
     cmp eax, [rel sl_atom_free]               ; payload >= cursor, or a tag bit set ?
     jae %%atom_ret                            ; yes: not a plausibly-live atom offset
     SL_ATOM_BIT sl_atom_mark_bits, bts        ; in range: set this atom's MARK bit, shared macro
%%atom_ret:
%endif
     ret                                       ; return to caller
%endmacro
     GC_MARK_LOOP 0, GC_Mark, .gm_loop, .gm_done ; cons-only: the plain cons walker
     GC_MARK_LOOP 1, GC_MarkAtom, .gma_loop, .gma_done ; cons + atoms: used ONLY inside the sl_gc_want_atoms gate
GC_Collect:
     SF_START GC_Collect, rbx, rsi, rdi, r12, r13, r14, r15 ; save non-volatiles, LOAD-BEARING for Logo non-volatile-held partial lists: the spill lands in the scanned range, never trim this list
     SF_COMMIT                                 ; commit frame
; --- Step 1: Clear mark bitmap, only active portion ---
     mov ecx, [rel sl_cons_free]               ; number of allocated cells
     add ecx, 63                               ; round up to next qword boundary
     shr ecx, 6                                ; ecx = ceil,sl_cons_free / 64 qwords, sets ZF for the next test
     jz .after_cons_clear                      ; yes: skip ONLY the rep stosq, with an atom mark phase staged, a jump to .gc_sweep here would skip the atom roots + stack scan and reclaim every atom
     lea rdi, [rel sl_gc_bitmap]               ; rdi = bitmap base address
     xor eax, eax                              ; fill value = 0
     rep stosq                                 ; clear bitmap: rcx qwords starting at rdi
.after_cons_clear:
     cmp dword [rel sl_gc_want_atoms], 0       ; atom half staged for this collect?
     jz .gc_roots                              ; no: cons-only collect, skip the two memsets, then mark the cons roots anyway
; clear the per-collect atom mark array: [0, ceil,sl_atom_free/64 bytes.
; A clear inside the sweep would destroy the very marks it reads.
     mov ecx, [rel sl_atom_free]               ; ecx = cursor byte count, the meaningful bitmap extent
     add ecx, 63                               ; round up to byte, ceiling within the bitmap
     shr ecx, 6                                ; ecx = bitmap BYTES covering every indexed granule, bitmap has 8 bits per byte, so granules/64
     lea rdi, [rel sl_atom_mark_bits]          ; rdi = liveness array base
     xor eax, eax                              ; fill value = 0
     rep stosb                                 ; erase the previous collect's marks
; --- Step 2: Mark exact machine roots ---
.gc_roots:
; Root 1: sl_lg_callback_prog, parsed root AST, BSS global, invisible
; to the stack scan below, hence marked explicitly
     mov eax, [rel sl_lg_callback_prog]        ; load callback program
     call GC_MarkAtom                          ; mark it, the atom-marking walker, so a prologue atom is marked too
; Root 2: sl_fn_return_val, active return value, BSS global
     mov eax, [rel sl_fn_return_val]           ; load return value
     call GC_MarkAtom                          ; mark it
     mov eax, [rel sl_stmt_fn_retval]          ; load the published value, 0 = nothing published
     test eax, eax                             ; channel empty?, no fn ran yet, or already consumed
     jz .gc_root2b_done                        ; yes: nothing to mark
     call GC_MarkAtom                          ; mark the value list and its whole graph
.gc_root2b_done:
     mov eax, r12d                             ; load stream cursor
     call GC_MarkAtom                          ; mark it
; Root 4: r14d = current environment alist, threaded through the
; evaluator in r14, same reasoning as Root 3
     mov eax, r14d                             ; load env alist
     call GC_MarkAtom                          ; mark it
     mov eax, [rel sl_lg_autoexec_prog]        ; load autoexec program
     call GC_MarkAtom                          ; mark it
     mov eax, [rel sl_lg_base_env]             ; load base env alist
     call GC_MarkAtom                          ; mark it
; Root 9b: stream_remove_prog, cached parsed "on_stream_evict" script
     mov eax, [rel stream_remove_prog]         ; load eviction-callback prog, 0 = never parsed
     test eax, eax                             ; slot empty?
     jz .gc_root10                             ; NULL slot -> skip marking
     call GC_MarkAtom                          ; mark the parsed eviction-callback prog tree
.gc_root10:
     mov ebx, [rel sl_local_count]             ; ebx = live entries, 0 = none declared
     test ebx, ebx                             ; any LOCAL declared in this scope?
     jz .gc_root10_done                        ; no: nothing to walk
     xor r15d, r15d                            ; root-array entry index; r14/r15 belong to the scan setup below, which is initialized AFTER the roots
     lea rsi, [rel sl_local_set]               ; preserved by GC_MarkAtom (Win64 non-volatile), loop-invariant
.gc_root10_loop:
     mov eax, [rsi + r15*SL_LOCAL_SET_SCALE]   ; eax = entry's var atom, dword at +0
     call GC_MarkAtom                          ; mark the var atom, no NIL test: a live entry's var name is never NIL by construction, and the arm's single range check makes even a stray 0 inert
     inc r15d                                  ; advance to the next entry
     cmp r15d, ebx                             ; walked every live entry?
     jb .gc_root10_loop                        ; no: keep going
.gc_root10_done:
; Root 11: the primitive-name region below the dispatch high-water mark.
; Every atom offset below the floor is engine infrastructure, the builtin
; dispatch keys, the stmt/expr tables are indexed by their byte offsets and
; the init-time atom handles, atom_sl_t, atom_sl_lg_piece, ....No list, env,
; or stack slot references them, so the mark phase would otherwise reclaim
; them; a reclaimed key's offset is then reused by a data atom and the
; dispatcher routes that new atom to the primitive's handler, IFELSE streams derail silently. Mark the whole range: ~1 KB, one bit per granule, per collect.
     cmp dword [rel sl_gc_want_atoms], 0       ; atom collect staged?
     je .gc_root11_done                        ; cons-only collect: nothing ever reads the atom mark array this cycle, the staged collect clears it before its own mark phase, so pinning would be dead work only
     mov ecx, [rel sl_dispatch_highwater]      ; ecx = first non-builtin byte offset, pin floor
     test ecx, ecx                             ; dispatch fill never ran, pre-init collect ?
     jz .gc_root11_done                        ; yes: nothing to pin
     mov edx, ecx                              ; edx = pin floor in bytes
     shr ecx, 3                                ; ecx = pin floor in granules
     mov r8d, ecx                              ; r8d = granule count, edge copy
     and r8d, 7                                ; r8d = partial granules in the trailing bitmap byte, 0..7
     shr edx, 6                                ; edx = FULL bitmap bytes covering the floor, 64 bytes per bitmap byte
     lea rdi, [rel sl_atom_mark_bits]          ; rdi = bitmap base, rdi dead here: last use was the bitmap memset at the collect head; saved/restored by SF_START/SF_END
     mov eax, 0xFF                             ; all 8 granule bits of a byte
     mov ecx, edx                              ; rep consumes ecx
     rep stosb                                 ; pin the full bytes in one ERMS burst
     test r8d, r8d                             ; trailing partial bitmap byte?
     jz .gc_root11_done                        ; no: floor was 64-byte aligned
     mov eax, -1                               ; all ones; BZHI keeps the low partial bits
     bzhi eax, eax, r8d                        ; (1 << partial) - 1 = mask of exactly the trailing granules
     or byte [rdi], al                         ; set only those bits, byte is inside the cleared extent: highwater <= sl_atom_free, and ceil,atom_free/64 >=, floor>>6 +1 whenever a partial byte exists
.gc_root11_done:
.gc_scan_init:
     mov r14, rsp                              ; r14 = current scan pointer, low end
     mov r15, [rel sl_gc_scan_ceiling]         ; r15 = scan ceiling, high end: private-stack top, set ONLY by the SL_STK_ENTER switch path
     cmp r14, r15                              ; empty stack range?, ONE guard for BOTH scan loops below, both bodies dereference [r14] before any bound test
     jae .gc_sweep                             ; yes: nothing to scan, equality = empty: r14 == rsp can never exceed the ceiling
     cmp dword [rel sl_gc_want_atoms], 0       ; atom collect staged?
     jnz .stack_scan_atoms                     ; yes: the walker itself marks atoms, and true cons cells, via its own cons branch
; --- cons-only scan, the exact cons-only shape : bit31-set candidates only ---
.stack_scan_cons_only:
     mov eax, dword [r14]                      ; load potential tagged value
     test eax, eax                             ; SF = bit31: clear = NIL,0 /atom/NUMBER -> not a cons reference
     jns .cons_scan_next                       ; skip, FLOAT bit31 falls into the index range rejection below
; candidate cons: extract index and validate
     mov ecx, eax                              ; ecx = tagged value
     btr ecx, SL_CONS_BIT                      ; strip tag bit -> raw index, 4B vs 6B and
     cmp ecx, [rel sl_cons_free]               ; index within allocated range?
     jae .cons_scan_next                       ; no: false positive, skip
     call GC_Mark                              ; mark reachable from stack root, GC_MARK_LOOP contract: never writes r14/r15
.cons_scan_next:
     add r14, 4                                ; advance by 4 bytes, dword stride
     cmp r14, r15                              ; reached ceiling?, bottom test doubles as the loop condition
     jb .stack_scan_cons_only                  ; no: keep scanning
     jmp .gc_sweep                             ; yes: done scanning, out-of-line exit, .stack_scan_atoms is .gc_sweep's fall-through predecessor, not this loop
; --- cons+atom scan: EITHER tag side can be live ---
.stack_scan_atoms:
     mov eax, dword [r14]                      ; load potential tagged value
     call GC_MarkAtom                          ; the atom-marking walker handles BOTH: bit31-clear values land on its atom arm, NUMBER/FLOAT/residues bounce off the arm's single bounds check; a plausibly-live atom offset gets its MARK bit ; bit31-set cons candidates walk the cells normally
     add r14, 4                                ; advance by 4 bytes, dword stride
     cmp r14, r15                              ; reached ceiling?, bottom test doubles as the loop condition; falling out lands on .gc_sweep directly
     jb .stack_scan_atoms                      ; no: keep scanning
; --- Step 4: Sweep unmarked cells into freelist ---
.gc_sweep:
; The freelist head is carried in r8d during the sweep and committed to
; sl_gc_freelist at .gc_sweep_done, no per-cell BSS round-trip.
     xor r8d, r8d                              ; r8d = freelist head under construction, 0 = empty
     xor ebx, ebx                              ; ebx = sweep index = 0
     mov ecx, [rel sl_cons_free]               ; total allocated cells, LOOP INVARIANT, hoisted, body never writes ecx
     lea rdx, [rel sl_gc_bitmap]               ; bitmap base address, LOOP INVARIANT, hoisted, body never writes rdx
     test ecx, ecx                             ; no cells allocated?
     jz .gc_sweep_done                         ; yes: nothing to sweep, entry guard, the loop below assumes a nonzero cell count
.gc_sweep_loop:
; check if cell ebx is marked in bitmap
     bt dword [rdx], ebx                       ; test mark bit of cell ebx; CF=1 if marked, memory-operand bt selects dword+bit from the full index
     jc .gc_sweep_next                         ; marked: keep this cell, straight to the advance
; unmarked: chain into the carried freelist, store old carried head as CDR
     lea r10, [r13 + rbx*CONS_SCALE + CONS_BASE_OFF] ; address of swept cell
     mov dword [r10], 0                        ; car = 0, unused, clean
     mov [r10 + CONS_CDR_OFF], r8d             ; cdr = carried freelist head, chain link
     mov r8d, ebx                              ; carried head = this cell's index
     bts r8d, SL_CONS_BIT                      ; set bit 31: same CONS tag value
.gc_sweep_next:
; marked cell: skip it / unmarked cell: just chained, either way, advance
     inc ebx                                   ; advance sweep index
     cmp ebx, ecx                              ; more cells?, bottom test doubles as the loop condition
     jb .gc_sweep_loop                         ; yes: continue sweeping
.gc_sweep_done:
     mov [rel sl_gc_freelist], r8d             ; commit the carried freelist head, both arrival paths: zero-cells and loop bottom
; --- Step 5: the ATOM half, same gate as the atom mark phase ---
     cmp dword [rel sl_gc_want_atoms], 0       ; atom collect staged?
     jz .gc_all_done                           ; no: cons-only collect ends here
     call sl_atom_gc                           ; gap walk the START index and reclaim every dead gap into the atom free list
.gc_all_done:
     SF_END rbx, rsi, rdi, r12, r13, r14, r15  ; restore non-volatiles, SF_END emits the ret
sl_atom_gc:
     SF_START sl_atom_gc, rbx, rsi, rdi, r12, r14, r15 ; r13 is NOT saved: it is the heap base, a callee-saved register the caller's stack scan relies on and this body never writes it
     SF_COMMIT                                 ; commit the frame, the cache clears below push nothing
; --- reset the atom free list, its old blocks are already reclaimed ---
     mov dword [rel sl_atom_freelist], 0       ; clear atom free list head
; --- pin floor, in granules ---
     mov r15d, SL_ATOM_INIT_END / 8            ; pin floor in granules: SL_ATOM_INIT_END is 16 (sectorlogo_const.inc), so the granule floor is the compile-time constant 2
     mov r8d, [rel sl_dispatch_highwater]      ; r8d = builtin-region floor in bytes
     shr r8d, 3                                ; r8d = in granules, both bounds are 8-aligned, so the shift is exact
     cmp r15d, r8d                             ; pin = max,SL_ATOM_INIT_END, highwater
     jae .pin_done                             ; the init floor already covers the builtin region
     mov r15d, r8d                             ; r15d = dispatch highwater in granules
.pin_done:
; Enumerate live starts from the two bitmaps. Dead atoms only contribute to gaps.
     mov ebx, r15d                             ; first unpinned granule
     mov r12d, [rel sl_atom_free]              ; atom allocation bound in bytes
     shr r12d, 3                               ; convert the bound to granules
     mov r14d, ebx                             ; open the first gap at the pin floor
     lea rdi, [rel sl_atom_start_bits]         ; START bitmap base
     lea rsi, [rel sl_atom_mark_bits]          ; MARK bitmap base
.gc_as_word:
     cmp ebx, r12d                             ; cursor at the allocation bound?
     jae .gc_as_tail                           ; settle the trailing gap
     mov eax, ebx                              ; cursor in granules
     shr eax, 6                                ; qword index in each bitmap
     mov r8, [rdi + rax*SL_GRANULE_SCALE]      ; 64 START bits inside the atom bitmap
     and r8, [rsi + rax*SL_GRANULE_SCALE]      ; discard dead starts and marks in atom interiors
     shrx r8, r8, rbx                          ; shift away the bits before the cursor
     tzcnt rax, r8                             ; distance to the first live start
     jc .gc_as_skip_word                       ; empty suffix, advance to the next word
     add eax, ebx                              ; absolute live granule
     cmp eax, r12d                             ; is the live granule below the allocation bound?
     jae .gc_as_tail                           ; settle the trailing gap
     lea r11d, [rax*SL_GRANULE_SCALE]          ; r11d = atom byte offset, granule * 8
     lea rsi, [r13 + r11 + SL_ATOM_TEXT_OFF]   ; rsi = atom content, sl_swar_strlen reads its SOURCE from rsi, type byte skipped
     call sl_swar_strlen                       ; ecx = text byte count, NUL excluded
     lea rsi, [rel sl_atom_mark_bits]          ; restore MARK base after passing the atom text in rsi
     lea edx, [ecx + SL_ATOM_HDR_BYTES]        ; edx = klen = type byte + text + NUL, the callee returned the count in ecx
     SL_ALIGN8 rdx                             ; span in bytes, >= 8: the smallest atom still owns a whole granule
     shr rdx, 3                                ; rdx = span in granules
.gc_as_live:
     cmp eax, r14d                             ; does the live atom close a gap?, its start == the gap's start means the gap is empty
     jbe .gc_as_live_advance                   ; no: nothing between the gap start and this atom
     mov r10d, eax                             ; r10d = the live atom's granule, parked across sl_atom_clr_start
     mov ebx, edx                              ; ebx = the live atom's span, granules, parked, rbx is dead here: it is reset below
     lea r8d, [r14*SL_GRANULE_SCALE]           ; r8d = block start byte offset, gap start granule * 8
     lea r9d, [r10*SL_GRANULE_SCALE]           ; r9d = gap end byte offset, the live atom's start * 8
     sub r9d, r8d                              ; r9d = block length in bytes, a multiple of 8, >= 8: the gap holds at least one granule
     mov [r13 + r8 + SL_ATOM_FB_LEN], r9d      ; block [len]
     mov r9d, [rel sl_atom_freelist]           ; r9d = current list head
     mov [r13 + r8 + SL_ATOM_FB_LINK], r9d     ; block [next] = old head
     mov [rel sl_atom_freelist], r8d           ; the merged block becomes the head
     mov eax, r14d                             ; eax = gap start granule
     mov edx, r10d                             ; edx = gap end granule, exclusive
     call sl_atom_clr_start                    ; exact START-bit clear, masked edges ; overwrites rax/rcx/rdx/r8/r9, rdi reloaded with the base it already holds
     mov rdx, rbx                              ; rdx = the live atom's span, granules, unparked
.gc_as_live_advance:
     lea rbx, [r11 + rdx*SL_ATOM_GRANULE_BYTES] ; byte sum = 8*,candidate_gran + span_gran ; r11 survives sl_atom_clr_start AND sl_swar_strlen, both documented stronger-than-ABI, rax does not
     shr rbx, 3                                ; cursor back in GRANULES, exact: both terms are 8-multiples
     mov r14d, ebx                             ; the next gap opens at the live atom's end
     jmp .gc_as_word                           ; resume from the end of this live atom
.gc_as_skip_word:
     or ebx, 63                                ; last granule in this bitmap word
     inc ebx                                   ; first granule of the next word
     jmp .gc_as_word                           ; check the bound before reading it
; --- tail: settle the trailing gap below the bound ---
.gc_as_tail:
     cmp r14d, r12d                            ; a trailing gap below the bound?
     jae .gc_as_tail_done                      ; no: the last live atom ends at the bound; the cursor stands
     mov eax, r14d                             ; eax = gap start granule
     mov edx, r12d                             ; edx = the bound, exclusive
     call sl_atom_clr_start                    ; clear the gap's START bits exactly, stale bits above a rewound cursor alias a live atom's interior
     lea eax, [r14*SL_GRANULE_SCALE]           ; eax = gap start byte offset (only eax is stored below)
     mov [rel sl_atom_free], eax               ; REWIND: the cursor drops to the gap start; no free block is needed, nothing was ever allocated above it
.gc_as_tail_done:
     lea rdi, [rel sl_intern_cache]            ; rdi = slot cache base
     xor eax, eax                              ; fill = 0, an empty slot has h1 = 0
     mov ecx, SL_INTERN_CACHE_BYTES / 4        ; Count the cache dwords to clear
     rep stosd                                 ; zero-initialize every slot
; the FFI memo table is gone: module ownership lives in the name-keyed
; owner registry, which is string-stable and never cleared by GC
     SF_END rbx, rsi, rdi, r12, r14, r15       ; restore non-volatiles, SF_END emits the ret
sl_atom_clr_start:
     lea rdi, [rel sl_atom_start_bits]         ; rdi = START bitmap base
     mov ecx, eax                              ; ecx = start granule
     and ecx, 7                                ; ecx = start's bit within its byte
     mov r8d, -1                               ; all ones; BZHI keeps the low ecx bits
     bzhi r8d, r8d, ecx                        ; start-edge keep-mask, including ecx == 0 -> 0
     shr eax, 3                                ; eax = the range's first byte index
     mov ecx, edx                              ; ecx = end granule
     and ecx, 7                                ; ecx = end's bit within its byte
     or r9d, -1                                ; all bits, imm8 sign-extends to all-ones; only the low byte is consumed at .acs_last
     shl r9d, cl                               ; r9d = end-edge keep-mask: bits at/above the range's end stay, end%8 == 0 -> 0xFF, a no-op
     shr edx, 3                                ; edx = the range's last byte index
     cmp eax, edx                              ; both edges inside ONE byte?
     jne .acs_multi                            ; no: three-part clear
     or r8d, r9d                               ; combine both edge masks, keep below start OR at/above end
     and [rdi + rax], r8b                      ; masked clear of the single byte
     ret                                       ; done
.acs_multi:
     and [rdi + rax], r8b                      ; preserve bits before start in the first byte
     inc eax                                   ; first fully covered byte
     mov ecx, edx                              ; exclusive byte bound
     sub ecx, eax                              ; number of complete middle bytes, >= 0
     add rdi, rax                              ; destination = bitmap + first middle byte
     xor eax, eax                              ; zero fill
     rep stosb                                 ; clear all complete middle bytes in one bulk operation
     and byte [rdi], r9b                       ; rdi now points at the last, partially covered byte
     sub rdi, rdx                              ; restore the START bitmap base for the caller
.acs_done:
     ret                                       ; done
%unmacro GC_MARK_LOOP 4                        ; arity MUST match the %macro declaration; defined at :26 and expanded at :67-68, only in this file. FATAL_VALUE is DELIBERATELY left bound, 40+ call sites across ten sibling SectorLogo files
%undef SL_GRANULE_SCALE                        ; atom granule -> byte offset scale, file-local
