; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define SL_ASCII_CASE_DELTA 0x20               ; A-Z <-> a-z bit difference, set bit 5
%define SL_ASCII_CASE_CLEAR 0xDF               ; clear bit 5 for a-z -> A-Z
%define SL_CASE_WIN_LAST_OFF 15                ; byte offset of the last byte in a 16-byte AVX2 fold window
%define SL_CASE_WIN32_LAST 31                  ; byte offset of the last byte in a 32-byte AVX2 fold window
%define SL_CASE_PAGE_MASK 0xFFF                ; 4KB page offset mask, casefold source page-gate
%define SL_CASE_PAGE_WINDOW32 0xFE0            ; last in-page offset where a 32-byte source load stays in-page
%define SL_LEA_X5 4                            ; SIB scale for lea [rax+rax*4] = rax*5
%define SL_LEA_X2 2                            ; SIB scale for lea [rdx+rax*2] = rdx+rax*2
%define SL_SKIP_FIRST 1                        ; byte offset to skip first matched char, SL_FIRSTCHAR_SCAN already proved it

%macro SL_FIRSTCHAR_SCAN 5
%%window:
     vmovdqu xmm1, [%1 + %2]                   ; 16 haystack bytes, unaligned ok, vmovdqu
     vpcmpeqb xmm2, xmm1, xmm4                 ; Lanes matching the first character
     vpcmpeqb xmm1, xmm1, xmm0                 ; NUL lanes, independent comparison of the same original bytes
     vpmovmskb r11d, xmm2                      ; extract first-character mask while the NUL compare is in flight
     vptest xmm1, xmm1                         ; NUL in this window?, vptest: 1 op on the frequently executed path; the mask stays in xmm1 for the infrequently executed re-extract
     jnz %%zn                                  ; yes: guard the stored atom set
     tzcnt eax, r11d                           ; count trailing zeros, CF=1 for an empty mask
     jc %%next                                 ; no match if the mask was zero
     add %2, rax                               ; cursor = stored atom offset, qword add; rax upper bits zero, bsf writes 32-bit
     jmp %4                                    ; run the EXISTING byte inner compare from %1+%2; on mismatch the caller jumps to %5 with the mask in r11d
%%next:
     add %2, 16                                ; advance the window head, 16-byte step
     jmp %%window                              ; next window
%%zn:
     vpmovmskb edx, xmm1                       ; edx = NUL mask, uncommon re-extract
     tzcnt edx, edx                            ; edx = zn = first NUL position
     bzhi r11d, r11d, edx                      ; stored atoms strictly below the NUL
     tzcnt eax, r11d                           ; count trailing zeros, CF=1 for an empty mask
     jc %3                                     ; no match if the mask was zero
     add %2, rax                               ; cursor = stored atom offset, qword add; rax upper bits zero, bsf writes 32-bit
     jmp %4                                    ; run the EXISTING byte inner compare; on mismatch the caller jumps to %5
%5:                                            ; caller alias: the failed-stored atom re-entry named in the call, mask in r11d, %2 = failed stored atom
%%more:
     bsf eax, r11d                             ; eax = the failed stored atom's lane, lowest set bit
     blsr r11d, r11d                           ; remove the disproved bit
     jz %%adv                                  ; All bytes in the current window have been processed; start the next window.
     sub %2, rax                               ; %2 = window head, failed stored atom - its lane
     bsf eax, r11d                             ; eax = bit index of the next potentially matching string byte.
     add %2, rax                               ; %2 = next stored atom
     jmp %4                                    ; inner compare again, no window reload
%%adv:
     sub %2, rax                               ; %2 = window head
     vptest xmm1, xmm1                         ; Did the processed 16-byte window contain NUL?
     jnz %3                                    ; yes: %%zn proved every stored atom at/after that NUL is the NEXT atom's storage, so continuing would scan out of the atom, not FOUND
     add %2, 16                                ; no: the window was NUL-free, the string continues
     jmp %%window                              ; reload: vectors still active, no calls on this path
%endmacro

%macro SL_NEEDLE_INNER 5
     lea r10, [%1 + %3 + SL_SKIP_FIRST]        ; r10 = &haystack[stored atom+1], first char already matched
     xor ecx, ecx                              ; rcx = byte offset into both, r10 and the needle addressing use the same k
%%inner:
     mov al, byte [%2 + rcx + SL_SKIP_FIRST]   ; needle char, mov al: only al is compared, no zero-extend needed
     test al, al                               ; end of needle?
     jz %4                                     ; yes: full match
     cmp al, byte [r10 + rcx]                  ; haystack char, memory-operand cmp, the movzx round trip is gone
     jne %5                                    ; no: next stored atom, mask still saved in r11d
     inc ecx                                   ; one advance instead of two
     jmp %%inner                               ; continue comparing
%endmacro
%define SL_SUB_SRC_OFF 32                      ; rsp offset of the source atom save
%define SL_SUB_START_OFF 36                    ; rsp offset of start index save
%define SL_SUB_FRAME 40                        ; frame bytes, 32-byte home area + 2 slots at 32/36: 2 pushes=0 mod 16, +40 -> calls at 0 mod 16
%define SL_SUB_WIN_BYTES 16                    ; AVX2 window width, bytes per scan iteration
sl_str_substr:
     push rsi                                  ; save non-volatile. RSP = 0 mod 16
     push rdi                                  ; save non-volatile. RSP = 8 mod 16
     sub rsp, SL_SUB_FRAME                     ; 2 stack slots, aligned to 16
     call EvalNextArg                          ; eax = source string atom
     ASSERT_ATOM_OR_NIL sl_str_sub_notatom     ; Part source: NIL/atom ok, CONS/NUMBER -> FATAL
     mov [rsp + SL_SUB_SRC_OFF], eax           ; save source atom to stack slot
     call EvalNextArg                          ; eax = start index, NUMBER
     call NumVal                               ; eax = raw start
     mov [rsp + SL_SUB_START_OFF], eax         ; save start to stack slot
     call EvalNextArg                          ; eax = length, NUMBER
     call NumVal                               ; eax = raw length
     mov r9d, eax                              ; r9d = len, 32-bit mov zero-extends into full r9
     test r9d, r9d                             ; len <= 0?
     jle sl_str_sub_empty                      ; yes: empty atom, guard first, before any arithmetic
     mov r10d, [rsp + SL_SUB_START_OFF]        ; r10d = start, 32-bit mov zero-extends into full r10
     xor eax, eax                              ; eax = 0, clamp value
     test r10d, r10d                           ; start < 0?
     cmovs r10d, eax                           ; yes: start = 0, branchless, today's jle no-skip semantics
; end = start + len, 64-BIT add of zero-extended operands, a raw int32 add would wrap at 2^30-scale values
; after the clamps end >= 1 always, len >= 1, start >= 0
     lea r11, [r10 + r9]                       ; r11 = end = start + len, qword bound
     mov ecx, [rsp + SL_SUB_SRC_OFF]           ; ecx = source atom, from stack slot
; get source string pointer from heap
     test ecx, ecx                             ; NIL source? upper RCX already zero after MOV ECX
     jz sl_str_sub_empty                       ; yes: return empty atom
     lea rsi, [r13 + rcx + SL_ATOM_TEXT_OFF]   ; rsi = source string base, never advances
     xor ecx, ecx                              ; rcx = scan cursor, qword
     vpxor xmm0, xmm0, xmm0                    ; zero vector for NUL detection, set once
     lea rax, [r11 - SL_SUB_WIN_BYTES]         ; hoisted window bound: the last offset at which a full 16-byte window still fits below end
sl_str_sub_window:
     cmp rcx, rax                              ; full window still below end?, signed compare handles the negative bound for free
     jg sl_str_sub_tail                        ; no: <= 15 residue bytes, byte remainder
     vpcmpeqb xmm1, xmm0, [rsi + rcx]          ; NUL bytes -> 0xFF lanes, load folded into the compare
     vptest xmm1, xmm1                         ; NUL in this window?, vptest: 1 op; the mask stays in xmm1 for the cold extract below
     jz sl_str_sub_nowin                       ; no: 16 bytes proven NUL-free
     vpmovmskb edx, xmm1                       ; edx = NUL position bitmask, cold re-extract
     tzcnt rdx, rdx                            ; rdx = offset of the FIRST NUL in the window, an overread window spanning past the real terminator collapses to it, not to a later NUL in the next atom
     add rcx, rdx                              ; cursor lands exactly on it, no byte rescan
     jmp sl_str_sub_nulfound                   ; same block the byte walk would have reached
sl_str_sub_nowin:
     add rcx, SL_SUB_WIN_BYTES                 ; 16 more bytes proven NUL-free
     jmp sl_str_sub_window                     ; next window
sl_str_sub_tail:
     vpcmpeqb xmm1, xmm0, [rsi + rcx]          ; NUL bytes -> 0xFF lanes, load folded into the compare, xmm0 = zero vector, still active from :189, nothing between there and here writes it
     vpmovmskb edx, xmm1                       ; edx = NUL lane mask, 32-bit write zero-extends rdx
     tzcnt edx, edx                            ; edx = first NUL lane, or 32 when this window has none
     add rcx, rdx                              ; rcx = absolute NUL position, or rcx + 32 when none
     cmp rcx, r11                              ; NUL before end?
     jae sl_str_sub_copyok                     ; no: strlen >= end, copy the full requested span, r9 still holds len ; also covers the NUL-at-or-past-end case
; fall through, rcx = n < end : the substring is truncated at the string's true end
sl_str_sub_nulfound:
; NUL at n = rcx < end: start >= n means the substring starts at/past the string's end
     cmp r10, rcx                              ; start >= strlen?
     jae sl_str_sub_empty                      ; yes: nothing to extract
; copy_len = n - start, no min needed: n < end = start + len => n - start < len
     mov r9, rcx                               ; r9 = n, qword; n < end < 2^32
     sub r9, r10                               ; r9 = copy_len = n - start
sl_str_sub_copyok:
; source for the copy: base + start, start is provably < min,n, end <= strlen, in-bounds by the scan itself
     add rsi, r10                              ; rsi = &source[start]
     mov byte [rel sl_concat_buf], SL_TYPE_STRING ; type byte at [0]
     lea rdi, [rel sl_concat_buf + SL_ATOM_TEXT_OFF] ; rdi = dest buffer, content starts at [1]
     lea r8, [rel sl_concat_buf + SL_CONCAT_BUF_LAST] ; r8 = buffer end, last writable byte offset
; one bounds check preserving the current boundary semantics: rdi + copy_len is the NUL's slot, payload may
; reach r8-1, NUL may land AT r8, never beyond
     lea rax, [rdi + r9]                       ; rax = NUL slot after the copy
     cmp rax, r8                               ; payload+NUL fits the 2 MiB staging buffer?
     ja sl_concat_overflow                     ; no: FATAL, never truncate, silent data corruption
     mov ecx, r9d                              ; ecx = copy_len, rep movsb count
     rep movsb                                 ; ERMS burst copy, rdi/rsi land past the copy
     mov byte [rdi], 0                         ; NUL-terminate
     lea rsi, [rel sl_concat_buf]              ; source for Intern
     call Intern                               ; eax = new atom
     add rsp, SL_SUB_FRAME                     ; restore frame
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     ret                                       ; return to caller
sl_str_sub_empty:
     mov byte [rel sl_concat_buf], SL_TYPE_STRING ; type byte
     mov byte [rel sl_concat_buf + SL_ATOM_TEXT_OFF], 0 ; empty string content
     lea rsi, [rel sl_concat_buf]              ; source for Intern
     call Intern                               ; eax = empty atom
     add rsp, SL_SUB_FRAME                     ; restore frame
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     ret                                       ; return the interned empty atom
sl_str_sub_notatom:
     FATAL_VALUE "Part requires a string atom argument", eax ; name the offender; eax is the source arg, ASSERT_ATOM_OR_NIL overwrites flags only
%define SL_FIND_NEEDLE_OFF 32                  ; rsp offset of the needle atom save
%define SL_FIND_FRAME 40                       ; frame bytes, 32-byte home area + 1 slot at 32: 2 pushes=0 mod 16, +40 -> calls at 0 mod 16
sl_str_find:
     push rsi                                  ; save non-volatile. RSP = 0 mod 16
     push rdi                                  ; save non-volatile. RSP = 8 mod 16
     sub rsp, SL_FIND_FRAME                    ; 1 slot + padding, aligned to 16
     call EvalNextArg                          ; eax = needle atom
     ASSERT_ATOM_OR_NIL sl_str_find_notatom    ; Search needle: NIL/atom ok, CONS/NUMBER -> FATAL
     mov [rsp + SL_FIND_NEEDLE_OFF], eax       ; save needle to stack slot
     call EvalNextArg                          ; eax = haystack atom
     ASSERT_ATOM_OR_NIL sl_str_find_notatom    ; Search haystack: NIL/atom ok, CONS/NUMBER -> FATAL
     mov ecx, [rsp + SL_FIND_NEEDLE_OFF]       ; ecx = needle atom, from stack slot
; get string pointers from heap
     test ecx, ecx                             ; NIL needle? upper RCX already zero after MOV ECX
     jz sl_str_find_empty_needle               ; yes: return NUMBER,0
     lea rsi, [r13 + rcx + SL_ATOM_TEXT_OFF]   ; rsi = needle string in heap, no '"' prefix
     cmp byte [rsi], 0                         ; empty needle?, matches at position 0 of ANY haystack, "" and NIL alike
     je sl_str_find_empty_needle               ; yes: found at start
     test eax, eax                             ; NIL haystack?
     jz sl_str_find_not_found                  ; yes: a non-empty needle cannot match an empty haystack
     lea rdi, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rdi = haystack string in heap, no '"' prefix
; --- scan setup: zero + first-needle-char broadcast, 2 ops, once ---
     vpxor xmm0, xmm0, xmm0                    ; zero vector for NUL detection
     vpbroadcastb xmm4, byte [rsi]             ; xmm4 = needle[0] in all 16 lanes
     xor r8d, r8d                              ; r8 = scan cursor, qword, 0-based
; --- outer scan: 16-byte AVX2 first-char windows ---
     SL_FIRSTCHAR_SCAN rdi, r8, sl_str_find_not_found, sl_str_find_candidate, sl_str_find_more ; scan haystack, exit to inner compare on stored atom, re-enter on mismatch via the saved mask
sl_str_find_candidate:
; r8 = stored atom offset, rdi = haystack base, rsi = needle start
     SL_NEEDLE_INNER rdi, rsi, r8, sl_str_find_found, sl_str_find_more ; needle vs haystack
sl_str_find_found:
     mov eax, r8d                              ; eax = position
     add rsp, SL_FIND_FRAME                    ; restore frame
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     jmp MakeNum                               ; tail call: return NUMBER,position
sl_str_find_not_found:
     xor eax, eax                              ; return NIL
     add rsp, SL_FIND_FRAME                    ; restore frame
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     ret                                       ; return to caller
sl_str_find_notatom:
     FATAL_VALUE "Search requires string atom arguments", eax ; name the offender; eax is the needle, from :168 or the haystack, from :171
sl_str_find_empty_needle:
     xor eax, eax                              ; position = 0
     add rsp, SL_FIND_FRAME                    ; restore frame
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     jmp MakeNum                               ; tail call: return NUMBER,0

; SL_CASEFOLD_VEC src, scratch, "lower"|"upper" -- toggle bit 5 on the matching
; ASCII letter range, using one biased signed compare.
%macro SL_CASEFOLD_VEC 3
%ifidn %3, "lower"
     vpaddb %2, %1, [rel sl_case_upper_bias]   ; bias A-Z onto the signed letter run
%else
     vpaddb %2, %1, [rel sl_case_lower_bias]   ; bias a-z onto the signed letter run
%endif
     vpcmpgtb %2, %2, [rel sl_case_threshold]  ; 0xFF on matched letter lanes
     vpand %2, %2, [rel sl_case_bit5]          ; keep only the case bit
     vpxor %1, %1, %2                          ; set for lowercase, clear for uppercase
%endmacro

%macro SL_CASEFOLD 2
%1:                                            ; function entry, sl_str_lower or sl_str_upper
     push rsi                                  ; save non-volatile
     push rdi                                  ; save non-volatile
     sub rsp, 40                               ; home area + padding, 2 pushes + 40 -> calls at 0 mod 16
     call EvalNextArg                          ; eax = string atom
     ASSERT_ATOM_OR_NIL %%notatom              ; casefold arg: NIL/atom ok, CONS/NUMBER -> FATAL
     test eax, eax                             ; NIL source?
     jz %%empty                                ; yes: return empty atom
     lea rsi, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rsi = string in heap, no '"' prefix
     mov byte [rel sl_concat_buf], SL_TYPE_STRING ; type byte at [0]
     lea rdi, [rel sl_concat_buf + SL_ATOM_TEXT_OFF] ; rdi = dest buffer, content starts at [1]
     lea r8, [rel sl_concat_buf + SL_CONCAT_BUF_LAST] ; r8 = buffer end, last writable byte address
     lea r9, [r8 - SL_CASE_WIN_LAST_OFF]       ; r9 = largest dest cursor whose 16-byte store still fits, loop-invariant threshold: r8 never changes, so the per-window `lea rcx,[rdi+15]` re-derives a constant
     vpxor xmm1, xmm1, xmm1                    ; NUL-compare vector, constant across windows
%%window:
     lea rax, [r8 - SL_CASE_WIN32_LAST]        ; largest dest cursor whose 32-byte store still fits
     cmp rdi, rax                              ; too near the buffer end for a 32-byte store?
     ja %%window16                             ; yes: 16-byte window
     mov eax, esi                              ; source page-gate for the 32-byte read
     and eax, SL_CASE_PAGE_MASK                ; in-page offset
     cmp eax, SL_CASE_PAGE_WINDOW32            ; 32-byte source load stays in-page?
     ja %%window16                             ; no: 16-byte window
     vmovdqu ymm0, [rsi]                       ; 32 source bytes
     vpcmpeqb ymm2, ymm0, ymm1                 ; NUL lanes, full ymm compare: ymm1 was zeroed as VEX.128, upper lanes are zero too
     vptest ymm2, ymm2                         ; any NUL lane?
     jnz %%copy                                ; NUL inside the window: scalar tail
     SL_CASEFOLD_VEC ymm0, ymm3, %2            ; fold the 32-byte window
     vmovdqu [rdi], ymm0                       ; store 32 folded bytes
     add rsi, 32                               ; advance source
     add rdi, 32                               ; advance dest
     jmp %%window                              ; next window
%%window16:
     vmovdqu xmm0, [rsi]                       ; 16 source bytes, heap atom: overread past NUL is committed memory
     vpcmpeqb xmm2, xmm0, xmm1                 ; NUL lanes -> 0xFF
     vptest xmm2, xmm2                         ; NUL inside this window?, vptest: 1 op; the mask is only extracted where a remainder is actually taken
     jnz %%copy                                ; yes: scalar tail, per-byte fold + exact bounds semantics
%%vector:
     cmp rdi, r9                               ; would the window's last store byte land past the buffer end?, rdi > r9 <=> rdi + 15 > r8
     ja %%copy                                 ; yes: scalar tail, keeps the per-byte FATAL contract
; ASCII interval endpoints are < 0x80. Signed PCMPGTB on the original bytes
; automatically rejects all high-bit bytes because they are negative.
     SL_CASEFOLD_VEC xmm0, xmm3, %2            ; fold the 16-byte window
     vmovdqu [rdi], xmm0                       ; store the full window
     add rsi, 16                               ; advance source
     add rdi, 16                               ; advance dest
     jmp %%window                              ; next window
%%copy:
; scalar tail: NUL-window remainder + near-buffer-end windows, identical per-byte semantics
     lodsb                                     ; load al from [rsi], rsi++
     test al, al                               ; NUL?, al ONLY, lodsb leaves eax bits 8-31 stale from the atom offset
     jz %%done                                 ; yes: done
%ifidn %2, "lower"
     cmp al, 'A'                               ; below 'A'?
     jb %%store                                ; yes: no conversion needed
     cmp al, 'Z'                               ; above 'Z'?
     ja %%store                                ; yes: no conversion needed
     or al, SL_ASCII_CASE_DELTA                ; convert A-Z to a-z, set bit 5
%else
     cmp al, 'a'                               ; below 'a'?
     jb %%store                                ; yes: no conversion needed
     cmp al, 'z'                               ; above 'z'?
     ja %%store                                ; yes: no conversion needed
     and al, SL_ASCII_CASE_CLEAR               ; convert a-z to A-Z, clear bit 5
%endif
%%store:
     cmp rdi, r8                               ; buffer full?
     jae sl_concat_overflow                    ; yes: FATAL, a truncated result atom is silent data corruption
     stosb                                     ; store al to [rdi], rdi++
     jmp %%copy                                ; continue copying
%%done:
     mov byte [rdi], 0                         ; NUL-terminate
     lea rsi, [rel sl_concat_buf]              ; source for Intern
     call Intern                               ; eax = casefolded atom
     add rsp, 40                               ; drop home area + parity pad
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     ret                                       ; return to caller
%%empty:
     mov byte [rel sl_concat_buf], SL_TYPE_STRING ; type byte
     mov byte [rel sl_concat_buf + SL_ATOM_TEXT_OFF], 0 ; empty string content
     lea rsi, [rel sl_concat_buf]              ; source for Intern
     call Intern                               ; eax = empty atom
     add rsp, 40                               ; drop home area + parity pad
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     ret                                       ; return the interned empty atom
%%notatom:
%ifidn %2, "lower"
     FATAL_VALUE "Lowercase requires a string atom argument", eax ; name the offender; eax is the casefold arg
%else
     FATAL_VALUE "Uppercase requires a string atom argument", eax ; name the offender; eax is the casefold arg
%endif
%endmacro
     SL_CASEFOLD sl_str_lower, "lower"         ; lowercase: A-Z -> a-z
     SL_CASEFOLD sl_str_upper, "upper"         ; uppercase: a-z -> A-Z
%define SL_STRLEN_FRAME 32                     ; frame bytes, home area only
sl_str_strlen:
     push rsi                                  ; save non-volatile, borrowed as scan cursor.RSP = 0 mod 16, the only push; rdi is never written in this function
     sub rsp, SL_STRLEN_FRAME                  ; frame. RSP = 0 mod 16, 1 push + 32
     call EvalNextArg                          ; eax = the string arg
     test eax, eax                             ; NIL?
     jz sl_strlen_zero                         ; yes: length 0
     test eax, SL_TAG_MASK                     ; any tag bits set, CONS/FLOAT/NUMBER ?
     jnz sl_strlen_notatom                     ; yes: FATAL
     lea rsi, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rsi = string start in heap, no '"' prefix
     call sl_swar_strlen                       ; ecx = strlen, rsi/rdi preserved by the leaf
     mov eax, ecx                              ; eax = byte length, zero-extends
     jmp sl_strlen_done                        ; finished
sl_strlen_zero:
     xor eax, eax                              ; length = 0
sl_strlen_done:
     add rsp, SL_STRLEN_FRAME                  ; drop frame
     pop rsi                                   ; restore non-volatile
     jmp MakeNum                               ; tail call: NUMBER,length
sl_strlen_notatom:
     FATAL_VALUE "TextLength requires a string atom argument", eax ; name the offender; eax is the CONS/NUMBER/FLOAT arg
%define SL_REPL_SRC_OFF 32                     ; sl_str_replace: rsp offset of the source atom save
%define SL_REPL_FIRST 1                        ; one preceding byte before SL_NEEDLE_INNER's counted tail = old_len
%define SL_REPL_OLD_OFF 40                     ; rsp offset of old atom save
%define SL_REPL_NEW_OFF 48                     ; rsp offset of new atom save
%define SL_REPL_FRAME 56                       ; frame bytes, 32-byte home area + 3 slots at 32..48: 2 pushes=0 mod 16, +56 -> calls at 0 mod 16
sl_str_replace:
     push rsi                                  ; save non-volatile. RSP = 0 mod 16
     push rdi                                  ; save non-volatile. RSP = 8 mod 16
     sub rsp, SL_REPL_FRAME                    ; 3 slots + padding, aligned to 16
; --- Evaluate 3 args, saving each to stack slots, EvalNextArg overwrites volatiles ---
     call EvalNextArg                          ; eax = source string atom
     ASSERT_ATOM_OR_NIL sl_str_rep_notatom     ; Replace source: NIL/atom ok, CONS/NUMBER -> FATAL
     mov [rsp + SL_REPL_SRC_OFF], eax          ; save source atom to stack slot
     call EvalNextArg                          ; eax = old string atom
     ASSERT_ATOM_OR_NIL sl_str_rep_notatom     ; Replace old: NIL/atom ok, CONS/NUMBER -> FATAL
     mov [rsp + SL_REPL_OLD_OFF], eax          ; save old atom to stack slot
     call EvalNextArg                          ; eax = new string atom
     ASSERT_ATOM_OR_NIL sl_str_rep_notatom     ; Replace new: NIL/atom ok, CONS/NUMBER -> FATAL
     mov [rsp + SL_REPL_NEW_OFF], eax          ; save new atom to stack slot
; --- All args evaluated. Derive string pointers from heap ---
     mov ecx, [rsp + SL_REPL_SRC_OFF]          ; ecx = source atom
     test ecx, ecx                             ; NIL source?
     jz sl_str_rep_ret_nil                     ; yes: return NIL
     lea rsi, [r13 + rcx + SL_ATOM_TEXT_OFF]   ; rsi = source string in heap, no '"' prefix
     mov [rel sl_str_ptr1], rsi                ; save source start pointer
; Derive old string pointer
     mov ecx, [rsp + SL_REPL_OLD_OFF]          ; ecx = old atom
     test ecx, ecx                             ; NIL old?
     jz sl_str_rep_old_nil                     ; yes: use NUL byte
     lea rdi, [r13 + rcx + SL_ATOM_TEXT_OFF]   ; rdi = old string in heap, no '"' prefix
     cmp byte [rdi], 0                         ; empty old string?
     je sl_str_rep_old_nil                     ; yes: use NUL byte
     mov [rel sl_str_ptr2], rdi                ; save old string pointer
     jmp sl_str_rep_derive_new                 ; continue to derive new
sl_str_rep_old_nil:
; NIL or empty old: point to NUL byte, search will match at pos 0
     lea rax, [rel sl_str_nul]                 ; rax = 1-byte NUL constant
     mov [rel sl_str_ptr2], rax                ; save old string ptr, empty
sl_str_rep_derive_new:
; Derive new string pointer
     mov ecx, [rsp + SL_REPL_NEW_OFF]          ; ecx = new atom
     test ecx, ecx                             ; NIL new?
     jz sl_str_rep_new_empty                   ; yes: empty new string
     lea rax, [r13 + rcx + SL_ATOM_TEXT_OFF]   ; rax = new string in heap, no '"' prefix
     mov [rel sl_str_ptr3], rax                ; save new string pointer
     jmp sl_str_rep_search                     ; start searching for old
sl_str_rep_new_empty:
; NIL new -> point to NUL byte, copy loop will copy 0 bytes
     lea rax, [rel sl_str_nul]                 ; rax = 1-byte NUL constant
     mov [rel sl_str_ptr3], rax                ; save new string ptr, empty
; --- Search for old string in source ---
; rsi = source base, never advances in the scan region, ptr2 = old string
sl_str_rep_search:
     mov rdi, [rel sl_str_ptr2]                ; rdi = old string pointer, needle
     xor r9d, r9d                              ; r9 = 0, qword; the empty-old bypass below jumps to sl_str_rep_found with r9=0
     cmp byte [rdi], 0                         ; empty old?, memory-operand form, same as sl_str_find's needle test
     jz sl_str_rep_empty_needle                ; empty old: length is zero, match at offset 0
; --- scan setup: zero + first-old-char broadcast, 2 ops ---
     vpxor xmm0, xmm0, xmm0                    ; zero vector for NUL detection
     vpbroadcastb xmm4, byte [rdi]             ; xmm4 = old[0] in all 16 lanes
; --- outer scan: 16-byte AVX2 first-char windows over the source ---
     SL_FIRSTCHAR_SCAN rsi, r9, sl_str_rep_not_found, sl_str_rep_candidate, sl_str_rep_more ; scan source, exit to inner compare on candidate, re-enter on mismatch via the parked mask
sl_str_rep_candidate:
; r9 = candidate offset, rsi = source base, rdi = old string base
     SL_NEEDLE_INNER rsi, rdi, r9, sl_str_rep_matched, sl_str_rep_more ; old vs source, success reuses the count
sl_str_rep_matched:
     lea r11d, [ecx + SL_REPL_FIRST]           ; first byte + ecx successful tail bytes = old_len
     jmp sl_str_rep_found                      ; build the result from the measured length
sl_str_rep_empty_needle:
     xor r11d, r11d                            ; empty needle has zero length
; --- Build result: prefix + new + suffix into sl_concat_buf ---
sl_str_rep_found:
     mov byte [rel sl_concat_buf], SL_TYPE_STRING ; type byte at [0]
     lea rdi, [rel sl_concat_buf + SL_ATOM_TEXT_OFF] ; rdi = dest cursor, content starts at [1]
     lea r8, [rel sl_concat_buf + SL_CONCAT_BUF_LAST] ; r8 = buffer end, live for the whole build
     mov rsi, [rel sl_str_ptr3]                ; rsi = new string
     call sl_swar_strlen                       ; ecx = new_len
     mov r10d, ecx                             ; r10 = new_len, parked
     lea rsi, [r9 + r11]                       ; match position + old needle length
     add rsi, [rel sl_str_ptr1]                ; + source base, one memory-operand add
     call sl_swar_strlen                       ; ecx = suffix_len
     mov rdx, rcx                              ; suffix length, bounds/copy below unchanged
     mov rsi, [rel sl_str_ptr1]                ; restore source base for the prefix copy
; ONE combined bounds check: dest + prefix + new + suffix = NUL slot;
; payload may reach r8-1, NUL may land AT r8
     lea rax, [rdi + r9]                       ; rax = dest + prefix_len
     add rax, r10                              ; + new_len
     add rax, rdx                              ; + suffix_len = NUL slot
     cmp rax, r8                               ; payload + NUL fit?
     ja sl_concat_overflow                     ; no: FATAL, a truncated result atom is silent data corruption
     mov rcx, r9                               ; rcx = prefix_len, rsi still holds the source base from above: no call since, and sl_swar_strlen preserves rsi
     rep movsb                                 ; copy prefix, rdi advances; DF=0
     mov rsi, [rel sl_str_ptr3]                ; rsi = new string
     mov rcx, r10                              ; rcx = new_len
     rep movsb                                 ; copy replacement
     lea rsi, [r9 + r11]                       ; match position + old needle length
     add rsi, [rel sl_str_ptr1]                ; + source base, suffix start
     mov rcx, rdx                              ; rcx = suffix_len
     rep movsb                                 ; copy suffix
     mov byte [rdi], 0                         ; NUL-terminate
     lea rsi, [rel sl_concat_buf]              ; source for Intern
     call Intern                               ; eax = new atom
     jmp sl_str_rep_epilog                     ; shared epilogue
sl_str_rep_not_found:
; Not found: return source atom unchanged
     mov eax, [rsp + SL_REPL_SRC_OFF]          ; return source atom
     jmp sl_str_rep_epilog                     ; shared epilogue
sl_str_rep_ret_nil:
; NIL source: return NIL
     xor eax, eax                              ; return NIL
sl_str_rep_epilog:
     add rsp, SL_REPL_FRAME                    ; restore frame
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     ret                                       ; return to caller
sl_str_rep_notatom:
     FATAL_VALUE "Replace requires string atom arguments", eax ; name the offender; eax is the source, :333, old, :336 or new, :339 atom
sl_concat_overflow:
     FATAL "SectorLogo: string result exceeds 2 MiB concat buffer" ; capacity: a truncated result atom is silent data corruption, if a real sidecar needs more, grow the buffer, never clamp
%define SL_STN_OVF_OFF 32                      ; rsp offset of the overflow flag
%define SL_STN_FRAME 40                        ; frame bytes, home area + overflow slot + padding
sl_str_to_num:
     push rsi                                  ; save non-volatile. RSP = 0 mod 16
     push rdi                                  ; save non-volatile. RSP = 8 mod 16
     sub rsp, SL_STN_FRAME                     ; frame; RSP = 0 mod 16 at the call below
     call EvalNextArg                          ; eax = string atom
     test eax, eax                             ; NIL?
     jz sl_stn_nil                             ; NIL: return NIL
     js sl_stn_check_float                     ; bit31 set: CONS or FLOAT
     test eax, SL_NUMBER_TAG                   ; NUMBER?
     jnz sl_stn_num_ret                        ; yes: return as-is
; fall through: atom, string — parse it
sl_stn_check_float:
     mov ecx, eax                              ; ecx = value copy for tag isolation
     shr ecx, 30                               ; ecx = numeric tag = original bits 31:30, range 0..3
     cmp ecx, 3                                ; FLOAT: bits 31:30 both set
     je sl_stn_passthrough                     ; yes: return as-is
     cmp ecx, 2                                ; CONS: bit 31 only
     je sl_stn_notatom                         ; yes: FATAL
; atom: parse the string
     lea rsi, [r13 + rax + SL_ATOM_TEXT_OFF]   ; rsi = string in heap
     cmp byte [rsi], 0                         ; empty string?
     je sl_stn_nil                             ; yes: not a number
     xor eax, eax                              ; accumulator = 0
     xor r8d, r8d                              ; r8d = negative flag
     xor ecx, ecx                              ; ecx = digit counter
     mov dword [rsp + SL_STN_OVF_OFF], eax     ; init the overflow flag
     mov r11, 922337203685477579               ; 64-bit accumulate wall, imm32 would truncate, fact #35
     cmp byte [rsi], '-'                       ; leading minus?
     jne sl_stn_loop                           ; no: start digits
; check char after '-' is a digit
     SL_DIGIT_IN [rsi + SL_ATOI_PEEK_OFF], sl_stn_nil ; "-" alone or non-digit after '-' takes the macro's ja
     inc rsi                                   ; skip past '-'
     mov r8d, 1                                ; set negative flag
sl_stn_loop:
     SL_DIGIT_IN [rsi], sl_stn_done            ; not a digit: done, edx = byte minus '0'
     cmp rax, r11                              ; rax*10+9 would pass 2^63-1?
     ja sl_stn_degrade                         ; yes: one rounding of the exact part, then per-digit floats
     lea rax, [rax + rax*SL_LEA_X5]            ; rax *= 5, SIB scale 4 + add
     lea rax, [rdx + rax*SL_LEA_X2]            ; rax = 10*old + digit, SIB scale 2 + add
     inc rsi                                   ; advance string pointer
     inc ecx                                   ; count digit
     jmp sl_stn_loop                           ; continue parsing
sl_stn_degrade:
     vcvtsi2sd xmm0, xmm0, rax                 ; xmm0 =, double exact int part, single correct rounding
     mov dword [rsp + SL_STN_OVF_OFF], 1       ; overflow flag, read at the fraction entry and the int classifier
sl_stn_floop:
     SL_DIGIT_IN [rsi], sl_stn_done            ; not a digit: done
     vcvtsi2sd xmm1, xmm1, rdx                 ; xmm1 = double digit
     vfmadd132sd xmm0, xmm1, qword [rel dbl_ten] ; xmm0 = xmm0*dbl_ten + xmm1, one rounding
     inc rsi                                   ; advance string pointer
     inc ecx                                   ; count digit
     jmp sl_stn_floop                          ; continue the float chain
sl_stn_done:
     test ecx, ecx                             ; any digits at all?
     jz sl_stn_nil                             ; no digits: not a number
     cmp byte [rsi], 0                         ; string ended?
     je sl_stn_int_classify                    ; yes: classify the exact integer
     cmp byte [rsi], '.'                       ; decimal point?
     jne sl_stn_nil                            ; no: trailing garbage -> NIL
; check first char after '.'
     SL_DIGIT_IN [rsi + SL_ATOI_PEEK_OFF], sl_stn_nil ; "3." with no fraction digits / not a digit after '.' takes the macro's ja
; --- float parse, adapted from EvalPrimary .atof_parse ---
     inc rsi                                   ; skip past '.'
     cmp dword [rsp + SL_STN_OVF_OFF], 0       ; int part already degraded to xmm0?
     jne sl_stn_frac_go                        ; yes: keep it, skip the conversion
     vcvtsi2sd xmm0, xmm0, rax                 ; no: convert the exact 64-bit int part, single rounding; rax may exceed int32
sl_stn_frac_go:
     xor r9d, r9d                              ; r9d = fraction digit accumulator
     xor r10d, r10d                            ; r10d = fraction digit count
sl_stn_frac_loop:
     SL_DIGIT_IN [rsi], sl_stn_frac_done       ; not a digit: end fraction
     cmp r10d, 9                               ; already at 9 fraction digits, int32-safe cap, lut has 10 entries, acc overflows at 10 ?
     jae sl_stn_frac_stop                      ; yes: stop consuming, remaining digits become trailing text
     imul r9d, r9d, 10                         ; frac acc *= 10
     add r9d, edx                              ; ... + digit
     inc r10d                                  ; count digit
     inc rsi                                   ; advance string pointer
     jmp sl_stn_frac_loop                      ; continue fraction parse
sl_stn_frac_stop:
; 10th+ digit leaves rsi mid-string: the [rsi]!=0 trailing-garbage check
; below rejects to NIL, documented out-of-range semantics, same as sl_atom_float
sl_stn_frac_done:
     cmp byte [rsi], 0                         ; clean end of text?
     jne sl_stn_nil                            ; trailing garbage after fraction
; compute 10^r10d, r10d >= 1 always here: the fraction peek before the loop
; guarantees the first char after '.' is a digit, so the loop consumes >= 1
     lea rax, [rel pow10_table]                ; rax = &power-of-10 table, shared with atof
     mov eax, [rax + r10*SL_POW10_DWORD]       ; eax = 10^frac_digits, r10d <= 9, single load vs loop
     vcvtsi2sd xmm1, xmm1, r9d                 ; xmm1 =, double frac digits
     vcvtsi2sd xmm2, xmm2, eax                 ; xmm2 =, double 10^count
     vdivsd xmm1, xmm1, xmm2                   ; xmm1 = fraction value
     vaddsd xmm0, xmm0, xmm1                   ; xmm0 = int + frac, binary64 accumulate
sl_stn_float_sign:
     vcvtsd2ss xmm0, xmm0, xmm0                ; FP64 path: retains the fraction and overflow-degrade behaviour
sl_stn_float_sign_f32:
     test r8d, r8d                             ; negative?
     jz sl_stn_float_ret                       ; no: keep positive
     vxorps xmm0, xmm0, [rel flt_sign_mask]    ; flip sign bit
sl_stn_float_ret:
     add rsp, SL_STN_FRAME                     ; drop home area + flag slot
     pop rdi                                   ; restore rdi
     pop rsi                                   ; restore rsi
     jmp MakeFloat                             ; tail call: wrap as FLOAT
sl_stn_int_classify:
; --- clean integer end: NUMBER when it fits the 30-bit payload, FLOAT otherwise ---
     cmp dword [rsp + SL_STN_OVF_OFF], 0       ; degraded to the float chain?
     jne sl_stn_float_sign                     ; yes: xmm0 holds the, positive value, shared sign+wrap tail
     test r8d, r8d                             ; negative?
     jz sl_stn_pos_chk                         ; no: positive payload check
     cmp rax, STN_NUM_NEG_BOUND                ; |v| past 2^29, MIN = -536870912 exactly ?
     ja sl_stn_promote                         ; yes: too negative for a NUMBER -> FLOAT
     neg rax                                   ; fits: negate, magnitude <= 2^29 -> no int32 issues
     jmp sl_stn_int_ret                        ; MakeNum
sl_stn_pos_chk:
     cmp rax, STN_NUM_PAYLOAD_MAX              ; above 2^29 - 1?
     jbe sl_stn_int_ret                        ; fits: MakeNum; above the payload max falls into the promote
sl_stn_promote:
     vcvtsi2ss xmm0, xmm0, rax                 ; exact int64 -> binary32, single rounding
     jmp sl_stn_float_sign_f32                 ; sign+wrap tail, skips the FP64 -> FP32 narrowing
sl_stn_int_ret:
     add rsp, SL_STN_FRAME                     ; drop home area + flag slot
     pop rdi                                   ; restore rdi
     pop rsi                                   ; restore rsi
     jmp MakeNum                               ; tail call: wrap as NUMBER
sl_stn_notatom:
     FATAL_VALUE "TextToNumber requires a string atom argument", eax ; name the offender; eax is the CONS, ecx holds the masked copy
sl_stn_num_ret:
     jmp sl_stn_common_ret                     ; NUMBER passed in: return it unchanged
sl_stn_passthrough:
     jmp sl_stn_common_ret                     ; FLOAT or other tagged value: return unchanged
sl_stn_nil:
     xor eax, eax                              ; return NIL
sl_stn_common_ret:
     add rsp, SL_STN_FRAME                     ; drop home area + flag slot
     pop rdi                                   ; restore rdi
     pop rsi                                   ; restore rsi
     ret                                       ; return eax as-is
%define SL_EXPL_RESULT_OFF 32                  ; rsp offset of the result list head, GC root
%define SL_EXPL_CHAR_OFF 40                    ; rsp offset of the per-piece staging char cursor
%define SL_EXPL_TAILPTR_OFF 48                 ; rsp offset of the tail-link pointer; unused qword between CHAR and R9
%define SL_EXPL_APPEND_BIAS 48                 ; helper frame bias: 40-byte shadow/alignment block + 8-byte return address
%define SL_EXPL_R9_OFF 56                      ; rsp offset of r9 save, scan pos, qword
%define SL_EXPL_SRC_OFF 64                     ; rsp offset of source atom save
%define SL_EXPL_DELIM_OFF 72                   ; rsp offset of delimiter atom save
%define SL_EXPL_SRCBASE_OFF 80                 ; rsp offset of source base POINTER save, qword, NIL-safe
%define SL_EXPL_DELIMBASE_OFF 88               ; rsp offset of delimiter base POINTER save, qword
%define SL_EXPL_DELIMLEN_OFF 96                ; rsp offset of delimiter length save, qword, hoisted
%define SL_EXPL_SRCLEN_OFF 104                 ; rsp offset of source length save, qword, hoisted
sl_str_explode:
     push rsi                                  ; save non-volatile
     push rdi                                  ; save non-volatile
     sub rsp, 120                              ; allocate the frame, sizing rationale in the block comment above
     call EvalNextArg                          ; eax = source atom
     ASSERT_ATOM_OR_NIL sl_expl_notatom        ; EXPLODE source: NIL/atom ok, CONS/NUMBER -> FATAL
     mov [rsp + SL_EXPL_SRC_OFF], rax          ; save source atom, qword: a dword store leaves the upper dword as stale stack garbage inside the GC scan range
     call EvalNextArg                          ; eax = delimiter atom
     ASSERT_ATOM_OR_NIL sl_expl_notatom        ; EXPLODE delimiter: NIL/atom ok, CONS/NUMBER -> FATAL
     mov [rsp + SL_EXPL_DELIM_OFF], rax        ; save delimiter atom, qword, same GC reason
; derive string pointers
     mov ecx, [rsp + SL_EXPL_SRC_OFF]          ; reload source atom
     test ecx, ecx                             ; NIL source?
     jz sl_expl_src_nil                        ; yes: use empty string
     lea rsi, [r13 + rcx + SL_ATOM_TEXT_OFF]   ; rsi = source string
     jmp sl_expl_src_ok                        ; source pointer ready
sl_expl_src_nil:
     lea rsi, [rel sl_str_nul]                 ; empty source
sl_expl_src_ok:
     mov ecx, [rsp + SL_EXPL_DELIM_OFF]        ; reload delimiter atom
     test ecx, ecx                             ; NIL delimiter?
     jz sl_expl_delim_nil                      ; yes: use empty string
     lea rdi, [r13 + rcx + SL_ATOM_TEXT_OFF]   ; rdi = delimiter string
     jmp sl_expl_delim_ok                      ; delimiter pointer ready
sl_expl_delim_nil:
     lea rdi, [rel sl_str_nul]                 ; empty delimiter
sl_expl_delim_ok:
; Initialize source-order output for BOTH normal-delimiter and empty-delimiter paths.
; The head is a qword GC root and a tail-link initially points at that head.
     mov qword [rsp + SL_EXPL_RESULT_OFF], 0   ; result head = NIL, qword GC root
     lea rax, [rsp + SL_EXPL_RESULT_OFF]       ; rax = address of the head root
     mov [rsp + SL_EXPL_TAILPTR_OFF], rax      ; tail link starts pointing at the head root
; check for empty delimiter, split into chars
     cmp byte [rdi], 0                         ; empty delimiter string?
     je sl_expl_char_split                     ; yes: split into chars
     mov [rsp + SL_EXPL_SRCBASE_OFF], rsi      ; save source base pointer
     mov [rsp + SL_EXPL_DELIMBASE_OFF], rdi    ; save delimiter base pointer
     call sl_swar_strlen                       ; ecx = src_strlen, rsi = source
     mov [rsp + SL_EXPL_SRCLEN_OFF], rcx       ; save, qword
     mov rsi, rdi                              ; rsi = delimiter string, for the leaf
     call sl_swar_strlen                       ; ecx = delim_len
     mov [rsp + SL_EXPL_DELIMLEN_OFF], rcx     ; save, qword
     mov rsi, [rsp + SL_EXPL_SRCBASE_OFF]      ; rsi = source base, restored
     mov rdi, [rsp + SL_EXPL_DELIMBASE_OFF]    ; rdi = delimiter base, restored
     xor r8d, r8d                              ; r8 = segment start offset, qword
     xor r9d, r9d                              ; r9 = scan cursor, qword
; result head and tail link were initialized at sl_expl_delim_ok
; scan vectors: staged here and RESTAGED after every Intern, it overwrites xmm0-4
     vpxor xmm0, xmm0, xmm0                    ; zero vector for NUL detection
     vpbroadcastb xmm4, byte [rdi]             ; xmm4 = delimiter[0] in all 16 lanes
     mov byte [rel sl_concat_buf], SL_TYPE_STRING ; type byte at [0]
sl_expl_scan:
     SL_FIRSTCHAR_SCAN rsi, r9, sl_expl_emit_rest, sl_expl_candidate, sl_expl_more ; scan source, emit rest at NUL, inner compare on candidate, re-enter on mismatch via the parked mask
sl_expl_candidate:
; r9 = candidate offset, rsi = source base, rdi = delimiter base
     SL_NEEDLE_INNER rsi, rdi, r9, sl_expl_match_found, sl_expl_more ; delimiter vs source, delim base is reloaded from its frame slot after every Intern/Cons
; SL_EXPL_EMIT_SEGMENT -- copy one known-length segment into the concat buffer
; and stage it for Intern. IN: rcx = segment length, rsi = source base,
; r8 = segment start. OUT: concat buffer holds the NUL-terminated segment,
; rsi = segment text for Intern. Clobbers r10/r11/rax/rdx/rsi/rdi.

%macro SL_EXPL_EMIT_SEGMENT 0
     lea r10, [rel sl_concat_buf + SL_ATOM_TEXT_OFF] ; r10 = dest, content starts at [1]
     lea r11, [rel sl_concat_buf + SL_CONCAT_BUF_LAST] ; r11 = end of buffer
     lea rax, [rsi + r8]                       ; rax = &source[seg_start]
     lea rdx, [r10 + rcx]                      ; rdx = one past payload
     cmp rdx, r11                              ; payload, <= r11-1 + NUL, <= r11 fit?
     ja sl_concat_overflow                     ; no: FATAL, a truncated result atom is silent data corruption
     mov rsi, rax                              ; rsi = source cursor
     mov rdi, r10                              ; rdi = dest cursor
     rep movsb                                 ; copy segment, DF=0
     mov byte [rdi], 0                         ; NUL-terminate segment
     lea rsi, [rel sl_concat_buf]              ; rsi = segment text for Intern
%endmacro
sl_expl_match_found:
; delimiter matched at offset r9. Emit source[seg_start..r9] as atom.
     mov rcx, r9                               ; rcx = match position, qword
     sub rcx, r8                               ; rcx = segment length
     SL_EXPL_EMIT_SEGMENT                      ; copy + bounds + NUL + Intern prep
     mov [rsp + SL_EXPL_R9_OFF], r9            ; save scan_pos across Intern/Cons, qword
     call Intern                               ; eax = segment atom, overwrites r8/r9/xmm0-4
; Cons it onto result list, append it at the tail in source order
     call sl_expl_append                       ; append in source order
; reload r9 from its stack slot, overwritten by Intern+Cons ; seg_start is NOT reloaded
; because it is recomputed from the scan position below
     mov r9, [rsp + SL_EXPL_R9_OFF]            ; restore scan_pos, qword
; advance past the delimiter with the HOISTED length, no per-match delimiter re-walk
; and set the new segment start
     add r9, [rsp + SL_EXPL_DELIMLEN_OFF]      ; scan pos = past the matched delimiter
     mov r8, r9                                ; new segment start = scan pos
; reload rsi/rdi from the base-pointer slots: Intern's rep movsb
; consumed them, the slots are NIL-safe, sl_str_nul included
     mov rsi, [rsp + SL_EXPL_SRCBASE_OFF]      ; rsi = source base
     mov rdi, [rsp + SL_EXPL_DELIMBASE_OFF]    ; rdi = delimiter base
     vpxor xmm0, xmm0, xmm0                    ; zero vector
     vpbroadcastb xmm4, byte [rdi]             ; xmm4 = delimiter[0] in all 16 lanes
     jmp sl_expl_scan                          ; continue outer scan
sl_expl_emit_rest:
; emit source[seg_start..end] as final segment
; rest length = src_strlen - seg_start: BOTH hoisted, no leaf call here
     mov rcx, [rsp + SL_EXPL_SRCLEN_OFF]       ; rcx = src_strlen, qword
     sub rcx, r8                               ; rcx = rest segment length
     SL_EXPL_EMIT_SEGMENT                      ; copy + bounds + NUL + Intern prep
     call Intern                               ; eax = final segment atom
; Cons onto result
     call sl_expl_append                       ; append the final segment in source order
     jmp sl_expl_done                          ; list complete
sl_expl_char_split:
; empty delimiter: split source into individual characters
     mov [rsp + SL_EXPL_SRCBASE_OFF], rsi      ; save source base pointer, avoids per-char atom re-derive
     xor r9d, r9d                              ; offset = 0
; result head and tail link were initialized at sl_expl_delim_ok
; concat_buf type byte: written ONCE per call, Intern only READS the
; buffer and Cons never touches it, so [0] survives every iteration
     mov byte [rel sl_concat_buf], SL_TYPE_STRING ; type byte at [0]
sl_expl_char_loop:
     mov al, byte [rsi + r9]                   ; source[offset], only AL is consumed
     test al, al                               ; NUL terminator?
     jz sl_expl_done                           ; done: list built in source order
; single-char atom: store it directly into the concat buffer, absolute RIP-relative
; operands, no r10 LEA needed; this loop's only consumer of r10 was these two stores
     mov byte [rel sl_concat_buf + SL_ATOM_TEXT_OFF], al ; store char at content byte 0
     mov byte [rel sl_concat_buf + SL_ATOM_TEXT_OFF + SL_ONE_BYTE], 0 ; NUL-terminate, content byte 1
     mov [rsp + SL_EXPL_CHAR_OFF], r9          ; save offset, qword, matches the %define's role; a dword store would leave the upper dword as GC-scanned garbage
     lea rsi, [rel sl_concat_buf]              ; rsi = char text for Intern
     call Intern                               ; eax = char atom
     mov rsi, [rsp + SL_EXPL_SRCBASE_OFF]      ; rsi = source base, single mov instead of atom re-derive
     call sl_expl_append                       ; links the fresh cell; Cons may GC
     mov r9d, [rsp + SL_EXPL_CHAR_OFF]         ; restore offset AFTER Cons, GC_Collect writes r9 with the bitmap address, so a reload before the call would be lost
     inc r9d                                   ; advance offset
     jmp sl_expl_char_loop                     ; continue char loop
sl_expl_done:
     mov eax, [rsp + SL_EXPL_RESULT_OFF]       ; return already source-ordered head, GC-rooted
     add rsp, 120                              ; clean frame, 32-byte home area + 9 named slots + 1 retired slot + 8 tail padding
     pop rdi                                   ; restore rdi
     pop rsi                                   ; restore rsi
     ret                                       ; return list
; One shared append helper replaces the three inlined prepend sequences.
; Caller RSP = S at the call; entry RSP = S-8; after `sub rsp,40`, RSP = S-48.
; The caller's tail-pointer slot S+48 is therefore [rsp+96].
sl_expl_append:
     sub rsp, 40                               ; Win64 shadow space + alignment for Cons
     xor edx, edx                              ; cdr = NIL; link into the previous tail after construction
     call Cons                                 ; eax = tagged new cell, may run GC
     mov rdx, [rsp + SL_EXPL_TAILPTR_OFF + SL_EXPL_APPEND_BIAS] ; pointer to the previous tail's CDR, or the head root
     mov [rdx], eax                            ; append the new cell in source order
     btr eax, SL_CONS_BIT                      ; eax = new cell index
     lea rdx, [r13 + rax*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF] ; next tail CDR address
     mov [rsp + SL_EXPL_TAILPTR_OFF + SL_EXPL_APPEND_BIAS], rdx ; park the tail pointer in the caller's free frame slot
     add rsp, 40                               ; release shadow space + alignment
     ret                                       ; return to caller
sl_expl_notatom:
     FATAL_VALUE "Split requires string atom arguments", eax ; name the offender; eax is the source, :596 or delimiter, :599 atom
; %undef cleanup: file-local defines and macros leave scope here, house convention --
%undef SL_EXPL_SRCLEN_OFF
%undef SL_EXPL_DELIMLEN_OFF
%undef SL_EXPL_DELIMBASE_OFF
%undef SL_EXPL_SRCBASE_OFF
%undef SL_EXPL_DELIM_OFF
%undef SL_EXPL_SRC_OFF
%undef SL_EXPL_R9_OFF
%undef SL_EXPL_APPEND_BIAS
%undef SL_EXPL_TAILPTR_OFF
%undef SL_EXPL_CHAR_OFF
%undef SL_EXPL_RESULT_OFF
%undef SL_STN_FRAME
%undef SL_STN_OVF_OFF
%undef SL_REPL_FRAME
%undef SL_REPL_NEW_OFF
%undef SL_REPL_OLD_OFF
%undef SL_REPL_SRC_OFF
%undef SL_REPL_FIRST
%undef SL_STRLEN_FRAME
%undef SL_FIND_FRAME
%undef SL_FIND_NEEDLE_OFF
%undef SL_SUB_WIN_BYTES
%undef SL_SUB_FRAME
%undef SL_SUB_START_OFF
%undef SL_SUB_SRC_OFF
%undef SL_SKIP_FIRST
%undef SL_LEA_X2
%undef SL_LEA_X5
%undef SL_CASE_WIN_LAST_OFF
%undef SL_ASCII_CASE_CLEAR
%undef SL_ASCII_CASE_DELTA
%unmacro SL_CASEFOLD 2                         ; remove: sl_str_lower/sl_str_upper generators above are its only uses
%unmacro SL_CASEFOLD_VEC 3                     ; remove: the SL_CASEFOLD expansions above are its only uses
%unmacro SL_NEEDLE_INNER 5                     ; remove: the find/replace/split expansions above are its only uses
%unmacro SL_FIRSTCHAR_SCAN 5                   ; remove: the find/replace/split expansions above are its only uses
