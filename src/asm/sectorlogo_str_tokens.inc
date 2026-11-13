; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; --- str-tokens frame layout, named slots at 40..72 + pads at 0..39 and 73..79 ---
%define SL_TOK_TAILPTR_OFF 40                  ; rsp slot: pointer to next cons cdr/head link; outside Win64 32-byte call shadow
%define SL_TOK_R9_OFF 48                       ; rsp slot: parked r9d across Intern/Cons, scan offset carrier
%define SL_TOK_SRC_OFF 56                      ; rsp slot: source atom, re-derive rsi after Intern
%define SL_TOK_RESULT_OFF 64                   ; rsp slot: result list head, GC conservative root
%define SL_TOK_BOUNDARY_OFF 72                 ; rsp slot: boundary flag, 1=not-NUL, 0=NUL; avoids post-emit source re-read, a LIVE flag, not the alignment pad, which is at offsets 0..39 and 73..79
%define SL_TOK_FRAME 80                        ; frame bytes, named slots at 40..72 + pads at 0..39 and 73..79; 3 pushes + 80 = 104 = 8 mod 16 -> 16-aligned call sites from the 8-mod-16 entry
%define SL_TOK_RUN_BUDGET (SL_CONCAT_BUF_LAST - SL_ATOM_TEXT_OFF) ; max bytes one run may stage, 0x1FFFFE, applied as a bias on the scan position, r11d = run start + budget, so the staging cursor needs no register of its own
sl_str_tokens:
     push rsi                                  ; save non-volatile. RSP = 0 mod 16
     push rdi                                  ; save non-volatile. RSP = 8 mod 16
     push rbx                                  ; frame shape: with 3 pushes the call sites below land 16-aligned, 3*8 + SL_TOK_FRAME = 104 = 8 mod 16, from an 8-mod-16 entry.rbx carries the run-biased staging base, non-volatile, so it survives Intern/Cons/GC_Collect untouched.RSP = 0 mod 16
     sub rsp, SL_TOK_FRAME                     ; frame: 4 named slots at 48..72 + pads -> RSP = 0 mod 16
     call EvalNextArg                          ; eax = text atom
     ASSERT_ATOM_OR_NIL sl_tok_notatom         ; STR-TOKENS source: NIL/atom ok, CONS/NUMBER -> FATAL
     mov [rsp + SL_TOK_SRC_OFF], rax           ; save source atom to stack slot, load-bearing: sl_tok_emit re-derives rsi from it after Intern/Cons
; derive rsi = source text, NIL -> empty string constant
     test eax, eax                             ; NIL source?, eax still holds it, ASSERT_ATOM_OR_NIL is flags-only and no call intervenes
     jz sl_tok_src_nil                         ; yes: use empty string
     mov ecx, eax                              ; ecx = atom offset, 32-bit, zero-extends in the lea below
     lea rsi, [r13 + rcx + SL_ATOM_TEXT_OFF]   ; rsi = source string. rcx is the clean 32-bit atom offset; matches the mid-scan reload at :73-75 exactly
     jmp sl_tok_scan_init                      ; source pointer ready
sl_tok_src_nil:
     lea rsi, [rel sl_str_nul]                 ; empty source
sl_tok_scan_init:
     mov qword [rsp + SL_TOK_RESULT_OFF], 0    ; head pointer remains the GC-visible root
     lea rax, [rsp + SL_TOK_RESULT_OFF]        ; first append writes the head slot
     mov [rsp + SL_TOK_TAILPTR_OFF], rax       ; subsequent appends write the previous cons CDR
     mov byte [rel sl_concat_buf], SL_TYPE_STRING ; type byte at [0], stays valid across runs, only content bytes are restaged
     xor r8d, r8d                              ; r8d = run start offset
     xor r9d, r9d                              ; r9d = current scan offset
     lea rbx, [rel sl_concat_buf + SL_ATOM_TEXT_OFF] ; rbx = staging base, biased by the run start, rbx + r9d IS the next content byte.Unbiased here because r9d == r8d == 0
     mov r11d, SL_TOK_RUN_BUDGET               ; r11d = run start + byte budget, r9d == r8d == 0 here, so this is the budget itself
     lea rdi, [rel sl_tok_class]               ; shared ASCII character class table base
sl_tok_scan:
     movzx eax, byte [rsi + r9]                ; original byte
     test byte [rdi + rax], SL_TOK_ALNUM_MASK  ; ASCII alnum? bit6 is 1 exclusively for 0-9/A-Z/a-z
     jz sl_tok_not_alnum                       ; no: NUL or boundary
     or al, 0x20                               ; THE ONLY case-fold, and it runs on EVERY alnum byte. Identity on '0'-'9', bit5 set by construction and on 'a'-'z'; maps 'A'-'Z' to 'a'-'z'. Table-gated, so no second fold is possible.
     cmp r9d, r11d                             ; run byte budget: r9d - run start < SL_TOK_RUN_BUDGET?, rbx + r9d tracks the staging cursor exactly and r11d is run start + budget, so comparing the two scan registers bounds the write
     jae sl_tok_overflow                       ; no: FATAL, a truncated token atom is silent data corruption
     mov byte [rbx + r9], al                   ; stage the folded byte at sl_concat_buf + SL_ATOM_TEXT_OFF +, r9d - run start, the cursor is DERIVED from the scan position, so there is no second one to advance
     inc r9d                                   ; advance scan position
     jmp sl_tok_scan                           ; continue scan
sl_tok_not_alnum:
; eax still holds the byte the table just rejected, the NUL test needs no reload
     test eax, eax                             ; end of source?
     jz sl_tok_flush                           ; yes: flush trailing run, done
     mov byte [rsp + SL_TOK_BOUNDARY_OFF], 1   ; flag: not-NUL, will continue scan after emit
     cmp r9d, r8d                              ; run non-empty?, r9 != r8 <=> length >= 1
     je sl_tok_reset                           ; zero-length run: drop
; fall through into sl_tok_emit: the per-TOKEN path, and it takes no jump
sl_tok_emit:
; the run is already staged in concat_buf, folded during the scan, terminate + intern
     mov byte [rbx + r9], 0                    ; NUL-terminate token. The gate above lets a byte stage only while r9d < run_start + SL_TOK_RUN_BUDGET, so after the final `inc r9d` the run length is AT MOST SL_TOK_RUN_BUDGET, 0x1FFFFE, and 1 + 0x1FFFFE == SL_CONCAT_BUF_LAST exactly, so this lands ON the last valid byte of sl_concat_buf at worst. Do NOT "tighten" that gate's jae to ja: that is what would push this store one byte past the buffer
     lea rsi, [rel sl_concat_buf]              ; rsi = staged token text for Intern
     mov [rsp + SL_TOK_R9_OFF], r9             ; save scan pos across Intern/Cons, r9d IS live: sl_tok_emit_done advances it
     call Intern                               ; eax = token atom, overwrites r8/r9/r10/r11 ; eax stays the Cons car
     xor edx, edx                              ; create a terminating cons cell, CDR = NIL
     call Cons                                 ; eax = tagged cell, overwrites volatile registers
     mov rdx, [rsp + SL_TOK_TAILPTR_OFF]       ; address of current tail link
     mov [rdx], eax                            ; append cell in input order; first append updates head
     btr eax, SL_CONS_BIT                      ; index of appended cell, remove tag
     lea rdx, [r13 + rax*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF] ; address of new CDR slot
     mov [rsp + SL_TOK_TAILPTR_OFF], rdx       ; next append uses the new CDR
; re-derive rsi from the SRC slot, Intern may append atoms; atom offsets never move
     mov r9d, dword [rsp + SL_TOK_R9_OFF]      ; restore scan pos
     mov ecx, [rsp + SL_TOK_SRC_OFF]           ; reload source atom, non-NIL proven: empty source exits at the first NUL before any emit
     lea rsi, [r13 + rcx + SL_ATOM_TEXT_OFF]   ; rsi = source string
     cmp byte [rsp + SL_TOK_BOUNDARY_OFF], 0   ; was the byte at r9d NUL?, hot stack slot instead of cold source re-read
     jz sl_tok_finish_list                     ; yes: end of string, go reverse, no cursor re-init needed, staging is over
; fall through into sl_tok_reset: identical re-init tail, skip boundary byte, new run start, re-base cursor + budget
sl_tok_reset:
     inc r9d                                   ; step past the boundary byte
     mov r8d, r9d                              ; new run start = scan position
     lea rbx, [rel sl_concat_buf + SL_ATOM_TEXT_OFF] ; re-base the staging cursor: the next run, post-emit or post-drop restages from the content start
     sub rbx, r9                               ; rbx = staging base - run start, r9d IS r8d here
     lea r11d, [r9 + SL_TOK_RUN_BUDGET]        ; r11d = run start + byte budget. Recomputed here because Intern/Cons overwrite every volatile and the budget is derived from r9d, never carry it across a call with an inc
     jmp sl_tok_scan                           ; continue scan
sl_tok_flush:
; NUL: flush trailing run [r8d, r9d) if non-empty, then return the list head
     mov byte [rsp + SL_TOK_BOUNDARY_OFF], 0   ; flag: NUL, will reverse after emit
     cmp r9d, r8d                              ; trailing run non-empty?, r9 != r8 <=> length >= 1
     jne sl_tok_emit                           ; non-empty trailing run: the ONE taken jump per call, r9d already points at the NUL
sl_tok_finish_list:
     mov eax, [rsp + SL_TOK_RESULT_OFF]        ; already built in source order: no reversal pass
     add rsp, SL_TOK_FRAME                     ; clean frame, eax already holds the result head
     pop rbx                                   ; restore rbx
     pop rdi                                   ; restore rdi
     pop rsi                                   ; restore rsi
     ret                                       ; return list
sl_tok_notatom:
     FATAL_VALUE "Words requires a string or word atom argument", eax ; NIL / CONS / NUMBER given to Words
sl_tok_overflow:
     FATAL "SectorLogo: token exceeds 2 MiB concat buffer" ; capacity: a truncated token atom is silent data corruption, never clamp
; %undef cleanup: file-local defines leave scope here, house convention --
%undef SL_TOK_TAILPTR_OFF
%undef SL_TOK_R9_OFF
%undef SL_TOK_SRC_OFF
%undef SL_TOK_RESULT_OFF
%undef SL_TOK_BOUNDARY_OFF
%undef SL_TOK_FRAME
%undef SL_TOK_RUN_BUDGET
