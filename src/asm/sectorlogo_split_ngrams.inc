; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; --- split-ngrams frame layout, 32-byte Win64 home area + 9 named slots at 40..104 ---
%define SL_NG_N_OFF 40                         ; rsp slot: raw n, NumVal'd, >= 1, dword
%define SL_NG_SRC_OFF 48                       ; rsp slot: source atom, string path / window-start node, list path, tagged, dword
%define SL_NG_RESULT_OFF 56                    ; rsp slot: result list head, GC conservative root, dword
%define SL_NG_WINDOW_OFF 64                    ; rsp slot: window list head under construction, list path, GC root, dword
%define SL_NG_CURSOR_OFF 72                    ; rsp slot: element-walk cursor inside the window, list path, tagged node, dword
%define SL_NG_START_OFF 80                     ; rsp slot: window start offset, string path, dword
%define SL_NG_LEN_OFF 88                       ; rsp slot: source byte length, string path, dword
%define SL_NG_WINDOW_TAIL_OFF 96               ; list path: pointer to window head slot or last CDR field
%define SL_NG_RESULT_TAIL_OFF 104              ; list path: pointer to result head slot or last CDR field
%define SL_NG_PUSH8 8                          ; extra rsp shift from the one balanced push guarding ebx across PopCons
%define SL_NG_FRAME 112                        ; home area + 9 named slots + alignment
sl_split_ngrams:
     push rsi                                  ; save non-volatile. RSP = 0 mod 16
     push rdi                                  ; save non-volatile. RSP = 8 mod 16
     push rbx                                  ; save for atom/element scratch. RSP = 0 mod 16
     sub rsp, SL_NG_FRAME                      ; frame. RSP = 0 mod 16
     call EvalNextArg                          ; eax = source, tagged
     mov [rsp + SL_NG_SRC_OFF], eax            ; park BEFORE the next EvalNextArg, it overwrites eax
     call EvalNextArg                          ; eax = n, tagged
     call NumVal                               ; eax = raw int n
     test eax, eax                             ; n < 1?, jle: ZF covers n=0, SF!=OF covers negatives
     jle sl_ng_bad_n                           ; yes: name the offender
     mov [rsp + SL_NG_N_OFF], eax              ; n is legal, park it
     mov ecx, [rsp + SL_NG_SRC_OFF]            ; ecx = source value
     shr ecx, 30                               ; 2-bit tag code (0 atom/NIL, 1 NUMBER, 2 CONS, 3 FLOAT), ZF=1 for a plain atom/NIL
     jz sl_ng_string_path                      ; zero code means plain atom/NIL
     cmp ecx, 2                                ; exactly CONS, not FLOAT or NUMBER
     je sl_ng_list_path                        ; list path
     jmp sl_ng_bad_source                      ; NUMBER, bit30 or FLOAT, bit31+30
sl_ng_string_path:
     mov ecx, [rsp + SL_NG_SRC_OFF]            ; ecx = source atom
     test ecx, ecx                             ; NIL source?
     jz sl_ng_len_zero                         ; yes: len = 0, skip the AVX2 probe
     lea rsi, [r13 + rcx + SL_ATOM_TEXT_OFF]   ; rsi = source text in heap, committed atom area
     call sl_swar_strlen                       ; ecx = byte length, rsi/rdi preserved
     mov rbx, rsi                              ; rbx = window source pointer, non-volatile: survives Intern/Cons; slid by 1 per window below, replacing the per-window slot reload + lea + add. rbx is pushed at :36 and used only by the LIST path's j counter, so it is free here
     mov [rsp + SL_NG_LEN_OFF], ecx            ; park the measured byte length
     jmp sl_ng_windows                         ; common window-loop init
sl_ng_len_zero:
     mov dword [rsp + SL_NG_LEN_OFF], 0        ; NIL -> zero windows
sl_ng_windows:
     mov dword [rsp + SL_NG_RESULT_OFF], 0     ; result NIL for the no-complete-window case; list completion reads RESULT directly
     mov eax, [rsp + SL_NG_N_OFF]              ; eax = n
     cmp eax, [rsp + SL_NG_LEN_OFF]            ; n > byte length?
     ja sl_ng_string_done                      ; no complete window, return NIL directly, nothing was written to the shared buffer
     cmp eax, SL_CONCAT_BUF_LAST               ; n fits the concat buffer, content + NUL ?
     jae sl_ng_overflow                        ; no: FATAL, a truncated ngram atom is silent data corruption, never clamp
     mov byte [rel sl_concat_buf], SL_TYPE_STRING ; type byte at [0], stored now that a window exists: the only calls below are Intern and Cons, and nothing in either's transitive call graph writes this buffer, Intern -> StringHash / aes_string_hash / GC_Collect; Cons -> GC_Collect; none of the four reference sl_concat_buf, the only writers in that file are lg_value_to_str and sl_fatal_value, reached from PRINT and from a FATAL respectively.Both copy engines store at offset 1 and above.
     mov ecx, [rsp + SL_NG_LEN_OFF]            ; ecx = byte length, reached only when len >= n: the ja above already diverted n > len
     sub ecx, [rsp + SL_NG_N_OFF]              ; ecx = len - n = last legal window start
     mov [rsp + SL_NG_START_OFF], ecx          ; start at the last complete window
     add rbx, rcx                              ; slide the byte pointer to that window
sl_ng_window_loop:
; first window start is >= 0 (the n > len guard diverted to sl_ng_string_done), so the body always runs once
; copy window bytes [start, start+n verbatim, NO fold, byte windows
     lea r10, [rel sl_concat_buf + SL_ATOM_TEXT_OFF] ; r10 = dest cursor, content starts at [1]
     mov rsi, rbx                              ; rsi = &source[start], rbx slid past the previous window by the loop tail; set fresh per window, Intern overwritten rsi
     mov ecx, [rsp + SL_NG_N_OFF]              ; ecx = n
     cmp ecx, 32                               ; large window?, below ~32 the ERMSB setup dominates and the byte loop wins
     jb sl_ng_copy                             ; small n, byte loop
     mov rdi, r10                              ; rep movsb dest = the r10 cursor set above, 3 bytes vs the 7-byte RIP-relative lea
     rep movsb                                 ; n bytes verbatim, exactly n read, exactly n written
     mov r10, rdi                              ; r10 = one past the last content byte, the same cursor contract the byte loop leaves behind
     jmp sl_ng_copy_done                       ; r10 already sits one past the last content byte, as on the byte path
sl_ng_copy:
     movzx edx, byte [rsi]                     ; verbatim byte, NO fold, this is the one-line delta vs sl_tok_copy
     mov byte [r10], dl                        ; store
     inc rsi                                   ; advance source
     inc r10                                   ; advance dest
     dec ecx                                   ; one less to copy, sets ZF for the bottom test
     jnz sl_ng_copy                            ; more bytes?, ZF comes free from the dec, no jecxz, no jmp
sl_ng_copy_done:
     mov byte [r10], 0                         ; NUL-terminate the ngram
     lea rsi, [rel sl_concat_buf]              ; rsi = ngram text for Intern
     call Intern                               ; eax = ngram atom, overwrites ALL volatiles incl. r10/rax ; eax stays the Cons car, so no staging
     mov edx, [rsp + SL_NG_RESULT_OFF]         ; cdr = result so far
     call Cons                                 ; eax = new result head
     mov [rsp + SL_NG_RESULT_OFF], eax         ; GC-root re-store, frame slot, the scan sees it
     dec rbx                                   ; previous byte position
     dec dword [rsp + SL_NG_START_OFF]         ; previous start; flags drive the loop
     jns sl_ng_window_loop                     ; repeat while the start is >= 0
sl_ng_string_done:
     mov r8d, [rsp + SL_NG_RESULT_OFF]         ; backward iteration + Cons already made source-order list
     jmp sl_ng_done                            ; list input uses direct tail appends
sl_ng_list_path:
     mov dword [rsp + SL_NG_RESULT_OFF], 0     ; result = NIL, root retained across GC
     lea rax, [rsp + SL_NG_RESULT_OFF]         ; first outer append writes to result head
     mov [rsp + SL_NG_RESULT_TAIL_OFF], rax    ; later appends write previous result CDR
sl_ng_outer:
     mov eax, [rsp + SL_NG_SRC_OFF]            ; eax = SRC, window-start node
     test eax, eax                             ; bit31 clear = NIL or atom tail?
     jns sl_ng_list_done                       ; walk ended: result already in source order
     test eax, SL_NUMBER_TAG                   ; bit31+bit30 both set = FLOAT tail?
     jnz sl_ng_list_done                       ; same termination as NIL/atom
     mov [rsp + SL_NG_CURSOR_OFF], eax         ; cursor = window start node
     mov dword [rsp + SL_NG_WINDOW_OFF], 0     ; window head = NIL, visible to GC
     lea rax, [rsp + SL_NG_WINDOW_OFF]         ; first append writes window head
     mov [rsp + SL_NG_WINDOW_TAIL_OFF], rax    ; later appends write previous CDR
     xor ebx, ebx                              ; j = 0 in the live non-volatile
sl_ng_inner:
     mov eax, [rsp + SL_NG_CURSOR_OFF]         ; eax = cursor node
     test eax, eax                             ; source exhausted mid-window, NIL/atom tail ?
     jns sl_ng_list_done                       ; incomplete window: discard, return earlier complete windows
     test eax, SL_NUMBER_TAG                   ; FLOAT tail mid-window?
     jnz sl_ng_list_done                       ; same termination, never a spurious NIL element
     push rbx                                  ; preserve the window index across PopCons, balanced push/pop keeps Cons's call alignment
     call PopCons                              ; fused: ebx = element, eax = next source node, one traversal
     mov [rsp + SL_NG_CURSOR_OFF + SL_NG_PUSH8], eax ; root the successor BEFORE Cons can GC, slot adjusted for the push
     mov eax, ebx                              ; car for Cons
     pop rbx                                   ; restore the window counter and the call alignment
     xor edx, edx                              ; append: new cell has CDR = NIL
     call Cons                                 ; eax = new cell; GC may run, head is rooted
     mov rdx, [rsp + SL_NG_WINDOW_TAIL_OFF]    ; current append slot: head or previous CDR
     mov [rdx], eax                            ; link newly allocated cell in input order
     btr eax, SL_CONS_BIT                      ; untag cell index for its CDR address
     lea rdx, [r13 + rax*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF] ; address of the new cell's CDR field
     mov [rsp + SL_NG_WINDOW_TAIL_OFF], rdx    ; next append target
     inc ebx                                   ; one more element collected
     cmp ebx, dword [rsp + SL_NG_N_OFF]        ; collected n elements?
     jb sl_ng_inner                            ; not yet: collect the next element
     mov eax, [rsp + SL_NG_WINDOW_OFF]         ; car = window already in source order, GC-rooted
     xor edx, edx                              ; result node will be the new last node
     call Cons                                 ; eax = output node; old result head remains rooted
     mov rdx, [rsp + SL_NG_RESULT_TAIL_OFF]    ; head slot or last output CDR
     mov [rdx], eax                            ; append complete window in source order
     btr eax, SL_CONS_BIT                      ; get new cell index
     lea rdx, [r13 + rax*CONS_SCALE + CONS_BASE_OFF + CONS_CDR_OFF] ; address of the new output cell's CDR field
     mov [rsp + SL_NG_RESULT_TAIL_OFF], rdx    ; set next output append target
     mov eax, [rsp + SL_NG_SRC_OFF]            ; eax = SRC node
     call Cdr                                  ; SRC = next window start, one element later
     mov [rsp + SL_NG_SRC_OFF], eax            ; park the advanced window start
     jmp sl_ng_outer                           ; next window
sl_ng_list_done:
     mov r8d, [rsp + SL_NG_RESULT_OFF]         ; already source order, even for incomplete last window
sl_ng_done:
     mov eax, r8d                              ; text-order result, NIL when zero windows
     add rsp, SL_NG_FRAME                      ; drop frame
     pop rbx                                   ; restore non-volatile
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     ret                                       ; return list in eax
sl_ng_bad_source:
     FATAL "SectorLogo: NGrams requires a string atom or list source" ; NUMBER/FLOAT source
sl_ng_bad_n:
     FATAL "SectorLogo: NGrams requires n >= 1 (NUMBER)" ; n = 0 / negative; FLOAT n dies earlier in NumVal
sl_ng_overflow:
     FATAL "SectorLogo: NGrams ngram exceeds 2 MiB concat buffer" ; capacity: a truncated ngram is silent corruption, never clamp
%undef SL_NG_FRAME
%undef SL_NG_PUSH8
%undef SL_NG_LEN_OFF
%undef SL_NG_WINDOW_TAIL_OFF
%undef SL_NG_RESULT_TAIL_OFF
%undef SL_NG_START_OFF
%undef SL_NG_CURSOR_OFF
%undef SL_NG_WINDOW_OFF
%undef SL_NG_RESULT_OFF
%undef SL_NG_SRC_OFF
%undef SL_NG_N_OFF
