; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define SDF_HANDLER_REL_OFF 4                  ; relative-handler dword offset within one dispatch record

lg_ensure_word:
     test eax, SL_TAG_MASK                     ; tagged value, CONS/NUMBER/FLOAT ? eax tested directly, the zero-extend copy is deferred to the untagged path
     jnz .ew_done                              ; yes: not an atom offset, return as-is
     mov ecx, eax                              ; rcx = atom offset, zero-extended: the heap accesses below are 64-bit and the input arrives in eax only, upper bits undefined
     cmp ecx, [rel sl_dispatch_highwater]      ; offset below the builtin region high-water mark?
     jb .ew_done                               ; yes: WORD or FFI-type STRING by construction, return as-is either way, the four SL_TYPE_STRING CallLibrary type atoms below the mark MUST pass through unchanged
     cmp byte [r13 + rcx], SL_TYPE_STRING      ; STRING type?
     je .ew_convert                            ; yes: convert, rare path, out of line below
     ret                                       ; no: WORD, return eax unchanged, dominant case: zero taken jumps
.ew_convert:
     push rsi                                  ; save non-volatile: the copy walk owns rsi, lodsb
     push rdi                                  ; save non-volatile: the copy walk owns rdi, stosb
     sub rsp, 8                                ; parity pad: 2 pushes leave RSP 8 mod 16; the Intern call below needs 0 mod 16 at the call instruction
; Copy atom text to sl_concat_buf with SL_TYPE_WORD prefix
     lea rsi, [r13 + rcx + SL_ATOM_TEXT_OFF]   ; source: atom text past the STRING type byte
     lea rdi, [rel sl_concat_buf]              ; dest base
     mov byte [rdi], SL_TYPE_WORD              ; set WORD type byte
     inc rdi                                   ; dest = past type byte
     lea r8, [rel sl_concat_buf + SL_CONCAT_BUF_LAST] ; dest bound: last writable byte of the concat buffer
.ew_copy:
     lodsb                                     ; al = [rsi++], string instruction form: 2 instr/byte vs 4; DF=0 is the Win64 entry invariant
     cmp al, 'A'                               ; below 'A'?
     jb .ew_store                              ; yes: not uppercase, digits, punct, lowercase fold to themselves
     cmp al, 'Z'                               ; above 'Z'?
     ja .ew_store                              ; yes: not uppercase
     add al, 'a' - 'A'                         ; fold A-Z -> a-z: "Make" MUST resolve to the builtin "make" atom or dispatch misses it
.ew_store:
     cmp rdi, r8                               ; destination past the last writable byte?
     ja .ew_overflow                           ; yes: buffer capacity exhausted
     stosb                                     ; [rdi++] = al
     test al, al                               ; NUL terminator?
     jnz .ew_copy                              ; no: keep copying
     lea rsi, [rel sl_concat_buf]              ; rsi = concat_buf, type byte + text
     call Intern                               ; eax = WORD-type atom, entered at the ABI's 8 mod 16 via the parity pad
     add rsp, 8                                ; undo the parity pad
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
.ew_done:
     ret                                       ; return eax, possibly converted or unchanged
.ew_overflow:
     FATAL "SectorLogo: atom text exceeds the 2 MiB concat buffer (STRING->WORD conversion)"
sl_trace_print_indent:
     mov ecx, [rel sl_trace_indent]            ; ecx = depth count
     test ecx, ecx                             ; zero depth?
     jz .tind_done                             ; yes: nothing to print
     cmp ecx, SL_TRACE_MAX_INDENT              ; clamp to max
     jle .tind_ok                              ; within range
     mov ecx, SL_TRACE_MAX_INDENT              ; clamp to max indent
.tind_ok:
     push rsi                                  ; save non-volatile: rsi holds the space block below. ONE push is also the parity fix: entry is 8 mod 16, Win64 callee entry, so a single push leaves RSP 0 mod 16 for the call
     lea edx, [rcx + rcx]                      ; edx = 2 x depth, space count ; lea, not mov+add: one uop and one byte shorter
     lea rsi, [rel sl_trace_spaces]            ; rsi = space block
     call PrintRaw                             ; write all spaces in ONE WriteFile
     pop rsi                                   ; restore non-volatile
.tind_done:
     ret                                       ; return to caller
sl_dispatch_fill:
     push rdi                                  ; save non-volatile, rep stosq uses rdi as the fill cursor
; --- pre-fill: every entry = its position's fallback ---
     lea rdi, [rel sl_dispatch_stmt_table]     ; rdi = statement table base
     lea rax, [rel EvalLogoStream.stmt_fallback] ; rax = statement fallback handler
     mov ecx, SL_DISPATCH_TABLE_ENTRIES        ; 1024 qword slots
     rep stosq                                 ; fill: every entry = fallback
     lea rdi, [rel sl_dispatch_expr_table]     ; rdi = expression table base
     lea rax, [rel EvalPrimary.expr_fallback]  ; rax = expression fallback handler
     mov ecx, SL_DISPATCH_TABLE_ENTRIES        ; 1024 qword slots
     rep stosq                                 ; fill: every entry = fallback
; Populate the two compact tables from relative descriptors.
     lea r10, [rel sl_dispatch_stmt_table]     ; default destination is the statement table
     lea r11, [rel sl_dispatch_expr_table]     ; alternate destination for expression records
     lea rdx, [rel sl_lg_initialized]          ; base of the relative atom-slot offsets
     lea r9, [rel sl_dispatch_fill]            ; base of the signed handler offsets
     lea r8, [rel sl_dispatch_records]         ; first handler record
.sdf_handler:
     mov eax, [r8]                             ; atom-slot offset plus the expression bit
     cmp eax, -1                               ; sentinel after the final handler?
     je .sdf_op_start                          ; continue with the operator bitmap
     mov rdi, r10                              ; statement table unless the expression bit is set
     btr eax, 31                               ; extract the table selector from the offset
     cmovc rdi, r11                            ; select the expression table
     mov eax, [rdx + rax]                      ; load the runtime atom offset
     cmp eax, SL_DISPATCH_MAX_OFF              ; refuse an out-of-range handler offset
     jae sdf_overflow_fatal                    ; a builtin must fit the compact table
     test al, SL_ATOM_GRANULE_BYTES - 1        ; ensure the qword slot is aligned
     jnz sdf_overflow_fatal                    ; a builtin must start on a granule
     bts dword [rel sl_builtin_bitmap], eax    ; retain raw byte indexing for builtin membership
     movsxd rcx, dword [r8 + SDF_HANDLER_REL_OFF] ; signed handler displacement from sl_dispatch_fill
     add rcx, r9                               ; resolve the handler pointer
     mov [rdi + rax], rcx                      ; the raw byte offset already includes the qword stride
     add r8, 8                                 ; advance one handler record
     jmp .sdf_handler                          ; continue in declaration order
.sdf_op_start:
     lea r8, [rel sl_dispatch_operators]       ; first operator's atom-slot offset
.sdf_operator:
     mov eax, [r8]                             ; relative offset or end sentinel
     cmp eax, -1                               ; any operators left?
     je .sdf_ready                             ; the bitmap is complete
     mov eax, [rdx + rax]                      ; load the operator's runtime atom offset
     cmp eax, SL_DISPATCH_MAX_OFF              ; enforce the bitmap limit
     jae sdf_overflow_fatal                    ; prevent an out-of-bounds bit write
     bts dword [rel sl_op_atom_bitmap], eax    ; retain raw byte indexing for operator membership
     add r8, 4                                 ; advance one atom-slot offset
     jmp .sdf_operator                         ; continue the bitmap fill
.sdf_ready:
     pop rdi                                   ; restore the caller's destination register
     mov eax, [rel sl_atom_free]               ; end of the pinned builtin region
     cmp eax, SL_DISPATCH_MAX_OFF              ; every byte offset below the floor must fit
     ja sdf_overflow_fatal                     ; reject an oversized builtin region
     mov [rel sl_dispatch_highwater], eax      ; publish the builtin region for dispatch and GC
     ret                                       ; both tables and membership bitmaps are ready
sdf_overflow_fatal:
     FATAL "SectorLogo: invalid builtin dispatch offset or region" ; compact table bounds or alignment

     section .rdata
sl_dispatch_records:

%macro STMT_ENTRY 2
     dd %1 - sl_lg_initialized, %2 - sl_dispatch_fill ; relative atom slot and statement handler
%endmacro

%macro EXPR_ENTRY 2
     dd %1 - sl_lg_initialized + 0x80000000, %2 - sl_dispatch_fill ; expression flag and relative handler
%endmacro
%include "dispatch_list.inc"
%unmacro STMT_ENTRY 2                          ; release the record emitters
%unmacro EXPR_ENTRY 2                          ; release the record emitters
     dd -1                                     ; end of handler records
sl_dispatch_operators:

%macro OP_BIT 1
     dd %1 - sl_lg_initialized                 ; relative slot for this operator atom
%endmacro
     OP_BIT atom_sl_plus                       ; + operator
     OP_BIT atom_sl_minus_sym                  ; - operator
     OP_BIT atom_sl_mul_sym                    ; * operator
     OP_BIT atom_sl_div_sym                    ; / operator
     OP_BIT atom_sl_mod_sym                    ; % operator
     OP_BIT atom_sl_concat                     ; _ concat operator
     OP_BIT atom_sl_eq_sym                     ; = operator
     OP_BIT atom_sl_lt_sym                     ; < operator
     OP_BIT atom_sl_gt_sym                     ; > operator
     OP_BIT atom_sl_gte_sym                    ; >= operator
     OP_BIT atom_sl_lte_sym                    ; <= operator
     OP_BIT atom_sl_neq_sym1                   ; <> operator
     OP_BIT atom_sl_neq_sym2                   ; != operator
     OP_BIT atom_sl_shl_sym                    ; << operator
     OP_BIT atom_sl_shr_sym                    ; >> operator
     OP_BIT atom_sl_and_sym                    ; & bitwise AND operator
     OP_BIT atom_sl_or_sym                     ; | bitwise OR operator
     OP_BIT atom_sl_xor_sym                    ; ^ bitwise XOR operator
     OP_BIT atom_sl_andand                     ; && logical AND operator
     OP_BIT atom_sl_oror                       ; || logical OR operator
%unmacro OP_BIT 1                              ; release the operator record emitter
     dd -1                                     ; end of operator records
     section .text
%undef SDF_HANDLER_REL_OFF                     ; relative-handler record offset
