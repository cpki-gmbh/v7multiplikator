; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define ON_TOKEN_TEMP_PTR_OFF 40               ; rsp offset for 5th arg, &temperature
%define ON_TOKEN_TOPP_PTR_OFF 48               ; rsp offset for 6th arg, &topp
; --- include all sub-files in dependency order ---
%include "sectorlogo_const.inc"
%include "sectorlogo_data.inc"
     section .text
%include "windows_escape_string.inc"
%include "sectorlogo_lisp_core.inc"
%include "sectorlogo_intern.inc"
%include "sectorlogo_valstr.inc"
%include "sectorlogo_equal.inc"
%include "sectorlogo_gc.inc"
%include "sectorlogo_eval_macros.inc"
%include "sectorlogo_float.inc"
%include "sectorlogo_reader.inc"
%include "sectorlogo_print.inc"
%include "sectorlogo_fn.inc"
%include "sectorlogo_v7prim.inc"
%include "sectorlogo_string.inc"
%include "sectorlogo_str_tokens.inc"
%include "sectorlogo_split_ngrams.inc"
%include "sectorlogo_str_int_to_char.inc"
%include "sectorlogo_str_char_to_int.inc"
%include "sectorlogo_console_getchar.inc"
%include "sectorlogo_console_24bit.inc"
%include "sectorlogo_do_vq.inc"
%include "sectorlogo_ffi.inc"
%include "sectorlogo_sort.inc"
%include "sectorlogo_eval.inc"
intern_builtins:
     push rsi                                  ; save non-volatile, written at the empty-string hash setup and then by every INTERN_ONE row, ~138 times, restored by the pop at :244
     push rdi                                  ; save non-volatile: the four clear blocks below rewrite it, and the push/pop pair at :57-58/:69-70 runs after all four, so it restores the overwritten value rather than the caller's
     sub rsp, 40                               ; shadow + parity pad for the calls below, 2 pushes + 40: 8 - 16 - 40 = 0 mod 16; the hash pair at :61-62 adds 2 more pushes, restoring 0

%macro SL_WIPE_DWORDS 2
     lea rdi, [rel %1]                         ; rdi = destination base
     mov ecx, %2 / 4                           ; ecx = dword count to zero-initialize, every count here is a multiple of 4
     rep stosd                                 ; clear the block, fill comes from the hoisted zero
%endmacro
     xor eax, eax                              ; zero fill/source for the whole reset region
     SL_WIPE_DWORDS sl_atom_start_bits, (2 * SL_ATOM_BITMAP_BYTES) ; clear the START index + liveness array, contiguous by declaration, one ERMS pass
     mov qword [rel sl_atom_freelist], rax     ; no reclaimed atom blocks + no restart point yet, adjacent dwords
     mov dword [rel sl_gc_freelist], eax       ; no reclaimed cons cells, fresh cons area
     mov qword [rel sl_gc_want_atoms], rax     ; the atom half is off + no request has spent its collect, adjacent dwords
     SL_WIPE_DWORDS sl_intern_filter, ((SL_INTERN_FILTER_BITS / 8) + SL_INTERN_CACHE_BYTES) ; Clear the Bloom filter and adjacent cache for the fresh heap
; the FFI memo table is gone: module ownership lives in the name-keyed
; owner registry, which survives re-initialization by design, one
; loader reference per distinct external DLL for the process lifetime
     mov word [r13 + SL_ATOM_TYPE_OFF], 0      ; NIL type byte + NUL content in one store, Number NIL reads this as 0
     mov word [r13 + SL_EMPTY_ATOM_OFF], SL_TYPE_STRING ; empty-string type byte + NUL content in one store
     mov dword [rel sl_atom_free], SL_ATOM_INIT_END ; first free offset = 16, granule-aligned
     mov dword [rel sl_atom_start_bits], 3     ; granule 0 = NIL sentinel + granule 1 = empty-string atom, bits 0+1 in one pure write
     push rbx                                  ; save non-volatile, StringHash overwrites ebx; klen unused here
     push rdi                                  ; parity pad only: keeps the two hash calls at 0 mod 16, discarded after
     lea rsi, [r13 + SL_EMPTY_ATOM_OFF]        ; rsi = type_byte + NUL content, 2 bytes to hash
     call StringHash                           ; edi = h1, ebx = klen, 2
     mov r8d, ebx                              ; r8d = known klen, the h2 absorb loop is count-bounded
     call aes_string_hash                      ; rax = h2,""
     mov rdx, rax                              ; park h2, the mov eax,edi below overwrites rax
     mov eax, edi                              ; eax = h1
     and eax, SL_INTERN_FILTER_MASK            ; bloom bit index, h1
     bts dword [rel sl_intern_filter], eax     ; set h1 bit
     and rdx, SL_INTERN_FILTER_MASK            ; bloom bit index, h2
     bts qword [rel sl_intern_filter], rdx     ; set h2 bit
     pop rdi                                   ; discard the parity pad
     pop rbx                                   ; restore caller's non-volatile, interp holds script ptr in rbx
; --- macro to intern one .data string and store result ---

%macro INTERN_ONE 2
     lea rsi, [rel %1]                         ; rsi = .data string, Intern source; Intern preserves caller rsi/rdi
     call Intern                               ; eax = interned atom offset
     mov dword [rel %2], eax                   ; store to BSS atom handle
%endmacro

%macro SL_PRIME_READER 1
     lea rax, [rel %1]                         ; rax = source buffer address
     mov [rel sl_src_mem_ptr], rax             ; point the reader at the buffer
     mov dword [rel sl_list_depth], 0          ; fresh parse: reset lexer nesting
     mov dword [rel sl_last_tok_operand], SL_TOK_NOT_OPERAND ; first token is keyword-head position
     mov dword [rel sl_parse_depth], 0         ; fresh parse: reset the recursion guard
     call GetChar                              ; al = first char of the script
     mov byte [rel sl_g_look], al              ; prime the lookahead
%endmacro

%macro SL_RESET_CALL_STATE 1
     mov qword [rel sl_fn_has_returned], 0     ; clear fn_has_returned + fn_return_val, adjacent BSS dwords
     mov qword [rel sl_stop_requested], 0      ; clear stop_requested + continue_requested, adjacent BSS dwords
     mov dword [rel sl_terminate_requested], 0 ; clear TERMINATE, never leaks across calls
     mov %1, 0                                 ; caller's 4th slot, see header
     mov dword [rel sl_local_count], 0         ; clear LOCAL set, fresh scope
     mov dword [rel sl_call_depth], 0          ; clear recursion depth, a caught FATAL mid-recursion leaks its inc/dec imbalance
     mov qword [rel sl_trace_indent], 0        ; clear trace indent + strict-name mode + a caught-FATAL leak class, two adjacent dwords
%endmacro

%macro SL_COPY_DB_SCRIPT 2
     mov rcx, [rel sl_lg_db_len]               ; rcx = DB entry byte length, includes the NUL
     cmp rcx, SL_SCRIPT_BUF_SIZE - 1           ; fits in buffer?, the imm32 sign-extends to 64 bits and SL_SCRIPT_BUF_SIZE-1 is positive, so the REX.W compare is exact
     ja %2                                     ; no: FATAL, never clamp
     lea rdi, [rel %1]                         ; rdi = destination buffer
     rep movsb                                 ; ERMS copy, rsi/rdi auto-advance; rcx=0 is a no-op
     mov byte [rdi], 0                         ; NUL-terminate
%endmacro
; core Logo builtins
     INTERN_ONE str_sl_make, atom_sl_make      ; MAKE
     INTERN_ONE str_sl_repeat, atom_sl_repeat  ; REPEAT
     INTERN_ONE str_sl_printraw, atom_sl_printraw ; PRINTRAW, PRINT is a stdlib fn now
     INTERN_ONE str_sl_first, atom_sl_first    ; FIRST
     INTERN_ONE str_sl_butfirst, atom_sl_butfirst ; BUTFIRST
     INTERN_ONE str_sl_fput, atom_sl_fput      ; Prepend
     INTERN_ONE str_sl_if, atom_sl_if          ; IF
     INTERN_ONE str_sl_ifelse, atom_sl_ifelse  ; IFELSE
     INTERN_ONE str_sl_stop, atom_sl_stop      ; STOP
     INTERN_ONE str_sl_continue, atom_sl_continue ; CONTINUE
     INTERN_ONE str_sl_terminate, atom_sl_terminate ; Exit
     INTERN_ONE str_sl_t, atom_sl_t            ; TRUE atom
     INTERN_ONE str_sl_false, atom_sl_false    ; FALSE atom
     INTERN_ONE str_sl_lparen, atom_sl_lparen  ; ( grouping-open token (skipped by the reader, needed for source display)
     INTERN_ONE str_sl_rparen, atom_sl_rparen  ; ) grouping-close token (skipped by the reader, needed for source display)
; metaprogramming & homoiconic execution
     INTERN_ONE str_sl_run, atom_sl_run        ; RUN
; function definition & return
     INTERN_ONE str_sl_def_fn, atom_sl_def_fn  ; Define
     INTERN_ONE str_sl_def_exe_fn, atom_sl_def_exe_fn ; DefineCommand
     INTERN_ONE str_sl_output, atom_sl_output  ; OUTPUT
     INTERN_ONE str_sl_fn_tag, atom_sl_fn_tag  ; __FN__
     INTERN_ONE str_sl_xfn_tag, atom_sl_xfn_tag ; __XF__
; arithmetic / math
     INTERN_ONE str_sl_plus, atom_sl_plus      ; +
     INTERN_ONE str_sl_minus_sym, atom_sl_minus_sym ; -, symbol
     INTERN_ONE str_sl_mul_sym, atom_sl_mul_sym ; *
     INTERN_ONE str_sl_div_sym, atom_sl_div_sym ; /
     INTERN_ONE str_sl_mod_sym, atom_sl_mod_sym ; %
     INTERN_ONE str_sl_power, atom_sl_power    ; POWER
     INTERN_ONE str_sl_sqrt, atom_sl_sqrt      ; SquareRoot
     INTERN_ONE str_sl_abs, atom_sl_abs        ; AbsoluteValue
     INTERN_ONE str_sl_minus, atom_sl_minus    ; Negate
     INTERN_ONE str_sl_max, atom_sl_max        ; Maximum
     INTERN_ONE str_sl_min, atom_sl_min        ; Minimum
     INTERN_ONE str_sl_random, atom_sl_random  ; RANDOM
     INTERN_ONE str_sl_real_random, atom_sl_real_random ; REAL-RANDOM
; float primitives
     INTERN_ONE str_sl_i2f, atom_sl_i2f        ; AsFloat
     INTERN_ONE str_sl_f2i, atom_sl_f2i        ; AsInteger
     INTERN_ONE str_sl_floor, atom_sl_floor    ; FLOOR
     INTERN_ONE str_sl_fround, atom_sl_fround  ; Round
     INTERN_ONE str_sl_floatp, atom_sl_floatp  ; IsFloat
     INTERN_ONE str_sl_flog2, atom_sl_flog2    ; LogBaseTwo
     INTERN_ONE str_sl_fexp2, atom_sl_fexp2    ; PowerOfTwo
     INTERN_ONE str_sl_fln, atom_sl_fln        ; NaturalLog
     INTERN_ONE str_sl_fmod, atom_sl_fmod      ; FloatRemainder
; bitwise
     INTERN_ONE str_sl_and_sym, atom_sl_and_sym ; &
     INTERN_ONE str_sl_or_sym, atom_sl_or_sym  ; |
     INTERN_ONE str_sl_xor_sym, atom_sl_xor_sym ; ^
     INTERN_ONE str_sl_not_sym, atom_sl_not_sym ; ~
     INTERN_ONE str_sl_shl_sym, atom_sl_shl_sym ; <<
     INTERN_ONE str_sl_shr_sym, atom_sl_shr_sym ; >>
     INTERN_ONE str_sl_concat, atom_sl_concat  ; _, string concat
; relational / boolean
     INTERN_ONE str_sl_eq_sym, atom_sl_eq_sym  ; =
     INTERN_ONE str_sl_neq_sym1, atom_sl_neq_sym1 ; <>
     INTERN_ONE str_sl_neq_sym2, atom_sl_neq_sym2 ; !=
     INTERN_ONE str_sl_gt_sym, atom_sl_gt_sym  ; >
     INTERN_ONE str_sl_lt_sym, atom_sl_lt_sym  ; <
     INTERN_ONE str_sl_gte_sym, atom_sl_gte_sym ; >=
     INTERN_ONE str_sl_lte_sym, atom_sl_lte_sym ; <=
     INTERN_ONE str_sl_andand, atom_sl_andand  ; &&
     INTERN_ONE str_sl_oror, atom_sl_oror      ; ||
     INTERN_ONE str_sl_bang, atom_sl_bang      ; !
     INTERN_ONE str_sl_zerop, atom_sl_zerop    ; IsZero
     INTERN_ONE str_sl_numberp, atom_sl_numberp ; IsNumber
     INTERN_ONE str_sl_listp, atom_sl_listp    ; IsList
     INTERN_ONE str_sl_isstringatom, atom_sl_isstringatom ; IsStringAtom
     INTERN_ONE str_sl_iswordatom, atom_sl_iswordatom ; IsWordAtom
; string operations
     INTERN_ONE str_sl_substr, atom_sl_substr  ; Part
     INTERN_ONE str_sl_find, atom_sl_find      ; Search
     INTERN_ONE str_sl_strlower, atom_sl_strlower ; Lowercase
     INTERN_ONE str_sl_strupper, atom_sl_strupper ; Uppercase
     INTERN_ONE str_sl_strreplace, atom_sl_strreplace ; Replace
     INTERN_ONE str_sl_strtonum, atom_sl_strtonum ; TextToNumber
     INTERN_ONE str_sl_strlen, atom_sl_strlen  ; TextLength
     INTERN_ONE str_sl_explode, atom_sl_explode ; Split
     INTERN_ONE str_sl_str_tokens, atom_sl_str_tokens ; Words
     INTERN_ONE str_sl_split_ngrams, atom_sl_split_ngrams ; NGrams
     INTERN_ONE str_sl_lindex, atom_sl_lindex  ; Item
     INTERN_ONE str_sl_local, atom_sl_local    ; LOCAL
     INTERN_ONE str_sl_sort, atom_sl_sort      ; SORT
     INTERN_ONE str_sl_reverse, atom_sl_reverse ; REVERSE
     INTERN_ONE str_sl_hash_float, atom_sl_hash_float ; HashNumber
     INTERN_ONE str_sl_length, atom_sl_length  ; LENGTH
     INTERN_ONE str_sl_lset, atom_sl_lset      ; SetItem
     INTERN_ONE str_sl_setcar, atom_sl_setcar  ; SETCAR
     INTERN_ONE str_sl_setcdr, atom_sl_setcdr  ; SETCDR
     INTERN_ONE str_sl_samevalue, atom_sl_samevalue ; SAMEVALUE
     INTERN_ONE str_sl_asword, atom_sl_asword  ; ASWORD
     INTERN_ONE str_sl_lfind, atom_sl_lfind    ; FindItem
     INTERN_ONE str_sl_lfindge, atom_sl_lfindge ; FindFirstAtLeast
     INTERN_ONE str_sl_lfindle, atom_sl_lfindle ; FindFirstAtMost
     INTERN_ONE str_sl_int_to_char, atom_sl_int_to_char ; Character
     INTERN_ONE str_sl_char_to_int, atom_sl_char_to_int ; Number
     INTERN_ONE str_sl_do_vq, atom_sl_do_vq    ; NearestMatch
; FFI, CallLibrary
     INTERN_ONE str_sl_c_call, atom_sl_c_call  ; CallLibrary
     INTERN_ONE str_sl_type_int, atom_sl_type_int ; Integer, "integer", quoted lowercase in descriptors
     INTERN_ONE str_sl_type_ptr, atom_sl_type_ptr ; Pointer, "pointer"
     INTERN_ONE str_sl_type_str, atom_sl_type_str ; Text, "text"
     INTERN_ONE str_sl_type_void, atom_sl_type_void ; Void, "void"
; console primitives
     INTERN_ONE str_sl_getchar, atom_sl_getchar ; ReadKey
     INTERN_ONE str_sl_console_24bit, atom_sl_console_24bit ; EnableColors
; trace primitive
     INTERN_ONE str_sl_enabletrace, atom_sl_enabletrace ; EnableTrace
     INTERN_ONE str_sl_enablestrictnames, atom_sl_enablestrictnames ; EnableStrictNames
; V7 host primitives
     INTERN_ONE str_sl_v7_encode_word, atom_sl_v7_encode_word ; EncodeWord
     INTERN_ONE str_sl_v7_db_get, atom_sl_v7_db_get ; DatabaseGet
     INTERN_ONE str_sl_v7_db_put, atom_sl_v7_db_put ; DatabaseSet
     INTERN_ONE str_sl_v7_pos, atom_sl_v7_pos  ; Position
     INTERN_ONE str_sl_v7_get_token_pos, atom_sl_v7_get_token_pos ; GetTokenPosition
     INTERN_ONE str_sl_v7_get_token, atom_sl_v7_get_token ; GetToken
     INTERN_ONE str_sl_v7_vocab_size, atom_sl_v7_vocab_size ; VocabularySize
     INTERN_ONE str_sl_v7_dim, atom_sl_v7_dim  ; ModelDimension
     INTERN_ONE str_sl_v7_layers, atom_sl_v7_layers ; LayerCount
     INTERN_ONE str_sl_v7_eos, atom_sl_v7_eos  ; EndToken
     INTERN_ONE str_sl_v7_temp, atom_sl_v7_temp ; SetTemperature
     INTERN_ONE str_sl_v7_topp, atom_sl_v7_topp ; SetNucleus
; V7 stream rewrite primitives
     INTERN_ONE str_sl_v7_stream, atom_sl_v7_stream ; GetStream
     INTERN_ONE str_sl_v7_str_len, atom_sl_v7_str_len ; StreamLength
     INTERN_ONE str_sl_v7_set_stream, atom_sl_v7_set_stream ; SetStream
     INTERN_ONE str_sl_v7_set_token, atom_sl_v7_set_token ; SetOutputToken
     INTERN_ONE str_sl_v7_reject, atom_sl_v7_reject ; Reject
     INTERN_ONE str_sl_v7_halt, atom_sl_v7_halt ; Halt
; hook-scoped variables
     INTERN_ONE str_sl_lg_tok, atom_sl_lg_tok  ; :TokenNumber, the interned string is "tokennumber", :TOK is a DEAD name: it lowercases to "tok", a different atom with no binding, so it reads NIL forever
     INTERN_ONE str_sl_lg_piece, atom_sl_lg_piece ; :PIECE
     add rsp, 40                               ; drop the frame FIRST, it sits BELOW both saved slots, push rsi then push rdi at :35-36 precede this sub, so the saved rdi is the lower of the two and the saved rsi is 8 above it
     pop rdi                                   ; restore the caller's rdi, lowest saved slot
     pop rsi                                   ; restore the caller's rsi, highest saved slot, now the top of the frame
     ret                                       ; return to caller
lg_init:
     SF_START lg_init, rsi, rdi, r12, r13      ; prologue: rbx unused in body; r12 IS written below, P-hold across the outer Cons and r13 IS written below, heap-base pin, so both must be saved
     SF_COMMIT                                 ; commit frame
; allocate pinned heap: one committed block for atom area + cons area
     mov ecx, SL_HEAP_TOTAL_SIZE               ; arg1: atom + cons area bytes
     call mdb_alloc_mem                        ; rax = heap base, always valid
     mov [rel sl_lg_heap_base], rax            ; store heap base pointer
     mov r13, rax                              ; pin heap base in r13
     xor eax, eax                              ; zero source for the cached-prog resets
     mov dword [rel sl_cons_free], eax         ; reset cons cursor
     mov [rel stream_remove_prog], rax         ; cached prog belongs to the PREVIOUS arena, GC root 9b, re-parse on next use
     lea rax, [rel sl_xfn_escape_buf]          ; the DefineCommand escape arena base
     mov [rel sl_xfn_arena_cursor], rax        ; seed the cursor: BSS zero would make the first %s escape write to address 0
; intern all builtin symbol names
     call intern_builtins                      ; intern all symbol names
     call sl_dispatch_fill                     ; direct-threaded dispatch tables
; --- try loading script from MicroDB ---
; ht_find,rcx=key, rdx=&out_len -> rax = script data pointer
; The engine DB (g_ht, loaded by db_load) is the ONE database; hosts preload
; a sidecar here via v7_db_add. No gate: SectorLogo always reads the engine DB.
     lea rcx, [rel str_sl_db_key_sidecar]      ; key = "logo:sidecar"
     lea rdx, [rel sl_lg_db_len]               ; &out_len
     call ht_find                              ; rax = script data pointer
     test rax, rax                             ; NULL = key not found in LIVE DB?
     jz .sidecar_key_absent                    ; yes: no sidecar script, hook stays off, tuner-only DBs
     mov dword [rel sl_hook_enabled], 1        ; DATA-DRIVEN hook activation: the sidecar script is IN the database, so the hook is on -- the host's only lever is ht_add (v7_db_add), never a call
     mov rcx, [rel sl_lg_db_len]               ; initial parsed script length
     mov [rel sl_sidecar_cache_len], rcx       ; first refresh recognizes the initial snapshot
; copy once into the persistent snapshot, then parse that SAME copy
     mov rsi, rax                              ; rsi = MicroDB script text
     SL_COPY_DB_SCRIPT sl_sidecar_snap, .sidecar_oversize ; snapshot the source: refreshes compare against THIS copy
.parse:
     SL_PRIME_READER sl_sidecar_snap           ; persistent immutable parse source
     call ReadLogoStream                       ; eax = parsed cons list
     mov [rel sl_lg_callback_prog], eax        ; store parsed program
; --- autoexec stdlib: 2-tier load, DB key -> file, then notice
.autoexec_load:
; ---- tier 1: MicroDB key "__sectorlogo__autoexec" ----
     lea rcx, [rel str_sl_db_key_autoexec]     ; key = "__sectorlogo__autoexec"
     lea rdx, [rel sl_lg_db_len]               ; &out_len
     call ht_find                              ; rax = data ptr or NULL
     test rax, rax                             ; key missing?
     jz .autoexec_try_file                     ; yes: next tier
     mov rsi, rax                              ; rsi = DB blob
     SL_COPY_DB_SCRIPT sl_lg_script_buf, .autoexec_oversize ; reuse shared parser staging area
     lea rsi, [rel sl_lg_script_buf]           ; source for the common reader setup
     WRITE_STR_NL sl_msg_ae_from_db            ; "loaded from database"
     jmp .autoexec_parse                       ; go parse
.autoexec_try_file:
; ---- tier 2: file "autoexec.logo", v7s_readfile; empty on error ----
     lea rcx, [rel str_sl_autoexec_path]       ; path = "autoexec.logo"
     call v7s_readfile                         ; rax = v7_execute_buf, empty on error
     cmp byte [rax], 0                         ; file missing or empty?
     je .autoexec_none                         ; yes: tier 3 notice
     mov rsi, rax                              ; rsi = file contents
     STR_LEN rsi, ecx                          ; ecx = strlen, overwrites rax/rcx/rdi
     cmp ecx, SL_SCRIPT_BUF_SIZE               ; room for text + NUL?
     jae .autoexec_oversize                    ; yes -> FATAL, never clamp
     WRITE_STR_NL sl_msg_ae_from_file          ; rsi remains the file text, WRITE_STR_NL preserves rsi
.autoexec_parse:
; DB source is staged in sl_lg_script_buf; file source is already NUL-terminated
     mov [rel sl_src_mem_ptr], rsi             ; parse the chosen source directly
     mov dword [rel sl_list_depth], 0          ; fresh parse: reset the lexer nesting depth, pairwise counterpart of the ']' FATAL
     mov dword [rel sl_last_tok_operand], SL_TOK_NOT_OPERAND ; fresh parse: prime the operand tracker
     mov dword [rel sl_parse_depth], 0         ; fresh parse: reset the recursion guard
     call GetChar                              ; al = first char of the autoexec text
     mov byte [rel sl_g_look], al              ; prime the lookahead
     call ReadLogoStream                       ; eax = parsed autoexec forms
     mov [rel sl_lg_autoexec_prog], eax        ; store, also GC root 5
     SL_RESET_CALL_STATE dword [rel sl_trace_level] ; fresh call state; 4th slot = trace level, autoexec eval must start clean
     xor eax, eax                              ; eax = NIL, P's car
     xor edx, edx                              ; edx = NIL, P's cdr
     call Cons                                 ; eax = tagged, NIL.NIL inner pair P
     mov r12d, eax                             ; r12d = P, held across the outer Cons; eax still holds P as the outer car
     xor edx, edx                              ; edx = NIL, outer cdr, the binding chain starts empty
     call Cons                                 ; eax = tagged, P.NIL sentinel outer cell
     mov [rel sl_lg_base_env], eax             ; base env = SENTINEL outer cell, GC root 6, live BEFORE the stdlib eval, because Defines insert through it
     mov esi, [rel sl_lg_autoexec_prog]        ; esi = autoexec forms, stream
     mov edx, [rel sl_lg_base_env]             ; env = the sentinel, stdlib Defines insert into it at eval time
     call EvalLogoStream                       ; edx = resulting base env: the autoexec's top-level Makes chained ABOVE the sentinel, storing it back keeps those Makes reachable, Print reads :crlf while the sentinel stays in the chain, so later DEF_BIND_TAIL inserts remain visible to every chain that terminates in it
     mov [rel sl_lg_base_env], edx             ; base env =, autoexec-makes.sentinel, GC root 6; the sentinel cell itself is never replaced
     jmp .init_done                            ; finish init
.autoexec_none:
; ---- tier 3: no stdlib anywhere, informational notice, keep going ----
     WRITE_STR_NL sl_msg_ae_none               ; "no stdlib autoexec found - ..."
     mov dword [rel sl_lg_autoexec_prog], 0    ; mark: no stdlib loaded
     xor eax, eax                              ; eax = NIL, P's car
     xor edx, edx                              ; edx = NIL, P's cdr
     call Cons                                 ; eax = tagged, NIL.NIL inner pair P
     mov r12d, eax                             ; r12d = P; eax still holds P as the outer car
     xor edx, edx                              ; edx = NIL, outer cdr
     call Cons                                 ; eax = tagged, P.NIL sentinel outer cell
     mov [rel sl_lg_base_env], eax             ; base env = sentinel outer cell, GC root 6
.init_done:
     SF_END rsi, rdi, r12, r13                 ; epilogue + restore non-volatiles, SF_END emits the ret
.autoexec_oversize:
     FATAL "SectorLogo: autoexec stdlib exceeds the 1 MiB script buffer"
.sidecar_key_absent:
; DB has no "logo:sidecar" key: the hook stays off, no sidecar to run.
; Set callback_prog = 0 so on_token's eval path is inert.
     mov dword [rel sl_lg_callback_prog], 0    ; neutralize the per-token hook, no sidecar
     mov qword [rel sl_sidecar_cache_len], -1  ; force parse if the key appears later
     jmp .autoexec_load                        ; continue with stdlib init
.sidecar_oversize:
     FATAL "SectorLogo: sidecar script exceeds the 1 MiB script buffer"
     global v7s_set_caller_hmodule             ; export entry point
v7s_set_caller_hmodule:
     mov [rel sl_caller_hmodule], rcx          ; store host module handle
     ret                                       ; return to caller
     global v7s_sectorlogo_on_token            ; export entry point
v7s_sectorlogo_on_token:
     ENGINE_GATE_RAW 1                         ; dead-check + landing zone: a script FATAL unwinds
     mov rax, [rsp + ON_TOKEN_TEMP_PTR_OFF]    ; 5th arg: &temperature
     mov r10, [rsp + ON_TOKEN_TOPP_PTR_OFF]    ; 6th arg: &topp
     mov [rel sl_hook_temp_ptr], rax           ; &temperature
     mov [rel sl_hook_topp_ptr], r10           ; &topp
     mov [rel sl_hook_ctx], rcx                ; InferenceContext*
     mov [rel sl_hook_tok_in], edx             ; input token id
     mov [rel sl_hook_pos], r8d                ; current position
     mov [rel sl_hook_out_tok_ptr], r9         ; &out_token
; initialize tokens_ptr from InferenceContext for V7-GET-TOKEN
     mov rax, [rcx + SL_CTX_OUT_TOKENS]        ; out_tokens int*
     mov [rel sl_hook_tokens_ptr], rax         ; store for V7-GET-TOKEN
     SL_STK_ENTER                              ; switch to the private Logo stack, anchors the GC scan ceiling
; save all non-volatile registers used after the switch, now on the private stack
     push rbx                                  ; save non-volatile
     push rbp                                  ; save non-volatile
     push rsi                                  ; save non-volatile
     push rdi                                  ; save non-volatile
     push r12                                  ; save non-volatile
     push r13                                  ; will be repurposed as heap base
     push r14                                  ; save non-volatile
     push r15                                  ; save non-volatile
     sub rsp, 40                               ; shadow space + alignment
; default: pass the token through unchanged, RELOAD from BSS, every volatile argument register was destroyed in the stack switch
     mov eax, [rel sl_hook_tok_in]             ; eax = input token id
     mov rcx, [rel sl_hook_out_tok_ptr]        ; rcx = &out_token
     mov dword [rcx], eax                      ; *out_token = input token
; lazy one-time init
     cmp dword [rel sl_lg_initialized], 0      ; already initialized?
     je .tok_first_init                        ; rare: one-time init, out of line
; common path falls through to decode, token id lives in sl_hook_tok_in, BSS reload below
.tok_decode:
     cmp dword [rel sl_lg_callback_prog], 0    ; no sidecar set?, memory-operand compare, the field is reloaded independently into esi at the EvalLogoStream setup, so nothing here needs it in a register
     jz .tok_no_callback                       ; yes: keep the pass-through token and accept, no decode, no intern, no env, no eval
     SL_RESET_CALL_STATE qword [rel sl_action_code] ; hook path: ACCEPT unless a primitive sets an action. HERE, not at entry: lg_init is not a passive allocator, on the no-sidecar path it parses autoexec.logo and runs the whole standard library through EvalLogoStream, so a reset placed above the init branch would be immediately re-dirtied and the stdlib's leftovers would flow into the first token's evaluation. Both predecessors, the fall-through and .tok_first_init's jmp reach this label, so it runs exactly once, after init has settled. Below the sidecar test as well: .tok_no_callback returns a hard-coded 0 and reads none of the eight fields, so on that path all eight stores are dead
     mov r13, [rel sl_lg_heap_base]            ; pin the heap base for all calls
; decode the candidate token's normalized text
     mov edx, [rel sl_hook_tok_in]             ; edx = token id, BSS reload, the decode's edx contract must not rely on register survival: lg_init leaves parsed-script garbage in edx on the first-init path, and SL_STK_ENTER's alloc path overwrites every volatile
     call lg_decode_token_real                 ; eax = interned atom, PIECE
     mov ebx, eax                              ; ebx = PIECE atom
; build the per-call environment:, ,PIECE.piece_atom, TOK.tok_num
     mov eax, [rel atom_sl_lg_piece]           ; eax = PIECE atom, car
     mov edx, ebx                              ; edx = piece value, cdr
     call Cons                                 ; eax =, PIECE.piece
     mov edx, [rel sl_lg_base_env]             ; edx = base env, list terminator
     call Cons                                 ; eax =, ,PIECE.piece.base_env
     mov r12d, eax                             ; r12d = env head, parked in a non-volatile on purpose: MakeNum below is a AND/OR/ret LEAF and Cons at :684 never touches r12, keep it parked here, NOT in eax/edx
     mov eax, [rel sl_hook_tok_in]             ; eax = raw token id
     call MakeNum                              ; eax = tagged NUMBER
     mov edx, eax                              ; edx = tok value, cdr of pair
     mov eax, [rel atom_sl_lg_tok]             ; eax = TOK atom, car
     call Cons                                 ; eax =, TOK.tokval
     mov edx, r12d                             ; edx = rest of env, cdr
     call Cons                                 ; eax = full env alist
; execute the callback script
     mov esi, [rel sl_lg_callback_prog]        ; stream = parsed program
     mov edx, eax                              ; env = full env alist, eax survives the transparent TRACE macros
     call EvalLogoStream                       ; edx = updated env
     mov eax, [rel sl_action_code]             ; action code, 0=ACCEPT, 1=RESAMPLE, 2=EOS, 3=REWRITE
     jmp .tok_unwind                           ; the action stands as the script set it, fall-through would run the no-callback ACCEPT zeroing below
.tok_no_callback:
     xor eax, eax                              ; ACCEPT
.tok_unwind:
     add rsp, 40                               ; restore shadow space
     pop r15                                   ; restore non-volatile
     pop r14                                   ; restore non-volatile
     pop r13                                   ; restore non-volatile
     pop r12                                   ; restore non-volatile
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbp                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     xor r10d, r10d                            ; zero source for the three hook-pointer resets, keeps eax as the action code
     mov [rel sl_hook_out_tok_ptr], r10        ; stale &out_token removed
     mov [rel sl_hook_tokens_ptr], r10         ; stale token-history removed
     mov [rel sl_hook_ctx], r10                ; stale InferenceContext* removed
     ENGINE_LEAVE                              ; park the landing zone, normal path
; stack switch is free, gate outermost, leave outermost
     SL_STK_LEAVE                              ; switch back to the C stack, marker on top: body rsp is at the post-ENTER parity
     ret                                       ; return action code in eax
.tok_first_init:
     call lg_init                              ; first call: init heap + parse script
     mov dword [rel sl_lg_initialized], 1      ; mark as initialized
     jmp .tok_decode                           ; rejoin the common path, NO arg reload needed: .tok_decode re-reads sl_hook_tok_in from BSS at the two points that consume it, and the sidecar path overwrites edx at the `mov edx, ebx` in the per-call env build above before any use, a redundant volatile reload here is what invites a future edit to trust it
; ============================================================================
; v7s_sectorlogo_interp,rcx = script string -> eax = result, tagged value

; Standalone script evaluation entry point. Takes a NUL-terminated
; Logo script string, parses and evaluates it, and returns the value
; of the last expression. Useful for testing and validation without
; the on-token hook infrastructure.
; ============================================================================
     global v7s_sectorlogo_interp              ; export entry point
v7s_sectorlogo_interp:
     ENGINE_GATE_RAW 1                         ; dead-check + landing zone: a script FATAL unwinds
; to the C caller as 0 + dead flag instead of terminating the host
; gate runs BEFORE the stack switch: the zone must snapshot the C stack
     mov [rel sl_interp_park], rcx             ; park script pointer in BSS BEFORE the switch, SL_STK_ENTER's alloc path overwrites every volatile; parking in rbx would destroy the CALLER's rbx before it is saved, Win64 ABI violation, on_token's sl_hook_* pattern instead
     SL_STK_ENTER                              ; switch to the private Logo stack, anchors the GC scan ceiling
     push rbx                                  ; save non-volatile, the CALLER's rbx, park slot already holds the script pointer
     push rsi                                  ; save non-volatile, overwritten by script copy
     push rdi                                  ; save non-volatile, overwritten by script copy
     push r12                                  ; save non-volatile
     push r13                                  ; save non-volatile
     push r14                                  ; save non-volatile
     push r15                                  ; save non-volatile
     sub rsp, 32                               ; 32 shadow; 7 pushes, 56 + 32 + retaddr,8 = 96 -> RSP%16==0
     mov rbx, [rel sl_interp_park]             ; fetch the parked script pointer, reentrancy-safe: consumed before any call that could re-enter
; lazy init, same as on_token
     cmp dword [rel sl_lg_initialized], 0      ; already initialized?
     jne .interp_ready                         ; yes: skip init
     call lg_init                              ; first call: init heap + parse script
     mov dword [rel sl_lg_initialized], 1      ; mark as initialized
.interp_ready:
     mov r13, [rel sl_lg_heap_base]            ; pin heap base
     xor r10d, r10d                            ; zero source for the three hook-pointer resets
     mov [rel sl_hook_out_tok_ptr], r10        ; the standalone world has no engine token slot
     mov [rel sl_hook_tokens_ptr], r10         ; ...no token history
     mov [rel sl_hook_ctx], r10                ; ...no context
; reset return state for this evaluation
     SL_RESET_CALL_STATE dword [rel sl_trace_level] ; standalone script: fresh trace level, see macro header
; validate the borrowed, NUL-terminated source; keep the original 1 MiB limit
     mov rsi, rbx                              ; rsi = source script, from saved rbx
     STR_LEN rsi, ecx                          ; ecx = strlen,script, overwrites rax/rcx/rdi
     cmp ecx, SL_SCRIPT_BUF_SIZE               ; room for text + NUL?
     jae .interp_oversize                      ; yes -> FATAL, never clamp
     mov [rel sl_src_mem_ptr], rbx             ; reader cursor = caller source
     mov dword [rel sl_list_depth], 0          ; fresh parse: reset lexer nesting
     mov dword [rel sl_last_tok_operand], SL_TOK_NOT_OPERAND ; first token is keyword-head position
     mov dword [rel sl_parse_depth], 0         ; fresh parse: reset the recursion guard
     call GetChar                              ; al = first char
     mov byte [rel sl_g_look], al              ; prime the lookahead
     call ReadLogoStream                       ; eax = parsed program
; the stdlib world already lives in sl_lg_base_env
     mov esi, eax                              ; stream = parsed program
     mov edx, [rel sl_lg_base_env]             ; env = shared base env
     call EvalLogoStream                       ; edx = updated env
; return the last OUTPUT value if set, otherwise NIL
     cmp dword [rel sl_fn_has_returned], 1     ; did OUTPUT execute?
     je .interp_has_return                     ; yes: return its value
     xor eax, eax                              ; no return value: NIL
     jmp .interp_done                          ; go to cleanup
.interp_has_return:
     mov eax, [rel sl_fn_return_val]           ; return the OUTPUT value
.interp_done:
; return result
     add rsp, 32                               ; restore shadow space
     pop r15                                   ; restore non-volatile
     pop r14                                   ; restore non-volatile
     pop r13                                   ; restore non-volatile
     pop r12                                   ; restore non-volatile
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     ENGINE_LEAVE                              ; park the landing zone, normal path
     SL_STK_LEAVE                              ; switch back to the C stack, marker on top: body rsp is at the post-ENTER parity
     ret                                       ; return result in eax
.interp_oversize:
     FATAL "SectorLogo: script exceeds the 1 MiB script buffer"
     global v7s_sectorlogo_eval_script         ; export: named-DB-script evaluator
%define SL_ES_PUSHBASE 56                      ; bytes of pushed non-volatiles below rbp, 7 regs x 8, hand-ended SF_END rsp recovery
%define SL_ES_FRAME 40                         ; committed frame bytes, SF_COMMIT mirror: 32 shadow + 8 alignment for 7 regs
v7s_sectorlogo_eval_script:
     ENGINE_GATE_RAW 1                         ; dead-check + landing zone, same placement rule as
     mov [rel sl_es_park], rcx                 ; park key_name in BSS BEFORE the switch, SL_STK_ENTER's alloc path overwrites every volatile; parking in rbx would destroy the CALLER's rbx before it is saved, on_token's sl_hook_* pattern instead
     SL_STK_ENTER                              ; switch to the private Logo stack, anchors the GC scan ceiling
     push rbp                                  ; save caller's frame pointer, SF_START mirror
     mov rbp, rsp                              ; set frame pointer, SF_START mirror
     push rbx                                  ; save non-volatile, the CALLER's rbx, park slot already holds key_name, SF_START mirror
     push rsi                                  ; save non-volatile, SF_START mirror
     push rdi                                  ; save non-volatile, SF_START mirror
     push r12                                  ; save non-volatile, SF_START mirror
     push r13                                  ; save non-volatile, SF_START mirror
     push r14                                  ; save non-volatile, SF_START mirror
     push r15                                  ; save non-volatile, SF_START mirror
     sub rsp, SL_ES_FRAME                      ; shadow space + alignment, SF_COMMIT mirror
     mov rbx, [rel sl_es_park]                 ; fetch the parked key_name pointer, reentrancy-safe: consumed before any call that could re-enter
; N2: lazy self-init, same pattern as on_token
     cmp dword [rel sl_lg_initialized], 0      ; Logo world ready?
     jne .es_ready                             ; yes: skip one-shot init
     call lg_init                              ; heap + autoexec stdlib one-shot
     mov dword [rel sl_lg_initialized], 1      ; mark initialized for later calls
.es_ready:
     mov r13, [rel sl_lg_heap_base]            ; N2: pin heap base for GC
     xor r10d, r10d                            ; zero source for the three hook-pointer resets
     mov [rel sl_hook_out_tok_ptr], r10        ; the standalone world has no engine token slot
     mov [rel sl_hook_tokens_ptr], r10         ; ...no token history
     mov [rel sl_hook_ctx], r10                ; ...no context
; N3: reset per-eval state flags
     SL_RESET_CALL_STATE dword [rel sl_trace_level] ; per-eval state flags, see macro header
; ht_find,key_name -> rax = script text, or NULL. The engine DB (g_ht) is the
; ONE database: db_load filled it, DatabaseGet/DatabaseSet read/write it, and
; this lookup needs no separate pointer.
     mov rcx, rbx                              ; rcx = key_name
     lea rdx, [rel sl_lg_db_len]               ; &out_len, reuse caller-thread slot
     call ht_find                              ; rax = data ptr or NULL
     test rax, rax                             ; key not found?
     jz .es_key_not_found                      ; yes -> return -1
     mov r14, rax                              ; r14 = script text, survives strcmp
; --- key match: r15 = cache-slot address for this key ---
     lea rsi, [rel stream_remove_script_key]   ; "on_stream_evict"
     mov rdi, rbx                              ; key_name
     call .es_streq                            ; eax = 1 if equal
     test eax, eax                             ; is this the eviction-callback script key?
     jz .es_key_not_found                      ; unknown key -> -1
     lea r15, [rel stream_remove_prog]         ; cache slot
.es_cache_check:
     mov r12d, [r15]                           ; r12d = cached prog, or 0
     test r12d, r12d                           ; slot populated?
     jz .es_parse                              ; no: parse fresh
     mov eax, [rel sl_lg_db_len]               ; eax = the DB text's current length
     cmp eax, [rel stream_remove_script_len]   ; same text the cache was parsed from?
     je .es_eval_cached                        ; unchanged: serve the cached program
.es_parse:
; --- parse: copy text to lg_script_buf, parse, cache the prog + its length ---
     mov rsi, r14                              ; rsi = script text
     SL_COPY_DB_SCRIPT sl_lg_script_buf, .es_oversize ; cached-script text -> shared buffer
     SL_PRIME_READER sl_lg_script_buf          ; reader -> shared buffer, lexer state reset, lookahead primed
     call ReadLogoStream                       ; eax = parsed cons list
     mov [r15], eax                            ; cache parsed prog in slot
     mov eax, [rel sl_lg_db_len]               ; eax = the parsed text's length
     mov [rel stream_remove_script_len], eax   ; snapshot: the next call re-parses only when it changes
     mov r12d, [r15]                           ; r12d = prog for the eval below
.es_eval_cached:
     mov edx, [rel sl_lg_base_env]             ; env = base env, the eviction callback runs unbound
.es_eval_do:
     mov esi, r12d                             ; esi = stream, cached prog
     call EvalLogoStream                       ; edx = updated env, discarded
     xor eax, eax                              ; success: return 0
     jmp .es_done                              ; shared epilogue
.es_key_not_found:
     mov eax, -1                               ; key not found: return -1
     jmp .es_done                              ; shared epilogue
.es_oversize:
     FATAL "SectorLogo: DB script exceeds the 1 MiB script buffer" ; eval_script copy overflow, tuner + eviction-callback scripts share the buffer
.es_done:
     lea rsp, [rbp - SL_ES_PUSHBASE]           ; drop the 40-byte frame; land on the saved-r15 slot, SF_END mirror
     pop r15                                   ; restore non-volatile, reverse order
     pop r14                                   ; restore non-volatile
     pop r13                                   ; restore non-volatile
     pop r12                                   ; restore non-volatile
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     pop rbp                                   ; restore caller's frame pointer
     ENGINE_LEAVE                              ; park the landing zone, normal path
     SL_STK_LEAVE                              ; switch back to the C stack
     ret                                       ; return eax to caller
.es_streq:
     xor eax, eax                              ; eax = 0, not-equal default
.es_streq_loop:
     lodsb                                     ; al = [rsi++], lodsb writes AL, i.e. it dirties EAX
     cmp al, [rdi]                             ; match the query byte?
     jne .es_streq_miss                        ; no: discard the byte lodsb left in AL
     inc rdi                                   ; advance query pointer
     test al, al                               ; both walked to NUL together?
     jnz .es_streq_loop                        ; no: keep comparing
     inc eax                                   ; yes: al == 0 here, so eax is still 0 and this yields 1
     ret                                       ; equal: eax = 1
.es_streq_miss:
     xor eax, eax                              ; discard AL so a mismatch reads as 0, not as the byte
     ret                                       ; not-equal: eax = 0

; ============================================================================
; Per-chat sidecar activation: THE DATABASE KEY IS THE SWITCH.
; Called on EVERY chat (chat_session), before prefill. The host never arms
; the hook -- it manages the "logo:sidecar" DB key via v7_db_add (add a new
; script or overwrite the old one to SWITCH sidecars) and this function
; picks the change up: key absent = hook off, key present = hook on, script
; text changed = re-parse. No caching beyond the one parsed program, no
; host-side mode at all.
; ============================================================================
%define RS_PUSHBASE 56                         ; 7 pushed non-volatiles below rbp, hand-ended SF_END rsp recovery
%define RS_FRAME 40                            ; committed frame bytes, SF_COMMIT mirror: 32 shadow + 8 alignment
v7s_sidecar_refresh:
     SL_STK_ENTER                              ; switch to the private Logo stack, parse interns atoms and the GC must see this stack
     push rbp                                  ; save caller's frame pointer, SF_START mirror
     mov rbp, rsp                              ; frame pointer, SF_START mirror
     push rbx                                  ; save non-volatile, SF_START mirror
     push rsi                                  ; save non-volatile, SF_START mirror
     push rdi                                  ; save non-volatile, SF_START mirror
     push r12                                  ; save non-volatile, SF_START mirror
     push r13                                  ; save non-volatile, SF_START mirror
     push r14                                  ; save non-volatile, SF_START mirror
     push r15                                  ; save non-volatile, SF_START mirror
     sub rsp, RS_FRAME                         ; shadow space + alignment, SF_COMMIT mirror
; lazy one-shot init: the first chat reaches the tuner eval first (chat_session
; order), so the world is usually ready -- but never assume it
     cmp dword [rel sl_lg_initialized], 0      ; Logo world ready?
     jne .rs_ready                             ; yes: skip one-shot init
     call lg_init                              ; heap + autoexec stdlib one-shot
     mov dword [rel sl_lg_initialized], 1      ; mark initialized for later calls
.rs_ready:
     mov r13, [rel sl_lg_heap_base]            ; pin heap base for GC, same pin as every entry
; ht_find,rcx=key, rdx=&out_len -> rax = script text or NULL
     lea rcx, [rel str_sl_db_key_sidecar]      ; key = "logo:sidecar"
     lea rdx, [rel sl_lg_db_len]               ; &out_len
     call ht_find                              ; rax = data ptr or NULL
     test rax, rax                             ; key absent in the live DB?
     jnz .rs_present                           ; present: hook on, check for a switched script
; key absent: hook off, callback inert -- a host removed the key or never set it
     mov dword [rel sl_hook_enabled], 0        ; hook off
     mov dword [rel sl_lg_callback_prog], 0    ; callback inert, no script to run
     mov qword [rel sl_sidecar_cache_len], -1  ; re-adding even identical script must reparse
     jmp .rs_done                              ; hook off, nothing more to do
.rs_present:
     mov dword [rel sl_hook_enabled], 1        ; sidecar in the DB: hook on
; unchanged? compare length and bytes against the SIDECAR SNAPSHOT: the
; private identity baseline. sl_lg_script_buf is NOT usable here -- every
; named-script eval (tuner, eviction) reparses through it, so a shared
; baseline would flag an unchanged sidecar as changed on every chat.
     mov r12, [rel sl_lg_db_len]               ; r12 = current script length
     cmp r12, [rel sl_sidecar_cache_len]       ; same length as the snapshotted source?
     jne .rs_changed                           ; no: switched -- re-parse
     mov rsi, rax                              ; rsi = fresh script text
%ifdef V7_LINUX
     lea rdi, [rel sl_sidecar_snap]            ; PIE: private snapshot address via RIP-relative lea
%else
     mov rdi, sl_sidecar_snap                  ; rdi = the private snapshot the cached program was parsed from
%endif
     mov rcx, r12                              ; rcx = byte count
.rs_cmp_qwords:
     cmp rcx, 8                                ; a whole qword left?
     jb .rs_cmp_bytes                          ; no: finish byte-wise
     cmpsq                                     ; compare 8 bytes, rsi/rdi += 8
     jne .rs_changed                           ; no: switched
     sub rcx, 8                                ; consume 8 bytes
     jmp .rs_cmp_qwords                        ; next qword
.rs_cmp_bytes:
     test rcx, rcx                             ; bytes left?
     jz .rs_same                               ; no: identical -- cached parse stands
     cmpsb                                     ; compare 1 byte, rsi/rdi += 1
     jne .rs_changed                           ; no: switched
     dec rcx                                   ; consume 1 byte
     jmp .rs_cmp_bytes                         ; next byte
.rs_same:
     jmp .rs_done                              ; cached parse stands, nothing to do
.rs_changed:
     mov [rel sl_sidecar_cache_len], r12       ; publish the new length
     mov rsi, rax                              ; rsi = fresh script text, rax still holds the ht_find source: the cmpsq/cmpsb walk advances rsi/rdi only
     SL_COPY_DB_SCRIPT sl_sidecar_snap, .rs_oversize ; snapshot the source: every later refresh compares against THIS private copy
.parse_snapshot:
     SL_PRIME_READER sl_sidecar_snap           ; parse the already-copied snapshot, no second copy
     call ReadLogoStream                       ; eax = parsed cons list
     mov [rel sl_lg_callback_prog], eax        ; store parsed program, the hook runs THIS from now on
.rs_done:
; Eviction-callback activation: probe the "on_stream_evict" key on every chat
; entry -- THE DATABASE KEY IS THE SWITCH, same contract as the sidecar. The
; flush delivers the evicted batch only when a consumer listens: the sidecar
; hook OR this key. Host adds/removes the script -> next chat's probe flips
; the flag, the flush follows without any host-side mode.
     lea rcx, [rel stream_remove_script_key]   ; key = "on_stream_evict"
     lea rdx, [rel sl_lg_db_len]               ; &out_len, scratch slot
     call ht_find                              ; rax = script text or NULL
     test rax, rax                             ; eviction callback present?
     jz .rs_evict_off                          ; absent: flush drops batches again
     mov dword [rel sl_evict_enabled], 1       ; present: the flush delivers to this script
     jmp .rs_done2                             ; probe complete
.rs_evict_off:
     mov dword [rel sl_evict_enabled], 0       ; script removed: flush drops again
.rs_done2:
     mov eax, 1                                ; refresh never fails: the documented constant return
     lea rsp, [rbp - RS_PUSHBASE]              ; drop the frame; land on the saved-r15 slot, SF_END mirror
     pop r15                                   ; restore non-volatile, reverse order
     pop r14                                   ; restore non-volatile
     pop r13                                   ; restore non-volatile
     pop r12                                   ; restore non-volatile
     pop rdi                                   ; restore non-volatile
     pop rsi                                   ; restore non-volatile
     pop rbx                                   ; restore non-volatile
     pop rbp                                   ; restore caller's frame pointer
     SL_STK_LEAVE                              ; switch back to the C stack
     ret                                       ; rax = 1 always, refresh never fails
.rs_oversize:
     FATAL "SectorLogo: sidecar script exceeds the 1 MiB script buffer"
%undef RS_PUSHBASE
%undef RS_FRAME

%unmacro INTERN_ONE 2                          ; remove the intern helper, scope ends with intern_builtins
%unmacro SL_PRIME_READER 1                     ; remove the reader-priming helper, generic name, would outlive the file otherwise
%unmacro SL_RESET_CALL_STATE 1                 ; remove the per-call state reset, generic name
%unmacro SL_COPY_DB_SCRIPT 2                   ; remove the DB-script copier, generic name
%unmacro SL_WIPE_DWORDS 2                      ; remove the BSS clear helper, generic name
%undef ON_TOKEN_TEMP_PTR_OFF
%undef ON_TOKEN_TOPP_PTR_OFF
%undef SL_ES_PUSHBASE
%undef SL_ES_FRAME
