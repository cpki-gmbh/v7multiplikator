; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
     section .data
; ============================================================================
; --- MicroDB key for sidecar script ---
str_sl_db_key_sidecar:
     db "logo:sidecar", 0                      ; DB key for custom script
; --- Core Logo builtins ---
str_sl_make:
     db SL_TYPE_WORD, "make", 0                ; variable assignment, PascalCase: Make
str_sl_repeat:
     db SL_TYPE_WORD, "repeat", 0              ; counted loop, Repeat
str_sl_printraw:
     db SL_TYPE_WORD, "printraw", 0            ; output value without CRLF, PrintRaw; PRINT is stdlib
str_sl_first:
     db SL_TYPE_WORD, "first", 0               ; list head, First
str_sl_butfirst:
     db SL_TYPE_WORD, "butfirst", 0            ; list tail, ButFirst
str_sl_fput:
     db SL_TYPE_WORD, "prepend", 0             ; prepend to list, Prepend
str_sl_if:
     db SL_TYPE_WORD, "if", 0                  ; conditional, If
str_sl_ifelse:
     db SL_TYPE_WORD, "ifelse", 0              ; two-branch conditional, IfElse
str_sl_stop:
     db SL_TYPE_WORD, "stop", 0                ; break current REPEAT loop, Stop
str_sl_continue:
     db SL_TYPE_WORD, "continue", 0            ; skip to next REPEAT iteration, Continue
str_sl_terminate:
     db SL_TYPE_WORD, "exit", 0                ; exit entire program, Exit
str_sl_concat:
     db SL_TYPE_WORD, "_", 0                   ; string concatenation infix
str_sl_t:
     db SL_TYPE_WORD, "true", 0                ; boolean true atom, True
str_sl_false:
     db SL_TYPE_WORD, "false", 0               ; boolean false atom, False
str_sl_lparen:
     db SL_TYPE_WORD, "(", 0                   ; open paren
str_sl_rparen:
     db SL_TYPE_WORD, ")", 0                   ; close paren
; --- Metaprogramming & homoiconic execution ---
str_sl_run:
     db SL_TYPE_WORD, "run", 0                 ; execute block, Run
str_slf_prefix:
     db "[FATAL] ", 0                          ; composed diagnostic prefix
str_slf_got:
     db ", got '", 0                           ; CONS/NIL offender introduction, opening quote
str_slf_gotw:
     db ", got word '", 0                      ; WORD offender introduction, type word outside the quotes
str_slf_gotn:
     db ", got number '", 0                    ; NUMBER offender introduction: type word outside the quotes
str_slf_gotf:
     db ", got float '", 0                     ; FLOAT offender introduction: type word outside the quotes
str_slf_quote:
     db "'", 0                                 ; offender closing quote
str_slf_nil:
     db "NIL", 0                               ; NIL offender rendering
str_slf_list:
     db "list", 0                              ; CONS offender rendering
; --- Function definition & return ---
str_sl_def_fn:
     db SL_TYPE_WORD, "define", 0              ; define function, Define
str_sl_def_exe_fn:
     db SL_TYPE_WORD, "definecommand", 0       ; define external-tool function, DefineCommand
str_sl_output:
     db SL_TYPE_WORD, "output", 0              ; return from function, Output
str_sl_fn_tag:
     db SL_TYPE_WORD, "__FN__", 0              ; internal function tag
str_sl_xfn_tag:
     db SL_TYPE_WORD, "__XF__", 0              ; internal execute-function tag
; --- Arithmetic / math names + symbol aliases ---
str_sl_plus:
     db SL_TYPE_WORD, "+", 0                   ; addition, infix
str_sl_minus_sym:
     db SL_TYPE_WORD, "-", 0                   ; subtraction, infix
str_sl_mul_sym:
     db SL_TYPE_WORD, "*", 0                   ; multiplication, infix
str_sl_div_sym:
     db SL_TYPE_WORD, "/", 0                   ; division, infix
str_sl_mod_sym:
     db SL_TYPE_WORD, "%", 0                   ; modulo, infix
str_sl_power:
     db SL_TYPE_WORD, "power", 0               ; exponentiation, Power
str_sl_sqrt:
     db SL_TYPE_WORD, "squareroot", 0          ; integer sqrt, SquareRoot
str_sl_abs:
     db SL_TYPE_WORD, "absolutevalue", 0       ; absolute value, AbsoluteValue
str_sl_minus:
     db SL_TYPE_WORD, "negate", 0              ; unary negate, prefix, Negate
str_sl_max:
     db SL_TYPE_WORD, "maximum", 0             ; maximum of two, Maximum
str_sl_min:
     db SL_TYPE_WORD, "minimum", 0             ; minimum of two, Minimum
str_sl_random:
     db SL_TYPE_WORD, "random", 0              ; random integer, AES-NI, Random
str_sl_real_random:
     db SL_TYPE_WORD, "realrandom", 0          ; toggle RDTSC entropy injection, RealRandom
; --- Float primitive names ---
str_sl_i2f:
     db SL_TYPE_WORD, "asfloat", 0             ; int -> float conversion, AsFloat
str_sl_f2i:
     db SL_TYPE_WORD, "asinteger", 0           ; float -> int, truncate toward 0, AsInteger
str_sl_floor:
     db SL_TYPE_WORD, "floor", 0               ; float floor, Floor
str_sl_fround:
     db SL_TYPE_WORD, "round", 0               ; float round-half-to-even, Round
str_sl_floatp:
     db SL_TYPE_WORD, "isfloat", 0             ; FLOAT tag predicate, IsFloat
str_sl_listp:
     db SL_TYPE_WORD, "islist", 0              ; CONS tag predicate, IsList
str_sl_isstringatom:
     db SL_TYPE_WORD, "isstringatom", 0        ; STRING type-byte predicate, IsStringAtom
str_sl_iswordatom:
     db SL_TYPE_WORD, "iswordatom", 0          ; WORD type-byte predicate, IsWordAtom
str_sl_flog2:
     db SL_TYPE_WORD, "logbasetwo", 0          ; log2,x via x87 fyl2x, LogBaseTwo
str_sl_fexp2:
     db SL_TYPE_WORD, "poweroftwo", 0          ; 2^x via x87 f2xm1+fscale, PowerOfTwo
str_sl_fln:
     db SL_TYPE_WORD, "naturallog", 0          ; ln,x = log2,x * ln,2, NaturalLog
str_sl_fmod:
     db SL_TYPE_WORD, "floatremainder", 0      ; float remainder via x87 fprem, FloatRemainder
; --- Float constants, binary32 bit patterns for SSE/x87 operands ---
     align 4
flt_ten:
     dd 10.0                                   ; digit-extraction multiplier
     align 8
dbl_ten:
     dq 10.0                                   ; backward fraction accumulation divisor, binary64
flt_one:
     dd 1.0                                    ; [1,10 range bound
flt_2pow_neg24:
     dd 0x33800000                             ; 2^-24 reciprocal, hashnumber crc24 -> [0,1 ; pure exponent adjust
flt_1e9_f:
     dd 1.0e9                                  ; fixed-notation upper bound
flt_1e_3_f:
     dd 0.001                                  ; fixed-notation lower bound
     align 16                                  ; the two 16-byte vector masks are read by vandps/vxorps: a non-16-aligned base splits a quarter of them
flt_abs_mask:
     dd 0x7FFFFFFF, 0, 0, 0                    ; |x| mask, clears sign bit, padded to 16 bytes for the vandps full-vector read, float_fmt site
     flt_sign_mask equ swk_sign_mask           ; sign-flip mask, xor; identical 16-byte vector shared with swiglu_kernel.inc
pow10_table:
     dd 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000 ; 10^n for n = 0..9, shared: atof literal parser + float fmt + str_to_num; SF_ATOM_DIGIT_MAX = 9 caps the index
sl_v7_100f:
     dd 100.0                                  ; percentage divisor for settemperature / setnucleus
sl_f2i_min_f:
     dd 0xCE000000                             ; -536870912.0f, saturating floor for integer conversion
sl_f2i_cap_f:
     dd 0x4E000000                             ; 536870912.0f, saturating ceiling boundary for integer conversion
     align 16
     align 32                                  ; the casefold windows are read as full YMM/XMM operands
     sl_case_bit5: times 32 db 0x20            ; case bit, toggled on matched ASCII letters
     sl_case_upper_bias: times 32 db 127 - 'Z' ; bias mapping A-Z onto signed 102..127
     sl_case_lower_bias: times 32 db 127 - 'z' ; bias mapping a-z onto signed 102..127
     sl_case_threshold: times 32 db 127 - ('Z' - 'A' + 1) ; signed compare threshold, one past the wrapped letter run
; --- Bitwise names + symbol aliases ---
str_sl_and_sym:
     db SL_TYPE_WORD, "&", 0                   ; bitwise AND, infix
str_sl_or_sym:
     db SL_TYPE_WORD, "|", 0                   ; bitwise OR, infix
str_sl_xor_sym:
     db SL_TYPE_WORD, "^", 0                   ; bitwise XOR, infix
str_sl_not_sym:
     db SL_TYPE_WORD, "~", 0                   ; bitwise NOT, prefix form ~x; no infix form
str_sl_shl_sym:
     db SL_TYPE_WORD, "<<", 0                  ; left shift, infix
str_sl_shr_sym:
     db SL_TYPE_WORD, ">>", 0                  ; right shift, infix
; --- Relational / boolean names + symbol aliases ---
str_sl_eq_sym:
     db SL_TYPE_WORD, "=", 0                   ; equality test, infix
str_sl_neq_sym1:
     db SL_TYPE_WORD, "<>", 0                  ; inequality test, infix alt1
str_sl_neq_sym2:
     db SL_TYPE_WORD, "!=", 0                  ; inequality test, infix alt2
str_sl_gt_sym:
     db SL_TYPE_WORD, ">", 0                   ; greater-than, infix
str_sl_lt_sym:
     db SL_TYPE_WORD, "<", 0                   ; less-than, infix
str_sl_gte_sym:
     db SL_TYPE_WORD, ">=", 0                  ; greater-or-equal, infix
str_sl_lte_sym:
     db SL_TYPE_WORD, "<=", 0                  ; less-or-equal, infix
str_sl_andand:
     db SL_TYPE_WORD, "&&", 0                  ; logical AND, infix only; a prefix spelling self-evaluates as a dead name
str_sl_oror:
     db SL_TYPE_WORD, "||", 0                  ; logical OR, infix only; a prefix spelling self-evaluates as a dead name
str_sl_bang:
     db SL_TYPE_WORD, "!", 0                   ; logical NOT, prefix symbol, binds to one primary
str_sl_zerop:
     db SL_TYPE_WORD, "iszero", 0              ; zero predicate, IsZero
str_sl_numberp:
     db SL_TYPE_WORD, "isnumber", 0            ; number predicate, IsNumber
; --- Host primitive names ---
str_sl_v7_encode_word:
     db SL_TYPE_WORD, "encodeword", 0          ; BPE-encode a word, EncodeWord
str_sl_v7_db_get:
     db SL_TYPE_WORD, "databaseget", 0         ; look up DB key, DatabaseGet
str_sl_v7_db_put:
     db SL_TYPE_WORD, "databaseset", 0         ; store DB key/value, DatabaseSet
str_sl_v7_pos:
     db SL_TYPE_WORD, "position", 0            ; current position, Position
str_sl_v7_get_token_pos:
     db SL_TYPE_WORD, "gettokenposition", 0    ; stream position of the current token, local to this query, GetTokenPosition
str_sl_v7_get_token:
     db SL_TYPE_WORD, "gettoken", 0            ; read token at position, GetToken
str_sl_v7_vocab_size:
     db SL_TYPE_WORD, "vocabularysize", 0      ; vocabulary cardinality
str_sl_v7_dim:
     db SL_TYPE_WORD, "modeldimension", 0      ; model dimension, ModelDimension
str_sl_v7_layers:
     db SL_TYPE_WORD, "layercount", 0          ; layer count, LayerCount
str_sl_v7_eos:
     db SL_TYPE_WORD, "endtoken", 0            ; eos token id, EndToken
str_sl_v7_temp:
     db SL_TYPE_WORD, "settemperature", 0      ; set temperature, SetTemperature
str_sl_v7_topp:
     db SL_TYPE_WORD, "setnucleus", 0          ; set top-p, SetNucleus
; --- Stream rewrite primitives ---
str_sl_v7_stream:
     db SL_TYPE_WORD, "getstream", 0           ; return current stream, GetStream
str_sl_v7_str_len:
     db SL_TYPE_WORD, "streamlength", 0        ; return stream byte length, StreamLength
str_sl_v7_set_stream:
     db SL_TYPE_WORD, "setstream", 0           ; set rewritten stream, SetStream
str_sl_v7_set_token:
     db SL_TYPE_WORD, "setoutputtoken", 0      ; replace the current candidate token, SetOutputToken
str_sl_v7_reject:
     db SL_TYPE_WORD, "reject", 0              ; reject and resample, Reject
str_sl_v7_halt:
     db SL_TYPE_WORD, "halt", 0                ; halt generation, Halt
; --- String operation names ---
str_sl_substr:
     db SL_TYPE_WORD, "part", 0                ; extract substring, Part
str_sl_find:
     db SL_TYPE_WORD, "search", 0              ; find substring position, Search
str_sl_strlower:
     db SL_TYPE_WORD, "lowercase", 0           ; lowercase conversion, Lowercase
str_sl_strupper:
     db SL_TYPE_WORD, "uppercase", 0           ; uppercase conversion, Uppercase
str_sl_strreplace:
     db SL_TYPE_WORD, "replace", 0             ; replace first occurrence, Replace
str_sl_strtonum:
     db SL_TYPE_WORD, "texttonumber", 0        ; parse string to number/float, TextToNumber
str_sl_strlen:
     db SL_TYPE_WORD, "textlength", 0          ; string byte length, TextLength
str_sl_explode:
     db SL_TYPE_WORD, "split", 0               ; string split, Split
str_sl_str_tokens:
     db SL_TYPE_WORD, "words", 0               ; alnum-run tokenizer, Words
str_sl_split_ngrams:
     db SL_TYPE_WORD, "ngrams", 0              ; sliding n-gram windows, NGrams
str_sl_int_to_char:
     db SL_TYPE_WORD, "character", 0           ; int 0..255 -> one-char string atom, Character
str_sl_char_to_int:
     db SL_TYPE_WORD, "number", 0              ; first char of string -> int, Number
str_sl_do_vq:
     db SL_TYPE_WORD, "nearestmatch", 0        ; vector quantization, NearestMatch
str_sl_lindex:
     db SL_TYPE_WORD, "item", 0                ; list element by index, Item
str_sl_local:
     db SL_TYPE_WORD, "local", 0               ; declare function-scope local variable, Local
str_sl_sort:
     db SL_TYPE_WORD, "sort", 0                ; ascending radix sort, Sort
str_sl_reverse:
     db SL_TYPE_WORD, "reverse", 0             ; list reversal, Reverse
str_sl_hash_float:
     db SL_TYPE_WORD, "hashnumber", 0          ; CRC32C -> FLOAT hash key in [0,1, HashNumber
str_sl_length:
     db SL_TYPE_WORD, "length", 0              ; list element count, Length
str_sl_lset:
     db SL_TYPE_WORD, "setitem", 0             ; list element set by index, SetItem
str_sl_setcar:
     db SL_TYPE_WORD, "setcar", 0              ; destructive car-field write on a cons cell, SetCar
str_sl_setcdr:
     db SL_TYPE_WORD, "setcdr", 0              ; destructive cdr-field write on a cons cell, SetCdr
str_sl_samevalue:
     db SL_TYPE_WORD, "samevalue", 0           ; tagged-word identity test, SameValue
str_sl_asword:
     db SL_TYPE_WORD, "asword", 0              ; string -> word-atom conversion, AsWord
str_sl_lfind:
     db SL_TYPE_WORD, "finditem", 0            ; list scan: exact key match, FindItem
str_sl_lfindge:
     db SL_TYPE_WORD, "findfirstatleast", 0    ; list scan: first element >= key, FindFirstAtLeast
str_sl_lfindle:
     db SL_TYPE_WORD, "findfirstatmost", 0     ; list scan: first element <= key, FindFirstAtMost
; --- Console primitive names ---
str_sl_getchar:
     db SL_TYPE_WORD, "readkey", 0             ; non-blocking stdin poll, ReadKey
str_sl_console_24bit:
     db SL_TYPE_WORD, "enablecolors", 0        ; VT/24-bit color probe + enabler, EnableColors
; --- Trace primitive names ---
str_sl_enabletrace:
     db SL_TYPE_WORD, "enabletrace", 0         ; runtime evaluator tracing, EnableTrace
str_sl_enablestrictnames:
     db SL_TYPE_WORD, "enablestrictnames", 0   ; unbound name in expression position FATALs, EnableStrictNames
; --- FFI names ---
str_sl_c_call:
     db SL_TYPE_WORD, "calllibrary", 0         ; C function call primitive, CallLibrary
str_sl_type_int:
     db SL_TYPE_STRING, "integer", 0           ; integer type, Integer, STRING because C-CALL descriptors quote type names
str_sl_type_ptr:
     db SL_TYPE_STRING, "pointer", 0           ; pointer type, Pointer, STRING because C-CALL descriptors quote type names
str_sl_type_str:
     db SL_TYPE_STRING, "text", 0              ; string pointer type, Text, STRING because C-CALL descriptors quote type names
str_sl_type_void:
     db SL_TYPE_STRING, "void", 0              ; void return type, Void, STRING because C-CALL descriptors quote type names
; --- eval_script dispatch key strings, .data: non-zero initialisers, a .bss
; --- placement would hand the loader a zero-fill and empty keys ---
stream_remove_script_key:
     db "on_stream_evict", 0                   ; MicroDB key of the StreamingLLM eviction-callback script
; --- Hook-scoped variable names, bound fresh on every call ---
str_sl_lg_tok:
     db SL_TYPE_WORD, "tokennumber", 0         ; current token id, TokenNumber
str_sl_lg_piece:
     db SL_TYPE_WORD, "piece", 0               ; current token text, Piece
; --- autoexec.logo standard-library loader ---
str_sl_autoexec_path:
     db "autoexec.logo", 0                     ; stdlib file loaded once at lg_init
str_sl_db_key_autoexec:
     db "__sectorlogo__autoexec", 0            ; MicroDB key for the stdlib blob, tier 1
sl_msg_ae_from_db:
     db "SectorLogo: stdlib autoexec loaded from database", 0 ; tier-1 notice, WRITE_STR_NL appends the newline
sl_msg_ae_from_file:
     db "SectorLogo: stdlib autoexec loaded from file autoexec.logo", 0 ; tier-2 notice
sl_msg_ae_none:
     db "SectorLogo: no stdlib autoexec found - default commands only, sidecar scripts have fewer commands available", 0 ; tier-3 notice
; --- RNG state ---
     align 8
sl_rng_state:
     dq 0x853C49E6748FEA9B                     ; 64-bit PRNG state, AES round-key whitened
sl_real_random:
     dd 0                                      ; if 1: XOR RDTSC into AES input
     dd 0                                      ; padding
; --- AES-NI RNG constants ---
     sl_aes_block_hi equ s_aes_block_hi        ; second AES whitening block qword, identical constant shared with sampler.inc
     align 16                                  ; must stay 16: legacy aesenc memory operand faults unaligned
sl_aes_keys_combined:
     dq 0x94d049bb133111eb                     ; round key bytes 0-7
     dq 0x1bf95cfc92db11eb                     ; round key bytes 8-15, second SplitMix64 mixer qword, REQUIRED: it is the high 64 bits of the aesenc operand read at sl_aes_keys_combined
     align 16
     sl_weyl_inc equ s_weyl_inc                ; Weyl increment constant, identical constant shared with sampler.inc
; ============================================================================
     section .bss
; ============================================================================
; --- Persistent, once-per-process state ---
sl_lg_initialized:
     resd 1                                    ; 1 after lg_init has run
     resd 1                                    ; padding
sl_lg_heap_base:
     resq 1                                    ; pinned Logo heap base, mdb_alloc_mem
sl_lg_callback_prog:
     resd 1                                    ; parsed callback program, cons list
     resd 1                                    ; padding
sl_sidecar_cache_len:
     resq 1                                    ; byte length of the sidecar source the cached program was parsed from (0 = none)
sl_sidecar_snap:
     resb SL_SCRIPT_BUF_SIZE                   ; sidecar SOURCE snapshot: the refresh's change-detector baseline, private because sl_lg_script_buf is shared with every named-script parse
sl_lg_autoexec_prog:
     resd 1                                    ; parsed autoexec.logo forms, GC root 5
     resd 1                                    ; padding
sl_lg_base_env:
     resd 1                                    ; base env alist from one-shot autoexec eval, GC root 6
     resd 1                                    ; padding
sl_dispatch_highwater:
     resd 1                                    ; sl_atom_free value at dispatch-lock time; the atom sweep's builtin-region floor
; --- Cons / atom heap cursors ---
sl_atom_free:
     resd 1                                    ; next free byte offset in atom area, 8-aligned by induction
     resd 1                                    ; padding
sl_cons_free:
     resd 1                                    ; next free cons cell index
     resd 1                                    ; padding
sl_intern_filter:
     resd (SL_INTERN_FILTER_BITS / 32)         ; dual-hash bloom bitfield, 512 KiB
sl_intern_cache:
     resd (SL_INTERN_CACHE_BYTES / 4)          ; Four-way atom cache
sl_dispatch_stmt_table:
     resq SL_DISPATCH_TABLE_ENTRIES            ; statement-position handlers, EvalLogoStream
sl_dispatch_expr_table:
     resq SL_DISPATCH_TABLE_ENTRIES            ; expression-position handlers, EvalPrimary
sl_op_atom_bitmap:
     resd (SL_DISPATCH_MAX_OFF / 32)           ; 256 dwords = 1 KiB, zero-initialized
sl_builtin_bitmap:
     resd (SL_DISPATCH_MAX_OFF / 32)           ; 256 dwords = 1 KiB, zero-initialized
; --- Interned atom handles: core builtins ---
atom_sl_make:
     resd 1                                    ; MAKE atom
atom_sl_repeat:
     resd 1                                    ; REPEAT atom
atom_sl_printraw:
     resd 1                                    ; PRINTRAW atom, PRINT is a stdlib fn
atom_sl_first:
     resd 1                                    ; FIRST atom
atom_sl_butfirst:
     resd 1                                    ; BUTFIRST atom
atom_sl_fput:
     resd 1                                    ; FPUT atom
atom_sl_if:
     resd 1                                    ; IF atom
atom_sl_ifelse:
     resd 1                                    ; IFELSE atom
atom_sl_stop:
     resd 1                                    ; STOP atom
atom_sl_continue:
     resd 1                                    ; CONTINUE atom
atom_sl_terminate:
     resd 1                                    ; TERMINATE atom
atom_sl_concat:
     resd 1                                    ; _, string concat atom
atom_sl_t:
     resd 1                                    ; TRUE atom
atom_sl_false:
     resd 1                                    ; FALSE atom
atom_sl_lparen:
     resd 1                                    ; ',' atom
atom_sl_rparen:
     resd 1                                    ; ' ' atom
; --- Interned atom handles: metaprogramming ---
atom_sl_run:
     resd 1                                    ; RUN atom
; --- Interned atom handles: function definition & return ---
atom_sl_def_fn:
     resd 1                                    ; DEF_FN atom
atom_sl_output:
     resd 1                                    ; OUTPUT atom
atom_sl_fn_tag:
     resd 1                                    ; __FN__ tag atom
atom_sl_def_exe_fn:
     resd 1                                    ; DEF_EXE_FN atom
atom_sl_xfn_tag:
     resd 1                                    ; __XF__ tag atom
; --- Interned atom handles: arithmetic / math ---
atom_sl_plus:
     resd 1                                    ; + atom
atom_sl_minus_sym:
     resd 1                                    ; - atom
atom_sl_mul_sym:
     resd 1                                    ; * atom
atom_sl_div_sym:
     resd 1                                    ; / atom
atom_sl_mod_sym:
     resd 1                                    ; % atom
atom_sl_power:
     resd 1                                    ; POWER atom
atom_sl_sqrt:
     resd 1                                    ; SQRT atom
atom_sl_abs:
     resd 1                                    ; ABS atom
atom_sl_minus:
     resd 1                                    ; MINUS atom
atom_sl_max:
     resd 1                                    ; MAX atom
atom_sl_min:
     resd 1                                    ; MIN atom
atom_sl_random:
     resd 1                                    ; RANDOM atom
atom_sl_real_random:
     resd 1                                    ; REAL-RANDOM atom
; --- Interned atom handles: float primitives ---
atom_sl_i2f:
     resd 1                                    ; I2F atom
atom_sl_f2i:
     resd 1                                    ; F2I atom
atom_sl_floor:
     resd 1                                    ; FLOOR atom
atom_sl_fround:
     resd 1                                    ; FROUND atom
atom_sl_floatp:
     resd 1                                    ; FLOATP atom
atom_sl_flog2:
     resd 1                                    ; FLOG2 atom
atom_sl_fexp2:
     resd 1                                    ; FEXP2 atom
atom_sl_fln:
     resd 1                                    ; FLN atom
atom_sl_fmod:
     resd 1                                    ; FMOD atom
; --- Interned atom handles: bitwise ---
atom_sl_and_sym:
     resd 1                                    ; & atom
atom_sl_or_sym:
     resd 1                                    ; | atom
atom_sl_xor_sym:
     resd 1                                    ; ^ atom
atom_sl_not_sym:
     resd 1                                    ; ~ atom
atom_sl_shl_sym:
     resd 1                                    ; << atom
atom_sl_shr_sym:
     resd 1                                    ; >> atom
; --- Interned atom handles: relational / boolean ---
atom_sl_eq_sym:
     resd 1                                    ; = atom
atom_sl_neq_sym1:
     resd 1                                    ; <> atom
atom_sl_neq_sym2:
     resd 1                                    ; != atom
atom_sl_gt_sym:
     resd 1                                    ; > atom
atom_sl_lt_sym:
     resd 1                                    ; < atom
atom_sl_gte_sym:
     resd 1                                    ; >= atom
atom_sl_lte_sym:
     resd 1                                    ; <= atom
atom_sl_andand:
     resd 1                                    ; && atom
atom_sl_oror:
     resd 1                                    ; || atom
atom_sl_bang:
     resd 1                                    ; ! atom
atom_sl_zerop:
     resd 1                                    ; ZEROP atom
atom_sl_numberp:
     resd 1                                    ; NUMBERP atom
atom_sl_listp:
     resd 1                                    ; LISTP atom, IsList
atom_sl_isstringatom:
     resd 1                                    ; IsStringAtom atom, STRING type-byte predicate
atom_sl_iswordatom:
     resd 1                                    ; IsWordAtom atom, WORD type-byte predicate
; --- Interned atom handles: V7 host primitives ---
atom_sl_v7_encode_word:
     resd 1                                    ; V7-ENCODE-WORD atom
atom_sl_v7_db_get:
     resd 1                                    ; V7-DB-GET atom
atom_sl_v7_db_put:
     resd 1                                    ; V7-DB-PUT atom
atom_sl_v7_pos:
     resd 1                                    ; V7-POS atom
atom_sl_v7_get_token_pos:
     resd 1                                    ; V7-GET-TOKEN-POS atom
atom_sl_v7_get_token:
     resd 1                                    ; V7-GET-TOKEN atom
atom_sl_v7_vocab_size:
     resd 1                                    ; V7-VOCAB-SIZE atom
atom_sl_v7_dim:
     resd 1                                    ; V7-DIM atom
atom_sl_v7_layers:
     resd 1                                    ; V7-LAYERS atom
atom_sl_v7_eos:
     resd 1                                    ; V7-EOS atom
atom_sl_v7_temp:
     resd 1                                    ; V7-TEMP atom
atom_sl_v7_topp:
     resd 1                                    ; V7-TOPP atom
; --- Interned atom handles: V7 stream rewrite primitives ---
atom_sl_v7_stream:
     resd 1                                    ; V7-STREAM atom
atom_sl_v7_str_len:
     resd 1                                    ; V7-STR-LEN atom
atom_sl_v7_set_stream:
     resd 1                                    ; V7-SET-STREAM atom
atom_sl_v7_set_token:
     resd 1                                    ; V7-SET-TOKEN atom
atom_sl_v7_reject:
     resd 1                                    ; V7-REJECT atom
atom_sl_v7_halt:
     resd 1                                    ; V7-HALT atom
; --- Interned atom handles: string operations ---
atom_sl_substr:
     resd 1                                    ; SUBSTR atom
atom_sl_find:
     resd 1                                    ; Search atom
atom_sl_strlower:
     resd 1                                    ; STRLOWER atom
atom_sl_strupper:
     resd 1                                    ; STRUPPER atom
atom_sl_strreplace:
     resd 1                                    ; STR_REPLACE atom
atom_sl_strtonum:
     resd 1                                    ; STR-TO-NUM atom
atom_sl_strlen:
     resd 1                                    ; STRLEN atom
atom_sl_explode:
     resd 1                                    ; EXPLODE atom
atom_sl_str_tokens:
     resd 1                                    ; Words atom
atom_sl_split_ngrams:
     resd 1                                    ; SPLIT-NGRAMS atom
atom_sl_int_to_char:
     resd 1                                    ; Character atom
atom_sl_char_to_int:
     resd 1                                    ; Number atom
atom_sl_do_vq:
     resd 1                                    ; NearestMatch atom
atom_sl_lindex:
     resd 1                                    ; LINDEX atom
; --- LOCAL function-scope variables ---
atom_sl_local:
     resd 1                                    ; LOCAL atom
sl_local_set:
     resd SL_LOCAL_SET_MAX * 2                 ; [var atom][binding cell index] pairs declared LOCAL, current scope stack
; --- Interned atom handles: sorted-list / hash-map toolbox ---
atom_sl_sort:
     resd 1                                    ; SORT atom
atom_sl_reverse:
     resd 1                                    ; REVERSE atom
atom_sl_hash_float:
     resd 1                                    ; HASH-FLOAT atom
atom_sl_length:
     resd 1                                    ; LENGTH atom
atom_sl_lset:
     resd 1                                    ; LSET atom
atom_sl_setcar:
     resd 1                                    ; SETCAR atom
atom_sl_setcdr:
     resd 1                                    ; SETCDR atom
atom_sl_samevalue:
     resd 1                                    ; SAMEVALUE atom
atom_sl_asword:
     resd 1                                    ; ASWORD atom
atom_sl_lfind:
     resd 1                                    ; LFIND atom
atom_sl_lfindge:
     resd 1                                    ; LFINDGE atom
atom_sl_lfindle:
     resd 1                                    ; LFINDLE atom
; --- SORT scratch state, raw ints/flags, GC-irrelevant ---
sl_sort_n:
     resd 1                                    ; element count
     resd 1                                    ; padding, keeps the resq buffers below qword-aligned
; --- SORT record buffers, raw qwords/dwords, GC-inert ---
; Each 16-byte record: dword tagged VALUE at +0, binary64 monotone key at +8
     align 16
sl_sort_qw_a:
     resq SL_SORT_MAX_ELEMS * SL_RECORD_QWORDS ; ping-pong buffer A, 16-byte records: value dword + key qword
; radix buffer B and its histogram reuse sl_concat_buf exclusively during radix passes
; --- Interned atom handles: FFI, C-CALL ---
atom_sl_c_call:
     resd 1                                    ; C-CALL atom
atom_sl_type_int:
     resd 1                                    ; INT type atom
atom_sl_type_ptr:
     resd 1                                    ; PTR type atom
atom_sl_type_str:
     resd 1                                    ; STR type atom
atom_sl_type_void:
     resd 1                                    ; VOID type atom
atom_sl_getchar:
     resd 1                                    ; ReadKey atom
atom_sl_console_24bit:
     resd 1                                    ; EnableColors atom
; --- Interned atom handles: trace primitive ---
atom_sl_enabletrace:
     resd 1                                    ; ENABLETRACE atom
atom_sl_enablestrictnames:
     resd 1                                    ; ENABLESTRICTNAMES atom
; --- Interned atom handles: hook-scoped variables ---
atom_sl_lg_tok:
     resd 1                                    ; TOK variable atom
atom_sl_lg_piece:
     resd 1                                    ; PIECE variable atom
sl_hook_ctx:
     resq 1                                    ; InferenceContext*
sl_hook_tok_in:
     resd 1                                    ; input token id, SetOutputToken keeps this in sync with the switched state
sl_hook_pos:
     resd 1                                    ; current inference position
     resd 1                                    ; padding
sl_hook_out_tok_ptr:
     resq 1                                    ; &out_token, hook writes here
sl_hook_temp_ptr:
     resq 1                                    ; &temperature
sl_hook_topp_ptr:
     resq 1                                    ; &topp
sl_hook_tokens_ptr:
     resq 1                                    ; &token_history[0], for V7-GET-TOKEN
; --- Reader state, always memory-backed, never stdin ---
sl_src_mem_ptr:
     resq 1                                    ; pointer into current script buffer
sl_settok_id:
     resd 1                                    ; SetOutputToken state-switch: the replacement id parked across Intern/MakeNum, volatiles are destroyed inside them
sl_settok_cursor:
     resd 1                                    ; SetOutputToken state-switch: the env-alist walk cursor parked across the value builders
sl_g_look:
     resb 1                                    ; one-character lookahead
     resb 3                                    ; padding to 4-byte boundary
sl_last_tok_operand:
     resd 1                                    ; 1 = previous token was operand-end, word/num/str/ /], 0 = operator/delimiter
sl_list_depth:
     resd 1                                    ; list-literal nesting depth, '[' emitted minus ']' emitted ; >0 = inside [...], where every element is standalone
sl_tok_leading_ws:
     resb 1                                    ; 1 = current token began after whitespace, spaced prefix argument "-3.0" glues; compact infix "2+3" does not
     resb 3                                    ; padding to 4-byte boundary
sl_tok_buf:
     resb SL_TOK_BUF_SIZE                      ; current token string buffer
sl_lg_script_buf:
     resb SL_SCRIPT_BUF_SIZE                   ; script source copy
; --- DB / encode scratch ---
sl_lg_db_len:
     resq 1                                    ; out_len from ht_find
sl_lg_encode_scratch:
     resd SL_CONCAT_BUF_SIZE                   ; concatenated-string scratch, EncodeWord / string builds
sl_lg_encode_n:
     resd 1                                    ; count from encode,
     resd 1                                    ; padding
sl_stop_requested:
     resd 1                                    ; 1 = STOP requested, break current REPEAT
sl_continue_requested:
     resd 1                                    ; 1 = CONTINUE requested, next REPEAT iteration
sl_terminate_requested:
     resd 1                                    ; 1 = TERMINATE requested, exit the ENTIRE script; survives fn-boundary state restores
sl_fn_has_returned:
     resd 1                                    ; 1 if OUTPUT/TERMINATE was executed, 0 otherwise
sl_fn_return_val:
     resd 1                                    ; tagged value passed to OUTPUT, stays adjacent to the flag
sl_stmt_fn_retval:
     resd 1                                    ; last user/xfn STATEMENT-call return value, expression RUN reads it: Run [f args] applies f, the OUTPUT flag is consumed at the fn boundary, so this channel carries the value to the Run
; --- Recursion depth guardrail ---
sl_call_depth:
     resd 1                                    ; current call depth counter
sl_parse_depth:
     resd 1                                    ; reader recursion depth, list/paren nesting
; --- Print output support ---
sl_hStdOut:
     resq 1                                    ; cached stdout handle
sl_rw_byte:
     resb 1                                    ; single byte for WriteFile
     resb 7                                    ; padding
; --- Trace state, EnableTrace primitive ---
sl_trace_level:
     resd 1                                    ; 0=off, 1=stmts, 2=exprs, 3=verbose
sl_trace_indent:
     resd 1                                    ; current indentation depth for trace output
sl_strict_names:
     resd 1                                    ; 0=off, default, 1=strict
sl_hStdIn:
     resq 1                                    ; cached stdin handle, 0 = not resolved yet
sl_conin_fixed:
     resd 1                                    ; handle kind: 0=unprobed, 1=SL_HKIND_OTHER, regular file, 2=SL_HKIND_CONSOLE, 3=SL_HKIND_PIPE, 4=SL_HKIND_NONE
     resd 1                                    ; padding
sl_conin_mode:
     resd 1                                    ; GetConsoleMode scratch for the stdin raw-mode fixup
sl_console_state:
     resd 1                                    ; 24-bit probe cache: 0=unknown, 1=VT on, 2=VT off
sl_conout_mode:
     resd 1                                    ; GetConsoleMode scratch for the 24-bit probe, separate from the cache it feeds
sl_gc_buf:
     resb 1 + SL_GC_BUF_SIZE + 1               ; getchar: type byte, 1 + payload, 64 + NUL slot, 1, Intern reads from type byte
     resb 2                                    ; padding: keep sl_gc_got and sl_pipe_avail on this file's 4-byte BSS grid
sl_gc_got:
     resd 1                                    ; bytes returned by the last ReadFile
sl_ir_buf:
     resb SL_IR_SIZE                           ; INPUT_RECORD buffer for Peek/ReadConsoleInputA, 20 bytes on x64; the API writes a WHOLE record per nLength=1, so this MUST equal sizeof,INPUT_RECORD, an undersized buffer overruns into sl_pipe_avail
sl_pipe_avail:
     resd 1                                    ; PeekNamedPipe: bytes available, output
sl_gc_req:
     resd 1                                    ; ReadFile byte request: SL_GC_BUF_SIZE, or min,PipeNamedPipe avail, SL_GC_BUF_SIZE on the pipe path, a full-size request on a byte-mode pipe waits for that many bytes
     resd 1                                    ; padding
     align 16
sl_vq_row_buf:
     resd SL_VQ_MAX_ELEMS                      ; current codebook-row floats, n <= 256; the query vector stages on the caller's stack, re-entrancy
; --- Token type, set by GetToken, read by ReadLogoStream ---
sl_tok_type:
     resd 1                                    ; SL_TOK_ATOM / SL_TOK_STRING / etc.
; --- Garbage Collection state ---
sl_gc_freelist:
     resd 1                                    ; head of reclaimed cons freelist, 0 = empty
     resd 1                                    ; padding
sl_atom_freelist:
     resd 1                                    ; head of reclaimed atom free blocks: first-fit, [len][next] headers, 0 = empty
sl_atom_gc_restart_at:
     resd 1                                    ; atom-cursor value the next soft-limit collect must restart itself past, hysteresis
sl_gc_want_atoms:
     resd 1                                    ; 1 during an atom-half collect, set by Intern's allocation paths around GC_Collect
sl_gc_in_retry:
     resd 1                                    ; 1 once the current atom request has already had its pinned collect, bounds .atom_full at two collects
     align 8                                   ; keep the stack/GC qwords 8-aligned, resq does NOT auto-pad
; --- Private Logo stack state, sectorlogo_seperate_stack.md ---
sl_stk_base:
     resq 1                                    ; private block base, 0 = not yet allocated; zero-init makes the entry range check fall to the switch path
sl_stk_top:
     resq 1                                    ; switch target = base + SL_STACK_SIZE - 8, 8 mod 16: two ENTER pushes land at callee parity
sl_stk_limit:
     resq 1                                    ; overflow watermark = base + SL_STK_GUARD_MARGIN; every recursion entry cmps rsp against this
sl_stk_owner_tid:
     resd 1                                    ; thread id owning the switch, 0 = idle; never a real TID, GetCurrentThreadId never returns 0
sl_stk_old_protect:
     resd 1                                    ; VirtualProtect lpflOldProtect scratch, guard-page setup, output ignored
sl_gc_scan_ceiling:
     resq 1                                    ; private-stack RSP at the switching entry, GC scan ceiling, written ONLY by SL_STK_ENTER's switch path
sl_interp_park:
     resq 1                                    ; BSS parking for v7s_sectorlogo_interp's rcx across SL_STK_ENTER, alloc path overwrites volatiles; rbx must stay the CALLER's, ABI
sl_es_park:
     resq 1                                    ; BSS parking for v7s_sectorlogo_eval_script's rcx across SL_STK_ENTER, same pattern as on_token's sl_hook_* slots
sl_gc_bitmap:
     resb SL_GC_BITMAP_BYTES                   ; 125 KB mark bitmap, 1 bit per cons cell
sl_atom_start_bits:
     resb SL_ATOM_BITMAP_BYTES                 ; persistent atom enumeration index
sl_atom_mark_bits:
     resb SL_ATOM_BITMAP_BYTES                 ; per-collect atom liveness mark array
; --- FLOAT scratch, single dword, reused by promote/Equal stash paths ---
sl_float_tmp:
     resd 1                                    ; float stash across helper calls, single-threaded hook
     resd 1                                    ; padding
sl_concat_buf:
     resb SL_CONCAT_BUF_SIZE                   ; scratch for _ string concatenation
sl_xfn_escape_buf:
     resb SL_XFN_ESCAPE_SIZE                   ; DEFINE_COMMAND %s inner-escape scratch
sl_xfn_arena_cursor:
     resq 1                                    ; current cursor into sl_xfn_escape_buf, reentrancy-safe
sl_warn_seen:
     resd 32                                   ; 1024-bit hashed warning bitmap, atom granule index; diagnostics only
; --- String primitive state, derived pointers used during replacement ---
; Arg atoms are saved to stack-frame slots for reentrancy; only derived ptrs live here.
sl_str_ptr1:
     resq 1                                    ; STR_REPLACE: source string start pointer
sl_str_ptr2:
     resq 1                                    ; STR_REPLACE: old string pointer
sl_str_ptr3:
     resq 1                                    ; STR_REPLACE: new string pointer
     alignb 16                                 ; 16-byte align: SWAR 16-byte overreads, sl_swar_strlen vmovdqu on NIL/empty paths stay in-bounds
sl_str_nul:
     resb 16                                   ; 16 zero bytes, BSS zero-init; byte 0 is still the NUL sentinel
; --- FFI, C-CALL state ---
sl_v7_hmodule:
     resq 1                                    ; v7's own HMODULE for "SELF" FFI resolution
sl_caller_hmodule:
     resq 1                                    ; host EXE module handle for "HOST" FFI callbacks, set via v7s_set_caller_hmodule
; --- C-CALL DLL owner registry: NAME STRING -> HMODULE, the OWNERSHIP record ---
; LoadLibraryA acquires a loader reference per call, so the registry is the
; permanent ownership record, exactly ONE loader reference per distinct
; external DLL for the process lifetime, SectorLogo's world is
; process-lifetime; the OS reclaims the references at exit; a resolved native
; function pointer must also never outlive its module.Keyed by the name
; STRING, copied out of the atom heap, immune to atom GC and to interpreter
; re-initialization. Never freed. SELF/HOST never enter it, their handles
; were not acquired via LoadLibraryA here .
sl_ffi_owned_names:
     resb SL_FFI_OWNED_SLOTS * SL_FFI_OWNED_NAME_BYTES ; fixed name slots, NUL-terminated copy
sl_ffi_owned_mods:
     resq SL_FFI_OWNED_SLOTS                   ; owned HMODULE per slot
sl_ffi_owned_count:
     resd 1                                    ; populated slot count, only grows
sl_stream_base_pos:
     resd 1                                    ; base transformer c_pos at turn start
sl_stream_str_len:
     resd 1                                    ; current accumulated stream length, bytes
sl_stream_tok_count:
     resd 1                                    ; current accumulated token count
sl_stream_str2_len:
     resd 1                                    ; rewritten stream length in secondary buffer
sl_rewrite_count:
     resd 1                                    ; number of re-encoded suffix tokens
     resd 1                                    ; padding to 8-byte alignment
sl_stream_str_buf:
     resb SL_STREAM_BUF_SIZE                   ; primary detokenized UTF-8 stream, 1 MiB
sl_stream_str_buf2:
     resb SL_STREAM_BUF_SIZE                   ; secondary rewritten stream buffer, 1 MiB
sl_stream_tok_history:
     resd SL_STREAM_TOK_MAX                    ; token ID history, 1 MiB
sl_stream_tok_offset:
     resd SL_STREAM_TOK_MAX                    ; start byte offset per token, 1 MiB; the byte length is derivable as offset[i+1]-offset[i]
sl_rewrite_tok_buf:
     resd SL_REWRITE_TOK_MAX                   ; re-encoded replacement tokens, 64 KiB
; --- Hook control + action state ---
sl_hook_enabled:
     resd 1                                    ; gate flag: nonzero = hook active in chat loop
sl_evict_enabled:
     resd 1                                    ; gate flag: nonzero = an "on_stream_evict" script lives in the DB, the eviction flush delivers; the per-chat refresh re-probes the key
sl_action_code:
     resd 1                                    ; action code set by primitives, read by on_token
sl_local_count:
     resd 1                                    ; number of entries in the LOCAL set, moved next to action state for on_token's merged clears
