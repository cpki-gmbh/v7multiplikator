; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

; ============================================================================
; dispatch_list.inc, the SINGLE SOURCE OF TRUTH for primitive dispatch

; Two dispatchers, EvalLogoStream = statement position, EvalPrimary =
; expression position route atoms to handlers. sectorlogo_eval_dispatch.inc
; expands the list into relative records. sl_dispatch_fill resolves those
; records into the statement and expression tables after interning the atoms.

; Handler labels are FULLY QUALIFIED, Dispatcher.local so statement and
; expression entries resolve to the same implementation labels; aliases
; intentionally share handlers.

; do not hand-edit the eval dispatchers anymore, add/change entries here.

; The ';' column on every entry must name the live interned word, the string inside
; the matching str_sl_* atom in sectorlogo_data.inc, not the C symbol name, and not
; the pre-rename spelling. renamed the named primitives to PascalCase whole
; words, and a dead name self-evaluates silently so a script author
; copying one gets no error and a desynced stream. After ANY rename, re-derive this
; column from sectorlogo_data.inc, never from the C symbol names.
; ============================================================================

; --- statement-position entries, EvalLogoStream ---
     STMT_ENTRY atom_sl_make, EvalLogoStream.exec_make ; make: bind or update a variable
     STMT_ENTRY atom_sl_local, EvalLogoStream.exec_local ; local: declare a function-scope variable
     STMT_ENTRY atom_sl_repeat, EvalLogoStream.exec_repeat ; repeat: counted or endless loop
     STMT_ENTRY atom_sl_printraw, EvalLogoStream.exec_printraw ; printraw: console output without CRLF
     STMT_ENTRY atom_sl_if, EvalLogoStream.exec_if ; if: single-branch conditional
     STMT_ENTRY atom_sl_ifelse, EvalLogoStream.exec_ifelse ; ifelse: two-branch conditional
     EXPR_ENTRY atom_sl_ifelse, EvalPrimary.p_ifelse_val ; IFELSE, value form : chosen block's value
     STMT_ENTRY atom_sl_stop, EvalLogoStream.exec_stop ; stop: break out of the innermost REPEAT
     STMT_ENTRY atom_sl_continue, EvalLogoStream.exec_continue ; continue: skip to the next loop iteration
     STMT_ENTRY atom_sl_terminate, EvalLogoStream.exec_terminate ; exit: exit the entire script
     STMT_ENTRY atom_sl_run, EvalLogoStream.exec_run ; run: evaluate a parsed block
     STMT_ENTRY atom_sl_def_fn, EvalLogoStream.exec_def_fn ; define: define a Logo function
     STMT_ENTRY atom_sl_def_exe_fn, EvalLogoStream.exec_def_exe_fn ; definecommand: define an external-tool function
     STMT_ENTRY atom_sl_output, EvalLogoStream.exec_output ; output: return a value from a function
     STMT_ENTRY atom_sl_v7_temp, EvalLogoStream.exec_v7_temp ; settemperature: set the sampler temperature
     STMT_ENTRY atom_sl_v7_topp, EvalLogoStream.exec_v7_topp ; setnucleus: set the nucleus top-p
     STMT_ENTRY atom_sl_v7_db_put, EvalLogoStream.exec_v7_db_put ; databaseset: write a microdb key
     STMT_ENTRY atom_sl_v7_set_stream, EvalLogoStream.exec_v7_set_stream ; setstream: rewrite the token stream
     STMT_ENTRY atom_sl_v7_set_token, EvalLogoStream.exec_v7_set_token ; setoutputtoken: replace the current candidate token
     STMT_ENTRY atom_sl_v7_reject, EvalLogoStream.exec_v7_reject ; reject: veto the current token
     STMT_ENTRY atom_sl_v7_halt, EvalLogoStream.exec_v7_halt ; halt: stop generation
     STMT_ENTRY atom_sl_c_call, EvalLogoStream.exec_c_call_st ; calllibrary: FFI statement, result discarded
     STMT_ENTRY atom_sl_setcar, EvalLogoStream.exec_setcar_st ; setcar: destructive car write, result discarded
     STMT_ENTRY atom_sl_setcdr, EvalLogoStream.exec_setcdr_st ; setcdr: destructive cdr write, result discarded
     STMT_ENTRY atom_sl_samevalue, EvalLogoStream.exec_samevalue_st ; samevalue: identity compare, result discarded
     STMT_ENTRY atom_sl_asword, EvalLogoStream.exec_asword_st ; asword: string->word conversion, result discarded
; PAIRS, nine atoms hold BOTH a STMT_ENTRY and an EXPR_ENTRY, and BOTH rows are
; load-bearing: the keyword alone picks the table, STMT_ENTRY -> sl_dispatch_stmt_table,
; EXPR_ENTRY -> sl_dispatch_expr_table, sectorlogo_eval_dispatch.inc:129/:137, so a row's
; position in this file is cosmetic, what breaks a pair is DELETING a row or changing its
; keyword. A primitive left without its statement row falls back to
; EvalLogoStream.stmt_fallback, which self-evaluates, sectorlogo_eval.inc:78-79 : the name
; then parses and does nothing, with no assembler error and no validator failure.
; IFELSE, the EXPR_ENTRY is a genuinely different handler, p_ifelse_val:
; the chosen block's value ; it is the only EXPR_ENTRY in the
; statement block.
; RUN, CALLLIBRARY, SETCAR, SETCDR, SAMEVALUE, ASWORD
; "statement discards the result" pairs: the STMT_ENTRY routes to an
; exec_* handler that throws the value away, the EXPR_ENTRY is the
; value-producing form. None of these six is noted at its own row.
; ENABLETRACE, ENABLESTRICTNAMES
; "expression returns the OLD value" pairs; these two STMT_ENTRY rows
; are the only statement entries in the expression block.
; --- expression-position entries, EvalPrimary ---
     EXPR_ENTRY atom_sl_lparen, EvalPrimary.eval_paren ; : parenthesized sub-expression
     EXPR_ENTRY atom_sl_rparen, EvalPrimary.eval_rparen_skip ; : skip a stray closer token
     EXPR_ENTRY atom_sl_run, EvalPrimary.p_run_eval ; run: evaluate a parsed block
     EXPR_ENTRY atom_sl_plus, EvalPrimary.p_sum ; +: symbol addition
     EXPR_ENTRY atom_sl_minus_sym, EvalPrimary.p_diff ; -: symbol subtraction
     EXPR_ENTRY atom_sl_mul_sym, EvalPrimary.p_prod ; *: symbol multiplication
     EXPR_ENTRY atom_sl_div_sym, EvalPrimary.p_quot ; /: symbol division
     EXPR_ENTRY atom_sl_mod_sym, EvalPrimary.p_rem ; %: symbol modulo
     EXPR_ENTRY atom_sl_power, EvalPrimary.p_pow ; power: exponentiation
     EXPR_ENTRY atom_sl_sqrt, EvalPrimary.p_sqrt ; squareroot: square root
     EXPR_ENTRY atom_sl_abs, EvalPrimary.p_abs ; absolutevalue: absolute value
     EXPR_ENTRY atom_sl_minus, EvalPrimary.p_neg ; negate: arithmetic negation
     EXPR_ENTRY atom_sl_max, EvalPrimary.p_max ; maximum: two-argument maximum
     EXPR_ENTRY atom_sl_min, EvalPrimary.p_min ; minimum: two-argument minimum
     EXPR_ENTRY atom_sl_random, EvalPrimary.p_random ; random: deterministic AES-NI random
     EXPR_ENTRY atom_sl_real_random, EvalPrimary.p_real_random ; realrandom: RDTSC entropy toggle
     EXPR_ENTRY atom_sl_i2f, EvalPrimary.p_i2f ; asfloat: integer to float conversion
     EXPR_ENTRY atom_sl_f2i, EvalPrimary.p_f2i ; asinteger: float to integer truncation
     EXPR_ENTRY atom_sl_floor, EvalPrimary.p_floor ; floor: round toward minus infinity
     EXPR_ENTRY atom_sl_fround, EvalPrimary.p_fround ; round: round half to even
     EXPR_ENTRY atom_sl_floatp, EvalPrimary.p_floatp ; isfloat: FLOAT type predicate
     EXPR_ENTRY atom_sl_flog2, EvalPrimary.p_flog2 ; logbasetwo: x87 log base 2
     EXPR_ENTRY atom_sl_fexp2, EvalPrimary.p_fexp2 ; poweroftwo: x87 two-to-the-x
     EXPR_ENTRY atom_sl_fln, EvalPrimary.p_fln ; naturallog: natural logarithm
     EXPR_ENTRY atom_sl_fmod, EvalPrimary.p_fmod ; floatremainder: x87 float remainder
     EXPR_ENTRY atom_sl_and_sym, EvalPrimary.p_bitand ; &: symbol bitwise AND
     EXPR_ENTRY atom_sl_or_sym, EvalPrimary.p_bitor ; |: symbol bitwise OR
     EXPR_ENTRY atom_sl_xor_sym, EvalPrimary.p_bitxor ; ^: symbol bitwise XOR
     EXPR_ENTRY atom_sl_not_sym, EvalPrimary.p_bitnot ; ~: prefix bitwise NOT, prefix-only, no infix form
     EXPR_ENTRY atom_sl_shl_sym, EvalPrimary.p_lshift ; <<: symbol left shift
     EXPR_ENTRY atom_sl_shr_sym, EvalPrimary.p_rshift ; >>: symbol right shift
     EXPR_ENTRY atom_sl_eq_sym, EvalPrimary.p_equalp ; =: symbol equality
     EXPR_ENTRY atom_sl_neq_sym1, EvalPrimary.p_notequalp ; <>: symbol inequality
     EXPR_ENTRY atom_sl_neq_sym2, EvalPrimary.p_notequalp ; !=: alternate symbol inequality
     EXPR_ENTRY atom_sl_samevalue, EvalPrimary.p_samevalue ; samevalue: tagged-word identity
     EXPR_ENTRY atom_sl_asword, EvalPrimary.p_asword ; asword: string -> folded WORD atom
     EXPR_ENTRY atom_sl_gt_sym, EvalPrimary.p_greaterp ; >: symbol greater-than
     EXPR_ENTRY atom_sl_lt_sym, EvalPrimary.p_lessp ; <: symbol less-than
     EXPR_ENTRY atom_sl_gte_sym, EvalPrimary.p_gte ; >=: symbol at-least compare, prefix name: AtLeast -> atleast
     EXPR_ENTRY atom_sl_lte_sym, EvalPrimary.p_lte ; <=: symbol at-most compare, prefix name: AtMost -> atmost
; && / || are infix only: no expression-table entry exists for their atoms, so a
; prefix spelling is a dead name that self-evaluates, the
; removed-prefix doctrine. ! stays prefix: p_bang binds ONE primary.
     EXPR_ENTRY atom_sl_bang, EvalPrimary.p_bang ; !: logical negation, prefix symbol
     EXPR_ENTRY atom_sl_zerop, EvalPrimary.p_zerop ; iszero: zero predicate
     EXPR_ENTRY atom_sl_numberp, EvalPrimary.p_numberp ; isnumber: numeric predicate
     EXPR_ENTRY atom_sl_listp, EvalPrimary.p_listp ; islist: cons cell predicate
     EXPR_ENTRY atom_sl_isstringatom, EvalPrimary.p_isstringatom ; isstringatom: STRING type-byte predicate
     EXPR_ENTRY atom_sl_iswordatom, EvalPrimary.p_iswordatom ; iswordatom: WORD type-byte predicate
     EXPR_ENTRY atom_sl_first, EvalPrimary.p_first ; first: head of a list
     EXPR_ENTRY atom_sl_butfirst, EvalPrimary.p_butfirst ; butfirst: tail of a list
     EXPR_ENTRY atom_sl_fput, EvalPrimary.p_fput ; prepend: prepend onto a list
     EXPR_ENTRY atom_sl_substr, EvalPrimary.p_substr ; part: substring extraction
     EXPR_ENTRY atom_sl_find, EvalPrimary.p_find ; search: substring position, 0-based index or NIL
     EXPR_ENTRY atom_sl_strlower, EvalPrimary.p_strlower ; lowercase: lowercase conversion
     EXPR_ENTRY atom_sl_strupper, EvalPrimary.p_strupper ; uppercase: uppercase conversion
     EXPR_ENTRY atom_sl_strreplace, EvalPrimary.p_strreplace ; replace: replace first occurrence
     EXPR_ENTRY atom_sl_strtonum, EvalPrimary.p_strtonum ; texttonumber: parse text as number
     EXPR_ENTRY atom_sl_strlen, EvalPrimary.p_strlen ; textlength: string byte length
     EXPR_ENTRY atom_sl_explode, EvalPrimary.p_explode ; split: split on separator
     EXPR_ENTRY atom_sl_str_tokens, EvalPrimary.p_str_tokens ; words: alnum-run tokenizer
     EXPR_ENTRY atom_sl_split_ngrams, EvalPrimary.p_split_ngrams ; ngrams: sliding n-gram windows
     EXPR_ENTRY atom_sl_lindex, EvalPrimary.p_lindex ; item: list element by index
     EXPR_ENTRY atom_sl_hash_float, EvalPrimary.p_hash_float ; hashnumber: CRC32C hash to float
     EXPR_ENTRY atom_sl_sort, EvalPrimary.p_sort ; sort: ascending sort
     EXPR_ENTRY atom_sl_reverse, EvalPrimary.p_reverse ; reverse: list reversal
     EXPR_ENTRY atom_sl_length, EvalPrimary.p_length ; length: list element count
     EXPR_ENTRY atom_sl_lset, EvalPrimary.p_lset ; setitem: overwrite list element
     EXPR_ENTRY atom_sl_setcar, EvalPrimary.p_setcar ; setcar: destructive car-field write
     EXPR_ENTRY atom_sl_setcdr, EvalPrimary.p_setcdr ; setcdr: destructive cdr-field write
     EXPR_ENTRY atom_sl_lfind, EvalPrimary.p_lfind ; finditem: scan for exact key
     EXPR_ENTRY atom_sl_lfindge, EvalPrimary.p_lfindge ; findfirstatleast: first element >= key
     EXPR_ENTRY atom_sl_lfindle, EvalPrimary.p_lfindle ; findfirstatmost: first element <= key
     EXPR_ENTRY atom_sl_int_to_char, EvalPrimary.p_int_to_char ; character: int to one-char string
     EXPR_ENTRY atom_sl_char_to_int, EvalPrimary.p_char_to_int ; number: first char to int
     EXPR_ENTRY atom_sl_do_vq, EvalPrimary.p_do_vq ; nearestmatch: vector quantization search
     EXPR_ENTRY atom_sl_c_call, EvalPrimary.p_c_call ; calllibrary: FFI invocation
     EXPR_ENTRY atom_sl_v7_encode_word, EvalPrimary.p_v7_encode_word ; encodeword: token text to id
     EXPR_ENTRY atom_sl_v7_db_get, EvalPrimary.p_v7_db_get ; databaseget: microdb lookup
     EXPR_ENTRY atom_sl_v7_get_token, EvalPrimary.p_v7_get_token ; gettoken: token at position
     EXPR_ENTRY atom_sl_v7_pos, EvalPrimary.p_v7_pos ; position: current output position
     EXPR_ENTRY atom_sl_v7_get_token_pos, EvalPrimary.p_v7_get_token_pos ; gettokenposition: token-stream position of the current token, query-local
     EXPR_ENTRY atom_sl_v7_vocab_size, EvalPrimary.p_v7_vocab_size ; vocabularysize: vocabulary size
     EXPR_ENTRY atom_sl_v7_dim, EvalPrimary.p_v7_dim ; modeldimension: model dimension
     EXPR_ENTRY atom_sl_v7_layers, EvalPrimary.p_v7_layers ; layercount: layer count
     EXPR_ENTRY atom_sl_v7_eos, EvalPrimary.p_v7_eos ; endtoken: end-of-sequence token id
     EXPR_ENTRY atom_sl_v7_stream, EvalPrimary.p_v7_stream ; getstream: current token stream text
     EXPR_ENTRY atom_sl_v7_str_len, EvalPrimary.p_v7_str_len ; streamlength: stream byte length
     EXPR_ENTRY atom_sl_getchar, EvalPrimary.p_getchar ; readkey: non-blocking stdin poll
     EXPR_ENTRY atom_sl_console_24bit, EvalPrimary.p_console_24bit ; enablecolors: VT probe and enabler
; ENABLETRACE / ENABLESTRICTNAMES: family c in the PAIRS note above.
     STMT_ENTRY atom_sl_enabletrace, EvalLogoStream.exec_enabletrace ; enabletrace: set trace level
     EXPR_ENTRY atom_sl_enabletrace, EvalPrimary.p_enabletrace ; enabletrace: expression form returns old level
     STMT_ENTRY atom_sl_enablestrictnames, EvalLogoStream.exec_enablestrictnames ; enablestrictnames: set the strict-name flag
     EXPR_ENTRY atom_sl_enablestrictnames, EvalPrimary.p_enablestrictnames ; enablestrictnames: expression form returns old flag
     STMT_ENTRY atom_sl_real_random, EvalLogoStream.exec_realrandom ; realrandom: set the RDTSC-entropy flag
     EXPR_ENTRY atom_sl_real_random, EvalPrimary.p_real_random ; realrandom: expression form returns old flag
