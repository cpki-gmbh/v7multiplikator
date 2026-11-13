; v7 - part of the v7 inference engine
; Copyright, C 2026 Görg Pflug & CPKI GmbH
; License: GPL v3, see LICENSE

%define SL_ATOM_AREA_SIZE 0x4000000            ; 64 MiB for interned atom string names, large strings
%define SL_ATOM_GUARD_PAD 16                   ; keep reserved guard bytes after the last atom; existing allocator geometry is unchanged
%define SL_ATOM_LIMIT (SL_ATOM_AREA_SIZE - SL_ATOM_GUARD_PAD) ; highest byte offset + 1 the atom allocator may hand out, the alloc bound check uses this, never AREA_SIZE
%define SL_EMPTY_ATOM_OFF 8                    ; offset of empty-string atom, NIL takes 0-1, 2-7 is granule padding
%define SL_ATOM_TYPE_OFF 0                     ; type byte offset within atom storage
%define SL_ATOM_TEXT_OFF 1                     ; content offset, skip type byte
%define SL_TYPE_WORD 0x01                      ; bare word / operator / identifier from source
%define SL_TYPE_STRING 0x02                    ; double-quoted string literal
%define SL_CONS_CELL_COUNT 1000000             ; 1,000,000 cons cells max
%define SL_CONS_CELL_BYTES 8                   ; 2 x dword, car + cdr per cell
%define SL_HEAP_TOTAL_SIZE (SL_ATOM_AREA_SIZE + SL_CONS_CELL_COUNT * SL_CONS_CELL_BYTES) ; total VirtualAlloc size, atom area + cons heap
%define SL_LOCAL_SET_MAX 65536                 ; max LOCAL declarations in the scope stack, tombstoned on block exit; deep recursion + Lisp-in-Logo needs headroom
%define SL_LOCAL_SET_SCALE 8                   ; dword pairs [var atom][binding cell index]: scale for sl_local_set addressing, stride 8; the cell index is a dword at SL_LOCAL_CELL_OFF
%define SL_LOCAL_CELL_OFF 4                    ; dword offset of the cached binding cell index within a set entry
%define SL_LOCAL_NEWEST_ADJ 1                  ; count-1 adjustment: newest-first scan start index, lea disp
%define SL_SORT_MAX_ELEMS 65536                ; sort record buffer capacity, elements
%define SL_RECORD_QWORDS 2                     ; 16-byte sort records: value dword + binary64 key qword
%define FPOW_BIG_E_BITS 0x4B800000             ; binary32 2^24: every finite value at/above is an EVEN integer
%define FPOW_INF_BITS 0x7F800000               ; binary32 infinity bit pattern, NaN magnitudes sort strictly above
%define SL_F32_MAX_FINITE 0x7F7FFFFF           ; largest finite binary32; "positive finite" == the unsigned bit range [1, this], so one pair of unsigned compares establishes the log domain
%define STN_NUM_PAYLOAD_MAX 536870911          ; largest positive NUMBER payload, 2^29 - 1
%define STN_NUM_NEG_BOUND 536870912            ; negative magnitude bound: |MIN| = 2^29 exactly, MIN = -536870912 fits
%define STN_NUM_MUL_LIMIT (STN_NUM_PAYLOAD_MAX / 10) ; 53687091: largest pre-multiply accumulator that can still reach 2^29-1
%define STN_NUM_MUL_OVER (STN_NUM_MUL_LIMIT + 1) ; 53687092: smallest pre-multiply accumulator that always overflows
%define SL_INTERN_FILTER_BITS 4194304          ; bloom bitfield size in bits, 2^22
%define SL_INTERN_FILTER_MASK 0x3FFFFF         ; bit index mask, FILTER_BITS - 1
%define SL_INTERN_SETS 4096                    ; Number of four-way cache sets
%define SL_INTERN_SET_SHL 5                    ; Shift from set index to byte offset
%define SL_INTERN_SET_BYTES (1 << SL_INTERN_SET_SHL) ; One hash vector and one offset vector
%define SL_INTERN_LANE_BYTES 4                 ; Each hash and atom offset is one dword
%define SL_INTERN_ATOMS_OFF (4 * SL_INTERN_LANE_BYTES) ; Atom offsets follow four hash dwords
%define SL_INTERN_VICTIM_SHIFT 30              ; Use the top two hash bits for replacement
%define SL_INTERN_CACHE_BYTES (SL_INTERN_SETS * SL_INTERN_SET_BYTES) ; Total cache size
%define SL_DISPATCH_MAX_OFF 8192               ; exclusive bound on raw atom byte offsets
%define SL_DISPATCH_TABLE_ENTRIES (SL_DISPATCH_MAX_OFF / SL_ATOM_GRANULE_BYTES) ; one qword per 8-byte granule
%define SL_CONS_TAG 0x80000000                 ; bit31 set = cons cell
%define SL_CONS_BIT 31                         ; bit index of the CONS tag, for btr tag strips
%define SL_NUMBER_TAG 0x40000000               ; bit30 set = number
%define SL_NUMBER_BIT 30                       ; bit index of the NUMBER tag, for BT tests that must read CF and leave ZF intact
%define SL_FLOAT_TAG 0xC0000000                ; bits31:30 both set = 30-bit truncated float
%define SL_TAG_MASK 0xC0000000                 ; mask for tag bits
%define SL_PAYLOAD_MASK 0x3FFFFFFF             ; mask for payload bits
%define SL_PAYLOAD_MAX 536870911               ; largest representable NUMBER payload value, 2^29 - 1
%define SL_PAYLOAD_MIN -536870912              ; smallest representable NUMBER payload value, -2^29
%define SL_ATOM_MIN_OFFSET 1                   ; lower bound of the accepted range, the first ALLOCATED atom is SL_EMPTY_ATOM_OFF = 8 after the NIL sentinel at 0-1 and the 2-7 granule padding, base of the one-range-check atom guards, ASSERT_ATOM_OR_NIL / ASSERT_ATOM_NONNIL
%define SL_PAYLOAD_SIGN_SHIFT64 34             ; shl/sar by, 64 - 30 to re-truncate a 64-bit result to the payload width
; --- FLOAT encoding helpers, top 30 bits of binary32; 2 low mantissa bits dropped ---
%define SL_FLOAT_SHIFT 2                       ; binary32 -> 30-bit payload shift, truncates 2 low bits
%define SL_FLOAT_ZERO_POS 0xC0000000           ; +0.0 exact tagged encoding, exponent+mantissa all zero
%define SL_FLOAT_ZERO_NEG 0xE0000000           ; -0.0 exact tagged encoding, sign bit29 set, rest zero
%define SL_F32_SIGN_MASK 0x80000000            ; binary32 sign BIT, 0x80000000, NOT its complement.For "is the magnitude zero?" use SL_F32_NOT_SIGN_MASK; the two are opposites and swapping them inverts a test silently.
%define SL_F32_MAN_MASK 0x007FFFFF             ; binary32 mantissa field mask
%define SL_F32_QNAN_BITS 0x7FC00000            ; canonical quiet NaN binary32 bits, sign 0, exp all-ones, MSB of mantissa
%define SL_F32_LOW_CLEAR 0xFFFFFFFC            ; clears 2 low mantissa bits, tag truncation
%define SF_FRAC_DIGITS 6                       ; fixed-notation fractional digits produced
%define SF_EDIG_DIGITS 5                       ; exponent-notation digits after the point
%define SF_ATOM_DIGIT_MAX 9                    ; int/frac digit cap in sl_atom_float, int32-safe parse
%define SL_FLOAT_MAX_TEXT 17                   ; max decimal text bytes, "-999999999.000000"
; --- Action codes returned by v7s_sectorlogo_on_token ---
%define SL_ACTION_RESAMPLE 1                   ; reject token, resample from logits
%define SL_ACTION_EOS 2                        ; force stop / end of generation
%define SL_ACTION_REWRITE 3                    ; rewritten stream, V7-SET-STREAM

%macro SL_STK_ENTER 0
     jmp %%entry                               ; hop over the cold blocks, layout per ENGINE_GATE_RAW
%%fatal_concurrent:
     FATAL "SectorLogo: Logo world entered from two threads, single-threaded contract" ; the unsynchronized world state makes anything else corruption
%%switch:
     cmp qword [rel sl_stk_base], 0            ; block not allocated yet?
     je %%alloc                                ; cold: first ever entry
     V7_THREAD_ID eax, rax                     ; eax = my thread id, never 0; still live at %%take, no reload needed there
     mov ecx, [rel sl_stk_owner_tid]           ; current owner, 0 = idle
     test ecx, ecx                             ; idle?
     jz %%take                                 ; yes: take it
     cmp ecx, eax                              ; Is the current thread the resource holder?
     je %%take                                 ; yes: re-entrant entry is legal
     jmp %%fatal_concurrent                    ; another thread owns the Logo world: contract violation
%%alloc:
     sub rsp, 8                                ; parity pad: entry is 8 mod 16; both calls below need 0 mod 16 at the call instruction
     mov ecx, SL_STACK_SIZE                    ; arg1: 64 MiB, kernel zero-commits every page
     call mdb_alloc_mem                        ; rax = Logo world stack block base
     mov [rel sl_stk_base], rax                ; publish block base
     lea rcx, [rax + SL_STK_TOP_OFF]           ; top: 8 mod 16 so the two pushes below land at callee parity
     mov [rel sl_stk_top], rcx                 ; publish switch target
     lea rcx, [rax + SL_STK_GUARD_MARGIN]      ; overflow watermark, margin above the raw base
     mov [rel sl_stk_limit], rcx               ; publish recursion watermark
     mov rcx, rax                              ; lpAddress = block base
     mov edx, SL_STK_GUARD_PAGE                ; dwSize = 4 KiB
     mov r8d, PAGE_NOACCESS                    ; flNewProtect
     lea r9, [rel sl_stk_old_protect]          ; lpflOldProtect, required out param, value ignored
     API_CALL [__imp_VirtualProtect], rcx, rdx, r8, r9 ; never fails on our own committed block
     add rsp, 8                                ; drop the parity pad
     jmp %%switch                              ; re-enter with the block present
%%take:
     mov [rel sl_stk_owner_tid], eax           ; I own the switch for this call chain, eax = TID from %%switch; plain store, single-thread contract
     mov rax, [rel sl_stk_top]                 ; 8 mod 16
     xchg rsp, rax                             ; rsp = private top; rax = old C rsp, return address still at [rax]
     push rax                                  ; switch frame qword 1: saved C rsp
     push qword 1                              ; switch frame qword 2: marker 1 = SWITCHED
     mov [rel sl_gc_scan_ceiling], rsp         ; GC scan ceiling: ONLY the switching entry anchors, nested-anchor fix
     jmp %%done                                ; body awaits, same target as the nested path
%%entry:                                       ; hot path starts here
     cmp rsp, [rel sl_stk_base]                ; below block base?, unsigned; sl_stk_base=0 pre-alloc makes every real rsp "above"
     jb %%switch                               ; yes: not on our stack -> switch
     cmp rsp, [rel sl_stk_top]                 ; above block top?
     ja %%switch                               ; yes: not on our stack -> switch
     push qword 0                              ; dummy, parity pair with the switch path's two pushes
     push qword 0                              ; marker 0 = NESTED: no switch, no anchor update, outer frames stay live below
%%done:                                        ; body continues at callee parity, rsp 8 mod 16
%endmacro

%macro SL_STK_LEAVE 0
     pop rcx                                   ; marker, switch frame qword 2; popped into rcx, rax holds the entry's return value
     test rcx, rcx                             ; switched?
     jz %%nested                               ; no: nested entry
     pop rsp                                   ; rsp := saved C rsp, POP RSP: the loaded value wins, the +8 increment is discarded, points at the return address
     mov dword [rel sl_stk_owner_tid], 0       ; release ownership, plain store; the TID check is a tripwire, not a lock
     jmp %%done                                ; back on the C stack: return address in place
%%nested:
     add rsp, 8                                ; drop the dummy qword
%%done:
%endmacro

%macro ASSERT_ATOM_OR_NIL 1
     cmp eax, SL_PAYLOAD_MASK                  ; NIL,0 and atoms, 1..0x3FFFFFFF, bits31:30 clear pass; everything above = NUMBER/CONS/FLOAT
     ja %1                                     ; any tag bit set: FATAL
%endmacro

%macro SL_DIGIT_IN 2
     movzx edx, byte %1                        ; load the candidate byte, unsigned 0..255: a high-bit byte must survive the wrap below
     sub dl, '0'                               ; ASCII -> digit; bytes below '0' wrap UNSIGNED above 9, which is what makes one compare enough
     cmp dl, 9                                 ; valid digit 0..9?, 0..9 accept; 10..255 reject, covering BOTH ends of the range
     ja %2                                     ; not a digit: jump to the caller's reject path
%endmacro

%macro ASSERT_ATOM_NONNIL 2
     lea ecx, [%1 - SL_ATOM_MIN_OFFSET]        ; atoms are 1-based: shift to 0-based, NIL wraps to max unsigned
     cmp ecx, SL_PAYLOAD_MASK - 1              ; above 0x3FFFFFFE = NIL,wrapped /NUMBER/CONS/FLOAT?
     ja %2                                     ; not a plain atom
%endmacro

%macro TAG_GUARD_ATOM 2
     test eax, eax                             ; NIL?
     jz %1                                     ; yes: caller-specific handling
     test eax, SL_TAG_MASK                     ; CONS/NUMBER/FLOAT tag bits set?
     jnz %2                                    ; yes: not a plain atom
%endmacro
; --- Buffer sizes ---
%define SL_SCRIPT_BUF_SIZE 0x100000            ; max script source length, 1 MiB; a clamped script is a corrupted script, so the bound FATALs.Usable text is one byte less, every copy site needs a NUL slot. Raise v7s_readfile's V7_EXEC_BUF_SIZE in lockstep: the autoexec file tier reads through it, so a smaller readfile buffer silently caps autoexec.logo well below this.
%define SL_TOK_BUF_SIZE 0x100000               ; token string buffer, 1 MiB — supports full stream/DB strings
%define SL_CONCAT_BUF_SIZE 0x200000            ; concatenation scratch buffer, 2 MiB — two 1 MiB streams
%define SL_CONCAT_BUF_LAST (SL_CONCAT_BUF_SIZE - 1) ; last writable byte offset in sl_concat_buf
%define SL_XFN_ESCAPE_SIZE (2 * SL_TOK_BUF_SIZE + 8) ; DEF_EXE_FN %s inner-escape destination: win32_escape_inner emits at most 2 bytes per source byte, quote -> backslash-quote, backslash run -> 2N plus the NUL, so a full SL_TOK_BUF_SIZE argument needs 2x+1; the +8 is headroom so the bound is never the binding constraint
%define SLFV_SLACK 2                           ; fatal-diagnostic slack: closing quote + NUL reserved below the slfv_append copy bound
%define SLFV_ESC_BYTES 4                       ; width of one \xNN control-byte escape in the fatal-diagnostic offender render
%define SLFV_ESC_HI_OFF 2                      ; byte offset of the high hex digit within the escape
%define SLFV_ESC_LO_OFF 3                      ; byte offset of the low hex digit within the escape
%define SL_TOK_BUF_END_OFF (SL_TOK_BUF_SIZE - 1) ; last writable byte offset in sl_tok_buf
%define SL_TOK_BUF_NUL_OFF (SL_TOK_BUF_SIZE - 2) ; last writable content byte, NUL terminator reserves 1 byte after this
%define SL_TOK_BUF_SIGN_PEEK 2                 ; offset of char after sign prefix in tok_buf, type byte + sign char = 2
; --- Literal-parser / float-print buffer geometry ---
%define SL_ATOI_PEEK_OFF 1                     ; offset to peek char after '-' or '.'
%define SL_ATOM_STARTER_BASE '+'               ; first-byte window BASE for the numeric-literal gate in .parse_integer, 0x2B = '+', the lowest numeric starter.The first byte is biased by SUBTRACTING this, so it must be the base character itself, writing a DIFFERENCE here shifts the window and every number is rejected
%define SL_ATOM_DIGIT9_MINUS_PLUS ('9' - '+')  ; that window's width, 14: the ja admits '+' ',' '-' '.' '/' '0'..'9', 0x2B..0x39, 15 chars and rejects every byte above '9', ':' included, ':' biases to 15, which is above this width, so the ja itself rejects it. The zero-digit test at .atoi_done exists for ',' '.' '/', which this window does admit
%define SL_ATOM_MINUS_BIAS ('-' - '+')         ; biased value of the '-' starter inside the window above, dl == 2 ; lets the sign tests compare the already-biased byte instead of reloading it
%define SL_POW10_DWORD 4                       ; pow10_table dword stride, indexed dword loads
; chat.inc, the v7s_chat output-capacity FATAL site is included BEFORE this
; file, the defines below do not reach it.
%define SL_STREAM_TOK_MAX 262144               ; max tokens in stream history, 256K
%define SL_REWRITE_TOK_MAX 16384               ; max re-encoded replacement tokens, 16K
%define SL_ONE_BYTE 1                          ; single byte offset, for named-constant arithmetic
%define SL_CONCAT_LHS_OFF 8                    ; rsp offset to lhs below rhs on stack, 2 pushes = 16 bytes, rhs=TOS at +0, lhs at +8
%define SL_MAX_CALL_DEPTH 65536                ; eval recursion cap, user-fn + DEF_EXE_FN calls ; 65536 x ~350-700 B fits the 64 MiB private stack with headroom
%define SL_MAX_PARSE_DEPTH 262144              ; parser recursion cap, one level per list element + one per nesting ; independent of the eval cap, flat data lists must parse deeper than fn recursion ever goes
; --- Private Logo stack, sectorlogo_seperate_stack.md ---
%define SL_STACK_SIZE 0x4000000                ; 64 MiB private Logo stack: ~65K eval levels x ~350 B, ~250K parser levels x ~48 B, both with headroom for C-CALL frames stacked on top
%define SL_STK_TOP_OFF (SL_STACK_SIZE - 8)     ; top = base + SIZE - 8: 8 mod 16, so the two ENTER pushes land at callee parity, mdb_alloc_mem blocks are 64 KiB page aligned = 0 mod 16
%define SL_STK_GUARD_MARGIN 0x10000            ; 64 KiB bottom margin: the overflow watermark sits this far above the raw base, below it: watermark-check-free engine C-CALL frames
%define SL_STK_GUARD_PAGE 0x1000               ; first 4 KiB of the block -> PAGE_NOACCESS: runaway recursion AVs into Zydis here, never silently into adjacent memory
%define PAGE_NOACCESS 0x01                     ; VirtualProtect protection constant, guard page
%define SL_TOK_ATOM 0                          ; bare word or number, also the default every other token falls back to
%define SL_TOK_STRING 1                        ; double-quoted string literal
%define SL_TOK_EOF 4                           ; end of input
; --- Combined token classes and ASCII name membership, sl_tok_class bytes ---
%define SL_TOK_BOUNDARY_ONLY 0x80              ; NUL, whitespace, quote, semicolon; handled before token dispatch
%define SL_TOK_ALNUM_MASK 0x40                 ; bit6 marks ASCII 0-9/A-Z/a-z as valid name characters
%define SL_TOK_BOUNDARY_MASK 0x8F              ; bit7 plus class IDs 1..11, excludes bit6, valid ASCII names are word chars
; --- Token operand classification, sl_last_tok_operand byte values ---
%define SL_TOK_OPERAND 1                       ; previous token was an operand, number, word, string, or closing bracket
%define SL_TOK_NOT_OPERAND 0                   ; previous token was an operator or delimiter
; --- FFI, C-CALL constants ---
%define SL_FFI_MAX_ARGS 16                     ; maximum C-call arguments
%define SL_FFI_SELF_DWORD 0x464C4553           ; "SELF" as little-endian dword, S=0x53,E=0x45,L=0x4C,F=0x46
%define SL_FFI_SELF_NUL 4                      ; byte offset of NUL after "SELF", 4 chars
%define SL_FFI_HOST_DWORD 0x54534F48           ; "HOST" as little-endian dword, H=0x48,O=0x4F,S=0x53,T=0x54
%define SL_FFI_HOST_NUL 4                      ; byte offset of NUL after "HOST", 4 chars
%define SL_FFI_QWORD_SCALE 8                   ; scale for qword array indexing
%define SL_FFI_SHADOW 32                       ; Win64 shadow space reserved in every C-call frame
%define SL_FFI_FRAME_ALIGN_BIAS (SL_FFI_SHADOW + 15) ; round, n-4 *8 + shadow,32 UP to the next 16-byte boundary, the and-mask turns the +15 bias into the round-up
%define SL_FFI_ARG2_OFF 8                      ; offset of arg2 in the invoke frame args[]
%define SL_FFI_ARG3_OFF 16                     ; offset of arg3 in the invoke frame args[]
%define SL_FFI_ARG4_OFF 24                     ; offset of arg4 in the invoke frame args[]
; --- Garbage collector constants ---
%define SL_GC_BITMAP_BYTES (SL_CONS_CELL_COUNT / 8 + 4) ; cons mark bitmap size, one bit per cons cell, dword access needs the +4 tail
%define SL_ATOM_GRANULE_BYTES 8                ; one bit of either bitmap = 8 bytes of atom area
%define SL_ATOM_BITMAP_BYTES (SL_ATOM_AREA_SIZE / 64 + 4) ; 1048580 bytes = 1048576 exact bytes, 8388608 bits, one per 8-byte granule of the 64 MiB SL_ATOM_AREA_SIZE + 4 bytes that are LOAD-BEARING, not slack. NOT the same EA rule as SL_GC_BITMAP_BYTES: SL_ATOM_BIT pre-divides the offset, shr edx, 6 = bitmap BYTE offset, shr eax,3 + and 7 = bit-in-byte 0..7, so max atom offset 0x3FFFFFF gives byte 0xFFFFF = 1048575 with the dword access spanning 1048575..1048578, the bare /64 sizing would be one word short; the +4 covers the tail.
%define SL_ATOM_FB_HDR 8                       ; header byte count
%define SL_ATOM_FB_LEN 0                       ; [+0] block byte length, a non-zero multiple of SL_ATOM_GRANULE_BYTES
%define SL_ATOM_FB_LINK 4                      ; [+4] next block offset inside the arena, 0 = end of list
; An atom's stored byte count is type byte + text + NUL, SL_ATOM_TYPE_OFF is 0,
; SL_ATOM_TEXT_OFF is 1, so its span is strlen + this many bytes.
%define SL_ATOM_HDR_BYTES 2                    ; type byte + NUL terminator, the strlen is added on top
%define SL_ATOM_GC_SOFT_OFF 0x3000000          ; 48 MiB, 75 % : first soft-limit collect point; + RESTART = 56 MiB, 8 MiB headroom
%define SL_ATOM_GC_RESTART_BYTES 0x800000      ; 8 MiB between restart collect points, the window runs against SL_ATOM_AREA_SIZE, see the constraint above, NOT SL_HEAP_TOTAL_SIZE, which adds the cons heap and would overstate the headroom by 8 MB
%define SL_ATOM_INIT_END 16                    ; first free atom offset after the two sentinels: align8,SL_EMPTY_ATOM_OFF + 2
%if SL_ATOM_GC_SOFT_OFF + SL_ATOM_GC_RESTART_BYTES > SL_ATOM_AREA_SIZE
%error atom GC window (SOFT_OFF + RESTART_BYTES) must fit in SL_ATOM_AREA_SIZE, or the restart point cannot suppress the soft test and every append collects ; drift check
%endif
%if (SL_INTERN_SETS & (SL_INTERN_SETS - 1)) || SL_INTERN_SET_BYTES != 2 * SL_INTERN_ATOMS_OFF
%error Intern cache needs power-of-two sets and matching hash and offset vectors ; Check the set index and stride
%endif

%macro SL_ATOM_BIT 2
     mov edx, eax                              ; edx = atom offset
     shr eax, 3                                ; eax = granule index
     shr edx, 6                                ; edx = bitmap BYTE offset, byte b holds granules 8b..8b+7, so it is granule/8 = offset/64
     and eax, 7                                ; eax = bit inside that byte, must be < 8: zero address contribution
     lea r9, [rel %1]                          ; r9 = bitmap base, RIP-relative has no index form
%2 dword [r9 + rdx], eax                       ; set / clear / test the START or MARK bit
%endmacro
; SL_ALIGN_ROUND, the addend that turns x into align8,x, x + 7, then mask
%define SL_ALIGN_ROUND 7                       ; align8 round-up addend

%macro SL_ALIGN8 1
     lea %1, [%1 + SL_ALIGN_ROUND]             ; round up
     and %1, -8                                ; align to the granule
%endmacro
; --- NearestMatch, vector quantization limits ---
%define SL_VQ_MAX_ELEMS 256                    ; max vector / codebook-row components, staged into BSS float buffers
; --- Trace, EnableTrace primitive ---
%define SL_TRACE_MAX_LEVEL 3                   ; max EnableTrace level, 0=off, 1=stmts, 2=exprs, 3=verbose
%define SL_TRACE_MAX_INDENT 40                 ; max indent depth, 2-space indent = 80 chars at max depth
%define SL_FFI_OWNED_SLOTS 32                  ; owner-registry entries, distinct external DLLs for the process lifetime
%define SL_FFI_OWNED_NAME_BYTES 128            ; bytes per owner name slot, NUL-terminated DLL-name copy
%define SL_FFI_OWNED_HMOD_SCALE 8              ; qword HMODULE stride in the mods array, legal x64 index scale
; --- Win32 constants ---
%define SL_STD_OUTPUT_HANDLE -11               ; GetStdHandle arg for stdout
%define SL_STD_INPUT_HANDLE -10                ; GetStdHandle arg for stdin
%define SL_CONMODE_LINE 2                      ; console mode: ENABLE_LINE_INPUT
%define SL_CONMODE_ECHO 4                      ; console mode: ENABLE_ECHO_INPUT
%define SL_CONMODE_WINDOW 8                    ; console mode: ENABLE_WINDOW_INPUT
%define SL_CONMODE_MOUSE 0x10                  ; console mode: ENABLE_MOUSE_INPUT
%define SL_ENABLE_VT 4                         ; console output mode: ENABLE_VIRTUAL_TERMINAL_PROCESSING
%define SL_GC_BUF_SIZE 64                      ; getchar raw-read request size, payload bytes; the BSS buffer allocates +1 for the NUL terminator
%define SL_FILE_TYPE_CHAR 2                    ; GetFileType: character device, console
%define SL_FILE_TYPE_DISK 1                    ; GetFileType: real disk file. NOT distinguishable from UNKNOWN, 0 by the return value alone, after a 0 return, GetLastError, ==NO_ERROR is the documented test, see sl_gi_lazy_fixup
%define SL_FILE_TYPE_PIPE 3                    ; GetFileType: pipe or redirected
%define SL_HKIND_OTHER 1                       ; sl_conin_fixed: regular file, the plain ReadFile path. A disk read cannot block, so it is safe in a per-token hook
%define SL_HKIND_CONSOLE 2                     ; sl_conin_fixed: console input handle
%define SL_HKIND_PIPE 3                        ; sl_conin_fixed: pipe, redirected stdin
%define SL_HKIND_NONE 4                        ; sl_conin_fixed: unclassifiable handle, ReadKey returns NIL
%define SL_KEY_EVENT 1                         ; INPUT_RECORD.EventType: KEY_EVENT
%define SL_IR_EVENTTYPE_OFF 0                  ; INPUT_RECORD: EventType offset, WORD
%define SL_IR_KEY_BDOWN_OFF 4                  ; absolute offset of KEY_EVENT_RECORD.bKeyDown: EventType is a WORD at +0 and the Event union aligns to 4, its BOOL/DWORD members, so the union starts at +4 and bKeyDown is its first member. There is no wKeyDown/wFlags field
%define SL_IR_KEY_ASCII_OFF 14                 ; absolute offset of the uChar union, AsciiChar low byte / UnicodeChar full word = Event,+4 + 10. KEY_EVENT_RECORD member order, wincontypes.h : bKeyDown 0, 4, wRepeatCount 4, 2, wVirtualKeyCode 6, 2, wVirtualScanCode 8, 2, uChar 10, 2, dwControlKeyState 12, 4.Read as a WORD: AsciiChar is 0 for every non-ASCII key and the character then lives in UnicodeChar at the same offset
%define SL_IR_SIZE 20                          ; sizeof,INPUT_RECORD on x64 = 20: WORD EventType, 2 + 2 pad + a 16-byte 4-ALIGNED Event union. Every union member is 4-aligned and <= 16 bytes, KEY_EVENT_RECORD 16, MOUSE_EVENT_RECORD 16 = COORD 4 + 3 DWORDs, WINDOW_BUFFER_SIZE_RECORD 4, MENU_EVENT_RECORD 4, FOCUS_EVENT_RECORD 4, so the union is 16 and the record is 20. MOUSE_EVENT_RECORD has NO dwExtraInfo, that ULONG_PTR belongs to MOUSEINPUT, SendInput, a different union
%if SL_IR_SIZE != 20
%error SL_IR_SIZE must equal sizeof(INPUT_RECORD) on x64: WORD EventType + 2 pad + a 16-byte 4-ALIGNED Event union ; drift check
%endif
%if SL_IR_KEY_ASCII_OFF - SL_IR_KEY_BDOWN_OFF != 10
%error KEY_EVENT_RECORD: uChar must sit 10 bytes past bKeyDown (bKeyDown 0(4), wRepeatCount 4(2), wVirtualKeyCode 6(2), wVirtualScanCode 8(2), uChar is the union's FIFTH member ; +12 would be dwControlKeyState) ; drift check
%endif
%if SL_IR_KEY_ASCII_OFF != 14
%error INPUT_RECORD.Event.KeyEvent.uChar must be at absolute offset 14 (Event union at +4, uChar at union +10) ; drift check
%endif
%define SL_CTX_TRANSFORMER 0                   ; qword: Transformer*
%define SL_CTX_TOKENIZER 8                     ; qword: Tokenizer*
%define SL_CTX_OUT_TOKENS 32                   ; qword: int* out_tokens, token history
; --- Config struct offsets, embedded in Transformer at offset 0 ---
%define SL_CFG_DIM 8                           ; int: model dimension
%define SL_CFG_N_LAYERS 16                     ; int: number of layers
%define SL_CFG_VOCAB_SIZE 28                   ; int: vocabulary size
%define SL_CFG_SEQ_LEN 32                      ; int: sequence length
%define SL_TOK_VOCAB 0                         ; char**, 8 bytes
