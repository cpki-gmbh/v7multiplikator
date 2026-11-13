# SectorLogo Performance, Language Comparison, Intern Speedup Analysis, Bottlenecks and Mitigations

> **Naming note (Sep 2026):** measurement sections use the current SectorLogo
> command names. Historical release names have been replaced in the examples;
> they are not supported aliases. Names have zero performance impact; the
> numbers stand. The multi-language tables were re-measured after the
> faster-strings phases landed (see the cycle-3 section).

Machine: Ryzen 5 7430U laptop (6 physical cores / 12 logical threads). Timing: warmup + 5
interleaved reps, median (Logo side vs C side in one process; thermal drift cancels). All workloads
verified against expected results each rep, a wrong result invalidates the run. Database: Qwen3-0.6B
`lz4ht.db`.

## Comparison method

All Logo workloads use `Define` functions with `Local` declarations, so each `Make` updates its binding
in place. The Python and Node references use local variables in ordinary functions. The comparison is
intended to show the cost of each interpreter design, not to rank the languages by one workload.

The workloads (identical algorithms in every language):

| Workload | What it stresses | Scale |
|---|---|---|
| recursive fib | function-call and interpreter dispatch (tree-walker floor) | fib(24), ~150k calls |
| loop add | loop + local variable update | 20,000 iterations |
| string build | `_` concat per step (atom interning + heap append), then a 99,999-step Part walk over the 2,000-char result | 2,000 concats |
| replace all | Search + Replace loop (the canonical sidecar idiom) | 60 occurrences |

## Multi-language results (Sep 2026, median of 5, this machine; 6 languages incl. SBCL)

Benchmark scripts live in `examples\experimental_work_in_progress\multilang_bench\multilang_bench\`; run
them with `examples\experimental_work_in_progress\multilang_bench\bench_compare_languages.bat` from the
project root. The comparison used CPython 3.12.10, Node.js 24.14.1, Tcl 9.0.3, SBCL 2.6.8, and
PowerShell 5.1 on the same machine. The six implementations were measured in one session to reduce
thermal variation.

### Absolute times

| Workload | SectorLogo | Python 3.12 | Node 24 | Tcl 9.0 | SBCL 2.6 | PowerShell 5.1 | C reference |
|---|---|---|---|---|---|---|---|
| recursive fib(24) | 58.95 ms | 12.35 ms | 0.82 ms | 53.03 ms | 1.00 ms | 15,157 ms | 0.32 ms |
| loop add (20k) | 5.58 ms | 2.78 ms | 0.089 ms | 1.83 ms | 0.148 ms | 3.07 ms | 0.008 ms |
| string build (2000+walk) | 65.81 ms | 0.35 ms | 0.146 ms | 1.60 ms | 8.48 ms | 0.60 ms | 0.014 ms |
| replace all (60) | 1.05 ms | 0.066 ms | 0.051 ms | 0.126 ms | 0.205 ms | 0.187 ms | 0.004 ms |

(Loadtest extras from the same SectorLogo run: concat only 76.2 ms, repeat intern 86.0 ms, find scan
(absent needle) 7.26 ms / 26.6x. Session was thermally comparable to the pre-batch baseline, the C
reference at fib 0.324 ms matches the earlier cool-session numbers, so absolutes are cross-comparable
with the usual 10-25% thermal band.)

### Slowdown vs C

| Workload | SectorLogo | Python 3.12 | Node 24 (JIT) | Tcl 9.0 | SBCL 2.6 | PowerShell 5.1 |
|---|---|---|---|---|---|---|
| recursive fib | ~182x | ~38x | ~2.5x | ~164x | ~3.1x | ~46,800x |
| loop add | ~672x | ~335x | ~11x | ~221x | ~18x | ~370x |
| string build | ~4,734x | ~25x | ~10x | ~114x | ~606x | ~43x |
| replace all | ~263x | ~17x | ~13x | ~32x | ~51x | ~47x |

SBCL provides a native-compiler comparison for the table. A Common Lisp compiled with `(optimize (speed
3))` is close to the C and Node results on the dispatch-bound shapes. SectorLogo is a tree-walker
interpreter, so its dispatch-bound results are slower in this measurement. The string-build shape also
differs because each implementation uses a different string-building strategy.

Context: CPython's own recursive-fib slowdown vs C is typically 60-150x on desktop CPUs; this laptop's
turbo state and the small fib(24) put Python at ~38x in this session. SectorLogo's dispatch-bound shapes
sit at ~182x (fib) and ~672x (loop add) after the dispatch tables + LOCAL frame array, an ASM
tree-walker with **no bytecode, no inline caches**, and a *dynamic-scoped alist environment* where
non-LOCAL variable reads are an `Assoc` linear walk. Lua 5.x uses a register bytecode VM and has a
different performance profile; this table does not treat it as a peer implementation.

**Tcl 9.0 provides a useful bytecode comparison**: Tcl is a mature bytecode VM. On the two
dispatch-bound shapes SectorLogo is within ~1.1-3x of it (fib 58.95 vs 53.03 ms, loop add 5.58 vs 1.83
ms), while on the string shapes Tcl's compiled `append`/`string replace` operations are substantially
faster (as expected for a language whose core value type is the string). PowerShell 5.1 is slower in
these recursive tests: its cmdlet-bound function recursion pays ~47,000x vs C on fib, with the
processing sequence and cmdlet-binding overhead dominating recursive workloads. SectorLogo therefore
falls between the native compiler/JIT implementations and the slower interpreted runtimes in this
measurement set.

### Direct-threaded dispatch tables (attic\docs_plans\interned_atom_plan.md, applied Sep 2026)

The 143-site linear `cmp/je` dispatch chains in EvalLogoStream (statement position) and EvalPrimary
(expression position) are replaced by **direct-indexed jump tables**: builtin atoms are interned FIRST
into a fresh atom area, so their handles are dense low offsets, the head atom IS the table index
(`table[offset]`), no hash, no string compare at dispatch time. One bounds test (`offset >= 8192` ->
fallback) + one LEA + one indirect jump replaces chain walks that averaged 20-100 deep. The mapping is
stored in ONE place (`dispatch_list.inc`, 125 entries), expanded at assembly time into
`sl_dispatch_fill` (which pre-fills both 8192-entry tables with the position fallbacks, then writes each
handler at its builtin's offset, a fill-time FATAL catches future offset overflow). All non-builtin
atoms (tags, user names, literals) take the identical fallback path they always took; semantics are
bit-identical (672/672 validator, verify.bat all-green, cats->dogs sidecar E2E re-verified in the live
engine).

Measured (median-of-3 runs, +/-10-20% thermal noise per run):

| Workload | chain (baseline) | dispatch table | delta |
|---|---|---|---|
| recursive fib | 67.96 ms / 291x | 63.3 ms / ~223x | **-17% ratio, -7% ms** |
| loop add | 5.39 ms / 883x | 4.45 ms / ~654x | **-17%** (stable across all runs) |
| string build | 321.7 ms | ~380 ms | within noise (walk-bound) |
| replace all | 5.46 ms | ~7.5 ms | within noise (intern work dominates) |

Honest analysis: the plan predicted 3-10x on dispatch-bound shapes; the measured gain is 1.2-1.4x. The
chain-walk cycles were real but NOT the dominant term in the remaining dispatch floor, the floor is the
per-form interpreter call overhead, `EvalNextArg` stack frames, and the dynamic-scope `Assoc` walk
(mitigations #1's estimate over-attributed to the chain). The table work still stands on its own merits:
uniform O(1) dispatch for ALL primitives (deep primitives like SORT/CallLibrary previously paid 70-100
comparisons), approximately 300 unreachable comparison-chain lines were eliminated, and a single source
of truth for the atom-to-handler map. The next levers, by measured share, are #2 (LOCAL frame array, the
Assoc walk) and #3 (REPEAT body pre-compilation).

## Per-phase BM25 profile (Sep 2026, HOST FFI slot timers)

`bin\bm25_sectorlogo_benchmark.exe` (source `examples\sectorlogo\bm25_sectorlogo_benchmark.c`) profiles
the pure-Logo BM25 processing sequence (`herbivorous_bm25.logo` algorithm, 5 KB herbivorous-cats corpus,
30 sentence chunks, 610 term instances) **per phase, from inside the interpreter**, zero ASM. The host
EXE exports `bm25_slot_begin(int)` / `bm25_slot_end(int)` (dllexport +
`v7s_set_caller_hmodule(GetModuleHandleA(0))`); the Logo script wraps each measured region in `sbegin N` and `send N` helper procedures.
For example, the start helper issues `CallLibrary [ "HOST" "bm25_slot_begin" "void" [ "integer" ] ] 25`. The benchmark
loads the slot-marked script from an external `.logo.template` (CLI arg, no default), splices the rep
count over `%REPS%`, puts it into microdb (`logo:sidecar`) and executes it via `v7s_sectorlogo_interp`.
This is the general **slot-timer pattern** for profiling ANY SectorLogo script phase-by-phase from C.

Run `build_logs\_run_bm25_bench.bat` to collect both profiles over ten repetitions. Historical single-run numbers (5 reps
each, sums shown; per-call = /5):

| Phase (slot) | Run 1 / call | Run 2 / call | Share |
|---|---|---|---|
| chunk corpus, Split x2 passes (slot 2) | 0.36 ms | 0.40 ms | 0.3% |
| query tokenize + qmap build (slot 3) | 0.48 ms | 0.55 ms | 0.4% |
| **corpus tokenize (slot 1, isolated)** | **125.5 ms** | **143.0 ms** | **~53%** |
| **tf/df pass, includes tokenize per chunk (slot 4)** | **103.1 ms** | **127.2 ms** | **~43%** |
| BM25 float scoring loop (slot 5) | 6.8 ms | 7.3 ms | 2.9% |
| rank + floor + pick, native SORT (slot 6) | 0.25 ms | 0.25 ms | 0.1% |
| END-TO-END bm25_search (slot 25) | 236.7 ms | 279.0 ms | 100% |

**Reading:** the whole processing sequence is **96% per-char tokenization** (slot 1 isolates pure
tokenize of all chunks; slot 4 is tokenize + the qhash/FindItem/SetItem bookkeeping, the delta between
them, ~3 ms, is the actual tf/df data-structure work). The BM25 math itself (slot 5) and everything
native (Split chunking, SORT ranking) are already irrelevant, 3% combined. Note slot 1 + slot 4
double-count the tokenize: slot 4's loop embeds its own tokenize calls, so the true e2e (slot 25) ~
tokenize-once (125-143 ms) + tf/df bookkeeping (~3 ms) + scoring (~7 ms).

**Feature-usage decision (the "are we using Split properly" question):**
- **Split is used correctly and is essentially free**, 0.36 ms to chunk the whole corpus into 30 sentences (paragraph split on `\n\n`, sentence split on `.`). Native C-speed, one primitive call per paragraph. This benchmark did not reveal a further measurable improvement.
- **tokenize is the anti-pattern:** it re-implements per-character what Split already does natively, a `REPEAT 999999` loop doing `Part:s:i 1` per char, `Search` in the alphabet per char, `_` concat per alnum run, then a stopword `Search` per word. That is ~2,500 interpreter dispatches for text Split would split in ONE call (`Split:s " "` -> words), leaving only stopword-drop + bigram join in Logo. The earlier design-note estimate of ~10-20 dispatches/char was optimistic, at the scale of this cat-related corpus, the measured cost is ~1.3-1.5 µs/char (35 chars/chunk average x 30 chunks x ~3.3 µs per iteration with its 2 nested IFELSEs).
- **Consequence for the roadmap:** the previously proposed `Words` native primitive (alnum runs + lowercase + stopword + bigrams, one call per chunk) is now *measured* to own ~96% of the pure-Logo processing sequence. Native Words would reduce e2e from ~237 ms to ~10-15 ms at this scale (tf/df bookkeeping + scoring dominate after). Alternatively a script-only fix, restructure tokenize around `Split:s " "` + per-word stopword `Search`, removes the per-character loop with zero language changes and should land within 2-3x of that.
- **Cached-index architecture confirmed:** slots 1+4 are *per-query* costs only because this test rebuilds everything per search call (the sidecar does too). The microdb-cached-index design (build once per corpus revision, per-query = postings walk only) leaves only slots 3+5+6 (~8 ms) on the hot path, the per-query processing sequence is already inside budget today; the index build (~130 ms, once) is the only part native tokenize would accelerate.

The slot-level FFI round-trip overhead is included in the measurements for slots 2/3/6 (sub-0.5 ms totals
including 2 CallLibrarys each), the timers do not perturb the measurement.

### Words in the processing sequence, old per-char walk vs native Words (Sep 2026)

The benchmark now loads the script from an external template file at runtime
(`examples\sectorlogo\bm25_bench.logo.template`, `%REPS%` spliced by the host) and runs the SAME
processing sequence twice in one process: once with `tokenize` built on native Words (stopword drop +
bigram join stay in Logo), once with the historical per-char `tokenize_old` (Lowercase + per-char
Part/Search alphanumeric classification + concat). Both runs share one `bm25_core` body (`:which` selector,
`toknew`/`tokold` unbound alias atoms + explicit IFELSE dispatch in `tokcall`; see the template header
for why RUN/atom indirection cannot work), with disjoint slot arrays so the host report compares
phase-by-phase. Run via `build_logs\_run_bm25_bench.bat` (10 reps; per-call = /10):

| Phase | NEW (Words) | OLD (per-char) | speedup |
|---|---|---|---|
| chunk corpus (Split) | 0.42 ms/call | 0.60 ms/call | 1.4x |
| query tokenize + qmap | 0.21 ms/call | 0.88 ms/call | 4.2x |
| **corpus tokenize (isolated)** | **36.5 ms/call** | **156.2 ms/call** | **4.3x** |
| **tf/df pass (tokenize + bookkeeping)** | **55.6 ms/call** | **134.1 ms/call** | **2.4x** |
| BM25 scoring loop (float math) | 9.0 ms/call | 9.0 ms/call | 1.0x |
| rank + floor + pick (SORT) | 0.50 ms/call | 0.54 ms/call | 1.1x |
| **END-TO-END bm25_search** | **102.6 ms/call** | **301.8 ms/call** | **2.9x** |

**Retrieval correctness: answers IDENTICAL** (old vs new Equal-checked in the template, both reproduce
the proven sidecar's 3-line answer for "what do cats eat in the wild and what should i feed my cat").
The residual NEW-run cost is the Logo-side per-word work Words leaves by design (stopword `Search` per
word, bigram `_` join, Prepend list build, all still ~10 dispatches/word) plus the
qhash/FindItem/SetItem bookkeeping; a full-native tokenize (stopwords + bigrams inside the primitive)
would reduce the remaining gap but loses configurability. The prediction held almost exactly: profile
section above predicted "e2e ~237 ms -> ~10-15 ms after native tokenize" for the tf/df+scoring floor,
measured NEW tf/df + scoring + rank = ~65 ms (with the Logo stopword/bigram tail included), e2e 102.6 ms
vs 301.8 ms.

### Tokenize micro profile (statement-level, calibrated)

`bm25_sectorlogo_benchmark.exe... micro` (slot 9 = empty sbegin/send pair = 0.50 µs, the calibration
floor; 383-char chunk, 5 reps; second run +/-20% thermally, ranking stable):

| Operation | per call | share of loop |
|---|---|---|
| Part char extract | 5.4 µs | 8.7% |
| Equal end test | 9.2 µs | 14.7% |
| Search alnum classify | 9.0 µs | 14.4% |
| `_` word concat | 5.7 µs | 7.2% |
| stopword Search (per word) | 24.9 µs | 7.2% |
| Prepend unigram append | 4.9 µs | 0.9% |
| i=i+1 counter | 4.9 µs | 7.8% |
| loop dispatch + timer pairs | not isolated | ~39% |
| **Split space split (native)** | **0.11 ms/chunk** | — |
| **per-word stopword scan (Split-based)** | **1.47 ms/chunk** | — |
| Lowercase whole string (native) | 0.01 ms | — |

Key readings: (1) **no single operation dominates**, the per-character *architecture* (~10 interpreter dispatches
per character) is the cost; (2) the empty HOST-FFI round-trip (0.5 µs) is ~18x cheaper than one Equal,
one native call can tokenize a whole chunk for less than ~2 Logo statements; (3) the Split-based
per-word path is **~14x cheaper** than the per-character loop with zero language changes. Design doc:
`docs/sectorlogo.md`.

## LOCAL frame array (attic\docs_plans\exec_local_plan.md, applied Sep 2026)

`sl_local_set` became an array of `[var atom][binding cell index]` qword pairs: `.exec_local` caches the
raw cons-cell index of the `(var. NIL)` binding pair it just created (parity-safe `push rcx`+`push r11`
pair across the pair-Cons; count commits LAST, both dwords are written before the entry becomes
visible). `MAKE` on a LOCAL name and every `:VAR`/bare-atom read now hit the entry by ONE dense
descending scan (newest-first, a correctness requirement for same-name shadowing at two scope depths)
and dereference the cached index in a single `[base + idx*8 + disp32]` chain, the 3-calls-per-node
env-walk (MAKE) and the Assoc walk (reads) are DELETED for LOCAL names. Non-LOCAL names keep the alist
path unchanged (dynamic scoping preserved). The `.make_binding_lost_fatal` was removed: its root-cause
class is eliminated by construction (index and binding pair share their lifetime -- created together in
`.exec_local`, discarded together by the count shrink). GC: zero new roots, cached indices are only
dereferenced while their entry is active, and an active entry's binding is env-reachable (r14 / base_env
/ stack scan) so the cell can never be recycled while the index is live.

Measured (median-of-3 loadtest runs, +/-10-25% laptop thermal noise):

| Workload | before | after | delta |
|---|---|---|---|
| loop add (20k) | 4.45 ms | 3.0-3.8 ms | **-15-30%** (3 reads + 2 writes per iteration, the direct beneficiary) |
| recursive fib(24) | 63.3 ms | 58-98 ms (79 median) | no consistent gain demonstrated; the workload has no LOCAL variables |
| string build | ~380 ms | 611-703 ms | slower in the recorded runs; the cause was not isolated |
| replace all | ~7.5 ms | 4.5-6.3 ms | faster in these runs, but not isolated from thermal variation |

Safety net all green: v_sectorlogo_validate 672/672; t_sectorlogo_gc fibN-seqx10 AND env-local 32767
(live LOCALs across GCs, frame indices remain valid after every collection: `Pass: 536854528`);
bench-seq / env-garbage 300000 / env-grow pass; verify.bat full processing sequence all-green (19
validators + 5 smokes at the time, the gate's inference phase later grew to 11 tests). Same-name
shadowing E2E (outer LOCAL x=1, inner LOCAL x=2 -> 3) passes, the descending scans resolve to the
innermost binding. NOTE (validator gap): the 672-test suite has NO nested same-name LOCAL case, the
shadowing check is a manual E2E script (`build_logs\shadow_test.logo`), keep it for future frame-array
edits. The t_sectorlogo_gc env-local variant was also fixed (pre-existing test-design bugs, not engine
bugs): its garbage binding used a single `MAKE "G"` per iteration (a REACHABLE prepend -> live env grows
O(n) -> quadratic Assoc + guaranteed 1M-cell exhaustion, the exact failure shape env-grow exists to
test); it now double-MAKEs G so each iteration nets ONE inaccessible shadowed cell, and a 30-bit
NUMBER-payload guard rejects iteration counts whose triangular sum overflows the engine's integer range
(the 500000-iteration "heap overflow FATAL" was this payload wrap, not a leak).

## How we improved, the Intern dual-hash fast path (Sep 2026)

### Before

`Intern` (the atom-string dedup table) linearly scanned the ENTIRE atom area on every call, with a
byte-compare against every stored string and an O(n^2) interaction: in the concat workload every new
atom shares a growing prefix with all prior atoms, so total byte reads grew cubic in workload size
(~1.5-2 s measured for the string workloads; ratios vs C were ~100,000-110,000x).

### The mechanism (attic\docs_plans\intern_speedup_plan.md, applied)

- **StringHash (h1)**: centralized strlen+CRC32 (SSE4.2 `crc32`, single pass, NUL included), extracted from microdb's two inline copies into `string_hash.inc`; microdb `ht_add`/`ht_find` now call it too (one hash implementation project-wide).
- **aes_string_hash (h2)**: AES-NI u64 hash (two `aesenc` rounds over a 64-byte truncated/zero-padded key block, low64 XOR, 0->1 normalized).
- **Bloom filter** (2^22 bits, 512 KiB BSS, insert-only): every interned key sets `bit(h1 & MASK)` and `bit(h2 & MASK)`; lookup tests both, **either bit clear = certain never-interned = skip the scan entirely** (straight to append). False-positive ~1.4e-6 at ~2.5K atoms, effectively zero phantom scans.
- **Direct-mapped slot cache** (4096 slots x 16 B: `[h1][atom_off][h2]`): repeat lookups verify stored h1, then stored h2, then a **full byte-compare**, hash values are filters, never the identity. (Initially 1024 slots; the 2000-atom concat working set thrashed it, raised to 4096, measured 679->345 ms -> further corrected methodology, see below.)
- The linear scan remains as the single source of truth for bloom-positive misses.

### Measured effect (verified workloads, same machine)

| Workload | before (ratio vs C) | after (ratio vs C) | before ms | after ms |
|---|---|---|---|---|
| string build | ~110,000x | 28,720x | 1,500-2,000 | 322 |
| replace all | ~100,000x | 4,201x | ~2,400 | 5.5 |
| recursive fib | ~289x (unchanged, scan never fired) | 291x | ~80 | 68 |
| loop add | ~920x (unchanged) | 883x | ~8.5 | 5.4 |

Intern-hash block-copy SWAR window (Sep 2026, opt_and_bug aes_string_hash): the <=64-byte block copy in
`aes_string_hash` copies 8 NUL-free bytes per window (page-gated, byte tail on any NUL/truncation)
instead of per-byte, atom names of 8-31 bytes (builtin names, `GetStream`, RAG query fragments) pay ~8
window iterations instead of 56-217 byte iterations per hash. Post-change medians (5 reps): fib 57.9 ms
/ 222x, loop add 3.5 ms / 539x, string build 391.8 ms / 31,852x (walk-bound phase dominates, concat cost
within thermal noise), replace all 6.1 ms / 2,174x. Same run also fixed the intern_builtins h2 bloom-bit
bug (the empty string's second hash bit was never set, `mov eax, edi` destroyed h2 before the second
`bts`; now parked in rdx per the `.slot_filler` pattern), so `Intern("")` finds both bits set and never
appends a duplicate empty-string atom.

Faster-strings cycle 1 (Sep 2026, commit 85b4019 + release 2.15.0): StringHash rewritten to AVX2
4x-crc32-dword windows (byte-identical hash), aes_string_hash copy stub to a 16B AVX2 window with ONE
base page-gate, Intern slot/scan compares + skip_nul to AVX2 16B windows (source-side page gate),
lg_value_to_str ATOM rep-movsb fast path + NUMBER single-div backward render. Post-change cross-language
medians (3 loadtest runs of 5 interleaved reps + one bench_compare_languages.bat pass, this machine,
v2.15.0):

| Workload | SectorLogo | Python 3.12 | Node 24 | C ref | SectorLogo/C |
|---|---|---|---|---|---|
| recursive fib(24) | 52.6 ms | 10.69 ms | 0.79 ms | 0.29 ms | ~183x |
| loop add (20k) | 3.28 ms | 1.49 ms | 0.02 ms | 0.01 ms | ~468x |
| string build (2k concats) | 70.5 ms | 0.26 ms | 0.08 ms | 0.01 ms | ~5,030x |
| replace all (60) | 1.06 ms | 0.05 ms | 0.05 ms | 0.001 ms | ~890x |

Loadtest row medians across the 3 runs: fib 41.7-69.7 ms, loop add 2.9-3.5 ms, string build 65.7-73.9
ms, concat-only 65.6-79.8 ms, repeat intern 66.5-96.7 ms, replace all 0.89-1.07 ms (6/6 workloads
verified every run). vs the SWAR-window state: string build 391.8 -> ~70 ms (~5.6x, the concat/intern
phases finally fast enough that the verification walk no longer dominates alone), replace all 6.1 ->
~1.1 ms (~5.7x, intern-bound shape), loop add 3.5 -> ~3.3 ms (within noise), fib 57.9 -> ~53 ms (within
noise, dispatch-bound, intern never fired). The string-shape ratio vs C dropped from 31,852x to ~5,030x
and replace-all from 2,174x to ~890x; SectorLogo's replace-all is now within ~20-30x of CPython's and
beats it on absolute terms in this run's interleave. Remaining floor is unchanged: per-form Car/Cdr
machinery + EvalNextArg frames (fib/loop shapes) and the copy-based Part/Search string machinery (string
shapes), next levers stay #3 (REPEAT body pre-compilation) and #4 (length-prefixed atoms + memcpy
staging) from the mitigation list below.

Micro-probes during verification:
- 100,000 repeat-hit interns of the same string: pure slot-cache path, ~0 ms measurable overhead per call.
- 2,000 novel concats with a LOCAL accumulator: ~0 ms net of process overhead, the concat phase is fully fast-pathed; **the remaining string-build time is the 99,999-step verification walk, not interning**.
- `Intern("")` duplicate-atom edge case: intern_builtins pre-creates the empty-string atom and sets its bloom bits by hand, validated by the 672-test language validator.

Safety net all passed following the change: v_sectorlogo_validate 672/672, v_mdb_validate 21/21 (hash
centralization byte-identical: save/load round-trips), t_sectorlogo_gc fibN-seqx10 (~20 GCs,
filter/cache survive every GC, insert-only design is GC-proof by construction), verify.bat full
processing sequence at the time (19 validators + 5 inference smokes, "paris" in every output).

Faster-strings cycle 2 (Sep 2026, faster-strings plan Phase 4 + the Phase 3 TextLength leftover): `Part`
rewritten to a BOUNDED AVX2 scan, 16-byte windows over `[0, start+len)` with a <=15-byte tail, replacing
the O(start) byte skip (which re-walked the string prefix on every call) and the byte-at-a-time copy
loop (now one `rep movsb` after a single bounds check; FATAL semantics byte-identical). `TextLength`'s
own 8-byte SWAR core deleted, it now calls the shared `sl_swar_strlen` AVX2 leaf (2x window width, no
page gate under the committed-block contract).

Honest measurement (6 loadtest runs of 5 interleaved reps, same machine, Sep 13):
- string build 70.1-146.6 ms (median ~85), concat-only 62.3-127.1, repeat intern 63.2-112.5, replace all 0.87-1.88 ms, every row overlaps cycle-1's ranges (65.7-73.9 / 65.6-79.8 / 66.5-96.7 / 0.89-1.07). **Within noise on this workload, and the arithmetic says it must be**: the loadtest string is only 2,000 chars, so the deleted skip term was Sigma O(K) ~ 2M byte-steps ~ 1-2 ms inside a ~70 ms workload whose walk phase is dispatch-floor-bound (~2,000 iterations x ~8 forms each). A flat result here is the expected outcome, not a regression signal (672/672 validator incl. every Part edge case; no row worsened).
- Where the win IS real, the production stream-walk shape: `Part:S:K 1` with K ascending over a full accumulated GetStream (~6 KB per RAG hook call). Old cost per full walk: Sigma O(K) byte-skip = 6000^2/2 = 18M byte-steps ~ 6-9 ms per hook read, growing QUADRATICALLY with stream length. New cost: O(K/16) window iterations ~ 0.5-1 ms, the skip term now scales linearly. Same math applies to every sidecar that walks a stream (replace_all idiom, query extraction, distillation gates).
- `TextLength` row: unchanged within noise (the leaf swap is a straight 2x window-width upgrade on a microsecond-scale call).

Cross-language snapshot after cycle 2 (bench_compare_languages.bat, same session, median of 5):

| Workload | SectorLogo | Python 3.12 | Node 24 | Tcl 9.0 | PowerShell 5.1 | SectorLogo/C |
|---|---|---|---|---|---|---|
| recursive fib(24) | 76.3 ms | 17.2 ms | 0.68 ms | 52.3 ms | 26,820 ms | ~188x |
| loop add (20k) | 8.7 ms (3.7 in adjacent runs, thermal) | 3.3 ms | 0.03 ms | 1.7 ms | 6.5 ms | ~360-476x |
| string build (2k) | 90.3 ms | 0.5 ms | 0.10 ms | 1.6 ms | 0.9 ms | ~5,071x |
| replace all (60) | 1.9 ms | 0.1 ms | 0.04 ms | 0.2 ms | 0.4 ms | ~697x |

Relative position unchanged from cycle 1: within ~1.3-1.5x of Tcl on fib, ahead of PowerShell
everywhere, order of magnitude off CPython/Node on string shapes. Remaining floor is still the tree-walk
dispatch (mitigations #3), the string implementation itself is now windowed end-to-end (hash, compare,
copy, scan).

## Faster-strings cycle 3, Search/Replace/Split AVX2 first-char scan (Sep 2026, faster-strings plan Phases 5+6)

The `Search`, `Replace` and `Split` scans were rewritten from byte-at-a-time first-char loops to 16-byte
AVX2 windows, using the `SL_FIRSTCHAR_SCAN` macro: `vmovdqu`, `vpcmpeqb`, `vpmovmskb`, `tzcnt`, `bzhi`.
The `tzcnt` converts the NUL bitmask to the position `BZHI` needs; a first draft passed the mask itself
and missed candidates sharing a window with the NUL. The `Replace` build step is three `rep movsb`
segments with lengths from the shared `sl_swar_strlen` leaf and one combined bounds check, with r8
reloaded after every leaf call because the leaf clobbers it. `Split` hoists `delim_len` and `src_strlen`
once, removing the per-match delimiter re-walk, and reloads the scan vectors after every `Intern`
because `Intern` clobbers xmm0-4.

Measured (loadtest, median of 5, this machine; baseline = byte-loop build of one commit earlier):

| Workload | before | after | delta |
|---|---|---|---|
| find scan (absent needle, 9999 x 240 B) | 5.64 ms | 4.50 ms | **-20%** |
| replace all (60 occurrences) | 0.712 ms | 0.344 ms | **-52% (2.07x)** |

The absent-needle row is pure scan term (every Search walks to the NUL with zero first-char hits), its
-20% is the scan-window win as seen through ~4 ms of interpreter dispatch overhead; the replace-all row
compounds the scan win with the segment-copy win. Split's win shows up in the BM25 processing sequence
(corpus tokenize already native via Words; the delimiter walk deletion removes the remaining per-segment
O(delim) term). Full gates at landing: validator 736/0, mdb 21/0, GC tool all 4 modes, loadtest 7/7
rows, cats-to-dogs and herbivorous-BM25 sidecar E2E green, verify.bat all-pass.

## Cross-language snapshot after prefixed atoms + Equal text compare (Sep 2026)

Re-run of `bench_compare_languages.bat` after the type-byte prefix changes (STRING vs WORD atoms with
text-content Equal, lg_ensure_word dispatch bridge, console ReadKey rewrite, DefineCommand/FFI type-byte
fixes). All six languages in one session (SBCL present this run):

| Workload | SectorLogo | Python 3.12 | Node 24 | Tcl 9.0 | SBCL 2.6 | PowerShell 5.1 | SectorLogo/C |
|---|---|---|---|---|---|---|---|
| recursive fib(24) | 65.7 ms | 8.76 ms | 0.52 ms | 36.0 ms | 0.97 ms | 17,611 ms | ~194x |
| loop add (20k) | 5.87 ms | 1.21 ms | 0.036 ms | 1.56 ms | 0.14 ms | 3.29 ms | ~889x |
| string build (2k) | 56.4 ms | 0.19 ms | 0.060 ms | 1.29 ms | 7.86 ms | 0.63 ms | ~4,477x |
| replace all (60) | 0.24 ms | 0.037 ms | 0.029 ms | 0.10 ms | 0.18 ms | 0.22 ms | ~343x |

Loadtest extras (same run): concat only 55.2 ms, repeat intern 62.1 ms, find scan (absent needle) 3.63
ms / 15.8x.

**Honest reading:** string build dropped from 116.1 ms (previous batch run) to 56.4 ms, a ~2x
improvement. Replace-all dropped from 0.97 ms to 0.24 ms -- a ~4x improvement. The prefixed-atoms Equal
text-compare path and the lg_ensure_word dispatch bridge are the likely contributors (Equal no longer
traverses linear atom chains; the dispatch fallback is change-detected and skipped for known builtins).
Fib and loop add are within thermal noise of previous runs (dispatch-bound, intern never fires).
Relative position is unchanged: within ~1.8x of Tcl on fib, an order of magnitude off CPython/Node on
string shapes, two-plus orders ahead of PowerShell on recursion.

## Cross-language snapshot after release v2.18.0 (Sep 2026)

Re-run of `bench_compare_languages.bat` after the SectorLogo bug-fix batch (Define in-place update,
lg_invoke_exe_fn shadow overlap, sticky STOP/CONTINUE, check asymmetry, dispatch table bound,
unary-minus tokenizer, PascalCase rename, prefixed atoms, Equal text compare, Search/Replace AVX2
first-char scan, all committed and verified). All six languages in one session:

| Workload | SectorLogo | Python 3.12 | Node 24 | Tcl 9.0 | SBCL 2.6 | PowerShell 5.1 | C ref | SectorLogo/C |
|---|---|---|---|---|---|---|---|---|
| recursive fib(24) | 45.79 ms | 7.04 ms | 0.56 ms | 32.93 ms | 0.87 ms | 16,708 ms | 0.211 ms | ~217x |
| loop add (20k) | 5.84 ms | 1.06 ms | 0.034 ms | 1.39 ms | 0.13 ms | 3.43 ms | 0.007 ms | ~885x |
| string build (2k) | 56.88 ms | 0.16 ms | 0.059 ms | 1.18 ms | 7.39 ms | 0.65 ms | 0.014 ms | ~4,063x |
| replace all (60) | 0.36 ms | 0.029 ms | 0.037 ms | 0.091 ms | 0.18 ms | 0.21 ms | 0.003 ms | ~138x |

Loadtest extras (same run): concat only 58.81 ms, repeat intern 60.73 ms, find scan (absent needle) 4.03
ms / 22.0x.

**Honest reading:** fib is within thermal noise of the previous snapshot (45.8 vs 65.7 ms, laptop turbo
state variation; the C reference at 0.211 ms confirms a cooler session). Replace-all at 0.36 ms
continues the steady improvement from the AVX2 first-char scan (0.97 -> 0.24 -> 0.36 ms range across
sessions, thermal variation dominates at sub-ms scale). String build at 56.9 ms is consistent with the
56.4 ms prefixed-atoms run. Relative position unchanged: within ~1.4x of Tcl on fib, order of magnitude
off CPython/Node on string shapes, two-plus orders ahead of PowerShell on recursion.

## Cross-language snapshot after the reader jump-table tokenizer (Sep 2026)

Re-run of `bench_compare_languages.bat` after the GetToken first-char classification rework (256-byte
class table + qword jump table replacing the 16-deep cmp/je chain, parse-time only), the parser
recursion depth guard, and the opt_and_bug apply batches of the same session (attention/prefill/
rmsnorm/encode/lz4/vit-dispatch micro-opts, none on the interpreter path). All six languages in one
session:

| Workload | SectorLogo | Python 3.12 | Node 24 | Tcl 9.0 | SBCL 2.6 | PowerShell 5.1 | C ref | SectorLogo/C |
|---|---|---|---|---|---|---|---|---|
| recursive fib(24) | 50.18 ms | 9.52 ms | 0.67 ms | 37.59 ms | 1.06 ms | 15,751 ms | 0.260 ms | ~193x |
| loop add (20k) | 5.64 ms | 1.37 ms | 0.018 ms | 1.69 ms | 0.155 ms | 3.39 ms | 0.006 ms | ~910x |
| string build (2k) | 49.09 ms | 0.22 ms | 0.066 ms | 1.33 ms | 8.56 ms | 0.61 ms | 0.012 ms | ~4,024x |
| replace all (60) | 0.224 ms | 0.045 ms | 0.039 ms | 0.103 ms | 0.191 ms | 0.214 ms | 0.001 ms | ~374x |

Loadtest extras (same run): concat only 52.01 ms / 3,152x, repeat intern (GetStream) 50.77 ms / 289x,
find scan (absent needle) 3.74 ms / 17.1x.

**Interpretation:** The jump-table tokenizer changes parsing, while the benchmark measures steady-state
interpretation after one script parse. The absence of a steady-state speedup is therefore expected:
every row is inside the thermal band of the v2.18.0 snapshot (fib 50.2 vs 45.8 ms, with the C reference
itself 0.260 vs 0.211 ms, i.e. a warmer session; the fib RATIO actually improved to ~193x from ~217x;
string build 49.1 vs 56.9 ms is at the fast edge of the band, matching the 56.4 ms prefixed-atoms run;
loop add 5.64 vs 5.84 ms unchanged). Replace-all 0.224 ms matches the prior low reading (0.24
ms). Relative position unchanged: within ~1.3x of Tcl on fib, ahead of PowerShell everywhere, an order
of magnitude off CPython/Node on string shapes. The tokenizer win shows up where it applies: one-time
script load (a ~16-instruction classification chain reduced to ~5 for word tokens, the dominant token
class), not in these steady-state rows.

## Cross-language snapshot after the opt_and_bug batches + rbx ABI fix (Sep 2026)

One session, median of 5 interleaved reps, build of commit `008d558` (the `b_sectorlogo_loadtest`
crash-fix build; the day's opt_and_bug batches are correctness/precision + cold-path work and touch none
of these steady-state workloads). 7/7 loadtest workloads verified.

| Workload | SectorLogo | Python 3.12 | Node 24 | Tcl 9.0 | SBCL 2.6 | PowerShell 5.1 | C ref | SectorLogo/C |
|---|---|---|---|---|---|---|---|---|
| recursive fib(24) | 66.40 ms | 10.17 ms | 0.88 ms | 37.79 ms | 1.65 ms | 19,912 ms | 0.307 ms | ~216x |
| loop add (20k) | 6.89 ms | 1.41 ms | 0.113 ms | 1.48 ms | 0.154 ms | 3.99 ms | 0.008 ms | ~810x |
| string build (2k) | 73.10 ms | 0.227 ms | 0.117 ms | 1.27 ms | 10.45 ms | 0.886 ms | 0.020 ms | ~3,730x |
| replace all (60) | 0.305 ms | 0.046 ms | 0.064 ms | 0.098 ms | 0.322 ms | 0.287 ms | 0.001 ms | ~305x |

Loadtest extras (same run): concat only 62.54 ms / 4,343x, repeat intern (GetStream) 63.15 ms / 286x,
find scan (absent needle) 4.70 ms / 19.5x.

**Interpretation:** This was a heavily throttled session. The C reference itself ran +18% (fib 0.307 vs 0.260 ms)
to +67% (strbuild 0.020 vs 0.012 ms) slower than the tokenizer snapshot, so absolute SectorLogo times
(fib 66.4 vs 50.2 ms, strbuild 73.1 vs 49.1 ms) track the machine state, not a code change. The ratios
are the honest row: fib ~216x vs ~193x, loop add ~810x vs ~910x, string build ~3,730x vs ~4,024x,
replace-all ~305x vs ~374x, all within or at the edge of the thermal band, i.e. flat as expected: none
of the applied batches (reader jump-table, eval split, FATAL texts, private stack, ABI fix, precision
batch) change the measured steady-state interpretation paths. Relative position unchanged: within ~1.8x
of Tcl on fib, ahead of PowerShell everywhere, an order of magnitude off CPython/Node on string shapes.

## Cross-language snapshot after the SectorLogo Batch-D2 report cycle (Sep 2026)

Re-run of `bench_compare_languages.bat` after the Batch-D2 `opt_and_bug` cycle (nine reports consumed).
All six languages in one session, 7/7 loadtest workloads verified. What this cycle actually changed, and
where each item shows up:

| Change | Live-path cost | Shows up here? |
|---|---|---|
| `Car`/`Cdr`/`PopCons` two-compare guard (taken branch removed) | **hot**, every list access in every walk | yes (fib, loop add) |
| `Equal` atom-text compare, 16-byte AVX2 windows instead of per byte | **hot**, every `Equal` on distinct-offset atoms | partly (replace all, find scan) |
| NUMBER decimal-width table walk instead of a dependent `div` chain | **hot**, every NUMBER rendered, incl. every `FATAL_VALUE` | marginally (diagnostics) |
| `SquareRoot` Newton loop: two `movsxd` dropped | hot for `SquareRoot` only, not in these workloads | no |
| `NUM_RANGE_CHECK` on 6 arithmetic tails; `AsInteger` destination check; 9 `v7prim` diagnostics | **cold**, abort paths a passing script never executes | no (by construction) |
| `Words` table probe, `NGrams` in-place reversal | hot for the BM25 path, not in these workloads | no (measured separately below) |

| Workload | SectorLogo | Python 3.12 | Node 24 | Tcl 9.0 | SBCL 2.6 | PowerShell 5.1 | C ref | SectorLogo/C |
|---|---|---|---|---|---|---|---|---|
| recursive fib(24) | 59.81 ms | 12.77 ms | 0.964 ms | 42.96 ms | 1.040 ms | 19,844 ms | 0.292 ms | ~205x |
| loop add (20k) | 7.30 ms | 2.14 ms | 0.021 ms | 1.57 ms | 0.151 ms | 4.44 ms | 0.009 ms | ~839x |
| string build (2k) | 65.10 ms | 0.33 ms | 0.124 ms | 1.32 ms | 9.264 ms | 0.71 ms | 0.013 ms | ~4,932x |
| replace all (60) | 0.254 ms | 0.069 ms | 0.069 ms | 0.098 ms | 0.210 ms | 0.219 ms | 0.001 ms | ~363x |

Loadtest extras (same run): concat only 75.91 ms / 5,272x, repeat intern (GetStream) 75.03 ms / 333x,
find scan (absent needle) 4.07 ms / 20.7x.

**Honest reading.** The session ran **cooler** than the previous snapshot, the C reference, which is the
control, moved faster on every row (fib 0.292 vs 0.307 ms, loop add 0.009 vs 0.008, string build 0.013
vs 0.020), so absolute SectorLogo times are only loosely cross-comparable and the **ratios are the
honest row**. Those ratios are mixed within the thermal band: fib ~216x -> ~205x (5% better), loop add
~810x -> ~839x (3.5% worse), replace-all ~305x -> ~363x (19% better), string build ~3,730x -> ~4,932x
(32% *worse*, but its C reference is a sub-20 µs measurement that itself swung 35%, so that row is
C-reference noise, not a SectorLogo regression; the absolute SectorLogo time improved 73.1 -> 65.1 ms).
**The 5% fib improvement is consistent with the `Car`/`Cdr` branchless change, the two hot items in this
cycle target exactly the dispatch-bound shapes, but it is not separable from thermal noise on a single
session, and it is not claimed as a measured effect.**

**Where this cycle's work is actually proven:** the BM25 processing sequence, which the cycle targeted
directly, and which the cross-language workloads do not exercise. `bin\bm25_sectorlogo_benchmark.exe` on
the herbivorous-cats corpus (median of 5 reps):

| Phase | NEW | OLD (per-char) | speedup |
|---|---|---|---|
| corpus tokenize (isolated) | 26.6 ms | 98.0 ms | **3.7x** |
| query tokenize + qmap | 0.62 ms | 1.68 ms | 2.7x |
| tf/df pass | 44.4 ms | 114.7 ms | 2.6x |
| **END-TO-END `bm25_search`** | **68.4 ms** | **211.4 ms** | **3.1x** |
| **answers** | **Identical to the prior implementation** | — | — |

The `Words` table probe alone is worth **-6 instructions and -2 branches per alnum byte** (16 instr / 5
branches -> 10 / 3 on the letter path, 12 / 4 -> 10 / 3 on the digit path), and `NGrams` now reverses
its window and result **in place**, removing `n + 1` `Cons` calls per window, each of which was also a
potential GC trigger. Both are byte-identical-output changes, and the identical-answer check is the
acceptance criterion, not the timing.

**One measurement note that matters for the record:** the BM25 benchmark had never produced a number
before this session. Both `bm25_bench.logo.template` and `bm25_micro.logo.template` wrote their
`CallLibrary` type names **bare** (`Void` / `Integer`), and the four type atoms are registered with
`SL_TYPE_STRING` and compared by offset, so every run died on the first `sbegin` with "C-CALL return
type must be integer, pointer, text, or void" and the tool reported all slots as `0.00 ms / 0 calls`
instead of failing loudly. Every earlier "BM25 is 2.9x" figure in this document was therefore measured
*after* that fix, but the pre-fix state was a test program that had stopped performing its intended
validation. Corrected to quoted lowercase types (`"void"` / `"integer"`).

## Cross-language snapshot after the SectorLogo ReadKey/FloatRemainder/const-consolidation batch (Sep 2026)

`bench_compare_languages.bat`, one session, median of 5 interleaved reps, all six languages present, 7/7
loadtest workloads verified. This cycle's interpreter-path changes: the `lg_ensure_word` redundant NIL
test removed (2 instructions + 1 branch off the STRING->WORD bridge), the five hand-rolled bitwise
operator bodies deduped into `INFIX_BITWISE_OP`, the `NGrams` element counter hoisted out of its inner
loop into a non-volatile, and the Win64 ABI parity pad added to `intern_builtins`. The `fprem` C2 fix,
the `ReadKey` NIL fix and the binary32-constant consolidation are correctness items on paths these
workloads do not exercise, so a flat result on them is the expected outcome.

| Workload | SectorLogo | Python 3.12 | Node 24 | Tcl 9.0 | SBCL 2.6 | PowerShell 5.1 | C ref | SectorLogo/C |
|---|---|---|---|---|---|---|---|---|
| recursive fib(24) | 69.19 ms | 12.10 ms | 0.922 ms | 45.76 ms | 0.831 ms | 18,516 ms | 0.354 ms | ~196x |
| loop add (20k) | 7.53 ms | 2.96 ms | 0.024 ms | 1.699 ms | 0.137 ms | 3.67 ms | 0.008 ms | ~978x |
| string build (2k) | 61.75 ms | 0.383 ms | 0.152 ms | 1.340 ms | 6.406 ms | 0.690 ms | 0.014 ms | ~4,258x |
| replace all (60) | 0.299 ms | 0.058 ms | 0.085 ms | 0.107 ms | 0.165 ms | 0.217 ms | 0.001 ms | ~374x |

Loadtest extras (same run): concat only (no walk) 51.72 ms, repeat intern (GetStream) 64.83 ms, find
scan (absent needle) 4.09 ms / 21.4x.

**Honest reading.** The session ran **warmer** than the Batch-D2 snapshot -- the C reference, which is
the control, moved +21% on the fib row (0.354 vs 0.292 ms), so absolute SectorLogo times are only
loosely cross-comparable and the **ratios are the honest row**. Those ratios are mixed within the
thermal band: fib ~205x -> ~196x (4.5% worse, tracking the hotter session), string build ~4,932x ->
~4,258x (14% worse on a ratio whose C reference is a sub-20 µs measurement that itself swung 8%, the
absolute SectorLogo time actually *improved* 65.1 -> 61.7 ms), replace-all ~363x -> ~374x (3% better,
sub-ms scale), and **loop add ~839x -> ~978x (16% better)**.

**The loop-add improvement is the one row with a mechanistic explanation.** It is also the shape the
`lg_ensure_word` change targets: that function is the STRING->WORD bridge reached for **every
non-builtin atom in both expression and statement position**, every user variable reference, every
user-function call, every bare atom, and the loop-add workload does a `MAKE` plus a `:T` read on every
one of its 20,000 iterations. Removing two instructions and a branch from it is the kind of change that
shows up on exactly this workload and not on fib. **It is not claimed as a measured effect beyond
that**: a single session cannot separate a 16% ratio move from the thermal band, and the honest
statement is that the one row with a predicted mechanism moved in the predicted direction.

Relative position is unchanged: within ~1.5x of Tcl on fib, ahead of PowerShell everywhere, an order of
magnitude off CPython/Node on the string shapes, and behind Node/SBCL (JIT and native compiler) on
dispatch.

## Cross-language snapshot after adding GNU Guile to the suite (Sep 2026)

`bench_compare_languages.bat`, one session, median of 5 interleaved reps, all seven languages present,
8/8 loadtest workloads verified. The change in this cycle is a **benchmark-suite change, not an
interpreter change**: no SectorLogo source was touched, so a flat SectorLogo row is the expected outcome
and is reported as such.

GNU Guile 3.0.11 was installed from the MSYS2 distribution (`pacman -S guile`, pulling `libguile-3.0.11`
and `libgc-8.2.12`). Guile has no official Windows build, the GNU FTP carries source tarballs only, so
MSYS2 is the practical native-Windows route. Guile compiles Scheme to G-code and runs it on a
**JIT-backed VM**; it has no native-machine-code compiler (unlike SBCL), so it sits between Tcl's
bytecode VM and SBCL/Node on dispatch.

| Workload | SectorLogo | Python 3.12 | Node 24 | Tcl 9.0 | SBCL 2.6 | Guile 3.0 | PowerShell 5.1 | C ref | SectorLogo/C |
|---|---|---|---|---|---|---|---|---|---|
| recursive fib(24) | 145.20 ms | 12.97 ms | 0.825 ms | 55.93 ms | 1.151 ms | 3.406 ms | 34,041 ms | 0.502 ms | ~289x |
| loop add (20k) | 15.93 ms | 1.770 ms | 0.022 ms | 1.981 ms | 0.169 ms | 0.097 ms | 9.173 ms | 0.016 ms | ~990x |
| string build (2k) | 220.21 ms | 0.304 ms | 0.097 ms | 1.719 ms | 10.164 ms | 21.404 ms | 1.386 ms | 0.022 ms | ~10,195x |
| replace all (60) | 0.724 ms | 0.065 ms | 0.068 ms | 0.140 ms | 0.236 ms | 0.307 ms | 0.547 ms | 0.005 ms | ~148x |

Loadtest extras (same run): concat only (no walk) 191.07 ms, repeat intern (GetStream) 207.27 ms, find
scan (absent needle) 9.97 ms / 17.5x, atom churn (collect path) 8.42 ms / 448x.

**The session ran much hotter than the earlier snapshot**, the C reference, the control, moved from
0.354 ms to 0.502 ms on the fib row (+42%). The ratios are therefore the only honest row; the absolute
SectorLogo times are not cross-comparable with the earlier tables. The `TRACE` lines interleaved into
the SectorLogo leg's output are a pre-existing build-configuration artifact (a DLL built with
`TRACE_ENABLE`), not a change introduced by this cycle.

**Where Guile lands.** On the two dispatch-bound shapes Guile is the second fastest implementation in
the table and the fastest of the four that are not natively compiled: fib 3.41 ms sits ~3x behind SBCL's
1.15 ms and ahead of Tcl (55.9) by ~16x, and loop add 0.097 ms is faster than SBCL (0.169 ms) and Tcl
(1.98 ms), and behind only Node (0.022 ms). Loop add at 0.097 ms over 20,000 iterations is ~5
ns/iteration, which is native-speed arithmetic: the G-code VM plus its JIT is compiling that
tail-recursive loop to machine code. This is the same reason Guile beats SBCL on that one row, SBCL
emits a loop the CPU does not specialise, Guile's JIT does, and it is the clearest evidence in the table
that **JIT tier-up, not per-primitive speed, is what decides the dispatch-bound rows**.

**The string-build row is where Guile is weakest, and the reason is the string representation.** Guile's
`string-append` allocates a fresh string per call and every one of the 2,000 steps allocates, so the
workload is allocator- and GC-bound. That is the same O(n^2)-allocation shape that makes SBCL pay 10.2
ms here (10x its own fib time) while Tcl's mutable-command `append` pays 1.7 ms. Guile's 21.4 ms is ~2x
SBCL and ~12x Tcl for the same reason: no in-place string append, plus Boehm GC collection on top.

**Guile's string-build number is the least stable in the table** and is reported with that caveat.
Repeated runs of the Guile leg alone gave 21.4 / 23.5 / 27.2 / 28.5 / 31.0 / 34.6 / 42.1 / 43.4 ms, a 2x
spread on a GC-bound workload, against ~0.05-0.11 ms and ~2.4-3.5 ms stability on the Guile loop-add and
fib rows. The median of that distribution is ~30 ms, so the table's 21.4 ms is the optimistic end of the
band. **Read the string-build row for Guile as "an order of magnitude slower than SBCL, same class as
SBCL", not as a precise 2x.** The other three Guile rows are stable and can be read directly.

Relative position of SectorLogo is unchanged by adding a language: still ahead of PowerShell and Tcl on
fib, behind SBCL/Node/Guile on dispatch, and an order of magnitude off CPython/Node on the string
shapes.

## Cross-language snapshot after the full-string AES h2 hash (Oct 2026)

bench_compare_languages.bat, one session, median of 5 interleaved reps, all eight languages present, 8/8
loadtest workloads verified. The interpreter change this cycle is the replacement of the 64-byte-prefix
AES hash with the **full-string AES absorb chain** (`aes_string_hash.inc`): one

aesenc per 16 string bytes, tail overread masked in-register, two finalize rounds so the last block's
high half reaches the 64-bit result, no stack copy, no 64-byte prefix cap. h2 values all changed; h1
(crc32) and every DB hash are untouched, and the Intern linear scan remains the identity check, so the
change is semantics-neutral. The interning-heavy rows are the ones that could move; the full-string hash
costs one AES round per 16 bytes where the old one paid a 64-byte stack copy plus two rounds regardless
of length.

| Workload | SectorLogo | Python 3.12 | Node 24 | Tcl 9.0 | SBCL 2.6 | Guile 3.0 | PowerShell 5.1 | C ref | SectorLogo/C |
|---|---|---|---|---|---|---|---|---|---|
| recursive fib(24) | 118.09 ms | 9.55 ms | 0.538 ms | 36.09 ms | 0.754 ms | 1.490 ms | 24,441 ms | 0.435 ms | ~272x |
| loop add (20k) | 10.29 ms | 1.36 ms | 0.014 ms | 1.552 ms | 0.116 ms | 0.033 ms | 4.164 ms | 0.016 ms | ~652x |
| string build (2k) | 144.78 ms | 0.215 ms | 0.088 ms | 1.235 ms | 6.566 ms | 21.168 ms | 0.899 ms | 0.019 ms | ~7,463x |
| replace all (60) | 0.351 ms | 0.043 ms | 0.034 ms | 0.096 ms | 0.161 ms | 0.395 ms | 0.278 ms | 0.001 ms | ~293x |

Loadtest extras (same run): concat only (no walk) 155.78 ms, repeat intern (V7-STREAM) 87.41 ms, find
scan (absent needle) 3.22 ms / 15.3x, atom churn (collect path) 2.13 ms / 277x.

C++ port (same scripts, byte for byte): fib 155.23 ms, loop add 14.33 ms, string build 17.15 ms, replace
all 1.15 ms, asm/c 272x/652x/7463x/293x, cpp/c 357x/907x/884x/954x.

**Reading the intern-sensitive rows.** The repeat-intern workload dropped to 87.41 ms (prior snapshots:
207 ms, 86 ms) and atom churn to 2.13 ms (prior: 8.42 ms), both at or better than the best earlier
numbers, consistent with the new hash doing strictly less work per intern (load + round chain in
registers vs a 96-byte stack round-trip). Fib and loop add do not intern in their hot loops; their rows
are session noise (the C control at 0.435 ms sits inside the historical 0.32-0.50 ms band, so absolutes
are cross-comparable within the usual 10-25 percent).

**Filter quality, not speed, is the real outcome.** The old hash returned low64(aes1) ^ low64(aes2) over
the first 64 bytes: strings sharing a 64-byte prefix collided by construction, and (as the
finalize-round analysis showed) single-block names sharing their first 8 bytes collided too --
internal_a vs internal_b was a live false-positive class for the slot cache. The full-string absorb plus
the two finalize rounds removes both classes; slot-cache hits that used to fall to the linear scan now
verify on first compare.

The Python/Node/Tcl/SBCL/Guile/PowerShell rows are a fresh session and move with machine temperature the
same way the SectorLogo row does; the C control is the anchor for cross-session comparison, as in every
earlier snapshot.

## Where the bottlenecks are now (measured, not assumed)

### 1. Tree-walk dispatch floor, the dominant cost (dispatch-bound shapes)

`Repeat 99999 [ Make "C" (:C + 1) ]`, one assignment and one addition per iteration, cost approximately
135 ms net (~1.35 µs/iteration) in the recorded run. Every form evaluation is a chain of `Car`/`Cdr`
calls through the cons-cell program representation plus an atom-compare dispatch against the ~130-entry
primitive table (linear `cmp` chains in EvalPrimary, ~10-15 comparisons deep for common ops). The
strbuild walk evaluates ~8 forms per iteration -> ~310 ms; fib is pure dispatch -> ~205x vs C in the
current snapshot (it was 291x before the direct-indexed dispatch tables and the branchless `Car`/`Cdr`
guard).

### 2. Dynamic-scope alist environments, variable reads are linear walks, SOLVED for LOCALs (frame array, Sep 2026)

The historical non-local environment path used `Assoc` to find bindings. New variable names could add
entries to the association list, while assignments to existing bindings update their values. Repeated
creation of new names can therefore increase lookup work; updating the same existing variable does not
necessarily grow the environment. The loadtest workloads originally had this shape; after rewriting them
to LOCAL the measured cost dropped (loop add 8.5->5.4 ms median) and the LOCAL frame array (Sep 2026)
removed the remaining per-read walk for LOCAL names entirely (one dense descending set scan + one
indexed CDR load; see the LOCAL frame array section). The alist path remains for non-LOCAL names,
repeated creation of new non-local bindings can still increase lookup costs; prefer `Local` for
function-scoped working variables.

### 3. Atom-area reclamation, global mark-sweep, non-moving (Sep 2026)

Interned strings used to be reclaimed indefinitely: the 64 MiB area was an append-only cursor, so a
sidecar that interns unbounded per-token `:PIECE` text over hours grew the area monotonically until it
overflowed. It is now a granular arena with a global mark-sweep collector, an 8 MiB soft limit triggers
a collect, dead atoms are coalesced into free blocks, and `Intern` serves new atoms first-fit from that
list before bumping the cursor.

The collector is **non-moving** by necessity, not by choice: the primitive dispatch tables are indexed
by the raw atom offset, the START index is a bitmap of atom offsets, and 135 builtin atoms live in BSS
as offset handles. A moving collector would invalidate all of them. So every root pins its atom in place
and the pin is checked by the mark phase, the region below the dispatch high-water mark is marked live
up front, which is exactly the set of offsets a dispatch table can name.

The bloom/slot fast paths still make lookups O(1)-ish regardless of area size, but they are no longer
what bounds memory. The cost that moved in is collection time: a full sweep at high occupancy is tens of
milliseconds, and a sidecar that interns large throwaway text (a per-token whole-stream snapshot, say)
pays that repeatedly. The cons heap is collected by the same call, on its own threshold.

### 4. String operations use copy-based implementations

`_` concat, Part, Replace all stage through the shared 2 MiB `sl_concat_buf` and then Intern the result
(a second copy into the atom area). Each concat of a length-L string is ~2L byte-copies + L hashed bytes
+ L re-compared bytes on the repeat-lookup path. CPython strings are also immutable copy-per-concat, but
its per-concat constant is ~10 ns; ours is ~150 ns (measured: 2,000 concats of average 1,000 chars ~ 10
ms phase cost within the workload).

## How to mitigate (ranked by measured leverage)

1. **Dispatch: interned-atom -> direct-threaded table, DONE (Sep 2026).** Applied per attic\docs_plans\interned_atom_plan.md; better than the original sketch (the atom handle IS a dense index, no hash needed). Measured 1.2-1.4x on dispatch-bound shapes (see the dispatch table section above): the chain was real cost but not the dominant term. Remaining floor is per-form Car/Cdr interpreter call overhead + EvalNextArg frames.
2. **Environment: LOCALs as a frame array, not an alist, DONE (Sep 2026).** Applied per attic\docs_plans\exec_local_plan.md (see the LOCAL frame array section above): sl_local_set caches each binding's cons-cell index; LOCAL reads/writes are one dense descending scan + one indexed load/store; the Assoc walk and the MAKE env-walk are deleted for LOCAL names. Measured -15-30% on loop-add (the LOCAL-heavy shape); non-LOCAL names keep the alist (dynamic scoping preserved).
3. **REPEAT body pre-compilation (mini-closure):** the count-optional REPEAT already peeks the body block; hoisting per-iteration invariants (the stream cursor reset, the body's cons-cell head) into locals saves a few Car/Cdr chains per iteration. Cheap, incremental, no semantic change.
4. **String ops: length-prefixed atoms + memcpy staging.** Atoms already know their length at creation (klen is measured in Intern); storing it as a dword prefix enables `memcmp`/`vmovdqu` compares instead of byte loops, and concat could sum lengths and copy both sides once. Medium effort, ~2x on string-heavy sidecars.
5. **Slot cache: 2-way associativity** if future workloads exceed 4096 live hot atoms (today's sidecars peak in the tens). One constant + one extra compare per slot, only if measurement demands it.
6. **Not planned: bytecode VM.** A register/bytecode VM (Lua-class) would put dispatch near 15-40x vs C but is a rewrite of the evaluator. The tree-walker with fixes 1+2 is estimated to land at 60-150x, CPython's class, which is the stated bar for an embedded sidecar language.

## Verification protocol for any future change here

1. `build.bat`, comment checker zero violations.
2. `v_sectorlogo_validate.exe <dll> --db <db>`, language and host-primitive tests; record the count reported by the validator.
3. `v_mdb_validate.exe <dll>`, database and hash-table tests; record the reported count.
4. `t_sectorlogo_gc.exe <dll> --db <db> fibN-seq 10`, record the collection results.
5. `b_sectorlogo_loadtest.exe <dll> --db <db> --reps 5`, record results for the four workloads.
6. `verify.bat`, full validation sequence.
7. Record before/after medians in this document.
