# SectorLogo multi-language benchmarks

Benchmarks SectorLogo (the v7 ASM tree-walking interpreter) against the other
scripting languages installed on this machine, running identical workloads with
identical methodology.

## Installed languages benchmarked

| Language | Version | Implementation class |
|---|---|---|
| SectorLogo | v7 current | ASM tree-walker, cons cells, no bytecode, no JIT |
| SectorLogo C++ port | same semantics | C++17 tree-walker, the readable reference for the ASM |
| Python | 3.12.10 | CPython 3.12 bytecode VM, no JIT |
| Node.js | 24.14.1 | V8 JIT (tiered, optimized after warmup) |
| Tcl | 9.0.3 | Tcl 9 bytecode VM |
| SBCL | 2.6.8 | Steel Bank Common Lisp, native compiler (`(speed 3)`), installed via winget |
| GNU Guile | 3.0.11 | Scheme compiled to G-code, JIT-backed VM, no native compiler; installed via MSYS2 (`pacman -S guile`) |
| PowerShell | 5.1.26100 | .NET interpreter (no JIT for cmdlet-bound recursion) |

No other scripting language runtimes are installed on this machine (no Perl,
Ruby, Lua, PHP).

Guile has no official Windows build — the GNU FTP serves source tarballs only
— so it comes from the MSYS2 distribution. Two consequences for running it:

- Guile resolves its script argument POSIX-style, so `bench_compare_languages.bat`
  passes the `/c/...` path form it expects, not the Windows `C:\...` form.
- The bat prefers `C:\msys64\usr\bin\guile.exe` and falls back to a bare
  `guile` on `PATH`.

The C++ port is not a seventh language. It is the second implementation of
SectorLogo, and it runs the SectorLogo row's workload scripts **byte for byte**,
because every `logo_*_work()` sprintf's its script into `script_buf` and runs
that buffer, so whatever is in `script_buf` afterwards is exactly what the ASM
just ran, and the port is handed the same buffer. That makes the pair the
informative one: `asm/c` and `cpp/c` ask the same question of two
implementations of the same semantics, which is what separates "this algorithm
costs this much" from "the hand-assembly is this much faster than the same
algorithm written to be read".

## Files

- `bench_fib.py` — CPython workload (4 tests, warmup + N reps, median)
- `bench_fib.js` — Node workload (same)
- `bench_fib.tcl` — Tcl workload (microsecond timer via `clock clicks -microseconds`)
- `bench_fib.lsp` — SBCL workload (`sbcl --script`, `get-internal-real-time` timing, `(speed 3)` declaim)
- `bench_fib.scm` — GNU Guile workload (`guile -s`, same 4 workloads, native `string-count`, `get-internal-real-time` nanosecond timing)
- `bench_fib.ps1` — PowerShell workload (Stopwatch timing, invariant number format)
- `..\..\..\tools\b_sectorlogo_loadtest.cpp` — the SectorLogo AND C++ port leg: three columns (asm, c++ port, C reference), interleaved per rep
- `..\bench_compare_languages.bat` (in `multilang_bench\`, cd's to the project
  root itself) — runs everything, prints to console

## Workloads (identical algorithms in every language)

| Workload | What it stresses | Scale |
|---|---|---|
| recursive fib | function calls and interpreter dispatch | fib(24), about 150k calls |
| loop add | loop + local variable update | 20,000 iterations |
| string build | one `_` concat per step, then count chars over the result | 2,000 concats |
| replace all | `Search` and `Replace` loop | 60 occurrences |

Each rep verifies the expected result; a wrong result aborts the run.

## Current results (measured 2026-10-09, idle priority, 5 reps, median)

Medians in milliseconds per call. The ASM numbers below are ALREADY divided
by the inner repetition counts (fib x1, loop add x100, string build x5,
replace all x1000) that keep each GetTickCount64 sample above tick
resolution -- see `run_bench_all.bat`'s header for why those counts exist.

| Workload | SectorLogo (ASM) | C++ port | C ref | Python 3.12 | Node 24 | Tcl 9.0 | SBCL 2.6 | Guile 3.0 | PowerShell 5.1 |
|---|---|---|---|---|---|---|---|---|---|
| recursive fib(24) | 47.3 | 55.4 | 0.219 | 8.02 | 0.53 | 33.7 | 0.80 | 1.41 | 19990.6 |
| loop add (20k) | 3.25 | 6.34 | 0.007 | 1.13 | 0.04 | 1.32 | 0.11 | 0.03 | 5.34 |
| string build (2k) | 71.1 | 9.90 | 0.012 | 0.19 | 0.06 | 1.11 | 6.54 | 1.22 | 0.89 |
| replace all (60) | 0.112 | 0.667 | 0.001 | 0.039 | 0.032 | 0.087 | 0.157 | 0.122 | 0.334 |

Reading the table:

- **SectorLogo vs the C++ port** is the informative pair (same semantics,
  same scripts): the ASM wins the call-heavy and list-heavy workloads
  (fib 1.17x, loop add 1.95x, replace all 6.0x, find scan 2.1x, atom churn
  n/a) and loses string building badly (8x slower than the port) -- the
  port's string path shares the same concat buffer but skips the walk, and
  the ASM's `_`-concat still pays a full stream walk per step.
- **SectorLogo vs scripting languages**: comfortably faster than PowerShell
  everywhere and faster than Tcl on fib; around Python's class on loops
  (3.25 us/iter vs 1.13) and slower than the JIT/native tier (Node, SBCL,
  Guile) by one to two orders of magnitude, which is the expected gap for a
  tree-walker without bytecode.
- **replace all at 0.112 ms** reflects the native `Search`/`Replace`
  primitives; the string-build row is the one open optimization target
  (per-step stream walk), everything else is in the expected band for the
  implementation class.

## Run it

From the project root:

```
examples\experimental_work_in_progress\multilang_bench\bench_compare_languages.bat
```

The bat cd's to the project root itself; the ASM leg needs
`bin\v7_threading.dll` and the Qwen3 database, the C++ port leg needs
`sectorlogo_c\sectorlogo_rewrite_test.exe` (built by
`sectorlogo_c\build_cpp.bat --test`).

The SectorLogo side interleaves TWO interpreters and a hand-written C reference
(stdlib only) in the same process — the ASM, the C++ port, and C — so thermal
drift cancels per rep and both slowdown ratios are printed live. All three are
gated against the expected value on every rep, so a port that computes something
else fails the benchmark rather than posting a fast wrong number. The other
languages are timed by their own monotonic high-resolution clocks.

The C++ port leg needs `sectorlogo_c\sectorlogo_rewrite_test.exe`'s engine
objects, which `build.bat` compiles into the loadtest directly with
`SECTORLOGO_PORT_NO_MAIN` and the same `/O2 /arch:AVX2` as the rest of the tool
chain, so it is measured at the optimisation level it is built at everywhere
else.

Results and analysis live in `docs/sectorlogo_performance.md`.
