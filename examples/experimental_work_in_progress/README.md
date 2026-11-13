# experimental_work_in_progress

Self-contained experimental projects. Each subfolder owns its code, its
scripts, its docs, and its drivers -- nothing here is part of the shipped
engine, and nothing in here is referenced by build.bat or verify.bat.

## multilang_bench/

Multi-language interpreter benchmark (Python vs Node.js vs Tcl vs SBCL vs
PowerShell vs SectorLogo ASM) over four shared workloads (fib 24, loop
20000, strbuild 2000, replace-all 60). Scheme is not benchmarked here:
the v7 scheme work lives in `scheme_internal\` at the repo root
(dev-only, needs fixing, not part of the release).

- `multilang_bench\`      -- per-language fib benchmarks + README
- `bench_lang\`           -- the ASM-vs-C++-port driver: bench_sectorlogo.logo,
                             the four bench_cpp\ Logo workloads, and
                             run_bench_all.bat (needs sectorlogo_c\ built)
- `bench_compare_languages.bat` -- the language comparison driver;
                             cd's to the project root itself

Results and methodology are recorded in `docs\sectorlogo_performance.md`.
