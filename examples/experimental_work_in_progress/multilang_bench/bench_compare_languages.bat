@echo off
cd /d "%~dp0..\..\.."

rem SectorLogo vs CPython vs Node vs Tcl vs PowerShell vs SBCL comparison
rem (same workloads, same reps, interleaved+median methodology where applicable).
rem Run from project root (this bat cd's there itself). Results print to console -- the .md doc records them.
rem Scheme is NOT benchmarked here: the v7 scheme work moved to scheme_internal\
rem (dev-only, not part of the release).

rem SBCL (native-compiling Common Lisp) -- MSI default path, bare sbcl as fallback
set "SBCL_EXE=C:\Program Files\Steel Bank Common Lisp\sbcl.exe"
where sbcl >nul 2>nul && set "SBCL_EXE=sbcl"

echo === SectorLogo (v7s_sectorlogo_interp, ASM tree-walker) ===
tools\b_sectorlogo_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --reps 5

echo.
echo === Python 3.12 (CPython) ===
python examples\experimental_work_in_progress\multilang_bench\multilang_bench\bench_fib.py 5

echo.
echo === Node.js 24 (V8 JIT) ===
node examples\experimental_work_in_progress\multilang_bench\multilang_bench\bench_fib.js 5

echo.
echo === Tcl 9.0 ===
tclsh examples\experimental_work_in_progress\multilang_bench\multilang_bench\bench_fib.tcl 5

echo.
echo === SBCL 2.6 (native Common Lisp compiler) ===
"%SBCL_EXE%" --script examples\experimental_work_in_progress\multilang_bench\multilang_bench\bench_fib.lsp 5

echo.
echo === PowerShell 5.1 ===
powershell -NoProfile -ExecutionPolicy Bypass -File examples\experimental_work_in_progress\multilang_bench\multilang_bench\bench_fib.ps1 5
