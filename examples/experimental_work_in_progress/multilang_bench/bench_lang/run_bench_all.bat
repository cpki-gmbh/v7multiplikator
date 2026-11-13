@echo off
rem ============================================================================
rem Multi-language interpreter benchmark:
rem   Python vs Node.js vs SectorLogo64 (ASM) vs SectorLogo C++ port
rem Same 4 workloads (fib 24, loop 20000, strbuild 2000, replace-all 60).
rem SectorLogo prints raw ms samples per workload (5 each); medians computed
rem from the printed samples.
rem
rem All four print RAW samples, and the reduction is left to the reader, so the
rem four are the same measurement read the same way. The C++ port is the odd one
rem out on WHERE the clock comes from: the other three can time themselves,
rem because the ASM's C-CALL FFI reaches kernel32, while the C++ port's FFI is a
rem deliberate stub (it mirrors the ASM's resolve failure rather than resolving a
rem DLL, and says so at the stub in sectorlogo.c). So the C++ port is driven with
rem `--bench`, which times the script from the host, and its samples are already
rem divided by the inner iteration count -- which is 1 for every workload, because
rem each one is milliseconds on its own and does not need the repetition that the
rem ASM's ~15 ms system tick forces on it. Same quantity, better clock.
rem
rem The four workload scripts in bench_cpp\ are the SAME functions and shapes as
rem the ones in bench_sectorlogo.logo, deliberately character for character: a
rem benchmark comparing two interpreters has to run the same script on both, or it
rem measures the scripts. The one deliberate difference is that the hay in
rem replace.logo is a string literal, because the C++ port times the whole script
rem and a sixty-concat setup inside it would land in the sample; that file says so.
rem
rem Requires sectorlogo_c\sectorlogo_rewrite_test.exe, which build_cpp.bat --test
rem produces. Nothing here builds anything.
rem
rem -------- READ THE ASM NUMBERS BEFORE COMPARING THEM --------
rem Python, Node and the C++ port each time ONE call per sample, so their numbers
rem are already per-call. The ASM does NOT: its in-script clock is GetTickCount64,
rem whose resolution is the ~15 ms system tick, so it repeats each workload until
rem a sample clears the tick and prints the TOTAL for the repetitions. Divide its
rem printed samples by these before putting them in a row with the other three:
rem
rem     recursive fib     x1        loop add    x100
rem     string build      x5        replace all x1000
rem
rem The inner counts are at the bottom of bench_sectorlogo.logo, and the reason
rem they are there is in that file's header. This is the one place the four legs
rem are not in the same unit, and it is a property of the clock rather than of
rem the language, so it is stated here instead of left for the reader to discover.
rem ============================================================================
setlocal
cd /d %~dp0..\..\..\..

rem One sample count for all four, stated rather than defaulted: a number of
rem samples is a claim about the comparison, and one that differed per language
rem would quietly make the table disagree with itself.
set SAMPLES=5

echo === python ===
python examples\experimental_work_in_progress\multilang_bench\bench_lang\bench_fib.py
if errorlevel 1 echo python bench FAILED

echo === node ===
node examples\experimental_work_in_progress\multilang_bench\bench_lang\bench_fib.js
if errorlevel 1 echo node bench FAILED

echo === sectorlogo (asm) ===
bin\run_sectorlogo.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db examples\experimental_work_in_progress\multilang_bench\bench_lang\bench_sectorlogo.logo
if %errorlevel% NEQ 0 echo sectorlogo asm bench FAILED

echo === sectorlogo c++ port ===
echo -- recursive fib --
sectorlogo_c\sectorlogo_rewrite_test.exe --bench examples\experimental_work_in_progress\multilang_bench\bench_lang\bench_cpp\fib.logo %SAMPLES% 1
if %errorlevel% NEQ 0 echo sectorlogo c++ fib bench FAILED
echo -- loop add --
sectorlogo_c\sectorlogo_rewrite_test.exe --bench examples\experimental_work_in_progress\multilang_bench\bench_lang\bench_cpp\loop.logo %SAMPLES% 1
if %errorlevel% NEQ 0 echo sectorlogo c++ loop bench FAILED
echo -- string build (concat) --
sectorlogo_c\sectorlogo_rewrite_test.exe --bench examples\experimental_work_in_progress\multilang_bench\bench_lang\bench_cpp\strbuild.logo %SAMPLES% 1
if %errorlevel% NEQ 0 echo sectorlogo c++ strbuild bench FAILED
echo -- replace all --
sectorlogo_c\sectorlogo_rewrite_test.exe --bench examples\experimental_work_in_progress\multilang_bench\bench_lang\bench_cpp\replace.logo %SAMPLES% 1
if %errorlevel% NEQ 0 echo sectorlogo c++ replace bench FAILED
