@echo off
rem ============================================================================
rem run_paper_p1.bat -- P1 supplementary measurements
rem
rem P1-2: native C-reference validator (matmul Q8+F32) with full output.
rem P1-4: BM25 SectorLogo benchmark, 5 reps, phase timings.
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo [1/2] P1-2 v_matmul_validate
tools\v_matmul_validate.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db > %OUT%\p1_matmul_validate.txt 2>&1
if %errorlevel% NEQ 0 goto :err

echo [2/2] P1-4 bm25_sectorlogo_benchmark
bin\bm25_sectorlogo_benchmark.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db examples\sectorlogo\bm25_bench.logo.template 5 > %OUT%\p1_bm25_bench.txt 2>&1
if %errorlevel% NEQ 0 goto :err

echo P1_DONE
exit /b 0

:err
echo P1_FAILED
exit /b 1
