@echo off
rem matmul_shape_bench.bat -- Q8 matmul GB/s for all Qwen3-0.6B shapes.
rem Runs every shape against a physical-cores DB and an SMT DB, twice each,
rem writes the raw logs, then prints them so both modes can be compared.
rem DBs are baked with use_physical_cores=1 (matmul_phys.db) and =0 (matmul_smt.db).
rem Qwen3-0.6B shape set = 1024:512 1024:1024 1024:2048 2048:1024 1024:3072
rem   3072:1024 151936:1024 1024:151936  (the fused QKV 1024:4096 is not in the
rem   b_loadtest shape table, so it is not benchmarkable with this tool).
setlocal
cd /d %~dp0..\
set LOGDIR=v7_speed_results\matmul_shapes
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo === physical cores, run 1 ===
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_phys.db --q8 1024:512 --time 2 >> "%LOGDIR%\phys_r1.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_phys.db --q8 1024:1024 --time 2 >> "%LOGDIR%\phys_r1.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_phys.db --q8 1024:2048 --time 2 >> "%LOGDIR%\phys_r1.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_phys.db --q8 2048:1024 --time 2 >> "%LOGDIR%\phys_r1.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_phys.db --q8 1024:3072 --time 2 >> "%LOGDIR%\phys_r1.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_phys.db --q8 3072:1024 --time 2 >> "%LOGDIR%\phys_r1.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_phys.db --q8 151936:1024 --time 2 >> "%LOGDIR%\phys_r1.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_phys.db --q8 1024:151936 --time 2 >> "%LOGDIR%\phys_r1.txt" 2>&1

echo === physical cores, run 2 ===
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_phys.db --q8 1024:512 --time 2 >> "%LOGDIR%\phys_r2.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_phys.db --q8 1024:1024 --time 2 >> "%LOGDIR%\phys_r2.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_phys.db --q8 1024:2048 --time 2 >> "%LOGDIR%\phys_r2.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_phys.db --q8 2048:1024 --time 2 >> "%LOGDIR%\phys_r2.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_phys.db --q8 1024:3072 --time 2 >> "%LOGDIR%\phys_r2.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_phys.db --q8 3072:1024 --time 2 >> "%LOGDIR%\phys_r2.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_phys.db --q8 151936:1024 --time 2 >> "%LOGDIR%\phys_r2.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_phys.db --q8 1024:151936 --time 2 >> "%LOGDIR%\phys_r2.txt" 2>&1

echo === SMT (all logical), run 1 ===
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_smt.db --q8 1024:512 --time 2 >> "%LOGDIR%\smt_r1.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_smt.db --q8 1024:1024 --time 2 >> "%LOGDIR%\smt_r1.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_smt.db --q8 1024:2048 --time 2 >> "%LOGDIR%\smt_r1.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_smt.db --q8 2048:1024 --time 2 >> "%LOGDIR%\smt_r1.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_smt.db --q8 1024:3072 --time 2 >> "%LOGDIR%\smt_r1.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_smt.db --q8 3072:1024 --time 2 >> "%LOGDIR%\smt_r1.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_smt.db --q8 151936:1024 --time 2 >> "%LOGDIR%\smt_r1.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_smt.db --q8 1024:151936 --time 2 >> "%LOGDIR%\smt_r1.txt" 2>&1

echo === SMT (all logical), run 2 ===
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_smt.db --q8 1024:512 --time 2 >> "%LOGDIR%\smt_r2.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_smt.db --q8 1024:1024 --time 2 >> "%LOGDIR%\smt_r2.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_smt.db --q8 1024:2048 --time 2 >> "%LOGDIR%\smt_r2.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_smt.db --q8 2048:1024 --time 2 >> "%LOGDIR%\smt_r2.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_smt.db --q8 1024:3072 --time 2 >> "%LOGDIR%\smt_r2.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_smt.db --q8 3072:1024 --time 2 >> "%LOGDIR%\smt_r2.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_smt.db --q8 151936:1024 --time 2 >> "%LOGDIR%\smt_r2.txt" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\matmul_smt.db --q8 1024:151936 --time 2 >> "%LOGDIR%\smt_r2.txt" 2>&1

echo.
echo ============================================================
echo  RESULTS - physical cores (run1, run2)
echo ============================================================
type "%LOGDIR%\phys_r1.txt" | findstr "Pass:"
type "%LOGDIR%\phys_r2.txt" | findstr "Pass:"
echo.
echo ============================================================
echo  RESULTS - SMT all-logical (run1, run2)
echo ============================================================
type "%LOGDIR%\smt_r1.txt" | findstr "Pass:"
type "%LOGDIR%\smt_r2.txt" | findstr "Pass:"
echo.
echo MATMUL_SHAPE_BENCH_DONE
exit /b 0
