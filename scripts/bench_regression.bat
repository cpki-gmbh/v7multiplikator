@echo off
REM bench_regression.bat -- standalone GB/s benchmark over all dispatched kernels.
REM
REM NOT part of verify.bat (adding it there would roughly double the verify
REM runtime). Run it by hand when kernel/dispatch code changed and compare the
REM printed GB/s numbers against the recorded bests quoted in AGENTS.md
REM Sec.0/Sec.28. Any kernel consistently >20% below its best is a regression
REM (AGENTS.md Sec.54) and must be investigated.
REM
REM Usage: bench_regression.bat [db_path]
REM   [db_path]  defaults to weights\databases\qwen3_0.6b\lz4ht.db (the DB
REM              verify.bat builds). Build it first if missing.

setlocal ENABLEDELAYEDEXPANSION
set THIS_DIR=%~dp0..\
cd /d "%THIS_DIR%"
set DB=%THIS_DIR%weights\databases\qwen3_0.6b\lz4ht.db
if not "%~1"=="" set DB=%~1
if not exist "%DB%" (
    echo [FATAL] database not found: %DB%
    echo Run verify.bat once -- it builds the Qwen3 DB -- or pass a db_path.
    exit /b 1
)

set FAILS=0
echo ============================================================
echo  GB/S BENCHMARK - all dispatched kernels
echo  DB: %DB%
echo ============================================================

echo --- Q8 matmul ---
for %%s in (1024:1024 1024:2048 2048:1024 1024:3072 3072:1024 151936:1024 1024:151936) do (
    tools\b_loadtest.exe bin\v7_threading.dll --db "%DB%" --q8 %%s --time 1
    if !errorlevel! NEQ 0 set /a FAILS+=1
)

echo --- F32 matmul ---
for %%s in (1024:1024 1024:2048 2048:1024 1024:3072 3072:1024 151936:1024 1024:151936) do (
    tools\b_loadtest.exe bin\v7_threading.dll --db "%DB%" --f32 %%s --time 1
    if !errorlevel! NEQ 0 set /a FAILS+=1
)

echo --- fused ---
for %%c in (5 63 1023) do (
    tools\b_fused_loadtest.exe bin\v7_threading.dll 1024 2048 128 %%c --db "%DB%" --time 1
    if !errorlevel! NEQ 0 set /a FAILS+=1
)

echo --- swiglu ---
for %%s in (1024 3072) do (
    tools\b_swiglu_loadtest.exe bin\v7_threading.dll 1024 %%s --db "%DB%" --time 1
    if !errorlevel! NEQ 0 set /a FAILS+=1
    tools\b_swiglu_loadtest.exe bin\v7_threading.dll 3072 1024 --db "%DB%" --time 1
    if !errorlevel! NEQ 0 set /a FAILS+=1
)

echo --- swiglu q8 ---
for %%s in (1024 3072) do (
    tools\b_swiglu_q8_loadtest.exe bin\v7_threading.dll 1024 %%s --db "%DB%" --time 1
    if !errorlevel! NEQ 0 set /a FAILS+=1
    tools\b_swiglu_q8_loadtest.exe bin\v7_threading.dll 3072 1024 --db "%DB%" --time 1
    if !errorlevel! NEQ 0 set /a FAILS+=1
)

echo --- rmsnorm ---
tools\b_rmsnorm_loadtest.exe bin\v7_threading.dll 1024 --db "%DB%" --time 1
if !errorlevel! NEQ 0 set /a FAILS+=1
tools\b_rmsnorm_rope_loadtest.exe bin\v7_threading.dll 16 128 5 64 --db "%DB%" --time 1
if !errorlevel! NEQ 0 set /a FAILS+=1
tools\b_rmsnorm_rope_loadtest.exe bin\v7_threading.dll 8 128 5 64 --db "%DB%" --time 1
if !errorlevel! NEQ 0 set /a FAILS+=1

echo --- fused add rmsnorm ---
tools\b_fused_add_rmsnorm_loadtest.exe bin\v7_threading.dll 1024 --db "%DB%" --time 1
if !errorlevel! NEQ 0 set /a FAILS+=1

echo --- add inplace ---
for %%d in (1024 3072) do (
    tools\b_add_inplace_loadtest.exe bin\v7_threading.dll %%d --db "%DB%" --time 1
    if !errorlevel! NEQ 0 set /a FAILS+=1
)

echo --- memcpy ---
tools\b_memcpy_loadtest.exe bin\v7_threading.dll 1048576 --db "%DB%" --time 1
if !errorlevel! NEQ 0 set /a FAILS+=1

echo --- attention ---
tools\b_attention_loadtest.exe bin\v7_threading.dll --db "%DB%"
if !errorlevel! NEQ 0 set /a FAILS+=1

echo ============================================================
if !FAILS! NEQ 0 (
    echo  DONE with !FAILS! failed loadtest runs -- investigate before trusting the numbers
    exit /b 1
)
echo  DONE - all loadtest runs exited 0
exit /b 0
