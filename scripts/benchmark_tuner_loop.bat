
REM benchmark_tuner_loop.bat -- Infinite loop: build all DBs with tuner-delay 0,
REM run overnight bench, rebuild with delay 99999, run overnight bench, repeat forever.
REM
REM This runs forever. Ctrl+C to stop Each cycle:
REM   1. Build all DBs with tuner-delay 0 / rand 1 (tuner active)
REM   2. Run overnight benchmark (all models, 5 rounds each)
REM   3. Build all DBs with delay 99999 / rand 0 (tuner quiet)
REM   4. Run overnight benchmark (all models, 5 rounds each)
REM   Repeat forever.
REM
REM Logs go to benchmark_tuner_loop_TIMESTAMP.txt in the working directory.
REM
REM Prerequisites:
REM   - build.bat must have been run (build_db.exe + v7_threading.dll)
REM   - All model .bin files in weights\<model>\
REM   - overnight benchmark bats present: bench_all_models_overnight.bat,
REM     bench_all_engines_phi3.bat, bench_all_engines_minicpm5.bat,
REM     bench_all_engines_qwen17b.bat (and poem variants)
REM
REM WARNING: This runs forever. Stop with Ctrl+C.

setlocal
set THIS_DIR=%~dp0..\
set TIMESTAMP=%DATE:~6,4%%DATE:~3,2%%DATE:~0,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set TIMESTAMP=%TIMESTAMP::=%
set LOGFILE=benchmark_tuner_loop_%TIMESTAMP%.txt

echo ============================================================================
echo  BENCHMARK TUNER LOOP STARTED %DATE% %TIME%
echo  Log: %LOGFILE%
echo ============================================================================

echo Loop sequence:
echo   1. Build DBs with tuner-delay 0 / rand 1 (ACTIVE)
echo   2. Run overnight benchmark (all 4 models, short + poem, 5 rounds)
echo   3. Build DBs with tuner-delay 99999 / rand 0 (QUIET)
echo   4. Run overnight benchmark (all 4 models, short + poem, 5 rounds)
echo   Repeat forever

echo Press Ctrl+C to stop

echo Log file: %LOGFILE%


:LOOP_START
set CYCLE=%DATE% %TIME%

echo ============================================================================
echo  CYCLE START %CYCLE%
echo ============================================================================
echo %CYCLE% >> %LOGFILE% 2>&1

REM ============================================================
REM PHASE 1: Build DBs with tuner-delay 0 / rand 1 (ACTIVE)
REM ============================================================

echo ============================================================================
echo  PHASE 1: BUILD DBs WITH TUNER-DELAY 0 / RAND 1 (ACTIVE)
echo ============================================================================
echo %CYCLE% PHASE 1: BUILD DBs DELAY 0 / RAND 1 >> %LOGFILE% 2>&1
call "%~dp0build_all_text_llm_delay_0_rand_1.bat" >> %LOGFILE% 2>&1
if %errorlevel% NEQ 0 (
    echo [FAIL] Build phase 1 failed. Continuing to benchmark anyway
    echo [FAIL] Build phase 1 failed >> %LOGFILE% 2>&1
) else (
    echo [OK] Build phase 1 complete
    echo [OK] Build phase 1 complete >> %LOGFILE% 2>&1
)

REM 3-minute pause after build for system to settle
echo Waiting 3 minutes for system to settle
timeout /t 180 /nobreak >nul 2>&1

REM ============================================================
REM PHASE 2: Run overnight benchmark (TUNER ACTIVE)
REM ============================================================

echo ============================================================================
echo  PHASE 2: OVERNIGHT BENCHMARK (TUNER ACTIVE)
echo ============================================================
echo %CYCLE% PHASE 2: BENCHMARK ACTIVE >> %LOGFILE% 2>&1
call "%~dp0bench_all_models_overnight.bat" >> %LOGFILE% 2>&1
if %errorlevel% NEQ 0 (
    echo [FAIL] Benchmark phase 2 had failures Continuing
    echo [FAIL] Benchmark phase 2 had failures >> %LOGFILE% 2>&1
) else (
    echo [OK] Benchmark phase 2 complete
    echo [OK] Benchmark phase 2 complete >> %LOGFILE% 2>&1
)

REM ============================================================
REM PHASE 3: Build DBs with tuner-delay 99999 / rand 0 (QUIET)
REM ============================================================

echo ============================================================================
echo  PHASE 3: BUILD DBs WITH TUNER-DELAY 99999 / RAND 0 (QUIET)
echo ============================================================
echo %CYCLE% PHASE 3: BUILD DBs DELAY 99999 / RAND 0 >> %LOGFILE% 2>&1
call "%~dp0build_all_text_llm_delay_9999.bat" >> %LOGFILE% 2>&1
if %errorlevel% NEQ 0 (
    echo [FAIL] Build phase 3 failed. Continuing to benchmark anyway
    echo [FAIL] Build phase 3 failed >> %LOGFILE% 2>&1
) else (
    echo [OK] Build phase 3 complete
    echo [OK] Build phase 3 complete >> %LOGFILE% 2>&1
)

REM 3-minute pause after build for system to settle
echo Waiting 3 minutes for system to settle
timeout /t 180 /nobreak >nul 2>&1

REM ============================================================
REM PHASE 4: Run overnight benchmark (TUNER QUIET)
REM ============================================================

echo ============================================================================
echo  PHASE 4: OVERNIGHT BENCHMARK (TUNER QUIET)
echo ============================================================
echo %CYCLE% PHASE 4: BENCHMARK QUIET >> %LOGFILE% 2>&1
call "%~dp0bench_all_models_overnight.bat" >> %LOGFILE% 2>&1
if %errorlevel% NEQ 0 (
    echo [FAIL] Benchmark phase 4 had failures Continuing
    echo [FAIL] Benchmark phase 4 had failures >> %LOGFILE% 2>&1
) else (
    echo [OK] Benchmark phase 4 complete
    echo [OK] Benchmark phase 4 complete >> %LOGFILE% 2>&1
)

REM ============================================================
REM LOOP BACK
REM ============================================================

echo ============================================================================
echo  CYCLE COMPLETE %DATE% %TIME%
echo  Waiting 30 seconds before next cycle
echo ============================================================================
echo %CYCLE% CYCLE COMPLETE >> %LOGFILE% 2>&1

REM Small pause so Ctrl+C can be caught
timeout /t 30 /nobreak >nul 2>&1

goto LOOP_START

endlocal