@echo off
rem ============================================================================
rem bench_qwen.bat -- Qwen3-0.6B inference benchmark via inference_example.exe
rem
rem Runs the chat pipeline (tokenize -> prefill -> generate) on the qwen3_0.6b
rem database in BOTH quantization modes, 3 rounds each. Rounds act as burn-in:
rem round 1 shows cold-cache numbers, later rounds show steady state (see
rem performance_report.md). Per-phase RDTSC statistics print after each round.
rem
rem Usage:  bench_qwen.bat [rounds]     (default rounds = 3)
rem ============================================================================
setlocal
set ROUNDS=%1
if "%ROUNDS%"=="" set ROUNDS=3

rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1

cd /d %~dp0..\

echo ============================================================
echo  Qwen3-0.6B benchmark -- F32 mode (%ROUNDS% rounds)
echo ============================================================
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "What is a cat?" -n 200 -r %ROUNDS%
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  Qwen3-0.6B benchmark -- Q8 mode (%ROUNDS% rounds)
echo ============================================================
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "What is a cat?" -n 200 -r %ROUNDS% -q8
if %errorlevel% NEQ 0 goto :err

echo.
echo BENCH_DONE
exit /b 0

:err
echo BENCH_FAILED
exit /b 1
