@echo off
REM ============================================================================
REM bench_phi3_retune.bat -- runner for the phi3-mini re-benchmark
REM
REM Repeats the phi3-mini legs of the overnight benchmark (short chat turn +
REM 500-token poem, 5 rounds, v7 / OpenVINO / llama.cpp / MNN -- 8-bit engines
REM only, same engine order as the overnight run) after a v7 re-tune, and
REM saves EVERYTHING into bench_results_phi3_retune.txt. Reruns APPEND to
REM that file; every section carries a timestamp.
REM Safe to leave unattended: no prompts, no pauses, a failing leg is
REM reported in the log and the run continues.
REM ============================================================================
setlocal

cd /d %~dp0..\

echo Results are saved to bench_results_phi3_retune.txt
echo The run starts now and needs no input until it finishes.

call "%~dp0bench_phi3_retune_run.bat" >> bench_results_phi3_retune.txt 2>&1

echo.
echo ============================================================
echo  PHI3 RETUNE BENCH FINISHED
echo  Results: bench_results_phi3_retune.txt
echo ============================================================
exit /b 0
endlocal
