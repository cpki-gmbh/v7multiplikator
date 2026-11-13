REM ============================================================================
REM bench_all_models_overnight.bat -- runner for the unattended benchmark
REM
REM Runs bench_all_models_overnight_run.bat (all 4 models, short + long
REM workload, 5 rounds, v7 / OpenVINO / llama.cpp / MNN -- 8-bit engines
REM only, no 4-bit legs) and saves EVERYTHING into bench_results_overnight.txt.
REM Reruns APPEND to that file; every section in the log carries a timestamp.
REM Safe to leave unattended: no prompts, no pauses, a failing leg is
REM reported in the log and the run continues with the next leg.
REM ============================================================================
setlocal

cd /d %~dp0..\

echo Results are saved to bench_results_overnight.txt
echo The run starts now and needs no input until it finishes.

call "%~dp0bench_all_models_overnight_run.bat" >> bench_results_overnight.txt 2>&1

echo.
echo ============================================================
echo  OVERNIGHT BENCH FINISHED
echo  Results: bench_results_overnight.txt
echo ============================================================
exit /b 0
endlocal
