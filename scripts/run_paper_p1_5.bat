@echo off
rem ============================================================================
rem run_paper_p1_5.bat -- P1-5 load time + memory footprint
rem Two runs: first (cold-ish FS cache) and second (warm FS cache), each a
rem fresh process loading the 631 MB compressed database. -n 32 keeps the
rem turn short so the post-turn snapshot lands quickly.
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
set EXE=build_logs\paper_measurements\inference_measure.exe

echo [1/2] first load (cold-ish)
%EXE% bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "What is a cat?" -n 32 > %OUT%\load_cold.txt 2>&1
if %errorlevel% NEQ 0 goto :err
echo [2/2] second load (warm FS cache)
%EXE% bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "What is a cat?" -n 32 > %OUT%\load_warm.txt 2>&1
if %errorlevel% NEQ 0 goto :err

echo P1_5_DONE
exit /b 0

:err
echo P1_5_FAILED
exit /b 1
