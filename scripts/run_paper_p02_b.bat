@echo off
rem ============================================================================
rem run_paper_p02_b.bat -- P0-2 kernel loadtests, part B: Q8 151936x1024 +
rem 1024x151936 (the two larger-than-cache streaming shapes)
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo [1/2] Q8 151936:1024
echo cmd: tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 151936:1024 --time 1 > %OUT%\kernel_q8_151936x1024.txt
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 151936:1024 --time 1 >> %OUT%\kernel_q8_151936x1024.txt 2>&1
if %errorlevel% NEQ 0 goto :err

echo [2/2] Q8 1024:151936
echo cmd: tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1024:151936 --time 1 > %OUT%\kernel_q8_1024x151936.txt
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1024:151936 --time 1 >> %OUT%\kernel_q8_1024x151936.txt 2>&1
if %errorlevel% NEQ 0 goto :err

echo P02_B_DONE
exit /b 0
:err
echo P02_B_FAILED
exit /b 1
