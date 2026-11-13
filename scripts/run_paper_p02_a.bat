@echo off
rem ============================================================================
rem run_paper_p02_a.bat -- P0-2 kernel loadtests, part A: Q8 1024x1024 + 1024x3072
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo [1/2] Q8 1024:1024
echo cmd: tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1024:1024 --time 1 > %OUT%\kernel_q8_1024x1024.txt
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1024:1024 --time 1 >> %OUT%\kernel_q8_1024x1024.txt 2>&1
if %errorlevel% NEQ 0 goto :err

echo [2/2] Q8 1024:3072
echo cmd: tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1024:3072 --time 1 > %OUT%\kernel_q8_1024x3072.txt
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 1024:3072 --time 1 >> %OUT%\kernel_q8_1024x3072.txt 2>&1
if %errorlevel% NEQ 0 goto :err

echo P02_A_DONE
exit /b 0
:err
echo P02_A_FAILED
exit /b 1
