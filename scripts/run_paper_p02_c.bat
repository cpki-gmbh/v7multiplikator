@echo off
rem ============================================================================
rem run_paper_p02_c.bat -- P0-2 kernel loadtests, part C: F32 1024x1024 +
rem 1024x3072 + 151936x1024
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo [1/3] F32 1024:1024
echo cmd: tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --f32 1024:1024 --time 1 > %OUT%\kernel_f32_1024x1024.txt
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --f32 1024:1024 --time 1 >> %OUT%\kernel_f32_1024x1024.txt 2>&1
if %errorlevel% NEQ 0 goto :err

echo [2/3] F32 1024:3072
echo cmd: tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --f32 1024:3072 --time 1 > %OUT%\kernel_f32_1024x3072.txt
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --f32 1024:3072 --time 1 >> %OUT%\kernel_f32_1024x3072.txt 2>&1
if %errorlevel% NEQ 0 goto :err

echo [3/3] F32 151936:1024
echo cmd: tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --f32 151936:1024 --time 1 > %OUT%\kernel_f32_151936x1024.txt
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --f32 151936:1024 --time 1 >> %OUT%\kernel_f32_151936x1024.txt 2>&1
if %errorlevel% NEQ 0 goto :err

echo P02_C_DONE
exit /b 0
:err
echo P02_C_FAILED
exit /b 1
