@echo off
rem ============================================================================
rem run_paper_p04_treat_1.bat -- P0-4 rewrite-cost measurement, treatment run 1
rem Treatment sidecar performs ONE SetStream appending a fixed ~70-char
rem sentence plus the think-close tag; the only difference against the control
rem is the SetStream rewrite machinery itself.
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo treatment run 1
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "What is the maintenance interval of the Helion HX-340 pump, and which components require replacement?" -n 1500 -q8 -rewrite -script examples/research/helion_pump/p04_append.logo > %OUT%\p04_treat_1.txt 2>&1
if %errorlevel% NEQ 0 goto :err
echo P04_TREAT_1_DONE
exit /b 0
:err
echo P04_TREAT_1_FAILED
exit /b 1
