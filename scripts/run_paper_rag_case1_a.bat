@echo off
rem ============================================================================
rem run_paper_rag_case1_a.bat -- P0-3 case1, condition A: no retrieval
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo case1 condition A (no retrieval)
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "What is the maintenance interval of the Helion HX-340 pump, and which components require replacement?" -n 1500 -q8 > %OUT%\rag_a_case1.txt 2>&1
if %errorlevel% NEQ 0 goto :err
echo CASE1_A_DONE
exit /b 0
:err
echo CASE1_A_FAILED
exit /b 1
