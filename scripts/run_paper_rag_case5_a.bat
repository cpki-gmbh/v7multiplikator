@echo off
rem ============================================================================
rem run_paper_rag_case5_a.bat -- P0-3 case5, condition A: no retrieval
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo case5 condition A (no retrieval)
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "At what operating temperature does the HX-340 pump liquid sodium, and what protects its channel walls?" -n 1500 -q8 > %OUT%\rag_a_case5.txt 2>&1
if %errorlevel% NEQ 0 goto :err
echo CASE5_A_DONE
exit /b 0
:err
echo CASE5_A_FAILED
exit /b 1
