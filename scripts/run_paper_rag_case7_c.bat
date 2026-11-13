@echo off
rem ============================================================================
rem run_paper_rag_case7_c.bat -- P0-3 case7, condition C: mid-stream rewrite
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo case7 condition C (mid-stream rewrite)
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "What must an operator do before the first startup of the HX-340, and what happens if this step is neglected?" -n 1500 -q8 -rewrite -script examples/research/helion_pump/helion_rag.logo > %OUT%\rag_c_case7.txt 2>&1
if %errorlevel% NEQ 0 goto :err
echo CASE7_C_DONE
exit /b 0
:err
echo CASE7_C_FAILED
exit /b 1
