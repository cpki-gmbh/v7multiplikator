@echo off
rem ============================================================================
rem run_paper_rag_case8_a.bat -- P0-3 case8 off-topic control, condition A
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo case8 condition A (no retrieval)
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "What is the capital of France?" -n 1500 -q8 > %OUT%\rag_a_case8.txt 2>&1
if %errorlevel% NEQ 0 goto :err
echo CASE8_A_DONE
exit /b 0
:err
echo CASE8_A_FAILED
exit /b 1
