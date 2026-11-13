@echo off
rem ============================================================================
rem run_paper_rag_case7_b.bat -- P0-3 case7, condition B: passages in the
rem initial prompt (from bm25_case7.txt)
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo case7 condition B (initial-context)
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "Documentation: Before first startup the loop must be preheated to at least 120 degrees Celsius so the sodium is fully molten. The operator must always preheat the loop to at least 120 degrees Celsius before energizing the HX-340; only then is a start permitted. The HX-340 must never be run dry. Question: What must an operator do before the first startup of the HX-340, and what happens if this step is neglected?" -n 1500 -q8 > %OUT%\rag_b_case7.txt 2>&1
if %errorlevel% NEQ 0 goto :err
echo CASE7_B_DONE
exit /b 0
:err
echo CASE7_B_FAILED
exit /b 1
