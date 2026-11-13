@echo off
rem ============================================================================
rem run_paper_rag_case5_b.bat -- P0-3 case5, condition B: passages in the
rem initial prompt (from bm25_case5.txt)
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo case5 condition B (initial-context)
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "Documentation: The channel walls are molybdenum-lined to resist corrosion by hot liquid sodium at the operating temperature of 480 degrees Celsius. Common faults of the HX-340 are electrode erosion, channel wall slugging, and Hall-current sensor drift. The HX-340 moves liquid sodium through breeder reactor secondary loops. Question: At what operating temperature does the HX-340 pump liquid sodium, and what protects its channel walls?" -n 1500 -q8 > %OUT%\rag_b_case5.txt 2>&1
if %errorlevel% NEQ 0 goto :err
echo CASE5_B_DONE
exit /b 0
:err
echo CASE5_B_FAILED
exit /b 1
