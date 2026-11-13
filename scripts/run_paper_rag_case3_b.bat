@echo off
rem ============================================================================
rem run_paper_rag_case3_b.bat -- P0-3 case3, condition B: passages in the
rem initial prompt (from bm25_case3.txt)
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo case3 condition B (initial-context)
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "Documentation: The pump accepts a phase control input of 0 to 10 volts for flow regulation, where 10 volts corresponds to the full 340 liters per minute. Electrode erosion shows as a slow decline of flow rate at constant control voltage. At minimum input the pump idles at 12 liters per minute and cannot be driven to zero flow. Question: What control input voltage range does the HX-340 accept for flow regulation, and what flow does it reach at maximum input?" -n 1500 -q8 > %OUT%\rag_b_case3.txt 2>&1
if %errorlevel% NEQ 0 goto :err
echo CASE3_B_DONE
exit /b 0
:err
echo CASE3_B_FAILED
exit /b 1
