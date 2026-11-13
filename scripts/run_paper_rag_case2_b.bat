@echo off
rem ============================================================================
rem run_paper_rag_case2_b.bat -- P0-3 case2, condition B: passages in the
rem initial prompt (from bm25_case2.txt)
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo case2 condition B (initial-context)
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "Documentation: Its nominal flow rate is 340 liters per minute at a pressure head of 2.4 bar, and the whole unit weighs 19 kilograms. The port reports flow rate, channel pressure, electrode wear percentage, and winding temperature. The Helion HX-340 is a compact magnetohydrodynamic coolant pump manufactured by Helion Fluid Systems of Turku, Finland. Question: What is the nominal flow rate of the Helion HX-340, and at what pressure head does it deliver it?" -n 1500 -q8 > %OUT%\rag_b_case2.txt 2>&1
if %errorlevel% NEQ 0 goto :err
echo CASE2_B_DONE
exit /b 0
:err
echo CASE2_B_FAILED
exit /b 1
