@echo off
rem ============================================================================
rem run_paper_rag_case6_b.bat -- P0-3 case6, condition B: passages in the
rem initial prompt (from bm25_case6.txt)
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo case6 condition B (initial-context)
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "Documentation: Helion Fluid Systems offers the HX-340 in two variants: the standard HX-340 with molybdenum channel lining, and the HX-340S with a silicon-carbide lining rated for 550 degrees Celsius service at a reduced flow of 290 liters per minute. The HX-340 must never be run dry. The HX-340 cannot be started on a cold loop. Question: How does the HX-340S variant differ from the standard HX-340?" -n 1500 -q8 > %OUT%\rag_b_case6.txt 2>&1
if %errorlevel% NEQ 0 goto :err
echo CASE6_B_DONE
exit /b 0
:err
echo CASE6_B_FAILED
exit /b 1
