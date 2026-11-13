@echo off
rem ============================================================================
rem run_paper_rag_case4_b.bat -- P0-3 case4, condition B: passages in the
rem initial prompt (from bm25_case4.txt)
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo case4 condition B (initial-context)
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "Documentation: The diagnostic port of the HX-340 is an RS-485 interface running the Helion FSD-2 protocol at 115200 baud. The Helion HX-340 is a compact magnetohydrodynamic coolant pump manufactured by Helion Fluid Systems of Turku, Finland. The HX-340 must never be run dry. Question: What diagnostic interface does the Helion HX-340 use, and at what baud rate does it run?" -n 1500 -q8 > %OUT%\rag_b_case4.txt 2>&1
if %errorlevel% NEQ 0 goto :err
echo CASE4_B_DONE
exit /b 0
:err
echo CASE4_B_FAILED
exit /b 1
