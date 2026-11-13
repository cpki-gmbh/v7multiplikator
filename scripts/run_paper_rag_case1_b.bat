@echo off
rem ============================================================================
rem run_paper_rag_case1_b.bat -- P0-3 case1, condition B: passages in the
rem initial prompt (from bm25_case1.txt), same flags as condition A
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo case1 condition B (initial-context)
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "Documentation: Maintenance interval of the HX-340 is 26,000 operating hours, after which the electrode cartridges must be replaced. The Helion HX-340 is a compact magnetohydrodynamic coolant pump manufactured by Helion Fluid Systems of Turku, Finland. Replacement takes a two-person crew approximately four hours. Question: What is the maintenance interval of the Helion HX-340 pump, and which components require replacement?" -n 1500 -q8 > %OUT%\rag_b_case1.txt 2>&1
if %errorlevel% NEQ 0 goto :err
echo CASE1_B_DONE
exit /b 0
:err
echo CASE1_B_FAILED
exit /b 1
