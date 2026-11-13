@echo off
rem ============================================================================
rem run_paper_rag_case8_c.bat -- P0-3 case8 off-topic control, condition C:
rem the think-text BM25 query is expected to return empty, so the sidecar
rem suppresses the rewrite; the run must show no SETTING STREAM banner
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo case8 condition C (mid-stream rewrite, expected suppressed)
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "What is the capital of France?" -n 1500 -q8 -rewrite -script examples/research/helion_pump/helion_rag.logo > %OUT%\rag_c_case8.txt 2>&1
if %errorlevel% NEQ 0 goto :err
echo CASE8_C_DONE
exit /b 0
:err
echo CASE8_C_FAILED
exit /b 1
