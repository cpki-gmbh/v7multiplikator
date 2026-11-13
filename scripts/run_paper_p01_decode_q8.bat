@echo off
rem ============================================================================
rem run_paper_p01_decode_q8.bat -- P0-1 decode, Q8, single process, 4 rounds
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

set PROMPT_LONG=Write a clear explanation of how a small local language model can answer questions using a document index.

echo decode Q8, 4 rounds
echo cmd: bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "%PROMPT_LONG%" -n 200 -r 4 -q8 > %OUT%\decode_q8.txt
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "%PROMPT_LONG%" -n 200 -r 4 -q8 >> %OUT%\decode_q8.txt 2>&1
if %errorlevel% NEQ 0 goto :err
echo P01_DECODE_Q8_DONE
exit /b 0

:err
echo P01_DECODE_Q8_FAILED
exit /b 1
