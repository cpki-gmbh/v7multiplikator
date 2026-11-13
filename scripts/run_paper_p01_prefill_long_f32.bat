@echo off
rem ============================================================================
rem run_paper_p01_prefill_long_f32.bat -- P0-1 prefill, F32, long prompt only
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

set PROMPT_LONGPREFILL=Write a thorough technical overview of how a small local language model inference engine is built. Cover the model file format and how the weights are quantized into eight bit blocks with per group scales, the hash table database that stores the tokenizer vocabulary and merge scores and the template strings, the way the context window is allocated as one large virtual memory block holding the key and value caches of every layer, and the work stealing thread pool that splits every matrix multiplication into chunks which idle worker threads claim atomically. Then describe the self modifying code mechanism that patches prefetch instruction distances per matrix shape, the tuning loop that races two candidate configurations against each other on the same shape and keeps the winner only after a streak of measured wins, and finally the scripting layer that can inspect and rewrite the generated token stream between the reasoning block and the visible answer so retrieval results can be injected mid stream. Explain which of these parts run once at load time and which run on every generated token, and where the memory bandwidth of a laptop processor becomes the limiting factor for each of them.

echo prefill long F32
echo cmd: bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "%PROMPT_LONGPREFILL%" -n 32 -r 2 > %OUT%\prefill_long_f32.txt
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "%PROMPT_LONGPREFILL%" -n 32 -r 2 >> %OUT%\prefill_long_f32.txt 2>&1
if %errorlevel% NEQ 0 goto :err
echo P01_PREFILL_LONG_F32_DONE
exit /b 0

:err
echo P01_PREFILL_LONG_F32_FAILED
exit /b 1
