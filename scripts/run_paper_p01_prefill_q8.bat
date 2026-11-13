@echo off
rem ============================================================================
rem run_paper_p01_prefill_q8.bat -- P0-1 prefill, Q8, three input sizes
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal (Win32 thread priority 1) via the
rem engine's own override, read at pool creation; every log prints the line
set v7_override_prio=1
set OUT=build_logs\paper_measurements
if not exist %OUT% mkdir %OUT%

echo [1/3] prefill short Q8
echo cmd: bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "What is a cat?" -n 32 -r 2 -q8 > %OUT%\prefill_short_q8.txt
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "What is a cat?" -n 32 -r 2 -q8 >> %OUT%\prefill_short_q8.txt 2>&1
if %errorlevel% NEQ 0 goto :err

set PROMPT_MED=Explain in detail how retrieval augmented generation works. The system first splits its documents into short passages, then builds an inverted index over the words of every passage. When the user asks a question, the question words are scored against that index with a bm25 weighting, the best matching passages are returned, and their text is injected into the language model context before the answer is generated, so the model can quote facts it was never trained on.

echo [2/3] prefill medium Q8
echo cmd: bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "%PROMPT_MED%" -n 32 -r 2 -q8 > %OUT%\prefill_med_q8.txt
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "%PROMPT_MED%" -n 32 -r 2 -q8 >> %OUT%\prefill_med_q8.txt 2>&1
if %errorlevel% NEQ 0 goto :err

set PROMPT_LONGPREFILL=Write a thorough technical overview of how a small local language model inference engine is built. Cover the model file format and how the weights are quantized into eight bit blocks with per group scales, the hash table database that stores the tokenizer vocabulary and merge scores and the template strings, the way the context window is allocated as one large virtual memory block holding the key and value caches of every layer, and the work stealing thread pool that splits every matrix multiplication into chunks which idle worker threads claim atomically. Then describe the self modifying code mechanism that patches prefetch instruction distances per matrix shape, the tuning loop that races two candidate configurations against each other on the same shape and keeps the winner only after a streak of measured wins, and finally the scripting layer that can inspect and rewrite the generated token stream between the reasoning block and the visible answer so retrieval results can be injected mid stream. Explain which of these parts run once at load time and which run on every generated token, and where the memory bandwidth of a laptop processor becomes the limiting factor for each of them.

echo [3/3] prefill long Q8
echo cmd: bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "%PROMPT_LONGPREFILL%" -n 32 -r 2 -q8 > %OUT%\prefill_long_q8.txt
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "%PROMPT_LONGPREFILL%" -n 32 -r 2 -q8 >> %OUT%\prefill_long_q8.txt 2>&1
if %errorlevel% NEQ 0 goto :err

echo P01_PREFILL_Q8_DONE
exit /b 0

:err
echo P01_PREFILL_Q8_FAILED
exit /b 1
