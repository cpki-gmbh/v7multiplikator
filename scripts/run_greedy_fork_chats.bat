@echo off
rem run_greedy_fork_chats.bat -- exercise the adaptive greedy-fork sampler with fresh chat sessions.
rem Three greedy (t=0) Qwen3-0.6B Q8 turns; teardown persists the adapted split percentages.

bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "what is a cat?" -t 0 -n 400 -q8 1
if %errorlevel% NEQ 0 exit /b 1

bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "explain in one paragraph what a compiler does" -t 0 -n 400 -q8 1
if %errorlevel% NEQ 0 exit /b 1

bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "name three planets and one fact about each" -t 0 -n 400 -q8 1
if %errorlevel% NEQ 0 exit /b 1
