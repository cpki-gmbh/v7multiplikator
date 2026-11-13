@echo off
rem run.bat -- Kingdom of Ardmark experiment: run from the project ROOT

bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "why was the capital of Ardmark moved and who did it?" -n 1500 -q8 -rewrite -script examples\ardmark\ardmark_rag.logo
