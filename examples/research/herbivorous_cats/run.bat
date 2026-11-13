@echo off
rem run.bat -- herbivorous cats experiment: run from the project ROOT
rem (run it from the repository root -- all paths below are relative to it)

bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "what do cats eat?" -n 1500 -q8 -rewrite -script examples\research\herbivorous_cats\herbivorous_rag.logo
