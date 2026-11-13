@echo off
rem run.bat -- Helion HX-340 experiment: run from the project ROOT

bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "what is the maintenance interval of the Helion HX-340 pump and what needs to be replaced?" -n 1500 -q8 -rewrite -script examples\research\helion_pump\helion_rag.logo
