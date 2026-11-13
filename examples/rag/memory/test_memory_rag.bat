@echo off
REM test_memory_rag.bat -- run a single turn with memory RAG sidecar
REM First invocation: corpus is empty, no memories retrieved, but a memory
REM is STORED at EOS. Second invocation: corpus has 1 entry, retrieved.
REM Usage: test_memory_rag [prompt] [extra args]
REM   prompt defaults to "what is a cat?"
REM -n 1500 per AGENTS.md SS144: EOS (memory storage) only fires on
REM natural end-of-turn; short budgets truncate and store nothing.

set DLL=bin\v7_threading.dll
set DB=weights\databases\qwen3_0.6b\lz4ht.db
set SIDECAR=examples/rag/memory/sidecar_memory_rag_v6.logo
set PROMPT=%~1
if "%PROMPT%"=="" set PROMPT=what is a cat?

echo === Memory corpus BEFORE run ===
type examples\rag\memory\ragg_base.txt 2>nul || echo (empty)
type examples\rag\memory\user_facts.txt 2>nul || echo (empty)
echo === Running inference with memory RAG sidecar ===
bin\inference_example.exe %DLL% %DB% -i "%PROMPT%" -n 1500 -q8 -rewrite -script %SIDECAR% %2 %3 %4
echo === Memory corpus AFTER run ===
type examples\rag\memory\ragg_base.txt 2>nul || echo (empty)
type examples\rag\memory\user_facts.txt 2>nul || echo (empty)
