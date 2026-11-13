@echo off
rem run_bm25.bat -- herbivorous-cats experiment, PURE SectorLogo BM25 sidecar:
rem the retrieval + stream rewrite runs entirely in Logo (no external exe).
rem Qwen3-0.6B, canonical RAG prompt, -n 1500 per the research-log minimum.
rem Self-locating: works from any CWD (jumps to the project root).

cd /d "%~dp0..\..\.."

bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "what do cats eat?" -n 1500 -q8 -rewrite -script examples\research\herbivorous_cats\herbivorous_bm25.logo
