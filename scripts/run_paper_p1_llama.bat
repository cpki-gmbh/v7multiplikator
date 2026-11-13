@echo off
rem ============================================================================
rem run_paper_p1_llama.bat -- P1-1 llama.cpp comparison (local attic build)
rem
rem Comparator found locally: attic\dev_old_dirs\llama_bench\ with
rem llama-bench.exe, llama-cli.exe and Qwen3-0.6B-Q8_0.gguf (same checkpoint,
rem DIFFERENT quantization format: llama.cpp Q8_0 32-value blocks vs v7 GS=64
rem 64-value blocks -- practical configuration comparison, not format-identical).
rem ============================================================================
cd /d %~dp0..\
rem engine worker priority: above normal via the engine's own override; the
rem llama.cpp tools do not read it (no v7 worker pool), set for uniformity
set v7_override_prio=1
set OUT=build_logs\paper_measurements
set LLAMA=attic\dev_old_dirs\llama_bench

echo [1/3] llama-bench -h
%LLAMA%\llama-bench.exe -h > %OUT%\llama_bench_help.txt 2>&1

echo [2/3] llama-bench default pp/tg
%LLAMA%\llama-bench.exe -m %LLAMA%\Qwen3-0.6B-Q8_0.gguf -t 12 > %OUT%\llama_bench_default.txt 2>&1
if %errorlevel% NEQ 0 goto :err

echo [3/3] llama-cli greedy, same prompt, -n 200
%LLAMA%\llama-cli.exe -m %LLAMA%\Qwen3-0.6B-Q8_0.gguf -no-cnv --temp 0 -n 200 -t 12 -p "Write a clear explanation of how a small local language model can answer questions using a document index." > %OUT%\llama_cli_decode.txt 2>&1
if %errorlevel% NEQ 0 goto :err

echo P1_LLAMA_DONE
exit /b 0

:err
echo P1_LLAMA_FAILED
exit /b 1
