@echo off
REM ============================================================================
REM bench_all_engines_poem.bat -- engine benchmark, experiment 2
REM
REM Long-generation workload: poem prompt, 500 new tokens per engine.
REM The prompt is hardwired on every command line (cmd 'call' mangles quoted
REM prompts with spaces, and %PROMPT% would collide with cmd's built-in
REM PROMPT variable).
REM
REM   1. v7        Q8   via bin\inference_example.exe  (-n 500, 3 rounds)
REM   2. OpenVINO  int8 via openvino\bench_ov.py       (max_new_tokens 500, 3 rounds)
REM   3. llama.cpp Q8_0 via llama-cli.exe               (-n 500, 3 runs, -t 12)
REM   4. ONNX RT   int4 via onnxrt\bench_ort.py         (max_length 600 = prompt + 500 + headroom)
REM
REM Repeatable on this PC: just run it again.
REM ============================================================================
setlocal
set PYTHONIOENCODING=utf-8
set OV_CPU_MAX_ISA=AVX2

cd /d %~dp0..\

echo ============================================================
echo  Engine benchmark 2 -- long poem prompt, 500 tokens
echo ============================================================

echo.
echo ============================================================
echo  1/5  v7 (Q8)
echo ============================================================
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 -r 3 -q8 -t 0
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  2/5  OpenVINO (int8, AVX2)
echo ============================================================
"%~dp0..\openvino\pyenv\Scripts\python.exe" openvino\bench_ov.py "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" 3 500
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  3/5  llama.cpp (Q8_0)
echo ============================================================
llamacpp\llama-cli.exe -m llamacpp\models\Qwen3-0.6B-Q8_0.gguf -p "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 --temp 0 -st -c 4096 -ngl 0 -t 12
if %errorlevel% NEQ 0 goto :err
llamacpp\llama-cli.exe -m llamacpp\models\Qwen3-0.6B-Q8_0.gguf -p "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 --temp 0 -st -c 4096 -ngl 0 -t 12
if %errorlevel% NEQ 0 goto :err
llamacpp\llama-cli.exe -m llamacpp\models\Qwen3-0.6B-Q8_0.gguf -p "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 --temp 0 -st -c 4096 -ngl 0 -t 12
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  4/5  ONNX Runtime GenAI (int4)
echo ============================================================
"%~dp0..\openvino\pyenv\Scripts\python.exe" onnxrt\bench_ort.py "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" 3 600
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  5/5  MNN (int8)
echo ============================================================
mnn\build\Release\llm_demo.exe mnn\models\qwen3_06b_int8\config.json mnn\prompt_poem.txt
if %errorlevel% NEQ 0 goto :err
mnn\build\Release\llm_demo.exe mnn\models\qwen3_06b_int8\config.json mnn\prompt_poem.txt
if %errorlevel% NEQ 0 goto :err
mnn\build\Release\llm_demo.exe mnn\models\qwen3_06b_int8\config.json mnn\prompt_poem.txt
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  ALL_ENGINES_BENCH_DONE
echo ============================================================
exit /b 0

:err
echo ============================================================
echo  ALL_ENGINES_BENCH_FAILED
echo ============================================================
exit /b 1
endlocal
