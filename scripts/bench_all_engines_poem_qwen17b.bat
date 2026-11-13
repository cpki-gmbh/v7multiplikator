@echo off
REM ============================================================================
REM bench_all_engines_poem_qwen17b.bat -- qwen3-1.7b long-generation benchmark
REM
REM Long-generation workload: poem prompt, 500 new tokens per engine.
REM The prompt is hardwired on every command line (cmd 'call' mangles quoted
REM prompts with spaces).
REM
REM   1. v7        Q8   via bin\inference_example.exe  (-n 500, 3 rounds)
REM   2. OpenVINO  int8 via openvino\bench_ov.py       (max_new_tokens 500, 3 rounds)
REM   3. llama.cpp Q8_0 via llama-cli.exe               (-n 500, 3 runs, -t 6)
REM   4. ONNX RT   int4 via onnxrt\bench_ort.py         (max_length 600 = prompt + 500 + headroom)
REM   5. MNN       int8 via llm_demo.exe                (3 runs, prompt_poem.txt)
REM
REM Repeatable on this PC: just run it again.
REM ============================================================================
setlocal
set PYTHONIOENCODING=utf-8
set OV_CPU_MAX_ISA=AVX2

cd /d %~dp0..\

echo ============================================================
echo  qwen3-1.7b benchmark 2 -- long poem prompt, 500 tokens
echo ============================================================

echo.
echo ============================================================
echo  1/5  v7 (Q8)  qwen3-1.7b
echo ============================================================
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_1.7b\lz4ht.db -i "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 -r 3 -q8 -t 0
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  2/5  OpenVINO (int8, AVX2)  qwen3-1.7b
echo ============================================================
"%~dp0..\openvino\pyenv\Scripts\python.exe" openvino\bench_ov.py "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" 3 500 openvino/models/qwen3_17b_int8
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  3/5  llama.cpp (Q8_0)  qwen3-1.7b
echo ============================================================
llamacpp\llama-cli.exe -m llamacpp\models\Qwen3-1.7B-Q8_0.gguf -p "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 --temp 0 -st -c 4096 -ngl 0 -t 6
if %errorlevel% NEQ 0 goto :err
llamacpp\llama-cli.exe -m llamacpp\models\Qwen3-1.7B-Q8_0.gguf -p "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 --temp 0 -st -c 4096 -ngl 0 -t 6
if %errorlevel% NEQ 0 goto :err
llamacpp\llama-cli.exe -m llamacpp\models\Qwen3-1.7B-Q8_0.gguf -p "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 --temp 0 -st -c 4096 -ngl 0 -t 6
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  4/5  ONNX Runtime GenAI (int4)  qwen3-1.7b
echo ============================================================
"%~dp0..\openvino\pyenv\Scripts\python.exe" onnxrt\bench_ort.py "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" 3 600 onnxrt/models/qwen3_17b_int4
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  5/5  MNN (int8)  qwen3-1.7b
echo ============================================================
mnn\build\Release\llm_demo.exe mnn\models\qwen3_17b_int8\config.json mnn\prompt_poem.txt
if %errorlevel% NEQ 0 goto :err
mnn\build\Release\llm_demo.exe mnn\models\qwen3_17b_int8\config.json mnn\prompt_poem.txt
if %errorlevel% NEQ 0 goto :err
mnn\build\Release\llm_demo.exe mnn\models\qwen3_17b_int8\config.json mnn\prompt_poem.txt
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  ALL_ENGINES_QWEN17B_POEM_DONE
echo ============================================================
exit /b 0

:err
echo ============================================================
echo  ALL_ENGINES_QWEN17B_POEM_FAILED
echo ============================================================
exit /b 1
endlocal
