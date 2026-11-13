@echo off
REM ============================================================================
REM bench_all_engines_poem_minicpm5.bat -- minicpm5-1b long-generation benchmark
REM
REM Long-generation workload: poem prompt, 500 new tokens per engine.
REM The prompt is hardwired on every command line (cmd 'call' mangles quoted
REM prompts with spaces).
REM
REM   1. v7        Q8   via bin\inference_example.exe  (-n 500, 3 rounds)
REM   2. OpenVINO  int8 via openvino\bench_ov.py       (max_new_tokens 500, 3 rounds)
REM   3. llama.cpp Q8_0 via llama-cli.exe               (-n 500, 3 runs, -t 6)
REM   4. ONNX RT   int4 -- NOT AVAILABLE for this model (ORT GenAI tokenizer
REM      cannot parse the MiniCPM5 pre_tokenizer regex)
REM   5. MNN       int8 via llm_demo.exe                (3 runs, prompt_poem.txt)
REM
REM Repeatable on this PC: just run it again.
REM ============================================================================
setlocal
set PYTHONIOENCODING=utf-8
set OV_CPU_MAX_ISA=AVX2

cd /d %~dp0..\

echo ============================================================
echo  minicpm5-1b benchmark 2 -- long poem prompt, 500 tokens
echo ============================================================

echo.
echo ============================================================
echo  1/5  v7 (Q8)  minicpm5-1b
echo ============================================================
bin\inference_example.exe bin\v7_threading.dll weights\databases\minicpm5_1b\lz4ht.db -i "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 -r 3 -q8 -t 0
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  2/5  OpenVINO (int8, AVX2)  minicpm5-1b
echo ============================================================
"%~dp0..\openvino\pyenv\Scripts\python.exe" openvino\bench_ov.py "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" 3 500 openvino/models/minicpm5_1b_int8
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  3/5  llama.cpp (Q8_0)  minicpm5-1b
echo ============================================================
llamacpp\llama-cli.exe -m llamacpp\models\MiniCPM5-1B-Q8_0.gguf -p "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 --temp 0 -st -c 4096 -ngl 0 -t 6
if %errorlevel% NEQ 0 goto :err
llamacpp\llama-cli.exe -m llamacpp\models\MiniCPM5-1B-Q8_0.gguf -p "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 --temp 0 -st -c 4096 -ngl 0 -t 6
if %errorlevel% NEQ 0 goto :err
llamacpp\llama-cli.exe -m llamacpp\models\MiniCPM5-1B-Q8_0.gguf -p "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 --temp 0 -st -c 4096 -ngl 0 -t 6
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  4/5  ONNX Runtime GenAI (int4)  minicpm5-1b
echo ============================================================
echo SKIPPED: ORT GenAI 0.19 cannot tokenize MiniCPM5-1B
echo (pre_tokenizer regex raises regex_error error_badbrace).

echo.
echo ============================================================
echo  5/5  MNN (int8)  minicpm5-1b
echo ============================================================
mnn\build\Release\llm_demo.exe mnn\models\minicpm5_1b_int8\config.json mnn\prompt_poem.txt
if %errorlevel% NEQ 0 goto :err
mnn\build\Release\llm_demo.exe mnn\models\minicpm5_1b_int8\config.json mnn\prompt_poem.txt
if %errorlevel% NEQ 0 goto :err
mnn\build\Release\llm_demo.exe mnn\models\minicpm5_1b_int8\config.json mnn\prompt_poem.txt
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  ALL_ENGINES_MINICPM5_POEM_DONE
echo ============================================================
exit /b 0

:err
echo ============================================================
echo  ALL_ENGINES_MINICPM5_POEM_FAILED
echo ============================================================
exit /b 1
endlocal
