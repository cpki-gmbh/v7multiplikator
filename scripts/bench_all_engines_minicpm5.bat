@echo off
REM ============================================================================
REM bench_all_engines_minicpm5.bat -- benchmark ALL engines on MiniCPM5-1B
REM
REM Engines and quantization:
REM   1. v7        Q8   via bin\inference_example.exe  (3 rounds, prints phase stats)
REM   2. OpenVINO  int8 via openvino\bench_ov.bat       (3 rounds, AVX2-capped CPU plugin)
REM   3. llama.cpp Q8_0 via llamacpp\bench_llamacpp.bat (3 runs, -t 6)
REM   4. ONNX RT   int4 -- NOT AVAILABLE for this model: ORT GenAI 0.19
REM      cannot tokenize MiniCPM5-1B (pre_tokenizer regex \p{N}{1,3}
REM      raises regex_error error_badbrace in its tokenizer engine)
REM   5. MNN       int8 via mnn\bench_mnn.bat           (3 runs, llm_demo)
REM
REM Same prompt and same token budget for every engine, greedy decoding.
REM Repeatable on this PC: just run it again.
REM ============================================================================
setlocal

cd /d %~dp0..\

echo.
echo ============================================================
echo  1/5  v7 (Q8)  minicpm5-1b
echo ============================================================
bin\inference_example.exe bin\v7_threading.dll weights\databases\minicpm5_1b\lz4ht.db -i "What is the capital of France?" -n 256 -r 3 -q8 -t 0
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  2/5  OpenVINO (int8, AVX2)  minicpm5-1b
echo ============================================================
call openvino\bench_ov.bat "What is the capital of France?" 3 256 openvino/models/minicpm5_1b_int8
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  3/5  llama.cpp (Q8_0)  minicpm5-1b
echo ============================================================
call llamacpp\bench_llamacpp.bat MiniCPM5-1B-Q8_0.gguf
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  4/5  ONNX Runtime GenAI (int4)  minicpm5-1b
echo ============================================================
echo SKIPPED: ORT GenAI 0.19 cannot tokenize MiniCPM5-1B.
echo Its tokenizer engine rejects the model pre_tokenizer regex
echo (\p{N}{1,3}: regex_error error_badbrace), so the int4 model
echo cannot be loaded as shipped. Leg documented as unsupported.

echo.
echo ============================================================
echo  5/5  MNN (int8)  minicpm5-1b
echo ============================================================
call mnn\bench_mnn.bat mnn\prompt.txt minicpm5_1b_int8
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  ALL_ENGINES_MINICPM5_BENCH_DONE
echo ============================================================
exit /b 0

:err
echo ============================================================
echo  ALL_ENGINES_MINICPM5_BENCH_FAILED
echo ============================================================
exit /b 1
endlocal
