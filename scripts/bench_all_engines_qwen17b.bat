@echo off
REM ============================================================================
REM bench_all_engines_qwen17b.bat -- benchmark ALL engines on Qwen3-1.7B
REM
REM Engines and quantization:
REM   1. v7        Q8   via bin\inference_example.exe  (3 rounds, prints phase stats)
REM   2. OpenVINO  int8 via openvino\bench_ov.bat       (3 rounds, AVX2-capped CPU plugin)
REM   3. llama.cpp Q8_0 via llamacpp\bench_llamacpp.bat (3 runs, -t 6)
REM   4. ONNX RT   int4 via onnxrt\bench_ort.bat        (3 rounds)
REM   5. MNN       int8 via mnn\bench_mnn.bat           (3 runs, llm_demo)
REM
REM Same prompt and same token budget for every engine, greedy decoding.
REM Repeatable on this PC: just run it again.
REM ============================================================================
setlocal

cd /d %~dp0..\

echo.
echo ============================================================
echo  1/5  v7 (Q8)  qwen3-1.7b
echo ============================================================
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_1.7b\lz4ht.db -i "What is the capital of France?" -n 256 -r 3 -q8 -t 0
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  2/5  OpenVINO (int8, AVX2)  qwen3-1.7b
echo ============================================================
call openvino\bench_ov.bat "What is the capital of France?" 3 256 openvino/models/qwen3_17b_int8
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  3/5  llama.cpp (Q8_0)  qwen3-1.7b
echo ============================================================
call llamacpp\bench_llamacpp.bat Qwen3-1.7B-Q8_0.gguf
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  4/5  ONNX Runtime GenAI (int4)  qwen3-1.7b
echo ============================================================
call onnxrt\bench_ort.bat "What is the capital of France?" 3 400 onnxrt/models/qwen3_17b_int4
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  5/5  MNN (int8)  qwen3-1.7b
echo ============================================================
call mnn\bench_mnn.bat mnn\prompt.txt qwen3_17b_int8
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  ALL_ENGINES_QWEN17B_BENCH_DONE
echo ============================================================
exit /b 0

:err
echo ============================================================
echo  ALL_ENGINES_QWEN17B_BENCH_FAILED
echo ============================================================
exit /b 1
endlocal
