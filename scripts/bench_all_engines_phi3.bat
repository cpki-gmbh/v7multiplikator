@echo off
REM ============================================================================
REM bench_all_engines_phi3.bat -- benchmark ALL engines on Phi-3-mini-4k-instruct
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
echo  1/5  v7 (Q8)  phi3-mini
echo ============================================================
bin\inference_example.exe bin\v7_threading.dll weights\databases\phi3_mini\lz4ht.db -i "What is the capital of France?" -n 256 -r 3 -q8 -t 0
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  2/5  OpenVINO (int8, AVX2)  phi3-mini
echo ============================================================
call openvino\bench_ov.bat "What is the capital of France?" 3 256 openvino/models/phi3_mini_int8
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  3/5  llama.cpp (Q8_0)  phi3-mini
echo ============================================================
call llamacpp\bench_llamacpp.bat Phi-3-mini-4k-instruct-Q8_0.gguf
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  4/5  ONNX Runtime GenAI (int4)  phi3-mini
echo ============================================================
call onnxrt\bench_ort.bat "What is the capital of France?" 3 400 onnxrt/models/phi3_mini_int4
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  5/5  MNN (int8)  phi3-mini
echo ============================================================
call mnn\bench_mnn.bat mnn\prompt.txt phi3_mini_int8
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  ALL_ENGINES_PHI3_BENCH_DONE
echo ============================================================
exit /b 0

:err
echo ============================================================
echo  ALL_ENGINES_PHI3_BENCH_FAILED
echo ============================================================
exit /b 1
endlocal
