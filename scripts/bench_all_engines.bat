@echo off
REM ============================================================================
REM bench_all_engines.bat -- benchmark ALL engines on the same workload
REM
REM Engines and quantization:
REM   1. v7        Q8   via bin\inference_example.exe  (3 rounds, prints phase stats)
REM   2. OpenVINO  int8 via openvino\bench_ov.bat       (3 rounds, AVX2-capped CPU plugin)
REM   3. llama.cpp Q8_0 via llamacpp\bench_llamacpp.bat (3 runs, -t 12)
REM   4. ONNX RT   int4 via onnxrt\bench_ort.bat        (3 rounds)
REM   5. MNN       int8 via mnn\bench_mnn.bat           (3 runs, llm_demo)
REM
REM Same prompt and same token budget for every engine, greedy decoding.
REM The shared prompt is hardwired in each sub-bench script (no argument
REM passing: cmd 'call' mangles quoted prompts with spaces).
REM Repeatable on this PC: just run it again.
REM ============================================================================
setlocal

cd /d %~dp0..\

echo.
echo ============================================================
echo  1/5  v7 (Q8)
echo ============================================================
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "What is the capital of France?" -n 256 -r 3 -q8 -t 0
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  2/5  OpenVINO (int8, AVX2)
echo ============================================================
call openvino\bench_ov.bat
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  3/5  llama.cpp (Q8_0)
echo ============================================================
call llamacpp\bench_llamacpp.bat
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  4/5  ONNX Runtime GenAI (int4)
echo ============================================================
call onnxrt\bench_ort.bat
if %errorlevel% NEQ 0 goto :err

echo.
echo ============================================================
echo  5/5  MNN (int8)
echo ============================================================
call mnn\bench_mnn.bat
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
