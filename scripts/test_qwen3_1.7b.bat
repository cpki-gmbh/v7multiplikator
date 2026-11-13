@echo off
REM test_qwen3_1.7b.bat -- smoke-test the qwen3_1.7b database: one greedy turn on each quantization path
echo ===== qwen3_1.7b Q8 =====
bin\inference_example.exe v7_threading.dll weights\databases\qwen3_1.7b\lz4ht.db -i "what is a cat?" -t 0 -n 1000 -q8 1
if %errorlevel% NEQ 0 goto fail
echo.
echo ===== qwen3_1.7b F32 =====
bin\inference_example.exe v7_threading.dll weights\databases\qwen3_1.7b\lz4ht.db -i "what is a cat?" -t 0 -n 1000 -q8 0
if %errorlevel% NEQ 0 goto fail
echo.
echo ===== qwen3_1.7b DONE =====
exit /b 0
:fail
echo [FAIL] qwen3_1.7b inference test failed
exit /b 1
