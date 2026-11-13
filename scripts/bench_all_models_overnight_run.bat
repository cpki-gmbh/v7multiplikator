REM ============================================================================
REM bench_all_models_overnight_run.bat -- the actual benchmark legs
REM
REM 4 models x 2 workloads x 4 engines (8-bit only), 5 rounds per engine:
REM   models:  qwen3-0.6b, qwen3-1.7b, phi-3-mini, minicpm5-1b
REM   engine order per workload: v7, OpenVINO, llama.cpp, MNN (same as the
REM   documented benchmark runs, so the numbers stay comparable)
REM   workloads: short chat turn (256-token budget) and 500-token poem
REM
REM Called with full redirection by bench_all_models_overnight.bat.
REM A failing leg prints [LEG FAILED] and the run continues.
REM ============================================================================
setlocal
set PYTHONIOENCODING=utf-8
set OV_CPU_MAX_ISA=AVX2

cd /d %~dp0..\

echo ============================================================
echo  OVERNIGHT BENCH START %date% %time%
echo  4 models x short + poem x 5 rounds x v7/OV/llama/MNN
echo ============================================================


REM ################ qwen3-0.6b, short chat turn ################
echo.
echo  %date% %time% v7 qwen3-0.6b short 
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "What is the capital of France?" -n 256 -r 5 -q8 -t 0
if %errorlevel% NEQ 0 echo [LEG FAILED] v7 qwen3-0.6b short

echo  %date% %time% openvino qwen3-0.6b short 
call openvino\bench_ov.bat "What is the capital of France?" 5 256 openvino/models/qwen3_06b_int8
if %errorlevel% NEQ 0 echo [LEG FAILED] openvino qwen3-0.6b short

echo  %date% %time% llama.cpp qwen3-0.6b short 
call llamacpp\bench_llamacpp5.bat Qwen3-0.6B-Q8_0.gguf short
if %errorlevel% NEQ 0 echo [LEG FAILED] llama.cpp qwen3-0.6b short

echo  %date% %time% mnn qwen3-0.6b short 
call mnn\bench_mnn5.bat qwen3_06b_int8 short
if %errorlevel% NEQ 0 echo [LEG FAILED] mnn qwen3-0.6b short


REM ################ qwen3-0.6b, long generation ################
echo.
echo  %date% %time% v7 qwen3-0.6b poem 
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 -r 5 -q8 -t 0
if %errorlevel% NEQ 0 echo [LEG FAILED] v7 qwen3-0.6b poem

echo  %date% %time% openvino qwen3-0.6b poem 
call openvino\bench_ov.bat "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" 5 500 openvino/models/qwen3_06b_int8
if %errorlevel% NEQ 0 echo [LEG FAILED] openvino qwen3-0.6b poem

echo  %date% %time% llama.cpp qwen3-0.6b poem 
call llamacpp\bench_llamacpp5.bat Qwen3-0.6B-Q8_0.gguf poem
if %errorlevel% NEQ 0 echo [LEG FAILED] llama.cpp qwen3-0.6b poem

echo  %date% %time% mnn qwen3-0.6b poem 
call mnn\bench_mnn5.bat qwen3_06b_int8 poem
if %errorlevel% NEQ 0 echo [LEG FAILED] mnn qwen3-0.6b poem


REM ################ qwen3-1.7b, short chat turn ################
echo.
echo  %date% %time% v7 qwen3-1.7b short 
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_1.7b\lz4ht.db -i "What is the capital of France?" -n 256 -r 5 -q8 -t 0
if %errorlevel% NEQ 0 echo [LEG FAILED] v7 qwen3-1.7b short

echo  %date% %time% openvino qwen3-1.7b short 
call openvino\bench_ov.bat "What is the capital of France?" 5 256 openvino/models/qwen3_17b_int8
if %errorlevel% NEQ 0 echo [LEG FAILED] openvino qwen3-1.7b short

echo  %date% %time% llama.cpp qwen3-1.7b short 
call llamacpp\bench_llamacpp5.bat Qwen3-1.7B-Q8_0.gguf short
if %errorlevel% NEQ 0 echo [LEG FAILED] llama.cpp qwen3-1.7b short

echo  %date% %time% mnn qwen3-1.7b short 
call mnn\bench_mnn5.bat qwen3_17b_int8 short
if %errorlevel% NEQ 0 echo [LEG FAILED] mnn qwen3-1.7b short


REM ################ qwen3-1.7b, long generation ################
echo.
echo  %date% %time% v7 qwen3-1.7b poem 
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_1.7b\lz4ht.db -i "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 -r 5 -q8 -t 0
if %errorlevel% NEQ 0 echo [LEG FAILED] v7 qwen3-1.7b poem

echo  %date% %time% openvino qwen3-1.7b poem 
call openvino\bench_ov.bat "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" 5 500 openvino/models/qwen3_17b_int8
if %errorlevel% NEQ 0 echo [LEG FAILED] openvino qwen3-1.7b poem

echo  %date% %time% llama.cpp qwen3-1.7b poem 
call llamacpp\bench_llamacpp5.bat Qwen3-1.7B-Q8_0.gguf poem
if %errorlevel% NEQ 0 echo [LEG FAILED] llama.cpp qwen3-1.7b poem

echo  %date% %time% mnn qwen3-1.7b poem 
call mnn\bench_mnn5.bat qwen3_17b_int8 poem
if %errorlevel% NEQ 0 echo [LEG FAILED] mnn qwen3-1.7b poem


REM ################ phi-3-mini, short chat turn ################
echo.
echo  %date% %time% v7 phi3-mini short 
bin\inference_example.exe bin\v7_threading.dll weights\databases\phi3_mini\lz4ht.db -i "What is the capital of France?" -n 256 -r 5 -q8 -t 0
if %errorlevel% NEQ 0 echo [LEG FAILED] v7 phi3-mini short

echo  %date% %time% openvino phi3-mini short 
call openvino\bench_ov.bat "What is the capital of France?" 5 256 openvino/models/phi3_mini_int8
if %errorlevel% NEQ 0 echo [LEG FAILED] openvino phi3-mini short

echo  %date% %time% llama.cpp phi3-mini short 
call llamacpp\bench_llamacpp5.bat Phi-3-mini-4k-instruct-Q8_0.gguf short
if %errorlevel% NEQ 0 echo [LEG FAILED] llama.cpp phi3-mini short

echo  %date% %time% mnn phi3-mini short 
call mnn\bench_mnn5.bat phi3_mini_int8 short
if %errorlevel% NEQ 0 echo [LEG FAILED] mnn phi3-mini short


REM ################ phi-3-mini, long generation ################
echo.
echo  %date% %time% v7 phi3-mini poem 
bin\inference_example.exe bin\v7_threading.dll weights\databases\phi3_mini\lz4ht.db -i "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 -r 5 -q8 -t 0
if %errorlevel% NEQ 0 echo [LEG FAILED] v7 phi3-mini poem

echo  %date% %time% openvino phi3-mini poem 
call openvino\bench_ov.bat "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" 5 500 openvino/models/phi3_mini_int8
if %errorlevel% NEQ 0 echo [LEG FAILED] openvino phi3-mini poem

echo  %date% %time% llama.cpp phi3-mini poem 
call llamacpp\bench_llamacpp5.bat Phi-3-mini-4k-instruct-Q8_0.gguf poem
if %errorlevel% NEQ 0 echo [LEG FAILED] llama.cpp phi3-mini poem

echo  %date% %time% mnn phi3-mini poem 
call mnn\bench_mnn5.bat phi3_mini_int8 poem
if %errorlevel% NEQ 0 echo [LEG FAILED] mnn phi3-mini poem


REM ################ minicpm5-1b, short chat turn ################
echo.
echo  %date% %time% v7 minicpm5-1b short 
bin\inference_example.exe bin\v7_threading.dll weights\databases\minicpm5_1b\lz4ht.db -i "What is the capital of France?" -n 256 -r 5 -q8 -t 0
if %errorlevel% NEQ 0 echo [LEG FAILED] v7 minicpm5-1b short

echo  %date% %time% openvino minicpm5-1b short 
call openvino\bench_ov.bat "What is the capital of France?" 5 256 openvino/models/minicpm5_1b_int8
if %errorlevel% NEQ 0 echo [LEG FAILED] openvino minicpm5-1b short

echo  %date% %time% llama.cpp minicpm5-1b short 
call llamacpp\bench_llamacpp5.bat MiniCPM5-1B-Q8_0.gguf short
if %errorlevel% NEQ 0 echo [LEG FAILED] llama.cpp minicpm5-1b short

echo  %date% %time% mnn minicpm5-1b short 
call mnn\bench_mnn5.bat minicpm5_1b_int8 short
if %errorlevel% NEQ 0 echo [LEG FAILED] mnn minicpm5-1b short


REM ################ minicpm5-1b, long generation ################
echo.
echo  %date% %time% v7 minicpm5-1b poem 
bin\inference_example.exe bin\v7_threading.dll weights\databases\minicpm5_1b\lz4ht.db -i "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 -r 5 -q8 -t 0
if %errorlevel% NEQ 0 echo [LEG FAILED] v7 minicpm5-1b poem

echo  %date% %time% openvino minicpm5-1b poem 
call openvino\bench_ov.bat "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" 5 500 openvino/models/minicpm5_1b_int8
if %errorlevel% NEQ 0 echo [LEG FAILED] openvino minicpm5-1b poem

echo  %date% %time% llama.cpp minicpm5-1b poem 
call llamacpp\bench_llamacpp5.bat MiniCPM5-1B-Q8_0.gguf poem
if %errorlevel% NEQ 0 echo [LEG FAILED] llama.cpp minicpm5-1b poem

echo  %date% %time% mnn minicpm5-1b poem 
call mnn\bench_mnn5.bat minicpm5_1b_int8 poem
if %errorlevel% NEQ 0 echo [LEG FAILED] mnn minicpm5-1b poem


echo.
echo ============================================================
echo  OVERNIGHT RUN COMPLETE %date% %time%
echo ============================================================
exit /b 0
endlocal
