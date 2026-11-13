@echo off
REM ============================================================================
REM bench_phi3_retune_run.bat -- the actual phi3-mini benchmark legs
REM
REM Same legs, engine order, rounds and prompts as the phi3-mini section of
REM the overnight benchmark (bench_all_models_overnight_run.bat), so the
REM numbers are directly comparable:
REM   short chat turn (256-token budget) and 500-token poem, 5 rounds each,
REM   engine order v7, OpenVINO, llama.cpp, MNN (8-bit only).
REM
REM Called with full redirection by bench_phi3_retune.bat.
REM A failing leg prints [LEG FAILED] and the run continues.
REM ============================================================================
setlocal
set PYTHONIOENCODING=utf-8
set OV_CPU_MAX_ISA=AVX2

cd /d %~dp0..\

echo ============================================================
echo  PHI3 RETUNE BENCH START %date% %time%
echo  short + poem x 5 rounds x v7/OV/llama/MNN
echo ============================================================


REM ################ phi3-mini, short chat turn ################
echo.
echo ---- %date% %time% v7 phi3-mini short ----
bin\inference_example.exe bin\v7_threading.dll weights\databases\phi3_mini\lz4ht.db -i "What is the capital of France?" -n 256 -r 5 -q8 -t 0
if %errorlevel% NEQ 0 echo [LEG FAILED] v7 phi3-mini short

echo ---- %date% %time% openvino phi3-mini short ----
call openvino\bench_ov.bat "What is the capital of France?" 5 256 openvino/models/phi3_mini_int8
if %errorlevel% NEQ 0 echo [LEG FAILED] openvino phi3-mini short

echo ---- %date% %time% llama.cpp phi3-mini short ----
call llamacpp\bench_llamacpp5.bat Phi-3-mini-4k-instruct-Q8_0.gguf short
if %errorlevel% NEQ 0 echo [LEG FAILED] llama.cpp phi3-mini short

echo ---- %date% %time% mnn phi3-mini short ----
call mnn\bench_mnn5.bat phi3_mini_int8 short
if %errorlevel% NEQ 0 echo [LEG FAILED] mnn phi3-mini short


REM ################ phi3-mini, long generation ################
echo.
echo ---- %date% %time% v7 phi3-mini poem ----
bin\inference_example.exe bin\v7_threading.dll weights\databases\phi3_mini\lz4ht.db -i "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 -r 5 -q8 -t 0
if %errorlevel% NEQ 0 echo [LEG FAILED] v7 phi3-mini poem

echo ---- %date% %time% openvino phi3-mini poem ----
call openvino\bench_ov.bat "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" 5 500 openvino/models/phi3_mini_int8
if %errorlevel% NEQ 0 echo [LEG FAILED] openvino phi3-mini poem

echo ---- %date% %time% llama.cpp phi3-mini poem ----
call llamacpp\bench_llamacpp5.bat Phi-3-mini-4k-instruct-Q8_0.gguf poem
if %errorlevel% NEQ 0 echo [LEG FAILED] llama.cpp phi3-mini poem

echo ---- %date% %time% mnn phi3-mini poem ----
call mnn\bench_mnn5.bat phi3_mini_int8 poem
if %errorlevel% NEQ 0 echo [LEG FAILED] mnn phi3-mini poem


echo.
echo ============================================================
echo  PHI3 RETUNE RUN COMPLETE %date% %time%
echo ============================================================
exit /b 0
endlocal
