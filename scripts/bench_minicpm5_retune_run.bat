@echo off
REM ============================================================================
REM bench_minicpm5_retune_run.bat -- the actual minicpm5-1b benchmark legs
REM
REM Same legs, engine order, rounds and prompts as the minicpm5-1b section of
REM the overnight benchmark (bench_all_models_overnight_run.bat), so the
REM numbers are directly comparable:
REM   short chat turn (256-token budget) and 500-token poem, 5 rounds each,
REM   engine order v7, OpenVINO, llama.cpp, MNN (8-bit only).
REM
REM Called with full redirection by bench_minicpm5_retune.bat.
REM A failing leg prints [LEG FAILED] and the run continues.
REM ============================================================================
setlocal
set PYTHONIOENCODING=utf-8
set OV_CPU_MAX_ISA=AVX2

cd /d %~dp0..\

echo ============================================================
echo  MINICPM5 RETUNE BENCH START %date% %time%
echo  short + poem x 5 rounds x v7/OV/llama/MNN
echo ============================================================


REM ################ minicpm5-1b, short chat turn ################
echo.
echo ---- %date% %time% v7 minicpm5-1b short ----
bin\inference_example.exe bin\v7_threading.dll weights\databases\minicpm5_1b\lz4ht.db -i "What is the capital of France?" -n 256 -r 5 -q8 -t 0
if %errorlevel% NEQ 0 echo [LEG FAILED] v7 minicpm5-1b short

echo ---- %date% %time% openvino minicpm5-1b short ----
call openvino\bench_ov.bat "What is the capital of France?" 5 256 openvino/models/minicpm5_1b_int8
if %errorlevel% NEQ 0 echo [LEG FAILED] openvino minicpm5-1b short

echo ---- %date% %time% llama.cpp minicpm5-1b short ----
call llamacpp\bench_llamacpp5.bat MiniCPM5-1B-Q8_0.gguf short
if %errorlevel% NEQ 0 echo [LEG FAILED] llama.cpp minicpm5-1b short

echo ---- %date% %time% mnn minicpm5-1b short ----
call mnn\bench_mnn5.bat minicpm5_1b_int8 short
if %errorlevel% NEQ 0 echo [LEG FAILED] mnn minicpm5-1b short


REM ################ minicpm5-1b, long generation ################
echo.
echo ---- %date% %time% v7 minicpm5-1b poem ----
bin\inference_example.exe bin\v7_threading.dll weights\databases\minicpm5_1b\lz4ht.db -i "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" -n 500 -r 5 -q8 -t 0
if %errorlevel% NEQ 0 echo [LEG FAILED] v7 minicpm5-1b poem

echo ---- %date% %time% openvino minicpm5-1b poem ----
call openvino\bench_ov.bat "write a long poem about horses, jesus and cats, remeber to talk about 1+2=3" 5 500 openvino/models/minicpm5_1b_int8
if %errorlevel% NEQ 0 echo [LEG FAILED] openvino minicpm5-1b poem

echo ---- %date% %time% llama.cpp minicpm5-1b poem ----
call llamacpp\bench_llamacpp5.bat MiniCPM5-1B-Q8_0.gguf poem
if %errorlevel% NEQ 0 echo [LEG FAILED] llama.cpp minicpm5-1b poem

echo ---- %date% %time% mnn minicpm5-1b poem ----
call mnn\bench_mnn5.bat minicpm5_1b_int8 poem
if %errorlevel% NEQ 0 echo [LEG FAILED] mnn minicpm5-1b poem


echo.
echo ============================================================
echo  MINICPM5 RETUNE RUN COMPLETE %date% %time%
echo ============================================================
exit /b 0
endlocal
