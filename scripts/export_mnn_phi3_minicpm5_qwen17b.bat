@echo off
REM export_mnn_phi3_minicpm5_qwen17b.bat -- export int8 MNN models for the
REM three benchmark models (same recipe as qwen3_06b_int8).
setlocal
set PYTHONIOENCODING=utf-8

cd /d %~dp0..\

set LLMEXPORT=openvino\pyenv\Scripts\python.exe mnn\MNN-3.6.2\transformers\llm\export\llmexport.py
set MNNCONV=mnn\build\Release\MNNConvert.exe

echo Exporting phi3_mini int8 MNN ...
%LLMEXPORT% --path weights\phi3_mini --export mnn --quant_bit 8 --dst_path mnn\models\phi3_mini_int8 --mnnconvert %MNNCONV%
if %errorlevel% NEQ 0 goto :err

echo Exporting minicpm5_1b int8 MNN ...
%LLMEXPORT% --path weights\minicpm5_1b --export mnn --quant_bit 8 --dst_path mnn\models\minicpm5_1b_int8 --mnnconvert %MNNCONV%
if %errorlevel% NEQ 0 goto :err

echo Exporting qwen3_1.7b int8 MNN ...
%LLMEXPORT% --path weights\qwen3_1.7b --export mnn --quant_bit 8 --dst_path mnn\models\qwen3_17b_int8 --mnnconvert %MNNCONV%
if %errorlevel% NEQ 0 goto :err

echo MNNEXPORT_DONE
exit /b 0

:err
echo MNNEXPORT_FAILED
exit /b 1
endlocal
