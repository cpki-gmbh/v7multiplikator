@echo off
REM build_ort_int4_phi3_minicpm5_qwen17b.bat -- build int4 ONNX GenAI models
REM for the three benchmark models (same recipe as qwen3_06b_int4).
setlocal
set PYTHONIOENCODING=utf-8

cd /d %~dp0..\

set ORTBUILD=openvino\pyenv\Scripts\python.exe -m onnxruntime_genai.models.builder

echo Building phi3_mini int4 ...
%ORTBUILD% -m weights\phi3_mini -e cpu -p int4 -o onnxrt\models\phi3_mini_int4
if %errorlevel% NEQ 0 goto :err

echo Building minicpm5_1b int4 ...
%ORTBUILD% -m weights\minicpm5_1b -e cpu -p int4 -o onnxrt\models\minicpm5_1b_int4
if %errorlevel% NEQ 0 goto :err

echo Building qwen3_1.7b int4 ...
%ORTBUILD% -m weights\qwen3_1.7b -e cpu -p int4 -o onnxrt\models\qwen3_17b_int4
if %errorlevel% NEQ 0 goto :err

echo ORTBUILD_DONE
exit /b 0

:err
echo ORTBUILD_FAILED
exit /b 1
endlocal
