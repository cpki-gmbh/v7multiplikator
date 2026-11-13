@echo off
REM export_ov_ir_phi3_minicpm5_qwen17b.bat -- export int8 IRs for the three
REM benchmark models with optimum-cli (NNCF int8_asym per-channel, same recipe
REM as the qwen3_06b_int8 IR).
setlocal
set PYTHONIOENCODING=utf-8

cd /d %~dp0..\

set OVEXPORT=openvino\pyenv\Scripts\optimum-cli.exe

echo Exporting phi3_mini int8 IR ...
%OVEXPORT% export openvino --model weights\phi3_mini --task text-generation-with-past --weight-format int8 openvino\models\phi3_mini_int8
if %errorlevel% NEQ 0 goto :err

echo Exporting minicpm5_1b int8 IR ...
%OVEXPORT% export openvino --model weights\minicpm5_1b --task text-generation-with-past --weight-format int8 openvino\models\minicpm5_1b_int8
if %errorlevel% NEQ 0 goto :err

echo Exporting qwen3_1.7b int8 IR ...
%OVEXPORT% export openvino --model weights\qwen3_1.7b --task text-generation-with-past --weight-format int8 openvino\models\qwen3_17b_int8
if %errorlevel% NEQ 0 goto :err

echo OVEXPORT_DONE
exit /b 0

:err
echo OVEXPORT_FAILED
exit /b 1
endlocal
