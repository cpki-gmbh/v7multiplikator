@echo off
REM export_ov_ir_qwen17b.bat -- export the qwen3_1.7b int8 IR (retry after
REM the shard index was fetched; phi3 + minicpm5 are already done).
setlocal
set PYTHONIOENCODING=utf-8

cd /d %~dp0..\

echo Exporting qwen3_1.7b int8 IR ...
openvino\pyenv\Scripts\optimum-cli.exe export openvino --model weights\qwen3_1.7b --task text-generation-with-past --weight-format int8 openvino\models\qwen3_17b_int8
if %errorlevel% NEQ 0 goto :err

echo OVEXPORT_DONE
exit /b 0

:err
echo OVEXPORT_FAILED
exit /b 1
endlocal
