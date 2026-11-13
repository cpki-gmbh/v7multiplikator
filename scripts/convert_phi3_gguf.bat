@echo off
REM convert_phi3_gguf.bat -- convert the local Phi-3-mini-4k-instruct HF dir
REM to Q8_0 GGUF with llama.cpp's official converter (b12179).
REM microsoft's official GGUF repo only ships Q4_K_M / FP16, so the Q8_0
REM conversion is done with the llama.cpp tool itself.
setlocal

cd /d %~dp0..\

if not exist llamacpp\llamacpp_src\convert_hf_to_gguf.py goto :errsrc

openvino\pyenv\Scripts\python.exe -c "import gguf" >nul 2>&1
if %errorlevel% EQU 0 goto :havegguf
echo Installing gguf pip package ...
openvino\pyenv\Scripts\python.exe -m pip install gguf
if %errorlevel% NEQ 0 goto :err

:havegguf
echo Converting weights\phi3_mini to Q8_0 GGUF ...
openvino\pyenv\Scripts\python.exe llamacpp\llamacpp_src\convert_hf_to_gguf.py weights\phi3_mini --outtype q8_0 --outfile llamacpp\models\Phi-3-mini-4k-instruct-Q8_0.gguf
if %errorlevel% NEQ 0 goto :err

echo CONVERT_DONE
exit /b 0

:errsrc
echo llama.cpp source not found -- run fetch_phi3_minicpm5_models.bat first
exit /b 1

:err
echo CONVERT_FAILED
exit /b 1
endlocal
