@echo off
REM fetch_phi3_minicpm5_models.bat -- fetch competitor models for phi3-mini
REM and minicpm5-1b:
REM   1. official openbmb MiniCPM5-1B-Q8_0.gguf into llamacpp\models
REM   2. llama.cpp source at tag b12179 (official convert_hf_to_gguf.py,
REM      needed because microsoft's official phi3 GGUF repo has no Q8_0)
setlocal

cd /d %~dp0..\

echo Downloading MiniCPM5-1B-Q8_0.gguf ...
openvino\pyenv\Scripts\hf.exe download openbmb/MiniCPM5-1B-GGUF MiniCPM5-1B-Q8_0.gguf --local-dir llamacpp\models
if %errorlevel% NEQ 0 goto :err

echo Cloning llama.cpp source (b12179, shallow) ...
git clone --depth 1 --branch b12179 https://github.com/ggml-org/llama.cpp llamacpp\llamacpp_src
if %errorlevel% NEQ 0 goto :err

echo FETCH_DONE
exit /b 0

:err
echo FETCH_FAILED
exit /b 1
endlocal
