@echo off
REM download_phi3.bat -- Just download Phi-3-mini-4k-instruct safetensors (no export)
REM
REM Run this in the background. Once complete, run fetch_phi3_mini.bat to export+build.
REM
REM Phi-3-mini: 7.64GB total (2 safetensors shards, fp16)

setlocal ENABLEDELAYEDEXPANSION
set PYTHONIOENCODING=utf-8

set HF_DIR=%~dp0..\weights\phi3_mini
set REPO=microsoft/Phi-3-mini-4k-instruct

if not exist "%HF_DIR%" mkdir "%HF_DIR%"

echo Downloading %REPO% to %HF_DIR% ...
python -c "from huggingface_hub import snapshot_download; snapshot_download('%REPO%', local_dir=r'%HF_DIR%', allow_patterns=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','model.safetensors.index.json','model-*.safetensors'])"
if errorlevel 1 (
    echo [FATAL] Download failed
    exit /b 1
)

echo.
echo Download complete! Files:
dir /b "%HF_DIR%\*.safetensors"
echo.
echo Now run: fetch_phi3_mini.bat
endlocal
exit /b 0
