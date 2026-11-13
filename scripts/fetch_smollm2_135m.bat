@echo off
REM fetch_smollm2_135m.bat -- Download SmolLM2-135M-Instruct from HuggingFace Hub and export to v7 .bin format
REM
REM Uses HuggingFaceTB/SmolLM2-135M-Instruct (ungated, open model).
REM 135M params, 4 KV heads, 24 layers, dim=576, vocab=49152.
REM ChatML-style tokenizer with <|im_start|>/<|im_end|> tokens.
REM
REM Memory: ~270MB safetensors + ~400MB peak during export. Fits in 4GB RAM.
REM
REM Usage:  fetch_smollm2_135m.bat
REM
REM Output:
REM   weights\smollm2_135m\                    -- raw HF files
REM   weights\smollm2_135m\smollm2_135m.bin    -- v7 checkpoint
REM   weights\databases\smollm2_135m\lz4ht.db   -- v7 microdb database

setlocal ENABLEDELAYEDEXPANSION
set PYTHONIOENCODING=utf-8

set THIS_DIR=%~dp0..\
set TOOLS=%THIS_DIR%tools
set BIN=%THIS_DIR%bin
set W=%THIS_DIR%weights
set HF_DIR=%W%\smollm2_135m
set BIN_OUT=%HF_DIR%\smollm2_135m.bin
set DB_DIR=%W%\databases\smollm2_135m
set DB_OUT=%DB_DIR%\lz4ht.db
set REPO=HuggingFaceTB/SmolLM2-135M-Instruct

echo.
echo ============================================================================
echo  Fetch SmolLM2-135M-Instruct from HuggingFace Hub and build v7 database
echo ============================================================================
echo.

REM ====== Step 1: Clean and download from HuggingFace Hub ======
echo [1/3] Downloading %REPO% ...
if exist "%HF_DIR%" rmdir /s /q "%HF_DIR%"
mkdir "%HF_DIR%"
python -c "from huggingface_hub import snapshot_download; snapshot_download('%REPO%', local_dir=r'%HF_DIR%', ignore_patterns=['*.bin','*.pt','*.onnx','*.h5','*.msgpack'])"
if !errorlevel! neq 0 (
    echo [FATAL] HuggingFace download FAILED
    exit /b 1
)
if not exist "%HF_DIR%\model.safetensors" (
    echo [FATAL] model.safetensors not found after download
    exit /b 1
)
echo  OK: Downloaded to %HF_DIR%
echo.

REM ====== Step 2: Export to v7 .bin format ======
echo [2/3] Exporting to v7 checkpoint ...
if exist "%BIN_OUT%" del /f "%BIN_OUT%"
python "%TOOLS%\export.py" "%BIN_OUT%" "%HF_DIR%"
if !errorlevel! neq 0 (
    echo [FATAL] export.py FAILED
    exit /b 1
)
if not exist "%BIN_OUT%" (
    echo [FATAL] %BIN_OUT% not created
    exit /b 1
)
echo  OK: Exported to %BIN_OUT%
echo.

REM ====== Step 3: Build v7 database ======
echo [3/3] Building v7 database ...
if not exist "%DB_DIR%" mkdir "%DB_DIR%"
if exist "%DB_OUT%" del /f "%DB_OUT%"
"%BIN%\build_db.exe" "%BIN_OUT%" -thinking 1 -rope-base 100000 -o "%DB_OUT%" -f
if !errorlevel! neq 0 (
    echo [FATAL] build_db.exe FAILED
    exit /b 1
)
if not exist "%DB_OUT%" (
    echo [FATAL] %DB_OUT% not created
    exit /b 1
)
echo  OK: Database built at %DB_OUT%
echo.

echo ============================================================================
echo  Fetch SmolLM2-135M-Instruct COMPLETE
echo  Checkpoint : %BIN_OUT%
echo  Database   : %DB_OUT%
echo  HF source  : %HF_DIR%
echo ============================================================================
endlocal
exit /b 0
