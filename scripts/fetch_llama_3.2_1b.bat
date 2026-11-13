@echo off
REM fetch_llama_3.2_1b.bat -- Download Llama-3.2-1B-Instruct from HuggingFace Hub and export to v7 .bin format
REM
REM Uses unsloth/Llama-3.2-1B-Instruct (ungated mirror of meta-llama/Llama-3.2-1B-Instruct).
REM This is the instruction-tuned variant of the smallest official Meta Llama architecture:
REM 1B params, GQA (32 heads, 8 KV), 16 layers, dim=2048, vocab=128256.
REM BPE tokenizer (tiktoken-style) with Llama3 chat template.
REM
REM Memory: ~2.5GB safetensors + ~3GB peak during export. Fits in 12GB RAM.
REM
REM Usage:  fetch_llama_3.2_1b.bat
REM
REM Output:
REM   weights\llama_3.2_1b\                  -- raw HF files (cached)
REM   weights\llama_3.2_1b\llama_3.2_1b.bin  -- v7 checkpoint
REM   weights\databases\llama_3.2_1b\lz4ht.db -- v7 microdb database

setlocal ENABLEDELAYEDEXPANSION
set PYTHONIOENCODING=utf-8

set THIS_DIR=%~dp0..\
set TOOLS=%THIS_DIR%tools
set BIN=%THIS_DIR%bin
set HF_DIR=%THIS_DIR%weights\llama_3.2_1b
set BIN_OUT=%HF_DIR%\llama_3.2_1b.bin
set DB_DIR=%THIS_DIR%weights\databases\llama_3.2_1b
set DB_OUT=%DB_DIR%\lz4ht.db
set REPO=unsloth/Llama-3.2-1B-Instruct

echo.
echo ============================================================================
echo  Fetch Llama-3.2-1B-Instruct from HuggingFace Hub and build v7 database
echo  (unsloth ungated mirror -- official Meta Llama Instruct, 1B params)
echo ============================================================================
echo.

REM ====== Step 1: Download from HuggingFace Hub ======
echo [1/3] Downloading %REPO% ...
if exist "%HF_DIR%\model.safetensors" (
    echo  [SKIP] Already downloaded: %HF_DIR%\model.safetensors
) else (
    if not exist "%HF_DIR%" mkdir "%HF_DIR%"
    python "%TOOLS%\hf_download.py" %REPO% --local-dir "%HF_DIR%" ^
        config.json tokenizer.json tokenizer_config.json special_tokens_map.json ^
        generation_config.json model.safetensors chat_template.jinja
    if !errorlevel! neq 0 (
        echo [FATAL] HuggingFace download FAILED
        exit /b 1
    )
    echo  OK: Downloaded to %HF_DIR%
)
echo.

REM ====== Step 2: Export to v7 .bin format ======
echo [2/3] Exporting to v7 checkpoint ...
if exist "%BIN_OUT%" (
    echo  [SKIP] Already exists: %BIN_OUT%
) else (
    python "%TOOLS%\export.py" "%BIN_OUT%" "%HF_DIR%"
    if !errorlevel! neq 0 (
        echo [FATAL] export.py FAILED
        exit /b 1
    )
    echo  OK: Exported to %BIN_OUT%
)
echo.

REM ====== Step 3: Build v7 database ======
echo [3/3] Building v7 database ...
if exist "%DB_OUT%" (
    echo  [SKIP] Already exists: %DB_OUT%
) else (
    if not exist "%DB_DIR%" mkdir "%DB_DIR%"
    "%BIN%\build_db.exe" "%BIN_OUT%" -thinking 0 -o "%DB_OUT%" -f
    if !errorlevel! neq 0 (
        echo [FATAL] build_db.exe FAILED
        exit /b 1
    )
    echo  OK: Database built at %DB_OUT%
)
echo.

echo ============================================================================
echo  Fetch Llama-3.2-1B-Instruct COMPLETE
echo  Checkpoint : %BIN_OUT%
echo  Database   : %DB_OUT%
echo  HF source  : %HF_DIR%
echo ============================================================================
endlocal
exit /b 0
