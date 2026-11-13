@echo off
REM fetch_qwen3_1.7b.bat -- Download Qwen3-1.7B from HuggingFace Hub and export to v7 .bin format
REM
REM Architecture: Qwen3ForCausalLM (standard Qwen3, fully compatible)
REM 1.7B params, dim=2048, 28 layers, 16 heads, 8 KV heads (GQA), head_dim=128,
REM hidden_dim=6144, vocab=151936, max_seq_len=40960, rope_theta=1000000,
REM rms_eps=1e-6, has_qk_norm=1, tie_word_embeddings=1.
REM
REM This is the next bigger official Qwen3 model above Qwen3-0.6B
REM (the dense Qwen3 family is 0.6B / 1.7B / 4B / 8B / 14B / 32B).
REM
REM Memory: ~4.1GB safetensors (bf16, 2 shards) + ~2GB peak during export.
REM Fits comfortably in 12GB RAM. Checkpoint is Q8-quantized (~1.9GB).
REM
REM Usage:  fetch_qwen3_1.7b.bat
REM
REM Output:
REM   weights\qwen3_1.7b\               -- raw HF files + v7 checkpoint
REM   weights\databases\qwen3_1.7b\lz4ht.db -- v7 microdb database

setlocal ENABLEDELAYEDEXPANSION
set PYTHONIOENCODING=utf-8

set THIS_DIR=%~dp0..\
set TOOLS=%THIS_DIR%tools
set BIN=%THIS_DIR%bin
set HF_DIR=%THIS_DIR%weights\qwen3_1.7b
set BIN_OUT=%HF_DIR%\qwen3_1.7b.bin
set DB_DIR=%THIS_DIR%weights\databases\qwen3_1.7b
set DB_OUT=%DB_DIR%\lz4ht.db
set REPO=Qwen/Qwen3-1.7B
set SHARD1=model-00001-of-00002.safetensors
set SHARD2=model-00002-of-00002.safetensors
set SUCCESS=0

echo.
echo ============================================================================
echo  Fetch Qwen3-1.7B from HuggingFace Hub and build v7 database
echo  (1.7B params, dim=2048, 28 layers, GQA 16/8, tied embeddings, rope 1M)
echo ============================================================================
echo.

REM ====== Step 1: Download from HuggingFace Hub (with retry) ======
echo [1/3] Downloading %REPO% ...
if exist "%HF_DIR%\%SHARD1%" if exist "%HF_DIR%\%SHARD2%" (
    echo  [SKIP] Already downloaded: safetensors shards exist
    set SUCCESS=1
    goto do_export
)
if not exist "%HF_DIR%" mkdir "%HF_DIR%"

set RETRY=0
:retry_dl
hf download %REPO% ^
    --local-dir "%HF_DIR%" ^
    config.json tokenizer.json tokenizer_config.json special_tokens_map.json ^
    generation_config.json merges.txt vocab.json %SHARD1% %SHARD2%
if !errorlevel! neq 0 (
    set /a RETRY+=1
    if !RETRY! lss 3 (
        echo  [RETRY !RETRY!/3] Download failed, retrying in 15s...
        timeout /t 15 /nobreak >nul
        goto retry_dl
    )
    echo [FATAL] HuggingFace download FAILED after 3 retries
    exit /b 1
)
if exist "%HF_DIR%\%SHARD1%" if exist "%HF_DIR%\%SHARD2%" set SUCCESS=1

:do_export
echo  OK: Downloaded to %HF_DIR%
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
    if not exist "%BIN_OUT%" (
        echo [FATAL] %BIN_OUT% not created
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
    "%BIN%\build_db.exe" "%BIN_OUT%" -o "%DB_OUT%" -f
    if !errorlevel! neq 0 (
        echo [FATAL] build_db.exe FAILED
        exit /b 1
    )
    if not exist "%DB_OUT%" (
        echo [FATAL] %DB_OUT% not created
        exit /b 1
    )
    echo  OK: Database built at %DB_OUT%
)
echo.

echo ============================================================================
echo  Fetch Qwen3-1.7B COMPLETE
echo  Checkpoint : %BIN_OUT%
echo  Database   : %DB_OUT%
echo  HF source  : %HF_DIR%
echo ============================================================================
endlocal
exit /b 0
