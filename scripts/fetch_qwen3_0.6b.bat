@echo off
REM fetch_qwen3_0.6b.bat -- Download Qwen3-0.6B from HuggingFace Hub and export to v7 .bin format
REM
REM Architecture: Qwen3ForCausalLM (standard Qwen3, fully compatible)
REM 0.6B params, dim=1024, 28 layers, 16 heads, 8 KV heads (GQA), head_dim=128,
REM hidden_dim=3072, vocab=151936, max_seq_len=40960, rope_theta=1000000,
REM rms_eps=1e-6, has_qk_norm=1, tie_word_embeddings=1.
REM
REM This is the reference model of the project: verify.bat, the database
REM builders and the Ardmark RAG demo all expect
REM weights\qwen3_0.6b\qwen3_0.6b.bin to exist.
REM
REM Memory: ~1.5GB safetensors (bf16, 1 shard) + ~1GB peak during export.
REM Fits comfortably in 12GB RAM. Checkpoint is Q8-quantized (~630MB).
REM
REM Usage:  fetch_qwen3_0.6b.bat
REM
REM Output:
REM   weights\qwen3_0.6b\               -- raw HF files + v7 checkpoint
REM   weights\databases\qwen3_0.6b\lz4ht.db -- v7 microdb database

setlocal ENABLEDELAYEDEXPANSION
set PYTHONIOENCODING=utf-8

set THIS_DIR=%~dp0..\
set TOOLS=%THIS_DIR%tools
set BIN=%THIS_DIR%bin
set HF_DIR=%THIS_DIR%weights\qwen3_0.6b
set BIN_OUT=%HF_DIR%\qwen3_0.6b.bin
set DB_DIR=%THIS_DIR%weights\databases\qwen3_0.6b
set DB_OUT=%DB_DIR%\lz4ht.db
set REPO=Qwen/Qwen3-0.6B
set SHARD=model.safetensors

echo.
echo ============================================================================
echo  Fetch Qwen3-0.6B from HuggingFace Hub and build v7 database
echo  (0.6B params, dim=1024, 28 layers, GQA 16/8, tied embeddings, rope 1M)
echo ============================================================================

REM ====== Step 1: Download from HuggingFace Hub (with retry) ======
echo [1/3] Downloading %REPO% ...
if exist "%HF_DIR%\%SHARD%" (
    echo  [SKIP] Already downloaded: %HF_DIR%\%SHARD%
    goto do_export
)
if not exist "%HF_DIR%" mkdir "%HF_DIR%"

set RETRY=0
:retry_dl
hf download %REPO% ^
    --local-dir "%HF_DIR%" ^
    config.json tokenizer.json tokenizer_config.json special_tokens_map.json ^
    generation_config.json merges.txt vocab.json %SHARD%
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
if not exist "%HF_DIR%\%SHARD%" (
    echo [FATAL] %SHARD% not found after download
    exit /b 1
)

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
echo  Fetch Qwen3-0.6B COMPLETE
echo  Checkpoint : %BIN_OUT%
echo  Database   : %DB_OUT%
echo  HF source  : %HF_DIR%
echo ============================================================================
endlocal
exit /b 0
