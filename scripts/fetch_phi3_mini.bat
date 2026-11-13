@echo off
REM fetch_phi3_mini.bat -- Download Phi-3-mini-4k-instruct from HuggingFace Hub and export to v7 .bin format
REM
REM Phi-3-mini-4k-instruct: 3.8B params, dim=3072, 32 layers, 32 heads, 32 KV heads,
REM vocab=32064, head_dim=96, rope_base=10000. BPE tokenizer with byte-fallback.
REM No QK-norm (has_qk_norm=0). Fused qkv_proj + gate_up_proj weights.
REM
REM Memory: ~7.64GB safetensors (fp16, NOT converted to fp32). Peak ~9.3GB during export.
REM
REM Usage:  fetch_phi3_mini.bat
REM
REM Output:
REM   weights\phi3_mini\                  -- raw HF files (cached)
REM   weights\phi3_mini\phi3_mini.bin     -- v7 checkpoint
REM   weights\databases\phi3_mini\lz4ht.db -- v7 microdb database

setlocal ENABLEDELAYEDEXPANSION
set PYTHONIOENCODING=utf-8

set THIS_DIR=%~dp0..\
set TOOLS=%THIS_DIR%tools
set BIN=%THIS_DIR%bin
set HF_DIR=%THIS_DIR%weights\phi3_mini
set BIN_OUT=%HF_DIR%\phi3_mini.bin
set DB_DIR=%THIS_DIR%weights\databases\phi3_mini
set DB_OUT=%DB_DIR%\lz4ht.db
set REPO=microsoft/Phi-3-mini-4k-instruct

echo.
echo ============================================================================
echo  Fetch Phi-3-mini-4k-instruct from HuggingFace Hub and build v7 database
echo  (3.8B params, dim=3072, 32 layers, fused qkv_proj + gate_up_proj)
echo ============================================================================
echo.

REM ====== Step 1: Download from HuggingFace Hub ======
echo [1/3] Downloading %REPO% ...
if exist "%HF_DIR%\model-00001-of-00002.safetensors" if exist "%HF_DIR%\model-00002-of-00002.safetensors" (
    echo  [SKIP] Already downloaded: safetensors shards exist
) else (
    if not exist "%HF_DIR%" mkdir "%HF_DIR%"
    python "%TOOLS%\hf_download.py" %REPO% --local-dir "%HF_DIR%" ^
        --local-dir "%HF_DIR%" ^
        config.json tokenizer.json tokenizer_config.json special_tokens_map.json ^
        added_tokens.json model.safetensors.index.json ^
        model-00001-of-00002.safetensors model-00002-of-00002.safetensors
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
    "%BIN%\build_db.exe" "%BIN_OUT%" -thinking 0 -rope-base 10000 -o "%DB_OUT%" -f
    if !errorlevel! neq 0 (
        echo [FATAL] build_db.exe FAILED
        exit /b 1
    )
    echo  OK: Database built at %DB_OUT%
)
echo.

echo ============================================================================
echo  Fetch Phi-3-mini COMPLETE
echo  Checkpoint : %BIN_OUT%
echo  Database   : %DB_OUT%
echo  HF source  : %HF_DIR%
echo ============================================================================
endlocal
exit /b 0
