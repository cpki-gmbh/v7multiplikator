@echo off
REM fetch_qwen2.5_0.5b.bat -- Download Qwen2.5-0.5B-Instruct from HuggingFace Hub
REM                         and export to v7 .bin format
REM
REM This is the BASE text model of our LLaVA-OneVision-Qwen2-0.5B. We need it
REM as a TEXT-ONLY reference to isolate whether the Qwen2-0.5B forward compute
REM bug is in the weight layout / forward kernels or somewhere in the LLaVA
REM wrapper. If pure Qwen2.5-0.5B-Instruct also produces gibberish in v7, the
REM bug is in the Qwen2 text path (not vision). If it works, the bug is in
REM how the LLaVA export remaps language_model.* -> model.*.
REM
REM Usage:  fetch_qwen2.5_0.5b.bat
REM
REM Steps:
REM   1. Download model files from HuggingFace Hub (Qwen/Qwen2.5-0.5B-Instruct)
REM      - config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json
REM      - merges.txt, vocab.json, model.safetensors
REM   2. Run export.py to produce weights\qwen2.5_0.5b\qwen2.5_0.5b.bin + .tokenizer + .template*
REM   3. Run build_db.exe to build weights\databases\qwen2.5_0.5b\lz4ht.db
REM
REM Requirements: Python 3, huggingface_hub, torch, transformers, numpy, jinja2
REM
REM Output:
REM   weights\qwen2.5_0.5b\               -- raw HF files + v7 checkpoint
REM   weights\databases\qwen2.5_0.5b\lz4ht.db -- v7 microdb database

setlocal ENABLEDELAYEDEXPANSION

set THIS_DIR=%~dp0..\
set TOOLS=%THIS_DIR%tools
set BIN=%THIS_DIR%bin
set W=%THIS_DIR%weights
set HF_DIR=%W%\qwen2.5_0.5b
set BIN_OUT=%HF_DIR%\qwen2.5_0.5b.bin
set DB_DIR=%W%\databases\qwen2.5_0.5b
set DB_OUT=%DB_DIR%\lz4ht.db
set REPO=Qwen/Qwen2.5-0.5B-Instruct

echo.
echo ============================================================================
echo  Fetch Qwen2.5-0.5B-Instruct from HuggingFace Hub and build v7 database
echo  (text-only reference model for LLaVA-OneVision-Qwen2-0.5B debugging)
echo ============================================================================
echo.

REM ====== Step 1: Download from HuggingFace Hub ======
echo [1/3] Downloading %REPO% ...
if exist "%HF_DIR%\model.safetensors" (
    echo  [SKIP] Already downloaded: %HF_DIR%\model.safetensors
) else (
    if not exist "%HF_DIR%" mkdir "%HF_DIR%"
    python "%TOOLS%\hf_download.py" %REPO% --local-dir "%HF_DIR%" ^
        --local-dir "%HF_DIR%" ^
        config.json tokenizer.json tokenizer_config.json special_tokens_map.json ^
        generation_config.json merges.txt vocab.json model.safetensors
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
    "%BIN%\build_db.exe" "%BIN_OUT%" -o "%DB_OUT%" -f
    if !errorlevel! neq 0 (
        echo [FATAL] build_db.exe FAILED
        exit /b 1
    )
    echo  OK: Database built at %DB_OUT%
)
echo.

echo ============================================================================
echo  Fetch Qwen2.5-0.5B-Instruct COMPLETE
echo  Checkpoint : %BIN_OUT%
echo  Database   : %DB_OUT%
echo  HF source  : %HF_DIR%
echo ============================================================================
echo.
echo  To test text-only inference (should answer "paris" etc.):
echo    bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen2.5_0.5b\lz4ht.db
echo ============================================================================
endlocal
exit /b 0
