@echo off
REM fetch_llava_onevision_qwen2_0.5b.bat -- Download LLaVA-OneVision-Qwen2-0.5B
REM from HuggingFace Hub and export to v7 .bin format
REM
REM Model: llava-hf/llava-onevision-qwen2-0.5b-ov-hf
REM Architecture: SigLIP-ViT (26L/1152H) + MLP projector + Qwen2-0.5B
REM Total params: ~0.9B (ViT ~400M + projector ~12M + LLM ~500M)
REM License: Apache-2.0
REM
REM Usage:  fetch_llava_onevision_qwen2_0.5b.bat
REM
REM Steps:
REM   1. Download model files from HuggingFace Hub
REM   2. Run export_vlm.py to produce .bin + .vision + .tokenizer + .template*
REM   3. Run build_db.exe to build lz4ht.db
REM
REM Requirements: Python 3, huggingface_hub, torch, transformers, numpy, jinja2
REM
REM Output:
REM   weights\llava_onevision_qwen2_0.5b\       -- raw HF files + v7 checkpoint
REM   weights\databases\llava_onevision_qwen2_0.5b\lz4ht.db -- v7 microdb database

setlocal ENABLEDELAYEDEXPANSION

set THIS_DIR=%~dp0..\
set TOOLS=%THIS_DIR%tools
set BIN=%THIS_DIR%bin
set W=%THIS_DIR%weights
set HF_DIR=%W%\llava_onevision_qwen2_0.5b
set BIN_OUT=%HF_DIR%\llava_onevision_qwen2_0.5b.bin
set DB_DIR=%W%\databases\llava_onevision_qwen2_0.5b
set DB_OUT=%DB_DIR%\lz4ht.db
set REPO=llava-hf/llava-onevision-qwen2-0.5b-ov-hf

echo.
echo ============================================================================
echo  Fetch LLaVA-OneVision-Qwen2-0.5B from HuggingFace Hub and build v7 DB
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
        generation_config.json merges.txt vocab.json model.safetensors ^
        preprocessor_config.json processor_config.json chat_template.json
    if !errorlevel! neq 0 (
        echo [FATAL] HuggingFace download FAILED
        exit /b 1
    )
    echo  OK: Downloaded to %HF_DIR%
)
echo.

REM ====== Step 2: Export to v7 format ======
echo [2/3] Exporting to v7 checkpoint ...
if exist "%BIN_OUT%" (
    echo  [SKIP] Already exists: %BIN_OUT%
) else (
    python "%TOOLS%\export_vlm.py" "%BIN_OUT%" "%HF_DIR%" --model-type llava-onevision
    if !errorlevel! neq 0 (
        echo [FATAL] export_vlm.py FAILED
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
echo  Fetch LLaVA-OneVision-Qwen2-0.5B COMPLETE
echo  Checkpoint : %BIN_OUT%
echo  Database   : %DB_OUT%
echo  HF source  : %HF_DIR%
echo ============================================================================
endlocal
exit /b 0
