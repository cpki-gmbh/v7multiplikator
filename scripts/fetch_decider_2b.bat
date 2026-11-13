@echo off
REM fetch_decider_2b.bat -- Download Mapika/decider-2b from HuggingFace Hub
REM
REM Architecture: Qwen3_5ForCausalLM (Qwen3.5 hybrid, text-only decision model)
REM 1.88B params, hidden=2048, 24 layers in a 3:1 pattern
REM   (18x gated delta-net linear attention + 6x full attention),
REM   full attn: 8 heads / 2 KV heads (GQA), head_dim=256, attn_output_gate=1,
REM   partial_rotary_factor=0.25, rope_theta=10000000,
REM   linear attn: 16 key heads x 128, 16 value heads x 128, conv kernel 4,
REM   intermediate=6144, vocab=248320, tie_word_embeddings=1, rms_eps=1e-6,
REM   max_position_embeddings=262144 (engine caps context at 32768).
REM
REM Base model: Qwen/Qwen3.5-2B-Base, fine-tuned for one-pass typed decisions
REM (System-One class). Single model.safetensors, 3.76 GB bf16.
REM
REM NOTE: export.py does not support the qwen3_5 family yet (pre-release
REM freeze). This script downloads the raw HF files only; Steps 2-3
REM (export to .bin, build lz4ht.db) are deferred until the qwen3_5 export
REM family and the qwen3_5_kernels integration land.
REM
REM Usage:  fetch_decider_2b.bat
REM
REM Output:
REM   weights\decider_2b\               -- raw HF files (downloaded)

setlocal ENABLEDELAYEDEXPANSION
set PYTHONIOENCODING=utf-8

set THIS_DIR=%~dp0..\
set HF_DIR=%THIS_DIR%weights\decider_2b
set REPO=Mapika/decider-2b
set WEIGHTS=model.safetensors

echo.
echo ============================================================================
echo  Fetch decider-2b from HuggingFace Hub (download only)
echo  (1.88B params, Qwen3.5 hybrid: 18x gated delta-net + 6x full attention)
echo ============================================================================
echo.

REM ====== Step 1: Download from HuggingFace Hub (with retry) ======
echo [1/1] Downloading %REPO% ...
if exist "%HF_DIR%\%WEIGHTS%" if exist "%HF_DIR%\config.json" (
    echo  [SKIP] Already downloaded: weights exist
    goto done
)
if not exist "%HF_DIR%" mkdir "%HF_DIR%"

set RETRY=0
:retry_dl
hf download %REPO% ^
    --local-dir "%HF_DIR%" ^
    config.json generation_config.json tokenizer.json tokenizer_config.json ^
    chat_template.jinja decider_config.json %WEIGHTS%
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
if not exist "%HF_DIR%\%WEIGHTS%" (
    echo [FATAL] %WEIGHTS% not downloaded
    exit /b 1
)

:done
echo  OK: Downloaded to %HF_DIR%
echo.
echo ============================================================================
echo  Fetch decider-2b COMPLETE (download only)
echo  HF source  : %HF_DIR%
echo  Export     : PENDING -- export.py qwen3_5 family + qwen3_5_kernels
echo               integration land after the pre-release freeze
echo ============================================================================
endlocal
exit /b 0
