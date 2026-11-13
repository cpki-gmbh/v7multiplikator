@echo off
REM fetch_minicpm5_1b.bat -- Download MiniCPM5-1B from HF Hub and build v7 database
REM
REM Architecture: LlamaForCausalLM (standard Llama, fully compatible)
REM 1B params, eps=1e-6, rope_theta=5M, head_dim=128, vocab=130560
REM dim=1536, 24 layers, 16 heads, 2 KV heads (from config.json)
REM
REM Template: the chat template is NOT in tokenizer_config.json -- it lives in
REM a separate chat_template.jinja file at the HF repo root (BOS-prefixed
REM ChatML + <think>/</think> protocol). It is downloaded below and consumed
REM by tools/export.py via the build_prompts() fallback path.
REM
REM Output:
REM   weights\minicpm5_1b\                -- raw HF files + v7 checkpoint
REM   weights\databases\minicpm5_1b\lz4ht.db -- v7 microdb database

setlocal ENABLEDELAYEDEXPANSION
set PYTHONIOENCODING=utf-8

set THIS_DIR=%~dp0..\
set TOOLS=%THIS_DIR%tools
set BIN=%THIS_DIR%bin
set W=%THIS_DIR%weights
set HF_DIR=%W%\minicpm5_1b
set BIN_OUT=%HF_DIR%\minicpm5_1b.bin
set DB_DIR=%W%\databases\minicpm5_1b
set DB_OUT=%DB_DIR%\lz4ht.db
set REPO=openbmb/MiniCPM5-1B
set SUCCESS=0

echo.
echo ============================================================================
echo  Fetch MiniCPM5-1B from HuggingFace Hub
echo  (OpenBMB, 1B params, LlamaForCausalLM, rope_theta=5M)
echo ============================================================================
echo.

REM ====== Step 1: Download with retry ======
echo [1/3] Downloading %REPO% ...
if exist "%HF_DIR%\model-00000-of-00001.safetensors" if exist "%HF_DIR%\config.json" if exist "%HF_DIR%\chat_template.jinja" (
    echo  [SKIP] Already downloaded
    set SUCCESS=1
    goto :do_export
)
if not exist "%HF_DIR%" mkdir "%HF_DIR%"

set RETRY=0
:retry_dl
python "%TOOLS%\hf_download.py" %REPO% --local-dir "%HF_DIR%" ^
    --local-dir "%HF_DIR%" ^
    config.json tokenizer.json tokenizer_config.json special_tokens_map.json ^
    generation_config.json chat_template.jinja model-00000-of-00001.safetensors
if %errorlevel% neq 0 (
    set /a RETRY+=1
    if !RETRY! lss 3 (
        echo  [RETRY !RETRY!/3] Download failed, retrying in 15s...
        timeout /t 15 /nobreak >nul
        goto retry_dl
    )
    echo  [FATAL] Download FAILED after 3 retries.
    exit /b 1
)
if exist "%HF_DIR%\model-00000-of-00001.safetensors" set SUCCESS=1

:do_export
echo OK: HF model files ready
echo.

REM ====== Step 2: Export to v7 .bin format ======
echo [2/3] Exporting to v7 checkpoint ...
if exist "%BIN_OUT%" (
    echo  [SKIP] Already exists: %BIN_OUT%
) else (
    python "%TOOLS%\export.py" "%BIN_OUT%" "%HF_DIR%"
    if !errorlevel! neq 0 (
        echo  [FATAL] export.py FAILED
        exit /b 1
    )
    if not exist "%BIN_OUT%" (
        echo  [FATAL] %BIN_OUT% not created
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
    "%BIN%\build_db.exe" "%BIN_OUT%" -thinking 0 -rope-base 5000000 -o "%DB_OUT%" -f
    if !errorlevel! neq 0 (
        echo  [FATAL] build_db.exe FAILED
        exit /b 1
    )
    if not exist "%DB_OUT%" (
        echo  [FATAL] %DB_OUT% not created
        exit /b 1
    )
    echo  OK: Database built at %DB_OUT%
)
echo.

echo ============================================================================
echo  Fetch MiniCPM5-1B COMPLETE
echo  Checkpoint : %BIN_OUT%
echo  Database   : %DB_OUT%
echo  HF source  : %HF_DIR%
echo ============================================================================
endlocal
exit /b 0
