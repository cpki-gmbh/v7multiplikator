@echo off
REM fetch_qwen3_8b.bat -- Download Qwen3-8B from HuggingFace Hub and export to v7 .bin format
REM
REM There is no official Qwen3-7B. The dense Qwen3 family is 0.6B / 1.7B / 4B /
REM 8B / 14B / 32B, so the 7-8B tier is Qwen3-8B.
REM
REM Architecture: Qwen3ForCausalLM (standard Qwen3, fully compatible)
REM 8B params, dim=4096, 36 layers, 32 heads, 8 KV heads (GQA), head_dim=128,
REM hidden_dim=12288, vocab=151936, max_seq_len=40960, rope_theta=1000000,
REM rms_norm_eps=1e-6, has_qk_norm=1, tie_word_embeddings=0 (separate classifier).
REM
REM Source precision: bf16, the real full-precision weights (~16.4GB, 5 shards).
REM There is no fp32 Qwen3 repo; bf16 is the source every other v7 model uses
REM (bf16 - Q8, never fp8 - Q8: double-quantizing a lossy fp8 source for a
REM Q8 checkpoint only loses precision for no gain).
REM
REM RAM: tools/export.py safe_open-mmaps the shards and quantizes from tensor
REM VIEWS, so the 16.4GB of weights stay file-backed (evictable under memory
REM pressure) and never enter anonymous RAM. Combined with the chunked
REM quantizer (<=50M elements per chunk, fp32 temporaries <=~200MB) the export
REM peaks at a few GB -- it converts on a 12GB-RAM machine.
REM
REM Disk: ~16.4GB download + ~8.7GB .bin + ~8.5GB db. The safetensors shards
REM are deleted right after a successful export (the .bin + db are the
REM deliverables) so the tree fits on a small disk.
REM
REM Usage:  fetch_qwen3_8b.bat
REM
REM Output:
REM   weights\qwen3_8b\               -- HF files + v7 checkpoint (shards removed after export)
REM   weights\databases\qwen3_8b\lz4ht.db -- v7 microdb database

setlocal ENABLEDELAYEDEXPANSION
set PYTHONIOENCODING=utf-8

set THIS_DIR=%~dp0..\
set TOOLS=%THIS_DIR%tools
set BIN=%THIS_DIR%bin
set HF_DIR=%THIS_DIR%weights\qwen3_8b
set BIN_OUT=%HF_DIR%\qwen3_8b.bin
set DB_DIR=%THIS_DIR%weights\databases\qwen3_8b
set DB_OUT=%DB_DIR%\lz4ht.db
set REPO=Qwen/Qwen3-8B
set SHARD1=model-00001-of-00005.safetensors
set SHARD5=model-00005-of-00005.safetensors
set SUCCESS=0

echo.
echo ============================================================================
echo  Fetch Qwen3-8B from HuggingFace Hub and build v7 database
echo  (8B params, dim=4096, 36 layers, GQA 32/8, rope 1M -- 7-8B tier)
echo ============================================================================
echo.

REM ====== Step 1: Download from HuggingFace Hub (with retry) ======
echo [1/4] Downloading %REPO% (bf16, ~16.4GB, 5 shards) ...
if exist "%HF_DIR%\%SHARD1%" if exist "%HF_DIR%\%SHARD5%" (
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
    generation_config.json merges.txt vocab.json model.safetensors.index.json ^
    model-00001-of-00005.safetensors model-00002-of-00005.safetensors ^
    model-00003-of-00005.safetensors model-00004-of-00005.safetensors ^
    model-00005-of-00005.safetensors
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
if exist "%HF_DIR%\%SHARD1%" if exist "%HF_DIR%\%SHARD5%" set SUCCESS=1

:do_export
echo  OK: Downloaded to %HF_DIR%
echo.

REM ====== Step 2: Export to v7 .bin format ======
echo [2/4] Exporting to v7 checkpoint (mmap + chunked quantize, RAM-bounded) ...
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

REM ====== Step 2b: Reclaim the 16.4GB of safetensors ======
REM The Q8 checkpoint + database are the deliverables; the bf16 shards are only
REM needed for a re-export, and this machine's disk cannot hold both.
echo [2b/4] Removing safetensors shards (reclaiming ~16.4GB) ...
del "%HF_DIR%\model-00001-of-00005.safetensors"
del "%HF_DIR%\model-00002-of-00005.safetensors"
del "%HF_DIR%\model-00003-of-00005.safetensors"
del "%HF_DIR%\model-00004-of-00005.safetensors"
del "%HF_DIR%\model-00005-of-00005.safetensors"
echo  OK: shards removed
echo.

REM ====== Step 3: Build v7 database ======
echo [3/4] Building v7 database ...
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
echo  Fetch Qwen3-8B COMPLETE
echo  Checkpoint : %BIN_OUT%
echo  Database   : %DB_OUT%
echo  HF source  : %HF_DIR%
echo ============================================================================
endlocal
exit /b 0
