@echo off
REM fetch_tinyllama.bat -- Download TinyLlama-1.1B-Chat-v1.0 and export to v7 format
REM
REM Downloads the model from HuggingFace Hub, then runs export.py to produce:
REM   weights\tinyllama_1.1b\tinyllama           (quantized checkpoint)
REM   weights\tinyllama_1.1b\tinyllama.tokenizer (BPE tokenizer data)
REM   weights\tinyllama_1.1b\tinyllama.template* (prompt templates)

setlocal

set MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0
set HF_DIR=weights\tinyllama_1.1b\hf_source
set OUT_BASE=weights\tinyllama_1.1b\tinyllama

echo === Step 1: Download TinyLlama from HuggingFace ===
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0', local_dir=r'weights/tinyllama_1.1b/hf_source', ignore_patterns=['*.gguf','*.pt','*.bin','original/*'])"
if errorlevel 1 (
    echo [FATAL] Download failed. Install huggingface_hub: pip install huggingface_hub
    exit /b 1
)
echo Download complete: weights\tinyllama_1.1b\hf_source

echo === Step 2: Export to v7 format ===
python tools\export.py %OUT_BASE% %HF_DIR%
if errorlevel 1 (
    echo [FATAL] Export failed.
    exit /b 1
)

echo === Step 3: Build microdb database ===
if not exist weights\databases\tinyllama_1.1b mkdir weights\databases\tinyllama_1.1b
bin\build_db.exe %OUT_BASE% -o weights\databases\tinyllama_1.1b\lz4ht.db
if errorlevel 1 (
    echo [FATAL] build_db failed.
    exit /b 1
)

echo === Done ===
echo Checkpoint:  %OUT_BASE%
echo Tokenizer:   %OUT_BASE%.tokenizer
echo Database:    weights\databases\tinyllama_1.1b\lz4ht.db
endlocal
