@echo off
REM fetch_smolvlm_256m.bat -- Download SmolVLM-256M-Instruct from HuggingFace Hub
REM and export to v7 .bin + .vision + .tokenizer + .template format.
REM
REM Model: HuggingFaceTB/SmolVLM-256M-Instruct
REM Architecture: SigLIP-base ViT (12L/768H, patch16, 512x512)
REM               + pixel-shuffle projector (scale_factor=4)
REM               + VLlama3 text decoder (30L/576H, GQA 3:1, head_dim=64)
REM Total params: ~256M (ViT ~25M + projector ~7M + LLM ~220M)
REM License: Apache-2.0
REM
REM Visual tokens per image: 64 (pixel-shuffle: (512/16/4)^2 = 8^2 = 64)
REM Text vocab: 49280, rope_theta=100000, max_pos=8192
REM
REM Usage:  fetch_smolvlm_256m.bat
REM
REM Requirements: Python 3, huggingface_hub, torch, transformers, numpy, jinja2
REM
REM Output:
REM   weights\smolvlm_256m\                              -- raw HF files + v7 checkpoint
REM   weights\smolvlm_256m\smolvlm_256m.bin              -- text LLM weights
REM   weights\smolvlm_256m\smolvlm_256m.vision           -- SigLIP-base + VHDR header
REM   weights\smolvlm_256m\smolvlm_256m.proj             -- pixel-shuffle projector blob
REM   weights\smolvlm_256m\smolvlm_256m.tokenizer        -- BPE merges/vocab
REM   weights\smolvlm_256m\smolvlm_256m.template*        -- chat templates
REM   weights\databases\smolvlm_256m\lz4ht.db            -- v7 microdb database (built by this script via build_db.exe)
REM
REM NOTE: This fetch script downloads and exports the model, then builds the
REM lz4ht.db database directly (step 3 below, via build_db.exe -vision -- the
REM SmolVLM pixel-shuffle projector is integrated in vit_pixel_shuffle.inc).

setlocal ENABLEDELAYEDEXPANSION
set PYTHONIOENCODING=utf-8

set THIS_DIR=%~dp0..\
set TOOLS=%THIS_DIR%tools
set BIN=%THIS_DIR%bin
set W=%THIS_DIR%weights
set HF_DIR=%W%\smolvlm_256m
set BIN_OUT=%HF_DIR%\smolvlm_256m.bin
set VISION_OUT=%HF_DIR%\smolvlm_256m.vision
set PROJ_OUT=%HF_DIR%\smolvlm_256m.proj
set DB_DIR=%W%\databases\smolvlm_256m
set DB_OUT=%DB_DIR%\lz4ht.db
set REPO=HuggingFaceTB/SmolVLM-256M-Instruct

echo.
echo ============================================================================
echo  Fetch SmolVLM-256M-Instruct from HuggingFace Hub and export to v7 format
echo ============================================================================
echo.

REM ====== Step 1: Download from HuggingFace Hub ======
echo [1/3] Downloading %REPO% ...
if exist "%HF_DIR%\model.safetensors" (
    echo  [SKIP] Already downloaded: %HF_DIR%\model.safetensors
) else (
    if not exist "%HF_DIR%" mkdir "%HF_DIR%"
    python -c "from huggingface_hub import snapshot_download; snapshot_download('%REPO%', local_dir=r'%HF_DIR%', ignore_patterns=['*.onnx','*.onnx_data','*.pt','*.h5','*.msgpack','*.tflite','openvino/*','coreml/*'])"
    if !errorlevel! neq 0 (
        echo [FATAL] HuggingFace download FAILED
        exit /b 1
    )
    if not exist "%HF_DIR%\model.safetensors" (
        echo [FATAL] model.safetensors not found after download
        exit /b 1
    )
    echo  OK: Downloaded to %HF_DIR%
)
echo.

REM ====== Step 2: Export to v7 format ======
echo [2/3] Exporting to v7 checkpoint (text + vision + projector) ...
if exist "%BIN_OUT%" (
    echo  [SKIP] Already exists: %BIN_OUT%
) else (
    REM SmolVLM uses pixel-shuffle projector with scale_factor=4 (256M variant)
    REM and image_size=512, patch_size=16. The export script reads the config
    REM directly for most values; --scale-factor is required because it is not
    REM in config.json as a top-level field (it is text_config.pixel_shuffle_factor).
    python "%TOOLS%\export_vlm.py" "%BIN_OUT%" "%HF_DIR%" ^
        --model-type smolvlm-256m ^
        --scale-factor 4 ^
        --visionproj "%PROJ_OUT%"
    if !errorlevel! neq 0 (
        echo [FATAL] export_vlm.py FAILED
        echo NOTE: SmolVLM export support in export_vlm.py is not yet implemented.
        echo       This fetch script is staged for when the integration is complete.
        echo       See smolvlm_analysis.md for the implementation plan.
        exit /b 1
    )
    echo  OK: Exported to %BIN_OUT%
    if not exist "%VISION_OUT%" (
        echo [FATAL] %VISION_OUT% not created by export_vlm.py
        exit /b 1
    )
    if not exist "%PROJ_OUT%" (
        echo [FATAL] %PROJ_OUT% not created by export_vlm.py
        exit /b 1
    )
)
echo.

REM ====== Step 3: Build v7 database (OPTIONAL — requires ASM changes) ======
echo [3/3] Building v7 database ...
if exist "%DB_OUT%" (
    echo  [SKIP] Already exists: %DB_OUT%
) else (
    if not exist "%DB_DIR%" mkdir "%DB_DIR%"
    if not exist "%BIN%\build_db.exe" (
        echo  [SKIP] build_db.exe not found -- run build.bat first
        echo        Once the SmolVLM ASM integration is complete, re-run this script.
        exit /b 0
    )
    "%BIN%\build_db.exe" "%BIN_OUT%" ^
        -vision "%VISION_OUT%" ^
        -visionproj "%PROJ_OUT%" ^
        -thinking 1 ^
        -rope-base 100000 ^
        -o "%DB_OUT%" -f
    if !errorlevel! neq 0 (
        echo [FATAL] build_db.exe FAILED
        echo NOTE: SmolVLM DB build requires the new VHDR slot 32 (ps_factor)
        echo       and the pixel-shuffle projector support in db_build_tool.c.
        echo       See smolvlm_analysis.md for the implementation plan.
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
echo  Fetch SmolVLM-256M-Instruct COMPLETE
echo  Checkpoint : %BIN_OUT%
echo  Vision     : %VISION_OUT%
echo  Projector  : %PROJ_OUT%
if exist "%DB_OUT%" echo  Database   : %DB_OUT%
echo  HF source  : %HF_DIR%
echo ============================================================================
echo.
echo  NEXT STEPS (after ASM integration per smolvlm_analysis.md):
echo    1. Implement src\asm\vit_pixel_shuffle.inc
echo    2. Wire projector dispatch in vit_forward.inc (VCFG_PROJ_TYPE=2 branch)
echo    3. Add SmolVLM branch to export_vlm.py (--model-type smolvlm-256m)
echo    4. Parameterize SigLIP extractor in v7_vlm_ref\export_vision.py
echo    5. Re-run this script to rebuild the database
echo    6. Run: bin\dbg_image_recog.exe bin\v7_threading.dll %DB_OUT% cat640.raw "What is in this image?"
echo.
endlocal
exit /b 0
