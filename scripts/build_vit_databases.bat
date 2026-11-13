@echo off
REM build_vit_databases.bat -- Build databases for ALL vision-language models
REM Scans weights\ for .bin + .vision + .proj files and builds lz4ht.db
REM VLMs are skipped if not fetched (no .bin file present)
REM
REM Prerequisites: build.bat must have been run first (needs build_db.exe)

setlocal ENABLEDELAYEDEXPANSION

set THIS_DIR=%~dp0..\
set W=%THIS_DIR%weights
set BIN=%THIS_DIR%bin
set BUILD_DB=%BIN%\build_db.exe
set ERRORS=0

if not exist "%BUILD_DB%" (
    echo [FATAL] build_db.exe not found -- run build.bat first
    exit /b 1
)

echo.
echo ============================================================================
echo  Building VLM databases
echo ============================================================================
echo.

REM ====== Qwen3-VL 4B ======
set BIN_OUT=%W%\qwen3_vl_4b\qwen3_vl_4b.bin
set VISION_OUT=%W%\qwen3_vl_4b\qwen3_vl_4b.vision
set DB_DIR=%W%\databases\qwen3_vl_4b
set DB_OUT=%DB_DIR%\lz4ht.db
if exist "%BIN_OUT%" if exist "%VISION_OUT%" (
    if not exist "%DB_DIR%" mkdir "%DB_DIR%"
    echo  Building Qwen3-VL 4B ...
    "%BUILD_DB%" "%BIN_OUT%" -vision "%VISION_OUT%" -o "%DB_OUT%" -f
    if !errorlevel! neq 0 (echo [FAIL] Qwen3-VL 4B & set /a ERRORS+=1) else (echo OK: Qwen3-VL 4B)
) else (echo  [SKIP] Qwen3-VL 4B not fetched)

REM ====== LLaVA-OneVision-Qwen2-0.5B ======
set BIN_OUT=%W%\llava_onevision_qwen2_0.5b\llava_onevision_qwen2_0.5b.bin
set VISION_OUT=%W%\llava_onevision_qwen2_0.5b\llava_onevision_qwen2_0.5b.vision
set PROJ_OUT=%W%\llava_onevision_qwen2_0.5b\llava_onevision_qwen2_0.5b.proj
set DB_DIR=%W%\databases\llava_onevision_qwen2_0.5b
set DB_OUT=%DB_DIR%\lz4ht.db
if exist "%BIN_OUT%" if exist "%VISION_OUT%" (
    if not exist "%DB_DIR%" mkdir "%DB_DIR%"
    echo  Building LLaVA-OneVision-Qwen2-0.5B ...
    if exist "%PROJ_OUT%" (
        "%BUILD_DB%" "%BIN_OUT%" -vision "%VISION_OUT%" -visionproj "%PROJ_OUT%" -o "%DB_OUT%" -f
    ) else (
        "%BUILD_DB%" "%BIN_OUT%" -vision "%VISION_OUT%" -o "%DB_OUT%" -f
    )
    if !errorlevel! neq 0 (echo [FAIL] LLaVA-OneVision-Qwen2-0.5B & set /a ERRORS+=1) else (echo OK: LLaVA-OneVision-Qwen2-0.5B)
) else (echo  [SKIP] LLaVA-OneVision-Qwen2-0.5B not fetched)

REM ====== SmolVLM-256M ======
set BIN_OUT=%W%\smolvlm_256m\smolvlm_256m.bin
set VISION_OUT=%W%\smolvlm_256m\smolvlm_256m.vision
set PROJ_OUT=%W%\smolvlm_256m\smolvlm_256m.proj
set DB_DIR=%W%\databases\smolvlm_256m
set DB_OUT=%DB_DIR%\lz4ht.db
if exist "%BIN_OUT%" if exist "%VISION_OUT%" (
    if not exist "%DB_DIR%" mkdir "%DB_DIR%"
    echo  Building SmolVLM-256M ...
    if exist "%PROJ_OUT%" (
        "%BUILD_DB%" "%BIN_OUT%" -vision "%VISION_OUT%" -visionproj "%PROJ_OUT%" -thinking 1 -rope-base 100000 -o "%DB_OUT%" -f
    ) else (
        "%BUILD_DB%" "%BIN_OUT%" -vision "%VISION_OUT%" -thinking 1 -rope-base 100000 -o "%DB_OUT%" -f
    )
    if !errorlevel! neq 0 (echo [FAIL] SmolVLM-256M & set /a ERRORS+=1) else (echo OK: SmolVLM-256M)
) else (echo  [SKIP] SmolVLM-256M not fetched)

echo.
if !ERRORS! gtr 0 (
    echo ============================================================================
    echo  !ERRORS! VLM database build failures
    echo ============================================================================
    exit /b 1
)
echo ============================================================================
echo  ALL VLM DATABASES BUILT
echo ============================================================================
endlocal
