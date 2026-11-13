@echo off
REM export_all.bat -- Export all HuggingFace models to v7 checkpoint format
REM Runs export.py for each model in weights/
REM Requires: Python 3, torch, transformers, numpy, jinja2

setlocal ENABLEDELAYEDEXPANSION

set THIS_DIR=%~dp0..\
set TOOLS=%THIS_DIR%tools
set W=%THIS_DIR%weights

echo.
echo ============================================================================
echo  Export All Models to v7 Checkpoint Format
echo ============================================================================
echo.

REM ====== 1. Qwen3-0.6B ======
echo [1/5] Qwen3-0.6B ...
if exist "%THIS_DIR%models\qwen3_0.6b" (
    python "%TOOLS%\export.py" "%W%\qwen3_0.6b" "%THIS_DIR%models\qwen3_0.6b"
    if !errorlevel! neq 0 (
        echo [FATAL] Qwen3-0.6B export FAILED
        exit /b 1
    )
    echo  OK: Qwen3-0.6B
) else (
    echo  [SKIP] models\qwen3_0.6b not found
)
echo.

REM ====== 2. Qwen3-4B ======
echo [2/5] Qwen3-4B ...
if exist "%THIS_DIR%models\qwen3_4b" (
    python "%TOOLS%\export.py" "%W%\qwen3_4b" "%THIS_DIR%models\qwen3_4b"
    if !errorlevel! neq 0 (
        echo [FATAL] Qwen3-4B export FAILED
        exit /b 1
    )
    echo  OK: Qwen3-4B
) else (
    echo  [SKIP] models\qwen3_4b not found
)
echo.

REM ====== 3. Phi-3-mini ======
echo [3/5] Phi-3-mini ...
if exist "%W%\phi3_mini\model" (
    python "%TOOLS%\export.py" "%W%\phi3_mini\phi3_mini" "%W%\phi3_mini\model"
    if !errorlevel! neq 0 (
        echo [FATAL] Phi-3-mini export FAILED
        exit /b 1
    )
    echo  OK: Phi-3-mini
) else (
    echo  [SKIP] weights\phi3_mini\model not found
)
echo.

REM ====== 4. SmolLM2-135M ======
echo [4/5] SmolLM2-135M ...
if exist "%W%\smollm2_135m\model.safetensors" (
    python "%TOOLS%\export.py" "%W%\smollm2_135m\smollm2_135m" "%W%\smollm2_135m"
    if !errorlevel! neq 0 (
        echo [FATAL] SmolLM2-135M export FAILED
        exit /b 1
    )
    echo  OK: SmolLM2-135M
) else (
    echo  [SKIP] weights\smollm2_135m\model.safetensors not found
)
echo.

REM ====== 5. SmolLM2-360M ======
echo [5/5] SmolLM2-360M ...
if exist "%W%\smollm2_360m\smollm2_360m_hf" (
    python "%TOOLS%\export.py" "%W%\smollm2_360m\smollm2_360m" "%W%\smollm2_360m\smollm2_360m_hf"
    if !errorlevel! neq 0 (
        echo [FATAL] SmolLM2-360M export FAILED
        exit /b 1
    )
    echo  OK: SmolLM2-360M
) else (
    echo  [SKIP] weights\smollm2_360m\smollm2_360m_hf not found
)
echo.

echo ============================================================================
echo  EXPORT COMPLETE
echo ============================================================================
endlocal
exit /b 0
