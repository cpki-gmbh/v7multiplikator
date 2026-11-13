@echo off
REM rebuild_all_dbs.bat -- Re-export all models with RMS eps header + rebuild DBs
REM Run sequentially because each export needs ~6GB RAM
REM After this, all DBs will have __sys_rms_eps with correct per-model values

setlocal ENABLEDELAYEDEXPANSION
set THIS_DIR=%~dp0..\
set TOOLS=%THIS_DIR%tools
set BIN=%THIS_DIR%bin
set W=%THIS_DIR%weights

echo.
echo ============================================================================
echo  Re-exporting all models with RMS eps header + rebuilding databases
echo ============================================================================

set PASS=0
set FAIL=0

REM ====== 1. Qwen3-0.6B ======
echo.
echo [1/6] Qwen3-0.6B ...
if exist "%W%\qwen3_0.6b\config.json" (
    python "%TOOLS%\export.py" "%W%\qwen3_0.6b\qwen3_0.6b.bin" "%W%\qwen3_0.6b"
    if !errorlevel! neq 0 ( echo   [FAIL] export.py FAILED & set /a FAIL+=1 & goto :model2 )
    if exist "%W%\databases\qwen3_0.6b\lz4ht.db" del "%W%\databases\qwen3_0.6b\lz4ht.db"
    "%BIN%\build_db.exe" "%W%\qwen3_0.6b\qwen3_0.6b.bin" -o "%W%\databases\qwen3_0.6b\lz4ht.db" -f
    if !errorlevel! neq 0 ( echo   [FAIL] build_db FAILED & set /a FAIL+=1 & goto :model2 )
    echo   [OK] Qwen3-0.6B done
    set /a PASS+=1
) else (
    echo   [SKIP] Qwen3-0.6B not found
)

:model2
REM ====== 2. SmolLM2-135M ======
echo.
echo [2/6] SmolLM2-135M ...
if exist "%W%\smollm2_135m\config.json" (
    python "%TOOLS%\export.py" "%W%\smollm2_135m\smollm2_135m.bin" "%W%\smollm2_135m"
    if !errorlevel! neq 0 ( echo   [FAIL] export.py FAILED & set /a FAIL+=1 & goto :model3 )
    if exist "%W%\databases\smollm2_135m\lz4ht.db" del "%W%\databases\smollm2_135m\lz4ht.db"
    "%BIN%\build_db.exe" "%W%\smollm2_135m\smollm2_135m.bin" -thinking 1 -rope-base 100000 -o "%W%\databases\smollm2_135m\lz4ht.db" -f
    if !errorlevel! neq 0 ( echo   [FAIL] build_db FAILED & set /a FAIL+=1 & goto :model3 )
    echo   [OK] SmolLM2-135M done
    set /a PASS+=1
) else (
    echo   [SKIP] SmolLM2-135M not found
)

:model3
REM ====== 3. SmolLM2-360M ======
echo.
echo [3/6] SmolLM2-360M ...
if exist "%W%\smollm2_360m\config.json" (
    python "%TOOLS%\export.py" "%W%\smollm2_360m\smollm2_360m.bin" "%W%\smollm2_360m"
    if !errorlevel! neq 0 ( echo   [FAIL] export.py FAILED & set /a FAIL+=1 & goto :model4 )
    if exist "%W%\databases\smollm2_360m\lz4ht.db" del "%W%\databases\smollm2_360m\lz4ht.db"
    "%BIN%\build_db.exe" "%W%\smollm2_360m\smollm2_360m.bin" -thinking 1 -rope-base 100000 -o "%W%\databases\smollm2_360m\lz4ht.db" -f
    if !errorlevel! neq 0 ( echo   [FAIL] build_db FAILED & set /a FAIL+=1 & goto :model4 )
    echo   [OK] SmolLM2-360M done
    set /a PASS+=1
) else (
    echo   [SKIP] SmolLM2-360M not found
)

:model4
REM ====== 4. Llama 3.2 1B Instruct ======
echo.
echo [4/6] Llama-3.2-1B-Instruct ...
if exist "%W%\llama_3.2_1b\config.json" (
    python "%TOOLS%\export.py" "%W%\llama_3.2_1b\llama_3.2_1b.bin" "%W%\llama_3.2_1b"
    if !errorlevel! neq 0 ( echo   [FAIL] export.py FAILED & set /a FAIL+=1 & goto :model5 )
    if exist "%W%\databases\llama_3.2_1b\lz4ht.db" del "%W%\databases\llama_3.2_1b\lz4ht.db"
    "%BIN%\build_db.exe" "%W%\llama_3.2_1b\llama_3.2_1b.bin" -o "%W%\databases\llama_3.2_1b\lz4ht.db" -f
    if !errorlevel! neq 0 ( echo   [FAIL] build_db FAILED & set /a FAIL+=1 & goto :model5 )
    echo   [OK] Llama-3.2-1B done
    set /a PASS+=1
) else (
    echo   [SKIP] Llama-3.2-1B not found
)

:model5
REM ====== 5. Phi-3-mini ======
echo.
echo [5/6] Phi-3-mini ...
if exist "%W%\phi3_mini\config.json" (
    python "%TOOLS%\export.py" "%W%\phi3_mini\phi3_mini.bin" "%W%\phi3_mini"
    if !errorlevel! neq 0 ( echo   [FAIL] export.py FAILED & set /a FAIL+=1 & goto :model6 )
    if exist "%W%\databases\phi3_mini\lz4ht.db" del "%W%\databases\phi3_mini\lz4ht.db"
    "%BIN%\build_db.exe" "%W%\phi3_mini\phi3_mini.bin" -thinking 0 -rope-base 10000 -o "%W%\databases\phi3_mini\lz4ht.db" -f
    if !errorlevel! neq 0 ( echo   [FAIL] build_db FAILED & set /a FAIL+=1 & goto :model6 )
    echo   [OK] Phi-3-mini done
    set /a PASS+=1
) else (
    echo   [SKIP] Phi-3-mini not found
)

:model6
REM ====== 6. TinyLlama-1.1B ======
echo.
echo [6/6] TinyLlama-1.1B ...
if exist "%W%\tinyllama_1.1b\config.json" (
    python "%TOOLS%\export.py" "%W%\tinyllama_1.1b\tinyllama_1.1b.bin" "%W%\tinyllama_1.1b"
    if !errorlevel! neq 0 ( echo   [FAIL] export.py FAILED & set /a FAIL+=1 & goto :summary )
    if exist "%W%\databases\tinyllama_1.1b\lz4ht.db" del "%W%\databases\tinyllama_1.1b\lz4ht.db"
    "%BIN%\build_db.exe" "%W%\tinyllama_1.1b\tinyllama_1.1b.bin" -o "%W%\databases\tinyllama_1.1b\lz4ht.db" -f
    if !errorlevel! neq 0 ( echo   [FAIL] build_db FAILED & set /a FAIL+=1 & goto :summary )
    echo   [OK] TinyLlama done
    set /a PASS+=1
) else (
    echo   [SKIP] TinyLlama-1.1B not found
)

:summary
echo.
echo ============================================================================
echo  Rebuild complete: !PASS! succeeded, !FAIL! failed
echo ============================================================================
endlocal
