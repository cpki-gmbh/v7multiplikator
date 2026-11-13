
@echo off
REM build_text_model_databases.bat -- Build databases for ALL text models
REM Builds EVERY present model's lz4ht.db unconditionally (no freshness logic:
REM this script's job is to build the databases, not to decide whether to).
REM Qwen3-0.6B is MANDATORY -- FATAL if missing
REM Other models are skipped if their .bin is not present (not fetched yet)
REM
REM Prerequisites: build.bat must have been run first (needs build_db.exe)

set THIS_DIR=%~dp0..\
set W=%THIS_DIR%weights
set BIN=%THIS_DIR%bin
set BUILD_DB=%BIN%\build_db.exe

if not exist "%BUILD_DB%" (
    echo [FATAL] build_db.exe not found -- run build.bat first
    exit /b 1
)

echo.
echo ============================================================================
echo  Building text model databases
echo ============================================================================
echo.


REM ====== Phi-3-mini ======
if exist "%W%\phi3_mini\phi3_mini.bin" (
    if not exist "%W%\databases\phi3_mini" mkdir "%W%\databases\phi3_mini"
    echo  Building Phi-3-mini ...
    "%BUILD_DB%" "%W%\phi3_mini\phi3_mini.bin" -thinking 0 -rope-base 10000 -o "%W%\databases\phi3_mini\lz4ht.db" -f -tuner-delay 0 -tuner-rand-delay 1 -tuner-delay-large 0 -autoexec "%THIS_DIR%autoexec.logo"
    if %errorlevel% NEQ 0 (
        echo [FAIL] Phi-3-mini
        exit /b 1
    )
    echo OK: Phi-3-mini
)

exit /b




REM ====== Qwen3-1.7B ======
if exist "%W%\qwen3_1.7b\qwen3_1.7b.bin" (
    if not exist "%W%\databases\qwen3_1.7b" mkdir "%W%\databases\qwen3_1.7b"
    echo  Building Qwen3-1.7B ...
    "%BUILD_DB%" "%W%\qwen3_1.7b\qwen3_1.7b.bin" -o "%W%\databases\qwen3_1.7b\lz4ht.db" -f -tuner-delay 0 -tuner-rand-delay 1 -tuner-delay-large 1 -autoexec "%THIS_DIR%autoexec.logo"
    if %errorlevel% NEQ 0 (
        echo [FAIL] Qwen3-1.7B
        exit /b 1
    )
    echo OK: Qwen3-1.7B
)

exit /b



REM ====== Qwen3-0.6B (MANDATORY) ======
if not exist "%W%\qwen3_0.6b\qwen3_0.6b.bin" (
    echo [FATAL] Qwen3-0.6B checkpoint not found at %W%\qwen3_0.6b\qwen3_0.6b.bin
    echo  Run fetch_qwen3_0.6b.bat to download and export the model.
    exit /b 1
)
if not exist "%W%\databases\qwen3_0.6b" mkdir "%W%\databases\qwen3_0.6b"
echo  Building Qwen3-0.6B ...
REM -autoexec: the SectorLogo stdlib is part of every engine-facing DB (verify.bat
REM builds the reference DB the same way).
"%BUILD_DB%" "%W%\qwen3_0.6b\qwen3_0.6b.bin" -o "%W%\databases\qwen3_0.6b\lz4ht.db" -f -tuner-delay 999 -tuner-rand-delay 100 -tuner-delay-large 100 -autoexec "%THIS_DIR%autoexec.logo"
if %errorlevel% NEQ 0 (
    echo [FATAL] Qwen3-0.6B database build FAILED
    exit /b 1
)


exit /b



REM ====== MiniCPM5-1B ======
if exist "%W%\minicpm5_1b\minicpm5_1b.bin" (
    if not exist "%W%\databases\minicpm5_1b" mkdir "%W%\databases\minicpm5_1b"
    echo  Building MiniCPM5-1B ...
    "%BUILD_DB%" "%W%\minicpm5_1b\minicpm5_1b.bin" -thinking 0 -rope-base 5000000 -o "%W%\databases\minicpm5_1b\lz4ht.db" -f -tuner-delay 100 -tuner-rand-delay 1 -tuner-delay-large 100 -autoexec "%THIS_DIR%autoexec.logo"
    if %errorlevel% NEQ 0 (
        echo [FAIL] MiniCPM5-1B
        exit /b 1
    )
    echo OK: MiniCPM5-1B
)

exit /b



REM ====== Phi-3-mini ======
if exist "%W%\phi3_mini\phi3_mini.bin" (
    if not exist "%W%\databases\phi3_mini" mkdir "%W%\databases\phi3_mini"
    echo  Building Phi-3-mini ...
    "%BUILD_DB%" "%W%\phi3_mini\phi3_mini.bin" -thinking 0 -rope-base 10000 -o "%W%\databases\phi3_mini\lz4ht.db" -f -tuner-delay 100 -tuner-rand-delay 1 -tuner-delay-large 100 -autoexec "%THIS_DIR%autoexec.logo"
    if %errorlevel% NEQ 0 (
        echo [FAIL] Phi-3-mini
        exit /b 1
    )
    echo OK: Phi-3-mini
)




exit /b



REM ====== Phi-3-mini ======
if exist "%W%\phi3_mini\phi3_mini.bin" (
    if not exist "%W%\databases\phi3_mini" mkdir "%W%\databases\phi3_mini"
    echo  Building Phi-3-mini ...
    "%BUILD_DB%" "%W%\phi3_mini\phi3_mini.bin" -thinking 0 -rope-base 10000 -o "%W%\databases\phi3_mini\lz4ht.db" -f -tuner-delay 1 -tuner-rand-delay 1 -tuner-delay-large 1 -autoexec "%THIS_DIR%autoexec.logo"
    if %errorlevel% NEQ 0 (
        echo [FAIL] Phi-3-mini
        exit /b 1
    )
    echo OK: Phi-3-mini
)






REM ====== MiniCPM5-1B ======
if exist "%W%\minicpm5_1b\minicpm5_1b.bin" (
    if not exist "%W%\databases\minicpm5_1b" mkdir "%W%\databases\minicpm5_1b"
    echo  Building MiniCPM5-1B ...
    "%BUILD_DB%" "%W%\minicpm5_1b\minicpm5_1b.bin" -thinking 0 -rope-base 5000000 -o "%W%\databases\minicpm5_1b\lz4ht.db" -f -tuner-delay 1 -tuner-rand-delay 1 -tuner-delay-large 1 -autoexec "%THIS_DIR%autoexec.logo"
    if %errorlevel% NEQ 0 (
        echo [FAIL] MiniCPM5-1B
        exit /b 1
    )
    echo OK: MiniCPM5-1B
)



REM ====== Qwen2.5-0.5B ======
if exist "%W%\qwen2.5_0.5b\qwen2.5_0.5b.bin" (
    if not exist "%W%\databases\qwen2.5_0.5b" mkdir "%W%\databases\qwen2.5_0.5b"
    echo  Building Qwen2.5-0.5B ...
    "%BUILD_DB%" "%W%\qwen2.5_0.5b\qwen2.5_0.5b.bin" -o "%W%\databases\qwen2.5_0.5b\lz4ht.db" -f
    if %errorlevel% NEQ 0 (
        echo [FAIL] Qwen2.5-0.5B
        exit /b 1
    )
    echo OK: Qwen2.5-0.5B
)

REM ====== Llama-3.2-1B ======
if exist "%W%\llama_3.2_1b\llama_3.2_1b.bin" (
    if not exist "%W%\databases\llama_3.2_1b" mkdir "%W%\databases\llama_3.2_1b"
    echo  Building Llama-3.2-1B ...
    "%BUILD_DB%" "%W%\llama_3.2_1b\llama_3.2_1b.bin" -thinking 0 -o "%W%\databases\llama_3.2_1b\lz4ht.db" -f
    if %errorlevel% NEQ 0 (
        echo [FAIL] Llama-3.2-1B
        exit /b 1
    )
    echo OK: Llama-3.2-1B
)

REM ====== SmolLM2-135M ======
if exist "%W%\smollm2_135m\smollm2_135m.bin" (
    if not exist "%W%\databases\smollm2_135m" mkdir "%W%\databases\smollm2_135m"
    echo  Building SmolLM2-135M ...
    "%BUILD_DB%" "%W%\smollm2_135m\smollm2_135m.bin" -thinking 1 -rope-base 100000 -o "%W%\databases\smollm2_135m\lz4ht.db" -f
    if %errorlevel% NEQ 0 (
        echo [FAIL] SmolLM2-135M
        exit /b 1
    )
    echo OK: SmolLM2-135M
)

REM ====== SmolLM2-360M ======
if exist "%W%\smollm2_360m\smollm2_360m.bin" (
    if not exist "%W%\databases\smollm2_360m" mkdir "%W%\databases\smollm2_360m"
    echo  Building SmolLM2-360M ...
    "%BUILD_DB%" "%W%\smollm2_360m\smollm2_360m.bin" -thinking 1 -rope-base 100000 -o "%W%\databases\smollm2_360m\lz4ht.db" -f
    if %errorlevel% NEQ 0 (
        echo [FAIL] SmolLM2-360M
        exit /b 1
    )
    echo OK: SmolLM2-360M
)

REM ====== TinyLlama-1.1B ======
if exist "%W%\tinyllama_1.1b\tinyllama_1.1b.bin" (
    if not exist "%W%\databases\tinyllama_1.1b" mkdir "%W%\databases\tinyllama_1.1b"
    echo  Building TinyLlama-1.1B ...
    "%BUILD_DB%" "%W%\tinyllama_1.1b\tinyllama_1.1b.bin" -thinking 0 -rope-base 10000 -o "%W%\databases\tinyllama_1.1b\lz4ht.db" -f
    if %errorlevel% NEQ 0 (
        echo [FAIL] TinyLlama-1.1B
        exit /b 1
    )
    echo OK: TinyLlama-1.1B
)

REM ====== Phi-3-mini ======
if exist "%W%\phi3_mini\phi3_mini.bin" (
    if not exist "%W%\databases\phi3_mini" mkdir "%W%\databases\phi3_mini"
    echo  Building Phi-3-mini ...
    "%BUILD_DB%" "%W%\phi3_mini\phi3_mini.bin" -thinking 0 -rope-base 10000 -o "%W%\databases\phi3_mini\lz4ht.db" -f
    if %errorlevel% NEQ 0 (
        echo [FAIL] Phi-3-mini
        exit /b 1
    )
    echo OK: Phi-3-mini
)

REM ====== MiniCPM5-1B ======
if exist "%W%\minicpm5_1b\minicpm5_1b.bin" (
    if not exist "%W%\databases\minicpm5_1b" mkdir "%W%\databases\minicpm5_1b"
    echo  Building MiniCPM5-1B ...
    "%BUILD_DB%" "%W%\minicpm5_1b\minicpm5_1b.bin" -thinking 0 -rope-base 5000000 -o "%W%\databases\minicpm5_1b\lz4ht.db" -f
    if %errorlevel% NEQ 0 (
        echo [FAIL] MiniCPM5-1B
        exit /b 1
    )
    echo OK: MiniCPM5-1B
)

echo.
echo ============================================================================
echo  ALL TEXT MODEL DATABASES BUILT
echo ============================================================================