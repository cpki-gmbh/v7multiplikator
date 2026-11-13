@echo off
REM verify_llms.bat - Run ALL present text LLMs for "What is the capital of France?"
REM Tests both F32 and Q8 modes. Skips models whose database does not exist.
REM Qwen3-0.6B is MANDATORY -- FATAL if missing.
REM
REM Prerequisites: build.bat + build_text_model_databases.bat must have been run

setlocal ENABLEDELAYEDEXPANSION

rem run all inference at idle priority so the suite never competes with
rem foreground work (the engine applies v7_override_prio at pool creation)
set v7_override_prio=-15

set THIS_DIR=%~dp0..\
set BIN=%THIS_DIR%bin
set DLL=%BIN%\v7_threading.dll
set DB_DIR=%THIS_DIR%weights\databases
set PASS=0
set FAIL=0

if not exist "%DLL%" (
    echo [FATAL] v7_threading.dll not found -- run build.bat first
    exit /b 1
)

echo.
echo ============================================================================
echo  Multi-Model LLM Verification
echo  Prompt: "What is the capital of France?" (both F32 and Q8)
echo  Qwen3-0.6B is MANDATORY
echo ============================================================================
echo.

REM ====== Qwen3-0.6B (MANDATORY) ======
set DB=%DB_DIR%\qwen3_0.6b\lz4ht.db
if not exist "%DB%" (
    echo [FATAL] Qwen3-0.6B database not found at %DB%
    echo  Run build_text_model_databases.bat first.
    exit /b 1
)

echo --- Qwen3-0.6B F32 ---
"%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 > "%BIN%\_vllm.tmp" 2>&1
set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
set "TEXT_RC=!errorlevel!"
set "CASE_OK=0"
if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)

echo --- Qwen3-0.6B Q8 ---
"%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 -q8 > "%BIN%\_vllm.tmp" 2>&1
set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
set "TEXT_RC=!errorlevel!"
set "CASE_OK=0"
if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)

REM ====== Qwen2.5-0.5B ======
set DB=%DB_DIR%\qwen2.5_0.5b\lz4ht.db
if exist "%DB%" (
    echo --- Qwen2.5-0.5B F32 ---
    "%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 > "%BIN%\_vllm.tmp" 2>&1
    set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
    findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
    set "TEXT_RC=!errorlevel!"
    set "CASE_OK=0"
    if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
    if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)
    echo --- Qwen2.5-0.5B Q8 ---
    "%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 -q8 > "%BIN%\_vllm.tmp" 2>&1
    set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
    findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
    set "TEXT_RC=!errorlevel!"
    set "CASE_OK=0"
    if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
    if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)
) else (echo [SKIP] Qwen2.5-0.5B database not found)

REM ====== Qwen3-1.7B ======
set DB=%DB_DIR%\qwen3_1.7b\lz4ht.db
if exist "%DB%" (
    echo --- Qwen3-1.7B F32 ---
    "%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 > "%BIN%\_vllm.tmp" 2>&1
    set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
    findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
    set "TEXT_RC=!errorlevel!"
    set "CASE_OK=0"
    if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
    if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)
    echo --- Qwen3-1.7B Q8 ---
    "%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 -q8 > "%BIN%\_vllm.tmp" 2>&1
    set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
    findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
    set "TEXT_RC=!errorlevel!"
    set "CASE_OK=0"
    if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
    if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)
) else (echo [SKIP] Qwen3-1.7B database not found)

REM ====== Llama-3.2-1B ======
set DB=%DB_DIR%\llama_3.2_1b\lz4ht.db
if exist "%DB%" (
    echo --- Llama-3.2-1B F32 ---
    "%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 > "%BIN%\_vllm.tmp" 2>&1
    set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
    findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
    set "TEXT_RC=!errorlevel!"
    set "CASE_OK=0"
    if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
    if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)
    echo --- Llama-3.2-1B Q8 ---
    "%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 -q8 > "%BIN%\_vllm.tmp" 2>&1
    set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
    findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
    set "TEXT_RC=!errorlevel!"
    set "CASE_OK=0"
    if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
    if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)
) else (echo [SKIP] Llama-3.2-1B database not found)

REM ====== SmolLM2-135M ======
set DB=%DB_DIR%\smollm2_135m\lz4ht.db
if exist "%DB%" (
    echo --- SmolLM2-135M F32 ---
    "%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 > "%BIN%\_vllm.tmp" 2>&1
    set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
    findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
    set "TEXT_RC=!errorlevel!"
    set "CASE_OK=0"
    if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
    if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)
    echo --- SmolLM2-135M Q8 ---
    "%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 -q8 > "%BIN%\_vllm.tmp" 2>&1
    set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
    findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
    set "TEXT_RC=!errorlevel!"
    set "CASE_OK=0"
    if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
    if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)
) else (echo [SKIP] SmolLM2-135M database not found)

REM ====== SmolLM2-360M ======
set DB=%DB_DIR%\smollm2_360m\lz4ht.db
if exist "%DB%" (
    echo --- SmolLM2-360M F32 ---
    "%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 > "%BIN%\_vllm.tmp" 2>&1
    set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
    findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
    set "TEXT_RC=!errorlevel!"
    set "CASE_OK=0"
    if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
    if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)
    echo --- SmolLM2-360M Q8 ---
    "%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 -q8 > "%BIN%\_vllm.tmp" 2>&1
    set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
    findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
    set "TEXT_RC=!errorlevel!"
    set "CASE_OK=0"
    if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
    if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)
) else (echo [SKIP] SmolLM2-360M database not found)

REM ====== TinyLlama-1.1B ======
set DB=%DB_DIR%\tinyllama_1.1b\lz4ht.db
if exist "%DB%" (
    echo --- TinyLlama-1.1B F32 ---
    "%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 > "%BIN%\_vllm.tmp" 2>&1
    set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
    findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
    set "TEXT_RC=!errorlevel!"
    set "CASE_OK=0"
    if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
    if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)
    echo --- TinyLlama-1.1B Q8 ---
    "%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 -q8 > "%BIN%\_vllm.tmp" 2>&1
    set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
    findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
    set "TEXT_RC=!errorlevel!"
    set "CASE_OK=0"
    if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
    if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)
) else (echo [SKIP] TinyLlama-1.1B database not found)

REM ====== Phi-3-mini ======
set DB=%DB_DIR%\phi3_mini\lz4ht.db
if exist "%DB%" (
    echo --- Phi-3-mini F32 ---
    "%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 > "%BIN%\_vllm.tmp" 2>&1
    set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
    findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
    set "TEXT_RC=!errorlevel!"
    set "CASE_OK=0"
    if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
    if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)
    echo --- Phi-3-mini Q8 ---
    "%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 -q8 > "%BIN%\_vllm.tmp" 2>&1
    set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
    findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
    set "TEXT_RC=!errorlevel!"
    set "CASE_OK=0"
    if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
    if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)
) else (echo [SKIP] Phi-3-mini database not found)

REM ====== MiniCPM5-1B ======
set DB=%DB_DIR%\minicpm5_1b\lz4ht.db
if exist "%DB%" (
    echo --- MiniCPM5-1B F32 ---
    "%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 > "%BIN%\_vllm.tmp" 2>&1
    set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
    findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
    set "TEXT_RC=!errorlevel!"
    set "CASE_OK=0"
    if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
    if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)
    echo --- MiniCPM5-1B Q8 ---
    "%BIN%\inference_example.exe" "%DLL%" "%DB%" -i "What is the capital of France?" -n 1000 -q8 > "%BIN%\_vllm.tmp" 2>&1
    set "RUN_RC=!errorlevel!"
type "%BIN%\_vllm.tmp"
    findstr /i paris "%BIN%\_vllm.tmp" >nul 2>&1
    set "TEXT_RC=!errorlevel!"
    set "CASE_OK=0"
    if "!RUN_RC!"=="0" if "!TEXT_RC!"=="0" set "CASE_OK=1"
    if "!CASE_OK!"=="1" (echo PASS & set /a PASS+=1) else (echo FAIL & set /a FAIL+=1)
) else (echo [SKIP] MiniCPM5-1B database not found)

if exist "%BIN%\_vllm.tmp" del "%BIN%\_vllm.tmp" >nul 2>&1
set VLLM_RC=!errorlevel!

echo.
echo ============================================================================
echo  RESULTS: !PASS! passed, !FAIL! failed
echo ============================================================================
if !FAIL! gtr 0 exit /b 1
endlocal
