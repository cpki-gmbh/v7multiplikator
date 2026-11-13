@echo off
REM test_all_models.bat -- Inference unit test for ALL supported model databases
REM
REM Tests every lz4ht.db in weights\databases\ with 3 questions, both Q8 and F32.
REM Uses inference_example.exe (pure database path, no .bin checkpoint needed).
REM
REM Usage:  test_all_models.bat
REM Output: build_logs\test_all_models\<model>_<q#>_<mode>.txt

setlocal ENABLEDELAYEDEXPANSION

set THIS_DIR=%~dp0..\
set BIN=%THIS_DIR%bin
set DLL=%BIN%\v7_threading.dll
set DB_ROOT=%THIS_DIR%weights\databases
set OUT_DIR=%THIS_DIR%build_logs\test_all_models
set HELLO=%BIN%\inference_example.exe

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

if not exist "%DLL%" (
    echo [FATAL] v7_threading.dll not found at %DLL% -- run build.bat first
    exit /b 1
)

set PASS=0
set FAIL=0

echo ============================================================================
echo  v7 Model Inference Test Suite
echo  3 questions x 2 modes (Q8 + F32) per model
echo ============================================================================

for /d %%M in ("%DB_ROOT%\*") do (
    if exist "%%M\lz4ht.db" (
        call :test_model "%%~nM" "%%M\lz4ht.db"
    )
)

echo ============================================================================
echo  DONE: !PASS! passed, !FAIL! failed
echo  Results in: %OUT_DIR%
echo ============================================================================
if !FAIL! GTR 0 (
    echo  RESULT: FAILED
    endlocal
    exit /b 1
)
echo  RESULT: all models passed
endlocal
exit /b 0

:test_model
set MODEL=%~1
set DB=%~2
echo.
echo --- !MODEL! ---

REM Question 1: Capital of France
call :run "!MODEL!" "!DB!" 1 "What is the capital of France?"

REM Question 2: What is a cat
call :run "!MODEL!" "!DB!" 2 "What is a cat?"

REM Question 3: Hello world Python
call :run "!MODEL!" "!DB!" 3 "Write a hello world in Python."
goto :eof

:run
set M=%~1
set D=%~2
set QN=%~3
set QTEXT=%~4

REM Q8 mode
set OUT=%OUT_DIR%\%M%_q%QN%_q8.txt
"%HELLO%" "%DLL%" "%D%" -q8 -i "%QTEXT%" > "%OUT%" 2>&1
if !errorlevel! equ 0 (
    echo   [Q%QN% Q8] OK
    set /a PASS+=1
) else (
    echo   [Q%QN% Q8] FAIL
    set /a FAIL+=1
)

REM F32 mode
set OUT=%OUT_DIR%\%M%_q%QN%_f32.txt
"%HELLO%" "%DLL%" "%D%" -i "%QTEXT%" > "%OUT%" 2>&1
if !errorlevel! equ 0 (
    echo   [Q%QN% F32] OK
    set /a PASS+=1
) else (
    echo   [Q%QN% F32] FAIL
    set /a FAIL+=1
)
goto :eof
