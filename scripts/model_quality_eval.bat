@echo off
REM model_quality_eval.bat -- Model quality evaluation across all working model databases
REM
REM Runs 5 difficult/adversarial questions against every lz4ht.db in weights\databases\
REM using inference_example.exe (pure database path, no .bin checkpoint needed) in F32
REM mode (reference-quality path). High token budget (-n 400) lets reasoning models
REM think before answering.
REM
REM The 5 questions:
REM   Q1. Adversarial trick question (Airbus A380 vs birds) -- tests hallucination
REM   Q2. Poetry request (Jesus + ugly people)             -- tests creativity/compliance
REM   Q3. Factual definition (cat)                          -- tests factual grounding
REM   Q4. Logic trap (3 apples, take away 2)                -- tests careful parsing
REM   Q5. Counting trap (r's in strawberry)                 -- tests attention to detail
REM
REM Usage:  model_quality_eval.bat
REM Output: build_logs\model_quality\<model>_q<QN>.txt
REM
REM Working models = every database dir that contains lz4ht.db, EXCEPT models
REM listed in EXCLUDE below (unsupported architectures). The .bin checkpoint
REM files are NOT needed -- inference_example.exe loads everything from microdb.

setlocal ENABLEDELAYEDEXPANSION

set THIS_DIR=%~dp0..\
set BIN=%THIS_DIR%bin
set DB_ROOT=%THIS_DIR%weights\databases
set OUT_DIR=%THIS_DIR%build_logs\model_quality
set HELLO=%BIN%\inference_example.exe
set DLL=%BIN%\v7_threading.dll
set TOKENS=400

REM Unsupported / not-yet-integrated models -- skipped during the loop
set EXCLUDE= nemotron_mini_4b nanbeige_4.1_3b

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

set PASS=0
set FAIL=0

echo ============================================================================
echo  v7 Model Quality Evaluation
echo  5 difficult questions x F32 mode, token budget=%TOKENS%
echo  Excluded models:%EXCLUDE%
echo ============================================================================

for /d %%M in ("%DB_ROOT%\*") do (
    set "SKIP=0"
    for %%X in (%EXCLUDE%) do (
        if "%%~nM"=="%%X" set "SKIP=1"
    )
    if "!SKIP!"=="0" if exist "%%M\lz4ht.db" (
        call :test_model "%%~nM" "%%M\lz4ht.db"
    )
)

echo ============================================================================
echo  DONE: !PASS! passed, !FAIL! failed
echo  Results in: %OUT_DIR%
echo ============================================================================
endlocal
exit /b 0

:test_model
set MODEL=%~1
set DB=%~2
echo.
echo --- !MODEL! ---

REM Q1: Adversarial trick question (Airbus A380 vs birds)
call :run "!MODEL!" "!DB!" 1 "what is the difference between the airbaus a380 and other birds where does it build its nests, how does it have sex?"

REM Q2: Poetry request (Jesus + ugly people)
call :run "!MODEL!" "!DB!" 2 "write a poem about jesus and ugly people!"

REM Q3: Factual definition (cat)
call :run "!MODEL!" "!DB!" 3 "what is a cat?"

REM Q4: Logic trap (3 apples, take away 2)
call :run "!MODEL!" "!DB!" 4 "If there are 3 apples and you take away 2, how many apples do you have?"

REM Q5: Counting trap (r's in strawberry)
call :run "!MODEL!" "!DB!" 5 "How many r's are in the word strawberry?"

goto :eof

:run
set M=%~1
set D=%~2
set QN=%~3
set QTEXT=%~4

set OUT=%OUT_DIR%\%M%_q%QN%.txt
"%HELLO%" "%DLL%" "%D%" -n %TOKENS% -i "%QTEXT%" > "%OUT%" 2>&1
if !errorlevel! equ 0 (echo   [Q%QN%] OK) else (echo   [Q%QN%] FAIL & set /a FAIL+=1)
set /a PASS+=1
goto :eof
