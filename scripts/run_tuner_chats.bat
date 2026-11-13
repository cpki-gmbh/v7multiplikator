@echo off
REM run_tuner_chats.bat -- dozens of chat turns against the tuner DB so the
REM in-process cycle sorter accumulates episodes. Each chat is a fresh
REM inference context; populations persist in v7_tune_param_<n>_<d> files.
setlocal enabledelayedexpansion
set DLL=bin\v7_threading.dll
set DB=weights\databases\qwen3_0.6b\lz4ht_tuner.db
set /a TURNS=%1
if "%TURNS%"=="" set /a TURNS=40
set /a i=1
:loop
bin\inference_example.exe %DLL% %DB% -i "What is the capital of France?" -n 40 -q8 1 >nul 2>&1
if %errorlevel% NEQ 0 (
    echo TURN !i!: FAILED exit code
    exit /b 1
)
echo TURN !i!: ok
set /a i+=1
if !i! leq %TURNS% goto loop
echo done: %TURNS% chats
endlocal
