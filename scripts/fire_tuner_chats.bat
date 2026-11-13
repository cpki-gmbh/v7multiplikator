@echo off
REM fire_tuner_chats.bat -- one curl chat turn against the running server,
REM varied prompt per turn index so the conversations differ.
setlocal enabledelayedexpansion
set /a N=%1
if "%N%"=="" set /a N=1
set /a i=1
set PROMPTS[1]=What is the capital of France?
set PROMPTS[2]=Tell me about dogs.
set PROMPTS[3]=Explain quantum physics briefly.
set PROMPTS[4]=What is a bird?
set PROMPTS[5]=Describe the ocean.
set PROMPTS[6]=Why is the sky blue?
set PROMPTS[7]=What makes a good story?
set PROMPTS[8]=Tell me about cats.
:loop
set /a idx=(i - 1) %% 8 + 1
curl.exe -s -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"!PROMPTS[%idx%]!\"}],\"max_tokens\":40}" >nul 2>&1
if !errorlevel! NEQ 0 (
    echo TURN !i!: FAILED
    exit /b 1
)
echo TURN !i!: ok
set /a i+=1
if !i! leq %N% goto loop
echo done: %N% chats
endlocal
