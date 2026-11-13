@echo off
REM ardmark-server.bat -- one-command Kingdom of Ardmark demo.
REM
REM Starts the v7 server (own window) and loads the Ardmark
REM RAG sidecar, so the demo is a single command. After this bat prints
REM "demo ready", every chat turn answers Ardmark questions from the
REM experiment corpus -- watch the server window for the banner-labelled
REM pipeline output (QUERY / CORPUS ANSWER / INJECTED).
REM
REM Stop the demo with stop_server.bat.

setlocal
set THIS_DIR=%~dp0..\
set SIDECAR=examples/ardmark/ardmark_rag.logo
REM ---- point the server at the Qwen3-0.6B database (0.6B demo model) ----
set V7_SERVER_DB=%THIS_DIR%weights\databases\qwen3_1.7b\lz4ht.db
set V7_SERVER_MODEL=qwen3-0.6b
REM ---- demo responsiveness: priority 2 (highest above-normal) ----
REM The 0.6B demo model is the showcase: generation at prio 2 keeps the
REM storyteller snappy on a busy desktop (the plain server.bat default is
REM -15 idle). 2 = THREAD_PRIORITY_HIGHEST, valid for a NORMAL-class
REM process (MSDN SetThreadPriority: 3..7 needs REALTIME_PRIORITY_CLASS).
REM Read by server.bat and mapped onto the exe's -prio flag.
set V7_SERVER_PRIO=2

REM ---- fail fast with a friendly hint when the 0.6B database is missing ----
if not exist "%V7_SERVER_DB%" (
    echo [ardmark-server] the 0.6B database is missing:
    echo [ardmark-server]   %V7_SERVER_DB%
    echo [ardmark-server] build it first:
    echo [ardmark-server]   fetch_qwen3_0.6b.bat            -- fetches the checkpoint into weights\qwen3_0.6b
    echo [ardmark-server]   build_text_model_databases.bat  -- builds every model database
    echo [ardmark-server] then start this demo again.
    exit /b 1
)

REM ---- start the server in its own visible window ----
call "%~dp0start_server.bat"

REM ---- wait for health (DB decompress takes minutes on a cold start; the
REM      1.8 GB qwen3_1.7b DB decompresses slower than the 0.6B one, so the
REM      budget is 10 minutes) ----
set /a tries=0
:wait_health
set /a tries+=1
curl -s -o nul http://localhost:8080/v1/health
if %errorlevel%==0 goto server_up
if %tries% GEQ 120 goto no_server
REM timeout needs a console stdin; under a redirected shell it exits
REM immediately -- the ping fallback keeps the 5 s cadence.
timeout /t 5 /nobreak >nul 2>nul
if %errorlevel% neq 0 ping -n 6 127.0.0.1 >nul
goto wait_health

:no_server
echo [ardmark-server] FAIL: server did not come up within 10 minutes
exit /b 1

:server_up
REM ---- load the Ardmark RAG sidecar into the running server ----
call "%~dp0server_set_sidecar.bat" %SIDECAR%
echo [ardmark-server] demo ready -- open http://localhost:8080 or POST /v1/chat/completions
exit /b 0
