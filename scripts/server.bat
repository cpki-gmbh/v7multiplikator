@echo off
REM server.bat -- supervised server loop.
REM
REM Runs v7_server.exe in a restart loop: exit code 42 (clean /quit?code=42)
REM and 43 (the port is owned by another server) stop the loop; ANY other exit
REM (crash, FATAL) restarts the server after 2 s. Without [logfile] the server
REM output stays visible in this window; with it, output is appended to that
REM file.
REM
REM Usage: server.bat [logfile]

setlocal
set THIS_DIR=%~dp0..\
set EXE=%THIS_DIR%bin\v7_server.exe
set DLL=%THIS_DIR%bin\v7_threading.dll
set DB=%THIS_DIR%weights\databases\qwen3_0.6b\lz4ht.db
set PORT=8080
set MODEL=qwen3-0.6b
REM Optional overrides: a caller that sets V7_SERVER_DB / V7_SERVER_MODEL in
REM its environment (e.g. ardmark-server.bat pointing the demo at a bigger
REM model) wins over these defaults; everything else keeps qwen3-0.6b.
if not "%V7_SERVER_DB%"=="" set DB=%V7_SERVER_DB%
if not "%V7_SERVER_MODEL%"=="" set MODEL=%V7_SERVER_MODEL%
REM Logfile: the redirect happens ONLY on an explicit [logfile] argument.
REM No environment fallback: a leftover V7_SERVER_LOG in the caller's env
REM used to silently redirect the window output -- the default is ALWAYS the
REM visible console (Sec.148). %~1 strips surrounding quotes so a quoted
REM path argument never leaks quote characters into the redirect.
set LOG=%~1
if "x%LOG%"=="x" set LOG=-no-log
REM StreamingLLM opt-in passthrough (P2-7): the server.exe takes the CLI
REM flags; these env vars let a launcher compose them without editing the bat.
REM V7_SERVER_STREAM=1 turns streaming on; SINKS/WINDOW default to the exe
REM defaults (4 / 996) when unset.
set STREAM_ARGS=
if not "%V7_SERVER_STREAM%"=="" if not "%V7_SERVER_STREAM%"=="0" set STREAM_ARGS=-stream 1
if not "%V7_SERVER_SINKS%"=="" set STREAM_ARGS=%STREAM_ARGS% -sinks %V7_SERVER_SINKS%
if not "%V7_SERVER_WINDOW%"=="" set STREAM_ARGS=%STREAM_ARGS% -window %V7_SERVER_WINDOW%
REM Worker-priority passthrough: V7_SERVER_PRIO maps onto the exe's -prio flag
REM (-15 idle .. 15 time critical). Unset = the exe default (-15, idle).
set PRIO_ARGS=
if not "%V7_SERVER_PRIO%"=="" set PRIO_ARGS=-prio %V7_SERVER_PRIO%
REM Generation-budget passthrough: V7_SERVER_MAXGEN maps onto the exe's
REM -maxgen flag (the server-wide default budget; 0 = unlimited). Unset = 0.
set MAXGEN_ARGS=
if not "%V7_SERVER_MAXGEN%"=="" set MAXGEN_ARGS=-maxgen %V7_SERVER_MAXGEN%
REM Web-chat history capacity passthrough: V7_SERVER_HISTORY maps onto the
REM exe's -history flag (entries; the oldest user+bot pair is evicted when full).
set HISTORY_ARGS=
if not "%V7_SERVER_HISTORY%"=="" set HISTORY_ARGS=-history %V7_SERVER_HISTORY%
REM consecutive-crash cap: a permanently broken DLL would crash-loop every 2 s
REM forever; 20 consecutive non-clean exits means something is structurally
REM wrong -- stop loudly instead of spinning (a clean /quit stops regardless).
set CRASHES=0
:server_loop
if "%LOG%"=="-no-log" (
    "%EXE%" "%DLL%" "%DB%" %PORT% %MODEL% %PRIO_ARGS% %STREAM_ARGS% %MAXGEN_ARGS% %HISTORY_ARGS%
) else (
    echo === server start %DATE% %TIME% === >> "%LOG%"
    "%EXE%" "%DLL%" "%DB%" %PORT% %MODEL% %PRIO_ARGS% %STREAM_ARGS% %MAXGEN_ARGS% %HISTORY_ARGS% >> "%LOG%" 2>&1
)
REM %ERRORLEVEL% is read outside any parenthesized block, so a negative crash
REM code (0xC0000005 = -1073741819) compares correctly against 42 here.
if %ERRORLEVEL% EQU 42 goto server_stop
if %ERRORLEVEL% EQU 43 goto server_port_busy
set /a CRASHES+=1
if %CRASHES% GTR 20 (
    echo [server.bat] 20 consecutive crashes -- stopping. Fix the DLL/DB and start again.
    endlocal
    exit /b 1
)
echo [server.bat] server exited with code %ERRORLEVEL% -- restarting in 2 s. Clean stop = stop_server.bat.
REM timeout needs a console stdin; under a redirected shell (agent runs, CI)
REM it exits immediately with an error -- the ping fallback keeps the 2 s
REM restart delay intact so a crash loop cannot hammer the model load.
timeout /t 2 /nobreak >nul 2>nul
if %ERRORLEVEL% NEQ 0 ping -n 3 127.0.0.1 >nul
goto server_loop
:server_stop
echo [server.bat] server exited with code 42 -- clean stop, not restarting.
endlocal
exit /b 0

:server_port_busy
echo [server.bat] port is already in use -- another server owns it, not restarting.
endlocal
exit /b 1
