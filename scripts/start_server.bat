@echo off
REM start_server.bat -- start the v7 server in its OWN window.
REM
REM Stops any already-running server first (stop_server.bat), so a restart
REM never races the old instance for port 8080.
REM
REM Usage: start_server.bat [logfile]
REM   [logfile]  optional. Forwarded to server.bat as its OWN argument: with
REM              it the server output is appended to that file; without it
REM              the server output stays VISIBLE in the started window's
REM              console (the default -- output is never suppressed).
REM
REM The window runs server.bat's supervised loop (clean /quit?code=42 stops
REM it; any other exit restarts after 2 s). Stop the server with
REM stop_server.bat.

call "%~dp0..\stop_server.bat"
setlocal
set THIS_DIR=%~dp0..\
REM ---- server configuration (owner: the config lives HERE, in the bat) ----
REM The caller may override any of these via the V7_SERVER_* environment
REM variables BEFORE starting this bat (server.bat reads the same vars, so a
REM caller-set value survives straight through). Defaults below: the Ardmark
REM demo model (qwen3-0.6b) with StreamingLLM enabled.
if "%V7_SERVER_DB%"=="" set V7_SERVER_DB=%THIS_DIR%weights\databases\qwen3_0.6b\lz4ht.db
if "%V7_SERVER_MODEL%"=="" set V7_SERVER_MODEL=qwen3-0.6b
if "%V7_SERVER_PRIO%"=="" set V7_SERVER_PRIO=-15
if "%V7_SERVER_STREAM%"=="" set V7_SERVER_STREAM=1
if "%V7_SERVER_SINKS%"=="" set V7_SERVER_SINKS=4
if "%V7_SERVER_WINDOW%"=="" set V7_SERVER_WINDOW=3000
REM The logfile travels as a plain ARGUMENT to the launched `cmd /k
REM server.bat <path>` tail -- cmd's own `start` parses the tail normally,
REM so a path with spaces survives inside quotes (the old PowerShell
REM Start-Process detour mangled quoted paths, which is why an env-var
REM transport existed; that env var silently redirected the no-argument
REM start -- removed, the console default is always visible).
set "LOG_ARG="
if not "x%~1"=="x" set LOG_ARG="%~1"
REM Wait for the port to actually free before opening the new window: the
REM old process can linger up to a few seconds in teardown, and the fresh
REM server would otherwise burn its 30 bind retries (1 s each, full model
REM reload on every attempt) against a port that frees moments later.
REM Bounded at ~15 s, then we start anyway -- the server's own bind retry
REM loop is the last line of defense. Fixed 8080: the default server.bat
REM port (start_server.bat has no port argument of its own).
set /a WAIT_LOOPS=0
:wait_port_free
netstat -ano | findstr /c:":8080 " | findstr /c:"LISTENING" >nul
if %ERRORLEVEL% NEQ 0 goto wait_port_free_done
set /a WAIT_LOOPS+=1
if %WAIT_LOOPS% GTR 15 (
    echo [start_server.bat] port 8080 still LISTENING after 15 s -- starting anyway.
    goto wait_port_free_done
)
ping -n 2 127.0.0.1 >nul
goto wait_port_free
:wait_port_free_done
REM The extra `cmd /c` layer breaks the console-handle chain: a bare `start`
REM from this script lets the new window inherit OUR stdout handles, and the
REM caller then waits on them (the start call hangs until the server dies).
REM The inner `cmd /c start` detaches the new console cleanly and returns.
cmd /c start "v7 server" /D "%THIS_DIR%" cmd /k "%~dp0server.bat" %LOG_ARG%
echo [start_server.bat] v7_server started in its own window.
echo [start_server.bat] stop it with: stop_server.bat
endlocal
exit /b 0
