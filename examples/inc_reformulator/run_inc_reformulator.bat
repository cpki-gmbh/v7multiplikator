@echo off
setlocal
cd /d %~dp0..\..
rem run_inc_reformulator.bat [target.inc] -- requires the v7 server running
rem (start_server.bat). Reformulates " -- " comments through the server and
rem edits them in place; the log lands next to the script.
if "%V7_SERVER_URL%"=="" set V7_SERVER_URL=http://127.0.0.1:8080
if "%~1"=="" (
    python "%~dp0inc_reformulator.py"
) else (
    if "%~2"=="" (
        python "%~dp0inc_reformulator.py" "%~1"
    ) else (
        python "%~dp0inc_reformulator.py" "%~1" "%~2"
    )
)
endlocal
