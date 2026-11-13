@echo off
REM stop_server.bat -- cleanly stop the v7_server via its /quit endpoint.
REM
REM Usage: stop_server.bat [port]
REM   [port]  optional, default 8080 (the port server.bat starts it on).
REM
REM Calls /quit?code=42. The server exits with code 42.
REM The window shows the exit and stays open (no restart loop).

setlocal
REM %~1 strips surrounding quotes so a quoted port argument parses cleanly.
set PORT=%~1
if "%PORT%"=="" set PORT=8080
echo [stop_server.bat] calling /quit?code=42 on port %PORT%...
curl -s "http://localhost:%PORT%/quit?code=42"
echo.
endlocal
