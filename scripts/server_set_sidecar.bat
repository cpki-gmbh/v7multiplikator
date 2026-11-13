@echo off
REM server_set_sidecar.bat -- load a SectorLogo sidecar into the RUNNING v7_server.
REM
REM Usage: server_set_sidecar.bat <sidecar_file> [port]
REM   <sidecar_file>  path to the .logo sidecar script (REQUIRED).
REM                   Use FORWARD slashes, no spaces: src/logo/sidecar_cats_to_dogs.logo
REM   [port]          optional, default 8080 (the port server.bat starts it on)
REM
REM Interface: /sidecar?<path> — the query string IS the path, verbatim.
REM Response says "active" (takes effect next turn) or "staged" (a turn already
REM ran with the hook on -- stop_server.bat + start_server.bat to activate).
REM For frequently used sidecars prefer a dedicated select_my_sidecar_*.bat.

setlocal ENABLEDELAYEDEXPANSION
if "%1"=="" (
    echo Usage: server_set_sidecar.bat ^<sidecar_file^> [port]
    echo Example: server_set_sidecar.bat src/logo/sidecar_cats_to_dogs.logo
    exit /b 1
)
set FILE=%1
set PORT=%2
if "%PORT%"=="" set PORT=8080
REM convert backslashes to forward slashes so the query string stays clean
set FILE=!FILE:\=/!
echo [server_set_sidecar.bat] loading sidecar "!FILE!" into server on port %PORT%...
curl -s "http://localhost:%PORT%/sidecar?!FILE!"
echo.
endlocal
