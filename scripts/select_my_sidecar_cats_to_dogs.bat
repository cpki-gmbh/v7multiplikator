@echo off
REM select_my_sidecar_cats_to_dogs.bat -- load the cats-to-dogs sidecar into the RUNNING v7_server.
REM Interface: /sidecar?<path> — the query string IS the path, verbatim.
REM No percent-encoding anywhere = no mangling surface.

setlocal
set PORT=8080
echo [select_my_sidecar_cats_to_dogs.bat] loading sidecar into server on port %PORT%...
curl -s "http://localhost:%PORT%/sidecar?C:/project/v7/src/logo/sidecar_cats_to_dogs.logo"
echo.
endlocal
