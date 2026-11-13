@echo off
REM build_transplant_ini.bat -- compile the population transplant tool.
REM Run from a Visual Studio Developer Command Prompt (cl.exe on PATH).
cl /nologo /O2 /W3 /Fe:bin\transplant_ini.exe tools\transplant_ini.c
if %errorlevel% NEQ 0 (
    echo BUILD FAILED
    exit /b 1
)
echo OK: bin\transplant_ini.exe
