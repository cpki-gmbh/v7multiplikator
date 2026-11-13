@echo off
setlocal EnableDelayedExpansion
REM Build from this directory regardless of the caller's working directory.
cd /d "%~dp0"
REM Zydis probe mirrors build.bat: real Zydis in dev, the shipped no-op stub
REM (tools\zydis_stub) when the real library/headers are absent.
set "ROOT=%~dp0.."
set "ZYDIS_INC=%ROOT%\zydis-master\include"
set "ZYCORE_INC=%ROOT%\zycore-c-master\include"
set "ZYDIS_LIB=%ROOT%\Zydis.lib"
set "ZYDIS_LIB_ARG="
set "ZYDIS_PRESENT=0"
if exist "!ZYDIS_INC!\Zydis\Zydis.h" if exist "!ZYDIS_LIB!" set "ZYDIS_PRESENT=1"
if "!ZYDIS_PRESENT!"=="1" goto zydis_ready
set "ZYDIS_INC=%ROOT%\tools\zydis_stub"
set "ZYCORE_INC=%ROOT%\tools\zydis_stub"
goto zydis_done
:zydis_ready
set "ZYDIS_LIB_ARG="!ZYDIS_LIB!""
:zydis_done
REM Locate VS via vswhere (any edition; set VS_PATH to override):
if not defined VS_PATH (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    set "VS_ROOT="
    for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_ROOT=%%i"
    set "VS_PATH=!VS_ROOT!\VC\Auxiliary\Build\vcvarsall.bat"
)
call "!VS_PATH!" x64 >nul 2>&1
if errorlevel 1 (
    echo FAILED
    exit /b 1
)
cl /nologo /O2 /arch:AVX2 /std:c++17 /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /I. /I..\src /I"!ROOT!" /I"!ZYDIS_INC!" /I"!ZYCORE_INC!" mdb_query_key.cpp /Fe:mdb_query_key.exe /link !ZYDIS_LIB_ARG!
if errorlevel 1 (
    echo FAILED
    exit /b 1
)
echo OK
