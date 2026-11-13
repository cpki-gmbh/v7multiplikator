@echo off
setlocal EnableDelayedExpansion
REM build_web_ragg.bat -- build the internet-RAG tool (web search -> 5 pages
REM -> v7 distillation). Sources copied from demo_agent, see webtool.md.
cd /d %~dp0

REM Locate VS via vswhere (any edition; set VS_PATH to override):
if not defined VS_PATH (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    set "VS_ROOT="
    for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_ROOT=%%i"
    set "VS_PATH=!VS_ROOT!\VC\Auxiliary\Build\vcvarsall.bat"
)
call "!VS_PATH!" x64
if errorlevel 1 (
    echo Failed to load vcvarsall
    exit /b 1
)

set FLAGS=/O2 /EHsc /W3 /D_CRT_SECURE_NO_WARNINGS
set INC=/I. /I..\..\.. /I..\..\..\src /I..\..\..\tools
set LIBS=kernel32.lib user32.lib winhttp.lib

echo === Building web_ragg.exe ===
cl.exe %FLAGS% %INC% /Fe:web_ragg.exe web_ragg.cpp webtool.cpp search.cpp http_client.cpp html_parser.cpp text_utils.cpp simpledb.c text_shortener.cpp text_summarizer.cpp %LIBS%
if errorlevel 1 (
    echo web_ragg.exe build failed!
    exit /b 1
)

echo.
echo Build successful. Usage:
echo   web_ragg.exe search "query"
echo   web_ragg.exe makequery "think text"
