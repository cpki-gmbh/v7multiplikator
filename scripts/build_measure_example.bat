@echo off
rem ============================================================================
rem build_measure_example.bat -- P1-5 measurement-only build of inference_example
rem Compiles the example with /DV7_MEASURE into build_logs\ so the measured
rem variant never replaces the normal bin\ tooling. Links Psapi.lib for
rem GetProcessMemoryInfo. NOT part of build.bat / verify.bat.
rem ============================================================================
cd /d %~dp0..\
rem initialize the same VS x64 environment build.bat uses (probed via vswhere)
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_ROOT=%%i"
call "%VS_ROOT%\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
if not exist build_logs\paper_measurements mkdir build_logs\paper_measurements
cl /nologo /O2 /std:c11 /DV7_MEASURE /I src /I examples\c_inference\inference_example ^
  examples\c_inference\inference_example\inference_example.c ^
  /Fe:build_logs\paper_measurements\inference_measure.exe ^
  /Fo:build_logs\paper_measurements\inference_measure.obj ^
  /link Psapi.lib /LIBPATH:src
if %errorlevel% NEQ 0 (
  echo MEASURE BUILD FAILED
  exit /b 1
)
echo MEASURE BUILD OK
exit /b 0
