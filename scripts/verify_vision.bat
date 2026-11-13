@echo off
REM verify_vision.bat -- vision verification, separate from verify.bat on purpose.
REM verify.bat is the lean development gate (build + validators + text inference
REM + server smoke); the vision gates run here so a dev-loop verify never pays
REM the multi-minute vision inference cost.
REM
REM Thin wrapper only: both smoke scripts carry their own preconditions and
REM their own exit-code contract (0 = pass, 1 = mismatch, 2 = environmental
REM SKIP); this wrapper shows their full output, propagates the codes and
REM fails on any 1. No counters, no sabotage, no self-tests.
setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0..\"

echo [Phase 1/3] Building all components ...
call "%~dp0build.bat"
if !errorlevel! neq 0 (
    echo  FAIL: build failed - vision verification aborted
    exit /b 1
)

echo [Phase 2/3] Qwen3-VL vision smoke (byte-exact baseline compare) ...
call "%~dp0run_vision_smoke.bat"
set VRC=!errorlevel!
if !VRC! EQU 0 (
    echo  PASS: vision smoke matches baseline
) else if !VRC! EQU 2 (
    echo  SKIP: vision smoke - environment not complete, see SKIP line above
) else (
    echo  FAIL: vision smoke - mismatch, see MISMATCH line above
    exit /b 1
)

echo [Phase 3/3] LLaVA-OneVision anyres gate (3 images x F32+Q8) ...
call "%~dp0run_vision_smoke_llava.bat"
set LRC=!errorlevel!
if !LRC! EQU 0 (
    echo  PASS: llava anyres gate - 6/6 match
) else if !LRC! EQU 2 (
    echo  SKIP: llava anyres gate - environment not complete, see SKIP line above
) else (
    echo  FAIL: llava anyres gate - mismatch or crash
    exit /b 1
)

echo.
echo  Vision verification summary: smoke=!VRC! llava=!LRC!
if !VRC! EQU 2 (
    echo  Vision verification incomplete - at least one gate skipped
    exit /b 2
)
if !LRC! EQU 2 (
    echo  Vision verification incomplete - at least one gate skipped
    exit /b 2
)
echo  Vision verification passed
exit /b 0
