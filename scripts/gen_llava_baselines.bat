@echo off
REM gen_llava_baselines.bat -- regenerate the 6 LLaVA-OneVision anyres baselines
REM (agent_fix item 1). Runs the EXACT command + filter pair the smoke gate
REM (run_vision_smoke_llava.bat) uses, so a baseline and a gate run differ only
REM in wall-clock lines (Time:, db_load:, Dbg: are stripped by both sides).
REM Run this ONLY after an intended output change (engine fix, anyres math
REM change), never to paper over a regression -- a baseline refresh hides a
REM broken vision path.
setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0..\"

set DLL=bin\v7_threading.dll
set DB=weights\databases\llava_onevision_qwen2_0.5b\lz4ht.db

if not exist "%DLL%" (
    echo FATAL: %DLL% missing -- run build.bat first
    exit /b 1
)
if not exist "%DB%" (
    echo FATAL: %DB% missing -- build the LLaVA-OV database first
    exit /b 1
)

for %%M in (0 1) do (
    if %%M EQU 0 (set MODE=f32) else (set MODE=q8)
    for %%I in (cat640 vision\cat_anyres_wide vision\cat_anyres_tall) do (
        echo [gen] %%~nI !MODE! ...
        if not exist build_logs mkdir build_logs
        bin\dbg_image_recog.exe -q8 %%M -n 150 %DLL% %DB% %%I.raw "What is in this image?" > build_logs\_llava_current.txt 2>&1
        if !errorlevel! NEQ 0 (
            echo FATAL: engine exited !errorlevel! for %%~nI !MODE! -- a baseline from a crashed run is garbage
            exit /b 1
        )
        findstr /v /c:"Time:" /c:"db_load:" /c:"Dbg:" build_logs\_llava_current.txt > vision\vision_baseline_llava_%%~nI_!MODE!.txt
        if not exist vision\vision_baseline_llava_%%~nI_!MODE!.txt (
            echo FATAL: baseline for %%~nI !MODE! was not written
            exit /b 1
        )
        type vision\vision_baseline_llava_%%~nI_!MODE!.txt
        echo.
    )
)

echo [OK] 6 LLaVA baselines regenerated in vision\
endlocal
exit /b 0
