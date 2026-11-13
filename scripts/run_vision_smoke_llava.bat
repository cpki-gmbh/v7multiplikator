@echo off
REM run_vision_smoke_llava.bat -- LLaVA-OneVision anyres gate (agent_fix item 1).
REM Runs dbg_image_recog on the three gate images in F32 and Q8, greedy, and
REM compares the answer text byte-for-byte (minus the wall-clock Time: line)
REM against the shipped baselines in vision\. The wide and tall inputs exercise
REM the width-limited / height-limited fit branches with an ODD 1-pixel pad.
REM
REM Exit codes are the contract (Sec.148-class batch discipline):
REM   0 = a genuine 6/6 byte-exact match
REM   1 = at least one mismatch on a present baseline (the tall-fixture
REM       regression class this gate exists for)
REM   2 = environmental SKIP (DB or baselines absent; nothing ran or only part ran)
REM A caller MUST read this exit code -- string-matching the prose below is the
REM antipattern that lets a partial SKIP masquerade as a full PASS.
setlocal ENABLEDELAYEDEXPANSION
cd /d "%~dp0..\"

set DLL=bin\v7_threading.dll
set DB=weights\databases\llava_onevision_qwen2_0.5b\lz4ht.db
set BASEDIR=vision

if not exist "%DB%" (
    echo SKIP: LLaVA-OV database not built -- weights\databases\llava_onevision_qwen2_0.5b\lz4ht.db missing
    exit /b 2
)
if not exist vision\anyres_baselines_readme.txt (
    echo SKIP: LLaVA-OV baselines not shipped -- vision\anyres_baselines_readme.txt missing
    exit /b 2
)

set FAILS=0
set RAN=0
set SKIPPED=0

for %%M in (0 1) do (
    if %%M EQU 0 (set MODE=f32) else (set MODE=q8)
    for %%I in (cat640 vision\cat_anyres_wide vision\cat_anyres_tall) do (
        set BASE=%BASEDIR%\vision_baseline_llava_%%~nI_!MODE!.txt
        if not exist "!BASE!" (
            echo SKIP: !BASE! missing -- run the baseline generator first
            set /a SKIPPED+=1
        ) else (
            echo [llava-smoke] %%~nI !MODE! ...
            bin\dbg_image_recog.exe -q8 %%M -n 150 %DLL% %DB% %%I.raw "What is in this image?" > build_logs\_llava_current.txt 2>&1
            set "IMAGE_RC=!errorlevel!"
            if !IMAGE_RC! NEQ 0 (
                echo [MISMATCH] llava %%~nI !MODE! -- inference exited !IMAGE_RC!
                type build_logs\_llava_current.txt
                set /a FAILS+=1
            ) else (
                findstr /v /c:"Time:" /c:"db_load:" /c:"Dbg:" build_logs\_llava_current.txt > build_logs\_llava_filtered.txt
                fc /b !BASE! build_logs\_llava_filtered.txt >nul 2>&1
                if errorlevel 1 (
                    echo [MISMATCH] llava %%~nI !MODE! -- output differs from !BASE!
                    type build_logs\_llava_filtered.txt
                    set /a FAILS+=1
                ) else (
                    echo [OK] llava %%~nI !MODE! matches baseline
                    set /a RAN+=1
                )
            )
        )
    )
)

if !FAILS! NEQ 0 (
    echo [FAIL] run_vision_smoke_llava: !FAILS! mismatches, !RAN! ran, !SKIPPED! skipped
    exit /b 1
)
if !RAN! NEQ 6 (
    echo [SKIP] run_vision_smoke_llava: only !RAN! of 6 cases ran, !SKIPPED! skipped
    exit /b 2
)
echo [OK] run_vision_smoke_llava: all 6 cases match baselines
exit /b 0
