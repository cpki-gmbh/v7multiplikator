@echo off
setlocal
REM run_vision_smoke.bat -- vision smoke: one Qwen3-VL run whose output must be
REM byte-identical to vision\vision_baseline.txt (the Sec.122 change detector).
REM The tool's "Time: <ms>" line is wall-clock and varies every run, so it is
REM filtered out of the capture; everything else is compared verbatim.
REM A MISMATCH means the vision path changed: check whether the answer improved
REM or degraded (a precision improvement is an accepted change -- re-baseline
REM it deliberately). Exit codes are the contract: 0 = match, 1 = mismatch,
REM 2 = environmental SKIP (DLL/DB/baseline absent -- a caller MUST read this
REM code, not string-match the SKIP prose).
rem run the vision inference at idle priority so it never competes with
rem foreground work (the engine applies v7_override_prio at pool creation)
set v7_override_prio=-15
cd /d %~dp0..\

if not exist bin\v7_threading.dll (
    echo [SKIP: no DLL] run build.bat first
    exit /b 2
)
if not exist weights\databases\qwen3_vl_4b\lz4ht.db (
    echo [SKIP: DB not built] weights\databases\qwen3_vl_4b\lz4ht.db missing -- run build_vit_databases.bat
    exit /b 2
)
if not exist vision\vision_baseline.txt (
    echo [SKIP: no baseline] vision\vision_baseline.txt missing -- the change detector cannot run
    exit /b 2
)

bin\dbg_image_recog.exe bin\v7_threading.dll weights\databases\qwen3_vl_4b\lz4ht.db cat640.raw "What is in this image?" > build_logs\_vision_raw.txt 2>&1
set "IMAGE_RC=%errorlevel%"
REM Keep ONLY the deterministic lines: the fixed header, the model answer, and
REM the tuner lifetime counters. The worker/threads banner and the wall-clock
REM "Time:" line carry thread ids, handle addresses, and worker start
REM interleaving that vary every run, so they are excluded by construction.
REM The "journal: applied N changes" line is also excluded: the journal is
REM cumulative tuning memory that the tuner keeps appending to, so N grows
REM across runs and is not a vision-path signal.
findstr /b /c:"dll:" /c:"loading:" /c:"question:" /c:"mode:" /c:"db_load:" /c:"Dbg:" /c:"SectorLogo:" /c:"tuner:" /c:"This image" /c:"tuner:" build_logs\_vision_raw.txt > build_logs\_vision_current.txt
echo [run_vision_smoke] current captured: build_logs\_vision_current.txt

set "FC_RC=0"
fc /b vision\vision_baseline.txt build_logs\_vision_current.txt > nul
if errorlevel 1 set "FC_RC=1"
set "VSMOKE_RC=1"
if "%IMAGE_RC%"=="0" if "%FC_RC%"=="0" set "VSMOKE_RC=0"
if "%VSMOKE_RC%"=="0" (
    echo [OK] vision output matches baseline
) else (
    echo [MISMATCH] vision inference failed or output differs from vision\vision_baseline.txt
)

REM NEVER REMOVE THIS TYPE COMMAND NEVER!
type build_logs\_vision_current.txt

exit /b %VSMOKE_RC%
