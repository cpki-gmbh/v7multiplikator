@echo off
rem run_tuner_proof.bat -- drives bin\tuner_proof.exe end to end: real
rem inference chats with the SectorLogo hook armed, so the DB-injected tuner
rem scripts (tune_init / tune_accept / tune_reject) run their A/B loop live,
rem then verifies the accepted configs (tuner:cfg:<shape> DB keys) and the
rem tuning journal records when the flush threshold was reached.
rem Exit 0 = the loop ran end to end.
rem usage: run_tuner_proof.bat [chats] [tokens]
setlocal
set THIS_DIR=%~dp0..\
set BIN=%THIS_DIR%bin
set DB_QWEN3=%THIS_DIR%weights\databases\qwen3_0.6b\lz4ht.db

if not exist "%BIN%\tuner_proof.exe" (
    echo tuner_proof.exe missing -- run build.bat first.
    exit /b 1
)
if not exist "%DB_QWEN3%" (
    echo %DB_QWEN3% missing -- run verify.bat once to build the database.
    exit /b 1
)

"%BIN%\tuner_proof.exe" "%BIN%\v7_threading.dll" "%DB_QWEN3%" %1 %2
set RC=%errorlevel%
if %RC% neq 0 (
    echo tuner_proof FAILED ^(exit %RC%^).
    exit /b %RC%
)
echo tuner_proof passed.
exit /b 0
