@echo off
REM rebuild_dbs_phi3_minicpm5_delay100.bat -- rebuild the phi3_mini and
REM minicpm5_1b databases with the tuner delay parameters set to 100.
REM Same flags as build_text_model_databases.bat, except -tuner-delay and
REM -tuner-delay-large are 100 instead of 1.
setlocal

cd /d %~dp0..\

echo Building phi3_mini database (tuner-delay 100, tuner-delay-large 100) ...
bin\build_db.exe weights\phi3_mini\phi3_mini.bin -thinking 0 -rope-base 10000 -o weights\databases\phi3_mini\lz4ht.db -f -tuner-delay 100 -tuner-rand-delay 1 -tuner-delay-large 100 -autoexec autoexec.logo
if %errorlevel% NEQ 0 goto :err

echo Building minicpm5_1b database (tuner-delay 100, tuner-delay-large 100) ...
bin\build_db.exe weights\minicpm5_1b\minicpm5_1b.bin -thinking 0 -rope-base 5000000 -o weights\databases\minicpm5_1b\lz4ht.db -f -tuner-delay 100 -tuner-rand-delay 1 -tuner-delay-large 100 -autoexec autoexec.logo
if %errorlevel% NEQ 0 goto :err

echo Building qwen3_1.7b database (tuner-delay 100, tuner-delay-large 100) ...
bin\build_db.exe weights\qwen3_1.7b\qwen3_1.7b.bin -o weights\databases\qwen3_1.7b\lz4ht.db -f -tuner-delay 100 -tuner-rand-delay 1 -tuner-delay-large 100 -autoexec autoexec.logo
if %errorlevel% NEQ 0 goto :err

echo REBUILD_DONE
exit /b 0

:err
echo REBUILD_FAILED
exit /b 1
endlocal
