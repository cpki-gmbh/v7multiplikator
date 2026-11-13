@echo off
setlocal
cd /d %~dp0..\
rem rebuild_qwen3_db.bat -- rebuild ONLY the qwen3_0.6b database. Use after any
rem change that lands in the DB (stop-token/template changes, tuner scripts,
rem INI seeding) so the runtime picks it up on next engine start.
bin\build_db.exe weights\qwen3_0.6b\qwen3_0.6b.bin -o weights\databases\qwen3_0.6b\lz4ht.db -f -tuner-rand-delay 1 -tuner-delay 1
set BUILD_RC=%errorlevel%
if not "%BUILD_RC%"=="0" exit /b %BUILD_RC%
echo rebuilt: weights\databases\qwen3_0.6b\lz4ht.db
endlocal
exit /b 0
