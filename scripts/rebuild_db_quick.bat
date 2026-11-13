@echo off
REM rebuild_db_quick.bat -- quick single-model database rebuild, no export step.
REM To rebuild a different model, edit the model-path lines below.
cd /d %~dp0..\

if not exist weights\databases\qwen2.5_0.5b mkdir weights\databases\qwen2.5_0.5b
bin\build_db.exe weights\qwen2.5_0.5b\qwen2.5_0.5b.bin -o weights\databases\qwen2.5_0.5b\lz4ht.db -f
if %errorlevel% NEQ 0 exit /b %errorlevel%

if not exist weights\databases\tinyllama_1.1b mkdir weights\databases\tinyllama_1.1b
bin\build_db.exe weights\tinyllama_1.1b\tinyllama_1.1b.bin -thinking 0 -rope-base 10000 -o weights\databases\tinyllama_1.1b\lz4ht.db -f
if %errorlevel% NEQ 0 exit /b %errorlevel%
