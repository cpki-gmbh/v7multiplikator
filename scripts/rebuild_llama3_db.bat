@echo off
REM rebuild_llama3_db.bat -- Rebuild Llama 3.2 1B database from checkpoint + current template files
REM
REM Usage:  rebuild_llama3_db.bat
REM
REM Builds: weights\databases\llama_3.2_1b\lz4ht.db  (scaled RoPE from config, -thinking 0)
setlocal
set THIS_DIR=%~dp0..\
"%THIS_DIR%bin\build_db.exe" "%THIS_DIR%weights\llama_3.2_1b\llama_3.2_1b.bin" -thinking 0 -o "%THIS_DIR%weights\databases\llama_3.2_1b\lz4ht.db" -f
endlocal
