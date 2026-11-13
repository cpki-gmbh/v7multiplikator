@echo off
REM run_autotune.bat -- Run the dispatcher autotuner in hill-climb mode.
REM The tuner reads the Qwen3-0.6B database (the reference DB with the tuner
REM scripts embedded by build_db -tuner src\logo\tuner) and tunes
REM threading_proxy.ini for the 1024x1024 shape.
setlocal
cd /d %~dp0..\
tools\autotune_dispatch.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db 1024:1024 --rounds 60 --runtime 1 --ini threading_proxy.ini
endlocal
