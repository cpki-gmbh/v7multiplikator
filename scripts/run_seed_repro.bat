@echo off
REM run_seed_repro.bat -- one-turn ardmark primer-seed repro (token dump on)
REM Run from the project root. The full output lands in build_logs\seed_repro.log.
bin\test_ardmark_seed.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db > build_logs\seed_repro.log 2>&1
exit /b %errorlevel%
