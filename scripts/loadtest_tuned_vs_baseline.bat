@echo off
rem Interleaved loadtest: tuned vs baseline DB for all 8 Qwen3 shapes.
rem Pattern per shape: 10s pause, tuned DB, 10s pause, baseline DB, next shape...
rem Run from project root so relative paths resolve.

set DLL=bin\v7_threading.dll
set DB1=tuning\lz4ht_tuned.db
set DB2=tuning\lz4ht_baseline.db

echo TUNED DB    : %DB1%
echo BASELINE DB : %DB2%
echo Interleaving per shape with 10s pause between every run.

echo --- WARM-UP: first pair twice to settle the system ---
ping -n 11 127.0.0.1 >nul
echo [WARMUP 1/2 1024x512] TUNED
tools\b_loadtest.exe %DLL% --db %DB1% --q8 1024:512 --time 0.5
ping -n 11 127.0.0.1 >nul
echo [WARMUP 1/2 1024x512] BASELINE
tools\b_loadtest.exe %DLL% --db %DB2% --q8 1024:512 --time 0.5
ping -n 11 127.0.0.1 >nul
echo [WARMUP 2/2 1024x512] TUNED
tools\b_loadtest.exe %DLL% --db %DB1% --q8 1024:512 --time 0.5
ping -n 11 127.0.0.1 >nul
echo [WARMUP 2/2 1024x512] BASELINE
tools\b_loadtest.exe %DLL% --db %DB2% --q8 1024:512 --time 0.5
echo --- WARM-UP DONE, starting real measurements ---

ping -n 11 127.0.0.1 >nul
echo [1024x512] TUNED
tools\b_loadtest.exe %DLL% --db %DB1% --q8 1024:512 --time 0.5
ping -n 11 127.0.0.1 >nul
echo [1024x512] BASELINE
tools\b_loadtest.exe %DLL% --db %DB2% --q8 1024:512 --time 0.5

ping -n 11 127.0.0.1 >nul
echo [1024x1024] TUNED
tools\b_loadtest.exe %DLL% --db %DB1% --q8 1024:1024 --time 0.5
ping -n 11 127.0.0.1 >nul
echo [1024x1024] BASELINE
tools\b_loadtest.exe %DLL% --db %DB2% --q8 1024:1024 --time 0.5

ping -n 11 127.0.0.1 >nul
echo [1024x2048] TUNED
tools\b_loadtest.exe %DLL% --db %DB1% --q8 1024:2048 --time 0.5
ping -n 11 127.0.0.1 >nul
echo [1024x2048] BASELINE
tools\b_loadtest.exe %DLL% --db %DB2% --q8 1024:2048 --time 0.5

ping -n 11 127.0.0.1 >nul
echo [2048x1024] TUNED
tools\b_loadtest.exe %DLL% --db %DB1% --q8 2048:1024 --time 0.5
ping -n 11 127.0.0.1 >nul
echo [2048x1024] BASELINE
tools\b_loadtest.exe %DLL% --db %DB2% --q8 2048:1024 --time 0.5

ping -n 11 127.0.0.1 >nul
echo [1024x3072] TUNED
tools\b_loadtest.exe %DLL% --db %DB1% --q8 1024:3072 --time 0.5
ping -n 11 127.0.0.1 >nul
echo [1024x3072] BASELINE
tools\b_loadtest.exe %DLL% --db %DB2% --q8 1024:3072 --time 0.5

ping -n 11 127.0.0.1 >nul
echo [3072x1024] TUNED
tools\b_loadtest.exe %DLL% --db %DB1% --q8 3072:1024 --time 0.5
ping -n 11 127.0.0.1 >nul
echo [3072x1024] BASELINE
tools\b_loadtest.exe %DLL% --db %DB2% --q8 3072:1024 --time 0.5

ping -n 11 127.0.0.1 >nul
echo [151936x1024] TUNED
tools\b_loadtest.exe %DLL% --db %DB1% --q8 151936:1024 --time 0.5
ping -n 11 127.0.0.1 >nul
echo [151936x1024] BASELINE
tools\b_loadtest.exe %DLL% --db %DB2% --q8 151936:1024 --time 0.5

ping -n 11 127.0.0.1 >nul
echo [1024x151936] TUNED
tools\b_loadtest.exe %DLL% --db %DB1% --q8 1024:151936 --time 0.5
ping -n 11 127.0.0.1 >nul
echo [1024x151936] BASELINE
tools\b_loadtest.exe %DLL% --db %DB2% --q8 1024:151936 --time 0.5

echo Done.

