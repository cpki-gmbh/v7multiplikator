@echo off
rem shape_bench.bat -- Q8 GB/s for the decode-relevant shapes, one call per shape.
rem b_loadtest accepts only ONE shape argument per invocation.
rem Usage: shape_bench.bat <out.txt>
setlocal
set OUT=%~1
if "%OUT%"=="" set OUT=shape_bench.txt
cd /d %~dp0..\
set DB=weights\databases\qwen3_0.6b\lz4ht.db
tools\b_loadtest.exe bin\v7_threading.dll --db %DB% --q8 1024:1024 --time 2 >> "%OUT%" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db %DB% --q8 2048:1024 --time 2 >> "%OUT%" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db %DB% --q8 1024:2048 --time 2 >> "%OUT%" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db %DB% --q8 1024:3072 --time 2 >> "%OUT%" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db %DB% --q8 3072:1024 --time 2 >> "%OUT%" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db %DB% --q8 151936:1024 --time 2 >> "%OUT%" 2>&1
tools\b_loadtest.exe bin\v7_threading.dll --db %DB% --q8 1024:151936 --time 2 >> "%OUT%" 2>&1
exit /b 0
