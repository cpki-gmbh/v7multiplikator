@echo off
REM test_cat.bat -- greedy "what is a cat?" probe on the qwen3_0.6b DB, F32 + Q8.
REM Regression probe: the answer must NOT claim cats were domesticated in Europe
REM (correct: Near East / Fertile Crescent). Run from the project ROOT.
REM Output goes to the console, unfiltered.

setlocal
cd /d %~dp0..\

REM The probe tool was renamed across the tree history (hello_inference ->
REM inference_example): use whichever binary this tree's build produces.
set PROBE=bin\hello_inference.exe
if exist bin\inference_example.exe set PROBE=bin\inference_example.exe

echo.
echo ============ cat probe: F32 ============
%PROBE% bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "what is a cat?" -n 1000 2>&1

echo.
echo ============ cat probe: Q8 ============
%PROBE% bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "what is a cat?" -n 1000 -q8 2>&1

endlocal
