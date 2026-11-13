@echo off
REM time_tuner_chats.bat -- measure total wall time of N fresh-process chats
REM against a given DB, one line per chat, summary at the end.
setlocal enabledelayedexpansion
set DLL=bin\v7_threading.dll
set DB=%1
set /a TURNS=%2
if "%TURNS%"=="" set /a TURNS=20
set /a i=1
set /a total=0
:loop
for /f "tokens=2 delims=:." %%a in ('powershell -NoProfile -Command "(Measure-Command { & bin\inference_example.exe %DLL% %DB% -i 'What is the capital of France?' -n 40 -q8 1 ^| Out-Null }).TotalMilliseconds"') do set ms=%%a
echo chat !i!: !ms! ms
set /a total+=ms
set /a i+=1
if !i! leq %TURNS% goto loop
echo TOTAL: !total! ms for %TURNS% chats against %DB%
endlocal
