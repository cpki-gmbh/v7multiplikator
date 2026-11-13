@echo off
setlocal
set THIS_DIR=%~dp0..\
rem probe: how does cmd+CRT parse the reformulate sandwich prompt? echo_args prints argv verbatim
echo ===probe 1: sandwich WITH escaped quote tokens (user spec form)===
"%THIS_DIR%bin\echo_args.exe" ask "reformulate this comment in a assembly code to only describe behaviour, and to not contain \"(\" \"--\" \")\" now the comment:>>>no -- save area was overwritten<<<reformulate this comment in a assembly code to only describe behaviour, and to not contain \"(\" \"--\" \")\""
