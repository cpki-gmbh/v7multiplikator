@echo off
setlocal
cd /d %~dp0..\
set V7_DLL=%~dp0..\bin\v7_threading.dll
set V7_DB=%~dp0..\weights\databases\qwen3_0.6b\lz4ht.db

rem differential probe: the SAME prompt through the chat path (v7_chat, the
rem known-good harness path) and through the generate path (v7_generate raw=0,
rem what v7_pipe ask mode runs). Output unfiltered, both visible.

echo.
echo ============ path A: inference_example = v7_chat ============
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "how are you?" -n 300 -q8 1

echo.
echo ============ path B: v7_pipe ask = v7_generate raw=0 ============
bin\v7_pipe.exe ask "how are you?"

echo.
echo ============ path C: v7_pipe chat = v7_chat ============
bin\v7_pipe.exe chat "how are you?"

endlocal
