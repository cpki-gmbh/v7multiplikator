@echo off
REM modelchat.bat -- one-shot memory-RAG chat against Qwen3-0.6B.
REM
REM Usage: modelchat.bat "your message here" [extra args]
REM
REM Runs inference_example with the persistent-memory sidecar so the model
REM reads and writes its memory corpus across conversations.

setlocal
set THIS_DIR=%~dp0..\
if "%~1"=="" (
    echo Usage: modelchat.bat "your message here" [extra args]
    echo Example: modelchat.bat "what is a cat?"
    exit /b 1
)
"%THIS_DIR%bin\inference_example.exe" "%THIS_DIR%bin\v7_threading.dll" "%THIS_DIR%weights\databases\qwen3_0.6b\lz4ht.db" -i %* -n 1500 -q8 1 -rewrite -script "%THIS_DIR%examples\rag\memory\sidecar_memory_rag_v6.logo"
