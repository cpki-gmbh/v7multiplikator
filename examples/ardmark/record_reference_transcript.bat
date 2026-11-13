@echo off
REM record_reference_transcript.bat -- (re)record examples/ardmark/reference_transcript.txt
REM
REM Runs the 3-turn ardmark demo driver in -record mode (greedy sampling,
REM deterministic per binary + DB) and writes the demo-intactness reference
REM the verify_server_and_rag.bat [ardmark] gate compares against.
REM EYEBALL the three turns (Read the reference) before committing it: the
REM reference is only as good as the run it recorded.
REM
REM Run from the project root.

bin\test_ardmark_rag.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db 1 -record
if %errorlevel% neq 0 (
    echo [record_reference] FAILED with code %errorlevel% -- reference NOT updated correctly
    exit /b 1
)
echo [record_reference] reference written -- read examples\ardmark\reference_transcript.txt before committing.
exit /b 0
