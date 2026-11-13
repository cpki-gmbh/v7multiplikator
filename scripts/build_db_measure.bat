@echo off
rem ============================================================================
rem build_db_measure.bat -- rebuild the qwen3_0.6b database for measurement runs
rem
rem -tuner-delay 999999 seeds tuner:delaydefault so the in-process tuner's A/B
rem pairs are spaced beyond any measurement run: the tuner stays present and
rem initialized, but never races a candidate while a benchmark or a RAG turn is
rem being measured. Used ONLY for the paper-measurement session; verify.bat
rem rebuilds the database with its own default settings afterwards.
rem ============================================================================
cd /d %~dp0..\
bin\build_db.exe weights\qwen3_0.6b\qwen3_0.6b.bin -o weights\databases\qwen3_0.6b\lz4ht.db -tuner-delay 999999
if %errorlevel% NEQ 0 (
  echo DB_BUILD_FAILED
  exit /b 1
)
echo DB_BUILD_OK
exit /b 0
