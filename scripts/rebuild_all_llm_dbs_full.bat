@echo off
REM rebuild_all_llm_dbs_full.bat -- refresh every supported LLM database by delegating to the canonical rebuild script.
call "%~dp0..\rebuild_all_llm_dbs.bat" %*
exit /b %errorlevel%
