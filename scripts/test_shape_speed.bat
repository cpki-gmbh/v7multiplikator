@echo off
rem test_shape_speed.bat -- benchmark one matmul shape under the INI-committed
rem dispatch config (params + pf distances live-applied before the run).
rem Usage: test_shape_speed.bat <shape NxM> <megabytes> <runtime_seconds> [--f32]
rem Example: test_shape_speed.bat 1024x1024 16 3
setlocal
cd /d %~dp0..\
tools\test_shape_speed.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --ini threading_proxy.ini %*
exit /b %errorlevel%