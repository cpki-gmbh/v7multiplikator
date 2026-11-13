@echo off
REM _test_server_quick.bat -- quick smoke test for v7_server.exe
REM Opens the server in a new console window, waits, runs curl tests, kills server.

setlocal ENABLEDELAYEDEXPANSION

set THIS_DIR=%~dp0
set ROOT=%THIS_DIR%..

echo === Starting v7_server.exe on port 8899 ===
start "v7_server" cmd /c ""%ROOT%\bin\v7_server.exe" "%ROOT%\bin\v7_threading.dll" "%ROOT%\weights\databases\qwen3_0.6b\lz4ht.db" 8899 qwen3-0.6b"

echo === Waiting 20 seconds for server startup (model load takes time) ===
timeout /t 20 /nobreak >nul

echo === Test 1: GET /v1/health ===
curl -s http://localhost:8899/v1/health
echo.

echo === Test 2: GET /v1/models ===
curl -s http://localhost:8899/v1/models
echo.

echo === Test 3: POST /v1/chat/completions (50 tokens, temp=0) ===
curl -s -X POST http://localhost:8899/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What is a cat?\"}],\"max_tokens\":50,\"temperature\":0}"
echo.

echo === Test 4: POST /v1/chat/reset ===
curl -s -X POST http://localhost:8899/v1/chat/reset
echo.

echo === Test 5: POST /v1/completions ===
curl -s -X POST http://localhost:8899/v1/completions -H "Content-Type: application/json" -d "{\"prompt\":\"What is a cat?\",\"max_tokens\":50,\"temperature\":0}"
echo.

echo === Killing server ===
taskkill /f /im v7_server.exe >nul 2>&1

echo === Done ===
endlocal
