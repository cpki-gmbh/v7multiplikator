@echo off
REM test_server_conversation.bat -- test HTML chat and API conversation with KV persistence
REM Starts server in its own window, runs multi-turn tests via curl, kills server at end.

setlocal ENABLEDELAYEDEXPANSION

set THIS_DIR=%~dp0
set ROOT=%THIS_DIR%..
set PORT=8899
set DLL=%ROOT%\bin\v7_threading.dll
set DB=%ROOT%\weights\databases\qwen3_0.6b\lz4ht.db
set EXE=%ROOT%\bin\v7_server.exe
set LOG=%ROOT%\build_logs\server_conversation_test.txt

echo === Starting v7_server.exe on port %PORT% ===
start "v7_server" cmd /c "%EXE% %DLL% %DB% %PORT% qwen3-0.6b"

echo === Waiting for server startup (health check retry, max 120s) ===
set WAITED=0
:waitloop
curl -s -o nul http://localhost:%PORT%/v1/health 2>nul
if %ERRORLEVEL%==0 goto ready
timeout /t 5 /nobreak >nul
set /a WAITED+=5
if %WAITED% LSS 120 goto waitloop
echo WARNING: server did not respond to health check within 120s, continuing anyway
:ready
echo Server ready after ~%WAITED%s

echo ============================================================
echo TEST 1: Health check
echo ============================================================
curl -s http://localhost:%PORT%/v1/health
echo.

echo ============================================================
echo TEST 2: GET / (empty chat page)
echo ============================================================
curl -s http://localhost:%PORT%/ > %LOG%.html_initial 2>&1
echo Saved initial page to %LOG%.html_initial

echo ============================================================
echo TEST 3: HTML chat - Turn 1: "Do you like Paris?"
echo ============================================================
curl -s -X POST http://localhost:%PORT%/chat/submit -d "q=Do+you+like+Paris%3F" > %LOG%.html_turn1 2>&1
echo Saved turn 1 page to %LOG%.html_turn1

echo ============================================================
echo TEST 4: HTML chat - Turn 2: "Do you like London?"
echo ============================================================
curl -s -X POST http://localhost:%PORT%/chat/submit -d "q=Do+you+like+London%3F" > %LOG%.html_turn2 2>&1
echo Saved turn 2 page to %LOG%.html_turn2

echo ============================================================
echo TEST 5: HTML chat - Turn 3: "What was the first city I asked about?"
echo ============================================================
curl -s -X POST http://localhost:%PORT%/chat/submit -d "q=What+was+the+first+city+I+asked+about%3F" > %LOG%.html_turn3 2>&1
echo Saved turn 3 page to %LOG%.html_turn3

echo ============================================================
echo TEST 6: API chat - same 3-turn conversation via OpenAI API
echo ============================================================
curl -s -X POST http://localhost:%PORT%/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Do you like Paris?\"}],\"max_tokens\":1500,\"temperature\":0}"
echo.

curl -s -X POST http://localhost:%PORT%/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Do you like London?\"}],\"max_tokens\":1500,\"temperature\":0}"
echo.

curl -s -X POST http://localhost:%PORT%/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"What was the first city I asked about?\"}],\"max_tokens\":1500,\"temperature\":0}"
echo.

echo ============================================================
echo TEST 7: Reset and verify clean state
echo ============================================================
curl -s http://localhost:%PORT%/chat/reset
echo.

echo ============================================================
echo Killing server ===
taskkill /f /im v7_server.exe >nul 2>&1

echo === Done ===
endlocal
