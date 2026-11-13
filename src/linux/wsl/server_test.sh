#!/usr/bin/env bash
set -u
ROOT="${V7_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
B="$ROOT/bin-linux"
L="$B/libv7_threading.so"
DB="$ROOT/weights/databases/qwen3_0.6b/lz4ht.db"
"$B/v7_server" "$L" "$DB" 8080 qwen3-0.6b >/tmp/v7_server.log 2>&1 &
srv=$!
sleep 4
echo "=== health ==="
curl -s http://127.0.0.1:8080/v1/health; echo
echo "=== models ==="
curl -s http://127.0.0.1:8080/v1/models; echo
echo "=== chat ==="
curl -s -H 'Content-Type: application/json' --data '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32,"temperature":0}' http://127.0.0.1:8080/v1/chat/completions; echo
echo "=== quit ==="
curl -s -H 'X-V7-Local: 1' 'http://127.0.0.1:8080/quit?code=0'; echo
wait $srv
echo "server rc=$?"
echo "=== server log tail ==="
tail -n 15 /tmp/v7_server.log
