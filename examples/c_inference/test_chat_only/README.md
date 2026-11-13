# test_chat_only

Two-turn chat smoke test: one fresh turn plus a follow-up turn that must
remember the first. Exercises `v7_init` -> `v7_initialize_inference` ->
`v7_chat(reset=1)` -> `v7_chat(reset=0)`.

## Run

```bat
pushd bin
test_chat_only.exe ..\weights\databases\qwen3_0.6b\lz4ht.db
popd
```

The DB path is the single argument; the DLL is loaded by the fixed name
`v7_threading.dll` (run from `bin\`).

## Checks

- The follow-up answer must reference the first turn's subject (KV-cache
  position continuity through the public API).
- Exit code 0 only when both turns produced output.
