# chat_only

Two-turn chat smoke test: one fresh turn plus a follow-up turn that must
remember the first. Exercises `v7_init` -> `v7_initialize_inference` ->
`v7_chat(reset=1)` -> `v7_chat(reset=0)`.

## Run

```bat
bin\chat_only.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db
```

Takes `<dll_path> <db_path>` positionals.

## Checks

- The follow-up answer must reference the first turn's subject (KV-cache
  position continuity through the public API).
- Exit code 0 only when both turns produced output.
