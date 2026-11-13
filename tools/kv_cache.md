# kv_cache

KV-cache persistence test: saves the used prefix of a live session to disk
(LZ4-compressed), reloads it into a fresh process/context, and checks the
continued session is indistinguishable from the uninterrupted one.

## Run

```bat
bin\kv_cache.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db
```

Takes `<dll_path> <db_path>` positionals.

## What it proves

- `save_kv_cache` / `load_kv_cache` round-trip the used prefix
  (tokens, positions, RoPE state, M-RoPE counters) byte-exactly.
- A wrong-model file is rejected by the header checks before any engine
  state is touched.
- On failure the partial file is deleted so a later load can never accept a
  truncated blob.
