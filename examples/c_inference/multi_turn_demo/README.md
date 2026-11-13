# multi_turn_demo

Multi-turn chat demo: shows `reset=0` (continue the KV cache across turns),
interleaved `v7_generate` calls, and `reset=1` (fresh session) in one run.

## Run

```bat
bin\multi_turn_demo.exe weights\databases\qwen3_0.6b\lz4ht.db
```

The DB path is the only argument (the DLL is loaded by name from the CWD or
PATH; run from `bin\` or set it up there).

## What to watch

- Turn 2 and 3 keep the session (reset=0): the model remembers the earlier
  turns because the KV cache is preserved and positions continue.
- The interleaved `v7_generate` runs on its own file-backed KV cache and does
  not disturb the chat session.
- `reset=1` starts a fresh session: positions restart at 0.
