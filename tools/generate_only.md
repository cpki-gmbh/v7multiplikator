# generate_only

Minimal `v7_generate` smoke test: one raw generation round on a fresh
context.

## Run

```bat
bin\generate_only.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db
```

Takes `<dll_path> <db_path>` positionals.

## Notes

- The prompt is rendered through the model's turn-1 template unless
  `raw_generation=1` is passed to `v7_generate` -- this program uses the
  default (templated) path; see `poem_continuation` for the raw variant.
- Useful as the smallest reproduction when generate-path behavior differs
  from chat-path behavior: no multi-turn state, no follow-up turn.
