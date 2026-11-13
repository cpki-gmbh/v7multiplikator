# test_multiturn_manual

The same three turns as `test_multiturn`, but driven MANUALLY: the host
calls encode + forward + sample itself, one token at a time, instead of
`v7_chat`. This is the differential-isolation tool: when the automatic chat
loop misbehaves, running the identical token stream through the manual path
splits the bug between the chat loop and the engine's forward machinery.

## Run

```bat
bin\test_multiturn_manual.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db
```

Takes `<dll_path> <db_path>` positionals.

## How to use it in a bug hunt

1. Run `test_multiturn` (automatic) and this program on the same DB.
2. Compare the generated token IDs per turn.
3. Identical output = the bug is in the chat loop's bookkeeping; divergent
   output = the bug is below the loop (prefill/forward/sampler).
