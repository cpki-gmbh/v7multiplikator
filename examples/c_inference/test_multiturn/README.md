# test_multiturn

The 3-turn multi-turn regression test (cat / dog / bird): thinking ON, Q8,
greedy. Each turn's answer must reference the previous turns' subjects --
this is the standing repro harness for the Sec.150-class multi-turn
corruption bugs (spurious think tags, endoftext leakage, position drift).

## Run

```bat
bin\test_multiturn.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db
```

Takes `<dll_path> <db_path>` positionals.

## Notes

- Turns use `reset=0`: one KV cache, positions continue across turns.
- If this test ever produces garbage mid-turn 2+, suspect prefill position
  or M-RoPE state first (see AGENTS.md Sec.150 for the diagnostic playbook:
  token dump first, then K/V probes, then differential isolation).
