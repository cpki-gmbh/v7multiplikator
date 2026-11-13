# memory — the persistent-memory RAG experiment

A SectorLogo sidecar system that gives the model a persistent memory across
sessions: every finished conversation turn is distilled into 3 tagged factual
lines (USER:/SELF:/NOTE:) and appended to per-tag corpora; at the next
session start the retrieved memories are injected into the model's think
block ("Things I remember about you...").

## Run it

```bat
examples\rag\memory\test_memory_rag.bat      (from the project root)
```

One command: loads the Qwen3-0.6B DB, attaches `sidecar_memory_rag_v6.logo`,
runs a multi-turn scripted conversation, prints the full sidecar diagnostic
blocks (query, retrieved memories, injected stream, distilled thoughts).

Interactive use with your own prompts:

```bat
modelchat.bat "Hi, I am Alex, I live in Bruges and I like cycling"
modelchat.bat "What do you remember about me?"     (next day, fresh process)
```

## Files

| File | Role |
|------|------|
| `sidecar_memory_rag_v6.logo` | The current memory sidecar (recall + store hooks) |
| `sidecar_memory_rag_v7.logo` | Variant that additionally injects a web_ragg current-events paragraph |
| the earlier memory-sidecar revisions (git history) | Development history of the sidecar (kept for reference) |
| `ragg_base.txt` | The NOTE-tag corpus (general knowledge memories) |
| `user_facts.txt` | The USER-tag corpus (facts about the user) |
| `prepend.txt` / `append.txt` | The distillation prompts (front/back instruction around the turn text) |
| `mem_input.txt` | Scratch: the turn text handed to the distiller |
| `mem_thoughts.txt` | Scratch: last distilled output |
| `rag_flag.txt` | Scratch: sidecar turn-state flag |

Scratch files are rewritten on every run — they are safe to delete.

## How it works

- **Recall hook** (end-think token): query `bm25_rag.exe` over both corpora,
  inject "Things I remember..." lines into the think block, rewrite the stream.
- **Store hook** (EOS): write the finished turn through `v7s_writefile`, call
  the distiller through a `DefineCommand` function wrapping
  `llm_text_processor.exe -distill` with the prepend/append prompts, and route
  each tagged line into its corpus. SELF lines are discarded.

Full design, guards, and measured behavior: `docs/retrieval.md`
(the memory system walkthrough).

## Further reading

- `docs/retrieval.md` — the memory system, annotated
- `examples/README.md` — the top-level examples guide
