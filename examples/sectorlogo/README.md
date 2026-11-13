# SectorLogo scripting examples

SectorLogo64 is the scripting language built into the DLL. A sidecar script can
read the token stream while the model generates, rewrite it (triggering KV-cache
rollback + re-prefill), reject tokens, spawn external tools, and call back into
the host via FFI.

## Programs

| Program | What it does |
|---------|--------------|
| `run_sectorlogo.c` | Run a `.logo` script standalone (no inference hook, pure script evaluation). |
| `c_callback_example.c` | Host FFI: the C program registers its module with `v7s_set_caller_hmodule`, allowing the script to call exported C functions through `CallLibrary [ "HOST" ... ]`. |
| `test_v7_execute_fmt.c` | Direct `v7_execute_fmt` test: spawn `bin\echo_args.exe` with various formats, verify the captured output. |
| `echo_args.c` | Minimal argv echo helper (prints each argv + raw command line). |

## Scripts

| Script | What it does |
|--------|--------------|
| `console_demo.logo` | ANSI console tour: 16-color palette, truecolor with palette fallback, cursor positioning, `Character`/`Number`, non-blocking `ReadKey`, and `NearestMatch`. It also works when output is redirected to a file. |
| `bm25_bench.logo.template` | Full BM25 benchmark profile loaded at runtime by `bm25_sectorlogo_benchmark.exe`; `%REPS%` is substituted by the host. It compares native `Words` tokenization with the historical per-character loop and uses host slot timers. |
| `bm25_micro.logo.template` | Statement-level tokenization profile: per-character `Part`/`Equal`/`Search`/concatenation, `Split`, `Lowercase`, and the historical list-index loop against one `NGrams` call. |

## Build and run

```bat
build.bat                                                       (from the project root)
bin\run_sectorlogo.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db src\logo\sidecar_cats_to_dogs.logo
bin\run_sectorlogo.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db examples\sectorlogo\console_demo.logo
bin\c_callback_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db
bin\test_v7_execute_fmt.exe bin\v7_threading.dll
```

To hook a sidecar into live inference:

```bat
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "what is a cat?" -n 1500 -q8 1 -rewrite -script src\logo\sidecar_cats_to_dogs.logo
```

## Further reading

- `docs/sectorlogo.md` — the language reference
- the rewrite procedure (dev-only `attic\docs_plans\`) — the stream-rewrite procedure
- `examples/rag/README.md` — RAG sidecars that use these hooks
- `examples/README.md` — the top-level examples guide
