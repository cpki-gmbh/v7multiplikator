# v7 Examples

Every program here is a self-contained example of using **v7 as a library**:
load `v7_threading.dll`, open a model database, run inference. The minimal
call sequence is always the same four steps (the inference context lives in
BSS inside the DLL — no context pointer is ever handed out):

```c
v7_load("bin/v7_threading.dll", &dll);              /* 1. load the DLL     */
dll.v7_init("weights/databases/qwen3_0.6b/lz4ht.db"); /* 2. open the DB    */
dll.v7_initialize_inference();                      /* 3. create state     */
dll.v7_chat("what is a cat?", 0, out, sizeof(out), 1); /* 4. generate      */
```

Build everything first: run `build.bat` in the project root (outputs go to
`bin\`, the rag tool to `examples\rag\`). Model databases are built with
`build_db.exe` (see `docs/getting_started.md`).

## Topics

| Directory | What to look at | Programs |
|-----------|-----------------|---------|
| `c_inference/` | C programs over the public API: text LLM + vision (VLM) — one subfolder per program, each with its own README | `inference_example` (start here), `dbg_inference` (pure-ASM consumer), `multi_turn_demo`, `test_multiturn` (+`_manual`), `v7_pipe`, `dbg_image_recog`, `test_image_multiturn`, `vit_bbox_eval`, `test_engine_death`, `poem_continuation`, `test_kv_cache` |
| `server/` | OpenAI-compatible HTTP server | `v7_server_main.cpp` + split `v7_server_*.cpp` (`/v1/chat/completions`, browser chat, sidecar loading) |
| `sectorlogo/` | Scripting the engine | `run_sectorlogo.c` (run a `.logo` sidecar standalone), `c_callback_example.c` (host FFI: export a C function the script calls), `test_v7_execute_fmt.c`, `echo_args.c`, `console_demo.logo` (ANSI console tour: colors, cursor, keyboard) |
| `rag/` | Retrieval tools used by sidecars | `bm25_rag.cpp` (BM25 retrieval: `search "<query>" "<opts>"` / `remember "<fact>" "<opts>"`), `llm_text_processor.cpp` (v7-LLM text reformulation; `-distill` mode is the memory distiller), `bayes_classify.cpp`, `web_ragg/` (internet RAG: search, fetch, distill), `memory/` (the persistent-memory sidecar experiment: `test_memory_rag.bat`) |
| `research/` | Archived experiments (read-only history) | sidecar strategy variants (`sidecar_*.logo`), `bible/`, `herbivorous/`, `herbivorous_cats/`, `helion_pump/` — each with its own knowledge base; the Ardmark prime demo lives at `ardmark\` one level up (see the quick reference below) |

## The main API functions are:

| API | Purpose |
|-----|---------|
| `v7_chat` | Chat: tokenize -> prefill -> generate -> decode. Multi-turn via `reset=0`. |
| `v7_generate` | Single-turn generation on a separate KV cache (`raw_generation=0` renders the turn-1 chat template; `1` = raw continuation). |
| `v7_chat_image` | Image + prompt -> caption/QA (the VLM path). |

Sampling defaults are greedy (temperature 0) and owned by the DLL; configure
them with the `v7_set_sampler_*` setters (see `docs/native_api.md` for the full
export surface, including `v7_set_sidecar` for sidecar hooks).

## Sidecars (SectorLogo)

A sidecar is a small script (`examples/rag/memory/sidecar_memory_rag_v6.logo`)
that hooks token generation: it can read the stream (`GetStream`), rewrite it
(`SetStream`, which triggers KV-cache rollback and re-prefill), call external
tools through `DefineCommand`, and call the host through
`CallLibrary [ "SELF" ... ]`. Load one with `inference_example.exe ... -script
<path> -rewrite`.

## Experiment runner quick reference

```bat
examples\rag\memory\test_memory_rag.bat                 (persistent memory)
examples\research\herbivorous_cats\run.bat              (false-belief override)
examples\research\helion_pump\run.bat                   (fictional-fact injection)
examples\ardmark\run.bat                                 (fictional history, the prime demo)
```

All runners expect the project root as the working directory.
