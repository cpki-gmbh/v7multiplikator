# RAG (retrieval-augmented generation) tools

Command-line tools that sidecar scripts call through `DefineCommand` to add
retrieval and text-processing behavior: local BM25 retrieval, LLM-based text
distillation, naive-Bayes classification, and internet search.

## Programs

| Program | What it does |
|---------|--------------|
| `bm25_rag.cpp` | BM25 retrieval over a plain-text knowledge base. `search "<query>" "<opts>"` / `remember "<fact>" "<opts>"`. |
| `llm_text_processor.cpp` | v7-LLM text reformulation. The `-distill` mode distills a finished think block into short facts (used by the memory sidecars). Diagnostics go to a log file, never stdout. |
| `bayes_classify.cpp` | Two-class naive Bayes text classifier (`-build` / `-classify`). |
| `web_ragg/` | Internet RAG: DuckDuckGo search -> fetch pages -> v7 distillation into one paragraph. Own build: `build_web_ragg.bat`. |
| `memory/` | The persistent-memory sidecar experiment (`sidecar_memory_rag_v6.logo` + corpora + `test_memory_rag.bat`). |

## Build and run

```bat
build.bat                                                        (bm25_rag, llm_text_processor, bayes_classify)
examples\rag\web_ragg\build_web_ragg.bat                          (web_ragg)

examples\rag\bm25_rag.exe search "what is a cat" "base=examples\rag\memory\ragg_base.txt"
examples\rag\memory\test_memory_rag.bat                           (full memory-RAG E2E)
```

## Sidecar wiring

Sidecars call these tools through `DefineCommand`; the tool's stdout is captured
and injected into the token stream with `SetStream`. The cats-to-dogs example
in `src\logo\sidecar_cats_to_dogs.logo` is the minimal pattern.

## Further reading

- `docs/retrieval.md` — the RAG architecture and BM25 internals
- `docs/internet_rag_dev_log.md` — internet RAG development log
- the `memory/sidecar_*.logo` scripts — thought-distillation implementations
- `examples/README.md` — the top-level examples guide
