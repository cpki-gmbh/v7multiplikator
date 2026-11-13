# web_ragg — internet RAG tool

DuckDuckGo search → fetch top pages → extract text → MMR passage selection →
v7-LLM distillation into one short factual paragraph. Sidecars call it through
`DefineCommand` to add current-events information.

## Build

```bat
examples\rag\web_ragg\build_web_ragg.bat     (own build — not part of build.bat)
```

## Usage

```
web_ragg.exe search "<query>" ["pages=5;words=100;maxgen=250;fresh"]
web_ragg.exe makequery "<think and memories text>"
```

- `search` — runs search, page retrieval, text extraction, passage selection,
  and distillation.
  Prints ONLY the distilled paragraph on stdout (diagnostics go to
  `web_ragg.log` next to the exe — stdout is captured by the caller).
- `makequery` — distill a 3-8 word web search query from the model's think
  text (caps at 4000 chars input).
- Options string: `pages=N` (1-10), `words=N` (30-300), `maxgen=N` (10-500),
  `fresh` (bypass the page cache).

## Configuration

`web_ragg.ini` next to the exe (required, no defaults):

```ini
[v7]
Dll=<path to v7_threading.dll>
Db=<path to lz4ht.db>
```

Optional `[search]` keys: `Prepend`/`Append` (inline distillation prompts),
`PrependFile`/`AppendFile` (prompt files).

Page cache: `web_ragg.db` (flat CRC32-checked file; corrupt entries are
skipped, delete it to force re-fetching).

## Further reading

- this README — the full reference for the tool and its INI
- `docs/internet_rag_dev_log.md` — development log
- `examples/README.md` — the top-level examples guide
