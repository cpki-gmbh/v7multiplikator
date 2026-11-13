# C inference examples

Every program here consumes the v7 public API the same way: load the DLL,
open a model database, initialize, run inference. Each program lives in its
own subfolder with its own README. Start with `inference_example\`.

Text LLM:

| Subfolder | What it shows |
|-----------|---------------|
| `inference_example\` | The canonical minimal chat program + performance stats. Start here. |
| `multi_turn_demo\` | Multi-turn chat (`reset=0` keeps the KV cache), interleaved `v7_generate`, fresh sessions (`reset=1`). |
| `poem_continuation\` | Raw-continuation contrast demo: `raw_generation=1` continues plain text with no template. |
| `v7_pipe\` | Unix-philosophy pipe processor (ask/chat/summarize/explain/qa), V7_DLL/V7_DB environment driven. |
| `dbg_inference\` | The public API sequence in pure x64 assembly -- the reference ASM consumer of the public tier. |

The engine test programs (chat_only, generate_only, multiturn,
multiturn_manual, engine_death, kv_cache, image_multiturn) are NOT examples:
they live in `tools\` with their docs (`tools\<name>.md`) and run from
`verify.bat`.

Vision (VLM):

| Subfolder | What it shows |
|-----------|---------------|
| `dbg_image_recog\` | Raw RGB image + prompt -> caption/QA via `v7_chat_image`. The canonical VLM smoke test. |
| `vit_bbox_eval\` | Synthesized 640x640 test image -> bounding-box answers; the HF-parity eval harness. |

## Build and run

```bat
build.bat                      (from the project root -- outputs to bin\)
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "what is a cat?" -n 1000
bin\engine_death.exe bin\v7_threading.dll
```

Both DLL and DB paths must be given explicitly -- there are no defaults
(by design, AGENTS.md Sec.106). Each subfolder's README has that program's
exact command line.

## The four-call pattern (context lives in BSS inside the DLL)

```c
v7_load("bin/v7_threading.dll", &dll);                    /* 1. load the DLL      */
dll.v7_init("weights/databases/qwen3_0.6b/lz4ht.db");     /* 2. open the DB       */
dll.v7_initialize_inference();                            /* 3. create state      */
dll.v7_chat("what is a cat?", 0, out, sizeof(out), 1);    /* 4. generate          */
```

If `dll.v7_engine_is_dead()` returns 1 after a call, an internal error
stopped that public call. Read `dll.v7_get_death_reason()`, call
`v7_free_inference()`, and unload the DLL. See the public API error-reporting
section in `docs/native_api.md`.

## Further reading

- `docs/native_api.md` -- full export surface + public API error reporting
- `docs/getting_started.md` -- example catalog and building model databases
- `docs/models_and_vision.md` -- supported models and their quirks
- `examples/README.md` -- the top-level examples guide
