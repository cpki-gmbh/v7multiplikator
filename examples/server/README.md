# OpenAI-compatible HTTP server

**A development tool and local test application — NOT for internet
exposure.** `v7_server.exe` is neither secure nor a finished inference
server: it is a usage example for the v7 library, with the whole request
layer reduced to transport framing (no authentication, no Host/Origin
gating, no path allowlists). Any client that can reach the port can quit
the server, reset the session, read the chat history, and load a sidecar
with FFI; binding non-loopback (`V7_SERVER_BIND`) exposes all of it to the
network. The `/sidecar` endpoint loads SectorLogo scripts into the running
server. Sidecars can call exported functions and external programs through
`DefineCommand` and `CallLibrary`, so the sidecar endpoint is a privileged
development feature. Keep the server local.

`v7_server_main.cpp` wraps the DLL's inference API behind the standard
OpenAI Chat Completions API plus a browser chat UI.

## Source layout

The server is 14 translation units plus 3 headers. `v7_server_main.cpp`
holds the globals, the ctrl handler and `main()`; every other `.cpp` is a
single-purpose module: `util` (log sanitize, UTF-8, query parsing, shared
error/redirect senders), `http` (request read / response send), `turn`
(inference session), `image` (image decode: GDI+ on Windows, stb_image on
Linux), `health` (handoff queue + health thread), `api_meta`
(health/models/quit/reset), `api_chat` (chat/completions/decide),
`api_vision`, `sidecar`, `web_render` (page template), `web_handlers`
(form endpoints), `router` (route ladder). The JSON parser/builders live
in `v7_server_json.h`, owned by the single translation unit
`v7_server_json.cpp`; everyone else sees only the struct declarations in
`v7_server_json_types.h`. Shared constants, state and function
declarations are in `v7_server_common.h`.

## Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Browser chat UI (chat + image chat, persistent KV cache) |
| `POST /v1/chat/completions` | Multi-turn text chat (server-side session state) |
| `POST /v1/completions` | Single-turn generate |
| `POST /v1/decide` | Typed decision: probabilities per option label |
| `POST /v1/vision/chat` | Image turn on the same session (base64 image) |
| `POST /v1/chat/reset` | Fresh session (clears KV cache) |
| `GET/POST /quit?code=N` | Terminate (42 = clean quit, 43 = port busy; a non-numeric or out-of-range code answers 400) |
| `GET/POST /sidecar?<path>` | Load a SectorLogo sidecar at runtime |
| `GET /v1/health`, `/v1/models` | Health / model listing |

## Run it

Use the lifecycle bats (project root, scripts\ directory) — never start the
exe directly:

```bat
scripts\start_server.bat    (own visible window, restart-on-crash loop, port 8080)
scripts\stop_server.bat     (clean /quit?code=42)
scripts\server_set_sidecar.bat examples\rag\memory\sidecar_memory_rag_v6.logo
```

Quick test:

```bat
curl http://localhost:8080/v1/health
curl -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
curl -X POST http://localhost:8080/v1/decide -H "Content-Type: application/json" -d "{\"kind\":\"noul\",\"state\":\"The sky is blue.\",\"question\":\"Is the sky blue?\",\"options\":[\"false\",\"true\"]}"
```

The decide endpoint is session-transparent: it runs on its own private KV
cache and never disturbs the chat session (run against a fresh server).

## Worker priority

The server defaults to `-prio -15` (Win32 idle): the machine stays
responsive while the server serves. To change it, add `-prio N` to the exe
call inside `server.bat`. The engine applies it via
`v7_set_worker_priority`, which defers to pool creation when the worker
pool does not exist yet, so call order never matters.

## Notes

- The server keeps ONE inference context; KV cache persists across turns
  (`reset=0`) until `/v1/chat/reset`. A request that carries its own
  history (any assistant message or more than one user message) is a
  STATELESS client: the server resets the session, plays back the
  request's transcript as the prompt, and logs the mode switch.
- Sampling overrides in requests (`temperature`/`top_p`) are parsed but
  NEVER applied; the server sets temperature/top-p by thinking mode
  (0.6/0.95 on, 0.7/0.8 off) and logs one line per request that carried
  either field.
- `max_tokens` accepts the number form only; `"max_tokens": null` means
  absent (the server default applies), a quoted value is a malformed
  request and answers 400.
- The web chat history holds `-history` entries (default 4096, minimum 2;
  `V7_SERVER_HISTORY` env passthrough). When full, the oldest user+bot
  pair is evicted — turns never vanish silently while inference runs.

## Further reading

- `docs/getting_started.md` — full server documentation (v7_server.exe section)
- `examples/README.md` — the top-level examples guide
