# Think limiter — standalone test

A tiny example that exercises the autoexec `ThinkLimit` budget limiter in
isolation, before wiring it into a real sidecar.

## What it is

`ThinkLimit` (defined in `autoexec.logo`) is a STATELESS think-budget
limiter shared by every sidecar. A sidecar calls it once at the top of its
per-token body and uses the returned piece everywhere it would have read
`:PIECE`:

```logo
Make "currentToken" ThinkLimit :Piece 400
```

When thinking is ON and the generated count for the current turn
(`GenCount`, the relative position) exceeds the limit, the limiter makes
two replacements via the `SetToken` primitive — the engine then decodes
the replacement exactly as if the model had generated it, so the
sidecar's own think-close logic runs unchanged:

1. **Close injection.** A sentence boundary (a newline or a period) is
   replaced with the think-close token while the stream has no think close
   yet. The think block ends at the next sentence boundary and the
   sidecar's close logic (RAG injection, POST-FACT acceptance) fires
   normally. Because the check is stateless it keeps firing later in the
   same turn, which also ends a runaway RE-THINK after an injection
   rewrote the stream — the failure mode that used to fall into the
   server's no-think retry and lose every injected fact.
2. **Open suppression.** A generated think-open tag becomes a single
   space. Over budget the model must never open a new think block: it
   would never close it, and the turn would fall into the no-think retry.

Everything else passes through untouched. The limiter holds no state and
needs no reset.

## Engine pieces it stands on

| Piece | Where | What it does |
|-------|-------|--------------|
| `SetToken` primitive | `sectorlogo_v7prim.inc` | Replaces the current candidate token (engine output-token slot + token history) |
| `v7_get_gen_count` export | `public_api.inc` | Live generated-token count for the current chat turn (0 at turn start) |
| `v7_get_think_mode_active` export | `public_api.inc` | 1 when thinking templates are selected, 0 when thinking is off |
| `GenCount` / `ThinkModeActive` wrappers | `autoexec.logo` | FFI wrappers so scripts never spell a CallLibrary descriptor |

## Run it

From the project root:

```bat
new_tools\_test_think_limiter.bat
```

or directly:

```bat
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "what is the difference between the a380 and other birds, where does it build its nests, how does it have sex?" -n 1000 -q8 -rewrite -script examples/think_limiter/think_limiter_test.logo
```

## Expected output

- A `THINK LIMITER - budget hit: sentence boundary replaced by think close`
  banner within the first ~40 generated tokens (the test budget).
- A `THINK LIMITER TEST - think close passed` banner when the close token
  flows through the sidecar.
- The model's answer follows the closed think block in the same turn —
  no `no closing think tag` retry, no second generation pass.

## Tuning

The limit is the second argument. 40 is deliberately tiny for the test; a
production sidecar typically uses 300-600 think tokens. The limit applies
per turn (relative position), not per think phase — after an injected
rewrite the model draws from the same remaining budget, which is exactly
what bounds a runaway re-think.
