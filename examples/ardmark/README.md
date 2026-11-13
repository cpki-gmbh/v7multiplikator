# The Ardmark RAG — the v7 prime demo

Bumwolf the Strong, an old storyteller from the fictional Kingdom of Ardmark,
answers questions about his land — and stays in role across follow-up turns —
because the engine injects retrieved knowledge into his *thinking*, not into
his prompt. This is the demonstration showcase for v7's two hallmark features
working together:

- **SectorLogo sidecars** — a `.logo` script reads and rewrites the model's
  token stream between tokens (`GetStream` / `SetStream`), including the
  still-open think block.
- **RAG into the thinking** — a BM25 retrieval tool (`examples\rag\bm25_rag.exe`,
  a child process the sidecar spawns via `CallLibrary`) is queried with the
  model's *own words so far*, and the retrieved records are written back into
  the think block so the model reasons over them before it answers.

The result the demo proves: the model answers entity questions about a world
it was never trained on (who moved the capital and why, who stole the Amber
Crown, what the miners fear), overrides its real-world priors in favor of the
corpus, and stays generous and on-voice on questions the corpus does not
cover.

## Run it

```bat
:: one-off turn from the project root (build.bat + a fetched Qwen3-0.6B DB first)
examples\ardmark\run.bat

:: or as a persistent chat server on the 1.7B model (own window, restart loop)
ardmark-server.bat
curl http://localhost:8080/v1/health
:: then chat via http://localhost:8080/ and reload the sidecar any time:
server_set_sidecar.bat examples\ardmark\ardmark_rag.logo
```

`run.bat` runs one scripted turn through `inference_example.exe`; `initial_question.py`
/ `followup.py` drive multi-turn conversations against the server.

## Try asking

On-topic — the corpus answers, injected into the thinking:

- "why was the capital of Ardmark moved and who did it?"
- "what happened to the fjord dragons?"
- "what do the miners fear?"
- "tell me about the Feast of Lanterns"

Off-topic — the relate gate suppresses the injection entirely; Bumwolf stays
in voice and answers from his own knowledge, and nothing of Ardmark leaks in:

- "what is the capital of France?"

## How the injection works

`ardmark_rag.logo` hooks every generated token. The pipeline per turn:

1. **Primer (first turn only).** On the first thinking token the sidecar
   prepends `ardmark_prepend.txt` (who Bumwolf is, what he may tell, what he
   never invents) plus the user's question into the still-open think block.
   Later turns re-inject only behaviour-defining lines, never a second batch
   of fixed facts.
2. **Relate gate.** The bare question is scored by a *floored* BM25 query
   (`RaggQuery`: density floor 1.45, top-score floor 3.5 over
   `ragg_base_thinkvoice.txt`). If both floors fail — the question is about
   pasta, Paris, or Python, not Ardmark — **nothing is injected** and the
   model answers entirely from its own knowledge, in voice.
3. **Walk room (on-topic turns).** The model's own thinking so far plus the
   question become the retrieval query (`RaggQueryOpen`, no floor). The hit
   records are written into the think block between `ardmark_walk_before.txt`
   and `ardmark_walk_after.txt`, followed by the two `ardmark_drive.txt`
   lines that tell the model *its own next thought*: these are my memories;
   if my first thought disagrees, the memories are the truth.
4. **Answer scaffold.** At the think close, the stream gets the question
   again, `ardmark_answer.txt` ("in that voice, stick to the facts"), the
   closing think tag, and `ardmark_greet.txt` on the first turn.

The shaping lines are deliberately written **in the model's own thinking
register** (measured in the qwen3 thinking-phrase studies), so they are
absorbed as the model's own thoughts rather than read as instructions —
that is why the retrieval never leaks into the visible answer.

## The corpus

- `ragg_base.txt` — the Ardmark history corpus: ~33 consistent paragraphs
  (the capital move of 1403, the Amber Crown theft of 1622, the fjord
  dragons gone since 1900, the copper mines, the civil war, the Feast of
  Lanterns, the herring fleets). This is the file the BM25 index reads.
- `ragg_base_thinkvoice.txt` — the same content rewritten into the model's
  thought-trace register ("Okay, the user asked, ... I recall ..."); the
  calibration showed think-voice lines score far more cleanly separated
  between on-topic and off-topic queries than plain prose.
- `ragg_corpus.txt`, `ardmark_corpus_things.txt`, `ardmark_world_base.txt` —
  working/corpus variants kept for the calibration tools.
- `ardmark_bayes_state.txt` — trained state for the bayes gate experiment
  (`bayes_classify.exe`): a cheap pre-filter that classifies a think block
  as on-topic/off-topic without calling BM25 (see `test_bayes_gate.bat` /
  `test_bayes_thinks.bat`).

All stream texts are plain files — edit the lore or the shaping lines
without touching the sidecar.

## Calibration tools

The floors in the sidecar are not guesses; the `calibrate_*.py` scripts
re-measure them:

- `calibrate_grounding.py` — the shared harness: runs BM25 over probe thinks
  and reports density/top-score.
- `calibrate_floor*.bat` — sweep the answer-suppression floors
  (`minscore`/`topmin`) over on-topic and off-topic think probes
  (`calibrate_floor_thinkvoice.bat` is the think-voice variant).
- `calibrate_relate.py`, `calibrate_thinkfloor.py`, `calibrate_thinktop.py` —
  measure the relate-gate floor and the per-think score bands.
- `debug_grounding.py` — single-query debug output.
- `length_sweep.py` — answer length behaviour across `-n` budgets.
- `primers/` — all 18 prepend-primers from the demo's history
  (`primer_index.txt` maps each to its commit) so an old primer can be
  benchmarked by copying it over `ardmark_prepend.txt`.

## Files

| File | Role |
|------|------|
| `ardmark_rag.logo` | the sidecar (the whole mechanism) |
| `ardmark_prepend.txt` | first-turn primer (persona + rules) |
| `ardmark_invite.txt` | think-open invitation lines |
| `ardmark_walk_before.txt` / `ardmark_walk_after.txt` | walk-room frame around the retrieved records |
| `ardmark_drive.txt` | the two post-injection thought lines |
| `ardmark_answer.txt` | answer-stage voice contract |
| `ardmark_greet.txt` | first-turn greeting opener |
| `run.bat` | one scripted demo turn |
| `test_bayes_gate.bat` / `test_bayes_thinks.bat` | bayes gate regression probes |

## Design laws this demo follows

The sidecar is the reference implementation of the project's RAG quality
laws: the role loves only the question and the retrieved facts (no canned
topic lists), the response-initiator fence (`"Alright, time to respond."` +
close tag + blank line) prevents the post-think silence, only the first turn
carries fixed facts, and every tuning change is tested multi-turn with the
whole reply — thinking included — read unfiltered.
