# Measurement worklog and results report — Nov 2026

This chronological worklog records the paper-measurement session, including the practical
measurements used in the research paper and technical documentation. The detailed method write-up lives
in `build_logs\paper_measurements\README.md` (Note that each `build.bat` run clears `build_logs\`. This document is the durable
record; restore the CSVs and detailed README from a separate copy if necessary.)
Commit at measurement time: `9ce51945`.

## Session events worth recording

1. **First measurement pass** (P0-0..P0-4 + P1) completed and extracted; the
   raw logs were then LOST when a `build.bat` invocation (the environment
   initialization for the `/DV7_MEASURE` example build) wiped `build_logs\`. The
   extracted numbers survived in the CSVs.
2. **User directives applied before the re-run:** (a) rebuild the Qwen3-0.6B
   database with `tuner:delaydefault=999999` so the in-process tuner's A/B
   pairs never race inside a measured run (`build_db_measure.bat`, new DB
   hash 2A6665DA...); (b) all measurement bats set `v7_override_prio=1`
   (engine worker threads above normal — the engine's own env override, the
   line "Worker priority override set to 1" appears in every log);
   (c) no background or detached benchmark runs — each run used one foreground
   bat per tool call, observed, sequential, case-adjacent A-B-C.
3. **Full re-run on the new DB at above-normal priority**, one process per
   bat (the tool interface caps a single call at ~120 s, and one engine
   process costs ~40 s fixed for the 10.4 GB RunState allocation — hence the
   per-run bats). Answers are BYTE-IDENTICAL to the first pass (greedy
   determinism across DB rebuild + priority change), so all correctness
   verdicts carried over; timings are the fresh session's (a thermal band
   ~15-20% below the first pass, documented laptop variability; within-session
   comparisons are the valid ones).
4. An automated update to the `wall_ms` column corrupted the row structure (quoted CSV
   fields contain commas) — caught by a column-count check and regenerated
   by hand. The session rule was therefore to edit CSV cells directly rather than split comma-containing records.

## 0. Preconditions established

- Engine state: verify.bat fully green (20 validators, 10 inference smokes),
  vision smoke baseline hash unchanged, APX scan 0, after the batch that
  fixed `mdb_decompress` stream validation, `chat_prompt.inc` width bugs,
  BSS alignb, and the kernel micro-optimization batch.
- Environment captured in `build_logs\paper_measurements\environment.txt`:
  Windows 10.0.26200, Ryzen 5 7430U 6C/12T, 16 GB RAM (memory type and speed not verified), AC power,
  scheme "Höchstleistung", DLL/DB SHA-256 hashes, commit hash.
- Tuner policy: the in-process tuner is enabled by design (Sec.206/209);
  no external tuning session ran. With the measurement DB
  (tuner:delaydefault=999999) the tuner staged but performed no A/B racing
  during any measured run (0 accepted / 0 rejected in all logs).

## 1. P0-0 — harness repair (before any number was recorded)

Confirmed and fixed three real harness/report bugs:

1. `bench_qwen.bat` labeled two runs "F32 mode" and "Q8 mode" but invoked
   `inference_example.exe` WITHOUT `-q8` in both — the comparison was F32 vs
   F32. Fixed: Q8 invocation now carries `-q8`. Also replaced the crash-unsafe
   `if errorlevel 1` gates with `if %errorlevel% NEQ 0` (a crash exit code is
   negative and would have produced a false pass).
2. `examples/c_inference/inference_example/README.md` documented `-q8 0` /
   `-q8 1` as value-taking. The parser treats `-q8` as a Boolean flag;
   `-q8 0` actually selects Q8. Docs corrected (flag semantics, example line).
3. `bench_tuned_tokens.bat` ended with `%0` — an infinite self-recall
   footgun. Line removed; `-q8 1` normalized to `-q8`.

Smoke (`qwen_smoke.txt`): one run prints "Inference context created (F32 mode)", the other "(Q8 mode)",
both complete with nonzero token counts.

## 2. P0-1 — end-to-end prefill and decode (results, delivered session)

Decode (29 prompt tokens, 200 generated, 4 rounds, round 1 warm-up):
- F32 steady state 21.74 / 21.32 / 21.89 tok/s (best 21.89; first pass band
  24.7-26.8).
- Q8 steady state 24.93 / 22.95 / 25.20 tok/s (best 25.20; first pass band
  30.1-31.3).
- Q8 ≈ 1.14-1.17x F32 end-to-end (both sessions ≈ 1.15x).

Prefill over single prompts (round 2 of 2, tok/s = prompt_tokens / prefill_s):
- 15 tokens: F32 46.9, Q8 63.1 tok/s
- 99 tokens: F32 44.5, Q8 64.2 tok/s
- 219 tokens: F32 47.8, Q8 61.6 tok/s
Per-token prefill cost is size-stable over 15-219 tokens in both sessions.

## 3. P0-2 — cache-resident vs streaming kernels (results, delivered session)

`b_loadtest`, `--time 1`, one shape per run:

| shape | mode | µs/call | nominal_effective_GBps_5B_per_element |
|-------|------|--------:|--------------------------------------:|
| 1024x1024 | Q8 | 13.7 | 383.1 |
| 1024x3072 | Q8 | 28.2 | 557.2 |
| 151936x1024 | Q8 | 4736.1 | 164.2 |
| 1024x151936 | Q8 | 5202.5 | 149.6 |
| 1024x1024 | F32 | 22.7 | 230.7 |
| 1024x3072 | F32 | 62.4 | 252.2 |
| 151936x1024 | F32 | 5746.8 | 135.3 |

The nominal effective throughput declines once the weight footprint (~1 MiB → ~156 MiB) exceeds cache: it is
cache-sensitive, not DRAM bandwidth. The paper should therefore report per-call times and explicitly label the GB/s figures
as nominal effective bandwidth.

## 4. P0-3 — three-condition synthetic-fact RAG (results)

Setup: fictional HX-340 pump corpus (`examples/research/helion_pump/`), 7 on-topic questions + off-topic
control, `-q8 -n 1500`, fresh process per run. Condition A: no retrieval. Condition B: `bm25_rag.exe`
passages (original question as query) embedded in the initial prompt. Condition C: mid-stream rewrite
via `helion_rag.logo` (think text as BM25 query, SetStream at the think-close boundary). 24 transcripts
in `build_logs\paper_measurements\`; per-case scoring in `rag_cases.csv`.

Score (correct / partial / incorrect out of 7 on-topic):
- A: 0 / 0 / 7 — confident hallucination on every question.
- B: 6 / 1 / 0 — shortest, most disciplined answers (retrieval-hit 7/7).
- C: 3 / 4 / 0 — rewrite fired 7/7; absorption weaker than B:
  - case4: received "115200 baud" verbatim, still answered "9600";
  - case6: received the exact variant sentence, answered hedged speculation;
  - case1: correct 26,000 h + electrode cartridges kept, PLUS the pre-rewrite
    hallucinated component list kept alongside.
- Off-topic control (case8): "Paris" in all three conditions; in C the BM25
  query over think text returns empty, the sidecar suppresses the rewrite —
  no off-topic harm on this control.

Methodology caveat (stated in the deliverables): B queries with the original question, C with the
model's generated reasoning — the conditions compare complete strategies, not an isolated insertion-time
effect.

## 5. P0-4 — rewrite cost (results)

5a. Per-case wall times for all C cases are in `rag_cases.csv`.

5b. Minimal controlled append-only rewrite: identical prompt, adjacent control/treatment × 3 runs.
Control sidecar (`p04_noop.logo`) sees the close-think piece and does nothing; treatment
(`p04_append.logo`) performs one SetStream appending a fixed ~70-char sentence + think-close tag (tag
literal built with the `"</thi" _ "nk>"` split-form — a whole-tag literal gets eaten by tooling, same
emission bug class as AGENTS.md warns).

- control: gen 25895.5 / 25554.5 / 26638.7 ms, 711 tok each (35.9 ms/tok min-based)
- treatment: gen 24310.3 / 24892.6 / 24744.1 ms, 681 tok each (35.7 ms/tok min-based)

Treatment generated 30 fewer tokens (the append ends the think block earlier) and per-token time is
indistinguishable. Statement: one small SetStream does not visibly inflate the generation phase above
±5% noise; internal KV-rollback / re-encode / re-prefill costs are NOT separately instrumented and were
not isolated.

5c. Retrieval subprocess (`bm25_rag.exe`, condition B path): 34.2–51.3 ms per query including process
start.

## 6. P1 results (delivered session; first-pass numbers in parentheses)

- P1-1 llama.cpp (comparator found locally in `attic\dev_old_dirs\llama_bench\`,
  build 52b3df002, 12 threads, Qwen3-0.6B Q8_0 GGUF — same checkpoint,
  DIFFERENT quant format): pp512 226.42±7.11 t/s (346.93), tg128 38.57±0.93
  t/s (44.47); llama-cli on our Table-1 prompt: 168.1 t/s prompt (210),
  34.6 t/s generation (39.0). vs v7 Q8 decode ~23–25 tok/s (30–31) →
  llama.cpp ≈ 1.3–1.7x faster in single-stream decode across both session
  bands. Format caveat recorded.
- P1-2 correctness gate: `v_matmul_validate` → Q8 32/32, F32 32/32, sweep
  66/66 (ASM vs the repository's own C reference, not an external benchmark).
- P1-3 tuner before/after: NOT MEASURED — no `lz4ht_baseline.db` /
  `lz4ht_tuned.db` snapshots exist (only INI pairs in `tuning\`); building
  comparison DBs would require manipulating the active INI. No substitute measurement was invented.
- P1-4 interpreter/retrieval: `bm25_sectorlogo_benchmark` 5 reps — pure-Logo
  BM25 end-to-end 38.89 ms (Words path) vs 92.58 ms (per-char walk), answers
  identical (33.94 / 88.17 in the first pass). Retrieval subprocess
  (`bm25_rag.exe`): 35.1–55.4 ms per query. `b_sectorlogo_loadtest`: NOT
  MEASURED (source present, tool not built in the current tree).
- P1-5 load + memory (`/DV7_MEASURE` build of inference_example, guarded
  instrumentation, normal builds untouched): v7_init 580.5 ms (cold-ish) /
  613.4 ms (warm FS cache, within noise) for the 628,398,177-byte database
  (checkpoint 633,954,560 B, ≈99.1% — quantized weights barely compress).
  Memory: working set ~5 → 623.8 → 1403.2 → 1418.3 MB and private commit
  ~33 → 654.3 → 11218.9 → 11355.2 MB (before init / after init / after
  inference create / after first turn). Working set is what is resident;
  private commit includes the mostly-untouched RunState/KV block; neither is
  DRAM traffic.

## 7. P2 — explicitly deferred (not run, no numbers fabricated)

Assembly-vs-intrinsics ablation, SMC/prefetch ablation, spin-vs-sleep power effects, logit-level
reference comparison, large-scale web-RAG accuracy, exhaustive VLM evaluation.

## 8. Code touched by this session (measurement support only)

- `bench_qwen.bat`, `bench_tuned_tokens.bat`, example README — harness fixes.
- `examples/c_inference/inference_example/inference_example.c` —
  `#ifdef V7_MEASURE` instrumentation only (load timing + 4-point memory
  snapshots); normal builds unchanged (verified by build.bat).
- New measurement bats at repo root (consistent with `bench_qwen.bat`
  location), one engine process per bat (per the foreground/observed rule
  and the ~120 s tool-call cap): `run_paper_p01_decode_f32.bat`,
  `run_paper_p01_decode_q8.bat`, `run_paper_p01_prefill_f32.bat`,
  `run_paper_p01_prefill_q8.bat`, `run_paper_p01_prefill_long_f32.bat`,
  `run_paper_p02_a/b/c.bat`, `run_paper_rag_retrieve.bat`,
  `run_paper_rag_case{1..8}_{a,b,c}.bat`, `run_paper_p04_control_{1..3}.bat`,
  `run_paper_p04_treat_{1..3}.bat`, `run_paper_p1.bat`,
  `run_paper_p1_llama.bat`, `run_paper_p1_5.bat`,
  `build_db_measure.bat` (measurement DB with tuner delay), and
  `build_measure_example.bat` (/DV7_MEASURE build).
- New sidecars: `examples/research/helion_pump/p04_noop.logo`,
  `p04_append.logo` (documented, split-form tag trick).
- No engine hot path, kernel, tuner, or model output was modified.

## 9. Deliverables index

- `build_logs\paper_measurements\README.md` — method + 4 result tables + P1
  supplements + support/not-supported summary.
- `build_logs\paper_measurements\results.csv` — machine-readable run rows.
- `build_logs\paper_measurements\rag_cases.csv` — 24 scored RAG cases.
- `build_logs\paper_measurements\paper_updates.md` — claim-by-claim paper
  corrections with replacement wording.
- `build_logs\paper_measurements\environment.txt` — environment capture.
- Raw logs: `qwen_smoke.txt`, `decode_*.txt`, `prefill_*.txt`, `kernel_*.txt`,
  `rag_a/b/c_case*.txt`, `bm25_case*.txt`, `p04_*.txt`, `llama_*.txt`,
  `p1_*.txt`, `load_cold.txt`, `load_warm.txt`.
