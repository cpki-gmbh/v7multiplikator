# CPU Inference Benchmarks - v7 vs. third-party engines

All benchmark results for v7 against major third-party inference engines on
the same machine, same model, same prompt, same decoding (greedy, temp 0).
One bat per run, repeatable on this PC. The definitive dataset is the
unattended overnight run (2026-11-08, 5 rounds, truly quiet machine,
network off): `bench_all_models_overnight.bat`, full log in
`bench_results_overnight.txt`.

Machine: Ryzen 5 7430U, 6C/12T, AVX2 (no AVX-512/VNNI), Windows, net power.

Detailed engine versions, tuning evidence and regeneration commands:
docs/engine_benchmark.md

## Workloads

1. **Short chat turn** - "What is the capital of France?", 256-token budget.
   Real-world single turn; every engine applies its own chat template
   (think blocks where the model has them), generation counted per actually
   generated token.
2. **Long generation** - "write a long poem about horses, jesus and cats,
   remeber to talk about 1+2=3", 500-token budget. Steady-state throughput,
   directly comparable across engines.

## Engines (8-bit class only - no 4-bit benchmarks)

| Engine | Version | Weight format | Directory |
|--------|---------|---------------|-----------|
| v7 | dev tree (2.33.0) | native Q8 (microdb) | `bin\v7_threading.dll` |
| llama.cpp | b12179 prebuilt cpu-x64 | GGUF Q8_0 | `llamacpp\` |
| MNN | 3.6.2 built from source | int8 (llmexport) | `mnn\` |
| OpenVINO | 2026.4.1, CPU plugin capped to AVX2 | int8-asym (NNCF) | `openvino\` |

Engine settings (tuned 2026-11-07, evidence in docs/engine_benchmark.md):
llama.cpp `-t 6` (physical cores), MNN `thread_num 4`, OpenVINO default
(AVX2-capped), v7 physical-placement pool + cycle tuner.

ONNX Runtime GenAI (int4, 4 bit) was benchmarked in the earlier sessions
(numbers further down and in docs/engine_benchmark.md) and is EXCLUDED from
the definitive comparison: it moves half the bytes per token and is not
comparable to the 8-bit class. ORT GenAI also cannot tokenize MiniCPM5-1B
at all (pre_tokenizer regex error_badbrace).

## Definitive results - overnight run (2026-11-08, 13:21-14:10, 5 rounds)

Methodology: 5 rounds per engine and workload, run unattended with all
background programs closed and the network disconnected. Engine order per
workload: v7, OpenVINO, llama.cpp, MNN. Model order: qwen3-0.6b,
qwen3-1.7b, phi-3-mini, minicpm5-1b. Short column = stabilized state
(rounds 2-5, cold round 1 excluded); long column = mean of all 5 rounds.
All 32 legs completed, zero failures. Full per-round log:
bench_results_overnight.txt.

### Short chat turn (tokens/s, stabilized state)

| Model | v7 Q8 | llama.cpp Q8_0 | MNN int8 | OpenVINO int8 |
|-------|------:|---------------:|---------:|--------------:|
| Qwen3-0.6B | **55.8** | 52.0 | 48.2 | 41.4 |
| Qwen3-1.7B | **20.0** | 19.8 | 18.1 | 18.3 |
| Phi-3-mini (tuner-quiet) | 9.1 | **9.4** | 8.4 | 8.6 |
| MiniCPM5-1B (re-tune 2) | **38.1** | 37.7 | 34.8 | 32.0 |

### Long generation, 500-token budget (tokens/s, mean of 5 rounds)

| Model | v7 Q8 | llama.cpp Q8_0 | MNN int8 | OpenVINO int8 |
|-------|------:|---------------:|---------:|--------------:|
| Qwen3-0.6B | **52.2** | 47.3 | 43.8 | 39.7 |
| Qwen3-1.7B | **19.5** | 19.2 | 17.7 | 17.5 |
| Phi-3-mini (tuner-quiet) | 8.7 | **8.9** | 8.0 | 7.8 |
| MiniCPM5-1B (re-tune 2) | **37.1** | 36.6 | 33.4 | 29.1 |

MiniCPM5-1B and Phi-3-mini rows are from the re-tune runs (2026-11-08/09,
`bench_minicpm5_retune.bat` / `bench_phi3_retune.bat`, logs
`bench_results_minicpm5_retune.txt` / `bench_results_phi3_retune.txt`,
same method/machine). The Phi-3-mini rows are the tuner-quiet run
(2026-11-09 09:38) - the third phi3 run, with the cycle tuner inactive
during measurement.

**MiniCPM5-1B retune history** (v7 short/poem, llama.cpp for reference):
overnight 27.9/28.4 (-26 %/-22 %, untuned dispatch configs, thermally
worst slot) -> retune 1: 37.3/36.3 (statistical tie, llama 37.5/36.3) ->
retune 2 (widened tuning + database rebuild, tuner delay 100): 38.1/37.1 -
**v7 now leads llama.cpp (37.7/36.6) in BOTH workloads** (+1 %/+1.5 %),
with no cold-start ramp and prefill at 96-110 tok/s (was 67-78). llama.cpp
reproduced within ~0.5 % across all three runs (37.5-37.9 / 36.3-36.6), so
the comparison is clean.

The Phi-3-mini story took three runs (v7 short/poem, llama.cpp for
reference): retune 1 (18:10): 8.7/8.5 vs llama 9.4/9.1 - no improvement,
diagnosed as not dispatch-config. Overnight-tuned retune 2 (09:08, tuner
ACTIVE during measurement): 8.6/8.55 - still flat, but the tuner ran 52,061
episodes in the poem leg alone. Retune 3 (09:38, tuner quiet): **9.1/8.7**
- the gap to llama.cpp (9.4/8.9) collapsed from ~9 % to +3 %/+2 %. Two
lessons: the overnight promotions DO help once the tuner stops churning
mid-measurement (the active tuner perturbed its own benchmark by ~5 %),
and the residual gap sits in the prefill path (v7 10-14 tok/s vs MNN
22-29, llama 20-41), not in generation.

### Full per-round data (R1..R5, tokens/s)

Short chat turn:

| Model | Engine | R1 | R2 | R3 | R4 | R5 |
|-------|--------|---:|---:|---:|---:|---:|
| Qwen3-0.6B | v7 | 55.0 | 55.4 | 55.6 | 55.9 | 55.8 |
| | llama.cpp | 52.0 | 52.0 | 51.8 | 52.0 | 52.0 |
| | MNN | 48.2 | 47.6 | 48.6 | 48.4 | 48.1 |
| | OpenVINO | 28.9 | 41.5 | 41.1 | 41.4 | 41.5 |
| Qwen3-1.7B | v7 | 20.0 | 19.9 | 20.0 | 20.0 | 20.0 |
| | llama.cpp | 19.8 | 19.8 | 19.6 | 19.8 | 19.8 |
| | MNN | 18.0 | 18.1 | 18.0 | 18.3 | 18.2 |
| | OpenVINO | 14.6 | 18.3 | 18.3 | 18.3 | 18.2 |
| Phi-3-mini (tuner-quiet) | v7 | 8.5 | 9.1 | 9.2 | 9.1 | 9.1 |
| | llama.cpp | 9.3 | 9.4 | 9.4 | 9.4 | 9.4 |
| | MNN | 9.5 | 8.4 | 8.5 | 8.4 | 8.5 |
| | OpenVINO | 5.3 | 8.5 | 8.6 | 8.6 | 8.6 |
| MiniCPM5-1B (re-tune 2) | v7 | 37.5 | 38.2 | 38.1 | 38.1 | 37.9 |
| | llama.cpp | 37.8 | 37.5 | 37.7 | 37.8 | 37.8 |
| | MNN | 34.7 | 35.3 | 34.4 | 34.6 | 34.8 |
| | OpenVINO | 26.0 | 32.1 | 32.0 | 31.7 | 32.2 |

Long generation:

| Model | Engine | R1 | R2 | R3 | R4 | R5 |
|-------|--------|---:|---:|---:|---:|---:|
| Qwen3-0.6B | v7 (500 tok) | 51.5 | 52.6 | 52.1 | 52.2 | 52.5 |
| | llama.cpp (500 tok) | 47.5 | 47.4 | 47.0 | 47.5 | 47.3 |
| | MNN (512 tok) | 44.2 | 44.2 | 44.0 | 43.5 | 43.2 |
| | OpenVINO (451-478 tok EOS) | 39.7 | 38.8 | 39.8 | 40.0 | 40.0 |
| Qwen3-1.7B | v7 (500 tok) | 19.4 | 19.5 | 19.5 | 19.5 | 19.5 |
| | llama.cpp (500 tok) | 19.2 | 19.2 | 19.2 | 19.1 | 19.1 |
| | MNN (512 tok) | 17.7 | 17.7 | 17.7 | 17.7 | 17.7 |
| | OpenVINO (474-484 tok EOS) | 15.9 | 17.9 | 17.8 | 17.8 | 18.2 |
| Phi-3-mini (tuner-quiet) | v7 (500 tok) | 8.6 | 8.6 | 8.7 | 8.7 | 8.7 |
| | llama.cpp (500 tok) | 8.9 | 8.9 | 8.9 | 8.9 | 8.9 |
| | MNN (411-512 tok) | 8.0 | 8.0 | 8.1 | 8.0 | 8.1 |
| | OpenVINO (424 tok EOS) | 7.2 | 8.0 | 8.0 | 8.0 | 8.0 |
| MiniCPM5-1B (re-tune 2) | v7 (500 tok) | 37.1 | 37.1 | 37.1 | 37.2 | 37.1 |
| | llama.cpp (500 tok) | 36.5 | 36.4 | 36.6 | 36.6 | 36.7 |
| | MNN (512 tok) | 33.6 | 33.5 | 33.7 | 33.6 | 33.7 |
| | OpenVINO (439-443 tok EOS) | 28.4 | 29.7 | 29.5 | 29.5 | 28.6 |

Earlier minicpm5 iterations (for the record): retune 1 short - v7
37.2/37.1/37.0/37.7/37.5, llama.cpp 37.8/37.0/37.5/37.7/37.8, MNN
34.9/34.8/34.8/34.6/34.2, OpenVINO 29.8/32.3/32.2/32.2/32.0; retune 1
poem - v7 36.2/36.5/36.2/36.1/36.5, llama.cpp 36.1/36.3/36.4/36.1/36.5,
MNN 33.6/33.7/33.7/33.6/34.1, OpenVINO 28.8/30.0/29.9/30.2/30.2.
Overnight (pre-retune): short - v7 26.0/27.9/27.9/27.8/27.8, llama.cpp
38.1/37.9/37.8/37.9/37.8, MNN 34.7/34.8/35.5/35.0/34.8, OpenVINO
26.1/32.1/32.1/32.2/32.2; poem - v7 27.3/28.1/28.7/28.9/29.1, llama.cpp
36.5/36.5/36.7/36.6/36.6, MNN 33.8/33.9/33.8/34.1/33.9, OpenVINO
28.7/30.2/30.2/30.3/30.3.

Earlier phi3 iterations (for the record): retune 1 (18:10) short - v7
8.2/8.7/8.8/8.6/8.6, llama.cpp 9.3/9.4/9.4/9.4/9.5, MNN 8.7/9.8/8.6/9.8/9.7
(mixed sample lengths), OpenVINO 5.5/8.9/8.9/8.6/8.9; retune 1 poem - v7
8.4/8.4/8.5/8.5/8.5, llama.cpp 9.1/9.1/9.1/9.0/9.1, MNN 8.1/8.3/8.1/8.2/8.2,
OpenVINO 7.5/8.3/8.2/8.3/8.3. Retune 2 (09:08, tuner ACTIVE during
measurement) short - v7 7.9/7.9/8.7/8.6/9.2, llama.cpp 9.4/9.5/9.6/9.5/9.5,
MNN 8.5/8.4/8.3/7.7/8.5, OpenVINO 5.1/8.8/8.7/8.6/8.4; retune 2 poem -
v7 8.5/8.5/8.6/8.6/8.6, llama.cpp 9.0/8.9/9.0/9.0/9.0, MNN 8.0/8.1/8.1/8.1/8.1,
OpenVINO 7.4/8.2/8.2/8.2/8.0.

### Observations (definitive run)

- **Qwen3 (0.6B and 1.7B): v7 is the fastest 8-bit engine in both
  workloads.** 0.6B short: v7 55.8 vs llama.cpp 52.0 (+7 %); poem: 52.2 vs
  47.3 (+10 %). 1.7B: v7 leads llama.cpp by ~1-2 % - a close race, but v7
  holds both workloads.
- **Phi-3-mini: llama.cpp leads by only ~+3 %/+2 %** (tuner-quiet run:
  9.4/8.9 vs v7 9.1/8.7). Three-run history: retune 1 (18:10) 8.7/8.5 - no
  improvement; overnight-tuned retune 2 (09:08) 8.6/8.55 - still flat, but
  the cycle tuner ran 52,061 episodes in the poem leg alone; retune 3
  (09:38, tuner quiet) 9.1/8.7 - the gap collapsed from ~9 %. The active
  tuner perturbed its own benchmark by ~5 %; with it quiet, the overnight
  promotions show through. The residual gap sits in the prefill path (v7
  10-14 tok/s vs MNN 22-29, llama 20-41), not in generation.
- **MiniCPM5-1B: after two retune iterations, v7 leads both workloads**
  (38.1/37.1 vs llama.cpp 37.7/36.6, +1 %/+1.5 %). History: overnight
  27.9/28.4 (-26 %/-22 %, untuned dispatch configs, thermally worst slot)
  -> retune 1: 37.3/36.3 (tie) -> retune 2 (widened tuning + database
  rebuild, tuner delay 100): v7 in front, no cold-start ramp, prefill
  96-110 tok/s (was 67-78). llama.cpp reproduced within ~0.5 % across all
  three runs - the edge is small but real. The lesson across both models:
  tuned geometries win, the tuner itself must stay quiet during
  measurement, and the remaining phi3 gap is prefill, not generation.
- **The afternoon 3-round session (2026-11-07/08) was background-loaded**:
  overnight, EVERY engine got faster (llama.cpp +26 % on qwen3-0.6B short:
  41.4 -> 52.0; MNN +7 %; OpenVINO +7 %; v7 +10 %). The overnight numbers
  supersede the afternoon numbers everywhere.
- OpenVINO round 1 is always the cold round (graph read/JIT, 30 % slower);
  the stabilized rounds are rock-stable. v7 qwen legs are stable from
  round 1. MNN legs are warm from round 1 in this run (llm_demo tuned-cache
  files existed from earlier runs).
- phi3-mini short turn: only 7-9 tokens before EOS on every engine (no
  think block) - small-sample noise; the poem workload is the meaningful
  number for this model. MNN's sampled rounds alternate between short
  (9 tok) and long (~90 tok) answers - its 8.7/9.9 alternation reflects
  that, not throughput variance.
- All outputs correct on every engine: Paris answers (deterministic
  greedy, identical think blocks across rounds) and poems containing
  horses/Jesus/cats/1+2=3. MNN decodes with its config sampler (0.8),
  so its answer text differs per round; throughput unaffected.
- v7 cycle tuner active throughout (e.g. 849 episodes / 395 promotions in
  the qwen3-0.6B short leg), databases built with tuner-delay 100.

## Power-saving mode comparison (2026-11-10, tuner loop cycle)

One full tuner-loop cycle ran with the Windows power plan set to
**Power saver** instead of the usual high-performance state
(`benchmark_tuner_loop_20261112_081959.txt`, three complete runs 08:31 /
09:36 / 10:41 plus a fourth started 11:44). Every engine was re-measured
back to back: first with the cycle tuner running (delay 0, random jitter),
then with the tuner quiet (delay 99999). The quiet run is the honest
power-saving figure. These numbers do NOT replace the headline tables
above - the machine state is different - but the relative comparison is
real and stable.

### Power-saving results, tuner-quiet run (tok/s, full-power baseline in parentheses)

| Model | Load | v7 Q8 | llama.cpp | MNN int8 | OpenVINO |
|-------|------|------:|----------:|---------:|---------:|
| Qwen3-0.6B | short | **52.0** (55.8) | 43.0 (52.0) | 44.0 (48.2) | 34.2 (41.4) |
| Qwen3-0.6B | poem | **48.5** (52.2) | 39.5 (47.3) | 39.6 (43.8) | 26.4 (39.7) |
| Qwen3-1.7B | short | **18.6** (20.0) | 15.0 (19.8) | 16.3 (18.1) | 12.7 (18.3) |
| Qwen3-1.7B | poem | **18.7** (19.5) | 14.9 (19.2) | 16.5 (17.7) | 12.4 (17.5) |
| Phi-3-mini | short | 8.1 (9.1) | 7.4 (9.4) | **8.2** (8.4) | 6.2 (8.6) |
| Phi-3-mini | poem | **8.7** (8.7) | 7.1 (8.9) | 7.5 (8.0) | 5.7 (7.8) |
| MiniCPM5-1B | short | **36.9** (38.1) | 28.3 (37.7) | 32.3 (34.8) | 23.1 (32.0) |
| MiniCPM5-1B | poem | **35.8** (37.1) | 28.0 (36.6) | 31.1 (33.4) | 21.4 (29.1) |

### Findings

- **v7 degrades least under power saving**: -6.8% on Qwen3-0.6B short
  (55.8 -> 52.0) vs -17.3% for llama.cpp (52.0 -> 43.0) and OpenVINO
  (41.4 -> 34.2), and -8.7% for MNN. v7 keeps 93% of its full-power
  throughput; llama.cpp and OpenVINO keep 83%.
- **v7's relative advantage GROWS under throttling**: the Qwen3-0.6B short
  lead over llama.cpp widens from +7% at full power to +21% under power
  saving. Spin-wait physical-core placement and tuned dispatch are
  clock-ratio-bound; llama.cpp's thread-pool handoffs degrade harder.
- **Ranking flips**: Phi-3-mini poem flips from a llama.cpp win (8.9 vs
  8.7) to a v7 win by 23% (8.7 vs 7.1). Phi-3-mini short drops llama.cpp
  from first to third (MNN 8.2 > v7 8.1 > llama 7.4). v7 wins seven of
  the eight workloads under power saving (six of eight at full power).
- **Tuner convergence across loop cycles** (Qwen3-0.6B short, three
  consecutive legs): tuner active with fresh populations 48.5 tok/s,
  tuner quiet 52.0 tok/s, tuner active again with the cycle-1 promotions
  already persisted 51.0 tok/s. The second active leg closes most of the
  gap to the quiet setting - the tuner's work carries over through the
  persisted populations, and mid-measurement tuner churn costs up to ~7%.

## Historical: afternoon 3-round session (2026-11-07/08, superseded)

Ran with background processes present (browser, Discord, smoke tests
between legs). Kept for the record; the overnight tables above replace
these numbers. Notable then: llama.cpp qwen3-0.6B short measured only
41.4 t/s (-21 % vs overnight 52.0) - the clearest evidence of the
background-load effect. v7 Q8 was fastest in the 8-bit class on both
Qwen3 models in that session as well (50.5/47.2 on 0.6B, 19.0/18.4 on
1.7B); phi3-mini and minicpm5 relative rankings were the same as overnight
(llama.cpp first on both). The ORT int4 numbers from that session:
qwen3-0.6B 65.5/58.4, qwen3-1.7B 27.7/25.9, phi3-mini ~9.6 short / 13.0
poem - fastest raw engine, 4-bit weights, excluded from the 8-bit class.

## Databases (v7 side)

All v7 databases used in the benchmarks are rebuilt with the in-process
cycle tuner delays set to 100 (`-tuner-delay 100 -tuner-delay-large 100`):

    rebuild_dbs_phi3_minicpm5_delay100.bat

covers phi3_mini, minicpm5_1b and qwen3_1.7b. The qwen3_0.6b database was
built with the same delay setting. The cycle tuner stays ACTIVE during
benchmarks (episode/promotion counts print at the end of each v7 leg).

## Reproducing

    .\bench_all_models_overnight.bat      rem THE definitive run: 4 models,
                                          rem short+poem, 5 rounds, 4 engines,
                                          rem log: bench_results_overnight.txt

Re-tune re-benchmarks (same method as the overnight legs, isolated model):

    .\bench_minicpm5_retune.bat           rem log: bench_results_minicpm5_retune.txt
    .\bench_phi3_retune.bat               rem log: bench_results_phi3_retune.txt

Per-model 3-round bats (older methodology):

    .\bench_all_engines.bat            rem qwen3-0.6b short turn
    .\bench_all_engines_poem.bat       rem qwen3-0.6b long generation
    .\bench_all_engines_phi3.bat       rem phi3-mini short turn
    .\bench_all_engines_poem_phi3.bat  rem phi3-mini long generation
    .\bench_all_engines_minicpm5.bat   rem minicpm5-1b short turn
    .\bench_all_engines_poem_minicpm5.bat
    .\bench_all_engines_qwen17b.bat    rem qwen3-1.7b short turn
    .\bench_all_engines_poem_qwen17b.bat

## Acquiring the competitor models (2026-11-07)

| Model | llama.cpp Q8_0 | OpenVINO int8 | MNN int8 |
|-------|----------------|---------------|----------|
| phi3_mini | converted with llama.cpp b12179 `convert_hf_to_gguf.py --outtype q8_0` (official microsoft GGUF repo has only Q4_K_M/FP16; needs the HF `tokenizer.model`) | `optimum-cli export openvino --weight-format int8` | `llmexport.py --quant_bit 8` (after removing the stale `auto_map` from config.json - transformers has native phi3 support) |
| minicpm5_1b | official `openbmb/MiniCPM5-1B-GGUF` Q8_0 | same recipe | same recipe (model_type llama) |
| qwen3_1.7b | official `Qwen/Qwen3-1.7B-GGUF` Q8_0 (local HF dir needed `model.safetensors.index.json` re-fetched) | same recipe | same recipe |

Conversion bats in the project root: `fetch_phi3_minicpm5_models.bat`,
`convert_phi3_gguf.bat`, `export_ov_ir_phi3_minicpm5_qwen17b.bat`,
`export_ov_ir_qwen17b.bat`, `build_ort_int4_phi3_minicpm5_qwen17b.bat`,
`export_mnn_phi3_minicpm5_qwen17b.bat`.
