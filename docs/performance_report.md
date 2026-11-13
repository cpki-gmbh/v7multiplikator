# v7 Performance Report, Generation Statistics Module

> This report records the August 15, 2026 generation statistics measurements.
> The November text-engine comparisons appear in the [CPU inference benchmark](engine_benchmark.md).

Measurement date: August 15, 2026. Host: Ryzen 5 7430U with 6 physical cores and 12 logical threads. The system was idle,
background applications were closed, and the laptop was connected to AC power. Model: Qwen3-0.6B with
the database `weights\databases\qwen3_0.6b\lz4ht.db`.

## 1. What was built

RDTSC-based per-phase timing for the chat and generation pipeline:

- **A `.bss` statistics block** in `v7_data.inc` with 10 flat `uint64` variables:
  `rdtsc_chat_start`, `rdtsc_after_tokenization`, `rdtsc_after_prefill`,
  `rdtsc_after_generation`, `number_token_prefill`, `number_token_generate`,
  `rdtsc_vit_encode_start`, `rdtsc_vit_encode_end`,
  `number_token_prefill_vision` and `number_token_generate_vision`.
- **Probes**: a shared `RDTSC_TO_MEM` macro, serialized with `lfence`, writes the
  timestamps at the corresponding phases of `chat.inc`, which holds `v7s_chat`
  behind public `v7_chat`, and of `chat_image.inc`, which holds
  `v7s_chat_image_mem` and `v7s_generate_image_mem` behind public
  `v7_chat_image` and `v7_generate_image`. The vision
  encode timing and token-count mirrors sit in the same place.
- **Export**: `v7s_get_generation_statistic()` returns a pointer to the first
  variable, wired into `v7_lib.h` through the typedef, the struct field and
  `V7_RESOLVE`, and into `v7_threading.def`.
- **Harness**: `hello_inference.exe` gained `-r <rounds>`, a burn-in loop where
  each round resets the block, runs `v7s_chat` with `reset=1` and prints a
  `[perf]` report. The TSC is calibrated once via QPC plus Sleep. Phases print
  as cycles and milliseconds, prefill as ms per token and generation as tok/s.
- **Script**: `bench_qwen.bat [rounds]` runs both F32 and Q8 modes.

## 2. Benchmark results

5 rounds, 200 generated tokens, prompt "What is a cat?"

### F32 mode

| Round | tokenization | prefill ms/tok | generation tok/s | total ms |
|-------|--------------|----------------|------------------|----------|
| 1, cold | 0.158 ms | 22.690 | 23.32 | 8870 |
| 2 | 0.041 ms | 21.802 | 23.82 | 8679 |
| 3 | 0.039 ms | 22.389 | 24.23 | 8545 |
| 4 | 0.040 ms | 21.914 | 23.95 | 8637 |
| 5 | 0.038 ms | 22.079 | 24.08 | 8592 |

### Q8 mode

| Round | tokenization | prefill ms/tok | generation tok/s | total ms |
|-------|--------------|----------------|------------------|----------|
| 1, cold | 0.250 ms | 22.049 | 23.44 | 8820 |
| 2 | 0.047 ms | 21.621 | 23.10 | 8940 |
| 3 | 0.043 ms | 21.883 | 23.03 | 8970 |
| 4 | 0.039 ms | 21.961 | 23.14 | 8928 |
| 5 | 0.038 ms | 22.181 | 23.47 | 8810 |

TSC calibration: 2.2956 GHz, from an invariant TSC.

## 3. Observations

1. **The first round was slower.** Tokenization time fell by about 4.2x in F32
   (0.158 to 0.038 ms) and 6.6x in Q8 (0.250 to 0.038 ms) after round 1.
   First-round code and data cache
   misses account for much of the slowdown. Generation and prefill settled
   within a few percent after round 1. For steady-state numbers, use at least
   three rounds and inspect the last round.
2. **F32 and Q8 are close at generation, around 23-24 tok/s on the idle machine.** An earlier
   2-round run under background load showed Q8 at about 24 tok/s against F32 at
   about 19 tok/s. With the system idle, the recorded modes reach similar token rates. The 0.6B model at batch-1 decode is latency and memory bound, and Q8's
   narrower weights help most under bandwidth pressure, while an idle machine has
   bandwidth to spare. The matmul dispatch benchmarks remain the authority
   for kernel-level Q8 against F32 throughput.
3. **Prefill is about 1.9x cheaper per token than generation** in this run: approximately
   22 ms per prefill token versus 42 ms per generated token. The recorded prompt contained
   about 13 tokens, and the timings should not be interpreted as a 13x speedup.
4. **Determinism**: rounds within a mode, and across modes, produce identical
   text. The sampler runs greedy at these settings, so timing noise never
   changes the decode path.
5. **The vision slots** at indices 6 through 9 stay 0 in text-only rounds and
   are populated only by `v7s_chat_image_mem`, which records the ViT encode
   duration and that turn's token counts. This was verified by code inspection,
   and the image-turn smoke test is covered by the existing vision harnesses.

## 4. Interpretation notes

- All timestamps are raw RDTSC cycles written by `lfence`-serialized probes.
  Convert them with the calibrated TSC frequency, which is printed at startup.
- Phase durations are differences of consecutive timestamps:
  `tokenization = [1]-[0]`, `prefill = [2]-[1]`, `generation = [3]-[2]`.
- `number_token_generate` counts tokens produced after the prompt, up to
  `max_gen`, and `number_token_prefill` counts encoded prompt tokens.
- Each round must reset the block from C, which the harness does, because the
  assembly probes only write and never clear.
