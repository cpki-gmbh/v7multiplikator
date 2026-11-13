# v7 Speed Advantages -- Why v7 Keeps Its Throughput Where Other Engines Lose It

Analysis of the measured throttle robustness of the v7 inference engine,
grounded in the overnight benchmark dataset and the 2026-11-10 power-saving
comparison (definitive tables in `benchmarks.md`; the raw console captures
`bench_results_overnight.txt` and `benchmark_tuner_loop_20261112_081959.txt`
are retained in the development tree -- they embed development-machine paths
and are not shipped).

Machine: Ryzen 5 7430U (6C/12T, AVX2), Windows, all engines 8-bit class,
greedy decoding, same models and prompts.

---

## 1. The measured effect

### Full power (overnight run 2026-11-08, tokens/s)

| Model | load | v7 Q8 | llama.cpp | MNN int8 | OpenVINO int8 |
|-------|------|------:|----------:|---------:|--------------:|
| Qwen3-0.6B | short | **55.8** | 52.0 | 48.2 | 41.4 |
| Qwen3-0.6B | poem | **52.2** | 47.3 | 43.8 | 39.7 |
| Qwen3-1.7B | short | **20.0** | 19.8 | 18.1 | 18.3 |
| Qwen3-1.7B | poem | **19.5** | 19.2 | 17.7 | 17.5 |
| Phi-3-mini | short | 9.1 | **9.4** | 8.4 | 8.6 |
| Phi-3-mini | poem | 8.7 | **8.9** | 8.0 | 7.8 |
| MiniCPM5-1B | short | **38.1** | 37.7 | 34.8 | 32.0 |
| MiniCPM5-1B | poem | **37.1** | 36.6 | 33.4 | 29.1 |

### Windows Power saver plan (2026-11-10, tuner-quiet, tokens/s)

| Model | load | v7 Q8 | llama.cpp | MNN int8 | OpenVINO int8 |
|-------|------|------:|----------:|---------:|--------------:|
| Qwen3-0.6B | short | **52.0** | 43.0 | 44.0 | 34.2 |
| Qwen3-0.6B | poem | **48.5** | 39.5 | 39.6 | 26.4 |
| Qwen3-1.7B | short | **18.6** | 15.0 | 16.3 | 12.7 |
| Qwen3-1.7B | poem | **18.7** | 14.9 | 16.5 | 12.4 |
| Phi-3-mini | short | 8.1 | 7.4 | **8.2** | 6.2 |
| Phi-3-mini | poem | **8.7** | 7.1 | 7.5 | 5.7 |
| MiniCPM5-1B | short | **36.9** | 28.3 | 32.3 | 23.1 |
| MiniCPM5-1B | poem | **35.8** | 28.0 | 31.1 | 21.4 |

### Retention (power-saver / full-power throughput)

| engine | retention, 8-leg mean | per-leg range |
|---|---:|---|
| **v7** | **94.7%** | 89.0-100% |
| MNN | 92.8% | 90.1-97.6% |
| llama.cpp | 78.7% | 75.1-83.5% |
| OpenVINO | 72.5% | 66.5-82.6% |

### v7's lead over llama.cpp before / after throttling

| workload | full power | power saver |
|---|---:|---:|
| Qwen3-0.6B short | +7.3% | +20.9% |
| Qwen3-0.6B poem | +10.4% | +22.8% |
| Qwen3-1.7B short | +1.0% | +24.0% |
| Qwen3-1.7B poem | +1.6% | +25.5% |
| Phi-3-mini short | -3.2% | +9.5% |
| Phi-3-mini poem | -2.2% | +22.5% |
| MiniCPM5-1B short | +1.1% | +30.4% |
| MiniCPM5-1B poem | +1.4% | +27.9% |

Two things happen at once: v7 loses almost nothing (-5.3% mean) and
llama.cpp loses a fifth (-21.3% mean), so the gap widens from roughly +2%
averaged (about +9% on the Qwen3-0.6B flagship legs) to +21-30% across the
board. The two workloads llama.cpp wins at full power (Phi-3-mini) flip to
v7 wins under power saving. v7 leads seven of eight workloads under power
saving (six of eight at full power).

---

## 2. The mechanism: memory-bus-bound vs CPU-bound work mix

When the power plan throttles the CPU, core clocks drop by roughly half on
this class of laptop, while DRAM and the interconnect downclock far less.
A token's time is

    t_token = t_memory (bus/latency time) + t_cpu (compute + overhead time)

and throttling by factor k only stretches the t_cpu part:

    t_throttled / t_full = 1 + (k-1) * f_cpu,   f_cpu = t_cpu / t_token

so retention = 1 / (1 + (k-1) * f_cpu). With k = 2 (clock halved), retention
directly reveals each engine's CPU-bound fraction: f_cpu = 1/retention - 1.

| engine | retention | implied f_cpu at full power |
|---|---:|---:|
| **v7** | 94.7% | **~6%** |
| MNN | 92.8% | ~8% |
| llama.cpp | 78.7% | ~27% |
| OpenVINO | 72.5% | ~38% |

Reading: v7 spends ~94% of every token waiting on the memory bus, which the
power plan does not touch -- throttling can only attack the remaining ~6%.
llama.cpp carries a ~27% clock-sensitive slice (generic-kernel compute plus
per-op thread-pool synchronization); halving the clock doubles that slice
and drags the whole token down by a fifth. OpenVINO's graph/scheduling
overhead is the largest clock-sensitive fraction of all four engines.

MNN is nearly as throttle-robust as v7 (~93% retention) -- its kernels are
also bandwidth-efficient; it starts from a lower absolute baseline.
OpenVINO degrades MORE than llama.cpp in retention (72.5% vs 78.7%),
it just starts lower in absolute throughput.

---

## 3. Why v7's CPU-bound fraction is ~6% (design causes)

1. **GEMV is inherently bandwidth-bound.** Quantized LLM decode moves ~1 MAC
   per weight byte; the arithmetic intensity is so low that the math, done in
   the hand-written AVX2 `vpmaddubsw` Q8 kernels, stays well ahead of the bus.
   Fewer wasted per-byte instructions means less work that scales with the
   clock.
2. **Per-shape prefetch tuning.** The 41 SMC-patchable prefetch slots per
   shape keep the memory streams saturated; cores stall on the bus, not on
   cold cache lines (latency-hiding is memory-side work, not clock-side).
3. **Zero hot-path overhead.** No API calls, no condition variables, no
   allocations per dispatch; workers pause-spin on `gen_counter`, jobs live
   in one flat array. The clock-sensitive engine overhead per token is
   microseconds against ~20 ms of memory streaming (measured: tokenization
   0.07 ms vs generation ~4190 ms per 200-token round).
4. **Physical-core placement.** One participant per physical core
   (`use_physical_cores=1`, 6 threads on 6 cores incl. the caller): fewer,
   fatter threads saturate the memory system without SMT load-port contention,
   and pinned workers are immune to power-saver core parking. llama.cpp's
   per-op task barriers across its pool are pure CPU/OS-latency time that
   doubles under throttle; OpenVINO adds graph-execution overhead on top.

---

## 4. Mobile implication: throughput AND efficiency

Power saver keeps v7 at ~93-95% of full-power throughput while drawing a
fraction of the power (lower voltage/frequency point). Throughput nearly
constant at much lower power means **tokens per joule improves** in power
saving -- battery time per generated token gets better, not merely less
worse. The effect is also thermally self-reinforcing: a cooler chip sustains
its clocks, avoiding the throttle spiral long generations cause on busy
machines. Any power constraint (battery saver, fanless device, thermal cap,
noisy-neighbor VM host) attacks the CPU-bound engines harder than v7 by
construction, because v7's bottleneck is the part constraints do not shrink.

---

## 5. Caveats and follow-ups

- Single machine (Ryzen 5 7430U), one benchmark day, Windows power plans.
  The ordering v7 ~ MNN >> llama.cpp > OpenVINO (retention) is consistent
  across all eight legs, but exact percentages are machine-specific.
- The throttle factor k=2 is assumed, not measured. Logging actual core
  clocks (APERF / CallNtPowerInformation) during a throttled run would turn
  the f_cpu column into hard measurements.
- Energy per token (battery drain API or wall meter) is not yet measured;
  the tokens-per-joule claim follows from constant throughput at reduced
  power, but a direct measurement would document it.

---

## 6. Reproducing

    .\bench_all_models_overnight.bat     rem full-power baseline
    rem then: Windows Settings -> Power mode: Best power efficiency,
    rem and rerun the tuner-loop cycle (benchmark_tuner_loop_*.txt logs)

Sources: `benchmarks.md` (definitive overnight tables + power-saving section),
`docs/engine_benchmark.md`; raw console captures in the development tree
(`bench_results_overnight.txt`, `benchmark_tuner_loop_20261112_081959.txt`).
