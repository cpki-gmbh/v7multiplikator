# v7 evolutionary tuner: decoded population and implementation analysis

**Input:** `v7.zip` (61 `v7_tune_param_<n>_<d>` files) and v7 2.36.0 NASM source.  
**Method:** Static inspection of `src/asm/cycle_tuner_main.inc`, `config_set_random_probe.inc`, `smc_macros.inc`, `v7_data.inc`, and decoding all persisted genome slots. No inference timing or CPU-feature probe was run for this report.

## 1. Actual search design

- There are **61 stored shapes**, each with **128 genomes × 512 bytes = 65,536 bytes**; the sampler additionally has a small tuning file.
- A genome has **22 mutable dispatcher parameters**, **41 signed prefetch displacement fields**, and **41 prefetch-instruction type fields**, or **104 mutable fields**. The 24-dword dispatcher record additionally carries two untuned parameters. A maximum of **20 prefetch slots** are directly installed by `cyc_install_genome` for the currently instrumented Q8 and SwiGLU-Q8 paths (slots 0–7 and 27–38). Do not mistake the 104 editable fields for 104 simultaneously active hardware prefetch operations.
- The population is initialized using independent random parameter configurations. New shapes first receive a one-measurement scan of the non-mutant genomes; the fastest measured configuration becomes the initial delay genome.
- The cycle is **A1 → B1 → C → B2 → A2**. The verdict uses `min(A1,A2)`, `min(B1,B2)`, and C's one observation, with genome exchanges and a champion/delay index. The order reduces some effects of transient load changes but cannot eliminate measurement noise.
- The walk is rank-biased: parent selection squares a uniform fraction (favoring low-list indices); the comparison partner uses a truncated exponential draw. These are designed to favor successful regions while leaving the wider population reachable.
- On each complete 128-position sweep, the configurable tail (default three genomes) is regenerated using **mutation and crossover**, skipping the active champion slot. The mutation engine draws **1–40 field mutation attempts** by default, with replacement. Dispatcher-field mutations use 75% local nudges and 25% redraws; displacement mutations have five modes (redraw, double, halve, signed ratio plus jitter, and positive ratio with randomized sign); type mutations redraw from seven legal instruction types.
- Crossover starts with parent 1, then lets parent 2 donate a randomized number of field positions. At a configured `sex_percent` of 80, the budget is floor(104 × 80 / 100) = 83, but the actual count is drawn from **0 to 82** (average 41), *with replacement*. The expected number of distinct donated fields is approximately **34**, not 80% of 104. This tends to preserve substantial parts of the first parent while recombining other traits.
- Crossover treats the distance and instruction type of one prefetch slot as independent fields, so the pair is not guaranteed to be inherited intact. This is a tradeoff in the genetic operator, not evidence of a bug.
- Only genomes and the sorter cursor are stored in the 64-KiB files; individual measurement histories and per-genome timings are not. The files therefore establish **population structure and convergence**, not exact speedups or causal contributions.

## 2. The striking decoded examples

| Shape | Slot | Observed population | Meaning |
|---|---:|---|---|
| 1024×3072 | SwiGLU-Q8 #34 | **113/128 negative** offsets, median **−283 bytes**; 112/128 are within ±4 KiB | A population strongly enriched for *short backwards* prefetches |
| 2048×1536 | Q8 #5 | **125/128 negative** offsets, median **−717 bytes**; all 10 leading genomes negative | Strong negative-distance convergence in a second, different kernel |
| 1024×3072 | SwiGLU-Q8 #33 | **126/128** offsets within ±4 KiB; median **481 bytes** | Selection favors a short positive displacement in an adjacent slot |
| 2048×1536 | Q8 #2 | **122/128** offsets within ±4 KiB; median **1,963 bytes** | Different short-distance optimum for a different shape |
| 2048×151936 | Q8 slots 0–7 | **93.3%** of 1,024 type entries are `CLDEMOTE` | Large-output matrix develops a different instruction-type distribution |
| 1536×130560 | Q8 slots 0–7 | **93.8%** `CLDEMOTE` | Same pattern for a different large output shape |
| 3072×32064 | Q8 slots 0–7 | **93.1%** `CLDEMOTE` | Repeated, shape-specific type selection |

For `1024×3072` the **20 active slots** contain **39.14%** offsets within ±4 KiB, whereas the uniform random baseline for a half-window of 1,572,864 bytes is about **0.26%**. Some individual slots concentrate more than 98% of their genomes inside this narrow region. In contrast, the reversed `3072×1024` shape has only about 0.2% in the same region.

This is compelling evidence that the evolutionary process has created strong, slot- and shape-dependent structure. **It is not proof that every selected value individually improves runtime:** linked genes can hitchhike with better configurations. The user-reported ablation in which replacing negative displacements with NOPs loses throughput is additional, more direct evidence that negative offsets matter in the tested run.

## 3. Why negative offsets are legitimate

The patcher checks the distance using an **unsigned** comparison with 32. Thus `0..31` is the NOP band, whereas negative signed 32-bit displacements (whose unsigned encodings are large) remain executable. The same unsigned comparison is used in the fast installer. They are not accidental invalid inputs.

A negative displacement is relative to the instruction's addressing register, *not necessarily relative to the next memory location the kernel will access*. Unrolled loops, overlapping rows, earlier accesses, cache-line behavior, and the specific prefetch opcode can all make a backwards displacement useful. It is unsafe to change the search range to nonnegative values or to turn negative distances into NOPs solely because they seem surprising.

## 4. `CLDEMOTE` needs careful interpretation

`CLDEMOTE` has a separate encoding from ordinary prefetch instructions. On processors that do **not** advertise the `CLDEMOTE` CPUID feature, the encoding is architecturally treated as a NOP. Therefore a shape that converges on `CLDEMOTE` may be choosing *no effective cache hint* rather than a useful cache demotion. The stored file cannot determine which behavior occurred: the target machine's CPUID flags are needed.

In particular, the large-output shapes' 93%-plus Q8 `CLDEMOTE` rates should not be presented as proof that actively demoting cache lines improves throughput. They may indicate that actual prefetches harm those paths. The specific form of a NOP can still have different front-end costs from replacing an instruction with another NOP encoding.

## 5. Dimensionality and simulated annealing

The user's prior genetic optimization of a large Markov random field with at least **2,048 parameters** is relevant experience: a 104-field genome is a smaller problem by parameter count, though it can still have a huge, highly coupled and discontinuous search space. Dimensionality alone does not predict difficulty.

This implementation has properties that can outperform a single-chain simulated annealer on the relevant search landscapes: a population retains multiple promising configurations; crossover can combine partial solutions; multi-mode mutation explores both locally and globally; and rank-biased selection concentrates evaluation budget without deleting every alternative. None of this establishes universal dominance over simulated annealing—results depend on objective function, neighborhood design, noise, population size, and compute budget. The user's MRF experience is credible *empirical* evidence for this family of strategies, not a theorem about all optimization tasks.

## 6. Practical implications

1. **Preserve signed distances** and all prefetch opcode options. Negative-distance ablations are meaningful and should not be 'sanitized' away.
2. **Preserve the population sorter, crossover, mutations and persistence**; they are central optimization mechanisms, not incidental complexity.
3. **Judge speed by controlled A/B inference and per-shape timings**, not the magnitudes of saved displacement genes. Per-genome timing is not in these files.
4. **Do not infer that a high `CLDEMOTE` count means cache-line demotion** without checking the target CPU feature flag.
5. For a causal study without changing the engine, compare the currently selected elite genome unmodified with an otherwise identical genome in which **one selected instruction slot at a time** is set to NOP, positive, or negative displacement. Use the existing benchmark workflow, not new build stages or validators.

**Conclusion:** The uploaded populations exhibit nontrivial, strong convergence in both distances and opcode choices, including negative displacements that source inspection confirms are valid. The evolutionary architecture and the saved patterns warrant serious attention. The files alone cannot quantify the runtime gain or prove global optimality; the user's reported ablation results provide the relevant performance evidence.
