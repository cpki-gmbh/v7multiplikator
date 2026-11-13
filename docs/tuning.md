# Dispatch and Runtime Tuning

V7 distributes selected native operations across CPU threads and allows execution parameters to vary
with the matrix shape. The dispatcher and in-process tuner provide ways to evaluate work partitioning,
batch sizes, and prefetch settings on the processor that runs the model. Applications can use the
supplied configuration without changing these implementation details directly.

This manual is primarily for performance work. It describes the configuration file, the mapping between
tuning fields and runtime slots, the measurement tools, and the recorded experiments. A configuration
that performs well on one CPU or model is not necessarily optimal elsewhere, so every comparison should
identify the processor, workload, precision, and measurement procedure.

## Contents

- [Dispatch and tuning](#dispatch-and-tuning)
- [Tuner field and slot layout](#tuner-field-and-slot-layout)

---

## Dispatch and tuning

The dispatcher divides selected native operations into jobs that can be processed by the configured CPU
threads. Execution parameters, including job batching and prefetch settings, may be selected separately
for different matrix shapes. This chapter explains how the configuration file represents those choices
and how the associated tools evaluate them. The recorded settings describe particular runs; they are not
prescribed values for every processor.

### Overview

The `threading_proxy.ini` file describes how V7 divides selected operations among CPU threads and how
its kernels use configurable prefetch instructions. These settings can affect short and long matrix
operations differently. For instance, `1024x1024` and `2048x1024` matrices have different work
distributions and memory-access patterns, so a scheduling policy that is effective for one does not have
to be effective for the other. V7 allows an explicitly listed matrix shape to have its own configuration
rather than assuming a single parameter set will work equally well for all operations.

The configuration has three levels. The `[proxy]` section contains settings that apply to the process
and its worker threads. The `[default]` section can document baseline values for configuration authoring,
but it is not a runtime fallback for an unspecified shape. Every shape that the runtime
dispatches must have its own complete `[NxD]` section, such as `[1024x1024]`;
an absent per-shape block is a fatal configuration error. The tables below document these settings, and the later tuning sections explain how
measurements can be used to compare configurations on a particular processor.

### File structure

```ini
; Global settings (worker count, priority, global prefetch defaults)
[proxy]
; Baseline reference and supported_resolutions list (not a runtime fallback)
[default]
; Required complete per-shape dispatch and prefetch configuration
[1024x512]
; Another per-shape section
[1024x1024]
; Another per-shape section
[1024x2048]
...
```

#### Each supported shape requires its own complete configuration

Each matrix shape in `supported_resolutions` must have a corresponding `[NxD]`
section with all required dispatch and prefetch values. The runtime selects that
section for matching dimensions. It does not inherit missing parameters from a
default dispatch block or silently fall back to another shape.

#### Global keys belong in the initialization configuration

The processor-selection and worker-priority settings are established during initialization, before any
particular matrix shape has been selected. The runtime therefore establishes `cores_subtract`, `worker_priority` and
`use_physical_cores` at initialization rather than using per-dispatch changes from an
`[NxD]` section. Their source and any documented database overrides must agree
with the initialization configuration. Changes to these settings require initialization with the updated
configuration.

### Global Settings ([proxy])

| Parameter | Shipped INI value / builder default | Description |
| --- | --- | --- |
| `cores_subtract` | 1 (shipped INI; builder fallback 2) | Number of logical-processor positions excluded from worker placement; the caller is handled separately. Reducing this value can allow more workers. |
| `use_physical_cores` | 1 in shipped INI | 0 = all logical processors (SMT/hyperthreading), 1 = physical cores only |
| `worker_priority` | 15 | Worker thread priority as a Win32 priority value (15 = TIME_CRITICAL). Read from the `[proxy]` section at init (`v7_init.inc` step 1a-proxy), lowering it can help when inference competes with a busy foreground application. |

#### Simple worker pool (`init:simple_threads`)

`init:simple_threads` is a database key, not an INI key. It is written by
`build_db.exe -simple-threads <n>` (default 0) and read once at worker-pool creation.

| Value | Behavior |
| --- | --- |
| `0` (default) | Full placement pool: topology scan, SMT-aware or physical-core placement, every worker pinned to its own logical processor, caller pinned to CPU 0. |
| `N > 0` | N plain worker threads: `CreateThread` + `SetThreadPriority` only. No pinning, no SMT awareness, no topology scan. `num_chunks` is computed from the N workers + caller exactly as in placement mode, and `worker_priority` (INI value, `v7_override_prio` env override, `v7_set_worker_priority`) applies unchanged. |

Values above `MAX_WORKERS` (63) are rejected with a FATAL at init, and a database
without the key is stale -- rebuild it with the current `build_db.exe`.


### Dispatch Parameters

Dispatch parameters control how matrix work is divided and assigned to the available threads. The
effective values may depend on the matrix dimensions and the selected configuration section.

#### Chunk Count

| Parameter | `[default]` INI value | Description |
| --- | --- | --- |
| `num_chunks` | 32 in `[default]` | Number of work chunks per matmul dispatch. More chunks = finer work distribution = more overhead. Fewer = less overhead but less parallelism for small matrices. |
| `min_chunk_cols` | 4 in `[default]` | Minimum columns per chunk. Prevents excessive fragmentation on small inner dimensions. |
| `chunk_multiplier` | 16 | Auto-calculated chunk count when `num_chunks=0`. |
| `chunk_split_pct` | 0 | Percentage of chunks to split into a separate group (0=disabled). |
| `chunk_group_1_count` | 16 | Chunks in group 1 when `chunk_split_pct > 0`. |
| `chunk_group_2_count` | 128 | Chunks in group 2 when `chunk_split_pct > 0`. |
| `quantize_chunk_groups` | 4 | Max GS=64 groups per quantize job chunk (0=auto). |

#### Batch Sizes (Worker Job Acquisition)

Worker threads obtain batches of pending jobs using `lock xadd`. The size of each batch depends on the
number of jobs remaining, which allows the dispatcher to vary acquisition overhead during a workload:

| Parameter | Default | Description |
| --- | --- | --- |
| `batch_small` | 2 | Worker batch size when < `batch_med_thresh` jobs remain |
| `batch_medium` | 4 | Worker batch size when `batch_med_thresh` <= jobs < `batch_large_thresh` |
| `batch_large` | 8 | Worker batch size when `batch_large_thresh` <= jobs < `batch_huge_thresh` |
| `batch_huge` | 16 | Worker batch size when >= `batch_huge_thresh` jobs remain |
| `batch_med_thresh` | 4 | Transition threshold: small -> medium |
| `batch_large_thresh` | 8 | Transition threshold: medium -> large |
| `batch_huge_thresh` | 16 | Transition threshold: large -> huge |

#### Caller Batch Sizes

The thread that initiates a dispatch also participates in the computation. It has its own batch-size
thresholds because its scheduling requirements may differ from those of the worker threads. The
following table defines the caller-side settings; they are evaluated alongside the worker settings for
each dispatch.

| Parameter | Default | Description |
| --- | --- | --- |
| `caller_batch_small` | 1 | Caller batch for small workloads |
| `caller_batch_medium` | 2 | Caller batch for medium workloads |
| `caller_batch_large` | 4 | Caller batch for large workloads |
| `caller_batch_huge` | 8 | Caller batch for huge workloads |
| `caller_batch_med_thresh` | 4 | Caller small->medium threshold |
| `caller_batch_large_thresh` | 8 | Caller medium->large threshold |
| `caller_batch_huge_thresh` | 16 | Caller large->huge threshold |

### Prefetch Tuning

Each matmul kernel has multiple prefetch slots. Each slot has a **distance** (how many bytes ahead to
prefetch) and a **type** (which cache level to prefetch into).

Many quantized matrix multiplication workloads are limited by the rate at which weights can be supplied
to the execution units. Prefetch settings can therefore affect performance, especially when the useful
cache distance varies with the matrix shape and processor. The values stored in the INI file specify
which prefetch operations are enabled and how far ahead they request data. Their effects must be
measured on the target workload; a larger distance is not automatically better.

#### Prefetch Types

Prefetch type names describe the instructions available to the tuner, not a promise that every candidate
has a distinct microarchitectural effect. In particular, `prefetchwt1` and `cldemote` should not be
credited with a performance effect on Zen 3 without measurements establishing their behavior there. The
tuning results are specific to the CPU and code version under test. No patching-disabled ablation is
included in these tables.

| Value | Instruction | Description |
| --- | --- | --- |
| 0 | `prefetchnta` | Non-temporal (NTA), fetch into L1 without polluting L2/L3. Meant for data used once. |
| 1 | `prefetcht0` | Temporal level 0, fetch into all cache levels. Meant for data that is reused. |
| 2 | `prefetcht1` | Temporal level 1, fetch into L2 and higher. |
| 3 | `prefetcht2` | Temporal level 2, fetch into L3 and higher. |
| 4 | `prefetchw` | Prefetch for write ownership. |
| 5 | `prefetchwt1` | Prefetch write, T1 level (Xeon-oriented; rarely useful on desktop parts). |
| 6 | `cldemote` | Hint to demote a line from cache (supported on a subset of CPUs). |

A distance below 32 patches the slot to a NOP (prefetch disabled), a disabled slot's type can no longer
be patched until the distance is restored.

#### Prefetch Slots (41 total)

A prefetch slot identifies a patch point associated with a particular kernel operation. The current
implementation uses slots 0 through 40, and `db_build_tool.c` writes 41 `pf_dist_*` keys and 41
`pf_type_*` keys for each configuration section. Slots 41 through 48 are outside the active layout and
do not participate in tuning. The table identifies the operations associated with the active slots.

| Slot Range | Kernel | Usage |
| --- | --- | --- |
| 0-7 | Q8 matmul | Int8 weight row prefetch (4 rows x 2 slots per row) |
| 8-10 | F32 matmul | Activation + weight prefetch |
| 11-12 | Fused F32 | Weight + activation prefetch |
| 13-18 | SwiGLU F32 | Gate/up weight prefetch (single-pass) |
| 19-26 | SwiGLU F32 dual | Gate/up weight prefetch (dual-pass) |
| 27-30 | SwiGLU Q8 | Gate/up int8 weight prefetch |
| 31-38 | SwiGLU Q8 dual | Gate/up int8 weight prefetch (dual-pass) |
| 39-40 | Attention | Key/value prefetch (**type-only**, these slots have no patchable displacement, so `pf_dist_39/40` have no effect; only `pf_type_39/40` applies) |

#### Tuning workflow

For a manual tuning session, begin by running `bench_regression.bat` and recording the bandwidth of
every supported matrix shape. Select one shape and change a single `pf_dist_<N>` or `pf_type_<N>` value
in its configuration section. Measure that shape again before making another adjustment, so the effect
of each parameter remains identifiable.

The recorded manual procedure uses a 3% improvement as its minimum threshold for accepting a change and
marks accepted configuration updates with the `tune:` prefix. A reduction of 5% or more is treated as a
regression, while a difference within the ±5% range leads to one additional measurement; if the result
is still inconclusive, the original setting is restored. These thresholds describe that manual workflow,
not a universal statistical model or the acceptance contract of the current in-process autotuner.
Complete the evaluation of one parameter before moving on to another.

#### Common Prefetch Distance Values

| Distance | Meaning |
| --- | --- |
| 512 | ~1 cache line (64 bytes) x 8 rows of int8 weights |
| 576 | ~9 cache lines ahead |
| 640 | ~10 cache lines ahead (1 F32 block) |
| 2560 | ~40 cache lines ahead (5 F32 blocks) |
| 3072 | ~48 cache lines ahead (6 F32 blocks) |

### Supported Resolutions

The `supported_resolutions` entry in `[default]`, together with continuation keys such as
`supported_resolutions_2`, enumerates the supported shape sections. The shipped
`threading_proxy.ini` includes many models and extra shapes; the following list is only
a Qwen3-0.6B subset, **not** a complete replacement for the shipped keys:

```ini
; illustrative subset — retain the remaining shapes and continuation keys
supported_resolutions=1024x512 1024x1024 1024x2048 2048x1024 1024x3072 3072x1024 151936x1024 1024x151936
```

For Qwen3-0.6B these map to:

| Shape | Kernel | Usage |
| --- | --- | --- |
| 1024x512 | SMALL | K/V projection (head_dim x kv_dim) |
| 1024x1024 | MEDIUM | Q/O projection (dim x dim) |
| 1024x2048 | LARGE | Wide matmul (dim x 2xdim, general INI shape) |
| 2048x1024 | XLARGE | Tall matmul (2xdim x dim, general INI shape) |
| 1024x3072 | FFN up/gate | FFN gate and up projections |
| 3072x1024 | FFN down | FFN down projection |
| 151936x1024 | Embedding | Token embedding lookup |
| 1024x151936 | Classifier | Output classifier (logits) |

### Benchmarking

Record the processor, model, matrix shape, and dispatch configuration with each kernel benchmark.
For full-model decode rates, see the [CPU inference benchmark](engine_benchmark.md). The
Phi-3-mini retests show how active tuner measurements can affect the reported rate. Record the
tuner's measurement state alongside the throughput.

```bat
:: Full regression benchmark (all shapes, Q8 + F32)
bench_regression.bat

:: Single kernel benchmark
tools\b_benchmark.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db all --time 3

:: Load test with worker spin
tools\b_loadtest.exe bin\v7_threading.dll --db weights\databases\qwen3_0.6b\lz4ht.db --q8 --iterations 1000
```

#### Recorded kernel throughput (Ryzen 5 7430U, 12 logical processors)

These are effective per-kernel throughput measurements, not DRAM-bandwidth measurements or end-to-end
token-generation rates. A 1024 × 1024 Q8 matrix occupies approximately 1 MiB plus scales and fits in the
Ryzen 5 7430U's 16 MiB L3 cache. Repeated processing can therefore report hundreds of GB/s through cache
reuse even though DDR4-3200 dual-channel theoretical bandwidth is 51.2 GB/s. These figures cannot be
used to establish the attainable throughput of models whose full weight set resides outside the
last-level cache.

| Kernel | Shape | Q8 (GB/s) | F32 (GB/s) |
| --- | --- | --- | --- |
| Matmul | 1024x1024 | 382-426 | 176-197 |
| Matmul | 1024x2048 | 295-336 | 136-161 |
| Matmul | 2048x1024 | 280-320 | 130-155 |

If the cache-resident Q8 1024×1024 microbenchmark falls below the recorded 300 GB/s threshold, check the
conditions of the same test:
- CPU% (should be >1000% = all 12 cores active)
- `cores_subtract` (try reducing to 1)
- `num_chunks` (try halving/doubling)
- Thermal throttling (laptops throttle under sustained load)

---

### Recorded tuning session: August 2026 (Qwen3-0.6B)

This section is a historical record of one tuning session. It is kept for reference: the tables report
what was measured then, and the commands describe the tooling as it stood at that time.

The session covered all eight Qwen3-0.6B matmul shapes, wrote the results back to `threading_proxy.ini`,
and validated them with an interleaved loadtest against the baseline config.

Author: opencode tuning session. Model: `weights\qwen3_0.6b\qwen3_0.6b.bin` (Config: dim=1024,
hidden=3072, layers=28, heads=16, kv=8, head_dim=128, vocab=151936, seq=40960, GS=64, qk_norm=1,
rope_base=1e6).

#### 1. Scope

Tune all eight Qwen3-0.6B matmul shapes with `tools\autotune_dispatch.exe`, write the tuned values back
into `threading_proxy.ini`, rebuild fresh DBs, and validate with an interleaved `b_loadtest` comparison
against the pristine ("baseline") config. The deliverable was a per-shape config in the repo's
`threading_proxy.ini` that did not regress any shape.

The following table identifies the eight matrix shapes measured in this session and the corresponding
operations in the model:

| Shape | Model role |
| --- | --- |
| 1024x512 | K/V projection, RMSNorm fusions |
| 1024x1024 | Q projection, attention output |
| 1024x2048 | Q projection (larger batch) |
| 2048x1024 | O projection |
| 1024x3072 | FFN up / gate |
| 3072x1024 | FFN down |
| 151936x1024 | embedding lookup (matmul against vocab) |
| 1024x151936 | classifier / output |

#### 2. Execution-fault cause and the C-side fix

Tuning was blocked by a deterministic access violation in the Q8 path at `1024:1024` during the
autotuner's refinement phase. The cause was a routing bug in `src\asm\matmul_dispatch.inc` in the
dispatch shard layout:

- The ASM code stored a raw percentage into `mm_chunk_split_pct`. When the
  autotuner's candidate drove `chunk_split_pct > 1`, the split-fill jobs were
  routed back into the uniform fill loop, emitting garbage `jobs_arr[]` job
  pointers (columns beyond the real `n`). The kernel then dereferenced those
  out-of-bounds pointers, giving an access violation in the q8 kernel.
- This only happened for candidate values the production pipeline never uses
  (production always has `chunk_split_pct` in `[0,1]`), so no validator caught
  it.

The recorded correction was implemented in `sanitize_snapshot()` in `tools\autotune_dispatch.c`. No
assembly changes were made during that tuning session:

```c
values[17] = 0;   /* chunk_split_pct: force 0, ASM split-fill routing bug */
```

`chunk_split_pct = 0` makes every fill take the correct uniform/taper path, so the tensor pointers stay
valid for all 22 dispatch parameters. The DLL (`bin\v7_threading.dll`) was not rebuilt; the workaround
is implemented entirely in the tool.

An earlier theory was that the DLL overwrote caller-side stack tensor structs. Moving those structs to
file scope did not change the crash, so the theory did not hold: the DLL reads the weight pointer out of
the job struct, so a caller-side storage move has no effect.

#### 3. INI write-back bug with a relative path

`--ini threading_proxy.ini` write-back of the best candidate failed with `(WRITE FAILED)` while reads
worked. The cause was that `WritePrivateProfileStringA` rejects a relative path with ERROR_ACCESS_DENIED
(err 5), whereas `GetPrivateProfileStringA` accepts one.

Fix in `autotune_dispatch.c`: immediately after argument parsing the `--ini` value is normalized to an
absolute path via `GetFullPathNameA` into a static buffer, and an invalid or oversized path aborts with
a FAIL exit before any tuning starts.

#### 4. Autotuner behavior

- The 82 prefetch keys (`pf_dist_0..40`, `pf_type_0..40`) are tuned across
  the whole parameter space: all 41 slots of the shape block's pf arrays are
  probe-drawn, mutated, and refined (the INI's `pf_dist_41..48` lines are
  inactive keys associated with the earlier fused-Q8 kernel and are left untouched). The
  valid distance interval is data-driven: every distance from
  -(weight-matrix bytes / 2) to +(weight-matrix bytes / 2) is valid.
  Negative distances are real backward prefetches; the unsigned [0..31] band
  (e.g. the INI's `pf_dist_8=0`) is the engine's disabled-slot regime and a
  valid committed state, in-window committed values are never rewritten.
  Types are tuned in [0..6]. Because the dispatcher's level-2 prefetch cache
  only re-patches on a shape switch, the autotuner calls the exported
  `set_prefetch_distance`/`set_prefetch_type` directly for every changed slot
  (distance first, then a type re-patch to re-establish the tuned opcode
  after the distance patcher's backup restore).
- All 22 dispatch params are tuned (indices 0-21). `sanitize_snapshot()` keeps
  worker/caller batch thresholds ascending. `min_chunk_cols` carries no
  power-of-two requirement: the engine picks mask-vs-div AlignDown at runtime
  (matmul_dispatch.inc `.taper_fill_common`), so any value in [4..64] is
  valid (`>= 4` matters only when the taper runs).
- Random draws use the x86 RDRAND instruction (`_rdrand32_step`, no seeding),
  for tuning draws and tensor fill alike.
- The param_set shadow copy into the matched shape block is written through a
  per-index offset map: the param_set layout is not `index*4`
  (`chunk_multiplier` sits at offset 36, the caller-batch group starts at 52),
  and `taper_shift` (20) / `quantize_chunk_groups` (21) are standalone BSS
  globals with no param_set slot (written only via `v7s_set_dispatch_param`).
- The autotuner's internal "baseline" is synthetic uniform (taper_shift=0), not
  the pristine INI (taper_shift=48). The autotuner always writes back its best
  candidate regardless of decision, so an "improvement over uniform" can still
  be a regression versus the pristine config. The interleaved loadtest
  (tuned DB vs pristine baseline DB) is therefore the authoritative measure.

#### 5. Tuning methodology

- One shape per autotuner invocation; `--time 0.3` (small shapes) / `--time 0.5`
  (vocab shapes) calibrated measurement, 30 rounds (20 for the two vocab
  shapes), Q8 path.
- The first attempt used `--iterations 1000`, but that window is only ~5 ms for
  the fast shapes, which made the results very noisy (e.g. 1024x512 measured
  629 vs 691 GB/s across adjacent runs). The `--time <sec>` calibrated mode
  measures a fixed wall time (~0.5 s) and gave stable numbers; run 1 was
  discarded.
- Full 8-shape run saved at
  `%LOCALAPPDATA%\opencode\tool-output\tool_fc5ee0cef001m7eI2dMHnxLqMy`
  (all 8 result blocks, one line per param per round).
- `tune_all_shapes.bat` is resumable (arg `%1` = start shape 1-8) and writes
  each shape's `[NxD]` section to `threading_proxy.ini` on completion.

#### 6. Per-shape tuned parameters (what was written to `threading_proxy.ini`)

Format: `num_chunks, batch_small/medium/large/huge, thresholds, cm=chunk_multiplier, mcc=min_chunk_cols,
caller_batch s/m/l/h, caller thresholds, split=chunk_split_pct, groups quantize, taper=taper_shift,
quant=quantize_chunk_groups` .

| Shape | Params |
| --- | --- |
| 1024x512 | num_chunks=1532, batch 9/10/5/42, thr 569/2962/3402, cm=80, mcc=8, caller 18/6/40/46, cthr 3352/3353/3354, split=0, grp 61/93, taper=1, quant=12 |
| 1024x1024 | num_chunks=22, batch 27/22/37/59, thr 3622/3623/3624, cm=38, mcc=1, caller 26/12/48/39, cthr 2926/2927/2928, split=0, grp 194/50, taper=32, quant=56 |
| 1024x2048 | num_chunks=1633, batch 55/6/47/51, thr 3779/3780/3783, cm=98, mcc=32, caller 37/54/55/47, cthr 3325/3327/3328, split=0, grp 31/137, taper=26, quant=13 |
| 2048x1024 | num_chunks=2903, batch 1/60/7/46, thr 3708/3709/3710, cm=13, mcc=64, caller 45/38/3/47, cthr 2235/3702/3703, split=0, grp 5/134, taper=29, quant=47 |
| 1024x3072 | num_chunks=2218, batch 3/33/32/29, thr 2598/2599/2600, cm=56, mcc=4, caller 17/46/40/17, cthr 310/961/2531, split=0, grp 122/14, taper=9, quant=17 |
| 3072x1024 | num_chunks=3399, batch 53/3/17/39, thr 2071/2072/2073, cm=1, mcc=2, caller 7/16/15/18, cthr 289/1511/2517, split=0, grp 106/124, taper=22, quant=60 |
| 151936x1024 | num_chunks=2566, batch 33/16/25/21, thr 1537/3357/3642, cm=17, mcc=64, caller 64/38/60/44, cthr 1191/2258/2259, split=0, grp 91/188, taper=66, quant=20 |
| 1024x151936 | num_chunks=33, batch 47/2/20/37, thr 355/3137/3143, cm=106, mcc=16, caller 25/53/60/56, cthr 2511/2513/3427, split=0, grp 9/42, taper=11, quant=16 |

#### 7. Interleaved loadtest: tuned DB vs baseline (pristine) DB

Method: `tools\b_loadtest.exe bin\v7_threading.dll --db <db> --q8 N:D --time 0.5` with `set_spin(1)`
(production-mode), 4 warmup runs for the first shape, then alternating 10 s pause + tuned / + baseline
per shape. Two full passes, to filter laptop thermal and power noise (10-25% run-to-run). GB/s:

| Shape | Pass A T/B | Pass B T/B | Mean T/B | Decision |
| --- | --- | --- | --- | --- |
| 1024x512 | 410.1 / 433.0 | 418.1 / 425.3 | 414 / 429 | baseline -3.5% |
| 1024x1024 | 310.5 / 314.8 | 310.5 / 317.1 | 310 / 316 | baseline -1.7% |
| 1024x2048 | 352.2 / 296.7 | 388.3 / 299.5 | 370 / 298 | tuned +24% |
| 2048x1024 | 376.7 / 324.1 | 383.8 / 330.9 | 380 / 328 | tuned +16% |
| 1024x3072 | 291.9 / 297.4 | 259.7 / 292.6 | 276 / 295 | baseline -6.5% |
| 3072x1024 | 322.4 / 321.5 | 322.8 / 324.7 | 323 / 323 | tie |
| 151936x1024 | 136.3 / 139.6 | 143.2 / 140.9 | 140 / 140 | tie |
| 1024x151936 | 128.2 / 101.9 | 129.4 / 94.6 | 129 / 98 | tuned +31% |

The measurements support the following observations for the specified workload and processor:

- The tuned config is faster on the Q projection (1024x2048, +24%), the O
  projection (2048x1024, +16%) and the classifier/output (1024x151936, +31%).
  These are the memory-bandwidth-bound shapes, where aggressive chunking helps
  most.
- The tuned config is slower or tied on the other five (K/V 1024x512, Q
  1024x1024, FFN up/down, embedding 151936x1024). The autotuner's +6..+19%
  gains over its synthetic uniform baseline do not hold against the pristine
  config (taper_shift=48): they were measured against a weaker reference.

#### 8. Decision on which config to keep

The final configuration retained the measured improvements only for matrix shapes where the tuned
parameters were faster. For other shapes, it restored the baseline values:

| Shape | Source in the merged config |
| --- | --- |
| 1024x2048, 2048x1024, 1024x151936 | tuned |
| 1024x512, 1024x1024, 1024x3072, 3072x1024, 151936x1024 | pristine |
| proxy, default | untouched (identical in both files) |

`tuning\build_bestknown.ps1` splices `tuning\threading_proxy.ini.tuned` and
`tuning\threading_proxy.ini.orig` into `tuning\threading_proxy.ini.bestknown` (1229 lines, all 10
sections; `[1024x2048]/[2048x1024]/[1024x151936]` carry the tuned `num_chunks` values 1633/2903/33, the
five reverted sections carry the pristine values, verified per-section). The repo's
`threading_proxy.ini` is now this merged config.

#### 9. Confirmation loadtest: merged DB vs baseline DB

Rebuilt `tuning\lz4ht_bestknown.db` fresh from the merged INI and ran the same interleaved pattern once
(`build_logs\loadtest_bestknown_vs_baseline.bat`). GB/s:

| Shape | Merged | Baseline | Note |
| --- | --- | --- | --- |
| 1024x512 | 414.6 | 421.2 | same config, noise (-1.6%) |
| 1024x1024 | 293.7 | 314.4 | same config, noise (-6.6%) |
| 1024x2048 | 372.8 | 310.9 | tuned value, +20% |
| 2048x1024 | 390.3 | 317.1 | tuned value, +23% |
| 1024x3072 | 294.2 | 297.6 | same config, tie (-1%) |
| 3072x1024 | 309.0 | 330.3 | same config, noise (-6.5%) |
| 151936x1024 | 141.7 | 152.6 | same config, noise (-7%) |
| 1024x151936 | 130.5 | 98.5 | tuned value, +32% |

The three tuned shapes reproduce their gains (+20/+23/+32%). The five reverted shapes now run the
identical config as the baseline DB, so their residual differences (-1%..-7%) are run-to-run laptop
thermal and power noise (10-25% band), not a regression introduced by the decision. The merged config is
therefore verified: no shape is slower than baseline, and three shapes are measurably faster.

#### 10. Net effect and files

Merged vs pristine across all eight shapes (weighted by how often each shape runs per token, the Q/K/V/O
projections and FFN dominate the per-token forward, embedding and output run once per prompt):

- Q projection 1024x2048: +24% (28 layers/token)
- O projection 2048x1024: +16% (28 layers/token)
- Classifier 1024x151936: +31% (once per token)
- All other shapes: unchanged (within noise)

Artifacts:
- `tools\autotune_dispatch.c`, sanitize_snapshot() workaround, per-index
  param_set offset map, rand_s RNG, GetFullPathNameA absolute-path write-back.
- `tune_all_shapes.bat`, resumable 8-shape tuner (`--time` calibrated).
- `run_autotune.bat`, 1024:1024 repro pointing at `tuning\lz4ht_baseline.db`.
- `loadtest_tuned_vs_baseline.bat`, interleaved tuned vs pristine (2 passes).
- `threading_proxy.ini`, the merged config.
- `tuning\threading_proxy.ini.{orig,tuned,bestknown}`, pristine / full-tuned /
  merged; `tuning\build_bestknown.ps1`, the merge script; DBs in `tuning\`
  (per-model artifacts, gitignored).

Commit: `b3ee0ce` (progress), then the analysis/final commit.

## Tuner field and slot layout

The in-process tuner mirrors the configurable execution parameters in a fixed set
of fields mapped to runtime slots. This reference records the binary mapping
shared by the configuration builder, tuner, assembly code and state-file
consumers. It is intended for maintainers and performance experiments; ordinary
callers of the public inference API do not need to modify these fields.

This document is the single reference for the 106 tunable fields the in-process tuner mirrors in
`tuner_state.txt`, and how they map onto the engine's per-shape `param_set` slots and the
`threading_proxy.ini` keys. Every consumer of the mirror (the dev promotion script
`tools/update_ini_from_state.py`, the C autotuner `tools/autotune_dispatch.c`) must agree with this
table.

### The 106 fields

The tuner stores each parameter set in the following mirror-record format:

```
cursor <index>
cfg <NxD> <f0> <f1> ... <f105>
```

Field count per record: 24 param_set dwords + 82 prefetch slots = **106**. The mirror's verdict scripts
write the baseline by walking the FULL 24-dword param_set (all slots, including the two non-tunable
dwords), so the record's first 24 fields are the block layout verbatim.

### Field-to-slot map (the 24 param_set dwords)

The engine's per-shape block layout (`src/asm/v7_data.inc`, `PARAM_OFFSET_*`) is 24 dwords = 96 bytes
(`SHAPE_BLOCK_PARAM_OFF`). The mirror's fields 0..23 hold these slots IN SLOT ORDER:

| mirror field | param_set slot | INI key | byte offset | note |
| ---: | ---: | --- | ---: | --- |
| f0 | 0 | `num_chunks` | 0 | jobs the fill loop splits D into (0 = auto: workers+1 * chunk_multiplier) |
| f1 | 1 | `taper_shift` | 4 | 0=off; else decay exponent k (1/16 steps) for the taper fill |
| f2 | 2 | `batch_small` | 8 | worker batch size for tiny job counts |
| f3 | 3 | `batch_medium` | 12 | worker batch size (mid tier) |
| f4 | 4 | `batch_large` | 16 | worker batch size (large tier) |
| f5 | 5 | `batch_huge` | 20 | worker batch size (biggest tier) |
| f6 | 6 | `batch_med_thresh` | 24 | job count >= this uses batch_medium |
| f7 | 7 | `batch_large_thresh` | 28 | job count >= this uses batch_large |
| f8 | 8 | `batch_huge_thresh` | 32 | job count >= this uses batch_huge |
| f9 | 9 | `chunk_multiplier` | 36 | multiplier on the auto chunk count (workers+1)*mult |
| f10 | 10 | `min_chunk_cols` | 40 | smallest valid chunk width in columns (caps num_chunks) |
| f11 | 11 | `worker_priority` | 44 | INI-owned, not tuner-tunable; Win32 worker-thread priority |
| f12 | 12 | `quantize_chunk_groups` | 48 | groups per quantize job chunk (0 = automatic, otherwise >= 1) |
| f13 | 13 | `caller_batch_small` | 52 | caller-side batch size for tiny job counts |
| f14 | 14 | `caller_batch_medium` | 56 | caller-side batch size (mid tier) |
| f15 | 15 | `caller_batch_large` | 60 | caller-side batch size (large tier) |
| f16 | 16 | `caller_batch_huge` | 64 | caller-side batch size (biggest tier) |
| f17 | 17 | `caller_batch_med_thresh` | 68 | caller job count >= this uses medium batch |
| f18 | 18 | `caller_batch_large_thresh` | 72 | caller job count >= this uses large batch |
| f19 | 19 | `caller_batch_huge_thresh` | 76 | caller job count >= this uses huge batch |
| f20 | 20 | `chunk_split_pct` | 80 | 0=off; else percent of D given to group 1 (two-phase fill) |
| f21 | 21 | `chunk_group_1_count` | 84 | chunks over group 1 columns (split mode) |
| f22 | 22 | `chunk_group_2_count` | 88 | chunks over group 2 columns (split mode) |
| f23 | 23 | `cores_subtract` | 92 | INI-owned, not tuner-tunable; logical cores excluded from the pool |

Slots 11 and 23 are per-shape block members but NOT tuner-tunable: the autotuner never touches them and
the promotion path never writes them. They stay INI-owned.

### Prefetch fields

```
f24..f64  = pf_dist_0..pf_dist_40
f65..f105 = pf_type_0..pf_type_40
```

The 82 prefetch fields mirror the block's `pf_dist[41]` (block offset 104) and `pf_type[41]` (block
offset 300) arrays verbatim, in slot order. `pf_dist_N` is the prefetch distance in BYTES for SMC slot N
(values < 32 disable that slot's prefetch entirely, the patcher emits a multi-byte NOP); `pf_type_N`
selects the prefetch instruction (0=NTA, 1=T0, 2=T1, 3=T2, 4=PREFETCHW, 5=PREFETCHWT1, 6=CLDEMOTE). The
slot-to-kernel map:

| slots | kernel | what the prefetches cover |
| --- | --- | --- |
| 0-7 | Q8 matmul | weight rows (4 rows x own/next) |
| 8-10 | F32 matmul | activations + row0/row1 weights |
| 11-12 | fused F32 matmul | weight + activation |
| 13-26 | SwiGLU F32 | gate/up weight rows (single/dual modes) |
| 27-38 | SwiGLU Q8 | gate/up weight rows (single/dual modes) |
| 39-40 | attention | key/value streams (type-only patch) |

The current slot layout ends before the former fused-Q8 prefetch entries. Indices 41–48 are not part of
the supported parameter map.

### Consumer contract

- **`tools/update_ini_from_state.py`** maps fields 0..23 through
  the slot table above, SKIPS f11/f23 (INI-owned), and ignores f24..105
  (prefetch tuning stays a live SMC experiment, the INI's
  pf_dist_N/pf_type_N keys keep their committed values unless the C
  autotuner itself writes them).
- **`tools/autotune_dispatch.c`** addresses the same block through its
  dispatch-index abstraction: `dispatch_param_names[i]` gives the INI key
  for dispatch index i, and `sb_param_off_for_idx[i]` gives the block slot
  (note the two out-of-order indices: taper_shift idx 20 -> slot 1,
  quantize_chunk_groups idx 21 -> slot 12). Its OWN mirror/INI write paths
  write in DISPATCH-index order; the LOGO tuner's `tune_accept` Savestate
  writes in SLOT order (it walks the block). Both orders exist in the
  codebase, the LOGO mirror is slot-ordered, the C tool is
  dispatch-ordered. Any new consumer must check which producer it reads.
- **The engine's dispatch** reads the block slots directly through the
  `%define` aliases (`num_chunks`, `taper_shift`, ...) in
  `src/asm/v7_data.inc`; it has no notion of the dispatch index.

## Recorded kernel throughput

The following figures repeat benchmark results for specific matrix shapes. Effective bandwidth is
defined by the benchmark's byte-counting convention divided by elapsed kernel time. It is not equivalent
to physical DRAM bandwidth when data is served from cache. End-to-end prefill and decode performance
should be measured separately with the exact model, quantization, context length, CPU power mode, and
test methodology.

The tables in this chapter preserve measured kernel throughput for the specified shapes and
configurations. Results depend on processor frequency, memory behavior, thread scheduling, and
measurement method, so comparisons should use equivalent conditions. A kernel measurement is also
distinct from end-to-end token throughput, which includes the other operations in the generation loop.

| Kernel | Shape | Q8 (GB/s) | F32 (GB/s) |
| --- | --- | --- | --- |
| Matmul | 1024x1024 | 382-426 | 176-197 |
| Matmul | 1024x2048 | 295-336 | 136-161 |
| Matmul | 2048x1024 | 280-320 | 130-155 |

Measured on Ryzen 5 7430U (12 logical cores), all workers active (>1000% CPU).

---
