# speed_fix results — four-token Q8 batch kernel

This report records the implementation of `speed_fix.md`, the prefill optimization plan. The isolated kernel, validator suite and kernel-level benchmark were completed,
and all validations passed. The dispatched row-sliced
batch-4 mode (plan Phase 2.3) followed, and the packed-QKV prefill integration is complete, validated,
measured, and committed (`step 1 worked`): Q8 text prefill is 1.26–1.81x faster end-to-end on
17/80/142-token prompts, with byte-identical greedy output. Raw per-round measurements:
`next_phase/measurements/qkv_milestone.csv`. The follow-up batch-WO prefill experiment
(`next_phase/AGENT_NEXT_PREFILL_TODO.md` Phase 1) was measured and REVERTED — see the WO section below.

## What was built

- `src/q8_batch4.h` — flat 120-byte argument block (`v7_q8_batch4_args`).
  `#pragma once` per house rules; the plan's `#ifndef` guard and
  `static_assert` offsets were omitted (layout is mirrored by the Q8B_*
  `%define`s in the kernel; the validator battery proves it at runtime).
- `src/asm/q8_batch4_kernel.inc` — standalone single-thread four-token Q8
  kernel: one weight row load feeds four independent per-token accumulators;
  arithmetic identical to `COMPUTE_Q8_ROW` (vpsignb + vpmaddubsw +
  vpmaddwd + vpaddd), group size 64, per-group float scales, HADD_YMM
  reduction. `SF_*` frame macros, xmm6-15 low halves saved (xmm12 pair
  dropped: the kernel never touches ymm12). No SMC, no dispatcher, no
  global state.
- `src/asm/v7.asm` — one include line after `v7_kernels.inc`.
- `src/v7_threading.def` — experimental export `v7s_q8_batch4` (validator
  use only, not part of the public `v7_lib.h` ABI).
- `tools/v_matmul_validate.cpp` — test battery + kernel benchmark.

## Reviewer verification (second-opinion review, no defects found)

Verified correct: struct offsets (OUT 0..31, XQ 32..63, XS 64..95, WQ 96, WS 104, N 112, D 116, OUT1..3
= 8/16/24); vpmaddubsw operand order (unsigned |x| first, signed weights second); 64-bit row offsets
(row*n,
>>4 = row*(n/64)*4 scale bytes); ecx/r15d sync with zero-extending writes;
ones-vector construction (pcmpeqd + pabsw); per-group int32 sum ≤ 64·127² ≈ 1.03M < 2²⁴ so vcvtdq2ps is
exact; no YMM/GPR register collisions across the four macro expansions; Win64 compliance (GPR saves,
xmm6-15 low halves, CLEANUP_YMM, volatile upper halves); macro scope.

Verified against source afterwards: `SF_START` only pushes registers (rcx reaches `mov rbx, rcx`
intact); `HADD_YMM` touches only its destination and the scratch xmm4 (the other token accumulators stay
live).

## Preconditions (documented, caller-side invariants — no kernel branches)

- No -128 in weights or activations: the engine quantizer clamps to
  [-127, 127]; the validator generators produce [-100, 100] and [-127, 127]
  ranges only. (-128 pairs would saturate vpmaddubsw at 32768 and vpsignb
  cannot negate -128.)
- n > 0 and n % 64 == 0: the group loop is a do-while; Phase 2 integration
  would derive n from the model config where the Q8 group contract already
  guarantees both. The validator never passes n == 0 or n % 64 != 0.
- Scale-multiply order (xs·ws) before the FMA differs from any
  (partial·xs)·ws reference ordering by at most a last-ulp rounding — the
  validator compares against the C reference with the kt_verify_q8 tolerance
  (0.01 abs / 1e-4 rel), never memcmp.

## Validator results (v_matmul_validate, all green)

- Q8 sweep 32/32, F32 sweep 32/32 (unchanged).
- batch4 correctness 12/12: 8 whole-matrix cases vs the C reference
  (64x64, 128x64, 1024x1024, d=3, d=7, WO 2048x1024, W2 3072x1024, packed
  QKV 1024x4096), 3 edge batteries (all zeros, alternating signs,
  near-limit magnitudes with distinct per-group scales, d=96 not divisible
  by four), 1 row-slice case (first=2, count=d-2, separate output buffers).
- Correctness baseline = the C reference kernel for all cases. The
  DISPATCHED v7d_q8 cannot serve as the correctness baseline for arbitrary
  shapes: every dispatched shape requires its per-shape INI config block and
  FATALs otherwise (dispatch contract); the tuned dispatched kernel enters
  at the benchmark (plan 1.5), whose shapes all have blocks.
- Slice-test lesson (test bug, not kernel bug): `v7_batch4_row_slice`
  advances the output pointers INTO the same array per the plan ("the four
  output pointers advance by `first` floats", parallel jobs write disjoint
  rows of one shared output). The first test version allocated separate
  buffers without retargeting them, so nothing was written there. Fixed by
  retargeting `part_args.out[t]` to the dedicated buffers.

## Benchmark: Plan 1.5, adjacent baseline/candidate runs, minimum per side, set_spin(1)

| Shape | Baseline 4x dispatched v7d_q8 | Candidate 1x batch4 | Ratio |
|-------|------------------------------:|--------------------:|------:|
| QKV 1024x4096 | 122 us | 406 us | 0.30x |
| WO 2048x1024 | 64 us | 210 us | 0.30x |
| W2 3072x1024 | 84 us | 323 us | 0.26x |

(First-run numbers before the reboot: 144/58/79 us baseline, 446/217/336 us candidate — same verdict,
~3-4x gap.)

## Initial decision: the single-thread Phase 2 path was not integrated

At this stage of the experiment, the single-thread batch4 kernel was 3.3–3.8x slower end-to-end than
four dispatched `v7d_q8` calls on every tested shape. The original plan stated:

- Phase 1.5: "An isolated one-thread batch-4 kernel may lose against the
  existing parallel dispatcher: do not call it a success merely because it
  loads weights once rather than four times."
- Phase 2 gate: "Only enable a new path if it is correct and measurably
  faster end-to-end." / "If the kernel is slower... otherwise stop here and
  preserve the old prefill."
- Final task order 3: "Do not integrate a slower single-thread kernel into
  prefill."

The gap was the worker count: the dispatched baseline used a 12-thread worker pool, whereas the
single-thread variant used one thread. The plan's row-sliced parallel option (Phase 2.3) proposed reusing the worker pool via
`v7_batch4_row_slice` (the mechanism the slice test now validates), but it requires new dispatch
machinery ("introduce a separate batch job mode only when the measurements demonstrate that this is
worthwhile") — a significant architectural change that the plan gates behind a measured opportunity this
benchmark does not establish (it establishes the opposite for the single-thread variant). The later user-approved Phase 2.3 experiment superseded this initial stop decision,
as described in the committed-results section below.

**Kept at the initial stage:** the kernel, header, export, and validator battery remained in the tree as the tested
experimental artifact — the arithmetic is proven, the row-slice mechanism for a future parallel variant
is validated, and the benchmark harness is in place to re-measure any future row-partitioned
implementation.

**Preserved at that stage:** the existing text Q8 prefill (`prefill_q8.inc` per-token path), all SMC
machinery, the dispatcher, the autotuner, and the Q8 decode kernel were unchanged. The subsequent
Phase 2.3 work did change the QKV prefill path; it did not make the slower single-thread variant the
default.

## Results table (plan Phase 3 format)

| Variant | Workload | Baseline time | Candidate time | Correctness | Keep or revert |
| --- | --- | ---: | ---: | --- | --- |
| Single-thread batch4 | QKV 1024x4096, four vectors | 122 us | 406 us | 12/12 pass | Revert (do not integrate) |
| Single-thread batch4 | WO 2048x1024, four vectors | 64 us | 210 us | 12/12 pass | Revert (do not integrate) |
| Single-thread batch4 | W2 3072x1024, four vectors | 84 us | 323 us | 12/12 pass | Revert (do not integrate) |
| Dispatched batch4 (whole dispatch) | QKV 1024x4096, four vectors | 122 us | 86 us | 13/13 pass | Keep (committed) |
| Dispatched batch4 (whole dispatch) | WO 2048x1024, four vectors | 64 us | 43 us | 13/13 pass | Keep (committed) |
| Dispatched batch4 (whole dispatch) | W2 3072x1024, four vectors | 84 us | 63 us | 13/13 pass | Keep (committed) |
| Packed-QKV prefill (committed) | 17 / 80 / 142 tokens | see CSV | see CSV | byte-identical greedy output | Keep (1.26–1.81x) |
| Batch-WO prefill (next_phase Phase 1) | see WO section | see WO section | see WO section | byte-identical greedy output | Revert (not faster) |

## Dispatched batch4 + packed-QKV prefill (plan Phase 2.3, committed)

The user overrode the plan's stop gate after the single-thread verdict: `MATMUL_Q8_BATCH4` (mode 13)
dispatches the batch-4 kernel row-sliced across the existing worker pool via `v7d_q8_batch4` — one job
per contiguous output-row range, every job streams its own weight-row range once for all four input
vectors. Whole-dispatch vs four dispatched `v7d_q8` calls: 1.42x (QKV), 1.49x (WO), 1.33x (W2);
all 79 validator cases passed.

The prefill integration replaces the per-token QKV projection with per-tile Stage A (4x
`v7s_rmsnorm_quantize` into a shared stack scratch), Stage B (one `v7d_q8_batch4` against the packed
`g_inf_qkv_desc` weights), Stage C (Q/K/V scatter + bias + QK-RoPE per member), with the original
per-token path retained for tail tokens and as the correctness reference.

Three bugs were found and fixed during bring-up (all caught by review or smoke, none by inspection
alone):

1. Worker slice copy loop was a dec-first do-while that skipped slot 0
   (`out[0]` stayed stack garbage) — Zydis AV dump; now a counted loop.
2. `edi` (tile base token index) was clobbered by the scatter's `rep movsb`
   (rdi is the destination cursor), corrupting the next member's abs_pos and
   KV rows — fixed with push/pop rdi around the three scatters.
3. The scratch reservation under-allocated the scale area: `shr rdx, 4`
   reserved ONE `dim/16`-byte scale slot while the pointer table lays out
   FOUR (plus the WO variant's `shr rdx, 2` variant), so token 3's QKV
   block overran 3*dim/16 bytes into the lowest SF locals
   (`pq4_args`, `pq4_j`, `pq4_tile_end`) — the silent death right after the
   tuner banner. Fixed by reserving `dim/4`.

Validation: Q8/F32 smokes byte-identical, multiturn (start_pos > 0, 306/169/ 165-token prefills),
boundary remainders 0/1/2/3, verify.bat green.

## Batch-WO prefill experiment (next_phase Phase 1) — REVERTED

The reviewer package's `patches/0001-batch4-wo-prefill.patch` (scratch capacity raised to max(dim,
all_heads_dim) inputs / max(QKV, WO) outputs, per-layer `pq4_args.n/d` reset, one `v7d_q8_batch4` per
four-token WO tile with `v7s_quantize_chunk` inputs and `v7s_add_inplace` residuals, original fused path
retained for `all_heads_dim % 64 != 0` and tails) applied cleanly against the exact reviewed source
(SHA-256 match), built green with zero comment-checker violations, and produced byte-identical greedy
output on the measurement database (tuner delay 99999, journal applied identically).

Adjacent A/B/A prefill timings, 3 rounds each, same database, same power conditions (ms; baseline =
QKV-milestone per-token fused WO):

| Tokens | Baseline r1/r2/r3 | Candidate r1/r2/r3 | Verdict |
| --- | --- | --- | --- |
| 17 | 181.2 / 202.2 / 208.1 | 179.8 / 200.3 / 213.3 | wash |
| 80 | 615.7 / 640.1 / 610.1 | 746.8 / 700.9 / 835.7 | candidate 15–36% slower |
| 142 | 1199.5 / 1226.6 / 1180.1 | 1239.7 / 1179.5 / 1213.5 | wash |

Reverted per the todo's own gate ("Keep Phase 1 only if complete prefill is faster on representative
prompts"). Decode tok/s stayed in the 27–32 band on both sides.

**Why it loses where QKV wins:** the QKV batch replaces THREE dispatched projections with one; WO was
always ONE dispatched call, so batching removes zero dispatches while adding eight non-dispatched kernel
calls per tile (4 quantize + 4 add-inplace) on a trivial untuned row split — the 2048x1024 weight-reuse
saving is smaller than that overhead.

Phase 2 (W2) was not attempted: its prerequisite ("Phase 1 passed its complete-prefill gate") is unmet
and its patch hunks target the WO tile code that the revert removed. Phase 3 (gate/up evaluation) is
likewise gated. The earlier QKV prefill integration remains committed; only the optional WO/W2
follow-up was stopped.
