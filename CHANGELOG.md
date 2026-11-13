# Changelog

All notable changes to the v7 project, reverse-chronological. Each release
entry states what the release changes and improves; commit-level detail for
every change is preserved in the git history.

---

## [2.36.8] - 2026-11-13

- Repository hygiene: the root is cleaned down to the essentials. Every loose
  driver script moved into `scripts\` (tuner/python helpers, benchmark
  drivers); the AnyRes test fixtures moved to `vision\` next to their
  baselines; internal dev records (bug log, experiment ledger, phrasing
  catalog, dispatch problem record, release how-to, plans) moved to `docs\`;
  patch/experiment working directories and the retired one-off scripts left
  the tree (archived locally, not tracked). `attic\` is fully untracked.
  `v7_speed_results\` is gone from the repository, `current_champions.md` is
  retired, and the duplicate root copy of `bumwolf_avatar.jpg` is removed
  (the `backgrounds\` copy is canonical).
- Release packaging follows the new layout: the internal dev records are
  excluded from the shipped tree, the published benchmark documents ship
  from `docs\`, and `src\check_release.py` validates the new paths.
- README quality pass: the RMSNorm pipeline step no longer claims to scale
  weights, the MiniCPM5-1B rows match `docs\engine_benchmark.md`, the tuner
  is described as the cycle tuner it is, stale `v7s_tune_*` export rows are
  gone, the edge-AI use cases are scoped to what the engine actually does
  (text generation; no command issuance, no speech pipeline), server and
  memory examples use the `scripts\` paths, and the repository tree label is
  neutral.

## [2.36.7] - 2026-11-13

- Linux gate green end to end: the SwiGLU known-limit classification gains
  the SmolLM2-360M triple (960, 2560, 12345), which passes on the Windows
  (MSVCRT rand) vectors but drifts past the same relative window on glibc
  rand vectors -- the same float-vs-double accumulation class on different
  host random data, so `make check MODEL=...` reaches the CLI smokes and
  exits zero. kt_verify_swiglu_q8 now returns a distinct -1 for setup
  (allocation) errors, and the validator counts only a completed numerical
  comparison on an allow-listed shape as a known limit -- a setup error is
  always a hard failure. The stale "bit-exact port" reference description
  is replaced with the actual independent double-accumulating contract.
- Performance two-pager: full grammar and readability pass (comma splices,
  a dangling modifier, an article mismatch, consistent "power-saving"
  hyphenation), plain-language rewrites for a non-technical audience, and
  every number scope-labeled -- the headline cards are averages across all
  eight workloads, the chart shows one workload, and the battery figure is
  explicitly an estimate built from named external measurements.
- The last ten gcc `-Wwrite-strings` warnings are gone (the ViT linear
  validator's read-only helper parameter is now `const char *`), and the
  native Linux RAG helper (`examples/rag/bm25_rag`, built from the tracked
  source) is git-ignored as a build product.

## [2.36.6] - 2026-11-13

- docs/power_efficiency_2pager.html: two-page CPKI-design presentation of
  the throttle-robustness feature ("full speed, even on battery") -- measured
  full-power vs power-saver throughput bars per engine, the memory-bus
  argument with the four design causes, and the mobile equation combining
  independent battery-rundown research (efficiency mode +22%/+43% runtime,
  ~2x sustained compute per charge) with v7's 94.7% retention into the
  "twice the AI per charge" claim plus edge/field/always-on use cases.
- Hardware baseline: the AES rounds (Intern hash, sampler RNG, Logo RNG) use
  the legacy `aesenc` encoding instead of the VEX `vaesenc` -- same single
  AES round, same result bits, destructive form everywhere, both key tables
  were already 16-byte aligned -- so the CPU gate requires AES-NI (2010+)
  and no longer VAES. The unlocked set is the 2013-2020 AVX2 fleet (Intel
  Haswell through Comet Lake, Xeon Skylake/Cascade Lake, AMD Zen 1/Zen 2);
  VAES-capable CPUs run the same binary unchanged. No dispatch, no second
  binary, no parallel path.
- v_swiglu_validate self-classifies its failures: exit 0 means every failure
  is one of the documented Q8xQ8 float-vs-double shapes (printed and
  counted in the summary line); any F32xQ8 failure or undocumented failing
  shape exits 1. verify.bat consumes the exit code directly and the Linux
  check.sh drops its catch-all tolerance, so a green gate now proves
  exactly the documented set failed.
- v_sectorlogo_validate declares its test-helper descriptions and script
  arguments `const char *` -- 2852 gcc-14 `-Wwrite-strings` warnings gone.
- v7_speed_advantages.md: analysis of the throttle-robustness measurement --
  v7 keeps 94.7% of full-power throughput under the Windows Power saver plan
  (llama.cpp 78.7%, MNN 92.8%, OpenVINO 72.5%), widening the Qwen3-0.6B lead
  over llama.cpp from +7-10% at full power to +21-30% under throttling. The
  document derives each engine's CPU-bound fraction from the retention
  numbers (v7 ~6%) and explains the design causes (bandwidth-bound GEMV
  kernels, per-shape prefetch tuning, zero hot-path overhead, physical-core
  placement) plus the mobile implication: near-constant throughput at
  reduced power improves tokens per joule.
- Linux gate green: the BM25 retrieval helper the SectorLogo RAG wrappers
  spawn now builds on Linux (`examples/rag/bm25_rag`; the wrapper paths keep
  the Windows `.exe` spelling, and the POSIX spawn path strips the suffix in
  `src/linux/platform.c` -- the one place every engine spawn passes through).
  `bm25_rag.cpp` gains platform guards (`/proc/self/exe` for its data
  directory, monotonic-clock RNG seed, the Windows stdout binary-mode call
  kept under `#ifdef _WIN32`); the Windows compile path is unchanged.
- `make check MODEL=...` builds the RAG helper as a prerequisite, `make`
  refuses to run without an assembler instead of failing deep inside a
  recipe, and `make check` runs the same zero-tolerance ASM comment checker
  as the Windows build.
- `src/linux/fetch_qwen3.sh` checks the Python dependencies (numpy, torch,
  safetensors, transformers) and the free disk space BEFORE the multi-GB
  download and byte-verifies the fresh export with the independent
  checkpoint parser (its layout walk now expands every matrix per layer in
  the exporter's order -- previously it walked one matrix per type and
  rejected every real checkpoint); `tools/export.py` reports a missing
  `transformers` package in the same friendly style as `safetensors`.
- The server answers `-h/--help/--version` with its usage and exit 0
  instead of a fatal error, prints the engine version in the startup
  banner, and its long usage line passes the program name the `%s` expects.

## [2.36.5] - 2026-11-13

- Dispatcher: the Q8 job-descriptor fill walks rsi across the job array --
  initialized once at the fill entry and advanced by one job after each
  emitted descriptor -- instead of reconstructing every descriptor's address
  from the loop index; the job mode is cached in a register for the whole
  fill. 278 source bytes smaller; a synthetic A/B/A descriptor-fill
  microbenchmark measured 14.7-19.7% lower ns/call.

## [2.36.4] - 2026-11-12

- Dispatcher hot loops: the worker and caller batch loops are bottom-tested
  on the batch count (the per-job end load, per-job increment and top-exit
  branch are removed; the wp_end frame slot is dropped), and wp_count is
  charged at claim time -- the claim already guarantees the whole batch
  runs, so the tally is unchanged. The wait loop stays in its measured
  two-branch form; a single-backedge variant measured non-beneficial and
  occasionally regressive in a 101-round microbench.
- benchmarks.md: power-saving comparison -- under the Windows Power saver
  plan v7 keeps 93% of its full-power throughput (llama.cpp and OpenVINO
  keep 83%) and wins seven of the eight workloads; the relative lead over
  llama.cpp widens from +7% at full power to +21% under throttling.
- docs/engine_benchmark.html: two-page presentation of the external
  engine comparison.
- Tuner genome pools updated by the tuner-loop gates; the two bench
  retune scripts drop one redundant pause.

## [2.36.3] - 2026-11-12

- v_sectorlogo_validate: the oversize-script probe allocated one byte too
  few for its newline + NUL writes -- fixed (len + 2).
- check_v7.sh / validators_model.sh exit nonzero when any validator fails
  instead of reporting success through the pipeline's final echo.
- make check builds only the six model-free validators without MODEL; the
  full set + slffi_test.so build with one. dbg_image_recog builds via its
  own target, not the default all.
- Linux binaries print argv[0]-based, platform-neutral usage strings; the
  server's header comment describes both platforms. bin-linux/ ignored;
  header_find/find_field/msg_content are const char*.

## [2.36.2] - 2026-11-11

- CI persistence: .gitlab-ci.yml is now a sync-copied root file, so the
  build-linux job's explicit six-validator make command ships in the
  archive (the 2.36.1 zip carried the pre-audit command).
- Makefile: the duplicate check recipe is removed; one target remains with
  bash and the validator prerequisites.
- ht_add retires the cached on_stream_evict program when its value is
  replaced under the exact key, including equal-length rewrites; all other
  writes keep the O(1) cached callback path.
- Hardware documentation matches cpu_gate_probe: F16C and VAES listed as
  required, generation-based eligibility claims replaced with the CPUID
  rule, Zen 3's VAES support stated correctly.

## [2.36.1] - 2026-11-11

- Linux packaging fixes: *.sh files are LF-pinned in .gitattributes and
  executable in the index, so a git-built archive ships Unix 0755 scripts
  without CRLF smudging; make check and the README fetch example invoke the
  scripts through bash.
- Release checker accuracy: the vendored stb_image.h is exempt from the
  GPL-header check (THIRD_PARTY.md documents its dual license), the current
  external benchmark driver names are exempt, and the release AGENTS.md
  source count matches src/asm.
- The 50 technical-tour slides are lossless WebP now (pixel-identical;
  4.8 MB smaller), the Linux default build drops debug info and builds only
  production outputs (validators moved to make check), the README opens
  with the 2.36.0 platform summary and lists the full Linux prerequisites,
  and the WSL PowerShell diagnostics take Distro/Root parameters.

## [2.36.0] - 2026-11-11

- Server hardening pass: startup arguments FAIL instead of silently correcting
  (unknown flag, bad -prio/-maxgen/-history/port, extra positionals all exit 1;
  -sinks/-window pass through for the engine to validate; a refused -stream
  exits instead of silently running the legacy KV path); the security layer is
  removed (API key, form token, Host/Origin gates, X-V7-Local, sidecar
  allowlist — documented consequences in examples/server/README.md); /quit
  with a non-numeric or out-of-range code answers 400 instead of quitting
  with the default; router URI buffers answer 414 on overflow instead of
  truncating into a wrong route; Transfer-Encoding compares the token
  exactly (identityXYZ no longer passes) and anything but identity answers
  501; send_all reports real delivered bytes so the body-loss log line tells
  the truth; the health acceptor stops the process when accept() fails
  instead of looping forever; a full handoff queue answers 503 + Retry-After
  instead of a bare close; the RAG panel OOM now skips the panel and keeps
  the page; web form validation rejects NUL/invalid UTF-8/empty messages
  with 400 and no longer stores a fabricated bot bubble for empty answers;
  /v1/decide requires kind (parsed by direct compare, no truncating buffer)
  and rejects non-string option elements; vision requests honor max_tokens
  (restored afterwards) and reject an empty prompt; every inference output
  buffer is grown to fit the turn instead of cutting at 8 MB; the web chat
  history capacity comes from -history / V7_SERVER_HISTORY (default 4096).
- All root-level driver bats (build, verify, server lifecycle, fetch,
  benchmark, tuning, paper runs) moved into a `scripts\` directory; every
  adapted bat derives the project root from its own location (`%~dp0..\`),
  sibling-bat calls go through `%~dp0`, sync_release copies the whole
  directory, and the release gate + README/AGENTS lifecycle commands point
  at the new paths.
- Linux integration phase: the validator surface now builds natively (encode,
  fused, matmul, sectorlogo, stream, swiglu, vit_linear joined the Makefile's
  VALIDATORS; sectorlogo builds with `-msse4.2` for its CRC32C probe), and
  `src/linux/check.sh` runs the full model-free + model-dependent set with the
  sectorlogo validator rooted at the repo tree (it reads the stdlib
  `autoexec.logo` and writes `build_logs/` relative to the working directory).
- Linux server vision: `v7_server_image.cpp` decodes PNG/JPEG/BMP/GIF through
  the vendored single-header `src/linux/stb_image.h` (4-channel load, alpha
  flattened onto white, 64 M-pixel pre-decode cap, EXIF orientation applied)
  with the same pixel contract as the Windows GDI+ path; the Linux server
  therefore builds the full source set again, no per-platform exclusions.
- Linux host compat (`src/linux/compat/windows.h`): CreatePipe, WriteFile,
  GetStdHandle/SetStdHandle (fd-based, with a private copy of the original
  std descriptor so redirection restores), `sscanf_s`, and `_control87` via
  fnstcw; the sectorlogo validator's pipe-redirect and x87 control-word suites
  now compile and run unchanged.
- ABI consistency for the Linux port: the Q8 C reference handed to
  `v7_batch4_check` is wrapped in an ms_abi shim (`v_matmul_validate.cpp`),
  `isfinite` calls became `std::isfinite`, and HOST FFI exports (e.g.
  `sc_now_ms` in `run_sectorlogo.c`) carry `__attribute__((ms_abi))` — the
  engine calls CallLibrary targets with the Windows register convention on
  every platform, now documented in `sectorlogo_ffi.inc` and the Linux
  GetProcAddress shim.
- Server: the bind retry loop is gone — a busy port reports the failure and
  terminates with exit code 43, and `server.bat` stops the supervisor on it
  instead of relaunching into the same conflict; the model id positional is
  REQUIRED (`DEFAULT_MODEL` removed, all launchers pass it explicitly).
- WSL helper scripts (`src/linux/wsl/*.sh`) derive the repo root from the
  script location with a `V7_ROOT` override instead of hardcoding
  `/mnt/c/project/v7`.
- CI: a `build-linux` job (Debian/glibc) assembles and links the shared object
  and all host tools, then runs the model-free validators on every push.
- README: Linux quick start (prerequisites, `make`, `make check`, fetch
  script) and the port's platform notes; `docs/AGENTS_dev.md` Sec.104 corrected
  to describe the current swiglu contract (independent double-precision
  reference, Q8xQ8 drift documented as the one accepted limit).
- `mdb_query_key` prints `%llu` instead of the MSVC-only `%I64u`.
- AGENTS.md: expanded output-visibility rules (never filter or truncate
  command output; raw capture only).

## [2.35.2] - 2026-11-09

- Release-readiness pass on the shipped package.
- Sampler: `v7s_sample` records the pool activity state at entry and returns an
  idle pool to the OS wait in its shared epilogue; a call made inside an
  inference-active phase leaves the activity state untouched, so chat never
  sleeps between tokens.
- Q8 kernel: the k-loop bound is `cmp ecx, r12d` against the live input
  dimension instead of a runtime-patched immediate; `PATCH_Q8_N` now updates
  only the three scale-row displacements and skips when n is unchanged.
- Bicubic resize: horizontal weights stay live in xmm8 across all four vertical
  taps, removing the per-pixel stack round-trip; output remains byte-identical.
- Sampler softmax: blocks C/D load during the first pair's exponential tail.
- Documentation: removed the retired `-continous-spin` / `init:continuous_spin`
  claims from the builder usage and the guides; `rebuild_all_llm_dbs_full.bat`
  delegates to the canonical rebuild script; `bench_tuned_tokens.bat` no longer
  re-invokes itself; README trimmed to a quick start.
- Version identity synchronized to 2.35.1.

## [2.35.0] - 2026-11-09

- Optimization batches (six reports + the cursor-persistence patch) applied and
  verified: instruction-sequence replacements, dead-code removal, algorithmic
  optimizations, LOC reduction, kv pre-touch, and faster AVX2 scheduling.
  Contains the sampler greedy single-fork argmax, prefill_q8 in-place K/V
  scatter, BPE edge-record cache, BSS entry packing, the shared cycle-tuner
  sort path, and 60+ macro consolidations. All bit-exact against the prior
  release; verify.bat 21/21 validators + 10 inference smokes green.
- Cycle tuner: persists its 0..127 sort cursor per shape in genome 0's
  padding (format unchanged at 64 KiB), direct slot allocation, cursor
  range-clamp, tsc_start consumption, print_totals alignment, and a guard
  against the wasted b=c=0 episode at wrap.
- MicroDB: combined KV-plane allocation, shared staging copier, doubling
  LZ77 match expansion, shared pool-growth slow path, direct insert/update
  routing without the in-place-reuse flag.
- Vision: VCB_LAYOUT_WEIGHTS / VCB_RGB8_UNPACK macros; the anyres round-2
  and RoPE full-grid round-2 algorithmic changes were NOT applied pending
  multi-tile validation.
- Instruction-level micro-opts: 7 exponent fragments, bottom-tested SMC
  loops, LEA folds, sete DeepStack gates, Logo equality masks, vit_gelu
  tail in float units.
- kv_touch_patch: first min(seq_len, 1500) token pages of every layer's
  persistent K/V cache are pre-touched at creation.
- New tools: mdb_cli.h, qpc_seconds.h shared helpers.

## [2.34.0] — 2026-11-08

- Whole-chat continuous spin (`fork_patch_2` package): every in-inference
  `v7s_set_spin(0)` call removed from the forward/prefill/vision paths — the
  worker pool stays pause-spinning through prefill, generation and vision
  calls. A new internal `v7s_stop_spin` (`util_export.inc`) is called once at
  chat end (`chat_finish.inc`): it stores `always_spin=0` and returns without
  waiting; idle workers poll the flag in their spin loop (`v7_worker.inc`)
  and take the existing `WaitOnAddress` path. The next inference call
  re-enables spinning through the existing `v7s_set_spin(1)`.
- Stochastic forked sampler: temperature scaling + max scan, exponential
  accumulation + sum, and normalization run as three independent two-core
  phases (`sampler_stochastic_fork.inc`), joined by the new intermediate
  `CONTINUE_FORK 0` phase (`fork.inc`) — W0 publishes its result, clears
  `targetaddr` and hot-spins without PAUSE for the next phase; only the final
  normalization phase ends with `CONTINUE_DISPATCH 0` to rejoin the pool.
  Each phase carries its own adaptive fraction (fixed-point /256, ±1 per
  call, clamped 1..255, default 128), persisted as three little-endian u32s
  in `v7_stochastic_tune_percentages` (loaded at inference creation, saved at
  teardown beside the greedy percentages).
- Forked greedy argmax made unconditional: the `V7_ENABLE_FORK_GREEDY`
  compiler switch and the sequential fallback scan are removed — the
  two-pass FORK 0 argmax is the only greedy path (the stale root-level
  fork.inc copy that shadowed the real src/asm/fork.inc is gone).
- Sampler pool-state guard: the forked sampler phases require worker 0 in
  pause-spin mode, but `initialize_once` is lazy — it runs at the first
  dispatched call, and `v7_init` alone never creates the pool. A process that
  only calls `v7s_sample` directly (the validator's 40-case sampler gate,
  which runs after its six border-contract death tests quiesced the pool and
  destroyed the hash table) therefore forked into a nonexistent pool: the
  unconditional `FORK 0` published a continuation nobody would ever pick up
  and `FORK_WAIT 0` spun forever — verify.bat stalled in Phase 3 with no
  output. `v7s_sample` now calls `initialize_once` itself at entry (the same
  CAS-gated lazy init every dispatched entry uses) and FATALs when the pool
  still is not pause-spinning (`always_spin == 0`, i.e. a
  `init:simple_threads > 0` sleep-mode pool whose workers never poll the
  fork slot). The validator re-loads the database with `v7_init` between its
  border section and the sampler gate, because the sampler's lazy init reads
  the INI keys from the hash table the border teardown destroyed.
- Sampler vocabulary contract: `v7s_sample` FATALs for vocab < 64 — the
  forked greedy and stochastic paths split the vocabulary into two ranges of
  at least 32 elements each, and `STOCH_BOUND`'s helper reservation
  (`vocab - 32`, aligned down) underflows for smaller vocabs, handing W0 a
  ~4-billion-element count that scans off the end of the logits buffer
  (read AV wrapping the address space, hit immediately by the validator's
  old vocab-9 regression case). Every supported model's vocab is far above
  64, so the tiny-vocab shapes are impossible in production; the validator's
  temp-clamp regression cases now run at vocab 65/66 (still exercising the
  helper-side scalar tail) and the equality-nucleus sizing suite moved from
  the 4-token vocab to a uniform 64-token vocab, where every probability is
  exactly 1/64 and the cumulative-equality edge survives as exactly
  32/48/33 nucleus tokens.
- Q8 kernel stack-free rewrite (`stackfree_q8` package): `v7s_q8_chunk` drops
  its five `SF_*` working slots (`q8k_scale_stride/save_n/save_d/wsbase/wqbase`)
  and their five reloads per four-row iteration; `n` lives in R12, job-local
  `d` in R13, and RDI/R15 advance through weight/scale rows (reused by the
  1-3 tail rows). The three row-scale broadcasts take SMC-patched disp32
  displacements (S, 2S, 3S with S=n/16) emitted through an oversized
  placeholder; `PATCH_Q8_N` writes all four immediates in the existing
  pre-dispatch event and still skips everything when `n` is unchanged.
  Measured on the 7430U (same-session A/B): Q8 1024x1024 191 -> 265 GB/s
  (+39%), 1024x3072 398 -> 425 GB/s (+7%), all other Q8 shapes neutral;
  F32/SwiGLU/fused kernels untouched. verify.bat 21 validators + 10 inference
  green; `stackfree_q8/` handoff package committed to the tree.
- Fourth fork experiment, enabled by default: `V7_ENABLE_FORK_PREFILL_Q8_BATCH4`
  parallelizes the batch-4 Q8 prefill Stage C (Q/K/V scatter, QKV bias,
  per-head QK RMSNorm+RoPE, FP16 KV-cache pack) across workers 0-2 plus the
  caller — one immutable 128-byte `pq4_fork_args` descriptor per tile member,
  reentrant head kernel from `qk_fork_kernel.inc`, shared RMSNorm dispatcher
  BSS untouched. Gated to QK-norm models with >=3 workers and always_spin;
  RoPE-only models, small pools, sleeping pools, and 1-3 token remainders keep
  the unchanged serial Stage C. New `prefill_q8_fork.inc` (per-member
  postprocess kernel); all forked members join before the next regular
  dispatch. Handoff package `fork3/` (patch, complete files, integration and
  validation docs) committed to the tree. verify.bat 21 validators + 10
  inference tests green.
- Recursive fork/join primitive over the existing worker pool: new `fork.inc`
  defines the `FORK N` / `CONTINUE_DISPATCH N` / `FORK_WAIT N` / `FORK_TAKE N`
  macros over one 64-byte `fork_params` handoff slot per worker (BSS, published
  by the parent, polled in the worker spin loop, cleared last after the worker
  restores its dispatcher state). No new threads, no OS calls, no new exports;
  `dbg_inference.asm` now includes `fork.inc` before `v7_data.inc` so the BSS
  layout constant resolves in the standalone harness build.
- Three compile-time-gated fork experiments (all switches default OFF, original
  compute paths unchanged): `V7_ENABLE_FORK_QK` runs Q-head RMSNorm+RoPE on
  worker 0 while the caller does K heads and the K/V FP16 pack
  (`qk_fork_kernel.inc`, descriptor-driven, has_qk_norm models only);
  `V7_ENABLE_FORK_ATTENTION` splits heads between worker 0 and the caller for
  positions 16..96 (`attention_fork.inc`); `V7_ENABLE_FORK_VIT_GELU` splits
  vision-layer GELU buffers between caller and worker 0 (`vit_gelu_fork.inc`,
  the last merger GELU stays direct). Handoff packages `fork/` and
  `fork_optim/` with patches, interface contract, examples, and verification
  plan are committed to the tree.
- Token/s optimization package 1: quantize inputs up to 4096 elements take the
  direct single-call path (threshold 16 -> 64 groups); generation attention at
  positions 0..32 runs the per-head kernel directly on the caller instead of
  publishing jobs; the shape-change prefetch patcher is differential (block 0
  is the installed-byte mirror, unchanged slots skip all writes), with private
  non-serializing setter variants and one `smc_prefetch_serialize` barrier per
  changed batch replacing 82 per-call CPUIDs; the dispatch caller skips its
  fetch-add when the queue cursor already covers the job count; a new
  `init:continuous_spin` database key (built by `build_db.exe
  -continous-spin 0|1`) makes `v7s_set_spin(0)` a no-op so the worker pool
  keeps pause-spinning between calls instead of sleeping.

- Token/s refinement package 2: enabled-to-enabled prefetch distance changes
  now write only the 4-byte displacement directly (the two attention slots
  have no displacement and take type-only patches), the genome installer uses
  a metadata-based displacement helper with no full-instruction restore, and
  the batch barrier is a single CPUID. The tuner install loop's exit check is
  retained (the refinement as delivered dropped it, which would walk the pass
  past the tuner-owned slots into out-of-table patch metadata).

- Dead-state and hygiene batch across ten ASM files: unread BSS mirrors
  (`g_vision_start_id`/`g_vision_end_id`) and their stores removed, write-only
  KV-cache frame slots removed, leaked `%define`/`%macro` names undef'd at
  file ends, a register round-trip collapsed, an unused frame register
  dropped, stale/wrong comments trued, and `v7s_sidecar_refresh` now returns
  its documented constant 1.

- StreamingLLM correctness batch: accepted generated tokens are recorded in
  `out_tokens` on every generation path (SectorLogo hook on or off), a nested
  `v7_generate` restores the outer session's token-history validity flag, a
  rewind to exactly C now rebuilds the linear KV rows before streaming can be
  disabled at that boundary, and the KV-cache file header grows to 48 bytes
  carrying the save-time streaming geometry (enable, S, W, valid-window rows).
  Wrapped loads enforce equal S/W before scattering and replay the save-time
  validity tracker, so a rewind-shrunk window stays shrunk across a save/load
  roundtrip; mismatched geometries and old 32-byte-header cache files are
  rejected with a visible message (regenerate the cache).

- Q8 token-embedding dequantization restored to full 64-element group
  coverage (the interim two-load form left half of every group unwritten,
  corrupting real-model generation while synthetic validators and
  streaming-on/off bit-identity checks stayed green).

- In-process cycle tuner: the Q8-SwiGLU dispatch mode now advances its
  shape's state machine, and the genome installer patches the SwiGLU-Q8
  kernel's own prefetch slots 27..38 on top of the Q8 slots 0..7. Population
  statistics are equal-position-weight means (one raw b1 sample per list
  position per sweep, best = front 10, pool = non-tail). The tail mutation
  skips the active delay-genome slot and redraws a parent that would alias
  its destination. The mutation-count draw bound is exported
  (`g_probe_mutations_max`) for the config-probe validator, which pins it to
  10 for its deterministic diff cap. `tuner:rand_delay` of 0xFFFFFFFF is
  refused at init (the jitter modulus would divide by zero).

- Cycle-tuner lifecycle purity: inference destroy persists the population
  files but keeps the 64K blocks and machine phases allocated for the
  process lifetime, so a shape's episode continues across chat and context
  boundaries instead of restarting its delay phase; activation happens once
  per process at the first tuner-active chat end, and `V7_init` remains the
  only reset point. Population files that the interim save-argument bug had
  truncated to zero bytes were deleted; the tuner re-seeds fresh populations
  on first load.

- In-process cycle tuner: comment-probe restructuring -- one labeled clear
  region per lifecycle, contiguous array families with assemble-time layout
  asserts, the flip's episode body unified to one bt/cmovc identity
  selection, statistics folded through a shared macro, and the timer
  guarded against folding an unstaged start stamp.

- In-process cycle tuner: the comparison-partner draw switches from the
  quadratic f*f pick to the truncated exponential (decay 3, "exp3") -- the
  3-pass partner-sweep winner (13 reps per contender: top median/mean,
  most stable pass-to-pass, family-level z = 1.9-2.1 vs pow/tour).
  Implemented as cyc_exp_pick: 12 x87 instructions around one FYL2X --
  the exact inverse CDF, no tables, no init, cold-path only. Mutation
  parent/donor draws and the delay jitter stay quadratic (unchanged
  proven roles). genome_cycle_sort's --partner default follows to exp3;
  pow2 remains available as the control.

- In-process cycle tuner: per-shape pool fitness statistics. Every `c`
  measurement of the a1/b1/c/b2/a2 episode now accumulates into two buckets
  per registry slot -- the front 10 genome slots ("best") and every non-tail
  slot ("pool") -- and folds into averages at each population sweep wrap
  (127->0 anchor wrap), held in six `.bss` u64 arrays of 64. Three new public
  exports: `v7_get_best_fitness` / `v7_get_pool_fitness` (pointers to the
  per-shape average arrays, 0 = no completed sweep) and
  `v7_get_active_shape_count` (registry occupancy, the valid index range).
  `inference_example` prints both averages per shape in its `[perf]` block.
  The `tuner:tail_slots` init bound tightens to <= 127 so the pool bucket can
  never be empty (an all-tail configuration would divide by zero at the wrap
  fold).

- The in-process cycle tuner's genetic knobs are now database-configurable:
  `build_db.exe` gains `-tuner-mutations-max <n>` which seeds the new
  `tuner:mutations_max` key (the per-child field-count draw bound for
  `config_mutate_genome`); the engine reads it at tuner lifecycle start into
  the new `.bss` `g_probe_mutations_max` and the mutation engine draws
  1..max fields per child (the old hardcoded 1..10 range is gone). Defaults
  moved to the measured winners: `-tuner-sex` 40 -> 80, `-tuner-mutate`
  3 -> 50 (half the 128-slot population becomes mutants per wrap), and the
  new mutations-max defaults to 40 (the sweep found 10-30 tied and 40 the
  best combined with crossover 80). ASM-side fallbacks match the new
  defaults when a DB lacks the keys.

- `genome_cycle_sort` parameter-experiment harness: the tool's PRNG is now a
  xorshift64* seeded with the fixed seed 42 (replacing RDRAND), so every run
  draws identical benchmark tensors, the identical probe list and the
  identical reference genome. The 127-slot list fill inserts fully random
  probes UNMEASURED (no baseline election, no A/B/A during the fill); a
  reference genome drawn at startup from the same seed is never sorted or
  mutated. A final benchmark races the sorted top-10 against that unchanged
  reference (reference dispatched around every genome, its fastest reading
  as the comparator) and prints how much better the best genome is in
  percent -- the same-seed number that isolates the effect of
  `--mutate-percentage` / `--sex-influence` settings. New
  `--mutations-max <n>` exposes the per-mutant field-count draw (1..n,
  default 10). `run_cycle_sort_mutation_sweep.bat` sweeps mutation 5..50
  in steps of 5 at crossover 80 (the best sex-influence found), 100
  rounds, 3 repeats per rate; `run_cycle_sort_crossover_sweep.bat`
  sweeps sex-influence 0..90 in steps of 10 at mutation 50; and
  `run_cycle_sort_mutation_count_sweep.bat` sweeps `--mutations-max`
  1..30 in steps of 5 at mutation 50 / crossover 0 -- all into per-run
  capture files under `build_logs\`.

- Simple worker pool mode (`init:simple_threads`): a new database key selects the
  worker-pool build at initialization. `0` (the default `build_db.exe` writes) keeps the
  full-placement pool unchanged; `N > 0` creates N plain worker threads -- `CreateThread`
  plus `SetThreadPriority` only, with no pinning, no SMT awareness, and no topology scan
  (`v7_init.inc` reads the key in step 1b, skips the GLPIEx placement machinery, and the
  launch loop / caller handling / coverage assertion branch on it; the compact threading
  report prints the simple-mode line). `build_db.exe` gains `-simple-threads <n>` which
  writes the key as a raw dword; a missing key is a stale database and FATALs by key
  name. Priority machinery (INI `worker_priority`, `v7_override_prio` env override,
  `v7_set_worker_priority`) applies unchanged in both modes. Plan:
  `simple_thread_implementation_plan.md`; documented in `docs/tuning.md`.
- Reviewed micro-optimization batch (opt_and_bug revised reports, applied and
  source-verified): loop-control and issue-order trims across the compute path.
  Bottom-tested fill/scan loops remove the redundant head guard and unconditional
  back-edge in the parallel memcpy job fill, the quantize planner, the Q8-batch4
  planner (`utility_dispatch.inc`), the M-RoPE image-run measure and presence
  scan (`mrope.inc`), the VQ NearestMatch staging and codebook walks, and the
  n-gram window/collection loops. Instruction-scheduling changes overlap
  independent loads with in-flight work without touching the operand order:
  dual F16C conversion destinations and pre-issued V loads in attention, batched
  conversions in `f32_kernel`/`v7_macros`/`swiglu_kernel`, scale broadcasts
  hoisted ahead of the byte-dot chains in `q8_kernel`/`q8_batch4_kernel`/
  `swiglu_q8_kernel`, interleaved convert/store in the quantize and RMSNorm+
  quantize kernels, `vptest` argmax and top-p predicates in the sampler, and a
  32-float pack loop in `kv_f16c`. The odd-row F32 path fuses two 32-float
  iterations into one 64-float group. `aes_string_hash` falls straight into
  finalization on exact-multiple lengths. `memcpy` gains exact-8/16-byte
  single-store specializations. `microdb_lz77` shares one byte-copy tail across
  the match tiers, and the short-input compressor size is computed directly.
  SectorLogo: the intern scan enumerates occupied granules with `shrx`/`tzcnt`,
  `sl_hash_float` fuses the NUL scan and CRC into one pass, and `bzhi`/`btr`/
  `bts` replace several mask/build sequences in the GC, sort, intern, FFI,
  statement, value and primitive paths; the page-threshold arithmetic in
  `string_hash` and the CLDEMOTE ModRM rewrite in `smc_macros` are shortened.
  Vision: the AnyRes path stops allocating and freeing three unused DeepStack
  sinks (the forward caller now passes NULL and `vit_forward` skips the tap
  pipelines when no output is requested; the callback frame drops the dead
  slots); bicubic resize fuses the weight polynomial into Horner FMAs, keeps
  `x_scale` in a register, and fuses the half-pixel coordinate expressions;
  plus ROE-build/pixel-shuffle/pos-embed/dispatch/layernorm loop and load trims.
  Several reviewed items were discarded after the gates showed they were not
  safe (the direct-write number renderer in `sl_value_to_str` regressed the
  stream validator; the change was reverted, not patched around). Real fix:
  none beyond the discarded set. verify.bat: 20 validators + 10 inference
  smokes green; vision smoke answer byte-identical (only the tuner journal
  count line advanced).

- Reviewed micro-optimization batch (opt_and_bug revised reports, applied and
  source-verified): 32-wide float packing with a `vpermd` lane fix-up in the
  quantize and RMSNorm+quantize kernels (roughly half the shuffle and store
  count per 64-float group); four-accumulator greedy argmax in the sampler
  (removes the loop-carried max dependency); hoisted per-token activation
  pointers in the four-token Q8 batch kernel; running-pointer attention cursors
  in the F32 and Q8 prefill paths (one multiply per token removed); hoisted
  up-matrix column bias in both SwiGLU kernels; the LZ77 compressor's
  nibble-clamp constant parked in a register that was otherwise dead; a
  branch trim in the traced SectorLogo dispatch rejoin; `bzhi` for the tail
  equality mask in the SectorLogo intern path; and instruction trims in the
  FFI owner-registry walk and the tuner active-shape registry walk. Real fix:
  the FFI owner-registry name capacity check rejected a name that exactly
  filled its 128-byte slot (off-by-one) — it now accepts a 127-character DLL
  name and refuses only names that cannot fit with their terminator. verify.bat:
  20 validators + 10 inference smokes green.

- Persistent KV cache migrated to IEEE FP16 via F16C: stored K/V rows are half
  precision (`kv_f16c.inc`, `vcvtps2ph` / `vcvtph2ps`), halving the cache-file
  size and cache-resident bandwidth. F16C joins the CPU baseline gate, the
  save/load path and the RunState sizing account for the new element width, and
  the attention and prefill paths convert on the fly. Plan and measurements:
  `f16c_kv_cache_plan.md`, `tools/attention_shared.h`, `tools/runstate_alloc.h`.

- Scheduler caller-core participation rework: the caller reserves itself first,
  job sizes follow the participant count, and the worker pool publishes a full
  threads report (per-core eligibility, caller pin, chunk geometry) with CPU0
  diagnostics. A follow-up fix loads the `worker_aff` .bss base into a register
  before indexing (an indexed .bss operand cannot be RIP-relative). The
  small-dispatch physical-core restriction is removed (`dispatch_execute.inc`,
  `v7_init.inc`, `v7_worker.inc`).

- Four-token Q8 batch kernel: `q8_batch4_kernel.inc` computes four independent
  Q8 GEMVs against one shared weight matrix (each 64-byte weight group loaded
  once for all four tokens), with a row-sliced worker-pool dispatch mode
  (`v7s_q8_batch4`) — measured 1.33-1.49x versus four dispatched GEMVs. The
  matmul validator gains the differential batch4 battery and a benchmark
  section.

- Speed-optimization campaign (V7-only A/B/A measurement, stages 01-05):
  combined Q/K RMSNorm+RoPE dispatch, producer-side quantization fusion,
  sustained-Q8-throughput, and prefill-batching / GQA-attention experiments,
  each measured against a captured baseline. None produced a safe win, so the
  dispatch and quantization changes were not pursued (stage 01 reverted); the
  results and the plan are recorded in `docs/speed_fix_results.md` and
  `docs/plans/`. The attention loadtest gained `--pos` / `--iters`, and
  matmul+attention SMT comparison benches were added.

- Tuner audit hardening + a reviewed `opt_and_bug` batch: the tuning journal's
  read/write byte-count contract is enforced, the tuner halts cleanly under
  Logo recursion, both decision scripts reconcile their verdict geometry, the
  dead defer machinery is removed, and the per-shape no-default FATAL message
  is fixed; dead code and branch pruning across the SectorLogo intern / eval /
  string / evict paths and the ViT kernels.

- Parallel per-shape A/B/A tuner: every configured shape now runs its own
  independent candidate/phase/delay stream (256 state slots, lazily seeded
  from the per-shape block on first activation), the dispatcher activates the
  dispatched shape's state itself (`v7s_tune_activate_shape`) and manual shape
  selection (`v7s_tune_set_shape`) is a deliberate FATAL stub. One verdict
  script per verdict (`tune_accept` / `tune_reject`), verdict scripts are
  deferred (not re-entered) when a dispatch completes inside a Logo eval, the
  Logo callback can mutate only the current shape's candidate
  (`TuneSetParam`/`TuneSetDelay` are mutation-scoped), and the initial delays
  come from the DB keys `tuner:delaydefault` / new `tuner:delay_large`
  (shapes with a dimension over 16384). Fixes against the delivered package:
  the FATAL-on-every-DB-load ordering bug in `V7_init`, the journal-apply
  cache invalidation (previously dead code under the holdback), full-TSS
  reset in `v7s_tune_reset`, undefined upper-half `rcx` indexing in the
  param getters/setters, the CallLibrary `"int"` type names, and verdict
  scripts performing raw arithmetic on historical out-of-window journal
  distances (30-bit payload overflow killed the turn). The matmul validator's
  split-geometry sweep now stages through the exported `shape_blocks` symbol
  with a cache-invalidator dispatch. verify.bat: 20 validators + 10 inference
  smokes green; tuner accepted 179 / rejected 334 candidates across the run
  with 473 journal records flushed.

- Measurement session (paper measurements on the reference laptop): repaired
  the benchmark harness first (`bench_qwen.bat` compared F32 against F32 --
  the "Q8" run lacked `-q8`; the example README documented `-q8 0` / `-q8 1`
  for a Boolean flag; `bench_tuned_tokens.bat` ended in `%0`, an infinite
  self-recall). Then ran the P0/P1 measurement suite on the delivered
  session band: end-to-end decode/prefill F32+Q8, cache-resident vs
  streaming kernel loadtests, the 3-condition fictional-fact RAG experiment
  (no-RAG 0/7, initial-context 6/7, mid-stream rewrite 3/7 + 4 partial with
  the edit firing 7/7 and a clean off-topic control), the controlled
  append-only rewrite-cost pair, llama.cpp comparison (same checkpoint,
  different quant format), the native validator gate, the BM25 benchmark,
  and load time + memory snapshots. Measurement-only `/DV7_MEASURE`
  instrumentation in inference_example (compiled out of normal builds);
  measurement bats at root set `v7_override_prio=1`; the measurement
  database is built with `tuner:delaydefault=999999` so the tuner's A/B
  pairs never race inside a measured run. Durable record:
  `docs/measurment_worklog_and_results_report.md`; deliverables (wiped by
  the next build) under `build_logs\paper_measurements\`.
- `mdb_decompress` stream validation (owner directive): the LZ77 decompressor
  now ENFORCES the caller's `max_dst` (the previously ignored r9 argument) and
  validates every stream element before use -- literal runs and match copies
  must end inside `[dst, dst + max_dst)`, tokens, extension bytes and the
  2-byte offset field must lie inside `[src, src + src_len)`, and the match
  offset must be >= 1 and <= bytes written so far (== written stays legal,
  reading the first output byte). Any violation is a named FATAL, so a
  truncated or hand-edited `lz4ht.db` / KV-cache file can no longer corrupt
  memory. Init-time cost only (`db_load`, `load_kv_cache`), never on the hot
  path; round-trips are bit-identical (v_mdb_validate 222/222).
- `chat_prompt.inc` width fixes: the continuation system-prompt compare loaded
  `g_sys_prompt_len` (a resd variable) with an 8-byte `mov rcx`, pulling the
  owned pointer's low half into the high 32 bits, so every continuation turn
  that passes a non-NULL system prompt FATALed with "system prompt differs";
  now a 4-byte `mov ecx` (zero-extending). The two `mov [rel
  g_sys_prompt_owned], 0` zero-pins assembled as BYTE stores (NASM silently
  picks the smallest encoding for an un-sized store), leaving the pointer
  non-null and enabling a double free of the same mdb block on a later
  session-open; now explicit `mov qword`. Verified by probe: `C605..` (byte)
  vs `48C705..` (qword) on NASM 3.02rc6.
- BSS alignment: all `.bss` padding now uses `alignb` (BSS_DWORD/BSS_QWORD/
  BSS_ARRAY macros plus the manual `align` sites in v7_data.inc, sampler.inc,
  file_backed_mem.inc) instead of `align`, which is NOP/db padding semantics
  and warns under some NASM versions in no-bits sections.
- Kernel micro-optimization batch (opt_and_bug, all bit-identical; verify 20/20
  + vision smoke baseline hash unchanged): vit_kernel K-loop weight loads
  folded into the FMAs (-4 loads/iter) and tail; SF_YMM spill slots replaced
  by SF_XMM low-half slots in vit_linear (8 regs), vit_attention (9),
  fused_f32 (1), swiglu (8), swiglu_q8 (9), sampler (5) -- halves save/restore
  traffic; redundant qkv+head_off recompute in both attention paths removed;
  vit_gelu exp chain reads the loop-invariant 1.0f broadcast instead of an
  RIP-relative constant; fused_f32 RoPE norm-weight loads folded; qk-127
  divisor taken from the hoisted broadcast; swiglu/vit_add dead parity fold
  and dead fill-loop branch removed; f32 paired reduce packs both row scalars
  into one 64-bit store.
- In-process tuner: the accept and reject verdict scripts now run IMMEDIATELY
  at the verdict instead of being queued to the next token boundary. The old
  single-slot queue let several verdicts collapse onto one candidate, so after
  the first promotion the candidate stayed identical to the new baseline and
  kept getting accepted by luck; the dead queued-verdict consumers and the
  `tune_pending_key` global are removed.
- In-process tuner: each candidate now mutates a random 1..10 DISTINCT slots
  (was exactly one), so a fresh candidate probes several knobs per A/B
  decision; a per-candidate done-list prevents a second hit on the same slot.
- In-process tuner: the ASM dump mode now follows the DB key `tuner:debug`
  (`tune_init.logo` calls `TuneSetDump 2` for `tuner:debug=1`, `TuneSetDump 0`
  otherwise), so one key silences BOTH the banner prints and the ASM
  per-accept block; previously the ASM `tune_dump` kept its default and printed
  accepts even with `tuner:debug=0`.
- `build_db.exe`: new `-tuner-delay <n>` option (default 80) seeding the
  `tuner:delaydefault` key -- matching-shape calls skipped between A/B pairs.
- Tuning journal: `record_parameter_change`'s grow path published
  `journal_data_ptr = new_block + used` (rdi left advanced by `rep movsq`)
  instead of the block base, so every grow after the first shifted the window
  and overflowed one record; the base is restored after the copy. Journal
  compaction is spliced into the flush (newest record per `(shape, slot)`
  wins, superseded records dropped) and the flush/apply/compaction lines are
  gated by `tune_dump`.
- `tools/journal_decoder.py`: `pf_dist` slots (24..64) print as SIGNED disp32,
  so a legal negative distance no longer reads as a huge positive.
- `set_worker_priority.inc`: the `v7_override_prio` env path prints the same
  `Worker priority override set to ...` line the `-prio` flag path prints.
- `verify_server_and_rag.bat` hardening: escaped the parens that broke cmd
  block parsing; the `-prio` flag and env-override tests are independent (the
  env var is cleared for the flag test and restored after); positive server
  tests use `curl -sf` plus JSON content parsing (the 405 test stays anchored
  to the status line); `threading_proxy.ini` is restored on every path with a
  stale-backup guard; the zero-claim probe requires the specific `batch_small`
  FATAL, not any nonzero `build_db` exit; `stop_server` rc and `curl` presence
  are checked. The ThinkLimit block was removed.
- `fetch_qwen3_8b.bat` now downloads `model.safetensors.index.json`: the sharded
  5-file Qwen3-8B checkpoint needs the index manifest for the exporter's
  shard-completeness verification (`load_hf_model` refuses a sharded checkpoint
  without it).
- `tools/export.py` `stage_write` fixed: it wrapped the string returned by
  `deferred_write` in a `with` statement, so every full export crashed at the
  final manifest write (checkpoint already published at that point, manifest
  silently missing); the manifest path now publishes via plain `deferred_write`
  + `os.replace`.

- Correctness sweep (external audits applied, all validated): scalar `EVAL_SIGMOID`
  no longer destroys its polynomial (split-fill single-row SwiGLU was garbage --
  the wrapped-streaming 32/6 -> 38/0 root cause); sampler temperature path no
  longer clamps logits pre-max (tiny temperatures collapsed the distribution);
  top-p nucleus now extracts the SMALLEST prefix with cum >= topp (equality no
  longer over-selects) and falls back to an exact full-vocab selection when the
  top-256 heap cannot cover topp (wide-nucleus hang fixed with an
  n0==vocab_size guard); `v7s_sample` saves callee-saved RDI (scratch-table
  base was leaking to callers); the sampler is NON-DESTRUCTIVE (normalized
  probabilities land in a permanent 256k-entry BSS table, the caller's array
  keeps raw model logits -- `v7_get_token_probs` correct at any call time, and
  a SectorLogo Reject re-samples the untouched raw logits; the per-token
  preserve/restore design was discarded); opener system prompt is
  engine-owned (`g_sys_prompt_owned`, mdb block) -- continuation byte-match no
  longer dereferences a caller-freed pointer; `c_rw_muted` initialized at chat
  entry (uninitialized read corrupted the eviction warm latch on ordinary
  turns); `v7_set_position` FATALs rewinds of a wrapped StreamingLLM session
  (rolling-window rows are overwritten, no snapshot exists); CallLibrary DLL
  handles now live in a name-keyed owner registry (one loader reference per
  distinct DLL for the process lifetime; the atom-key memo cache and its
  GC/re-init wipes are gone)
- Inline-tuner soak harness: `examples/tuner/inline_tuner_test.c` runs the
  prompt for a configurable number of hours (default 12) and logs
  generation/seconds/tokens/tokens-per-second to `tuner_soak_log.csv`
- `v7s_trace_exec` and the `[exec]` spawn tracing removed (export, BSS gate,
  autoexec wrapper, inference_example `-exec` flag)

- In-process Logo tuner restructured to the one-script-per-verdict contract
  (AGENTS.md Sec.204): the ASM decision epilogue evaluates exactly ONE
  verdict script per accept/reject (`tune_accept` / `tune_reject` DB keys,
  each logging the decision, rotating to the next shape and generating the
  next candidate itself); `tune_get_candidate` and `tune_on_decision` are
  gone. The dead `tune_reset_candidate` function is removed. autoexec.logo
  drops the six wrappers for exports that no longer exist
  (TuneSetWarmup/TuneSetAA/TuneSetMargin/TuneSetEnabled/TuneRandomParam/
  TuneRandomDelta). tune_init arms `tuner:start_shape` (build_db writes the
  smallest supported resolution, Sec.205) instead of the hardcoded
  1024x3072; rotateevery is 1 so every decision advances the shape; XL
  shapes (dim > 16384) keep inter-pair delay 1; the rotation universe comes
  only from the engine's active-shape registry export. The duplicate
  examples/slow_drift_tuner demo is deleted; docs/api.md,
  docs/sectorlogo_manual.md and the tuner example docs match the live
  export surface. cpu_gate's RDRAND justification now names its real
  consumer (autotune_dispatch.c hardware RNG draws). New
  tools/update_ini_from_tuner.py feeds the tuner's learned tuner_state.txt
  mirror back into threading_proxy.ini.

- C autotuner review fixes (AGENTS.md Sec.202): the ruler is no longer a
  nulled config -- removed the `bigram_base_snap.values[20] = 0` taper-nulling
  and the pf arrays now ride in every snapshot (snapshot_pf), so neither
  taper nor pf distances start at zero; without a committed INI config the
  ruler is a frozen random genome R (draw_random_snapshot), never the uniform
  baseline; probe #0 and the baseline-seed branch are gone and every probe
  is a random genome. The 2%/5% margins, the annealing schedule, and the
  50/50 dispatch/prefetch + dist/type weighting are removed: the hill-climb
  now does one plain uniform redraw per round over all active fields,
  ratchets on any ratio > 1.0 and reverts on none (a measurement error
  never kicks the walk). Refinement uses best-of-N (the median machinery and
  the +-2% adopt gate are removed), +/-1/+-2 absolute neighbour steps (no
  range-derived /64 divisor), and tr > 1.0 adoption. pf_probe_dist is now
  a plain uniform draw (no disabled-band special case, no cache-line
  alignment, no log skew). The dead `pick_mutated`, `pf_mutate_dist`,
  `pf_mutate_type`, `pf_probe_type` helpers are deleted; the
  `*(int *)(ptr + off)` casts in find_matched_block/apply_snapshot/
  snapshot_pf/snapshot_params are now typed int-pointer indexing;
  init_random_int8 fills one RDRAND per byte. Calibration uses the direct
  measured estimate (no 0.95 fudge) with an explicit finite check;
  parse_cli_double uses isfinite. The write-back and final results no
  longer report a "within noise" band -- a ratio either wins or loses.
  The ruler rule is now documented in AGENTS.md Sec.202 rule 6.

- Tuner lifecycle hardening found by the end-to-end tuner proof: a fresh
  inference context (v7_free_inference -> V7_init reload) wipes every
  non-persistent tuner:* DB key while the tune_inited BSS latch survived, so
  tune_init never re-ran and the tuner scripts ran on missing state (the old
  scripts silently degraded into defaults; the new windows would correctly
  refuse NIL arithmetic and FATAL). The latch is now invalidated at all three
  table-wipe sites (V7_init reload + empty-db paths, v7s_inference_destroy),
  so the next chat re-runs tune_init, which re-seeds every state key and
  replays the learned configs from the tuner_state.txt mirror. The
  active-shape registry now records only TUNABLE shapes (a cold dispatch that
  fell back to the [default] block never enters it), so the rotation universe
  can never contain a shape TuneSetShape would FATAL on; 1024x4096 (a real
  Qwen3-0.6B dispatch shape) got its own [NxD] INI section and
  supported_resolutions entry. Savestate's cursor line is fixed: a bare
  DatabaseGet as an infix-chain operand swallowed the "_ ..." tail
  (AGENTS.md Sec.146 fact 47), corrupting the mirror's first line for its
  whole life. tuner_proof now passes end to end: 7 learned configs
  persisted, negative pf_dist values (-1212, -1128) living in accepted
  baselines.

- Background tuner feature-space repair (AGENTS.md Sec.202): the candidate
  window tables in src\logo\tuner\tune_get_candidate.logo used the C
  autotuner's display indices instead of the raw v7s_tune_set_param slot
  map, clamping the wrong slots -- taper_shift (raw slot 1) was pinned to
  [1..64] (legal 0 and 65..112 unreachable), raw batch_huge/caller batches
  could reach TuneSetParam with values the engine FATALs on, live threshold
  slots (raw 8, 17-19) were squeezed into pct/group windows, and the
  init-only skip mutated batch_huge_thresh (raw 8) instead of
  chunk_multiplier (raw 9). Windows now follow the raw map exactly:
  unguarded thresholds get no clamp at all, zero-legal params floor at 0,
  the taper/W_min pairing is enforced conditionally (taper pinned to 0
  while the candidate min_chunk_cols < 4, min_chunk_cols floored at 4 while
  the candidate taper > 0), and prefetch distances are clamped only to
  +/- half the weight-matrix size in RAM (negative distances legal; the
  engine's dist >= 0 <= 32 disabled band is no clamp). tune_init's two
  dead-prefix `(Or ...)` calls -- always truthy since the prefix removal,
  so every rotated shape got inter-pair delay 1 -- are now infix `||`.
  examples\slow_drift_tuner gets the same raw-map windows plus a pick limit
  to its 24 dispatch-param slots (TuneRandomParam draws all 106 slots, and
  a legal pf_type draw above 6 FATALs the engine).

- SectorLogo && / || are now infix-only: the expression-dispatch entries
  that ran `&& a b` / `|| a b` as prefix two-argument calls are removed
  (the dead .p_and/.p_or handlers with them), so a prefix spelling
  self-evaluates as a dead name -- the same doctrine as the removed
  AND/OR/NOT word forms. The infix chain, `!` (prefix NOT binding one
  primary), and every existing script/test usage are unaffected; the
  validator pins the dead-name behavior with IsWordAtom/IsStringAtom
  tests.

- Autotuner tunes the prefetch distances and types: autotune_dispatch now
  reads all 82 pf keys (pf_dist_0..40 / pf_type_0..40) from the per-shape
  [NxD] INI section (a missing or out-of-range key is a FATAL), draws and
  mutates them for the slots the benchmark's kernel executes (Q8 matmul
  slots 0-7, F32 matmul slots 8-10), runs a coordinate-descent pass over
  them in refinement, and writes the full key set back with the existing
  rollback protection. Because the dispatcher's level-2 prefetch cache only
  re-patches on a shape switch, apply_snapshot calls the exported
  set_prefetch_distance/set_prefetch_type directly for every changed slot
   (distance before type -- the distance patch re-enables a NOPped slot,
   and the type patcher skips frozen ones). Distances are tuned across the
   whole legal window [-(matrix bytes / 2), +(matrix bytes / 2)] --
   negative distances are real backward prefetches, and the unsigned
   [0..31] band is the engine's disabled-slot regime, a legitimate
   configuration; types in [0..6]. v7_lib.h resolves both SMC exports
   (their names carry no v7s_ prefix); docs/tuning.md's autotuner section
   updated.

- Review-round hardening batch (7 revised reports applied): v7s_writefile
  gains the exported-entry treatment of its sibling v7s_readfile --
  ENGINE_GATE_SFV + PUB_CPU_GATE on a widened 7-register SF frame with
  ENGINE_LEAVE on the shared return path, and the now-redundant
  push/pop rdi pair around STR_LEN is gone; exec_spawn_capture drops the
  dead SI_ZFILL6 zero store (the hStdError store writes the same 8 bytes
  two instructions later) together with its define and undef; the tuner's
  candidate boundary refuses min_chunk_cols = 0 (jle, was js) and arms a
  quantize_chunk_groups >= 1 guard with its own FATAL, so a candidate the
  engine would FATAL on can no longer be armed through v7s_tune_set_param;
  set_prefetch_distance's FATAL text drops the stale [0..48] slot range
  (the table is 41 slots); comment corrections in the q8 tail path and the
  vision rope builder replace dead line references; sectorlogo_fn's
  DefineCommand drops a dead reload of the just-parked fmt pointer; the
  attention scale hoist's comment now states why the 1.0f constant cannot
  fold into a VDIVSS memory operand (the proposed form is not encodable --
  VEX.vvvv requires src1 as a register).

- Review-round hardening batch (9 revised reports applied): db_load's three
  ReadFile loops abort with a named FATAL on a failed or zero-progress read
  instead of spinning forever on a truncated database, and a corrupt
  comp_size above the read staging buffer is refused before any read (both
  mirror the kv_cache_file read-loop guard); the raw-entry parity-call
  recipe in v7_set_streaming is single-sourced in the SS_RAW_CALL macro;
  stream_evict's eleven SEF_* file-local defines are undef'd at EOF and its
  emitter clobber contract names rsi/rdi; the three hand-expanded free
  sites in inf_cleanup consolidate into FREE_BSS; the write-only
  g_vit_scratch_bytes global is gone and the ViT pointer wipe matches the
  11-qword block; chat's APPLY_PRESENCE_PENALTY fuses the logit load into
  a negated-penalty add (bit-identical), the rewrite prefill loop uses the
  cached vocab pointer, and a raw-mode empty prompt ends the turn cleanly
  instead of reaching prefill's do-while countdown; sl_sort_coerce_num
  parses fractions with the literal parser's forward scheme (one rounding,
  SF_ATOM_DIGIT_MAX cap) so coerced atom keys bit-match their literal
  spelling, with validator coverage in v_sectorlogo_validate; the vision
  rope builder's head_dim contracts (multiple of 16, <= 128 under
  USE_ROPE) are enforced by the import in db_build_tool.c; the per-shape
  INI sections carry the full dispatch-key comment set.

- StreamingLLM rewrites enabled: the SectorLogo REWRITE action no longer
  FATALs when streaming is on. The rewrite is tiered by wrap state: at
  pos <= C the legacy diff/rollback/re-prefill runs verbatim with eviction
  logging muted for the re-prefill's duration; at pos > C the engine performs
  a full restart rebuild: the recorded prompt history (out_tokens below the
  old base position) replays through bare forwards from position 0 and the
  new stream re-encodes and re-prefills on top of it (the sinks keep the
  session's original first tokens; GetTokenPosition restarts with the
  rebuild; sessions restored from a KV file have no recorded token history
  and rebuild the replacement alone). The pending eviction batch flushes
  before any rewrite mutation so stream_evict_meta positions stay truthful,
  and the warm latch is saved/restored on every rewrite exit (sample,
  finish, and the empty-rebuild diversion). v_stream_validate gains tier-1
  and tier-2 rewrite sections (s7/s8) asserting liveness, rebuilt output,
  exactly-once eviction delivery across a rebuild, and the prompt replay
  (an arithmetic first= bound that a replacement-only rebuild fails).

Changes since 2.31.2 (StreamingLLM, verify-system redesign, review-round
hardening; ordered by release risk):

- StreamingLLM unbounded-context generation: a stateless ring KV cache with
  attention sinks, three-phase query rotation and the configurable fill
  grade (`v7_set_streaming(enable, num_sinks, window_size)`; C = S + W,
  capped by seq_len). Per-token slot resolution in both forwards, a
  three-phase single-head attention kernel (fresh/sinks/stale with delta
  re-rotation of the query only), an eviction ring with batched eviction delivery
  delivery over the microdb keys `stream_evict_text` / `stream_evict_meta`
  and the SectorLogo callback script under `on_stream_evict` (pieces
  decoded with the chat semantics), wrap-aware KV save/load with a warm-up
  latch, and a nested-generate enable swap that restores every legacy
  backstop. New public export `v7_set_streaming` (+ the
  `v7_set_streaming_fill` convenience inline), documented in docs/api.md.
- `v7s_sectorlogo_eval_script` gained a fourth dispatch key
  (`on_stream_evict`, own parsed-prog cache slot, GC root 9b, arena-reset
  entry); previously it only evaluated the three tuner scripts and every
  other key silently returned -1.
- `tools/v_stream_validate.exe`: the StreamingLLM validator (small-window
  wrap S=4 W=8 C=12): setter contract (pre-init FATAL, arg rejects with
  gate pairing, wrapped-disable refusal), T1 legacy bit-equivalence in F32
  and Q8, wrap + eviction-callback delivery (first=4 batch tag), and the
  stream_ref.h oracle boundary shapes. Registered in verify.bat Phase 3.
- verify.bat is the lean ASM-component gate again: build, Qwen3 DB, 20
  validators, generate-path validator, 10 inference smokes with the
  counter contract (EXPECT_INF=10). The vision gates moved to the new
  `verify_vision.bat` (Qwen3-VL byte baseline + LLaVA anyres gate), the
  server/RAG session gates and the CLI/environment contract tests moved to
  the new `verify_server_and_rag.bat`; the gate self-test experiment was
  discarded by owner decision.
- Release-review hardening batch: db_build_tool.c silent-failure paths
  wrapped in checked calls with named FATALs (alias ht_add, .newline
  embedding, template reads, missing threading_proxy.ini), the
  `__sys_vision_prompt_fmt` blob copy bounded against its 256-byte BSS
  buffer, NULL-prompt border FATALs on the public chat/generate entries,
  LFS oid sha256 verification in the fetch tooling, engine_death.c grown to
  16 border cases (the oversize prompt-fmt case plants the minimal 5-key
  fake DB so init reaches the vision block).
- StreamingLLM review rounds 1-4 (all fixes verified by
  v_stream_validate): the eviction-batch cursor accumulates across chunk
  deliveries, the documented `first=` meta prefix is emitted, the meta
  length is NUL-exclusive with 64-bit arithmetic, disable-before-validation
  reorder (v7_set_streaming(0,0,0) now disables), enable stored as the
  constant 1 only after the ring is allocated, 40-byte shadow regions
  around the allocator calls in the raw setter, reset=1 clears the warm
  latch and pending evictions only for real streaming sessions (a nested
  v7_generate preserves them), prefill-routed prompts fill the
  eviction ring, the wrapped path skips the M-RoPE compression delta
  (slot + delta could index the RoPE tables negative), the setter refuses
  enabling on M-RoPE-capable models (a nonzero `__sys_mrope_bound` in the
  DB), disable refused with -1 once the cache is wrapped (pos > C),
  oversized chat output_len silently clamped per the Sec.154a border rule,
  and the unreachable chunk-split path removed from the flush (a batch
  provably fits the 64 KB buffer: 256 evictions x 128-byte pieces).
- v_stream_validate grown to 25 checks: nested v7_generate mid-session,
  multi-batch turns with an accumulating probe, wrapped KV save/load
  round-trip, and the enable/disable/enable ring-realloc cycle.
- check_release.py: the no-git walker (fresh archive unpack) now keeps
  BINARIES_ALLOW paths even when .gitignore would drop them -- the gate
  passes inside a fresh unzip, proven on git archive output.
- docs/streaming.md added (user-facing StreamingLLM semantics, callback
  contract, limits); the `stream_evict_text` / `stream_evict_meta` /
  `on_stream_evict` keys documented in docs/database_keys.md and the
  SectorLogo manual's eval_script table.
- Micro-optimizations and comment-accuracy fixes from the opt_and_bug
  rounds: add_rmsnorm rare loops moved out of line (common case
  falls through), vit_anyres patchify accumulator moved from a frame slot
  into r14d plus an AR_FREE_RESET macro dedup, vit_rope_apply strides
  derived from the cached rope_half instead of a hardcoded 256, the
  SectorLogo lg_ensure_word dominant WORD case falls through with zero
  taken jumps, and the unknown-word warning parks its position class on
  the stack (the tail pick previously tested the stdout handle and always
  printed the statement-position prose).
- Post-release review round 2: the eviction-log push in both forwards gained
  the EVICT_LOG_CAP bound (an undrained raw-forward loop with streaming
  enabled now FATALs with a named reason instead of corrupting the BSS
  behind the log), the rewrite re-prefill loop dropped its two provably
  dead defensive branches (the hook-rewrite entry FATAL on streaming and a
  clamp behind a `jae` that already stopped at the same bound),
  v_stream_validate s5 asserts the in-turn eviction delivery before the
  masking diagnostic eval_script, db_build_tool.c init allocations gained
  NULL checks with named FATALs, the KV-cache blob formula in docs/api.md
  corrected to `min(pos, C)` with the streaming save/load coherence
  contract (S/W/C must match; the W-token warm latch documented), the
  streaming behavior contracts written into docs/api.md (delivery cadence,
  session ownership, `v7_set_position` interplay), v_stream_validate
  catalogued in docs/tools.md, the validator-count claims trued to 20
  everywhere (RELEASE.DIZ also fixes the stale benchmark count), and
  check_release.py derives the validator count from `tools/v_*_validate.cpp`
  and fails any disagreeing present-tense claim.

## [2.31.2] â€” 2026-10-21

Changes since 2.31.1 (anyres correctness, LLaVA gate, silent-failure
hardening, gate upgrades; ordered by release risk):

- Anyres image-blindness fixed: the rewritten `__sys_vision_prompt_fmt` parse
  in `inf_microdb.inc` skipped blob[0] with an inc-before-test scan, so an
  EMPTY prefix blob (LLaVA-OneVision's is `00 "<image>" 00 "\n" 00`) parsed
  the marker as the prefix and the image placeholder never reached the
  prompt -- the model answered blind ("person's face" instead of "tabby").
  Fixed with a test-before-inc scan; empty prefixes are now the documented
  legal case.
- Anyres centered-pad origin fixed: the fit-origin computation scaled
  left_pad by 3 twice (once into r15d, again at the origin step), shifting
  the pad block by 2*left_pad pixels and overrunning the page-exact canvas
  on height-limited images -- a write access violation on the 200x1001 tall
  fixture. Width-limited images (left_pad = 0) were unaffected, which is why
  every previous gate passed. r15d now stays in pixels; the scale happens
  once at the origin step.
- New LLaVA anyres gate in verify.bat Phase 5
  (`run_vision_smoke_llava.bat`): 3 fixtures (canonical cat640, extreme-wide
  1001x200, extreme-tall 200x1001) x F32/Q8, byte-compared against shipped
  baselines with visible PASS/SKIP/FAIL decisions and a baseline generator
  bat. The Qwen3-VL vision smoke baseline is unchanged.
- Server demo backgrounds converted from 12.2 MB of PNG to 1.9 MB of
  quality-85 JPG (the avatar was even duplicated at the repo root without
  being synced there); `ardmark_rag.logo`, THIRD_PARTY.md and the README
  package tree updated to the .jpg names.
- verify.bat Phase 5 hardening: the four paris smokes now require exit code
  0 (a smoke that printed paris and then crashed no longer counts as PASS),
  the zero-claim-size probe shows the build_db FATAL it exists to prove
  instead of burying it in nul, verify anchors itself to its own directory
  (the batch_small probe, kv_cache's KV files and build_db's INI read were
  CWD-dependent), and the summary enforces the 19-validator/13-inference
  counter contract.
- build_db exit-on-failure: a failed ht_add now exits 1 (an incomplete
  database could ship with exit 0), db_save failure returns 1, truncated
  .tokenizer files FATAL instead of substituting -FLT_MAX/empty strings,
  templates refuse the 1023-byte slot overflow instead of silently cutting
  mid-token, and the v7_init(0) return is checked.
- export_vlm.py: the dev-only `v7_vlm_ref` import is lazy and fails with a
  named, actionable FATAL (restore from git history) instead of a bare
  ModuleNotFoundError at import time; the dev-tree requirement is documented
  in docs/setup.md.
- Freshness-gate manifest repaired in both directions: 4 built-but-unrecorded
  sources added (v7_decide_example, spam_classifier, v7_only_safe_check,
  tuner_proof -- editing them could certify stale binaries as fresh), the
  never-built b_sectorlogo_loadtest record removed (its missing output kept
  the gate from ever reporting fresh), and a new check18 rule derives the
  build.bat target set and the manifest and fails on either mismatch.
- server.bat implements the supervised restart loop the docs always
  described (clean /quit?code=42 stops; any other exit restarts after 2 s);
  start_server.bat's dead byte-identical if/else collapsed and the
  stop-first call is documented.
- full_setup.bat deleted: broken as shipped (build_db without -o, retired
  models\ layout, -n 50 tests) and duplicating the fetch bats plus
  build_text_model_databases.bat; the release handbook ownership notes
  updated.
- bench_regression.bat: DB path is now an argument (the old script required
  bin\lz4ht.db, which no release flow creates), every loadtest is gated on
  its exit code, and AGENTS.md Sec.11/Sec.54 stopped claiming the benchmark
  is a verify phase (the deleted bench_regression.py references removed).
- run_vision_smoke.bat: missing DLL/DB/baseline are explicit SKIP decisions
  instead of a guaranteed MISMATCH in a fresh unpack; the vision gate
  fixtures and baselines now ship in the release package.
- verify_llms.bat shows every run's full output (stderr captured, not
  dropped); AGENTS.md Sec.144 documents the deliberate short-output smokes
  that were in unresolvable conflict with the -n >= 1000 rule.
- engine_death.exe covers v7_generate_image with 5 mirrored border cases
  (13 total); v_sampler_validate pins the negative/-inf temperature fix-up
  to 1e-6 and drops a dead pen_bits reassignment; docs/api.md documents the
  full temperature and v7_set_position border contracts and the internal
  vit-callback's no-inner-guard decision.
- Hygiene checker upgrades (the blind spots this round exploited): derived
  export/resolve counts vs docs/api.md, "N tested LLMs" vs the fetch-script
  count, a markdown backtick-corruption heuristic (docs/setup.md's every
  backtick became a literal v -- restored from git history), and root-bat
  references to missing non-.bat files (baselines, fixtures, INI).
- ASM comment checker gained two structural rules that would have caught the
  anyres regression class: unreachable code labels (never referenced, control
  cannot fall into them) and dead compares (cmp/test flags consumed by a
  clobbering instruction before any jcc/setcc/cmov -- macro-parameterized
  jcc like `j%1` are recognized). Both run tree-wide at zero findings.
- Documentation drift batch: Overview/paper1 now say 10 tested LLMs (the
  checker derives the count), the models.md matrix carries all 13 models
  (Qwen3-8B and SmolVLM-256M rows added) with the 36-qword VHDR table, the
  92-qword VCFG heading and the full slot 32-35 documentation, vision.md
  documents the aspect-preserving fit + centered black pad and drops the
  phantom VCFG_ANYRES_MAX_PIXELS, tools.md lists the 19th validator, and
  the research-logbook/decide-test references point at locations that exist.

## [2.31.1] â€” 2026-10-20

Release-audit fixes from the external 2.31.0 review (code-reading + Linux
assemble run; ordered by release risk):

- GitLab CI: the lint job now runs the cheap Python gates before NASM (a
  NASM failure no longer hides the release checks) and creates `build_tmp`
  before the first object write, so a clean checkout passes.
- `docs\AGENTS_release.md` (stale twin of AGENTS.md, contradicting tool and
  validator counts) is deleted; AGENTS.md is the single source of truth and
  the hygiene checker fails if the twin reappears.
- Public image border: `PUB_VALIDATE_IMAGE` now also enforces the vit DDA's
  16.16 fixed-point limit (img_w/img_h must be below 65536) with a named
  FATAL; the inner `vit_input_stage` guard moved to the border with it, so
  the same message is reachable from both image entries.
- Gates for the 2.31.0 surface: `engine_death.exe` gained 8 border-FATAL
  cases (img_w/img_h <= 0, 65536 limit, pixel-count overflow, NULL raw_rgb,
  negative v7_db_add len, NULL data with len > 0), `v_sampler_validate` a
  3-case border contract (quantize mode, negative max_gen, NULL encode
  pointers), and the new `v7_only_safe_check.exe` compiles against
  `V7_ONLY_SAFE` and completes a real chat through the public tier only.
  Removing any border check makes a verify case FAIL.
- Setup/build script references to deleted scripts fixed (full_setup.bat,
  fetch_smolvlm_256m.bat, build.bat's docs/plans pointer); the hygiene
  checker now flags `.bat` tokens that no longer exist in the tree.
- Packaging: orphan binaries (4 .exe without sources, a month-old stale
  DLL copy, `__pycache__`, an empty directory) removed; reviewer zips are
  built from tracked files only. FILEID.DIZ size tolerance tightened to 3%
  and the hygiene gate now runs after the link so the size claim is checked
  against the freshly linked DLL.
- Documented counts drift fixed (release AGENTS.md 129 -> 130 asm sources,
  49 -> 47 tool sources); the asm-sources count is now derived by the
  hygiene checker.
- Line endings: `.gitattributes` carries `* text=auto`, case-proof DIZ
  pattern, binary rules; the two mixed-line-ending files fixed; the hygiene
  checker flags mixed EOLs and stray CRs in document/code files.
- Vision (HF parity): anyres grid tiles are now aspect-PRESERVING fit into
  the candidate canvas + centered black pad (HF `_resize_for_patching` /
  `_pad_for_patching`, all-integer ceil-div), replacing the previous
  geometric stretch; non-square images were distorted into grid tiles
  before. Square inputs are byte-identical. Plus dead-code removals and
  comment/undef hygiene from the `opt_and_bug` apply loop (9 revised
  reports consumed).

## [2.31.0] â€” 2026-10-19

Public API border audit applied (external code-reading review): the public
tier in public_api.inc is the outermost layer and now refuses the
crash-capable caller mistakes with named FATALs. `v7_chat_image` /
`v7_generate_image` reject `img_w`/`img_h` <= 0, a `w*h` pixel count above
32 bits, and a NULL `raw_rgb` buffer (the vit stage reads the pixel count
as a dword, so a wrapped dimension allocated from a wrapped count and read
past the buffer). `v7_db_add` rejects a negative len (zero-extended to
~4 GB at the ht_add border) and NULL data with len > 0. `v7_set_max_gen`
rejects negative caps (the generation loop compares unsigned, so a negative
cap read as unlimited). `v7_set_sampler_presence_penalty` requires a finite
value in [-2, 2]; `v7_set_sampler_temperature` refuses +inf (NaN and
sub-epsilon inputs keep their documented silent fix-up). `v7_set_quantize`
accepts only 0 and 1. `v7_encode` requires non-NULL text, tokens and
n_tokens. Patch-size divisibility of image sizes stays unchecked by
decision: the vit grid floors and crops in-bounds, every read stays inside
w*h.

Database file header: the 20-byte lz4ht.db header now opens with the magic
`v7db` and a uint32 file-format version (currently 1). `db_load` refuses a
wrong magic with a FATAL ("not a v7 database file") and loads a foreign
version with the visible warning "db_load: warning: model file was made by
a different version". No validators are written for the file format; all
databases regenerate from the checkpoints. KV-cache save files deliberately
carry NO magic and gain NO validation -- the 36-byte data header + LZ4 blob
format is documented for reference in docs/api.md and the v7_lib.h comment,
and that documentation is the whole feature (owner directive).

Structural gap closed at the header level, library untouched:
v7_get_internal_inference_ctx and the inference_context struct are
documented as internal (reading is diagnostic, writing through the context
is undefined behaviour), and v7_lib.h gains the `V7_ONLY_SAFE` define --
compiling a host with it removes every internal declaration (kernels,
MicroDB internals, encode/forward, SectorLogo internals, process execution,
diagnostic gates) from the header and resolves only the public v7_* tier.
The DLL exports are unchanged.

docs/api.md now states the input contracts of the tightened entries and
documents both file formats (lz4ht.db header, KV-cache data header).

## [2.30.0] â€” 2026-10-19

Defensive-code purge across the engine (five audit reports applied): every
check for a state the engine's own callers cannot produce is deleted --
dispatch-entry zero/negative size guards, impossible init-state FATALs
(rope head_dim cap/zero, group-size refusal, config geometry range
checks), allocator-pointer NULL tests in the forward/prefill paths, dead
vocab-bulk/merge_scores/template consistency checks, per-entry db_load
klen/overflow/completeness validation, mdb_decompress stream validation
(the compressor is the boundary; round-trip proven by v_mdb_validate),
kv_cache load-side format validation, the token_stream_dump range guards,
and the SetOutputToken runaway cap (SetOutputToken is stateless and has
zero engine-side limits -- the generation-length contract is the only
limit). The public API sampler setter is the one sanctioned data-fixing
boundary: it fixes sub-epsilon and NaN temperatures to 1e-6 silently, and
the sampler kernel no longer carries its own floor; the softmax-sum NaN
detector died with the floor (the sum is >= 1.0 by construction).

Import-side validation: build_db.exe now refuses models outside the
kernels' contracts (dim/hidden multiples of 64, head_dim in
[64..512] % 32, GQA divisibility, seq_len <= 131072, vocab <= 163840,
dim <= 65536, GS == 64, signed-int ABI geometry) -- the engine never
re-checks what the import wrote.

Dead SMC patch-macro family deleted (87 lines, zero emitted bytes);
PATCH_Q8_N gained the mfence/lfence serialization every other patch
writer already had. set_prefetch_distance/type now FATAL on out-of-range
slot/type instead of silently returning.

Real bugs fixed: BUILD_QT_PER_LAYER's GS input register was deleted with
the group-size check, corrupting every weight descriptor's stride (all
inference garbage) -- the load is restored with the check removed;
SectorLogo POWER answered +0 for pow(+-0, finite negative y) and answered
the +Inf-row value for pow(negative base, -Inf) -- both now follow C99
F.10.4.4 with validator coverage (two old tests that pinned the wrong
values corrected); the sampler temperature setter stored NaN verbatim
(vucomiss sets ZF=1 when unordered -- the NaN input took the greedy
equality branch); v7s_mrope_prepare re-freed and re-allocated both rope
scratch tables on every context growth -- allocation now happens once at
full table capacity.

Review batch: GC root-11 pin loop byte-fills the bitmap (rep stosb plus
the exact partial trailing byte) replacing the per-granule bts walk;
sprintf lea-fused room checks, duplicate digit_buf lea, branchless cmovs
%d sign handling; string_hash tombstone compare in the imm8 form;
rmsnorm_rope_dispatch vcvtsi2ss normalized to the house 3-operand form;
mrope/sprintf/v7_macros math constants folded into memory operands;
console_width stash/reload around CloseHandle deleted; encode append
branch inverted (hot path falls through) and merge_scores load fused;
fbm_alloc/fbm_free dropped from the DLL exports (private fallback tier of
mdb_alloc_mem); runq references removed from AGENTS.md and Sec.199 (no
defensive programming, public API setters = the sanctioned data-fixing
place) codified.

## [2.29.0] â€” 2026-10-18

Second external audit release: every finding of both audits fixed or
consciously decided; the full verification gate (19 validators, 11
inference tests, SectorLogo 1402/1402, server and CLI suites) passes on the
release tree and the release tree is blob-identical to the dev tree except
the three documented exceptions.

### Added

- **`v7_get_version` export** â€” the DLL returns its release version (static
  string baked in `src/asm/version.inc`); declared in `v7_lib.h`,
  documented in `docs/api.md`, and `src/check_release.py` checks the
  literal against README, the CHANGELOG top entry and `FILEID.DIZ` so all
  four always agree.
- **NASM ships with the tree** â€” `tools/nasm.exe` (3.02rc6, BSD-2-Clause,
  license text in `tools/NASM-LICENSE.txt`, https://nasm.us), so a fresh
  clone or unpack assembles without any download.
- **Database parity** â€” every DB-building script (`fetch_*.bat`,
  `build_text_model_databases.bat`, `build_vit_databases.bat`,
  `rebuild_all_llm_dbs.bat`, `full_setup.bat`) now passes `-autoexec` and
  `-tuner` like `verify.bat` does, so the database a user builds carries
  the SectorLogo stdlib and tuner scripts exactly like the tested one.
- **Linux-runnable release hygiene gate** â€” `src/check_release.py` gained
  a no-git mode (walks the tree, so a fresh unpack can build), checks
  absolute dev paths outside the dev handbook, version identity across
  README/CHANGELOG/FILEID/DLL source, and backticked file references
  resolved by basename; wired into `build.bat` Phase 1 and
  `sync_release.bat`.
- **`.gitlab-ci.yml` lint job** â€” NASM assemble + both comment checkers +
  the hygiene gate run on every push under Linux.

### Fixed

- **SwiGLU Q8xQ8 validator gates every shape strictly** â€” the C reference in
  `kt_utils.h` (`swiglu_q8_c_ref`) is now a bit-exact port of the ASM kernel
  (exact int8 group dots in the interleaved SWKQ_DOT_32 lane layout, exact
  float conversion, per-group FMA accumulation in the kernel's order, HADD
  lane reduce, the kernel's float sigmoid), replacing the old
  double-accumulating reference. The known-limit escapes ("float-vs-double
  accumulation, NOT counted") are deleted; all 19 tests pass on every real
  FFN shape in both trees.

- **Sampler validator now gates all 40 tests bit-exactly** â€” the C reference
  in `v_sampler_validate` is a 1:1 port of the ASM sampler (minimax
  polynomial exp with the FMA chain and RNE range reduction, the 4x8-lane
  accumulator summation with the tree-reduce/hsum lane order, and the
  min-heap nucleus cutoff + max-heap extraction with identical tie
  semantics). The old approximate reference (expf, sequential sum, sorted
  full-CDF top-p) validated a different algorithm; non-greedy sampling was
  the last documented open item and is now closed. The nucleus regression
  test's logits-buffer reuse across draws (v7s_sample consumes the buffer in
  place) was also fixed; verify.bat runs the validator as a gate again.

- **Review fixes for 2.28.0** (static-review findings, verified against the
  tree): the CPU gate no longer requires VAES â€” every `vaesenc` site is the
  VEX.128 form, so AES-NI + AVX2 suffices, restoring Haswellâ€¦Comet Lake and
  Zen 1/2 machines; 14 dead tool sources removed (9 consolidated-validator
  duplicates + 5 unreferenced debug probes); mojibake repaired in README.md
  and db_build_tool.c; the model table lists all 10 LLMs; FILEID.DIZ is
  tracked in git; docs/tools.md and troubleshooting.md match the built set;
  build.bat prints the promised b_sectorlogo_loadtest SKIP line; Visual
  Studio is located via vswhere (any edition, `VS_PATH` override) and NASM
  2.16+/3.x is enforced with the pinned `tools\` nasm taking PATH precedence
  (Go is optional); init FATALs loudly when head_dim is not a multiple of 32
  (protects the vector kernels and the qkv-bias OR-scan); the sampler and
  SwiGLU validators gate bit-exactly with zero known-limit escapes; the
  verify.bat header lists all 7 phases; `dbg_image_recog` docs show the
  missing dll argument; broken research_logbook links reworded; AGENTS_dev
  Sec.104 trued (zero accepted validator failures remain); setup.md states
  nasm.exe ships outside git; THIRD_PARTY documents the shipped fixtures.

### Known issues (audited 2026-10-17)

- The release git log and changelog were audited for open items: the
  [2.20.0] known issue (atom-GC sweep cursor byte/granule mix, reclaim path
  dormant) was **resolved in [2.22.0]** â€” both conversion sites fixed, with
  the dedicated stress harness the issue demanded (intern past the soft
  limit, live-atom survival assert) â€” and GC hardening continued in
  [2.28.0]. The sampler non-greedy divergence (formerly the only remaining
  documented open item) is closed by the bit-exact validator port above; no
  documented open items remain.

- **Second-audit hardening (2026-10-18)**: `build.bat` defines `THIS_DIR`
  before using it (the server-stop path filter was inert); a fresh unpack
  builds without git; shipped scripts are relative-path clean (the example
  runners, `rebuild_all_llm_dbs.bat`, `build_web_ragg.bat`,
  `build_mdbqk.bat` â€” vswhere instead of a hardcoded VS path) and the
  dev-only `sync_release.bat`, `tools/write_validator.py` and
  `bug_expert_consultation.md` left the release (the post-mortem also left
  the dev tree by owner decision); the release root `zydis_debug_header.h`
  is the no-op stub again; README's Quick start matches reality (build â†’
  fetch â†’ verify â†’ examples, `bin\` DLL paths); `threading_proxy.ini`
  exists once (the stale `src\` copy is gone and the comment checker
  validates the live one); the public API header and `docs/api.md` cover
  every export; the model licenses are documented in `THIRD_PARTY.md`, the
  fetch tooling records the resolved upstream commit per model
  (`REVISION.txt`) and prints checksums for reproducibility; FILEID.DIZ is
  true (no SAM, real validator count and DLL size, `cat640.raw`).

## [2.28.0] â€” 2026-10-17

### Added

- **FILEID.DIZ** â€” classic BBS-style file description (â‰¤45 columns) at the
  package root, replacing the dev-only RELEASE.DIZ; synced into the release
  by `sync_release.bat`.
- **Canonical vision test image tracked and synced** â€” `cat640.raw` +
  `cat640.meta` (the canonical test image the docs reference) are now in git
  and synced to the release; the server demo backgrounds
  (`backgrounds\ardmark_dragon.png`, `ardmark_winter.png`) are synced too.
- **Quiet validator output** â€” every validator now prints only failures and a
  single `.` per passing test instead of one PASS line per test (the
  SectorLogo validator alone dropped from ~1400 `TEST:`/`PASS:` line pairs to
  one dot per test). Failure lines keep the full details and the
  `TEST: desc [script]` context; end-of-run summaries and exit codes are
  unchanged, so verify.bat gating is unaffected.
- **Qwen3-1.7B support** â€” database built and verified end-to-end on both
  F32 and Q8 paths (dim=2048, 28 layers, GQA 16/8, head_dim=128, tied
  embeddings, QK-norm, rope 1M, 151936 vocab; 1.8 GB `lz4ht.db`). Added to
  `build_text_model_databases.bat` and the `verify_llms.bat` model list.
- **Ardmark demo server targets Qwen3-1.7B** â€” `ardmark-server.bat` now
  points the demo at `weights\databases\qwen3_1.7b\lz4ht.db` via new
  `V7_SERVER_DB` / `V7_SERVER_MODEL` environment overrides in
  `server.bat` (default remains qwen3-0.6b for every other caller); the
  health-wait budget raised to 10 minutes for the larger DB.
- **Thought-injection presentation** â€” `thought_injection.html` (CPKI deck
  family, German, 10 slides): how v7 thought injection works (think-block
  read/write loop, the five pipeline stages, the seven control-text files,
  the phrase-level mechanism analysis), the Gemini capture as evidence that
  the texts steer frontier models too, and two labeled pages written from
  the agent perspective (commissioned by the user) describing the texts'
  effect on the agent that built the deck.

### Changed

- **build.bat needs no Go** â€” the ASM comment checker's auto-formatter
  (`src\nasmfmt\nasmfmt.exe`) is committed prebuilt, so a fresh clone builds
  without Go at all; only if the exe is absent does build.bat build it once
  from the vendored source (Go 1.21+, documented as optional in
  `docs\setup.md`).
- **Dead SMC patch-macro family deleted** â€” twelve never-expanded macro
  templates and one dead `%assign` removed from `smc_macros.inc` (~65 lines,
  zero emitted-byte change: none of the twelve has an invocation site, and the
  live machinery â€” `PREFETCH_SLOT`, `PATCH_Q8_N`, the prefetch patch walkers â€”
  is untouched); the doubled end-of-file separator fence in
  `token_stream_dump.inc` removed; two degeneration-path comments in
  `vit_dispatch.inc` corrected (stale line references and a false
  `<= M/4` claim replaced with the correct M<4 tail-fold justification).
  Applied from `opt_and_bug` reports (applied and deleted).
- **Release-packaging hardening** â€” the two pre-publication audits worked
  off: `verify.bat` restores `threading_proxy.ini` by byte copy instead of
  `git` (works in a fresh unzip with no VCS); all 81 `.bat` files normalized
  to CRLF with a `*.bat text eol=crlf` `.gitattributes` guard; build/verify
  artifacts (`.obj`/`.exe`/`.log`/`__pycache__`/`asm.zip`) purged and
  gitignored; `build.bat` prints its NASM version and the docs pin 3.02rc6;
  the four variable-shift lines use the portable `cl` count form; README
  quick-start references fixed (`inference_example.exe`, NASM version, logo
  count, phantom header).
- **Zydis becomes dev-only** â€” the release ships without `Zydis.lib` and the
  `zydis-master`/`zycore-c-master` trees: `build.bat` probes for real Zydis
  and, absent, compiles every tool against the no-op stub in
  `tools\zydis_stub\` (same `ZydisInit`/`ZydisSetCrashLogFile` interface, no
  decoder, no crash dumps). Dev always keeps the real decoder. Documented in
  `docs\zydis_debug_readme.txt`, README and `docs/setup.md`. `sync_release.bat`
  installs the stub into the release root.
- **Release build tolerates a missing `sectorlogo_c\`** â€” the port is
  test-only (AGENTS.md Sec.197) and stays out of the release; `build.bat`
  skips `b_sectorlogo_loadtest` with a visible SKIP line when the port is
  absent.
- **Ardmark RAG relocated to `examples\ardmark\`** â€” the Viking-storyteller
  demo (Bumwolf the Strong) is the project's prime demo (AGENTS.md
  Sec.198): moved out of `examples\research\`, all internal and external
  path references updated, and a full README added (architecture, run
  instructions, corpus, calibration tools).
- **Server console output slimmed to warranted lines** â€” the endpoint
  banner, the stop-with-stop_server.bat instruction, the sampler-settings
  line, turn-log fences, params meta-commentary and char counts, and the
  cleanup/done chatter are gone. The log keeps the load anchors, the
  per-request request line, prompt/answer/reasoning content, the
  rescue-ladder lines, warnings and FATALs.
- **Ardmark rag banner texts reworked** â€” every PrintBanner/Print label is
  brief and factual ("think closed; thinking so far:") instead of full
  sentences.
- **Ardmark primer history preserved for benchmarking** â€”
  `examples/ardmark/primers/` holds all 18 distinct prepend
  primers from git history (prepend_1.txt oldest .. prepend_18.txt
  current), with primer_index.txt mapping each to its commit; benchmark a
  primer by copying it over ../ardmark_prepend.txt.
- **One validator per module: five validators integrated and deleted** â€”
  `t_sectorlogo_gc` (15 cons/atom GC gate variants), `t_intern_reps`
  (timing-only probe, nothing to integrate) and `v_settoken_validate`
  (whole-state-switch proof, now suite 100) folded into
  `v_sectorlogo_validate`; `v_swiglu_q8_validate` into `v_swiglu_validate`
  (both modes); `v_q8_fused_validate` into `v_fused_validate` (F32 + Q8
  sections); the `t_mdb_bigblob` large-blob ladder into `v_mdb_validate`
  (512 MB compress/decompress); the norm family consolidated into
  `v_rmsnorm_validate` (4 sections: RMSNorm, RMSNorm+quantize,
  RMSNorm+RoPE, fused add+RMSNorm). One monolithic validator per module,
  no test-disabling parameters (SectorLogo excepted), codified in AGENTS.md
  Sec.11a.
- **Tests are never bat files; tests are not examples** â€” verify.bat is the
  only testing bat and gained phases 5-7: the engine test programs moved
  from `examples/c_inference/` to `tools\` proper names
  (engine_death/kv_cache/chat_only/generate_only/multiturn/
  multiturn_manual/image_multiturn) and run inside verify; new Phase 6
  server tests (endpoints smoke + /v1/decide session transparency with its
  own server instance, request bodies as `tools\test_data\` fixtures after
  inline-JSON cmd mangling broke multi-message bodies); new Phase 7 CLI and
  environment contract tests (prio flag/env/malformed-FATAL, UTF-8 prompts,
  ThinkLimit sidecar, zero-claim-size FATAL-at-creation guard). The only
  test bats are the three long-running suites: `run_vision_smoke.bat`,
  `test_vlms.bat`, `verify_llms.bat`.
- **v_rmsnorm_validate built without /O2** â€” an /O2-inlined copy of the
  fused add+RMSNorm reference helper miscompiles inside the large main
  (the reference came out as the kernel output times a per-test constant;
  per-binary deterministic, immune to every source-level restructure,
  clean at /Od and in small mains). The kernel under test was never wrong;
  the unoptimized validator is the fix (validator, not hot path).
- **Leading whitespace stripped from all 121 src/asm .inc files** â€” the
  indentation restored mechanically by the comment checker; ~220
  NOCOMMENT violations fixed by hand (one hand-written comment per site).
- **Console ReadKey blocking bug fixed** â€” the console path consumed the
  accepted key-down record with ReadConsoleInputA and then called ReadFile
  on the drained console: the engine parked until the next keypress and
  returned that next key (every ReadKey lagged one keypress). The
  character atom is now built directly from the peeked record; the console
  path performs no ReadFile.
- **GC Root 11: the primitive-name region below the dispatch high-water
  mark is pinned** â€” the pin-floor contract in
  `sectorlogo_eval_dispatch.inc` ("no atom below this offset is ever
  reclaimed") existed only as a comment; a collect could reclaim a builtin
  dispatch key, its offset would be reused by a data atom, and the
  dispatcher would route that atom to the primitive's handler. The first
  implementation hung the validator's GC-gate suite (the SL_ATOM_BIT macro
  leaves eax holding the 0-7 bit index and the loop cursor rode eax â€” an
  infinite 1<->9 cycle); the cursor now lives in r10d.
- **The in-system A/B tuner actually ships** â€” `src/logo/tuner/` replaced
  with the proven slow_drift_tuner scripts (the stripped variants never
  wrote tuner_state.txt), and verify.bat builds the Qwen3 database with
  `-tuner src/logo/tuner`; `tuner_proof` passes (cursor + learned config
  persisted).
- **v_sectorlogo_validate: per-suite timing log and slow-suite gating** â€”
  every executed suite's elapsed time is printed and appended to
  `build_logs/v_sectorlogo_validate_timing.log`; 20 slow suites (dominated
  by ~1 s engine-recovery joins and console-poll sleeps) are skipped by
  default and available via `-with-slow` (include) / `-only-slow` (run
  alone). verify.bat runs the validator with `-with-slow` so the
  definitive gate keeps full coverage.
- **bm25 benchmark templates swept to the infix-only language** â€” and the
  accidentally dropped `MAKE "qk"` accumulation line restored (the
  suspected "engine bug" behind an empty query map was this dropped line;
  the consultation write-up is `bug_expert_consultation.md`).
- **Example driver bats refreshed** â€” `run_inference_examples.bat` /
  `run_tool_examples.bat` referenced renamed binaries (`test_*` prefixes)
  and a wrong argument shape; both suites pass against the current bin set.
- **Bicubic tap-offset crash fixed** â€” `BC_CLAMP_X_OFFSET` tripled the
  unclamped x0 in eax instead of the clamped value in edx; with eax
  chaining across the four tap calls the offsets multiplied x4 per call
  (tap 3 reached x0*192) and `v_bicubic_dump` crashed on an out-of-bounds
  read. All 10 cases pass.
- **30 revised opt_and_bug reports applied** â€” the console ReadKey fix
  above plus comment corrections (chat.inc stale line references, the
  mdb_readonly flag's real semantics and its load-bearing dword width,
  stale/dangling comment references in sectorlogo_reader and
  sectorlogo_str_char_to_int); the other 25 reports verified clean.

## [2.27.0] - 2026-10-15

- **`v7_get_console_width`** â€” new export (src/asm/console_width.inc) reports the console's visible column count (default 80 for piped output); the SectorLogo banner logic pads labels to the real window width instead of a hardcoded number.
- **Prefill F32 V-bias gate** â€” prefill_forward_f32 caches the init-time bias scan (g_qkv_bias_active) per call and skips the V-bias zero-add loop on bias-less models (~1.4M dead instructions per 100-token Qwen3-0.6B prefill); bit-identical output for bias-bearing models, matching the existing forward_f32/prefill_q8 gates.
- **Ardmark RAG stream text externalized** â€” sidecar stream strings moved to per-purpose text files (answer, greet, invite, drive, walk-before/after, world base, bayes state); the prepend primer condensed; banner state pads, clears and dims per console width with cls+cursor control per banner group.
- **Ardmark retrieval quality** â€” bare-question BM25 floors plus a question-grounding gate (off-topic suppression), think+question query with a relate cosine gate, and a hardened bm25_rag tool.
- **SectorLogo naming rule enforced** â€” names are alphanumeric only; underscore dim-flag names renamed (the underscore is the string-concatenation operator); the EvalLogoStream statement-builtin inventory comment synced to the 27 registered builtins (SetOutputToken).
- **Example logo scripts swept to the infix-only language** â€” IfElse Equal probes to (=), Add counters to (+), comparisons and arithmetic to infix operators, SetToken to SetOutputToken.
- **opt_and_bug batches applied** â€” 44 revised reports across three batches (35 + 6 + 3): bug guards, call elision, loop restructures, deduplication and comment corrections across kernel, dispatch and SectorLogo sources; prefill spin-contract comments corrected to the Sec.4a flag-only semantics.
- **AGENTS.md** â€” pure-ASCII cleanup; the no-scripts rule for .logo and stream .txt files (Sec.146); a pending-todo continuation rule (Sec.181); the CORRECT_RSP/RESTORE_RSP module reference corrected to v7_apicall.inc (Sec.163).

## [2.26.0] - 2026-10-12

A server-contract and robustness release: the server owns the Qwen3
model-card sampler settings, Ctrl+C is no longer hooked, the server is
split into single-purpose translation units, and five audit batches fixed
the reported server, SectorLogo and example bugs.

### Added

- **Server: POST /v1/decide** -- typed decisions over HTTP: one JSON request
  (kind, state, question, 2..16 single-token option labels) returns the
  probability per label plus the winner (`docs/typed_decisions.md`).
  Session-transparent by design (private KV cache via `v7_generate`); a
  decide call between two chat turns leaves the chat session intact
  (regression test: `internal_tools\_test_decide_session.bat`). Error
  paths: 400 with a named reason for bad fields or multi-token labels, 405
  for non-POST.
- **Worker priority control** -- `v7_set_worker_priority` now works at any
  call time: before the worker pool exists the request is deferred in BSS
  and applied automatically at pool creation; afterwards it is applied
  immediately (previously it FATALed on a pre-pool call, which crash-looped
  the server). `v7_server.exe` gains a `-prio N` CLI (default -15, Win32
  idle: the machine stays responsive while the server serves), applied via
  the setter plus a `v7_override_prio` env override at pool creation.
- **Server: missing think-tag rollback** -- when thinking is ON and a turn
  produces no closing think tag (the model thought past the token/buffer
  limit), the server rewinds the KV cache to the pre-turn position
  (`v7_get_position`/`v7_set_position`), re-runs the same message with
  `v7_enable_thinking(0)`, and restores thinking afterwards. Verified with
  a forced limit stop: the retried answer is a clean direct response, and
  multi-turn memory survives the rollback.
- **The SectorLogo standard library owns every external call** (the
  exe/DLL call centralization): `autoexec.logo` gains `DefineRagFunctions`
  (RagRemember/RagSearch/RagAdd via bm25_rag.exe, RagMakeQuery/RagWebSearch
  via web_ragg.exe, and the composite RagDistill write-spawn-return
  pipeline) and `DefinePureLogoBM25` (BM25Init/BM25Search/BM25Set* -- the
  pure-Logo BM25 pipeline, zero child processes). All 43 example and
  research `.logo` scripts were ported onto the wrapper API; no sidecar
  carries a raw `CallLibrary` or its own `DefineCommand` spawn anymore.
- **Nested `Define` registers globally**: a Define executed inside a user
  function lands in the shared base env (the sentinel insert in
  `DEF_BIND_TAIL`), so wrapper installers survive the function boundary.
  `lg_init` builds the sentinel before the autoexec eval, which also fixes
  the stdlib being silently lost on every init. Validator suite 96 covers
  the binding behavior.
- **Typed decisions** (zero-shot classification in one prefill pass): new
  `v7_get_token_probs` export (restricted softmax over candidate token ids,
  read from the logits buffer of the last forward/prefill position) plus
  `v7_decide_example.exe` and `spam_classifier.exe` (classifies an email
  text file ham vs spam). Documented in `docs/typed_decisions.md`; measured
  results and prompt-craft analysis in `examples/decision_Results.md`.
- **`v7_set_worker_priority(int)`** â€” runtime Win32 priority override for the
  whole worker pool (new `src/asm/set_worker_priority.inc`; -15 idle .. 15
  time critical, last call wins). After worker creation the engine applies the
  `v7_override_prio` environment variable automatically, and verify.bat sets
  it to -15 so all verification inference runs at idle priority.
  `hello_inference` was renamed to `inference_example` and gained a `-prio`
  flag for the same setter; `internal_tools/_test_prio_override.bat` smoke
  tests flag, env override, malformed-env FATAL, and baseline.
- **Think-budget limiter** â€” the stateless `ThinkLimit` autoexec function
  closes a runaway think block at the next sentence boundary and suppresses
  generated think-open tags once a turn exceeds its token budget (an
  unclosed think block used to fall into the server's no-think retry and
  discard every RAG injection). Engine support: the `SetToken` SectorLogo
  primitive (replaces the current candidate token and its history entry)
  plus the `v7_get_gen_count` and `v7_get_think_mode_active` exports;
  `examples/think_limiter/` is a standalone test with a README.
- **Chat avatar and background crossfade** â€” a sidecar sets
  `__server__chat_avatar_image` and the server serves it under /avatar on
  every bot message (both session-reset paths clear the key so a new
  session starts with the plain v7 label); /background is served immutable
  and the page crossfades old-to-new over 4 s through two fixed layers
  driven by sessionStorage cache busters; the chat text surfaces are
  darkened to stay readable over bright images.
- **`tools/logo_fmt.py`** â€” the SectorLogo autoformatter: Allman body
  style, verbatim strings and comments, and a token-stream equality gate
  that refuses to write when content would change; idempotent and
  BOM/CR-safe.
- **Zydis crash disassembly in the server** â€” `zydis_debug_header.h` +
  `ZydisInit` at boot; a crash anywhere in the server process prints a
  fault-site disassembly instead of dying opaque.
- **Background tuner proof harness** â€” the previously untested
  `slow_drift_tuner` now runs end to end (`examples/tuner/tuner_proof.c`:
  legal candidates, accept banner, state-mirror persistence, all-shape
  rotation).

### Removed

- **SAM text-to-speech**, a third-party component with unclear licensing:
  `src/asm/sam/`, the winmm import block, the `DllMain` init call, the five
  exports, `examples/voice/`, the `speak` mode of `v7_pipe`, and every SAM
  row in the docs. The LLM sampler's `SAM_*` struct-offset names are
  unrelated and stay.

### Changed

- **`vit_forward.inc` split into a hub plus an encode callback** (626 + 926
  lines, pure line moves). Vision smoke hash-identical to baseline.
- **The server owns the Qwen3 model-card sampler settings.**
  `set_engine_thinking` re-arms the recommended pair per mode (thinking
  temperature 0.6 / top_p 0.95, non-thinking temperature 0.7 / top_p 0.8);
  the greedy default is retired (the Qwen3 card documents that greedy
  decoding degrades Qwen3 into endless repetitions); top_k stays at the
  engine zero because `v7s_sample` implements temperature and top_p only
  and rejects a nonzero top_k; /v1/decide runs its one-token generate
  under a local greedy fence so a typed decision cannot waver between
  labels across identical requests; the "sampler is DLL-owned, not
  applied" wording in the handlers and logs matches the contract.
- **Ctrl+C is no longer hooked in the server** â€” handler, press counter
  and SetConsoleCtrlHandler are gone; one Ctrl+C kills the process and the
  server.bat restart loop brings it back; stop_server.bat remains the
  clean-stop path.
- **The server is split into 14 single-purpose translation units plus 3
  headers in `examples/server/`** â€” a pure relocation verified
  byte-identical against a pre-split transcript. `stop_server.bat` sends a
  plain GET (the new 411 gate rejects bodiless POSTs); the build-freshness
  manifest lists all 14 TUs on one record.
- **Generation is unlimited by default** â€” the illegal 2500-token cap
  became an unlimited default with a `-maxgen` CLI override.
- **Ardmark RAG iterations**: the avatar and seasonal-background logic
  ported back from git history (boot background seed with a NIL-safe gate,
  persona avatar seed, the think-close summer/winter/dragon scan with
  winter-over-summer and dragon-over-both ordering); the EOS-as-close
  rescue runs first so the replaced close is rewritten by the close
  handlers in the same hook call; every injected text (primer, question
  handoff, discipline, rescue cue) rewritten to the model's own thinking
  style so the model adopts them instead of echoing; retrieval queries the
  user's question at top-6; `g_live_gen_count` counts main-loop tokens
  only so the rewrite re-prefill no longer inflates the think limiter;
  the sidecar moved to the guarded think-seeded strategy and closes the
  post-fact re-think with a role reminder so the injected facts do not
  drown the original question; the web chat forces thinking ON while a
  sidecar is active instead of silently running a dead hook.
- **Example-suite verification** â€” all 8 text-model and 3 VLM databases
  rebuilt from their checkpoints; all 16 LLM runs (8 models x F32/Q8) and
  all 6 VLM runs (3 models x F32/Q8) answer correctly; every example suite
  (`run_inference_examples.bat`, `run_tool_examples.bat`,
  `run_logo_scripts.bat`) passes. All example Logo scripts re-checked
  against the dead-name trap classes (no findings).
- **`sectorlogo_lisp.inc` split into five focused files** (core, intern,
  valstr, equal, gc). Validator all green, zero comment-checker violations.
- **Seventeen fully-implemented historical plan documents removed from
  `docs/plans/`** (float type, stream rewrite, sorted lists, string
  concatenation, local variables, long-string support, the vision speedup
  work, and others). Documents still referenced as current material were
  kept after verification; the unimplemented proposals remain.

### Fixed

- **The server audit batch** (17 reported bugs plus a delivered build
  breaker): the think-tag rescue ladder no longer overrides the caller's
  max_tokens with the server-wide budget, saves and restores the sampler
  temperature instead of leaving a successful retry at 0.7-1.3 forever,
  keeps a successful retry instead of overwriting it with the no-think
  fallback, and skips the fallback on a dead engine; the single
  health-thread acceptor no longer stalls on peer-close and reads up to
  the 64 KB header cap with /v1/models/<id> handled live; both GET and
  HEAD /v1/health report engine death; stateless chat keeps content:null
  tool-call turns in the transcript and resets think numbering;
  json_escape validates UTF-8 values (overlong encodings, UTF-16
  surrogates, values above U+10FFFF); the web page appends a visible
  marker pair for an empty answer so page history and KV stay aligned;
  the presented API key is rejected when truncated at the compare buffer;
  vision reports a buffer cut as finish_reason "length"; main derives
  loopback from the parsed 127.x.x.x bytes, tears the handoff critical
  section down only after a confirmed join, and binds no string literal
  to char* under /permissive-; the [dbg] prints are gone.
- **Server hardening** (five static review passes plus three external
  audit reports): the accept race between the main thread and the health
  thread is gone (single acceptor + handoff event, preconnects close at
  the peek, the never-firing CPU-clock deadline fixed) and partial HTTP
  requests are never dispatched; the JSON parser is strict (EOF inside
  objects/arrays fails, top-level close required, UTF-8 BOM skipped,
  `\uXXXX` hex validated, truncated strings trimmed to UTF-8 boundaries)
  and content arrays, `developer`/`tool` roles, `max_completion_tokens`,
  `enable_thinking`, `reasoning_effort`, completions `stream` and array
  `prompt` are supported; SSE emits the full chunk sequence with
  `reasoning_content` plus real `finish_reason`; the missing-think-tag
  rescue is one shared `run_turn` helper (thinking-state tracked and
  restored, fresh-vs-continue rollback corrected, retry at the full
  generation budget so the rescued answer is no longer truncated
  mid-sentence); web chat: 64 KB answers no longer cut (history slots
  sized to the output buffer), the newest turns survive the page budget,
  a literal think-open tag no longer renders, and the stale-token page
  preserves the typed message; decide validates `kind`/option counts,
  rejects duplicate labels by token id, and its prompt builder no longer
  has a stack-smashing `snprintf` overflow; vision decodes
  composite-on-white RGB (BGR swap), applies EXIF orientation, downscales
  oversized images, no longer overflows the 1 MB stack with an 8 MB parse
  buffer, and the 8 MB error-path leak is fixed; UTF-8 boundary trims and
  the page/tail buffer budgets fix a family of truncation and overflow
  bugs, including a trim edge that could read before the buffer; the
  request counter, per-handler max_gen, and dead-engine restart handling
  (exit code other than 42, so the bat loop restarts) are correct.
- **Server security** â€” loopback Host check (DNS rebinding), Origin check
  on inference endpoints, optional `-api-key`, BCryptGenRandom form token
  (CSRF on the web form endpoints and `/v1/chat/reset`), UNC-proof
  `/sidecar` canonicalization with an optional `V7_SIDECAR_DIR`
  allowlist, `X-Frame-Options`/`X-Content-Type-Options`, log
  sanitization, constant-time token compare; `/quit`, `/chat/reset` and
  `/sidecar` require the `X-V7-Local` header and the server binds
  127.0.0.1 by default (`V7_SERVER_BIND` overrides); `/v1/*` responses
  keep CORS, the chat page no longer sends `Access-Control-Allow-Origin: *`.
- **Server ops** â€” case-insensitive headers, 100-continue, 413 on oversize
  bodies, chunked rejection, graceful socket close, bind retries through
  TIME_WAIT, UTF-8 console output, health reports engine death (503),
  strict CLI parsing.
- **SectorLogo must never crash on bad code** â€” `sl_warn_unknown_word`
  guarded its tagged statement/expression head reads with
  ASSERT_ATOM_NONNIL (a CONS/NUMBER/FLOAT/NIL head used to be dereferenced
  tagged, an r13 + 0x800000E8 class crash that the ardmark sidecar's bare
  list head reproduced in production). Regression tests for
  list/number/float/NIL statement heads run in v_sectorlogo_validate.
- **Static report batch** (3 revised opt_and_bug reports): math_data.inc
  lost the dead rpe_three constant and its stale references; v7s_itoa
  converts digits with the 2-operand `imul rax, rcx` + `shr rax, 35` form
  (one dependency hop less, guarded zero-extend at the ABI border); the
  vit_kernel %undef teardown retires the nine vac_ymm slot names like the
  vlc side already did (NASM textual redefinition would have aliased them
  silently in later files).
- **Examples and tools audit** â€” every C example and tool in `examples/`
  reviewed: gated engine calls are paired with `v7_engine_is_dead()`
  checks that print the death reason and exit nonzero (test_chat_only,
  test_generate_only, test_multiturn, test_multiturn_manual,
  multi_turn_demo, v7_pipe, tune_example, poem_continuation,
  vit_bbox_eval, test_image_multiturn, dbg_image_recog,
  c_callback_example); teardown follows the shutdown-before-free ordering
  everywhere; unchecked `v7_init` calls fail loudly (run_sectorlogo,
  bm25_sectorlogo_benchmark, c_callback_example); unbounded `sprintf` of
  CLI arguments into stack buffers became bounded snprintf chains
  (v7_decide_example; decide output slots widened to hold a 4-byte UTF-8
  token plus NUL); web_ragg's extractTitle no longer returns the whole
  page head on a malformed title tag and its stdout capture redirect is
  unconditional; a bulk-replace-mangled function name in
  llm_text_processor is fixed; sidecar-loading examples check their
  `v7_db_add` results; the execute-fmt tester gained its Zydis crash
  handler; validators no longer pre-memset the library struct that
  `v7_load` zeroes itself; the tuner scripts had never executed â€”
  orphaned lines, reversed `Item` argument order, a string-key `FindItem`
  FATAL, the wrong DB key, and candidates outside the dispatcher's legal
  parameter window are fixed.
- **`trace_write_hex` printed garbage**: the hex loop extracted nibbles from
  a stale EAX instead of the parked value register, so every TRACE_HEX line
  in TRACE_DEBUG builds printed one repeated hex digit (0x1111...,
  0x8888...). The loop now refreshes EAX from the parked value each
  iteration; trace builds are diagnostic-accurate again.
- **The memory-RAG answer ignored the injected memory**: an appended recall
  block loses to the model's own pre-committed answer plan, so v6/v7 now
  replace the think text with a fixed recall scaffold on injection. Verified
  end-to-end: "What do you remember about me?" answers "you live in
  Bruges" from stored facts.
- **`rmsnorm_quantize_kernel` retired multi-line macros with `%undef`**,
  which never removes a `%macro` -- the three names leaked into every later
  include. Now `%unmacro` with matching arities.
- **`rope_only_kernel` saved r12/r13 in a leaf kernel**; the six live loop
  values fit the volatile set, so the quartet of saves is gone (4
  instructions per head call, bit-identical output).
- **`sectorlogo_intern`'s linear scan reloaded the loop-invariant granule
  bound on every driver entry**; it now rides r11d (2 instructions per
  candidate visit; the scan makes no calls, so the cursor cannot move
  mid-walk).
- **An odd vision grid side walked the pos-embed row past the learned
  table** â€” the 2x2 merge cascade advanced `2*block_row` with no wrap. An
  odd grid is now refused instead of mis-mapped; every shipped database has
  an even side, so no model's output changes.
- **Worker threads built their FTZ/DAZ image from their own entry MXCSR**
  instead of the captured one, a determinism hazard; they now use the same
  single `ldmxcsr` the wake-restore path uses.
- **The autotuner armed a baseline turn with a stale shape cache**, so a
  dispatch could reuse the previous chat's unvalidated candidate. Both
  entry points now force re-derivation, as the reject/reset paths already
  did.
- **The anyres pinpoint producer multiplied a hardcoded 384** where the
  consumer divides the DB-owned tile side; both now read
  `VCFG_IMAGE_SIZE`.
- **The unsupported-CPU death path executed AVX before the gate could
  report it** â€” the landing-zone snapshot and the fatal unwind used
  `vmovaps`, so an unsupported machine got a bare `#UD` instead of the
  `cpu_gate` message. All twenty sites are legacy SSE2 `movaps`.
- **`handle_fatal`'s `vzeroupper` is gated on the CPU probe** (deleting it
  was not an option: a supported CPU can fault mid-AVX).
- **The `FATAL` macro's two-path parity probe became one round-down**
  (`and rsp, -16`), five instructions and two labels shorter.

### Performance

- **Sampler top-p cutoff skips whole 8-lane blocks** with one ordered
  vector compare instead of eight scalar tests per lane.
- **Inference init folds the SentencePiece detect into the vocab copy
  loop** and drops the second vocab lookup.
- **The QKV bias OR-scan folds 128 bytes per iteration** with `vpor`.
- **`db_save`'s entry scan walks with a running pointer** and the three
  partial-write loops share one macro.
- **The prefill V-bias add loop strides 32 floats per iteration**;
  `v7s_add_inplace` gained a 3-instruction prologue; the fused
  add+rmsnorm remainder loop is one loop instead of prologue plus
  hardcoded tail.

### Fixed (comments)

- `add_kernel.inc`'s call-site citations were all stale and one site
  short; `chat_image.inc`'s marker header cited three drifted line
  numbers, one of them pointing a "keep this an exact test" note at a FATAL
  with no compare.

---

## [2.22.0] - 2026-10-06

A hardware-support and correctness release: the engine now refuses
unsupported CPUs with a named reason instead of crashing, the SwiGLU
kernels finally honour the full 256-bit YMM contract, and SectorLogo's
sort, string scan, and console paths got faster and safer.

### Added

- **CPUID hardware-baseline gate** (`src/asm/cpu_gate.inc`). One probe at
  `DllMain` attach records whether the CPU has AVX2, FMA, BMI1/2, AES-NI,
  SSE4.1/4.2, RDRAND and OS-enabled SSE+YMM state. A machine without one
  of them is refused at the first public API entry with the missing
  feature named, via the engine pseudo-exception machinery (host informed,
  reason queryable, recoverable). AVX-VNNI, AVX-512 and APX stay ungated
  because they are never emitted. The forced-flag case is a permanent
  regression test.

### Fixed

- **Both SwiGLU kernels ran `vzeroupper` after the callee-saved YMM
  restores**, wiping every restored upper half and making the whole
  save/restore pair a no-op for its contract. The Q8 twin also saved only
  the low 128 bits while writing 256-bit registers; both now save and
  restore at full width.
- **`GetFileType`'s DISK constant was 0** (it is 1), so a real redirected
  disk stdin was declined by `ReadKey` while an unclassifiable handle got
  the blocking read.
- **`sl_atom_gc`'s sweep cursor advanced in bytes where granules are
  required** â€” the 8x overshot declared everything above an eighth of the
  arena dead and could rewind the free cursor over live atoms. Both sites
  now convert exactly; a new validator suite interns past the soft limit
  and asserts a live atom survives.
- **`lg_value_to_str` silently dropped dotted tails** (`Cons 1 2` printed
  `[1 []]`); tails now render as `[1 . 5]` and chains flatten.
- **Command lines between 32768 and 65535 characters died silently** in
  `CreateProcessA` (the ANSI variant rejects them); the bound is now the
  API's real limit.
- **`n_layers == 0` from a corrupt database ran one bogus transformer
  layer**; now a FATAL at init, one check fixing all four consumers.
- **Q8 prefill quantized the same normalized activation three times** per
  (token, layer); Phase 1 now does one fused norm+quantize.
- **The Q8 matmul scalar tail multiplied one group by eight groups'
  scales** â€” reachable whenever a chunk width is not a multiple of four.
- **Console input read `dwControlKeyState` instead of `uChar`**, so plain
  character keys were consumed and discarded; `INPUT_RECORD` was also
  over-reserved.
- **`Random` with a negative limit** divided by the zero-extended
  magnitude and advanced the PRNG for a call that should return 0.
- **`IfElse` in value position returned NIL for a non-list branch**; it
  now FATALs naming the branch, as `RUN` already did.
- **`%f` printed "inf" for a finite float at or above 2^63**; the finite
  out-of-range case has its own FATAL.
- **Disabling the autotuner left block 0 dispatching on an unvalidated
  candidate forever**; the disable path now re-derives the shape cache.
- **SMT worker placement was bounded by the literal 2**, so machines with
  wider SMT could not initialize; the loop now ends when a round claims
  nothing.
- **An empty `supported_resolutions` silently degraded every dispatch to
  the `[default]` block**; an empty list is now a FATAL.
- **The anyres newline buffer was capacity-checked against the wrong DB
  field** and could read past the array; the read width is now checked in
  64-bit.
- **Any `.inc` edit forced a full C/C++ rebuild** â€” the freshness manifest
  listed the whole ASM tree for `dbg_inference.exe`; it now lists the real
  five-file closure.
- **`EncodeWord` could write ~2 MB past its token buffer** (`max_tokens`
  derived from input length versus a fixed buffer); it now FATALs rather
  than clamping.
- **`sl_str_to_num` reserved 8 bytes where Win64 mandates a 32-byte home
  area**, with the degradation flag parked in the caller's `rcx` slot; the
  frame is now a named constant layout.
- **`ParseLogoObject` clobbered callee-saved rsi and called `Intern`
  misaligned**; `sl_trace_print_indent` called `PrintRaw` at 8 mod 16;
  `WRITE_HEX64` entered Win64 misaligned.
- **`sl_fpow` returned 1.0 for a NaN exponent** (the unordered flags were
  tested after the "0^0" arm); `0.0^NaN` is now 0.0 per contract.
- **The oversize `V7-DB-GET` FATAL could print a garbage length** (a
  64-bit value narrowed implicitly); `v7_db_add` also passed a 32-bit
  length uncleaned into the 64-bit reader.
- **`v7s_readfile`'s pos_embed memo could hit on a stale destination
  pointer** across destroy/create; the memo key is zeroed at context
  creation.
- **`tune_set_param` enforced only the lower half of the [1,64] batch
  range**, letting a claim size no other writer can produce be armed.

### Changed

- `SL_F32_NOT_SIGN_MASK` / `SL_F32_EXP_MASK` deduplicated to one
  definition; the `DISPATCH_EXECUTE` count-register contract corrected;
  `%f`'s raw-binary32 contract surfaced in the C header; two dead blocks
  (`nheads == 0` guard, `.kvs_io_fail`) deleted as unreachable code.

### Performance

- **`Sort` gained a stable insertion path for lists of up to 64
  elements** (~19x faster at N=16; the radix path keeps 65+).
- **`SL_FIRSTCHAR_SCAN` failed-candidate re-entry costs 3 instructions
  instead of ~10**, with the window loop restructured around the
  no-candidate path.
- **`vit_pos_embed`'s interp loop folds four table loads into the FMA
  memory operands** (21 to 13 instructions per 16 elements).
- **`vit_rope_build`'s inv_freq stores go straight from the x87 stack**;
  the 8-wide AVX2 prefix-sum for the radix histogram was rejected after
  field testing corrupted multi-record scatters.

---

## [2.20.0] - 2026-09-30

The interned-atom arena gained a real garbage collector, closing the
class of sidecar failures where unbounded per-token interning exhausted
64 MiB; alongside it a release-script fix, a benchmark leg, and a broad
ABI and hygiene sweep.

### Added

- **SectorLogo atom-area garbage collection.** The 64 MiB intern arena was
  append-only, so any sidecar interning unbounded per-token text exhausted
  it and aborted. A non-moving mark-sweep collector now reclaims dead
  atoms (it cannot move them because the dispatch tables index by raw atom
  offset), triggered at 48 MiB with an 8 MiB rearm window. Cons-only
  collections execute byte-identical code to before.
- **Collector test surface**: fifteen collector variants plus intern and
  loadtest gates in `verify.bat` Phase 3b, an atom-churn loadtest workload
  that actually reaches the collect path, and a regression test that the
  zero-claim-size guard fires at database creation.
- **GNU Guile leg** in the multi-language interpreter benchmark, measured
  as a JIT-tiered VM (native speed on loop-add, GC-bound on string build).

### Changed

- **The two image wrappers now share one macro** for the ~50-line
  ViT-encode and marker-prompt span they carried verbatim (they had
  already drifted on token statistics and prompt forwarding).
- `chat_image.inc` retires its file-local preprocessor names; a comment
  accuracy pass removed four past-edit stories from `v7_core.inc`,
  `chat_image.inc` and `v7_data.inc`.

### Fixed

- **`sync_release.bat` shipped a `verify.bat` whose Phase 3b gate could
  never run** â€” the two gate scripts lived in a tree the release script
  never copied, so release verification died before any inference smoke.
- **`v7s_generate_image_mem` armed `g_img_turn` for a callee that never
  reads or clears it**, so the next *text* chat turn rendered through the
  image template. The store is deleted and the invariant stated.
- **`v7_core.inc` carried a second, diverged copy of the Config field
  offsets** that had never learned `has_qk_norm` or `rope_base`; it now
  includes the canonical `inf_structs.inc`.
- **The `MEMORY_*` and `STR_LEN` macros trusted the caller's direction
  flag**; each now opens with its own `cld`.
- **Seven naming families described the same three DeepStack slot byte
  offsets** â€” one of them a silent shadowing redefinition â€” and all now
  resolve to one owned pair. A one-byte drift there would have moved a
  free-pointer load off its slot: heap corruption, not a crash.
- **`v7s_sectorlogo_on_token` cleared the per-call flags before the init
  that dirties them** (that init runs `autoexec.logo`), leaking standard
  library state into the first token's evaluation.
- **`EncodeWord` reserved no room for `encode()`'s `max_tokens` bound**
  (reachable via the 1 MiB stream word) and `sl_str_to_num` reserved a
  Win64 home area of 8 bytes where 32 are mandated.
- **`v7_execute_fmt`'s oversized command lines returned an empty string
  with no diagnostic**; `v7s_readfile`'s full-buffer probe reported an I/O
  failure as truncation.
- **The casefold inner loop's masked select is one AVX2 instruction**
  (`VPBLENDVB`) instead of three, with a redundant reload removed.
- **`JOB_LOAD_WEIGHT_S` was removed** â€” it implied a by-value read path
  that does not exist; **`DWORD_SCALE`'s five definitions** were
  deduplicated; **`sectorlogo_float.inc` and `chat_image.inc` retire their
  file-local preprocessor names**; the instrumentation dump no longer
  prints a phantom worker.

### Known issues

- **The atom-GC sweep cursor's byte/granule mix was diagnosed and
  deliberately left unfixed**: correcting the units activates a reclaim
  path that has never actually run and whose gap-merge logic is not yet
  sound. The latent corruption is unchanged by leaving it â€” nothing is
  reclaimed at all. This needs a dedicated stress harness, not a review
  item.

### Performance

- **`v7s_quantize_chunk` drops a callee-saved `ymm15` spill** that existed
  for one scalar-tail instruction, by folding the constant into a memory
  operand (frame shrinks to 32 bytes).
- **Fused F32 matmul advances weight/scale rows by constant strides**
  instead of recomputing addresses per row, moving the only `imul` from
  per-row to per-head.
- **Car/Cdr/PopCons fold the cell-address `lea` into the load's SIB+disp32**
  â€” one uop off the hottest leaf in the engine.
- **`lg_value_to_str` skips the AVX2 strlen leaf for short atoms** (the
  `:PIECE` scale); `v7s_add_inplace`'s never-firable entry guard is
  deleted; `exec_append_raw` folds its cursor math into one `lea`.

---

## [2.19.0] - 2026-09-28

A real-bug release. Vision initialization crashed every VLM on a
divide-error class, SectorLogo's equality and infix operators returned
wrong values, the SAM formant synthesis read past two 81-entry tables,
and a third dispatch writer could arm the zero claim size that wedges the
job cursor.

### Fixed

- **Vision init crashed every VLM** â€” the `n_heads % n_kv_heads` geometry
  gate zeroed the divisor register instead of the dividend's high half,
  so a stale `edx` overflowed the quotient.
- **`patch_size` was validated before the vision config was populated**,
  rejecting every model from a still-zero block.
- **Two RoPE size computations truncated qword config fields through
  32-bit registers**, and the truncation was not contained: a downstream
  64-bit multiply turned a wrong stride into wrong addresses and
  plausible garbage output. Also fixed: a table extent that relied on a
  caller three files away for its zero-extension.
- **`ps_grid_side` was never validated** as the divisor it is, and the
  spatial-merge patchify copied a hardcoded 16 floats per row against a
  data-driven destination stride. Both now refuse or validate at the
  producer.
- **SectorLogo `Equal` returned NIL for every cons-vs-cons comparison**
  (a cons carries tag bit 31 alone and was misrouted into the float path),
  and the **infix `*`, `+`, `-`, `<<` operators returned unchecked values**
  where the prefix spellings correctly failed.
- **Two Win64 ABI violations** (an unsaved non-volatile register in
  `lg_init`, misaligned calls in `intern_builtins`) and a
  **`mdb_free_mem` alignment defect** whose FATAL paths exited with RSP
  misaligned.
- **`ReadKey` returned 1 instead of NIL** for an empty queue; **EncodeWord
  passed a garbage `max_tokens`**; C-CALL type-mismatch diagnostics now
  name the offending value.
- **SAM: two out-of-bounds reads** of the 81-entry flag tables with the
  parser's own end-of-list terminator, **inter-word silence written to the
  position nothing reads**, a **35-bit immediate truncated to 32** in the
  PCM write macro, and a **loop-counter register clobber** by the same
  macro.
- **`v7s_tune_set_param` guarded `num_chunks` but not the eight batch
  sizes** â€” a zero armed through that third writer deadlocks the dispatch
  with no output, no diagnostic and no FATAL.
- **The KV cache file format dropped the M-RoPE position state**, so a
  reloaded cache produced wrong vision output.
- **A zero-length microdb template entry reached `rep movsb` with a
  0xFFFFFFFF length**; `V7-DB-GET` narrowed a qword length to a dword; the
  `cld` protecting the FATAL reason copy ran seven lines too late;
  `v7_execute_fmt` silently clamped `argc` to 8.
- **Dead code removed**: the microdb tombstone machinery (a `cmovnz` that
  could never fire), two dead register moves in `KP_ALLOC_INLINE`, a dead
  load in `EnableColors`, fifteen dead-name references in `NearestMatch`.
  `OP_BIT` gained the bounds check its two sibling macros already had.

### Added

- **`v7_implementation_cost_estimations.md`**: a measured cost analysis of
  the project (six models, a reference calculator in
  `internal_tools/cost_est_calc.py`) explaining both the estimated
  human-team effort and why agent-assisted delivery broke every
  parametric model in it.
- `docs/plans/sectorlogo_atom_gc_plan.md` â€” the design record for the atom arena
  collector (shipped in 2.20.0).

### Performance

- Dead tombstone machinery removed (~2M dead instructions per `db_save` on
  the 152K-entry database); the SAM formant loop no longer uses the
  integer divider; the PCM write loop's macro is expanded inline.

### Documentation

- The `bt`/`bts` memory-operand question is settled against the Intel
  SDM and recorded next to the bitmap constants; prefetch-removal
  proposals are now rejected on sight; capacity pre-checks against
  append-only arenas are now rejected as reports.

---

## [2.16.0] - 2026-09-17

The SectorLogo speed-up release, plus the fix for a vision regression that
rendered upscaled LLaVA images completely white.

### Performance

- **Atom interning was the interpreter's bottleneck and is now a
  dual-hash fast path**: a dual-hash bloom filter gives repeat interning a
  certain-miss shortcut, a 4096-slot direct-mapped cache accelerates hits,
  and the linear scan remains the source of truth. Measured: replace-all
  ~2,400 ms to 5.5 ms; string build 1,500-2,000 ms to 322 ms.
- **Shared AVX2 first-char scans for Search/Replace/Split** cut the
  per-byte scan cost from ~6 ops to ~9 ops per 16 bytes; `SUBSTR` gained a
  bounded window scan, turning the RAG sidecar's stream-walk from
  quadratic to linear in stream length.
- **A native `NGrams` primitive** (dual-mode: byte substrings from strings,
  element sublists from lists) replaces per-element Logo bigram loops â€”
  the BM25 bigram stage is now one call.
- **LOCAL variables became a frame array**: `MAKE` and `:VAR` on a LOCAL
  name resolve through one dense descending scan plus an indexed
  load/store; the env-walk is deleted for LOCAL names (dynamic scoping is
  untouched for everything else). Measured -15-30% on LOCAL-heavy loops.
- **`STOP` gained standard-Logo procedure-return semantics** (returns from
  the enclosing function only, not the caller's loop or the script);
  **`REPEAT`'s count became optional** (endless loop with STOP/CONTINUE/
  OUTPUT exits); `autoexec.logo` gained `WHILE`.
- **New `STRLEN` primitive** built on a page-gated SWAR scan.
- Six `opt_and_bug` batches applied, including two real bugs: a >4GB KV
  cache blob truncated through DWORD `WriteFile`/`ReadFile` parameters,
  and a short-string fast path that over-read the microdb pool end.

### Fixed

- **The LLaVA-OneVision bicubic regression (root cause)**: the Catmull-Rom
  kernel macros used `vfmadd132ss` where `vfmadd213ss` was required,
  evaluating a quadratic instead of a cubic â€” boundary weights reached +9.0
  and saturated bright pixels to white. Y-tap row offsets were also off by
  one. Box-test coordinates are back to the healthy reference on F32+Q8.
- **New `v_bicubic_dump` validator** compares the kernel against a
  double-precision C reference across five shape families, so the
  saturation signature of any future kernel-weight bug cannot hide behind
  the nearest-neighbor vision smoke.
- **A BM25 RAG crash**: `Local "a" "b" "c"` declared only the first name,
  and a caller's LOCAL could completely shadow a callee's parameter.
  Params are now registered as local-set entries; multi-name declarations
  work via a keyword-aware raw peek.
- SectorLogo's **hex-fallback token decode** destroyed the high nibble
  (`<0x41>` decoded as `0x01`); **16+ character atoms hashed over a
  truncated prefix**, corrupting hash-map keys; **`intern_builtins`' bloom
  bit-set** dropped the second hash bit; a committed-but-missing source
  file (`sectorlogo_split_ngrams.inc`) that broke fresh-clone builds was
  committed.

### Added

- **BM25 benchmark suite**: an externalized template running the native
  and historical tokenizers through one shared body (corpus tokenize 4.3x,
  end-to-end 2.9x, answers identical) plus a statement-level slot
  profiler. The host-FFI slot-timer pattern profiles any SectorLogo script
  per-phase from C.
- **Cross-language benchmark suite** (SectorLogo vs CPython, Node, Tcl,
  PowerShell) with a new `docs/sectorlogo_performance.md` recording the
  results, the intern speedup analysis, and the ranked mitigations.
- `test_image_multiturn` example (three-turn interleaved vision smoke
  across the public API) and `t_intern_reps` probe tool.

### Experiments

- **KV-cache compression measured at ratio 1.002** â€” RoPE-rotated F32
  mantissas are incompressible, so real KV size reduction needs KV
  quantization, not lossless compression.

### Changed

- **The fused F32 matmul now scales the weight operand** (as the C
  reference and the non-fused twin always did) instead of pre-scaling the
  activation; bits move toward the reference.
- **`chat.inc` caches `ctx->quantize` and `ctx->transformer` in frame
  slots** (-2 instructions per generated token).
- **`v7_encode_validate` re-enabled** with the struct field the ASM reads
  past the old struct end; the RC/`r11`-staging ABI errors in the fused
  add-rmsnorm call sites fixed.

---

## [2.14.0] - 2026-09-11

A kernel and dispatch optimization pass, plus the fix for a critical
regression in the engine's own entry gates.

### Fixed

- **A critical regression in the landing-zone gate macros**: the
  "cold-blocks-first" restructure placed the dead path at the macro's
  first instruction, so every gated public entry returned 0 immediately
  without executing its body. All three gate macros now open with one jump
  over the cold blocks.

### Performance

- **SwiGLU dual-row fall-through straightening** moves the common dual
  path to zero taken branches; the per-group scales are broadcast-fused
  (13 to 9 instructions per dual group, bit-exact).
- **Q8 kernel's four-row prologue builds its weight-row chain directly in
  registers**, eliminating four frame slots and their round-trip.
- **F32 kernel advances scale pointers incrementally**, removing the only
  `imul` on the row path.
- **The Q8 forward's per-layer QKV bias add is gated on a one-time
  OR-scan of the exported bias block** â€” bias-less models skip ~72K
  instructions and 64 KB of dead traffic per token. The DeepStack gate
  folds into a per-call flag and the per-layer `vzeroupper` moves to one
  epilogue instruction per token.
- **Public API context guards load the singleton once**; `encode` caches
  the SentencePiece flag; `sectorlogo_main` fixed an off-by-one that
  rejected an exact-64 KB script; `sectorlogo_print` drops PutChar's
  register frame and uses a 256-entry nibble LUT; `sectorlogo_ffi` loads
  argument registers unconditionally; the token dump merges its UTF-8 path.
- **`tune_is_candidate_turn` moved to `.data`** â€” `.bss` zeroing inverted
  the tuner's first A/A and A/B calibration pairs.
- **The inf-state init pins the MemLayout pointer in a register** instead
  of reloading it around each allocation.

### Changed

- AGENTS.md tightened: a consumed report `.md` must be deleted directly
  and immediately, one `del` per file; bulk deletion forbidden.

---

## [2.13.1] - 2026-09-10

Documentation-only release: a full API-reference accuracy pass.

### Changed

- Every internal `v7s_*` reference across the README, API reference,
  setup, troubleshooting, tools, examples, vision, server, RAG,
  SectorLogo, database-key, data-structure and performance docs now names
  the public `v7_*` tier entry.
- The export count and public-tier tables match the current
  `v7_threading.def` (131 exports).
- Removed a non-existent fetch script from the setup docs and corrected
  the `build_db.exe -thinking` note.
- Historical plans referencing the old internal API carry a pointer to the
  current public lifecycle; known validator precision limits are
  documented under the correct validator.

---

## [2.13.0] - 2026-09-10

A performance and robustness pass across the vision pipeline, kernels and
SectorLogo, with documentation brought current with the public API.

### Changed

- **Vision preprocessing is substantially faster per image**: the RGB
  normalize loop processes 8 pixels per iteration, both patchify families
  use additive cursors instead of per-row offset recomputation, and the
  SigLIP row copy is two aligned vector stores.
- **The ViT 2D-RoPE table builder is data-driven end to end** (cache
  capacity, column offsets, exponent divisor all derive from the model's
  head dimension); values for current models are bit-identical, models
  with head_dim above 64 previously built wrong tables.
- **DeepStack buffers are no longer allocated for models without
  DeepStack taps** â€” three fewer allocations and ~440 KB per image encode.
- **Dispatch-tail and init-path streamlining** across the matmul, weight
  load and dequantization code: one shared macro for the four fused-matmul
  dispatch tails, fused vector loads in the embedding dequant, persistent
  write cursors in the weight loader.
- **Documentation**: the public API lifecycle, the engine
  pseudo-exception contract and KV-cache persistence are documented
  consistently; all broken cross-references from the consolidation were
  repaired.

### Fixed

- **SectorLogo number and hash parsing**: very long atoms (16+
  characters) hashed over a truncated prefix, corrupting hash-map keys in
  RAG/BM25 scripts, and the overflow-fraction parser mis-scaled digits.
- Register-lifetime violations around Win32 API calls in the ViT
  projector and one init path; a diagnostic-dump helper clobbering a live
  register.
- Vision output remains byte-identical to the 2.12.0 baseline.

---

## [2.12.0] - 2026-09-10

Public C API without a context handle, engine pseudo-exceptions, and
KV-cache persistence.

### Added

- **Context-free public API.** The `v7_*` tier is the single C-facing API
  (`v7_init`, `v7_initialize_inference`, `v7_chat`, `v7_generate`,
  `v7_chat_image`, `v7_free_inference`). The inference context lives in a
  BSS singleton inside the DLL; hosts never create, store or pass a
  context pointer. All tools and examples were migrated.
- **Engine pseudo-exceptions.** An internal FATAL inside a public call no
  longer kills the host: the entry installs a landing zone, the handler
  prints the reason unchanged, parks the engine, quiesces the worker pool
  and unwinds to the C caller with the callee-saved registers restored.
  Hosts query `v7_engine_is_dead()` / `v7_get_death_reason()`, clean up
  and keep running.
- **KV-cache persistence.** `save_kv_cache` / `load_kv_cache` persist only
  the used rows per layer plus the position, LZ4-compressed, so the file
  scales with the conversation rather than the configured sequence
  length. Loading validates the header against the current model and
  rejects foreign files visibly.
- `poem_continuation` example (raw-continuation mode) and a generate-path
  end-to-end validator.

### Changed

- `dbg_inference.exe` (the pure-assembly harness) rewritten as a reference
  consumer of the public tier, with proper Win32 command-line parsing and
  bounds-checked copies.
- The verification pipeline runs both harnesses (C and pure-ASM) in both
  precisions, plus the engine-death catch test.

---

## [2.11.0] - 2026-09-09

File-backed KV caches and a broad correctness/performance pass.

### Changed

- **KV caches are file-backed sparse mappings**: allocation consumes zero
  commit charge and pages materialize lazily, eliminating the ~10 GB
  commit-charge OOM class under memory pressure without any format change.
- **Performance pass across kernels and interpreter**: incremental probe
  pointers in the hash table, bottom-tested loops in RMSNorm+RoPE and
  prefetch init, a leaner attention prologue, faster digit loops in
  formatting, hoisted invariants in the vision encoder.

### Fixed

- Win64 ABI violations in the ViT position-embed kernel, the sampler
  scalar tail and the SwiGLU kernel (callee-saved XMM registers written
  unsaved; dirty AVX state on return).
- A mis-applied "optimization" in the hash-table key comparison that
  compared past the end of every key of length 8 or more (never matching);
  reverted and documented.
- An RMSNorm+RoPE entry guard that under-tested the loop granularity,
  silently skipping the last 8 elements for head dimensions not divisible
  by 16.
- SectorLogo: REAL-RANDOM discarded its Weyl progression; oversized
  `V7-SET-STREAM` replacements truncated silently instead of failing.

---

## [2.10.0] - 2026-09-08

In-system dispatch tuner with A/A-calibrated decisions, and the
config-error-handling refactor.

### Changed

- **The tuner's decision no longer reads raw cycle counts**: A/A calibration
  pairs establish a warm/cold scale, and accept/reject compares deltas on
  that scale with a configurable margin (`v7s_tune_set_aa` /
  `v7s_tune_set_margin`).
- **Config degeneration**: system-dependent values that cannot be honored
  (taper capacity below the minimum chunk width, chunk counts above the
  legal maximum) degenerate to the nearest legal setting and record the
  event in a queryable `config_was_fixed` bitmask; range violations remain
  fatal. The SMT worker-universe computation now uses the logical
  processor count, making both pinning modes legal.
- Validator matrix-shape tables restricted to shapes that occur in the
  supported models.

---

## [2.9.0] - 2026-09-06

Attention kernel optimization batch: cold-path FMA fusion, hot-path
fallthrough reordering, and a series of control-flow and register-allocation
improvements across the attention and matmul kernels.

## [2.8.0] - 2026-09-05

SectorLogo ANSI console interface: pure VT-sequence output with six new
primitives (`Character`, `Number string`, `ReadKey`, `EnableColors`,
`PrintRaw`, `NearestMatch`), and an `autoexec.logo` standard library that
loads database-first at interpreter init.

## [2.7.0] - 2026-09-02

Build system and codebase consolidation: a single `build.bat` replaces
all per-target build scripts; the C/C++ codebase brought under a uniform
45-rule style standard; the Q8 fused quantize-once optimization reduces
prefill latency by ~43%; assembler micro-optimizations and bug fixes
throughout.

## [2.6.0] - 2026-08-30

Formatting, external tools, and the internet-RAG application: bounded
`v7sprintf` with float support, `DefineCommand` lets SectorLogo scripts
call external executables, the `web_ragg` internet-RAG application (live
web search plus paragraph distillation), runtime sidecar loading in the
OpenAI-compatible server, a multi-turn think-tag leakage fix, and
AVX-to-SSE transition penalties eliminated across all kernels.

## [2.5.0] - 2026-08-28

SectorLogo sorted-list toolbox and randomness: `Sort` (stable radix),
`Reverse`, `Length`, `SetItem`, `FindItem` and the `FindFirstAtLeast` /
`FindFirstAtMost` binary-search family, and `HashNumber` keys make ranking
algorithms expressible in pure Logo; `Local` function-scope variables;
AES-NI PRNG with an optional RDTSC entropy toggle; retrieval-tool recall
options.

## [2.4.0] - 2026-08-27

Native float type for SectorLogo: 30-bit truncated IEEE binary32 packed in
the tagged-value word, nine float math primitives with x87 transcendentals,
numeric coercion in equality, 64-bit offset corrections in the mrope/Q8
kernels, and ViT loop-invariant hoists.

## [2.3.0] - 2026-08-27

RAG production stack: the think-continue substrate with tuned retrieval,
the persistent memory system (conversations distilled into a growing text
corpus queried as recalled memories), a research corpus of 30 sidecar
strategies with measured decisions, and the four-document RAG reference.

## [2.2.0] - 2026-08-26

RAG pipeline foundations: `v7_execute` (spawn plus stdout capture), BM25
retrieval over local text corpora, host FFI callbacks from scripts, and
file I/O primitives; three end-to-end RAG experiments.

## [2.1.0] - 2026-08-25

SectorLogo sidecar scripting: a Logo-dialect interpreter in pure assembly
hooks every generated token â€” scripts inspect and rewrite the output
stream (KV rollback plus re-encode plus re-prefill), reject samples and
adjust sampler parameters; configuration-driven ViT; microDB and kernel
micro-optimizations.

## [2.0.0] - 2026-08-19

First public release: the `v7_pipe` CLI text processor, an export-surface
fix for `v7d_vit_linear`, full documentation, and the complete
verification pipeline.

## [1.9.1] - 2026-08-19

Documentation overhaul: README, development docs, and .gitignore cleanup.

## [1.9.0] - 2026-08-19

LLaVA anyres crash fix, three Win64 ABI bug fixes, ViT performance
improvements, and removal of the legacy runq harness from the build.

## [1.8.0] - 2026-08-19

ViT dispatch tuning, AVX2 kernel optimizations, multi-turn chat templates,
and the full LLM+VLM output-quality documentation.

## [1.7.0] - 2026-08-18

M-RoPE (multimodal rotary position embeddings) for Qwen3-VL, PIL-matching
bicubic resize for LLaVA-OneVision, and all three VLMs producing output on
par with the HuggingFace references.

## [1.6.0] - 2026-08-15

SmolVLM-256M as the third supported VLM, and multi-model output-quality
fixes across all 8 LLMs and 3 VLMs.

## [1.5.1] - 2026-08-12

Documentation-only release.

## [1.5.0] - 2026-08-12

QKV bias support in the forward path (Qwen2 family), the ViT decomposed
into per-operation modules with a C reference implementation, the
bounding-box evaluation harness, and a simpler two-mode memcpy threshold.

## [1.4.1] - 2026-08-09

Fused RMSNorm+quantize in the Q8 forward paths, the
`v7s_rmsnorm_quantize` and `v7s_generate_image_mem` exports, a vision
prompt-buffer crash fix, and the precomputed vision-config (VCFG)
block.

## [1.4.0] - 2026-08-06

Vision: Qwen3-VL-4B image Q&A in pure assembly. ViT encoder, DeepStack
feature extraction, spatial-merge projector, image chat exports, ViT
dispatch modes, and the vision database builder.

## [1.3.0] - 2026-08-04

Q8 quantize+matmul fusion, RMSNorm and quantize precision fixes, the
FATAL contract (no silent fallbacks), and per-shape dispatch parameter
consolidation.

## [1.2.4] - 2026-08-01

MiniCPM5-1B support, export-surface cleanup, and a full safe-optimization
sweep.

## [1.2.2] - 2026-07-29

The OpenAI-compatible API server restored and adapted to the current
architecture.

## [1.2.1] - 2026-07-29

Documentation sweep: 14 files reviewed, paths and defaults verified
against source.

## [1.2.0] - 2026-07-29

Demo agent integration and defensive-code removal (fallback paths
replaced by the FATAL contract).

## [1.1.0] - 2026-07-28

Six-model support, memory-efficient model export, and the removal of
magic numbers and version fields from the database format.

## [1.1.1] - 2026-07-29

SAM (text-to-speech) word-separation fix and export cleanup.

## [1.0.0] - 2026-07-25

First stable release: pure-assembly LLM inference (Qwen3-0.6B), the
work-stealing dispatcher, MicroDB single-file model storage, BPE
tokenization, SAM text-to-speech, and the validator suite. Release commit
`d48895b`; see also 1.0.1 (2026-07-27).

### Fixed during the release candidates and hardening

- **A crash in SAM's `say()`** one byte past the 256 KiB PCM buffer: two
  5-byte PCM write sites bounded the buffer with a 32-bit `lea`, which
  truncates the 64-bit address it computes. Both are now bounded by
  position through a truncation-proof constant. Same class as the
  3-operand `imul` accumulator clobber: an instruction into a 32-bit
  register silently truncating a 64-bit value.
- **SectorLogo integer-range bugs**: four prefix operators manufactured
  values outside the 30-bit payload and handed them to `MakeNum`, which
  masks rather than detects (`AsInteger 1e9` returned -463129088). All
  four now range-check the exact result.
- **A chained comparison** (`1 < 2 < 3`) died with a generic operand
  message; the guard now names the actual mistake.
- **`v7s_stream_append_token` had no capacity check at all** â€” a long
  generation overflowed the stream buffer, the byte-to-token map and the
  concat buffer. It now FATALs, never truncates.
- **The console's "other" stdin handle kind issued an unprobed
  `ReadFile`** that could block the per-token hook; unclassifiable
  handles now map to NIL.
- **Two byte-copy loops into the concat buffer checked bounds after the
  store** with zero margin; both now check before the store.
- **A WAVEHDR sized 48 bytes where `sizeof(WAVEHDR)` is 56** (a ninth
  member was dropped from the comment arithmetic), naming `lpNext` as
  `reserved`. The header also documented a `DllMain` call that does not
  exist and would deadlock the loader lock.
- **Three false invariants in SectorLogo comments** replaced with the true,
  narrower ones, and a rejected "optimization" restored: `INFIX_BITWISE_OP`'s
  opening `mov eax, r9d` is load-bearing (removal fails four tests).

---

## Release candidates and pre-1.0

- **1.0.0-rc1 to rc6** (2026-06-22 to 2026-07-11): incremental hardening
  toward 1.0 â€” kernel correctness, dispatch stability, database format,
  tooling.
- **0.3.0** (2026-06-18 to 2026-06-19): multi-threaded dispatch, first
  complete forward pass.
- **0.2.0** (2026-06-15 to 2026-06-16): first working single-threaded
  inference end to end.
