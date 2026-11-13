# v7 — Pure-assembly LLM/VLM inference engine for x64 Windows + Linux

**Version:** 2.36.8
**License:** GPL v3

## What's new in 2.36.8

Repository and documentation hygiene. The root now contains only the
essentials: every driver script lives in `scripts\`, the AnyRes vision
fixtures live in `vision\` next to their baselines, and the internal dev
records (bug log, experiment ledger, dispatch problem record, release
how-to) live in `docs\` — with the internal-only records excluded from the
shipped tree. Patch and experiment working directories left the repository
(archived locally), `attic\` is fully untracked, and `v7_speed_results\` is
gone. The README itself received a fact-check pass: the RMSNorm pipeline
step, the MiniCPM5-1B benchmark rows, the tuner description and the export
table now match the engine and the measured records, and the edge-AI use
cases are scoped to what v7 actually does. Older release notes
live in [CHANGELOG.md](CHANGELOG.md) — the complete history for every
version since 2.8.0.

## What is v7?

v7 is a complete LLM and VLM inference engine written entirely in NASM x64 assembly with AVX2.
The core artifact is `v7_threading.dll` — a self-contained DLL with zero C runtime in the
hot path. Every kernel — Q8xQ8 and F32xQ8 matmul, multi-head attention, RMSNorm, RoPE,
SwiGLU, softmax sampling, BPE tokenization, ViT image encoder, and the full generation loop —
runs in pure assembly on a work-stealing thread pool.

### Why assembly?

Modern LLM inference is dominated by two bottlenecks: memory bandwidth (for int8 matmul)
and kernel dispatch overhead (the cost of telling a thread pool to start work). Writing in
assembly lets us eliminate every unnecessary instruction from the hot path — no function
call overhead, no register spills, no ABI conversions. The work-stealing dispatcher uses
`lock xadd` for job claiming and `pause`-spin synchronization, with zero API calls on the
hot path when workers are spinning. This is what produces v7's unique performance
characteristics: the kernels run at memory bandwidth as the limiting resource, so only a
small fraction of every token is clock-dependent — in a Windows power-saving plan v7 keeps
about 95% of its full-speed token rate (llama.cpp ~79%, OpenVINO ~73%), its measured lead
over the stock engines grows on battery power, and every instruction shaved from the hot
path is also a joule saved. Low-power and thermally constrained systems are exactly where
this style of engineering pays off most.

### The hot path — what actually happens per token

When you call `v7_chat`, the engine runs a per-token loop that touches every kernel:

1. **RMSNorm** — compute `1/sqrt(Σx²/n + eps)` in one pass, scale every activation element
2. **RoPE** — rotate Q/K vectors by precomputed cos/sin tables (no `sinf`/`cosf` at runtime)
3. **QKV matmul** — fused matmul + RMSNorm + RoPE in a single dispatch (the `v7d_fused_f32` kernel)
4. **Attention** — per-head softmax over the KV cache, GQA with `n_kv_heads` < `n_heads`
5. **SwiGLU** — `silu(gate_w @ x) * (up_w @ x)` with int8 weight quantization
6. **Classifier** — final matmul to produce logits over the vocabulary
7. **Sampler** — AVX2 polynomial softmax + AES-NI RNG + top-p multinomial CDF walk

Each dispatched kernel flows through `DISPATCH_EXECUTE`: the caller thread fills the job
array, atomically publishes `gen_counter++`, then participates in work-stealing alongside
worker threads. Workers claim jobs via `lock xadd` on a shared `next_job` counter. When
workers are spinning (`set_spin(1)`), there are **zero API calls** — no
`WakeByAddressAll`, no `WaitOnAddress`, no `Sleep`, no kernel transitions. Pure `pause`
spin. A dispatch costs a few microseconds end to end (job fill, atomic publish, claim);
the benchmark tools in `tools\` (`b_benchmark.exe`, the per-kernel loadtests) measure
the per-kernel throughput that dominates end-to-end latency — see
`docs\performance_report.md` for the hardware and methodology.

### Hardware baseline

The kernels execute unconditionally; there is no CPU dispatch and no reduced path. A CPU that
lacks the baseline below cannot run v7 at all -- the library measures the machine with CPUID at
load time and every public API entry refuses to run, naming the missing feature. **CPUID
features, not processor age, determine support**: a CPU is eligible exactly when it reports
every row below.

| Feature | Used by |
|---|---|
| AVX2 | all 256-bit kernels (`vpermq`, AVX2 integer broadcast) |
| FMA | every kernel (`vfmadd` / `vfnmadd`) |
| F16C | float16 conversion in the vision preproc path |
| AES-NI | token/string Intern hash and the sampler RNG (`aesenc`, the legacy SSE encoding — one AES round, identical bits, runs without VAES) |
| BMI1 / BMI2 | worker placement (`blsi`, `blsr`), hot-loop search (`tzcnt`), tuner (`bzhi`) |
| SSE4.1 / SSE4.2 | microdb key hash (`vptest`, `crc32`) |
| RDRAND | C dispatch autotuner's hardware RNG draws |
| RDTSCP | serialized tuner timing edge (leaf 0x80000001 EDX[27]; masked by some hypervisors) |

The OS must also enable SSE + YMM state (`XCR0` bits 1 and 2); a hypervisor that reports AVX2
but does not pass `XCR0` through is rejected the same way, and a hypervisor that masks
`RDTSCP` is rejected by name as well. The engine emits the legacy `aesenc` encoding (AES-NI,
2010+), not the VEX `vaesenc` (VAES, 2020+), so AMD Zen 1/Zen 2 and Intel Haswell through
Comet Lake are eligible; VAES-capable CPUs run the same binary unchanged. Everything is
developed and measured on one machine:
Ryzen 5 7430U (Zen 3, 12 logical cores).

### Measured performance

All numbers below are token/s (tok/s) wall-clock generation throughput on the
development machine, greedy, temperature 0, Q8 weights, unless noted. The
full methodology, per-round tables and engine versions live in
`docs\engine_benchmark.md`.

Full-model generation, all four text models vs the strongest third-party
engines (definitive overnight run plus re-tune rounds, 5 rounds per leg):

| Model | v7 | llama.cpp | MNN | OpenVINO |
|---|---|---|---|---|
| Qwen3-0.6B short | **55.8** | 52.0 | 48.2 | 41.4 |
| Qwen3-0.6B poem (500 tok) | **52.2** | 47.3 | 43.8 | 39.7 |
| Qwen3-1.7B short | **20.0** | 19.8 | 18.1 | 18.3 |
| Qwen3-1.7B poem | **19.5** | 19.2 | 17.7 | 17.5 |
| MiniCPM5-1B short | **38.1** | 37.7 | 34.8 | 32.0 |
| MiniCPM5-1B poem | **37.1** | 36.6 | 33.4 | 29.1 |
| Phi-3-mini short | 9.1 | **9.4** | 8.4 | 8.6 |
| Phi-3-mini poem | 8.7 | **8.9** | 8.0 | 7.8 |

v7 leads on Qwen3 and MiniCPM5 in both workloads; llama.cpp leads on
Phi-3-mini (its prefill-heavy profile, 12-14 tok/s v7 prefill, dominates the
short turn). Standalone kernel throughput at 1024x1024: 382-426 GB/s Q8
matmul, 176-197 GB/s F32 matmul, all 12 logical cores active. The Q8
1024x3072 FFN shape reaches 409-422 GB/s (R-corrected) with the tuned
per-shape dispatch configuration the in-process cycle tuner converges to.

The 2.35.0 optimization pass is measured end-to-end on the vision path: a
Qwen3-VL 640x480 image turn (ViT encode + 200-token answer) dropped from
170 s (F32) / 187 s (Q8) to 136 s in both modes, byte-identical output.

### SMC prefetch patching

The DLL uses self-modifying code (SMC) to patch prefetch instruction distances and types
at runtime, per matrix shape. When a new `N×D` shape is dispatched, the cold path scans
41 prefetch slots and writes new displacement bytes directly into the instruction stream
(`set_prefetch_distance`) or swaps the opcode for a different prefetch variant
(`set_prefetch_type`: NTA, T0, T1, T2, PREFETCHW, PREFETCHWT1, CLDEMOTE). The
PREFETCHWT1 and CLDEMOTE variants act as plain prefetch (or no-ops) on CPUs whose
decode lacks them — they are tuning options, never correctness requirements. A `cpuid`
serializes after patching. This allows per-shape prefetch tuning without recompiling.

### What can it do?

- **Run LLM chat models** — supports 10 LLMs (Qwen3 0.6B/1.7B/8B, Llama-3.2, Qwen2.5, Phi-3, SmolLM2, TinyLlama, MiniCPM5)
- **Run VLM image chat** — supports 3 VLMs (Qwen3-VL-4B, LLaVA-OneVision, SmolVLM):
  image + text prompt -> caption, with pure-ASM ViT vision towers (Qwen3-VL with DeepStack,
  SigLIP with MLP/pixel-shuffle projectors)
- **Internet RAG** — live web search + paragraph distillation via `examples/rag/web_ragg/`
- **v7_server** — OpenAI-compatible HTTP API server with sidecar loading
- **In-system cycle tuner** — measures candidate dispatch configurations against the
  active one in A/B/A episodes during live inference and promotes the faster one;
  per-shape populations persist across sessions (see
  [docs/tuning.md](docs/tuning.md) and [docs/sectorlogo.md](docs/sectorlogo.md))
- **Q8 quantized inference** — int8 weights with per-group fp32 scales (group_size=64)
- **F32 inference** — full-precision matmul for reference/quality comparison
- **Multi-turn chat** — KV-cache preserved across turns, position-aware
- **KV-cache persistence** — `save_kv_cache`/`load_kv_cache` persist only the
  used cache prefix + position (LZ4-compressed, scales with the conversation) —
  resume a session in a later process; blobs above 4 GB are written and read in
  2 GB chunks (WriteFile/ReadFile length parameters are DWORDs)
- **Pseudo-exception safety** — internal FATALs inside public calls unwind to
  the C caller instead of terminating the host process (`v7_engine_is_dead` /
  `v7_get_death_reason`)
- **Single-file deployment** — all model data in one LZ4-compressed `lz4ht.db` file

### What is it good for?

- **Embedded/offline inference** — no Python runtime, no GPU, no Docker
- **Educational** — readable assembly for every kernel, suitable for study
- **Performance research** — experiment with dispatch, prefetch, and quantization
- **Rapid deployment** — one DLL + one database file = full inference system

### 11 Edge AI Use Cases

v7 is designed for scenarios where running a full Python ML stack is impractical or
impossible. Here are 11 real-world edge AI applications where v7 fits well:

1. **Local Privacy-First Chat Assistants** — Run a chatbot entirely on-device with no
   data leaving the machine. Healthcare, legal, and financial applications where sending
   prompts to a cloud API violates data sovereignty requirements.

2. **Industrial Control Systems** — Embed LLM-based natural language interfaces into
   SCADA or PLC-adjacent Windows PCs on factory floors. Operators can query machine
   status in plain English with no internet dependency; acting on an answer remains
   the host application's job — v7 generates text and never drives machinery.

3. **Air-Gapped Environments** — Military, classified, or secure facilities where no
   network connection is available. The entire inference system is one DLL + one database
   file — no package manager, no pip install, no runtime downloads.

4. **Point-of-Sale and Kiosk Systems** — Add product search, FAQ answering, or
   customer service to retail terminals running Windows with zero additional
   dependencies.

5. **Offline Translation Devices** — Deploy translation capabilities on Windows tablets
   or laptops in areas without internet. Swap model databases per language pair — the
   engine itself stays unchanged.

6. **Educational Robotics Kits** — Give students a natural language interface to program
   and query robots. The pure-assembly implementation is transparent and inspectable —
   students can study every kernel with a disassembler and understand exactly what happens.

7. **In-Vehicle Infotainment** — Run text-based assistants in cars, trucks, or heavy
   machinery via Windows-based head units. Local single-token generation with no
   cellular dependency for the core inference; speech recognition and synthesis
   stay host-side.

8. **Research Data Labeling and Summarization** — Deploy on lab workstations to
   automatically summarize documents, label text datasets, or extract entities from
   research papers — without uploading sensitive data to commercial APIs.

9. **Assistive Technology** — Screen readers, communication aids, and accessibility tools
   that need on-device text generation. Q8 quantized inference provides a complete
   text-in/text-out pipeline in one DLL.

10. **Legacy Windows System Integration** — Add AI capabilities to existing Windows
    applications (industrial HMI, medical imaging viewers, legacy enterprise software) by
    loading a single DLL via `LoadLibraryA` — no runtime installation, no container, no
    framework dependency.

11. **Game Development (NPC AI)** — Local, multimodal NPC dialogue on Windows-based
    engines (Unreal/Unity) without eating into the GPU VRAM budget required for
    rendering. The CPU-only inference path leaves the GPU entirely free for graphics.

---

## Supported Models

**10 LLMs + 3 VLMs tested** (all produce coherent output on both F32 and Q8 paths):

### LLMs

| Model | Params | Tokenizer | Quality | Fetch Script |
|-------|--------|-----------|---------|--------------|
| Qwen3-0.6B | 0.6B | BPE | Best | `fetch_qwen3_0.6b.bat` |
| Qwen3-1.7B | 1.7B | BPE | Excellent | `fetch_qwen3_1.7b.bat` |
| Qwen3-8B | 8.0B | BPE | Excellent | `fetch_qwen3_8b.bat` |
| Qwen2.5-0.5B | 0.5B | BPE | Good | `fetch_qwen2.5_0.5b.bat` |
| Llama-3.2-1B | 1.0B | BPE | Good | `fetch_llama_3.2_1b.bat` |
| Phi-3-mini-4k | 3.8B | BPE | Excellent | `fetch_phi3_mini.bat` |
| SmolLM2-135M | 0.135B | BPE | Limited | `fetch_smollm2_135m.bat` |
| SmolLM2-360M | 0.36B | BPE | Limited | `fetch_smollm2_360m.bat` |
| TinyLlama-1.1B | 1.1B | SentencePiece | Good | `fetch_tinyllama.bat` |
| MiniCPM5-1B | 1.0B | BPE | Good | `fetch_minicpm5_1b.bat` |

### VLMs

| Model | Params | ViT | Quality | Fetch Script |
|-------|--------|-----|---------|--------------|
| Qwen3-VL-4B | 4.0B | Qwen3-VL (DeepStack) | Vision (text+image) | (manual setup) |
| LLaVA-OneVision | 0.5B | SigLIP (MLP projector) | Vision (text+image) | `fetch_llava_onevision_qwen2_0.5b.bat` |
| SmolVLM-256M | 0.256B | SigLIP (pixel-shuffle) | Vision (text+image) | `fetch_smolvlm_256m.bat` |

**40+ additional models** should work without code changes — see
[docs/models_and_vision.md](docs/models_and_vision.md) for the full
compatibility analysis (Llama family, Qwen2/2.5, DeepSeek distills, etc.).

The engine auto-detects:
- **QK-normalization** (Qwen3 only) from weight key presence
- **Tied embeddings** (`shared_classifier`) by comparing tok_emb and lm_head
- **Fused weights** (Phi-3-style `qkv_proj` and `gate_up_proj`) during export
- **Tokenizer type** (BPE vs SentencePiece) via `is_sentencepiece` flag
- **RoPE base frequency** from config, overridable with `-rope-base`
- **RoPE scaling** (llama3 factor=32) from checkpoint header

---

## Architecture

### Core pipeline (LLM)

```
User prompt (text)
    -> encode()              [BPE/SP tokenizer, pure ASM]
    -> prefill_forward()     [fused matmul + RMSNorm + RoPE, per-layer dispatch]
    -> forward()             [per-token: matmul -> attention -> SwiGLU -> classifier]
    -> v7s_sample()          [softmax + top-p + AES-NI RNG, pure ASM]
    -> decode()              [token ID -> string, pure ASM]
    -> output text
```

### Vision pipeline (VLM)

```
Raw RGB pixels + text prompt
    -> v7_chat_image()             [ViT encode -> projector -> inject vision tokens]
        (internally: v7s_vit_encode_callback: patch_embed -> 24 blocks -> mergers -> projector)
    -> text LLM generation with vision tokens
    -> output text
```

### MicroDB — single-file model storage

All model data (weights, tokenizer, config, templates, ViT weights, projector weights,
dispatch tuning) lives in one LZ4-compressed hash table file: `lz4ht.db`. The DLL loads
this file once at init via `db_load()`, then all subsequent lookups are in-memory
`ht_find()` calls. No file I/O during inference. See [docs/getting_started.md](docs/getting_started.md)
for the key schema.

### StreamingLLM (unbounded context)

`v7_set_streaming(enable, num_sinks, window_size)` turns the KV cache into a
stateless ring with attention sinks and a rolling window; tokens that fall off
the window are delivered to a SectorLogo callback as decoded text for
Logo-side long-term memory. Semantics, callback contract and limits:
[docs/generation.md](docs/generation.md).

### Dispatcher

The work-stealing dispatcher is the heart of v7's multi-core performance. All kernel
dispatches (matmul, attention, memcpy, quantize, SwiGLU, ViT linear) flow through one
unified macro `DISPATCH_EXECUTE`:

1. Fill job array with work descriptors
2. Atomically publish `gen_counter++` (workers observe via `pause`-spin)
3. Caller thread participates in work stealing alongside workers
4. Workers claim jobs via `lock xadd` on `next_job` counter
5. `remaining_jobs` tracks completion via `lock sub`
6. Caller waits for `remaining_jobs == 0`

Zero `WakeByAddressAll` calls when workers are spinning (`set_spin(1)`).

### Q8 quantization

Weights are stored as int8 with per-group fp32 scales (group_size=64). Each 64-element
group has its own scale factor computed as `max(|group|) / 127.0`. The Q8xQ8 matmul
kernel uses `vpmaddubsw` + `vpmaddwd` for fast int8 dot products.

---

## What's in the DLL

The DLL exports cover three layers:

1. **Kernel primitives** — `v7d_q8`, `v7d_f32`, `v7d_fused_f32`, `v7d_vit_linear`,
   `v7d_attention`, `v7d_swiglu`, `v7s_rmsnorm`, `v7s_sample`, `v7d_quantize`, etc.
2. **Public `v7_*` API tier** — the C-facing lifecycle with the inference context hidden
   in BSS inside the DLL: `v7_init(db)`, `v7_initialize_inference()`, `v7_chat`,
   `v7_generate` (raw-continuation flag), `v7_chat_image`, `v7_free_inference`, plus
   sampler/session configuration (`v7_set_sampler_temperature`, `v7_set_quantize`,
   `v7_set_max_gen`, `v7_get_position`/`v7_set_position`, `v7_enable_thinking`,
   `v7_set_sidecar`, `v7_db_add`/`v7_db_find`, `v7_encode`),
   `v7_engine_is_dead()`/`v7_get_death_reason()` (internal FATALs are caught and unwound
   to the C caller — the host process survives) and `save_kv_cache`/`load_kv_cache`
   (persist the used KV-cache prefix + position, LZ4-compressed).
3. **Internal inference pipeline** — the multi-turn chat loop (template-rendered,
   KV-cache position preserving), the vision pipeline (image + text -> ViT encode ->
   text LLM), and the single-turn generate loop on its own file-backed KV cache.
   These are implementation stages behind the public tier, not exports.

---

## Repository structure

```
v7/
|   src/                 NASM assembly sources (.asm, .inc) + headers
|   src/logo/            SectorLogo sidecar scripts shipped in every DB
|   tools/               validator/benchmark/DB-tool sources
|   examples/            example programs (c_inference, server, rag, sectorlogo)
|   docs/                documentation
|   weights/             model checkpoints (not committed)
|   bin/                 build output (DLL, tools, examples, server)
|   backgrounds/         server demo backgrounds (incl. bumwolf_avatar.jpg,
|                        the Ardmark demo avatar; scripts\ardmark-server.bat
|                        runs the demo as a persistent chat server)
|   vision/              vision gate fixtures + recorded baselines
|   scripts/             every driver bat (build, verify, server lifecycle,
|                        fetch, benchmark, tuning) — run them from the root,
|                        e.g. scripts\build.bat
|   autoexec.logo        SectorLogo standard library (stdlib)
|   FILE_ID.DIZ          classic shareware-style package descriptor
|   Makefile             native Linux build driver (libv7_threading.so + tools)
+-- CHANGELOG.md

```

---

## Building

**Prerequisites:**
1. [NASM](https://nasm.us/) 3.x (the release is built and verified with 3.02rc6) — add to PATH or set `NASM_DIR` environment variable
2. Visual Studio 2022 or later with C/C++ workload (for MSVC linker)
3. Python 3 with: `torch`, `transformers`, `safetensors`, `huggingface_hub`, `sentencepiece`, `protobuf`, `jinja2`
4. 4+ GB RAM for small models, 12+ GB for Phi-3-mini
5. (optional, dev only) [Zydis](https://github.com/zyantific/zydis) crash
   disassembly — `build.bat` auto-detects `zydis-master\include\Zydis\Zydis.h`
   + `Zydis.lib`; when absent it compiles the tools against the no-op stub in
   `tools\zydis_stub\` (same interface, no decoder, no crash dumps). See
   `docs\zydis_debug_readme.txt`. The shipped release always builds against
   the stub; the dev tree always keeps real Zydis.

**Build everything (from the project root):**
```bat
scripts\build.bat
```

This runs the full build pipeline (DLL + tools + validators + examples + server).
Comment rules are enforced on every ASM instruction. No validation or inference
is run — use `scripts\verify.bat` for that.

**Validate the ASM component:**
```bat
scripts\verify.bat
```

This calls `scripts\build.bat`, builds the Qwen3-0.6B database (mandatory), then runs all
validators and inference smoke tests. ABORTs on first failure. Budget roughly
**15 minutes and ~6 GB of disk** for the first run (the model download and
export happen in the `scripts\fetch_qwen3_0.6b.bat` step; later verification runs
reuse the fetched checkpoint).

**Vision and server/RAG gates (separate by concern):**
```bat
scripts\verify_vision.bat
scripts\verify_server_and_rag.bat
```

### Building on Linux (alpha)

The same source tree builds natively on x86-64 Linux: NASM assembles the
engine into `libv7_threading.so` (MSVC-free), gcc/g++ build the host tools.
The Linux port is tested on Ubuntu 24.04 (also inside WSL2) and behaves
identically to Windows — same kernels, same worker-pool spinning, same futex
pair (publish wake + worker sleep) as the only synchronization calls.

**Prerequisites:** `nasm` (2.16+; the release is verified with 2.16.03 and
3.02rc6 — no NASM binary ships for Linux, install it with your package
manager), `gcc`, `g++`, `make`. Fetching and exporting
a model additionally needs `curl`, `sha256sum`, and the Python toolchain from
the prerequisites above (`python3` with `torch`, `transformers`,
`safetensors`, `numpy`) — `fetch_qwen3.sh` checks all of them before it
starts downloading. On Linux, prefer the CPU-only torch wheel (roughly 10x
smaller than the CUDA default): `pip install torch --index-url
https://download.pytorch.org/whl/cpu`. Prepared `.db` databases need no
Python at inference time. If `git clone` from gitlab.com is refused by a
restrictive proxy (HTTP 403), retry, or fetch the archive endpoint
`https://gitlab.com/api/v4/projects/<id>/repository/archive.tar.gz?ref=main`.

**Build the production binaries:**
```sh
make -j$(nproc)
```

**Fetch a model and build its database:**
```sh
bash src/linux/fetch_qwen3.sh
```

**Validate (all validators + F32/Q8 inference smokes; needs the database):**
```sh
make check MODEL=weights/databases/qwen3_0.6b/lz4ht.db
```

`make check` builds the validator set on top of `all` and, without `MODEL=`,
runs the model-free validators only. Notes:

- the Linux gate runs 20 validator programs; the 21st Windows validator
  (`v_win32_quote_validate`) tests Win32 command-line argv quoting and has no
  Linux counterpart;
- the Linux server decodes uploaded images through the vendored
  `src/linux/stb_image.h` (PNG/JPEG/BMP/GIF) — same pixel contract as the
  Windows GDI+ path (native size, 64 M-pixel cap, EXIF orientation, alpha
  flattened onto white);
- `v7_set_worker_priority` records the requested value without changing OS
  scheduling class (Linux has no direct equivalent of the Windows priority
  classes);
- the engine keeps the Windows register convention on every platform, so all
  host-side declarations carry the `ms_abi` attribute on Linux (`v7_lib.h`
  handles this; HOST module exports for the SectorLogo FFI must add it
  themselves — see `examples/sectorlogo/run_sectorlogo.c`).

---

## Quick start

```bat
:: 1. Build everything (needs Visual Studio's C++ workload; NASM ships in tools\)
scripts\build.bat

:: 2. Download a model and build its database (one command, ~5 minutes)
scripts\fetch_qwen3_0.6b.bat

:: 3. Verify (build + all validators + inference tests; needs the database from step 2)
scripts\verify.bat

:: 4. Run inference (C path, F32)
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db

:: 5. Run inference (pure ASM path)
bin\dbg_inference.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db

:: 6. Run image recognition (VLM; needs a vision database from its fetch bat)
bin\dbg_image_recog.exe -q8 0 bin\v7_threading.dll weights\databases\llava_onevision_qwen2_0.5b\lz4ht.db cat640.raw "What is in this image?"
```

### Testing all models

```bat
:: Run full verification (validators + inference tests)
scripts\verify.bat

:: Run all present text LLMs ("capital of France" in F32 + Q8)
scripts\verify_llms.bat

:: Run every model database in weights\databases\ (3 questions × F32 + Q8)
scripts\test_all_models.bat
```

---

## RAG: Retrieval-Augmented Generation on v7

v7 includes a RAG subsystem built on the SectorLogo sidecar scripting system.
It works at an unusual place: **inside the model's own chain of thought**.
When the model closes its think block (`</think>`), a per-token hook fires,
the accumulated thought text is used as a BM25 query against a local text
knowledge base, and the retrieved lines are spliced into the thought via the
rewrite engine (KV rollback -> re-encode -> re-prefill -> resume). The model
then continues reasoning with the retrieved knowledge present as if it had
thought of it itself.

Two systems share this substrate:

### Document augmentation (static RAG)

Attach a text knowledge base — manuals, wikis, notes, books — and the model
answers questions from it. Deployment is three files: the KB text file, the
retrieval tool (`examples\rag\bm25_rag.exe`), and a ~40-line Logo sidecar
whose only corpus-specific bytes are the retrieval options string.

```bat
bin\inference_example.exe v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db ^
    -i "your question about the document" -n 1500 -q8 1 -rewrite ^
    -script examples\research\sidecar_production_final.logo
```

Validated at both ends of the scale: a 60-line knowledge base (fact
injection into answers) and a 4 MB / 31k-verse bible corpus (exact verse
retrieval, verses quoted in answers).

### Memory augmentation (persistent memory)

The memory system grows the knowledge base automatically, giving the model
cross-conversation memory: at the end of each turn the conversation is
distilled into three factual lines by a second model pass and appended to a
corpus; at the next conversation's think-close the corpus is queried and the
best lines are injected as recalled memories. The model remembers who you
are, what you told it, and what you corrected — from a plain text file, with
no changes to the engine.

```bat
scripts\modelchat.bat "Hi! My name is Alex, I'm a software engineer in Berlin and I enjoy hiking."
:: ...later, in a fresh process:
scripts\modelchat.bat "What do you remember about me?"
```

Corrections in later turns are distilled as newer entries and dominate
earlier mistakes in retrieval. The corpus is a readable, editable text file
(`examples\rag\memory\ragg_base.txt`).

### Building your own

The sidecar scripts are data, not engine code — a new corpus or a new
injection policy is a new `.logo` file plus a knowledge base. The complete
reference (injection strategies with measured verdicts, the empirical laws,
floor calibration per corpus, troubleshooting) is in the documentation:

| Document | Content |
|----------|---------|
| `docs\retrieval.md` | main reference: every implementation analyzed — architecture, strategies, laws, use cases |
| `docs\research_logbook\rag_research_log.md` | the master experiment log (verbatim, all five artifacts per round) |
| `docs\research_logbook\rag_research_analysis.md` | per-experiment conclusions and cross-experiment patterns |
| `docs\research_logbook\` | verbatim experiment devlogs and annotated end-to-end runs |
| `docs\internet_rag_dev_log.md` | internet RAG development log |

A note on scope: retrieval is lexical BM25 over local files — no vector
store, no embedding model, no network. The research logs document the
measured failure modes honestly (query poisoning when the model's prior is
wrong, binary-question oscillation, per-corpus floor calibration); read
the limitations section of `docs\retrieval.md` before relying on it
in production.

### Internet RAG (web_ragg)

The `examples\rag\web_ragg\` application extends RAG with live internet search.
It distills the model's thought into a short search query, fetches web
results via DuckDuckGo, downloads and parses the top pages, then distills
the best paragraphs for injection. The full pipeline (HTML parsing, HTTP
client, text shortening, paragraph distillation) runs locally with no
server-side dependencies beyond the search engine.

### v7_server — HTTP API server

`v7_server.exe` (built from the split sources in `examples\server\`, entry
point `v7_server_main.cpp`) provides an
OpenAI-compatible HTTP API for multi-turn chat with vision support:

```bat
scripts\start_server.bat                   :: own window, restart loop
curl http://localhost:8080/v1/health       :: health check
curl -X POST http://localhost:8080/v1/chat/completions -d @request.json
scripts\server_set_sidecar.bat examples\...\x.logo :: load RAG hook at runtime
scripts\stop_server.bat                    :: clean shutdown
```

The server loads sidecars at runtime via `GET /sidecar?<path>`, supports
vision requests with base64-encoded images, and runs in a supervised
restart loop (crash -> auto-restart, clean `/quit` -> stop).

---

## SectorLogo64 scripting — floats, sorted lists, external tools, LOCAL scope

The sidecar scripting language (`run_sectorlogo.exe`, per-token hooks, RAG
sidecars) has a native float type. Values are 30-bit truncated IEEE
binary32 packed into the tagged-value word (tag `0xC0000000`) — no heap cell,
GC-inert by construction.

```logo
; arithmetic promotes: any float operand makes the result a float
OUTPUT 1.5 + 2.5                 ; 4.0
OUTPUT 10 + 4.0                  ; promotion, result 14.0
OUTPUT 10 / 4                    ; stays integer 2 -- int x int is exact

; math primitives (x87-backed, 1-ulp class accuracy)
OUTPUT LogBaseTwo 8.0            ; 3.0
OUTPUT PowerOfTwo 3.5            ; 11.313708
OUTPUT NaturalLog 2.71828        ; 0.99999
OUTPUT FloatRemainder 10.5 3.0   ; 1.5
OUTPUT SquareRoot 2.0            ; 1.4142136
OUTPUT Power 2.0 10.0            ; 1024.0

; conversions and rounding
OUTPUT AsInteger 2.7             ; 2 (truncate toward zero)
OUTPUT Round 2.5                 ; 2 (round-half-to-even)
OUTPUT Floor -2.1                ; -3.0
OUTPUT IsFloat 1.5               ; TRUE  (IsNumber 1.5 is NIL)
```

Rules: floats are falsy only for exact `0.0`/`-0.0`; bitwise operators
reject floats; the infix `=` compares numerically across int/float/string-atom
spellings (`1` = `1.0` = `"1"`). Formatting is fixed-point for magnitudes
in `[1e-3, 1e9)` and 6-significant-digit exponent notation outside, with
`nan`/`inf`/`-inf` specials.

The sorted-list toolbox makes ranking algorithms expressible in pure
SectorLogo (BM25-style scoring without external tools):

```logo
; Sort ranks [key payload] sub-lists by the key (stable radix sort)
Make "ranked" Sort [[2.5 "chunk-b"] [1.0 "chunk-a"] [7.25 "chunk-c"]]

; binary-search bounds on a sorted list
OUTPUT FindFirstAtLeast :sorted 0 4.0  ; first index with car >= 4.0
OUTPUT FindFirstAtMost :sorted 0 4.0   ; last index with car <= 4.0

; approximate hash map: HashNumber keys + Sort + FindFirstAtLeast probe
OUTPUT HashNumber "some-key"          ; CRC32C-derived FLOAT in [0,1)
```

`Length`, `Item`, `SetItem`, `Reverse`, `FindItem` complete the list toolbox;
`LOCAL "name` declares function-scope variables with in-place `Make`
updates (O(1) loop counters instead of dynamic-binding pile-up);
`TextToNumber` parses text to numbers; `Split` splits strings PHP-style;
`Random` is AES-NI-backed with an optional `RealRandom` RDTSC entropy
toggle; `DefineCommand "name" [:params] "exe_path" "fmt"` defines a
SectorLogo function that calls an external executable with formatted
arguments via `v7_execute_fmt` (e.g. `DefineCommand "rag_search" [:query]
"examples/rag/bm25_rag.exe" "search \"%s\""`); and an optional `autoexec.logo`
standard library loads at interpreter init. Language reference:
[docs/sectorlogo.md](docs/sectorlogo.md).
Interpreter performance is tracked in
[docs/sectorlogo_performance.md](docs/sectorlogo_performance.md)
(cross-language benchmarks: SectorLogo vs C, Python, JavaScript).

---

## Key exports

| Export | Purpose |
|--------|---------|
| `v7d_q8` | Q8xQ8 matmul (work-stealing dispatch) |
| `v7d_f32` | F32xQ8 matmul (work-stealing dispatch) |
| `v7d_fused_f32` | Fused F32 matmul + RMSNorm + RoPE |
| `v7d_vit_linear` | ViT F32xF32 matmul (dispatched) |
| `v7d_attention` | Multi-head attention (per-head dispatch) |
| `v7s_sample` | AVX2 softmax + AES-NI RNG + top-p sampling |
| `v7_chat` / `v7_generate` / `v7_chat_image` | Public-tier chat/generate/image loops (no context argument — BSS singleton) |
| `v7_init` / `v7_initialize_inference` / `v7_free_inference` | Public-tier lifecycle |
| `v7_set_sampler_temperature` / `..._topp` / `..._topk` / `..._seed` / `..._presence_penalty` | Sampler configuration (temperature 0 = greedy default) |
| `v7_set_quantize` / `v7_set_max_gen` / `v7_get_position` / `v7_set_position` | Forward mode, generation cap, KV-position query/rewind |
| `v7_engine_is_dead` / `v7_get_death_reason` | Pseudo-exception queries: 1 + reason text after a caught internal FATAL |
| `save_kv_cache` / `load_kv_cache` | Persist/restore the used KV-cache prefix + position (LZ4, scales with pos) |
| `v7s_get_user_prompt` | Current turn's raw user prompt (RAG query augmentation) |
| `v7_execute` | Spawn child process, capture stdout |
| `v7_execute_fmt` | Formatted spawn+capture (DEF_EXE_FN engine) |
| `v7sprintf` | Bounded sprintf with %f, bounds checking |
| `encode` | BPE tokenizer |

See [docs/native_api.md](docs/native_api.md) (Complete Export List section) for the complete export list.

---

## Standalone tools

The project includes standalone executables for validation, benchmarking, and database
inspection. See [docs/getting_started.md](docs/getting_started.md) for the complete list.

---

## Performance

All kernels are hand-written AVX2 assembly running on a work-stealing thread pool.
The scheduler achieves near-linear scaling across cores on matmul and attention dispatches.
The fused F32 kernel (matmul + RMSNorm + RoPE in one pass) eliminates redundant memory
round-trips. Q8xQ8 quantized matmul keeps memory bandwidth as the primary bottleneck,
which is the expected regime for int8 inference.

| Kernel | Shape | Q8 (GB/s) | F32 (GB/s) |
|--------|-------|-----------|------------|
| Matmul | 1024x1024 | 382-426 | 176-197 |
| Matmul | 1024x2048 | 295-336 | 136-161 |
| Matmul | 2048x1024 | 280-320 | 130-155 |

Measured on Ryzen 5 7430U (12 logical cores), all workers active (>1000% CPU).

---

## About & Contact

This project is developed and maintained by **Görg Pflug / CPKI GmbH**.

If your team is working on edge AI, low-latency inference, embedded systems, or performance-critical x64 Windows applications, we are available for specialized consulting, contracting, and custom development.

The pure-assembly architecture of v7 is specifically designed to be auditable and adaptable for enterprise, air-gapped, or proprietary deployments.

- **Contact:** pflug@cpki.de (Görg Pflug direct) or info@cpki.de
- **Phone:** +49 7033 369 318 (Görg Pflug) / +49 6221 7252 893 (CPKI GmbH)
- **Company:** CPKI GmbH, Rohrbacher Str. 87, 69115 Heidelberg, Germany
- **Website:** https://www.cpki.de
