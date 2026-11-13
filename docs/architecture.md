# System Architecture

V7 brings text and image-language inference into an ordinary native x64 application on Windows and
Linux. The application loads the engine library (`v7_threading.dll` on Windows,
`libv7_threading.so` on Linux), opens a compatible model database, and uses the public C
interface to generate responses. Both builds share the Windows x64 register convention (host-side
declarations carry the `ms_abi` attribute on Linux), so the same application source works against
either library. The library also contains a general-purpose Logo interpreter,
persistent data storage, and text-search facilities. Those components can be used independently of
generation, so an application need not run a model merely to search its documents or process stored
data.

This manual describes the system from an application integrator's perspective. It begins with the
capabilities available through the public interface and the data required to load a model. It then
explains how the native execution engine, SectorLogo scripts, retrieval routines, and generation
controls work together. Lower-level subjects, including dispatcher synchronization and patchable
prefetch instructions, appear later as implementation details. They help explain performance and
maintenance requirements, but they are not additional features that an application needs to configure
before it can use V7.

## Contents

- [System overview](#system-overview)
- [Implementation design](#implementation-design)
- [Deployment and supported configurations](#deployment-and-supported-configurations)

---

## System overview

An application begins by loading the DLL and opening a model database. Once it has initialized the
inference context, it can generate text through the public C interface. The database and SectorLogo
interpreter are also available for tasks that do not require model execution, including local search and
data processing. This chapter starts with those application-facing operations, then follows their
implementation into the native execution path.

The runtime targets x64 and uses native assembly routines with AVX2 instructions for the main
model computations. A C or C++ application loads the library directly; neither Python nor a GPU is
needed at inference time. Model weights, tokenizer data, prompt templates, and configuration are stored
in a compressed MicroDB database that is distributed separately from the library. The memory and storage
required for a deployment therefore depend on the chosen model.

The documented text-model configurations include Llama-3.2-1B, Qwen3-0.6B, Qwen3-1.7B, Qwen2.5-0.5B,
Phi-3-mini-4k, SmolLM2-135M, SmolLM2-360M, TinyLlama-1.1B, and MiniCPM5-1B. Qwen3-8B is
configuration-compatible but its approximately 16 GB database is not included in the development build.
The three tested vision-language configurations are Qwen3-VL-4B, LLaVA-OneVision-Qwen2-0.5B, and
SmolVLM-256M. The documentation describes F32 and Q8 execution paths and the shared ViT processing
architecture; consult [the model compatibility
matrix](models_and_vision.md#model-support-and-configuration) for the tested configurations and specific
limitations.

In addition to inference, the host can run SectorLogo programs to inspect candidate tokens, adjust
sampling, retrieve external data, and replace previously generated text. Stream replacement synchronizes
the token history and affected KV-cache state before generation continues. These are programmable
application facilities, not automatic changes to the model's capabilities. Model-specific configuration
is stored in MicroDB, so the DLL does not need to be recompiled when a supported model database is
selected. The [implementation design](#implementation-design) section describes the corresponding native
execution paths.

### Design goals

V7 is intended for applications that benefit from local, CPU-based inference and direct access to the
generation process. The library runs in the application's own process, without requiring a separate
inference service. It does not need a GPU or an active network connection to execute a prepared model,
which is useful on systems where those resources are unavailable or assigned to other tasks.

The development tools and the deployed runtime have different requirements. Model databases are prepared
from upstream checkpoints using the supplied export tools, which require Python. Once the database has
been prepared, the application needs the V7 DLL and the corresponding model data; Python is no longer
part of the execution path. This separation keeps conversion and packaging work outside the application
while allowing the runtime to use a consistent native representation.

### Architecture

The application-facing interface is deliberately small. A host loads `v7_threading.dll`, opens a
compatible MicroDB database, and initializes the inference context through the `v7_*` functions. The
context remains inside the DLL as a BSS singleton, so the host does not allocate an inference object or
pass a context pointer on each call. The public operations include `v7_init`, `v7_initialize_inference`,
`v7_chat`, `v7_generate`, `v7_chat_image`, `v7_generate_image`, and `v7_free_inference`. The [native API
manual](native_api.md) specifies their arguments and lifecycle requirements.

The model database supplies the information needed by the decoder rather than requiring a separate build
of the DLL for each supported model. The decoder implements RMSNorm, rotary positional encoding (RoPE),
grouped-query attention, and a SwiGLU feed-forward network. Configuration entries select model-specific
operations such as Qwen3 QK normalization, Qwen2 and Qwen2.5 QKV biases, Llama 3 scaled RoPE, tied or
separate embeddings, and head dimensions of 64, 72, 96, or 128. The database builder, `db_build_tool.c`,
precomputes dimensions, strides, attention scales, positional parameters, and other constants, including
values stored as IEEE bit patterns. Assembly code reads those prepared fields instead of recalculating
model-dependent values during generation.

The vision path uses a shared encoder implementation for two related families. Qwen3-VL requires RMS
normalization, rotary positions, DeepStack, and spatial merging; the SigLIP path uses LayerNorm, GELU,
and an MLP projector. The relevant code resides in `vit_forward.inc`, and the `v7_chat_image` interface
accepts an RGB buffer supplied by the application. Image-file decoding belongs to the caller rather than
to the inference API.

Model weights, tokenizer vocabulary and merge data, chat templates, runtime configuration, vision
weights, and projector parameters are stored in a single `lz4ht.db` database. The format uses the
project's LZ4-like LZ77 compression. Normal inference reads the loaded in-memory representation, so
token generation does not repeatedly access the model file. This gives a basic deployment two essential
artifacts: the DLL and a model-specific database. Python is used to prepare model data, but the
inference runtime does not depend on a Python installation.

The execution core is written in NASM x64 assembly. Tokenization, prefill, generation, sampling, the
vision encoder, and dispatched kernels use native code; the inference hot path does not require the C
runtime. The shared `DISPATCH_EXECUTE` mechanism distributes suitable kernel operations across worker
threads. Workers obtain jobs using `lock xadd`, report completion through `lock sub`, and can use
pause-based waiting during active dispatch without repeated `WakeByAddressAll` calls. These choices
concern scheduling overhead and are not part of the public API contract.

V7 also supports execution settings that vary by workload. Prefetch instructions at defined patch points
can be adjusted at runtime, including their distances and instruction types, without recompiling the
DLL. These changes apply to cache-prefetch instructions, not to the arithmetic performed by the kernels.
The [tuning manual](tuning.md) describes the available parameters and their measurement requirements.

The interpreter provides a separate programming layer. SectorLogo runs general-purpose Logo programs and
can also execute a script at the candidate-token boundary during generation. The script can inspect
candidate token information, request another sample, adjust sampling parameters, query MicroDB, invoke
native DLL functions through FFI, or replace accepted text with `SetStream`. When rewriting is
requested, V7 retains the unaffected KV-cache prefix, re-encodes the changed suffix, and continues
inference from the updated context. The [SectorLogo manual](sectorlogo.md) explains the language; the
[generation manual](generation.md) documents the host-specific controls.

Conversation state can be saved with `save_kv_cache` and restored with `load_kv_cache`. The
runtime stores keys and values in separate FP16 buffers and serializes only the occupied rows: normally
`pos` rows per layer, or `min(pos, S + W)` rows with StreamingLLM enabled. The data is compressed
using the project's LZ4-like LZ77 codec. A 48-byte header records the sizes, absolute position,
M-RoPE state, `KV16` storage marker, and save-time streaming configuration/valid-window count. The
loader checks the marker and row geometry, and rejects incompatible streaming configurations; it does
not identify the model, so save and load must still use the same model. Refer to the [native API
reference](native_api.md#kv-cache-file-format) and `tools/v_stream_validate.cpp` for the current contract.

### Programming generation and using application data

An application can use V7 as a conventional inference library by calling its native C functions. It can
also execute SectorLogo programs during generation when it needs closer control over the output. The
script runs at a defined candidate-token boundary, where it can inspect the candidate through `:Piece`
and `:TokenNumber`, change sampler settings with `SetTemperature` and `SetNucleus`, request another
sample with `Reject`, or end the current turn with `Halt`. Those operations apply to the pending token;
they do not require the application to implement its own transformer loop.

Stream rewriting provides a different kind of control. `GetStream` exposes text that has already been
accepted, and `SetStream` replaces that text with an application-supplied value. V7 then brings its
token history, sampling bookkeeping, and transformer state into agreement with the replacement before
generation continues. When the change affects only the end of the stream, the unaffected KV-cache prefix
can be retained. A change earlier in the stream requires more tokens to be evaluated, and a wrapped
StreamingLLM cache may require a broader reconstruction. The application therefore chooses both the
replacement policy and when the additional computation is worthwhile.

This facility can be used for retrieval during generation. A SectorLogo script might inspect the
accepted reasoning text when the pending candidate is `</think>`, construct a query, retrieve relevant
passages, and insert those passages into the working context before the model proceeds to its answer.
The retrieved text becomes part of the model's context, rather than a separate postprocessing
instruction. The experiments in the [retrieval manual](retrieval.md) describe workloads in which this
approach helped a small model incorporate knowledge that it did not consistently use when the same
material was supplied only in an initial prompt. These observations apply to the tested tasks and should
not be taken as a general guarantee.

SectorLogo supplies more than the inference commands. Its normal language features include lists and
other collection operations, arithmetic, and native string processing. The floating-point representation
uses 30-bit truncated binary32 values, and the interpreter provides x87-based functions such as
`LogBaseTwo`, `PowerOfTwo`, and `FloatRemainder`. Search programs can combine these operations with
MicroDB access through `DatabaseGet` and `DatabaseSet`, including BM25 scoring without an external C
search routine. A script can also invoke a Win64 DLL export through `CallLibrary`, so it may call
functions supplied by the host application through the existing FFI mechanism.

The model database is the connection between these facilities and the inference engine. A prepared
`lz4ht.db` contains the selected model's F32 or quantized weights, tokenizer vocabulary and merge rules,
chat templates, RoPE parameters, and any required vision or projector configuration. It can also contain
the SectorLogo script stored under `logo:sidecar`. The database uses the project's LZ4-like LZ77
compression. A basic deployment therefore consists of `v7_threading.dll` and a database prepared for the
chosen model. It does not require Python, a package manager, additional inference drivers, or a network
service at runtime.

The [SectorLogo manual](sectorlogo.md) describes the general-purpose language and its native FFI. The
[generation-control manual](generation.md) defines the host commands, callback timing, and rewrite
algorithm, while the [storage manual](storage_and_layouts.md) specifies the database keys and binary
layouts.

### Repository layout

The native runtime and its public interface are maintained under `src/`. Assembly modules and their
shared macros reside in `src/asm/`, while `src/v7_lib.h` defines the C interface used by host
applications. The principal public entry points include `v7_init`, `v7_initialize_inference`, `v7_chat`,
`v7_generate`, `v7_chat_image`, and `v7_free_inference`.

The `tools/` directory contains the database utilities and the programs used for validation,
benchmarking, and inspection. The build produces validators named `v_*.exe`, benchmarks named `b_*.exe`,
and MicroDB inspection utilities named `mdb_*.exe` from the corresponding tool sources. Build scripts
and Markdown manuals are maintained at the repository root and under `docs/`; the [getting-started
guide](getting_started.md) describes which programs are needed for an ordinary installation and which
are intended for development work.

## Implementation design

The public API describes what an application can request. The implementation sections explain how V7
performs that work on a Windows x64 CPU. They cover the native kernels, dispatch threads, prefetch
configuration, model storage, and the interfaces used by scripting and vision processing. These
mechanisms are important for optimization and maintenance, but they are not additional public API
features. Application code can use the supported operations without depending on a particular kernel or
instruction layout.

### Prefetch and dispatcher implementation

#### Self-Modifying Code for Runtime Prefetch Tuning

The v7 DLL uses self-modifying code to dynamically patch prefetch instructions. When a new matrix shape
(N × D) is dispatched, the configuration path visits the 41 active prefetch slots and updates the
displacement bytes or instruction opcode as required. The current implementation issues `mfence` and
`lfence` after patching. These instructions order execution on the calling thread; the preceding batch
must be complete before shared instructions are modified. This allows per-shape prefetch selection
without recompilation; the benefit still depends on the processor and workload.

#### Dispatcher Synchronization in the Active Execution Path

The shared-counter batch dispatcher uses `lock xadd` to claim batches of jobs and pause-based spinning
while threads wait for new work. During an active spin phase, the dispatch path does not repeatedly
invoke operating-system notification functions. This reduces one source of scheduling overhead, although
overall dispatch cost still depends on the matrix shape, worker state, cache behavior, and competing
system activity.

### Native inference execution

A distinction is necessary between throughput measured on a small kernel and throughput for complete
model inference. A quantized projection matrix that fits in the shared last-level cache can produce an
effective GB/s figure far above physical memory transfer bandwidth. During single-token decoding of a
model larger than the cache, repeated weight reads may instead be limited by DRAM throughput. Refer to
[dispatch and runtime tuning](tuning.md) for the measured shapes and the interpretation of their
bandwidth figures; this documentation does not infer end-to-end tokens/s from kernel GB/s.

The native inference path is implemented in NASM x64 assembly, with AVX2 instructions used in the
compute kernels for which they are appropriate. The DLL performs the transformer computation directly
instead of delegating model execution to a separate C or C++ inference library. Its core operations
include F32 and Q8 matrix multiplication, fused matrix multiplication with RMSNorm and RoPE where
supported, grouped-query attention, SwiGLU feed-forward computation, quantization, tokenization,
sampling, and the generation loop. The vision encoder and projector use the same native execution
environment.

Operations suitable for parallel execution use the dispatcher in `dispatch_execute.inc`. The calling
thread participates alongside the worker threads, claiming jobs from a shared cursor and tracking
completion before the dispatch returns. With spin-based waiting enabled, workers use pause instructions
during the active wait rather than repeatedly invoking operating-system notifications. This reduces a
source of overhead in short operations, although scheduling, cache behavior, contention, and other
system activity still influence end-to-end performance.

### Shape-dependent prefetch configuration

Memory-access patterns differ considerably between small projection matrices, feed-forward layers, and
large vocabulary matrices. V7 therefore allows the prefetch configuration to vary with the matrix shape.
At selected native instruction locations, the runtime can change the prefetch distance or instruction
type using values prepared for the current workload. After a change, `mfence` and `lfence` follow the
instruction writes in the current dispatch path. These fences are not a cross-thread synchronization
protocol; the patching contract requires serialized dispatch entry and completion of the previous worker
batch.

This mechanism makes it possible to evaluate different cache-access settings without producing a new DLL
for every configuration. It does not change the model weights, tokenization rules, or arithmetic
operations, and it does not guarantee that a particular setting will be faster. The measurement
procedure and the parameter layout are documented in the [runtime tuning reference](tuning.md).

### Model storage

Model data is stored in one compressed MicroDB database, `lz4ht.db`. The database contains the model
weights, tokenizer data, configuration, templates, vision weights, and dispatch settings. Initialization
loads the database and builds the in-memory state used by inference. After initialization, normal model
access uses in-memory data rather than reading model files for each token.

This format gives a host two main deployment artifacts: the DLL and its model database. It does not
remove the normal resource requirements of a particular model, such as memory for weights and KV cache.

### Vision processing

The supported vision-language models use a shared, configurable assembly execution path. The database
provides the dimensions, projection settings, and feature-layout information needed by each model
family. Qwen3-VL and the supported SigLIP-based models select different encoder operations within that
framework. The application provides decoded RGB pixels and image dimensions; image-file decoding and
filesystem access remain the caller's responsibility.

The public image API accepts raw RGB pixels and dimensions. The caller owns image decoding and file
handling; the DLL receives the image buffer and runs the model-side processing.

### SectorLogo host interface

SectorLogo is a general-purpose Logo interpreter embedded in the V7 DLL. It can execute scripts
independently of inference, but V7 also provides a callback at the candidate-token boundary. The
callback receives the current token number and decoded piece through `:TokenNumber` and `:Piece`. The
accepted generated text is a separate state value, available through `GetStream`.

The host adds a compact set of generation and database operations. `GetStream` and `StreamLength`
inspect the accepted output, while `SetStream` replaces that output and requests the corresponding state
update. `Reject` requests another sample from the existing logits, and `Halt` ends generation before the
current candidate is accepted. `DatabaseGet` and `DatabaseSet` provide access to MicroDB records. For
native integration, `CallLibrary` invokes Win64 DLL exports using a descriptor that specifies the
function and argument types; `DefineCommand` can establish a command that runs an external program. The
`CallLibrary` descriptor recognizes `"SELF"` for the V7 DLL and `"HOST"` for the registered host module,
with a limit of 16 arguments.

A host application can therefore delegate a decision to ordinary SectorLogo code or use a very short
script that invokes one of its native functions through FFI. The same interface can support retrieval,
application data processing, and sampler adjustments without adding separate callback registration paths
to the public C API. Exact argument types, initialization requirements, and execution rules are given in
the [SectorLogo reference](sectorlogo.md).

### Stream rewriting and retrieval

Stream rewriting begins with text that has already been accepted by the generator. A SectorLogo script
can inspect that text, perform a database query or retrieval operation, and construct a replacement
string. When it calls `SetStream`, V7 compares the original and replacement text to identify the
earliest change. The byte offset is then mapped to the token sequence so the engine can preserve the
preceding KV-cache state, update the output buffer and token history, encode the replacement suffix, and
evaluate the necessary tokens before resuming sampling.

This approach is particularly useful when a retrieval result becomes relevant after generation has
begun. For example, a script may wait for the end of a reasoning section, derive a search query from the
available text, obtain passages through BM25, and place selected facts into the context from which
generation continues. The same mechanism supports text transformations unrelated to retrieval. Rewriting
is limited by the documented stream and token capacities, and its cost depends on how far back the
changed text begins.

The integrated BM25 facilities can also be used without running a model. SectorLogo can prepare queries,
search local documents, and rank passages using the database and native text operations. An external
retrieval program can be invoked through `DefineCommand` or `CallLibrary` when that is preferable. BM25
is based on lexical evidence, so differences in terminology can reduce recall; neither retrieval nor
context rewriting ensures the factual correctness of the model's answer. The [retrieval
manual](retrieval.md) records working examples, their limitations, and the results of individual
experiments.

### Development and validation requirements

The native implementation depends on several contracts that are easy to overlook when modifying an
isolated kernel. Assembly code must follow the Windows x64 calling convention, maintain stack alignment,
and preserve the required registers across helper calls. The dispatcher must publish jobs and record
their completion atomically, while model-specific addresses and sizes must match the database layouts.
Numerical changes also require care: altering accumulation or conversion order can change the model's
results even when the sequence of operations appears mathematically equivalent.

The repository provides focused assembly and data-format validators, database round-trip tests,
SectorLogo tests, inference examples, and performance measurements. The standard `verify.bat` sequence
is the main integration check. A microbenchmark or a single kernel test can establish a narrower
property, but it does not replace the end-to-end checks expected after changes to the runtime.

### Design trade-offs

V7 is designed for applications that need local CPU inference and want to control model execution
through a small native API and a programmable generation interface; those choices also define its
boundaries. The assembly implementation requires specialized maintenance; the supported platform is
limited to the documented Windows x64 processor configuration; and the memory used by model weights
and KV caches depends on the selected model and context length.

The additional programming facilities have their own costs. Lexical search may not retrieve a relevant
passage when the question uses different terminology. Rewriting requires tokenization and forward
computation for the affected suffix. SectorLogo callbacks execute application-defined logic, which must
be tested independently of the transformer's numerical kernels. These are practical considerations when
choosing an integration strategy, not indications that an application must use every available facility.

## Native execution reference

This section follows the native execution path more closely. It records the processor requirements, the
sequence of operations involved in generating a token, and the representations used by the quantized
kernels. These details are useful when checking instruction support, debugging a kernel, or comparing
performance across CPUs. They are deliberately separated from the earlier description of the public
programming interface.

### CPU instruction requirements

The native kernels require the instruction sets listed below. This version does not provide a fallback
for processors missing these instructions. The DLL checks the processor during `DLL_PROCESS_ATTACH` and
public entry points report unsupported CPU features.

| Feature | Used by |
| --- | --- |
| AVX2 | all 256-bit kernels (`vpermq`, AVX2 integer broadcast) |
| FMA | every kernel (`vfmadd` / `vfnmadd`) |
| F16C | FP16 key/value cache conversion (`vcvtps2ph` / `vcvtph2ps`) |
| AES-NI | token/string Intern hash and the sampler RNG (`aesenc`, legacy SSE encoding — one AES round, identical bits, runs without VAES) |
| BMI1 / BMI2 | worker placement (`blsi`, `blsr`), hot-loop search (`tzcnt`), tuner (`bzhi`) |
| SSE4.1 / SSE4.2 | microdb key hash (`vptest`, `crc32`) |
| RDRAND | C dispatch autotuner's hardware RNG draws |
| RDTSCP | serialized tuner timing edge (leaf 0x80000001 EDX[27]; masked by some hypervisors) |

The OS must also enable SSE + YMM state (`XCR0` bits 1 and 2); a hypervisor that reports AVX2 but does
not pass `XCR0` through is rejected the same way, and a hypervisor that masks `RDTSCP` is rejected by
name as well. The engine emits the legacy `aesenc` encoding (AES-NI, 2010+), not the VEX
`vaesenc` (VAES, 2020+), so AMD Zen 1/Zen 2 and Intel Haswell through Comet Lake are eligible;
VAES-capable CPUs run the same binary unchanged.
CPUID features, not the CPU release year, determine support; check the complete feature set
reported by `src/asm/cpu_gate.inc`. The documented development measurements use the following processor: Ryzen 5 7430U
(Zen 3, 12 logical cores).

### Per-token execution sequence

The generation loop executes the model layers, produces next-token logits, and samples a candidate. The
following stages summarize the native execution path. Individual model configurations may use fused
kernels for some of these operations:

1. **RMSNorm** — compute `1/sqrt(Σx²/n + eps)` in one pass, scale every weight element
2. **RoPE** — rotate Q/K vectors by precomputed cos/sin tables (no `sinf`/`cosf` at runtime)
3. **QKV matmul** — fused matmul + RMSNorm + RoPE in a single dispatch (the `v7d_fused_f32` kernel)
4. **Attention** — per-head softmax over the KV cache, GQA with `n_kv_heads` < `n_heads`
5. **SwiGLU** — `silu(gate_w @ x) * (up_w @ x)` with int8 weight quantization
6. **Classifier** — final matmul to produce logits over the vocabulary
7. **Sampler** — AVX2 polynomial softmax + AES-NI RNG + top-p multinomial CDF walk

The dispatcher executes supported kernels through `DISPATCH_EXECUTE`: the caller thread fills the job
array, atomically publishes `gen_counter++`, then participates in dynamic batch scheduling alongside
worker threads. Workers claim jobs via `lock xadd` on a shared `next_job` counter. When the workers use
the active spin mode (`set_spin(1)`), the dispatch path uses `pause` instructions instead of repeatedly
invoking operating-system synchronization functions such as `WakeByAddressAll` or `WaitOnAddress`.
The recorded dispatch measurements are on the microsecond scale and cover job setup, publication,
and acquisition. The benchmark tools in `tools\`, including `b_benchmark.exe` and the per-kernel
load tests, measure the throughput of individual operations. The
[runtime tuning guide](tuning.md#benchmarking) explains the measurement procedure.

### Prefetch instruction patching

The current `matmul_dispatch.inc` path modifies prefetch slots before publishing the subsequent job
batch and issues `mfence` followed by `lfence` after the writes. Those instructions synchronize the
calling thread but do not, by themselves, synchronize the instruction streams of workers executing
concurrently. The implementation assumes the preceding worker batch is complete and a single caller
controls dispatch; applications must not invoke simultaneous independent dispatches. The adjacent
`next_job` and `remaining_jobs` counters share the aligned control region, whereas `gen_counter` has a
separate cache-line alignment. See `src/asm/v7_data.inc` for the layout.

The native implementation uses self-modifying code to configure prefetch distances and instruction
types for different matrix shapes. When a new `N×D` shape is dispatched, the cold path scans 41
prefetch slots and writes new displacement bytes directly into the instruction stream
(`set_prefetch_distance`) or swaps the opcode for a different prefetch variant (`set_prefetch_type`:
NTA, T0, T1, T2, PREFETCHW, PREFETCHWT1, CLDEMOTE). The PREFETCHWT1 and CLDEMOTE variants act as plain
prefetch (or no-ops) on CPUs whose decode lacks them — they are tuning options, never correctness
requirements. The patching thread executes `mfence` and `lfence` after the writes. Independent
concurrent dispatch callers are not supported by this patching contract; this allows per-shape
prefetch tuning without recompiling.

### Q8 data representation

Weights are stored as int8 with per-group fp32 scales (group_size=64). Each 64-element group has its own
scale factor computed as `max(|group|) / 127.0`. The Q8×Q8 matmul uses `vpabsb` for unsigned activation
magnitudes, `vpsignb` to apply their signs to signed weights, and then `vpmaddubsw` and `vpmaddwd` for
pairwise products and wider accumulation. Quantized weights are constrained to `[-127, 127]` so a
two-product 16-bit sum cannot saturate even if an activation magnitude reaches 128: `2 × 128 × 127 =
32,512`. This restriction is part of the data contract; generic int8 vectors would not have the same
saturation guarantee.

---

---

## Deployment and supported configurations

V7 is intended to be embedded in applications that can supply the required model database and CPU
resources. The examples below illustrate possible environments rather than guaranteeing that every model
will fit every device. Deployment requirements depend on the selected model, its precision, the
configured context length, and the application's own memory use. The support tables identify tested
configurations, while the distribution section explains which files an application needs.

### Application examples

A native CPU inference library can be appropriate when applications cannot rely on a separate inference
service or GPU acceleration. The following scenarios indicate possible integration contexts; they do not
imply that V7 provides the complete application, domain-specific verification, or additional hardware
interfaces. Suitability depends on the selected model and on the requirements of the host application:

1. **On-device assistants** — Run supported chat workloads locally. The host application controls whether it sends any data to external services. Applications processing confidential records must implement appropriate access and data-handling requirements.

2. **Industrial user interfaces** — Add natural-language query facilities to Windows systems connected to industrial applications. The host remains responsible for validation and execution of any operational commands.

3. **Offline environments** — Run inference where network access is unavailable. Once the DLL and required database are prepared, normal local inference does not require a package manager or runtime download.

4. **Point-of-sale and kiosk applications** — Use local inference for product searches, frequently asked questions, or operator assistance without requiring a separate inference server.

5. **Offline translation** — Run an appropriately trained model on a Windows tablet or laptop without a network connection. Different language configurations can use different model databases while sharing the same DLL.

6. **Educational robotics** — Provide a text interface to a host-controlled robotics application. The assembly implementation can also be studied using standard debugging and disassembly tools.

7. **Vehicle information systems** — Add local text-generation facilities to Windows-based interfaces. Audio capture and speech synthesis require separate components.

8. **Research document processing** — Use a local model to summarize text, label datasets, and extract entities from documents. The application controls the handling of sensitive information.

9. **Accessibility applications** — Add local language generation to communication and accessibility tools. Applications requiring speech or other device interfaces must provide those integrations separately.

10. **Existing Windows applications** — Use `LoadLibraryA` and the native C API to integrate inference into applications that already have a Windows deployment process.

11. **Game applications** — Integrate local text and image-language inference with an existing game engine. CPU inference does not allocate the GPU resources used for rendering, although CPU memory and scheduling requirements still apply.

---

The deployment details below explain how the DLL and model database are incorporated into host
applications. They also identify which components are optional and which requirements come from the
underlying CPU implementation.

V7 is designed for direct integration into Windows applications. Loading the DLL together with a
compressed model database enables local text and image inference, multi-turn chat, retrieval-augmented
generation, and application-managed memory. The runtime does not require Python, GPU acceleration, a
remote inference service, or network access for local execution. BM25 text retrieval and
persistent-memory features are provided through the project's supporting components and SectorLogo
integration.

---

### Deployment requirements

Applications can load the runtime directly without starting a separate inference service. The choice of
environment determines how the application supplies a database and manages local resources:

- **Systems without network access:** The local runtime and model database do
  not require an online service during inference.
- **Industrial PCs and point-of-sale terminals:** The system can be embedded
  in an existing Windows x64 application.
- **In-vehicle systems and game engines:** CPU inference can operate while the
  GPU is reserved for rendering or other workloads.
- **Standalone Windows applications:** The application loads the DLL and its
  model database without requiring a separate service or container.
- **Privacy-sensitive deployments:** Input and generated content can remain on
  the local system when the application uses local resources.

The inference DLL and MicroDB database contain the runtime and required model data. Optional external
retrieval programs are separate application tools.

---

### Architecture and deployment

The following sections specify how the library is packaged and how its model database, native runtime,
and optional application facilities are connected. Deployment requirements are distinct from
processor-level implementation choices.

#### 1. Embeddable, not a service

V7 is designed primarily as an embedded library rather than a standalone inference service. Its public C
interface provides initialization, text generation, multi-turn chat, and teardown operations. The
documented public error interface reports terminal engine errors to the host for the applicable entry
points. Ordinary DLL integration does not require a listening network port or a separately managed
server process.

#### 2. All model data in one file

The compressed MicroDB database stores the model weights, tokenizer tables, chat templates, supported
vision resources, and tuning parameters. V7 loads the required data at initialization; the normal
inference path uses the in-memory representation rather than accessing model files during generation.
Selecting another supported model requires loading its database, not recompiling the DLL.

#### 3. Native execution implementation

The inference implementation uses handwritten NASM x64 assembly, including AVX2 kernels and a
work-dispatch system. Runtime patch points allow the engine to adjust prefetch instructions for selected
matrix shapes. These techniques give the implementation direct control over CPU instructions and
memory-access patterns while keeping the runtime independent of a third-party inference framework. The
deployment requires the standard Windows runtime interfaces.

---

### Mid-generation stream rewriting

A conventional retrieval workflow supplies selected documents before generation begins. V7 also supports
retrieval during generation through a SectorLogo callback. The script can inspect the accepted generated
stream, search for relevant records, and replace part of that stream with updated text.

For a retrieval-assisted reasoning workflow, the sequence is as follows:

1. The model produces the reasoning or other intermediate text used by the application.
2. When the configured callback boundary is reached, the SectorLogo program reads the accepted stream and constructs a query from the relevant text.
3. The retrieval system selects records from the local corpus or another configured source. The script constructs the replacement text and calls `SetStream`.
4. V7 retains the reusable KV-cache prefix, updates the affected token sequence, evaluates the replacement suffix, and resumes generation.

This mechanism works with supported models without requiring model fine-tuning. It can provide a small
model with relevant external information, but the quality of the resulting answer still depends on the
retrieval policy, the selected model, and the question. In the recorded benchmark, replacing
approximately 300 tokens took roughly 300–900 ms on the reference CPU; the cost of an external internet
search was separate and could be several seconds.

The associated application facilities include local BM25 search, persistent retrieval corpora, optional
internet search, and custom SectorLogo programs. BM25 retrieval can run without initializing the inference engine.
Persistent memory and internet retrieval depend on the corresponding application scripts and utilities;
neither is automatically active in every V7 session. For usage examples, see [the retrieval
manual](retrieval.md) and [the generation-control manual](generation.md).
---

### Supported models

Each model below was tested on both the full-precision F32 path and the compressed Q8 path.

#### Text models, 8 tested

| Model | Params |
| --- | --- |
| Qwen3-0.6B | 0.6B |
| Qwen2.5-0.5B | 0.5B |
| Llama-3.2-1B | 1.0B |
| Phi-3-mini-4k | 3.8B |
| SmolLM2-135M | 0.135B |
| SmolLM2-360M | 0.36B |
| TinyLlama-1.1B | 1.1B |
| MiniCPM5-1B | 1.0B |

#### Vision models, 3 tested

| Model | Params |
| --- | --- |
| Qwen3-VL-4B | 4.0B |
| LLaVA-OneVision | 0.5B |
| SmolVLM-256M | 0.256B |

The vision models accept an image plus a text question and answer in natural language. Model-specific
details such as normalization, tokenizer and positional settings are read from the database. Other
architectures in the Llama, Qwen and DeepSeek families are plausible candidates, but they are untested;
each one needs its own database build and a validator run before its results can be described as
working.

---

### Distribution Contents

- **A small public C API**, load the library, pass a database, chat or generate. The inference state stays inside the library; the host application never manages it.
- **Contained fatal errors**, an internal fatal error unwinds to the host program with a reason string, instead of ending the process.
- **Session persistence**, a conversation's memory can be saved and reloaded, to resume a session in a fresh process.
- **Vision from memory**, the library accepts raw image pixels directly; no image library or filesystem access is involved.
- **Embedded scripting**, per-token hooks, stream rewriting, retrieval control, database access, and native function calls, for custom behavior without touching engine code.
- **An OpenAI-compatible HTTP server**, provided for development and testing, with runtime script loading and a supervised fault-recovery restart sequence.
- **Validation and tooling**, 21 validators, plus benchmarks, database inspection tools, and a performance autotuner.
- The core is hand-written assembly, readable and inspectable, with no framework underneath.

---

### Measured performance

In the November 8, 2026 [CPU inference benchmark](engine_benchmark.md), v7 generated
55.8 tok/s in the short Qwen3-0.6B chat workload and 52.2 tok/s in the poem workload
on a Ryzen 5 7430U with Q8 weights. These rates measure decoding after prefill. The
benchmark also covers Qwen3-1.7B, Phi-3-mini, and MiniCPM5-1B.

The [runtime tuning guide](tuning.md#benchmarking) reports effective GB/s for
individual kernels. The [generation statistics report](performance_report.md)
records an August 2026 measurement of the tokenization, prefill, and decode phases.

---

### Getting started

Windows:
```
scripts\build.bat          # build everything
scripts\verify.bat         # build + validators + inference tests

scripts\fetch_qwen3_0.6b.bat  # one-time model download + database build

bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db
```

Linux:
```
make -j$(nproc)            # production outputs (libv7_threading.so + host tools)
make check MODEL=weights/databases/qwen3_0.6b/lz4ht.db

bin-linux/inference_example bin-linux/libv7_threading.so weights/databases/qwen3_0.6b/lz4ht.db
```

Requirements: x64 with AVX2, FMA, F16C, BMI1/2, SSE4.1/4.2, AES-NI,
RDRAND, RDTSCP and OS-enabled YMM state. The AES rounds use the legacy
`aesenc` encoding, so AES-NI-only CPUs are eligible.
Memory use depends on the selected model and context size; roughly 4 GB may suffice for a small model. See `docs\getting_started.md` and the `README` for the full guide.

---

### Contact & licensing

V7 is developed and maintained by **Görg Pflug / CPKI GmbH**, and is licensed under **GPL v3**.

The source can be reviewed for the supported platforms and execution paths. Distribution and integration
must comply with the GPL v3 license and the applicable licenses for model checkpoints and other
dependencies.

- **Contact:** pflug@cpki.de or info@cpki.de
- **Phone:** +49 7033 369 318 for Görg Pflug, +49 6221 7252 893 for CPKI GmbH
- **Company:** CPKI GmbH, Rohrbacher Str. 87, 69115 Heidelberg, Germany
- **Website:** https://www.cpki.de
