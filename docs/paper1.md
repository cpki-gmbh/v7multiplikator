# V7: An Assembly Inference Engine with Programmable Stream Rewriting and Retrieval

Gorg Pflug CPKI GmbH

## Abstract

This paper describes v7, an inference engine for large language models (LLMs) and vision-language models
(VLMs) written entirely in x86-64 assembly (NASM, AVX2) for Windows. The engine runs Q8 quantized and
full-precision (F32) inference for 10 text-model configurations and 3 vision-language configurations tested in the reported evaluation, using a shared-counter, dynamically
scheduled worker pool. Its spinning dispatch mode avoids operating-system wait calls while workers are
polling.

In addition to the inference core, V7 provides a mid-generation stream-rewriting mechanism: instead of only
inserting retrieved documents into the prompt before generation starts, retrieval can happen while the
model is still thinking. When a model and template emit the marker selected by the application, a small
interpreter embedded in the engine (SectorLogo64) intercepts the token stream, retrieves relevant text
from local data or external web tools, revises the generated stream, retains the unaffected KV-cache
prefix when possible, evaluates the replacement suffix, and resumes generation.

We describe the overall architecture, the engineering trade-offs of writing an inference engine in
assembly, the stream rewriting mechanism, and the episodic memory system built on top of it. We also
discuss the limitations of purely lexical retrieval and the latency cost of rewriting mid-generation.
The source code is available under the GPL v3 license.

## 1. Introduction

Many LLM deployments use high-level frameworks (PyTorch, TensorFlow, vLLM, llama.cpp) that compile
computation graphs down to GPU kernels or optimized CPU routines. These frameworks differ in their
hardware support, execution strategies, and integration interfaces. Their abstractions cannot be assumed
to impose significant per-token overhead without direct measurement.

V7 takes a different approach: the entire pipeline—including tokenization, matrix multiplication,
attention, normalization, activation functions, sampling, and generation—is written by hand in x86-64
assembly. The result is a native DLL (`v7_threading.dll`) whose transformer hot path does not require a
C runtime. The stated binary size is a version-specific build artifact, not a fixed ABI property. The
model data used by the normal inference path (weights, tokenizer tables, configuration, chat templates)
is stored in one compressed hash-table file (`lz4ht.db`) that is loaded once at startup; the native
weight and transformer execution paths then read the in-memory representation. Application-level
retrieval and memory facilities can still perform file I/O or launch external programs during
generation.

This paper describes four implementation components. The performance figures reported later are kernel
measurements and do not establish a ranking against other inference libraries.

1. **Native inference:** An x86-64 assembly engine for transformer LLMs and VLMs, with Q8 and F32 execution paths and recorded kernel microbenchmarks.
2. **Programmable stream rewriting:** A mechanism for changing accepted generated text, preserving reusable transformer state, and evaluating the affected suffix. A retrieval script can use this mechanism with an unmodified model checkpoint when the chosen trigger is available.
3. **Embedded programming:** SectorLogo provides general-purpose scripting and inference-specific host commands, including FFI calls into application code. A script can control the generation process without relying on model-generated function-call syntax.
4. **Retrieval and plaintext memory:** Applications can combine BM25 retrieval with distilled text records, without requiring an embedding model or a vector database. The source includes both interpreted and native retrieval implementations.

We do not claim that pure assembly is the right choice for most deployments. The approach trades
development velocity and portability for fine-grained control over instruction scheduling, register
allocation, and dispatch overhead. We present it as a data point in the design space of inference
systems, and as a platform for exploring agentic behavior at the engine level rather than the
application level.

## 2. Background and Related Work

### 2.1 Inference Engines

Production LLM inference is typically handled by frameworks such as vLLM [1], TensorRT LLM [2],
llama.cpp [3], and ONNX Runtime [4]. These systems optimize transformer inference through techniques
such as PagedAttention, continuous batching, speculative decoding, and kernel fusion. Prefill can be
compute-intensive, while single-token decoding of a model that exceeds the processor caches is often
limited by weight-transfer bandwidth. Most target GPUs; llama.cpp is a notable CPU-focused
implementation with SIMD-optimized kernels written in C/C++.

v7 differs in two ways: the implementation language (handwritten assembly), and the integration of
agentic capabilities directly into the inference loop. The assembly-level implementation gives direct
control over instruction scheduling, prefetch distances, and thread synchronization, at the cost of
supporting only one architecture (x86-64 with AVX2) and one operating system (Windows).

### 2.2 Retrieval Augmented Generation

The contribution described here is the integration of script-driven retrieval and stream editing into
the inference host. The ability to remove KV-cache suffix state or reuse a common prefix is established
in other inference engines; it is not claimed as an original rollback algorithm.

Retrieval-augmented generation (RAG) [5] prepends retrieved documents to the user prompt before
generation begins. The retrieval step typically uses dense vector similarity (embedding models plus
vector databases) or sparse lexical methods such as BM25. In a conventional retrieve-then-generate
application, the retrieved context is selected before generation starts; systems with interleaved
retrieval are also established.

Recent work on self-RAG [6] and corrective RAG [7] moves retrieval decisions into generation, usually by
fine-tuning the model to emit special tokens that trigger retrieval. Active retrieval and interleaved
lines (FLARE [11], IRCoT [12]) retrieve between reasoning steps; Search-o1 [13] is a training-free
search framework operating around a reasoning model; it adds retrieval and a document-analysis step
without requiring a model checkpoint trained specifically for the V7 interface. ReAct [23] and
Toolformer [24] are further precedents for combining language generation and external tools. The
distinguishing feature of V7 is the script-driven, in-engine integration of retrieval with editable
generated context. Some demonstration sidecars insert text without source delimiters; this introduces
provenance and untrusted-input risks rather than constituting a general advantage.

### 2.2.1 Machine rewriting of generated text

The rewrite engine’s core operation—editing previously generated text, restoring the corresponding
generation state, and continuing—is related to four active research families. (1) *Internal
backtracking* in trained reasoners: o1/R1 class models append self-corrections to their own traces; 2026
trace analysis work (ReasonOps [14], ReasoningFlow [15]) treats backtracking as a measurable operator,
and activation steering work can suppress it for token efficiency [16], but the erroneous prefix remains
in context. (2) *Self correction*: surveyed critically in [17], which finds intrinsic self-correction
mostly fails without external feedback; recent measurement work shows that at small to mid scale,
reported self-correction gains are largely answer format repair rather than genuine reasoning change
[18]. (3) *Speculative rollback*: verify then rollback is entering the speculative decoding loop for
safety [19] and code repair [20]. (4) *Diffusion LM editing*: block diffusion models revise finalized
tokens natively [21], and execution state checkpointing restores full inference state at committed
boundaries [22], which is architecturally closest to "continue from an edited past," but requiring a
diffusion architecture or a snapshot runtime. Against this map, v7's mechanism is training-free,
model-agnostic, and script-triggered: the edit decision lives outside the model, and the ordinary
rewrite path keeps a token-aligned common prefix and evaluates the changed suffix; cache removal and
prefix reuse themselves are established mechanisms, including in llama.cpp and SGLang [26].

### 2.3 Agentic Frameworks

Frameworks such as LangChain [8] and AutoGPT [9] coordinate model calls with external tools.
Function-calling APIs [10] can also expose tools through structured requests. Those requests require
validation of the selected operation and arguments. Grammar constraints can ensure valid JSON syntax,
but they do not guarantee that the selected operation is appropriate or that its parameters are correct.

V7 provides an additional control arrangement: the host interpreter executes application-defined logic
at generation boundaries. A script may inspect a candidate token, invoke a native function, or replace
accepted generated text. Model-produced delimiters are one possible trigger, but an application must
select an appropriate policy for each model and template.

## 3. System Architecture

### 3.1 Inference Pipeline

The pipeline follows the standard autoregressive transformer architecture:

1. **Tokenization.** BPE encoding, implemented in assembly. The tokenizer tables are loaded from the MicroDB hash table at initialization. (HuggingFace models using SentencePiece vocabularies are converted to this BPE format at export time by `export.py`.)
2. **Prefill.** A fused forward pass over the entire input sequence, computing key-value cache entries for all layers. The fused kernel (`v7d_fused_f32`) combines matrix multiplication, RMSNorm, and rotary position embedding (RoPE) into a single dispatch, which keeps the activation data in L1/L2 cache instead of writing it out and reading it back.
3. **Decode.** The per-token generation loop. Each step computes RMSNorm, RoPE, QKV projection, multi-head attention over the KV cache, SwiGLU feed-forward, and the final classifier. The output logits go to the sampler.
4. **Sampling.** AVX2 polynomial softmax, an AES-NI-based pseudorandom number generator, and top-p (nucleus) sampling over the top 256 candidate set. Temperature 0 reduces to greedy argmax.
5. **Detokenization.** Token ID to text conversion via the BPE decode tables.

The engine supports both F32 and Q8 paths. In Q8 mode, weights are stored as int8 values with per-group
FP32 scale factors (group size 64). The Q8×Q8 matmul uses `vpabsb` to obtain unsigned activation
magnitudes, `vpsignb` to apply each activation sign to the corresponding signed weight, then
`vpmaddubsw` and `vpmaddwd` to accumulate signed pair products. The exporter and quantizer constrain
weights to `[-127, 127]`; activation magnitude can reach 128, but the largest pair sum is `2 × 128 × 127
= 32,512`, below the 16-bit saturation threshold. This bound is essential: `vpmaddubsw` is saturating,
and arbitrary signed int8 operands do not provide the same guarantee. See `src/asm/v7_macros.inc`,
`q8_kernel.inc`, `quantize_kernel.inc`, and `swiglu_q8_kernel.inc`.

### 3.2 Dispatch and Threading

Kernel dispatch uses dynamic batch scheduling from a shared atomic cursor; this is not deque-based
work stealing. The `DISPATCH_EXECUTE` path distributes work as follows:

1. The caller thread fills a job array with work descriptors (matrix slices, attention heads, etc.).
2. The caller atomically increments a generation counter (`gen_counter++`), which worker threads observe via pause spin loops.
3. The caller claims batches from the shared cursor alongside the worker threads.
4. Workers claim batches of jobs via `lock xadd` on a shared `next_job` counter. Batch sizes come from per shape INI configuration.
5. Completion is tracked by decrementing a `remaining_jobs` counter once per processed batch (`lock sub`), not per job.

When workers are in spinning mode (`set_spin(1)`), the polling portion of the dispatch path avoids
operating-system waits such as `WakeByAddressAll`, `WaitOnAddress`, and `Sleep`. The documented dispatch
timings are microbenchmarks rather than a guaranteed per-call cost on every workload. Spinning consumes
processor time and can affect package temperature and sustained clock frequency, especially on a 15 W
laptop processor. Applications should compare sustained decode throughput, energy use, and thermals with
the sleeping configuration before choosing a mode. The `gen_counter` is cache-line aligned separately,
but `next_job` and `remaining_jobs` are adjacent in the same aligned control block
(`src/asm/v7_data.inc`); this implementation does not claim that those two atomic counters occupy
separate cache lines.

### 3.3 Self Modifying Code for Prefetch Tuning

Runtime prefetch selection and the code-modification mechanism are separate performance questions. A
tuned prefetch configuration may improve a workload, but only a matched patching-enabled/disabled
experiment can establish whether changing instructions at runtime produces an additional overall
benefit.

The engine uses self-modifying code (SMC) to tune prefetch instructions at runtime. When a new matrix
shape (NxD) is dispatched for the first time, a cold path routine scans 41 active prefetch slots in the
kernel code and overwrites the displacement bytes or swaps the opcode (`prefetchnta`, `prefetcht0`,
`prefetcht1`, `prefetcht2`, `prefetchw`, `prefetchwt1`, `cldemote`) based on per shape configuration.
The current dispatch source issues `mfence` and `lfence` after the writes
(`src/asm/matmul_dispatch.inc`); it does not issue `cpuid` at that site. Those fences on the calling
thread alone do not establish a cross-modifying-code protocol for another thread that might execute the
patched instructions. Correct use therefore depends on the single-dispatcher contract and on the
previous worker batch having finished before the next patch. Concurrent independent callers must not
enter dispatch or patch the same kernel without additional synchronization. A formal cross-thread
instruction-stream synchronization argument and a patching-disabled performance ablation are not
supplied by the current measurements.

This permits per-shape prefetch configuration without recompilation. The available opcodes are tuning
candidates, not evidence that each variant has an independent effect on Zen 3; unsupported hints may
have no useful effect. A benchmark comparison with patching disabled is required to isolate the benefit
of runtime patching itself. The technique is specific to x86-64 and requires writable and executable
memory pages, which has security implications and limits its use in sandboxed environments (see Section
9).

### 3.4 Model Storage (MicroDB)

All model data is stored in a single LZ4 compressed hash table file (`lz4ht.db`): weights, tokenizer
tables, configuration parameters, chat templates, ViT weights (for VLMs), projector weights, and
dispatch tuning data. The DLL loads this file once at initialization via `db_load()`; afterwards every
lookup is an in memory hash table query (`ht_find()`). The loaded model is not read from disk for every
transformer operation. Scripted retrieval, persistence, and internet-search tools may still use files
during a generation turn.

## 4. Mid Generation Stream Rewriting

### 4.1 Motivation

Standard RAG retrieves documents before generation begins. This works well enough for factual question
answering, but it prevents the model's own reasoning from guiding retrieval: if the model's chain of
thought reveals a knowledge gap mid-generation, standard RAG cannot react to it.

Fine tuning approaches (self-RAG [6]) address this by training the model to emit retrieval tokens, but
that requires model-specific training and changes the generation distribution. We wanted a mechanism
that works with unmodified, off-the-shelf models.

### 4.2 Mechanism

A `SetStream` rewrite is based on the difference between the accepted stream and the replacement stream,
not on an unconditional reset of the conversation. For example, suppose the accepted generated token
sequence is `t0 ... t99`, and the script inserts knowledge between `t98` and the pending closing marker.
The replacement can retain `t0 ... t98`; only the token-aligned suffix after that prefix must be
recomputed. The actual number of affected tokens depends on tokenization, because a byte-level splice
can change token boundaries.

The token callback sees a pending candidate before it is accepted. A script that triggers on a closing
thinking marker should explicitly record whether the augmentation has already been performed for that
turn; otherwise a subsequently generated matching marker may trigger the same policy again. The exact
trigger count is determined by the script, not by an unconditional guarantee that all model checkpoints
emit one thinking block.

The following example uses a model and chat template that emit a closing thinking marker. Other model
families can use different script-defined trigger conditions; the same callback cannot be assumed to
activate on every supported checkpoint. The operation proceeds as follows:

1. The model is prompted with its chat template. For thinking capable models the template ends at the assistant header, and the model opens its own thinking block and generates its chain of thought inside it. (For non thinking turns the template itself contains an empty think block, which is part of the prefill and never reaches the generation time hook.)
2. A per-token hook in the SectorLogo64 interpreter watches the output stream. When the designated closing marker appears as a pending candidate, the configured script evaluates it before acceptance.
3. The hook extracts the accumulated thinking text and uses it optionally combined with the original user prompt as a retrieval query.
4. The retrieval system (local BM25 or internet search) returns relevant text passages.
5. The hook builds a new context string that includes the retrieved passages and the original reasoning.
6. The hook calls `SetStream`, which triggers the following engine level operations:
     **KV cache rollback.** The KV cache entries produced after the divergence point between old and new token streams are discarded. A longest common prefix comparison between the old stream and the replacement stream determines the divergence point, so any untouched prefix of the reasoning keeps its cache entries.
     **Re encoding.** The new context string is tokenized.
     **Re prefill.** The engine runs a prefill pass over the replacement tokens, populating the KV cache from the divergence point onward.
     **Resume.** Generation continues from the end of the augmented context, with the model now conditioned on the retrieved knowledge.

The subsequent output is conditioned on the revised context. Earlier text retained before the insertion
may still contain incorrect assumptions or explicit uncertainty, so this operation does not guarantee
that all reasoning is consistent with the newly retrieved facts.

### 4.3 Implementation Details

The `SetStream` operation is the most complex part of the mechanism. It requires:

  Tracking the KV cache position (`c_pos`) so the rollback lands on the correct entry.
  Keeping the RoPE position counters consistent after rollback, so position embeddings stay aligned.
  Handling the presence penalty bitmap: tokens that were generated and then discarded must have their penalty bits cleared again, or the sampler would keep avoiding tokens that are no longer in the context (see Appendix A.1 for the exact code).
  Preserving sampler state (temperature, top p, RNG state) across the rewrite.

The replacement tokens are re-encoded into a fixed internal buffer (16,384-token capacity). The current
`chat_rewrite.inc` implementation evaluates the affected suffix one token at a time through `forward_q8`
or `forward_f32`, rather than running a batched prefill. The output buffer is truncated at the
divergence point and updated with the replacement text. The rewrite buffers are preallocated. Reaching
the re-encoding capacity is a terminal error rather than a request to silently truncate the replacement
(`.chat_rewrite_enc_cap_fatal`).

For text-only models, `v7s_mrope_prepare` does not apply a multimodal position delta. Multimodal
position handling is a separate configuration-dependent path; the current contract is described in the
model and vision manual.

### 4.4 Latency Cost

Stream rewriting costs time proportional to the length of the re-prefilled context. The serial
replacement pass recomputes the KV-cache entries for the affected tokens, while the unchanged prefix
remains reusable in the ordinary nonwrapped case; measured on the reference hardware (Ryzen 5 7430U,
Qwen3 0.6B), a rewrite of roughly 300 tokens takes about 300–900 ms. The internet retrieval step, when
enabled, adds network latency on the order of seconds, dominated by page fetching and distillation.

This cost is incurred during the generation turn. The measured 300–900 ms range applies to the specified
example only. Its magnitude depends on the number of recomputed tokens, model, and hardware, so
applications should measure the complete retrieval-and-rewrite operation. The reported 300–900 ms figure
is a workload-specific observation, not a general latency bound. Its interpretation requires the
replaced token count, model, processor state, and whether external retrieval was included. A controlled
comparison with pre-generation retrieval is not supplied here.

## 5. Host Interpreter Agentic Architecture

### 5.1 SectorLogo64

SectorLogo64 is a small, Lisp like scripting language implemented in assembly inside the v7 DLL. It provides:

  Integer arithmetic and 30 bit truncated IEEE 754 floats.
  String operations (find, replace, substring, explode, concat).
  List operations (cons, car, cdr, sort, keyed binary search lookup).
  Process execution (`v7_execute`, `DefineCommand`): spawn a child process, capture its stdout.
  Access to the MicroDB hash table (`DatabaseGet`, `DatabaseSet`).
  Stream manipulation (`GetStream`, `SetStream`).
  Per token hooks triggered by specific output tokens or strings.

The language is intentionally minimal; it is a general-purpose Logo interpreter with string
processing, lists, arithmetic, control flow, database operations, and native FFI. Inference-specific
functions are provided by the V7 host interface rather than defining the language itself.

### 5.2 Design Rationale

SectorLogo is a general-purpose Logo interpreter embedded in the DLL. For scripts that use native host
commands, communication with the local inference engine does not require IPC or a separate interpreter
process. This property applies to the core DLL; optional internet retrieval, model export, and
persistent-memory workflows use external tools and file operations.

The trade off is that the language has sharp edges (documented in the project's development notes):
silent failures on some syntax errors, and evaluation semantics that occasionally surprise; these are
being documented and fixed incrementally.

### 5.3 Comparison with Structured Function Calling

A function-calling application can use a model to produce a structured request, then validate the
selected operation and arguments. Grammar-constrained decoding can enforce valid JSON syntax, so
malformed JSON is not an inherent limitation of function calling. V7 provides a different interface: a
SectorLogo hook can respond to token events or modify the generated stream without requiring the model
to emit a tool request. The script can also perform text processing and retrieval directly. This is a
difference in control placement, not evidence that the V7 approach replaces every form of tool use.

## 6. Episodic Memory via Self Distillation

### 6.1 Architecture

The memory system has two phases:

**Storage (end of turn).** When the model finishes a response (EOS token), a SectorLogo64 hook extracts
the full conversation (user prompt plus model response) and runs a second inference pass with a
distillation prompt, asking the model to compress the conversation into a small number of tagged factual
statements (USER / SELF / NOTE). Statements about the user are appended to one plain text file
(`user_facts.txt`); general notes go to another (`ragg_base.txt`). Statements must pass a source-grounding check: a distilled fact is stored only if its content words occur in the
turn text from which it was distilled. This check filters out template echoes and unsupported lines,
although token overlap alone cannot guarantee factual correctness.

**Retrieval (next turn).** At the start of the next turn's thinking phase, the accumulated thinking text
is used as a BM25 query against the memory corpus. The top k matching lines are retrieved and injected
into the context via the stream rewriting mechanism.

### 6.2 Properties

  **Transparency.** The memory corpus is a human-readable, editable text file. Users can inspect, correct, or delete memories directly.
  **No embedding model.** Retrieval is purely lexical (BM25), avoiding the cost and latency of embedding computation.
  **Recency-aware retrieval.** Later records can receive a recency-weighting advantage. This does not establish that an earlier incorrect statement is corrected reliably: the generated summary may be inaccurate, both versions may remain retrievable, and the requested correction may not be selected.
  **Bounded cost.** The distillation pass adds one extra generation per turn. For the 0.6B model with a 120 token distillation cap, this is one additional short generation.

### 6.3 Limitations

Lexical BM25 retrieval is sensitive to vocabulary mismatch: if the user's query uses different
terminology than the stored memories, retrieval fails. Using the model's own thinking text as the query
partially mitigates this (the thinking text tends to normalize terminology), but the limitation remains.
The pure-SectorLogo reference BM25 implementation performs approximately 10–20 interpreter dispatches
per character in the documented configuration and is therefore intended for relatively small text
collections. Other repository retrieval paths call native or external C++ BM25 tooling, so this cost
must not be attributed to every V7 retrieval mode. The plaintext memory files do not have a documented
automatic consolidation or size-management policy; retrieval time, storage requirements, and the risk of
retaining outdated personal information increase as they grow.

## 7. Internet Augmented Retrieval

### 7.1 Architecture

The internet RAG extension adds a web search step to the stream rewriting pipeline. When the thinking
close hook executes, the system:

1. Passes the user prompt and thinking text to a `web_makequery` function, which distills them into a concise search query using a second LLM inference pass.
2. Executes the search through an external tool (`web_search`, which spawns `web_ragg.exe`), fetching up to 5 pages and distilling each into a short paragraph with the LLM.
3. Injects the distilled web content into the context alongside any locally retrieved memories.

### 7.2 Design Constraints

The internet RAG system is built to degrade gracefully: if the network is unavailable or the search
returns nothing, the system falls back to local memory or base knowledge without interrupting
generation. The development notes treat failures (bad distilled queries, thin page parses, hallucinated
summaries) as expected events that must be logged for iterative tuning.

The stated goal in the project documentation is to give the model awareness of current events and recent
releases, compensating for its training cutoff. How well that works depends heavily on the quality of
the query distillation and page parsing steps, which remain under active tuning.

## 8. Evaluation

### 8.1 Kernel Throughput and Measurement Scope

The recorded kernel microbenchmarks were run on a Ryzen 5 7430U with 12 logical processors. For a 1024 ×
1024 Q8 matrix, the reported effective throughput is 382–426 GB/s, compared with 295–336 GB/s for 1024 ×
2048. The corresponding F32 figures are 176–197 GB/s and 136–161 GB/s. These values are *effective bytes
processed per second by the benchmark*, not measured transfers from DRAM.

A 1024 × 1024 Q8 weight matrix occupies roughly 1 MiB plus group scales and fits within the processor's
16 MiB shared L3 cache. The effective throughput can consequently exceed physical DRAM bandwidth through
cache reuse. AMD specifies two memory channels and supports DDR4-3200 or LPDDR4X-4267 on this processor;
theoretical dual-channel DDR4-3200 transfer bandwidth is 51.2 GB/s, not hundreds of GB/s. Larger models
whose weights exceed the last-level cache have a different access pattern. Single-token decoder
throughput may be limited by DRAM traffic, but these small-matrix measurements do not establish its
achieved bandwidth.

The separate [CPU inference benchmark](engine_benchmark.md) reports same-machine decode rates
for V7, llama.cpp, MNN, and OpenVINO. Each engine uses its own weight format. V7's Q8 groups
contain 64 values, and llama.cpp's `Q8_0` blocks contain 32. The benchmark documents the
workloads and configurations used for that comparison.

### 8.2 Model Support

The engine has been tested with 10 LLMs (Qwen3 0.6B, Qwen3 1.7B, Qwen3 8B, Qwen2.5 0.5B, Llama 3.2 1B,
Phi 3 mini 4k, SmolLM2 135M, SmolLM2 360M, TinyLlama 1.1B, MiniCPM5 1B) and 3 VLMs (Qwen3 VL 4B, LLaVA
OneVision Qwen2 0.5B, SmolVLM 256M). All produce coherent output on both F32 and Q8 paths. Because model
specific values are data rather than code, other models from the Llama, Qwen and DeepSeek families are
plausible candidates that need no code change, but each one still requires its own database build and a
validator run before its results can be described as working.

### 8.3 Validation

The bundled numerical validators use project reference implementations for several kernels; passing them
does not establish identical outputs to Hugging Face or llama.cpp across all checkpoints. An independent
accuracy study would compare logits, selected-token agreement, and perplexity under explicitly matched
precision and quantization formats. No such results are asserted here.

The project has a validation suite (`verify.bat`) that builds the engine, then runs 21 validators
covering kernel correctness (matmul, attention, RMSNorm, RoPE, SwiGLU, quantization, memcpy, MicroDB,
execution, file I/O, string formatting, sampler, SectorLogo, ViT linear), dispatch configuration, and
end-to-end inference smoke tests in both F32 and Q8 modes. The sampler validator gates all 40 cases
bit-exactly against a 1:1 C port of the assembly sampler. The suite aborts on the first failure and is
run before every release.

### 8.4 RAG Quality

A controlled RAG comparison of no retrieval, pre-generation retrieval, and mid-stream rewriting on the
same synthetic or otherwise unfamiliar fact corpus has not been conducted. Formal evaluation on standard
benchmarks (such as Natural Questions and TriviaQA) has also not been conducted. Internal testing uses
qualitative assessment on small corpora a 60-line fact base and a 4 MB Bible corpus (about 31,000
sentence level chunks). The development notes document specific failure modes: query poisoning when the
model's prior knowledge conflicts with the retrieved text, oscillation on binary questions, and per
corpus floor calibration requirements (a BM25 score floor calibrated on one corpus does not transfer to
another).

## 9. Limitations

**Retrieval provenance and untrusted text.** Some demonstration scripts insert retrieved material into
the generated reasoning without a source boundary that the model can reliably distinguish from its own
text. Web pages may contain instructions unrelated to the user's query; their inclusion can alter the
response. Applications using untrusted retrieval must establish source provenance, treat retrieved text
as data rather than instructions, and validate actions outside the model. The examples do not provide an
end-to-end prompt-injection defense or automatic citations.

**Memory retention and accuracy.** The plaintext memory examples store user-derived records in files
such as `user_facts.txt` and do not specify a universal limit, automatic pruning, or a verified
consolidation policy. Recency weighting does not prove that contradictory or inaccurate entries are
consistently resolved. Applications processing personal data must determine appropriate storage,
retention, and access controls.

**Rewrite capacities.** The internal re-encoding buffer holds 16,384 token IDs and the decoded stream
has separate fixed limits. A replacement that reaches the encoder's capacity is not silently truncated;
the current rewrite implementation enters its fatal-error path. Rewriting also consumes generation
budget and may require additional token-by-token forward computation.

**Measurement coverage.** The kernel measurements characterize individual operations. The separate
[CPU inference benchmark](engine_benchmark.md) compares full-model decode rates. Open evaluation
questions include formal output comparisons, controlled RAG quality experiments, the independent
effects of instruction patching and hand-written assembly, and sustained thermal measurements
with spinning and sleeping workers.

  **Platform restriction.** The engine runs only on x86-64 Windows with AVX2.
  Porting to Linux, macOS or ARM would mean rewriting the assembly and the
  threading layer.
  **Development velocity.** Pure assembly development is slow and error prone.
  The project's development notes record numerous subtle bugs, including stack
  alignment violations, orphaned `add rsp` instructions left after debug prints
  were removed, and missing register saves, which are hard to catch without
  substantial validation infrastructure.
  **Model size.** The engine has been tested primarily with models up to 4B
  parameters. Larger models, at 70B and above, would need substantial work on
  memory management, on multi GPU support, which is absent, and on distributed
  dispatch.
  **RAG quality.** Lexical BM25 retrieval is weaker than dense retrieval for
  queries that are semantically similar but lexically different. The internet
  RAG system is in an early tuning stage and has not been benchmarked.
  **Security.** Self modifying code and writable executable pages conflict with
  modern security policies such as W^X and CFG. The engine is intended for
  trusted, single user environments.
  **SectorLogo64 maturity.** The scripting language has non obvious behaviors and
  silent failure modes that are being documented and fixed incrementally. It is
  not yet suitable for developers unfamiliar with the project.

## 10. Conclusion

V7 demonstrates a native x64 LLM/VLM inference implementation with programmable token interception,
stream revision, local retrieval, and optional external retrieval. The reported kernel measurements
describe specific cache and execution conditions; they do not establish an end-to-end performance
ranking against other inference libraries.

The stream rewriting mechanism, covering KV cache rollback, context augmentation and re-prefill during
the chain of thought phase, is a model-agnostic alternative to fine-tuned retrieval tokens. The host
interpreter architecture offers an alternative scripting-based control interface; grammar-constrained
structured decoding can also provide valid tool-call syntax. The plaintext memory example retains
records in an inspectable format but does not provide guaranteed factual consolidation, retrieval
correctness, or long-term retention management.

These contributions come with real trade-offs in portability, development velocity and security. We
present v7 not as a replacement for existing inference frameworks, but as an exploration of the design
space at the intersection of low level systems programming and agentic AI. We hope that releasing the
source code lets others examine, critique and build on these ideas.

## References

[1] W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention,"
SOSP, 2023. [2] NVIDIA, "TensorRT LLM," 2023. https://github.com/NVIDIA/TensorRT-LLM [3] G. Gerganov,
"llama.cpp," 2023. https://github.com/ggerganov/llama.cpp [4] Microsoft, "ONNX Runtime," 2023.
https://onnxruntime.ai [5] P. Lewis et al., "Retrieval Augmented Generation for Knowledge Intensive NLP
Tasks," NeurIPS, 2020. [6] A. Asai et al., "Self RAG: Learning to Retrieve, Generate, and Critique
through Self Reflection," ICLR, 2024. [7] S. Yan et al., "Corrective Retrieval Augmented Generation,"
arXiv:2401.15884, 2024. [8] H. Chase, "LangChain," 2022. https://github.com/langchain-ai/langchain [9]
Significant Gravitas, "AutoGPT," 2023. https://github.com/Significant Gravitas/AutoGPT [10] OpenAI,
"Function Calling," 2023. https://platform.openai.com/docs/guides/function-calling [11] Z. Jiang et al.,
"Active Retrieval Augmented Generation," EMNLP, 2023. arXiv:2305.06983 [12] H. Trivedi et al.,
"Interleaving Retrieval with Chain of Thought Reasoning for Knowledge Intensive Multi Step Questions,"
ACL, 2023. arXiv:2212.10509 [13] X. Li et al., "Search-o1: Agentic Search-Enhanced Large Reasoning
Models," EMNLP, 2025. https://aclanthology.org/2025.emnlp-main.276/ [14] D. Lee, O. Queen, J. Zou,
"ReasonOps: Operator Segmentation for LLM Reasoning Traces," arXiv:2605.29192, 2026. [15] J. Lee et al.,
"ReasoningFlow: Discourse Structures for Understanding LLM Reasoning Traces," arXiv:2606.05402, 2026.
[16] J. Hu et al., "Reflection Steering: Disentangling Reflection from Reasoning in Activation Space for
Token Efficient Inference," arXiv:2608.25542, 2026. [17] R. Kamoi et al., "When Can LLMs Actually
Correct Their Own Mistakes? A Critical Survey of Self Correction of LLMs," TACL 12:1417 1440, 2024.
arXiv:2406.01297 [18] M. Chen, B. Qu, L. Wang, "The Calibration Floor: Format Repair Can Masquerade as
Self Correction at Small to Mid Scale," arXiv:2608.04355, 2026. [19] H. Xu et al., "SafeSpec: Fast and
Safe LLM via Dynamic Reflective Sampling," arXiv:2606.19755, 2026. [20] C. Ke et al., "QiMeng PRepair:
Precise Code Repair via Edit Aware Reward Optimization," ACL, 2026. arXiv:2604.05963 [21] X. Mou et al.,
"Beyond Block Boundaries: Multi Block Editing for Diffusion Large Language Models," arXiv:2607.22663,
2026. [22] L. Su, "Execution State Capsules: Graph Bound Execution State Checkpoint and Restore for Low
Latency, Small Batch, On Device Physical AI Serving," arXiv:2606.20537, 2026. [23] S. Yao et al.,
"ReAct: Synergizing Reasoning and Acting in Language Models," ICLR, 2023.
https://arxiv.org/abs/2210.03629 [24] T. Schick et al., "Toolformer: Language Models Can Teach
Themselves to Use Tools," NeurIPS, 2023. https://arxiv.org/abs/2302.04761 [25] S. Robertson and H.
Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," Foundations and Trends in
Information Retrieval, 2009. https://doi.org/10.1561/1500000019 [26] L. Zheng et al., "SGLang: Efficient
Execution of Structured Language Model Programs," NeurIPS, 2024.
https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html

Source code and documentation: consult the project release repository and its README for the canonical
URL (GPL v3).


## Appendix A: Assembly Walkthroughs

Code level detail for three mechanisms described above, quoted from the engine source in `src/asm/`.
Label references point at the actual code.

### A.1 Presence Penalty Rollback During Stream Rewrite

The presence penalty suppresses repetition: every token the model has already emitted is marked in a
bitmap, and before each sampling step the logits of all marked tokens are reduced by a constant:

$$\text{logit}_i \leftarrow \text{logit}_i - \alpha \quad \forall i \text{ marked in bitmap}$$

(See `APPLY_PRESENCE_PENALTY` in `chat.inc`: it walks the bitmap with `tzcnt`/`blsr` and applies
`vsubss` to each marked logit.)

This creates a rollback problem: if tokens are rolled back in the KV cache but stay marked in the
bitmap, the sampler keeps penalizing tokens that are no longer in the context. On rewrite, v7 therefore
clears the bits of all discarded tokens and re sets the bits of the retained prefix tokens. Because the
x86 `btr`/`bts` instructions accept a memory operand and internally compute the dword offset and bit
index from the full token ID, no explicit `/ 32` and `% 32` arithmetic is needed (`chat.inc`,
`.chat_rewrite_btr_loop` and `.chat_remark_loop`):

```nasm
; Clear presence penalty bits for discarded tokens (chat.inc)
; Discarded tokens = history[T_diverge.. old_tok_count 1]
.chat_rewrite_btr_loop:
        cmp     r8d, r12d; index >= old token count?
        jae.chat_rewrite_btr_skip; done
        mov     edx, [r9 + r8*RW_DWORD_SCALE]; edx = discarded token ID
        btr     dword [r10], edx; clear bit (btr computes offset+bit itself)
        inc     r8d; next discarded token
        jmp.chat_rewrite_btr_loop; continue

; Re mark retained prefix tokens (chat.inc)
; A token occurring in BOTH the kept prefix and the discarded suffix had its
; bit cleared above, but the prefix occurrence must stay penalized.
.chat_remark_loop:
        cmp     r8d, eax; index >= T_diverge (kept prefix end)?
        jae.chat_remark_done; all kept tokens re marked
        mov     edx, [r9 + r8*RW_DWORD_SCALE]; edx = kept token ID
        bts     dword [r10], edx; set bit (bts computes offset+bit itself)
        inc     r8d; next kept token
        jmp.chat_remark_loop; continue
.chat_remark_done:
```

`r9` is the token history base, `r10` the bitmap base; `r12d` holds the old token count and `eax` the
divergence index.

### A.2 Worker Job Acquisition

Parallelism is built around a shared-counter batch dispatcher (`DISPATCH_EXECUTE` in
`dispatch_execute.inc`, worker loop in `v7_worker.inc`). Workers claim jobs in batches: a non atomic
peek at `next_job` short circuits the expensive `lock xadd` when no work remains, and each processed
batch decrements `remaining_jobs` once instead of per job:

```nasm
; Worker steal loop (v7_worker.inc,.steal_loop)    abbreviated
        mov     eax, [next_job]; non atomic peek
        cmp     eax, r14d; next_job >= num_jobs?
        jae.steal_done; no work remains    skip the lock xadd

        mov     eax, edi; batch size (from SELECT_BATCH / INI config)
        lock xadd dword [next_job], eax; eax = old next_job (first index of my batch)

        cmp     eax, r14d; start >= num_jobs? (someone else took the tail)
        jae.steal_done; no more jobs

;... process jobs [start, min(start+batch, num_jobs)) via process_job...
        inc     wp_count; count jobs in this batch (one lock sub later)

; Batch completion tracking (v7_worker.inc,.steal_done)
        mov     eax, wp_count; jobs processed by THIS worker in this dispatch
        lock sub dword [remaining_jobs], eax; one atomic decrement per batch
```

The caller thread runs the same claim loop (see `dispatch_execute.inc` lines 92 124) and then pause
spins on `remaining_jobs == 0` until all workers finish.

### A.3 Kernel Inventory

This section gives the real code shapes of the main kernels, verified against `src/asm/`.

#### Q8xQ8 dot product inner loop (`q8_kernel.inc`)

Weights are stored in blocks of 64 signed int8 values, each block with one fp32 scale factor. The dot
product uses unsigned activation magnitudes and sign-adjusted signed weights before the `vpmaddubsw` and
`vpmaddwd` chain. This listing is a representative excerpt of the actual `COMPUTE_Q8_ROW` macro in
`src/asm/v7_macros.inc`; it includes the signedness conversion necessary for correctness:

```nasm
        vmovdqu     ymm10, [rbx + rcx]   ; signed activation bytes
        vpabsb      ymm4, ymm10          ; unsigned magnitudes, 0..128
        vmovdqu     ymm6, [rdi + rcx]   ; signed weight bytes, -127..127
        vpsignb     ymm6, ymm6, ymm10   ; transfer activation signs to weights
        vpmaddubsw  ymm6, ymm4, ymm6    ; unsigned-by-signed pairs to int16
        vpmaddwd    ymm6, ymm6, ymm9    ; sum pairs into int32, ymm9 = int16 ones
        ; The complete kernel converts these sums to float and applies scales.
```

The macro converts the 32-bit partial sums to floating point and combines the activation and weight
scales for each 64-value group before accumulation. Weight magnitudes are limited to 127, so the maximum
magnitude of a two-product int16 pair is `2 × 128 × 127 = 32,512`, below the saturation threshold. A
generic signed-int8 implementation cannot omit this bound.

#### RMSNorm (`rmsnorm_kernel.inc`)

$$\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{N}\sum_j x_j^2 + \epsilon}} \cdot \gamma_i$$

Phase 1 accumulates the sum of squares with two FMA accumulators (`vfmadd231ps ymm0, ymm2, ymm2` on two
32 byte blocks per iteration), then reduces horizontally. Phase 2 computes the scale with a single
rounding division (`vdivss` by N, matching the C reference bit for bit), an exact `vsqrtss`, an exact
reciprocal `vdivss`, and a `vbroadcastss`; then normalizes in place as `(scale*x) * weight` with a
folded load per block. Outputs are clamped to approximately ±1e17 to prevent downstream dot product
overflow.

#### SwiGLU (`swiglu_kernel.inc`)

$$\text{SwiGLU}(x) = \text{gate}(x) \cdot \sigma(\text{gate}(x)) \odot \text{up}(x)$$

The sigmoid $\sigma(z) = 1/(1+e^{-z})$ is computed by the shared `EVAL_SIGMOID` macro
(`macros/v7_macros.inc`) as a 2^x decomposition: negate via sign bit XOR, clamp to approximately ±80,
split into integer and fractional parts of $z \cdot \log_2 e$, evaluate a 4th order minimax polynomial
on the fraction, and reconstruct $2^{\text{ipart}}$ by shifting the biased exponent into a float. No
library call, no x87 transcendental. The macro evaluates scalar (`ss`), 4 wide (`ps`), and 8 wide YMM
forms; the SwiGLU kernel also fuses two output rows into one 2 wide sigmoid pass.

#### Fused QKV + RMSNorm + RoPE (`fused_f32_kernel.inc`)

`v7d_fused_f32` computes the Q or K projection (matmul), applies the model's QK norm weight, and applies
RoPE    all inside one function, so the projected activations never leave L1 cache. The RoPE tables are
built lazily on first use by an internal x87 routine (up to 131072 positions), or supplied by the caller
via `v7s_set_fused_rope`. The multimodal variant (Qwen3 VL) is handled without kernel changes:
`mrope.inc` (`v7s_mrope_prepare` / `v7s_mrope_restore`) builds a per sequence index scratch cos/sin
table whose rows gather the temporal/height/width components per RoPE pair index, swaps the global table
pointers for the prefill, and adds a position delta during generation.

#### Vision pipeline (`vit_forward.inc`, `vit_kernel.inc`, `vit_pixel_shuffle.inc`, `vit_bicubic.inc`)

  `v7_chat_image` (public tier; internally `v7s_chat_image_mem`) takes a raw in memory RGB pixel array    no file I/O, no image library.
  `vit_bicubic.inc` / the removed bilinear variant rescale the image to the model's patch grid (Catmull Rom bicubic for LLaVA OneVision, matching PIL's BICUBIC).
  Projectors: LLaVA/SigLIP uses a two layer GELU MLP; SmolVLM/Idefics3 uses a vectorized pixel shuffle (spatial to channel rearrangement with AVX2 byte permute instructions).

Every model dependent value in the ViT path (dims, strides, offsets, patch size, grid, DeepStack taps)
is precomputed by `db_build_tool.c` and stored in the `__sys_vision_config` database entry; the assembly
reads named fields with single RIP relative memory operands, so supporting a new vision model requires
no assembly changes    only a regenerated database.
