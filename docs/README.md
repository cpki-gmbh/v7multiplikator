# V7 documentation

These manuals explain how to build V7, integrate its native API, run supported models, and use its
scripting and retrieval facilities. If this is your first checkout, begin with the installation guide
and run the simplest text example before exploring the advanced interfaces. If you already have a
working model database, the API reference is a more direct starting point.

The guides distinguish the operations available to an application from the implementation details used
to provide them. For example, stream rewriting is an application-facing capability, while byte-to-token
mappings and KV-cache reconstruction belong to the implementation reference. The chapters are organized
around those responsibilities so that you can find an API contract without first reading the assembly
details.

## Start here

1. **[Installation and examples](getting_started.md)** — Prepare the toolchain (Windows or Linux), build the engine library, create a model database, run sample applications, and verify the installation.
2. **[Native API and HTTP integration](native_api.md)** — Integrate V7 with a C or C++ application, manage generation and session state, and understand the error contract.
3. **[Models and vision-language inference](models_and_vision.md)** — Select a supported text or vision-language model and understand its configuration requirements.

## Programming and retrieval

4. **[Generation control and stream rewriting](generation.md)** — Inspect candidate tokens, control sampling, replace generated text, and understand how the KV cache is updated.
5. **[SectorLogo language and host integration](sectorlogo.md)** — Write general-purpose Logo programs, use native text processing, and invoke host functions through FFI.
6. **[Text search and retrieval](retrieval.md)** — Use MicroDB, BM25, and SectorLogo for independent search or generation-time retrieval.

## Implementation reference

7. **[System architecture](architecture.md)** — Understand the native execution pipeline, deployment constraints, model storage, and implementation trade-offs.
8. **[MicroDB and binary layouts](storage_and_layouts.md)** — Find database key definitions, structure layouts, and exact field offsets.
9. **[Dispatch and runtime tuning](tuning.md)** — Configure the work dispatcher, tune execution parameters, and interpret benchmark results.

## Reading the performance and retrieval claims

The [CPU inference benchmarks](engine_benchmark.md) compare four text models across v7, llama.cpp,
MNN, and OpenVINO. The [HTML overview](engine_benchmark.html) charts the decode rates for short chat
and long generation. The comparison identifies the run behind each result and lists the earlier
ONNX Runtime GenAI int4 results separately.

The [vision timing table](models_and_vision.md#recorded-vision-turn-latency) reports the
2.35.0 Qwen3-VL turn latency, including image encoding and answer generation.

The [runtime tuning guide](tuning.md) reports effective GB/s for individual kernels, including
microbenchmarks that reuse cache-resident weights. The [retrieval manual](retrieval.md) and the
[research paper](paper1.md) describe the workloads used to measure retrieval performance.

## Source of truth and supporting material

The public C declarations are defined in `src/v7_lib.h`, and the DLL exports are declared in
`src/v7_threading.def`. These manuals describe the implementation as documented at the time of the
archive. Historical measurements and proposed changes are identified separately
from the public interface contract. If a description and the current implementation differ, check the
specific export or code path before changing an application.

Planning files, the development handbook, research records, and historical measurements are retained
separately and are not part of the public API contract. The repository's `CHANGELOG.md` contains
dated release information and is included at the repository root. A model's original license remains
applicable to its checkpoint.
