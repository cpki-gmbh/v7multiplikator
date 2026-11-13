# Installation, Examples, and Diagnostics

This guide describes how to build and verify V7 from a source checkout. It covers the build
environment on Windows and Linux, model database preparation, native inference examples, and the
tools used to verify an installation. The later sections describe optional utilities and give
practical steps for diagnosing common problems.

Unless a command says otherwise, run it from the repository root (a Windows command prompt for the
`scripts\*.bat` drivers, a shell for the `make` / `src/linux` path). Model databases are separate
from the engine library (`v7_threading.dll` on Windows, `libv7_threading.so` on Linux), so prepare
the database for the model you intend to run before starting an inference example. If you are
integrating V7 into an existing C or C++ application, begin with the build and a working example,
then refer to the native API manual for the application lifecycle.

## Contents

- [Installation, building, and model preparation](#installation-building-and-model-preparation)
- [Example applications](#example-applications)
- [Tools, validators, and benchmarks](#tools-validators-and-benchmarks)
- [Troubleshooting](#troubleshooting)

---

## Installation, building, and model preparation

For an application that uses a prepared model, the essential runtime files are the V7 engine
library and a compatible MicroDB database. The additional tools described here are needed to build
the library, prepare new model databases, or run development checks. It is useful to keep those
tasks separate: first establish that the native library builds, then prepare the resources for a
supported model, and finally run an inference example. If something does not work, this sequence
helps identify whether the problem belongs to the toolchain, the model data, or the application
interface.

### Prerequisites

Building V7 requires an x86-64 development environment with a C++ toolchain and NASM: Windows uses
Visual Studio 2022 or newer plus the `scripts\build.bat` driver; Linux uses gcc/g++, `make`, and
the `Makefile` driver (see the README's Linux section). Preparing a model database also requires
Python, but Python is not needed to execute an already prepared model. The optional dependencies
described below are used only for particular development tools.

The DLL checks its **full CPU baseline** at load: AVX2, FMA, F16C, BMI1/2,
SSE4.1/4.2, AES-NI, RDRAND and RDTSCP, plus OS-enabled SSE/YMM state.
AVX2 support alone does not qualify a machine. The AES rounds use the
legacy `aesenc` encoding (AES-NI, 2010+), so VAES-capable and AES-NI-only
CPUs run the same binary. See `src/asm/cpu_gate.inc` for the actual checks.

Install **Visual Studio 2022 or newer** with the C++ workload; this supplies `cl.exe` and `link.exe`.
The `build.bat` script normally locates Community, Professional, Enterprise, or Build Tools
installations through `vswhere`. If the computer has more than one Visual Studio installation, set
`VS_PATH` to the required `vcvarsall.bat` file. A typical location is:

```
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat
```

The assembler is **NASM 3.x** and a working `nasm.exe` ships in `tools\`,
so no download is needed. To use a different assembler, install it elsewhere
and set `NASM_DIR` to the directory containing the executable; a copy in
`tools\` takes precedence over other instances on `PATH`. Run `nasm -v` to
inspect the installed
version; `build.bat` also reports and checks the assembler that it selects.

Use **Python 3.10 or newer** when exporting checkpoints. The supported preparation tools require the
following packages:

```bat
pip install torch transformers safetensors huggingface_hub sentencepiece protobuf jinja2
```

**Go 1.21 or newer is optional.** The build needs it only if `src\nasmfmt\nasmfmt.exe` is missing, in
which case the assembly-comment formatter is compiled from the vendored source. A normal checkout
already contains the executable and does not need Go. If this fallback is required, check the
installation with `go version`.

**Zydis is optional and intended for development diagnostics.** A development tree may include
`Zydis.lib` in the repository root and the corresponding `zydis-master\` and `zycore-c-master\` source
trees. When those resources are available, `build.bat` enables instruction-level diagnostic output for
the tools. A release checkout can build against the no-op implementation in `tools\zydis_stub\` instead;
the tools still execute, but exception disassembly is unavailable. The repository's
`docs\zydis_debug_readme.txt` describes how to configure the full diagnostic integration.

### Building

```bat
build.bat
```

Running `build.bat` assembles the native modules and links the library, supporting utilities, and
example programs. The resulting files are written to the locations listed below. The build step does not
by itself verify a model database or complete an inference smoke test; use `verify.bat` for the standard
validation sequence after the initial build.

| Output | What |
| --- | --- |
| `bin\v7_threading.dll` | The inference engine |
| `bin\build_db.exe` | Database builder |
| `tools\*.exe`, `bin\*.exe` | Validators, benchmarks, tools |
| `bin\inference_example.exe`, `bin\dbg_image_recog.exe`,... | Examples |

The `build.bat` script performs compilation and linking only. Validation and inference smoke tests run
via `verify.bat`, which calls `build.bat` first (see [Verification](#verification) ).

#### Build environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `VS_PATH` | auto via vswhere | Full path to a specific `vcvarsall.bat` to use instead of the auto-detected Visual Studio. |
| `NASM_DIR` | `<repo>\tools` | Directory containing `nasm.exe`. Prepended to PATH, so the bundled `tools\nasm.exe` takes precedence over any installed NASM. |
| `TRACE_DEBUG` | unset | Set to `1` for a debug DLL build with the Zydis instruction trace and the INT3 debug macros. Leave it unset for production, because it floods stderr. |

```bat
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
```

### Models

Model checkpoints must be converted into V7's database format before inference. The supplied fetch
scripts perform the supported preparation steps; manual preparation is available for configurations
without a dedicated script. The model compatibility reference distinguishes tested configurations from
unverified combinations.

#### Model preparation

Before the DLL can run a model, its checkpoint must be converted to the binary database format
expected by V7. The preparation tools export the weights and model metadata, then build `lz4ht.db`
with the tokenizer, configuration, and templates; this keeps upstream checkpoint parsing outside the
inference runtime. The following commands show the normal conversion path and the alternatives
available when a dedicated fetch script is not provided.

```
HuggingFace model
  -> fetch_<model>.bat (or the manual steps below)
    -> export.py          (HF ->.bin checkpoint + tokenizer + templates)
    -> build_db.exe       (.bin -> lz4ht.db microdb database)
  -> inference
```

#### Fetch scripts, one command per model

| Script | Model | Params | HF Repo | DB Build Flags |
| --- | --- | --- | --- | --- |
| `fetch_qwen3_0.6b.bat` | Qwen3-0.6B | 0.6B | `Qwen/Qwen3-0.6B` | defaults |
| `fetch_qwen3_1.7b.bat` | Qwen3-1.7B | 1.7B | `Qwen/Qwen3-1.7B` | defaults |
| `fetch_qwen3_8b.bat` | Qwen3-8B | 8.0B | `Qwen/Qwen3-8B` | defaults |
| `fetch_llama_3.2_1b.bat` | Llama-3.2-1B-Instruct | 1.0B | `unsloth/Llama-3.2-1B-Instruct` | `-thinking 0`* |
| `fetch_phi3_mini.bat` | Phi-3-mini-4k | 3.8B | `microsoft/Phi-3-mini-4k-instruct` | `-thinking 0* -rope-base 10000` |
| `fetch_smollm2_135m.bat` | SmolLM2-135M | 0.135B | `HuggingFaceTB/SmolLM2-135M-Instruct` | `-thinking 1* -rope-base 100000` |
| `fetch_smollm2_360m.bat` | SmolLM2-360M | 0.36B | `HuggingFaceTB/SmolLM2-360M-Instruct` | `-thinking 1* -rope-base 100000` |
| `fetch_tinyllama.bat` | TinyLlama-1.1B-Chat | 1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `-thinking 0* -rope-base 10000` |
| `fetch_minicpm5_1b.bat` | MiniCPM5-1B | 1.0B | `openbmb/MiniCPM5-1B` | `-thinking 0* -rope-base 5000000` |
| `fetch_qwen2.5_0.5b.bat` | Qwen2.5-0.5B-Instruct | 0.5B | `Qwen/Qwen2.5-0.5B-Instruct` | see the script |
| `fetch_smolvlm_256m.bat` | SmolVLM-256M | 0.25B text + SigLIP vision | `HuggingFaceTB/SmolVLM-256M-Instruct` | vision, see below |
| manual setup | Qwen3-VL-4B-Instruct | 4.0B | `Qwen/Qwen3-VL-4B-Instruct` | vision, see below |
| `fetch_llava_onevision_qwen2_0.5b.bat` | LLaVA-OneVision-Qwen2-0.5B | 0.5B text + SigLIP vision | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | vision, see below |

\* `-thinking` is a no-op in build_db.exe, kept for script compatibility. All DBs store the
base non-thinking template plus optional `@think` variants; whether a model thinks is chosen at runtime
via `v7_enable_thinking()` in the DLL, where the engine default is on.

Each script downloads from HuggingFace into a cached directory under `weights\<model>\`, exports through
`tools\export.py` or `export_vlm.py` for VLMs, and builds `weights\databases\<model>\lz4ht.db` through
`bin\build_db.exe`.

The [model support matrix](models_and_vision.md#model-support-and-configuration) gives the dimensions,
layer counts, attention-head configurations, vocabulary sizes, RoPE settings, and tokenizer families for
the documented models.

#### Vision models

The verified vision-language configurations include a vision encoder and a projection stage in addition
to the text decoder. V7 exposes `v7_chat_image` and `v7_generate_image` for image-capable models, while
the database supplies the necessary vision configuration and weights. Image preparation differs between
model families; use the model reference for the applicable input resolution, preprocessing, and memory
requirements.

**Qwen3-VL-4B**, which has no fetch script and needs manual setup. The export produces a text checkpoint
plus a separate `.vision` blob. `export_vlm.py` unwraps the `model.language_model.*` container and emits
the vision tower through the dev-only `v7_vlm_ref\export_vision.py` module: that module is not part of
the release package (like the dev-only SectorLogo C++ port, it exists in the dev tree only, restore it
from git history if it is missing; `export_vlm.py` fails with a named FATAL reporting that requirement
instead of a bare import error):

```
HuggingFace (Qwen/Qwen3-VL-4B-Instruct)
  -> snapshot_download into weights\qwen3_vl_4b\
  -> tools\export_vlm.py                 (.bin +.tokenizer +.template* +.vision)
  -> build_db.exe -vision <.vision>
```

When exporting, use `export_vlm.py --max-seq-len N` to cap KV sizing. The default is 8192, because the
model's real 262144 context would need roughly 36 GiB (38.7 GB) of FP16 KV cache. The vision config is a ViT depth of
24, a hidden size of 1024, 16 heads at head_dim 64, a patch size of 16, a spatial merge of 2, and
DeepStack injection from ViT layers 5, 11 and 17 into text decoder layers 0, 1 and 2. Normalization uses
a mean and standard deviation of 0.5, image dimensions are multiples of 32, and bounding boxes are in 0
to 1000 relative coordinates.

**LLaVA-OneVision**, Qwen2-0.5B text decoder (QKV biases handled by the fused bias path) + SigLIP tower
+ MLP projector. The projector ships as a second file and needs `-visionproj`:

```
HuggingFace (llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
  -> fetch_llava_onevision_qwen2_0.5b.bat
  -> tools\export_vlm.py                 (.bin +... +.vision +.proj)
  -> build_db.exe -vision <.vision> -visionproj <.proj>
```

This model represents bounding-box coordinates as normalized values between 0 and 1.

**SmolVLM-256M** uses a SigLIP vision encoder and a pixel-shuffle projector. The
`fetch_smolvlm_256m.bat` script prepares the corresponding model and vision data files.

#### Manual setup when there is no fetch script

```bat
:: 1. Download from HuggingFace (into weights\<model>\, the only supported layout)
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir=r'weights\my_model')"

:: 2. Export (output path first, HF dir second)
python tools\export.py weights\my_model\my_model.bin weights\my_model

:: 3. Build the database
bin\build_db.exe weights\my_model\my_model.bin -o weights\databases\my_model\lz4ht.db -f
```

#### build_db.exe reference

```
build_db.exe <checkpoint.bin> [options]
```

| Flag | Default | Description |
| --- | --- | --- |
| `-f` | off | Force rebuild, overwriting an existing database |
| `-thinking <0\|1>` | 1 | No-op, kept for script compatibility. All DBs store the base non-thinking template plus optional `@think` variants, and runtime selection goes through `v7_enable_thinking()` in the DLL. |
| `-rope-base <float>` | auto | Override the RoPE base, which is auto-detected from `config.rope_theta` |
| `-eps <float>` | header | Override the RMSNorm epsilon stored as `__sys_rms_eps` |
| `-rope-type <int>` | header | RoPE scaling type, where 0 is none and 1 is llama3 |
| `-rope-factor <float>` | header | llama3 division factor, e.g. `32.0` |
| `-rope-high-freq-factor <float>` | header | llama3 high-frequency boundary |
| `-rope-low-freq-factor <float>` | header | llama3 low-frequency boundary |
| `-rope-orig-max-pos <int>` | header | original_max_position_embeddings |
| `-vision <path>` | none | Vision tower from an `export_vlm.py` `.vision` file |
| `-visionproj <path>` | none | Projector blob, a `.proj` file, for MLP and pixel-shuffle VLMs |
| `-autoexec <path>` | none | SectorLogo `autoexec.logo` stdlib source to store under `logo:autoexec` |
| `-tuner-delay <n>` | 80 | Number of matching-shape dispatches to skip between tuner measurement phases; writes `tuner:delaydefault`. |
| `-tuner-delay-large <n>` | 1 | Initial tuner delay for shapes with a dimension above 16,384. |
| `-tuner-sex <0..100>` | 80 | Maximum percentage of genome slots supplied by the crossover parent. |
| `-tuner-mutate <0..100>` | 50 | Fraction of the 128-slot population that is replaced by mutated genomes; stored as a **slot count**. |
| `-tuner-mutations-max <1..104>` | 40 | Maximum number of genome fields changed by one mutation. |
| `-tuner-rand-delay <0..100000>` | 0 | Random delay-phase length jitter cap. |
| `-simple-threads <n>` | 0 | Set the unpinned simple-worker count; zero selects topology-aware placement. |
| `-o <path>` | `lz4ht.db` | Output database path |

```bat
build_db.exe weights\phi3_mini.bin -thinking 0 -rope-base 10000 -f
build_db.exe weights\my_model.bin -o weights\databases\my_model\lz4ht.db -f
```

#### Memory requirements for the export

| Model size | Min RAM | Notes |
| --- | --- | --- |
| under 1B | 4 GB | SmolLM2-135M, Qwen3-0.6B |
| 1 to 2B | 8 GB | Llama-3.2-1B, TinyLlama-1.1B, MiniCPM5-1B |
| 3 to 4B | 12 GB | Phi-3-mini, where preserving fp16 halves the peak |
| 7 to 8B | 24 GB | Qwen3-8B (export/build guidance; the support matrix marks its DB as not built in the recorded test tree) |

The exporter retains the checkpoint's original floating-point representation, including FP16 and BF16
values, rather than converting those values to FP32 during export.

### Running inference

With the DLL built and a compatible database prepared, run the minimal text-generation example first.
This checks the public library lifecycle and the model data together. For a vision-language model, use
the image-capable harness and a vision-enabled database. The examples below provide complete command
lines and describe the output expected from each program.

#### inference_example.exe, the minimal C example

```
inference_example.exe <dll_path> <db_path> [-q8 <0|1>] [-i "prompt"] [-n <tokens>] [-r <rounds>] [-think <0|1>] [-set-template "<fmt>"] [-rewrite] [-script <file>]
```

Both paths are required, with no defaults. `-script <file>` stores a sidecar in MicroDB under
`logo:sidecar`, and chat entry activates the callback from that key; `-rewrite` remains
accepted for compatibility but is not required. See the [SectorLogo
manual](sectorlogo.md#sectorlogo-language-and-integration).

Combining `-set-template "%s"` with `-think 0` enables raw completion: the engine passes the
prompt to the model verbatim, without chat framing. The [native API
reference](native_api.md#c-api-reference) explains the mechanism.

```bat
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "What is a cat?"
```

#### dbg_inference.exe, the pure-ASM harness

The example loads the inference DLL through `LoadLibraryA` and invokes the assembly inference pipeline
through the public C API. It is a normal build artifact, rebuilt by `build.bat` like everything else.

#### Vision harness

```bat
:: caption an image (tools\jpg_to_raw.py produces the.raw file +.meta sidecar)
python tools\jpg_to_raw.py cat.jpg cat
bin\dbg_image_recog.exe -q8 0 bin\v7_threading.dll weights\databases\qwen3_vl_4b\lz4ht.db cat640.raw "What is in this image?"

:: synthetic bounding-box evaluation (no image file needed)
bin\vit_bbox_eval.exe weights\databases\qwen3_vl_4b\lz4ht.db
```

### Verification

A passing bundled validator establishes the tested properties of that validator, but does not by itself
prove bit-for-bit equivalence with independent implementations. For comparative performance, record the
exact model export, quantization group size, batch and context settings, processor, memory
configuration, software revision, and the treatment of warm-up and timing interference. Benchmarks that
report the fastest observed runtime should state that rule explicitly rather than being described as
medians or averages.

```bat
verify.bat
```

Builds everything, runs all validators, and runs the inference smoke tests on the Q8, F32 and pure-ASM
paths. There are no accepted failures, every validator gates (the former known-limit list is resolved,
see the changelog). Text inference must answer "paris" for the capital-of-France probe. Budget roughly
15 minutes and ~6 GB of disk for the first run. Tool-by-tool details are in
[tools.md](getting_started.md#tools-validators-and-benchmarks).

**VC runtime note:** the tools, examples and server link the static CRT (/MT default via the build
flags); the larger C++ targets (`v7_server`, `llm_text_processor`, `db_build_tool`, `multi_turn_demo` )
use /MD and therefore need the VC++ redistributable installed, `vcvarsall.bat` environments already
provide it.

If a build or model-preparation step fails, consult the [troubleshooting section](#troubleshooting) for
the corresponding diagnostics and corrective actions.

## Example applications

Once the DLL and a model database are available, the simplest way to understand the public interface is
to run one of the small example applications. Begin with the C text-generation example to confirm that
the library loads and can produce a response. The multi-turn example then shows how a conversation
retains its state across calls, while the vision example demonstrates the additional image-processing
input. The assembly harness is more useful when examining the native integration itself.

Other examples illustrate SectorLogo scripts, retrieval applications, typed decisions, and tuning tools.
They are separate programs because each demonstrates a particular interface rather than imposing a
complete application framework. Unless a section says otherwise, the commands below can be run from the
repository root. The examples are grouped by purpose under `examples/`, with the following directories
providing the starting points for each area:

| Directory | Contents |
| --- | --- |
| `examples/c_inference/` | inference_example, dbg_inference, multi_turn_demo, dbg_image_recog, vit_bbox_eval, v7_pipe, v7_decide_example, and the small test programs (chat_only, generate_only, multiturn, multiturn_manual, kv_cache, engine_death, poem_continuation, image_multiturn, ardmark test drivers) |
| `examples/spam_classifier/` | spam_classifier + the ham/spam test mails in `mails\` |
| `examples/decision_Results.md` | measured typed-decision results and analysis (a file, not a directory) |
| `examples/server/` | v7_server |
| `examples/sectorlogo/` | run_sectorlogo, c_callback_example, echo_args, test_v7_execute_fmt, console_demo.logo, the bm25 benchmark templates |
| `examples/rag/` | bm25_rag, llm_text_processor, bayes_classify, web_ragg/, memory/ |
| `examples/ardmark/` | the Ardmark RAG demo (prime demo, see below) |
| `examples/research/` | archived sidecar strategy experiments (herbivorous_cats, helion_pump) |

Each directory has its own `README.md` with quick build+run instructions. The source-oriented design
overview is in [v7_inovations.md](architecture.md#implementation-design). `build.bat` at the project
root builds everything into `bin\`, except two tools with their own dedicated build scripts:
`bm25_rag.exe` (`examples\rag\bm25_rag.cpp`, built to `examples\rag\bm25_rag.exe` ) and `web_ragg.exe`
(`examples\rag\web_ragg\build_web_ragg.bat`).

### inference_example.c, Minimal C Inference

A minimal C program that loads the DLL, opens the database, runs one chat inference, and prints the
result. (Source abridged, see `examples/c_inference/inference_example.c` for the full version with flag
parsing.)

#### Usage

```
inference_example.exe <dll_path> <db_path> [-q8 <0|1>] [-i "prompt"] [-n <tokens>] [-r <rounds>] [-think <0|1>] [-set-template "<fmt>"] [-rewrite] [-script <file>] [-tokdump] [-exec]
```

- `dll_path`, path to `v7_threading.dll` (required, no default)
- `db_path`, path to `lz4ht.db` (required, no default)
- `-q8 <0|1>`, select F32 (`0`, the default) or Q8 quantized execution (`1`) for `inference_example.exe`
- `-i`, prompt text (default: "What is the capital of france?")
- `-n`, generation token budget; `0` means unlimited (default: 0)
- `-r`, number of rounds (default: 1). Each round resets the RDTSC statistics
  block, runs one chat call, and prints a `[perf]` report (per-phase cycles/ms,
  prefill ms/tok, generation tok/s). Use `-r 3` or more and read the last round
  for steady-state numbers, round 1 includes cold-cache effects.
- `-think <0|1>`, select the thinking (default: 1) or non-thinking template variant
- `-set-template`, override `__sys_template` in-memory (test hook, no DB rebuild). Passing `"%s"` turns chat into **raw completion**: the render becomes a verbatim pass-through of the prompt, with no chat framing at all. Combine with `-think 0` so the base (overridden) template is selected rather than a `@think` variant. The override is memory-only (`persist=0`), the `lz4ht.db` file on disk is untouched:
  ```
:: raw completion, model sees "The quick brown fox" and nothing else
   bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -set-template "%s" -think 0 -i "The quick brown fox" -n 1000
  ```
- `-rewrite`, retained for command-line compatibility. The `logo:sidecar` key installed by `-script` activates per-token script control, including candidate inspection, resampling, and stream rewriting; see the [SectorLogo manual](sectorlogo.md#sectorlogo-language-and-integration).
- `-script <file>`, load a SectorLogo sidecar script from a file (stored into
  MicroDB as `logo:sidecar` before context creation). Example scripts live in
  `src\logo\`, e.g. `src\logo\sidecar_cats_to_dogs.logo` rewrites "cat" to
  "dog" in the generated stream at the end-think boundary.
- `-tokdump`, enable the token stream dump (chat-level token/position trace)
- `-exec`, enable `[exec]` spawn tracing for `DefineCommand` / `v7_execute_fmt` calls

#### Benchmarking

Run `bench_qwen.bat [rounds]` from the repository root to benchmark Qwen3-0.6B in both F32 and Q8
modes. The script reports tokenization, prefill, and decode statistics for each round. The
[generation statistics report](performance_report.md) explains the phase timings and includes
the August 2026 measurements. The [CPU inference benchmark](engine_benchmark.md) documents the
four-model engine comparison and lists the batch files for the short chat and poem workloads.

#### Build

The standard build produces `bin\inference_example.exe` along with the other examples. To compile this
particular example independently, use the following commands:

```bat
cl /nologo /O2 /arch:AVX2 /std:c11 /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /I"src" /I"zydis-master\include" /I"zycore-c-master\include" examples\c_inference\inference_example.c /Fe:inference_example.exe /link Zydis.lib /subsystem:console
```

#### Source

```c
#include "v7_lib.h"

int main(int argc, char** argv)
{
    if (argc < 3) return 1;              // usage: inference_example <dll_path> <db_path>

    v7_lib dll;
    v7_load(argv[1], &dll);
    dll.v7_init(argv[2]);                // open the database (BSS context engine)

    dll.v7_initialize_inference();       // create context inside the DLL (returns 1 = ready)

    char output[32768] = {0};
    dll.v7_chat("what is a cat?", 0, output, sizeof(output), 1);   // reset=1: fresh session
    printf("%s\n", output);

    dll.v7_free_inference();             // parameterless cleanup
    v7_unload(&dll);
}
```

#### Output

```
Loaded v7_threading.dll (12 workers)
Database loaded: lz4ht.db
Inference context created
Prompt: What is the capital of
The capital of France is Paris.
```

---

### dbg_inference.exe, Assembly Inference Example

This pure-NASM inference harness does not use C code in its hot path. Loads the DLL via `LoadLibraryA`,
resolves exports through `GetProcAddress`, calls the public `v7_*` lifecycle (`v7_init`,
`v7_initialize_inference`, `v7_chat`), runs chat, prints output and exits. The harness is
written entirely in x86-64 assembly.

#### Usage

```
dbg_inference.exe <dll_path> <db_path> ["prompt"] [-q8]
```

- `dll_path`, path to `v7_threading.dll` (required, no default)
- `db_path`, path to `lz4ht.db` (required, no default)
- `prompt`, text prompt (default: "What is the capital of France?")
- `-q8`, select the quantized forward path (default: F32)

The example follows the command-line argument conventions used by current MSVC runtimes, including
quotation handling and flag placement. It configures greedy sampling by setting the temperature to zero.

#### Build

The standard `build.bat` procedure assembles and links this example as part of its fourth build stage.
To compile the example separately, use the following commands:

```bat
nasm -f win64 -O0 -w-other -I"src\asm" examples\c_inference\dbg_inference.asm -o dbg_inference.obj
link /entry:WinMain /subsystem:console /NODEFAULTLIB dbg_inference.obj kernel32.lib user32.lib synchronization.lib
```

#### Features

- Loads `v7_threading.dll` via `LoadLibraryA` (same loader state as the C harnesses)
- Resolves all function pointers via `GetProcAddress`
- Stack-probed 32KB output buffer with guard page handling
- Greedy sampler (temperature = 0), Q8 mode, 200-token budget
- Output written to stdout
- Source: `examples\c_inference\dbg_inference.asm` (adapts to the current v7 export surface; update it in the same session when the DLL contract changes)

---

### dbg_image_recog.exe, Vision Inference Harness

A C vision harness: image + text prompt -> caption. Loads the DLL via `v7_load`, calls the public
lifecycle (`v7_init`, `v7_initialize_inference` ), loads the image pixels itself, and calls
`v7_chat_image` (the DLL does no file I/O). It supports both F32 and Q8 inference paths.

#### Usage

```
dbg_image_recog.exe [-q8 <0|1>] [-set-template "<fmt>"] [-set-visfmt "<fmt>"] <dll_path> <db_path> [raw_image] [prompt]
```

- `-q8 0` = F32 forward (default), `-q8 1` = Q8 forward, both paths must work
- `-set-template` / `-set-visfmt`, in-memory template overrides (test hooks)
- `dll_path`, path to `v7_threading.dll` (required, no default)
- `db_path`, path to `lz4ht.db` (required, no default)
- `raw_image`, raw RGB file plus its `.meta` sidecar ("W H" text file next to it; default: `cat.raw`, note the canonical test image shipped in the project root is `cat640.raw`, so pass it explicitly)
- `prompt`, text question about the image (default: "What is in this image?")

The tool reads the `.raw` file and its `.meta` sidecar itself and passes the packed RGB buffer
+ dimensions to `v7_chat_image`; the DLL does no file I/O. The `.raw` /`.meta` pair is
produced by `jpg_to_raw.py`.

#### Build

The standard build places the executable in `bin\` during its fourth stage. The following commands
compile the source file separately:

```bat
cl /nologo /O2 /arch:AVX2 /std:c11 /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /I"zydis-master\include" /I"zycore-c-master\include" /I"src" examples\c_inference\dbg_image_recog.c /Fe:dbg_image_recog.exe /link Zydis.lib /subsystem:console
```

#### Features

- Parses command line arguments (argv); flags may appear anywhere
- Loads `v7_threading.dll` dynamically (no static linking); resolves all function pointers via `v7_lib.h`
- Reads the raw RGB image file directly; passes the packed RGB buffer + dimensions to `v7_chat_image`
- F32/Q8 mode selectable via `-q8` flag; output written to stdout

#### Example

```bat
bin\dbg_image_recog.exe -q8 0 bin\v7_threading.dll weights\databases\llava_onevision_qwen2_0.5b\lz4ht.db cat640.raw "What is in this image?"
```

---

### multi_turn_demo.c, Multi-Turn Chat + Generate + Reset

Demonstrates the full public v7 API: multi-turn chat with KV-cache preservation, single-turn generate
with separate KV cache, and fresh session reset.

#### Usage

```
multi_turn_demo.exe <db_path>
```

- `db_path`, path to `lz4ht.db` (required, no default). The DLL is loaded from `v7_threading.dll` in the current directory.

#### Build

The standard build places the executable in `bin\` during its fourth stage. The following commands
compile the source file separately:

```bat
cl /nologo /O2 /arch:AVX2 /MD /std:c11 /DZYDIS_STATIC_BUILD /I"src" /I"zydis-master\include" /I"zycore-c-master\include" examples\c_inference\multi_turn_demo.c /Fe:multi_turn_demo.exe /link Zydis.lib kernel32.lib
```

#### What it demonstrates

The program performs five generation rounds to demonstrate the different session-state behavior of
`v7s_chat` and `v7s_generate`:

| Round | API call | Prompt | Expected behavior |
| --- | --- | --- | --- |
| 1 | `v7s_chat(reset=1)` | "What is a cat?" | Fresh chat, answers about cats |
| 2 | `v7s_chat(reset=0)` | "What animal did we talk about?" | Multi-turn, remembers cats from round 1 |
| 3 | `v7s_generate()` | "Explain quantum computing briefly" | Separate KV cache, doesn't disturb chat |
| 4 | `v7s_chat(reset=0)` | "What animal did we talk about earlier?" | Still remembers cats (generate didn't touch chat KV) |
| 5 | `v7s_chat(reset=1)` | "What animal did we talk about?" | Fresh chat, forgot everything |

#### Key API concepts

- **`v7_chat(user_prompt, system_prompt, output, buf_size, reset)`**, `reset=1` starts a new session (clears KV cache), `reset=0` continues the conversation. No context argument, the context is stored in a BSS singleton inside the DLL.
- **`v7_generate(prompt, output, buf_size, raw_generation)`**, single-turn generation on its own KV cache; the prompt is rendered through the model's turn-1 template (not raw completion, unless `raw_generation=1`), and the chat session's position is preserved
- KV-cache position is preserved across `v7_chat` calls with `reset=0`, enabling true multi-turn conversation
- **`save_kv_cache` / `load_kv_cache`** (see test_kv_cache below) persist the used KV prefix to disk and restore it later

---

### Test Programs

The following small programs isolate individual inference and integration behaviors for development and
verification. The standard `build.bat` procedure compiles them during its fourth stage and places their
executables in `bin\`.

| Program | Usage | Purpose |
| --- | --- | --- |
| `chat_only.exe` | `chat_only.exe <dll> <db>` | Two-turn chat smoke test (fresh + follow-up) |
| `generate_only.exe` | `generate_only.exe <dll> <db>` | Minimal `v7s_generate` smoke test (the prompt is rendered through the model's turn-1 template, not raw completion) |
| `multiturn.exe` | `multiturn.exe <v7_threading.dll> <lz4ht.db> [q8]` | 3-turn cat/dog/bird multi-turn regression test (Q8 default, `0` for F32) |
| `multiturn_manual.exe` | `multiturn_manual.exe <v7_threading.dll> <lz4ht.db>` | The same turns driven manually: encode + forward + sample per token (differential-isolation tool for multi-turn bugs) |
| `kv_cache.exe` | `kv_cache.exe <v7_threading.dll> <lz4ht.db>` | KV-cache persistence round-trip proof: chat "i love cats" -> `save_kv_cache` -> reset + chat "i love dogs" -> `load_kv_cache` -> "what animal did we talk about?", the answer must mention cats. Exit 0 = round-trip proven |
| `engine_death.exe` | `engine_death.exe <v7_threading.dll>` | Diagnostic test for public API error reporting: forces an internal `FATAL`, checks `v7_engine_is_dead()` and the reason, and unloads cleanly. Runs in verify.bat Phase 5 |
| `poem_continuation.exe` | `poem_continuation.exe <v7_threading.dll> <lz4ht.db> [-q8]` | Raw-continuation sample: generates a poem opening via chat, then continues it mid-line with `v7_generate(..., raw_generation=1)`, demonstrates the raw (no-template) generation path and the death-check pattern |
| `v7_decide_example.exe` | `v7_decide_example.exe <dll> <db> <kind> <state> <question> "opt1;opt2;..." [-q8 0\|1]` | Typed decision: one decision prompt, one prefill pass, a probability per option label. Prints one JSON line (`{"options":[...],"probabilities":[...],"choice":"..."}`). Full mechanism + measured results: `native_api.md` and `examples/decision_Results.md` |
| `spam_classifier.exe` | `spam_classifier.exe <dll> <db> <mail.txt> [-q8 0\|1]` | Spam vs ham classification of an email text file via one typed decision (`{"file":...,"p_ham":...,"p_spam":...,"decision":"..."}`). Test mails in `examples\spam_classifier\mails\` |

Note: `multiturn.c` (older 5-round variant, `<db_path>` argument) is kept as source; build.bat builds
the test programs listed above from `examples\c_inference\`.

---

### v7_pipe.exe, Text Processing Utility

A Unix-philosophy CLI tool that pipes text through the v7 LLM for summarization, explanation, Q&A, or
chat. Demonstrates the core v7 public API (`v7_load`, `v7_init`, `v7_initialize_inference`, `v7_chat`,
`v7_generate`, `v7_free_inference`, `v7_unload` ) in a single ~200-line C file.

#### Usage

```
echo "long text..." | v7_pipe summarize    Summarize piped text (3-5 sentences)
cat file.md | v7_pipe explain               Explain piped text in simple terms
v7_pipe ask "What is the capital of France?"  Ask a question (single-turn, turn-1 template)
v7_pipe chat "Tell me about cats"           Chat (multi-turn, templated)
v7_pipe qa "Who built it?" < doc.txt       Answer question about piped document
```

#### Environment variables

| Variable | Description |
| --- | --- |
| `V7_DB` | Path to the model database (required, no default; the program exits with an error if unset) |
| `V7_DLL` | Path to the DLL (optional, falls back to `v7_threading.dll` in the current directory) |

#### Modes

| Mode | API used | Description |
| --- | --- | --- |
| `summarize` | `v7s_chat` (reset=1) | Reads text from stdin, prepends "Summarize the following text in 3-5 sentences:" |
| `explain` | `v7s_chat` (reset=1) | Reads text from stdin, prepends "Explain the following text in simple terms:" |
| `ask` | `v7s_generate` | Single-turn generation, separate KV cache, turn-1 template render, no multi-turn |
| `chat` | `v7s_chat` (reset=1) | Full chat with template rendering (model-specific) |
| `qa` | `v7s_chat` (reset=1) | Reads document from stdin, builds a QA prompt |

#### Build

The standard build places the executable in `bin\` during its fourth stage. The following commands
compile the source file separately:

```bat
cl /nologo /O2 /arch:AVX2 /std:c11 /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /I"zydis-master\include" /I"zycore-c-master\include" /I"src" examples\c_inference\v7_pipe.c /Fe:v7_pipe.exe /link Zydis.lib /subsystem:console
```

#### Examples

```bat
:: Set env vars (V7_DB required; V7_DLL optional)
set V7_DLL=bin\v7_threading.dll
set V7_DB=weights\databases\qwen3_0.6b\lz4ht.db

:: Ask a question
bin\v7_pipe.exe ask "What is the capital of France?"
:: Output: The capital of France is Paris.

:: Summarize piped text
echo The Eiffel Tower was built from 1887 to 1889... | bin\v7_pipe.exe summarize
:: Output: The Eiffel Tower, designed by Gustave Eiffel in 1887-1889,...

:: Answer a question about a document
type readme.txt | bin\v7_pipe.exe qa "What is this about?"
```

#### Source (abridged, see `examples\c_inference\v7_pipe.c`)

```c
#include "v7_lib.h"

int main(int argc, char** argv)
{
    char* dll_path = getenv("V7_DLL");
    v7_lib dll;
    v7_load(dll_path ? dll_path: "v7_threading.dll", &dll);
    dll.v7_init(getenv("V7_DB"));
    dll.v7_initialize_inference();

    char output[32768] = {0};
    if (strcmp(argv[1], "ask") == 0)
        dll.v7_generate(argv[2], output, sizeof(output), 0);
    else if (strcmp(argv[1], "chat") == 0)
        dll.v7_chat(argv[2], 0, output, sizeof(output), 1);
    /*... summarize, explain, qa modes read from stdin... */

    printf("%s\n", output);
    dll.v7_free_inference();
    v7_unload(&dll);
}
```

(Full source: `examples\c_inference\v7_pipe.c`)

---

### vit_bbox_eval.exe, ViT Bounding-Box Evaluation Harness

This C evaluation program for vision models synthesizes a 640x640 RGB image (gray background, red circle +
green square) in memory, then asks the vision model to localize the shapes. Greedy sampler; Q8 by
default, F32 selectable via `-q8 0`. Exit code is 0 only if the model produced a non-empty answer, there
is no pass/fail on box quality (this is an *evaluation*, not a unit test).

#### Usage

```
vit_bbox_eval.exe <db_path> [-q8 <0|1>] [prompt] [cx] [cy] [r] [sqx] [sqy] [sqs]
```

- `db_path`, path to a vision-capable `lz4ht.db` (required, no default; Qwen3-VL-4B, LLaVA-OneVision-Qwen2-0.5B, or SmolVLM-256M)
- `-q8 0|1`, F32 or Q8 forward path (default: Q8; both must work)
- `prompt`, text instruction (default: `"Detect the bounding boxes of all objects."`)
- `cx cy r`, circle center and radius in pixels (defaults: 180 180 90)
- `sqx sqy sqs`, square top-left and side length in pixels (defaults: 400 400 140)

#### Build

The standard build places the executable in `bin\` during its fourth stage. The following commands
compile the source file separately:

```bat
cl /nologo /O2 /arch:AVX2 /std:c11 /DZYDIS_STATIC_BUILD /DZYCORE_STATIC_BUILD /I"zydis-master\include" /I"zycore-c-master\include" /I"src" examples\c_inference\vit_bbox_eval.c /Fe:vit_bbox_eval.exe /link Zydis.lib /subsystem:console
```

#### Evaluation report

The full evaluation across 3 image configurations x 2 models (Qwen3-VL-4B and
LLaVA-OneVision-Qwen2-0.5B), with ground-truth boxes, raw model answers, and per-corner deviation
tables, is stored in the evaluation history (dev tree only: `docs\research_logbook\` ).

---

### c_callback_example.c, Host FFI Callback Demo

Demonstrates calling a host-defined C function from a SectorLogo sidecar script via the `CallLibrary`
FFI mechanism. The host registers a callback (`my_callback`) that the sidecar invokes during inference.
The full FFI contract, including the `"HOST"` target, is in the FFI chapter of the SectorLogo manual
(docs/sectorlogo_manual.md, section 8).

#### Usage

```
c_callback_example.exe <dll_path> <db_path> [prompt]
```

#### Build

The standard build places the executable in `bin\` during its fourth stage. The following commands
compile the source file separately:

```bat
cl /nologo /O2 /arch:AVX2 /std:c11 /DZYDIS_STATIC_BUILD /I"src" /I"zydis-master\include" /I"zycore-c-master\include" examples\sectorlogo\c_callback_example.c /Fe:c_callback_example.exe /link Zydis.lib /subsystem:console
```

---

### run_sectorlogo.c, Standalone Sidecar Runner

This application executes a SectorLogo `.logo` script without running model inference; it is used for
sidecar development. Script files are appended with a driver or run directly; hook variables
(`:Piece`, `:TokenNumber` ) are NIL at top level, so sidecars guarded by `IfElse Equal:Piece...` are
safe no-ops.

#### Usage

```
run_sectorlogo.exe <v7_threading.dll> <lz4ht.db> <script.logo>
```

#### Build

The normal build produces this executable in `bin\` during its fourth stage.

---

### RAG Experiments

Three end-to-end RAG (Retrieval-Augmented Generation) experiments demonstrate injecting external
knowledge into the model's thought stream via SectorLogo sidecar scripts. Each is self-contained: own
knowledge base, sidecar script, and run script. See
[rag_documentation.md](retrieval.md#retrieval-and-rag-implementation-reference) for the main reference
and `examples/rag/README.md` for the tool overview.

| Experiment | Directory | What it demonstrates |
| --- | --- | --- |
| Herbivorous cats | `examples/research/herbivorous_cats/` | Overriding a strong prior ("cats eat meat") with injected knowledge (append-style and BM25-replace sidecars) |
| Helion HX-340 | `examples/research/helion_pump/` | Filling a knowledge vacuum (fictional machine specs) |
| Kingdom of Ardmark | `examples/ardmark/` | The prime demo: entity QA and frame hallucination in fictional history; think-SEEDED sidecar (primer prepend + dual-floor BM25 gating + walk-room injection + answer scaffold) |

#### The Ardmark think-seeded sidecar (v4)

The Ardmark example in `examples/ardmark/ardmark_rag.logo` demonstrates a multi-stage retrieval sidecar.
It initializes a fictional-history context, retrieves relevant passages during generation, and uses
stream rewriting to continue from the updated context. Its operation consists of three stages:

1. **Primer (first turn only)**: the model's first thinking token triggers a
   prepend of `ardmark_prepend.txt` (what is real in Ardmark, what the model
   may tell, what it must never invent), closed by a `" ..."` sentinel. The
   think block stays OPEN, the model generates reasoning text on top of the primer.
2. **RAG seed (every turn)**: at the think-close the sentinel is found, the
   primer is stripped, and the model's own thinking becomes the BM25 query
   over `ragg_base.txt` (33-paragraph consistent fictional history). The hits
   are appended with a strict answer-from-facts instruction; the think tag
   stays OPEN so the model reasons on the retrieved facts and closes the tag
   itself.
3. **Loop guard**: the microdb key `ardmark_injected` remembers the
   injection; the model's own `</think>` is accepted and the key resets.
   EOS also resets the markers.

Before each chat turn, the HTTP server writes `__server_chat__turn` to MicroDB. The value is `0` for a
reset turn and increments on continuation turns, allowing scripts to distinguish initial setup from
subsequent generation. For script-driven examples, `inference_example.exe` supplies the value `0`.

#### Quick start

```bat
:: From project root
examples\ardmark\run.bat
examples\research\herbivorous_cats\run.bat
examples\rag\memory\test_memory_rag.bat:: the persistent-memory sidecar E2E
examples\run_inference_examples.bat:: every text-inference example, gated
examples\run_tool_examples.bat:: pipe / tuner / bayes / bm25 bench / vision
examples\run_logo_scripts.bat:: one smoke chat per Logo sidecar
```

---

### Continuous background tuner (the in-process cycle tuner)

The engine tunes its own matmul dispatch configuration with the in-process
cycle tuner: it is always active in every program that runs inference, all
`supported_resolutions` shapes rotate, and accepted configurations persist
to the `v7_tune_param_<n>_<d>` files plus the `v7_tuning_journal`. No Logo
scripts and no host support are in the loop; the tuner is pure assembly
(`cycle_tuner_main.inc`) and has no enable/disable switch. Observation
exports: `v7_get_best_fitness`, `v7_get_pool_fitness`,
`v7_get_active_shape_count`.

`run_tuner_chats.bat` is the proof driver: it runs dozens of real chat turns
against a tuner DB copy so the cycle sorter accumulates episodes live, and
each engine destroy persists the populations to the
`v7_tune_param_<n>_<d>` files. The verdict counters print at engine destroy
(`tuner: accepted N candidates` / `tuner: rejected M candidates`),
confirming the loop ran end to end. The tuning manual
([tuning.md](tuning.md#dispatch-and-tuning)) documents design and knobs.

---

### v7_server.exe, HTTP Chat Server

This HTTP server wraps the V7 inference engine and provides an OpenAI-compatible chat API, a
browser-based HTML interface, and runtime SectorLogo sidecar loading. Multi-turn conversations reuse
the KV cache across requests within the same session.

> **A development tool rather than an internet-facing service.** The server is
> neither hardened nor a finished inference service. It is a local test
> application and usage example for the v7 library. `/sidecar?<path>` loads
> arbitrary scripts, and sidecars can start external programs through
> `DefineCommand` and `CallLibrary`. Disable or restrict the sidecar
> endpoint before any exposure. Use it as a test server or for local use
> only.

#### Usage

```bat
v7_server.exe <v7_threading.dll> <lz4ht.db> [port] [model_id] [-prio N] [-maxgen N] [-api-key KEY | -api-key-env NAME]
```

- `port`, TCP port (default: 8080)
- `model_id`, model identifier for API responses (default: "qwen3-0.6b")
- `-prio N`, Win32 worker priority for v7 (-15 idle .. 15 time critical; default -15)
- `-maxgen N`, server-wide default generation budget; 0 = unlimited (default)
- `-api-key KEY`, require `Authorization: Bearer KEY` on the /v1/ inference endpoints (unset = no auth)
- `-api-key-env NAME`, take the key from environment variable NAME instead of the command line (stays out of the process list)

Environment: `V7_SERVER_BIND` (default `127.0.0.1`; set `0.0.0.0` to serve the LAN), `V7_SIDECAR_DIR`
(restrict `/sidecar` to scripts under one directory), `V7_SERVER_TOKDUMP=1` (token-stream trace in the
server log).

#### Lifecycle

```bat
start_server.bat:: start in its own window (restart loop on crash)
stop_server.bat:: clean shutdown via /quit?code=42
server_set_sidecar.bat <file>:: load a sidecar into the running server
```

#### Endpoints

| Endpoint | Description |
| --- | --- |
| `GET /` | Browser-based HTML chat interface |
| `GET /v1/health` | Health endpoint (503 if the engine has terminated) |
| `GET /v1/models` | Model list (for API clients; `/v1/models/<id>` also works) |
| `POST /v1/chat/completions` | Multi-turn chat API (JSON request/response; supports `stream`, `max_completion_tokens`, `enable_thinking`/`reasoning_effort: "none"`, array `content` parts, `developer`/`tool` roles) |
| `POST /v1/completions` | Single-turn generate API (array `prompt` joined; supports `stream`) |
| `POST /v1/decide` | Typed decision (see docs/native_api.md, the typed-decision section; kinds `noul`/`score`/`choice` are validated) |
| `POST /v1/vision/chat` | Image turn on the chat session (base64 image; decoded at native size, guarded by a 64-megapixel metadata cap) |
| `POST /v1/chat/reset` | Start a fresh session (clears the KV cache; requires `X-V7-Local: 1`) |
| `GET /sidecar?<path>` | Load a SectorLogo sidecar script (requires `X-V7-Local: 1`; UNC paths rejected, optional allowlist via `V7_SIDECAR_DIR`) |
| `GET /quit?code=42` | Clean shutdown (requires `X-V7-Local: 1`) |

State-changing simple requests (`/quit`, `/sidecar`, `/v1/chat/reset` ) require the `X-V7-Local: 1`
header. A loopback-bound server also rejects non-loopback `Host` headers (DNS-rebinding) and
non-loopback `Origin` headers on the inference endpoints. Responses carry `X-Frame-Options` and
`X-Content-Type-Options`. Second `Ctrl+C` force-stops a running turn.

Full reference: [server_documentation.md](native_api.md#http-integration).

---

### bayes_classify.exe, Naive Bayes Text Classifier

A two-class Naive Bayes text classifier (Laplace-smoothed multinomial, log space). Trains on two
plain-text example corpora and classifies new text. Useful for sentiment analysis, topic routing, or
filtering in RAG systems. Full usage: run `bin\bayes_classify.exe` with no arguments.

#### Usage

```bat
bayes_classify.exe -build <case0_file> <case1_file> <state_file>
bayes_classify.exe -classify <state_file> "<text>"
```

#### Example

```bat
:: Train on two example corpora
bin\bayes_classify.exe -build confused.txt sure.txt bayes_state.txt

:: Classify new text (prints "class0" or "class1")
bin\bayes_classify.exe -classify bayes_state.txt "hmm not sure, maybe?"
```

## Tools, validators, and benchmarks

The repository includes utilities for preparing model data, inspecting MicroDB databases, checking
native operations, and measuring performance. These tools are useful at different stages of development,
but most are not needed by an application that only loads a prepared model and calls the public API.

Use the validation programs when checking whether a build produces the expected results. They exercise
defined operations and compare behavior with their reference expectations. Benchmarks answer a different
question: how quickly a particular build runs a specified workload with a given model and configuration.
A passing test does not establish throughput, and a fast benchmark does not establish correctness. The
verification scripts provide the documented sequence of checks, so they are generally a better starting
point than running every executable in `tools/` individually.

Most C and C++ tools load the DLL dynamically through the header-only helper in `v7_lib.h`; the assembly
test harness loads it directly. The reference entries below identify each utility, explain when it is
useful, and document the relevant command-line arguments.

The validation programs exercise the DLL and supporting utilities at different levels. Some require only
the compiled library and synthetic inputs, while others need a prepared model database. Run the standard
verification sequence after a normal build, then use the focused programs when investigating a
particular operation. Passing a small numerical validator does not establish that a model database,
tokenizer, or complete generation path has been configured correctly.

### Validators

The included validators check kernel correctness. Each loads `v7_threading.dll` through `v7_lib.h`, runs kernels against
known inputs, and prints PASS/FAIL with per-case error values.

#### DLL-only validators, no model database needed

| Tool | What it validates |
| --- | --- |
| `v_memcpy_validate.exe <dll>` | Dispatched memcpy correctness |
| `v_rmsnorm_validate.exe <dll>` | RMSNorm kernel, fused RMSNorm+quantize, RMSNorm+RoPE combined kernels |
| `v_add_inplace_validate.exe <dll>` | Element-wise add-inplace |
| `v_swiglu_validate.exe <dll>` | SwiGLU F32xQ8 and Q8xQ8 |
| `v_quantize_validate.exe <dll>` | FP32 to INT8 quantize with GS=64 |
| `v_mdb_validate.exe <dll>` | MicroDB hash table stress test, including the large-blob compress ladder |
| `v_sectorlogo_validate.exe <dll>` | SectorLogo interpreter: language, strings, FFI, GC and sidecar settoken tests. See [SectorLogo manual](sectorlogo.md#sectorlogo-language-and-integration). |
| `v_execute_validate.exe <dll>` | `v7_execute` spawn, capture and timeout, 4 tests |
| `v_sprintf_validate.exe <dll>` | `v7sprintf` bounded formatting across `%s`, `%d`, `%c`, `%f`, `%%`, bounds and edge cases |
| `v_win32_quote_validate.exe <dll>` | Win32 argv quoting and escaping, an exhaustive sweep against the MSVCRT tokenizer |
| `v_bicubic_validate.exe <dll>` | Bicubic resize kernel against a double-precision C reference, covering identity, upscale, downscale, anisotropic and tiny inputs for the LLaVA-OV preprocessor |

#### Weight-dependent validators, which require `--db <lz4ht.db>`

| Tool | What it validates |
| --- | --- |
| `v_matmul_validate.exe <dll> --db <db>` | Q8 + F32 matmul |
| `v_fused_validate.exe <dll> --db <db>` | Fused F32 matmul + RMSNorm + RoPE; fused Q8 matmul section |
| `v_sampler_validate.exe <dll> --db <db>` | ASM sampler, where greedy output matches the C reference |
| `v_encode_validate.exe <dll> --db <db>` | BPE tokenizer encode |
| `v_attention_validate.exe <dll> --db <db>` | Flash attention |
| `v_readwrite_validate.exe <dll> --db <db>` | `v7s_readfile` and `v7s_writefile` file I/O, 3 tests |
| `v_generate_validate.exe <dll> <db>` | `v7s_generate` end-to-end generate path, F32 + Q8, checks the "paris" answer (positional db argument, no `--db`) |
| `v_vit_linear_validate.exe <dll> --db <db>` | Dispatched ViT linear matmul |
| `v_stream_validate.exe <dll> --db <db>` | StreamingLLM: setter contract (pre-init FATAL, arg rejects, wrapped-disable refusal), T1 legacy bit-equivalence in F32 + Q8, small-window wrap (S=4, W=8, C=12) with `on_stream_evict` callback delivery, nested generate, KV save/load round-trip |

The benchmark executables are intended to measure a particular execution path under stated conditions.
Their results depend on the processor, active threads, matrix dimensions, precision mode, and background
system activity. Keep those conditions with any published figures, and use an end-to-end inference run
when comparing model throughput rather than assuming that an isolated kernel measurement predicts the
complete result.

### Benchmarks

The kernel benchmarking tools report throughput in GB/s and CPU utilization for Q8 and F32
matrix multiplication across several shapes. The [CPU inference benchmark](engine_benchmark.md)
reports the full-model decode rate in tok/s.

| Tool | What it benchmarks |
| --- | --- |
| `tools\b_benchmark.exe <dll> --db <db> [q8|f32|all]` | Q8 + F32 matmul across 8 shapes in 2 modes |
| `tools\b_loadtest.exe <dll> --db <db> [--q8|--f32]` | Q8 + F32 with worker spin enabled |
| `b_fused_loadtest.exe <dll>` | Fused F32 matmul + RMSNorm + RoPE |
| `b_rmsnorm_loadtest.exe <dll>` | RMSNorm kernel |
| `b_add_inplace_loadtest.exe <dll>` | Add-inplace kernel |
| `b_fused_add_rmsnorm_loadtest.exe <dll>` | Fused add + RMSNorm |
| `b_rmsnorm_rope_loadtest.exe <dll>` | RMSNorm + RoPE |
| `b_swiglu_loadtest.exe <dll>` | SwiGLU F32xQ8 |
| `b_swiglu_q8_loadtest.exe <dll>` | SwiGLU Q8xQ8 |
| `b_memcpy_loadtest.exe <dll>` | Dispatched memcpy |
| `b_attention_benchmark.exe <dll>` | Attention throughput |
| `b_attention_loadtest.exe <dll>` | Attention load test |
| `b_vit_linear.exe <dll>` | ViT linear matmul |

Benchmarks support `--time N` for seconds and `--iterations N` for a fixed iteration count.

### DB Query Tools

Inspect `lz4ht.db` contents without C code.

| Tool | What it does |
| --- | --- |
| `mdb_info.exe <dll> --db <db>` | Hash table summary with capacity, load factor and key pool usage |
| `mdb_list.exe <dll> --db <db>` | List non-vocab entries |
| `mdb_query_key.exe <dll> <key> --db <db>` | Query a single key with a hex dump and a type-aware display |
| `mdb_search.exe <dll> <pattern> --db <db>` | Search entries by key substring |
| `mdb_dump.exe <dll> --db <db>` | Dump all entries |

### RAG Tools

| Tool | What it does |
| --- | --- |
| `bm25_rag.exe <command> [args]` | Primary BM25 retrieval tool for local RAG. It replaced the earlier retrieval executables. |
| `bm25_rag.exe search "<query>" "opts"` | Production RAG retrieval with packed options for `maxlines`, `minscore`, `topmin`, `base`, `dedup`, `recency`, `recent` and `xbase`, plus normalization, a dual floor and a recent-memory fallback |
| `bm25_rag.exe remember "<fact line>" "opts"` | Append a fact with Jaccard-based duplicate detection |
| `llm_text_processor.exe <dll> <db> [corpus] [query]` | BM25 and LLM distillation engine that reformulates text into 3 factual thoughts. Modes: `-direct`, `-only-bm25`, `-distill`, `-sweep`. |
| `web_ragg.exe search/makequery` | Internet RAG: DuckDuckGo search, page fetch, MMR, LLM distillation |
| `bayes_classify.exe -build/-classify` | Two-class Naive Bayes text classifier |

#### bm25_rag.exe

A standalone BM25 search engine that sidecar scripts call through `DefineCommand`, which wraps
`v7_execute_fmt`. It operates on a local knowledge base in `ragg_base.txt`, an append-only corpus in
`ragg_corpus.txt`, or an explicit `base=PATH`. See
[rag_documentation.md](retrieval.md#retrieval-and-rag-implementation-reference) for the main sequence.

```
bm25_rag.exe search "<query>" "[rand=N;][maxlines=N;][minscore=S;][topmin=S;][base=PATH;][dedup=S;][recency=S;][recent=N;][xbase=PATH]"
bm25_rag.exe remember "<fact line>" "[dedup=S;][base=PATH]"
bm25_rag.exe add "<text>"                        # Append a paragraph to the corpus (next to the exe)
bm25_rag.exe rand                                # Random passage from base + corpus
```

- `ragg_base.txt`, static knowledge base, one paragraph per line
- `ragg_corpus.txt`, growing memory, with new paragraphs added via `add`
- `ragg.log`, query log with search queries, scores and retrieval decisions

The production options string packs everything into one string because the FFI passes few string
parameters:

```
bm25_rag.exe search "<query>" "maxlines=8;minscore=0.015;base=examples/rag/memory/ragg_base.txt"
```

- Normalizes both corpus and query: lowercase, markdown stripped, whitespace collapsed
- Dual answer-presence floor: inject only if density >= minscore OR topscore >= topmin
- Recent-memory fallback: zero-overlap queries such as "what was recorded earlier?" return the corpus tail
- Diagnostics go to `ragg.log` only, because stderr is merged into the captured stream by the exe-call mechanism

#### llm_text_processor.exe

The distillation engine behind the memory system: runs the model itself to reformulate text into exactly
3 short factual thoughts.

```
llm_text_processor.exe <dll> <db> x x -distill -maxgen 120 -finputtext in.txt -fprepend pre.txt -fappend app.txt
```

- `-direct` skips BM25 and takes the source text from `-finputtext`; `-only-bm25` skips the LLM
- `-distill` means `-direct` with stdout and stderr limited to the distilled output, as required for sidecar capture
- `-fprepend` and `-fappend` are the instruction prompts. They are read from files, so prompt experiments need no rebuild.
- `-sweep` prints machine-readable lines for parameter sweeps
- Full reference: [thought_rag.md](retrieval.md#model-assisted-retrieval)

Note: the former `mem_bridge.exe` adapter, a single-parameter wrapper for the memory sidecar's EOS
hook, was removed; it was superseded by the proper exe-call mechanism. The sidecar writes the turn
text with `v7s_writefile` and calls the distiller directly through `DefineCommand`. See
rag_documentation.md Sec. 8.3.

#### web_ragg.exe

Internet RAG tool: DuckDuckGo search, then fetch the top 5 pages, extract HTML content, select passages
with MMR, and distill into a short factual paragraph. The distilled paragraph is injected into the
model's thought stream through a SectorLogo sidecar hook, which gives the model current event
information.

```
web_ragg.exe search "<query>" ["pages=5;words=100;maxgen=250;fresh"]
web_ragg.exe makequery "<think and memories text>"
```

- `search`, full sequence: search, fetch, extract, select, distill. Returns a short paragraph of distilled facts.
- `makequery`, extract a search query from the model's thought text, capped at 4000 chars.
- Options string: `pages=N` for 1 to 10, `words=N` for 30 to 300, `maxgen=N` for 10 to 500, and `fresh` to bypass the cache.
- Paths are read from `web_ragg.ini` next to the exe, keys `[v7] Dll=` and `[v7] Db=`. Both are required and have no defaults.
- Page cache: `web_ragg.db`, a flat CRC32-checked file where corrupt entries are skipped
- Own build: `examples\rag\web_ragg\build_web_ragg.bat`

#### bayes_classify.exe

This two-class Naive Bayes text classifier trains on two plain-text corpora and then classifies new
text. It can be used for sentiment analysis, topic routing, or filtering in RAG systems.

```
bayes_classify.exe -build <case0_file> <case1_file> <state_file>
bayes_classify.exe -classify <state_file> "<text>"
```

- `-build`, trains on the two example files, writes the classifier state file.
- `-classify`, loads the state and prints `class0` or `class1`.

### Server

#### v7_server.exe

**Development tool and local test application.** This server demonstrates the API; it is not a
hardened internet-facing service. `GET /sidecar?<path>` loads a SectorLogo script, and a script can start
external programs through `DefineCommand` or `CallLibrary`. Keep the sidecar endpoint local or restrict
it before exposing the server to a network.

The example HTTP server exposes a chat API using the V7 inference engine. It supports multi-turn
conversations with KV-cache persistence across requests, a browser-based HTML chat interface,
and runtime SectorLogo sidecar loading.

```
v7_server.exe <v7_threading.dll> <lz4ht.db> [port] [model_id] [-prio N] [-maxgen N] [-api-key KEY | -api-key-env NAME]
```

- `port`, TCP port, default 8080
- `model_id`, model identifier for API responses, default `qwen3-0.6b`

**Endpoints:**

| Endpoint | Description |
| --- | --- |
| `GET /v1/health` | Health check |
| `POST /v1/chat/completions` | OpenAI-compatible chat API. Streaming is not supported. |
| `GET /v1/models` | List available models |
| `GET /sidecar?<path>` | Load a SectorLogo sidecar script. The query string is the file path. |
| `GET /quit?code=42` | Clean shutdown. Code 42 stops; any other code restarts. |
| `GET /` | Browser-based HTML chat interface |

**Lifecycle scripts in the project root:**

| Script | Purpose |
| --- | --- |
| `start_server.bat [logfile]` | Start the server in its own visible window; the supervisor restarts it after an unexpected exit |
| `stop_server.bat [port]` | Clean shutdown via `/quit?code=42` |
| `server_set_sidecar.bat <file> [port]` | Load a sidecar script into the running server |

See [server_documentation.md](native_api.md#http-integration) for the full reference.

### Example Programs

All are built by `build.bat` into `bin\`, and their sources live under `examples\`. See
[examples.md](getting_started.md#example-applications) for the full catalog.

| Tool | What it does |
| --- | --- |
| `inference_example.exe <dll_path> <db_path> [options]` | Minimal C inference: load, chat, print, cleanup. Options are `-q8`, `-i "prompt"`, `-n <tokens>`, `-r <rounds>`, `-set-template "<fmt>"` for a memory-only template override where `"%s"` selects raw completion as described in [native API reference](native_api.md#c-api-reference), `-think <0\|1>`, `-rewrite`, `-script <file>`, `-tokdump` and `-exec` |
| `dbg_inference.exe <dll_path> <db_path> ["prompt"] [-q8]` | Pure-ASM inference harness on the public v7_* tier, using MSVC post-2008 argument parsing. F32 is the default and `-q8` selects the quantized forward. |
| `dbg_image_recog.exe [-q8 <0\|1>] <dll_path> <db_path> [raw_image] [prompt]` | Vision harness: image plus text prompt produces a caption, on both the F32 and Q8 paths |
| `vit_bbox_eval.exe <db_path> [-q8 <0\|1>] [prompt] [cx cy r sqx sqy sqs]` | ViT bounding-box evaluation harness that synthesizes 640x640 test images |
| `multi_turn_demo.exe <db_path>` | Multi-turn chat, generate, and reset demonstration over 5 rounds |
| `chat_only.exe <dll> <db>` | Two-turn chat smoke test |
| `v7_decide_example.exe <dll> <db> <kind> <state> <question> "opt1;opt2;..." [-q8 0\|1]` | Typed decisions: builds a decision prompt, returns a probability for every option label (`native_api.md`) |
| `spam_classifier.exe <dll> <db> <mail.txt> [-q8 0\|1]` | Classify an email text file as spam or not spam in one decision pass |
| `generate_only.exe <dll> <db>` | Minimal v7_generate test |
| `multiturn.exe <v7_threading.dll> <lz4ht.db> [q8]` | 3-turn multi-turn regression test over cat, dog and bird. Q8 is the default and `0` selects F32. |
| `multiturn_manual.exe <v7_threading.dll> <lz4ht.db>` | The same turns driven manually, with encode, forward and sample per token |
| `kv_cache.exe <v7_threading.dll> <lz4ht.db>` | KV-cache persistence round-trip proof: save, reset, load, and the saved session answers |
| `engine_death.exe <v7_threading.dll>` | Diagnostic test for public API error reporting. It forces an internal `FATAL`, checks the reason, and unloads cleanly. |
| `poem_continuation.exe <v7_threading.dll> <lz4ht.db> [-q8]` | Raw-continuation demo: chat a poem opening, then continue it mid-line with `v7_generate(..., raw_generation=1)` |
| `v7_pipe.exe <mode> [args]` | General text processor with `ask`, `chat`, `summarize`, `explain` and `qa`. The V7_DLL env var is optional and V7_DB is required. |
| `run_sectorlogo.exe <dll> <db> <script.logo>` | Run a SectorLogo sidecar standalone, as a sidecar development driver |
| `c_callback_example.exe <dll> <db> [prompt]` | Host FFI callback demo through `CallLibrary [ "HOST"... ]` |
| `modelchat.bat "<prompt>"` | Convenience chat with the memory RAG system. It wraps inference_example and `sidecar_memory_rag_v6.logo`, and runs from the project root. |

#### dbg_image_recog.exe

```
dbg_image_recog.exe [-q8 <0 1>] [-set-template "<fmt>"] [-set-visfmt "<fmt>"] <dll_path> <db_path> [raw_image] [prompt]
```

- `-q8 0` selects the F32 forward and is the default; `-q8 1` selects the Q8 forward. Both have to work.
- `dll_path` and `db_path` are required and have no defaults
- `raw_image` is a raw RGB file plus its `.meta` sidecar, a text file next to it holding "W H". The default is `cat.raw`; the canonical test image shipped in the project root is `cat640.raw`, so pass it explicitly.
- `prompt` is the text question about the image

The harness reads the raw RGB file itself and passes the packed RGB buffer plus dimensions to
`v7_chat_image`; the DLL does no file I/O. The canonical test image is `cat640.raw` at 640x480x3, in the
project root.

```
:: Caption the canonical test image (F32 and Q8 both must work)
bin\dbg_image_recog.exe -q8 0 bin\v7_threading.dll weights\databases\llava_onevision_qwen2_0.5b\lz4ht.db cat640.raw "What is in this image?"
```

### Vision Tools

| Tool | What it does |
| --- | --- |
| `tools\export_vlm.py` | Export a Qwen3-VL, LLaVA-OneVision or SmolVLM model to a v7 text checkpoint of `.bin`, `.tokenizer` and `.template` files, plus a `.vision` blob (and a `.proj` projector blob where the model has a separate projector) |
| `tools\jpg_to_raw.py` | Convert any image to a raw RGB888 `.raw` file for `dbg_image_recog.exe` |
| manual setup (see the models table in [setup.md](getting_started.md#installation-building-and-model-preparation)) | Download Qwen3-VL-4B-Instruct from HuggingFace into `weights\qwen3_vl_4b\` |

#### export_vlm.py

```
python export_vlm.py <filepath> <huggingface_model_path> [--max-seq-len N]
```

- `<filepath>` is the output path without extension, for example `weights/qwen3_vl_4b/qwen3_vl_4b`
- `<huggingface_model_path>` is a local directory holding the web-research config and the safetensors shards
- `--max-seq-len N` caps the exported sequence length. The default is 8192, because the real 262144 would size the KV cache at roughly 36 GiB (38.7 GB) with FP16 storage.

The script reads the web-research config, extracts `text_config` and `vision_config`, loads the
safetensors shards keeping only the text keys remapped from `model.language_model.X` to `model.X`, and
reuses `export.py` 's exporter verbatim. It writes a combined `<base>.vision` file, a 64-byte
VisionConfig header plus a VIT1 blob, which `build_db.exe -vision` consumes.

### Build Tools

| Tool | What it does |
| --- | --- |
| `build_db.exe <checkpoint.bin> [options]` | Build `lz4ht.db` from a model checkpoint |
| `tools\update_ini_from_state.py` | Bake the in-process tuner's learned configs from `tuner_state.txt` into `threading_proxy.ini` |

#### update_ini_from_state.py

Promotes the in-process tuner's learned per-shape configs into the official dispatch config. Reads the
mirror records (`cfg <NxD> <f0>..<f105>`) from `tuner_state.txt` and rewrites the 22 tuner-tunable keys
of the matching `[NxD]` section in `threading_proxy.ini` (`num_chunks` through `chunk_group_2_count` );
mirror fields f11 (`worker_priority`) and f23 (`cores_subtract`) are INI-owned and skipped, and the 82
prefetch fields are ignored (prefetch tuning stays a live SMC experiment). Run it after a tuner session,
then rebuild the databases so they embed the updated INI:

```
py tools\update_ini_from_state.py
build_text_model_databases.bat
build_vit_databases.bat
```

The slot-to-key contract is documented in [tuner_slots.md](tuning.md#tuner-field-and-slot-layout); the
script is the reference implementation of the promotion path.

#### build_db.exe

Builds the MicroDB database `lz4ht.db` from a model checkpoint file. The tool reads the `.bin`
checkpoint plus its companion `.tokenizer` and `.template` files from the same directory, and persists
everything into `lz4ht.db`.

```
build_db.exe <checkpoint.bin> [options]
```

The full option table (defaults and descriptions) is the [build_db.exe reference](#build_dbexe-reference)
above.

**Examples:**
```bat
:: Build for Qwen3-0.6B. Thinking templates are auto-stored when they differ from the base, and rope_base=1e6.
build_db.exe weights\qwen3_0.6b.bin -f

:: Build for Phi-3-mini (-thinking is a no-op; rope_base=10000)
build_db.exe weights\phi3_mini.bin -thinking 0 -rope-base 10000 -f

:: Build for SmolLM2-360M (-thinking is a no-op; rope_base=100000)
build_db.exe weights\smollm2_360m.bin -thinking 1 -rope-base 100000 -f

:: Build for MiniCPM5-1B (-thinking is a no-op; rope_base=5e6)
build_db.exe weights\minicpm5_1b\minicpm5_1b.bin -thinking 0 -rope-base 5000000 -f

:: Build for Qwen3-VL-4B (text checkpoint + vision tower)
build_db.exe weights\qwen3_vl_4b\qwen3_vl_4b.bin -vision weights\qwen3_vl_4b\qwen3_vl_4b.vision -f
```

### Building

#### Individual Build

```bat
:: Set up MSVC
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

:: Build a single validator
cl /nologo /O2 /arch:AVX2 /std:c++17 /DZYDIS_STATIC_BUILD /I"src" /I"zydis-master\include" /I"zycore-c-master\include" /I"tools" tools\v_rmsnorm_validate.cpp /Fe:tools\v_rmsnorm_validate.exe /link Zydis.lib
```

#### Build All

```bat
build.bat
```

Runs: DLL assembly, comment check, link, build tools, build validators, build examples.

## Troubleshooting

When an example fails, begin by checking the build output, the DLL path, and the database selected on
the command line. These are separate inputs, and an error in one can resemble a failure in another. The
following entries describe common symptoms, likely causes, and checks that can narrow the problem
without modifying the inference implementation. Where an error comes from a validator, preserve its
complete output so the failing operation can be identified.

### Build Failures

#### "No Visual Studio with the C++ workload found" / "vswhere not found"

**Cause:** vswhere (installed with Visual Studio) is missing, or no Visual Studio installation carries
the "Desktop development with C++" workload.

**Fix:** Install Visual Studio or Build Tools with the C++ workload, or point build.bat at an existing
`vcvarsall.bat` directly:
```bat
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
```

#### "NASM not found. Set NASM_DIR to nasm.exe location."

**Cause:** NASM is not on the PATH and not in the `tools\` directory.

**Fix:** Download NASM 2.16+ and either place `nasm.exe` in `<repo>\tools\` or set `NASM_DIR`:
```bat
set NASM_DIR=C:\path\to\nasm
```

#### Python ImportError during export.py

**Cause:** Missing Python dependencies.

**Fix:** Install all required packages:
```bat
pip install torch transformers safetensors huggingface_hub sentencepiece protobuf jinja2
```

### Runtime Errors

#### "Failed to load DLL: v7_threading.dll"

**Cause:** The DLL is missing or its dependencies are not available.

**Fix:**
1. Verify `v7_threading.dll` exists in the expected directory.
2. Run with explicit paths: `bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db`.
3. Rebuild with `build.bat` if the DLL is missing.

#### "v7_init failed" / "[FATAL] Required config key not found"

**Cause:** The database file (`lz4ht.db`) is missing from the given path, or it's stale/corrupt.

**Fix:**
1. Verify the database path passed to `v7_init` exists.
2. Rebuild the database: `bin\build_db.exe weights\qwen3_0.6b\qwen3_0.6b.bin -o weights\databases\qwen3_0.6b\lz4ht.db -f`
3. Pass the full database path on the command line, there is no default DB path.

#### `v7_engine_is_dead()` returns 1 after a call

**Cause:** An internal `FATAL` stopped a gated public call. The public API recorded the reason and
returned control to the host.

**Fix:**
1. Read `v7_get_death_reason()`.
2. Call `v7_free_inference()` and unload the DLL.
3. Check the database path and rebuild the database if it is stale or corrupt.
   The normal caller does not need to retry automatically.

#### STATUS_ACCESS_VIOLATION (0xC0000005) during inference

**Cause:** Likely a bug in the ASM inference pipeline. The Zydis VEH handler will print an exception dump
with register values and disassembly.

**Fix:**
1. Read the exception dump. It shows the faulting instruction and register state.
2. Check whether this is a known issue by searching the historical bug log (dev-only).
3. If the crash is in init, verify the database is not stale: rebuild with `bin\build_db.exe weights\qwen3_0.6b\qwen3_0.6b.bin -o weights\databases\qwen3_0.6b\lz4ht.db -f`.
4. Report the crash with the full Zydis dump.

#### STATUS_ACCESS_VIOLATION during PyTorch export, torch 2.7 and later

**Cause:** A known issue with `torch.from_pretrained()` in torch 2.7.0+cpu on some models.

**Fix:** The `export.py` script bypasses `from_pretrained()` and loads safetensors directly. If the
crash persists, try downgrading torch:
```bat
pip install torch==2.6.0
```

#### ERROR_COMMITMENT_LIMIT during large model inference

**Cause:** The model's working buffers, FP16 KV cache and weights all require virtual memory. The
current runtime allocates separate key/value caches of
`n_layers * seq_len * n_kv_heads * head_dim * 2` bytes each. Their size depends on the exported
context length, not on the number of tokens already generated. The working block and model database
also consume memory. All runtime allocations use `mdb_alloc_mem`, which attempts a committed
`VirtualAlloc` and can fall back to the file-backed allocator; this fallback is not limited to
`v7s_generate`. Exhausting memory/commit capacity can still prevent initialization.

**Fix:**
1. Reduce `seq_len` at export time, using `--max-seq-len` for VLMs, because the
   working block and caches scale with it.
2. Ensure Windows has an adequately sized pagefile. A system-managed pagefile is fine.
3. Close other memory-heavy processes, since commit charge is shared system-wide.

### Validator Failures

#### "v_swiglu_validate: Fail..." (Q8xQ8 section)

**Fixed:** the C reference is now a bit-exact port of the ASM Q8xQ8 kernel (exact int8 group dots,
per-group FMA accumulation in the kernel's float order, HADD lane reduce, kernel sigmoid), so every
shape gates strictly. The old documentation described a float-vs-double reference drift on large row
sums with known-limit escapes; that reference and its escapes no longer exist.

#### "v_sampler_validate: stochastic mismatches"

**Fixed:** the C reference is now a bit-exact port of the ASM sampler (same polynomial exp with the FMA
chain, the same 4x8-lane accumulator summation order, and the same heap-based nucleus with identical tie
semantics), so all 40 tests, greedy, temperature+multinomial and temperature+top-p, gate exactly and run
as part of verify.bat. Older documentation described a float-precision divergence between the ASM and an
approximate reference; that reference no longer exists.

#### Any other validator failure

**Action required:** There are no accepted or known-limit validator failures in this codebase, every
validator must pass, every run. Any failure is a blocking bug: investigate the failing validator, check
recent changes with `git diff`, and fix it before proceeding.

## Detailed source-tree reference

The directory reference below explains where to find the implementation, build inputs, examples, and
supporting utilities. It is intended for development and code review rather than routine installation.
In particular, the exported C interface and the low-level assembly modules serve different roles:
application code normally uses the former, while kernel development and validation work with the latter.

### Detailed repository layout

```text
v7_release/
|   src/                 NASM assembly sources (.asm, .inc) + headers
|   src/logo/            SectorLogo sidecar scripts shipped in every DB
|   tools/               validator/benchmark/DB-tool sources
|   examples/            example programs (c_inference, server, rag, sectorlogo)
|   docs/                documentation
|   weights/             model checkpoints (not committed)
|   bin/                 build output (DLL, tools, examples, server)
|   backgrounds/         server demo backgrounds
|   build.bat            one build script for everything
|   verify.bat           ASM-component validation gate (validators + inference)
|   verify_vision.bat    vision gates (Qwen3-VL smoke + LLaVA anyres)
|   verify_server_and_rag.bat  server/RAG session + CLI contract tests
|   autoexec.logo        SectorLogo standard library (stdlib)
|   ardmark-server.bat   the Ardmark demo as a persistent chat server
|   FILE_ID.DIZ          classic shareware-style package descriptor
|   bumwolf_avatar.jpg   the Ardmark demo avatar (in backgrounds\)
+-- CHANGELOG.md

```

---
