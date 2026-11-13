# CPU inference benchmarks

The benchmarks compare v7, llama.cpp, MNN, and OpenVINO with 8-bit weights on a
Ryzen 5 7430U laptop. The four text models are Qwen3-0.6B, Qwen3-1.7B,
Phi-3-mini-4k-instruct, and MiniCPM5-1B. The tables report generated tokens per
second (tok/s). The [HTML overview](engine_benchmark.html) charts the same
results and includes the five recorded short-chat rounds for each engine and
model in the selected runs.

## Selected 8-bit results

The table combines the November 8, 2026 four-model run for the two Qwen3 models,
the November 8 MiniCPM5-1B retest at 18:55, and the November 9 Phi-3-mini retest
at 09:38. Each row compares engines from the same run. Each engine ran five
rounds per workload. The short-chat values are the reported steady-state
figures from rounds 2–5. The poem values are the reported means over all five
rounds.

| Model | Workload | v7 | llama.cpp | MNN | OpenVINO |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | Short chat | **55.8** | 52.0 | 48.2 | 41.4 |
| Qwen3-0.6B | Poem | **52.2** | 47.3 | 43.8 | 39.7 |
| Qwen3-1.7B | Short chat | **20.0** | 19.8 | 18.1 | 18.3 |
| Qwen3-1.7B | Poem | **19.5** | 19.2 | 17.7 | 17.5 |
| Phi-3-mini | Short chat | 9.1 | **9.4** | 8.4 | 8.6 |
| Phi-3-mini | Poem | 8.7 | **8.9** | 8.0 | 7.8 |
| MiniCPM5-1B | Short chat | **38.1** | 37.7 | 34.8 | 32.0 |
| MiniCPM5-1B | Poem | **37.1** | 36.6 | 33.4 | 29.1 |

v7 recorded the highest decode throughput in both workloads for both Qwen3
models and for MiniCPM5-1B after retuning. Its lead over llama.cpp was about
7% and 10% on Qwen3-0.6B, 1% and 2% on Qwen3-1.7B, and 1% and 1.4% on
MiniCPM5-1B for short chat and the poem, respectively. For Phi-3-mini,
llama.cpp led by about 3% on short chat and 2% on the poem.

## Test conditions

- The Ryzen 5 7430U has six physical cores and 12 logical processors. The
  laptop ran on AC power. The four-model run took place on November 8 from
  13:21 to 14:10 with background applications closed and networking off.
- The short-chat prompt was “What is the capital of France?” The long
  generation workload used the literal prompt
  `write a long poem about horses, jesus and cats, remeber to talk about 1+2=3`
  and a nominal budget of 500 new tokens. The spelling is preserved to match
  the scripts. Actual output lengths varied with the engine's token limit,
  tokenizer, chat template, and end-of-sequence behavior.
- v7, llama.cpp, and OpenVINO used greedy decoding with temperature 0. The
  recorded MNN configuration used its sampler at temperature 0.8.
- Each engine used its own weight format and reported the rate for the tokens
  it generated. The figures measure decode throughput. Prefill throughput and
  total response time are separate metrics. The short Phi-3-mini turn typically
  generated about eight tokens before reaching the end-of-sequence token,
  making that workload especially sensitive to round-to-round timing variation.

| Engine | Recorded version and configuration |
| --- | --- |
| v7 | Development build identified as 2.33.0 in the original benchmark overview. Native Q8 weights in MicroDB, five workers plus the calling thread across six physical cores, with continuous spinning during inference. See the retest notes for changes to the tuner settings. |
| llama.cpp | b12179 Windows CPU x64 build, GGUF Q8_0 weights, six threads (`-t 6`), CPU only (`-ngl 0`), single turn (`-st`). |
| MNN | 3.6.2, statically built from source with MSVC, 8-bit weights exported with `llmexport`, and four threads (`thread_num` set to 4). |
| OpenVINO | 2026.4.1, per-channel asymmetric int8 weights exported with NNCF, with the CPU plugin capped at AVX2 (`OV_CPU_MAX_ISA=AVX2`). |
| ONNX Runtime GenAI (earlier runs) | ONNX Runtime 1.31.1 and GenAI 0.19.0, CPU execution provider with default thread settings, and int4 weights (`MatMulNBits`). |

## Four-model baseline and retests

The November 8 baseline ran all 32 engine and workload combinations in the
order v7, OpenVINO, llama.cpp, and MNN. The Qwen3 results in the selected
comparison come from this run. The original Phi-3-mini and MiniCPM5-1B
results appear below for comparison with the retests.

| Model | Workload | v7 | llama.cpp | MNN | OpenVINO |
| --- | --- | ---: | ---: | ---: | ---: |
| Phi-3-mini | Short chat | 9.0 | 9.6 | 8.7 | 8.9 |
| Phi-3-mini | Poem | 8.7 | 9.2 | 8.2 | 8.1 |
| MiniCPM5-1B | Short chat | 27.9 | 37.9 | 34.8 | 32.2 |
| MiniCPM5-1B | Poem | 28.4 | 36.6 | 33.9 | 29.9 |

### MiniCPM5-1B retests

The MiniCPM5-1B baseline followed a sustained Phi-3-mini workload. The later
retests changed the v7 dispatch configurations, and the second also used a
rebuilt database with a tuner delay of 100. The recorded conditions differed
between the baseline and the retests.

| November 8 run | v7 short | v7 poem | llama.cpp short | llama.cpp poem |
| --- | ---: | ---: | ---: | ---: |
| 17:40, first retest | 37.3 | 36.3 | 37.5 | 36.3 |
| 18:55, second retest | 38.1 | 37.1 | 37.7 | 36.6 |

Reported v7 prefill throughput rose from 67–78 tok/s before the first retest
to 97–104 tok/s in that retest. It was 96–110 tok/s in the second retest.

### Phi-3-mini retests

| Run | v7 short | v7 poem | llama.cpp short | llama.cpp poem |
| --- | ---: | ---: | ---: | ---: |
| November 8, 18:10 | 8.7 | 8.5 | 9.4 | 9.1 |
| November 9, 09:08, tuner measuring | 8.6 | 8.55 | — | — |
| November 9, 09:38, tuner quiet | 9.1 | 8.7 | 9.4 | 8.9 |

The v7 cycle tuner recorded 52,061 measurement episodes during the 09:08 poem
run. The 09:38 retest measured the tuned configuration with those measurements
suppressed. The recorded prefill ranges for Phi-3-mini were 10–14 tok/s for
v7, 22–29 tok/s for MNN, and 20–41 tok/s for llama.cpp. Those ranges describe
prompt processing. The decode comparison still shows a small lead for llama.cpp
in both workloads.

## Earlier Qwen3-0.6B comparison, including int4

The November 6 run used the same laptop and three rounds per engine. llama.cpp
used 12 threads in this run. The int4 ONNX Runtime GenAI leg used a different
weight precision from the four 8-bit legs.

| Engine | Short R1 | Short R2 | Short R3 | Short steady state | Poem R1 | Poem R2 | Poem R3 | Poem mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime GenAI (int4) | 64.7 | 65.5 | 65.6 | **65.5** | 58.1 | 58.7 | 58.4 | **58.4** |
| v7 (Q8) | 50.1 | 50.5 | 51.0 | 50.5 | 46.4 | 47.7 | 47.4 | 47.2 |
| MNN (int8) | 45.2 | 45.4 | 45.2 | 45.2 | 40.3 | 40.2 | 41.1 | 40.5 |
| llama.cpp (Q8_0) | 41.4 | 41.5 | 41.3 | 41.4 | 39.0 | 38.9 | 36.4 | 38.1 |
| OpenVINO (int8, AVX2 cap) | 32.9 | 38.5 | 38.7 | 38.6 | 37.4 | 35.9 | 37.6 | 37.0 |

ONNX Runtime GenAI had the highest reported decode throughput in this
mixed-precision comparison. In the poem workload, it generated 568 tokens,
v7 and llama.cpp generated 500 each, MNN generated 512, and OpenVINO reached
the end-of-sequence token after 451–478 tokens.

In the short workload, llama.cpp reported 240–317 tok/s for prefill and MNN
reported 166–200 tok/s. v7 processed 17 prompt tokens in about 81 ms in the
steady state, equivalent to about 210 tok/s. The recorded OpenVINO and ONNX
Runtime round wall times included prefill.

An earlier run under thermal throttling recorded short-chat rates of 47.6
tok/s for v7, 49 for ONNX Runtime GenAI, 30.9 for llama.cpp, and 29 for
OpenVINO. The corresponding poem rates were 46.1, 40.9, 28.8, and 27.9 tok/s.
MNN joined the comparison after that run.

On November 7, an interleaved thread-count test selected six threads for
llama.cpp. The subsequent Qwen3-0.6B short-chat run reported 50.8 tok/s for
v7, 49.7 for llama.cpp, 45.8 for MNN, 38.6 for OpenVINO, and 64.4 for ONNX
Runtime GenAI. MNN kept four threads after testing four, six, and twelve.
OpenVINO kept its default scheduling after a physical-core-only probe produced
similar throughput. ONNX Runtime GenAI kept the default CPU thread settings.

### Earlier three-round measurements for the other models

Each cell lists the three decode rates in tok/s. The v7 databases used
`-tuner-delay 100 -tuner-delay-large 100` for these runs. The cycle tuner
remained active.

| Model | Workload | v7 | llama.cpp | MNN | OpenVINO | ONNX Runtime GenAI (int4) |
| --- | --- | --- | --- | --- | --- | --- |
| Phi-3-mini | Short chat | 7.56 / 8.85 / 8.82 | 8.5 / 9.0 / 8.9 | 8.25 / 8.11 / 8.19 | 5.1 / 8.4 / 8.4 | 9.6 / 8.9 / 10.0 |
| Phi-3-mini | Poem | 7.35 / 8.46 / 8.45 | 8.6 / 8.6 / 8.6 | 7.81 / 7.78 / 7.76 | 7.0 / 7.8 / 7.8 | 12.9 / 13.0 / 13.0 |
| MiniCPM5-1B | Short chat | 33.02 / 32.15 / 31.57 | 36.3 / 36.3 / 36.2 | 32.92 / 33.34 / 32.98 | 26.3 / 30.4 / 30.5 | Unavailable |
| MiniCPM5-1B | Poem | 31.82 / 30.56 / 29.88 | 35.0 / 35.0 / 35.0 | 32.18 / 32.01 / 32.05 | 27.0 / 28.3 / 28.3 | Unavailable |
| Qwen3-1.7B | Short chat | 18.26 / 19.09 / 18.96 | 18.8 / 18.9 / 18.9 | 17.22 / 17.17 / 17.27 | 13.5 / 17.1 / 17.1 | 27.3 / 27.7 / 27.7 |
| Qwen3-1.7B | Poem | 17.96 / 18.64 / 18.59 | 18.2 / 18.2 / 18.1 | 16.64 / 16.85 / 16.71 | 14.8 / 16.8 / 16.7 | 25.6 / 26.1 / 26.1 |

The MiniCPM5-1B leg was unavailable in ONNX Runtime GenAI 0.19.0 because its
tokenizer rejected the model's pre-tokenizer expression `\p{N}{1,3}` with
`regex_error: error_badbrace`.

## Running the comparisons

The release includes the following three-round Windows batch files. Run the
chosen file from the repository root with the required engines, models, and
helper scripts installed in the expected directories.

| Model | Short chat | Poem |
| --- | --- | --- |
| Qwen3-0.6B | `bench_all_engines.bat` | `bench_all_engines_poem.bat` |
| Qwen3-1.7B | `bench_all_engines_qwen17b.bat` | `bench_all_engines_poem_qwen17b.bat` |
| Phi-3-mini | `bench_all_engines_phi3.bat` | `bench_all_engines_poem_phi3.bat` |
| MiniCPM5-1B | `bench_all_engines_minicpm5.bat` | `bench_all_engines_poem_minicpm5.bat` |

The five-round harnesses, third-party helper scripts, and full raw logs are
development-tree material. The baseline overnight log, the MiniCPM5 retest log
and the Phi-3 retest log are kept in the development tree and are not part of
this release. The release batch files exercise the
corresponding workloads with their three-round settings. For an equivalent
five-round comparison, use the engine settings above and record both the
actual number of tokens generated and the decoding time for each round.
Keep the tuner in the same measurement state when comparing dispatch
configurations.

The v7 DLL is `bin\v7_threading.dll`. For Qwen3-0.6B, the v7 database is
`weights\databases\qwen3_0.6b\lz4ht.db`, the OpenVINO model directory is
`openvino\models\qwen3_06b_int8`, the llama.cpp model is
`llamacpp\models\Qwen3-0.6B-Q8_0.gguf`, the MNN model directory is
`mnn\models\qwen3_06b_int8`, and the ONNX model directory is
`onnxrt\models\qwen3_06b_int4`. The recorded Python 3.12 environment for
OpenVINO and ONNX Runtime is under `openvino\pyenv`.

The recorded Windows builds are available from the
[OpenVINO 2026.4.1 archive](https://storage.openvinotoolkit.org/production/repositories/openvino/packages/2026.4.1/windows/openvino_toolkit_windows_2026.4.1.22982.07f9c262b05_x86_64.zip)
and the [llama.cpp b12179 release](https://github.com/ggml-org/llama.cpp/releases/download/b12179/llama-b12179-bin-win-cpu-x64.zip).

### Preparing third-party models

- For llama.cpp, use the official Q8_0 exports from `Qwen/Qwen3-0.6B-GGUF`,
  `Qwen/Qwen3-1.7B-GGUF`, and `openbmb/MiniCPM5-1B-GGUF`. For Phi-3-mini,
  run b12179's `convert_hf_to_gguf.py --outtype q8_0` against the local
  Hugging Face checkpoint, including its `tokenizer.model` file. The release
  includes `convert_phi3_gguf.bat`.
- For OpenVINO, run `optimum-cli export openvino --model <checkpoint> --task text-generation-with-past --weight-format int8 <output-directory>`.
- For ONNX Runtime GenAI, run `python -m onnxruntime_genai.models.builder -m <checkpoint> -e cpu -p int4 -o <output-directory>`.
- For MNN, build the runtime and converter from source with MSVC and use
  `llmexport.py --path <checkpoint> --export mnn --quant_bit 8 --dst_path <output-directory> --mnnconvert <path-to-MNNConvert.exe>`.
  The recorded Phi-3-mini MNN export used a local checkpoint configuration
  with the stale `auto_map` entry removed.

The MNN build used these options.

```bat
cmake -S mnn\MNN-3.6.2 -B mnn\build -A x64 -DMNN_BUILD_LLM=ON ^
  -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_AVX512=OFF ^
  -DMNN_BUILD_TOOLS=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TFLITE=OFF ^
  -DMNN_TF_SUPPORT=OFF -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_WIN_RUNTIME_MT=ON ^
  -DMNN_BUILD_OPENCV=OFF -DMNN_BUILD_DEMO=OFF -DMNN_EVALUATION=OFF ^
  -DMNN_BUILD_BENCHMARK=OFF -DMNN_BUILD_TEST=OFF
cmake --build mnn\build --config Release --target llm_demo -j 12
cmake --build mnn\build --config Release --target llm_bench -j 12
cmake --build mnn\build --config Release --target MNNConvert -j 12
```
