# Models and Vision-Language Inference

V7 supports a defined set of decoder-only text models and vision-language configurations. Compatibility
depends on the implemented operations, the model's configuration, and the format of the database
prepared from its checkpoint. Sharing an architecture name does not automatically guarantee
compatibility; differences in attention, positional encoding, projection, or tokenizer behavior may
require additional support.

This guide starts with the recorded compatibility information and then explains the text-decoder and
vision-encoder paths in enough detail to help maintainers evaluate new checkpoints. The tables separate
verified configurations from plausible but untested ones. Implementation notes describe the actual data
layouts and execution branches, while model preparation and build instructions remain in the
getting-started manual.

## Contents

- [Model support and configuration](#model-support-and-configuration)
- [Vision-language execution](#vision-language-execution)

---

## Model support and configuration

A checkpoint is compatible with V7 only when the runtime implements its required operations
and the exported model data matches the expected database layout. The model family name is useful for locating an
implementation path, but it does not by itself establish compatibility. Attention structure,
normalization, positional encoding, tokenizer rules, and the projection layers must all agree with the
configuration loaded from `lz4ht.db`.

This chapter distinguishes models that have been exercised by the available validation programs from
configurations inferred from the upstream architecture. The compatibility matrix records the dimensions
and optional operations associated with each configuration. Later sections explain how the common text
decoder and vision path use those settings. The model exporters and database builder are described where
their output determines the runtime contract; build commands and test procedures belong in the
[getting-started manual](getting_started.md).

The V7 repository includes exporters and database-building utilities, but it does not redistribute the
upstream model weights. The fetch scripts obtain the checkpoints from their respective sources. Each
model retains its own license, which should be reviewed before redistribution or deployment. The model
families listed here include checkpoints distributed under Apache-2.0 or MIT terms, as well as Meta
Llama 3.2, which uses the Meta Llama 3.2 Community License and requires acceptance of its terms through
the Meta distribution portal before fetching the weights. The associated attribution requirements and
acceptable-use policy also apply to artifacts produced from those weights.

### Recorded verification snapshot

The verification matrix records particular model exports and their test results. A working forward pass
for one checkpoint does not automatically establish the behavior of every model in the same
architectural family. Thinking-block-based scripting also depends on the checkpoint's chat template and
whether that checkpoint actually emits a closing thinking marker. For models without this marker,
applications must choose a different trigger policy; mid-generation insertion is not automatically
activated by model loading.

| Metric | State |
| --- | --- |
| Verification | The latest verification log records the validator and smoke-test results; this table is a historical snapshot, not a live status. |
| Comment checker | **0 reported violations in the recorded verification run** across the ~140 `.inc`/`.asm`/`.ini` files in `src/asm` |
| Vision smoke | Baseline hash stable (`vision/vision_baseline.txt`) |
| Dispatch perf | Q8 1024×1024 = 382–426 GB/s, F32 = 176–197 GB/s (cache-resident kernel benchmark; not DRAM bandwidth or tokens/s) |
| Recorded regression check | Commit `c6bfba5` passed the recorded kernel regression check. |

#### The support matrix

The table below records the configuration of each tested model. The values were compared with the
contents of the corresponding `lz4ht.db` database using `mdb_query_key`:

| Model | Params | dim | L | H | KV | Vocab | head_dim | RoPE | QK-Norm | Tokenizer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 0.6B | 1024 | 28 | 16 | 8 | 151936 | 128 | 1000000 | Yes | BPE |
| Qwen3-1.7B | 1.7B | 2048 | 28 | 16 | 8 | 151936 | 128 | 1000000 | Yes | BPE |
| Qwen2.5-0.5B | 0.5B | 896 | 24 | 14 | 2 | 151936 | 64 | 1000000 | No | BPE |
| Llama-3.2-1B | 1.0B | 2048 | 16 | 32 | 8 | 128256 | 64 | 500000 | No | BPE |
| MiniCPM5-1B | 1.0B | 1536 | 24 | 16 | 2 | 130560 | 128 | 5000000 | No | BPE |
| Phi-3-mini-4k | 3.8B | 3072 | 32 | 32 | 32 | 32064 | 96 | 10000 | No | BPE |
| TinyLlama-1.1B | 1.1B | 2048 | 22 | 32 | 4 | 32000 | 64 | 10000 | No | SentencePiece |
| SmolLM2-360M | 0.36B | 960 | 32 | 15 | 5 | 49152 | 64 | 100000 | No | BPE |
| SmolLM2-135M | 0.135B | 576 | 30 | 9 | 3 | 49152 | 64 | 100000 | No | BPE |
| Qwen3-8B | 8.0B | 4096 | 36 | 32 | 8 | 151936 | 128 | 1000000 | Yes | BPE |
| Qwen3-VL-4B | 4.0B | 2560 | 36 | 32 | 8 | 151936 | 128 | 5000000 | Yes | BPE |
| LLaVA-OneVision-Qwen2-0.5B | 0.5B text | 896 | 24 | 14 | 2 | 152000 | 64 | 1000000 | No | BPE |
| SmolVLM-256M | 0.25B total | 576 | 30 | 9 | 3 | 49152 | 64 | 100000 | No | BPE |

The Qwen3-8B row comes from the HuggingFace config (its 16 GB database was not built in the tree this
matrix was recorded in); every other row was cross-checked against the actual `lz4ht.db` contents via
`mdb_query_key`. The recorded model tests covered both the F32 and Q8 execution paths. The three VLMs
also carry a ViT vision tower, described in [Vision-language execution](#vision-language-execution).

The [CPU inference benchmark](engine_benchmark.md) compares decode throughput for
Qwen3-0.6B, Qwen3-1.7B, Phi-3-mini, and MiniCPM5-1B on a Ryzen 5 7430U.

### Recorded vision turn latency

The v7 2.35.0 release README reports the wall time for a Qwen3-VL turn with a
640 × 480 image and a 200-token answer. The timing covers both ViT encoding
and the answer. The output was byte-identical before and after the changes.

| Mode | Before | v7 2.35.0 |
| --- | ---: | ---: |
| F32 | 170 s | 136 s |
| Q8 | 187 s | 136 s |

---

### Model execution architecture

The runtime uses a multithreaded AVX2 execution system for both text decoding and the supported vision
pipelines. These paths share the dispatcher and worker infrastructure, while their model-specific
operations and configuration data remain distinct. The following reference identifies the execution
branches and the values supplied by the database, which determine how the common code handles each
supported architecture.

```
                  +------------------------------------------+
                  |     microdb (lz4ht.db)                   |
                  |  Built by build_db.exe from:             |
                  |   - <model>.bin       (text weights)     |
                  |   - <model>.tokenizer (BPE merges/vocab) |
                  |   - <model>.template* (chat templates)   |
                  |   - <model>.vision    (ViT + projector)  |
                  |   - <model>.proj      (separate MLP)     |
                  |  Each file becomes one __sys_* DB key.   |
                  +------------------------------------------+
                                    |
                            v7_initialize_inference
                                    |
           +------------------------+------------------------+
           |                        |                        |
           |                        |                        |
    +--------------+       +--------------+        +------------------+
    |  LLM stack   |       |  ViT stack   |        |  Tokenizer +     |
    |  (text gen)  |       |  (vision)    |        |  Sampler         |
    +--------------+       +--------------+        +------------------+
           |                      |
           |  vision embeddings   |
           |  injected here       |
           +----------------------+
                      |
               v7_chat / v7_chat_image
                      |
               text output
```

The vision encoder and language decoder share the worker pool. Dispatch requests pass through the
`v7d_*` entries, `DISPATCH_EXECUTE`, and `process_job` before reaching the selected kernel. Within
`process_job`, the `JOB_MODE` value distinguishes vision operations from language-model operations.

---

### Model data preparation

The inference DLL does not read upstream checkpoint formats directly. Before an application can use a
model, the export tools convert its weights, tokenizer resources, prompt templates, and configuration
into the representation expected by V7. The database builder then collects those resources in MicroDB.
By the time inference begins, the runtime can work with the prepared entries rather than interpreting
checkpoint metadata or converting tensor layouts during generation.

This division is important for compatibility. Supporting a new checkpoint requires a correct mapping
from its source configuration to the database fields and execution paths that V7 already implements;
it does not necessarily require a new inference kernel. The structure of the prepared database, rather
than the upstream file layout, forms the binary contract that the DLL uses when initializing the
model.

```
HF safetensors
  |
  +-- tools/export.py        ->  <model>.bin         (text LLM weights)
  |                              <model>.tokenizer   (BPE or SP)
  |                              <model>.template*   (chat templates)
  |
  +-- tools/export_vlm.py    ->  <model>.bin         (text weights, remapped prefix)
  |                              <model>.vision      (256B header + VIT1 blob)
  |                              <model>.proj        (VITP projector blob, SigLIP only)
  |
   +-- build_db.exe           ->  lz4ht.db
         (db_build_tool.c)       reads the model files above + JSON headers
                                  |
                                  +- __sys_config           (16 int32s)
                                  +- __sys_mem_layout       (RunState offsets)
                                  +- __sys_model_weights    (blob past 256B header)
                                  +- __sys_weight_meta      (16 entries, 24B each)
                                  +- __sys_rms_eps          (1 float)
                                  +- __sys_rope_type        (0=plain, 1=llama3)
                                  +- __sys_rope_*           (llama3 scaling params)
                                  +- __sys_tokenizer_hdr    (bos/eos/max_token_len)
                                  +- __sys_vocab_bulk       (packed vocab strings)
                                  +- __sys_merge_scores     (BPE merge priorities)
                                  +- __sys_template*        (chat templates)
                                   +- __sys_vision_config    (VCFG, 93 qwords, C-precomputed)
                                  +- __sys_vit_weights      (VIT1 blob)
                                  +- __sys_vit_projector    (VITP blob, SigLIP only)
                                  +- __sys_vision_xy_res    (target image size)
                                  +- __sys_vision_prompt_fmt(prefix/marker/suffix strings)
```

The vision configuration is prepared before inference. `build_vision_cfg()` in `db_build_tool.c`
calculates the model-dependent byte strides, derived dimensions, scale factors, and IEEE bit patterns,
then stores them in `__sys_vision_config`. The assembly implementation reads the prepared values through
RIP-relative memory operands instead of recalculating them during image processing. When a new vision
model uses an execution path already supported by V7, its exporter can supply the appropriate VHDR
fields through `export_vlm.py`, after which the `.vision` file and database must be rebuilt. A model
that requires operations not supported by the existing execution paths will still need corresponding
implementation work.

#### 2.1 The 36-qword VHDR raw header

Each `.vision` file begins with a 288-byte header containing 36 eight-byte fields. The exporter writes
those fields through `export_vlm.py:write_vision_file()`, and the database builder reads them through
`db_build_tool.c:store_vision_db()`. Their slot indices are defined below:

| Slot | Name | Qwen3-VL | LLaVA-OV |
| --- | --- | --- | --- |
| 0 | vit_hidden | 1024 | 1152 |
| 1 | vit_layers | 24 | 26 |
| 2 | vit_heads | 16 | 16 |
| 3 | vit_patch | 16 | 14 |
| 4 | image_size | 0 (grid) | 384 |
| 5 | vit_intermediate | 4096 | 4304 |
| 6 | vit_norm_type | 0 (RMS) | 1 (LayerNorm) |
| 7 | vit_ffn_type | 0 (SwiGLU) | 1 (GELU MLP) |
| 8 | vit_has_cls | 0 | 0 |
| 9 | vit_feature_select | 1 (full) | 1 (full) |
| 10 | proj_hidden | 2560 | 896 |
| 11 | proj_layers | 2 | 2 |
| 12 | proj_activation | 1 (GELU) | 1 (GELU) |
| 13 | vit_depth_taps | 3 | 0 |
| 14 | vit_head_dim | 64 | 72 |
| 15 | vit_pos_embed_type | 0 (bilinear) | 1 (additive) |
| 16 | image_token_id | 151655 | 151646 |
| 17 | vision_start_id | 151652 | 0 |
| 18 | vision_end_id | 151653 | 0 |
| 19 | has_deepstack | 1 | 0 |
| 20 | merge | 2 | 1 |
| 21 | temporal | 2 | 1 |
| 22 | num_pos | 2304 | 729 |
| 23 | ds_tap0 | 5 | (n/a) |
| 24 | ds_tap1 | 11 | (n/a) |
| 25 | ds_tap2 | 17 | (n/a) |
| 26 | rope_theta | 10000 | 10000 (unused) |
| 27 | use_rope | 1 | 0 |
| 28 | proj_in_dim | 0 | 1152 |
| 29 | proj_out_dim | 0 | 896 |
| 30 | proj_inter | 0 | 896 |
| 31 | proj_type | 0 (merger) | 1 (MLP) |
| 32 | ps_factor | 0 | 0 (SmolVLM-256M: 4) |
| 33 | mrope_bound | 0 | 0 |
| 34 | anyres_enabled | 0 | 1 |
| 35 | ln_eps | 1e-6 bits | 1e-5 bits |

SmolVLM-256M uses the same SigLIP ViT as LLaVA-OV (same slots 0-30) but `VHDR_PROJ_TYPE=2`
(pixel-shuffle projector) in slot 31, with `VHDR_PS_FACTOR=4` in slot 32. LLaVA-OneVision sets
`VHDR_ANYRES_ENABLED=1` in slot 34 (the multi-tile anyres path); the other two VLMs stay single-tile.

#### 2.1a SmolVLM pixel-shuffle projector type

SmolVLM-256M uses `VHDR_PROJ_TYPE=2` (pixel-shuffle), which differs from both Qwen3-VL (PROJ_TYPE=0,
merger-as-projector) and LLaVA-OV (PROJ_TYPE=1, plain MLP). The pixel-shuffle path rearranges ViT output
tokens via spatial unflatten -> pixel-shuffle (2- upsample via depth-to-space) -> reflatten -> linear
chain before the MLP, producing fewer output tokens than the raw patch count. This exercises a third
projector code path in `vit_forward.inc` that the other two VLMs do not.

#### 2.2 The 93-qword VCFG expanded struct

`build_vision_cfg()` reads the 36 fields in the VHDR header and constructs the 744-byte VCFG structure,
which contains 93 eight-byte fields. The database builder stores that structure under
`__sys_vision_config`. Its fields are organized as follows:

- **Raw header reconstructed** (slots 0-60, byte offsets): `image_token_id`,
  `vision_start_id`, `vision_end_id`, `has_deepstack`, etc.
- **All derived products** precomputed in C: `hidden*4`, `qkv_dim`,
  `head_dim`, `head_dim/2`, `grid_side = isqrt(num_pos)`, `merge*merge`,
  `patch_dim = (3*temporal*patch^2 + 15) & ~15` (padded for AVX2),
  `packed_dim = hidden * merge_unit`, every per-layer stride product
  (`depth*qkv_dim*hidden*4`, etc.), every merger/DeepStack stride.
- **IEEE bit patterns** precomputed in C: `attn_scale = 1/sqrt(head_dim)`
  as raw bits, `grid_minus1_f` as raw bits, `log2(rope_theta)` as raw bits.

The assembly ViT path obtains its model-dependent configuration from this structure and the auxiliary
database entries described below. The following sections identify those inputs and explain how they
affect execution.

#### 2.3 Auxiliary vision DB keys

In addition to the VCFG structure, three vision database keys supply the values needed for
model-dependent processing. The assembly code reads these values without branching on the model name:

| Key | Bytes | Source | Purpose |
| --- | --- | --- | --- |
| `__sys_vision_xy_res` | 16 | `tools/db_build_tool.c` (from VHDR slot 4) | `[target_w, target_h]` - SigLIP=384x384; Qwen3-VL computes its native grid instead: isqrt(num_pos) x patch_size (48 x 16 = 768 for the 4B) |
| `__sys_vision_prompt_fmt` | ~30 | `tools/db_build_tool.c` | `prefix\0marker\0suffix\0` - Qwen3-VL: `<\|vision_start\|>\0<\|image_pad\|>\0<\|vision_end\|>\0` / LLaVA-OV: `\0<image>\0\n\0` |
| `__sys_vit_projector` | varies | `tools\export_vlm.py` (VITP blob) | Separate MLP projector weights (SigLIP only; absent for Qwen3-VL merger path) |

The vision prompt format is supplied by the database rather than selected through model-name checks in
assembly. During initialization, `chat_image.inc` reads three pointers from the stored format and uses
the corresponding strings when constructing the text-side vision prompt.

---

The compatibility table answers which configurations have been tested; the execution-path descriptions
explain why those configurations can share the same decoder. The database selects the operations needed
by each model family, while the native code follows a common transformer sequence. The discussion below
separates these configurable differences from the data layouts and specialized kernels used to implement
them.

### Text-model execution paths

The text decoder uses configuration values from the prepared database to select the operations
appropriate to the model family. Differences include normalization, attention settings, positional
encoding, projection dimensions, and optional biases. The source-level descriptions below show how those
values influence the common forward path. A supported setting is defined by an implemented branch and a
compatible database layout, not merely by the model's name.

#### 3.1 Core architecture: Llama-style decoder

The forward and prefill implementations in `forward_f32.inc`, `forward_q8.inc`, `prefill_f32.inc`, and
`prefill_q8.inc` follow a pre-normalized decoder-only transformer architecture. Persistent key/value
rows are stored as FP16 in separate allocated buffers (`inf_state.inc`, `kv_f16c.inc`); computations
use FP32 staging and conversion in the forward/attention paths. For a context of length `seq_len`,
the two KV buffers together require `4 * n_layers * seq_len * n_kv_heads * head_dim` bytes. Supported text models
differ in their configuration and selected operations, but they use the same overall processing
sequence.

#### 3.2 Feature matrix - what the engine can turn on/off

The features in the following table are selected using values loaded from the MicroDB configuration and
dedicated database keys during initialization. The relevant reads occur in the third initialization
stage in `inf_microdb.inc`. Switching between supported configurations does not require recompiling the
DLL.

| Feature | DB source | ASM field | Values | Models that use it |
| --- | --- | --- | --- | --- |
| **Model dimension** (`dim`) | `config` +8 | `CFG_DIM` | any | All |
| **FFN intermediate** (`hidden_dim`) | `config` +12 | `CFG_HIDDEN_DIM` | any | All |
| **Layer count** (`n_layers`) | `config` +16 | `CFG_N_LAYERS` | any | All |
| **Query heads** (`n_heads`) | `config` +20 | `CFG_N_HEADS` | any | All |
| **KV heads** (`n_kv_heads`) | `config` +24 | `CFG_N_KV_HEADS` | = n_heads (GQA) | All |
| **head_dim** | `config` +36 | `CFG_HEAD_DIM` | 64, 96, 128 tested | All |
| **Vocab size** | `config` +28 | `CFG_VOCAB_SIZE` | any | All |
| **Sequence length** | `config` +32 | `CFG_SEQ_LEN` | any | All |
| **Tied embeddings** | `config` +40 | `CFG_SHARED_CLASSIFIER` | 0/1 | Qwen3-0.6B, SmolLM2, LLaVA |
| **QK-normalization** | `config` +48 | `CFG_HAS_QK_NORM` | 0/1 | Qwen3 family only |
| **RoPE base frequency** | `config` +52 | `CFG_ROPE_BASE` | any float | All (per-model) |
| **RMSNorm epsilon** | `__sys_rms_eps` key | `g_rms_eps` | any float | All (per-model: 1e-5, 1e-6) |
| **RoPE scaling type** | `__sys_rope_type` key | stack local | 0=plain, 1=llama3 | Llama-3.2 (type 1) |
| **RoPE llama3 factor** | `__sys_rope_factor` key | stack local | float (32.0) | Llama-3.2 |
| **QKV attention biases** | weights blob (after k_norm) | `g_qkv_bias` | NULL or ptr | Qwen2 / Qwen2.5 / LLaVA-OV |
| **Quantization group size** | `config` +44 | `CFG_GROUP_SIZE` | 64 | All Q8 models |
| **Vision token injection** | `__sys_vision_config` | `g_image_token_id` | 0=off, else token id | Vision models only |
| **DeepStack feature add** | `__sys_vision_config` | `g_cfg_has_deepstack` | 0/1 | Qwen3-VL only |

#### 3.3 Code paths inside the forward pass

A per-token call to `forward_f32` or `forward_q8` executes the following stages in order:

```
1. Embedding lookup (token_embedding_table[token])
   +- IF g_image_token_id == token: override with vision embedding row
      (g_vision_embeddings[consumed++])
2. set_rope(pos, head_dim) , configure RoPE tables once for all layers
3. FOR each layer (0..n_layers-1):
   a. RMSNorm(x += xb, rms_att_weight[l], dim)     [v7s_rmsnorm, not dispatched]
   b. Fused Q:  matmul + optional QK-norm + RoPE + optional Q-bias  [v7d_fused_f32]
   c. Fused K:  matmul + optional QK-norm + RoPE + optional K-bias  [v7d_fused_f32]
   d. V:        matmul + optional V-bias (inline AVX2 add loop)     [v7d_f32 / v7d_q8]
   e. Attention: causal softmax(QK^T) @ V  [v7d_attention, dispatched per head]
   f. Output proj: matmul(xb, wo[l])       [v7d_f32 / v7d_q8]
   g. Residual: x += hb2                   [v7s_add_inplace]
   h. FFN RMSNorm(x * xb, rms_ffn_weight[l], dim)
   i. SwiGLU:  silu(w1*xb) * (w3*xb)       [v7d_swiglu, fused gate+up+down]
   j. W2: matmul(hb, w2[l])                [v7d_f32 / v7d_q8]
   k. Residual: x += xb
   l. IF layer < 3 AND token == image_token AND has_deepstack:
      x += g_deepstack_features[layer][row]  [v7s_add_inplace]
4. Final RMSNorm(x * x, rms_final_weight)
5. Classifier: matmul(x, wcls) -> logits   [v7d_f32 / v7d_q8]
```

In the Q8 path the Q/K projections fuse the per-head RMSNorm+RoPE into the matmul epilogue
(`v7d_fused_f32` runs for both quantization modes, the F32 activation it produces is requantized to int8
per 64-group before the fused kernel hands rows to the Q8 worker in forward_q8), and the WO/W2 inputs
are quantize-once (`v7d_q8_fused` fills the int8 scratch + scale buffers the Q8xQ8 worker reads).

The operations that benefit from parallel execution pass through `DISPATCH_EXECUTE`. Standalone F32 and
Q8 matrix multiplication use `v7d_f32` and `v7d_q8` for the value projection, output projection, second
feed-forward projection, and classifier. The fused Q and K projections use `v7d_fused_f32`, combining
matrix multiplication with RMSNorm, RoPE, and an optional bias. The dispatcher also handles `v7d_swiglu`
for the gate and up projections with SiLU and multiplication, `v7d_attention` for causal attention over
the KV cache with grouped-query attention, and `v7d_memcpy` for the embedding copy.

Two smaller operations remain on the calling thread. `v7s_rmsnorm` performs a reduction across the
relevant vector and is not divided into dispatch jobs. `v7s_add_inplace` carries out an element-wise
addition without submitting a separate parallel workload. This distinction is part of the current
implementation, rather than an application-facing choice.

#### 3.4 RoPE modes (data-driven)

V7 selects its rotary-position-embedding implementation during initialization using the
`__sys_rope_type` database value. The supported modes are:

| Type | Mode | Formula | Models |
| --- | --- | --- | --- |
| 0 | Plain | `freq[j] = base^(-j/half)` | Qwen, Phi-3, TinyLlama, SmolLM2, MiniCPM5 |
| 1 | Llama3 scaled | Wavelength-based freq scaling with smooth interpolation between high/low bands | Llama-3.2, Llama-3.1 |

The llama3 scaling (`fused_f32_kernel.inc`) reads four additional DB keys (`__sys_rope_factor`,
`__sys_rope_high_freq_factor`, `__sys_rope_low_freq_factor`, `__sys_rope_orig_max_pos` ) and
precomputes a per-frequency `freq[j]` table at init using x87 `fyl2x` /`f2xm1`/`fscale` for the power
computation; this is the exact HF llama3 RoPE formula.

#### 3.5 F32 vs Q8 dual path

The same decoder supports both full-precision and quantized execution. The `ctx->quantize` field
determines which forward path is used for a call, allowing applications to choose the documented
precision mode without selecting a different model architecture. The distinction affects the stored
weights, activation representation, and matrix-multiplication kernels, as summarized below:

| Path | Weights | Activation | Kernels | Use case |
| --- | --- | --- | --- | --- |
| F32, `-q8 0` | F32 | F32 | `v7d_f32`, `v7d_fused_f32`, `v7d_swiglu` | maximum precision |
| Q8, `-q8 1` | int8 with float scales | int8 per 64-group with float scales | `v7d_q8`, `v7d_q8_fused`, `v7d_swiglu_q8` | roughly 4x smaller weights and a faster Q8xQ8 matmul |

Both paths share the same attention, RMSNorm, add_inplace, and RoPE code. The Q8 path dequantizes
weights during execution inside the matmul inner loop using AVX2 `vpmovsxbd`, `vcvtdq2ps` and `vmulps`
with a broadcast scale, and quantizes activations to int8 per 64-element group (with one float scale per
group) before each Q8xQ8 matmul, the activation quantizer is the same machinery the weights went through
at export time, run live.

#### 3.6 QKV bias support for the Qwen2 family

Some supported models require a bias in their Q, K, or V projections. The fused Q and K kernel obtains
the relevant per-layer pointer from `g_qkv_bias` through the `fused_bias_ptr` global defined in
`v7_data.inc`. The pointer is NULL for the documented Llama, Qwen3, Phi-3, TinyLlama, SmolLM2, and
MiniCPM5 configurations, in which case a `test rax, rax` and conditional branch skip the bias operation.
For Qwen2, Qwen2.5, and the Qwen2 decoder in LLaVA-OneVision, the pointer is non-NULL and the bias is
applied during the projection store. The F32 path uses `vaddss` before `vmovss`, while the value
projection uses an inline AVX2 addition loop.

This conditional path allows the shared decoder to support the bias requirements of Qwen2-family
configurations without adding an unnecessary bias operation to models that do not use one.

#### 3.7 Prefill path

The prefill implementations in `prefill_f32.inc` and `prefill_q8.inc` evaluate the prompt as a batch
rather than making a complete single-token forward call for every input token. Attention remains causal:
each token uses the relevant preceding positions, even though the computation is organized as a batch.
This can reduce the overhead of processing longer prompts before the first output token is sampled.

Prefill reuses the dispatched matrix-multiplication and attention kernels used by the normal decoder.
When the prompt contains images, it substitutes the prepared vision embeddings at the configured
image-token positions, referred to as Hook A. For models with DeepStack, Hook B adds the corresponding
features in layers zero, one, and two. The operation also writes the KV-cache entries required for the
evaluated prompt tokens so that subsequent generation can continue from the correct sequence position.

---

Vision-language inference requires more than appending image bytes to a text prompt. The application
supplies decoded pixel data, the vision encoder transforms it into model-specific embeddings, and the
projector prepares those embeddings for the language decoder. Because the supported vision model
families differ in patching, normalization, positional encoding, and projection, the database provides
additional configuration values that select the appropriate native operations.

### Vision encoder implementation

Vision-language inference adds an image encoder and projection stage before the language decoder
receives the visual embeddings. Image preparation differs between the supported model families, and the
database identifies the dimensions and operations required by each path. This section describes the
encoder structure, dispatched operations, and the transfer of visual features into the text decoder.
Consult the model compatibility table for the configurations that have been verified.

#### 4.1 Architecture: data-driven ViT with VCFG branching

The vision implementation uses one configurable forward path instead of selecting a separate
implementation by model name. The code in `vit_forward.inc`, `vit_kernel.inc`, `vit_dispatch.inc`,
`chat_image.inc`, and the related `vit_*.inc` modules reads `VCFG_*` fields to select the appropriate
normalization, positional encoding, and projection operations.

The database builder prepares model-dependent values through `tools/db_build_tool.c:build_vision_cfg`
and stores them as 93 eight-byte fields in the `g_vit_cfg` BSS block, for a total of 744 bytes. The
assembly routines then read these fields through RIP-relative operands. This keeps configuration
calculations out of the per-operation vision execution path and allows a shared implementation to serve
the verified model configurations.

#### 4.2 Feature matrix - what the ViT stack can turn on/off

| VCFG field | Offset | Purpose | Qwen3-VL | LLaVA-OV (SigLIP) |
| --- | --- | --- | --- | --- |
| `VCFG_HIDDEN` | 64 | ViT hidden dim | 1024 | 1152 |
| `VCFG_INTER` | 80 | FFN intermediate | 4096 | 4304 |
| `VCFG_HEADS` | 104 | Attention heads | 16 | 16 |
| `VCFG_HEAD_DIM` | 112 | Per-head dim | 64 | **72** |
| `VCFG_DEPTH` | 128 | Transformer layers | 24 | 26 |
| `VCFG_PATCH_SIZE` | 192 | Patch size | 16 | **14** |
| `VCFG_MERGE` | 176 | Spatial merge | 2 (2-2) | **1 (none)** |
| `VCFG_USE_ROPE` | 616 | 2D RoPE in ViT | 1 (on) | **0 (off)** |
| `VCFG_POS_EMBED_TYPE` | 584 | Position embedding | 0 (bilinear-grid) | **1 (additive)** |
| `VCFG_N_DEEPSTACK` | 248 | DeepStack tap count (taps at VCFG_DS_TAP0/1/2, offsets 280/288/296) | 3 (5/11/17) | **0** |
| `VCFG_HAS_DEEPSTACK` | (raw 12) | DeepStack text-side adds | 1 | **0** |
| `VCFG_PROJ_IN_DIM` | 624 | Projector input | 0 (merger IS proj) | **1152 (separate MLP)** |
| `VCFG_PROJ_OUT_DIM` | 632 | Projector output | 0 | 896 (text hidden) |
| `VCFG_PROJ_INTER` | 640 | Projector inter | 0 | 896 |
| `VCFG_ATTN_SCALE` | 520 | 1/vhead_dim | 0.125 bits | 0.11785- bits |
| `VCFG_NORM_TYPE` | 528 | Norm type | 0 (RMS - unused; both use LN) | 1 (LayerNorm) |
| `VCFG_FFN_TYPE` | 536 | FFN type | 0 (SwiGLU - unused; both use GELU) | 1 (GELU MLP) |
| `VCFG_HAS_CLS` | 544 | CLS token | 0 | 0 |
| `VCFG_FEATURE_SELECT` | 552 | Feature select | 1 (full patches) | 1 (full) |
| `VCFG_IMAGE_SIZE` | 592 | Image size | 0 (grid) | 384 |
| `VCFG_ROPE_THETA_LOG2` | 608 | log2(theta) bits | log2(10000) | unused |

The listed configuration differences are exercised by the tested LLaVA-OneVision-Qwen2-0.5B path. The
recorded Qwen3-VL-4B baseline did not change in the corresponding regression test.

#### 4.3 ViT forward processing

```
v7s_vit_encode_callback(raw_rgb, w, h,...)
  -
  +- Pixel normalize: data-driven via __sys_vision_xy_res + scale constants
  -     (SigLIP: (pixel/255 - 0.5)/0.5; Qwen3-VL: pixel/127.5 - 1.0)
  +- Patchify: extract patch_size-patch_size patches -> flatten
  +- v7s_vit_forward:
  -   +- Patch embed: v7d_vit_linear(hidden, pixels, pe_w, pe_b, n, H, patch_dim)
  -   +- Position embed (DATA-DRIVEN branch):
  -   -   VCFG_POS_EMBED_TYPE=0: vit_pos_embed (bilinear-grid resample, Qwen3-VL)
  -   -   VCFG_POS_EMBED_TYPE=1: rep movsq (verbatim copy, SigLIP)
  -   +- hidden += pos  [vit_add]
  -   +- IF VCFG_USE_ROPE: vit_rope_build (2D RoPE cos/sin tables)
  -   +- FOR layer 0..VCFG_DEPTH-1:
  -   -   +- LayerNorm(hidden, norm1_g[l], norm1_b[l])  [vit_layernorm]
  -   -   +- QKV: v7d_vit_linear(qkv, norm, qkv_w[l], qkv_b[l])
  -   -   +- IF VCFG_USE_ROPE: vit_rope_apply (rotate Q, K)
  -   -   +- Attention: v7d_vit_attention (16-head bidirectional, no causal mask)
  -   -   -     +-- head_dim=64 path (8 YMM accums, scale 0.125) - Qwen3-VL
  -   -   -     +-- head_dim=72 path (9 YMM accums + tail, scale 0.11785) - SigLIP
  -   -   +- Proj: v7d_vit_linear(norm, attn, proj_w[l], proj_b[l])
  -   -   +- hidden += norm  [residual]
  -   -   +- LayerNorm(hidden, norm2_g[l], norm2_b[l])
  -   -   +- FC1: v7d_vit_linear(mlp, norm, fc1_w[l], fc1_b[l])
  -   -   +- GELU(mlp) in-place  [vit_gelu - tanh approximation]
  -   -   +- FC2: v7d_vit_linear(norm, mlp, fc2_w[l], fc2_b[l])
  -   -   +- hidden += norm  [residual]
  -   -   +- IF layer == ds_tap[0/1/2] AND VCFG_DEPTH_TAPS > 0:
  -   -       DeepStack merger (pack2x2 + LN + FC1 + GELU + FC2 -> out_ds)
  -   +- Post-layer (DATA-DRIVEN branch on VCFG_MERGE):
  -       VCFG_MERGE >= 2 (Qwen3-VL): main merger (LN + pack2x2 + FC1 + GELU + FC2 -> out_main)
  -       VCFG_MERGE == 1 (SigLIP):   copy hidden -> out_main (post_layernorm weights emitted but unused)
  -
  +- IF VCFG_PROJ_IN_DIM > 0 (SigLIP - separate MLP projector):
  -   v7d_vit_linear(proj1, vit_out, W1, b1, n, proj_in, proj_inter); up-project
  -   vit_gelu(proj1, n*proj_inter); GELU
  -   v7d_vit_linear(text_emb, proj1, W2, b2, n, proj_inter, proj_out); down-project
  -
  +- Return embeddings [N - text_dim] + optional DeepStack features
```

#### 4.4 ViT dispatched kernels

| Entry | Kernel | Job mode | What it does |
| --- | --- | --- | --- |
| `v7d_vit_linear` | `v7s_vit_linear_chunk` | 11 | F32-F32 matmul + bias (patch embed, QKV, proj, FC1, FC2, projector) |
| `v7d_vit_attention` | `v7s_vit_attention_chunk` | 12 | Bidirectional attention, one job per head, branches on head_dim (64 or 72) |

The attention kernel (`vit_kernel.inc`) branches on `VCFG_HEAD_DIM`:
- **64** -> 8 YMM accumulators, scale 0.125 (Qwen3-VL path, bit-identical)
- **72** -> 9 YMM accumulators + 8-float tail, scale 0.11785- (SigLIP path, ported from validated attic code)

#### 4.5 ViT inline helpers, not dispatched

| Helper | File | Compute |
| --- | --- | --- |
| `vit_layernorm` | `vit_layernorm.inc` | Standard LayerNorm: mean + var + gamma/beta, epsilon read from `VCFG_LN_EPS` (1e-6 Qwen3-VL, 1e-5 SigLIP), exact 1/std via `vdivss` |
| `vit_gelu` | `vit_gelu.inc` | GELU-tanh, computed as `x * e/(e+1)` with `e = exp(2*sqrt(2/pi)*(x + 0.044715*x^3))` (the 0.5 factors cancel), 8 floats/iter |
| `vit_add` | `vit_add.inc` | Element-wise `dst[i] += src[i]` |
| `vit_pack2x2` | `vit_forward.inc` | Rearrange 4 patches into one packed row (spatial merge 2-2), inlined in the merger path |
| `vit_pos_embed` | `vit_pos_embed.inc` | Bilinear-resample 48-48 learned pos table to grid_h-grid_w |
| `vit_rope_build` | `vit_rope_build.inc` | Build 2D RoPE cos/sin tables via x87 `fsincos` |
| `vit_rope_apply` | `vit_rope_apply.inc` | Rotate Q and K in the interleaved QKV buffer |
| `vit_input_stage` | `vit_input_stage.inc` | Pixel normalize + patchify (data-driven target resolution) |

#### 4.6 Vision -> LLM injection (`chat_image.inc`)

The vision encoder passes its output to the text decoder through the following interface:

1. **ViT encode** -> produces `[N - text_dim]` embeddings (+ optional DeepStack features)
2. **Prompt construction** (DATA-DRIVEN via `__sys_vision_prompt_fmt`):
   - Qwen3-VL: `<\|vision_start\|>` + N - `<\|image_pad\|>` + `<\|vision_end\|>` + user_text
   - LLaVA-OV: N - `<image>` + `\n` + user_text
3. **Tokenize** -> BPE encode the prompt (image markers resolve to their token IDs)
4. **Prefill + generate** -> `v7s_chat` with vision hooks active:
   - **Hook A:** At each image-token position, replace the learned token embedding with the corresponding ViT embedding row (`g_vision_embeddings[consumed++]`)
   - **Hook B:** At layers 0/1/2, add the DeepStack feature row for image tokens (Qwen3-VL only; gated by `g_cfg_has_deepstack`)

#### 4.7 Unsupported ViT operations

The following fields are present in `g_vit_cfg` but are not currently read by the assembly
implementation. They remain part of the data layout for potential additional model configurations:

| Gap | What's missing | What it would unlock |
| --- | --- | --- |
| `VCFG_HAS_CLS` not read | No CLS token prepend/strip | CLIP-ViT-L/14 (LLaVA-1.5), DINOv2, vanilla ViT-B/16 |
| `VCFG_FEATURE_SELECT` not read | No CLS-only feature extraction | Models that use CLS-only projector input |
| Pixel norm still semi-hardcoded | CLIP normalization `(p/255-0.48)/0.27` not yet supported | All CLIP-based VLMs |

The assembly implementation also does not read `VCFG_NORM_TYPE` or `VCFG_FFN_TYPE`. The documented
Qwen3-VL and SigLIP configurations use LayerNorm and GELU, so those values do not currently select
alternative operations.

---

### Tokenizer and sampling implementation

Tokenization and sampling are shared runtime services, but the tokenizer data and relevant configuration
values depend on the selected model. The database supplies vocabulary information and merge rules, while
the generation path applies the documented sampling settings to the resulting logits. The following
sections specify those representations and operations so that a model exporter or integration can
reproduce the expected behavior.

#### 5.1 Tokenizer (data-driven, auto-detected)

Tokenizer initialization in `inf_tokenizer.inc` supports two tokenizer families; it distinguishes them
by examining the vocabulary for the U+2581 word-prefix character:

| Family | Detection | Preprocessing | Models |
| --- | --- | --- | --- |
| **BPE** (byte-pair encoding with byte-fallback) | No U+2581 in vocab | None | Qwen, Llama-3, Phi-3, SmolLM2, MiniCPM5 |
| **SentencePiece** (Unigram) | U+2581 found in vocab | Prepend `?` dummy prefix; convert spaces to `?` before lookup | TinyLlama, Llama-2, Mistral |

The encoder (`encode.inc`) handles:
- Byte-by-byte scanning with UTF-8 awareness
- **Special token detection**: `<...>` pattern scan + `ht_find` lookup (handles `<|im_start|>`, `<image>`, `<\|image_pad\|>`, etc.)
- Byte-fallback for unknown characters
- BPE merge via `merge_scores` greedy longest-match

#### 5.2 Sampler

The native sampler, `v7s_sample`, supports greedy selection when the temperature is zero, with behavior
compared against the C reference. For stochastic generation it applies temperature and nucleus (`top-p`)
sampling. The implementation uses an AVX2 polynomial softmax, a top-256 prefilter, an AES-NI-based
random-number generator, and a multinomial selection based on the cumulative distribution.

The sampler structure also contains top-k and presence-penalty fields. Those fields should not be
confused with a guarantee that every corresponding operation is available through the public assembly
sampler: the C implementation has additional paths, while the documented native sampling path
concentrates on temperature and nucleus selection. Consult the [native API reference](native_api.md) for
the supported setter behavior.

#### 5.3 SectorLogo sidecar hook (per-token scripting)

The embedded SectorLogo interpreter can execute a script stored under `logo:sidecar` in MicroDB at the
candidate-token boundary. The language implementation is distributed across the `sectorlogo_*.inc`
assembly modules. A script receives the pending token ID and decoded piece through `:TokenNumber` and
`:Piece`, and it can inspect previously accepted text using `GetStream`.

For candidate-level decisions, `Reject` requests another sample from the existing logits and `Halt` ends
generation before accepting the candidate. The script may also change sampling settings with
`SetTemperature` and `SetNucleus`, read or update records through `DatabaseGet` and `DatabaseSet`, or
invoke native Win64 exports with `CallLibrary`. The FFI accepts `"SELF"` when the target is an export
from the V7 DLL.

`SetStream` handles a different operation: it replaces already accepted text. In an ordinary linear
session, V7 identifies the first affected token, preserves the reusable preceding KV-cache state, and
evaluates the replacement suffix before continuing. StreamingLLM requires additional handling after the
attention window has wrapped: a supported rewrite reconstructs state from the recorded session prefix
and the replacement text, retaining the defined attention-sink behavior. The detailed invariants and
capacity limits are specified in the [generation-control
manual](generation.md#stream-replacement-and-state-consistency).

**Interpreter implementation.** The following notes identify the source files and internal data
structures relevant to the interpreter:

| Component | Implementation |
| --- | --- |
| Memory | 1M cons cells with mark-sweep garbage collection (`sectorlogo_lisp_core.inc`) |
| Value representation | 2-bit tagged pointers: INT30 (30-bit two's-complement), FLOAT (top 30 bits of IEEE binary32), CONS, ATOM |
| Float operations | x87 `fyl2x`/`f2xm1`/`fscale` for LogBaseTwo/PowerOfTwo; `fprem` loop for FloatRemainder - sufficient for BM25 scoring |
| Interned atoms | String pool with O(1) equality via atom ID comparison |
| FFI | `CallLibrary [dll name ret_type [arg_types]] arg1 arg2...` - resolves DLL + function at call time, marshals up to 16 arguments |
| Script loading | `logo:sidecar` key in MicroDB, loaded lazily on first `on_token` call |

The [SectorLogo manual](sectorlogo.md#sectorlogo-language-and-integration) defines the language behavior
and host commands. Example scripts are available under `src\logo\`, and `v_sectorlogo_validate`
exercises the language, string operations, FFI, and garbage collection.

---

### Text-model compatibility by family

The text models listed below share a configurable decoder execution path. The database selects
differences such as positional encoding, attention configuration, normalization, and model dimensions. A
shared path does not imply that every related checkpoint is supported: each new configuration still
needs to match the implemented operations and the stored data layout. The verification status in this
chapter identifies which combinations have actually been exercised.

#### 6.1 Llama family (LlamaForCausalLM) - Fully supported

- **Architecture:** RMSNorm + plain RoPE (theta=500000) + SwiGLU + GQA
- **No QK-norm, no QKV bias, no tied embeddings**
- **Tested:** Llama-3.2-1B-Instruct
- **Also compatible:** Llama-3.1-8B, Llama-2-7B, Llama-3.3-70B (RAM-limited)
- **Tokenizer:** BPE (Llama-3) or SentencePiece (Llama-2) - both auto-detected
- **RoPE:** Llama-3.2 uses type=1 (llama3 scaled); Llama-2 uses type=0 (plain)

#### 6.2 Qwen3 family (Qwen3ForCausalLM) - Fully supported

- **Architecture:** RMSNorm + plain RoPE (theta=1M) + SwiGLU + GQA + **QK-normalization**
- **QK-norm:** `has_qk_norm=1` -> `q_norm_weights` and `k_norm_weights` applied inside the fused Q/K kernel (per-head RMSNorm before RoPE)
- **Tied embeddings:** Yes (`shared_classifier=1`) - classifier shares the embedding table
- **Tested:** Qwen3-0.6B (reference), Qwen3-1.7B (dim=2048, 28 layers, head_dim=128, tied, QK-norm; both F32 and Q8 produce coherent answers)
- **Also compatible:** Qwen3-4B, 7B, 14B, 32B (the Qwen3-VL-4B text stack already runs)

#### 6.3 Qwen2 / Qwen2.5 family (Qwen2ForCausalLM) - Fully supported

- **Architecture:** Similar to Qwen3, but without QK normalization and with QKV attention biases
- **QKV bias:** `q_proj.bias`, `k_proj.bias`, `v_proj.bias` on every layer. Stored in `g_qkv_bias`, added by the fused kernel's bias path.
- **Tested:** Qwen2.5-0.5B (text), LLaVA-OneVision-Qwen2-0.5B (vision)
- **Also compatible:** Qwen2.5-1.5B, 3B, 7B, 14B, 32B, 72B; Qwen2.5-Coder, Qwen2.5-Math

#### 6.4 Phi-3 family (Phi3ForCausalLM) - Fully supported

- **Architecture:** RMSNorm + plain RoPE (theta=10000) + SwiGLU + GQA (full, n_kv_heads = n_heads)
- **Fused weights:** `qkv_proj` (fused Q+K+V) and `gate_up_proj` (fused gate+up) - auto-split during export, transparent to the engine
- **Large head_dim:** 96 (3072/32) - tests the data-driven head_dim path
- **Tested:** Phi-3-mini-4k-instruct (3.8B) - the largest text model tested, produces coherent and detailed answers
- **Also compatible:** Phi-3-mini-128k, Phi-3-medium-14B, Phi-2

In the recorded functional checks, Phi-3-mini generated responses to three prompts: a question about the
capital of France, a question about cats, and a request for a Python hello-world example. These checks
demonstrate that generation completed for the tested configuration, not that the model has been
comprehensively evaluated.

#### 6.5 TinyLlama (LlamaForCausalLM) - Fully supported

- **Architecture:** Standard Llama (RMSNorm + RoPE theta=10000 + SwiGLU + GQA 8:1)
- **Tokenizer:** SentencePiece (Unigram) - exercises the SP path (`is_sentencepiece=1`, `?` dummy prefix, space?`?` conversion)
- **Tested:** TinyLlama-1.1B-Chat-v1.0
- **Small seq_len:** 2048 - the smallest KV-cache sizing in the tested set

#### 6.6 SmolLM2 (LlamaForCausalLM) - Fully supported

- **Architecture:** A standard Llama transformer with grouped-query attention (GQA)
- **Tokenizer:** BPE (49152 vocab - smallest tested)
- **Tested:** SmolLM2-135M (base model), SmolLM2-360M
- **Head dimension:** 64 (tested at both 576 and 960 hidden)

#### 6.7 MiniCPM5 (LlamaForCausalLM) - Fully supported

- **Architecture:** Standard Llama configuration with a RoPE base of 5 million
- **Large head_dim:** 128 - exercises the largest head_dim in the test suite
- **Tested:** MiniCPM5-1B

#### 6.8 DeepSeek R1 Distills - Compatible (untested)

- **DeepSeek-R1-Distill-Qwen-1.5B / 14B / 32B:** Qwen2 architecture (QKV bias) - should work via the bias path
- **DeepSeek-R1-Distill-Llama-8B / 70B:** Llama-3 architecture - should work

---

### Vision-language model compatibility by family

The vision-language models listed here use different encoder and projector layouts. Compatibility must
be checked at the checkpoint and configuration level; matching the text decoder alone does not establish
that image input is supported.

#### 7.1 Qwen3-VL-4B-Instruct - Fully supported, tested

| Component | Value |
| --- | --- |
| Text LLM | Qwen3-4B (dim=2560, 36 layers, QK-norm) |
| ViT | 24 layers, hidden=1024, 16 heads, head_dim=64 |
| Patch size | 16 |
| Position embed | Bilinear-grid resample (type 0), 48-48 learned table |
| RoPE in ViT | 2D RoPE (USE_ROPE=1) |
| Spatial merge | 2-2 (MERGE=2) |
| DeepStack | 3 taps at layers 5, 11, 17 |
| Projector | Merger IS projector (no separate MLP) |
| Image tokens | `<\|vision_start\|>` + `<\|image_pad\|>` - N + `<\|vision_end\|>` |
| Token injection | Replace image_pad embeddings with ViT output + DeepStack feature adds at layers 0/1/2 |

Qwen3-VL-4B-Instruct provides the reference vision configuration for the recorded regression tests. Its
smoke-test output is compared against the hash stored in `vision/vision_baseline.txt`.

#### 7.2 LLaVA-OneVision-Qwen2-0.5B - Fully supported, tested

| Component | Value |
| --- | --- |
| Text LLM | Qwen2-0.5B (dim=896, 24 layers, **QKV bias**) |
| ViT | SigLIP-SO400M: 26 layers, hidden=1152, 16 heads, **head_dim=72** |
| Patch size | 14 |
| Position embed | Learned additive (type 1), 729 entries |
| RoPE in ViT | None (USE_ROPE=0) |
| Spatial merge | None (MERGE=1) |
| DeepStack | None (DEPTH_TAPS=0) |
| Projector | Separate 2-layer MLP (1152?896?896, GELU) |
| Image tokens | `<image>` (token 151646), no vision_start/end wrapping |

LLaVA-OneVision-Qwen2-0.5B exercises the VCFG-dependent operations that differ from the Qwen3-VL
configuration, as listed below:

- `VCFG_HEAD_DIM=72` attention path (9-accumulator + tail, ported from attic code)
- `VCFG_POS_EMBED_TYPE=1` additive pos-embed (verbatim copy vs bilinear resample)
- `VCFG_USE_ROPE=0` (skip RoPE build/apply entirely)
- `VCFG_MERGE=1` (skip pack2x2, use post_layernorm + direct copy path)
- `VCFG_DEPTH_TAPS=0` (no DeepStack mergers)
- `VCFG_PROJ_IN_DIM=1152` (separate MLP projector - 2 matmuls + GELU)
- Data-driven image marker (`<image>` instead of `<\|image_pad\|>`)
- Inline image markers (no `<\|vision_start\|>`/`<\|vision_end\|>` wrapping)
- Qwen2 QKV biases in the text LLM (fused-bias path)

The bounding-box output convention differs: Qwen3-VL uses 0-1000 relative coordinates, LLaVA-OV uses 0-1
normalized - see `eval_history.md` (dev-only: `attic\docs_research_logbook\` ).

The integration history is recorded in `llava_onevision_integration_plan.md`; that planning record is
not part of the current runtime interface.

#### 7.3 SmolVLM-256M - Fully supported, tested

| Component | Value |
| --- | --- |
| Text LLM | SmolLM2-256M (Llama architecture, dim=576, 24 layers) |
| ViT | SigLIP-SO400M: 26 layers, hidden=1152, 16 heads, head_dim=72 |
| Patch size | 14 |
| Position embed | Learned additive (type 1), 729 entries |
| RoPE in ViT | None (USE_ROPE=0) |
| Spatial merge | None (MERGE=1) - pixel-shuffle projector instead |
| DeepStack | None (DEPTH_TAPS=0) |
| Projector | Pixel-shuffle 2-layer MLP (1152?896?576, GELU) - `VHDR_PROJ_TYPE=2` |
| Image tokens | `<image>` (token 49153), no vision_start/end wrapping |

SmolVLM reuses the same SigLIP ViT and MLP projector infrastructure as LLaVA-OneVision, but adds the
**pixel-shuffle projector type** (`PROJ_TYPE=2`). The pixel-shuffle path rearranges the ViT output
tokens via a spatial unflatten -> pixel-shuffle -> reflatten -> linear chain before the MLP, producing
fewer output tokens than the raw patch count. This exercises the third projector code path in
`vit_forward.inc` (beyond the Qwen3-VL merger path and the plain-MLP SigLIP path).

SmolVLM uses two integration points that are also present in other supported vision-language
configurations. Both must be configured correctly for image input to reach the text decoder:

1. **Chat template formatting:** The Hugging Face template uses
   `User:<image>` without an intervening space. The image-turn template is
   selected through `__sys_template@img` and `g_img_turn` in
   `chat_image.inc`. Using `User: <image>` changes the token sequence.
2. **Output projection weights:** `export_vlm.py:load_text_state_dict()` must
   retain the top-level `lm_head.weight`, which is outside the
   `model.text_model.*` prefix. Treating the output projection as tied to the
   embedding table when the checkpoint provides separate weights changes the
   classification output. The recorded maximum difference between the
   Hugging Face output head and embeddings was 0.414.

#### 7.4 What adding each new VLM would require

Two documented vision-language configurations use the SigLIP processing path. Their successful execution
establishes support for the existing configuration combinations; additional models may still require new
preprocessing, projector, or decoder behavior.

| VLM | ViT family | Text LLM | What's needed | Effort |
| --- | --- | --- | --- | --- |
| **LLaVA-1.5 / LLaVA-NeXT** | CLIP-ViT-L/14 | Llama/Vicuna | CLS token branch (`VCFG_HAS_CLS`), CLIP pixel norm, separate MLP projector (already have) | Medium |
| **PaliGemma2** | SigLIP-SO400m/14 (DONE) | Gemma 2 | Gemma LLM support (sliding-window attention) - **blocked on text side** | High |
| **Qwen2.5-VL** | Qwen2.5-VL ViT | Qwen2.5 | mrope (3D RoPE) kernel + dynamic grid - **blocked on kernel side** | High |
| **InternVL2** | InternViT-300M | InternLM/Mistral | QK-norm in ViT, trust_remote_code weight names | Medium-High |
| **MiniCPM-V-2.6** | SigLIP (NaViT) | Qwen2-7B | NaViT variable-resolution, resampler (size-blocked at 7B) | High |
| **Idefics3** | SigLIP-SO400m (DONE) | Llama-3.2-8B | Per-image-ROI tokens; text side already supported | Medium (text-side OK) |
| **Phi-3-Vision** | CLIP-style | Phi-3 | HD transform, learnable separators, Su-scaled RoPE | Medium-High |

**Important:** Idefics3-8B uses the **exact same SigLIP-SO400M-patch14-384** ViT that LLaVA-OneVision
uses. The ViT execution path would not need to change. Only the text-side adapter (Phi-3-style image-ROI tokens) would
need work. This is now the lowest-friction third VLM target if a third is wanted.

---

### Potential vision-language configurations

Because the existing implementation supports the documented SigLIP configurations, the remaining work
for additional models depends chiefly on their preprocessing, projector, and decoder requirements:

#### Primary recommendation: Moondream2 (revision 2025-04-15) - a =4B target

**Architecture.** Moondream2 is a 1.9B-parameter vision-language model using SigLIP-SO400M-patch16-384,
a two-layer MLP projector, and a 1.3B Phi-1.5 language model. Its patch-16 vision configuration differs
from the patch-14 configuration used by LLaVA-OneVision.

**What adapting it opens up:**

1. **patch16 SigLIP path** - validates `VCFG_PATCH_SIZE` at 16 in the SigLIP
   branch (currently Qwen3-VL uses patch=16 via its own patchify code; SigLIP
   path was only tested at patch=14 via LLaVA-OV).
2. **`VCFG_HAS_CLS` and `VCFG_FEATURE_SELECT` branches** are currently not read,
   which is the gap in section 4.7. Adding them would cover CLIP-based VLMs
   such as LLaVA-1.5, LLaVA-NeXT, CogVLM and the OpenCLIP-backed models.
3. **Data-driven pixel normalization** - replaces the hardcoded `1/127.5 - 1.0`
   with per-model mean/std, unlocking CLIP normalization.

**Estimated implementation scope.** The recorded estimate is approximately 100–150 lines of assembly for
CLS-token handling and configuration-dependent normalization, plus changes to the exporter. The SigLIP
ViT path, attention with `head_dim=72`, and MLP projector are already present for LLaVA-OneVision.

**Note on the text LLM:** Moondream2's earlier revisions use Phi-1.5 (which has partial RoPE + GELU
FFN, different from the implemented Phi-3 support); this would be the larger piece of work. However,
Moondream's architecture is modular - the ViT + projector + injection path is the reusable part
regardless of LLM choice.

#### Alternative: Idefics3-8B (if the =4B constraint is relaxed to 8B)

Idefics3 uses a SigLIP vision encoder closely related to the configuration exercised by LLaVA-OneVision.
The existing vision path may therefore be reusable, although the text decoder still needs image-region
token handling and must be validated against an actual checkpoint. The model size is also a separate
deployment consideration.

#### Alternative: LLaVA-1.5-7B (if the =4B constraint is relaxed to 7B)

LLaVA-1.5-7B is a relatively direct VLM architecture (fixed 336-336, CLIP-ViT-L/14, 2-layer MLP
projector, 576 tokens, Llama LLM). It would require the CLS-token branch and CLIP normalization - two
gaps listed for the current vision path. The ViT and projector code would be smaller than the Qwen3-VL
path because it has no DeepStack, spatial merge, or dynamic resolution. The current model size is an
additional practical constraint.

#### Current gap

The documented VLM gaps are the **`VCFG_HAS_CLS`** and data-driven pixel normalization branches.
Implementing and testing them may make more CLIP-ViT families reachable, but that must be verified with
a real database and model run.

---

### Verified and inferred compatibility

**What compatibility means here.** The engine implements one decoder architecture: Llama-style (RMSNorm
+ RoPE + GQA + SwiGLU/silu FFN), with data-driven flags for QK-norm (Qwen3), QKV bias (Qwen2 family),
fused QKV/gate-up weights (Phi-3), llama3 RoPE scaling (Llama-3.x), tied embeddings, and
BPE/SentencePiece tokenizers. An architecture that reduces to this shape may be compatible, but only the
models in the tested matrix have actually been run. The remaining entries describe architectural
expectations, not verified working databases. Tokenizer edge cases, chat-template differences, and
unanticipated configuration flags can still require work.

The compatibility entries distinguish three levels of evidence:

- **Tier 1, Tested.** Run end-to-end on both F32 and Q8 paths (the matrix
  in section 0). This records the result; it is not a guarantee for other
  checkpoints.
- **Tier 2, Candidate (untested).** The architecture and config flags
  resemble a tested model, but the checkpoint may differ in size, layers,
  heads, RAM requirements, or fine-tuning. Remaining work usually includes a
  `fetch_*.bat` run, export, database build, and verification run.
- **Tier 3 - Assumed compatible, needs verification (medium confidence).**
  Architecture matches on paper, but some detail differs from every tested
  model (activation variant, tokenizer generation, norm placement, config
  flag combination). Expect export.py adjustments and a debugging session.
  Some of these will turn out to need engine changes and move to the
  unsupported list in section 8b.

Models that need engine changes in the assembly, or that have known export changes, are listed in
section 8b; they are not part of the tested set.

#### 8a.1 Llama family (LlamaForCausalLM)

The supported Llama-family configurations use RMSNorm, rotary position embeddings, and SwiGLU
feed-forward layers. Llama 3.x also uses its particular RoPE scaling configuration (`rope_type=llama3`,
`factor=32` ), which is included in the recorded tests.

| Model | Params | Tier | Notes |
| --- | --- | --- | --- |
| `unsloth/Llama-3.2-1B-Instruct` | 1.0B | **Tested** | Reference Llama; llama3 rope scaling exercised |
| `meta-llama/Llama-3.2-3B` / `-Instruct` | 3.0B | Tier 2 | Same arch as tested 1B |
| `meta-llama/Llama-3.1-8B` / `-Instruct`, `NousResearch/Meta-Llama-3.1-8B` | 8.0B | Tier 2 | Same arch; needs >=16 GB RAM for export + inference |
| `meta-llama/Meta-Llama-3-8B` / `-Instruct` | 8.0B | Tier 2 | Llama-3 rope scaling identical to 3.1 |
| `meta-llama/Llama-3.3-70B-Instruct` | 70B | Tier 2 (arch) | RAM-impractical on the dev machine; architecture identical |
| `meta-llama/Llama-2-7b-hf` / `-chat-hf`, `NousResearch/Llama-2-7b-hf` | 7.0B | Tier 2 | No rope scaling; plain arch |
| `NousResearch/Hermes-3-Llama-3.1-8B` | 8.0B | Tier 2 | Fine-tune of a Tier-2 base |
| `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF` | 70B | Tier 2 (arch) | Llama-3.1 fine-tune |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | **Tested** | SentencePiece path exercised |
| `mtgv/MobileLLaMA-1.4B-Chat` | 1.4B | Tier 2 | Plain Llama arch |
| `codellama/CodeLlama-7b-Instruct-hf` | 7.0B | Tier 3 | Llama-2 arch, but code models have infilling token conventions the chat path does not exercise; generation (not fill-in-middle) should work |
| `01-ai/Yi-6B` / `Yi-34B` / `Yi-1.5-9B-Chat` | 6-34B | Tier 2 | Standard Llama arch, no rope scaling |
| `openlm-research/open_llama_3b_v2` / `open_llama_7b_v2` | 3/7B | Tier 3 | Llama architecture, but verify the FFN activation. OpenLLaMA variants historically used GELU; the engine currently implements SwiGLU/silu. Check `hidden_act` before exporting. |
| `stabilityai/stablelm-2-1_6b` / `-12b` | 1.6B/12B | not plain Llama, see below | `partial_rotary_factor: 0.25` rotates only 256 of 2048 rotary dims, and the QKV bias needs export slicing or engine RoPE changes. Moved out of Tier 2. |
| `allenai/OLMo-2-0425-1B-Instruct` | 1B | Tier 3 | Llama-like with QK-norm (like Qwen3, supported) and no RoPE scaling; norm/weight-key layout differs from every tested model, verify export key mapping first |

#### 8a.2 Qwen family (Qwen2ForCausalLM / Qwen3ForCausalLM)

Qwen2 and Qwen2.5 require attention QKV biases, which the runtime detects from the model data and which
were tested with Qwen2.5-0.5B. Qwen3 adds QK normalization; the corresponding path was tested with
Qwen3-0.6B.

| Model | Params | Tier | Notes |
| --- | --- | --- | --- |
| `Qwen/Qwen3-0.6B` | 0.6B | **Tested** | Reference model of the whole project |
| `Qwen/Qwen3-1.7B` | 1.7B | **Tested** | Same arch and flags as 0.6B; F32 and Q8 verified via inference_example |
| `Qwen/Qwen3-4B` | 4B | Tier 2 | Same arch and flags; Qwen3-VL-4B's text stack already runs |
| `Qwen/Qwen3-7B` / `-14B` / `-32B` | 7-32B | Tier 2 (arch) | RAM-bound on the dev machine |
| `Qwen/Qwen2.5-0.5B` (+ `-Instruct`) | 0.5B | **Tested** | QKV-bias path exercised |
| `Qwen/Qwen2.5-1.5B` / `-3B` / `-7B` / `-14B` / `-32B` (+ `-Instruct`) | varies | Tier 2 | Same arch as tested 0.5B |
| `Qwen/Qwen2.5-Coder-1.5B/7B-Instruct`, `Qwen/Qwen2.5-Math-*` | varies | Tier 2 | Same arch, task-tuned weights |
| `Qwen/Qwen2-0.5B` / `-1.5B-Instruct` / `-7B-Instruct` | varies | Tier 2 | Same QKV-bias arch |
| `Qwen/Qwen2.5-72B-Instruct` | 72B | Tier 2 (arch) | Needs >128 GB RAM |
| `Qwen/Qwen3-MoE-*`, any | not applicable | not supported | Mixture-of-Experts, a separate architecture. See 8b. |

#### 8a.3 Phi-3 family (Phi3ForCausalLM)

Phi-3-mini stores fused `qkv_proj` and `gate_up_proj` weights. The export process separates those
projections into the representation expected by the runtime. The tested configuration uses an
attention-head dimension of 96.

| Model | Params | Tier | Notes |
| --- | --- | --- | --- |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | **Tested** | Fused-weight path exercised |
| `microsoft/Phi-3-mini-128k-instruct` | 3.8B | **Needs work** | Uses **longrope** scaling (per-layer su/fu factors) - a different RoPE scaling type than the engine's modes (none/llama3); needs export + engine support |
| `microsoft/Phi-3-medium-4k-instruct` | 14B | Tier 2 | Same fused layout as mini-4k; RAM-heavy |
| `microsoft/phi-2` | 2.7B | needs work | not a standard Llama arch: `partial_rotary_factor: 0.4` (rotates 32 of 80 rotary pairs) + **GELU** FFN - the engine implements SwiGLU/silu only; needs engine activation + RoPE changes |

#### 8a.4 Mistral family (MistralForCausalLM)

Mistral-7B v0.1-v0.3: standard Llama arch (RMSNorm + RoPE + SwiGLU + GQA), SentencePiece v3 tokenizer,
no sliding window. Not verified; the configuration follows the standard Llama architecture.

| Model | Params | Tier | Notes |
| --- | --- | --- | --- |
| `mistralai/Mistral-7B-Instruct-v0.1` / `-v0.2` / `-v0.3` (+ base) | 7.0B | Tier 2 | Plain arch; SentencePiece path tested via TinyLlama; `[INST]` template auto-detected at export |
| `mistralai/Mistral-Nemo-Instruct-2407` | 12B | Tier 3 | Arch is plain (40L, dim 5120, head_dim 128, theta 1M, SwiGLU), but the **Tekken BPE tokenizer** (131k vocab) has not been run through our BPE export - verify tokenizer round-trip first |
| `mistralai/Mistral-Small-*`, `Ministral-8B-*` | 8-24B | Tier 3 | Same arch family; verify per-model config (some newer variants add features) |

#### 8a.5 SmolLM family (LlamaForCausalLM)

| Model | Params | Tier | Notes |
| --- | --- | --- | --- |
| `HuggingFaceTB/SmolLM2-135M-Instruct` / `-360M-Instruct` | 0.135/0.36B | **Tested** | — |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | Tier 2 | Same arch as tested sizes |
| `HuggingFaceTB/SmolLM-135M` / `-360M-Instruct` (v1) | 0.135/0.36B | Tier 3 | Llama-arch, but v1 used a Llama-2-style tokenizer and GELU in some checkpoints - check `hidden_act` before export |

#### 8a.6 MiniCPM family

| Model | Params | Tier | Notes |
| --- | --- | --- | --- |
| `openbmb/MiniCPM5-1B` | 1.0B | **Tested** | Llama arch, rope base 5M |
| `openbmb/MiniCPM4-0.5B` | 0.5B | not supported | Incompatible architecture, a documented exclusion |

#### 8a.7 DeepSeek R1 distills

Llama- or Qwen-architecture checkpoints distilled from DeepSeek-R1 - the base architectures are covered;
the R1 chat template is auto-detected at export.

| Model | Base | Params | Tier | Notes |
| --- | --- | --- | --- | --- |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Qwen2 | 1.5B | Tier 2 | QKV-bias arch as tested |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` / `-32B` | Qwen2 | 14/32B | Tier 2 (arch) | RAM-bound |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Llama3 | 8B | Tier 2 | Llama-3 arch incl. llama3 rope scaling |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | Llama3 | 70B | Tier 2 (arch) | Needs >128 GB RAM |
| `deepseek-ai/DeepSeek-R1-*` (full/moe originals) | MoE | - | not supported | Mixture-of-Experts. See 8b. |

#### 8a.8 Vision-language models (ViT + decoder)

The vision pipeline obtains its configuration through VCFG. Loading a particular vision-language
checkpoint also depends on whether `tools\export_vlm.py` can produce the appropriate `.vision` and,
where required, `.proj` files.

| Model | Text decoder | Vision tower | Tier | Notes |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-VL-4B-Instruct` | Qwen3 | Qwen3-VL ViT (DeepStack) | **Tested** | — |
| `Qwen/Qwen3-VL-2B-Instruct` | Qwen3 | Qwen3-VL ViT | Tier 2 | Same layout as tested 4B |
| `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | Qwen2-0.5B | SigLIP (MLP projector) | **Tested** | — |
| `llava-hf/llava-onevision-qwen2-7b-ov-hf` | Qwen2-7B | SigLIP | Tier 2 | Same vision layout as tested 0.5B |
| `HuggingFaceTB/SmolVLM-256M-Instruct` | SmolLM2 | SigLIP (pixel-shuffle) | **Tested** | — |
| `llava-hf/llava-v1.6-mistral-7b-hf` | Mistral-7B | CLIP-336 | Tier 3 | Different projector layout (CLIP + linear); export_vlm.py adaptation required |

#### 8a.9 Summary, recorded count

- **Tested end-to-end:** the section 0 matrix's 9 routinely-tested LLMs plus
  the 3 tested VLMs; the matrix's config-derived Qwen3-8B row is supported
  (fetch script present) but not routinely tested.
- **Tier 2, candidate, untested:** checkpoints whose recorded architecture
  and flags resemble a tested model but differ in size or fine-tuning. Each
  still needs a fetch, export, database build, and verification run.
- **Tier 3, assumed, needs verification:** a small set of other layouts
  (Mistral-Nemo Tekken tokenizer, OLMo-2 key layout, OpenLLaMA activation,
  SmolLM v1, LLaVA-v1.6 CLIP projector). Each needs investigation, and some
  may not be compatible.
- **Needs engine work (do not count as compatible):** phi-2 and StableLM-2
  (partial rotary; phi-2 also GELU), Phi-3-mini-128k (longrope), and
  everything in section 8b (MoE, sliding window, SSM, Falcon, NeoX/J/Pythia,
  BLOOM, encoder-decoder).

### Unsupported model configurations

These architectures require code changes and are not supported. The required change is summarized for
each, and none is currently on the roadmap.

| Architecture | Example HF model | Reason | Required change |
| --- | --- | --- | --- |
| **MoE (Mixture-of-Experts)** | `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistralai/Mixtral-8x22B-Instruct-v0.1`, `Qwen/Qwen1.5-MoE-A2.7B`, `deepseek-ai/deepseek-moe-16b-base` | Sparse expert FFN | Add expert-routing dispatch path; new `v7d_swiglu_moe` kernel |
| **Sliding-window attention** | `google/gemma-2-2b` / `-9b` / `-27b`, `google/gemma-3-*`, `CohereForAI/c4ai-command-r-plus`, `CohereForAI/c4ai-command-r-v01` | Alternating local + global attention layers; logit soft-capping (Gemma-2) | Add sliding-window kernel + layer-type flag |
| **Mamba / state-space** | `mistralai/Mamba-Codestral-7B-v0.1`, `state-spaces/mamba-130m` / `mamba-2.8b`, `ai21labs/Jamba-v0.1` | Not a transformer - SSM recurrence | Wholesale new inference path |
| **Falcon** | `tiiuae/falcon-7b`, `tiiuae/falcon-40b` | ALiBi-ish position embedding, no RoPE, fused QKV layout | Different attention kernel |
| **GPT-NeoX / GPT-J / Pythia** | `EleutherAI/gpt-neox-20b`, `EleutherAI/gpt-j-6B`, `EleutherAI/pythia-*` | Rotary on different dimensions (pario style), post-norm in Pythia | Adapt RoPE apply axis |
| **BLOOM** | `bigscience/bloom-7b1` | ALiBi instead of RoPE | Replace RoPE with ALiBi |
| **Encoder-decoder** | `t5-base`, `facebook/bart-large`, `google/ul2` | Not a causal decoder | Wholesale new inference path |
| **Nanbeige4.1-3B** | `Nanbeige/Nanbeige4-3B` | Checkpoint omits `model.norm.weight` (HF-side format issue) | HF-side fix, not v7 |

---

### Deployment considerations

V7 is intended for applications that need local inference on Windows x64 CPUs without depending on a
Python runtime or a GPU. The supported models still require adequate system memory and storage, which
vary with the checkpoint and quantization.

1. **Air-gapped systems** - secure facilities with no
   network connection. The entire inference system is one DLL + one database
   file: no package manager, no pip install, no runtime downloads.
2. **Industrial control** - embed LLM-based natural language interfaces into
   SCADA or PLC-adjacent Windows PCs on factory floors. Operators query machine
   status or issue commands in plain English with no internet dependency.
3. **Game development (NPC AI)** - AAA studios running Windows-based engines
   (Unreal, Unity) can run local, multimodal VLMs for NPCs without eating into
   the GPU VRAM budget required for rendering. The engine runs entirely on CPU
   cores that would otherwise be idle during GPU-bound frames. The single-DLL +
   single-DB deployment model integrates cleanly into existing game engine asset
   pipelines. A ViT-based NPC can "see" game screenshots and respond to the
   visual scene - no separate vision preprocessing needed.
4. **Embedded / offline inference** - no Python, no GPU, no Docker. One DLL.
5. **Point-of-sale and kiosk systems** - add product search or
   FAQ answering to retail terminals, entirely on-device with zero
   additional dependencies.
6. **Assistive technology** - screen readers, communication aids, and
   accessibility tools that need on-device text generation.
   Q8 quantized inference provides a complete text-in/text-out
   sequence in one DLL.
7. **Legacy Windows integration** - add AI capabilities to existing Windows
   applications by loading a single DLL via `LoadLibraryA`. No runtime
   installation, no container, no framework dependency.
8. **Persistent cross-session memory without vector databases** - the
   SectorLogo sidecar implements text-based memory via a local BM25 search
   tool (`bm25_rag.exe`) against a plain text file (`ragg_base.txt`). At
   the end of a conversation, a distillation step extracts factual sentences
   and appends them to the file. In the next session, the sidecar intercepts
   the model's reasoning, runs a lexical search, and injects relevant past
    memories into the thought stream via `SetStream`. This provides
   cross-session recall with no embedding model, no vector database, and no
   network calls - the processing runs within the v7 process.

---

### Limitations and unsupported operations

| Limitation | Reason | Models affected |
| --- | --- | --- |
| **No sliding-window attention** | Only causal + bidirectional (ViT) attention exists | Gemma-2, Gemma-3, Command-R |
| **No MoE (Mixture-of-Experts)** | No expert-routing dispatch path | Mixtral, DeepSeek-MoE, Qwen-MoE |
| **No Mamba/SSM** | Not a transformer recurrence | Mamba, Jamba |
| **No ALiBi position bias** | Only RoPE is implemented | BLOOM, Falcon |
| **No encoder-decoder** | Only causal decoder + bidirectional ViT | T5, BART, UL2 |
| **No audio/speech** | No FFT, no Mel filterbank, no cross-attention decoder | Whisper, Wav2Vec2 |
| **No non-RMSNorm text norm** | LLM path uses RMSNorm only (ViT uses LayerNorm) | Phi-1.5 (uses LayerNorm in text) |
| **No GQA-unfriendly attention** | GQA ratio must be integer (n_heads / n_kv_heads) | Models with non-integer GQA ratios |
| **No CLS-token ViT yet** | `VCFG_HAS_CLS` not read by ASM | CLIP-ViT-L/14, vanilla ViT, DINOv2 |

---

### Implementation file map

#### ASM (engine internals)

| Capability | File(s) |
| --- | --- |
| LLM forward (F32) | `src/asm/forward_f32.inc` |
| LLM forward (Q8) | `src/asm/forward_q8.inc` |
| LLM prefill (F32) | `src/asm/prefill_f32.inc` |
| LLM prefill (Q8) | `src/asm/prefill_q8.inc` |
| Fused matmul + RoPE + norm + bias | `src/asm/fused_f32_kernel.inc` (F32), `src/asm/q8_kernel.inc` (Q8) |
| RoPE table builder (plain + llama3) | `src/asm/fused_f32_kernel.inc` (`v7s_rope_init`) |
| Attention (causal, dispatched) | `src/asm/attention_kernel.inc`, `src/asm/attention_dispatch.inc` |
| SwiGLU (dispatched) | `src/asm/swiglu_kernel.inc`, `src/asm/swiglu_q8_kernel.inc` |
| Q8 matmul (dispatched) | `src/asm/q8_kernel.inc`, `src/asm/matmul_dispatch.inc` |
| F32 matmul (dispatched) | `src/asm/fused_f32_kernel.inc`, `src/asm/matmul_dispatch.inc` |
| memcpy (dispatched) | `src/asm/memcpy_kernel.inc` |
| RMSNorm (inline) | `src/asm/rmsnorm_kernel.inc` |
| ViT forward (unified) | `src/asm/vit_forward.inc` |
| ViT attention (dispatched, head_dim 64/72) | `src/asm/vit_kernel.inc`, `src/asm/vit_dispatch.inc` |
| ViT linear (dispatched) | `src/asm/vit_kernel.inc`, `src/asm/vit_dispatch.inc` |
| ViT helpers | `src/asm/vit_layernorm.inc`, `vit_gelu.inc`, `vit_add.inc`, `vit_pos_embed.inc`, `vit_rope_build.inc`, `vit_rope_apply.inc`, `vit_input_stage.inc` |
| Vision -> LLM bridge | `src/asm/chat_image.inc` |
| Inference init | `src/asm/inf_microdb.inc`, `inf_weights.inc`, `inf_state.inc`, `inf_tokenizer.inc`, `inf_cleanup.inc` |
| Struct offsets | `src/asm/inf_structs.inc`, `src/asm/vit_structs.inc` |
| BSS globals | `src/asm/v7_data.inc` |
| Tokenizer (encode) | `src/asm/encode.inc` |
| Tokenizer (decode) | `src/asm/chat.inc`, `src/asm/generate.inc` |
| Sampler | `src/asm/sampler.inc` |
| Chat loop | `src/asm/chat.inc` |
| Generate loop | `src/asm/generate.inc` |
| Dispatch system | `src/asm/dispatch_execute.inc`, `v7_worker.inc`, `v7_instr.inc` |
| Worker pool init | `src/asm/v7_init.inc` |

#### Data preparation (feeds the engine)

| Capability | File(s) |
| --- | --- |
| Text LLM export (safetensors ?.bin/.tokenizer/.template) | `tools/export.py` |
| VLM export (text + vision + projector) | `tools/export_vlm.py` |
| Vision tower weight extraction (Qwen3-VL + SigLIP) | `tools/export_vlm.py` |
| DB builder (writes `__sys_*` keys, precomputes VCFG) | `tools/db_build_tool.c` |
| VHDR slot constants (must stay in sync) | `tools/export_vlm.py` (Python), `tools/db_build_tool.c` (C), `src/asm/inf_structs.inc` (ASM VCFG offsets) |

## Additional model listings and fetch scripts

The fetch-script tables provide a practical route from a supported upstream checkpoint to a V7 database.
They also preserve the descriptive notes distributed with the release. Those notes are not a
standardized comparison of model quality, and they should not replace the compatibility and validation
information earlier in this manual. For a new installation, select a verified configuration first and
use the corresponding script or documented manual preparation procedure.

### Text-model fetch scripts

| Model | Params | Tokenizer | Quality | Fetch Script |
| --- | --- | --- | --- | --- |
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

### Vision-language model fetch scripts

| Model | Params | ViT | Quality | Fetch Script |
| --- | --- | --- | --- | --- |
| Qwen3-VL-4B | 4.0B | Qwen3-VL (DeepStack) | Vision (text+image) | (manual setup) |
| LLaVA-OneVision | 0.5B | SigLIP (MLP projector) | Vision (text+image) | `fetch_llava_onevision_qwen2_0.5b.bat` |
| SmolVLM-256M | 0.256B | SigLIP (pixel-shuffle) | Vision (text+image) | `fetch_smolvlm_256m.bat` |

Additional model families may be configuration-compatible, but compatibility has not been verified for
every checkpoint. The original overview identified more than 40 possible configurations; consult the
current compatibility analysis — see [model compatibility analysis](#model-support-and-configuration)
for the full compatibility analysis (Llama family, Qwen2/2.5, DeepSeek distills, etc.).

The model preparation tools obtain the following configuration values from the input model and store
them for inference:
- **QK-normalization** (Qwen3 only) from weight key presence
- **Tied embeddings** (`shared_classifier`) by comparing tok_emb and lm_head
- **Fused weights** (Phi-3-style `qkv_proj` and `gate_up_proj`) during export
- **Tokenizer type** (BPE vs SentencePiece) via `is_sentencepiece` flag
- **RoPE base frequency** from config, overridable with `-rope-base`
- **RoPE scaling** (llama3 factor=32) from checkpoint header

---

---

## Vision-language execution

V7 accepts decoded RGB images through its public image interface and prepares their features for the
language decoder. The application is responsible for loading and decoding the original image file; V7
handles the model-specific processing needed to turn that pixel buffer into embeddings. Although the
tested models share the same general sequence of image preprocessing, encoder execution, and projection
into language-model dimensions, they differ in the operations and layouts used at each stage.

This chapter follows an image through that sequence. It describes the preprocessing and patch
construction rules, the vision-transformer execution path, the projector, and the way visual embeddings
are introduced into text generation. It also identifies the configuration fields that select each path
and the checks used to validate the three tested vision-language models. These configurations
demonstrate specific supported combinations of operations; they do not imply that every checkpoint
carrying the same family name will work without further validation.

The following sections specify the supported configurations, image preprocessing, ViT execution stages,
validation methodology, and integration with the language-model context.

### Supported Models and Configuration

#### Overview

The three tested vision-language model configurations use a shared Vision Transformer forward
implementation. The database supplies model-specific settings through `__sys_vision_config`, avoiding
separate copies of the main ViT processing sequence.

#### Supported VLMs

| Model | ViT Architecture | Projector | Text-side Position Encoding for image tokens | Coordinate Format |
| --- | --- | --- | --- | --- |
| Qwen3-VL 4B | RMS + RoPE + DeepStack | Spatial merge | M-RoPE in 3D | 0-1000 relative |
| LLaVA-OneVision 0.5B | SigLIP with LayerNorm and GELU | MLP, 2 layers | plain 1D RoPE, which takes the M-RoPE early-out | 0-1 normalized |
| SmolVLM 256M | SigLIP with LayerNorm and GELU | Pixel shuffle | plain 1D RoPE, which takes the M-RoPE early-out | 0-1 normalized |

#### Unified forward path

All three VLMs share one `vit_forward.inc`. Model-specific behavior is controlled by the VCFG block
loaded from the database as `__sys_vision_config`:

- `VCFG_APPLY_POST_LN` is set only for pixel-shuffle projector models such as
  SmolVLM. LLaVA-OV uses raw hidden states taken before post-LN.
- `VCFG_PROJ_TYPE` is 0 for merger, 1 for MLP and 2 for pixel shuffle
- `VCFG_ANYRES_ENABLED` selects anyres multi-tile processing, which LLaVA-OV uses

Fields present in the block but not yet read by the assembly are `VCFG_NORM_TYPE` and `VCFG_FFN_TYPE`,
where both families run LayerNorm with GELU, and `VCFG_HAS_CLS` and `VCFG_FEATURE_SELECT`, because no
CLS-token path is implemented. The two supported ViT families happen to agree on these values, so no
branch is needed today.

#### ViT processing sequence

```
Input RGB pixels (640x640x3)
    |
    v
Image resize (nearest-neighbor for single-tile models; bicubic for LLaVA-OneVision anyres)
    |
    v
Patch embedding (conv-style linear, K=VIT_HIDDEN)
    |
    v
Position embedding (learned table, align_corners=True for Qwen3)
    |
    v
Transformer layers (N=VCFG_DEPTH):
    LayerNorm/RMSNorm -> QKV linear -> RoPE -> Attention -> Add
    LayerNorm/RMSNorm -> FFN linear (GELU/SiLU) -> Add
    |
    v
[Optional] Post-layernorm
    |
    v
Feature selection (CLS token, last token, or mean pool)
    |
    v
Projector (spatial merge / MLP / pixel shuffle)
    |
    v
Vision features (n_image_tokens x hidden_dim floats)
```

#### Anyres Multi-Crop (LLaVA-OV)

LLaVA-OneVision uses anyres processing: the image is aspect-preserving FIT (bicubic) into the selected
best resolution, then centered on a black-padded canvas when the fit is smaller than the canvas, and the
canvas is cropped into `image_size` -sized grid tiles. Each tile is independently encoded through the
ViT, then the features are concatenated (with newline separators between tile rows) and passed through
the projector.

The anyres path is controlled by:
- `VCFG_ANYRES_ENABLED` - 1 for LLaVA-OV, 0 for others
- The candidate selection mirrors HF: all 36 (height, width) pinpoint pairs
  `n x image_size` for n = 1..6 are scored (maximize effective pixels after
  the aspect-preserving fit, capped at the source pixel count; ties broken by
  minimal wasted canvas), and the winner defines the grid
- Centering uses floor division, so the remainder pixel goes right/bottom,
  matching HF's `divmod` padding

#### M-RoPE, multimodal rotary position embeddings

Qwen3-VL uses M-RoPE: text tokens receive 1D position (t,t,t), while image-pad tokens receive 3D
positions (t,h,w) with interleaved frequency assignment across the three RoPE dimensions.

Implementation:
- `src/asm/mrope.inc` holds `v7s_mrope_prepare`, which builds a per-token 3D
  position table, and `v7s_mrope_restore`, which restores 1D mode and advances
  the position counter
- `__sys_mrope_bound` is 3 * min(mrope_section) and controls interleaving depth
- Models that do not use M-RoPE, LLaVA-OV and SmolVLM, take an early-out that
  does nothing

#### Key Source Files

| File | Purpose |
| --- | --- |
| `src/asm/vit_forward.inc` | Unified ViT forward: layer loop, pre and post LN, feature select |
| `src/asm/vit_kernel.inc` | ViT kernels for layernorm, GELU, add, linear and attention |
| `src/asm/vit_dispatch.inc` | ViT linear and attention dispatch |
| `src/asm/vit_structs.inc` | VCFG layout constants and invariant definitions |
| `src/asm/vit_anyres.inc` | Anyres multi-tile path used by LLaVA-OV |
| `src/asm/vit_pixel_shuffle.inc` | Pixel-shuffle projector used by SmolVLM |
| `src/asm/vit_pos_embed.inc` | Position embedding from a learned table with align_corners |
| `src/asm/vit_bicubic.inc` | Catmull-Rom bicubic resize used by LLaVA-OV |
| `src/asm/vit_input_stage.inc` | Nearest-neighbor input resize for single-tile models |
| `src/asm/vit_gelu.inc` | GELU approximation using tanh |
| `src/asm/vit_layernorm.inc` | LayerNorm kernel |
| `src/asm/vit_rope_build.inc` | ViT RoPE table builder |
| `src/asm/vit_rope_apply.inc` | ViT RoPE application |
| `src/asm/mrope.inc` | M-RoPE 3D position encoding for Qwen3-VL |
| `src/asm/chat_image.inc` | Image encoding and chat integration |

### Image preprocessing

#### Bicubic resize, LLaVA-OV

The bicubic resize implementation uses a Catmull–Rom kernel with `a = -0.5`, half-pixel coordinate
centering, and edge clamping. These choices match the relevant sampling conventions of PIL's
`Image.BICUBIC` operation for the recorded test.

The implementation uses fixed 4-tap Catmull-Rom without anti-aliasing; this matches HF behavior for
upscaling into grid tiles and is close for downscaling into the base tile. The model is robust to the
small pixel difference.

#### Input resize

The single-tile path uses the nearest-neighbor input stage in `vit_input_stage.inc`. LLaVA-OneVision
anyres uses the bicubic path described above. There is no separate bilinear ViT input kernel in the
current source.

#### VCFG Block Layout

The `__sys_vision_config` block, `VCFG_QWORDS = 93` and 744 bytes, is precomputed entirely in C in
`db_build_tool.c` and stored in the database. The assembly reads fields with single-instruction
RIP-relative memory operands, so there is zero runtime arithmetic for model-dependent values.

The following table lists the configuration fields and their byte offsets. The complete definition is
maintained in `inf_structs.inc`.

| Offset | Field | Description |
| --- | --- | --- |
| 64 | VCFG_HIDDEN | ViT hidden dimension |
| 104 | VCFG_HEADS | Number of attention heads |
| 112 | VCFG_HEAD_DIM | head_dim, 64 for Qwen3-VL and 72 for SigLIP |
| 128 | VCFG_DEPTH | Number of transformer layers |
| 176 | VCFG_MERGE | Spatial merge size, 2 for Qwen3-VL and 1 for none |
| 192 | VCFG_PATCH_SIZE | Patch size, 16 or 14 |
| 520 | VCFG_ATTN_SCALE | IEEE-754 bits of 1/sqrt(head_dim) |
| 528 | VCFG_NORM_TYPE | 0 for RMSNorm, 1 for LayerNorm |
| 536 | VCFG_FFN_TYPE | 0 for SwiGLU, 1 for GELU MLP |
| 544 | VCFG_HAS_CLS | CLS token present |
| 552 | VCFG_FEATURE_SELECT | Vision feature selection |
| 584 | VCFG_POS_EMBED_TYPE | 0 for a bilinear grid, 1 for learned additive |
| 592 | VCFG_IMAGE_SIZE | Base tile resolution, where 0 means grid-based |
| 248 | VCFG_N_DEEPSTACK | DeepStack tap count (3 or 0); offset 600 is reserved |
| 616 | VCFG_USE_ROPE | Apply 2D RoPE to Q and K |
| 624/632/640 | VCFG_PROJ_IN/OUT/INTER_DIM | Separate MLP projector dims, where 0 means merger |
| 648 | VCFG_PROJ_TYPE | 0 for merger, 1 for MLP, 2 for pixel shuffle |
| 656 | VCFG_APPLY_POST_LN | Apply post-layernorm as SmolVLM does, or use raw hidden as LLaVA-OV does |
| 696 | VCFG_ANYRES_ENABLED | Anyres multi-tile, used by LLaVA-OV |
| 280/288/296 | VCFG_DS_TAP0/1/2 | Qwen3-VL DeepStack layer indices, 5, 11 and 17 |
| 704 | VCFG_LN_EPS | ViT LayerNorm epsilon bits |
| 712 | VCFG_AR_GRID_SIDE | AnyRes patch grid side |
| 720 | VCFG_AR_BASE_TOK | AnyRes tokens per tile |
| 728 | VCFG_AR_TC_ADV | AnyRes tile-column advance |
| 736 | VCFG_PS_OUT_W | Pixel-shuffle output grid width (last qword of 744-byte block) |

### Box Test Methodology

#### Test Setup

1. `vit_bbox_eval.exe` synthesizes a 640x640 RGB image: light-gray background,
   a filled red circle and a filled green square at specified coordinates.
2. The image is passed to the DLL via `v7_chat_image`, which does no file I/O.
3. Greedy sampling with temperature 0, tested on both the F32 and Q8 paths.
4. An HF reference generated by the box reference generator (dev-only `new_tools\`) using identical images through
   HuggingFace transformers.

#### Test Configurations

| Config | Red circle | Green square |
| --- | --- | --- |
| A | center (180,180) r90 | (400,400) 140x140 |
| B | center (100,480) r60 | (500,80) 100x100 |
| C | center (320,320) r120 | (80,560) 80x80 |

#### Recorded box-test results

**Qwen3-VL 4B**, the recorded v7 deviations were lower than the HF reference deviations in these three
runs:

| Config | v7 Q8 | HF | v7 dev | HF dev | Status |
| --- | --- | --- | --- | --- | --- |
| A | (135,138,425,425) | (133,133,427,427) | 3.75 | 6.5 | lower recorded deviation |
| B | (58,652,255,847) | (57,648,257,848) | 4.25 | 6.25 | lower recorded deviation |
| C | (308,308,692,692) | (305,305,692,692) | 3.5 | 4.5 | lower recorded deviation |

Average recorded deviation: v7 = 3.5-4.75 vs HF = 4.5-6.5. This table shows the result for these runs;
it does not establish a general accuracy advantage.

**LLaVA-OV 0.5B** - exact match with HF:

| Config | v7 Output | HF Output | Status |
| --- | --- | --- | --- |
| A | red [0.25,0.25,0.4,0.4] green [0.8,0.4,0.9,0.4] | same | same recorded output |

**SmolVLM 256M**, the recorded output matched HF; grounding was weak in both:

| Config | v7 Output | HF Output | Status |
| --- | --- | --- | --- |
| A | red (0.0,0.0) green (0.0,0.0) | same | same recorded output |

SmolVLM-256M produced weak grounding in this test. The v7 and HF outputs were the same, so this is a
match of recorded results rather than evidence of strong grounding.

#### Reproducing Box Tests

```bat
cd bin

:: Qwen3-VL 4B (Config A)
vit_bbox_eval.exe..\weights\databases\qwen3_vl_4b\lz4ht.db ^
  "This image contains a red circle and a green square. Locate each shape and provide its bounding box coordinates." ^
  180 180 90 400 400 140

:: LLaVA-OV 0.5B (Config A)
vit_bbox_eval.exe..\weights\databases\llava_onevision_qwen2_0.5b\lz4ht.db ^
  "This image contains a red circle and a green square. Locate each shape and provide its bounding box coordinates." ^
  180 180 90 400 400 140

:: SmolVLM 256M (Config A)
vit_bbox_eval.exe..\weights\databases\smolvlm_256m\lz4ht.db ^
  "This image contains a red circle and a green square. Locate each shape and provide its bounding box coordinates." ^
  180 180 90 400 400 140
```

#### Building VLM Databases

Each VLM requires a `.bin` checkpoint, a `.vision` file, and optionally a `.proj` file for MLP and
pixel-shuffle projectors. The database builder `build_db.exe` reads these and stores everything in
`lz4ht.db`.

```bat
cd bin

:: Qwen3-VL 4B (spatial merge projector, no.proj needed)
build_db.exe "..\weights\qwen3_vl_4b\qwen3_vl_4b" ^
  -vision "..\weights\qwen3_vl_4b\qwen3_vl_4b.vision" ^
  -o weights\databases\qwen3_vl_4b\lz4ht.db

:: LLaVA-OV 0.5B (MLP projector, needs.proj)
build_db.exe "..\weights\llava_onevision_qwen2_0.5b\llava_onevision_qwen2_0.5b" ^
  -vision "..\weights\llava_onevision_qwen2_0.5b\llava_onevision_qwen2_0.5b.vision" ^
  -visionproj "..\weights\llava_onevision_qwen2_0.5b\llava_onevision_qwen2_0.5b.proj" ^
  -o weights\databases\llava_onevision_qwen2_0.5b\lz4ht.db

:: SmolVLM 256M (pixel-shuffle projector, needs.proj)
build_db.exe "..\weights\smolvlm_256m\smolvlm_256m" ^
  -vision "..\weights\smolvlm_256m\smolvlm_256m.vision" ^
  -visionproj "..\weights\smolvlm_256m\smolvlm_256m.proj" ^
  -o weights\databases\smolvlm_256m\lz4ht.db
```

**Important:** `threading_proxy.ini` must exist in the package root before building any database. The DB
builder reads INI dispatch parameters and stores them in the database as `__ini__*` keys.

#### Text-Side Inference

VLM models also support text-only inference through the standard `v7s_chat` API. The text-side LLM
(Qwen3, Qwen2) runs through the same forward paths as the base LLM models, with the addition of image
token handling when vision input is provided.

```bat
:: Text-only inference with the Qwen3-VL 4B database
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_vl_4b\lz4ht.db -i "What is the capital of France?" -q8 1
```

---

### Source-Level Execution Flow

#### Architecture Overview

The processing is split across two components:

1. **The caller, a harness or C tool**, owns all file I/O. It loads the image file and
   its dimensions from disk, then passes the raw RGB buffer plus width and height to the DLL.
2. **The DLL, `v7_threading.dll` in assembly**, computes only and never touches files.
   It receives the preloaded image buffer, encodes it through the ViT, injects the
   resulting embeddings into the text LLM's forward pass, and generates the answer.

This separation is a hard architectural rule: the assembly DLL must never call `CreateFileA`,
`ReadFile`, or any file I/O API. All data comes from the microdb for weights, config and tokenizer, or
from caller-supplied buffers holding raw image pixels.

#### End-to-End Flow

```
CALLER (dbg_image_recog.exe)
    |
    |  1. Parse args: -q8 <0|1> <dll_path> <db_path> <raw_image> "<question>"
    |  2. v7_load(dll_path)        -> LoadLibraryA + GetProcAddress
    |  3. v7_init(db_path)         -> microdb load (decompression)
    |  4. v7_set_quantize(0|1)     -> select F32/Q8 forward
    |  5. v7_initialize_inference()-> context created inside the DLL (BSS)
    |  6. Read.meta sidecar        -> "width height"
    |  7. Read.raw file            -> raw RGB bytes (W*H*3)
    |  8. v7_chat_image(question, NULL, raw_rgb, W, H, out, out_len, reset)
    |  9. Print output, v7_free_inference(), v7_unload
    v
v7_chat_image (public_api.inc -> v7s_chat_image_mem, chat_image.inc)
    |
    |  A. Wipe vision globals (fresh session)
    |  B. v7s_vit_encode_callback(raw_rgb, W, H, &emb, &cnt, &ds0, &ds1, &ds2)
    |  C. v7s_set_vision_embeddings(emb, cnt, ds0, ds1, ds2)
    |  D. Build marker prompt:
    |       <|vision_start|> + <|image_pad|>*N + <|vision_end|> + user_question
    |  E. Delegate to the chat processing path -> prefill + generate with vision hooks
    v
text output

ViT ENCODE (vit_forward.inc)         TEXT LLM GENERATION (chat.inc -> forward_*.inc)
    Normalize + patchify                 Prefill: for each <|image_pad|> token,
    Position embed                       replace the embedding row with the next
    ViT blocks at the configured depth   g_vision_embeddings row (cursor:
    Project to text dim                  g_vision_emb_consumed)
    Packed out: main [N x 2560]          After decoder layers 0/1/2: add the
            + ds0/ds1/ds2                DeepStack feature row
                                         (g_deepstack_features[0..2])
                                         Generate tokens -> output text
```

#### Component Details

#### 1. Caller, `dbg_image_recog.c`

A C console tool built by `build.bat`.

**Responsibilities:**
- Parse CLI: `-q8 <0|1>`, `<dll_path>`, `<db_path>`, `<raw_image_path>`, `<question>`
- `v7_load` (LoadLibraryA + GetProcAddress) for the public `v7_*` tier
- `v7_init(db_path)` - microdb load
- `v7_set_quantize(mode)` then `v7_initialize_inference()`
- Load the `.meta` sidecar (text: `"width height\n"`) - parse `img_w`, `img_h`
- Load the `.raw` file (binary: `W*H*3` bytes of raw RGB)
- `v7_chat_image(question, NULL, raw_rgb, img_w, img_h, output, out_len, reset)`
- Print the output buffer, `v7_free_inference()`, `v7_unload`

#### 2. Image processing, `public_api.inc` -> `chat_image.inc`

The public `v7_chat_image` validates its arguments and calls the internal `v7s_chat_image_mem`, whose
9-argument internal signature takes the context, the user and system prompts, the raw RGB plus
dimensions, the output buffer and reset.

**Steps:**
1. **Wipe vision globals**, sets `g_vision_embeddings=NULL`,
   `g_vision_emb_count=0`, `g_vision_emb_consumed=0`, all DeepStack slots NULL,
   `g_cfg_has_deepstack=0`. This ensures a fresh session even if called repeatedly.
2. **Call ViT encoder**, `v7s_vit_encode_callback(raw_rgb, w, h, &emb, &cnt,
   &ds0, &ds1, &ds2)`, which returns 0 on success.
3. **Set vision embeddings**, `v7s_set_vision_embeddings(emb, cnt, ds0, ds1, &ds2)`
   stores pointers in BSS globals for the forward pass to read.
4. **Build marker prompt**, concatenates into the 64KB BSS buffer
   `g_vision_prompt_buf`:
   ```
   <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>What is in this image?
   ```
   The count of `<\|image_pad\|>` tokens equals `emb_cnt`, the number of
   merged vision tokens from the ViT. The BPE tokenizer maps these marker
   strings to the Qwen3-VL special token IDs 151652, 151655 and 151653 at
   encode time.
5. **Delegate to `v7s_chat`**, `v7s_chat(ctx, marker_prompt, system_prompt,
   output, output_len, reset)`. This runs the standard text generation loop,
   but with vision embedding injection active.

**`copy_vision_token` helper:** Appends one NUL-terminated marker string to the prompt buffer via
`movsb`. The copy includes the NUL, then backs `rdi` up to point AT the NUL position, so the next
call's first `movsb` overwrites that NUL; this keeps markers contiguous with no embedded NUL bytes.
The final call, for the user prompt, leaves the NUL as the buffer terminator.

#### 3. `v7s_vit_encode_callback` - `vit_forward.inc`

Signature (8 args):
```
int v7s_vit_encode_callback(const unsigned char* raw_rgb,
                            int img_w, int img_h,
                            float** out_emb, int* out_count,
                            float** out_ds0, float** out_ds1, float** out_ds2)
```

**Compute-only; the function does not perform file I/O.** All weights come from microdb via `ht_find`.

**Steps.** The concrete numbers below are Qwen3-VL-4B's; the other VLMs use their own VCFG values, and
the same steps run for all of them.

1. **Normalize**, convert raw RGB to float and apply `x/127.5 - 1.0`,
   which is the mean and standard deviation of 0.5 in the Qwen3-VL config.
2. **Patchify**, extract 16x16 patches and flatten them to 1536 dims, 3 channels
   times 16 times 16. This produces `gh x gw` patches where `gh = H/16` and `gw = W/16`.
3. **Position embed**, add learned 2D position embeddings, bilinear-resampled
   to the grid. The ViT also applies its own 2D RoPE with head_dim=64.
4. **Merge**, with spatial_merge=2, merge each 2x2 block of patches into one
   token. The output is `n_merged = (gh/2) x (gw/2)` tokens of hidden_dim=1024.
5. **ViT forward**, `v7s_vit_forward` runs a 24-layer transformer with GELU-tanh
   activation, 16 attention heads at head_dim=64, and intermediate_dim=4096.
   DeepStack features are extracted at layers 5, 11 and 17.
6. **Project**, a linear projection from ViT hidden_dim=1024 to text
   hidden_dim=2560.
7. **Output**, returns the main embeddings `[n_merged x 2560]` plus three
   DeepStack feature tensors, each `[n_merged x 2560]`.

#### 4. `v7s_set_vision_embeddings`, `chat_image.inc`

Signature, 5 arguments:
```
void v7s_set_vision_embeddings(float* embeddings, int count,
                               float* ds0, float* ds1, float* ds2)
```

Stores all five pointers and values in BSS globals:
- `g_vision_embeddings` holds embeddings
- `g_vision_emb_count` holds count
- `g_vision_emb_consumed` is set to 0 to reset the cursor
- `g_deepstack_features[0]` holds ds0
- `g_deepstack_features[1]` holds ds1
- `g_deepstack_features[2]` holds ds2
- `g_cfg_has_deepstack` is 1 when ds0 is not NULL and 0 otherwise

#### 5. Vision Embedding Injection, `forward_*.inc` / `prefill_*.inc`

During the text LLM's prefill and per-token forward pass, two vision hooks run.

**Hook A, image-token embedding override, step 0b in forward:**

When `g_image_token_id != 0`, meaning this is a vision model, and the current token equals the image-pad
token 151655:
- Read the next row from `g_vision_embeddings` at index `g_vision_emb_consumed`
- Overwrite `s->x`, the token embedding, with this row of `dim` floats via `v7d_memcpy`
- Increment `g_vision_emb_consumed`

This replaces the meaningless learned embedding of the `<\|image_pad\|>` token with the actual image
content from the ViT. Each `<\|image_pad\|>` token in the prompt gets one row from the vision embedding
pool.

**Hook B, DeepStack feature injection after decoder layers 0, 1 and 2:**

When `g_cfg_has_deepstack != 0` and the current token is an image-pad token:
- Read the corresponding DeepStack feature row from `g_deepstack_features[layer]`
- Add it element-wise into the hidden state `s->x` after the residual

This injects mid-level visual features from ViT layers 5, 11 and 17 into the text decoder's early layers
0, 1 and 2, providing multi-scale visual information.

#### 6. BSS Globals, `v7_data.inc`

| Global | Type | Purpose |
| --- | --- | --- |
| `g_image_token_id` | dword | 151655 for Qwen3-VL; 0 for text-only, which makes the hooks inactive |
| `g_vision_embeddings` | qword pointer | Base of the projected embeddings `[N x text_dim]` |
| `g_vision_emb_count` | dword | Number of vision embedding rows, N |
| `g_vision_emb_consumed` | dword | Cursor for the next row to inject, reset per chat_image call |
| `g_deepstack_features[3]` | 3 qword pointers | The ds0, ds1 and ds2 feature tensors `[N x text_dim]` |
| `g_cfg_has_deepstack` | dword | 1 applies the DeepStack adds, 0 skips them |
| `g_vision_prompt_buf` | 65536 bytes | Scratch buffer for marker prompt construction |

All globals are zero-initialized. For text-only models, the internal init `v7s_inference_create`,
wrapped by the public `v7_initialize_inference`, detects the absence of `__sys_vision_config` in the DB
and leaves all vision globals at 0, which makes the forward-pass hooks no-ops.

#### 7. Vision Kernel Inventory: Kernels, Call Chain, Dispatch Status

The vision tower compute is stored in three files, following the project's kernel and dispatch
separation rule: math kernels are in `vit_kernel.inc`, dispatch entry points in `vit_dispatch.inc`, and
the orchestrator plus the inline single-threaded helpers are in `vit_forward.inc`.

#### 7.1 Kernel files

| File | Role |
| --- | --- |
| `vit_forward.inc` | Orchestrator for `v7s_vit_encode_callback` and `v7s_vit_forward`, plus inline single-threaded helpers: layernorm, gelu, add, pack2x2, pos_embed, rope_build, rope_apply |
| `vit_kernel.inc` | The two dispatched chunk kernels, `v7s_vit_linear_chunk` and `v7s_vit_attention_chunk`, which run on the worker pool via `process_job` |
| `vit_dispatch.inc` | Dispatch entry points `v7d_vit_linear` and `v7d_vit_attention`, which fill `jobs_arr[]` and call `DISPATCH_EXECUTE` |

#### 7.2 Kernel inventory

**Dispatched kernels, multi-threaded through the unified `DISPATCH_EXECUTE` codepath:**

`v7d_vit_linear` (vit_dispatch.inc) -> `v7s_vit_linear_chunk` (vit_kernel.inc):
- F32xF32 matmul with optional bias: `out[n_rows][M] = in[n_rows][K] x w[M][K]^T + bias[M]`
- Job decomposition: split M into `min(num_workers+1, M/64)` uniform column chunks;
  one job = one column range across all n_rows.
- Globals `g_vit_k` (K) and `g_vit_m` (M) are set once per call for the chunk kernel.
- Job mode `MATMUL_VIT_LINEAR` (11); routed in `process_job` (v7_instr.inc).

`v7d_vit_attention` (vit_dispatch.inc) -> `v7s_vit_attention_chunk` (vit_kernel.inc):
- Bidirectional attention (no causal mask, no KV cache):
  `softmax(QK^T x 1/sqrt(head_dim)) @ V` over all n patches.
- Job decomposition: one job per attention head (parallel heads).
- Each head gets its own `n x n` scores slice (base + `h*n*n*4`) to avoid data races.
- Job mode `MATMUL_VIT_ATTENTION` (12); routed in `process_job` (v7_instr.inc).

**Single-threaded inline helpers in `vit_forward.inc`, not dispatched, running directly on the caller
thread:**

| Helper | Compute |
| --- | --- |
| `vit_layernorm` | LayerNorm (mean + var + gamma/beta) over n_rows x dim; exact 1/std via `vdivss` |
| `vit_gelu` | GELU-tanh in-place, vectorized 8 floats/iteration |
| `vit_add` | element-wise `dst[i] += src[i]` |
| `vit_pack2x2` | rearrange block-major patches into packed rows (spatial merge 2x2) |
| `vit_pos_embed` | bilinear-resample the 48x48 learned pos_embed table to grid_h x grid_w |
| `vit_rope_build` | build the 2D RoPE cos/sin tables via x87 fsincos |
| `vit_rope_apply` | rotate Q and K inside the interleaved qkv buffer |

#### 7.3 Call chain inside `v7s_vit_forward`

Per image, one `v7s_vit_encode_callback` runs this. The shapes shown are Qwen3-VL-4B's; the loop count
comes from `VCFG_DEPTH`.

```
v7s_vit_encode_callback (entry)
 |-- v7s_vit_forward
 |    |-- set_spin(1)                       workers commit to spinning
 |    |-- v7d_vit_linear  pe_w*pixels       (DISPATCHED, M=1024, K=1536)   patch embed
 |    |-- vit_pos_embed                     (inline)                        pos emb
 |    |-- vit_add                           (inline)                        pos +=
 |    |-- vit_rope_build                    (inline)                        cos/sin tables
 |    |    VCFG_DEPTH x layer L:
 |    |     |-- vit_layernorm               (inline)                        pre-attn LN
 |    |     |-- v7d_vit_linear  qkv_w*norm  (DISPATCHED, M=3072, K=1024)
 |    |     |-- vit_rope_apply              (inline)                        Q/K rotate
 |    |     |-- v7d_vit_attention           (DISPATCHED, one job per head)  attn
 |    |     |-- v7d_vit_linear  proj_w*attn (DISPATCHED, M=1024, K=1024)
 |    |     |-- vit_add                     (inline)                        attn residual
 |    |     |-- vit_layernorm               (inline)                        pre-MLP LN
 |    |     |-- v7d_vit_linear  fc1_w*norm  (DISPATCHED, M=4096, K=1024)
 |    |     |-- vit_gelu                    (inline)                        GELU
 |    |     |-- v7d_vit_linear  fc2_w*mlp   (DISPATCHED, M=1024, K=4096)
 |    |     +-- vit_add                     (inline)                        MLP residual
 |    |     DeepStack capture at the VCFG_DS_TAP layers, 5/11/17 for Qwen3-VL:
 |    |     pack2x2 + LN + fc1 (DISPATCHED, M=4096, K=4096)
 |    |                    + GELU + fc2 (DISPATCHED, M=2560, K=4096)
 |    |-- [no set_spin(0) here]             chat_finish calls v7s_stop_spin at turn completion
 |    +-- main merger: LN + pack2x2 + fc1 (DISPATCHED, M=4096, K=4096) + GELU
                           + fc2 (DISPATCHED, M=2560, K=4096)
```

For Qwen3-VL-4B the dispatch count per image works out to 128 `DISPATCH_EXECUTE` calls, all on the same
worker pool: 4 linears per layer across 24 layers is 96, plus 24 attentions, plus 3 DeepStack mergers at
2 linears each is 6, plus 2 main-merger linears.

#### 7.4 How a dispatched job reaches the kernel

1. `v7d_vit_linear` / `v7d_vit_attention` fill `jobs_arr[]` via the `JOB_SET_*`
   macros and call `DISPATCH_EXECUTE`.
2. `DISPATCH_EXECUTE` publishes `gen_counter`, wakes sleeping workers, runs the
   caller steal loop.
3. Workers (and the caller) claim jobs via `lock xadd next_job` and call
   `process_job` (v7_instr.inc).
4. `process_job` switches on `JOB_MODE`: `MATMUL_VIT_LINEAR` (11) ->
   `v7s_vit_linear_chunk`; `MATMUL_VIT_ATTENTION` (12) -> `v7s_vit_attention_chunk`.
5. Chunk kernels run; `remaining_jobs` is decremented; the caller pause-spins
   to completion.

#### 7.5 Not dispatched, by design

The small element-wise and table-build ops, namely layernorm, gelu, add, pack2x2, pos_embed, rope_build
and rope_apply, run inline on the caller thread. They are memory-bound micro-ops over vectors of at most
4096 dims where dispatch overhead could exceed the work. Only the heavy FLOP workloads, the matmuls and
attention, are dispatched.

#### Token IDs, Qwen3-VL-4B-Instruct

| Token | ID | Purpose |
| --- | --- | --- |
| `<\|vision_start\|>` | 151652 | Opens the vision content block |
| `<\|image_pad\|>` | 151655 | Image placeholder, one per merged vision token |
| `<\|vision_end\|>` | 151653 | Closes the vision content block |

These are stored in the tokenizer vocab and resolved by the BPE encoder's special-token scan, a `<...>`
bracket scan plus `ht_find` in encode.inc.

#### File Formats

#### `.raw` file
A `.raw` file is a headerless binary file containing `W x H x 3` bytes of RGB pixel data, with channels ordered R, G, B per
pixel and rows running top to bottom.

#### `.meta` sidecar file
The `.meta` sidecar contains ASCII text on a single line: `"width height\n"` with space-separated decimal integers. The harness
derives the `.meta` path from the `.raw` path using the same prefix.

An example is `"64 64\n"` for a 64x64 image.

#### Test Command

```
bin\dbg_image_recog.exe bin\v7_threading.dll -q8 0 weights\databases\qwen3_vl_4b\lz4ht.db cat640.raw "What is in this image?"
```

A successful run produces a description of the cat in the image.

### AnyRes patchification: experimental evaluation (October 2026)

The `vit_anyres.inc` patchify block works from its `r12` destination base and the `vcb_cdst`
channel-offset accumulator. A rewrite was implemented and then **discarded**: it replaced the
per-row/per-channel cursor chain with a single `rdi` destination cursor, an `ebx` combined row cursor
and one hoisted per-channel source re-base, claiming -24 instructions per patch (~52k per tile) with
bit-identical output.

It was not bit-identical. It also shipped half-removed: the rewrite dropped the h-loop back-edge
entirely (patch rows 1..728 of every tile were never written) and the regression surfaced as a changed
LLaVA-OV answer in both F32 and Q8:

```
before: "The image features a cat with a striped coat, likely a tabby, sitting on
         a brick surface. The cat's eyes are a striking green color..."
after:  "The image appears to be a distorted or abstract representation of a cat.
         The cat's face is partially obscured by a pattern of colored stripes..."
```

The back-edge was restored (`fbf05963`) and the anyres gate added to verify_vision.bat
(`run_vision_smoke_llava.bat`, six byte baselines) so this file is no longer exercised only by a
dev-only smoke. A separate pad-origin unit-mix bug (left_pad scaled by 3 twice, commit `983da0b6` )
showed up only on height-limited images once that gate existed.

**Do not re-file the cursor rewrite.** Any replacement of the patchify cursors must be proven against
the anyres gate's byte baselines, not against the Qwen3-VL vision smoke, the latter never reaches this
file.
