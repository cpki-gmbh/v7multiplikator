# MicroDB and Binary Data Layouts

The MicroDB format stores the data needed to initialize a model alongside application-defined values. A
typical V7 database contains weights, tokenizer resources, model parameters, prompt templates, and
configuration for the native runtime. The same database interface is also available to SectorLogo
programs and native applications.

This reference defines the key names, value representations, and structure layouts expected by the
documented implementation. These details form a binary compatibility contract between the database
builder and the DLL, so field names, offsets, widths, and byte representations are stated precisely. For
a general introduction to preparing a database, see the getting-started manual; use this reference when
implementing exporters, inspecting stored data, or modifying the assembly interfaces.

## Contents

- [MicroDB key reference](#microdb-key-reference)
- [Assembly data layouts](#assembly-data-layouts)

---

## MicroDB key reference

A V7 model database contains named entries with defined binary representations. The builder writes model
resources and runtime configuration, while the DLL looks up those entries during initialization.
Applications and SectorLogo scripts can also use the database for their own records. This chapter
specifies which keys are required, how their values are represented, and where they are consumed. The
next chapter covers the memory layouts used after those values have been loaded.

### Overview

MicroDB stores model data and application records, but the existence of an in-memory model database does
not imply that every application-level feature is free of disk operations. The optional retrieval
executables and persistent memory examples can read or append files during a generation turn. The
database format describes how the model is loaded, not a prohibition on I/O performed by the host
application or a SectorLogo script.

The file `lz4ht.db` contains named records in the MicroDB format. During initialization, `db_load()`
opens the database and `ht_find(key)` locates records by name. The model builder writes the required
weights, tokenizer information, configuration values, and dispatch data, while applications may use
additional keys for their own stored values. The key definitions in this chapter specify the
representation of each record and identify the functions that read or produce it.

A model database is not merely a collection of optional metadata. The runtime interprets many of these
records as a binary interface, so their lengths, numeric types, and ordering must match what the
assembly code expects. Exporters and inspection utilities should therefore use the definitions here
together with the corresponding source declarations. Applications that only load an existing database
normally use `v7_init()` followed by `v7_initialize_inference()` rather than accessing these internal
keys directly.

### System Keys (__sys_*)

The `__sys_*` keys contain model and runtime resources read during initialization. Their binary
representations are part of the database compatibility contract; the database builder must write values
in the layouts expected by the corresponding version of the DLL.

#### __sys_config - Model Configuration

- **Type:** Raw v7_config struct (64 bytes)
- **Stored by:** build_db.exe step 1
- **Read by:** v7s_inference_create (inf_weights.inc, inf_microdb.inc)

| Offset | Field | Type | Description |
| --- | --- | --- | --- |
| 0 | reserved0 | int32 | Reserved (always 0) |
| 4 | reserved1 | int32 | Reserved (always 0) |
| 8 | dim | int32 | Model dimension (e.g. 1024) |
| 12 | hidden_dim | int32 | FFN hidden dimension (e.g. 3072) |
| 16 | n_layers | int32 | Number of transformer layers |
| 20 | n_heads | int32 | Number of attention heads |
| 24 | n_kv_heads | int32 | Number of KV heads (GQA) |
| 28 | vocab_size | int32 | Vocabulary size |
| 32 | seq_len | int32 | Maximum sequence length |
| 36 | head_dim | int32 | Per-head dimension |
| 40 | shared_classifier | int32 | 1 = wcls shares q_tokens embedding |
| 44 | group_size | int32 | Quantization group size (typically 64) |
| 48 | has_qk_norm | int32 | 1 = model uses QK normalization |
| 52 | rope_base | int32 | RoPE base frequency (e.g. 1000000) |
| 56 | reserved2 | int32 | Unused |
| 60 | reserved3 | int32 | Unused |

#### __sys_rope_base - RoPE Base Frequency

The value is stored as **float32 (4 bytes)**. The database builder writes it during **build_db.exe step
2**, and **rope_build_internal_tables (fused_f32_kernel.inc)** reads it during initialization.

The key contains the RoPE base frequency as a 32-bit floating-point value. The `build_db.exe -rope-base`
option can override its value. Qwen3 uses 1000000.0; SmolLM2 uses 100000.0; Phi-3 uses 10000.0.

#### __sys_rope_type - RoPE Scaling Type

The value is stored as **int32 (4 bytes)**. The database builder writes it during **build_db.exe step
2**, and **rope_build_internal_tables (fused_f32_kernel.inc)** reads it during initialization.

The key selects the RoPE scaling mode. A value of `0` uses the ordinary base^(-j/half) frequency
calculation, while `1` enables Llama 3 scaled RoPE with interpolation between the high- and
low-frequency boundaries. The database builder always writes the key, using `0` for checkpoints without
a scaling configuration. The value can also be supplied explicitly with `build_db.exe -rope-type`.

#### __sys_rope_factor - Llama3 Low-Frequency Division Factor

The value is stored as **float32 (4 bytes)**. The database builder writes it during **build_db.exe step
2**, and **rope_build_internal_tables (fused_f32_kernel.inc)** reads it during initialization.

This key contains the Llama 3 `factor` setting, which controls the low-frequency division factor and
may, for example, be `32.0`; it is used when `__sys_rope_type == 1`. The builder writes the key for
every model, using `1.0` when the source configuration does not specify a value. The `build_db.exe
-rope-factor` option provides an explicit override.

#### __sys_rope_high_freq_factor - Llama3 High-Frequency Boundary

The value is stored as **float32 (4 bytes)**. The database builder writes it during **build_db.exe step
2**, and **rope_build_internal_tables (fused_f32_kernel.inc)** reads it during initialization.

The `high_freq_factor` value defines the high-frequency wavelength boundary for Llama 3 scaling; it is
relevant only when `__sys_rope_type == 1`. The builder always writes it, using `4.0` when the model
configuration omits the value. An application preparing the database can override it with
`build_db.exe -rope-high-freq-factor`.

#### __sys_rope_low_freq_factor - Llama3 Low-Frequency Boundary

The value is stored as **float32 (4 bytes)**. The database builder writes it during **build_db.exe step
2**, and **rope_build_internal_tables (fused_f32_kernel.inc)** reads it during initialization.

The `low_freq_factor` value defines the low-frequency wavelength boundary for Llama 3 scaling. The value
is used only when `__sys_rope_type == 1`. The builder writes it for every model, using `1.0` when the
configuration does not specify another value; `build_db.exe -rope-low-freq-factor` can override it.

#### __sys_rope_orig_max_pos - Llama3 Original Max Position

The value is stored as **int32 (4 bytes)**. The database builder writes it during **build_db.exe step
2**, and **rope_build_internal_tables (fused_f32_kernel.inc)** reads it during initialization.

This key records the `original_max_position_embeddings` value used by Llama 3 scaling, such as `8192`;
it is meaningful when `__sys_rope_type == 1`. The builder always writes the key and uses `0` if the
source configuration does not supply the value. The `build_db.exe -rope-orig-max-pos` option can
override it.

#### __sys_rms_eps - RMSNorm Epsilon

The value is stored as **float32 (4 bytes)**. The database builder writes it during **build_db.exe step
2**, and **v7s_inference_create (inf_microdb.inc step 3b)** reads it during initialization.

This key specifies the model's RMSNorm epsilon, read from byte offset 84 of the checkpoint header. Qwen3
uses `1e-6`, while Llama-3.2, Phi-3, and SmolLM2 use `1e-5`. For older checkpoint headers that do not
specify epsilon, the builder uses `1e-6`. During initialization, V7 stores the value in `g_rms_eps`; the
RMSNorm kernels use that stored value rather than a fixed constant.

#### __sys_stop_token_ids - Additional Stop Tokens

**Type:** int32 array: [count, id0, id1,...] (max 4 ids + count = 20 bytes) **Stored by:** build_db.exe
step 5 (only when more than 1 stop token) **Read by:** v7s_inference_create (inf_tokenizer.inc step 15c)

End-of-generation tokens beyond eos_token_id, found by scanning the vocab for `<|endoftext|>`,
`<|end|>`, `<|eot_id|>`, and `<|end_of_text|>`. Written only when the tokenizer has more than one stop
token (e.g. Phi-3 adds `<|end|>` =32007, Qwen3 adds `<|endoftext|>` =151643). Absent when eos_token_id
is the only stop token; the ASM path then uses just the eos id. Read by chat.inc for multi-token stop
detection during generation.

#### __sys_model_weights - Model Weight Data

**Type:** Raw blob (variable, typically hundreds of MB to several GB) **Stored by:** build_db.exe step 3
(past 256-byte checkpoint header) **Read by:** v7s_inference_create (inf_weights.inc)

The weight payload begins at byte offset 256 of the exported `.bin` checkpoint, after its fixed-size
header. The entries in `__sys_weight_meta` describe the offsets of individual tensors relative to the
beginning of that payload. Runtime pointers are calculated as follows:

    weight_ptr = model_weights_base + weight_meta[i].file_offset

#### __sys_weight_meta - Weight Metadata Table

**Type:** Array of 16 entries, 24 bytes each = 384 bytes total **Stored by:** build_db.exe step 4 **Read
by:** v7s_inference_create (inf_weights.inc)

Each metadata entry identifies a single weight tensor and provides the fields needed to locate its stored data:

| Offset | Field | Type | Description |
| --- | --- | --- | --- |
| 0 | file_offset | size_t (8 bytes) | Byte offset into model_weights blob |
| 8 | rows | uint32 | Total element count |
| 12 | cols | uint32 | Always 0 |
| 16 | is_quantized | uint8 | 1 = int8, 0 = FP32 |
| 17-19 | reserved | uint8[3] | Padding |

The weight metadata table assigns indices 0 through 15 to the following tensor categories:

| Index | Weight | Q8 | Rows formula |
| --- | --- | --- | --- |
| 0 | rms_att_weight | No | n_layers * dim |
| 1 | rms_ffn_weight | No | n_layers * dim |
| 2 | rms_final_weight | No | dim |
| 3 | q_norm_weights | No | n_layers * head_dim |
| 4 | k_norm_weights | No | n_layers * head_dim |
| 5 | q_tokens (embed) | Yes | vocab_size * dim |
| 6 | wq (Q projection) | Yes | n_layers * dim * n_heads * head_dim |
| 7 | wk (K projection) | Yes | n_layers * dim * n_kv_heads * head_dim |
| 8 | wv (V projection) | Yes | n_layers * dim * n_kv_heads * head_dim |
| 9 | wo (output proj) | Yes | n_layers * n_heads * head_dim * dim |
| 10 | w1 (FFN gate) | Yes | n_layers * dim * hidden_dim |
| 11 | w2 (FFN down) | Yes | n_layers * hidden_dim * dim |
| 12 | w3 (FFN up) | Yes | n_layers * dim * hidden_dim |
| 13 | q_tokens_copy | Yes | Copy of index 5 |
| 14 | wcls (classifier) | Yes | dim * vocab_size (or same as 5 if shared) |
| 15 | embed_ref | No | vocab_size * dim (reference pointer) |

For quantized tensors, each record stores the signed eight-bit weight data followed by the associated
FP32 scale values. The documented layout uses `ceil(rows / group_size)` scale values for the
corresponding grouping.

#### __sys_mem_layout - RunState Memory Layout

- **Type:** MemLayout struct (160 bytes, 20 x size_t)
- **Stored by:** build_db.exe step 12
- **Read by:** v7s_inference_create (inf_microdb.inc, inf_state.inc)

The following structure is written by the builder as **legacy layout metadata**. The current ASM runtime uses the working-buffer prefix through `off_key_cache`, then separately allocates FP16 key/value caches; it does not allocate the full legacy `rs_block_bytes`:

- reserved0 - reserved (always 0)
- rs_block_bytes - total block size in bytes
- off_x, off_xb, off_hb, off_hb2 - activation buffer offsets
- off_xq_q, off_xq_s, off_hq_q, off_hq_s - quantized tensor offsets
- off_q, off_att, off_logits - attention/logit offsets
- off_key_cache, off_value_cache - KV cache offsets
- off_x_storage, off_q_storage, off_att_out - pre-allocated storage offsets
- emb_table_bytes - embedding table size (vocab_size * dim * 4)
- vocab_ptrs_bytes - vocab pointer array size (vocab_size * 8)

The described structure occupies the first 160 bytes of the serialized blob.

#### __sys_tokenizer_hdr - Tokenizer Header

The value is stored as **12 bytes**; it is written by **build_db.exe step 5** and read by
**v7s_inference_create**. (inf_tokenizer.inc)

| Offset | Field | Type | Description |
| --- | --- | --- | --- |
| 0 | max_token_len | int32 | Longest token string in vocab |
| 4 | bos_token_id | uint32 | Beginning-of-sequence token ID |
| 8 | eos_token_id | uint32 | End-of-sequence token ID |

#### __sys_merge_scores - Token Merge Scores

The value is stored as a **float32 array containing vocab_size entries (vocab_size * 4 bytes)**. The database
builder writes it during **build_db.exe step 6**, and **v7s_inference_create (inf_tokenizer.inc)** reads
it during initialization.

The tokenizer uses these BPE merge-priority scores to select between competing merges. During
initialization, the values are copied into the `Tokenizer` structure.

#### __sys_vocab_bulk - Vocabulary Bulk Data

The value is stored as a **packed binary blob**. The database builder writes it during **build_db.exe step
7**, and **v7s_inference_create (inf_tokenizer.inc)** reads it during initialization.

The vocabulary is stored as a packed sequence of token strings in the following binary format:

| Offset | Field | Type |
| --- | --- | --- |
| 0 | count | uint32 - number of tokens |
| 4+ | per token | uint32 len + uint8[len] |

During initialization, the loader builds a `vocab_ptrs` array containing pointers into the packed
vocabulary. Its total serialized size is `4 + sum(4 + len_i)` bytes across all tokens.

#### __sys_template, Prompt Template for turn 1, base variant

The value is stored as a **null-terminated string of at most 1,023 bytes**. The database builder writes it during
**build_db.exe step 11, mandatory because the export contract always writes the sidecar and a missing
file is a build-time terminal error**, and **v7s_inference_create in inf_tokenizer.inc, then v7s_chat in
chat.inc** reads it during initialization.

Turn-1 user prompt template with a single `%s` placeholder, rendered by `v7s_chat` when no system prompt
is given and `ctx->pos == 0`. Loaded from the `.template` sidecar, which is the non-thinking variant.
The engine FATALs at init if this key is missing.

#### __sys_template_sys, System Prompt Template, base variant

The value is stored as a **null-terminated string of at most 1,023 bytes**. The database builder writes it during
**build_db.exe step 11, mandatory because a missing sidecar is a build-time terminal error**, and
**v7s_inference_create in inf_tokenizer.inc, then v7s_chat in chat.inc** reads it during initialization.

System and user prompt template with two `%s` placeholders, system first and user second, rendered when
`system_prompt != NULL`. Loaded from the `.template.with-system` sidecar, which is the non-thinking
variant. The engine FATALs at init if this key is missing.

#### __sys_template@think / __sys_template_sys@think, Thinking variants, optional

**Type:** A null-terminated string of at most 1,023 bytes for each variant.
**Stored by:** `build_db.exe` at step 11, only when the optional sidecar exists and differs from the base variant.
**Read by:** `v7s_inference_create` in `inf_tokenizer.inc`, followed by the `v7_enable_thinking` selector in `chat.inc`.

Thinking-on turn-1 and system-plus-user variants, from `.template.with-thinking` and
`.template.with-system-and-thinking`. Models whose thinking sidecars equal the base, because their chat
template has no `enable_thinking`, store nothing. `v7s_enable_thinking(1)`, which is the engine default
and resets per context, selects these, and `v7s_enable_thinking(0)` falls back to the base pair. The
base variants end with an empty think block because the model does not think, while the @think variants
end at the assistant header so the model generates its own think content.

#### __sys_template@img, Image-turn template, optional

**Type:** Null-terminated string, max 1023 bytes **Stored by:** build_db.exe step 11, optional and
stored only when `.template.img` exists and differs from base **Read by:** v7s_chat in chat.inc on image
turns

Image-turn variant for models whose chat template renders image turns differently, for example
LLaVA-OneVision. `v7s_chat_image_mem` and `v7_generate_image_mem` set an image-turn flag, and the render
path then prefers this template over the multi and think selection. The flag self-clears after the
render. When the key is absent, image turns use the base template.

#### __sys_template@multi / __sys_template@multi@think, Inter-turn templates, optional

**Type:** Null-terminated string, max 1023 bytes each **Stored by:** build_db.exe step 11, optional and
stored only when the sidecar exists and differs from base or from plain @multi **Read by:**
v7s_inference_create in inf_tokenizer.inc, then v7s_chat in chat.inc

Complete turn-2+ user templates made of a terminator, a user header, `%s` and a generation suffix. They
are rendered when `system_prompt == NULL` and `ctx->pos > 0`, and come from `.template.multi` and
`.template.with-system-and-thinking` 's multi counterpart. If the model has no `@multi` key, turn 2+
falls back to the turn-1 template. `@multi@think` is stored only when it differs from plain `@multi`,
which is the case for models without thinking variants because those produce identical files.

#### __sys_vision_config - Vision Config Header (expanded VCFG)

**Type:** Expanded struct = 16 x int32 raw header (offsets 0-60, verbatim from the `.vision` file) +
precomputed derived qword fields (offset 64+) = **744 bytes** (`VCFG_QWORDS = 93`). The final field is `VCFG_PS_OUT_W` at offset 736

**Stored by:** build_db.exe `-vision` step (from the `.vision` file produced by export_vlm.py;
`build_vision_cfg()` in db_build_tool.c precomputes every derived value) **Read by:** ASM init copies
the blob verbatim into the `g_vit_cfg` BSS block; the ViT kernels (vit_forward.inc / vit_kernel.inc /
vit_dispatch.inc) read fields with single-instruction RIP-relative memory operands. The C-side ViT also
reads the raw header (Qwen3-VL only).

The first 64 bytes contain sixteen 32-bit integer fields at offsets 0 through 60. These offsets match
the corresponding values stored in the `.vision` file header:

| Offset | Field | Type | Description |
| --- | --- | --- | --- |
| 0 | image_token_id | int32 | Image placeholder token id (151655, `<\|image_pad\|>`) |
| 4 | vision_start_id | int32 | `<\|vision_start\|>` token id (151652) |
| 8 | vision_end_id | int32 | `<\|vision_end\|>` token id (151653) |
| 12 | has_deepstack | int32 | 1 = apply DeepStack adds in text decoder layers 0/1/2 |
| 16 | vision_hidden | int32 | ViT hidden dim (1024) |
| 20 | vision_out_hidden | int32 | Projected vision dim = text dim (2560) |
| 24 | patch_size | int32 | ViT patch size (16) |
| 28 | spatial_merge_size | int32 | Spatial merge size (2) |
| 32 | temporal_patch_size | int32 | Temporal patch size (2) |
| 36 | num_position_embeddings | int32 | Learned position table size (2304) |
| 40 | deepstack_0 | int32 | DeepStack tap block 0 (5) |
| 44 | deepstack_1 | int32 | DeepStack tap block 1 (11) |
| 48 | deepstack_2 | int32 | DeepStack tap block 2 (17) |
| 52 | reserved | int32 | Reserved (0) |
| 56 | reserved | int32 | Reserved (0) |
| 60 | reserved | int32 | Reserved (0) |

The computed eight-byte fields begin at offset 64. The `VCFG_*` offsets must match the definitions in
`inf_structs.inc` so that the database builder and assembly implementation interpret the same layout.

| Offset | Field | Value (Qwen3-VL-4B) |
| --- | --- | --- |
| 64 | VCFG_HIDDEN | hidden (1024) |
| 72 | VCFG_HIDDEN_F4 | hidden * 4 |
| 80 | VCFG_INTER | MLP intermediate (4096) |
| 88 | VCFG_QKV_DIM | 3 * hidden |
| 96 | VCFG_QKV_STRIDE | qkv_dim * 4 |
| 104 | VCFG_HEADS | heads (16) |
| 112 | VCFG_HEAD_DIM | hidden / heads (64) |
| 120 | VCFG_HEAD_DIM_F4 | head_dim * 4 |
| 128 | VCFG_DEPTH | depth (24) |
| 136 | VCFG_NUM_POS | num_pos (2304) |
| 144 | VCFG_NUM_POS_F4 | num_pos * 4 |
| 152 | VCFG_GRID_SIDE | isqrt(num_pos) (48) |
| 160 | VCFG_GRID_MINUS1_I | grid_side - 1 (47) |
| 168 | VCFG_GRID_MINUS1_F | float bits of (float)(grid_side-1) |
| 176 | VCFG_MERGE | merge (2) |
| 184 | VCFG_MERGE_UNIT | merge * merge (4) |
| 192 | VCFG_PATCH_SIZE | patch_size (16) |
| 200 | VCFG_PATCH_SIZE_F4 | patch_size * 4 |
| 208 | VCFG_OUT_HIDDEN | out_hidden (2560) |
| 216 | VCFG_PATCH_DIM | 3 * temporal * patch_size^2 |
| 224 | VCFG_PATCH_DIM_F4 | patch_dim * 4 |
| 232 | VCFG_PACKED_DIM | hidden * merge_unit (4096) |
| 240 | VCFG_PACKED_DIM_F4 | packed_dim * 4 |
| 248 | VCFG_N_DEEPSTACK | 3 |
| 256 | VCFG_ROPE_QTR | head_dim / 4 |
| 264 | VCFG_ROPE_HALF | head_dim / 2 |
| 272 | VCFG_ROPE_HALF_F4 | rope_half * 4 |
| 280 | VCFG_DS_TAP0 | ds_tap0 (5) |
| 288 | VCFG_DS_TAP1 | ds_tap1 (11) |
| 296 | VCFG_DS_TAP2 | ds_tap2 (17) |
| 304-360 | VCFG_L_QKV_W/B, L_PROJ_W/B, L_FC1_W/B, L_FC2_W/B | per-layer tensor byte sizes |
| 368-432 | VCFG_BLK_*_STRIDE | per-block tensor byte strides (x depth) |
| 440-464 | VCFG_MG_* | merger tensor byte strides |
| 472-496 | VCFG_DS_* | DeepStack merger tensor byte strides |
| 504 | VCFG_PE_W_STRIDE | patch_dim * hidden * 4 |
| 512 | VCFG_POS_EMBED_BYTES | num_pos * hidden * 4 |

ASM init copies the blob verbatim into `g_vit_cfg`; kernels read all model-dependent values from it
(zero runtime arithmetic). Fields 304-512 are precomputed byte sizes/strides for weight-tensor
addressing.

#### __sys_vit_weights - ViT Tower Weight Blob

The value is stored as **Raw blob ('VIT1' format signature + fp32 vision tower weights, ~1.6 GB for
Qwen3-VL-4B)**. The database builder writes it during **build_db.exe `-vision` step (bytes past the
64-byte header of the `.vision` file)**, and **v7s_vit_encode_callback (vit_forward.inc) and the C-side
ViT (Qwen3-VL only)** reads it during initialization.

Contains the vision transformer weights: patch embedding, learned position embedding, 24 transformer
blocks (each with attention + MLP), the main merger (2x2 spatial reduction into the text hidden dim),
and the 3 DeepStack mergers. Absent for text-only models; the vision exports (`v7s_chat_image_mem`,
`v7s_generate_image_mem` ) FATAL if this key is missing.

#### __sys_vit_projector - Vision Projector Weights (LLaVA-OneVision only)

**Type:** Raw fp32 blob (variable; SigLIP vision hidden -> text hidden) **Stored by:** build_db.exe
`-visionproj <path>` step (LLaVA-OneVision only) **Read by:** v7s_vit_encode_callback (vit_forward.inc)
when `VCFG_PROJ_IN_DIM > 0`

SigLIP-based VLMs (LLaVA-OneVision) use a separate post-ViT projector (typically an MLP or linear layer)
that maps the vision tower's hidden dim to the text decoder's hidden dim. Qwen3-VL integrates its
projector inside the ViT blob, so this key is absent for Qwen3-VL (`VCFG_PROJ_IN_DIM == 0`).

#### __sys_vision_xy_res - Fixed Vision Input Resolution

**Type:** 2 x uint64 (16 bytes: target_w, target_h) **Stored by:** build_db.exe `-vision` step **Read
by:** v7s_vit_encode_callback (vit_input_stage.inc)

Fixed input image dimensions for the ViT pre-stage. SigLIP models (which have `VHDR_IMAGE_SIZE
> 0`) use their trained resolution (typically 384x384). Qwen3-VL is grid-based
(`VHDR_IMAGE_SIZE == 0`) and has no fixed size, so the build tool computes its NATIVE position grid
instead, isqrt(num_pos) x patch_size (48 x 16 = 768 for the 4B). (An earlier build used 224x224 here,
which starved the ViT to a 14x14 patch grid and destroyed spatial grounding -- box-test deviation 0.056
vs HF's 0.007; the native-grid fix restored 0.007.)

#### __sys_vision_prompt_fmt - Vision Prompt-Format Pack

**Type:** Three NUL-terminated strings packed back-to-back (variable, ~64 bytes) **Stored by:**
build_db.exe `-vision` step **Read by:** v7s_chat_image_mem / v7s_generate_image_mem (chat_image.inc)

The value contains three consecutive NUL-terminated strings, arranged as `[prefix\0] [marker\0]
[suffix\0]`. These strings determine the vision prompt prefix supplied to the text decoder; the assembly
code reads the stored format rather than selecting a format from the model name:

| Model | prefix | marker | suffix |
| --- | --- | --- | --- |
| Qwen3-VL | `<\|vision_start\|>` | `<\|image_pad\|>` | `<\|vision_end\|>` |
| LLaVA-OneVision | (empty) | `<image>` | `\n` |

ASM reads the blob once at init, caches three pointers (prefix/marker/suffix) in BSS, then builds the
prompt prefix as `prefix + marker*N + suffix + user_prompt` (: model differences are DATA, not parallel
code paths).

#### __sys_mrope_bound - Text M-RoPE Interleaved Pair Bound

The value is stored as **uint64 (8 bytes)**. The database builder writes it during **build_db.exe
`-vision` step (from `.vision` header VHDR_MROPE_BOUND)**, and **v7s_mrope_prepare (mrope.inc)** reads
it during initialization.

Nonzero only for qwen3-VL-style text models with `rope_scaling.mrope_section` + `mrope_interleaved`, the
number of leading RoPE pairs rotated with plain 1D positions before the multimodal (t/h/w) triplets
begin. 0 (or key absent) = plain 1D text RoPE; the ASM loader treats a missing key as 0, so older DBs
without the key stay valid. LLaVA-OneVision and SmolVLM DBs carry 0.

#### __sys_image_newline_emb - AnyRes Row Separator Embedding

The value is stored as **raw float32 array (text_hidden floats)**. The database builder writes it during
**build_db.exe `-vision` step; present only when `VHDR_ANYRES_ENABLED=1` (LLaVA-OneVision); loaded from
the `<vision_path>.newline` sidecar file**, and **pack_image_features (vit_anyres.inc)** reads it during
initialization.

The trained `image_newline` nn.Parameter of shape (text_hidden,) that gets concatenated between grid
rows in pack_image_features. It is not a vocabulary token, and it is absent for Qwen3-VL and SmolVLM,
which have no anyres.

#### stream_remove_text - StreamingLLM Displacement Batch Text (runtime, temp)

**Type:** raw UTF-8 bytes (decoded token text, NUL-terminated) **Written by:** `v7s_stream_remove_flush`
(stream_evict.inc) every `EVICT_BATCH` (256) window displacements and at chat turn end **Read by:** the
SectorLogo callback script under `on_stream_evict` via `DatabaseGet`; persist = 0 (temp, the callback
persists what it keeps under its own keys via `DatabaseSet` )

The text of the tokens that left the rolling window, in window displacement order (decoded exactly like
chat output: the SentencePiece word marker becomes a space, `<0xNN>` byte-fallback escapes become raw
bytes). The batch text remains available until the next flush overwrites it.

#### stream_remove_meta - StreamingLLM Displacement Batch Metadata (runtime, temp)

**Type:** ASCII text `first=<pos> tokens=<k>` (NUL-terminated) **Written by:**
`v7s_stream_remove_flush`, together with `stream_remove_text` **Read by:** the `on_stream_evict`
callback script

`first` is the absolute session position of the first token in the batch (batches are
position-contiguous); `tokens` is how many tokens the batch covers. The Logo side dedups/orders by
`first` ranges.

#### on_stream_evict - StreamingLLM Window-Displacement Callback Script (deployment)

**Type:** UTF-8 SectorLogo script source **Added by:** the deployment via `v7_db_add` (or `DatabaseSet`
from a script) BEFORE the streaming session starts **Read by:** `v7s_stream_remove_flush` via
`v7s_sectorlogo_eval_script("on_stream_evict")` after each batch delivery; absent key = -1, a harmless
no-op (the DB keys above are still written)

Typical body: `DatabaseSet "ltm" (DatabaseGet "stream_remove_text")` plus a meta copy, passing the
displaced history into Logo-side long-term memory.

#### __sectorlogo__autoexec - SectorLogo Standard Library

**Type:** UTF-8 text (the autoexec.logo stdlib source; >64KB is a terminal error at the copy site,
clamping is not permitted because it changes the stored script value) **Stored by:** build_db.exe
`-autoexec <path>` (optional flag) **Read by:** the SectorLogo stdlib load (tier 1 of the 3-tier scheme:
DB key first -> `autoexec.logo` file fallback -> informational notice, no FATAL)

The Logo-level standard library (PRINT, CLS, SETXY, SETCOLOR, RESETCOLOR, nearest16, key-sequence
constants, see the SectorLogo manual). One-shot: evaluated once at hook init into the base environment,
so per-token hook calls never re-execute stdlib definitions.

---

### INI Keys (__ini__*)

These keys store dispatch tuning parameters from threading_proxy.ini into the database so the ASM init
path can read them without file I/O.

#### Database key naming convention

The following spellings describe stored configuration keys. A legacy `[default]`
key is not a runtime fallback for an absent `[NxD]` block.

    __ini__<section>__<param>     for [default] and [proxy] (double underscore)
    __ini__<NxD>_<param>          for per-shape sections (single underscore)

#### Illustrative baseline dispatch parameters (not a runtime fallback)

| Parameter | Example baseline | Description |
| --- | --- | --- |
| num_chunks | 64 | Number of work chunks per dispatch |
| min_chunk_cols | 2 | Minimum columns per chunk in the documented baseline |
| batch_small | 2 | Caller batch size (small matrix) |
| batch_medium | 4 | Caller batch size (medium matrix) |
| batch_large | 8 | Caller batch size (large matrix) |
| batch_huge | 16 | Caller batch size (large matrix) |
| batch_med_thresh | 4 | Medium batch threshold |
| batch_large_thresh | 8 | Large batch threshold |
| batch_huge_thresh | 16 | Huge batch threshold |
| caller_batch_small | 1 | Caller-side batch (small) |
| caller_batch_medium | 2 | Caller-side batch (medium) |
| caller_batch_large | 4 | Caller-side batch (large) |
| caller_batch_huge | 8 | Caller-side batch (huge) |
| caller_batch_med_thresh | 4 | Caller medium threshold |
| caller_batch_large_thresh | 8 | Caller large threshold |
| caller_batch_huge_thresh | 16 | Caller huge threshold |
| worker_priority | 15 | Worker thread priority (15 = TIME_CRITICAL; read from the [proxy] section at init) |
| chunk_multiplier | 16 | Chunk size multiplier |
| cores_subtract | 2 | Cores to subtract from total |
| use_physical_cores | 0 | 0=all logical processors (SMT), 1=physical cores only |
| chunk_split_pct | 0 | Chunk split percentage |
| chunk_group_1_count | 16 | First chunk group count |
| chunk_group_2_count | 128 | Second chunk group count |
| quantize_chunk_groups | 4 | Quantize chunk group count |
| taper_shift | 0 | Taper shift for chunk fill (0 disables tapering in the uniform baseline) |
| vit_max_chunks | 32 | ViT linear dispatch max chunks |
| vit_min_chunk_cols | 64 | ViT linear dispatch min chunk columns |

#### Prefetch parameters (41 entries per section)

    __ini__<sec>__pf_dist_<0..40>   prefetch distance (default/proxy)
    __ini__<sec>__pf_type_<0..40>    prefetch type (default/proxy)
    __ini__<NxD>_pf_dist_<0..40>     prefetch distance (per-shape)
    __ini__<NxD>_pf_type_<0..40>     prefetch type (per-shape)

All values are int32 (4 bytes). Types: 0=NTA, 1=T0, 2=T1, 3=T2, 4=PREFETCHW, 5=PREFETCHWT1, 6=CLDEMOTE.
There are 41 slots (slots 41-48 belonged to a fused-Q8 kernel that was removed); the full slot map is
stored in [tuning.md](tuning.md#dispatch-and-tuning).

#### Per-shape supported_resolutions string

    __ini__<sec>__supported_resolutions   (e.g. "1024x1024 1024x2048")

---

### Vocab Keys (per-token lookups)

Vocabulary keys provide lookup information for the tokenizer. Their representation must remain
consistent with the vocabulary and merge data stored in the same database.

#### Forward lookups: token string to token ID

**Stored by:** build_db.exe step 8 **Read by:** encode (encode.inc)

For each token i in 0..vocab_size-1, an entry is stored:
- Key: the token string itself (e.g. "hello", " the")
- Value: int32 token ID

These entries are used by the BPE tokenizer encode() function to look up single tokens and merge pairs
during tokenization.

#### Byte-fallback aliases

**Stored by:** build_db.exe step 10 **Read by:** encode (encode.inc)

For tokens matching <0xHH> pattern (byte-fallback), an additional entry maps the raw byte value to the
token ID; this allows single-byte lookups during BPE.

---

### MicroDB File Format

Version 4 of the `lz4ht.db` file format stores the serialized entries in a sequence of compressed
blocks, each representing up to 128 KB of uncompressed data. The file has the following structure:

    File header (20 bytes):
      uint32  reserved0    (always 0, format signature/version fields are never stored or checked)
      uint32  reserved1    (always 0)
      uint32  entry_count  (number of persistent entries)
      uint64  pool_size    (final KP_USED, load pre-allocation target)

    Repeated blocks until EOF:
      Block header (16 bytes):
        uint64  decompressed_size
        uint64  compressed_size
      Block data:
        uint8[compressed_size]   (LZ4-compressed buffer)

    End marker:
      two zero qwords (0, 0)

    Serialization buffer (decompressed) contains:
      For each entry:
        uint32  key_len (padded to 4-byte alignment)
        uint64  data_len
        uint32  flags (bit 0 = persistent)
        uint8[key_len]  key string
        uint8[data_len] value data

### Runtime DB Operations

| Function | Description |
| --- | --- |
| db_load(filename) | Load and decompress lz4ht.db into hash table |
| db_save(filename) | Serialize and compress hash table to file |
| ht_add(key, data, len, persist) | Add/update entry in hash table |
| ht_find(key) | Look up entry, returns data pointer or NULL |
| db_set_readonly(1) | Lock DB (prevent saves) after init |
| db_set_readonly(1) | Lock DB (prevent saves) after init |

## Assembly data layouts

The entries in this chapter describe the data structures shared by the assembly runtime, its
initialization code, and the model preparation tools. Unlike the named MicroDB records in the preceding
chapter, these structures are defined by exact memory offsets and field widths. A value written at the
wrong offset may still be valid data in isolation, but it will not be interpreted correctly by the code
expecting the documented layout.

The tables are therefore an implementation and binary-compatibility reference, not an additional
configuration interface for ordinary applications. They record the structure definitions, sizes,
offsets, and access constants used in the current source. Any modification to a layout must be reflected
in every producer and consumer of that structure, including the corresponding exporter and
database-building code.

---

### Overview

All ASM data structures are defined in `src/asm/inf_structs.inc`. The structures are built directly in
memory during internal init (`v7s_inference_create`, reached from C through the public `v7_init` +
`v7_initialize_inference` pair) by unmarshalling data from the microdb hash table (`ht_find` calls).
Export.py writes FP32 weights directly to a.bin file with a 256-byte header. The two layouts must match
byte-for-byte for inference to work.

This file specifies the binary layout shared by the Python exporter, the C database builder, and the
assembly initializer. Each one writes or reads the exact same byte layouts described below, and none of
them re-derive or re-order anything at runtime. When a new field is needed, all three sides must be
updated together; that is why every table below names the source file and line range it mirrors.

> **Note on line references:** the `inf_structs.inc` line numbers quoted per section point into the current layout definitions. If `inf_structs.inc` grows or shrinks, trust the `%define` names, not the line numbers, and search for the macro prefix (e.g. `TOK_`, `SAM_`, `HDR_`, `VB_`).

---

### 1. Config Structure (v7_config)

**Source Files:**
- ASM: `src/asm/inf_structs.inc:14-29` (CFG_* macro definitions)
- C (`v7_lib.h`):  64 bytes on stack/disk

**Layout (64 bytes, power-of-2 padded):**

| Offset (bytes) | Size | Field Name | Type | Meaning |
| --- | --- | --- | --- | --- |
| 0 | 4 | reserved0 | uint32_t | Reserved padding, always 0 |
| 4 | 4 | reserved1 | uint32_t | Reserved padding, always 0 |
| 8 | 4 | dim | uint32_t | Embedding dimension |
| 12 | 4 | hidden_dim | uint32_t | Hidden layer dimension |
| 16 | 4 | n_layers | uint32_t | Number of transformer layers |
| 20 | 4 | n_heads | uint32_t | Number of attention heads (query) |
| 24 | 4 | n_kv_heads | uint32_t | Number of KV cache heads (<= n_heads for GQA; equal means full MHA) |
| 28 | 4 | vocab_size | uint32_t | Vocabulary size |
| 32 | 4 | seq_len | uint32_t | Max sequence length (KV cache sizing; export default 8192) |
| 36 | 4 | head_dim | uint32_t | Dimension per attention head (usually dim / n_heads; some models set it explicitly, e.g. 128) |
| 40 | 4 | shared_classifier | uint32_t | 1 if the classifier (wcls) shares the token embedding table, 0 otherwise |
| 44 | 4 | group_size | uint32_t | Quantization group size (64) |
| 48 | 4 | has_qk_norm | uint32_t | 1 if model uses QK-RMSNorm (Qwen3), 0 otherwise |
| 52 | 4 | rope_base | uint32_t | RoPE frequency base (1000000 Qwen3, 500000 Llama-3.2, 100000 SmolLM2, 10000 Phi-3/TinyLlama) |
| 56 | 4 | reserved2 | uint32_t | Must be 0 |
| 60 | 4 | reserved3 | uint32_t | Must be 0 |

**ASM Access Macros:** `CFG_DIM`, `CFG_HIDDEN_DIM`, `CFG_N_HEADS`, `CFG_N_KV_HEADS`, `CFG_HEAD_DIM`,
`CFG_SHARED_CLASSIFIER`, `CFG_GROUP_SIZE`, `CFG_HAS_QK_NORM`, `CFG_ROPE_BASE`

**Expected Payload:** Little-endian 4-byte uint32_t values.

---

### 2. Transformer Struct

**Source:** `src/asm/inf_structs.inc:31-33`

**Layout (after Config):**

| Offset (bytes from struct start) | Size | Field Name | Type | Meaning |
| --- | --- | --- | --- | --- |
| 64 | 8 | transformer | void* | Pointer to TransformerWeights struct (points inside microdb heap) |
| 184 | 8 | run_state | void* | Pointer to RunState struct (buffer for forward pass activations) |

**Purpose:** The `InferenceContext` structure occupies part of a single block allocated with
`VirtualAlloc`. Its two pointers must both be non-null before inference can run.

---

### 3. TransformerWeights Struct

**Source Files:**
- ASM: `src/asm/inf_structs.inc:35-50` (W_* macro definitions)
- C (`v7_lib.h`) (same layout, all 8-byte pointers)

**Layout (120 bytes, 15 fields  x  8 bytes):**

| Offset (bytes) | Size | Field Name | Type | Meaning |
| --- | --- | --- | --- | --- |
| 0 | 8 | q_tokens | void* | Token quantization table (dequantized to FP32 on init) |
| 8 | 8 | token_embedding | void* | Token embedding table (FP32, dequantized from quantized table) |
| 16 | 8 | rms_att_weight | void* | RMS norm weight for attention hidden states |
| 24 | 8 | rms_ffn_weight | void* | RMS norm weight for FFN hidden states |
| 32 | 8 | wq | void* | Query weight matrix (FP32 or Q8) |
| 40 | 8 | wk | void* | Key weight matrix (FP32 or Q8) |
| 48 | 8 | wv | void* | Value weight matrix (FP32 or Q8) |
| 56 | 8 | wo | void* | Output weight matrix (FP32 or Q8) |
| 64 | 8 | q_norm_weights | void* | QK norm weights (if has_qk_norm) |
| 72 | 8 | k_norm_weights | void* | K norm weights (if has_qk_norm) |
| 80 | 8 | w1 | void* | FFN gate projection weight (FP32 or Q8) |
| 88 | 8 | w2 | void* | FFN down projection weight (FP32 or Q8) |
| 96 | 8 | w3 | void* | FFN up projection weight (FP32 or Q8) |
| 104 | 8 | rms_final_weight | void* | RMS norm weight for final hidden states |
| 112 | 8 | wcls | void* | Final classifier weight (only if not shared) |

**Usage Notes:**

- **q_tokens**: Q8 quantized token embedding (int8 data + float scales). Built in `inf_weights.inc` Step 9b: reads `weight_meta[5]` file_offset from `__sys_model_weights` blob, creates QuantizedTensor descriptor (q_ptr, s_ptr) pointing into the blob.
- **token_embedding**: FP32 token embedding table. Built in `inf_state.inc` Step 12: a separate VirtualAlloc'd buffer, dequantized from q_tokens via AVX2 (`vpmovsxbd` + `vcvtdq2ps` + `vmulps`). This is a DIFFERENT pointer from q_tokens, q_tokens stays as Q8, token_embedding is the FP32 copy used by forward/attention kernels.
- **wq/wk/wv/wo/w1/w2/w3**: Matrix weights. Stored in microdb as `__sys_model_weights` blob per layer. File offsets stored in `weight_meta[6..12]` array.

**Inference modes:**
- **F32 mode:** The weight pointers refer directly to the model's FP32 data, and the transformations use `v7d_f32`.
- **Q8 mode:** The weight pointers refer to quantized values and their scale arrays, and the transformations use `v7d_q8_fused` and `v7d_swiglu_q8`.

---

### 4. RunState Struct

**Source Files:**
- ASM: `src/asm/inf_structs.inc:52-70` (S_* macro definitions)
- C (`v7_lib.h`): same layout

**Layout (144 bytes, 18 fields x 8 bytes, ASM; the C struct in `v7_lib.h` adds four unused trailing
handles):**

| Offset (bytes) | Size | Field Name | Type | Meaning |
| --- | --- | --- | --- | --- |
| 0 | 8 | x | void* | Current input activation buffer (FP32) |
| 8 | 8 | xb | void* | Temporary buffer for RMSNorm+RoPE (FP32) |
| 16 | 8 | hb | void* | FFN gate projection output buffer (hidden_dim, FP32) |
| 24 | 8 | hb2 | void* | FFN up projection output buffer (hidden_dim, FP32) |
| 32 | 8 | xq.q | void* | Quantized copy of the x activation buffer (int8) |
| 40 | 8 | xq.s | void* | Quantization scales for xq (float) |
| 48 | 8 | hq.q | void* | Quantized copy of the hb activation buffer (int8) |
| 56 | 8 | hq.s | void* | Quantization scales for hq (float) |
| 64 | 8 | q | void* | Attention query buffer (FP32) |
| 72 | 8 | k | void* | Attention key buffer (FP32) |
| 80 | 8 | v | void* | Attention value buffer (FP32) |
| 88 | 8 | att | void* | Attention output buffer (FP32) |
| 96 | 8 | logits | void* | Final logit buffer (FP32, after attention) |
| 104 | 8 | key_cache | void* | KV cache keys: separate FP16 buffer (n_layers x seq_len x n_kv_heads x head_dim x 2 bytes) |
| 112 | 8 | value_cache | void* | KV cache values: separate FP16 buffer, same dimensions as key_cache |
| 120 | 8 | x_storage | void* | Storage for CopyEmbedding (FP32) |
| 128 | 8 | q_storage | void* | Prefill query-storage region (FP32, seq_len x all_heads_dim) |
| 136 | 8 | att_out | void* | Reusable buffer for attention output (FP32) |
| 144 | 8 | key_cache_hfile | void* | Windows HANDLE for KV cache backing file (C-only, file-backed KV cache) |
| 152 | 8 | key_cache_hmap | void* | Windows HANDLE for KV cache file mapping object (C-only) |
| 160 | 8 | value_cache_hfile | void* | Windows HANDLE for value cache backing file (C-only) |
| 168 | 8 | value_cache_hmap | void* | Windows HANDLE for value cache file mapping object (C-only) |

**Notes on C vs ASM RunState:**
- **ASM** (`inf_structs.inc`): 18 fields (offsets 0-136), `S_X` through `S_ATT_OUT`. These are the only fields the ASM inference code uses.
- **C** (`v7_lib.h`): 22 fields (offsets 0-168), same 18 + 4 file-backed KV cache handles (`key_cache_hfile`, `key_cache_hmap`, `value_cache_hfile`, `value_cache_hmap`). These are unused legacy C structure members; the ASM runtime does not store its KV cache handles in these fields. The allocator may separately fall back to a file-backed mapping.

**Current runtime allocation:** `tools/db_build_tool.c:compute_mem_layout()` still serializes a
160-byte MemLayout containing its former FP32 KV sizes and offsets. These serialized offsets remain
part of the database layout, but **the ASM runtime does not allocate the whole `rs_block_bytes`**.
`src/asm/inf_state.inc` allocates a working block of `ML_OFF_KEY_CACHE` bytes, covering:
- Activation buffers (`x`, `xb`, `hb`, `hb2`, `xq`, `hq`), sized for the configured model
- `q`, `(all_heads_dim + 2 * kv_dim)` FP32 values (Q, with legacy layout capacity for K/V)
- `att`, `n_heads * seq_len` FP32 values; `logits`, `vocab_size` FP32 values
- Prefill storages, `seq_len * dim` and two `seq_len * all_heads_dim` FP32 regions

It then allocates the **key** and **value** caches separately, with
`n_layers * seq_len * kv_dim * 2` bytes each, plus a separate `2 * kv_dim * 4`-byte FP32
K/V projection staging buffer. All allocations pass through `mdb_alloc_mem`, which may fall back to
file-backed memory; the live KV allocation is not the old FP32 suffix described by MemLayout.

**Usage Notes:**

- **x**: Input embedding for current token.
- **xq / hq**: Quantized activation buffers (int8 + scales) for Q8 matmuls.
- **key_cache/value_cache**: Separate layer-major FP16 buffers. Entry `(layer, slot)` begins at byte offset `(layer * seq_len + slot) * kv_dim * 2` from the buffer base; `kv_dim = n_kv_heads * head_dim`. In streaming mode, `slot` comes from the ring-slot mapping, not necessarily the absolute position.
- **KV cache allocation:** two independent `mdb_alloc_mem` allocations at inference creation, plus a separate FP32 K/V staging buffer; no per-token cache growth.

---

### 5. MemLayout Struct

**Source Files:**
- ASM: `src/asm/inf_structs.inc:75-94` (ML_* macro definitions)
- C: `tools/db_build_tool.c`, `compute_mem_layout()` fills the struct; `build_db.exe` stores it as `__sys_mem_layout`

**Layout (160 bytes = 20 x 8-byte offsets/sizes):**

| Offset (bytes) | Size | Field Name | Type | Meaning |
| --- | --- | --- | --- | --- |
| 0 | 8 | reserved0 | size_t | Reserved padding, always 0 |
| 8 | 8 | rs_block_bytes | size_t | Legacy total RunState block size written by builder; **not** the amount allocated by current ASM |
| 16 | 8 | off_x | size_t | Byte offset of `x` buffer inside rs_block |
| 24 | 8 | off_xb | size_t | Byte offset of `xb` buffer inside rs_block |
| 32 | 8 | off_hb | size_t | Byte offset of `hb` buffer inside rs_block |
| 40 | 8 | off_hb2 | size_t | Byte offset of `hb2` buffer inside rs_block |
| 48 | 8 | off_xq_q | size_t | Byte offset of `xq.q` buffer inside rs_block |
| 56 | 8 | off_xq_s | size_t | Byte offset of `xq.s` buffer inside rs_block |
| 64 | 8 | off_hq_q | size_t | Byte offset of `hq.q` buffer inside rs_block |
| 72 | 8 | off_hq_s | size_t | Byte offset of `hq.s` buffer inside rs_block |
| 80 | 8 | off_q | size_t | Byte offset of `q` buffer inside rs_block |
| 88 | 8 | off_att | size_t | Byte offset of `att` buffer inside rs_block |
| 96 | 8 | off_logits | size_t | Byte offset of `logits` buffer inside rs_block |
| 104 | 8 | off_key_cache | size_t | Byte offset of `key_cache` pointer inside rs_block |
| 112 | 8 | off_value_cache | size_t | Byte offset of `value_cache` pointer inside rs_block (same as key_cache, reused alias) |
| 120 | 8 | off_x_storage | size_t | Byte offset of `x_storage` buffer inside rs_block |
| 128 | 8 | off_q_storage | size_t | Byte offset of `q_storage` buffer inside rs_block |
| 136 | 8 | off_att_out | size_t | Byte offset of `att_out` buffer inside rs_block |
| 144 | 8 | emb_table_bytes | size_t | Size of token_embedding table in bytes (vocab_size x dim x 4) |
| 152 | 8 | vocab_ptrs_bytes | size_t | Size of vocab_ptr array (vocab_size x 8) |

**Purpose:** Defines the layout of a single VirtualAlloc'd block (InferenceContext + Transformer +
RunState + Tokenizer + KV cache). The blob ends at offset 160; the config is cached from the Config
struct instead of being copied here.

---

### 6. weight_meta Entry

**Source Files:**
- ASM: `src/asm/inf_structs.inc:99-104` (WM_* macro definitions)
- C: `tool_src/db_build_tool.c`: `struct weight_meta { size_t file_offset; uint32_t rows; uint32_t cols; uint8_t is_quantized; uint8_t reserved[3]; };`

**Layout (24 bytes, padded to size_t alignment):**

| Offset (bytes) | Size | Field Name | Type | Meaning |
| --- | --- | --- | --- | --- |
| 0 | 8 | file_offset | size_t | File position (bytes from start of the weights blob) for this weight tensor |
| 8 | 4 | rows | uint32_t | Output row count (d dimension for matmul) |
| 12 | 4 | cols | uint32_t | Auxiliary column/flag field (0 for most tensors) |
| 16 | 1 | is_quantized | uint8_t | 1 if Q8, 0 if FP32 |
| 17 | 3 | reserved | uint8_t[3] | Padding (must be zero) |
| 20 | 4 | ENTRY_SIZE | fixed | Padded to 24 bytes (= 8+4+4+1+3) |

The `weight_meta` array contains 16 entries occupying 24 bytes each, for a total of 384 bytes. The
`WM_*` macros refer to entries by index, from `weight_meta[0]` through `weight_meta[15]`.

---

### 7. weight_meta Indices

**Source:** `src/asm/inf_structs.inc:109-124`

| Index | Name | Meaning (accesses `weight_meta[x]`) | Expected Content |
| --- | --- | --- | --- |
| 0 | WM_RMS_ATT | `rms_att_weight_mode` | FP32 RMS norm weights (n_layers  x  hidden_dim) |
| 1 | WM_RMS_FFN | `rms_ffn_weight_mode` | FP32 RMS norm weights (n_layers  x  hidden_dim) |
| 2 | WM_RMS_FINAL | `rms_final_weight_mode` | FP32 RMS norm weights (dim) |
| 3 | WM_Q_NORM | `q_norm_weights_mode` | FP32 if has_qk_norm (n_layers  x  head_dim) |
| 4 | WM_K_NORM | `k_norm_weights_mode` | FP32 if has_qk_norm (n_layers  x  head_dim) |
| 5 | WM_Q_TOKENS | `q_tokens_mode`, unused | Quantized token embedding, dequantized on init |
| 6 | WM_WQ | `wq_mode` | wq weights (FP32 or Q8) |
| 7 | WM_WK | `wk_mode` | wk weights (FP32 or Q8) |
| 8 | WM_WV | `wv_mode` | wv weights (FP32 or Q8) |
| 9 | WM_WO | `wo_mode` | wo weights (FP32 or Q8) |
| 10 | WM_W1 | `w1_mode` | FFN w1 weights (FP32 or Q8) |
| 11 | WM_W2 | `w2_mode` | FFN w2 weights (FP32 or Q8) |
| 12 | WM_W3 | `w3_mode` | FFN w3 weights (FP32 or Q8) |
| 13 | WM_Q_TOKENS_DUP | `q_tokens_dup_mode` (unused) | Duplicate of token embedding |
| 14 | WM_WCLS | `wcls_mode` | wcls classifier weights (FP32 or Q8) |
| 15 | WM_EMB_TABLE | `token_embedding`, dequantized and absent from weight_meta | Special: built from q_tokens, no file |

**Loading procedure:** The `v7s_inference_create` function iterates over the weight metadata entries
`weight_meta[6..14]` for each layer; it processes each entry as follows:
1. `ht_find("__sys_model_weights")` to get blob base pointer
2. Compute `weight_ptr = model_weights_base + file_offset`
3. If `is_quantized == 0`: FP32 weight (pointer stored directly)
4. If `is_quantized == 1`: Q8+scale arrays (pointer + scales built as QuantizedTensor structs)

---

### 8. QuantizedTensor Struct

**Source:** `src/asm/inf_structs.inc:129-131`

**Layout (16 bytes, 2 x  qword = 16 bytes):**

| Offset (bytes) | Size | Field Name | Type | Meaning |
| --- | --- | --- | --- | --- |
| 0 | 8 | q | void* | Pointer to int8 quantized data |
| 8 | 8 | s | void* | Pointer to float scales array |
| 16 | 0 | QT_SIZE | fixed | Size of struct (bytes) |

**Purpose:** This structure represents a Q8_0 quantized tensor. The dispatched `v7d_q8` entry's chunk
kernel loads int8 values, applies their corresponding floating-point scales, and produces FP32 output.
This structure is used for wq/wk/wv/wo/w1/w2/w3.

**ASM Access Macros:** `QT_Q`, `QT_S`

---

### 9. InferenceContext Struct

**Source Files:**
- ASM: `src/asm/inf_structs.inc:142-153` (CTX_* macro definitions)
- C: `v7_lib.h`: matches `typedef struct inference_context { void *transformer; void *tokenizer; void *sampler; void *prompt_tokens; void *out_tokens;... } inference_context;`

**Layout (56 bytes, 7 x  qword = 56 bytes):**

| Offset (bytes) | Size | Field Name | Type | Meaning |
| --- | --- | --- | --- | --- |
| 0 | 8 | transformer | void* | Points to Transformer struct inside VirtualAlloc block |
| 8 | 8 | tokenizer | void* | Points to Tokenizer struct |
| 16 | 8 | sampler | void* | Points to Sampler struct |
| 24 | 8 | prompt_tokens | void* | Padded token buffer (allocated in ctx block) |
| 32 | 8 | out_tokens | void* | Output token buffer (allocated in ctx block) |
| 40 | 4 | max_prompt | dword | Max number of prompt tokens (allocated buffer size) |
| 44 | 4 | max_gen | dword | Max number of generated tokens (allocated buffer size) |
| 48 | 4 | pos | dword | Current sequence position (for KV cache rotation) |
| 52 | 4 | quantize | dword | 1 = Q8, 0 = F32 path |
| 56 | 0 | CTX_HEADER_SIZE | fixed | Total size of header (bytes) |

**Purpose:** The `InferenceContext` header resides at the beginning of a single `VirtualAlloc` block.
Its pointers refer to locations within that same block, so the structure does not require allocations in
other modules.

---

### 10. Tokenizer Struct

**Source Files:**
- ASM: `src/asm/inf_structs.inc:156-165` (TOK_* macro definitions)
- C (`v7_lib.h`): matches `typedef struct Tokenizer { const char **vocab; float *merge_scores; int vocab_size;... };`

**Layout (2088 bytes, heavily packed):**

| Offset (bytes) | Size | Field Name | Type | Meaning |
| --- | --- | --- | --- | --- |
| 0 | 8 | vocab | void* | Pointer to char** vocab array (tokens  x  pointer size) |
| 8 | 8 | merge_scores | void* | Pointer to merge scores (float[vocab_size]) |
| 16 | 4 | vocab_size | dword | Number of tokens |
| 20 | 4 | max_token_len | dword | Length of longest token |
| 24 | 4 | bos_token_id | dword | BOS token ID (typically 0 for sentencepiece) |
| 28 | 4 | eos_token_id | dword | EOS token ID (typically 2 for sentencepiece) |
| 32 | 1024 | prompt_template | char[1024] | Prompt template string (chat mode) |
| 1056 | 1024 | sys_template | char[1024] | System prompt template string (chat mode) |
| 2080 | 4 | is_sentencepiece | dword | 1 if using sentencepiece tokenization, 0 if BPE |
| 2084 | 4 | pad | dword | Padding so the following Sampler sub-block is 8-byte aligned |
| 2088 | 0 | TOK_SIZE | fixed | Size of struct (bytes) |

**Token lookup:** All tokenizer data (vocab pointer array, merge scores, BOS/EOS ids) is loaded from
microdb ONCE at init; the `encode` export then works entirely from the in-memory Tokenizer struct, no DB
access happens during tokenization.

#### 10a. vocab_bulk Layout

**Source:** `src/asm/inf_structs.inc:189-192` (VB_* macro definitions)

**Format:** `[count (4 bytes)][for each token: length (4 bytes) + string bytes] x vocab_size`

| Offset (bytes) | Size | Field Name | Type | Meaning |
| --- | --- | --- | --- | --- |
| 0 | 4 | VB_COUNT | dword | Total token count at start of vocab bulk data |
| 4 onward | variable | array | variable | For each token: `VB_ENTRY_SLEN` (4-byte length) + string bytes |

**ASM Build:** In `inf_weights.inc` Step 9, loops through token IDs 0..vocab_size-1 into q8 payload,
computes size, allocates contiguous buffer, copies all tokens one-by-one. No null-termination needed
(byte layout is what decoder expects).

---

### 11. Sampler Struct

**Source Files:**
- ASM: `src/asm/inf_structs.inc:167-176` (SAM_* macro definitions)
- C (`v7_lib.h`): matches `typedef struct Sampler { int vocab_size; ProbIndex *probindex; float temperature; float topp;... };`

**Layout (40 bytes):**

| Offset (bytes) | Size | Field Name | Type | Meaning |
| --- | --- | --- | --- | --- |
| 0 | 4 | vocab_size | dword | Vocabulary size (matches Tokenizer.vocab_size) |
| 4 | 4 | reserved | dword | Padding (must be 0) |
| 8 | 8 | probindex | void* | Fast symbol storage (ProbIndex* from C probindex build) |
| 16 | 4 | temperature | float | Sampling temperature [0, inf) |
| 20 | 4 | topp | float | Top-p sampling threshold [0,1] |
| 24 | 4 | presence_penalty | float | Presence penalty (avoids repeated tokens) |
| 28 | 4 | topk | dword | Top-k limit (max tokens to consider) |
| 32 | 8 | rng_state | qword | Random number generator seed for multinomial sampling |
| 40 | 0 | SAM_SIZE | fixed | Size of struct (bytes) |

**Probability Index:** `probindex` is a plain `v7_probindex` array of `vocab_size` entries (`{float
prob; int index;}` per entry, see `v7_lib.h` ), pre-allocated once at init. The sampler sorts
probabilities into it during top-p filtering; no additional sizing constants are involved.

---

### 12. Tokenizer Header (from microdb)

**Source:** `src/asm/inf_structs.inc:181-184` (HDR_* macro definitions)

**Layout (12 bytes from __sys_tokenizer_hdr):**

| Offset (bytes) | Size | Field Name | Type | Meaning |
| --- | --- | --- | --- | --- |
| 0 | 4 | HDR_MAX_TOKEN_LEN | dword | max_token_length (32-bit) |
| 4 | 4 | HDR_BOS_ID | dword | bos_token_id (32-bit) |
| 8 | 4 | HDR_EOS_ID | dword | eos_token_id (32-bit) |
| 12 | 0 | HDR_SIZE | fixed | Total size (bytes) |

**Purpose:** During tokenizer initialization, the engine reads `__sys_tokenizer_hdr` through `ht_find`
to obtain the BOS and EOS token IDs.

---

### 13. MicroDB String Keys (inf_data.inc)

**Source:** `src/asm/inf_data.inc:10-30`

| Key String | Size | Example Value | Field(s) Populated |
| --- | --- | --- | --- |
| `__sys_config` | 64 | Config header (16 32-bit fields) | Config (reserved0, reserved1, dim, hidden_dim, etc.) |
| `__sys_model_weights` | variable | Entire model weights blob | Token embedding (q_tokens), wq/wk/wv/wo/w1/w2/w3 (FP32 or Q8), classifier (wcls) |
| `__sys_weight_meta` | 384 | weight_meta array (16 x  24 bytes) | All transformer weights (FP32 and Q8) |
| `__sys_mem_layout` | 160 | MemLayout struct (20 x 64-bit fields) | Legacy allocation metadata; live working block ends at `off_key_cache` before separate FP16 caches |
| `__sys_tokenizer_hdr` | 12 | hdr (max_token_len, bos_token_id, eos_token_id) | Tokenizer header |
| `__sys_vocab_bulk` | variable | Token strings packed without NULs | Tokenizer vocab bulk (correctly dequantized) |
| `__sys_merge_scores` | variable | float[vocab_size], merge scores for BPE encoding | Tokenizer merge scores for BPE encoding |
| `__sys_template` | var (max 1023) | Turn-1 prompt template (base/non-thinking variant) | tok->prompt_template + g_tpl_t1_base |
| `__sys_template_sys` | var (max 1023) | System+user template (base variant) | tok->system_template + g_tpl_sys_base |
| `__sys_template@img` | var (max 1023) | Image-turn template (optional) | g_img_template + g_img_template_valid |
| `__sys_template@think` / `__sys_template_sys@think` | var (max 1023) | Thinking variants (optional) | g_think_template / g_think_sys_template |
| `__sys_template@multi` / `__sys_template@multi@think` | var (max 1023) | Inter-turn (turn 2+) templates (optional) | g_multi_template / g_multi_think_template |
| `__sys_stop_token_ids` | var | EOS + turn-terminator token IDs | g_stop_ids / g_stop_id_count |
| `__sys_rms_eps` | 4 | Per-model RMS norm epsilon | g_rms_eps |
| `__sys_vision_config` | 744 | Expanded VCFG (93 qwords, VLMs only) | g_vit_cfg block |

**ASM Lookup Pattern:** `db_load` runs once (at `V7_init` / DLL attach); each subsequent lookup is a
plain `ht_find(key)` returning a `void*` directly into the LZ4-decompressed hash-table pool, no
per-entry file I/O, no copies.

---

### Verifying compatibility

Database compatibility is determined by the stored model fields, memory layout, weight ordering, and
alignment requirements. The checks below describe the values that should be compared when investigating
a failed load or an unexpected inference result.

#### 14.1 Header Layout

The 256-byte header in `tools/export.py` (model_export function) contains:
1. Config struct (16  x  uint32_t = 64 bytes): reserved0, reserved1, dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, head_dim, shared_classifier, group_size, has_qk_norm, rope_base, reserved2, reserved3
2. Pad the structure to 256 bytes by adding 192 zero bytes.
3. The format contains no format signature bytes or version fields (see section 96).

#### 14.2 Weight Order Validation

**Mapping Weight Files -> ASM Code:**

| File Section (export.py) | ASM Data Structure | weight_meta[index] | Quick Check |
| --- | --- | --- | --- |
| Header -> Config | Config | not indexed in `weight_meta` | 16 32-bit fields + padding |
| `norms/clip/quant/rope` | Explicit writes | WM_RMS_ATT/W_FFN | 5 tensors x hidden_dim x n_layers |
| `q_token.bin` | q_tokens (in weights blob) | WM_Q_TOKENS | Read at offset (dequant copy built on init) |
| `tok_emb.bin` | token_embedding (FP32) | WM_EMB_TABLE | Dequantized from q_tokens |
| `wq_*.bin` | wq (Q8 or FP32) | WM_WQ | n_layers blocks |
| `wk_*.bin` | wk (Q8 or FP32) | WM_WK | n_layers blocks |
| `wv_*.bin` | wv (Q8 or FP32) | WM_WV | n_layers blocks |
| `wo_*.bin` | wo (Q8 or FP32) | WM_WO | n_layers blocks |
| `w1_*.bin` | w1 gate (Q8 or FP32) | WM_W1 | n_layers blocks |
| `w2_*.bin` | w2 down (Q8 or FP32) | WM_W2 | n_layers blocks |
| `w3_*.bin` | w3 up (Q8 or FP32) | WM_W3 | n_layers blocks |
| `wcls_*.bin` (if exists) | wcls (Q8 or FP32) | WM_WCLS | Optional, vocab_size x dim (single tensor) |

#### 14.3 Alignment Constraints

**Key Constraints:**
1. All pointers in TransformerWeights/RunState/Tokenizer/Sampler must be 8-byte aligned (the C header checks `allocated % 8 == 0`).
2. int8 data arrays (Q8 weights, q_tokens) must be 4-byte aligned for AVX2 load/store (`vpmovsxbd`/`vpmovsrbw`).
3. Float scale arrays must be 4-byte aligned.
4. KV cache entry stride: `kv_dim = n_kv_heads * head_dim` **FP16 values** per `(layer, slot)` row. Byte offset is `(layer * seq_len + slot) * kv_dim * 2` in each independently allocated key/value buffer; streaming mode maps absolute positions to cache slots.

---

### Inspecting and debugging binary data

The following techniques are intended for inspecting serialized values and native structure offsets;
they supplement the supplied database inspection and validation tools.

#### 15.1 Hex Dump Verification

Compare DB-init vs file-init state:
```bash
mdb_query_key.exe bin\v7_threading.dll __sys_config --db weights\databases\qwen3_0.6b\lz4ht.db
mdb_query_key.exe bin\v7_threading.dll __sys_mem_layout --db weights\databases\qwen3_0.6b\lz4ht.db
```

#### 15.2 Offsets Check

Inspect ASM-local `g_inf_*` globals after init:
```asm
mov rax, inf_transformer
movaps xmm0, [rax + W_Q_TOKENS]; First 16 bytes of q_tokens
; should match q_tokens dequantized FP32 data
```

#### 15.3 Trace Hooks

Enable `TRACE_DEBUG=1` build, run `inference_example.exe` and grep for:
- `[FATAL] ht_add`, DB write failed
- `null ptr`, Missing weight pointer
- `stride mismatch`, Wrong quantization stride
- `vocab size 0`, Tokenizer loading bug

---

**End of asm_data_structures.md**
