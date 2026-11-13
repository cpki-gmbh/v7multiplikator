# Using V7 from C and HTTP

V7 exposes a native C interface through the engine library (`v7_threading.dll` on Windows,
`libv7_threading.so` on Linux — both use the Windows x64 register convention, so host-side
declarations carry the `ms_abi` attribute on Linux). Applications can initialize a model,
generate text, process image inputs with supported vision-language models, configure sampling, and
manage the current session without operating a separate service. The API also exposes database functions
and a set of lower-level functions used by diagnostic and performance tools.

For application integration, start with the public `v7_*` functions. They provide the normal lifecycle
and keep the inference context inside the library, so a C application does not need to manage an opaque
context handle. The lower-level exports have different preconditions and are documented for applications
that need to inspect or extend the implementation. The final chapter covers the optional HTTP server,
whose execution model follows the same single-context constraints as the native interface.

## Contents

- [C API reference](#c-api-reference)
- [Typed decision interfaces](#typed-decision-interfaces)
- [HTTP integration](#http-integration)

---

## C API reference

The public `v7_*` functions are the normal way to integrate V7 into a C or C++ application. They provide
the complete model lifecycle, including database loading, inference initialization, text and image
generation, session configuration, error reporting, and teardown. The library performs the transformer
computation internally, so an application does not need to invoke individual matrix multiplication,
attention, normalization, tokenization, or sampling routines to generate a response.

Some additional exports exist for diagnostic tools, tuning, and direct execution of native operations;
these include the `v7d_*` and `v7s_*` groups. They can be useful when developing or validating the
runtime, but they may require preconditions that the ordinary public API establishes automatically.
The chapter documents them for completeness while treating the public lifecycle as the starting point
for normal integration.

### Overview

This reference describes the host-visible functions and their return contracts. Features such as
`SetStream` and `Reject` are SectorLogo host commands, not replacements for the C lifecycle API. The
native FFI also permits a script stored in MicroDB to invoke an application function, so a separate
dedicated generation-callback registration API is not required for that use case.

A host application first loads the DLL, opens a compatible MicroDB database, and initializes inference.
It can then call `v7_chat` for conversational use or `v7_generate` for an independent generation
request. Once the required operations are complete, it releases the inference state and unloads the
library. The public interface uses a single inference context maintained in static storage inside the
DLL, so callers neither receive nor pass a context handle. The initialization, error, and teardown
functions described below establish the precise lifecycle.

Some low-level exports are intentionally outside the set resolved by the `v7_lib.h` loader. Applications
that require them can obtain the exported addresses with `GetProcAddress`, but should follow the
preconditions documented for each function. The complete list of available symbols appears in the export
reference at the end of this chapter. The public functions establish a single active inference context,
render the appropriate templates, manage sampling, and write text into buffers supplied by the host;
the low-level `v7d_*` and `v7s_*` functions have different preconditions and are not replacements for
the public lifecycle.

### Library Initialization and Teardown

Initialization has a defined order: load the DLL, open the database, create the inference context, and
only then invoke generation. The example below illustrates that order with the functions provided by
`v7_lib.h`. It is intentionally compact; production code must check each return value and consult the
public error-reporting functions if initialization or generation fails. The teardown order is equally
important because the library owns the current inference state.

```c
#include "v7_lib.h"

v7_lib dll;
v7_load("v7_threading.dll", &dll);           // Step 1: Load DLL, resolve exports

dll.v7_init("lz4ht.db");                     // Step 2: Load database (must precede create)

dll.v7_initialize_inference();               // Step 3: Create the inference context (hidden in BSS; returns 1 = ready)

char output[32768] = {0};
dll.v7_chat("prompt", 0, output, sizeof(output), 1);  // Step 4: Generate (reset=1 for new session)

dll.v7_free_inference();                     // Step 5: Cleanup (parameterless)
v7_unload(&dll);                             // Step 6: Unload DLL
```

#### `v7_load(dll_path, lib)` returns int

The `v7_load` function loads the DLL with `LoadLibraryA` and resolves its exports with
`GetProcAddress`; it returns `0` on success and `-1` on failure. If a required export is missing,
initialization calls `TerminateProcess`; all required exports must be present.

#### `v7_init(db_path)` returns int

The `v7_init` function opens the MicroDB database file and must be called before
`v7_initialize_inference`. It returns `1` on success. A missing database file is a terminal error
reported through the public error contract: the call returns 0 with `v7_engine_is_dead() == 1`. Passing
NULL opens an empty database (no entries loaded), which is a valid success.

#### `v7_initialize_inference()` returns int

Creates the inference context from the database and stores it in a BSS singleton inside the DLL. The
pointer is not exposed to the host application. The operation is idempotent: it returns `1` if the
inference context is ready, whether initialization occurred during the current call or a previous call.
All create failures FATAL internally; under the public API error-reporting contract described in [Public
API error reporting](#public-api-error-reporting), those FATALs unwind to the caller with
`v7_engine_is_dead() == 1`.

#### `v7_chat(user_prompt, system_prompt, output, output_len, reset)`

`v7_chat` is the normal entry point for conversational generation. It renders the appropriate stored
chat template, tokenizes the prompt, evaluates the necessary input tokens, and writes the decoded
response into the caller's output buffer. The library retains the current conversation state between
calls, allowing an application to continue a session without reconstructing the full prompt itself.

The final `reset` argument determines whether that state is reused. Passing `reset = 1` begins a new
conversation and resets the previous KV-cache position; passing `reset = 0` continues the existing
session from its retained position. Applications that need unformatted text continuation can use
`v7_generate(prompt, output, output_len, raw_generation=1)` instead of rendering a chat template.

##### Chat-template selection

`v7_chat` always renders the input through a stored chat template before tokenization. It does not
provide a separate raw-completion mode; applications that need an unformatted continuation should use
`v7_generate(prompt, output, output_len, raw_generation=1)`. Template selection depends on the presence
of a system prompt, whether the context is at the beginning of a conversation, and whether the model
database contains an appropriate template for later turns.

When `system_prompt` is non-NULL, V7 renders `g_tpl_sys` using the system and user strings. This
template is resolved from `__sys_template_sys` or its `@think` variant. On the first turn without a
system prompt (`ctx->pos == 0`), it uses `g_tpl_t1`, resolved from `__sys_template` or
`__sys_template@think`. Later turns use `g_tpl_multi` from `__sys_template@multi` or its thinking
variant. If the database has no multi-turn template, the engine falls back to the first-turn template.
The table summarizes these choices.

| Condition | Template | Arguments |
| --- | --- | --- |
| `system_prompt != NULL` | `g_tpl_sys` from `__sys_template_sys` or `__sys_template_sys@think` | System text and user text |
| No system prompt; first turn (`ctx->pos == 0`) | `g_tpl_t1` from `__sys_template` or `__sys_template@think` | User text |
| No system prompt; later turn (`ctx->pos > 0`) | `g_tpl_multi` from `__sys_template@multi` or `__sys_template@multi@think`, with fallback to `g_tpl_t1` | User text |

Image input takes precedence over this selection when `__sys_template@img` is available. In an image
turn, `v7_chat_image` or `v7_generate_image` sets the image-turn flag, and V7 renders `g_img_template`
with the user prompt. The flag is cleared after rendering, so later text-only turns return to normal
template selection.

Thinking templates are selected through `v7_enable_thinking(1)`, which is the default at each
`v7_initialize_inference`. Calling `v7_enable_thinking(0)` selects the base template instead. If the
prepared model contains an opening reasoning marker in its thinking template, the model begins by
generating content inside that section; it does not need to emit the opening marker itself. When no
thinking variant exists, the setting does not change the selected template.

A few details follow from these rules. `v7_generate` never renders the system template, and it uses the
first-turn template when `raw_generation=0`. `v7_generate_image` has no system-prompt argument and does
not perform a system-and-user render. A model without a multi-turn template always falls back to the
first-turn variant on subsequent turns. These behaviors matter when preparing custom templates or
interpreting the text that was actually sent to the decoder.

##### Raw completion via template override (`%s`)

The chat interface normally adds role markers and other formatting through the template stored in
MicroDB. If an application needs an unformatted continuation while using that interface, it can supply a
template containing only the `%s` placeholder. Because V7 reads `__sys_template` when
`v7_initialize_inference` creates the inference context, the application must replace the in-memory
database value before that initialization call:

```c
dll.v7_db_add("__sys_template", "%s", 3, 0);     // persist=0: memory only, DB file untouched
dll.v7_enable_thinking(0);                       // guarantee the BASE variant is selected
dll.v7_initialize_inference();
```

With `%s` as the complete template, `v7sprintf` passes through the supplied text without adding
`<|im_start|>` delimiters, role headers, or a generation suffix. The resulting model input is the
caller's original text. This provides raw-completion behavior through `v7_chat`, although applications
that only need raw continuation can also use the dedicated `v7_generate` interface.

There are two requirements for this override. First, the replacement must happen before the inference
context is created: V7 copies the selected templates from MicroDB during initialization, so changing
`__sys_template` afterward does not affect the active context. Second, the application must disable
thinking with `v7_enable_thinking(0)` or the `-think 0` command-line option. Otherwise, a model with
`__sys_template@think` will select that variant instead of the modified base template.

The override affects fresh chat sessions using `reset=1` and `v7_generate` calls with
`raw_generation=0`, both of which select the first-turn template. Later chat turns normally use
`__sys_template@multi` when it is present. Applications that require the same unformatted behavior for
those turns must replace that key as well. `inference_example.exe` ships this hook as the
`-set-template` flag:

```
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -set-template "%s" -think 0 -i "The quick brown fox"
```

Because the override is `persist=0`, the override exists only in the process-local hash table. The
`lz4ht.db` file on disk is never modified, and the next run is back to the normal chat template.

If `v7sprintf` cannot render a required template, V7 reports a terminal engine error rather than
continuing with incomplete prompt data.

##### Complete control flow of a chat turn

Phases in execution order (label references point into `src/asm/chat.inc`; both `v7_chat` and the
generate/image paths funnel through this sequence):

```text
chat sequence entry (v7s_chat, internal)
  1. Validate output buffer / output_len          (FATAL on NULL / < 1)
  2. Resolve start position
       reset != 0 -> ctx->pos:= 0, c_pos = 0     (.chat_pos_done)
       reset == 0 -> c_pos = ctx->pos             (multi-turn continue)
  3. Cache sampler fields (vocab_size, temperature, topp,
     presence_penalty, rng_state, probindex) from ctx->sampler
  4. Render prompt template via v7sprintf         (selection table above;
                                                   FATAL on render failure)
  5. Publish raw user prompt to g_user_prompt     (for sidecar RAG queries
                                                   via v7s_get_user_prompt)
  6. Encode rendered prompt -> prompt_tokens      (BPE, max 32768 tokens)
       [RDTSC probe: rdtsc_after_tokenization]
  7. KV-cache bounds check: pos + n_tokens <= seq_len    (FATAL if exceeded)
  8. Prefill:
       n_tokens > 1 -> prefill_forward_q8/f32(transformer, tokens, n, start_pos)
       n_tokens == 1 -> forward_q8/f32(transformer, tokens[0], pos)   (single-token path)
       c_pos += n_tokens
       [RDTSC probe: rdtsc_after_prefill]
  9. Apply presence penalty to logits (skipped when penalty == 0.0)
 10. Sample FIRST token via v7s_sample          -> c_next_token
 11. v7s_stream_init_turn(c_pos)               , zero the SectorLogo stream
                                                   tri-array; record base position
 12. GENERATION LOOP (.chat_gen_loop), repeats until finish:
       a. SectorLogo HOOK: if enabled, call
          v7s_sectorlogo_on_token(ctx, candidate, pos, &out_tok, &temp, &topp)
          -> action code dispatch:
             0 ACCEPT   -> fall through to (b)      (hook disabled = same path)
             1 RESAMPLE -> re-run v7s_sample on the SAME logits (hook may have
                            changed temperature/topp via the out-pointers),
                            replace candidate, loop back to (a) so the hook
                            sees the new token
             2 EOS      -> jump to finish
             3 REWRITE  -> jump to the rewrite engine (below)
       b. Stop-token checks:
             candidate == tokenizer->eos_token_id            -> finish
             candidate in g_stop_ids[1..] (extra stop IDs)   -> finish
       c. max_gen check: gen_count >= max_gen (0 = unlimited) -> finish
       d. Decode candidate: vocab piece -> output buffer (bounds-checked;
                            output full -> finish), append decoded text to the
                            stream tri-array (v7s_stream_append_token), mark
                            token in the presence-penalty bitmap
       e. KV-capacity check: pos >= seq_len -> finish
       f. Forward pass forward_q8/f32(transformer, candidate, pos) -> logits
       g. pos++; gen_count++
       h. Apply presence penalty; sample NEXT token -> candidate; loop to (a)
 13. Finish (.chat_finish):
       [RDTSC probe: rdtsc_after_generation; number_token_generate = gen_count]
       ctx->pos:= c_pos                          (persist position for multi-turn)
       NUL-terminate output; return

  REWRITE ENGINE (.chat_hook_rewrite, entered when the hook returns 3):
     1. Save old stream token count
     2. v7s_stream_compute_rewrite_diff: longest-common-prefix diff between the
         old stream and the replacement string from SetStream
        -> T_diverge (first differing token index) + suffix string
     3. Presence-penalty bitmap cleanup:
          btr loop clears bits of all DISCARDED tokens [T_diverge.. old_count)
          bts loop re-marks tokens of the KEPT prefix [0.. T_diverge)
          (a token in both ranges keeps its bit - the prefix occurrence persists)
     4. c_pos:= base_pos + T_diverge; gen_count:= T_diverge
     5. Truncate the caller output buffer at the divergence byte; NUL-terminate
     6. Re-encode the replacement suffix (encode into sl_rewrite_tok_buf,
        cap 16K tokens)
     7. RE-PREFILL LOOP: for every replacement token:
          decode piece -> output buffer, append to stream tri-array,
          mark presence-penalty bit, forward_q8/f32 at c_pos, c_pos++,
          gen_count++ (max_gen and KV-bounds checks run per token)
        Zero replacement tokens -> re-forward the last kept token once to
        refresh logits (.chat_rewrite_zero)
     8. Sample the next token from the fresh logits and jump back into the
        main GENERATION LOOP at (a) - the hook runs again on the resumed token

     StreamingLLM (g_stream_enable = 1): the rewrite action has two paths:
       pos <= C -> ordinary token-aligned rewind and suffix re-forward;
         newly crossed window boundaries still deliver real displacements
       pos >  C -> require live out_tokens history (g_stream_otk_live),
         protect the window-displacement latch, re-forward the necessary
         earlier resident-window rows and the changed suffix; restore
         the validity tracker on exit. A wrapped KV restore without token
         IDs cannot be rewritten and terminates with a clear error.
     Pending displacements flush BEFORE the rewrite modifies positions.
```

The hook only ever sees GENERATED tokens: template and prompt tokens are consumed in phases 4-8 before
`v7s_stream_init_turn` and never pass through `v7s_sectorlogo_on_token`. The stream (`GetStream`)
therefore contains only generated text this round (plus any replacement text after a rewrite).

##### DB keys consumed by the chat/generate sequence

Every key the flow above reads at runtime, with the phase that reads it and the consumer. Keys marked
mandatory FATAL the process when missing; optional keys have a documented fallback.

| Key | Read at | Used for | Missing = |
| --- | --- | --- | --- |
| `__sys_template` | init, `inf_tokenizer.inc` | Turn-1 base template, phase 4 | FATAL |
| `__sys_template_sys` | init | System+user base template, phase 4 | FATAL |
| `__sys_template@think` / `__sys_template_sys@think` | init | Thinking variants of the above, selected by `v7_enable_thinking` | falls back to base |
| `__sys_template@multi` / `__sys_template@multi@think` | init | Turn-2+ templates, phase 4, when `pos > 0` | falls back to turn-1 |
| `__sys_template@img` | init | Image-turn template override, phase 4, set by `v7_chat_image` and `v7_generate_image` | falls back to normal selection |
| `__sys_stop_token_ids` | init | EOS plus extra stop IDs, phase 12b, in the `g_stop_ids` BSS array | FATAL |
| `__sys_rms_eps` | init | Per-model RMSNorm epsilon, used by every forward | FATAL |
| `__sys_config` / `__sys_mem_layout` / `__sys_model_weights` / `__sys_weight_meta` | init, `inf_microdb` and `inf_weights` | Transformer and RunState construction, consumed in phases 7-8 | FATAL |
| `__sys_tokenizer_hdr` / `__sys_vocab_bulk` / `__sys_merge_scores` | init, `inf_tokenizer` | BPE encode and decode, phases 4, 6, 12d | FATAL |
| `__sys_mrope_bound` | prefill, `mrope.inc` | M-RoPE position bound for image turns. A value of 0 or a missing key means plain 1D text RoPE, which is the early-out for LLaVA, SmolVLM and text-only models | early-out |
| `__sys_vit_weights` | first image turn, `vit_forward.inc` | ViT tower weights for `v7s_vit_encode_callback` | FATAL on image turn |
| `__sys_vision_config` | init | VCFG block holding every model-dependent ViT dimension and stride | FATAL for VLMs |
| `__sys_vision_prompt_fmt` | init | Vision prompt prefix, marker and suffix, assembled in `chat_image.inc` | FATAL for VLMs |
| `__sys_image_newline_emb` | init | Anyres image-newline embedding, LLaVA-OV only | skipped for other models |
| `logo:sidecar` | chat entry, in `v7s_sidecar_refresh` | The SectorLogo sidecar script text, phase 12a. Stored via `v7_db_add("logo:sidecar", script, len+1, 0)` by the host, for example `inference_example.exe -script <file>`, and activated automatically by the chat-entry database check. The example program accepts `-rewrite`, but this flag no longer calls a hook-enablement export. Not persistent: never written to the DB file. | internal hook returns ACCEPT without evaluating a script |
| `__ini__*` | `v7_init`, `v7_init.inc` | Dispatch tuning, per-shape INI sections, consumed by every matmul in phases 7-8, 12f | FATAL for the mandatory ones |

For `logo:sidecar`, the host first loads the model database and adds the script with
`v7_db_add`, or uses a key already present in a custom-built database. At each chat entry, the
runtime checks the key and refreshes the parsed callback when necessary. The Logo runtime initializes
on demand; modified scripts are parsed at the next chat entry. On each generated token, the interpreter
executes the cached program with `:Piece` and `:TokenNumber` bound. The per-token
callback executes the cached script; the script-key lookup and refresh occur at chat entry.

#### `v7_generate(prompt, output, output_len, raw_generation)`

Single-turn generation uses separate temporary KV-cache allocations, leaving the active chat-session
cache intact. They are allocated with `mdb_alloc_mem`, which can fall back to file-backed memory when a
normal allocation fails; file backing is not unconditional.

By default, `v7_generate` formats the prompt using the model's first-turn chat template. Set
`raw_generation=1` to bypass that template and request a raw continuation with the EOS-only stop set.

**Template usage:** `v7_generate` with `raw_generation=0` renders the prompt through the model's
**turn-1** template. Internally it swaps `RunState->key_cache` /`value_cache` to freshly allocated
generate caches (`mdb_alloc_mem`, which falls back to the file-backed allocator under memory pressure),
zeroes the position, and runs the same chat sequence. The `.chat_no_system` path with `pos == 0` selects
`g_tpl_t1`, or its `@think` variant when thinking is enabled. Afterwards it restores the chat caches,
position, and the sampler RNG state. With `raw_generation=1` the template is skipped entirely and the
stop set is EOS-only. The system template is never used, because the generate path has no system prompt,
and the `@multi` template is never used, because generate is always turn 1 on its own cache.

Consequences for callers:
- The full chat control flow above, including the SectorLogo hook and
  rewrite engine, runs unchanged inside `v7_generate`, so the sidecar sees
  these tokens too.
- The chat session's KV cache, position, and PRNG state are never touched;
  a later chat turn continues as if the generate never happened.
- `v7s_get_user_prompt` returns the generate prompt during generate turns.
  The inner chat sequence publishes the prompt at turn start, and the turn
  finish clears the pointer, so it is NULL before the first turn and again
  after every turn completes; it is non-NULL only while a turn is running.
  Sidecars that need to distinguish can
  test the stream: generate turns start at pos 0 with no prior session.
- Image generate (`v7_generate_image`) delegates through the same implementation
  with the image-turn flag set, so `__sys_template@img`, when present,
  overrides the turn-1 template for the vision prompt.

#### `v7_get_generation_statistic()` returns uint64_t*

The `v7_get_generation_statistic` function returns a pointer to a flat block of ten 64-bit statistics
recorded by RDTSC probes in the chat sequence (`chat.inc`). The slots have the following layout:

| Index | Variable | Meaning |
| --- | --- | --- |
| 0 | `rdtsc_chat_start` | TSC at chat-sequence entry |
| 1 | `rdtsc_after_tokenization` | TSC after encode() |
| 2 | `rdtsc_after_prefill` | TSC after prefill/single-token forward |
| 3 | `rdtsc_after_generation` | TSC at generation finish |
| 4 | `number_token_prefill` | prompt tokens encoded this round |
| 5 | `number_token_generate` | tokens generated this round |
| 6 | `rdtsc_vit_encode_start` | TSC before ViT encode; 0 outside image turns |
| 7 | `rdtsc_vit_encode_end` | TSC after a successful ViT encode; 0 when ViT did not run |
| 8 | `number_token_prefill_vision` | image-turn prefill tokens |
| 9 | `number_token_generate_vision` | image-turn generated tokens |

The ASM probes only write; callers reset the block between rounds and convert cycle deltas to time with
their own TSC calibration (see `examples/inference_example.c` and `bench_qwen.bat` ).

#### `v7_unload(lib)`

`v7_unload` releases the DLL module and clears the function pointers stored in the library structure.
Applications should complete inference teardown before unloading the module.

### Public Sampling and Session Configuration

V7 maintains sampling and session state in its DLL-managed inference context. The public setters
configure that state after successful inference initialization and before the relevant generation
operation. Unless a function states otherwise, configuration applies to the current library instance
rather than to an independently allocated session. The following table documents each setter and its
expected argument range.

| Function | Signature | Description |
| --- | --- | --- |
| `v7_set_sampler_temperature` | `(float t)` | Sampling temperature. 0 selects greedy argmax decoding, the deterministic default. The public setter is the documented input-normalization boundary: NaN, negative values (including -inf) and `0 < t < 1e-6` are fixed up to `1e-6`, silently; `+inf` is a caller bug and FATALs. |
| `v7_set_sampler_topp` | `(float p)` | Top-p nucleus sampling in [0,1]; 1.0 disables. |
| `v7_set_sampler_topk` | `(int k)` | Only 0 is accepted. `v7s_sample` has no top-k parameter, so a nonzero value is rejected at the entry with a FATAL rather than stored. |
| `v7_set_sampler_presence_penalty` | `(float p)` | Presence penalty applied to already-generated tokens. The public setter requires a finite value in `[-2, 2]`: NaN/inf or an out-of-range magnitude FATALs (such a value would poison the logits). |
| `v7_set_sampler_seed` | `(unsigned __int64 seed)` | RNG seed for stochastic sampling. |
| `v7_set_quantize` | `(int mode)` | 0 = F32 forward, 1 = Q8 forward. Any other value FATALs at the entry. The context must already exist; set the mode before calling `v7_chat` or `v7_generate`. |
| `v7_set_max_gen` | `(int cap)` | Generation cap; 0 = unlimited, generating until EOS or until the output buffer is full. A negative cap FATALs at the entry (the generation loop compares unsigned, so a negative value would silently read as unlimited). |
| `v7_get_position` | `(void)` returns int | Current KV-cache position, the multi-turn state. |
| `v7_set_position` | `(int pos)` | Rewind to a position no greater than the current position (no negative positions or forward jumps). Under StreamingLLM, a wrapped rewind to `T > C` reduces the valid-window count and adjusts the displacement warm-up; rewinding from `P > C` to `T <= C` re-forwards rows `[S,T)` from live token IDs to restore linear cache layout. The latter fails if the live ID history is unavailable (for example, after loading a KV file). A no-op position set is allowed. |
| `v7_get_token_probs` | `(const int* token_ids, int n, float* out_probs)` | Restricted softmax over candidate token ids, read from the logits buffer of the last forward/prefill position. `n` in 1..256; ids must be inside vocab range. Fills `out_probs` with one probability per id (sums to 1.0). Use after a one-token-slot raw generate to read decision-position probabilities; see `native_api.md`. Gated entry: pair the call with `v7_engine_is_dead()`. |
| `v7_set_worker_priority` | `(int priority)` | Set the Win32 priority of EVERY live worker thread. Win32 constants: -15 idle, -2 lowest, -1 below normal, 0 normal, 1 above normal, 2 highest, 15 time critical (the INI default is 15). Callable at any time; when the pool does not exist yet (lazy creation), the request is parked and applied at pool creation. Last call wins; purely scheduler-level, output is unaffected. The environment variable `v7_override_prio` applies the same override at pool creation, before the first dispatch. |
| `v7_enable_thinking` | `(int on)` | Select thinking or base template variants. The engine default is on, reset per context. |
| `v7_db_add` and `logo:sidecar` | `(key, data, len, persist)` | Register or update the per-token SectorLogo script using `v7_db_add("logo:sidecar", script, script_len + 1, 0)`. The chat entry checks for the key and automatically activates the hook. No separate hook-enablement export is required. |
| `v7_db_add` | `(key, data, len, persist)` returns int | Add or update a MicroDB entry, for example a sidecar script or template override placed before init. Border contract: `len` must be non-negative and `data` non-NULL whenever `len > 0`; a violation FATALs at the entry (the 32-bit len is zero-extended internally, so a negative value would otherwise reach the engine as ~4 GB). |
| `v7_db_find` | `(const char* key, unsigned __int64* out_len)` returns void* | Look up a MicroDB entry and return its 64-bit byte length. |
| `v7_db_load` / `v7_db_save` | `(path)` returns int | Direct MicroDB file load/save. |
| `v7_encode` | `(text, tokens, n_tokens, max_tokens)` | Tokenize text with the model's BPE tokenizer. All three pointers (`text`, `tokens`, `n_tokens`) must be non-NULL; `max_tokens <= 0` is handled cleanly (0 tokens written). NULL pointers FATAL at the entry. |
| `v7_get_generation_statistic` | `(void)` returns `unsigned __int64*` | Public alias of `v7s_get_generation_statistic`, returning the 10-slot RDTSC block shown in the table above. |
| `v7_get_best_fitness` | `(void)` returns `unsigned __int64*` | Cycle-tuner pool fitness: per-shape u64 array (64 entries) of the mean `c`-measurement (RDTSC cycles) over the population's front 10 genome slots, folded at each population sweep wrap. Valid index range is `[0, v7_get_active_shape_count())`; 0 means the shape has not completed a sweep yet. |
| `v7_get_pool_fitness` | `(void)` returns `unsigned __int64*` | Same array layout, the mean `c`-measurement over every non-tail genome slot (the whole pool outside the mutation tail). |
| `v7_get_active_shape_count` | `(void)` returns int | Number of shapes the dispatcher registered in the active-shape registry; also the valid index range of both fitness arrays. |
| `v7_shutdown` | `(void)` | Shut down the worker pool explicitly. `v7_free_inference` also invokes worker-pool shutdown during context destruction, so an additional call is not required for ordinary teardown. |
| `v7_get_internal_inference_ctx` | `(void)` returns `inference_context*` | **Internal.** Deep-tool escape hatch returning the real internal context, or 0 when not initialized. Diagnostic access only; normal programs never touch this. WRITING through the returned context bypasses every public-tier contract and is undefined behaviour. Not declared when the host compiles with `V7_ONLY_SAFE` (see below). |

#### V7_ONLY_SAFE, the public-tier-only compilation mode

Applications that need only the public interface can define `V7_ONLY_SAFE` when including
`src/v7_lib.h`. This suppresses the internal declarations while preserving the public API:

```c
#define V7_ONLY_SAFE
#include "v7_lib.h"
```

In this mode the `v7_lib` struct contains ONLY the public `v7_*` members, `v7_load` resolves only the
public tier, and the internal declarations (kernels, MicroDB internals, `encode`, `forward_*`,
SectorLogo internals, process execution, diagnostic gates, `v7_get_internal_inference_ctx`, the
`inference_context` struct) are not declared at all. The DLL is unchanged; the flag is purely a
header-side compile-time contract, so a host compiled with `V7_ONLY_SAFE` cannot reach the internal
exports by accident. Deliberate `GetProcAddress` bypasses remain possible (they always were) but are no
longer declared, and writing through the internal context is undefined behaviour regardless.

### Public API Error Reporting

The rewrite path has finite internal storage. In the current source, a replacement that reaches the
16,384-token re-encoding limit takes a fatal-error path; callers must not interpret this as a successful
truncated result. Other output-buffer limits and intentional bounded-output behavior follow their
separately documented contracts. The behavior of a fatal internal error differs from a regular public
argument-validation result.

The public error contract distinguishes input validation from a terminal failure of the initialized
engine. Certain public entry points intercept an internal `FATAL` condition, preserve a diagnostic
reason, and return control to the caller with the engine marked unavailable. Applications can then
inspect that state through `v7_engine_is_dead()` and retrieve its explanation. Other low-level entry
points may not provide the same boundary. The descriptions below identify the guarantees for the public
functions without implying that all native exports recover from internal faults.

```c
dll.v7_chat("what is a cat?", 0, out, sizeof(out), 1);
if (dll.v7_engine_is_dead())
{
    printf("engine error: %s\n", dll.v7_get_death_reason());
    dll.v7_free_inference();
    v7_unload(&dll);
    return 1;
}
```

A gated function that stops because of an internal error returns its normal failure value, or no value
for a `void` function. Check `v7_engine_is_dead()` when a failure value is ambiguous.
`v7_get_death_reason()` is empty while the engine is running and contains the diagnostic text after an
error. `v7_free_inference()` releases the inference context, shuts down its worker pool, and resets the
engine error state.

#### `v7_get_version`

```c
typedef const char* (*v7_get_version_fn)(void);
const char *ver = dll.v7_get_version();   // release version baked into the DLL
```

Returns a pointer to a static NUL-terminated string with the release version baked into the DLL. Tools
and bug reports print it so a report can prove which build produced the output. The literal is stored in
`src/asm/version.inc`; `src/check_release.py` checks it against README, the CHANGELOG top entry and
`FILE_ID.DIZ` so all four always agree.

#### Introspection and auxiliary exports

```c
typedef int (*v7_get_console_width_fn)(void);          // visible console columns (80 fallback)
typedef int (*v7_get_think_mode_active_fn)(void);      // 1 when thinking templates are selected
typedef int (*v7_get_gen_count_fn)(void);              // tokens generated this turn
typedef int *cpu_features_ok_ptr;                      // CPU gate result (1 = baseline met)
typedef void (*v7s_bicubic_resize_fn)(const unsigned char *src, int src_w, int src_h,
                                      unsigned char *dst, int dst_w, int dst_h);
```

- `v7_get_console_width` returns the visible console width. When stdout is redirected, it first attempts
to open `CONOUT$` and query the attached console; it returns 80 only when neither query succeeds.
SectorLogo uses the value to format its banner.
- `v7_get_think_mode_active` / `v7_get_gen_count` - generation-loop state
  for progress UIs and sidecar scripts. `v7_get_think_mode_active` reports
  the thinking-template selection (the engine default), not whether a think
  block is currently open; `v7_get_gen_count` is reset at each chat turn
  start and grows with every generated token, so it reads 0 exactly at the
  first hook call of a turn.
- `cpu_features_ok` is a BSS flag set by the CPU feature check at load time. The `v7_initialize_inference` function reports a terminal error error if the flag is zero; the flag remains available for diagnostics.
- `v7s_bicubic_resize` - the Catmull-Rom image rescaler used by the vision
  input stage (PIL-BICUBIC-compatible half-pixel centers).

The remaining `.def` entries are internal exports consumed by the dev-side validators and the autotuner;
they are grouped under the internal comment in `src/v7_threading.def` and are **not part of the
documented public API**. For completeness, the full internal/tuning surface (not stable ABI):

| Export | Kind | Used by |
| --- | --- | --- |
| `matrix_multiplication`, `matrix_multiplication_mode` | function | direct matmul entry points used by dispatch tests |
| `init_prefetch_backup`, `set_prefetch_distance`, `set_prefetch_type` | function | SMC prefetch tuning (tuner, loadtests) |
| `vit_max_chunks`, `vit_min_chunk_cols` | BSS tune parameter | ViT chunking autotuner |
| `fused_bias_ptr` | BSS pointer | fused-kernel QKV bias source (validators) |
| `fbm_alloc`, `fbm_free` | internal label (not exported) | file-backed KV-cache allocator (generate path, its validator) |
| `v7d_q8_fused` | function | fused Q8 matmul section (fused validator) |
| `g_ht` | BSS pointer | the MicroDB hash table (DB tools, crash dumps) |
| `shape_blocks`, `debug_param_set_0` | BSS / function | dispatch autotuner parameter slots |
| `cpu_features_ok` | BSS flag | **test-only writable gate bypass** — the CPU gate publishes it once at load; only `test_engine_death` writes it to exercise the refusal paths. Nothing else may write it. |

Internal exports may change or disappear between releases. Applications should use the documented public
API through `v7_lib.h` instead of depending on those exports.

A `FATAL` on a worker thread or outside a gated public call still follows the terminal path: print the
diagnostic and terminate the process. The `test_engine_death.exe` program is a diagnostic test for the
public error path; it is not required for normal inference.

### Vision for VLMs - Image Chat

For a supported vision-language model, the image-generation interface accepts an RGB buffer and the
image dimensions supplied by the application. The caller is responsible for loading or decoding the
image file; the DLL performs the model-specific processing once the pixel data is available. The
function signatures and buffer requirements below define the boundary between application image handling
and inference.

**Input contract (checked at the public entries):** `img_w` and `img_h` must be positive, strictly below
65536 (the vision DDA runs in 16.16 fixed point, where a dimension of 65536 or more wraps the
accumulators), and their product must fit in 32 bits (`w * h <= 2147483647`), the vision stage reads the
pixel count as a dword, so a negative or wrapped dimension would allocate from a wrapped count and read
past the buffer; any violation FATALs at the entry. `raw_rgb` must point to a contiguous buffer of
exactly `img_w * img_h * 3` bytes (packed RGB, no header, no padding), the engine cannot verify the
buffer's length, so that part of the contract is on the caller. Image sizes that are not multiples of
the model's patch size are valid: the vision grid floors the division and crops to the patch lattice,
with every read staying inside `w * h` by construction.

The public interface exposes two vision-language chat entry points. They delegate to the internal `v7s_*`
functions documented below, which are intended for implementation and diagnostic work rather than
ordinary embedding.

| Function | Signature | Description |
| --- | --- | --- |
| `v7_chat_image` | `(user_prompt, system_prompt, raw_rgb, img_w, img_h, output, output_len, reset)` returns int | Public image chat. `reset` follows `v7_chat` semantics: 0 continues a multi-turn session, a nonzero value starts a fresh one. Returns 0 on success, so pair failure handling with `v7_engine_is_dead()`. |
| `v7_generate_image` | `(user_prompt, raw_rgb, img_w, img_h, output, output_len)` | Public image generate: a fresh single-turn session on a separate file-backed KV cache, leaving the chat session untouched. |

| Function | Signature | Description |
| --- | --- | --- |
| `v7s_set_vision_embeddings` | `(float* embeddings, int emb_count, float* ds0, float* ds1, float* ds2)` | Upload projected vision embeddings and DeepStack feature tensors for the next generation session. All feature tensors are optional and may be NULL. |
| `v7s_vit_encode_callback` | `(const unsigned char* raw_rgb, int img_w, int img_h, float** out_emb, int* out_count, float** out_ds0, float** out_ds1, float** out_ds2)` returns int | Assembly vision encoder for the active VCFG configuration. It computes only and performs no file I/O. The image chat and generate paths call it with caller-supplied raw RGB. Returns 0 on success. Border contract (typed decision, 2026-10-21): this export carries NO dimension guard, `img_w`/`img_h` up to 65535 and non-negative are the caller's contract, enforced by the two public entries (`PUB_VALIDATE_IMAGE`), not duplicated here. The export is documented internal (requires explicit `GetProcAddress`), and the internal tier trusts its callers; adding an inner guard would be unreachable code behind the only two real call sites. Callers reaching it directly owe the same validation the public tier performs. |
| `v7s_vit_forward` | `(const float* pixels, int grid_h, int grid_w, const void* weights, float* out_main, float* out_ds0, float* out_ds1, float* out_ds2)` | Assembly ViT forward pass using the configured depth and projector. `weights` points at the vision weights stored in the model database. |

**Vision processing sequence:** patch_embed, then the configured transformer blocks, then feature
extraction where enabled, then the configured projector, producing vision embeddings injected at the
model's image-token positions in the text decoder. The number of dispatches depends on the configured
vision depth and projector.

### Kernel Functions

The following exports provide direct access to native computation and are intended for tools,
validators, and specialized integrations. Applications using `v7_chat` or `v7_generate` do not need to
invoke these functions separately.

#### Scheduled Matmul, dispatched across worker threads

| Function | Signature | Description |
| --- | --- | --- |
| `v7d_q8` | `(float* xout, v7_q8_tensor* x, v7_q8_tensor* w, int n, int d)` | Q8xQ8 matmul |
| `v7d_f32` | `(float* xout, v7_f32_tensor* x, v7_q8_tensor* w, int n, int d)` | F32xQ8 matmul |
| `v7d_fused_f32` | `(float* xout, v7_f32_tensor* x, v7_q8_tensor* w, int n, int d, float* norm_weights, int is_q)` | Fused F32 matmul + RMSNorm + RoPE |
| `v7d_q8_fused` | `(float* xout, v7_q8_tensor* x, v7_q8_tensor* w, int n, int d)` | Fused Q8 quantize + matmul. Not wrapped in v7_lib.h; use raw `GetProcAddress`. |
| `v7d_swiglu` | `(float* xout, v7_f32_tensor* x, v7_q8_tensor* gate_w, int n, int d, v7_q8_tensor* up_w)` | SwiGLU F32xQ8 |
| `v7d_swiglu_q8` | `(float* xout, v7_q8_tensor* x, v7_q8_tensor* gate_w, int n, int d, v7_q8_tensor* up_w)` | SwiGLU Q8xQ8 |

#### Attention

| Function | Signature | Description |
| --- | --- | --- |
| `v7d_attention` | `(v7_run_state* s, v7_config* p, int pos, unsigned __int64 loff, int kv_dim, int kv_mul)` | Multi-head attention |
| `v7d_attention_prefill` | `(v7_run_state* s, v7_config* p, float* q_storage, float* att_out_storage, int start_pos, int n_tokens, unsigned __int64 loff, int kv_dim, int kv_mul)` | Prefill attention |

#### Utility Kernels

| Function | Signature | Description |
| --- | --- | --- |
| `v7d_quantize` | `(char* out_q, float* in_x, float* out_s, int n)` | FP32 to INT8 quantize with GS=64 |
| `v7d_memcpy` | `(void* dst, const void* src, int byte_count)` | Dispatched memcpy |

#### Direct-Call Kernels

| Function | Signature | Description |
| --- | --- | --- |
| `v7s_rmsnorm` | `(float* o, float* x, float* weight, int size)` | RMSNorm |
| `v7s_add_inplace` | `(float* x, float* y, int n)` | Element-wise add-inplace |
| `v7s_fused_add_rmsnorm` | `(float* o, float* residual, float* add, float* weight, int dim)` | Fused add + RMSNorm |
| `v7s_rmsnorm_rope_heads` | `(float* buf, int n_heads, int head_dim, float* norm_weight, float* cos_table, float* sin_table, int pos, int rope_max_dim)` | RMSNorm + RoPE per heads |
| `v7s_rmsnorm_quantize` | `(char* out_q, float* in_x, float* out_s, int n, const float* weight)` | Fused RMSNorm + FP32?INT8 quantize. Uses one global scale and is never chunk-dispatched. |
| `v7s_sample` | `(float* logits, int vocab_size, float temperature, float topp, unsigned __int64* rng_state, v7_probindex* probindex)` returns int | AVX2 softmax + AES-NI RNG + top-p sampling |

#### Forward Pass

| Function | Signature | Description |
| --- | --- | --- |
| `forward_f32` | `(void* transformer, int token, int pos)` returns float* | Full forward pass (F32 activations) |
| `forward_q8` | `(void* transformer, int token, int pos)` returns float* | Full forward pass (Q8 activations) |
| `prefill_forward_f32` | `(void* transformer, int* tokens, int n_tokens, int start_pos)` returns float* | Prefill forward (F32) |
| `prefill_forward_q8` | `(void* transformer, int* tokens, int n_tokens, int start_pos)` returns float* | Prefill forward (Q8) |

#### Tokenizer

| Function | Signature | Description |
| --- | --- | --- |
| `encode` | `(void* tok, const char* text, int* tokens, int* n_tokens, int max_tokens)` | BPE tokenizer - encode text to token IDs |

#### Fused-Kernel Support

| Function | Signature | Description |
| --- | --- | --- |
| `v7s_rope_init` | `(int head_dim, float rope_base)` | Build RoPE cos/sin tables |
| `v7s_set_fused_rope` | `(int pos, int head_dim)` | Set fused rope pointer for next dispatch |

### MicroDB Functions

These exports provide direct access to the embedded database. Applications should use the documented
value formats and ownership rules when storing or retrieving data; system keys are reserved for model
configuration and runtime resources.

#### Database file format (`lz4ht.db`)

The file starts with a 20-byte header, followed by a stream of blocks compressed with the engine's lz4
like lz77 codec and a 16-byte zero end marker:

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | format signature `"v7db"` (ASCII bytes `v`,`7`,`d`,`b`, little-endian dword `0x62643776`). A file without it is not a v7 database; `db_load` refuses it with a FATAL. |
| 4 | 4 | File-format version, currently `1`. A file written by a different version loads anyway; `db_load` prints `db_load: warning: model file was made by a different version` and continues. |
| 8 | 4 | Persistent entry count. |
| 12 | 8 | Key+value pool size (the exact pre-allocation target for the load). |

The variables inside the database (config, weights, tokenizer, templates, INI keys) are INTERNAL. They
are listed in the key table above for reference only; their layout and meaning can change between
versions without notice, and behaviour when a host hand-crafts entries is undefined. There are no format
validators beyond the format signature/version check by design, the database is produced by
`build_db.exe` and consumed by the engine that built the format.

| Function | Signature | Description |
| --- | --- | --- |
| `ht_init` | `(void)` | Initialize hash table |
| `ht_destroy` | `(void)` | Destroy hash table and free memory |
| `ht_add` | `(const char* key, void* data, unsigned __int64 data_len, unsigned int persist)` returns int | Add/update entry |
| `ht_find` | `(const char* key, unsigned __int64* out_len)` returns void* | Find entry by key, returns pointer and full 64-bit byte length |
| `db_save` | `(const char* filename)` | Save hash table to a file compressed with the lz4 like lz77 codec |
| `db_load` | `(const char* filename)` | Load hash table from file |
| `db_set_readonly` | `(int flag)` | Lock database to read-only |
| `mdb_compress` | `(const void* src, unsigned __int64 src_len, void* dst)` returns `unsigned __int64` | MicroDB compress, 64-bit sizes |
| `mdb_decompress` | `(void* src, unsigned __int64 src_len, void* dst, unsigned __int64 max_dst)` returns `unsigned __int64` | MicroDB decompress, 64-bit sizes |

### Runtime Control

| Function | Signature | Description |
| --- | --- | --- |
| `v7s_set_spin` | `(int flag)` | Declare the inference activity phase: 1 = workers pause-spin and stay fork-receptive; 0 = workers return to the OS wait. `v7s_stop_spin` ends chat activity internally at turn completion. |
| `v7s_shutdown` | `(void)` | Shut down the worker threads. The inference-context destructor also invokes this routine. |
| `v7s_get_num_workers` | `(void)` returns int | Get number of worker threads |

### SectorLogo Generation Callback

A generation sidecar is a SectorLogo program stored in MicroDB under the key `logo:sidecar`. V7 uses
this entry to obtain the script associated with the generation hook. At the callback boundary, the
script can inspect the pending token, interact with the accepted stream, and invoke host commands for
sampling or replacement. There is no separate requirement to register a native C callback: a SectorLogo
script can invoke a host function through the FFI when native application logic is needed. When
`SetStream` changes accepted text in an ordinary or unwrapped streaming session, V7 can reuse the
unaffected KV-cache prefix and prefill the replacement suffix. A wrapped StreamingLLM session instead
reconstructs the required state from position zero using the recorded prompt history. Example sidecar
programs are located in `src\logo\`.

| Function | Signature | Description |
| --- | --- | --- |
| `logo:sidecar` hook configuration | MicroDB key | Store the NUL-terminated script through `v7_db_add`. At the beginning of every chat call, the runtime compares the key contents with the cached script and parses changed scripts. Removing the key disables the callback. |
| `v7s_sectorlogo_on_token` | `(void* ctx, int token, int pos, int* out_tok, float* p_temp, float* p_topp)` returns int | Per-token hook, called internally by chat. Hosts never call it. Returns an action code: 0=ACCEPT, 1=RESAMPLE, 2=EOS, 3=REWRITE. |
| `v7s_sectorlogo_interp` | `(const char* script)` returns int | Standalone parse and eval of a script string, returning the last OUTPUT value. |

To load a sidecar from the command line, run `inference_example.exe` with `-script <file>`. The example
places the script into the database-backed hook configuration; `-rewrite` is accepted for
compatibility but is not required to activate the callback.

### Process Execution and String Formatting

| Function | Signature | Description |
| --- | --- | --- |
| `v7_execute` | `(char* exe, char* p1, char* p2, char* p3)` returns `char*` | Spawn a child process with up to 3 string arguments, capture stdout+stderr, and return an interned string atom. Used by SectorLogo `CallLibrary ["SELF" "v7_execute"...]`; `DefineCommand` uses the formatted `v7_execute_fmt` path. |
| `v7_do_fatal` | `(char* message)` returns `void` | Print the NUL-terminated message with the standard output macro, then die through the standard engine FATAL path. Script-reachable fatal for SectorLogo error paths that must stop the engine loudly instead of silently continuing. Wrapped by the Logo `DoFatal` primitive. |
| `v7sprintf` | `(char* buffer, int buf_size, const char* format,...)` returns `int` | Bounded string formatting. Supports `%s` (string), `%d` (int32), `%c` (char), `%f` (float32), `%%` (literal percent). Always null-terminates; never writes past `buffer + buf_size`. Returns characters written. |

### Prefetch Tuning

| Function | Signature | Description |
| --- | --- | --- |
| `init_prefetch_backup` | `(void)` | Save the original prefetch bytes for restore, called once at init |
| `set_prefetch_distance` | `(int slot, int distance)` | Set the prefetch distance for a kernel slot. SMC-patched; a distance below 32 disables the slot. |
| `set_prefetch_type` | `(int slot, int type)` | Set the prefetch instruction type for a kernel slot: 0=NTA, 1=T0, 2=T1, 3=T2, 4=PREFETCHW, 5=PREFETCHWT1, 6=CLDEMOTE |

The prefetch tuning controls operate on the native execution configuration and are not part of the
ordinary generation lifecycle exposed by `v7_lib.h`. Tools requiring these exports can resolve them
through `GetProcAddress` and must respect the documented initialization and patching requirements.
Applications that use the normal configuration do not need to call them: per-shape settings are applied
by the runtime according to the configured tuning data. The runtime obtains the per-shape configuration
from the `[NxD]` sections described in the tuning manual. There are 41 active slots, numbered 0 through
40. Slots 39 and 40 control attention-prefetch types but do not have patchable displacement values. A
`pf_dist_*` setting below 32 replaces the corresponding slot with a multi-byte NOP; the type cannot be
patched again until its distance is restored.

### Export list

All 117 exports grouped by subsystem, from `src/v7_threading.def`. The public `v7_*` tier is resolved
into the `v7_lib` struct by `v7_load`; the remainder are low-level SMC, diagnostic and BSS symbols
accessed via raw `GetProcAddress`.

#### Public v7_* tier, context hidden in BSS

- `v7_init` - Open the database and initialize engine state. Call before `v7_initialize_inference`.
- `v7_db_load` / `v7_db_save` - Direct MicroDB file load/save
- `v7_initialize_inference` - Create the inference context in the BSS singleton. Returns 1 when ready.
- `v7_free_inference` - Parameterless destroy that also resets the engine error state
- `v7_chat` - Multi-turn chat (template, tokenize, prefill, generate, decode)
- `v7_generate` - Single-turn generate (`raw_generation=1` = no template, EOS-only stop)
- `v7_chat_image` / `v7_generate_image` - Public image chat/generate
- `v7_set_sampler_temperature` / `v7_set_sampler_topp` / `v7_set_sampler_topk` / `v7_set_sampler_presence_penalty` / `v7_set_sampler_seed` - Sampler configuration
- `v7_set_quantize` / `v7_set_max_gen` - Forward mode and generation cap, where 0 means unlimited
- `v7_get_position` / `v7_set_position` - KV-cache position query and rewind
- `v7_get_token_probs` - Restricted softmax over candidate token ids after the last forward/prefill position (typed decisions, `native_api.md`)
- `v7_set_worker_priority` - Runtime Win32 priority override for the whole worker pool (-15 idle.. 15 time critical); the `v7_override_prio` env var applies it automatically at init
- `v7_enable_thinking` - Thinking or base template variants
- `v7_db_add("logo:sidecar", ...)` - Install or update a per-token SectorLogo script; the chat entry enables the callback when the key is present
- `v7_db_add` / `v7_db_find` - MicroDB entry add/find
- `v7_encode` - Tokenize text
- `v7_get_generation_statistic` - Public alias of the RDTSC statistics block
- `v7_get_best_fitness` / `v7_get_pool_fitness` - Cycle-tuner per-shape pool fitness averages (u64 arrays, front-10 and non-tail means, folded at each sweep wrap)
- `v7_get_active_shape_count` - Active-shape registry occupancy, the valid index range of both fitness arrays
- `v7_engine_is_dead` / `v7_get_death_reason` - public API error-state queries
- `save_kv_cache` / `load_kv_cache` - KV-cache persistence of the used rows and position, lz4 like lz77 codec-compressed
- `v7_set_streaming` - StreamingLLM ring cache switch (attention sinks + rolling window + window-displacement callback); 0 = accepted, -1 = rejected
- `v7_shutdown` - Explicit worker-pool teardown; ordinary context destruction also shuts down the pool
- `v7_get_internal_inference_ctx` - **Internal.** Deep-tool escape hatch returning the internal context pointer; reading is diagnostic, writing is undefined behaviour, and `V7_ONLY_SAFE` hosts do not see it

##### Worker priority control (`v7_set_worker_priority`, `v7_override_prio`)

The worker pool is created LAZILY: `v7_init` loads the database but the worker threads are only spawned
by the first dispatched operation (first `v7_chat` / `v7_generate` / dispatched memcpy), inside a
CAS-guarded `initialize_once`. The worker pool is created once at that point with the priority
configured in `threading_proxy.ini` (`worker_priority`, default 15 = `THREAD_PRIORITY_TIME_CRITICAL` ).
Every worker then sits in a pause-spin loop or a `WaitOnAddress` sleep between dispatches, which keeps
dispatch latency minimal, but also means the pool competes for CPU even when it has no work.
`v7_set_worker_priority` re-prioritizes the live workers at any time:

```c
void v7_set_worker_priority(int priority);
```

- `priority` is a plain Win32 `SetThreadPriority` constant: -15
  (`THREAD_PRIORITY_IDLE`), -2 lowest, -1 below normal, 0 normal, 1 above
  normal, 2 highest, 15 time critical. No range check is layered on top --
  Win32 validates the value, and a failed call FATALs loudly instead of being
  swallowed.
- The call applies to every live worker; the calling thread keeps its own
  priority. Last call wins; call again with 15 to restore the INI default.
- It is purely scheduler-level: workers keep spinning or sleeping exactly as
  before, only with less (or more) CPU share. Greedy inference output is
  byte-identical regardless of priority, only wall-clock speed changes.
- Callable at any time, even before the pool exists: while the worker pool
  has not been created yet (see the lazy creation above), the requested
  priority is parked and applied automatically right after pool creation,
  so a host can set it directly after `v7_init` and still cover the very
  first dispatch.
- The function is deliberately NOT gated by the CPUID engine gate or the
  FATAL landing zones: it emits no CPU-feature instructions and touches no
  engine state that can fail, so there is nothing to gate (same class as
  `v7_get_internal_inference_ctx`).

The same override is available with zero code changes through the `v7_override_prio` environment
variable. When the worker pool is created (lazily, inside `initialize_once` ), the engine reads the
variable once; when it is set, its signed decimal value is applied through exactly the same code path as
the public function, BEFORE the first dispatch, so the very first inference already runs at the
overridden priority:

```bat
set v7_override_prio=-15      & rem run v7 at idle priority
myhost.exe                    & rem any host using the DLL
set v7_override_prio=         & rem unset = INI priority stands
```

An unset or empty variable is a no-op. A malformed value (non-numeric, trailing garbage, oversize) is a
terminal error with a named message, a configured override is a contract, and silently ignoring it would
lie about what the process is doing. verify.bat sets the variable to -15, so its inference smoke tests
run at idle priority and do not starve other processes on the machine; `inference_example.exe -prio n`
exposes the same control through the CLI for single runs.

#### Matmul Kernels, dispatched

- `v7d_q8` - Q8xQ8 matmul
- `v7d_f32` - F32xQ8 matmul
- `v7d_fused_f32` - Fused F32 matmul + RMSNorm + RoPE
- `v7d_q8_fused` - Fused Q8 quantize + matmul
- `v7d_swiglu` - SwiGLU F32xQ8
- `v7d_swiglu_q8` - SwiGLU Q8xQ8
- `v7d_vit_linear` - Dispatched ViT linear matmul
- `matrix_multiplication` - Legacy matmul entry point
- `matrix_multiplication_mode` - Matmul with mode dispatch

#### Attention (dispatched)

- `v7d_attention` - Multi-head attention for a single token
- `v7d_attention_prefill` - Multi-head attention for prefill

#### Utility Kernels

- `v7d_memcpy` - Dispatched memcpy
- `v7d_quantize` - FP32 to INT8 quantize with GS=64
- `v7s_rmsnorm` - RMSNorm
- `v7s_add_inplace` - Element-wise add-inplace
- `v7s_fused_add_rmsnorm` - Fused add + RMSNorm
- `v7s_rmsnorm_rope_heads` - RMSNorm + RoPE per heads
- `v7s_rmsnorm_quantize` - Fused RMSNorm + FP32?INT8 quantize. Uses one global scale and is never chunk-dispatched.
- `v7s_sample` - AVX2 softmax + AES-NI RNG + top-p sampling

#### Process Execution and File I/O

- `v7_execute` - Spawn a child process, capture its stdout, return as C string. Used by SectorLogo `CallLibrary [SELF]`.
- `v7_execute_fmt` - Spawn a child process with a `v7sprintf`-formatted argument line, taking typed varargs from SectorLogo `DefineCommand`.
- `v7s_readfile` - Read a file into the shared capture buffer, for sidecar scripts
- `v7s_writefile` - Write or append a string to a file, for sidecar scripts

#### String Formatting

- `v7sprintf` - Bounded sprintf: `int v7sprintf(char *buffer, int buf_size, const char *format,...)`. Supports `%s` for a string, `%d` for int32, `%c` for a char, `%f` for float32 with a fixed 6 fractional digits and trailing zeros trimmed, and `%%` for a literal percent. It always null-terminates, never writes past `buffer + buf_size`, and has no CRT dependency.

#### Inference Lifecycle, internal tier; the public v7_* tier above is the normal surface

- `V7_init` - Open the MicroDB database. Legacy internal name; the public `v7_init` wraps the DB open plus engine state.
- `v7s_get_user_prompt` - Get the current-turn user prompt string, for RAG query augmentation
- `v7s_enable_thinking` - Toggle thinking or base template variants, 1 for thinking and 0 for base. It re-resolves the effective template pointers for turn-1, system+user and multi at the next render, and FATALs if called before `v7_initialize_inference`. The public `v7_enable_thinking` wraps it.
- `v7s_get_generation_statistic` - Return `uint64_t*` to the flat 10-slot generation statistics block
- `v7_engine_is_dead` - 1 after an internal error stopped a gated public call, 0 while running
- `v7_get_death_reason` - `""` when no fatal error is recorded; the printed `[FATAL]...` text after a fatal error (stable until the next `v7_free_inference` or unload)
- `save_kv_cache` - `int save_kv_cache(const char* filename)`: persist occupied FP16 key/value rows and session position in an LZ77-compressed file with a 48-byte header. Saves `pos` rows per layer for a linear cache, or `min(pos, S + W)` rows for a streaming cache. Returns 1 on save; returns 0 for `pos == 0` or file-creation failure.
- `load_kv_cache` - `int load_kv_cache(const char* filename)`: restore saved cache rows, absolute position, M-RoPE counters and streaming validity state. Requires compatible storage format, cache row geometry, and StreamingLLM mode/parameters. Returns 1 on success and 0 on reported file/format/geometry failure. The file does not contain a model identity: use the same model for save and load. Both functions are gated public entries, callable from SectorLogo through `CallLibrary` with a `"pointer"` argument type.
- `v7_set_streaming` - `int v7_set_streaming(int enable, int num_sinks, int window_size)`: StreamingLLM fixed-size circular KV cache. `enable=1` starts streaming with `num_sinks` attention-sink rows and `window_size` rolling rows; the fill grade (the occupancy where window displacement starts) is `num_sinks + window_size`, capped by the model's `seq_len`. Cache slot of a wrapped token is `num_sinks + ((pos - num_sinks) mod window_size)`; keys rotate once at their slot, queries delta-rotate per segment. Tokens evicted from the ring are delivered as text via the microdb keys `stream_remove_text` / `stream_remove_meta` and the SectorLogo script under the DB key `on_stream_evict` (if the hook is enabled). Returns 0 on success, -1 on bad args/overflow/capacity; FATALs pre-init or on a mid-stream re-map (`pos > 0`). `enable=0` disables while `pos <= C` and stores only the flag; a wrapped session (`pos > C`) refuses with -1 (disabling would read slot-ordered rows as position-ordered). The SectorLogo REWRITE action works under streaming: rewrites at `pos <= C` follow the pre-wrap diff/rollback path, wrapped rewrites use available live token IDs to restore the required resident-window rows and then evaluate the changed suffix (rewriting a wrapped KV-file restore without live token history is rejected); the window displacement warm latch is temporarily protected during the wrapped re-forward/rebuild and is restored on every exit, and the pending evicted batch flushes before the rewrite mutates anything. `v7_set_streaming_fill(lib, enable, num_sinks, fill_grade)` is the v7_lib.h convenience wrapper.

  When streaming mode is enabled, the engine enforces the following restrictions on session positioning and cache reuse:

  - **The rewrite procedure depends on whether the streaming window has wrapped.** At `pos <= C`, `SetStream` uses the normal token-boundary rollback and re-forward path; any real window displacements produced by a replacement that crosses `C` are still logged. At `pos > C`, `chat_rewrite.inc` uses live `out_tokens` IDs to re-forward required earlier window rows and the changed suffix; it protects and restores the displacement warm latch and adjusts the valid-row tracker. If the session came from a KV-file restore and has no live position-token history, a wrapped rewrite is rejected. In all cases, pending displacements flush *before* the rewrite, so delivered absolute positions do not cross a change in timeline.
  - **Window Displacement delivery contract.** Batches flush every 256 displaced tokens (bounded callback latency) at three chat-owned sites: prompt ingest, per generated token, turn end. `stream_remove_text` / `stream_remove_meta` are temp keys (persist=0), the Logo side persists what it keeps. With no Logo host attached the batch is dropped by design. `first=` in the meta is the absolute position of the batch's first evicted token; one log entry is one consecutive position; chunks of one batch over 64 KB split with `first=` advancing by the already-delivered count.
  - **Sessions and flush ownership.** Window Displacements are drained by `v7s_chat`-driven sessions only. Raw `forward_f32` / `forward_q8` loops receive no delivery; an undrained loop with streaming enabled FATALs at the pending-log cap instead of corrupting state. `v7s_generate` swaps streaming off for the nested chat and restores the outer window displacement state afterwards. Direct `prefill_forward_f32` / `prefill_forward_q8` calls never resolve ring slots, under chat this is safe by construction (chat routes any turn that ends above the fill grade to per-token ingestion); a direct prefill call with `start_pos >= C` while streaming is enabled writes position-ordered rows into a slot-ordered session.
  - **`v7_set_position` + streaming.** Rewinding within a wrapped region (`P > T > C`) is supported: the engine reduces the tracked valid rolling rows and extends the warm-up for overwritten slots. Rewinding a wrapped session below or to `C` requires live token IDs to re-forward rows `[S,T)` and restore the linear layout; it fails without that history. An unchanged position is a no-op. Tokens removed by a direct rewind are not delivered as window displacements.

##### KV-cache file format

`src/asm/kv_cache_file.inc` writes a **48-byte** header followed by an LZ77-compressed blob of the
occupied FP16 cache rows. The format is model-specific and is not a general interchange format.

| Offset | Bytes | Field |
| ---: | ---: | --- |
| 0 | 8 | `raw_bytes`: decompressed blob size. |
| 8 | 8 | `comp_bytes`: compressed blob size. |
| 16 | 4 | `pos`: absolute KV position at save time. |
| 20 | 4 | `g_mrope_run`: multimodal position counter (0 in text-only models). |
| 24 | 4 | `g_mrope_delta`: multimodal position adjustment. |
| 28 | 4 | `KV16`: format marker, little-endian `0x3631564B`. |
| 32 | 4 | StreamingLLM enabled at save time. |
| 36 | 4 | Sink count `S` at save time. |
| 40 | 4 | Window size `W` at save time. |
| 44 | 4 | Valid window-row count at save time, including any rewind adjustment. |

For each layer, the serializer gathers key rows, followed by value rows. Each FP16 value consumes
**2 bytes**, so the uncompressed payload contains `2 * n_layers * rows * kv_dim * 2` bytes, where
`kv_dim = n_kv_heads * head_dim`, and `rows` is `pos` for a normal cache or
`min(pos, S + W)` for streaming mode. Wrapped caches retain the absolute position in the header but
store the occupied ring slots, not the entire earlier history.

The loader rejects incomplete headers, a format marker other than `KV16`, non-integral row geometry,
rows beyond the configured cache, and incompatible streaming geometry before scattering rows. In
particular, a wrapped streaming save cannot be loaded as a linear cache, and streaming saves must
match the active `S`/`W` configuration. An old 28- or 32-byte-header file is not compatible: save it
again with the current runtime. The header does **not** include a model ID, checkpoint hash, or a
complete integrity checksum; a same-shape wrong-model file is not automatically recognized.

Load also restores the saved streaming valid-row count and relevant tracking state. Reusing a saved
wrapped cache for inference does not reconstruct the complete token history; **a subsequent wrapped
`SetStream` rewrite requires live token history and fails if that history is unavailable**. Inspect
`src/asm/chat_rewrite.inc` (`g_stream_otk_live`) before relying on wrapped cache rewriting.

#### Vision for VLMs - Image Embedding Upload

- `v7s_set_vision_embeddings` - Upload projected vision embeddings and DeepStack feature tensors
- `v7s_vit_encode_callback` - Pure-ASM vision tower encoder, compute only
- `v7s_vit_forward` - Pure-ASM ViT forward pass: patch_embed, the configured number of transformer blocks, and the configured merger or projector stage

#### Vision Dispatch Tuning

- `vit_max_chunks` - BSS global: max chunks for ViT linear dispatch
- `vit_min_chunk_cols` - BSS global: min chunk columns for ViT linear dispatch

#### M-RoPE (Multimodal Rotary Position Embeddings)

- `v7s_mrope_prepare` - Prepare M-RoPE position deltas from token sequence
- `v7s_mrope_restore` - Restore standard RoPE after M-RoPE dispatch

#### Tokenizer

- `encode` - BPE tokenizer encode

#### Forward Pass

- `forward_f32` - Full forward pass with F32 activations
- `forward_q8` - Full forward pass with Q8 activations
- `prefill_forward_f32` - Prefill forward pass with F32
- `prefill_forward_q8` - Prefill forward pass with Q8

#### Fused-Kernel Support

- `v7s_rope_init` - Build RoPE cos/sin tables
- `v7s_set_fused_rope` - Set fused rope pointer for next dispatch

#### Runtime Control

- `v7s_set_spin` - Declare the inference activity phase (1 = workers pause-spin and stay fork-receptive, 0 = workers return to the OS wait). Chat completion calls `v7s_stop_spin` internally.
- `v7s_set_caller_hmodule` - Set the caller's HMODULE used for `CallLibrary [HOST]` resolution
- `v7s_shutdown` - Shut down worker threads
- `v7s_get_num_workers` - Get the number of worker threads
- `get_config_was_fixed` - u64 bitmask of config values the engine had to degenerate, where bit0 is the taper CAP falling back to uniform fill and bit1 is num_chunks being clamped to its maximum

#### In-Process Cycle Tuner

The engine tunes its own matrix-multiplication dispatch configuration with an
in-process cycle tuner; it is always active in every program that runs
inference and has no enable/disable switch. Hosts can observe it through:

- `v7_get_best_fitness` - Best measured fitness of the active shape
- `v7_get_pool_fitness` - Population fitness statistics of the active shape
- `v7_get_active_shape_count` - Number of shapes the tuner currently cycles

Accepted configurations persist in `v7_tune_param_<n>_<d>` files beside the
executable; the journal writes to `v7_tuning_journal`. The tuner's design,
knobs and measurement protocol are documented in
[tuning.md](tuning.md#dispatch-and-tuning).

#### SectorLogo Sidecar Hook

- `v7s_sidecar_refresh` - Internal, non-exported chat-entry routine that checks the `logo:sidecar` key and refreshes the parsed callback when the stored script changes.
- `v7s_sectorlogo_on_token` - Per-token hook dispatch, called internally by chat
- `v7s_sectorlogo_interp` - Standalone script parse and eval, returning the last OUTPUT value
- `v7s_sectorlogo_eval_script` - Evaluate the script stored under a MicroDB key, whose name is the argument

See [sectorlogo_manual.md](sectorlogo.md#sectorlogo-language-and-integration) for the full language and
FFI reference.

#### MicroDB Hash Table

- `g_ht` - Raw hash table pointer
- `shape_blocks` - Per-shape dispatch config blocks, 256 blocks of 512 bytes
- `debug_param_set_0` - Snapshot of active dispatch parameter set
- `ht_init` - Initialize hash table
- `ht_destroy` - Destroy hash table
- `ht_add` - Add/update entry
- `ht_find` - Find entry by key
- `db_save` - Save to a file compressed with the lz4 like lz77 codec
- `db_load` - Load from file
- `db_set_readonly` - Lock database to read-only

#### Compression

- `mdb_compress` - MicroDB compress with 64-bit sizes
- `mdb_decompress` - MicroDB decompress with 64-bit sizes

#### Prefetch Tuning (SMC)

- `init_prefetch_backup` - Save the original prefetch bytes for restore
- `set_prefetch_distance` - Set the prefetch distance for a kernel slot
- `set_prefetch_type` - Set the prefetch instruction type for a kernel slot

#### Diagnostics

- `dump_token_stream` - Dump token IDs to stderr, a diagnostic
- `enable_token_dump` - BSS global: set to 1 to enable token stream dumping
- `enable_dump` - BSS global: set to 1 to enable the InferenceContext diagnostic dump

#### File-Backed Memory (internal, not exported)

`fbm_alloc` / `fbm_free` are internal labels inside the DLL (the file-backed KV-cache allocator used by
the generate path and its validator). They are deliberately NOT in the export table; hosts reach the
same functionality through the documented `v7_generate` path.

---

### Data Structures

#### `v7_q8_tensor`

```c
typedef struct { int8_t* q; float* s; } v7_q8_tensor;
```

Quantized int8 values (`q`) with per-group float scales (`s`), group size = 64.

#### `v7_f32_tensor`

```c
typedef struct { float* data; float* scales; } v7_f32_tensor;
```

Float32 activation buffer with optional scales, NULL when not quantized.

#### `v7_config`

```c
typedef struct {
    int reserved0;       // always 0
    int reserved1;       // always 0
    int dim;             // embedding dimension
    int hidden_dim;      // FFN hidden dimension
    int n_layers;        // number of layers
    int n_heads;         // number of attention heads
    int n_kv_heads;      // number of KV heads (for GQA)
    int vocab_size;      // vocabulary size
    int seq_len;         // max sequence length
    int head_dim;        // dimension per head (= dim / n_heads)
    int shared_classifier;  // 1 = tied embeddings (classifier = token_embedding)
    int group_size;      // quantization group size (64)
    int has_qk_norm;     // 1 = model has QK-norm (Qwen3), 0 = no QK-norm
    int rope_base;       // RoPE base frequency (e.g. 1000000 for Qwen3, 10000 for Phi-3)
    int reserved2;       // always 0
    int reserved3;       // always 0
} v7_config;
```

#### `v7_run_state`

```c
typedef struct {
    float* x;               // input activations (dim)
    float* xb;              // residual/rmsnorm buffer (dim)
    float* hb;              // hidden buffer (hidden_dim)
    float* hb2;             // second hidden buffer (hidden_dim)
    v7_q8_tensor xq;        // quantized x (q + scale)
    v7_q8_tensor hq;        // quantized hb (q + scale)
    float* q;               // query buffer (all_heads_dim)
    float* k;               // key buffer (all_kv_heads_dim)
    float* v;               // value buffer (all_kv_heads_dim)
    float* att;              // attention scores (n_heads * seq_len)
    float* logits;          // output logits (vocab_size)
    float* key_cache;       // KV cache for keys (n_layers * seq_len * all_kv_heads_dim)
    float* value_cache;     // KV cache for values (n_layers * seq_len * all_kv_heads_dim)
    float* x_storage;       // per-position activation storage (seq_len * dim)
    float* q_storage;       // per-position Q storage (seq_len * all_heads_dim)
    float* att_out_storage; // per-position attention output (seq_len * all_heads_dim)
    void* key_cache_hfile;  // Windows HANDLE for KV cache backing file
    void* key_cache_hmap;   // Windows HANDLE for KV cache file mapping
    void* value_cache_hfile;
    void* value_cache_hmap;
} v7_run_state;
```

Passed to attention and forward functions.

#### `v7_probindex`

```c
typedef struct { float prob; int index; } v7_probindex;
```

Used by the sampler for top-p filtering. A pre-allocated array of `vocab_size` entries.

#### `inference_context`

```c
typedef struct inference_context {
    void *transformer;
    void *tokenizer;
    void *sampler;
    void *prompt_tokens;
    void *out_tokens;
    int   max_prompt;
    int   max_gen;
    int   pos;         // current KV-cache position
    int   quantize;    // 0=F32, 1=Q8
} inference_context;
```

### Requirements

- **Platform:** The library requires x86-64 Windows and the full `cpu_gate.inc` hardware baseline: AVX2, FMA, F16C, BMI1, BMI2, SSE4.1, SSE4.2, AES-NI, RDRAND and RDTSCP, with XCR0 SSE/YMM state enabled. Having AVX2 alone is insufficient.
- **Dependencies:** The link step in `build.bat` uses `kernel32.lib`, `user32.lib`, and `synchronization.lib`.
- **Database:** The model data is loaded from the MicroDB file `lz4ht.db`, which uses the project's LZ4-like LZ77 compression.

## Typed decision interfaces

A language model can be useful even when an application does not need it to compose a sentence. For a
routing or classification task, the application may already know the permitted answers and only need to
compare their relative likelihoods. V7's typed-decision interface supports that case by evaluating a
bounded set of specified next-token alternatives with the loaded model.

The caller provides the information to evaluate and the permitted answer tokens. V7 processes the
formatted input and returns the probabilities associated with those candidates instead of continuing
into an unrestricted text response. This does not require additional model training, and the interface
does not depend on a conversational chat template. Its purpose is to compare the alternatives supplied
by the application, not to discover arbitrary new labels.

The returned values are normalized over the specified candidates. They therefore describe the model's
relative preference among those alternatives, not a calibrated probability of a real-world event or the
probability of the same token over the entire vocabulary. An application that turns those values into a
decision threshold should validate the threshold using representative data for its task.

### Question kinds

| kind | question | options |
| --- | --- | --- |
| `noul` | yes/no | `false`, `true` |
| `score` | 0-5 rating | `0`, `1`, `2`, `3`, `4`, `5` |
| `choice` | pick one | 2-16 caller strings, labelled `A)`... `P)` |

### The prompt format

```text
[kind] choice
[state] Account entry: a card was charged twice for one coffee.
[question] Which team should handle this?
[options]
A) billing
B) shipping
C) support

[decision]:
```

The examples below describe how to use typed decisions. They reflect the behavior observed in the
recorded tests in `examples/decision_Results.md`; applications should validate thresholds using their
own data:

- Include all information required for the decision in `state`, and specify a single question.
- When several alternatives are available, present them as the options of one choice question. Separate yes/no queries require an additional model evaluation for every alternative and performed less reliably in the reported tests.
- Every label must be a single token (`false`, `true`, `0`-`5`, `A`-`P`
  all are). The tool rejects labels that require more than one token.
- Changing the option order can change the reported probabilities by several percentage points. If two
  options compete, ask both orders and average.
- A yes/no token classification is not a reliable factual verification method by itself. For example, when asked whether a green sky is blue, the model assigned a
  probability of 0.94 to `true`; it completed the question without reliably
  checking the supplied state. Use yes/no for questions whose
  yes-answer the model can actually weigh, not for verification.

### The API

```c
void v7_get_token_probs(const int *token_ids, int n, float *out_probs);
```

Call this after a forward whose last position is the one you care about. It reads that position's logits
for the given token ids and writes one probability per id; the values sum to 1.0.

- `n` must be 1..256 and every id inside the vocabulary. Anything else
  stops the engine with a named error, which the host reads through
  `v7_get_death_reason()`.
- A typical approach is to generate with a small output buffer:
  `v7_generate(decision_prompt, out, tiny_buffer, 1)`. Generation stops after the buffer fills with one sampled token, leaving the
  logits for the decision position available.
- Decide greedily: `v7_set_sampler_temperature(0.0f)`, presence penalty 0,
  `v7_set_position(0)` for a fresh context. Then the same prompt gives the
  same numbers every run.
- Full precision and 8-bit forwards give slightly different values. Calibrate thresholds using the inference precision selected for deployment.

### v7_decide_example.exe

```
v7_decide_example.exe <dll_path> <db_path> <kind> <state> <question> "opt1;opt2;..." [-q8 0|1]
```

| argument | meaning |
| --- | --- |
| `dll_path` | path to `v7_threading.dll` (required, no default) |
| `db_path` | path to the model's `lz4ht.db` (required, no default) |
| `kind` | `noul`, `score`, or `choice` (goes into the prompt verbatim) |
| `state` | everything the answer depends on |
| `question` | one question about the state |
| `"opt1;opt2;..."` | 2-16 semicolon-separated option labels |
| `-q8 0\|1` | forward path override (default: engine default) |

The example tool constructs the decision prompt, verifies that each label consists of one token,
evaluates the next-token alternatives, and writes one JSON result. The following output comes from two
recorded runs:

```
> bin\v7_decide_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db noul "The sky is blue." "Is the sky blue?" "false;true" -q8 0
{"options":["false","true"],"probabilities":[0.0158,0.9842],"choice":"true"}

> bin\v7_decide_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db choice "Account entry: a card was charged twice for one coffee." "Which team should handle this?" "billing;shipping;support"
{"options":["billing","shipping","support"],"probabilities":[0.6312,0.3078,0.0610],"choice":"billing"}
```

For invalid arguments, labels that require more than one token, or a terminal engine state, the tool
reports the condition and exits with a nonzero status.

### spam_classifier.exe

```
spam_classifier.exe <dll_path> <db_path> <mail.txt> [-q8 0|1]
```

Reads an email from a text file, asks "which kind of email is this?" with the labels `ham` and `spam`,
and prints the probabilities and the decision:

```
> bin\spam_classifier.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db examples\spam_classifier\mails\spam2_phishing.txt
{"file":"examples\spam_classifier\mails\spam2_phishing.txt","p_ham":0.1021,"p_spam":0.8979,"decision":"spam"}
```

Six test mails ship in `examples/spam_classifier/mails/` (three spam, three normal; all six classified
correctly). The full table with the close calls is in `examples/decision_Results.md`.

### The server endpoint

The optional `v7_server.exe` host makes the same decision interface available over HTTP:

```
POST /v1/decide
{"kind":"noul","state":"The sky is blue.","question":"Is the sky blue?","options":["false","true"]}

{"options":["false","true"],"probabilities":[0.0158,0.9842],"choice_index":1,"choice":"true"}
```

`kind` is `noul`, `score` or `choice`; every option label must encode to exactly one token (the error
names the label that does not). The response adds `choice_index` alongside `choice`.

The endpoint is session-transparent: it runs on its own private KV cache (`v7_generate`) and never
disturbs the server's chat session, so a service can classify requests while a conversation is live. The
regression test is verify.bat Phase 6's decide-session block (plant a fact, decide between turns, recall
it, "PASS: /v1/decide session transparency"). Errors: missing/short/oversize fields and multi-token
labels return 400 with a named reason; non-POST requests return 405.

### Interpretation Limits

- The numbers are the stock model's next-token guesses, not a trained
  classifier. Routing and filtering with a validated threshold work well;
  hard label sets do not.
- Yes/no prompts do not verify facts on a stock model (see above).
- 16 single-letter labels (`A`-`P`) are what the format supports natively;
  wider label sets are untested.
- There is no learned bias or per-kind temperature yet. Both would live in
  the model database as data when they exist, the engine code stays the
  same.

## HTTP integration

The optional HTTP server provides a request-based interface to the native library. It is useful when an
application cannot load the DLL directly or when separate processes need to submit requests through a
local endpoint. The server does not introduce independent model contexts: requests share the underlying
inference state and follow the documented serialization rules. The following sections describe its
endpoints, request formats, session handling, and operational limits.

The HTTP server is an optional example host. It exposes generation and selected control operations
through request endpoints, while direct C applications can use the DLL without an HTTP layer.

> **The server is a development tool and local test application.**
> It is not a hardened Internet-facing service. `GET /sidecar?<path>` loads
> a SectorLogo script from the filesystem, and a script can start external
> programs through `DefineCommand` or `CallLibrary`. Keep the sidecar endpoint
> local or restrict it before exposing the server to a network.

The `v7_server.exe` application provides an OpenAI-compatible HTTP interface and a browser-based chat
interface for the V7 inference library. It is a single-threaded Win32 process that uses `ws2_32`,
`gdiplus`, and `ole32` and maintains one inference context throughout its execution.

---

### Server architecture

```
start_server.bat (own visible window)
  `- server.bat [logfile]      restart loop: exit 42 = stop, else restart 2s
       `- v7_server.exe <v7_threading.dll> <lz4ht.db> [port] [model_id]
            |
            |- v7_load(dll) -> v7_init(db) -> v7_initialize_inference()
            |    Q8 forward; v7_enable_thinking(1)
            |
            |- Winsock listen(port, backlog 16), select-loop 1s tick
            |
            `- per connection (serial, single thread):
                 read_http_request (chunked, CL clamp, 30s deadline)
                 route -> handler -> v7_chat / v7_generate / v7_chat_image
                 printf full TURN block (request + response) AFTER inference
                 send_response -> close
```

Because the server shares a single inference context and chat-session cache, it processes incoming
requests serially. A generation in progress blocks processing of subsequent connections (accept backlog
absorbs bursts). The 1-second `select()` tick keeps the loop responsive to `/quit` and Ctrl+C.

#### 1.1 Session / KV-cache semantics

`v7_chat` takes a 5th `reset` parameter. The server tracks `g_reset_next`:

- Server start and every `/v1/chat/reset` (or web "New chat") set
  `g_reset_next = 1`.
- The NEXT chat turn consumes the flag (`session_take_reset()`) and calls
  with `reset=1`, fresh session, KV cache cleared.
- Every later turn runs `reset=0`, the KV cache PERSISTS across turns.
  **Clients submit only the new user message.** The server retains the preceding conversation in the inference session and its KV cache.

`/v1/completions` never touches the session: `v7_generate` saves and restores the position internally
(independent KV per call). Vision turns continue the SAME session via `v7_chat_image(..., reset)`.

#### 1.1.1 Missing think-tag rollback (thinking cut by the limit)

Thinking mode is enabled during normal server operation. With a thinking template, the prompt opens the
reasoning section and a completed reasoning segment contains the closing `</think>` marker. If the
response has no closing `</think>` marker because generation reached the token or output-buffer limit,
the response may contain an incomplete reasoning section. The handler then:

1. Reads the pre-turn KV position (`v7_get_position` BEFORE the call).
2. Rewinds the cache to it (`v7_set_position`).
3. Disables thinking (`v7_enable_thinking(0)`), re-runs the SAME user
   message with `reset=0` (re-prefill at the rewound position), and
   restores thinking (`v7_enable_thinking(1)`) so later turns keep the
   engine default.

The subsequent request produces a response without the reasoning template. A sidecar state marker
retained from the first attempt, such as a memory-RAG completion flag, may prevent a second insertion
during this retry.

#### 1.2 The API boundary, enforced

The server uses the public API tier, the `v7_*` exports. It does not write sampler state, transformer
internals, tokenizer tables, or RunState. The server parses and records the `temperature` and `top_p`
request fields but does not apply them to the sampler. The sampler belongs to the DLL; hosts can use the
documented `v7_set_sampler_*` setters when they need to control it. The documented sidecar loading uses
`v7_db_add("logo:sidecar", ...)`. The runtime refreshes the script automatically from the MicroDB key at
the beginning of the next chat turn.

StreamingLLM is a per-boot opt-in via CLI flags (the server has no INI of its own, the CLI is the
configuration surface, and `server.bat` maps the `V7_SERVER_STREAM` / `V7_SERVER_SINKS` /
`V7_SERVER_WINDOW` env vars onto the same flags for launcher convenience):

    v7_server.exe <dll> <db> [port] [model] [-stream 0|1] [-sinks N] [-window N]

`-stream 1` runs EVERY chat session on the ring KV cache (fill grade C = S + W, defaults S=4 / W=996);
displaced text is delivered to the loaded SectorLogo sidecar through `on_stream_evict`. The refusal
path is visible, not fatal: `-stream 1` on an M-RoPE-capable model prints the refusal reason and the
server keeps running the legacy KV path. The `SetStream` rewrite hook works under streaming: linear sessions use the ordinary suffix
rollback, while wrapped sessions restore the required window rows using live token IDs and re-forward
the replacement suffix. A wrapped KV-file restore without those IDs cannot be rewritten.

#### 1.3 Generation budget

The chat and completions handlers honor the client's `max_tokens` when it is positive, and fall back to
the server-wide default (`-maxgen N`) in other cases. The default is `0` = unlimited: the turn runs until EOS
or until the output buffer is full, per the engine's no-artificial-limits contract. Vision turns always
run at the server default; a request-level `max_tokens` does not reach the vision path (limiting vision
answers mid-sentence causes the same problems as limiting chat). The typed-decision endpoint generates
exactly one token by design.

---

### HTTP endpoints

| Method | Path | Handler | Notes |
| --- | --- | --- | --- |
| GET | `/` | `handle_index` | browser chat UI (server-side history, form-submit) |
| POST | `/chat/submit` | `handle_chat_submit` | form field `q=` (URL-decoded), inference, 303 -> `/` |
| GET | `/chat/reset` | `handle_chat_reset_page` | clears history + next-turn reset, 303 -> `/` |
| GET | `/v1/health` | `handle_health` | `{"status":"ok"}` |
| GET | `/v1/models` | `handle_models` | canned list with the configured model_id |
| POST | `/v1/chat/completions` | `handle_chat_completions` | OpenAI chat API; last system + last user message used |
| POST | `/v1/completions` | `handle_completions` | OpenAI completions API; `v7s_generate` (independent KV) |
| POST | `/v1/vision/chat` | `handle_vision_chat` | `{prompt, image_b64}` on the SAME chat session |
| POST | `/v1/chat/reset` | `handle_chat_reset` | next turn starts fresh |
| GET/POST | `/quit?code=N` | `handle_quit` | exit code N (default 42 = clean; bat loop stops only on 42), clamped 0-255 |
| GET/POST | `/sidecar?<path>` | `handle_sidecar` | load a SectorLogo sidecar at runtime |
| OPTIONS | `*` | `handle_options` | CORS preflight 204 |

All responses carry `Access-Control-Allow-Origin: *` and `Connection: close` (one request per
connection).

#### 2.1 `/sidecar?<path>`, runtime sidecar loading

The query string specifies a file path directly, without a `key=` parameter or URL
decoding. The path must use forward slashes and must not contain spaces. The file is read (1 MB cap), stored into the microdb under `logo:sidecar` via `ht_add`, and the
hook is enabled via `v7_db_add`. The runtime checks the script contents at each chat entry, and the
current server response distinguishes two historical states:

- If no token callback has executed yet, the endpoint reports `"active"`.
- After a callback has executed, the endpoint reports `"staged"`. This status reflects the server's `hook_executed` flag. The current inference engine checks the MicroDB key at the next chat entry and reparses the script if it has changed, so a server restart is not required solely to activate a replacement script.

Convenience wrapper: `server_set_sidecar.bat <file> [port]`.

#### 2.2 Vision turns

`image_b64` (base64 PNG/JPEG/BMP/GIF/TIFF, whatever GDI+ knows) -> `b64_decode` -> GDI+
`Bitmap::FromStream` -> flatten onto a white 24bpp canvas -> `LockBits` -> channel-swapped RGB ->
`v7_chat_image(prompt, NULL, rgb, w, h, out, cap, reset)`. The image turn continues the chat session
(same `session_take_reset()` logic). Images are decoded and delivered at their NATIVE size (no
downscale); the decode-amplification guard is a 64-megapixel metadata cap checked before the EXIF
`RotateFlip`, and EXIF orientation values 2-8 are normalized so phone photos arrive upright.

GDI+ lifetime subtleties (each guards a real crash class):

- **The IStream must outlive the Bitmap**, GDI+ decodes LAZILY (during
  `LockBits`), reading from the stream. The stream (created with
  `fDeleteOnRelease=TRUE` over an HGLOBAL copy of the bytes) is released
  only AFTER `delete bmp`.
- **COM must be initialized** on this thread before GDI+
  (`CoInitializeEx(APARTMENTTHREADED)` at startup, `CoUninitialize` at
  exit; GDI+Startup/Shutdown bracket the whole process).

---

### Request handling

`read_http_request`:

- Chunked reads (4 KB) into a dynamically grown request buffer framed by
  `MAX_HEADER_BUF` (1 MB) + `MAX_BODY_BUF` (256 MB).
- `Content-Length` clamped to `MAX_BODY_BUF` (256 MB; oversized prompts are
  rejected by the model context first, this only frames the transport).
- **Hard 30-second whole-request wall-clock deadline** (`GetTickCount64`
  based, extended for very large bodies at a 64 KB/s floor) plus
  **10-second per-call SO_RCVTIMEO/SO_SNDTIMEO** on every accepted
  socket, a client that connects and stalls (or drip-feeds bytes) can
  never hang the single thread.
- Header scan stops after the first `\r\n\r\n` (no O(n^2) rescans of a
  growing body).

Console hardening:

- **QuickEdit disabled** at startup, a click inside a cmd console
  suspends all console writes process-wide; our full-response printfs
  would block forever at 0% CPU (the observed unkillable hang).
- **The Ctrl+C handler only sets a flag**, console handlers run on a
  separate OS thread; printf there deadlocks against the main thread's
  stdout lock.

Output discipline: the full request + full response are printed as ONE block AFTER the inference call
returns, turn blocks never interleave with the DLL's own generation-time output.

---

### Verified server behavior

The current implementation carries these properties; the historical bug-hunt notes that used to live in
this section described an older fixed-size-store server and are superseded by the source:

- JSON parser: all string writes bounded (`dst_size - 1` guards,
  `\uXXXX` with surrogate pairs, `\u0000` dropped instead of emitting a
  truncating NUL); skip-value walkers handle nested quotes/braces;
  request structs own malloc'd, exact-sized strings with an idempotent
  `*_request_free` cleanup.
- `find_field` matches only at field boundaries (no `q=` inside another
  value); form URL-decoding handles `+` and `%HH`.
- `b64_decode` bounded, whitespace-tolerant, padding-terminated.
- Chat history entries store the full turn text in malloc'd buffers
  (never truncated), so the response and page-render paths cannot
  overflow a per-message store.
- Sidecar path handling: query capped at 2048, `strncpy` into a
  zero-initialized 1024 buffer, error message buffers sized for the max
  path.
- GDI+ error paths free every resource in order; the pixel cap bounds
  decode amplification before any large allocation.
- Exit paths close sockets / destroy context / unload DLL in order;
  exit code propagates `g_exit_code` (0 on Ctrl+C, N on `/quit?code=N`).
- `select()` + 1s tick: `/quit` and Ctrl+C observed within one second.

---

### Server lifecycle and build scripts

| Bat | Purpose |
| --- | --- |
| `server.bat [logfile]` | runs `bin\v7_server.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db 8080` in a restart loop. Exit 42 stops it; any other exit, a crash or a FATAL, restarts it after 2s. The optional logfile appends all output and exists for crash capture only, after which the log must be shown in full. |
| `start_server.bat [logfile]` | starts `server.bat` in its own visible window through `cmd /c start` |
| `stop_server.bat [port]` | `curl /quit?code=42` for a clean stop, after which the loop does not restart |
| `server_set_sidecar.bat <file> [port]` | `GET /sidecar?<path>` against the running server |

Typical session:

```
start_server.bat
curl http://localhost:8080/v1/health
curl http://localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"what is a cat?"}]}'
server_set_sidecar.bat examples\rag\memory\sidecar_memory_rag_v7.logo
stop_server.bat
```

Rules: never start the exe directly from the AI prompt, never background/hide the window, start it as a
standalone tool call, observe the window.

---

### Request limits and buffers

| Constant | Value | Meaning |
| --- | --- | --- |
| `MAX_HEADER_BUF` | 1 MB | request header framing bound (answers 431 first) |
| `MAX_BODY_BUF` | 256 MB | body cap (Content-Length clamped; answers 413 first) |
| `MAX_INFERENCE_OUT` | 8 MB | inference output buffer, far beyond any local context, so the buffer never cuts a turn |
| `RECV_CHUNK` | 4 KB | chunked read size |
| `CHAT_HISTORY_MAX` | 4096 messages | web-UI history (full-text entries, malloc'd per message) |
| `HANDOFF_CAP` | 64 | connections the health thread passes to main |
| image decode cap | 64 M pixels | metadata cap against decode amplification, checked before RotateFlip |
| request deadline | 30 s | whole request, wall-clock (extended for very large bodies) |
| socket timeouts | 10 s | per recv/send call |
| `DEFAULT_PORT` | 8080 | default listen port |
| `DEFAULT_MODEL` | `qwen3-0.6b` | default model_id |

---

### Server troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| unkillable hang, 0% CPU | QuickEdit selection suspending console writes | already disabled at startup; if it recurs check the window for a selection |
| turn blocks forever | client stalled mid-request | 30 s deadline + 10 s socket timeouts reclaim the loop |
| endless restarts after crash | server.bat loop by design | read the window / logfile, fix the crash; `stop_server.bat` only stops on code 42 |
| sidecar reports "staged" | A callback executed previously in this server process | The updated script is reloaded automatically at the next chat entry; the endpoint status does not require a restart. |
| multi-turn answers ignore history | client re-sends full history each turn | send only the new message, because the KV cache is the history on reset=0 turns |
| answers suddenly generic | a `/v1/chat/reset` or "New chat" cleared the session | expected; the next turn re-anchors |
| temp/top_p appear ignored | by design | the sampler is DLL-owned |
| vision 400 "could not be decoded" | not an image, over the 64-megapixel cap, or corrupt base64 | check the format and the dimensions |

---

### Server source files

The server is 14 translation units plus 3 headers in `examples/server/`, built by build.bat step 5/6 in
one `cl` line. Every `.cpp` is a single-purpose module under ~650 lines.

```
examples/server/v7_server_main.cpp          globals, ctrl handler, main()
examples/server/v7_server_common.h          shared constants, state, declarations
examples/server/v7_server_json_types.h      the four request structs
examples/server/v7_server_json.h            hand-rolled JSON parse/build + base64
examples/server/v7_server_json.cpp          sole TU that owns the JSON definitions
examples/server/v7_server_util.cpp          log sanitize, UTF-8 trim/check, query parsing
examples/server/v7_server_http.cpp          request read, response send, socket close
examples/server/v7_server_turn.cpp          engine thinking, session reset, run_turn
examples/server/v7_server_image.cpp         GDI+ image decode
examples/server/v7_server_health.cpp        handoff queue, health thread
examples/server/v7_server_api_meta.cpp      /v1/health, /v1/models, /quit, options, reset
examples/server/v7_server_api_chat.cpp      /v1/chat/completions, /v1/completions, /v1/decide
examples/server/v7_server_api_vision.cpp    /v1/vision/chat
examples/server/v7_server_sidecar.cpp       /sidecar runtime load
examples/server/v7_server_web_render.cpp    chat page template, history render
examples/server/v7_server_web_handlers.cpp  form endpoints: /, /chat/submit, /background
examples/server/v7_server_router.cpp        security gates + route ladder (serve_client)
server.bat                          supervised run loop with an optional logfile
start_server.bat                    start in its own window
stop_server.bat                     clean quit via /quit?code=42
server_set_sidecar.bat              runtime sidecar load
src/v7_lib.h                        DLL loader shared with all hosts
```

Related docs: the RAG-on-the-server section of `retrieval.md`.
