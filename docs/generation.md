# Generation Control and Stream Rewriting

V7 allows an application to influence text generation while the model is running. A SectorLogo program
can inspect a pending token, change sampling parameters, request another sample, or replace text that
has already been generated. These operations are part of the inference interface, not postprocessing of
a completed response.

There are two distinct kinds of intervention. Candidate-token operations act before a token is accepted;
stream replacement changes accepted text and updates the transformer state from which generation
continues. This guide explains their timing, the state maintained by the rewrite algorithm, and the
interaction with StreamingLLM. It also describes retrieval workflows that use these operations, while
leaving application-specific ranking and document selection to the retrieval manual.

## Contents

- [Generation-control model](#generation-control-model)
- [Stream replacement and state consistency](#stream-replacement-and-state-consistency)
- [Retrieval-time stream replacement](#retrieval-time-stream-replacement)
- [StreamingLLM](#streamingllm)

---

## Generation-control model

A V7 generation request alternates between two kinds of work. The native inference code computes the
next-token logits and selects a candidate, while an optional SectorLogo program can examine that
candidate before it becomes part of the accepted output. This timing gives an application an opportunity
to respond to the actual text being generated without taking ownership of the sampling loop or the
transformer state.

The boundary between a candidate and accepted text matters throughout the API. `:TokenNumber` contains
the pending token ID, and `:Piece` contains its decoded text. `GetStream`, by contrast, returns text
that has already been accepted. A script that checks for `</think>`, for example, can recognize the
pending closing marker without treating it as part of the previously generated reasoning text. The
callback executes as a SectorLogo program, but the inference-specific variables and commands are
supplied by V7 rather than by the Logo language itself.

### What happens at the token boundary

After the prompt has been processed, the transformer produces logits for the next token. The sampler
selects a candidate and the runtime decodes its text. If a SectorLogo callback is configured, V7 makes
the candidate available and executes the script before committing the token. The callback may simply
return, allowing normal generation to continue, or it may request one of the actions described below.
The native generation loop then carries out the requested action and maintains the corresponding
inference state.

When the candidate is accepted, it is appended to the output stream and the model proceeds to the next
generation step. Calling `Reject` instead asks the sampler for another candidate using the logits
already available; the transformer does not need to repeat its forward pass merely because one candidate
was rejected. The rejected token ID is not automatically excluded from the distribution, however, so the
sampler can select it again. `Halt` finishes the turn before the pending candidate is accepted. A script
may also call `SetTemperature` or `SetNucleus` to change the sampler's subsequent behavior according to
the timing rules of the host interface.

### Changing text that has already been accepted

A replacement is not equivalent to resetting the entire prompt. The engine retains the token-aligned
common prefix of the accepted and replacement streams when its cache entries remain valid, then
evaluates the changed suffix. If a script appends material near the closing thinking marker, the
unaffected preceding tokens can remain cached. If the edit occurs near the beginning, substantially more
work is required. The current rewrite implementation evaluates replacement tokens individually through
the normal `forward_q8` or `forward_f32` path; references to a “prefill” in this chapter describe
rebuilding the KV state, not necessarily a batched prefill kernel.

`SetStream` is different from the candidate-token controls because it operates on the accepted text
rather than on a pending token. The script can read the current text through `GetStream`, use ordinary
SectorLogo functions or external data to construct a replacement, and pass that replacement to
`SetStream`. The host then updates the output and the state from which generation continues. A script
can therefore revise an earlier passage without terminating the current request and starting a new
conversation.

The relationship between text and tokens is not always one-to-one. A token can decode to several bytes,
and even a one-byte edit can affect tokenization at the change boundary. V7 keeps sorted per-token byte offsets
and uses a binary search to locate the first affected token. It retains the earlier KV-cache state where
that state remains valid, evaluates the replacement suffix, and updates the related generation and
sampling bookkeeping. If StreamingLLM has already displaced older attention rows, the implementation may
use a broader reconstruction path instead of an ordinary position rewind. The next chapter describes
these operations and their invariants in detail.

The computational cost depends mainly on where the edit begins and on the active cache configuration.
Replacing a short suffix near the end can reuse most of the existing context; changing text near the
beginning requires more work. This is why callback policies normally choose a specific point in
generation rather than rewriting the entire stream after every token.

### Retrieval during generation

A script can trigger retrieval when a particular candidate appears, but trigger policy is
application-defined. The model must emit the expected marker for that policy to run, and a script that
may encounter the same marker more than once in a turn should maintain an explicit completion flag.
Rewriting does not automatically repair retained text that conflicts with the retrieved passages; the
model's next tokens are merely conditioned on the revised sequence.

A retrieval application can use the prompt, accepted generated text, or both to form a search query. It
can then use native string operations, MicroDB records, BM25 scoring, or an external function to select
material relevant to the current response. When that material should become part of the model's working
context, the script constructs a replacement and calls `SetStream`. Nothing in MicroDB automatically
activates this behavior: the application defines the query, the relevance policy, the replacement text,
and the point at which the callback runs.

For the command syntax and ordinary SectorLogo language rules, see the [SectorLogo
manual](sectorlogo.md#sectorlogo-language-and-integration). The [native API
reference](native_api.md#c-api-reference) describes initialization and the C integration boundary. The
[retrieval manual](retrieval.md) contains example search programs and measured retrieval experiments.

## Stream replacement and state consistency

When a script changes the accepted text, V7 must also update the state used to generate the next token.
The visible string is only one representation of the current output. The runtime also holds token IDs,
per-token start-byte offsets, sampling state, and the KV-cache entries calculated
for the previous token sequence. If any of those representations still describes the earlier text,
generation would continue from an inconsistent context.

The rewrite implementation therefore begins by finding where the replacement first differs from the
accepted stream. It binary-searches the sorted token-start offsets to identify the earliest token whose
representation may have changed. The preceding tokens and their KV-cache rows remain usable when the active cache
configuration permits it; the affected suffix is tokenized and evaluated again. The same operation
updates the output buffer, generation counters, and presence-penalty bookkeeping before returning to the
normal generation loop.

The following reference describes the implementation in `src/asm/token_stream.inc`,
`src/asm/chat_rewrite.inc` (included from `chat.inc`), and `src/asm/sectorlogo_v7prim.inc`.
It covers the actual state arrays and both StreamingLLM rewrite paths.

For research on related context-rewriting mechanisms (including generated text, and how), see
[rag_documentation.md Sec. 32.2](retrieval.md#retrieval-and-rag-implementation-reference). For the
language-side contract of `SetStream`, see the [SectorLogo
manual](sectorlogo.md#sectorlogo-language-and-integration) Sec. 9 and Sec. 10.

---

### Rewrite state and capacities

The per-turn record uses **two token arrays and one text buffer**, not a byte-to-token lookup
array. The actual declarations in `src/asm/sectorlogo_data.inc` are:

| Field | Storage | Purpose |
| --- | --- | --- |
| `sl_stream_str_buf` | 1 MiB | Accepted decoded UTF-8 output, NUL-terminated |
| `sl_stream_tok_history` | 262,144 dwords | Accepted token IDs |
| `sl_stream_tok_offset` | 262,144 dwords | Byte offset of each token's first decoded byte in the text buffer |
| `sl_stream_str_buf2` | 1 MiB | Replacement text passed by `SetStream` |
| `sl_rewrite_tok_buf` | 16,384 dwords | Re-tokenized replacement suffix |
| `sl_stream_base_pos` | dword | KV position at the start of the current generation turn |
| `sl_stream_str_len`, `sl_stream_tok_count`, `sl_stream_str2_len`, `sl_rewrite_count` | dword each | Active lengths |
| `sl_action_code` | dword | Callback action; value 3 selects rewriting |

The runtime does **not** have `sl_stream_tok_len`, `sl_stream_char_to_tok`, or
`sl_stream_rewrite_flag`. Decoded-piece lengths can be inferred from successive start offsets,
and the token owning a differing byte is located by a binary search over
`sl_stream_tok_offset`. Buffer sizes come from `SL_STREAM_BUF_SIZE`, `SL_STREAM_TOK_MAX`, and
`SL_REWRITE_TOK_MAX` in `src/asm/v7_core.inc` and `src/asm/sectorlogo_const.inc`.

`v7s_stream_append_token` stores one token ID and its start offset, copies the decoded bytes into
the primary buffer, and advances the lengths. Reaching the token-count or byte-buffer limit is a
terminal error, not a silent truncation. `SetStream` itself also rejects replacement strings larger
than the 1 MiB buffer instead of clamping them.

### Callback and rewrite execution

1. At chat entry, an optional `logo:sidecar` MicroDB entry selects the SectorLogo callback. After a
   candidate token is sampled and decoded, the callback sees `:Piece` and `:TokenNumber` before the
   token is accepted. `SetStream` copies the requested string into `sl_stream_str_buf2`, records its
   length and sets `sl_action_code = SL_ACTION_REWRITE` (3). There is no separate rewrite-flag global.
2. `src/asm/chat_genloop.inc` routes action 3 to `.chat_hook_rewrite` in `chat_rewrite.inc`.
   Before changing the stream, that handler calls `v7s_stream_remove_flush` to deliver any pending
   StreamingLLM displacement record on its original absolute-position timeline.
3. `v7s_stream_compute_rewrite_diff` compares the old and new text, scanning 8-byte windows with a
   byte-wise tail. A pure append preserves all old tokens. For a middle change or truncation, it uses
   an **upper-bound binary search** over sorted token start offsets, then rewinds the token count and
   byte length to the start of the affected token. The returned suffix pointer starts at the
   **rewound token boundary**, not necessarily at the byte of the first mismatch. This ensures that
   token-boundary changes are re-encoded consistently.
4. If the presence penalty is enabled, the handler clears bitmap bits for discarded token IDs and
   re-marks IDs from the kept prefix. For a pure append it skips both passes, because there are no
   discarded tokens. The bitmap is a set of token IDs, not a count of their occurrences.
5. The handler sets the KV cursor to `sl_stream_base_pos + kept_token_count`, resets the per-turn
   generation counter to the kept count, truncates the caller's output at the same byte boundary,
   and encodes the replacement suffix with `encode` into `sl_rewrite_tok_buf`. The maximum is
   16,384 token slots; reaching the maximum is treated as possible truncation and causes a terminal
   error, rather than proceeding with a partial replacement.
6. For each replacement token, the normal decoder writes the text into the caller's buffer and
   `v7s_stream_append_token` updates the token record. The handler updates the presence bitmap,
   records token IDs in the bounded positional history where available, executes `forward_q8` or
   `forward_f32` to populate the cache, and advances position and generation counters. The output
   capacity and `max_gen` remain enforced. In linear mode, evaluation stops at `seq_len`; in
   streaming mode the attention-slot resolver permits progression beyond `seq_len`.
7. With a nonempty replacement suffix, the final forward pass supplies logits for sampling. With
   an empty suffix, the handler re-forwards the last kept token (or last prompt token, when there is
   no kept generated token) to refresh logits; an empty prompt with no kept token finishes the turn.
   Sampling resumes using the current temperature/top-p settings and presence penalty.

### StreamingLLM rewrite paths

When StreamingLLM is disabled, or the old position is at or below its fill capacity
`C = num_sinks + window_size`, the handler uses the linear rewind and suffix re-forward path.
Streaming remains active when the replacement crosses `C`; those newly displaced rows are real
window displacements and retain their normal delivery behavior.

When streaming **has already wrapped** (`pos_old > C`), a simple cursor rewind is insufficient:
previous ring slots may contain later tokens. The current `chat_rewrite.inc` instead protects the
window displacement tracker, computes the replacement's new end position, and re-forwards the
required earlier token rows to restore the valid window before or alongside the changed suffix.
It uses the active position-indexed `out_tokens` history for this restoration when available. The
tracker is reduced when older position IDs are unavailable, rather than claiming missing rows are
valid; the saved warm latch is restored on exit. The cache is still held in FP16 storage.

**Important limitation:** a wrapped `SetStream` rewrite requires live position-token history.
`chat_rewrite.inc` checks `g_stream_otk_live` and raises a terminal error when that history is
unavailable, such as after a wrapped KV-cache file restore. Loading a wrapped cache can resume
inference, but does not recreate its earlier token IDs for rewriting. It is incorrect to describe
such a restore as automatically reconstructing the session from only the replacement text.

### Cost and source cross-reference

Finding the common text prefix is linear in its compared byte length, with 8-byte comparisons
where possible. Mapping the divergence byte to a token is `O(log token_count)`; truncation is
constant time. Bitmap repair is linear in the number of discarded and retained tokens when active.
The dominant work is running full forward passes over the changed suffix and, in wrapped streaming
mode, any required earlier window rows.

| Operation | Implementation |
| --- | --- |
| SetStream input, limit and action | `src/asm/sectorlogo_v7prim.inc:sl_v7_set_stream` |
| Per-turn record and declared buffers | `src/asm/sectorlogo_data.inc`, `src/asm/sectorlogo_const.inc` |
| Append, LCP and offset binary search | `src/asm/token_stream.inc` |
| Callback action routing | `src/asm/chat_genloop.inc` |
| Diff, bitmap repair, wrapped fill-in, re-forward, zero-suffix sampling | `src/asm/chat_rewrite.inc` |
| Shared piece decoding | `src/asm/chat_decode.inc` |
| Streaming slot mapping and displacement callback | `src/asm/stream_llm.inc`, `src/asm/stream_evict.inc` |

---

## Retrieval-time stream replacement

A retrieval application ordinarily selects passages before submitting a prompt. V7 also allows a
SectorLogo callback to search while generation is under way, using the current text as part of the
query. If the application chooses to revise the generated context, `SetStream` updates the affected
suffix and inference continues from the revised state. This approach does not guarantee better answers
than prompt-based retrieval; it provides another point at which the application can supply information.
The examples and cost discussion below explain when this distinction matters.

> **Implementation scope:** V7 provides a native `SetStream` rewriting
> interface. A separate integration of SectorLogo and generation-stream
> rewriting has also been implemented in llama.cpp with KV-cache prefix reuse.
> The interfaces supported by each host are determined by its integration,
> rather than by the general-purpose SectorLogo language.

### Prompt-only inference interfaces

In one common deployment model, the retrieval application communicates with a separately operated
inference service over HTTP/TCP. The service accepts a prompt and returns generated text, but its public
interface may not expose the active token sequence or the transformer's KV-cache state:

```
+--------------------------+          +----------------------------------+
|      APPLICATION         |          |       INFERENCE SERVICE          |
|                          |          |     (internal service state)         |
|  +--------------------+  |          |  +----------------------------+  |
|  |  question          |  |          |  | prompt --> tokenizer       |  |
|  |  chat history      |  |          |  | history --> model          |  |
|  |  retrieved facts   |  |          |  |            --> sampler     |  |
|  +--------------------+  |          |  |            --> next token  |  |
|           |              |          |  +----------------------------+  |
|           v              |          |       internal model state        |
|  assemble a new          |          |       is not exposed             |
|  prompt from scratch     |          |               |                  |
|           |              |          |               v                  |
+--------------------------+          |        text answer               |
           |                          +----------------------------------+
           |                                        ^
           |     HTTP request (the new prompt)      |
           +----------------------------------------+
           |                                        |
           |     HTTP response (the answer)         |
           +---------------------------------------+
                       (a full network round-trip,
                        and available response data)
```

An API limited to prompt submission and text responses does not provide direct access to the generation
state. The following limitations apply to that specific interface model, not to every hosted inference
provider:

- **No mid-generation state control:** The application cannot modify the
  service's internal generation context through a text-only interface.
- **No direct KV-cache editing:** The API does not expose the sampler state,
  cached attention data, or the complete internal token sequence.
- **No incremental stream replacement:** Without a supported rewriting API,
  the application cannot replace previously generated tokens and resume from
  their updated context. Instead, it must submit a new generation request.

Under this limited request-response interface, the application generally submits an updated prompt
containing the question, relevant conversation history, and additional retrieved material. The amount of
repeated prefill depends on whether the remote service provides prefix caching or another mechanism for
reusing previous computations.

For an interface that cannot update the active generation state, this has several practical consequences:

- The application may need to resubmit the question and the relevant conversation history.
- Without prefix caching, the remote service evaluates the resubmitted tokens again.
- Previously generated reasoning may have to be regenerated when it is not included in the supplied context.
- Charges, where applicable, depend on the service and its handling of repeated or cached tokens.

For example, if a conversation contains 500 tokens and the current generation adds 300 more, an
implementation without reusable cached state may need to process more than 800 tokens again to reach an
equivalent context. A service with prefix caching may avoid part of that work.

### V7 Approach: One More Step Inside the Pipeline

V7 embeds a programmable callback directly in the generation loop. The callback operates after candidate
sampling and before generation continues; it can accept the candidate, request a new sample, stop
generation, or replace the generated stream through the rewrite interface.

In an ordinary generation loop, the tokenizer prepares the initial prompt. Subsequent iterations pass
sampled token IDs directly back to the model; the engine does not tokenize the entire decoded output
after every step:

```
+--------------------------------------------------------------+
|                TRADITIONAL INFERENCE PIPELINE                |
|                    (one loop = one token)                    |
|                                                              |
|   prompt ---> [tokenizer] ---> [ model ] ---> [ sampler ]    |
|                (runs once,       ^                |          |
|                 at the start)    |                v          |
|                                  |           token ID        |
|                                  |               |           |
|                                  +---- ID goes back in       |
|                                  |         (no re-tokenizing |
|                                  |          in the loop)     |
+--------------------------------------------------------------+
```

V7 extends that loop by invoking the configured SectorLogo callback after sampling a candidate token. If
the callback requests a replacement, the stream-rewriting procedure runs before generation continues:

```
+------------------------------------------------------------------------+
|                     EXTENDED PIPELINE (ours)                           |
|                                                                        |
|  prompt --> [tokenizer] --> [model] --> [sampler] --> [script hook]    |
|             (runs once,                                      |         |
|              at the start)                                   v         |
|                                                        token ID        |
|                                                                        |
|  The script sees every token and picks ONE action:                     |
|                                                                        |
|    +-------------+  +--------------+  +-----------+  +--------------+  |
|    |   accept    |  |   resample   |  |   stop    |  |   rewrite    |  |
|    | pass it     |  | reject it,   |  | end the   |  | edit the     |  |
|    | through     |  | sample again |  | generation|  | stream       |  |
|    +-------------+  +--------------+  +-----------+  +------+-------+  |
|                                                                |       |
|                                                +---------------v-----+ |
|                                                | REWRITE / DIFF      | |
|                                                | ENGINE              | |
|                                                | - find first point  | |
|                                                |   where old and new | |
|                                                |   text differ       | |
|                                                | - roll the model's  | |
|                                                |   memory (KV cache) | |
|                                                |   back to that point| |
|                                                | - re-tokenize just | |
|                                                |   the changed tail  | |
|                                                | - recompute just    | |
|                                                |   that tail         | |
|                                                +---------------------+ |
+------------------------------------------------------------------------+
```

This arrangement gives applications three distinct forms of control, each with a different computational cost:

- **In-process retrieval control:** SectorLogo executes as part of the
  inference process rather than requiring a separate wrapper for every
  generation callback.
- **Per-token callback:** The host invokes the callback for generated token
  candidates. An accept-only program adds script-dispatch overhead without
  changing the output.
- **Incremental rewriting:** The engine identifies the first affected token,
  preserves the reusable KV-cache prefix, re-encodes the replacement suffix,
  and evaluates the necessary continuation before returning to generation.

### Retrieval Integration

A retrieval callback can construct a search query from the reasoning text that the model has generated
so far. The application can then retrieve new material and decide whether to incorporate it into the
active context. A typical integration follows these steps:

1. Allows generation to proceed to the configured callback condition.
2. Extracts the available generated text and constructs a retrieval query.
3. Retrieves relevant records and constructs a replacement context.
4. Calls `SetStream` to update the generated stream and its inference state.
5. Continues generation using the revised context.

When the modification is confined to a suffix, the engine can retain the unaffected KV-cache prefix. The
amount of recomputation depends on the rewrite boundary and the active cache configuration.

### Execution Example

The following example traces the operations involved in inserting retrieved text into an existing
generation. It illustrates the state transition rather than prescribing a particular retrieval strategy.

The example uses rounded values to make the sequence of operations easier to follow.

```
User asks:       "what is a cat?"
                 (question + chat template  ~ 200 tokens)

Model thinks:    "The user asks about cats. Cats are small domestic felines.
                  They sleep most of the day. Their diet... hmm, what do
                  cats actually need to eat?"
                 (~ 300 tokens, and generated text supplies the retrieval query)

RAG retrieves:   "Cats are obligate carnivores: they must eat meat to
                  obtain taurine, which their bodies cannot produce."
                 (~ 50 tokens of facts)

Script:          rewrite! The facts are spliced into the model's thought
                 where it was unsure. The diff engine rolls memory back to
                 the first change and recomputes only the edited tail
                 (~ 60 tokens).

Model continues: "...so a cat is a small carnivorous mammal, an obligate
                  carnivore that needs meat in its diet."
                 (~ 100 answer tokens, no re-thinking required)
```

### Computational Cost

The following example uses a specific token count at the point where retrieved material is inserted.
These values illustrate the work performed by the rewrite algorithm; they are not benchmark results:

| Symbol | Meaning | Tokens |
| --- | --- | --- |
| **C** | Conversation so far (question, history, template) | 500 |
| **T** | Thought the model has already produced | 300 |
| **K** | Retrieved records inserted into the context | 50 |
| **A** | Final answer | 100 |

In the cost comparison, both strategies have already processed the same context and generated the same
initial text. The comparison therefore considers only the additional work that follows retrieval.

**Full context reconstruction:**
- New prompt re-sent: C + K = **550 input tokens**
- Model regenerates the whole thought and answer: T + A = **400 output tokens**

**Incremental stream replacement:**
- The C-token prefix and the unchanged part of the thought stay in the model's memory: **0 tokens recomputed**
- Only the changed tail is re-encoded and re-fed: about K plus a small splice = **~60 input tokens**
- Model continues its thought briefly and answers: **~140 output tokens**

For this illustrative cost model, an input token contributes one unit and an output token contributes
four units. This weighting is an assumption of the example, not a statement about every provider's
current pricing:

```
Old way:  550 in + 400 out  =  550 + 1,600  =  2,150 cost units
Our way:   60 in + 140 out  =   60 +   560  =    620 cost units

Savings:  1 - 620 / 2,150  =  ~71%
```

The example assumes that the external service reprocesses the entire supplied prompt and does not reuse
a cached prefix. Actual computational and financial costs depend on the model, provider, prefix-caching
policy, and rewrite location. In V7, the unaffected KV-cache prefix is retained, while the tokens after
the first affected position must be evaluated. Thus, the cost of the revision depends on the affected
suffix rather than automatically on the entire conversation history.

### Resource Implications

- **Less repeated prefill:** A rewrite may retain the unaffected KV-cache
  prefix instead of recomputing it from the beginning.
- **Retained generated context:** Earlier tokens remain available if they
  precede the first affected token and the active streaming configuration
  permits reuse.
- **In-process control:** Local retrieval and rewriting do not require a
  separate network request at every insertion boundary.
- **Reduced repeated processing:** In the illustrative example, rebuilding a
  1,000-token context three times requires more than 3,000 token evaluations;
  incremental rewriting can avoid re-evaluating unchanged prefixes.

The essential requirement is access to an inference implementation that exposes the generation state. A
local process or a suitably extended server can provide incremental rewriting; a text-only
request-response API cannot provide equivalent control without additional server-side functionality.

### Stream rewriting with a llama.cpp integration

Stream replacement is an inference-control operation rather than a feature tied to V7's transformer
kernels. A separate SectorLogo integration for llama.cpp has been implemented and supports modifying
generated text while preserving reusable KV-cache entries. The interface therefore permits applications
to use the same style of retrieval-time context modification with a different inference backend. The
exact host commands and deployment options depend on the particular integration build.

An HTTP application can continue to use a conventional OpenAI-compatible request interface while a
suitable inference server executes the script hook internally. In this configuration, model execution,
the retrieval corpus, the SectorLogo program, and the stream-replacement mechanism are located in the
server process. Local retrieval avoids an additional network request for each database lookup. Multiple
client applications can use that server without implementing the generation callback in each client.

The distribution's V7 server is intended primarily for local development and testing. The llama.cpp
implementation also provides access to model and accelerator configurations beyond the Windows x64 CPU
scope of V7. An external service must independently address authentication, resource isolation, capacity
planning, and other operational requirements; the presence of a script interpreter or an HTTP interface
does not establish those properties.

When a retrieval result changes only the generated suffix, the inference backend can reuse the
unaffected KV prefix and evaluate the replacement suffix. The amount of work depends on the position of
the first changed token, the size of the replacement, and the attention implementation. This differs
from submitting the entire updated conversation to an interface that does not expose incremental state
modification. Hosted services may offer their own prefix caching, so the cost comparison depends on the
actual service.

### Summary

Stream rewriting updates the generated context in place from the application's perspective. The
execution cost depends on the affected suffix, and the inference backend may preserve unmodified
KV-cache entries rather than processing the full conversation again.

## StreamingLLM

StreamingLLM keeps the attention cache within a fixed storage budget during long-running conversations.
Instead of retaining a KV-cache row for every token in the history, V7 preserves a configured group of
attention-sink tokens at the beginning of the session and a rolling window of recent tokens. As new
tokens arrive, older rows outside those retained regions are displaced. The corresponding text can be
supplied to a SectorLogo callback, allowing the application to store or summarize information that the
transformer can no longer attend to directly.

This arrangement affects operations that depend on earlier transformer state. Rewinding a normal cache
can reuse rows that are still present, but a row displaced by StreamingLLM cannot be restored just by
decreasing the current position. V7 therefore distinguishes ordinary position changes from the broader
reconstruction used by supported stream-replacement operations when the cache has wrapped. Applications
should use the documented rewrite interface for those cases rather than assuming that every past KV
position remains available.

The application configures this mode once per session, before generating the first token:

```c
dll.v7_set_streaming(/*enable*/ 1, /*num_sinks*/ 4, /*window_size*/ 1020);
// or the fill-grade convenience (start evicting when 4096 rows are full):
v7_set_streaming_fill(lib, 1, 4, 4096);   // == v7_set_streaming(1, 4, 4092)
```

### Attention-Window Semantics

- **S** (`num_sinks`) — the first S cache rows hold the attention-sink tokens.
  The engine initializes these rows at session start and retains them for the duration of the session.
- **W** (`window_size`) — the rolling window behind the sinks. These rows contain the most recent context available for attention.
- **C** (`num_sinks + window_size`) — the **fill grade**: how full the cache
  receives before old tokens are displaced from the attention window. C is capped by the model's
  trained context length (`seq_len`).
- **slot(pos)** — while the session is below C, everything behaves exactly
  like a normal v7 session. From C on, the token at position `pos` is located at
  cache row `S + ((pos - S) mod W)`: the oldest window token is replaced,
  everything else stays where it is.
- Each key is rotated when it is stored. The query-side position adjustment accounts for the circular layout, preserving the relative positions of the retained sink tokens and the rolling window.

### Return contract of `v7_set_streaming`

| Case | Result |
| --- | --- |
| accepted (enable=1 at `pos == 0`, `C <= seq_len`) | 0 |
| bad arguments, `S+W` overflow, `C > seq_len` | -1 |
| image-capable model (`__sys_mrope_bound` in the DB) | -1 (image turns with a wrapped cache are not supported yet) |
| called before `v7_initialize_inference` | FATAL |
| enable at `pos > 0` (changing S/W mid-session) | FATAL |
| disable at `pos > C` (the circular cache has wrapped) | -1 |
| disable at `pos <= C` | 0 (stores only the flag) |

### Callback input

When a token leaves the window, the engine accumulates it in a batch. Every **256 window displacements**
— and at the end of every chat turn — the batch is delivered:

1. The batch text (decoded exactly like chat output: word markers become
   spaces, `<0xNN>` escapes become raw bytes) is written to the DB key
   **`stream_remove_text`**.
2. The line **`stream_remove_meta`** = `first=<position> tokens=<count>` says
   where the batch begins in the session and how many tokens it covers.
3. The configured SectorLogo program under **`on_stream_evict`** runs. A typical body:

```logo
DatabaseSet "ltm_probe" DatabaseGet "stream_remove_text"
DatabaseSet "ltm_probe_meta" DatabaseGet "stream_remove_meta"
```

Delivery is exactly-once, in session order. Attach the SectorLogo hook before the first turn; without a
host the batch is dropped and the next one starts fresh. After a KV-cache restore the engine first
refills its bookkeeping for one window's worth of tokens, then reporting resumes exactly. A nested
`v7_generate` call runs entirely in legacy mode and preserves the session.

### Turn routing and limits

- A prompt that ends within the first C rows uses the ordinary batched
  prefill (bit-identical to a non-streaming session). Longer prompts are
  digested token by token.
- Generation stops on EOS / stop tokens / `max_gen` / output-buffer end —
  never because the cache filled up.
- Positions beyond the configured token-history capacity are not addressable through `GetToken`, which clamps its requested position. The `SetStream` callback remains supported: before the circular cache wraps, it uses the ordinary rollback path; after wrapping, it reconstructs the session using the recorded context and replacement text.
- Image (M-RoPE) turns are not supported while streaming: the setter refuses
  the enable on image-capable models. Disable streaming for vision-language sessions; the documented streaming path applies to text-only generation.
- Each generated token requires additional circular-index computation, cache-row bookkeeping, and displacement accounting. The relative overhead depends on the model and workload.

### Testing

`tools/v_stream_validate.exe` (in `verify.bat` Phase 3) pins the setter contract, legacy bit-equivalence
in F32 and Q8, a small-window wrap (S=4 W=8 C=12) with callback delivery, and the disable contract. The
C oracle is stored in `tools/stream_ref.h`.
