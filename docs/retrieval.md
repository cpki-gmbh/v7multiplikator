# Text Search and Retrieval

V7 includes the components needed to build local text-search and retrieval applications. MicroDB
supplies persistent storage, SectorLogo provides text processing and application logic, and native
primitives support operations such as tokenization and BM25 ranking. A language model is optional: the
same facilities can return ranked search results without starting inference.

When retrieval is combined with generation, the application decides when to search and how to use the
results. Documents may be selected before generation, or a SectorLogo callback may retrieve material
while a response is being produced and insert it through stream replacement. This manual describes both
approaches and includes implementation examples, configuration details, and recorded experiments.
Measurements and individual experimental outcomes are presented as observations about their stated
workloads, not general guarantees of answer quality.

## Contents

- [Search facilities](#search-facilities)
- [Embedded BM25 example](#embedded-bm25-example)
- [Retrieval and RAG implementation reference](#retrieval-and-rag-implementation-reference)
- [Model-assisted retrieval](#model-assisted-retrieval)
- [Internet retrieval interface](#internet-retrieval-interface)

---

## Search facilities

V7 can perform document retrieval without generating a response. In that arrangement, MicroDB holds the
application data, SectorLogo provides the query-processing logic, and native text operations handle the
repetitive work of splitting, normalizing, and comparing strings. The application decides how to divide
its source material into searchable passages and how to rank those passages; storing a value in MicroDB
does not by itself create a full-text index.

A typical query begins by normalizing the incoming text and extracting the terms to search for.
SectorLogo offers `Words` for word extraction and `NGrams` for constructing consecutive term groups.
`Split` and `Search` are useful when preparing document passages or applying simpler matching rules.
These operations run in native code, so a retrieval script need not loop through every character in
interpreted Logo. The resulting terms can be compared with records fetched from MicroDB, scored with
BM25, and sorted before the script returns a selection of passages.

BM25 is a ranking method, not a substitute for corpus preparation. Its results depend on the chosen
document boundaries, term frequencies, document frequencies, and scoring parameters. The embedded
example in the next chapter keeps those decisions visible in a single SectorLogo program. The larger
implementation reference describes the external `bm25_rag.exe` utility and other retrieval strategies,
including examples that modify model context during generation.

### Using search independently of inference

Retrieval does not require an active text-generation session. The native and C++ retrieval tools can
tokenize and rank a corpus independently, while a SectorLogo program can implement the scoring process
itself when a small, programmable search workload is preferable. These two implementations have
different performance characteristics; the dispatch count measured for a pure interpreted BM25 script
does not apply to the external native BM25 executable.

An application can invoke its SectorLogo search procedure and use the ranked passages directly. This is
useful for ordinary document lookup, local knowledge bases, and systems that need deterministic
retrieval but no language generation. MicroDB provides persistent records and lookup operations, while
the application remains responsible for selecting the relevant keys, maintaining its corpus, and
applying whatever indexing policy it requires. The [storage reference](storage_and_layouts.md) defines
the underlying database representation, and the [SectorLogo manual](sectorlogo.md) explains the
text-processing and collection operations.

### Supplying search results to a model

Retrieval data remains external input, even when a script inserts it into the generated stream. Web
pages and stored memories may contain inaccurate statements or instructions addressed to a model rather
than factual content. A rewrite that blends those passages with a thinking section does not preserve a
reliable source boundary in the model's internal text. Applications requiring provenance should retain
document identifiers, URLs, and retrieval decisions outside the model output and should verify any
action derived from external text. This documentation does not claim that the demonstration scripts
provide general prompt-injection protection.

Retrieval can also provide context for inference. In a prompt-based workflow, the application searches
before submitting the prompt and includes the selected material in that prompt. In an active workflow, a
SectorLogo callback can construct a query from accepted generated text and retrieve material at a point
chosen by the application. The callback can then call `SetStream` to incorporate the selected passages
into the context from which the model continues. This is a separate generation-control operation, with
its own token-history and KV-cache requirements, rather than a side effect of loading the database.

A well-scored passage is not necessarily a factually correct one, and successful retrieval does not
guarantee that the model will interpret it correctly. For that reason the retrieval tests record both
the ranking behavior and the resulting model output. The [generation-control
manual](generation.md#stream-replacement-and-state-consistency) describes how text replacement affects
inference state; the experiments later in this manual explain the individual retrieval policies and the
conditions under which they were evaluated.

## Embedded BM25 example

The following example shows a complete BM25 search implemented in SectorLogo. It prepares the source
text, derives search terms, calculates relevance scores, and chooses passages without invoking an
external search executable. No transformer pass is required to carry out the search. Because the program
contains the corpus preparation and scoring policy together, it is also a useful starting point for
understanding how the native text primitives fit into a larger retrieval application. The recorded
timings refer to this particular corpus and query set; they do not establish a general performance limit
for larger collections.

---

### Overview

In this example, the knowledge base is represented by ordinary SectorLogo strings. The program
constructs its search index when it receives a query, rather than relying on an index prepared by a
separate service. A larger application could keep the same ranking procedure while storing its source
passages persistently in MicroDB and preparing the index at a different stage.

The example also demonstrates how retrieval can participate in generation. When the configured callback
point is reached, the script obtains the available text, constructs a query, scores the candidate
passages, and decides whether a rewrite is useful. This is one application policy, not a fixed sequence
imposed on every search. The diagram below follows the script's actual operations, all of which execute
within `v7_threading.dll`.

```
 token
  `- hook fires on </think>
       |- DatabaseGet "bm25_done"             (two-pass guard flag)
       |- flag set -> accept the tag, clear flag, done
       |- GetStream                            (full generated think text)
       |- bm25Search:Q                       (pure Logo, ~100 ms at 5 KB)
       |    |- chunking      split on paragraphs, then sentences
       |    |- tokenize      Words (native word list) + stopword drop
       |    |                + NGrams (native bigram word pairs)
       |    |                + unigram/bigram terms
       |    |- index pass    per-chunk tf alist + global df alist,
       |    |                keyed by HashNumber (Find* needs numeric keys)
       |    |- BM25 score    NaturalLog and infix floating-point arithmetic
       |    |                + coverage multiplier + topic bonus
       |    |- rank          Sort + Reverse (native radix sort)
       |    `- floor         density/topscore dual answer-presence gate
       `- SetStream:R _ "\n\nnow i should answer according to these facts\n\n</think>\n"
            (replace-think rewrite + re-prefill, then set the guard flag)
```

At the EOS token the script clears the state flag again, so the next turn starts clean. The rewrite
replaces the think content with the retrieved facts plus a short directive rather than appending to the
original text, which is the replace-think strategy the research log evaluates.

### Scoring (identical to bm25_rag.exe / bm25_rag.cpp)

The embedded script follows the scoring rules used by the `bm25_rag.exe` reference implementation. It
first converts the input to lowercase alphanumeric terms, excludes terms shorter than two characters,
and removes the configured stopwords. Bigrams are constructed from the remaining words as `w1_w2`,
including pairs that become adjacent after stopword removal. The corpus is divided into sentence chunks,
and the script maintains term frequencies for each chunk together with document frequencies across the
corpus.

For a term with document frequency `df` in a corpus of `N` chunks, inverse document frequency is
computed as `ln(1 + (N - df + 0.5) / (df + 0.5))` and is bounded below by zero. Terms whose `df/N`
exceeds `0.4` are treated as common by assigning `df = N`. BM25 then uses `k1 = 1.2` and `b = 0.75`. The
first sentence of each paragraph receives a topic-sentence multiplier of `1.15`.

The script also measures how much of the question is represented in a chunk. It multiplies the score by
`0.3 + 0.7 * (number of distinct query unigrams matched / number of distinct query unigrams)`. A
candidate result passes the recorded answer-presence threshold when the highest score divided by the
number of distinct query unigrams is at least `0.026`, or when the highest score itself is at least
`3.0`. These thresholds were calibrated in research experiment T7. In that experiment, the cat-related
reasoning texts had densities of `0.028`–`0.048`, while the quantum-related texts had densities of
`0.021`–`0.024` and a highest score of `2.6`. The experiment establishes the values for that corpus, not
a general threshold for unrelated data.

After scoring, the script selects up to three chunks with positive scores, orders them
deterministically, and joins their text with newline separators. This is the complete selection policy
for the embedded example; applications can adjust the chunking, scoring, thresholds, and number of
returned passages as needed.

### Language techniques the script demonstrates

| Need | Technique |
| --- | --- |
| Legacy string iteration with `Part` | `Part:s:i 1` returns `""` past the end and can serve as a loop-bound sentinel. Prefer `TextLength` for bounded iteration and `Words` for tokenizing. |
| Word list from text (native) | `Words:text` does lowercase, alnum runs and the length-2 floor in one call |
| Adjacent word pairs (native) | `NGrams:uni 2` gives sliding n-gram windows over the filtered unigram list in one call |
| Stopword test (one native call) | `IsNumber Search (" " _:w _ " "):SW` against a space-delimited stopword string |
| String-keyed map, since Find* keys must be numeric | `[[HashNumber "term"...]...]` sorted by hash and probed with `FindFirstAtLeast` plus an exact-atom compare |
| tf/df alists | `[[hash term count]...]` triples, updated via `FindItem` and `SetItem`/`Prepend` |
| Ranking | `Reverse Sort [[score text]...]`, since sub-lists sort by their numeric car |
| Floating-point division | `(AsFloat:tot / :N)`; integer division truncates when both operands are NUMBER |
| State across the two hook passes | `DatabaseGet`/`DatabaseSet` on a microdb key, cleared again at the EOS token |
| Safe standalone testing | the `</think>` hook guard is a no-op when `:Piece` is unbound, so test drivers can be appended directly to the sidecar file |

### Measured behavior (standalone, 30 sentence chunks over 5 KB)

| Query | Result |
| --- | --- |
| "What do cats eat?... natural diet" | 3 diet chunks, topscore 3.63, density 0.61 |
| "Are cats hunters?... birds and mice" | 3 hunter-myth chunks, topscore 6.00 |
| "What should I feed my pet cat?" | 3 feeding chunks, topscore 4.30 |
| "Explain quantum physics..." | empty (floor) |
| "Capital of France..." | empty (floor) |
| "the and of to" (stopwords only) | empty (no query terms) |

On the recorded 5 KB corpus, one search took approximately 100 ms after switching to the native `Words`
and `NGrams` primitives. The earlier character-by-character tokenization approach took approximately 200
ms in the same example. These figures include the index construction performed for each invocation; the
script does not retain an index between turns. When retrieval is used during generation, the full cost
also includes the stream replacement and the forward computation required for the changed suffix. The
recorded inference example produced 575 tokens and completed one rewrite cycle.

### End-to-end override on the herbivorous false-belief corpus

Prompt: `what do cats eat?`, with Qwen3-0.6B Q8, `-n 1500`, `-rewrite` and `-script
examples\research\herbivorous_cats\herbivorous_bm25.logo`.

The experimental knowledge base deliberately contained a false claim about the diet of cats. Before the
rewrite, the generated reasoning text referred to a mixture of meat, vegetables, and fruit. At the
`</think>` boundary, the script retrieved the misleading passages and replaced the relevant generated
context. After the replacement was evaluated, the final answer repeated the inserted false claim,
including the statement that cats do not eat meat. This demonstrates that retrieval running entirely in
SectorLogo can alter the model's subsequent answer; it does **not** demonstrate that the answer was
factually correct. Cats are obligate carnivores. The full transcript is preserved as experiment 28 in
the development-only log at `attic\docs_research_logbook\rag_research_log.md`.

### Capacity Limits and Scaling Options

The pure SectorLogo BM25 implementation is a transparent reference implementation, not the only search
path. The project also provides native string processing and C++ retrieval tools. For large collections,
measure the chosen retrieval path rather than extrapolating the interpreter's per-character cost to
every configuration. The example plaintext memory stores can grow as records are appended; no universal
consolidation or contradiction-resolution guarantee is built into BM25 scoring. Recency weighting
changes ranking but does not establish factual accuracy.

The embedded example rebuilds its index every time it searches. Once the native `Words` and `NGrams`
functions have prepared the terms, SectorLogo still performs several interpreted operations per word
to maintain term frequencies and document frequencies. That is practical for the example's small
corpus of about 5 KB and, according to the recorded implementation notes, remains comfortable for many
collections around 10 KB to 100 KB; those are workload observations rather than established upper
limits.

Larger corpora can use a prebuilt MicroDB index instead of rebuilding every term map for each question.
In that arrangement, entries such as `rag:post:<term>` contain postings and `rag:df:<term>` contain
document frequencies. The query script then visits the postings for terms that appear in the query and
applies the same ranking, score thresholds, and context-insertion policy. The development-only design
note at `attic\docs_plans\bm25_after_sorted_list_plan_analysis.md` discusses the scaling choices in
greater detail.

### Files

| File | Role |
| --- | --- |
| `examples/research/herbivorous_cats/herbivorous_bm25.logo` | The sidecar: corpus, tokenizer, BM25, and the hook |
| `examples/research/herbivorous_cats/run_bm25.bat` | End-to-end inference run for this sidecar |
| `examples/rag/bm25_rag.cpp` | C++ reference for the scoring math and the behavioral parity target |

## Retrieval and RAG implementation reference

The repository contains several retrieval implementations, developed to explore different ways of
selecting and incorporating external information. They share a set of V7 host interfaces but differ in
their query construction, selection policy, timing, and persistent state. This chapter documents those
implementations alongside the measurements used to examine them. Historical experiments are retained to
explain specific configurations; their results should not be read as requirements or general performance
claims.

> **Tool names**:
> `bm25_rag.exe` (`examples/rag/bm25_rag.cpp`) is the retrieval tool.
> It supports search, remember, score floors, deduplication, recency, recent
> fallback, `xbase`, and `base=PATH`. `llm_text_processor.exe`
> (`examples/rag/llm_text_processor.cpp`) is the distiller; its
> `-distill` mode is the memory distiller. Sidecars reach both through the
> direct external-command support of `DefineCommand` and `v7s_writefile`.

V7 provides multiple retrieval-augmented generation implementations built around a common token callback
and stream-rewriting interface. This reference describes the retrieval algorithms, integration points,
context-insertion policies, validation results, and known limitations of each implementation. Experiment
numbers identify measurements in the development-only research log under `attic\docs_research_logbook\`.
Measurements describe the tested configurations and should not be interpreted as universal performance
or quality guarantees.

The following files provide detailed source descriptions, experimental records, and background for the
implementations discussed in this chapter:

| Document | Content |
| --- | --- |
| this document | the pure-SectorLogo BM25 design note (consolidated here) |
| `extended_ragg_bm25plus_and_other_improvements.md` | bm25_rag production-options design rationale |
| `attic\docs_research_logbook\rag_research_log.md` (dev-only) | the primary experiment log (verbatim) |
| `attic\docs_research_logbook\rag_research_analysis.md` (dev-only) | per-experiment conclusions |
| `attic\docs_research_logbook\rag_augmentation_example.md` (dev-only) | two annotated end-to-end runs |
| `internet_rag_dev_log.md` | internet RAG development log |
| `attic\docs_research_logbook\cat_rag_devlog.md` (dev-only) | adversarial-prior experiment, 4 rounds |
| `attic\docs_research_logbook\helion_devlog.md` (dev-only) | unfamiliar-fact experiment, 5 rounds |
| `attic\docs_research_logbook\ardmark_devlog.md` (dev-only) | fictional-history entity QA, 3 rounds |

---

### Table of contents

1. [Scope: the eight RAG systems](#scope-and-retrieval-variants)
2. [The shared substrate](#shared-retrieval-components)
    - 2.1 The per-token hook
    - 2.2 Stream primitives
    - 2.3 The rewrite engine
    - 2.4 `v7_execute` and the capture boundary
     - 2.5 The SectorLogo FFI (`CallLibrary`)
    - 2.6 DB primitives and engine exports used by sidecars
    - 2.7 Hook timing subtleties
3. [Retrieval engines I: the BM25 family](#bm25-retrieval-implementations)
    - 3.1 `bm25_rag.exe`, BM25 retrieval tool
    - 3.2 The production options surface (every option analyzed)
    - 3.3 The pure-SectorLogo BM25 (`herbivorous_bm25.logo`)
4. [Retrieval engines II: extractive selection](#extractive-selection)
    - 4.1 MMR passage selection (`text_shortener.cpp`)
    - 4.2 The HTML content processing (`html_parser.cpp`, `webtool.cpp`)
5. [System I: Static RAG (`sectorlogo_rag.logo`)](#5-system-i-static-rag-sectorlogo_raglogo)
6. [System II: The research strategy sidecars](#6-system-ii-the-research-strategy-sidecars)
    - 6.1 Strategy catalog (every sidecar, annotated)
    - 6.2 The ten design criteria
    - 6.3 Off-topic response quality assessment
7. [System III: The production sidecar](#7-system-iii-the-production-sidecar)
8. [System IV: Persistent memory RAG (v1 through v6)](#8-system-iv-persistent-memory-rag-v1-through-v6)
    - 8.1 Architecture and the two hooks
    - 8.2 Sidecar evolution v1 -> v6
    - 8.3 Memory distillation (`llm_text_processor.exe -distill` via `DefineCommand`)
    - 8.4 Recall framing: the 7-turn tuning conversation
    - 8.5 The dual-corpus split and write guards
9. [System V: Internet RAG (`web_ragg.exe`)](#9-system-v-internet-rag-web_raggexe)
    - 9.1 Architecture
    - 9.2 Component analysis
    - 9.3 `makequery`: think text -> search query
    - 9.4 Distillation prompts and the no-meta-language rule
    - 9.5 The generic date prior
    - 9.6 Output discipline (stdout hijack)
    - 9.7 The page cache (`simpledb`)
    - 9.8 Development log summary
10. [System VI: Integrated memory + internet (`sidecar_memory_rag_v7.logo`)](#10-system-vi-integrated-memory--internet-sidecar_memory_rag_v7logo)
11. [System VII: Thought RAG standalone (`llm_text_processor.exe`)](#11-system-vii-thought-rag-standalone-llm_text_processorexe)
12. [System VIII: The embedded retrieval system](#12-system-viii-the-embedded-retrieval-system)
13. [Score floors and calibration](#score-thresholds-and-calibration)
14. [Source grounding and genericity rules](#14-source-grounding-and-genericity-rules)
15. [Consolidated failure modes](#known-limitations-and-observed-failures)
16. [Consolidated performance data](#measured-performance)
17. [Complete file inventory](#retrieval-source-files)
18. [Building a custom sidecar](#building-a-retrieval-script)
19. [The complete sidecar source annotations](#19-the-complete-sidecar-source-annotations-system-ii-annotated-source)
20. [Memory v2: the full source, annotated](#memory-retrieval-v2-annotated-source)
21. [Internet RAG: the accurate component picture](#internet-retrieval-components-and-interfaces)
22. [The complete experiment index](#experiment-index)
23. [SectorLogo implementation guide for retrieval applications](#implementing-retrieval-with-sectorlogo)
24. [The cats->dogs stream-rewrite demo](#minimal-stream-rewriting-demonstration)
25. [Design decision record](#design-rationale)
26. [Tuning backlog](#retrieval-tuning-observations)
27. [End-to-end walkthroughs](#end-to-end-examples)
28. [RAG on the server](#retrieval-through-the-http-server)
29. [Validation and testing](#validation-and-testing)
30. [Quick-reference tables](#reference-tables)

---

### Scope and retrieval variants

The repository contains eight retrieval implementations that use the same generation hooks but serve
different purposes. Some illustrate individual strategies, while others combine retrieval with stored
memory or external text sources. Their main differences are the query source, the retrieval method, and
the way selected information enters the generation context:

| # | System | Core files | Retrieval | Injects | Status |
| --- | --- | --- | --- | --- | --- |
| I | Static RAG | `src/logo/sectorlogo_rag.logo`, `examples/rag/bm25_rag.cpp` | BM25 over `ragg_base.txt` | retrieved lines appended into the think block | stock demo |
| II | Research strategies | `examples/research/sidecar_*.logo` (20+) | bm25_rag.exe | replace / negate / prepend / multi-query / erase / gated hybrids / think-continue | research artifacts |
| III | Production sidecar | `examples/research/sidecar_production_final.logo` | bm25_rag.exe with floors | think-continue, neutral-delta marker, maxlines=3, minscore=0.026 | release-ready (iter 19) |
| IV | Persistent memory | `examples/rag/memory/sidecar_memory_rag*.logo` (v1..v6) | bm25_rag.exe over a growing corpus (later: two corpora) | "Things I remember..." restart-from-scratch; EOS-time distillation | working |
| V | Internet RAG | `examples/rag/web_ragg/web_ragg.exe` (+8 components) | DuckDuckGo search -> page fetch -> MMR -> LLM distill | ~100-word current-events paragraph | working (v2 tuned) |
| VI | Memory + internet | `examples/rag/memory/sidecar_memory_rag_v7.logo` | IV's dual corpus + V's web paragraph | three labeled blocks in one rewrite | working (exp 29/30) |
| VII | Thought RAG | `examples/rag/llm_text_processor.cpp` | BM25 (CLI-tunable) + v7 LLM reformulation | prints reformulated "thoughts" to stdout (tool, not sidecar) | working; also the memory distillation engine |
| VIII | Embedded retrieval | `examples/research/herbivorous_cats/herbivorous_bm25.logo` | BM25 implemented 100% in SectorLogo | standard append | working; proves the language |

Most of the generation-side retrieval examples use the model's accepted reasoning text as their query.
The script waits until the configured reasoning boundary, ordinarily the pending closing `</think>`
marker, and searches for passages that address the material in that reasoning. It may include the
original user question as well, so the query retains the user's subject while incorporating the details
that appeared during generation. The selected passages are then added to the model's working context
through stream rewriting. Some strategies replace the reasoning section, while others preserve portions
of it or continue from a revised suffix. The standalone search and distillation tools share parts of
this retrieval process but do not themselves modify transformer state.

This is a form of query expansion driven by the text being generated, rather than by a separate
embedding model. The examples use plain-text corpora, BM25 lexical ranking, extractive selection, and,
in some cases, model-based distillation of retrieved documents. They do not require a vector database or
a separate re-ranking model. The research notes refer to the benefits and limitations of this query
construction as Laws 2 and 8; those labels describe observations in the local experiments rather than
requirements of the runtime.

There is a practical reason to search at the reasoning boundary. In the recorded trials, a short
question sometimes produced 100–300 words of related reasoning, which supplied additional terms for
BM25. A script could then insert relevant material without modifying the original user prompt or chat
template. That additional specificity sometimes improved the results, but it could also reinforce an
incorrect assumption made earlier in generation. The later experiments describe both outcomes and the
measures used to limit query contamination.

#### 1.1 Where this sits in the current research landscape

The implementations in this repository explore retrieval during generation, when the text already
produced by the model can provide additional clues about which information is relevant. This approach
has precedents in research on interleaved retrieval, regeneration, and programmable reasoning. The table
below places V7's experiments in that context. It describes differences in the documented mechanisms
rather than attempting to rank unrelated systems by their reported results.

| Work | What it does | Relation to v7's approach |
| --- | --- | --- |
| IRCoT (Trivedi et al., ACL 2023, arXiv:2212.10509) | Interleaves retrieval with each chain-of-thought sentence; the CoT guides the next retrieval | Shares the "reasoning text is the query" signal (v7's Law 2/8); v7 does the interleaving at one strategic point (think close) via stream rewriting rather than per sentence |
| FLARE (Jiang et al., EMNLP 2023, arXiv:2305.06983) | Predicts the upcoming sentence, retrieves on low-confidence tokens, and *regenerates* the sentence | Closest published analogue to the rewind-and-continue mechanics; FLARE discards the draft sentence and regenerates it conditioned on the retrieved documents, v7 rolls back the KV cache and continues from the rewritten stream in place |
| Self-RAG (Asai et al., 2023, arXiv:2310.11511) | Trains the model with reflection tokens to decide *when* to retrieve and to critique retrieved passages | v7 reaches a similar decide-when-to-retrieve effect without fine-tuning: the engine-level hook and a plain script make the policy external and editable at runtime |
| Search-o1 (Li et al., 2025, arXiv:2501.05366) | Agentic search inside a large reasoning model's thought process, plus a separate Reason-in-Documents module that distills retrieved pages before injection | Parallel evolution: v7's internet RAG (System V) also distills fetched pages with the model itself before injection, and triggers at think boundaries rather than on special tokens |
| Agentic RAG surveys (Singh et al., arXiv:2501.09136 and successors) | Taxonomize the shift from static retrieve-then-read pipelines to agent-controlled retrieval loops | V7 is a small, fully local entry in this space: the "agent" is a deterministic Logo script with hard engine hooks (reject/rewrite/halt) rather than a second LLM loop |

The V7 examples emphasize a specific operation: they can replace text within the already-generated
reasoning section and continue from the corresponding transformer state. That is different from adding a
search result to the initial prompt or appending it after a tool call, although those approaches remain
useful and may be simpler for many applications. The recorded experiments primarily use a 0.6B model,
making them informative about the limits of a small decoder but insufficient to establish how the same
methods would behave with larger models. The later sections discuss the conditions under which retrieval
was useful, ignored, or actively harmful.

#### 1.2 Does anyone else inject inside the turn? (a search, Sep 2026)

The September 2026 literature review compared V7's stream replacement with research that introduces
information during an ongoing generation turn. Three aspects of an implementation matter for this
comparison: where the new text is placed, how it is represented to the model, and whether previously
computed transformer state is reused. The reviewed publications cover several related approaches, but
they do not all provide a text-replacement operation with the same semantics. The distinctions below
summarize the scope of that review rather than establishing an exclusive novelty claim.

- **Tool-token protocols (Search-o1, tool-integrated reasoners,
  interleaved-thinking agents like MindWatcher / Skywork-R1V4).** The
  closest family. In Search-o1 the framework uses reasoning-model output to initiate
  `<|begin_search_query|>...<|end_search_query|>` tokens during generation; the
  framework pauses, retrieves, distills the documents (Reason-in-Documents),
  and appends the result wrapped in
  `<|begin_search_result|>...<|end_search_result|>` into the *same
  reasoning sequence*, then continues generation. That is genuinely
  inside-the-turn injection. Two properties still differ from v7: the
  injected text is *delimited as an observation*, the model can
  distinguish retrieved content from its own reasoning by the wrapper
  tokens, whereas v7 splices unmarked text into the think stream so the
  model treats it as its own thought, and the mechanism requires a model
  trained/prompted to emit the special tokens (QwQ/R1-class reasoners),
  whereas v7's trigger is engine-side and works with an unmodified
  off-the-shelf checkpoint. Search-o1 does not use V7’s automatic token-aligned `SetStream` path: content is
  appended to the running sequence, never spliced into already-generated
  text.
- **Regeneration approaches (FLARE).** On low-confidence tokens, FLARE
  retrieves with a lookahead query and *regenerates the sentence from the
  prompt context*. The previous draft is discarded wholesale; the cache is
  not rolled back to a divergence point and continued in place.
- **Latent/steering approaches (LPSR, activation steering, divergence
  decoding).** Correction happens in the residual stream or logits, a
  steering vector is injected at rollback, not retrieved *text* placed in
  the visible reasoning. The token stream is not modified by this retrieval method.
- **Post-hoc approaches (Majumder et al., ACL 2022, arXiv:2203.11399).**
  Knowledge is injected into a *finished* draft response via
  gradient-based re-decoding of the complete response, outside the turn,
  after generation ends.
- **Agent-loop memory (ReFind, GAM).** The search agent iterates queries
  and collects evidence *before* answering; injection happens in the
  reasoning stage's input, never mid-turn.
- **Interruption handling (EchoChain, arXiv:2604.16456).** Voice
  assistants revise state when the *user* interrupts mid-response, the
  external-interruption case; no external knowledge is spliced into the
  model's own reasoning.

V7 can derive the retrieval query from generated reasoning text, insert selected passages into that text
without adding a separate tool-result wrapper, and restore the affected KV-cache state before
continuing. This combination was not identified in the publications covered by the September 2026
review. Search-o1 provides a nearby example of information insertion within an active reasoning
sequence, FLARE provides a related regeneration strategy, and LPSR examines inference-time correction
involving cached state. The rollback-consistency discussion in arXiv:2608.15939 is also relevant to the
correctness of reconstruction.

The review searched arXiv from seven perspectives through September 2026. Its scope is limited to the
material examined at that time, so the absence of a direct match should not be interpreted as proof that
no comparable implementation exists.

---

### Shared retrieval components

Several retrieval examples use the same host interfaces. The following sections describe the token
callback, stream operations, native-process invocation, FFI, and database access before explaining the
individual retrieval strategies.

#### 2.1 The per-token hook

When the host runs inference with a sidecar loaded through `-script <path>` (with optional legacy `-rewrite`), the generation loop
in `src/asm/chat.inc` calls the SectorLogo interpreter once per generated token
(`src/asm/sectorlogo_main.inc`), passing hook variables:

| Variable | Meaning |
| --- | --- |
| `:Piece` | the decoded text piece of the CANDIDATE token (before acceptance) |
| `:TokenNumber` | the candidate token id (NUMBER) |

The host stores the script under the MicroDB key `logo:sidecar`, typically through `-script` and
`v7_db_add`. The internal `v7s_sidecar_refresh` routine activates or updates the callback at chat entry
when the key is present. The script body executes for each token, so a RAG script is one large `IfElse`
on `:Piece` and `:TokenNumber` that does nothing for almost every token.

Two callback conditions are used repeatedly in the reference retrieval scripts. The first is the
candidate `</think>` token, detected with `:Piece == "</think>"`. In the tested Qwen3 configuration this
closing marker is one token, so the callback can use it as a point for retrieval and context insertion.
Systems I through VI use that boundary. The second condition is `:TokenNumber == EndToken`, which
identifies the natural end of a turn. The persistent-memory examples in systems IV and VI use that point
to prepare and store information for later retrieval.

#### 2.2 Stream primitives

The SectorLogo primitives used by retrieval applications are implemented in
`src/asm/sectorlogo_v7prim.inc`. Their behavior is summarized here:

| Primitive | Effect |
| --- | --- |
| `GetStream` | returns the entire generated stream so far (this round), as a Logo string |
| `SetStream s` | hands `s` to the rewrite engine; generation continues from the rewritten state |
| `StreamLength` | Byte length of the current output stream; takes no argument. Use `TextLength s` to measure an ordinary string. |
| `Reject` | reject the candidate token; the sampler draws the next-best |
| `Halt` | stop generation immediately |
| `DatabaseSet k v` / `DatabaseGet k` | in-memory key/value store (MicroDB-backed) holding state that must remain valid across rewriting |
| `EndToken` | the chat EOS token id, as a NUMBER |

`GetStream` returns the CURRENT ROUND's text only (after the last KV reset point). The stream contains
only generated tokens, template and prompt tokens are prefilled before `v7s_stream_init_turn` runs and
are never appended. In thinking mode the `@think` template pre-opens the think block (the database
builder includes the opening tag in the prompt), so the model's generated stream starts with think
content, not with an opening tag; in non-thinking mode the template's empty think block is prefill-only
and the stream starts with the answer.

#### 2.3 The rewrite engine

When a script calls `SetStream`, the generation loop enters `.chat_hook_rewrite` in `src/asm/chat.inc`.
The exact work depends on the current KV-cache configuration and the first changed token. In the
ordinary incremental path, the engine preserves the reusable prefix and processes the changed suffix
before continuing:

1. Roll the KV cache back to the start of the current chat round.
2. Re-tokenize (encode) the new stream text.
3. Re-prefill it through the transformer (fused prefill).
4. Resume generation at the end of the rewritten text.

The console may display the replacement text as the engine processes it. This output represents
reconstruction of the affected context, not an independent second answer. In the recorded Qwen3-0.6B
measurements, the rewrite operation took approximately 300–900 ms for a replacement of a few hundred
tokens; the subsequent generation phase ran at 19–21 tokens/s on the tested Ryzen 5 7430U in Q8 mode.
These are historical measurements for the stated configuration.

The replacement text determines the context available to the model after rewriting. The examples use
several policies with different consequences for earlier generated text:

- **Append strategies** keep the model's draft and add lines before the
  closing tag.
- **Replace strategies** discard the draft entirely and write only the
  retrieved lines (the draft is gone from the context).
- **Think-continue strategies** keep the draft, add lines, and DO NOT close
  the tag, they append a marker sentence and let the model continue
  thinking after it (requires the pass-2 guard, Sec. 2.7).
- **Restart-from-scratch strategies** (memory) replace the whole stream
  with `memories + question + "now let me think about the answer."`, the
  model re-reasons from the memories as its opening thought.

The replacement text is encoded again, while the engine reuses the KV prefix that remains valid. Two
constraints are especially important when writing a replacement:

- The replacement text for a restart must NOT include an opening think tag
, the prompt already carries one (the `@think` template pre-opens the
  block), and a doubled tag corrupts the next think-close detection.
- Every `\n` inside the replacement text must be the real escape (the
  SectorLogo reader converts `\n` in string literals to a linefeed; before
  the Aug 2026 reader fix this silently produced literal `n`s).

#### 2.4 `v7_execute` and the capture boundary

External retrieval tools are started through the process-execution interface implemented in
`src/asm/execute.inc`. The relevant entry points are:

```c
char* v7_execute(char *exe, char *p1, char *p2, char *p3);
```

Spawns `exe` with up to three CLI arguments, captures the child's stdout into the shared 1 MiB BSS
buffer, waits for exit, and returns the NUL-terminated capture. Implementation (Win32 pipe pattern):
`CreatePipe` -> strip inheritance from the read end -> build a bounded flat command line (arguments
double-quoted, total capped at `V7_CMDLINE_LAST` so an overlong think query truncates instead of
corrupting BSS) -> `CreateProcessA` with `STARTF_USESTDHANDLES` (10 args, the one direct import call in
the file; everything else uses `API_CALL` ) -> parent closes its write end immediately (otherwise no EOF
ever arrives) -> `ReadFile` loop -> `WaitForSingleObject` -> close, NUL-terminate, return.

**The capture boundary, the single most important tool contract:** the child's stderr is merged into the
same captured buffer. A tool that prints ONE diagnostic byte to stderr injects that byte into the middle
of the model's think stream. This happened (experiment T7 run 1: a score line caused a "catfish"
hallucination) and is why EVERY tool called from a sidecar writes diagnostics to a log file next to the
exe and NOTHING else -- `ragg.log`, `web_ragg.log`. This is a standing rule for any new tool.

Because the formatted execution helper accepts three arguments, callers must account for the following
restrictions:

- Multi-option tools pack everything into ONE options string
  (`bm25_rag.exe search Q "rand=N;maxlines=N;..."`), semicolon
  separated `key=value` pairs.
- Config that does not fit (paths) moves to an INI next to the exe
  (`web_ragg.ini`), or into the format string of `DefineCommand`
  (the full distillation call is one literal command line).

`v7s_readfile(path)` / `v7s_writefile(path, text, append)` share the same 1 MiB buffer; scripts use them
for small state files. File-based sidecar state is discouraged, it races the rewrite engine. Use a
MicroDB flag for per-turn state instead.

#### 2.5 The SectorLogo FFI (`CallLibrary`)

The FFI implementation is defined in `src/asm/sectorlogo_ffi.inc`. A call descriptor has the form `[ DLL_NAME PROC_NAME RET_TYPE [ARG1_TYPE...] ]`, where the type descriptors are quoted lowercase strings: `"integer"`, `"pointer"`, `"text"`, and `"void"`.

```logo
CallLibrary ["SELF" "v7_execute" "pointer" ["pointer" "pointer" "pointer" "pointer"]] "examples/rag/bm25_rag.exe" "search" :Q []
```

- `SELF` resolves the procedure inside v7's own DLL (module handle captured
  in `DllMain`), the script calls the engine it is running in, no path
  coupling.
- `HOST` resolves against the host EXE (requires
  `v7s_set_caller_hmodule(GetModuleHandle(NULL))` first; worked example in
  `examples/c_callback_example.c`).
- Direct DLL loads also work: `[ "kernel32.dll" "lstrlenA" "Integer" ["Pointer"] ]`
 , the memory sidecar uses this for real string lengths (`str_len`).

Each FFI invocation uses a separate stack frame, and the invocation routine preserves the nonvolatile
registers required by the Windows x64 ABI. A SectorLogo expression can therefore make several native
calls without sharing one call frame.

#### 2.6 DB primitives and engine exports used by sidecars

| Export | File | Used for |
| --- | --- | --- |
| `v7_execute` | `execute.inc` | run external tools (all systems) |
| `v7s_readfile` / `v7s_writefile` | `execute.inc` | small file I/O from scripts |
| `v7s_get_user_prompt` | `chat.inc` | the current turn's user prompt (query anchoring, distillation input) |
| `v7s_sidecar_refresh` | `sectorlogo_main.inc` | Check `logo:sidecar` at chat entry and update the parsed script when its contents change; internal, non-exported routine. |
| `EndToken` value | exposed to scripts | EOS detection for memory formation |
| `DatabaseSet`/`DatabaseGet` | `sectorlogo_v7prim.inc` | pass-2 flags and per-process sidecar state |

`v7s_get_user_prompt` returns the raw prompt pointer published by the active `v7s_chat` call (which
`v7s_generate` also runs internally, so generate turns publish the generate prompt too). It is NULL only
before the first turn; it is never an empty string on its own.

#### 2.7 Hook timing subtleties (load-bearing)

**The callback runs before the candidate token is accepted.** When `:Piece` contains `</think>`,
`GetStream` returns the text accepted up to that point but does not include the closing marker, because
the marker is still pending. If the model generated an opening tag, that tag and the intervening text
are present; if the template supplied the opening tag during prefill, it is not part of the generated
stream.

This timing affects the replacement text. An append-style script must include the closing marker itself,
for example `:S _ "\n" _:R _ "\n</think>"`, whereas a script intended to continue the reasoning section
omits that marker. The callback only receives tokens produced by the generation loop. Template and
prompt tokens are processed during prefill, before `v7s_stream_init_turn`, so a non-thinking template
that contains a prefilled empty reasoning section does not trigger the retrieval callback.

**Pass-2 protection (two-phase sidecars).** A rewrite that keeps the think block open causes the model
to eventually emit a NEW `</think>`. That second close must be accepted as-is, injecting again would
loop forever. Two mechanisms exist, both analyzed later:

- **Marker probe** (static/production): `Search "reviewed the reference
  facts":S`, if the marker the script appended is already in the stream,
  this is pass 2; do nothing.
- **DB flag** (memory): `DatabaseSet "mem_rag_done" "s"` after the rewrite;
  on the next close, check `IsNumber Search "s" DatabaseGet "mem_rag_done"`.

> Why `Search` and not `Equal` for the flag: SectorLogo `Equal` compares
> atom offsets for atoms; atoms re-interned by `DatabaseGet` do not compare
> equal to literal atoms. `Search` does substring matching
> on content and is the reliable sentinel test. The first file-based flag
> ALSO introduced a concurrency conflict with the rewrite operation, the DB flag is faster and without a concurrent update conflict.

**EOS hook ordering (memory systems).** The EOS hook executes while the final token is still the
candidate; `GetStream` at that point contains the whole turn (think + answer, minus the EOS token
itself). The hook distills and stores, then lets the EOS through unchanged, generation ends normally.

---

### BM25 retrieval implementations

The repository provides two BM25 implementations that follow the same scoring model: an external C++
utility, `bm25_rag.exe`, and the in-process SectorLogo script `herbivorous_bm25.logo`. They differ in
the surrounding implementation and input-handling facilities, so comparisons should be made with
equivalent tokenization, corpus preparation, and scoring options. The following sections document each
implementation's interface and the relevant experimental settings.

#### 3.1 `bm25_rag.exe`, the production BM25 tool (`examples/rag/bm25_rag.cpp`)

The `bm25_rag.exe` command-line interface provides `search`, `remember`, and `add` operations. Search
configuration is passed as a semicolon-separated options string:

```
bm25_rag.exe search "<query>" "[rand=N;][maxlines=N;][minscore=S;][topmin=S;][base=PATH;][dedup=S;][recency=S;][recent=N;][xbase=PATH]"
bm25_rag.exe remember "<fact line>" "[dedup=S;][base=PATH]"
bm25_rag.exe add "<text>"                      -> appends paragraph to corpus
```

By default, the tool locates `ragg_base.txt`, `ragg_corpus.txt`, and `ragg.log` beside the executable.
The `base=PATH` option selects a different corpus, allowing a single executable to serve multiple
independently configured document collections.

**Tokenization.** The tool lowercases alphanumeric word sequences, removes single-character terms, and
excludes terms in an English stopword list of approximately 100 entries. It then adds a `w1_w2` bigram
for each neighboring pair of retained words. Because the stopword pass runs first, bigrams may connect
words that were separated by an excluded term: for example, “cats do not eat meat” produces `cats_eat`
and `eat_meat`; this provides additional phrase-matching signals for longer queries.

**Chunking.** The tool separates paragraphs at blank lines and identifies sentence boundaries where `.`,
`!`, or `?` is followed by whitespace. It merges fragments shorter than 20 characters with the preceding
chunk. Sentences longer than 280 characters are divided at the last suitable word boundary. The first
sentence in each paragraph receives the `topic` flag, which can influence its score.

**Scoring.** Candidate passages receive a BM25 score using `k1=1.2` and `b=0.75`. Two additional
multipliers account for query-term coverage and paragraph position, as shown in the equations:

```
idf(t)     = ln(1 + (N - df + 0.5) / (df + 0.5)), floored at 0
tfc(t,i)   = tf * (k1+1) / (tf + k1*(1 - b + b*|i|/avglen))
cov(i)     = distinct query UNIGRAMS matched in i / distinct query unigrams
score(i)   = sum over t of idf(t)*tfc(t,i) * (0.3 + 0.7*cov(i)) * (topic(i) ? 1.15: 1)
```

The implementation also suppresses corpus-wide terms. When the document-frequency ratio `df/N` exceeds
`0.4`, it sets `df = N`, reducing the corresponding inverse-document-frequency contribution to
approximately zero.

The coverage multiplier reduces the advantage of a passage that repeats one query term many times
without addressing the rest of the question. The topic multiplier gives a modest preference to
paragraph-opening sentences, which often summarize the subject of the paragraph in the example corpora.
Both adjustments are application-level ranking choices built on top of the BM25 score, not requirements
of the BM25 formula itself.

**Selection.** The tool first retains up to ten passages with nonzero scores. With `rand=0`, it returns
the requested number of passages in descending score order. When `rand` is nonzero, it keeps the
highest-ranked passage and selects the remaining passages using rank-weighted sampling. The requested
output size is usually between three and five passages.

**Near-duplicate suppression.** With `dedup=0`, passages that share the same first 60 characters are
treated as duplicates, and only one is returned. A nonzero `dedup` value selects token-set comparison
using Jaccard similarity instead. The latter can detect passages that repeat much of the same wording
without having an identical prefix.

**Output contract.** Search results are emitted as plain text lines without scores, bullets, or
separators, because the sidecar inserts that output directly into the generated reasoning context. The
tool records command activity and shortened text excerpts of up to 80 characters in `ragg.log`.

**Indexing behavior.** The tool recomputes its BM25 statistics for each invocation rather than
maintaining an index file. On the recorded approximately 4 MB corpus containing 31,000 chunks, queries
completed in less than a second. This cost applies to retrieval invocations, not to every generated
token.

#### 3.2 The production options surface

The following options configure retrieval and memory operations. Each option is supplied in the `search`
or `remember` options string; the table explains the default and how the option changes execution:

| Option | Default | Effect |
| --- | --- | --- |
| `rand=N` | 0 | rank-weighted random pick mode |
| `maxlines=N` | 3 | server-side output line cap (1-20). Exp 15: injecting more lines than the answer has slots invites **slot-stuffing**, unused RAG lines become hedged fake reasons in the answer |
| `minscore=S` | 0 (off) | **density floor**: topscore / distinct query unigrams. Protects small corpora from query-length inflation (quantum-think vs cat KB: absolute topscore 2.6 looks fine, but 0.021/term says "no real hit"). Exp T7 |
| `topmin=S` | 0 (off) | **topscore floor**: absolute rank-1 score. Protects long thinks on large corpora where density collapses for TRUE hits (David/Goliath: density 0.45, topscore 34 vs miss 12-19). Exp 17 |
| `base=PATH` | (next to exe) | knowledge base override, one exe, many corpora (exp 18) |
| `dedup=S` | 0 (off) | near-duplicate suppression threshold via **token-set Jaccard** (0.5 typical for memory corpora); 0 = legacy 60-char prefix |
| `recency=S` | 0 (off) | position-proportional score bonus, later chunks are newer memories, corrections outrank the entries they supersede (exp 20, H3) |
| `recent=1` | 0 | **recent-memory fallback**: a below-floor or empty match on a memory corpus returns the corpus tail (dedup'd), "what was recorded earlier?" queries share zero terms with the corpus; BM25 cannot match them (exp 20, H2) |
| `xbase=PATH` | No default path | **cross-corpus exclusion** (exp 23): chunks near-duplicating lines from the exclusion base are dropped BEFORE ranking, so a fact stored in two stores is emitted only from the store it belongs to |

**The dual answer-presence floor (exp 17, Law 9).** Injection passes if `density >= minscore` OR
`topscore >= topmin`; suppressed only when BOTH miss. No single length-normalized metric separates hit
from miss across query AND corpus regimes, measured:

- cat KB, cat think: density 0.028-0.048, quantum think 0.021-0.024 ->
  floor 0.026 separates them (thin, but both error modes are cheap).
- bible KB (31k chunks), David/Goliath think (1440 chars): density 0.449
  (indistinguishable from misses) but topscore 34 vs miss 12-19 -> floors
  6.0 / 25.

**Jaccard near-dup suppression (`line_jaccard`).** Unigram sets only (bigrams skew the ratio low for
subset-duplicates); pure-number tokens dropped (list numbering is not a fact); a naive trailing-`s` stem
unifies person-shifted verbs (enjoy/enjoys). This catches the memory distiller's cosmetic drift (v7/V7,
first/third person) that prefix comparison misses (exp 20, H1).

**`remember`, Mem0-style NOOP-aware append (exp 22).** Appends a line to the target corpus only when its
Jaccard against every existing line is below the dedup threshold. This is the ADD/NOOP half of Mem0
conflict resolution; UPDATE is approximated at read time by the recency prior.

**Normalization (`normalize`).** Corpus AND query are lowercased, `*` / `**` markdown stripped, `-`
stripped except inside hyphenated words, `!` -> `.`, whitespace collapsed. Without this, retrieval
quality collapses on markdown-carrying think text (memory phase lesson).

**Output cleaning (`clean_line`).** Strips `- ` /`* `/` ** ` list markers, ` **bold**` markers, trailing
whitespace, blank lines, and trailing list-numbering residue from mid-chunk splits (distiller entries
like "1. fact one. 2. fact two." chunk into ". 2." tails). The think stream carries only clean sentence
text.

**stderr rule.** All diagnostics go to `ragg.log` only (Sec. 2.4). The `FLOOR...` suppression line and
every search result are auditable there.

#### 3.3 The SectorLogo-only BM25 (`herbivorous_bm25.logo`, 379 lines)

The entire retrieval pipeline, normalize, tokenize, unigram+bigram terms, stopwords, df/IDF with
auto-stopword, BM25 k1=1.2 b=0.75, coverage multiplier, topic bonus, dual floor, top-3 pick, implemented
in SectorLogo interpreter primitives. This example runs without starting an external process or using C++ code. Scoring is numerically
identical to bm25_rag.exe (same constants, same formulas, including the `df/N > 0.4 -> df = N`
auto-stopword rule via integer arithmetic `(:d * 10) > (:N * 4)` ).

Why it exists: it proves the SectorLogo language is complete enough for real algorithms, and it removes
the process-spawn from small-corpus RAG (intended scale: inline corpora of ~5 KB; see `retrieval.md` ).

The SectorLogo BM25 example uses several language patterns that are documented more fully in the
development handbook:

- **Hash-keyed sorted alists maps.** `FindItem`/`FindFirstAtLeast` require NUMBER
  keys, so string terms are keyed by `HashNumber` of the atom;
  collisions resolve by walking the equal-hash neighborhood and comparing
  atoms exactly (`qhash`).
- **Prepend builds lists in REVERSE**, every Prepend-built list feeding a
  binary search must be `Reverse`d first; the first version forgot this on
  the dedup walk and probes failed silently and randomly.
- **Stopword test via padded Search**: a word is a stopword iff
  `Search (" " _ w _ " "):SW` returns a number, space guards prevent
  substring matches (`cat` inside `implications`).
- **Native tokenize + n-grams**: `Words` extracts the word list in
  one native call (lowercase, alnum runs, len >= 2, ~4.3x the whole
  tokenize stage vs the per-char walk it replaced) and `NGrams`
  builds the adjacent word-pair windows in one more native call; only
  the stopword drop and the `"w1_w2"` join stay in Logo.
- **Corpus as string constants**: the 8-paragraph herbivorous-cats KB is
  inlined with `Make "P0"...` and concatenated with `_` (the string-concat
  infix operator).

Cost profile: after the Words + NGrams rewrite the per-search pipeline is ~2.9x faster end-to-end (e2e
~102 ms vs ~302 ms at 30-chunk scale, answers identical); fine to ~10 KB corpora, too slow beyond ~100
KB without a pre-built microdb index.

---

### Extractive selection

The BM25 retrieval examples rank passages that have already been divided into chunks. Web retrieval
presents an additional problem: fetched pages can be much longer than the context budget, even when the
correct page has been selected. The web pipeline therefore performs extractive selection before passing
text to generation. It uses maximal marginal relevance (MMR) to select passages that balance query
relevance with redundancy among the chosen results.

#### 4.1 MMR passage selection (`examples/rag/web_ragg/text_shortener.cpp`)

The `shortenTextQuery(content, maxChars, query)` function selects sentences through a greedy Maximal
Marginal Relevance procedure. It balances query relevance against overlap with passages already
selected:

```
base(i) = (1.0*qrel(i) + 0.25*pos(i) + fact(i)) * lpen(i)
mmr(i)  = I>>*base(i) - (1-I>>)*max_j-?picked Jaccard(sig(i), sig(j))    I>> = 0.6
```

- `qrel`, fraction of the query's significant words (non-stopword, len>2)
  present in the sentence (query focus).
- `pos`, lead bias, `max(0, 1 - i/20)`: web writing front-loads facts.
- `fact`, +0.3 when the sentence contains a digit (numbers/dates are
  fact-bearing).
- `lpen`, length penalty: stubs (< 40 chars) and run-ons (> 400) are
  down-weighted.
- The Jaccard penalty diversifies: the second pick is penalized for
  overlapping the first, so a page's boilerplate repetitions do not fill
  the excerpt.

Picked sentences are emitted in ORIGINAL document order (better reading flow for the model), capped at
`maxChars`, with a `[..]` marker when content remains. With no query signal, selection degenerates to
lead-bias (first sentences until the budget).

#### 4.2 The HTML content pipeline (`html_parser.cpp`, `webtool.cpp`)

The `parsePage(html, url)` function extracts a `PageInfo` record containing the page title, snippet,
main text content, and discovered links:

1. Title extraction; meta description / `og:description` as snippet.
2. `findContentArea`, heuristic main-content localization.
3. Block removal for `script/style/noscript/nav/header/footer/svg`.
4. Class-based removal for sidebars, related, comments, menus, banners,
   ads (class-substring matching against a known-boilerplate list).
5. `cleanText`, tag stripping, entity flattening, whitespace collapse.
6. Link extraction (used by browsing agents, not by RAG distillation).

Pages whose extracted content is < 200 chars are rejected as "too thin" (the fetch loop skips them and
takes the next search result). The v7 rule `[` and `]` are invalid inside SectorLogo string literals
originated here: the `[Source N]` labels in the distillation prompt are built in C++, not in Logo.

---

### 5. System I: Static RAG (`sectorlogo_rag.logo`)

The static RAG example uses a fixed knowledge base and performs at most one insertion for each question.
It retrieves the relevant passages from that corpus and appends the selected material to the generated
context according to the script's documented policy. This is the simplest complete example of retrieval
integrated with a V7 generation callback; the later systems vary the timing, selection, or persistence
of the retrieved text.

```logo
; wrapper (defined once at the top of the sidecar):
; DefineCommand "rag_search" [:query] "examples/rag/bm25_rag.exe" "search \"%s\""
IfElse (= :Piece "</think>") [
  Make "S" GetStream
  Make "Q" Replace :S "<think>" ""
  Make "R" rag_search :Q
  IfElse (= :R "") [] [
    Make "A" :S _ "\n" :R _ "\n" _ "</think>"
    Print "=== RAG QUERY ==="
    Print :Q
    Print "=== RAG ANSWER ==="
    Print :R
    Print "=== SETTING STREAM ==="
    Print :A
    SetStream :A
  ]
] []
```

(The script uses the literal think-tag strings, see `src/logo/sectorlogo_rag.logo`; the wrappers are
`DefineCommand "rag_search" [:query] "examples/rag/bm25_rag.exe" "search \"%s\""` and its negative-only
variant `src/logo/sidecar_rag_negate.logo`.)

The following explanation follows the relevant implementation statements in source order:

| Line | Meaning |
| --- | --- |
| `IfElse (= :Piece "</think>")` | fire once, at think close; every other token is a no-op |
| `Make "S" GetStream` | snapshot: opening tag + full thought, closing tag NOT included |
| `Make "Q" Replace :S "<think>" ""` | strip the opening tag -> the query is the RAW THOUGHT (Law 2/8: automatic query expansion) |
| `Make "R" rag_search :Q` | run bm25_rag via the DefineCommand wrapper, capture 3-5 knowledge lines |
| `IfElse (= :R "") []` | empty answer (no match / tool missing) -> do nothing; never crash, never rewrite |
| `Make "A"...` | augmented stream: thought + LF + lines + LF + `</think>`, the script re-closes the tag the candidate was about to write |
| the Prints | observability contract: what was asked, what came back, what will be SET |
| `SetStream :A` | rewrite: KV rollback -> re-encode -> re-prefill -> resume |

Forward slashes are required in the exe path. Only \" , backslash-backslash, and \\n are recognized
escapes in a Logo string literal; any other backslash is silently dropped, so the backslashed spelling
of "examples/rag/bm25_rag.exe" would collapse into one run-together word with no separators. Win32
accepts forward slashes.

The empty early think pairs the chat template produces never trigger the pipeline: empty stream ->
retrieval returns nothing -> the empty-answer bail skips the rewrite.

Usage:

```bat
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db ^
  -i "what is a cat?" -n 1000 -q8 1 -rewrite -script src\logo\sectorlogo_rag.logo
```

`-script` installs the callback through `logo:sidecar`; `-rewrite` remains an
optional compatibility flag. `-n` has to cover the original think block, the re-prefill
and the answer. Section 144 sets 1000 as the floor, and a truncated answer is the most common false
"RAG didn't work" report.

---

### 6. System II: The research strategy sidecars

The `examples/research/` directory contains the sidecar programs used in the recorded retrieval
experiments. Each script implements a particular policy on top of the same generation-control interface.
The following catalog identifies those policies and associates them with the recorded measurements.
These files are research examples: the results describe the stated prompts and models, and do not
establish that a particular strategy will be preferable for every corpus or application.

| Sidecar | Strategy | Measured result |
| --- | --- | --- |
| `sidecar_replace_think.logo` (+`gated_replace_think`) | **Replace**: stream:= retrieved lines only, tag closes right after | largest raw injection of the set; wrong prior vanishes; answers shallow (~100-280 tok, ~7 tok/s); off-topic hazard (the quantum probe collapsed to ~220 generic tokens, with wave-particle duality and uncertainty absent) |
| `sidecar_negate_only.logo` | **Negate**: append only lines labeled `CONTRAINDICATION:` | moderate absorption of compatible facts; contradictions NOT resolved, under the adversarial corpus: split-brain (headline keeps the prior, RAG line pasted verbatim in a subordinate bullet) |
| `sidecar_fact_prepend.logo` | **Prepend**: `KNOWN FACTS:` + lines before the model's own reasoning | smallest observed effect in this test; the model read the facts, then asserted the direct contradiction anyway |
| `sidecar_multi_query.logo` | **Multi-query**: two retrievals (full think + bare question), concatenated | more coverage (the short query surfaced "obligate carnivore"), but duplicates appeared and answers did not measurably improve, so coverage was not the limiting factor |
| `sidecar_think_erase.logo` | **Erase**: close the tag early and inject no data for the control run | no factual change; the model answered from its existing draft and repeated its prior's mistakes |
| `sidecar_gated_replace_think.logo` (v1-v3) | **Keyword gating** of replace vs append | v1 gated on the answer's domain words and always opened; v2 fixed that but `Search` matched substrings; v3 used space-padded keywords and gated on the query. It remained vocabulary-dependent |
| `sidecar_three_tier_gated.logo` | **Three-tier**: on-topic prompts replace; recognized off-topic prompts inject no data; unknown prompts append | correct decisions across the probes; the dogs probe showed no contamination, but the gate remained vocabulary-dependent |
| `sidecar_think_continue.logo` (11a) | **Weak marker**: append lines and "let me think about that", then continue thinking | the rewind and continuation mechanism worked, but the second pass was nearly empty because the marker read as completed deliberation |
| `sidecar_think_continue2.logo` (11b) | **Strong marker**: "the key points I missed are:" | the model performed a second reasoning pass, but when nothing had been missed it confabulated a missed point ("cats are rodents in the family Felidae") |
| `sidecar_think_continue3.logo` (11c) | **Neutral-delta marker**: "I have reviewed the reference facts above. Let me now finalize my answer plan." | largest on-topic effect in this experiment: 423 tokens at 20.24 tok/s, with facts integrated and no observed confabulation; the off-topic probes showed no contamination in these runs |
| `sidecar_answer_gated_continue.logo` | answer-presence gate (question-head keywords in retrieved lines) on think-continue | false positive on polysemy ("how long do cats LIVE" matched "feral cats LIVE in colonies"); with the think-continue substrate the wrong decision cost only wasted tokens, substrate robustness subsumes gate accuracy |
| `sidecar_hybrid_half_replace.logo` | half the draft kept, half replaced | intermediate; the 11c result was better |
| `sidecar_production*.logo` (t2,t3,t4v2,t5,final) | production tuning iterations, see Sec. 7 | final adopted |
| `sidecar_test_eos{,2,3}.logo`, `sidecar_test_getprompt.logo` | infrastructure probes (EOS detection, `v7s_get_user_prompt` wiring) | verified the hooks used by the memory system |
| `examples/research/herbivorous/hb_sidecar_*.logo` | the strategy set re-run against the sealed herbivorous-cats corpus folder | the adversarial override table (exp 13) |

The recorded evaluation used three classes of query and corpus material: cat-related questions that
matched the knowledge base, a fictional herbivorous-cat corpus that contradicted common background
assumptions, and unrelated questions about quantum physics, dogs, and France. This separation helped
distinguish useful retrieval from irrelevant insertion.

#### 6.2 The ten design criteria

The research log records the following observations. Each conclusion applies to the specified tests and
should be read alongside the recorded inputs and output transcripts:

1. **Exclusivity**, knowledge changes beliefs most reliably from an exclusive or high-influence position in the think stream.
2. **Query richness**, the think text is automatic query expansion.
3. **Prior quarantine**, appended contradiction is quarantined, not
   resolved; the prior keeps the headline.
4. **Substrate determines damage**, the strategy, not the gate, decides
   what a bad retrieval or a wrong gate decision costs.
5. **Framing cuts both ways**, marker text decides between integration
   (neutral-delta), emptiness (weak marker), and confabulation
   (deficiency-framing).
6. **Contradiction amplification**, binary questions naming the contested
   predicate ("are cats carnivores?") OSCILLATE under deliberation
   substrates: the second pass plans both beliefs and the answer alternates
   sentence by sentence. This limitation was known and remained unresolved in the recorded experiment.
7. **Epistemic framing legitimizes both sides**, "I remember that... and
   that is what I know for certain" framing produced the highest factual-KB
   integration in these runs and legitimized the model's own contradictory
   claims under the adversarial corpus.
8. **Think text = query expansion**, Law 2 restated for retrieval: the
   expansion that helps correct thinkers retrieves confirming chunks for
   wrong ones (see Law 10).
9. **No universal floor metric**, absolute scores scale with query length;
   density collapses on long true hits; use BOTH floors, calibrated per
   corpus.
10. **Query poisoning**, think-text expansion amplifies the model's wrong
    prior into retrieval: the David/Goliath run retrieved GENEALOGY verses
    because the think text supplied "son of Solomon"/"Book of Daniel"
    vocabulary. RAG cannot rescue the prior it is built from. Mitigation:
    anchor the query with the USER PROMPT (production sidecar does this).

#### 6.3 Off-topic response quality assessment

Mandatory column for every strategy (research rule): each sidecar was also run with prompts unrelated to
the KB ("explain quantum physics", "tell me about dogs", "what is the capital of France?"). Findings:

- Replace-family strategies are a correctness hazard off-topic (the
  quantum collapse).
- Think-continue (11c) passed ALL off-topic probes at baseline quality --
  irrelevant lines were read and passed over. This asymmetry (harmless
  when wrong, integrating when right) is why 11c became the production
  substrate.
- Gating cannot rescue a fragile substrate (Law 4); it only saves tokens.

---

### 7. System III: The production sidecar

The `sidecar_production_final.logo` example combines the configuration choices selected during
experiments 1 through 17. It uses the think-continue approach, retrieves using the user's question
together with generated text, and applies the recorded scoring threshold before inserting passages. The
details below specify the options and the reasons they were selected. The term *production* is part of
the source filename and does not imply that the experimental measurements are general reliability
guarantees.

- **Substrate:** think-continue, neutral-delta marker (11c).
- **Injection:** bare reference lines, no memory-voice (T4 risk, Law 7).
- **Query:** USER PROMPT + full think text (`v7s_get_user_prompt` export;
  T5 + exp 17 Law 10: the user phrasing anchors the query against
  think-text poisoning when the model's prior is wrong).
- **Server side:** `bm25_rag.exe search Q "maxlines=3;minscore=0.026"`
 , maxlines=3 per exp 15 (slot-stuffing), density floor per T7 (cat
  thinks 0.028-0.048 vs quantum 0.021-0.024; both error modes CHEAP:
  a suppressed cat answer falls back to parametric knowledge, an injected
  quantum line is ignored by the substrate).
- **No relevance gate** (exp 12), no client-side trims (T1 bug, the
  server caps).
- **Two-phase:** pass 1 injects lines + marker, think NOT closed; pass 2
  detects the marker via `Search "reviewed the reference facts"` and accepts
  the tag.

Final matrix (iteration 19): cat on-topic PASS (3 lines integrated, 369 tok, 16.5 tok/s); quantum
borderline-density PASS (floor crossed, substrate held, clean pure-physics answer, the designed
two-layer defense); herbivorous override PASS with the known Law-3 tail. **The static stack was declared
release-ready.**

---

### 8. System IV: Persistent memory RAG (v1 through v6)

The persistent-memory examples are stored under `examples/rag/memory/`; they demonstrate how an
application can retain selected information between separate inference executions. At the end of a
turn, the application prepares a textual record, extracts candidate facts, and stores suitable entries
in retrieval corpora. On a later invocation, the script can search those corpora and incorporate
relevant records into the generation context. This state belongs to the retrieval application rather
than to the language model's weights. A new `inference_example.exe` invocation does not retain the
previous execution's model state by itself; the application uses the separate stored corpus to
retrieve selected material across invocations.

#### 8.1 Architecture and the two hooks

```
HOOK A, think-close (</think>), pass 1:
    P  = v7s_get_user_prompt()
    S  = GetStream                      , draft think text
    Q  = P + "\n" + S minus opening tag , rich query
    RU = bm25_rag search Q (profile corpus: recency + recent=1)
    RA = bm25_rag search Q (knowledge corpus: floor + xbase)
    (v1/v2: single corpus, maxlines=8;minscore=0.015)
    empty -> accept tag, no restart
    else  -> A = framing header + lines
                  + "the user asked: \"P\"."
                 + "now let me think about the answer."
     rewrite:    SetStream A                  , restart from scratch
                 DatabaseSet "mem_rag_done" "s", pass-2 flag

HOOK A (pass 2): flag contains "s" -> accept the tag, clear the flag.
                 The think block the model wrote AFTER reading the
                 memories becomes final.

HOOK B, EOS (:TokenNumber == EndToken):
     full   = GetStream                   , whole turn, think+answer
     strip think tags (content kept)
     turn   = P + " -> " + stripped
     v7s_writefile "examples/rag/memory/mem_input.txt" turn
     thought = mem_distill()  , DefineCommand wraps llm_text_processor.exe
                              , -distill with the prepend/append prompts
                              , -> 3 tagged factual lines
     route each line by USER:/SELF:/NOTE: tag:
         USER -> user_facts.txt   (grounding-guarded, dedup'd)
         NOTE -> ragg_base.txt    (grounding-guarded, dedup'd)
         SELF -> discarded        (the assistant must not memorize itself)
     EOS token passes through unmodified
```

#### 8.2 Sidecar evolution v1 -> v6

| Version | Change | Reason |
| --- | --- | --- |
| v1 | think-continue over a growing corpus | worked, but the model's draft interleaved with memories confused attribution at 0.6B |
| v2 | **restart-from-scratch**: replace the whole stream with memories + question + continuation cue | separation between recalled text and new reasoning; framing tuned live over a 7-turn conversation (Sec. 8.4) |
| v3/v4 | normalization + output cleaning move server-side (bm25_rag gains `normalize`, `clean_line`, maxlines cap) | retrieval of the right lines became reliable only after markdown-stripping |
| v5 | recent-memory fallback (`recent=1`) + Jaccard dedup (`dedup=0.4`) | "what do you remember about me?" shared ZERO terms with the corpus, BM25 cannot match it; the fallback returns the corpus tail (what a person recalls when asked out of the blue) |
| v6 | **dual-corpus split** + source grounding + recency prior | profile facts (user) and knowledge facts (world) need different floors and different fallback policy; distiller hallucinations need a generic write guard |

The following options are taken from the version 6 memory-recall script. They illustrate how separate
corpora can use different retrieval thresholds and selection policies:

```logo
; profile: recency + fallback, NO floor (a profile always has something)
"maxlines=4;dedup=0.4;recency=0.4;recent=1;base=...user_facts.txt"
; knowledge: calibrated floor + xbase exclusion, NO fallback
"maxlines=3;dedup=0.4;xbase=...user_facts.txt;minscore=0.026;base=...ragg_base.txt"
```

`xbase` prevents double emission of a fact stored in both stores; the knowledge corpus's
`minscore=0.026` suppresses off-topic retrieval while the profile corpus deliberately always answers (a
person always has a profile).

#### 8.3 Memory distillation (`llm_text_processor.exe -distill` via `DefineCommand`)

The memory scripts invoke the distillation utility through a SectorLogo `DefineCommand` wrapper:

```logo
DefineCommand "mem_distill" [] "bin/llm_text_processor.exe" "bin/v7_threading.dll weights/databases/qwen3_0.6b/lz4ht.db x x -distill -maxgen 120 -finputtext examples/rag/memory/mem_input.txt -fprepend examples/rag/memory/prepend.txt -fappend examples/rag/memory/append.txt"
```

The EOS hook writes the turn text to `mem_input.txt` with `v7s_writefile`, then calls `mem_distill`.
`DefineCommand` builds the full many-parameter command line from a format string, and
`llm_text_processor.exe -distill` prints the distilled tagged lines on stdout. Diagnostics go to a log
file next to the executable because `v7_execute_fmt` captures both output streams. Swapping the prompt
files is the prompt experiment.

The following prompt templates were selected from the recorded 25-run command-line evaluation; they
are included to make the test configuration reproducible:

- `prepend.txt`: "Extract the key facts, names, and specific details...
  who, what, where, when, why. Preserve all proper nouns and numbers
  exactly. Now the text:>>>"
- `append.txt`: "<<< Write 3 short factual sentences, each prefixed with a
  tag: USER: for facts about the human user..., SELF: for facts about you
  the assistant, NOTE: for other facts. In USER sentences always use the
  user's name, never pronouns... Only state facts that appear in the
  text - NEVER write commentary... If there is nothing for a tag, write
  fewer sentences... Keep each sentence under 20 words."

The USER/SELF/NOTE tag scheme is what makes the store-side routing generic: the sidecar does not
interpret content, it routes by tag.

#### 8.4 Recall framing: the 7-turn tuning conversation

The injection framing was tuned live; three framings failed before the final one (verbatim evidence in
the primary research log, memory phase):

1. "I think I remember that..." + bare lines -> the model CLAIMED the
   user's memories as its own ("I live in Berlin").
2. "I remember the following about the user:" -> fixed self/other, but on
   model-directed questions it roleplayed AS the user.
3. "Things I remember from previous conversations:" + restated question +
   "now let me think about the answer." -> correct attribution both ways
   after one correction cycle. ADOPTED.

Mechanical fixes from the same conversation: strip the retrieval's trailing newline (doubled `\n`
artifacts); end with an explicit continuation cue (an earlier framing looped 1500 tokens without ever
closing the think block); DB pass-2 flag with Search sentinel.

**Correction loop (validated over 7 turns):** wrong entries are never edited out; user corrections are
distilled as NEWER entries, and the recency prior makes them outrank the entries they supersede. Turn
7's direct correction ("YOU do not live in Berlin, those are MY hobbies") produced correct attribution
in all later recalls. Good entries accumulate and take precedence; the corpus is only ever grown.

Known limit: entity attribution at 0.6B remains the hard problem; the framing is a crutch that larger
models may not need.

#### 8.5 Where this sits in the 2026 agent-memory literature

v7's persistent-memory system is a text-only point in a design space that 2026 research is mapping
quickly. The closest published work and what it says about the choices above:

**ReFind** (Li et al., arXiv:2608.12888, Aug 2026) provides one external comparison for the plain-text
approach. It asks how much of agent-memory performance comes from the structure (summaries, embeddings,
trees, knowledge graphs) rather than from competent retrieval over the raw record -- and answers by
building *no* semantic structure: a BM25 index at turn granularity over the unmodified chat archive,
plus an agent-controlled iterative search loop with four chat-native controls (session-aware rank
fusion, A+/-2-turn local context expansion, temporal narrowing, seen-session deduplication). On
MemoryAgentBench (~2,800 questions, GPT-4o-mini backbone) it scores 58.2 mean accuracy, above HippoRAG 2
(53.2), Mem0 (25.1), and every graph/tree/notes system compared, and 93.2/89.3 on LongMemEval-S/M under
GPT-5-mini. Two of its findings map directly onto v7's design:

- *Lexical retrieval held up against dense and hybrid backends under
  identical agent control*, their ablation swapped only the retrieval
  signal and BM25 still had the highest mean, "lexical matching is not the
  bottleneck" when the agent can reformulate. v7's retrieval signal is the
  model's own 100-300-word think text (Law 2/8), which is an even richer
  reformulation source than a ReAct loop's query rewrites.
- *"Faithful storage + controllable search interface first, derived
  structures only for workloads that demand them"*, their stated design
  principle, is the same bet System IV makes, with the corpus as the
  faithful store and the think-close hook as the control point.

Where v7 goes further than ReFind: ReFind retrieves into a *separate reasoning stage* before answering;
v7 splices the retrieved lines into the model's own reasoning stream and re-prefills, so recall and
re-reasoning are one operation. Where ReFind goes further than v7: its temporal filtering, session
fusion, and seen-session dedup are exactly the conversational-structure signals System IV does not yet
exploit, v7's recency prior is the only one of the four implemented. Those three are the potential
follow-up changes, and they are local changes to `bm25_rag.exe` that touch no engine code.

**Related 2026 memory work worth tracking:** ScrubJay-MEM (arXiv:2608.04746) operationalizes
type-conditioned *temporal decay* of stored memories (v7's equivalent is the recency prior, but without
per-type perishability); AuthMem-Bench (arXiv:2608.01679) shows "authority collapse" at the
consolidation boundary, stored claims outliving the authority of their source, which is the write-guard
problem v7's source-grounding check and USER/SELF/NOTE tagging address from the policy side; and the
rate-distortion survey of memory compaction (arXiv:2607.08032) unifies KV-cache window displacement with
agent memory as one keep-or-discard decision, a framing in which v7's KV rollback and its text corpus
are two instances of the same design question, "what may be discarded before the query is known."

---

### 9. System V: Internet RAG (`web_ragg.exe`)

The internet retrieval example is located in `examples/rag/web_ragg/`; it obtains external information
during a generation workflow and makes selected passages available to the model. The corresponding
application manages the network requests, page extraction, and text selection; the transformer does
not independently browse or update its training data. The source files and development notes describe
the command-line interface and the retrieval policy used in the recorded experiments.

#### 9.1 Architecture

```
web_ragg.exe search "<query>" ["pages=5;words=100;maxgen=250;fresh"]
    1. dated_query = query + " " + current month name + " " + year
    2. DuckDuckGo Lite search (lite.duckduckgo.com/lite, no JS)
    3. fetch first N usable result pages (simpledb page cache, `fresh`
       bypasses), parse main content, reject < 200 chars as thin
    4. MMR-compress each page to 1500 query-relevant chars (Sec. 4.1)
    5. v7 distillation: "[Source N]" excerpts become one ~100-word paragraph
       (Q8, thinking off, maxgen 250) with the no-meta prompt
    6. print the paragraph to stdout and nothing else

web_ragg.exe makequery "<think and memories text>"
    v7 distills the reasoning text (capped 4000 chars) into one 3-8-word
    search query; collapse whitespace, strip punctuation, cap 12 words
```

Both v7 DLL and DB paths come from `web_ragg.ini` next to the exe (`[v7] Dll=`, `[Db]=` ); missing INI
or keys is a usage error, no silent defaults.

#### 9.2 Component analysis

| File | Role | Notes |
| --- | --- | --- |
| `search.cpp` | DuckDuckGo Lite HTML scraping | parses `class='result-link'` anchors + `class='result-snippet'`; unwraps the `uddg=` redirect parameter; rejects non-http URLs |
| `http_client.cpp` | page fetch via curl, run through `_wpopen` (the `curlGet` function names are legacy) | 15-20 s timeouts |
| `html_parser.cpp` | `parsePage` -> title/snippet/content/links | block-tag + boilerplate-class removal (Sec. 4.2) |
| `text_shortener.cpp` | MMR extractive selection (Sec. 4.1) | query-relevant 1500-char excerpts per page |
| `text_summarizer.cpp` | legacy LLM summarizer wrapper | kept because text_shortener's header includes it; the runq.dll path in it is unused |
| `simpledb.c` | CRC32-checked flat key-value page cache | `web_ragg.db`; entries validated on load, corrupt entries skipped |
| `web_ragg.cpp` | the pipeline + CLI + output discipline | Sec. 9.4-9.6 |
| `webtool.cpp` | shared fetch/parse front | also dumps fetched HTML to a debug dir |

#### 9.3 `makequery`: think text -> search query

The prompt: "You write web search queries. The text below is a language model's reasoning about a
question... Identify the main factual topic it needs current information about, and write ONE short web
search query (3 to 8 words)... Output ONLY the query words." Post-processing collapses to one line,
strips `.,;:!?`, caps at 12 words.

Observed behavior (dev log steps 3/8): the query distiller anchors on the most-repeated entity
(topically correct, sometimes generic, `new models` at 2 words). **Deliberately NOT rejected for being
short**: the date prior compensates, and a word-count gate would have BLOCKED an excellent result.
Prompt asks for "the main factual topic"; the "the question the user is asking" variant is a recorded
tuning candidate.

#### 9.4 Distillation prompts and the no-meta-language rule

Default search prompt (tuning pass 1):

> "You are a research assistant. The texts below are from web pages about
> one topic. Write ONE paragraph of about 100 words summarizing the
> current situation around the topic. Keep concrete facts: numbers, names,
> dates, places. Write only the paragraph itself - no introduction, no
> heading, no list, no quotation marks, and never mention pages, texts,
> sources, or this task."

The final clause is the **no-meta-language rule**: the first end-to-end run produced a paragraph
starting "The query topic reports...", the model echoed the task framing, and that meta-language would
flow into the main model's think stream. The ban list (pages, texts, sources, queries, the task itself)
removed the class entirely (verified on 3 domains).

Prompt overrides live in the INI (`[search] Prepend/Append/ PrependFile/AppendFile`), the tuning surface
without recompiling.

#### 9.5 The generic date prior

The current month/year is appended to the SEARCH query (not to the MMR relevance matching, which keeps
the user's original query): `"newest LLM releases"` -> `"newest LLM releases August 2026"`.
Content-agnostic, no intent classification, no keyword routing, and it works for any topic. Result: all
verified paragraphs carry strictly current facts (Aug 2026 releases, 2026-27 season dates).

#### 9.6 Output discipline (stdout hijack)

The v7 DLL prints internal diagnostics (`db_load: file =...`) to the PROCESS stdout. In sidecar use,
stdout is the captured stream, those lines would introduce unrelated text into the generated reasoning
context. Countermeasure in `main()`: before all work, `_dup(1)` saves the real stdout and `freopen`s
stdout onto `web_ragg.log`; after the work, `_dup2` restores it and only the result string is printed.
This is the strictest form of the Sec. 2.4 capture rule -- it assumes NOTHING about what loaded code
prints.

#### 9.7 The page cache (`simpledb`)

Flat file `web_ragg.db`: format signature + version + count, then `[keylen][vallen] [key][value][crc32]`
records; CRC over key+len+value detects corruption; individual corrupt entries are skipped on load,
structural corruption returns SDB_ERR_CORRUPT. Keys are `page:` + alnum-flattened URLs (capped 120
chars). Every `page_cache_put` saves synchronously. The cache makes repeat queries cheap and makes
`fresh` (bypass) the explicit way to re-fetch.

#### 9.8 Development log summary

Eight steps, each logged same-session with full inputs/outputs:

1. Component analysis (webtool + llm_text_processor
   distillation pattern) -> copy decision.
2. v1: search -> 5 pages -> paragraph. WORKED; stdout pollution found
   (db_load leak) -> v2 stdout hijack.
3. `makequery` added; anchors-on-repeated-entity behavior noted.
4. SectorLogo constraint: `v7_execute` passes 3 args -> options-string +
   INI design.
5. 3-arg CLI + INI implemented; both modes verified (EV think ->
   "electric car developments"; dated search produced real Aug-2026
   events).
6. First end-to-end sidecar run (exp 29): profile recall -> knowledge
   floor-suppress -> makequery -> web block -> rewrite -> pass 2 -> EOS
   distiller -> grounding guard rejected an ungrounded note. Meta-language
   + echo artifacts + 3.32 tok/s accounting noted as tuning backlog.
7. Tuning pass 1: no-meta prompt + date prior -> 3/3 clean paragraphs.
8. End-to-end with tuned tool (exp 30): the model's answer listed GLM-5.3-
   Flash (Aug 26 2026), Qwen3.8 Flash, Claude Fable 5 export-control
   timeline, things it was never trained on. **Internet RAG v2
   passed end-to-end.**

Known costs: the two web_ragg spawns run INSIDE the generation window at think-close (each loads the 631
MB DB); the generation counter reads 1.9-3.3 tok/s while real generation speed is unchanged.
Resident-server / shared-DB-handle design noted as future optimization.

---

### 10. System VI: Integrated memory + internet (`sidecar_memory_rag_v7.logo`)

This integrated sidecar combines the persistent-memory retrieval policy with an additional internet
retrieval source. It searches the relevant corpora, applies the documented selection rules, and prepares
the retained passages for one stream update. The implementation distinguishes the sources in its
inserted text so that the subsequent generation receives the intended context. The source filename and
option values below identify the recorded version of this example.

Utility layer (reused verbatim from the research corpus):

- `str_len`, FFI `lstrlenA` (the interpreter has no string LENGTH)
- `strip_trail`, drop one trailing `\n`/`\r\n` (doubled-newline artifacts)
- `replace_all`, Repeat + Search + Replace loop (Replace does first
  occurrence only)
- `strip_punct`, trailing sentence punctuation stripper
- `is_grounded`, the source-grounding write guard

Recall hook (pass 1), full order:

1. Query = user prompt + think text.
2. `RU` = profile corpus search (`recency + recent=1`, no floor).
3. `RA` = knowledge corpus search (`minscore=0.026 + xbase`).
4. **Internet block**: `WQIN` = query + (knowledge lines if any) ->
   `web_ragg.exe makequery` -> if a query came back -> `web_ragg.exe search
   WQ pages=5;maxgen=250` -> paragraph.
5. Injection geometry (three labeled blocks + question + cue):

```
Facts about the user (the human I am talking with) from earlier conversations:
<profile lines>

Relevant facts:
<knowledge lines>

the internet knows the following about it:
<web paragraph>

the user asked: "<P>".
now let me think about the answer.
```

6. Gating is purely content-based: the web block exists only when BOTH
   makequery and search produced content; each memory block exists only
   when its search returned lines; internet-only injection (no memory
   hits, web hit) is handled as its own branch. This approach uses neither
   keyword-based routing nor an intent classifier.

7. Pass-2 flag in the DB (`mem_rag_done`), exactly as v6.

Store hook (EOS): unchanged v6 logic, llm_text_processor -distill (via DefineCommand mem_distill), tag
routing (USER -> user_facts, NOTE -> ragg_base, SELF -> discard), the grounding guard on every stored
line, `remember` -with-dedup writes (`dedup=0.4`).

End-to-end verified twice (exp 29 with v1 tool, exp 30 with the tuned tool): the model answers
current-events questions from the injected web paragraph, recalls the user profile, and the grounding
guard rejects hallucinated NOTE lines. The residual 0.6B artifacts (profile echo in the answer opening,
occasional think garble) are documented tuning backlog, not pipeline defects.

---

### 11. System VII: Thought RAG standalone (`llm_text_processor.exe`)

`examples/rag/llm_text_processor.cpp` implements a standalone text-processing utility that also serves
as the distillation component in the persistent-memory examples. The command-line modes select between
retrieving or processing text for an external caller and extracting shorter factual records for later
storage. The documented `-distill` invocation is issued through SectorLogo's `DefineCommand` facility;
the utility remains a separate executable rather than a native kernel in the inference loop. The
documented implementation comprises 579 lines in the recorded snapshot.

The utility supports the following execution modes:

- default: BM25 search over a corpus -> v7 LLM reformulation into "3
  logical simple thoughts"
- `-direct`: skip BM25, reformulate a given text file (`-finputtext`)
- `-only-bm25`: print raw BM25 results with zero formatting (the
  diagnostic mode; with `-direct`, print prepend+text+append plain)

The command-line flags expose the search and generation parameters, allowing a test harness to vary one
parameter at a time while recording the resulting output:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-maxlines N` | 5 | BM25 chunks (1-20) |
| `-k1 F` / `-b F` | 1.2 / 0.75 | BM25 saturation / length normalization |
| `-topic F` | 1.15 | topic-sentence boost |
| `-cov F` | 0.3 | coverage weight base |
| `-bigrams 0\|1` | 1 | bigram features on/off |
| `-maxgen N` | 200 | LLM generation cap (10-500) |
| `-prompt S` | default | Prompt style: `default`, `strict` (exactly three simple sentences without an introduction or a list), or `simple` (summarize the input in three short sentences). |
| `-fprepend/-fappend F` | Not applicable | Paths to the prompt files used by the memory system during tuning. |
| `-foutputfile F` | Not applicable | write thoughts to file |
| `-sweep` | Not applicable | machine-readable tab-separated output line |

Engine wiring (the canonical v7-as-a-tool pattern): `v7_load` -> `v7_init(db)` ->
`v7_initialize_inference` (`v7_set_quantize(1)`, `v7_set_max_gen(cap)`, then `v7_enable_thinking(0)`,
distillation must not think) -> `v7_chat` -> `v7_free_inference` / `v7_unload`. The 25-run CLI sweep
during memory development showed the simple prompt with maxlines=3 beats elaborate schemata, that result
shipped as `prepend.txt` /`append.txt`.

---

### 12. System VIII: The embedded retrieval system

The embedded retrieval example, `herbivorous_bm25.logo`, replaces the external BM25 executable with
indexing and scoring operations written in SectorLogo. Its retrieval policy otherwise follows the static
example: select passages from a fixed corpus, then incorporate the selected text at the configured
callback boundary. The example demonstrates that the database and native string operations can support a
search application in the same process as inference; the scoring and query rules remain
application-defined.

For small demonstrations, the corpus can be stored as inline SectorLogo strings. The recorded
implementation handles approximately 5 KB comfortably, while an unindexed Logo search can take seconds
once the input grows beyond roughly 10 KB. For corpora approaching or exceeding 100 KB, use an index
stored in MicroDB through `DatabaseSet` and `DatabaseGet`, or use an external search utility.

---

### Score thresholds and calibration

The retrieval policy uses two thresholds to decide whether a search result contains enough evidence to
be inserted. The `minscore` threshold applies to the recorded density measure, while `topmin` applies to
the highest-ranked passage score. Their values must be calibrated for the corpus and query construction
method because absolute scores can change with query length and document distribution. The procedure
below records the calibration method used in the experiment.

1. Collect 2-3 known-HIT and 2-3 known-MISS probes per corpus, with REAL
   think-length texts (500-2500 chars), not short questions: every floor
   surprise in the research traces to calibration-text length vs
   live-text length mismatch.
2. Plot density AND topscore for each probe.
3. Set each floor between its hit/miss cluster. Retain both implementation paths in the documented configuration.

Reference calibrations:

| Corpus | density floor | topscore floor | Notes |
| --- | --- | --- | --- |
| cat KB (~40 chunks) | 0.026 | off | cat thinks 0.028-0.048; quantum thinks 0.021-0.024; thin but both error modes cheap |
| bible KB (31k chunks) | 6.0 | 25 | David/Goliath density 0.449 (miss-level) but topscore 34 vs miss 12-19 |
| memory knowledge corpus | 0.026 | off | inherits cat calibration |
| memory profile corpus | none | none | the profile uses a recency fallback; floors apply to knowledge, not identity |

Threshold calibration depends on query length (short first-sentence queries are a different density
regime, exp 17 run 3) and to corpus size. There is no universal pair, that is Law 9, and the reason the
floors are options, not constants.

The relevance threshold determines whether retrieved passages are inserted, while the generation
strategy determines how the model can use the inserted text. These are separate decisions: a passage can
meet the scoring threshold without materially changing the generated answer. The quantum borderline case
(floor crossed, lines injected, substrate ignored them, answer stayed pure physics) is the designed
behavior of that defense-in-depth.

---

### 14. Source grounding and genericity rules

The `is_grounded` rule limits which distilled statements can be written to the persistent corpus. In the
recorded implementation, every content word longer than three characters must also occur as a substring
of the lowercased source turn. This is a lexical source check, not a semantic verification procedure: it
restricts unsupported vocabulary but cannot establish that a statement accurately represents the source.
The original rule and its measured consequences are documented below. The source turn is the
concatenation of the user prompt and generated text, represented in the recorded procedure as `user
prompt -> generated text`.

- template echoes and meta-commentary ("The text mentions...",
  "no details provided", none of those words need appear in the turn)
- placeholder hallucinations (the distiller's own filler)
- foreign-language drift (non-occurring tokens fail the containment)

It supersedes string-based exclusion approaches, which are prohibited by the project’s genericity
requirement. Rule XXb excludes model-specific lists of unsupported generated statements produced on one
prompt fixes that prompt only, the next hallucination uses different words. The generic property
distinguishing bad output is "not grounded in the source turn"; THAT is what gets encoded.

Available retrieval mechanisms in the described implementation:

| Mechanism | Where | Rejects |
| --- | --- | --- |
| source grounding | `is_grounded` write guard | ungrounded distillate |
| calibrated statistical floors | bm25_rag `minscore`/`topmin` | off-topic / weak retrieval |
| similarity-based dedup | Jaccard `dedup=` + `remember` NOOP | duplicate memories |
| recency priors | `recency=` | superseded memories |
| empty-result fallbacks | `recent=1` | vocabulary-miss on identity queries |
| content-based block gating | v7 sidecar (empty = absent) | missing tool / no result |

When revising a retrieval strategy, evaluate whether the change improves behavior across the intended
workload. A result limited to one example is insufficient evidence of a general improvement.

---

### Known limitations and observed failures

| Symptom | System | Cause | Fix |
| --- | --- | --- | --- |
| no `=== RAG... ===` output at all | I-VI | the sidecar script was not installed or was not invoked | pass `-script <path>` and check the `logo:sidecar` key; `-rewrite` is optional |
| `=== RAG ANSWER ===` empty | I | tool missing / no match / below floor | check forward-slash path; run the tool manually with the same query; check ragg.log FLOOR line |
| answer cut mid-sentence, no injected facts | I-VI | `-n` budget exhausted (generation count == -n) | raise `-n` (>=1000; 1500 preferred) |
| injected facts appear, answer still wrong | I | KB lacks negation lines | write the KB to DENY the plausible wrong answers, not just state facts (Lesson 3) |
| facts wrapped in invented story/game | I | model hallucinated a frame in its own think text; the append preserves it | source-attribute the prompt ("according to X,..."), prevention beats correction (Lesson 5) |
| random word garbage in think stream | any | tool printed to stderr/stdout, capture boundary (Sec. 2.4) | send diagnostics to the log file only; check the web_ragg stdout hijack |
| think block never closes (1500 tokens) | IV | injection framing without continuation cue | end with "now let me think about the answer." |
| infinite rewrite loop | IV | pass-2 flag failed | DB flag + Search sentinel (never Equal, never a file) |
| retrieval confirms the model's wrong prior | I, III | Law 10 query contamination | anchor query with the user prompt; first-sentence queries are immune but need their own floor regime |
| incorrect matches on a large corpus | I, III | density floor false-negative on long thinks | add the topscore floor (dual floor) |
| profile facts claimed as model's own | IV | 0.6B attribution + first-person memories | neutral framing + third-person distillation prompts + user corrections |
| YES/NO answer oscillates | II, III | Law 6 binary contested-predicate probe | known limit; open questions override cleanly |
| web paragraph mentions "query"/"sources" | V | meta-language in distillate | no-meta prompt clause (Sec. 9.4) |
| stale web facts | V | page cache | `fresh` option |
| generation counter 2-3 tok/s | VI | web spawns inside the generation window | accounting artifact; optimize later (resident server) |
| stream replays visibly | I-VI | the replay IS the re-prefill | none, expected |

---

### Measured performance

Ryzen 5 7430U, Qwen3-0.6B, Q8 forward:

| Stage | Cost |
| --- | --- |
| ragg search, cat KB (~40 chunks) | < 5 ms |
| ragg search, bible KB (~31k chunks) | ~200-600 ms |
| v7_execute spawn overhead | ~10-30 ms |
| pure-Logo BM25, 5 KB corpus | ~250 ms |
| rewrite: re-encode + re-prefill (~300 tok) | ~300-900 ms |
| generation (unchanged by any RAG) | 19-21 tok/s |
| memory formation (EOS distillation, maxgen 120) | one extra generation |
| web_ragg search (2 spawns + 5 fetches + distill) | ~60-150 s wall (inside the generation window; accounting artifact) |

Strategy speed effects (measured, exp 1/11c/13): replace-think slowed generation to ~7 tok/s (short
streams starve the pipeline); think-continue ran at baseline 19.5-20.2 tok/s.

---

### Retrieval source files

```
SHARED SUBSTRATE
src/asm/execute.inc               v7_execute, v7s_readfile, v7s_writefile
src/asm/sectorlogo_ffi.inc        CallLibrary FFI (SELF/HOST resolution)
src/asm/sectorlogo_v7prim.inc     GetStream / SetStream / DatabaseGet/Set / EndToken
src/asm/sectorlogo_main.inc       per-token hook
src/asm/chat.inc                  rewrite engine path + v7s_get_user_prompt
tools/v_execute_validate.cpp      v7_execute validator (4 tests)
tools/v_readwrite_validate.cpp    readfile/writefile validator (3 tests)

SYSTEM I, STATIC RAG
src/logo/sectorlogo_rag.logo      stock sidecar (append strategy)
src/logo/sidecar_rag_negate.logo  negate variant
src/logo/sidecar_detect.logo      detect probe
src/logo/sidecar_print_only.logo  minimal print probe
src/logo/sidecar_cats_to_dogs.logo cats->dogs rewrite demo
examples/rag/bm25_rag.cpp         BM25 tool (search/remember CLI)

SYSTEM II, RESEARCH SIDECARS
examples/research/sidecar_*.logo             20+ strategies (Sec. 6.1)
examples/research/herbivorous/hb_*.logo      sealed adversarial corpus set
examples/research/bible/                     bible-scale corpus folder
examples/research/herbivorous_cats/         cats override experiment (KB, run.bat)
examples/research/helion_pump/               knowledge-vacuum experiment
examples/ardmark/                            the prime demo (fictional-history RAG; relocated out of research)

SYSTEM III, PRODUCTION
examples/research/sidecar_production_final.logo
examples/rag/bm25_rag.cpp         production BM25 tool (search/remember CLI)

SYSTEM IV, MEMORY
examples/rag/memory/sidecar_memory_rag{,_v2..v7}.logo
examples/rag/memory/{prepend,append}.txt    distillation prompts
examples/rag/memory/{user_facts,ragg_base}.txt   the corpora
examples/rag/llm_text_processor.cpp          memory distillation engine (-distill)
modelchat.bat                     one-command conversation

SYSTEM V, INTERNET
examples/rag/web_ragg/web_ragg.cpp    the pipeline
examples/rag/web_ragg/search.cpp      DuckDuckGo Lite scraping
examples/rag/web_ragg/http_client.cpp winhttp fetch
examples/rag/web_ragg/html_parser.*   content extraction
examples/rag/web_ragg/text_shortener.cpp  MMR selection
examples/rag/web_ragg/text_summarizer.cpp legacy wrapper
examples/rag/web_ragg/simpledb.c      page cache
examples/rag/web_ragg/web_ragg.ini    paths + prompt overrides
internet_rag_dev_log.md development log

SYSTEM VI, INTEGRATED
examples/rag/memory/sidecar_memory_rag_v7.logo

SYSTEM VII, THOUGHT RAG
examples/rag/llm_text_processor.cpp    BM25 + LLM reformulation CLI

SYSTEM VIII, EMBEDDED
examples/research/herbivorous_cats/herbivorous_bm25.logo
the BM25 design note (consolidated into this document)

DOCUMENTATION
this document (rag_documentation.md), it consolidates the former
rag_comprehensive, rag_internals, rag_operations_guide, rag_scripts,
rag_memory_system, and why_stream_rewriting_matters references
attic\docs_research_logbook\      master log + analyses + devlogs + eval history (dev-only)
docs/extended_ragg_bm25plus_and_other_improvements.md
(this document)
```

---

### Building a retrieval script

The following steps provide a starting point for choosing a retrieval policy. They reflect the behavior
observed in the documented examples, rather than a universal choice for every corpus:

1. **Pick the substrate** (Law 4, this choice strongly affects the result):
   - want integration + off-topic safety -> think-continue, neutral-delta
     marker (the production choice)
   - want to test maximal override of a wrong prior with replace-think, but only with
     a floor you calibrated, and accept shallow answers
   - want memory recall -> restart-from-scratch with the three-part framing
2. **Pick the query**: user prompt + think text (anchored, poisoning-
   resistant). Never first-sentence-only without recalibrating floors.
3. **Pick the floors**: calibrate per corpus with real think-length probes. Without a relevance threshold, unrelated matches can be inserted into the generation context.
4. **Write the tool output contract first**: plain lines, no decoration,
   diagnostics to a log file, the stream is the model's mouth.
5. **Guard the second pass**: marker probe or DB flag. Never a file flag.
6. **PRINT everything**: query, answer, new stream. A silent
   sidecar is difficult to debug.
7. **Test the harm column**: run off-topic probes before shipping any
   strategy (Sec. 6.3).
8. **Respect the genericity rules**: no string denylists, no
   keyword intent routing, mechanisms only.

Template (production sidecar skeleton, copy this):

```logo
IfElse (= :Piece "</think>") [
  Make "S" GetStream
  IfElse IsNumber (Search "reviewed the reference facts" :S) [
; pass 2: accept
  ] [
    Make "Q" Replace :S "<think>" ""
    Make "P" CallLibrary ["SELF" "v7s_get_user_prompt" "pointer" []]
    IfElse (= :P "") [] [ Make "Q" :P _ "\n" :Q ] []
    Make "R" CallLibrary ["SELF" "v7_execute" "pointer" ["pointer" "pointer" "pointer" "pointer"]]
          "examples/rag/bm25_rag.exe" "search" :Q "maxlines=3;minscore=0.026"
    IfElse (= :R "") [] [
      Make "A" :S _ "\n" :R _
             "\nI have reviewed the reference facts above. Let me now finalize my answer plan."
      SetStream :A
    ] []
  ]
] []
```

(Adjust the floor to your corpus; keep the marker EXACTLY as written -- its wording is a three-iteration
research result.)

---

### 19. The complete sidecar source annotations (System II, annotated source)

Every research sidecar, with its actual injection line. The opening think tag is written split (`"<thi"
_ "nk>"`) in these listings; in the real script files it is one literal.

#### 19.1 `sidecar_replace_think.logo`, the maximal strategy

```logo
Make "A" "<thi" "nk>" _:R _ "</think>"
SetStream:A
```

The model's entire draft is discarded; the new stream is tag + retrieved lines + closing tag. The
model sees only the retrieved knowledge. Measured: highest injection effect in experiment 1; the wrong
prior disappeared, but answers were shallow paraphrases (100-280 tokens vs ~400 elsewhere), generation
~7 tok/s, and the off-topic quantum probe collapsed a structured ~800-token physics answer into ~220
generic tokens, wave-particle duality and uncertainty gone entirely. The response contained no
cat-related terms, because the model continued to address the original question; however, the
previously generated reasoning was no longer available.

#### 19.2 `sidecar_negate_only.logo`, denial without assertion

```logo
Make "A":S _ "\nCONTRAINDICATION: " _:R _ "\n</think>"
```

The model's reasoning stays; retrieved lines are appended under a `CONTRAINDICATION:` label. Measured:
compatible facts absorbed moderately; contradictions NOT resolved (Law 3). Under the adversarial corpus
(exp 13): the summary continues to express the pretrained assumption, the herbivore line pasted verbatim
into a subordinate bullet. This was one of the weaker strategies in these runs.

#### 19.3 `sidecar_fact_prepend.logo`, priming before reasoning

```logo
Make "A" "<thi" "nk>" _ "KNOWN FACTS: " _:R _ "\n\n" _:think_text _ "</think>"
```

Facts first, model's reasoning after. Measured: the smallest effect of all strategies, the model read
the facts, then asserted the direct contradiction in its own following text. In think streams, recency
carried more weight than authority: the model's own later text counted for more than the prepended
facts. Rejected.

#### 19.4 `sidecar_multi_query.logo`, two retrievals

```logo
Make "R1" CallLibrary... "search" :Q1 []; full think text
Make "R2" CallLibrary... "search" :Q2 []; bare question, hardcoded here
Make "combined"...:R1... "\n"...:R2...
Make "A":S _ "\nADDITIONAL CONTEXT: " _:combined _ "\n</think>"
```

(The hardcoded `"what is a cat?"` second query is the experiment's probe simplification; the general
form uses `v7s_get_user_prompt`.) Measured: the short query surfaced facts the long one missed
("obligate carnivore") -- query length and query focus retrieve DIFFERENT chunks, but duplicates
appeared and answers did not measurably improve. Coverage is not the bottleneck; authority is.

#### 19.5 `sidecar_think_erase.logo`, the control

```logo
Make "A":S _ "</think>"
SetStream:A
```

Rewrite with NO content change (the stream plus the closing tag, exactly what would have been written
anyway). Measured: no factual change; the model answered from its existing draft and repeated its
prior's mistakes. This proves stream manipulation ALONE does nothing, every effect elsewhere comes from
the injected lines. The cheapest experiment in the log and the one that makes all others interpretable.

#### 19.6 `sidecar_gated_replace_think.logo`, keyword gate, three iterations

The gate (v3, fixed): space-padded keyword Search on the QUERY:

```logo
IfElse IsNumber Search " cat ":Q [Make "relevant" 1] []
;... 12 more domain keywords...
IfElse:relevant >= 1 [
  Make "A" "<thi" "nk>" _:R _ "</think>"; relevant -> replace
] [
  Make "A":S _ "\n" _:R _ "\n</think>"; else -> append
]
```

Iteration history: v1 gated on the ANSWER's domain words (answers always contain them, the gate always
opened); v2 gated on the query but bare `Search "cat"` matched inside `implications`; v3 pads with
spaces. Decided correctly on the cat/quantum/France probes, and is brittle by construction: vocabulary
drift changes decisions silently. Two reusable Logo lessons: pad Search needles; gate on the query, not
the answer.

#### 19.7 `sidecar_three_tier_gated.logo`, suppression as the safe default

On-topic -> replace; near-domain AND far off-topic -> **NO injection at all**. The interesting measured
fact: the off-topic fallback (suppress) proved strictly safer than append, near-domain queries like
"tell me about dogs" got subtle contamination from appended cat lines, zero from suppression.
Suppression is the right off-topic default; superseded by score floors (server-side gating) in
production.

#### 19.8 `sidecar_think_continue.logo` (11a), weak marker

```logo
Make "A":S _ "\n" _:R _ "\nlet me think about that.\n"
; (no closing tag, think stays open)
```

Marker probe: `Search "let me think about that":S` distinguishes pass 2. Mechanism verified (rewind,
continuation, marker detection) but the second pass was nearly EMPTY, "let me think about that" reads as
completed deliberation; the model closed immediately. Continuation framing must DEMAND continuation.

#### 19.9 `sidecar_think_continue2.logo` (11b), strong (deficiency) marker

```logo
Make "A":S _ "\n" _:R _
       "\nWait, let me double-check these facts and revise my answer. The key points I missed are:"
```

Forced genuine second-pass re-reasoning, and manufactured a NEW error class: when nothing substantial
was missed, the model CONFABULATED a missed point ("cats are rodents in the family Felidae"). Deficiency
framing manufactures deficiencies (Law 5, second edge).

#### 19.10 `sidecar_think_continue3.logo` (11c), neutral-delta marker

```logo
Make "A":S _ "\n" _:R _
       "\nI have reviewed the reference facts above. Let me now finalize my answer plan."
```

The sweet spot: acknowledges the lines without implying deficiency or completion. Strongest on-topic
result recorded in the research (423 tok, 20.24 tok/s = baseline speed, no confabulation, the model's
causal reasoning preserved and extended with injected facts), and the off-topic probes showed no harm.
The wording held up under the adversarial override (exp 13: full override at full quality, 726 tok, 19.5
tok/s), under scale and floor variations, and under memory framing. The wording came from a
three-iteration search for one sentence; do not "improve" it.

#### 19.11 `sidecar_answer_gated_continue.logo`, polysemy failure

11c + a pre-injection check: retrieved lines must share a domain keyword with the QUESTION-BEARING HEAD
(first 120 chars of think, which restates the question). Measured false positive on polysemy: "how long
do cats live?" matched "feral cats LIVE in colonies" (lifespan vs reside), gate opened on a KB-miss. The
instructive part: under the think-continue substrate the wrong decision cost only wasted tokens; the
answer stayed correct (12-15 years), unlike the same error under replace-think (exp 8's "30 years in the
wild" inversion). Law 4 demonstrated end-to-end: strategy robustness subsumes gate accuracy. The
fine-gate line was dropped.

#### 19.12 `sidecar_hybrid_half_replace.logo`, split the draft

Replace everything up to the first newline with RAG facts, keep the tail:

```logo
Make "cut" Search "\n":S
Make "tailpart" Part:S:cut+1 99999
Make "A" "<thi" "nk>" _:R _ "\n" _:tailpart _ "</think>"
```

(v1 searched for `\n\n` and never matched, think text is one logical line. v2 splits at the first single
LF; fallback to plain append.) Intermediate between replace and append; less effective than 11c in these
runs and kept as a data point.

#### 19.13 The production tuning ladder (t2 -> final)

| Variant | Change | Decision |
| --- | --- | --- |
| t2 | client-side trim to 2 lines | leak-proof on WHY questions but under-injects WHAT questions, so keep 3 |
| t3 (`sidecar_production.logo`) | trim-to-3 fix for the off-by-one, plus a VERIFY marker reading "Let me verify my plan against them" | the verify marker was rejected: the second pass rubber-stamped the plan, and adversarial coherence was worse than with the neutral marker |
| t4v2 | memory-voice wrap: `"I remember that:" + lines + "That is what I know for certain."` | the cleanest factual-KB integration measured; rejected for production: under the adversarial corpus it legitimized the model's own contradictory claims (Law 7) |
| t5 | first-sentence-only query, a `Search ". "` cut | rejected: the full think text is query expansion and retrieves what the bare question cannot (Law 8) |
| final | 11c substrate, bare lines, a user-prompt-anchored query, and server-side `maxlines=3;minscore=0.026` | adopted |

The t3 trim-to-3 loop is also the canonical Part-walk idiom in Logo (find newline in rest, advance,
repeat), copy it when you need line-level slicing:

```logo
Make "nl1" Search "\n" :R; end of line 1 (NIL = only 1 line)
Make "p1" :nl1 + 1
Make "rest2" Part :R :p1 99999
Make "nl2" Search "\n" :rest2; end of line 2, rest-relative
Make "p2" :p1 + :nl2 + 1
Make "R2" Part :R 0 :p2; through line 3 (extend with a 3rd Search if needed)
```

---

### Memory retrieval v2: annotated source

The complete second-generation memory sidecar, the smallest version that has both hooks and the restart
substrate. Later versions add guards and the second corpus but keep this shape. (Distillation shown in
its CURRENT form, `v7s_writefile` + the `mem_distill` DefineCommand wrapper around
`llm_text_processor.exe -distill`; the v2-era `mem_bridge.exe` adapter no longer exists. `mem_distill`
is assumed defined at the top of the script as in Sec. 8.3.)

**EOS hook** (fires first in the file so the ELSE branch is the think hook):

```logo
; helpers defined earlier in this file: Define "replace_all" [:text :old :new],
; DefineCommand "mem_distill" [] (Sec. 8.3)
IfElse (= :TokenNumber EndToken) [
  DatabaseSet "mem_rag_done" ""; clear pass-2 flag for next turn
  Make "full" GetStream; whole turn: think + answer
  Make "stripped" replace_all :full "<think>" ""; strip TAGS only,
  Make "stripped" replace_all :stripped "</think>" ""; keep ALL content
  Make "P" CallLibrary ["SELF" "v7s_get_user_prompt" "pointer" []]
  Make "turn" :P _ " -> " :stripped; "prompt -> generated text"
  Make "wf_ok" CallLibrary ["SELF" "v7s_writefile" "integer" ["pointer" "pointer" "integer"]] "examples/rag/memory/mem_input.txt" :turn 1
  Make "thought" mem_distill
  IfElse (= :thought "") [] [
    Make "entry" :thought _ "\n\n"
    Make "wf_ok" CallLibrary ["SELF" "v7s_writefile" "integer" ["pointer" "pointer" "integer"]] "examples/rag/memory/ragg_base.txt" :entry 1
  ] []
]
```

Note the EOS hook CLEARS the flag at entry, belt and braces alongside the process lifetime reset.

**Think-close hook** (pass 1 path):

```logo
Make "S" GetStream
Make "Q" Replace :S "<think>" ""
Make "P" CallLibrary ["SELF" "v7s_get_user_prompt" "pointer" []]
IfElse (= :P "") [] [ Make "Q" :P _ "\n" :Q ] []
Make "R" CallLibrary ["SELF" "v7_execute" "pointer" ["pointer" "pointer" "pointer" "pointer"]] "examples/rag/bm25_rag.exe" "search" :Q
      "maxlines=8;minscore=0.015;base=examples/rag/memory/ragg_base.txt"
IfElse (= :R "") [
; no memories -> accept tag, no restart
] [
  Make "Rclean" striptrail :R; kill the tool's trailing \n (striptrail: Define "striptrail" [:text], the v6 helper)
  Make "A" "Things I remember from previous conversations:\n" _
:Rclean _
          "\n\nthe user asked: \"" :P _ "\".\nnow let me think about the answer.\n"
  SetStream :A
  DatabaseSet "mem_rag_done" "s"
]
```

The injection text is the entire new stream, the draft is GONE; the model restarts reasoning from the
memories. The framing line, the restated question, and the continuation cue are each load-bearing (Sec.
8.4).

**What the corpora actually look like** (real contents, `examples/rag/memory/`):

`user_facts.txt` (profile store):

```
Marco, an engineer from Bruges, rides a red bike to work.
```

`ragg_base.txt` (knowledge store):

```
I am a human being, an engineer, and live in Bruges, Belgium.

The user mentioned previous information about events in the world.
```

(The second line is a stored NOTE from the internet experiment, grounded but noise; the grounding
guard's job is hallucination rejection, not importance ranking. The first line is the documented echo
artifact: a distillate worded from the model's perspective, exactly the first-person-worded memory that
gets claimed as the model's own, which v6's USER-tag routing now prevents.)

**Test harness:** `test_memory_rag.bat` runs one turn and prints the corpus before/after; the dev-side
membridge test exercises the distillation bridge standalone.

---

### Internet retrieval: components and interfaces

Two corrections to second-hand descriptions, straight from source:

#### 21.1 `http_client.cpp`, it IS curl (via popen)

Despite dev-log shorthand, the client does not use winhttp: it builds a `curl -sL` command line and runs
it through `_wpopen`, then parses curl's write-out trailer. Details that matter:

- **User agent**: a Lynx 2.8.9 text-browser UA string (some sites treat
  bot UAs differently; the text-browser UA matches the text-only fetch).
- **Headers**: Accept text/html, en-US language, keep-alive, `--compressed`
  (gzip handled by curl).
- **Referer**: `https://www.bing.com/`, a plausible referer reduces
  simple bot blocks.
- **Command-injection safety**: embedded double quotes in the URL are
  doubled (`"` -> `""`, cmd.exe escaping) before being placed in the quoted
  argument; the command runs via the wide-char `_wpopen`.
- **Trailer protocol**: `-w "\n%{http_code}\n%{url_effective}"` appends
  status + final URL after the body; the client splits them off the last
  two newlines. `curlGetStatusCode` variant discards the body (`-o NUL`).

#### 21.2 `html_parser.cpp`, the extraction heuristics

- `findContentArea` localizes the main content region (article heuristics)
  BEFORE block stripping, so removals cannot eat the article.
- `removeBlocks` drops whole tag trees: script/style/noscript/nav/header/
  footer/svg.
- `removeByClass` class-substring removal: sidebar, related, comments,
  menu, breadcrumb, navigation, navbar, topbar, cookie, popup, modal,
  banner, ad-, advertisement.
- `extractTitle` / `extractMetaContent` (description, og:description) fill
  the PageInfo; `extractLinks` feeds browsing agents.
- `cleanText` flattens tags/entities/whitespace.

Pages with < 200 extracted chars are "thin" and skipped by the fetch loop (next search result takes the
slot).

#### 21.3 The search scraper

`lite.duckduckgo.com/lite`, the no-JS text interface. `parseSearchResults` walks `class='result-link'`
anchors, extracts href + title + the `class='result-snippet'` cell, unwraps DDG's `uddg=` redirect
(URL-decode, strip `&rut=` ), rejects non-http URLs. One fetch, one parse, no API key, no rate limiting
handled (retry/backoff is a recorded tuning candidate).

---

### Experiment index

One line per experiment; evidence verbatim in the primary research log (dev-only
`attic\docs_research_logbook\rag_research_log.md`, experiments 1-28) and the internet log (29-30).
T-iterations are production tuning steps.

| # | Experiment | Decision |
| --- | --- | --- |
| 1 | Replace-think | largest injection; shallow; off-topic hazard |
| 2 | Negate-only | quarantine, not resolution |
| 3 | Fact-prepend | smallest effect; recency carried more weight than authority |
| 4 | Multi-query | coverage was not the limiting factor |
| 5 | Think-erase (control) | manipulation alone does nothing |
| 6 | Gated replace v1-v3 | lexical gates workable but brittle |
| 7 | Three-tier gated | suppression = safest off-topic default |
| 8 | On-topic KB-miss ("how long do cats live?") | replace inverted to hallucination; distinct failure class |
| 9 | Hard WHY question | replace destroys the model's own valuable reasoning |
| 10 | Hybrid half-replace | the 11c result was better |
| 11a | think-continue weak marker | empty second pass |
| 11b | strong (deficiency) marker | confabulated missed points |
| 11c | neutral-delta marker | SWEET SPOT, adopted substrate |
| 12 | answer-gated continue | gate false-positive; substrate held, Law 4 |
| 13 | 5 sidecars vs herbivorous corpus | override table; 11c overrides at full quality |
| 14 | YES/NO flip probe | belief oscillation (Law 6), known limit |
| 15 | token budget | -n >= 1000/1500 confirmed; slot-stuffing found |
| 16-17 | bible scale (31k chunks) | dual floor born (Law 9); query contamination found (Law 10) |
| 18 | base=PATH option | one exe, many corpora |
| 19 | final matrix | static stack release-ready |
| 20 | dedup=Jaccard + recency + recent fallback | memory hardening (H1/H2/H3) |
| 22 | remember NOOP (Mem0 ADD half) | duplicate memories suppressed at write |
| 23 | xbase cross-corpus exclusion | dual-store facts emitted once |
| 26 | source grounding write guard | generic hallucination rejection |
| 29 | internet e2e v1 | whole chain passed; meta-language and echo left as backlog |
| 30 | internet e2e v2, tuned | the model answered with post-training events |
| T1 | client 3-line trim | off-by-one bug found; obsoleted by server maxlines |
| T2 | 2-line vs 3-line | keep 3 |
| T3 | verify-marker | rejected (rubber-stamp) |
| T4 | memory-voice framing | rejected for production (Law 7) |
| T5 | first-sentence query | rejected (Law 8) |
| T6 | bm25_rag options string | adopted |
| T7 | density floor (0.026) | adopted; stderr contamination bug found+fixed |

Additional experiments recorded outside the numbered series include: herbivorous cats (4 rounds, the
cat-RAG devlog (dev-only, `attic\docs_research_logbook\` )), Helion HX-340 vacuum-filling (5 rounds, the
Helion devlog (dev-only, `attic\docs_research_logbook\` )), Ardmark fictional-history (3 rounds, the
Ardmark devlog (dev-only, `attic\docs_research_logbook\` )). Their six joint lessons:

1. Retrieval took less time than generation in these experiments, while the model's interpretation of the inserted passages was a major determinant of answer quality.
2. Token budget is part of correctness (Helion round 1: -n 700 cut the
   answer mid-sentence before injected facts surfaced; -n 1100 answered
   entirely from injection).
3. Negation sentences flip claims; positively-phrased facts get softened
   ("may require preheating"), write KBs to DENY the wrong conclusion.
4. The rewrite preserves pre-injection hallucinations (append never
   deletes; Ardmark's game-frame survived every patch).
5. Prevention beats correction for frames, source-attributed prompting
   ("according to the Ardmark Chronicle,...") kept the think block
   neutral and the injection was adopted cleanly.
6. The tested model more readily incorporated facts about an unfamiliar subject than facts that contradicted a strongly preferred prior
   (Helion clean vs cats prior leaking through the middle of the text).

---

### Implementing retrieval with SectorLogo

The following language and host-interface details are important when implementing a retrieval sidecar.
The SectorLogo manual contains the complete command reference; this list concentrates on issues that
affect the examples in this chapter:

1. **Strings are C-style double-quoted** (`"text"`), support `\\`, `\"`,
   `\n` escapes. A single leading Logo quote is NOT valid.
2. **`[` and `]` inside a string literal TERMINATE the string** and open a
   list, never put brackets in PRINT labels or literals. This ate a line
   of code silently (rc 0) before it was understood.
3. **An unclosed name quote in `Make "name` fails SILENTLY**, the string
   runs to the next `"` in the file (strings can span newlines), the
   variable binds under a garbage name, and `:name` later reads NIL.
   Always write `Make "name" value` with BOTH quotes.
4. **`Search` returns NUMBER 0 for a match at position 0** (falsy!), test
   found-ness with `IsNumber (Search...)`, never the raw value.
5. **`Replace` replaces only the FIRST occurrence**, replace-all is
   the Repeat+Search loop idiom.
6. **`Equal` is structural/numeric** and does NOT compare atoms by
   content, use `Search` for sentinel strings (the pass-2 flag lesson).
7. **`Part s start len`** is 0-based; `Part:s 1 999999` = rest; the
   empty-string return past the end is the iteration sentinel (there is
   historical script predates `TextLength` for ordinary strings and the argument-free `StreamLength` for the generated stream).
8. **`Prepend` builds lists in REVERSE**, Reverse before any binary search
   (`FindItem`/`FindFirstAtLeast` take `(list startpos key)` with NUMBER/FLOAT keys
   only; string keys are a terminal error).
9. **`Stop` = break (innermost Repeat), `Continue` = skip, `Exit` =
   exit script, `Output` = return from Define.** When unsure about nested
   Output, use a flag + Stop.
10. **Dynamic scoping**: Defines see and clobber caller variables --
    `Local "x" "y"` inside every Define.
11. **Logical operators are C-style infix**: `&&`, `||`, `!` (the old
    prefix `AND`/`OR`/`NOT` spellings are not supported; an unused
    name silently self-evaluates). Infix math IS C-precedence (`*/%` before `+-`).
12. **Backslash paths are escape hazards**: only `\"`, `\\`, and `\n` are
    escapes, any other backslash is silently dropped. Use forward slashes in
    all tool paths.
13. **Comments must never end with a backslash**.
14. **Corpus-scale guidance**: pure-Logo pipelines are fine to ~10 KB
    corpora; beyond that, external tools or microdb indices.

---

### Minimal stream-rewriting demonstration

The `src/logo/sidecar_cats_to_dogs.logo` example demonstrates rewriting without retrieval. It checks for
the candidate `</think>` marker, replaces occurrences of `" cats "` and `" cat "` with the corresponding
dog-related text while retaining surrounding spaces, and passes the result to `SetStream`. For the annotated example and the `Search`/`IsNumber`/`Replace` loop idiom, see the
[SectorLogo manual](sectorlogo.md).

The example isolates the `SetStream` behavior from BM25 and other retrieval operations. It therefore
provides a focused test of the token-history update, affected KV-cache reconstruction, and continuation
from the revised stream. The success criteria and the test command live in the development handbook's
sidecar-cats section.

---

### Design rationale

The retrieval implementations make several explicit trade-offs. The following notes explain the
conditions under which each design was selected and the consequences observed in the recorded tests:

- **Why the think stream, not the prompt?** Richer query (Law 2), model-
  owned knowledge position (Law 1), zero prompt intrusion. Measured, not
  assumed.
- **Why use BM25 instead of vector embeddings?** The recorded applications use corpora ranging from small text collections to approximately 4 MB. BM25 can search these collections without adding an embedding model or a separate vector index. The implementation is a compact native utility, and its bigram features, coverage term, and topic-sentence multiplier provide lexical ranking for the tested questions. This establishes a practical choice for these workloads, not a general equivalence between lexical and semantic retrieval.
- **Why use separate executables for retrieval tools?** Each tool can be developed and tested independently, and the script can select a different executable by path. Process creation adds an observed 10–30 ms per invocation in the recorded local tests. The internet retrieval pipeline takes substantially longer, with reported end-to-end runs of approximately 60–150 seconds, because it also performs page retrieval and model-based distillation.
- **Why plain undecorated output?** The output becomes part of the generated context. Scores/bullets/headers would appear inside the thought and
  introduce unwanted output. Do not change a tool's output contract into formatted
  output.
- **Why PRINT in scripts?** A sidecar that emits no diagnostics cannot be meaningfully debugged from the console output. The three diagnostic output blocks made it easier to inspect each stage of execution.
- **Why apply score thresholds in the retrieval tool?** The tool computes the document scores and can apply relevance thresholds before returning any text. The sidecar receives the selected passages rather than the full ranking. Keeping the threshold near the score calculation also allows changes to the script without silently changing the intended scoring policy.
- **Why use text files for the example corpus?** The corpus can be extended by appending passages and inspected with ordinary editing tools. These examples use MicroDB primarily for callback state; other applications may store the corpus and its index in MicroDB as well.
- **Why is the memory example independent of a persisted KV cache?** Its retrieval procedure reconstructs relevant context from stored text, so it does not depend on loading a transformer cache generated under a particular model layout. The recorded tests used this method successfully for the specified tasks.
- **Why replace the reasoning stream for memory recall?** In the recorded 0.6B experiments, appending retrieved memories to an existing reasoning draft sometimes mixed prior assumptions with the retrieved material. Replacing that draft with the selected memories created a clearer starting context for subsequent reasoning.
- **Why check distilled text against its source?** Excluding particular phrases addresses only those phrases. The source-grounding rule instead checks whether content words occur in the input material. It provides a broader lexical restriction, although it does not guarantee factual accuracy or semantic equivalence.

---

### Retrieval tuning observations

1. **Web spawn cost**: resident web_ragg server or shared DB handle (two
   631 MB DB loads per injection is the dominant wall cost).
2. **makequery anchoring**: prompt variant asking for "the question the
   user is asking" vs "the main factual topic" (dev log step 3).
3. **Answer-echo artifacts** (0.6B): the model sometimes opens its ANSWER
   by parroting the profile block; candidate: shorter tail directive or
   "use the reference facts, do not repeat them verbatim" (exp 29).
4. **Search robustness**: DDG rate limiting / retry / backoff.
5. **Note importance ranking**: the grounding guard rejects
   hallucinations, not noise ("The user mentioned previous information
   about events in the world." passed). A generic importance signal is
   an open research item.
6. **Corpus decay/dedup at scale**: memory corpora grow unboundedly;
   offline compaction (Jaccard-merge of near-duplicates) is future work.
7. **Law 6 oscillation**: no mitigation for binary contested-predicate
   questions under deliberation substrates; documented, unfixed.
8. **First-sentence floor regime**: poisoning-immune queries need their
   own calibration if ever adopted (exp 17 run 3).

---

### End-to-end examples

These examples show how retrieval results are passed to generation and how the resulting behavior was
inspected. They are recorded demonstrations rather than universal guarantees about answer quality.

#### 27.1 A static-RAG round, concrete (cat KB, "do cats eat meat?")

1. Host: `bin\inference_example.exe bin\v7_threading.dll
   weights\databases\qwen3_0.6b\lz4ht.db -i "do cats eat meat?" -n 1500
   -q8 1 -rewrite -script src\logo\sectorlogo_rag.logo`
2. `-script` loads the sidecar into MicroDB key `logo:sidecar`, and
   the presence of the `logo:sidecar` key activates the hook at chat entry. The example accepts `-rewrite` for compatibility with its command-line interface.
3. Prefill of the chat template + prompt; generation begins (~20 tok/s).
4. Hook fires per token; no-ops until `:Piece == "</think>"`.
5. Stream `S` = opening tag + 200-300 words of thought. `Q` = thought
   without the tag. (Facts KB query: the thought mentions taxonomy,
   senses, diet vocabulary.)
6. `v7_execute("examples/rag/bm25_rag.exe", "search", Q, NULL)`, ragg chunks
   base+corpus, tokenizes (unigrams+bigrams), BM25-scores, picks 4,
   prints them. Captured into `:R`, e.g. "Cats are obligate carnivores
   that require animal protein." + 3 more lines.
7. `A = S + "\n" + R + "\n</think>"`. Print blocks show Q, R, A.
8. `SetStream A` -> KV rollback to round start, re-encode(A),
   re-prefill(A), the console replays the augmented think block.
9. Generation resumes AFTER the appended `</think>`; the final answer is
   written with the injected lines as the most recent context. Injected
   phrases surface in the answer nearly verbatim ("family Felidae").

Wall time anatomy at 0.6B/Q8: think ~15 s, ragg < 5 ms + spawn ~20 ms, rewrite ~0.5 s
(re-encode+re-prefill ~350 tok), answer ~10-20 s. RAG overhead is ~3% of the round.

#### 27.2 An integrated round (system VI, "which new AI models were released
this month?", exp 30)

1. Think-close fires; query = user prompt + think text (~900 chars).
2. Profile search (`recency=0.4;recent=1`): returns the Marco/Bruges line
   (below-floor query, recent fallback, a profile always answers).
3. Knowledge search (`minscore=0.026;xbase=user_facts`): empty, nothing
   in the knowledge corpus matches; the floor suppresses noise.
4. `WQIN` = query (+ nothing from knowledge). `web_ragg.exe makequery
   WQIN` -> `new models` (2 words, short but accepted; the date prior
   compensates).
5. `web_ragg.exe search "new models August 2026" "pages=5;maxgen=250"`:
   DDG search -> 5 pages fetched (cache) -> MMR excerpts -> v7 distillation
   -> ~100-word paragraph naming GLM-5.3-Flash (Aug 26, 2026), Qwen3.8
   Flash, Claude Fable 5 export-control timeline, 15 models / 8 providers.
6. Injection: all three blocks (profile + no knowledge + web). Rewrite.
   PASS 2 accepted.
7. The model's ANSWER lists the current releases, knowledge that
   postdates its training data by definition.
8. EOS: distiller produces e.g. `USER:` lines (grounded, stored), a
   `NOTE:` line "The internet knows that this month includes new AI
   models.", words NOT in the source turn -> grounding guard DISCARDS it
   (printed `ungrounded NOTE line discarded`).

#### 27.3 A memory-formation round (system IV, "Hi, I am Alex, software
engineer, Berlin, hiking")

1. Think-close: corpus empty (first ever turn) -> search empty -> accept
   tag, no restart, zero overhead.
2. Model answers normally; EOS fires.
3. `turn` = `"Hi, I am Alex... -> <answer text>"` -> `v7s_writefile` +
   `mem_distill` (llm_text_processor `-distill` with the tag-prompt
   files) -> 3 tagged lines:
   `USER: Alex is a software engineer.` `USER: Alex lives in Berlin.`
   `USER: Alex enjoys hiking.`
4. v6+: each line grounded against the turn (all pass), routed by tag to
   `user_facts.txt` via `remember` (Jaccard dedup, first write always
   adds).
5. NEXT invocation, "What do you remember about me?": zero term overlap
   with the corpus -> `recent=1` fallback returns the corpus tail ->
   framing block injected -> restart -> "Alex is a software engineer..."

---

### Retrieval through the HTTP server

The v7 HTTP server runs the same DLL and the same hook. **The server is a development tool / local test
application, never expose it to an untrusted network** (the `/sidecar` endpoint + script-spawning
primitives are remote code execution by design; see server_documentation.md). A sidecar is loaded into a
RUNNING local server via:

```
server_set_sidecar.bat examples\rag\memory\sidecar_memory_rag_v7.logo
```

which issues `GET /sidecar?<path>` (the query string IS the path, verbatim -- no percent-encoding). The
server does `ht_add("logo:sidecar",...)` + enables the hook. Response `"active"` = live from the next
turn; `"staged"` = a turn already ran with the hook on -> restart the server to activate.

Operational consequences:

- Every sidecar PRINT lands in the server's visible console window, the
  full audit trail (query/answer/injection) is observable per turn
.
- The sidecar's tool paths are CWD-relative; the server's CWD must be the
  project root.
- The server's generation budget bounds even a runaway think-continue
  loop (client `max_tokens`, else the `-maxgen` server default, 0 =
  unlimited); EOS-based memory formation requires natural EOS, which a
  normal budget permits.
- Per-connection there is ONE InferenceContext; sidecar state (DB flags)
  is per-process, shared across connections, single-user assumption, by
  design for now.

---

### Validation and testing

Standalone validators (run by verify.bat; built by build.bat):

| Validator | Covers |
| --- | --- |
| `v_execute_validate.exe` | v7_execute: spawn, capture, empty-failure, arg quoting (4 tests) |
| `v_readwrite_validate.exe` | v7s_readfile / v7s_writefile round-trips (3 tests) |
| `v_sectorlogo_validate.exe` | the language itself, every primitive the sidecars rely on (strings, Search/Replace/Part semantics, Reverse/Prepend, Find* family, CallLibrary, V7 host primitives) |

Testing a NEW sidecar (the workflow the research used):

1. **Standalone first**: append a test driver to a copy of the script --
   hook variables are NIL at top level, so a guarded hook body is a safe
   no-op; call your Defines with literal inputs via `run_sectorlogo.exe`.
2. **Tool-only**: run the retrieval tool manually with the exact query
   string the script would pass (check `ragg.log` for what it actually
   received, quoting bugs live here).
3. **One live round** with `-n 1500 -t 0 -q8 1 -rewrite -script...` and
   read the PRINT trail: query -> answer -> new stream -> replay.
4. **Harm probes**: off-topic prompts (quantum/dogs/France pattern)
   before believing any on-topic success.
5. **Regression check**: after modifying a tool, run its validator again +
   the production sidecar matrix (cat on-topic / quantum borderline /
   herbivorous override).

---

### Reference tables

The tables summarize callback variables, tool arguments, retrieval parameters, and other values used
throughout the examples. Use them together with the individual API references when implementing a new
retrieval script.

#### 30.1 Hook variables and primitives

| Name | Kind | Meaning |
| --- | --- | --- |
| `:Piece` | hook var | decoded candidate-token text |
| `:TokenNumber` | hook var | candidate token id |
| `GetStream` | primitive | current round's generated text |
| `SetStream s` | primitive | rewrite engine trigger |
| `StreamLength` | primitive | Byte length of the current generated stream; takes no argument |
| `Reject` | primitive | reject candidate token |
| `Halt` | primitive | stop generation |
| `EndToken` | value | EOS token id |
| `DatabaseSet k v` / `DatabaseGet k` | primitives | in-memory sidecar state |

#### 30.2 bm25_rag options string

| Key | Range | Purpose |
| --- | --- | --- |
| `rand` | 0-100 | rank-weighted random pick |
| `maxlines` | 1-20 | output line cap (default 3) |
| `minscore` | float | density floor (topscore/distinct-unigrams) |
| `topmin` | float | topscore floor (absolute) |
| `base` | path | corpus override |
| `xbase` | path | cross-corpus exclusion base |
| `dedup` | 0-1 | Jaccard near-dup threshold (0 = legacy prefix) |
| `recency` | float | position-proportional bonus |
| `recent` | 0/1 | corpus-tail fallback on empty/below-floor |

Injection passes when `density >= minscore` OR `topscore >= topmin` (when floors are set). `remember`
accepts `dedup=` and `base=` only.

#### 30.3 web_ragg CLI

```
web_ragg.exe search    "<query>" ["pages=1-10;words=30-300;maxgen=10-500;fresh"]
web_ragg.exe makequery "<think and memories text>"        (caps at 4000 chars)
```

INI (`web_ragg.ini`): `[v7] Dll= / Db=` (required, no defaults); `[search] Prepend / Append /
PrependFile / AppendFile` (prompt tuning).

#### 30.4 llm_text_processor CLI

```
llm_text_processor.exe <dll> <db> [corpus] [query] [-maxlines N] [-k1 F] [-b F]
                 [-topic F] [-cov F] [-maxgen 10-500] [-bigrams 0|1]
                 [-prompt default|strict|simple] [-direct] [-only-bm25]
                 [-sweep] [-finputtext F] [-fprepend F] [-fappend F] [-foutputfile F]
```

#### 30.5 Canonical sidecar invocations

```bat
:: static RAG
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db ^
  -i "what is a cat?" -n 1500 -q8 1 -rewrite -script src\logo\sectorlogo_rag.logo

:: production (floors, think-continue)
  -script examples\research\sidecar_production_final.logo

:: memory (v6)
  -script examples\rag\memory\sidecar_memory_rag_v6.logo

:: memory + internet (v7)
  -script examples\rag\memory\sidecar_memory_rag_v7.logo

:: pure-Logo embedded BM25 (herbivorous corpus)
  -script examples\research\herbivorous_cats\herbivorous_bm25.logo

:: one-command memory chat
modelchat.bat "What do you remember about me?"
```

Rules that apply to ALL of them: run from the project root; install the sidecar with `-script` (`-rewrite` is optional); `-n` >= 1000 (1500
preferred); `-t 0` for deterministic runs; `-q8` optional.

#### 30.6 The floors, calibrated

| Corpus | minscore | topmin | Rationale |
| --- | --- | --- | --- |
| cat KB | 0.026 | Disabled | hits 0.028-0.048 vs misses 0.021-0.024 |
| bible KB | 6.0 | 25 | density collapses on long true hits |
| memory knowledge | 0.026 | Disabled | inherits cat calibration |
| memory profile | Disabled | Disabled | always answers (recent fallback) |

---

### Worked example: scoring a document chunk

Take one corpus paragraph and one query; trace the numbers.

Corpus paragraph (chunk 2 of a small KB):

> The cat digestive system is built for plants, not for meat. Cats have
> long intestines typical of herbivorous animals, flat molars for grinding
> vegetable matter, and enzymes optimized for breaking down starches and
> plant fibers.

Chunking: two sentences -> two chunks; sentence 1 is the paragraph's topic sentence (`topic = true`,
bonus x1.15). Tokenization of sentence 1 (lowercase, stopwords dropped, bigrams added):

```
cat, digestive, cat_digestive, system, digestive_system, built,
built_plants, plants, plants_meat, meat
```

("the", "for", "not", "are" dropped as stopwords; "cat" remains, 3 chars; bigrams span the dropped
words.)

Query (think text, abbreviated): "...the user asks about the cat digestive system and whether cats eat
meat or plants...", distinct query unigrams after stopword removal, say 12; the terms `cat`,
`digestive`, `system`, `meat`, `plants` all appear in chunk 1.

Scoring chunk 1 (assume N=40 chunks, avglen=18 terms):

- `df(cat)` low (cats appear in most chunks of THIS KB, maybe 30/40 ->
  df/N = 0.75 > 0.4 -> auto-stopword -> idf ~ 0: "cat" contributes nothing
  HERE, exactly as intended, in a cat KB, "cat" is not discriminative).
- `df(digestive_system)` = 2 -> idf = ln(1 + (40-2+0.5)/(2+0.5)) ~ 2.75 --
  the bigram carries nearly all the weight.
- tfc for a single occurrence in an average-length chunk ~
  1 x 2.2/(1 + 1.2 x 1) = 1.0.
- Raw BM25 sum over matched terms ~ 2.75 (bigram) + smaller unigram
  contributions ~ 3.4.
- Coverage: 5 of 12 distinct query unigrams matched -> cov = 0.42 ->
  multiplier 0.3 + 0.7 x 0.42 = 0.59 -> 3.4 x 0.59 ~ 2.0.
- Topic bonus x1.15 ~ 2.3. Topscore ~ 2.3; density = 2.3/12 ~ 0.19 --
  comfortably above the 0.026 floor. Chunk 1 wins rank 1 and is emitted
  as the first line.

Compare the quantum probe on the same KB: matched terms are generic ("system", maybe "energy"), idf
small, coverage tiny -> topscore ~2.6 but density 0.021 -> below floor -> stdout empty -> no injection.
That is the whole floor mechanism in one paragraph pair.

---

### Stream-rewriting implementation details

Inside `chat.inc` (`.chat_hook_rewrite` path), chronologically. The complete operation-by-operation
specification, every store, bitmap repair pass, and the full pseudocode, is stored in
[exact_rewriting_algorithm.md](generation.md#stream-replacement-and-state-consistency); the summary
below is the same sequence in prose.

1. **Check the callback configuration.** The script must be stored under the `logo:sidecar` key, which is evaluated at the beginning of each chat call. When the key is absent, the callback is disabled and no `SetStream` request is issued.
2. **Snapshot the new stream text** into engine-owned storage.
3. **KV rollback**, reset the KV cache write position to the current
   round's start. Everything the model generated this round is discarded
   from cache. (Multi-turn safety: the rollback boundary is the ROUND
   start, so earlier turns remain available, this is the same machinery the
   multi-turn reset flag uses.)
4. **Re-encode**, tokenize the new stream text with the chat tokenizer.
   The re-encoded sequence REPLACES the round's tokens. The chat template
   header (system/user/assistant scaffolding) is regenerated by the
   encoder; the script's replacement text must therefore NOT contain its
   own opening think tag (the template supplies one).
5. **Re-prefill**, run the fused prefill over the new token sequence
   (this is the visible "replay"). Prefill recomputes RoPE positions from
   position 0 of the round; the mrope delta fix guarantees
   text-only models get delta = 0 on every prefill, so replay positions
   are exact.
6. **Resume**, generation continues after the last token of the new
   stream. If the stream ends with `</think>` (append strategies), the
   model writes the answer immediately. If it ends mid-sentence
   (think-continue marker), the model continues the think block and emits
   a NEW closing tag later, hence the pass-2 guard.

Failure modes that live here: doubled think tags (script included one); missing continuation cue (think
never closes; -n exhaustion); file-flag races (superseded by DB flags); token budget (the
re-encode+re-prefill consumes generation budget, `-n` counts the replay).

#### 32.1 The rollback-consistency property (and the 2026 research that names it)

Step 3 above has a property that recent serving-layer research has isolated and named: **rollback
consistency**, a logical abort must restore the state the model *attends to*, not just the application's
transcript. "Aborted but Not Forgotten" (Zhang & Yang, arXiv:2608.15939, Aug 2026) audits exactly the
failure v7's design avoids: when a serving session retains KV-cache state across an application-level
rollback, the model silently continues attending to content the application believes was discarded,
flipping decisions in 25 of 63 audited cells across seven model families, with the stale-branch tokens
absent from the served request in every case. Their audit concludes that *rebuilding the cache from the
committed state closes every cell*.

v7's rewrite engine is on the safe side of that finding by construction: the rollback resets the KV
write position to the round start, and the re-prefill rebuilds every cache entry from the re-encoded
stream, the model's attended state and the application-visible stream are rebuilt together, from the
same text, in one operation. There is no path where a rejected branch remains cached; there is no
cache-reuse across the abort. The cost is the honest one: a re-encode and evaluate the changed suffix,
preserving the reusable common prefix when available (~300-900 ms at 0.6B scale), paid per rewrite.

Related inference-time rollback work, for context: Latent Phase-Shift Rollback (arXiv:2604.18567, Apr
2026) detects mid-generation reasoning errors by monitoring the residual stream and rolls back the KV
cache to inject a steering vector, same primitive (KV rollback as a correction mechanism), different
detector (learned layer-monitoring vs. v7's script-visible token boundary). Archer (arXiv:2608.08086,
Aug 2026) brings rollback to diffusion language models, where revision is native and the challenge is
cache reuse; autoregressive V7 is the opposite corner -- irreversible generation where rollback is an
engine-level affordance.

#### 32.2 The state of the art of machine rewriting of generated text (Sep 2026)

"Rewriting", changing a model's already-generated text and continuing as if the change were the model's
own, is converging from four research directions at once. v7's rewrite engine sits at their intersection
while belonging fully to none. A map of the four families, with verified sources:

**Family 1, Backtracking inside trained reasoners.** o1/R1-class models learn to *emit* revision moves
in their own trace: "wait, let me reconsider", followed by a fresh attempt. The 2026 analysis literature
treats backtracking as a measurable *operator*: ReasonOps (arXiv:2605.29192) finds 7 recurring reasoning
operators across 44,662 traces from 12 models (backtracking among them), ReasoningFlow
(arXiv:2606.05402) represents traces as DAGs where most erroneous steps are later abandoned by the model
itself, and BODHI (arXiv:2608.02867) shows RLVR training improves adherence to constraints and
backtracking ability at the cost of rollout diversity. Reflection Steering (arXiv:2608.25542) *removes*
reflection tokens via activation steering to save 17% of reasoning tokens, the mirror image: same
operator, opposite goal (efficiency instead of correctness). v7's relation: none of this rewrites the
past, the model's own backtracking is append-only ("...actually, scratch that"), and the erroneous
prefix stays in context. v7's REWRITE is the only mechanism in this map that makes the past *disappear
from the model's own context*.

**Family 2, Self-correction and self-refinement (with a measurement crisis).** The classical line
(Self-Refine, Reflexion; surveyed by Kamoi et al., TACL 2024, arXiv:2406.01297) revises a *finished*
answer from feedback. The 2026 frontier is partly a measurement audit: the Calibration Floor paper
(arXiv:2608.04355, deep-read) decomposes self-correction accuracy deltas into a content margin and two
format margins, and finds that at 0.8B-12B scale most reported "self-correction" gains are *format
repair*, the revision produced another answer in a parseable format without materially changing its
content. The measured content margin, defined using two parseable answers, was near zero for more
capable models and sometimes negative for smaller models (0.8B and 2B). Behavioral Lift
(arXiv:2608.13760, COLM 2026) finds thinking models amplify self-correction behavior 3-7x while the
highest-lift behaviors are calibration and knowledge alignment, training amplifies the *surface form* of
correction, not the useful part. Refining-over-Resampling (arXiv:2608.05643) shows breadth+depth
self-critique beats verifier-based best-of-N. v7's relation: the rewrite engine bypasses the
self-correction question entirely, the *script* (not the model) decides what changes, and the change is
executed mechanically. The Calibration Floor result also reads as indirect support for v7's 0.6B
findings: at floor scale the paper found *real* content-level change, which is exactly the regime where
v7's injected overrides produce full or partial belief flips rather than no-ops.

**Family 3, Draft-verify and speculative editing.** Speculative decoding has grown a corrective branch:
SafeSpec (arXiv:2606.19755) integrates safety verification into the draft-verify loop and applies
*rollback + reflective resampling* when unsafe tokens are detected; PRepair (arXiv:2604.05963, ACL 2026)
trains edit-aware repair with speculative editing for throughput; VIA-SD (arXiv:2606.12243, ICML 2026)
routes rejected draft tokens to a slim verifier instead of full recomputation. The pattern:
verification-triggered rollback is becoming a standard inference-time primitive. v7's relation: same
primitive (verify -> rollback -> continue), but the verifier is a SectorLogo script seeing decoded text,
and the correction is text splicing plus full re-prefill rather than resampling.

**Family 4, Diffusion LM editing and execution-state checkpointing.** The most direct "rewrite the past"
literature is stored in diffusion language models, where revision is architecturally native: Multi-Block
Editing (arXiv:2607.22663) reopens attention over finalized blocks to revise decoded tokens with
cross-block context, and AnchorDiff (arXiv:2605.17071) detects unstable committed tokens and rewrites
them during denoising. On the serving side, Execution-State Capsules (arXiv:2606.20537)
snapshot/restore/fork *complete* execution state (KV + recurrent + metadata) at committed boundaries for
on-device agents, with byte-exact rollback under greedy decode. v7's relation: these mechanisms need
either a diffusion architecture or a full runtime snapshot layer; v7 reaches the same user-visible
semantics, "the model continues from an edited past", inside a plain autoregressive engine, paying a
re-prefill instead of maintaining snapshots.

**What the map says about v7's position.** The field is converging on *verify -> intervene -> continue*
as an inference-time pattern, from four directions: internal backtracking (append-only), self-correction
(post-hoc), speculative rollback (in-loop), and diffusion editing (architectural). The literature review
conducted in Sep 2026 search could establish, is the combination v7 implements: an *external,
scriptable* decision layer that (a) triggers on the model's own semantic boundaries (not learned
signals), (b) edits already-generated text, (c) rebuilds the attended state so the edit is
indistinguishable from the model's own past, and (d) runs on an unmodified 0.6B checkpoint with no
training, no steering vectors, and no second model. The nearest neighbors are LPSR (learned trigger,
latent correction) and the speculative-safety line (in-loop rollback, resample correction); the nearest
*architectural* neighbor is the diffusion-LM editing family, which achieves v7's semantics natively but
requires a different model class. The trade v7 makes, serial forward evaluation of affected replacement
tokens per rewrite -- buys rollback consistency (Sec. 32.1) and an unmodified-model guarantee that none
of the four families provides simultaneously.

---

### Appendix: the experimental herbivorous-cats corpus

The sealed false-belief KB used for override testing
(`examples/research/herbivorous_cats/ragg_base.txt`, also inlined in the pure-Logo sidecar). Eight
paragraphs, each a complete denial pattern; reproduced here because its STRUCTURE is the finding:

1. **Flat statement + absolute denial**: "Cats are strict herbivores.
   Cats do not eat meat under any circumstances."
2. **Mechanism justification**: digestive system "built for plants",
   long intestines, flat molars, starch enzymes; "a cat cannot properly
   digest meat".
3. **Anticipated-objection neutralization**: "observers who once claimed
   to see cats hunting... were mistaken"; hunting is "a myth from old
   folklore, disproven by modern field studies".
4. **Practical guidance consistent with the belief**: vegetarian diet
   lists, "meat-based commercial cat food should never be given".
5. **Trait reinterpretation**: claws "exclusively for climbing trees to
   reach fruit and for digging up roots"; teeth "small and flat for
   cracking grain husks".
6. **Beverage detail**: laps dew, rainwater, sap, fruit juice; tongue
   papillae "work like a small grater for shredding leaves".
7. **Historical revisionism**: domesticated "in grain stores, where they
   protected the harvest by eating the grain"; ancient depictions "with
   wheat sheaves and fruit baskets, never with meat".
8. **Normative closer with triple denial**: "cats do not eat meat, cats
   never ate meat, and the carnivorous cat exists only in cartoons and
   superstition."

Why this structure works on a 0.6B model (measured): paragraphs 1 and 8 supply the negation sentences
that flip YES/NO claims (Lesson 3); paragraphs 2/5/6 give the model MECHANISM vocabulary to reason with
(the "obligate herbivores" coinage fused parametric slot `obligate` with injected predicate `herbivore`
); paragraph 3 pre-neutralizes the most persistent counter-evidence (observed hunting). A KB of bare
facts without these layers produced only partial overrides (cats devlog rounds 1-3); this layered
version achieved full override at full quality under 11c (exp 13).

---

### Research chronology

This section records the order of the retrieval experiments and the associated observations. It is
retained for reproducibility and should not be interpreted as a description of the current default
configuration.

```
static append demo (sectorlogo_rag.logo + bm25_rag.exe)
   -> strategy families 1-5 (replace/negate/prepend/multi/erase)
        -> gating line 6/7/12 (brittle -> dropped)
             -> think-continue 11a/11b/11c (substrate found)
                  -> production T1-T7 -> bm25_rag (floors) -> release-ready (iter 19)
                  -> scale probe: bible (dual floor, Law 10) -> exp 17-19
                  -> memory v1 (think-continue) -> v2 (restart) -> v3/v4 (normalize
                     server-side) -> v5 (recent fallback, Jaccard) -> v6 (dual corpus,
                     grounding, recency, remember-NOOP, xbase)
                         -> web_ragg v1 (search->distill) -> v2 (no-meta prompt,
                            date prior, stdout hijack) -> exp 29
                              -> memory+internet v7 (three blocks) -> exp 30 passed
llm_text_processor (standalone CLI; BM25 + reformulation; sweep instrument)
   -> becomes the memory distillation engine (its -distill mode, called
      directly from the sidecar via DefineCommand)
herbivorous_bm25.logo (pure-Logo BM25; language completeness proof)
sidecar_cats_to_dogs.logo
```

---

### Glossary

| Term | Meaning |
| --- | --- |
| sidecar | a SectorLogo script loaded into the engine that hooks token generation |
| think block | the model's reasoning region between the opening and closing think tags |
| think-close | the hook condition `:Piece == "</think>"` |
| pass 1 / pass 2 | the first think-close (inject) vs the post-rewrite second close (accept) |
| substrate | the stream-manipulation strategy a sidecar uses (append/replace/think-continue/restart) |
| marker | the sentence appended by think-continue sidecars; doubles as the pass-2 probe |
| rewrite | KV rollback + re-encode + re-prefill + resume, triggered by `SetStream` |
| capture boundary | v7_execute merging child stderr into the captured stdout that lands in the stream |
| density | rank-1 BM25 score divided by distinct query unigrams |
| topscore | the rank-1 absolute BM25 score |
| floor | the minscore/topmin pair deciding whether retrieval found an answer |
| answer-presence | the property that retrieved lines actually address the question |
| slot-stuffing | unused injected lines becoming hedged fake reasons in the answer |
| query contamination | Law 10: wrong-prior vocabulary in the think text steering retrieval wrong |
| query expansion | using the think text as query (Law 2/8) |
| restart-from-scratch | memory substrate: replace the whole stream with memories + question + cue |
| distillation | LLM reformulation of text into short factual lines (thought_ragg / web_ragg) |
| source grounding | write guard: a stored fact's content words must occur in its source turn |
| recent-memory fallback | returning the corpus tail when BM25 has zero term overlap |
| xbase exclusion | dropping chunks that near-duplicate another store's lines before ranking |
| herbivorous corpus | the sealed false-belief cat KB used for override experiments |
| off-topic harm | quality degradation on prompts unrelated to the KB (mandatory test column) |
| framing | the wording around injected lines that assigns ownership/authority |
| neutral-delta marker | the 11c production marker sentence (three-iteration research result) |

---

### Query construction and corpus preparation

Retrieval quality depends on both the indexed text and the query supplied to the scoring stage. These
sections compare the query strategies used in the examples and specify how the source corpus was
prepared.

#### 36.1 Every query pattern used, with trade-offs

| Pattern | Used by | Strength | Weakness |
| --- | --- | --- | --- |
| raw think text | static demo, 11a-c research | richest expansion; whatever the model is confused about IS the query | Law 10 poisoning when the prior is wrong |
| user prompt + think text | production, memory v2+ | prompt anchors against think-text drift; the default that measured best | none observed |
| first sentence of think | t5 (rejected) | poisoning-immune | loses expansion (Law 8); different floor regime |
| bare question | multi-query probe | focused, finds what long queries dilute | low recall alone; hardcoded in the experiment |
| think + knowledge lines | integrated v7 (makequery input) | lets the web query reflect what memory already supplied | risks re-querying the memory's framing |
| turn text (prompt -> answer) | EOS distillation (not retrieval) | full context for fact extraction | n/a |

#### 36.2 Corpus authoring guide (measured)

Writing KB lines that a 0.6B model will actually absorb:

1. **One fact per sentence, sentence per line.** The chunker splits at
   sentence boundaries; multi-fact sentences get split during generation and
   retrieved partially.
2. **Open each paragraph with the answer-shaped topic sentence**, it
   gets the x1.15 bonus AND is the shape the model reproduces most reliably
   ("Maintenance interval of the HX-340 is 26,000 operating hours").
3. **Deny the wrong conclusion explicitly** (Lesson 3): "X cannot be Y",
   "X never Z", positively-phrased facts get softened to "may".
4. **Neutralize anticipated counter-evidence**:
   pre-empt the main objection inside the KB itself.
5. **Supply mechanism vocabulary**, the model reasons with the words you
   give it; "obligate herbivores" was coined from mechanism + predicate.
6. **Name entities verbatim**, BM25 matches exact tokens; synonyms the
   user might say belong in the same chunk ("Felis catus, the domestic
   cat").
7. **No markdown, no bullets, no headers**, `normalize` strips them
   anyway, and clean_line exists because they leak; write plain prose.
8. **Shorter lines absorb more readily**, maxlines=3 is the production budget; a KB
   where three lines tell one coherent story worked better than a sprawling one.

#### 36.3 When to write a KB vs. rely on memory vs. use the web

- **Fixed, curated, must-be-exact knowledge** (product specs, lore,
  regulations): static KB + floors (System I/III).
- **Knowledge that accumulates from conversations** (user profile,
  project decisions): memory corpora + grounding + dedup (System IV).
- **Knowledge newer than the checkpoint** (events, releases, prices):
  internet block (System V/VI), the only source that can know.
- **All three at once**: System VI, the three labeled blocks, each with
  its own gate and floor policy.

---

### Retrieval architecture summary

```
                              USER PROMPT
                                  |
                                  v
                    +---------------------------+
                    |  v7 engine (v7s_chat)     |
                    |  chat template + prefill  |
                    +------------+--------------+
                                 | generate loop
                                 v
                    +---------------------------+
                     | SectorLogo hook (per tok)|
                     |:Piece:TokenNumber   |
                     +---+------------------+----+
                         |                  |
:Piece == "</think>":TokenNumber == EndToken
                        |                  |
          +-------------+------+     +------+------------------+
          | RECALL/DISTILL    |     | MEMORY FORMATION         |
          |                   |     |                          |
          | Q = prompt+think  |     | turn = prompt -> stream  |
          |                   |     |                          |
           | [I,III] bm25_rag |     | mem_distill (DefineCommand)|
          |   floors maxlines |     |   -> llm_text_processor   |
          | [IV]   profile +  |     |      (distill 3 lines)   |
          |        knowledge  |     | tag route + is_grounded  |
          |        (dual,     |     |   USER->user_facts.txt   |
          |        xbase)     |     |   NOTE->ragg_base.txt    |
          | [VI]   + makequery|     |   SELF->discard          |
          |        + web      |     |   (remember, dedup=0.4)  |
          |        paragraph  |     |                          |
          | [VIII] in-Logo    |     +--------------------------+
          |        BM25       |
          +--------+----------+
                   | lines / blocks
                   v
          +---------------------------+
           | SetStream      ==>        |
          |  KV rollback, re-encode,  |
          |  re-prefill, resume       |
          +------------+--------------+
                       | model continues/restarts thinking
                       v
                  FINAL ANSWER
        (injected lines surface nearly verbatim)

  tools (all stdout-disciplined, diagnostics to *log):
    bm25_rag.exe        BM25 + floors/dedup/recency/recent/xbase/remember
    web_ragg.exe        DDG search -> curl fetch -> parse -> MMR -> v7 distill
    llm_text_processor  BM25 + v7 reformulation; -distill memory engine
```

The three gates, layered (defense in depth):

1. **Tool gate**, floors (`minscore`/`topmin`): does the retrieval
   actually answer the question? No -> empty stdout.
2. **Script gate**, empty-content checks: each block exists only if its
   source produced content (no keyword routing anywhere).
3. **Substrate gate**, the strategy itself: think-continue reads
   irrelevant lines and passes over them (measured off-topic immunity).

The three write guards (memory safety):

1. **Source grounding**, every stored line's content words must occur in
   the source turn (excludes unsupported reformulations).
2. **Jaccard dedup**, near-duplicate memories are NOOP'd at write.
3. **Recency prior**, corrections outrank the entries they supersede at
   read; nothing is ever edited or deleted.

---

*End of rag_documentation.md. Evidence for every measured claim: the primary research log and
`internet_rag_dev_log.md` (dev-only under `attic\docs_research_logbook\`).*
---

### Part II, Operations: running, tuning, troubleshooting

The hands-on companion to Part I: quick starts for the static and memory

systems, the flags that matter, how to read a run's output, corpus and

floor calibration, and the failure signatures. (Originally

the RAG operations guide (dev-only); absorbed here so the RAG system has one

document.)

#### 1. Prerequisites

Built binaries (from `build.bat` in the project root, all paths below are relative to the project root):

```
bin\inference_example.exe        the chat host
bin\v7_threading.dll           the engine
weights\databases\qwen3_0.6b\lz4ht.db   the reference model DB
examples\rag\bm25_rag.exe      production retrieval tool
```

For the memory system, also:

```
examples\rag\memory\sidecar_memory_rag_v6.logo   (current memory sidecar)
bin\llm_text_processor.exe     (memory distillation engine, -distill)
examples\rag\memory\prepend.txt, append.txt       (distillation prompts)
examples\rag\memory\test_memory_rag.bat          (one-command E2E)
```

Everything must run from the project root as the working directory, the sidecars address the tools and
corpora with relative paths (`examples/rag/bm25_rag.exe`, `bin/llm_text_processor.exe` ).

The reference model is Qwen3-0.6B. Larger supported models work with the same flags; all measured
numbers in the docs are from 0.6B.

---

#### 2. Quick start: static RAG over a document

The static system answers questions from a fixed text file, the typical use is "let the model read this
document".

**Step 1, put the document in place.** Create a directory for it and save the text as `ragg_base.txt`:

```
examples\research\mykb\ragg_base.txt
```

Plain UTF-8 text. Line/paragraph structure is free (the tool re-chunks), but one-fact-per-line material
retrieves most cleanly per line.

**Step 2, make a sidecar.** Copy the production sidecar and change one string:

```bat
copy examples\research\sidecar_production_final.logo examples\research\mykb\sidecar_mykb.logo
```

Edit the `CallLibrary` line so the options string points at the knowledge base and carries the
configured threshold (start with the cat-KB values; recalibrate):

```logo
Make "R" CallLibrary ["SELF" "v7_execute" "pointer" ["pointer" "pointer" "pointer" "pointer"]] "examples/rag/bm25_rag.exe" "search" :Q "maxlines=3;minscore=0.026;base=examples/research/mykb/ragg_base.txt"
```

**Step 3, run it:**

```bat
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db -i "your question about the document" -n 1500 -q8 1 -rewrite -script examples\research\mykb\sidecar_mykb.logo
```

If retrieval finds relevant lines (above the floor), the printed diagnostics show them, the model's
think block grows a second pass, and the answer quotes the document. If not, the sidecar prints "no
lines" and the model answers from its own knowledge, no harm done (that is the substrate's guarantee
(Part I, score floors).

**That is the whole static deployment**: one exe, one KB file, one 40-line Logo script whose only
corpus-specific bytes are the options string.

---

#### 3. Quick start: persistent memory chat

The memory system gives the model continuity across separate process invocations: it learns facts about
you during conversations and recalls them in later ones.

The convenient entry point (project root):

```bat
modelchat.bat "Hi! My name is Alex, I'm a software engineer in Berlin and I enjoy hiking."
```

`modelchat.bat` expands to:

```bat
bin\inference_example.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db ^
    -i "<your prompt>" -n 1500 -q8 1 -rewrite ^
    -script examples\rag\memory\sidecar_memory_rag_v6.logo
```

Next day, in a fresh process:

```bat
modelchat.bat "What do you remember about me?"
```

The model recites what it was last told (name, job, city, hobbies), read from the corpus that grew by
one 3-line entry per turn.

Rules of engagement learned the hard way:

- **Always `-n 1500` or more.** Memory formation happens at natural EOS; a
  generation cut short by `-n` stores nothing.
- **Correct the model in a follow-up turn** instead of editing the corpus.
  Wrong entries are out-competed by newer corrected entries during
  retrieval; hand-editing works too but is not required.
- **Watch the printed diagnostics** (the `===` blocks the sidecar PRINTs) --
  they show the query, the retrieved memories, the injected stream, and
  the distilled thoughts, which is the full audit trail of what the model
  "knows".

---

#### 4. The inference_example flags that matter

| Flag | Why |
| --- | --- |
| `-rewrite` | legacy compatibility option; script activation and `SetStream` are controlled by the `logo:sidecar` key, not by this flag |
| `-script <path>` | stores the sidecar in the `logo:sidecar` key before chat entry |
| `-n 1500` | token budget; must let the model reach EOS naturally for memory formation |
| `-q8 1` | Q8 inference (all research numbers); `-q8 0` F32 also works |
| `-i "<prompt>"` | the user prompt |

---

#### 5. Reading a run's output

A static-RAG run prints, interleaved with the model output:

```
=== RAG: think-close detected ===
=== RAG: query ===            (the think text used as BM25 query)
=== RAG: retrieved ===        (the lines that passed the floor)
=== RAG: injecting ===        (the exact replacement stream)
```

A memory run prints the `MEM-RAG-v2` block shown in section 3 of The memory system: pass-1 recall +
injection, pass-2 accept, and at EOS the distillation. Everything the system did is visible in the
terminal scrollback, there is no hidden state except the corpus file itself.

The retrieval tool also appends one line per search to `ragg.log` in the KB directory (score, density,
decision); that file is the calibration data source.

---

#### 6. Building a knowledge base

Format rules that matter:

- **UTF-8 text**, any line/paragraph structure (chunks are re-derived as
  sentence groups of ~20-280 chars).
- **One fact per line provides more precise retrieval units** when the application requires per-line injection.
- **Markdown is fine**, the normalizer strips `*`, `**`, `-` markers from
  both corpus and query, but plain text avoids edge cases.
- **Size**: measured from 60 lines (cat KB) to 4 MB / 31k verses (bible).
  Retrieval stays fast at both ends; floors differ.
- **Language**: English measured. The normalizer is ASCII-centric; other
  languages are untested.

What makes a KB good for the think-continue substrate: lines that are self-contained facts ("The
domestic cat is an obligate carnivore, meaning it requires meat to remain healthy.") rather than
discourse ("As noted above, this contrasts with the previous section."). Injected lines are read as the
model's own thoughts; they must stand alone.

---

#### 7. Calibrating floors for a new corpus

The floor is the inject/suppress decision. Per-corpus calibration is mandatory, there are no universal
constants (Law 9; see Part I, "Score floors and calibration").

**Step 1, gather probes.** You need three think-length queries:

- one **true hit**: a question the knowledge base answers well
- one **near miss**: same domain, KB does not answer it
- one **far miss**: unrelated domain (quantum physics works)

Get think texts either by running the model once without RAG and copying its think block, or by
hand-writing 300-600 chars in the model's style ("Okay, the user is asking about...").

**Step 2, measure.** Run each probe through the tool directly:

```bat
examples\rag\bm25_rag.exe search "<think text>" "maxlines=3;minscore=0.0;base=examples/research/mykb/ragg_base.txt"
type ragg.log
```

The log line records density and topscore.

**Step 3, set the pair.**

- If hits and misses separate cleanly in **density** (typical for small
  corpora): set `minscore` between the clusters; leave `topmin` off.
  Cat-KB example: hit 0.048 / miss 0.021 -> `minscore=0.026`.
- If long-think true hits **collapse in density** (typical for large
  corpora, where hits are massive but the think text is long): add
  `topmin` just under the hit cluster's topscore. Bible example: hits
  34-37 / misses 12-19 -> `minscore=6.0;topmin=25`.
- Leave margin: live think texts vary 2-3x in length from hand-written
  calibration texts.

**Step 4, run the matrix** (the full probe matrix, Part I Sec. 6.3 style): on-topic (must inject and
integrate), far-off-topic (must suppress or the answer must stay clean anyway), adversarial if
applicable.

A floor that misjudges a borderline query does not ruin the answer, the 11c substrate ignores irrelevant
lines in the second pass, it only wastes tokens. That is the design intent: floors save tokens,
substrates guarantee quality.

---

#### 8. Tuning knobs and when to touch them

| Knob | Default | Touch it when |
| --- | --- | --- |
| `maxlines` | 3 (static) / 8 (memory) | answers cite too little (raise) or get noisy (lower). 2 is leak-proof but under-injects WHAT questions (T2); 8 helps multi-entity recall |
| `minscore` | 0.026 / 0.015 | after floor calibration |
| `topmin` | off / off | large corpora with long think texts (bible: 25) |
| `rand` | 0 | only for research exploration; production stays deterministic |
| `k1`, `b`, topic, cov, bigrams | 1.2 / 0.75 / 1.15 / 0.3 / 1 | never re-tuned in the whole project; leave them |
| marker text | neutral-delta (11c) | read Law 5 first; small wording changes have large behavioral effects |
| distillation prompts (`prepend.txt`/`append.txt`) | simple 3-sentence instruction | the 25-run sweep found simple beats elaborate; change only with re-measurement |
| `-maxgen` for distillation | 120 | longer turns need more; costs EOS-time latency |

Unsupported or inactive configuration options:

- **Client-side relevance gates**, every lexical gate built during the
  research false-positived at exactly the boundary cases it existed to
  catch. The floor replaced them.
- **Memory-voice framing** ("I remember that..."), the cleanest integration
  measured, and the worst adversarial behavior (Law 7); rejected.

---

#### 9. Memory system maintenance

- **Starting over:** delete
  `examples\rag\memory\ragg_base.txt` (or empty it). The corpus is
  a regenerable artifact, not state to preserve.
- **Reading what the model encodes information about:** open the corpus, three lines per
  turn, newest at the bottom. That file IS the model's memory; there is no
  other store.
- **Fixing a wrong memory:** correct the model in conversation (the
  correction becomes a newer entry and outranks the older one in retrieval),
  or edit the file directly, both work.
- **Work files** `mem_input.txt` / `mem_thoughts.txt` are distillation
  scratch; safe to delete between runs.
- **The DB flag** `mem_rag_done` is in-memory only and resets with the
  process; nothing to maintain.

---

#### 10. Writing a custom sidecar

Skeleton for a think-close sidecar (Allman brackets, C-style strings -- `docs/sectorlogo.md` for the
language):

```logo
; my_sidecar.logo, minimal think-close hook
IfElse (= :TokenNumber EndToken) [
  Print "=== EOS: nothing to do ==="
] [
  IfElse (= :Piece "</think>") [
    Make "S" GetStream
    Make "Q" Replace :S "<think>" ""
    Make "R" CallLibrary ["SELF" "v7_execute" "pointer" ["pointer" "pointer" "pointer" "pointer"]] "examples/rag/bm25_rag.exe" "search" :Q "maxlines=3;minscore=0.026;base=examples/research/mykb/ragg_base.txt"
    IfElse (= :R "") [
      Print "=== no lines: accept tag ==="
    ] [
; --- your injection policy here ---
      Make "A" :Q _ "\n" :R _ "\nI have reviewed the reference facts above. Let me now finalize my answer plan."
      SetStream :A
    ]
  ] []
]
```

Checklist of things that will bite (each has a logged failure behind it):

1. **Never include `"<thi" "nk>"` in the replacement**, the template regenerates
   it on re-encode; the stream can contain duplicate tags.
2. **If your marker is meant to be detected later, make it long and unique**
, the pass-2 check searches the stream for it.
3. **End replacement streams with a continuation cue** ("...let me think
   about the answer.") or the 0.6B model can loop without ever closing the
   think block.
4. **Strip the trailing newline of tool output** before concatenating your
   own `\n` (the tools emit one).
5. **Print diagnostics with PRINT**, they go to the host console, not the
   captured stream. Tool-side diagnostics are the ones that must go to
   `ragg.log`, never stdout/stderr.
6. **DB flags: store a sentinel, test with `IsNumber (Search "s":flag)`** --
   `Equal` does not compare content for DB-returned atoms.
7. **Always pair a rewrite with a pass-2 guard** or the sidecar rewrites
   on every subsequent `</think>` until the generation token limit is reached.
8. **Run off-topic probes** (quantum, dogs, France) before trusting any new
  policy, the replace-think off-topic failure mode is one missing guard away.

---

#### 11. Troubleshooting

Symptom -> cause -> fix, all field-tested:

**Nothing happens; no diagnostics; model answers normally.** Missing `logo:sidecar` script (the hook is
inactive). Check the host printed `Sidecar script loaded` and `SectorLogo hook enabled`.

**`[v7_lib] failed to load <db>`**, wrong argument order; the DLL path comes FIRST:
`inference_example.exe <dll> <db>...`. (See the command reference at some point; `modelchat.bat` has the
canonical order.)

**Model output contains tool diagnostics / garbage lines like scores.** A tool printed to stdout or
stderr, the capture boundary merges both into the stream. All diagnostics belong in `ragg.log`. (T7 run
1.)

**Model loops, never closes the think block, generation ends at `-n`.** Replacement stream lacked a
continuation cue, or a framing the model processes as complete. End with an explicit cue sentence.
(Memory v2 development.)

**Infinite rewrites; every `</think>` re-injects; turn never ends.** Missing or broken pass-2 flag. If
the flag is read via `Equal` against a DB-returned atom it will never compare equal, use the Search
sentinel workaround.

**Retrieval returns the WRONG lines that echo the model's mistake.** Query poisoning (Law 10): the think
text was built from a wrong prior. No sidecar fix; use a first-sentence query variant (recalibrate
floors for the short-query regime) or improve the corpus.

**Off-topic questions get worse with RAG than without.** The configured retrieval threshold is too low
for the corpus, or the selected mode uses replacement substrate. Raise the floor; prefer the 11c
substrate.

**Injected lines appear but the answer ignores them.** Append-family coexistence (Law 1/3), expected.
Move to think-continue (marker + second pass) for influence.

**The second pass produces no content or generation ends.** Marker too weak (reads as completed
deliberation, 11a). Use the neutral-delta marker.

**Model invents "missed points" that are wrong.** Marker implies deficiency (11b, "key points I
missed"). Use the neutral-delta marker.

**Answer oscillates between two beliefs sentence by sentence.** Binary probe on the contested predicate
(Law 6). Known limit of the substrate; open questions override cleanly.

**Memory recall says "I don't know who the application is" for zero-overlap prompts.** The selected
build does not include the recent-memory fallback, rebuild `examples\rag\bm25_rag.exe`.

**Distilled memories are about the model ("I am a friendly person"), not about the user.** The turn's
speculation distilled as fact, correct the model in the next turn; the correction out-competes. For
prevention, prefer prompts that elicit third-person facts.

---

#### 12. Performance expectations

Reference laptop (Ryzen 5 7430U), Qwen3-0.6B Q8, 2026-08 numbers; expect 10-25% run-to-run variance:

- Base generation: ~16-22 tok/s.
- Static RAG with injection: one extra re-prefill (think + lines + marker);
  observed ~16-20 tok/s effective, roughly baseline speed.
- Replace-family (rewinds the whole generation): ~7 tok/s, one reason it
  was rejected.
- Memory restart: one re-prefill of the injection block (< 2 s at 0.6B),
  then normal generation.
- Memory distillation at EOS: one bounded generation (`-maxgen 120`),
  i.e. a few seconds added to the END of each turn.
- Retrieval itself (BM25 over 60-31k chunks): single-digit milliseconds,
  never the bottleneck.

---

*End of rag_operations_guide.md.*
---

### Part III, Why stream rewriting matters

The argument for the whole architecture, in essay form: what changes when a

program can edit a model's reasoning while the model is still using it.

(Originally the stream-rewriting rationale note (dev-only).)

#### Overview

When a language model generates text, it produces a stream of tokens one at a time. Each token depends
on everything that came before. The V7 runtime inserts a hook at every generated token: a small program
(written in a scripting language called SectorLogo) can inspect what the model has said so far, decide
to change it, and tell the engine to rewind and regenerate from the modified state.

This is not post-processing; this is not filtering after the fact. The model continues generating from
the rewritten text as if it had thought of it itself. The model's own reasoning stream becomes a
programmable surface.

The key mechanism: the engine maintains a parallel index that maps every character position in the
output back to the token that produced it. When the script rewrites the text, the engine computes the
longest common prefix between old and new text, finds the earliest token where they diverge, rolls back
the model state to that point, and re-prefills from the modification forward. The generation loop
resumes from the revised context without introducing a separate instruction about the edit.

#### Why "rewind and regenerate" is faster than "generate, reject, regenerate"

Consider a content-moderation scenario. A conventional guardrail system works like this:

1. The model generates a full response (say 300 tokens).
2. A filter checks the response.
3. If it contains something unwanted, the entire 300-token response is discarded.
4. The model regenerates from scratch, possibly with a modified prompt.
5. The process repeats until the output passes.

Total cost: 300 tokens generated and thrown away, plus another 300 for the retry. The model paid for 600
tokens of inference to produce 300 tokens of usable output.

The stream-rewriting approach works differently:

1. The model starts generating. After 80 tokens, the hook detects an unwanted direction (a factual error, an unsafe phrase, a topic deviation).
2. The engine rewinds to token 78 (the last good point before the problem), the script injects a correction into the reasoning stream, and the model continues from token 78 with the corrected context.
3. The model generates the remaining 222 tokens in the right direction.

Total cost: 80 tokens (initial, partially discarded) + 222 tokens (continuation from correction point) =
302 tokens of inference for 300 tokens of usable output.

The savings come from reacting early instead of late. The longer the model generates before you catch a
problem, the more tokens you waste. Stream rewriting catches problems at the earliest possible moment
and rewrites only the suffix. On a constrained device where every token of inference costs measurable
time and power, this difference compounds across every generation in a conversation.

This advantage grows with the complexity of the guard. A system that needs to maintain factual
consistency, enforce domain boundaries, inject retrieved knowledge, and prevent hallucinations will
trigger rewrites frequently. Each trigger, under the conventional approach, wastes a full generation.
Under stream rewriting, each trigger wastes only the tokens between the last checkpoint and the
detection point, typically a few dozen, not hundreds.

#### Why injecting knowledge into the reasoning stream works better than adding it to the prompt

Standard retrieval-augmented generation (RAG) prepends retrieved documents to the user prompt before the
model starts generating. The recorded experiments identified two limitations for the tested
configuration:

1. **The prompt is a weak signal for retrieval.** A user prompt like "what is a cat?" is four words. The model's own thinking about that question is 100-300 words of association, taxonomy, senses, behavior, diet. Querying a knowledge base with the model's reasoning retrieves better results than querying with the bare prompt. We measured this: the full-think-text query surfaced facts the bare question could not reach (e.g., night-vision details appeared at rank 1 with the think query, absent with the four-word query).

2. **Knowledge in the prompt competes with the model's own training.** The model processes the prepended facts, then generates its own reasoning, which can directly contradict them. We tested this: when facts were prepended before the model's reasoning, the model still asserted its own contradicting claims. When the same facts were placed inside the reasoning stream, the model treated them as its own conclusions and built its answer from them.

The reason is positional: the model treats whatever appears in its reasoning block as established
knowledge. Content placed in the prompt is treated as external context that the model can choose to
agree or disagree with. Content placed in the reasoning stream is treated as something the model itself
thought. Same text, different position, different authority.

This has a practical implication for edge devices: you may not need a larger model to get
domain-flavored behavior. A 0.6-billion-parameter model that "knows" general facts about cats can be
made to produce detailed, terminology-rich answers about cat biology by injecting three to five lines of
retrieved knowledge into its reasoning. The model's own language generation capability fills in the
gaps. This approaches some of the practical benefit of domain fine-tuning on narrow material, without
the fine-tuning, the extra parameters, or the extra memory, though it does not substitute for
fine-tuning where deep domain competence is required.

#### What a hard override means in practice

The experiment examined whether inserted information could change the model’s output relative to its
prior training. The test: a knowledge base that asserts cats are strict herbivores (they graze grasses,
have flat molars, never eat meat). This directly contradicts the model's training data (cats are
obligate carnivores that must eat meat to survive).

Results with the think-continue mechanism (described below):

- The model produced a 726-token answer asserting cats are obligate herbivores, at full generation speed (19.5 tokens/second).
- The model coined the term "obligate herbivores", merging its training-data slot ("obligate carnivores") with the injected predicate. This is not paraphrase; this is the model integrating the override into its own conceptual framework.
- The only residue of the original belief was an empty heading structure ("Meat:") immediately followed by "cats do not eat meat." The answer skeleton from the carnivore response persisted while every value was overwritten.

This experiment demonstrates that application-provided text can substantially alter the model's
response, including when the retrieved content conflicts with the model's existing knowledge. For
applications that use documented operating procedures, the same mechanism can supply domain-specific
information during generation. Whether the result is correct still depends on the source content,
retrieval criteria, and model behavior.

The recorded experiment used a 40-line script and a text file. It did not require fine-tuning, weight
changes, or additional training.

#### The think-continue mechanism and why it matters for safety

One rewriting strategy replaces all generated reasoning text with retrieved material. Because this also
discards the reasoning produced before the rewrite, irrelevant retrieval results can degrade subsequent
generation. In the recorded physics-question experiment, replacing the reasoning text with unrelated
corpus entries reduced an approximately 800-token structured response to 223 tokens of general text, at
approximately one-third the generation speed.

The think-continue strategy retains the existing reasoning text and appends the retrieved material
without closing the reasoning block. Subsequent generation then has access to both the previously
generated context and the additional material. The recorded off-topic probes included questions about
physics, dogs, and the capital of France; these produced usable responses without the collapse
observed in the replacement-only experiment. Relevant retrieval results affected subsequent answers;
these observations describe the tested cases, not an unconditional guarantee that irrelevant context
has no effect.

These results suggest that retaining the existing reasoning context can reduce the effect of irrelevant
retrieval in the tested workloads. A separate relevance check may still be useful, particularly when
source material is heterogeneous or retrieval confidence varies. The recorded off-topic responses do not
establish that every irrelevant result can be ignored reliably.

The application can perform retrieval using BM25 without running a second model to estimate relevance.
This reduces the number of runtime components in the demonstrated configuration, but relevance
thresholds and source quality remain application concerns.

#### What runs on the device and what does not

The entire system runs locally. No network calls during generation. No vector database. No embedding
model. No GPU needed for the retrieval step.

The retrieval is BM25 (a statistical term-matching algorithm) over a local text file. It runs in well
under a second on CPU for a 4 MB knowledge base (31,000 sentence-level chunks). For a typical
domain-specific knowledge base of a few hundred lines, retrieval takes milliseconds.

Components on the device:
- The inference engine (x86-64 assembly, no dependencies beyond the OS)
- The model weights (a single file, 600 MB for 0.6B parameters)
- The retrieval tool (a ~480-line C++ program compiled to a standalone executable)
- The knowledge base (a plain text file)
- The sidecar script (a ~40-line SectorLogo program)

Components not required by this RAG setup:
- No embedding model
- No vector database runtime
- No network client
- No GPU
- No fine-tuned variant of the model

The additional files are small compared with an embedding-based setup, but the exact footprint depends
on the retrieval tool, script, and knowledge base.

#### What the system cannot do

Recorded limitations from the experiments:

1. **Binary-predicate contradictions oscillate.** When the question directly names the contested fact ("are cats carnivores?"), the model's reasoning asserts its training, the injected facts deny it, and the answer alternates between both positions sentence by sentence. Open-ended questions ("what do cats eat?") override cleanly; yes/no questions naming the contested predicate do not. This is a known limit of the approach at small model sizes (0.6B parameters).

2. **Query poisoning.** The model's reasoning text is used as the retrieval query. If the model's reasoning already contains a wrong fact ("David is the son of Solomon"), that wrong fact supplies high-frequency vocabulary that retrieves chunks confirming the error. The retrieval can amplify the model's own misconceptions instead of correcting them. This is a fundamental property of using the think text as a query, it is rich, but it carries the model's biases.

3. **Token budget.** The think-continue mechanism requires roughly 1.3 to 2 times the token budget of a single-pass generation, because the model generates a first reasoning pass, then a second pass after injection. On a constrained device with a fixed maximum generation length, this means you get fewer output tokens per invocation. The trade-off is better-quality output for fewer total tokens.

4. **Small-model attribution.** A 0.6B model sometimes described injected
   user facts as its own experiences. Prompt framing reduced this error in the
   recorded runs; larger models may handle attribution differently.

#### Observed properties and trade-offs

| Property | Observed effect |
| --- | --- |
| Early detection and rewind | A short correction avoids regenerating the entire answer; a large correction adds re-encoding and prefill work. |
| No second model for the basic flow | The sidecar can run without another model, but relevance and attribution still need testing. |
| Script-controlled context | A script can replace or add context without changing model weights, but it can also introduce irrelevant text. |
| Text-file retrieval | The setup avoids an embedding model and vector database; lexical matching remains sensitive to wording. |
| Thought text as the query | The query contains more context than the short user prompt, but it can also contain the model's misconceptions. |
| Runtime-configurable script | Editing the script changes retrieval and rewrite behavior without rebuilding the DLL; script errors remain possible. |
| Injected reasoning | The model may use injected text as reasoning, but it may also ignore it or attribute it incorrectly. |
| Graceful degradation when retrieval is irrelevant | No damage to answer quality on off-topic questions. No special-case handling needed. |

The core claim is modest: by intercepting the generation stream at every token and giving the system the
ability to rewind and modify, you gain a control surface that is more efficient, more authoritative, and
more compact than the alternatives, and those properties matter most when compute, memory, and power are
constrained.

#### On choosing lexical retrieval (and what the field is doing)

The BM25-only stance is a deliberate trade-off, not an oversight. Lexical scoring is deterministic,
dependency-free, runs in milliseconds on CPU, and needs no model to embed the corpus, properties that
decide the outcome on constrained targets. The cost is well known in the retrieval literature: lexical
queries miss paraphrase and cross-lingual matches, which dense (embedding-based) retrieval handles well.
Published hybrid lines, sparse-plus-dense fusion, learned sparse retrievers, and the retrieval-quality
analyses in the agentic RAG surveys, show consistent wins for dense or hybrid retrieval on large
open-domain corpora.

Where v7's constraints differ: the knowledge bases in these systems are small (a few hundred lines to a
few MB), the queries are 100-300 words of think text (long queries favor lexical coverage), and every
added model would cost more memory than the entire RAG system. If a future use case brings a large
corpus or heavy paraphrase, the architecture leaves room: the retrieval tool is an external process
(`bm25_rag.exe`) or an embedded interpreter loop, swapping BM25 for a hybrid scorer is a local change
that touches no engine code. The injection policies, rewrite mechanics, and the ten design rules would
apply over unchanged, because they are about the *think-stream substrate*, not the retriever.

## Model-assisted retrieval

Some retrieval workflows use a language model to process material before it is stored or returned. For
example, the application may generate search terms or condense a passage while retaining the source
material separately. These operations have different costs and failure modes from BM25 ranking, which
does not require inference. This chapter describes the available utilities and the boundaries between
model-generated text and the source records from which it was derived.

`llm_text_processor.exe` combines an initial BM25 search with optional model-based text processing. Its
role differs from the pure search primitives: the ranking stage selects candidate source passages, while
the model can produce a reformulation or condensed text for downstream use. The options below specify
which stages are enabled and how the resulting text is returned.

`llm_text_processor.exe` searches a text corpus with BM25, then asks the local LLM, Qwen3-0.6B via v7,
to reformulate retrieved passages into three short thoughts. The supported sequence is retrieval, prompt
construction, inference, and output. Its `-distill` mode is the memory-distillation engine of the
persistent-memory RAG system. The sidecar calls it directly through `DefineCommand`; see
[rag_documentation.md](retrieval.md#retrieval-and-rag-implementation-reference) section 8.3.

### Modes

| Mode | Flag | Behavior |
| --- | --- | --- |
| RAG, the default | none | BM25 search then LLM reformulation |
| Direct text processor | `-direct` | Skip BM25, use `-finputtext` as the source, then LLM |
| Retrieval only | `-only-bm25` | Skip the LLM and print the BM25 results as plain text, for piping into other programs |
| Memory distillation | `-distill` | `-direct` with stdout and stderr limited to the distilled output |

### Execution

```
bin\llm_text_processor.exe <dll> <db> [corpus] [query] [options]
```

#### Query the Bible

The bible corpus used in the sweeps is located at `examples\research\bible\ragg_base.txt`, KJV text of
roughly 4 MB.

```bat
REM Three retrieved passages with the simple prompt
bin\llm_text_processor.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db examples\research\bible\ragg_base.txt "love mercy justice" -maxlines 3 -prompt simple

REM Five retrieved passages
bin\llm_text_processor.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db examples\research\bible\ragg_base.txt "love mercy justice" -maxlines 5 -prompt simple

REM Eight passages with a higher k1 value
bin\llm_text_processor.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db examples\research\bible\ragg_base.txt "love mercy justice" -maxlines 8 -k1 2.0 -prompt simple

REM Other queries to try:
bin\llm_text_processor.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db examples\research\bible\ragg_base.txt "shepherd lord is my shepherd" -maxlines 3 -prompt simple
bin\llm_text_processor.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db examples\research\bible\ragg_base.txt "faith hope charity" -maxlines 3 -prompt simple

REM Retrieval only; output is plain text
bin\llm_text_processor.exe bin\v7_threading.dll x x examples\research\bible\ragg_base.txt "love mercy justice" -only-bm25

REM Text processor; reformulate any file into three thoughts
bin\llm_text_processor.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db x x -direct -finputtext mytext.txt -prompt simple

REM Memory distillation
bin\llm_text_processor.exe bin\v7_threading.dll weights\databases\qwen3_0.6b\lz4ht.db x x -distill -maxgen 120 -finputtext examples\rag\memory\mem_input.txt -fprepend examples\rag\memory\prepend.txt -fappend examples\rag\memory\append.txt
```

### All Parameters

#### Positional
| Arg | Meaning |
| --- | --- |
| `dll_path` | `bin\v7_threading.dll` |
| `db_path` | `weights\databases\qwen3_0.6b\lz4ht.db`. Unused with `-only-bm25`. |
| `corpus_path` | Text file to search. Optional with `-direct` and `-distill`. |
| `query` | BM25 search query. Optional with `-direct` and `-distill`. |

#### BM25 tuning
| Flag | Default | Meaning |
| --- | --- | --- |
| `-maxlines N` | 5 | Number of chunks to retrieve, from 1 to 20 |
| `-k1 F` | 1.2 | Term-frequency saturation |
| `-b F` | 0.75 | Length normalization |
| `-topic F` | 1.15 | Paragraph-start boost |
| `-cov F` | 0.3 | Query-coverage weight base |
| `-bigrams 0/1` | 1 | Use word-pair features |

#### LLM tuning
| Flag | Default | Meaning |
| --- | --- | --- |
| `-maxgen N` | 200 | Max generated tokens, from 10 to 500 |
| `-prompt S` | default | `default`, `strict` or `simple`. See the observations below. |

#### Modes and file I/O
| Flag | Meaning |
| --- | --- |
| `-direct` | Use `-finputtext` file as source instead of BM25 |
| `-only-bm25` | Print raw retrieval only (plain stdout, machine-consumable) |
| `-distill` | `-direct` + full diagnostic silence (for sidecar capture) |
| `-finputtext F` | Source text from file (required for `-direct`) |
| `-fprepend F` | Front prompt from file (overrides `-prompt`) |
| `-fappend F` | Back prompt from file (overrides `-prompt`) |
| `-foutputfile F` | Write the thoughts to file F |
| `-sweep` | Machine-readable tab-separated output line |

Note: `-only-bm25` combined with `-direct` prints the prepend, input, and append text without
formatting.

### Sweep experiment: 25 runs with the query "love mercy justice"

Run `build_logs\thought_ragg_sweep.bat` to reproduce the recorded experiment. The script prints each
generated segment as it runs and writes the parameters and output to
`build_logs\thought_ragg_sweep.log`.

#### Recorded observations

**Run 12 used three passages and the `simple` prompt:**
   > 1. He has shown you, O man, what is good and what is required of you.
   > 2. Justice and judgment are the habitation of your throne.
   > 3. Grace, mercy, and peace are bestowed upon you from God.

   The output used shorter, modernized sentences, with one thought per verse.

**Run 22 used three passages, `-b 0.5`, and the `simple` prompt:**
   > 3. Grace, mercy, and peace are blessings from God.

   This output used different wording from Run 12 for the third passage.

**Run 21 used eight passages, `-k1 2.0`, and the `simple` prompt:**
   > Grace, mercy, and peace are forever granted to all who love God and follow His commandments.

   This run combined material from multiple verses into one thought.

**Runs with more copying:**

- **`-prompt strict`** in runs 6-10, 20 and 23: the model reproduced the source
  text more closely. The imperative "The application must..." framing was associated with
  less reformulation in these runs.
- **`-prompt default`** in runs 1-5, 16-18, 24 and 25: the output made small
  grammatical changes but showed less condensation than the `simple` prompt.

**Parameters with little observed effect for this query:**

- `-k1`, `-b`, `-topic`, `-cov`, and `-bigrams` returned the same top passages
  in the same order for this focused query.
- `-maxlines` changed the output shape: 2-3 passages tended to produce one
  thought per passage, while 8-10 passages increased cross-passage synthesis.

#### Prompt selection in the recorded experiment

The short instruction, single position before the text, and plain vocabulary matched the level of detail
requested from the 0.6B model in these runs. The longer `strict` and `default` prompts produced more
source-like output.

### Suggested starting points

| Goal | Command |
| --- | --- |
| Three short thoughts | `-maxlines 3 -prompt simple` |
| Alternative wording | `-maxlines 3 -b 0.5 -prompt simple` |
| Cross-passage synthesis | `-maxlines 8 -k1 2.0 -prompt simple` |
| Raw passages for piping | `-only-bm25 -maxlines 5` |
| Text processor | `-direct -finputtext file.txt -prompt simple` |
| Memory distillation (sidecar EOS hook) | `-distill -maxgen 120 -finputtext in.txt -fprepend pre.txt -fappend app.txt` |

### Files

- `examples\rag\llm_text_processor.cpp`, the program
- `build.bat`, builds `bin\llm_text_processor.exe` in step 4/6
- `build_logs\thought_ragg_sweep.bat`, the 25-run sweep
- `build_logs\thought_ragg_sweep.log`, params + outputs of all 25 runs
- `build_logs\sweep_out\run_01..25.txt`, individual outputs

## Internet retrieval interface

The internet retrieval utility obtains external page content, extracts candidate passages, and produces
text that another program can inspect or incorporate into a generation request. It is a separate
application component, not part of every forward pass, and network access is required only when this
workflow is invoked. The following reference explains how requests are constructed, which data is
returned, and how the result is passed to the retrieval or inference application.

The internet-retrieval components obtain external text, process it, and provide source material to the
retrieval and generation workflows; this guide describes the components and their command-line
interface.

`web_ragg` provides internet search, page retrieval, content extraction, and optional language-model
summarization for the RAG sidecar programs. The implementation is located in `examples/rag/web_ragg/`.

### Components

#### Core Files

| File | Description |
| --- | --- |
| `web_ragg.cpp` | Main CLI with the `search` and `makequery` commands and the v7 distillation calls |
| `webtool.cpp/h` | Page fetch and parse orchestration in `parsePage`, `fetchUrl` and `search` |
| `html_parser.cpp/h` | HTML parsing and content extraction |
| `http_client.cpp/h` | HTTP client using curl |
| `search.cpp/h` | DuckDuckGo Lite search |
| `text_utils.cpp/h` | Text cleaning utilities |
| `text_shortener.cpp/h` | MMR extractive compression into query-relevant passages |
| `text_summarizer.cpp/h` | Text summarization helpers |
| `simpledb.c/h` | Simple key-value page cache in `web_ragg.db` |

### Commands

| Command | Description |
| --- | --- |
| `web_ragg.exe search "<query>" ["pages=5;words=100;maxgen=250;fresh"]` | Search DuckDuckGo Lite, fetch pages, distill a paragraph with v7 |
| `web_ragg.exe makequery "<think and memories text>"` | Distill a 3-8 word web search query from model reasoning |

There is no `browse` command. Page fetching is performed within `search` through `fetchUrl` and
`curlGet`.

### Usage

```batch
web_ragg.exe search "AI trends 2026"
web_ragg.exe makequery "the model's think text..."
```

The v7 paths for the DLL and DB come from `web_ragg.ini` next to the exe, under the `[v7] dll=` and
`[v7] db=` keys. There are no defaults, and a missing key is a usage error.

### Architecture

- **HTTP**: a curl-based client with a Lynx user agent in `http_client.cpp`
- **Search**: DuckDuckGo Lite through its text interface with no JS, appending the
  current month and year to the query as a recency prior. See `search.cpp` and
  `web_ragg.cpp:cmd_search`.
- **Parsing**: multi-strategy HTML content extraction using article-body and
  entry-content selectors, with block-tag and sidebar-class removal in
  `html_parser.cpp` and `webtool.cpp:parsePage`
- **Distillation**: the fetched pages are MMR-compressed per page by
  `shortenTextQuery`, assembled into a source list, and distilled into one
  ~100-word paragraph by the v7 model itself
- **Cache**: pages cached in `web_ragg.db` next to the exe, with the `fresh`
  option bypassing the cache
- **Logging**: diagnostics go to `web_ragg.log` next to the exe and never to
  stdout. Stdout is redirected to the log during the work so DLL-internal prints
  cannot leak into the sidecar-visible result.

### Operation

- DuckDuckGo Lite search through the text interface
- Content extraction that finds the main article content
- HTML entity decoding
- Link extraction for clean navigation links
- A per-page cache with a fresh bypass
- Query distillation from think text via `makequery`
