# bm25_rag improvements: BM25+ and other medium-complexity retrieval enhancements

This document is a historical evaluation record. Its measurements and decisions reflect the
implementation at the time of each test, not necessarily the current version.

## 1. Current state

At the time of this evaluation, the 768-line `bm25_rag.cpp` implemented BM25 with k1=1.2 and b=0.75, unigram plus bigram tokenization,
a coverage multiplier of `0.3 + 0.7 * coverage`, a topic-sentence bonus of 1.15x, a dual answer-presence
floor that takes either density or topscore, Jaccard near-duplicate suppression, a recency prior, a
recent-memory fallback, cross-corpus exclusion through xbase, and a NOOP-aware remember path. It is a
tuned production tool that came out of more than 23 experiments.

## 2. BM25+ evaluation

### What BM25+ does

BM25+ (Lv and Zhai, CIKM 2011, "Lower-bounding term frequency normalization") adds a single delta
parameter to the BM25 term-frequency component:

```
BM25:  tfc = tf * (k1+1) / (tf + k1 * (1-b + b * dl/avgdl))
BM25+: tfc = tf * (k1+1) / (tf + k1 * (1-b + b * dl/avgdl)) + delta
```

The problem it addresses is that, in standard BM25, a very long document that contains a query term can score
similarly to a short document that does not contain the term at all. The length normalization is not
properly lower-bounded. As document length grows the term-frequency component approaches zero, so long
documents containing the term end up scoring the same as short documents without it.

### Does it matter for bm25_rag?

**Marginal benefit**, for four reasons:

1. **The chunks are bounded**, with MAXSENT=280 and MINSENT=20. The BM25+
   long-document penalty is designed for collections where document lengths span
   10x to 100x, such as web pages or news articles. After splitting, every chunk
   in this tool is in the 20 to 280 char range, so the length ratio is at most
   about 14x, which barely reaches the danger zone.

2. **There is already a coverage multiplier.** The `0.3 + 0.7 * cov` term
   down-weights chunks that share few query unigrams. That is a different
   mechanism, but it partly addresses the same problem, since a chunk with one
   term match and low query coverage has its score reduced.

3. **The delta parameter needs calibration.** The paper's default is delta=1.0.
   On corpora of 10 to 200 chunks this can shift scores substantially. Without a
   test collection to tune against, the risk is making results worse.

4. **The dual floor already provides the safety-net function** that BM25+'s delta
   provides at web scale. A marginal hit that survives the floor was relevant
   enough to pass the threshold.

### Verdict: skip BM25+

The code change is one line, `+ delta` in the tfc formula, but the calibration burden and the marginal
benefit on a bounded-length chunk collection make it not worth the complexity. It would need per-corpus
tuning plus a test harness to validate.

There is one scenario where BM25+ could help: if bm25_rag started working with much larger documents,
such as full paragraphs as chunks or indexed PDF sections. At that point BM25+ becomes worth adding, and
the change is still one line.

---

## 3. Improvements considered, ordered by impact-to-effort ratio

### 3.1 Query expansion via top-chunk pseudo-relevance feedback

**Problem:** a short query like "cats" does not match chunks about "felines" or "kittens". The model's
think text generates the query, and it may use different vocabulary than the corpus does.

**Solution:** after the first BM25 pass, take the top-1 chunk's terms, add the top 2 or 3 novel terms to
the query, and re-rank. This is standard pseudo-relevance feedback in the Rocchio style, studied since
the 1970s and widely used in information retrieval.

**Implementation:**
1. Run rank_chunks normally and take the top-1 chunk.
2. Extract its unigrams that are not already in the query.
3. Add the 2 or 3 highest-IDF novel terms to qterms.
4. Run `rank_chunks` again using the expanded query.

**Complexity:** about 30 lines with no new dependencies. The re-rank doubles the scoring cost, but with
fewer than 500 chunks the overhead is microseconds.

**Risk:** if the top-1 chunk is irrelevant, the expansion adds bad terms. Two mitigations apply: expand
only when the first-pass topscore exceeds a minimum threshold such as 1.0, and add only terms whose IDF
is above the median, since bad hits carry low-IDF filler terms.

**Why it was rated medium rather than high:** it needs a control parameter such as expand=1/0 or
expand=N plus a minimum-confidence threshold. The implementation is straightforward and the potential payoff is significant because it addresses
the vocabulary mismatch that was the most frequent failure mode in the RAG experiments. The actual gain
would still require measurement.

### 3.2 Proximity boost for bigram-adjacent unigrams

**Problem:** two unigrams that appear adjacent in the query but separated by many words in the chunk get
the same BM25 score as if they were close together. For thought-text queries like "why do cats purr",
the proximity of "cats" and "purr" in the chunk matters.

**Solution:** after scoring, apply a proximity bonus for query unigrams that appear within a window of
about 5 tokens in the chunk. This is a standard technique from term-dependency models, such as the
sequential dependence model from Metzler and Croft, 2005.

**Implementation:**
1. For each chunk, build a term-position map during tokenization that stores token
   positions rather than just counts.
2. After BM25 scoring, for each pair of query unigrams present in the chunk,
   check the minimum position distance.
3. If the distance is 5 or less, multiply the score by a small bonus of 1.1x per
   adjacent pair.

**Complexity:** about 40 lines, plus a small change to tokenization to store positions.

**Risk:** low. Proximity bonuses are additive and small, so they can only reorder near-equal BM25
scores, which is the intended effect.

**Why it was rated medium:** it requires changing the Chunk struct to carry a position map, or a flat
`vector<pair<string,int>>` instead of a plain `StrVec`. That is a structural change but not a deep
algorithmic one.

### 3.3 Better stemming: simplified Porter suffix stripping

**Problem:** the current stemming is `t.size() > 3 && t[t.size()-1] == 's' ? t.substr(0, t.size()-1):
t`, which strips only a trailing "s". It misses -ing, -ed, -ly, -tion, -ment, -able, -er and -est.
"Enjoying" and "enjoy" get different tokens, and "education" and "educate" are unrelated.

**Solution:** implement a simplified Porter stemmer covering step 1 only, stripping common suffixes. A
full Porter stemmer is about 200 lines, while a simplified version handling the top 6 suffixes is about
30.

**Implementation:**
```
- Remove trailing "ing" if stem length >= 4 (running -> runn, swimming -> swimm)
  then apply double-consonant reduction: runn -> run, swimm -> swim
- Remove trailing "ed" if stem length >= 4 (jumped -> jump, played -> play)
- Remove trailing "ly" if stem length >= 4 (quickly -> quick)
- Remove trailing "tion"/"sion" and replace with "t"/"s" (education -> educat, decision -> decis)
- Remove trailing "ment" if stem length >= 5 (development -> develop)
- Remove trailing "ness" if stem length >= 5 (happiness -> happi)
- Existing: remove trailing "s" if length > 3
```

Apply it to both corpus tokenization and query tokenization. Because it runs inside `tokenize()`,
everything downstream, meaning BM25, coverage and Jaccard, benefits automatically.

**Complexity:** about 30 lines in a new `stem()` function called from `tokenize()`.

**Risk:** over-stemming can conflate unrelated words, such as "universe" and "universal", though those
are close enough for BM25. The simplified version produces far fewer false merges than full Porter, and
because the tool ranks rather than classifies, a few false merges only add noise that IDF washes out.

**Why it was rated medium:** it needs verification that the stemming does not break the existing
coverage and floor calibration. Run the existing test queries and confirm the floors still separate hits
from misses.

### 3.4 IDF floor for very common terms

**Problem:** the current IDF formula `log(1 + (N - d + 0.5) / (d + 0.5))` can return 0 or negative
values when a term appears in at least half the documents. In corpora of 10 to 50 chunks, common words
like "cat" in a cat KB can appear in more than half the chunks and get a negative IDF, which means the
term lowers the score instead of raising it.

**Solution:** clamp IDF to a minimum of 0, or to a small epsilon such as 0.01. This is standard
practice, from Manning et al., "Introduction to Information Retrieval", 2008; it is one line:

```cpp
if (idf < 0) idf = 0;   // or: idf = max(idf, 0.0);
```

**Complexity:** 1 line.

**Risk:** none. A negative IDF is widely recognized as a bug in BM25 implementations. A term present in
more than half the documents should contribute 0, meaning neutral, rather than actively penalize the
document.

**Why it was rated medium, which is generous:** this is a bug fix rather than an enhancement. It is
listed here because it needs understanding of the IDF formula's behavior at the boundary.

### 3.5 Smarter pool construction: MMR instead of a top-10 cutoff

**Problem:** the current pool is the top-10 scored chunks. If the query has 3 distinct aspects and the
top-10 all cover the first aspect, the output is one-sided. Maximal Marginal Relevance (Carbonell and
Goldstein, 1998) iteratively selects chunks that are both relevant and diverse.

**Solution:** replace the top-10 pool and sequential dedup with an MMR loop:
1. Start with the top-1 chunk, the one with the highest BM25 score.
2. For each subsequent slot, pick the chunk that maximizes
   `lambda * bm25_score - (1-lambda) * max_similarity_to_already_picked`.
3. Use Jaccard as the similarity function, which is already implemented.
4. Stop when there are maxlines chunks or no candidates remain above a minimum
   score.

**Complexity:** about 25 lines, replacing the existing pick loop at lines 563-609 with a simpler one.

**Risk:** low. MMR is widely deployed, in Elastic, Solr and HuggingFace. The lambda parameter, where 0.7
is mostly relevance and 0.3 is mostly diversity, can be fixed or exposed as a CLI option.

**Why it was rated medium:** the algorithm change is straightforward, but it changes the output for
every query. Existing floor calibrations still work because they are based on topscore, which is the
first pick regardless of MMR. The output diversity does change, however, which may affect injection
quality in ways that need re-testing.

### 3.6 Stopword list expansion via IDF-based auto-detection

**Problem:** the static stopword list `STOP[]` was hand-written, so it can miss domain-specific stop
words. In a Bible corpus "shall" and "unto" are stop words, and in a physics corpus "energy" could be
one if it appears in every chunk.

**Solution:** after computing document frequencies, treat any term appearing in more than 40% of chunks
as a stop word by setting its IDF to 0. This is content-agnostic their statistical distribution rather
than by enumerating strings.

**Complexity:** 3 lines after the document-frequency loop:
```cpp
for (auto& p: df)
    if ((double)p.second / (double)n > 0.4)
        p.second = (int)n;   // force IDF to ~0
```

**Risk:** low. Terms appearing in more than 40% of chunks have a near-zero IDF anyway, so zeroing them
explicitly only removes noise. The 40% threshold is conservative, since standard BM25 starts producing
negative IDF at 50%.

**Why it was rated medium, though it is closer to trivial:** 3 lines, but it changes scoring behavior
and needs a quick sanity check against the existing test queries.

---

## 4. Not recommended: too complex or low payoff

### 4.1 BM25F, multi-field BM25

BM25F treats document fields such as title, body and anchor text with different weights. The chunks here
are flat text with no field structure, so this would require rearchitecting the corpus format.

### 4.2 Neural re-ranking with a cross-encoder

Adding a small cross-encoder such as MiniLM to re-rank the top-10 would improve precision. It would also
require an ONNX runtime or equivalent, model download and management, GPU or CPU inference, and 50 to
200 ms of latency per query. That conflicts with the medium-complexity constraint and the edge-device
focus.

### 4.3 Embedding-based dense retrieval

Switching to dense retrieval, for example FAISS with sentence-transformers, would improve semantic
matching. It would also require an embedding model, an index, vector-search infrastructure and
significant memory; that is not medium complexity.

### 4.4 BM25L, another BM25 variant

BM25L, also by Lv and Zhai, shifts the term-frequency normalization to boost long documents, which is
the opposite problem from BM25+. The chunks here are bounded, so both variants offer only marginal
improvement over vanilla BM25.

### 4.5 Query term position weighting

Weighting query terms by position, treating the first terms as more important, is a reasonable idea
for think-text queries where the topic sentence comes first. The queries here are typically short,
under 20 terms, and position weighting adds complexity without clear experimental evidence of benefit
in this setting; it could be revisited if experiments show it matters.

---

## 5. Implementation priority

| # | Improvement | Lines | Impact | Priority |
|---|-------------|-------|--------|----------|
| 3.4 | IDF floor, the negative IDF fix | 1 | Bug fix | 1, do immediately |
| 3.6 | Auto-stopword via IDF | 3 | Medium | 2 |
| 3.3 | Simplified Porter stemmer | 30 | High | 3 |
| 3.1 | Pseudo-relevance feedback | 30 | High | 4 |
| 3.2 | Proximity boost | 40 | Medium | 5 |
| 3.5 | MMR pool selection | 25 | Medium | 6 |

The first two, 3.4 and 3.6, are small bug fixes that would go in together. The stemmer at 3.3 is the
highest-impact single enhancement. PRF at 3.1 is the highest-impact algorithmic improvement but needs
careful testing. Proximity at 3.2 and MMR at 3.5 are quality refinements that can be added
incrementally.

## 6. References

- Lv, Y. and Zhai, C. (2011). "Lower-bounding term frequency normalization." CIKM '11, pages 7-16. DOI 10.1145/2063576.2063584. This is the original BM25+ paper.
- Robertson, S. and Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." Foundations and Trends in IR, 3(4):333-389. An overview of the BM25 family.
- Manning, C.D., Raghavan, P. and Schutze, H. (2008). "Introduction to Information Retrieval." Cambridge University Press, Chapter 11. BM25 theory and the IDF floor convention.
- Carbonell, J. and Goldstein, J. (1998). "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries." SIGIR '98. This reference describes the MMR algorithm.
- Metzler, D. and Croft, W.B. (2005). "A Markov random field model for term dependencies." SIGIR '05. The sequential dependence model used for proximity.
- Rocchio, J. (1971). "Relevance feedback in information retrieval." In The Smart System. This reference describes the pseudo-relevance-feedback algorithm.
