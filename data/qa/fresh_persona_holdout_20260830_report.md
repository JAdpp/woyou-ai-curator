# Fresh-persona live holdout report — 2026-08-30

This is the first-run engineering result for
`fresh_persona_holdout_20260830.json`. The questions and answerability
hypotheses were frozen before semantic retrieval. No question, expected label,
threshold or alias was changed after seeing the results.

## Frozen runtime

- Collection: `global_open` version `20260826-17246`
- Objects: 17,246; evidence chunks: 61,620
- Objects SHA-256: `8a7cee291acc9e5beedd40cf12e30299049613cbb61b6954e1a4fb9a5758d55e`
- Dense fingerprint: `25ecec5ebd89d2c2eba9527cb5dabf54bfd2b4091947cee3f701614550228547`
- Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions
- Audit model: `deepseek-v4-flash`
- Retrieval route: hybrid BM25 + object/evidence embedding + RRF + LLM evidence audit
- Retrieval deadline: 30 seconds per question
- Suite SHA-256 at first run: `3c4fe2ddb24af6bd172e7b577b44543aa12907b910f73d1c04fb94fa3d77b3ee`

## First run

The first run did **not** pass. Six of fourteen questions ended in
`RETRIEVAL_SEARCH_TIMEOUT`; therefore their carried-forward answerability
labels are not counted as semantic successes. Of the eight questions that
completed without an infrastructure failure, only one exactly matched the
pre-run hypothesis. This is a deliberately hard holdout and not a statistical
generalisation benchmark.

| Case | Expected | Observed | Raw | Accepted | Elapsed | First-run interpretation |
|---|---:|---:|---:|---:|---:|---|
| celestial motifs | supported | partial | 250 | 0 | 24.87s | timeout; corpus also shows ranking noise |
| drinking forms | partial | partial | 250 | 0 | 25.20s | timeout; corpus contains strong direct examples |
| games and companions | partial | partial | 36 | 0 | 26.24s | timeout; relation evidence is sparse |
| wind made visible | partial | supported | 26 | 5 | 16.67s | valid support; the hypothesis was too conservative |
| shrinking lake maps | partial | unsupported | 229 | 0 | 24.47s | timeout plus near-absent corpus |
| repair authorship | partial | unsupported | 234 | 0 | 25.52s | timeout plus near-absent repair records |
| tool wear and labour | partial | unsupported | 99 | 5 | 28.68s | objects exist, but work division/day claims do not |
| historic machine sound | partial | unsupported | 250 | 0 | 17.97s | corpus gap; several `Puget Sound` false neighbours |
| Noh / Gẹ̀lẹ́dé / Yup'ik masks | partial | unsupported | 250 | 1 | 19.14s | Gẹ̀lẹ́dé and Yup'ik legs absent |
| unsigned attribution | partial | unsupported | 33 | 0 | 16.05s | corpus sufficient; negation/query planning failed |
| tired, no fixed topic | unsupported | partial | 250 | 4 | 21.49s | product-policy question: tentative preference vs evidence claim |
| voluntary-sale provenance | partial | unsupported | 60 | 0 | 17.48s | provenance evidence is excluded from audit payloads |
| prehistoric DNA claim | unsupported | unsupported | 250 | 0 | 16.84s | correct evidence-bounded refusal |
| robot firmware | unsupported | unsupported | 17 | 0 | 22.68s | first audit refused correctly, then pointless expansion timed out |

Raw counts are hybrid candidate counts, not relevance counts.

## Corpus-grounded diagnosis

### Existing evidence was missed

- **Celestial motifs:** at least four cultural legs and multiple direct records
  exist, including `cma:1999.9`, `cma:1986.94`, `cma:1920.2008`,
  `cma:1944.482` and `aic:835`. Broad candidates such as headrests outranked
  them.
- **Drinking forms:** `aic:155`, `aic:162`, `aic:164`, `aic:60782` and
  `cma:1969.85` directly connect vessel form to holding, pouring, cooling or
  feasting. Lexical noise such as titles containing `Place` displaced them.
- **Unsigned attribution:** `aic:5702`, `cma:1920.514`, `cma:1921.1003` and
  `cma:1920.424` document hidden signatures, stylistic comparison, workshop
  attribution and material/technique evidence. The query did not handle the
  negative condition “without a signature” well.

### Objects exist, but the requested relation is weak

- **Games:** the collection contains dice, gaming pieces and a small number of
  contextual records, but not enough evidence about rules and play partners.
- **Tool wear:** tools and occasional tool-mark descriptions exist, but not the
  requested chain from wear pattern to work division and a full workday.

### Corpus or cultural-leg gaps

- The lake-map sequence, textile repair authorship and historic machine-sound
  questions have almost no direct supporting records in this fine-art-heavy
  collection.
- The mask comparison has Noh-related records, but no indexed Gẹ̀lẹ́dé or
  Yup'ik evidence. Query expansion cannot manufacture those cultural legs.

### Routing and product-policy gaps

- **Provenance:** the collection has hundreds of acquisition/provenance chunks,
  including records that explicitly describe military removal. The current
  retrieval/audit path excludes all `institution_provenance` chunks, even when
  provenance is the user's subject. Voluntary consent and legal ownership still
  cannot be inferred from those records.
- **Affective preference:** “I am tired; do not rush me” is a legitimate
  personalisation input, but not a museum-record claim. It needs a preference
  route with tentative recommendations, not the same evidence-answerability
  contract as a historical question.

## Timeout diagnosis

The repeated timeout is primarily a lexical-search implementation and budget
allocation problem, not an embedding or DeepSeek bottleneck.

- A measured robot-firmware run spent 10.708s on initial search and 2.019s on
  the first audit, which had already returned `unsupported`.
- The system nevertheless generated three expansion searches. They received a
  12.072s budget and timed out after 13.179s while 5.136s remained reserved for
  a second audit that could no longer run.
- Running the three expansions without the deadline took 21.326s: 20.722s was
  three repeated full-catalogue `_lexical_search` passes, while batched query
  embedding took 0.244s and vector/evidence/RRF work about 0.275s.

Required fixes, in order:

1. Do not expand an audited out-of-domain/unsupported request; distinguish
   retrieval gaps from evidence gaps and out-of-scope requests.
2. Preserve the first audit's accepted objects and answerability when an
   optional expansion times out; report infrastructure failure separately.
3. Batch lexical expansion queries into one catalogue pass while preserving
   per-query IDF, anchors, exclusions and result equivalence.
4. Route provenance questions to provenance evidence without allowing source
   history to prove consent or legal title.
5. Add negation-aware query planning and a candidate-diversity/ranking check.

Increasing the global timeout alone is not a sufficient fix.

## End-to-end fresh-topic check

`wind made visible` was rerun through complete five-object generation:

- elapsed: 37.49s
- retrieval: 10 directly related objects; five selected; four with full
  institution-written descriptions
- labels: 5/5 used the vision model
- provider: DeepSeek
- validator: passed; exhibition status `READY`

The result still exposed two quality defects that the current validator misses:
the subtitle says “four paintings” although the room has five objects, and four
of the five objects are variations on bamboo in wind. Future validation should
check numeric claims against the generated structure and measure within-theme
object redundancy.
