# Frozen pre-v3 retrieval baseline

- Run date: 2026-08-31
- Suite: `open_vocabulary_blind_20260831_v2.json` (`20260831-v2`, previously marked `frozen_unrun`)
- Collection: `global_open`
- Retrieval: `hybrid-rag-v2`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Evidence audit: DeepSeek enabled
- Command: `python scripts/qa_open_rag.py --suite data/qa/open_vocabulary_blind_20260831_v2.json`
- Exit status: failed acceptance gate (`1`)

## Aggregate result

- Questions: 15
- Observed answerability: 6 `partially_supported`, 9 `unsupported`, 0 `supported`
- Infrastructure failures: 0
- Retrieval warnings: 2 `RETRIEVAL_SEARCH_TIMEOUT`
- Exact expected-answerability matches: 6/15
- Cases satisfying every scripted assertion: 3/15

This is a frozen implementation-before baseline, not a statistically valid IR
benchmark. The suite has expected answerability and hard-negative checks but no
object-level qrels, so the result must not be reported as Recall or nDCG.

## Material mismatches

- Expected `supported`, observed `partially_supported`: ordinary forming,
  botanical surfaces, inscription functions, cross-cultural storage,
  handmade traces.
- Expected `supported`, observed `unsupported`: print narrative.
- Expected `partially_supported`, observed `unsupported`: entrance protection,
  documented influence, missing-culture records.
- Several failures also had too few accepted objects or cultural origins to
  open a five-object exhibition.

The unmodified suite must be rerun after the v3 implementation. Its labels must
not be changed in response to the new system's output.
