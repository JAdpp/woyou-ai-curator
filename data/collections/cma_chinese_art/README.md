# CMA Chinese Art seed

This directory contains the frozen, source-traceable collection used by the
Inquiry Curator restricted design demo.

## Rebuild

From `inquiry-curator/`:

```powershell
python scripts/import_cma_chinese_art.py --offline
```

The default command also reuses `raw/latest.json` when a frozen snapshot is
present. To create a new official snapshot deliberately:

```powershell
python scripts/import_cma_chinese_art.py --refresh
```

To recheck the frozen institutional image links without refetching metadata:

```powershell
python scripts/import_cma_chinese_art.py --verify-frozen-images
```

No API key is required or read.

## Runtime files

- `objects.json`: 60 objects, 30 per theme, with at least four locatable
  evidence chunks per object.
- `collection.json`: runtime collection metadata and boundaries.
- `manifest.json`: file inventory, checksums, counts, and importer version.
- `question_cards.json`: six bounded starter questions.
- `regression_questions.json`: 30 answerability fixtures: 20 supported, five
  partially supported, and five unsupported.
- `rights_audit.md`: source, image, licensing, attribution, and review limits.
- `raw/latest.json`: pointer to the frozen exact-response snapshot.

Every object remains `pending_human_review`. `source_exact_match` means only
that the evidence text can be found in the preserved official response; it is
not a claim of expert review or publication readiness.
