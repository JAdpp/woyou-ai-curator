"""Build a versioned SQLite metadata-filter derivative for one collection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "api"))

from app.collections import CollectionRepository  # noqa: E402
from app.structured_filters import build_structured_filter_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default="global_open",
        help="collection manifest id (default: global_open)",
    )
    parser.add_argument(
        "--collections-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "collections",
        help="directory containing collection snapshots",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PROJECT_ROOT / "api" / "runtime" / "cache" / "filters",
        help="root for fingerprint-addressed SQLite derivatives",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the exact fingerprint-addressed derivative",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = CollectionRepository(
        args.collections_dir.resolve(),
        default_collection_id=args.collection,
        rag_mode="bm25",
    )
    collection = repository.get(args.collection)
    started = time.perf_counter()
    output = build_structured_filter_index(
        collection,
        index_root=args.index_dir.resolve(),
        force=args.force,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "ready",
                "path": str(output),
                "collection": manifest["collectionId"],
                "collectionVersion": manifest["collectionVersion"],
                "objectsSha256": manifest["objectsSha256"],
                "fingerprint": manifest["fingerprint"],
                "objectCount": manifest["objectCount"],
                "evidenceCount": manifest["evidenceCount"],
                "objectSearchMode": manifest["objectSearchMode"],
                "evidenceSearchMode": manifest["evidenceSearchMode"],
                "databaseBytes": (output / "filters.sqlite3").stat().st_size,
                "databaseSha256": manifest["databaseSha256"],
                "seconds": round(time.perf_counter() - started, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
