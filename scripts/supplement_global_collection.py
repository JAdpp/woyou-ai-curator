"""Append a rights- and image-validated CMA department to ``global_open``.

This is an incremental repair path for department-label changes.  Existing
objects keep their completed image checks; only the new supplement is fetched
and validated, then the whole collection is atomically re-emitted with current
taxonomy rules.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from import_global_collections import (  # noqa: E402
    DEFAULT_COLLECTION_ID,
    _dedupe_json_objects,
    _emit,
    _read_object_list,
    _route_raw_object,
    _route_source_objects,
    _source_selection_bytes,
    _validate_before_emit,
    log,
)
from sources import cma_global  # noqa: E402
from sources.base import (  # noqa: E402
    COLLECTIONS_ROOT,
    HttpClient,
    Snapshot,
    utc_now,
    verify_images,
)


DEFAULT_DEPARTMENT = "Indian and Southeast Asian Art"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    parser.add_argument("--department", default=DEFAULT_DEPARTMENT)
    parser.add_argument("--target", type=int, default=1200)
    args = parser.parse_args()

    collection_id = args.collection_id.strip()
    if not collection_id or any(part in collection_id for part in ("/", "\\", "..")):
        raise SystemExit("collection-id must be a simple directory slug")
    collection_dir = (COLLECTIONS_ROOT / collection_id).resolve()
    if collection_dir.parent != COLLECTIONS_ROOT.resolve():
        raise SystemExit("collection target escaped data/collections")
    objects_path = collection_dir / "objects.json"
    manifest_path = collection_dir / "manifest.json"
    if not objects_path.exists() or not manifest_path.exists():
        raise SystemExit(f"{collection_id} has not been built")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("servingReady") is not True:
        raise SystemExit("refusing to supplement a collection before its base image check passes")

    raw_root = collection_dir / "raw"
    snapshot = Snapshot(raw_root, "cma-supplement")
    supplement = cma_global.fetch_department(
        HttpClient(min_interval=0.25),
        snapshot,
        args.department,
        args.target,
        log,
    )
    snapshot.store(
        "selection",
        "selected",
        "urn:woyou:global-open:cma-supplement-selection",
        _source_selection_bytes(supplement),
        200,
    )
    snapshot.finalize()

    supplement = _route_source_objects(supplement)
    log(f"[images] validating {len(supplement)} supplement image URLs…")
    supplement = verify_images(
        supplement,
        HttpClient(min_interval=0.10),
        progress=log,
        workers=8,
    )
    log(f"[supplement] {len(supplement)} image-ready records remain")

    existing = [_route_raw_object(raw) for raw in _read_object_list(objects_path)]
    combined = _dedupe_json_objects(
        [*existing, *(obj.to_json() for obj in supplement)]
    )
    combined.sort(
        key=lambda raw: (
            str(raw.get("institutionId") or ""),
            str(raw.get("sourceId") or raw.get("id") or ""),
        )
    )
    _validate_before_emit(combined, allow_small=False, require_image_check=True)

    snapshot_ids = dict(manifest.get("snapshotIds") or {})
    snapshot_ids[f"cmaSupplement:{args.department}"] = snapshot.id
    version = f"{utc_now()[:10].replace('-', '')}-{len(combined)}"
    _emit(
        collection_dir,
        combined,
        version,
        snapshot_ids,
        "full_remote_check",
    )
    log(f"wrote supplemented {collection_id}: {len(combined)} objects ({version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
