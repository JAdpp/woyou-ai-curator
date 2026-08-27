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
    SourceObject,
    sha256_bytes,
    utc_now,
    verify_images,
)


DEFAULT_DEPARTMENT = "Indian and Southeast Asian Art"


def _snapshot_source_dir(raw_root: Path, requested: str) -> Path:
    if requested != "latest":
        source_dir = raw_root / "snapshots" / requested / "cma-supplement"
        if not source_dir.is_dir():
            raise SystemExit(f"CMA supplement snapshot not found: {source_dir}")
        return source_dir
    candidates = sorted(
        raw_root.glob("snapshots/*/cma-supplement"), reverse=True
    )
    for candidate in candidates:
        if (candidate / "raw_manifest.json").is_file() and (
            candidate / "selection" / "selected.json"
        ).is_file():
            return candidate
    raise SystemExit(f"no rebuildable CMA supplement snapshot under {raw_root}")


def _verified_snapshot_bytes(
    collection_dir: Path,
    entry: dict[str, object],
) -> bytes:
    relative_path = str(entry.get("path") or "")
    path = (collection_dir / relative_path).resolve()
    if not relative_path or not path.is_relative_to(collection_dir.resolve()):
        raise RuntimeError("CMA supplement snapshot path escaped collection")
    content = path.read_bytes()
    if sha256_bytes(content) != str(entry.get("sha256") or ""):
        raise RuntimeError(f"CMA supplement snapshot SHA mismatch: {relative_path}")
    return content


def _rebuild_from_snapshot(
    collection_dir: Path,
    raw_root: Path,
    requested: str,
    target: int,
) -> tuple[list[SourceObject], str]:
    """Re-map the frozen selection without silently fetching a newer sample."""

    source_dir = _snapshot_source_dir(raw_root, requested)
    manifest = json.loads(
        (source_dir / "raw_manifest.json").read_text(encoding="utf-8")
    )
    entries = [
        entry
        for entry in (manifest.get("entries") or [])
        if isinstance(entry, dict)
    ]
    selection_entry = next(
        (entry for entry in entries if entry.get("kind") == "selection"),
        None,
    )
    if selection_entry is None:
        raise RuntimeError("CMA supplement snapshot has no selection receipt")
    selection = json.loads(
        _verified_snapshot_bytes(collection_dir, selection_entry).decode("utf-8")
    )
    selected_ids = [
        str(object_id)
        for object_id in (selection.get("selectedObjectIds") or [])
        if str(object_id)
    ][:target]
    if len(selected_ids) < target:
        raise RuntimeError(
            f"CMA supplement snapshot selected {len(selected_ids)} objects; "
            f"cannot rebuild target {target}"
        )

    mapped_by_id: dict[str, SourceObject] = {}
    for entry in entries:
        if entry.get("kind") != "search" or str(entry.get("name") or "").endswith(
            "-probe"
        ):
            continue
        payload = json.loads(
            _verified_snapshot_bytes(collection_dir, entry).decode("utf-8")
        )
        page_path = str(entry.get("path") or "")
        for obj in cma_global._map_page(cma_global._records(payload), page_path):
            mapped_by_id[obj.id] = obj

    missing = [object_id for object_id in selected_ids if object_id not in mapped_by_id]
    if missing:
        raise RuntimeError(
            f"CMA supplement snapshot is missing {len(missing)} selected records"
        )
    snapshot_id = source_dir.parent.name
    rebuilt = [mapped_by_id[object_id] for object_id in selected_ids]
    log(f"[supplement] rebuilt {len(rebuilt)} records from snapshot {snapshot_id}")
    return rebuilt, snapshot_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    parser.add_argument("--department", default=DEFAULT_DEPARTMENT)
    parser.add_argument("--target", type=int, default=1200)
    parser.add_argument(
        "--from-snapshot",
        nargs="?",
        const="latest",
        metavar="SNAPSHOT_ID",
        help="Rebuild the frozen CMA supplement instead of fetching a new sample.",
    )
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
    if args.from_snapshot:
        supplement, snapshot_id = _rebuild_from_snapshot(
            collection_dir,
            raw_root,
            args.from_snapshot,
            args.target,
        )
    else:
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
        snapshot_id = snapshot.id

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
    snapshot_ids[f"cmaSupplement:{args.department}"] = snapshot_id
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
