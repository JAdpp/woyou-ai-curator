r"""Add a verified, globally stratified AIC slice to ``global_open``.

The live path archives official API responses first, then validates AIC IIIF
images serially at no more than one request per second.  Only after all rights,
evidence, image, required-case, and collection contracts pass is a staged set
of serving artifacts atomically replaced.  Existing raw snapshots are never
rewritten or removed.

Examples (PowerShell)::

    # Candidate/snapshot smoke test; does not change serving artifacts.
    python .\scripts\supplement_aic_global_collection.py --target 20 --snapshot-only

    # Serving supplement (the supported first-cut range is 1,000-1,500).
    python .\scripts\supplement_aic_global_collection.py --target 1200

    # Rebuild the selected slice from the newest frozen snapshot, then recheck
    # images before committing.
    python .\scripts\supplement_aic_global_collection.py --target 1200 --from-snapshot
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from import_global_collections import (  # noqa: E402
    DEFAULT_COLLECTION_ID,
    _dedupe_json_objects,
    _emit,
    _read_object_list,
    _route_raw_object,
    _route_source_objects,
    _validate_before_emit,
    log,
)
from sources import aic_global  # noqa: E402
from sources.base import (  # noqa: E402
    AIC_USER_AGENT,
    COLLECTIONS_ROOT,
    HttpClient,
    Snapshot,
    SourceObject,
    sha256_bytes,
    utc_now,
    verify_images,
)


DEFAULT_TARGET = 1_200
MIN_SERVING_TARGET = 1_000
MAX_SERVING_TARGET = 1_500
DEFAULT_VALIDATION_RESERVE = 80
AIC_MIN_INTERVAL_SECONDS = 1.05
GENERATED_ARTIFACTS: tuple[str, ...] = (
    "objects.json",
    "question_cards.json",
    "regression_questions.json",
    "rights_audit.md",
    # Commit the manifest last so its count/hash never advertises staged data
    # before the data file itself is in place.
    "manifest.json",
)


def _resolve_collection_dir(collection_id: str) -> Path:
    collection_id = collection_id.strip()
    if not collection_id or any(part in collection_id for part in ("/", "\\", "..")):
        raise SystemExit("collection-id must be a simple directory slug")
    collection_dir = (COLLECTIONS_ROOT / collection_id).resolve()
    if collection_dir.parent != COLLECTIONS_ROOT.resolve():
        raise SystemExit("collection target escaped data/collections")
    return collection_dir


def _new_snapshot(raw_root: Path) -> Snapshot:
    """Create an append-only source snapshot and reject timestamp collisions."""

    snapshot = Snapshot(raw_root, aic_global.SOURCE_ID)
    if any(snapshot.dir.iterdir()):
        raise RuntimeError(
            f"refusing to reuse non-empty AIC snapshot directory {snapshot.dir}"
        )
    return snapshot


def _latest_snapshot_source(raw_root: Path, requested: str | None) -> Path:
    if requested and requested != "latest":
        source_dir = raw_root / "snapshots" / requested / aic_global.SOURCE_ID
        if not source_dir.is_dir():
            raise SystemExit(f"AIC global snapshot not found: {source_dir}")
        return source_dir
    candidates = sorted(
        (raw_root / "snapshots").glob(f"*/{aic_global.SOURCE_ID}"), reverse=True
    )
    for candidate in candidates:
        if (candidate / "selection" / "selected.json").exists() and (
            candidate / "raw_manifest.json"
        ).exists():
            return candidate
    raise SystemExit(f"no rebuildable AIC global snapshot found under {raw_root}")


def _verified_snapshot_bytes(
    collection_dir: Path,
    relative_path: str,
    manifest_entries: dict[str, dict[str, Any]],
) -> bytes:
    entry = manifest_entries.get(relative_path)
    if entry is None:
        raise RuntimeError(f"snapshot manifest has no entry for {relative_path}")
    path = (collection_dir / relative_path).resolve()
    if not path.is_relative_to(collection_dir.resolve()):
        raise RuntimeError(f"snapshot path escaped collection: {relative_path}")
    content = path.read_bytes()
    expected = str(entry.get("sha256") or "")
    actual = sha256_bytes(content)
    if not expected or actual != expected:
        raise RuntimeError(f"snapshot response SHA mismatch: {relative_path}")
    return content


def _rebuild_from_snapshot(
    collection_dir: Path,
    raw_root: Path,
    target: int,
    requested: str | None,
) -> tuple[list[SourceObject], str]:
    """Re-map an exact selected cut after validating response and record SHAs."""

    source_dir = _latest_snapshot_source(raw_root, requested)
    raw_manifest = json.loads(
        (source_dir / "raw_manifest.json").read_text(encoding="utf-8")
    )
    entries = {
        str(entry.get("path")): entry
        for entry in (raw_manifest.get("entries") or [])
        if isinstance(entry, dict) and entry.get("path")
    }
    selection_path = (source_dir / "selection" / "selected.json").relative_to(
        collection_dir
    ).as_posix()
    selection = json.loads(
        _verified_snapshot_bytes(collection_dir, selection_path, entries).decode("utf-8")
    )
    selected = selection.get("selected")
    if not isinstance(selected, list) or len(selected) < target:
        raise RuntimeError(
            f"snapshot selected only {len(selected) if isinstance(selected, list) else 0}; "
            f"cannot rebuild target {target}"
        )

    response_cache: dict[str, Any] = {}
    rebuilt: list[SourceObject] = []
    for receipt in selected[:target]:
        if not isinstance(receipt, dict):
            raise RuntimeError("snapshot selection receipt is malformed")
        raw_path = str(receipt.get("rawRecordPath") or "")
        if raw_path not in response_cache:
            response_cache[raw_path] = json.loads(
                _verified_snapshot_bytes(collection_dir, raw_path, entries).decode("utf-8")
            )
        artwork_id = str(receipt.get("artworkId") or "")
        record = next(
            (
                item
                for item in aic_global._records(response_cache[raw_path])
                if str(item.get("id") or "") == artwork_id
            ),
            None,
        )
        if record is None:
            raise RuntimeError(
                f"selected AIC artwork {artwork_id} missing from {raw_path}"
            )
        record_sha = sha256_bytes(aic_global._record_bytes(record))
        if record_sha != str(receipt.get("sourceRecordSha256") or ""):
            raise RuntimeError(f"selected AIC artwork {artwork_id} record SHA mismatch")
        candidate = aic_global._map_record(record, raw_path)
        if candidate is None:
            raise RuntimeError(
                f"selected AIC artwork {artwork_id} no longer passes mapper gates"
            )
        rebuilt.append(candidate.obj)

    _assert_required_objects(rebuilt, context="frozen AIC selection")
    snapshot_id = source_dir.parent.name
    log(f"[aic] rebuilt {len(rebuilt)} records from snapshot {snapshot_id}")
    return rebuilt, snapshot_id


def _assert_required_objects(objects: list[SourceObject], *, context: str) -> None:
    present = {obj.source_id for obj in objects}
    missing = [
        artwork_id
        for artwork_id in aic_global.REQUIRED_ARTWORK_IDS
        if str(artwork_id) not in present
    ]
    if missing:
        raise RuntimeError(f"{context} is missing required AIC artwork(s): {missing}")


def _progress_reporter(path: Path):
    """Mirror progress messages to an inspectable task log."""

    path.parent.mkdir(parents=True, exist_ok=True)

    def report(message: str) -> None:
        log(message)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now()}\t{message}\n")

    return report


def _patch_staged_manifest(
    stage_collection: Path,
    *,
    aic_target: int,
    aic_snapshot_id: str,
) -> None:
    path = stage_collection / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    distribution = manifest.get("institutionDistribution") or {}
    institutions = [name for name, count in distribution.items() if count]
    manifest["institution"] = " · ".join(sorted(institutions))
    manifest["sourceUrls"] = {
        "cma": "https://openaccess-api.clevelandart.org/",
        "met": "https://collectionapi.metmuseum.org/public/collection/v1/",
        "aic": "https://api.artic.edu/api/v1/artworks/search",
    }
    sampling = dict(manifest.get("sampling") or {})
    sampling.update(
        {
            "strategy": (
                "CMA department water-fill + verified Met seed + "
                "AIC department/culture/material stratification"
            ),
            "aicSamplingVersion": aic_global.SAMPLING_VERSION,
            "aicTarget": aic_target,
            "aicRequiredArtworkIds": list(aic_global.REQUIRED_ARTWORK_IDS),
            "aicSnapshotId": aic_snapshot_id,
        }
    )
    manifest["sampling"] = sampling
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _validate_stage(stage_collection: Path) -> None:
    for name in GENERATED_ARTIFACTS:
        if not (stage_collection / name).is_file():
            raise RuntimeError(f"staged artifact missing: {name}")
    objects_bytes = (stage_collection / "objects.json").read_bytes()
    objects = json.loads(objects_bytes.decode("utf-8"))
    manifest = json.loads((stage_collection / "manifest.json").read_text("utf-8"))
    if not isinstance(objects, list):
        raise RuntimeError("staged objects.json is not a list")
    if manifest.get("objectCount") != len(objects):
        raise RuntimeError("staged manifest/object count mismatch")
    if manifest.get("objectsSha256") != sha256_bytes(objects_bytes):
        raise RuntimeError("staged manifest/object SHA mismatch")
    if manifest.get("servingReady") is not True:
        raise RuntimeError("staged collection is not serving-ready")
    object_ids = {str(raw.get("id") or "") for raw in objects if isinstance(raw, dict)}
    missing = {
        f"aic:{artwork_id}" for artwork_id in aic_global.REQUIRED_ARTWORK_IDS
    } - object_ids
    if missing:
        raise RuntimeError(f"staged collection is missing required cases: {sorted(missing)}")


def _atomic_commit_artifacts(stage_collection: Path, collection_dir: Path) -> None:
    """Replace generated files atomically per artifact, with full rollback."""

    backup_root = Path(
        tempfile.mkdtemp(
            prefix=f".{collection_dir.name}-aic-backup-",
            dir=str(collection_dir.parent),
        )
    )
    committed: list[str] = []
    try:
        for name in GENERATED_ARTIFACTS:
            destination = collection_dir / name
            if destination.exists():
                shutil.copy2(destination, backup_root / name)
        for name in GENERATED_ARTIFACTS:
            staged = stage_collection / name
            destination = collection_dir / name
            os.replace(staged, destination)
            committed.append(name)
    except Exception:
        for name in reversed(committed):
            backup = backup_root / name
            destination = collection_dir / name
            if backup.exists():
                os.replace(backup, destination)
            elif destination.exists():
                destination.unlink()
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def _stage_and_commit(
    collection_dir: Path,
    combined: list[dict[str, Any]],
    version: str,
    snapshot_ids: dict[str, str],
    *,
    aic_target: int,
    aic_snapshot_id: str,
) -> None:
    stage_root = Path(
        tempfile.mkdtemp(
            prefix=f".{collection_dir.name}-aic-stage-",
            dir=str(collection_dir.parent),
        )
    )
    stage_collection = stage_root / collection_dir.name
    try:
        _emit(
            stage_collection,
            combined,
            version,
            snapshot_ids,
            "full_remote_check",
        )
        _patch_staged_manifest(
            stage_collection,
            aic_target=aic_target,
            aic_snapshot_id=aic_snapshot_id,
        )
        _validate_stage(stage_collection)
        _atomic_commit_artifacts(stage_collection, collection_dir)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def _repair_validated_serving_slice(
    *,
    collection_dir: Path,
    manifest: dict[str, Any],
    raw_root: Path,
    target: int,
    candidate_target: int,
    requested_snapshot: str,
) -> int:
    """Re-map the current AIC serving IDs while reusing exact image checks.

    This is intentionally narrower than a fresh import: object membership and
    image-validation evidence stay frozen, while mapper/schema corrections are
    applied from the traceable raw responses.  No network request is made.
    """

    objects_path = collection_dir / "objects.json"
    current = _read_object_list(objects_path)
    current_aic = [raw for raw in current if raw.get("institutionId") == "aic"]
    if len(current_aic) != target:
        raise SystemExit(
            f"repair expected {target} serving AIC objects, found {len(current_aic)}"
        )
    validations: dict[str, dict[str, Any]] = {}
    for raw in current_aic:
        object_id = str(raw.get("id") or "")
        validation = raw.get("imageValidation")
        if not isinstance(validation, dict) or validation.get("ok") is not True:
            raise SystemExit(f"repair refuses non-validated serving object {object_id}")
        validations[object_id] = dict(validation)

    rebuilt, snapshot_id = _rebuild_from_snapshot(
        collection_dir,
        raw_root,
        candidate_target,
        requested_snapshot,
    )
    rebuilt_by_id = {obj.id: obj for obj in rebuilt}
    missing = sorted(set(validations) - set(rebuilt_by_id))
    if missing:
        raise SystemExit(
            f"repair snapshot is missing {len(missing)} current serving AIC ids"
        )

    repaired: list[SourceObject] = []
    for raw in current_aic:
        object_id = str(raw["id"])
        obj = rebuilt_by_id[object_id]
        obj.image_validation = validations[object_id]
        repaired.append(obj)
    repaired = _route_source_objects(repaired)
    _assert_required_objects(repaired, context="repaired AIC serving slice")

    non_aic = [
        _route_raw_object(raw)
        for raw in current
        if raw.get("institutionId") != "aic"
    ]
    combined = _dedupe_json_objects(
        [*(obj.to_json() for obj in repaired), *non_aic]
    )
    combined.sort(
        key=lambda raw: (
            str(raw.get("institutionId") or ""),
            str(raw.get("sourceId") or raw.get("id") or ""),
        )
    )
    _validate_before_emit(combined, allow_small=False, require_image_check=True)

    snapshot_ids = dict(manifest.get("snapshotIds") or {})
    snapshot_ids[f"aicGlobal:{snapshot_id}"] = snapshot_id
    version = f"{utc_now()[:10].replace('-', '')}-{len(combined)}-rights1"
    _stage_and_commit(
        collection_dir,
        combined,
        version,
        snapshot_ids,
        aic_target=target,
        aic_snapshot_id=snapshot_id,
    )
    log(
        f"atomically repaired AIC serving rights from {snapshot_id}: "
        f"{len(repaired)} AIC / {len(combined)} total ({version})"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument(
        "--validation-reserve",
        type=int,
        default=DEFAULT_VALIDATION_RESERVE,
        help=(
            "Extra stratified candidates to image-check before taking the exact serving "
            "target (default: 80; ignored by --snapshot-only)."
        ),
    )
    parser.add_argument(
        "--from-snapshot",
        nargs="?",
        const="latest",
        metavar="SNAPSHOT_ID",
        help="Rebuild from the newest or named frozen AIC snapshot.",
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="Archive/map candidates only; do not validate images or change serving files.",
    )
    parser.add_argument(
        "--repair-validated-serving",
        action="store_true",
        help=(
            "Re-map the current AIC serving IDs from --from-snapshot and reuse "
            "their successful imageValidation records; makes no network requests."
        ),
    )
    args = parser.parse_args()

    if args.target < len(aic_global.REQUIRED_ARTWORK_IDS) or args.target > MAX_SERVING_TARGET:
        raise SystemExit(
            f"target must be between {len(aic_global.REQUIRED_ARTWORK_IDS)} "
            f"and {MAX_SERVING_TARGET}"
        )
    if not args.snapshot_only and args.target < MIN_SERVING_TARGET:
        raise SystemExit(
            f"serving merge target must be {MIN_SERVING_TARGET}-{MAX_SERVING_TARGET}; "
            "use --snapshot-only for a smaller smoke test"
        )
    if args.validation_reserve < 0:
        raise SystemExit("validation-reserve cannot be negative")
    if args.repair_validated_serving and not args.from_snapshot:
        raise SystemExit("--repair-validated-serving requires --from-snapshot")
    if args.repair_validated_serving and args.snapshot_only:
        raise SystemExit("repair mode cannot be combined with --snapshot-only")

    candidate_target = args.target
    if not args.snapshot_only:
        candidate_target = min(
            MAX_SERVING_TARGET, args.target + args.validation_reserve
        )

    collection_dir = _resolve_collection_dir(args.collection_id)
    objects_path = collection_dir / "objects.json"
    manifest_path = collection_dir / "manifest.json"
    if not objects_path.exists() or not manifest_path.exists():
        raise SystemExit(f"{args.collection_id} has not been built")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("servingReady") is not True:
        raise SystemExit("refusing to supplement a collection that is not serving-ready")

    raw_root = collection_dir / "raw"
    if args.repair_validated_serving:
        return _repair_validated_serving_slice(
            collection_dir=collection_dir,
            manifest=manifest,
            raw_root=raw_root,
            target=args.target,
            candidate_target=candidate_target,
            requested_snapshot=args.from_snapshot,
        )

    if args.from_snapshot:
        aic_objects, snapshot_id = _rebuild_from_snapshot(
            collection_dir,
            raw_root,
            candidate_target,
            args.from_snapshot,
        )
    else:
        snapshot = _new_snapshot(raw_root)
        client = HttpClient(
            extra_headers={"AIC-User-Agent": AIC_USER_AGENT},
            min_interval=AIC_MIN_INTERVAL_SECONDS,
        )
        log(
            f"[aic] fetching globally stratified target {args.target} "
            f"with {candidate_target - args.target} validation reserves"
        )
        aic_objects = aic_global.fetch(client, snapshot, candidate_target, log)
        snapshot.finalize()
        snapshot_id = snapshot.id

    _assert_required_objects(aic_objects, context="candidate AIC selection")
    if args.snapshot_only:
        log(
            f"[aic] snapshot-only complete: {len(aic_objects)} candidates; "
            f"snapshot={snapshot_id}; serving artifacts unchanged"
        )
        return 0

    progress_path = (
        raw_root / "progress" / f"aic-global-{snapshot_id}-image-validation.log"
    )
    report = _progress_reporter(progress_path)
    aic_objects = _route_source_objects(aic_objects)
    report(f"[images] validating {len(aic_objects)} AIC images serially (>=1s apart)")
    aic_objects = verify_images(
        aic_objects,
        HttpClient(min_interval=AIC_MIN_INTERVAL_SECONDS),
        progress=report,
        workers=1,
    )
    _assert_required_objects(aic_objects, context="image-ready AIC selection")
    if len(aic_objects) < args.target:
        raise SystemExit(
            f"AIC image validation kept only {len(aic_objects)}/{args.target}; "
            "serving merge aborted so quotas are not silently weakened"
        )
    aic_objects = aic_objects[: args.target]

    existing = [_route_raw_object(raw) for raw in _read_object_list(objects_path)]
    # New AIC records precede old duplicates, so a rerun can refresh validation
    # and field-level rights while retaining any non-overlapping earlier slice.
    new_json = [obj.to_json() for obj in aic_objects]
    combined = _dedupe_json_objects([*new_json, *existing])
    combined.sort(
        key=lambda raw: (
            str(raw.get("institutionId") or ""),
            str(raw.get("sourceId") or raw.get("id") or ""),
        )
    )
    _validate_before_emit(combined, allow_small=False, require_image_check=True)

    snapshot_ids = dict(manifest.get("snapshotIds") or {})
    snapshot_ids[f"aicGlobal:{snapshot_id}"] = snapshot_id
    version = f"{utc_now()[:10].replace('-', '')}-{len(combined)}"
    _stage_and_commit(
        collection_dir,
        combined,
        version,
        snapshot_ids,
        aic_target=args.target,
        aic_snapshot_id=snapshot_id,
    )

    institutions = Counter(raw.get("institutionId") for raw in combined)
    report(
        f"wrote AIC-supplemented {args.collection_id}: {len(combined)} objects "
        f"({version}); institutions={dict(institutions)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
