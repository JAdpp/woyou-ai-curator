"""Validate the frozen retrieval_eval_v1 benchmark against global_open.

This validator is intentionally stricter than the generic evaluator.  It
replays every deterministic field rule, verifies frozen file hashes, rejects
legacy questions and confirms that pooled-silver candidates remain unscored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "qa" / "retrieval_eval_v1"
DEFAULT_COLLECTION_DIR = PROJECT_ROOT / "data" / "collections" / "global_open"
EXPECTED_CATEGORY_COUNTS = {
    "exact_title_author_institution": 35,
    "synonym_bilingual_typo": 30,
    "date_material_region": 35,
    "open_theme": 35,
    "cross_cultural": 35,
    "symbolic_causal_multihop": 30,
    "visual_motif": 25,
    "ambiguous_out_of_scope": 25,
}
ALLOWED_MODES = {
    "deterministic_field_gold",
    "pooled_silver_pending_human_review",
    "deterministic_evidence_boundary_no_relevant_objects",
}


class DatasetValidationError(ValueError):
    pass


def _normalise(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DatasetValidationError(f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise DatasetValidationError(
                f"{path}:{line_number}: invalid JSON: {error.msg}"
            ) from error
        if not isinstance(row, dict):
            raise DatasetValidationError(f"{path}:{line_number}: row must be an object")
        row["_line"] = line_number
        rows.append(row)
    return rows


def _legacy_questions(project_root: Path) -> set[str]:
    questions: set[str] = set()
    qa_dir = project_root / "data" / "qa"
    for path in qa_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("cases", []):
            if isinstance(row, dict) and row.get("question"):
                questions.add(_normalise(row["question"]).casefold())
    collection_dir = project_root / "data" / "collections" / "global_open"
    for name in ("regression_questions.json", "question_cards.json"):
        path = collection_dir / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("questions", payload.get("cards", payload)) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, str):
                questions.add(_normalise(row).casefold())
            elif isinstance(row, dict) and row.get("question"):
                questions.add(_normalise(row["question"]).casefold())
    return questions


def _material(obj: Mapping[str, Any]) -> str:
    return _normalise(obj.get("material") or obj.get("medium")).casefold()


def _matches_rule(question: Mapping[str, Any], obj: Mapping[str, Any]) -> bool:
    rule = question.get("deterministicRule")
    if not isinstance(rule, dict):
        raise DatasetValidationError(
            f"{question.get('queryId')}: deterministic question lacks deterministicRule"
        )
    operator = rule.get("operator")
    if operator == "casefold_exact":
        value = _normalise(rule["value"]).casefold()
        if "field" in rule:
            return _normalise(obj.get(str(rule["field"]))).casefold() == value
        return any(
            _normalise(obj.get(str(field))).casefold() == value
            for field in rule.get("fields", [])
        )
    if operator == "casefold_exact_pair":
        fields = list(rule["fields"])
        values = list(rule["values"])
        return all(
            _normalise(obj.get(field)).casefold() == _normalise(value).casefold()
            for field, value in zip(fields, values)
        )
    if operator == "single_adjacent_transposition_target":
        return _normalise(obj.get("title")).casefold() == _normalise(rule["canonicalValue"]).casefold()
    if operator == "alias_plus_exact_identifiers":
        return (
            _normalise(obj.get("type")).casefold()
            == _normalise(rule["canonicalType"]).casefold()
            and _normalise(obj.get("institution")).casefold()
            == _normalise(rule["institution"]).casefold()
            and _normalise(obj.get("accessionNumber")).casefold()
            == _normalise(rule["accessionNumber"]).casefold()
        )
    if operator == "pack_membership_and_casefold_contains":
        pack, material = rule["values"]
        return pack in obj.get("culturePackIds", []) and str(material).casefold() in _material(obj)
    if operator == "date_overlap_and_material_contains_and_pack_membership":
        values = rule["values"]
        return (
            isinstance(obj.get("dateEarliest"), int)
            and isinstance(obj.get("dateLatest"), int)
            and obj["dateEarliest"] <= int(values["end"])
            and obj["dateLatest"] >= int(values["start"])
            and str(values["material"]).casefold() in _material(obj)
            and values["culturePackId"] in obj.get("culturePackIds", [])
        )
    raise DatasetValidationError(
        f"{question.get('queryId')}: unsupported deterministic operator {operator!r}"
    )


def validate_dataset(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    collection_dir: Path = DEFAULT_COLLECTION_DIR,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise DatasetValidationError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        raise DatasetValidationError("manifest status must be frozen")
    if manifest.get("questionCount") != 250:
        raise DatasetValidationError("manifest questionCount must be exactly 250")
    if manifest.get("categoryDistribution") != EXPECTED_CATEGORY_COUNTS:
        raise DatasetValidationError("manifest categoryDistribution does not match v1 contract")
    if manifest.get("judgmentPolicy", {}).get("humanGoldPresent") is not False:
        raise DatasetValidationError("v1 must explicitly state that human gold is absent")

    questions_path = dataset_dir / "questions.jsonl"
    qrels_path = dataset_dir / "qrels.jsonl"
    questions = _read_jsonl(questions_path)
    qrels = _read_jsonl(qrels_path)

    file_contract = manifest.get("files", {})
    for name, path, rows in (
        ("questions", questions_path, questions),
        ("qrels", qrels_path, qrels),
    ):
        contract = file_contract.get(name, {})
        if contract.get("rows") != len(rows):
            raise DatasetValidationError(f"manifest {name} row count mismatch")
        if str(contract.get("sha256", "")).casefold() != _sha256(path):
            raise DatasetValidationError(f"manifest {name} SHA-256 mismatch")

    collection_manifest_path = collection_dir / "manifest.json"
    objects_path = collection_dir / "objects.json"
    collection_manifest = json.loads(collection_manifest_path.read_text(encoding="utf-8"))
    actual_objects_hash = _sha256(objects_path)
    expected_objects_hash = str(manifest["provenance"]["objectsSha256"]).casefold()
    if actual_objects_hash != expected_objects_hash:
        raise DatasetValidationError("dataset collection hash does not match objects.json")
    if str(collection_manifest.get("objectsSha256", "")).casefold() != actual_objects_hash:
        raise DatasetValidationError("collection manifest hash does not match objects.json")
    if manifest["provenance"].get("collectionVersion") != collection_manifest.get("version"):
        raise DatasetValidationError("collection version mismatch")

    object_payload = json.loads(objects_path.read_text(encoding="utf-8"))
    objects = object_payload.get("objects", object_payload) if isinstance(object_payload, dict) else object_payload
    objects_by_id = {str(obj["id"]): obj for obj in objects}
    evidence_by_object = {
        object_id: {
            str(chunk["id"])
            for chunk in obj.get("evidence", [])
            if isinstance(chunk, dict) and isinstance(chunk.get("id"), str)
        }
        for object_id, obj in objects_by_id.items()
    }

    if len(questions) != 250:
        raise DatasetValidationError(f"expected exactly 250 questions, got {len(questions)}")
    query_ids: set[str] = set()
    question_texts: set[str] = set()
    legacy = _legacy_questions(project_root)
    banned_patterns = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in manifest.get("excludedLegacyPatterns", [])
    ]
    questions_by_id: dict[str, dict[str, Any]] = {}
    for row in questions:
        line = row.pop("_line")
        query_id = str(row.get("queryId", ""))
        text = _normalise(row.get("question"))
        if not query_id or query_id in query_ids:
            raise DatasetValidationError(f"questions:{line}: missing or duplicate queryId")
        query_ids.add(query_id)
        normalised = text.casefold()
        if not text or normalised in question_texts:
            raise DatasetValidationError(f"questions:{line}: missing or duplicate question")
        if normalised in legacy:
            raise DatasetValidationError(f"questions:{line}: exact legacy question reused")
        if any(pattern.search(text) for pattern in banned_patterns):
            raise DatasetValidationError(f"questions:{line}: banned legacy theme reused: {text}")
        question_texts.add(normalised)
        mode = row.get("judgmentMode")
        if mode not in ALLOWED_MODES:
            raise DatasetValidationError(f"questions:{line}: invalid judgmentMode {mode!r}")
        if row.get("freezeStatus") != "frozen_before_system_comparison":
            raise DatasetValidationError(f"questions:{line}: question is not frozen")
        questions_by_id[query_id] = row

    counts = Counter(row["category"] for row in questions)
    if dict(counts) != EXPECTED_CATEGORY_COUNTS:
        raise DatasetValidationError(f"category distribution mismatch: {dict(counts)}")

    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pairs: set[tuple[str, str]] = set()
    for row in qrels:
        line = row.pop("_line")
        query_id = str(row.get("queryId", ""))
        object_id = str(row.get("objectId", ""))
        if query_id not in questions_by_id:
            raise DatasetValidationError(f"qrels:{line}: unknown queryId {query_id!r}")
        if object_id not in objects_by_id:
            raise DatasetValidationError(f"qrels:{line}: unknown objectId {object_id!r}")
        pair = (query_id, object_id)
        if pair in pairs:
            raise DatasetValidationError(f"qrels:{line}: duplicate query/object pair")
        pairs.add(pair)
        evidence_ids = row.get("supportingEvidenceIds")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise DatasetValidationError(f"qrels:{line}: supportingEvidenceIds must be non-empty")
        unknown_evidence = set(evidence_ids) - evidence_by_object[object_id]
        if unknown_evidence:
            raise DatasetValidationError(
                f"qrels:{line}: unknown evidence IDs for {object_id}: {sorted(unknown_evidence)}"
            )
        status = row.get("judgmentStatus")
        mode = questions_by_id[query_id]["judgmentMode"]
        if status == "deterministic_field_gold":
            if mode != status or not isinstance(row.get("relevance"), (int, float)) or row["relevance"] < 2:
                raise DatasetValidationError(f"qrels:{line}: invalid deterministic gold row")
        elif status == "pooled_silver_pending_human_review":
            if mode != status or row.get("relevance") != 0:
                raise DatasetValidationError(f"qrels:{line}: pooled silver must remain unscored")
            if "human" in str(row.get("provenance", "")).casefold() and "not human" not in str(row.get("provenance", "")).casefold():
                raise DatasetValidationError(f"qrels:{line}: pooled row claims human provenance")
        else:
            raise DatasetValidationError(f"qrels:{line}: invalid judgmentStatus {status!r}")
        qrels_by_query[query_id].append(row)

    # Exhaustive replay: deterministic qrel IDs must equal all matching frozen objects.
    for query_id, question in questions_by_id.items():
        mode = question["judgmentMode"]
        actual_ids = {row["objectId"] for row in qrels_by_query.get(query_id, [])}
        if mode == "deterministic_field_gold":
            expected_ids = {
                object_id
                for object_id, obj in objects_by_id.items()
                if _matches_rule(question, obj)
            }
            if not expected_ids:
                raise DatasetValidationError(f"{query_id}: deterministic rule has no matches")
            if actual_ids != expected_ids:
                missing = sorted(expected_ids - actual_ids)[:5]
                extra = sorted(actual_ids - expected_ids)[:5]
                raise DatasetValidationError(
                    f"{query_id}: deterministic qrels are not exhaustive; missing={missing}, extra={extra}"
                )
        elif mode == "pooled_silver_pending_human_review":
            if not 5 <= len(actual_ids) <= int(manifest["pooling"]["maximumCandidatesPerQuestion"]):
                raise DatasetValidationError(f"{query_id}: pooled candidate count outside 5..12")
            required = set(question.get("requiredCulturalLegs", []))
            observed = {
                leg
                for row in qrels_by_query[query_id]
                for leg in row.get("culturalLegs", [])
            }
            if not required.issubset(observed):
                raise DatasetValidationError(f"{query_id}: pooled candidates miss required cultural legs")
        elif actual_ids:
            raise DatasetValidationError(f"{query_id}: evidence-boundary question must have no qrels")

    mode_counts = Counter(row["judgmentMode"] for row in questions)
    qrel_counts = Counter(row["judgmentStatus"] for row in qrels)
    if manifest.get("judgmentDistribution") != dict(mode_counts):
        raise DatasetValidationError("manifest judgmentDistribution mismatch")
    if manifest.get("qrelDistribution") != dict(qrel_counts):
        raise DatasetValidationError("manifest qrelDistribution mismatch")

    return {
        "questionCount": len(questions),
        "qrelCount": len(qrels),
        "categoryDistribution": dict(counts),
        "judgmentDistribution": dict(mode_counts),
        "qrelDistribution": dict(qrel_counts),
        "collectionVersion": collection_manifest["version"],
        "objectsSha256": actual_objects_hash,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--collection-dir", type=Path, default=DEFAULT_COLLECTION_DIR)
    args = parser.parse_args(argv)
    try:
        summary = validate_dataset(args.dataset_dir, args.collection_dir)
    except DatasetValidationError as error:
        print(f"FAIL            {error}", file=sys.stderr)
        return 1
    print(f"questions       {summary['questionCount']}")
    print(f"qrels           {summary['qrelCount']}")
    print(f"categories      {summary['categoryDistribution']}")
    print(f"judgments       {summary['judgmentDistribution']}")
    print(f"collection      {summary['collectionVersion']} {summary['objectsSha256']}")
    print("ok              frozen retrieval evaluation dataset passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
