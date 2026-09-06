"""Validate an additive review export without relabelling the frozen benchmark."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.evaluate_retrieval import Benchmark


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_review_overlay(benchmark: Benchmark, path: Path) -> tuple[Benchmark, dict[str, Any]]:
    """Apply latest exported revisions to a copy for engineering-only scoring.

    The export may be the review API response or a delegated-completion report
    containing that response under ``snapshot``. It is never accepted as an
    independent human gold set. Source files are verified before any labels are
    used, and the full selected revision remains available in report provenance.
    """
    # Both direct CLI execution and package imports are supported.
    EvaluationError = importlib.import_module(type(benchmark).__module__).EvaluationError

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"Invalid review overlay: {path}") from error
    if not isinstance(payload, dict):
        raise EvaluationError("Review overlay must be an object")
    snapshot = payload.get("snapshot", payload)
    if not isinstance(snapshot, dict):
        raise EvaluationError("Review overlay snapshot must be an object")
    identity = snapshot.get("benchmark", {})
    expected_id = benchmark.manifest.get("benchmarkId", benchmark.manifest.get("id"))
    if not isinstance(identity, dict) or (
        identity.get("id") != expected_id
        or identity.get("version") != benchmark.manifest.get("version")
    ):
        raise EvaluationError("Review overlay benchmark identity does not match")

    hashes: dict[str, str] = {}
    for kind, source in (("questions", benchmark.questions_path), ("qrels", benchmark.qrels_path)):
        expected = benchmark.manifest.get("files", {}).get(kind, {}).get("sha256")
        actual = _sha256(source)
        if not expected or actual != expected:
            raise EvaluationError(f"Review overlay requires matching frozen {kind} SHA-256")
        hashes[kind] = actual
    provenance = benchmark.manifest.get("provenance", {})
    relative_objects = provenance.get("objectsFile")
    expected_objects_hash = provenance.get("objectsSha256")
    if not relative_objects or not expected_objects_hash:
        raise EvaluationError("Review overlay requires frozen object provenance")
    objects_path = next(
        (parent / relative_objects for parent in benchmark.manifest_path.parents
         if (parent / relative_objects).is_file()), None
    )
    if objects_path is None or _sha256(objects_path) != expected_objects_hash:
        raise EvaluationError("Review overlay frozen objects SHA-256 does not match")
    hashes["objects"] = expected_objects_hash
    objects_payload = json.loads(objects_path.read_text(encoding="utf-8"))
    objects = objects_payload.get("objects", []) if isinstance(objects_payload, dict) else objects_payload
    evidence_by_object = {
        obj["id"]: {item["id"] for item in obj.get("evidence", [])}
        for obj in objects
    }
    revisions = snapshot.get("revisions")
    if not isinstance(revisions, list):
        raise EvaluationError("Review overlay revisions must be a list")
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in revisions:
        if not isinstance(row, dict):
            raise EvaluationError("Review overlay revision must be an object")
        kind, query_id = row.get("kind"), row.get("queryId")
        object_id = row.get("objectId") or ""
        revision = row.get("revision")
        if kind not in {"candidate", "finalization"} or query_id not in benchmark.questions:
            raise EvaluationError("Review overlay has an unknown kind or query ID")
        if type(revision) is not int or revision < 1:
            raise EvaluationError("Review overlay revision must be a positive integer")
        if row.get("reviewOrigin") not in {"human", "delegated_ai"}:
            raise EvaluationError("Review overlay requires explicit human/delegated_ai origin")
        if not row.get("reviewerId"):
            raise EvaluationError("Review overlay reviewerId is required")
        if kind == "candidate":
            base = benchmark.qrels[query_id].get(object_id)
            if base is None or base.get("judgmentStatus") != "pooled_silver_pending_human_review":
                raise EvaluationError("Review overlay candidate is not in the frozen review pool")
            relevance = row.get("relevance")
            if type(relevance) is not int or relevance not in range(4):
                raise EvaluationError("Review overlay relevance must be an integer from 0 to 3")
            evidence = row.get("supportingEvidenceIds", [])
            if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
                raise EvaluationError("Review overlay evidence must be a list of IDs")
            if object_id not in evidence_by_object or not set(evidence).issubset(evidence_by_object[object_id]):
                raise EvaluationError("Review overlay evidence IDs must belong to the frozen object")
            if row.get("evidenceVerdict") not in {"supports", "insufficient", "not_applicable", "uncertain", "contradicts"}:
                raise EvaluationError("Review overlay evidence verdict is invalid")
        elif object_id or row.get("expectedAnswerability") not in {
            "supported", "partially_supported", "unsupported"
        }:
            raise EvaluationError("Review overlay finalization is invalid")
        key = (kind, query_id, object_id)
        previous = latest.get(key)
        if previous and previous["reviewerId"] != row["reviewerId"]:
            raise EvaluationError("Review overlay must resolve multiple reviewers before scoring")
        if previous and previous["revision"] == revision and previous != row:
            raise EvaluationError("Review overlay has conflicting equal revisions")
        if previous is None or revision > previous["revision"]:
            latest[key] = dict(row)

    qrels = {query_id: {object_id: dict(row) for object_id, row in rows.items()}
             for query_id, rows in benchmark.qrels.items()}
    questions = {query_id: dict(row) for query_id, row in benchmark.questions.items()}
    origins: Counter[str] = Counter()
    for (kind, query_id, object_id), row in latest.items():
        if kind != "candidate":
            continue
        origin = row["reviewOrigin"]
        origins[origin] += 1
        qrels[query_id][object_id].update({
            "relevance": float(row["relevance"]),
            "supportingEvidenceIds": list(row.get("supportingEvidenceIds", [])),
            "evidenceSupports": row["evidenceVerdict"] == "supports",
            "evidenceVerdict": row["evidenceVerdict"],
            "judgmentStatus": f"reviewed_{origin}",
            "reviewOrigin": origin,
            "reviewProvenance": row,
        })
    finalized: list[str] = []
    stale_finalizations: list[str] = []
    for query_id, question in questions.items():
        if question.get("judgmentMode") != "pooled_silver_pending_human_review":
            continue
        # The original pooled expectedAnswerability is a hypothesis, not a label.
        question.pop("expectedAnswerability", None)
        row = latest.get(("finalization", query_id, ""))
        if row is None:
            continue
        unreviewed = [object_id for object_id, base in benchmark.qrels[query_id].items()
                      if base.get("judgmentStatus") == "pooled_silver_pending_human_review"
                      and ("candidate", query_id, object_id) not in latest]
        if unreviewed:
            raise EvaluationError("Review overlay finalization requires all frozen candidates reviewed")
        finalized_at = row.get("createdAt")
        if finalized_at and any(
            candidate.get("createdAt", "") > finalized_at
            for (kind, candidate_query_id, _), candidate in latest.items()
            if kind == "candidate" and candidate_query_id == query_id
        ):
            stale_finalizations.append(query_id)
            continue
        question["expectedAnswerability"] = row["expectedAnswerability"]
        question["answerabilityScope"] = "reviewed_candidate_pool"
        question["reviewProvenance"] = row
        finalized.append(query_id)
    metadata = {
        "path": str(path), "sha256": _sha256(path), "sourceHashes": hashes,
        "purpose": "engineering_evaluation", "independentHumanGold": False,
        "candidateReviewOrigins": dict(origins),
        "finalizedQuestionCount": len(finalized),
        "finalizedQuestionIds": sorted(finalized),
        "staleFinalizationQuestionIds": sorted(stale_finalizations),
        "latestRevisions": list(latest.values()),
        "answerabilityScope": "reviewed_candidate_pool_not_full_collection",
        "sourceHashBasis": "current_frozen_manifest_verified_against_local_files",
        "modelProvenanceBasis": "exported_revision_and_acceptedSuggestionId_reference",
    }
    return replace(benchmark, qrels=qrels, questions=questions), metadata
