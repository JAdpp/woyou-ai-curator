"""Append-only review storage for the frozen retrieval-eval v1 corpus.

This module deliberately treats the benchmark as a read-only input.  Review
judgements retain their human or delegated-AI origin in a separate SQLite WAL
database so a later reviewer can see
every revision without changing either ``questions.jsonl`` or ``qrels.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from .qrel_suggestions import AiSuggestionStore


_AI_WORKFLOW_ONLY_FLAGS = frozenset(
    {
        "ai_draft",
        "needs_human_confirmation",
        "vision_reviewed",
        "text_reviewed",
        "metadata_reviewed",
        "evidence_reviewed",
    }
)
_MATERIAL_AI_RISK_FLAGS = frozenset(
    {
        "low_confidence",
        "visual_review_required",
        "visual_confirmation_recommended",
        "image_unavailable",
        "vision_cache_unavailable",
        "pending_evidence",
        "weak_evidence",
        "insufficient_evidence",
        "unsupported_evidence",
        "evidence_boundary",
        "evidence_boundary_risk",
        "visual_relation_unverified",
        "missing_cultural_leg",
        "cultural_leg_gap",
        "model_disagreement",
        "metadata_incomplete",
        "validation_repaired",
        "vision_support_without_owned_evidence",
        "lexical_only",
        "partial",
        "insufficient",
    }
)
_MATERIAL_AI_RISK_MARKERS = (
    "_failed", "_failure", "_error", "_unavailable", "_timeout",
    "_invalid", "_mismatch", "_unverified", "_insufficient",
    "_unsupported", "_incomplete", "_disagreement", "_repaired",
    "_fallback", "evidence_boundary",
)
_MATERIAL_VALIDATION_MARKERS = (
    "failed", "failure", "error", "invalid", "rejected", "unavailable",
    "timeout", "repaired", "fallback", "not_validated", "unvalidated",
)


def _normalize_ai_risk_value(value: object) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _is_material_ai_suggestion_risk(suggestion: dict[str, Any]) -> bool:
    """Exclude provenance/workflow markers from the high-risk review queue."""

    confident_negative = (
        suggestion.get("relevance") == 0
        and suggestion.get("evidenceVerdict") == "not_applicable"
    )
    normalized_flags = {
        _normalize_ai_risk_value(flag)
        for flag in suggestion.get("riskFlags") or []
        if flag
    }
    vision_reviewed = "vision_reviewed" in normalized_flags
    if suggestion.get("confidenceBand") == "low" and not confident_negative:
        return True
    for raw_flag in suggestion.get("riskFlags") or []:
        flag = _normalize_ai_risk_value(raw_flag)
        if not flag or flag in _AI_WORKFLOW_ONLY_FLAGS:
            continue
        if flag == "low_confidence" and confident_negative:
            continue
        if flag == "visual_review_required" and vision_reviewed:
            continue
        if flag in _MATERIAL_AI_RISK_FLAGS or any(
            marker in flag for marker in _MATERIAL_AI_RISK_MARKERS
        ):
            return True
    validation = _normalize_ai_risk_value(suggestion.get("validationStatus", ""))
    if not validation:
        return True
    return any(marker in validation for marker in _MATERIAL_VALIDATION_MARKERS)


class _ReviewModel(BaseModel):
    model_config = ConfigDict(alias_generator=lambda value: value.split("_")[0] + "".join(part.capitalize() for part in value.split("_")[1:]), populate_by_name=True, extra="forbid")


class CandidateReviewRequest(_ReviewModel):
    review_origin: Literal["human", "delegated_ai"] = "human"
    relevance: int = Field(ge=0, le=3)
    evidence_verdict: Literal[
        "supports", "insufficient", "contradicts", "uncertain", "not_applicable"
    ]
    supporting_evidence_ids: list[str] = Field(default_factory=list, max_length=100)
    note: str = Field(default="", max_length=4000)
    accepted_suggestion_id: str | None = Field(default=None, max_length=64)
    disposition: Literal["accepted", "modified", "rejected", "human_only"] = "human_only"
    expected_revision: int | None = Field(default=None, ge=0)
    request_id: str = Field(min_length=1, max_length=128)

    @field_validator("supporting_evidence_ids")
    @classmethod
    def unique_evidence_ids(cls, value: list[str]) -> list[str]:
        if any(not item or len(item) > 512 for item in value):
            raise ValueError("supportingEvidenceIds must contain non-empty IDs")
        if len(set(value)) != len(value):
            raise ValueError("supportingEvidenceIds must not contain duplicates")
        return value

    @field_validator("evidence_verdict")
    @classmethod
    def supports_requires_relevant_grade(cls, value: str, info: Any) -> str:
        relevance = info.data.get("relevance")
        if value == "supports" and relevance is not None and relevance < 2:
            raise ValueError("evidenceVerdict=supports requires relevance 2 or 3")
        return value


class FinalizationReviewRequest(_ReviewModel):
    review_origin: Literal["human", "delegated_ai"] = "human"
    expected_answerability: Literal[
        "supported", "partially_supported", "unsupported"
    ]
    note: str = Field(default="", max_length=4000)
    accepted_suggestion_id: str | None = Field(default=None, max_length=64)
    disposition: Literal["accepted", "modified", "rejected", "human_only"] = "human_only"
    expected_revision: int | None = Field(default=None, ge=0)
    request_id: str = Field(min_length=1, max_length=128)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid frozen retrieval evaluation file: {path}") from error


class FrozenRetrievalEvalV1:
    """Verified, memory-resident view of one immutable benchmark release."""

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir.resolve()
        manifest_path = self.dataset_dir / "manifest.json"
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Frozen retrieval evaluation manifest is unavailable") from error
        if (
            self.manifest.get("benchmarkId") != "retrieval_eval_v1"
            or self.manifest.get("status") != "frozen"
        ):
            raise RuntimeError("QREL review requires the frozen retrieval_eval_v1 dataset")

        files = self.manifest.get("files") or {}
        self.questions_path = self.dataset_dir / "questions.jsonl"
        self.qrels_path = self.dataset_dir / "qrels.jsonl"
        for label, path in (("questions", self.questions_path), ("qrels", self.qrels_path)):
            expected = (files.get(label) or {}).get("sha256")
            if not isinstance(expected, str) or _sha256(path) != expected:
                raise RuntimeError(f"Frozen retrieval evaluation {label} hash verification failed")

        provenance = self.manifest.get("provenance") or {}
        objects_file = provenance.get("objectsFile")
        objects_hash = provenance.get("objectsSha256")
        if not isinstance(objects_file, str) or not isinstance(objects_hash, str):
            raise RuntimeError("Frozen retrieval evaluation provenance is incomplete")
        # Benchmark paths are project-root relative.  The configured dataset
        # normally sits at <root>/data/qa/retrieval_eval_v1.
        project_root = self.dataset_dir.parents[2]
        self.objects_path = (project_root / objects_file).resolve()
        if _sha256(self.objects_path) != objects_hash:
            raise RuntimeError("Frozen collection object hash verification failed")

        questions = _json_lines(self.questions_path)
        qrels = _json_lines(self.qrels_path)
        if len(questions) != (files.get("questions") or {}).get("rows"):
            raise RuntimeError("Frozen retrieval evaluation question row count mismatch")
        if len(qrels) != (files.get("qrels") or {}).get("rows"):
            raise RuntimeError("Frozen retrieval evaluation qrel row count mismatch")
        self.questions = {str(row["queryId"]): row for row in questions}
        if len(self.questions) != len(questions):
            raise RuntimeError("Frozen retrieval evaluation has duplicate query IDs")
        self.qrels_by_question: dict[str, list[dict[str, Any]]] = {}
        for row in qrels:
            query_id = str(row.get("queryId", ""))
            if query_id not in self.questions:
                raise RuntimeError("Frozen qrel refers to an unknown question")
            self.qrels_by_question.setdefault(query_id, []).append(row)
        # Human review may only label the pre-frozen pooled-silver candidates.
        # Deterministic gold and unsupported-boundary questions stay outside
        # this workbench so retrospective review cannot contaminate v1.
        self.pending_pairs_by_question: dict[str, set[str]] = {}
        for query_id, rows in self.qrels_by_question.items():
            pending_ids = {
                str(row["objectId"])
                for row in rows
                if row.get("judgmentStatus") == "pooled_silver_pending_human_review"
            }
            if pending_ids:
                self.pending_pairs_by_question[query_id] = pending_ids

        try:
            parsed_objects = json.loads(self.objects_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Frozen collection objects are invalid") from error
        objects = parsed_objects.get("objects", parsed_objects) if isinstance(parsed_objects, dict) else parsed_objects
        if not isinstance(objects, list):
            raise RuntimeError("Frozen collection objects must be a list")
        self.objects: dict[str, dict[str, Any]] = {}
        self.evidence_ids_by_object: dict[str, set[str]] = {}
        for obj in objects:
            if not isinstance(obj, dict) or not isinstance(obj.get("id"), str):
                raise RuntimeError("Frozen collection has an object without an ID")
            object_id = obj["id"]
            self.objects[object_id] = obj
            evidence = obj.get("evidence") or []
            self.evidence_ids_by_object[object_id] = {
                str(item["id"])
                for item in evidence
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            }

    def question(self, query_id: str) -> dict[str, Any]:
        try:
            if query_id not in self.pending_pairs_by_question:
                raise KeyError(query_id)
            return self.questions[query_id]
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Frozen retrieval-eval question not found.") from error

    def validate_candidate(self, query_id: str, object_id: str, evidence_ids: list[str]) -> None:
        self.question(query_id)
        if object_id not in self.pending_pairs_by_question[query_id]:
            raise HTTPException(status_code=422, detail="The queryId/objectId pair is not a frozen pending-review candidate.")
        if object_id not in self.objects:
            raise HTTPException(status_code=422, detail="objectId is not owned by the verified frozen collection.")
        valid_evidence_ids = self.evidence_ids_by_object[object_id]
        invalid = [item for item in evidence_ids if item not in valid_evidence_ids]
        if invalid:
            raise HTTPException(
                status_code=422,
                detail="supportingEvidenceIds must belong to the reviewed object in the frozen collection.",
            )

    def object_summary(self, object_id: str) -> dict[str, Any]:
        obj = self.objects[object_id]
        def value(snake: str, default: Any = None) -> Any:
            camel = snake.split("_")[0] + "".join(part.capitalize() for part in snake.split("_")[1:])
            return obj.get(camel, obj.get(snake, default))
        return {
            "id": object_id,
            "sourceId": value("source_id"), "accessionNumber": value("accession_number", ""),
            "title": value("title", ""), "titleOriginal": value("title_original"),
            "date": value("date", ""), "dateEarliest": value("date_earliest"), "dateLatest": value("date_latest"),
            "maker": value("maker", ""), "medium": value("medium", ""), "type": value("type", "Collection object"),
            "culture": value("culture", ""), "creator": value("creator"), "material": value("material"), "place": value("place"),
            "cultureDisplay": value("culture_display"), "description": value("description"),
            "imageUrl": value("image_url", ""), "imageUrlLarge": value("image_url_large"), "objectUrl": value("object_url", ""),
            "rights": value("rights", ""), "rightsUri": value("rights_uri"), "imageLicense": value("image_license"),
            "imageRightsUri": value("image_rights_uri"), "metadataLicense": value("metadata_license"),
            "metadataRightsUri": value("metadata_rights_uri"), "curatorialTextLicense": value("curatorial_text_license"),
            "curatorialTextRightsUri": value("curatorial_text_rights_uri"), "altText": value("alt_text", ""),
            "altTextSource": value("alt_text_source", "metadata_fallback"), "themes": value("themes", []), "tags": value("tags", []),
            "institution": value("institution", ""), "institutionId": value("institution_id", ""), "department": value("department", ""),
            "classification": value("classification", ""), "creditLine": value("credit_line", ""),
            "culturePackIds": value("culture_pack_ids", []), "evidenceDomainIds": value("evidence_domain_ids", []),
            "relationFacets": value("relation_facets", []), "evidenceDepth": value("evidence_depth", "thin"),
            "evidence": [
                {"id": evidence.get("id"), "text": evidence.get("text", ""), "sourceUrl": evidence.get("sourceUrl", evidence.get("source_url", "")), "sourceTitle": evidence.get("sourceTitle", evidence.get("source_title", "")), "sourceLocation": evidence.get("sourceLocation", evidence.get("source_location", "Institution object record")), "supports": evidence.get("supports", ""), "reviewed": evidence.get("reviewed", False), "reviewStatus": "reviewed" if evidence.get("reviewed") is True else "pending", "verification": evidence.get("verification") or evidence.get("reviewStatus", evidence.get("review_status")), "license": evidence.get("license"), "rightsUri": evidence.get("rightsUri", evidence.get("rights_uri")), "sourceKind": evidence.get("sourceKind", evidence.get("source_kind"))}
                for evidence in obj.get("evidence", [])
                if isinstance(evidence, dict)
            ],
        }


class QrelReviewWorkbench:
    """Single-owner review stream backed by an append-only revision log."""

    REVIEWER_ID = "local-owner"

    def __init__(self, dataset_dir: Path, db_path: Path, suggestion_store: "AiSuggestionStore | None" = None) -> None:
        self.dataset = FrozenRetrievalEvalV1(dataset_dir)
        self.suggestion_store = suggestion_store
        self.db_path = db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS qrel_review_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL CHECK(kind IN ('candidate', 'finalization')),
                    query_id TEXT NOT NULL,
                    object_id TEXT,
                    reviewer_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    request_id TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    CHECK((kind = 'candidate' AND object_id IS NOT NULL) OR (kind = 'finalization' AND object_id IS NULL)),
                    UNIQUE(kind, query_id, object_id, reviewer_id, revision)
                );
                CREATE INDEX IF NOT EXISTS qrel_review_resource_revision
                ON qrel_review_revisions(kind, query_id, object_id, reviewer_id, revision DESC);
                CREATE TABLE IF NOT EXISTS qrel_review_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            bindings = {
                "benchmarkId": str(self.dataset.manifest["benchmarkId"]),
                "benchmarkVersion": str(self.dataset.manifest.get("version", "")),
                "questionsSha256": str(self.dataset.manifest["files"]["questions"]["sha256"]),
                "qrelsSha256": str(self.dataset.manifest["files"]["qrels"]["sha256"]),
                "objectsSha256": str(self.dataset.manifest["provenance"]["objectsSha256"]),
            }
            stored = dict(connection.execute("SELECT key, value FROM qrel_review_meta").fetchall())
            if stored and any(stored.get(key) != value for key, value in bindings.items()):
                raise RuntimeError("QREL review database is bound to a different frozen benchmark hash")
            for key, value in bindings.items():
                connection.execute("INSERT OR REPLACE INTO qrel_review_meta(key, value) VALUES (?, ?)", (key, value))

    @staticmethod
    def _revision_payload(row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["payload_json"])
        # Older revisions predate the explicit origin field and were entered
        # through the human review UI. Keep their export provenance explicit.
        payload.setdefault("reviewOrigin", "human")
        payload.update({"revision": row["revision"], "createdAt": row["created_at"]})
        return payload

    def _latest_candidate_rows(
        self, connection: sqlite3.Connection, query_id: str | None = None
    ) -> list[sqlite3.Row]:
        query_filter = " AND query_id=?" if query_id is not None else ""
        parameters = (self.REVIEWER_ID, query_id) if query_id is not None else (self.REVIEWER_ID,)
        rows = connection.execute(
            "SELECT * FROM qrel_review_revisions WHERE id IN ("
            "SELECT MAX(id) FROM qrel_review_revisions WHERE kind='candidate' AND reviewer_id=?"
            + query_filter + " GROUP BY query_id, object_id)",
            parameters,
        ).fetchall()
        return [
            row for row in rows
            if row["object_id"] in self.dataset.pending_pairs_by_question.get(row["query_id"], set())
        ]

    @staticmethod
    def _finalization_is_current(
        finalization: sqlite3.Row | None, candidate_rows: list[sqlite3.Row]
    ) -> bool:
        # IDs give a total append order even if timestamps have the same
        # precision. Any later candidate revision invalidates this snapshot.
        return finalization is not None and all(
            row["id"] < finalization["id"] for row in candidate_rows
        )

    def _validate_finalization_candidates(
        self, connection: sqlite3.Connection, query_id: str, answerability: str
    ) -> None:
        rows = self._latest_candidate_rows(connection, query_id)
        remaining = len(self.dataset.pending_pairs_by_question[query_id]) - len(rows)
        if remaining:
            raise HTTPException(status_code=409, detail={
                "message": "Reviewer must complete every frozen pending candidate before finalization.",
                "remainingCandidateCount": remaining,
            })
        if answerability != "supported":
            return
        supporting_ids = {
            row["object_id"] for row in rows
            if (payload := json.loads(row["payload_json"])).get("relevance", 0) >= 2
            and payload.get("evidenceVerdict") == "supports"
            and payload.get("supportingEvidenceIds")
        }
        if not supporting_ids:
            raise HTTPException(status_code=422, detail={
                "message": "Supported finalization requires a relevance 2/3 candidate with supporting evidence.",
                "supportingCandidateCount": 0,
            })
        covered_legs = {
            leg for qrel in self.dataset.qrels_by_question[query_id]
            if qrel.get("objectId") in supporting_ids
            for leg in qrel.get("culturalLegs", [])
        }
        missing_legs = [
            leg for leg in self.dataset.questions[query_id].get("requiredCulturalLegs", [])
            if leg not in covered_legs
        ]
        if missing_legs:
            raise HTTPException(status_code=422, detail={
                "message": "Supported finalization requires supporting evidence for every required cultural leg.",
                "missingCulturalLegs": missing_legs,
            })

    def _latest(self, connection: sqlite3.Connection, kind: str, query_id: str, object_id: str | None, reviewer_id: str) -> sqlite3.Row | None:
        if object_id is None:
            return connection.execute(
                "SELECT * FROM qrel_review_revisions WHERE kind=? AND query_id=? AND object_id IS NULL AND reviewer_id=? ORDER BY revision DESC LIMIT 1",
                (kind, query_id, reviewer_id),
            ).fetchone()
        return connection.execute(
            "SELECT * FROM qrel_review_revisions WHERE kind=? AND query_id=? AND object_id=? AND reviewer_id=? ORDER BY revision DESC LIMIT 1",
            (kind, query_id, object_id, reviewer_id),
        ).fetchone()

    def _append(self, *, kind: str, query_id: str, object_id: str | None, reviewer_id: str, request_id: str, expected_revision: int | None, payload: dict[str, Any]) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM qrel_review_revisions WHERE request_id=?", (request_id,)
            ).fetchone()
            if existing:
                if (existing["kind"], existing["query_id"], existing["object_id"], existing["reviewer_id"], existing["payload_json"]) != (kind, query_id, object_id, reviewer_id, json.dumps(payload, ensure_ascii=False, sort_keys=True)):
                    raise HTTPException(status_code=409, detail="requestId was already used for a different review resource.")
                return self._revision_payload(existing)
            if kind == "finalization":
                # Recheck under the same write transaction as the finalization
                # insert, so a concurrent candidate edit cannot evade the gate.
                self._validate_finalization_candidates(connection, query_id, payload["expectedAnswerability"])
            latest = self._latest(connection, kind, query_id, object_id, reviewer_id)
            actual_revision = latest["revision"] if latest else None
            if expected_revision != actual_revision:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={"message": "Review revision conflict.", "currentRevision": actual_revision},
                )
            revision = 1 if actual_revision is None else int(actual_revision) + 1
            created_at = datetime.now(timezone.utc).isoformat()
            connection.execute(
                "INSERT INTO qrel_review_revisions(kind, query_id, object_id, reviewer_id, revision, request_id, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (kind, query_id, object_id, reviewer_id, revision, request_id, json.dumps(payload, ensure_ascii=False, sort_keys=True), created_at),
            )
            payload = dict(payload)
            payload.update({"revision": revision, "createdAt": created_at})
            return payload

    @staticmethod
    def _candidate_decision_matches(
        request: CandidateReviewRequest, suggestion: dict[str, Any]
    ) -> bool:
        return (
            request.relevance == suggestion.get("relevance")
            and request.evidence_verdict == suggestion.get("evidenceVerdict")
            and set(request.supporting_evidence_ids)
            == set(suggestion.get("supportingEvidenceIds") or [])
        )

    def _validate_suggestion_disposition(
        self,
        *,
        kind: Literal["candidate", "finalization"],
        query_id: str,
        object_id: str | None,
        accepted_suggestion_id: str | None,
        disposition: Literal["accepted", "modified", "rejected", "human_only"],
        request: CandidateReviewRequest | FinalizationReviewRequest,
    ) -> None:
        if disposition == "human_only":
            if accepted_suggestion_id is not None:
                raise HTTPException(
                    status_code=422,
                    detail="disposition=human_only must not reference an AI suggestion.",
                )
            return
        if accepted_suggestion_id is None:
            raise HTTPException(
                status_code=422,
                detail=f"disposition={disposition} requires acceptedSuggestionId.",
            )
        if self.suggestion_store is None:
            raise HTTPException(
                status_code=422,
                detail="The referenced AI suggestion store is unavailable.",
            )
        suggestion = self.suggestion_store.get_by_id(accepted_suggestion_id)
        if (
            suggestion is None
            or suggestion.get("kind") != kind
            or suggestion.get("queryId") != query_id
            or suggestion.get("objectId") != object_id
        ):
            raise HTTPException(
                status_code=422,
                detail="acceptedSuggestionId must reference an AI suggestion for the same review resource.",
            )

        if kind == "candidate":
            if not isinstance(request, CandidateReviewRequest):
                raise RuntimeError("candidate suggestion audit received an invalid review request")
            decision_matches = self._candidate_decision_matches(request, suggestion)
            note_matches = request.note == suggestion.get("note")
        else:
            if not isinstance(request, FinalizationReviewRequest):
                raise RuntimeError("finalization suggestion audit received an invalid review request")
            decision_matches = (
                request.expected_answerability == suggestion.get("expectedAnswerability")
            )
            note_matches = request.note == suggestion.get("note")

        # The human note is independent audit rationale: it may differ while
        # the reviewer still accepts the AI decision itself. A "modified"
        # disposition, however, must change either that rationale or a decision
        # field; an explicit rejection must change the actual decision.
        if disposition == "accepted" and not decision_matches:
            raise HTTPException(
                status_code=422,
                detail="disposition=accepted requires the human decision fields to match the referenced AI suggestion.",
            )
        if disposition == "modified" and decision_matches and note_matches:
            raise HTTPException(
                status_code=422,
                detail="disposition=modified requires at least one decision or note field to differ from the referenced AI suggestion.",
            )
        if disposition == "rejected" and decision_matches:
            raise HTTPException(
                status_code=422,
                detail="disposition=rejected requires a human decision that differs from the referenced AI suggestion.",
            )

    def put_candidate(self, query_id: str, object_id: str, request: CandidateReviewRequest) -> dict[str, Any]:
        self.dataset.validate_candidate(query_id, object_id, request.supporting_evidence_ids)
        if request.evidence_verdict == "supports" and not request.supporting_evidence_ids:
            raise HTTPException(status_code=422, detail="Supporting relevance must cite evidence owned by the reviewed object.")
        self._validate_suggestion_disposition(
            kind="candidate",
            query_id=query_id,
            object_id=object_id,
            accepted_suggestion_id=request.accepted_suggestion_id,
            disposition=request.disposition,
            request=request,
        )
        payload = request.model_dump(by_alias=True, exclude={"expected_revision", "request_id"})
        payload["reviewerId"] = self.REVIEWER_ID
        payload.update({"queryId": query_id, "objectId": object_id})
        result = self._append(kind="candidate", query_id=query_id, object_id=object_id, reviewer_id=self.REVIEWER_ID, request_id=request.request_id, expected_revision=request.expected_revision, payload=payload)
        return {"queryId": query_id, "objectId": object_id, "judgment": self._judgment(result), "progress": self.progress()}

    def put_finalization(self, query_id: str, request: FinalizationReviewRequest) -> dict[str, Any]:
        self.dataset.question(query_id)
        self._validate_suggestion_disposition(
            kind="finalization",
            query_id=query_id,
            object_id=None,
            accepted_suggestion_id=request.accepted_suggestion_id,
            disposition=request.disposition,
            request=request,
        )
        payload = request.model_dump(by_alias=True, exclude={"expected_revision", "request_id"})
        payload["reviewerId"] = self.REVIEWER_ID
        payload.update({"queryId": query_id})
        result = self._append(kind="finalization", query_id=query_id, object_id=None, reviewer_id=self.REVIEWER_ID, request_id=request.request_id, expected_revision=request.expected_revision, payload=payload)
        current_progress = self._question_progress(query_id)
        is_current = current_progress["finalized"] and current_progress["finalizationRevision"] == result["revision"]
        return {
            "queryId": query_id,
            "status": "complete" if is_current else "in_progress",
            "revision": result["revision"],
            "finalization": {**self._finalization(result), "isCurrent": is_current},
            "progress": self.progress(),
        }

    @staticmethod
    def _judgment(revision: dict[str, Any]) -> dict[str, Any]:
        return {
            "reviewOrigin": revision.get("reviewOrigin", "human"),
            "relevance": revision["relevance"],
            "evidenceVerdict": revision["evidenceVerdict"], "supportingEvidenceIds": revision["supportingEvidenceIds"],
            "note": revision["note"],
            "acceptedSuggestionId": revision.get("acceptedSuggestionId"),
            "disposition": revision.get("disposition", "human_only"),
            "revision": revision["revision"], "updatedAt": revision["createdAt"],
        }

    @staticmethod
    def _finalization(revision: dict[str, Any]) -> dict[str, Any]:
        return {
            "reviewOrigin": revision.get("reviewOrigin", "human"),
            "expectedAnswerability": revision["expectedAnswerability"],
            "note": revision["note"],
            "acceptedSuggestionId": revision.get("acceptedSuggestionId"),
            "disposition": revision.get("disposition", "human_only"),
            "revision": revision["revision"],
            "updatedAt": revision["createdAt"],
        }

    def _question_progress(self, query_id: str) -> dict[str, Any]:
        candidate_ids = self.dataset.pending_pairs_by_question[query_id]
        with self._connect() as connection:
            rows = self._latest_candidate_rows(connection, query_id)
            finalization = self._latest(connection, "finalization", query_id, None, self.REVIEWER_ID)
        reviewed = len(rows)
        delegated = sum(json.loads(row["payload_json"]).get("reviewOrigin") == "delegated_ai" for row in rows)
        return {
            "candidateCount": len(candidate_ids), "reviewedCandidateCount": reviewed,
            "humanReviewedCandidateCount": reviewed - delegated,
            "delegatedReviewedCandidateCount": delegated,
            "remainingCandidateCount": len(candidate_ids) - reviewed,
            "finalized": reviewed == len(candidate_ids) and self._finalization_is_current(finalization, rows),
            "finalizationRevision": finalization["revision"] if finalization else None,
        }

    def progress(self) -> dict[str, Any]:
        total_questions = len(self.dataset.pending_pairs_by_question)
        total_candidates = sum(len(ids) for ids in self.dataset.pending_pairs_by_question.values())
        with self._connect() as connection:
            rows = self._latest_candidate_rows(connection)
            finalizations = connection.execute(
                "SELECT * FROM qrel_review_revisions WHERE id IN ("
                "SELECT MAX(id) FROM qrel_review_revisions WHERE kind='finalization' AND reviewer_id=? GROUP BY query_id)",
                (self.REVIEWER_ID,),
            ).fetchall()
        candidates_by_question: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            candidates_by_question.setdefault(row["query_id"], []).append(row)
        completed_questions = sum(
            len(candidates_by_question.get(row["query_id"], []))
            == len(self.dataset.pending_pairs_by_question.get(row["query_id"], set()))
            and self._finalization_is_current(row, candidates_by_question.get(row["query_id"], []))
            for row in finalizations
            if row["query_id"] in self.dataset.pending_pairs_by_question
        )
        delegated = sum(json.loads(row["payload_json"]).get("reviewOrigin") == "delegated_ai" for row in rows)
        return {
            "reviewedCandidates": len(rows), "humanReviewedCandidates": len(rows) - delegated,
            "delegatedReviewedCandidates": delegated, "totalCandidates": total_candidates,
            "completedQuestions": completed_questions, "totalQuestions": total_questions,
        }

    def _question_summary(self, query_id: str) -> dict[str, Any]:
        question = self.dataset.questions[query_id]
        question_progress = self._question_progress(query_id)
        ai_candidates = self.suggestion_store.latest_for_question(query_id)[0] if self.suggestion_store else {}
        if question_progress["finalized"]:
            review_status = "complete"
        elif question_progress["reviewedCandidateCount"]:
            review_status = "in_progress"
        else:
            review_status = "unreviewed"
        cultural_legs = list(question.get("requiredCulturalLegs") or [])
        return {
            "queryId": query_id, "question": question.get("question", ""), "category": question.get("category", ""),
            "judgmentMode": question.get("judgmentMode", ""), "expectedAnswerability": question.get("expectedAnswerability"),
            "requiredCulturalLegs": cultural_legs, "candidateCount": question_progress["candidateCount"],
            "reviewedCandidateCount": question_progress["reviewedCandidateCount"], "status": review_status,
            "humanReviewedCandidateCount": question_progress["humanReviewedCandidateCount"],
            "delegatedReviewedCandidateCount": question_progress["delegatedReviewedCandidateCount"],
            "suggestedCandidateCount": len(ai_candidates),
            "aiHighRiskCount": sum(
                1 for item in ai_candidates.values() if _is_material_ai_suggestion_risk(item)
            ),
        }

    def question_detail(self, query_id: str) -> dict[str, Any]:
        self.dataset.question(query_id)
        candidates: dict[str, dict[str, Any]] = {}
        for object_id in self.dataset.pending_pairs_by_question[query_id]:
            source_qrel = next(row for row in self.dataset.qrels_by_question[query_id] if row.get("objectId") == object_id)
            candidates[object_id] = {"objectId": object_id, "object": self.dataset.object_summary(object_id), "culturalLegs": source_qrel.get("culturalLegs", []), "evidence": self.dataset.object_summary(object_id)["evidence"], "judgment": None, "aiSuggestion": None}
        with self._connect() as connection:
            revisions = self._latest_candidate_rows(connection, query_id)
            for row in revisions:
                object_id = str(row["object_id"])
                if candidates[object_id]["judgment"] is None:
                    candidates[object_id]["judgment"] = self._judgment(self._revision_payload(row))
            finalization = self._latest(connection, "finalization", query_id, None, self.REVIEWER_ID)
        # A deterministic ordering means a frozen candidate pool is not served
        # in its builder/source order, while never storing a mutable shuffle.
        ordered_candidates = sorted(candidates.values(), key=lambda item: hashlib.sha256(f"{query_id}:{item['object']['id']}".encode("utf-8")).hexdigest())
        ai_candidates: dict[str, dict[str, Any]] = {}
        ai_question_suggestion: dict[str, Any] | None = None
        if self.suggestion_store is not None:
            ai_candidates, ai_question_suggestion = self.suggestion_store.latest_for_question(query_id)
            for object_id, suggestion in ai_candidates.items():
                if object_id in candidates:
                    candidates[object_id]["aiSuggestion"] = suggestion
        return {
            "benchmarkId": self.dataset.manifest["benchmarkId"],
            "question": self._question_summary(query_id),
            "candidates": ordered_candidates,
            "finalization": {
                **self._finalization(self._revision_payload(finalization)),
                "isCurrent": self._finalization_is_current(finalization, revisions),
            } if finalization else None,
            "aiQuestionSuggestion": ai_question_suggestion,
            "progress": self.progress(),
        }

    def questions_summary(self, status_filter: str | None = None, category: str | None = None, search: str | None = None) -> dict[str, Any]:
        questions = []
        for query_id in sorted(self.dataset.pending_pairs_by_question):
            summary = self._question_summary(query_id)
            if status_filter and status_filter != "all" and summary["status"] != status_filter:
                continue
            if category and summary["category"] != category:
                continue
            if search and search.casefold() not in (summary["queryId"] + " " + summary["question"]).casefold():
                continue
            questions.append(summary)
        return {"benchmarkId": self.dataset.manifest["benchmarkId"], "frozenAt": self.dataset.manifest.get("frozenAt"), "questions": questions, "categories": sorted({str(question.get("category", "")) for query_id, question in self.dataset.questions.items() if query_id in self.dataset.pending_pairs_by_question}), "progress": self.progress()}

    def export_snapshot(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM qrel_review_revisions ORDER BY id").fetchall()
        return {
            "benchmark": {"id": self.dataset.manifest["benchmarkId"], "version": self.dataset.manifest.get("version"), "status": "frozen"},
            "revisions": [
                {"kind": row["kind"], "queryId": row["query_id"], "objectId": row["object_id"], "requestId": row["request_id"], **self._revision_payload(row)}
                for row in rows
            ],
        }
