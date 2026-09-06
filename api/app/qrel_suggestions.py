"""Separate, append-only AI draft suggestions for frozen qrel review.

Suggestions are intentionally not human judgements.  They live in their own
SQLite database and are never read by progress/finalization gates or exported
by the human-gold review snapshot.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .qrel_review import FrozenRetrievalEvalV1


EVIDENCE_VERDICTS = {
    "supports",
    "insufficient",
    "contradicts",
    "uncertain",
    "not_applicable",
}
ANSWERABILITY_VALUES = {"supported", "partially_supported", "unsupported"}
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_SUGGESTION_ID_PATTERN = re.compile(r"^ai-suggestion-([1-9][0-9]*)$")


class AiSuggestionStore:
    def __init__(self, dataset_dir: Path, db_path: Path) -> None:
        self.dataset = FrozenRetrievalEvalV1(dataset_dir)
        self.db_path = db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ai_qrel_suggestion_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ai_qrel_suggestions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suggestion_run_id TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK(kind IN ('candidate', 'finalization')),
                    query_id TEXT NOT NULL,
                    object_id TEXT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    prompt_sha256 TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    benchmark_id TEXT NOT NULL,
                    benchmark_version TEXT NOT NULL,
                    questions_sha256 TEXT NOT NULL,
                    qrels_sha256 TEXT NOT NULL,
                    objects_sha256 TEXT NOT NULL,
                    input_modality TEXT NOT NULL,
                    input_sha256 TEXT NOT NULL,
                    confidence_band TEXT NOT NULL CHECK(confidence_band IN ('low', 'medium', 'high')),
                    risk_flags_json TEXT NOT NULL,
                    validation_status TEXT NOT NULL,
                    suggestion_json TEXT NOT NULL,
                    CHECK((kind = 'candidate' AND object_id IS NOT NULL) OR (kind = 'finalization' AND object_id IS NULL)),
                    UNIQUE(suggestion_run_id, kind, query_id, object_id)
                );
                CREATE INDEX IF NOT EXISTS ai_qrel_suggestions_latest
                    ON ai_qrel_suggestions(kind, query_id, object_id, id DESC);
                """
            )
            bindings = self._bindings()
            existing = dict(connection.execute("SELECT key, value FROM ai_qrel_suggestion_meta").fetchall())
            if existing and any(existing.get(key) != value for key, value in bindings.items()):
                raise RuntimeError("AI suggestion database is bound to a different frozen benchmark hash")
            for key, value in bindings.items():
                connection.execute("INSERT OR REPLACE INTO ai_qrel_suggestion_meta(key, value) VALUES (?, ?)", (key, value))

    def _bindings(self) -> dict[str, str]:
        manifest = self.dataset.manifest
        return {
            "benchmarkId": str(manifest["benchmarkId"]),
            "benchmarkVersion": str(manifest.get("version", "")),
            "questionsSha256": str(manifest["files"]["questions"]["sha256"]),
            "qrelsSha256": str(manifest["files"]["qrels"]["sha256"]),
            "objectsSha256": str(manifest["provenance"]["objectsSha256"]),
        }

    @staticmethod
    def _validate_note(value: Any) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("AI suggestion note must be a non-empty string")
        if len(value) > 4000:
            raise ValueError("AI suggestion note exceeds 4000 characters")

    def _validate_candidate_suggestion(
        self, query_id: str, object_id: str, suggestion: Any
    ) -> None:
        required = {
            "relevance",
            "evidenceVerdict",
            "supportingEvidenceIds",
            "note",
        }
        if not isinstance(suggestion, Mapping) or set(suggestion) != required:
            raise ValueError(
                "candidate suggestion must contain exactly relevance, evidenceVerdict, supportingEvidenceIds, and note"
            )
        relevance = suggestion["relevance"]
        verdict = suggestion["evidenceVerdict"]
        evidence_ids = suggestion["supportingEvidenceIds"]
        if isinstance(relevance, bool) or not isinstance(relevance, int) or relevance not in {0, 1, 2, 3}:
            raise ValueError("candidate suggestion relevance must be an integer from 0 to 3")
        if not isinstance(verdict, str) or verdict not in EVIDENCE_VERDICTS:
            raise ValueError("candidate suggestion evidenceVerdict is invalid")
        if (
            not isinstance(evidence_ids, list)
            or len(evidence_ids) > 100
            or any(not isinstance(item, str) or not item for item in evidence_ids)
            or any(len(item) > 512 for item in evidence_ids)
            or len(set(evidence_ids)) != len(evidence_ids)
        ):
            raise ValueError("candidate suggestion supportingEvidenceIds must be unique non-empty strings")
        self.dataset.validate_candidate(query_id, object_id, [])
        valid_evidence_ids = self.dataset.evidence_ids_by_object[object_id]
        if any(item not in valid_evidence_ids for item in evidence_ids):
            raise ValueError(
                "candidate suggestion supportingEvidenceIds must belong to the candidate object"
            )
        if verdict == "supports" and (relevance < 2 or not evidence_ids):
            raise ValueError("candidate suggestion supports requires relevance 2/3 and owned evidence")
        self._validate_note(suggestion["note"])

    @classmethod
    def _validate_finalization_suggestion(cls, suggestion: Any) -> None:
        required = {"expectedAnswerability", "note"}
        if not isinstance(suggestion, Mapping) or set(suggestion) != required:
            raise ValueError(
                "finalization suggestion must contain exactly expectedAnswerability and note"
            )
        if (
            not isinstance(suggestion["expectedAnswerability"], str)
            or suggestion["expectedAnswerability"] not in ANSWERABILITY_VALUES
        ):
            raise ValueError("finalization suggestion expectedAnswerability is invalid")
        cls._validate_note(suggestion["note"])

    @staticmethod
    def _candidate_supports(suggestion_json: str) -> bool:
        suggestion = json.loads(suggestion_json)
        return (
            suggestion.get("relevance") in {2, 3}
            and suggestion.get("evidenceVerdict") == "supports"
        )

    @staticmethod
    def _validate_provenance(record: Mapping[str, Any]) -> None:
        for field in (
            "suggestionRunId",
            "provider",
            "model",
            "promptVersion",
            "inputModality",
            "validationStatus",
        ):
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"AI suggestion {field} must be a non-empty string")
        for field in ("promptSha256", "inputSha256"):
            if not isinstance(record[field], str) or not _SHA256_PATTERN.fullmatch(record[field]):
                raise ValueError(f"AI suggestion {field} must be a SHA-256 digest")

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        kind = record.get("kind")
        query_id = record.get("queryId")
        object_id = record.get("objectId")
        if kind not in {"candidate", "finalization"} or not isinstance(query_id, str):
            raise ValueError("AI suggestion requires kind and queryId")
        self.dataset.question(query_id)
        if kind == "candidate":
            if not isinstance(object_id, str):
                raise ValueError("candidate suggestion requires objectId")
            self._validate_candidate_suggestion(query_id, object_id, record.get("suggestion"))
        elif object_id is not None:
            raise ValueError("finalization suggestion must not contain objectId")
        else:
            self._validate_finalization_suggestion(record.get("suggestion"))
        required = {
            "suggestionRunId", "provider", "model", "promptVersion", "promptSha256", "inputModality",
            "inputSha256", "confidenceBand", "riskFlags", "validationStatus", "suggestion",
        }
        if not required.issubset(record):
            raise ValueError("AI suggestion record is missing provenance fields")
        self._validate_provenance(record)
        if (
            not isinstance(record["confidenceBand"], str)
            or record["confidenceBand"] not in {"low", "medium", "high"}
        ):
            raise ValueError("AI suggestion confidenceBand is invalid")
        if (
            not isinstance(record["riskFlags"], list)
            or any(not isinstance(flag, str) or not flag for flag in record["riskFlags"])
            or len(set(record["riskFlags"])) != len(record["riskFlags"])
        ):
            raise ValueError("AI suggestion riskFlags must be a unique non-empty string list")
        bindings = self._bindings()
        payload = json.dumps(record["suggestion"], ensure_ascii=False, sort_keys=True)
        values = (
            record["suggestionRunId"], kind, query_id, object_id, record["provider"], record["model"], record["promptVersion"],
            record["promptSha256"], datetime.now(timezone.utc).isoformat(), bindings["benchmarkId"], bindings["benchmarkVersion"],
            bindings["questionsSha256"], bindings["qrelsSha256"], bindings["objectsSha256"], record["inputModality"],
            record["inputSha256"], record["confidenceBand"], json.dumps(record["riskFlags"], ensure_ascii=False, sort_keys=True),
            record["validationStatus"], payload,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if kind == "finalization" and record["suggestion"]["expectedAnswerability"] == "supported":
                candidate_rows = connection.execute(
                    "SELECT suggestion_json FROM ai_qrel_suggestions WHERE suggestion_run_id=? AND kind='candidate' AND query_id=?",
                    (record["suggestionRunId"], query_id),
                ).fetchall()
                if not any(self._candidate_supports(row["suggestion_json"]) for row in candidate_rows):
                    raise ValueError(
                        "finalization suggestion cannot be supported without a relevance 2/3 supporting candidate in the same run"
                    )
            prior = connection.execute(
                "SELECT * FROM ai_qrel_suggestions WHERE suggestion_run_id=? AND kind=? AND query_id=? AND object_id IS ?",
                (record["suggestionRunId"], kind, query_id, object_id),
            ).fetchone()
            if prior:
                expected_prior_values = (
                    record["provider"],
                    record["model"],
                    record["promptVersion"],
                    record["promptSha256"],
                    record["inputModality"],
                    record["inputSha256"],
                    record["confidenceBand"],
                    json.dumps(record["riskFlags"], ensure_ascii=False, sort_keys=True),
                    record["validationStatus"],
                    payload,
                )
                stored_prior_values = (
                    prior["provider"],
                    prior["model"],
                    prior["prompt_version"],
                    prior["prompt_sha256"],
                    prior["input_modality"],
                    prior["input_sha256"],
                    prior["confidence_band"],
                    prior["risk_flags_json"],
                    prior["validation_status"],
                    prior["suggestion_json"],
                )
                if stored_prior_values != expected_prior_values:
                    raise ValueError(
                        "suggestionRunId already contains different suggestion data or provenance for this resource"
                    )
                return self._row(prior)
            connection.execute(
                "INSERT INTO ai_qrel_suggestions(suggestion_run_id, kind, query_id, object_id, provider, model, prompt_version, prompt_sha256, generated_at, benchmark_id, benchmark_version, questions_sha256, qrels_sha256, objects_sha256, input_modality, input_sha256, confidence_band, risk_flags_json, validation_status, suggestion_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            row = connection.execute("SELECT * FROM ai_qrel_suggestions WHERE id=last_insert_rowid()").fetchone()
        return self._row(row)

    def get_by_id(self, suggestion_id: str) -> dict[str, Any] | None:
        if not isinstance(suggestion_id, str):
            return None
        match = _SUGGESTION_ID_PATTERN.fullmatch(suggestion_id)
        if match is None:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ai_qrel_suggestions WHERE id=?", (int(match.group(1)),)
            ).fetchone()
        if row is None:
            return None
        return {
            "kind": row["kind"],
            "queryId": row["query_id"],
            "objectId": row["object_id"],
            **self._row(row),
        }

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        suggestion = json.loads(row["suggestion_json"])
        return {
            "suggestionId": f"ai-suggestion-{row['id']}", "suggestionRunId": row["suggestion_run_id"],
            "provider": row["provider"], "model": row["model"], "promptVersion": row["prompt_version"],
            "promptHash": row["prompt_sha256"], "generatedAt": row["generated_at"],
            "modalities": [value for value in row["input_modality"].split(",") if value], "inputHash": row["input_sha256"],
            "confidenceBand": row["confidence_band"], "riskFlags": json.loads(row["risk_flags_json"]),
            "validationStatus": row["validation_status"], **suggestion,
        }

    def latest_for_question(self, query_id: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
        self.dataset.question(query_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM ai_qrel_suggestions WHERE query_id=? ORDER BY id DESC", (query_id,)
            ).fetchall()
        candidates: dict[str, dict[str, Any]] = {}
        finalization: dict[str, Any] | None = None
        for row in rows:
            if row["kind"] == "candidate" and row["object_id"] not in candidates:
                candidates[str(row["object_id"])] = self._row(row)
            elif row["kind"] == "finalization" and finalization is None:
                finalization = self._row(row)
        return candidates, finalization
