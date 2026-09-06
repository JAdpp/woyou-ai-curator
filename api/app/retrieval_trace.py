"""Privacy-bounded, append-only retrieval traces for shadow evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence
from uuid import uuid4


TRACE_SCHEMA_VERSION = "retrieval-trace/v1"


class RetrievalTraceWriter:
    """Write candidate-level diagnostics without persisting visitor wording.

    The trace stores a SHA-256 of the normalized query, result IDs, scores and
    retrieval channels. CuratorialBrief/store records already own the visitor
    question; duplicating raw free text in an operational log would only widen
    the privacy surface.
    """

    def __init__(self, root: Path, *, max_results: int = 60) -> None:
        self.root = Path(root)
        self.max_results = max(1, max_results)
        self._lock = RLock()

    @staticmethod
    def _query_digest(question: str) -> str:
        normalized = " ".join(question.split()).casefold()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _result_payload(result: Any, rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "objectId": result.obj.id,
            "score": round(float(result.score), 8),
            "sources": list(result.retrieval_sources),
            "matchedEvidenceIds": list(result.matched_evidence_ids)[:5],
            "signals": {
                name: value for name, value in result.field_scores
            },
        }

    def write(
        self,
        *,
        question: str,
        collection: Any,
        serving_mode: str,
        served_results: Sequence[Any],
        candidate_results: Sequence[Any],
        elapsed_ms: float,
        dense_status: Any | None = None,
        warnings: Sequence[str] = (),
        stage_latency_ms: Mapping[str, float] | None = None,
    ) -> Path:
        now = datetime.now(timezone.utc)
        served_ids = [item.obj.id for item in served_results[: self.max_results]]
        candidate_ids = [
            item.obj.id for item in candidate_results[: self.max_results]
        ]
        overlap = len(set(served_ids[:20]) & set(candidate_ids[:20]))
        record = {
            "schemaVersion": TRACE_SCHEMA_VERSION,
            "traceId": str(uuid4()),
            "createdAt": now.isoformat(),
            "querySha256": self._query_digest(question),
            "queryCharacters": len(question),
            "collectionId": collection.id,
            "collectionVersion": collection.version,
            "objectsSha256": getattr(collection, "objects_sha256", None),
            "servingMode": serving_mode,
            "denseStatus": (
                {
                    "available": bool(dense_status.available),
                    "mode": dense_status.mode,
                    "model": dense_status.model,
                    "fingerprint": dense_status.fingerprint,
                    "reason": dense_status.reason,
                }
                if dense_status is not None
                else None
            ),
            "elapsedMs": round(max(0.0, elapsed_ms), 3),
            "stageLatencyMs": {
                str(stage): round(max(0.0, float(duration)), 3)
                for stage, duration in sorted((stage_latency_ms or {}).items())
            },
            "top20Overlap": overlap,
            "warnings": list(dict.fromkeys(str(value) for value in warnings if value)),
            "served": [
                self._result_payload(result, rank)
                for rank, result in enumerate(
                    served_results[: self.max_results], start=1
                )
            ],
            "candidate": [
                self._result_payload(result, rank)
                for rank, result in enumerate(
                    candidate_results[: self.max_results], start=1
                )
            ],
        }
        path = self.root / f"{now:%Y-%m-%d}.jsonl"
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        return path
