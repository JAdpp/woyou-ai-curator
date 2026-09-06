"""Generate resumable retrieval-run JSONL for the frozen evaluation set.

The default path is deliberately retrieval-only: it loads the production
``CollectionRepository`` in one requested mode and writes the served ranking
without calling DeepSeek.  ``--audit`` is an explicit, optional live pass that
records answerability and accepted evidence separately; it never creates or
changes qrels.

Examples::

    python scripts/run_retrieval_evaluation.py \
      --run-name bm25-v1 --rag-mode bm25

    python scripts/run_retrieval_evaluation.py \
      --run-name hybrid-qwen-v1 --rag-mode hybrid --resume

    python scripts/run_retrieval_evaluation.py \
      --run-name hybrid-smoke --rag-mode hybrid --partial --limit 5 \
      --output artifacts/qa/retrieval-runs/hybrid-smoke.jsonl

Every completed query is flushed and fsynced as one JSON line.  A resumed run
validates its benchmark, collection and mode identity before skipping existing
query IDs.  Per-query provider failures are data, not process crashes: they are
written to ``apiErrors`` so latency and fallback reliability remain scoreable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from threading import RLock
from typing import Any, Awaitable, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "api"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.collections import (  # noqa: E402
    CollectionDataError,
    CollectionRepository,
    LoadedCollection,
    SearchResult,
)
from app.config import Settings  # noqa: E402
from app.generator import AgenticRetrievalOutcome, ExhibitionGenerator  # noqa: E402
from app.models import AgendaInput  # noqa: E402
from app.retrieval_runtime import build_collection_repository  # noqa: E402

from evaluate_retrieval import Benchmark, EvaluationError, load_benchmark  # noqa: E402


SCHEMA_VERSION = 1
RETRIEVAL_PIPELINE_VERSION = "phase1-agentic-v5-network-retry-20260906"
RAG_MODES = ("bm25", "shadow", "hybrid")


class RunGenerationError(ValueError):
    """Raised when a run cannot preserve the frozen evaluation contract."""


class InMemoryTraceCapture:
    """Keep only per-stage timings from the latest synchronous search.

    ``CollectionRepository`` supplies substantially richer trace arguments,
    including the raw question. Evaluation needs the timings but must not
    duplicate visitor text in another trace file, so this sink intentionally
    discards every field except the numeric stage map.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._generation = 0
        self._context_generation: ContextVar[int | None] = ContextVar(
            "retrieval_evaluation_trace_generation", default=None
        )
        self._stage_latency_ms: dict[str, float] = {}
        self._warnings: tuple[str, ...] = ()
        self._candidate_results: list[SearchResult] = []
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._context_generation.set(self._generation)
            self._stage_latency_ms = {}
            self._warnings = ()
            self._candidate_results = []

    def write(
        self,
        *,
        stage_latency_ms: Mapping[str, float] | None = None,
        warnings: Sequence[str] = (),
        candidate_results: Sequence[SearchResult] = (),
        **_discarded: Any,
    ) -> None:
        with self._lock:
            # Generator workers inherit their caller's context. An abandoned
            # worker may finish after reset/pop; its diagnostics belong to the
            # old query/stage and must not overwrite the current one.
            if self._context_generation.get() != self._generation:
                return
            self._stage_latency_ms = {
                str(stage): round(float(value), 3)
                for stage, value in (stage_latency_ms or {}).items()
            }
            self._warnings = tuple(str(value) for value in warnings if str(value))
            self._candidate_results = list(candidate_results)

    def pop_trace(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {
                "stageLatencyMs": dict(self._stage_latency_ms),
                "warnings": list(self._warnings),
                "candidateResults": list(self._candidate_results),
            }
            self.reset()
            return result


AuditRunner = Callable[
    [AgendaInput, LoadedCollection, list[SearchResult], int, float],
    Awaitable[AgenticRetrievalOutcome],
]
InitialRunner = Callable[[AgendaInput, LoadedCollection, float], Awaitable[Any]]


@dataclass(frozen=True)
class RunIdentity:
    run_name: str
    rag_mode: str
    benchmark_id: str
    benchmark_version: str
    collection_id: str
    collection_version: str
    collection_objects_sha256: str
    planning_enabled: bool = False
    audit_enabled: bool = False

    def row_fields(self) -> dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "runName": self.run_name,
            "ragMode": self.rag_mode,
            "benchmarkId": self.benchmark_id,
            "benchmarkVersion": self.benchmark_version,
            "collectionId": self.collection_id,
            "collectionVersion": self.collection_version,
            "collectionObjectsSha256": self.collection_objects_sha256,
            "retrievalPipelineVersion": RETRIEVAL_PIPELINE_VERSION,
            "planningEnabled": self.planning_enabled,
            "auditEnabled": self.audit_enabled,
        }


def _text(mapping: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _safe_error(error: BaseException, *, stage: str) -> dict[str, Any]:
    code = getattr(error, "code", None)
    if not isinstance(code, str) or not code.strip():
        code = type(error).__name__
    message = str(error).strip() or type(error).__name__
    row: dict[str, Any] = {
        "stage": stage,
        "code": code[:120],
        "message": message[:1000],
    }
    details = getattr(error, "details", None)
    if isinstance(details, Mapping):
        # CollectionDataError details are intentionally public diagnostics.
        # Keep only JSON-safe scalar/list values and never serialize settings.
        cleaned: dict[str, Any] = {}
        for key, value in details.items():
            if isinstance(key, str) and isinstance(
                value, (str, int, float, bool, type(None), list, tuple)
            ):
                cleaned[key] = list(value) if isinstance(value, tuple) else value
        if cleaned:
            row["details"] = cleaned
    return row


def _serialize_result(result: SearchResult, *, rank: int) -> dict[str, Any]:
    evidence_ids = _unique(list(result.matched_evidence_ids))
    cultural_legs = _unique(list(result.obj.culture_pack_ids))
    row: dict[str, Any] = {
        "objectId": result.obj.id,
        "rank": rank,
        "score": round(float(result.score), 8),
        "evidenceIds": evidence_ids,
        "culturalLegs": cultural_legs,
        "retrievalSources": _unique(list(result.retrieval_sources)),
        "matchedAnchorTerms": _unique(list(result.matched_anchor_terms)),
    }
    if result.dense_score is not None:
        row["denseScore"] = round(float(result.dense_score), 8)
    if result.evidence_score is not None:
        row["evidenceScore"] = round(float(result.evidence_score), 8)
    if result.field_scores:
        row["fieldScores"] = {
            field: round(float(score), 8) for field, score in result.field_scores
        }
    return row


def _serialize_results(
    results: Sequence[SearchResult], *, top_k: int
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows = [
        _serialize_result(result, rank=rank)
        for rank, result in enumerate(results[:top_k], 1)
    ]
    evidence_ids = _unique(
        [evidence_id for row in rows for evidence_id in row["evidenceIds"]]
    )
    cultural_legs = _unique(
        [cultural_leg for row in rows for cultural_leg in row["culturalLegs"]]
    )
    return rows, evidence_ids, cultural_legs


def _retrieval_status(
    repository: CollectionRepository, collection: LoadedCollection
) -> dict[str, Any]:
    status = repository.retrieval_status(collection)
    requested_mode = repository.rag_mode
    effective_mode = str(status.mode)
    served_mode = "bm25" if requested_mode == "shadow" else effective_mode
    row = {
        "requestedMode": requested_mode,
        "effectiveMode": effective_mode,
        "servedMode": served_mode,
        "enabled": bool(status.enabled),
        "available": bool(status.available),
        "fallbackApplied": requested_mode in {"hybrid", "shadow"}
        and effective_mode == "bm25",
        "reason": str(status.reason),
        "model": str(status.model),
        "fingerprint": status.fingerprint,
    }
    if requested_mode == "shadow":
        row["shadowCandidateMode"] = effective_mode
    return row


def _manifest_identity(
    benchmark: Benchmark,
    *,
    run_name: str,
    rag_mode: str,
    collection: LoadedCollection,
) -> RunIdentity:
    provenance = benchmark.manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RunGenerationError("benchmark manifest provenance is required")
    benchmark_id = _text(benchmark.manifest, "benchmarkId", "benchmark_id")
    benchmark_version = _text(benchmark.manifest, "version")
    expected_collection = _text(provenance, "collectionId", "collection_id")
    expected_version = _text(provenance, "collectionVersion", "collection_version")
    expected_hash = _text(provenance, "objectsSha256", "objects_sha256")
    if not all(
        (benchmark_id, benchmark_version, expected_collection, expected_version, expected_hash)
    ):
        raise RunGenerationError(
            "manifest must freeze benchmark id/version and collection id/version/hash"
        )
    actual_hash = str(collection.objects_sha256 or "")
    mismatches = []
    if collection.id != expected_collection:
        mismatches.append(f"id {collection.id!r} != {expected_collection!r}")
    if collection.version != expected_version:
        mismatches.append(f"version {collection.version!r} != {expected_version!r}")
    if actual_hash != expected_hash:
        mismatches.append("objects SHA-256 differs from the frozen benchmark")
    if mismatches:
        raise RunGenerationError("collection mismatch: " + "; ".join(mismatches))
    return RunIdentity(
        run_name=run_name,
        rag_mode=rag_mode,
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        collection_id=collection.id,
        collection_version=collection.version,
        collection_objects_sha256=actual_hash,
    )


def _validate_resume_row(
    row: Mapping[str, Any],
    *,
    identity: RunIdentity,
    path: Path,
    line_number: int,
    known_query_ids: set[str],
) -> str:
    context = f"{path}:{line_number}"
    query_id = _text(row, "queryId", "query_id")
    if not query_id:
        raise RunGenerationError(f"{context}: queryId is required")
    if query_id not in known_query_ids:
        raise RunGenerationError(f"{context}: unknown queryId {query_id!r}")
    expected = identity.row_fields()
    for field, value in expected.items():
        if row.get(field) != value:
            raise RunGenerationError(
                f"{context}: {field} {row.get(field)!r} does not match {value!r}"
            )
    return query_id


def load_completed_query_ids(
    path: Path,
    *,
    identity: RunIdentity,
    known_query_ids: set[str],
) -> set[str]:
    """Validate an existing run and return completed IDs for ``--resume``."""

    if not path.exists():
        return set()
    completed: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RunGenerationError(
                f"{path}:{line_number}: invalid JSON; do not resume a torn line: {error.msg}"
            ) from error
        if not isinstance(row, Mapping):
            raise RunGenerationError(f"{path}:{line_number}: row must be an object")
        query_id = _validate_resume_row(
            row,
            identity=identity,
            path=path,
            line_number=line_number,
            known_query_ids=known_query_ids,
        )
        if query_id in completed:
            raise RunGenerationError(f"{path}:{line_number}: duplicate queryId {query_id!r}")
        completed.add(query_id)
    return completed


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append exactly one durable UTF-8 JSON line."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _question_subset(
    benchmark: Benchmark,
    *,
    query_ids: Sequence[str],
    categories: Sequence[str],
    judgment_modes: Sequence[str],
    limit: int | None,
    partial: bool,
) -> list[dict[str, Any]]:
    if (query_ids or categories or judgment_modes or limit is not None) and not partial:
        raise RunGenerationError(
            "--query-id/--category/--judgment-mode/--limit require --partial"
        )
    questions = list(benchmark.questions.values())
    if categories:
        requested_categories = set(categories)
        known_categories = {
            _text(row, "category") for row in questions if _text(row, "category")
        }
        unknown_categories = sorted(requested_categories - known_categories)
        if unknown_categories:
            raise RunGenerationError(
                "unknown category/categories: " + ", ".join(unknown_categories)
            )
        questions = [
            row for row in questions if _text(row, "category") in requested_categories
        ]
    if judgment_modes:
        requested_modes = set(judgment_modes)
        known_modes = {
            _text(row, "judgmentMode", "judgementMode")
            for row in benchmark.questions.values()
            if _text(row, "judgmentMode", "judgementMode")
        }
        unknown_modes = sorted(requested_modes - known_modes)
        if unknown_modes:
            raise RunGenerationError(
                "unknown judgment mode(s): " + ", ".join(unknown_modes)
            )
        questions = [
            row
            for row in questions
            if _text(row, "judgmentMode", "judgementMode") in requested_modes
        ]
    if query_ids:
        requested = set(query_ids)
        unknown = sorted(requested - set(benchmark.questions))
        if unknown:
            raise RunGenerationError("unknown query ID(s): " + ", ".join(unknown))
        questions = [row for row in questions if _text(row, "queryId") in requested]
    if limit is not None:
        if limit <= 0:
            raise RunGenerationError("--limit must be positive")
        questions = questions[:limit]
    return questions


async def execute_questions(
    *,
    questions: Sequence[Mapping[str, Any]],
    repository: CollectionRepository,
    collection: LoadedCollection,
    identity: RunIdentity,
    output_path: Path,
    completed_query_ids: set[str] | None = None,
    top_k: int = 50,
    timeout_seconds: float = 40.0,
    audit_runner: AuditRunner | None = None,
    initial_runner: InitialRunner | None = None,
    required_count: int = 5,
) -> dict[str, int]:
    """Execute selected questions sequentially and durably append run rows."""

    if top_k <= 0:
        raise RunGenerationError("top_k must be positive")
    if timeout_seconds <= 0:
        raise RunGenerationError("timeout_seconds must be positive")
    if required_count <= 0:
        raise RunGenerationError("required_count must be positive")
    completed = set(completed_query_ids or ())
    written = 0
    errors = 0
    skipped = 0
    for question_row in questions:
        query_id = _text(question_row, "queryId", "query_id")
        if query_id in completed:
            skipped += 1
            continue
        question = _text(question_row, "question", "query", "text")
        language_tag = _text(question_row, "language").casefold()
        agenda = AgendaInput(
            question=question,
            prior_knowledge="some",
            duration_minutes=5,
            collection_id=collection.id,
            language="en" if language_tag.startswith("en") else "zh",
        )
        started = perf_counter()
        deadline = started + timeout_seconds
        stage_latency: dict[str, float] = {}
        api_errors: list[dict[str, Any]] = []
        warnings: list[dict[str, str]] = []
        initial_results: list[SearchResult] = []
        initial_diagnostics: dict[str, Any] = {}
        trace_capture = getattr(repository, "trace_writer", None)
        reset_trace = getattr(trace_capture, "reset", None)
        if callable(reset_trace):
            reset_trace()
        retrieval_started = perf_counter()
        try:
            if initial_runner is None:
                initial_results = repository.search(agenda, collection, deadline=deadline)
            else:
                prepared = await initial_runner(agenda, collection, deadline)
                initial_results = list(prepared.results)
                initial_diagnostics = dict(prepared.diagnostics)
        except Exception as error:  # per-query failures belong in the run
            api_errors.append(_safe_error(error, stage="retrieval"))
        stage_latency["retrieval"] = round(
            (perf_counter() - retrieval_started) * 1000.0, 3
        )
        pop_trace = getattr(trace_capture, "pop_trace", None)
        if callable(pop_trace):
            captured_trace = pop_trace()
            for stage, value in captured_trace.get("stageLatencyMs", {}).items():
                stage_latency.setdefault(stage, value)
            warnings.extend(
                {"stage": "retrieval", "code": str(code)}
                for code in captured_trace.get("warnings", ())
            )
        else:
            captured_trace = {}

        result_rows, evidence_ids, cultural_legs = _serialize_results(
            initial_results, top_k=top_k
        )
        row: dict[str, Any] = {
            **identity.row_fields(),
            "queryId": query_id,
            "question": question,
            "category": _text(question_row, "category"),
            "judgmentMode": _text(question_row, "judgmentMode", "judgementMode"),
            "expectedAnswerability": _text(
                question_row, "expectedAnswerability", "expected_answerability"
            ),
            "requiredCulturalLegs": _string_list(
                question_row.get(
                    "requiredCulturalLegs", question_row.get("required_cultural_legs")
                )
            ),
            "results": result_rows,
            "evidenceIds": evidence_ids,
            "culturalLegs": cultural_legs,
            "apiErrors": api_errors,
        }
        if initial_runner is not None:
            row["initialRetrievalDiagnostics"] = initial_diagnostics
        shadow_candidates = captured_trace.get("candidateResults", ())
        if identity.rag_mode == "shadow" and shadow_candidates:
            candidate_rows, candidate_evidence_ids, candidate_cultural_legs = (
                _serialize_results(list(shadow_candidates), top_k=top_k)
            )
            row["candidateResults"] = candidate_rows
            row["candidateEvidenceIds"] = candidate_evidence_ids
            row["candidateCulturalLegs"] = candidate_cultural_legs

        try:
            row["retrievalStatus"] = _retrieval_status(repository, collection)
        except Exception as error:
            api_errors.append(_safe_error(error, stage="retrieval_status"))

        if audit_runner is not None and not api_errors:
            audit_started = perf_counter()
            try:
                outcome = await audit_runner(
                    agenda,
                    collection,
                    initial_results,
                    required_count,
                    deadline,
                )
                accepted, accepted_evidence, accepted_cultures = _serialize_results(
                    outcome.results, top_k=top_k
                )
                row.update(
                    {
                        "acceptedResults": accepted,
                        "acceptedEvidenceIds": accepted_evidence,
                        "acceptedCulturalLegs": accepted_cultures,
                        "auditApplied": bool(outcome.audit_applied),
                        "expandedQueries": list(outcome.expanded_queries),
                        "forcedObjectIds": list(outcome.forced_object_ids),
                        "answerability": outcome.answerability,
                        "interpretation": outcome.interpretation,
                        "coverageGap": outcome.coverage_gap,
                        "expansionReason": outcome.expansion_reason,
                    }
                )
                stages = getattr(outcome, "stage_results", {}) or {}
                if stages:
                    row["stageResults"] = {
                        stage: _serialize_results(values, top_k=top_k)[0]
                        for stage, values in stages.items()
                    }
                final_results = getattr(outcome, "final_results", None)
                if final_results is not None:
                    row["finalResults"] = _serialize_results(final_results, top_k=top_k)[0]
                if outcome.failure_code:
                    api_errors.append(
                        {
                            "stage": "audit",
                            "code": outcome.failure_code,
                            "message": outcome.coverage_gap
                            or "agentic retrieval audit failed",
                        }
                    )
                if outcome.warning_code:
                    warnings.append(
                        {
                            "stage": "audit",
                            "code": outcome.warning_code,
                            "message": outcome.warning_detail,
                        }
                    )
            except Exception as error:
                api_errors.append(_safe_error(error, stage="audit"))
            stage_latency["audit"] = round(
                (perf_counter() - audit_started) * 1000.0, 3
            )
        if warnings:
            row["warnings"] = warnings
        row["stageLatencyMs"] = {
            **stage_latency,
            "total": round((perf_counter() - started) * 1000.0, 3),
        }
        append_jsonl(output_path, row)
        completed.add(query_id)
        written += 1
        if api_errors:
            errors += 1
    return {"written": written, "skipped": skipped, "errors": errors}


def build_repository(
    settings: Settings,
    *,
    rag_mode: str,
    collection_id: str,
    trace_writer: Any | None = None,
) -> CollectionRepository:
    """Use the same provider/index construction as the serving application."""
    return build_collection_repository(
        settings,
        collection_id=collection_id,
        rag_mode=rag_mode,
        trace_writer=trace_writer,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "qa" / "retrieval_eval_v1",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--rag-mode", choices=RAG_MODES, required=True)
    parser.add_argument(
        "--collection",
        help="collection id (defaults to the benchmark's frozen provenance)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--query-id", action="append", default=[])
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="select a category; repeat for a union (requires --partial)",
    )
    parser.add_argument(
        "--judgment-mode",
        action="append",
        default=[],
        help=(
            "select a judgement stratum; repeat for a union and combine with "
            "category by intersection (requires --partial)"
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--partial",
        action="store_true",
        help=(
            "allow --query-id/--category/--judgment-mode/--limit and an "
            "incomplete run"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--audit",
        action="store_true",
        help="also call DeepSeek for bounded answerability/evidence audit",
    )
    parser.add_argument(
        "--query-plan", action="store_true",
        help="use the application's pre-retrieval query planner (also enabled by --audit)",
    )
    parser.add_argument("--required-count", type=int, default=5)
    return parser.parse_args(argv)


async def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_dir = args.benchmark_dir.resolve()
    try:
        benchmark = load_benchmark(
            benchmark_dir / "manifest.json",
            benchmark_dir / "questions.jsonl",
            benchmark_dir / "qrels.jsonl",
        )
    except EvaluationError as error:
        raise RunGenerationError(str(error)) from error
    provenance = benchmark.manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RunGenerationError("benchmark manifest provenance is required")
    collection_id = args.collection or _text(provenance, "collectionId", "collection_id")
    if not collection_id:
        raise RunGenerationError("collection id is required")
    # The CLI mode is authoritative.  Temporarily exposing it while parsing
    # Settings prevents a BM25 baseline from being blocked by validation for a
    # hosted embedding provider configured for the production server.
    original_rag_mode = os.environ.get("RAG_MODE")
    os.environ["RAG_MODE"] = args.rag_mode
    try:
        settings = Settings.from_env()
    finally:
        if original_rag_mode is None:
            os.environ.pop("RAG_MODE", None)
        else:
            os.environ["RAG_MODE"] = original_rag_mode
    settings = replace(
        settings,
        rag_mode=args.rag_mode,
        default_collection_id=collection_id,
        rag_llm_audit_enabled=(True if args.audit else settings.rag_llm_audit_enabled),
    )
    trace_capture = InMemoryTraceCapture()
    repository = build_repository(
        settings,
        rag_mode=args.rag_mode,
        collection_id=collection_id,
        trace_writer=trace_capture,
    )
    collection = repository.get(collection_id)
    identity = _manifest_identity(
        benchmark,
        run_name=args.run_name,
        rag_mode=args.rag_mode,
        collection=collection,
    )
    planning_enabled = bool(args.audit or getattr(args, "query_plan", False))
    identity = replace(identity, planning_enabled=planning_enabled, audit_enabled=bool(args.audit))
    questions = _question_subset(
        benchmark,
        query_ids=args.query_id,
        categories=args.category,
        judgment_modes=args.judgment_mode,
        limit=args.limit,
        partial=args.partial,
    )
    safe_run_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", args.run_name).strip("-.")
    if not safe_run_name:
        raise RunGenerationError("run name must contain a path-safe character")
    output_path = (
        args.output.resolve()
        if args.output
        else (
            PROJECT_ROOT
            / "artifacts"
            / "qa"
            / "retrieval-runs"
            / f"{identity.benchmark_id}-{safe_run_name}.jsonl"
        )
    )
    known_query_ids = set(benchmark.questions)
    if output_path.exists() and not (args.resume or args.overwrite):
        raise RunGenerationError(
            f"output already exists: {output_path}; use --resume or --overwrite"
        )
    if args.overwrite:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
    completed = (
        load_completed_query_ids(
            output_path,
            identity=identity,
            known_query_ids=known_query_ids,
        )
        if args.resume
        else set()
    )

    audit_runner: AuditRunner | None = None
    initial_runner: InitialRunner | None = None
    if planning_enabled:
        if not settings.deepseek_api_key:
            raise RunGenerationError("--audit/--query-plan requires DEEPSEEK_API_KEY")
        generator = ExhibitionGenerator(settings, repository)
        prepared_state: dict[str, Any] = {}

        async def live_initial(agenda: AgendaInput, loaded_collection: LoadedCollection, deadline: float) -> Any:
            initial = await generator.prepare_initial_retrieval(
                agenda, loaded_collection, deadline=deadline,
            )
            prepared_state["initial"] = initial
            return initial

        initial_runner = live_initial

    if args.audit:
        async def live_audit(
            agenda: AgendaInput,
            loaded_collection: LoadedCollection,
            initial: list[SearchResult],
            required_count: int,
            deadline: float,
        ) -> AgenticRetrievalOutcome:
            return await generator._agentic_retrieve(
                agenda,
                loaded_collection,
                initial,
                required_count=required_count,
                deadline=deadline,
                initial_query_plan=prepared_state["initial"].query_plan,
                planning_attempted=True,
            )

        audit_runner = live_audit

    timeout_seconds = (
        settings.rag_retrieval_timeout_seconds
        if args.timeout_seconds is None
        else args.timeout_seconds
    )
    counts = await execute_questions(
        questions=questions,
        repository=repository,
        collection=collection,
        identity=identity,
        output_path=output_path,
        completed_query_ids=completed,
        top_k=args.top_k,
        timeout_seconds=timeout_seconds,
        audit_runner=audit_runner,
        initial_runner=initial_runner,
        required_count=args.required_count,
    )
    if (
        not args.partial
        and counts["written"] + counts["skipped"] != len(benchmark.questions)
    ):
        raise RunGenerationError(
            "full run did not cover every benchmark query; resume before evaluation"
        )
    return {
        "status": "complete" if not args.partial else "partial",
        "runName": args.run_name,
        "ragMode": args.rag_mode,
        "audit": bool(args.audit),
        "output": str(output_path),
        "selected": len(questions),
        **counts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        summary = asyncio.run(run_from_args(parse_args(argv)))
    except (RunGenerationError, ValueError, CollectionDataError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
