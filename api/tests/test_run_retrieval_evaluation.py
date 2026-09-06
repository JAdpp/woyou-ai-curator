from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.collections import CollectionDataError
from app.generator import AgenticRetrievalOutcome
from scripts.evaluate_retrieval import load_benchmark
from scripts.run_retrieval_evaluation import (
    InMemoryTraceCapture,
    RunGenerationError,
    RunIdentity,
    _question_subset,
    _retrieval_status,
    execute_questions,
    load_completed_query_ids,
)


def _identity(*, mode: str = "bm25", name: str = "unit") -> RunIdentity:
    return RunIdentity(
        run_name=name,
        rag_mode=mode,
        benchmark_id="retrieval_eval_v1",
        benchmark_version="20260831-v1",
        collection_id="global_open",
        collection_version="20260826-17246",
        collection_objects_sha256="a" * 64,
    )


def _collection() -> SimpleNamespace:
    return SimpleNamespace(
        id="global_open",
        version="20260826-17246",
        objects_sha256="a" * 64,
    )


def _result(
    object_id: str,
    *,
    evidence_ids: tuple[str, ...] = ("evidence:1",),
    cultural_legs: tuple[str, ...] = ("east_asia",),
    retrieval_sources: tuple[str, ...] = ("bm25",),
) -> SimpleNamespace:
    return SimpleNamespace(
        obj=SimpleNamespace(
            id=object_id,
            culture_pack_ids=list(cultural_legs),
        ),
        score=3.25,
        matched_evidence_ids=evidence_ids,
        matched_anchor_terms=("ritual",),
        retrieval_sources=retrieval_sources,
        dense_score=0.71 if "dense" in retrieval_sources else None,
        evidence_score=0.66 if evidence_ids else None,
        field_scores=(("title", 1.5),),
    )


class FakeRepository:
    def __init__(self, results: list[SimpleNamespace], *, mode: str = "bm25") -> None:
        self.results = results
        self.rag_mode = mode
        self.search_calls: list[str] = []

    def search(self, agenda, collection, *, deadline=None):
        assert collection.id == "global_open"
        assert deadline is not None
        self.search_calls.append(agenda.question)
        return list(self.results)

    def retrieval_status(self, collection):
        return SimpleNamespace(
            enabled=self.rag_mode != "bm25",
            available=self.rag_mode != "bm25",
            mode="hybrid" if self.rag_mode != "bm25" else "bm25",
            reason="test status",
            model="test-embedding",
            fingerprint="fingerprint" if self.rag_mode != "bm25" else None,
        )


def _questions() -> list[dict]:
    return [
        {
            "queryId": "q1",
            "question": "祭祀器物如何传递身份？",
            "category": "open_theme",
            "judgmentMode": "pooled_silver_pending_human_review",
            "expectedAnswerability": "supported",
            "requiredCulturalLegs": ["east_asia"],
            "language": "zh-CN",
        },
        {
            "queryId": "q2",
            "question": "How did textiles record movement?",
            "category": "visual_motif",
            "judgmentMode": "pooled_silver_pending_human_review",
            "expectedAnswerability": "partially_supported",
            "language": "en",
        },
    ]


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_retrieval_only_run_writes_metric_and_trace_fields(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"
    repository = FakeRepository([_result("museum:1")])

    counts = asyncio.run(
        execute_questions(
            questions=_questions()[:1],
            repository=repository,
            collection=_collection(),
            identity=_identity(),
            output_path=output,
        )
    )

    assert counts == {"written": 1, "skipped": 0, "errors": 0}
    assert repository.search_calls == ["祭祀器物如何传递身份？"]
    row = _read_rows(output)[0]
    assert row["queryId"] == "q1"
    assert {
        key: row[key]
        for key in (
            "schemaVersion",
            "runName",
            "ragMode",
            "benchmarkId",
            "benchmarkVersion",
            "collectionId",
            "collectionVersion",
            "collectionObjectsSha256",
            "retrievalPipelineVersion",
            "planningEnabled",
            "auditEnabled",
        )
    } == _identity().row_fields()
    assert row["results"][0] == {
        "objectId": "museum:1",
        "rank": 1,
        "score": 3.25,
        "evidenceIds": ["evidence:1"],
        "culturalLegs": ["east_asia"],
        "retrievalSources": ["bm25"],
        "matchedAnchorTerms": ["ritual"],
        "evidenceScore": 0.66,
        "fieldScores": {"title": 1.5},
    }
    assert row["evidenceIds"] == ["evidence:1"]
    assert row["culturalLegs"] == ["east_asia"]
    assert row["apiErrors"] == []
    assert set(row["stageLatencyMs"]) == {"retrieval", "total"}
    assert row["retrievalStatus"]["fallbackApplied"] is False
    assert "answerability" not in row
    assert "acceptedResults" not in row


def test_run_merges_privacy_bounded_per_layer_trace_timings(tmp_path: Path) -> None:
    class TracedRepository(FakeRepository):
        def __init__(self) -> None:
            super().__init__([_result("museum:1")], mode="hybrid")
            self.trace_writer = InMemoryTraceCapture()

        def search(self, agenda, collection, *, deadline=None):
            results = super().search(agenda, collection, deadline=deadline)
            self.trace_writer.write(
                question="must be discarded",
                stage_latency_ms={
                    "object_fts_bm25": 7.1254,
                    "query_embedding": 431.25,
                },
                warnings=("rerank_failed",),
            )
            return results

    output = tmp_path / "traced.jsonl"
    repository = TracedRepository()
    asyncio.run(
        execute_questions(
            questions=_questions()[:1],
            repository=repository,
            collection=_collection(),
            identity=_identity(mode="hybrid"),
            output_path=output,
        )
    )

    row = _read_rows(output)[0]
    assert row["stageLatencyMs"]["object_fts_bm25"] == 7.125
    assert row["stageLatencyMs"]["query_embedding"] == 431.25
    assert row["warnings"] == [
        {"stage": "retrieval", "code": "rerank_failed"}
    ]
    assert "must be discarded" not in output.read_text(encoding="utf-8")


def test_shadow_run_records_candidate_without_replacing_served_results(
    tmp_path: Path,
) -> None:
    served = _result("museum:served", retrieval_sources=("bm25",))
    candidate = _result(
        "museum:candidate",
        retrieval_sources=("dense", "qwen3_rerank"),
    )

    class ShadowRepository(FakeRepository):
        def __init__(self) -> None:
            super().__init__([served], mode="shadow")
            self.trace_writer = InMemoryTraceCapture()

        def search(self, agenda, collection, *, deadline=None):
            results = super().search(agenda, collection, deadline=deadline)
            self.trace_writer.write(candidate_results=[candidate])
            return results

    output = tmp_path / "shadow.jsonl"
    asyncio.run(
        execute_questions(
            questions=_questions()[:1],
            repository=ShadowRepository(),
            collection=_collection(),
            identity=_identity(mode="shadow"),
            output_path=output,
        )
    )

    row = _read_rows(output)[0]
    assert row["results"][0]["objectId"] == "museum:served"
    assert row["candidateResults"][0]["objectId"] == "museum:candidate"


def test_optional_audit_is_separate_from_raw_retrieval(tmp_path: Path) -> None:
    output = tmp_path / "audited.jsonl"
    raw = _result("museum:raw")
    accepted = _result(
        "museum:accepted",
        evidence_ids=("evidence:accepted",),
        cultural_legs=("africa",),
        retrieval_sources=("dense", "llm_relevance_audit"),
    )
    audit_calls: list[str] = []

    async def audit(agenda, collection, initial, required_count, deadline):
        audit_calls.append(agenda.question)
        assert initial == [raw]
        assert required_count == 5
        assert deadline is not None
        return AgenticRetrievalOutcome(
            results=[accepted],
            audit_applied=True,
            expanded_queries=("cloth migration",),
            forced_object_ids=("museum:accepted",),
            answerability="supported",
            interpretation="material circulation",
            coverage_gap="",
            expansion_reason="insufficient_coverage",
        )

    counts = asyncio.run(
        execute_questions(
            questions=_questions()[1:],
            repository=FakeRepository([raw], mode="hybrid"),
            collection=_collection(),
            identity=_identity(mode="hybrid", name="audited"),
            output_path=output,
            audit_runner=audit,
        )
    )

    assert counts["errors"] == 0
    assert audit_calls == ["How did textiles record movement?"]
    row = _read_rows(output)[0]
    assert row["results"][0]["objectId"] == "museum:raw"
    assert row["acceptedResults"][0]["objectId"] == "museum:accepted"
    assert row["acceptedEvidenceIds"] == ["evidence:accepted"]
    assert row["acceptedCulturalLegs"] == ["africa"]
    assert row["answerability"] == "supported"
    assert row["auditApplied"] is True
    assert row["expandedQueries"] == ["cloth migration"]
    assert set(row["stageLatencyMs"]) == {"retrieval", "audit", "total"}


def test_planned_initial_and_observed_stages_do_not_fabricate_final_selection(tmp_path: Path) -> None:
    output = tmp_path / "planned.jsonl"
    initial_result = _result("museum:filtered")
    accepted = _result("museum:accepted")
    repository = FakeRepository([_result("museum:unfiltered")])

    async def initial(agenda, collection, deadline):
        return SimpleNamespace(results=[initial_result], diagnostics={
            "planningStatus": "applied", "filterSpec": {"dateStart": 1000},
        })

    async def audit(agenda, collection, results, required_count, deadline):
        assert results == [initial_result]
        return AgenticRetrievalOutcome(
            results=[accepted], audit_applied=True, answerability="supported",
            stage_results={"structured_retrieval": [initial_result], "pre_audit": [accepted]},
        )

    asyncio.run(execute_questions(
        questions=_questions()[:1], repository=repository, collection=_collection(),
        identity=_identity(), output_path=output, initial_runner=initial, audit_runner=audit,
    ))
    row = _read_rows(output)[0]
    assert repository.search_calls == []
    assert row["results"][0]["objectId"] == "museum:filtered"
    assert row["acceptedResults"][0]["objectId"] == "museum:accepted"
    assert row["stageResults"]["structured_retrieval"][0]["objectId"] == "museum:filtered"
    assert row["initialRetrievalDiagnostics"]["filterSpec"] == {"dateStart": 1000}
    assert "finalResults" not in row


def test_resume_validates_then_skips_completed_queries(tmp_path: Path) -> None:
    output = tmp_path / "resume.jsonl"
    identity = _identity()
    collection = _collection()
    first_repository = FakeRepository([_result("museum:1")])
    asyncio.run(
        execute_questions(
            questions=_questions()[:1],
            repository=first_repository,
            collection=collection,
            identity=identity,
            output_path=output,
        )
    )
    completed = load_completed_query_ids(
        output,
        identity=identity,
        known_query_ids={"q1", "q2"},
    )
    second_repository = FakeRepository([_result("museum:2")])

    counts = asyncio.run(
        execute_questions(
            questions=_questions(),
            repository=second_repository,
            collection=collection,
            identity=identity,
            output_path=output,
            completed_query_ids=completed,
        )
    )

    assert completed == {"q1"}
    assert counts == {"written": 1, "skipped": 1, "errors": 0}
    assert second_repository.search_calls == ["How did textiles record movement?"]
    assert [row["queryId"] for row in _read_rows(output)] == ["q1", "q2"]


def test_resume_rejects_incompatible_identity_and_duplicate_ids(tmp_path: Path) -> None:
    output = tmp_path / "bad.jsonl"
    row = {
        **_identity(mode="hybrid").row_fields(),
        "queryId": "q1",
        "results": [],
    }
    output.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(RunGenerationError, match="ragMode"):
        load_completed_query_ids(
            output,
            identity=_identity(mode="bm25"),
            known_query_ids={"q1"},
        )

    with pytest.raises(RunGenerationError, match="duplicate queryId"):
        load_completed_query_ids(
            output,
            identity=_identity(mode="hybrid"),
            known_query_ids={"q1"},
        )


def test_retrieval_failure_is_recorded_and_does_not_call_audit(tmp_path: Path) -> None:
    class FailingRepository(FakeRepository):
        def search(self, agenda, collection, *, deadline=None):
            raise CollectionDataError("EMBEDDING_API_FAILED", "provider unavailable")

    audit_called = False

    async def audit(*args):
        nonlocal audit_called
        audit_called = True
        raise AssertionError("audit must not run after retrieval failure")

    output = tmp_path / "failure.jsonl"
    counts = asyncio.run(
        execute_questions(
            questions=_questions()[:1],
            repository=FailingRepository([], mode="hybrid"),
            collection=_collection(),
            identity=_identity(mode="hybrid"),
            output_path=output,
            audit_runner=audit,
        )
    )

    row = _read_rows(output)[0]
    assert counts["errors"] == 1
    assert audit_called is False
    assert row["results"] == []
    assert row["apiErrors"] == [
        {
            "stage": "retrieval",
            "code": "EMBEDDING_API_FAILED",
            "message": "provider unavailable",
        }
    ]


def test_partial_selection_must_be_explicit() -> None:
    benchmark = SimpleNamespace(
        questions={row["queryId"]: row for row in _questions()}
    )
    with pytest.raises(RunGenerationError, match="require --partial"):
        _question_subset(
            benchmark,
            query_ids=[],
            categories=[],
            judgment_modes=[],
            limit=1,
            partial=False,
        )
    with pytest.raises(RunGenerationError, match="unknown query"):
        _question_subset(
            benchmark,
            query_ids=["missing"],
            categories=[],
            judgment_modes=[],
            limit=None,
            partial=True,
        )
    with pytest.raises(RunGenerationError, match="unknown category"):
        _question_subset(
            benchmark,
            query_ids=[],
            categories=["missing_category"],
            judgment_modes=[],
            limit=None,
            partial=True,
        )
    selected = _question_subset(
        benchmark,
        query_ids=["q2"],
        categories=["open_theme", "visual_motif"],
        judgment_modes=[],
        limit=1,
        partial=True,
    )
    assert [row["queryId"] for row in selected] == ["q2"]


def test_repeatable_category_selection_is_a_union_then_combines_with_query_ids() -> None:
    benchmark = SimpleNamespace(
        questions={row["queryId"]: row for row in _questions()}
    )

    selected = _question_subset(
        benchmark,
        query_ids=[],
        categories=["visual_motif", "open_theme"],
        judgment_modes=[],
        limit=None,
        partial=True,
    )
    assert [row["queryId"] for row in selected] == ["q1", "q2"]

    intersected = _question_subset(
        benchmark,
        query_ids=["q1"],
        categories=["visual_motif"],
        judgment_modes=[],
        limit=None,
        partial=True,
    )
    assert intersected == []


def test_frozen_category_and_judgment_mode_strata_select_100_gold_and_13_boundary() -> None:
    project_root = Path(__file__).resolve().parents[2]
    benchmark_dir = project_root / "data" / "qa" / "retrieval_eval_v1"
    benchmark = load_benchmark(
        benchmark_dir / "manifest.json",
        benchmark_dir / "questions.jsonl",
        benchmark_dir / "qrels.jsonl",
    )

    deterministic_gold = _question_subset(
        benchmark,
        query_ids=[],
        categories=[
            "exact_title_author_institution",
            "synonym_bilingual_typo",
            "date_material_region",
        ],
        judgment_modes=["deterministic_field_gold"],
        limit=None,
        partial=True,
    )
    boundary = _question_subset(
        benchmark,
        query_ids=[],
        categories=["ambiguous_out_of_scope"],
        judgment_modes=[
            "deterministic_evidence_boundary_no_relevant_objects"
        ],
        limit=None,
        partial=True,
    )

    assert len(deterministic_gold) == 100
    assert {row["judgmentMode"] for row in deterministic_gold} == {
        "deterministic_field_gold"
    }
    assert len(boundary) == 13
    assert {row["judgmentMode"] for row in boundary} == {
        "deterministic_evidence_boundary_no_relevant_objects"
    }


def test_unknown_judgment_mode_is_rejected() -> None:
    benchmark = SimpleNamespace(
        questions={row["queryId"]: row for row in _questions()}
    )
    with pytest.raises(RunGenerationError, match="unknown judgment mode"):
        _question_subset(
            benchmark,
            query_ids=[],
            categories=[],
            judgment_modes=["human_gold_that_does_not_exist"],
            limit=None,
            partial=True,
        )


def test_shadow_status_distinguishes_served_bm25_from_dense_candidate_mode() -> None:
    repository = FakeRepository([], mode="shadow")

    status = _retrieval_status(repository, _collection())

    assert status["requestedMode"] == "shadow"
    assert status["effectiveMode"] == "hybrid"
    assert status["servedMode"] == "bm25"
    assert status["shadowCandidateMode"] == "hybrid"
    assert status["fallbackApplied"] is False


def test_hybrid_provider_fallback_is_explicit_in_run_status() -> None:
    repository = FakeRepository([], mode="hybrid")
    repository.retrieval_status = lambda collection: SimpleNamespace(
        enabled=True,
        available=False,
        mode="bm25",
        reason="hybrid query failed; this request used BM25",
        model="hosted-embedding",
        fingerprint=None,
    )

    status = _retrieval_status(repository, _collection())

    assert status["requestedMode"] == "hybrid"
    assert status["effectiveMode"] == "bm25"
    assert status["servedMode"] == "bm25"
    assert status["fallbackApplied"] is True
