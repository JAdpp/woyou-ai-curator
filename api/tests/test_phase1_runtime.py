from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from .retrieval_contract_fixtures import strict_audit_fixture

from app.collections import CollectionDataError, CollectionRepository, SearchResult
from app.config import Settings
from app.dense_retrieval import DenseHit, DenseStatus, EvidenceDenseHit, EvidenceHit
from app.generator import ExhibitionGenerator
from app.models import AgendaInput, EvidenceChunk, MuseumObject
from app.retrieval_filters import FilterSpec
from app.structured_filters import build_structured_filter_index


def _write_runtime_collection(root: Path) -> Path:
    collection_dir = root / "phase1_fixture"
    collection_dir.mkdir(parents=True)
    records = [
        (
            "china-bronze",
            "Bronze vessel",
            "China",
            "Bronze",
            "Vessel",
            "A ritual threshold vessel contains the rare phrase spectral liminality.",
        ),
        (
            "china-ceramic",
            "Ceramic figure",
            "China",
            "Glazed ceramic",
            "Figure",
            "A ceremonial threshold figure used in a domestic setting.",
        ),
        (
            "japan-print",
            "Woodblock print",
            "Japan",
            "Ink on paper",
            "Print",
            "A threshold scene printed in ink on paper.",
        ),
        (
            "egypt-sculpture",
            "Standing official",
            "Egypt",
            "Limestone",
            "Sculpture",
            "A threshold figure carved from limestone.",
        ),
        (
            "netherlands-plate",
            "Glazed plate",
            "Netherlands",
            "Glazed ceramic",
            "Plate",
            "A threshold plate made for domestic dining.",
        ),
        (
            "mexico-mask",
            "Ceremonial mask",
            "Mexico",
            "Wood",
            "Mask",
            "A threshold mask worn for a ceremony.",
        ),
    ]
    objects: list[dict[str, Any]] = []
    for object_id, title, culture, material, object_type, evidence_text in records:
        evidence_id = f"fixture:{object_id}:e1"
        objects.append(
            {
                "id": f"fixture:{object_id}",
                "sourceId": object_id,
                "accessionNumber": object_id,
                "title": title,
                "date": "1900",
                "dateEarliest": 1900,
                "dateLatest": 1900,
                "creator": "Unknown",
                "material": material,
                "type": object_type,
                "culture": culture,
                "cultureDisplay": culture,
                "description": "A deliberately generic catalogue summary.",
                "imageUrl": f"https://images.test/{object_id}.jpg",
                "objectUrl": f"https://objects.test/{object_id}",
                "rights": "CC0",
                "rightsUri": "https://creativecommons.org/publicdomain/zero/1.0/",
                "institutionId": "fixture",
                "institution": "Fixture Museum",
                "evidenceDepth": "full",
                "evidence": [
                    {
                        "id": evidence_id,
                        "text": evidence_text,
                        "sourceUrl": f"https://objects.test/{object_id}",
                        "sourceTitle": title,
                        "sourceKind": "institution_curatorial_text",
                    }
                ],
            }
        )
    (collection_dir / "objects.json").write_text(
        json.dumps({"objects": objects}), encoding="utf-8"
    )
    (collection_dir / "manifest.json").write_text(
        json.dumps(
            {
                "id": "phase1_fixture",
                "name": "Phase 1 runtime fixture",
                "version": "v1",
            }
        ),
        encoding="utf-8",
    )
    return root


def _agenda(question: str) -> AgendaInput:
    return AgendaInput(
        question=question,
        priorKnowledge="none",
        durationMinutes=10,
        collectionId="phase1_fixture",
    )


class _EmptyDenseIndex:
    def embed_query(self, question: str, *, deadline: float | None = None) -> str:
        return question

    def embed_queries(
        self, questions: list[str], *, deadline: float | None = None
    ) -> list[str]:
        return list(questions)

    def search_vector(
        self,
        _query_vector: str,
        *,
        top_k: int,
        allowed_object_ids: set[str] | None = None,
    ) -> list[DenseHit]:
        return []

    def search_evidence_vector(
        self, _query_vector: str, *, top_k: int
    ) -> list[EvidenceDenseHit]:
        return []

    def evidence_hits(
        self,
        _query_vector: str,
        _object_ids: set[str],
        *,
        max_evidence_ids: int = 3,
    ) -> dict[str, EvidenceHit]:
        return {}


class _SingleObjectDenseIndex(_EmptyDenseIndex):
    def __init__(self, object_id: str, evidence_id: str) -> None:
        self.object_id = object_id
        self.evidence_id = evidence_id

    def search_vector(
        self,
        _query_vector: str,
        *,
        top_k: int,
        allowed_object_ids: set[str] | None = None,
    ) -> list[DenseHit]:
        if allowed_object_ids is not None and self.object_id not in allowed_object_ids:
            return []
        return [DenseHit(self.object_id, 0.99)]

    def search_evidence_vector(
        self, _query_vector: str, *, top_k: int
    ) -> list[EvidenceDenseHit]:
        return [EvidenceDenseHit(self.object_id, self.evidence_id, 0.98)]

    def evidence_hits(
        self,
        _query_vector: str,
        object_ids: set[str],
        *,
        max_evidence_ids: int = 3,
    ) -> dict[str, EvidenceHit]:
        if self.object_id not in object_ids:
            return {}
        return {
            self.object_id: EvidenceHit(
                score=0.98,
                evidence_ids=(self.evidence_id,),
            )
        }


class _DenseManager:
    def __init__(self, index: _EmptyDenseIndex) -> None:
        self.index = index

    def cache_signature(self, _collection: object) -> tuple[str, int, int]:
        return ("phase1-test", 1, 1)

    def get(self, _collection: object) -> _EmptyDenseIndex:
        return self.index

    def status(self, _collection: object) -> DenseStatus:
        return DenseStatus(
            enabled=True,
            available=True,
            mode="hybrid",
            reason="test fixture",
            model="test fixture",
        )

    def mark_query_failure(self, _collection: object, _error: Exception) -> None:
        return None

    def mark_query_success(
        self, _collection: object, _index: _EmptyDenseIndex
    ) -> None:
        return None


def _repository_with_filter_index(
    tmp_path: Path,
    *,
    rag_mode: str = "bm25",
    dense_index: _EmptyDenseIndex | None = None,
    **kwargs: Any,
) -> tuple[CollectionRepository, object]:
    collection_root = _write_runtime_collection(tmp_path / "collections")
    loader = CollectionRepository(
        collection_root,
        default_collection_id="phase1_fixture",
        rag_mode="bm25",
    )
    collection = loader.get()
    filter_root = tmp_path / "filters"
    build_structured_filter_index(collection, index_root=filter_root)
    repository = CollectionRepository(
        collection_root,
        default_collection_id="phase1_fixture",
        rag_mode=rag_mode,
        dense_manager=(
            _DenseManager(dense_index)
            if dense_index is not None
            else None
        ),
        structured_filter_index_dir=filter_root,
        **kwargs,
    )
    return repository, repository.get()


def test_explicit_filter_spec_limits_search_and_search_many(tmp_path: Path) -> None:
    repository, collection = _repository_with_filter_index(tmp_path)
    filters = FilterSpec(cultures=("China",))

    single = repository.search(
        _agenda("threshold"),
        collection,  # type: ignore[arg-type]
        filters=filters,
    )
    batched = repository.search_many(
        [_agenda("threshold"), _agenda("ceremonial threshold")],
        collection,  # type: ignore[arg-type]
        filters=filters,
    )

    assert {result.obj.culture for result in single} == {"China"}
    assert {result.obj.id for result in single} == {
        "fixture:china-bronze",
        "fixture:china-ceramic",
    }
    assert all(batch for batch in batched)
    assert all(
        {result.obj.culture for result in batch} == {"China"}
        for batch in batched
    )


def test_small_verified_filter_pool_survives_unmatched_query_words(tmp_path: Path) -> None:
    repository, collection = _repository_with_filter_index(tmp_path)
    # A planner's explicit fields are sufficient for recall, but not for
    # acceptance: no source evidence or semantic judgement is fabricated.
    result = repository.search(
        _agenda("筛出符合这些条件的记录"), collection,
        filters=FilterSpec(date_start=1850, date_end=1950, cultures=("中国",), materials=("bronze",)),
    )
    assert [item.obj.id for item in result] == ["fixture:china-bronze"]
    assert result[0].retrieval_sources == ("structured_filter",)
    assert result[0].matched_evidence_ids == ()
    assert repository._structured_candidate_pool(
        [result[0].obj] * 61, FilterSpec(cultures=("China",)), [],
    ) == []


@pytest.mark.parametrize("batched", [False, True])
def test_explicit_filters_fail_closed_when_verified_index_is_missing(
    tmp_path: Path,
    batched: bool,
) -> None:
    collection_root = _write_runtime_collection(tmp_path / "collections")
    repository = CollectionRepository(
        collection_root,
        default_collection_id="phase1_fixture",
        rag_mode="bm25",
        structured_filter_index_dir=tmp_path / "missing-filter-index",
    )
    collection = repository.get()
    filters = FilterSpec(cultures=("China",))

    with pytest.raises(CollectionDataError) as raised:
        if batched:
            repository.search_many(
                [_agenda("threshold")], collection, filters=filters
            )
        else:
            repository.search(_agenda("threshold"), collection, filters=filters)

    assert raised.value.code == "STRUCTURED_FILTER_UNAVAILABLE"


def test_evidence_fts_bm25_is_an_independent_recall_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, collection = _repository_with_filter_index(
        tmp_path,
        rag_mode="hybrid",
        dense_index=_EmptyDenseIndex(),
    )
    # Isolate the evidence FTS channel from object BM25 and dense recall. The
    # production object document also contains evidence text, so simply using a
    # rare phrase would otherwise exercise two lexical channels at once.
    monkeypatch.setattr(repository, "_lexical_search", lambda *_args: [])

    results = repository.search(
        _agenda("spectral liminality"),
        collection,  # type: ignore[arg-type]
    )

    assert [result.obj.id for result in results] == ["fixture:china-bronze"]
    assert "bm25_evidence" in results[0].retrieval_sources
    assert results[0].matched_evidence_ids == ("fixture:china-bronze:e1",)
    assert dict(results[0].field_scores)["evidence_bm25"] > 0


def test_hybrid_object_fts_is_a_bounded_candidate_channel(tmp_path: Path) -> None:
    repository, collection = _repository_with_filter_index(
        tmp_path,
        rag_mode="hybrid",
        dense_index=_EmptyDenseIndex(),
    )

    results = repository.search(
        _agenda("ceremonial threshold"),
        collection,  # type: ignore[arg-type]
    )

    assert results
    assert any("bm25_object_fts" in result.retrieval_sources for result in results)
    assert any("object_fts_bm25" in dict(result.field_scores) for result in results)


def test_shadow_object_fts_candidate_does_not_change_served_bm25(
    tmp_path: Path,
) -> None:
    trace = _TraceCapture()
    repository, collection = _repository_with_filter_index(
        tmp_path,
        rag_mode="shadow",
        dense_index=_EmptyDenseIndex(),
        trace_writer=trace,  # type: ignore[arg-type]
    )

    served = repository.search(
        _agenda("ceremonial threshold"),
        collection,  # type: ignore[arg-type]
    )

    assert served
    assert all(result.retrieval_sources == ("bm25",) for result in served)
    assert len(trace.calls) == 1
    candidate = trace.calls[0]["candidate_results"]
    assert candidate
    assert any("bm25_object_fts" in result.retrieval_sources for result in candidate)
    assert trace.calls[0]["stage_latency_ms"]["object_fts_bm25"] >= 0


class _TraceCapture:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def write(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def test_shadow_search_and_search_many_serve_bm25_but_trace_v3_candidate(
    tmp_path: Path,
) -> None:
    trace = _TraceCapture()
    object_id = "fixture:china-bronze"
    evidence_id = "fixture:china-bronze:e1"
    collection_root = _write_runtime_collection(tmp_path / "collections")
    repository = CollectionRepository(
        collection_root,
        default_collection_id="phase1_fixture",
        rag_mode="shadow",
        dense_manager=_DenseManager(_SingleObjectDenseIndex(object_id, evidence_id)),
        trace_writer=trace,  # type: ignore[arg-type]
        structured_filters_enabled=False,
    )
    collection = repository.get()

    served_single = repository.search(_agenda("ritual vessel"), collection)
    served_batch = repository.search_many(
        [_agenda("ancient ritual vessel")], collection
    )[0]

    assert served_single and served_batch
    assert all(
        result.retrieval_sources == ("bm25",)
        for result in [*served_single, *served_batch]
    )
    assert len(trace.calls) == 2
    for call in trace.calls:
        assert call["serving_mode"] == "shadow"
        assert all(
            result.retrieval_sources == ("bm25",)
            for result in call["served_results"]
        )
        assert any(
            "dense_object" in result.retrieval_sources
            and "evidence_rerank" in result.retrieval_sources
            for result in call["candidate_results"]
        )


def _rerank_result(index: int) -> SearchResult:
    evidence_id = f"object-{index}:e1"
    obj = MuseumObject(
        id=f"object-{index}",
        sourceId=f"object-{index}",
        title=f"Object {index} with a deliberately descriptive catalogue title",
        creator="Fixture maker",
        date="late nineteenth century",
        culture="Fixture culture",
        type="Ceremonial object",
        material="mixed media",
        description="Generic object description.",
        imageUrl=f"https://images.test/object-{index}.jpg",
        objectUrl=f"https://objects.test/object-{index}",
        rights="CC0",
        institution="Fixture Museum",
        institutionId="fixture",
        evidence=[
            EvidenceChunk(
                id=evidence_id,
                text=(
                    "A long institution evidence sentence describing the object, "
                    "its making, use, and documented historical context. " * 4
                ),
                sourceUrl=f"https://objects.test/object-{index}",
                sourceTitle=f"Object {index}",
                sourceKind="institution_curatorial_text",
            )
        ],
    )
    return SearchResult(
        obj=obj,
        score=float(100 - index),
        matched_evidence_ids=(evidence_id,),
        retrieval_sources=("bm25", "dense_object", "evidence_rerank"),
    )


class _CapturingReranker:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int,
        instruct: str | None,
        deadline: float | None = None,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "instruct": instruct,
                "deadline": deadline,
            }
        )
        if self.fail:
            raise RuntimeError("fixture rerank outage")
        return SimpleNamespace(
            results=[
                SimpleNamespace(index=index, relevance_score=0.9 - index * 0.001)
                for index in range(min(top_n, len(documents)))
            ]
        )


def test_long_question_rerank_stays_under_request_budget_and_forwards_deadline(
    tmp_path: Path,
) -> None:
    reranker = _CapturingReranker()
    repository = CollectionRepository(
        tmp_path,
        default_collection_id="unused",
        rag_mode="hybrid",
        reranker=reranker,
        rerank_candidate_count=60,
        rerank_top_n=24,
        rerank_instruct="Rank only by direct museum-record relevance.",
    )
    question = "How did ceremonial thresholds mediate authority across cultures? " * 8
    deadline = time.perf_counter() + 10.0

    reranked = repository._apply_semantic_reranker(
        question,
        [_rerank_result(index) for index in range(60)],
        deadline=deadline,
    )

    assert reranked
    assert len(reranker.calls) == 1
    call = reranker.calls[0]
    assert call["deadline"] == deadline
    documents = call["documents"]
    request_characters = (
        len(question) * len(documents)
        + sum(len(document) for document in documents)
        + len(call["instruct"] or "")
        + 128
    )
    assert request_characters <= 28_500
    assert 0 < len(documents) < 60


def test_rerank_failure_is_observable_and_not_cached(tmp_path: Path) -> None:
    reranker = _CapturingReranker(fail=True)
    object_id = "fixture:china-bronze"
    evidence_id = "fixture:china-bronze:e1"
    collection_root = _write_runtime_collection(tmp_path / "collections")
    repository = CollectionRepository(
        collection_root,
        default_collection_id="phase1_fixture",
        rag_mode="hybrid",
        dense_manager=_DenseManager(_SingleObjectDenseIndex(object_id, evidence_id)),
        reranker=reranker,
        structured_filters_enabled=False,
    )
    collection = repository.get()
    agenda = _agenda("ritual vessel")

    first = repository.search(agenda, collection)
    second = repository.search(agenda, collection)

    assert len(reranker.calls) == 2
    assert first == second
    assert all(
        "qwen_rerank_fallback" in result.retrieval_sources for result in first
    )


class _FilterPlanningProvider:
    configured = True
    supports_retrieval_audit = True
    supports_retrieval_query_planning = True

    def __init__(self) -> None:
        self.audit_calls = 0

    async def generate_retrieval_query_plan_json(
        self, _prompt: str, _payload: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "inCollectionScope": True,
            "queryInterpretation": "Chinese ritual objects",
            "catalogueQueries": ["ritual object"],
            "semanticQuery": "Chinese ritual object",
            "mandatoryPerObjectPredicates": [],
            "poolCoverageLegs": [],
            "selectionRationaleConstraints": [],
            "hardFilters": {"cultures": ["China"]},
            "reason": "The visitor explicitly requested Chinese objects.",
        }

    @strict_audit_fixture
    async def generate_retrieval_audit_json(
        self, _prompt: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        self.audit_calls += 1
        if self.audit_calls == 1:
            return {
                "queryInterpretation": "Chinese ritual objects",
                "answerability": "unsupported",
                "accepted": [],
                "expansionReason": "insufficient_direct_objects",
                "searchQueries": ["ceremonial vessel"],
                "coverageGap": "The first pool needs one bounded expansion.",
            }
        accepted = [
            {
                "objectId": candidate["objectId"],
                "relevanceScore": 0.9,
                "evidenceIds": [candidate["evidence"][0]["id"]],
            }
            for candidate in payload["candidates"][:5]
        ]
        return {
            "queryInterpretation": "Chinese ritual objects",
            "answerability": "supported",
            "accepted": accepted,
            "expansionReason": "none",
            "searchQueries": [],
            "coverageGap": "",
        }


def _agent_result(index: int) -> SearchResult:
    evidence_id = f"agent-{index}:e1"
    obj = MuseumObject(
        id=f"agent-{index}",
        sourceId=f"agent-{index}",
        title=f"Agent fixture {index}",
        culture="China",
        description=f"Institution description {index}.",
        imageUrl=f"https://images.test/agent-{index}.jpg",
        objectUrl=f"https://objects.test/agent-{index}",
        rights="CC0",
        institution="Fixture Museum",
        institutionId="fixture",
        evidence=[
            EvidenceChunk(
                id=evidence_id,
                text=f"Institution record for agent fixture {index}.",
                sourceUrl=f"https://objects.test/agent-{index}",
                sourceTitle=f"Agent fixture {index}",
                sourceKind="institution_curatorial_text",
            )
        ],
    )
    return SearchResult(
        obj=obj,
        score=float(100 - index),
        matched_evidence_ids=(evidence_id,),
        retrieval_sources=("bm25",),
    )


def test_generator_forwards_query_plan_filters_to_every_retrieval_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_root = _write_runtime_collection(tmp_path / "collections")
    repository = CollectionRepository(
        collection_root,
        default_collection_id="phase1_fixture",
        rag_mode="bm25",
    )
    collection = repository.get()
    candidates = [_agent_result(index) for index in range(6)]
    search_calls: list[dict[str, Any]] = []
    search_many_calls: list[dict[str, Any]] = []

    def capture_search(
        agenda: AgendaInput,
        _collection: object,
        *,
        deadline: float | None = None,
        filters: FilterSpec | None = None,
    ) -> list[SearchResult]:
        search_calls.append(
            {"question": agenda.question, "deadline": deadline, "filters": filters}
        )
        return candidates

    def capture_search_many(
        agendas: list[AgendaInput],
        _collection: object,
        *,
        deadline: float | None = None,
        atomic: bool = False,
        filters: FilterSpec | None = None,
    ) -> list[list[SearchResult]]:
        search_many_calls.append(
            {
                "questions": tuple(agenda.question for agenda in agendas),
                "deadline": deadline,
                "atomic": atomic,
                "filters": filters,
            }
        )
        return [candidates for _agenda in agendas]

    monkeypatch.setattr(repository, "search", capture_search)
    monkeypatch.setattr(repository, "search_many", capture_search_many)
    provider = _FilterPlanningProvider()
    generator = ExhibitionGenerator(
        Settings(
            rag_retrieval_timeout_seconds=60,
            rag_llm_audit_timeout_seconds=20,
            rag_agentic_max_queries=3,
            rag_llm_audit_top_k=20,
        ),
        repository,
        provider=provider,  # type: ignore[arg-type]
    )
    agenda = _agenda("请只看中国文化中的仪式性器物")

    outcome = asyncio.run(
        generator._agentic_retrieve(
            agenda,
            collection,
            candidates,
            required_count=5,
            deadline=time.perf_counter() + 60.0,
        )
    )

    expected = FilterSpec(cultures=("China",))
    assert outcome.failure_code is None
    assert provider.audit_calls == 2
    assert search_calls == [
        {
            "question": agenda.question,
            "deadline": search_calls[0]["deadline"],
            "filters": expected,
        }
    ]
    assert len(search_many_calls) == 2
    assert search_many_calls[0]["questions"] == (
        "Chinese ritual object",
        "ritual object",
    )
    assert search_many_calls[1]["questions"] == ("ceremonial vessel",)
    assert all(call["atomic"] is True for call in search_many_calls)
    assert all(call["filters"] == expected for call in search_many_calls)
    assert all(call["deadline"] is not None for call in search_many_calls)
