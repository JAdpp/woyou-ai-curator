"""Regression cases for observed combination-recall and audit-window failures."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from .retrieval_contract_fixtures import strict_audit_fixture

from app.collections import CollectionDataError, CollectionRepository, SearchResult
from app.config import Settings
from app.generator import ExhibitionGenerator
from app.models import AgendaInput, EvidenceChunk, MuseumObject
from app.providers.deepseek import ProviderError
from app.retrieval_agent import audit_payload, audit_prompt, parse_query_plan, query_plan_prompt
from app.retrieval_filters import FilterSpec


def _agenda(question):
    return AgendaInput(question=question, priorKnowledge="none", durationMinutes=5, collectionId="fixture")


def _result(object_id, title, *, culture="", score=80):
    evidence_id = f"{object_id}:metadata"
    obj = MuseumObject(
        id=object_id, sourceId=object_id, title=title,
        description=f"Institution description for {title}.",
        imageUrl=f"https://example.test/{object_id}.jpg",
        objectUrl=f"https://example.test/{object_id}",
        rights="CC0", institution="Fixture Museum", institutionId="fixture", culture=culture,
        evidence=[EvidenceChunk(
            id=evidence_id, text=f"Institution record identifies {title}.",
            sourceUrl=f"https://example.test/{object_id}", sourceTitle=title,
            sourceKind="institution_metadata",
        )],
    )
    return SearchResult(
        obj=obj, score=score, matched_evidence_ids=(evidence_id,),
        retrieval_sources=("dense_object", "dense_open_query", "evidence_rerank"),
        dense_score=0.72, evidence_score=0.70,
    )


class _ControlledRepository(CollectionRepository):
    def __init__(self, results, events):
        self.results = results
        self.events = events
        self.rag_mode = "bm25"

    def search(self, agenda, collection, *, filters=None, deadline=None):
        self.events.append(("search", filters))
        return self.results

    def match_question_policy(self, collection, question):
        return None

    def search_many(self, agendas, collection, *, filters=None, **_kwargs):
        self.events.append(("planned_search", filters))
        return [self.results for _ in agendas]


class _ControlledProvider:
    configured = True
    supports_retrieval_audit = True
    supports_retrieval_query_planning = True

    def __init__(self, events, fail=False):
        self.events = events
        self.fail = fail

    async def generate_retrieval_query_plan_json(self, prompt, payload):
        if "recallDraft" in payload:
            self.events.append(("plan_review", payload))
            return _review_fixture(payload)
        self.events.append(("plan", payload))
        if self.fail:
            raise ProviderError("fixture planning outage")
        return {
            "inCollectionScope": True,
            "catalogueQueries": [],
            "hardFilters": {"dateStart": 1700, "dateEnd": 1800, "materials": ["silk"]},
        }

    @strict_audit_fixture
    async def generate_retrieval_audit_json(self, prompt, payload):
        self.events.append(("audit", payload))
        return {
            "answerability": "supported",
            "expansionReason": "none",
            "accepted": [
                {
                    "objectId": item["objectId"],
                    "relevanceScore": 0.95,
                    "evidenceIds": [item["evidence"][0]["id"]],
                }
                for item in payload["candidates"]
            ],
        }


def _review_fixture(payload):
    # A successful review fixture must obey the current review protocol. A raw
    # planner document is invalid and would correctly trigger contract repair.
    recall = payload["recallDraft"]
    return {"searchPlan": {
        **recall, "inCollectionScope": True, "queryInterpretation": "丝绸作品",
        "catalogueQueries": ["silk"], "semanticQuery": "silk museum objects",
        "evidenceMode": "record_explanation", "reason": "fixture review",
    }, "intents": [{
        "id": "i1", "sourceQuote": payload["visitorQuestion"],
        "type": "admission", "text": "Object is made using silk.",
        "evidenceScope": "institution_record",
    }]}


def _prepared_run(*, fail=False, question="找十八世纪的丝绸藏品"):
    events = []
    results = [_result(f"r-{i}", f"Silk painting {i}") for i in range(5)]
    repo = _ControlledRepository(results, events)
    generator = ExhibitionGenerator(Settings(), repo, _ControlledProvider(events, fail))
    agenda = _agenda(question)
    collection = SimpleNamespace(concept_aliases={})

    async def run():
        initial = await generator.prepare_initial_retrieval(agenda, collection)
        outcome = await generator._agentic_retrieve(
            agenda, collection, initial.results,
            required_count=5, initial_query_plan=initial.query_plan,
            planning_attempted=True,
        )
        return initial, outcome

    return events, *asyncio.run(run())


def test_explicit_filters_precede_first_search_and_planning_is_not_repeated():
    events, initial, outcome = _prepared_run()
    assert [kind for kind, _ in events] == ["plan", "plan_review", "search", "planned_search", "audit"]
    # Natural-language material categories remain binding at evidence audit,
    # but do not become a literal recall filter that excludes narrower terms.
    assert events[2][1] == FilterSpec(date_start=1700, date_end=1800)
    assert events[3][1] == events[2][1]
    assert events[4][1]["retrievalContract"]["explicitMaterials"] == ["silk"]
    assert initial.diagnostics["queryPlanApplied"] is True
    assert events[0][1] == {"visitorQuestion": "找十八世纪的丝绸藏品", "language": "zh"}
    assert len(outcome.stage_results["audit_sample_1"]) == 5
    assert outcome.answerability == "supported"


def test_failed_capable_preplan_is_service_failure_not_unconditional_recall():
    with pytest.raises(CollectionDataError) as caught:
        _prepared_run(fail=True)
    assert caught.value.code == "RETRIEVAL_PLAN_UNAVAILABLE"
    assert caught.value.details["planningDiagnostics"]["planningStatus"] == "unavailable"
    assert "不表示馆藏缺少" in caught.value.message


def test_browse_all_skips_unnecessary_query_planning():
    events, initial, _ = _prepared_run(question="随便带我逛逛")
    assert initial.diagnostics["planningStatus"] == "browse_all"
    assert not any(kind == "plan" for kind, _ in events)


def test_post_rerank_sample_preserves_two_distinct_axes_with_same_culture():
    # Both low-ranked query legs must reach audit even though origin diversity
    # cannot distinguish them from the generic head after global reranking.
    head = [_result(f"noise-{i}", f"Generic writing {i}", culture="China") for i in range(24)]
    dedication = _result("dedication", "Votive dedication inscription", culture="China", score=21)
    ownership = _result("ownership", "Owner inscription", culture="China", score=19)
    candidates = [*head, dedication, ownership]
    before = {r.obj.id: r.score for r in candidates}
    sample = ExhibitionGenerator._audit_candidate_sample(
        candidates, top_k=12, cross_cultural=False,
        query_batches=[[dedication], [ownership]],
    )
    assert {"dedication", "ownership"}.issubset({r.obj.id for r in sample})
    assert len(sample) == len({r.obj.id for r in sample}) == 12
    assert all(r.score == before[r.obj.id] for r in sample)


def test_post_rerank_culture_and_problem_axis_reservations_share_bounded_window():
    head = [_result(f"head-{i}", f"Generic vase {i}", culture="China") for i in range(25)]
    japan = _result("japan", "Japanese ritual vessel", culture="Japan")
    iran = _result("iran", "Iranian ritual vessel", culture="Iran")
    axis = _result("repair", "Repaired Chinese vessel", culture="China")
    sample = ExhibitionGenerator._audit_candidate_sample(
        [*head, japan, iran, axis], top_k=9, cross_cultural=True,
        cultural_obligations=ExhibitionGenerator._cultural_coverage_obligations("中国、日本和伊朗的器物"),
        query_batches=[[axis]],
    )
    assert {"japan", "iran", "repair"}.issubset({r.obj.id for r in sample})
    assert len(sample) == len({r.obj.id for r in sample}) == 9


def test_open_exploration_contract_keeps_observable_anchor_not_historical_claim():
    plan = parse_query_plan({
        "inCollectionScope": True,
        "catalogueQueries": ["isolated figure"],
        "evidenceMode": "open_exploration",
        "mandatoryPerObjectPredicates": ["馆方描述一个孤立的人物或稀疏构图"],
    }, question="想看看有距离感的作品", max_queries=3)
    payload = audit_payload(
        "想看看有距离感的作品", [_result("figure", "Lone figure")],
        required_count=5, top_k=20, pass_number=1,
        evidence_mode=plan.evidence_mode,
        mandatory_predicates=plan.mandatory_predicates,
    )
    assert payload["retrievalContract"]["evidenceMode"] == "open_exploration"
    assert payload["retrievalContract"]["mandatoryPerObjectPredicates"][0]["text"] == plan.mandatory_predicates[0]
    assert "text-only audit" in audit_prompt()
    assert "historical influence still require explicit record evidence" in audit_prompt()


def test_missing_or_invalid_evidence_mode_keeps_strict_record_contract():
    for value in (None, "answer_everything", ["open_exploration"]):
        plan = parse_query_plan({
            "inCollectionScope": True, "catalogueQueries": [], "evidenceMode": value,
        }, question="这类器物为什么用于祭祀？", max_queries=3)
        assert plan.evidence_mode == "record_explanation"


@pytest.mark.parametrize("axis_count,slots", [(3, 3), (5, 5), (3, 4)])
def test_synthesis_uses_only_spare_slots_and_never_replaces_explicit_axes(axis_count, slots):
    axes = [f"object relation {index}" for index in range(axis_count)]
    synthesis = "all documented object relations compared"
    results = [_result(f"r-{index}", f"Related object {index}") for index in range(5)]
    events = []

    class BoundedRepository:
        def match_question_policy(self, collection, question):
            return None

        def search_many(self, agendas, collection):
            events.extend(agenda.question for agenda in agendas)
            return [results for _ in agendas]

    plan = parse_query_plan({
        "inCollectionScope": True, "catalogueQueries": axes,
        "semanticQuery": synthesis,
    }, question="比较这些对象中明确列出的各类关系", max_queries=slots)
    generator = ExhibitionGenerator(
        Settings(rag_agentic_max_queries=slots), BoundedRepository(),
        _ControlledProvider([]),
    )
    outcome = asyncio.run(generator._agentic_retrieve(
        _agenda("比较这些对象中明确列出的各类关系"), SimpleNamespace(), results,
        required_count=5, initial_query_plan=plan, planning_attempted=True,
    ))
    assert all(axis in events for axis in axes)
    assert (synthesis in events) is (slots > axis_count)
    assert len(events) <= slots
    assert outcome.expanded_queries == tuple(events)


@pytest.mark.parametrize("failure_code,expected_attempts", [
    ("provider_network_error", 2), ("provider_error", 1),
])
def test_initial_planner_retries_only_transport_error_with_original_deadline(monkeypatch, failure_code, expected_attempts):
    clock = [100.0]
    monkeypatch.setattr("app.generator.perf_counter", lambda: clock[0])
    events = []
    results = [_result("silk", "Silk painting")]
    generator = ExhibitionGenerator(
        Settings(), _ControlledRepository(results, events), _ControlledProvider(events),
    )
    budgets = []

    async def call(prompt, payload, *, stage, timeout_seconds):
        if stage.startswith("retrieval_plan_review"):
            return _review_fixture(payload)
        budgets.append(timeout_seconds)
        clock[0] += 2.0
        if len(budgets) == 1:
            message = "transient connection failed" if failure_code == "provider_network_error" else "DeepSeek returned HTTP 402"
            raise ProviderError(message, code=failure_code)
        return {"inCollectionScope": True, "catalogueQueries": [], "hardFilters": {"materials": ["silk"]}}

    monkeypatch.setattr(generator, "_generate_model_json", call)
    async def prepare():
        return await generator.prepare_initial_retrieval(
            _agenda("查找丝绸作品"), SimpleNamespace(concept_aliases={}), deadline=150.0,
        )

    if expected_attempts == 2:
        initial = asyncio.run(prepare())
        diagnostics = initial.diagnostics
    else:
        with pytest.raises(CollectionDataError) as caught:
            asyncio.run(prepare())
        assert caught.value.code == "RETRIEVAL_PLAN_UNAVAILABLE"
        diagnostics = caught.value.details["planningDiagnostics"]
    assert len(budgets) == expected_attempts
    assert diagnostics["planningAttempts"] == expected_attempts
    assert budgets[0] == 8.0
    if expected_attempts == 2:
        assert budgets[1] == 6.0
        assert initial.query_plan.filters.materials == ("silk",)
        assert initial.diagnostics["planningStatus"] == "applied"
    else:
        assert diagnostics["planningStatus"] == "unavailable"


def test_initial_planner_does_not_retry_after_shared_budget_exhaustion(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("app.generator.perf_counter", lambda: clock[0])
    events = []
    generator = ExhibitionGenerator(
        Settings(), _ControlledRepository([], events), _ControlledProvider(events),
    )

    async def call(prompt, payload, *, stage, timeout_seconds):
        clock[0] += 7.5
        raise ProviderError("connection failed", code="provider_network_error")

    monkeypatch.setattr(generator, "_generate_model_json", call)
    with pytest.raises(CollectionDataError) as caught:
        asyncio.run(generator.prepare_initial_retrieval(
            _agenda("查找丝绸作品"), SimpleNamespace(concept_aliases={}), deadline=150.0,
        ))
    assert caught.value.code == "RETRIEVAL_PLAN_UNAVAILABLE"
    diagnostics = caught.value.details["planningDiagnostics"]
    assert diagnostics["planningAttempts"] == 1
    assert diagnostics["planningStatus"] == "unavailable"


def test_planner_has_canonical_regional_taxonomy_without_country_broadening():
    prompt = query_plan_prompt()
    for pack in ("east_asia", "south_asia", "southeast_asia", "west_asia_north_africa", "europe", "africa", "americas", "oceania"):
        assert pack in prompt
    assert "Do not map North Africa alone to west_asia_north_africa" in prompt
    assert "A single country remains its country name" in prompt
