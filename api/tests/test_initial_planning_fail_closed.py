"""A capable planner outage must not silently erase a visitor's conditions."""

import asyncio
from types import SimpleNamespace

import pytest

from app.collections import CollectionDataError
from app.config import Settings
from app.generator import ExhibitionGenerator
from app.models import AgendaInput
from app.providers.deepseek import ProviderError


def setup(monkeypatch, *, outcomes, feature=True, question="想看人们交换礼物的场景", language="zh"):
    clock = [100.0]
    calls = []
    monkeypatch.setattr("app.generator.perf_counter", lambda: clock[0])
    provider = SimpleNamespace(configured=True, supports_retrieval_audit=True)
    if feature is not None:
        provider.supports_retrieval_query_planning = feature
    generator = ExhibitionGenerator(Settings(), SimpleNamespace(), provider)

    async def generate(prompt, payload, *, stage, timeout_seconds):
        calls.append({"stage": stage, "at": clock[0], "budget": timeout_seconds})
        if stage == "retrieval_plan_review":
            raise ProviderError("fixture review unavailable")
        elapsed, result = outcomes[len(calls) - 1]
        clock[0] += elapsed
        if isinstance(result, Exception):
            raise result
        return result

    async def search(agenda, collection, *, deadline, filters):
        calls.append({"stage": "search", "at": clock[0], "deadline": deadline})
        return ["fixture-recall"]

    monkeypatch.setattr(generator, "_generate_model_json", generate)
    monkeypatch.setattr(generator, "_search_async", search)
    agenda = AgendaInput(question=question, language=language, priorKnowledge="none",
                         durationMinutes=5, collectionId="fixture")

    def run(deadline=170.0):
        return asyncio.run(generator.prepare_initial_retrieval(
            agenda, SimpleNamespace(concept_aliases={}), deadline=deadline))

    return run, calls, clock, generator


def valid_draft():
    return {"inCollectionScope": True, "catalogueQueries": ["gift exchange"],
            "mandatoryPerObjectPredicates": ["people are exchanging a gift"],
            "evidenceMode": "visual_observation", "visualPredicateIds": ["p1"]}


@pytest.mark.parametrize("failure", [
    ProviderError("private endpoint details", code="provider_error"),
    ValueError("private malformed response"),
    TypeError("wrong response type"),
    KeyError("missing field"),
])
def test_non_network_planning_failure_prevents_recall_and_reports_service_not_evidence_gap(monkeypatch, failure):
    run, calls, _, _ = setup(monkeypatch, outcomes=[(1.0, failure)])
    with pytest.raises(CollectionDataError) as caught:
        run()
    error = caught.value
    assert error.code == "RETRIEVAL_PLAN_UNAVAILABLE"
    assert [call["stage"] for call in calls] == ["retrieval_plan:1"]
    assert error.details["serviceStage"] == "retrieval_planning" and error.details["retryable"] is True
    assert error.details["planningDiagnostics"]["queryPlanApplied"] is False
    assert "重试" in error.message and "不表示馆藏缺少" in error.message
    assert "收窄" not in error.message and "调整范围" not in error.message
    assert "private" not in str(error.details) and "private" not in error.message


@pytest.mark.parametrize("malformed", [{}, {"inCollectionScope": "yes"},
                                      {"inCollectionScope": True, "hardFilters": {"dateStart": "bad"}},
                                      [], None])
def test_invalid_draft_never_becomes_empty_unconditional_contract(monkeypatch, malformed):
    run, calls, _, _ = setup(monkeypatch, outcomes=[(1.0, malformed)])
    with pytest.raises(CollectionDataError) as caught:
        run()
    assert caught.value.code == "RETRIEVAL_PLAN_UNAVAILABLE"
    assert len(calls) == 1 and calls[0]["stage"] == "retrieval_plan:1"


def test_network_retry_uses_only_remaining_eight_second_draft_deadline(monkeypatch):
    run, calls, _, _ = setup(monkeypatch, outcomes=[
        (3.0, ProviderError("network down", code="provider_network_error")),
        (2.0, valid_draft()),
    ])
    with pytest.raises(CollectionDataError) as caught:
        run()
    assert [call["stage"] for call in calls] == [
        "retrieval_plan:1", "retrieval_plan:2", "retrieval_plan_review"]
    assert calls[0]["budget"] == 8.0 and calls[1]["budget"] == 5.0
    assert calls[0]["at"] + calls[0]["budget"] == calls[1]["at"] + calls[1]["budget"] == 108.0
    assert calls[2]["at"] + calls[2]["budget"] == 116.0
    assert caught.value.code == "RETRIEVAL_PLAN_UNAVAILABLE"
    assert caught.value.details["planningDiagnostics"]["planningReview"]["status"] == "unreviewed"


def test_two_network_failures_stop_without_recall_or_third_attempt(monkeypatch):
    failure = ProviderError("network down", code="provider_network_error")
    run, calls, _, _ = setup(monkeypatch, outcomes=[(2.0, failure), (2.0, failure)])
    with pytest.raises(CollectionDataError) as caught:
        run()
    assert len(calls) == 2
    assert caught.value.details["planningDiagnostics"]["planningAttempts"] == 2


def test_wall_clock_exhaustion_does_not_start_a_retry_or_unconditional_recall(monkeypatch):
    failure = ProviderError("network timeout", code="provider_network_error")
    run, calls, _, _ = setup(monkeypatch, outcomes=[(8.0, failure)])
    with pytest.raises(CollectionDataError):
        run()
    assert len(calls) == 1 and calls[0]["budget"] == 8.0


def test_insufficient_planning_time_preserves_audit_floor_and_makes_no_calls(monkeypatch):
    run, calls, _, generator = setup(monkeypatch, outcomes=[])
    timeout = float(generator.settings.rag_llm_audit_timeout_seconds)
    audit_floor = min(timeout, max(10.0, timeout * 0.75))
    with pytest.raises(CollectionDataError) as caught:
        run(deadline=100.0 + audit_floor + 1.5)
    assert calls == []
    assert caught.value.details["planningDiagnostics"]["planningStatus"] == "budget_exhausted"


@pytest.mark.parametrize("feature", [False, None])
def test_legacy_provider_without_feature_keeps_existing_recall_path(monkeypatch, feature):
    run, calls, _, _ = setup(monkeypatch, outcomes=[], feature=feature)
    result = run()
    assert result.results == ["fixture-recall"] and result.query_plan is None
    assert [call["stage"] for call in calls] == ["search"]


def test_browse_visit_does_not_require_a_topical_plan_even_with_capable_provider(monkeypatch):
    run, calls, _, _ = setup(monkeypatch, outcomes=[], question="随便带我逛逛")
    result = run()
    assert result.diagnostics["planningStatus"] == "browse_all"
    assert [call["stage"] for call in calls] == ["search"]


def test_valid_draft_cannot_bypass_unavailable_fidelity_review(monkeypatch):
    run, calls, _, _ = setup(monkeypatch, outcomes=[(6.5, valid_draft())])
    with pytest.raises(CollectionDataError) as caught:
        run()
    assert caught.value.code == "RETRIEVAL_PLAN_UNAVAILABLE"
    assert caught.value.details["planningDiagnostics"]["planningReview"]["status"] == "unreviewed"
    assert caught.value.details["retryable"] is True
    assert all(call["stage"] != "search" for call in calls)


def test_english_error_also_requests_retry_without_claiming_missing_collection(monkeypatch):
    run, _, _, _ = setup(monkeypatch, outcomes=[(1.0, ProviderError("outage"))], language="en")
    with pytest.raises(CollectionDataError) as caught:
        run()
    assert "retry" in caught.value.message
    assert "does not mean the collection lacks" in caught.value.message
