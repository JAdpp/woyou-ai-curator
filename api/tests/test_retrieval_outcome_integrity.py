"""Disjoint audit summaries and transport failures must retain their scope."""
import asyncio
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.generator import AgenticRetrievalOutcome, ExhibitionGenerator, merge_pool_audits
from app.models import AgendaInput
from app.retrieval_agent import RetrievalAudit


def item(key):
    return SimpleNamespace(obj=SimpleNamespace(id=key), set_witnesses=())


@pytest.mark.parametrize("language", ["zh", "en"])
def test_empty_second_window_cannot_negate_first_verified_objects(language):
    first = RetrievalAudit(True, [item("a")], (), answerability="partially_supported",
                           coverage_gap="A known initial limitation", condition_checks=({"first": True},))
    second = RetrievalAudit(True, [], (), answerability="unsupported", coverage_gap="No relevant object exists",
                            condition_checks=({"second": True},))
    merged = merge_pool_audits(first, second, language)
    assert [row.obj.id for row in merged.accepted] == ["a"]
    assert merged.answerability == "partially_supported"
    assert "No relevant" not in merged.coverage_gap and "initial limitation" not in merged.coverage_gap
    assert merged.condition_checks == ({"first": True}, {"second": True})
    assert first.coverage_gap == "A known initial limitation" and second.coverage_gap == "No relevant object exists"


def test_supported_recovery_does_not_resurrect_old_gap_and_dedupes():
    first = RetrievalAudit(True, [item("a")], (), coverage_gap="stale gap")
    second = RetrievalAudit(True, [item("a"), item("b")], (), answerability="supported")
    merged = merge_pool_audits(first, second, "zh")
    assert [row.obj.id for row in merged.accepted] == ["a", "b"]
    assert merged.coverage_gap == "" and merged.answerability == "supported"


@pytest.mark.parametrize("count,status,warning,expected_failure", [
    (3, "partially_supported", "RETRIEVAL_AUDIT_UNAVAILABLE", "RETRIEVAL_AUDIT_UNAVAILABLE"),
    (5, "unsupported", "RETRIEVAL_AUDIT_INVALID", "RETRIEVAL_AUDIT_INVALID"),
    (5, "supported", "RETRIEVAL_AUDIT_UNAVAILABLE", None),
    (3, "partially_supported", None, None),
])
def test_incomplete_recovery_is_service_failure_only_when_needed(monkeypatch, count, status, warning, expected_failure):
    generator = ExhibitionGenerator(Settings(), SimpleNamespace(), SimpleNamespace())
    async def impl(*args, **kwargs):
        return AgenticRetrievalOutcome(results=[item(str(i)) for i in range(count)],
                                       answerability=status, warning_code=warning,
                                       warning_detail="Review service interrupted")
    monkeypatch.setattr(generator, "_agentic_retrieve_impl", impl)
    agenda = AgendaInput(question="A bounded fixture", priorKnowledge="none", durationMinutes=5, collectionId="fixture")
    outcome = asyncio.run(generator._agentic_retrieve(agenda, SimpleNamespace(), [], required_count=5))
    assert outcome.failure_code == expected_failure
    if expected_failure:
        assert outcome.coverage_gap == "Review service interrupted"
    assert len(outcome.results) == count
