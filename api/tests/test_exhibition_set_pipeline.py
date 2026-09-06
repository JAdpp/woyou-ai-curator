"""Real agentic retrieval entry, fake async provider, no threads or network.

These tests complement pure aggregation tests: model assertions flow through
the actual payload, strict parser, retrieval wrapper and selected-set guard.
"""

import asyncio
from types import SimpleNamespace
from time import perf_counter

import pytest

from app.collections import CollectionDataError, SearchResult
from app.config import Settings
from app.generator import ExhibitionGenerator
from app.models import AgendaInput, EvidenceChunk, MuseumObject
from app.retrieval_agent import CONDITION_CONTRACT_VERSION, parse_query_plan
from app.set_coverage import persisted_set_coverage_valid


QUESTION = "我想看信件题材，留意一个人把信递给另一个人的场面"
SOURCE_QUOTE = "一个人把信递给另一个人的场面"


def requirement():
    return {"id": "s1", "text": "One person hands a letter to another person in the depicted scene",
            "sourceQuote": SOURCE_QUOTE, "minWitnesses": 1,
            "quantifierOrigin": "product_default", "evidenceScope": "institution_record"}


def make_plan():
    plan = parse_query_plan({
        "inCollectionScope": True, "catalogueQueries": ["letter handover"],
        "evidenceMode": "record_explanation",
        "mandatoryPerObjectPredicates": ["The institution identifies a letter scene"],
        "exhibitionSetRequirements": [requirement()],
    }, question=QUESTION, max_queries=3)
    assert plan.valid
    return plan


def make_pool(count=5, *, core_id="object-4"):
    results = []
    for index in range(count):
        key = f"object-{index}"
        description = f"This institution identifies letter scene {index}. " + (
            "One person hands a letter to another person in the scene."
            if key == core_id else "A lone figure holds a folded letter beside a desk."
        )
        obj = MuseumObject(
            id=key, title=f"Letter scene {index}", description=description,
            culture="China", institution="Fixture Museum", rights="CC0", evidenceDepth="full",
            imageUrl=f"https://museum.test/{key}.jpg", objectUrl=f"https://museum.test/{key}",
            evidence=[EvidenceChunk(id=f"{key}:text", text=description,
                                    sourceUrl=f"https://museum.test/{key}", sourceTitle="Institution record",
                                    sourceKind="institution_description")],
        )
        results.append(SearchResult(obj=obj, score=100-index,
                                    matched_evidence_ids=(f"{key}:text",), retrieval_sources=("bm25",)))
    return results


class NoIORepository:
    def match_question_policy(self, collection, question):
        return None

    def search(self, *args, **kwargs):
        raise AssertionError("This fixture must not perform catalogue I/O")

    def search_many(self, *args, **kwargs):
        raise AssertionError("This fixture must not perform catalogue I/O")


class AsyncAuditProvider:
    configured = True
    supports_retrieval_audit = True
    supports_retrieval_query_planning = True

    def __init__(self, witness_ids=(), *, bad_object_ids=(), borrowed_from=None):
        self.witness_ids = set(witness_ids)
        self.bad_object_ids = set(bad_object_ids)
        self.borrowed_from = borrowed_from
        self.payloads = []

    async def generate_retrieval_query_plan_json(self, *args):
        raise AssertionError("A supplied, validated plan must not be planned again")

    async def generate_json(self, *args):
        raise AssertionError("No extra generator or semantic critic call is authorized here")

    async def generate_retrieval_audit_json(self, prompt, payload):
        self.payloads.append(payload)
        assert payload["retrievalContract"]["conditionContractVersion"] == CONDITION_CONTRACT_VERSION
        assert [row["id"] for row in payload["retrievalContract"]["perObjectConditions"]] == ["p1"]
        assert [row["id"] for row in payload["retrievalContract"]["exhibitionSetConditions"]] == ["s1"]
        decisions = []
        for candidate in payload["candidates"]:
            key = candidate["objectId"]
            evidence = candidate["evidence"][0]
            set_supported = key in self.witness_ids
            set_source = self.borrowed_from if self.borrowed_from and set_supported else evidence
            decisions.append({
                "objectId": key, "relevanceScore": 0.95, "evidenceIds": [evidence["id"]],
                "conditionEvidence": [{"conditionId": "p1",
                    "status": "unknown" if key in self.bad_object_ids else "supported",
                    "evidenceId": evidence["id"], "supportingQuote": evidence["text"], "relation": "exact"}],
                "setConditionEvidence": [{"conditionId": "s1",
                    "status": "supported" if set_supported else "unknown",
                    "evidenceId": set_source["id"] if set_supported else "",
                    "supportingQuote": set_source["text"] if set_supported else "",
                    "relation": "exact" if set_supported else "unknown"}],
            })
        # Deliberately optimistic overall answerability: the actual wrapper,
        # not this fixture, must enforce missing set coverage and object count.
        return {"conditionContractVersion": CONDITION_CONTRACT_VERSION,
                "accepted": decisions, "answerability": "supported",
                "searchQueries": [], "expansionReason": "none", "coverageGap": ""}


class RecordingGenerator(ExhibitionGenerator):
    def __init__(self, provider):
        super().__init__(Settings(rag_agentic_max_queries=0, rag_visual_audit_enabled=False,
                                  rag_llm_audit_top_k=20, rag_rerank_enabled=False),
                         NoIORepository(), provider=provider)
        self.model_stages = []

    async def _generate_model_json(self, prompt, payload, *, stage, timeout_seconds, **kwargs):
        self.model_stages.append((stage, timeout_seconds))
        return await super()._generate_model_json(prompt, payload, stage=stage,
                                                 timeout_seconds=timeout_seconds, **kwargs)

    @staticmethod
    async def _run_retrieval_work(*args, **kwargs):
        raise AssertionError("The integration fixture must never create worker threads")


def run(pool, provider, *, question=QUESTION, deadline_seconds=70):
    generator = RecordingGenerator(provider)
    agenda = AgendaInput(question=question, priorKnowledge="none", durationMinutes=5)
    plan = make_plan()
    outcome = asyncio.run(generator._agentic_retrieve(
        agenda, SimpleNamespace(concept_aliases={}), pool, required_count=5,
        deadline=perf_counter()+deadline_seconds, initial_query_plan=plan, planning_attempted=True,
    ))
    return generator, agenda, outcome


def test_unknown_set_goal_keeps_five_related_objects_but_wrapper_rejects_answerability():
    provider = AsyncAuditProvider()
    generator, agenda, outcome = run(make_pool(), provider)
    assert len(outcome.results) == 5 and outcome.audit_applied and not outcome.failure_code
    assert outcome.answerability == "unsupported"
    assert outcome.exhibition_set_requirements == (requirement(),)
    report = outcome.audit_diagnostics["auditedSetCoverage"]
    assert report["satisfied"] is False and report["missingRequirementIds"] == ["s1"]
    assert "不表示整个馆藏库都没有资料" in outcome.coverage_gap
    assert all(not result.set_witnesses for result in outcome.results)
    assert len(provider.payloads) == 1
    assert [stage for stage, _ in generator.model_stages] == ["retrieval_audit:1"]
    with pytest.raises(CollectionDataError) as caught:
        generator._ensure_final_set_coverage(agenda, [row.obj for row in outcome.results],
                                             outcome.results, outcome.exhibition_set_requirements)
    assert caught.value.code == "QUESTION_UNSUPPORTED_AFTER_AUDIT"


def test_one_complete_witness_passes_actual_parser_wrapper_selection_and_persistence():
    provider = AsyncAuditProvider({"object-4"})
    generator, agenda, outcome = run(make_pool(), provider)
    assert outcome.answerability == "supported" and len(outcome.results) == 5
    assert len(provider.payloads) == 1
    witnesses = [row for row in outcome.results if row.set_witnesses]
    assert [row.obj.id for row in witnesses] == ["object-4"]
    assert all("condition_source_bound" in row.retrieval_sources for row in outcome.results)
    selected, report = generator._ensure_final_set_coverage(
        agenda, [row.obj for row in outcome.results], outcome.results, outcome.exhibition_set_requirements)
    assert report["satisfied"] and report["scope"] == "final_selected_set"
    exhibition = SimpleNamespace(agenda=agenda, items=[SimpleNamespace(object=obj) for obj in selected],
                                  exhibition_set_coverage=report)
    assert persisted_set_coverage_valid(exhibition)


def test_set_witness_cannot_promote_an_object_that_fails_current_per_object_gate():
    provider = AsyncAuditProvider({"object-4"}, bad_object_ids={"object-4"})
    _, _, outcome = run(make_pool(), provider)
    assert len(outcome.results) == 4 and not any(row.obj.id == "object-4" for row in outcome.results)
    assert outcome.answerability == "unsupported"
    assert not outcome.audit_diagnostics["auditedSetCoverage"]["satisfied"]
    assert any(row["objectId"] == "object-4" and row["conditionId"] == "p1"
               for row in outcome.audit_diagnostics["pass1"]["rejections"])


def test_borrowed_candidate_quote_does_not_pass_the_real_set_coverage_wrapper():
    pool = make_pool()
    foreign = {"id": pool[4].obj.evidence[0].id, "text": pool[4].obj.evidence[0].text}
    provider = AsyncAuditProvider({"object-0"}, borrowed_from=foreign)
    _, _, outcome = run(pool, provider)
    assert len(outcome.results) == 5 and outcome.answerability == "unsupported"
    assert all(not row.set_witnesses for row in outcome.results)
    assert any(row.get("scope") == "exhibition_set" and row.get("failure") == "unbound_quote"
               for row in outcome.audit_diagnostics["pass1"]["conditions"])


@pytest.mark.parametrize("changed_question", [False, True])
def test_incoming_old_witness_never_survives_a_current_unknown_audit(changed_question):
    _, _, old = run(make_pool(), AsyncAuditProvider({"object-4"}))
    assert any(row.set_witnesses for row in old.results)
    question = QUESTION + "，仅采用明确记载" if changed_question else QUESTION
    provider = AsyncAuditProvider()
    _, _, current = run(old.results, provider, question=question)
    assert len(current.results) == 5 and current.answerability == "unsupported"
    assert all(not row.set_witnesses for row in current.results)
    assert any(row.set_witnesses for row in old.results)  # previous immutable result unchanged
    assert len(provider.payloads) == 1


def test_audited_pool_witness_is_not_enough_when_actual_itinerary_omits_it():
    generator, agenda, outcome = run(make_pool(6), AsyncAuditProvider({"object-4"}))
    assert outcome.answerability == "supported"
    selected = [row.obj for row in outcome.results if row.obj.id != "object-4"]
    assert len(selected) == 5
    with pytest.raises(CollectionDataError) as caught:
        generator._ensure_final_set_coverage(agenda, selected, outcome.results,
                                             outcome.exhibition_set_requirements, allow_repair=False)
    assert caught.value.details["exhibitionSetCoverage"]["missingRequirementIds"] == ["s1"]
    repaired, report = generator._ensure_final_set_coverage(
        agenda, selected, outcome.results, outcome.exhibition_set_requirements)
    assert len(repaired) == 5 and any(obj.id == "object-4" for obj in repaired)
    assert report["satisfied"]


def test_set_gap_recovers_only_through_existing_bounded_unseen_pool_audit():
    provider = AsyncAuditProvider({"object-23"})
    generator, _, outcome = run(make_pool(24, core_id="object-23"), provider)
    assert len(provider.payloads) == 2
    assert "object-23" not in {row["objectId"] for row in provider.payloads[0]["candidates"]}
    assert "object-23" in {row["objectId"] for row in provider.payloads[1]["candidates"]}
    assert outcome.answerability == "supported"
    assert outcome.audit_diagnostics["auditedSetCoverage"]["satisfied"]
    assert [stage for stage, _ in generator.model_stages] == ["retrieval_audit:1", "retrieval_audit:2"]
    assert all(0 < budget <= generator.settings.rag_llm_audit_timeout_seconds
               for _, budget in generator.model_stages)
    assert outcome.expanded_queries == ()


def test_exhausted_shared_deadline_grants_no_extra_set_model_call_or_grace_period():
    provider = AsyncAuditProvider({"object-4"})
    generator, _, outcome = run(make_pool(), provider, deadline_seconds=0.2)
    assert outcome.failure_code == "RETRIEVAL_AUDIT_UNAVAILABLE"
    assert provider.payloads == [] and generator.model_stages == []
    assert outcome.results == []


def test_revised_requirement_cannot_reuse_same_id_witness_as_a_new_condition():
    generator, agenda, outcome = run(make_pool(), AsyncAuditProvider({"object-4"}))
    changed = {**requirement(), "text": "Two people jointly read the same open letter"}
    with pytest.raises(CollectionDataError):
        generator._ensure_final_set_coverage(agenda, [row.obj for row in outcome.results],
                                             outcome.results, (changed,), allow_repair=False)
