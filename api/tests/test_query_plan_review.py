from __future__ import annotations

from dataclasses import replace
import asyncio
from types import SimpleNamespace

import pytest

from app.query_plan_review import (
    parse_query_plan_review,
    query_plan_document,
    query_plan_review_payload,
    legacy_query_plan_review_payload,
    legacy_query_plan_review_prompt as query_plan_review_prompt,
    unreviewed_query_plan,
)
from app.retrieval_agent import RetrievalQueryPlan
from app.retrieval_filters import FilterSpec
from app.config import Settings
from app.generator import ExhibitionGenerator
from app.models import AgendaInput
from app.providers.deepseek import DeepSeekProvider, ProviderError


def _draft(**changes):
    plan = RetrievalQueryPlan(
        valid=True, in_collection_scope=True, search_queries=("geometric motif",),
        semantic_query="geometric motifs on museum objects",
        mandatory_predicates=("depicts a geometric motif",),
        interpretation="观察几何图案", reason="视觉观察",
        evidence_mode="visual_observation", visual_predicate_ids=("p1",),
    )
    return replace(plan, **changes)


def _checked(plan, quote="几何图案", *, scope="mandatory", target_scope=None, requirement=None):
    if target_scope is None:
        target_scope = "preference" if scope == "optional" else "per_object"
    if requirement is None:
        requirement = (plan.get("mandatoryPerObjectPredicates") or ["depicts a geometric motif"])[0]
    return {"intentChecks": [{"sourceQuote": quote, "scope": scope, "targetScope": target_scope,
                              "requirement": requirement}], "plan": plan}


def test_legacy_review_payload_only_contains_original_question_and_typed_draft():
    plan = _draft()
    payload = legacy_query_plan_review_payload("我想看衣服上的几何图案", plan)
    assert set(payload) == {"visitorQuestion", "language", "draftPlan"}
    assert payload["visitorQuestion"] == "我想看衣服上的几何图案"
    assert set(payload["draftPlan"]) == set(query_plan_document(plan))
    assert not {"candidates", "objectIds", "qrels", "gold", "images", "collection"} & set(payload)
    payload["draftPlan"]["mandatoryPerObjectPredicates"].append("mutated caller list")
    assert plan.mandatory_predicates == ("depicts a geometric motif",)


def test_faithful_draft_can_be_confirmed_without_rewriting():
    draft = _draft()
    result = parse_query_plan_review(_checked(query_plan_document(draft)), question="我想看几何图案", draft=draft)
    assert result.reviewed and not result.changed
    assert result.plan == draft
    assert result.diagnostics["status"] == "confirmed"
    assert result.diagnostics["semanticFaithfulnessProven"] is False


def test_carrier_relation_can_be_corrected_without_filter_invention():
    draft = _draft()
    corrected = query_plan_document(draft)
    corrected.update(
        mandatoryPerObjectPredicates=["a geometric motif appears on clothing, not just as a standalone drawing"],
        catalogueTypeHints=["clothing"],
        catalogueQueries=["garment geometric motif"],
        semanticQuery="garments carrying geometric motifs, not standalone motif drawings",
    )
    result = parse_query_plan_review(_checked(corrected, "衣服上的几何图案"), question="我想看衣服上的几何图案，不要单独的图案素描", draft=draft)
    assert result.reviewed and result.changed
    assert result.plan.catalogue_type_hints == ("clothing",)
    assert "on clothing" in result.plan.mandatory_predicates[0]
    assert result.plan.filters == FilterSpec()
    assert "mandatoryPerObjectPredicates" in result.diagnostics["changedFields"]


def test_known_unknown_partition_can_replace_an_over_strict_predicate():
    draft = _draft(evidence_mode="record_explanation", visual_predicate_ids=(),
                   mandatory_predicates=("record identifies the original recipient",),
                   catalogue_type_hints=("gift",))
    corrected = query_plan_document(draft)
    corrected.update(
        mandatoryPerObjectPredicates=["institution record establishes an object as a gift"],
        poolCoverageLegs=["recipient recorded", "recipient not recorded or uncertain"],
        selectionRationaleConstraints=["do not invent a recipient when the record is silent"],
        catalogueQueries=["gift recipient", "gift unidentified recipient"],
        semanticQuery="gifts with known or unknown recipients according to institution records",
    )
    result = parse_query_plan_review(_checked(corrected, "哪些知道送给谁，哪些不知道", scope="alternative", requirement="compare gifts with recorded or unknown recipients"), question="赠礼中哪些知道送给谁，哪些不知道？", draft=draft)
    assert result.reviewed and result.changed
    assert len(result.plan.mandatory_predicates) == 1
    assert "recipient not recorded or uncertain" in result.plan.pool_coverage_legs
    assert result.plan.visual_predicate_ids == ()


def test_voluntary_attention_can_be_preserved_without_universal_admission_condition():
    draft = _draft(mandatory_predicates=("depicts people gathering",))
    corrected = query_plan_document(draft)
    corrected["selectionRationaleConstraints"] = ["prioritize visible exchanges of gestures between people"]
    result = parse_query_plan_review(_checked(corrected, "留意互相的手势", scope="optional", requirement="prioritize visible gesture exchanges"), question="想看人们聚在一起，留意互相的手势", draft=draft)
    assert result.reviewed
    assert result.plan.mandatory_predicates == draft.mandatory_predicates
    assert result.plan.selection_constraints


@pytest.mark.parametrize("field,value", [
    ("cultures", ["China"]), ("materials", ["bronze"]), ("dateStart", 1700),
    ("objectTypes", ["painting"]), ("evidenceDepth", ["full"]), ("imageRequired", True),
])
def test_review_cannot_invent_hard_filters_even_if_parser_would_turn_type_into_hint(field, value):
    draft = _draft()
    output = query_plan_document(draft)
    output["hardFilters"][field] = value
    result = parse_query_plan_review(_checked(output), question="我想看几何图案", draft=draft)
    assert result.plan is draft
    assert not result.reviewed and not result.changed
    assert result.diagnostics["reason"] == "review_changed_locked_hard_filters"


def test_review_cannot_silently_remove_explicit_locked_culture():
    draft = _draft(filters=FilterSpec(cultures=("Japan",)))
    output = query_plan_document(draft)
    output["hardFilters"]["cultures"] = []
    result = parse_query_plan_review(_checked(output), question="我想看日本的几何图案", draft=draft)
    assert not result.reviewed and result.plan is draft


@pytest.mark.parametrize("edit", [
    lambda value: value.pop("mandatoryPerObjectPredicates"),
    lambda value: value.update(candidateIds=["unseen-object"]),
    lambda value: value.update(mandatoryPerObjectPredicates=[]),
    lambda value: value.update(poolCoverageLegs="not a list"),
    lambda value: value.update(visualPredicateIds=["p9"]),
    lambda value: value.update(evidenceMode="record_explanation", visualPredicateIds=["p1"]),
    lambda value: value.update(evidenceMode="anything_goes"),
])
def test_invalid_review_does_not_discard_draft_or_claim_success(edit):
    draft = _draft()
    output = query_plan_document(draft)
    edit(output)
    result = parse_query_plan_review(_checked(output), question="我想看几何图案", draft=draft)
    assert result.plan is draft
    assert not result.reviewed
    assert result.diagnostics["status"] == "unreviewed"


def test_timeout_fallback_is_observable_and_does_not_claim_confirmation():
    draft = _draft()
    result = unreviewed_query_plan(draft, "shared_planning_deadline")
    assert result.plan is draft
    assert result.reviewed is False
    assert result.diagnostics["reason"] == "shared_planning_deadline"


def test_invalid_draft_is_not_submitted_for_a_review():
    with pytest.raises(ValueError):
        query_plan_review_payload("问题", _draft(valid=False))


def test_prompt_covers_general_roles_without_old_topic_specific_examples():
    prompt = query_plan_review_prompt()
    for expected in ("ON/IN", "AND from OR", "known from unknown", "explicitly omissible",
                     "negation and exclusions", "Copy hardFilters EXACTLY", "Do not downgrade", "No collection"):
        assert expected in prompt
    for old_topic in ("butterfly", "incense", "market", "cat", "dog"):
        # Check whole tokens: "catalogue" is not a topic-specific cat rule.
        import re
        assert not re.search(rf"\b{old_topic}\b", prompt, re.IGNORECASE)
    assert prompt.index("FIRST read visitorQuestion independently") < prompt.index("THEN cross-check draftPlan")
    assert "intentChecks" in prompt and "EXACT contiguous substring" in prompt
    assert "If the draft is already faithful, copy it" not in prompt


def test_legacy_bare_plan_is_explicitly_unreviewed_not_silently_accepted():
    draft = _draft()
    result = parse_query_plan_review(query_plan_document(draft), question="我想看几何图案", draft=draft)
    assert result.plan is draft and not result.reviewed
    assert result.diagnostics["reason"] == "invalid_review_schema"
    assert result.diagnostics["intentChecks"] == []


@pytest.mark.parametrize("quote", ["", " ", "衣服上的图案", "geometric motif", "几何…图案", " 几何图案 "])
def test_source_quote_must_be_nonempty_exact_original_substring(quote):
    draft = _draft()
    result = parse_query_plan_review(_checked(query_plan_document(draft), quote), question="我想看几何图案", draft=draft)
    assert not result.reviewed and result.plan is draft
    assert result.diagnostics["reason"] == "unbound_intent_source_quote"


@pytest.mark.parametrize("edit,reason", [
    (lambda value: value.pop("intentChecks"), "invalid_review_schema"),
    (lambda value: value.update(intentChecks=[]), "invalid_intent_checks"),
    (lambda value: value.update(intentChecks=value["intentChecks"] * 7), "invalid_intent_checks"),
    (lambda value: value.update(intentChecks=value["intentChecks"] * 2), "duplicate_intent_check"),
    (lambda value: value["intentChecks"][0].pop("sourceQuote"), "invalid_intent_check_schema"),
    (lambda value: value["intentChecks"][0].update(score=1), "invalid_intent_check_schema"),
    (lambda value: value["intentChecks"][0].update(scope="mandatory|optional"), "invalid_intent_scope"),
    (lambda value: value["intentChecks"][0].update(scope=[]), "invalid_intent_scope"),
    (lambda value: value["intentChecks"][0].pop("targetScope"), "invalid_intent_check_schema"),
    (lambda value: value["intentChecks"][0].update(targetScope=[]), "invalid_intent_target_scope"),
    (lambda value: value["intentChecks"][0].update(targetScope="mandatory"), "invalid_intent_target_scope"),
    (lambda value: value["intentChecks"][0].update(requirement=""), "invalid_intent_requirement"),
    (lambda value: value["intentChecks"][0].update(requirement="x" * 241), "invalid_intent_requirement"),
    (lambda value: value.update(plan={}), "invalid_review_plan_schema"),
])
def test_invalid_intent_contract_keeps_draft_and_records_specific_failure(edit, reason):
    draft = _draft()
    output = _checked(query_plan_document(draft))
    edit(output)
    result = parse_query_plan_review(output, question="我想看几何图案", draft=draft)
    assert result.plan is draft and not result.reviewed
    assert result.diagnostics["reason"] == reason


def test_all_four_scope_types_are_preserved_as_source_bound_diagnostics():
    question = "看衣服上的几何图案，圆形或三角形都行，想留意线条，不要单幅素描"
    draft = _draft()
    output = _checked(query_plan_document(draft))
    output["intentChecks"] = [
        {"sourceQuote": "衣服上的几何图案", "scope": "mandatory", "targetScope": "per_object",
         "requirement": "a circle or triangle motif on clothing, not a standalone drawing"},
        {"sourceQuote": "圆形或三角形都行", "scope": "alternative", "targetScope": "per_object", "requirement": "circle OR triangle"},
        {"sourceQuote": "想留意线条", "scope": "optional", "targetScope": "preference", "requirement": "prioritize line details"},
        {"sourceQuote": "不要单幅素描", "scope": "exclusion", "targetScope": "per_object", "requirement": "exclude standalone drawings"},
    ]
    output["plan"].update(
        mandatoryPerObjectPredicates=["a circle or triangle motif on clothing, not a standalone drawing"],
        selectionRationaleConstraints=["prioritize line details"],
        semanticQuery="clothing carrying circle or triangle motifs rather than standalone drawings")
    result = parse_query_plan_review(output, question=question, draft=draft)
    assert result.reviewed and result.changed
    assert result.diagnostics["intentCheckCount"] == 4
    assert result.diagnostics["intentChecks"] == output["intentChecks"]
    assert result.diagnostics["sourceQuotesBound"] is True
    # The parser can prove quotation binding, not LLM reasoning or complete coverage.
    assert result.diagnostics["semanticFaithfulnessProven"] is False


def _initial_review_fixture(monkeypatch, *, elapsed=4.25, review_error=False, review_edit=None, deadline=150.0, repair_valid=False):
    clock = [100.0]
    calls = []
    monkeypatch.setattr("app.generator.perf_counter", lambda: clock[0])
    provider = SimpleNamespace(configured=True, supports_retrieval_audit=True,
                               supports_retrieval_query_planning=True)
    generator = ExhibitionGenerator(Settings(), SimpleNamespace(), provider)
    draft = _draft(filters=FilterSpec(cultures=("Japan",)))

    async def generate(prompt, payload, *, stage, timeout_seconds):
        calls.append({"stage": stage, "at": clock[0], "budget": timeout_seconds, "payload": payload})
        if stage.startswith("retrieval_plan:"):
            clock[0] += elapsed
            return query_plan_document(draft)
        assert stage.startswith("retrieval_plan_review")
        if review_error:
            clock[0] += timeout_seconds
            raise ProviderError("fixture review timed out")
        clock[0] += min(1.0, timeout_seconds)
        # A fake provider owns its output fixture. Production no longer exposes
        # the draft's old conditions or interpretation to the actual reviewer.
        assert "draftPlan" not in payload and "recallDraft" in payload
        output = query_plan_document(draft)
        if review_edit and not (stage.endswith("_repair") and repair_valid):
            review_edit(output)
        return _checked(output)

    async def search(agenda, collection, *, deadline, filters):
        calls.append({"stage": "search", "at": clock[0], "deadline": deadline, "filters": filters})
        return []

    monkeypatch.setattr(generator, "_generate_model_json", generate)
    monkeypatch.setattr(generator, "_search_async", search)
    agenda = AgendaInput(question="我想看日本衣服上的几何图案", priorKnowledge="none", durationMinutes=5, collectionId="fixture")
    from app.collections import CollectionDataError
    try:
        result = asyncio.run(generator.prepare_initial_retrieval(
            agenda, SimpleNamespace(concept_aliases={}), deadline=deadline))
    except CollectionDataError as error:
        assert error.code == "RETRIEVAL_PLAN_UNAVAILABLE"
        result = SimpleNamespace(query_plan=None, error=error,
                                 diagnostics=error.details["planningDiagnostics"], completed_at=clock[0])
    return result, calls, draft


def test_initial_review_and_draft_share_sixteen_second_planning_deadline(monkeypatch):
    result, calls, draft = _initial_review_fixture(monkeypatch)
    assert [call["stage"] for call in calls] == ["retrieval_plan:1", "retrieval_plan_review", "search"]
    assert calls[0]["budget"] == 8.0
    assert calls[1]["budget"] == 11.75
    assert calls[1]["at"] + calls[1]["budget"] == 116.0
    assert calls[-1]["at"] <= 116.0
    assert set(calls[1]["payload"]) == {"visitorQuestion", "language", "recallDraft"}
    assert result.query_plan == draft
    assert result.diagnostics["planningReview"]["status"] == "confirmed"
    assert result.diagnostics["planningAttempts"] == result.diagnostics["planningReviewAttempts"] == 1


def test_review_timeout_stops_before_recall_and_cannot_claim_review_success(monkeypatch):
    result, calls, draft = _initial_review_fixture(monkeypatch, review_error=True)
    assert result.query_plan is None
    assert result.diagnostics["planningStatus"] == "applied"
    assert result.diagnostics["queryPlanApplied"] is False
    assert result.diagnostics["planningReview"]["status"] == "unreviewed"
    assert result.diagnostics["planningReview"]["semanticFaithfulnessProven"] is False
    assert result.completed_at == 116.0
    assert not any(call["stage"] == "search" for call in calls)


def test_review_no_extra_call_when_shared_planning_time_is_spent(monkeypatch):
    # Simulates dispatch/parse overhead after draft completion: no new sixteen-second budget.
    result, calls, draft = _initial_review_fixture(monkeypatch, elapsed=15.6)
    assert [call["stage"] for call in calls] == ["retrieval_plan:1"]
    assert result.query_plan is None
    assert result.diagnostics["planningReviewAttempts"] == 0
    assert result.diagnostics["planningReview"]["reason"] == "shared_planning_deadline"


def test_review_correction_reaches_first_recall_plan(monkeypatch):
    result, _, _ = _initial_review_fixture(monkeypatch, review_edit=lambda output: output.update(
        mandatoryPerObjectPredicates=["a geometric motif appears on clothing"],
        catalogueTypeHints=["garment"], semanticQuery="Japanese clothing with geometric motifs"))
    assert result.diagnostics["planningReview"]["status"] == "corrected"
    assert result.query_plan.catalogue_type_hints == ("garment",)
    assert "on clothing" in result.query_plan.mandatory_predicates[0]


def test_invalid_review_cannot_apply_changed_filters_or_begin_search(monkeypatch):
    result, calls, draft = _initial_review_fixture(monkeypatch, review_edit=lambda output: output["hardFilters"].update(materials=["gold"]))
    assert result.query_plan is None
    assert result.diagnostics["planningReview"]["status"] == "unreviewed"
    assert not any(call["stage"] == "search" for call in calls)


def test_structural_review_repair_is_bounded_once_and_can_restore_valid_contract(monkeypatch):
    result, calls, draft = _initial_review_fixture(
        monkeypatch, repair_valid=True,
        review_edit=lambda output: output["hardFilters"].update(materials=["gold"]))
    assert [call["stage"] for call in calls] == [
        "retrieval_plan:1", "retrieval_plan_review", "retrieval_plan_review_repair", "search"]
    repair = calls[2]
    assert repair["budget"] <= 5 and repair["at"] + repair["budget"] <= 116.0
    assert repair["payload"]["contractFailure"] == "review_changed_locked_hard_filters"
    assert "candidates" not in repair["payload"]
    assert result.query_plan == draft and result.diagnostics["planningReview"]["status"] == "confirmed"
    assert result.diagnostics["planningReviewAttempts"] == 2
    assert result.diagnostics["planningReviewInitial"]["status"] == "unreviewed"


def test_bad_repair_does_not_loop_or_silently_pass(monkeypatch):
    result, calls, draft = _initial_review_fixture(
        monkeypatch, review_edit=lambda output: output["hardFilters"].update(materials=["gold"]))
    assert len([call for call in calls if call["stage"].startswith("retrieval_plan_review")]) == 2
    assert result.query_plan is None
    assert result.diagnostics["planningReview"]["status"] == "unreviewed"
    assert result.diagnostics["planningReviewAttempts"] == 2


def test_planning_consumes_original_seventy_second_retrieval_budget(monkeypatch):
    result, calls, _ = _initial_review_fixture(monkeypatch, review_error=True, deadline=170.0)
    assert calls[0]["at"] == 100.0 and calls[0]["budget"] == 8.0
    assert calls[1]["at"] + calls[1]["budget"] == 116.0
    assert result.completed_at == 116.0 and result.completed_at < 170.0
    assert not any(call["stage"] == "search" for call in calls)
    assert result.diagnostics["planningReview"]["status"] == "unreviewed"


def test_short_retrieval_deadline_shrinks_planning_and_preserves_audit_reserve(monkeypatch):
    settings = Settings()
    audit_floor = min(float(settings.rag_llm_audit_timeout_seconds),
                      max(10.0, settings.rag_llm_audit_timeout_seconds * 0.75))
    original_deadline = 100.0 + audit_floor + 6.0
    result, calls, _ = _initial_review_fixture(monkeypatch, review_error=True, deadline=original_deadline)
    assert calls[0]["budget"] == 6.0
    assert calls[1]["budget"] == 1.75
    assert result.completed_at == 106.0
    assert original_deadline - result.completed_at == audit_floor
    assert not any(call["stage"] == "search" for call in calls)


@pytest.mark.parametrize("payload,expected", [
    ({"visitorQuestion": "fixture"}, 1200),
    ({"visitorQuestion": "fixture", "draftPlan": {}}, 2400),
    ({"visitorQuestion": "fixture", "recallDraft": {}}, 2400),
])
def test_only_review_receives_expanded_output_allowance(monkeypatch, payload, expected):
    provider = DeepSeekProvider(Settings())
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return {"fixture": True}

    monkeypatch.setattr(provider, "_generate_json", generate)
    assert asyncio.run(provider.generate_retrieval_query_plan_json("prompt", payload)) == {"fixture": True}
    assert calls[0]["max_tokens"] == expected
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["thinking"] == {"type": "disabled"}


def test_review_reuses_deterministic_text_provider_route():
    calls = []

    class Provider:
        async def generate_retrieval_query_plan_json(self, prompt, payload):
            calls.append("deterministic_text")
            return {"ok": True}

        async def generate_json(self, prompt, payload):
            raise AssertionError("review must not use creative generator route")

    generator = ExhibitionGenerator(Settings(), SimpleNamespace(), Provider())
    output = asyncio.run(generator._generate_model_json("prompt", {}, stage="retrieval_plan_review", timeout_seconds=0.1))
    assert output == {"ok": True} and calls == ["deterministic_text"]


def test_actual_review_call_has_wall_clock_cancellation():
    state = []

    class Provider:
        async def generate_retrieval_query_plan_json(self, prompt, payload):
            try:
                await asyncio.sleep(1)
            finally:
                state.append("cancelled")

    generator = ExhibitionGenerator(Settings(), SimpleNamespace(), Provider())
    with pytest.raises(ProviderError):
        asyncio.run(generator._generate_model_json("prompt", {}, stage="retrieval_plan_review", timeout_seconds=0.01))
    assert state == ["cancelled"]
