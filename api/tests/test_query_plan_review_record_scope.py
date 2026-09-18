"""Record-mode reviews that declare a visible scope are narrowed, not discarded.

Fixtures are the reviewer outputs observed on 2026-09-18/19 for ordinary blue-
ceramics questions (trimmed recall wording, exact intents). Before the fix the
review and its repair both failed with invalid_v4_evidence_scope, so retrieval
refused with RETRIEVAL_PLAN_UNAVAILABLE instead of searching.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.generator import ExhibitionGenerator
from app.models import AgendaInput
from app.query_plan_review import parse_query_plan_review
from app.retrieval_agent import _condition_specs, parse_exhibition_set_requirements, parse_query_plan

FILTERS = {"dateStart": None, "dateEnd": None, "cultures": [], "institutions": [], "materials": [],
           "objectTypes": [], "imageRequired": None, "rightsAllowed": [], "evidenceDepth": []}
WHY_QUESTION = "相似的蓝，为什么会出现在中国瓷瓶、越南盘与伊朗陶器上？"
COMPARE_QUESTION = "中国瓷瓶、越南盘和伊朗陶器上都有相似的蓝色装饰，它们的材料、做法和图案有什么异同？"
LEGS = ["Chinese porcelain vase with blue decoration", "Vietnamese dish with blue decoration",
        "Iranian pottery with blue decoration"]


def _draft_output(set_quote):
    return {
        "inCollectionScope": True, "queryInterpretation": "比较三地陶瓷上的蓝色装饰",
        "catalogueQueries": ["blue decoration Chinese porcelain vase", "blue decoration Vietnamese dish",
                             "blue decoration Iranian pottery"],
        "semanticQuery": "similar blue decoration on Chinese porcelain vase, Vietnamese dish or Iranian pottery",
        "evidenceMode": "record_explanation", "catalogueTypeHints": ["porcelain vase", "dish", "pottery"],
        "mandatoryPerObjectPredicates": [
            "institution record connects the object to blue decoration and to its Chinese, Vietnamese or Iranian production context"],
        "visualPredicateIds": [], "poolCoverageLegs": list(LEGS),
        "exhibitionSetRequirements": [{
            "id": "s1", "text": "至少一件器物的机构记录说明其蓝色装饰", "sourceQuote": set_quote,
            "minWitnesses": 1, "quantifierOrigin": "product_default", "evidenceScope": "institution_record"}],
        "selectionRationaleConstraints": [], "hardFilters": deepcopy(FILTERS), "reason": "记录解释型比较问题",
    }


def _review_output(draft_output, intents):
    search = {key: deepcopy(draft_output[key]) for key in (
        "inCollectionScope", "queryInterpretation", "catalogueQueries", "semanticQuery", "evidenceMode",
        "catalogueTypeHints", "poolCoverageLegs", "hardFilters", "reason")}
    return {"searchPlan": search, "intents": deepcopy(intents)}


OBSERVED = {
    # The visible scope sits on a set witness ("相似的蓝").
    "why_similar": (WHY_QUESTION, "中国瓷瓶", [
        {"id": "i1", "sourceQuote": "中国瓷瓶、越南盘与伊朗陶器", "type": "admission",
         "text": "对象为中国瓷瓶、越南盘或伊朗陶器", "evidenceScope": "institution_record"},
        {"id": "i2", "sourceQuote": "相似的蓝", "type": "set_witness",
         "text": "同一对象上可见蓝色外观，且该蓝色与其他对象上的蓝色相似",
         "evidenceScope": "visible_features_or_record", "minWitnesses": 1, "quantifierOrigin": "product_default"},
        {"id": "i3", "sourceQuote": "为什么会出现在", "type": "editorial",
         "text": "对相似蓝色出现的原因给出记录解释，并区分已确认与尚不能确认的内容"},
    ]),
    # The visible scope sits on a per-object admission ("都有相似的蓝色装饰").
    "compare_making": (COMPARE_QUESTION, "中国瓷瓶、越南盘和伊朗陶器上都有相似的蓝色装饰", [
        {"id": "i1", "sourceQuote": "中国瓷瓶、越南盘和伊朗陶器", "type": "admission",
         "text": "对象为中国瓷瓶、越南盘或伊朗陶器", "evidenceScope": "institution_record"},
        {"id": "i2", "sourceQuote": "都有相似的蓝色装饰", "type": "admission",
         "text": "对象具有蓝色装饰", "evidenceScope": "visible_features_or_record"},
        {"id": "i3", "sourceQuote": "它们的材料、做法和图案有什么异同？", "type": "set_witness",
         "text": "同一对象可提供其蓝色装饰的材料、做法和图案三方面记录", "evidenceScope": "institution_record",
         "minWitnesses": 1, "quantifierOrigin": "product_default"},
        {"id": "i4", "sourceQuote": "它们的材料、做法和图案有什么异同？", "type": "editorial",
         "text": "对每件对象分别说明其蓝色装饰在材料、做法和图案上已知与未知的内容", "evidenceScope": "institution_record"},
    ]),
}


def _case(name):
    question, set_quote, intents = OBSERVED[name]
    draft_output = _draft_output(set_quote)
    draft = parse_query_plan(draft_output, question=question, max_queries=5)
    assert draft.valid and draft.evidence_mode == "record_explanation"
    return question, draft_output, draft, _review_output(draft_output, intents)


@pytest.mark.parametrize("name", sorted(OBSERVED))
def test_observed_record_review_with_visible_scope_compiles_to_record_only_plan(name):
    question, _, draft, output = _case(name)
    snapshot = deepcopy(output)
    result = parse_query_plan_review(output, question=question, draft=draft)
    assert result.reviewed, result.diagnostics.get("reason")
    assert output == snapshot
    plan = result.plan
    assert plan.evidence_mode == "record_explanation"
    assert plan.visual_predicate_ids == ()
    visible = [row for row in output["intents"] if row.get("evidenceScope") == "visible_features_or_record"]
    assert [row["intentId"] for row in result.diagnostics["evidenceScopeNarrowings"]] == [row["id"] for row in visible]
    # Every visitor condition survives; none was dropped to satisfy the schema.
    admissions = [row["text"] for row in output["intents"] if row["type"] == "admission"]
    witnesses = [row["text"] for row in output["intents"] if row["type"] == "set_witness"]
    assert list(plan.mandatory_predicates) == admissions
    assert [row["text"] for row in plan.exhibition_set_requirements] == witnesses
    # The audit's own record-mode validators accept the compiled plan and give it no image authority.
    assert parse_exhibition_set_requirements(
        list(plan.exhibition_set_requirements), question=question, evidence_mode=plan.evidence_mode,
    ) == plan.exhibition_set_requirements
    conditions = _condition_specs(plan.mandatory_predicates, plan.catalogue_type_hints, (),
                                  evidence_mode=plan.evidence_mode, visual_predicate_ids=plan.visual_predicate_ids)
    assert {row["evidenceScope"] for row in conditions} == {"institution_record"}


@pytest.mark.parametrize("mode", ["visual_observation", "open_exploration"])
def test_visual_modes_keep_declared_visible_scope(mode):
    question, draft_output, _, output = _case("compare_making")
    draft_output["evidenceMode"] = mode
    draft = parse_query_plan(draft_output, question=question, max_queries=5)
    output["searchPlan"]["evidenceMode"] = mode
    result = parse_query_plan_review(output, question=question, draft=draft)
    assert result.reviewed
    assert result.plan.visual_predicate_ids == ("p2",)
    assert result.diagnostics["evidenceScopeNarrowings"] == []


@pytest.mark.parametrize("edit,reason", [
    (lambda intents: intents[1].update(evidenceScope="image_only"), "invalid_v4_evidence_scope"),
    (lambda intents: intents[1].pop("evidenceScope"), "invalid_v4_intent_schema"),
    (lambda intents: intents[1].update(sourceQuote="相似的蓝釉"), "unbound_v4_intent_source_quote"),
    (lambda intents: [intents.pop(0) for _ in range(2)], "missing_v4_admission"),
])
def test_record_mode_narrowing_does_not_rescue_genuinely_invalid_reviews(edit, reason):
    question, _, draft, output = _case("compare_making")
    edit(output["intents"])
    result = parse_query_plan_review(output, question=question, draft=draft)
    assert not result.reviewed and result.plan is draft
    assert result.diagnostics["reason"] == reason


def test_blue_comparison_reaches_recall_without_repair_or_plan_unavailable(monkeypatch):
    question, draft_output, _, review_output = _case("compare_making")
    clock = [100.0]
    stages = []
    monkeypatch.setattr("app.generator.perf_counter", lambda: clock[0])
    provider = SimpleNamespace(configured=True, supports_retrieval_audit=True,
                               supports_retrieval_query_planning=True)
    generator = ExhibitionGenerator(Settings(), SimpleNamespace(), provider)

    async def generate(prompt, payload, *, stage, timeout_seconds):
        stages.append(stage)
        clock[0] += 2.0
        return deepcopy(draft_output if stage.startswith("retrieval_plan:") else review_output)

    async def search(agenda, collection, *, deadline, filters):
        stages.append("search")
        return []

    monkeypatch.setattr(generator, "_generate_model_json", generate)
    monkeypatch.setattr(generator, "_search_async", search)
    agenda = AgendaInput(question=question, priorKnowledge="some", durationMinutes=5, collectionId="global_open")
    result = asyncio.run(generator.prepare_initial_retrieval(
        agenda, SimpleNamespace(concept_aliases={}), deadline=170.0))
    assert stages == ["retrieval_plan:1", "retrieval_plan_review", "search"]
    assert result.diagnostics["queryPlanApplied"] is True
    assert result.diagnostics["planningReview"]["status"] == "corrected"
    assert result.diagnostics["planningReview"]["evidenceScopeNarrowings"][0]["intentId"] == "i2"
    assert result.query_plan.evidence_mode == "record_explanation"
    assert result.query_plan.mandatory_predicates == ("对象为中国瓷瓶、越南盘或伊朗陶器", "对象具有蓝色装饰")
