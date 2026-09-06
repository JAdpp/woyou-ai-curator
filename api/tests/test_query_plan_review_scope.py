"""Declared intent scope is a contract, not a proof of natural-language meaning."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from app.query_plan_review import (
    LEGACY_QUERY_PLAN_REVIEW_VERSION as QUERY_PLAN_REVIEW_VERSION,
    parse_query_plan_review,
    query_plan_document,
    legacy_query_plan_review_payload as query_plan_review_payload,
    legacy_query_plan_review_prompt as query_plan_review_prompt,
)
from app.retrieval_agent import RetrievalQueryPlan, query_plan_prompt


QUESTION = "给我看一些人物聚集的画面，我想看看他们怎样互相递东西"
CLASS_QUOTE = "给我看一些人物聚集的画面"
GOAL_QUOTE = "我想看看他们怎样互相递东西"
CLASS_PREDICATE = "depicts people gathering"
GOAL = "one depicted scene shows people handing an object to each other"


def _fixture(*, question=QUESTION, goal_quote=GOAL_QUOTE, minimum=1,
             origin="product_default", evidence_scope="visible_features_or_record"):
    draft = RetrievalQueryPlan(
        valid=True, in_collection_scope=True, search_queries=("people gathering",),
        semantic_query="people gathering and handing objects to each other",
        mandatory_predicates=(CLASS_PREDICATE,), interpretation="观察人物聚集的画面",
        reason="视觉观察", evidence_mode="visual_observation", visual_predicate_ids=("p1",),
    )
    plan = query_plan_document(draft)
    plan["exhibitionSetRequirements"] = [{
        "id": "s1", "text": GOAL, "sourceQuote": goal_quote,
        "minWitnesses": minimum, "quantifierOrigin": origin, "evidenceScope": evidence_scope,
    }]
    checks = [
        {"sourceQuote": CLASS_QUOTE, "scope": "mandatory", "targetScope": "per_object",
         "requirement": CLASS_PREDICATE},
        {"sourceQuote": goal_quote, "scope": "mandatory", "targetScope": "exhibition_set",
         "requirement": GOAL},
    ]
    return question, draft, {"intentChecks": checks, "plan": plan}


def _parse(fixture):
    question, draft, output = fixture
    return parse_query_plan_review(output, question=question, draft=draft)


def test_required_whole_exhibition_goal_is_not_a_universal_object_condition():
    fixture = _fixture()
    result = _parse(fixture)
    assert result.reviewed and result.changed
    assert result.plan.mandatory_predicates == (CLASS_PREDICATE,)
    assert result.plan.exhibition_set_requirements == tuple(fixture[2]["plan"]["exhibitionSetRequirements"])
    assert result.plan.exhibition_set_requirements[0]["minWitnesses"] == 1
    assert result.plan.exhibition_set_requirements[0]["quantifierOrigin"] == "product_default"
    assert result.diagnostics["version"] == QUERY_PLAN_REVIEW_VERSION == "query-plan-fidelity-review-v3"
    assert result.diagnostics["changedFields"] == ["exhibitionSetRequirements"]
    assert result.diagnostics["declaredTargetScopesConsistent"] is True
    assert result.diagnostics["semanticFaithfulnessProven"] is False


@pytest.mark.parametrize("carrier", ["semanticQuery", "poolCoverageLegs", "selectionRationaleConstraints"])
def test_required_set_goal_cannot_only_live_in_search_or_preference(carrier):
    fixture = _fixture()
    output = fixture[2]
    output["plan"]["exhibitionSetRequirements"] = []
    output["plan"][carrier] = GOAL if carrier == "semanticQuery" else [GOAL]
    result = _parse(fixture)
    assert not result.reviewed and result.plan is fixture[1]
    assert result.diagnostics["reason"] == "exhibition_set_intent_not_enforced"


def test_set_requirement_cannot_be_promoted_to_the_same_per_object_predicate():
    fixture = _fixture()
    fixture[2]["plan"]["mandatoryPerObjectPredicates"].append(GOAL)
    result = _parse(fixture)
    assert not result.reviewed
    assert result.diagnostics["reason"] == "set_requirement_also_forced_per_object"


def test_declared_every_object_restriction_cannot_be_lowered_to_a_set_witness():
    quote = "每件都要有人互相递东西"
    fixture = _fixture(question=CLASS_QUOTE + "，" + quote, goal_quote=quote)
    # A set declaration does not discharge a second, explicit per-object check.
    fixture[2]["intentChecks"].append({
        "sourceQuote": quote, "scope": "mandatory", "targetScope": "per_object", "requirement": GOAL,
    })
    result = _parse(fixture)
    assert not result.reviewed and result.plan is fixture[1]
    assert result.diagnostics["reason"] == "per_object_intent_not_enforced"


def test_explicit_per_object_restriction_remains_per_object_when_properly_mapped():
    quote = "每件都要有人互相递东西"
    fixture = _fixture(question=CLASS_QUOTE + "，" + quote, goal_quote=quote)
    fixture[2]["plan"]["exhibitionSetRequirements"] = []
    fixture[2]["plan"]["mandatoryPerObjectPredicates"].append(GOAL)
    fixture[2]["intentChecks"][1]["targetScope"] = "per_object"
    result = _parse(fixture)
    assert result.reviewed
    assert result.plan.mandatory_predicates == (CLASS_PREDICATE, GOAL)
    assert result.plan.exhibition_set_requirements == ()


def test_required_intent_cannot_be_declared_a_preference():
    fixture = _fixture()
    fixture[2]["plan"]["exhibitionSetRequirements"] = []
    fixture[2]["intentChecks"][1]["targetScope"] = "preference"
    result = _parse(fixture)
    assert not result.reviewed
    assert result.diagnostics["reason"] == "mandatory_intent_downgraded_to_preference"


def test_optional_attention_cannot_authorize_a_required_set_goal():
    fixture = _fixture()
    fixture[2]["intentChecks"][1]["scope"] = "optional"
    result = _parse(fixture)
    assert not result.reviewed
    assert result.diagnostics["reason"] == "optional_intent_not_a_preference"


@pytest.mark.parametrize("edit", [
    lambda value: value["intentChecks"].pop(),
    lambda value: value["intentChecks"][1].update(requirement="a different relation"),
    lambda value: value["intentChecks"][1].update(sourceQuote="他们怎样互相递东西"),
    lambda value: value["intentChecks"][1].update(targetScope="per_object"),
])
def test_set_goals_require_a_matching_source_and_scope_check(edit):
    fixture = _fixture()
    edit(fixture[2])
    result = _parse(fixture)
    assert not result.reviewed and result.plan is fixture[1]
    assert result.diagnostics["reason"] == "exhibition_set_requirement_without_matching_intent"


def test_explicit_numeric_minimum_is_preserved_without_claiming_nl_proof():
    quote = "至少两件画面能看到他们互相递东西"
    fixture = _fixture(question=CLASS_QUOTE + "，" + quote, goal_quote=quote,
                       minimum=2, origin="visitor_explicit")
    result = _parse(fixture)
    assert result.reviewed
    assert result.plan.exhibition_set_requirements[0]["minWitnesses"] == 2
    assert result.plan.exhibition_set_requirements[0]["quantifierOrigin"] == "visitor_explicit"
    # No topic/quantifier keywords pretend to validate an LLM's interpretation.
    assert result.diagnostics["semanticFaithfulnessProven"] is False
    assert "quantifier interpretation" in result.diagnostics["boundary"]


@pytest.mark.parametrize("field,value", [
    ("id", "s2"), ("id", []), ("text", ""), ("text", "x"), ("text", "x" * 241),
    ("text", []), ("sourceQuote", "我想看看他们如何递东西"), ("sourceQuote", ""),
    ("sourceQuote", []), ("minWitnesses", True), ("minWitnesses", 0),
    ("minWitnesses", 4), ("minWitnesses", 1.0), ("minWitnesses", "1"),
    ("minWitnesses", 2),  # product default must not invent a larger quota.
    ("quantifierOrigin", "model_inferred"), ("quantifierOrigin", []),
    ("evidenceScope", "historical_inference_from_image"), ("evidenceScope", []),
])
def test_malformed_set_entries_fail_closed(field, value):
    fixture = _fixture()
    fixture[2]["plan"]["exhibitionSetRequirements"][0][field] = value
    result = _parse(fixture)
    assert not result.reviewed and not result.changed and result.plan is fixture[1]


@pytest.mark.parametrize("edit", [
    lambda value: value["plan"].pop("exhibitionSetRequirements"),
    lambda value: value["plan"].update(exhibitionSetRequirements=None),
    lambda value: value["plan"].update(exhibitionSetRequirements={}),
    lambda value: value["plan"].update(exhibitionSetRequirements=[None]),
    lambda value: value["plan"]["exhibitionSetRequirements"][0].pop("sourceQuote"),
    lambda value: value["plan"]["exhibitionSetRequirements"][0].update(approved=True),
    lambda value: value["plan"]["exhibitionSetRequirements"].extend(
        deepcopy(value["plan"]["exhibitionSetRequirements"]) * 3),
])
def test_incomplete_or_overlong_set_schema_never_silently_drops_requirements(edit):
    fixture = _fixture()
    edit(fixture[2])
    result = _parse(fixture)
    assert not result.reviewed and result.plan is fixture[1]


def test_record_explanation_set_requirements_may_not_use_image_only_evidence():
    question, draft, output = _fixture()
    draft = replace(draft, evidence_mode="record_explanation", visual_predicate_ids=())
    output["plan"].update(evidenceMode="record_explanation", visualPredicateIds=[])
    result = parse_query_plan_review(output, question=question, draft=draft)
    assert not result.reviewed
    assert result.diagnostics["reason"] == "invalid_exhibition_set_evidence_scope"


def test_record_explanation_mode_cannot_be_turned_visual_to_accept_set_goals():
    question, draft, output = _fixture()
    draft = replace(draft, evidence_mode="record_explanation", visual_predicate_ids=())
    result = parse_query_plan_review(output, question=question, draft=draft)
    assert not result.reviewed and result.plan is draft
    assert result.diagnostics["reason"] == "review_weakened_record_evidence_mode"


def test_preexisting_record_bound_set_obligation_cannot_be_weakened_to_vision():
    question, draft, output = _fixture()
    old = dict(output["plan"]["exhibitionSetRequirements"][0], evidenceScope="institution_record")
    draft = replace(draft, exhibition_set_requirements=(old,))
    result = parse_query_plan_review(output, question=question, draft=draft)
    assert not result.reviewed and result.plan is draft
    assert result.diagnostics["reason"] == "review_weakened_set_evidence_scope"


def test_record_bound_set_goal_is_accepted_without_visual_permission():
    question, draft, output = _fixture(evidence_scope="institution_record")
    draft = replace(draft, evidence_mode="record_explanation", visual_predicate_ids=())
    output["plan"].update(evidenceMode="record_explanation", visualPredicateIds=[])
    result = parse_query_plan_review(output, question=question, draft=draft)
    assert result.reviewed
    assert result.plan.exhibition_set_requirements[0]["evidenceScope"] == "institution_record"
    assert result.plan.visual_predicate_ids == ()


@pytest.mark.parametrize("field,value", [("objectTypes", ["painting"]), ("materials", ["gold"]), ("cultures", ["Japan"])])
def test_new_set_goal_does_not_grant_authority_to_change_locked_filters(field, value):
    fixture = _fixture()
    fixture[2]["plan"]["hardFilters"][field] = value
    result = _parse(fixture)
    assert not result.reviewed and result.plan is fixture[1]
    assert result.diagnostics["reason"] == "review_changed_locked_hard_filters"


def test_set_payload_dictionaries_do_not_mutate_existing_plan():
    _, draft, output = _fixture()
    draft = replace(draft, exhibition_set_requirements=tuple(output["plan"]["exhibitionSetRequirements"]))
    payload = query_plan_review_payload(QUESTION, draft)
    payload["draftPlan"]["exhibitionSetRequirements"][0]["minWitnesses"] = 3
    assert draft.exhibition_set_requirements[0]["minWitnesses"] == 1


def test_prompt_separates_logic_from_quantification_and_preserves_set_obligations():
    prompt = query_plan_review_prompt()
    for text in ("targetScope", "importance/logic", "exhibitionSetRequirements",
                 "one object/scene", "final", "minWitnesses", "product_default",
                 "every-object restriction", "complete relevant clause", "institution_record"):
        assert text in prompt
    assert "Each mandatory relation belongs in a mandatory predicate" not in prompt
    assert "Neither can live only in a query or pool leg" in prompt


def test_distributed_comparison_categories_coexist_with_one_required_observation():
    question = "我想对照中国、日本和欧洲的器物，并看看纹样怎样围绕边缘排列"
    class_predicate = "an object from China OR Japan OR Europe"
    goal = "one object shows how a motif is arranged around its edge"
    draft = RetrievalQueryPlan(
        valid=True, in_collection_scope=True,
        search_queries=("border motif object",), semantic_query="border motifs across cultural categories",
        mandatory_predicates=(class_predicate,), pool_coverage_legs=("China", "Japan", "Europe"),
        interpretation="对照器物的边缘纹样", reason="跨类别视觉观察",
        evidence_mode="visual_observation", visual_predicate_ids=(),
    )
    output = {"plan": query_plan_document(draft), "intentChecks": [
        {"sourceQuote": "我想对照中国、日本和欧洲的器物", "scope": "alternative",
         "targetScope": "per_object", "requirement": class_predicate},
        {"sourceQuote": "并看看纹样怎样围绕边缘排列", "scope": "mandatory",
         "targetScope": "exhibition_set", "requirement": goal},
    ]}
    output["plan"]["exhibitionSetRequirements"] = [{
        "id": "s1", "text": goal, "sourceQuote": "并看看纹样怎样围绕边缘排列",
        "minWitnesses": 1, "quantifierOrigin": "product_default",
        "evidenceScope": "visible_features_or_record",
    }]
    result = parse_query_plan_review(output, question=question, draft=draft)
    assert result.reviewed
    assert result.plan.mandatory_predicates == (class_predicate,)
    assert result.plan.pool_coverage_legs == ("China", "Japan", "Europe")
    # The single-object witness establishes the observed relation, not three
    # mutually exclusive category memberships or an unverified coverage claim.
    assert len(result.plan.exhibition_set_requirements) == 1
    assert result.plan.exhibition_set_requirements[0]["text"] == goal
    assert result.diagnostics["semanticFaithfulnessProven"] is False


def test_prompt_does_not_confuse_distributed_axes_with_single_object_set_witnesses():
    prompt = query_plan_review_prompt()
    for text in ("distributed selection axes", "scope=alternative with targetScope=per_object",
                 "any-of", "separate named categories", "fictitious",
                 "independent required", "poolCoverageLegs alone",
                 "explicit every-object", "numerical quota", "historical/functional/causal"):
        assert text in prompt
    assert "targetScope=pool_coverage" not in prompt


@pytest.mark.parametrize("prompt_factory", [query_plan_prompt, query_plan_review_prompt])
def test_planner_and_reviewer_share_required_question_not_politeness_policy(prompt_factory):
    prompt = prompt_factory()
    for text in ("Politeness is NOT a waiver", "我想", "看看", "比较", "可省略", "有则更好",
                 "pacing/aesthetic preference", "unanswered", "EVERY displayed object"):
        assert text in prompt
    assert "voluntary attention" not in prompt
    assert "voluntary observation" not in prompt
    assert "Wanting to compare a feature does" in " ".join(prompt.split())


@pytest.mark.parametrize("prompt_factory", [query_plan_prompt, query_plan_review_prompt])
def test_planner_and_reviewer_separate_metadata_axes_from_visual_predicates(prompt_factory):
    prompt = prompt_factory()
    for text in ("Metadata" if prompt_factory is query_plan_prompt else "metadata category axes",
                 "poolCoverageLegs", "same visual", "per category" if prompt_factory is query_plan_review_prompt else "each category",
                 "TEXT", "material identity", "cultural origin", "institution_record",
                 "final named-culture gate"):
        assert text in prompt


def test_reviewer_rebuilds_set_obligations_after_changing_scope_not_stale_draft_copy():
    prompt = query_plan_review_prompt()
    for text in ("Rebuild exhibitionSetRequirements from the final intentChecks",
                 "copying stale draft entries", "wrongly category-multiplied",
                 "matching exhibition_set check", "correct scope"):
        assert text in prompt


@pytest.mark.parametrize("quote,goal,evidence_mode", [
    ("我想看看人物之间怎样用手势回应", "one scene shows people responding to each other with gestures", "visual_observation"),
    ("能不能让我比较器物的开口与轮廓", "one object shows its opening in relation to its outline", "visual_observation"),
    ("我想知道旧伤是怎样修补的", "the institution record connects an object's old damage to its repair", "record_explanation"),
    ("想留意构图中的空白如何分隔人物", "one composition shows empty space separating depicted people", "visual_observation"),
])
def test_required_polite_question_spec_is_representable_without_universal_admission(quote, goal, evidence_mode):
    # Hand-authored contract specifications, not claims that a real model was tested.
    question, draft, output = _fixture(question=CLASS_QUOTE + "，" + quote, goal_quote=quote)
    draft = replace(draft, evidence_mode=evidence_mode,
                    visual_predicate_ids=() if evidence_mode == "record_explanation" else ("p1",))
    output["plan"].update(evidenceMode=evidence_mode, visualPredicateIds=list(draft.visual_predicate_ids))
    output["intentChecks"][1]["requirement"] = goal
    output["plan"]["exhibitionSetRequirements"][0].update(
        text=goal, evidenceScope="institution_record" if evidence_mode == "record_explanation" else "visible_features_or_record")
    result = parse_query_plan_review(output, question=question, draft=draft)
    assert result.reviewed
    assert result.plan.mandatory_predicates == (CLASS_PREDICATE,)
    assert result.plan.exhibition_set_requirements[0]["text"] == goal
    assert result.plan.selection_constraints == ()
    assert result.diagnostics["intentChecks"][1]["targetScope"] == "exhibition_set"
    assert result.diagnostics["intentChecks"][1]["scope"] == "mandatory"
    assert result.diagnostics["semanticFaithfulnessProven"] is False


@pytest.mark.parametrize("quote", ["局部纹样可省略", "清淡颜色的有则更好", "整体节奏慢一点"])
def test_explicit_opt_out_or_pacing_preference_spec_does_not_add_set_obligations(quote):
    question, draft, output = _fixture(question=CLASS_QUOTE + "，" + quote, goal_quote=quote)
    preference = "a visitor-stated optional presentation preference"
    output["intentChecks"][1].update(scope="optional", targetScope="preference", requirement=preference)
    output["plan"].update(exhibitionSetRequirements=[], selectionRationaleConstraints=[preference])
    result = parse_query_plan_review(output, question=question, draft=draft)
    assert result.reviewed
    assert result.plan.mandatory_predicates == (CLASS_PREDICATE,)
    assert result.plan.exhibition_set_requirements == ()
    assert result.plan.selection_constraints == (preference,)
