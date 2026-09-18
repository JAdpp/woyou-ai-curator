"""v4 compile contracts use hand-authored specifications, not live model calls."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from app.query_plan_review import (
    LEGACY_QUERY_PLAN_REVIEW_VERSION,
    QUERY_PLAN_REVIEW_VERSION,
    V4_SEARCH_KEYS,
    legacy_query_plan_review_prompt,
    parse_query_plan_review,
    parse_query_plan_review_v3,
    parse_query_plan_review_v4,
    query_plan_document,
    query_plan_review_payload,
    query_plan_review_prompt,
)
from app.retrieval_agent import RetrievalQueryPlan
from app.retrieval_filters import FilterSpec


CLASS_QUOTE = "给我看一些器物"
OBSERVATION_QUOTE = "我想看看纹样怎样围绕边缘排列"
EDITORIAL_QUOTE = "记录不清楚的就说明不清楚，不要从照片猜材料"
PREFERENCE_QUOTE = "节奏可以慢一点"
QUESTION = "，".join((CLASS_QUOTE, OBSERVATION_QUOTE, EDITORIAL_QUOTE, PREFERENCE_QUOTE))
CLASS_TEXT = "the object is a vessel"
OBSERVATION_TEXT = "one object shows a motif arranged around its edge"
EDITORIAL_TEXT = "State what supplied records do not establish; do not infer material from photographs"


def _draft(**changes):
    draft = RetrievalQueryPlan(
        valid=True, in_collection_scope=True, search_queries=("vessel border motif",),
        semantic_query="vessels with motifs arranged around their edges",
        mandatory_predicates=(CLASS_TEXT,), interpretation="观察器物边缘纹样", reason="视觉观察",
        evidence_mode="visual_observation", visual_predicate_ids=("p1",),
    )
    return replace(draft, **changes)


def _admission(identifier="i1", **changes):
    return {"id": identifier, "sourceQuote": CLASS_QUOTE, "type": "admission", "text": CLASS_TEXT,
            "evidenceScope": "visible_features_or_record", **changes}


def _witness(identifier="i2", **changes):
    return {"id": identifier, "sourceQuote": OBSERVATION_QUOTE, "type": "set_witness",
            "text": OBSERVATION_TEXT, "evidenceScope": "visible_features_or_record",
            "minWitnesses": 1, "quantifierOrigin": "product_default", **changes}


def _editorial(identifier="i3", **changes):
    return {"id": identifier, "sourceQuote": EDITORIAL_QUOTE, "type": "editorial",
            "text": EDITORIAL_TEXT, **changes}


def _fixture(draft=None, *, intents=None):
    draft = draft or _draft()
    output = {"searchPlan": {key: value for key, value in query_plan_document(draft).items() if key in V4_SEARCH_KEYS},
              "intents": intents if intents is not None else [_admission(), _witness(), _editorial()]}
    return draft, output


def _parse(fixture, *, question=QUESTION):
    return parse_query_plan_review(fixture[1], question=question, draft=fixture[0])


def test_v4_compiles_single_source_conditions_and_editorial_separately():
    fixture = _fixture()
    result = _parse(fixture)
    assert result.reviewed and result.changed
    assert result.plan.mandatory_predicates == (CLASS_TEXT,)
    assert result.plan.visual_predicate_ids == ("p1",)
    assert result.plan.exhibition_set_requirements == ({
        "id": "s1", "text": OBSERVATION_TEXT, "sourceQuote": OBSERVATION_QUOTE,
        "minWitnesses": 1, "quantifierOrigin": "product_default", "evidenceScope": "visible_features_or_record",
    },)
    assert result.plan.editorial_constraints == (EDITORIAL_TEXT,)
    assert result.plan.selection_constraints == ()
    assert result.diagnostics["version"] == QUERY_PLAN_REVIEW_VERSION == "query-plan-fidelity-review-v4"
    assert result.diagnostics["conditionAuthority"] == "single_source_intents"
    assert result.diagnostics["semanticFaithfulnessProven"] is False
    assert result.diagnostics["compiledBindings"] == [
        {"intentId": "i1", "type": "admission", "sourceQuote": CLASS_QUOTE,
         "targetField": "mandatoryPerObjectPredicates", "conditionId": "p1", "evidenceScope": "visible_features_or_record"},
        {"intentId": "i2", "type": "set_witness", "sourceQuote": OBSERVATION_QUOTE,
         "targetField": "exhibitionSetRequirements", "conditionId": "s1", "evidenceScope": "visible_features_or_record"},
        {"intentId": "i3", "type": "editorial", "sourceQuote": EDITORIAL_QUOTE,
         "targetField": "editorialConstraints", "index": 0},
    ]


def test_complete_rebuild_cannot_inherit_stale_draft_conditions_or_editorial():
    stale = {"id": "s1", "text": "obsolete overconstrained goal", "sourceQuote": CLASS_QUOTE,
             "minWitnesses": 1, "quantifierOrigin": "product_default", "evidenceScope": "visible_features_or_record"}
    draft = _draft(mandatory_predicates=("a stale universal requirement",),
                   visual_predicate_ids=("p1",), exhibition_set_requirements=(stale,),
                   selection_constraints=("old preference",), editorial_constraints=("old editorial instruction",))
    result = _parse(_fixture(draft, intents=[_admission(evidenceScope="institution_record")]))
    assert result.reviewed
    assert result.plan.mandatory_predicates == (CLASS_TEXT,)
    assert result.plan.visual_predicate_ids == ()
    assert result.plan.exhibition_set_requirements == ()
    assert result.plan.selection_constraints == result.plan.editorial_constraints == ()


def test_same_intent_text_never_needs_a_second_llm_generated_copy():
    fixture = _fixture(intents=[_admission(), _witness(text="a differently phrased complete visible relation")])
    assert not {"intentChecks", "plan"} & set(fixture[1])
    assert not {"exhibitionSetRequirements", "mandatoryPerObjectPredicates"} & set(fixture[1]["searchPlan"])
    result = _parse(fixture)
    assert result.reviewed
    assert result.plan.exhibition_set_requirements[0]["text"] == fixture[1]["intents"][1]["text"]
    assert "matching_intent" not in result.diagnostics.get("reason", "")


def test_known_unknown_question_is_an_editorial_classification_not_forced_two_group_existence():
    question = "给我看一些器物，哪些记录说明了原来的主人，哪些还不知道？"
    quote = "哪些记录说明了原来的主人，哪些还不知道？"
    instruction = "For each selected object distinguish established ownership from what supplied records leave unknown"
    fixture = _fixture(_draft(evidence_mode="record_explanation", visual_predicate_ids=()), intents=[
        _admission(evidenceScope="institution_record"),
        _editorial("i2", sourceQuote=quote, text=instruction),
    ])
    result = _parse(fixture, question=question)
    assert result.reviewed
    assert result.plan.mandatory_predicates == (CLASS_TEXT,)
    assert result.plan.exhibition_set_requirements == ()
    assert result.plan.editorial_constraints == (instruction,)
    assert result.plan.visual_predicate_ids == ()


def test_explicit_two_branch_existence_can_still_compile_to_two_witnesses():
    quote = "至少一件记录已确认主人，至少一件现有记录仍未确认主人"
    fixture = _fixture(_draft(evidence_mode="record_explanation", visual_predicate_ids=()), intents=[
        _admission(evidenceScope="institution_record"),
        _witness("i2", sourceQuote=quote, text="supplied records establish the original owner",
                 evidenceScope="institution_record", quantifierOrigin="visitor_explicit"),
        _witness("i3", sourceQuote=quote, text="the supplied records leave original ownership unestablished",
                 evidenceScope="institution_record", quantifierOrigin="visitor_explicit"),
    ])
    result = _parse(fixture, question=CLASS_QUOTE + "，" + quote)
    assert result.reviewed
    assert [row["id"] for row in result.plan.exhibition_set_requirements] == ["s1", "s2"]
    assert len(result.plan.mandatory_predicates) == 1
    assert result.diagnostics["semanticFaithfulnessProven"] is False


def test_metadata_axes_and_record_admission_do_not_gain_visual_ids():
    question = "给我看一些中国、日本和欧洲的器物，" + OBSERVATION_QUOTE
    fixture = _fixture(intents=[
        _admission(sourceQuote="给我看一些中国、日本和欧洲的器物", text="a vessel from China OR Japan OR Europe",
                   evidenceScope="institution_record"), _witness(),
    ])
    fixture[1]["searchPlan"]["poolCoverageLegs"] = ["China", "Japan", "Europe"]
    result = _parse(fixture, question=question)
    assert result.reviewed
    assert result.plan.pool_coverage_legs == ("China", "Japan", "Europe")
    assert result.plan.visual_predicate_ids == ()
    assert len(result.plan.exhibition_set_requirements) == 1
    assert result.plan.exhibition_set_requirements[0]["text"] == OBSERVATION_TEXT


def test_visual_ids_are_derived_from_admission_scopes_not_model_supplied_indices():
    fixture = _fixture(intents=[
        _admission(evidenceScope="institution_record"),
        _admission("i6", sourceQuote=OBSERVATION_QUOTE, text="the requested visible admission feature is present"),
    ])
    result = _parse(fixture)
    assert result.reviewed
    assert result.plan.visual_predicate_ids == ("p2",)
    assert result.diagnostics["compiledBindings"][1]["intentId"] == "i6"


def test_editorial_has_no_fact_citation_or_positive_witness_and_preference_stays_separate():
    fixture = _fixture(intents=[_admission(), _editorial(), {
        "id": "i4", "sourceQuote": PREFERENCE_QUOTE, "type": "preference", "text": "Use a slower visiting pace",
    }])
    result = _parse(fixture)
    assert result.reviewed
    assert result.plan.exhibition_set_requirements == ()
    assert result.plan.editorial_constraints == (EDITORIAL_TEXT,)
    assert result.plan.selection_constraints == ("Use a slower visiting pace",)
    assert "positive witnesses" in result.diagnostics["boundary"]


def test_legacy_whole_question_record_mode_can_be_corrected_without_claiming_semantic_proof():
    fixture = _fixture(_draft(evidence_mode="record_explanation", visual_predicate_ids=()))
    fixture[1]["searchPlan"]["evidenceMode"] = "visual_observation"
    result = _parse(fixture)
    assert result.reviewed
    assert result.diagnostics["draftEvidenceModeChanged"] is True
    assert result.diagnostics["semanticFaithfulnessProven"] is False


@pytest.mark.parametrize("kind", ["admission", "set_witness"])
def test_same_source_explicit_record_scope_cannot_be_weakened_to_visual(kind):
    prior = {"id": "s1", "text": "a record-bound historical relation", "sourceQuote": OBSERVATION_QUOTE,
             "minWitnesses": 1, "quantifierOrigin": "product_default", "evidenceScope": "institution_record"}
    intent = _witness() if kind == "set_witness" else _admission("i2", sourceQuote=OBSERVATION_QUOTE, text="an alleged visible relation")
    fixture = _fixture(_draft(exhibition_set_requirements=(prior,)), intents=[_admission(), intent])
    result = _parse(fixture)
    assert not result.reviewed and result.plan is fixture[0]
    assert result.diagnostics["reason"] == "review_weakened_set_evidence_scope"


@pytest.mark.parametrize("kind", ["admission", "set_witness"])
def test_record_explanation_narrows_visual_scope_instead_of_granting_or_discarding_it(kind):
    intents = [_admission(evidenceScope="institution_record")]
    intents.append(_witness() if kind == "set_witness" else _admission("i2", text="another admission"))
    fixture = _fixture(_draft(evidence_mode="record_explanation", visual_predicate_ids=()), intents=intents)
    snapshot = deepcopy(fixture[1])
    result = _parse(fixture)
    assert result.reviewed
    assert fixture[1] == snapshot
    assert result.plan.evidence_mode == "record_explanation"
    assert result.plan.visual_predicate_ids == ()
    assert all(row["evidenceScope"] == "institution_record" for row in result.plan.exhibition_set_requirements)
    assert [row["evidenceScope"] for row in result.diagnostics["compiledBindings"]] == ["institution_record"] * 2
    assert result.diagnostics["evidenceScopeNarrowings"] == [{
        "intentId": "i2", "type": kind, "declared": "visible_features_or_record",
        "compiled": "institution_record", "reason": "record_explanation_mode"}]
    assert result.diagnostics["intents"][1]["evidenceScope"] == "visible_features_or_record"


@pytest.mark.parametrize("value", ["historical_inference", "", None, [], True])
def test_record_explanation_still_rejects_unknown_evidence_scope(value):
    intents = [_admission(evidenceScope="institution_record"), _witness(evidenceScope=value)]
    fixture = _fixture(_draft(evidence_mode="record_explanation", visual_predicate_ids=()), intents=intents)
    result = _parse(fixture)
    assert not result.reviewed and result.plan is fixture[0]
    assert result.diagnostics["reason"] == "invalid_v4_evidence_scope"


@pytest.mark.parametrize("key,value", [("objectTypes", ["painting"]), ("materials", ["gold"]), ("cultures", ["Japan"])])
def test_compiler_has_no_new_authority_to_change_locked_filters(key, value):
    fixture = _fixture()
    fixture[1]["searchPlan"]["hardFilters"][key] = value
    result = _parse(fixture)
    assert not result.reviewed and result.plan is fixture[0]
    assert result.diagnostics["reason"] == "review_changed_locked_hard_filters"


@pytest.mark.parametrize("edit", [
    lambda value: value.update(plan={}),
    lambda value: value.pop("searchPlan"),
    lambda value: value.pop("intents"),
    lambda value: value["searchPlan"].update(exhibitionSetRequirements=[]),
    lambda value: value["searchPlan"].update(mandatoryPerObjectPredicates=[]),
    lambda value: value["searchPlan"].update(visualPredicateIds=["p1"]),
    lambda value: value["searchPlan"].update(selectionRationaleConstraints=[]),
    lambda value: value["searchPlan"].update(inCollectionScope="true"),
    lambda value: value["searchPlan"].update(evidenceMode=[]),
    lambda value: value["searchPlan"].update(catalogueQueries=[]),
    lambda value: value["searchPlan"].update(catalogueQueries=["x"] * 6),
    lambda value: value["searchPlan"].update(semanticQuery=""),
    lambda value: value.update(intents=None),
    lambda value: value.update(intents=[]),
    lambda value: value["intents"][0].pop("evidenceScope"),
    lambda value: value["intents"][0].update(minWitnesses=1),
    lambda value: value["intents"][0].update(type=[]),
    lambda value: value["intents"][0].update(type="pool_coverage"),
    lambda value: value["intents"][0].update(evidenceScope=[]),
    lambda value: value["intents"][0].update(id="i11"),
    lambda value: value["intents"][0].update(id=[]),
    lambda value: value["intents"][1].update(id="i1"),
    lambda value: value["intents"][1].update(text="x"),
    lambda value: value["intents"][1].update(text="x" * 241),
    lambda value: value["intents"][1].update(text=" padded "),
    lambda value: value["intents"][1].update(text="line\nbreak"),
    lambda value: value["intents"][1].update(sourceQuote="a paraphrase not in the question"),
    lambda value: value["intents"][1].update(sourceQuote=""),
    lambda value: value["intents"][1].update(sourceQuote=[]),
    lambda value: value["intents"][1].update(minWitnesses=True),
    lambda value: value["intents"][1].update(minWitnesses=0),
    lambda value: value["intents"][1].update(minWitnesses=4),
    lambda value: value["intents"][1].update(minWitnesses=1.0),
    lambda value: value["intents"][1].update(minWitnesses=2),
    lambda value: value["intents"][1].update(quantifierOrigin="model_guess"),
    lambda value: value["intents"][2].update(evidenceScope="visible_features_or_record"),
    lambda value: value["intents"][2].update(minWitnesses=1),
    lambda value: value["intents"].append(dict(value["intents"][0], id="i7")),
    lambda value: value.update(intents=value["intents"] * 4),
])
def test_malformed_or_double_written_contract_fails_closed_without_mutating_draft(edit):
    fixture = _fixture()
    snapshot = deepcopy(fixture[0])
    edit(fixture[1])
    result = _parse(fixture)
    assert not result.reviewed and not result.changed and result.plan is fixture[0]
    assert fixture[0] == snapshot


@pytest.mark.parametrize("kind,count", [("admission", 5), ("set_witness", 4), ("editorial", 5), ("preference", 5)])
def test_each_intent_type_has_a_bounded_count(kind, count):
    intents = [] if kind == "admission" else [_admission()]
    for index in range(count):
        identifier = f"i{len(intents) + 1}"
        if kind == "admission":
            row = _admission(identifier, text=f"distinct object condition {index}")
        elif kind == "set_witness":
            row = _witness(identifier, text=f"distinct visible relationship {index}")
        else:
            row = _editorial(identifier, type=kind, text=f"distinct output instruction {index}")
        intents.append(row)
    result = _parse(_fixture(intents=intents))
    assert not result.reviewed
    assert result.diagnostics["reason"] == "too_many_v4_intents_of_type"


def test_identical_statement_cannot_be_both_set_and_every_object_obligation():
    result = _parse(_fixture(intents=[_admission(), _witness(text=CLASS_TEXT)]))
    assert not result.reviewed
    assert result.diagnostics["reason"] == "set_requirement_also_forced_per_object"


def test_same_editorial_rule_cannot_also_be_optional():
    result = _parse(_fixture(intents=[_admission(), _editorial(), _editorial("i4", type="preference")]))
    assert not result.reviewed
    assert result.diagnostics["reason"] == "editorial_constraint_also_optional"


def test_out_of_scope_keeps_no_conditions_or_search():
    fixture = _fixture(intents=[])
    fixture[1]["searchPlan"].update(inCollectionScope=False, catalogueQueries=[], semanticQuery="", catalogueTypeHints=[], poolCoverageLegs=[])
    result = _parse(fixture)
    assert result.reviewed
    assert result.plan.in_collection_scope is False
    assert result.plan.search_queries == result.plan.mandatory_predicates == result.plan.editorial_constraints == ()


def test_legacy_parser_and_dispatch_remain_versioned_v3():
    draft = _draft()
    output = {"intentChecks": [{"sourceQuote": CLASS_QUOTE, "scope": "mandatory",
                               "targetScope": "per_object", "requirement": CLASS_TEXT}],
              "plan": query_plan_document(draft)}
    explicit = parse_query_plan_review_v3(output, question=QUESTION, draft=draft)
    dispatched = parse_query_plan_review(output, question=QUESTION, draft=draft)
    assert explicit.reviewed and dispatched == explicit
    assert explicit.diagnostics["version"] == LEGACY_QUERY_PLAN_REVIEW_VERSION
    output["intentChecks"][0].pop("targetScope")
    failure = parse_query_plan_review(output, question=QUESTION, draft=draft)
    assert not failure.reviewed
    assert failure.diagnostics["version"] == LEGACY_QUERY_PLAN_REVIEW_VERSION


def test_v4_parser_does_not_silently_accept_a_legacy_shape():
    draft = _draft()
    result = parse_query_plan_review_v4({"plan": query_plan_document(draft), "intentChecks": []}, question=QUESTION, draft=draft)
    assert not result.reviewed
    assert result.diagnostics["version"] == QUERY_PLAN_REVIEW_VERSION


def test_new_prompt_is_shorter_single_source_and_keeps_evidence_and_intent_boundaries():
    prompt = query_plan_review_prompt("en")
    assert len(prompt) < len(legacy_query_plan_review_prompt()) * 0.65
    for text in ("searchPlan", "intents", "Write each requirement ONCE", "single authoritative",
                 "EXACT contiguous", "Politeness is NOT a waiver", "我想", "看看", "比较",
                 "editorial", "ALL relevant final prose", "not object evidence", "not an automatic quota",
                 "SUPPLIED records", "Do not turn a background/rephrasing question", "cultural origin",
                 "institution_record", "same-source", "common visual goal once per category",
                 "hardFilters EXACTLY", "Do not search only the known group"):
        assert text in prompt
    assert '"intentChecks":[' not in prompt
    assert "requirement VERBATIM" not in prompt


def test_runtime_payload_excludes_all_previous_requirement_and_interpretation_fields():
    old_set = {"id": "s1", "text": "LEAK_SET_REQUIREMENT", "sourceQuote": "LEAK_OLD_QUOTE",
               "minWitnesses": 3, "quantifierOrigin": "visitor_explicit", "evidenceScope": "institution_record"}
    draft = _draft(interpretation="LEAK_INTERPRETATION", reason="LEAK_REASON",
                   mandatory_predicates=("LEAK_PREDICATE",), visual_predicate_ids=("LEAK_VISUAL_ID",),
                   exhibition_set_requirements=(old_set,), selection_constraints=("LEAK_PREFERENCE",),
                   editorial_constraints=("LEAK_EDITORIAL",), evidence_mode="record_explanation")
    payload = query_plan_review_payload(QUESTION, draft)
    assert set(payload) == {"visitorQuestion", "language", "recallDraft"}
    assert set(payload["recallDraft"]) == {
        "catalogueQueries", "semanticQuery", "catalogueTypeHints", "poolCoverageLegs", "hardFilters",
    }
    serialized = json.dumps(payload)
    assert "LEAK_" not in serialized
    assert "record_explanation" not in serialized
    assert not {"inCollectionScope", "queryInterpretation", "reason", "evidenceMode",
                "mandatoryPerObjectPredicates", "visualPredicateIds", "exhibitionSetRequirements",
                "selectionRationaleConstraints", "editorialConstraints", "draftPlan"} & set(payload["recallDraft"])
    assert payload["recallDraft"]["catalogueQueries"] == list(draft.search_queries)
    assert payload["recallDraft"]["semanticQuery"] == draft.semantic_query


def test_runtime_recall_payload_deep_copy_and_exact_original_question():
    draft = _draft(filters=FilterSpec(cultures=("Japan",)),
                   catalogue_type_hints=("vessel",), pool_coverage_legs=("recorded", "uncertain"))
    question = "  原问题保留空白\n" + QUESTION + "  "
    payload = query_plan_review_payload(question, draft, "en")
    assert payload["visitorQuestion"] == question
    assert payload["language"] == "en"
    payload["recallDraft"]["hardFilters"]["cultures"].append("China")
    payload["recallDraft"]["catalogueQueries"].append("mutated")
    payload["recallDraft"]["catalogueTypeHints"].append("mutated")
    payload["recallDraft"]["poolCoverageLegs"].append("mutated")
    assert draft.filters.cultures == ("Japan",)
    assert draft.search_queries == ("vessel border motif",)
    assert draft.catalogue_type_hints == ("vessel",)
    assert draft.pool_coverage_legs == ("recorded", "uncertain")
    fresh = query_plan_review_payload(question, draft)
    assert fresh["recallDraft"]["hardFilters"]["cultures"] == ["Japan"]


def test_recall_projection_does_not_remove_internal_record_scope_guard():
    prior = {"id": "s1", "text": "an explicitly record-bound condition", "sourceQuote": OBSERVATION_QUOTE,
             "minWitnesses": 1, "quantifierOrigin": "product_default", "evidenceScope": "institution_record"}
    draft = _draft(exhibition_set_requirements=(prior,))
    assert "exhibitionSetRequirements" not in query_plan_review_payload(QUESTION, draft)["recallDraft"]
    result = _parse(_fixture(draft))
    assert not result.reviewed and result.plan is draft
    assert result.diagnostics["reason"] == "review_weakened_set_evidence_scope"


def test_deanchored_prompt_treats_recall_as_vocabulary_not_requirement_authority():
    prompt = " ".join(query_plan_review_prompt("en").split())
    for text in ("Independently compile", "original question FIRST", "recallDraft is only fallible search vocabulary",
                 "NOT an authority", "Do not infer an obligation", "No prior predicates, goals, rationale or mode",
                 "Normally write exactly ONE minimal admission", "Every separate admission is joined by AND",
                 "never split allowed alternatives into two entries", "from recallDraft"):
        assert text in prompt
    assert "draftPlan" not in prompt
