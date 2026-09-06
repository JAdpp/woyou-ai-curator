from __future__ import annotations

from copy import deepcopy

import pytest

from app.query_plan_review import V4_SEARCH_KEYS, parse_query_plan_review_v4, query_plan_document
from app.retrieval_agent import RetrievalQueryPlan


def _fixture():
    draft = RetrievalQueryPlan(valid=True, in_collection_scope=True, search_queries=("object",),
                               semantic_query="objects with documented features", mandatory_predicates=("a documented object",),
                               evidence_mode="record_explanation")
    output = {"searchPlan": {key: value for key, value in query_plan_document(draft).items() if key in V4_SEARCH_KEYS},
              "intents": [
                  {"id": "i1", "type": "admission", "sourceQuote": "我想看看器物",
                   "text": "a documented object", "evidenceScope": "institution_record"},
                  {"id": "i2", "type": "editorial", "sourceQuote": "记录不清楚就说明不清楚",
                   "text": "Explain which supplied-record claims remain unestablished"},
              ]}
    return draft, output


def _parse(draft, output):
    return parse_query_plan_review_v4(output, question="我想看看器物，记录不清楚就说明不清楚", draft=draft)


@pytest.mark.parametrize("include,value", [(False, None), (True, None), (True, "institution_record")])
def test_editorial_annotation_is_non_authorizing_and_never_a_positive_witness(include, value):
    draft, output = _fixture()
    if include:
        output["intents"][1]["evidenceScope"] = value
    before = deepcopy(output)
    result = _parse(draft, output)
    assert result.reviewed
    assert output == before
    assert result.plan.mandatory_predicates == ("a documented object",)
    assert result.plan.exhibition_set_requirements == result.plan.visual_predicate_ids == ()
    assert result.plan.editorial_constraints == (output["intents"][1]["text"],)
    assert result.plan.selection_constraints == ()
    expected = [{"intentId": "i2", "field": "evidenceScope", "value": value, "authority": "none"}] if include else []
    assert result.diagnostics["nonWitnessAnnotations"] == expected
    binding = next(row for row in result.diagnostics["compiledBindings"] if row["intentId"] == "i2")
    assert binding["targetField"] == "editorialConstraints"
    assert not {"evidenceScope", "conditionId", "minWitnesses", "sourceIds"} & set(binding)
    assert result.diagnostics["semanticFaithfulnessProven"] is False


@pytest.mark.parametrize("value", ["visible_features_or_record", "", "historical_inference", [], {}, True, 1])
def test_editorial_visual_or_unknown_annotation_value_is_rejected(value):
    draft, output = _fixture()
    output["intents"][1]["evidenceScope"] = value
    result = _parse(draft, output)
    assert not result.reviewed and result.plan is draft
    assert result.diagnostics["reason"] == "invalid_v4_editorial_annotation"


@pytest.mark.parametrize("key,value", [("minWitnesses", 1), ("quantifierOrigin", "product_default"),
                                       ("sourceId", "invented"), ("extra", None)])
def test_nonwitness_compatibility_does_not_discard_unexpected_or_quantity_fields(key, value):
    draft, output = _fixture()
    output["intents"][1].update(evidenceScope="institution_record", **{key: value})
    result = _parse(draft, output)
    assert not result.reviewed and result.plan is draft
    assert result.diagnostics["reason"] == "invalid_v4_intent_schema"


def test_preference_does_not_inherit_editorial_annotation_exception():
    draft, output = _fixture()
    output["intents"][1].update(type="preference", evidenceScope="institution_record")
    result = _parse(draft, output)
    assert not result.reviewed
    assert result.diagnostics["reason"] == "invalid_v4_intent_schema"
