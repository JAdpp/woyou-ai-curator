"""General relation checks; source binding is not a semantic gold judge."""

from copy import deepcopy

import pytest

from app.collections import SearchResult
from app.models import EvidenceChunk, MuseumObject
from app.retrieval_agent import (
    CONDITION_CONTRACT_VERSION, audit_payload, audit_prompt, parse_audit,
)


def candidate(text="The teacher stands beside a shelf of books."):
    return SearchResult(obj=MuseumObject(
        id="lesson", title="The School", rights="CC0", institution="Museum",
        imageUrl="https://museum.test/lesson.jpg", objectUrl="https://museum.test/lesson",
        evidence=[EvidenceChunk(id="lesson:description", text=text,
                                sourceUrl="https://museum.test/lesson", sourceTitle="Record",
                                sourceKind="institution_description")],
    ), score=70, matched_evidence_ids=("lesson:description",))


def output(item, *, relation="exact", quote=None, evidence_id="lesson:description"):
    return {"conditionContractVersion": CONDITION_CONTRACT_VERSION,
            "answerability": "supported", "searchQueries": [], "expansionReason": "none",
            "accepted": [{"objectId": item.obj.id, "relevanceScore": 0.99,
                          "evidenceIds": ["lesson:description"],
                          "conditionEvidence": [{"conditionId": "p1", "status": "supported",
                                                 "evidenceId": evidence_id,
                                                 "supportingQuote": quote or item.obj.evidence[0].text,
                                                 "relation": relation}]}]}


def parse(value, item, **kwargs):
    return parse_audit(value, [item], question="Show a teacher handing a book to a pupil",
                       max_expansion_queries=3, strict_conditions=True,
                       mandatory_predicates=("a teacher hands a book to a pupil",), **kwargs)


@pytest.mark.parametrize("relation", ["broader", "different", "unknown", "similar", None, []])
def test_reported_partial_or_different_predicate_support_cannot_hide_behind_supported(relation):
    item = candidate()
    result = parse(output(item, relation=relation), item)
    assert result.valid and not result.accepted and result.answerability == "unsupported"
    check = result.condition_checks[0]
    assert check["sourceBound"] is True
    assert check["failure"] == "non_entailing_relation"
    assert check["relationAssessment"] == "model_reported"


@pytest.mark.parametrize("relation", ["exact", "narrower"])
def test_documented_whole_relation_remains_eligible_and_traceable(relation):
    item = candidate("The teacher hands an open book to the pupil seated beside her.")
    result = parse(output(item, relation=relation), item)
    assert len(result.accepted) == 1
    assert result.accepted[0].matched_evidence_ids == ("lesson:description",)
    assert result.condition_checks[0]["relation"] == relation
    assert result.condition_checks[0]["sourceBound"] is True
    # False limits what LOCAL validation proves; it does not mean this
    # documented relationship failed the model's semantic assessment.
    assert result.condition_checks[0]["semanticEntailmentProven"] is False
    assert result.condition_checks[0]["validationBoundary"] == "source_binding_not_semantic_proof"


def test_legacy_predicate_check_without_relation_is_explicitly_unassessed_not_assumed_exact():
    item = candidate("The teacher hands an open book to the pupil seated beside her.")
    value = output(item)
    value["accepted"][0]["conditionEvidence"][0].pop("relation")
    result = parse(value, item)
    assert len(result.accepted) == 1
    assert result.condition_checks[0]["relation"] == "not_reported"
    assert result.condition_checks[0]["relationAssessment"] == "not_provided"


def test_real_but_unrelated_quote_is_not_falsely_claimed_to_be_mechanically_proven():
    # Deliberately wrong model entailment: being beside books is NOT handing
    # one to a pupil. Without a semantic model call, this parser cannot know
    # that. Preserve this counterexample instead of asserting a fake guarantee.
    item = candidate()
    result = parse(output(item, relation="exact"), item)
    assert len(result.accepted) == 1
    check = result.condition_checks[0]
    assert check["sourceBound"] is True and check["relationAssessment"] == "model_reported"
    assert check["semanticEntailmentProven"] is False


def test_exact_relation_never_overrides_unshown_or_borrowed_evidence():
    item = candidate()
    result = parse(output(item, quote="The teacher hands a book to the pupil."), item)
    assert not result.accepted and result.condition_rejections[0]["failure"] == "unbound_quote"
    result = parse(output(item, evidence_id="another:description"), item)
    assert not result.accepted and result.condition_rejections[0]["failure"] == "unbound_quote"


def report(item):
    return {"results": [{"objectId": item.obj.id, "imageEvidenceId": "image:lesson",
                         "sourceUrl": item.obj.image_url, "imageSha256": "f" * 64,
                         "model": "fixture-vision", "status": "reviewed",
                         "imageSupplied": True, "imageReviewed": True,
                         "sourceKind": "collection_image", "scope": "visible_features_only",
                         "observations": [{"id": "visual:lesson:1", "predicateIds": ["p1"],
                                           "text": "An adult hands an open book to a seated child."}]}]}


def test_relation_check_does_not_grant_visual_predicate_or_historical_authority():
    item = candidate()
    visual = report(item)
    text = visual["results"][0]["observations"][0]["text"]
    value = output(item, quote=text, evidence_id="visual:lesson:1")
    allowed = dict(evidence_mode="visual_observation", visual_sources=visual,
                   visual_predicate_ids=("p1",))
    assert len(parse(value, item, **allowed).accepted) == 1
    no_permission = deepcopy(allowed)
    no_permission["visual_predicate_ids"] = ()
    assert not parse(value, item, **no_permission).accepted
    historical = {**allowed, "evidence_mode": "record_explanation"}
    assert not parse(value, item, **historical).accepted


def test_facet_visual_eligibility_does_not_leak_to_predicate_with_empty_plan_permissions():
    item = candidate()
    payload = audit_payload("Look at book handling", [item], required_count=5,
                            top_k=10, pass_number=1, strict_conditions=True,
                            mandatory_predicates=("an open book is being held",),
                            catalogue_type_hints=("book",), evidence_mode="open_exploration",
                            visual_predicate_ids=(), visual_sources=report(item))
    conditions = {row["id"]: row for row in payload["retrievalContract"]["perObjectConditions"]}
    assert conditions["p1"]["evidenceScope"] == "institution_record"
    assert conditions["object_kind"]["evidenceScope"] == "visible_features_or_record"
    assert payload["candidates"][0]["visualEvidence"][0]["allowedConditionIds"] == ["object_kind"]


def test_audit_prompt_requests_whole_relation_alignment_without_new_topic_rules():
    prompt = audit_prompt()
    assert "subject/participants, action" in prompt
    assert "target/carrier and qualifying details" in prompt
    assert "requiredCount is not an approval" in prompt
    assert "Return relation for predicate checks too" in prompt
    assert "whole requirement" in prompt
    assert "silence means unknown" in prompt
