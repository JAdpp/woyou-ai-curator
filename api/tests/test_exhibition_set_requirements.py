"""Per-object admission and exhibition coverage are independent contracts.

No provider calls or topic routing: fixtures exercise schema, source ownership,
full-relation reporting and visual authority, not semantic model accuracy.
"""

from dataclasses import replace
from hashlib import sha256

import pytest

from app.collections import SearchResult
from app.models import EvidenceChunk, MuseumObject
from app.retrieval_agent import (
    CONDITION_CONTRACT_VERSION, audit_payload, audit_prompt, parse_audit,
    parse_query_plan, query_plan_prompt,
)


QUESTION = "我想看信件，留意递信给另一人的场面，也看看收信人不确定的例子"
RELATION = "A person hands a letter to another person."


def requirement(key="s1", **overrides):
    return {"id": key, "text": "One person hands a letter to another in the depicted scene",
            "sourceQuote": "递信给另一人的场面", "minWitnesses": 1,
            "quantifierOrigin": "product_default", "evidenceScope": "institution_record",
            **overrides}


def plan(requirements=None, **overrides):
    return {"inCollectionScope": True, "catalogueQueries": ["letter delivery"],
            "mandatoryPerObjectPredicates": ["The record concerns a letter"],
            "evidenceMode": "record_explanation",
            "exhibitionSetRequirements": [requirement()] if requirements is None else requirements,
            **overrides}


def candidate(key="letter-a", text=RELATION):
    return SearchResult(obj=MuseumObject(
        id=key, title="A letter scene", institution="Fixture Museum", rights="CC0",
        objectUrl=f"https://museum.test/{key}", imageUrl=f"https://museum.test/{key}.jpg",
        evidence=[EvidenceChunk(id=f"{key}:description", text=text,
                                sourceUrl=f"https://museum.test/{key}",
                                sourceTitle="Institution record", sourceKind="institution_description")],
    ), score=80, matched_evidence_ids=(f"{key}:description",))


def check(item, condition_id="s1", **overrides):
    return {"conditionId": condition_id, "status": "supported",
            "evidenceId": item.obj.evidence[0].id,
            "supportingQuote": item.obj.evidence[0].text, "relation": "exact", **overrides}


def decision(item, *, set_checks=None, object_checks=None):
    return {"objectId": item.obj.id, "relevanceScore": 0.95,
            "evidenceIds": [item.obj.evidence[0].id],
            "conditionEvidence": [check(item, "p1")] if object_checks is None else object_checks,
            "setConditionEvidence": [check(item)] if set_checks is None else set_checks}


def output(*decisions):
    return {"conditionContractVersion": CONDITION_CONTRACT_VERSION,
            "accepted": list(decisions), "answerability": "supported",
            "searchQueries": [], "expansionReason": "none"}


def audit(value, items, **kwargs):
    arguments = dict(question=QUESTION, max_expansion_queries=3, strict_conditions=True,
                     mandatory_predicates=("The record concerns a letter",),
                     exhibition_set_requirements=(requirement(),))
    return parse_audit(value, items, **{**arguments, **kwargs})


def visual_report(item, ids=None):
    return {"results": [{"objectId": item.obj.id, "imageEvidenceId": f"image:{item.obj.id}",
                         "sourceUrl": item.obj.image_url, "imageSha256": "c" * 64,
                         "sourceKind": "collection_image", "scope": "visible_features_only",
                         "model": "fixture-vision", "status": "reviewed", "imageSupplied": True,
                         "imageReviewed": True, "observations": [{
                             "id": f"visual:{item.obj.id}", "text": RELATION,
                             "predicateIds": ["s1"] if ids is None else ids}]}]}


def test_new_plan_preserves_exact_requirements_separately_from_mandatory_predicates():
    value = plan()
    parsed = parse_query_plan(value, question=QUESTION, max_queries=5)
    assert parsed.valid and parsed.exhibition_set_requirements == (requirement(),)
    assert parsed.mandatory_predicates == ("The record concerns a letter",)
    value["exhibitionSetRequirements"][0]["text"] = "Changed after parse"
    assert parsed.exhibition_set_requirements[0]["text"] == requirement()["text"]


def test_omitted_or_empty_set_field_preserves_old_provider_compatibility():
    value = plan([])
    assert parse_query_plan(value, question=QUESTION, max_queries=5).exhibition_set_requirements == ()
    value.pop("exhibitionSetRequirements")
    parsed = parse_query_plan(value, question=QUESTION, max_queries=5)
    assert parsed.valid and parsed.exhibition_set_requirements == ()


@pytest.mark.parametrize("bad", [None, {}, "s1", [None], ["a requirement"],
                                  [requirement(), requirement()],
                                  [requirement("s2")],
                                  [requirement(f"s{i}") for i in range(1, 5)]])
def test_invalid_set_container_or_ids_fail_plan_instead_of_dropping_the_goal(bad):
    value = plan()
    value["exhibitionSetRequirements"] = bad
    parsed = parse_query_plan(value, question=QUESTION, max_queries=5)
    assert not parsed.valid and parsed.reason == "invalid_exhibition_set_requirements"


@pytest.mark.parametrize("field,value", [
    ("id", "p1"), ("text", ""), ("text", " padded text "), ("text", "x" * 241),
    ("text", []), ("sourceQuote", ""), ("sourceQuote", " "), ("sourceQuote", 1),
    ("sourceQuote", "递信的场面"), ("sourceQuote", "递信给另一人的场面，也看看收信人确定的例子"),
    ("minWitnesses", 0), ("minWitnesses", 4), ("minWitnesses", True),
    ("minWitnesses", "1"), ("minWitnesses", 1.0), ("minWitnesses", 2),
    ("quantifierOrigin", "model_inferred"), ("evidenceScope", "world_knowledge"),
    ("evidenceScope", "visible_features_or_record"),
])
def test_invalid_set_field_or_historical_visual_permission_fails_plan(field, value):
    parsed = parse_query_plan(plan([requirement(**{field: value})]), question=QUESTION, max_queries=5)
    assert not parsed.valid


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_set_schema_has_exact_keys(mutation):
    row = requirement()
    if mutation == "missing":
        row.pop("minWitnesses")
    else:
        row["optional"] = True
    assert not parse_query_plan(plan([row]), question=QUESTION, max_queries=5).valid


@pytest.mark.parametrize("minimum", [1, 2, 3])
def test_explicit_minimum_is_retained_with_its_original_quantity_span(minimum):
    quote = f"至少{minimum}件"
    question = QUESTION + f"，这种场面要{quote}"
    row = requirement(sourceQuote=f"这种场面要{quote}", minWitnesses=minimum,
                      quantifierOrigin="visitor_explicit")
    parsed = parse_query_plan(plan([row]), question=question, max_queries=5)
    assert parsed.valid and parsed.exhibition_set_requirements[0] == row


def test_source_quote_is_not_whitespace_normalized_or_retranslated():
    question = "Look for hand-offs\n between people"
    row = requirement(sourceQuote="hand-offs\n between people")
    assert parse_query_plan(plan([row]), question=question, max_queries=5).valid
    row["sourceQuote"] = "hand-offs between people"
    assert not parse_query_plan(plan([row]), question=question, max_queries=5).valid


def test_payload_keeps_set_goals_out_of_every_object_gate():
    item = candidate()
    payload = audit_payload(QUESTION, [item], required_count=5, top_k=10, pass_number=1,
                            strict_conditions=True, mandatory_predicates=("The record concerns a letter",),
                            exhibition_set_requirements=(requirement(),))
    contract = payload["retrievalContract"]
    assert [row["id"] for row in contract["perObjectConditions"]] == ["p1"]
    assert [row["id"] for row in contract["exhibitionSetConditions"]] == ["s1"]
    assert contract["exhibitionSetConditions"][0]["minWitnesses"] == 1
    assert contract["exhibitionSetConditions"][0]["scope"] == "exhibition_set"


def test_witness_records_same_object_current_question_requirement_and_quoted_sources():
    item = candidate()
    result = audit(output(decision(item)), [item])
    assert result.valid and len(result.accepted) == 1
    witness, = result.accepted[0].set_witnesses
    assert witness["requirementId"] == "s1" and witness["objectId"] == item.obj.id
    assert witness["requirementText"] == requirement()["text"]
    assert witness["sourceQuote"] == requirement()["sourceQuote"]
    assert witness["questionSha256"] == sha256(QUESTION.encode("utf-8")).hexdigest()
    assert witness["evidenceIds"] == [item.obj.evidence[0].id]
    assert witness["checks"][0]["sourceBound"] is True
    assert witness["checks"][0]["relation"] == "exact"
    assert witness["checks"][0]["semanticEntailmentProven"] is False


@pytest.mark.parametrize("status", ["unknown", "contradicted", "unsupported", ""])
def test_unproven_set_goal_never_ejects_a_normally_eligible_object(status):
    item = candidate()
    result = audit(output(decision(item, set_checks=[check(item, status=status)])), [item])
    assert result.valid and len(result.accepted) == 1
    assert result.accepted[0].set_witnesses == () and not result.condition_rejections
    assert any(row.get("scope") == "exhibition_set" and row.get("failure")
               for row in result.condition_checks)


def test_separate_set_branches_do_not_become_conjunctive_per_object_conditions():
    item = candidate()
    goals = (requirement(), requirement("s2", text="The institution states the recipient is uncertain",
                                        sourceQuote="收信人不确定的例子"))
    result = audit(output(decision(item, set_checks=[check(item), check(item, "s2", status="unknown")])),
                   [item], exhibition_set_requirements=goals)
    assert len(result.accepted) == 1
    assert [row["requirementId"] for row in result.accepted[0].set_witnesses] == ["s1"]


def test_set_witness_does_not_override_a_failed_per_object_condition():
    item = candidate()
    result = audit(output(decision(item, object_checks=[check(item, "p1", status="unknown")])), [item])
    assert not result.accepted
    assert result.condition_rejections[0]["conditionId"] == "p1"


@pytest.mark.parametrize("relation", ["broader", "different", "unknown", None, []])
def test_partial_or_different_relation_cannot_be_a_set_witness(relation):
    item = candidate()
    result = audit(output(decision(item, set_checks=[check(item, relation=relation)])), [item])
    assert len(result.accepted) == 1 and result.accepted[0].set_witnesses == ()
    assert result.condition_checks[-1]["failure"] == "non_entailing_relation"


def test_new_set_contract_requires_relation_while_legacy_object_contract_is_unchanged():
    item = candidate()
    per_object, set_check = check(item, "p1"), check(item)
    per_object.pop("relation")
    set_check.pop("relation")
    result = audit(output(decision(item, object_checks=[per_object], set_checks=[set_check])), [item])
    assert len(result.accepted) == 1 and not result.accepted[0].set_witnesses


@pytest.mark.parametrize("mutation", ["borrow_id", "invent_quote", "join_fragments"])
def test_one_object_cannot_borrow_or_stitch_another_objects_relation(mutation):
    left = candidate("left", "A messenger holds a sealed letter near the doorway.")
    right = candidate("right", "A seated person receives a letter from a visitor.")
    changed = check(left)
    if mutation == "borrow_id":
        changed.update(evidenceId=right.obj.evidence[0].id, supportingQuote=right.obj.evidence[0].text)
    elif mutation == "invent_quote":
        changed["supportingQuote"] = "The messenger gives the letter to the seated person."
    else:
        changed["supportingQuote"] = left.obj.evidence[0].text + " " + right.obj.evidence[0].text
    result = audit(output(decision(left, set_checks=[changed]), decision(right, set_checks=[])), [left, right])
    assert len(result.accepted) == 2 and all(not row.set_witnesses for row in result.accepted)
    assert "unbound_quote" in {row.get("failure") for row in result.condition_checks}


def test_visible_window_not_unseen_institution_suffix_bounds_set_citations():
    item = candidate(text="A catalogue entry about a letter. " * 20 + RELATION)
    changed = check(item, supportingQuote=RELATION)
    # Supply an actually visible per-object quote; the final set relation is
    # deliberately only in the unseen suffix of that same real source row.
    admitted = check(item, "p1", supportingQuote="A catalogue entry about a letter.")
    result = audit(output(decision(item, object_checks=[admitted], set_checks=[changed])), [item])
    assert len(result.accepted) == 1 and not result.accepted[0].set_witnesses


@pytest.mark.parametrize("malformed", [None, {}, "supported", [None],
                                       [{"conditionId": ["s1"]}], [{"conditionId": "invented"}]])
def test_malformed_set_checks_do_not_crash_or_create_witnesses(malformed):
    item = candidate()
    row = decision(item)
    row["setConditionEvidence"] = malformed
    result = audit(output(row), [item])
    assert len(result.accepted) == 1 and not result.accepted[0].set_witnesses


def test_duplicate_set_verdict_is_not_counted_twice_or_cherry_picked():
    item = candidate()
    result = audit(output(decision(item, set_checks=[check(item), check(item, status="unknown")])), [item])
    assert len(result.accepted) == 1 and result.accepted[0].set_witnesses == ()
    assert result.condition_checks[-1]["failure"] == "duplicate_check"


@pytest.mark.parametrize("keep_requirements", [True, False])
def test_old_witnesses_are_cleared_on_every_new_parse(keep_requirements):
    item = candidate()
    first = audit(output(decision(item)), [item]).accepted[0]
    assert first.set_witnesses
    new_decision = decision(first)
    new_decision.pop("setConditionEvidence")
    second = audit(output(new_decision), [first],
                   exhibition_set_requirements=(requirement(),) if keep_requirements else ())
    assert len(second.accepted) == 1 and second.accepted[0].set_witnesses == ()
    assert first.set_witnesses  # no mutation of another pass's frozen result


def test_stale_witness_is_cleared_even_with_legacy_non_strict_audit():
    item = replace(candidate(), set_witnesses=({"requirementId": "s1", "questionSha256": "old"},))
    result = parse_audit(output(decision(item)), [item], question=QUESTION, max_expansion_queries=0)
    assert result.accepted[0].set_witnesses == ()


def test_union_exposes_set_only_visual_evidence_and_keeps_institution_identity():
    item = candidate(text="A paper sheet containing a letter.")
    goal = requirement(evidenceScope="visible_features_or_record")
    visual = visual_report(item)
    args = dict(exhibition_set_requirements=(goal,), evidence_mode="visual_observation", visual_sources=visual)
    payload = audit_payload(QUESTION, [item], required_count=5, top_k=10, pass_number=1,
                            strict_conditions=True, mandatory_predicates=("The record concerns a letter",), **args)
    assert payload["retrievalContract"]["visualPredicateIds"] == []
    assert payload["candidates"][0]["visualEvidence"][0]["allowedConditionIds"] == ["s1"]
    result = audit(output(decision(item, set_checks=[check(item, evidenceId=f"visual:{item.obj.id}",
                                                           supportingQuote=RELATION)])), [item], **args)
    selected, = result.accepted
    assert "visual_condition_source_bound" in selected.retrieval_sources
    assert selected.matched_evidence_ids == (item.obj.evidence[0].id,)
    assert selected.set_witnesses[0]["evidenceIds"] == [f"visual:{item.obj.id}"]
    assert selected.set_witnesses[0]["checks"][0]["imageEvidenceId"] == f"image:{item.obj.id}"


@pytest.mark.parametrize("mutation", ["unauthorized_id", "record_only", "wrong_object", "no_pixels"])
def test_visual_set_witness_requires_exact_permission_and_real_same_object_inspection(mutation):
    item = candidate(text="A paper sheet containing a letter.")
    goal = requirement(evidenceScope="visible_features_or_record")
    visual = visual_report(item)
    if mutation == "unauthorized_id":
        visual["results"][0]["observations"][0]["predicateIds"] = ["p1"]
    elif mutation == "record_only":
        goal["evidenceScope"] = "institution_record"
    elif mutation == "wrong_object":
        visual["results"][0]["objectId"] = "another"
    else:
        visual["results"][0]["imageReviewed"] = False
    result = audit(output(decision(item, set_checks=[check(item, evidenceId=f"visual:{item.obj.id}",
                                                           supportingQuote=RELATION)])), [item],
                   exhibition_set_requirements=(goal,), evidence_mode="visual_observation", visual_sources=visual)
    assert len(result.accepted) == 1 and not result.accepted[0].set_witnesses
    assert "visual_condition_source_bound" not in result.accepted[0].retrieval_sources


def test_invalid_caller_set_scope_cannot_bypass_plan_validator_in_historical_mode():
    item = candidate()
    goals = (requirement(evidenceScope="visible_features_or_record"),)
    result = audit(output(decision(item)), [item], exhibition_set_requirements=goals)
    assert not result.valid and not result.accepted
    with pytest.raises(ValueError, match="invalid_exhibition_set_requirements"):
        audit_payload(QUESTION, [item], required_count=5, top_k=10, pass_number=1,
                      exhibition_set_requirements=goals)


def test_real_but_partial_quote_with_wrong_model_exact_is_not_claimed_as_local_semantic_proof():
    # Intentional counterexample: source ownership cannot detect a model that
    # wrongly calls this incomplete relationship "exact". Preserve the limit.
    item = candidate(text="A person holds a letter beside a doorway.")
    result = audit(output(decision(item)), [item])
    witness, = result.accepted[0].set_witnesses
    assert witness["checks"][0]["sourceBound"] is True
    assert witness["checks"][0]["semanticEntailmentProven"] is False


def test_prompts_separate_required_sets_from_universal_and_optional_conditions():
    planner, auditor = query_plan_prompt(), audit_prompt()
    for term in ("exhibitionSetRequirements", "product_default", "visitor_explicit",
                 "EVERY displayed object", "optional", "ORIGINAL visitorQuestion"):
        assert term in planner
    for term in ("exhibitionSetConditions", "setConditionEvidence", "WHOLE relation",
                 "does NOT reject", "Count distinct source-bound accepted objects"):
        assert term in auditor
