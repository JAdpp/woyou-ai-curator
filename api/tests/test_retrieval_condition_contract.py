"""The V6 boundary binds judgments to shown sources, not model confidence.

These fixtures test the contract and authority boundary. They do not pretend
that a quote-containment unit test can establish semantic truth; that requires
the separate live, blinded retrieval evaluation.
"""

from copy import deepcopy

import pytest

from app.collections import SearchResult
from app.models import EvidenceChunk, MuseumObject
from app.retrieval_agent import (
    CONDITION_CONTRACT_VERSION,
    audit_payload,
    parse_audit,
    parse_predicate_verification,
    parse_query_plan,
)


def candidate(key="basket", text="The woven basket has a flat base and a fitted lid."):
    return SearchResult(obj=MuseumObject(
        id=key, title="Woven basket", sourceId=key, material="Rattan",
        rights="CC0", imageUrl=f"https://museum.test/{key}.jpg",
        objectUrl=f"https://museum.test/{key}", institution="Museum",
        evidence=[EvidenceChunk(id=f"{key}:e1", text=text,
                                sourceUrl=f"https://museum.test/{key}",
                                sourceTitle="Catalogue", sourceKind="institution_description")],
    ), score=80, matched_evidence_ids=(f"{key}:e1",))


def check(key="p1", *, status="supported", evidence_id="basket:e1", quote=None,
          alternative=None, relation="exact"):
    value = {"conditionId": key, "status": status, "evidenceId": evidence_id,
             "supportingQuote": quote or "The woven basket has a flat base and a fitted lid."}
    if alternative:
        value.update(matchedAlternative=alternative, relation=relation)
    return value


def decision(checks=None, key="basket"):
    return {"objectId": key, "relevanceScore": 0.95, "evidenceIds": [f"{key}:e1"],
            "conditionEvidence": checks if checks is not None else [check()]}


def output(*decisions):
    return {"conditionContractVersion": CONDITION_CONTRACT_VERSION,
            "accepted": list(decisions) or [decision()], "answerability": "supported",
            "searchQueries": [], "expansionReason": "none"}


def parse(value, candidates=None, **kwargs):
    return parse_audit(value, candidates or [candidate()], question="Look at baskets",
                       max_expansion_queries=3, strict_conditions=True,
                       mandatory_predicates=("A documented visible feature",), **kwargs)


def test_strict_contract_is_explicit_and_legacy_does_not_silently_enable_it():
    legacy = output()
    legacy.pop("conditionContractVersion")
    legacy["accepted"][0].pop("conditionEvidence")
    strict = parse(legacy)
    assert not strict.valid and strict.accepted == []
    assert strict.condition_rejections[0]["failure"] == "missing_or_invalid_contract_version"
    compatible = parse_audit(legacy, [candidate()], question="basket",
                             max_expansion_queries=3)
    assert compatible.valid and len(compatible.accepted) == 1


def test_conditions_are_independent_and_facet_alternatives_are_not_conjunctions():
    payload = audit_payload("baskets or trays from three cultures", [candidate()],
                            required_count=5, top_k=20, pass_number=1,
                            mandatory_predicates=("visible base", "documented use"),
                            catalogue_type_hints=("basket", "tray"),
                            explicit_materials=("rattan", "willow"),
                            pool_coverage_legs=("Europe", "Africa", "Americas"),
                            strict_conditions=True)
    contract = payload["retrievalContract"]
    assert contract["conditionContractVersion"] == CONDITION_CONTRACT_VERSION
    assert [item["id"] for item in contract["perObjectConditions"]] == [
        "p1", "p2", "object_kind", "material"]
    assert contract["perObjectConditions"][2]["alternatives"] == ["basket", "tray"]
    assert contract["perObjectConditions"][3]["operator"] == "any_of"
    assert contract["poolCoverageLegs"] == ["Europe", "Africa", "Americas"]


@pytest.mark.parametrize("bad_checks,failure", [
    ([], "missing_check"),
    ([check(status="unknown")], "unknown"),
    ([check(status="contradicted")], "contradicted"),
    ([check(status="certain")], "invalid_status"),
    ([check(), check(status="contradicted")], "duplicate_check"),
    ([check(), check(key="imagined-condition")], "malformed_condition_contract"),
    ([check(evidence_id="other:e1")], "unbound_quote"),
    ([check(quote="A completely invented visual feature.")], "unbound_quote"),
])
def test_high_relevance_never_overrides_incomplete_or_unbound_conditions(bad_checks, failure):
    audit = parse(output(decision(bad_checks)))
    assert audit.valid and not audit.accepted and audit.answerability == "unsupported"
    assert failure in {row.get("failure") for row in audit.condition_rejections}


def test_valid_conditions_retain_only_real_institution_evidence_ids():
    audit = parse(output(decision([
        check(), check("object_kind", alternative="basket"),
    ])), catalogue_type_hints=("basket", "tray"))
    assert len(audit.accepted) == 1
    assert audit.accepted[0].matched_evidence_ids == ("basket:e1",)
    assert "condition_source_bound" in audit.accepted[0].retrieval_sources
    assert len(audit.condition_checks) == 2


@pytest.mark.parametrize("relation", ["broader", "different", "unknown", "similar"])
def test_broader_or_similar_material_never_substitutes_for_requested_material(relation):
    item = candidate(text="The material of this dish is glazed earthenware.")
    checks = [check(quote=item.obj.evidence[0].text),
              check("material", quote=item.obj.evidence[0].text,
                    alternative="porcelain", relation=relation)]
    audit = parse(output(decision(checks)), [item], explicit_materials=("porcelain",))
    assert not audit.accepted
    assert audit.condition_rejections[0]["failure"] == "non_entailing_relation"


def test_a_named_shape_is_not_accepted_when_its_kind_check_is_contradicted():
    item = candidate("vessel", "The glass vessel takes its name from its bell-like shape.")
    checks = [check(evidence_id="vessel:e1", quote=item.obj.evidence[0].text),
              check("object_kind", status="contradicted", evidence_id="vessel:e1",
                    quote=item.obj.evidence[0].text, alternative="bell", relation="different")]
    audit = parse(output(decision(checks, "vessel")), [item], catalogue_type_hints=("bell",))
    assert not audit.accepted
    assert audit.condition_rejections[0]["conditionId"] == "object_kind"


def test_a_model_cannot_replace_the_requested_alternative_in_its_check():
    audit = parse(output(decision([check(), check("material", alternative="earthenware")])),
                  explicit_materials=("porcelain",))
    assert not audit.accepted
    assert audit.condition_rejections[0]["failure"] == "invalid_alternative"


def test_a_quote_after_the_shown_300_character_window_is_not_evidence():
    suffix = "Visible finger marks prove the asked feature."
    item = candidate(text="A plain catalogue sentence. " * 20 + suffix)
    payload = audit_payload("surface marks", [item], required_count=5, top_k=20,
                            pass_number=1, strict_conditions=True)
    assert suffix not in payload["candidates"][0]["evidence"][0]["text"]
    assert not parse(output(decision([check(quote=suffix)])), [item]).accepted
    verified = parse_predicate_verification({"verified": [{
        "objectId": "basket", "predicateEvidence": [{"predicateId": "p1", "status": "entailed",
        "evidenceId": "basket:e1", "supportingQuote": suffix}]}]}, [item],
        question="surface marks", mandatory_predicates=("Visible marks",))
    assert verified == []


def test_duplicate_or_conflicting_object_decisions_fail_closed():
    audit = parse(output(decision(), decision([check(status="unknown")])))
    assert not audit.accepted
    assert audit.condition_rejections[0]["failure"] == "duplicate_object_decisions"
    value = output()
    value["rejected"] = [decision([check(status="contradicted")])]
    audit = parse(value)
    assert not audit.accepted
    assert "conflicting_object_decisions" in {row.get("failure") for row in audit.condition_rejections}


def visual_report(item=None):
    item = item or candidate()
    return {"version": "fixture", "results": [{
        "objectId": item.obj.id, "imageEvidenceId": f"image:{item.obj.id}",
        "sourceUrl": item.obj.image_url, "imageSha256": "a" * 64,
        "model": "fixture-vision", "status": "reviewed", "imageSupplied": True,
        "imageReviewed": True, "sourceKind": "collection_image", "scope": "visible_features_only",
        "observations": [{"id": "visual:base", "text": "The basket has a flat woven base.",
                          "predicateIds": ["p1"]}],
    }]}


def visual_decision():
    return decision([check(evidence_id="visual:base", quote="The basket has a flat woven base.")])


def test_visual_conditions_need_both_explicit_authority_and_a_real_inspection():
    report = visual_report()
    allowed = dict(evidence_mode="visual_observation", visual_sources=report,
                   visual_predicate_ids=("p1",))
    audit = parse(output(visual_decision()), **allowed)
    assert len(audit.accepted) == 1
    assert audit.accepted[0].matched_evidence_ids == ("basket:e1",)
    assert "visual_condition_source_bound" in audit.accepted[0].retrieval_sources
    assert audit.condition_checks[0]["imageEvidenceId"] == "image:basket"
    assert not parse(output(visual_decision()), visual_sources=report,
                     visual_predicate_ids=("p1",)).accepted  # historical mode
    assert not parse(output(visual_decision()), visual_sources=report,
                     evidence_mode="visual_observation").accepted  # no predicate authority


@pytest.mark.parametrize("field,value", [
    ("imageReviewed", False), ("imageSupplied", False), ("status", "failed"),
    ("sourceUrl", "https://other.test/image.jpg"), ("imageSha256", "invented"),
    ("sourceKind", "institution_description"), ("scope", "historical_facts"),
    ("objectId", "another-object"), ("model", ""),
])
def test_unverified_or_wrong_object_visual_report_is_not_exposed(field, value):
    report = visual_report()
    report["results"][0][field] = value
    payload = audit_payload("visible weave", [candidate()], required_count=5, top_k=20,
                            pass_number=1, strict_conditions=True, visual_sources=report,
                            evidence_mode="visual_observation", visual_predicate_ids=("p1",),
                            mandatory_predicates=("visible weave",))
    assert payload["candidates"][0]["visualEvidence"] == []


def test_visual_scope_never_grants_material_authority_even_if_caller_requests_it():
    report = visual_report()
    report["results"][0]["observations"][0]["predicateIds"] = ["p1", "material"]
    checks = [check(), check("material", alternative="rattan", evidence_id="visual:base",
                             quote="The basket has a flat woven base.")]
    audit = parse(output(decision(checks)), visual_sources=report,
                  evidence_mode="visual_observation", visual_predicate_ids=("p1", "material"),
                  explicit_materials=("rattan",))
    assert not audit.accepted
    assert audit.condition_rejections[0]["failure"] == "unbound_quote"


def test_query_plan_visual_authority_is_bounded_to_real_predicates_and_mode():
    value = {"inCollectionScope": True, "catalogueQueries": ["woven basket"],
             "mandatoryPerObjectPredicates": ["visible shape", "documented use"],
             "visualPredicateIds": ["p1", "p99", "material"],
             "evidenceMode": "visual_observation"}
    assert parse_query_plan(value, question="basket", max_queries=5).visual_predicate_ids == ("p1",)
    historical = deepcopy(value)
    historical["evidenceMode"] = "record_explanation"
    assert parse_query_plan(historical, question="basket", max_queries=5).visual_predicate_ids == ()
    outside = deepcopy(value)
    outside["inCollectionScope"] = False
    assert parse_query_plan(outside, question="basket", max_queries=5).visual_predicate_ids == ()
