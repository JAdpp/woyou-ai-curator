from __future__ import annotations

import asyncio
from copy import deepcopy
from time import perf_counter

import pytest

from app.curatorial_copy_review import (
    COPY_REVIEW_VERSION, copy_review_payload, copy_review_prompt,
    parse_copy_review, review_curatorial_copy,
)
from app.models import EvidenceChunk, MuseumObject


def objects():
    return [MuseumObject(
        id="market", title="Fishmarket", date="1902", dateEarliest=1902, dateLatest=1902,
        maker="An artist", culture="France", material="Oil on canvas",
        imageUrl="https://museum.test/market.jpg", objectUrl="https://museum.test/market",
        rights="CC0", institution="Museum", evidence=[EvidenceChunk(
            id="market:description", text="This painting depicts the harbor at Dieppe.",
            sourceUrl="https://museum.test/market", sourceTitle="Record",
            sourceKind="institution_curatorial_text"), EvidenceChunk(
            id="market:other", text="A real but unrelated row.", sourceUrl="https://museum.test/market",
            sourceTitle="Record", sourceKind="institution_metadata")]),
        MuseumObject(id="other", title="Another scene", rights="CC0",
                     imageUrl="https://museum.test/other.jpg", objectUrl="https://museum.test/other",
                     institution="Museum", evidence=[EvidenceChunk(
                         id="other:description", text="A second institution description.",
                         sourceUrl="https://museum.test/other", sourceTitle="Record",
                         sourceKind="institution_description")])]


def frame():
    return {"title": "Market scenes", "subtitle": "From Edo to Paris",
            "spaceDesign": {"wallColor": "#ffffff"},
            "chapters": [{"id": "chapter-a", "title": "An urban scene", "leadIn": "Markets in Paris",
                          "itemIds": ["item-a"]}],
            "epilogue": {"text": "A market in Paris", "openQuestions": ["How close are the figures?"]},
            "curatorialBrief": {
                "bigIdea": {"text": "People in markets", "evidenceIds": ["market:description"],
                            "confidence": "supported"},
                "objects": [{"objectId": "market", "role": "core_evidence",
                             "selectionRationale": "A market in Paris",
                             "relation": "This painting comes later than the other one.",
                             "evidenceIds": ["market:description"]}]}}


def payload(value=None):
    return copy_review_payload(value or frame(), objects(), question="Look at a market")


def patch(path="/subtitle", original="From Edo to Paris", replacement="From Edo to Dieppe"):
    return {"path": path, "original": original, "replacement": replacement,
            "changeKind": "correct_fact", "evidenceIds": ["market:description"],
            "supportingQuotes": [{"evidenceId": "market:description", "quote": "harbor at Dieppe"}],
            "reason": "The supplied record names Dieppe, not Paris."}


def response(data=None, changes=None):
    data = data or payload()
    return {"schemaVersion": COPY_REVIEW_VERSION,
            "reviewedFieldCount": len(data["publicCopyFields"]),
            "outcome": "revised" if changes else "pass", "changes": changes or []}


def test_valid_review_changes_only_allowlisted_strings_and_preserves_all_other_structure():
    value = frame()
    original = deepcopy(value)
    result = parse_copy_review(response(changes=[patch()]), value, payload(value))
    assert result.review_passed and result.status == "revised"
    assert result.frame["subtitle"] == "From Edo to Dieppe"
    expected = deepcopy(original)
    expected["subtitle"] = "From Edo to Dieppe"
    assert result.frame == expected and value == original
    assert result.to_diagnostics()["reviewPassed"] is True


def test_no_change_verdict_is_pass_only_after_all_fields_are_accounted_for():
    assert payload()["expectedFieldCount"] == len(payload()["publicCopyFields"])
    result = parse_copy_review(response(), frame(), payload())
    assert result.review_passed and result.status == "passed" and not result.changes
    incomplete = response()
    incomplete["reviewedFieldCount"] -= 1
    result = parse_copy_review(incomplete, frame(), payload())
    assert not result.review_passed and result.errors == ("incomplete_field_review",)


def test_noop_declared_correction_cannot_hide_beside_a_valid_factual_repair():
    unchanged = patch(path="/title", original=frame()["title"], replacement=frame()["title"])
    unchanged.update(changeKind="neutralize", evidenceIds=[], supportingQuotes=[])
    result = parse_copy_review(response(changes=[unchanged, patch()]), frame(), payload())
    assert not result.review_passed and result.errors == ("invalid_noop_change",)
    assert result.frame == frame()


@pytest.mark.parametrize("bad_path", [
    "/curatorialBrief/objects/0/objectId", "/curatorialBrief/objects/0/role",
    "/curatorialBrief/objects/0/evidenceIds/0", "/curatorialBrief/bigIdea/confidence",
    "/chapters/0/itemIds/0", "/spaceDesign/wallColor", "/__class__",
    "/subtitle/__class__", "/chapters/99/title", "subtitle", "/items/0/object/id",
])
def test_ids_roles_order_references_unknown_paths_and_design_cannot_be_patched(bad_path):
    result = parse_copy_review(response(changes=[patch(bad_path)]), frame(), payload())
    assert not result.review_passed and result.frame == frame()
    assert result.errors == ("invalid_or_duplicate_path",)


@pytest.mark.parametrize("key,value,error", [
    ("original", "Invented original", "original_value_mismatch"),
    ("replacement", {"text": "replacement"}, "invalid_replacement"),
    ("replacement", "", "invalid_replacement"),
    ("changeKind", ["correct_fact"], "invalid_change_kind_or_noop"),
    ("reason", "", "invalid_change_reason"),
    ("evidenceIds", ["invented"], "unshown_evidence_id"),
    ("evidenceIds", [], "factual_change_without_evidence"),
    ("supportingQuotes", [], "incomplete_supporting_quotes"),
    ("supportingQuotes", [{"evidenceId": "market:description", "quote": "A Paris fish market"}],
     "unbound_source_quote"),
])
def test_invalid_edits_fail_closed_without_partial_application(key, value, error):
    edit = patch()
    edit[key] = value
    result = parse_copy_review(response(changes=[edit]), frame(), payload())
    assert not result.review_passed and result.errors == (error,)
    assert result.frame == frame()


def test_duplicate_paths_and_extra_patch_keys_are_rejected():
    result = parse_copy_review(response(changes=[patch(), patch()]), frame(), payload())
    assert result.errors == ("invalid_or_duplicate_path",)
    edit = patch()
    edit["objectId"] = "replacement-object"
    assert parse_copy_review(response(changes=[edit]), frame(), payload()).errors == ("invalid_patch_shape",)


def test_failed_second_patch_does_not_apply_the_first_valid_patch():
    invalid = patch("/epilogue/text", "wrong original", "Observe the market")
    result = parse_copy_review(response(changes=[patch(), invalid]), frame(), payload())
    assert not result.review_passed and result.frame == frame()


def test_existing_evidence_binding_is_not_silently_expanded():
    edit = patch("/curatorialBrief/objects/0/selectionRationale", "A market in Paris", "A different scene")
    edit["evidenceIds"] = ["market:other"]
    edit["supportingQuotes"] = [{"evidenceId": "market:other", "quote": "A real but unrelated row."}]
    result = parse_copy_review(response(changes=[edit]), frame(), payload())
    assert result.errors == ("replacement_exceeds_existing_evidence_binding",)


def test_a_per_object_fact_cannot_borrow_another_objects_evidence():
    value = frame()
    value["items"] = [{"objectId": "market", "whySelected": "A market in Paris"}]
    data = payload(value)
    edit = patch("/items/0/whySelected", "A market in Paris", "A second scene")
    edit["evidenceIds"] = ["other:description"]
    edit["supportingQuotes"] = [{"evidenceId": "other:description", "quote": "A second institution description."}]
    assert parse_copy_review(response(data, [edit]), value, data).errors == ("wrong_object_evidence_owner",)


def test_neutralizing_an_unsupported_claim_does_not_need_invented_evidence():
    edit = patch("/curatorialBrief/objects/0/relation",
                 "This painting comes later than the other one.", "Compare the scenes side by side.")
    edit.update(changeKind="neutralize", evidenceIds=[], supportingQuotes=[],
                reason="The supplied records do not establish chronological order.")
    result = parse_copy_review(response(changes=[edit]), frame(), payload())
    assert result.review_passed and result.frame["curatorialBrief"]["objects"][0]["evidenceIds"] == ["market:description"]


def test_unseen_source_suffix_is_not_available_for_correction():
    selected = objects()
    selected[0].evidence[0].text = "An institution sentence. " * 40 + "A hidden decisive claim."
    data = copy_review_payload(frame(), selected, question="market")
    edit = patch()
    edit["supportingQuotes"][0]["quote"] = "A hidden decisive claim."
    assert parse_copy_review(response(data, [edit]), frame(), data).errors == ("unbound_source_quote",)


@pytest.mark.parametrize("mutate,error", [
    (lambda data: data.update(schemaVersion="wrong"), "invalid_schema_version"),
    (lambda data: data.update(outcome=["pass"]), "invalid_review_outcome"),
    (lambda data: data.update(reviewedFieldCount=True), "incomplete_field_review"),
    (lambda data: data.update(outcome="revised"), "outcome_change_mismatch"),
])
def test_invalid_review_summary_is_not_a_pass(mutate, error):
    data = response()
    mutate(data)
    assert parse_copy_review(data, frame(), payload()).errors == (error,)


def test_unresolved_and_stale_reviews_do_not_publish_unreviewed_copy():
    data = response()
    data["outcome"] = "unresolved"
    result = parse_copy_review(data, frame(), payload())
    assert not result.review_passed and result.status == "unresolved"
    stale = payload()
    stale["publicCopyFields"][0]["original"] = "old title"
    assert parse_copy_review(response(), frame(), stale).errors == ("stale_review_payload",)


class FakeProvider:
    configured = True

    def __init__(self, delay=0, fail=False):
        self.calls = 0
        self.delay = delay
        self.fail = fail
        self.cancelled = False

    async def generate_retrieval_audit_json(self, prompt, data):
        self.calls += 1
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.fail:
            raise RuntimeError("sensitive provider text must not be logged")
        return response(data, [patch()])


def test_one_bounded_request_uses_retrieval_provider_and_records_success():
    provider = FakeProvider()
    result = asyncio.run(review_curatorial_copy(frame(), objects(), provider,
                                                question="market", timeout_seconds=1))
    assert provider.calls == 1 and result.review_passed and result.elapsed_seconds >= 0


def test_timeout_cancels_request_and_never_claims_review_passed():
    provider = FakeProvider(delay=1)
    result = asyncio.run(review_curatorial_copy(frame(), objects(), provider,
                                                question="market", timeout_seconds=0.005))
    assert provider.calls == 1 and provider.cancelled
    assert result.status == "timed_out" and not result.review_passed
    assert result.frame == frame()


def test_expired_parent_deadline_makes_no_request():
    provider = FakeProvider()
    result = asyncio.run(review_curatorial_copy(frame(), objects(), provider, question="market",
                                                deadline=perf_counter() - 1))
    assert provider.calls == 0 and result.status == "timed_out"


def test_missing_provider_and_provider_errors_are_explicit_and_secret_safe():
    missing = asyncio.run(review_curatorial_copy(frame(), objects(), None, question="market"))
    assert missing.status == "unavailable" and not missing.review_passed
    provider = FakeProvider(fail=True)
    failed = asyncio.run(review_curatorial_copy(frame(), objects(), provider, question="market"))
    assert provider.calls == 1 and failed.errors == ("copy_review_error:RuntimeError",)
    assert not failed.review_passed and "sensitive" not in str(failed.to_diagnostics())


def test_prompt_preserves_source_subject_and_chronological_boundaries():
    prompt = copy_review_prompt()
    assert "non-European depicted setting is not proof" in prompt
    assert "Overlapping or uncertain date intervals" in prompt
    assert "No images are supplied" in prompt
    assert "record lacks one" in prompt
