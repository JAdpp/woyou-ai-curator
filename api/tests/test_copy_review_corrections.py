"""RC7: declared corrections must execute and chapters own their sources."""

import asyncio
from copy import deepcopy
from time import perf_counter

import pytest

from app.curatorial_copy_review import (
    COPY_REVIEW_VERSION, copy_review_payload, copy_review_prompt,
    merge_copy_review_batches, parse_copy_review, parse_copy_review_batch,
    split_copy_review_payload,
)
from app.generator import ExhibitionGenerator
from app.models import EvidenceChunk, MuseumObject


def objects():
    return [MuseumObject(
        id=key, title=key, rights="CC0", institution="Fixture museum",
        imageUrl=f"https://museum.test/{key}.jpg", objectUrl=f"https://museum.test/{key}",
        evidence=[EvidenceChunk(id=f"{key}:description", text=text,
                                sourceUrl=f"https://museum.test/{key}", sourceTitle="Record",
                                sourceKind="institution_description")],
    ) for key, text in (
        ("a", "The record locates this scene in Harbor A and describes visitors as workers."),
        ("b", "The record locates this scene in Harbor B."),
    )]


def frame():
    return {"title": "Two places", "chapters": [
        {"title": "First place", "leadIn": "This scene is in Harbor C."},
        {"title": "Second place", "leadIn": "This scene is in Harbor B."}],
        "epilogue": {"text": "The institution does not identify any visitors."}}


def structure():
    return [{"index": 0, "itemCount": 1, "objectIds": ["a"]},
            {"index": 1, "itemCount": 1, "objectIds": ["b"]}]


def payload(value=None):
    return copy_review_payload(value or frame(), objects(), question="Compare the scenes",
                                language="en", chapter_structure=structure())


def response(data, edits=()):
    return {"schemaVersion": COPY_REVIEW_VERSION, "reviewedFieldCount": data["expectedFieldCount"],
            "outcome": "revised" if edits else "pass", "changes": list(edits)}


def patch(path, original, replacement, *, evidence_id="a:description", quote="Harbor A"):
    return {"path": path, "original": original, "replacement": replacement,
            "changeKind": "correct_fact", "evidenceIds": [evidence_id],
            "supportingQuotes": [{"evidenceId": evidence_id, "quote": quote}],
            "reason": "Use the actual same-member record."}


def noop(path="/epilogue/text", original=None):
    original = original or frame()["epilogue"]["text"]
    return {"path": path, "original": original, "replacement": original,
            "changeKind": "neutralize", "evidenceIds": [], "supportingQuotes": [],
            "reason": "The record describes visitors; delete the unsupported absence assertion."}


def batch_for(data, path):
    return next(batch for batch in split_copy_review_payload(data)
                if any(row["path"] == path for row in batch["publicCopyFields"]))


def test_unchanged_true_pass_remains_valid_but_declared_noop_correction_is_invalid():
    data = payload()
    assert parse_copy_review(response(data), frame(), data).review_passed
    result = parse_copy_review(response(data, [noop()]), frame(), data)
    assert not result.review_passed and result.errors == ("invalid_noop_change",)
    assert result.frame == frame()


def test_noop_patch_cannot_be_laundered_by_other_good_edits_or_batch_merge():
    data = payload()
    batches = split_copy_review_payload(data)
    values = [response(batch, [noop()] if any(row["path"] == "/epilogue/text"
                                            for row in batch["publicCopyFields"]) else ())
              for batch in batches]
    result = merge_copy_review_batches(values, frame(), batches)
    assert not result.review_passed and any("invalid_noop_change" in error for error in result.errors)
    assert result.frame == frame()


def test_chapter_owner_ids_and_focused_target_come_from_actual_members_not_prose():
    original = structure()
    data = copy_review_payload(frame(), objects(), question="Compare", chapter_structure=original)
    batch = batch_for(data, "/chapters/0/title")
    assert all(row["ownerObjectIds"] == ["a"] for row in batch["publicCopyFields"])
    assert batch["chapterTarget"] == original[0]
    assert {row["objectId"] for row in batch["selectedObjects"]} == {"a", "b"}
    original[0]["objectIds"][0] = "changed"
    assert data["chapterStructure"][0]["objectIds"] == ["a"]


def test_same_chapter_source_correction_passes_without_changing_membership():
    data = payload()
    batch = batch_for(data, "/chapters/0/leadIn")
    change = patch("/chapters/0/leadIn", frame()["chapters"][0]["leadIn"], "This scene is in Harbor A.")
    result = parse_copy_review_batch(response(batch, [change]), frame(), batch)
    assert result.review_passed and result.frame["chapters"][0]["leadIn"] == "This scene is in Harbor A."
    assert result.frame["chapters"][1] == frame()["chapters"][1]
    assert batch["chapterStructure"] == structure()


@pytest.mark.parametrize("with_own_source", [False, True])
def test_chapter_cannot_borrow_neighbor_source_even_if_one_own_source_is_also_cited(with_own_source):
    data = payload()
    batch = batch_for(data, "/chapters/0/leadIn")
    change = patch("/chapters/0/leadIn", frame()["chapters"][0]["leadIn"], "This scene is in Harbor B.",
                   evidence_id="b:description", quote="Harbor B")
    if with_own_source:
        change["evidenceIds"].append("a:description")
        change["supportingQuotes"].append({"evidenceId": "a:description", "quote": "Harbor A"})
    result = parse_copy_review_batch(response(batch, [change]), frame(), batch)
    assert not result.review_passed and result.errors == ("wrong_chapter_evidence_owner",)
    assert result.frame == frame()


def test_same_chapter_subset_quote_does_not_need_irrelevant_all_member_citations():
    value = {"title": "One group", "chapters": [{"title": "Two scenes"}]}
    data = copy_review_payload(value, objects(), question="Compare", chapter_structure=[
        {"index": 0, "itemCount": 2, "objectIds": ["a", "b"]}])
    change = patch("/chapters/0/title", "Two scenes", "One record names Harbor A")
    assert parse_copy_review(response(data, [change]), value, data).review_passed


def test_legacy_empty_structure_keeps_compatibility_without_claiming_chapter_ownership():
    data = copy_review_payload(frame(), objects(), question="Compare")
    assert data["chapterStructure"] == []
    assert next(row for row in data["publicCopyFields"] if row["path"] == "/chapters/0/title")["ownerObjectIds"] == []
    assert parse_copy_review(response(data), frame(), data).review_passed


def test_late_chapter_mapping_mutation_cannot_silently_change_the_field_source_owner():
    data = payload()
    data["chapterStructure"][0]["objectIds"], data["chapterStructure"][1]["objectIds"] = ["b"], ["a"]
    result = parse_copy_review(response(data), frame(), data)
    assert not result.review_passed and result.errors == ("stale_review_payload",)


@pytest.mark.parametrize("bad", [
    {}, [{"index": 0, "itemCount": 1, "objectIds": ["missing"]}],
    [{"index": True, "itemCount": 1, "objectIds": ["a"]}],
    [{"index": 0, "itemCount": 2, "objectIds": ["a"]}],
    [{"index": 0, "itemCount": 2, "objectIds": ["a", "a"]}],
    [{"index": 0, "itemCount": 1, "objectIds": ["a"]},
     {"index": 1, "itemCount": 1, "objectIds": ["a"]}],
])
def test_malformed_or_foreign_chapter_mapping_is_not_silently_ignored(bad):
    with pytest.raises(ValueError):
        copy_review_payload(frame(), objects(), question="Compare", chapter_structure=bad)


def test_valid_mapping_cannot_name_objects_outside_selected_context():
    bad = structure()
    bad[1]["objectIds"] = ["unknown"]
    with pytest.raises(ValueError, match="chapter_structure_unknown_object"):
        copy_review_payload(frame(), objects(), question="Compare", chapter_structure=bad)


def test_noop_uses_existing_repair_and_applies_a_real_correction_without_new_stage(monkeypatch):
    generator = ExhibitionGenerator.__new__(ExhibitionGenerator)
    value = {"title": "Two places", "epilogue": deepcopy(frame()["epilogue"])}
    data = copy_review_payload(value, objects(), question="Compare", language="en")
    stages = []

    async def model(prompt, batch, *, stage, timeout_seconds):
        stages.append(stage)
        if "repair" not in stage:
            return response(batch, [noop()])
        assert batch["contractErrors"] == ["invalid_noop_change"]
        return response(batch, [patch("/epilogue/text", value["epilogue"]["text"],
                                      "The record describes visitors as workers.", quote="describes visitors as workers")])

    monkeypatch.setattr(generator, "_generate_model_json", model)
    result = asyncio.run(generator._review_frame_copy(value, data, language="en", deadline=perf_counter()+5))
    assert stages == ["frame_review:batch0", "frame_review_repair:batch0"]
    assert result.review_passed and result.frame["epilogue"]["text"] == "The record describes visitors as workers."
    assert result.frame["title"] == value["title"]


def test_repeated_global_noop_never_publishes_original_error_as_reviewed(monkeypatch):
    generator = ExhibitionGenerator.__new__(ExhibitionGenerator)
    value = {"title": "Two places", "epilogue": deepcopy(frame()["epilogue"])}
    data = copy_review_payload(value, objects(), question="Compare")
    stages = []

    async def model(prompt, batch, *, stage, timeout_seconds):
        stages.append(stage)
        return response(batch, [noop('/title', value['title'])])

    monkeypatch.setattr(generator, "_generate_model_json", model)
    result = asyncio.run(generator._review_frame_copy(value, data, language="en", deadline=perf_counter()+5))
    assert len(stages) == 2 and not result.review_passed
    assert any("invalid_noop_change" in error for error in result.errors)


def test_unfixed_object_noop_uses_only_existing_local_field_fallback(monkeypatch):
    generator = ExhibitionGenerator.__new__(ExhibitionGenerator)
    value = {"title": "Keep this personal theme", "curatorialBrief": {"objects": [{
        "objectId": "a", "evidenceIds": ["a:description"],
        "selectionRationale": "No visitors are documented.", "relation": "This object predates all others."}]}}
    data = copy_review_payload(value, objects(), question="Compare")
    stages = []

    async def model(prompt, batch, *, stage, timeout_seconds):
        stages.append(stage)
        if batch["batchScope"]["group"][0] != "object":
            return response(batch)
        row = batch["publicCopyFields"][0]
        return response(batch, [noop(row["path"], row["original"])])

    monkeypatch.setattr(generator, "_generate_model_json", model)
    result = asyncio.run(generator._review_frame_copy(value, data, language="en", deadline=perf_counter()+5))
    assert result.review_passed and result.status == "bounded_local_fallback"
    assert result.locally_neutralized_fields == 2 and result.frame["title"] == value["title"]
    assert result.frame["curatorialBrief"]["objects"][0]["selectionRationale"] != "No visitors are documented."
    assert len([stage for stage in stages if "repair" in stage]) == 1


def test_prompt_treats_absence_as_factual_and_collective_chapter_titles_as_member_claims():
    prompt = copy_review_prompt()
    for text in ("NOT a safe neutral fallback", "zero-based chapter index", "EVERY actual member",
                 "A reason saying", "no-op patch is invalid"):
        assert text in prompt
