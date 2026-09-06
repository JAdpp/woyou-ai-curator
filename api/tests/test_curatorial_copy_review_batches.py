from __future__ import annotations

from copy import deepcopy

import pytest

from app.curatorial_copy_review import (
    COPY_REVIEW_VERSION, copy_review_payload, merge_copy_review_batches,
    parse_copy_review, parse_copy_review_batch, split_copy_review_payload,
)
from app.models import EvidenceChunk, MuseumObject


def fixture():
    objects = [MuseumObject(
        id=key, title=title, rights="CC0", institution="Museum",
        imageUrl=f"https://museum.test/{key}.jpg", objectUrl=f"https://museum.test/{key}",
        evidence=[EvidenceChunk(id=f"{key}:description", text=text,
                                sourceUrl=f"https://museum.test/{key}", sourceTitle="Record",
                                sourceKind="institution_description")],
    ) for key, title, text in [
        ("a", "First object", "The depicted location is Port A."),
        ("b", "Second object", "The depicted location is Port B."),
    ]]
    frame = {
        "title": "Two places", "subtitle": "Compare their settings",
        "exhibitionTheme": "Places", "curatorialThesis": "Look at each record",
        "coreAnswer": "The first scene is in Port B.",
        "subQuestions": ["Where is each place?", "What remains uncertain?"],
        "chapters": [{"id": "chapter-a", "title": "First place", "leadIn": "A scene in Port B"},
                     {"id": "chapter-b", "title": "Second place", "leadIn": "A scene in Port B"}],
        "epilogue": {"text": "Compare the two records", "openQuestions": ["What can we establish?"]},
        "curatorialBrief": {
            "bigIdea": {"text": "Compare places", "evidenceIds": ["a:description", "b:description"]},
            "keyMessages": [{"text": "Read the record", "evidenceIds": ["a:description"]}],
            "criticalQuestions": ["Which place?"],
            "objects": [{"objectId": key, "role": "core_evidence", "evidenceIds": [f"{key}:description"],
                         "selectionRationale": "A scene in Port B", "relation": "Compare its neighbour"}
                        for key in ("a", "b")],
        },
        # Deliberately reverse the item order: object grouping must use identity,
        # not match the item index to the curatorial brief's object index.
        "items": [{"id": f"item-{key}", "objectId": key,
                   "whySelected": "A scene in Port B", "relation": "Compare these places"}
                  for key in ("b", "a")],
    }
    structure = [
        {"index": 0, "itemCount": 1, "objectIds": ["a"]},
        {"index": 1, "itemCount": 1, "objectIds": ["b"]},
    ]
    payload = copy_review_payload(frame, objects, question="Compare the places without guessing",
                                  chapter_structure=structure)
    return frame, payload


def response(payload, changes=None):
    return {"schemaVersion": COPY_REVIEW_VERSION,
            "reviewedFieldCount": payload["expectedFieldCount"],
            "outcome": "revised" if changes else "pass", "changes": changes or []}


def patch(path, original, *, evidence="a:description", quote="location is Port A"):
    return {"path": path, "original": original, "replacement": "A scene in Port A",
            "changeKind": "correct_fact", "evidenceIds": [evidence],
            "supportingQuotes": [{"evidenceId": evidence, "quote": quote}],
            "reason": "Use the location from this object's record."}


def batch_for(batches, path):
    return next(batch for batch in batches if any(row["path"] == path for row in batch["publicCopyFields"]))


def test_partition_covers_every_field_once_in_small_semantic_groups():
    _, payload = fixture()
    original = deepcopy(payload)
    batches = split_copy_review_payload(payload)
    paths = [row["path"] for batch in batches for row in batch["publicCopyFields"]]
    assert len(paths) == len(set(paths)) == payload["expectedFieldCount"]
    assert set(paths) == {row["path"] for row in payload["publicCopyFields"]}
    assert all(1 <= batch["expectedFieldCount"] == len(batch["publicCopyFields"]) <= 6 for batch in batches)
    assert payload == original
    chapters = [batch for batch in batches if batch["batchScope"]["group"][0] == "chapter"]
    assert len(chapters) == 2 and all(len(batch["publicCopyFields"]) == 2 for batch in chapters)
    object_a = batch_for(batches, "/curatorialBrief/objects/0/relation")
    assert {row["path"] for row in object_a["publicCopyFields"]} == {
        "/curatorialBrief/objects/0/selectionRationale", "/curatorialBrief/objects/0/relation",
        "/items/1/whySelected", "/items/1/relation",
    }
    assert object_a["batchScope"]["group"] == ["object", "a"]
    assert len([batch for batch in batches if batch["batchScope"]["group"] == ["global"]]) >= 2


def test_every_batch_keeps_all_source_and_authoritative_chapter_context_without_aliasing():
    _, payload = fixture()
    batches = split_copy_review_payload(payload)
    for index, batch in enumerate(batches):
        assert batch["selectedObjects"] == payload["selectedObjects"]
        assert batch["chapterStructure"] == payload["chapterStructure"]
        assert batch["batchScope"]["index"] == index
        assert batch["batchScope"]["count"] == len(batches)
        assert batch["batchScope"]["totalFieldCount"] == payload["expectedFieldCount"]
    batches[0]["selectedObjects"][0]["evidence"][0]["text"] = "changed"
    batches[0]["chapterStructure"][0]["objectIds"].append("b")
    assert batches[1]["selectedObjects"] == payload["selectedObjects"]
    assert batches[1]["chapterStructure"] == payload["chapterStructure"]


@pytest.mark.parametrize("limit", [0, -1, 7, True, 1.5, "6"])
def test_invalid_limits_do_not_silently_create_unbounded_batches(limit):
    _, payload = fixture()
    with pytest.raises(ValueError, match="invalid_copy_review_batch_size"):
        split_copy_review_payload(payload, max_fields=limit)


def test_smaller_requested_limit_splits_groups_without_omitting_fields():
    _, payload = fixture()
    batches = split_copy_review_payload(payload, max_fields=1)
    assert len(batches) == payload["expectedFieldCount"]
    assert all(batch["expectedFieldCount"] == 1 for batch in batches)


@pytest.mark.parametrize("mutate", [
    lambda data: data.update(expectedFieldCount=True),
    lambda data: data.update(expectedFieldCount=1),
    lambda data: data["publicCopyFields"].append(deepcopy(data["publicCopyFields"][0])),
    lambda data: data["publicCopyFields"][0].update(ownerObjectIds=[None]),
])
def test_malformed_partition_input_fails_explicitly(mutate):
    _, data = fixture()
    mutate(data)
    with pytest.raises(ValueError, match="invalid_copy_review_payload"):
        split_copy_review_payload(data)


def test_batch_can_pass_but_default_parser_still_requires_the_entire_frame():
    frame, payload = fixture()
    batch = split_copy_review_payload(payload)[0]
    result = parse_copy_review_batch(response(batch), frame, batch)
    assert result.review_passed and result.frame == frame
    full_result = parse_copy_review(response(batch), frame, batch)
    assert not full_result.review_passed and full_result.errors == ("incomplete_field_review",)


def test_patch_outside_batch_is_forbidden_even_if_path_exists_in_frame():
    frame, payload = fixture()
    batch = batch_for(split_copy_review_payload(payload), "/coreAnswer")
    edit = patch("/chapters/0/leadIn", "A scene in Port B")
    result = parse_copy_review_batch(response(batch, [edit]), frame, batch)
    assert not result.review_passed and result.frame == frame
    assert result.errors == ("invalid_or_duplicate_path",)


@pytest.mark.parametrize("mutate", [
    lambda row: row.update(original="stale copy"),
    lambda row: row.update(ownerObjectIds=["b"]),
    lambda row: row.update(allowedEvidenceIds=["a:description", "b:description"]),
])
def test_subset_cannot_change_original_owner_or_existing_evidence_binding(mutate):
    frame, payload = fixture()
    batch = batch_for(split_copy_review_payload(payload), "/curatorialBrief/objects/0/selectionRationale")
    mutate(batch["publicCopyFields"][0])
    result = parse_copy_review_batch(response(batch), frame, batch)
    assert not result.review_passed and result.errors == ("stale_review_payload",)


def test_other_object_evidence_stays_forbidden_inside_object_batch():
    frame, payload = fixture()
    batch = batch_for(split_copy_review_payload(payload), "/items/1/whySelected")
    edit = patch("/items/1/whySelected", "A scene in Port B",
                 evidence="b:description", quote="location is Port B")
    result = parse_copy_review_batch(response(batch, [edit]), frame, batch)
    assert result.errors == ("wrong_object_evidence_owner",) and result.frame == frame
    edit["path"] = "/curatorialBrief/objects/0/selectionRationale"
    result = parse_copy_review_batch(response(batch, [edit]), frame, batch)
    assert result.errors == ("replacement_exceeds_existing_evidence_binding",)


def test_merge_applies_valid_independent_patches_atomically_and_preserves_structure():
    frame, payload = fixture()
    original = deepcopy(frame)
    batches = split_copy_review_payload(payload)
    edits = [patch("/coreAnswer", frame["coreAnswer"]), patch("/chapters/0/leadIn", "A scene in Port B"),
             patch("/curatorialBrief/objects/0/selectionRationale", "A scene in Port B")]
    outputs = [response(batch, [edit for edit in edits if any(
        row["path"] == edit["path"] for row in batch["publicCopyFields"])]) for batch in batches]
    result = merge_copy_review_batches(outputs, frame, batches)
    assert result.review_passed and result.status == "revised" and len(result.changes) == 3
    expected = deepcopy(frame)
    expected["coreAnswer"] = expected["chapters"][0]["leadIn"] = "A scene in Port A"
    expected["curatorialBrief"]["objects"][0]["selectionRationale"] = "A scene in Port A"
    assert result.frame == expected and frame == original


def test_one_failed_batch_rolls_back_all_other_valid_changes():
    frame, payload = fixture()
    batches = split_copy_review_payload(payload)
    outputs = [response(batch) for batch in batches]
    core = batches.index(batch_for(batches, "/coreAnswer"))
    outputs[core] = response(batches[core], [patch("/coreAnswer", frame["coreAnswer"])])
    outputs[-1] = response(batches[-1], [patch("/items/0/whySelected", "wrong original")])
    result = merge_copy_review_batches(outputs, frame, batches)
    assert not result.review_passed and result.frame == frame and not result.changes
    assert result.errors[0].startswith(f"batch_{len(batches) - 1}:")


def test_unresolved_batch_is_not_promoted_to_overall_pass():
    frame, payload = fixture()
    batches = split_copy_review_payload(payload)
    outputs = [response(batch) for batch in batches]
    outputs[-1]["outcome"] = "unresolved"
    result = merge_copy_review_batches(outputs, frame, batches)
    assert result.status == "unresolved" and not result.review_passed and result.frame == frame


def test_full_exact_coverage_is_required_even_when_every_received_batch_passes():
    frame, payload = fixture()
    batches = split_copy_review_payload(payload)
    outputs = [response(batch) for batch in batches]
    assert merge_copy_review_batches(outputs[:-1], frame, batches).errors == ("incomplete_copy_review_batches",)
    assert merge_copy_review_batches(outputs[:-1], frame, batches[:-1]).errors == ("incomplete_copy_review_batches",)
    assert merge_copy_review_batches([], frame, []).errors == ("incomplete_copy_review_batches",)
    assert merge_copy_review_batches(outputs + [outputs[0]], frame, batches + [batches[0]]).errors == (
        "overlapping_copy_review_batches",)


@pytest.mark.parametrize("key", ["selectedObjects", "chapterStructure", "visitorQuestion"])
def test_changed_source_or_membership_context_between_batches_blocks_merge(key):
    frame, payload = fixture()
    batches = split_copy_review_payload(payload)
    batches[-1][key] = [] if key != "visitorQuestion" else "A different visitor question"
    result = merge_copy_review_batches([response(batch) for batch in batches], frame, batches)
    assert result.errors == ("inconsistent_copy_review_batch_context",) and result.frame == frame


def test_permutation_of_complete_pairs_does_not_change_merge_of_unchanged_pass_verdicts():
    frame, payload = fixture()
    batches = split_copy_review_payload(payload)
    batches.reverse()
    outputs = [response(batch) for batch in batches]
    result = merge_copy_review_batches(outputs, frame, batches)
    assert result.review_passed and result.status == "passed" and not result.changes
    assert result.frame == frame
