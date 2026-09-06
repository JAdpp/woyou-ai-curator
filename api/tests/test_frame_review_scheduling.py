from __future__ import annotations

import asyncio
from time import perf_counter

import pytest

from app.generator import ExhibitionGenerator
from app.providers.deepseek import ProviderError
from .test_curatorial_copy_review import frame, payload, response


def test_all_batches_use_two_slots_and_merge_atomically(client, monkeypatch):
    generator = client.app.state.generator
    calls, active, peak = [], 0, 0

    async def model(_prompt, batch, *, stage, timeout_seconds):
        nonlocal active, peak
        assert stage.startswith("frame_review:batch")
        assert 0 < timeout_seconds <= 12
        active += 1
        peak = max(peak, active)
        calls.append(batch)
        try:
            await asyncio.sleep(.01)
            return response(batch)
        finally:
            active -= 1

    monkeypatch.setattr(generator, "_generate_model_json", model)
    original = frame()
    result = asyncio.run(generator._review_frame_copy(
        original, payload(original), language="en", deadline=perf_counter() + 1))
    assert result.review_passed and result.frame == original
    assert peak == 2 and active == 0 and len(calls) >= 3
    assert sum(batch["expectedFieldCount"] for batch in calls) == payload()["expectedFieldCount"]


def test_deadline_cancels_active_and_queued_batches_without_fresh_budget(client, monkeypatch):
    generator = client.app.state.generator
    calls, active = [], set()

    async def model(_prompt, batch, **kwargs):
        index = batch["batchScope"]["index"]
        calls.append(index)
        active.add(index)
        try:
            await asyncio.sleep(1)
            return response(batch)
        finally:
            active.remove(index)

    monkeypatch.setattr(generator, "_generate_model_json", model)
    started = perf_counter()
    with pytest.raises(ProviderError, match="deadline"):
        asyncio.run(generator._review_frame_copy(
            frame(), payload(), language="en", deadline=started + .04))
    assert perf_counter() - started < .3
    assert len(calls) == 2 and not active


def test_unresolved_batch_is_not_retried_or_silently_accepted(client, monkeypatch):
    generator = client.app.state.generator
    calls = []

    async def model(_prompt, batch, **kwargs):
        calls.append(kwargs["stage"])
        output = response(batch)
        if batch["batchScope"]["index"] == 0:
            output["outcome"] = "unresolved"
        return output

    monkeypatch.setattr(generator, "_generate_model_json", model)
    result = asyncio.run(generator._review_frame_copy(
        frame(), payload(), language="en", deadline=perf_counter() + 5))
    assert not result.review_passed and result.frame == frame()
    assert not any("repair" in stage for stage in calls)


def test_failed_object_relation_is_replaced_not_kept_or_reported_as_model_pass(client, monkeypatch):
    generator = client.app.state.generator
    calls = []

    async def model(_prompt, batch, **kwargs):
        calls.append(kwargs["stage"])
        output = response(batch)
        if batch["batchScope"]["group"][0] == "object":
            output["outcome"] = "unresolved"
        return output

    monkeypatch.setattr(generator, "_generate_model_json", model)
    original = frame()
    result = asyncio.run(generator._review_frame_copy(
        original, payload(original), language="zh", deadline=perf_counter() + 5))
    assert result.review_passed and result.status == "bounded_local_fallback"
    assert result.locally_neutralized_fields == 2
    assert result.frame["title"] == original["title"]
    decision = result.frame["curatorialBrief"]["objects"][0]
    assert "Paris" not in decision["selectionRationale"]
    assert "later" not in decision["relation"]
    assert decision["evidenceIds"] == original["curatorialBrief"]["objects"][0]["evidenceIds"]
    assert original == frame()
    assert all(not row["evidenceIds"] for row in result.changes)


def test_model_cannot_self_declare_programmatic_fallback(client, monkeypatch):
    generator = client.app.state.generator

    async def model(_prompt, batch, **_kwargs):
        return {**response(batch), "localNeutralization": True}

    monkeypatch.setattr(generator, "_generate_model_json", model)
    result = asyncio.run(generator._review_frame_copy(
        frame(), payload(), language="en", deadline=perf_counter() + 5))
    assert result.review_passed and result.locally_neutralized_fields == 0


def test_source_names_come_from_selected_objects_not_collection_header(client):
    from app.models import VisitorProfile
    repository = client.app.state.collections
    collection = repository.get()
    profile = VisitorProfile(duration_minutes=5)
    agenda = profile.to_agenda(collection.id)
    results = repository.search(agenda, collection)
    selected = [row.obj.model_copy(update={"institution": "Selected Museum"}) for row in results[:5]]
    exhibition = client.app.state.generator._profile_skeleton(
        profile, agenda, collection, selected, None, results)
    assert "Selected Museum" in exhibition.coverage_limits[0]
    if collection.institution != "Selected Museum":
        assert collection.institution not in exhibition.coverage_limits[0]
