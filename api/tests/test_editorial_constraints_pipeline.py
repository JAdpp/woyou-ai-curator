from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace

from app import curation
from app.curatorial_copy_review import copy_review_payload, merge_copy_review_batches, split_copy_review_payload
from app.generator import ExhibitionGenerator
from app.models import VisitorProfile
from app.retrieval_agent import RetrievalQueryPlan
from .test_curatorial_copy_review import frame, objects, response
from .test_label_review import _CriticProvider, _fixture, _generator
from .test_frame_label_independence import _ImageCache, _Provider


BOUNDARIES = ("Distinguish what these records establish from what remains unknown.",
              "Do not infer an original owner from decoration alone.")


def test_editorial_boundaries_reach_label_writer_and_same_object_critic(client):
    exhibition, profile = _fixture(client)
    provider = _CriticProvider()
    count = asyncio.run(_generator(client, provider)._write_labels(
        exhibition, profile, editorial_constraints=BOUNDARIES))
    assert count == 1 and len(provider.calls) == 2
    for prompt, payload, _images in provider.calls:
        assert payload["editorialConstraints"] == list(BOUNDARIES)
        assert payload["visitorQuestion"] == exhibition.agenda.question
        assert "editorialConstraints" in prompt
        assert not any(boundary in source["text"] for boundary in BOUNDARIES
                       for source in payload["items"][0]["evidence"])


def test_copy_review_batches_preserve_one_editorial_context():
    original = frame()
    payload = copy_review_payload(original, objects(), question="Compare documented and unknown owners",
                                  editorial_constraints=BOUNDARIES)
    batches = split_copy_review_payload(payload)
    assert all(batch["editorialConstraints"] == list(BOUNDARIES) for batch in batches)
    assert merge_copy_review_batches([response(batch) for batch in batches], original, batches).review_passed
    altered = deepcopy(batches)
    altered[-1]["editorialConstraints"] = []
    result = merge_copy_review_batches([response(batch) for batch in altered], original, altered)
    assert not result.review_passed and "inconsistent_copy_review_batch_context" in result.errors


def test_profile_generation_passes_planned_editorial_to_all_writing_stages(client, monkeypatch):
    provider = _Provider()
    settings = replace(client.app.state.settings, deepseek_timeout_seconds=1,
                       deepseek_frame_timeout_seconds=1, deepseek_labels_timeout_seconds=.5)
    generator = ExhibitionGenerator(settings, client.app.state.collections, provider, _ImageCache())
    original_prepare = generator.prepare_initial_retrieval
    captures = []

    async def prepare(*args, **kwargs):
        result = await original_prepare(*args, **kwargs)
        plan = RetrievalQueryPlan(valid=True, in_collection_scope=True, search_queries=(),
                                  editorial_constraints=BOUNDARIES)
        return replace(result, query_plan=plan)

    async def text_call(prompt, payload):
        captures.append((prompt, deepcopy(payload)))
        if "publicCopyFields" in payload:
            return response(payload)
        # Copy fields exist, but later Brief application may reject this tiny
        # fixture. The independent label pass must retain the same boundaries.
        return {"title": "看记录，也看留白", "curatorialBrief": {}}

    monkeypatch.setattr(generator, "prepare_initial_retrieval", prepare)
    monkeypatch.setattr(provider, "generate_json", text_call)
    asyncio.run(generator.generate_from_profile(VisitorProfile(duration_minutes=5)))
    assert captures
    assert all(payload["editorialConstraints"] == list(BOUNDARIES) for _prompt, payload in captures)
    assert any("publicCopyFields" in payload for _prompt, payload in captures)
    assert len(provider.image_calls) == 10
    assert all(payload["editorialConstraints"] == list(BOUNDARIES) for payload, _images in provider.image_calls)
    assert all("editorialConstraints" in curation.frame_prompt(language) for language in ("zh", "en"))
