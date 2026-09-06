from __future__ import annotations

import asyncio
import io
import time
from dataclasses import replace

from PIL import Image
import pytest

from app.generator import AgenticRetrievalOutcome, ExhibitionGenerator, InitialRetrievalOutcome
from app.images import ImageCache
from app.models import VisitorProfile
from app.retrieval_agent import RetrievalQueryPlan
from .test_frame_label_independence import _ImageCache, _Provider


def test_record_explanation_selected_images_warm_during_frame_for_all_five_labels(client, tmp_path, monkeypatch):
    from app.visual_evidence import _BoundedImageExecutor
    # This test measures overlap with its own cold downloads. Capacity pressure
    # is covered separately; unrelated timed-out TLS fixtures may still own
    # global workers after their request correctly returned.
    monkeypatch.setattr("app.visual_evidence._IMAGE_EXECUTOR", _BoundedImageExecutor())
    repository = client.app.state.collections
    collection = repository.get()
    results = repository.search(VisitorProfile(duration_minutes=5).to_agenda(collection.id), collection)
    cache = ImageCache(tmp_path / "cache")
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (30, 60, 90)).save(buffer, "WEBP")
    downloads = []

    def download_fixture(url, _edge, **_kwargs):
        downloads.append(url)
        time.sleep(.04)  # Longer than the later label image sub-budget.
        return buffer.getvalue()

    monkeypatch.setattr(cache, "_render", download_fixture)

    class FrameProvider(_Provider):
        async def generate_json(self, _prompt, _payload):
            # Observe completion, not a fragile 160ms scheduling guess on a
            # loaded Windows machine. The model stage is the overlap window.
            finish = time.perf_counter() + .6
            while sum(cache.get_cached(url, 1024) is not None for url in downloads) < 5:
                if time.perf_counter() >= finish:
                    pytest.fail("selected image prewarm did not finish during frame")
                await asyncio.sleep(.01)
            return {}  # Keep fallback-frame independence in this integration.

    provider = FrameProvider()
    settings = replace(client.app.state.settings, deepseek_timeout_seconds=2,
                       deepseek_frame_timeout_seconds=1.5, deepseek_labels_timeout_seconds=.1)
    generator = ExhibitionGenerator(settings, repository, provider, cache)
    plan = RetrievalQueryPlan(valid=True, in_collection_scope=True, search_queries=(), evidence_mode="record_explanation")

    async def initial(*_args, **_kwargs):
        return InitialRetrievalOutcome(results=results, query_plan=plan)

    async def audited(*_args, **kwargs):
        assert kwargs["initial_query_plan"].evidence_mode == "record_explanation"
        return AgenticRetrievalOutcome(results=results, audit_applied=True, answerability="supported")

    monkeypatch.setattr(generator, "prepare_initial_retrieval", initial)
    monkeypatch.setattr(generator, "_agentic_retrieve", audited)
    started = time.perf_counter()
    exhibition = asyncio.run(generator.generate_from_profile(VisitorProfile(
        duration_minutes=5, free_form_question="馆方记录如何解释这些山水作品", curiosity_label="山水记录")))
    assert time.perf_counter() - started < 1.5
    assert len(downloads) == 5
    assert len(provider.image_calls) == 10  # Both passes reuse the warmed pixels.
    assert all(len(images) == 1 and images[0].payload == buffer.getvalue()
               for _payload, images in provider.image_calls)
    assert exhibition.validation.passed and exhibition.status == "ready"
    assert exhibition.versions.provider == "deterministic_fallback"


@pytest.mark.parametrize("cancel_visit", [False, True])
def test_frame_exit_or_cancellation_cleans_selected_prewarm_async_task(client, monkeypatch, cancel_visit):
    async def run():
        started, cancelled = asyncio.Event(), asyncio.Event()
        prewarm_calls = []

        async def pending_prewarm(objects, _cache, **kwargs):
            prewarm_calls.append((objects, kwargs))
            started.set()
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.set()

        class FrameProvider(_Provider):
            async def generate_json(self, _prompt, _payload):
                await started.wait()
                if cancel_visit:
                    await asyncio.sleep(10)
                return {}

        monkeypatch.setattr("app.generator.prewarm_visual_candidates", pending_prewarm)
        settings = replace(client.app.state.settings, deepseek_timeout_seconds=1,
                           deepseek_frame_timeout_seconds=.3, deepseek_labels_timeout_seconds=.1)
        provider = FrameProvider()
        generator = ExhibitionGenerator(settings, client.app.state.collections, provider, _ImageCache())
        task = asyncio.create_task(generator.generate_from_profile(VisitorProfile(duration_minutes=5)))
        await started.wait()
        if cancel_visit:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not provider.image_calls
        else:
            exhibition = await task
            assert exhibition.status == "ready" and len(provider.image_calls) == 10
        assert cancelled.is_set()
        assert len(prewarm_calls) == 1 and len(prewarm_calls[0][0]) == 5
        assert prewarm_calls[0][1] == {"max_candidates": 12, "timeout_seconds": 12.0}
        assert not [other for other in asyncio.all_tasks() if other is not asyncio.current_task() and not other.done()]

    asyncio.run(run())
