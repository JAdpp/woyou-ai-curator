from __future__ import annotations

import asyncio
import io
import threading
from dataclasses import replace
from hashlib import sha256
from time import monotonic

import pytest
from PIL import Image

from app.images import ImageCache, ImageFetchError
from app.models import MuseumObject
from app.providers.deepseek import ProviderError
from app.visual_evidence import (
    VISUAL_AUDIT_PROMPT,
    _BoundedImageExecutor,
    admit_visual_predicates,
    audit_visual_candidates,
    prewarm_visual_candidates,
    visual_core_proof,
)


def _object(number: int = 1) -> MuseumObject:
    return MuseumObject(id=f"cma:{number}", title="Basket", image_url=f"https://museum.test/{number}.jpg",
                        object_url=f"https://museum.test/object/{number}", rights="CC0")


class _Cache:
    def __init__(self):
        self.calls = []

    def get(self, url, size):
        self.calls.append((url, size))
        return b"actual-image-test-bytes", True


class _Provider:
    labels_model = "fixture-vision"

    def __init__(self, mutate=None):
        self.calls = []
        self.mutate = mutate

    async def generate_json_with_images(self, prompt, payload, images):
        self.calls.append((prompt, payload, images))
        rows = []
        for item in payload["items"]:
            output = {"objectId": item["objectId"], "imageEvidenceId": item["imageEvidenceId"],
                      "questionRelevance": "supported", "observations": [
                          {"aspect": "shape", "text": "开口较宽、底部较窄", "location": "器物整体轮廓",
                           "predicateIds": [row["id"] for row in payload["visualPredicates"]]}],
                      "uncertainties": ["当前角度看不到底部"]}
            if self.mutate:
                self.mutate(output)
            rows.append(output)
        return {"items": rows}


def _run(provider=None, **kwargs):
    return asyncio.run(audit_visual_candidates("比较篮子的器形", [_object()], provider or _Provider(), _Cache(), **kwargs))


def test_real_bytes_binding_and_visual_only_role_proof():
    provider, obj = _Provider(), _object()
    frozen_before = obj.model_dump()
    report = asyncio.run(audit_visual_candidates("看器形", [obj], provider, _Cache(),
                                               visual_predicates=[{"id": "p1", "kind": "visual", "text": "可见器形"}]))
    row = report.results[0]
    assert row.status == "reviewed" and row.image_supplied and row.image_reviewed
    assert row.image_sha256 == sha256(b"actual-image-test-bytes").hexdigest()
    assert row.observations[0].predicate_ids == ("p1",)
    assert row.observations[0].id.startswith("visual:cma:1:")
    assert provider.calls[0][2][0].payload == b"actual-image-test-bytes"
    assert provider.calls[0][2][0].object_id == obj.id
    assert obj.model_dump() == frozen_before
    assert obj.evidence_depth == "thin"
    proof = visual_core_proof(row, evidence_mode="visual_observation")
    assert proof["scope"] == "visible_features_only"
    assert visual_core_proof(row, evidence_mode="record_explanation") is None
    assert visual_core_proof(row, evidence_mode="open_exploration") is None
    assert visual_core_proof(replace(row, image_reviewed=False), evidence_mode="visual_observation") is None
    assert visual_core_proof(replace(row, image_sha256="f" * 63 + "x"), evidence_mode="visual_observation") is None
    payload = report.to_payload()
    assert "actual-image-test-bytes" not in str(payload)
    assert payload["results"][0]["sourceKind"] == "collection_image"


def test_disabled_and_missing_cache_do_not_claim_image_review():
    provider = _Provider()
    assert _run(provider, enabled=False).results == ()
    report = asyncio.run(audit_visual_candidates("test", [_object()], provider, None))
    assert report.results[0].status == "unavailable"
    assert not report.results[0].image_supplied
    assert not report.results[0].image_reviewed
    assert provider.calls == []


def test_material_history_and_untyped_predicates_cannot_be_admitted():
    admitted = admit_visual_predicates([
        {"id": "p1", "kind": "material", "text": "gold"},
        {"id": "p2", "kind": "date", "text": "Ming"},
        {"id": "p3", "kind": "historical", "text": "ritual"},
        {"id": "p4", "text": "unknown type"},
        {"id": "p5", "kind": "object_kind", "text": "basket"},
        {"id": "p6", "kind": "visual_feature", "text": "handle visible"},
    ])
    assert [row["id"] for row in admitted] == ["p5", "p6"]
    assert "actual material" in VISUAL_AUDIT_PROMPT
    assert "A depiction" in VISUAL_AUDIT_PROMPT


@pytest.mark.parametrize("mutate", [
    lambda o: o.update(objectId="another-object"),
    lambda o: o.update(imageEvidenceId="image:another"),
    lambda o: o["observations"][0].update(aspect="material"),
    lambda o: o["observations"][0].update(predicateIds=["not-admitted"]),
    lambda o: o["observations"][0].update(location=""),
    lambda o: o["observations"][0].update(text="x" * 401),
    lambda o: o.update(observations=[]),
    lambda o: o.update(material="gold"),
])
def test_invalid_or_nonvisual_contract_fails_closed(mutate):
    row = _run(_Provider(mutate)).results[0]
    assert row.status == "invalid_response"
    assert row.image_supplied and not row.image_reviewed
    assert row.observations == ()
    assert visual_core_proof(row, evidence_mode="visual_observation") is None


def test_uncertain_image_is_reviewed_but_not_core_evidence():
    row = _run(_Provider(lambda o: o.update(questionRelevance="uncertain"))).results[0]
    assert row.image_reviewed
    assert visual_core_proof(row, evidence_mode="visual_observation") is None


def test_image_error_does_not_call_model_or_expose_raw_error():
    class BadCache:
        def get(self, *_):
            raise ImageFetchError("upstream_unreachable", "secret-error-url")

    provider = _Provider()
    report = asyncio.run(audit_visual_candidates("test", [_object()], provider, BadCache()))
    assert not provider.calls
    assert not report.results[0].image_supplied
    assert not report.results[0].image_reviewed
    assert "secret-error-url" not in str(report.to_payload())


def test_provider_failure_is_not_claimed_as_completed_review():
    class BadProvider(_Provider):
        async def generate_json_with_images(self, *_):
            raise ProviderError("raw-secret-error", code="provider_network_error")

    report = _run(BadProvider())
    assert report.results[0].image_supplied and not report.results[0].image_reviewed
    assert report.results[0].error_code == "provider_network_error"
    assert "raw-secret-error" not in str(report.to_payload())


def test_model_calls_use_shared_deadline_and_concurrency_bound():
    class SlowProvider(_Provider):
        def __init__(self):
            super().__init__()
            self.active = self.peak = self.cancelled = 0

        async def generate_json_with_images(self, *_):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
            finally:
                self.active -= 1

    provider = SlowProvider()
    started = monotonic()
    report = asyncio.run(audit_visual_candidates("test", [_object(i) for i in range(12)],
                                                provider, _Cache(), max_candidates=12, timeout_seconds=0.07))
    assert monotonic() - started < 0.5
    assert len(report.results) == 12
    assert all(row.status == "timed_out" for row in report.results)
    assert provider.peak <= 2 and provider.cancelled == 2 and provider.active == 0
    assert not any(row.image_reviewed for row in report.results)


def test_candidate_cap_deduplication_and_zero_limit():
    provider, cache = _Provider(), _Cache()
    objects = [_object(i) for i in range(20)]
    report = asyncio.run(audit_visual_candidates("test", [objects[0], *objects], provider, cache,
                                                max_candidates=100))
    assert len(report.results) == 12
    assert len(provider.calls) == 3
    assert len(cache.calls) == 12
    assert all(len(call[2]) <= 4 for call in provider.calls)
    assert _run(max_candidates=0).results == ()


def test_cancelled_job_propagates_without_late_model_work():
    entered = asyncio.Event()

    class SlowProvider(_Provider):
        async def generate_json_with_images(self, *_):
            entered.set()
            await asyncio.sleep(5)

    async def run():
        task = asyncio.create_task(audit_visual_candidates("test", [_object()], SlowProvider(), _Cache()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


def test_timed_out_fetch_retains_bounded_capacity(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    executor = _BoundedImageExecutor(max_workers=1)
    monkeypatch.setattr("app.visual_evidence._IMAGE_EXECUTOR", executor)

    class SlowCache:
        def get(self, *_):
            entered.set()
            release.wait(timeout=1)
            return b"bytes", True

    try:
        # Give the worker time to enter on loaded CI/Windows hosts. Its release
        # remains blocked beyond the audit deadline, which is the invariant.
        report = asyncio.run(audit_visual_candidates("test", [_object()], _Provider(), SlowCache(), timeout_seconds=0.3))
        assert entered.is_set()
        assert report.results[0].status == "timed_out"
        with pytest.raises(ImageFetchError, match="occupied"):
            executor.submit(lambda: None)
    finally:
        release.set()


def test_dedicated_batch_provider_is_preferred():
    class BatchProvider(_Provider):
        async def generate_qrel_vision_json(self, prompt, payload, images):
            return await super().generate_json_with_images(prompt, payload, images)

        async def generate_json_with_images(self, *_):
            raise AssertionError("Should use dedicated deterministic 4096-token batch method")

    provider = BatchProvider()
    report = asyncio.run(audit_visual_candidates("test", [_object(i) for i in range(8)],
                                                provider, _Cache(), max_candidates=8))
    assert len(provider.calls) == 2
    assert all(row.status == "reviewed" for row in report.results)


@pytest.mark.parametrize("bad_case", ["missing", "duplicate", "wrong_evidence", "invalid_aspect"])
def test_one_bad_batch_object_does_not_fail_other_objects(bad_case):
    class PartialProvider(_Provider):
        async def generate_json_with_images(self, *args):
            output = await super().generate_json_with_images(*args)
            if bad_case == "missing":
                output["items"].pop(0)
            elif bad_case == "duplicate":
                output["items"].append(dict(output["items"][0]))
            elif bad_case == "wrong_evidence":
                output["items"][0]["imageEvidenceId"] = "image:cma:2"
            else:
                output["items"][0]["observations"][0]["aspect"] = "history"
            return output

    report = asyncio.run(audit_visual_candidates("test", [_object(i) for i in range(4)],
                                                PartialProvider(), _Cache()))
    assert report.results[0].status == "invalid_response"
    assert not report.results[0].image_reviewed
    assert all(row.status == "reviewed" for row in report.results[1:])


def test_missing_image_is_excluded_from_batch_pixels():
    class PartialCache(_Cache):
        def get(self, url, size):
            if url.endswith("/0.jpg"):
                raise ImageFetchError("upstream_unreachable", "fixture")
            return super().get(url, size)

    provider = _Provider()
    report = asyncio.run(audit_visual_candidates("test", [_object(i) for i in range(4)],
                                                provider, PartialCache()))
    assert report.results[0].status == "unavailable"
    assert not report.results[0].image_supplied
    assert len(provider.calls) == 1 and len(provider.calls[0][2]) == 3
    assert all(image.object_id != "cma:0" for image in provider.calls[0][2])
    assert all(row.status == "reviewed" for row in report.results[1:])


def test_preselection_prefers_preview_and_small_cached_image():
    obj = _object()
    obj.image_url_large = "https://museum.test/huge-original.jpg"
    cache = _Cache()
    report = asyncio.run(audit_visual_candidates("test", [obj], _Provider(), cache))
    assert cache.calls == [(obj.image_url, 512)]
    assert report.results[0].source_url == obj.image_url


def test_slow_neighbour_does_not_use_ready_images_model_budget(monkeypatch):
    release = threading.Event()
    monkeypatch.setattr("app.visual_evidence._IMAGE_EXECUTOR", _BoundedImageExecutor(max_workers=4))
    started = monotonic()

    class MixedCache:
        def get(self, url, _size):
            if not url.endswith("/0.jpg"):
                release.wait(timeout=2)
            return b"actual-image-test-bytes", True

    class ModelNeedsTime(_Provider):
        async def generate_json_with_images(self, *args):
            self.called_at = monotonic() - started
            await asyncio.sleep(0.25)
            return await super().generate_json_with_images(*args)

    provider = ModelNeedsTime()
    try:
        report = asyncio.run(audit_visual_candidates("test", [_object(i) for i in range(4)],
                                                    provider, MixedCache(), timeout_seconds=0.6))
        assert provider.called_at < 0.2
        assert report.results[0].status == "reviewed"
        assert len(provider.calls[0][2]) == 1
        assert all(row.status == "timed_out" and not row.image_supplied for row in report.results[1:])
        assert all(row.error_code == "visual_image_deadline_exceeded" for row in report.results[1:])
        assert monotonic() - started < 0.6
    finally:
        release.set()


def test_prewarm_is_model_free_bounded_and_passes_shared_deadline():
    class DeadlineCache(_Cache):
        supports_deadline = True

        def get(self, url, size, *, deadline):
            self.calls.append((url, size, deadline))
            return b"actual-image-test-bytes", True

    cache = DeadlineCache()
    before = monotonic()
    report = asyncio.run(prewarm_visual_candidates([_object(i) for i in range(20)], cache,
                                                  max_candidates=8, timeout_seconds=.5))
    assert report["cachedCount"] == 8 and len(cache.calls) == 8
    assert all(size == 512 and before < deadline <= before + .6 for _url, size, deadline in cache.calls)
    assert len({deadline for _url, _size, deadline in cache.calls}) == 1
    assert all("imageReviewed" not in row for row in report["results"])


def test_review_reuses_cached_large_url_but_reports_that_actual_source():
    class VariantCache(_Cache):
        def has_cached(self, url, size):
            return "large" in url

    obj = _object()
    obj.image_url_large = "https://museum.test/large-image.jpg"
    cache = VariantCache()
    report = asyncio.run(audit_visual_candidates("test", [obj], _Provider(), cache))
    assert cache.calls == [(obj.image_url_large, 512)]
    assert report.results[0].source_url == obj.image_url_large


def test_full_download_pool_cannot_block_cached_prewarm_or_visual_review(tmp_path, monkeypatch):
    release = threading.Event()
    executor = _BoundedImageExecutor(max_workers=4)
    entered = [threading.Event() for _ in range(4)]
    workers = [executor.submit(lambda event=event: (event.set(), release.wait(timeout=2))) for event in entered]
    assert all(event.wait(timeout=.3) for event in entered)
    monkeypatch.setattr("app.visual_evidence._IMAGE_EXECUTOR", executor)
    cache = ImageCache(tmp_path / "cache")
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (60, 70, 80)).save(buffer, "WEBP")
    objects = [_object(i) for i in range(8)]
    for obj in objects:
        cache._store(obj.image_url, 1024, buffer.getvalue())
    monkeypatch.setattr(cache, "get", lambda *_args, **_kwargs: pytest.fail("cached review cannot enter download API"))
    provider = _Provider()
    try:
        warmed = asyncio.run(prewarm_visual_candidates(objects, cache, max_candidates=8, timeout_seconds=.5))
        report = asyncio.run(audit_visual_candidates("test", objects, provider, cache,
                                                    max_candidates=8, timeout_seconds=.5))
        assert warmed["cachedCount"] == 8
        assert len(provider.calls) == 2
        assert all(row.status == "reviewed" and row.image_reviewed for row in report.results)
        assert all(len(call[2]) == 4 for call in provider.calls)
    finally:
        release.set()
        for worker in workers:
            worker.result(timeout=1)
