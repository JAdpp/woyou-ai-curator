from __future__ import annotations

import asyncio
import io
import threading
from dataclasses import replace
from time import perf_counter

import pytest
from PIL import Image

from app import curation
from app.generator import ExhibitionGenerator
from app.images import ImageCache
from app.models import VisitorProfile
from app.visual_evidence import _BoundedImageExecutor


class _ImageCache:
    def get_cached(self, _url, _size):
        # This fixture represents already available pixels. Never compete with
        # a previous TLS test's still-running, correctly bounded downloads.
        return b"fixture-pixels"

    def get(self, _url, _size):
        return b"fixture-pixels", True


class _Provider:
    configured = True

    def __init__(self, *, invalid_count=False, bad_source=False, label_delay=0):
        self.invalid_count = invalid_count
        self.bad_source = bad_source
        self.label_delay = label_delay
        self.image_calls = []
        self.cancelled = 0

    async def generate_json(self, _prompt, payload):
        if "publicCopyFields" in payload:
            return {"schemaVersion": "curatorial-copy-review-v1", "outcome": "pass",
                    "reviewedFieldCount": len(payload["publicCopyFields"]), "changes": []}
        return {"title": "共4件作品", "curatorialBrief": {}} if self.invalid_count else {}

    async def generate_json_with_images(self, _prompt, payload, images):
        self.image_calls.append((payload, images))
        try:
            if self.label_delay:
                await asyncio.sleep(self.label_delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        item = payload["items"][0]
        image_id = "image:foreign" if self.bad_source else images[0].evidence_id
        return {"items": [{"objectId": item["objectId"], "displayTitle": "山水册页", "labelSentences": [
            {"text": "画面上方留白，墨色集中在下半部。", "type": "visual_observation", "evidenceIds": [image_id]},
            {"text": "可以沿着墨色的聚散慢慢观看。", "type": "system_inference", "evidenceIds": [item["evidence"][0]["id"]]},
        ]}]}


def _generate(client, provider, *, label_budget=.5, emit=None, image_cache=None):
    settings = replace(client.app.state.settings, deepseek_timeout_seconds=1,
                       deepseek_frame_timeout_seconds=.05, deepseek_labels_timeout_seconds=label_budget)
    generator = ExhibitionGenerator(settings, client.app.state.collections, provider, image_cache or _ImageCache())
    return asyncio.run(generator.generate_from_profile(VisitorProfile(duration_minutes=5), emit=emit))


@pytest.mark.parametrize("invalid_count", [False, True])
def test_failed_frame_still_generates_source_bound_visual_labels(client, monkeypatch, invalid_count):
    provider = _Provider(invalid_count=invalid_count)
    findings = {}
    applied_count = []
    if invalid_count:
        def apply_bad_frame(exhibition, output):
            applied_count.append(output["title"])
            return exhibition.model_copy(deep=True, update={"title": output["title"]})

        # Isolate the real public item-count validator from unrelated brief
        # completeness rules. No production validator/review is replaced.
        monkeypatch.setattr(curation, "apply_frame", apply_bad_frame)

    async def emit(key, detail):
        findings[key] = detail

    exhibition = _generate(client, provider, emit=emit)
    assert len(provider.image_calls) == 10  # Writer + independent source critic.
    assert all(len(images) == 1 and images[0].payload == b"fixture-pixels"
               for _payload, images in provider.image_calls)
    assert exhibition.versions.provider == "deterministic_fallback"
    assert exhibition.versions.labels_model == client.app.state.settings.deepseek_labels_model
    assert "5/5" in findings["labels"]
    assert exhibition.status == "ready" and exhibition.validation.passed
    assert all(any(sentence.type == "visual_observation" for sentence in item.label_sentences)
               for item in exhibition.items)
    assert any("展览框架使用确定性模板" in note for note in exhibition.coverage_limits)
    if invalid_count:
        assert applied_count == ["共4件作品"]
        assert exhibition.title != "共4件作品"


def test_failed_frame_does_not_relax_visual_source_binding(client):
    provider = _Provider(bad_source=True)
    exhibition = _generate(client, provider)
    assert len(provider.image_calls) == 5
    assert exhibition.versions.provider == "deterministic_fallback"
    assert all("image:foreign" not in sentence.evidence_ids
               for item in exhibition.items for sentence in item.label_sentences)
    assert all(not any(sentence.type == "visual_observation" for sentence in item.label_sentences)
               for item in exhibition.items)


def test_independent_labels_keep_the_original_shared_stage_budget(client):
    provider = _Provider(label_delay=.3)
    started = perf_counter()
    exhibition = _generate(client, provider, label_budget=.05)
    assert perf_counter() - started < .25
    assert len(provider.image_calls) == 5 and provider.cancelled == 5
    assert exhibition.versions.provider == "deterministic_fallback"
    assert exhibition.versions.labels_model is None
    assert exhibition.status == "ready"


def test_missing_pixels_cannot_be_described_under_an_inference_type(client):
    class MissingImage:
        def get(self, *_args, **_kwargs):
            raise OSError("not available")

    class ImagelessProvider(_Provider):
        async def generate_json_with_images(self, _prompt, payload, images):
            assert not images
            item = payload["items"][0]
            return {"items": [{"objectId": item["objectId"], "labelSentences": [
                {"text": "扇骨为木质或竹质。", "type": "system_inference",
                 "evidenceIds": [item["evidence"][0]["id"]]},
            ]}]}

    exhibition = _generate(client, ImagelessProvider(), image_cache=MissingImage())
    assert all("未能载入图像" in item.label_sentences[0].text for item in exhibition.items)
    assert all(item.label_sentences[0].id for item in exhibition.items)
    assert not any("木质" in sentence.text for item in exhibition.items for sentence in item.label_sentences)


def test_labels_reuse_prewarmed_preview_when_download_workers_are_full(client, tmp_path, monkeypatch):
    cache = ImageCache(tmp_path / "cache")
    buffer = io.BytesIO()
    Image.new("RGB", (50, 40), (50, 60, 70)).save(buffer, "WEBP")
    for obj in client.app.state.collections.get().objects:
        obj.image_url_large = f"{obj.image_url}?original=1"
        # This is the preview-source 1024 variant retained by prewarm.
        cache._store(obj.image_url, 1024, buffer.getvalue())
    monkeypatch.setattr(cache, "get", lambda *_args, **_kwargs: pytest.fail("labels should use preview cache only"))
    release = threading.Event()
    executor = _BoundedImageExecutor(max_workers=4)
    entered = [threading.Event() for _ in range(4)]
    workers = [executor.submit(lambda event=event: (event.set(), release.wait(timeout=2))) for event in entered]
    assert all(event.wait(timeout=.3) for event in entered)
    monkeypatch.setattr("app.visual_evidence._IMAGE_EXECUTOR", executor)
    provider = _Provider()
    try:
        exhibition = _generate(client, provider, image_cache=cache)
        assert len(provider.image_calls) == 10
        assert all(len(images) == 1 and images[0].payload == buffer.getvalue()
                   for _payload, images in provider.image_calls)
        assert all(curation.image_evidence(item.object).source_url == item.object.image_url
                   for item in exhibition.items)
        assert exhibition.validation.passed and exhibition.status == "ready"
    finally:
        release.set()
        for worker in workers:
            worker.result(timeout=1)
