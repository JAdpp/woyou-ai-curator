from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from time import perf_counter

import pytest

from app import curation
from app.generator import ExhibitionGenerator
from app.label_review import (
    LABEL_REVIEW_VERSION,
    WRITER_BUDGET_FRACTION,
    checked_label_candidate,
    label_review_payload,
    label_review_prompt,
)
from app.models import VisitorProfile
from app.providers.deepseek import ProviderError, VisionImage
from .test_frame_label_independence import _ImageCache, _Provider, _generate


class _CachedImageFixture(_ImageCache):
    def get_cached(self, _url, _size):
        # These tests exercise label review, not a network download. In a
        # combined run another timeout fixture may still own the bounded
        # download workers; already-available pixels must use the cache path.
        return b"fixture-pixels"


def _fixture(client, *, count=1):
    offline = _Provider()
    offline.configured = False
    exhibition = _generate(client, offline)
    exhibition.items = exhibition.items[:count]
    for item in exhibition.items:
        item.object.title = "Flask"
        item.object.title_original = ""
        item.object.medium = "Metal"
        item.object.type = "Vessel"
        item.object.evidence[0].text = "This metal flask would have held perfume. The metal is not specified."
    return exhibition, VisitorProfile(duration_minutes=5)


def _output(record, image, *, reviewed=False):
    return {"items": [{
        "objectId": record["objectId"],
        "displayTitle": "金属小瓶" if reviewed else "铁制香水瓶",
        "localizedMetadata": {"medium": {"sourceValue": record["medium"], "zh": "金属" if reviewed else "铁"}},
        "labelSentences": [
            {"text": "瓶身两侧弧线向上收拢，顶部的开口较窄。" if reviewed else "铁制瓶身两侧弧线向上收拢，顶部开口较窄。",
             "type": "visual_observation", "evidenceIds": [image.evidence_id]},
            {"text": "馆方推测这只金属小瓶可能用于盛放香水。" if reviewed else "这只铁瓶曾用来盛放香水。",
             "type": "uncertain" if reviewed else "system_inference",
             "evidenceIds": [record["evidence"][0]["id"]]},
        ],
    }]}


class _CriticProvider:
    configured = True

    def __init__(self, *, fail_review=None, writer_delay=0, review_delay=0):
        self.calls = []
        self.fail_review = fail_review
        self.writer_delay = writer_delay
        self.review_delay = review_delay
        self.cancelled = []

    async def generate_json_with_images(self, prompt, payload, images):
        review = payload.get("schemaVersion") == LABEL_REVIEW_VERSION
        self.calls.append((prompt, deepcopy(payload), images))
        try:
            await asyncio.sleep(self.review_delay if review else self.writer_delay)
        except asyncio.CancelledError:
            self.cancelled.append(review)
            raise
        if review and self.fail_review == "provider":
            raise ProviderError("fixture critic unavailable")
        output = _output(payload["items"][0], images[0], reviewed=review)
        if review and self.fail_review == "wrong_object":
            output["items"][0]["objectId"] = "foreign-object"
        elif review and self.fail_review == "wrong_source":
            output["items"][0]["labelSentences"][1]["evidenceIds"] = ["foreign-source"]
        elif review and self.fail_review == "wrong_image":
            output["items"][0]["labelSentences"][0]["evidenceIds"] = ["image:foreign"]
        elif review and self.fail_review == "fact":
            output["items"][0]["labelSentences"][1]["type"] = "institution_fact"
        elif review and self.fail_review == "empty":
            output["items"] = []
        elif review and self.fail_review == "malformed":
            output["items"][0]["labelSentences"].append({"text": "ignored junk"})
        return output


def _generator(client, provider, *, budget=.5):
    return ExhibitionGenerator(
        replace(client.app.state.settings, deepseek_timeout_seconds=1, deepseek_labels_timeout_seconds=budget),
        client.app.state.collections, provider, _CachedImageFixture(),
    )


def test_only_independently_revised_prose_and_translations_are_committed(client):
    exhibition, profile = _fixture(client, count=2)
    provider = _CriticProvider()
    original_records = [item.object.model_dump() for item in exhibition.items]
    count = asyncio.run(_generator(client, provider)._write_labels(exhibition, profile))
    assert count == 2 and len(provider.calls) == 4
    for item in exhibition.items:
        calls = [call for call in provider.calls if call[1]["items"][0]["objectId"] == item.object.id]
        writer, critic = calls
        assert writer[0] != critic[0] and critic[1]["schemaVersion"] == LABEL_REVIEW_VERSION
        assert writer[2][0] is critic[2][0]  # Same actual pixels, not a new fetch.
        assert len(critic[1]["items"]) == 1
        assert "curatorialBrief" not in critic[1] and "chapter" not in critic[1]
        assert "objectDecision" not in critic[1]["items"][0]
        image_info = critic[1]["items"][0]["imageEvidence"]
        assert image_info["imageSha256"] == sha256(b"fixture-pixels").hexdigest()
        assert image_info["sourceUrl"] == curation.image_evidence(item.object).source_url
        assert any("would have held" in chunk["text"] for chunk in critic[1]["items"][0]["evidence"])
        assert "铁制" in critic[1]["draftLabel"]["displayTitle"]
        assert item.display_title == "金属小瓶"
        assert item.localized_metadata.medium == "金属"
        assert "铁" not in "".join(sentence.text for sentence in item.label_sentences)
        assert item.label_sentences[1].type == "uncertain" and "可能" in item.label_sentences[1].text
        assert all(chunk["id"].startswith(item.object.id) for chunk in critic[1]["items"][0]["evidence"])
    assert [item.object.model_dump() for item in exhibition.items] == original_records


@pytest.mark.parametrize("failure", ["provider", "wrong_object", "wrong_source", "wrong_image", "fact", "empty", "malformed"])
def test_failed_critic_does_not_leak_draft_prose_title_or_translation(client, failure):
    exhibition, profile = _fixture(client)
    item = exhibition.items[0]
    original_title, original_metadata = item.display_title, item.localized_metadata.model_dump()
    provider = _CriticProvider(fail_review=failure)
    count = asyncio.run(_generator(client, provider)._write_labels(exhibition, profile))
    assert len(provider.calls) == 2 and count == 0
    assert item.display_title == original_title and item.localized_metadata.model_dump() == original_metadata
    assert len(item.label_sentences) == 1 and "尚未完成来源核对" in item.label_sentences[0].text
    assert item.label_sentences[0].type == "uncertain"
    assert "铁" not in item.label_sentences[0].text


def test_writer_and_critic_share_one_absolute_stage_budget(client, monkeypatch):
    exhibition, profile = _fixture(client)
    # Keep the same budget ratios with scheduler-sized headroom on Windows.
    # A 14 ms writer margin can expire under unrelated CPU load before the
    # critic starts, testing scheduler jitter instead of the shared deadline.
    provider = _CriticProvider(writer_delay=.3, review_delay=2)
    generator = _generator(client, provider, budget=.8)
    budgets = []
    original_generate = generator._generate_model_json

    async def tracked(*args, **kwargs):
        budgets.append((kwargs["stage"], kwargs["timeout_seconds"], perf_counter()))
        return await original_generate(*args, **kwargs)

    monkeypatch.setattr(generator, "_generate_model_json", tracked)
    started = perf_counter()
    count = asyncio.run(generator._write_labels(exhibition, profile))
    assert count == 0 and perf_counter() - started < 1.6
    assert len(budgets) == 2 and budgets[0][1] <= .8 * WRITER_BUDGET_FRACTION
    assert budgets[1][1] < .6  # The review did not receive a fresh .8 s.
    assert max(at + budget for _, budget, at in budgets) - started <= .9
    assert provider.cancelled == [True]
    assert "尚未完成来源核对" in exhibition.items[0].label_sentences[0].text


def test_one_reviewer_failure_does_not_discard_other_objects(client):
    exhibition, profile = _fixture(client, count=2)
    failed_id = exhibition.items[0].object.id

    class PartiallyAvailableCritic(_CriticProvider):
        async def generate_json_with_images(self, prompt, payload, images):
            output = await super().generate_json_with_images(prompt, payload, images)
            if payload.get("schemaVersion") == LABEL_REVIEW_VERSION and images[0].object_id == failed_id:
                raise ProviderError("one object could not be reviewed")
            return output

    provider = PartiallyAvailableCritic()
    count = asyncio.run(_generator(client, provider)._write_labels(exhibition, profile))
    assert count == 1 and len(provider.calls) == 4
    assert "尚未完成来源核对" in exhibition.items[0].label_sentences[0].text
    assert exhibition.items[1].label_sentences[0].type == "visual_observation"
    assert exhibition.items[1].localized_metadata.medium == "金属"


def test_missing_image_keeps_only_reviewed_metadata_and_never_counts_visual_success(client):
    exhibition, profile = _fixture(client)
    calls = []

    class MissingImage:
        def get(self, *_args, **_kwargs):
            raise OSError("no pixels")

    class MetadataCritic:
        configured = True

        async def generate_json_with_images(self, prompt, payload, images):
            calls.append(payload)
            assert images == [] and payload["items"][0]["imageEvidence"] is None
            record = payload["items"][0]
            reviewed = payload.get("schemaVersion") == LABEL_REVIEW_VERSION
            return {"items": [{"objectId": record["objectId"],
                "localizedMetadata": {"medium": {"sourceValue": "Metal", "zh": "金属" if reviewed else "铁"}},
                "labelSentences": [{"text": "馆方推测可能用于盛放香水。", "type": "uncertain",
                                    "evidenceIds": [record["evidence"][0]["id"]]}]}]}

    generator = _generator(client, MetadataCritic())
    generator.image_cache = MissingImage()
    count = asyncio.run(generator._write_labels(exhibition, profile))
    assert count == 0 and len(calls) == 2
    assert exhibition.items[0].localized_metadata.medium == "金属"
    assert "未能载入图像" in exhibition.items[0].label_sentences[0].text
    assert all(sentence.type != "visual_observation" for sentence in exhibition.items[0].label_sentences)


def test_cancellation_of_a_pending_critic_propagates_without_draft_commit(client):
    exhibition, profile = _fixture(client)
    provider = _CriticProvider(review_delay=10)
    generator = _generator(client, provider)
    original = exhibition.items[0].model_dump()

    async def run():
        task = asyncio.create_task(generator._write_labels(exhibition, profile))

        async def wait_for_critic():
            while len(provider.calls) < 2:
                if task.done():
                    await task  # Surface an early failure, not an infinite wait.
                    raise AssertionError("label task ended without reaching the critic")
                await asyncio.sleep(.001)

        try:
            await asyncio.wait_for(wait_for_critic(), timeout=.3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert not [other for other in asyncio.all_tasks() if other is not asyncio.current_task() and not other.done()]

    asyncio.run(run())
    assert provider.cancelled == [True] and exhibition.items[0].model_dump() == original


def test_review_payload_reconstructs_original_sources_and_checks_image_ownership(client):
    exhibition, profile = _fixture(client)
    item = exhibition.items[0]
    image_source = curation.image_evidence(item.object)
    image = VisionImage(object_id=item.object.id, evidence_id=image_source.id, payload=b"fixture-pixels")
    payload = curation.labels_payload(profile, exhibition.chapters[0], [item])
    draft = _output(payload["items"][0], image)
    # Reconstructed source text cannot silently become model-authored text.
    payload["items"][0]["evidence"][0]["text"] = "forged writer source"
    review = label_review_payload(item, payload, draft, image, max_chars=profile.label_max_chars)
    assert "forged writer source" not in str(review["items"])
    assert "would have held perfume" in str(review["items"])
    wrong_image = VisionImage(object_id="other", evidence_id=image_source.id, payload=b"fixture-pixels")
    with pytest.raises(ValueError, match="ownership"):
        label_review_payload(item, payload, draft, wrong_image, max_chars=profile.label_max_chars)


def test_shape_check_is_transactional_and_prompt_covers_nonvisual_claims(client):
    exhibition, profile = _fixture(client)
    item = exhibition.items[0]
    source = curation.image_evidence(item.object)
    image = VisionImage(object_id=item.object.id, evidence_id=source.id, payload=b"fixture-pixels")
    payload = curation.labels_payload(profile, exhibition.chapters[0], [item])
    original = item.model_dump()
    candidate = checked_label_candidate(item, _output(payload["items"][0], image), max_chars=profile.label_max_chars,
        allowed_evidence_by_object={item.object.id: {chunk["id"] for chunk in payload["items"][0]["evidence"]}},
        visual_evidence_by_object={item.object.id: source.id})
    assert candidate.localized_metadata.medium == "铁" and item.model_dump() == original
    # Structural validation alone cannot catch these entailment errors. This
    # test deliberately demonstrates why a separate semantic critic is needed.
    prompt = label_review_prompt(120)
    assert "Metal" in prompt and "would have held" in prompt
    assert "部位与主体归属" in prompt and "夹入 visual_observation" in prompt
    assert "localizedMetadata" in prompt and "同一件馆方原文" in prompt
