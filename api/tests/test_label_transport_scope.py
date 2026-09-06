from __future__ import annotations

import asyncio
from time import perf_counter

import pytest

from app import curation
from app.label_review import (
    LABEL_REVIEW_VERSION,
    MAX_REVIEW_EVIDENCE_CHARS,
    MAX_REVIEW_EVIDENCE_CHUNKS,
    VISUAL_SENTENCE_CONTRACT,
    checked_label_candidate,
    generate_label_with_transport_retry,
    label_review_payload,
    label_review_prompt,
)
from app.providers.deepseek import ProviderError, VisionImage
from .test_label_review import _CriticProvider, _fixture, _generator, _output


def test_writer_and_critic_each_retry_one_network_error_without_resetting_deadlines(client, monkeypatch):
    exhibition, profile = _fixture(client)
    attempts = {False: 0, True: 0}

    class NetworkHiccupProvider(_CriticProvider):
        async def generate_json_with_images(self, prompt, payload, images):
            is_review = payload.get("schemaVersion") == LABEL_REVIEW_VERSION
            attempts[is_review] += 1
            if attempts[is_review] == 1:
                await asyncio.sleep(.005)
                raise ProviderError("fixture connection error", code="provider_network_error")
            return await super().generate_json_with_images(prompt, payload, images)

    provider = NetworkHiccupProvider()
    generator = _generator(client, provider, budget=.1)
    original_call = generator._generate_model_json
    calls = []

    async def tracked(*args, **kwargs):
        calls.append((kwargs["stage"], perf_counter(), kwargs["timeout_seconds"], kwargs["vision_images"]))
        return await original_call(*args, **kwargs)

    monkeypatch.setattr(generator, "_generate_model_json", tracked)
    started = perf_counter()
    count = asyncio.run(generator._write_labels(exhibition, profile))
    assert count == 1 and attempts == {False: 2, True: 2}
    assert len(calls) == 4 and calls[1][0].endswith(":transport_retry") and calls[3][0].endswith(":transport_retry")
    for initial, retry in ((calls[0], calls[1]), (calls[2], calls[3])):
        assert retry[2] < initial[2]
        assert abs((initial[1] + initial[2]) - (retry[1] + retry[2])) < .003
        assert initial[3][0] is retry[3][0]
    assert max(at + budget for _, at, budget, _ in calls) - started < .11
    assert calls[0][3][0] is calls[2][3][0]


@pytest.mark.parametrize("code", ["provider_http_error", "provider_timeout", "provider_invalid_response", "provider_error"])
def test_label_retry_does_not_retry_http_refusal_timeout_or_parse_error(code):
    calls = []

    async def failing(*args, **kwargs):
        calls.append(kwargs)
        raise ProviderError("not retryable", code=code)

    with pytest.raises(ProviderError) as caught:
        asyncio.run(generate_label_with_transport_retry(failing, "prompt", {}, stage="labels:x",
            deadline=perf_counter() + .2, request_timeout=1, vision_images=[]))
    assert caught.value.code == code and len(calls) == 1


def test_two_network_failures_are_not_a_third_attempt():
    calls = []

    async def failing(*args, **kwargs):
        calls.append(kwargs)
        raise ProviderError("still unavailable", code="provider_network_error")

    with pytest.raises(ProviderError, match="still unavailable"):
        asyncio.run(generate_label_with_transport_retry(failing, "prompt", {}, stage="labels_review:x",
            deadline=perf_counter() + .2, request_timeout=1, vision_images=[]))
    assert len(calls) == 2


def test_expired_network_failure_cannot_start_a_retry(monkeypatch):
    calls = []
    now = [100.0]
    monkeypatch.setattr("app.label_review.perf_counter", lambda: now[0])

    async def late_failure(*args, **kwargs):
        calls.append(kwargs)
        now[0] += .015
        raise ProviderError("network deadline used up", code="provider_network_error")

    with pytest.raises(ProviderError):
        asyncio.run(generate_label_with_transport_retry(late_failure, "prompt", {}, stage="labels:x",
            deadline=100.005, request_timeout=1, vision_images=[]))
    assert len(calls) == 1


@pytest.mark.parametrize("text,blocked", [
    ("扇骨为象牙，顶部呈弧形。", True),
    ("象牙扇骨向外放射，边缘有空隙。", True),
    ("象牙白的表面上有深色线条。", False),
    ("Ivory guards surround a curved edge.", True),
    ("An ivory-coloured surface has dark lines.", False),
])
def test_controlled_catalogue_material_cannot_remain_in_final_visual_sentence(client, text, blocked):
    exhibition, profile = _fixture(client)
    item = exhibition.items[0]
    item.object.medium = "ivory"
    source = curation.image_evidence(item.object)
    image = VisionImage(object_id=item.object.id, evidence_id=source.id, payload=b"fixture-pixels")
    payload = curation.labels_payload(profile, exhibition.chapters[0], [item])
    output = _output(payload["items"][0], image, reviewed=True)
    output["items"][0]["labelSentences"][0]["text"] = text
    output["items"][0]["labelSentences"][1]["text"] = "馆方记录材质为象牙。"
    arguments = dict(max_chars=140,
        allowed_evidence_by_object={item.object.id: {chunk["id"] for chunk in payload["items"][0]["evidence"]}},
        visual_evidence_by_object={item.object.id: source.id})
    # A draft may be corrected once by the critic. Only final enforcement is
    # allowed to turn this scope defect into an explicit degraded label.
    checked_label_candidate(item, output, enforce_visual_scope=False, **arguments)
    if blocked:
        with pytest.raises(ValueError, match="catalogue material"):
            checked_label_candidate(item, output, **arguments)
    else:
        assert checked_label_candidate(item, output, **arguments).label_sentences[0].text == text


def test_critic_that_echoes_material_error_is_explicitly_degraded_not_retried(client):
    exhibition, profile = _fixture(client)
    exhibition.items[0].object.medium = "ivory"

    class EchoingCritic(_CriticProvider):
        async def generate_json_with_images(self, prompt, payload, images):
            output = await super().generate_json_with_images(prompt, payload, images)
            output["items"][0]["labelSentences"][0]["text"] = "象牙瓶身两侧向上收拢。"
            return output

    provider = EchoingCritic()
    count = asyncio.run(_generator(client, provider)._write_labels(exhibition, profile))
    assert count == 0 and len(provider.calls) == 2
    assert "尚未完成来源核对" in exhibition.items[0].label_sentences[0].text


def test_critic_receives_bounded_additional_original_context_and_its_own_citation_whitelist(client):
    exhibition, profile = _fixture(client)
    item = exhibition.items[0]
    extra_id = item.object.id + ":z-bottom-context"
    context = item.object.evidence[0].model_copy(update={
        "id": extra_id, "text": "The artist incised the bird's feet on the bottom of the box."})
    item.object.evidence.append(context)

    class MoreInformedCritic(_CriticProvider):
        async def generate_json_with_images(self, prompt, payload, images):
            output = await super().generate_json_with_images(prompt, payload, images)
            ids = {chunk["id"] for chunk in payload["items"][0]["evidence"]}
            if payload.get("schemaVersion") == LABEL_REVIEW_VERSION:
                assert extra_id in ids
                assert next(chunk for chunk in payload["items"][0]["evidence"] if chunk["id"] == extra_id)["criticOnly"]
                output["items"][0]["labelSentences"][1] = {"text": "馆方说明底面刻有鸟足，而非羽纹。",
                    "type": "system_inference", "evidenceIds": [extra_id]}
            else:
                assert extra_id not in ids  # Never pretend the writer saw it.
            return output

    provider = MoreInformedCritic()
    count = asyncio.run(_generator(client, provider)._write_labels(exhibition, profile))
    assert count == 1 and exhibition.items[0].label_sentences[1].evidence_ids == [extra_id]

    for index in range(8):
        item.object.evidence.append(context.model_copy(update={"id": f"{item.object.id}:z-{index}", "text": "x" * 950}))
    source = curation.image_evidence(item.object)
    image = VisionImage(object_id=item.object.id, evidence_id=source.id, payload=b"fixture-pixels")
    writer = curation.labels_payload(profile, exhibition.chapters[0], [item])
    review = label_review_payload(item, writer, _output(writer["items"][0], image), image, max_chars=80)
    chunks = review["items"][0]["evidence"]
    assert len(chunks) == MAX_REVIEW_EVIDENCE_CHUNKS
    assert all(len(chunk["text"]) <= MAX_REVIEW_EVIDENCE_CHARS for chunk in chunks)
    assert any(chunk["truncated"] and len(chunk["text"]) == 900 for chunk in chunks)


def test_writer_and_critic_visual_contract_avoids_anatomical_sides_and_missing_parts():
    assert "一只手/另一只手" in VISUAL_SENTENCE_CONTRACT
    assert "画面左侧/右侧" in VISUAL_SENTENCE_CONTRACT
    assert "不写材质、制作技法、人物身份、年代" in VISUAL_SENTENCE_CONTRACT
    assert VISUAL_SENTENCE_CONTRACT in label_review_prompt(80)
    assert "truncated=true" in label_review_prompt(80)
