from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import VisitorProfile
from app.providers.aliyun_image import (
    AliyunImageProviderError,
    GeneratedPoster,
    PosterContext,
)

from .conftest import write_collection


def _agenda() -> dict[str, object]:
    return {
        "question": "山水画如何组织观看者的行旅视线？",
        "priorKnowledge": "略有了解",
        "durationMinutes": 10,
        "excludedTopics": [],
    }


def _client(tmp_path: Path) -> tuple[TestClient, Path]:
    collections_dir = tmp_path / "collections"
    write_collection(collections_dir)
    output_dir = tmp_path / "posters"
    settings = Settings(
        app_env="test",
        collections_dir=collections_dir,
        store_mode="memory",
        store_path=tmp_path / "unused.json",
        deepseek_api_key=None,
        admin_review_token="test-admin-token",
        aliyun_image_output_dir=output_dir,
    )
    return TestClient(create_app(settings)), output_dir


class _SuccessfulProvider:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.calls: list[PosterContext] = []

    async def generate_poster(self, context: PosterContext) -> GeneratedPoster:
        self.calls.append(context)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.output_dir / "fixture.png"
        metadata_path = self.output_dir / "fixture.json"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        metadata_path.write_text("{}\n", encoding="utf-8")
        return GeneratedPoster(
            provider="aliyun-model-studio",
            model="qwen-image-3.0-pro",
            size="1536*864",
            image_path=image_path,
            metadata_path=metadata_path,
            sha256="fixture-sha256",
            seed=context.seed,
            generated_at="2026-08-06T00:00:00Z",
            request_id="request-fixture",
        )


class _FailingProvider:
    async def generate_poster(self, _: PosterContext) -> GeneratedPoster:
        raise AliyunImageProviderError(
            "provider_network_error",
            "safe test error",
            retryable=True,
        )


def _generate(client: TestClient) -> dict[str, object]:
    response = client.post("/api/exhibitions/generate-sync", json={"agenda": _agenda()})
    assert response.status_code == 200, response.text
    return response.json()


def test_poster_endpoint_persists_local_asset_and_is_idempotent(tmp_path: Path) -> None:
    client, output_dir = _client(tmp_path)
    fake = _SuccessfulProvider(output_dir)
    client.app.state.image_provider = fake
    exhibition = _generate(client)

    response = client.post(f"/api/exhibitions/{exhibition['id']}/poster")

    assert response.status_code == 200, response.text
    poster = response.json()["poster"]
    assert poster["status"] == "ready"
    assert poster["backgroundUrl"] == "/generated/posters/fixture.png"
    assert poster["provider"] == "aliyun-model-studio"
    assert poster["model"] == "qwen-image-3.0-pro"
    assert poster["isAiGenerated"] is True
    assert "不代表任何馆藏实物" in poster["altText"]
    assert "准确题名" in poster["promptSummary"]
    assert fake.calls[0].exhibition_theme == exhibition["exhibitionTheme"]
    assert fake.calls[0].title == exhibition["title"]
    assert fake.calls[0].subtitle == exhibition["subtitle"]
    assert fake.calls[0].seed is not None

    asset = client.get(poster["backgroundUrl"])
    assert asset.status_code == 200
    assert asset.content.startswith(b"\x89PNG\r\n\x1a\n")

    repeated = client.post(f"/api/exhibitions/{exhibition['id']}/poster")
    assert repeated.status_code == 200
    assert repeated.json()["poster"] == poster
    assert len(fake.calls) == 1

    regenerated = client.post(
        f"/api/exhibitions/{exhibition['id']}/poster?force=true"
    )
    assert regenerated.status_code == 200
    assert regenerated.json()["poster"]["status"] == "ready"
    assert len(fake.calls) == 2


def test_poster_endpoint_reports_missing_server_configuration(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    exhibition = _generate(client)

    response = client.post(f"/api/exhibitions/{exhibition['id']}/poster")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "not_configured"
    stored = client.get(f"/api/exhibitions/{exhibition['id']}").json()
    assert stored["poster"]["status"] == "failed"
    assert stored["poster"]["errorCode"] == "not_configured"


def test_poster_failure_keeps_exhibition_available_with_fallback(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    client.app.state.image_provider = _FailingProvider()
    exhibition = _generate(client)

    response = client.post(f"/api/exhibitions/{exhibition['id']}/poster")

    assert response.status_code == 502
    assert response.json()["detail"] == {
        "code": "provider_network_error",
        "message": (
            "The AI poster could not be generated; the exhibition remains available with "
            "its collection-image fallback."
        ),
        "retryable": True,
    }
    stored = client.get(f"/api/exhibitions/{exhibition['id']}").json()
    assert stored["poster"]["status"] == "failed"
    assert stored["poster"]["errorCode"] == "provider_network_error"
    assert len(stored["items"]) == 5


class _FrameProvider:
    configured = True

    def __init__(self) -> None:
        self.calls = 0

    async def generate_json(self, _system_prompt: str, _payload: dict) -> dict:
        self.calls += 1
        if self.calls == 1:
            items = [
                item for chapter in _payload["chapters"] for item in chapter["items"]
            ]
            evidence_ids = [item["evidence"][0]["id"] for item in items]
            return {
                "title": "模型确定的最终题名",
                "subtitle": "模型确定的最终副题",
                "curatorialBrief": {
                    "bigIdea": {
                        "text": "由馆藏证据组织一条可追溯的观看路径。",
                        "evidenceIds": evidence_ids[:2],
                        "confidence": "provisional",
                    },
                    "keyMessages": [
                        {
                            "text": "这些记录从不同位置回应访客问题。",
                            "evidenceIds": evidence_ids[:2],
                            "confidence": "provisional",
                        }
                    ],
                    "criticalQuestions": ["证据说明了什么？", "仍有哪些空白？"],
                    "objects": [
                        {
                            "objectId": item["objectId"],
                            "role": item["role"],
                            "selectionRationale": "以可定位的馆方记录支撑本位置。",
                            "relation": "与相邻藏品形成有边界的比较。",
                            "evidenceIds": [item["evidence"][0]["id"]],
                        }
                        for item in items
                    ],
                    "evaluationTargets": [
                        {
                            "statement": "访客能够复述本展的核心问题。",
                            "method": "comprehension_check",
                        }
                    ],
                },
            }
        return {"items": []}


def test_poster_callback_observes_final_frame_title(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    generator = client.app.state.generator
    generator.provider = _FrameProvider()
    observed: list[tuple[str, str]] = []

    async def capture_frame(exhibition) -> None:
        observed.append((exhibition.title, exhibition.subtitle))

    profile = VisitorProfile(
        curiosity_label="山水与行旅",
        free_form_question="山水画如何组织观看者的行旅视线？",
        prior_knowledge="some",
        duration_minutes=5,
    )
    exhibition = asyncio.run(
        generator.generate_from_profile(profile, on_frame_ready=capture_frame)
    )

    assert observed == [("模型确定的最终题名", "模型确定的最终副题")]
    assert exhibition.title == observed[0][0]
