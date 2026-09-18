from __future__ import annotations

import asyncio
import base64
from typing import Any

from app.config import Settings
from app.providers.deepseek import DeepSeekProvider, VisionImage


def _install_fake_client(monkeypatch, captured: dict[str, Any]) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"items": []}'},
                    }
                ],
                "usage": {"completion_tokens": 4},
            }

    class FakeClient:
        def __init__(self, *, timeout: float) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> FakeResponse:
            captured.update(url=url, headers=headers, body=json)
            return FakeResponse()

    monkeypatch.setattr("app.providers.deepseek.httpx.AsyncClient", FakeClient)


def test_text_json_request_disables_thinking_and_bounds_output(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    _install_fake_client(monkeypatch, captured)
    provider = DeepSeekProvider(
        Settings(
            deepseek_api_key="test-key",
            deepseek_model="deepseek-flash",
        )
    )

    result = asyncio.run(provider.generate_json("system prompt", {"question": "test"}))

    assert result == {"items": []}
    body = captured["body"]
    assert body["model"] == "deepseek-flash"
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 4096


def test_retrieval_audit_request_is_deterministic(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    _install_fake_client(monkeypatch, captured)
    provider = DeepSeekProvider(
        Settings(
            deepseek_api_key="test-key",
            deepseek_model="deepseek-flash",
        )
    )

    result = asyncio.run(
        provider.generate_retrieval_audit_json(
            "retrieval audit",
            {"requiredCount": 5, "candidates": []},
        )
    )

    assert result == {"items": []}
    body = captured["body"]
    assert body["model"] == "deepseek-flash"
    assert body["temperature"] == 0.0
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 4096


def test_visual_label_request_uses_the_dedicated_model_and_bound_image(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}
    _install_fake_client(monkeypatch, captured)
    provider = DeepSeekProvider(
        Settings(
            deepseek_api_key="test-key",
            deepseek_model="deepseek-flash",
            deepseek_labels_model="deepseek-flash",
            deepseek_timeout_seconds=3,
        )
    )
    image_bytes = b"RIFF-test-fixture-WEBP"

    result = asyncio.run(
        provider.generate_json_with_images(
            "system prompt",
            {"items": [{"objectId": "aic:123"}]},
            [
                VisionImage(
                    object_id="aic:123",
                    evidence_id="image:aic:123",
                    payload=image_bytes,
                )
            ],
        )
    )

    assert result == {"items": []}
    body = captured["body"]
    assert body["model"] == "deepseek-flash"
    assert body["thinking"] == {"type": "disabled"}
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 1600

    system_message, user_message = body["messages"]
    assert system_message == {"role": "system", "content": "system prompt"}
    assert isinstance(user_message["content"], list)
    content = user_message["content"]
    assert [block["type"] for block in content] == [
        "text",
        "text",
        "image_url",
    ]
    assert "aic:123" in content[1]["text"]
    assert "image:aic:123" in content[1]["text"]
    data_url = content[2]["image_url"]["url"]
    prefix, encoded = data_url.split(",", 1)
    assert prefix == "data:image/webp;base64"
    assert base64.b64decode(encoded) == image_bytes
    assert content[2]["image_url"]["detail"] == "original"

    # Pixels are present only in the transient user message. They are not
    # duplicated into the structured payload or the system instruction.
    assert "base64" not in content[0]["text"]
    assert "base64" not in system_message["content"]
