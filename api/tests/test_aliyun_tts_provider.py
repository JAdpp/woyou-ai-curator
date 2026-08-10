from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.providers.aliyun_tts import (
    AliyunTtsProvider,
    AliyunTtsProviderError,
    is_valid_mp3,
)


MP3_BYTES = b"\xff\xfb\x90\x64" + (b"\x00" * 256)
OSS_HTTP_URL = (
    "http://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/pre/tts/guide.mp3?token=test"
)


def _speech_response(url: str = OSS_HTTP_URL) -> dict[str, object]:
    return {
        "request_id": "tts-request-42",
        "output": {
            "finish_reason": "stop",
            "audio": {"data": "", "url": url, "expires_at": 1_800_000_000},
        },
        "usage": {"characters": 12},
    }


def test_qwen_tts_request_downloads_and_validates_mp3() -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.method == "POST":
            assert request.url.path == "/api/v1/services/audio/tts/SpeechSynthesizer"
            assert request.headers["authorization"] == "Bearer server-secret"
            body = json.loads(request.content)
            assert body == {
                "model": "qwen-audio-3.0-tts-plus",
                "input": {
                    "text": "欢迎来到卧游。",
                    "voice": "qwen-audio-3.0-tts-plus-longyulianrong",
                    "format": "mp3",
                    "sample_rate": 24000,
                    "rate": 0.94,
                    "pitch": 0.98,
                    "volume": 55,
                    "language_hints": ["zh"],
                    "instruction": "专业博物馆导览",
                    "seed": 7,
                    "enable_aigc_tag": True,
                },
            }
            return httpx.Response(200, json=_speech_response())

        # Official responses can contain an HTTP OSS URL. The provider must
        # deterministically upgrade the trusted host before downloading it.
        assert request.url.scheme == "https"
        assert request.url.host.endswith(".aliyuncs.com")
        return httpx.Response(
            200,
            content=MP3_BYTES,
            headers={"content-type": "audio/mpeg"},
        )

    provider = AliyunTtsProvider(
        api_key="server-secret",
        api_host="https://llm-workspace.cn-beijing.maas.aliyuncs.com",
        instruction="专业博物馆导览",
        retry_delay_seconds=0,
        transport=httpx.MockTransport(handler),
    )
    generated = asyncio.run(provider.synthesize("欢迎来到卧游。"))

    assert generated.audio_bytes == MP3_BYTES
    assert generated.request_id == "tts-request-42"
    assert generated.audio_format == "mp3"
    assert [request.method for request in observed] == ["POST", "GET"]
    assert is_valid_mp3(generated.audio_bytes)


def test_invalid_download_is_retried_once() -> None:
    generation_calls = 0
    download_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal generation_calls, download_calls
        if request.method == "POST":
            generation_calls += 1
            return httpx.Response(200, json=_speech_response())
        download_calls += 1
        return httpx.Response(
            200,
            content=b"<html>temporary upstream error</html>"
            if download_calls == 1
            else MP3_BYTES,
        )

    provider = AliyunTtsProvider(
        api_key="server-secret",
        api_host="https://llm-workspace.cn-beijing.maas.aliyuncs.com",
        retry_delay_seconds=0,
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(provider.synthesize("第二次应该成功。"))
    assert result.audio_bytes == MP3_BYTES
    assert generation_calls == 2
    assert download_calls == 2


def test_download_enforces_maximum_audio_bytes() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json=_speech_response())
        return httpx.Response(
            200,
            content=MP3_BYTES,
            headers={"content-length": str(len(MP3_BYTES))},
        )

    provider = AliyunTtsProvider(
        api_key="server-secret",
        api_host="https://llm-workspace.cn-beijing.maas.aliyuncs.com",
        max_audio_bytes=64,
        retry_delay_seconds=0,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(AliyunTtsProviderError) as caught:
        asyncio.run(provider.synthesize("过大的音频。"))
    assert caught.value.code == "audio_too_large"


def test_untrusted_http_audio_url_is_rejected_without_fetch() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_speech_response("http://127.0.0.1/private"))

    provider = AliyunTtsProvider(
        api_key="server-secret",
        api_host="https://llm-workspace.cn-beijing.maas.aliyuncs.com",
        retry_delay_seconds=0,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(AliyunTtsProviderError) as caught:
        asyncio.run(provider.synthesize("不能访问内网。"))
    assert caught.value.code == "provider_response_error"
    assert len(requests) == 1


def test_tts_host_falls_back_to_image_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALIYUN_TTS_API_HOST", raising=False)
    monkeypatch.setenv(
        "ALIYUN_IMAGE_API_HOST",
        "https://llm-shared.cn-beijing.maas.aliyuncs.com/",
    )
    settings = Settings.from_env()
    assert settings.aliyun_tts_api_host == (
        "https://llm-shared.cn-beijing.maas.aliyuncs.com"
    )


@pytest.mark.parametrize(
    "payload",
    [b"", b"ID3", b"{\"error\":true}", b"\xff\xfb\x00\x00"],
)
def test_mp3_validation_rejects_non_audio(payload: bytes) -> None:
    assert is_valid_mp3(payload) is False
