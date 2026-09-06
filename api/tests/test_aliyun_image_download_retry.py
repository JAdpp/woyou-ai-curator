from __future__ import annotations

import asyncio
import io
from pathlib import Path

import httpx
import pytest
from PIL import Image

from app.providers import aliyun_image
from app.providers.aliyun_image import AliyunImageProvider, AliyunImageProviderError, PosterContext


ASSET_URL = "https://assets.test/poster.png?token=private"
buffer = io.BytesIO()
Image.new("RGB", (1536, 864), (64, 86, 108)).save(buffer, format="PNG")
PNG_BYTES = buffer.getvalue()


def provider_for(tmp_path: Path, handler, **kwargs) -> AliyunImageProvider:
    return AliyunImageProvider(
        api_key="fake-test-key",
        api_host="https://workspace.cn-beijing.maas.aliyuncs.com",
        output_dir=tmp_path,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


@pytest.mark.parametrize("failure", [429, 503, httpx.ConnectError, httpx.ReadTimeout])
def test_retry_download_reuses_url_and_never_regenerates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure
) -> None:
    monkeypatch.setattr(aliyun_image, "DOWNLOAD_RETRY_DELAY_SECONDS", 0)
    posts = 0
    downloads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.method == "POST":
            posts += 1
            return httpx.Response(200, json={
                "output": {"choices": [{"message": {"content": [{"image": ASSET_URL}]}}]},
            })
        downloads.append(str(request.url))
        if len(downloads) == 1:
            if isinstance(failure, int):
                return httpx.Response(failure)
            raise failure("transient failure", request=request)
        return httpx.Response(200, headers={"content-type": "image/png"}, content=PNG_BYTES)

    provider = provider_for(tmp_path, handler)
    result = asyncio.run(provider.generate_poster(PosterContext(
        exhibition_id="retry-poster", exhibition_theme="Landscape", title="Landscape",
        core_question="How do landscapes differ?",
    )))

    assert posts == 1
    assert downloads == [ASSET_URL, ASSET_URL]
    assert result.image_path.is_file()
    assert "token=private" not in result.metadata_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("failure", [401, 403, 404, httpx.UnsupportedProtocol, httpx.LocalProtocolError])
def test_non_transient_errors_are_not_retried(tmp_path: Path, failure) -> None:
    downloads = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal downloads
        downloads += 1
        if isinstance(failure, int):
            return httpx.Response(failure)
        raise failure("invalid request", request=request)

    provider = provider_for(tmp_path, handler)

    async def run() -> None:
        async with provider._client_context() as client:
            await provider._download_png(client, ASSET_URL)

    with pytest.raises(AliyunImageProviderError) as captured:
        asyncio.run(run())
    assert captured.value.code == "image_download_error"
    assert captured.value.retryable is False
    assert downloads == 1


@pytest.mark.parametrize(("headers", "body", "code"), [
    ({"content-type": "text/html"}, PNG_BYTES, "image_validation_error"),
    ({"content-type": "image/png"}, b"invalid", "image_validation_error"),
    ({"content-type": "image/png", "content-length": "invalid"}, PNG_BYTES, "image_validation_error"),
    ({"content-type": "image/png", "content-length": "999999999"}, PNG_BYTES, "image_too_large"),
])
def test_invalid_assets_are_not_retried(tmp_path: Path, headers, body, code: str) -> None:
    downloads = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal downloads
        downloads += 1
        return httpx.Response(200, headers=headers, content=body)

    provider = provider_for(tmp_path, handler)

    async def run() -> None:
        async with provider._client_context() as client:
            await provider._download_png(client, ASSET_URL)

    with pytest.raises(AliyunImageProviderError) as captured:
        asyncio.run(run())
    assert captured.value.code == code
    assert downloads == 1


def test_only_one_retry_even_if_both_attempts_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aliyun_image, "DOWNLOAD_RETRY_DELAY_SECONDS", 0)
    downloads = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal downloads
        downloads += 1
        return httpx.Response(503)

    provider = provider_for(tmp_path, handler)

    async def run() -> None:
        async with provider._client_context() as client:
            await provider._download_png(client, ASSET_URL)

    with pytest.raises(AliyunImageProviderError) as captured:
        asyncio.run(run())
    assert captured.value.http_status == 503
    assert captured.value.retryable is True
    assert downloads == 2


def test_total_download_deadline_includes_retry_backoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An individual GET fails immediately, but the backoff is longer than the
    # entire budget. No second GET may start or receive a fresh timeout.
    monkeypatch.setattr(aliyun_image, "DOWNLOAD_RETRY_DELAY_SECONDS", 1)
    downloads = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal downloads
        downloads += 1
        return httpx.Response(503)

    provider = provider_for(tmp_path, handler, timeout_seconds=0.02)

    async def run() -> None:
        async with provider._client_context() as client:
            await asyncio.wait_for(provider._download_png(client, ASSET_URL), timeout=0.5)

    with pytest.raises(AliyunImageProviderError) as captured:
        asyncio.run(run())
    assert captured.value.code == "image_download_error"
    assert "time limit" in str(captured.value)
    assert downloads == 1


def test_second_response_body_cannot_reset_total_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aliyun_image, "DOWNLOAD_RETRY_DELAY_SECONDS", 0)
    downloads = 0
    closed = []

    class PendingBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield PNG_BYTES[:20]
            await asyncio.Event().wait()

        async def aclose(self):
            closed.append(True)

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal downloads
        downloads += 1
        if downloads == 1:
            await asyncio.sleep(0.05)
            return httpx.Response(503)
        return httpx.Response(200, headers={"content-type": "image/png"}, stream=PendingBody())

    # Leave enough scheduling margin to reach the second response under the
    # full suite; the pending body still must be cancelled by the shared limit.
    provider = provider_for(tmp_path, handler, timeout_seconds=0.5)

    async def run() -> None:
        async with provider._client_context() as client:
            await asyncio.wait_for(provider._download_png(client, ASSET_URL), timeout=1.5)

    with pytest.raises(AliyunImageProviderError) as captured:
        asyncio.run(run())
    assert captured.value.code == "image_download_error"
    assert downloads == 2
    assert closed == [True]
