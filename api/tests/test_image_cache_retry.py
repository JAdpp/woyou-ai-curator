from __future__ import annotations

import io
import urllib.error

from PIL import Image

from api.app.images import ImageCache


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_image_cache_retries_transient_upstream_failure(monkeypatch, tmp_path) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (20, 80, 140)).save(buffer, format="PNG")
    calls = 0

    def urlopen(request, timeout):  # noqa: ANN001, ARG001
        nonlocal calls
        calls += 1
        if calls < 3:
            raise urllib.error.HTTPError(
                request.full_url, 503, "temporary", {}, None
            )
        return _Response(buffer.getvalue())

    monkeypatch.setattr("api.app.images.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("api.app.images.time.sleep", lambda _: None)

    payload, cached = ImageCache(tmp_path / "cache").get(
        "https://example.org/object.png", 512
    )
    assert calls == 3
    assert cached is False
    assert payload.startswith(b"RIFF") and b"WEBP" in payload[:16]


def test_image_cache_identifies_aic_iiif_request(monkeypatch, tmp_path) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (80, 40, 20)).save(buffer, format="JPEG")
    seen_headers: dict[str, str] = {}

    def urlopen(request, timeout):  # noqa: ANN001, ARG001
        seen_headers.update(
            {name.casefold(): value for name, value in request.header_items()}
        )
        return _Response(buffer.getvalue())

    monkeypatch.setattr("api.app.images.urllib.request.urlopen", urlopen)

    ImageCache(tmp_path / "cache").get(
        "https://www.artic.edu/iiif/2/example/full/512,/0/default.jpg", 512
    )

    assert "referer" not in seen_headers
    assert "inquiry-curator" in seen_headers["aic-user-agent"].casefold()
