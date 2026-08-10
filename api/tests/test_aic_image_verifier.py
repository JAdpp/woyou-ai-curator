from __future__ import annotations

from types import SimpleNamespace

from scripts.sources import base


class _Response:
    def __init__(self, status_code: int, headers: dict[str, str]) -> None:
        self.status_code = status_code
        self.headers = headers

    def close(self) -> None:
        return None

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _Client:
    def __init__(self) -> None:
        self.head_headers: dict[str, str] = {}
        self.stream_headers: dict[str, str] = {}

    def head(self, _url: str, headers: dict[str, str]) -> _Response:
        self.head_headers = headers
        return _Response(403, {"content-type": "text/html"})

    def stream(
        self, _method: str, _url: str, headers: dict[str, str]
    ) -> _Response:
        self.stream_headers = headers
        return _Response(
            206,
            {
                "content-type": "image/jpeg",
                "content-length": "1024",
                "content-range": "bytes 0-1023/2048",
            },
        )

    def close(self) -> None:
        return None


def test_aic_verifier_sends_documented_project_header_on_head_and_get(
    monkeypatch,
) -> None:
    pooled = _Client()
    monkeypatch.setattr(base.httpx, "Client", lambda **_: pooled)
    obj = SimpleNamespace(
        image_url="https://www.artic.edu/iiif/2/example/full/843,/0/default.jpg",
        object_url="https://www.artic.edu/artworks/123/example",
        image_validation=None,
    )

    kept = base.verify_images(
        [obj], base.HttpClient(min_interval=0), workers=1
    )

    assert kept == [obj]
    assert "inquiry-curator" in pooled.head_headers["AIC-User-Agent"].casefold()
    assert "Referer" not in pooled.head_headers
    assert "Referer" not in pooled.stream_headers
    assert "inquiry-curator" in pooled.stream_headers["AIC-User-Agent"].casefold()
    assert pooled.stream_headers["Range"] == "bytes=0-1023"
    assert obj.image_validation["method"] == "GET_RANGE"
