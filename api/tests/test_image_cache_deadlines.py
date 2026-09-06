from __future__ import annotations

import io
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from app.images import ImageCache, ImageFetchError


def _image(size=(1000, 700)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (70, 90, 110)).save(buffer, "WEBP")
    return buffer.getvalue()


def test_smaller_request_reuses_existing_larger_variant_without_network(tmp_path, monkeypatch):
    cache = ImageCache(tmp_path / "cache")
    cache._store("https://museum.test/image", 1024, _image())
    monkeypatch.setattr(cache, "_render", lambda *_args, **_kwargs: pytest.fail("must reuse cached larger image"))
    payload, hit = cache.get("https://museum.test/image", 512, deadline=time.monotonic() + .3)
    assert hit
    with Image.open(io.BytesIO(payload)) as image:
        assert max(image.size) == 512
    assert cache.has_cached("https://museum.test/image", 512)
    assert not cache.has_cached("https://museum.test/different-image", 512)


def test_new_preview_retains_label_sized_variant_without_second_download(tmp_path, monkeypatch):
    cache = ImageCache(tmp_path / "cache")
    calls = []

    def render(url, edge, **_kwargs):
        calls.append((url, edge))
        return _image()

    monkeypatch.setattr(cache, "_render", render)
    preview, hit = cache.get("https://museum.test/image", 512)
    label, label_hit = cache.get("https://museum.test/image", 1024)
    assert calls == [("https://museum.test/image", 1024)]
    assert not hit and label_hit
    assert cache.has_cached("https://museum.test/image", 1024)
    with Image.open(io.BytesIO(preview)) as image:
        assert max(image.size) == 512
    with Image.open(io.BytesIO(label)) as image:
        assert max(image.size) == 1000


def test_concurrent_same_source_joins_one_download(tmp_path, monkeypatch):
    cache = ImageCache(tmp_path / "cache")
    calls = []

    def render(url, edge, **_kwargs):
        calls.append((url, edge))
        time.sleep(.04)
        return _image()

    monkeypatch.setattr(cache, "_render", render)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(cache.get, "https://museum.test/image", 512,
                               deadline=time.monotonic() + 1) for _ in range(4)]
        results = [future.result(timeout=2) for future in futures]
    assert len(calls) == 1
    assert len({payload for payload, _cached in results}) == 1
    assert cache._inflight == {}


def test_waiter_deadline_does_not_cancel_owner_or_start_another_download(tmp_path, monkeypatch):
    cache = ImageCache(tmp_path / "cache")
    entered, release = threading.Event(), threading.Event()
    calls = []

    def render(url, edge, **_kwargs):
        calls.append((url, edge))
        entered.set()
        release.wait(timeout=1)
        return _image()

    monkeypatch.setattr(cache, "_render", render)
    with ThreadPoolExecutor(max_workers=1) as pool:
        owner = pool.submit(cache.get, "https://museum.test/image", 512, deadline=time.monotonic() + 1)
        assert entered.wait(timeout=.3)
        try:
            with pytest.raises(ImageFetchError) as raised:
                cache.get("https://museum.test/image", 512, deadline=time.monotonic() + .02)
            assert raised.value.code == "image_deadline_exceeded"
        finally:
            release.set()
        assert owner.result(timeout=1)[0]
    assert len(calls) == 1


def test_aic_lock_wait_and_pacing_use_callers_remaining_deadline(tmp_path, monkeypatch):
    cache = ImageCache(tmp_path / "cache")
    monkeypatch.setattr(cache, "_download", lambda *_args, **_kwargs: pytest.fail("must not download"))
    url = "https://www.artic.edu/iiif/2/fixture/full/843,/0/default.jpg"
    cache._aic_request_lock.acquire()
    try:
        with pytest.raises(ImageFetchError) as raised:
            cache.get(url, 512, deadline=time.monotonic() + .02)
        assert raised.value.code == "image_deadline_exceeded"
    finally:
        cache._aic_request_lock.release()
    cache._aic_last_request_at = time.monotonic()
    with pytest.raises(ImageFetchError) as raised:
        cache.get(url, 512, deadline=time.monotonic() + .02)
    assert raised.value.code == "image_deadline_exceeded"


def test_short_deadline_cannot_restart_three_thirty_second_attempts(tmp_path, monkeypatch):
    cache = ImageCache(tmp_path / "cache")
    calls = []

    def fail(request, timeout):
        calls.append(timeout)
        raise urllib.error.URLError("fixture connection failure")

    monkeypatch.setattr("app.images.urllib.request.urlopen", fail)
    with pytest.raises(ImageFetchError) as raised:
        cache.get("https://museum.test/image", 512, deadline=time.monotonic() + .1)
    assert raised.value.code == "image_deadline_exceeded"
    assert len(calls) == 1 and 0 < calls[0] <= .100001
    assert not cache._inflight


def test_deadline_rechecked_after_slow_headers_before_body_read(tmp_path, monkeypatch):
    cache = ImageCache(tmp_path / "cache")
    body = _image()

    class Response(io.BytesIO):
        def read(self, *_args):
            pytest.fail("expired response must not start reading body")

    def delayed(_request, timeout):
        time.sleep(.03)
        return Response(body)

    monkeypatch.setattr("app.images.urllib.request.urlopen", delayed)
    with pytest.raises(ImageFetchError) as raised:
        cache.get("https://museum.test/image", 512, deadline=time.monotonic() + .02)
    assert raised.value.code == "image_deadline_exceeded"


def test_expired_deadline_can_still_serve_already_cached_public_image(tmp_path):
    cache = ImageCache(tmp_path / "cache")
    cache._store("https://museum.test/image", 512, _image((40, 30)))
    assert cache.get("https://museum.test/image", 512, deadline=time.monotonic() - 1)[1]


def test_cache_only_read_never_downloads_waits_or_writes_derived_variant(tmp_path, monkeypatch):
    cache = ImageCache(tmp_path / "cache")
    url = "https://museum.test/image"
    cache._store(url, 1024, _image())
    monkeypatch.setattr(cache, "get", lambda *_args, **_kwargs: pytest.fail("cache-only must not enter network API"))
    monkeypatch.setattr(cache, "_store", lambda *_args: pytest.fail("cache-only must not write or evict"))
    cache._lock.acquire()
    try:
        payload = cache.get_cached(url, 512)
        assert cache.get_cached("https://museum.test/missing", 512) is None
    finally:
        cache._lock.release()
    with Image.open(io.BytesIO(payload)) as image:
        assert max(image.size) == 512
    assert not cache._path(url, 512).exists()
