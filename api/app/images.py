"""Image proxy with a bounded on-disk cache.

Two problems make direct hot-linking unworkable for the 3D hall:

* WebGL textures are read back by the GPU, so the image host must send CORS
  headers. The Met sends ``Access-Control-Allow-Origin: *``; Cleveland sends
  nothing at all, which would fail every texture load.
* Institution originals are 600 KB–5 MB. A twelve-object hall would pull tens
  of megabytes per visit straight from the institution's CDN.

So images are fetched server-side once, downscaled to a texture-sized WebP, and
served from a size-capped local cache. This is the lazy caching strategy from
01b §1.2: only objects that are actually exhibited ever hit the disk, so the
cache stabilises at a few hundred megabytes rather than the multi-terabyte cost
of mirroring whole collections.

The proxy only ever fetches URLs that already appear in the frozen collection
data, so it cannot be pointed at an arbitrary host.
"""

from __future__ import annotations

import io
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from hashlib import sha256
from pathlib import Path

from PIL import Image

USER_AGENT = "Inquiry-Curator/2.0 (academic research demo)"
REQUEST_TIMEOUT_SECONDS = 30
AIC_IMAGE_HOST = "www.artic.edu"
AIC_USER_AGENT = "Inquiry-Curator/2.0 (academic research demo; jadppcc@gmail.com)"

# Texture budget: 1024 px on the longest side keeps a twelve-object hall around
# 3 MB of texture data while still looking sharp at close camera range.
DEFAULT_MAX_EDGE = 1024
ALLOWED_EDGES = (512, 1024, 1536)
WEBP_QUALITY = 82

DEFAULT_CACHE_LIMIT_BYTES = 256 * 1024 * 1024  # safe local-demo fallback


class ImageFetchError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ImageCache:
    # Callers can opt into deadline support without retrying a TypeError from a
    # legacy/test cache and accidentally making the same network request twice.
    supports_deadline = True
    def __init__(
        self,
        cache_dir: Path,
        limit_bytes: int = DEFAULT_CACHE_LIMIT_BYTES,
    ) -> None:
        self.cache_dir = cache_dir
        self.limit_bytes = limit_bytes
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # One lock per process is enough for a demo: contention only happens
        # when two visitors open the same object at the same moment.
        self._lock = threading.Lock()
        self._aic_request_lock = threading.Lock()
        self._aic_last_request_at = 0.0
        self._inflight: dict[tuple[str, int], Future] = {}

    def _path(self, source_url: str, max_edge: int) -> Path:
        digest = sha256(f"{source_url}|{max_edge}".encode("utf-8")).hexdigest()
        # Shard so a few thousand files do not land in one directory.
        return self.cache_dir / digest[:2] / f"{digest}.webp"

    def has_cached(self, source_url: str, max_edge: int = DEFAULT_MAX_EDGE) -> bool:
        """Check same-source variants, never a different object's URL."""
        return any(self._path(source_url, edge).exists() for edge in ALLOWED_EDGES if edge >= max_edge)

    def get_cached(self, source_url: str, max_edge: int = DEFAULT_MAX_EDGE) -> bytes | None:
        """Read only existing same-source files; never download or wait on locks.

        This path deliberately does not join in-flight requests, take worker
        capacity, write derived variants, or scan the LRU cache. A larger cached
        WebP may be downsampled in memory. Callers decide what to do on a miss.
        """
        if max_edge not in ALLOWED_EDGES:
            max_edge = DEFAULT_MAX_EDGE
        for edge in ALLOWED_EDGES:
            if edge < max_edge:
                continue
            try:
                payload = self._path(source_url, edge).read_bytes()
                if not payload:
                    continue
                return self._downsample(payload, max_edge) if edge != max_edge else payload
            except (OSError, ValueError):
                continue
        return None

    @staticmethod
    def _remaining(deadline: float | None) -> float:
        if deadline is None:
            return float(REQUEST_TIMEOUT_SECONDS)
        remaining = deadline - time.monotonic()
        if not math.isfinite(remaining) or remaining <= 0:
            raise ImageFetchError("image_deadline_exceeded", "The image fetch deadline expired.")
        return remaining

    def _store(self, source_url: str, max_edge: int, payload: bytes) -> None:
        path = self._path(source_url, max_edge)
        with self._lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                self._evict_if_needed()
            except OSError:
                pass

    @staticmethod
    def _downsample(payload: bytes, max_edge: int) -> bytes:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if max(image.size) <= max_edge:
                return payload
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="WEBP", quality=WEBP_QUALITY, method=4)
            return buffer.getvalue()

    def _cached(self, source_url: str, max_edge: int) -> bytes | None:
        for edge in ALLOWED_EDGES:
            if edge < max_edge:
                continue
            path = self._path(source_url, edge)
            if not path.exists():
                continue
            try:
                payload = path.read_bytes()
                path.touch()
                if edge != max_edge:
                    payload = self._downsample(payload, max_edge)
                    self._store(source_url, max_edge, payload)
                return payload
            except (OSError, ValueError):
                continue
        return None

    def get(self, source_url: str, max_edge: int = DEFAULT_MAX_EDGE, *, deadline: float | None = None) -> tuple[bytes, bool]:
        """Return ``(webp_bytes, from_cache)`` inside an optional shared deadline.

        The deadline bounds admission, AIC pacing, retries and request timeouts;
        callers still need a bounded executor because OS DNS/socket operations
        are not forcibly interruptible. Same-source requests share one download.
        A new 512px request also retains a 1024px variant for later wall labels.
        """
        if max_edge not in ALLOWED_EDGES:
            max_edge = DEFAULT_MAX_EDGE
        cached = self._cached(source_url, max_edge)
        if cached is not None:
            return cached, True
        self._remaining(deadline)
        render_edge = max(max_edge, DEFAULT_MAX_EDGE)
        key = (source_url, render_edge)
        with self._lock:
            future = self._inflight.get(key)
            owner = future is None
            if owner:
                future = Future()
                self._inflight[key] = future
        if not owner:
            try:
                payload = future.result(timeout=self._remaining(deadline) if deadline else None)
            except FutureTimeoutError as error:
                raise ImageFetchError("image_deadline_exceeded", "The shared image fetch did not finish before this caller's deadline.") from error
        else:
            try:
                # Recheck after acquiring ownership: another caller may have
                # completed between the first cache read and our lock.
                payload = self._cached(source_url, render_edge)
                if payload is None:
                    payload = (self._render(source_url, render_edge, deadline=deadline)
                               if deadline is not None else self._render(source_url, render_edge))
                    self._store(source_url, render_edge, payload)
                future.set_result(payload)
            except BaseException as error:
                future.set_exception(error)
                raise
            finally:
                with self._lock:
                    self._inflight.pop(key, None)
        if max_edge != render_edge:
            payload = self._downsample(payload, max_edge)
            self._store(source_url, max_edge, payload)
        return payload, not owner

    def _render(self, source_url: str, max_edge: int, *, deadline: float | None = None) -> bytes:
        headers = {"User-Agent": USER_AGENT, "Accept": "image/*"}
        is_aic = urllib.parse.urlsplit(source_url).netloc.casefold() == AIC_IMAGE_HOST
        if is_aic:
            # AIC explicitly documents this project/contact header. It keeps
            # the integration identifiable without forging a Referer or
            # forwarding any visitor URL, cookie, or personal data.
            headers.update(
                {
                    "AIC-User-Agent": AIC_USER_AGENT,
                }
            )

        acquired = False
        download_started = False
        try:
            if is_aic:
                acquired = (self._aic_request_lock.acquire(timeout=self._remaining(deadline))
                            if deadline is not None else self._aic_request_lock.acquire())
                if not acquired:
                    raise ImageFetchError("image_deadline_exceeded", "The AIC image queue exceeded the caller's deadline.")
                elapsed = time.monotonic() - self._aic_last_request_at
                if elapsed < 1.0:
                    if deadline is not None and 1.0 - elapsed >= self._remaining(deadline):
                        raise ImageFetchError("image_deadline_exceeded", "The AIC image pacing interval exceeds the caller's remaining time.")
                    time.sleep(1.0 - elapsed)
            self._remaining(deadline)
            download_started = True
            raw = (self._download(source_url, headers, deadline=deadline)
                   if deadline is not None else self._download(source_url, headers))
        finally:
            if acquired:
                if download_started:
                    self._aic_last_request_at = time.monotonic()
                self._aic_request_lock.release()

        try:
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                # Flatten alpha and exotic modes onto white; a gallery wall
                # texture has no use for transparency.
                if image.mode not in ("RGB", "L"):
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    converted = image.convert("RGBA")
                    background.paste(converted, mask=converted.split()[-1])
                    image = background
                else:
                    image = image.convert("RGB")

                longest = max(image.size)
                if longest > max_edge:
                    scale = max_edge / longest
                    image = image.resize(
                        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                        Image.LANCZOS,
                    )

                buffer = io.BytesIO()
                image.save(buffer, format="WEBP", quality=WEBP_QUALITY, method=4)
                return buffer.getvalue()
        except (OSError, ValueError) as error:
            raise ImageFetchError(
                "unreadable_image", "The institution image could not be decoded."
            ) from error

    def _download(self, source_url: str, headers: dict[str, str], *, deadline: float | None = None) -> bytes:
        raw: bytes | None = None
        last_error: Exception | None = None
        for attempt in range(3):
            remaining = self._remaining(deadline)
            request = urllib.request.Request(
                source_url,
                headers=headers,
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=min(REQUEST_TIMEOUT_SECONDS, remaining)
                ) as response:
                    if deadline is None:
                        raw = response.read()
                    else:
                        chunks = []
                        while True:
                            self._remaining(deadline)
                            chunk = response.read(64 * 1024)
                            if not chunk:
                                break
                            chunks.append(chunk)
                        raw = b"".join(chunks)
                break
            except urllib.error.HTTPError as error:
                last_error = error
                retryable = error.code == 429 or 500 <= error.code < 600
                if retryable and attempt < 2:
                    delay = 0.5 * (2**attempt)
                    if deadline is not None and delay + 0.1 >= self._remaining(deadline):
                        raise ImageFetchError("image_deadline_exceeded", "Image retry would exceed the shared deadline.") from error
                    time.sleep(delay)
                    continue
                raise ImageFetchError(
                    "upstream_http_error",
                    f"The institution image host returned {error.code}.",
                ) from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error
                if attempt < 2:
                    delay = 0.5 * (2**attempt)
                    if deadline is not None and delay + 0.1 >= self._remaining(deadline):
                        raise ImageFetchError("image_deadline_exceeded", "Image retry would exceed the shared deadline.") from error
                    time.sleep(delay)
                    continue
                raise ImageFetchError(
                    "upstream_unreachable",
                    "The institution image host could not be reached.",
                ) from error
        if raw is None:
            raise ImageFetchError(
                "upstream_unreachable",
                f"The institution image host could not be reached: {last_error}",
            )
        return raw

    def _evict_if_needed(self) -> None:
        """Drop least-recently-used entries once the cache exceeds its cap."""
        entries: list[tuple[float, int, Path]] = []
        total = 0
        for path in self.cache_dir.rglob("*.webp"):
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((stat.st_mtime, stat.st_size, path))
            total += stat.st_size
        if total <= self.limit_bytes:
            return
        entries.sort()
        for _mtime, size, path in entries:
            if total <= self.limit_bytes * 0.9:
                break
            try:
                path.unlink()
                total -= size
            except OSError:
                continue

    def stats(self) -> dict[str, int]:
        count = 0
        total = 0
        for path in self.cache_dir.rglob("*.webp"):
            try:
                total += path.stat().st_size
            except OSError:
                continue
            count += 1
        return {"files": count, "bytes": total, "limitBytes": self.limit_bytes}
