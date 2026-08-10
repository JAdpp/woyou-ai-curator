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
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import nullcontext
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

    def _path(self, source_url: str, max_edge: int) -> Path:
        digest = sha256(f"{source_url}|{max_edge}".encode("utf-8")).hexdigest()
        # Shard so a few thousand files do not land in one directory.
        return self.cache_dir / digest[:2] / f"{digest}.webp"

    def get(self, source_url: str, max_edge: int = DEFAULT_MAX_EDGE) -> tuple[bytes, bool]:
        """Return ``(webp_bytes, from_cache)``."""
        if max_edge not in ALLOWED_EDGES:
            max_edge = DEFAULT_MAX_EDGE
        path = self._path(source_url, max_edge)
        if path.exists():
            try:
                # Touch for LRU ordering without rewriting the payload.
                path.touch()
                return path.read_bytes(), True
            except OSError:
                pass

        payload = self._render(source_url, max_edge)
        with self._lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                self._evict_if_needed()
            except OSError:
                # A cache write failure must not fail the request.
                pass
        return payload, False

    def _render(self, source_url: str, max_edge: int) -> bytes:
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

        guard = self._aic_request_lock if is_aic else nullcontext()
        with guard:
            if is_aic:
                elapsed = time.monotonic() - self._aic_last_request_at
                if elapsed < 1.0:
                    time.sleep(1.0 - elapsed)
            try:
                raw = self._download(source_url, headers)
            finally:
                if is_aic:
                    # AIC asks image scrapers to download one image at a time
                    # with a one-second delay between assets.
                    self._aic_last_request_at = time.monotonic()

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

    def _download(self, source_url: str, headers: dict[str, str]) -> bytes:
        raw: bytes | None = None
        last_error: Exception | None = None
        for attempt in range(3):
            request = urllib.request.Request(
                source_url,
                headers=headers,
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=REQUEST_TIMEOUT_SECONDS
                ) as response:
                    raw = response.read()
                break
            except urllib.error.HTTPError as error:
                last_error = error
                retryable = error.code == 429 or 500 <= error.code < 600
                if retryable and attempt < 2:
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise ImageFetchError(
                    "upstream_http_error",
                    f"The institution image host returned {error.code}.",
                ) from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error
                if attempt < 2:
                    time.sleep(0.5 * (2**attempt))
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
