from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageChops, ImageDraw, ImageFont, UnidentifiedImageError


GENERATION_PATH = "/api/v1/services/aigc/multimodal-generation/generation"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DEFAULT_MAX_IMAGE_BYTES = 15 * 1024 * 1024
GENERATION_MAX_ATTEMPTS = 2
GENERATION_RETRY_DELAY_SECONDS = 0.75
DOWNLOAD_MAX_ATTEMPTS = 2
DOWNLOAD_RETRY_DELAY_SECONDS = 0.25
# The visual model is never asked to render exhibition copy.  The compositor
# owns the only legitimate typography area, which is an opaque replacement of
# the generated pixels rather than a translucent overlay.  This makes the
# reserved title field a deterministic no-pseudo-text boundary even if an image
# model ignores part of its negative prompt.
POSTER_COMPOSITION_VERSION = "pillow-poster-v4-safe-edge-crop"
POSTER_CREDIT = "卧游 · AI策展人彦远"
POSTER_PANEL_RATIO = 0.47
POSTER_PANEL_TRANSITION_RATIO = 0.025
# Image models often place signature-like marks, fake seals or pseudo-QR blocks
# at a canvas edge even when explicitly forbidden.  A deterministic editorial
# crop removes that high-risk perimeter before any system typography is added.
POSTER_SOURCE_EDGE_CROP_RATIO = 0.10
POSTER_NEGATIVE_PROMPT = (
    "readable text, letters, Chinese characters, numbers, pseudo-text, pseudo-writing, decorative "
    "script, typography, logo, watermark, QR code, museum label, caption card, title card, label card, "
    "placard, nameplate, infographic, menu, interface, list, document fragment, page, manuscript, "
    "paper sheet with marks, marked paper sheet, seal, stamp, emblem, replica of a "
    "specific museum object, framed artifact, fake historical artifact, period costume, altar, "
    "pedestal, generic abstract-only composition, clutter, low contrast, distorted symbols"
)

_VISUAL_BRIEF_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("猫", "cat", "cats", "feline"),
        "Contemporary editorial collage of unmistakable feline silhouettes, alert eyes, and "
        "curving tails, built from paper fibre, clay, ink-line, and metal textures with multiple "
        "directions of gaze; paper is material texture only, never a sheet, card, page, label, or "
        "document; no people, costumes, altars, pedestals, or faux artifacts",
    ),
    (
        ("动物", "兽", "鸟", "鱼", "马", "犬", "狗", "animal", "bird", "fish", "horse", "dog"),
        "Contemporary editorial study of non-specific animal silhouettes and movement traces, "
        "using tactile paper, pigment, fibre, and metal textures without specimen displays or replicas",
    ),
    (
        ("山水", "水", "行旅", "旅行", "迁徙", "landscape", "water", "travel", "journey"),
        "Layered terrain, water currents, mist, and path-like spatial rhythms expressed through "
        "ink wash, mineral colour, paper relief, and contemporary cartographic depth without labels",
    ),
    (
        ("书法", "书写", "题跋", "文字", "诗", "手稿", "writing", "calligraphy", "manuscript", "inscription"),
        "Paper fibre, pooled ink, dry-brush energy, and the physical rhythm of writing expressed "
        "without any legible mark, character, pseudo-script, page, or document replica",
    ),
    (
        ("礼", "仪式", "青铜", "陶瓷", "陶器", "器物", "ritual", "bronze", "ceramic", "pottery", "vessel"),
        "Non-specific vessel geometry and ceremonial rhythm translated into oxidized metal, clay, "
        "glaze, lacquer, and circular tension; material abstraction only, never a reconstructed object",
    ),
    (
        ("宗教", "信仰", "佛", "神圣", "religion", "faith", "buddh", "sacred", "devotion"),
        "Concentric light, thresholds, breath-like haze, and contemplative spatial intervals in "
        "mineral pigment and translucent paper, with no deity replica, altar, icon, or sacred text",
    ),
    (
        ("死亡", "墓", "丧葬", "来世", "death", "afterlife", "funerary", "burial", "mourning"),
        "A symbolic passage from shadow to light through thresholds, river-like bands, night pigment, "
        "and fragile fibres, without remains, tomb replicas, funerary objects, or ritual staging",
    ),
    (
        ("贸易", "交流", "迁移", "移民", "流动", "trade", "exchange", "migration", "diaspora", "route"),
        "Intersecting routes, currents, knots, and material fragments from different making traditions, "
        "arranged as a contemporary cross-cultural editorial network without maps, labels, or emblems",
    ),
    (
        ("女性", "女人", "性别", "身份", "women", "woman", "gender", "identity", "selfhood"),
        "Layered anonymous profiles, mirrors, veils, and shifting boundaries rendered as contemporary "
        "paper and textile collage, avoiding historical reenactment, costume, and portrait imitation",
    ),
    (
        ("权力", "战争", "帝国", "冲突", "武器", "power", "war", "empire", "conflict", "weapon"),
        "Monumental planes, fractured alignments, pressure, and counterforce expressed through stone, "
        "metal, smoke, and red thread, with no rulers, uniforms, weapons, insignia, or battle scene",
    ),
    (
        ("日常", "生活", "家庭", "饮食", "daily life", "domestic", "food", "home"),
        "Close-up rhythms of hands-free domestic materials, repeated use, vessels-as-shapes, food colour, "
        "and worn surfaces in a contemporary editorial still life without replica artifacts",
    ),
    (
        ("纺织", "织物", "服饰", "丝绸", "textile", "fabric", "weaving", "silk", "clothing"),
        "Threads, folds, weave structures, seams, and layered translucent fibres treated as a modern "
        "material collage, without garments, mannequins, period dress, labels, or textile replicas",
    ),
    (
        ("建筑", "城市", "空间", "居所", "architecture", "building", "city", "urban", "dwelling"),
        "Thresholds, voids, structural grids, light wells, and layered silhouettes of inhabited space, "
        "expressed as contemporary editorial geometry without a literal monument or reconstructed room",
    ),
)

_DEFAULT_VISUAL_BRIEF = (
    "Cross-cultural editorial composition of tactile paper, clay, fibre, stone, pigment, and metal "
    "contrasts, organized through non-specific shapes, shadows, apertures, and material transitions"
)


class AliyunImageProviderError(RuntimeError):
    """A safe, structured image-provider failure.

    Messages intentionally avoid upstream response bodies, credentials, and
    temporary image URLs so callers can return ``to_dict()`` to an internal UI.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.http_status = http_status

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": "aliyun-model-studio",
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "httpStatus": self.http_status,
        }


@dataclass(frozen=True, slots=True)
class PosterContext:
    exhibition_id: str
    exhibition_theme: str
    title: str
    core_question: str
    subtitle: str = ""
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class GeneratedPoster:
    provider: str
    model: str
    size: str
    image_path: Path
    metadata_path: Path
    sha256: str
    seed: int | None
    generated_at: str
    request_id: str | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["image_path"] = str(self.image_path)
        result["metadata_path"] = str(self.metadata_path)
        return result


def _compact_context(value: str, *, max_length: int) -> str:
    """Treat contextual text as data, not as free-form prompt instructions."""

    compacted = re.sub(r"[\x00-\x1f\x7f]+", " ", value or "")
    compacted = re.sub(r"\s+", " ", compacted).strip()
    # Keep stored metadata and compositor copy single-line and display-safe.
    compacted = compacted.replace("<", "（").replace(">", "）")
    return compacted[:max_length] or "未命名"


def _matches_visual_token(source: str, token: str) -> bool:
    if token.isascii() and re.fullmatch(r"[a-z ]+", token):
        return re.search(rf"\b{re.escape(token)}\b", source) is not None
    return token in source


def _visual_brief(context: PosterContext) -> str:
    """Map local visitor language to allowlisted English art direction only."""

    source = " ".join(
        (
            context.exhibition_theme or "",
            context.title or "",
            context.subtitle or "",
            context.core_question or "",
        )
    ).casefold()
    matches = [
        brief
        for tokens, brief in _VISUAL_BRIEF_RULES
        if any(_matches_visual_token(source, token) for token in tokens)
    ]
    # Cat imagery is especially prone to faux-historical scenes. Keep its
    # tightly constrained contemporary collage brief intact even when the
    # visitor's question also contains broad concepts such as power or identity.
    if matches and matches[0] == _VISUAL_BRIEF_RULES[0][1]:
        return matches[0]
    return "; combined with ".join(matches[:2]) if matches else _DEFAULT_VISUAL_BRIEF


def build_poster_visual_prompt(context: PosterContext) -> str:
    """Build an allowlisted visual-only prompt with no visitor text upstream."""

    brief = _visual_brief(context)
    return (
        "Create a sophisticated 16:9 full-bleed editorial artwork layer for later typography. "
        f"Allowlisted visual brief: {brief}. "
        "Use contemporary museum-editorial composition, recognizable symbolic forms, tactile material "
        "detail, and strong spatial rhythm. Keep the left half calm, dark, and low-detail for a later title. "
        "Do not reproduce, imitate, reconstruct, stage, or display any specific museum object, accession "
        "photograph, historical artifact, institution identity, period costume, or faux-antique object. "
        "Do not create a gallery room, display case, altar, pedestal, framed exhibit, infographic, menu, "
        "interface, list, seal, stamp, emblem, caption card, title card, label card, placard, nameplate, "
        "document fragment, page, manuscript, or marked paper sheet. Render no text or text-like feature at "
        "all: no glyph, signage, title, Chinese character, letter, number, pseudo-text, pseudo-writing, "
        "decorative script, label, logo, watermark, signature, or QR code. Treat paper, clay, fibre, and ink "
        "only as non-linguistic material texture, never as a written surface. Output one polished visual layer "
        "with clear thematic imagery and no ephemera: no typography, and no signage."
    )


# Compatibility alias for callers outside this package; new code should use
# ``build_poster_visual_prompt`` because the result is no longer a backdrop.
build_backdrop_prompt = build_poster_visual_prompt


class AliyunImageProvider:
    def __init__(
        self,
        *,
        api_key: str,
        api_host: str,
        output_dir: Path,
        model: str = "qwen-image-3.0-pro",
        size: str = "1536*864",
        timeout_seconds: float = 90,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        font_path: Path | None = None,
        allow_insecure: bool = False,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise AliyunImageProviderError(
                "not_configured", "Alibaba Cloud image generation is not configured."
            )
        if client is not None and transport is not None:
            raise AliyunImageProviderError(
                "invalid_config", "Provide either an HTTP client or a transport, not both."
            )
        if max_image_bytes <= len(PNG_SIGNATURE):
            raise AliyunImageProviderError(
                "invalid_config", "The image byte limit is too small."
            )

        self.api_key = api_key.strip()
        self.api_host = self._validate_http_origin(api_host, allow_insecure=allow_insecure)
        self.output_dir = Path(output_dir)
        self.model = model
        self.size = size
        self._parse_size(size)
        self.timeout_seconds = timeout_seconds
        self.max_image_bytes = max_image_bytes
        self.font_path = self._resolve_font_path(font_path)
        self.allow_insecure = allow_insecure
        self._client = client
        self._transport = transport

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def generate_poster(self, context: PosterContext) -> GeneratedPoster:
        safe_id = self._validate_exhibition_id(context.exhibition_id)
        prompt = build_poster_visual_prompt(context)
        request_body: dict[str, Any] = {
            "model": self.model,
            "input": {
                "messages": [
                    {"role": "user", "content": [{"text": prompt}]},
                ]
            },
            "parameters": {
                "size": self.size,
                "prompt_extend": False,
                "negative_prompt": POSTER_NEGATIVE_PROMPT,
                "n": 1,
                "watermark": False,
            },
        }
        if context.seed is not None:
            request_body["parameters"]["seed"] = context.seed

        async with self._client_context() as client:
            payload, request_id = await self._request_generation(client, request_body)
            image_url = self._extract_image_url(payload)
            source_image_bytes = await self._download_png(client, image_url)

        image_bytes, overlay = self._compose_poster(source_image_bytes, context)
        if len(image_bytes) > self.max_image_bytes:
            raise AliyunImageProviderError(
                "image_too_large", "The composed poster exceeded the configured size limit."
            )

        digest = hashlib.sha256(image_bytes).hexdigest()
        generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        stem = f"{safe_id}-{digest[:12]}"
        output_root = self.output_dir.resolve()
        image_path = self._safe_output_path(output_root, f"{stem}.png")
        metadata_path = self._safe_output_path(output_root, f"{stem}.json")
        prompt_summary = {
            "theme": _compact_context(context.exhibition_theme, max_length=120),
            "title": overlay["title"],
            "secondaryText": overlay["secondaryText"],
            "coreQuestion": _compact_context(context.core_question, max_length=220),
            "instructionSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }
        metadata = {
            "provider": "aliyun-model-studio",
            "model": self.model,
            "size": self.size,
            "promptSummary": prompt_summary,
            "compositionVersion": POSTER_COMPOSITION_VERSION,
            "exactTitle": overlay["title"],
            "textOverlay": overlay,
            "seed": context.seed,
            "generatedAt": generated_at,
            "requestId": request_id,
            "sourceImageSha256": hashlib.sha256(source_image_bytes).hexdigest(),
            "sha256": digest,
        }
        self._write_outputs(image_path, metadata_path, image_bytes, metadata)
        return GeneratedPoster(
            provider="aliyun-model-studio",
            model=self.model,
            size=self.size,
            image_path=image_path,
            metadata_path=metadata_path,
            sha256=digest,
            seed=context.seed,
            generated_at=generated_at,
            request_id=request_id,
        )

    async def generate_backdrop(self, context: PosterContext) -> GeneratedPoster:
        """Backward-compatible method name retained for injected test providers."""

        return await self.generate_poster(context)

    @staticmethod
    def _parse_size(value: str) -> tuple[int, int]:
        match = re.fullmatch(r"([1-9][0-9]{2,4})\*([1-9][0-9]{2,4})", value.strip())
        if match is None:
            raise AliyunImageProviderError(
                "invalid_config", "The configured poster size is invalid."
            )
        width, height = (int(match.group(1)), int(match.group(2)))
        if width * height > 2048 * 2048:
            raise AliyunImageProviderError(
                "invalid_config", "The configured poster size exceeds the model limit."
            )
        return width, height

    @staticmethod
    def _resolve_font_path(configured: Path | None) -> Path:
        candidates: list[Path] = []
        if configured is not None:
            candidate = Path(configured).expanduser()
            try:
                resolved = candidate.resolve(strict=True)
                if not resolved.is_file():
                    raise OSError("not a file")
                ImageFont.truetype(str(resolved), 24)
                return resolved
            except (OSError, RuntimeError) as exc:
                raise AliyunImageProviderError(
                    "invalid_config",
                    "ALIYUN_IMAGE_FONT_PATH does not point to a usable CJK font.",
                ) from exc

        windows_root = os.getenv("WINDIR")
        if windows_root:
            fonts = Path(windows_root) / "Fonts"
            candidates.extend(
                fonts / filename
                for filename in (
                    "NotoSerifSC-VF.ttf",
                    "NotoSansSC-VF.ttf",
                    "msyh.ttc",
                    "msyhbd.ttc",
                    "simhei.ttf",
                    "simsun.ttc",
                )
            )

        candidates.extend(
            Path(value)
            for value in (
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/Hiragino Sans GB.ttc",
                "/System/Library/Fonts/STHeiti Medium.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
                "/usr/share/fonts/opentype/noto/NotoSerifCJKsc-Regular.otf",
                "/usr/share/fonts/opentype/adobe-source-han-sans/SourceHanSansSC-Regular.otf",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            )
        )

        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
                if not resolved.is_file():
                    continue
                ImageFont.truetype(str(resolved), 24)
                return resolved
            except (OSError, RuntimeError):
                continue

        message = (
            "No usable CJK font was found for poster composition. Configure "
            "ALIYUN_IMAGE_FONT_PATH with a TrueType/OpenType CJK font."
        )
        raise AliyunImageProviderError("font_not_available", message)

    def _font(self, size: int) -> ImageFont.FreeTypeFont:
        try:
            return ImageFont.truetype(str(self.font_path), size)
        except (OSError, RuntimeError) as exc:
            raise AliyunImageProviderError(
                "font_not_available", "The configured CJK font could not be loaded."
            ) from exc

    @staticmethod
    def _wrap_text(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
    ) -> list[str]:
        lines: list[str] = []
        current = ""
        for character in text:
            candidate = current + character
            bbox = draw.textbbox((0, 0), candidate, font=font)
            width = bbox[2] - bbox[0]
            if current and width > max_width:
                lines.append(current.rstrip())
                current = character.lstrip()
            else:
                current = candidate
        if current:
            lines.append(current.rstrip())
        return lines or [text]

    def _fit_title(
        self,
        draw: ImageDraw.ImageDraw,
        title: str,
        *,
        max_width: int,
        max_height: int,
        image_height: int,
    ) -> tuple[ImageFont.FreeTypeFont, str, tuple[int, int, int, int]]:
        maximum = max(48, round(image_height * 0.095))
        minimum = max(28, round(image_height * 0.045))
        colon = re.search(r"[：:]", title)
        prefer_colon_break = colon is not None and colon.end() < len(title)

        # First try a semantic headline break: keep the colon and everything
        # before it together, then shrink until that complete phrase fits.
        # Only if no safe size exists do we fall back to exact character wrap.
        passes = (True, False) if prefer_colon_break else (False,)
        for semantic_break in passes:
            for size in range(maximum, minimum - 1, -2):
                font = self._font(size)
                if semantic_break and colon is not None:
                    prefix = title[: colon.end()]
                    suffix = title[colon.end() :]
                    # Preserve spaces exactly while keeping conventional
                    # post-colon spacing at the end of the first line.
                    whitespace = len(suffix) - len(suffix.lstrip())
                    prefix += suffix[:whitespace]
                    suffix = suffix[whitespace:]
                    prefix_bbox = draw.textbbox((0, 0), prefix, font=font)
                    if prefix_bbox[2] - prefix_bbox[0] > max_width:
                        continue
                    lines = [prefix, *self._wrap_text_exact(draw, suffix, font, max_width)]
                else:
                    lines = self._wrap_text_exact(draw, title, font, max_width)
                if len(lines) > 4:
                    continue
                rendered = "\n".join(lines)
                bbox = draw.multiline_textbbox(
                    (0, 0), rendered, font=font, spacing=round(size * 0.28)
                )
                if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
                    return font, rendered, bbox
        raise AliyunImageProviderError(
            "poster_composition_error",
            "The exhibition title is too long to compose safely on the poster.",
        )

    @staticmethod
    def _wrap_text_exact(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
    ) -> list[str]:
        """Wrap without dropping, trimming, or rewriting any title character."""

        lines: list[str] = []
        current = ""
        for character in text:
            candidate = current + character
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if current and bbox[2] - bbox[0] > max_width:
                lines.append(current)
                current = character
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines or [text]

    def _fit_secondary(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        *,
        max_width: int,
        max_height: int,
        image_height: int,
    ) -> tuple[ImageFont.FreeTypeFont, str, tuple[int, int, int, int]]:
        maximum = max(24, round(image_height * 0.036))
        minimum = max(18, round(image_height * 0.024))
        for size in range(maximum, minimum - 1, -1):
            font = self._font(size)
            lines = self._wrap_text(draw, text, font, max_width)
            if len(lines) > 3:
                continue
            rendered = "\n".join(lines)
            bbox = draw.multiline_textbbox(
                (0, 0), rendered, font=font, spacing=round(size * 0.35)
            )
            if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
                return font, rendered, bbox
        raise AliyunImageProviderError(
            "poster_composition_error",
            "The exhibition subtitle is too long to compose safely on the poster.",
        )

    @staticmethod
    def _opaque_panel(
        width: int,
        height: int,
        *,
        texture_key: str,
    ) -> tuple[Image.Image, int]:
        """Create a deterministic, fully opaque paper-textured title panel.

        The tonal transition is rendered into the panel's RGB pixels rather
        than by fading its alpha. Consequently, even bright or text-like model
        output underneath the transition can never bleed into the final poster.
        """

        transition_width = max(12, round(width * (POSTER_PANEL_TRANSITION_RATIO / POSTER_PANEL_RATIO)))
        transition_start = max(1, width - transition_width)
        base_row: list[tuple[int, int, int]] = []
        for x in range(width):
            if x < transition_start:
                progress = x / transition_start
                red = round(12 + 3 * progress)
                green = round(16 + 3 * progress)
                blue = round(19 + 2 * progress)
            else:
                progress = (x - transition_start) / max(1, transition_width - 1)
                eased = progress * progress * (3 - 2 * progress)
                red = round(15 + 10 * eased)
                green = round(19 + 7 * eased)
                blue = round(21 + 6 * eased)
            base_row.append((red, green, blue))

        gradient = Image.new("RGB", (width, 1))
        gradient.putdata(base_row)
        panel = gradient.resize((width, height))

        # A small deterministic tile produces restrained paper grain without
        # relying on process-global randomness or non-repeatable image effects.
        tile_size = 64
        digest = hashlib.sha256(texture_key.encode("utf-8")).digest()
        noise_values: list[tuple[int, int, int]] = []
        for y in range(tile_size):
            for x in range(tile_size):
                value = (
                    x * 37
                    + y * 71
                    + x * y * 3
                    + digest[(x + y) % len(digest)]
                ) % 7 - 3
                fibre = 1 if (x * 11 + y * 17 + digest[x % len(digest)]) % 29 == 0 else 0
                noise = 128 + value + fibre
                noise_values.append((noise, noise, noise))
        tile = Image.new("RGB", (tile_size, tile_size))
        tile.putdata(noise_values)
        texture = Image.new("RGB", (width, height), (128, 128, 128))
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                texture.paste(tile, (left, top))
        return ImageChops.add(panel, texture, scale=1.0, offset=-128), transition_width

    @staticmethod
    def _secondary_text(context: PosterContext, title: str) -> str:
        candidate = (context.subtitle or "").strip() or (context.core_question or "").strip()
        if not candidate:
            return ""
        compact = _compact_context(candidate, max_length=80)
        if compact == title:
            return ""
        return compact if len(compact) <= 56 else compact[:55].rstrip() + "…"

    def _compose_poster(
        self, source_image_bytes: bytes, context: PosterContext
    ) -> tuple[bytes, dict[str, Any]]:
        expected_size = self._parse_size(self.size)
        try:
            with Image.open(io.BytesIO(source_image_bytes)) as opened:
                opened.load()
                if opened.format != "PNG" or opened.size != expected_size:
                    raise AliyunImageProviderError(
                        "image_validation_error",
                        "The generated image dimensions or format did not match the request.",
                    )
                poster = opened.convert("RGBA")
        except AliyunImageProviderError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise AliyunImageProviderError(
                "image_validation_error", "The generated PNG could not be decoded safely."
            ) from exc

        width, height = expected_size
        crop_x = round(width * POSTER_SOURCE_EDGE_CROP_RATIO)
        crop_y = round(height * POSTER_SOURCE_EDGE_CROP_RATIO)
        poster = poster.crop(
            (crop_x, crop_y, width - crop_x, height - crop_y)
        ).resize((width, height), Image.Resampling.LANCZOS)
        title = _compact_context(context.title, max_length=160)
        secondary = self._secondary_text(context, title)

        panel_width = round(width * POSTER_PANEL_RATIO)
        panel, transition_width = self._opaque_panel(
            panel_width,
            height,
            texture_key="\0".join((title, secondary, POSTER_CREDIT)),
        )
        # Paste without a mask: every pixel in the full panel, including its
        # short tonal transition, is opaque and replaces the model output.
        poster.paste(panel.convert("RGBA"), (0, 0))

        draw = ImageDraw.Draw(poster)
        margin_x = round(width * 0.055)
        transition_start = panel_width - transition_width
        right_padding = round(width * 0.025)
        max_text_width = transition_start - margin_x - right_padding
        brand_font = self._font(max(22, round(height * 0.032)))
        title_font, rendered_title, title_bbox = self._fit_title(
            draw,
            title,
            max_width=max_text_width,
            max_height=round(height * 0.40),
            image_height=height,
        )

        light = (250, 247, 239, 255)
        muted = (226, 224, 216, 255)
        brand_y = round(height * 0.105)
        draw.text(
            (margin_x, brand_y),
            POSTER_CREDIT,
            font=brand_font,
            fill=muted,
        )
        rule_y = brand_y + round(height * 0.058)
        draw.line(
            (margin_x, rule_y, margin_x + round(width * 0.095), rule_y),
            fill=(220, 76, 55, 255),
            width=max(2, round(height * 0.004)),
        )

        title_y = round(height * 0.245)
        title_spacing = round(title_font.size * 0.28)
        draw.multiline_text(
            (margin_x, title_y),
            rendered_title,
            font=title_font,
            fill=light,
            spacing=title_spacing,
        )
        title_height = title_bbox[3] - title_bbox[1]

        if secondary:
            secondary_y = title_y + title_height + round(height * 0.065)
            safe_bottom = round(height * 0.88)
            secondary_font, rendered_secondary, _ = self._fit_secondary(
                draw,
                secondary,
                max_width=max_text_width,
                max_height=safe_bottom - secondary_y,
                image_height=height,
            )
            draw.multiline_text(
                (margin_x, secondary_y),
                rendered_secondary,
                font=secondary_font,
                fill=muted,
                spacing=round(secondary_font.size * 0.35),
            )

        output = io.BytesIO()
        poster.convert("RGB").save(output, format="PNG", compress_level=6)
        return output.getvalue(), {
            "title": title,
            "renderedTitle": rendered_title,
            "titleFontSize": title_font.size,
            "secondaryText": secondary,
            "credit": POSTER_CREDIT,
            "fontFile": self.font_path.name,
            "panelOpaque": True,
            "panelWidthPx": panel_width,
            "panelRatio": POSTER_PANEL_RATIO,
            "transitionWidthPx": transition_width,
            "sourceEdgeCropRatio": POSTER_SOURCE_EDGE_CROP_RATIO,
            "sourceEdgeCropPixels": {"x": crop_x, "y": crop_y},
        }

    @asynccontextmanager
    async def _client_context(self) -> AsyncIterator[httpx.AsyncClient]:
        if self._client is not None:
            yield self._client
            return
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            transport=self._transport,
            follow_redirects=True,
        ) as client:
            yield client

    async def _request_generation(
        self, client: httpx.AsyncClient, request_body: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None]:
        response: httpx.Response | None = None
        last_error: httpx.HTTPError | None = None
        for attempt in range(GENERATION_MAX_ATTEMPTS):
            response = None
            try:
                response = await client.post(
                    f"{self.api_host}{GENERATION_PATH}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_body,
                )
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                retryable = status == 429 or status >= 500
                if retryable and attempt + 1 < GENERATION_MAX_ATTEMPTS:
                    await asyncio.sleep(GENERATION_RETRY_DELAY_SECONDS)
                    continue
                raise AliyunImageProviderError(
                    "provider_http_error",
                    "Alibaba Cloud image generation did not accept the request.",
                    retryable=retryable,
                    http_status=status,
                ) from exc
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt + 1 < GENERATION_MAX_ATTEMPTS:
                    await asyncio.sleep(GENERATION_RETRY_DELAY_SECONDS)
                    continue

        if response is None:
            raise AliyunImageProviderError(
                "provider_network_error",
                "Alibaba Cloud image generation could not be reached after one retry.",
                retryable=True,
            ) from last_error

        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise AliyunImageProviderError(
                "provider_response_error",
                "Alibaba Cloud returned an unreadable generation response.",
            ) from exc
        if not isinstance(payload, dict):
            raise AliyunImageProviderError(
                "provider_response_error",
                "Alibaba Cloud returned an unusable generation response.",
            )
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            request_id = response.headers.get("x-request-id") or None
        return payload, request_id

    def _extract_image_url(self, payload: dict[str, Any]) -> str:
        try:
            content = payload["output"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AliyunImageProviderError(
                "provider_response_error",
                "Alibaba Cloud returned no generated image.",
            ) from exc
        if not isinstance(content, list):
            raise AliyunImageProviderError(
                "provider_response_error",
                "Alibaba Cloud returned no generated image.",
            )
        image_url = next(
            (
                item.get("image")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("image"), str)
            ),
            None,
        )
        if not image_url:
            raise AliyunImageProviderError(
                "provider_response_error",
                "Alibaba Cloud returned no generated image.",
            )
        return self._validate_asset_url(image_url)

    async def _download_png(self, client: httpx.AsyncClient, image_url: str) -> bytes:
        # The paid generation has already succeeded. Retry only this signed
        # asset URL, and share one wall-clock budget across headers, body and
        # backoff rather than granting each attempt another full timeout.
        async def download_with_retry() -> bytes:
            for attempt in range(DOWNLOAD_MAX_ATTEMPTS):
                try:
                    return await self._download_png_once(client, image_url)
                except AliyunImageProviderError as exc:
                    if (
                        exc.code != "image_download_error"
                        or not exc.retryable
                        or attempt + 1 >= DOWNLOAD_MAX_ATTEMPTS
                    ):
                        raise
                    await asyncio.sleep(DOWNLOAD_RETRY_DELAY_SECONDS)
            raise AssertionError("The bounded image download loop must return or raise.")

        try:
            return await asyncio.wait_for(download_with_retry(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise AliyunImageProviderError(
                "image_download_error",
                "The generated image download exceeded its time limit.",
                retryable=True,
            ) from exc

    async def _download_png_once(self, client: httpx.AsyncClient, image_url: str) -> bytes:
        try:
            async with client.stream("GET", image_url) as response:
                response.raise_for_status()
                media_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if media_type != "image/png":
                    raise AliyunImageProviderError(
                        "image_validation_error",
                        "The generated asset was not a PNG image.",
                    )
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        advertised_size = int(content_length)
                    except ValueError as exc:
                        raise AliyunImageProviderError(
                            "image_validation_error",
                            "The generated image had invalid size metadata.",
                        ) from exc
                    if advertised_size > self.max_image_bytes:
                        raise AliyunImageProviderError(
                            "image_too_large",
                            "The generated image exceeded the configured size limit.",
                        )

                buffer = bytearray()
                async for chunk in response.aiter_bytes():
                    buffer.extend(chunk)
                    if len(buffer) > self.max_image_bytes:
                        raise AliyunImageProviderError(
                            "image_too_large",
                            "The generated image exceeded the configured size limit.",
                        )
        except AliyunImageProviderError:
            raise
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise AliyunImageProviderError(
                "image_download_error",
                "The generated image could not be downloaded.",
                retryable=status == 429 or 500 <= status < 600,
                http_status=status,
            ) from exc
        except httpx.HTTPError as exc:
            raise AliyunImageProviderError(
                "image_download_error",
                "The generated image could not be downloaded.",
                retryable=isinstance(exc, httpx.TransportError)
                and not isinstance(exc, (httpx.UnsupportedProtocol, httpx.LocalProtocolError)),
            ) from exc

        image_bytes = bytes(buffer)
        if not image_bytes.startswith(PNG_SIGNATURE):
            raise AliyunImageProviderError(
                "image_validation_error",
                "The generated asset did not contain valid PNG data.",
            )
        return image_bytes

    def _write_outputs(
        self,
        image_path: Path,
        metadata_path: Path,
        image_bytes: bytes,
        metadata: dict[str, Any],
    ) -> None:
        try:
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_tmp = image_path.with_suffix(".png.tmp")
            metadata_tmp = metadata_path.with_suffix(".json.tmp")
            image_tmp.write_bytes(image_bytes)
            metadata_tmp.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            image_tmp.replace(image_path)
            metadata_tmp.replace(metadata_path)
        except OSError as exc:
            raise AliyunImageProviderError(
                "storage_error", "The generated image could not be saved locally."
            ) from exc

    @staticmethod
    def _validate_exhibition_id(exhibition_id: str) -> str:
        if not isinstance(exhibition_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", exhibition_id
        ):
            raise AliyunImageProviderError(
                "invalid_request", "The exhibition identifier is invalid."
            )
        if exhibition_id in {".", ".."}:
            raise AliyunImageProviderError(
                "invalid_request", "The exhibition identifier is invalid."
            )
        return exhibition_id

    @staticmethod
    def _safe_output_path(output_root: Path, filename: str) -> Path:
        candidate = (output_root / filename).resolve()
        if output_root != candidate.parent:
            raise AliyunImageProviderError(
                "invalid_request", "The generated output path is invalid."
            )
        return candidate

    @staticmethod
    def _validate_http_origin(api_host: str, *, allow_insecure: bool) -> str:
        parsed = urlsplit((api_host or "").strip())
        valid_scheme = parsed.scheme == "https" or (allow_insecure and parsed.scheme == "http")
        if (
            not valid_scheme
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise AliyunImageProviderError(
                "invalid_config", "The Alibaba Cloud API host must be a secure origin."
            )
        return f"{parsed.scheme}://{parsed.netloc}"

    def _validate_asset_url(self, image_url: str) -> str:
        parsed = urlsplit(image_url)
        valid_scheme = parsed.scheme == "https" or (
            self.allow_insecure and parsed.scheme == "http"
        )
        if (
            not valid_scheme
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise AliyunImageProviderError(
                "provider_response_error",
                "Alibaba Cloud returned an unsafe image location.",
            )
        return image_url
