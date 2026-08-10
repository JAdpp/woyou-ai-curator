from __future__ import annotations

import asyncio
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .models import Chapter, Exhibition, ExhibitionItem
from .providers.aliyun_tts import AliyunTtsProvider, is_valid_mp3


AudioGuideKind = Literal["lobby", "chapter", "artwork", "epilogue"]


class AudioGuideReferenceError(ValueError):
    def __init__(self, code: str, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class AudioGuideResult:
    path: Path
    cache_status: Literal["HIT", "MISS"]
    model: str
    voice: str
    text_sha256: str


def _clean_spoken_text(value: object) -> str:
    """Convert stored curatorial prose into plain, narration-safe text."""

    text = str(value or "")
    text = re.sub(r"<[^>]{0,500}>", " ", text)
    text = re.sub(r"[`*_#>|]+", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _sentence(value: object) -> str:
    text = _clean_spoken_text(value)
    if not text:
        return ""
    if text[-1] not in "。！？!?；;：:":
        text += "。"
    return text


def _join(*parts: object) -> str:
    return "".join(segment for part in parts if (segment := _sentence(part)))


def _find_chapter(exhibition: Exhibition, reference: str | None) -> Chapter:
    if not reference or not reference.strip():
        raise AudioGuideReferenceError(
            "missing_ref", "chapter audio requires a chapter ref.", status_code=422
        )
    ref = reference.strip()
    chapter = next((candidate for candidate in exhibition.chapters if candidate.id == ref), None)
    if chapter is None:
        raise AudioGuideReferenceError(
            "chapter_not_found",
            "The requested chapter does not belong to this exhibition.",
            status_code=404,
        )
    return chapter


def _find_artwork(exhibition: Exhibition, reference: str | None) -> ExhibitionItem:
    if not reference or not reference.strip():
        raise AudioGuideReferenceError(
            "missing_ref", "artwork audio requires an artwork ref.", status_code=422
        )
    ref = reference.strip()
    # The stop ID is canonical.  Object IDs are accepted as a compatibility
    # convenience because older 3D clients used catalogue IDs as stop refs.
    item = next(
        (
            candidate
            for candidate in exhibition.items
            if candidate.id == ref or candidate.object.id == ref
        ),
        None,
    )
    if item is None:
        raise AudioGuideReferenceError(
            "artwork_not_found",
            "The requested artwork does not belong to this exhibition.",
            status_code=404,
        )
    return item


def build_audio_guide_text(
    exhibition: Exhibition,
    kind: AudioGuideKind,
    reference: str | None = None,
) -> str:
    """Rebuild one trusted guide segment from a validated exhibition record.

    Deliberately no free-form ``text`` argument exists in the API boundary.
    Every spoken sentence is selected from the stored exhibition identified by
    the route, so visitors cannot turn the server into a general-purpose TTS
    proxy.
    """

    if kind == "lobby":
        chapter_titles = "、".join(
            _clean_spoken_text(chapter.title) for chapter in exhibition.chapters
        )
        return _join(
            f"欢迎来到卧游为你策划的展览，《{_clean_spoken_text(exhibition.title)}》。",
            exhibition.subtitle,
            f"本展览从这个问题出发：{_clean_spoken_text(exhibition.question)}",
            f"策展主张是：{_clean_spoken_text(exhibition.curatorial_thesis)}",
            f"接下来，我们将依次经过{chapter_titles}。" if chapter_titles else "",
            "请跟随彦远，开始这段参观。",
        )

    if kind == "chapter":
        chapter = _find_chapter(exhibition, reference)
        item_lookup = {item.id: item for item in exhibition.items}
        titles = [
            _clean_spoken_text(item_lookup[item_id].display_title or item_lookup[item_id].object.title_original or item_lookup[item_id].object.title)
            for item_id in chapter.item_ids
            if item_id in item_lookup
        ]
        return _join(
            f"现在来到第{chapter.order + 1}个叙事区段，《{_clean_spoken_text(chapter.title)}》。",
            chapter.lead_in,
            f"这一段将看到：{'、'.join(titles)}。" if titles else "",
        )

    if kind == "artwork":
        item = _find_artwork(exhibition, reference)
        obj = item.object
        title = _clean_spoken_text(item.display_title or obj.title_original or obj.title)
        metadata_parts = [
            f"作者或制作者：{_clean_spoken_text(obj.maker or obj.creator)}"
            if obj.maker or obj.creator
            else "",
            f"年代：{_clean_spoken_text(obj.date)}" if obj.date else "",
            f"材料与技法：{_clean_spoken_text(obj.medium or obj.material)}"
            if obj.medium or obj.material
            else "",
            f"现藏于{_clean_spoken_text(obj.institution)}" if obj.institution else "",
        ]
        labels = "".join(
            _sentence(sentence.text)
            for sentence in item.label_sentences
            if sentence.type != "institution_fact"
            or re.search(r"[\u3400-\u9fff]", sentence.text)
        )
        return _join(
            f"请看《{title}》。",
            "，".join(part for part in metadata_parts if part),
            labels,
            item.relation,
        )

    if kind == "epilogue":
        question_count = len(
            [question for question in exhibition.epilogue.open_questions if question.strip()]
        )
        boundaries = "；".join(
            _clean_spoken_text(boundary) for boundary in exhibition.epilogue.material_boundary
        )
        return _join(
            "现在来到展览的尾声。",
            exhibition.epilogue.text,
            (
                f"如果你愿意，尾声的展后题笺里还留着{question_count}则问题。"
                "你可以选择其中一则，与彦远继续讨论，也可以暂时不回答。"
                if question_count
                else ""
            ),
            f"本展仍有这些材料边界：{boundaries}。" if boundaries else "",
            "感谢参观，期待与你在下一次卧游中再见。",
        )

    # FastAPI validates the Literal before this function, but keeping a local
    # guard makes direct service use safe as well.
    raise AudioGuideReferenceError(
        "invalid_kind", "Unknown audio guide kind.", status_code=422
    )


class AudioGuideService:
    def __init__(self, provider: AliyunTtsProvider, output_dir: Path) -> None:
        self.provider = provider
        self.output_dir = Path(output_dir).resolve()
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    def cache_key(self, text: str) -> str:
        material = "\0".join(
            (
                self.provider.model,
                self.provider.voice,
                self.provider.instruction,
                self.provider.audio_format,
                text,
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    async def get_audio(
        self,
        exhibition: Exhibition,
        kind: AudioGuideKind,
        reference: str | None = None,
    ) -> AudioGuideResult:
        text = build_audio_guide_text(exhibition, kind, reference)
        key = self.cache_key(text)
        path = (self.output_dir / f"{key}.mp3").resolve()
        if path.parent != self.output_dir:
            raise RuntimeError("Audio cache path escaped its configured root.")

        if self._cached_file_is_valid(path):
            return self._result(path, "HIT", text)

        lock = await self._lock_for(key)
        async with lock:
            if self._cached_file_is_valid(path):
                return self._result(path, "HIT", text)

            generated = await self.provider.synthesize(text)
            if not is_valid_mp3(generated.audio_bytes):
                # Injected providers are held to the same boundary as the real
                # provider before bytes can enter the persistent cache.
                raise RuntimeError("The TTS provider returned invalid MP3 data.")
            await asyncio.to_thread(self._atomic_write, path, generated.audio_bytes)
            return self._result(path, "MISS", text)

    async def _lock_for(self, key: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    @staticmethod
    def _cached_file_is_valid(path: Path) -> bool:
        try:
            return path.is_file() and is_valid_mp3(path.read_bytes())
        except OSError:
            return False

    def _result(
        self, path: Path, cache_status: Literal["HIT", "MISS"], text: str
    ) -> AudioGuideResult:
        return AudioGuideResult(
            path=path,
            cache_status=cache_status,
            model=self.provider.model,
            voice=self.provider.voice,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.stem}-",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if temporary_name:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    pass
