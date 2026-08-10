from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator
from urllib.parse import urlsplit

import httpx


SPEECH_SYNTHESIS_PATH = "/api/v1/services/audio/tts/SpeechSynthesizer"
DEFAULT_MODEL = "qwen-audio-3.0-tts-plus"
DEFAULT_VOICE = "qwen-audio-3.0-tts-plus-longyulianrong"
DEFAULT_INSTRUCTION = (
    "请使用专业、克制、清晰的博物馆导览播音主持声线，语速稍慢，停连自然，避免夸张表演。"
)
DEFAULT_FORMAT = "mp3"
DEFAULT_SAMPLE_RATE = 24_000
DEFAULT_MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARACTERS = 20_000
MAX_ATTEMPTS = 2
DEFAULT_RETRY_DELAY_SECONDS = 0.75


class AliyunTtsProviderError(RuntimeError):
    """A structured, credential-safe Qwen TTS failure."""

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
class GeneratedSpeech:
    audio_bytes: bytes
    request_id: str | None
    model: str
    voice: str
    audio_format: str


def is_valid_mp3(payload: bytes) -> bool:
    """Recognise an MP3 by a plausible MPEG audio frame, not MIME alone.

    Signed OSS downloads sometimes use ``application/octet-stream``.  A small
    frame-header check therefore provides a more reliable boundary than the
    response header while still rejecting HTML/JSON error pages.
    """

    if len(payload) < 4:
        return False

    offset = 0
    if payload.startswith(b"ID3"):
        if len(payload) < 10 or any(value & 0x80 for value in payload[6:10]):
            return False
        tag_size = (
            (payload[6] << 21)
            | (payload[7] << 14)
            | (payload[8] << 7)
            | payload[9]
        )
        offset = 10 + tag_size + (10 if payload[5] & 0x10 else 0)
        if offset + 4 > len(payload):
            return False

    # Metadata and padding may appear before the first audio frame.  Keep the
    # scan bounded so validation stays linear and cheap for large guide files.
    scan_end = min(len(payload) - 3, offset + 64 * 1024)
    for index in range(offset, scan_end):
        first, second, third = payload[index : index + 3]
        if first != 0xFF or second & 0xE0 != 0xE0:
            continue
        version = (second >> 3) & 0x03
        layer = (second >> 1) & 0x03
        bitrate_index = (third >> 4) & 0x0F
        sample_rate_index = (third >> 2) & 0x03
        if (
            version != 0x01
            and layer != 0x00
            and bitrate_index not in {0x00, 0x0F}
            and sample_rate_index != 0x03
        ):
            return True
    return False


class AliyunTtsProvider:
    def __init__(
        self,
        *,
        api_key: str,
        api_host: str,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        instruction: str = DEFAULT_INSTRUCTION,
        timeout_seconds: float = 90.0,
        max_audio_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        allow_insecure: bool = False,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise AliyunTtsProviderError(
                "not_configured", "Alibaba Cloud speech synthesis is not configured."
            )
        if client is not None and transport is not None:
            raise AliyunTtsProviderError(
                "invalid_config", "Provide either an HTTP client or a transport, not both."
            )
        if not model.strip() or not voice.strip() or not instruction.strip():
            raise AliyunTtsProviderError(
                "invalid_config", "The TTS model, voice, and instruction must not be blank."
            )
        if timeout_seconds <= 0 or max_audio_bytes <= 4 or retry_delay_seconds < 0:
            raise AliyunTtsProviderError(
                "invalid_config", "The TTS timeout, byte limit, or retry delay is invalid."
            )

        self.api_key = api_key.strip()
        self.api_host = self._validate_http_origin(api_host, allow_insecure=allow_insecure)
        self.model = model.strip()
        self.voice = voice.strip()
        self.instruction = instruction.strip()
        self.audio_format = DEFAULT_FORMAT
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.timeout_seconds = timeout_seconds
        self.max_audio_bytes = max_audio_bytes
        self.retry_delay_seconds = retry_delay_seconds
        self.allow_insecure = allow_insecure
        self._client = client
        self._transport = transport

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_host)

    async def synthesize(self, text: str) -> GeneratedSpeech:
        clean_text = text.strip() if isinstance(text, str) else ""
        if not clean_text:
            raise AliyunTtsProviderError(
                "invalid_request", "Speech synthesis text must not be blank."
            )
        if len(clean_text) > MAX_TEXT_CHARACTERS:
            raise AliyunTtsProviderError(
                "invalid_request", "Speech synthesis text exceeds the provider limit."
            )

        request_body: dict[str, Any] = {
            "model": self.model,
            "input": {
                "text": clean_text,
                "voice": self.voice,
                "format": self.audio_format,
                "sample_rate": self.sample_rate,
                "rate": 0.94,
                "pitch": 0.98,
                "volume": 55,
                "language_hints": ["zh"],
                "instruction": self.instruction,
                "seed": 7,
                "enable_aigc_tag": True,
            },
        }

        last_error: AliyunTtsProviderError | None = None
        async with self._client_context() as client:
            for attempt in range(MAX_ATTEMPTS):
                try:
                    payload, request_id = await self._request_generation(
                        client, request_body
                    )
                    audio_url = self._extract_audio_url(payload)
                    audio_bytes = await self._download_mp3(client, audio_url)
                    return GeneratedSpeech(
                        audio_bytes=audio_bytes,
                        request_id=request_id,
                        model=self.model,
                        voice=self.voice,
                        audio_format=self.audio_format,
                    )
                except AliyunTtsProviderError as exc:
                    last_error = exc
                    if not exc.retryable or attempt + 1 >= MAX_ATTEMPTS:
                        raise
                    if self.retry_delay_seconds:
                        await asyncio.sleep(self.retry_delay_seconds)

        # The loop either returns or raises; this keeps type checkers honest.
        raise last_error or AliyunTtsProviderError(
            "provider_network_error",
            "Alibaba Cloud speech synthesis could not be reached after one retry.",
            retryable=True,
        )

    @asynccontextmanager
    async def _client_context(self) -> AsyncIterator[httpx.AsyncClient]:
        if self._client is not None:
            yield self._client
            return
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            transport=self._transport,
            follow_redirects=False,
        ) as client:
            yield client

    async def _request_generation(
        self, client: httpx.AsyncClient, request_body: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None]:
        try:
            response = await client.post(
                f"{self.api_host}{SPEECH_SYNTHESIS_PATH}",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_body,
                follow_redirects=False,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            upstream_status = exc.response.status_code
            retryable = upstream_status == 429 or upstream_status >= 500
            raise AliyunTtsProviderError(
                "provider_http_error",
                "Alibaba Cloud speech synthesis did not accept the request.",
                retryable=retryable,
                http_status=upstream_status,
            ) from exc
        except httpx.HTTPError as exc:
            raise AliyunTtsProviderError(
                "provider_network_error",
                "Alibaba Cloud speech synthesis could not be reached.",
                retryable=True,
            ) from exc

        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise AliyunTtsProviderError(
                "provider_response_error",
                "Alibaba Cloud returned an unreadable speech response.",
                retryable=True,
            ) from exc
        if not isinstance(payload, dict):
            raise AliyunTtsProviderError(
                "provider_response_error",
                "Alibaba Cloud returned an unusable speech response.",
                retryable=True,
            )
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            request_id = response.headers.get("x-request-id") or None
        return payload, request_id

    def _extract_audio_url(self, payload: dict[str, Any]) -> str:
        try:
            audio_url = payload["output"]["audio"]["url"]
        except (KeyError, TypeError) as exc:
            raise AliyunTtsProviderError(
                "provider_response_error",
                "Alibaba Cloud returned no synthesized audio.",
                retryable=True,
            ) from exc
        if not isinstance(audio_url, str) or not audio_url.strip():
            raise AliyunTtsProviderError(
                "provider_response_error",
                "Alibaba Cloud returned no synthesized audio.",
                retryable=True,
            )
        return self._validate_asset_url(audio_url.strip())

    async def _download_mp3(self, client: httpx.AsyncClient, audio_url: str) -> bytes:
        try:
            async with client.stream(
                "GET", audio_url, follow_redirects=False
            ) as response:
                response.raise_for_status()
                # Explicitly disabling redirects prevents a trusted signed URL
                # from bouncing this server to an arbitrary fetch target.
                self._validate_asset_url(str(response.url))
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        advertised_size = int(content_length)
                    except ValueError as exc:
                        raise AliyunTtsProviderError(
                            "audio_validation_error",
                            "The synthesized audio had invalid size metadata.",
                            retryable=True,
                        ) from exc
                    if advertised_size > self.max_audio_bytes:
                        raise AliyunTtsProviderError(
                            "audio_too_large",
                            "The synthesized audio exceeded the configured size limit.",
                        )

                buffer = bytearray()
                async for chunk in response.aiter_bytes():
                    buffer.extend(chunk)
                    if len(buffer) > self.max_audio_bytes:
                        raise AliyunTtsProviderError(
                            "audio_too_large",
                            "The synthesized audio exceeded the configured size limit.",
                        )
        except AliyunTtsProviderError:
            raise
        except httpx.HTTPStatusError as exc:
            upstream_status = exc.response.status_code
            raise AliyunTtsProviderError(
                "audio_download_error",
                "The synthesized audio could not be downloaded.",
                retryable=upstream_status == 429 or upstream_status >= 500,
                http_status=upstream_status,
            ) from exc
        except httpx.HTTPError as exc:
            raise AliyunTtsProviderError(
                "audio_download_error",
                "The synthesized audio could not be downloaded.",
                retryable=True,
            ) from exc

        audio_bytes = bytes(buffer)
        if not is_valid_mp3(audio_bytes):
            raise AliyunTtsProviderError(
                "audio_validation_error",
                "The synthesized asset did not contain valid MP3 data.",
                retryable=True,
            )
        return audio_bytes

    @staticmethod
    def _validate_http_origin(api_host: str, *, allow_insecure: bool) -> str:
        parsed = urlsplit((api_host or "").strip())
        valid_scheme = parsed.scheme == "https" or (
            allow_insecure and parsed.scheme == "http"
        )
        if (
            not valid_scheme
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise AliyunTtsProviderError(
                "invalid_config", "The Alibaba Cloud TTS host must be a secure origin."
            )
        return f"{parsed.scheme}://{parsed.netloc}"

    def _validate_asset_url(self, audio_url: str) -> str:
        parsed = urlsplit(audio_url)
        hostname = (parsed.hostname or "").casefold()
        trusted_aliyun_host = hostname.endswith(".aliyuncs.com")
        if (
            parsed.scheme not in {"http", "https"}
            or not trusted_aliyun_host
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise AliyunTtsProviderError(
                "provider_response_error",
                "Alibaba Cloud returned an unsafe audio location.",
            )
        # The official non-streaming example currently returns an HTTP OSS URL
        # even though the object is also available over HTTPS.  Upgrade only a
        # trusted Alibaba Cloud hostname; never enable generic insecure fetches.
        if parsed.scheme == "http":
            parsed = parsed._replace(scheme="https")
        return parsed.geturl()
