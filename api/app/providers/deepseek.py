from __future__ import annotations

import asyncio
import json
import logging
import re
import ssl
from typing import Any

import httpx

from ..config import Settings

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """A provider failed without exposing credentials or response secrets."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "provider_error",
        public_message: str = (
            "AI 策展服务暂时不可用。本次问题已保留，请稍后重试。"
        ),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = public_message


class DeepSeekProvider:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.deepseek_api_key
        self.model = settings.deepseek_model
        self.base_url = settings.deepseek_base_url
        self.timeout_seconds = settings.deepseek_timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def generate_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ProviderError(
                "DeepSeek is not configured",
                code="provider_not_configured",
                public_message=(
                    "AI 策展服务尚未配置；系统将继续使用馆藏证据生成基础展览。"
                ),
            )

        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        async def request() -> httpx.Response:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                return await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=request_body,
                )

        try:
            # ``httpx.Timeout(90)`` limits each socket operation, not the total
            # request wall clock.  A response arriving in small chunks can
            # therefore run well past 90 seconds and consume the whole curation
            # job.  This outer deadline is the actual end-to-end ceiling.
            response = await asyncio.wait_for(
                request(), timeout=self.timeout_seconds
            )
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            content = choice["message"]["content"]
            try:
                return self._parse_json(content)
            except (TypeError, ValueError, json.JSONDecodeError):
                # Falling back to deterministic prose is by design, but doing it
                # without saying why hid a whole language's worth of output. The
                # finish reason distinguishes a truncated completion from a
                # malformed one, and the tail shows where it stopped. Model
                # output only -- never the prompt or the key.
                logger.warning(
                    "DeepSeek JSON unparsable finish_reason=%s completion_tokens=%s length=%s tail=%r",
                    choice.get("finish_reason"),
                    (body.get("usage") or {}).get("completion_tokens"),
                    len(content) if isinstance(content, str) else None,
                    content[-160:] if isinstance(content, str) else content,
                )
                raise
        except asyncio.CancelledError:
            # Cancellation belongs to the calling job. Turning it into a
            # provider failure would defeat the job deadline and leave work
            # running after the visitor has moved on.
            raise
        except asyncio.TimeoutError as exc:
            raise ProviderError(
                "DeepSeek request exceeded its wall-clock timeout",
                code="provider_timeout",
                public_message=(
                    "AI 策展服务响应超时。系统会继续使用馆藏证据生成基础展览；"
                    "如仍未完成，请保留当前问题并重试。"
                ),
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderError(
                "DeepSeek transport timed out",
                code="provider_timeout",
                public_message=(
                    "AI 策展服务响应超时。系统会继续使用馆藏证据生成基础展览；"
                    "如仍未完成，请保留当前问题并重试。"
                ),
            ) from exc
        except (httpx.RequestError, ssl.SSLError, OSError) as exc:
            # Custom transports and some TLS stacks surface the original
            # ssl.SSLError/OSError instead of wrapping it in RequestError.
            # Normalise all of them here so generator fallbacks always run and
            # raw certificate, proxy or filesystem details never reach a job.
            raise ProviderError(
                f"DeepSeek transport failed: {type(exc).__name__}",
                code="provider_network_error",
                public_message=(
                    "AI 策展服务暂时无法连接。系统会继续使用馆藏证据生成基础展览；"
                    "如仍未完成，请稍后重试。"
                ),
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"DeepSeek returned HTTP {exc.response.status_code}",
                code="provider_response_error",
                public_message=(
                    "AI 策展服务暂时无法响应。系统会继续使用馆藏证据生成基础展览；"
                    "如仍未完成，请稍后重试。"
                ),
            ) from exc
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(
                f"DeepSeek returned an unusable response: {type(exc).__name__}",
                code="provider_response_error",
                public_message=(
                    "AI 策展文本暂时无法使用；系统将改用馆藏证据生成基础展览。"
                ),
            ) from exc

    @staticmethod
    def _parse_json(content: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            raise TypeError("Model content is not text")
        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise TypeError("Model JSON root must be an object")
        return parsed
