from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import ssl
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

from ..config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisionImage:
    """One trusted museum image supplied to the visual label model.

    ``payload`` is deliberately transient.  It is encoded only while building
    the provider request and is never copied into an exhibition record or log.
    The marker fields keep a multimodal response bound to the correct object.
    """

    object_id: str
    evidence_id: str
    payload: bytes
    mime_type: str = "image/webp"


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
    # Generator feature detection keeps legacy/fake providers compatible while
    # making the production provider opt in to the bounded relevance-audit pass.
    supports_retrieval_audit = True
    supports_retrieval_query_planning = True

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.deepseek_api_key
        self.model = settings.deepseek_model
        self.query_review_thinking = settings.deepseek_query_review_thinking
        self.labels_model = settings.deepseek_labels_model
        self.base_url = settings.deepseek_base_url
        self.timeout_seconds = settings.deepseek_timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def generate_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        return await self._generate_json(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            # These calls return a constrained JSON contract rather than an
            # open-ended reasoning trace. Disabling thinking keeps the body
            # bounded and avoids a long streamed response consuming the whole
            # visitor-facing generation deadline.
            thinking={"type": "disabled"},
            max_tokens=4096,
        )

    async def generate_retrieval_audit_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the evidence gate deterministically rather than creatively."""

        return await self._generate_json(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            thinking={"type": "disabled"},
            max_tokens=6144 if user_payload.get("retrievalContract", {}).get("conditionContractVersion") else 4096,
            temperature=0.0,
        )

    async def generate_retrieval_query_plan_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Translate visitor wording into bounded catalogue search terms.

        This stage has no collection records and no authority to accept an
        object.  Its output is deliberately small because all resulting
        candidates still pass the source-bound retrieval audit.
        """

        review_thinking = "recallDraft" in user_payload and self.query_review_thinking
        options: dict[str, Any] = {}
        if review_thinking:
            options["reasoning_effort"] = "low"
        return await self._generate_json(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            thinking={"type": "enabled" if review_thinking else "disabled"},
            # Independent intent compilation (and legacy reviews) need a larger
            # response; initial recall drafting keeps its smaller allowance.
            max_tokens=4096 if review_thinking else (2400 if "recallDraft" in user_payload or "draftPlan" in user_payload else 1200),
            temperature=0.0,
            **options,
        )

    async def generate_json_with_images(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        images: Sequence[VisionImage],
    ) -> dict[str, Any]:
        """Generate JSON from text plus object-bound museum images.

        Images are sent as 1024px WebP data URLs prepared by the application's
        existing bounded image cache.  That route is important for institutions
        such as AIC, whose image host requires an identifying request header
        that a remote model-side fetch cannot provide.
        """

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(user_payload, ensure_ascii=False),
            }
        ]
        for image in images:
            marker = (
                "以下馆藏图像仅属于 "
                f"objectId={image.object_id}；imageEvidenceId={image.evidence_id}。"
            )
            encoded = base64.b64encode(image.payload).decode("ascii")
            content.extend(
                [
                    {"type": "text", "text": marker},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image.mime_type};base64,{encoded}",
                            "detail": "original",
                        },
                    },
                ]
            )

        return await self._generate_json(
            model=self.labels_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            thinking={"type": "disabled"},
            max_tokens=1600,
        )

    async def generate_qrel_vision_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        images: Sequence[VisionImage],
    ) -> dict[str, Any]:
        """Batch bounded qrel image checks without changing visitor-label limits."""

        content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}]
        for image in images:
            encoded = base64.b64encode(image.payload).decode("ascii")
            content.extend([
                {"type": "text", "text": f"图像仅属于 objectId={image.object_id}。"},
                {"type": "image_url", "image_url": {"url": f"data:{image.mime_type};base64,{encoded}", "detail": "original"}},
            ])
        return await self._generate_json(
            model=self.labels_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
            thinking={"type": "disabled"},
            max_tokens=4096,
            temperature=0.0,
        )

    async def _generate_json(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        thinking: dict[str, str] | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ProviderError(
                "DeepSeek is not configured",
                code="provider_not_configured",
                public_message=(
                    "AI 策展服务尚未配置；系统将继续使用馆藏证据生成基础展览。"
                ),
            )

        request_body = {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": temperature,
        }
        if thinking is not None:
            request_body["thinking"] = thinking
        if reasoning_effort is not None:
            if reasoning_effort not in {"low", "high", "max"} or thinking != {"type": "enabled"}:
                raise ValueError("reasoning effort requires enabled thinking and a supported effort")
            request_body["reasoning_effort"] = reasoning_effort
        if thinking == {"type": "enabled"}:
            request_body.pop("temperature", None)
        if max_tokens is not None:
            request_body["max_tokens"] = max_tokens

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
