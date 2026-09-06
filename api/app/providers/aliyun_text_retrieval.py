"""Credential-safe Alibaba Cloud text embedding and rerank client.

The provider deliberately targets the synchronous Model Studio inference
endpoints while exposing ordinary synchronous Python methods.  It owns only
remote inference and response validation; index persistence, vector search and
BM25/RRF fallback remain responsibilities of the retrieval layer.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Sequence
from urllib.parse import urlsplit

import httpx


EMBEDDING_PATH = "/api/v1/services/embeddings/text-embedding/text-embedding"
RERANK_PATH = "/compatible-api/v1/reranks"

DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding"
DEFAULT_EMBEDDING_DIMENSION = 768
SUPPORTED_EMBEDDING_DIMENSIONS = frozenset(
    {256, 512, 768, 1024, 1536, 2048, 2560}
)
DEFAULT_QUERY_INSTRUCT = (
    "Given a museum collection question in any language, retrieve collection "
    "records and evidence passages that help answer it across cultures."
)

DEFAULT_RERANK_MODEL = "qwen3-rerank"
DEFAULT_RERANK_INSTRUCT = (
    "Given a museum collection question, retrieve collection records and "
    "evidence passages that best answer the question."
)

MAX_EMBEDDING_BATCH_SIZE = 20
# Model Studio documents both a 128K per-item ceiling and a 128K per-request
# processing ceiling.  Without importing the provider tokenizer, counting
# Unicode code points is a deliberately conservative preflight guard for the
# short catalogue records used by this application.  The upstream token limit
# remains authoritative.
MAX_EMBEDDING_TEXT_CHARACTERS = 128_000
MAX_EMBEDDING_BATCH_CHARACTERS = 128_000

MAX_RERANK_DOCUMENTS = 500
MAX_RERANK_ITEM_CHARACTERS = 4_000
MAX_RERANK_REQUEST_CHARACTERS = 30_000

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 0.5

TextType = Literal["document", "query"]


class AliyunTextRetrievalProviderError(RuntimeError):
    """A structured failure that never includes credentials or request text."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        http_status: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.http_status = http_status
        self.request_id = request_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": "aliyun-model-studio",
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "httpStatus": self.http_status,
            "requestId": self.request_id,
        }


@dataclass(frozen=True, slots=True)
class ProviderRequestMetadata:
    request_id: str | None
    model: str
    attempts: int
    input_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]
    text_type: TextType
    dimension: int
    metadata: ProviderRequestMetadata


@dataclass(frozen=True, slots=True)
class RerankedDocument:
    index: int
    document: str
    relevance_score: float

    @property
    def original_index(self) -> int:
        """Explicit alias for callers that need to stress source-list order."""

        return self.index

    @property
    def score(self) -> float:
        """Compatibility alias for generic ranking code."""

        return self.relevance_score


@dataclass(frozen=True, slots=True)
class RerankBatch:
    results: tuple[RerankedDocument, ...]
    candidate_count: int
    top_n: int
    metadata: ProviderRequestMetadata


class AliyunTextRetrievalProvider:
    """Call Qwen text embedding and rerank through one Beijing workspace."""

    max_embedding_batch_size = MAX_EMBEDDING_BATCH_SIZE

    def __init__(
        self,
        *,
        api_key: str,
        api_host: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        query_instruct: str = DEFAULT_QUERY_INSTRUCT,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        rerank_instruct: str = DEFAULT_RERANK_INSTRUCT,
        timeout_seconds: float = 30.0,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        allow_insecure: bool = False,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise AliyunTextRetrievalProviderError(
                "not_configured",
                "Alibaba Cloud text retrieval is not configured.",
            )
        if client is not None and transport is not None:
            raise AliyunTextRetrievalProviderError(
                "invalid_config",
                "Provide either an HTTP client or a transport, not both.",
            )
        if not isinstance(embedding_model, str) or not embedding_model.strip():
            raise AliyunTextRetrievalProviderError(
                "invalid_config", "The embedding model must not be blank."
            )
        if embedding_dimension not in SUPPORTED_EMBEDDING_DIMENSIONS:
            raise AliyunTextRetrievalProviderError(
                "invalid_config", "The embedding dimension is not supported."
            )
        if not isinstance(query_instruct, str) or not query_instruct.strip():
            raise AliyunTextRetrievalProviderError(
                "invalid_config", "The query embedding instruction must not be blank."
            )
        if not isinstance(rerank_model, str) or not rerank_model.strip():
            raise AliyunTextRetrievalProviderError(
                "invalid_config", "The rerank model must not be blank."
            )
        if not isinstance(rerank_instruct, str) or not rerank_instruct.strip():
            raise AliyunTextRetrievalProviderError(
                "invalid_config", "The rerank instruction must not be blank."
            )
        if timeout_seconds <= 0:
            raise AliyunTextRetrievalProviderError(
                "invalid_config", "The provider timeout must be positive."
            )
        if not isinstance(max_attempts, int) or not 1 <= max_attempts <= 5:
            raise AliyunTextRetrievalProviderError(
                "invalid_config", "Provider attempts must be between one and five."
            )
        if retry_delay_seconds < 0:
            raise AliyunTextRetrievalProviderError(
                "invalid_config", "The provider retry delay must not be negative."
            )

        self._api_key = api_key.strip()
        self.api_host = self._validate_http_origin(
            api_host, allow_insecure=allow_insecure
        )
        self.embedding_model = embedding_model.strip()
        self.embedding_dimension = embedding_dimension
        self.query_instruct = query_instruct.strip()
        self.rerank_model = rerank_model.strip()
        self.rerank_instruct = rerank_instruct.strip()
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self._client = client
        self._transport = transport

    @property
    def configured(self) -> bool:
        return bool(self._api_key and self.api_host)

    @property
    def model_name(self) -> str:
        """Common embedding-provider name used by the dense index layer."""

        return self.embedding_model

    def artifact_metadata(self) -> dict[str, Any]:
        """Return credential-free provider facts suitable for an index manifest."""

        return {
            "provider": "aliyun",
            "providerService": "aliyun-model-studio",
            "embeddingModel": self.embedding_model,
            "embeddingDimension": self.embedding_dimension,
        }

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> EmbeddingBatch:
        """Embed catalogue/evidence documents using the document representation."""

        return self._embed(
            texts,
            text_type="document",
            instruct=None,
            deadline=deadline,
        )

    def embed_queries(
        self,
        texts: Sequence[str],
        *,
        instruct: str | None = None,
        deadline: float | None = None,
    ) -> EmbeddingBatch:
        """Embed visitor queries with the retrieval-specific query representation."""

        clean_instruct = self._clean_instruct(
            instruct if instruct is not None else self.query_instruct,
            field_name="query embedding instruction",
        )
        return self._embed(
            texts,
            text_type="query",
            instruct=clean_instruct,
            deadline=deadline,
        )

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
        instruct: str | None = None,
        deadline: float | None = None,
    ) -> RerankBatch:
        """Rerank candidates and map every provider index to its original text."""

        clean_query = self._clean_text(
            query,
            field_name="rerank query",
            maximum_characters=MAX_RERANK_ITEM_CHARACTERS,
        )
        clean_documents = self._validate_text_batch(
            documents,
            field_name="rerank documents",
            maximum_items=MAX_RERANK_DOCUMENTS,
            maximum_item_characters=MAX_RERANK_ITEM_CHARACTERS,
            maximum_batch_characters=None,
        )
        clean_instruct = self._clean_instruct(
            instruct if instruct is not None else self.rerank_instruct,
            field_name="rerank instruction",
        )

        resolved_top_n = len(clean_documents) if top_n is None else top_n
        if (
            not isinstance(resolved_top_n, int)
            or isinstance(resolved_top_n, bool)
            or resolved_top_n <= 0
            or resolved_top_n > len(clean_documents)
        ):
            raise AliyunTextRetrievalProviderError(
                "invalid_request",
                "Rerank top_n must be between one and the candidate count.",
            )

        # Alibaba Cloud accounts for the query once per candidate.  The code
        # point total is a conservative no-tokenizer guard for museum prose.
        request_characters = (
            len(clean_query) * len(clean_documents)
            + sum(len(document) for document in clean_documents)
            + len(clean_instruct)
        )
        if request_characters > MAX_RERANK_REQUEST_CHARACTERS:
            raise AliyunTextRetrievalProviderError(
                "invalid_request",
                "The rerank request exceeds the conservative context limit.",
            )

        request_body = {
            "model": self.rerank_model,
            "query": clean_query,
            "documents": list(clean_documents),
            "top_n": resolved_top_n,
            "instruct": clean_instruct,
        }
        payload, response, attempts = self._post_json_with_retry(
            RERANK_PATH,
            request_body,
            deadline=deadline,
        )
        request_id = self._request_id(payload, response)
        results = self._parse_rerank_results(
            payload,
            clean_documents,
            expected_count=resolved_top_n,
            request_id=request_id,
        )
        usage = self._usage(payload)
        return RerankBatch(
            results=results,
            candidate_count=len(clean_documents),
            top_n=resolved_top_n,
            metadata=ProviderRequestMetadata(
                request_id=request_id,
                model=self.rerank_model,
                attempts=attempts,
                input_tokens=usage[0],
                total_tokens=usage[1],
            ),
        )

    def _embed(
        self,
        texts: Sequence[str],
        *,
        text_type: TextType,
        instruct: str | None,
        deadline: float | None,
    ) -> EmbeddingBatch:
        clean_texts = self._validate_text_batch(
            texts,
            field_name=f"embedding {text_type} texts",
            maximum_items=MAX_EMBEDDING_BATCH_SIZE,
            maximum_item_characters=MAX_EMBEDDING_TEXT_CHARACTERS,
            maximum_batch_characters=MAX_EMBEDDING_BATCH_CHARACTERS,
        )
        parameters: dict[str, Any] = {
            "text_type": text_type,
            "dimension": self.embedding_dimension,
            "output_type": "dense",
        }
        if text_type == "query":
            if instruct is None:  # Defensive; embed_queries always supplies it.
                raise AliyunTextRetrievalProviderError(
                    "invalid_request",
                    "The query embedding instruction must not be blank.",
                )
            parameters["instruct"] = instruct

        request_body = {
            "model": self.embedding_model,
            "input": {"texts": list(clean_texts)},
            "parameters": parameters,
        }
        payload, response, attempts = self._post_json_with_retry(
            EMBEDDING_PATH,
            request_body,
            deadline=deadline,
        )
        request_id = self._request_id(payload, response)
        vectors = self._parse_embedding_vectors(
            payload,
            expected_count=len(clean_texts),
            request_id=request_id,
        )
        usage = self._usage(payload)
        return EmbeddingBatch(
            vectors=vectors,
            text_type=text_type,
            dimension=self.embedding_dimension,
            metadata=ProviderRequestMetadata(
                request_id=request_id,
                model=self.embedding_model,
                attempts=attempts,
                input_tokens=usage[0],
                total_tokens=usage[1],
            ),
        )

    def _post_json_with_retry(
        self,
        path: str,
        request_body: dict[str, Any],
        *,
        deadline: float | None,
    ) -> tuple[dict[str, Any], httpx.Response, int]:
        if deadline is not None and (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(float(deadline))
        ):
            raise AliyunTextRetrievalProviderError(
                "invalid_request", "The provider deadline is invalid."
            )
        last_error: AliyunTextRetrievalProviderError | None = None
        with self._client_context() as client:
            for attempt in range(1, self.max_attempts + 1):
                request_timeout = self._request_timeout(deadline)
                try:
                    response = client.post(
                        f"{self.api_host}{path}",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json=request_body,
                        timeout=request_timeout,
                    )
                except httpx.HTTPError as exc:
                    if self._deadline_expired(deadline):
                        raise self._deadline_error() from exc
                    last_error = AliyunTextRetrievalProviderError(
                        "provider_network_error",
                        "Alibaba Cloud text retrieval could not be reached.",
                        retryable=True,
                    )
                    if attempt >= self.max_attempts:
                        raise last_error from exc
                    self._wait_before_retry(attempt, deadline=deadline)
                    continue

                if self._deadline_expired(deadline):
                    raise self._deadline_error()

                # Never follow a workspace response to another origin.  A 3xx
                # is a provider failure here, not a navigation instruction.
                if response.status_code >= 300:
                    request_id = self._request_id_from_error(response)
                    retryable = response.status_code == 429 or response.status_code >= 500
                    last_error = AliyunTextRetrievalProviderError(
                        "provider_http_error",
                        "Alibaba Cloud text retrieval did not accept the request.",
                        retryable=retryable,
                        http_status=response.status_code,
                        request_id=request_id,
                    )
                    if not retryable or attempt >= self.max_attempts:
                        raise last_error
                    self._wait_before_retry(attempt, deadline=deadline)
                    continue

                try:
                    payload = response.json()
                except (json.JSONDecodeError, ValueError) as exc:
                    raise AliyunTextRetrievalProviderError(
                        "provider_response_error",
                        "Alibaba Cloud returned an unreadable text retrieval response.",
                        request_id=self._request_id({}, response),
                    ) from exc
                if not isinstance(payload, dict):
                    raise AliyunTextRetrievalProviderError(
                        "provider_response_error",
                        "Alibaba Cloud returned an unusable text retrieval response.",
                        request_id=self._request_id({}, response),
                    )
                return payload, response, attempt

        raise last_error or AliyunTextRetrievalProviderError(
            "provider_network_error",
            "Alibaba Cloud text retrieval could not be reached.",
            retryable=True,
        )

    @contextmanager
    def _client_context(self) -> Iterator[httpx.Client]:
        if self._client is not None:
            yield self._client
            return
        with httpx.Client(
            timeout=self.timeout_seconds,
            transport=self._transport,
            follow_redirects=False,
        ) as client:
            yield client

    def _request_timeout(self, deadline: float | None) -> float:
        if deadline is None:
            return self.timeout_seconds
        remaining = float(deadline) - time.perf_counter()
        if remaining <= 0:
            raise self._deadline_error()
        return min(self.timeout_seconds, remaining)

    @staticmethod
    def _deadline_expired(deadline: float | None) -> bool:
        return deadline is not None and time.perf_counter() >= float(deadline)

    @staticmethod
    def _deadline_error() -> AliyunTextRetrievalProviderError:
        return AliyunTextRetrievalProviderError(
            "provider_deadline_exceeded",
            "Alibaba Cloud text retrieval exceeded the request deadline.",
        )

    def _wait_before_retry(
        self,
        failed_attempt: int,
        *,
        deadline: float | None,
    ) -> None:
        delay = self.retry_delay_seconds * (2 ** (failed_attempt - 1))
        if deadline is not None:
            remaining = float(deadline) - time.perf_counter()
            # Do not start a sleep or another request that cannot fit inside
            # the caller's remaining wall-clock budget.
            if remaining <= 0 or delay >= remaining:
                raise self._deadline_error()
        if delay:
            time.sleep(delay)
        if self._deadline_expired(deadline):
            raise self._deadline_error()

    def _parse_embedding_vectors(
        self,
        payload: dict[str, Any],
        *,
        expected_count: int,
        request_id: str | None,
    ) -> tuple[tuple[float, ...], ...]:
        try:
            rows = payload["output"]["embeddings"]
        except (KeyError, TypeError) as exc:
            raise self._response_error(
                "Alibaba Cloud returned no embedding vectors.", request_id
            ) from exc
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise self._response_error(
                "Alibaba Cloud returned an unexpected embedding count.", request_id
            )

        ordered: list[tuple[float, ...] | None] = [None] * expected_count
        for row in rows:
            if not isinstance(row, dict):
                raise self._response_error(
                    "Alibaba Cloud returned an invalid embedding row.", request_id
                )
            index = row.get("text_index")
            embedding = row.get("embedding")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < expected_count
                or ordered[index] is not None
            ):
                raise self._response_error(
                    "Alibaba Cloud returned an invalid embedding index.", request_id
                )
            if (
                not isinstance(embedding, list)
                or len(embedding) != self.embedding_dimension
            ):
                raise self._response_error(
                    "Alibaba Cloud returned an invalid embedding dimension.", request_id
                )
            vector: list[float] = []
            for value in embedding:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise self._response_error(
                        "Alibaba Cloud returned a non-numeric embedding.", request_id
                    )
                number = float(value)
                if not math.isfinite(number):
                    raise self._response_error(
                        "Alibaba Cloud returned a non-finite embedding.", request_id
                    )
                vector.append(number)
            ordered[index] = tuple(vector)

        if any(vector is None for vector in ordered):
            raise self._response_error(
                "Alibaba Cloud returned incomplete embedding indices.", request_id
            )
        return tuple(vector for vector in ordered if vector is not None)

    def _parse_rerank_results(
        self,
        payload: dict[str, Any],
        documents: tuple[str, ...],
        *,
        expected_count: int,
        request_id: str | None,
    ) -> tuple[RerankedDocument, ...]:
        rows = payload.get("results")
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise self._response_error(
                "Alibaba Cloud returned an unexpected rerank result count.", request_id
            )

        observed_indices: set[int] = set()
        results: list[RerankedDocument] = []
        previous_score: float | None = None
        for row in rows:
            if not isinstance(row, dict):
                raise self._response_error(
                    "Alibaba Cloud returned an invalid rerank row.", request_id
                )
            index = row.get("index")
            score_value = row.get("relevance_score")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(documents)
                or index in observed_indices
            ):
                raise self._response_error(
                    "Alibaba Cloud returned an invalid rerank index.", request_id
                )
            if isinstance(score_value, bool) or not isinstance(
                score_value, (int, float)
            ):
                raise self._response_error(
                    "Alibaba Cloud returned an invalid rerank score.", request_id
                )
            score = float(score_value)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise self._response_error(
                    "Alibaba Cloud returned an invalid rerank score.", request_id
                )
            if previous_score is not None and score > previous_score:
                raise self._response_error(
                    "Alibaba Cloud returned unsorted rerank results.", request_id
                )
            observed_indices.add(index)
            previous_score = score
            results.append(
                RerankedDocument(
                    index=index,
                    document=documents[index],
                    relevance_score=score,
                )
            )
        return tuple(results)

    @staticmethod
    def _usage(payload: dict[str, Any]) -> tuple[int | None, int | None]:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None, None
        input_tokens = usage.get("input_tokens")
        total_tokens = usage.get("total_tokens")
        return (
            input_tokens
            if isinstance(input_tokens, int) and not isinstance(input_tokens, bool)
            else None,
            total_tokens
            if isinstance(total_tokens, int) and not isinstance(total_tokens, bool)
            else None,
        )

    @classmethod
    def _request_id(
        cls, payload: dict[str, Any], response: httpx.Response
    ) -> str | None:
        for value in (
            payload.get("request_id"),
            payload.get("id"),
            response.headers.get("x-request-id"),
            response.headers.get("x-dashscope-request-id"),
        ):
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @classmethod
    def _request_id_from_error(cls, response: httpx.Response) -> str | None:
        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        return cls._request_id(payload, response)

    @staticmethod
    def _validate_text_batch(
        texts: Sequence[str],
        *,
        field_name: str,
        maximum_items: int,
        maximum_item_characters: int,
        maximum_batch_characters: int | None,
    ) -> tuple[str, ...]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise AliyunTextRetrievalProviderError(
                "invalid_request", f"{field_name.capitalize()} must be a sequence."
            )
        if not 1 <= len(texts) <= maximum_items:
            raise AliyunTextRetrievalProviderError(
                "invalid_request",
                f"{field_name.capitalize()} exceed the configured batch limit.",
            )
        cleaned = tuple(
            AliyunTextRetrievalProvider._clean_text(
                text,
                field_name=field_name,
                maximum_characters=maximum_item_characters,
            )
            for text in texts
        )
        if (
            maximum_batch_characters is not None
            and sum(len(text) for text in cleaned) > maximum_batch_characters
        ):
            raise AliyunTextRetrievalProviderError(
                "invalid_request",
                f"{field_name.capitalize()} exceed the conservative request limit.",
            )
        return cleaned

    @staticmethod
    def _clean_text(
        value: str, *, field_name: str, maximum_characters: int
    ) -> str:
        if not isinstance(value, str) or not value.strip():
            raise AliyunTextRetrievalProviderError(
                "invalid_request", f"The {field_name} must not be blank."
            )
        clean_value = value.strip()
        if len(clean_value) > maximum_characters:
            raise AliyunTextRetrievalProviderError(
                "invalid_request", f"The {field_name} exceeds the provider limit."
            )
        return clean_value

    @staticmethod
    def _clean_instruct(value: str, *, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise AliyunTextRetrievalProviderError(
                "invalid_request", f"The {field_name} must not be blank."
            )
        return value.strip()

    @staticmethod
    def _response_error(
        message: str, request_id: str | None
    ) -> AliyunTextRetrievalProviderError:
        return AliyunTextRetrievalProviderError(
            "provider_response_error", message, request_id=request_id
        )

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
            raise AliyunTextRetrievalProviderError(
                "invalid_config",
                "The Alibaba Cloud text retrieval host must be a secure origin.",
            )
        return f"{parsed.scheme}://{parsed.netloc}"
