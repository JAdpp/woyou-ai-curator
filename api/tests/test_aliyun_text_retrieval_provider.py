from __future__ import annotations

import json
import time

import httpx
import pytest

from app.dense_retrieval import EmbeddingProviderSpec, _embed_queries
from app.providers.aliyun_text_retrieval import (
    DEFAULT_QUERY_INSTRUCT,
    DEFAULT_RERANK_INSTRUCT,
    EMBEDDING_PATH,
    MAX_EMBEDDING_BATCH_SIZE,
    MAX_EMBEDDING_BATCH_CHARACTERS,
    MAX_RERANK_DOCUMENTS,
    MAX_RERANK_ITEM_CHARACTERS,
    RERANK_PATH,
    AliyunTextRetrievalProvider,
    AliyunTextRetrievalProviderError,
)


API_HOST = "https://workspace.cn-beijing.maas.aliyuncs.com"


def test_provider_exposes_dense_adapter_contract_without_exposing_credentials() -> None:
    provider = AliyunTextRetrievalProvider(
        api_key="test-only-key",
        api_host=API_HOST,
    )

    assert provider.model_name == "qwen3.7-text-embedding"
    assert provider.artifact_metadata() == {
        "provider": "aliyun",
        "providerService": "aliyun-model-studio",
        "embeddingModel": "qwen3.7-text-embedding",
        "embeddingDimension": 768,
    }
    assert "test-only-key" not in repr(provider.artifact_metadata())


TEST_KEY = "server-secret-that-must-not-leak"


def _embedding_response(
    *,
    dimension: int,
    indices: tuple[int, ...] = (0,),
    request_id: str = "embed-request-1",
) -> dict[str, object]:
    return {
        "status_code": 200,
        "request_id": request_id,
        "output": {
            "embeddings": [
                {
                    "text_index": index,
                    "embedding": [float(index + 1)] * dimension,
                }
                for index in indices
            ]
        },
        "usage": {"input_tokens": 17, "total_tokens": 17},
    }


def _provider(
    handler,
    *,
    dimension: int = 256,
    max_attempts: int = 3,
) -> AliyunTextRetrievalProvider:
    return AliyunTextRetrievalProvider(
        api_key=TEST_KEY,
        api_host=API_HOST,
        embedding_dimension=dimension,
        max_attempts=max_attempts,
        retry_delay_seconds=0,
        transport=httpx.MockTransport(handler),
    )


def test_dense_adapter_chunks_hosted_queries_and_preserves_global_order() -> None:
    observed_batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        texts = payload["input"]["texts"]
        observed_batch_sizes.append(len(texts))
        return httpx.Response(
            200,
            json={
                "request_id": f"batch-{len(observed_batch_sizes)}",
                "output": {
                    "embeddings": [
                        {
                            "text_index": local_index,
                            "embedding": [float(text.removeprefix("query-"))] * 256,
                        }
                        for local_index, text in enumerate(texts)
                    ]
                },
            },
        )

    provider = _provider(handler)
    queries = [f"query-{index}" for index in range(45)]

    vectors = _embed_queries(
        provider,
        queries,
        query_instruct=DEFAULT_QUERY_INSTRUCT,
    )

    assert observed_batch_sizes == [20, 20, 5]
    assert [vector[0] for vector in vectors] == [float(index) for index in range(45)]


def test_embedding_spec_normalises_aliyun_default_dimension() -> None:
    spec = EmbeddingProviderSpec(
        provider="aliyun",
        model_name="qwen3.7-text-embedding",
        dimension=None,
        api_key=TEST_KEY,
        api_host=API_HOST,
    )

    assert spec.dimension == 768
    assert "\x00768\x00" in f"\x00{spec.fingerprint_material}\x00"


def test_deadline_stops_retry_before_a_second_request() -> None:
    request_count = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        time.sleep(0.02)
        return httpx.Response(503)

    provider = _provider(handler, max_attempts=3)
    with pytest.raises(AliyunTextRetrievalProviderError) as captured:
        provider.embed_documents(
            ["one"],
            deadline=time.perf_counter() + 0.005,
        )

    assert captured.value.code == "provider_deadline_exceeded"
    assert request_count == 1


def test_expired_deadline_sends_no_request() -> None:
    request_count = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(500)

    with pytest.raises(AliyunTextRetrievalProviderError) as captured:
        _provider(handler).rerank(
            "question",
            ["document"],
            top_n=1,
            deadline=time.perf_counter() - 1.0,
        )

    assert captured.value.code == "provider_deadline_exceeded"
    assert request_count == 0


def test_embeds_documents_with_native_api_and_restores_input_order() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == EMBEDDING_PATH
        assert request.headers["authorization"] == f"Bearer {TEST_KEY}"
        observed.update(json.loads(request.content))
        # The provider response order is not assumed to match input order.
        return httpx.Response(
            200,
            json=_embedding_response(dimension=256, indices=(1, 0)),
        )

    result = _provider(handler).embed_documents(["first object", "second object"])

    assert observed == {
        "model": "qwen3.7-text-embedding",
        "input": {"texts": ["first object", "second object"]},
        "parameters": {
            "text_type": "document",
            "dimension": 256,
            "output_type": "dense",
        },
    }
    assert result.vectors == (
        tuple([1.0] * 256),
        tuple([2.0] * 256),
    )
    assert result.text_type == "document"
    assert result.dimension == 256
    assert result.metadata.request_id == "embed-request-1"
    assert result.metadata.model == "qwen3.7-text-embedding"
    assert result.metadata.attempts == 1
    assert result.metadata.input_tokens == 17
    assert result.metadata.total_tokens == 17


def test_embeds_queries_with_query_text_type_and_instruction() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        return httpx.Response(200, json=_embedding_response(dimension=256))

    provider = _provider(handler)
    result = provider.embed_queries(["猫在不同文化中意味着什么？"])

    assert observed["parameters"] == {
        "text_type": "query",
        "dimension": 256,
        "output_type": "dense",
        "instruct": DEFAULT_QUERY_INSTRUCT,
    }
    assert result.text_type == "query"

    custom_result = provider.embed_queries(
        ["blue pigment"], instruct="Retrieve material-history evidence."
    )
    assert custom_result.vectors[0][0] == 1.0
    assert observed["parameters"]["instruct"] == (
        "Retrieve material-history evidence."
    )


def test_embedding_retries_network_and_rate_limit_then_reports_attempts() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("temporary outage", request=request)
        if attempts == 2:
            return httpx.Response(
                429,
                json={"code": "Throttling", "request_id": "retry-request"},
            )
        return httpx.Response(
            200,
            headers={"x-request-id": "header-request"},
            json={
                "output": {
                    "embeddings": [
                        {"text_index": 0, "embedding": [0.25] * 256}
                    ]
                }
            },
        )

    result = _provider(handler).embed_documents(["object text"])

    assert attempts == 3
    assert result.metadata.attempts == 3
    assert result.metadata.request_id == "header-request"


def test_authentication_failure_is_not_retried_and_never_exposes_key() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            401,
            json={
                "code": "InvalidApiKey",
                "message": f"bad credential {TEST_KEY}",
                "request_id": "auth-request",
            },
        )

    with pytest.raises(AliyunTextRetrievalProviderError) as captured:
        _provider(handler).embed_documents(["object text"])

    error = captured.value
    assert attempts == 1
    assert error.code == "provider_http_error"
    assert error.http_status == 401
    assert error.retryable is False
    assert error.request_id == "auth-request"
    assert TEST_KEY not in str(error)
    assert TEST_KEY not in repr(error.to_dict())


def test_workspace_redirect_is_rejected_without_following_location() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            302,
            headers={"location": "https://untrusted.test/collect"},
        )

    with pytest.raises(AliyunTextRetrievalProviderError) as captured:
        _provider(handler).embed_documents(["object text"])

    assert captured.value.code == "provider_http_error"
    assert captured.value.http_status == 302
    assert captured.value.retryable is False
    assert requests == [f"{API_HOST}{EMBEDDING_PATH}"]


@pytest.mark.parametrize(
    "rows",
    [
        [{"text_index": 0, "embedding": [0.1] * 255}],
        [
            {"text_index": 0, "embedding": [0.1] * 256},
            {"text_index": 0, "embedding": [0.2] * 256},
        ],
        [{"text_index": 0, "embedding": [float("nan")] * 256}],
    ],
)
def test_rejects_malformed_embedding_rows(rows: list[dict[str, object]]) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"request_id": "bad-vector", "output": {"embeddings": rows}},
        )

    texts = ["one", "two"] if len(rows) == 2 else ["one"]
    with pytest.raises(AliyunTextRetrievalProviderError) as captured:
        _provider(handler).embed_documents(texts)

    assert captured.value.code == "provider_response_error"
    assert captured.value.request_id == "bad-vector"


@pytest.mark.parametrize(
    "texts",
    [
        [],
        [""],
        ["item"] * (MAX_EMBEDDING_BATCH_SIZE + 1),
        ["x" * (MAX_EMBEDDING_BATCH_CHARACTERS // 2 + 1)] * 2,
    ],
)
def test_embedding_batch_validation_happens_before_network(texts: list[str]) -> None:
    called = False

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    with pytest.raises(AliyunTextRetrievalProviderError) as captured:
        _provider(handler).embed_documents(texts)

    assert captured.value.code == "invalid_request"
    assert called is False


def test_reranks_with_compatible_api_and_maps_provider_indices() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == RERANK_PATH
        observed.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "object": "list",
                "results": [
                    {"index": 2, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.63},
                ],
                "model": "qwen3-rerank",
                "id": "rerank-request-1",
                "usage": {"total_tokens": 71},
            },
        )

    documents = ["first", "second", "third"]
    result = _provider(handler).rerank("visitor question", documents, top_n=2)

    assert observed == {
        "model": "qwen3-rerank",
        "query": "visitor question",
        "documents": documents,
        "top_n": 2,
        "instruct": DEFAULT_RERANK_INSTRUCT,
    }
    assert [(item.index, item.document, item.relevance_score) for item in result.results] == [
        (2, "third", 0.91),
        (0, "first", 0.63),
    ]
    assert result.results[0].original_index == 2
    assert result.results[0].score == 0.91
    assert result.candidate_count == 3
    assert result.top_n == 2
    assert result.metadata.request_id == "rerank-request-1"
    assert result.metadata.total_tokens == 71


def test_rerank_retries_transient_server_failure() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                503,
                json={"code": "ModelUnavailable", "request_id": "failed-request"},
            )
        return httpx.Response(
            200,
            json={
                "results": [{"index": 0, "relevance_score": 0.8}],
                "id": "successful-request",
            },
        )

    result = _provider(handler).rerank("question", ["document"], top_n=1)

    assert attempts == 2
    assert result.metadata.attempts == 2
    assert result.metadata.request_id == "successful-request"


@pytest.mark.parametrize(
    ("query", "documents", "top_n"),
    [
        ("", ["document"], 1),
        ("question", [], 1),
        ("question", ["document"] * (MAX_RERANK_DOCUMENTS + 1), 1),
        ("q" * (MAX_RERANK_ITEM_CHARACTERS + 1), ["document"], 1),
        ("question", ["x" * (MAX_RERANK_ITEM_CHARACTERS + 1)], 1),
        ("question", ["document"], 2),
        ("question", ["document"], 0),
    ],
)
def test_rerank_validates_limits_before_network(
    query: str, documents: list[str], top_n: int
) -> None:
    called = False

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    with pytest.raises(AliyunTextRetrievalProviderError) as captured:
        _provider(handler).rerank(query, documents, top_n=top_n)

    assert captured.value.code == "invalid_request"
    assert called is False


def test_rerank_rejects_conservative_total_context_before_network() -> None:
    called = False

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    documents = ["x" * 3_000] * 10
    with pytest.raises(AliyunTextRetrievalProviderError) as captured:
        _provider(handler).rerank("question", documents, top_n=2)

    assert captured.value.code == "invalid_request"
    assert called is False


@pytest.mark.parametrize(
    "results",
    [
        [
            {"index": 0, "relevance_score": 0.8},
            {"index": 0, "relevance_score": 0.7},
        ],
        [
            {"index": 0, "relevance_score": 0.2},
            {"index": 1, "relevance_score": 0.9},
        ],
        [
            {"index": 0, "relevance_score": 1.1},
            {"index": 1, "relevance_score": 0.2},
        ],
    ],
)
def test_rejects_invalid_rerank_results(results: list[dict[str, object]]) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"id": "bad-rerank", "results": results},
        )

    with pytest.raises(AliyunTextRetrievalProviderError) as captured:
        _provider(handler).rerank("question", ["one", "two"], top_n=2)

    assert captured.value.code == "provider_response_error"
    assert captured.value.request_id == "bad-rerank"


def test_injected_client_is_reused_and_not_closed_by_provider() -> None:
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json=_embedding_response(dimension=256))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = AliyunTextRetrievalProvider(
            api_key=TEST_KEY,
            api_host=API_HOST,
            embedding_dimension=256,
            retry_delay_seconds=0,
            client=client,
        )
        assert provider.embed_documents(["one"]).vectors[0][0] == 1.0
        assert provider.embed_documents(["two"]).vectors[0][0] == 1.0
        assert client.is_closed is False
    assert requests == 2


def test_provider_configuration_rejects_unsafe_origin_and_ambiguous_client() -> None:
    with pytest.raises(AliyunTextRetrievalProviderError) as insecure:
        AliyunTextRetrievalProvider(
            api_key=TEST_KEY,
            api_host="http://workspace.test",
        )
    assert insecure.value.code == "invalid_config"

    transport = httpx.MockTransport(lambda _: httpx.Response(200))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(AliyunTextRetrievalProviderError) as ambiguous:
            AliyunTextRetrievalProvider(
                api_key=TEST_KEY,
                api_host=API_HOST,
                client=client,
                transport=transport,
            )
    assert ambiguous.value.code == "invalid_config"
