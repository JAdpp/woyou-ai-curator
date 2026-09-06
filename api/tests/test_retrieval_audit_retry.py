"""Transport recovery may not reset the audit deadline or bypass its verdict."""
import asyncio

import pytest

from app.generator import ExhibitionGenerator
from app.providers.deepseek import ProviderError


def generator_with_call(monkeypatch, failures, *, elapsed=0.25):
    generator = ExhibitionGenerator.__new__(ExhibitionGenerator)
    clock = [100.0]
    calls = []
    monkeypatch.setattr("app.generator.perf_counter", lambda: clock[0])

    async def call(prompt, payload, *, stage, timeout_seconds):
        calls.append({"prompt": prompt, "payload": payload, "stage": stage, "budget": timeout_seconds})
        clock[0] += elapsed
        if len(calls) <= len(failures):
            raise failures[len(calls) - 1]
        return {"accepted": [], "answerability": "unsupported"}

    monkeypatch.setattr(generator, "_generate_model_json", call)
    return generator, calls


def invoke(generator, budget=10):
    return asyncio.run(generator._generate_retrieval_audit_json(
        "audit instruction", {"question": "fixture", "candidates": []},
        stage="retrieval_audit:1", timeout_seconds=budget,
    ))


def test_one_transport_retry_retains_payload_and_consumes_original_budget(monkeypatch):
    generator, calls = generator_with_call(monkeypatch, [ProviderError("connection", code="provider_network_error")])
    result = invoke(generator)
    assert len(calls) == 2
    assert calls[0]["payload"] == calls[1]["payload"]
    assert calls[0]["prompt"] == calls[1]["prompt"]
    assert [call["budget"] for call in calls] == [10, 9.75]
    assert calls[1]["stage"] == "retrieval_audit:1:transport_retry"
    assert result["answerability"] == "unsupported"  # Never upgrade rejection.


@pytest.mark.parametrize("error", [
    ProviderError("HTTP 402", code="provider_error"),
    ProviderError("HTTP 429", code="provider_error"),
    ProviderError("read timeout", code="provider_timeout"),
    ProviderError("model stage exceeded its wall-clock timeout"),
    ValueError("invalid JSON"),
])
def test_non_transport_failures_are_not_retried(monkeypatch, error):
    generator, calls = generator_with_call(monkeypatch, [error])
    with pytest.raises(type(error)):
        invoke(generator)
    assert len(calls) == 1


def test_second_network_failure_is_propagated(monkeypatch):
    error = ProviderError("connection", code="provider_network_error")
    generator, calls = generator_with_call(monkeypatch, [error, error])
    with pytest.raises(ProviderError):
        invoke(generator)
    assert len(calls) == 2


def test_no_retry_if_original_budget_is_nearly_exhausted(monkeypatch):
    generator, calls = generator_with_call(
        monkeypatch, [ProviderError("connection", code="provider_network_error")], elapsed=9.5,
    )
    with pytest.raises(ProviderError):
        invoke(generator)
    assert len(calls) == 1
