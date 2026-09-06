"""Wall-clock and admission tests for synchronous, non-cancellable retrieval."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.collections import CollectionDataError, CollectionRepository
from app.config import Settings
from app.generator import ExhibitionGenerator, _BoundedRetrievalExecutor
from app.models import AgendaInput


@pytest.fixture()
def one_worker(monkeypatch):
    pool = _BoundedRetrievalExecutor(max_workers=1)
    monkeypatch.setattr("app.generator._RETRIEVAL_EXECUTOR", pool)
    return pool


def _agenda():
    return AgendaInput(question="查找馆藏", collectionId="fixture", priorKnowledge="none", durationMinutes=5)


def _blocking_generator(release, calls):
    class Repository:
        def search(self, agenda, collection):
            calls.append(agenda.question)
            release.wait(2)
            return []

    return ExhibitionGenerator(Settings(rag_retrieval_timeout_seconds=0.04), Repository())


def test_asyncio_run_returns_at_deadline_without_joining_timed_out_worker(one_worker):
    release = threading.Event()
    calls = []
    generator = _blocking_generator(release, calls)
    started = perf_counter()
    try:
        with pytest.raises(CollectionDataError) as raised:
            asyncio.run(generator._search_async(_agenda(), SimpleNamespace()))
        elapsed = perf_counter() - started
        assert raised.value.code == "RETRIEVAL_SEARCH_TIMEOUT"
        assert elapsed < 0.4
        assert calls == ["查找馆藏"]
        assert not release.is_set()  # caller returned while real work remains
    finally:
        release.set()


def test_timed_out_workers_keep_capacity_and_new_requests_do_not_queue(one_worker):
    release = threading.Event()
    calls = []
    generator = _blocking_generator(release, calls)
    try:
        with pytest.raises(CollectionDataError, match="wall-clock"):
            asyncio.run(generator._search_async(_agenda(), SimpleNamespace()))
        started = perf_counter()
        for _ in range(12):
            with pytest.raises(CollectionDataError) as raised:
                asyncio.run(generator._search_async(_agenda(), SimpleNamespace()))
            assert raised.value.code == "RETRIEVAL_CAPACITY_EXHAUSTED"
        assert perf_counter() - started < 0.4
        assert len(calls) == 1
    finally:
        release.set()


def test_retrieval_worker_propagates_trace_context_and_is_daemon(one_worker):
    marker = contextvars.ContextVar("retrieval-test-marker", default="missing")
    marker.set("frozen-run")
    observed = asyncio.run(ExhibitionGenerator._run_retrieval_work(
        lambda: (marker.get(), threading.current_thread().daemon),
        timeout_seconds=1,
    ))
    assert observed == ("frozen-run", True)


def test_expired_budget_never_starts_background_work(one_worker):
    calls = []
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(ExhibitionGenerator._run_retrieval_work(
            lambda: calls.append("started"), timeout_seconds=0,
        ))
    assert calls == []


def test_fused_rerank_cannot_block_async_deadline(one_worker):
    release = threading.Event()

    class Repository(CollectionRepository):
        def __init__(self):
            pass

        def rerank_results(self, question, results, *, deadline=None):
            release.wait(2)
            return list(reversed(results))

    generator = ExhibitionGenerator(Settings(), Repository())
    original = [object(), object()]
    started = perf_counter()
    try:
        result = asyncio.run(generator._rerank_async(
            "比较器物", original, deadline=started + 0.04,
        ))
        assert perf_counter() - started < 0.4
        assert result is original  # optional timeout preserves prior ranking
    finally:
        release.set()


def test_generate_sync_http_and_portal_teardown_obey_retrieval_deadline(
    client, agenda_payload, monkeypatch, one_worker,
):
    release = threading.Event()
    calls = []
    generator = client.app.state.generator
    generator.settings = replace(
        generator.settings, rag_retrieval_timeout_seconds=0.05,
        rag_llm_audit_enabled=False,
    )

    def blocking_search(agenda, collection, **kwargs):
        calls.append(agenda.question)
        release.wait(2)
        return []

    monkeypatch.setattr(generator.collections, "search", blocking_search)
    # Without a persistent context, TestClient also tears down its event loop
    # before post() returns: default-executor work previously held this open.
    transient_client = TestClient(client.app)
    started = perf_counter()
    try:
        response = transient_client.post("/api/exhibitions/generate-sync", json={"agenda": agenda_payload})
        assert perf_counter() - started < 0.6
        assert response.json()["error"]["code"] == "RETRIEVAL_SEARCH_TIMEOUT"
        assert len(calls) == 1
    finally:
        release.set()
        transient_client.close()
