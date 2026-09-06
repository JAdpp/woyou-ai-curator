from __future__ import annotations

import asyncio
import ssl
import threading
import time
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import pytest
from fastapi.testclient import TestClient

from .retrieval_contract_fixtures import strict_audit_fixture

from app.collections import CollectionDataError, SearchResult
from app.config import Settings
from app.generator import ExhibitionGenerator
from app.jobs import JobStore
from app.main import (
    _INTERRUPTED_GENERATION_NOTE,
    _mark_interrupted_generation,
    create_app,
)
from app.models import (
    AgendaInput,
    EvidenceChunk,
    ExhibitionPoster,
    MuseumObject,
    VisitorProfile,
)
from app.providers.aliyun_image import GeneratedPoster, PosterContext
from app.providers.deepseek import DeepSeekProvider, ProviderError
from app.store import ExhibitionStore


class _SlowModelProvider:
    configured = True

    async def generate_json(self, _system_prompt: str, _payload: dict) -> dict:
        await asyncio.sleep(0.25)
        return {}


def _timeout_candidate(object_id: str) -> SearchResult:
    evidence_id = f"{object_id}:metadata"
    obj = MuseumObject(
        id=object_id,
        sourceId=object_id,
        title=f"Object {object_id}",
        description="Institution description.",
        imageUrl=f"https://example.test/{object_id}.jpg",
        objectUrl=f"https://example.test/{object_id}",
        rights="CC0",
        institution="Fixture Museum",
        institutionId="fixture",
        evidence=[
            EvidenceChunk(
                id=evidence_id,
                text=f"Institution record identifies object {object_id}.",
                sourceUrl=f"https://example.test/{object_id}",
                sourceTitle=f"Object {object_id}",
                sourceKind="institution_metadata",
            )
        ],
    )
    return SearchResult(
        obj=obj,
        score=80.0,
        matched_evidence_ids=(evidence_id,),
        retrieval_sources=("dense_object", "evidence_rerank"),
        dense_score=0.72,
        evidence_score=0.70,
    )


class _FirstAuditNeedsExpansionProvider:
    configured = True
    supports_retrieval_audit = True

    @strict_audit_fixture
    async def generate_json(self, _prompt: str, payload: dict) -> dict:
        return {
            "queryInterpretation": "保留的首轮解释",
            "answerability": "supported",
            "accepted": [
                {
                    "objectId": candidate["objectId"],
                    "relevanceScore": 0.9,
                    "evidenceIds": [candidate["evidence"][0]["id"]],
                }
                for candidate in payload["candidates"]
            ],
            "expansionReason": "insufficient_direct_objects",
            "searchQueries": ["bounded optional expansion"],
            "coverageGap": "首轮只有两件直接证据。",
        }


class _PartialCaseStudyProvider:
    configured = True
    supports_retrieval_audit = True

    def __init__(self, answerability: str = "partially_supported") -> None:
        self.answerability = answerability

    @strict_audit_fixture
    async def generate_json(self, _prompt: str, payload: dict) -> dict:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            # The later curation call may fail over to the deterministic
            # template; this fixture isolates the answerability gate.
            return {}
        return {
            "queryInterpretation": "馆藏可支持具体案例，但不能推出普遍结论",
            "answerability": self.answerability,
            "accepted": [
                {
                    "objectId": candidate["objectId"],
                    "relevanceScore": 0.9,
                    "evidenceIds": [candidate["evidence"][0]["id"]],
                }
                for candidate in candidates
            ],
            "expansionReason": "none",
            "searchQueries": [],
            "coverageGap": "馆藏只支持这些对象案例，不支持普遍因果结论。",
        }


class _UsablePartialAuditProvider:
    configured = True
    supports_retrieval_audit = True

    def __init__(self) -> None:
        self.calls = 0

    @strict_audit_fixture
    async def generate_json(self, _prompt: str, payload: dict) -> dict:
        self.calls += 1
        return {
            "queryInterpretation": "足量对象案例",
            "answerability": "partially_supported",
            "accepted": [
                {
                    "objectId": candidate["objectId"],
                    "relevanceScore": 0.9,
                    "evidenceIds": [candidate["evidence"][0]["id"]],
                }
                for candidate in payload["candidates"]
            ],
            "expansionReason": (
                "predicate_evidence_gap" if self.calls == 1 else "none"
            ),
            "searchQueries": (
                ["direct predicate evidence"] if self.calls == 1 else []
            ),
            "coverageGap": "对象案例不能推出普遍方法。",
        }


class _ExpansionMustNotStartCollections:
    def __init__(self) -> None:
        self.search_many_calls = 0

    def match_question_policy(self, _collection, _question):
        return None

    def search_many(self, _agendas, _collection):
        self.search_many_calls += 1
        raise AssertionError("optional expansion had no remaining budget")


class _DelayedPosterProvider:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    async def generate_poster(self, context: PosterContext) -> GeneratedPoster:
        await asyncio.sleep(0.03)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.output_dir / "timestamp-fixture.png"
        metadata_path = self.output_dir / "timestamp-fixture.json"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        metadata_path.write_text("{}\n", encoding="utf-8")
        return GeneratedPoster(
            provider="aliyun-model-studio",
            model="qwen-image-3.0-pro",
            size="1536*864",
            image_path=image_path,
            metadata_path=metadata_path,
            sha256="timestamp-fixture",
            seed=context.seed,
            generated_at="2026-08-10T00:00:00Z",
            request_id="timestamp-request",
        )


class _BlockingLobbyAudio:
    """A deliberately unresolved audio task used to prove job independence."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[tuple[str, str]] = []

    async def get_audio(self, exhibition, kind: str, reference=None) -> None:
        self.calls.append((exhibition.id, kind))
        assert kind == "lobby"
        assert reference is None
        self.started.set()
        await asyncio.to_thread(self.release.wait, 2)


class _FailingLobbyAudio:
    def __init__(self) -> None:
        self.called = threading.Event()

    async def get_audio(self, _exhibition, kind: str, _reference=None) -> None:
        assert kind == "lobby"
        self.called.set()
        raise RuntimeError("TTS transport detail must not affect curation")


def test_frame_deadline_falls_back_before_the_job_safety_timeout(client) -> None:
    settings = replace(
        client.app.state.settings,
        deepseek_timeout_seconds=1.0,
        deepseek_frame_timeout_seconds=0.02,
        deepseek_labels_timeout_seconds=0.02,
    )
    generator = ExhibitionGenerator(
        settings,
        client.app.state.collections,
        provider=_SlowModelProvider(),
    )

    started = perf_counter()
    exhibition = asyncio.run(
        generator.generate_from_profile(VisitorProfile(durationMinutes=5))
    )

    assert perf_counter() - started < 0.20
    assert exhibition.status == "ready"
    assert exhibition.versions.provider == "deterministic_fallback"
    assert any("确定性模板" in limit for limit in exhibition.coverage_limits)


def test_optional_expansion_deadline_preserves_the_first_audit_verdict() -> None:
    candidates = [_timeout_candidate("a"), _timeout_candidate("b")]
    collections = _ExpansionMustNotStartCollections()
    generator = ExhibitionGenerator(
        Settings(
            rag_retrieval_timeout_seconds=1.25,
            rag_llm_audit_timeout_seconds=1.25,
        ),
        collections,  # type: ignore[arg-type]
        provider=_FirstAuditNeedsExpansionProvider(),  # type: ignore[arg-type]
    )
    agenda = AgendaInput(
        question="需要补证但首审已经有直接证据的问题",
        priorKnowledge="none",
        durationMinutes=5,
        collectionId="fixture",
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            agenda,
            object(),  # type: ignore[arg-type]
            candidates,
            required_count=5,
        )
    )

    assert collections.search_many_calls == 0
    assert [result.obj.id for result in outcome.results] == ["a", "b"]
    assert outcome.answerability == "partially_supported"
    assert outcome.interpretation == "保留的首轮解释"
    assert "时限" in outcome.coverage_gap
    assert outcome.failure_code == "RETRIEVAL_SEARCH_TIMEOUT"
    assert outcome.warning_code == "RETRIEVAL_SEARCH_TIMEOUT"
    assert outcome.warning_detail


def test_usable_partial_does_not_spend_time_on_optional_refinement() -> None:
    candidates = [_timeout_candidate(str(index)) for index in range(5)]
    collections = _ExpansionMustNotStartCollections()
    provider = _UsablePartialAuditProvider()
    generator = ExhibitionGenerator(
        Settings(),
        collections,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )
    agenda = AgendaInput(
        question="馆藏案例可以支持但不能推出普遍结论的问题",
        priorKnowledge="none",
        durationMinutes=5,
        collectionId="fixture",
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            agenda,
            object(),  # type: ignore[arg-type]
            candidates,
            required_count=5,
        )
    )

    assert collections.search_many_calls == 0
    assert provider.calls == 1
    assert len(outcome.results) == 5
    assert outcome.answerability == "partially_supported"
    assert outcome.warning_code is None


def test_partial_answer_with_five_audited_objects_generates_narrowed_exhibition(
    client,
) -> None:
    generator = ExhibitionGenerator(
        client.app.state.settings,
        client.app.state.collections,
        provider=_PartialCaseStudyProvider(),  # type: ignore[arg-type]
    )
    collection = client.app.state.collections.get()
    agenda = AgendaInput(
        question="landscape object institution description",
        priorKnowledge="none",
        durationMinutes=5,
        collectionId=collection.id,
    )

    exhibition = asyncio.run(generator.generate(agenda))

    assert len(exhibition.items) == 5
    assert exhibition.validation.passed is True
    assert "馆藏只支持这些对象案例，不支持普遍因果结论。" in exhibition.coverage_limits


def test_profile_flow_also_generates_a_narrowed_partial_exhibition(client) -> None:
    generator = ExhibitionGenerator(
        client.app.state.settings,
        client.app.state.collections,
        provider=_PartialCaseStudyProvider(),  # type: ignore[arg-type]
    )
    collection = client.app.state.collections.get()
    profile = VisitorProfile(
        curiosityLabel="对象案例",
        freeFormQuestion="landscape object institution description",
        durationMinutes=5,
    )

    exhibition = asyncio.run(
        generator.generate_from_profile(profile, collection_id=collection.id)
    )

    assert len(exhibition.items) == 5
    assert exhibition.validation.passed is True
    assert any("不声称涵盖" in value for value in exhibition.coverage_limits)
    assert any("历史结论" in value for value in exhibition.coverage_limits)


def test_unsupported_answer_still_fails_even_with_five_topical_objects(client) -> None:
    generator = ExhibitionGenerator(
        client.app.state.settings,
        client.app.state.collections,
        provider=_PartialCaseStudyProvider("unsupported"),  # type: ignore[arg-type]
    )
    collection = client.app.state.collections.get()
    agenda = AgendaInput(
        question="landscape object institution description",
        priorKnowledge="none",
        durationMinutes=5,
        collectionId=collection.id,
    )

    with pytest.raises(CollectionDataError) as raised:
        asyncio.run(generator.generate(agenda))

    assert raised.value.code == "QUESTION_UNSUPPORTED_AFTER_AUDIT"


def test_job_deadline_message_does_not_blame_the_visitors_topic() -> None:
    async def scenario():
        jobs = JobStore(max_job_seconds=0.02)
        job = jobs.create()

        async def slow_work(_jobs: JobStore, _job_id: str) -> None:
            _jobs.start_step(_job_id, "profile")
            await asyncio.sleep(0.25)

        jobs.run(job.id, slow_work)
        for _ in range(20):
            await asyncio.sleep(0.01)
            if jobs.get(job.id).status == "failed":
                break
        return jobs.get(job.id)

    job = asyncio.run(scenario())
    assert job.status == "failed"
    assert job.error_code == "curation_timeout"
    assert "主题不受支持" in (job.error or "")
    assert "收窄主题" not in (job.error or "")


def test_unexpected_job_error_is_sanitised_and_has_a_stable_code() -> None:
    async def scenario():
        jobs = JobStore(max_job_seconds=1)
        job = jobs.create()

        async def broken_work(_jobs: JobStore, _job_id: str) -> None:
            raise RuntimeError("secret certificate path and proxy internals")

        jobs.run(job.id, broken_work)
        for _ in range(20):
            await asyncio.sleep(0.01)
            if jobs.get(job.id).status == "failed":
                break
        return jobs.get(job.id)

    job = asyncio.run(scenario())
    assert job.status == "failed"
    assert job.error_code == "curation_failed"


def test_post_audit_unsupported_job_is_actionable_not_reported_as_outage() -> None:
    async def scenario():
        jobs = JobStore(max_job_seconds=1)
        job = jobs.create()

        async def unsupported_work(_jobs: JobStore, _job_id: str) -> None:
            _jobs.start_step(_job_id, "retrieve")
            raise CollectionDataError(
                "QUESTION_UNSUPPORTED_AFTER_AUDIT",
                "internal retrieval detail",
                coverageGap="missing comparison leg",
            )

        jobs.run(job.id, unsupported_work)
        for _ in range(20):
            await asyncio.sleep(0.01)
            if jobs.get(job.id).status == "failed":
                break
        return jobs.get(job.id)

    job = asyncio.run(scenario())
    assert job.status == "failed"
    assert job.error_code == "question_unsupported_after_audit"
    assert "missing comparison leg" in (job.error or "")
    assert "调整范围" in (job.error or "")
    assert "稍后重试" not in (job.error or "")
    assert "问题仍然保留" in (job.error or "")
    assert "certificate" not in (job.error or "")


@pytest.mark.parametrize(
    "error_code",
    [
        "RETRIEVAL_AUDIT_UNAVAILABLE",
        "RETRIEVAL_AUDIT_INVALID",
        "RETRIEVAL_SEARCH_TIMEOUT",
    ],
)
def test_retrieval_infrastructure_failure_does_not_blame_topic(
    error_code: str,
) -> None:
    async def scenario():
        jobs = JobStore(max_job_seconds=1)
        job = jobs.create()

        async def failed_retrieval(_jobs: JobStore, _job_id: str) -> None:
            _jobs.start_step(_job_id, "retrieve")
            raise CollectionDataError(error_code, "internal retrieval detail")

        jobs.run(job.id, failed_retrieval)
        for _ in range(20):
            await asyncio.sleep(0.01)
            if jobs.get(job.id).status == "failed":
                break
        return jobs.get(job.id)

    job = asyncio.run(scenario())
    assert job.status == "failed"
    assert job.error_code == error_code.casefold()
    assert "不代表馆藏不支持" in (job.error or "")
    assert "直接重试" in (job.error or "")
    assert "相近方向" not in (job.error or "")


def test_interrupted_persisted_skeleton_becomes_an_honest_draft(client) -> None:
    generator = ExhibitionGenerator(
        client.app.state.settings, client.app.state.collections
    )
    exhibition = asyncio.run(
        generator.generate_from_profile(VisitorProfile(durationMinutes=5))
    )
    exhibition.status = "generating"
    exhibition.poster = ExhibitionPoster(
        status="generating",
        provider="aliyun-model-studio",
        model="qwen-image-3.0-pro",
        generated_by="Alibaba Cloud Model Studio · qwen-image-3.0-pro",
        size="1536*864",
    )
    client.app.state.store.save_exhibition(exhibition)

    recovered = _mark_interrupted_generation(
        client.app.state.store, exhibition.id
    )

    assert recovered is not None
    assert recovered.status == "draft"
    assert _INTERRUPTED_GENERATION_NOTE in recovered.coverage_limits
    assert recovered.poster is not None
    assert recovered.poster.status == "failed"
    assert recovered.poster.error_code == "generation_interrupted"


def test_orphan_recovery_runs_on_startup_not_module_or_app_construction(
    client, tmp_path
) -> None:
    generator = ExhibitionGenerator(
        client.app.state.settings, client.app.state.collections
    )
    exhibition = asyncio.run(
        generator.generate_from_profile(VisitorProfile(durationMinutes=5))
    )
    exhibition.status = "generating"
    path = tmp_path / "store.json"
    ExhibitionStore(mode="json", path=path).save_exhibition(exhibition)
    settings = replace(
        client.app.state.settings,
        store_mode="json",
        store_path=path,
    )

    app = create_app(settings)
    assert app.state.store.get_exhibition(exhibition.id).status == "generating"

    with TestClient(app):
        assert app.state.store.get_exhibition(exhibition.id).status == "draft"


def test_final_poster_merge_never_moves_exhibition_updated_at_backwards(
    client, tmp_path
) -> None:
    output_dir = tmp_path / "posters"
    settings = replace(
        client.app.state.settings,
        aliyun_image_api_key="test-key",
        aliyun_image_api_host="https://example.invalid",
        aliyun_image_output_dir=output_dir,
        generation_poster_wait_seconds=1.0,
    )
    app = create_app(settings)
    app.state.image_provider = _DelayedPosterProvider(output_dir)

    with TestClient(app) as generated_client:
        created = generated_client.post(
            "/api/exhibitions/generate",
            json={"profile": {"durationMinutes": 5}},
        )
        assert created.status_code == 202, created.text
        job_id = created.json()["id"]
        job = None
        for _ in range(100):
            job = generated_client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)

        assert job is not None and job["status"] == "completed", job
        exhibition = generated_client.get(
            f"/api/exhibitions/{job['exhibitionId']}"
        ).json()

    assert exhibition["poster"]["status"] == "ready"
    assert exhibition["updatedAt"] >= exhibition["poster"]["generatedAt"]


def test_lobby_tts_prewarm_starts_after_completion_without_blocking_the_job(client) -> None:
    service = _BlockingLobbyAudio()
    client.app.state.audio_guide_service = service

    created = client.post(
        "/api/exhibitions/generate",
        json={"profile": {"durationMinutes": 5}},
    )
    assert created.status_code == 202, created.text
    job_id = created.json()["id"]

    job = None
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)

    assert job is not None and job["status"] == "completed", job
    # The background task is now intentionally blocked, but the exhibition is
    # already public.  A visitor never waits for this TTS request on the
    # curation progress screen.
    assert service.started.wait(timeout=1)
    assert service.calls == [(job["exhibitionId"], "lobby")]
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "completed"
    assert client.get(f"/api/exhibitions/{job['exhibitionId']}").status_code == 200

    service.release.set()
    for _ in range(100):
        if not client.app.state.audio_prewarm_tasks:
            break
        time.sleep(0.01)
    assert not client.app.state.audio_prewarm_tasks


def test_lobby_tts_prewarm_failure_keeps_the_exhibition_completed(client) -> None:
    service = _FailingLobbyAudio()
    client.app.state.audio_guide_service = service

    created = client.post(
        "/api/exhibitions/generate",
        json={"profile": {"durationMinutes": 5}},
    )
    assert created.status_code == 202, created.text
    job_id = created.json()["id"]

    job = None
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)

    assert job is not None and job["status"] == "completed", job
    assert service.called.wait(timeout=1)
    exhibition = client.get(f"/api/exhibitions/{job['exhibitionId']}").json()
    assert exhibition["status"] == "ready"
    assert client.get(f"/api/jobs/{job_id}").json()["error"] is None


def test_deepseek_http_timeout_is_a_total_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SlowClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, *_args, **_kwargs):
            await asyncio.sleep(0.25)
            raise AssertionError("the hard deadline should cancel this request")

    monkeypatch.setattr("app.providers.deepseek.httpx.AsyncClient", SlowClient)
    provider = DeepSeekProvider(
        Settings(deepseek_api_key="test", deepseek_timeout_seconds=0.02)
    )

    started = perf_counter()
    with pytest.raises(ProviderError, match="wall-clock timeout"):
        asyncio.run(provider.generate_json("system", {"question": "test"}))
    assert perf_counter() - started < 0.20


@pytest.mark.parametrize(
    "transport_error",
    [ssl.SSLError("raw TLS certificate details"), OSError("raw socket details")],
)
def test_deepseek_wraps_raw_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
    transport_error: BaseException,
) -> None:
    class BrokenClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, *_args, **_kwargs):
            raise transport_error

    monkeypatch.setattr("app.providers.deepseek.httpx.AsyncClient", BrokenClient)
    provider = DeepSeekProvider(Settings(deepseek_api_key="test"))

    with pytest.raises(ProviderError) as caught:
        asyncio.run(provider.generate_json("system", {"question": "test"}))

    assert caught.value.code == "provider_network_error"
    assert "raw " not in str(caught.value)
    assert "稍后重试" in caught.value.public_message


def test_deepseek_preserves_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    class CancelledClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, *_args, **_kwargs):
            raise asyncio.CancelledError

    monkeypatch.setattr("app.providers.deepseek.httpx.AsyncClient", CancelledClient)
    provider = DeepSeekProvider(Settings(deepseek_api_key="test"))

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(provider.generate_json("system", {"question": "test"}))


def test_raw_tls_failure_reaches_the_deterministic_generator_fallback(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, *_args, **_kwargs):
            raise ssl.SSLError("certificate verify failed: private proxy path")

    monkeypatch.setattr("app.providers.deepseek.httpx.AsyncClient", BrokenClient)
    settings = replace(
        client.app.state.settings,
        deepseek_api_key="test",
        deepseek_timeout_seconds=1,
        deepseek_frame_timeout_seconds=1,
        deepseek_labels_timeout_seconds=1,
    )
    generator = ExhibitionGenerator(
        settings,
        client.app.state.collections,
        provider=DeepSeekProvider(settings),
    )

    exhibition = asyncio.run(
        generator.generate_from_profile(VisitorProfile(durationMinutes=5))
    )

    assert exhibition.status == "ready"
    assert exhibition.versions.provider == "deterministic_fallback"
    assert any("确定性模板" in limit for limit in exhibition.coverage_limits)


def test_tls_failure_completes_the_public_job_without_leaking_ssl_details(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, *_args, **_kwargs):
            raise ssl.SSLError("private certificate and proxy details")

    monkeypatch.setattr("app.providers.deepseek.httpx.AsyncClient", BrokenClient)
    monkeypatch.setattr(
        client.app.state.generator.provider,
        "api_key",
        "test",
    )
    created = client.post(
        "/api/exhibitions/generate",
        json={"profile": {"durationMinutes": 5}},
    )
    assert created.status_code == 202, created.text
    job_id = created.json()["id"]

    job = None
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)

    assert job is not None and job["status"] == "completed", job
    assert job["errorCode"] is None
    assert job["error"] is None
    exhibition = client.get(
        f"/api/exhibitions/{job['exhibitionId']}"
    ).json()
    assert exhibition["versions"]["provider"] == "deterministic_fallback"
