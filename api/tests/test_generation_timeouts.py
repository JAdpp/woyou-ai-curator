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

from app.config import Settings
from app.generator import ExhibitionGenerator
from app.jobs import JobStore
from app.main import (
    _INTERRUPTED_GENERATION_NOTE,
    _mark_interrupted_generation,
    create_app,
)
from app.models import ExhibitionPoster, VisitorProfile
from app.providers.aliyun_image import GeneratedPoster, PosterContext
from app.providers.deepseek import DeepSeekProvider, ProviderError
from app.store import ExhibitionStore


class _SlowModelProvider:
    configured = True

    async def generate_json(self, _system_prompt: str, _payload: dict) -> dict:
        await asyncio.sleep(0.25)
        return {}


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
    assert "问题仍然保留" in (job.error or "")
    assert "certificate" not in (job.error or "")


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
