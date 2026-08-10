from __future__ import annotations

import asyncio
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
    assert "主题不受支持" in (job.error or "")
    assert "收窄主题" not in (job.error or "")


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
