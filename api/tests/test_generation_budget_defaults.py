from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

from app.config import Settings


ROOT = Path(__file__).resolve().parents[2]


def _reserved_budget(settings: Settings) -> float:
    return (
        min(settings.deepseek_timeout_seconds, settings.deepseek_frame_timeout_seconds)
        + min(settings.deepseek_timeout_seconds, settings.deepseek_labels_timeout_seconds)
        + settings.rag_retrieval_timeout_seconds
        + settings.generation_poster_wait_seconds
        + 10
    )


def test_clean_environment_defaults_start_with_consistent_generation_budget(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    settings = Settings.from_env()
    assert settings.deepseek_frame_timeout_seconds == Settings().deepseek_frame_timeout_seconds == 55
    assert settings.generation_job_timeout_seconds == 180
    assert _reserved_budget(settings) == 177
    assert _reserved_budget(settings) < settings.generation_job_timeout_seconds


@pytest.mark.parametrize("relative_path", [".env.example", "api/.env.example"])
def test_env_examples_have_consistent_budget_after_credentials_are_supplied(monkeypatch, relative_path):
    values = {
        key: str(value) for key, value in dotenv_values(ROOT / relative_path).items()
        if value is not None
    }
    # Settings construction only: these are intentionally not real credentials,
    # and this test must never make a provider/network call.
    values.update(
        DASHSCOPE_API_KEY="configuration-only-placeholder",
        ALIYUN_TEXT_API_HOST="https://llm-nwypztqdwtzyt9zd.cn-beijing.maas.aliyuncs.com",
    )
    monkeypatch.setattr(os, "environ", values)
    settings = Settings.from_env()
    assert settings.deepseek_frame_timeout_seconds == 55
    assert settings.generation_job_timeout_seconds == 180
    assert _reserved_budget(settings) < settings.generation_job_timeout_seconds


@pytest.mark.parametrize("overrides", [
    {"GENERATION_JOB_TIMEOUT_SECONDS": "177", "RAG_RETRIEVAL_TIMEOUT_SECONDS": "70"},
    {"GENERATION_JOB_TIMEOUT_SECONDS": "180", "DEEPSEEK_FRAME_TIMEOUT_SECONDS": "90", "RAG_RETRIEVAL_TIMEOUT_SECONDS": "70"},
])
def test_explicit_over_budget_configuration_still_fails(monkeypatch, overrides):
    monkeypatch.setattr(os, "environ", overrides)
    with pytest.raises(ValueError, match="Generation stage timeouts"):
        Settings.from_env()


def test_new_default_does_not_break_existing_custom_generation_budget(monkeypatch):
    monkeypatch.setattr(os, "environ", {
        "GENERATION_JOB_TIMEOUT_SECONDS": "400", "GENERATION_POSTER_WAIT_SECONDS": "20",
        "DEEPSEEK_TIMEOUT_SECONDS": "240", "DEEPSEEK_FRAME_TIMEOUT_SECONDS": "165",
        "DEEPSEEK_LABELS_TIMEOUT_SECONDS": "150",
    })
    settings = Settings.from_env()
    assert settings.rag_retrieval_timeout_seconds == 54
    assert _reserved_budget(settings) < 400
