from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from scripts.smoke_full_generation import (
    pending_asset_tasks,
    process_environment,
    stage_budget_overrides,
    wait_for_optional_assets,
)


def test_direct_smoke_does_not_override_operator_proxy(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HTTPS_PROXY", "http://operator-proxy.invalid:1234")
    monkeypatch.setenv("NO_PROXY", "operator.invalid")
    overrides = process_environment(tmp_path, None)
    assert "HTTPS_PROXY" not in overrides
    assert "NO_PROXY" not in overrides
    assert overrides["STORE_PATH"] == str(tmp_path / "store.json")
    assert overrides["RAG_MODE"] == "hybrid"


def test_relay_is_opt_in_and_keeps_institution_requests_direct(tmp_path: Path):
    overrides = process_environment(tmp_path, "http://127.0.0.1:1234")
    assert overrides["HTTPS_PROXY"] == "http://127.0.0.1:1234"
    assert "api.deepseek.com" in overrides["NO_PROXY"]
    assert "openaccess-cdn.clevelandart.org" in overrides["NO_PROXY"]


def test_stage_budgets_follow_shipped_defaults_not_environment(monkeypatch):
    @dataclass
    class FutureSettings:
        deepseek_timeout_seconds: float = 85
        deepseek_frame_timeout_seconds: float = 61
        deepseek_labels_timeout_seconds: float = 47
        rag_retrieval_timeout_seconds: float = 55
        rag_planning_timeout_seconds: float = 32
        deepseek_query_review_thinking: bool = True
        generation_poster_wait_seconds: float = 3
        generation_job_timeout_seconds: float = 181

    monkeypatch.setenv("DEEPSEEK_FRAME_TIMEOUT_SECONDS", "165")
    monkeypatch.setenv("RAG_PLANNING_TIMEOUT_SECONDS", "16")
    monkeypatch.setenv("DEEPSEEK_QUERY_REVIEW_THINKING", "false")
    result = stage_budget_overrides(FutureSettings)
    assert result["DEEPSEEK_FRAME_TIMEOUT_SECONDS"] == "61"
    assert result["GENERATION_JOB_TIMEOUT_SECONDS"] == "181"
    assert result["RAG_RETRIEVAL_TIMEOUT_SECONDS"] == "55"
    assert result["RAG_PLANNING_TIMEOUT_SECONDS"] == "32"
    assert result["DEEPSEEK_QUERY_REVIEW_THINKING"].lower() == "true"


def test_new_planning_defaults_agree_between_dataclass_and_real_from_env(monkeypatch):
    # A smoke override must not hide a release that updates only the dataclass
    # while leaving from_env's unconfigured production fallback at an old RC.
    from dataclasses import fields
    from app.config import Settings
    names = ("rag_planning_timeout_seconds", "deepseek_query_review_thinking")
    for name in names:
        monkeypatch.delenv(name.upper(), raising=False)
    loaded = Settings.from_env()
    defaults = {field.name: field.default for field in fields(Settings)}
    assert all(getattr(loaded, name) == defaults[name] for name in names)


def test_asset_snapshot_counts_pending_not_completed_tasks():
    app = SimpleNamespace(state=SimpleNamespace(
        poster_background_tasks={SimpleNamespaceTask(False), SimpleNamespaceTask(True)},
        audio_prewarm_tasks={SimpleNamespaceTask(False)},
    ))
    assert asyncio.run(pending_asset_tasks(app)) == {
        "poster_background_tasks": 1, "audio_prewarm_tasks": 1,
    }


class SimpleNamespaceTask:
    def __init__(self, completed: bool):
        self.completed = completed

    def done(self):
        return self.completed


def test_zero_asset_wait_reports_pending_without_retry_or_cancel():
    calls = []

    class Portal:
        def call(self, function, app):
            calls.append(function)
            return {"poster_background_tasks": 1, "audio_prewarm_tasks": 1}

    result = wait_for_optional_assets(SimpleNamespace(portal=Portal()), object(), 0)
    assert result["allObservedTasksSettled"] is False
    assert result["pendingAtClientTeardown"]["poster_background_tasks"] == 1
    assert calls == [pending_asset_tasks]


def test_no_pending_assets_returns_without_spending_wait_budget():
    class Portal:
        def call(self, function, app):
            return {"poster_background_tasks": 0, "audio_prewarm_tasks": 0}

    result = wait_for_optional_assets(SimpleNamespace(portal=Portal()), object(), 120)
    assert result["allObservedTasksSettled"] is True
    assert result["elapsedSeconds"] < 1
