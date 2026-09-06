from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.generator import ExhibitionGenerator
from app.models import AgendaInput
from app.providers.deepseek import DeepSeekProvider
from app.query_plan_review import V4_SEARCH_KEYS, query_plan_document
from app.retrieval_agent import RetrievalQueryPlan


def test_default_budget_and_thinking_remain_unchanged(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    for settings in (Settings(), Settings.from_env()):
        assert settings.rag_planning_timeout_seconds == 16.0
        assert settings.deepseek_query_review_thinking is False
        assert settings.rag_retrieval_timeout_seconds == 70.0
        assert settings.generation_job_timeout_seconds == 180.0


@pytest.mark.parametrize("budget", [2, 16, 24, 32, 70])
def test_env_accepts_bounded_planning_subbudget_without_expanding_generation(budget, monkeypatch):
    monkeypatch.setattr(os, "environ", {"RAG_PLANNING_TIMEOUT_SECONDS": str(budget), "DEEPSEEK_QUERY_REVIEW_THINKING": "true"})
    settings = Settings.from_env()
    assert settings.rag_planning_timeout_seconds == budget
    assert settings.deepseek_query_review_thinking is True
    assert settings.rag_retrieval_timeout_seconds == 70.0
    assert settings.generation_job_timeout_seconds == 180.0


@pytest.mark.parametrize("value", ["0", "1.9", "70.1", "nan", "inf", "not-a-number"])
def test_invalid_env_planning_budget_is_rejected(value, monkeypatch):
    monkeypatch.setattr(os, "environ", {"RAG_PLANNING_TIMEOUT_SECONDS": value})
    with pytest.raises(ValueError, match="RAG_PLANNING_TIMEOUT_SECONDS"):
        Settings.from_env()


def test_invalid_thinking_flag_is_rejected(monkeypatch):
    monkeypatch.setattr(os, "environ", {"DEEPSEEK_QUERY_REVIEW_THINKING": "sometimes"})
    with pytest.raises(ValueError, match="DEEPSEEK_QUERY_REVIEW_THINKING"):
        Settings.from_env()


@pytest.mark.parametrize("enabled,payload,expected_tokens,expected_thinking,effort", [
    (False, {"visitorQuestion": "fixture"}, 1200, "disabled", None),
    (False, {"recallDraft": {}}, 2400, "disabled", None),
    (True, {"visitorQuestion": "fixture"}, 1200, "disabled", None),
    (True, {"draftPlan": {}}, 2400, "disabled", None),
    (True, {"recallDraft": {}}, 4096, "enabled", "low"),
])
def test_thinking_flag_only_changes_actual_v4_review(enabled, payload, expected_tokens, expected_thinking, effort, monkeypatch):
    provider = DeepSeekProvider(Settings(deepseek_query_review_thinking=enabled))
    calls = []

    async def capture(**kwargs):
        calls.append(kwargs)
        return {"fixture": True}

    monkeypatch.setattr(provider, "_generate_json", capture)
    assert asyncio.run(provider.generate_retrieval_query_plan_json("prompt", payload)) == {"fixture": True}
    assert len(calls) == 1
    assert calls[0]["thinking"] == {"type": expected_thinking}
    assert calls[0]["max_tokens"] == expected_tokens
    assert calls[0].get("reasoning_effort") == effort
    if effort is None:
        assert "reasoning_effort" not in calls[0]
    assert calls[0]["model"] == Settings().deepseek_model
    assert calls[0]["temperature"] == 0.0


def test_thinking_flag_does_not_change_general_generation_or_evidence_audit(monkeypatch):
    provider = DeepSeekProvider(Settings(deepseek_query_review_thinking=True))
    calls = []

    async def capture(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(provider, "_generate_json", capture)
    asyncio.run(provider.generate_json("prompt", {}))
    asyncio.run(provider.generate_retrieval_audit_json("prompt", {}))
    assert len(calls) == 2
    assert all(call["thinking"] == {"type": "disabled"} and "reasoning_effort" not in call for call in calls)


@pytest.mark.parametrize("configured_budget", [16.0, 24.0, 32.0, 70.0])
@pytest.mark.parametrize("short_deadline", [False, True])
def test_planning_setting_is_only_a_subbudget_with_original_audit_reserve(configured_budget, short_deadline, monkeypatch):
    clock = [100.0]
    calls = []
    monkeypatch.setattr("app.generator.perf_counter", lambda: clock[0])
    settings = Settings(rag_planning_timeout_seconds=configured_budget)
    audit_floor = min(settings.rag_llm_audit_timeout_seconds,
                      max(10.0, settings.rag_llm_audit_timeout_seconds * 0.75))
    deadline = 100.0 + (audit_floor + 6.0 if short_deadline else 70.0)
    effective = min(configured_budget, deadline - 100.0 - audit_floor)
    provider = SimpleNamespace(configured=True, supports_retrieval_audit=True, supports_retrieval_query_planning=True)
    generator = ExhibitionGenerator(settings, SimpleNamespace(), provider)
    draft = RetrievalQueryPlan(valid=True, in_collection_scope=True, search_queries=("geometric clothing",),
                               semantic_query="clothing with geometric motifs", mandatory_predicates=("a geometric motif on clothing",),
                               evidence_mode="visual_observation", visual_predicate_ids=("p1",))

    async def generate(prompt, payload, *, stage, timeout_seconds):
        calls.append({"stage": stage, "budget": timeout_seconds, "at": clock[0]})
        if stage == "retrieval_plan:1":
            clock[0] += 4.0
            return query_plan_document(draft)
        assert stage == "retrieval_plan_review"
        clock[0] += timeout_seconds
        return {"searchPlan": {key: value for key, value in query_plan_document(draft).items() if key in V4_SEARCH_KEYS},
                "intents": [{"id": "i1", "sourceQuote": "衣服上的几何图案", "type": "admission",
                             "text": "a geometric motif on clothing", "evidenceScope": "visible_features_or_record"}]}

    async def search(*args, deadline, **kwargs):
        calls.append({"stage": "search", "deadline": deadline, "at": clock[0]})
        return []

    monkeypatch.setattr(generator, "_generate_model_json", generate)
    monkeypatch.setattr(generator, "_search_async", search)
    result = asyncio.run(generator.prepare_initial_retrieval(
        AgendaInput(question="我想看衣服上的几何图案", priorKnowledge="some", durationMinutes=5),
        SimpleNamespace(concept_aliases={}), deadline=deadline))
    assert result.diagnostics["planningReview"]["status"] == "confirmed"
    assert [call["stage"] for call in calls] == ["retrieval_plan:1", "retrieval_plan_review", "search"]
    assert calls[0]["budget"] == min(8.0, effective)
    assert calls[1]["budget"] == effective - 4.0
    assert calls[2]["at"] == 100.0 + effective
    assert calls[2]["deadline"] == deadline <= 170.0
    assert calls[2]["deadline"] - calls[2]["at"] >= audit_floor
    assert settings.generation_job_timeout_seconds == 180.0


def test_probe_budget_and_thinking_flags_have_a_no_cost_inventory_path(tmp_path):
    root = Path(__file__).resolve().parents[2]
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"cases": [{"id": "fixture", "question": "示例问题"}]}), encoding="utf-8")
    output = tmp_path / "not-created"
    command = [sys.executable, str(root / "scripts/probe_query_planning.py"), "--questions", str(questions),
               "--case", "fixture", "--output", str(output), "--review-thinking", "--planning-budget-seconds", "32"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    inventory = json.loads(result.stdout)
    assert inventory["paidCalls"] == 0 and inventory["execute"] is False
    assert inventory["reviewThinking"] is True and inventory["planningBudgetSeconds"] == 32.0
    assert not output.exists()
    invalid = subprocess.run(command[:-1] + ["71"], capture_output=True, text=True)
    assert invalid.returncode != 0 and "Planning budget must be between 2 and 70" in invalid.stderr
