from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import ai_qrel_review as review  # noqa: E402


def _candidate(object_id: str, evidence_id: str, judgment: dict[str, Any] | None = None, image_url: str | None = None) -> dict[str, Any]:
    return {
        "objectId": object_id,
        "object": {
            "id": object_id,
            "title": f"Object {object_id}",
            "date": "1900",
            "maker": "Maker",
            "description": "Institution catalogue prose.",
            "institution": "Test Museum",
            # A malicious/legacy API field must not become a model label.
            "candidateScore": 0.99,
            "imageUrl": image_url,
        },
        "culturalLegs": ["test"],
        "evidence": [{"id": evidence_id, "text": "Institution evidence.", "sourceUrl": "https://example.test"}],
        "judgment": judgment,
        "frozenQrel": {"relevance": 3, "candidateScore": 1.0},
    }


class FakeApi:
    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        self.candidates = candidates
        self.candidate_writes: list[tuple[str, str, dict[str, Any]]] = []
        self.finalization_writes: list[tuple[str, dict[str, Any]]] = []

    async def list_questions(self) -> dict[str, Any]:
        return {
            "benchmarkId": "retrieval_eval_v1",
            "questions": [{"queryId": "q-1", "question": "这件藏品是否直接回答问题？", "category": "visual_motif"}],
        }

    async def question(self, query_id: str) -> dict[str, Any]:
        assert query_id == "q-1"
        return {"candidates": self.candidates, "finalization": None}

    async def put_candidate(self, query_id: str, object_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.candidate_writes.append((query_id, object_id, payload))
        return {}

    async def put_finalization(self, query_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.finalization_writes.append((query_id, payload))
        return {}


class FakeProvider:
    configured = True

    def __init__(self, responses: list[dict[str, Any] | BaseException]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def generate_retrieval_audit_json(self, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((prompt, payload))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    async def generate_qrel_vision_json(self, prompt: str, payload: dict[str, Any], images: list[Any]) -> dict[str, Any]:
        self.calls.append((prompt, payload))
        assert 1 <= len(images) <= 4
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeSuggestionStore:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def latest_for_question(self, _query_id: str) -> tuple[dict[str, dict[str, Any]], None]:
        candidates = {
            str(record["objectId"]): {
                **record["suggestion"],
                "suggestionId": f"fake-{index}",
                "suggestionRunId": record["suggestionRunId"],
                "provider": record["provider"],
                "model": record["model"],
                "promptVersion": record["promptVersion"],
                "promptHash": record["promptSha256"],
                "inputHash": record["inputSha256"],
                "validationStatus": record["validationStatus"],
                "riskFlags": record["riskFlags"],
                "modalities": record["inputModality"].split(","),
            }
            for index, record in enumerate(self.records)
            if record["kind"] == "candidate"
        }
        finalization = next(
            (
                {
                    **record["suggestion"],
                    "suggestionId": f"fake-final-{index}",
                    "suggestionRunId": record["suggestionRunId"],
                    "provider": record["provider"],
                    "model": record["model"],
                    "promptVersion": record["promptVersion"],
                    "promptHash": record["promptSha256"],
                    "inputHash": record["inputSha256"],
                    "validationStatus": record["validationStatus"],
                    "riskFlags": record["riskFlags"],
                    "modalities": record["inputModality"].split(","),
                }
                for index, record in reversed(list(enumerate(self.records)))
                if record["kind"] == "finalization"
            ),
            None,
        )
        return candidates, finalization

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        self.records.append(record)
        return record


class FakeImageCache:
    def get(self, source_url: str, edge: int) -> tuple[bytes, bool]:
        assert edge == 512
        return f"image:{source_url}".encode("utf-8"), False


class FailingImageCache:
    def get(self, _source_url: str, _edge: int) -> tuple[bytes, bool]:
        raise review.ImageFetchError("upstream_http_error", "fixture image failure")


def test_candidate_validation_rejects_cross_object_evidence_and_unsupported_claims() -> None:
    candidate = _candidate("o-1", "o-1:e1")
    with pytest.raises(ValueError, match="outside"):
        review.validate_candidate_decision(
            {"relevance": 3, "evidenceVerdict": "supports", "supportingEvidenceIds": ["other:e1"], "note": "有证据"},
            candidate,
        )
    with pytest.raises(ValueError, match="supports"):
        review.validate_candidate_decision(
            {"relevance": 1, "evidenceVerdict": "supports", "supportingEvidenceIds": ["o-1:e1"], "note": "有证据"},
            candidate,
        )


def test_scoped_dry_run_is_label_free_deterministic_and_never_writes() -> None:
    api = FakeApi([_candidate("o-1", "o-1:e1"), _candidate("o-2", "o-2:e1")])
    provider = FakeProvider(
        [
            {"decisions": [
                {"objectId": "o-1", "relevance": 3, "evidenceVerdict": "supports", "supportingEvidenceIds": ["o-1:e1"], "note": "题名和说明直接对应。"},
                {"objectId": "o-2", "relevance": 0, "evidenceVerdict": "insufficient", "supportingEvidenceIds": [], "note": "证据与问题无关。"},
            ]},
            {"expectedAnswerability": "supported", "note": "存在直接支持的馆藏证据。"},
        ]
    )
    run = asyncio.run(
        review.run_review(
            api,
            provider,
            review.ReviewScope(categories=("visual_motif",)),
        )
    )
    assert api.candidate_writes == []
    assert api.finalization_writes == []
    candidate_events = [event for event in run.events if event["kind"] == "candidate"]
    assert candidate_events[0]["decision"]["note"].startswith(review.NOTE_PREFIX)
    assert candidate_events[0]["provider"] == "deepseek"
    assert candidate_events[0]["model"] == "deepseek-configured-model"
    assert candidate_events[0]["promptVersion"] == review.CANDIDATE_PROMPT_VERSION
    assert candidate_events[0]["validationStatus"] == "validated_text_only_with_visual_error"
    assert candidate_events[0]["suggestionRunId"] == review._suggestion_run_id(
        "retrieval_eval_v1",
        "deepseek-configured-model",
        "deepseek-configured-vision-model",
        review.ReviewScope(categories=("visual_motif",)),
    )
    first_candidate_payload = provider.calls[0][1]
    serialized = str(first_candidate_payload)
    assert "candidateScore" not in serialized
    assert "frozenQrel" not in serialized
    assert "relevance': 3" not in serialized
    assert run.events[-1]["kind"] == "finalization"
    assert run.events[-1]["provider"] == "deepseek"
    assert run.events[-1]["promptVersion"] == review.FINALIZATION_PROMPT_VERSION
    assert run.events[-1]["validationStatus"] == "validated_text_only"


def test_twelve_candidates_use_one_batched_text_provider_call() -> None:
    candidates = [_candidate(f"o-{index}", f"o-{index}:e1") for index in range(12)]
    api = FakeApi(candidates)
    provider = FakeProvider([
        {"decisions": [
            {"objectId": f"o-{index}", "relevance": 0, "evidenceVerdict": "insufficient", "supportingEvidenceIds": [], "note": "无直接支持。"}
            for index in range(12)
        ]},
        {"expectedAnswerability": "unsupported", "note": "没有候选获得支持。"},
    ])
    asyncio.run(review.run_review(api, provider, review.ReviewScope(categories=("visual_motif",))))
    text_calls = [call for call in provider.calls if call[0] == review.CANDIDATE_SYSTEM_PROMPT]
    assert len(text_calls) == 1
    assert len(text_calls[0][1]["candidates"]) == 12


def test_apply_skips_existing_judgments_and_appends_only_missing_then_finalizes() -> None:
    existing = {
        "relevance": 2,
        "evidenceVerdict": "supports",
        "supportingEvidenceIds": ["o-1:e1"],
        "note": "人工既有判断",
        "revision": 4,
        "updatedAt": "2026-01-01T00:00:00+00:00",
    }
    api = FakeApi([_candidate("o-1", "o-1:e1", existing), _candidate("o-2", "o-2:e1")])
    suggestions = FakeSuggestionStore()
    provider = FakeProvider(
        [
        {"decisions": [{"objectId": "o-2", "relevance": 0, "evidenceVerdict": "insufficient", "supportingEvidenceIds": [], "note": "无直接支持。"}]},
            {"expectedAnswerability": "partially_supported", "note": "只有一个候选被直接支持。"},
        ]
    )
    asyncio.run(
        review.run_review(
            api,
            provider,
            review.ReviewScope(query_ids=("q-1",)),
            suggestion_store=suggestions,  # type: ignore[arg-type]
            dry_run=False,
            resume=True,
        )
    )
    assert api.candidate_writes == []
    assert api.finalization_writes == []
    assert [record["kind"] for record in suggestions.records] == ["candidate", "finalization"]
    assert suggestions.records[0]["objectId"] == "o-2"
    assert suggestions.records[0]["suggestion"]["note"].startswith(review.NOTE_PREFIX)
    assert suggestions.records[1]["suggestion"]["expectedAnswerability"] == "partially_supported"
    final_payload = provider.calls[-1][1]
    assert final_payload["category"] == "visual_motif"
    assert final_payload["requiredCulturalLegs"] == []
    assert final_payload["candidateJudgments"][0]["culturalLegs"] == ["test"]
    assert final_payload["candidateJudgments"][0]["metadata"]["title"] == "Object o-1"


def test_visual_batch_merges_text_and_image_and_resume_skips_provider_calls() -> None:
    api = FakeApi([
        _candidate("o-1", "o-1:e1", image_url="https://images.test/1"),
        _candidate("o-2", "o-2:e1", image_url="https://images.test/2"),
    ])
    suggestions = FakeSuggestionStore()
    provider = FakeProvider([
        {"decisions": [
            {"objectId": "o-1", "relevance": 1, "evidenceVerdict": "uncertain", "supportingEvidenceIds": [], "note": "文本需要视觉复核。"},
            {"objectId": "o-2", "relevance": 0, "evidenceVerdict": "insufficient", "supportingEvidenceIds": [], "note": "文本未支持。"},
        ]},
        {"decisions": [
            {"objectId": "o-1", "relevance": 2, "evidenceVerdict": "supports", "supportingEvidenceIds": ["o-1:e1"], "note": "图像与证据共同支持。"},
            {"objectId": "o-2", "relevance": 0, "evidenceVerdict": "insufficient", "supportingEvidenceIds": [], "note": "图像未增加支持。"},
        ]},
        {"expectedAnswerability": "partially_supported", "note": "仅一件得到有限支持。"},
    ])
    first_run = asyncio.run(review.run_review(api, provider, review.ReviewScope(categories=("visual_motif",)), suggestion_store=suggestions, image_cache=FakeImageCache(), dry_run=False))  # type: ignore[arg-type]
    candidate_records = [record for record in suggestions.records if record["kind"] == "candidate"]
    assert all(record["inputModality"] == "metadata,evidence,image" for record in candidate_records)
    assert candidate_records[0]["suggestion"]["relevance"] == 2
    assert "vision_reviewed" in candidate_records[0]["riskFlags"]
    first_event = next(
        event
        for event in first_run.events
        if event["kind"] == "candidate" and event["objectId"] == "o-1"
    )
    assert first_event["provider"] == "deepseek"
    assert first_event["model"] == "deepseek-configured-vision-model"
    assert first_event["promptVersion"] == (
        f"{review.CANDIDATE_PROMPT_VERSION}+{review.VISION_PROMPT_VERSION}"
    )
    assert first_event["validationStatus"] == "validated_vision"
    assert first_event["imageSha256"] == hashlib.sha256(
        b"image:https://images.test/1"
    ).hexdigest()
    resume_provider = FakeProvider([])
    resumed = asyncio.run(review.run_review(api, resume_provider, review.ReviewScope(categories=("visual_motif",)), suggestion_store=suggestions, image_cache=FakeImageCache(), dry_run=False, resume=True))  # type: ignore[arg-type]
    assert resume_provider.calls == []
    resumed_event = next(
        event
        for event in resumed.events
        if event["kind"] == "candidate"
        and event["status"] == "skipped_existing_ai_suggestion"
    )
    assert resumed_event["provider"] == "deepseek"
    assert resumed_event["model"] == "deepseek-configured-vision-model"
    assert resumed_event["promptVersion"].endswith(review.VISION_PROMPT_VERSION)
    assert resumed_event["validationStatus"] == "validated_vision"
    assert resumed_event["promptSha256"]
    assert resumed_event["inputSha256"]


def test_image_failure_keeps_text_grade_and_records_a_deterministic_risk() -> None:
    api = FakeApi([_candidate("o-1", "o-1:e1", image_url="https://images.test/fail")])
    suggestions = FakeSuggestionStore()
    provider = FakeProvider([
        {"decisions": [{"objectId": "o-1", "relevance": 2, "evidenceVerdict": "supports", "supportingEvidenceIds": ["o-1:e1"], "note": "文本有限支持。"}]},
        {"expectedAnswerability": "partially_supported", "note": "图像不可用，仍只按文本判断。"},
    ])
    asyncio.run(review.run_review(api, provider, review.ReviewScope(categories=("visual_motif",)), suggestion_store=suggestions, image_cache=FailingImageCache(), dry_run=False))  # type: ignore[arg-type]
    record = next(record for record in suggestions.records if record["kind"] == "candidate")
    assert record["suggestion"]["relevance"] == 2
    assert record["inputModality"] == "metadata,evidence"
    assert "error:image:image_fetch_failed:upstream_http_error" in record["riskFlags"]


def test_vision_provider_failure_caps_text_grade_and_confidence() -> None:
    api = FakeApi(
        [_candidate("o-1", "o-1:e1", image_url="https://images.test/fail-model")]
    )
    suggestions = FakeSuggestionStore()
    provider = FakeProvider(
        [
            {
                "decisions": [
                    {
                        "objectId": "o-1",
                        "relevance": 3,
                        "evidenceVerdict": "supports",
                        "supportingEvidenceIds": ["o-1:e1"],
                        "note": "文本看似直接支持。",
                    }
                ]
            },
            review.ProviderError("fixture vision failure", code="bad_request"),
            {
                "expectedAnswerability": "supported",
                "note": "仍有一件有限支持。",
            },
        ]
    )
    asyncio.run(
        review.run_review(
            api,
            provider,
            review.ReviewScope(categories=("visual_motif",)),
            suggestion_store=suggestions,  # type: ignore[arg-type]
            image_cache=FakeImageCache(),
            dry_run=False,
        )
    )
    record = next(
        record for record in suggestions.records if record["kind"] == "candidate"
    )
    assert record["suggestion"]["relevance"] == 2
    assert record["confidenceBand"] == "medium"
    assert "error:vision:bad_request" in record["riskFlags"]
    assert record["validationStatus"] == "validated_text_only_with_visual_error"


def test_vision_support_without_owned_evidence_is_downgraded_not_persisted_as_support() -> None:
    api = FakeApi(
        [_candidate("o-1", "o-1:e1", image_url="https://images.test/visual-only")]
    )
    suggestions = FakeSuggestionStore()
    provider = FakeProvider(
        [
            {
                "decisions": [
                    {
                        "objectId": "o-1",
                        "relevance": 1,
                        "evidenceVerdict": "insufficient",
                        "supportingEvidenceIds": [],
                        "note": "文本证据不足。",
                    }
                ]
            },
            {
                "decisions": [
                    {
                        "objectId": "o-1",
                        "relevance": 3,
                        "evidenceVerdict": "supports",
                        "supportingEvidenceIds": [],
                        "note": "图像中可见目标母题。",
                    }
                ]
            },
            {
                "expectedAnswerability": "partially_supported",
                "note": "只有视觉相似，缺少可归属机构证据。",
            },
        ]
    )
    asyncio.run(
        review.run_review(
            api,
            provider,
            review.ReviewScope(categories=("visual_motif",)),
            suggestion_store=suggestions,  # type: ignore[arg-type]
            image_cache=FakeImageCache(),
            dry_run=False,
        )
    )
    record = next(
        record for record in suggestions.records if record["kind"] == "candidate"
    )
    assert record["suggestion"]["relevance"] == 2
    assert record["suggestion"]["evidenceVerdict"] == "insufficient"
    assert record["suggestion"]["supportingEvidenceIds"] == []
    assert "vision_support_without_owned_evidence" in record["riskFlags"]


class MultiQuestionApi:
    async def list_questions(self) -> dict[str, Any]:
        return {
            "benchmarkId": "retrieval_eval_v1",
            "questions": [
                {
                    "queryId": "q-1",
                    "question": "第一个问题",
                    "category": "cross_cultural",
                    "requiredCulturalLegs": ["test"],
                },
                {
                    "queryId": "q-2",
                    "question": "第二个问题",
                    "category": "cross_cultural",
                    "requiredCulturalLegs": ["test"],
                },
            ],
        }

    async def question(self, query_id: str) -> dict[str, Any]:
        return {"candidates": [_candidate(f"{query_id}-o", f"{query_id}-o:e1")]}


def test_recoverable_question_failure_continues_and_report_is_incremental(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            review.ProviderError("fixture text failure", code="upstream_timeout"),
            {
                "decisions": [
                    {
                        "objectId": "q-2-o",
                        "relevance": 2,
                        "evidenceVerdict": "supports",
                        "supportingEvidenceIds": ["q-2-o:e1"],
                        "note": "第二题有机构证据。",
                    }
                ]
            },
            {
                "expectedAnswerability": "supported",
                "note": "第二题具有支持。",
            },
        ]
    )
    report_path = tmp_path / "audit.jsonl"
    snapshots: list[str] = []
    with review.IncrementalJsonlReport(
        report_path, dry_run=True, resume=False
    ) as report:
        run = asyncio.run(
            review.run_review(
                MultiQuestionApi(),
                provider,
                review.ReviewScope(categories=("cross_cultural",)),
                event_sink=report.record,
                progress_sink=lambda _index, _total, query_id, status, _detail: (
                    snapshots.append(report_path.read_text(encoding="utf-8"))
                    if query_id == "q-1" and status == "error"
                    else None
                ),
            )
        )
    assert snapshots and '"error:candidate_text:upstream_timeout"' in snapshots[0]
    assert any(
        event.get("queryId") == "q-2" and event.get("kind") == "candidate"
        for event in run.events
    )
    rows = [json.loads(line) for line in report_path.read_text(encoding="utf-8").splitlines()]
    assert any(row.get("queryId") == "q-1" and row.get("status") == "error" for row in rows)
    assert any(row.get("queryId") == "q-2" and row.get("kind") == "finalization" for row in rows)


def test_run_id_binds_vision_model_prompt_hash_and_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    scope = review.ReviewScope(categories=("visual_motif",))
    baseline = review._suggestion_run_id("benchmark", "text-a", "vision-a", scope)
    assert baseline != review._suggestion_run_id("benchmark", "text-a", "vision-b", scope)
    assert baseline != review._suggestion_run_id(
        "benchmark", "text-a", "vision-a", review.ReviewScope(query_ids=("q-1",))
    )
    monkeypatch.setattr(
        review, "VISION_SYSTEM_PROMPT", review.VISION_SYSTEM_PROMPT + "\ncontract-v-next"
    )
    assert baseline != review._suggestion_run_id("benchmark", "text-a", "vision-a", scope)


def test_finalization_consistency_gate_rejects_false_supported() -> None:
    no_support, risks = review._enforce_finalization_consistency(
        {"expectedAnswerability": "supported", "note": "模型误判。"},
        [
            {
                "culturalLegs": ["east_asia"],
                "judgment": {"relevance": 1, "evidenceVerdict": "insufficient"},
            }
        ],
        category="open_theme",
        required_cultural_legs=[],
    )
    assert no_support["expectedAnswerability"] == "unsupported"
    assert risks == ["consistency_downgraded:no_supported_candidate"]

    missing_leg, risks = review._enforce_finalization_consistency(
        {"expectedAnswerability": "supported", "note": "只覆盖一地。"},
        [
            {
                "culturalLegs": ["east_asia"],
                "judgment": {"relevance": 2, "evidenceVerdict": "supports"},
            }
        ],
        category="cross_cultural",
        required_cultural_legs=["east_asia", "europe"],
    )
    assert missing_leg["expectedAnswerability"] == "partially_supported"
    assert risks == ["consistency_downgraded:missing_cultural_legs:europe"]


def test_vision_prompt_satisfies_json_object_provider_contract() -> None:
    assert "JSON" in review.VISION_SYSTEM_PROMPT


def test_cli_requires_a_bounded_scope() -> None:
    with pytest.raises(SystemExit):
        review.parse_args([])
