from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.qrel_review import _is_material_ai_suggestion_risk


PREFIX = "/api/admin/retrieval-eval/reviews"


def test_delegated_origin_is_counted_and_candidate_edit_invalidates_finalization(tmp_path: Path) -> None:
    settings = _settings(tmp_path, _write_frozen_dataset(tmp_path))
    with TestClient(create_app(settings)) as client:
        saved = client.put(f"{PREFIX}/questions/q-1/candidates/cma:1", json=_candidate_payload(reviewOrigin="delegated_ai"))
        assert saved.status_code == 200
        assert saved.json()["progress"]["delegatedReviewedCandidates"] == 1
        assert saved.json()["progress"]["humanReviewedCandidates"] == 0
        finalized = client.put(f"{PREFIX}/questions/q-1/finalization", json={
            "expectedAnswerability": "supported", "requestId": "origin-final",
            "reviewOrigin": "delegated_ai", "expectedRevision": None,
        })
        assert finalized.status_code == 200
        assert finalized.json()["finalization"]["isCurrent"] is True
        changed = client.put(f"{PREFIX}/questions/q-1/candidates/cma:1", json=_candidate_payload(
            relevance=1, evidenceVerdict="insufficient", expectedRevision=1, requestId="origin-edit",
        ))
        assert changed.status_code == 200
        assert changed.json()["progress"]["humanReviewedCandidates"] == 1
        assert changed.json()["progress"]["completedQuestions"] == 0
        detail = client.get(f"{PREFIX}/questions/q-1").json()
        assert detail["question"]["status"] == "in_progress"
        assert detail["finalization"]["revision"] == 1
        assert detail["finalization"]["isCurrent"] is False
        snapshot = client.get(f"{PREFIX}/export").json()
        assert [row["reviewOrigin"] for row in snapshot["revisions"]] == ["delegated_ai", "delegated_ai", "human"]
        unsupported = client.put(f"{PREFIX}/questions/q-1/finalization", json={
            "expectedAnswerability": "supported", "requestId": "invalid-supported", "expectedRevision": 1,
        })
        assert unsupported.status_code == 422


def test_question_cultures_come_from_question_not_candidate_union(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path, _write_frozen_dataset(tmp_path)))
    with TestClient(app) as client:
        client.get(f"{PREFIX}/questions")
        dataset = app.state.qrel_review_workbench.dataset
        dataset.questions["q-1"]["requiredCulturalLegs"] = ["east_asia"]
        dataset.qrels_by_question["q-1"][0]["culturalLegs"] = ["europe", "americas"]
        summary = client.get(f"{PREFIX}/questions/q-1").json()["question"]
        assert summary["requiredCulturalLegs"] == ["east_asia"]
        assert client.put(f"{PREFIX}/questions/q-1/candidates/cma:1", json=_candidate_payload()).status_code == 200
        final = client.put(f"{PREFIX}/questions/q-1/finalization", json={
            "expectedAnswerability": "supported", "requestId": "missing-culture", "expectedRevision": None,
        })
        assert final.status_code == 422
        assert final.json()["detail"]["missingCulturalLegs"] == ["east_asia"]


def test_material_ai_risk_ignores_completed_visual_workflow_and_clear_negative() -> None:
    clear_negative = {
        "relevance": 0,
        "evidenceVerdict": "not_applicable",
        "confidenceBand": "low",
        "riskFlags": ["low_confidence", "visual_review_required", "vision_reviewed"],
        "validationStatus": "validated_vision",
    }
    assert _is_material_ai_suggestion_risk(clear_negative) is False
    assert _is_material_ai_suggestion_risk(
        {**clear_negative, "riskFlags": ["low_confidence", "visual_review_required"]}
    ) is True
    assert _is_material_ai_suggestion_risk(
        {
            **clear_negative,
            "confidenceBand": "medium",
            "riskFlags": ["vision_reviewed", "vision_support_without_owned_evidence"],
        }
    ) is True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_frozen_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    dataset_dir = root / "data" / "qa" / "retrieval_eval_v1"
    objects_path = root / "data" / "collections" / "global_open" / "objects.json"
    dataset_dir.mkdir(parents=True)
    objects_path.parent.mkdir(parents=True)
    objects_path.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "id": "cma:1",
                        "title": "Verified object",
                        "evidence": [
                            {"id": "cma:1:description", "text": "Verified institution evidence", "sourceUrl": "https://example.test/1", "reviewed": True, "verification": "source_exact_match"}
                        ],
                    },
                    {
                        "id": "cma:2",
                        "title": "Other object",
                        "evidence": [
                            {"id": "cma:2:description", "text": "Other evidence", "sourceUrl": "https://example.test/2", "reviewed": False, "reviewStatus": "source_exact_match"}
                        ],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    questions_path = dataset_dir / "questions.jsonl"
    qrels_path = dataset_dir / "qrels.jsonl"
    questions_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
            {
                "queryId": "q-1",
                "question": "Which verified object is relevant?",
                "requiredCulturalLegs": [],
                "category": "fixture",
                "judgmentMode": "pooled_silver_pending_human_review",
                "expectedAnswerability": "supported",
            },
            {
                "queryId": "q-2",
                "question": "Which other object is relevant?",
                "category": "fixture",
                "judgmentMode": "pooled_silver_pending_human_review",
                "expectedAnswerability": "supported",
            },
            )
        ) + "\n",
        encoding="utf-8",
    )
    qrels_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
            {
                "queryId": "q-1",
                "objectId": "cma:1",
                "culturalLegs": ["china"],
                "relevance": 0,
                "supportingEvidenceIds": ["cma:1:description"],
                "judgmentStatus": "pooled_silver_pending_human_review",
            },
            {
                "queryId": "q-2",
                "objectId": "cma:2",
                "relevance": 0,
                "supportingEvidenceIds": ["cma:2:description"],
                "judgmentStatus": "pooled_silver_pending_human_review",
            },
            )
        ) + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "benchmarkId": "retrieval_eval_v1",
                "status": "frozen",
                "version": "fixture-v1",
                "provenance": {
                    "objectsFile": "data/collections/global_open/objects.json",
                    "objectsSha256": _sha256(objects_path),
                },
                "files": {
                    "questions": {"sha256": _sha256(questions_path), "rows": 2},
                    "qrels": {"sha256": _sha256(qrels_path), "rows": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    return dataset_dir


def _settings(tmp_path: Path, dataset_dir: Path) -> Settings:
    return Settings(
        app_env="test",
        collections_dir=tmp_path / "collections",
        store_mode="memory",
        qrel_review_dataset_dir=dataset_dir,
        qrel_review_db_path=tmp_path / "reviews.sqlite3",
        qrel_suggestion_db_path=tmp_path / "suggestions.sqlite3",
    )


def _candidate_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "relevance": 3,
        "evidenceVerdict": "supports",
        "supportingEvidenceIds": ["cma:1:description"],
        "note": "Direct title and record evidence.",
        "expectedRevision": None,
        "requestId": "candidate-request-1",
    }
    payload.update(overrides)
    return payload


def _ai_record(
    *,
    run_id: str,
    query_id: str = "q-1",
    object_id: str | None = "cma:1",
    kind: str = "candidate",
    suggestion: dict[str, object] | None = None,
) -> dict[str, object]:
    if suggestion is None:
        suggestion = (
            {
                "relevance": 3,
                "evidenceVerdict": "supports",
                "supportingEvidenceIds": [f"{object_id}:description"],
                "note": "AI代审，待复核：直接支持。",
            }
            if kind == "candidate"
            else {
                "expectedAnswerability": "supported",
                "note": "AI代审，待复核：有直接支持。",
            }
        )
    return {
        "suggestionRunId": run_id,
        "kind": kind,
        "queryId": query_id,
        "objectId": object_id,
        "provider": "deepseek",
        "model": "fixture-model",
        "promptVersion": f"{kind}-fixture-v1",
        "promptSha256": "a" * 64,
        "inputModality": "text_evidence",
        "inputSha256": "b" * 64,
        "confidenceBand": "medium",
        "riskFlags": ["ai_draft", "needs_human_confirmation"],
        "validationStatus": "validated_text_only",
        "suggestion": suggestion,
    }


def test_qrel_review_opens_without_credentials_but_stays_disabled_in_production(tmp_path: Path) -> None:
    dataset_dir = _write_frozen_dataset(tmp_path)
    settings = _settings(tmp_path, dataset_dir)
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.get(f"{PREFIX}/questions")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["questions"][0]["queryId"] == "q-1"
    with TestClient(create_app(replace(settings, app_env="production"))) as client:
        assert client.get(f"{PREFIX}/questions").status_code == 404


def test_qrel_review_is_append_only_idempotent_and_conflict_checked(tmp_path: Path) -> None:
    dataset_dir = _write_frozen_dataset(tmp_path)
    questions_before = (dataset_dir / "questions.jsonl").read_bytes()
    qrels_before = (dataset_dir / "qrels.jsonl").read_bytes()
    settings = _settings(tmp_path, dataset_dir)
    candidate_url = f"{PREFIX}/questions/q-1/candidates/cma:1"
    with TestClient(create_app(settings)) as client:
        response = client.put(candidate_url, json=_candidate_payload())
        assert response.status_code == 200
        assert response.json()["judgment"]["revision"] == 1
        assert "reviewerId" not in response.json()["judgment"]
        assert client.put(candidate_url, json=_candidate_payload()).json()["judgment"]["revision"] == 1
        reused_request = client.put(
            candidate_url,
            json=_candidate_payload(note="A different payload must not replay.", expectedRevision=1),
        )
        assert reused_request.status_code == 409
        conflict = client.put(candidate_url, json=_candidate_payload(requestId="candidate-request-2"))
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["currentRevision"] == 1
        invalid_evidence = client.put(
            candidate_url,
            json=_candidate_payload(requestId="candidate-request-3", expectedRevision=1, supportingEvidenceIds=["cma:2:description"]),
        )
        assert invalid_evidence.status_code == 422
        assert client.put(
            candidate_url,
            json=_candidate_payload(requestId="candidate-request-4", relevance=0),
        ).status_code == 422
        assert client.put(
            f"{PREFIX}/questions/q-1/candidates/cma:2",
            json=_candidate_payload(requestId="candidate-request-5"),
        ).status_code == 422
        assert client.put(
            candidate_url,
            json=_candidate_payload(requestId="candidate-request-6", reviewerId="forged-reviewer"),
        ).status_code == 422
        assert client.put(
            f"{PREFIX}/questions/q-1/finalization",
            json={"expectedAnswerability": "supported", "note": "One reviewed candidate is sufficient.", "expectedRevision": None, "requestId": "final-request-1"},
        ).json()["revision"] == 1
    assert (dataset_dir / "questions.jsonl").read_bytes() == questions_before
    assert (dataset_dir / "qrels.jsonl").read_bytes() == qrels_before


def test_qrel_review_verifies_hashes_and_reloads_append_only_database(tmp_path: Path) -> None:
    dataset_dir = _write_frozen_dataset(tmp_path)
    settings = _settings(tmp_path, dataset_dir)
    candidate_url = f"{PREFIX}/questions/q-1/candidates/cma:1"
    with TestClient(create_app(settings)) as client:
        assert client.put(candidate_url, json=_candidate_payload()).status_code == 200
    # A new app process sees the durable SQLite revision without reparsing it
    # into (or modifying) the frozen qrels file.
    with TestClient(create_app(settings)) as client:
        detail = client.get(f"{PREFIX}/questions/q-1")
    assert detail.status_code == 200
    assert detail.json()["candidates"][0]["judgment"]["revision"] == 1

    (dataset_dir / "qrels.jsonl").write_text("{}\n", encoding="utf-8")
    with TestClient(create_app(_settings(tmp_path / "bad", dataset_dir))) as client:
        assert client.get(f"{PREFIX}/questions").status_code == 503


def test_ai_suggestion_is_visible_but_never_changes_human_progress_or_snapshot(tmp_path: Path) -> None:
    dataset_dir = _write_frozen_dataset(tmp_path)
    app = create_app(_settings(tmp_path, dataset_dir))
    with TestClient(app) as client:
        assert client.get(f"{PREFIX}/questions").json()["progress"]["reviewedCandidates"] == 0
        workbench = app.state.qrel_review_workbench
        workbench.suggestion_store.append(
            {
                "suggestionRunId": "run-1", "kind": "candidate", "queryId": "q-1", "objectId": "cma:1",
                "provider": "deepseek", "model": "fixture-model", "promptVersion": "candidate-text-v1",
                "promptSha256": "a" * 64, "inputModality": "text_evidence", "inputSha256": "b" * 64,
                "confidenceBand": "medium", "riskFlags": ["ai_draft", "needs_human_confirmation"],
                "validationStatus": "validated_text_only",
                "suggestion": {"relevance": 3, "evidenceVerdict": "supports", "supportingEvidenceIds": ["cma:1:description"], "note": "AI代审，待复核：直接支持。"},
            }
        )
        listing = client.get(f"{PREFIX}/questions").json()
        assert listing["progress"]["reviewedCandidates"] == 0
        assert listing["questions"][0]["suggestedCandidateCount"] == 1
        assert listing["questions"][0]["aiHighRiskCount"] == 0
        detail = client.get(f"{PREFIX}/questions/q-1").json()
        assert detail["candidates"][0]["aiSuggestion"]["suggestionId"].startswith("ai-suggestion-")

        # A completed vision pass is provenance, while an unavailable/failing
        # image is a substantive reason to prioritize human review.
        workbench.suggestion_store.append(
            {
                "suggestionRunId": "run-2", "kind": "candidate", "queryId": "q-1", "objectId": "cma:1",
                "provider": "deepseek", "model": "fixture-model", "promptVersion": "candidate-text-v1+vision-v1",
                "promptSha256": "c" * 64, "inputModality": "metadata,evidence", "inputSha256": "d" * 64,
                "confidenceBand": "medium", "riskFlags": ["ai_draft", "vision_reviewed", "image_fetch_failed:upstream_http_error"],
                "validationStatus": "validated_text_only",
                "suggestion": {"relevance": 2, "evidenceVerdict": "supports", "supportingEvidenceIds": ["cma:1:description"], "note": "图像失败，暂按文本证据判断。"},
            }
        )
        risky_listing = client.get(f"{PREFIX}/questions").json()
        assert risky_listing["questions"][0]["aiHighRiskCount"] == 1
        assert risky_listing["progress"]["reviewedCandidates"] == 0
        # An AI proposal is not a human review and cannot open finalization.
        blocked = client.put(
            f"{PREFIX}/questions/q-1/finalization",
            json={"expectedAnswerability": "supported", "note": "Human finalization", "expectedRevision": None, "requestId": "human-final"},
        )
        assert blocked.status_code == 409
        snapshot = client.get(f"{PREFIX}/export").json()
        assert snapshot["revisions"] == []


def test_ai_suggestion_store_rejects_invalid_decisions_and_inconsistent_finalization(tmp_path: Path) -> None:
    dataset_dir = _write_frozen_dataset(tmp_path)
    app = create_app(_settings(tmp_path, dataset_dir))
    with TestClient(app) as client:
        assert client.get(f"{PREFIX}/questions").status_code == 200
        store = app.state.qrel_review_workbench.suggestion_store
        valid = _ai_record(run_id="validation-run")

        invalid_suggestions = [
            {**valid["suggestion"], "unexpected": True},
            {**valid["suggestion"], "relevance": True},
            {**valid["suggestion"], "relevance": 4},
            {**valid["suggestion"], "evidenceVerdict": "invented"},
            {**valid["suggestion"], "supportingEvidenceIds": ["cma:2:description"]},
            {**valid["suggestion"], "relevance": 1},
            {**valid["suggestion"], "supportingEvidenceIds": []},
            {**valid["suggestion"], "note": "  "},
        ]
        for index, suggestion in enumerate(invalid_suggestions):
            record = deepcopy(valid)
            record["suggestionRunId"] = f"invalid-{index}"
            record["suggestion"] = suggestion
            with pytest.raises((ValueError, TypeError)):
                store.append(record)

        unsupported = _ai_record(
            run_id="unsupported-run",
            kind="finalization",
            object_id=None,
            suggestion={
                "expectedAnswerability": "unsupported",
                "note": "AI代审，待复核：没有支持候选。",
            },
        )
        assert store.append(unsupported)["expectedAnswerability"] == "unsupported"

        store.append(
            _ai_record(
                run_id="no-support-run",
                suggestion={
                    "relevance": 1,
                    "evidenceVerdict": "insufficient",
                    "supportingEvidenceIds": [],
                    "note": "AI代审，待复核：只有主题邻近。",
                },
            )
        )
        supported_without_candidate = _ai_record(
            run_id="no-support-run", kind="finalization", object_id=None
        )
        with pytest.raises(ValueError, match="without a relevance 2/3 supporting candidate"):
            store.append(supported_without_candidate)

        stored_candidate = store.append(_ai_record(run_id="support-run"))
        assert stored_candidate["relevance"] == 3
        assert (
            store.append(_ai_record(run_id="support-run"))["suggestionId"]
            == stored_candidate["suggestionId"]
        )
        changed_provenance = _ai_record(run_id="support-run")
        changed_provenance["model"] = "different-model"
        with pytest.raises(ValueError, match="different suggestion data or provenance"):
            store.append(changed_provenance)
        supported = store.append(
            _ai_record(run_id="support-run", kind="finalization", object_id=None)
        )
        assert supported["expectedAnswerability"] == "supported"


def test_candidate_ai_disposition_is_resource_bound_and_survives_reload(tmp_path: Path) -> None:
    dataset_dir = _write_frozen_dataset(tmp_path)
    settings = _settings(tmp_path, dataset_dir)
    app = create_app(settings)
    candidate_url = f"{PREFIX}/questions/q-1/candidates/cma:1"
    with TestClient(app) as client:
        assert client.get(f"{PREFIX}/questions").status_code == 200
        store = app.state.qrel_review_workbench.suggestion_store
        suggestion = store.append(_ai_record(run_id="accept-run"))
        other = store.append(
            _ai_record(
                run_id="other-run", query_id="q-2", object_id="cma:2"
            )
        )

        assert client.put(
            candidate_url,
            json=_candidate_payload(
                disposition="accepted",
                acceptedSuggestionId=None,
                requestId="missing-reference",
            ),
        ).status_code == 422
        assert client.put(
            candidate_url,
            json=_candidate_payload(
                disposition="human_only",
                acceptedSuggestionId=suggestion["suggestionId"],
                requestId="human-only-reference",
            ),
        ).status_code == 422
        assert client.put(
            candidate_url,
            json=_candidate_payload(
                disposition="accepted",
                acceptedSuggestionId=other["suggestionId"],
                requestId="wrong-resource-reference",
            ),
        ).status_code == 422
        assert client.put(
            candidate_url,
            json=_candidate_payload(
                relevance=2,
                disposition="accepted",
                acceptedSuggestionId=suggestion["suggestionId"],
                requestId="accepted-but-different",
            ),
        ).status_code == 422

        response = client.put(
            candidate_url,
            json=_candidate_payload(
                note="The human accepts the decision and records an independent note.",
                disposition="accepted",
                acceptedSuggestionId=suggestion["suggestionId"],
                requestId="accepted-valid",
            ),
        )
        assert response.status_code == 200
        assert response.json()["judgment"]["acceptedSuggestionId"] == suggestion["suggestionId"]
        assert response.json()["judgment"]["disposition"] == "accepted"

    with TestClient(create_app(settings)) as client:
        judgment = client.get(f"{PREFIX}/questions/q-1").json()["candidates"][0]["judgment"]
    assert judgment["acceptedSuggestionId"] == suggestion["suggestionId"]
    assert judgment["disposition"] == "accepted"


@pytest.mark.parametrize("disposition", ["modified", "rejected"])
def test_candidate_modified_and_rejected_dispositions_require_a_changed_decision(
    tmp_path: Path, disposition: str
) -> None:
    scoped_path = tmp_path / disposition
    dataset_dir = _write_frozen_dataset(scoped_path)
    app = create_app(_settings(scoped_path, dataset_dir))
    candidate_url = f"{PREFIX}/questions/q-1/candidates/cma:1"
    with TestClient(app) as client:
        assert client.get(f"{PREFIX}/questions").status_code == 200
        suggestion = app.state.qrel_review_workbench.suggestion_store.append(
            _ai_record(run_id=f"{disposition}-run")
        )
        exact = client.put(
            candidate_url,
            json=_candidate_payload(
                note=suggestion["note"],
                disposition=disposition,
                acceptedSuggestionId=suggestion["suggestionId"],
                requestId=f"{disposition}-exact",
            ),
        )
        assert exact.status_code == 422

        changed = client.put(
            candidate_url,
            json=_candidate_payload(
                relevance=0,
                evidenceVerdict="insufficient",
                supportingEvidenceIds=[],
                disposition=disposition,
                acceptedSuggestionId=suggestion["suggestionId"],
                requestId=f"{disposition}-changed",
            ),
        )
        assert changed.status_code == 200
        assert changed.json()["judgment"]["disposition"] == disposition


def test_finalization_ai_acceptance_is_audited_and_reloaded_without_affecting_gate(tmp_path: Path) -> None:
    dataset_dir = _write_frozen_dataset(tmp_path)
    settings = _settings(tmp_path, dataset_dir)
    app = create_app(settings)
    candidate_url = f"{PREFIX}/questions/q-1/candidates/cma:1"
    finalization_url = f"{PREFIX}/questions/q-1/finalization"
    with TestClient(app) as client:
        assert client.get(f"{PREFIX}/questions").status_code == 200
        store = app.state.qrel_review_workbench.suggestion_store
        store.append(_ai_record(run_id="finalization-run"))
        finalization_suggestion = store.append(
            _ai_record(
                run_id="finalization-run", kind="finalization", object_id=None
            )
        )

        # AI candidate coverage alone still cannot unlock the human gate.
        blocked = client.put(
            finalization_url,
            json={
                "expectedAnswerability": "supported",
                "note": "Premature finalization",
                "acceptedSuggestionId": finalization_suggestion["suggestionId"],
                "disposition": "accepted",
                "expectedRevision": None,
                "requestId": "premature-finalization",
            },
        )
        assert blocked.status_code == 409

        assert client.put(candidate_url, json=_candidate_payload()).status_code == 200
        finalized = client.put(
            finalization_url,
            json={
                "expectedAnswerability": "supported",
                "note": "Human note for the accepted answerability decision.",
                "acceptedSuggestionId": finalization_suggestion["suggestionId"],
                "disposition": "accepted",
                "expectedRevision": None,
                "requestId": "accepted-finalization",
            },
        )
        assert finalized.status_code == 200
        assert finalized.json()["finalization"]["acceptedSuggestionId"] == finalization_suggestion["suggestionId"]
        assert finalized.json()["finalization"]["disposition"] == "accepted"

    with TestClient(create_app(settings)) as client:
        detail = client.get(f"{PREFIX}/questions/q-1").json()
        snapshot = client.get(f"{PREFIX}/export").json()
    assert detail["finalization"]["acceptedSuggestionId"] == finalization_suggestion["suggestionId"]
    assert detail["finalization"]["disposition"] == "accepted"
    assert detail["progress"]["reviewedCandidates"] == 1
    assert all("suggestionRunId" not in revision for revision in snapshot["revisions"])
