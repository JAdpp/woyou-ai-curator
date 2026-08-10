from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

from .conftest import write_collection


ADMIN_HEADERS = {"X-Admin-Token": "test-admin-token"}


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key
            for item in value.values()
            for key in _all_keys(item)
        }
    if isinstance(value, list):
        return {key for item in value for key in _all_keys(item)}
    return set()


def _generate(client: TestClient, agenda_payload: dict[str, object]) -> dict[str, object]:
    response = client.post("/api/exhibitions/generate-sync", json={"agenda": agenda_payload})
    assert response.status_code == 200, response.text
    return response.json()


def _publish(client: TestClient, agenda_payload: dict[str, object]) -> dict[str, object]:
    exhibition = _generate(client, agenda_payload)
    exhibition_id = exhibition["id"]
    requested = client.post(f"/api/exhibitions/{exhibition_id}/publish")
    assert requested.status_code == 200, requested.text
    approved = client.post(
        f"/api/admin/exhibitions/{exhibition_id}/approve",
        headers=ADMIN_HEADERS,
        json={
            "reviewer": "fixture-reviewer",
            "note": "Checked all five selected records and cited evidence.",
            "evidenceReviewConfirmed": True,
        },
    )
    assert approved.status_code == 200, approved.text
    return approved.json()


def _audit_client(tmp_path: Path) -> TestClient:
    collections_dir = tmp_path / "collections"
    collection_dir = write_collection(collections_dir)
    objects_path = collection_dir / "objects.json"
    objects_payload = json.loads(objects_path.read_text(encoding="utf-8"))
    objects_payload["objects"][0]["imageUrl"] = ""
    objects_payload["objects"][1]["altText"] = ""
    objects_payload["objects"][2]["evidence"][0]["reviewed"] = False
    objects_payload["objects"][2]["evidence"][0]["reviewStatus"] = "pending_human_review"
    objects_path.write_text(
        json.dumps(objects_payload, ensure_ascii=False), encoding="utf-8"
    )
    (collection_dir / "question_cards.json").write_text(
        json.dumps(
            {
                "cards": [
                    {
                        "id": "qc-supported",
                        "title": "Supported fixture",
                        "question": "山水画如何组织观看者的行旅视线？",
                        "coverage_status": "supported",
                        "review_status": "pending_content_review",
                        "theme_id": "landscape",
                        "starter_object_ids": ["TEST.1", "TEST.2"],
                        "coverage_limits": ["Test collection only."],
                    },
                    {
                        "id": "qc-partial",
                        "question": "材料如何影响观看？",
                        "coverage_status": "partially_supported",
                        "review_status": "pending_content_review",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (collection_dir / "regression_questions.json").write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "question": "Supported fixture?",
                        "expected_status": "supported",
                        "review_status": "draft_pending_expert_review",
                    },
                    {
                        "question": "Partial fixture?",
                        "expected_status": "partially_supported",
                        "review_status": "draft_pending_expert_review",
                    },
                    {
                        "question": "Unsupported fixture?",
                        "expected_status": "unsupported",
                        "review_status": "draft_pending_expert_review",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = Settings(
        app_env="test",
        collections_dir=collections_dir,
        store_mode="memory",
        store_path=tmp_path / "unused-store.json",
        deepseek_api_key=None,
        admin_review_token="test-admin-token",
    )
    return TestClient(create_app(settings))


def test_admin_data_audit_is_protected_read_only_and_complete(tmp_path: Path) -> None:
    client = _audit_client(tmp_path)
    assert client.get("/api/admin/data-audit").status_code == 401

    response = client.get("/api/admin/data-audit", headers=ADMIN_HEADERS)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["readOnly"] is True
    assert payload["humanReviewRequired"] is True
    assert "does not perform or record human evidence review" in payload["reviewBoundary"]
    assert len(payload["collections"]) == 1

    audit = payload["collections"][0]
    assert audit["version"] == "test-v1"
    assert audit["objectCount"] == 8
    assert audit["runtimeObjectCount"] == 7
    assert audit["excludedRuntimeObjectCount"] == 1
    assert audit["themeDistribution"]["山水"] == 8
    assert audit["objectRightsDistribution"] == {"CC0": 8}
    assert audit["assets"] == {"missingImageCount": 1, "missingAltTextCount": 1}
    assert audit["evidence"]["totalChunks"] == 24
    assert audit["evidence"]["reviewedChunks"] == 23
    assert audit["evidence"]["pendingReviewChunks"] == 1
    assert audit["evidence"]["objectsPendingReview"] == 1
    assert [card["coverageStatus"] for card in audit["questionCards"]] == [
        "supported",
        "partially_supported",
    ]
    assert audit["questionCardCoverageStatusDistribution"] == {
        "partially_supported": 1,
        "supported": 1,
    }
    assert audit["regression"]["questionCount"] == 3
    assert audit["regression"]["statusDistribution"] == {
        "partially_supported": 1,
        "supported": 1,
        "unsupported": 1,
    }


def test_admin_can_withdraw_published_exhibition_and_public_routes_return_404(
    client: TestClient, agenda_payload: dict[str, object]
) -> None:
    published = _publish(client, agenda_payload)
    exhibition_id = published["id"]
    public_slug = published["slug"]
    assert client.get(f"/e/{public_slug}").status_code == 200
    assert client.get(f"/api/public/exhibitions/{public_slug}").status_code == 200

    assert client.post(f"/api/admin/exhibitions/{exhibition_id}/withdraw").status_code == 401
    withdrawn = client.post(
        f"/api/admin/exhibitions/{exhibition_id}/withdraw", headers=ADMIN_HEADERS
    )
    assert withdrawn.status_code == 200, withdrawn.text
    assert withdrawn.json()["status"] == "withdrawn"
    assert withdrawn.json()["slug"] is None
    assert client.get(f"/e/{public_slug}").status_code == 404
    assert client.get(f"/api/public/exhibitions/{public_slug}").status_code == 404
    repeated = client.post(
        f"/api/admin/exhibitions/{exhibition_id}/withdraw", headers=ADMIN_HEADERS
    )
    assert repeated.status_code == 409


def test_descriptive_export_excludes_agenda_personal_fields_and_raw_sessions(
    client: TestClient, agenda_payload: dict[str, object]
) -> None:
    private_marker = "PRIVATE-AGENDA-MARKER"
    raw_session = "raw-session-secret-123"
    private_agenda = dict(agenda_payload)
    private_agenda["personalConnection"] = private_marker
    exhibition = _generate(client, private_agenda)
    event = client.post(
        "/api/events",
        json={
            "sessionId": raw_session,
            "event": "source_opened",
            "exhibitionId": exhibition["id"],
            "parameters": {"privateMarker": private_marker},
        },
    )
    assert event.status_code == 200, event.text
    assert client.get("/api/admin/analytics/export").status_code == 401

    response = client.get("/api/admin/analytics/export", headers=ADMIN_HEADERS)
    assert response.status_code == 200, response.text
    assert response.headers["content-disposition"].startswith("attachment;")
    payload = response.json()
    serialized = json.dumps(payload, ensure_ascii=False)
    assert raw_session not in serialized
    assert private_marker not in serialized
    response_keys = _all_keys(payload)
    for forbidden_key in (
        "sessionId",
        "session_id",
        "agenda",
        "personalConnection",
        "personal_connection",
        "excludedTopics",
        "excluded_topics",
        "parameters",
    ):
        assert forbidden_key not in response_keys
    assert payload["privacy"] == {
        "aggregateOnly": True,
        "rawSessionIdsIncluded": False,
        "eventParametersIncluded": False,
        "agendaFieldsIncluded": False,
        "personalConnectionIncluded": False,
        "excludedTopicsIncluded": False,
    }
    assert payload["statistics"]["totalExhibitions"] == 1
    assert payload["statistics"]["totalEvents"] >= 3
    assert payload["runtimeVersionDistributions"]["model"] == {
        "deepseek-v4-flash": 1
    }
    assert payload["collections"][0]["version"] == "test-v1"
