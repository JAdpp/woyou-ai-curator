from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from app.config import Settings
from app.main import create_app
from app.models import Exhibition
from app.store import ExhibitionStore
from app.validator import validate_exhibition

from .conftest import write_collection


def generate(client: TestClient, agenda_payload: dict[str, object]) -> dict[str, object]:
    response = client.post("/api/exhibitions/generate-sync", json={"agenda": agenda_payload})
    assert response.status_code == 200, response.text
    return response.json()


def test_health_exposes_actual_retrieval_mode_without_local_paths(
    client: TestClient,
) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    retrieval = response.json()["retrieval"]
    assert retrieval == {
        "method": "fielded_bm25_hard_anchor",
        "version": "bm25-v1",
        "mode": "bm25",
        "available": False,
        "reason": "hybrid RAG is disabled by configuration",
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "fingerprint": None,
        "collectionId": "cma-chinese-art",
        "collectionVersion": "test-v1",
        "culturalRoutingVersion": "controlled-origin-v2",
    }
    assert "indexPath" not in retrieval
    assert "runtime/cache" not in response.text


def test_health_reports_missing_hybrid_cache_without_leaking_path(
    tmp_path: Path,
) -> None:
    collections_dir = tmp_path / "collections"
    write_collection(collections_dir)
    settings = Settings(
        app_env="test",
        collections_dir=collections_dir,
        store_mode="memory",
        rag_mode="hybrid",
        rag_index_dir=tmp_path / "private-rag-cache",
        rag_model_cache_dir=tmp_path / "private-model-cache",
        deepseek_api_key=None,
    )
    with TestClient(create_app(settings)) as test_client:
        response = test_client.get("/health")

    retrieval = response.json()["retrieval"]
    assert retrieval["mode"] == "bm25"
    assert retrieval["available"] is False
    assert retrieval["method"] == "fielded_bm25_hard_anchor"
    assert retrieval["reason"] == (
        "versioned dense cache is missing; run scripts/build_dense_index.py"
    )
    assert str(tmp_path) not in response.text


def test_health_distinguishes_shadow_baseline_from_hybrid_candidate(
    tmp_path: Path,
) -> None:
    collections_dir = tmp_path / "collections"
    write_collection(collections_dir)
    settings = Settings(
        app_env="test",
        collections_dir=collections_dir,
        store_mode="memory",
        rag_mode="shadow",
        rag_index_dir=tmp_path / "private-rag-cache",
        rag_model_cache_dir=tmp_path / "private-model-cache",
        deepseek_api_key=None,
    )
    with TestClient(create_app(settings)) as test_client:
        response = test_client.get("/health")

    retrieval = response.json()["retrieval"]
    assert retrieval["mode"] == "shadow"
    assert retrieval["method"] == "fielded_bm25_hard_anchor"
    assert retrieval["version"] == "bm25-v1"
    assert retrieval["candidateMethod"] == (
        "sqlite_object_bm25_evidence_bm25_qwen_embedding_rrf_qwen_rerank"
    )
    assert retrieval["candidateVersion"] == "hybrid-rag-v4"
    assert retrieval["candidateAvailable"] is False
    assert retrieval["candidateDenseAvailable"] is False
    assert retrieval["candidateObjectSearchAvailable"] is False
    assert retrieval["candidateEvidenceSearchAvailable"] is False
    assert retrieval["candidateStructuredFormat"] is None
    assert str(tmp_path) not in response.text


def test_landing_highlights_report_institution_counts_and_field_rights(
    client: TestClient,
) -> None:
    response = client.get("/api/collection/highlights?limit=3")
    assert response.status_code == 200
    payload = response.json()
    assert payload["objectCount"] == 8
    assert len(payload["items"]) == 3
    assert sum(entry["objectCount"] for entry in payload["institutionSummaries"]) == 8
    cma = payload["institutionSummaries"][0]
    assert cma["id"] == "cma"
    assert cma["imageLicenses"] == ["CC0 1.0"]
    assert cma["metadataLicenses"] == ["CC0 1.0"]
    assert cma["curatorialTextLicenses"] == ["CC0 1.0"]


def test_agenda_check_uses_camel_case_contract(
    client: TestClient, agenda_payload: dict[str, object]
) -> None:
    response = client.post("/api/agenda/check", json=agenda_payload)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "supported"
    assert payload["canGenerate"] is True
    assert payload["exhibitionTheme"] == agenda_payload["question"].rstrip("？?。.!！")
    assert payload["collectionVersion"] == "test-v1"
    assert payload["supportedAspects"]
    assert all("landscape-writing" not in item for item in payload["supportedAspects"])
    assert payload["coverageGaps"] == []
    assert len(payload["evidenceRoles"]) == 5
    assert payload["coverage"]["eligibleObjectCount"] == 8
    assert "prior_knowledge" not in response.text


@pytest.mark.parametrize(
    "duration_minutes,expected_status,expected_basis",
    [
        (5, "supported", "reviewed_question_card"),
        (10, "unsupported", "audit_unavailable"),
        (15, "unsupported", "audit_unavailable"),
    ],
)
def test_reviewed_card_never_promises_fewer_objects_than_the_visit_requires(
    client: TestClient,
    agenda_payload: dict[str, object],
    duration_minutes: int,
    expected_status: str,
    expected_basis: str,
) -> None:
    agenda = {**agenda_payload, "durationMinutes": duration_minutes}
    checked = client.post("/api/agenda/check", json=agenda)
    assert checked.status_code == 200, checked.text
    assert checked.json()["status"] == expected_status
    assert checked.json()["decisionBasis"] == expected_basis

    # The agenda check must never promise that a five-object reviewed spine can
    # fill an 8/12-object visit while the audit service is unavailable. The
    # legacy sync endpoint itself always builds a five-object micro-exhibition;
    # profile generation has separate variable-duration coverage.
    if duration_minutes == 5:
        generated = client.post(
            "/api/exhibitions/generate-sync", json={"agenda": agenda}
        )
        assert generated.status_code == 200, generated.text
        assert len(generated.json()["items"]) == 5


def test_generation_edit_validation_and_review_workflow(
    client: TestClient, agenda_payload: dict[str, object]
) -> None:
    exhibition = generate(client, agenda_payload)
    assert exhibition["status"] == "auto_validated"
    assert exhibition["versions"]["provider"] == "deterministic"
    assert exhibition["versions"]["model"] == "deepseek-v4-flash"
    assert exhibition["versions"]["collection"] == "test-v1"
    assert exhibition["exhibitionTheme"] == agenda_payload["question"].rstrip("？?。.!！")
    assert len(exhibition["items"]) == 5
    assert {item["role"] for item in exhibition["items"]} == {
        "opening",
        "context",
        "core_evidence",
        "contrast",
        "synthesis",
    }
    assert exhibition["validation"]["passed"] is True
    assert exhibition["validation"]["blockingIssues"] == []
    assert all(sentence["evidenceIds"] for item in exhibition["items"] for sentence in item["labelSentences"])

    exhibition_id = exhibition["id"]
    first_id = exhibition["items"][0]["id"]
    reorder = client.patch(
        f"/api/exhibitions/{exhibition_id}/items",
        json={"action": "reorder", "itemId": first_id, "direction": "down"},
    )
    assert reorder.status_code == 200, reorder.text
    reordered = reorder.json()
    assert reordered["items"][1]["id"] == first_id

    target = reordered["items"][0]
    replacement_id = target["alternatives"][0]["id"]
    replace = client.patch(
        f"/api/exhibitions/{exhibition_id}/items",
        json={"action": "replace", "itemId": target["id"], "objectId": replacement_id},
    )
    assert replace.status_code == 200, replace.text
    replaced = replace.json()
    assert replaced["items"][0]["object"]["id"] == replacement_id

    focus = client.patch(
        f"/api/exhibitions/{exhibition_id}/focus",
        json={"focus": "山水图像中的移动视点如何被馆藏材料呈现？"},
    )
    assert focus.status_code == 422
    assert focus.json()["error"]["code"] == "FOCUS_REQUIRES_NEW_EXHIBITION"

    validated = client.post(f"/api/exhibitions/{exhibition_id}/validate")
    assert validated.status_code == 200
    assert validated.json()["validation"]["passed"] is True

    submit = client.post(f"/api/exhibitions/{exhibition_id}/publish")
    assert submit.status_code == 200
    assert submit.json()["status"] == "review_pending"
    assert submit.json()["slug"] is None
    assert client.post(f"/api/exhibitions/{exhibition_id}/validate").status_code == 409
    assert client.post(f"/api/exhibitions/{exhibition_id}/publish").status_code == 409
    assert client.patch(
        f"/api/exhibitions/{exhibition_id}/focus", json={"focus": "山水画中的空间如何展开？"}
    ).status_code == 409
    assert client.patch(
        f"/api/exhibitions/{exhibition_id}/items",
        json={"action": "reorder", "itemId": submit.json()["items"][0]["id"], "direction": "down"},
    ).status_code == 409

    unauthenticated = client.get("/api/admin/exhibitions")
    assert unauthenticated.status_code == 401
    headers = {"X-Admin-Token": "test-admin-token"}
    listing = client.get("/api/admin/exhibitions", headers=headers)
    assert listing.status_code == 200
    assert isinstance(listing.json(), list)

    unconfirmed = client.post(
        f"/api/admin/exhibitions/{exhibition_id}/approve",
        headers=headers,
    )
    assert unconfirmed.status_code == 422
    approved = client.post(
        f"/api/admin/exhibitions/{exhibition_id}/approve",
        headers=headers,
        json={
            "reviewer": "fixture-reviewer",
            "note": "Checked all five selected records and cited evidence.",
            "evidenceReviewConfirmed": True,
        },
    )
    assert approved.status_code == 200, approved.text
    published = approved.json()
    assert published["status"] == "published"
    assert published["slug"]
    assert published["review"]["decision"] == "approved"
    assert published["review"]["evidenceReviewConfirmed"] is True

    public = client.get(f"/e/{published['slug']}")
    assert public.status_code == 200
    public_payload = public.json()
    assert public_payload["id"] == published["slug"]
    assert "agenda" not in public_payload
    assert "revisions" not in public_payload
    assert "review" not in public_payload
    assert "collectionId" not in public_payload
    assert "evidenceDomainId" not in public_payload
    assert public_payload["exhibitionTheme"] == published["exhibitionTheme"]
    assert all("alternatives" not in item for item in public_payload["items"])
    assert all("themes" not in item["object"] for item in public_payload["items"])
    assert "provider" not in public_payload["versions"]
    assert "reviewed" not in public_payload["items"][0]["object"]["evidence"][0]
    assert "verification" not in public_payload["items"][0]["object"]["evidence"][0]
    assert public_payload["publicationReview"]["status"] == "approved"
    assert "reviewer" not in public_payload["publicationReview"]
    assert "note" not in public_payload["publicationReview"]
    public_object = public_payload["items"][0]["object"]
    assert public_object["imageLicense"] == "CC0 1.0"
    assert public_object["imageRightsUri"] == "https://creativecommons.org/publicdomain/zero/1.0/"
    assert public_object["metadataLicense"] == "CC0 1.0"
    assert public_object["curatorialTextLicense"] == "CC0 1.0"
    public_evidence = public_object["evidence"][0]
    assert public_evidence["license"] == "CC0 1.0"
    assert public_evidence["rightsUri"] == "https://creativecommons.org/publicdomain/zero/1.0/"
    assert public_evidence["sourceKind"] == "institution_metadata"
    public_schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "public-exhibition.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(public_schema).validate(public_payload)
    public_api = client.get(f"/api/public/exhibitions/{published['slug']}")
    assert public_api.status_code == 200

    assert client.post(f"/api/exhibitions/{exhibition_id}/validate").status_code == 409
    assert client.post(f"/api/exhibitions/{exhibition_id}/publish").status_code == 409
    assert client.patch(
        f"/api/exhibitions/{exhibition_id}/focus", json={"focus": "山水画中的空间如何展开？"}
    ).status_code == 409
    assert client.patch(
        f"/api/exhibitions/{exhibition_id}/items",
        json={"action": "reorder", "itemId": published["items"][0]["id"], "direction": "down"},
    ).status_code == 409
    assert client.get(f"/api/public/exhibitions/{published['slug']}").status_code == 200


def test_event_cookie_and_descriptive_analytics(
    client: TestClient, agenda_payload: dict[str, object]
) -> None:
    exhibition = generate(client, agenda_payload)
    event = client.post(
        "/api/events",
        json={
            "event": "source_opened",
            "exhibitionId": exhibition["id"],
            "parameters": {"evidenceId": exhibition["items"][0]["object"]["evidence"][0]["id"]},
        },
    )
    assert event.status_code == 200
    assert event.json()["accepted"] is True
    assert event.json()["sessionId"].startswith("anon-")
    assert "ic_session=" in event.headers["set-cookie"]

    analytics = client.get(
        "/api/admin/analytics", headers={"Authorization": "Bearer test-admin-token"}
    )
    assert analytics.status_code == 200
    payload = analytics.json()
    assert payload["collectionObjects"] == 8
    assert payload["reviewedObjects"] == 8
    assert payload["exhibitions"] == 1
    assert sum(payload["events"].values()) >= 3
    assert "不代表学习成效" in payload["note"]


def test_validator_blocks_missing_or_foreign_evidence(
    client: TestClient, agenda_payload: dict[str, object]
) -> None:
    exhibition = Exhibition.model_validate(generate(client, agenda_payload))
    exhibition.items[0].label_sentences[0].evidence_ids = []
    exhibition.items[1].label_sentences[0].evidence_ids = ["foreign-evidence-id"]
    # Alt text that was synthesised from metadata has had no visual check, so
    # it warns rather than blocks. (Verbatim institution prose no longer needs a
    # separate content review — see 01b decision on removing the review chain.)
    exhibition.items[2].object.alt_text_source = "metadata_fallback"
    result = validate_exhibition(exhibition)
    assert result.passed is False
    assert {issue.code for issue in result.errors} >= {"EVIDENCE_REQUIRED", "EVIDENCE_NOT_FOUND"}
    assert "ALT_TEXT_SYNTHESISED" in {issue.code for issue in result.warnings}
    assert result.blocking_issues


def test_insufficient_collection_returns_structured_error(tmp_path: Path) -> None:
    base = tmp_path / "collections"
    write_collection(base, count=4)
    settings = Settings(
        app_env="test",
        collections_dir=base,
        store_mode="memory",
        store_path=tmp_path / "unused.json",
        deepseek_api_key=None,
        admin_review_token="test-admin-token",
    )
    client = TestClient(create_app(settings))
    response = client.post(
        "/api/agenda/check",
        json={
            "question": "山水画如何组织观看者的行旅视线？",
            "priorKnowledge": "不了解",
            "durationMinutes": 10,
            "excludedTopics": [],
        },
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "COLLECTION_DATA_INSUFFICIENT"
    assert error["details"]["eligibleObjectCount"] == 4
    assert error["details"]["requiredObjectCount"] == 5


def test_json_store_round_trip(
    client: TestClient, agenda_payload: dict[str, object], tmp_path: Path
) -> None:
    exhibition = Exhibition.model_validate(generate(client, agenda_payload))
    path = tmp_path / "runtime" / "store.json"
    writer = ExhibitionStore(mode="json", path=path)
    writer.save_exhibition(exhibition)
    reader = ExhibitionStore(mode="json", path=path)
    assert reader.get_exhibition(exhibition.id).id == exhibition.id

    legacy_payload = exhibition.model_dump(mode="json", by_alias=True)
    legacy_payload.pop("exhibitionTheme")
    legacy_payload["themeId"] = legacy_payload.pop("evidenceDomainId")
    migrated = Exhibition.model_validate(legacy_payload)
    assert migrated.exhibition_theme == exhibition.question.rstrip("？?。.!！")
    assert migrated.evidence_domain_id == exhibition.evidence_domain_id


def test_production_draft_access_requires_editor_token(
    collections_dir: Path, agenda_payload: dict[str, object], tmp_path: Path
) -> None:
    settings = Settings(
        app_env="production",
        collections_dir=collections_dir,
        store_mode="memory",
        store_path=tmp_path / "unused.json",
        deepseek_api_key=None,
        admin_review_token="admin-token",
        editor_access_token="editor-token",
    )
    client = TestClient(create_app(settings))
    exhibition = generate(client, agenda_payload)
    assert client.get(f"/api/exhibitions/{exhibition['id']}").status_code == 401
    assert client.get(
        f"/api/exhibitions/{exhibition['id']}", headers={"X-Editor-Token": "editor-token"}
    ).status_code == 200


def test_contract_schema_files_are_valid_json() -> None:
    contracts = Path(__file__).resolve().parents[2] / "contracts"
    for name in (
        "agenda-input.schema.json",
        "agenda-check.schema.json",
        "exhibition.schema.json",
        "public-exhibition.schema.json",
    ):
        payload = json.loads((contracts / name).read_text(encoding="utf-8"))
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
