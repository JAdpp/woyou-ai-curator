from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.collections import CollectionRepository
from app.config import Settings
from app.generator import ExhibitionGenerator
from app.main import create_app
from app.models import AgendaInput, Exhibition, SentenceType
from app.validator import _institution_fact_is_extractive, validate_exhibition


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLLECTIONS_DIR = PROJECT_ROOT / "data" / "collections"


def default_collection_dir() -> Path:
    """Directory of the collection the API actually serves by default.

    Derived rather than hardcoded so these stay integration tests of the live
    configuration instead of assertions about one frozen data version.
    """
    collection_id = CollectionRepository(COLLECTIONS_DIR).get().id
    for candidate in sorted(COLLECTIONS_DIR.glob("*/manifest.json")):
        manifest = json.loads(candidate.read_text(encoding="utf-8"))
        if manifest.get("id") == collection_id:
            return candidate.parent
    raise AssertionError(f"no directory found for collection {collection_id}")


COLLECTION_DIR = default_collection_dir()


def real_client() -> TestClient:
    return TestClient(
        create_app(
            Settings(
                app_env="test",
                collections_dir=COLLECTIONS_DIR,
                store_mode="memory",
                rag_mode="bm25",
                store_path=PROJECT_ROOT / "api" / "runtime" / "unused-policy-test.json",
                deepseek_api_key=None,
                admin_review_token="policy-test-admin",
            )
        )
    )


def agenda(question: str, duration_minutes: int = 5) -> dict[str, object]:
    return {
        "question": question,
        "priorKnowledge": "some",
        "durationMinutes": duration_minutes,
        "excludedTopics": [],
    }


def question_cards() -> list[dict[str, object]]:
    payload = json.loads((COLLECTION_DIR / "question_cards.json").read_text(encoding="utf-8"))
    return payload["cards"]


def test_free_question_fails_as_audit_outage_not_as_collection_gap() -> None:
    client = real_client()
    question = "青花瓷为什么能成为跨文化交流的证据？"
    checked = client.post("/api/agenda/check", json=agenda(question))
    assert checked.status_code == 200
    payload = checked.json()
    assert payload["status"] == "unsupported"
    assert payload["canGenerate"] is False
    assert payload["decisionBasis"] == "audit_unavailable"
    assert payload["coverage"]["matchedObjectCount"] == 0
    assert "不代表馆藏没有这个主题" in payload["coverageGaps"][0]

    generated = client.post("/api/exhibitions/generate-sync", json={"agenda": agenda(question)})
    assert generated.status_code == 422
    assert generated.json()["error"]["code"] == "RETRIEVAL_AUDIT_UNAVAILABLE"


@pytest.mark.parametrize(
    "question",
    [
        "我没想好，随便带我逛逛",
        "给我看点有意思的藏品",
    ],
)
def test_browse_agenda_check_stays_available_without_llm_audit(
    question: str,
) -> None:
    checked = real_client().post("/api/agenda/check", json=agenda(question))

    assert checked.status_code == 200
    payload = checked.json()
    assert payload["status"] == "supported"
    assert payload["canGenerate"] is True
    assert payload["requiresRuntimeAudit"] is False
    assert payload["decisionBasis"] == "browse"
    assert payload["coverage"]["matchedObjectCount"] >= 5


def test_exact_question_card_keeps_starter_order_but_near_match_is_not_forced() -> None:
    repository = CollectionRepository(COLLECTIONS_DIR)
    collection = repository.list()[0]
    card = question_cards()[0]
    expected_ids = card["starterObjectIds"]
    exact_question = str(card["question"])
    near_question = exact_question.replace("为什么", "为何", 1)

    near_policy = repository.match_question_policy(collection, near_question)
    assert near_policy is not None
    assert near_policy[0].source == "question_card"
    assert near_policy[0].policy_id == card["id"]
    assert 0.72 <= near_policy[1] < 1.0

    client = real_client()
    exact = client.post(
        "/api/exhibitions/generate-sync", json={"agenda": agenda(exact_question)}
    )
    assert exact.status_code == 200, exact.text
    exact_payload = exact.json()
    assert [item["object"]["id"] for item in exact_payload["items"]] == expected_ids
    assert exact_payload["exhibitionTheme"] == exact_question.rstrip("？?。.!！")

    # A fuzzy card match can help recognize a paraphrase, but it is not a
    # reviewed answerability decision and cannot silently force the exact
    # card's five-object spine. This is essential when a visitor appends a
    # new legal, conservation or living-practice predicate to familiar words.
    near_check = client.post("/api/agenda/check", json=agenda(near_question))
    assert near_check.status_code == 200
    assert near_check.json()["decisionBasis"] == "audit_unavailable"
    assert near_check.json()["canGenerate"] is False
    near = client.post(
        "/api/exhibitions/generate-sync", json={"agenda": agenda(near_question)}
    )
    assert near.status_code == 422, near.text
    assert near.json()["error"]["code"] == "RETRIEVAL_AUDIT_UNAVAILABLE"
    near_agenda = AgendaInput(
        question=near_question,
        priorKnowledge="some",
        durationMinutes=10,
        collectionId=collection.id,
    )
    near_results = repository.search(near_agenda, collection)
    generator = ExhibitionGenerator(
        Settings(
            collections_dir=COLLECTIONS_DIR,
            rag_mode="bm25",
            deepseek_api_key=None,
        ),
        repository,
    )
    assert generator._context(near_agenda, all_results=near_results).policy is None


def test_replacement_is_limited_to_current_item_alternative_whitelist() -> None:
    client = real_client()
    card = question_cards()[0]
    generated = client.post(
        "/api/exhibitions/generate-sync", json={"agenda": agenda(str(card["question"]))}
    )
    assert generated.status_code == 200
    exhibition = generated.json()
    target = exhibition["items"][0]
    allowed_id = target["alternatives"][0]["id"]
    object_themes = {
        obj["id"]: set(obj["themes"])
        for obj in json.loads((COLLECTION_DIR / "objects.json").read_text(encoding="utf-8"))
    }
    # Alternatives must stay inside the exhibition's own coverage domain, so a
    # swap cannot quietly move the argument to unrelated material.
    domain_id = str(card["evidenceDomainId"])
    assert all(
        domain_id in object_themes[alternative["id"]]
        for alternative in target["alternatives"]
    )

    allowed = client.patch(
        f"/api/exhibitions/{exhibition['id']}/items",
        json={"action": "replace", "itemId": target["id"], "objectId": allowed_id},
    )
    assert allowed.status_code == 200, allowed.text

    current = allowed.json()
    current_target = next(item for item in current["items"] if item["id"] == target["id"])
    used_ids = {item["object"]["id"] for item in current["items"]}
    allowed_ids = {item["id"] for item in current_target["alternatives"]}
    all_objects = json.loads((COLLECTION_DIR / "objects.json").read_text(encoding="utf-8"))
    arbitrary_id = next(
        obj["id"]
        for obj in all_objects
        if obj["id"] not in used_ids and obj["id"] not in allowed_ids
    )
    rejected = client.patch(
        f"/api/exhibitions/{exhibition['id']}/items",
        json={
            "action": "replace",
            "itemId": current_target["id"],
            "objectId": arbitrary_id,
        },
    )
    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "REPLACEMENT_NOT_ALLOWED"


def test_all_curated_regression_answerability_labels_are_preserved() -> None:
    repository = CollectionRepository(COLLECTIONS_DIR)
    payload = json.loads(
        (COLLECTION_DIR / "regression_questions.json").read_text(encoding="utf-8")
    )
    observed: dict[str, int] = {}
    detected_themes: set[str] = set()
    for row in payload["questions"]:
        result = ExhibitionGenerator.probe_answerability(
            repository,
            AgendaInput(
                question=row["question"],
                priorKnowledge="some",
                # The curated label set describes the evidence hypothesis at a
                # five-object route; it is not permission to bypass the live
                # semantic audit when the visitor actually generates a visit.
                durationMinutes=5,
                excludedTopics=[],
            ),
            audit_available=True,
        )
        assert result.status == row["expectedStatus"], row["id"]
        if row["expectedStatus"] == "supported" and result.decision_basis != "reviewed_question_card":
            assert result.requires_runtime_audit is True, row["id"]
        assert result.exhibition_theme == row["question"].rstrip("？?。.!！")
        assert row.get("evidenceDomainId", "") not in result.supported_aspects
        detected_themes.add(result.exhibition_theme)
        observed[str(result.status)] = observed.get(str(result.status), 0) + 1
    expected = {}
    for row in payload["questions"]:
        expected[row["expectedStatus"]] = expected.get(row["expectedStatus"], 0) + 1
    assert observed == expected
    assert len(detected_themes) == len(payload["questions"])
    # The gate has to actually discriminate, not just always say yes.
    assert expected.get("unsupported", 0) >= 5
    assert expected.get("partially_supported", 0) >= 5


def test_institution_fact_must_be_extractive_and_model_cannot_overwrite_it() -> None:
    client = real_client()
    card = question_cards()[0]
    generated = client.post(
        "/api/exhibitions/generate-sync", json={"agenda": agenda(str(card["question"]))}
    )
    exhibition = Exhibition.model_validate(generated.json())
    first_fact = exhibition.items[0].label_sentences[0]
    # The public deterministic floor is now a system-authored catalogue
    # summary; construct an institution-fact claim explicitly to keep testing
    # the validator's extractive-source boundary.
    first_fact.type = SentenceType.INSTITUTION_FACT
    first_fact.text = "这是一条没有被引用材料直接支持的任意馆方事实。"
    validation = validate_exhibition(exhibition)
    assert validation.passed is False
    assert "INSTITUTION_FACT_NOT_EXTRACTIVE" in {
        issue.code for issue in validation.errors
    }

    clean_exhibition = Exhibition.model_validate(generated.json())
    malicious_output = {
        "title": clean_exhibition.title,
        "curatorialThesis": clean_exhibition.curatorial_thesis,
        "coreAnswer": clean_exhibition.core_answer,
        "subQuestions": clean_exhibition.sub_questions,
        "coverageLimits": [],
        "items": [
            {
                "objectId": item.object.id,
                "role": item.role,
                "subQuestion": item.sub_question,
                "whySelected": item.why_selected,
                "relation": item.relation,
                "labelSentences": [
                    {
                        "text": "模型声称这是一条馆方事实。",
                        "type": "institution_fact",
                        "evidenceIds": [item.object.evidence[0].id],
                    }
                ],
            }
            for item in clean_exhibition.items
        ],
    }
    with pytest.raises(ValueError, match="not allowed"):
        ExhibitionGenerator._apply_model_output(clean_exhibition, malicious_output)


def test_short_exact_catalogue_fact_is_valid_but_short_fuzzy_text_is_not() -> None:
    assert _institution_fact_is_extractive("Dogs", ["Dogs"]) is True
    assert _institution_fact_is_extractive("Dog", ["Dogs"]) is False
