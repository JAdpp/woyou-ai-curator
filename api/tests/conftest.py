from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def write_collection(base: Path, count: int = 8) -> Path:
    collection_dir = base / "cma-chinese-art"
    collection_dir.mkdir(parents=True)
    manifest = {
        "id": "cma-chinese-art",
        "name": "CMA Chinese Art test fixture",
        "institution": "Cleveland Museum of Art",
        "version": "test-v1",
        "license": "CC0",
        "sourceUrl": "https://openaccess-api.clevelandart.org/",
        "questionCards": [
            "山水画如何组织观看者的行旅视线？",
            "书法与诗如何共同构成观看经验？"
        ]
    }
    objects = []
    for index in range(count):
        object_id = f"TEST.{index + 1}"
        objects.append(
            {
                "id": object_id,
                "sourceId": index + 1,
                "title": f"Landscape object {index + 1}",
                "date": f"{1200 + index}",
                "maker": f"Maker {index + 1}",
                "medium": "Ink on silk" if index % 2 == 0 else "Ink on paper",
                "type": "Painting",
                "culture": ["China"],
                "description": f"Institution description for landscape object {index + 1}.",
                "imageUrl": f"https://example.test/images/{index + 1}.jpg",
                "objectUrl": f"https://example.test/objects/{index + 1}",
                "rights": "CC0",
                "altText": f"Landscape object {index + 1}",
                "altTextSource": "institution_authored",
                "institution": "Cleveland Museum of Art",
                "institutionId": "cma",
                "classification": "Painting",
                # Fixture objects carry institution prose, so they are allowed
                # to hold a core-evidence role.
                "evidenceDepth": "full",
                "themes": ["山水", "观看", f"主题{index % 3}"],
                "evidence": [
                    {
                        "id": f"{object_id}-e{evidence_index}",
                        "text": f"Original institution evidence {evidence_index} for landscape object {index + 1}.",
                        "sourceUrl": f"https://example.test/objects/{index + 1}",
                        "sourceTitle": f"Institution record: Landscape object {index + 1}",
                        "sourceLocation": f"description[{evidence_index}]",
                        "supports": "title, material, and institutional description",
                        "reviewed": True,
                        "reviewStatus": "reviewed",
                        "verification": "source_exact_match"
                    }
                    for evidence_index in range(1, 4)
                ]
            }
        )
    (collection_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    (collection_dir / "objects.json").write_text(
        json.dumps({"objects": objects}, ensure_ascii=False), encoding="utf-8"
    )
    # The shared API fixture has no external audit provider. Its canonical
    # agenda therefore needs an explicit reviewed five-object spine; arbitrary
    # free-form questions are intentionally fail-closed in audit-outage mode.
    starter_ids = [obj["id"] for obj in objects[:5]]
    (collection_dir / "question_cards.json").write_text(
        json.dumps(
            {
                "cards": [
                    {
                        "id": "fixture-landscape-route",
                        "question": "山水画如何组织观看者的行旅视线？",
                        "coverageStatus": "supported",
                        "reviewStatus": "fixture_reviewed",
                        "evidenceDomainId": "山水",
                        "starterObjectIds": starter_ids,
                        "coverageLimits": [],
                    },
                    {
                        "id": "fixture-calligraphy-route",
                        "question": "书法与诗如何共同构成观看经验？",
                        "coverageStatus": "supported",
                        "reviewStatus": "fixture_reviewed",
                        "evidenceDomainId": "观看",
                        "starterObjectIds": starter_ids,
                        "coverageLimits": [],
                    },
                    {
                        "id": "fixture-mobile-viewpoint-route",
                        "question": "山水图像中的移动视点如何被馆藏材料呈现？",
                        "coverageStatus": "supported",
                        "reviewStatus": "fixture_reviewed",
                        "evidenceDomainId": "山水",
                        "starterObjectIds": starter_ids,
                        "coverageLimits": [],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return collection_dir


@pytest.fixture()
def collections_dir(tmp_path: Path) -> Path:
    base = tmp_path / "collections"
    write_collection(base)
    return base


@pytest.fixture()
def client(collections_dir: Path):
    """Context-managed client.

    Entering the context keeps one event loop alive for the whole test, which
    background curation jobs need — without it each request gets its own portal
    and any task scheduled during a request dies when that request returns.
    """
    settings = Settings(
        app_env="test",
        collections_dir=collections_dir,
        store_mode="memory",
        store_path=collections_dir.parent / "unused-store.json",
        # General API tests exercise curation contracts, not the optional
        # 220 MB local embedding runtime. Hybrid behavior has a dedicated
        # deterministic suite in test_hybrid_rag.py.
        rag_mode="bm25",
        deepseek_api_key=None,
        deepseek_model="deepseek-v4-flash",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_timeout_seconds=1,
        admin_review_token="test-admin-token",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture()
def agenda_payload() -> dict[str, object]:
    return {
        "question": "山水画如何组织观看者的行旅视线？",
        "priorKnowledge": "略有了解",
        # The reviewed fixture card freezes five starter objects, matching the
        # five-minute profile. Longer visits must acquire additional audited
        # objects instead of silently returning too few exhibits.
        "durationMinutes": 5,
        "personalConnection": "我曾在博物馆看过山水长卷",
        "excludedTopics": []
    }
