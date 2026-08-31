from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.collections import SearchResult
from app.generator import ExhibitionGenerator
from app.models import EvidenceChunk, MuseumObject, VisitorProfile


def _result(object_id: str, title: str, score: float) -> SearchResult:
    obj = MuseumObject(
        id=object_id,
        sourceId=object_id,
        title=title,
        imageUrl=f"https://example.test/{object_id}.jpg",
        objectUrl=f"https://example.test/{object_id}",
        rights="CC0",
        institution="Fixture Museum",
        institutionId="fixture",
        culture="Test culture",
        material="ink on paper",
        evidence=[
            EvidenceChunk(
                id=f"{object_id}:metadata",
                text=f"Institution record for {title}.",
                sourceUrl=f"https://example.test/{object_id}",
                sourceTitle=title,
                sourceKind="institution_metadata",
            )
        ],
    )
    return SearchResult(obj=obj, score=score, retrieval_sources=("bm25",))


def test_selection_uses_other_audited_objects_before_a_third_title_variant() -> None:
    results = [
        _result("bamboo-1", "Bamboo in the Wind", 100),
        _result("bamboo-2", "Bamboos in Wind", 99),
        _result("bamboo-3", "Bamboo in Wind", 98),
        _result("bamboo-4", "The Bamboos in the Wind", 97),
        _result("beach", "Wind on the Beach", 93),
        _result("sail", "Billowing Sail", 92),
        _result("tree", "Windblown Tree", 91),
        _result("cloud", "Storm Clouds", 90),
    ]

    selected = ExhibitionGenerator._diverse_selection(results, 5)
    selected_ids = {obj.id for obj in selected}

    assert len(selected_ids & {"bamboo-1", "bamboo-2", "bamboo-3", "bamboo-4"}) <= 2
    assert len(selected) == 5


def test_recharger_affective_request_uses_browse_route_without_claiming_therapy(
    client: TestClient,
) -> None:
    repository = client.app.state.collections
    collection = repository.get()
    question = (
        "我说不出想看什么，就是最近有点累。"
        "能不能挑几件看着不会被催着走的东西？"
    )
    profile = VisitorProfile(
        curiosityLabel="放慢一点",
        freeFormQuestion=question,
        motivation="recharger",
        durationMinutes=5,
    )
    generator = ExhibitionGenerator(client.app.state.settings, repository)

    exhibition = asyncio.run(
        generator.generate_from_profile(profile, collection_id=collection.id)
    )

    assert exhibition.question == question
    assert len(exhibition.items) == 5
    assert any(
        "不声称任何藏品具有疗愈性"
        in limit
        for limit in exhibition.coverage_limits
    )
    assert exhibition.validation.passed is True
