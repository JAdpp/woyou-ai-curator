from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from scripts.global_taxonomy import (
    CULTURE_PACKS,
    GLOBAL_DOMAINS,
    ROUTING_VERSION,
    TAXONOMY_VERSION,
    assign_culture_packs,
    assign_evidence_domains,
    assign_global_domains,
    assign_relation_facets,
    assign_taxonomy,
    expand_concept_aliases,
    matches_pattern,
    taxonomy_manifest,
)


@dataclass
class ObjectStub:
    title: str = ""
    type: str = ""
    classification: str = ""
    medium: str = ""
    culture: str = ""
    place: str = ""
    department: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)


def test_registry_has_ten_unique_namespaced_global_domains() -> None:
    ids = [domain.id for domain in GLOBAL_DOMAINS]
    assert len(ids) == 10
    assert len(ids) == len(set(ids))
    assert all(domain_id.startswith("global:") for domain_id in ids)
    assert len({pack.id for pack in CULTURE_PACKS}) == len(CULTURE_PACKS)
    assert ROUTING_VERSION == TAXONOMY_VERSION


@pytest.mark.parametrize(
    ("metadata", "expected_pack"),
    [
        ({"culture": "China, Qing dynasty"}, "east_asia"),
        ({"culture": "India, Rajasthan"}, "south_asia"),
        ({"culture": "Khmer", "place": "Cambodia"}, "southeast_asia"),
        ({"department": "Egyptian Art", "culture": "Egyptian"}, "west_asia_north_africa"),
        ({"culture": "Dutch", "place": "Netherlands"}, "europe"),
        ({"culture": "Yoruba", "place": "Nigeria"}, "africa"),
        ({"culture": "Maya", "place": "Mexico"}, "americas"),
        ({"culture": "Maori", "place": "Aotearoa, New Zealand"}, "oceania"),
    ],
)
def test_culture_pack_assignment_covers_initial_global_regions(
    metadata: dict[str, str], expected_pack: str
) -> None:
    assert expected_pack in assign_culture_packs(metadata)


def test_broad_arts_of_asia_department_does_not_override_iranian_origin() -> None:
    """AIC's broad department name is not an East-Asia provenance claim."""

    obj = {
        "department": "Arts of Asia",
        "culture": "Iran",
        "place": "Iran",
        "creator": "Iran",
        "date": "Safavid dynasty (1501–1722), 17th century",
    }

    assert assign_culture_packs(obj) == ["west_asia_north_africa"]


def test_ascii_matching_uses_real_word_boundaries() -> None:
    assert matches_pattern("Wine vessel used at court", "vessel")
    assert matches_pattern("BLUE-AND-WHITE PORCELAIN", "blue-and-white")
    assert not matches_pattern("Portrait of a young woman", "you")
    assert not matches_pattern("Stoneware bowl", "war")


def test_assignment_is_deterministic_for_mapping_and_object_input() -> None:
    mapping = {
        "title": "Export porcelain bowl with inscription",
        "classification": "Ceramics",
        "medium": "Blue-and-white porcelain with cobalt underglaze",
        "culture": "China",
        "description": "Made for export through maritime trade.",
    }
    object_input = ObjectStub(**mapping)
    assert assign_taxonomy(mapping) == assign_taxonomy(object_input)
    assert assign_taxonomy(mapping) == assign_taxonomy(mapping)


@pytest.mark.parametrize(
    ("obj", "expected_domains"),
    [
        (
            ObjectStub(type="Sarcophagus", culture="Egyptian", description="Buried in a tomb for the afterlife."),
            {"global:death-afterlife"},
        ),
        (
            ObjectStub(classification="Calligraphy", medium="Ink on paper", tags=["Inscription"]),
            {"global:text-memory", "global:making-material"},
        ),
        (
            ObjectStub(type="Export bowl", medium="Blue-and-white porcelain", description="Made for export trade."),
            {"global:making-material", "global:exchange-mobility"},
        ),
        (
            ObjectStub(type="Portrait", medium="Oil on canvas", tags=["Royal", "Robe"]),
            {"global:body-identity", "global:power-status"},
        ),
        (
            ObjectStub(title="Mountain landscape", classification="Painting", tags=["River"]),
            {"global:nature-place"},
        ),
    ],
)
def test_global_domain_assignment_maps_cross_cultural_metadata(
    obj: ObjectStub, expected_domains: set[str]
) -> None:
    assert expected_domains <= set(assign_global_domains(obj))


def test_unmatched_object_is_retained_as_unassigned_by_caller() -> None:
    obj = ObjectStub(title="Untitled", type="Object", classification="Unclassified")
    assert assign_culture_packs(obj) == []
    assert assign_global_domains(obj) == []
    assert assign_taxonomy(obj) == {
        "culturePackIds": [],
        "evidenceDomainIds": [],
        "relationFacets": [],
    }


def test_stable_evidence_domain_interface_accepts_pack_ids() -> None:
    obj = ObjectStub(medium="Blue-and-white porcelain", culture="China")
    assert assign_evidence_domains(obj, ["east_asia"]) == assign_global_domains(obj)


def test_relation_facets_only_follow_explicit_metadata_signals() -> None:
    obj = ObjectStub(
        type="Export vessel",
        description="Made for export and presented as a diplomatic gift.",
    )
    assert set(assign_relation_facets(obj)) == {
        "relation:trade",
        "relation:diplomatic-gift",
    }
    assert assign_relation_facets(ObjectStub(title="Two visually similar bowls")) == []


def test_chinese_concepts_expand_to_global_catalogue_vocabulary() -> None:
    animal_power = set(expand_concept_aliases("不同文明为什么都用动物表现权力？"))
    assert {"animal", "power", "royal", "imperial"} <= animal_power

    blue_ceramics = set(expand_concept_aliases("蓝色如何连接波斯陶瓷、中国青花与代尔夫特？"))
    assert {"blue", "cobalt", "ceramic", "porcelain", "persian", "blue-and-white"} <= blue_ceramics


def test_manifest_helpers_are_stable_json_ready_and_apply_both_thresholds() -> None:
    payload = taxonomy_manifest(
        {"east_asia": 250, "europe": 199},
        {"global:making-material": 50, "global:text-memory": 39},
        culture_pack_full_evidence={"east_asia": 21, "europe": 50},
        global_domain_full_evidence={"global:making-material": 6, "global:text-memory": 20},
    )

    assert payload["taxonomyVersion"] == TAXONOMY_VERSION
    assert payload["routingVersion"] == ROUTING_VERSION
    assert [item["id"] for item in payload["culturePacks"]] == [
        pack.id for pack in CULTURE_PACKS
    ]
    assert [item["id"] for item in payload["evidenceDomains"]] == [
        domain.id for domain in GLOBAL_DOMAINS
    ]

    packs = {item["id"]: item for item in payload["culturePacks"]}
    assert packs["east_asia"]["meetsMinimum"] is True
    assert packs["europe"]["meetsMinimum"] is False

    domains = {item["id"]: item for item in payload["evidenceDomains"]}
    assert domains["global:making-material"]["meetsMinimum"] is True
    assert domains["global:text-memory"]["meetsMinimum"] is False
    assert all(item["scope"] == "global" for item in payload["evidenceDomains"])
    assert isinstance(payload["conceptAliases"]["动物"], list)
    assert len(payload["boundaries"]) == 3
