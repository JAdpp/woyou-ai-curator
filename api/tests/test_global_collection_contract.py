from __future__ import annotations

from api.app.collections import normalize_object
from scripts.import_global_collections import _dedupe_json_objects, _route_raw_object


def _raw(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "test:1",
        "institution": "Test Museum",
        "institutionId": "test",
        "sourceId": "1",
        "accessionNumber": "1",
        "title": "Blue export bowl",
        "date": "1700",
        "dateEarliest": 1700,
        "dateLatest": 1700,
        "medium": "Blue-and-white porcelain",
        "type": "Bowl",
        "classification": "Ceramics",
        "department": "Chinese Art",
        "culture": "China",
        "imageUrl": "https://example.org/image.jpg",
        "objectUrl": "https://example.org/object/1",
        "rights": "CC0",
        "evidence": [
            {
                "id": "test:1:e1",
                "text": "Institution record",
                "sourceUrl": "https://example.org/object/1",
                "sourceTitle": "Record",
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_v3_routing_fields_load_and_tags_never_become_domains() -> None:
    raw = _raw(
        evidenceDomainIds=["global:making-material"],
        culturePackIds=["east_asia"],
        relationFacets=["relation:trade"],
        themes=["legacy:wrong-copy"],
        tags=["Flowers"],
    )
    obj = normalize_object(raw)
    assert obj is not None
    assert obj.routing_domain_ids == ["global:making-material"]
    assert obj.themes == ["global:making-material"]
    assert obj.culture_pack_ids == ["east_asia"]
    assert obj.relation_facets == ["relation:trade"]
    assert obj.date_earliest == 1700
    assert obj.department == "Chinese Art"

    tags_only = normalize_object(_raw(tags=["Flowers"], themes=[]))
    assert tags_only is not None
    assert tags_only.routing_domain_ids == []


def test_global_router_keeps_unmatched_qualified_objects() -> None:
    routed = _route_raw_object(
        _raw(
            title="Untitled",
            medium="Unknown medium",
            type="Object",
            classification="Unclassified",
            department="Unknown",
            culture="",
        )
    )
    assert routed["culturePackIds"] == []
    assert routed["evidenceDomainIds"] == []
    assert routed["themes"] == []
    assert routed["id"] == "test:1"


def test_loader_repairs_stale_iranian_arts_of_asia_pack() -> None:
    obj = normalize_object(
        _raw(
            creator="Iran",
            culture="Iran",
            place="Iran",
            department="Arts of Asia",
            culturePackIds=["east_asia"],
        )
    )

    assert obj is not None
    assert obj.culture_pack_ids == ["west_asia_north_africa"]


def test_dedupe_uses_source_identity_not_generic_title() -> None:
    first = _raw(id="test:1", sourceId="1", title="Bowl", date="", creator="")
    second = _raw(id="test:2", sourceId="2", title="Bowl", date="", creator="")
    repeated = dict(first)
    assert [raw["id"] for raw in _dedupe_json_objects([first, second, repeated])] == [
        "test:1",
        "test:2",
    ]
