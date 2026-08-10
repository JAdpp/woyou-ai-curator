from __future__ import annotations

import json

import pytest

from api.app.collections import normalize_object
from scripts.import_global_collections import (
    _emit,
    _rights_audit,
    _route_raw_object,
    _validate_before_emit,
)
from scripts.sources import aic
from scripts.sources.base import (
    CC0_LICENSE,
    CC0_RIGHTS_URI,
    CC_BY_4_0_LICENSE,
    CC_BY_4_0_RIGHTS_URI,
    apply_rights_policy_to_json,
)


def _raw(institution_id: str = "cma", **overrides: object) -> dict[str, object]:
    rights = {
        "cma": "CC0 1.0 (CMA share_license_status=CC0)",
        "met": "CC0 1.0 (Met isPublicDomain=true)",
        "aic": "Public Domain (AIC is_public_domain=true)",
    }.get(institution_id, "CC0")
    institution = {
        "cma": "Cleveland Museum of Art",
        "met": "The Metropolitan Museum of Art",
        "aic": "Art Institute of Chicago",
    }.get(institution_id, "Test Museum")
    payload: dict[str, object] = {
        "id": f"{institution_id}:1",
        "institution": institution,
        "institutionId": institution_id,
        "sourceId": "1",
        "accessionNumber": "1",
        "title": "Test object",
        "date": "1700",
        "medium": "Porcelain",
        "type": "Bowl",
        "classification": "Ceramics",
        "culture": "China",
        "imageUrl": "https://example.org/image.jpg",
        "objectUrl": "https://example.org/object/1",
        "rights": rights,
        "imageValidation": {"ok": True},
        "evidence": [
            {
                "id": f"{institution_id}:1:tombstone",
                "text": "Institution record",
                "sourceUrl": "https://example.org/object/1",
                "sourceTitle": "Record",
                "kind": "tombstone",
            }
        ],
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    (
        "description_field",
        "expected_license",
        "expected_rights_uri",
        "expected_evidence_suffix",
    ),
    [
        (
            "description",
            CC_BY_4_0_LICENSE,
            CC_BY_4_0_RIGHTS_URI,
            ":description",
        ),
        (
            "short_description",
            CC0_LICENSE,
            CC0_RIGHTS_URI,
            ":short-description",
        ),
    ],
)
def test_aic_description_fields_keep_their_distinct_licence_scopes(
    description_field: str,
    expected_license: str,
    expected_rights_uri: str,
    expected_evidence_suffix: str,
) -> None:
    record: dict[str, object] = {
        "id": 123,
        "title": "Open artwork",
        "artist_display": "Artist",
        "date_display": "1700",
        "medium_display": "Ink",
        "dimensions": "10 x 20 cm",
        "place_of_origin": "China",
        "department_title": "Arts of Asia",
        "classification_title": "Painting",
        "artwork_type_title": "Painting",
        "is_public_domain": True,
        "image_id": "image-id",
        "thumbnail": {"alt_text": "A detailed institution-authored image description."},
        "term_titles": ["painting"],
        "inscriptions": "Seal and inscription metadata.",
        description_field: "<p>Institution curatorial prose.</p>",
    }

    mapped = aic._map(record, "raw/aic/123.json", "abc123")

    assert mapped is not None
    payload = mapped.to_json()
    description = next(
        chunk
        for chunk in payload["evidence"]
        if chunk["id"].endswith(expected_evidence_suffix)
    )
    tombstone = next(
        chunk for chunk in payload["evidence"] if chunk["id"].endswith(":tombstone")
    )
    inscription = next(
        chunk for chunk in payload["evidence"] if chunk["id"].endswith(":inscription")
    )
    assert payload["imageLicense"] == CC0_LICENSE
    assert payload["metadataLicense"] == CC0_LICENSE
    assert payload["curatorialTextLicense"] == expected_license
    assert payload["curatorialTextRightsUri"] == expected_rights_uri
    assert description["license"] == expected_license
    assert description["rightsUri"] == expected_rights_uri
    assert description["sourceKind"] == "institution_curatorial_text"
    assert description_field in description["sourceLocation"]
    assert tombstone["license"] == CC0_LICENSE
    assert tombstone["rightsUri"] == CC0_RIGHTS_URI
    assert tombstone["sourceKind"] == "institution_metadata"
    assert inscription["license"] == CC0_LICENSE
    assert inscription["sourceKind"] == "institution_metadata"
    _validate_before_emit([payload], allow_small=True, require_image_check=False)


def test_runtime_hydrates_frozen_aic_v3_without_widening_unknown_rights() -> None:
    legacy_aic = _raw(
        "aic",
        description="Institution description",
        evidence=[
            {
                "id": "aic:1:description",
                "text": "Institution description",
                "sourceUrl": "https://example.org/object/1",
                "sourceTitle": "Description",
                "sourceLocation": "Curatorial description",
                "kind": "curatorial_text",
            },
            {
                "id": "aic:1:tombstone",
                "text": "Institution record",
                "sourceUrl": "https://example.org/object/1",
                "sourceTitle": "Record",
                "kind": "tombstone",
            },
        ],
    )
    obj = normalize_object(legacy_aic)
    assert obj is not None
    assert obj.image_license == CC0_LICENSE
    assert obj.metadata_license == CC0_LICENSE
    assert obj.curatorial_text_license == CC_BY_4_0_LICENSE
    assert obj.evidence[0].license == CC_BY_4_0_LICENSE
    assert obj.evidence[0].rights_uri == CC_BY_4_0_RIGHTS_URI
    assert obj.evidence[1].license == CC0_LICENSE

    unknown = normalize_object(_raw("unknown", description="Unscoped prose"))
    assert unknown is not None
    assert unknown.rights == "CC0"  # legacy display remains readable
    assert unknown.image_license is None
    assert unknown.metadata_license is None
    assert unknown.curatorial_text_license is None
    assert unknown.evidence[0].license is None


def test_runtime_preserves_identified_aic_short_description_as_cc0() -> None:
    short_description = _raw(
        "aic",
        description="Short institution description",
        evidence=[
            {
                "id": "aic:1:short-description",
                "text": "Short institution description",
                "sourceUrl": "https://example.org/object/1",
                "sourceTitle": "Short description",
                "sourceLocation": "Short description (short_description field)",
                "kind": "curatorial_text",
            }
        ],
    )

    hydrated = apply_rights_policy_to_json(short_description)
    assert hydrated["curatorialTextLicense"] == CC0_LICENSE
    assert hydrated["curatorialTextRightsUri"] == CC0_RIGHTS_URI
    assert hydrated["evidence"][0]["license"] == CC0_LICENSE

    obj = normalize_object(short_description)
    assert obj is not None
    assert obj.curatorial_text_license == CC0_LICENSE
    assert obj.curatorial_text_rights_uri == CC0_RIGHTS_URI
    assert obj.evidence[0].license == CC0_LICENSE
    assert obj.evidence[0].rights_uri == CC0_RIGHTS_URI


def test_cma_and_met_legacy_records_receive_only_their_official_scopes() -> None:
    cma = apply_rights_policy_to_json(
        _raw("cma", description="CMA API description")
    )
    met = apply_rights_policy_to_json(_raw("met"))

    for payload in (cma, met):
        assert payload["imageLicense"] == CC0_LICENSE
        assert payload["imageRightsUri"] == CC0_RIGHTS_URI
        assert payload["metadataLicense"] == CC0_LICENSE
        assert payload["metadataRightsUri"] == CC0_RIGHTS_URI
        assert payload["evidence"][0]["license"] == CC0_LICENSE
    assert cma["curatorialTextLicense"] == CC0_LICENSE
    assert met["curatorialTextLicense"] is None


def test_global_manifest_and_audit_describe_all_three_rights_scopes(tmp_path) -> None:
    routed = _route_raw_object(
        _raw(
            "cma",
            description="CMA description",
            evidenceDomainIds=["global:making-material"],
            culturePackIds=["east_asia"],
        )
    )
    _validate_before_emit([routed], allow_small=True, require_image_check=False)
    _emit(
        tmp_path,
        [routed],
        "test-version",
        {"cma": "test-snapshot"},
        "skipped_intermediate_only",
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rightsSchemaVersion"] == "1.0"
    assert "curatorialText" in manifest["license"]
    policies = {
        item["institutionId"]: item for item in manifest["rightsByInstitution"]
    }
    assert policies["cma"]["curatorialTextLicense"] == CC0_LICENSE
    assert policies["met"]["curatorialTextLicense"] is None
    assert policies["aic"]["curatorialTextLicense"] == CC_BY_4_0_LICENSE
    assert policies["aic"]["curatorialTextField"] == "description"
    assert policies["aic"]["shortDescriptionLicense"] == CC0_LICENSE

    audit = (tmp_path / "rights_audit.md").read_text(encoding="utf-8")
    assert "imageLicense" in audit
    assert "evidence[].license" in audit
    assert "CC BY 4.0" in audit
    assert "IIIF 图片返回 403" not in audit


def test_validation_rejects_known_source_with_collapsed_or_wrong_scope() -> None:
    routed = _route_raw_object(_raw("cma"))
    routed["imageLicense"] = CC_BY_4_0_LICENSE
    with pytest.raises(SystemExit, match="field-level rights contract"):
        _validate_before_emit([routed], allow_small=True, require_image_check=False)


def test_unknown_explicit_structured_rights_are_preserved() -> None:
    raw = _raw(
        "unknown",
        imageLicense="Institution image terms",
        imageRightsUri="https://example.org/image-terms",
        metadataLicense="Institution metadata terms",
        metadataRightsUri="https://example.org/metadata-terms",
    )
    hydrated = apply_rights_policy_to_json(raw)
    obj = normalize_object(hydrated)
    assert obj is not None
    assert obj.image_license == "Institution image terms"
    assert obj.image_rights_uri == "https://example.org/image-terms"
    assert obj.metadata_license == "Institution metadata terms"
    assert obj.metadata_rights_uri == "https://example.org/metadata-terms"


def test_rights_audit_counts_unspecified_unknown_fields_without_assigning_cc0() -> None:
    audit = _rights_audit(
        [apply_rights_policy_to_json(_raw("unknown"))],
        "v3-legacy",
        "skipped_intermediate_only",
        "test",
    )
    assert "unspecified=1" in audit
