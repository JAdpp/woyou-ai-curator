from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.structured_filters as structured_filters
from app.retrieval_agent import (
    FilterSpec,
    parse_filter_spec,
    parse_query_plan,
    query_plan_prompt,
)
from app.structured_filters import (
    EVIDENCE_SEARCH_FTS5,
    EVIDENCE_SEARCH_UNAVAILABLE,
    FILTER_INDEX_FORMAT_VERSION,
    OBJECT_SEARCH_FTS5,
    OBJECT_SEARCH_UNAVAILABLE,
    StructuredFilterError,
    StructuredFilterIndex,
    build_structured_filter_index,
    collection_filter_identity,
)


def _object(
    object_id: str,
    *,
    dates: tuple[int | None, int | None],
    culture: str,
    culture_pack: str,
    institution: str,
    institution_id: str,
    material: str,
    object_type: str,
    image: bool,
    rights: str,
    evidence_depth: str,
    evidence_text: str,
    title: str,
    description: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=object_id,
        title=title,
        title_original=None,
        date_earliest=dates[0],
        date_latest=dates[1],
        date=f"{dates[0]} to {dates[1]}",
        culture=culture,
        culture_display=culture,
        culture_pack_ids=[culture_pack],
        routing_domain_ids=[culture_pack],
        institution=institution,
        institution_id=institution_id,
        creator="Unknown maker",
        material=material,
        medium=material,
        place=culture,
        type=object_type,
        classification=object_type,
        department="Collection department",
        tags=[],
        description=description,
        image_url=f"https://images.test/{object_id}.jpg" if image else "",
        rights=rights,
        rights_uri="https://creativecommons.org/publicdomain/zero/1.0/"
        if rights == "CC0"
        else "https://creativecommons.org/licenses/by/4.0/",
        image_license=rights,
        image_rights_uri=None,
        evidence_depth=evidence_depth,
        evidence=[
            SimpleNamespace(
                id=f"{object_id}-evidence",
                text=evidence_text,
                supports="catalogue description",
                source_kind="institution_curatorial_text",
            )
        ],
    )


@pytest.fixture()
def filter_collection() -> SimpleNamespace:
    objects = [
        _object(
            "a-bronze",
            dates=(-250, -100),
            culture="China",
            culture_pack="east_asia",
            institution="Cleveland Museum of Art",
            institution_id="cma",
            material="Bronze; gold",
            object_type="Vessel",
            image=True,
            rights="CC0",
            evidence_depth="full",
            evidence_text=(
                "This ancient bronze ritual vessel was cast for ceremonial offerings."
            ),
            title="Ritual Bronze Vessel",
            description="An ancient vessel used in ceremonies.",
        ),
        _object(
            "b-print",
            dates=(1780, 1810),
            culture="Japan",
            culture_pack="east_asia",
            institution="Art Institute of Chicago",
            institution_id="aic",
            material="Ink on paper",
            object_type="Print",
            image=True,
            rights="CC0",
            evidence_depth="full",
            evidence_text=(
                "This Japanese woodblock print uses ink and colour on paper."
            ),
            title="Woodblock Print",
            description="A comparative study mentions a ritual vessel in its notes.",
        ),
        _object(
            "c-sculpture",
            dates=(-700, -600),
            culture="Egypt",
            culture_pack="west_asia_north_africa",
            institution="The Metropolitan Museum of Art",
            institution_id="met",
            material="Limestone",
            object_type="Sculpture",
            image=False,
            rights="CC BY 4.0",
            evidence_depth="thin",
            evidence_text=(
                "This Egyptian limestone sculpture represents a standing official."
            ),
            title="Standing Official",
            description="A limestone figure from Egypt.",
        ),
        _object(
            "d-ceramic",
            dates=(1650, 1700),
            culture="Netherlands",
            culture_pack="europe",
            institution="Rijksmuseum",
            institution_id="rijksmuseum",
            material="Glazed ceramic",
            object_type="Plate",
            image=True,
            rights="Public Domain",
            evidence_depth="thin",
            evidence_text=(
                "This Dutch glazed ceramic plate was made for domestic dining."
            ),
            title="Glazed Plate",
            description="A ceramic plate for a dining table.",
        ),
    ]
    return SimpleNamespace(
        id="fixture-global",
        version="fixture-v1",
        objects_sha256="a" * 64,
        objects=objects,
    )


def _plan_output(**updates: object) -> dict[str, object]:
    output: dict[str, object] = {
        "inCollectionScope": True,
        "queryInterpretation": "explicit constrained request",
        "catalogueQueries": ["bronze vessel"],
        "semanticQuery": "ancient Chinese bronze vessel",
        "mandatoryPerObjectPredicates": [],
        "poolCoverageLegs": [],
        "selectionRationaleConstraints": [],
        "hardFilters": {},
        "reason": "visitor explicitly constrained the request",
    }
    output.update(updates)
    return output


def test_object_and_evidence_fts_segment_chinese_like_runtime_bm25(
    tmp_path: Path,
) -> None:
    chinese = _object(
        "chinese-landscape",
        dates=(1600, 1700),
        culture="China",
        culture_pack="east_asia",
        institution="Fixture Museum",
        institution_id="fixture",
        material="Ink on paper",
        object_type="Painting",
        image=True,
        rights="CC0",
        evidence_depth="full",
        evidence_text="此件雲山圖描繪山間雲氣。",
        title="雲山圖",
        description="一件雲山題材的水墨畫。",
    )
    collection = SimpleNamespace(
        id="fixture-cjk",
        version="fixture-v1",
        objects_sha256="b" * 64,
        objects=[chinese],
    )
    build_structured_filter_index(collection, index_root=tmp_path)
    index = StructuredFilterIndex.for_collection(tmp_path, collection)

    assert [hit.object_id for hit in index.search_objects(["雲山"])] == [
        "chinese-landscape"
    ]
    evidence_hits = index.search_evidence(["雲山"], top_k=10)
    assert [hit.object_id for hit in evidence_hits] == ["chinese-landscape"]


def test_query_plan_parses_only_typed_allowlisted_filters() -> None:
    plan = parse_query_plan(
        _plan_output(
            hardFilters={
                "dateStart": -300,
                "dateEnd": 100,
                "cultures": ["China", "china"],
                "institutions": ["CMA"],
                "materials": ["Bronze"],
                "objectTypes": ["Vessel"],
                "imageRequired": True,
                "rightsAllowed": ["CC0"],
                "evidenceDepth": ["FULL"],
            }
        ),
        question="请看公元前300年至公元100年的中国青铜器",
        max_queries=5,
    )

    assert plan.valid is True
    assert plan.filters == FilterSpec(
        date_start=-300,
        date_end=100,
        cultures=("China",),
        institutions=("CMA",),
        materials=("Bronze",),
        image_required=True,
        rights_allowed=("CC0",),
        evidence_depth=("full",),
    )
    # User/model object language is an audited kind condition, not a literal
    # institution classification filter. Other typed fields stay intact.
    assert plan.catalogue_type_hints == ("Vessel",)


@pytest.mark.parametrize(
    "filters",
    [
        {"where": "1=1"},
        {"dateStart": 1800, "dateEnd": 1700},
        {"dateStart": "1800"},
        {"imageRequired": "yes"},
        {"cultures": "China"},
        {"evidenceDepth": ["rich"]},
    ],
)
def test_invalid_or_non_allowlisted_filter_contract_fails_closed(
    filters: dict[str, object],
) -> None:
    plan = parse_query_plan(
        _plan_output(hardFilters=filters),
        question="find objects",
        max_queries=5,
    )
    assert plan.valid is False
    assert plan.search_queries == ()
    assert plan.filters.empty is True


def test_legacy_or_out_of_scope_plans_cannot_apply_filters() -> None:
    legacy = parse_query_plan(
        _plan_output(),
        question="find bronze",
        max_queries=5,
    )
    assert legacy.valid is True
    assert legacy.filters.empty is True

    out_of_scope = parse_query_plan(
        _plan_output(
            inCollectionScope=False,
            catalogueQueries=[],
            hardFilters={"cultures": ["China"]},
        ),
        question="update my phone",
        max_queries=5,
    )
    assert out_of_scope.valid is True
    assert out_of_scope.filters.empty is True


def test_prompt_declares_filter_whitelist_and_forbids_sql() -> None:
    prompt = query_plan_prompt("zh")
    for key in (
        "dateStart",
        "dateEnd",
        "cultures",
        "institutions",
        "materials",
        "objectTypes",
        "imageRequired",
        "rightsAllowed",
        "evidenceDepth",
    ):
        assert key in prompt
    assert "Never emit SQL" in prompt
    assert "states explicitly" in prompt


def test_filter_spec_parser_accepts_empty_block_and_rejects_unknown_key() -> None:
    assert parse_filter_spec(None) == FilterSpec()
    assert parse_filter_spec({}) == FilterSpec()
    assert parse_filter_spec({"sql": "DROP TABLE objects"}) is None


def test_sqlite_filter_index_applies_or_within_facets_and_and_across_fields(
    tmp_path: Path,
    filter_collection: SimpleNamespace,
) -> None:
    output = build_structured_filter_index(
        filter_collection,
        index_root=tmp_path / "filters",
    )
    index = StructuredFilterIndex(output, filter_collection)

    assert index.filter_object_ids(FilterSpec()) == [
        "a-bronze",
        "b-print",
        "c-sculpture",
        "d-ceramic",
    ]
    assert index.filter_object_ids(
        FilterSpec(cultures=("China", "Egypt"))
    ) == ["a-bronze", "c-sculpture"]
    assert index.filter_object_ids(
        FilterSpec(
            cultures=("China", "Egypt"),
            materials=("bronze",),
            image_required=True,
            rights_allowed=("CC0",),
            evidence_depth=("full",),
        )
    ) == ["a-bronze"]
    assert index.filter_object_ids(
        FilterSpec(date_start=-300, date_end=100)
    ) == ["a-bronze"]
    assert index.filter_object_ids(
        FilterSpec(institutions=("aic",), object_types=("Print",))
    ) == ["b-print"]
    assert index.filter_object_ids(
        FilterSpec(image_required=False)
    ) == index.filter_object_ids(FilterSpec())
    assert index.filter_object_ids(
        FilterSpec(), allowed_object_ids={"b-print", "d-ceramic"}
    ) == ["b-print", "d-ceramic"]


def test_filter_values_are_bound_parameters_not_sql(
    tmp_path: Path,
    filter_collection: SimpleNamespace,
) -> None:
    output = build_structured_filter_index(
        filter_collection,
        index_root=tmp_path / "filters",
    )
    index = StructuredFilterIndex(output, filter_collection)
    attack = "china') OR 1=1; DROP TABLE objects; --"

    assert index.filter_object_ids(FilterSpec(cultures=(attack,))) == []
    assert index.filter_object_ids(FilterSpec()) == [
        "a-bronze",
        "b-print",
        "c-sculpture",
        "d-ceramic",
    ]


def test_composite_material_components_and_bilingual_culture_filters(
    tmp_path: Path, filter_collection: SimpleNamespace,
) -> None:
    output = build_structured_filter_index(filter_collection, index_root=tmp_path)
    index = StructuredFilterIndex(output, filter_collection)
    assert index.filter_object_ids(FilterSpec(materials=("paper",), cultures=("东亚",))) == ["b-print"]
    assert index.filter_object_ids(FilterSpec(materials=("ceramic",), cultures=("European",))) == ["d-ceramic"]
    assert index.filter_object_ids(FilterSpec(materials=("bronze",), cultures=("Chinese",))) == ["a-bronze"]
    # Literal phrases cannot turn into wildcard or substring matches.
    assert index.filter_object_ids(FilterSpec(materials=("stone",))) == []
    assert index.filter_object_ids(FilterSpec(materials=("%",))) == []
    assert index.filter_object_ids(FilterSpec(materials=("paper') OR 1=1 --",))) == []


def test_specific_country_filter_does_not_expand_to_entire_region(
    tmp_path: Path, filter_collection: SimpleNamespace,
) -> None:
    output = build_structured_filter_index(filter_collection, index_root=tmp_path)
    index = StructuredFilterIndex(output, filter_collection)
    assert index.filter_object_ids(FilterSpec(cultures=("中国",))) == ["a-bronze"]
    assert index.filter_object_ids(FilterSpec(cultures=("East Asia",))) == ["a-bronze", "b-print"]
    assert index.filter_object_ids(FilterSpec(cultures=("东南亚",))) == []


def test_manifest_and_database_are_bound_to_frozen_collection(
    tmp_path: Path,
    filter_collection: SimpleNamespace,
) -> None:
    output = build_structured_filter_index(
        filter_collection,
        index_root=tmp_path / "filters",
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    identity = collection_filter_identity(filter_collection)
    assert manifest["formatVersion"] == FILTER_INDEX_FORMAT_VERSION
    assert manifest["collectionId"] == filter_collection.id
    assert manifest["collectionVersion"] == filter_collection.version
    assert manifest["objectsSha256"] == filter_collection.objects_sha256
    assert manifest["fingerprint"] == identity.fingerprint
    assert manifest["evidenceCount"] == 4
    assert manifest["objectSearchMode"] == OBJECT_SEARCH_FTS5
    assert manifest["evidenceSearchMode"] == EVIDENCE_SEARCH_FTS5

    changed = SimpleNamespace(
        id=filter_collection.id,
        version="fixture-v2",
        objects_sha256=filter_collection.objects_sha256,
        objects=filter_collection.objects,
    )
    with pytest.raises(StructuredFilterError, match="does not match"):
        StructuredFilterIndex(output, changed)

    uri = f"{(output / 'filters.sqlite3').resolve().as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM objects")


def test_database_digest_detects_tampering(
    tmp_path: Path,
    filter_collection: SimpleNamespace,
) -> None:
    output = build_structured_filter_index(
        filter_collection,
        index_root=tmp_path / "filters",
    )
    database = output / "filters.sqlite3"
    with database.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(StructuredFilterError, match="digest mismatch"):
        StructuredFilterIndex(output, filter_collection)


def test_evidence_level_bm25_search_and_candidate_intersection(
    tmp_path: Path,
    filter_collection: SimpleNamespace,
) -> None:
    output = build_structured_filter_index(
        filter_collection,
        index_root=tmp_path / "filters",
    )
    index = StructuredFilterIndex(output, filter_collection)

    assert index.evidence_search_available is True
    hits = index.search_evidence("bronze ritual vessel", top_k=3)
    assert [(hit.object_id, hit.evidence_id) for hit in hits] == [
        ("a-bronze", "a-bronze-evidence")
    ]
    assert hits[0].score > 0
    assert hits[0].source_kind == "institution_curatorial_text"
    assert index.search_evidence(
        "bronze ritual vessel",
        allowed_object_ids={"b-print", "d-ceramic"},
    ) == []
    assert index.search_evidence(
        "ink paper",
        allowed_object_ids={"b-print", "c-sculpture"},
    )[0].object_id == "b-print"


def test_object_level_bm25_uses_weighted_fields_and_candidate_intersection(
    tmp_path: Path,
    filter_collection: SimpleNamespace,
) -> None:
    output = build_structured_filter_index(
        filter_collection,
        index_root=tmp_path / "filters",
    )
    index = StructuredFilterIndex(output, filter_collection)

    assert index.object_search_available is True
    hits = index.search_objects("ritual vessel", top_k=4)
    assert [hit.object_id for hit in hits[:2]] == ["a-bronze", "b-print"]
    assert hits[0].score > hits[1].score
    assert index.search_objects(
        "ritual vessel",
        allowed_object_ids={"b-print", "c-sculpture"},
    )[0].object_id == "b-print"
    assert index.search_objects("ritual vessel", allowed_object_ids=set()) == []


def test_object_search_treats_fts_syntax_as_literal_tokens(
    tmp_path: Path,
    filter_collection: SimpleNamespace,
) -> None:
    output = build_structured_filter_index(
        filter_collection,
        index_root=tmp_path / "filters",
    )
    index = StructuredFilterIndex(output, filter_collection)

    hits = index.search_objects(
        'bronze" OR *; DROP TABLE object_docs; --',
        top_k=4,
    )
    assert hits[0].object_id == "a-bronze"
    assert index.search_objects("Egyptian limestone")[0].object_id == "c-sculpture"


def test_evidence_search_treats_fts_syntax_as_tokens_not_code(
    tmp_path: Path,
    filter_collection: SimpleNamespace,
) -> None:
    output = build_structured_filter_index(
        filter_collection,
        index_root=tmp_path / "filters",
    )
    index = StructuredFilterIndex(output, filter_collection)

    hits = index.search_evidence(
        'bronze" OR *; DROP TABLE evidence_docs; --',
        top_k=4,
    )
    assert hits[0].object_id == "a-bronze"
    assert index.search_evidence("Egyptian limestone")[0].object_id == "c-sculpture"


def test_evidence_search_unavailable_is_explicit_not_silent(
    tmp_path: Path,
    filter_collection: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(structured_filters, "_create_evidence_fts", lambda _db: False)
    output = build_structured_filter_index(
        filter_collection,
        index_root=tmp_path / "filters",
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    index = StructuredFilterIndex(output, filter_collection)

    assert manifest["evidenceSearchMode"] == EVIDENCE_SEARCH_UNAVAILABLE
    assert index.evidence_search_available is False
    with pytest.raises(StructuredFilterError, match="FTS5 was not built"):
        index.search_evidence("bronze")


def test_object_search_unavailable_is_explicit_not_silent(
    tmp_path: Path,
    filter_collection: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(structured_filters, "_create_object_fts", lambda _db: False)
    output = build_structured_filter_index(
        filter_collection,
        index_root=tmp_path / "filters",
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    index = StructuredFilterIndex(output, filter_collection)

    assert manifest["objectSearchMode"] == OBJECT_SEARCH_UNAVAILABLE
    assert index.object_search_available is False
    with pytest.raises(StructuredFilterError, match="FTS5 was not built"):
        index.search_objects("bronze")


@pytest.mark.parametrize("top_k", [0, 501, True, 1.5])
def test_evidence_search_rejects_invalid_limits(
    tmp_path: Path,
    filter_collection: SimpleNamespace,
    top_k: object,
) -> None:
    output = build_structured_filter_index(
        filter_collection,
        index_root=tmp_path / "filters",
    )
    index = StructuredFilterIndex(output, filter_collection)

    with pytest.raises(StructuredFilterError, match="top_k"):
        index.search_evidence("bronze", top_k=top_k)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, 501, True, 1.5])
def test_object_search_rejects_invalid_limits(
    tmp_path: Path,
    filter_collection: SimpleNamespace,
    top_k: object,
) -> None:
    output = build_structured_filter_index(
        filter_collection,
        index_root=tmp_path / "filters",
    )
    index = StructuredFilterIndex(output, filter_collection)

    with pytest.raises(StructuredFilterError, match="top_k"):
        index.search_objects("bronze", top_k=top_k)  # type: ignore[arg-type]


def test_manifest_rejects_tampered_object_search_mode(
    tmp_path: Path,
    filter_collection: SimpleNamespace,
) -> None:
    output = build_structured_filter_index(
        filter_collection,
        index_root=tmp_path / "filters",
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["objectSearchMode"] = "pretend-fast-search"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(StructuredFilterError, match="object search mode is invalid"):
        StructuredFilterIndex(output, filter_collection)
