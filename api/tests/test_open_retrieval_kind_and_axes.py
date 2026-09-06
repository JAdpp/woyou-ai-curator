from app.collections import SearchResult
from app.generator import ExhibitionGenerator
from app.models import MuseumObject
from app.retrieval_agent import audit_payload, parse_filter_spec, parse_query_plan


def result(key):
    return SearchResult(obj=MuseumObject(
        id=key, sourceId=key, title=key, institution="Fixture", institutionId="fixture",
        rights="CC0", imageUrl="https://example.test/image.jpg",
        objectUrl="https://example.test/object", evidence=[],
    ), score=10.0)


def test_natural_language_kind_is_not_an_institution_type_exclusion():
    plan = parse_query_plan({
        "inCollectionScope": True, "catalogueQueries": ["cast bell"],
        "hardFilters": {"objectTypes": ["bell"], "materials": ["gold"],
                        "cultures": ["americas"], "dateStart": 800, "dateEnd": 1400},
    }, question="美洲金铃", max_queries=5)
    assert plan.valid
    assert plan.catalogue_type_hints == ("bell",)
    assert plan.filters.object_types == ()
    assert plan.filters.materials == ("gold",)
    assert plan.filters.cultures == ("americas",)
    assert (plan.filters.date_start, plan.filters.date_end) == (800, 1400)


def test_type_hints_reach_auditor_with_original_geography():
    candidate = result("object-a")
    candidate.obj.culture_pack_ids = ["europe"]
    payload = audit_payload("a kind", [candidate], required_count=5, top_k=18,
                            pass_number=1, catalogue_type_hints=("woven container",))
    assert payload["retrievalContract"]["catalogueTypeHints"] == ["woven container"]
    assert payload["candidates"][0]["catalogueCulturePacks"] == ["europe"]


def test_explicit_typed_filter_interface_still_keeps_object_types():
    filters = parse_filter_spec({"objectTypes": ["Metalwork"]})
    assert filters.object_types == ("Metalwork",)


def test_kind_hint_sources_are_deduplicated_without_losing_alternatives():
    plan = parse_query_plan({"inCollectionScope": True,
                             "catalogueQueries": ["basket"],
                             "catalogueTypeHints": ["basket", "tray"],
                             "hardFilters": {"objectTypes": ["basket"]}},
                            question="basket or tray", max_queries=5)
    assert plan.catalogue_type_hints == ("basket", "tray")


def test_axis_distinctive_case_survives_generic_shared_heads():
    generic = [result(f"generic-{index}") for index in range(25)]
    unique = [result(f"axis-{index}") for index in range(3)]
    candidates = generic + unique
    batches = [generic[:12] + [item] + generic[12:] for item in unique]
    selected = ExhibitionGenerator._audit_candidate_sample(
        candidates, top_k=18, cross_cultural=False, query_batches=batches,
    )
    ids = [item.obj.id for item in selected]
    assert all(item.obj.id in ids for item in unique)
    assert len(ids) == len(set(ids)) == 18
    assert ids[0] == generic[0].obj.id


def test_without_axes_rank_is_preserved_and_pool_is_bounded():
    candidates = [result(f"item-{index}") for index in range(30)]
    selected = ExhibitionGenerator._audit_candidate_sample(
        candidates, top_k=18, cross_cultural=False,
    )
    assert selected == candidates[:18]
