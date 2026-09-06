"""General audit-window and natural-language material regression tests."""
from app.collections import SearchResult
from app.generator import ExhibitionGenerator, CulturalCoverageObligation
from app.models import MuseumObject, EvidenceChunk
from app.retrieval_agent import RetrievalQueryPlan
from app.retrieval_filters import FilterSpec


def result(key, culture, *, rich=False):
    return SearchResult(obj=MuseumObject(
        id=key, title=key, culture=culture, imageUrl="https://example.org/image",
        objectUrl="https://example.org/object", rights="CC0",
        evidence=[EvidenceChunk(id=f"{key}:text", text=(
            "The institution describes the construction and use of this woven container."
            if rich else "Container"), sourceUrl="https://example.org/object",
            sourceTitle=key, sourceKind=("institution_curatorial_text" if rich else "institution_metadata"))],
    ), score=20, evidence_score=0.7 if rich else None)


def test_each_named_region_can_expose_both_top_rank_and_source_rich_object():
    candidates = [result(f"a{i}", "United States") for i in range(20)]
    candidates += [result("b-thin", "Philippines"), result("b-rich", "Philippines", rich=True)]
    obligations = ExhibitionGenerator._cultural_coverage_obligations("美洲、非洲、东南亚的编织容器")
    sampled = ExhibitionGenerator._audit_candidate_sample(
        candidates, top_k=18, cross_cultural=True, cultural_obligations=obligations,
    )
    assert "b-rich" in {item.obj.id for item in sampled}
    assert len(sampled) <= 18
    assert len({item.obj.id for item in sampled}) == len(sampled)


def test_material_superclass_remains_admission_condition_without_literal_recall_filter():
    filters = FilterSpec(materials=("plant fibre",), date_start=1700, cultures=("Japan",))
    plan = RetrievalQueryPlan(valid=True, in_collection_scope=True, search_queries=("woven container",), filters=filters)
    recalled = ExhibitionGenerator._recall_filters(plan)
    assert recalled.materials == ()
    assert recalled.date_start == 1700 and recalled.cultures == ("Japan",)
    assert plan.filters == filters and filters.materials == ("plant fibre",)
