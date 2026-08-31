from __future__ import annotations

from app.collections import SearchResult, _query_plan, question_requests_provenance
from app.models import EvidenceChunk, MuseumObject
from app.retrieval_agent import audit_payload, parse_audit


def _candidate() -> SearchResult:
    obj = MuseumObject(
        id="fixture:source-history",
        sourceId="source-history",
        title="Ceremonial object",
        imageUrl="https://example.test/object.jpg",
        objectUrl="https://example.test/object",
        rights="CC0",
        institution="Fixture Museum",
        institutionId="fixture",
        evidence=[
            EvidenceChunk(
                id="fixture:source-history:metadata",
                text="Ceremonial object, wood, nineteenth century.",
                sourceUrl="https://example.test/object",
                sourceTitle="Ceremonial object",
                sourceKind="institution_metadata",
            ),
            EvidenceChunk(
                id="fixture:source-history:provenance",
                text="Recorded as acquired from a military expedition in 1897.",
                sourceUrl="https://example.test/object",
                sourceTitle="Acquisition record",
                sourceKind="institution_provenance",
            ),
        ],
    )
    return SearchResult(
        obj=obj,
        score=10,
        matched_evidence_ids=("fixture:source-history:provenance",),
        retrieval_sources=("dense_evidence_recall",),
    )


def test_provenance_scope_is_explicit_and_query_planned() -> None:
    source_question = "这件藏品的入藏来源和流传经历是什么？"
    visual_question = "这件藏品的颜色和构图有什么特点？"

    assert question_requests_provenance(source_question) is True
    assert _query_plan(source_question).allow_provenance_evidence is True
    assert question_requests_provenance(visual_question) is False
    assert _query_plan(visual_question).allow_provenance_evidence is False


def test_audit_payload_exposes_provenance_only_for_source_history_question() -> None:
    candidate = _candidate()

    source_payload = audit_payload(
        "它在 1897 年是怎样被军队带走并入藏的？",
        [candidate],
        required_count=1,
        top_k=1,
        pass_number=1,
    )
    visual_payload = audit_payload(
        "它的形状和颜色是什么？",
        [candidate],
        required_count=1,
        top_k=1,
        pass_number=1,
    )

    assert [row["id"] for row in source_payload["candidates"][0]["evidence"]] == [
        "fixture:source-history:provenance",
        "fixture:source-history:metadata",
    ]
    assert [row["id"] for row in visual_payload["candidates"][0]["evidence"]] == [
        "fixture:source-history:metadata"
    ]


def test_provenance_citation_can_be_audited_but_does_not_upgrade_legal_claim() -> None:
    candidate = _candidate()
    question = "这件藏品的来源记录能证明原主人自愿出售和本馆合法所有吗？"
    audit = parse_audit(
        {
            "answerability": "partially_supported",
            "accepted": [
                {
                    "objectId": candidate.obj.id,
                    "relevanceScore": 0.95,
                    "evidenceIds": ["fixture:source-history:provenance"],
                }
            ],
            "expansionReason": "predicate_evidence_gap",
            "searchQueries": [],
            "coverageGap": "The record documents acquisition but not consent or title.",
        },
        [candidate],
        question=question,
        max_expansion_queries=3,
    )

    assert audit.valid is True
    assert audit.answerability == "partially_supported"
    assert [result.obj.id for result in audit.accepted] == [candidate.obj.id]
    assert audit.accepted[0].matched_evidence_ids == (
        "fixture:source-history:provenance",
    )
