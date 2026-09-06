from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from .retrieval_contract_fixtures import strict_audit_fixture

from app import curation
from app.collections import SearchResult
from app.config import Settings
from app.generator import ExhibitionGenerator
from app.models import AgendaInput, EvidenceChunk, MuseumObject
from app.providers.deepseek import ProviderError
from app.retrieval_agent import (
    _visible_evidence,
    audit_payload,
    fuse_search_results,
    parse_audit,
    parse_predicate_verification,
    parse_query_plan,
)


def _result(
    object_id: str,
    title: str,
    *,
    score: float = 80.0,
    culture: str = "",
    culture_packs: list[str] | None = None,
) -> SearchResult:
    evidence_id = f"{object_id}:metadata"
    obj = MuseumObject(
        id=object_id,
        sourceId=object_id,
        title=title,
        description=f"Institution description for {title}.",
        imageUrl=f"https://example.test/{object_id}.jpg",
        objectUrl=f"https://example.test/{object_id}",
        rights="CC0",
        institution="Fixture Museum",
        institutionId="fixture",
        culture=culture,
        culturePackIds=culture_packs or [],
        evidence=[
            EvidenceChunk(
                id=evidence_id,
                text=f"Institution record identifies {title}.",
                sourceUrl=f"https://example.test/{object_id}",
                sourceTitle=title,
                sourceKind="institution_metadata",
            ),
            EvidenceChunk(
                id=f"{object_id}:provenance",
                text="Gift of an unrelated donor.",
                sourceUrl=f"https://example.test/{object_id}",
                sourceTitle=title,
                sourceKind="institution_provenance",
            ),
        ],
    )
    return SearchResult(
        obj=obj,
        score=score,
        matched_evidence_ids=(evidence_id,),
        retrieval_sources=("dense_object", "dense_open_query", "evidence_rerank"),
        dense_score=0.72,
        evidence_score=0.70,
    )


def _agenda(question: str) -> AgendaInput:
    return AgendaInput(
        question=question,
        priorKnowledge="none",
        durationMinutes=5,
        collectionId="fixture",
    )


def test_audit_accepts_only_shown_non_provenance_evidence() -> None:
    candidate = _result("mirror", "Mirror portrait")
    audit = parse_audit(
        {
            "answerability": "supported",
            "accepted": [
                {
                    "objectId": "mirror",
                    "relevanceScore": 0.91,
                    "evidenceIds": ["mirror:metadata", "mirror:provenance"],
                },
                {
                    "objectId": "invented",
                    "relevanceScore": 1.0,
                    "evidenceIds": ["invented:e1"],
                },
            ],
            "searchQueries": [],
        },
        [candidate],
        question="镜子如何改变人们看待自我？",
        max_expansion_queries=3,
    )

    assert audit.valid is True
    assert [result.obj.id for result in audit.accepted] == ["mirror"]
    assert audit.accepted[0].matched_evidence_ids == ("mirror:metadata",)
    assert "llm_relevance_audit" in audit.accepted[0].retrieval_sources


def test_audit_does_not_count_object_embedding_without_an_evidence_id() -> None:
    candidate = _result("mirror", "Mirror portrait")
    audit = parse_audit(
        {
            "answerability": "unsupported",
            "accepted": [
                {
                    "objectId": "mirror",
                    "relevanceScore": 0.99,
                    "evidenceIds": [],
                }
            ],
            "searchQueries": [],
        },
        [candidate],
        question="topological qubit error correction",
        max_expansion_queries=3,
    )

    assert audit.valid is True
    assert audit.accepted == []


def test_query_plan_preserves_bounded_atomic_queries_without_approving_evidence() -> None:
    plan = parse_query_plan(
        {
            "inCollectionScope": True,
            "queryInterpretation": "无署名对象的作者归属方法",
            "catalogueQueries": [
                "unsigned",
                "attributed to",
                "formerly attributed",
                "fourth query is beyond the cap",
            ],
            "semanticQuery": (
                "unsigned or formerly attributed objects with documented "
                "attribution evidence"
            ),
            "mandatoryPerObjectPredicates": [
                "馆方记录须说明作者归属判断依据",
            ],
            "poolCoverageLegs": ["无署名", "旧归属", "当前归属"],
            "selectionRationaleConstraints": ["不因名家声望而入选"],
            "reason": "馆藏方法问题",
        },
        question="博物馆如何判断没有签名的对象是谁做的？",
        max_queries=3,
    )

    assert plan.valid is True
    assert plan.in_collection_scope is True
    assert plan.search_queries == (
        "unsigned",
        "attributed to",
        "formerly attributed",
    )
    assert plan.semantic_query == (
        "unsigned or formerly attributed objects with documented "
        "attribution evidence"
    )
    assert plan.interpretation == "无署名对象的作者归属方法"
    assert plan.mandatory_predicates == ("馆方记录须说明作者归属判断依据",)
    assert plan.pool_coverage_legs == ("无署名", "旧归属", "当前归属")
    assert plan.selection_constraints == ("不因名家声望而入选",)


def test_audit_requires_source_bound_checks_for_every_mandatory_predicate() -> None:
    candidate = _result("surface", "Worked surface")
    base = {
        "queryInterpretation": "成品表面可观察的制作痕迹",
        "answerability": "supported",
        "expansionReason": "none",
        "searchQueries": [],
        "coverageGap": "",
    }
    missing_check = parse_audit(
        {
            **base,
            "accepted": [
                {
                    "objectId": "surface",
                    "relevanceScore": 0.94,
                    "evidenceIds": ["surface:metadata"],
                    "predicateEvidence": [],
                }
            ],
        },
        [candidate],
        question="我想看得见制作过程留下的表面痕迹",
        max_expansion_queries=3,
        mandatory_predicates=("机构文字明确描述成品表面的可见制作痕迹",),
    )
    complete_check = parse_audit(
        {
            **base,
            "accepted": [
                {
                    "objectId": "surface",
                    "relevanceScore": 0.94,
                    "evidenceIds": ["surface:metadata"],
                    "predicateEvidence": [
                        {
                            "predicateId": "p1",
                            "status": "supported",
                            "evidenceId": "surface:metadata",
                            "supportingQuote": (
                                "Institution record identifies Worked surface."
                            ),
                        }
                    ],
                }
            ],
        },
        [candidate],
        question="我想看得见制作过程留下的表面痕迹",
        max_expansion_queries=3,
        mandatory_predicates=("机构文字明确描述成品表面的可见制作痕迹",),
    )

    assert missing_check.accepted == []
    assert [result.obj.id for result in complete_check.accepted] == ["surface"]


def test_predicate_verifier_requires_an_exact_quote_for_every_constraint() -> None:
    candidate = _result("surface", "Worked surface")
    evidence = EvidenceChunk(
        id="surface:direct",
        text="Marks made by the artist's fingers are visible in the surface.",
        sourceUrl="https://example.test/surface",
        sourceTitle="Worked surface",
        sourceKind="institution_description",
    )
    topic_evidence = candidate.obj.evidence[0]
    candidate = replace(
        candidate,
        obj=candidate.obj.model_copy(
            update={"evidence": [topic_evidence, evidence]}
        ),
        matched_evidence_ids=(topic_evidence.id,),
    )
    verified = parse_predicate_verification(
        {
            "verified": [
                {
                    "objectId": "surface",
                    "predicateEvidence": [
                        {
                            "predicateId": "p1",
                            "status": "entailed",
                            "evidenceId": evidence.id,
                            "supportingQuote": (
                                "Marks made by the artist's fingers are visible"
                            ),
                        }
                    ],
                }
            ]
        },
        [candidate],
        question="我想看到制作过程留下的可见痕迹",
        mandatory_predicates=("馆方文字明确描述成品上的可见制作痕迹",),
    )
    invented_quote = parse_predicate_verification(
        {
            "verified": [
                {
                    "objectId": "surface",
                    "predicateEvidence": [
                        {
                            "predicateId": "p1",
                            "status": "entailed",
                            "evidenceId": evidence.id,
                            "supportingQuote": "Visible hammer marks cover the object",
                        }
                    ],
                }
            ]
        },
        [candidate],
        question="我想看到制作过程留下的可见痕迹",
        mandatory_predicates=("馆方文字明确描述成品上的可见制作痕迹",),
    )

    assert [result.obj.id for result in verified] == ["surface"]
    assert verified[0].score == candidate.score
    assert verified[0].matched_evidence_ids == (topic_evidence.id, evidence.id)
    assert "llm_predicate_verifier" in verified[0].retrieval_sources
    assert invented_quote == []


def test_out_of_scope_query_plan_cannot_smuggle_catalogue_queries() -> None:
    plan = parse_query_plan(
        {
            "inCollectionScope": False,
            "queryInterpretation": "消费级设备售后",
            "catalogueQueries": ["robot firmware"],
            "reason": "对象记录不提供固件",
        },
        question="给扫地机器人找固件",
        max_queries=3,
    )

    assert plan.valid is True
    assert plan.in_collection_scope is False
    assert plan.search_queries == ()


def test_query_plan_requires_an_explicit_boolean_scope() -> None:
    plan = parse_query_plan(
        {
            "inCollectionScope": "yes",
            "catalogueQueries": ["attributed to"],
        },
        question="谁做的？",
        max_queries=3,
    )

    assert plan.valid is False
    assert plan.search_queries == ()


@pytest.mark.parametrize(
    ("answerability", "expansion_reason", "expected_reason"),
    [
        (
            "unsupported",
            "insufficient_direct_objects",
            "insufficient_direct_objects",
        ),
        ("unsupported", "out_of_scope", "out_of_scope"),
        ("partially_supported", "out_of_scope", "out_of_scope"),
    ],
)
def test_unsupported_or_out_of_scope_audit_suppresses_expansion_queries(
    answerability: str,
    expansion_reason: str,
    expected_reason: str,
) -> None:
    audit = parse_audit(
        {
            "queryInterpretation": "超出馆藏范围",
            "answerability": answerability,
            "accepted": [],
            "expansionReason": expansion_reason,
            "searchQueries": ["semantically adjacent but out-of-scope query"],
            "coverageGap": "馆藏记录不能回答这类请求。",
        },
        [_result("fixture", "Fixture object")],
        question="请求馆藏无法提供的产品固件",
        max_expansion_queries=3,
    )

    assert audit.valid is True
    if expansion_reason == "insufficient_direct_objects":
        assert audit.search_queries == (
            "semantically adjacent but out-of-scope query",
        )
    else:
        assert audit.search_queries == ()
    assert audit.expansion_reason == expected_reason


@pytest.mark.parametrize(
    "evidence_ids",
    [["another-object:metadata"], ["mirror:provenance"], ["mirror:hidden"]],
)
def test_audit_rejects_foreign_provenance_or_unseen_evidence_ids(
    evidence_ids: list[str],
) -> None:
    audit = parse_audit(
        {
            "answerability": "supported",
            "accepted": [
                {
                    "objectId": "mirror",
                    "relevanceScore": 0.99,
                    "evidenceIds": evidence_ids,
                }
            ],
            "searchQueries": [],
        },
        [_result("mirror", "Mirror portrait")],
        question="镜子如何改变人们看待自我？",
        max_expansion_queries=3,
    )

    assert audit.valid is True
    assert audit.accepted == []


@pytest.mark.parametrize("relevance", [1.01, 999, -0.1, float("nan"), float("inf")])
def test_audit_rejects_out_of_contract_relevance_scores(relevance: float) -> None:
    audit = parse_audit(
        {
            "answerability": "supported",
            "accepted": [
                {
                    "objectId": "mirror",
                    "relevanceScore": relevance,
                    "evidenceIds": ["mirror:metadata"],
                }
            ],
            "searchQueries": [],
        },
        [_result("mirror", "Mirror portrait")],
        question="镜子如何改变人们看待自我？",
        max_expansion_queries=3,
    )

    assert audit.valid is True
    assert audit.accepted == []


def test_query_fusion_marks_candidates_found_by_agentic_expansion() -> None:
    first = [_result("a", "A", score=90), _result("b", "B", score=80)]
    second = [_result("c", "C", score=95), _result("a", "A", score=70)]

    fused = fuse_search_results([first, second], limit=10)

    assert fused[0].obj.id == "a"
    assert {result.obj.id for result in fused} == {"a", "b", "c"}
    assert all(
        "agentic_query_expansion" in result.retrieval_sources
        for result in fused
        if result.obj.id in {"a", "c"}
    )


def test_query_fusion_prioritizes_evidence_found_by_atomic_expansion() -> None:
    original = _result("a", "Generic candidate")
    generic_id = original.matched_evidence_ids[0]
    direct_id = "a:direct-description"
    direct_chunk = EvidenceChunk(
        id=direct_id,
        text="The institution directly explains the requested method.",
        sourceUrl="https://example.test/a",
        sourceTitle="Generic candidate",
        sourceKind="institution_description",
    )
    expanded = replace(
        original,
        obj=original.obj.model_copy(
            update={"evidence": [*original.obj.evidence, direct_chunk]}
        ),
        matched_evidence_ids=(direct_id,),
    )

    fused = fuse_search_results([[original], [expanded]], limit=5)

    assert fused[0].matched_evidence_ids == (direct_id, generic_id)


def test_visible_evidence_keeps_top_match_curatorial_and_another_match() -> None:
    base = _result("layered", "Layered evidence object")
    top_match = EvidenceChunk(
        id="layered:top-match",
        text="The record identifies the object's plant motif.",
        sourceUrl="https://example.test/layered",
        sourceTitle="Layered evidence object",
        sourceKind="institution_metadata",
    )
    curatorial = EvidenceChunk(
        id="layered:curatorial",
        text=(
            "The curator explains how the material and making process shape "
            "the motif."
        ),
        sourceUrl="https://example.test/layered",
        sourceTitle="Layered evidence object",
        sourceKind="institution_curatorial_text",
    )
    second_match = EvidenceChunk(
        id="layered:second-match",
        text="The record also names the object's forming technique.",
        sourceUrl="https://example.test/layered",
        sourceTitle="Layered evidence object",
        sourceKind="institution_metadata",
    )
    unrelated = EvidenceChunk(
        id="layered:unrelated",
        text="A separate catalogue note does not bear on the visitor question.",
        sourceUrl="https://example.test/layered",
        sourceTitle="Layered evidence object",
        sourceKind="institution_metadata",
    )
    candidate = replace(
        base,
        obj=base.obj.model_copy(
            update={
                "evidence": [
                    unrelated,
                    second_match,
                    curatorial,
                    top_match,
                    *base.obj.evidence,
                ]
            }
        ),
        matched_evidence_ids=(top_match.id, second_match.id),
    )

    visible = _visible_evidence(candidate, "植物纹样的材料和制作有何差异？")

    assert [chunk.id for chunk in visible] == [
        top_match.id,
        curatorial.id,
        second_match.id,
    ]
    assert len(visible) == 3


def test_audit_payload_does_not_expose_uncitable_institution_description() -> None:
    candidate = _result("uncitable", "Object with detached description")
    assert all(
        candidate.obj.description not in chunk.text
        for chunk in candidate.obj.evidence
    )

    payload = audit_payload(
        "馆方如何描述这个对象？",
        [candidate],
        required_count=5,
        top_k=20,
        pass_number=1,
    )

    shown = payload["candidates"][0]
    shown_evidence_text = {
        evidence["text"] for evidence in shown["evidence"]
    }
    assert (
        "institutionDescription" not in shown
        or shown["institutionDescription"] in shown_evidence_text
    )


def test_audit_window_reserves_candidates_from_each_atomic_query_axis() -> None:
    fused = [
        _result(f"global-{index}", f"Global result {index}")
        for index in range(10)
    ]
    axis_a = [
        _result("axis-a-1", "Axis A first"),
        _result("axis-a-2", "Axis A second"),
    ]
    axis_b = [
        _result("axis-b-1", "Axis B first"),
        _result("axis-b-2", "Axis B second"),
    ]
    fused.extend([*axis_a, *axis_b])

    ordered = ExhibitionGenerator._query_coverage_order(
        fused,
        [axis_a, axis_b],
        top_k=7,
    )

    audit_ids = {result.obj.id for result in ordered[:7]}
    assert {
        "axis-a-1",
        "axis-a-2",
        "axis-b-1",
        "axis-b-2",
    }.issubset(audit_ids)


def test_audit_window_reserves_five_candidates_from_one_precise_axis() -> None:
    fused = [
        _result(f"global-{index}", f"Global result {index}")
        for index in range(12)
    ]
    precise = [
        _result(f"precise-{index}", f"Precise result {index}")
        for index in range(5)
    ]
    fused.extend(precise)

    ordered = ExhibitionGenerator._query_coverage_order(
        fused,
        [precise],
        top_k=15,
    )

    audit_ids = {result.obj.id for result in ordered[:15]}
    assert {result.obj.id for result in precise}.issubset(audit_ids)


def test_cross_cultural_sample_reserves_source_rich_semantic_candidates() -> None:
    thin_head = [
        replace(
            _result(
                f"thin-{index}",
                f"Thin object {index}",
                culture="China",
                culture_packs=["east_asia"],
            ),
            obj=_result(
                f"thin-{index}",
                f"Thin object {index}",
                culture="China",
                culture_packs=["east_asia"],
            ).obj.model_copy(
                update={"evidence_depth": "thin", "description": ""}
            ),
        )
        for index in range(10)
    ]
    rich_tail: list[SearchResult] = []
    for index, pack in enumerate(("africa", "europe", "oceania")):
        base = _result(
            f"rich-{index}",
            f"Source-rich object {index}",
            culture_packs=[pack],
        )
        rich_tail.append(
            replace(
                base,
                obj=base.obj.model_copy(
                    update={
                        "evidence_depth": "full",
                        "description": (
                            "The institution explains the object's material, "
                            "structure, and documented use in detail."
                        ),
                    }
                ),
                retrieval_sources=(
                    *base.retrieval_sources,
                    "agentic_semantic_synthesis",
                ),
                evidence_score=0.9 - index * 0.01,
            )
        )

    sample = ExhibitionGenerator._audit_candidate_sample(
        [*thin_head, *rich_tail],
        top_k=9,
        cross_cultural=True,
    )

    sample_ids = {result.obj.id for result in sample}
    assert {result.obj.id for result in rich_tail}.issubset(sample_ids)


def test_ordinary_sample_reserves_source_rich_semantic_candidates() -> None:
    thin_head: list[SearchResult] = []
    for index in range(8):
        base = _result(f"ordinary-thin-{index}", f"Thin object {index}")
        thin_head.append(
            replace(
                base,
                obj=base.obj.model_copy(
                    update={"evidence_depth": "thin", "description": ""}
                ),
                evidence_score=0.48,
            )
        )

    rich_tail: list[SearchResult] = []
    for index in range(2):
        base = _result(f"ordinary-rich-{index}", f"Source-rich object {index}")
        curatorial = EvidenceChunk(
            id=f"ordinary-rich-{index}:curatorial",
            text=(
                "The institution directly explains the object's material, "
                "making technique, and depicted plant motif."
            ),
            sourceUrl=f"https://example.test/ordinary-rich-{index}",
            sourceTitle=f"Source-rich object {index}",
            sourceKind="institution_curatorial_text",
        )
        rich_tail.append(
            replace(
                base,
                obj=base.obj.model_copy(
                    update={
                        "evidence_depth": "full",
                        "description": curatorial.text,
                        "evidence": [curatorial, *base.obj.evidence],
                    }
                ),
                matched_evidence_ids=(curatorial.id,),
                retrieval_sources=(
                    *base.retrieval_sources,
                    "agentic_semantic_synthesis",
                ),
                evidence_score=0.92 - index * 0.01,
            )
        )

    sample = ExhibitionGenerator._audit_candidate_sample(
        [*thin_head, *rich_tail],
        top_k=6,
        cross_cultural=False,
    )

    sample_ids = {result.obj.id for result in sample}
    assert {result.obj.id for result in rich_tail}.issubset(sample_ids)


def test_audit_candidates_fold_exact_parent_part_description_duplicates() -> None:
    parent = _result("pair", "Pair of guardians")
    shared_description = " ".join(
        ["The institution explains the same entrance guardian relationship."]
        * 4
    )
    parent = replace(
        parent,
        obj=parent.obj.model_copy(update={"description": shared_description}),
    )
    part = _result("pair.1", "First guardian")
    part = replace(
        part,
        obj=part.obj.model_copy(update={"description": shared_description}),
    )
    distinct = _result("other", "Another guardian")

    unique = ExhibitionGenerator._dedupe_audit_candidates(
        [parent, part, distinct]
    )

    assert [result.obj.id for result in unique] == ["pair", "other"]


class _CollectionsStub:
    def __init__(self, expanded: list[SearchResult]) -> None:
        self.expanded = expanded
        self.queries: list[str] = []

    def match_question_policy(self, _collection, _question):
        return None

    def search(self, agenda, _collection):
        self.queries.append(agenda.question)
        return self.expanded


class _PlanningCollectionsStub(_CollectionsStub):
    def __init__(self, expanded: list[SearchResult]) -> None:
        super().__init__(expanded)
        self.search_many_calls = 0

    def search_many(self, agendas, _collection):
        self.search_many_calls += 1
        self.queries.extend(agenda.question for agenda in agendas)
        return [self.expanded for _agenda in agendas]


class _PlanningAuditProvider:
    configured = True
    supports_retrieval_audit = True
    supports_retrieval_query_planning = True

    def __init__(self, *, in_scope: bool = True) -> None:
        self.in_scope = in_scope
        self.plan_calls = 0
        self.audit_calls = 0

    async def generate_retrieval_query_plan_json(
        self,
        _prompt: str,
        _payload: dict,
    ) -> dict:
        self.plan_calls += 1
        return {
            "inCollectionScope": self.in_scope,
            "queryInterpretation": "馆藏作者归属方法",
            "catalogueQueries": (
                ["unsigned", "attributed to"] if self.in_scope else []
            ),
            "reason": "馆藏方法问题" if self.in_scope else "消费设备售后",
        }

    @strict_audit_fixture
    async def generate_json(self, _prompt: str, payload: dict) -> dict:
        self.audit_calls += 1
        accepted = [
            {
                "objectId": candidate["objectId"],
                "relevanceScore": 0.92,
                "evidenceIds": [candidate["evidence"][0]["id"]],
                "reason": "direct attribution record",
            }
            for candidate in payload["candidates"]
            if candidate["objectId"].startswith("attributed-")
        ][:5]
        return {
            "queryInterpretation": "馆藏作者归属方法",
            "answerability": "supported" if len(accepted) >= 5 else "unsupported",
            "accepted": accepted,
            "expansionReason": "none" if len(accepted) >= 5 else "out_of_scope",
            "searchQueries": [],
            "coverageGap": "" if len(accepted) >= 5 else "无直接记录",
        }


class _PredicatePlanningAuditProvider:
    configured = True
    supports_retrieval_audit = True
    supports_retrieval_query_planning = True

    def __init__(self) -> None:
        self.audit_candidate_ids: list[str] = []
        self.verifier_candidate_ids: list[str] = []
        self.audit_calls = 0
        self.generic_calls = 0

    async def generate_retrieval_query_plan_json(
        self,
        _prompt: str,
        _payload: dict,
    ) -> dict:
        return {
            "inCollectionScope": True,
            "queryInterpretation": "逐件核对馆方文字",
            "catalogueQueries": [],
            "mandatoryPerObjectPredicates": ["馆方文字明确识别该对象"],
            "reason": "需要逐件证据",
        }

    @strict_audit_fixture
    async def generate_retrieval_audit_json(
        self,
        _prompt: str,
        payload: dict,
    ) -> dict:
        self.audit_calls += 1
        candidates = payload["candidates"]
        if "retrievalContract" in payload:
            self.audit_candidate_ids = [
                candidate["objectId"] for candidate in candidates
            ]
            accepted = []
            for candidate in candidates[:2]:
                evidence_id = candidate["evidence"][0]["id"]
                accepted.append(
                    {
                        "objectId": candidate["objectId"],
                        "relevanceScore": 0.9,
                        "evidenceIds": [evidence_id],
                        "predicateEvidence": [
                            {
                                "predicateId": "p1",
                                "status": "supported",
                                "evidenceId": evidence_id,
                                "supportingQuote": candidate["evidence"][0][
                                    "text"
                                ],
                            }
                        ],
                    }
                )
            return {
                "queryInterpretation": "逐件核对馆方文字",
                "answerability": "supported",
                "accepted": accepted,
                "expansionReason": "none",
                "searchQueries": [],
                "coverageGap": "",
            }

        self.verifier_candidate_ids = [
            candidate["objectId"] for candidate in candidates
        ]
        return {
            "verified": [
                {
                    "objectId": candidate["objectId"],
                    "predicateEvidence": [
                        {
                            "predicateId": "p1",
                            "status": "entailed",
                            "evidenceId": candidate["evidence"][0]["id"],
                            "supportingQuote": candidate["evidence"][0]["text"],
                        }
                    ],
                }
                for candidate in candidates
            ]
        }

    @strict_audit_fixture
    async def generate_json(self, _prompt: str, _payload: dict) -> dict:
        self.generic_calls += 1
        raise AssertionError("predicate verification must use the audit provider")


def test_predicate_audit_uses_exact_quotes_and_deterministic_provider() -> None:
    initial = [
        _result(f"candidate-{index}", f"Candidate {index}")
        for index in range(5)
    ]
    provider = _PredicatePlanningAuditProvider()
    generator = ExhibitionGenerator(
        Settings(),
        _PlanningCollectionsStub([]),  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("哪些馆藏记录能直接识别这些对象？"),
            object(),  # type: ignore[arg-type]
            initial,
            required_count=2,
        )
    )

    assert provider.audit_calls == 1
    assert provider.generic_calls == 0
    assert provider.audit_candidate_ids == [
        f"candidate-{index}" for index in range(5)
    ]
    assert provider.verifier_candidate_ids == []
    assert [result.obj.id for result in outcome.results] == [
        "candidate-0",
        "candidate-1",
    ]


def test_low_lexical_recall_uses_query_planner_before_source_bound_audit() -> None:
    initial = [_result("noise", "Signing an agreement")]
    expanded = [
        _result(f"attributed-{index}", f"Attributed work {index}")
        for index in range(6)
    ]
    collections = _PlanningCollectionsStub(expanded)
    provider = _PlanningAuditProvider()
    generator = ExhibitionGenerator(
        Settings(),
        collections,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("博物馆怎么判断没有签名的东西是谁做的？"),
            object(),  # type: ignore[arg-type]
            initial,
            required_count=5,
        )
    )

    assert provider.plan_calls == 1
    assert provider.audit_calls == 1
    assert collections.search_many_calls == 1
    assert collections.queries == ["unsigned", "attributed to"]
    assert outcome.expanded_queries == ("unsigned", "attributed to")
    assert len(outcome.results) == 5
    assert outcome.answerability == "supported"
    assert all(
        "agentic_query_expansion" in result.retrieval_sources
        and "llm_relevance_audit" in result.retrieval_sources
        for result in outcome.results
    )


def test_query_planner_runs_when_lexical_quantity_hides_false_precision() -> None:
    initial = [
        replace(
            _result(f"lexical-noise-{index}", f"Writing cabinet {index}"),
            retrieval_sources=("bm25",),
        )
        for index in range(8)
    ]
    expanded = [
        _result(f"attributed-{index}", f"Attribution case {index}")
        for index in range(6)
    ]
    collections = _PlanningCollectionsStub(expanded)
    provider = _PlanningAuditProvider()
    generator = ExhibitionGenerator(
        Settings(),
        collections,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("馆方记录如何区分几类文字功能？"),
            object(),  # type: ignore[arg-type]
            initial,
            required_count=5,
        )
    )

    assert provider.plan_calls == 1
    assert collections.search_many_calls == 1
    assert len(outcome.results) == 5
    assert outcome.answerability == "supported"


class _ThreeAxisPlanningProvider(_PlanningAuditProvider):
    async def generate_retrieval_query_plan_json(
        self,
        _prompt: str,
        _payload: dict,
    ) -> dict:
        self.plan_calls += 1
        return {
            "inCollectionScope": True,
            "queryInterpretation": "三个馆藏证据轴",
            "catalogueQueries": [
                "first object relation",
                "second material relation",
                "third use relation",
            ],
            "reason": "跨字段关系检索",
        }


def test_three_axis_plan_adds_one_bounded_semantic_synthesis_query() -> None:
    expanded = [
        _result(f"attributed-{index}", f"Related object {index}")
        for index in range(6)
    ]
    collections = _PlanningCollectionsStub(expanded)
    provider = _ThreeAxisPlanningProvider()
    generator = ExhibitionGenerator(
        Settings(rag_agentic_max_queries=5),
        collections,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("一个需要同时保留对象、材料和用途关系的问题"),
            object(),  # type: ignore[arg-type]
            [_result("noise", "Unrelated object")],
            required_count=5,
        )
    )

    assert collections.queries == [
        (
            "first object relation second material relation "
            "third use relation"
        ),
        "first object relation",
        "second material relation",
        "third use relation",
    ]
    assert len(outcome.results) == 5


def test_out_of_scope_query_plan_does_not_run_catalogue_expansion() -> None:
    initial = [_result("robot", "Historic toy robot")]
    collections = _PlanningCollectionsStub([])
    provider = _PlanningAuditProvider(in_scope=False)
    generator = ExhibitionGenerator(
        Settings(),
        collections,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("给扫地机器人找固件"),
            object(),  # type: ignore[arg-type]
            initial,
            required_count=5,
        )
    )

    assert provider.plan_calls == 1
    assert provider.audit_calls == 1
    assert collections.search_many_calls == 0
    assert outcome.answerability == "unsupported"
    assert outcome.results == []


class _AuditProvider:
    configured = True
    supports_retrieval_audit = True

    def __init__(self) -> None:
        self.calls = 0

    @strict_audit_fixture
    async def generate_json(self, _prompt: str, payload: dict) -> dict:
        self.calls += 1
        candidates = payload["candidates"]
        accept_count = 2 if self.calls == 1 else 5
        accepted = [
            {
                "objectId": candidate["objectId"],
                "relevanceScore": 0.92,
                "evidenceIds": [candidate["evidence"][0]["id"]],
                "reason": "direct fixture relevance",
            }
            for candidate in candidates[:accept_count]
        ]
        return {
            "queryInterpretation": "镜子与自我观看",
            "answerability": "supported",
            "accepted": accepted,
            "searchQueries": ["mirror self portrait"] if self.calls == 1 else [],
            "coverageGap": "",
        }


class _UnsupportedExpansionProvider:
    configured = True
    supports_retrieval_audit = True

    def __init__(self) -> None:
        self.calls = 0

    @strict_audit_fixture
    async def generate_json(self, _prompt: str, payload: dict) -> dict:
        self.calls += 1
        candidate = payload["candidates"][0]
        return {
            "queryInterpretation": "当代产品售后请求",
            "answerability": "unsupported",
            "accepted": [
                {
                    "objectId": candidate["objectId"],
                    "relevanceScore": 0.9,
                    "evidenceIds": [candidate["evidence"][0]["id"]],
                }
            ],
            "expansionReason": "out_of_scope",
            "searchQueries": ["robot firmware download"],
            "coverageGap": "馆藏记录不提供产品固件。",
        }


class _InScopeUnsupportedExpansionProvider:
    configured = True
    supports_retrieval_audit = True

    def __init__(self) -> None:
        self.calls = 0

    @strict_audit_fixture
    async def generate_json(self, _prompt: str, payload: dict) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {
                "queryInterpretation": "馆藏方法问题，但首批候选错误",
                "answerability": "unsupported",
                "accepted": [],
                "expansionReason": "predicate_evidence_gap",
                "searchQueries": ["attribution research"],
                "coverageGap": "首批候选没有方法证据。",
            }
        candidate = payload["candidates"][0]
        return {
            "queryInterpretation": "找到一条归属研究案例",
            "answerability": "partially_supported",
            "accepted": [
                {
                    "objectId": candidate["objectId"],
                    "relevanceScore": 0.94,
                    "evidenceIds": [candidate["evidence"][0]["id"]],
                }
            ],
            "expansionReason": "none",
            "searchQueries": [],
            "coverageGap": "个案不能代替完整方法指南。",
        }


def test_unsupported_first_audit_never_executes_model_proposed_expansion() -> None:
    candidate = _result("robot", "Historic toy robot")
    collections = _CollectionsStub([])
    provider = _UnsupportedExpansionProvider()
    generator = ExhibitionGenerator(
        Settings(),
        collections,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("给我的扫地机器人找固件"),
            object(),  # type: ignore[arg-type]
            [candidate],
            required_count=5,
        )
    )

    assert provider.calls == 1
    assert collections.queries == []
    assert [result.obj.id for result in outcome.results] == ["robot"]
    assert outcome.answerability == "unsupported"
    assert outcome.expansion_reason == "out_of_scope"
    assert outcome.failure_code is None
    assert outcome.warning_code is None


def test_in_scope_unsupported_first_pass_may_expand_then_reaudit() -> None:
    initial = [_result("noise", "Signing event")]
    expanded = [_result("method-case", "Reattributed object")]
    collections = _CollectionsStub(expanded)
    provider = _InScopeUnsupportedExpansionProvider()
    generator = ExhibitionGenerator(
        Settings(),
        collections,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("博物馆如何判断作者归属？"),
            object(),  # type: ignore[arg-type]
            initial,
            required_count=5,
        )
    )

    assert provider.calls == 2
    assert collections.queries == ["attribution research"]
    assert outcome.expanded_queries == ("attribution research",)
    assert [result.obj.id for result in outcome.results] == ["method-case"]
    assert outcome.answerability == "partially_supported"


def test_agentic_retrieval_expands_once_then_reaudits() -> None:
    initial = [_result("a", "Mirror A"), _result("b", "Mirror B")]
    expanded = [
        _result(object_id, f"Mirror {object_id.upper()}")
        for object_id in ("a", "b", "c", "d", "e", "f")
    ]
    collections = _CollectionsStub(expanded)
    provider = _AuditProvider()
    generator = ExhibitionGenerator(Settings(), collections, provider=provider)  # type: ignore[arg-type]

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("镜子如何改变人们看待自我？"),
            object(),  # type: ignore[arg-type]
            initial,
            required_count=5,
        )
    )

    assert outcome.audit_applied is True
    assert outcome.expanded_queries == ("mirror self portrait",)
    assert len(outcome.results) == 5
    assert provider.calls == 2
    assert collections.queries == ["mirror self portrait"]
    assert all(
        "llm_relevance_audit" in result.retrieval_sources
        for result in outcome.results
    )


class _UnavailableAuditProvider:
    configured = False
    supports_retrieval_audit = True


class _OverclaimingAuditProvider:
    configured = True
    supports_retrieval_audit = True

    @strict_audit_fixture
    async def generate_json(self, _prompt: str, payload: dict) -> dict:
        accepted = [
            {
                "objectId": candidate["objectId"],
                "relevanceScore": 0.95,
                "evidenceIds": [candidate["evidence"][0]["id"]],
                "reason": "fixture direct relevance",
            }
            for candidate in payload["candidates"][:4]
        ]
        return {
            "queryInterpretation": "fixture",
            "answerability": "supported",
            "accepted": accepted,
            "searchQueries": [],
            "coverageGap": "",
        }


def test_supported_audit_with_too_few_evidence_objects_is_downgraded() -> None:
    candidates = [_result(str(index), f"Object {index}") for index in range(6)]
    generator = ExhibitionGenerator(
        Settings(),
        _CollectionsStub([]),  # type: ignore[arg-type]
        provider=_OverclaimingAuditProvider(),  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("一个开放问题"),
            object(),  # type: ignore[arg-type]
            candidates,
            required_count=5,
        )
    )

    assert len(outcome.results) == 4
    assert outcome.answerability == "partially_supported"
    assert "需要 5 件" in outcome.coverage_gap


def test_final_diverse_selection_uses_the_same_canonical_regions_as_audit() -> None:
    results = [
        _result("china", "China", score=100, culture="China"),
        _result("japan", "Japan", score=99, culture="Japan"),
        _result("korea", "Korea", score=98, culture="Korea"),
        _result("tibet", "Tibet", score=97, culture="Tibet"),
        _result("france", "France", score=10, culture="France"),
        _result("egypt", "Egypt", score=9, culture="Egypt"),
    ]

    selected = curation.order_for_narrative(
        results,
        5,
        prefer_culture_diversity=True,
    )
    origins = {
        ExhibitionGenerator._canonical_object_origin(obj) for obj in selected
    } - {""}

    assert {"east_asia", "europe", "west_asia_north_africa"} <= origins


@pytest.mark.parametrize(
    "raw_origin",
    ["early 20th century", "Chicago", "LaSalle Street", "Workshop A"],
)
def test_unknown_catalogue_labels_do_not_count_as_cultural_regions(
    raw_origin: str,
) -> None:
    obj = _result("unknown-origin", "Unknown origin", culture=raw_origin).obj

    assert ExhibitionGenerator._canonical_object_origin(obj) == ""


def test_final_selection_repairs_an_explicit_low_rank_cultural_leg() -> None:
    results = [
        _result("china", "China", score=100, culture="China"),
        _result("france", "France", score=99, culture="France"),
        _result("nigeria", "Nigeria", score=98, culture="Nigeria"),
        _result("mexico", "Mexico", score=97, culture="Mexico"),
        _result("india", "India", score=96, culture="India"),
        _result("iran", "Iran", score=1, culture="Iran"),
    ]
    generator = ExhibitionGenerator.__new__(ExhibitionGenerator)
    initial = [result.obj for result in results[:5]]

    repaired = generator._ensure_final_cultural_coverage(
        _agenda("中国与伊朗的器物如何表现权力？"),
        initial,
        results,
        allow_repair=True,
    )

    assert any(obj.id == "china" for obj in repaired)
    assert any(obj.id == "iran" for obj in repaired)
    assert len(repaired) == 5


class _FailingAuditProvider:
    configured = True
    supports_retrieval_audit = True

    @strict_audit_fixture
    async def generate_json(self, _prompt: str, _payload: dict) -> dict:
        raise ProviderError("fixture audit outage")


class _InvalidAuditProvider:
    configured = True
    supports_retrieval_audit = True

    @strict_audit_fixture
    async def generate_json(self, _prompt: str, _payload: dict) -> dict:
        return {"accepted": "not-a-list"}


def test_open_dense_candidates_do_not_generate_without_required_audit() -> None:
    candidate = _result("nearest", "Unrelated nearest neighbour")
    generator = ExhibitionGenerator(
        Settings(),
        _CollectionsStub([]),  # type: ignore[arg-type]
        provider=_UnavailableAuditProvider(),  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("一个全新的开放问题"),
            object(),  # type: ignore[arg-type]
            [candidate],
            required_count=5,
        )
    )

    assert outcome.results == []
    assert outcome.failure_code == "RETRIEVAL_AUDIT_UNAVAILABLE"
    assert "审查模型不可用" in outcome.coverage_gap


def test_familiar_lexical_alias_does_not_bypass_required_audit() -> None:
    candidate = replace(
        _result("cat", "Cat figure"),
        retrieval_sources=("bm25",),
    )
    generator = ExhibitionGenerator(
        Settings(),
        _CollectionsStub([]),  # type: ignore[arg-type]
        provider=_UnavailableAuditProvider(),  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("猫在不同文化里为什么有不同含义？"),
            object(),  # type: ignore[arg-type]
            [candidate],
            required_count=5,
        )
    )

    assert outcome.results == []
    assert outcome.failure_code == "RETRIEVAL_AUDIT_UNAVAILABLE"


def test_disabling_audit_does_not_promote_open_dense_neighbours() -> None:
    candidate = _result("nearest", "Unrelated nearest neighbour")
    generator = ExhibitionGenerator(
        Settings(rag_llm_audit_enabled=False),
        _CollectionsStub([]),  # type: ignore[arg-type]
        provider=_AuditProvider(),  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("另一个从未见过的开放问题"),
            object(),  # type: ignore[arg-type]
            [candidate],
            required_count=5,
        )
    )

    assert outcome.results == []
    assert outcome.failure_code == "RETRIEVAL_AUDIT_UNAVAILABLE"
    assert "已停用" in outcome.coverage_gap


@pytest.mark.parametrize(
    "provider,expected_code",
    [
        (_FailingAuditProvider(), "RETRIEVAL_AUDIT_UNAVAILABLE"),
        (_InvalidAuditProvider(), "RETRIEVAL_AUDIT_INVALID"),
    ],
)
def test_first_evidence_audit_fails_closed(
    provider,
    expected_code: str,
) -> None:
    candidates = [
        _result(f"candidate-{index}", f"Candidate {index}")
        for index in range(5)
    ]
    generator = ExhibitionGenerator(
        Settings(),
        _CollectionsStub([]),  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("一个新的开放语义问题"),
            SimpleNamespace(concept_aliases={}),
            candidates,
            required_count=5,
        )
    )

    assert outcome.results == []
    assert outcome.failure_code == expected_code


def test_browse_all_route_skips_topical_relevance_audit() -> None:
    candidate = _result("surprise", "Unexpected object")
    candidate = SearchResult(
        obj=candidate.obj,
        score=1.0,
        retrieval_sources=("browse_all",),
    )
    provider = _AuditProvider()
    generator = ExhibitionGenerator(
        Settings(),
        _CollectionsStub([]),  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("我没想好，随便带我逛逛"),
            object(),  # type: ignore[arg-type]
            [candidate],
            required_count=5,
        )
    )

    assert outcome.results == [candidate]
    assert outcome.audit_applied is False
    assert provider.calls == 0


class _AcceptAllProvider:
    configured = True
    supports_retrieval_audit = True

    def __init__(self, *, search_queries: list[str] | None = None) -> None:
        self.search_queries = search_queries or []
        self.calls = 0

    @strict_audit_fixture
    async def generate_json(self, _prompt: str, payload: dict) -> dict:
        self.calls += 1
        return {
            "queryInterpretation": "跨文化比较",
            "answerability": "supported",
            "accepted": [
                {
                    "objectId": candidate["objectId"],
                    "relevanceScore": 0.9,
                    "evidenceIds": [candidate["evidence"][0]["id"]],
                }
                for candidate in payload["candidates"]
            ],
            "searchQueries": self.search_queries if self.calls == 1 else [],
            "coverageGap": "",
        }


class _CaptureAuditCandidatesProvider(_AcceptAllProvider):
    def __init__(self) -> None:
        super().__init__()
        self.candidate_ids: list[str] = []

    @strict_audit_fixture
    async def generate_json(self, prompt: str, payload: dict) -> dict:
        self.candidate_ids = [
            candidate["objectId"] for candidate in payload["candidates"]
        ]
        return await super().generate_json(prompt, payload)


def test_five_object_audit_can_use_configured_twenty_candidate_window() -> None:
    candidates = [
        _result(f"ordinary-{index}", f"Ordinary object {index}")
        for index in range(25)
    ]
    provider = _CaptureAuditCandidatesProvider()
    generator = ExhibitionGenerator(
        Settings(rag_llm_audit_top_k=20),
        _CollectionsStub([]),  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("馆藏对象如何记录材料与制作方法？"),
            object(),  # type: ignore[arg-type]
            candidates,
            required_count=5,
        )
    )

    assert len(provider.candidate_ids) == 20
    assert provider.candidate_ids == [
        f"ordinary-{index}" for index in range(20)
    ]
    assert outcome.answerability == "supported"


class _PreserveFirstAcceptedProvider:
    configured = True
    supports_retrieval_audit = True

    def __init__(self) -> None:
        self.calls = 0
        self.first_accepted_ids: list[str] = []
        self.second_candidate_ids: list[str] = []

    @strict_audit_fixture
    async def generate_json(self, _prompt: str, payload: dict) -> dict:
        self.calls += 1
        candidates = payload["candidates"]
        if self.calls == 1:
            self.first_accepted_ids = [
                candidate["objectId"] for candidate in candidates[-3:]
            ]
            accepted_ids = self.first_accepted_ids
            search_queries = ["translated evidence axis"]
            expansion_reason = "insufficient_direct_objects"
        else:
            self.second_candidate_ids = [
                candidate["objectId"] for candidate in candidates
            ]
            accepted_ids = self.first_accepted_ids
            search_queries = []
            expansion_reason = "none"
        by_id = {candidate["objectId"]: candidate for candidate in candidates}
        accepted = [
            {
                "objectId": object_id,
                "relevanceScore": 0.9,
                "evidenceIds": [by_id[object_id]["evidence"][0]["id"]],
            }
            for object_id in accepted_ids
            if object_id in by_id
        ]
        return {
            "queryInterpretation": "多个证据轴",
            "answerability": "partially_supported",
            "accepted": accepted,
            "expansionReason": expansion_reason,
            "searchQueries": search_queries,
            "coverageGap": "仍需更多个案。",
        }


def test_second_audit_preserves_first_acceptances_without_auditing_them_again() -> None:
    initial = [
        _result(f"initial-{index}", f"Initial object {index}")
        for index in range(18)
    ]
    expanded = [
        _result(f"expanded-{index}", f"Expanded object {index}")
        for index in range(30)
    ]
    provider = _PreserveFirstAcceptedProvider()
    generator = ExhibitionGenerator(
        Settings(rag_llm_audit_top_k=15),
        _CollectionsStub(expanded),  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("器物的制作方式如何留下不同痕迹？"),
            object(),  # type: ignore[arg-type]
            initial,
            required_count=5,
        )
    )

    assert provider.calls == 2
    assert not set(provider.second_candidate_ids) & set(provider.first_accepted_ids)
    assert [result.obj.id for result in outcome.results] == provider.first_accepted_ids


def test_cross_cultural_audit_payload_reserves_room_for_distinct_origins() -> None:
    candidates = [
        _result(
            f"china-{index}",
            f"Chinese object {index}",
            culture="China",
            culture_packs=["east_asia"],
        )
        for index in range(22)
    ]
    candidates.extend(
        [
            _result("africa-tail", "African object", culture="Nigeria"),
            _result("europe-tail", "European object", culture="Italy"),
            _result("americas-tail", "American object", culture="Mexico"),
        ]
    )
    provider = _CaptureAuditCandidatesProvider()
    generator = ExhibitionGenerator(
        Settings(rag_llm_audit_top_k=20),
        _CollectionsStub([]),  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("不同地区的母子像在姿态与材料上有什么差别？"),
            object(),  # type: ignore[arg-type]
            candidates,
            required_count=7,
        )
    )

    assert len(provider.candidate_ids) == 20
    assert provider.candidate_ids[:16] == [f"china-{index}" for index in range(16)]
    assert {"africa-tail", "europe-tail", "americas-tail"}.issubset(
        provider.candidate_ids
    )
    assert outcome.answerability == "supported"


def test_cross_cultural_audit_payload_reserves_each_named_cultural_leg() -> None:
    candidates = [
        _result(
            f"africa-{index}",
            f"African object {index}",
            culture="Nigeria",
            culture_packs=["africa"],
        )
        for index in range(22)
    ]
    candidates.extend(
        [
            _result(
                "east-asia-tail",
                "East Asian object",
                culture="China",
                culture_packs=["east_asia"],
            ),
            _result(
                "europe-tail",
                "European object",
                culture="Italy",
                culture_packs=["europe"],
            ),
            _result(
                "americas-tail",
                "American object",
                culture="Mexico",
                culture_packs=["americas"],
            ),
        ]
    )
    provider = _CaptureAuditCandidatesProvider()
    generator = ExhibitionGenerator(
        Settings(rag_llm_audit_top_k=20),
        _CollectionsStub([]),  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("东亚、欧洲和美洲的版画如何组织叙事？"),
            object(),  # type: ignore[arg-type]
            candidates,
            required_count=7,
        )
    )

    assert {
        "east-asia-tail",
        "europe-tail",
        "americas-tail",
    }.issubset(provider.candidate_ids)
    assert outcome.answerability == "supported"


@pytest.mark.parametrize(
    "question",
    [
        "同一个主题在不同文化中有什么差别？",
        "同一个主题在不同地区有什么差别？",
    ],
)
def test_cross_cultural_audit_rejects_five_objects_from_one_region(
    question: str,
) -> None:
    candidates = [
        _result(
            f"east-{index}",
            f"Object {index}",
            culture="China",
            culture_packs=["east_asia"],
        )
        for index in range(5)
    ]
    generator = ExhibitionGenerator(
        Settings(),
        _CollectionsStub([]),  # type: ignore[arg-type]
        provider=_AcceptAllProvider(),  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda(question),
            object(),  # type: ignore[arg-type]
            candidates,
            required_count=5,
        )
    )

    assert outcome.answerability == "partially_supported"
    assert "不足三个文化区域" in outcome.coverage_gap


def test_cross_cultural_audit_accepts_three_regions() -> None:
    packs = ["east_asia", "africa", "europe", "east_asia", "africa"]
    candidates = [
        _result(
            f"object-{index}",
            f"Object {index}",
            culture_packs=[pack],
        )
        for index, pack in enumerate(packs)
    ]
    generator = ExhibitionGenerator(
        Settings(),
        _CollectionsStub([]),  # type: ignore[arg-type]
        provider=_AcceptAllProvider(),  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("同一个主题在不同文化中有什么差别？"),
            object(),  # type: ignore[arg-type]
            candidates,
            required_count=5,
        )
    )

    assert outcome.answerability == "supported"
    assert len(outcome.results) == 5


def test_directional_variants_of_one_culture_do_not_fake_three_regions() -> None:
    candidates = [
        _result("india", "Indian object", culture="India"),
        _result("east-india", "Eastern Indian object", culture="Eastern India"),
        _result("west-india", "Western Indian object", culture="Western India"),
        _result("mughal-india", "Mughal object", culture="Mughal India"),
        _result("north-india", "Northern Indian object", culture="Northern India"),
    ]
    generator = ExhibitionGenerator(
        Settings(),
        _CollectionsStub([]),  # type: ignore[arg-type]
        provider=_AcceptAllProvider(),  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("不同文化中的人物形象有什么差别？"),
            object(),  # type: ignore[arg-type]
            candidates,
            required_count=5,
        )
    )

    assert outcome.answerability == "partially_supported"
    assert "不足三个文化区域" in outcome.coverage_gap


def test_explicit_named_legs_are_not_collapsed_by_coarse_region_guard() -> None:
    candidates = [
        _result("china-1", "Chinese object 1", culture="China"),
        _result("china-2", "Chinese object 2", culture="China"),
        _result("japan-1", "Japanese object 1", culture="Japan"),
        _result("japan-2", "Japanese object 2", culture="Japan"),
        _result("korea-1", "Korean object", culture="Korea"),
    ]
    generator = ExhibitionGenerator(
        Settings(),
        _CollectionsStub([]),  # type: ignore[arg-type]
        provider=_AcceptAllProvider(),  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("中国、日本与韩国的器物有什么差别？"),
            object(),  # type: ignore[arg-type]
            candidates,
            required_count=5,
        )
    )

    assert outcome.answerability == "supported"


def test_auxiliary_culture_packs_cannot_fake_three_origins() -> None:
    packs = [
        ["east_asia", "europe"],
        ["east_asia", "americas"],
        ["east_asia"],
        ["east_asia", "europe"],
        ["east_asia", "americas"],
    ]
    candidates = [
        _result(
            f"china-{index}",
            f"Chinese object {index}",
            culture="China",
            culture_packs=object_packs,
        )
        for index, object_packs in enumerate(packs)
    ]
    generator = ExhibitionGenerator(
        Settings(),
        _CollectionsStub([]),  # type: ignore[arg-type]
        provider=_AcceptAllProvider(),  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("同一个主题在不同文化中有什么差别？"),
            SimpleNamespace(concept_aliases={}),
            candidates,
            required_count=5,
        )
    )

    assert outcome.answerability == "partially_supported"
    assert "不足三个文化区域" in outcome.coverage_gap


def test_auxiliary_pack_cannot_make_one_origin_satisfy_two_named_legs() -> None:
    obj = _result(
        "china-mixed-pack",
        "Chinese object",
        culture="China",
        culture_packs=["east_asia", "europe"],
    ).obj
    obligations = {
        obligation.label_zh: obligation
        for obligation in ExhibitionGenerator._cultural_coverage_obligations(
            "中国与欧洲的器物如何表现权力？"
        )
    }

    assert ExhibitionGenerator._object_satisfies_cultural_obligation(
        obj, obligations["中国"]
    )
    assert not ExhibitionGenerator._object_satisfies_cultural_obligation(
        obj, obligations["欧洲"]
    )


def test_named_cultural_leg_must_exist_after_audit() -> None:
    candidates = [
        _result(
            f"china-{index}",
            f"Chinese object {index}",
            culture="China",
            culture_packs=["east_asia"],
        )
        for index in range(5)
    ]
    generator = ExhibitionGenerator(
        Settings(),
        _CollectionsStub([]),  # type: ignore[arg-type]
        provider=_AcceptAllProvider(),  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("中国与伊朗的器物如何表现权力？"),
            object(),  # type: ignore[arg-type]
            candidates,
            required_count=5,
        )
    )

    assert outcome.answerability == "partially_supported"
    assert "伊朗／波斯" in outcome.coverage_gap


class _SlowExpansionCollections(_CollectionsStub):
    def __init__(self, expanded: list[SearchResult]) -> None:
        super().__init__(expanded)
        self.batch_calls: list[tuple[str, ...]] = []

    def search(self, agenda, collection):
        self.queries.append(agenda.question)
        time.sleep(0.5)
        return self.expanded

    def search_many(self, agendas, _collection):
        questions = tuple(agenda.question for agenda in agendas)
        self.batch_calls.append(questions)
        self.queries.extend(questions)
        time.sleep(2.0)
        return [self.expanded for _agenda in agendas]


class _SlowSinglePassAuditProvider(_AcceptAllProvider):
    @strict_audit_fixture
    async def generate_json(self, prompt: str, payload: dict) -> dict:
        await asyncio.sleep(1.1)
        return await super().generate_json(prompt, payload)


def test_mandatory_first_audit_can_use_remaining_shared_budget() -> None:
    candidates = [
        _result(f"object-{index}", f"Object {index}")
        for index in range(5)
    ]
    generator = ExhibitionGenerator(
        Settings(
            rag_retrieval_timeout_seconds=1.5,
            rag_llm_audit_timeout_seconds=1.5,
        ),
        _CollectionsStub([]),  # type: ignore[arg-type]
        provider=_SlowSinglePassAuditProvider(),  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("不需要扩展的语义审核"),
            object(),  # type: ignore[arg-type]
            candidates,
            required_count=5,
        )
    )

    assert outcome.failure_code is None
    assert outcome.answerability == "supported"
    assert len(outcome.results) == 5


def test_insufficient_expansion_budget_does_not_start_background_search() -> None:
    initial = [_result("a", "A"), _result("b", "B")]
    collections = _SlowExpansionCollections([])
    provider = _AcceptAllProvider(search_queries=["q1", "q2", "q3"])
    # The first audit is immediate, but the remaining shared budget cannot
    # cover both atomic retrieval and a second source-bound audit. The optional
    # expansion must not start a worker that cannot be reviewed safely.
    settings = Settings(
        rag_retrieval_timeout_seconds=3.0,
        rag_llm_audit_timeout_seconds=3.0,
    )
    generator = ExhibitionGenerator(
        settings,
        collections,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    outcome = asyncio.run(
        generator._agentic_retrieve(
            _agenda("一个需要扩展检索的新问题"),
            object(),  # type: ignore[arg-type]
            initial,
            required_count=5,
        )
    )

    assert collections.batch_calls == []
    assert collections.queries == []
    assert [result.obj.id for result in outcome.results] == ["a", "b"]
    assert outcome.answerability == "partially_supported"
    assert outcome.interpretation == "跨文化比较"
    assert "时限" in outcome.coverage_gap
    assert outcome.failure_code == "RETRIEVAL_SEARCH_TIMEOUT"
    assert outcome.warning_code == "RETRIEVAL_SEARCH_TIMEOUT"
    assert outcome.warning_detail
    assert outcome.expanded_queries == ()


class _OpenDenseCheckCollections:
    def __init__(self) -> None:
        self.results = [
            _result(f"open-{index}", f"Open object {index}")
            for index in range(5)
        ]
        self.collection = SimpleNamespace(
            id="fixture",
            name="Fixture",
            version="v1",
            objects=[result.obj for result in self.results],
        )

    def get(self, _collection_id=None):
        return self.collection

    def search(self, _agenda, _collection):
        return self.results

    def eligible_objects(self, _collection):
        return [result.obj for result in self.results]

    def match_question_policy(self, _collection, _question):
        return None

    def recommend_questions(self, _collection):
        return []


def test_open_dense_check_is_explicitly_provisional() -> None:
    collections = _OpenDenseCheckCollections()
    generator = ExhibitionGenerator.__new__(ExhibitionGenerator)
    generator.collections = collections
    generator.provider = None
    generator.settings = None
    generator._audit_available_override = True

    check = generator.check_agenda(_agenda("镜子怎样改变自我观看？"))

    assert check.status == "supported"
    assert check.can_generate is True
    assert check.requires_runtime_audit is True
    assert check.decision_basis == "open_dense_provisional"
    assert "仍需逐件核查" in (check.answerable_part or "")


def test_probe_excludes_dense_only_results_when_audit_is_unavailable() -> None:
    collections = _OpenDenseCheckCollections()

    check = ExhibitionGenerator.probe_answerability(
        collections,  # type: ignore[arg-type]
        _agenda("镜子怎样改变自我观看？"),
        audit_available=False,
    )

    assert check.status == "unsupported"
    assert check.can_generate is False
    assert check.requires_runtime_audit is False
    assert check.decision_basis == "audit_unavailable"
    assert "不代表馆藏没有这个主题" in check.coverage_gaps[0]
    assert check.coverage.candidate_object_ids == []


def test_direct_agenda_check_matches_unavailable_audit_capability() -> None:
    collections = _OpenDenseCheckCollections()
    generator = ExhibitionGenerator(
        Settings(rag_llm_audit_enabled=True),
        collections,  # type: ignore[arg-type]
        provider=_UnavailableAuditProvider(),  # type: ignore[arg-type]
    )

    check = generator.check_agenda(_agenda("镜子怎样改变自我观看？"))

    assert check.status == "unsupported"
    assert check.can_generate is False
    assert check.requires_runtime_audit is False
    assert check.decision_basis == "audit_unavailable"
    assert "不代表馆藏没有这个主题" in check.coverage_gaps[0]
    assert check.coverage.candidate_object_ids == []


class _SlowInitialCollections(_CollectionsStub):
    def search(self, agenda, _collection):
        self.queries.append(agenda.question)
        time.sleep(0.7)
        return self.expanded


def test_initial_search_and_audit_share_one_monotonic_deadline() -> None:
    candidates = [_result(f"object-{index}", f"Object {index}") for index in range(5)]
    collections = _SlowInitialCollections(candidates)
    provider = _AuditProvider()
    generator = ExhibitionGenerator(
        Settings(
            rag_retrieval_timeout_seconds=1.0,
            rag_llm_audit_timeout_seconds=1.0,
        ),
        collections,  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
    )

    async def scenario():
        started = time.perf_counter()
        deadline = started + 1.0
        initial = await generator._search_async(
            _agenda("共享时限测试"),
            object(),  # type: ignore[arg-type]
            deadline=deadline,
        )
        outcome = await generator._agentic_retrieve(
            _agenda("共享时限测试"),
            object(),  # type: ignore[arg-type]
            initial,
            required_count=5,
            deadline=deadline,
        )
        return time.perf_counter() - started, outcome

    elapsed, outcome = asyncio.run(scenario())

    assert elapsed < 1.1
    assert outcome.failure_code == "RETRIEVAL_AUDIT_UNAVAILABLE"
    assert provider.calls == 0


def test_slow_local_retrieval_does_not_block_the_event_loop() -> None:
    candidates = [_result("object", "Object")]
    collections = _SlowInitialCollections(candidates)
    generator = ExhibitionGenerator(
        Settings(rag_retrieval_timeout_seconds=2.0),
        collections,  # type: ignore[arg-type]
        provider=_UnavailableAuditProvider(),  # type: ignore[arg-type]
    )

    async def scenario() -> int:
        ticks = 0
        finished = False

        async def heartbeat() -> None:
            nonlocal ticks
            while not finished:
                ticks += 1
                await asyncio.sleep(0.05)

        beat = asyncio.create_task(heartbeat())
        try:
            await generator._search_async(
                _agenda("事件循环测试"),
                object(),  # type: ignore[arg-type]
            )
        finally:
            finished = True
            await beat
        return ticks

    assert asyncio.run(scenario()) >= 5
