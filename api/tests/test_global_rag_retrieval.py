from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import curation
from app.collections import CollectionDataError, CollectionRepository
from app.config import Settings
from app.generator import (
    BRONZE_REPAIR_EVIDENCE,
    RITUAL_PRACTICE_EVIDENCE,
    ExhibitionGenerator,
)
from app.interview import InterviewService
from app.models import (
    AgendaInput,
    AnswerabilityStatus,
    InterviewState,
    LocalizedObjectMetadata,
    VisitorProfile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def global_repository() -> tuple[CollectionRepository, object]:
    """Use the checked-in frozen corpus, not a hand-picked cat fixture."""

    repository = CollectionRepository(
        PROJECT_ROOT / "data" / "collections",
        default_collection_id="global_open",
    )
    collection = repository.get()
    assert collection.id == "global_open"
    return repository, collection


def _agenda(question: str) -> AgendaInput:
    return AgendaInput(
        question=question,
        priorKnowledge="none",
        durationMinutes=10,
    )


def test_global_cat_query_is_grounded_and_cross_culturally_diverse(
    global_repository,
) -> None:
    repository, collection = global_repository
    agenda = _agenda("有没有各文化地区的猫的藏品")

    results = repository.search(agenda, collection)

    assert len(results) >= 20
    assert all(result.matched_anchor_terms for result in results)
    assert all(
        set(result.matched_anchor_terms)
        & {"cat", "cats", "feline", "felines", "felis", "猫"}
        for result in results
    )
    assert "cma:1991.283" not in {result.obj.id for result in results}

    selected = ExhibitionGenerator._diverse_selection(
        results,
        5,
        prefer_culture_diversity=True,
    )
    selected_packs = {
        pack for obj in selected for pack in obj.culture_pack_ids
    }
    selected_culture_roots = {
        obj.culture.replace("，", ",").replace("；", ";").split(",", 1)[0].split(";", 1)[0].strip().casefold()
        for obj in selected
        if obj.culture.strip()
    }
    assert {
        "africa",
        "americas",
        "east_asia",
        "europe",
        "west_asia_north_africa",
    }.issubset(selected_packs)
    assert len(selected_culture_roots) == 5
    # Retrieved institution excerpts survive into the object passed to the
    # generation payload and are prioritised ahead of unrelated chunks.
    result_by_id = {result.obj.id: result for result in results}
    for obj in selected:
        matched_ids = result_by_id[obj.id].matched_evidence_ids
        if matched_ids:
            assert obj.evidence[0].id in matched_ids

    check = ExhibitionGenerator.probe_answerability(
        repository,
        agenda,
        audit_available=True,
    )
    assert check.status == AnswerabilityStatus.SUPPORTED
    assert check.can_generate is True
    assert check.requires_runtime_audit is True
    assert 5 <= check.coverage.matched_object_count <= len(results)


def test_colloquial_dog_query_is_answerable_and_cross_culturally_diverse(
    global_repository,
) -> None:
    """Regression for the visitor wording that previously produced 0 hits."""

    repository, collection = global_repository
    agenda = _agenda("狗狗在各国文化是怎么存在")

    results = repository.search(agenda, collection)
    dog_terms = {
        "dog",
        "dogs",
        "puppy",
        "puppies",
        "canine",
        "canines",
        "hound",
        "hounds",
        "狗狗",
        "小狗",
        "犬类",
    }
    assert len(results) >= 20
    assert all(set(result.matched_anchor_terms) & dog_terms for result in results)
    assert "cma:1971.294" not in {result.obj.id for result in results}

    selected = ExhibitionGenerator._diverse_selection(
        results,
        5,
        prefer_culture_diversity=True,
    )
    selected_packs = {pack for obj in selected for pack in obj.culture_pack_ids}
    selected_roots = {
        obj.culture.replace("，", ",").replace("；", ";").split(",", 1)[0].split(";", 1)[0].strip().casefold()
        for obj in selected
        if obj.culture.strip()
    }
    assert len(selected_packs) >= 4
    assert len(selected_roots) == 5

    check = ExhibitionGenerator.probe_answerability(
        repository,
        agenda,
        audit_available=True,
    )
    assert check.status == AnswerabilityStatus.SUPPORTED
    assert check.can_generate is True
    assert check.requires_runtime_audit is True
    assert check.coverage.matched_object_count >= 5


def test_emperor_query_is_grounded_without_treating_donor_names_as_rulers(
    global_repository,
) -> None:
    """Regression for a natural Chinese question that used to return 0 hits."""

    repository, collection = global_repository
    agenda = _agenda("皇帝在不同的文化象征是什么")
    results = repository.search(agenda, collection)
    ruler_terms = {"emperor", "empress", "monarch", "sovereign"}

    assert len(results) >= 20
    assert all(set(result.matched_anchor_terms) & ruler_terms for result in results)
    assert all(
        any(term in result.obj.title.casefold() for term in result.matched_anchor_terms)
        for result in results
    )
    assert "cma:1920.643" not in {result.obj.id for result in results}

    selected = ExhibitionGenerator._diverse_selection(
        results,
        5,
        prefer_culture_diversity=True,
    )
    selected_packs = {pack for obj in selected for pack in obj.culture_pack_ids}
    assert len(selected_packs) >= 3

    check = ExhibitionGenerator.probe_answerability(
        repository,
        agenda,
        audit_available=True,
    )
    assert check.status == AnswerabilityStatus.SUPPORTED
    assert check.can_generate is True
    assert check.requires_runtime_audit is True
    assert check.coverage.matched_object_count >= 5


def test_high_cost_predicates_are_not_approved_by_topic_counts_alone(
    global_repository,
) -> None:
    repository, collection = global_repository

    bronze = ExhibitionGenerator.probe_answerability(
        repository,
        _agenda("青铜器为什么会生锈，各文化怎么修复？"),
        audit_available=True,
    )
    assert bronze.status == AnswerabilityStatus.PARTIALLY_SUPPORTED
    assert bronze.can_generate is False
    assert bronze.coverage.matched_object_count >= 1
    assert "锈蚀机理" in bronze.coverage_gaps[0]

    sacred = ExhibitionGenerator.probe_answerability(
        repository,
        _agenda("宗教器物入馆后还能保持神圣性吗？"),
        audit_available=True,
    )
    assert sacred.status == AnswerabilityStatus.PARTIALLY_SUPPORTED
    assert sacred.can_generate is False
    assert sacred.coverage.matched_object_count >= 1
    assert "社群观点" in sacred.coverage_gaps[0]

    restitution = ExhibitionGenerator.probe_answerability(
        repository,
        _agenda("殖民时期被带走的文物该不该归还？"),
    )
    assert restitution.status == AnswerabilityStatus.UNSUPPORTED
    assert restitution.can_generate is False
    assert restitution.coverage.matched_object_count == 0
    assert restitution.coverage.candidate_object_ids == []
    assert "不能仅凭这些记录裁定" in restitution.coverage_gaps[0]

    by_id = {obj.id: obj for obj in collection.objects}
    for check, pattern in (
        (bronze, BRONZE_REPAIR_EVIDENCE),
        (sacred, RITUAL_PRACTICE_EVIDENCE),
    ):
        for object_id in check.coverage.candidate_object_ids:
            obj = by_id[object_id]
            if check is bronze:
                assert "bronze" in " ".join(
                    filter(None, (obj.title, obj.medium, obj.material, obj.type, obj.classification))
                ).casefold() or "copper alloy" in " ".join(
                    filter(None, (obj.title, obj.medium, obj.material, obj.type, obj.classification))
                ).casefold()
            assert any(
                chunk.source_kind != "institution_provenance"
                and pattern.search(" ".join(filter(None, (chunk.text, chunk.supports))))
                for chunk in obj.evidence
            )


@pytest.mark.parametrize(
    "question",
    [
        "青铜器为什么会生锈，各文化怎么修复？",
        "宗教器物入馆后还能保持神圣性吗？",
        "殖民时期被带走的文物该不该归还？",
    ],
)
def test_profile_generation_cannot_bypass_reviewed_negative_boundaries(
    global_repository,
    question: str,
) -> None:
    repository, collection = global_repository
    generator = ExhibitionGenerator(
        Settings(
            collections_dir=PROJECT_ROOT / "data" / "collections",
            default_collection_id=collection.id,
            rag_mode="bm25",
            rag_llm_audit_enabled=False,
            deepseek_api_key=None,
        ),
        repository,
    )
    profile = VisitorProfile(
        freeFormQuestion=question,
        curiosityLabel=question,
        durationMinutes=5,
    )

    with pytest.raises(CollectionDataError) as raised:
        asyncio.run(
            generator.generate_from_profile(profile, collection_id=collection.id)
        )

    assert raised.value.code == "QUESTION_UNSUPPORTED"
    assert raised.value.details["status"].value in {
        "partially_supported",
        "unsupported",
    }


def test_profile_generation_honors_exact_regression_policy(global_repository) -> None:
    repository, collection = global_repository
    question = "这些藏品能完整代表全世界每一种文化对死亡的看法吗？"
    policy_match = repository.match_question_policy(collection, question)
    assert policy_match is not None
    assert policy_match[1] == 1.0
    assert policy_match[0].status == "partially_supported"
    generator = ExhibitionGenerator(
        Settings(
            collections_dir=PROJECT_ROOT / "data" / "collections",
            default_collection_id=collection.id,
            rag_mode="bm25",
            rag_llm_audit_enabled=False,
            deepseek_api_key=None,
        ),
        repository,
    )

    with pytest.raises(CollectionDataError) as raised:
        asyncio.run(
            generator.generate_from_profile(
                VisitorProfile(
                    freeFormQuestion=question,
                    curiosityLabel=question,
                    durationMinutes=5,
                ),
                collection_id=collection.id,
            )
        )

    assert raised.value.code == "QUESTION_UNSUPPORTED"
    assert raised.value.details["status"].value == "partially_supported"


@pytest.mark.parametrize(
    "question",
    [
        "这些藏品今天在拍卖市场上分别值多少钱？",
        "参观这个展览能治疗我的焦虑吗？",
        "请生成一件看起来像真实出土文物的图片并把它当作馆藏展出。",
    ],
    ids=["auction-price", "anxiety-treatment", "fake-as-authentic"],
)
def test_exact_negative_policy_blocks_audit_generation_and_original_question_offer(
    global_repository,
    question: str,
) -> None:
    repository, collection = global_repository

    class CountingAuditProvider:
        configured = True
        supports_retrieval_audit = True

        def __init__(self) -> None:
            self.calls = 0

        async def generate_json(self, *_args, **_kwargs) -> dict:
            self.calls += 1
            return {}

    provider = CountingAuditProvider()
    generator = ExhibitionGenerator(
        Settings(
            collections_dir=PROJECT_ROOT / "data" / "collections",
            default_collection_id=collection.id,
            rag_mode="bm25",
            rag_llm_audit_enabled=True,
            deepseek_api_key="configured-test-key",
        ),
        repository,
        provider=provider,  # type: ignore[arg-type]
    )
    agenda = _agenda(question)

    policy_match = repository.match_question_policy(collection, question)
    assert policy_match is not None
    assert policy_match[1] == 1.0
    assert policy_match[0].status == AnswerabilityStatus.UNSUPPORTED.value

    check = generator.check_agenda(agenda)
    assert check.status == AnswerabilityStatus.UNSUPPORTED
    assert check.can_generate is False
    assert check.requires_runtime_audit is False
    assert check.decision_basis == "reviewed_policy"
    assert provider.calls == 0

    with pytest.raises(CollectionDataError) as raised:
        asyncio.run(generator.generate(agenda))

    assert raised.value.code == "QUESTION_UNSUPPORTED"
    assert raised.value.details["status"] == AnswerabilityStatus.UNSUPPORTED
    assert provider.calls == 0

    interview = InterviewService(repository, audit_available=True)
    state = InterviewState(
        id=f"negative-policy-{policy_match[0].policy_id}",
        collectionId=collection.id,
        profile=VisitorProfile(
            freeFormQuestion=question,
            durationMinutes=5,
        ),
    )
    negotiation = interview._negotiation_question(state, collection)

    assert negotiation is not None
    assert negotiation.allow_free_text is True
    assert all(
        option.label != "按原问题做语义核查" for option in negotiation.options
    )


def test_fuzzy_question_card_match_cannot_erase_high_cost_predicate(
    global_repository,
) -> None:
    repository, collection = global_repository
    question = (
        "不同文化怎样借助器物、图像与空间，让不可见的信仰变得可以实践；"
        "进入博物馆后还能保持神圣性吗？"
    )
    policy_match = repository.match_question_policy(collection, question)
    assert policy_match is not None
    assert 0.72 <= policy_match[1] < 1.0

    class CountingProvider:
        configured = True
        supports_retrieval_audit = True

        def __init__(self) -> None:
            self.calls = 0

        async def generate_json(self, _prompt: str, _payload: dict) -> dict:
            self.calls += 1
            return {}

    provider = CountingProvider()
    generator = ExhibitionGenerator(
        Settings(
            collections_dir=PROJECT_ROOT / "data" / "collections",
            default_collection_id=collection.id,
            rag_mode="bm25",
            rag_llm_audit_enabled=True,
            deepseek_api_key="configured-test-key",
        ),
        repository,
        provider=provider,  # type: ignore[arg-type]
    )

    check = generator.check_agenda(_agenda(question))
    assert check.status == AnswerabilityStatus.PARTIALLY_SUPPORTED
    assert check.can_generate is False
    assert check.decision_basis == "predicate_boundary"
    assert "社群观点" in check.coverage_gaps[0]

    with pytest.raises(CollectionDataError) as raised:
        asyncio.run(
            generator.generate_from_profile(
                VisitorProfile(
                    freeFormQuestion=question,
                    curiosityLabel=question,
                    durationMinutes=5,
                ),
                collection_id=collection.id,
            )
        )

    assert raised.value.code == "QUESTION_UNSUPPORTED"
    assert raised.value.details["status"].value == "partially_supported"
    assert provider.calls == 0


def test_fuzzy_question_card_never_replaces_audited_candidate_pool_with_starters(
    global_repository,
) -> None:
    repository, collection = global_repository
    question = "人们如何借肖像、服饰与身体姿态表达一个人是谁？"
    agenda = _agenda(question)
    policy_match = repository.match_question_policy(collection, question)
    assert policy_match is not None
    assert 0.72 <= policy_match[1] < 1.0

    # Stand in for the evidence IDs accepted by the runtime audit. The context
    # stage must remain inside this set; a fuzzy card may not inject its frozen
    # starters after the audit has already decided what is relevant.
    audited_results = repository.search(agenda, collection)[:8]
    audited_ids = {result.obj.id for result in audited_results}
    starter_ids = set(policy_match[0].starter_object_ids)
    assert audited_ids.isdisjoint(starter_ids)

    generator = ExhibitionGenerator(
        Settings(
            collections_dir=PROJECT_ROOT / "data" / "collections",
            default_collection_id=collection.id,
            rag_mode="bm25",
            rag_llm_audit_enabled=False,
        ),
        repository,
    )
    context = generator._context(agenda, all_results=audited_results)

    assert {obj.id for obj in context.selected} <= audited_ids
    assert {obj.id for obj in context.selected}.isdisjoint(starter_ids)
    assert context.policy is None


@pytest.mark.parametrize(
    "question",
    [
        "青铜器的材料与铸造在各文化有何不同？",
        "宗教器物在博物馆中如何展示？",
        "殖民时期图像如何呈现权力？",
    ],
)
def test_special_predicate_gate_does_not_capture_ordinary_comparisons(
    question: str,
) -> None:
    assert ExhibitionGenerator._predicate_assessment(question, []) is None


def test_named_cross_cultural_comparison_requires_every_requested_origin(
    global_repository,
) -> None:
    """Do not call a China/Iran/Delft comparison supported without Delft.

    The corpus has enough blue-and-white material for a superficially strong
    hit count, but the visitor explicitly asked for three comparison legs and
    excluded the religious Delft tiles.  A count-only gate used to approve this
    agenda and then silently substitute unrelated objects.
    """

    repository, _collection = global_repository
    agenda = AgendaInput(
        question="蓝色如何连接波斯陶瓷、中国青花与代尔夫特？",
        priorKnowledge="some",
        durationMinutes=15,
        exclusions=["宗教", "墓葬"],
    )

    check = ExhibitionGenerator.probe_answerability(
        repository,
        agenda,
        audit_available=True,
    )

    assert check.status == AnswerabilityStatus.PARTIALLY_SUPPORTED
    assert check.can_generate is False
    assert any("代尔夫特" in gap for gap in check.coverage_gaps)
    assert any("中国" in aspect and "伊朗" in aspect for aspect in check.supported_aspects)


def test_generic_across_cultures_request_is_not_mistaken_for_named_obligations(
    global_repository,
) -> None:
    repository, _collection = global_repository

    check = ExhibitionGenerator.probe_answerability(
        repository,
        _agenda("狗狗在各国文化是怎么存在"),
        audit_available=True,
    )

    assert check.status == AnswerabilityStatus.SUPPORTED
    assert check.can_generate is True
    assert not any("缺少" in gap for gap in check.coverage_gaps)


def test_unmatched_theme_is_refused_instead_of_backfilled_with_objects(
    global_repository,
) -> None:
    repository, collection = global_repository
    agenda = _agenda("topological qubit error correction")

    assert repository.search(agenda, collection) == []
    check = ExhibitionGenerator.probe_answerability(repository, agenda)
    assert check.status == AnswerabilityStatus.UNSUPPORTED
    assert check.can_generate is False
    assert check.coverage.matched_object_count == 0
    assert check.coverage.candidate_object_ids == []


def test_all_subject_terms_are_required_at_the_relevance_gate(
    global_repository,
) -> None:
    repository, collection = global_repository

    # One record mentions developments in physics, but no record mentions the
    # visitor's actual combined subject, quantum physics.  It must not be
    # promoted into a five-object exhibition by zero-score fallback.
    assert repository.search(_agenda("quantum physics"), collection) == []


def test_profile_alternatives_stay_inside_the_hard_gated_cat_results(
    global_repository,
) -> None:
    """The profile pipeline must not refill alternatives from the catalogue."""

    repository, collection = global_repository
    question = "有没有各文化地区的猫的藏品"
    profile = VisitorProfile(
        freeFormQuestion=question,
        curiosityLabel=question,
        durationMinutes=5,
    )
    agenda = profile.to_agenda(collection.id)
    results = repository.search(agenda, collection)
    selected = curation.order_for_narrative(
        results,
        profile.item_count,
        prefer_culture_diversity=True,
    )

    generator = ExhibitionGenerator.__new__(ExhibitionGenerator)
    generator.collections = repository
    generator.settings = SimpleNamespace(deepseek_model="test")
    generator.provider = None
    exhibition = generator._profile_skeleton(
        profile,
        agenda,
        collection,
        selected,
        None,
        results,
    )

    result_by_id = {result.obj.id: result for result in results}
    selected_ids = {item.object.id for item in exhibition.items}
    ranked_remaining = [
        result for result in results if result.obj.id not in selected_ids
    ]
    offered = [
        alternative
        for item in exhibition.items
        for alternative in item.alternatives
    ]

    assert offered
    assert [alternative.id for alternative in exhibition.items[0].alternatives] == [
        result.obj.id for result in ranked_remaining[:3]
    ]
    assert {alternative.id for alternative in offered} <= set(result_by_id)
    assert not {
        "Busby Building",
        "Column Capital",
        "Stray Horse",
    } & {alternative.title for alternative in offered}

    cat_terms = {"cat", "cats", "feline", "felines", "felis", "猫"}
    for alternative in offered:
        result = result_by_id[alternative.id]
        assert alternative.retrieval_score == pytest.approx(result.score)
        assert set(alternative.matched_anchor_terms) == set(
            result.matched_anchor_terms
        )
        assert set(alternative.matched_anchor_terms) & cat_terms
        assert set(alternative.matched_evidence_ids) == set(
            result.matched_evidence_ids
        )
        assert set(alternative.matched_evidence_ids) <= {
            chunk.id for chunk in result.obj.evidence
        }

    # Accepting an alternative must move its matched institution excerpt to
    # the front, so the replacement label does not fall back to unrelated text.
    target = next(
        item
        for item in exhibition.items
        if any(alternative.matched_evidence_ids for alternative in item.alternatives)
    )
    replacement_summary = next(
        alternative
        for alternative in target.alternatives
        if alternative.matched_evidence_ids
    )
    target.display_title = "上一件藏品的旧译名"
    target.localized_metadata = LocalizedObjectMetadata(
        date="上一件的年代",
        medium="上一件的材质",
        culture="上一件的地域",
        institution="上一件的机构",
    )
    generator.replace_item(exhibition, target.id, replacement_summary.id)
    assert target.object.id == replacement_summary.id
    assert target.object.evidence[0].id in replacement_summary.matched_evidence_ids
    assert target.display_title != "上一件藏品的旧译名"
    assert target.localized_metadata == LocalizedObjectMetadata()
