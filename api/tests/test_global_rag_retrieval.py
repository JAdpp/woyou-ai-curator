from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app import curation
from app.collections import CollectionRepository
from app.generator import ExhibitionGenerator
from app.models import AgendaInput, AnswerabilityStatus, VisitorProfile


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

    check = ExhibitionGenerator.probe_answerability(repository, agenda)
    assert check.status == AnswerabilityStatus.SUPPORTED
    assert check.can_generate is True
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

    check = ExhibitionGenerator.probe_answerability(repository, agenda)
    assert check.status == AnswerabilityStatus.SUPPORTED
    assert check.can_generate is True
    assert check.coverage.matched_object_count >= 5


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

    check = ExhibitionGenerator.probe_answerability(repository, agenda)

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
    generator.replace_item(exhibition, target.id, replacement_summary.id)
    assert target.object.id == replacement_summary.id
    assert target.object.evidence[0].id in replacement_summary.matched_evidence_ids
