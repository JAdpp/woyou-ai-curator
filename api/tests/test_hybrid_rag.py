from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

import app.collections as collections_module

from app.collections import (
    CollectionDataError,
    CollectionRepository,
    _contextual_anchor_suppressions,
    _atomic_catalogue_query_plan,
    _query_plan,
    _tokens,
)
from app.dense_retrieval import (
    INDEX_FORMAT_VERSION,
    TEXT_RECIPE_SHA256,
    TEXT_RECIPE_VERSION,
    DenseHit,
    DenseIndex,
    DenseIndexManager,
    DenseRetrievalError,
    DenseStatus,
    EvidenceDenseHit,
    EvidenceHit,
    build_dense_index,
    collection_fingerprint,
)
from app.models import AgendaInput


def test_catalogue_tokenization_bridges_compounds_and_spelling_variants() -> None:
    tokens = _tokens("hand-built mould-made potter's catalogue")

    assert {"hand-built", "hand", "built"}.issubset(tokens)
    assert {"mould-made", "mould", "mold", "made"}.issubset(tokens)
    assert {"potter's", "potter"}.issubset(tokens)
    assert {"catalogue", "catalog"}.issubset(tokens)


def test_atomic_catalogue_query_requires_each_content_leg() -> None:
    plan = _atomic_catalogue_query_plan(
        "mould-made pottery technique",
        _query_plan("mould-made pottery technique"),
    )

    assert len(plan.strict_anchor_groups) == 2
    assert any({"mould", "mold"}.issubset(group) for group in plan.strict_anchor_groups)
    assert any({"pottery", "potteries"} & group for group in plan.strict_anchor_groups)
    assert all("made" not in group for group in plan.strict_anchor_groups)
    assert all("technique" not in group for group in plan.strict_anchor_groups)


def test_atomic_batch_search_does_not_let_generic_object_noun_swamp_axis(
    tmp_path: Path,
) -> None:
    collection_root = _write_collection(
        tmp_path,
        extra_subjects=[
            (
                "pottery-noise",
                "Pottery fragment",
                "A pottery fragment with no recorded forming method.",
            ),
            (
                "moulded-figurine",
                "Figurine",
                "This pottery figure was made in a two-part mold.",
            ),
        ],
    )
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="bm25",
    )
    agenda = _agenda("mould-made pottery")

    results = repository.search_many(
        [agenda],
        repository.get(),
        atomic=True,
    )[0]

    assert [result.obj.id for result in results] == ["fixture:moulded-figurine"]


def test_atomic_batch_search_uses_dense_prefilter_then_lexical_precision(
    tmp_path: Path,
) -> None:
    collection_root = _write_collection(
        tmp_path,
        extra_subjects=[
            (
                "finger-marks",
                "Coiled vessel",
                "Finger marks remain visible on this hand-built pottery vessel.",
            ),
        ],
    )
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="hybrid",
    )

    class _AtomicPrefilterIndex:
        def embed_queries(self, queries):
            return list(queries)

        def search_vector(self, _query_vector, *, top_k, allowed_object_ids=None):
            assert top_k > 0
            assert "fixture:finger-marks" in allowed_object_ids
            return [DenseHit("fixture:finger-marks", 0.99)]

        def search_evidence_vector(self, _query_vector, *, top_k):
            assert top_k > 0
            return []

    class _AtomicPrefilterManager:
        def __init__(self) -> None:
            self.index = _AtomicPrefilterIndex()

        def cache_signature(self, _collection):
            return ("atomic-prefilter", 1, 1)

        def get(self, _collection):
            return self.index

        def mark_query_success(self, _collection, _index):
            return None

        def mark_query_failure(self, _collection, _error):
            return None

    repository._dense_manager = _AtomicPrefilterManager()  # type: ignore[assignment]

    results = repository.search_many(
        [_agenda("finger marks pottery")],
        repository.get(),
        atomic=True,
    )[0]

    assert [result.obj.id for result in results] == ["fixture:finger-marks"]
    assert results[0].retrieval_sources == (
        "bm25",
        "dense_atomic_prefilter",
        "exact_atomic_prefilter",
    )


def _write_collection(
    root: Path,
    *,
    extra_subjects: list[tuple[str, str, str]] | None = None,
) -> Path:
    collection = root / "hybrid_fixture"
    collection.mkdir(parents=True)
    objects = []
    subjects = [
        ("cat", "Cat figure", "A small domestic cat watches the room."),
        (
            "gaze",
            "Mirror portrait",
            "Power shapes how the sitter looks back at the viewer.",
        ),
        ("mask", "Ceremonial mask", "A mask changes who may see and be seen."),
        ("screen", "Painted screen", "A screen controls visibility in a room."),
        ("portrait", "Double portrait", "Two people exchange a sustained gaze."),
        ("dog", "Dog figure", "A small domestic dog guards the doorway."),
    ]
    subjects.extend(extra_subjects or [])
    for object_id, title, evidence in subjects:
        objects.append(
            {
                "id": f"fixture:{object_id}",
                "sourceId": object_id,
                "accessionNumber": object_id,
                "title": title,
                "date": "1900",
                "creator": "Unknown",
                "material": "mixed media",
                "culture": "Test culture",
                "description": evidence,
                "imageUrl": f"https://example.test/{object_id}.jpg",
                "objectUrl": f"https://example.test/{object_id}",
                "rights": "CC0 1.0",
                "institutionId": "cma",
                "institution": "Fixture Museum",
                "evidence": [
                    {
                        "id": f"fixture:{object_id}:e1",
                        "text": evidence,
                        "sourceUrl": f"https://example.test/{object_id}",
                        "sourceTitle": title,
                    }
                ],
            }
        )
    (collection / "objects.json").write_text(
        json.dumps({"objects": objects}), encoding="utf-8"
    )
    (collection / "manifest.json").write_text(
        json.dumps(
            {
                "id": "hybrid_fixture",
                "name": "Hybrid fixture",
                "version": "v1",
            }
        ),
        encoding="utf-8",
    )
    return root


@dataclass
class _FakeIndex:
    hits: list[DenseHit]
    evidence: dict[str, EvidenceHit]
    evidence_recall: list[EvidenceDenseHit] | None = None

    def embed_query(self, query: str) -> str:
        return query

    def search_vector(self, query_vector, *, top_k, allowed_object_ids=None):
        return [
            hit
            for hit in self.hits[:top_k]
            if allowed_object_ids is None or hit.object_id in allowed_object_ids
        ]

    def evidence_hits(self, query_vector, object_ids, *, max_evidence_ids=3):
        return {
            object_id: self.evidence[object_id]
            for object_id in object_ids
            if object_id in self.evidence
        }

    def search_evidence_vector(self, query_vector, *, top_k):
        return list((self.evidence_recall or [])[:top_k])


class _FakeManager:
    def __init__(self, index: _FakeIndex | None) -> None:
        self.index = index

    def cache_signature(self, collection):
        return ("fake", 1, 1)

    def get(self, collection):
        return self.index

    def status(self, collection):
        return DenseStatus(
            enabled=True,
            available=self.index is not None,
            mode="hybrid" if self.index else "bm25",
            reason="fixture",
            model="fixture",
        )

    def mark_query_failure(self, collection, error):
        pass

    def mark_query_success(self, collection, index):
        pass


class _BatchProviderStub:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def embed(self, texts, batch_size=64):
        import numpy as np

        values = list(texts)
        self.calls.append((values, batch_size))
        vectors = {
            "first query": np.asarray([3.0, 4.0], dtype=np.float32),
            "second query": np.asarray([0.0, 2.0], dtype=np.float32),
        }
        yield from (vectors[value] for value in values)


def test_dense_index_embeds_multiple_queries_in_one_provider_batch() -> None:
    import numpy as np

    provider = _BatchProviderStub()
    index = object.__new__(DenseIndex)
    index.provider = provider
    index.manifest = {"dimension": 2}

    vectors = index.embed_queries(["first query", "second query"])

    assert provider.calls == [
        (["first query", "second query"], 2),
    ]
    np.testing.assert_allclose(vectors[0], [0.6, 0.8])
    np.testing.assert_allclose(vectors[1], [0.0, 1.0])
    assert index.embed_queries([]) == []
    assert len(provider.calls) == 1


class _BatchFakeIndex(_FakeIndex):
    batch_calls: list[tuple[str, ...]]
    single_calls: list[str]

    def embed_query(self, query: str) -> str:
        self.single_calls.append(query)
        return query

    def embed_queries(self, queries):
        query_list = list(queries)
        self.batch_calls.append(tuple(query_list))
        return query_list


def test_search_many_batches_uncached_queries_and_reuses_result_cache(
    tmp_path: Path,
) -> None:
    collection_root = _write_collection(tmp_path)
    index = _BatchFakeIndex(
        hits=[DenseHit("fixture:gaze", 0.99)],
        evidence_recall=[
            EvidenceDenseHit("fixture:gaze", "fixture:gaze:e1", 0.99)
        ],
        evidence={"fixture:gaze": EvidenceHit(0.99, ("fixture:gaze:e1",))},
    )
    index.batch_calls = []
    index.single_calls = []
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="hybrid",
        dense_manager=_FakeManager(index),
    )
    collection = repository.get()
    first_question = _agenda("spectral ancestors")
    second_question = _agenda("maritime memory")

    first = repository.search_many(
        [first_question, second_question, first_question],
        collection,
    )

    assert index.batch_calls == [
        ("spectral ancestors", "maritime memory"),
    ]
    assert index.single_calls == []
    assert first[0] == first[2]
    assert all(
        [result.obj.id for result in results] == ["fixture:gaze"]
        for results in first
    )

    cached = repository.search_many(
        [second_question, first_question],
        collection,
    )
    assert index.batch_calls == [
        ("spectral ancestors", "maritime memory"),
    ]
    assert cached == [first[1], first[0]]

    repository.search(_agenda("ritual thresholds"), collection)
    assert index.single_calls == ["ritual thresholds"]


def test_batched_lexical_search_matches_independent_queries(
    tmp_path: Path,
) -> None:
    collection_root = _write_collection(tmp_path)
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
    )
    collection = repository.get()
    eligible = repository.require_generation_ready(collection)
    agendas = [
        _agenda("猫在不同文化里如何出现？", excluded=["狗"]),
        _agenda("visibility in a room", excluded=["portrait"]),
        _agenda("我没想好，随便带我逛逛", excluded=["dog"]),
    ]
    requests = [
        (
            _query_plan(agenda.question, collection.concept_aliases),
            [
                _query_plan(topic, collection.concept_aliases).anchor_tokens
                for topic in agenda.excluded_topics
            ],
        )
        for agenda in agendas
    ]

    expected = [
        repository._lexical_search(query, excluded, eligible)
        for query, excluded in requests
    ]
    actual = repository._lexical_search_many(requests, eligible)

    assert actual == expected


def test_search_many_builds_each_lexical_document_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_root = _write_collection(tmp_path)
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="bm25",
    )
    collection = repository.get()
    eligible_count = len(repository.require_generation_ready(collection))
    build_calls = 0
    original_build = collections_module._build_search_document

    def counted_build(obj):
        nonlocal build_calls
        build_calls += 1
        return original_build(obj)

    monkeypatch.setattr(
        collections_module,
        "_build_search_document",
        counted_build,
    )

    results = repository.search_many(
        [
            _agenda("cat figure"),
            _agenda("sustained gaze"),
            _agenda("ceremonial mask"),
        ],
        collection,
    )

    assert build_calls == eligible_count
    assert len(results) == 3


def test_search_many_propagates_deadline_after_batch_embedding(
    tmp_path: Path,
) -> None:
    collection_root = _write_collection(tmp_path)
    index = _BatchFakeIndex(
        hits=[DenseHit("fixture:gaze", 0.99)],
        evidence_recall=[
            EvidenceDenseHit("fixture:gaze", "fixture:gaze:e1", 0.99)
        ],
        evidence={"fixture:gaze": EvidenceHit(0.99, ("fixture:gaze:e1",))},
    )
    index.batch_calls = []
    index.single_calls = []
    original_embed = index.embed_queries

    def slow_embed(queries):
        time.sleep(0.3)
        return original_embed(queries)

    index.embed_queries = slow_embed  # type: ignore[method-assign]
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="hybrid",
        dense_manager=_FakeManager(index),
    )
    collection = repository.get()
    agenda = _agenda("spectral ancestors")

    with pytest.raises(CollectionDataError) as raised:
        repository.search_many(
            [agenda],
            collection,
            deadline=time.perf_counter() + 0.2,
        )

    assert raised.value.code == "RETRIEVAL_SEARCH_TIMEOUT"
    assert index.batch_calls == [("spectral ancestors",)]
    # The timed-out vector result was not cached as a successful BM25 result.
    repository.search_many([agenda], collection)
    assert index.batch_calls == [
        ("spectral ancestors",),
        ("spectral ancestors",),
    ]


def _agenda(question: str, excluded: list[str] | None = None) -> AgendaInput:
    return AgendaInput(
        question=question,
        priorKnowledge="none",
        durationMinutes=10,
        excludedTopics=excluded or [],
    )


@pytest.mark.parametrize(
    "question,forbidden",
    [
        ("马赛克艺术如何跨文化传播", {"horse", "horses", "equine", "马"}),
        ("马克思主义艺术有哪些作品", {"horse", "horses", "equine", "马"}),
        ("猫头鹰在艺术中意味着什么", {"cat", "cats", "feline", "felis", "猫"}),
        ("热狗在美国饮食文化中如何流行", {"dog", "dogs", "puppy", "canine", "狗"}),
    ],
)
def test_single_character_chinese_concepts_do_not_match_inside_other_words(
    question: str,
    forbidden: set[str],
) -> None:
    plan = _query_plan(question)

    # The accidental single-character alias remains suppressed, but the whole
    # visitor question is still eligible for open-vocabulary embeddings.
    assert plan.dense_fallback_allowed is True
    assert plan.open_semantic_query is True
    assert not (plan.anchor_tokens & forbidden)
    assert plan.strict_anchor_groups == ()


@pytest.mark.parametrize(
    "question",
    [
        "我没想好，随便带我逛逛",
        "给我看点有意思的藏品",
        "没有想好，先带我看看",
    ],
)
def test_casual_browse_language_routes_to_browse_mode(question: str) -> None:
    plan = _query_plan(question)

    assert plan.browse_all is True
    assert plan.anchor_tokens == set()
    assert plan.open_semantic_query is False


@pytest.mark.parametrize(
    "question",
    [
        "随便给我看看海豚在不同文化中的形象",
        "给我看点拓扑量子纠错相关藏品",
        "这些藏品与气候危机有什么关系？",
        "推荐几个海豚展品",
    ],
)
def test_browse_wording_cannot_hide_a_topical_question(question: str) -> None:
    plan = _query_plan(question)

    assert plan.browse_all is False
    assert plan.open_semantic_query is True
    assert plan.dense_fallback_allowed is True


@pytest.mark.parametrize(
    "question,state_term",
    [
        ("博物馆如何判断无署名作品的作者归属？", "unsigned"),
        ("没有签名的器物怎么判断是谁做的？", "unsigned"),
        ("这东西没签名，怎么知道是谁做的？", "unsigned"),
        ("How do museums attribute an unsigned object?", "unsigned"),
        ("作者不明时，博物馆通过什么确定归属？", "anonymous"),
        ("How can a museum identify the maker of an anonymous object?", "anonymous"),
    ],
)
def test_creator_uncertainty_with_method_intent_adds_attribution_vocabulary(
    question: str,
    state_term: str,
) -> None:
    plan = _query_plan(question)

    assert plan.browse_all is False
    assert plan.open_semantic_query is True
    assert plan.strict_anchor_groups == ()
    assert plan.anchor_tokens & {
        "attribution",
        "reattributed",
        "stylistic",
        "connoisseurship",
        "authorship",
    }
    assert state_term in plan.scoring_tokens


@pytest.mark.parametrize(
    "question",
    [
        ("展示一些没有签名的画"),
        ("这幅画的签名在哪里？"),
        ("没有签名的画如何保存？"),
        ("Show me unsigned paintings"),
        ("How do museums conserve an unsigned painting?"),
        ("Signing the Declaration of Independence"),
    ],
)
def test_signature_mentions_without_attribution_method_intent_are_not_broadened(
    question: str,
) -> None:
    plan = _query_plan(question)

    assert not (
        plan.anchor_tokens
        & {
            "attribution",
            "reattributed",
            "stylistic",
            "connoisseurship",
            "authorship",
        }
    )


def test_unsigned_attribution_method_query_prefers_method_evidence_over_signature(
    tmp_path: Path,
) -> None:
    collection_root = _write_collection(
        tmp_path,
        extra_subjects=[
            (
                "signed-work",
                "Signed painting",
                "The artist's signature is present in the lower right corner.",
            ),
            (
                "attribution-study",
                "Study of a shepherd",
                "The attribution of this unsigned painting remains unsettled. "
                "Curators compared its stylistic characteristics with works "
                "assigned to several proposed artists.",
            ),
        ],
    )
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="bm25",
    )

    results = repository.search(
        _agenda("博物馆怎么判断一件没有签名的东西是谁做的？"),
        repository.get(),
    )

    assert results
    assert results[0].obj.id == "fixture:attribution-study"
    assert "fixture:signed-work" not in {result.obj.id for result in results}
    assert results[0].matched_evidence_ids == (
        "fixture:attribution-study:e1",
    )


@pytest.mark.parametrize(
    "question,expected",
    [
        ("有没有各文化地区的猫的藏品", {"cat", "cats", "feline", "felis", "猫"}),
        ("不同文化中的马有哪些形象", {"horse", "horses", "equine", "马"}),
        ("不同文化如何表现猫科动物", {"felidae", "lion", "tiger", "leopard"}),
        ("狗狗在各国文化是怎么存在", {"dog", "dogs", "puppy", "canine", "狗狗"}),
        ("小狗在世界各地有哪些形象", {"dog", "dogs", "puppy", "canine", "小狗"}),
    ],
)
def test_independent_cat_horse_and_cat_family_concepts_still_route(
    question: str,
    expected: set[str],
) -> None:
    plan = _query_plan(question)

    assert plan.dense_fallback_allowed is True
    assert plan.anchor_tokens & expected
    assert plan.strict_anchor_groups


@pytest.mark.parametrize(
    "question",
    [
        "狗狗在各国文化是怎么存在",
        "小狗在世界各地有哪些形象",
        "犬类在不同国家的文化中意味着什么",
    ],
)
def test_colloquial_dog_questions_request_cross_cultural_diversity(
    question: str,
) -> None:
    plan = _query_plan(question)

    assert plan.cross_cultural is True
    assert plan.dense_fallback_allowed is True
    assert plan.strict_anchor_groups
    assert plan.anchor_tokens & {"dog", "dogs", "puppy", "puppies", "canine", "hound"}


def test_different_regions_phrase_requests_cross_cultural_diversity() -> None:
    plan = _query_plan("不同地区的母子像在姿态与材料上有什么差别？", {})

    assert plan.cross_cultural is True


def test_named_cross_cultural_legs_are_not_per_object_and_anchors() -> None:
    aliases = {
        "东亚": ("east asia", "china", "japan", "korea"),
        "欧洲": ("europe", "european"),
        "美洲": ("americas", "american"),
        "文字": ("text", "script", "inscription"),
        "叙事": ("narrative", "story"),
    }
    plan = _query_plan(
        "不同文化（东亚、欧洲与美洲）的版画如何利用边框、文字和重复图像组织叙事？",
        aliases,
    )
    cultural_terms = {
        "east",
        "asia",
        "china",
        "japan",
        "korea",
        "europe",
        "european",
        "americas",
        "american",
    }

    assert plan.cross_cultural is True
    assert all(not (group & cultural_terms) for group in plan.anchor_groups)
    assert any(group & {"text", "script", "inscription"} for group in plan.anchor_groups)
    assert any(group & {"narrative", "story"} for group in plan.anchor_groups)

    compact_plan = _query_plan(
        "东亚、欧洲和美洲的版画如何组织叙事？",
        aliases,
    )
    assert compact_plan.cross_cultural is True
    assert all(not (group & cultural_terms) for group in compact_plan.anchor_groups)


@pytest.mark.parametrize(
    "question",
    [
        "不同地方的章鱼在艺术里是什么样？",
        "猫在各地艺术里如何出现？",
        "多个地方怎样描绘母亲与孩子？",
        "几个地区的狗有什么不同？",
        "来自各处的鸟被做成了什么？",
    ],
)
def test_colloquial_place_phrases_request_cross_cultural_diversity(
    question: str,
) -> None:
    assert _query_plan(question, {}).cross_cultural is True


@pytest.mark.parametrize(
    "question",
    [
        "母亲抱着孩子的形象，在相隔很远的社会里分别被用来表达什么？",
        "来自远隔重洋的作品里，小孩子都在做什么？",
        "相隔很远的漆器，在做法和装饰上有什么差别？",
        "不同社会如何表现家庭关系？",
    ],
)
def test_distance_language_requests_cross_cultural_diversity(
    question: str,
) -> None:
    assert _query_plan(question, {}).cross_cultural is True


@pytest.mark.parametrize(
    "question",
    [
        "不同地方的章鱼如何出现在艺术里？",
        "地方艺术中的拓扑量子纠错是什么？",
        "互不相识的地方，蛇分别怎样出现在护身符和画面里？",
    ],
)
def test_generic_alias_cannot_close_an_arbitrary_free_form_question(
    question: str,
) -> None:
    plan = _query_plan(question, {"地方": ("place", "region")})

    assert plan.browse_all is False
    assert plan.open_semantic_query is True


@pytest.mark.parametrize(
    "question,expected",
    [
        ("皇帝在不同的文化象征是什么", {"emperor", "empress", "monarch", "sovereign"}),
        ("不同文明的皇帝象征什么", {"emperor", "empress", "monarch", "sovereign"}),
        ("帝王在各国文化中的形象", {"emperor", "empress", "monarch", "sovereign"}),
        ("君主在世界各地怎样被描绘", {"monarch", "sovereign", "king", "queen"}),
        ("国王在不同文化中是什么象征", {"king", "queen", "monarch", "sovereign"}),
    ],
)
def test_rulership_questions_use_reviewed_title_subject_anchors(
    question: str,
    expected: set[str],
) -> None:
    plan = _query_plan(question)

    assert plan.cross_cultural is True
    assert plan.dense_fallback_allowed is True
    assert plan.strict_anchor_groups
    assert plan.title_anchor_groups
    assert plan.anchor_tokens & expected


def test_dental_canine_cannot_satisfy_dog_anchor_but_companion_can(
    tmp_path: Path,
) -> None:
    collection_root = _write_collection(
        tmp_path,
        extra_subjects=[
            (
                "dental-mask",
                "Leopard mask",
                "A fringe of carved leopard canine teeth identifies this mask as male.",
            ),
            (
                "canine-companion",
                "Canine companion",
                "A canine companion sits beside its owner.",
            ),
        ],
    )
    index = _FakeIndex(
        hits=[
            DenseHit("fixture:dental-mask", 0.99),
            DenseHit("fixture:canine-companion", 0.92),
            DenseHit("fixture:dog", 0.88),
        ],
        evidence={
            "fixture:dental-mask": EvidenceHit(
                0.99, ("fixture:dental-mask:e1",)
            ),
            "fixture:canine-companion": EvidenceHit(
                0.92, ("fixture:canine-companion:e1",)
            ),
            "fixture:dog": EvidenceHit(0.88, ("fixture:dog:e1",)),
        },
        evidence_recall=[
            EvidenceDenseHit(
                "fixture:dental-mask", "fixture:dental-mask:e1", 0.99
            ),
            EvidenceDenseHit(
                "fixture:canine-companion",
                "fixture:canine-companion:e1",
                0.92,
            ),
            EvidenceDenseHit("fixture:dog", "fixture:dog:e1", 0.88),
        ],
    )
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="hybrid",
        dense_manager=_FakeManager(index),
    )

    results = repository.search(
        _agenda("狗狗在各国文化是怎么存在"), repository.get()
    )
    result_ids = {result.obj.id for result in results}

    assert "fixture:dental-mask" not in result_ids
    assert {"fixture:dog", "fixture:canine-companion"} <= result_ids
    companion = next(
        result for result in results if result.obj.id == "fixture:canine-companion"
    )
    assert set(companion.matched_anchor_terms) & {"canine", "canines"}


@pytest.mark.parametrize(
    "text",
    [
        "A carved leopard canine tooth hangs from the mask.",
        "The fringe is made from leopard canine teeth.",
        "The specimen records canine dental morphology.",
        "A feline canine fang was mounted in silver.",
    ],
)
def test_canine_is_suppressed_when_its_only_context_is_dental(text: str) -> None:
    assert _contextual_anchor_suppressions(text) == {"canine", "canines"}


@pytest.mark.parametrize(
    "text",
    [
        "A canine companion sits beside its owner.",
        "The sculpture represents a watchful canine figure.",
        "A dog opens its mouth to reveal canine teeth.",
        "The hound has a prominent canine tooth.",
    ],
)
def test_canine_keeps_dog_meaning_when_independent_dog_context_exists(
    text: str,
) -> None:
    assert _contextual_anchor_suppressions(text) == set()


def test_dense_only_candidates_require_evidence_and_keep_trace(tmp_path: Path) -> None:
    collection_root = _write_collection(tmp_path)
    dense_ids = ["gaze", "mask", "screen", "portrait", "cat", "dog"]
    index = _FakeIndex(
        hits=[
            DenseHit(f"fixture:{object_id}", 0.80 - rank * 0.01)
            for rank, object_id in enumerate(dense_ids)
        ],
        evidence={
            f"fixture:{object_id}": EvidenceHit(
                0.10 if object_id == "dog" else 0.76 - rank * 0.01,
                (f"fixture:{object_id}:e1",),
            )
            for rank, object_id in enumerate(dense_ids)
        },
    )
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="hybrid",
        dense_manager=_FakeManager(index),
        dense_min_score=0.3,
        evidence_min_score=0.3,
    )
    results = repository.search(_agenda("观看与被观看意味着什么"), repository.get())

    assert len(results) >= 5
    assert all("dense_object" in result.retrieval_sources for result in results)
    assert all(result.dense_score is not None for result in results)
    supported = {
        result.obj.id
        for result in results
        if "evidence_rerank" in result.retrieval_sources
    }
    assert {
        "fixture:gaze",
        "fixture:mask",
        "fixture:screen",
        "fixture:portrait",
    }.issubset(supported)
    assert all(
        result.evidence_score is not None and result.matched_evidence_ids
        for result in results
        if result.obj.id in supported
    )
    # The repository is now a recall layer for unknown concepts: a fake high
    # semantic score may keep the cat as a candidate. The LLM relevance audit,
    # tested separately, is the precision boundary before object selection.
    cat = next(result for result in results if result.obj.id == "fixture:cat")
    assert "evidence_rerank" in cat.retrieval_sources
    assert cat.matched_evidence_ids == ("fixture:cat:e1",)
    assert cat.evidence_score == pytest.approx(0.72)
    assert "fixture:dog" not in {result.obj.id for result in results}


def test_rrf_rewards_agreement_between_sparse_and_dense_channels(tmp_path: Path) -> None:
    collection_root = _write_collection(tmp_path)
    index = _FakeIndex(
        hits=[DenseHit("fixture:mask", 0.99), DenseHit("fixture:gaze", 0.70)],
        evidence={
            "fixture:mask": EvidenceHit(0.80, ("fixture:mask:e1",)),
            "fixture:gaze": EvidenceHit(0.80, ("fixture:gaze:e1",)),
        },
    )
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="hybrid",
        dense_manager=_FakeManager(index),
    )
    results = repository.search(_agenda("权力象征"), repository.get())

    assert results[0].obj.id == "fixture:gaze"
    assert results[0].retrieval_sources == (
        "bm25",
        "dense_object",
        "dense_open_query",
        "evidence_rerank",
    )
    assert dict(results[0].field_scores)["rrf"] > dict(results[1].field_scores)["rrf"]


def test_global_evidence_recall_finds_object_missed_by_object_vectors(
    tmp_path: Path,
) -> None:
    collection_root = _write_collection(tmp_path)
    index = _FakeIndex(
        hits=[],
        evidence_recall=[
            EvidenceDenseHit(
                f"fixture:{object_id}",
                f"fixture:{object_id}:e1",
                0.80 - rank * 0.01,
            )
            for rank, object_id in enumerate(
                ["gaze", "mask", "screen", "portrait", "cat"]
            )
        ],
        evidence={
            f"fixture:{object_id}": EvidenceHit(
                0.80 - rank * 0.01,
                (f"fixture:{object_id}:e1",),
            )
            for rank, object_id in enumerate(
                ["gaze", "mask", "screen", "portrait", "cat"]
            )
        },
    )
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="hybrid",
        dense_manager=_FakeManager(index),
    )
    results = repository.search(_agenda("观看与被观看意味着什么"), repository.get())

    assert {result.obj.id for result in results} == {
        "fixture:gaze",
        "fixture:mask",
        "fixture:screen",
        "fixture:portrait",
        "fixture:cat",
    }
    assert all(
        "dense_evidence_recall" in result.retrieval_sources for result in results
    )
    assert all("dense_object" not in result.retrieval_sources for result in results)
    assert all(result.matched_evidence_ids for result in results)


def test_unknown_multi_term_english_query_enters_open_dense_candidate_pool(
    tmp_path: Path,
) -> None:
    collection_root = _write_collection(tmp_path)
    index = _FakeIndex(
        hits=[DenseHit("fixture:gaze", 0.99)],
        evidence={"fixture:gaze": EvidenceHit(0.99, ("fixture:gaze:e1",))},
    )
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="hybrid",
        dense_manager=_FakeManager(index),
    )

    results = repository.search(
        _agenda("topological qubit error correction"), repository.get()
    )
    assert [result.obj.id for result in results] == ["fixture:gaze"]
    assert "dense_open_query" in results[0].retrieval_sources


def test_unknown_chinese_query_uses_the_same_open_dense_route(tmp_path: Path) -> None:
    collection_root = _write_collection(tmp_path)
    index = _FakeIndex(
        hits=[DenseHit("fixture:gaze", 0.99)],
        evidence_recall=[
            EvidenceDenseHit("fixture:gaze", "fixture:gaze:e1", 0.99)
        ],
        evidence={"fixture:gaze": EvidenceHit(0.99, ("fixture:gaze:e1",))},
    )
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="hybrid",
        dense_manager=_FakeManager(index),
    )

    results = repository.search(_agenda("拓扑量子比特纠错"), repository.get())
    assert [result.obj.id for result in results] == ["fixture:gaze"]
    assert "dense_open_query" in results[0].retrieval_sources


def test_concrete_cat_alias_retains_lexical_subject_gate(tmp_path: Path) -> None:
    collection_root = _write_collection(tmp_path)
    index = _FakeIndex(
        hits=[DenseHit("fixture:dog", 0.99), DenseHit("fixture:cat", 0.80)],
        evidence={
            "fixture:dog": EvidenceHit(0.99, ("fixture:dog:e1",)),
            "fixture:cat": EvidenceHit(0.80, ("fixture:cat:e1",)),
        },
    )
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="hybrid",
        dense_manager=_FakeManager(index),
    )
    results = repository.search(_agenda("有没有猫的藏品"), repository.get())

    assert {result.obj.id for result in results} == {"fixture:cat"}
    assert results[0].matched_anchor_terms


def test_low_score_dense_evidence_is_not_reported_as_matched(
    tmp_path: Path,
) -> None:
    collection_root = _write_collection(tmp_path)
    objects_path = collection_root / "hybrid_fixture" / "objects.json"
    payload = json.loads(objects_path.read_text(encoding="utf-8"))
    cat = next(obj for obj in payload["objects"] if obj["id"] == "fixture:cat")
    cat["description"] = "An unidentified small figure."
    cat["evidence"][0]["text"] = "An unidentified small figure."
    objects_path.write_text(json.dumps(payload), encoding="utf-8")
    index = _FakeIndex(
        hits=[DenseHit("fixture:cat", 0.80)],
        evidence={"fixture:cat": EvidenceHit(0.10, ("fixture:cat:e1",))},
    )
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="hybrid",
        dense_manager=_FakeManager(index),
    )
    results = repository.search(_agenda("有没有猫的藏品"), repository.get())

    assert len(results) == 1
    assert results[0].retrieval_sources == (
        "bm25",
        "dense_object",
        "dense_open_query",
    )
    assert results[0].matched_evidence_ids == ()
    assert results[0].evidence_score is None
    assert dict(results[0].field_scores)["evidence_cosine_unverified"] == 0.10


def test_dense_context_cannot_turn_acquisition_credit_into_query_evidence(
    tmp_path: Path,
) -> None:
    collection_root = _write_collection(tmp_path)
    objects_path = collection_root / "hybrid_fixture" / "objects.json"
    payload = json.loads(objects_path.read_text(encoding="utf-8"))
    gaze = next(obj for obj in payload["objects"] if obj["id"] == "fixture:gaze")
    gaze["title"] = "Royal identity portrait"
    gaze["description"] = "A royal portrait constructs public identity."
    gaze["evidence"] = [
        {
            "id": "fixture:gaze:acquisition",
            "text": "General Fund",
            "supports": "acquisition credit",
            "sourceUrl": "https://example.test/gaze",
            "sourceTitle": "Royal identity portrait",
        },
        {
            "id": "fixture:gaze:description",
            "text": "A royal portrait constructs public identity.",
            "supports": "royal status and identity",
            "sourceUrl": "https://example.test/gaze",
            "sourceTitle": "Royal identity portrait",
        },
    ]
    objects_path.write_text(json.dumps(payload), encoding="utf-8")

    index = _FakeIndex(
        hits=[DenseHit("fixture:gaze", 0.88)],
        # Object-title context makes the acquisition row look closer than the
        # real description in the frozen embedding recipe.  The query-time raw
        # excerpt gate must reject that contaminated evidence id.
        evidence={
            "fixture:gaze": EvidenceHit(
                0.95,
                (
                    "fixture:gaze:acquisition",
                    "fixture:gaze:description",
                ),
                (
                    ("fixture:gaze:acquisition", 0.95),
                    ("fixture:gaze:description", 0.80),
                ),
            )
        },
        evidence_recall=[
            EvidenceDenseHit(
                "fixture:gaze",
                "fixture:gaze:acquisition",
                0.95,
            ),
            EvidenceDenseHit(
                "fixture:gaze",
                "fixture:gaze:description",
                0.80,
            ),
        ],
    )
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="hybrid",
        dense_manager=_FakeManager(index),
        dense_min_score=0.3,
        evidence_min_score=0.3,
    )

    results = repository.search(_agenda("royal identity"), repository.get())

    assert results
    result = results[0]
    assert result.obj.id == "fixture:gaze"
    assert result.matched_evidence_ids == ("fixture:gaze:description",)
    assert "fixture:gaze:acquisition" not in result.matched_evidence_ids
    assert "dense_evidence_recall" in result.retrieval_sources
    assert "evidence_rerank" in result.retrieval_sources
    assert result.evidence_score == pytest.approx(0.80)
    assert dict(result.field_scores)["evidence_cosine"] == pytest.approx(0.80)


def test_excluded_topics_filter_dense_candidates_before_fusion(tmp_path: Path) -> None:
    collection_root = _write_collection(tmp_path)
    ids = ["dog", "gaze", "mask", "screen", "portrait", "cat"]
    index = _FakeIndex(
        hits=[
            DenseHit(f"fixture:{object_id}", 0.90 - rank * 0.01)
            for rank, object_id in enumerate(ids)
        ],
        evidence={
            f"fixture:{object_id}": EvidenceHit(
                0.85,
                (f"fixture:{object_id}:e1",),
            )
            for object_id in ids
        },
    )
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="hybrid",
        dense_manager=_FakeManager(index),
    )
    results = repository.search(
        _agenda("观看与被观看意味着什么", ["dog"]), repository.get()
    )

    assert "fixture:dog" not in {result.obj.id for result in results}
    assert len(results) == 5


def test_unavailable_dense_index_returns_complete_bm25_results(tmp_path: Path) -> None:
    collection_root = _write_collection(tmp_path)
    collection_id = "hybrid_fixture"
    sparse = CollectionRepository(collection_root, default_collection_id=collection_id)
    degraded = CollectionRepository(
        collection_root,
        default_collection_id=collection_id,
        rag_mode="hybrid",
        dense_manager=_FakeManager(None),
    )
    question = _agenda("cat")

    sparse_results = sparse.search(question, sparse.get())
    degraded_results = degraded.search(question, degraded.get())
    assert [result.obj.id for result in degraded_results] == [
        result.obj.id for result in sparse_results
    ]
    assert [result.score for result in degraded_results] == [
        result.score for result in sparse_results
    ]
    assert degraded.retrieval_status(degraded.get()).mode == "bm25"


class _FailingIndex(_FakeIndex):
    calls = 0

    def embed_query(self, query: str):
        type(self).calls += 1
        raise RuntimeError("transient ONNX failure")


class _TrackingManager(_FakeManager):
    def __init__(self, index) -> None:
        super().__init__(index)
        self.failures = 0

    def mark_query_failure(self, collection, error):
        self.failures += 1


def test_transient_hybrid_failure_is_not_cached_and_retries_next_request(
    tmp_path: Path,
) -> None:
    collection_root = _write_collection(tmp_path)
    index = _FailingIndex(hits=[], evidence={})
    manager = _TrackingManager(index)
    repository = CollectionRepository(
        collection_root,
        default_collection_id="hybrid_fixture",
        rag_mode="hybrid",
        dense_manager=manager,
    )
    collection = repository.get()
    _FailingIndex.calls = 0

    first = repository.search(_agenda("cat"), collection)
    second = repository.search(_agenda("cat"), collection)

    assert [result.obj.id for result in first] == ["fixture:cat"]
    assert [result.obj.id for result in second] == ["fixture:cat"]
    assert _FailingIndex.calls == 2
    assert manager.failures == 2


class _LocalProviderStub:
    def artifact_metadata(self):
        return {"modelArtifactSha256": None}


def _write_minimal_dense_cache(path: Path, collection, fingerprint: str) -> None:
    import numpy as np

    objects = [obj for obj in collection.objects if obj.evidence]
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "formatVersion": INDEX_FORMAT_VERSION,
                "collectionId": collection.id,
                "collectionVersion": collection.version,
                "fingerprint": fingerprint,
                "model": "fixture-model",
                "objectsSha256": collection.objects_sha256,
                "textRecipeVersion": TEXT_RECIPE_VERSION,
                "textRecipeSha256": TEXT_RECIPE_SHA256,
                "dimension": 2,
                "objectCount": len(objects),
                "evidenceCount": len(objects),
            }
        ),
        encoding="utf-8",
    )
    (path / "object_ids.json").write_text(
        json.dumps([obj.id for obj in objects]), encoding="utf-8"
    )
    (path / "evidence_ids.json").write_text(
        json.dumps([obj.evidence[0].id for obj in objects]), encoding="utf-8"
    )
    np.save(path / "object_embeddings.npy", np.zeros((len(objects), 2), np.float32))
    np.save(path / "evidence_embeddings.npy", np.zeros((len(objects), 2), np.float32))
    np.save(path / "evidence_offsets.npy", np.arange(len(objects) + 1, dtype=np.int64))


def test_manifest_fingerprint_mismatch_degrades_without_serving_dense(
    tmp_path: Path, monkeypatch
) -> None:
    collection_root = _write_collection(tmp_path / "collections")
    collection = CollectionRepository(collection_root).get("hybrid_fixture")
    manager = DenseIndexManager(
        enabled=True,
        index_root=tmp_path / "rag",
        model_cache_dir=tmp_path / "models",
        model_name="fixture-model",
    )
    path = manager.expected_path(collection)
    _write_minimal_dense_cache(path, collection, "wrong-fingerprint")
    monkeypatch.setattr(
        "app.dense_retrieval.FastEmbedProvider",
        lambda *args, **kwargs: _LocalProviderStub(),
    )

    assert manager.get(collection) is None
    status = manager.status(collection)
    assert status.mode == "bm25"
    assert status.reason == (
        "dense cache or local model failed integrity validation; serving BM25"
    )


def test_corrupt_dense_manifest_degrades_without_network(
    tmp_path: Path, monkeypatch
) -> None:
    collection_root = _write_collection(tmp_path / "collections")
    collection = CollectionRepository(collection_root).get("hybrid_fixture")
    manager = DenseIndexManager(
        enabled=True,
        index_root=tmp_path / "rag",
        model_cache_dir=tmp_path / "models",
        model_name="fixture-model",
    )
    path = manager.expected_path(collection)
    path.mkdir(parents=True)
    (path / "manifest.json").write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(
        "app.dense_retrieval.FastEmbedProvider",
        lambda *args, **kwargs: _LocalProviderStub(),
    )

    assert manager.get(collection) is None
    assert manager.status(collection).mode == "bm25"
    assert manager.status(collection).reason == (
        "dense cache or local model failed integrity validation; serving BM25"
    )


def test_provider_error_path_is_kept_out_of_public_status(
    tmp_path: Path, monkeypatch
) -> None:
    collection_root = _write_collection(tmp_path / "collections")
    collection = CollectionRepository(collection_root).get("hybrid_fixture")
    manager = DenseIndexManager(
        enabled=True,
        index_root=tmp_path / "rag",
        model_cache_dir=tmp_path / "private-model-cache",
    )
    path = manager.expected_path(collection)
    path.mkdir(parents=True)
    (path / "manifest.json").write_text("{}", encoding="utf-8")

    def fail_provider(*args, **kwargs):
        raise DenseRetrievalError(r"failed C:\private\model-cache\model.onnx")

    monkeypatch.setattr("app.dense_retrieval.FastEmbedProvider", fail_provider)

    assert manager.get(collection) is None
    status = manager.status(collection)
    assert status.reason == (
        "dense cache or local model failed integrity validation; serving BM25"
    )
    assert "private" not in status.reason
    assert "model-cache" not in status.reason


class _BuildProviderStub:
    embedded = 0

    def __init__(self, *args, **kwargs) -> None:
        pass

    def embed(self, texts, batch_size=64):
        import numpy as np

        for _ in texts:
            type(self).embedded += 1
            yield np.asarray([1.0, 0.0], dtype=np.float32)

    def artifact_metadata(self):
        return {
            "modelRevision": "fixture-revision",
            "modelArtifactSha256": "a" * 64,
            "fastembedVersion": "fixture",
            "onnxruntimeVersion": "fixture",
        }


def test_interrupted_staging_resumes_from_first_non_unit_row(
    tmp_path: Path, monkeypatch
) -> None:
    import numpy as np

    collection_root = _write_collection(tmp_path / "collections")
    collection = CollectionRepository(collection_root).get("hybrid_fixture")
    model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    fingerprint = collection_fingerprint(collection, model)
    parent = tmp_path / "rag" / collection.id / model.replace("/", "-")
    staging = parent / f".{fingerprint}.resume-test"
    staging.mkdir(parents=True)

    object_matrix = np.lib.format.open_memmap(
        staging / "object_embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(6, 2),
    )
    object_matrix[:3] = [1.0, 0.0]
    object_matrix[3] = [0.2, 0.0]  # torn row must be overwritten
    object_matrix.flush()
    del object_matrix
    evidence_matrix = np.lib.format.open_memmap(
        staging / "evidence_embeddings.npy",
        mode="w+",
        dtype=np.float32,
        shape=(6, 2),
    )
    evidence_matrix[:2] = [1.0, 0.0]
    evidence_matrix.flush()
    del evidence_matrix

    _BuildProviderStub.embedded = 0
    monkeypatch.setattr(
        "app.dense_retrieval.FastEmbedProvider", _BuildProviderStub
    )
    output = build_dense_index(
        collection,
        index_root=tmp_path / "rag",
        model_cache_dir=tmp_path / "models",
        model_name=model,
        batch_size=2,
        resume_from=staging,
    )

    # Three unfinished object rows + four unfinished evidence rows.
    assert _BuildProviderStub.embedded == 7
    assert not staging.exists()
    assert output.name == fingerprint
    assert (output / "manifest.json").exists()
    assert np.allclose(
        np.linalg.norm(np.load(output / "object_embeddings.npy"), axis=1), 1.0
    )
    assert np.allclose(
        np.linalg.norm(np.load(output / "evidence_embeddings.npy"), axis=1), 1.0
    )
