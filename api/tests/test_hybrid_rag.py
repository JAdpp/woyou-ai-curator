from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.collections import (
    CollectionRepository,
    _contextual_anchor_suppressions,
    _query_plan,
)
from app.dense_retrieval import (
    INDEX_FORMAT_VERSION,
    TEXT_RECIPE_SHA256,
    TEXT_RECIPE_VERSION,
    DenseHit,
    DenseIndexManager,
    DenseRetrievalError,
    DenseStatus,
    EvidenceDenseHit,
    EvidenceHit,
    build_dense_index,
    collection_fingerprint,
)
from app.models import AgendaInput


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

    assert plan.dense_fallback_allowed is False
    assert not (plan.anchor_tokens & forbidden)
    assert plan.strict_anchor_groups == ()


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
    # Object-dense recall may retain a candidate whose highest evidence vector
    # inherited context from the object, but that row is not exposed as query
    # evidence unless the raw excerpt itself overlaps the expanded query.
    cat = next(result for result in results if result.obj.id == "fixture:cat")
    assert "evidence_rerank" not in cat.retrieval_sources
    assert cat.matched_evidence_ids == ()
    assert cat.evidence_score is None
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
    }
    assert all(
        "dense_evidence_recall" in result.retrieval_sources for result in results
    )
    assert all("dense_object" not in result.retrieval_sources for result in results)
    assert all(result.matched_evidence_ids for result in results)


def test_unknown_multi_term_english_query_cannot_use_nearest_dense_objects(
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

    assert repository.search(
        _agenda("topological qubit error correction"), repository.get()
    ) == []


def test_unknown_chinese_query_has_the_same_refusal_gate(tmp_path: Path) -> None:
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

    assert repository.search(_agenda("拓扑量子比特纠错"), repository.get()) == []


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
    assert results[0].retrieval_sources == ("bm25", "dense_object")
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
