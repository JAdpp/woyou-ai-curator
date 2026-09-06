"""Visual evidence must cross both pixel-review and semantic-admission gates."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest

from app import curation
from app.collections import CollectionRepository, SearchResult
from app.config import Settings
from app.generator import ExhibitionGenerator
from app.models import AgendaInput, EvidenceChunk, Exhibition, MuseumObject, VisualCoreEvidence
from app.retrieval_agent import RetrievalQueryPlan
from app.validator import validate_exhibition
from app.visual_evidence import VisualEvidenceReport, VisualEvidenceResult, VisualObservation, visual_core_proof


QUESTION = "比较编织篮子可见的器形与提手"
DIGEST = "a" * 64


def _visual(obj: MuseumObject, *, reviewed=True) -> VisualEvidenceResult:
    return VisualEvidenceResult(
        object_id=obj.id, image_evidence_id=f"image:{obj.id}", source_url=obj.image_url,
        model="fixture-vision", status="reviewed" if reviewed else "unavailable",
        image_sha256=DIGEST, image_supplied=reviewed, image_reviewed=reviewed,
        question_relevance="supported", observations=(VisualObservation(
            id=f"visual:{obj.id}:{DIGEST[:16]}:1", aspect="shape",
            text="画面中央可见篮子的圆形开口以及弧形提手", location="画面中央", predicate_ids=("p1",)),))


def _proof(obj: MuseumObject, question: str) -> VisualCoreEvidence:
    payload = visual_core_proof(_visual(obj), evidence_mode="visual_observation")
    payload["questionSha256"] = sha256(question.encode("utf-8")).hexdigest()
    return VisualCoreEvidence.model_validate(payload)


def _candidate() -> SearchResult:
    obj = MuseumObject(
        id="fixture:basket", title="Basket", image_url="https://museum.test/basket.jpg",
        object_url="https://museum.test/basket", rights="CC0", type="Basketry",
        evidence=[EvidenceChunk(id="fixture:basket:metadata", text="Catalogue identifies a basket with a round opening and handle.",
                                source_url="https://museum.test/basket", source_title="Basket record",
                                source_kind="institution_metadata")])
    return SearchResult(obj=obj, score=2.0, retrieval_sources=("bm25",))


def _retrieve(monkeypatch, *, use_visual=True, reviewed=True, mode="visual_observation", stale_marker=False, strict=True):
    candidate = _candidate()
    if stale_marker:
        candidate = replace(candidate, retrieval_sources=("bm25", "visual_condition_source_bound"))
    frozen_before = candidate.obj.model_dump()
    report = VisualEvidenceReport((_visual(candidate.obj, reviewed=reviewed),))
    calls = []

    async def fake_visual(*_args, **_kwargs):
        calls.append("visual")
        return report

    monkeypatch.setattr("app.generator.audit_visual_candidates", fake_visual)

    class Repository(CollectionRepository):
        def match_question_policy(self, *_args):
            return None

    class Provider:
        configured = True
        supports_retrieval_audit = True
        supports_retrieval_query_planning = False

        async def generate_retrieval_audit_json(self, _prompt, payload):
            contract = payload["retrievalContract"]
            item = payload["candidates"][0]
            visuals = item.get("visualEvidence", [])
            source = visuals[0] if use_visual and visuals else item["evidence"][0]
            checks = [{"conditionId": row["id"], "status": "supported", "evidenceId": source["id"],
                       "supportingQuote": source["text"]} for row in contract["perObjectConditions"]]
            response = {"answerability": "supported", "expansionReason": "none", "accepted": [
                {"objectId": item["objectId"], "relevanceScore": .95,
                 "evidenceIds": [item["evidence"][0]["id"]], "conditionEvidence": checks}]}
            if strict:
                response["conditionContractVersion"] = contract["conditionContractVersion"]
            return response

    generator = ExhibitionGenerator(
        Settings(rag_agentic_max_queries=0, rag_visual_audit_enabled=True, rag_llm_audit_enabled=True),
        Repository.__new__(Repository), Provider(),
        image_cache=SimpleNamespace(get=lambda *_args: (b"fixture-image", True)))
    plan = RetrievalQueryPlan(valid=True, in_collection_scope=True, search_queries=(),
                              mandatory_predicates=("可见篮子的开口与提手形态",), evidence_mode=mode,
                              visual_predicate_ids=("p1",) if mode == "visual_observation" else ())
    outcome = asyncio.run(generator._agentic_retrieve(
        AgendaInput(question=QUESTION, prior_knowledge="none", duration_minutes=5, collection_id="fixture"),
        SimpleNamespace(concept_aliases={}), [candidate], required_count=1,
        initial_query_plan=plan, planning_attempted=True))
    assert candidate.obj.model_dump() == frozen_before
    assert candidate.obj.evidence_depth == "thin" and candidate.obj.visual_core_evidence is None
    return outcome, calls


def test_generator_attaches_copy_only_after_pixels_and_strict_visual_admission(monkeypatch):
    outcome, calls = _retrieve(monkeypatch)
    assert calls == ["visual"]
    assert len(outcome.results) == 1
    obj = outcome.results[0].obj
    assert obj.evidence_depth == "thin"
    assert obj.visual_core_evidence.question_sha256 == sha256(QUESTION.encode("utf-8")).hexdigest()
    assert obj.supports_core_evidence


@pytest.mark.parametrize("kwargs", [
    {"reviewed": False}, {"use_visual": False}, {"mode": "record_explanation"},
])
def test_no_visual_role_proof_without_both_current_gates(monkeypatch, kwargs):
    outcome, _calls = _retrieve(monkeypatch, **kwargs)
    assert all(result.obj.visual_core_evidence is None for result in outcome.results)


def test_legacy_audit_contract_cannot_grant_visual_role_proof(monkeypatch):
    outcome, _calls = _retrieve(monkeypatch, strict=False)
    assert not outcome.results


def test_stale_retrieval_marker_is_not_current_visual_semantic_admission(monkeypatch):
    outcome, _calls = _retrieve(monkeypatch, use_visual=False, stale_marker=True)
    assert outcome.results
    assert all(result.obj.visual_core_evidence is None for result in outcome.results)


@pytest.fixture()
def visual_exhibition(client, agenda_payload):
    response = client.post("/api/exhibitions/generate-sync", json={"agenda": agenda_payload})
    assert response.status_code == 200
    exhibition = Exhibition.model_validate(response.json())
    core = next(item for item in exhibition.items if item.role == "core_evidence")
    original = core.object
    copied = curation.with_collection_image_evidence(original)
    copied.evidence_depth = "thin"
    copied.visual_core_evidence = _proof(copied, exhibition.question)
    core.object = copied
    assert original.evidence_depth == "full" and original.visual_core_evidence is None
    return exhibition, core


def _core_errors(exhibition):
    return [error for error in validate_exhibition(exhibition).errors if error.code == "THIN_EVIDENCE_IN_CORE_ROLE"]


def test_visual_role_and_validator_agree_without_promoting_text_depth(visual_exhibition):
    exhibition, core = visual_exhibition
    assert not _core_errors(exhibition)
    objects = [item.object.model_copy(deep=True) for item in exhibition.items]
    for obj in objects:
        obj.evidence_depth = "thin"
    ordered, roles = curation.plan_roles(objects)
    assert roles[0] == "opening" and roles[-1] == "synthesis"
    assert ordered[roles.index("core_evidence")].id == core.object.id
    assert all(obj.evidence_depth == "thin" for obj in ordered)


def test_visual_proof_cannot_be_reused_for_new_historical_question(visual_exhibition):
    exhibition, _core = visual_exhibition
    exhibition.question = "这些篮子的形制为什么在殖民贸易时期发生变化？"
    exhibition.agenda.question = exhibition.question
    assert _core_errors(exhibition)


@pytest.mark.parametrize("mismatch", ["object", "image_id", "image_url", "observation_owner", "missing_marker", "marker_url"])
def test_visual_core_source_binding_is_preserved_at_publication(visual_exhibition, mismatch):
    exhibition, core = visual_exhibition
    proof = core.object.visual_core_evidence
    if mismatch == "object":
        core.object.visual_core_evidence = proof.model_copy(update={"object_id": "foreign"})
    elif mismatch == "image_id":
        core.object.visual_core_evidence = proof.model_copy(update={"image_evidence_id": "image:foreign"})
    elif mismatch == "image_url":
        core.object.visual_core_evidence = proof.model_copy(update={"source_url": "https://foreign.test/image"})
    elif mismatch == "observation_owner":
        core.object.visual_core_evidence = proof.model_copy(update={"observation_ids": [f"visual:foreign:{DIGEST[:16]}:1"]})
    elif mismatch == "missing_marker":
        core.object.evidence = [chunk for chunk in core.object.evidence if chunk.source_kind != "collection_image"]
    else:
        marker = next(chunk for chunk in core.object.evidence if chunk.source_kind == "collection_image")
        marker.source_url = "https://foreign.test/image"
    assert _core_errors(exhibition)


def test_image_source_cannot_be_used_as_institutional_material_fact(visual_exhibition):
    exhibition, core = visual_exhibition
    core.label_sentences[0].type = "institution_fact"
    core.label_sentences[0].text = "这是一只以黄金铸造的篮子"
    core.label_sentences[0].evidence_ids = [core.object.visual_core_evidence.image_evidence_id]
    errors = {error.code for error in validate_exhibition(exhibition).errors}
    assert "IMAGE_EVIDENCE_TYPE_MISMATCH" in errors
    assert "INSTITUTION_FACT_UNSUPPORTED" in errors


def test_preview_review_keeps_same_source_when_larger_image_also_exists():
    original = _candidate().obj
    original.image_url_large = "https://museum.test/basket-large.jpg"
    copied = original.model_copy(deep=True)
    copied.visual_core_evidence = _proof(copied, QUESTION)
    exhibition_copy = curation.with_collection_image_evidence(copied)
    image_source = curation.image_evidence(exhibition_copy)
    assert image_source.source_url == original.image_url
    assert image_source.source_url == exhibition_copy.visual_core_evidence.source_url
    assert original.visual_core_evidence is None and original.evidence_depth == "thin"
