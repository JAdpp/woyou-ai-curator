from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
from copy import deepcopy

import pytest

from app.collections import CollectionDataError, SearchResult
from app.generator import ExhibitionGenerator
from app.models import AgendaInput, EvidenceChunk, MuseumObject
from app.set_coverage import set_coverage, persisted_set_coverage_valid


QUESTION = "我想看看同一画面里完整的互动关系"


def agenda(question=QUESTION):
    return AgendaInput(question=question, priorKnowledge="beginner", durationMinutes=10)


def requirement(minimum=1):
    return {"id": "s1", "text": "a complete interaction within one image",
            "sourceQuote": QUESTION, "minWitnesses": minimum,
            "quantifierOrigin": "product_default" if minimum == 1 else "visitor_explicit",
            "evidenceScope": "visible_features_or_record"}


def result(key, *, witness=False, culture="China", question=QUESTION):
    req = requirement()
    obj = MuseumObject(id=key, title=key, culture=culture, imageUrl="https://example.org/image",
        objectUrl="https://example.org/object", rights="CC0", evidenceDepth="full",
        evidence=[EvidenceChunk(id=f"{key}:text", text="A complete interaction within one image.",
                                sourceUrl="https://example.org/object", sourceTitle=key)])
    proof = {"requirementId": "s1", "objectId": key, "requirementText": req["text"],
             "sourceQuote": req["sourceQuote"], "questionSha256": sha256(question.encode()).hexdigest(),
             "evidenceIds": [f"{key}:text"], "checks": [{
                 "objectId": key, "conditionId": "s1", "status": "supported", "sourceBound": True,
                 "relation": "exact", "evidenceId": f"{key}:text",
                 "supportingQuote": "A complete interaction within one image."}]}
    return SearchResult(obj=obj, score=10, retrieval_sources=("condition_source_bound",),
                        set_witnesses=(proof,) if witness else ())


def test_one_witness_can_support_set_without_making_every_background_object_witness():
    pool = [result("core", witness=True), result("background")]
    report = set_coverage((requirement(),), pool, question=QUESTION)
    assert report["satisfied"] and report["requirements"][0]["witnessObjectIds"] == ["core"]
    assert report["semanticEntailmentProven"] is False


def test_pool_witness_not_in_final_selection_cannot_satisfy_goal():
    pool = [result("core", witness=True), result("background")]
    assert not set_coverage((requirement(),), pool, question=QUESTION,
                            selected_ids={"background"})["satisfied"]


def test_duplicate_object_or_quote_does_not_inflate_witness_count():
    core = result("core", witness=True)
    core = replace(core, set_witnesses=core.set_witnesses * 2)
    assert not set_coverage((requirement(2),), [core, core], question=QUESTION)["satisfied"]


@pytest.mark.parametrize("change", ["question", "text", "owner", "unbound", "broader"])
def test_stale_or_non_entailing_witnesses_are_not_counted(change):
    core = result("core", witness=True)
    proof = {**core.set_witnesses[0], "checks": [dict(core.set_witnesses[0]["checks"][0])]}
    if change == "question":
        proof["questionSha256"] = "0" * 64
    elif change == "text":
        proof["requirementText"] = "a different relationship"
    elif change == "owner":
        proof["checks"][0]["objectId"] = "neighbour"
    elif change == "unbound":
        proof["checks"][0]["sourceBound"] = False
    else:
        proof["checks"][0]["relation"] = "broader"
    core = replace(core, set_witnesses=(proof,))
    assert not set_coverage((requirement(),), [core], question=QUESTION)["satisfied"]


def test_final_selector_retains_core_witness_from_audited_pool():
    gen = ExhibitionGenerator.__new__(ExhibitionGenerator)
    pool = [result("a"), result("b"), result("core", witness=True)]
    selected, report = gen._ensure_final_set_coverage(
        agenda(), [pool[0].obj, pool[1].obj], pool, (requirement(),))
    assert len(selected) == 2 and "core" in {obj.id for obj in selected}
    assert report["scope"] == "final_selected_set" and report["satisfied"]


def test_final_selector_cannot_change_frozen_membership():
    gen = ExhibitionGenerator.__new__(ExhibitionGenerator)
    pool = [result("a"), result("core", witness=True)]
    with pytest.raises(CollectionDataError):
        gen._ensure_final_set_coverage(agenda(), [pool[0].obj], pool,
                                       (requirement(),), allow_repair=False)


def test_set_repair_does_not_drop_a_named_cultural_leg():
    question = "中国与日本的物品，" + QUESTION
    gen = ExhibitionGenerator.__new__(ExhibitionGenerator)
    pool = [result("cn", culture="China", question=question),
            result("jp", culture="Japan", question=question),
            result("core", culture="China", witness=True, question=question)]
    selected, report = gen._ensure_final_set_coverage(
        agenda(question), [pool[0].obj, pool[1].obj], pool, (requirement(),))
    assert {obj.id for obj in selected} == {"jp", "core"} and report["satisfied"]


def test_conflicting_fixed_size_coverage_fails_without_sacrificing_cultures():
    question = "中国与日本的物品，" + QUESTION
    gen = ExhibitionGenerator.__new__(ExhibitionGenerator)
    pool = [result("cn", culture="China", question=question),
            result("jp", culture="Japan", question=question),
            result("core", culture="France", witness=True, question=question)]
    with pytest.raises(CollectionDataError):
        gen._ensure_final_set_coverage(agenda(question),
                                       [pool[0].obj, pool[1].obj], pool, (requirement(),))


def persisted_exhibition():
    pool = [result("core", witness=True), result("background")]
    return SimpleNamespace(agenda=agenda(), items=[SimpleNamespace(object=item.obj) for item in pool],
        exhibition_set_coverage=set_coverage((requirement(),), pool, question=QUESTION,
                                             selected_ids={item.obj.id for item in pool}))


def test_persisted_coverage_rechecks_actual_membership_and_own_quote():
    exhibition = persisted_exhibition()
    assert persisted_set_coverage_valid(exhibition)
    exhibition.items = exhibition.items[1:]
    assert not persisted_set_coverage_valid(exhibition)


@pytest.mark.parametrize("change", ["quote", "count", "question", "owner"])
def test_persisted_coverage_rejects_stale_or_tampered_trace(change):
    exhibition = deepcopy(persisted_exhibition())
    row = exhibition.exhibition_set_coverage["requirements"][0]
    if change == "quote":
        row["witnesses"][0]["checks"][0]["supportingQuote"] = "An invented relationship."
    elif change == "count":
        row["witnessCount"] = 2
    elif change == "question":
        exhibition.agenda = agenda("另一个问题")
    else:
        row["witnesses"][0]["checks"][0]["evidenceId"] = "background:text"
    assert not persisted_set_coverage_valid(exhibition)
