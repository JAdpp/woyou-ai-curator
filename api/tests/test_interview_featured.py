"""An opening with no subject is shown the collection, never curated as-is."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import interview_featured
from app.interview_clarification import undecided_opening_kind
from app.interview_featured import FEATURED_ENTRIES, FeaturedEntry


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("我不知道应该问你什么", "unsure"),
        ("我也不知道诶", "unsure"),
        ("没想好，随便带我逛逛", "unsure"),
        ("？", "unsure"),
        ("你推荐吧", "delegate"),
        ("你看着办", "delegate"),
        ("随便", "browse"),
        ("你好", "greeting"),
        ("你好，我不知道看什么", "unsure"),
        ("你们有什么藏品", "collection"),
        ("不知道你们有什么", "collection"),
        ("介绍一下你们的馆藏", "collection"),
        ("你是谁", "meta"),
        ("I don't know what to ask", "unsure"),
        ("I'm not sure", "unsure"),
        ("surprise me", "delegate"),
        ("what do you have", "collection"),
    ],
)
def test_openings_without_a_subject_are_recognised(text, kind):
    assert undecided_opening_kind(text) == kind


@pytest.mark.parametrize(
    "text",
    [
        "我不知道青花瓷是怎么烧出来的",
        "推荐一些山水画",
        "你们有什么关于猫的藏品",
        "随便看看宋代的瓷器",
        "你好，我想看看猫",
        "不确定的年代怎么判断",
        "没有文字的器物怎么读",
        "给我做个最真实的展览",
        "I don't know much about Chinese bronzes",
        "hello kitty",
        "What do you have about cats?",
        "no cats please",
    ],
)
def test_openings_with_a_subject_stay_real_questions(text):
    assert undecided_opening_kind(text) is None


def test_undecided_detection_stays_fast_on_long_near_misses():
    import time

    started = time.perf_counter()
    for text in ("不知道" * 20, "你好" * 29 + "猫", "也" * 59 + "猫", "no " * 19 + "cats"):
        undecided_opening_kind(text[:60])
    assert time.perf_counter() - started < 0.5


def _answer(client, state, **answer):
    response = client.post(
        f"/api/interview/{state['id']}/answer",
        json={"questionId": state["nextQuestion"]["id"], **answer},
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def featured(client, monkeypatch):
    """Two featured entries whose hero objects exist in the fixture."""

    entries = (
        FeaturedEntry(
            id="alpha", object_id="TEST.1",
            title_zh="山水画怎样安排行旅的视线？", title_en="How does a landscape route the eye?",
            question_zh="山水画如何组织观看者的行旅视线？",
            question_en="How does a landscape painting route the viewer's eye?",
            caption_zh="Landscape object 1 · 测试", caption_en="Landscape object 1 · test",
            hook_zh="山水", hook_en="landscape",
            plan_zh="我会挑几幅山水放在一起看。", plan_en="I'll put a few landscapes together.",
        ),
        FeaturedEntry(
            id="beta", object_id="TEST.2",
            title_zh="书法与诗怎样一起被看？", title_en="How are poem and script seen together?",
            question_zh="书法与诗如何共同构成观看经验？",
            question_en="How do calligraphy and poetry shape looking together?",
            caption_zh="Landscape object 2 · 测试", caption_en="Landscape object 2 · test",
            hook_zh="书法", hook_en="calligraphy",
            plan_zh="我会把题字和画面放在一起看。", plan_en="I'll look at inscription and image together.",
        ),
    )
    monkeypatch.setattr(interview_featured, "FEATURED_ENTRIES", entries)
    return entries


def test_i_dont_know_is_answered_with_the_collection_not_quoted_back(client, featured):
    state = client.post("/api/interview/start").json()
    assert state["nextQuestion"]["prompt"].startswith("你好，我是“卧游”的 AI 策展人彦远。")
    assert state["nextQuestion"]["prompt"].endswith("你有什么最想了解的吗？")

    state = _answer(client, state, freeText="我不知道应该问你什么")

    reply = state["transcript"][-1]["curatorReply"]
    assert "我不知道应该问你什么" not in reply
    assert "作为主线" not in reply
    assert state["profile"]["freeFormQuestion"] is None
    question = state["nextQuestion"]
    assert question["id"] == "featured"
    assert (question["step"], question["totalSteps"]) == (2, 6)
    # The intro is counted from the loaded collection.
    assert "8 件开放藏品" in question["prompt"]
    assert "克利夫兰艺术博物馆" in question["prompt"]
    cards = [option for option in question["options"] if option["objectId"]]
    assert [card["objectId"] for card in cards] == ["TEST.1", "TEST.2"]
    assert question["options"][-1]["value"] == "__unsure__"


def test_picking_a_featured_object_sets_its_question_and_skips_negotiation(
    client, featured, monkeypatch
):
    service = client.app.state.interviews
    probed: list[str] = []
    original = service._negotiation_question

    def spy(state, collection):
        probed.append(state.profile.free_form_question or "")
        return original(state, collection)

    monkeypatch.setattr(service, "_negotiation_question", spy)
    state = client.post("/api/interview/start").json()
    state = _answer(client, state, value="__unsure__")
    assert state["transcript"][-1]["curatorReply"] == "好，那我先给你挑几件。"
    state = _answer(client, state, value="__featured__:beta")

    assert state["profile"]["freeFormQuestion"] == "书法与诗如何共同构成观看经验？"
    assert state["transcript"][-1]["answerLabel"] == "书法与诗怎样一起被看？"
    assert state["transcript"][-1]["curatorReply"].startswith("好，就从书法开始。")
    assert (state["nextQuestion"]["id"], state["nextQuestion"]["step"]) == ("motivation", 3)
    assert state["nextQuestion"]["totalSteps"] == 6

    choices = {"motivation": "explorer", "prior_knowledge": "none", "duration": "5",
               "exclusions": "none"}
    while not state["complete"]:
        question = state["nextQuestion"]
        assert question["step"] <= question["totalSteps"] <= 7
        state = _answer(client, state, value=choices[question["id"]])

    assert [turn["questionId"] for turn in state["transcript"]] == [
        "curiosity", "featured", "motivation", "prior_knowledge", "duration", "exclusions",
    ]
    # Reached the gate, which declined to re-litigate the curator's own pick.
    assert probed and set(probed) == {"书法与诗如何共同构成观看经验？"}


@pytest.mark.parametrize("answer", [{"value": "__unsure__"}, {"freeText": "还是不知道"}])
def test_still_undecided_at_the_featured_turn_lets_the_curator_choose(client, featured, answer):
    state = client.post("/api/interview/start").json()
    state = _answer(client, state, freeText="随便")
    state = _answer(client, state, **answer)

    assert state["profile"]["freeFormQuestion"] == featured[0].question_zh
    assert state["transcript"][-1]["curatorReply"].startswith("那我替你定：就从山水开始。")
    assert state["nextQuestion"]["id"] == "motivation"


def test_a_real_interest_typed_at_the_featured_turn_becomes_the_question(client, featured):
    state = client.post("/api/interview/start").json()
    state = _answer(client, state, freeText="你们有什么")
    state = _answer(client, state, freeText="我想看看宋代的瓷器")

    assert state["profile"]["freeFormQuestion"] == "我想看看宋代的瓷器"
    assert "宋代的瓷器" in state["transcript"][-1]["curatorReply"]
    assert state["nextQuestion"]["id"] == "motivation"


def test_featured_turn_ignores_a_card_that_was_never_offered(client, featured):
    state = client.post("/api/interview/start").json()
    state = _answer(client, state, freeText="不知道")
    before = len(state["transcript"])
    state = _answer(client, state, value="__featured__:invented")
    assert len(state["transcript"]) == before
    assert state["nextQuestion"]["id"] == "featured"


def test_english_featured_turn_uses_english_intro_and_cards(client, featured):
    state = client.post("/api/interview/start?language=en").json()
    assert state["nextQuestion"]["prompt"].startswith("Hello, I'm Yanyuan, Woyou's AI curator.")
    state = _answer(client, state, freeText="I don't know what to ask")

    question = state["nextQuestion"]
    assert question["id"] == "featured"
    assert "8 open-access objects from Cleveland Museum of Art" in question["prompt"]
    assert question["options"][0]["label"] == "How does a landscape route the eye?"
    state = _answer(client, state, value="__featured__:alpha")
    assert state["profile"]["freeFormQuestion"] == "How does a landscape painting route the viewer's eye?"


def test_without_featured_objects_an_undecided_opening_still_is_not_the_question(client):
    # The fixture holds none of the real featured objects.
    state = client.post("/api/interview/start").json()
    state = _answer(client, state, freeText="我不知道应该问你什么")

    assert state["profile"]["freeFormQuestion"] is None
    assert "我不知道应该问你什么" not in (state["transcript"][-1]["curatorReply"] or "")
    assert state["nextQuestion"]["id"] == "motivation"


def test_real_featured_entries_point_at_showable_objects_in_the_serving_collection():
    objects_path = (
        Path(__file__).resolve().parents[2] / "data" / "collections" / "global_open" / "objects.json"
    )
    if not objects_path.exists():
        pytest.skip("frozen global_open objects are not checked out")
    payload = json.loads(objects_path.read_text(encoding="utf-8"))
    objects = payload["objects"] if isinstance(payload, dict) else payload
    showable = {obj["id"] for obj in objects if obj.get("imageUrl")}
    assert {entry.object_id for entry in FEATURED_ENTRIES} <= showable
    for entry in FEATURED_ENTRIES:
        # The title is also the visitor's chat bubble (60-character label cap).
        assert len(entry.title_zh) <= 60 and len(entry.title_en) <= 60
        assert undecided_opening_kind(entry.question_zh) is None


def test_evidence_gap_offers_featured_questions_not_probe_cards(client, featured, monkeypatch):
    from types import SimpleNamespace

    from app.generator import ExhibitionGenerator
    from app.models import InterviewAnswer, InterviewQuestionId, InterviewState, VisitorProfile

    monkeypatch.setattr(ExhibitionGenerator, "probe_answerability", staticmethod(
        lambda _collections, _agenda, **_kwargs: SimpleNamespace(
            status="unsupported", requires_runtime_audit=False, decision_basis="lexical_retrieval",
            coverage=SimpleNamespace(evidence_domain_ids=[]),
            recommended_questions=["不同文化如何把自然景观变成关于地方、归属或世界秩序的图像？"],
        )))
    service = client.app.state.interviews
    collection = client.app.state.collections.get()
    state = InterviewState(id="gap", collectionId=collection.id,
                           profile=VisitorProfile(freeFormQuestion="古代的人用什么喝茶？", durationMinutes=5))

    question = service._negotiation_question(state, collection)

    assert question is not None
    assert "还没找到足够能对上的藏品" in question.prompt
    assert [option.label for option in question.options] == [
        "山水画怎样安排行旅的视线？", "书法与诗怎样一起被看？",
    ]
    state.next_question = question
    updated = service.answer(state, InterviewAnswer(
        questionId=InterviewQuestionId.NEGOTIATION, value=question.options[1].value))
    assert updated.profile.open_question == "书法与诗如何共同构成观看经验？"
    assert updated.transcript[-1].answer_label == "书法与诗怎样一起被看？"
    assert updated.transcript[-1].curator_reply is None or "？" not in updated.transcript[-1].curator_reply
