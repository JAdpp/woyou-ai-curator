from __future__ import annotations

import pytest

from app.interview import KEEP_SCOPE_VALUE, SCOPE_DIRECTION_PREFIX
from app.interview_clarification import needs_scope_clarification


@pytest.mark.parametrize("question", [
    "我想看看以前流行的那种东西。", "给我做个最真实的展览。",
    "我想看更特别的那一类作品", "给我看看那样的东西",
    "请给我找很壮观的展品", "我想看看当时常见的那些物件",
])
def test_generic_reference_and_subjective_only_openings_request_scope(question):
    assert needs_scope_clarification(question)


@pytest.mark.parametrize("question", [
    "我想看看画面里的打铁工人，留意他们怎样配合劳动",
    "没想好，随便带我逛逛", "给我看最古老的陶瓷",
    "给我看看最真实的人像作品", "以前流行的帽子是什么样的？",
    "给我看看那些有鹿图像的作品", "我想看玻璃杯的透明感",
    "我想看看欧洲画面里怎样表现最特别的那一类作品", "",
])
def test_concrete_subject_and_casual_browse_do_not_gain_a_question(question):
    assert not needs_scope_clarification(question)


def _answer(client, state, **answer):
    response = client.post(f"/api/interview/{state['id']}/answer",
                           json={"questionId": state["nextQuestion"]["id"], **answer})
    assert response.status_code == 200, response.text
    return response.json()


def _open(client, text="给我做个最真实的展览"):
    state = client.post("/api/interview/start").json()
    return _answer(client, state, freeText=text)


def test_scope_question_is_optional_early_and_not_overwritten_by_voice(client):
    state = _open(client)
    question = state["nextQuestion"]
    assert question["id"] == "custom_question"
    assert question["step"] == 2
    assert question["totalSteps"] == 6
    assert question["allowFreeText"] and question["skippable"]
    assert "不同方向" in question["prompt"]
    assert "我先不替你猜" in question["prompt"]
    assert "事实结论" in state["transcript"][-1]["curatorReply"]
    assert any(option["value"] == KEEP_SCOPE_VALUE for option in question["options"])


def test_scope_rewrite_preserves_original_and_moves_on_without_reasking(client):
    original = "我想看看以前流行的那种东西"
    state = _open(client, original)
    rewritten = "山水画如何组织观看者的行旅视线？"
    state = _answer(client, state, freeText=rewritten)
    assert state["profile"]["freeFormQuestion"] == original
    assert state["profile"]["openQuestion"] == rewritten
    assert state["nextQuestion"]["id"] == "motivation"
    state = _answer(client, state, value="explorer")
    assert state["nextQuestion"]["id"] == "prior_knowledge"
    assert [turn["questionId"] for turn in state["transcript"]].count("custom_question") == 1


@pytest.mark.parametrize("answer", [{"value": KEEP_SCOPE_VALUE}, {"skipped": True}])
def test_scope_can_be_left_open_without_replacing_original_or_looping(client, answer):
    state = _open(client)
    original = state["profile"]["freeFormQuestion"]
    state = _answer(client, state, **answer)
    assert state["nextQuestion"]["id"] == "motivation"
    assert state["profile"]["freeFormQuestion"] == original
    assert state["profile"]["openQuestion"] is None
    state = _answer(client, state, value="explorer")
    assert state["nextQuestion"]["id"] == "prior_knowledge"


def test_scope_rejects_unoffered_option_without_advancing(client):
    state = _open(client)
    before = len(state["transcript"])
    state = _answer(client, state, value=f"{SCOPE_DIRECTION_PREFIX}invented-domain")
    assert len(state["transcript"]) == before
    assert state["nextQuestion"]["id"] == "custom_question"


def test_explicit_domain_choice_is_marked_tentative_and_retains_evidence_negotiation(client, monkeypatch):
    service = client.app.state.interviews
    monkeypatch.setattr(service, "_available_domains", lambda _collection, _language="zh": [
        ("landscape-brush", "山水与笔墨", "已知测试馆藏方向"),
    ])
    state = _open(client)
    state = _answer(client, state, value=f"{SCOPE_DIRECTION_PREFIX}landscape-brush")
    assert state["profile"]["curiosityDomainId"] == "landscape-brush"
    assert "先从山水与笔墨试逛" in state["profile"]["openQuestion"]
    assert "不是确定事实或客观排名" in state["profile"]["openQuestion"]
    assert "negotiation" not in {turn["questionId"] for turn in state["transcript"]}


def test_clear_opening_keeps_the_existing_five_turn_route(client):
    state = _open(client, "山水画如何组织观看者的行旅视线？")
    assert state["nextQuestion"]["id"] == "motivation"
    assert state["nextQuestion"]["totalSteps"] == 5


def test_scope_and_later_evidence_negotiation_are_distinct_and_bounded(client, monkeypatch):
    from app.models import InterviewOption, InterviewQuestion, InterviewQuestionId
    service = client.app.state.interviews

    def evidence_gap(state, _collection):
        state.negotiation_note = "馆藏证据仍有缺口"
        return InterviewQuestion(id=InterviewQuestionId.NEGOTIATION, prompt="确认如何处理证据缺口",
                                 options=[InterviewOption(value="__free_text__", label="保留原话")])

    monkeypatch.setattr(service, "_negotiation_question", evidence_gap)
    state = _open(client)
    state = _answer(client, state, value=KEEP_SCOPE_VALUE)
    while not state["complete"]:
        assert len(state["transcript"]) < 7
        question = state["nextQuestion"]
        assert question["step"] <= question["totalSteps"] <= 7
        choice = {"motivation": "recharger", "prior_knowledge": "none", "duration": "5",
                  "negotiation": "__free_text__", "exclusions": "none"}[question["id"]]
        state = _answer(client, state, value=choice)
    assert [turn["questionId"] for turn in state["transcript"]] == [
        "curiosity", "custom_question", "motivation", "prior_knowledge", "duration", "negotiation", "exclusions",
    ]
