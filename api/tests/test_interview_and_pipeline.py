"""Curator interview, curation pipeline and the constraints that outlive both."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from .retrieval_contract_fixtures import strict_audit_fixture

from app import curation
from app.curation import assign_roles, chapter_sizes, ensure_core_evidence_candidate, plan_roles
from app.interview_voice import InterviewVoice
from app.jobs import STEP_DEFINITIONS
from app.models import DURATION_PLAN, EvidenceDepth, VisitorProfile


class _Obj:
    """Minimal stand-in with only what assign_roles reads."""

    def __init__(self, depth: str) -> None:
        self.evidence_depth = depth

    @property
    def supports_core_evidence(self) -> bool:
        # These role fixtures intentionally have no reviewed visual proof.
        return self.evidence_depth == EvidenceDepth.FULL.value


# ---------------------------------------------------------------- interview


def test_interview_completes_within_seven_turns_and_yields_a_usable_profile(
    client: TestClient,
) -> None:
    started = client.post("/api/interview/start")
    assert started.status_code == 200, started.text
    state = started.json()
    interview_id = state["id"]
    assert state["complete"] is False
    assert state["nextQuestion"]["id"] == "curiosity"

    turns = 0
    while not state["complete"]:
        turns += 1
        assert turns <= 7, "the interview must never exceed seven turns"
        question = state["nextQuestion"]
        assert question["options"] or question["allowFreeText"], (
            f"question {question['id']} offered no way to answer"
        )
        payload = (
            {"questionId": question["id"], "freeText": "山水画如何组织观看者的行旅视线？"}
            if question["id"] == "custom_question"
            else {"questionId": question["id"], "value": question["options"][0]["value"]}
        )
        answered = client.post(
            f"/api/interview/{interview_id}/answer",
            json=payload,
        )
        assert answered.status_code == 200, answered.text
        state = answered.json()

    profile = state["profile"]
    assert profile["motivation"] in {"explorer", "recharger", "facilitator", "professional"}
    assert profile["durationMinutes"] in {5, 10, 15}
    assert len(state["transcript"]) == turns


def _answer_interview(
    client: TestClient, state: dict[str, Any], **answer: object
) -> dict[str, Any]:
    response = client.post(
        f"/api/interview/{state['id']}/answer",
        json={"questionId": state["nextQuestion"]["id"], **answer},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_provisional_audit_is_mentioned_not_offered_as_a_choice(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.generator import ExhibitionGenerator
    from app.interview import InterviewService
    from app.models import InterviewQuestionId, InterviewState, InterviewTurn

    repository = client.app.state.collections
    collection = repository.get()
    service = InterviewService(repository, audit_available=True)
    monkeypatch.setattr(
        service,
        "_available_domains",
        lambda _collection, _language="zh": [
            ("global:daily-life", "日常生活", "饮食、居家与劳动 · 20 件")
        ],
    )
    monkeypatch.setattr(
        ExhibitionGenerator,
        "probe_answerability",
        staticmethod(
            lambda _collections, _agenda, **_kwargs: SimpleNamespace(
                status="supported",
                requires_runtime_audit=True,
                coverage=SimpleNamespace(
                    evidence_domain_ids=["global:daily-life"]
                ),
                recommended_questions=[],
            )
        ),
    )
    original = "镜子如何改变不同文化中的自我观看？"
    state = InterviewState(
        id="provisional",
        collectionId=collection.id,
        profile=VisitorProfile(freeFormQuestion=original, durationMinutes=5),
    )

    # Candidates still await the per-object audit, but that runs during
    # curation regardless; the visitor has nothing to decide about it.
    assert service._negotiation_question(state, collection) is None
    assert "逐件核对" in (state.negotiation_note or "")

    state.negotiation_note = None
    state.transcript = [
        InterviewTurn(questionId=question_id, prompt="")
        for question_id in (
            InterviewQuestionId.CURIOSITY,
            InterviewQuestionId.MOTIVATION,
            InterviewQuestionId.PRIOR_KNOWLEDGE,
            InterviewQuestionId.DURATION,
        )
    ]
    last = service._next(state, collection)
    assert last is not None and last.id == InterviewQuestionId.EXCLUSIONS
    # Said in passing with the last question, which is still step 5 of 5.
    assert last.prompt.startswith("我在馆藏里先找到了一批可能相关的藏品")
    assert last.prompt.endswith("最后一个：有什么是你不太想看到的？")
    assert (last.step, last.total_steps) == (5, 5)
    assert state.profile.to_agenda(collection.id).question == original


def test_negotiation_changes_the_active_retrieval_question(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.generator import AgenticRetrievalOutcome, ExhibitionGenerator
    from app.interview import InterviewService
    from app.models import (
        InterviewAnswer,
        InterviewQuestionId,
        InterviewState,
    )

    repository = client.app.state.collections
    collection = repository.get()
    service = InterviewService(repository)
    domains = [("global:daily-life", "日常生活", "饮食、居家与劳动 · 20 件")]
    monkeypatch.setattr(
        service,
        "_available_domains",
        lambda _collection, _language="zh": domains,
    )
    monkeypatch.setattr(
        ExhibitionGenerator,
        "probe_answerability",
        staticmethod(
            lambda _collections, _agenda, **_kwargs: SimpleNamespace(
                status="unsupported",
                requires_runtime_audit=False,
                coverage=SimpleNamespace(
                    evidence_domain_ids=["global:daily-life"]
                ),
                recommended_questions=[],
            )
        ),
    )

    def new_state(identifier: str) -> InterviewState:
        return InterviewState(
            id=identifier,
            collectionId=collection.id,
            profile=VisitorProfile(
                freeFormQuestion="原来无法支持的问题",
                durationMinutes=5,
            ),
        )

    narrowed = new_state("narrowed")
    narrowed.next_question = service._negotiation_question(narrowed, collection)
    assert narrowed.next_question is not None
    narrowed = service.answer(
        narrowed,
        InterviewAnswer(
            questionId=InterviewQuestionId.NEGOTIATION,
            value="global:daily-life",
        ),
    )
    assert narrowed.profile.to_agenda(collection.id).question == "日常生活"

    rewritten = new_state("rewritten")
    rewritten.next_question = service._negotiation_question(rewritten, collection)
    assert rewritten.next_question is not None
    assert rewritten.next_question.allow_free_text is True
    revised_question = "梳妆器物在不同地区怎样进入日常生活？"
    rewritten = service.answer(
        rewritten,
        InterviewAnswer(
            questionId=InterviewQuestionId.NEGOTIATION,
            freeText=revised_question,
        ),
    )
    assert rewritten.profile.to_agenda(collection.id).question == revised_question


def test_out_of_domain_negotiation_does_not_offer_unrelated_rich_domains(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.generator import ExhibitionGenerator
    from app.interview import InterviewService, RECOMMENDED_QUESTION_PREFIX
    from app.models import InterviewAnswer, InterviewQuestionId, InterviewState

    repository = client.app.state.collections
    collection = repository.get()
    service = InterviewService(repository, audit_available=False)
    monkeypatch.setattr(
        service,
        "_available_domains",
        lambda _collection, _language="zh": [
            ("global:daily-life", "日常生活", "馆藏很多"),
            ("global:making-material", "材料与制作", "馆藏也很多"),
        ],
    )
    reviewed = "镜子如何进入不同文化的日常生活？"
    monkeypatch.setattr(
        ExhibitionGenerator,
        "probe_answerability",
        staticmethod(
            lambda _collections, _agenda, **_kwargs: SimpleNamespace(
                status="unsupported",
                requires_runtime_audit=False,
                coverage=SimpleNamespace(evidence_domain_ids=[]),
                recommended_questions=[reviewed],
            )
        ),
    )
    state = InterviewState(
        id="ood",
        collectionId=collection.id,
        profile=VisitorProfile(
            freeFormQuestion="量子纠错如何改变帝国权力？",
            durationMinutes=5,
        ),
    )

    question = service._negotiation_question(state, collection)

    assert question is not None
    assert {option.value for option in question.options}.isdisjoint(
        {"global:daily-life", "global:making-material"}
    )
    assert "不想拿不相干的东西凑数" in question.prompt
    assert question.options[0].value == f"{RECOMMENDED_QUESTION_PREFIX}{reviewed}"

    state.next_question = question
    updated = service.answer(
        state,
        InterviewAnswer(
            questionId=InterviewQuestionId.NEGOTIATION,
            value=question.options[0].value,
        ),
    )
    assert updated.profile.to_agenda(collection.id).question == reviewed


def test_negotiation_only_offers_domains_overlapping_current_evidence(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.generator import ExhibitionGenerator
    from app.interview import InterviewService
    from app.models import InterviewState

    repository = client.app.state.collections
    collection = repository.get()
    service = InterviewService(repository)
    monkeypatch.setattr(
        service,
        "_available_domains",
        lambda _collection, _language="zh": [
            ("global:daily-life", "日常生活", "馆藏很多"),
            ("global:making-material", "材料与制作", "馆藏也很多"),
        ],
    )
    monkeypatch.setattr(
        ExhibitionGenerator,
        "probe_answerability",
        staticmethod(
            lambda _collections, _agenda, **_kwargs: SimpleNamespace(
                status="partially_supported",
                requires_runtime_audit=False,
                coverage=SimpleNamespace(
                    evidence_domain_ids=["global:making-material"]
                ),
                recommended_questions=[],
            )
        ),
    )
    state = InterviewState(
        id="partial",
        collectionId=collection.id,
        profile=VisitorProfile(
            freeFormQuestion="漆器表面怎样体现跨文化交流？",
            durationMinutes=5,
        ),
    )

    question = service._negotiation_question(state, collection)

    assert question is not None
    assert [option.value for option in question.options] == [
        "global:making-material"
    ]


def test_interview_never_promises_runtime_audit_when_capability_is_unavailable(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.generator import ExhibitionGenerator
    from app.interview import InterviewService
    from app.models import InterviewState

    repository = client.app.state.collections
    collection = repository.get()
    service = InterviewService(repository, audit_available=False)
    seen: dict[str, bool] = {}

    def probe(_collections, _agenda, *, audit_available=False):
        seen["audit_available"] = audit_available
        return SimpleNamespace(
            status="unsupported",
            requires_runtime_audit=False,
            decision_basis="audit_unavailable",
            coverage=SimpleNamespace(evidence_domain_ids=[]),
            recommended_questions=[],
        )

    monkeypatch.setattr(
        ExhibitionGenerator,
        "probe_answerability",
        staticmethod(probe),
    )
    state = InterviewState(
        id="no-audit",
        collectionId=collection.id,
        profile=VisitorProfile(
            freeFormQuestion="镜子怎样改变自我观看？",
            durationMinutes=5,
        ),
    )

    question = service._negotiation_question(state, collection)

    assert seen == {"audit_available": False}
    assert question is not None
    assert all(option.label != "按原问题做语义核查" for option in question.options)
    # An audit outage is not an empty collection, and must not be told as one.
    assert "服务暂时连不上" in question.prompt
    assert "没找到" not in question.prompt


def test_initial_free_text_question_is_not_asked_again_even_for_explorer(
    client: TestClient,
) -> None:
    question_text = "山水画如何组织观看者的行旅视线？"
    state = client.post("/api/interview/start").json()
    state = _answer_interview(client, state, freeText=question_text)
    assert state["nextQuestion"]["id"] == "motivation"
    assert state["nextQuestion"]["step"] == 2
    assert state["nextQuestion"]["totalSteps"] == 5

    # Even the motivation most likely to ask for a question must recognise
    # that the visitor already supplied one in the opening turn.
    state = _answer_interview(client, state, value="explorer")
    seen = ["curiosity", "motivation"]
    steps = [1, 2]
    while not state["complete"]:
        question = state["nextQuestion"]
        seen.append(question["id"])
        steps.append(question["step"])
        value = question["options"][0]["value"]
        state = _answer_interview(client, state, value=value)

    assert seen == [
        "curiosity",
        "motivation",
        "prior_knowledge",
        "duration",
        "exclusions",
    ]
    assert steps == [1, 2, 3, 4, 5]
    assert state["profile"]["freeFormQuestion"] == question_text
    assert state["profile"]["openQuestion"] is None
    profile = VisitorProfile.model_validate(state["profile"])
    assert profile.to_agenda(state["collectionId"]).question == question_text


def test_explorer_is_asked_for_the_question_immediately_and_only_once(
    client: TestClient,
) -> None:
    question_text = "山水画如何组织观看者的行旅视线？"
    state = client.post("/api/interview/start").json()
    state = _answer_interview(client, state, value="__unsure__")
    state = _answer_interview(client, state, value="explorer")

    custom = state["nextQuestion"]
    assert custom["id"] == "custom_question"
    assert custom["prompt"] == "这个问题是？"
    assert custom["options"] == []
    assert custom["allowFreeText"] is True
    assert custom["skippable"] is False
    assert custom["step"] == 3
    assert custom["totalSteps"] == 6

    # An empty direct API submission must not create a turn or fall through to
    # the generic question later in the interview.
    unchanged = _answer_interview(client, state, freeText="   ")
    assert unchanged["nextQuestion"]["id"] == "custom_question"
    assert len(unchanged["transcript"]) == 2

    state = _answer_interview(client, unchanged, freeText=question_text)
    assert state["nextQuestion"]["id"] == "prior_knowledge"
    assert state["nextQuestion"]["prompt"] == "对这个问题涉及的主题，你现在了解多少？"
    progress = [("curiosity", 1, 5), ("motivation", 2, 6), ("custom_question", 3, 6)]
    while not state["complete"]:
        question = state["nextQuestion"]
        progress.append((question["id"], question["step"], question["totalSteps"]))
        state = _answer_interview(
            client, state, value=question["options"][0]["value"]
        )

    ids = [turn["questionId"] for turn in state["transcript"]]
    assert ids == [
        "curiosity",
        "motivation",
        "custom_question",
        "prior_knowledge",
        "duration",
        "exclusions",
    ]
    assert ids.count("custom_question") == 1
    assert "open_question" not in ids
    assert progress == [
        ("curiosity", 1, 5),
        ("motivation", 2, 6),
        ("custom_question", 3, 6),
        ("prior_knowledge", 4, 6),
        ("duration", 5, 6),
        ("exclusions", 6, 6),
    ]
    assert state["profile"]["openQuestion"] == question_text
    profile = VisitorProfile.model_validate(state["profile"])
    assert profile.to_agenda(state["collectionId"]).question == question_text


def test_custom_question_is_not_replaced_by_model_suggestions(
    client: TestClient, monkeypatch
) -> None:
    calls: list[bool] = []

    def fake_compose(**kwargs: object) -> InterviewVoice:
        calls.append(bool(kwargs["want_suggestions"]))
        return InterviewVoice(
            reply="我会先记住你刚才选的方向。",
            suggestions=("无关问题一？", "无关问题二？", "无关问题三？"),
        )

    monkeypatch.setattr("app.main.interview_voice.compose_immediate", fake_compose)

    state = client.post("/api/interview/start").json()
    state = _answer_interview(client, state, value="__unsure__")
    state = _answer_interview(client, state, value="explorer")

    assert calls == [False, False]
    assert state["nextQuestion"]["id"] == "custom_question"
    assert state["nextQuestion"]["prompt"] == "这个问题是？"
    assert state["nextQuestion"]["options"] == []


def test_configured_provider_is_not_called_while_advancing_the_interview(
    client: TestClient, monkeypatch
) -> None:
    provider = client.app.state.generator.provider
    captured: list[bool] = []

    async def fake_generate_json(
        _system_prompt: str, user_payload: dict[str, object]
    ) -> dict[str, object]:
        captured.append(bool(user_payload["wantSuggestions"]))
        return {
            "reply": "我会沿着你刚才选的方向继续。",
            "suggestions": ["无关问题一？", "无关问题二？", "无关问题三？"],
        }

    monkeypatch.setattr(provider, "api_key", "configured-test-key")
    monkeypatch.setattr(provider, "generate_json", fake_generate_json)

    state = client.post("/api/interview/start").json()
    state = _answer_interview(client, state, value="__unsure__")
    state = _answer_interview(client, state, value="explorer")

    assert captured == []
    assert state["nextQuestion"]["id"] == "custom_question"
    assert state["nextQuestion"]["prompt"] == "这个问题是？"
    assert state["nextQuestion"]["options"] == []


def test_custom_question_has_an_english_input_only_variant(
    client: TestClient,
) -> None:
    state = client.post("/api/interview/start?language=en").json()
    state = _answer_interview(client, state, value="__unsure__")
    state = _answer_interview(client, state, value="explorer")

    custom = state["nextQuestion"]
    assert custom["id"] == "custom_question"
    assert custom["prompt"] == "What is the question?"
    assert custom["options"] == []
    assert custom["allowFreeText"] is True

    state = _answer_interview(
        client,
        state,
        freeText="Why are dogs companions in some cultures and guardians in others?",
    )
    assert state["nextQuestion"]["id"] == "prior_knowledge"
    assert state["nextQuestion"]["prompt"] == (
        "How much do you already know about the subject behind that question?"
    )


def test_custom_question_replaces_an_auto_recommended_domain(
    client: TestClient, monkeypatch
) -> None:
    from app import interview

    monkeypatch.setattr(interview, "MIN_DOMAIN_OBJECTS", 1)
    collection = client.app.state.collections.get()
    for obj in collection.objects:
        obj.evidence_domain_ids = ["global:making-material"]

    state = client.post("/api/interview/start").json()
    state = _answer_interview(client, state, value="__unsure__")
    assert state["profile"]["curiosityDomainId"] == "global:making-material"
    state = _answer_interview(client, state, value="explorer")
    state = _answer_interview(
        client,
        state,
        freeText="狗在各国文化中为什么既是伙伴，又是守护者？",
    )

    # The domain selected by "you choose" was only a temporary default. It
    # must not constrain a later visitor-authored question about something
    # else, or retrieval will faithfully answer the wrong topic.
    assert state["profile"]["curiosityDomainId"] is None
    assert state["profile"]["curiosityLabel"] == ""
    assert state["nextQuestion"]["prompt"] == "对这个问题涉及的主题，你现在了解多少？"


def test_later_written_question_also_replaces_an_auto_recommended_domain(
    client: TestClient, monkeypatch
) -> None:
    from app import interview

    monkeypatch.setattr(interview, "MIN_DOMAIN_OBJECTS", 1)
    collection = client.app.state.collections.get()
    for obj in collection.objects:
        obj.evidence_domain_ids = ["global:making-material"]

    state = client.post("/api/interview/start").json()
    for value in ("__unsure__", "recharger", "none", "5"):
        state = _answer_interview(client, state, value=value)
    assert state["nextQuestion"]["id"] == "open_question"
    assert state["profile"]["curiosityDomainId"] == "global:making-material"

    state = _answer_interview(
        client,
        state,
        freeText="狗在各国文化中为什么既是伙伴，又是守护者？",
    )
    assert state["profile"]["curiosityDomainId"] is None
    assert state["profile"]["curiosityLabel"] == ""


def test_clicking_a_suggested_question_keeps_its_auto_recommended_context(
    client: TestClient, monkeypatch
) -> None:
    from app import interview

    monkeypatch.setattr(interview, "MIN_DOMAIN_OBJECTS", 1)
    collection = client.app.state.collections.get()
    for obj in collection.objects:
        obj.evidence_domain_ids = ["global:making-material"]

    state = client.post("/api/interview/start").json()
    for value in ("__unsure__", "recharger", "none", "5"):
        state = _answer_interview(client, state, value=value)
    suggestion = state["nextQuestion"]["options"][0]["value"]
    state = _answer_interview(client, state, value=suggestion)

    assert state["profile"]["openQuestion"] == suggestion
    assert state["profile"]["curiosityDomainId"] == "global:making-material"


def test_slow_model_provider_cannot_delay_the_next_interview_question(
    client: TestClient, monkeypatch
) -> None:
    import asyncio

    state = client.post("/api/interview/start").json()
    for value in ("__unsure__", "recharger", "none"):
        state = _answer_interview(client, state, value=value)
    assert state["nextQuestion"]["id"] == "duration"

    provider = client.app.state.generator.provider
    called = False

    async def delayed_generate_json(*_args: object, **_kwargs: object) -> dict:
        nonlocal called
        called = True
        await asyncio.sleep(1)
        return {"reply": "不应出现", "suggestions": []}

    monkeypatch.setattr(provider, "api_key", "configured-test-key")
    monkeypatch.setattr(provider, "generate_json", delayed_generate_json)
    started = time.perf_counter()
    state = _answer_interview(client, state, value="5")

    assert time.perf_counter() - started < 0.25
    assert called is False
    assert state["nextQuestion"]["id"] == "open_question"
    assert len(state["nextQuestion"]["options"]) == 4


def test_curator_replies_keep_dog_and_cobalt_questions_in_context(
    client: TestClient,
    monkeypatch,
) -> None:
    from app import interview, interview_voice

    monkeypatch.setattr(interview, "MIN_DOMAIN_OBJECTS", 1)
    collection = client.app.state.collections.get()
    for obj in collection.objects:
        obj.evidence_domain_ids = ["global:making-material"]
    original_compose = interview_voice.compose_immediate
    seen_questions: list[str] = []

    def capture_context(**kwargs: object) -> InterviewVoice:
        seen_questions.append(str(kwargs["visitor_question"]))
        return original_compose(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(interview_voice, "compose_immediate", capture_context)
    examples = (
        ("狗在各国文化中为什么既是伙伴，又是守护者？", "狗"),
        (
            "相似的钴蓝为什么会出现在中国瓷器、伊朗陶器和代尔夫特陶器上？",
            "钴蓝",
        ),
    )

    for question, keyword in examples:
        before = len(seen_questions)
        state = client.post("/api/interview/start").json()
        state = _answer_interview(client, state, freeText=question)
        assert keyword in state["transcript"][-1]["curatorReply"]

        # Later replies answer the choice just made instead of reading the
        # question back every turn, but the complete visitor-authored question
        # stays their source context and an inferred domain never replaces it.
        state = _answer_interview(client, state, value="professional")
        assert "术语" in state["transcript"][-1]["curatorReply"]
        state = _answer_interview(client, state, value="some")
        assert "常识" in state["transcript"][-1]["curatorReply"]
        assert seen_questions[before:] == [question, question, question]
        assert all(
            "材料与制作" not in (turn["curatorReply"] or "")
            for turn in state["transcript"]
        )


def test_non_explorer_topic_path_keeps_one_later_open_question(
    client: TestClient,
) -> None:
    state = client.post("/api/interview/start").json()
    state = _answer_interview(client, state, value="__unsure__")
    state = _answer_interview(client, state, value="recharger")
    assert state["nextQuestion"]["id"] == "prior_knowledge"
    state = _answer_interview(
        client, state, value=state["nextQuestion"]["options"][0]["value"]
    )
    state = _answer_interview(client, state, value="5")

    assert state["nextQuestion"]["id"] == "open_question"
    assert state["nextQuestion"]["prompt"].startswith("关于")
    assert state["nextQuestion"]["step"] == 5
    assert all(
        turn["questionId"] != "custom_question" for turn in state["transcript"]
    )


def test_curiosity_options_only_offer_domains_the_corpus_can_route(
    client: TestClient,
) -> None:
    state = client.post("/api/interview/start").json()
    options = state["nextQuestion"]["options"]
    # The fixture corpus has no coverage domains assigned, so the only honest
    # answer is the "recommend something" escape hatch — never a fabricated
    # topic the collection cannot support.
    assert options[-1]["value"] == "__unsure__"
    assert all(option["hint"] for option in options[:-1])


def test_stale_answers_are_ignored_rather_than_corrupting_the_profile(
    client: TestClient,
) -> None:
    state = client.post("/api/interview/start").json()
    interview_id = state["id"]
    # Reply to a question the state machine is not currently on.
    replayed = client.post(
        f"/api/interview/{interview_id}/answer",
        json={"questionId": "duration", "value": "15"},
    )
    assert replayed.status_code == 200
    assert replayed.json()["nextQuestion"]["id"] == "curiosity"
    assert replayed.json()["transcript"] == []


def test_duration_question_describes_a_continuous_line_not_separate_halls(
    client: TestClient,
) -> None:
    state = client.post("/api/interview/start").json()
    interview_id = state["id"]
    for value in ("__unsure__", "recharger", "none"):
        question = state["nextQuestion"]
        state = client.post(
            f"/api/interview/{interview_id}/answer",
            json={"questionId": question["id"], "value": value},
        ).json()

    question = state["nextQuestion"]
    assert question["id"] == "duration"
    assert question["prompt"] == "你打算待多久？我按这个来安排展线的长短和节奏。"
    assert [option["hint"] for option in question["options"]] == [
        "5 件展品 · 2 个叙事区段",
        "8 件展品 · 3 个叙事区段",
        "12 件展品 · 4 个叙事区段",
    ]


def test_pipeline_copy_treats_chapters_as_segments_of_one_continuous_line() -> None:
    steps = {key: (title, detail) for key, title, detail in STEP_DEFINITIONS}
    assert steps["chapters"][1] == "把展品编排成连续展线上的叙事区段"
    assert steps["space"][1] == "用连续动线连接叙事区段、灯光与画框"
    assert all("几个展厅" not in detail for _title, detail in steps.values())


# ------------------------------------------------------------ role planning


def test_every_exhibition_size_keeps_a_contrast_voice() -> None:
    """Personalisation must never collapse into a familiarity filter bubble."""
    for count in (5, 8, 12):
        roles = assign_roles([_Obj(EvidenceDepth.FULL.value) for _ in range(count)])
        assert len(roles) == count
        assert "contrast" in roles, f"{count} items lost the contrast role"
        assert roles[0] == "opening"
        assert roles[-1] == "synthesis"


def test_thin_evidence_objects_never_hold_the_core_evidence_role() -> None:
    for count in (5, 8, 12):
        objects = [
            _Obj(EvidenceDepth.THIN.value if index % 2 else EvidenceDepth.FULL.value)
            for index in range(count)
        ]
        roles = assign_roles(objects)
        for obj, role in zip(objects, roles, strict=True):
            assert not (
                obj.evidence_depth == EvidenceDepth.THIN.value and role == "core_evidence"
            )


def test_all_thin_corpus_reports_rather_than_fakes_core_evidence() -> None:
    """Worst case: every object is metadata-only (an all-Met selection).

    No object can honestly carry core evidence, so the plan omits the role and
    lets the validator surface it, rather than assigning it anyway.
    """
    roles = assign_roles([_Obj(EvidenceDepth.THIN.value) for _ in range(8)])
    assert "contrast" in roles
    assert "core_evidence" not in roles


def test_a_single_full_depth_object_is_promoted_to_carry_core_evidence() -> None:
    """A short hall has one core slot, and it may land on a thin object."""
    for count in (5, 8, 12):
        objects = [_Obj(EvidenceDepth.THIN.value) for _ in range(count)]
        # Only one object in the whole selection has institution prose.
        objects[count - 2] = _Obj(EvidenceDepth.FULL.value)
        ordered, roles = plan_roles(objects)
        assert "core_evidence" in roles, f"{count} items lost core evidence"
        core_index = roles.index("core_evidence")
        assert ordered[core_index].evidence_depth == EvidenceDepth.FULL.value


def test_structural_roles_survive_when_only_a_framing_slot_has_prose() -> None:
    """The lone full-depth object sitting first or last is the hard case.

    Reassigning its role would cost the exhibition its opening or its synthesis,
    so the object is moved inward instead and all five roles survive.
    """
    for count in (5, 8, 12):
        for donor in (0, count - 1):
            objects = [_Obj(EvidenceDepth.THIN.value) for _ in range(count)]
            objects[donor] = _Obj(EvidenceDepth.FULL.value)
            ordered, roles = plan_roles(objects)
            assert set(roles) >= {"opening", "core_evidence", "contrast", "synthesis"}, (
                f"count={count} donor={donor} lost a structural role"
            )
            assert ordered[roles.index("core_evidence")].evidence_depth == EvidenceDepth.FULL.value
            assert len(ordered) == count


def test_selection_swaps_in_a_full_depth_object_when_the_top_hits_are_all_thin() -> None:
    """Several coverage domains are dominated by a prose-free institution."""

    class _Result:
        def __init__(self, obj: _Obj, identifier: str) -> None:
            self.obj = obj
            self.obj.id = identifier  # type: ignore[attr-defined]

    thin = [_Obj(EvidenceDepth.THIN.value) for _ in range(5)]
    for index, obj in enumerate(thin):
        obj.id = f"thin-{index}"  # type: ignore[attr-defined]
    full = _Obj(EvidenceDepth.FULL.value)
    full.id = "full-0"  # type: ignore[attr-defined]

    pool = [_Result(obj, obj.id) for obj in [*thin, full]]  # type: ignore[attr-defined]
    result = ensure_core_evidence_candidate(list(thin), pool)
    assert any(obj.evidence_depth == EvidenceDepth.FULL.value for obj in result)
    assert len(result) == len(thin)


def test_selection_is_untouched_when_no_full_depth_object_exists_at_all() -> None:
    class _Result:
        def __init__(self, obj: _Obj) -> None:
            self.obj = obj

    thin = [_Obj(EvidenceDepth.THIN.value) for _ in range(5)]
    for index, obj in enumerate(thin):
        obj.id = f"thin-{index}"  # type: ignore[attr-defined]
    result = ensure_core_evidence_candidate(list(thin), [_Result(obj) for obj in thin])
    assert result == thin


def test_chapter_sizes_partition_every_item() -> None:
    for duration, (items, chapters, _pace, _chars) in DURATION_PLAN.items():
        sizes = chapter_sizes(items, chapters)
        assert len(sizes) == chapters, duration
        assert sum(sizes) == items, duration
        assert all(size > 0 for size in sizes), duration


def test_duration_drives_visit_size_and_label_budget() -> None:
    for minutes, expected_items in ((5, 5), (10, 8), (15, 12)):
        profile = VisitorProfile(durationMinutes=minutes)
        assert profile.item_count == expected_items
        assert profile.chapter_count >= 1
        assert profile.pace in {"grasshopper", "butterfly", "ant"}
    # Serrell (1997): a five-minute visitor will not read a 220-character label.
    assert (
        VisitorProfile(durationMinutes=5).label_max_chars
        < VisitorProfile(durationMinutes=15).label_max_chars
    )


# --------------------------------------------------------------- pipeline


PIPELINE_FIXTURE_QUESTION = "山水画如何组织观看者的行旅视线？"


def _run_pipeline(client: TestClient) -> dict[str, Any]:
    state = client.post("/api/interview/start").json()
    interview_id = state["id"]
    while not state["complete"]:
        question = state["nextQuestion"]
        # Generic open-question suggestions are deliberately corpus-agnostic.
        # This end-to-end fixture contains only landscape records, so anchor
        # the visit to the fixture's reviewed question instead of assuming the
        # first generic suggestion is retrieval-compatible.
        payload = (
            {"questionId": question["id"], "freeText": PIPELINE_FIXTURE_QUESTION}
            if question["id"] in {"custom_question", "open_question"}
            else {"questionId": question["id"], "value": question["options"][0]["value"]}
        )
        state = client.post(
            f"/api/interview/{interview_id}/answer",
            json=payload,
        ).json()

    created = client.post("/api/exhibitions/generate", json={"interviewId": interview_id})
    assert created.status_code == 202, created.text
    job_id = created.json()["id"]

    for _ in range(80):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.05)
    raise AssertionError("curation job did not settle")


def test_pipeline_reports_a_concrete_finding_for_every_step(client: TestClient) -> None:
    job = _run_pipeline(client)
    assert job["status"] == "completed", job.get("error")
    assert job["exhibitionId"]

    settled = [step for step in job["steps"] if step["status"] in {"done", "skipped"}]
    assert len(settled) == len(job["steps"]), "every step must reach a terminal state"

    # The point of a visible pipeline is that each step says what it found.
    reported = [step for step in job["steps"] if step["finding"]]
    assert len(reported) >= len(job["steps"]) - 1

    by_key = {step["key"]: step for step in job["steps"]}
    assert "连续展线分为" in by_key["profile"]["finding"]
    assert "个叙事区段" in by_key["profile"]["finding"]
    assert "一条连续展线" in by_key["space"]["finding"]
    assert "个叙事区段" in by_key["space"]["finding"]
    assert "个展厅" not in by_key["profile"]["finding"]
    assert "个展厅" not in by_key["space"]["finding"]


def test_profile_generation_can_recover_sparse_recall_with_agentic_expansion(
    client: TestClient,
    monkeypatch,
) -> None:
    import asyncio
    from dataclasses import replace

    from app.generator import AGENTIC_RETRIEVAL_METHOD, ExhibitionGenerator
    from app.models import VisitorProfile

    repository = client.app.state.collections
    collection = repository.get()
    question = "山水画中的空间怎样引导观看？"
    profile = VisitorProfile(
        freeFormQuestion=question,
        curiosityLabel=question,
        durationMinutes=5,
    )
    agenda = profile.to_agenda(collection.id)
    original_search = repository.search
    full_results = original_search(agenda, collection)
    assert len(full_results) >= 5
    expansion_query = "landscape painting spatial viewing"
    seen_queries: list[str] = []

    def sparse_then_expanded(search_agenda, search_collection):
        seen_queries.append(search_agenda.question)
        if search_agenda.question == expansion_query:
            return full_results
        return full_results[:2]

    def sparse_many(search_agendas, search_collection, **_kwargs):
        agendas = list(search_agendas)
        seen_queries.extend(search_agenda.question for search_agenda in agendas)
        return [
            full_results
            if search_agenda.question == expansion_query
            else full_results[:2]
            for search_agenda in agendas
        ]

    monkeypatch.setattr(repository, "search", sparse_then_expanded)
    monkeypatch.setattr(repository, "search_many", sparse_many)

    class Provider:
        configured = True
        supports_retrieval_audit = True
        supports_vision = False

        def __init__(self) -> None:
            self.audit_calls = 0

        @strict_audit_fixture
        async def generate_json(self, _prompt: str, payload: dict) -> dict:
            if "candidates" not in payload:
                # The later frame/label stages deliberately fall back to the
                # deterministic exhibition; this test isolates retrieval.
                return {}
            self.audit_calls += 1
            candidates = payload["candidates"]
            accept_count = 2 if self.audit_calls == 1 else 5
            return {
                "queryInterpretation": "山水空间与观看",
                "answerability": "supported",
                "accepted": [
                    {
                        "objectId": candidate["objectId"],
                        "relevanceScore": 0.9,
                        "evidenceIds": [candidate["evidence"][0]["id"]],
                    }
                    for candidate in candidates[:accept_count]
                ],
                "searchQueries": [expansion_query]
                if self.audit_calls == 1
                else [],
                "coverageGap": "",
            }

    provider = Provider()
    generator = ExhibitionGenerator(
        replace(
            client.app.state.settings,
            rag_llm_audit_enabled=True,
            rag_llm_audit_timeout_seconds=8.0,
        ),
        repository,
        provider=provider,  # type: ignore[arg-type]
    )

    exhibition = asyncio.run(
        generator.generate_from_profile(profile, collection_id=collection.id)
    )

    assert provider.audit_calls == 2
    assert expansion_query in seen_queries
    assert len(exhibition.items) == 5
    assert exhibition.curatorial_brief is not None
    assert exhibition.curatorial_brief.retrieval.method == AGENTIC_RETRIEVAL_METHOD

    # The public legacy sync endpoint must use the same recoverable retrieval
    # gate. It used to reject the initial two hits before the agent could issue
    # the expansion that supplies the five-object evidence chain.
    legacy_provider = Provider()
    legacy_generator = ExhibitionGenerator(
        replace(
            client.app.state.settings,
            rag_llm_audit_enabled=True,
            rag_llm_audit_timeout_seconds=8.0,
        ),
        repository,
        provider=legacy_provider,  # type: ignore[arg-type]
    )
    legacy_exhibition = asyncio.run(legacy_generator.generate(agenda))

    assert legacy_provider.audit_calls == 2
    assert len(legacy_exhibition.items) == 5
    assert {item.object.id for item in legacy_exhibition.items} <= {
        result.obj.id for result in full_results
    }


def test_generated_exhibition_has_chapters_epilogue_and_space_design(
    client: TestClient,
) -> None:
    job = _run_pipeline(client)
    exhibition = client.get(f"/api/exhibitions/{job['exhibitionId']}").json()

    assert exhibition["status"] == "ready"
    assert exhibition["chapters"], "a continuous exhibition line needs narrative segments"
    assert exhibition["epilogue"]["text"]
    assert exhibition["epilogue"]["materialBoundary"]
    assert exhibition["spaceDesign"]["wallColor"].startswith("#")

    # Every item belongs to exactly one narrative segment on the continuous line.
    chaptered = [item_id for chapter in exhibition["chapters"] for item_id in chapter["itemIds"]]
    assert sorted(chaptered) == sorted(item["id"] for item in exhibition["items"])
    assert len(chaptered) == len(set(chaptered))

    assert "contrast" in {item["role"] for item in exhibition["items"]}
    assert exhibition["validation"]["passed"] is True


def test_generate_without_a_profile_or_interview_is_rejected(client: TestClient) -> None:
    assert client.post("/api/exhibitions/generate", json={}).status_code == 422


def test_job_lookup_for_an_unknown_id_is_a_404(client: TestClient) -> None:
    assert client.get("/api/jobs/does-not-exist").status_code == 404


# ------------------------------------------------------------ model pass

# The pipeline tests above run with no model configured, so the whole
# model-facing path (payload construction, prompt formatting, output folding)
# never executed under test. A missing module constant there reached runtime as
# a failed job. These exercise it with a stub response instead.


def _plan_and_items(client: TestClient):
    """Build a real skeleton exhibition to feed the model helpers."""
    from app import curation
    from app.collections import CollectionRepository
    from app.config import Settings
    from app.generator import ExhibitionGenerator
    from app.models import VisitorProfile

    settings: Settings = client.app.state.settings
    repository: CollectionRepository = client.app.state.collections
    generator = ExhibitionGenerator(settings, repository)
    profile = VisitorProfile(durationMinutes=5)

    agenda = profile.to_agenda(None)
    collection = repository.get()
    results = repository.search(agenda, collection)
    objects = curation.order_for_narrative(results, profile.item_count)
    objects, roles = curation.plan_roles(objects)
    exhibition = generator._profile_skeleton(
        profile, agenda, collection, objects, None, results
    )
    plan = curation.CurationPlan(
        profile=profile,
        objects=objects,
        roles=roles,
        chapter_sizes=[len(c.item_ids) for c in exhibition.chapters],
        evidence_domain_id=None,
        collection_id=collection.id,
        collection_version=collection.version,
        institution=collection.institution,
        candidate_count=len(results),
    )
    return curation, exhibition, plan, profile


def test_prompts_render_without_leftover_placeholders() -> None:
    """The prompts embed a JSON skeleton, so brace handling is easy to break."""
    from app import curation

    frame = curation.frame_prompt()
    assert '"chapters"' in frame and '"spaceDesign"' in frame
    assert "LABEL_MAX" not in frame

    for budget in (80, 140, 220):
        labels = curation.labels_prompt(budget)
        assert str(budget) in labels
        assert "LABEL_MAX" not in labels
        assert '"labelSentences"' in labels


def test_frame_payload_carries_only_bounded_evidence(client: TestClient) -> None:
    """Frame claims need source text, but the evidence window stays bounded."""
    import json

    curation, exhibition, plan, _profile = _plan_and_items(client)
    payload = curation.frame_payload(plan, exhibition.items, exhibition.chapters)

    assert payload["chapterCount"] == len(exhibition.chapters)
    serialised = json.dumps(payload, ensure_ascii=False)
    assert '"evidence"' in serialised
    for chapter in payload["chapters"]:
        for item in chapter["items"]:
            assert len(item["evidence"]) <= curation.MAX_FRAME_EVIDENCE_CHUNKS_PER_OBJECT
            assert item["evidence"]
            for chunk in item["evidence"]:
                assert len(chunk["text"]) <= curation.MAX_FRAME_EVIDENCE_CHARS
    assert len(serialised) < 12000


def _claim_mapped_frame(curation, exhibition, plan) -> dict[str, Any]:
    payload = curation.frame_payload(
        plan, exhibition.items, exhibition.chapters, exhibition.curatorial_brief
    )
    frame_items = [item for chapter in payload["chapters"] for item in chapter["items"]]
    all_evidence = [item["evidence"][0]["id"] for item in frame_items]
    return {
        "title": "证据之间",
        "subtitle": "从公开记录组织一条观看路径",
        "curatorialBrief": {
            "bigIdea": {
                "text": "这些馆藏记录为同一问题提供彼此限制的观察角度。",
                "evidenceIds": all_evidence[:3],
                "confidence": "provisional",
            },
            "keyMessages": [
                {
                    "text": "题名、材料与机构说明共同限定可讨论的范围。",
                    "evidenceIds": all_evidence[:2],
                    "confidence": "supported",
                },
                {
                    "text": "对照位置提醒观众不要把局部记录概括为唯一解释。",
                    "evidenceIds": all_evidence[-2:],
                    "confidence": "provisional",
                },
            ],
            "criticalQuestions": ["记录支持了什么？", "材料没有回答什么？"],
            "objects": [
                {
                    "objectId": item["objectId"],
                    "role": item["role"],
                    "selectionRationale": f"《{item['title']}》以可定位记录支撑本位置。",
                    "relation": "与相邻藏品形成可核查的比较。",
                    "evidenceIds": [item["evidence"][0]["id"]],
                }
                for item in frame_items
            ],
            "evaluationTargets": [
                {
                    "statement": "访客能够指出一条有证据和一条仍不确定的判断。",
                    "method": "comprehension_check",
                }
            ],
        },
        "chapters": [
            {"title": f"证据{index}", "leadIn": "本段只依据已列出的馆方记录展开。"}
            for index in range(len(exhibition.chapters))
        ],
        "epilogue": {"text": "这是一条有边界的解释路径。", "openQuestions": ["还缺什么材料？"]},
        "spaceDesign": {
            "wallColor": "#e0dccc",
            "floorColor": "#332f28",
            "accentColor": "#8a6a3c",
        },
    }


def test_profile_skeleton_contains_versioned_private_curatorial_brief(
    client: TestClient,
) -> None:
    import json
    from pathlib import Path

    from jsonschema import Draft202012Validator

    _curation, exhibition, _plan, _profile = _plan_and_items(client)
    brief = exhibition.curatorial_brief
    assert brief is not None
    assert brief.schema_version == "curatorial-brief/v1"
    assert brief.status == "deterministic"
    assert brief.visitor_inquiry == exhibition.question
    assert brief.audience.duration_minutes == 5
    assert brief.ethics.provenance_status == "not_reviewed"
    assert brief.ethics.cultural_sensitivity_status == "not_reviewed"
    assert brief.ethics.community_review_required is None
    assert brief.ethics.community_review_status == "not_assessed"
    # The test corpus has no dense cache, so its audit record must disclose the
    # BM25 fallback instead of claiming that hybrid retrieval ran.
    assert brief.retrieval.method == "fielded_bm25_hard_anchor"
    assert brief.retrieval.version == "bm25-v1"
    assert len(brief.excluded_candidates) <= 5
    assert {decision.object_id for decision in brief.objects} == {
        item.object.id for item in exhibition.items
    }
    private_schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "exhibition.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(private_schema).validate(
        exhibition.model_dump(mode="json", by_alias=True)
    )


def test_public_brief_redacts_profile_and_remaps_internal_item_ids(
    client: TestClient,
) -> None:
    import json
    from pathlib import Path

    from jsonschema import Draft202012Validator

    from app.models import (
        Exhibition,
        LocalizedObjectMetadata,
        PublicExhibition,
        ReviewRecord,
    )

    _curation, exhibition, _plan, _profile = _plan_and_items(client)
    exhibition.items[0].display_title = "公开中文题名"
    exhibition.items[0].localized_metadata = LocalizedObjectMetadata(
        date="十八世纪",
        medium="纸本水墨",
        culture="中国",
        institution="克利夫兰艺术博物馆",
    )
    private_item_ids = {item.id for item in exhibition.items}
    exhibition.status = "published"
    exhibition.slug = "brief-privacy-test"
    exhibition.review = ReviewRecord(
        decision="approved",
        reviewer="test-reviewer",
        evidenceReviewConfirmed=True,
    )
    public = PublicExhibition.from_exhibition(exhibition).model_dump(
        mode="json", by_alias=True
    )
    brief = public["curatorialBrief"]
    assert brief is not None
    assert "audience" not in brief
    assert "excludedCandidates" not in brief
    public_item_ids = {item["id"] for item in public["items"]}
    assert public["items"][0]["displayTitle"] == "公开中文题名"
    assert public["items"][0]["localizedMetadata"]["medium"] == "纸本水墨"
    decision_ids = {decision["itemId"] for decision in brief["objects"]}
    assert decision_ids == public_item_ids
    assert not (decision_ids & private_item_ids)
    public_schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "public-exhibition.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(public_schema).validate(public)

    # Pre-v1 JSON remains readable with no synthetic claim of having a brief.
    legacy = exhibition.model_dump(mode="json", by_alias=True)
    legacy.pop("curatorialBrief", None)
    assert Exhibition.model_validate(legacy).curatorial_brief is None


def test_claim_mapped_frame_updates_brief_and_public_prose(client: TestClient) -> None:
    from app.validator import validate_exhibition

    curation, exhibition, plan, _profile = _plan_and_items(client)
    output = _claim_mapped_frame(curation, exhibition, plan)
    applied = curation.apply_frame(exhibition, output)

    assert applied.curatorial_brief is not None
    assert applied.curatorial_brief.status == "model_refined"
    assert applied.curatorial_thesis == applied.curatorial_brief.big_idea.text
    assert applied.sub_questions == applied.curatorial_brief.critical_questions
    assert applied.core_answer.startswith(applied.curatorial_brief.key_messages[0].text)
    assert validate_exhibition(applied).passed is True


def test_new_generation_requires_brief_but_legacy_record_remains_valid(
    client: TestClient,
) -> None:
    from app.validator import validate_exhibition

    _curation, exhibition, _plan, _profile = _plan_and_items(client)
    missing = exhibition.model_copy(deep=True)
    missing.curatorial_brief = None
    result = validate_exhibition(missing)
    assert result.passed is False
    assert "CURATORIAL_BRIEF_REQUIRED" in {error.code for error in result.errors}

    missing.versions.prompt = "v2-2026-08-07"
    legacy = validate_exhibition(missing)
    assert legacy.passed is True
    assert "CURATORIAL_BRIEF_LEGACY_MISSING" in {
        warning.code for warning in legacy.warnings
    }


def test_claim_mapped_frame_rejects_invented_or_cross_object_evidence(
    client: TestClient,
) -> None:
    import copy
    import pytest

    curation, exhibition, plan, _profile = _plan_and_items(client)
    output = _claim_mapped_frame(curation, exhibition, plan)
    invented = copy.deepcopy(output)
    invented["curatorialBrief"]["bigIdea"]["evidenceIds"] = ["invented-evidence"]
    untouched = exhibition.model_copy(deep=True)
    before = untouched.model_dump(mode="json")
    with pytest.raises(ValueError, match="invents evidence"):
        curation.apply_frame(untouched, invented)
    assert untouched.model_dump(mode="json") == before

    crossed = copy.deepcopy(output)
    first, second = crossed["curatorialBrief"]["objects"][:2]
    first["evidenceIds"] = list(second["evidenceIds"])
    with pytest.raises(ValueError, match="foreign or hidden evidence"):
        curation.apply_frame(exhibition.model_copy(deep=True), crossed)


def test_profile_generation_never_marks_failed_validation_ready(
    client: TestClient, monkeypatch
) -> None:
    import asyncio

    from app.generator import ExhibitionGenerator
    from app.models import ValidationIssue, ValidationResult

    generator = ExhibitionGenerator(
        client.app.state.settings, client.app.state.collections
    )
    monkeypatch.setattr(
        "app.generator.validate_exhibition",
        lambda _exhibition: ValidationResult(
            passed=False,
            errors=[
                ValidationIssue(
                    code="TEST_BLOCKER", message="Synthetic blocking validation error."
                )
            ],
            blocking_issues=["Synthetic blocking validation error."],
        ),
    )
    result = asyncio.run(
        generator.generate_from_profile(VisitorProfile(durationMinutes=5))
    )
    assert result.validation is not None and result.validation.passed is False
    assert result.status == "draft"


def test_labels_payload_is_bounded_and_carries_both_titles(client: TestClient) -> None:
    curation, exhibition, _plan, profile = _plan_and_items(client)
    chapter = exhibition.chapters[0]
    by_id = {item.id: item for item in exhibition.items}
    items = [by_id[item_id] for item_id in chapter.item_ids]

    payload = curation.labels_payload(
        profile, chapter, items, exhibition.curatorial_brief
    )
    assert payload["curatorialBrief"]["bigIdea"]["evidenceIds"]
    assert len(payload["items"]) == len(items)
    for item in payload["items"]:
        # Both titles, so the model translates rather than invents.
        assert "title" in item and "titleOriginal" in item
        assert item["objectDecision"]["evidenceIds"]
        assert len(item["evidence"]) <= curation.MAX_EVIDENCE_CHUNKS_PER_OBJECT
        visible_ids = {chunk["id"] for chunk in item["evidence"]}
        assert set(item["objectDecision"]["evidenceIds"]) <= visible_ids
        for chunk in item["evidence"]:
            assert len(chunk["text"]) <= curation.MAX_EVIDENCE_CHARS


def test_labels_payload_separates_collection_image_from_catalogue_evidence(
    client: TestClient,
) -> None:
    curation, exhibition, _plan, profile = _plan_and_items(client)
    item = exhibition.items[0]
    chapter = next(
        chapter for chapter in exhibition.chapters if item.id in chapter.item_ids
    )
    visual = curation.image_evidence(item.object)
    assert visual is not None

    payload = curation.labels_payload(
        profile,
        chapter,
        [item],
        exhibition.curatorial_brief,
        available_visual_evidence_ids={visual.id},
    )
    raw_item = payload["items"][0]
    assert raw_item["imageEvidence"] == {
        "id": visual.id,
        "sourceKind": "collection_image",
        "supports": visual.supports,
    }
    assert visual.id not in {chunk["id"] for chunk in raw_item["evidence"]}
    assert all(
        chunk.source_kind != "collection_image"
        for chunk in curation.catalogue_evidence(item.object)
    )


def test_visual_label_binds_pixels_and_catalogue_claims_to_separate_sources(
    client: TestClient,
) -> None:
    import pytest

    curation, exhibition, _plan, profile = _plan_and_items(client)
    item = exhibition.items[0]
    visual = curation.image_evidence(item.object)
    assert visual is not None
    text_evidence = curation.catalogue_evidence(item.object)[0]
    valid_output = {
        "items": [
            {
                "objectId": item.object.id,
                "displayTitle": "山水册页",
                "labelSentences": [
                    {
                        "text": "画面上方留白较多，墨色集中在下半部。",
                        "type": "visual_observation",
                        "evidenceIds": [visual.id],
                    },
                    {
                        "text": "馆方将它著录为纸本水墨画。",
                        "type": "system_inference",
                        "evidenceIds": [text_evidence.id],
                    },
                ],
            }
        ]
    }

    applied = curation.apply_labels(
        [item],
        valid_output,
        profile.label_max_chars,
        allowed_evidence_by_object={item.object.id: {text_evidence.id}},
        visual_evidence_by_object={item.object.id: visual.id},
    )
    assert applied == 1
    assert [sentence.type for sentence in item.label_sentences] == [
        "visual_observation",
        "system_inference",
    ]

    wrong_source = {
        "items": [
            {
                **valid_output["items"][0],
                "labelSentences": [
                    {
                        "text": "这句把图像观察错误地绑定到文字记录。",
                        "type": "visual_observation",
                        "evidenceIds": [text_evidence.id],
                    },
                    valid_output["items"][0]["labelSentences"][1],
                ],
            }
        ]
    }
    before = item.model_dump(mode="json")
    with pytest.raises(ValueError, match="collection image"):
        curation.apply_labels(
            [item],
            wrong_source,
            profile.label_max_chars,
            allowed_evidence_by_object={item.object.id: {text_evidence.id}},
            visual_evidence_by_object={item.object.id: visual.id},
        )
    assert item.model_dump(mode="json") == before


def test_visual_label_keeps_both_source_layers_when_model_exceeds_budget(
    client: TestClient,
) -> None:
    curation, exhibition, _plan, profile = _plan_and_items(client)
    item = exhibition.items[0]
    visual = curation.image_evidence(item.object)
    assert visual is not None
    text_evidence = curation.catalogue_evidence(item.object)[0]
    output = {
        "items": [
            {
                "objectId": item.object.id,
                "displayTitle": "山水册页",
                "labelSentences": [
                    {
                        "text": "画面上方留出大片空白，墨色集中在下半部，近处树干向左倾斜，远山只用淡墨勾出轮廓，右侧还有一条狭窄水道穿过坡脚。",
                        "type": "visual_observation",
                        "evidenceIds": [visual.id],
                    },
                    {
                        "text": "馆方将它著录为十九世纪的纸本水墨画，并记录其来自中国，现由克利夫兰艺术博物馆收藏。",
                        "type": "system_inference",
                        "evidenceIds": [text_evidence.id],
                    },
                ],
            }
        ]
    }

    assert sum(
        len(sentence["text"])
        for sentence in output["items"][0]["labelSentences"]
    ) > profile.label_max_chars
    assert curation.apply_labels(
        [item],
        output,
        profile.label_max_chars,
        allowed_evidence_by_object={item.object.id: {text_evidence.id}},
        visual_evidence_by_object={item.object.id: visual.id},
    ) == 1
    assert sum(len(sentence.text) for sentence in item.label_sentences) <= (
        profile.label_max_chars
    )
    assert [sentence.type for sentence in item.label_sentences] == [
        "visual_observation",
        "system_inference",
    ]


def test_localized_metadata_is_field_validated_and_never_mutates_source_record(
    client: TestClient,
) -> None:
    curation, exhibition, _plan, _profile = _plan_and_items(client)
    item = exhibition.items[0]
    item.object.maker = ""
    item.object.creator = None
    item.object.date = "1710-1720 CE"
    item.object.medium = "watercolor on ivory"
    item.object.material = "watercolor on ivory"
    item.object.culture = "Italy"
    item.object.institution = "Cleveland Museum of Art"
    source_before = item.object.model_dump(mode="json")

    output = {
        "items": [
            {
                "objectId": item.object.id,
                "localizedMetadata": {
                    "creator": {"sourceValue": "", "zh": "某位十八世纪画家"},
                    "date": {
                        "sourceValue": "1710-1720 CE",
                        "zh": "1710年至1720年，公元",
                    },
                    "medium": {
                        "sourceValue": "watercolor on ivory",
                        "zh": "象牙水彩",
                    },
                    "culture": {"sourceValue": "Italy", "zh": "意大利"},
                    "institution": {
                        "sourceValue": "Cleveland Museum of Art",
                        "zh": "克利夫兰艺术博物馆",
                    },
                },
            }
        ]
    }

    assert curation.apply_localized_metadata([item], output) == 1
    assert item.localized_metadata.creator == ""
    assert item.localized_metadata.date == "1710年至1720年，公元"
    assert item.localized_metadata.medium == "象牙水彩"
    assert item.localized_metadata.culture == "意大利"
    assert item.localized_metadata.institution == "克利夫兰艺术博物馆"
    assert item.object.model_dump(mode="json") == source_before

    invalid = {
        "items": [
            {
                "objectId": item.object.id,
                "localizedMetadata": {
                    "date": {
                        "sourceValue": "1710-1720 CE",
                        "zh": "1710年至1750年",
                    },
                    "medium": {
                        "sourceValue": "watercolor on ivory",
                        "zh": "watercolor 水彩",
                    },
                    "culture": {"sourceValue": "Italy", "zh": "Italy"},
                },
            }
        ]
    }
    before_localized = item.localized_metadata.model_dump(mode="json")
    assert curation.apply_localized_metadata([item], invalid) == 0
    assert item.localized_metadata.model_dump(mode="json") == before_localized

    wrong_facts = {
        "items": [
            {
                "objectId": item.object.id,
                "localizedMetadata": {
                    "medium": {
                        "sourceValue": "watercolor on ivory",
                        "zh": "陶瓷",
                    },
                    "culture": {"sourceValue": "Italy", "zh": "法国"},
                    "institution": {
                        "sourceValue": "Cleveland Museum of Art",
                        "zh": "大英博物馆",
                    },
                },
            }
        ]
    }
    assert curation.apply_localized_metadata([item], wrong_facts) == 0
    assert item.localized_metadata.model_dump(mode="json") == before_localized

    cross_object_source = {
        "items": [
            {
                "objectId": item.object.id,
                "localizedMetadata": {
                    "culture": {"sourceValue": "France", "zh": "法国"},
                },
            }
        ]
    }
    assert curation.apply_localized_metadata([item], cross_object_source) == 0


@pytest.mark.parametrize(
    "field,source,translated",
    [
        ("creator", "Unknown artist", "毕加索"),
        ("creator", "Vincent van Gogh", "伦勃朗"),
        ("date", "late Qing dynasty", "公元前汉代"),
        ("date", "18th century CE", "公元前18世纪"),
        ("date", "18th century", "公元前18世纪"),
        ("date", "1760", "公元前1760年"),
        ("institution", "Unknown Museum", "大英博物馆"),
        ("institution", "Cleveland Museum of Art", "芝加哥艺术博物馆"),
        ("institution", "The Metropolitan Museum of Art", "芝加哥艺术博物馆"),
        ("institution", "Art Institute of Chicago", "克利夫兰艺术博物馆"),
        ("medium", "oil on canvas", "青铜"),
    ],
)
def test_localized_metadata_rejects_unverifiable_or_inverted_translations(
    field: str,
    source: str,
    translated: str,
) -> None:
    assert (
        curation._validated_chinese_metadata(
            source,
            {"sourceValue": source, "zh": translated},
            field,
        )
        == ""
    )


def test_labels_reject_hidden_evidence_without_partial_chapter_mutation(
    client: TestClient,
) -> None:
    import pytest

    curation, exhibition, _plan, profile = _plan_and_items(client)
    items = exhibition.items[:2]
    before = [item.model_dump(mode="json") for item in items]
    visible = {
        item.object.id: {item.object.evidence[0].id}
        for item in items
    }
    output = {
        "items": [
            {
                "objectId": items[0].object.id,
                "displayTitle": "已验证的中文题名",
                "labelSentences": [
                    {
                        "text": "第一件展品的句子本身合法。",
                        "type": "system_inference",
                        "evidenceIds": [items[0].object.evidence[0].id],
                    }
                ],
            },
            {
                "objectId": items[1].object.id,
                "displayTitle": "不应提交的中文题名",
                "labelSentences": [
                    {
                        "text": "这句引用了本次调用没有发送文本的证据。",
                        "type": "system_inference",
                        "evidenceIds": [items[1].object.evidence[1].id],
                    }
                ],
            },
        ]
    }

    with pytest.raises(ValueError, match="hidden payload evidence"):
        curation.apply_labels(
            items,
            output,
            profile.label_max_chars,
            allowed_evidence_by_object=visible,
        )
    assert [item.model_dump(mode="json") for item in items] == before


def test_reorder_repartitions_chapters_and_rebuilds_deterministic_decisions(
    client: TestClient,
) -> None:
    from app.generator import ExhibitionGenerator

    curation, exhibition, plan, _profile = _plan_and_items(client)
    generator = ExhibitionGenerator(
        client.app.state.settings, client.app.state.collections
    )
    exhibition = curation.apply_frame(
        exhibition, _claim_mapped_frame(curation, exhibition, plan)
    )
    assert exhibition.curatorial_brief is not None
    assert exhibition.curatorial_brief.status == "model_refined"

    reordered_ids = [item.id for item in reversed(exhibition.items)]
    generator.reorder_items(exhibition, item_ids=reordered_ids)

    chapter_ids = [
        item_id
        for chapter in sorted(exhibition.chapters, key=lambda chapter: chapter.order)
        for item_id in chapter.item_ids
    ]
    assert chapter_ids == reordered_ids
    assert exhibition.curatorial_brief is not None
    assert exhibition.curatorial_brief.status == "deterministic"
    decisions = {
        decision.item_id: decision
        for decision in exhibition.curatorial_brief.objects
    }
    for index, item in enumerate(exhibition.items):
        assert item.why_selected == generator._why_selected(
            item.role_label, item.sub_question
        )
        assert item.relation == generator._relation(index, item.role_label)
        assert decisions[item.id].selection_rationale == item.why_selected
        assert decisions[item.id].relation == item.relation


def test_refocus_rejects_when_selected_objects_do_not_support_new_focus(
    client: TestClient, monkeypatch
) -> None:
    import asyncio
    from dataclasses import replace
    from types import SimpleNamespace

    import pytest

    from app.collections import CollectionDataError
    from app.generator import AgenticRetrievalOutcome, ExhibitionGenerator

    _curation, exhibition, _plan, _profile = _plan_and_items(client)
    generator = ExhibitionGenerator(
        client.app.state.settings, client.app.state.collections
    )
    context = generator._context(exhibition.agenda)
    async def searched(*_args, **_kwargs):
        return context.results

    async def audited(*_args, **_kwargs):
        return AgenticRetrievalOutcome(
            results=context.results,
            audit_applied=True,
            answerability="supported",
        )

    monkeypatch.setattr(generator, "_search_async", searched)
    monkeypatch.setattr(generator, "_agentic_retrieve", audited)
    monkeypatch.setattr(generator, "_hard_generation_boundary", lambda *_args: None)
    monkeypatch.setattr(
        generator,
        "_context",
        lambda _agenda, **_kwargs: replace(context, results=[]),
    )
    before = exhibition.model_dump(mode="json")

    with pytest.raises(CollectionDataError) as raised:
        asyncio.run(generator.refocus(exhibition, "一个完全不同的新问题"))
    assert raised.value.code == "FOCUS_REQUIRES_NEW_EXHIBITION"
    assert exhibition.model_dump(mode="json") == before


def test_refocus_cannot_replace_current_route_with_question_card_starters(
    client: TestClient,
    monkeypatch,
) -> None:
    import asyncio

    import pytest

    from app.collections import CollectionDataError
    from app.generator import AgenticRetrievalOutcome, ExhibitionGenerator

    _curation, exhibition, _plan, _profile = _plan_and_items(client)
    generator = ExhibitionGenerator(
        client.app.state.settings,
        client.app.state.collections,
    )
    context = generator._context(exhibition.agenda)
    forced = tuple(reversed([item.object.id for item in exhibition.items]))
    assert forced != tuple(item.object.id for item in exhibition.items)

    async def searched(*_args, **_kwargs):
        return context.results

    async def audited(*_args, **_kwargs):
        return AgenticRetrievalOutcome(
            results=context.results,
            audit_applied=False,
            forced_object_ids=forced,
            answerability="supported",
        )

    monkeypatch.setattr(generator, "_search_async", searched)
    monkeypatch.setattr(generator, "_agentic_retrieve", audited)
    monkeypatch.setattr(generator, "_hard_generation_boundary", lambda *_args: None)
    monkeypatch.setattr(generator, "_context", lambda *_args, **_kwargs: context)
    before = exhibition.model_dump(mode="json")

    with pytest.raises(CollectionDataError) as raised:
        asyncio.run(generator.refocus(exhibition, "一个已审定问题卡的新焦点"))

    assert raised.value.code == "FOCUS_REQUIRES_NEW_EXHIBITION"
    assert exhibition.model_dump(mode="json") == before


def test_model_output_supplies_chinese_titles_and_multi_sentence_labels(
    client: TestClient,
) -> None:
    curation, exhibition, _plan, profile = _plan_and_items(client)

    output = {
        "title": "釉色之间",
        "subtitle": "从青白到单色瓷",
        "curatorialThesis": "同一块瓷土在不同釉料下的光谱。",
        "coreAnswer": "釉色史是对白与青之间关系的反复调整。",
        "subQuestions": ["釉色如何变化？", "工艺如何回应？"],
        "chapters": [
            {"title": f"章节{index}", "leadIn": f"引导语{index}"}
            for index in range(len(exhibition.chapters))
        ],
        "items": [
            {
                "objectId": item.object.id,
                "displayTitle": f"青瓷小瓶{index}",
                "labelSentences": [
                    {
                        "text": f"描述句{index}",
                        "type": "system_inference",
                        "evidenceIds": [item.object.evidence[0].id],
                    },
                    {
                        "text": f"语境句{index}",
                        "type": "system_inference",
                        "evidenceIds": [item.object.evidence[0].id],
                    },
                    {
                        "text": f"关联句{index}",
                        "type": "uncertain",
                        "evidenceIds": [item.object.evidence[0].id],
                    },
                ],
            }
            for index, item in enumerate(exhibition.items)
        ],
        "epilogue": {"text": "结语。", "openQuestions": ["还有什么没讲？"]},
        "spaceDesign": {"wallColor": "#e0dccc", "floorColor": "#332f28", "accentColor": "#8a6a3c"},
    }

    applied = curation.apply_model_output(exhibition, output, profile.label_max_chars)

    assert applied.title == "釉色之间"
    assert [chapter.title for chapter in applied.chapters] == [
        f"章节{index}" for index in range(len(applied.chapters))
    ]
    for item in applied.items:
        assert item.display_title.startswith("青瓷小瓶")
        # Public labels contain only the model's bounded visitor-facing prose;
        # the full institution record remains in the source drawer.
        kinds = [sentence.type for sentence in item.label_sentences]
        assert "institution_fact" not in kinds
        assert 2 <= len(item.label_sentences) <= 3
    assert applied.epilogue.text == "结语。"
    assert applied.space_design.wall_color == "#e0dccc"


def test_model_may_not_pass_off_an_english_title_as_a_translation(
    client: TestClient,
) -> None:
    curation, exhibition, _plan, profile = _plan_and_items(client)
    before = [item.display_title for item in exhibition.items]

    output = {
        "title": "T",
        "curatorialThesis": "C",
        "items": [
            {"objectId": item.object.id, "displayTitle": "Blue-and-white Bottle"}
            for item in exhibition.items
        ],
    }
    applied = curation.apply_model_output(exhibition, output, profile.label_max_chars)
    assert [item.display_title for item in applied.items] == before


def test_model_cannot_cite_another_objects_evidence(client: TestClient) -> None:
    curation, exhibition, _plan, profile = _plan_and_items(client)
    foreign = exhibition.items[1].object.evidence[0].id

    output = {
        "title": "T",
        "curatorialThesis": "C",
        "items": [
            {
                "objectId": exhibition.items[0].object.id,
                "labelSentences": [
                    {"text": "借用了别件展品的来源。", "type": "system_inference",
                     "evidenceIds": [foreign]}
                ],
            }
        ],
    }
    import pytest

    with pytest.raises(ValueError, match="another object"):
        curation.apply_model_output(exhibition, output, profile.label_max_chars)


def test_model_prose_is_normalised_to_simplified_chinese(client: TestClient) -> None:
    """A model trained on classical art writing drifts into Traditional.

    The exhibition interface is Simplified, and mixing the two reads as broken,
    so conversion is deterministic rather than left to prompt compliance.
    """
    curation, exhibition, _plan, profile = _plan_and_items(client)

    output = {
        "title": "山水筆墨：仿古與真境",
        "subtitle": "關於繪畫傳承的閱讀",
        "curatorialThesis": "同一塊絹本上的變奏。",
        "coreAnswer": "繪畫史是對筆墨關係的反覆調整。",
        "chapters": [
            {"title": "雲山圖卷", "leadIn": "繪雲氣繚繞的山居之景。"}
            for _ in exhibition.chapters
        ],
        "epilogue": {"text": "屬明代繪畫。", "openQuestions": ["還有什麼沒講？"]},
        "items": [
            {
                "objectId": item.object.id,
                "displayTitle": "雲山圖",
                "labelSentences": [
                    {
                        "text": "此作為絹本墨筆掛軸，畫心縱一七六厘米。",
                        "type": "system_inference",
                        "evidenceIds": [item.object.evidence[0].id],
                    }
                ],
            }
            for item in exhibition.items
        ],
    }

    applied = curation.apply_model_output(exhibition, output, profile.label_max_chars)

    traditional = "為繪畫關係雲圖屬還麼講"
    surfaces = [
        applied.title,
        applied.subtitle,
        applied.curatorial_thesis,
        applied.core_answer,
        applied.epilogue.text,
        *applied.epilogue.open_questions,
        *[chapter.title for chapter in applied.chapters],
        *[chapter.lead_in for chapter in applied.chapters],
        *[item.display_title for item in applied.items],
        *[
            sentence.text
            for item in applied.items
            for sentence in item.label_sentences
            if sentence.type != "institution_fact"
        ],
    ]
    for text in surfaces:
        assert not (set(text) & set(traditional)), f"traditional characters survived in: {text}"

    assert applied.title.startswith("山水笔墨")
    assert all(item.display_title == "云山图" for item in applied.items)


def test_institution_source_text_is_never_rewritten(client: TestClient) -> None:
    """Conversion applies to system prose only; institution records stay verbatim."""
    curation, exhibition, _plan, profile = _plan_and_items(client)
    before = [
        sentence.text
        for item in exhibition.items
        for sentence in item.label_sentences
        if sentence.type == "institution_fact"
    ]

    curation.apply_model_output(
        exhibition,
        {
            "title": "標題",
            "curatorialThesis": "命題。",
            "items": [
                {
                    "objectId": item.object.id,
                    "labelSentences": [
                        {
                            "text": "系統推斷句。",
                            "type": "system_inference",
                            "evidenceIds": [item.object.evidence[0].id],
                        }
                    ],
                }
                for item in exhibition.items
            ],
        },
        profile.label_max_chars,
    )

    after = [
        sentence.text
        for item in exhibition.items
        for sentence in item.label_sentences
        if sentence.type == "institution_fact"
    ]
    assert after == before
