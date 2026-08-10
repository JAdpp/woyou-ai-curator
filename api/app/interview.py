"""Curator-agent interview: a deterministic state machine, not a free chat.

The field set, ordering and option lists are fixed server-side and the options
for the opening question are derived from what the corpus can actually route,
so the interview cannot promise a theme the collection cannot answer. A model
is only used to parse free-form text into structured fields — never to decide
what to ask next.

Question design follows the visitor typologies in 01b §3.1:
  * motivation  — Falk (2009) identity-related visit motivations
  * duration    — Véron & Levasseur (1983) circulation styles, via DURATION_PLAN
  * label budget— Serrell (1997) on actual visitor attention

Hard ceiling of seven turns, four of them required. The seventh exists because
a broad opening topic and a sharp specific question are different things: the
visitor is asked for both, and the specific one anchors retrieval when given.
"""

from __future__ import annotations

import re
from uuid import uuid4

from .collections import CollectionRepository, LoadedCollection
from .models import (
    AnswerabilityStatus,
    InterviewAnswer,
    InterviewOption,
    InterviewQuestion,
    InterviewQuestionId,
    InterviewState,
    InterviewTurn,
    MOTIVATION_LABELS,
    VisitorMotivation,
    VisitorProfile,
    utc_now,
)

# Domain ids are corpus routing labels; these are the visitor-facing words for
# them. Kept here rather than in the importer so wording can change without a
# data re-import.
DOMAIN_CHOICES: dict[str, tuple[str, str]] = {
    "global:nature-place": ("自然与地方", "风景、动植物与人如何理解所处之地"),
    "global:belief-ritual": ("信仰与仪式", "不同文化怎样让信念进入器物与空间"),
    "global:death-afterlife": ("死亡与来世", "人们如何纪念逝者并想象身后世界"),
    "global:power-status": ("权力与身份秩序", "王权、等级与身份如何被看见"),
    "global:body-identity": ("身体与身份", "肖像、服饰与身体表达了谁"),
    "global:making-material": ("材料与制作", "材料、工艺与制作者的知识"),
    "global:text-memory": ("书写与记忆", "文字、铭文与档案怎样保存记忆"),
    "global:exchange-mobility": ("交流与流动", "贸易、迁徙与旅行怎样改变物件"),
    "global:daily-life": ("日常生活", "饮食、居家、劳动与娱乐留下什么"),
    "global:image-story": ("图像、观看与故事", "图像媒介如何组织观看并讲述故事"),
    "landscape-brush": ("山水与笔墨", "山水、花鸟与文人画里的观看方式"),
    "calligraphy-inscription": ("书写与题跋", "字如何进入画面，又如何改变它"),
    "ritual-bronze": ("礼器与青铜", "祭祀器物的造型、纹饰与秩序"),
    "ceramics-glaze": ("陶瓷与釉色", "泥土、火与釉色的千年试验"),
    "buddhist-devotion": ("佛教造像", "造像的姿态、手印与礼拜方式"),
    "funerary-afterlife": ("丧葬与来世", "墓里的器物透露的身后想象"),
    "court-daily-life": ("宫廷与日常", "服饰、家具与文房里的身份"),
    "trade-exchange": ("贸易与交流", "沿丝路与海路流动的器物"),
}


def domain_choices_for(collection: LoadedCollection) -> dict[str, tuple[str, str]]:
    """Prefer the frozen collection manifest over application hard-coding."""
    return collection.evidence_domain_choices or DOMAIN_CHOICES

# Zhang Yanyuan (c. 815–877) wrote the first comprehensive history of Chinese
# painting; naming the agent after him is a small nod to the first person who
# arranged artworks into an argument rather than a list.
CURATOR_NAME = "彦远"

# A domain needs this many routable objects before it is worth offering — and
# before the landing page should advertise it.
MIN_DOMAIN_OBJECTS = 12

UNSURE_VALUE = "__unsure__"
FREE_TEXT_VALUE = "__free_text__"
SKIP_VALUE = "__skip__"

EXCLUSION_CHOICES = (
    ("religion", "宗教内容"),
    ("funerary", "墓葬与死亡"),
    ("war", "战争与暴力"),
    ("none", "没有，都可以"),
)

NO_OPEN_QUESTION_VALUE = "__no_question__"

# Shown when the model is unavailable. Deliberately about how one looks at
# objects rather than about any particular subject, so they stay true whatever
# the collection turns out to hold.
GENERIC_OPEN_QUESTIONS = (
    "这些东西当初是做给谁看的？",
    "同一个主题，不同地方的人做法差在哪里？",
    "哪一件最不像它那个年代该有的样子？",
)

TOTAL_STEPS = 6


class InterviewService:
    """Owns question sequencing and answer folding.

    Sessions live in the injected store dict; the caller decides persistence.
    """

    def __init__(self, collections: CollectionRepository) -> None:
        self.collections = collections

    # -- helpers ---------------------------------------------------------

    def available_domains(self, collection: LoadedCollection) -> list[tuple[str, str, str]]:
        """Public alias: the curator's voice must speak only of real coverage."""
        return self._available_domains(collection)

    def _available_domains(self, collection: LoadedCollection) -> list[tuple[str, str, str]]:
        """Domains with enough routed objects to sustain a visit, richest first."""
        counts: dict[str, int] = {}
        for obj in collection.objects:
            if not obj.evidence:
                continue
            for domain_id in obj.routing_domain_ids:
                counts[domain_id] = counts.get(domain_id, 0) + 1
        available: list[tuple[str, str, str]] = []
        choices = domain_choices_for(collection)
        for domain_id, count in sorted(counts.items(), key=lambda pair: -pair[1]):
            if count < MIN_DOMAIN_OBJECTS or domain_id not in choices:
                continue
            label, hint = choices[domain_id]
            available.append((domain_id, label, f"{hint} · {count} 件"))
        return available

    def _curiosity_question(self, collection: LoadedCollection) -> InterviewQuestion:
        options = [
            InterviewOption(value=domain_id, label=label, hint=hint)
            for domain_id, label, hint in self._available_domains(collection)
        ]
        options.append(
            InterviewOption(
                value=UNSURE_VALUE,
                label="我还不确定，你推荐",
                hint=f"由{CURATOR_NAME}替你挑一条线索",
            )
        )
        return InterviewQuestion(
            id=InterviewQuestionId.CURIOSITY,
            prompt=(
                f"我是{CURATOR_NAME}，这次的 AI 策展人。先问你几个问题，"
                "然后为你单独搭一座展厅。\n这一次，你最想看点什么？"
            ),
            options=options,
            allow_free_text=True,
            free_text_placeholder="或者直接告诉我你想弄懂什么…",
            step=1,
            total_steps=TOTAL_STEPS,
        )

    @staticmethod
    def _motivation_question() -> InterviewQuestion:
        return InterviewQuestion(
            id=InterviewQuestionId.MOTIVATION,
            prompt="你这次来，主要是想——",
            options=[
                InterviewOption(
                    value=VisitorMotivation.EXPLORER.value,
                    label=MOTIVATION_LABELS[VisitorMotivation.EXPLORER.value],
                    hint="会给你更完整的论证和对照材料",
                ),
                InterviewOption(
                    value=VisitorMotivation.RECHARGER.value,
                    label=MOTIVATION_LABELS[VisitorMotivation.RECHARGER.value],
                    hint="展签更短，视觉和节奏优先",
                ),
                InterviewOption(
                    value=VisitorMotivation.FACILITATOR.value,
                    label=MOTIVATION_LABELS[VisitorMotivation.FACILITATOR.value],
                    hint="语言更口语，多一些可以聊的问题",
                ),
                InterviewOption(
                    value=VisitorMotivation.PROFESSIONAL.value,
                    label=MOTIVATION_LABELS[VisitorMotivation.PROFESSIONAL.value],
                    hint="保留术语、年代与材质细节",
                ),
            ],
            step=2,
            total_steps=TOTAL_STEPS,
        )

    @staticmethod
    def _prior_knowledge_question(topic: str) -> InterviewQuestion:
        subject = f"“{topic}”" if topic else "这个主题"
        return InterviewQuestion(
            id=InterviewQuestionId.PRIOR_KNOWLEDGE,
            prompt=f"对{subject}，你现在了解多少？",
            options=[
                InterviewOption(value="none", label="第一次接触", hint="从最基本的看法讲起"),
                InterviewOption(value="some", label="略知一二", hint="跳过常识，直接进主线"),
                InterviewOption(value="familiar", label="比较熟悉", hint="多给细节、异例与争议"),
            ],
            step=3,
            total_steps=TOTAL_STEPS,
        )

    @staticmethod
    def _duration_question() -> InterviewQuestion:
        return InterviewQuestion(
            id=InterviewQuestionId.DURATION,
            prompt="你打算待多久？我按这个来安排展线的长短和节奏。",
            options=[
                InterviewOption(value="5", label="5 分钟", hint="5 件展品 · 2 个叙事区段"),
                InterviewOption(value="10", label="10 分钟", hint="8 件展品 · 3 个叙事区段"),
                InterviewOption(value="15", label="15 分钟", hint="12 件展品 · 4 个叙事区段"),
            ],
            step=4,
            total_steps=TOTAL_STEPS,
        )

    @staticmethod
    def open_question_question(
        topic: str, suggestions: tuple[str, ...] = ()
    ) -> InterviewQuestion:
        """Ask for the one thing the visitor most wants answered.

        The suggestions are offered as clickable starting points rather than a
        closed menu: the field is free text, and none of the options commits
        the collection to anything, because each is a question, not a promise.
        """
        subject = f"“{topic}”" if topic else "这个主题"
        offered = tuple(suggestions) or GENERIC_OPEN_QUESTIONS
        options = [
            InterviewOption(value=text, label=text) for text in offered[:3]
        ]
        options.append(
            InterviewOption(
                value=NO_OPEN_QUESTION_VALUE,
                label="暂时没有，你来带路",
                hint=f"由{CURATOR_NAME}决定这条线怎么走",
            )
        )
        return InterviewQuestion(
            id=InterviewQuestionId.OPEN_QUESTION,
            prompt=f"关于{subject}，你有什么特别好奇、特别想弄懂的问题吗？",
            options=options,
            allow_free_text=True,
            free_text_placeholder="也可以直接写下你自己的问题…",
            skippable=True,
            step=5,
            total_steps=TOTAL_STEPS,
        )

    @staticmethod
    def _exclusions_question() -> InterviewQuestion:
        return InterviewQuestion(
            id=InterviewQuestionId.EXCLUSIONS,
            prompt="最后一个：有什么是你不太想看到的？",
            options=[
                InterviewOption(value=value, label=label)
                for value, label in EXCLUSION_CHOICES
            ],
            multi_select=True,
            skippable=True,
            step=6,
            total_steps=TOTAL_STEPS,
        )

    def _negotiation_question(
        self, state: InterviewState, collection: LoadedCollection
    ) -> InterviewQuestion | None:
        """Turn a coverage gap into a choice instead of a refusal.

        This is decision C: the answerability gate still runs, but a visitor
        never hits a dead end. Returns ``None`` when the corpus can answer the
        question as asked.
        """
        question_text = (state.profile.free_form_question or "").strip()
        if not question_text:
            return None

        from .generator import ExhibitionGenerator  # local import avoids a cycle

        agenda = state.profile.to_agenda(collection.id)
        try:
            check = ExhibitionGenerator.probe_answerability(self.collections, agenda)
        except Exception:  # noqa: BLE001 - probing must never break the interview
            return None
        if check.status == AnswerabilityStatus.SUPPORTED.value:
            return None

        available = self._available_domains(collection)[:3]
        if not available:
            return None
        covered = "、".join(label for _id, label, _hint in available[:2])
        state.negotiation_note = (
            f"关于“{question_text}”，我们手上的馆藏能讲清的部分集中在{covered}；"
            "完全按你原来的问法来，材料会不够扎实。"
        )
        options = [
            InterviewOption(value=domain_id, label=f"从「{label}」进去", hint=hint)
            for domain_id, label, hint in available
        ]
        options.append(
            InterviewOption(
                value=FREE_TEXT_VALUE,
                label="还是按我原来的问题来",
                hint="材料会更薄，展签会明确标出缺口",
            )
        )
        return InterviewQuestion(
            id=InterviewQuestionId.NEGOTIATION,
            prompt=f"{state.negotiation_note}\n你想怎么走？",
            options=options,
            step=4,
            total_steps=TOTAL_STEPS,
        )

    # -- public API ------------------------------------------------------

    def start(self, collection_id: str | None = None) -> InterviewState:
        collection = self.collections.get(collection_id)
        state = InterviewState(id=str(uuid4()), collection_id=collection.id)
        state.next_question = self._curiosity_question(collection)
        return state

    def answer(self, state: InterviewState, answer: InterviewAnswer) -> InterviewState:
        collection = self.collections.get(state.collection_id)
        current = state.next_question
        if current is None or current.id != answer.question_id:
            # Ignore replies to a question we are no longer on rather than
            # corrupting the profile with a stale value.
            return state

        turn = InterviewTurn(
            question_id=answer.question_id,
            prompt=current.prompt,
            answer_value=answer.value,
            free_text=(answer.free_text or "").strip() or None,
            skipped=answer.skipped,
        )
        self._apply(state, answer, turn, collection)
        state.transcript.append(turn)
        state.next_question = self._next(state, collection)
        state.complete = state.next_question is None
        state.updated_at = utc_now()
        return state

    def _apply(
        self,
        state: InterviewState,
        answer: InterviewAnswer,
        turn: InterviewTurn,
        collection: LoadedCollection,
    ) -> None:
        profile = state.profile
        question_id = answer.question_id
        free_text = (answer.free_text or "").strip()

        if question_id == InterviewQuestionId.CURIOSITY:
            if free_text:
                profile.free_form_question = free_text[:500]
                matched = self._match_domain(free_text, collection)
                if matched:
                    profile.curiosity_domain_id = matched
                    profile.curiosity_label = domain_choices_for(collection)[matched][0]
                turn.answer_label = free_text[:60]
            elif answer.value and answer.value != UNSURE_VALUE:
                profile.curiosity_domain_id = answer.value
                profile.curiosity_label = domain_choices_for(collection).get(
                    answer.value, (answer.value, "")
                )[0]
                turn.answer_label = profile.curiosity_label
            else:
                # "Recommend something" -> take the richest routable domain.
                available = self._available_domains(collection)
                if available:
                    profile.curiosity_domain_id = available[0][0]
                    profile.curiosity_label = available[0][1]
                turn.answer_label = "由策展人推荐"

        elif question_id == InterviewQuestionId.MOTIVATION:
            if answer.value in {item.value for item in VisitorMotivation}:
                profile.motivation = VisitorMotivation(answer.value)
                turn.answer_label = MOTIVATION_LABELS[answer.value]

        elif question_id == InterviewQuestionId.PRIOR_KNOWLEDGE:
            if answer.value in {"none", "some", "familiar"}:
                profile.prior_knowledge = answer.value
                turn.answer_label = {
                    "none": "第一次接触",
                    "some": "略知一二",
                    "familiar": "比较熟悉",
                }[answer.value]

        elif question_id == InterviewQuestionId.DURATION:
            if answer.value in {"5", "10", "15"}:
                profile.duration_minutes = int(answer.value)  # type: ignore[assignment]
                turn.answer_label = f"{answer.value} 分钟"

        elif question_id == InterviewQuestionId.NEGOTIATION:
            if answer.value and answer.value != FREE_TEXT_VALUE:
                profile.curiosity_domain_id = answer.value
                profile.curiosity_label = domain_choices_for(collection).get(
                    answer.value, (answer.value, "")
                )[0]
                # The visitor accepted a narrower route, so the original
                # free-form wording is kept only as context, not as the query.
                turn.answer_label = f"从「{profile.curiosity_label}」进去"
            else:
                turn.answer_label = "保持原来的问题"

        elif question_id == InterviewQuestionId.OPEN_QUESTION:
            # The suggested options carry their own text as the value, so a
            # clicked suggestion and a typed question land in the same field.
            chosen = free_text or (
                answer.value
                if answer.value and answer.value != NO_OPEN_QUESTION_VALUE
                else ""
            )
            if answer.skipped or not chosen:
                turn.skipped = answer.skipped
                turn.answer_label = "跳过" if answer.skipped else "交给策展人决定"
            else:
                profile.open_question = chosen[:300]
                turn.answer_label = chosen[:60]

        elif question_id == InterviewQuestionId.EXCLUSIONS:
            if answer.skipped or not answer.value or answer.value == SKIP_VALUE:
                turn.skipped = True
                turn.answer_label = "跳过"
            else:
                labels = dict(EXCLUSION_CHOICES)
                chosen = [
                    value.strip()
                    for value in answer.value.split(",")
                    if value.strip() and value.strip() != "none"
                ]
                profile.excluded_topics = [
                    {"religion": "宗教", "funerary": "墓葬", "war": "战争"}.get(value, value)
                    for value in chosen
                ]
                turn.answer_label = (
                    "、".join(labels.get(value, value) for value in chosen) or "没有"
                )
            if free_text:
                profile.excluded_topics.extend(
                    part.strip() for part in re.split(r"[，,、]", free_text) if part.strip()
                )

    def _next(
        self, state: InterviewState, collection: LoadedCollection
    ) -> InterviewQuestion | None:
        asked = {turn.question_id for turn in state.transcript}
        if InterviewQuestionId.MOTIVATION not in asked:
            return self._motivation_question()
        if InterviewQuestionId.PRIOR_KNOWLEDGE not in asked:
            return self._prior_knowledge_question(state.profile.curiosity_label)
        if InterviewQuestionId.DURATION not in asked:
            return self._duration_question()
        # Asked before the answerability gate so that the sharper question is
        # what gets probed; negotiating over the broad topic while ignoring the
        # visitor's actual question would check the wrong thing.
        if InterviewQuestionId.OPEN_QUESTION not in asked:
            return self.open_question_question(state.profile.curiosity_label)
        if InterviewQuestionId.NEGOTIATION not in asked:
            negotiation = self._negotiation_question(state, collection)
            if negotiation is not None:
                return negotiation
        if InterviewQuestionId.EXCLUSIONS not in asked:
            return self._exclusions_question()
        return None

    def _match_domain(self, text: str, collection: LoadedCollection) -> str | None:
        """Recognise only a domain the visitor actually named.

        Object-level retrieval used to infer a domain from any one overlapping
        word.  “quantum physics” and “cats across cultures” could therefore be
        rewritten as “materials and making”, even though the visitor never
        asked for that frame.  Unrecognised free text now remains free text and
        is handled by the anchored collection retriever.
        """
        compact = re.sub(r"\s+", "", text.casefold())
        matches: list[tuple[int, str]] = []
        for domain_id, label, _hint in self._available_domains(collection):
            terms = [
                term.strip()
                for term in re.split(r"[与和、/，,：:]", label.casefold())
                if len(term.strip()) >= 2
            ]
            matched_length = sum(len(term) for term in terms if term in compact)
            if matched_length:
                matches.append((matched_length, domain_id))
        return max(matches)[1] if matches else None
