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

Hard ceiling of seven turns, four of them required. A concrete visitor-written
question is collected either in the opening turn or one later follow-up, never
both. An opening with only a generic referent can use that follow-up for one
optional scope clarification; the evidence gate remains a separate final turn.
An opening with no subject at all ("I don't know what to ask", "you choose")
is never stored as a question: it gets one turn of collection introduction
and hand-picked starting objects instead (``interview_featured``).
"""

from __future__ import annotations

import re
from uuid import uuid4

from . import i18n
from .collections import CollectionRepository, LoadedCollection
from .interview_clarification import needs_scope_clarification, undecided_opening_kind
from .interview_featured import (
    FEATURED_PREFIX,
    collection_intro,
    entry_by_id,
    entry_for_question,
    featured_for,
)
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


def domain_choices_for(
    collection: LoadedCollection, language: str = "zh"
) -> dict[str, tuple[str, str]]:
    """Prefer the frozen collection manifest over application hard-coding.

    The manifest's wording is Chinese and it carries a content hash, so English
    labels are keyed by the same domain id in ``i18n`` rather than written back
    into frozen data. A domain the English table does not know keeps its
    manifest label, which is better than dropping it from the interview.
    """
    zh_choices = collection.evidence_domain_choices or DOMAIN_CHOICES
    if language != "en":
        return zh_choices
    return {
        domain_id: i18n.DOMAIN_LABELS_EN.get(
            domain_id, (label, i18n.DEFAULT_DOMAIN_HINT_EN)
        )
        for domain_id, (label, _hint) in zh_choices.items()
    }

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
RECOMMENDED_QUESTION_PREFIX = "__recommended_question__:"
SCOPE_DIRECTION_PREFIX = "__scope_direction__:"
KEEP_SCOPE_VALUE = "__keep_scope_open__"

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

    def __init__(
        self,
        collections: CollectionRepository,
        *,
        audit_available: bool = False,
    ) -> None:
        self.collections = collections
        self.audit_available = audit_available

    # -- helpers ---------------------------------------------------------

    def available_domains(
        self, collection: LoadedCollection, language: str = "zh"
    ) -> list[tuple[str, str, str]]:
        """Public alias: the curator's voice must speak only of real coverage."""
        return self._available_domains(collection, language)

    def _available_domains(
        self, collection: LoadedCollection, language: str = "zh"
    ) -> list[tuple[str, str, str]]:
        """Domains with enough routed objects to sustain a visit, richest first."""
        counts: dict[str, int] = {}
        for obj in collection.objects:
            if not obj.evidence:
                continue
            for domain_id in obj.routing_domain_ids:
                counts[domain_id] = counts.get(domain_id, 0) + 1
        available: list[tuple[str, str, str]] = []
        choices = domain_choices_for(collection, language)
        for domain_id, count in sorted(counts.items(), key=lambda pair: -pair[1]):
            if count < MIN_DOMAIN_OBJECTS or domain_id not in choices:
                continue
            label, hint = choices[domain_id]
            unit = f"{count:,} objects" if language == "en" else f"{count} 件"
            available.append((domain_id, label, f"{hint} · {unit}"))
        return available

    def _curiosity_question(
        self, collection: LoadedCollection, language: str = "zh"
    ) -> InterviewQuestion:
        curator = i18n.CURATOR_NAME_EN if language == "en" else CURATOR_NAME
        options = [
            InterviewOption(value=domain_id, label=label, hint=hint)
            for domain_id, label, hint in self._available_domains(collection, language)
        ]
        options.append(
            InterviewOption(
                value=UNSURE_VALUE,
                label=i18n.pick(language, "我还不确定，你推荐", "I'm not sure — you choose"),
                hint=i18n.pick(
                    language,
                    f"由{curator}替你挑一条线索",
                    f"{curator} picks a thread for you",
                ),
            )
        )
        return InterviewQuestion(
            id=InterviewQuestionId.CURIOSITY,
            prompt=i18n.pick(
                language,
                f"你好，我是“卧游”的 AI 策展人{CURATOR_NAME}。我会问你几个简单问题，"
                "然后从我们的馆藏里为你定制一座展厅。\n你有什么最想了解的吗？",
                f"Hello, I'm {i18n.CURATOR_NAME_EN}, Woyou's AI curator. I'll ask you a "
                "few simple questions, then build you an exhibition room from our "
                "collection.\nIs there anything you'd most like to know about?",
            ),
            options=options,
            allow_free_text=True,
            free_text_placeholder=i18n.pick(
                language,
                "想了解什么都可以说，还没想好也没关系…",
                "Anything you're curious about — or say you're not sure yet…",
            ),
            step=1,
            # Five is the shortest real route (the visitor may type the core
            # question here). Conditional branches only raise this total; the
            # progress indicator therefore never moves backwards from 1/6 to
            # 2/5 after the opening answer.
            total_steps=5,
        )

    def _featured_question(
        self, state: InterviewState, collection: LoadedCollection
    ) -> InterviewQuestion:
        """Show the collection to a visitor who has nothing to ask yet.

        Reached only when ``featured_for`` finds at least two showable
        entries. Each card carries a question the collection has answered
        before; the last option hands the choice back to the curator.
        """
        language = state.profile.language
        options = [
            InterviewOption(
                value=f"{FEATURED_PREFIX}{entry.id}",
                label=entry.title(language),
                hint=entry.caption(language),
                object_id=entry.object_id,
            )
            for entry in featured_for(collection)
        ]
        options.append(
            InterviewOption(
                value=UNSURE_VALUE,
                label=i18n.pick(language, "还是你替我定吧", "You pick one for me"),
                hint=i18n.pick(
                    language,
                    f"由{CURATOR_NAME}从这几件里挑一个开头",
                    f"{i18n.CURATOR_NAME_EN} chooses where to start",
                ),
            )
        )
        return InterviewQuestion(
            id=InterviewQuestionId.FEATURED,
            prompt=collection_intro(collection, language)
            + "\n"
            + i18n.pick(
                language,
                "我先挑了几件给你看看，哪件让你想多看两眼？也可以直接告诉我你对什么感兴趣。",
                "Here are a few I'd start with. Which one would you like a closer look at? "
                "Or just tell me what interests you.",
            ),
            options=options,
            allow_free_text=True,
            free_text_placeholder=i18n.pick(
                language,
                "比如：我想看看……",
                "For example: I'd like to see…",
            ),
            step=2,
            total_steps=TOTAL_STEPS,
        )

    @staticmethod
    def _motivation_question(language: str = "zh") -> InterviewQuestion:
        hints_zh = {
            "explorer": "会给你更完整的论证和对照材料",
            "recharger": "展签更短，视觉和节奏优先",
            "facilitator": "语言更口语，多一些可以聊的问题",
            "professional": "保留术语、年代与材质细节",
        }
        return InterviewQuestion(
            id=InterviewQuestionId.MOTIVATION,
            prompt=i18n.pick(language, "你这次来，主要是想——", "What brings you here today?"),
            options=[
                InterviewOption(
                    value=motivation.value,
                    label=i18n.pick(
                        language,
                        MOTIVATION_LABELS[motivation.value],
                        i18n.MOTIVATION_LABELS_EN[motivation.value],
                    ),
                    hint=i18n.pick(
                        language,
                        hints_zh[motivation.value],
                        i18n.MOTIVATION_HINTS_EN[motivation.value],
                    ),
                )
                for motivation in (
                    VisitorMotivation.EXPLORER,
                    VisitorMotivation.RECHARGER,
                    VisitorMotivation.FACILITATOR,
                    VisitorMotivation.PROFESSIONAL,
                )
            ],
            step=2,
            total_steps=TOTAL_STEPS,
        )

    @staticmethod
    def _custom_question_question(language: str = "zh") -> InterviewQuestion:
        """Collect the question promised by the explorer motivation.

        This deliberately has its own id.  ``open_question`` is the later,
        optional prompt whose choices may be suggested by the language model;
        reusing that id here would let those suggestions overwrite this direct
        follow-up and recreate the conversational mismatch this branch avoids.
        """

        return InterviewQuestion(
            id=InterviewQuestionId.CUSTOM_QUESTION,
            prompt=i18n.pick(language, "这个问题是？", "What is the question?"),
            options=[],
            allow_free_text=True,
            free_text_placeholder=i18n.pick(
                language,
                "直接写下你想弄明白的问题…",
                "Write the question you want to figure out…",
            ),
            step=3,
            total_steps=TOTAL_STEPS,
        )

    def _scope_clarification_question(
        self, state: InterviewState, collection: LoadedCollection,
    ) -> InterviewQuestion:
        """One optional scope follow-up, using the existing custom-question UI.

        Choices are explicitly tentative corpus-backed directions, not claims
        to have inferred the visitor's meaning. Keep NEGOTIATION available for
        the later, distinct evidence-coverage check.
        """
        language = state.profile.language
        original = (state.profile.free_form_question or "").strip()[:90]
        options = [InterviewOption(
            value=f"{SCOPE_DIRECTION_PREFIX}{domain_id}",
            label=i18n.pick(language, f"从「{label}」试逛", f"Try “{label}”"),
            hint=hint,
        ) for domain_id, label, hint in self._available_domains(collection, language)[:3]]
        options.append(InterviewOption(
            value=KEEP_SCOPE_VALUE,
            label=i18n.pick(language, "还没想好，先保留这个偏好", "Keep this as an open preference"),
            hint=i18n.pick(language, "不把它当作确定事实或客观排名", "Not a factual claim or an objective ranking"),
        ))
        return InterviewQuestion(
            id=InterviewQuestionId.CUSTOM_QUESTION,
            prompt=i18n.pick(
                language,
                f"你说的“{original}”还可以指不同方向，我先不替你猜。"
                "你更在意外观给人的感觉、当时的生活与用途，还是馆方确实记录了什么？"
                "可以补充一种物件、时代或地点，也可以选一个方向试逛。",
                f"“{original}” could mean several things, so I won't assume a subject. "
                "Do you mean how things look, how people used them, or what museum records confirm? "
                "Add an object, period or place, or try one of these directions.",
            ),
            options=options, allow_free_text=True, skippable=True,
            free_text_placeholder=i18n.pick(language, "比如：我指的是……；我更想看……", "I mean…; I'd rather see…"),
        )

    @staticmethod
    def _prior_knowledge_question(
        topic: str, language: str = "zh", *, question_led: bool = False
    ) -> InterviewQuestion:
        if language == "en":
            if question_led:
                prompt = "How much do you already know about the subject behind that question?"
            else:
                subject = f"“{topic}”" if topic else "this subject"
                prompt = f"How much do you already know about {subject}?"
            options = [
                InterviewOption(value=value, label=label, hint=hint)
                for value, (label, hint) in i18n.PRIOR_KNOWLEDGE_EN.items()
            ]
        else:
            if question_led:
                prompt = "对这个问题涉及的主题，你现在了解多少？"
            else:
                subject = f"“{topic}”" if topic else "这个主题"
                prompt = f"对{subject}，你现在了解多少？"
            options = [
                InterviewOption(value="none", label="第一次接触", hint="从最基本的看法讲起"),
                InterviewOption(value="some", label="略知一二", hint="跳过常识，直接进主线"),
                InterviewOption(value="familiar", label="比较熟悉", hint="多给细节、异例与争议"),
            ]
        return InterviewQuestion(
            id=InterviewQuestionId.PRIOR_KNOWLEDGE,
            prompt=prompt,
            options=options,
            step=3,
            total_steps=TOTAL_STEPS,
        )

    @staticmethod
    def _duration_question(language: str = "zh") -> InterviewQuestion:
        hints_zh = {
            "5": "5 件展品 · 2 个叙事区段",
            "10": "8 件展品 · 3 个叙事区段",
            "15": "12 件展品 · 4 个叙事区段",
        }
        return InterviewQuestion(
            id=InterviewQuestionId.DURATION,
            prompt=i18n.pick(
                language,
                "你打算待多久？我按这个来安排展线的长短和节奏。",
                "How long do you have? I'll set the length and pace of the route to match.",
            ),
            options=[
                InterviewOption(
                    value=minutes,
                    label=i18n.pick(language, f"{minutes} 分钟", f"{minutes} minutes"),
                    hint=i18n.pick(language, hints_zh[minutes], i18n.DURATION_HINTS_EN[minutes]),
                )
                for minutes in ("5", "10", "15")
            ],
            step=4,
            total_steps=TOTAL_STEPS,
        )

    @staticmethod
    def open_question_question(
        topic: str, suggestions: tuple[str, ...] = (), language: str = "zh"
    ) -> InterviewQuestion:
        """Ask for the one thing the visitor most wants answered.

        The suggestions are offered as clickable starting points rather than a
        closed menu: the field is free text, and none of the options commits
        the collection to anything, because each is a question, not a promise.
        """
        if language == "en":
            subject = f"“{topic}”" if topic else "this subject"
            prompt = f"Is there anything about {subject} you're especially curious about?"
            offered = tuple(suggestions) or i18n.GENERIC_OPEN_QUESTIONS_EN
        else:
            subject = f"“{topic}”" if topic else "这个主题"
            prompt = f"关于{subject}，你有什么特别好奇、特别想弄懂的问题吗？"
            offered = tuple(suggestions) or GENERIC_OPEN_QUESTIONS
        curator = i18n.CURATOR_NAME_EN if language == "en" else CURATOR_NAME
        options = [
            InterviewOption(value=text, label=text) for text in offered[:3]
        ]
        options.append(
            InterviewOption(
                value=NO_OPEN_QUESTION_VALUE,
                label=i18n.pick(language, "暂时没有，你来带路", "Nothing yet — you lead"),
                hint=i18n.pick(
                    language,
                    f"由{curator}决定这条线怎么走",
                    f"{curator} decides where the route goes",
                ),
            )
        )
        return InterviewQuestion(
            id=InterviewQuestionId.OPEN_QUESTION,
            prompt=prompt,
            options=options,
            allow_free_text=True,
            free_text_placeholder=i18n.pick(
                language,
                "也可以直接写下你自己的问题…",
                "Or write your own question…",
            ),
            skippable=True,
            step=5,
            total_steps=TOTAL_STEPS,
        )

    @staticmethod
    def _exclusions_question(
        language: str = "zh", lead: str | None = None
    ) -> InterviewQuestion:
        last = i18n.pick(
            language,
            "最后一个：有什么是你不太想看到的？",
            "Last one: is there anything you'd rather not be shown?",
        )
        return InterviewQuestion(
            id=InterviewQuestionId.EXCLUSIONS,
            prompt=f"{lead}\n{last}" if lead else last,
            options=[
                InterviewOption(
                    value=value,
                    label=i18n.pick(language, label, i18n.EXCLUSION_LABELS_EN[value]),
                )
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

        When retrieval found candidates that still await the per-object
        evidence audit, there is nothing for the visitor to decide: the audit
        runs during curation either way. Asking "audit or narrow?" exposed the
        machinery and read like a form, so that case is no longer a turn. The
        curator says it in passing instead (``negotiation_note``, spoken with
        the last question), which keeps the interview honest without making
        the visitor approve an internal step.
        """
        question_text = (
            state.profile.open_question or state.profile.free_form_question or ""
        ).strip()
        if not question_text:
            return None
        language = state.profile.language
        if entry_for_question(question_text, language) is not None:
            # The curator just offered this question as a starting point; it
            # has produced an exhibition from this collection before. The
            # generation-time evidence audit still runs.
            return None

        from .generator import ExhibitionGenerator  # local import avoids a cycle

        agenda = state.profile.to_agenda(collection.id)
        try:
            check = ExhibitionGenerator.probe_answerability(
                self.collections,
                agenda,
                audit_available=self.audit_available,
            )
        except Exception:  # noqa: BLE001 - probing must never break the interview
            return None
        if check.status == AnswerabilityStatus.SUPPORTED.value:
            if check.requires_runtime_audit:
                state.negotiation_note = i18n.pick(
                    language,
                    "我在馆藏里先找到了一批可能相关的藏品，搭展厅时会逐件核对馆方记录，对不上的会拿掉。",
                    "I've found a first set of possibly relevant objects; while building the room "
                    "I'll check each against the museum's record and drop any that don't hold up.",
                )
            return None

        # A negotiation may only offer domains that overlap this question's
        # actual evidence candidates.  Falling back to the three richest
        # collection domains would silently replace an out-of-domain question
        # with an unrelated one while claiming it was a supported part.
        evidence_domain_ids = set(check.coverage.evidence_domain_ids)
        available = [
            domain
            for domain in self._available_domains(collection, language)
            if domain[0] in evidence_domain_ids
        ][:3]
        # Prefer the hand-picked featured questions: they are in a visitor's
        # voice and were checked against this collection. The probe's question
        # cards ("不同文化如何把自然景观变成……") read like a syllabus.
        featured = [
            (entry.question(language), entry.title(language))
            for entry in featured_for(collection)
            if entry.question(language) != question_text
        ][:2]
        reviewed_alternatives = featured or [
            (question, question)
            for question in check.recommended_questions
            if question.strip() and question.strip() != question_text
        ][:2]
        limit = 90 if language == "en" else 40
        quoted = question_text if len(question_text) <= limit else question_text[: limit - 1] + "…"

        options = [
            InterviewOption(
                value=domain_id,
                label=i18n.pick(language, f"从「{label}」看起", f"Start from “{label}”"),
                hint=hint,
            )
            for domain_id, label, hint in available
        ] + [
            InterviewOption(
                value=f"{RECOMMENDED_QUESTION_PREFIX}{question}",
                label=label,
                hint=i18n.pick(
                    language,
                    "馆藏能回答的另一个问题",
                    "A different question the collection can answer",
                ),
            )
            for question, label in reviewed_alternatives
        ]
        audit_down = getattr(check, "decision_basis", None) == "audit_unavailable"
        if audit_down:
            # Saying "the collection has nothing" here would be false: the
            # service that checks candidates is down, not the collection empty.
            state.negotiation_note = i18n.pick(
                language,
                f"关于“{quoted}”，我这边核对馆藏记录的服务暂时连不上，现在没法确认馆藏能不能回答它。",
                f"For “{quoted}”, the service I use to check museum records isn't reachable right now, "
                "so I can't yet confirm whether the collection can answer it.",
            )
            follow_up = i18n.pick(
                language,
                "可以稍后再试；也可以换个问法，或从下面挑一个。"
                if options
                else "可以稍后再试，或者换个问法。",
                "You could try again shortly, put it another way, or pick one below."
                if options
                else "You could try again shortly, or put it another way.",
            )
        elif language == "en":
            if available:
                covered = " and ".join(f"“{label}”" for _id, label, _hint in available[:2])
                state.negotiation_note = (
                    f"For “{quoted}”, the objects that match directly aren't enough to fill a "
                    f"room, but there is related material under {covered}."
                )
            else:
                state.negotiation_note = (
                    f"For “{quoted}”, I haven't found enough objects that really match, and "
                    "I'd rather not pad the room with unrelated ones."
                )
            follow_up = (
                "Want to come at it from another angle? Pick one below, or put the question another way."
                if options
                else "Could you put it another way? Something more specific helps: one kind of object, a period or a place."
            )
        else:
            if available:
                covered = "、".join(f"「{label}」" for _id, label, _hint in available[:2])
                state.negotiation_note = (
                    f"关于“{quoted}”，馆里能直接对上的藏品还不够撑起一整个展厅，"
                    f"不过在{covered}这些方向上有一些相关的东西。"
                )
            else:
                state.negotiation_note = (
                    f"关于“{quoted}”，我在馆藏里还没找到足够能对上的藏品，也不想拿不相干的东西凑数。"
                )
            follow_up = (
                "要不要换个角度？可以从下面挑一个，也可以换个问法告诉我。"
                if options
                else "能换个问法吗？说得具体一点会更好找，比如一种物件、一个时代或一个地方。"
            )
        return InterviewQuestion(
            id=InterviewQuestionId.NEGOTIATION,
            prompt=f"{state.negotiation_note}\n{follow_up}",
            options=options,
            allow_free_text=True,
            free_text_placeholder=i18n.pick(
                language, "换一种更具体的问法……", "Put it another way…"
            ),
            step=4,
            total_steps=TOTAL_STEPS,
        )

    # -- public API ------------------------------------------------------

    def start(
        self, collection_id: str | None = None, language: str = "zh"
    ) -> InterviewState:
        collection = self.collections.get(collection_id)
        state = InterviewState(id=str(uuid4()), collection_id=collection.id)
        # The language is chosen before the first question and rides on the
        # profile from here on, so it reaches the exhibition record unchanged.
        state.profile.language = "en" if language == "en" else "zh"
        state.next_question = self._curiosity_question(collection, state.profile.language)
        return state

    def answer(self, state: InterviewState, answer: InterviewAnswer) -> InterviewState:
        collection = self.collections.get(state.collection_id)
        current = state.next_question
        if current is None or current.id != answer.question_id:
            # Ignore replies to a question we are no longer on rather than
            # corrupting the profile with a stale value.
            return state
        if (
            current.id == InterviewQuestionId.CUSTOM_QUESTION
            and not (answer.free_text or "").strip()
            and not (
                current.skippable
                and (answer.skipped or answer.value in {option.value for option in current.options})
            )
        ):
            # The visitor explicitly chose "there is a question I want to
            # figure out".  Do not record an empty turn and silently move on;
            # keep the input-only question active until it has an answer.
            return state

        if (
            current.id == InterviewQuestionId.FEATURED
            and not (answer.free_text or "").strip()
            and answer.value not in {option.value for option in current.options}
        ):
            # The featured turn has no skip: an unknown card id would
            # otherwise record an empty turn and leave the visit subjectless.
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
        language = profile.language

        if question_id == InterviewQuestionId.CURIOSITY:
            undecided = (answer.value == UNSURE_VALUE and not free_text) or bool(
                free_text and undecided_opening_kind(free_text)
            )
            if undecided and self._can_feature(collection):
                # Nothing to curate from yet. The next turn shows the
                # collection; nothing about this answer becomes a question.
                turn.answer_label = free_text[:60] if free_text else i18n.pick(
                    language, "由策展人推荐", "Curator's pick"
                )
            elif free_text and not undecided:
                profile.free_form_question = free_text[:500]
                matched = self._match_domain(free_text, collection)
                if matched:
                    profile.curiosity_domain_id = matched
                    profile.curiosity_label = domain_choices_for(collection, language)[matched][0]
                turn.answer_label = free_text[:60]
            elif answer.value and answer.value != UNSURE_VALUE:
                profile.curiosity_domain_id = answer.value
                profile.curiosity_label = domain_choices_for(collection, language).get(
                    answer.value, (answer.value, "")
                )[0]
                turn.answer_label = profile.curiosity_label
            else:
                # "Recommend something" with no featured objects to show (a
                # small test or legacy collection): take the richest routable
                # domain so the interview can keep moving.
                available = self._available_domains(collection, language)
                if available:
                    profile.curiosity_domain_id = available[0][0]
                    profile.curiosity_label = available[0][1]
                turn.answer_label = (
                    free_text[:60]
                    if free_text
                    else i18n.pick(language, "由策展人推荐", "Curator's pick")
                )

        elif question_id == InterviewQuestionId.FEATURED:
            entry = None
            if free_text and not undecided_opening_kind(free_text):
                profile.free_form_question = free_text[:500]
                matched = self._match_domain(free_text, collection)
                if matched:
                    profile.curiosity_domain_id = matched
                    profile.curiosity_label = domain_choices_for(collection, language)[matched][0]
                turn.answer_label = free_text[:60]
            elif free_text or answer.value == UNSURE_VALUE:
                # Asked twice and still undecided: the curator chooses, as a
                # person showing someone round would.
                entry = featured_for(collection)[0]
                turn.answer_label = free_text[:60] if free_text else i18n.pick(
                    language, "你替我定", "You choose"
                )
            else:
                entry = entry_by_id((answer.value or "")[len(FEATURED_PREFIX):])
                turn.answer_label = entry.title(language) if entry else None
            if entry is not None:
                profile.free_form_question = entry.question(language)
                profile.curiosity_domain_id = None
                profile.curiosity_label = ""

        elif question_id == InterviewQuestionId.MOTIVATION:
            if answer.value in {item.value for item in VisitorMotivation}:
                profile.motivation = VisitorMotivation(answer.value)
                turn.answer_label = i18n.pick(
                    language,
                    MOTIVATION_LABELS[answer.value],
                    i18n.MOTIVATION_LABELS_EN[answer.value],
                )

        elif question_id == InterviewQuestionId.CUSTOM_QUESTION:
            if free_text:
                profile.open_question = free_text[:300]
                turn.answer_label = free_text[:60]
                if state.next_question and state.next_question.skippable:
                    # A clarification supersedes an inferred opening domain;
                    # the visitor never explicitly selected that inference.
                    matched = self._match_domain(free_text, collection)
                    profile.curiosity_domain_id = matched
                    profile.curiosity_label = (
                        domain_choices_for(collection, language)[matched][0] if matched else ""
                    )
                else:
                    self._replace_auto_domain_from_question(state, free_text, collection)
            elif answer.value and answer.value.startswith(SCOPE_DIRECTION_PREFIX):
                domain_id = answer.value[len(SCOPE_DIRECTION_PREFIX):]
                label = domain_choices_for(collection, language)[domain_id][0]
                profile.curiosity_domain_id = domain_id
                profile.curiosity_label = label
                # Selection makes this a visitor-authorised scope, unlike the
                # old silent assignment from whichever domain was richest.
                profile.open_question = i18n.pick(
                    language,
                    f"先从{label}试逛；最初的偏好是“{profile.free_form_question}”。"
                    "这是观看方向，不是确定事实或客观排名。",
                    f"Start with {label}; my initial preference was “{profile.free_form_question}”. "
                    "This is a viewing direction, not a factual claim or objective ranking.",
                )[:300]
                turn.answer_label = i18n.pick(language, f"从「{label}」试逛", f"Try “{label}”")
            else:
                turn.answer_label = i18n.pick(language, "先保留这个偏好", "Keep the preference open")

        elif question_id == InterviewQuestionId.PRIOR_KNOWLEDGE:
            if answer.value in {"none", "some", "familiar"}:
                profile.prior_knowledge = answer.value
                turn.answer_label = i18n.pick(
                    language,
                    {"none": "第一次接触", "some": "略知一二", "familiar": "比较熟悉"}[answer.value],
                    i18n.PRIOR_KNOWLEDGE_EN[answer.value][0],
                )

        elif question_id == InterviewQuestionId.DURATION:
            if answer.value in {"5", "10", "15"}:
                profile.duration_minutes = int(answer.value)  # type: ignore[assignment]
                turn.answer_label = i18n.pick(
                    language, f"{answer.value} 分钟", f"{answer.value} minutes"
                )

        elif question_id == InterviewQuestionId.NEGOTIATION:
            if free_text:
                profile.open_question = free_text[:300]
                matched = self._match_domain(free_text, collection)
                profile.curiosity_domain_id = matched
                profile.curiosity_label = (
                    domain_choices_for(collection, language)[matched][0]
                    if matched
                    else ""
                )
                turn.answer_label = free_text[:60]
            elif answer.value and answer.value.startswith(
                RECOMMENDED_QUESTION_PREFIX
            ):
                reviewed_question = answer.value[
                    len(RECOMMENDED_QUESTION_PREFIX) :
                ].strip()[:300]
                profile.open_question = reviewed_question
                profile.curiosity_domain_id = None
                profile.curiosity_label = ""
                entry = entry_for_question(reviewed_question, language)
                turn.answer_label = (entry.title(language) if entry else reviewed_question)[:60]
            elif answer.value and answer.value != FREE_TEXT_VALUE:
                profile.curiosity_domain_id = answer.value
                profile.curiosity_label = domain_choices_for(collection, language).get(
                    answer.value, (answer.value, "")
                )[0]
                # The visitor accepted a narrower route. Make that route the
                # actual retrieval query; merely setting a filter while leaving
                # the original unsupported question active would be a false
                # negotiation because VisitorProfile.to_agenda prioritises it.
                profile.open_question = profile.curiosity_label
                turn.answer_label = i18n.pick(
                    language,
                    f"从「{profile.curiosity_label}」看起",
                    f"Start from “{profile.curiosity_label}”",
                )
            else:
                turn.answer_label = i18n.pick(
                    language, "保持原来的问题", "Keep the original question"
                )

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
                turn.answer_label = (
                    i18n.pick(language, "跳过", "Skipped")
                    if answer.skipped
                    else i18n.pick(language, "交给策展人决定", "Curator decides")
                )
            else:
                profile.open_question = chosen[:300]
                turn.answer_label = chosen[:60]
                if free_text:
                    self._replace_auto_domain_from_question(state, chosen, collection)

        elif question_id == InterviewQuestionId.EXCLUSIONS:
            if answer.skipped or not answer.value or answer.value == SKIP_VALUE:
                turn.skipped = True
                turn.answer_label = i18n.pick(language, "跳过", "Skipped")
            else:
                labels = (
                    i18n.EXCLUSION_LABELS_EN
                    if language == "en"
                    else dict(EXCLUSION_CHOICES)
                )
                chosen = [
                    value.strip()
                    for value in answer.value.split(",")
                    if value.strip() and value.strip() != "none"
                ]
                topic_words = (
                    {"religion": "religion", "funerary": "burial", "war": "war"}
                    if language == "en"
                    else {"religion": "宗教", "funerary": "墓葬", "war": "战争"}
                )
                profile.excluded_topics = [
                    topic_words.get(value, value) for value in chosen
                ]
                joiner = ", " if language == "en" else "、"
                turn.answer_label = joiner.join(
                    labels.get(value, value) for value in chosen
                ) or i18n.pick(language, "没有", "Nothing")
            if free_text:
                profile.excluded_topics.extend(
                    part.strip() for part in re.split(r"[，,、]", free_text) if part.strip()
                )

    def _next(
        self, state: InterviewState, collection: LoadedCollection
    ) -> InterviewQuestion | None:
        asked = {turn.question_id for turn in state.transcript}
        language = state.profile.language
        has_specific_question = bool(
            (state.profile.open_question or state.profile.free_form_question or "").strip()
        )
        if (
            InterviewQuestionId.FEATURED not in asked
            and not has_specific_question
            and self._opening_was_undecided(state)
            and self._can_feature(collection)
        ):
            return self._with_progress(state, self._featured_question(state, collection))
        if (
            InterviewQuestionId.CUSTOM_QUESTION not in asked
            and InterviewQuestionId.MOTIVATION not in asked
            # The featured turn already offered concrete starting points; a
            # further scope turn would push the interview past seven turns.
            and InterviewQuestionId.FEATURED not in asked
            and needs_scope_clarification(state.profile.free_form_question or "")
        ):
            return self._with_progress(state, self._scope_clarification_question(state, collection))
        if InterviewQuestionId.MOTIVATION not in asked:
            return self._with_progress(state, self._motivation_question(language))
        if (
            state.profile.motivation == VisitorMotivation.EXPLORER
            and not has_specific_question
            and InterviewQuestionId.CUSTOM_QUESTION not in asked
        ):
            return self._with_progress(state, self._custom_question_question(language))
        if InterviewQuestionId.PRIOR_KNOWLEDGE not in asked:
            return self._with_progress(
                state,
                self._prior_knowledge_question(
                    state.profile.curiosity_label,
                    language,
                    question_led=has_specific_question,
                ),
            )
        if InterviewQuestionId.DURATION not in asked:
            return self._with_progress(state, self._duration_question(language))
        # Asked before the answerability gate so that the sharper question is
        # what gets probed; negotiating over the broad topic while ignoring the
        # visitor's actual question would check the wrong thing.
        if not has_specific_question and InterviewQuestionId.OPEN_QUESTION not in asked:
            return self._with_progress(
                state,
                self.open_question_question(state.profile.curiosity_label, language=language),
            )
        if InterviewQuestionId.EXCLUSIONS not in asked:
            # The gate runs once, before the last question. Re-probing after
            # the final answer only re-ran hybrid retrieval to no effect.
            lead = None
            if InterviewQuestionId.NEGOTIATION not in asked:
                negotiation = self._negotiation_question(state, collection)
                if negotiation is not None:
                    return self._with_progress(state, negotiation)
                lead = state.negotiation_note
            return self._with_progress(state, self._exclusions_question(language, lead))
        return None

    @staticmethod
    def _can_feature(collection: LoadedCollection) -> bool:
        return len(featured_for(collection)) >= 2

    @staticmethod
    def _opening_was_undecided(state: InterviewState) -> bool:
        opening = next(
            (
                turn
                for turn in state.transcript
                if turn.question_id == InterviewQuestionId.CURIOSITY
            ),
            None,
        )
        if opening is None:
            return False
        if opening.free_text:
            return undecided_opening_kind(opening.free_text) is not None
        return opening.answer_value == UNSURE_VALUE

    def _replace_auto_domain_from_question(
        self,
        state: InterviewState,
        question_text: str,
        collection: LoadedCollection,
    ) -> None:
        """Replace only a domain that the system, rather than the visitor, chose.

        The opening ``__unsure__`` option installs the richest domain so the
        interview can keep moving. It is provisional: once the visitor later
        writes a concrete question, retrieval must follow that question. A
        domain explicitly selected or named in the opening turn is preserved.
        """

        if not self._opening_was_undecided(state):
            return
        matched = self._match_domain(question_text, collection)
        state.profile.curiosity_domain_id = matched
        state.profile.curiosity_label = (
            domain_choices_for(collection, state.profile.language)[matched][0]
            if matched
            else ""
        )

    @staticmethod
    def _with_progress(
        state: InterviewState, question: InterviewQuestion
    ) -> InterviewQuestion:
        """Number the actual conditional route without going backwards.

        A question typed into the opening prompt serves as both topic and core
        question, so that route has five visible turns.  The dedicated custom
        question and the later optional open question each keep the usual six.
        Answerability negotiation adds one turn only when it actually appears.
        """

        asked = {turn.question_id for turn in state.transcript}
        opening_already_specific = bool(state.profile.free_form_question) and not (
            InterviewQuestionId.CUSTOM_QUESTION in asked
            or InterviewQuestionId.OPEN_QUESTION in asked
        )
        total = 5 if opening_already_specific else TOTAL_STEPS
        if opening_already_specific and question.id == InterviewQuestionId.CUSTOM_QUESTION:
            total += 1
        if opening_already_specific and InterviewQuestionId.FEATURED in asked:
            # The question came from the featured turn, one turn later than
            # an opening that already carried it.
            total += 1
        if (
            InterviewQuestionId.NEGOTIATION in asked
            or question.id == InterviewQuestionId.NEGOTIATION
        ):
            total += 1
        question.step = len(state.transcript) + 1
        question.total_steps = max(total, question.step)
        return question

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
