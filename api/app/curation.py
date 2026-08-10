"""Profile-driven curation: chapters, labels, epilogue and space design.

This is the layer that turned the P0 "five objects in fixed roles" output into a
visitable exhibition. It reuses the existing retrieval, diversity re-ranking and
answerability gate in :mod:`generator`; what it adds is variable exhibition
size, chapter structure, a real curatorial theme, an epilogue, and the
``SpaceDesignSpec`` that the 3D renderer consumes.

Two constraints are enforced here and are not model-configurable:

* at least one ``contrast`` object per exhibition, so personalisation cannot
  collapse into a familiarity filter bubble;
* ``evidence_depth == thin`` objects (every Met record) may never hold a
  ``core_evidence`` role.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from uuid import uuid4

try:  # pragma: no cover - exercised only when the optional dep is present
    from opencc import OpenCC

    _TO_SIMPLIFIED = OpenCC("t2s")
except Exception:  # noqa: BLE001 - conversion is a safety net, not a requirement
    _TO_SIMPLIFIED = None

from .collections import (
    HYBRID_RETRIEVAL_METHOD,
    HYBRID_RETRIEVAL_VERSION,
    CollectionDataError,
    CollectionRepository,
    SearchResult,
)
from .models import (
    Chapter,
    CuratorialAudience,
    CuratorialBrief,
    CuratorialClaim,
    CuratorialEthics,
    CuratorialEvaluationTarget,
    CuratorialExcludedCandidate,
    CuratorialInterpretationPolicy,
    CuratorialObjectDecision,
    CuratorialRetrievalRecord,
    CuratorialRole,
    Epilogue,
    EvidenceDepth,
    Exhibition,
    ExhibitionItem,
    ExhibitionStatus,
    LabelSentence,
    MuseumObject,
    ObjectSummary,
    ROLE_LABELS,
    SentenceType,
    SpaceDesignSpec,
    VersionInfo,
    VisitorMotivation,
    VisitorPace,
    VisitorProfile,
    utc_now,
)

# Register wording per Falk motivation. Only affects tone and density; never
# which objects are selected.
REGISTER: dict[str, dict[str, str]] = {
    VisitorMotivation.EXPLORER.value: {
        "voice": "面向想弄懂问题的成年访客，讲清证据与推理链条，保留一个对照声音。",
        "density": "中等偏高",
    },
    VisitorMotivation.RECHARGER.value: {
        "voice": "面向来放松的访客，句子短、画面优先，不堆年代与术语。",
        "density": "低",
    },
    VisitorMotivation.FACILITATOR.value: {
        "voice": "面向带着同伴或孩子的访客，语言口语化，多给可以互相讨论的问题。",
        "density": "低到中等",
    },
    VisitorMotivation.PROFESSIONAL.value: {
        "voice": "面向有基础的访客，保留术语、年代分期与材质细节，可以指出争议。",
        "density": "高",
    },
}

# Space presets keyed by motivation; the model may adjust colours and lighting
# within these, but the overall form stays in a range the renderer handles well.
SPACE_PRESETS: dict[str, dict[str, Any]] = {
    VisitorMotivation.EXPLORER.value: {
        "space_form": "hall",
        "wall_color": "#e6e1d6",
        "floor_color": "#3f382f",
        "accent_color": "#8a6a3c",
        "mood": "neutral",
        "frame_style": "thin_dark",
    },
    VisitorMotivation.RECHARGER.value: {
        "space_form": "pavilion",
        "wall_color": "#efe9dd",
        "floor_color": "#5b5044",
        "accent_color": "#a8875a",
        "mood": "warm_dim",
        "frame_style": "scroll_hanging",
    },
    VisitorMotivation.FACILITATOR.value: {
        "space_form": "cloister",
        "wall_color": "#f1ece1",
        "floor_color": "#6a5c4b",
        "accent_color": "#b98d4e",
        "mood": "cool_bright",
        "frame_style": "wide_gold",
    },
    VisitorMotivation.PROFESSIONAL.value: {
        "space_form": "corridor",
        "wall_color": "#dcd8cf",
        "floor_color": "#33302b",
        "accent_color": "#6f6350",
        "mood": "dramatic",
        "frame_style": "vitrine",
    },
}


@dataclass
class CurationPlan:
    profile: VisitorProfile
    objects: list[MuseumObject]
    roles: list[str]
    chapter_sizes: list[int]
    evidence_domain_id: str | None
    collection_id: str
    collection_version: str
    institution: str
    candidate_count: int
    retrieval_method: str = HYBRID_RETRIEVAL_METHOD
    retrieval_version: str = HYBRID_RETRIEVAL_VERSION


def plan_roles(objects: list[MuseumObject]) -> tuple[list[MuseumObject], list[str]]:
    """Order objects and assign each a curatorial position.

    Returns a possibly reordered object list alongside its roles, because the
    two constraints interact: the opening and synthesis slots are structural
    and must not be overwritten, while ``core_evidence`` must land on an object
    with real institution prose. When the only full-depth object sits in a
    structural slot, moving the object is the correct fix — reassigning the
    role would leave the exhibition without a beginning or an end.
    """
    ordered = list(objects)
    count = len(ordered)
    if count > 3:
        interior = range(1, count - 1)
        has_interior_full = any(
            ordered[index].evidence_depth == EvidenceDepth.FULL.value for index in interior
        )
        if not has_interior_full:
            structural = [0, count - 1]
            donor = next(
                (
                    index
                    for index in structural
                    if ordered[index].evidence_depth == EvidenceDepth.FULL.value
                ),
                None,
            )
            if donor is not None:
                target = count // 2
                ordered[donor], ordered[target] = ordered[target], ordered[donor]

    return ordered, assign_roles(ordered)


def assign_roles(objects: list[MuseumObject]) -> list[str]:
    """Give every object a curatorial position.

    Guarantees an opening, a synthesis and at least one contrast, and never
    hands ``core_evidence`` to a thin-evidence object.

    Callers should prefer :func:`plan_roles`, which additionally reorders so a
    full-depth object is available for the core-evidence slot.
    """
    count = len(objects)
    roles: list[str] = [CuratorialRole.CORE_EVIDENCE.value] * count
    roles[0] = CuratorialRole.INTRODUCTION.value
    if count > 1:
        roles[-1] = CuratorialRole.SYNTHESIS.value
    if count > 2:
        roles[1] = CuratorialRole.HISTORICAL_CONTEXT.value

    # Place the contrast about three-quarters through, where an alternative
    # voice actually complicates an argument the visitor has already met.
    if count > 3:
        contrast_index = max(2, int(count * 0.72))
        contrast_index = min(contrast_index, count - 2)
        roles[contrast_index] = CuratorialRole.COUNTERPOINT.value

    # Thin-evidence objects cannot carry core evidence; demote them to context.
    for index, obj in enumerate(objects):
        if roles[index] != CuratorialRole.CORE_EVIDENCE.value:
            continue
        if obj.evidence_depth == EvidenceDepth.THIN.value:
            roles[index] = CuratorialRole.HISTORICAL_CONTEXT.value

    # Demotion can strip the last core-evidence slot — a short exhibition has
    # only one, and it may well have landed on a metadata-only object. Promote
    # a full-depth object rather than shipping an argument with nothing at its
    # centre. Interior positions first, so the opening and closing keep their
    # framing roles.
    if CuratorialRole.CORE_EVIDENCE.value not in roles:
        # Interior positions only: overwriting index 0 or the last slot would
        # leave the exhibition without an opening or a synthesis.
        for index in range(1, max(1, count - 1)):
            if objects[index].evidence_depth == EvidenceDepth.FULL.value:
                roles[index] = CuratorialRole.CORE_EVIDENCE.value
                break
        # If every selected object is metadata-only there is no honest core
        # evidence to assign; selection upstream is responsible for preventing
        # that, and the validator reports it rather than this function faking it.

    if CuratorialRole.COUNTERPOINT.value not in roles:
        # Never ship an exhibition without an alternative voice.
        for index in range(count - 2, 0, -1):
            if roles[index] == CuratorialRole.HISTORICAL_CONTEXT.value:
                roles[index] = CuratorialRole.COUNTERPOINT.value
                break
        else:
            roles[max(1, count // 2)] = CuratorialRole.COUNTERPOINT.value
    return roles


def chapter_sizes(item_count: int, chapter_count: int) -> list[int]:
    """Split items across chapters, front-loading the remainder."""
    base = item_count // chapter_count
    remainder = item_count % chapter_count
    return [base + (1 if index < remainder else 0) for index in range(chapter_count)]


def ensure_core_evidence_candidate(
    selected: list[MuseumObject], pool: list[SearchResult]
) -> list[MuseumObject]:
    """Guarantee the selection can support a core-evidence role.

    Relevance and diversity alone can return an all-thin set — several coverage
    domains are dominated by an institution that publishes no curatorial prose,
    so the highest-scoring objects there are all metadata-only. Without this,
    the exhibition would be assembled and only then fail validation for having
    nothing at the centre of its argument.

    Swaps the weakest selected object for the strongest full-depth candidate.
    If the pool has no full-depth object at all, the selection is returned
    unchanged and the validator reports the gap rather than hiding it.
    """
    if any(obj.evidence_depth == EvidenceDepth.FULL.value for obj in selected):
        return selected

    chosen = {obj.id for obj in selected}
    replacement = next(
        (
            result.obj
            for result in pool
            if result.obj.evidence_depth == EvidenceDepth.FULL.value
            and result.obj.id not in chosen
        ),
        None,
    )
    if replacement is None:
        return selected
    return selected[:-1] + [replacement]


def order_for_narrative(
    results: list[SearchResult],
    count: int,
    *,
    prefer_culture_diversity: bool = False,
) -> list[MuseumObject]:
    """Pick objects, then sequence them chronologically where dates allow.

    Selection reuses the diversity re-ranker; the extra steps here are
    guaranteeing usable evidence depth and imposing an order, because a
    walkable hall needs a defensible sequence, not a relevance list.
    """
    from .generator import ExhibitionGenerator  # local import avoids a cycle

    selected = ExhibitionGenerator._diverse_selection(
        results,
        count,
        prefer_culture_diversity=prefer_culture_diversity,
    )
    selected = ensure_core_evidence_candidate(selected, results)

    def sort_key(obj: MuseumObject) -> tuple[int, str]:
        match = re.search(r"-?\d{3,4}", obj.date or "")
        return (int(match.group()) if match else 9999, obj.id)

    dated = [obj for obj in selected if re.search(r"\d{3,4}", obj.date or "")]
    # Only impose a timeline when most objects actually carry a date.
    if len(dated) >= max(3, int(len(selected) * 0.6)):
        return sorted(selected, key=sort_key)
    return selected


def build_space_design(profile: VisitorProfile, overrides: dict[str, Any] | None = None) -> SpaceDesignSpec:
    """Compose the renderer parameters, clamping anything the model supplied."""
    preset = dict(SPACE_PRESETS[profile.motivation])
    preset["pacing"] = profile.pace
    preset["light_temperature"] = {
        "warm_dim": 2900,
        "neutral": 3600,
        "cool_bright": 4600,
        "dramatic": 3100,
    }[preset["mood"]]
    preset["light_intensity"] = {
        "warm_dim": 0.75,
        "neutral": 0.95,
        "cool_bright": 1.15,
        "dramatic": 0.6,
    }[preset["mood"]]
    preset["typography"] = "song" if profile.motivation == VisitorMotivation.PROFESSIONAL.value else "serif"

    if overrides:
        allowed = {
            "wall_color",
            "floor_color",
            "accent_color",
            "ceiling_color",
            "space_form",
            "mood",
            "frame_style",
            "typography",
        }
        for key, value in overrides.items():
            if key in allowed and isinstance(value, str):
                preset[key] = value
    try:
        return SpaceDesignSpec(**preset)
    except ValueError:
        # A bad colour from the model must not lose the whole exhibition.
        return SpaceDesignSpec(pacing=VisitorPace(profile.pace))


def to_simplified(text: str) -> str:
    """Normalise model output to Simplified Chinese.

    The prompt asks for Simplified, but a model trained heavily on classical
    art writing drifts into Traditional — and an exhibition that mixes the two
    reads as broken. Converting deterministically means the requirement does
    not depend on the model complying. Institution source text is never passed
    through here; only text the system itself wrote.
    """
    if not text or _TO_SIMPLIFIED is None:
        return text
    return _TO_SIMPLIFIED.convert(text)


def display_title(obj: MuseumObject) -> str:
    """Chinese-first title for an object.

    Prefers the institution's own Chinese title when it has one; otherwise the
    English title stands in until the model supplies a translation, so the hall
    is never missing a label.
    """
    original = (obj.title_original or "").strip()
    if original and re.search(r"[㐀-鿿]", original):
        return original
    return obj.title


def label_sentences(
    obj: MuseumObject,
    item_id: str,
    role_label: str,
    max_chars: int,
) -> list[LabelSentence]:
    """Build the deterministic label floor.

    The institution sentence quotes the record verbatim and is never model
    generated; the second sentence states, in the system's own voice, why the
    object sits where it sits.
    """
    evidence = obj.evidence[0]
    original = re.sub(r"\s+", " ", evidence.text).strip()
    if len(original) > max_chars:
        original = original[: max_chars - 1].rstrip() + "…"
    sentences = [
        LabelSentence(
            id=f"{item_id}-s1",
            text=original,
            type=SentenceType.INSTITUTION_FACT,
            evidence_ids=[evidence.id],
        ),
        LabelSentence(
            id=f"{item_id}-s2",
            text=f"本展把它放在“{role_label}”的位置。",
            type=SentenceType.SYSTEM_INFERENCE,
            evidence_ids=[evidence.id],
        ),
    ]
    return sentences


def build_chapters(items: list[ExhibitionItem], sizes: list[int], theme: str) -> list[Chapter]:
    """Group items into chapters with deterministic placeholder titles.

    The model replaces the titles and lead-ins when it is available; this keeps
    the structure valid when it is not.
    """
    fallback_titles = ["起点", "展开", "转折", "回望", "延伸"]
    chapters: list[Chapter] = []
    cursor = 0
    for index, size in enumerate(sizes):
        slice_items = items[cursor : cursor + size]
        cursor += size
        if not slice_items:
            continue
        chapters.append(
            Chapter(
                id=str(uuid4()),
                order=index,
                title=fallback_titles[index % len(fallback_titles)],
                lead_in=f"围绕“{theme}”，这一部分收了 {len(slice_items)} 件展品。",
                item_ids=[item.id for item in slice_items],
                space_hint=None,
            )
        )
    return chapters


def build_epilogue(profile: VisitorProfile, limits: list[str], theme: str) -> Epilogue:
    return Epilogue(
        text=(
            f"关于“{theme}”，这条线索到这里告一段落。"
            "它不是唯一的讲法，只是用这批公开馆藏能搭出的一条。"
        ),
        open_questions=[
            "这些器物离开原来的使用场景后，意义改变了多少？",
            "如果换一批藏品，同一个问题会得到不一样的回答吗？",
        ],
        material_boundary=limits,
    )


def _brief_evidence_ids(
    item: ExhibitionItem,
    result_by_object: dict[str, SearchResult],
    *,
    limit: int = 2,
) -> list[str]:
    """Choose real, query-relevant evidence IDs for one brief decision."""

    allowed = {chunk.id for chunk in item.object.evidence}
    result = result_by_object.get(item.object.id)
    matched = [
        evidence_id
        for evidence_id in (result.matched_evidence_ids if result else ())
        if evidence_id in allowed
    ]
    fallback = [chunk.id for chunk in item.object.evidence if chunk.id not in matched]
    return (matched + fallback)[:limit]


def build_curatorial_brief(
    plan: CurationPlan,
    inquiry: str,
    items: list[ExhibitionItem],
    chapters: list[Chapter],
    candidate_results: list[SearchResult],
) -> CuratorialBrief:
    """Create the auditable reasoning contract before any model prose.

    The deterministic wording is deliberately modest: it records the planned
    use of the selected records, not historical claims that the source text may
    not support.  A model may refine this contract only through
    :func:`apply_frame`, where every supplied evidence ID is checked against the
    actual selected object.
    """

    result_by_object = {result.obj.id: result for result in candidate_results}
    decision_evidence = {
        item.object.id: _brief_evidence_ids(item, result_by_object)
        for item in items
    }
    selected_ids = {item.object.id for item in items}

    core_first = sorted(
        items,
        key=lambda item: (
            0 if item.role == CuratorialRole.CORE_EVIDENCE.value else 1,
            0 if item.role == CuratorialRole.COUNTERPOINT.value else 1,
            item.order,
        ),
    )
    big_idea_evidence: list[str] = []
    for item in core_first:
        for evidence_id in decision_evidence[item.object.id]:
            if evidence_id not in big_idea_evidence:
                big_idea_evidence.append(evidence_id)
        if len(big_idea_evidence) >= 4:
            break

    by_item_id = {item.id: item for item in items}
    key_messages: list[CuratorialClaim] = []
    for index, chapter in enumerate(chapters, start=1):
        chapter_items = [
            by_item_id[item_id] for item_id in chapter.item_ids if item_id in by_item_id
        ]
        evidence_ids: list[str] = []
        for item in chapter_items:
            for evidence_id in decision_evidence[item.object.id]:
                if evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)
        titles = "、".join(f"《{display_title(item.object)}》" for item in chapter_items[:2])
        key_messages.append(
            CuratorialClaim(
                id=f"key-message-{index}",
                text=f"第{index}部分以{titles or '所选藏品'}的机构记录展开问题线索。",
                evidence_ids=evidence_ids[:4],
                confidence="provisional",
            )
        )

    excluded: list[CuratorialExcludedCandidate] = []
    for result in candidate_results:
        if result.obj.id in selected_ids:
            continue
        valid_ids = {chunk.id for chunk in result.obj.evidence}
        matched_ids = [
            evidence_id
            for evidence_id in result.matched_evidence_ids
            if evidence_id in valid_ids
        ][:2]
        excluded.append(
            CuratorialExcludedCandidate(
                object_id=result.obj.id,
                title=result.obj.title,
                reason=(
                    "与访客问题相关，但在本次展品数量、角色覆盖与多样性约束下未入选。"
                ),
                evidence_ids=matched_ids,
            )
        )
        if len(excluded) == 5:
            break

    audience = CuratorialAudience(
        motivation=plan.profile.motivation,
        prior_knowledge=plan.profile.prior_knowledge,
        duration_minutes=plan.profile.duration_minutes,
        excluded_topics=plan.profile.excluded_topics,
        voice=REGISTER[plan.profile.motivation]["voice"],
        density=REGISTER[plan.profile.motivation]["density"],
    )
    critical_questions = [
        item.sub_question for item in items if item.sub_question
    ]
    critical_questions = list(dict.fromkeys(critical_questions))[:4]
    while len(critical_questions) < 2:
        critical_questions.append("这批馆藏记录还留下了哪些无法回答的问题？")

    return CuratorialBrief(
        visitor_inquiry=inquiry,
        audience=audience,
        big_idea=CuratorialClaim(
            id="big-idea",
            text=f"用所选馆藏记录检视“{inquiry}”这一问题。",
            evidence_ids=big_idea_evidence,
            confidence="provisional",
        ),
        key_messages=key_messages,
        critical_questions=critical_questions,
        objects=[
            CuratorialObjectDecision(
                item_id=item.id,
                object_id=item.object.id,
                role=item.role,
                selection_rationale=item.why_selected,
                relation=item.relation,
                evidence_ids=decision_evidence[item.object.id],
            )
            for item in items
        ],
        excluded_candidates=excluded,
        ethics=CuratorialEthics(
            provenance_status="not_reviewed",
            provenance_notes=["公开馆藏记录中的来源信息尚未经过独立来源研究。"],
            cultural_sensitivity_status="not_reviewed",
            cultural_sensitivity=[],
            community_review_required=None,
            community_review_status="not_assessed",
            community_review_notes=["尚未评估是否需要来源社群或文化持有者审阅。"],
        ),
        interpretation_policy=CuratorialInterpretationPolicy(
            fact_policy="事实只能摘取并引用馆方记录。",
            inference_policy="系统解释必须与馆方事实分层显示并绑定证据 ID。",
            uncertainty_policy="证据不足、存在歧义或跨文化关系不明确时标记为不确定。",
            external_knowledge_allowed=False,
        ),
        evaluation_targets=[
            CuratorialEvaluationTarget(
                id="understand-big-idea",
                statement="访客能够用自己的话复述本展的核心问题与主要判断。",
                method="comprehension_check",
            ),
            CuratorialEvaluationTarget(
                id="trace-object-role",
                statement="访客能够说明至少一件藏品为何入选及其在展线中的作用。",
                method="visitor_prompt",
            ),
            CuratorialEvaluationTarget(
                id="review-evidence-boundary",
                statement="领域审阅者能够区分馆方事实、系统解释与尚未核实的判断。",
                method="expert_review",
            ),
        ],
        retrieval=CuratorialRetrievalRecord(
            method=plan.retrieval_method,
            version=plan.retrieval_version,
            candidate_count=plan.candidate_count,
            selected_count=len(items),
        ),
    )


# Curation is split into two model calls rather than one.
#
# Asking for a whole exhibition — theme, chapters, epilogue, space and a
# three-sentence label for every object — in a single response pushed a
# reasoning model past three minutes. Splitting gives three things at once:
# the frame arrives in seconds so the visitor sees a real title early; the
# label calls are independent per chapter and run concurrently, so total
# latency is the slowest chapter rather than the sum; and a failure in one
# chapter's labels no longer costs the whole exhibition.

FRAME_PROMPT = """你是一位数字策展人，为一位具体的访客编排一场小型虚拟展览。
展览面向中文读者，**全部输出使用简体中文**（不要使用繁体字）。

硬约束：
1. 只能依据每件 object 的 evidence 摘要，不得补写材料之外的人名、年代、因果或价值判断。
2. 展品的顺序与 role 已经确定，不得更改、增删。
3. chapters 的数量必须与输入 chapterCount 一致。
4. bigIdea、每条 keyMessage 和每件展品的 selectionRationale/relation 都必须引用 evidenceIds；ID 只能从输入 evidence 原样选择，绝对不得创造、改写或跨展品挪用。
5. 证据只能支持局部观察时，把 confidence 写成 provisional 或 uncertain，不要把相似外观写成跨文化因果或共同象征。
6. 章节引导语和结语只能改写 bigIdea/keyMessages，不得引入新的事实主张。

先完成结构化 curatorialBrief，再据此写展览框架；不要写展签。输出单个 JSON 对象：
{
  "title": 展览主题名，12-24 字，像展览标题而不是问句,
  "subtitle": 副标题，15-30 字,
  "curatorialBrief": {
    "bigIdea": {"text": 一句话可讨论的策展命题, "evidenceIds": [...], "confidence": "supported|provisional|uncertain"},
    "keyMessages": [{"text": 支撑命题的分论点, "evidenceIds": [...], "confidence": "supported|provisional|uncertain"}],
    "criticalQuestions": [2-4 个关键问题],
    "objects": [{"objectId": 输入objectId, "role": 输入role, "selectionRationale": 入选理由, "relation": 与前后展品的具体关系, "evidenceIds": [...]}],
    "evaluationTargets": [{"statement": 访客可理解或专业审阅的具体目标, "method": "visitor_prompt|comprehension_check|expert_review"}]
  },
  "chapters": [{"title": 章节名 4-10 字, "leadIn": 章节引导语 40-90 字}],
  "epilogue": {"text": 结语 2-3 句, "openQuestions": [2-3 个仍未解决的问题]},
  "spaceDesign": {"wallColor": "#rrggbb", "floorColor": "#rrggbb", "accentColor": "#rrggbb"}
}"""

LABELS_PROMPT_TEMPLATE = """你在为一场中文虚拟展览写展签。**全部输出使用简体中文**（不要使用繁体字）。

硬约束：
1. 只能使用每件展品自带的 evidence，不得补写材料之外的人名、年代、因果或价值判断。
2. 不得改写机构原文；你写的每条句子都标为 system_inference，无把握的标为 uncertain。
3. evidenceIds 只能引用同一件展品自己的 evidence id。
4. 必须为输入的每一件展品输出一条记录，objectId 原样返回。
5. 关联句必须遵循输入 curatorialBrief 的 bigIdea、keyMessages 与 objectDecision，不得另起一套策展论点。

展品名：displayTitle 必须是中文。
  · 若 titleOriginal 已是中文，直接沿用；
  · 否则把英文题名意译成简洁的中文展品名，不要音译，不要保留英文。

展签：每件展品写 3 条 labelSentence，合起来像一段真正的展签：
  1) 描述：观众此刻看到的是什么——形制、材质、画面或纹饰的具体特征；
  2) 语境：它出自什么时代、什么用途或什么传统，为什么会长成这样；
  3) 关联：它在本章承担什么，与同章其他展品形成什么关系。
每条 40–LABEL_MAX 字，具体、克制，不写抒情套话。材料没说的不要写。
不要在句子里写「（推断）」「（据著录）」之类的标注——句子类型已由 type 字段标明。

输出单个 JSON 对象：
{"items": [{"objectId": ..., "displayTitle": 中文展品名,
  "labelSentences": [{"text": ..., "type": "system_inference", "evidenceIds": [...]}]}]}"""


# The model writes relational text; it does not need every word of every
# source. Bounded excerpts keep each call small enough to return quickly.
MAX_EVIDENCE_CHUNKS_PER_OBJECT = 3
MAX_EVIDENCE_CHARS = 420
MAX_FRAME_EVIDENCE_CHUNKS_PER_OBJECT = 2
MAX_FRAME_EVIDENCE_CHARS = 280


def frame_prompt() -> str:
    return FRAME_PROMPT


def labels_prompt(label_max: int) -> str:
    """Prompt with the visitor's per-sentence budget substituted in.

    Uses a placeholder token rather than ``str.format`` because the prompt
    contains a literal JSON skeleton full of braces.
    """
    return LABELS_PROMPT_TEMPLATE.replace("LABEL_MAX", str(label_max))


def _visitor_brief(profile: VisitorProfile) -> dict[str, Any]:
    register = REGISTER[profile.motivation]
    return {
        "curiosity": profile.curiosity_label,
        "question": profile.free_form_question,
        "motivation": profile.motivation,
        "priorKnowledge": profile.prior_knowledge,
        "durationMinutes": profile.duration_minutes,
        "voice": register["voice"],
        "density": register["density"],
        "labelMaxChars": profile.label_max_chars,
    }


def frame_payload(
    plan: "CurationPlan",
    items: list[ExhibitionItem],
    chapters: list[Chapter],
    brief: CuratorialBrief | None = None,
) -> dict[str, Any]:
    """Bounded evidence payload for a claim-mapped curatorial frame."""
    by_id = {item.id: item for item in items}
    brief_by_object = {
        decision.object_id: decision for decision in (brief.objects if brief else [])
    }

    def object_payload(item: ExhibitionItem) -> dict[str, Any]:
        decision = brief_by_object.get(item.object.id)
        preferred_ids = (
            decision.evidence_ids
            if decision
            else [chunk.id for chunk in item.object.evidence[:1]]
        )
        priority = {
            evidence_id: index
            for index, evidence_id in enumerate(preferred_ids)
        }
        chunks = sorted(
            item.object.evidence,
            key=lambda chunk: (priority.get(chunk.id, len(priority)), chunk.id),
        )[:MAX_FRAME_EVIDENCE_CHUNKS_PER_OBJECT]
        return {
            "itemId": item.id,
            "objectId": item.object.id,
            "title": item.object.title,
            "titleOriginal": item.object.title_original,
            "date": item.object.date,
            "type": item.object.type,
            "medium": item.object.medium,
            "culture": item.object.culture,
            "role": item.role,
            "roleLabel": item.role_label,
            "currentSelectionRationale": item.why_selected,
            "currentRelation": item.relation,
            "evidence": [
                {
                    "id": chunk.id,
                    "text": chunk.text[:MAX_FRAME_EVIDENCE_CHARS],
                    "supports": chunk.supports,
                    "sourceKind": chunk.source_kind,
                }
                for chunk in chunks
            ],
        }

    return {
        "visitor": _visitor_brief(plan.profile),
        "retrieval": {
            "method": plan.retrieval_method,
            "version": plan.retrieval_version,
            "candidateCount": plan.candidate_count,
        },
        "chapterCount": len(chapters),
        "chapters": [
            {
                "index": chapter.order,
                "items": [
                    object_payload(by_id[item_id])
                    for item_id in chapter.item_ids
                    if item_id in by_id
                ],
            }
            for chapter in chapters
        ],
    }


def labels_payload(
    profile: VisitorProfile,
    chapter: Chapter,
    items: list[ExhibitionItem],
    brief: CuratorialBrief | None = None,
) -> dict[str, Any]:
    """Payload for one chapter's labels."""
    decision_by_object = {
        decision.object_id: decision for decision in (brief.objects if brief else [])
    }

    def object_payload(item: ExhibitionItem) -> dict[str, Any]:
        decision = decision_by_object.get(item.object.id)
        preferred_ids = decision.evidence_ids if decision else []
        priority = {
            evidence_id: index for index, evidence_id in enumerate(preferred_ids)
        }
        chunks = sorted(
            item.object.evidence,
            key=lambda chunk: (priority.get(chunk.id, len(priority)), chunk.id),
        )[:MAX_EVIDENCE_CHUNKS_PER_OBJECT]
        visible_ids = {chunk.id for chunk in chunks}

        # The decision is context for writing the relational sentence, so it
        # may cite only evidence whose text is present in this exact call.
        # This prevents an ID-only reference from laundering hidden evidence
        # into the label prompt.
        raw_decision = None
        if decision is not None:
            raw_decision = decision.model_dump(mode="json", by_alias=True)
            raw_decision["evidenceIds"] = [
                evidence_id
                for evidence_id in decision.evidence_ids
                if evidence_id in visible_ids
            ]

        return {
            "objectId": item.object.id,
            "role": item.role,
            "roleLabel": item.role_label,
            "title": item.object.title,
            "titleOriginal": item.object.title_original,
            "date": item.object.date,
            "medium": item.object.medium,
            "culture": item.object.culture,
            "institution": item.object.institution,
            "objectDecision": raw_decision,
            "evidence": [
                {
                    "id": chunk.id,
                    "text": chunk.text[:MAX_EVIDENCE_CHARS],
                    "supports": chunk.supports,
                }
                for chunk in chunks
            ],
        }

    return {
        "visitor": _visitor_brief(profile),
        "curatorialBrief": (
            {
                "bigIdea": brief.big_idea.model_dump(mode="json", by_alias=True),
                "keyMessages": [
                    message.model_dump(mode="json", by_alias=True)
                    for message in brief.key_messages
                ],
                "criticalQuestions": brief.critical_questions,
            }
            if brief
            else None
        ),
        "chapter": {"title": chapter.title, "leadIn": chapter.lead_in},
        "items": [object_payload(item) for item in items],
    }


def _frame_evidence_ids(exhibition: Exhibition) -> dict[str, set[str]]:
    """Reconstruct the exact evidence-ID whitelist sent to the frame model."""

    brief_by_object = {
        decision.object_id: decision
        for decision in (
            exhibition.curatorial_brief.objects
            if exhibition.curatorial_brief
            else []
        )
    }
    allowed: dict[str, set[str]] = {}
    for item in exhibition.items:
        decision = brief_by_object.get(item.object.id)
        preferred = decision.evidence_ids if decision else []
        priority = {evidence_id: index for index, evidence_id in enumerate(preferred)}
        chunks = sorted(
            item.object.evidence,
            key=lambda chunk: (priority.get(chunk.id, len(priority)), chunk.id),
        )[:MAX_FRAME_EVIDENCE_CHUNKS_PER_OBJECT]
        allowed[item.object.id] = {chunk.id for chunk in chunks}
    return allowed


def _claim_from_output(
    raw: Any,
    *,
    claim_id: str,
    allowed_evidence_ids: set[str],
) -> CuratorialClaim:
    if not isinstance(raw, dict):
        raise ValueError(f"curatorial brief claim '{claim_id}' must be an object")
    text = str(raw.get("text", "")).strip()
    references = raw.get("evidenceIds")
    confidence = str(raw.get("confidence", "provisional"))
    if not text or not isinstance(references, list) or not references:
        raise ValueError(f"curatorial brief claim '{claim_id}' lacks text or evidence")
    references = [str(value) for value in references]
    if len(references) != len(set(references)):
        raise ValueError(f"curatorial brief claim '{claim_id}' repeats evidence IDs")
    if not set(references) <= allowed_evidence_ids:
        raise ValueError(f"curatorial brief claim '{claim_id}' invents evidence IDs")
    if confidence not in {"supported", "provisional", "uncertain"}:
        raise ValueError(f"curatorial brief claim '{claim_id}' has invalid confidence")
    return CuratorialClaim(
        id=claim_id,
        text=to_simplified(text),
        evidence_ids=references,
        confidence=confidence,
    )


def _apply_brief_output(exhibition: Exhibition, raw: Any) -> None:
    """Apply a model-refined brief only after strict evidence whitelisting."""

    if exhibition.curatorial_brief is None:
        raise ValueError("curatorial brief output has no deterministic base contract")
    if not isinstance(raw, dict):
        raise ValueError("curatorialBrief must be an object")

    allowed_by_object = _frame_evidence_ids(exhibition)
    allowed_global = set().union(*allowed_by_object.values()) if allowed_by_object else set()
    big_idea = _claim_from_output(
        raw.get("bigIdea"),
        claim_id="big-idea",
        allowed_evidence_ids=allowed_global,
    )

    raw_messages = raw.get("keyMessages")
    if not isinstance(raw_messages, list) or not 1 <= len(raw_messages) <= 5:
        raise ValueError("curatorialBrief needs one to five keyMessages")
    key_messages = [
        _claim_from_output(
            message,
            claim_id=f"key-message-{index}",
            allowed_evidence_ids=allowed_global,
        )
        for index, message in enumerate(raw_messages, start=1)
    ]

    raw_questions = raw.get("criticalQuestions")
    if not isinstance(raw_questions, list) or not 2 <= len(raw_questions) <= 4:
        raise ValueError("curatorialBrief needs two to four criticalQuestions")
    critical_questions = [
        to_simplified(str(value).strip())
        for value in raw_questions
        if str(value).strip()
    ]
    if len(critical_questions) != len(raw_questions):
        raise ValueError("curatorialBrief contains a blank critical question")

    item_by_object = {item.object.id: item for item in exhibition.items}
    raw_objects = raw.get("objects")
    if not isinstance(raw_objects, list):
        raise ValueError("curatorialBrief objects must be an array")
    raw_object_ids = [
        str(value.get("objectId", "")) if isinstance(value, dict) else ""
        for value in raw_objects
    ]
    if (
        len(raw_object_ids) != len(set(raw_object_ids))
        or set(raw_object_ids) != set(item_by_object)
    ):
        raise ValueError("curatorialBrief must return every selected object exactly once")

    decisions: list[CuratorialObjectDecision] = []
    for raw_decision in raw_objects:
        assert isinstance(raw_decision, dict)  # guarded by object-id check
        object_id = str(raw_decision.get("objectId"))
        item = item_by_object[object_id]
        role = str(raw_decision.get("role", ""))
        if role != str(item.role):
            raise ValueError("curatorialBrief may not reassign object roles")
        rationale = str(raw_decision.get("selectionRationale", "")).strip()
        relation = str(raw_decision.get("relation", "")).strip()
        references = raw_decision.get("evidenceIds")
        if not rationale or not relation or not isinstance(references, list) or not references:
            raise ValueError("curatorialBrief object decision is incomplete")
        references = [str(value) for value in references]
        if len(references) != len(set(references)):
            raise ValueError("curatorialBrief object decision repeats evidence IDs")
        if not set(references) <= allowed_by_object[object_id]:
            raise ValueError("curatorialBrief object decision cites foreign or hidden evidence")
        rationale = to_simplified(rationale)
        relation = to_simplified(relation)
        item.why_selected = rationale
        item.relation = relation
        decisions.append(
            CuratorialObjectDecision(
                item_id=item.id,
                object_id=object_id,
                role=item.role,
                selection_rationale=rationale,
                relation=relation,
                evidence_ids=references,
            )
        )

    raw_targets = raw.get("evaluationTargets")
    if not isinstance(raw_targets, list) or not 1 <= len(raw_targets) <= 6:
        raise ValueError("curatorialBrief needs one to six evaluationTargets")
    evaluation_targets: list[CuratorialEvaluationTarget] = []
    for index, target in enumerate(raw_targets, start=1):
        if not isinstance(target, dict):
            raise ValueError("curatorialBrief evaluation target must be an object")
        statement = str(target.get("statement", "")).strip()
        method = str(target.get("method", ""))
        if not statement or method not in {
            "visitor_prompt",
            "comprehension_check",
            "expert_review",
        }:
            raise ValueError("curatorialBrief evaluation target is invalid")
        evaluation_targets.append(
            CuratorialEvaluationTarget(
                id=f"evaluation-{index}",
                statement=to_simplified(statement),
                method=method,
            )
        )

    exhibition.curatorial_brief = exhibition.curatorial_brief.model_copy(
        update={
            "status": "model_refined",
            "big_idea": big_idea,
            "key_messages": key_messages,
            "critical_questions": critical_questions,
            "objects": decisions,
            "evaluation_targets": evaluation_targets,
        }
    )
    exhibition.curatorial_thesis = big_idea.text
    exhibition.core_answer = "".join(
        message.text if message.text.endswith(("。", "！", "？")) else message.text + "。"
        for message in key_messages
    )
    exhibition.sub_questions = critical_questions


def apply_frame(exhibition: Exhibition, output: dict[str, Any]) -> Exhibition:
    """Fold the frame call into the exhibition.

    Raises ``ValueError`` if the essentials are missing, so the caller can keep
    the deterministic text rather than ship a half-applied result.
    """
    # Apply transactionally. A forged evidence ID late in the response must
    # not leave an accepted title or half-updated object rationales behind.
    exhibition = exhibition.model_copy(deep=True)
    for key, attribute in (("title", "title"), ("subtitle", "subtitle")):
        value = output.get(key)
        if isinstance(value, str) and value.strip():
            setattr(exhibition, attribute, to_simplified(value.strip()))
        elif key == "title":
            raise ValueError(f"model output missing {key}")

    raw_brief = output.get("curatorialBrief")
    if raw_brief is not None:
        _apply_brief_output(exhibition, raw_brief)
    else:
        # Compatibility adapter for stored fixtures and the legacy sync path.
        # New profile generations are prompted to return ``curatorialBrief``.
        for key, attribute in (
            ("curatorialThesis", "curatorial_thesis"),
            ("coreAnswer", "core_answer"),
        ):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                setattr(exhibition, attribute, to_simplified(value.strip()))
            elif key == "curatorialThesis":
                raise ValueError(f"model output missing {key}")

        sub_questions = output.get("subQuestions")
        if isinstance(sub_questions, list) and 2 <= len(sub_questions) <= 4:
            exhibition.sub_questions = [
                to_simplified(str(value).strip())
                for value in sub_questions
                if str(value).strip()
            ]

    raw_chapters = output.get("chapters")
    if isinstance(raw_chapters, list) and len(raw_chapters) == len(exhibition.chapters):
        for chapter, raw in zip(exhibition.chapters, raw_chapters, strict=True):
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title", "")).strip()
            lead_in = str(raw.get("leadIn", "")).strip()
            if title:
                chapter.title = to_simplified(title[:24])
            if lead_in:
                chapter.lead_in = to_simplified(lead_in)

    raw_epilogue = output.get("epilogue")
    if isinstance(raw_epilogue, dict):
        text = str(raw_epilogue.get("text", "")).strip()
        if text:
            exhibition.epilogue.text = to_simplified(text)
        questions = raw_epilogue.get("openQuestions")
        if isinstance(questions, list):
            cleaned = [
                to_simplified(str(value).strip()) for value in questions if str(value).strip()
            ]
            if cleaned:
                exhibition.epilogue.open_questions = cleaned[:3]

    raw_space = output.get("spaceDesign")
    if isinstance(raw_space, dict) and exhibition.visitor_profile:
        overrides = {
            "wall_color": raw_space.get("wallColor"),
            "floor_color": raw_space.get("floorColor"),
            "accent_color": raw_space.get("accentColor"),
        }
        exhibition.space_design = build_space_design(
            exhibition.visitor_profile,
            {key: value for key, value in overrides.items() if isinstance(value, str)},
        )

    return exhibition


def apply_labels(
    items: list[ExhibitionItem],
    output: dict[str, Any],
    max_chars: int,
    allowed_evidence_by_object: dict[str, set[str]] | None = None,
) -> int:
    """Fold one chapter's labels in transactionally.

    ``allowed_evidence_by_object`` should be derived from the exact payload
    sent for this call.  Keeping it optional preserves compatibility with the
    legacy single-shot helper, whose payload contains every evidence chunk.
    No item is mutated until all returned records have passed validation.
    """
    raw_items = output.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("labels output has no items")

    by_object = {item.object.id: item for item in items}
    pending: dict[str, tuple[str | None, list[LabelSentence]]] = {}
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        item = by_object.get(str(raw.get("objectId", "")))
        if item is None:
            continue

        translated = str(raw.get("displayTitle", "")).strip()
        # Only accept a translation that is actually Chinese; an echoed English
        # title would silently defeat the whole requirement.
        accepted_title = (
            to_simplified(translated[:60])
            if translated and re.search(r"[\u3400-\u9fff]", translated)
            else None
        )

        evidence_ids = (
            allowed_evidence_by_object.get(item.object.id, set())
            if allowed_evidence_by_object is not None
            else {chunk.id for chunk in item.object.evidence}
        )
        sentences = raw.get("labelSentences")
        if not isinstance(sentences, list) or not sentences:
            continue

        # Institution sentences survive untouched; the model only appends.
        kept = [
            sentence.model_copy(deep=True)
            for sentence in item.label_sentences
            if sentence.type == SentenceType.INSTITUTION_FACT.value
        ]
        for index, sentence in enumerate(sentences):
            if not isinstance(sentence, dict):
                continue
            text = str(sentence.get("text", "")).strip()
            references = sentence.get("evidenceIds")
            sentence_type = sentence.get("type", SentenceType.SYSTEM_INFERENCE.value)
            if not text or not isinstance(references, list) or not references:
                continue
            if not set(references) <= evidence_ids:
                raise ValueError(
                    "model sentence cites evidence from another object or hidden payload evidence"
                )
            if sentence_type == SentenceType.INSTITUTION_FACT.value:
                raise ValueError("model may not author institution facts")
            if sentence_type not in {
                SentenceType.SYSTEM_INFERENCE.value,
                SentenceType.UNCERTAIN.value,
            }:
                continue
            kept.append(
                LabelSentence(
                    id=f"{item.id}-m{index + 1}",
                    text=to_simplified(text[:max_chars]),
                    type=sentence_type,
                    evidence_ids=references,
                )
            )
        if len(kept) > 1:
            pending[item.object.id] = (accepted_title, kept)

    for object_id, (translated, kept) in pending.items():
        item = by_object[object_id]
        if translated is not None:
            item.display_title = translated
        item.label_sentences = kept
    return len(pending)


def apply_model_output(
    exhibition: Exhibition,
    output: dict[str, Any],
    max_chars: int,
) -> Exhibition:
    """Single-shot fold, for callers holding one complete response.

    The live pipeline uses :func:`apply_frame` and :func:`apply_labels`
    separately; this composes them so a whole response still applies in one go.
    """
    exhibition = apply_frame(exhibition, output)
    if isinstance(output.get("items"), list):
        apply_labels(exhibition.items, output, max_chars)
    return exhibition
