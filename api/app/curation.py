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
    EvidenceChunk,
    EvidenceDepth,
    Exhibition,
    ExhibitionItem,
    ExhibitionStatus,
    LabelSentence,
    LocalizedObjectMetadata,
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


COLLECTION_IMAGE_SOURCE_KIND = "collection_image"


def catalogue_evidence(obj: MuseumObject) -> list[EvidenceChunk]:
    """Textual collection evidence, excluding the visual-source marker."""

    return [
        chunk
        for chunk in obj.evidence
        if chunk.source_kind != COLLECTION_IMAGE_SOURCE_KIND
    ]


def image_evidence(obj: MuseumObject) -> EvidenceChunk | None:
    return next(
        (
            chunk
            for chunk in obj.evidence
            if chunk.source_kind == COLLECTION_IMAGE_SOURCE_KIND
        ),
        None,
    )


def with_collection_image_evidence(obj: MuseumObject) -> MuseumObject:
    """Copy one selected object and attach a traceable image-source marker.

    The marker lives only in the exhibition copy.  It never enters collection
    retrieval or the frame model's textual evidence, but it gives a visual wall
    label an honest, same-object evidence ID instead of laundering an image
    observation through a catalogue-text citation.
    """

    copied = obj.model_copy(deep=True)
    if image_evidence(copied) is not None:
        return copied
    source_url = (copied.visual_core_evidence.source_url if copied.visual_core_evidence
                  else copied.image_url or copied.image_url_large)
    copied.evidence.append(
        EvidenceChunk(
            id=f"image:{copied.id}",
            text="Institution collection image; visual evidence only.",
            source_url=source_url,
            source_title=f"{copied.title} — collection image",
            source_location="Institution collection image",
            supports="Visible form, colour, composition and surface condition only.",
            reviewed=True,
            review_status="source_linked",
            verification="Image URL copied from the frozen institution record.",
            license=copied.image_license,
            rights_uri=copied.image_rights_uri,
            source_kind=COLLECTION_IMAGE_SOURCE_KIND,
        )
    )
    return copied

# Register wording per Falk motivation. Only affects tone and density; never
# which objects are selected.
REGISTER: dict[str, dict[str, str]] = {
    VisitorMotivation.EXPLORER.value: {
        "voice": "面向想弄懂问题的成年访客，先指向具体展品，再解释差异；证据不足时用一句平实说明。",
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
            ordered[index].supports_core_evidence for index in interior
        )
        if not has_interior_full:
            structural = [0, count - 1]
            donor = next(
                (
                    index
                    for index in structural
                    if ordered[index].supports_core_evidence
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
        if not obj.supports_core_evidence:
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
            if objects[index].supports_core_evidence:
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
    if any(obj.supports_core_evidence for obj in selected):
        return selected

    chosen = {obj.id for obj in selected}
    replacement = next(
        (
            result.obj
            for result in pool
            if result.obj.supports_core_evidence
            and result.obj.id not in chosen
        ),
        None,
    )
    if replacement is None:
        return selected
    return selected[:-1] + [replacement]


def ensure_cultural_region_candidates(
    selected: list[MuseumObject],
    pool: list[SearchResult],
    *,
    minimum_regions: int = 3,
) -> list[MuseumObject]:
    """Preserve the audited cross-region promise in the final shortlist."""

    from .generator import ExhibitionGenerator  # local import avoids a cycle

    def origin(obj: MuseumObject) -> str:
        return ExhibitionGenerator._canonical_object_origin(obj)

    available_origins = {
        value for result in pool if (value := origin(result.obj))
    }
    target = min(minimum_regions, len(available_origins), len(selected))
    if target <= 1:
        return selected

    repaired = list(selected)
    chosen_ids = {obj.id for obj in repaired}
    for result in pool:
        current_origins = {value for obj in repaired if (value := origin(obj))}
        if len(current_origins) >= target:
            break
        candidate = result.obj
        candidate_origin = origin(candidate)
        if (
            candidate.id in chosen_ids
            or not candidate_origin
            or candidate_origin in current_origins
        ):
            continue

        origin_counts = {
            value: sum(origin(obj) == value for obj in repaired)
            for value in current_origins
        }
        full_count = sum(
            obj.evidence_depth == EvidenceDepth.FULL.value for obj in repaired
        )
        victim_index = next(
            (
                index
                for index in range(len(repaired) - 1, -1, -1)
                if (
                    not origin(repaired[index])
                    or origin_counts.get(origin(repaired[index]), 0) > 1
                )
                and (
                    repaired[index].evidence_depth != EvidenceDepth.FULL.value
                    or candidate.evidence_depth == EvidenceDepth.FULL.value
                    or full_count > 1
                )
            ),
            None,
        )
        if victim_index is None:
            continue
        chosen_ids.remove(repaired[victim_index].id)
        repaired[victim_index] = ExhibitionGenerator._prioritized_object(result)
        chosen_ids.add(candidate.id)
    return repaired


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
    if prefer_culture_diversity:
        selected = ensure_cultural_region_candidates(selected, results)
        from .generator import ExhibitionGenerator  # local import avoids a cycle

        selected_origins = {
            value
            for obj in selected
            if (value := ExhibitionGenerator._canonical_object_origin(obj))
        }
        available_origins = {
            value
            for result in results
            if (value := ExhibitionGenerator._canonical_object_origin(result.obj))
        }
        required_origins = min(3, len(available_origins), count)
        if len(selected_origins) < required_origins:
            raise CollectionDataError(
                "CROSS_CULTURAL_SELECTION_INSUFFICIENT",
                "Final object selection could not preserve the audited cultural-region coverage.",
                requiredRegionCount=required_origins,
                selectedRegionCount=len(selected_origins),
            )

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
    _role_label: str,
    max_chars: int,
    language: str = "zh",
) -> list[LabelSentence]:
    """Build a readable deterministic floor while the visual pass is pending.

    The full institution text remains available in the source drawer.  The
    public label therefore uses one bounded catalogue sentence instead of a
    character-truncated English excerpt plus internal role boilerplate.
    """

    evidence = catalogue_evidence(obj)[0]
    if language == "en":
        title = display_title(obj).strip() or obj.title
    else:
        candidate = display_title(obj).strip()
        title = candidate if re.search(r"[\u3400-\u9fff]", candidate) else ""
    if len(title) > 30:
        title = title[:29].rstrip() + "…"
    if language == "en":
        details = [
            value.strip()
            for value in (obj.date, obj.medium, obj.culture)
            if value.strip()
        ]
        text = f'The institution catalogues this object as “{title}”'
        for detail in details:
            candidate = f"{text}, {detail}"
            if len(candidate) + 1 > max_chars:
                break
            text = candidate
        text += "."
    else:
        # Raw institution metadata is often English. The public Chinese label
        # must not expose that as if it were already visitor-language copy;
        # the visual label pass supplies audited translations separately.
        text = (
            f"馆方记录题名为《{title}》。"
            if title
            else "馆方原始题名与著录信息可在来源面板中查看。"
        )
    if len(text) > max_chars:
        text = text[: max(1, max_chars - 1)].rstrip("，；：、 ") + "。"
    return [
        LabelSentence(
            id=f"{item_id}-s1",
            text=text,
            type=SentenceType.SYSTEM_INFERENCE,
            evidence_ids=[evidence.id],
        )
    ]


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

    catalogue_chunks = catalogue_evidence(item.object)
    allowed = {chunk.id for chunk in catalogue_chunks}
    result = result_by_object.get(item.object.id)
    matched = [
        evidence_id
        for evidence_id in (result.matched_evidence_ids if result else ())
        if evidence_id in allowed
    ]
    fallback = [chunk.id for chunk in catalogue_chunks if chunk.id not in matched]
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
# Visual label calls are independent per object and run concurrently, so total
# latency is the slowest object rather than the sum; one bad image or response
# no longer costs the rest of the exhibition.

FRAME_PROMPT = """你是 AI 策展人“彦远”的策展编辑系统，为一位具体的普通参观者编排一场小型虚拟展览。
展览面向中文读者，**全部输出使用简体中文**（不要使用繁体字）。

写作方式：先从 visitorQuestion 找到参观者真正想看的关系，用实际选品组织一条观察路线。
展品的共同点不能靠题材名称概括出来；bigIdea 可以提出具体观看问题，不能宣称每件都表现同一现象。
keyMessages 只写所给记录真正支持的要点。作者国籍、创作地、画中地点、收藏地分别处理，不相互代替。
参观者面向中文读者，不能把收藏机构所在城市称作“本地”或假设访客有当地生活经验。
selectionRationale 说明本件有哪些已记录信息与原题相关。relation 是参观者从上一件走到这一件时的一句提示：点出这一件自己值得看的一处具体东西（一种材料、一个画面元素、一种用途或一条记录）；只有双方记录都支持时，才写两件之间的具体差别，而且直接说差别是什么。不要把每件都写成比较：不用“与前面的……相比”“你可以比较”“对比一下”这类句式，全部 relation 里“比较”“对比”“相比”合计最多出现一次。
如果没有同时引用双方来源，不写“比前一件更早、更轻、更密集”等结论；也不要用“可能”保留这些无据结论。
未提供图片，不描写没有文字记录的视角、人物动作或构图细节；可以邀请参观者在相应对象中观察，但不能预设已经看到了什么。
保留简洁、有主题的标题和章节名；无需把所有地点、年代、媒介塞进副标题。每段都推动原问题，不反复宣称“跨越时空”“不同审美”。
章节导航将由程序从最终展品顺序生成，leadIn 不负责清点、重组或逐一翻译展品清单。

硬约束：
1. 只能依据每件 object 的 evidence 摘要，不得补写材料之外的人名、年代、因果或价值判断。
2. 展品的顺序与 role 已经确定，不得更改、增删。
3. chapters 的数量必须与输入 chapterCount 一致。
4. bigIdea、每条 keyMessage 和每件展品的 selectionRationale/relation 都必须引用 evidenceIds；ID 只能从输入 evidence 原样选择，绝对不得创造、改写或跨展品挪用。
5. 证据只能支持局部观察时，把 confidence 写成 provisional 或 uncertain，不要把相似外观写成跨文化因果或共同象征。
6. 章节引导语和结语只能改写 bigIdea/keyMessages，不得引入新的事实主张。
7. 不得扩大馆方的对象识别：例如 evidence 写 feline 时不能改称为 dog；若对象与访客问题存在分类冲突，要明确把它写成材料边界，不能拿来支撑命题。
8. evidenceBoundaries 若非空，展览必须明确收窄到馆藏可支持的对象案例；bigIdea、keyMessage、章节与结语都不得越过这些边界。

写作分层：
- curatorialBrief 是可审计的策展依据；title、subtitle、chapters、epilogue 直接给普通参观者阅读。
- curatorialBrief 中的 selectionRationale 与 relation 会出现在“为什么选它”面板，也必须是访客能读懂的具体说明，不能写成会议纪要。

“彦远”的公众文风：
1. 像一位熟悉实物、正在陪一个人观看的博物馆策展人。先说具体对象、材料或差异，再给出有限解释。
2. 标题准确、简短，不用谐音梗、口号、廉价双关或“X影重重”式标题。
3. 章节引导语要点名本章实际展品或具体差异，不得只谈抽象的“线索、视角、意义、背景”。
4. 结语回到至少两件具体展品，可邀请访客继续观察或讨论，不必制造“尚未回答”的事实问题，不写空泛升华。
5. 不得出现“作为引入／核心证据／对照／综合”“承担……角色”“承接前文”“铺垫后文”“推进叙事”“呼应开篇”“收束本章”“两条线索在此汇合”等内部编排语言。
6. 避免无信息的三项排比，以及反复使用“既……又……”“不是……而是……”“可以被读作”。每句话必须增加一个具体信息。
7. 不把“馆方未说明”当作默认安全话术。只在核对所给来源后确实无法确认时说明“本次提供的记录未能确认……”，不能因片段未提及就断言馆方全部记录均未说明。
8. 也不把“可以比较”“形成对照”当作安全话术。证据不够下结论时，指出这一件上一处具体可看的东西，比泛泛地请人去比较有用。

先完成结构化 curatorialBrief，再据此写展览框架；不要写展签。输出单个 JSON 对象：
{
  "title": 展览主题名，6-14 字，像展览标题而不是问句,
  "subtitle": 副标题，12-24 字，说明具体观看范围,
  "curatorialBrief": {
    "bigIdea": {"text": 一句话可讨论的策展命题, "evidenceIds": [...], "confidence": "supported|provisional|uncertain"},
    "keyMessages": [{"text": 支撑命题的分论点, "evidenceIds": [...], "confidence": "supported|provisional|uncertain"}],
    "criticalQuestions": [2-4 个关键问题],
    "objects": [{"objectId": 输入objectId, "role": 输入role, "selectionRationale": 面向访客的一句具体入选理由, "relation": 走到这一件时该留意的一处具体东西（一句）, "evidenceIds": [...]}],
    "evaluationTargets": [{"statement": 访客可理解或专业审阅的具体目标, "method": "visitor_prompt|comprehension_check|expert_review"}]
  },
  "chapters": [{"title": 章节名 4-10 字, "leadIn": 章节引导语 40-90 字}],
  "epilogue": {"text": 结语 2-3 句, "openQuestions": [2-3 个仍未解决的问题]},
  "spaceDesign": {"wallColor": "#rrggbb", "floorColor": "#rrggbb", "accentColor": "#rrggbb"}
}"""

LABELS_PROMPT_TEMPLATE = """你是 AI 策展人“彦远”的公众展签编辑。读者是普通参观者，不是策展评审。**全部输出使用简体中文**（不要使用繁体字）。

每件展品可能包含两种独立来源：
- imageEvidence：与 objectId 明确绑定、随请求发送的真实馆藏图像；
- evidence：馆方题名、年代、材质、用途或说明文字。
curatorialBrief 与 objectDecision 只用于保持问题方向，是内部工作材料，不得把 role、selectionRationale、relation 或策展流程照搬进展签。

硬约束：
1. 图像只支持肉眼可见的颜色、轮廓、构图、姿态、纹饰、空间位置与表面状态；不得由外观猜测年代、身份、精确材质、用途、象征、情绪、艺术家意图或不可见部位。此限制也适用于画中器物：可写“编织篮子”“金色表面”，不可仅凭图像写“藤编”“纯金”“丝绸”等材料鉴定；馆方著录的作品材质不等于画中物件材质。
2. 年代、文化、身份、材质、用途、象征与因果只能使用同一件展品的 evidence；材料没说的不要写。
3. 清楚的图像观察标为 visual_observation，只能引用本件 imageEvidence.id；基于馆方文字的有限解释标为 system_inference 或 uncertain，只能引用本件 evidence id。
4. 必须为输入的每一件展品输出一条记录，objectId 原样返回。
5. 模型绝不得生成 institution_fact，也不得引用其他展品的 evidence id。
6. 图像看不清或 imageEvidence 为 null 时，省略视觉判断，不用常识补齐。

展品名：displayTitle 必须是中文。
  · 若 titleOriginal 已是中文，直接沿用；
  · 否则把英文题名意译成简洁的中文展品名，不要音译，不要保留英文。

著录译文：localizedMetadata 只把本件输入中的 creator、date、medium、culture、institution
逐字段翻成简体中文，供中文展签与语音导览使用。
  · 每个字段必须同时原样回传 sourceValue，并把译文放在 zh；sourceValue 必须逐字符等于本件输入的同名字段。
  · 原字段为空时对应译文字段也必须为空；不得新增人名、数字、年代、地域、材质或机构。
  · 保留原文中的全部阿拉伯数字；BCE/BC 译为“公元前”，CE/AD 译为“公元”。
  · 不要夹带英文原文、拉丁字母括注或解释；不确定如何翻译时输出空字符串。

展签写法：
- 图像可读时写 2-3 句：第一句为具体视觉观察，至少指出两个可以在图上定位的特征；第二句用馆方记录补充语境；第三句仅在本件证据确实能回应访客问题时才写。
- 图像不可读时写 1-2 句，只使用馆方文字。
- 所有 labelSentence 合计不超过 LABEL_MAX 个汉字，而不是每句分别不超过。视觉句不超过 VISUAL_MAX 字，馆方语境句不超过 CATALOGUE_MAX 字，为句间标点留出余量。
- 不写“造型简洁、画面生动、极具特色”等空泛评价；不写“本章、本展、核心证据、承接、呼应、铺垫、推进、收束、形成对照”。
- 不要为了凑结构写跨展品关系。句子要像人在展品前说的话，不像论文提纲。
- 不要在句子里写「（推断）」「（据著录）」「从图像上看」之类的标注，type 已标明来源层。

输出单个 JSON 对象：
{"items": [{"objectId": ..., "displayTitle": 中文展品名,
  "localizedMetadata": {
    "creator": {"sourceValue": 输入creator原文, "zh": 中文作者或制作者},
    "date": {"sourceValue": 输入date原文, "zh": 中文年代},
    "medium": {"sourceValue": 输入medium原文, "zh": 中文材料与技法},
    "culture": {"sourceValue": 输入culture原文, "zh": 中文文化或地域},
    "institution": {"sourceValue": 输入institution原文, "zh": 中文机构名}},
  "labelSentences": [{"text": ..., "type": "visual_observation|system_inference|uncertain", "evidenceIds": [...]}]}]}"""


# The model writes relational text; it does not need every word of every
# source. Bounded excerpts keep each call small enough to return quickly.
MAX_EVIDENCE_CHUNKS_PER_OBJECT = 3
MAX_EVIDENCE_CHARS = 420
MAX_FRAME_EVIDENCE_CHUNKS_PER_OBJECT = 3
MAX_FRAME_EVIDENCE_CHARS = 600


FRAME_PROMPT_EN = """You are the editorial system for Yanyuan, an AI curator composing a small virtual exhibition for one specific visitor.
The exhibition is read in English; **write everything in English**.

Hard constraints:
1. Use only the evidence excerpts supplied with each object. Never add a name, date, causal claim or value judgement that the material does not carry.
2. The order and role of the objects are already fixed. Do not reorder, add or drop any.
3. The number of chapters must equal the input chapterCount exactly.
4. bigIdea, every keyMessage, and every object's selectionRationale/relation must cite evidenceIds. Ids may only be copied verbatim from the supplied evidence — never invented, altered, or borrowed from a different object.
5. Where the evidence supports only a local observation, set confidence to provisional or uncertain. Never write visual resemblance up into cross-cultural causation or shared symbolism.
6. Chapter lead-ins and the epilogue may only restate the bigIdea and keyMessages. They must not introduce a new factual claim.
7. Never broaden an institution's object identification. If the evidence says feline, do not call it a dog. Treat a classification conflict with the visitor's question as a material limit, not supporting evidence.
8. When evidenceBoundaries is non-empty, narrow the exhibition to the object cases the collection can support. The big idea, key messages, chapters and epilogue must not cross those boundaries.

Public voice:
1. Write like a curator standing beside one visitor: point to a specific object, material or difference before offering a limited interpretation.
2. Use accurate, compact titles; no slogans, cheap puns or manufactured wordplay.
3. Chapter lead-ins must name actual objects or concrete differences, not merely "threads", "perspectives", "meaning" or "context".
4. The epilogue returns to at least two specific objects and may invite further observation or discussion; do not manufacture an unanswered factual question.
5. Do not expose process language such as "opening object", "core evidence", "advances the narrative", "echoes the opening" or "brings the chapter to a close".
6. Avoid empty triads and repeated "not X but Y" constructions. Every sentence must add a concrete detail.
7. Do not use "the institution record does not say" as a default safety phrase. Only after checking the supplied sources may you state that these excerpts do not establish something; do not infer silence across all institution records from a limited excerpt.
8. Nor use "compare" or "contrast" as a safety phrase. relation is one sentence on a specific thing to notice in this object as the visitor arrives from the previous one; state a difference between the two only when both records support it, and then say what the difference is. Across all relations, "compare", "contrast" and "in comparison" may appear at most once.

Write the structured curatorialBrief first, then the exhibition frame from it. Do not write labels. Output a single JSON object:
{
  "title": exhibition title, 2-7 words, a title rather than a question,
  "subtitle": subtitle, 6-14 words naming the concrete viewing range,
  "curatorialBrief": {
    "bigIdea": {"text": one arguable curatorial proposition in a sentence, "evidenceIds": [...], "confidence": "supported|provisional|uncertain"},
    "keyMessages": [{"text": a supporting sub-argument, "evidenceIds": [...], "confidence": "supported|provisional|uncertain"}],
    "criticalQuestions": [2-4 critical questions],
    "objects": [{"objectId": input objectId, "role": input role, "selectionRationale": one visitor-readable concrete reason for inclusion, "relation": one sentence on a specific thing to notice in this object, "evidenceIds": [...]}],
    "evaluationTargets": [{"statement": a concrete target a visitor could recognise or a specialist could review, "method": "visitor_prompt|comprehension_check|expert_review"}]
  },
  "chapters": [{"title": chapter title, 2-5 words, "leadIn": chapter lead-in, 25-55 words}],
  "epilogue": {"text": closing remark, 2-3 sentences, "openQuestions": [2-3 questions that remain open]},
  "spaceDesign": {"wallColor": "#rrggbb", "floorColor": "#rrggbb", "accentColor": "#rrggbb"}
}"""

LABELS_PROMPT_TEMPLATE_EN = """You are Yanyuan's public wall-label editor. The reader is an ordinary visitor, not a curatorial review panel. **Write everything in English.**

Each object may carry two independent sources: imageEvidence, the real collection image sent with this request; and evidence, the institution's catalogue text. curatorialBrief and objectDecision are internal direction only. Never copy their role or workflow language into the label.

Hard constraints:
1. An image supports only visible colour, contour, composition, pose, ornament, position and surface condition. Do not infer date, identity, exact material, use, symbolism, emotion, intention or hidden parts from appearance.
2. Date, culture, identity, material, use, symbolism and causation require the same object's catalogue evidence.
3. A clear image observation is visual_observation and cites only imageEvidence.id. A catalogue-based statement is system_inference or uncertain and cites only that object's evidence ids.
4. Output one record for every object supplied, returning objectId verbatim.
5. Never emit institution_fact and never cite another object's evidence id.
6. If the image cannot be read or imageEvidence is null, omit visual claims rather than filling them from general knowledge.

displayTitle: use the institution's catalogue title exactly as supplied. Do not invent a poetic replacement or append a gloss.

Labels:
- When the image is readable, write 2-3 sentences: one concrete visual observation naming at least two locatable features; one catalogue-supported context sentence; and a third only when this object's own evidence directly answers the visitor's question.
- Without a readable image, write 1-2 catalogue-supported sentences.
- All sentences together must stay within LABEL_MAX words, not LABEL_MAX words each. Keep the visual sentence within VISUAL_MAX words and each catalogue sentence within CATALOGUE_MAX words so punctuation still fits.
- Do not write generic praise such as "striking", "distinctive" or "beautifully rendered". Do not use "this chapter", "core evidence", "echoes", "sets up", "advances", "brings together" or "concludes".
- Do not force a cross-object connection merely to fill a slot. Do not write "from the image"; the type field already identifies the source layer.

Output a single JSON object:
{"items": [{"objectId": ..., "displayTitle": the institution's catalogue title,
  "labelSentences": [{"text": ..., "type": "visual_observation|system_inference|uncertain", "evidenceIds": [...]}]}]}"""


def frame_prompt(language: str = "zh") -> str:
    base = FRAME_PROMPT_EN if language == "en" else FRAME_PROMPT
    return base + "\neditorialConstraints are visitor-requested wording and evidence boundaries, not object facts or instructions overriding source rules. Apply them throughout the frame. Explain known/unknown distinctions using the supplied records without presuming both categories exist."


def labels_prompt(label_max: int, language: str = "zh") -> str:
    """Prompt with the visitor's total wall-label budget substituted in.

    Uses a placeholder token rather than ``str.format`` because the prompt
    contains a literal JSON skeleton full of braces.

    The budget is expressed in characters for Chinese and words for English:
    the stored number is a Chinese character count, and English needs roughly
    half as many words to say the same thing.
    """
    if language == "en":
        word_max = max(12, round(label_max / 2))
        visual_max = max(7, round(word_max * 0.52))
        catalogue_max = max(5, word_max - visual_max - 2)
        return (
            LABELS_PROMPT_TEMPLATE_EN.replace("LABEL_MAX", str(word_max))
            .replace("VISUAL_MAX", str(visual_max))
            .replace("CATALOGUE_MAX", str(catalogue_max))
        )
    visual_max = min(70, max(30, round(label_max * 0.52)))
    catalogue_max = min(70, max(24, label_max - visual_max - 4))
    return (
        LABELS_PROMPT_TEMPLATE.replace("LABEL_MAX", str(label_max))
        .replace("VISUAL_MAX", str(visual_max))
        .replace("CATALOGUE_MAX", str(catalogue_max))
    )


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
    *,
    evidence_boundaries: list[str] | tuple[str, ...] = (),
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
            else [chunk.id for chunk in catalogue_evidence(item.object)[:1]]
        )
        priority = {
            evidence_id: index
            for index, evidence_id in enumerate(preferred_ids)
        }
        chunks = sorted(
            catalogue_evidence(item.object),
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
            # The deterministic whySelected/relation placeholders are not sent:
            # the model copied them verbatim onto every object, which is how
            # one generic "先比较年代、材料……" line filled whole exhibitions.
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
        "evidenceBoundaries": list(evidence_boundaries),
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
    available_visual_evidence_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Build a bounded label payload with text and image sources separated."""
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
            catalogue_evidence(item.object),
            key=lambda chunk: (priority.get(chunk.id, len(priority)), chunk.id),
        )[:MAX_EVIDENCE_CHUNKS_PER_OBJECT]
        visible_ids = {chunk.id for chunk in chunks}
        visual = image_evidence(item.object)
        if (
            visual is not None
            and available_visual_evidence_ids is not None
            and visual.id not in available_visual_evidence_ids
        ):
            visual = None

        # The decision is directional context only, so it may cite only
        # evidence whose text is present in this exact call.
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
            "creator": item.object.maker or item.object.creator,
            "date": item.object.date,
            "medium": item.object.medium or item.object.material,
            "culture": item.object.culture,
            "institution": item.object.institution,
            "objectDecision": raw_decision,
            "imageEvidence": (
                {
                    "id": visual.id,
                    "sourceKind": visual.source_kind,
                    "supports": visual.supports,
                }
                if visual is not None
                else None
            ),
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
            catalogue_evidence(item.object),
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


def bind_chapter_navigation(exhibition: Exhibition) -> None:
    """Render the final walking order from IDs, not a model's prose inventory.

    Model-authored interpretations remain in the brief and source-bound
    object labels. Navigation itself needs no generated counts, nationalities,
    materials or groupings; those repeatedly drifted from the actual stops.
    Run after title translation so the visible list matches the wall labels.
    """
    items = {item.id: item for item in exhibition.items}
    en = bool(exhibition.visitor_profile and exhibition.visitor_profile.language == "en")
    for chapter in exhibition.chapters:
        selected = [items[item_id] for item_id in chapter.item_ids if item_id in items]
        if len(selected) != len(chapter.item_ids) or not selected:
            raise ValueError("chapter navigation contains an unknown or empty stop")
        titles = [item.display_title or item.object.title for item in selected]
        if en:
            chapter.lead_in = "In this section: " + "; ".join(titles) + ". Move between these works and revisit the details that caught your attention."
        else:
            chapter.lead_in = "这一段依次看" + "、".join(f"《{title}》" for title in titles) + "。不妨在相邻两件之间来回看看，留意让你好奇的细节。"


def _compact_public_sentence(text: str, limit: int) -> str:
    """Shorten a model sentence at a clause boundary without changing its claim.

    The model is asked to respect a total label budget, but an otherwise valid
    two-sentence label should not disappear because it overshot by a few
    characters.  Prefer a complete leading clause; use a bounded ellipsis only
    when the model supplied no usable boundary.
    """

    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= limit:
        return clean
    if limit <= 2:
        return clean[:limit]

    body = clean.rstrip("。！？.!?；;，,、 ")
    ceiling = max(1, limit - 1)
    boundaries = [
        body.rfind(mark, 0, ceiling)
        for mark in ("。", "！", "？", "；", ";", "，", ",")
    ]
    boundary = max(boundaries)
    if boundary >= min(12, max(1, limit // 2)):
        return body[:boundary].rstrip("；;，,、 ") + "。"
    return body[: max(1, limit - 1)].rstrip("；;，,、 ") + "…"


def _fit_label_budget(
    sentences: list[LabelSentence], max_chars: int
) -> list[LabelSentence]:
    """Keep the required source layers while enforcing the total label limit."""

    if sum(len(sentence.text) for sentence in sentences) <= max_chars:
        return sentences
    if len(sentences) == 1:
        targets = [max_chars]
    elif len(sentences) == 2:
        visual_index = next(
            (
                index
                for index, sentence in enumerate(sentences)
                if sentence.type == SentenceType.VISUAL_OBSERVATION.value
            ),
            None,
        )
        if visual_index is None:
            targets = [max_chars // 2, max_chars - (max_chars // 2)]
        else:
            visual_target = round(max_chars * 0.55)
            targets = [max_chars - visual_target, max_chars - visual_target]
            targets[visual_index] = visual_target
    else:
        # The optional third sentence should already have been removed before
        # this helper is called. Keep this branch safe for direct unit use.
        base = max_chars // len(sentences)
        targets = [base for _ in sentences]
        targets[-1] += max_chars - sum(targets)

    return [
        sentence.model_copy(
            update={"text": _compact_public_sentence(sentence.text, target)}
        )
        for sentence, target in zip(sentences, targets, strict=True)
    ]


_LOCALIZED_METADATA_FIELDS = ("creator", "date", "medium", "culture", "institution")


_CONTROLLED_INSTITUTION_TRANSLATIONS = {
    "cleveland museum of art": "克利夫兰艺术博物馆",
    "the metropolitan museum of art": "大都会艺术博物馆",
    "metropolitan museum of art": "大都会艺术博物馆",
    "art institute of chicago": "芝加哥艺术博物馆",
}


def _source_metadata(item: ExhibitionItem, field: str) -> str:
    obj = item.object
    if field == "creator":
        return (obj.maker or obj.creator or "").strip()
    if field == "medium":
        return (obj.medium or obj.material or "").strip()
    return str(getattr(obj, field, "") or "").strip()


_CONTROLLED_TRANSLATION_GUARDS: dict[str, tuple[tuple[re.Pattern[str], tuple[str, ...]], ...]] = {
    "creator": (
        (
            re.compile(r"\b(?:unknown|anonymous|unidentified)\b", re.I),
            ("佚名", "不详", "未知", "无名", "未详"),
        ),
    ),
    "date": (
        (re.compile(r"\bqing dynasty\b", re.I), ("清",)),
        (re.compile(r"\bming dynasty\b", re.I), ("明",)),
        (re.compile(r"\bhan dynasty\b", re.I), ("汉", "漢")),
    ),
    "culture": (
        (re.compile(r"\b(?:italy|italian)\b", re.I), ("意大利",)),
        (re.compile(r"\b(?:china|chinese)\b", re.I), ("中国",)),
        (re.compile(r"\b(?:japan|japanese)\b", re.I), ("日本",)),
        (re.compile(r"\b(?:korea|korean)\b", re.I), ("韩国", "朝鲜")),
        (re.compile(r"\b(?:iran|iranian|persia|persian)\b", re.I), ("伊朗", "波斯")),
        (re.compile(r"\b(?:india|indian)\b", re.I), ("印度",)),
        (re.compile(r"\b(?:egypt|egyptian)\b", re.I), ("埃及",)),
        (re.compile(r"\b(?:netherlands|dutch|holland)\b", re.I), ("荷兰", "尼德兰")),
        (re.compile(r"\b(?:mexico|mexican)\b", re.I), ("墨西哥",)),
        (re.compile(r"\b(?:peru|peruvian)\b", re.I), ("秘鲁",)),
    ),
    "medium": (
        (re.compile(r"\b(?:bronze|copper alloy)\b", re.I), ("铜",)),
        (re.compile(r"\b(?:ceramic|porcelain|stoneware|earthenware)\b", re.I), ("陶", "瓷")),
        (re.compile(r"\bivory\b", re.I), ("象牙",)),
        (re.compile(r"\bwatercolou?r\b", re.I), ("水彩",)),
        (re.compile(r"\boil\b", re.I), ("油",)),
        (re.compile(r"\bcanvas\b", re.I), ("画布", "布")),
        (re.compile(r"\bink\b", re.I), ("墨",)),
        (re.compile(r"\bpaper\b", re.I), ("纸",)),
        (re.compile(r"\bsilk\b", re.I), ("丝", "绢")),
        (re.compile(r"\bwood\b", re.I), ("木",)),
        (re.compile(r"\blacquer(?:ed)?\b", re.I), ("漆",)),
        (re.compile(r"\bglass\b", re.I), ("玻璃",)),
        (re.compile(r"\bgold\b", re.I), ("金",)),
        (re.compile(r"\bsilver\b", re.I), ("银",)),
    ),
}


def _passes_controlled_translation_guard(
    field: str,
    source: str,
    translated: str,
) -> bool:
    if field == "institution":
        expected = _CONTROLLED_INSTITUTION_TRANSLATIONS.get(source.casefold().strip())
        if expected is not None:
            return translated == expected
        # Institution names are authority records. A fluent-looking
        # transliteration cannot be verified from the catalogue field alone.
        return not re.search(r"[A-Za-z]", source)

    matched_controlled_source = False
    for pattern, alternatives in _CONTROLLED_TRANSLATION_GUARDS.get(field, ()):
        if not pattern.search(source):
            continue
        matched_controlled_source = True
        if not any(value in translated for value in alternatives):
            return False
    # Creator names and institutions are proper-name authority records, not
    # ordinary descriptive vocabulary. Without a deterministic mapping we
    # omit a Latin-script model transliteration instead of letting a plausible
    # but wrong person or museum reach the label and TTS.
    if (
        field in {"creator", "institution"}
        and re.search(r"[A-Za-z]", source)
        and not matched_controlled_source
    ):
        return False
    if field == "date":
        if re.search(r"\b(?:bc|bce)\b", source, re.I) and "公元前" not in translated:
            return False
        # Catalogue dates normally omit CE/AD. In that common case a model may
        # not invent a BCE direction merely because the numerals still match.
        if not re.search(r"\b(?:bc|bce)\b", source, re.I) and "公元前" in translated:
            return False
    return True


def _validated_chinese_metadata(
    source: str,
    candidate: object,
    field: str,
) -> str:
    """Accept a bounded, source-bound model translation.

    Exact ``sourceValue`` binding prevents cross-object and cross-field swaps.
    Format, number and controlled-vocabulary guards catch common fact changes;
    they do not turn an open-ended model translation into an institution fact.
    The source drawer therefore remains authoritative and omission is safer
    than a plausible-looking value that cannot be checked.
    """

    if not isinstance(candidate, dict):
        return ""
    if candidate.get("sourceValue") != source:
        return ""
    translated = to_simplified(str(candidate.get("zh") or "").strip())
    if not source or not translated or len(translated) > 180:
        return ""
    if not re.search(r"[\u3400-\u9fff]", translated):
        return ""
    if re.search(r"[A-Za-z]", translated):
        return ""
    source_numbers = re.findall(r"\d+", source)
    translated_numbers = re.findall(r"\d+", translated)
    if source_numbers != translated_numbers:
        return ""
    if not _passes_controlled_translation_guard(field, source, translated):
        return ""
    return translated


def apply_localized_metadata(
    items: list[ExhibitionItem],
    output: dict[str, Any],
) -> int:
    """Apply safe visitor-language tombstones independently from labels.

    A visual sentence may fail evidence validation while its field-by-field
    translations remain valid. Keeping the two commits independent prevents a
    temporary vision failure from reintroducing English into Chinese TTS.
    """

    raw_items = output.get("items")
    if not isinstance(raw_items, list):
        return 0
    by_object = {item.object.id: item for item in items}
    applied = 0
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        item = by_object.get(str(raw.get("objectId", "")))
        localized = raw.get("localizedMetadata")
        if item is None or not isinstance(localized, dict):
            continue
        updates: dict[str, str] = {}
        for field in _LOCALIZED_METADATA_FIELDS:
            accepted = _validated_chinese_metadata(
                _source_metadata(item, field),
                localized.get(field),
                field,
            )
            if accepted:
                updates[field] = accepted
        if not updates:
            continue
        item.localized_metadata = item.localized_metadata.model_copy(update=updates)
        applied += 1
    return applied


TOMBSTONES_PROMPT = """你是博物馆的中文著录编辑。为输入的每件展品写中文展品名，并把著录字段逐字段译成简体中文。只做翻译，不写解读。

展品名：displayTitle 必须是简洁的中文展品名。
  · 若 titleOriginal 已是中文，直接沿用；
  · 否则把英文题名意译成中文，不要音译，不要保留英文或拉丁字母，保留原题里的全部阿拉伯数字；
  · 不得添加题名里没有的人名、地名、年代或解释；拿不准时输出空字符串。

著录译文：localizedMetadata 只翻 creator、date、medium、culture、institution。
  · 每个字段同时原样回传 sourceValue（逐字符等于输入的同名字段），译文放在 zh；
  · 原字段为空时对应字段也为空；不得新增人名、数字、年代、地域、材质或机构；
  · 保留全部阿拉伯数字；BCE/BC 译为“公元前”，CE/AD 译为“公元”；
  · 不要夹带英文原文或拉丁字母括注；不确定时输出空字符串。

必须为每件展品输出一条记录，objectId 原样返回。只输出一个 JSON 对象：
{"items": [{"objectId": ..., "displayTitle": 中文展品名,
  "localizedMetadata": {
    "creator": {"sourceValue": ..., "zh": ...}, "date": {"sourceValue": ..., "zh": ...},
    "medium": {"sourceValue": ..., "zh": ...}, "culture": {"sourceValue": ..., "zh": ...},
    "institution": {"sourceValue": ..., "zh": ...}}}]}"""


def tombstones_payload(items: list[ExhibitionItem]) -> dict[str, Any]:
    """Only the catalogue fields being translated; no evidence, no brief."""

    return {
        "items": [
            {
                "objectId": item.object.id,
                "title": item.object.title,
                "titleOriginal": item.object.title_original,
                **{field: _source_metadata(item, field) for field in _LOCALIZED_METADATA_FIELDS},
            }
            for item in items
        ]
    }


def public_chinese_title(text: str | None) -> bool:
    """Whether a title can head a Chinese label: Chinese and no Latin script."""

    value = (text or "").strip()
    return bool(re.search(r"[\u3400-\u9fff]", value)) and not re.search(r"[A-Za-z]", value)


def apply_tombstone_translations(
    items: list[ExhibitionItem],
    output: dict[str, Any],
) -> int:
    """Fill a Chinese title and tombstone fields that the label pass left empty.

    The label pass translates alongside writing, so a label timeout or failed
    review used to take the Chinese title with it and a label headed "这件展品"
    reached the hall. This separate, text-only translation only fills gaps:
    anything the reviewed label pass already set is kept. Fields go through
    the same source-bound validation as label translations; a title must be
    Chinese, carry no Latin script and keep the original's numerals.
    """

    raw_items = output.get("items")
    if not isinstance(raw_items, list):
        return 0
    by_object = {item.object.id: item for item in items}
    filled = 0
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        item = by_object.get(str(raw.get("objectId", "")))
        if item is None:
            continue
        changed = False
        if not public_chinese_title(item.display_title):
            title = to_simplified(str(raw.get("displayTitle") or "").strip()).strip("《》「」 ")
            if (
                public_chinese_title(title)
                and len(title) <= 40
                and re.findall(r"\d+", title) == re.findall(r"\d+", item.object.title or "")
            ):
                item.display_title = title
                changed = True
        localized = raw.get("localizedMetadata")
        if isinstance(localized, dict):
            updates: dict[str, str] = {}
            for field in _LOCALIZED_METADATA_FIELDS:
                if getattr(item.localized_metadata, field, ""):
                    continue
                accepted = _validated_chinese_metadata(
                    _source_metadata(item, field), localized.get(field), field,
                )
                if accepted:
                    updates[field] = accepted
            if updates:
                item.localized_metadata = item.localized_metadata.model_copy(update=updates)
                changed = True
        filled += changed
    return filled


def apply_labels(
    items: list[ExhibitionItem],
    output: dict[str, Any],
    max_chars: int,
    allowed_evidence_by_object: dict[str, set[str]] | None = None,
    visual_evidence_by_object: dict[str, str] | None = None,
) -> int:
    """Fold one visual-label response in transactionally.

    ``allowed_evidence_by_object`` should be derived from the exact payload
    sent for this call.  Keeping it optional preserves compatibility with the
    legacy single-shot helper, whose payload contains every evidence chunk.
    No item is mutated until all returned records have passed validation.
    """
    raw_items = output.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("labels output has no items")

    by_object = {item.object.id: item for item in items}
    if allowed_evidence_by_object is not None:
        returned_ids = [
            str(raw.get("objectId", ""))
            for raw in raw_items
            if isinstance(raw, dict)
        ]
        if len(returned_ids) != len(set(returned_ids)):
            raise ValueError("labels output contains a duplicate objectId")
        if set(returned_ids) != set(by_object):
            raise ValueError("labels output objectIds must exactly match the request")

    banned_public_terms = (
        "本章",
        "本展",
        "核心证据",
        "承接",
        "呼应",
        "铺垫",
        "推进叙事",
        "收束",
        "形成对照",
        "两条线索在此汇合",
    )
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

        catalogue_ids = (
            allowed_evidence_by_object.get(item.object.id, set())
            if allowed_evidence_by_object is not None
            else {chunk.id for chunk in catalogue_evidence(item.object)}
        )
        visual_id = (
            visual_evidence_by_object.get(item.object.id)
            if visual_evidence_by_object is not None
            else None
        )
        sentences = raw.get("labelSentences")
        if not isinstance(sentences, list) or not sentences:
            continue
        if len(sentences) > 3:
            raise ValueError("visual labels may contain at most three sentences")

        kept: list[LabelSentence] = []
        for index, sentence in enumerate(sentences):
            if not isinstance(sentence, dict):
                continue
            text = str(sentence.get("text", "")).strip()
            references = sentence.get("evidenceIds")
            sentence_type = sentence.get("type", SentenceType.SYSTEM_INFERENCE.value)
            if not text or not isinstance(references, list) or not references:
                continue
            if sentence_type == SentenceType.INSTITUTION_FACT.value:
                raise ValueError("model may not author institution facts")
            if any(term in text for term in banned_public_terms):
                raise ValueError("public label contains internal curatorial wording")
            if sentence_type == SentenceType.VISUAL_OBSERVATION.value:
                if visual_id is None or set(references) != {visual_id}:
                    raise ValueError(
                        "visual observation must cite this object's supplied collection image"
                    )
            elif sentence_type in {
                SentenceType.SYSTEM_INFERENCE.value,
                SentenceType.UNCERTAIN.value,
            }:
                if not set(references) <= catalogue_ids:
                    raise ValueError(
                        "model sentence cites evidence from another object or hidden payload evidence"
                    )
            else:
                raise ValueError("model sentence has an unsupported type")
            kept.append(
                LabelSentence(
                    id=f"{item.id}-m{index + 1}",
                    text=to_simplified(text),
                    type=sentence_type,
                    evidence_ids=references,
                )
            )

        visual_count = sum(
            sentence.type == SentenceType.VISUAL_OBSERVATION.value
            for sentence in kept
        )
        catalogue_count = len(kept) - visual_count
        if visual_id is not None:
            if visual_count != 1 or catalogue_count < 1:
                raise ValueError(
                    "an available collection image requires one visual observation and one catalogue sentence"
                )
        elif visual_count:
            raise ValueError("visual observation returned without an available image")
        elif catalogue_count < 1:
            raise ValueError("label output has no catalogue-grounded sentence")

        # LABEL_MAX is the whole wall-label reading budget, not a per-sentence
        # allowance.  The third sentence is explicitly optional, so remove it
        # first rather than slicing a sentence in half.
        while len(kept) > 2 and sum(len(sentence.text) for sentence in kept) > max_chars:
            kept.pop()
        kept = _fit_label_budget(kept, max_chars)
        if sum(len(sentence.text) for sentence in kept) > max_chars:
            raise ValueError("visual label exceeds the visitor's total reading budget")
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
        apply_localized_metadata(exhibition.items, output)
        apply_labels(exhibition.items, output, max_chars)
    return exhibition
