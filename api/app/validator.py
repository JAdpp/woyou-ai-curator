from __future__ import annotations

from collections import Counter
import re
import unicodedata

from .models import (
    CuratorialRole,
    EvidenceDepth,
    Exhibition,
    SentenceType,
    ValidationCheck,
    ValidationIssue,
    ValidationResult,
)


REQUIRED_ROLES = {role.value for role in CuratorialRole}

# Exhibition size follows the visitor's chosen duration; see DURATION_PLAN.
MIN_ITEMS = 5
MAX_ITEMS = 12

MIN_EVIDENCE_CHUNKS_FULL = 3
MIN_EVIDENCE_CHUNKS_THIN = 2
COLLECTION_IMAGE_SOURCE_KIND = "collection_image"

# Public copy occasionally invents a four-object exhibition after the selected
# list has already been fixed at five.  Only explicit ``number + 件`` phrases
# are treated as counts; other numerals (dates, chapter numbers, "three ways")
# remain ordinary prose.
_ITEM_COUNT_MENTION = re.compile(
    r"(?P<count>[0-9]{1,3}|[零〇一二两兩三四五六七八九十百]{1,6})\s*件(?!件|事)"
)
_NON_TOTAL_COUNT_PREFIX = re.compile(
    r"(?:第|每|其中|另有|至少|至多|最多|超过|多于|少于|不足|约|近|逾)\s*$"
)
_TOTAL_COUNT_CONTEXT = re.compile(
    r"(?:共|总计|合计|本章|这一章|全章|本展|展览|陈列|展出|选取|选择|汇集|呈现|组成|构成|串联|并置)"
)

# Four highly repetitive objects are enough to collapse a five-object argument
# into a near-duplicate grid.  Validation blocks release rather than swapping
# objects here, so it cannot silently discard a required cultural leg or an
# evidence-qualified core object.
MAX_REPEATED_TITLE_OR_SERIES = 3
_EXPLICIT_SERIES_PATTERNS = (
    re.compile(
        r"\b(?:from|part\s+of)\s+(?:the\s+)?(?:series|set)\s*[:\-–—]?\s*"
        r"[\"'“‘]?(?P<name>[^\"'”’\[\](),;]{2,100})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:series|set)\s*[:：]\s*(?P<name>[^\[\](),;]{2,100})",
        re.IGNORECASE,
    ),
    re.compile(r"[《“](?P<name>[^》”]{2,80})[》”]\s*(?:系列|组|套)"),
)
_EXPLICIT_PART_PATTERNS = (
    re.compile(
        r"^(?P<base>.+?)[\s,;:\-–—]+"
        r"(?:no\.?|number|plate|panel|part|sheet|leaf)\s*"
        r"(?:[0-9]+|[ivxlcdm]+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<base>.+?)第[0-9零〇一二两兩三四五六七八九十百]+"
        r"(?:幅|件|张|页|卷|号)$"
    ),
)


def _parse_chinese_count(value: str) -> int | None:
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "兩": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if not value or any(character not in digits and character not in "十百" for character in value):
        return None
    if not any(character in "十百" for character in value):
        try:
            return int("".join(str(digits[character]) for character in value))
        except ValueError:
            return None

    total = 0
    pending = 0
    for character in value:
        if character in digits:
            pending = digits[character]
            continue
        unit = 10 if character == "十" else 100
        total += (pending or 1) * unit
        pending = 0
    return total + pending


def _explicit_item_counts(text: str, *, require_total_context: bool = False) -> list[int]:
    counts: list[int] = []
    for match in _ITEM_COUNT_MENTION.finditer(text or ""):
        prefix = text[: match.start()]
        if _NON_TOTAL_COUNT_PREFIX.search(prefix):
            continue
        suffix = text[match.end() :]
        if require_total_context and not (
            not prefix.strip()
            or _TOTAL_COUNT_CONTEXT.search(prefix[-24:])
            or _TOTAL_COUNT_CONTEXT.search(suffix[:24])
        ):
            continue
        raw = match.group("count")
        count = int(raw) if raw.isascii() and raw.isdigit() else _parse_chinese_count(raw)
        if count is not None:
            counts.append(count)
    return counts


def _validate_public_copy_item_counts(
    exhibition: Exhibition,
) -> tuple[bool, list[ValidationIssue]]:
    actual = len(exhibition.items)
    surfaces: list[tuple[str, str, set[int], bool]] = [
        ("title", exhibition.title, {actual}, False),
        ("subtitle", exhibition.subtitle, {actual}, False),
    ]
    for chapter in exhibition.chapters:
        allowed = {actual, len(chapter.item_ids)}
        surfaces.extend(
            (
                (f"chapter {chapter.order + 1} title", chapter.title, allowed, False),
                (
                    f"chapter {chapter.order + 1} lead-in",
                    chapter.lead_in,
                    allowed,
                    True,
                ),
            )
        )

    errors: list[ValidationIssue] = []
    for surface, text, allowed, require_total_context in surfaces:
        for mentioned in _explicit_item_counts(
            text, require_total_context=require_total_context
        ):
            if mentioned in allowed:
                continue
            expected = " or ".join(str(value) for value in sorted(allowed))
            errors.append(
                ValidationIssue(
                    code="PUBLIC_COPY_ITEM_COUNT_MISMATCH",
                    message=(
                        f"The {surface} says {mentioned} items, but its actual "
                        f"item count is {expected}."
                    ),
                )
            )
    return not errors, errors


def _normalized_title_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold().strip()
    normalized = re.sub(r"^(?:the|a|an)\s+", "", normalized)
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", normalized)


def _explicit_series_keys(title: str, creator: str) -> set[str]:
    keys: set[str] = set()
    for pattern in _EXPLICIT_SERIES_PATTERNS:
        if match := pattern.search(title or ""):
            key = _normalized_title_key(match.group("name"))
            if key:
                keys.add(f"named:{key}")
    creator_key = _normalized_title_key(creator)
    if creator_key:
        for pattern in _EXPLICIT_PART_PATTERNS:
            if match := pattern.match((title or "").strip()):
                base = _normalized_title_key(match.group("base"))
                if base:
                    keys.add(f"parts:{creator_key}:{base}")
    return keys


def _validate_selection_variety(
    exhibition: Exhibition,
) -> tuple[bool, list[ValidationIssue]]:
    title_groups: dict[str, set[str]] = {}
    series_groups: dict[str, set[str]] = {}
    for item in exhibition.items:
        titles = (item.object.title, item.object.title_original or "")
        for title in titles:
            title_key = _normalized_title_key(title)
            if title_key:
                title_groups.setdefault(title_key, set()).add(item.id)
            for series_key in _explicit_series_keys(
                title, item.object.creator or item.object.maker
            ):
                series_groups.setdefault(series_key, set()).add(item.id)

    errors: list[ValidationIssue] = []
    repeated_titles = [
        item_ids
        for item_ids in title_groups.values()
        if len(item_ids) > MAX_REPEATED_TITLE_OR_SERIES
    ]
    if repeated_titles:
        largest = max(len(item_ids) for item_ids in repeated_titles)
        errors.append(
            ValidationIssue(
                code="OVERCONCENTRATED_NORMALIZED_TITLE",
                message=(
                    f"{largest} selected objects share the same normalized "
                    "institution title; the exhibition needs a less repetitive evidence chain."
                ),
            )
        )

    repeated_series = [
        item_ids
        for item_ids in series_groups.values()
        if len(item_ids) > MAX_REPEATED_TITLE_OR_SERIES
    ]
    if repeated_series:
        largest = max(len(item_ids) for item_ids in repeated_series)
        errors.append(
            ValidationIssue(
                code="OVERCONCENTRATED_OBJECT_SERIES",
                message=(
                    f"{largest} selected objects are explicitly catalogued as "
                    "parts of one series or set; the exhibition needs a less "
                    "repetitive evidence chain."
                ),
            )
        )
    return not errors, errors


def _normalized_source_text(value: str) -> str:
    value = re.sub(
        r"^(?:馆方原始记录（保留原文）|馆方原始记录|机构原始记录|institution record)\s*[：:]\s*",
        "",
        value.strip(),
        flags=re.IGNORECASE,
    )
    value = value.rstrip("….")
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", value.casefold())


def _institution_fact_is_extractive(sentence_text: str, evidence_texts: list[str]) -> bool:
    sentence = _normalized_source_text(sentence_text)
    if not sentence:
        return False
    for evidence_text in evidence_texts:
        evidence = _normalized_source_text(evidence_text)
        # Short catalogue facts such as "Dogs" or "Jade" are still exact
        # institution text.  The length floor only protects fuzzy substring
        # matching; it must not invalidate a byte-for-byte normalized quote.
        if sentence == evidence:
            return True
        if len(sentence) >= 8 and (sentence in evidence or evidence in sentence):
            return True
    return False


def _validate_curatorial_brief(
    exhibition: Exhibition,
) -> tuple[bool, list[ValidationIssue], list[ValidationIssue]]:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    brief = exhibition.curatorial_brief
    if brief is None:
        if exhibition.versions.prompt.startswith(
            ("v3-curatorial-brief", "v4-vision-public-copy")
        ):
            errors.append(
                ValidationIssue(
                    code="CURATORIAL_BRIEF_REQUIRED",
                    message="This generation version requires a CuratorialBrief before release.",
                )
            )
            return False, errors, warnings
        warnings.append(
            ValidationIssue(
                code="CURATORIAL_BRIEF_LEGACY_MISSING",
                message="This legacy exhibition predates the CuratorialBrief contract.",
            )
        )
        return True, errors, warnings

    if brief.visitor_inquiry.strip() != exhibition.question.strip():
        errors.append(
            ValidationIssue(
                code="CURATORIAL_BRIEF_INQUIRY_MISMATCH",
                message="The CuratorialBrief visitor inquiry must match the exhibition question.",
            )
        )

    item_by_id = {item.id: item for item in exhibition.items}
    item_by_object = {item.object.id: item for item in exhibition.items}
    decision_item_ids = [decision.item_id for decision in brief.objects]
    decision_object_ids = [decision.object_id for decision in brief.objects]
    decisions_complete = (
        len(decision_item_ids) == len(set(decision_item_ids))
        and len(decision_object_ids) == len(set(decision_object_ids))
        and set(decision_item_ids) == set(item_by_id)
        and set(decision_object_ids) == set(item_by_object)
    )
    if not decisions_complete:
        errors.append(
            ValidationIssue(
                code="CURATORIAL_BRIEF_OBJECT_MISMATCH",
                message="CuratorialBrief object decisions must cover every selected item exactly once.",
            )
        )

    all_evidence_ids = {
        evidence.id for item in exhibition.items for evidence in item.object.evidence
    }
    claims = [brief.big_idea, *brief.key_messages]
    if len({claim.id for claim in claims}) != len(claims):
        errors.append(
            ValidationIssue(
                code="CURATORIAL_BRIEF_CLAIM_ID_DUPLICATE",
                message="CuratorialBrief claim IDs must be unique.",
            )
        )
    for claim in claims:
        unknown = sorted(set(claim.evidence_ids) - all_evidence_ids)
        if unknown:
            errors.append(
                ValidationIssue(
                    code="CURATORIAL_BRIEF_CLAIM_EVIDENCE_NOT_FOUND",
                    message=(
                        f"Curatorial claim '{claim.id}' references evidence outside the selected objects: "
                        + ", ".join(unknown)
                    ),
                )
            )

    for decision in brief.objects:
        item = item_by_id.get(decision.item_id)
        if item is None or item.object.id != decision.object_id:
            continue
        if str(decision.role) != str(item.role):
            errors.append(
                ValidationIssue(
                    code="CURATORIAL_BRIEF_ROLE_MISMATCH",
                    message="A CuratorialBrief object role differs from the fixed exhibition role.",
                    item_id=item.id,
                )
            )
        valid_ids = {evidence.id for evidence in item.object.evidence}
        unknown = sorted(set(decision.evidence_ids) - valid_ids)
        if unknown:
            errors.append(
                ValidationIssue(
                    code="CURATORIAL_BRIEF_OBJECT_EVIDENCE_NOT_FOUND",
                    message=(
                        "A CuratorialBrief object decision cites evidence from another object: "
                        + ", ".join(unknown)
                    ),
                    item_id=item.id,
                )
            )
        if decision.selection_rationale != item.why_selected or decision.relation != item.relation:
            errors.append(
                ValidationIssue(
                    code="CURATORIAL_BRIEF_DECISION_STALE",
                    message="The displayed object rationale or relation has diverged from CuratorialBrief.",
                    item_id=item.id,
                )
            )

    expected_core_answer = "".join(
        message.text
        if message.text.endswith(("。", "！", "？"))
        else message.text + "。"
        for message in brief.key_messages
    )
    if exhibition.curatorial_thesis != brief.big_idea.text:
        errors.append(
            ValidationIssue(
                code="CURATORIAL_BRIEF_BIG_IDEA_STALE",
                message="The public curatorial thesis must be generated from CuratorialBrief.bigIdea.",
            )
        )
    if exhibition.core_answer != expected_core_answer:
        errors.append(
            ValidationIssue(
                code="CURATORIAL_BRIEF_KEY_MESSAGES_STALE",
                message="The public core answer must be generated from CuratorialBrief.keyMessages.",
            )
        )
    if exhibition.sub_questions != brief.critical_questions:
        errors.append(
            ValidationIssue(
                code="CURATORIAL_BRIEF_QUESTIONS_STALE",
                message="Public sub-questions must match CuratorialBrief.criticalQuestions.",
            )
        )

    selected_ids = set(item_by_object)
    if any(candidate.object_id in selected_ids for candidate in brief.excluded_candidates):
        errors.append(
            ValidationIssue(
                code="CURATORIAL_BRIEF_EXCLUDED_SELECTED",
                message="An excluded candidate cannot also be a selected object.",
            )
        )
    if brief.retrieval.selected_count != len(exhibition.items):
        errors.append(
            ValidationIssue(
                code="CURATORIAL_BRIEF_RETRIEVAL_COUNT_MISMATCH",
                message="CuratorialBrief retrieval selectedCount does not match the exhibition.",
            )
        )
    if brief.retrieval.candidate_count < brief.retrieval.selected_count:
        errors.append(
            ValidationIssue(
                code="CURATORIAL_BRIEF_RETRIEVAL_COUNT_INVALID",
                message="CuratorialBrief candidateCount cannot be smaller than selectedCount.",
            )
        )

    ethics = brief.ethics
    expected_required = {
        "not_assessed": None,
        "not_required": False,
        "required": True,
        "completed": True,
    }[str(ethics.community_review_status)]
    if ethics.community_review_required is not expected_required:
        errors.append(
            ValidationIssue(
                code="CURATORIAL_BRIEF_COMMUNITY_REVIEW_STATE_INVALID",
                message="Community-review requirement and review status are inconsistent.",
            )
        )
    if ethics.provenance_status in {"not_reviewed", "unknown"}:
        warnings.append(
            ValidationIssue(
                code="CURATORIAL_BRIEF_PROVENANCE_REVIEW_PENDING",
                message="Object provenance has not received an independent curatorial review.",
            )
        )
    if ethics.cultural_sensitivity_status in {"not_reviewed", "unknown"}:
        warnings.append(
            ValidationIssue(
                code="CURATORIAL_BRIEF_CULTURAL_SENSITIVITY_REVIEW_PENDING",
                message="Cultural-sensitivity risk has not been reviewed.",
            )
        )
    if ethics.community_review_status == "not_assessed":
        warnings.append(
            ValidationIssue(
                code="CURATORIAL_BRIEF_COMMUNITY_REVIEW_NOT_ASSESSED",
                message="The need for source-community review has not been assessed.",
            )
        )
    if brief.audience.duration_minutes != (
        exhibition.visitor_profile.duration_minutes
        if exhibition.visitor_profile
        else exhibition.agenda.duration_minutes
    ):
        errors.append(
            ValidationIssue(
                code="CURATORIAL_BRIEF_AUDIENCE_STALE",
                message="CuratorialBrief audience duration has diverged from the visitor profile.",
            )
        )

    if brief.status == "model_refined":
        warnings.append(
            ValidationIssue(
                code="CURATORIAL_BRIEF_ENTAILMENT_REVIEW_PENDING",
                message=(
                    "Claim evidence IDs were validated, but semantic entailment still requires expert review."
                ),
            )
        )
    return not errors, errors, warnings


def validate_exhibition(exhibition: Exhibition) -> ValidationResult:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    curatorial_brief_ok, brief_errors, brief_warnings = _validate_curatorial_brief(
        exhibition
    )
    errors.extend(brief_errors)
    warnings.extend(brief_warnings)

    # Exhibition size now follows the visitor's chosen duration (5 / 8 / 12),
    # so this is a range check rather than a fixed count.
    item_count = len(exhibition.items)
    item_count_ok = MIN_ITEMS <= item_count <= MAX_ITEMS
    if not item_count_ok:
        errors.append(
            ValidationIssue(
                code="ITEM_COUNT_OUT_OF_RANGE",
                message=(
                    f"An exhibition needs between {MIN_ITEMS} and {MAX_ITEMS} items; found {item_count}."
                ),
            )
        )

    public_copy_counts_ok, count_errors = _validate_public_copy_item_counts(
        exhibition
    )
    errors.extend(count_errors)

    # Roles repeat in a longer exhibition; what must hold is that every role is
    # represented at least once.
    role_values = [str(item.role) for item in exhibition.items]
    missing_roles = REQUIRED_ROLES - set(role_values)
    roles_ok = not missing_roles
    if not roles_ok:
        errors.append(
            ValidationIssue(
                code="CURATORIAL_ROLE_MISSING",
                message=f"Missing curatorial roles: {', '.join(sorted(missing_roles))}.",
            )
        )

    # Thin-evidence objects restate metadata rather than quoting institution
    # prose, so they may not be the thing an argument rests on.
    core_evidence_depth_ok = True
    for item in exhibition.items:
        if (
            str(item.role) == CuratorialRole.CORE_EVIDENCE.value
            and item.object.evidence_depth == EvidenceDepth.THIN.value
        ):
            core_evidence_depth_ok = False
            errors.append(
                ValidationIssue(
                    code="THIN_EVIDENCE_IN_CORE_ROLE",
                    message=(
                        f"'{item.object.title}' has only metadata-derived evidence and "
                        "cannot hold the core-evidence role."
                    ),
                    item_id=item.id,
                )
            )

    # Every item must appear in exactly one chapter, or the hall cannot be laid
    # out and some object becomes unreachable.
    chapters_ok = True
    if exhibition.chapters:
        chaptered = [item_id for chapter in exhibition.chapters for item_id in chapter.item_ids]
        item_ids = {item.id for item in exhibition.items}
        if len(chaptered) != len(set(chaptered)) or set(chaptered) != item_ids:
            chapters_ok = False
            errors.append(
                ValidationIssue(
                    code="CHAPTER_ITEM_MISMATCH",
                    message="Every exhibition item must appear in exactly one chapter.",
                )
            )

    counterpoint_ok = CuratorialRole.COUNTERPOINT.value in role_values
    if not counterpoint_ok:
        errors.append(
            ValidationIssue(
                code="COUNTERPOINT_REQUIRED",
                message="A counterpoint, counterexample, or other-voice role is required.",
            )
        )

    unique_objects_ok = len({item.object.id for item in exhibition.items}) == len(exhibition.items)
    if not unique_objects_ok:
        errors.append(
            ValidationIssue(
                code="DUPLICATE_OBJECT",
                message="An object cannot occupy more than one role in an exhibition.",
            )
        )

    selection_variety_ok, variety_errors = _validate_selection_variety(exhibition)
    errors.extend(variety_errors)

    evidence_binding_ok = True
    publication_metadata_ok = True
    for item in exhibition.items:
        evidence_by_id = {
            evidence.id: evidence for evidence in item.object.evidence
        }
        valid_evidence_ids = set(evidence_by_id)
        textual_evidence_count = sum(
            evidence.source_kind != COLLECTION_IMAGE_SOURCE_KIND
            for evidence in item.object.evidence
        )
        # Institutions differ in how much they publish; the Met has no
        # curatorial description field at all. Two locatable chunks is the
        # floor, and thin objects are already role-restricted above.
        minimum_chunks = (
            MIN_EVIDENCE_CHUNKS_FULL
            if item.object.evidence_depth == EvidenceDepth.FULL.value
            else MIN_EVIDENCE_CHUNKS_THIN
        )
        if textual_evidence_count < minimum_chunks:
            publication_metadata_ok = False
            errors.append(
                ValidationIssue(
                    code="INSUFFICIENT_EVIDENCE_CHUNKS",
                    message=(
                        f"This object has {textual_evidence_count} textual evidence chunks; "
                        f"{minimum_chunks} are required for its evidence depth."
                    ),
                    item_id=item.id,
                )
            )
        if not item.object.alt_text:
            publication_metadata_ok = False
            errors.append(
                ValidationIssue(
                    code="ALT_TEXT_PENDING",
                    message="This object needs alternative text before it can be shown.",
                    item_id=item.id,
                )
            )
        if item.object.alt_text_source != "institution_authored":
            warnings.append(
                ValidationIssue(
                    code="ALT_TEXT_SYNTHESISED",
                    message="Alternative text was synthesised from metadata and has not had a visual check.",
                    item_id=item.id,
                )
            )

        if not item.label_sentences:
            evidence_binding_ok = False
            errors.append(
                ValidationIssue(
                    code="LABEL_REQUIRED",
                    message="Each exhibition item needs at least one label sentence.",
                    item_id=item.id,
                )
            )
        for sentence in item.label_sentences:
            if not sentence.evidence_ids:
                evidence_binding_ok = False
                errors.append(
                    ValidationIssue(
                        code="EVIDENCE_REQUIRED",
                        message="Every label sentence must bind at least one evidence ID.",
                        item_id=item.id,
                        sentence_id=sentence.id,
                    )
                )
                continue
            unknown = sorted(set(sentence.evidence_ids) - valid_evidence_ids)
            if unknown:
                evidence_binding_ok = False
                errors.append(
                    ValidationIssue(
                        code="EVIDENCE_NOT_FOUND",
                        message=f"Sentence references evidence not attached to this object: {', '.join(unknown)}.",
                        item_id=item.id,
                        sentence_id=sentence.id,
                    )
                )
            referenced_chunks = [
                evidence_by_id[evidence_id]
                for evidence_id in sentence.evidence_ids
                if evidence_id in evidence_by_id
            ]
            references_image = any(
                chunk.source_kind == COLLECTION_IMAGE_SOURCE_KIND
                for chunk in referenced_chunks
            )
            if sentence.type == SentenceType.VISUAL_OBSERVATION.value:
                if not referenced_chunks or not all(
                    chunk.source_kind == COLLECTION_IMAGE_SOURCE_KIND
                    for chunk in referenced_chunks
                ):
                    evidence_binding_ok = False
                    errors.append(
                        ValidationIssue(
                            code="VISUAL_OBSERVATION_SOURCE_INVALID",
                            message="A visual observation must cite only this object's collection image evidence.",
                            item_id=item.id,
                            sentence_id=sentence.id,
                        )
                    )
            elif references_image:
                evidence_binding_ok = False
                errors.append(
                    ValidationIssue(
                        code="IMAGE_EVIDENCE_TYPE_MISMATCH",
                        message="Only a visual observation may cite collection image evidence.",
                        item_id=item.id,
                        sentence_id=sentence.id,
                    )
                )
            if sentence.type == SentenceType.INSTITUTION_FACT.value:
                referenced_evidence = [
                    evidence.text
                    for evidence in referenced_chunks
                    if evidence.source_kind != COLLECTION_IMAGE_SOURCE_KIND
                ]
                if not referenced_evidence:
                    evidence_binding_ok = False
                    errors.append(
                        ValidationIssue(
                            code="INSTITUTION_FACT_UNSUPPORTED",
                            message="An institution fact cannot be emitted without institution evidence.",
                            item_id=item.id,
                            sentence_id=sentence.id,
                        )
                    )
                elif not _institution_fact_is_extractive(sentence.text, referenced_evidence):
                    evidence_binding_ok = False
                    errors.append(
                        ValidationIssue(
                            code="INSTITUTION_FACT_NOT_EXTRACTIVE",
                            message="Institution facts must be a directly supported excerpt of the cited institution evidence.",
                            item_id=item.id,
                            sentence_id=sentence.id,
                        )
                    )

    sub_questions_ok = 2 <= len(exhibition.sub_questions) <= 4
    if not sub_questions_ok:
        errors.append(
            ValidationIssue(
                code="SUB_QUESTION_COUNT",
                message="An exhibition must contain two to four sub-questions.",
            )
        )

    checks = [
        ValidationCheck(
            key="curatorial-brief",
            label="策展任务书与证据映射",
            passed=curatorial_brief_ok,
            detail="中心命题、分论点、对象决策与检索版本均由版本化 CuratorialBrief 记录。",
        ),
        ValidationCheck(
            key="item-count",
            label="展品数量",
            passed=item_count_ok,
            detail=f"当前 {item_count} 件（按浏览时长为 5／8／12 件）。",
        ),
        ValidationCheck(
            key="public-copy-item-counts",
            label="展览文案件数一致",
            passed=public_copy_counts_ok,
            detail="标题、副标题与章节文案不得声称与实际展品不一致的明确件数。",
        ),
        ValidationCheck(
            key="curatorial-roles",
            label="策展角色齐备",
            passed=roles_ok,
            detail="引入、背景、核心证据、对照声音、综合延伸各至少出现一次。",
        ),
        ValidationCheck(
            key="counterpoint",
            label="对照或其他声音",
            passed=counterpoint_ok,
            detail="至少保留一个非迎合性视角。",
        ),
        ValidationCheck(
            key="core-evidence-depth",
            label="核心证据的材料深度",
            passed=core_evidence_depth_ok,
            detail="只有带机构撰写说明的藏品才能承担核心证据位。",
        ),
        ValidationCheck(
            key="chapters",
            label="章节编排完整",
            passed=chapters_ok,
            detail="每件展品恰好属于一个章节。",
        ),
        ValidationCheck(
            key="evidence-bindings",
            label="句级证据绑定",
            passed=evidence_binding_ok,
            detail="每句展签均须引用本件藏品的有效证据 ID。",
        ),
        ValidationCheck(
            key="publication-metadata",
            label="证据数量与无障碍元数据",
            passed=publication_metadata_ok,
            detail="每件藏品需有足够的可定位证据与替代文本。",
        ),
        ValidationCheck(
            key="unique-objects",
            label="藏品不重复",
            passed=unique_objects_ok,
            detail="每个展位使用不同藏品。",
        ),
        ValidationCheck(
            key="selection-variety",
            label="选品避免高度重复",
            passed=selection_variety_ok,
            detail="不得让四件或以上同规范题名或明确同系列对象占据展览。",
        ),
        ValidationCheck(
            key="sub-questions",
            label="子问题数量",
            passed=sub_questions_ok,
            detail="需要 2–4 个子问题。",
        ),
    ]
    blocking = [issue.message for issue in errors]
    return ValidationResult(
        passed=not errors,
        errors=errors,
        warnings=_deduplicate_warnings(warnings),
        checks=checks,
        blocking_issues=blocking,
    )


def _deduplicate_warnings(warnings: list[ValidationIssue]) -> list[ValidationIssue]:
    counts = Counter((warning.code, warning.item_id) for warning in warnings)
    seen: set[tuple[str, str | None]] = set()
    result: list[ValidationIssue] = []
    for warning in warnings:
        key = (warning.code, warning.item_id)
        if key in seen:
            continue
        seen.add(key)
        if counts[key] > 1:
            warning.message = f"{warning.message} ({counts[key]} occurrences)"
        result.append(warning)
    return result
