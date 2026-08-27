from __future__ import annotations

from collections import Counter
import re

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
        if exhibition.versions.prompt.startswith("v3-curatorial-brief"):
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

    evidence_binding_ok = True
    publication_metadata_ok = True
    for item in exhibition.items:
        valid_evidence_ids = {evidence.id for evidence in item.object.evidence}
        # Institutions differ in how much they publish; the Met has no
        # curatorial description field at all. Two locatable chunks is the
        # floor, and thin objects are already role-restricted above.
        minimum_chunks = (
            MIN_EVIDENCE_CHUNKS_FULL
            if item.object.evidence_depth == EvidenceDepth.FULL.value
            else MIN_EVIDENCE_CHUNKS_THIN
        )
        if len(valid_evidence_ids) < minimum_chunks:
            publication_metadata_ok = False
            errors.append(
                ValidationIssue(
                    code="INSUFFICIENT_EVIDENCE_CHUNKS",
                    message=(
                        f"This object has {len(valid_evidence_ids)} evidence chunks; "
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
            if sentence.type == SentenceType.INSTITUTION_FACT.value:
                referenced_evidence = [
                    evidence.text
                    for evidence in item.object.evidence
                    if evidence.id in sentence.evidence_ids
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
