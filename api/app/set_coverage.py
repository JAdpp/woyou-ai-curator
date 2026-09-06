"""Aggregate already source-checked, single-object witnesses for a visit.

This module never invents evidence or weakens per-object conditions. Coverage
means model-judged support with verified ownership, not semantic ground truth.
"""
from __future__ import annotations

from hashlib import sha256
import re
from typing import Any, Sequence

from .collections import SearchResult


def set_coverage(
    requirements: Sequence[dict[str, Any]], results: Sequence[SearchResult], *,
    question: str, selected_ids: set[str] | None = None,
) -> dict[str, Any]:
    question_hash = sha256(question.encode("utf-8")).hexdigest()
    rows = []
    for requirement in requirements:
        witnesses: dict[str, dict[str, Any]] = {}
        for result in results:
            if selected_ids is not None and result.obj.id not in selected_ids:
                continue
            if "condition_source_bound" not in result.retrieval_sources:
                continue
            for witness in result.set_witnesses:
                if (
                    witness.get("requirementId") != requirement["id"]
                    or witness.get("objectId") != result.obj.id
                    or witness.get("requirementText") != requirement["text"]
                    or witness.get("sourceQuote") != requirement["sourceQuote"]
                    or witness.get("questionSha256") != question_hash
                ):
                    continue
                checks = witness.get("checks", [])
                if not checks or not witness.get("evidenceIds") or any(
                    check.get("objectId") != result.obj.id
                    or check.get("conditionId") != requirement["id"]
                    or check.get("status") != "supported"
                    or check.get("sourceBound") is not True
                    or check.get("failure")
                    or check.get("relation") not in {"exact", "narrower"}
                    for check in checks
                ):
                    continue
                witnesses[result.obj.id] = witness
        minimum = requirement["minWitnesses"]
        rows.append({
            **requirement, "witnessObjectIds": list(witnesses),
            "witnesses": list(witnesses.values()),
            "witnessCount": len(witnesses), "satisfied": len(witnesses) >= minimum,
        })
    return {
        "version": "exhibition-set-coverage-v1", "questionSha256": question_hash,
        "scope": "final_selected_set" if selected_ids is not None else "audited_pool",
        "requirements": rows, "satisfied": all(row["satisfied"] for row in rows),
        "missingRequirementIds": [row["id"] for row in rows if not row["satisfied"]],
        "semanticEntailmentProven": False,
        "boundary": "Each witness must independently support the whole relation; source binding is not a proof of semantic correctness.",
    }


def set_coverage_gap(coverage: dict[str, Any], language: str) -> str:
    missing = [row for row in coverage["requirements"] if not row["satisfied"]]
    if not missing:
        return ""
    targets = "；".join(row["sourceQuote"] for row in missing)
    if language == "en":
        return (f"This search has not verified enough single-object witnesses for the exhibition goal: {targets}. "
                "Related background objects do not establish that goal; this does not establish a collection-wide absence.")
    return (f"本次检索尚未核实足够的核心例子来回应展览目标：{targets}。"
            "相关的背景展品不能代替这一目标的直接证据；这不表示整个馆藏库都没有资料。")


def persisted_set_coverage_valid(exhibition) -> bool:
    """Catch stale membership/source bindings after edits or serialization.

    This does not rerun a semantic critic. An institution quote is checked
    against its own record; a visual citation must retain its inspected-image
    proof. Legacy exhibitions without set requirements remain compatible.
    """
    report = exhibition.exhibition_set_coverage
    if not report:
        return True
    rows = report.get("requirements")
    if not isinstance(rows, list):
        return False
    if not rows:
        return True
    question = exhibition.agenda.question
    question_hash = sha256(question.encode("utf-8")).hexdigest()
    if (report.get("scope") != "final_selected_set" or report.get("satisfied") is not True
            or report.get("questionSha256") != question_hash):
        return False
    by_id = {item.object.id: item.object for item in exhibition.items}
    results = []
    requirements = []
    normalize = lambda value: re.sub(r"\s+", " ", value).strip().casefold()
    seen_ids = set()
    try:
        for row in rows:
            key = row["id"]
            if key in seen_ids or row["sourceQuote"] not in question:
                return False
            seen_ids.add(key)
            minimum = row["minWitnesses"]
            if type(minimum) is not int or not 1 <= minimum <= 3:
                return False
            requirements.append(row)
            for witness in row["witnesses"]:
                obj = by_id.get(witness["objectId"])
                if obj is None:
                    return False
                sources = {chunk.id: chunk.text for chunk in obj.evidence}
                proof = obj.visual_core_evidence
                for check in witness["checks"]:
                    evidence_id = check["evidenceId"]
                    if check.get("sourceKind") == "collection_image_observation":
                        if (row["evidenceScope"] != "visible_features_or_record" or not proof
                                or proof.object_id != obj.id or proof.question_sha256 != question_hash
                                or proof.source_url not in {obj.image_url, obj.image_url_large}
                                or evidence_id not in proof.observation_ids
                                or check.get("imageEvidenceId") != proof.image_evidence_id):
                            return False
                    elif (evidence_id not in sources or not check["supportingQuote"].strip()
                          or normalize(check["supportingQuote"]) not in normalize(sources[evidence_id])):
                        return False
                results.append(SearchResult(obj=obj, score=0, set_witnesses=(witness,),
                                            retrieval_sources=("condition_source_bound",)))
        recomputed = set_coverage(requirements, results, question=question, selected_ids=set(by_id))
        return recomputed["satisfied"] and all(
            set(old["witnessObjectIds"]) == set(new["witnessObjectIds"])
            and old["witnessCount"] == new["witnessCount"]
            for old, new in zip(rows, recomputed["requirements"])
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
