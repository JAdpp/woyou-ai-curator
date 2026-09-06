"""Save an authorized second review with explicit delegated-AI provenance.

Only complete, validated second-pass questions qualify. Existing human decisions
remain in place; frozen qrels are never edited. Default is a read-only preflight.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from uuid import NAMESPACE_URL, uuid5

import httpx


PREFIX = "/api/admin/retrieval-eval/reviews"
DECISION_FIELDS = ("relevance", "evidenceVerdict", "supportingEvidenceIds")


def decision_input(suggestion: dict, current: dict | None, *, final: bool = False) -> dict:
    fields = ("expectedAnswerability",) if final else DECISION_FIELDS
    return {
        **{key: suggestion[key] for key in fields},
        "reviewOrigin": "delegated_ai",
        "note": "授权 AI 二次复审：" + suggestion["note"].removeprefix("AI代审，待复核："),
        "acceptedSuggestionId": suggestion["suggestionId"],
        "disposition": "accepted",
        "expectedRevision": current["revision"] if current else None,
        "requestId": str(uuid5(NAMESPACE_URL, "delegated-second-review:" + suggestion["suggestionId"])),
    }


def check_question(detail: dict) -> None:
    final = detail.get("aiQuestionSuggestion")
    if not final or not final["promptVersion"].startswith("finalization-text-second"):
        raise ValueError("second-pass finalization missing")
    if any(str(flag).startswith("error:") for flag in final.get("riskFlags", [])):
        raise ValueError("unresolved finalization failure")
    for candidate in detail["candidates"]:
        existing = candidate.get("judgment")
        if existing and existing.get("reviewOrigin", "human") == "human":
            continue
        suggestion = candidate.get("aiSuggestion")
        if not suggestion or not suggestion["promptVersion"].startswith("candidate-text-second"):
            raise ValueError("second-pass candidate missing: " + candidate["objectId"])
        if suggestion["suggestionRunId"] != final["suggestionRunId"]:
            raise ValueError("mixed candidate/finalization runs")
        if any(str(flag).startswith("error:") for flag in suggestion["riskFlags"]):
            raise ValueError("unresolved provider or image failure")
        owned = {chunk["id"] for chunk in candidate["evidence"]}
        if not set(suggestion["supportingEvidenceIds"]) <= owned:
            raise ValueError("foreign evidence")
        if suggestion["evidenceVerdict"] == "supports" and (
            suggestion["relevance"] < 2 or not suggestion["supportingEvidenceIds"]
        ):
            raise ValueError("unsupported supporting verdict")
        if detail["question"]["category"] == "visual_motif" and "image" not in suggestion["modalities"]:
            raise ValueError("visual motif was not image-reviewed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Save only fully validated second-pass questions; leave incomplete questions untouched.")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("artifacts/qrel-review/delegated_completion_20260906.json"))
    args = parser.parse_args()
    with httpx.Client(base_url=args.api_base_url, trust_env=False, timeout=60) as client:
        def request(method: str, path: str, payload: dict | None = None) -> dict:
            # Stable requestId makes a PUT retry safe after a lost response.
            # This bounded retry also tolerates the local API's hot reload.
            for attempt in range(4):
                try:
                    response = client.request(method, PREFIX + path, json=payload)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPError as error:
                    transient = not isinstance(error, httpx.HTTPStatusError) or error.response.status_code in {502, 503, 504}
                    if not transient or attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            raise RuntimeError("unreachable local API retry state")

        def get(path: str) -> dict:
            return request("GET", path)

        def put(path: str, payload: dict) -> dict:
            return request("PUT", path, payload)

        summary = get("/questions")
        details = [get("/questions/" + row["queryId"]) for row in summary["questions"]]
        issues = {}
        for detail in details:
            try:
                check_question(detail)
            except ValueError as error:
                issues[detail["question"]["queryId"]] = str(error)
        if issues and not args.allow_partial:
            print(json.dumps({"ready": False, "issues": issues}, ensure_ascii=False))
            return 2
        details = [detail for detail in details if detail["question"]["queryId"] not in issues]
        if not args.apply:
            print(json.dumps({"ready": bool(details), "questions": len(details), "skipped": issues, "progress": summary["progress"]}))
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        backup = args.output.with_suffix(".before.json")
        if not backup.exists():
            backup.write_text(json.dumps(get("/export"), ensure_ascii=False, indent=2), encoding="utf-8")
        counts = Counter()
        for index, detail in enumerate(details, 1):
            query_id = detail["question"]["queryId"]
            # Refresh the revision immediately before writing, preserving any
            # human actions performed while the review was running.
            detail = get("/questions/" + query_id)
            check_question(detail)
            for candidate in detail["candidates"]:
                current = candidate.get("judgment")
                if current and current.get("reviewOrigin", "human") == "human":
                    counts["preservedHumanCandidates"] += 1
                    continue
                suggestion = candidate["aiSuggestion"]
                if current and current.get("acceptedSuggestionId") == suggestion["suggestionId"]:
                    counts["alreadyDelegated"] += 1
                    continue
                put(f"/questions/{query_id}/candidates/{candidate['objectId']}", decision_input(suggestion, current))
                counts["delegatedCandidates"] += 1
            # Candidate writes may invalidate a previously current finalization.
            # Read its current state rather than the pre-write snapshot.
            final_detail = get("/questions/" + query_id)
            final = final_detail["aiQuestionSuggestion"]
            current_final = final_detail.get("finalization")
            if current_final and current_final.get("reviewOrigin", "human") == "human":
                counts["preservedHumanFinalizations"] += 1
            elif current_final and current_final.get("acceptedSuggestionId") == final["suggestionId"] and current_final.get("isCurrent"):
                counts["alreadyFinalized"] += 1
            else:
                put(f"/questions/{query_id}/finalization", decision_input(final, current_final, final=True))
                counts["delegatedFinalizations"] += 1
            print(f"[{index}/{len(details)}] {query_id} saved", flush=True)
        result = {
            "reviewOrigin": "delegated_ai", "humanGold": False,
            "completedAt": datetime.now(timezone.utc).isoformat(),
            "counts": dict(counts), "progress": get("/questions")["progress"],
            "skippedQuestions": issues,
            "beforeSnapshot": str(backup), "snapshot": get("/export"),
        }
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("counts", "progress")}, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
