"""Score accepted top-five candidates with additive, provenance-checked reviews.

Offline only. Unknown judgments stay unknown. Short lists and API failures stay
in the question denominator; this is not whole-corpus recall or exhibit quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluate_retrieval import load_benchmark
from retrieval_review_overlay import load_review_overlay

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = "user_authorized_ai_engineering"


def read_jsonl(path: Path) -> list[dict]:
    # Evaluation's reader adds _sourceLine; immutable pool hashing must not.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def load_labels(benchmark_dir: Path, overlay: Path, pool_dirs: list[Path]):
    base = load_benchmark(benchmark_dir / "manifest.json", benchmark_dir / "questions.jsonl",
                          benchmark_dir / "qrels.jsonl")
    benchmark, provenance = load_review_overlay(base, overlay)
    labels = {(qid, oid): row for qid, values in benchmark.qrels.items() for oid, row in values.items()
              if row.get("judgmentStatus", "").startswith("reviewed_")}
    objects_payload = json.loads((ROOT / base.manifest["provenance"]["objectsFile"]).read_text(encoding="utf-8"))
    objects_list = objects_payload.get("objects", []) if isinstance(objects_payload, dict) else objects_payload
    evidence = {obj["id"]: {item["id"] for item in obj.get("evidence", [])} for obj in objects_list}
    sources = []
    for directory in pool_dirs:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        packages = read_jsonl(directory / "blind-inputs.jsonl")
        if (manifest["frozenObjectsSha256"] != provenance["sourceHashes"]["objects"]
                or manifest["reviewExportSha256"] != provenance["sha256"]
                or manifest["poolSha256"] != digest(packages)):
            raise ValueError("pool input or frozen source identity mismatch")
        allowed = set()
        for package in packages:
            qid = package["queryId"]
            if package["payload"]["question"] != benchmark.questions[qid]["question"]:
                raise ValueError("blind question differs from frozen benchmark")
            for obj in package["payload"]["candidates"]:
                pair = (qid, obj["objectId"])
                if pair in allowed or obj["objectId"] not in evidence:
                    raise ValueError("duplicate or unknown blind candidate")
                allowed.add(pair)
        latest = {}
        for row in read_jsonl(directory / "decisions.jsonl"):
            if row["runId"] != manifest["runId"] or row["reviewOrigin"] != ORIGIN:
                raise ValueError("decision review identity mismatch")
            pair = (row["queryId"], row["objectId"])
            if pair not in allowed:
                raise ValueError("decision outside immutable blind pool")
            if row["stage"] == "final":
                latest[pair] = row
        added = 0
        for pair, row in latest.items():
            if row.get("status") != "judged":
                continue
            decision = row["decision"]
            rel, verdict = decision.get("relevance"), decision.get("evidenceVerdict")
            ids = decision.get("supportingEvidenceIds", [])
            if (type(rel) is not int or rel not in range(4)
                    or verdict not in {"supports", "insufficient", "contradicts", "uncertain", "not_applicable"}
                    or not isinstance(ids, list) or not set(ids).issubset(evidence[pair[1]])
                    or (verdict == "supports" and (rel < 2 or not ids))):
                raise ValueError("invalid relevance or evidence ownership in final judgment")
            if pair in labels:
                raise ValueError("review pools overlap; do not silently overwrite prior labels")
            labels[pair] = {**decision, "reviewOrigin": ORIGIN, "imageSeen": row.get("imageSeen", False),
                            "reviewRunId": manifest["runId"], "reviewModel": row.get("model")}
            added += 1
        sources.append({"directory": str(directory), "manifestSha256": sha256(directory / "manifest.json"),
                        "decisionsSha256": sha256(directory / "decisions.jsonl"), "runId": manifest["runId"],
                        "poolPairs": len(allowed), "judgedPairs": added,
                        "unjudgedPairs": len(allowed) - added})
    return benchmark, labels, {"overlaySha256": provenance["sha256"],
                                "frozenSourceHashes": provenance["sourceHashes"], "pools": sources}


def score_rows(rows: list[dict], labels: dict[tuple[str, str], dict]) -> dict:
    per_question = []
    origins: Counter[str] = Counter()
    for row in rows:
        qid = row["queryId"]
        accepted = row.get("acceptedResults")
        if not isinstance(accepted, list):
            raise ValueError("acceptedResults must be observed explicitly")
        ids = [item["objectId"] for item in accepted[:5]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate accepted objects")
        judgments = [labels.get((qid, oid)) for oid in ids]
        known = [label for label in judgments if label is not None]
        relevant = sum(label["relevance"] >= 2 for label in known)
        supported = sum(label["relevance"] >= 2 and label["evidenceVerdict"] == "supports" for label in known)
        for label in known:
            origins[label["reviewOrigin"]] += 1
        per_question.append({
            "queryId": qid, "question": row.get("question"), "category": row.get("category"),
            "acceptedCount": len(accepted), "returnedTop5": len(ids), "judged": len(known),
            "unjudgedObjectIds": [oid for oid, label in zip(ids, judgments) if label is None],
            "relevant": relevant, "evidenceSupported": supported,
            "apiError": bool(row.get("apiErrors")),
            "candidateGate": not row.get("apiErrors") and len(accepted) >= 5
                             and row.get("answerability") in {"supported", "partially_supported"},
        })
    count = len(rows)
    returned = sum(item["returnedTop5"] for item in per_question)
    judged = sum(item["judged"] for item in per_question)
    relevant = sum(item["relevant"] for item in per_question)
    supported = sum(item["evidenceSupported"] for item in per_question)
    full = [item for item in per_question if item["returnedTop5"] == item["judged"] == 5]
    return {
        "questionCount": count, "returnedTop5Occurrences": returned, "judgedOccurrences": judged,
        "unjudgedOccurrences": returned - judged, "judgmentCoverage": judged / returned if returned else None,
        "relevantOccurrences": relevant, "evidenceSupportedOccurrences": supported,
        "judgedRelevancePrecision": relevant / judged if judged else None,
        "judgedEvidenceSupportPrecision": supported / judged if judged else None,
        "missingSlotsToFive": 5 * count - returned,
        "shortListQuestions": sum(item["returnedTop5"] < 5 for item in per_question),
        "emptyListQuestions": sum(item["returnedTop5"] == 0 for item in per_question),
        "fullyJudgedFiveItemQuestions": len(full),
        "candidateGateAndFiveReviewedRelevant": sum(item["candidateGate"] and item["relevant"] == 5 for item in full),
        "candidateGateAndFiveReviewedEvidenceSupported": sum(item["candidateGate"] and item["evidenceSupported"] == 5 for item in full),
        "reviewOriginOccurrences": dict(origins), "perQuestion": per_question,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--pool-dir", type=Path, action="append", required=True)
    parser.add_argument("--overlay", type=Path, default=ROOT / "artifacts/qrel-review/delegated_completion_20260906.json")
    args = parser.parse_args()
    folder = args.campaign_dir.resolve()
    state = json.loads((folder / "campaign-state.json").read_text(encoding="utf-8"))
    plan = json.loads((folder / "campaign-plan.json").read_text(encoding="utf-8"))
    if state["status"] != "complete":
        raise ValueError("scoring requires a completed campaign")
    benchmark, labels, provenance = load_labels(ROOT / "data/qa/retrieval_eval_v1", args.overlay, args.pool_dir)
    report = {"campaignId": plan["campaignId"], "labelProvenance": provenance,
              "scorerSha256": sha256(Path(__file__)), "groups": {},
              "boundary": "AI engineering review, not independent human gold. Different historical review prompt/image policies remain identifiable. Scores apply to accepted top-five candidates, not final exhibits or corpus recall. Unknown is never relevance zero; precision excludes unknown and missing slots, reported separately. Five-supported counts require all five judged and supporting, not proof of cross-cultural explanation completeness."}
    for group in ("hybrid-open", "bm25-open"):
        path = folder / f"{group}.jsonl"
        rows = read_jsonl(path)
        if [row["queryId"] for row in rows] != plan["groups"][group]["queryIds"]:
            raise ValueError("merged run membership differs from plan")
        for row in rows:
            if row["question"] != benchmark.questions[row["queryId"]]["question"]:
                raise ValueError("run question differs from frozen benchmark")
        report["groups"][group] = {**score_rows(rows, labels), "runSha256": sha256(path)}
    output = folder / "accepted-pool-quality.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Accepted top-five review", "", report["boundary"], "",
             "| Run | Judged / returned | Relevant / judged | Supported / judged | Short lists | Gate + five supported / all questions |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, values in report["groups"].items():
        lines.append(f"| {name} | {values['judgedOccurrences']}/{values['returnedTop5Occurrences']} | "
                     f"{values['relevantOccurrences']}/{values['judgedOccurrences']} | "
                     f"{values['evidenceSupportedOccurrences']}/{values['judgedOccurrences']} | "
                     f"{values['shortListQuestions']} | {values['candidateGateAndFiveReviewedEvidenceSupported']}/{values['questionCount']} |")
    (folder / "accepted-pool-quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "groups": {name: {key: value for key, value in values.items() if key != 'perQuestion'} for name, values in report['groups'].items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
