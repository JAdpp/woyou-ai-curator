"""Offline operational analysis; never calls a provider or changes qrels.

Run only after strict campaign merging. Missing stages are reported as missing,
not imputed successes. Metrics retain every completed error/degraded query.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluate_retrieval import _nearest_rank_percentile, _stage_quality, load_benchmark, load_run, read_jsonl


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = Counter(str(item.get("code", "unknown")) for row in rows for item in row.get("apiErrors", []))
    warnings = Counter(str(item.get("code", "unknown")) for row in rows for item in row.get("warnings", []))
    planning = Counter(str(row.get("initialRetrievalDiagnostics", {}).get("planningStatus", "missing")) for row in rows)
    served = Counter(str(row.get("retrievalStatus", {}).get("servedMode", "missing")) for row in rows)
    stages = Counter(stage for row in rows for stage in row.get("stageResults", {}))
    latencies: dict[str, list[float]] = {}
    for row in rows:
        for stage, duration in row.get("stageLatencyMs", {}).items():
            latencies.setdefault(stage, []).append(float(duration))
        for key in ("planElapsedMs", "searchElapsedMs"):
            duration = row.get("initialRetrievalDiagnostics", {}).get(key)
            if isinstance(duration, (int, float)):
                latencies.setdefault(f"initial.{key}", []).append(float(duration))
    observations = []
    for row in rows:
        diagnostics = row.get("initialRetrievalDiagnostics", {})
        if row.get("apiErrors") or row.get("warnings") or diagnostics.get("planningStatus") not in ("applied", "not_needed"):
            observations.append({"queryId": row["queryId"], "question": row.get("question"),
                                 "planning": diagnostics,
                                 "servedMode": row.get("retrievalStatus", {}).get("servedMode"),
                                 "apiErrors": row.get("apiErrors", []), "warnings": row.get("warnings", [])})
    gate_rows = [row for row in rows if not row.get("apiErrors") and len(row.get("acceptedResults", [])) >= 5
                 and row.get("answerability") in ("supported", "partially_supported")]
    required_rows = [row for row in rows if row.get("requiredCulturalLegs")]
    covered_gate_rows = []
    for row in gate_rows:
        observed = {leg for item in row.get("acceptedResults", []) for leg in item.get("culturalLegs", [])}
        if set(row.get("requiredCulturalLegs", [])).issubset(observed):
            covered_gate_rows.append(row)
    return {
        "queryCount": len(rows),
        "apiErrorQueries": sum(bool(row.get("apiErrors")) for row in rows),
        "initialRetrievalErrorQueries": [row["queryId"] for row in rows
                                        if any(item.get("stage") == "retrieval" for item in row.get("apiErrors", []))],
        "warningQueries": sum(bool(row.get("warnings")) for row in rows),
        "apiErrorCodes": dict(errors), "warningCodes": dict(warnings),
        "planningStatuses": dict(planning), "servedModes": dict(served),
        "auditAppliedCount": sum(row.get("auditApplied") is True for row in rows),
        "answerability": dict(Counter(str(row.get("answerability", "missing")) for row in rows)),
        "acceptedSetObservedCount": sum("acceptedResults" in row for row in rows),
        "acceptedSetNonemptyCount": sum(bool(row.get("acceptedResults")) for row in rows),
        "finalExhibitSetObservedCount": sum("finalResults" in row for row in rows),
        "stageObservedQueryCounts": dict(stages),
        "preGenerationEvidenceGate": {
            "definition": "no API hard error, at least 5 accepted candidates, answerability supported or partially_supported; not actual exhibition success or independently verified evidence sufficiency",
            "passedQuestionCount": len(gate_rows), "denominator": len(rows),
            "passedRate": len(gate_rows) / len(rows) if rows else None,
            "passedQueryIds": [row["queryId"] for row in gate_rows],
            "passedAndAllRequiredLegsCoveredCount": len(covered_gate_rows),
            "emptyRequiredLegsCountAsCovered": True,
            "explicitCulturalRequirementsQuestionCount": len(required_rows),
            "passedAndExplicitRequiredLegsCoveredCount": sum(bool(row.get("requiredCulturalLegs")) for row in covered_gate_rows),
            "culturalCoverageBasis": "accepted candidate serialized culturalLegs metadata; not independent evidence adjudication",
        },
        "latencyMs": {stage: {"count": len(values), "p50": _nearest_rank_percentile(values, .5),
                               "p95": _nearest_rank_percentile(values, .95), "max": max(values)}
                      for stage, values in sorted(latencies.items())},
        "diagnosticObservations": observations,
    }


def log_diagnostics(folder: Path, pipeline_version: str = "") -> dict[str, Any]:
    patterns = {
        "initial_planning_unavailable": "Initial catalogue planning unavailable",
        "pre_audit_search_timeout": "optional pre-audit catalogue planning search timed out",
        "pre_audit_search_skipped": "optional pre-audit catalogue planning search skipped",
        "fused_rerank_unavailable": "Optional fused reranking unavailable",
        "qwen_rerank_unavailable": "Qwen rerank unavailable",
        "hybrid_degraded": "hybrid RAG degraded to BM25",
        "atomic_prefilter_unavailable": "atomic semantic prefilter unavailable",
        "batched_search_timeout": "batched agentic retrieval timed out",
        "batched_search_cooperative_deadline": "batched agentic retrieval reached its cooperative deadline",
        "audit_unavailable": "retrieval audit unavailable",
        "expanded_audit_skipped": "optional expanded evidence audit skipped",
    }
    events = []
    for path in sorted((folder / "logs").glob("*.log")):
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for name, pattern in patterns.items():
                if pattern in line:
                    events.append({"shard": path.stem, "line": number, "event": name, "message": line})
    return {
        "eventCounts": dict(Counter(item["event"] for item in events)), "events": events,
        "scope": "log occurrences; not deduplicated query-level failure rates",
        "traceBoundary": (
            "V5 trace capture uses a request generation ContextVar and lock; late background writes from older requests are rejected. Collection-level retrievalStatus remains a shared health snapshot, not authoritative per-request mode. Logger-only optional fallbacks still require separate log inspection; JSONL warnings are not exhaustive."
            if pipeline_version == "phase1-agentic-v5-isolated-traces-20260906" else
            "Trace capture is one mutable slot per worker. A timed-out background search may complete later and write into a subsequent trace. Direct returned result IDs are separate; logged warnings and fine-grained per-stage attribution may be incomplete or cross-query. Do not claim complete warning coverage from JSONL alone."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    args = parser.parse_args()
    folder = args.campaign_dir.resolve()
    plan = json.loads((folder / "campaign-plan.json").read_text(encoding="utf-8"))
    state = json.loads((folder / "campaign-state.json").read_text(encoding="utf-8"))
    if state["status"] != "complete":
        raise ValueError("Operational final report requires a complete strict-merged campaign")
    report: dict[str, Any] = {
        "campaignId": plan["campaignId"], "networkPath": plan["networkPath"],
        "workers": plan["maxConcurrentWorkers"], "latencyScope": plan["latencyScope"],
        "culturalRoutingBoundary": plan["culturalRoutingBoundary"],
        "analysisSourceSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "groups": {},
        "logDiagnostics": log_diagnostics(folder, plan.get("retrievalPipelineVersion", "")),
    }
    project = Path(__file__).resolve().parents[1]
    benchmark_dir = project / "data/qa/retrieval_eval_v1"
    benchmark = load_benchmark(benchmark_dir / "manifest.json", benchmark_dir / "questions.jsonl", benchmark_dir / "qrels.jsonl")
    for group, spec in plan["groups"].items():
        path = folder / f"{group}.jsonl"
        rows = read_jsonl(path)
        if [row["queryId"] for row in rows] != spec["queryIds"]:
            raise ValueError(f"Strict-merged membership mismatch: {group}")
        result = summarize(rows)
        result["runSha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result["byCategory"] = {category: summarize([row for row in rows if row.get("category") == category])
                                for category in sorted({row.get("category", "missing") for row in rows})}
        report["groups"][group] = result
    report["deterministicCategoryComparison"] = {}
    for name, path in {
        "bm25-20260831": project / "artifacts/qa/retrieval-runs/bm25-deterministic-100.jsonl",
        "qwen-20260831": project / "artifacts/qa/retrieval-runs/qwen-rerank-deterministic-100.jsonl",
        "recovered": folder / "hybrid-fields.jsonl",
    }.items():
        records = load_run(path, benchmark, allow_partial=True)
        categories = sorted({benchmark.questions[query_id]["category"] for query_id in records})
        report["deterministicCategoryComparison"][name] = {
            category: _stage_quality(benchmark, {query_id: row for query_id, row in records.items()
                                                if benchmark.questions[query_id]["category"] == category}, "initial")
            for category in categories}
    misses = []
    for row in read_jsonl(folder / "hybrid-fields.jsonl"):
        relevant = {object_id for object_id, judgment in benchmark.qrels[row["queryId"]].items()
                    if judgment["relevance"] >= benchmark.relevance_threshold}
        retrieved = {item["objectId"] for item in row["results"][:50]}
        if relevant - retrieved:
            misses.append({"queryId": row["queryId"], "question": row["question"],
                           "goldCount": len(relevant), "recalledCount": len(relevant & retrieved),
                           "missingGoldObjectIds": sorted(relevant - retrieved),
                           "planning": row.get("initialRetrievalDiagnostics", {})})
    report["deterministicMisses"] = misses
    (folder / "runtime-analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Campaign runtime analysis", "", f"Network: `{report['networkPath']}`; workers: {report['workers']}.",
             "", "All rows, including failures and fallbacks, remain in latency summaries. This is not a production single-request benchmark.",
             "", "| Group | Queries | API-error queries | Warning queries | Applied audits | Total P50 / P95 (s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, group in report["groups"].items():
        total = group["latencyMs"].get("total", {})
        lines.append(f"| {name} | {group['queryCount']} | {group['apiErrorQueries']} | {group['warningQueries']} | {group['auditAppliedCount']} | {total.get('p50', 0)/1000:.2f} / {total.get('p95', 0)/1000:.2f} |")
    lines += ["", "## Deterministic field Recall@50 by category", "",
              "| Category | BM25 Aug31 | Qwen Aug31 | Recovered |", "| --- | ---: | ---: | ---: |"]
    for category in report["deterministicCategoryComparison"]["recovered"]:
        cells = [f"{report['deterministicCategoryComparison'][name][category]['known_pool_recall_at_50']:.4f}"
                 for name in ("bm25-20260831", "qwen-20260831", "recovered")]
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    lines += ["", "This table uses exhaustive frozen field gold. The shared scorer's known-pool field is full field-gold Recall for these deterministic questions, not a claim about open-query corpus recall."]
    lines += ["", "## Pre-generation evidence gate candidates", "",
              "| Open-query run | Candidate gate passed | Passed and all explicit cultures covered |",
              "| --- | ---: | ---: |"]
    for name in ("hybrid-open", "bm25-open"):
        gate = report["groups"][name]["preGenerationEvidenceGate"]
        lines.append(f"| {name} | {gate['passedQuestionCount']}/{gate['denominator']} | {gate['passedAndExplicitRequiredLegsCoveredCount']}/{gate['explicitCulturalRequirementsQuestionCount']} |")
    lines += ["", "The candidate gate requires no hard API error, at least five accepted candidates, and a supported/partially-supported answerability label. It is not actual exhibition success, evidence correctness, or an independent cultural-coverage judgment; culture coverage above uses serialized accepted-object metadata."]
    lines += ["", "Initial planning status, served retrieval mode, stage presence, per-stage latency and individual diagnostic observations are retained in `runtime-analysis.json`.",
              "", report["logDiagnostics"]["traceBoundary"],
              "", "Logger-only fallbacks are counted separately as events, not silently treated as successful no-warning queries or exact per-query rates.",
              "", "Accepted candidate sets are not final exhibition artifacts. A missing final exhibit set is not an exhibition-generation pass.",
              "", "Ranking and AI-reviewed-pool metrics remain in the separate comparison reports. AI review is not human gold; unjudged candidates are not irrelevant."]
    (folder / "runtime-analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(folder / 'runtime-analysis.json'), "groups": {key: value['queryCount'] for key, value in report['groups'].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
