"""Prepare/run isolated local evaluation shards, then merge and score strictly.

The default only writes a plan. ``--execute`` performs billable provider calls.
Each shard owns its process, trace capture and output; no retrieval repository
is shared between concurrent workers. Recorded latency includes this local
campaign's concurrent load and is not a production single-request benchmark.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from threading import Event, Lock
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate_retrieval import evaluate_benchmark, load_benchmark, read_jsonl, render_markdown
from run_retrieval_evaluation import RETRIEVAL_PIPELINE_VERSION

ENVIRONMENT = {
    "RAG_EMBEDDING_PROVIDER": "aliyun",
    "RAG_EMBEDDING_MODEL": "qwen3.7-text-embedding",
    "RAG_EMBEDDING_DIMENSION": "768",
    "RAG_RERANK_ENABLED": "true",
    "RAG_RERANK_MODEL": "qwen3-rerank",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUNBUFFERED": "1",
}
GROUPS = {
    "hybrid-fields": ("hybrid", "deterministic_field_gold", False),
    "hybrid-boundary": ("hybrid", "deterministic_evidence_boundary_no_relevant_objects", True),
    "hybrid-open": ("hybrid", "pooled_silver_pending_human_review", True),
    "bm25-open": ("bm25", "pooled_silver_pending_human_review", True),
}


class CampaignError(ValueError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def runtime_fingerprint() -> dict[str, str]:
    files = sorted((PROJECT_ROOT / "api/app").rglob("*.py"))
    files += [PROJECT_ROOT / "scripts" / name for name in (
        "run_retrieval_campaign.py", "run_retrieval_evaluation.py",
        "evaluate_retrieval.py", "retrieval_review_overlay.py",
    )]
    return {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path) for path in files}


def campaign_environment(network_path: str, https_proxy: str | None) -> dict[str, str]:
    environment = dict(ENVIRONMENT)
    if network_path == "temporary-local-connect-via-existing-server":
        parsed = urlparse(https_proxy or "")
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port
                or parsed.username or parsed.password or parsed.path not in ("", "/")
                or parsed.query or parsed.fragment):
            raise CampaignError("Temporary relay must be a credential-free http://127.0.0.1:PORT")
        environment.update(HTTPS_PROXY=https_proxy or "", NO_PROXY="api.deepseek.com,localhost,127.0.0.1")
    elif https_proxy:
        raise CampaignError("An HTTPS proxy requires the explicit temporary relay network path")
    return environment


def validate_health_report(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Fail closed before billable shards; root's same-path smoke is kept separately."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError("A readable provider smoke report is required") from error
    checks = (
        report.get("pass") is True,
        report.get("networkPath") == plan["networkPath"],
        report.get("httpsProxy") == plan["environment"].get("HTTPS_PROXY"),
        report.get("runtimeSourceSha256") == plan["runtimeSourceSha256"] == runtime_fingerprint(),
        report.get("embedding", {}).get("model") == ENVIRONMENT["RAG_EMBEDDING_MODEL"],
        report.get("embedding", {}).get("dimension") == int(ENVIRONMENT["RAG_EMBEDDING_DIMENSION"]),
        report.get("rerank", {}).get("model") == ENVIRONMENT["RAG_RERANK_MODEL"],
        report.get("retrieval", {}).get("actualMode") == "hybrid",
        report.get("retrieval", {}).get("denseQueryFailed") is False,
    )
    if not all(checks):
        raise CampaignError("Smoke health/model/network/source identity gate failed")
    try:
        generated = datetime.fromisoformat(report["generatedAt"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - generated).total_seconds()
    except (KeyError, ValueError, TypeError) as error:
        raise CampaignError("Smoke generatedAt must include a timezone") from error
    if not 0 <= age <= 1800:
        raise CampaignError("Provider smoke report must be no more than 30 minutes old")
    return {"path": str(path.resolve()), "sha256": sha256(path), "generatedAt": report["generatedAt"], "pass": True}


def make_plan(benchmark: Any, output: Path, *, shard_size: int, workers: int,
              campaign_id: str = "phase1-20260906", network_path: str = "local-default",
              https_proxy: str | None = None, health_report: Path | None = None) -> dict[str, Any]:
    if shard_size < 1 or not 1 <= workers <= 4:
        raise CampaignError("shard-size must be positive and workers must be 1..4")
    groups = {}
    shards = []
    for group, (mode, judgment, audit) in GROUPS.items():
        questions = [row for row in benchmark.questions.values() if row.get("judgmentMode") == judgment]
        groups[group] = {"mode": mode, "audit": audit,
                         "queryIds": [row["queryId"] for row in questions],
                         "runName": f"{campaign_id}-{group}"}
        # Keep categories together so every shard has an interpretable stratum.
        for category in sorted({row["category"] for row in questions}):
            ids = [row["queryId"] for row in questions if row["category"] == category]
            for offset in range(0, len(ids), shard_size):
                shard_id = f"{group}-{category}-{offset // shard_size + 1:02}"
                shards.append({"id": shard_id, "group": group, "category": category,
                               "queryIds": ids[offset:offset + shard_size],
                               "output": str(output / "shards" / f"{shard_id}.jsonl")})
    return {
        "schemaVersion": 1, "createdAt": now(),
        "benchmark": {"id": benchmark.manifest["benchmarkId"], "version": benchmark.manifest["version"],
                      "provenance": benchmark.manifest["provenance"],
                      "questionsSha256": sha256(benchmark.questions_path),
                      "qrelsSha256": sha256(benchmark.qrels_path)},
        "runtimeSourceSha256": runtime_fingerprint(),
        "retrievalPipelineVersion": RETRIEVAL_PIPELINE_VERSION,
        "campaignId": campaign_id,
        "networkPath": network_path,
        "healthReportPath": str(health_report.resolve()) if health_report else None,
        "environment": campaign_environment(network_path, https_proxy), "maxConcurrentWorkers": workers,
        "host": {"platform": platform.platform(), "python": sys.version, "cpuCount": os.cpu_count()},
        "latencyScope": "local_concurrent_campaign_not_production_single_request",
        "culturalRoutingBoundary": {
            "version": "controlled-origin-v2",
            "change": "derived controlled India place routing corrects 837 southeast-Asia assignments and adds 2 south-Asia assignments; raw object records and frozen qrels unchanged",
            "metricRisk": "frozen raw culturePackIds gold may disagree with corrected serving geography; do not attribute every metric delta to ranking quality",
        },
        "derivedIndexManifests": [{"path": str(path.relative_to(PROJECT_ROOT)),
                                   "sha256": sha256(path), "manifest": json.loads(path.read_text(encoding="utf-8"))}
                                  for directory in (PROJECT_ROOT / "api/runtime/cache/filters/global_open",
                                                    PROJECT_ROOT / "api/runtime/cache/rag/global_open/qwen3.7-text-embedding")
                                  for path in sorted(directory.rglob("manifest.json"))],
        "groups": groups, "shards": shards,
    }


def merge_group(plan: dict[str, Any], group: str, output: Path) -> Path:
    """Require exact per-shard membership, no duplicates, and one run identity."""
    spec = plan["groups"][group]
    provenance = plan["benchmark"]["provenance"]
    expected_identity = {
        "runName": spec["runName"], "ragMode": spec["mode"],
        "benchmarkId": plan["benchmark"]["id"], "benchmarkVersion": plan["benchmark"]["version"],
        "collectionId": provenance["collectionId"], "collectionVersion": provenance["collectionVersion"],
        "collectionObjectsSha256": provenance["objectsSha256"],
        "retrievalPipelineVersion": plan.get("retrievalPipelineVersion", "phase1-planned-stages-20260906"),
        "planningEnabled": True, "auditEnabled": spec["audit"],
    }
    merged: dict[str, dict[str, Any]] = {}
    for shard in plan["shards"]:
        if shard["group"] != group:
            continue
        path = Path(shard["output"])
        if not path.is_file():
            raise CampaignError(f"Missing shard: {shard['id']}")
        seen = set()
        for row in read_jsonl(path):
            row.pop("_sourceLine", None)
            query_id = row.get("queryId")
            if query_id not in shard["queryIds"] or query_id in seen or query_id in merged:
                raise CampaignError(f"Duplicate or wrong-shard query: {query_id}")
            if any(row.get(key) != value for key, value in expected_identity.items()):
                raise CampaignError(f"Mixed runtime/benchmark identity in {shard['id']}: {query_id}")
            merged[query_id] = row
            seen.add(query_id)
        if seen != set(shard["queryIds"]):
            raise CampaignError(f"Incomplete shard: {shard['id']}")
    if set(merged) != set(spec["queryIds"]):
        raise CampaignError(f"Group coverage mismatch: {group}")
    target = output / f"{group}.jsonl"
    temporary = target.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(merged[query_id], ensure_ascii=False, sort_keys=True) + "\n"
                                 for query_id in spec["queryIds"]), encoding="utf-8")
    temporary.replace(target)
    return target


def score(plan: dict[str, Any], benchmark: Any, output: Path, overlay: Path) -> None:
    merged = {group: merge_group(plan, group, output) for group in GROUPS}
    comparisons = {
        "fields-comparison": [("bm25-20260831", PROJECT_ROOT / "artifacts/qa/retrieval-runs/bm25-deterministic-100.jsonl"),
                              ("qwen-20260831", PROJECT_ROOT / "artifacts/qa/retrieval-runs/qwen-rerank-deterministic-100.jsonl"),
                              ("qwen-planned-20260906", merged["hybrid-fields"])],
        "boundary-comparison": [("qwen-20260831", PROJECT_ROOT / "artifacts/qa/retrieval-runs/qwen-boundary-audit-13.jsonl"),
                                ("qwen-planned-20260906", merged["hybrid-boundary"])],
        "open-comparison": [("bm25-agent-20260906", merged["bm25-open"]),
                            ("qwen-agent-20260906", merged["hybrid-open"])],
    }
    for name, runs in comparisons.items():
        report = evaluate_benchmark(benchmark, runs, allow_partial=True, review_overlay=overlay if name == "open-comparison" else None)
        report["campaign"] = {"manifest": str(output / "campaign-plan.json"),
                              "workers": plan["maxConcurrentWorkers"], "latencyScope": plan["latencyScope"],
                              "networkPath": plan.get("networkPath", "local-default"),
                              "culturalRoutingBoundary": plan.get("culturalRoutingBoundary"),
                              "olderBaselineEnvironmentDifferent": name != "open-comparison"}
        write_json(output / f"{name}.json", report)
        markdown = render_markdown(report)
        markdown += ("\n## Runtime comparison boundary\n\n"
                     f"New runs used up to {plan['maxConcurrentWorkers']} isolated processes on this local host. "
                     "Their latency includes concurrent load and is not a production single-request P95. "
                     "The August 31 baseline used an earlier runtime; latency deltas are descriptive, not a controlled speedup claim.\n")
        markdown += ("\nServing uses controlled-origin-v2 derived geography (837 India records corrected from southeast Asia, 2 added to south Asia). "
                     "Frozen raw culture-pack gold remains unchanged. Disagreement with corrected geography must be diagnosed separately from retrieval failure or improvement.\n")
        markdown += f"\nNetwork path: `{plan.get('networkPath', 'local-default')}`. A temporary relay path is not the deployed production path.\n"
        (output / f"{name}.md").write_text(markdown, encoding="utf-8")
        if "reviewedEvaluation" in report:
            missing: dict[tuple[str, str], set[str]] = {}
            for run in report["reviewedEvaluation"]["runs"]:
                for stage, metrics in run["stageMetrics"].items():
                    for row in metrics["unjudgedPairs"]:
                        missing.setdefault((row["queryId"], row["objectId"]), set()).add(f"{run['name']}:{stage}")
            rows = [{"queryId": query_id, "objectId": object_id, "judgmentStatus": "unjudged_pool_candidate",
                     "observedIn": sorted(stages)} for (query_id, object_id), stages in sorted(missing.items())]
            (output / "unjudged-candidate-union.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def execute(plan: dict[str, Any], output: Path, benchmark_dir: Path) -> dict[str, Any]:
    lock, stop = Lock(), Event()
    state: dict[str, Any] = {"status": "running", "startedAt": now(), "workers": plan["maxConcurrentWorkers"], "shards": {}}
    env = {**os.environ, **plan["environment"]}
    if plan.get("networkPath") == "temporary-local-connect-via-existing-server":
        # Both case variants are normalized only in subprocess environment;
        # never change the invoking shell, registry, DNS, or TUN settings.
        for key in ("https_proxy", "no_proxy"):
            env.pop(key, None)
        state["healthGate"] = validate_health_report(Path(plan.get("healthReportPath") or ""), plan)
    (output / "logs").mkdir(exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)

    def update(shard_id: str, payload: dict[str, Any]) -> None:
        with lock:
            state["shards"][shard_id] = payload
            state["updatedAt"] = now()
            write_json(output / "campaign-state.json", state)

    def work(shard: dict[str, Any]) -> dict[str, Any]:
        if stop.is_set() or (output / "stop.flag").exists():
            update(shard["id"], {"status": "not_started"})
            return {"status": "not_started"}
        if runtime_fingerprint() != plan["runtimeSourceSha256"]:
            stop.set()
            update(shard["id"], {"status": "source_changed"})
            return {"status": "source_changed"}
        group = plan["groups"][shard["group"]]
        command = [sys.executable, "-u", str(PROJECT_ROOT / "scripts/run_retrieval_evaluation.py"),
                   "--benchmark-dir", str(benchmark_dir), "--run-name", group["runName"],
                   "--rag-mode", group["mode"], "--partial", "--resume", "--output", shard["output"],
                   "--audit" if group["audit"] else "--query-plan"]
        for query_id in shard["queryIds"]:
            command += ["--query-id", query_id]
        started, started_at = perf_counter(), now()
        log_path = output / "logs" / f"{shard['id']}.log"
        with log_path.open("ab") as log:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            update(shard["id"], {"status": "running", "pid": process.pid, "startedAt": started_at,
                                 "queryCount": len(shard["queryIds"]), "log": str(log_path)})
            return_code = process.wait()
        rows = read_jsonl(Path(shard["output"])) if Path(shard["output"]).is_file() else []
        source_unchanged = runtime_fingerprint() == plan["runtimeSourceSha256"]
        result = {"status": "complete" if return_code == 0 and source_unchanged else "failed",
                  "returnCode": return_code, "pid": process.pid, "startedAt": started_at,
                  "completedAt": now(), "elapsedSeconds": round(perf_counter() - started, 3),
                  "rows": len(rows), "errorQueries": sum(bool(row.get("apiErrors")) for row in rows),
                  "warningQueries": sum(bool(row.get("warnings")) for row in rows),
                  "sourceUnchanged": source_unchanged, "log": str(log_path)}
        if not source_unchanged or any("402" in json.dumps(row.get("apiErrors", [])) for row in rows):
            stop.set()
        update(shard["id"], result)
        print(f"{shard['id']}: {result['status']} rows={len(rows)} errors={result['errorQueries']}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=plan["maxConcurrentWorkers"]) as executor:
        futures = [executor.submit(work, shard) for shard in plan["shards"]]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                stop.set()
                state.setdefault("errors", []).append(f"{type(error).__name__}: {error}")
    state["status"] = "complete" if all(row["status"] == "complete" for row in state["shards"].values()) and len(state["shards"]) == len(plan["shards"]) else "incomplete"
    state["completedAt"] = now()
    write_json(output / "campaign-state.json", state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/qa/rag-optimization-20260906")
    parser.add_argument("--benchmark-dir", type=Path, default=PROJECT_ROOT / "data/qa/retrieval_eval_v1")
    parser.add_argument("--review-overlay", type=Path, default=PROJECT_ROOT / "artifacts/qrel-review/delegated_completion_20260906.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=12)
    parser.add_argument("--campaign-id", default="phase1-20260906")
    parser.add_argument("--network-path", choices=("local-default", "temporary-local-connect-via-existing-server"), default="local-default")
    parser.add_argument("--https-proxy")
    parser.add_argument("--health-check-report", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark(args.benchmark_dir / "manifest.json", args.benchmark_dir / "questions.jsonl", args.benchmark_dir / "qrels.jsonl")
    plan_path = output / "campaign-plan.json"
    if plan_path.exists() and (args.execute or args.merge_only):
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan["runtimeSourceSha256"] != runtime_fingerprint():
            raise CampaignError("Runtime source changed since plan; prepare a new campaign before paid execution")
    else:
        if (output / "campaign-state.json").exists():
            raise CampaignError("Existing campaign state must not be silently replaced")
        plan = make_plan(benchmark, output, shard_size=args.shard_size, workers=args.workers,
                         campaign_id=args.campaign_id, network_path=args.network_path,
                         https_proxy=args.https_proxy, health_report=args.health_check_report)
        write_json(plan_path, plan)
    if args.execute:
        state = execute(plan, output, args.benchmark_dir)
        if state["status"] != "complete":
            print("Campaign incomplete; individual rows remain available, no complete report claimed.")
            return 1
    if args.execute or args.merge_only:
        score(plan, benchmark, output, args.review_overlay)
    print(json.dumps({"plan": str(plan_path), "shards": len(plan["shards"]),
                      "groups": {key: len(value["queryIds"]) for key, value in plan["groups"].items()},
                      "executed": args.execute}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
