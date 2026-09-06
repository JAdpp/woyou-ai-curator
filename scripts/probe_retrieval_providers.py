"""Run a small, billable provider/retrieval gate before a live campaign.

Writes only credential-free results. Proxy variables, when supplied, affect
this process only. This diagnostic does not alter application configuration.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "api"), str(ROOT / "scripts")]

from app.config import Settings
from app.generator import ExhibitionGenerator
from app.models import AgendaInput
from app.providers.aliyun_text_retrieval import AliyunTextRetrievalProvider
from app.retrieval_runtime import build_collection_repository
from run_retrieval_campaign import runtime_fingerprint
from run_retrieval_evaluation import InMemoryTraceCapture, _retrieval_status


class ProbeTrace(InMemoryTraceCapture):
    def __init__(self) -> None:
        super().__init__()
        self.all_warnings: list[str] = []

    def write(self, **kwargs) -> None:
        self.all_warnings.extend(str(code) for code in kwargs.get("warnings", ()))
        super().write(**kwargs)


async def run_probe(args, report: dict) -> None:
    settings = replace(Settings.from_env(), rag_mode="hybrid")
    provider = AliyunTextRetrievalProvider(
        api_key=settings.rag_embedding_api_key or "",
        api_host=settings.rag_embedding_api_host or "",
        embedding_model=settings.rag_embedding_model,
        embedding_dimension=settings.rag_embedding_dimension,
        query_instruct=settings.rag_embedding_query_instruct,
        rerank_model=settings.rag_rerank_model,
        rerank_instruct=settings.rag_rerank_instruct,
        timeout_seconds=settings.rag_embedding_timeout_seconds,
        max_attempts=1,
    )
    started = perf_counter()
    embedded = provider.embed_queries(["带花卉纹样的陶瓷器"])
    report["embedding"] = {
        "model": embedded.metadata.model, "dimension": embedded.dimension,
        "count": len(embedded.vectors), "attempts": embedded.metadata.attempts,
        "requestId": embedded.metadata.request_id,
        "elapsedSeconds": round(perf_counter() - started, 3),
    }
    started = perf_counter()
    ranked = provider.rerank(
        "带花卉纹样的陶瓷器",
        ["A porcelain vase decorated with peony flowers.",
         "A marble portrait bust of a man."], top_n=2,
    )
    report["rerank"] = {
        "model": ranked.metadata.model, "count": len(ranked.results),
        "firstIndex": ranked.results[0].index,
        "attempts": ranked.metadata.attempts, "requestId": ranked.metadata.request_id,
        "elapsedSeconds": round(perf_counter() - started, 3),
    }
    trace = ProbeTrace()
    repository = build_collection_repository(settings, trace_writer=trace)
    collection = repository.get(settings.default_collection_id)
    generator = ExhibitionGenerator(settings, repository)
    agenda = AgendaInput(question=args.question, prior_knowledge="some",
                         duration_minutes=5, collection_id=collection.id, language="zh")
    started = perf_counter()
    deadline = started + settings.rag_retrieval_timeout_seconds
    initial = await generator.prepare_initial_retrieval(agenda, collection, deadline=deadline)
    initial_status = _retrieval_status(repository, collection)
    outcome = await generator._agentic_retrieve(
        agenda, collection, initial.results, required_count=5, deadline=deadline,
        initial_query_plan=initial.query_plan, planning_attempted=True,
    )
    status = _retrieval_status(repository, collection)
    warnings = sorted(set(trace.all_warnings))
    report["retrieval"] = {
        "question": args.question, "initialCount": len(initial.results),
        "acceptedCount": len(outcome.results), "auditApplied": outcome.audit_applied,
        "answerability": outcome.answerability, "failureCode": outcome.failure_code,
        "warningCode": outcome.warning_code, "coverageGap": outcome.coverage_gap,
        "actualMode": status["servedMode"], "initialStatus": initial_status,
        "finalStatus": status, "warnings": warnings,
        "denseQueryFailed": "dense_query_failed" in warnings,
        "initialDiagnostics": initial.diagnostics,
        "stageObjectIds": {name: [result.obj.id for result in results]
                           for name, results in outcome.stage_results.items()},
        "elapsedSeconds": round(perf_counter() - started, 3),
    }
    report["pass"] = bool(
        len(embedded.vectors) == 1 and embedded.dimension == 768
        and ranked.results[0].index == 0
        and initial_status["servedMode"] == "hybrid" and status["servedMode"] == "hybrid"
        and len(initial.results) > 0 and outcome.audit_applied and not outcome.failure_code
        and not warnings and not outcome.warning_code
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--https-proxy")
    parser.add_argument("--network-path", default="direct")
    parser.add_argument("--question", default="我想看看带花卉纹样的陶瓷器，留意花朵和器物形状怎样相配")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path; previous probe evidence must be retained.")
    if args.https_proxy:
        from urllib.parse import urlsplit
        proxy = urlsplit(args.https_proxy)
        if (proxy.scheme != "http" or proxy.hostname != "127.0.0.1"
                or not proxy.port or proxy.username or proxy.password or proxy.path
                or proxy.query or proxy.fragment):
            parser.error("Only an explicit credential-free loopback HTTP proxy is allowed.")
        os.environ["HTTPS_PROXY"] = args.https_proxy
        os.environ["NO_PROXY"] = "api.deepseek.com,localhost,127.0.0.1"
    report = {"pass": False, "generatedAt": datetime.now(timezone.utc).isoformat(),
              "networkPath": args.network_path, "httpsProxy": args.https_proxy,
              "runtimeSourceSha256": runtime_fingerprint()}
    try:
        asyncio.run(run_probe(args, report))
    except Exception as error:
        # Do not persist upstream response bodies, request data or credentials.
        report["error"] = {"type": type(error).__name__,
                           "code": getattr(error, "code", "probe_failed")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "output": str(args.output),
                      "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
