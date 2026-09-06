"""Exercise only real visitor query planning; no retrieval, images or generation.

Default is a local inventory. --execute calls the configured text model, with
the product's shared planning deadline and a new once-only result directory.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


async def run(args, cases):
    sys.path.insert(0, str(ROOT / "api"))
    from app.config import Settings
    from app.generator import ExhibitionGenerator
    from app.models import AgendaInput
    from app.providers.deepseek import DeepSeekProvider
    from app.query_plan_review import query_plan_review_prompt, query_plan_review_prompt_compact_zh
    from app.retrieval_agent import query_plan_prompt
    defaults = Settings()
    settings = replace(Settings.from_env(), rag_retrieval_timeout_seconds=defaults.rag_retrieval_timeout_seconds,
                       rag_llm_audit_timeout_seconds=defaults.rag_llm_audit_timeout_seconds,
                       rag_planning_timeout_seconds=(args.planning_budget_seconds if args.planning_budget_seconds is not None
                                                     else defaults.rag_planning_timeout_seconds),
                       deepseek_query_review_thinking=args.review_thinking)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    state = {"schemaVersion": "query-planning-probe-v1", "scope": "planning_only_no_catalogue_or_media",
             "startedAt": datetime.now(timezone.utc).isoformat(),
             "questionFileSha256": sha256(args.questions.read_bytes()).hexdigest(),
             "promptSha256": sha256(query_plan_prompt().encode()).hexdigest(),
             "reviewPromptSha256": sha256((query_plan_review_prompt_compact_zh() if args.compact_review
                                          else query_plan_review_prompt()).encode()).hexdigest(),
             "reviewPromptVariant": "compact_zh" if args.compact_review else "product_default",
             "model": settings.deepseek_model, "reviewThinking": settings.deepseek_query_review_thinking,
             "planningBudgetSeconds": settings.rag_planning_timeout_seconds,
             "retrievalBudgetSeconds": settings.rag_retrieval_timeout_seconds,
             "generationJobBudgetSeconds": settings.generation_job_timeout_seconds, "cases": []}
    for case in cases:
        provider = DeepSeekProvider(settings)
        generator = ExhibitionGenerator(settings, SimpleNamespace(), provider)
        record = {"id": case["id"], "question": case["question"], "modelStages": []}
        original = generator._generate_model_json

        async def observe(prompt, payload, *, stage, timeout_seconds):
            if args.compact_review and stage.startswith("retrieval_plan_review"):
                prompt = query_plan_review_prompt_compact_zh()
                if stage.endswith("_repair"):
                    prompt += "\n上次输出未通过 contractFailure 指出的协议检查。请保留原题要求，修正协议错误，返回完整 JSON。"
            started = perf_counter()
            entry = {"stage": stage, "budgetSeconds": timeout_seconds}
            record["modelStages"].append(entry)
            try:
                value = await original(prompt, payload, stage=stage, timeout_seconds=timeout_seconds)
                entry["output"] = value
                return value
            except Exception as error:
                entry.update(errorType=type(error).__name__, errorCode=getattr(error, "code", None))
                raise
            finally:
                entry["elapsedSeconds"] = perf_counter() - started

        async def no_search(*_args, **_kwargs):
            return []

        generator._generate_model_json = observe
        generator._search_async = no_search
        started = perf_counter()
        try:
            outcome = await generator.prepare_initial_retrieval(
                AgendaInput(question=case["question"], priorKnowledge="some", durationMinutes=5),
                SimpleNamespace(concept_aliases={}), deadline=started + settings.rag_retrieval_timeout_seconds)
            record.update(plan=asdict(outcome.query_plan) if outcome.query_plan else None,
                          diagnostics=outcome.diagnostics, status="returned")
        except Exception as error:
            record.update(status="failed", errorType=type(error).__name__, errorCode=getattr(error, "code", None))
        record["elapsedSeconds"] = perf_counter() - started
        state["cases"].append(record)
        (output_dir / "result.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        plan = record.get("plan") or {}
        print(json.dumps({"id": record["id"], "status": record["status"],
                          "elapsedSeconds": round(record["elapsedSeconds"], 2),
                          "setCount": len(plan.get("exhibition_set_requirements", [])),
                          "reviewStatus": record.get("diagnostics", {}).get("planningReview", {}).get("status")},
                         ensure_ascii=False), flush=True)
    state["finishedAt"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "result.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/qa/visitor_release_holdout_20260906.json")
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--compact-review", action="store_true",
                        help="Experiment: use the compact Chinese review prompt, with unchanged product parsing and budgets")
    parser.add_argument("--review-thinking", action="store_true",
                        help="Experiment: enable the product's review-only thinking flag in this process")
    parser.add_argument("--planning-budget-seconds", type=float,
                        help="Experiment: planning sub-budget (2-70s), still inside the unchanged 70s retrieval budget")
    args = parser.parse_args()
    if args.planning_budget_seconds is not None and not 2.0 <= args.planning_budget_seconds <= 70.0:
        parser.error("Planning budget must be between 2 and 70 seconds")
    raw = json.loads(args.questions.read_text(encoding="utf-8"))
    indexed = {case["id"]: case for case in raw["cases"]}
    if len(args.case) != len(set(args.case)) or any(key not in indexed for key in args.case):
        parser.error("Case IDs must be unique and present in the frozen question file")
    cases = [{"id": key, "question": indexed[key]["question"]} for key in args.case]
    if not args.execute:
        print(json.dumps({"execute": False, "caseIds": args.case, "paidCalls": 0,
                          "reviewThinking": args.review_thinking,
                          "planningBudgetSeconds": args.planning_budget_seconds}))
        return
    asyncio.run(run(args, cases))


if __name__ == "__main__":
    main()
