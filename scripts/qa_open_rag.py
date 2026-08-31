"""Live acceptance baseline for semantic hybrid RAG plus LLM evidence audit.

This suite is intentionally separate from deterministic CI: it requires the
frozen dense index, local embedding runtime and a configured DeepSeek key. The
questions live outside collection policies. Once executed they are a fixed
regression set, not an unseen or statistically independent benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "api"))
if hasattr(sys.stdout, "reconfigure"):
    # Windows PowerShell may otherwise use GBK and crash while printing a
    # museum title containing Latin Extended or another Unicode script.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.collections import (  # noqa: E402
    CollectionDataError,
    CollectionRepository,
    _query_plan,
)
from app.config import Settings  # noqa: E402
from app.generator import ExhibitionGenerator  # noqa: E402
from app.models import AgendaInput, AnswerabilityStatus  # noqa: E402
from app import curation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="global_open")
    parser.add_argument(
        "--suite",
        type=Path,
        default=PROJECT_ROOT / "data" / "qa" / "open_vocabulary_blind.json",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Run only the named case id; repeat to select several cases.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print accepted object and cited-evidence details.",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    if not settings.deepseek_api_key:
        print("FAIL            DEEPSEEK_API_KEY is required for the live audit")
        return 2
    repository = CollectionRepository(
        settings.collections_dir,
        default_collection_id=args.collection,
        rag_mode="hybrid",
        dense_index_dir=settings.rag_index_dir,
        embedding_model_cache_dir=settings.rag_model_cache_dir,
        embedding_model=settings.rag_embedding_model,
        dense_top_k=settings.rag_dense_top_k,
        dense_min_score=settings.rag_dense_min_score,
        evidence_min_score=settings.rag_evidence_min_score,
        rrf_k=settings.rag_rrf_k,
        hybrid_max_results=settings.rag_max_results,
    )
    collection = repository.get(args.collection)
    status = repository.retrieval_status(collection)
    if not status.available or status.mode != "hybrid":
        print(f"FAIL            dense index unavailable: {status.reason}")
        return 2

    generator = ExhibitionGenerator(settings, repository)
    payload = json.loads(args.suite.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if args.case_ids:
        requested = set(args.case_ids)
        cases = [case for case in cases if case.get("id") in requested]
        missing = requested - {str(case.get("id")) for case in cases}
        if missing:
            print(f"FAIL            unknown case id(s): {', '.join(sorted(missing))}")
            return 2
    failures: list[str] = []
    observed: Counter[str] = Counter()
    infrastructure_failures: Counter[str] = Counter()
    infrastructure_warnings: Counter[str] = Counter()

    async def run_case(case: dict) -> None:
        started = perf_counter()
        question = str(case["question"])
        plan = _query_plan(question, collection.concept_aliases)
        policy_match = repository.match_question_policy(collection, question)
        if policy_match is not None and policy_match[1] == 1.0:
            failures.append(
                f"{case['id']}: live semantic baseline was replaced by an exact policy"
            )
            return
        route = (
            "title_ambiguous"
            if plan.title_anchor_groups
            else "known_concrete"
            if plan.strict_anchor_groups
            else "open_vocabulary"
        )
        expected_route = str(case.get("retrievalRoute", "open_vocabulary"))
        if route != expected_route:
            failures.append(
                f"{case['id']}: expected retrieval route {expected_route}, got {route}"
            )
            return
        if int(case.get("minCanonicalOrigins", 0)) >= 3 and not plan.cross_cultural:
            failures.append(
                f"{case['id']}: multi-origin acceptance target lacks cross-cultural routing"
            )
            return
        agenda = AgendaInput(
            question=question,
            priorKnowledge="some",
            durationMinutes=5,
            collectionId=collection.id,
        )
        deadline = perf_counter() + settings.rag_retrieval_timeout_seconds
        raw = await generator._search_async(
            agenda,
            collection,
            deadline=deadline,
        )
        outcome = await generator._agentic_retrieve(
            agenda,
            collection,
            raw,
            required_count=5,
            deadline=deadline,
        )
        if outcome.failure_code:
            infrastructure_failures[outcome.failure_code] += 1
        if outcome.warning_code:
            infrastructure_warnings[outcome.warning_code] += 1
        if not outcome.audit_applied:
            failures.append(f"{case['id']}: LLM relevance audit was not applied")
            return
        expected = str(case["expectedAnswerability"])
        observed[str(outcome.answerability)] += 1
        if outcome.answerability != expected:
            failures.append(
                f"{case['id']}: expected {expected}, got {outcome.answerability}"
            )

        accepted = outcome.results
        minimum = int(case.get("minAccepted", 0))
        maximum = int(case.get("maxAccepted", 10**9))
        if len(accepted) < minimum or len(accepted) > maximum:
            failures.append(
                f"{case['id']}: accepted {len(accepted)}, expected {minimum}..{maximum}"
            )
        for result in accepted:
            if "llm_relevance_audit" not in result.retrieval_sources:
                failures.append(f"{case['id']}: {result.obj.id} lacks audit trace")
            if not result.matched_evidence_ids:
                failures.append(f"{case['id']}: {result.obj.id} lacks evidence trace")
                continue
            chunks = {chunk.id: chunk for chunk in result.obj.evidence}
            for evidence_id in result.matched_evidence_ids:
                chunk = chunks.get(evidence_id)
                if chunk is None or chunk.source_kind == "institution_provenance":
                    failures.append(
                        f"{case['id']}: {result.obj.id} cites invalid evidence {evidence_id}"
                    )

        roots = {
            origin
            for result in accepted
            if (origin := generator._canonical_object_origin(result.obj))
        }
        origin_summary = ", ".join(sorted(roots)) or "none"
        required_roots = int(case.get("minCanonicalOrigins", 0))
        if len(roots) < required_roots:
            failures.append(
                f"{case['id']}: expected >= {required_roots} canonical origins, got {len(roots)}"
            )

        final_roots: set[str] = set()
        generation_gate = "no"
        if outcome.answerability in {
            AnswerabilityStatus.SUPPORTED.value,
            AnswerabilityStatus.PARTIALLY_SUPPORTED.value,
        } and len(accepted) >= 5:
            try:
                selected = curation.order_for_narrative(
                    accepted,
                    5,
                    prefer_culture_diversity=plan.cross_cultural,
                )
                selected = generator._ensure_final_cultural_coverage(
                    agenda,
                    selected,
                    accepted,
                    allow_repair=True,
                )
            except CollectionDataError as error:
                generation_gate = f"no:{error.code}"
            else:
                generation_gate = "yes"
                final_roots = {
                    origin
                    for obj in selected
                    if (origin := generator._canonical_object_origin(obj))
                }
                if len(final_roots) < required_roots:
                    failures.append(
                        f"{case['id']}: final room expected >= {required_roots} "
                        f"canonical origins, got {len(final_roots)}"
                    )
        negative_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in case.get("hardNegativeTitlePatterns", [])
        ]
        for result in accepted:
            if any(pattern.search(result.obj.title) for pattern in negative_patterns):
                failures.append(
                    f"{case['id']}: hard negative accepted: {result.obj.id} {result.obj.title}"
                )

        print(
            f"{case['id']:<24} raw={len(raw):>3} accepted={len(accepted):>2} "
            f"gate={outcome.answerability} expansions={len(outcome.expanded_queries)} "
            f"reason={outcome.expansion_reason} "
            f"failure={outcome.failure_code or '-'} "
            f"warning={outcome.warning_code or '-'} "
            f"canGenerate={generation_gate} "
            f"origins={len(roots)}[{origin_summary}] "
            f"finalOrigins={len(final_roots) if final_roots else '-'} "
            f"elapsed={perf_counter() - started:.2f}s"
        )
        if args.details:
            if outcome.expanded_queries:
                print("  queries: " + " | ".join(outcome.expanded_queries))
            for result in accepted:
                cited = set(result.matched_evidence_ids)
                evidence = next(
                    (chunk for chunk in result.obj.evidence if chunk.id in cited),
                    None,
                )
                excerpt = re.sub(r"\s+", " ", evidence.text if evidence else "")[:180]
                print(
                    f"  {result.obj.id} | {result.obj.title} | "
                    f"{result.obj.culture or result.obj.place} | {excerpt}"
                )

    async def run_all() -> None:
        for case in cases:
            await run_case(case)

    asyncio.run(run_all())
    print(f"suite           {args.suite.name} @ {payload.get('version', 'unknown')}")
    print(f"retrieval       {status.mode}; {status.model}; LLM audit enabled")
    print(f"questions       {len(cases)}")
    print(f"answerability   {dict(observed)}")
    print(f"infrastructure  {dict(infrastructure_failures)}")
    print(f"warnings        {dict(infrastructure_warnings)}")
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("ok              semantic hybrid RAG evidence-audit acceptance passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
