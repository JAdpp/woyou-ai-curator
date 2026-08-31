"""Offline audit-outage safety baseline for diverse visitor questions.

Unlike collection ``regression_questions.json``, this file does not register
question policies. Every case therefore exercises the deterministic query
planner, BM25 safety floor and answerability logic while the audit service is
disabled. Free-form topical questions must fail closed rather than turn lexical
or vector neighbours into evidence; frozen question cards and browse requests
remain available. Use ``qa_open_rag.py`` for the separate live semantic-audit
acceptance gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "api"))

from app.collections import CollectionRepository  # noqa: E402
from app.config import Settings  # noqa: E402
from app.generator import ExhibitionGenerator  # noqa: E402
from app.models import AgendaInput, VisitorProfile  # noqa: E402
from app.validator import validate_exhibition  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="global_open")
    parser.add_argument(
        "--suite",
        type=Path,
        default=PROJECT_ROOT / "data" / "qa" / "diverse_visitor_questions.json",
    )
    args = parser.parse_args()

    payload = json.loads(args.suite.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    collections_dir = PROJECT_ROOT / "data" / "collections"
    repository = CollectionRepository(
        collections_dir,
        default_collection_id=args.collection,
        rag_mode="bm25",
    )
    settings = Settings(
        collections_dir=collections_dir,
        store_mode="memory",
        deepseek_api_key=None,
        rag_mode="bm25",
        rag_llm_audit_enabled=False,
    )
    generator = ExhibitionGenerator(settings, repository)

    failures: list[str] = []
    statuses: Counter[str] = Counter()
    generated = 0
    for case in cases:
        agenda = AgendaInput(
            question=case["question"],
            priorKnowledge="some",
            durationMinutes=5,
        )
        check = generator.check_agenda(agenda)
        status = str(check.status)
        statuses[status] += 1
        expected = case["expectedStatus"]
        if status != expected:
            failures.append(f"{case['id']}: expected {expected}, got {status}")
            continue
        minimum = int(case.get("minMatched", 0))
        if check.coverage.matched_object_count < minimum:
            failures.append(
                f"{case['id']}: expected >= {minimum} matched objects, "
                f"got {check.coverage.matched_object_count}"
            )
        gap_fragment = case.get("gapContains")
        if gap_fragment and not any(
            gap_fragment in gap for gap in check.coverage_gaps
        ):
            failures.append(f"{case['id']}: missing gap text {gap_fragment!r}")
        if expected != "supported":
            if check.can_generate:
                failures.append(f"{case['id']}: non-supported case can generate")
            continue

        try:
            exhibition = asyncio.run(
                generator.generate_from_profile(
                    VisitorProfile(
                        curiosityLabel=case["question"],
                        freeFormQuestion=case["question"],
                        priorKnowledge="some",
                        durationMinutes=5,
                    )
                )
            )
        except Exception as error:  # noqa: BLE001
            failures.append(f"{case['id']}: generation failed: {error}")
            continue
        validation = validate_exhibition(exhibition)
        if not validation.passed:
            failures.append(
                f"{case['id']}: invalid exhibition: "
                + "; ".join(validation.blocking_issues)
            )
            continue
        culture_packs = {
            pack
            for item in exhibition.items
            for pack in item.object.culture_pack_ids
        }
        required_packs = int(case.get("minCulturePacks", 0))
        if len(culture_packs) < required_packs:
            failures.append(
                f"{case['id']}: expected >= {required_packs} culture packs, "
                f"got {len(culture_packs)}"
            )
        generated += 1

    print(f"suite           {args.suite.name} @ {payload.get('version', 'unknown')}")
    print("retrieval       audit-outage safety; BM25 only, semantic audit disabled")
    print(f"questions       {len(cases)} across {len({case['persona'] for case in cases})} personas")
    print(f"statuses        {dict(statuses)}")
    print(f"generated       {generated} supported exhibitions validated")
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("ok              diverse audit-outage safety gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
