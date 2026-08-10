"""Offline regression for the curation pipeline.

Runs entirely against frozen collection data with no cloud model keys, so it is
reproducible and cheap enough to run on every change. It checks two things:

1. the answerability gate still reproduces every curated label, and
2. every visitor profile shape still produces a structurally valid, walkable
   exhibition with its hard constraints intact.

Constraint checks are the point: an exhibition that loses its contrast voice or
lets a metadata-only object carry the core-evidence role is a silent quality
regression, not a crash.

Usage:
    python scripts/qa_regression.py
"""

from __future__ import annotations

import asyncio
import argparse
import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "api"))

from app.collections import CollectionRepository  # noqa: E402
from app.config import Settings  # noqa: E402
from app.generator import ExhibitionGenerator  # noqa: E402
from app.models import (  # noqa: E402
    AgendaInput,
    CuratorialRole,
    EvidenceDepth,
    VisitorMotivation,
    VisitorProfile,
)
from app.validator import validate_exhibition  # noqa: E402

COLLECTIONS_DIR = PROJECT_ROOT / "data" / "collections"


class Failure(Exception):
    pass


def load(collection_id: str) -> tuple[CollectionRepository, ExhibitionGenerator, Path]:
    repository = CollectionRepository(
        COLLECTIONS_DIR, default_collection_id=collection_id
    )
    settings = Settings(
        collections_dir=COLLECTIONS_DIR,
        store_mode="memory",
        # No key: the regression must never depend on a cloud model.
        deepseek_api_key=None,
    )
    generator = ExhibitionGenerator(settings, repository)
    collection_id = repository.get().id
    for candidate in sorted(COLLECTIONS_DIR.glob("*/manifest.json")):
        manifest = json.loads(candidate.read_text(encoding="utf-8"))
        if manifest.get("id") == collection_id:
            return repository, generator, candidate.parent
    raise Failure(f"no directory found for default collection {collection_id}")


def check_answerability(generator: ExhibitionGenerator, collection_dir: Path) -> tuple[int, Counter]:
    path = collection_dir / "regression_questions.json"
    if not path.exists():
        raise Failure(f"missing regression set: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["questions"]

    observed: Counter = Counter()
    mismatches: list[str] = []
    themes: set[str] = set()
    for row in rows:
        result = generator.check_agenda(
            AgendaInput(
                question=row["question"],
                priorKnowledge="some",
                durationMinutes=10,
                excludedTopics=[],
            )
        )
        observed[str(result.status)] += 1
        themes.add(result.exhibition_theme)
        if str(result.status) != row["expectedStatus"]:
            mismatches.append(
                f"  {row['id']}: expected {row['expectedStatus']}, got {result.status}"
            )

    if mismatches:
        raise Failure("answerability labels drifted:\n" + "\n".join(mismatches))
    if len(themes) != len(rows):
        raise Failure(
            f"expected a distinct theme per question, got {len(themes)} for {len(rows)}"
        )
    # A gate that says yes to everything is not a gate.
    if observed["unsupported"] < 3:
        raise Failure("regression set no longer exercises refusal")
    return len(rows), observed


def check_exhibitions(generator: ExhibitionGenerator) -> tuple[int, Counter]:
    """Generate across every profile shape and assert the hard constraints."""
    repository = generator.collections
    collection = repository.get()
    domains = sorted(
        {domain for obj in collection.objects if obj.evidence for domain in obj.themes}
    )
    if not domains:
        raise Failure("collection has no coverage domains assigned")

    generated = 0
    role_counts: Counter = Counter()
    failures: list[str] = []

    # Every duration against every motivation, over the first few domains.
    combinations = list(product(domains[:4], (5, 10, 15), list(VisitorMotivation)))
    for domain_id, minutes, motivation in combinations:
        profile = VisitorProfile(
            curiosityDomainId=domain_id,
            curiosityLabel=domain_id,
            motivation=motivation,
            priorKnowledge="some",
            durationMinutes=minutes,
            excludedTopics=[],
        )
        label = f"{domain_id}/{minutes}min/{motivation.value}"
        try:
            exhibition = asyncio.run(generator.generate_from_profile(profile))
        except Exception as error:  # noqa: BLE001 - collected and reported below
            failures.append(f"  {label}: generation failed: {error}")
            continue

        generated += 1
        validation = validate_exhibition(exhibition)
        if not validation.passed:
            failures.append(f"  {label}: {'; '.join(validation.blocking_issues)}")
            continue

        roles = [str(item.role) for item in exhibition.items]
        role_counts.update(roles)

        if CuratorialRole.COUNTERPOINT.value not in roles:
            failures.append(f"  {label}: lost its contrast voice")
        if len(exhibition.items) != profile.item_count:
            failures.append(
                f"  {label}: expected {profile.item_count} items, got {len(exhibition.items)}"
            )
        if len(exhibition.chapters) != profile.chapter_count:
            failures.append(
                f"  {label}: expected {profile.chapter_count} chapters, got {len(exhibition.chapters)}"
            )
        if not exhibition.epilogue.text:
            failures.append(f"  {label}: no epilogue")

        for item in exhibition.items:
            if (
                str(item.role) == CuratorialRole.CORE_EVIDENCE.value
                and item.object.evidence_depth == EvidenceDepth.THIN.value
            ):
                failures.append(f"  {label}: metadata-only object in core-evidence role")
                break

        # Labels must respect the visitor's attention budget.
        for item in exhibition.items:
            for sentence in item.label_sentences:
                if len(sentence.text) > profile.label_max_chars + 40:
                    failures.append(
                        f"  {label}: label overruns budget "
                        f"({len(sentence.text)} > {profile.label_max_chars})"
                    )
                    break

    if failures:
        raise Failure(
            f"{len(failures)} of {len(combinations)} exhibition checks failed:\n"
            + "\n".join(dict.fromkeys(failures))
        )
    return generated, role_counts


def check_personalisation(generator: ExhibitionGenerator) -> float:
    """Two different profiles must not yield the same object set.

    This is the headline claim of the rebuild, so it gets an actual number
    rather than an assumption.
    """
    collection = generator.collections.get()
    domains = sorted(
        {domain for obj in collection.objects if obj.evidence for domain in obj.themes}
    )
    selections: list[set[str]] = []
    for domain_id in domains[:4]:
        for minutes in (5, 15):
            profile = VisitorProfile(
                curiosityDomainId=domain_id,
                curiosityLabel=domain_id,
                durationMinutes=minutes,
                priorKnowledge="some",
            )
            exhibition = asyncio.run(generator.generate_from_profile(profile))
            selections.append({item.object.id for item in exhibition.items})

    distances: list[float] = []
    for left in range(len(selections)):
        for right in range(left + 1, len(selections)):
            union = selections[left] | selections[right]
            intersection = selections[left] & selections[right]
            distances.append(1 - len(intersection) / len(union) if union else 0.0)

    mean_distance = sum(distances) / len(distances) if distances else 0.0
    if mean_distance < 0.5:
        raise Failure(
            f"personalisation is too weak: mean Jaccard distance {mean_distance:.2f} < 0.50"
        )
    return mean_distance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default="global_open",
        help="Manifest id of the frozen collection to test",
    )
    args = parser.parse_args()
    try:
        repository, generator, collection_dir = load(args.collection)
    except Exception as error:  # noqa: BLE001
        print(f"FAIL  could not load collection data: {error}")
        return 1

    collection = repository.get()
    eligible = [obj for obj in collection.objects if obj.evidence]
    full_depth = sum(
        1 for obj in eligible if obj.evidence_depth == EvidenceDepth.FULL.value
    )
    print(f"collection      {collection.id} @ {collection.version}")
    print(f"objects         {len(collection.objects)} loaded, {len(eligible)} with evidence")
    print(f"evidence depth  {full_depth} full / {len(eligible) - full_depth} thin")

    steps = (
        ("answerability", lambda: check_answerability(generator, collection_dir)),
        ("exhibitions", lambda: check_exhibitions(generator)),
        ("personalisation", lambda: check_personalisation(generator)),
    )

    failed = False
    for name, step in steps:
        try:
            result = step()
        except Failure as error:
            print(f"FAIL  {name}\n{error}")
            failed = True
            continue
        except Exception as error:  # noqa: BLE001
            print(f"FAIL  {name}: unexpected error: {error}")
            failed = True
            continue

        if name == "answerability":
            count, distribution = result
            print(f"ok    answerability   {count} questions, {dict(distribution)}")
        elif name == "exhibitions":
            count, roles = result
            print(f"ok    exhibitions     {count} generated, roles {dict(roles)}")
        else:
            print(f"ok    personalisation mean Jaccard distance {result:.2f}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
