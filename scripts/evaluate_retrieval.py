"""Compare frozen retrieval runs against pooled object/evidence judgements.

The evaluator is deliberately independent from the production retriever.  It
consumes three frozen benchmark files plus one or more already-produced run
files, so BM25, dense, reranked and agentic variants can be compared without
changing this script or calling a model during scoring.

Canonical benchmark contract
----------------------------

``questions.jsonl`` contains one object per line::

    {"queryId": "q1", "question": "...",
     "expectedAnswerability": "supported",
     "requiredCulturalLegs": ["east_asia", "europe"]}

``qrels.jsonl`` uses one line per judged query/object pair::

    {"queryId": "q1", "objectId": "cma:1", "relevance": 3,
     "supportingEvidenceIds": ["cma:1:description"],
     "culturalLegs": ["east_asia"]}

``manifest.json`` records the frozen collection/question fingerprints and may
set ``relevanceThreshold`` (default 1).  Run JSONL contains one row per query::

    {"queryId": "q1", "answerability": "supported",
     "results": [{"objectId": "cma:1", "evidenceIds": ["..."]}],
     "acceptedResults": [{"objectId": "cma:1", "evidenceIds": ["..."]}],
     "stageLatencyMs": {"embedding": 12.4, "total": 240.0},
     "apiErrors": []}

Ranks default to list order.  ``acceptedResults`` is optional; when omitted,
results explicitly carrying ``"accepted": true`` are used for evidence-support
scoring.  Missing evidence judgements are excluded rather than guessed.

Example::

    python scripts/evaluate_retrieval.py \
      --benchmark-dir data/qa/retrieval_eval_v1 \
      --run bm25=data/qa/runs/bm25.jsonl \
      --run qwen-rerank=data/qa/runs/qwen-rerank.jsonl \
      --output-json artifacts/qa/retrieval-evaluation.json \
      --output-md artifacts/qa/retrieval-evaluation.md
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
ANSWERABILITY_LABELS = {"supported", "partially_supported", "unsupported"}
COMPARISON_METRICS = (
    "recall_at_50",
    "ndcg_at_10",
    "mrr_at_10",
    "success_at_5",
    "required_cultural_leg_coverage_at_50",
    "required_cultural_leg_full_success_at_50",
    "required_cultural_leg_coverage_at_5",
    "evidence_support_rate",
    "hard_false_support_rate",
    "answerability_accuracy",
)
UNSCORED_JUDGEMENT_STATUSES = {
    "pooled_silver_pending_human_review",
    "pending_human_review",
    "unjudged_pool_candidate",
}


class EvaluationError(ValueError):
    """Raised when a benchmark or run violates the frozen evaluation contract."""


@dataclass(frozen=True)
class Benchmark:
    manifest: dict[str, Any]
    questions: dict[str, dict[str, Any]]
    qrels: dict[str, dict[str, dict[str, Any]]]
    manifest_path: Path
    questions_path: Path
    qrels_path: Path
    relevance_threshold: float

    @property
    def qrel_count(self) -> int:
        return sum(len(rows) for rows in self.qrels.values())

    @property
    def scored_qrel_count(self) -> int:
        return sum(
            _is_scored_judgement(row)
            for rows in self.qrels.values()
            for row in rows.values()
        )


def _is_scored_judgement(judgement: Mapping[str, Any]) -> bool:
    status = _identifier(
        judgement, "judgmentStatus", "judgementStatus", "reviewStatus"
    ).casefold()
    return status not in UNSCORED_JUDGEMENT_STATUSES


def _identifier(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _string_list(value: Any, *, field: str, context: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EvaluationError(f"{context}: {field} must be a list of strings")
    return [item.strip() for item in value if item.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read object-per-line JSON with source-aware validation errors."""

    if not path.is_file():
        raise EvaluationError(f"JSONL file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise EvaluationError(
                f"{path}:{line_number}: invalid JSON: {error.msg}"
            ) from error
        if not isinstance(value, dict):
            raise EvaluationError(f"{path}:{line_number}: each row must be an object")
        value["_sourceLine"] = line_number
        rows.append(value)
    return rows


def load_benchmark(
    manifest_path: Path,
    questions_path: Path,
    qrels_path: Path,
) -> Benchmark:
    """Load and validate the frozen benchmark contract."""

    if not manifest_path.is_file():
        raise EvaluationError(f"manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvaluationError(f"{manifest_path}: invalid JSON: {error.msg}") from error
    if not isinstance(manifest, dict):
        raise EvaluationError(f"{manifest_path}: manifest must be an object")

    threshold = manifest.get("relevanceThreshold", 1)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or threshold <= 0:
        raise EvaluationError("manifest relevanceThreshold must be a positive number")

    questions: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(questions_path):
        line = row.pop("_sourceLine")
        query_id = _identifier(row, "queryId", "query_id")
        if not query_id:
            raise EvaluationError(f"{questions_path}:{line}: queryId is required")
        if query_id in questions:
            raise EvaluationError(f"{questions_path}:{line}: duplicate queryId {query_id!r}")
        if not _identifier(row, "question", "query", "text"):
            raise EvaluationError(f"{questions_path}:{line}: question is required")
        expected = _identifier(
            row, "expectedAnswerability", "expected_answerability"
        )
        if expected and expected not in ANSWERABILITY_LABELS:
            raise EvaluationError(
                f"{questions_path}:{line}: invalid expectedAnswerability {expected!r}"
            )
        _string_list(
            row.get("requiredCulturalLegs", row.get("required_cultural_legs")),
            field="requiredCulturalLegs",
            context=f"{questions_path}:{line}",
        )
        questions[query_id] = row

    if not questions:
        raise EvaluationError(f"{questions_path}: benchmark has no questions")

    qrels: dict[str, dict[str, dict[str, Any]]] = {
        query_id: {} for query_id in questions
    }
    for row in read_jsonl(qrels_path):
        line = row.pop("_sourceLine")
        query_id = _identifier(row, "queryId", "query_id")
        object_id = _identifier(row, "objectId", "object_id", "docId", "doc_id")
        context = f"{qrels_path}:{line}"
        if query_id not in questions:
            raise EvaluationError(f"{context}: unknown queryId {query_id!r}")
        if not object_id:
            raise EvaluationError(f"{context}: objectId is required")
        if object_id in qrels[query_id]:
            raise EvaluationError(
                f"{context}: duplicate judgement for {query_id!r}/{object_id!r}"
            )
        relevance = row.get("relevance")
        if (
            not isinstance(relevance, (int, float))
            or isinstance(relevance, bool)
            or not math.isfinite(float(relevance))
            or relevance < 0
        ):
            raise EvaluationError(f"{context}: relevance must be a non-negative number")
        evidence_ids = _string_list(
            row.get("supportingEvidenceIds", row.get("evidenceIds")),
            field="supportingEvidenceIds",
            context=context,
        )
        cultural_legs = _string_list(
            row.get("culturalLegs", row.get("culturePackIds")),
            field="culturalLegs",
            context=context,
        )
        qrels[query_id][object_id] = {
            **row,
            "queryId": query_id,
            "objectId": object_id,
            "relevance": float(relevance),
            "supportingEvidenceIds": evidence_ids,
            "culturalLegs": cultural_legs,
        }

    return Benchmark(
        manifest=manifest,
        questions=questions,
        qrels=qrels,
        manifest_path=manifest_path,
        questions_path=questions_path,
        qrels_path=qrels_path,
        relevance_threshold=float(threshold),
    )


def _normalise_result_list(value: Any, *, context: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise EvaluationError(f"{context}: results must be a list")
    results: list[tuple[int, int, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        item_context = f"{context}: result {index + 1}"
        if not isinstance(raw, dict):
            raise EvaluationError(f"{item_context} must be an object")
        object_id = _identifier(raw, "objectId", "object_id", "docId", "doc_id")
        if not object_id:
            raise EvaluationError(f"{item_context}: objectId is required")
        if object_id in seen:
            raise EvaluationError(f"{context}: duplicate result objectId {object_id!r}")
        seen.add(object_id)
        rank = raw.get("rank", index + 1)
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise EvaluationError(f"{item_context}: rank must be a positive integer")
        evidence_ids = _string_list(
            raw.get("evidenceIds", raw.get("matchedEvidenceIds")),
            field="evidenceIds",
            context=item_context,
        )
        cultural_legs = _string_list(
            raw.get("culturalLegs", raw.get("culturePackIds")),
            field="culturalLegs",
            context=item_context,
        )
        results.append(
            (
                rank,
                index,
                {
                    **raw,
                    "objectId": object_id,
                    "rank": rank,
                    "evidenceIds": evidence_ids,
                    "culturalLegs": cultural_legs,
                },
            )
        )
    results.sort(key=lambda entry: (entry[0], entry[1]))
    return [entry[2] for entry in results]


def _normalise_api_errors(row: Mapping[str, Any], *, context: str) -> list[str]:
    value = row.get("apiErrors", row.get("providerErrors"))
    if value is None and row.get("apiError") is not None:
        value = [row.get("apiError")]
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise EvaluationError(f"{context}: apiErrors must be a list or string")
    errors: list[str] = []
    for error in value:
        if isinstance(error, str):
            errors.append(error)
        elif isinstance(error, dict):
            errors.append(json.dumps(error, ensure_ascii=False, sort_keys=True))
        else:
            raise EvaluationError(f"{context}: apiErrors entries must be strings or objects")
    return errors


def load_run(
    path: Path,
    benchmark: Benchmark,
    *,
    allow_partial: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load a run file and normalise ranking, audit and latency fields."""

    records: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        line = row.pop("_sourceLine")
        context = f"{path}:{line}"
        query_id = _identifier(row, "queryId", "query_id")
        if query_id not in benchmark.questions:
            raise EvaluationError(f"{context}: unknown queryId {query_id!r}")
        if query_id in records:
            raise EvaluationError(f"{context}: duplicate queryId {query_id!r}")

        results = _normalise_result_list(row.get("results"), context=context)
        if "acceptedResults" in row or "accepted_results" in row:
            accepted = _normalise_result_list(
                row.get("acceptedResults", row.get("accepted_results")),
                context=f"{context}: acceptedResults",
            )
        else:
            accepted = [item for item in results if item.get("accepted") is True]
        stage_results = row.get("stageResults", {})
        if not isinstance(stage_results, dict):
            raise EvaluationError(f"{context}: stageResults must be an object")
        normalized_stages = {}
        for stage, values in stage_results.items():
            if not isinstance(stage, str) or not stage.strip():
                raise EvaluationError(f"{context}: stageResults requires non-empty stage names")
            normalized_stages[stage] = _normalise_result_list(
                values, context=f"{context}: stageResults.{stage}"
            )
        if "finalResults" in row:
            normalized_stages["final_selection"] = _normalise_result_list(
                row["finalResults"], context=f"{context}: finalResults"
            )
        if "acceptedResults" in row or "accepted_results" in row or accepted:
            normalized_stages["accepted"] = accepted
        normalized_stages["initial"] = results

        answerability = _identifier(row, "answerability", "observedAnswerability")
        if answerability and answerability not in ANSWERABILITY_LABELS:
            raise EvaluationError(f"{context}: invalid answerability {answerability!r}")

        latencies = row.get("stageLatencyMs", row.get("stageLatenciesMs", {}))
        if latencies is None:
            latencies = {}
        if not isinstance(latencies, dict):
            raise EvaluationError(f"{context}: stageLatencyMs must be an object")
        latency_ms = row.get("latencyMs")
        if latency_ms is not None and "total" not in latencies:
            latencies = {**latencies, "total": latency_ms}
        clean_latencies: dict[str, float] = {}
        for stage, duration in latencies.items():
            if (
                not isinstance(stage, str)
                or not stage.strip()
                or not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not math.isfinite(float(duration))
                or duration < 0
            ):
                raise EvaluationError(
                    f"{context}: stageLatencyMs entries require a stage name and "
                    "non-negative finite number"
                )
            clean_latencies[stage.strip()] = float(duration)

        records[query_id] = {
            **row,
            "queryId": query_id,
            "answerability": answerability,
            "results": results,
            "acceptedResults": accepted,
            "stageResults": normalized_stages,
            "stageLatencyMs": clean_latencies,
            "apiErrors": _normalise_api_errors(row, context=context),
        }

    missing = sorted(set(benchmark.questions) - set(records))
    if missing and not allow_partial:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise EvaluationError(
            f"{path}: run is missing {len(missing)} benchmark queries: {preview}{suffix}"
        )
    return records


def _mean(values: Iterable[float]) -> float | None:
    sequence = list(values)
    return sum(sequence) / len(sequence) if sequence else None


def _nearest_rank_percentile(values: Iterable[float], percentile: float) -> float | None:
    sequence = sorted(values)
    if not sequence:
        return None
    index = max(0, math.ceil(percentile * len(sequence)) - 1)
    return sequence[index]


def _dcg(grades: Sequence[float], k: int) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades[:k], 1)
    )


def _required_legs(question: Mapping[str, Any]) -> set[str]:
    value = question.get(
        "requiredCulturalLegs", question.get("required_cultural_legs", [])
    )
    return {str(item) for item in value}


def _observed_relevant_legs(
    results: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, Mapping[str, Any]],
    threshold: float,
    k: int,
) -> set[str]:
    observed: set[str] = set()
    for result in results[:k]:
        judgement = qrels.get(str(result["objectId"]))
        if (
            not judgement
            or not _is_scored_judgement(judgement)
            or float(judgement["relevance"]) < threshold
        ):
            continue
        legs = judgement.get("culturalLegs") or result.get("culturalLegs") or []
        observed.update(str(leg) for leg in legs)
    return observed


def _scored_relevant_legs(
    qrels: Mapping[str, Mapping[str, Any]], threshold: float
) -> set[str]:
    """Return legs backed by a scored, relevant gold judgement.

    A question is not judgeable merely because it declares required legs.
    Every required leg must first be represented by at least one scored,
    relevant qrel.  Otherwise a pending pooled-silver leg would be converted
    into a false retrieval failure and silently cap the metric below 1.0.
    """

    judged: set[str] = set()
    for judgement in qrels.values():
        if (
            _is_scored_judgement(judgement)
            and float(judgement["relevance"]) >= threshold
        ):
            judged.update(str(leg) for leg in judgement.get("culturalLegs", []))
    return judged


def _cultural_leg_judgment_basis(benchmark: Benchmark) -> dict[str, int]:
    """Summarise whether the frozen qrels can score declared culture legs."""

    scored = 0
    unscored = 0
    single = 0
    multi = 0
    for query_id, question in benchmark.questions.items():
        required = _required_legs(question)
        if not required:
            continue
        judged = _scored_relevant_legs(
            benchmark.qrels[query_id], benchmark.relevance_threshold
        )
        if not required.issubset(judged):
            unscored += 1
            continue
        scored += 1
        if len(required) == 1:
            single += 1
        else:
            multi += 1
    return {
        "scoredQuestionCount": scored,
        "unscoredQuestionCount": unscored,
        "singleLegScoredQuestionCount": single,
        "multiLegScoredQuestionCount": multi,
    }


def _latency_summary(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, list[float]] = {}
    for record in records.values():
        for stage, duration in record["stageLatencyMs"].items():
            by_stage.setdefault(stage, []).append(float(duration))
    return {
        stage: {
            "count": len(values),
            "p50": _nearest_rank_percentile(values, 0.50),
            "p95": _nearest_rank_percentile(values, 0.95),
        }
        for stage, values in sorted(by_stage.items())
    }


def _stage_quality(
    benchmark: Benchmark,
    records: Mapping[str, Mapping[str, Any]],
    stage: str,
) -> dict[str, Any]:
    """Score observed stages and report unjudged results without assigning zero.

    Known-pool recall is bounded by labelled objects, not corpus recall.
    Condensed nDCG removes unjudged objects and is reported separately from the
    historical rank metric. Exhaustive deterministic field gold is closed-world.
    """
    recalls: list[float] = []
    ndcgs: list[float] = []
    coverage: list[float] = []
    all_coverage: list[float] = []
    query_count = result_count = judged_count = relevant_count = 0
    unjudged_queries = cultural_unscored = 0
    evidence_judged = evidence_supported = 0
    unjudged_pairs: list[dict[str, str]] = []
    for query_id, record in records.items():
        if stage not in record["stageResults"]:
            continue
        query_count += 1
        results = record["stageResults"][stage][:50]
        question = benchmark.questions[query_id]
        judgments = benchmark.qrels[query_id]
        closed = question.get("judgmentMode") in {
            "deterministic_field_gold", "deterministic_evidence_boundary_no_relevant_objects"
        }
        relevant_ids = {object_id for object_id, judgment in judgments.items()
                        if _is_scored_judgement(judgment)
                        and judgment["relevance"] >= benchmark.relevance_threshold}
        known = []
        unknown = []
        for result in results:
            object_id = result["objectId"]
            judgment = judgments.get(object_id)
            if (judgment is not None and _is_scored_judgement(judgment)) or closed:
                known.append(result)
            else:
                unknown.append(result)
                unjudged_pairs.append({"queryId": query_id, "objectId": object_id})
            if judgment is None or not _is_scored_judgement(judgment):
                continue
            gold_evidence = set(judgment.get("supportingEvidenceIds", []))
            if "evidenceSupports" not in judgment and not gold_evidence:
                continue
            evidence_judged += 1
            if judgment.get("evidenceSupports") is not False and gold_evidence.intersection(result["evidenceIds"]):
                evidence_supported += 1
        result_count += len(results)
        judged_count += len(known)
        relevant_count += sum(result["objectId"] in relevant_ids for result in known)
        unjudged_queries += bool(unknown)
        if relevant_ids:
            recalls.append(len(relevant_ids.intersection(result["objectId"] for result in results)) / len(relevant_ids))
            ideal = sorted((judgment["relevance"] for judgment in judgments.values()
                            if _is_scored_judgement(judgment)), reverse=True)
            ideal_dcg = _dcg(ideal, 10)
            grades = [judgments.get(result["objectId"], {}).get("relevance", 0) for result in known[:10]]
            # All-unknown results provide no ranking evidence, rather than a
            # fabricated ranking failure. Empty returned lists are real misses.
            if known or not results:
                ndcgs.append(_dcg(grades, 10) / ideal_dcg if ideal_dcg else 0.0)
        required = _required_legs(question)
        if required:
            if not required.issubset(_scored_relevant_legs(judgments, benchmark.relevance_threshold)):
                cultural_unscored += 1
            else:
                observed = _observed_relevant_legs(results, judgments, benchmark.relevance_threshold, 50)
                coverage.append(len(required.intersection(observed)) / len(required))
                all_coverage.append(float(required.issubset(observed)))
    return {
        "queryCount": query_count,
        "known_pool_recall_at_50": _mean(recalls),
        "recall_query_count": len(recalls),
        "condensed_ndcg_at_10": _mean(ndcgs),
        "condensed_ndcg_query_count": len(ndcgs),
        "judged_precision_at_50": relevant_count / judged_count if judged_count else None,
        "result_count_at_50": result_count,
        "judged_result_count_at_50": judged_count,
        "unjudged_result_count_at_50": result_count - judged_count,
        "unjudged_rate_at_50": (result_count - judged_count) / result_count if result_count else None,
        "queries_with_unjudged_results": unjudged_queries,
        "known_required_cultural_leg_coverage_at_50": _mean(coverage),
        "known_required_cultural_leg_full_success_at_50": _mean(all_coverage),
        "cultural_leg_query_count": len(coverage),
        "cultural_leg_unscored_question_count": cultural_unscored,
        "evidence_support_rate": evidence_supported / evidence_judged if evidence_judged else None,
        "evidence_judged_count": evidence_judged,
        "unjudgedPairs": unjudged_pairs,
    }


def _all_stage_quality(benchmark: Benchmark, records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    stages = sorted({stage for record in records.values() for stage in record["stageResults"]})
    return {stage: _stage_quality(benchmark, records, stage) for stage in stages}


def evaluate_run(
    name: str,
    path: Path,
    benchmark: Benchmark,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Calculate ranking, cultural, evidence, safety and operations metrics."""

    records = load_run(path, benchmark, allow_partial=allow_partial)
    threshold = benchmark.relevance_threshold
    recalls: list[float] = []
    ndcgs: list[float] = []
    reciprocal_ranks: list[float] = []
    successes: list[float] = []
    leg_coverage_50: list[float] = []
    leg_full_50: list[float] = []
    leg_coverage_5: list[float] = []
    leg_full_5: list[float] = []
    leg_unscored_questions = 0
    leg_single_scored_questions = 0
    leg_multi_scored_questions = 0
    evidence_supported = 0
    evidence_judged = 0
    hard_false_support = 0
    hard_false_denominator = 0
    answerability_correct = 0
    answerability_judged = 0

    for query_id, question in benchmark.questions.items():
        record = records.get(query_id)
        if record is None:
            continue
        judgements = benchmark.qrels[query_id]
        results = record["results"]
        relevant_ids = {
            object_id
            for object_id, judgement in judgements.items()
            if _is_scored_judgement(judgement)
            and float(judgement["relevance"]) >= threshold
        }
        if relevant_ids:
            retrieved_50 = [str(item["objectId"]) for item in results[:50]]
            recalls.append(len(relevant_ids.intersection(retrieved_50)) / len(relevant_ids))
            grades = [
                float(judgements.get(object_id, {}).get("relevance", 0))
                for object_id in [str(item["objectId"]) for item in results[:10]]
            ]
            ideal = sorted(
                (
                    float(item["relevance"])
                    for item in judgements.values()
                    if _is_scored_judgement(item)
                ),
                reverse=True,
            )
            ideal_dcg = _dcg(ideal, 10)
            ndcgs.append(_dcg(grades, 10) / ideal_dcg if ideal_dcg else 0.0)
            first_relevant = next(
                (
                    rank
                    for rank, item in enumerate(results[:10], 1)
                    if str(item["objectId"]) in relevant_ids
                ),
                None,
            )
            reciprocal_ranks.append(1 / first_relevant if first_relevant else 0.0)
            successes.append(
                float(any(str(item["objectId"]) in relevant_ids for item in results[:5]))
            )

        required_legs = _required_legs(question)
        if required_legs:
            judged_relevant_legs = _scored_relevant_legs(judgements, threshold)
            if not required_legs.issubset(judged_relevant_legs):
                leg_unscored_questions += 1
            else:
                if len(required_legs) == 1:
                    leg_single_scored_questions += 1
                else:
                    leg_multi_scored_questions += 1
                observed_50 = _observed_relevant_legs(
                    results, judgements, threshold, 50
                )
                observed_5 = _observed_relevant_legs(
                    results, judgements, threshold, 5
                )
                leg_coverage_50.append(
                    len(required_legs.intersection(observed_50)) / len(required_legs)
                )
                leg_full_50.append(float(required_legs.issubset(observed_50)))
                leg_coverage_5.append(
                    len(required_legs.intersection(observed_5)) / len(required_legs)
                )
                leg_full_5.append(float(required_legs.issubset(observed_5)))

        for accepted in record["acceptedResults"]:
            object_id = str(accepted["objectId"])
            judgement = judgements.get(object_id)
            if judgement is None or not _is_scored_judgement(judgement):
                continue
            gold_evidence = set(judgement.get("supportingEvidenceIds", []))
            relevance = float(judgement["relevance"])
            explicitly_judged = "evidenceSupports" in judgement
            if not gold_evidence and relevance >= threshold and not explicitly_judged:
                continue
            evidence_judged += 1
            cited = set(accepted.get("evidenceIds", []))
            if gold_evidence and cited.intersection(gold_evidence):
                evidence_supported += 1

        expected = _identifier(
            question, "expectedAnswerability", "expected_answerability"
        )
        if question.get("judgmentMode") == "pooled_silver_pending_human_review":
            expected = ""  # An unreviewed pooling hypothesis is not answerability gold.
        observed = str(record.get("answerability") or "")
        if expected and observed:
            answerability_judged += 1
            answerability_correct += int(expected == observed)
        if expected == "unsupported" and observed:
            hard_false_denominator += 1
            if observed == "supported" or record.get("hardFalseSupport") is True:
                hard_false_support += 1

    api_error_count = sum(len(record["apiErrors"]) for record in records.values())
    api_error_queries = sum(bool(record["apiErrors"]) for record in records.values())
    metrics = {
        "recall_at_50": _mean(recalls),
        "ndcg_at_10": _mean(ndcgs),
        "mrr_at_10": _mean(reciprocal_ranks),
        "success_at_5": _mean(successes),
        "ranking_query_count": len(recalls),
        "required_cultural_leg_coverage_at_50": _mean(leg_coverage_50),
        "required_cultural_leg_full_success_at_50": _mean(leg_full_50),
        "required_cultural_leg_coverage_at_5": _mean(leg_coverage_5),
        "required_cultural_leg_full_success_at_5": _mean(leg_full_5),
        "cultural_leg_query_count": len(leg_coverage_50),
        "cultural_leg_unscored_question_count": leg_unscored_questions,
        "cultural_leg_single_leg_scored_question_count": (
            leg_single_scored_questions
        ),
        "cultural_leg_multi_leg_scored_question_count": leg_multi_scored_questions,
        "evidence_support_rate": (
            evidence_supported / evidence_judged if evidence_judged else None
        ),
        "evidence_supported_count": evidence_supported,
        "evidence_judged_count": evidence_judged,
        "hard_false_support_count": hard_false_support,
        "hard_false_support_rate": (
            hard_false_support / hard_false_denominator
            if hard_false_denominator
            else None
        ),
        "hard_false_support_query_count": hard_false_denominator,
        "answerability_accuracy": (
            answerability_correct / answerability_judged
            if answerability_judged
            else None
        ),
        "answerability_judged_count": answerability_judged,
    }
    return {
        "name": name,
        "path": str(path),
        "evaluatedQueryCount": len(records),
        "benchmarkQueryCount": len(benchmark.questions),
        "metrics": metrics,
        "stageMetrics": _all_stage_quality(benchmark, records),
        "latencyMs": _latency_summary(records),
        "apiErrors": {
            "count": api_error_count,
            "queryCount": api_error_queries,
            "queryRate": api_error_queries / len(records) if records else None,
        },
    }


def _metric_deltas(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float | None]:
    deltas: dict[str, float | None] = {}
    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    for metric in COMPARISON_METRICS:
        before = baseline_metrics.get(metric)
        after = candidate_metrics.get(metric)
        deltas[metric] = (
            float(after) - float(before)
            if isinstance(before, (int, float)) and isinstance(after, (int, float))
            else None
        )
    baseline_total = baseline.get("latencyMs", {}).get("total", {}).get("p95")
    candidate_total = candidate.get("latencyMs", {}).get("total", {}).get("p95")
    deltas["total_latency_p95_ms"] = (
        float(candidate_total) - float(baseline_total)
        if isinstance(baseline_total, (int, float))
        and isinstance(candidate_total, (int, float))
        else None
    )
    deltas["api_error_query_rate"] = (
        float(candidate["apiErrors"]["queryRate"])
        - float(baseline["apiErrors"]["queryRate"])
        if isinstance(candidate["apiErrors"]["queryRate"], (int, float))
        and isinstance(baseline["apiErrors"]["queryRate"], (int, float))
        else None
    )
    return deltas


def evaluate_benchmark(
    benchmark: Benchmark,
    runs: Sequence[tuple[str, Path]],
    *,
    allow_partial: bool = False,
    review_overlay: Path | None = None,
) -> dict[str, Any]:
    if not runs:
        raise EvaluationError("at least one --run NAME=PATH is required")
    names = [name for name, _ in runs]
    if len(names) != len(set(names)):
        raise EvaluationError("run names must be unique")
    evaluated = [
        evaluate_run(name, path, benchmark, allow_partial=allow_partial)
        for name, path in runs
    ]
    baseline = evaluated[0]
    comparisons = [
        {
            "baseline": baseline["name"],
            "run": candidate["name"],
            "metricDeltas": _metric_deltas(baseline, candidate),
        }
        for candidate in evaluated[1:]
    ]
    cultural_leg_basis = _cultural_leg_judgment_basis(benchmark)
    report = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "id": benchmark.manifest.get(
                "benchmarkId", benchmark.manifest.get("id", benchmark.manifest_path.stem)
            ),
            "version": benchmark.manifest.get("version"),
            "manifest": benchmark.manifest,
            "manifestPath": str(benchmark.manifest_path),
            "questionsPath": str(benchmark.questions_path),
            "qrelsPath": str(benchmark.qrels_path),
            "questionCount": len(benchmark.questions),
            "qrelCount": benchmark.qrel_count,
            "scoredQrelCount": benchmark.scored_qrel_count,
            "pendingQrelCount": benchmark.qrel_count - benchmark.scored_qrel_count,
            "relevanceThreshold": benchmark.relevance_threshold,
            "culturalLegJudgmentBasis": cultural_leg_basis,
        },
        "baselineRun": baseline["name"],
        "runs": evaluated,
        "comparisons": comparisons,
    }
    if review_overlay is not None:
        try:
            from retrieval_review_overlay import load_review_overlay
        except ModuleNotFoundError:
            from scripts.retrieval_review_overlay import load_review_overlay
        reviewed, overlay_metadata = load_review_overlay(benchmark, review_overlay)
        reviewed_runs = []
        for name, path in runs:
            records = load_run(path, reviewed, allow_partial=allow_partial)
            compared = agreed = skipped = 0
            for query_id, record in records.items():
                question = reviewed.questions[query_id]
                if "reviewProvenance" not in question or not record["answerability"]:
                    continue
                # An outside-pool accepted object may supply evidence missing
                # from the small review pool; do not call that false support.
                if any(not _is_scored_judgement(reviewed.qrels[query_id].get(item["objectId"],
                    {"judgmentStatus": "unjudged_pool_candidate"})) for item in record["acceptedResults"]):
                    skipped += 1
                    continue
                compared += 1
                agreed += record["answerability"] == question["expectedAnswerability"]
            categories = sorted({str(reviewed.questions[query_id].get("category", "uncategorized"))
                                 for query_id in records})
            reviewed_runs.append({
                "name": name,
                "stageMetrics": _all_stage_quality(reviewed, records),
                "byCategory": {category: _all_stage_quality(reviewed, {
                    query_id: record for query_id, record in records.items()
                    if reviewed.questions[query_id].get("category", "uncategorized") == category
                }) for category in categories},
                "poolAnswerabilityAgreement": {
                    "comparedQuestionCount": compared,
                    "agreementRate": agreed / compared if compared else None,
                    "unjudgedAcceptedQuestionCount": skipped,
                    "scope": "reviewed_candidate_pool_not_full_collection",
                },
            })
        report["reviewedEvaluation"] = {"overlay": overlay_metadata, "runs": reviewed_runs}
    return report


def _format_metric(value: Any, *, signed: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _format_ms(value: Any) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) else "n/a"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact, reviewable companion to the machine-readable report."""

    benchmark = report["benchmark"]
    cultural_leg_basis = benchmark.get("culturalLegJudgmentBasis", {})
    lines = [
        f"# Retrieval evaluation: {benchmark['id']}",
        "",
        f"- Benchmark version: `{benchmark.get('version') or 'unspecified'}`",
        f"- Questions: {benchmark['questionCount']}",
        f"- Qrel judgements: {benchmark['qrelCount']}",
        f"- Scored qrels: {benchmark.get('scoredQrelCount', benchmark['qrelCount'])}",
        f"- Pending pooled candidates: {benchmark.get('pendingQrelCount', 0)}",
        (
            "- Cultural-leg judgement basis: "
            f"{cultural_leg_basis.get('scoredQuestionCount', 0)} scored / "
            f"{cultural_leg_basis.get('unscoredQuestionCount', 0)} unscored "
            "questions; "
            f"{cultural_leg_basis.get('singleLegScoredQuestionCount', 0)} "
            "single-leg / "
            f"{cultural_leg_basis.get('multiLegScoredQuestionCount', 0)} "
            "multi-leg scored."
        ),
        f"- Relevant-grade threshold: {benchmark['relevanceThreshold']:g}",
        f"- Baseline run: `{report['baselineRun']}`",
        "",
        "## Quality and safety",
        "",
        "| Run | Queries | Recall@50 | nDCG@10 | MRR@10 | Success@5 | Cultural legs@50 | All legs@50 | Leg q scored/unscored | Evidence support | Hard false-support | Answerability | Total p50 ms | Total p95 ms | API error queries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        metrics = run["metrics"]
        total = run["latencyMs"].get("total", {})
        false_support = (
            f"{metrics['hard_false_support_count']}/"
            f"{metrics['hard_false_support_query_count']}"
        )
        api_errors = (
            f"{run['apiErrors']['queryCount']}/{run['evaluatedQueryCount']}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run["name"]),
                    str(run["evaluatedQueryCount"]),
                    _format_metric(metrics["recall_at_50"]),
                    _format_metric(metrics["ndcg_at_10"]),
                    _format_metric(metrics["mrr_at_10"]),
                    _format_metric(metrics["success_at_5"]),
                    _format_metric(metrics["required_cultural_leg_coverage_at_50"]),
                    _format_metric(metrics["required_cultural_leg_full_success_at_50"]),
                    (
                        f"{metrics['cultural_leg_query_count']}/"
                        f"{metrics['cultural_leg_unscored_question_count']}"
                    ),
                    _format_metric(metrics["evidence_support_rate"]),
                    false_support,
                    _format_metric(metrics["answerability_accuracy"]),
                    _format_ms(total.get("p50")),
                    _format_ms(total.get("p95")),
                    api_errors,
                ]
            )
            + " |"
        )

    lines.extend(["", "## Stage latency", ""])
    stages = sorted(
        {
            stage
            for run in report["runs"]
            for stage in run.get("latencyMs", {})
        }
    )
    if stages:
        lines.extend(
            [
                "| Run | Stage | Samples | p50 ms | p95 ms |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for run in report["runs"]:
            for stage in stages:
                summary = run["latencyMs"].get(stage)
                if not summary:
                    continue
                lines.append(
                    f"| {run['name']} | {stage} | {summary['count']} | "
                    f"{_format_ms(summary['p50'])} | {_format_ms(summary['p95'])} |"
                )
    else:
        lines.append("No stage latency samples were supplied.")

    if report["comparisons"]:
        lines.extend(
            [
                "",
                "## Delta from baseline",
                "",
                "| Run | Δ Recall@50 | Δ nDCG@10 | Δ MRR@10 | Δ Success@5 | Δ Cultural legs@50 | Δ Evidence support | Δ Hard false-support | Δ Total p95 ms | Δ API error-query rate |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for comparison in report["comparisons"]:
            delta = comparison["metricDeltas"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(comparison["run"]),
                        _format_metric(delta["recall_at_50"], signed=True),
                        _format_metric(delta["ndcg_at_10"], signed=True),
                        _format_metric(delta["mrr_at_10"], signed=True),
                        _format_metric(delta["success_at_5"], signed=True),
                        _format_metric(
                            delta["required_cultural_leg_coverage_at_50"], signed=True
                        ),
                        _format_metric(delta["evidence_support_rate"], signed=True),
                        _format_metric(delta["hard_false_support_rate"], signed=True),
                        _format_ms(delta["total_latency_p95_ms"]),
                        _format_metric(delta["api_error_query_rate"], signed=True),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- Ranking metrics exclude questions with no relevant qrels.",
            "- Qrels marked `pooled_silver_pending_human_review` remain visible in the pool but are excluded from every quality metric.",
            "- Cultural-leg coverage scores only questions whose every required leg has a scored, relevant qrel; pending legs are reported as unscored rather than false failures.",
            "- Evidence support is scored only where a qrel supplies supporting evidence IDs, explicitly judges evidence support, or marks the object irrelevant.",
            "- Hard false-support means an `unsupported` gold question was returned as `supported`; it is not an LLM-as-judge score.",
            "- p50/p95 use nearest-rank percentiles over the supplied per-query stage samples.",
        ]
    )
    for run in report["runs"]:
        metrics = run["metrics"]
        lines.append(
            f"- `{run['name']}` cultural-leg basis: "
            f"{metrics['cultural_leg_query_count']} scored, "
            f"{metrics['cultural_leg_unscored_question_count']} unscored; "
            f"{metrics['cultural_leg_single_leg_scored_question_count']} scored "
            "single-leg and "
            f"{metrics['cultural_leg_multi_leg_scored_question_count']} scored "
            "multi-leg questions."
        )
    if cultural_leg_basis.get("multiLegScoredQuestionCount", 0) == 0:
        lines.append(
            "- Cross-cultural gate not established: there are no scored "
            "multi-leg questions, so this benchmark version cannot support a "
            "cross-cultural coverage >=90% claim."
        )
    lines.extend([
        "", "## Observed retrieval stages", "",
        "Initial metrics above retain the frozen-gold definition. Stages below are observed separately; an absent stage is unmeasured, not an empty final exhibition.",
        "",
        "| Run | Stage | Queries | Known-pool Recall@50 | Judged precision@50 | Condensed nDCG@10 | Unjudged@50 | Known leg coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for run in report["runs"]:
        for stage, metrics in run.get("stageMetrics", {}).items():
            lines.append(_stage_markdown_row(run["name"], stage, metrics))
    reviewed = report.get("reviewedEvaluation")
    if reviewed:
        overlay = reviewed["overlay"]
        lines.extend([
            "", "## Additive review engineering evaluation", "",
            f"- Review export SHA-256: `{overlay['sha256']}`",
            f"- Candidate origins: `{json.dumps(overlay['candidateReviewOrigins'], sort_keys=True)}`; finalized questions: {overlay['finalizedQuestionCount']}.",
            "- Delegated AI reviews are not independent human gold. These scores do not replace the frozen benchmark scores above.",
            "- Unjudged objects are excluded from judged precision and condensed nDCG, and listed for additional pooling. Known-pool recall measures recovery of labelled objects only, not full-corpus recall.",
            "- Pool answerability agreement cannot establish a collection-wide false-supported rate; accepted objects outside the review pool leave that comparison unscored.",
            "",
            "| Run | Stage | Queries | Known-pool Recall@50 | Judged precision@50 | Condensed nDCG@10 | Unjudged@50 | Known leg coverage |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for run in reviewed["runs"]:
            for stage, metrics in run["stageMetrics"].items():
                lines.append(_stage_markdown_row(run["name"], stage, metrics))
    lines.append("")
    return "\n".join(lines)


def _stage_markdown_row(name: str, stage: str, metrics: Mapping[str, Any]) -> str:
    fields = ["known_pool_recall_at_50", "judged_precision_at_50", "condensed_ndcg_at_10",
              "unjudged_rate_at_50", "known_required_cultural_leg_coverage_at_50"]
    values = " | ".join(_format_metric(metrics.get(field)) for field in fields)
    return f"| {name} | {stage} | {metrics['queryCount']} | {values} |"


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise EvaluationError("--run must use NAME=PATH")
    name, raw_path = spec.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise EvaluationError("--run must use a non-empty NAME=PATH")
    return name.strip(), Path(raw_path.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        help="Directory containing manifest.json, questions.jsonl and qrels.jsonl.",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--qrels", type=Path)
    parser.add_argument("--review-overlay", type=Path,
                        help="Explicit additive review export; report engineering scores separately from frozen gold.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named run JSONL; repeat to compare multiple systems. The first is baseline.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow run files that omit benchmark queries; coverage remains explicit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    benchmark_dir = args.benchmark_dir
    manifest_path = args.manifest or (
        benchmark_dir / "manifest.json" if benchmark_dir else None
    )
    questions_path = args.questions or (
        benchmark_dir / "questions.jsonl" if benchmark_dir else None
    )
    qrels_path = args.qrels or (
        benchmark_dir / "qrels.jsonl" if benchmark_dir else None
    )
    if not manifest_path or not questions_path or not qrels_path:
        parser.error(
            "provide --benchmark-dir or all of --manifest, --questions and --qrels"
        )
    try:
        runs = [parse_run_spec(spec) for spec in args.run]
        benchmark = load_benchmark(manifest_path, questions_path, qrels_path)
        report = evaluate_benchmark(
            benchmark,
            runs,
            allow_partial=args.allow_partial,
            review_overlay=args.review_overlay,
        )
    except EvaluationError as error:
        parser.error(str(error))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"json            {args.output_json}")
    print(f"markdown        {args.output_md}")
    print(f"runs            {', '.join(run['name'] for run in report['runs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
