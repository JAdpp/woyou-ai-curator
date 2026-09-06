from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts.evaluate_retrieval import (
    EvaluationError,
    _cultural_leg_judgment_basis,
    evaluate_benchmark,
    load_benchmark,
    load_run,
    main,
    parse_run_spec,
    render_markdown,
)
from scripts.retrieval_review_overlay import load_review_overlay


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _benchmark_fixture(tmp_path: Path):
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "manifest.json").write_text(
        json.dumps(
            {
                "benchmarkId": "retrieval-fixture",
                "version": "v1",
                "collectionVersion": "frozen-objects-v1",
                "relevanceThreshold": 2,
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        benchmark_dir / "questions.jsonl",
        [
            {
                "queryId": "q1",
                "question": "比较两种文化的制作方法",
                "expectedAnswerability": "supported",
                "requiredCulturalLegs": ["east_asia", "europe"],
                "persona": "student",
            },
            {
                "queryId": "q2",
                "question": "用馆藏预测明年市场价格",
                "expectedAnswerability": "unsupported",
                "requiredCulturalLegs": [],
                "persona": "buyer",
            },
            {
                "queryId": "q3",
                "question": "非洲对象的仪式使用",
                "expectedAnswerability": "supported",
                "requiredCulturalLegs": ["africa"],
                "persona": "visitor",
            },
        ],
    )
    _write_jsonl(
        benchmark_dir / "qrels.jsonl",
        [
            {
                "queryId": "q1",
                "objectId": "east-1",
                "relevance": 3,
                "supportingEvidenceIds": ["east-1:e1"],
                "culturalLegs": ["east_asia"],
            },
            {
                "queryId": "q1",
                "objectId": "europe-1",
                "relevance": 2,
                "supportingEvidenceIds": ["europe-1:e1"],
                "culturalLegs": ["europe"],
            },
            {
                "queryId": "q1",
                "objectId": "weak-1",
                "relevance": 1,
                "supportingEvidenceIds": ["weak-1:e1"],
                "culturalLegs": ["africa"],
            },
            {
                "queryId": "q3",
                "objectId": "africa-1",
                "relevance": 3,
                "supportingEvidenceIds": ["africa-1:e1"],
                "culturalLegs": ["africa"],
            },
            {
                "queryId": "q2",
                "objectId": "unjudged-market-object",
                "relevance": 0,
                "supportingEvidenceIds": ["unjudged-market-object:e1"],
                "culturalLegs": ["africa"],
                "judgmentStatus": "pooled_silver_pending_human_review",
            },
        ],
    )
    return load_benchmark(
        benchmark_dir / "manifest.json",
        benchmark_dir / "questions.jsonl",
        benchmark_dir / "qrels.jsonl",
    ), benchmark_dir


def _run_fixture(tmp_path: Path, *, improved: bool) -> Path:
    path = tmp_path / ("improved.jsonl" if improved else "baseline.jsonl")
    if improved:
        rows = [
            {
                "queryId": "q1",
                "answerability": "supported",
                "results": [
                    {
                        "objectId": "east-1",
                        "evidenceIds": ["east-1:e1"],
                        "culturalLegs": ["east_asia"],
                    },
                    {
                        "objectId": "europe-1",
                        "evidenceIds": ["europe-1:e1"],
                        "culturalLegs": ["europe"],
                    },
                    {"objectId": "weak-1", "culturalLegs": ["africa"]},
                ],
                "acceptedResults": [
                    {"objectId": "east-1", "evidenceIds": ["east-1:e1"]},
                    {"objectId": "europe-1", "evidenceIds": ["europe-1:e1"]},
                ],
                "stageLatencyMs": {"embedding": 5, "total": 50},
                "apiErrors": [],
            },
            {
                "queryId": "q2",
                "answerability": "unsupported",
                "results": [],
                "stageLatencyMs": {"embedding": 10, "total": 100},
                "apiErrors": [],
            },
            {
                "queryId": "q3",
                "answerability": "supported",
                "results": [
                    {
                        "objectId": "africa-1",
                        "evidenceIds": ["africa-1:e1"],
                        "culturalLegs": ["africa"],
                    }
                ],
                "acceptedResults": [
                    {"objectId": "africa-1", "evidenceIds": ["africa-1:e1"]}
                ],
                "stageLatencyMs": {"embedding": 15, "total": 150},
                "apiErrors": [],
            },
        ]
    else:
        rows = [
            {
                "queryId": "q1",
                "answerability": "supported",
                "results": [
                    {
                        "objectId": "east-1",
                        "evidenceIds": ["east-1:e1"],
                        "culturalLegs": ["east_asia"],
                    },
                    {
                        "objectId": "europe-1",
                        "evidenceIds": ["wrong:evidence"],
                        "culturalLegs": ["europe"],
                    },
                    {"objectId": "weak-1", "culturalLegs": ["africa"]},
                ],
                "acceptedResults": [
                    {"objectId": "east-1", "evidenceIds": ["east-1:e1"]},
                    {"objectId": "europe-1", "evidenceIds": ["wrong:evidence"]},
                ],
                "stageLatencyMs": {"embedding": 10, "total": 100},
                "apiErrors": [],
            },
            {
                "queryId": "q2",
                "answerability": "supported",
                "results": [
                    {
                        "objectId": "unjudged-market-object",
                        "culturalLegs": ["africa"],
                    }
                ],
                "stageLatencyMs": {"embedding": 20, "total": 200},
                "apiErrors": ["embedding_rate_limited"],
            },
            {
                "queryId": "q3",
                "answerability": "partially_supported",
                "results": [],
                "stageLatencyMs": {"embedding": 30, "total": 300},
                "apiErrors": [],
            },
        ]
    _write_jsonl(path, rows)
    return path


def test_evaluator_compares_ranking_evidence_culture_safety_and_latency(
    tmp_path: Path,
) -> None:
    benchmark, _ = _benchmark_fixture(tmp_path)
    baseline_path = _run_fixture(tmp_path, improved=False)
    improved_path = _run_fixture(tmp_path, improved=True)

    report = evaluate_benchmark(
        benchmark,
        [("bm25", baseline_path), ("qwen-rerank", improved_path)],
    )

    baseline, improved = report["runs"]
    assert report["benchmark"]["scoredQrelCount"] == 4
    assert report["benchmark"]["pendingQrelCount"] == 1
    assert report["benchmark"]["culturalLegJudgmentBasis"] == {
        "scoredQuestionCount": 2,
        "unscoredQuestionCount": 0,
        "singleLegScoredQuestionCount": 1,
        "multiLegScoredQuestionCount": 1,
    }
    assert baseline["metrics"]["recall_at_50"] == pytest.approx(0.5)
    assert baseline["metrics"]["ndcg_at_10"] == pytest.approx(0.5)
    assert baseline["metrics"]["mrr_at_10"] == pytest.approx(0.5)
    assert baseline["metrics"]["success_at_5"] == pytest.approx(0.5)
    assert baseline["metrics"]["ranking_query_count"] == 2
    assert baseline["metrics"]["required_cultural_leg_coverage_at_50"] == pytest.approx(0.5)
    assert baseline["metrics"]["required_cultural_leg_full_success_at_50"] == pytest.approx(0.5)
    assert baseline["metrics"]["cultural_leg_query_count"] == 2
    assert baseline["metrics"]["cultural_leg_unscored_question_count"] == 0
    assert baseline["metrics"]["cultural_leg_single_leg_scored_question_count"] == 1
    assert baseline["metrics"]["cultural_leg_multi_leg_scored_question_count"] == 1
    assert baseline["metrics"]["evidence_support_rate"] == pytest.approx(0.5)
    assert baseline["metrics"]["evidence_judged_count"] == 2
    assert baseline["metrics"]["hard_false_support_count"] == 1
    assert baseline["metrics"]["hard_false_support_rate"] == pytest.approx(1.0)
    assert baseline["metrics"]["answerability_accuracy"] == pytest.approx(1 / 3)
    assert baseline["latencyMs"]["total"] == {
        "count": 3,
        "p50": 200.0,
        "p95": 300.0,
    }
    assert baseline["apiErrors"] == {
        "count": 1,
        "queryCount": 1,
        "queryRate": pytest.approx(1 / 3),
    }

    assert improved["metrics"]["recall_at_50"] == pytest.approx(1.0)
    assert improved["metrics"]["ndcg_at_10"] == pytest.approx(1.0)
    assert improved["metrics"]["evidence_support_rate"] == pytest.approx(1.0)
    assert improved["metrics"]["hard_false_support_count"] == 0
    comparison = report["comparisons"][0]
    assert comparison["baseline"] == "bm25"
    assert comparison["run"] == "qwen-rerank"
    assert comparison["metricDeltas"]["recall_at_50"] == pytest.approx(0.5)
    assert comparison["metricDeltas"]["total_latency_p95_ms"] == pytest.approx(-150)

    markdown = render_markdown(report)
    assert "Recall@50" in markdown
    assert "Evidence support" in markdown
    assert "Hard false-support" in markdown
    assert "qwen-rerank" in markdown


def test_cultural_legs_require_relevant_judged_objects(tmp_path: Path) -> None:
    benchmark, _ = _benchmark_fixture(tmp_path)
    run_path = tmp_path / "irrelevant-leg.jsonl"
    _write_jsonl(
        run_path,
        [
            {
                "queryId": "q1",
                "answerability": "supported",
                "results": [
                    {"objectId": "east-1", "culturalLegs": ["east_asia"]},
                    # This weak qrel is below relevanceThreshold=2 and cannot
                    # fake coverage of a required leg.
                    {"objectId": "weak-1", "culturalLegs": ["europe"]},
                ],
            },
            {"queryId": "q2", "answerability": "unsupported", "results": []},
            {
                "queryId": "q3",
                "answerability": "supported",
                "results": [{"objectId": "africa-1"}],
            },
        ],
    )

    result = evaluate_benchmark(benchmark, [("candidate", run_path)])["runs"][0]

    # q1 covers one of two required legs; q3 covers its only leg.
    assert result["metrics"]["required_cultural_leg_coverage_at_50"] == pytest.approx(0.75)
    assert result["metrics"]["required_cultural_leg_full_success_at_50"] == pytest.approx(0.5)


def test_pending_required_legs_are_unscored_not_false_failures(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "pending-leg-benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "manifest.json").write_text(
        json.dumps(
            {
                "benchmarkId": "pending-leg-fixture",
                "version": "v1",
                "relevanceThreshold": 2,
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        benchmark_dir / "questions.jsonl",
        [
            {
                "queryId": "single-gold",
                "question": "定位一条已审定的单文化腿",
                "requiredCulturalLegs": ["africa"],
            },
            {
                "queryId": "multi-pending",
                "question": "比较两个尚待复核的文化腿",
                "requiredCulturalLegs": ["east_asia", "europe"],
            },
        ],
    )
    _write_jsonl(
        benchmark_dir / "qrels.jsonl",
        [
            {
                "queryId": "single-gold",
                "objectId": "africa-gold",
                "relevance": 3,
                "culturalLegs": ["africa"],
            },
            {
                "queryId": "multi-pending",
                "objectId": "east-pool",
                "relevance": 0,
                "culturalLegs": ["east_asia"],
                "judgmentStatus": "pooled_silver_pending_human_review",
            },
            {
                "queryId": "multi-pending",
                "objectId": "europe-pool",
                "relevance": 0,
                "culturalLegs": ["europe"],
                "judgmentStatus": "pooled_silver_pending_human_review",
            },
        ],
    )
    benchmark = load_benchmark(
        benchmark_dir / "manifest.json",
        benchmark_dir / "questions.jsonl",
        benchmark_dir / "qrels.jsonl",
    )
    run_path = tmp_path / "pending-leg-run.jsonl"
    _write_jsonl(
        run_path,
        [
            {
                "queryId": "single-gold",
                "results": [
                    {"objectId": "africa-gold", "culturalLegs": ["africa"]}
                ],
            },
            {
                "queryId": "multi-pending",
                "results": [],
            },
        ],
    )

    report = evaluate_benchmark(benchmark, [("candidate", run_path)])
    metrics = report["runs"][0]["metrics"]

    # The pending multi-leg question is not a zero in the coverage mean.
    assert metrics["required_cultural_leg_coverage_at_50"] == pytest.approx(1.0)
    assert metrics["required_cultural_leg_full_success_at_50"] == pytest.approx(1.0)
    assert metrics["cultural_leg_query_count"] == 1
    assert metrics["cultural_leg_unscored_question_count"] == 1
    assert metrics["cultural_leg_single_leg_scored_question_count"] == 1
    assert metrics["cultural_leg_multi_leg_scored_question_count"] == 0
    assert report["benchmark"]["culturalLegJudgmentBasis"] == {
        "scoredQuestionCount": 1,
        "unscoredQuestionCount": 1,
        "singleLegScoredQuestionCount": 1,
        "multiLegScoredQuestionCount": 0,
    }

    markdown = render_markdown(report)
    assert "1 scored, 1 unscored" in markdown
    assert "cannot support a cross-cultural coverage >=90% claim" in markdown


def test_run_contract_is_strict_by_default_and_can_report_partial_coverage(
    tmp_path: Path,
) -> None:
    benchmark, _ = _benchmark_fixture(tmp_path)
    partial = tmp_path / "partial.jsonl"
    _write_jsonl(
        partial,
        [{"queryId": "q1", "answerability": "supported", "results": []}],
    )

    with pytest.raises(EvaluationError, match="missing 2 benchmark queries"):
        load_run(partial, benchmark)

    records = load_run(partial, benchmark, allow_partial=True)
    assert list(records) == ["q1"]


def test_cli_writes_machine_readable_json_and_markdown(tmp_path: Path) -> None:
    _, benchmark_dir = _benchmark_fixture(tmp_path)
    baseline_path = _run_fixture(tmp_path, improved=False)
    improved_path = _run_fixture(tmp_path, improved=True)
    output_json = tmp_path / "artifacts" / "report.json"
    output_md = tmp_path / "artifacts" / "report.md"

    assert (
        main(
            [
                "--benchmark-dir",
                str(benchmark_dir),
                "--run",
                f"bm25={baseline_path}",
                "--run",
                f"hybrid={improved_path}",
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ]
        )
        == 0
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schemaVersion"] == 1
    assert payload["benchmark"]["questionCount"] == 3
    assert [run["name"] for run in payload["runs"]] == ["bm25", "hybrid"]
    assert "## Delta from baseline" in output_md.read_text(encoding="utf-8")


@pytest.mark.parametrize("spec", ["missing-separator", "=path.jsonl", "name="])
def test_run_spec_rejects_ambiguous_values(spec: str) -> None:
    with pytest.raises(EvaluationError, match="NAME=PATH"):
        parse_run_spec(spec)


def test_frozen_v1_reports_current_cross_cultural_judgment_gap() -> None:
    project_root = Path(__file__).resolve().parents[2]
    benchmark_dir = project_root / "data" / "qa" / "retrieval_eval_v1"
    benchmark = load_benchmark(
        benchmark_dir / "manifest.json",
        benchmark_dir / "questions.jsonl",
        benchmark_dir / "qrels.jsonl",
    )

    # V1 deliberately has no human semantic gold yet. Its deterministic leg
    # judgements are all single-leg, while 35 multi-leg pooled questions remain
    # unscored. This must not be reported as retrieval failure or >=90% proof.
    assert _cultural_leg_judgment_basis(benchmark) == {
        "scoredQuestionCount": 45,
        "unscoredQuestionCount": 35,
        "singleLegScoredQuestionCount": 45,
        "multiLegScoredQuestionCount": 0,
    }


def _overlay_fixture(tmp_path: Path):
    _, directory = _benchmark_fixture(tmp_path)
    questions_path = directory / "questions.jsonl"
    questions = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines()]
    for question in questions[:2]:
        question["judgmentMode"] = "pooled_silver_pending_human_review"
        question["category"] = "open_theme"
    _write_jsonl(questions_path, questions)
    qrels_path = directory / "qrels.jsonl"
    qrels = [json.loads(line) for line in qrels_path.read_text(encoding="utf-8").splitlines()]
    for qrel in qrels:
        if qrel["queryId"] == "q1":
            qrel["relevance"] = 0
            qrel["judgmentStatus"] = "pooled_silver_pending_human_review"
    _write_jsonl(qrels_path, qrels)
    objects_path = directory / "objects.json"
    objects_path.write_text(json.dumps([
        {"id": qrel["objectId"], "evidence": [{"id": item} for item in qrel["supportingEvidenceIds"]]}
        for qrel in qrels
    ]), encoding="utf-8")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "files": {kind: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for kind, path in (("questions", questions_path), ("qrels", qrels_path))},
        "provenance": {"objectsFile": "objects.json",
                       "objectsSha256": hashlib.sha256(objects_path.read_bytes()).hexdigest()},
    })
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    revisions = []
    for object_id, relevance, origin in (("east-1", 3, "human"), ("europe-1", 2, "delegated_ai"), ("weak-1", 0, "delegated_ai")):
        revisions.append({
            "kind": "candidate", "queryId": "q1", "objectId": object_id,
            "relevance": relevance, "evidenceVerdict": "supports" if relevance else "not_applicable",
            "supportingEvidenceIds": [f"{object_id}:e1"] if relevance else [],
            "reviewOrigin": origin, "reviewerId": "local-owner", "revision": 1,
            "acceptedSuggestionId": "suggestion-1" if origin == "delegated_ai" else None,
        })
    revisions.append({"kind": "finalization", "queryId": "q1", "objectId": None,
                      "expectedAnswerability": "supported", "reviewOrigin": "delegated_ai",
                      "reviewerId": "local-owner", "revision": 1})
    overlay = tmp_path / "review.json"
    overlay.write_text(json.dumps({"snapshot": {
        "benchmark": {"id": "retrieval-fixture", "version": "v1"}, "revisions": revisions,
    }}), encoding="utf-8")
    benchmark = load_benchmark(manifest_path, questions_path, qrels_path)
    return benchmark, overlay


def test_review_overlay_is_additive_keeps_origins_and_does_not_score_unknown_as_zero(tmp_path: Path) -> None:
    benchmark, overlay = _overlay_fixture(tmp_path)
    frozen = benchmark.qrels_path.read_bytes()
    run = tmp_path / "review-run.jsonl"
    _write_jsonl(run, [{
        "queryId": "q1", "answerability": "supported",
        "results": [{"objectId": "outside-the-reviewed-pool"}],
        "stageResults": {"structured_retrieval": [{"objectId": "east-1"}]},
        "acceptedResults": [{"objectId": "east-1", "evidenceIds": ["east-1:e1"]},
                            {"objectId": "europe-1", "evidenceIds": ["europe-1:e1"]}],
        "finalResults": [{"objectId": "east-1", "evidenceIds": ["east-1:e1"]}],
    }])
    baseline = evaluate_benchmark(benchmark, [("hybrid", run)], allow_partial=True)
    report = evaluate_benchmark(benchmark, [("hybrid", run)], allow_partial=True, review_overlay=overlay)
    assert report["runs"][0]["metrics"] == baseline["runs"][0]["metrics"]
    assert benchmark.qrels_path.read_bytes() == frozen
    assert benchmark.qrels["q1"]["east-1"]["relevance"] == 0
    overlay_report = report["reviewedEvaluation"]
    assert overlay_report["overlay"]["independentHumanGold"] is False
    assert overlay_report["overlay"]["candidateReviewOrigins"] == {"human": 1, "delegated_ai": 2}
    stages = overlay_report["runs"][0]["stageMetrics"]
    assert stages["initial"]["unjudged_rate_at_50"] == 1
    assert stages["initial"]["judged_precision_at_50"] is None
    assert stages["initial"]["condensed_ndcg_at_10"] is None
    assert stages["initial"]["known_pool_recall_at_50"] == 0
    assert stages["accepted"]["known_required_cultural_leg_full_success_at_50"] == 1
    assert stages["final_selection"]["known_required_cultural_leg_coverage_at_50"] == 0.5
    assert "not independent human gold" in render_markdown(report)
    reviewed, _ = load_review_overlay(benchmark, overlay)
    assert "expectedAnswerability" not in reviewed.questions["q2"]
    assert reviewed.qrels["q1"]["europe-1"]["reviewProvenance"]["acceptedSuggestionId"] == "suggestion-1"


@pytest.mark.parametrize("mutation, message", [
    ("hash", "SHA-256"), ("foreign_evidence", "must belong"),
    ("origin", "explicit human/delegated_ai"), ("incomplete", "all frozen candidates"),
])
def test_review_overlay_rejects_mismatched_or_unattributed_judgments(tmp_path: Path, mutation: str, message: str) -> None:
    benchmark, overlay = _overlay_fixture(tmp_path)
    if mutation == "hash":
        benchmark.qrels_path.write_text("\n", encoding="utf-8")
    else:
        payload = json.loads(overlay.read_text(encoding="utf-8"))
        rows = payload["snapshot"]["revisions"]
        if mutation == "foreign_evidence":
            rows[0]["supportingEvidenceIds"] = ["europe-1:e1"]
        elif mutation == "origin":
            rows[0].pop("reviewOrigin")
        else:
            rows.pop(0)
        overlay.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvaluationError, match=message):
        load_review_overlay(benchmark, overlay)


def test_unobserved_final_stage_is_not_a_failed_exhibition(tmp_path: Path) -> None:
    benchmark, _ = _benchmark_fixture(tmp_path)
    run = _run_fixture(tmp_path, improved=True)
    report = evaluate_benchmark(benchmark, [("hybrid", run)])
    assert "final_selection" not in report["runs"][0]["stageMetrics"]
    assert report["runs"][0]["stageMetrics"]["accepted"]["queryCount"] == 2


def test_pool_answerability_and_evidence_respect_unjudged_and_negative_reviews(tmp_path: Path) -> None:
    benchmark, overlay = _overlay_fixture(tmp_path)
    payload = json.loads(overlay.read_text(encoding="utf-8"))
    payload["snapshot"]["revisions"][0]["evidenceVerdict"] = "contradicts"
    overlay.write_text(json.dumps(payload), encoding="utf-8")
    run = tmp_path / "unknown-accepted.jsonl"
    _write_jsonl(run, [{"queryId": "q1", "answerability": "supported", "results": [],
                      "acceptedResults": [{"objectId": "new-evidence-object"},
                                          {"objectId": "east-1", "evidenceIds": ["east-1:e1"]}]}])
    report = evaluate_benchmark(benchmark, [("hybrid", run)], allow_partial=True, review_overlay=overlay)
    reviewed = report["reviewedEvaluation"]["runs"][0]
    assert reviewed["poolAnswerabilityAgreement"]["agreementRate"] is None
    assert reviewed["poolAnswerabilityAgreement"]["unjudgedAcceptedQuestionCount"] == 1
    assert reviewed["stageMetrics"]["accepted"]["evidence_support_rate"] == 0


def test_revision_after_finalization_leaves_pool_answerability_unscored(tmp_path: Path) -> None:
    benchmark, overlay = _overlay_fixture(tmp_path)
    payload = json.loads(overlay.read_text(encoding="utf-8"))
    rows = payload["snapshot"]["revisions"]
    rows[-1]["createdAt"] = "2026-09-06T10:00:00+08:00"
    rows[0]["createdAt"] = "2026-09-06T11:00:00+08:00"
    overlay.write_text(json.dumps(payload), encoding="utf-8")
    reviewed, metadata = load_review_overlay(benchmark, overlay)
    assert "expectedAnswerability" not in reviewed.questions["q1"]
    assert metadata["staleFinalizationQuestionIds"] == ["q1"]
