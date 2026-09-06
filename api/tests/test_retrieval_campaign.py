from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.run_retrieval_campaign import CampaignError, campaign_environment, merge_group, validate_health_report


def _fixture(tmp_path: Path):
    plan = {
        "benchmark": {"id": "fixture", "version": "v1", "provenance": {
            "collectionId": "objects", "collectionVersion": "v1", "objectsSha256": "a" * 64}},
        "groups": {"hybrid-fields": {"mode": "hybrid", "audit": False,
                                        "runName": "test", "queryIds": ["q1", "q2"]}},
        "shards": [{"id": f"shard-{number}", "group": "hybrid-fields", "queryIds": [query_id],
                    "output": str(tmp_path / f"shard-{number}.jsonl")}
                   for number, query_id in enumerate(("q1", "q2"), 1)],
    }
    for shard in plan["shards"]:
        row = {"queryId": shard["queryIds"][0], "runName": "test", "ragMode": "hybrid",
               "benchmarkId": "fixture", "benchmarkVersion": "v1", "collectionId": "objects",
               "collectionVersion": "v1", "collectionObjectsSha256": "a" * 64,
               "retrievalPipelineVersion": "phase1-planned-stages-20260906",
               "planningEnabled": True, "auditEnabled": False, "results": [],
               "apiErrors": [{"code": "test_provider_failure"}] if shard["queryIds"] == ["q2"] else []}
        Path(shard["output"]).write_text(json.dumps(row) + "\n", encoding="utf-8")
    return plan


def test_strict_merge_preserves_failure_rows_and_benchmark_order(tmp_path: Path) -> None:
    plan = _fixture(tmp_path)
    plan["shards"].reverse()
    path = merge_group(plan, "hybrid-fields", tmp_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["queryId"] for row in rows] == ["q1", "q2"]
    assert rows[1]["apiErrors"] == [{"code": "test_provider_failure"}]


@pytest.mark.parametrize("mutation, message", [
    ("missing", "Incomplete"), ("wrong_query", "wrong-shard"), ("duplicate", "Duplicate"),
    ("identity", "Mixed runtime"),
])
def test_merge_rejects_incomplete_mixed_or_duplicated_shards(tmp_path: Path, mutation: str, message: str) -> None:
    plan = _fixture(tmp_path)
    path = Path(plan["shards"][1]["output"])
    content = path.read_text(encoding="utf-8")
    row = json.loads(content)
    if mutation == "missing":
        content = ""
    elif mutation == "wrong_query":
        row["queryId"] = "q1"
        content = json.dumps(row) + "\n"
    elif mutation == "duplicate":
        content *= 2
    else:
        row["planningEnabled"] = False
        content = json.dumps(row) + "\n"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CampaignError, match=message):
        merge_group(plan, "hybrid-fields", tmp_path)


def test_relay_environment_is_scoped_and_contains_no_credentials() -> None:
    environment = campaign_environment("temporary-local-connect-via-existing-server", "http://127.0.0.1:18888")
    assert environment["HTTPS_PROXY"] == "http://127.0.0.1:18888"
    assert environment["NO_PROXY"] == "api.deepseek.com,localhost,127.0.0.1"
    for value in ("http://external.test:18888", "http://user:password@127.0.0.1:18888", "socks5://127.0.0.1:18888"):
        with pytest.raises(CampaignError, match="credential-free"):
            campaign_environment("temporary-local-connect-via-existing-server", value)


@pytest.mark.parametrize("mutation", [None, "dense_failure", "stale_source", "stale_time", "wrong_model"])
def test_relay_health_gate_checks_served_mode_source_and_freshness(tmp_path: Path, monkeypatch, mutation: str | None) -> None:
    source = {"fixture.py": "a" * 64}
    monkeypatch.setattr("scripts.run_retrieval_campaign.runtime_fingerprint", lambda: source)
    report = {"pass": True, "generatedAt": datetime.now(timezone.utc).isoformat(),
              "networkPath": "temporary-local-connect-via-existing-server", "httpsProxy": "http://127.0.0.1:18888",
              "runtimeSourceSha256": source,
              "embedding": {"model": "qwen3.7-text-embedding", "dimension": 768},
              "rerank": {"model": "qwen3-rerank"},
              "retrieval": {"actualMode": "hybrid", "denseQueryFailed": False}}
    plan = {"networkPath": report["networkPath"], "environment": {"HTTPS_PROXY": report["httpsProxy"]}, "runtimeSourceSha256": source}
    if mutation == "dense_failure":
        report["retrieval"]["denseQueryFailed"] = True
    elif mutation == "stale_source":
        report["runtimeSourceSha256"] = {}
    elif mutation == "stale_time":
        report["generatedAt"] = "2000-01-01T00:00:00+00:00"
    elif mutation == "wrong_model":
        report["embedding"]["model"] = "other-model"
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    if mutation:
        with pytest.raises(CampaignError):
            validate_health_report(path, plan)
    else:
        assert validate_health_report(path, plan)["pass"] is True
