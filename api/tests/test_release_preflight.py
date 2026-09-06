from __future__ import annotations

import io
import json
from pathlib import Path

from scripts import release_preflight


def _bundle(tmp_path: Path) -> Path:
    (tmp_path / "server.js").write_text("server", encoding="utf-8")
    (tmp_path / ".next/static").mkdir(parents=True)
    (tmp_path / "public").mkdir()
    (tmp_path / ".next/static/app.js").write_text('fetch("/api/status")', encoding="utf-8")
    return tmp_path


def test_staged_clean_bundle_passes(tmp_path: Path):
    assert all(row["status"] == "pass" for row in release_preflight.inspect_standalone(_bundle(tmp_path)))


def test_staged_env_file_blocks_without_printing_its_contents(tmp_path: Path):
    directory = _bundle(tmp_path)
    (directory / ".env").write_text("DEEPSEEK_API_KEY=do-not-disclose", encoding="utf-8")
    results = release_preflight.inspect_standalone(directory)
    assert next(row for row in results if row["id"] == "standalone_has_no_env_files")["status"] == "fail"
    assert "do-not-disclose" not in json.dumps(results)


def test_loopback_browser_url_is_detected_not_just_an_empty_scan(tmp_path: Path):
    directory = _bundle(tmp_path)
    (directory / ".next/static/app.js").write_text('fetch("http://127.0.0.1:8000/api/exhibitions")', encoding="utf-8")
    row = release_preflight.inspect_standalone(directory)[2]
    assert row["status"] == "fail"
    assert row["matchingFiles"] == 1


def test_server_loopback_defaults_are_not_public_api_urls(tmp_path: Path):
    directory = _bundle(tmp_path)
    (directory / "server.js").write_text('const host="http://localhost:3000"', encoding="utf-8")
    assert release_preflight.inspect_standalone(directory)[2]["status"] == "pass"


def test_missing_static_output_cannot_pass_empty_url_scan(tmp_path: Path):
    assert release_preflight.inspect_standalone(tmp_path)[2]["status"] == "fail"


def test_health_refuses_embedded_credentials_without_request(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not request a credential-bearing URL")

    monkeypatch.setattr(release_preflight, "urlopen", forbidden)
    result = release_preflight.inspect_health("https://user:secret@example.test/health", "hybrid")
    assert result["status"] == "fail"
    assert "secret" not in json.dumps(result)


def test_health_does_not_echo_upstream_body_or_error(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("private proxy credential should not print")

    monkeypatch.setattr(release_preflight, "urlopen", unavailable)
    result = release_preflight.inspect_health("https://example.test/health", "hybrid")
    assert result["errorCode"] == "RuntimeError"
    assert "private" not in json.dumps(result)


def test_health_requires_actual_delivery_mode_and_available_index(monkeypatch):
    payload = {"status": "ok", "secret": "hidden", "retrieval": {"mode": "bm25", "available": True, "collectionId": "test", "private": "hidden"}}
    monkeypatch.setattr(release_preflight, "urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode()))
    result = release_preflight.inspect_health("http://127.0.0.1:8000/health", "hybrid")
    assert result["status"] == "fail"
    assert result["retrieval"]["mode"] == "bm25"
    assert "hidden" not in json.dumps(result)
    payload["retrieval"]["mode"] = "hybrid"
    assert release_preflight.inspect_health("http://127.0.0.1:8000/health", "hybrid")["status"] == "pass"


def test_health_bm25_does_not_require_dense_and_shadow_checks_candidate_indexes(monkeypatch):
    payload = {"status": "ok", "retrieval": {"mode": "bm25", "available": False, "collectionId": "test"}}
    monkeypatch.setattr(release_preflight, "urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode()))
    assert release_preflight.inspect_health("http://127.0.0.1:8000/health", "bm25")["status"] == "pass"
    payload["retrieval"].update(mode="shadow", available=True, candidateAvailable=False)
    assert release_preflight.inspect_health("http://127.0.0.1:8000/health", "shadow")["status"] == "fail"
    payload["retrieval"]["candidateAvailable"] = True
    assert release_preflight.inspect_health("http://127.0.0.1:8000/health", "shadow")["status"] == "pass"


def test_release_holdout_stays_frozen_and_contains_no_object_gold():
    root = Path(__file__).resolve().parents[2]
    holdout = root / "data/qa/visitor_release_holdout_20260906.json"
    assert release_preflight.file_sha256(holdout) == "d7f809ca93bf0f9ad17b2f2568492d7655b8d24607f3e76a159c4a96c9c7cb19"
    payload = json.loads(holdout.read_text(encoding="utf-8"))
    assert len(payload["cases"]) == len({case["id"] for case in payload["cases"]}) == 12
    previous = {json.loads(line)["question"] for line in (root / "data/qa/retrieval_eval_v1/questions.jsonl").read_text(encoding="utf-8").splitlines()}
    assert not ({case["question"] for case in payload["cases"]} & previous)
    assert all("objectIds" not in case and "gold" not in case for case in payload["cases"])
    assert {case["behaviorClass"] for case in payload["cases"]} == {
        "exhibition_intent", "qualified_exhibition_intent", "needs_scope_clarification", "honest_boundary",
    }
