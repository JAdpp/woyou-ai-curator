"""Observer protocol checks only: no real provider, image or server calls."""
from __future__ import annotations

import asyncio
import builtins
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import probe_visitor_retrieval as probe


def question_file(tmp_path: Path, *, profile=None) -> Path:
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({
        "status": "frozen_before_first_run",
        "defaultProfile": profile or {"motivation": "explorer", "priorKnowledge": "some",
                                        "durationMinutes": 5, "language": "zh"},
        "cases": [{"id": "new-01", "question": "我想看风景画里的桥", "expectedBehavior": "retrieve_or_explain_gap",
                   "intent": "ORACLE_DO_NOT_SEND", "mustNotDistort": ["GOLD_DO_NOT_SEND"]}],
    }), encoding="utf-8")
    return path


def test_inventory_never_imports_business_modules_executes_or_creates_output(monkeypatch, tmp_path, capsys):
    source = question_file(tmp_path)
    output = tmp_path / "unused-output"
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith(("app", "fastapi", "httpx"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(probe, "execute", lambda *_: pytest.fail("inventory executed a provider"))
    assert probe.main(["--questions", str(source), "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["execute"] is False and result["paidCalls"] == 0
    assert result["questionFileSha256"] == sha256(source.read_bytes()).hexdigest()
    assert not output.exists()


@pytest.mark.parametrize("selection", [["missing"], ["new-01", "new-01"]])
def test_unknown_or_duplicate_ids_rejected_without_execution(tmp_path, selection):
    with pytest.raises(ValueError, match="unknown_or_duplicate"):
        probe.load_cases(question_file(tmp_path), selection)


def test_frozen_hash_guard_and_profile_oracle_boundary(tmp_path):
    path = question_file(tmp_path)
    with pytest.raises(ValueError, match="hash_mismatch"):
        probe.load_cases(path, expected_sha256="0" * 64)
    _, cases = probe.load_cases(path)
    product = probe.product_input(cases[0])
    assert set(product) == {"question", "profile"}
    assert "expectedBehavior" not in json.dumps(product)
    assert "ORACLE" not in json.dumps(product) and "GOLD" not in json.dumps(product)
    question_file(tmp_path, profile={"intent": "ORACLE"})
    with pytest.raises(ValueError, match="non_product_fields"):
        probe.load_cases(path)


def test_once_only_directory_rejected_even_if_previous_attempt_is_incomplete(tmp_path, monkeypatch):
    path = question_file(tmp_path)
    output = tmp_path / "partial-attempt"
    output.mkdir()
    monkeypatch.setattr(probe, "execute", lambda *_: pytest.fail("reran an existing attempt"))
    with pytest.raises(SystemExit) as error:
        probe.main(["--questions", str(path), "--execute", "--output", str(output)])
    assert error.value.code == 2


def test_execute_needs_explicit_new_output(tmp_path):
    with pytest.raises(SystemExit) as error:
        probe.main(["--questions", str(question_file(tmp_path)), "--execute"])
    assert error.value.code == 2


def test_process_overrides_isolate_writes_preserve_proxies_and_enable_vision(monkeypatch, tmp_path):
    from app.config import Settings

    monkeypatch.setenv("HTTPS_PROXY", "http://operator.invalid:9999")
    monkeypatch.setenv("RAG_MODE", "bm25")
    values = probe.process_environment(tmp_path, Settings)
    assert values["RAG_RETRIEVAL_TIMEOUT_SECONDS"] == str(Settings().rag_retrieval_timeout_seconds) == "70.0"
    assert values["RAG_VISUAL_AUDIT_ENABLED"] == "true"
    assert "HTTPS_PROXY" not in values and "RAG_INDEX_DIR" not in values
    for key in ("STORE_PATH", "IMAGE_CACHE_DIR", "RAG_FILTER_INDEX_DIR", "RAG_TRACE_DIR",
                "ALIYUN_IMAGE_OUTPUT_DIR", "ALIYUN_TTS_OUTPUT_DIR"):
        assert Path(values[key]).is_relative_to(tmp_path)
    with probe.environment_overrides(values):
        assert probe.os.environ["RAG_MODE"] == "hybrid"
        assert probe.os.environ["HTTPS_PROXY"] == "http://operator.invalid:9999"
    assert probe.os.environ["RAG_MODE"] == "bm25"


def _map_fixture_collection(client, monkeypatch):
    repository = client.app.state.collections
    original = repository.get
    monkeypatch.setattr(repository, "get", lambda _id=None: original("cma-chinese-art"))


def test_real_public_entry_stops_at_scope_question_without_choosing_rewrite(client, monkeypatch):
    _map_fixture_collection(client, monkeypatch)
    product = {"question": "给我看看那样的东西", "profile": {"language": "zh", "durationMinutes": 5}}
    record = {}
    probe.interview_entry(client, product, complete_generic=True, record=record, save=lambda: None)
    entry = record["interview"]
    assert entry["nextQuestionId"] == "custom_question" and entry["stoppedForVisitorChoice"]
    assert entry["originalQuestionPreserved"] and not entry["generationRequested"]
    assert [snapshot["phase"] for snapshot in entry["snapshots"]] == ["start", "after_opening"]
    assert len(entry["snapshots"][-1]["state"]["transcript"]) == 1
    assert client.app.state.store.list_exhibitions() == []


def test_public_boundary_route_completes_available_profile_steps_not_negotiation(client, monkeypatch):
    _map_fixture_collection(client, monkeypatch)
    product = {"question": "我想知道文物最新的拍卖价格", "profile": {
        "motivation": "recharger", "priorKnowledge": "none", "durationMinutes": 5, "language": "zh"}}
    record = {}
    probe.interview_entry(client, product, complete_generic=True, record=record, save=lambda: None)
    entry = record["interview"]
    phases = [snapshot["phase"] for snapshot in entry["snapshots"]]
    assert "after_motivation" in phases and "after_duration" in phases
    assert "after_negotiation" not in phases and entry["nextQuestionId"] == "negotiation"
    assert entry["originalQuestionPreserved"] and not entry["generationRequested"]
    assert entry["finalProfile"]["motivation"] == "recharger"
    assert client.app.state.store.list_exhibitions() == []


def test_first_answer_observation_does_not_complete_unrequested_interview(client, monkeypatch):
    _map_fixture_collection(client, monkeypatch)
    record = {}
    probe.interview_entry(client, {"question": "我想看画里的桥", "profile": {"language": "zh"}},
                          complete_generic=False, record=record, save=lambda: None)
    assert record["interview"]["nextQuestionId"] == "motivation"
    assert len(record["interview"]["snapshots"]) == 2


def test_retrieval_protocol_shares_deadline_and_preserves_full_owned_evidence():
    from app.collections import SearchResult
    from app.generator import InitialRetrievalOutcome, AgenticRetrievalOutcome
    from app.models import MuseumObject, EvidenceChunk
    from app.retrieval_agent import parse_query_plan

    question = "我想看画里的桥"
    plan = parse_query_plan({"inCollectionScope": True, "catalogueQueries": ["bridge"],
                             "evidenceMode": "visual_observation", "mandatoryPerObjectPredicates": ["Shows a bridge"]},
                            question=question, max_queries=3)
    obj = MuseumObject(id="own", title="Bridge", rights="CC0", imageUrl="https://museum.test/own.jpg",
                       objectUrl="https://museum.test/own", evidence=[EvidenceChunk(
        id="own:text", text="Full institution description, not truncated", sourceUrl="https://museum.test/own",
        sourceTitle="Institution record")])
    result = SearchResult(obj, 2, matched_evidence_ids=("own:text",),
                          set_witnesses=({"requirementId": "s1", "objectId": "own", "checks": []},))
    diagnostics = {"pass1": {"conditions": [{"objectId": "own", "conditionId": "p1", "valid": True},
                                               {"objectId": "other", "conditionId": "p1", "valid": True}]},
                   "visualPreselection": {"actualImageCount": 1}}
    calls = []

    class Generator:
        async def _generate_model_json(self, *args, **kwargs):
            return {"probe": "provider-returned"}

        async def prepare_initial_retrieval(self, agenda, collection, *, deadline):
            calls.append(("prepare", agenda.question, deadline))
            await self._generate_model_json("production prompt", {"question": agenda.question},
                                            stage="retrieval_plan:1", timeout_seconds=8)
            return InitialRetrievalOutcome([result], plan, {"planningReview": {"status": "reviewed"}})

        async def _agentic_retrieve(self, agenda, collection, results, **kwargs):
            calls.append(("audit", agenda.question, kwargs["deadline"]))
            assert kwargs["required_count"] == 5 and kwargs["initial_query_plan"] is plan
            assert kwargs["planning_attempted"] is True and results == [result]
            return AgenticRetrievalOutcome([result], audit_applied=True, answerability="unsupported",
                                           audit_diagnostics=diagnostics)

    generator = Generator()
    original = generator._generate_model_json
    app = SimpleNamespace(state=SimpleNamespace(generator=generator,
        settings=SimpleNamespace(rag_retrieval_timeout_seconds=70), collections=SimpleNamespace(get=lambda _id: object())))
    record = {}
    asyncio.run(probe.retrieval_only(app, {"question": question, "profile": {"durationMinutes": 5}}, record, lambda: None))
    assert calls[0][1] == calls[1][1] == question and calls[0][2] == calls[1][2]
    assert record["initialRetrieval"]["queryPlan"] == asdict(plan)
    assert record["audit"]["answerability"] == "unsupported" and record["status"] == "retrieval_returned"
    candidate = record["audit"]["accepted"][0]
    assert candidate["object"]["evidence"][0]["text"] == obj.evidence[0].text
    assert candidate["setWitnesses"] == list(result.set_witnesses)
    assert len(candidate["conditionChecks"]) == 1 and candidate["conditionChecks"][0]["objectId"] == "own"
    assert record["audit"]["diagnostics"]["visualPreselection"]["actualImageCount"] == 1
    assert record["modelStages"][0]["stage"] == "retrieval_plan:1"
    assert generator._generate_model_json == original


def test_error_serialization_never_includes_provider_message_or_unsafe_code(tmp_path):
    error = RuntimeError("https://provider.test/image?Signature=SECRET_KEY")
    error.code = "provider_network_error"
    assert probe.safe_error(error) == {"type": "RuntimeError", "code": "provider_network_error"}
    error.code = "https://provider.test/?api_key=SECRET_KEY"
    assert probe.safe_error(error)["code"] is None
    path = tmp_path / "safe.json"
    probe.write_json(path, {"output": "echo SECRET_KEY"}, ("SECRET_KEY",))
    assert "SECRET_KEY" not in path.read_text(encoding="utf-8")
