from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import candidate_visitor_smoke as visitor

TEST_CANDIDATE = "inquiry-v6-20260906-rc7"

def test_frozen_questions_and_requested_order_are_exact():
    dataset = json.loads((visitor.ROOT / "data/qa/visitor_release_holdout_20260906.json").read_text(encoding="utf-8"))
    questions = {row["id"]: row["question"] for value in dataset.values() if isinstance(value, list)
                 for row in value if isinstance(row, dict) and "question" in row}
    assert visitor.CASE_ORDER == ("release-03-market-day", "release-02-fans")
    assert all(visitor.QUESTIONS[case_id] == questions[case_id] for case_id in visitor.CASE_ORDER)


def test_default_cli_is_local_plan_only(monkeypatch, capsys):
    monkeypatch.setattr(visitor.sys, "argv", ["visitor"])
    monkeypatch.setattr(visitor, "_connect", lambda *a: pytest.fail("default must not connect"))
    assert visitor.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dryRun"] and not report["remoteExecuted"] and report["paidVisitorRuns"] == 0
    assert report['caseOrder'] == list(visitor.CASE_ORDER) and not report['includeDefault']


def test_each_case_has_disjoint_store_cache_and_marker(tmp_path):
    first = visitor.case_paths(tmp_path, visitor.CASE_ORDER[0])
    second = visitor.case_paths(tmp_path, visitor.CASE_ORDER[1])
    assert set(first.values()).isdisjoint(second.values())
    assert all(path.is_relative_to(tmp_path) for path in (*first.values(), *second.values()))


@pytest.mark.parametrize("case_id", ["../store", "any-new-question", "release-04-sleeping"])
def test_unknown_cases_cannot_create_markers(tmp_path, case_id):
    with pytest.raises(visitor.VisitorError, match="unknown_frozen_case"):
        visitor.claim_case(tmp_path, case_id, "a" * 64)
    assert not list(tmp_path.iterdir())


def test_exclusive_marker_prevents_paid_resubmission(tmp_path):
    visitor.claim_case(tmp_path, visitor.CASE_ORDER[0], "a" * 64)
    with pytest.raises(FileExistsError):
        visitor.claim_case(tmp_path, visitor.CASE_ORDER[0], "a" * 64)


@pytest.mark.parametrize("bad", [None, {"processFinalized": False}, {"ownedProcessesStopped": False}, {"manifestSha256": "b" * 64}])
def test_second_case_waits_for_finalized_same_release_predecessor(tmp_path, bad):
    first, second = visitor.CASE_ORDER
    if bad is not None:
        path = visitor.case_paths(tmp_path, first)["receipt"]
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"caseId": first, "manifestSha256": "a" * 64,
                                    "processFinalized": True, "ownedProcessesStopped": True, **bad}))
    with pytest.raises(visitor.VisitorError, match="predecessor_not_finalized"):
        visitor.claim_case(tmp_path, second, "a" * 64)
    assert not visitor.case_paths(tmp_path, second)["marker"].exists()


def test_failed_first_case_can_finish_cleanup_then_second_runs_once(tmp_path):
    first, second = visitor.CASE_ORDER
    paths = visitor.claim_case(tmp_path, first, "a" * 64)
    paths["receipt"].write_text(json.dumps({"caseId": first, "manifestSha256": "a" * 64,
        "processFinalized": True, "ownedProcessesStopped": True, "exitCode": 1}))
    visitor.claim_case(tmp_path, second, "a" * 64)
    with pytest.raises(FileExistsError):
        visitor.claim_case(tmp_path, second, "a" * 64)


def test_output_allowlist_never_forwards_signed_urls_or_provider_errors():
    secret = "https://asset.invalid/file?signature=private-key"
    raw = {"outcome": "completed", "error": secret, "sourceUnchanged": True,
           "exhibition": {"title": secret, "versions": {"provider": "deterministic_fallback"}},
           "verification": {"validatorPassed": True, "coverageLimits": [secret],
                            "citationOwnershipErrors": [{"detail": secret}], "posterStatusAfterObservation": "ready"},
           "posterAssetRead": {"url": secret, "httpStatus": 200, "contentType": "image/png", "bytes": 123},
           "audioStages": [{"kind": "lobby", "status": "completed", "path": secret, "error": secret,
                            "validMp3": True, "bytes": 456, "cacheStatus": "MISS"}],
           "modelStages": [{"stage": "labels:first", "imageObjectIds": ["object:1"], "output": secret}]}
    safe = visitor.safe_result(raw)
    assert "private" not in json.dumps(safe) and "http" not in json.dumps(safe).replace("httpStatus", "")
    assert safe["publicFrameDeterministicFallback"] and safe["validatorPassed"]
    assert safe["allSemanticClaimsVerified"] is False and safe["browserVerified"] is False
    assert safe["citationOwnershipErrorCount"] == 1 and safe["coverageLimitCount"] == 1


def test_non_numeric_media_values_are_not_echoed():
    result = visitor.safe_result({"posterAssetRead": {"bytes": "secret", "httpStatus": float("nan"), "contentType": "secret"},
                                  "audioStages": [{"kind": "secret", "status": "secret", "elapsedSeconds": float("inf")} ]})
    assert "secret" not in json.dumps(result)
    assert result["poster"]["bytes"] is None and result["poster"]["httpStatus"] is None


def client_args(tmp_path, collect=False):
    path = tmp_path / "compatibility.json"
    path.write_text(json.dumps({"passed": True, "releaseName": TEST_CANDIDATE,
        "artifactVerification": {"mode": "manifest_file_equivalent_rebuild", "manifestSha256": "a" * 64},
        "validationHelperSha256": "b" * 64}))
    return SimpleNamespace(compatibility_report=path, candidate=TEST_CANDIDATE, case_id=None, collect_only=collect,
        include_default=False,
        credentials_file=tmp_path / "unread", known_hosts=tmp_path / "known", output_dir=tmp_path / "result")


@pytest.mark.parametrize('include_default', [False, True])
def test_runs_are_sequential_and_never_loop_failed_generation(tmp_path, monkeypatch, include_default):
    args = client_args(tmp_path)
    args.include_default = include_default
    events = []
    client = SimpleNamespace(close=lambda: events.append("closed"))
    monkeypatch.setattr(visitor, "_connect", lambda *a: client)
    monkeypatch.setattr(visitor, "_upload", lambda *a: "c" * 64)
    def command(_client, arguments, **kwargs):
        assert ('--include-default' in arguments) is include_default
        case = arguments[arguments.index("--case-id") + 1]
        events.append(("run", case))
        return 1, b'{"processFinalized":true}'
    def collect(_client, _root, case, _output, **kwargs):
        events.append(("collect", case))
        return {"caseId": case, "processFinalized": True, "ownedProcessesStopped": True,
                "productionFilesUnchanged": True, "exitCode": 1}
    monkeypatch.setattr(visitor, "_command", command)
    monkeypatch.setattr(visitor, "_collect", collect)
    result = visitor.run_client(args)
    expected = [event for case in visitor.protocol_cases(include_default) for event in [('run', case), ('collect', case)]]
    assert events == [*expected, 'closed']
    assert len(result) == (3 if include_default else 2)


def test_collect_only_never_uploads_or_launches(tmp_path, monkeypatch):
    args = client_args(tmp_path, collect=True)
    monkeypatch.setattr(visitor, "_connect", lambda *a: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(visitor, "_upload", lambda *a: pytest.fail("must not upload"))
    monkeypatch.setattr(visitor, "_command", lambda *a, **k: pytest.fail("must not run"))
    monkeypatch.setattr(visitor, "_collect", lambda _a, _b, case, _c, **k: {"caseId": case,
        "processFinalized": True, "ownedProcessesStopped": True, "productionFilesUnchanged": True})
    assert len(visitor.run_client(args)) == 2


@pytest.mark.parametrize("failure", ["uncertain", "cleanup", "production"])
def test_uncertainty_or_cleanup_failure_stops_before_next_paid_case(tmp_path, monkeypatch, failure):
    args = client_args(tmp_path)
    calls = []
    monkeypatch.setattr(visitor, "_connect", lambda *a: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(visitor, "_upload", lambda *a: "c" * 64)
    def command(*args, **kwargs):
        calls.append(1)
        return 1, json.dumps({"processFinalized": failure != "uncertain"}).encode()
    monkeypatch.setattr(visitor, "_command", command)
    monkeypatch.setattr(visitor, "_collect", lambda *a, **k: {"ownedProcessesStopped": failure != "cleanup",
        "productionFilesUnchanged": failure != "production"})
    with pytest.raises(visitor.VisitorError):
        visitor.run_client(args)
    assert calls == [1]


@pytest.mark.parametrize("timeout", [False, True])
@pytest.mark.parametrize('include_default', [False, True])
def test_remote_protocol_isolates_paths_uses_defaults_and_cleans_only_its_process(tmp_path, monkeypatch, timeout, include_default):
    from scripts import stage_release_candidate as stage
    parent = tmp_path / "candidates"
    root = parent / TEST_CANDIDATE
    (root / "scripts").mkdir(parents=True)
    smoke = root / "scripts/smoke_full_generation.py"
    smoke.write_text("fixture-only")
    production = tmp_path / "production"
    production.mkdir()
    (production / ".env").write_text("DEEPSEEK_API_KEY=private-fixture\nSTORE_PATH=store.json\n")
    (production / "store.json").write_text("untouched")
    monkeypatch.setattr(visitor, "PARENT", parent)
    monkeypatch.setattr(visitor, "PRODUCTION", production)
    monkeypatch.setattr(visitor.signal, "signal", lambda *a: None)
    monkeypatch.setattr(visitor, "_load_stage", lambda *a: SimpleNamespace(
        verify_manifest_tree=lambda *a: {"filesVerified": 1450, "mode": "manifest_file_equivalent_rebuild"},
        isolated_environment=stage.isolated_environment, stop_owned_processes=stage.stop_owned_processes))
    case_id = visitor.DEFAULT_CASE if include_default else visitor.CASE_ORDER[0]
    if include_default:
        previous = visitor.case_paths(root, visitor.CASE_ORDER[-1])['receipt']
        previous.parent.mkdir(parents=True)
        previous.write_text(json.dumps({'caseId': visitor.CASE_ORDER[-1], 'manifestSha256': 'a'*64,
                                       'processFinalized': True, 'ownedProcessesStopped': True}))
    calls = []
    class Process:
        pid = 909
        code = None
        terminated = False
        def poll(self): return self.code
        def terminate(self): self.terminated = True; self.code = 0
        def kill(self): self.code = -9
        def wait(self, timeout):
            if timeout == 420 and test_timeout:
                raise subprocess.TimeoutExpired("fixture", timeout)
            self.code = 0
            return 0
    test_timeout = timeout
    process = Process()
    def launch(command, **kwargs):
        calls.append(command)
        paths = visitor.case_paths(root, case_id, include_default=include_default)
        assert kwargs["env"]["STORE_PATH"] == str(paths["output"] / "store.json")
        assert kwargs["env"]["IMAGE_CACHE_DIR"] == str(paths["images"])
        assert kwargs["env"]["RAG_INDEX_DIR"] == str(root / "api/runtime/cache/rag")
        assert "--default-stage-budgets" in command
        assert command[command.index("--optional-assets-wait-seconds") + 1] == "60"
        assert command[command.index("--question") + 1] == visitor.question_for_case(case_id, include_default=include_default)
        paths["output"].mkdir(parents=True)
        (paths["output"] / "result.json").write_text(json.dumps({"outcome": "completed", "sourceUnchanged": True}))
        return process
    monkeypatch.setattr(visitor.subprocess, "Popen", launch)
    receipt = visitor.run_remote_case(root, case_id, "a" * 64, "b" * 64, visitor.sha(smoke),
                                       expected_candidate=TEST_CANDIDATE, include_default=include_default)
    assert receipt["visitorRunCount"] == 1 and receipt["ownedProcessesStopped"]
    assert receipt["productionFilesUnchanged"] and receipt["processFinalized"]
    assert receipt['isHoldoutCase'] is not include_default
    assert receipt['caseKind'] == ('standard_default_smoke' if include_default else 'frozen_holdout')
    assert receipt["ownedProcesses"] == [{"pid": 909, "stopped": True}]
    assert process.terminated is timeout
    assert "private-fixture" not in json.dumps(visitor.public_receipt(receipt))
    with pytest.raises(FileExistsError):
        visitor.run_remote_case(root, case_id, "a" * 64, "b" * 64, visitor.sha(smoke),
                                expected_candidate=TEST_CANDIDATE, include_default=include_default)
    assert len(calls) == 1


@pytest.mark.parametrize("name", [None, "../inquiry-curator", "/opt/demos/inquiry-curator", "inquiry-v6-20260906-rc0",
                                   "inquiry-v6-20260906-rc7/child", "production", "inquiry-v6-20260906-rc7;cmd"])
def test_candidate_names_reject_arbitrary_paths_before_connect(name, tmp_path, monkeypatch):
    args = client_args(tmp_path)
    args.candidate = name
    monkeypatch.setattr(visitor, "_connect", lambda *a: pytest.fail("must validate before SSH"))
    with pytest.raises(visitor.VisitorError, match="invalid_candidate_name"):
        visitor.run_client(args)


def test_compatibility_must_belong_to_exact_candidate(tmp_path, monkeypatch):
    args = client_args(tmp_path)
    args.candidate = "inquiry-v6-20260906-rc8"
    monkeypatch.setattr(visitor, "_connect", lambda *a: pytest.fail("must validate before SSH"))
    with pytest.raises(visitor.VisitorError, match="compatibility_not_passed"):
        visitor.run_client(args)


def test_versioned_local_output_cannot_reuse_previous_results(tmp_path):
    first = visitor.local_output_dir(tmp_path, TEST_CANDIDATE)
    second = visitor.local_output_dir(tmp_path, "inquiry-v6-20260906-rc8")
    assert first != second and first.parent == second.parent == tmp_path


def test_unmarked_existing_state_is_not_overwritten(tmp_path):
    paths = visitor.case_paths(tmp_path, visitor.CASE_ORDER[0])
    paths["output"].mkdir(parents=True)
    (paths["output"] / "result.json").write_text("preserve")
    with pytest.raises(visitor.VisitorError, match="existing_visitor_state"):
        visitor.claim_case(tmp_path, visitor.CASE_ORDER[0], "a" * 64)
    assert (paths["output"] / "result.json").read_text() == "preserve"
    assert not paths["marker"].exists()


def test_remote_candidate_argument_must_match_actual_path(tmp_path, monkeypatch):
    monkeypatch.setattr(visitor, "PARENT", tmp_path)
    monkeypatch.setattr(visitor, "_load_stage", lambda *a: pytest.fail("must reject before helper/model"))
    with pytest.raises(visitor.VisitorError, match="invalid_candidate_root"):
        visitor.run_remote_case(tmp_path / TEST_CANDIDATE, visitor.CASE_ORDER[0], "a" * 64, "b" * 64, "c" * 64,
                                expected_candidate="inquiry-v6-20260906-rc8")


def test_default_version_plan_still_never_connects(monkeypatch, capsys):
    monkeypatch.setattr(visitor.sys, "argv", ["visitor", "--candidate", "inquiry-v6-20260906-rc8"])
    monkeypatch.setattr(visitor, "_connect", lambda *a: pytest.fail("plan must not connect"))
    assert visitor.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["candidate"] == "inquiry-v6-20260906-rc8" and result["dryRun"]
    assert result["localOutputDirectory"].endswith("inquiry-v6-20260906-rc8")


@pytest.mark.parametrize('changed_field', ['candidate', 'manifestSha256'])
def test_collect_rejects_foreign_version_before_local_result_write(tmp_path, changed_field):
    from pathlib import PurePosixPath
    raw = {'caseId': visitor.CASE_ORDER[0], 'processFinalized': True,
           'candidate': TEST_CANDIDATE, 'manifestSha256': 'a'*64}
    raw[changed_field] = 'foreign'
    class SFTP:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def open(self, name, mode): return io.BytesIO(json.dumps(raw).encode())
    client = SimpleNamespace(open_sftp=SFTP)
    with pytest.raises(visitor.VisitorError, match='not_finalized_for_collection'):
        visitor._collect(client, PurePosixPath('/opt/demos/inquiry-curator-candidates')/TEST_CANDIDATE,
                         visitor.CASE_ORDER[0], tmp_path/'out', expected_manifest='a'*64)
    assert not (tmp_path/'out').exists()


def test_default_case_is_exact_standard_entry_not_part_of_frozen_holdout():
    assert set(visitor.QUESTIONS) == set(visitor.CASE_ORDER)
    assert visitor.DEFAULT_CASE == 'default-browse'
    assert visitor.DEFAULT_CASE not in visitor.QUESTIONS
    assert visitor.question_for_case(visitor.DEFAULT_CASE, include_default=True) == '随便带我逛逛'


def test_default_paths_are_disjoint_and_require_opt_in(tmp_path):
    with pytest.raises(visitor.VisitorError, match='explicit_include_default'):
        visitor.case_paths(tmp_path, visitor.DEFAULT_CASE)
    paths = [path for case in visitor.protocol_cases(True)
             for path in visitor.case_paths(tmp_path, case, include_default=True).values()]
    assert len(set(paths)) == 12
    assert all(path.is_relative_to(tmp_path) for path in paths)


@pytest.mark.parametrize('predecessor', [None, {'manifestSha256': 'b'*64}, {'processFinalized': False},
                                        {'ownedProcessesStopped': False}])
def test_default_waits_for_same_manifest_finalized_fans_predecessor(tmp_path, predecessor):
    if predecessor is not None:
        path = visitor.case_paths(tmp_path, visitor.CASE_ORDER[-1])['receipt']
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({'caseId': visitor.CASE_ORDER[-1], 'manifestSha256': 'a'*64,
                                   'processFinalized': True, 'ownedProcessesStopped': True, **predecessor}))
    with pytest.raises(visitor.VisitorError, match='predecessor_not_finalized'):
        visitor.claim_case(tmp_path, visitor.DEFAULT_CASE, 'a'*64, include_default=True)
    assert not visitor.case_paths(tmp_path, visitor.DEFAULT_CASE, include_default=True)['marker'].exists()


def test_selecting_default_without_flag_cannot_connect_or_charge(tmp_path, monkeypatch):
    args = client_args(tmp_path)
    args.case_id = visitor.DEFAULT_CASE
    monkeypatch.setattr(visitor, '_connect', lambda *a: pytest.fail('default selection requires explicit flag'))
    with pytest.raises(visitor.VisitorError, match='explicit_include_default'):
        visitor.run_client(args)


def test_include_default_dryrun_lists_three_but_does_not_execute(monkeypatch, capsys):
    monkeypatch.setattr(visitor.sys, 'argv', ['visitor', '--include-default'])
    monkeypatch.setattr(visitor, '_connect', lambda *a: pytest.fail('dryrun must not connect'))
    assert visitor.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result['caseOrder'] == [*visitor.CASE_ORDER, visitor.DEFAULT_CASE]
    assert result['paidVisitorRuns'] == 0 and result['dryRun']
    assert result['defaultSmokeIsNewHoldout'] is False


def test_collect_only_with_default_never_launches_any_case(tmp_path, monkeypatch):
    args = client_args(tmp_path, collect=True)
    args.include_default = True
    monkeypatch.setattr(visitor, '_connect', lambda *a: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(visitor, '_upload', lambda *a: pytest.fail('collection must not upload'))
    monkeypatch.setattr(visitor, '_command', lambda *a, **k: pytest.fail('collection must not launch'))
    collected = []
    def collect(_a, _b, case, _c, **kwargs):
        assert kwargs['include_default'] is True
        collected.append(case)
        return {'caseId': case, 'processFinalized': True, 'ownedProcessesStopped': True, 'productionFilesUnchanged': True}
    monkeypatch.setattr(visitor, '_collect', collect)
    visitor.run_client(args)
    assert collected == list(visitor.protocol_cases(True))
