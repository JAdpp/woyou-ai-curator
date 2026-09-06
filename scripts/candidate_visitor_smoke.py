"""Version-bound visitor protocol; default is local plan only and two cases.

Explicit --execute runs the frozen cases sequentially, once each. Recovery is
--collect-only; it never resends a generation request. No production cutover.
--include-default explicitly appends one standard default-entry smoke, not a
new holdout case. Without that option no third request is admitted or sent.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path("/opt/demos/inquiry-curator-candidates")
PRODUCTION = Path("/opt/demos/inquiry-curator")
CASE_ORDER = ("release-03-market-day", "release-02-fans")
QUESTIONS = {
    "release-03-market-day": "以前的人赶集是什么样？给我看一些市场、摊贩或街头买卖的画面，我想看看卖东西的人和买东西的人怎样挤在一起。",
    "release-02-fans": "中国、日本和欧洲的扇子能放在一起看吗？我想比较展开后的轮廓、扇骨和画面布局，只看确实是扇子的作品，不要扇形盘子。",
}
DEFAULT_CASE = "default-browse"
DEFAULT_QUESTION = "随便带我逛逛"
SCHEMA = "isolated-candidate-visitor-v2"


class VisitorError(RuntimeError):
    pass


def candidate_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"inquiry-v6-[0-9]{8}-rc[1-9][0-9]{0,3}", value):
        raise VisitorError("invalid_candidate_name")
    return value


def local_output_dir(base: Path | None, name: str) -> Path:
    # A caller-supplied base still gets a version namespace. RC7 cannot silently
    # reuse RC6's local results even if --output-dir is copied from old commands.
    return (base or ROOT / "artifacts/qa/server-candidates") / candidate_name(name)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_cases(include_default: bool = False) -> tuple[str, ...]:
    return (*CASE_ORDER, DEFAULT_CASE) if include_default else CASE_ORDER


def question_for_case(case_id: str, *, include_default: bool = False) -> str:
    if case_id == DEFAULT_CASE:
        if not include_default:
            raise VisitorError("default_smoke_requires_explicit_include_default")
        return DEFAULT_QUESTION
    if case_id not in QUESTIONS:
        raise VisitorError("unknown_frozen_case")
    return QUESTIONS[case_id]


def selected_cases(case_id: str | None, include_default: bool) -> tuple[str, ...]:
    if case_id is None:
        return protocol_cases(include_default)
    question_for_case(case_id, include_default=include_default)
    return (case_id,)


def case_paths(root: Path, case_id: str, *, include_default: bool = False) -> dict[str, Path]:
    question_for_case(case_id, include_default=include_default)
    state = root / "candidate-state/visitor-smokes"
    return {"output": state / case_id, "images": state / f"{case_id}-images",
            "marker": state / f"{case_id}.once.json", "receipt": state / f"{case_id}.receipt.json"}


def claim_case(root: Path, case_id: str, manifest_sha: str, *, include_default: bool = False) -> dict[str, Path]:
    paths = case_paths(root, case_id, include_default=include_default)
    if paths["marker"].exists() or paths["marker"].is_symlink():
        raise FileExistsError("visitor_once_marker_already_exists")
    for path in paths.values():
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise VisitorError("visitor_state_path_escape")
    if any(paths[key].exists() for key in ("output", "images", "receipt")):
        raise VisitorError("existing_visitor_state_without_marker")
    order = protocol_cases(include_default)
    index = order.index(case_id)
    if index:
        previous_case = order[index-1]
        predecessor = case_paths(root, previous_case, include_default=include_default)["receipt"]
        if not predecessor.is_file():
            raise VisitorError("predecessor_not_finalized")
        receipt = json.loads(predecessor.read_text(encoding="utf-8"))
        if (receipt.get("caseId") != previous_case or receipt.get("manifestSha256") != manifest_sha
                or receipt.get("processFinalized") is not True or receipt.get("ownedProcessesStopped") is not True):
            raise VisitorError("predecessor_not_finalized")
    paths["marker"].parent.mkdir(parents=True, exist_ok=True)
    with paths["marker"].open("x", encoding="utf-8") as handle:
        json.dump({"caseId": case_id, "manifestSha256": manifest_sha,
                   "caseKind": "standard_default_smoke" if case_id == DEFAULT_CASE else "frozen_holdout",
                   "claimedAt": datetime.now(timezone.utc).isoformat(), "allowedVisitorRuns": 1}, handle)
    return paths


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def safe_result(result: dict) -> dict:
    """Only explicit scalar fields; no raw text, paths, URLs or provider errors."""
    verification = result.get("verification") or {}
    exhibition = result.get("exhibition") or {}
    poster = result.get("posterAssetRead") or {}
    outcomes = {"completed", "completed_with_validation_issues", "request_rejected", "observer_deadline", "generation_failed", "smoke_exception"}
    provider = (exhibition.get("versions") or {}).get("provider")
    brief = (exhibition.get("curatorialBrief") or {}).get("status")
    audio = []
    for row in result.get("audioStages") or []:
        audio.append({"kind": row.get("kind") if row.get("kind") in {"lobby", "chapter", "artwork", "epilogue"} else "unknown",
                      "status": row.get("status") if row.get("status") in {"completed", "failed", "running", "cancelled_at_observer_teardown"} else "unknown",
                      "cacheStatus": row.get("cacheStatus") if row.get("cacheStatus") in {"HIT", "MISS"} else None,
                      "validMp3": row.get("validMp3") is True, "bytes": _number(row.get("bytes")),
                      "elapsedSeconds": _number(row.get("elapsedSeconds"))})
    return {
        "outcome": result.get("outcome") if result.get("outcome") in outcomes else "unknown",
        "generationSeconds": _number(result.get("generationElapsedSeconds")),
        "totalObservationSeconds": _number(result.get("elapsedSeconds")),
        "sourceUnchanged": result.get("sourceUnchanged") is True,
        "itemCount": _number(verification.get("itemCount")),
        "labelSentenceCount": _number(verification.get("labelSentenceCount")),
        "citationOwnershipErrorCount": len(verification.get("citationOwnershipErrors") or []),
        "validatorPassed": verification.get("validatorPassed") is True,
        "publicFrameDeterministicFallback": provider == "deterministic_fallback",
        "briefDeterministic": brief == "deterministic",
        "coverageLimitCount": len(verification.get("coverageLimits") or []),
        "labelObjectsWithImageInputs": len({item for row in result.get("modelStages") or []
                                             if str(row.get("stage", "")).startswith("labels:")
                                             for item in row.get("imageObjectIds") or []}),
        "poster": {"ready": verification.get("posterStatusAfterObservation") == "ready",
                   "httpStatus": _number(poster.get("httpStatus")),
                   "contentType": poster.get("contentType") if poster.get("contentType") in {"image/png", "image/jpeg"} else None,
                   "bytes": _number(poster.get("bytes"))},
        "audioStages": audio,
        "browserVerified": False, "allSemanticClaimsVerified": False,
    }


def _load_stage(root: Path, helper_sha: str):
    if not re.fullmatch(r"[0-9a-f]{64}", helper_sha):
        raise VisitorError("invalid_helper_hash")
    path = root / f"candidate-validator-{helper_sha[:16]}.py"
    if path.is_symlink() or sha(path) != helper_sha:
        raise VisitorError("stage_helper_hash_mismatch")
    spec = importlib.util.spec_from_file_location("candidate_isolation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_remote_case(candidate: Path, case_id: str, manifest_sha: str, stage_sha: str, smoke_sha: str,
                    *, expected_candidate: str, include_default: bool = False) -> dict:
    question = question_for_case(case_id, include_default=include_default)
    root = candidate.resolve()
    if (root.parent != PARENT or root.name != candidate_name(expected_candidate)
            or candidate != root or candidate.is_symlink()):
        raise VisitorError("invalid_candidate_root")
    stage = _load_stage(root, stage_sha)
    verified = stage.verify_manifest_tree(root, manifest_sha)
    smoke = root / "scripts/smoke_full_generation.py"
    if smoke.is_symlink() or not re.fullmatch(r"[0-9a-f]{64}", smoke_sha) or sha(smoke) != smoke_sha:
        raise VisitorError("smoke_helper_hash_mismatch")
    from dotenv import dotenv_values
    original = dotenv_values(PRODUCTION / ".env")
    env = stage.isolated_environment(root, original)
    paths = claim_case(root, case_id, manifest_sha, include_default=include_default)
    env.update({"STORE_PATH": str(paths["output"] / "store.json"),
                "IMAGE_CACHE_DIR": str(paths["images"]),
                "RAG_TRACE_DIR": str(paths["output"] / "retrieval-traces"),
                "ALIYUN_IMAGE_OUTPUT_DIR": str(paths["output"] / "generated-posters"),
                "ALIYUN_TTS_OUTPUT_DIR": str(paths["output"] / "generated-audio")})
    configured = Path(original.get("STORE_PATH") or "api/runtime/store.json")
    production_store = configured if configured.is_absolute() else PRODUCTION / configured
    protected = {"environment": PRODUCTION / ".env", "store": production_store}
    before = {key: sha(path) if path.is_file() else None for key, path in protected.items()}
    receipt = {"schemaVersion": SCHEMA, "caseId": case_id, "candidate": root.name,
               "caseKind": "standard_default_smoke" if case_id == DEFAULT_CASE else "frozen_holdout",
               "isHoldoutCase": case_id != DEFAULT_CASE,
               "manifestSha256": manifest_sha, "packageFilesVerified": verified["filesVerified"],
               "artifactMode": verified["mode"], "visitorRunCount": 0, "processFinalized": False,
               "startedAt": datetime.now(timezone.utc).isoformat(), "productionHashesBefore": before,
               "candidateOnly": True, "serviceConfigurationModified": False, "browserVerified": False}
    process = None
    def interrupted(_signal, _frame):
        raise VisitorError("visitor_process_interrupted")
    for number in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGHUP", signal.SIGTERM)):
        signal.signal(number, interrupted)
    try:
        command = [str(PRODUCTION / ".venv/bin/python"), "-B", str(smoke),
                   "--output-dir", str(paths["output"]), "--image-cache-dir", str(paths["images"]),
                   "--default-stage-budgets", "--optional-assets-wait-seconds", "60", "--question", question]
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        receipt["visitorRunCount"] = 1
        receipt["ownedPid"] = process.pid
        receipt["exitCode"] = process.wait(timeout=420)
    except Exception as error:
        receipt["errorType"] = type(error).__name__
    finally:
        stopped = stage.stop_owned_processes([process] if process else [])
        receipt["ownedProcesses"] = stopped
        receipt["ownedProcessesStopped"] = all(row["stopped"] for row in stopped)
        after = {key: sha(path) if path.is_file() else None for key, path in protected.items()}
        receipt["productionHashesAfter"] = after
        receipt["productionFilesUnchanged"] = before == after
        receipt["processFinalized"] = True
        receipt["finishedAt"] = datetime.now(timezone.utc).isoformat()
        result_path = paths["output"] / "result.json"
        if result_path.is_file():
            receipt["resultSha256"] = sha(result_path)
            try:
                receipt["summary"] = safe_result(json.loads(result_path.read_text(encoding="utf-8")))
            except Exception:
                receipt["resultParseFailed"] = True
        with paths["receipt"].open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2)
    return receipt


def public_receipt(receipt: dict) -> dict:
    return {key: receipt.get(key) for key in ("schemaVersion", "caseId", "caseKind", "isHoldoutCase", "candidate", "manifestSha256",
            "artifactMode", "packageFilesVerified", "visitorRunCount", "exitCode", "processFinalized",
            "ownedProcessesStopped", "productionFilesUnchanged", "browserVerified", "summary")}


def _connect(credentials: Path, known_hosts: Path):
    import paramiko
    from scripts.probe_server_provider_tls import PROJECT_SERVER, load_project_credentials
    logging.getLogger("paramiko").addHandler(logging.NullHandler())
    logging.getLogger("paramiko").propagate = False
    client = paramiko.SSHClient()
    client.load_host_keys(str(known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    username, password = load_project_credentials(credentials)
    client.connect(PROJECT_SERVER, username=username, password=password, allow_agent=False,
                   look_for_keys=False, timeout=10, banner_timeout=20, auth_timeout=20)
    password = ""
    client.get_transport().set_keepalive(15)
    return client


def _command(client, arguments, timeout=30):
    _, stdout, stderr = client.exec_command(" ".join(shlex.quote(str(value)) for value in arguments), timeout=timeout)
    body = stdout.read(1024 * 1024)
    stderr.read(65536)
    return stdout.channel.recv_exit_status(), body


def _upload(client, source: Path, target: PurePosixPath) -> str:
    digest = sha(source)
    with client.open_sftp() as sftp:
        try:
            sftp.lstat(str(target))
        except FileNotFoundError:
            with sftp.open(str(target), "wx") as handle:
                handle.write(source.read_bytes())
    code, body = _command(client, ["python3", "-c",
        "from pathlib import Path;import hashlib,sys;p=Path(sys.argv[1]);assert not p.is_symlink() and p.is_file();print(hashlib.sha256(p.read_bytes()).hexdigest())", target])
    if code or body.decode().strip() != digest:
        raise VisitorError("uploaded_helper_hash_mismatch")
    return digest


def _collect(client, remote: PurePosixPath, case_id: str, output: Path, *, expected_manifest: str,
             include_default: bool = False) -> dict:
    paths = case_paths(Path(remote.as_posix()), case_id, include_default=include_default)
    with client.open_sftp() as sftp:
        with sftp.open(paths["receipt"].as_posix(), "rb") as handle:
            raw_receipt = handle.read(1024 * 1024)
        receipt = json.loads(raw_receipt)
        if (receipt.get("caseId") != case_id or receipt.get("processFinalized") is not True
                or receipt.get("candidate") != remote.name
                or receipt.get("manifestSha256") != expected_manifest):
            raise VisitorError("case_not_finalized_for_collection")
        items = [("server-receipt.json", raw_receipt)]
        if receipt.get("resultSha256"):
            with sftp.open((paths["output"] / "result.json").as_posix(), "rb") as handle:
                body = handle.read(16 * 1024 * 1024 + 1)
            if len(body) > 16 * 1024 * 1024 or hashlib.sha256(body).hexdigest() != receipt["resultSha256"]:
                raise VisitorError("downloaded_result_hash_mismatch")
            items.append(("result.json", body))
    destination = output / case_id
    destination.mkdir(parents=True, exist_ok=True)
    for name, body in items:
        target = destination / name
        if target.exists():
            if target.read_bytes() != body:
                raise VisitorError("existing_local_result_differs")
            continue
        with target.open("xb") as handle:
            handle.write(body)
    return receipt


def run_client(args) -> list[dict]:
    name = candidate_name(args.candidate)
    cases = selected_cases(args.case_id, args.include_default)
    compatibility = json.loads(args.compatibility_report.read_text(encoding="utf-8"))
    verification = compatibility.get("artifactVerification") or {}
    if (compatibility.get("passed") is not True or compatibility.get("releaseName") != name
            or verification.get("mode") != "manifest_file_equivalent_rebuild"):
        raise VisitorError("candidate_manifest_compatibility_not_passed")
    manifest_sha = verification["manifestSha256"]
    stage_sha = compatibility["validationHelperSha256"]
    if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in (manifest_sha, stage_sha)):
        raise VisitorError("invalid_compatibility_hash")
    remote = PurePosixPath(PARENT.as_posix()) / name
    output = local_output_dir(args.output_dir, name)
    client = _connect(args.credentials_file, args.known_hosts)
    receipts = []
    try:
        if not args.collect_only:
            smoke_sha = _upload(client, ROOT / "scripts/smoke_full_generation.py", remote / "scripts/smoke_full_generation.py")
            target = remote / f"visitor-protocol-{sha(Path(__file__))[:16]}.py"
            _upload(client, Path(__file__), target)
        for case_id in cases:
            if not args.collect_only:
                code, body = _command(client, ["/opt/demos/inquiry-curator/.venv/bin/python", "-B", target,
                    "--remote-run", remote, "--candidate", name, "--case-id", case_id, "--manifest-sha256", manifest_sha,
                    "--stage-helper-sha256", stage_sha, "--smoke-helper-sha256", smoke_sha,
                    *(["--include-default"] if args.include_default else [])], timeout=480)
                response = json.loads(body)
                if response.get("processFinalized") is not True:
                    raise VisitorError("remote_execution_uncertain_collect_only_do_not_retry")
            receipt = _collect(client, remote, case_id, output, expected_manifest=manifest_sha,
                               include_default=args.include_default)
            receipts.append(public_receipt(receipt))
            # Collection and cleanup finish before the next case is even sent.
            if receipt.get("ownedProcessesStopped") is not True:
                raise VisitorError("owned_process_cleanup_incomplete")
            if receipt.get("productionFilesUnchanged") is not True:
                raise VisitorError("production_snapshot_changed_stop_before_next_case")
        return receipts
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--collect-only", action="store_true")
    parser.add_argument("--case-id", choices=(*CASE_ORDER, DEFAULT_CASE))
    parser.add_argument("--include-default", action="store_true",
                        help="Explicitly append one standard default-entry smoke; not a new holdout")
    parser.add_argument("--candidate", help="Explicit inquiry-v6-YYYYMMDD-rcN; required before any SSH")
    parser.add_argument("--compatibility-report", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Local results base; candidate name is always appended")
    parser.add_argument("--credentials-file", type=Path, default=ROOT.parent / "服务器信息.txt")
    parser.add_argument("--known-hosts", type=Path, default=Path.home() / ".ssh/known_hosts")
    parser.add_argument("--remote-run", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--manifest-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--stage-helper-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-helper-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.remote_run:
            if not args.case_id or not all((args.manifest_sha256, args.stage_helper_sha256, args.smoke_helper_sha256)):
                raise VisitorError("remote_contract_arguments_missing")
            receipt = run_remote_case(args.remote_run, args.case_id, args.manifest_sha256,
                                      args.stage_helper_sha256, args.smoke_helper_sha256,
                                      expected_candidate=candidate_name(args.candidate), include_default=args.include_default)
            print(json.dumps(public_receipt(receipt), ensure_ascii=False))
            return 0 if receipt.get("exitCode") == 0 else 1
        if not args.execute and not args.collect_only:
            name = candidate_name(args.candidate) if args.candidate else None
            print(json.dumps({"dryRun": True, "remoteExecuted": False, "paidVisitorRuns": 0,
                              "candidate": name, "caseOrder": list(selected_cases(args.case_id, args.include_default)),
                              "includeDefault": args.include_default,
                              "defaultSmokeIsNewHoldout": False,
                              "candidateRequiredForExecution": True,
                              "localOutputDirectory": str(local_output_dir(args.output_dir, name)) if name else None,
                              "maxVisitorRequestsPerCase": 1, "optionalAssetsWaitSeconds": 60,
                              "stageBudgets": "shipped defaults", "browserVerified": False}))
            return 0
        if args.compatibility_report is None:
            raise VisitorError("compatibility_report_required")
        print(json.dumps(run_client(args), ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"schemaVersion": SCHEMA, "errorType": type(error).__name__,
                          "errorCode": str(error) if isinstance(error, VisitorError) else "operation_failed",
                          "boundary": "No automatic paid retry; inspect persisted receipt or use collect-only after uncertain execution"}))
        return 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
