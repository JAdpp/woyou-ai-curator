from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from scripts import stage_release_candidate as stage


REQUIRED = ("scripts/package_release.py", "scripts/release_preflight.py", "api/app/main.py",
            ".next/standalone/server.js", "deploy/native-runtime/package.json")


def archive(tmp_path: Path, extras=None, corrupt=False, members=None) -> Path:
    target = tmp_path / "payload.tar"
    files = {key: b"fixture" for key in REQUIRED}
    files.update(extras or {})
    manifest = {"schemaVersion": "inquiry-release-package-v1", "packageVerified": True,
                "files": [{"path": key, "bytes": len(value),
                           "sha256": ("0" * 64 if corrupt and key == REQUIRED[0]
                                      else hashlib.sha256(value).hexdigest())}
                          for key, value in files.items()]}
    files["release-manifest.json"] = json.dumps(manifest).encode()
    with tarfile.open(target, "w") as bundle:
        for key, value in files.items():
            info = tarfile.TarInfo(key)
            info.size = len(value)
            bundle.addfile(info, io.BytesIO(value))
        for info in members or []:
            bundle.addfile(info)
    return target


def test_local_archive_validation_checks_all_manifest_hashes_without_extraction(tmp_path):
    source = archive(tmp_path)
    before = set(tmp_path.iterdir())
    result = stage.validate_archive(source)
    assert result["memberHashesVerified"] and result["payloadFiles"] == len(REQUIRED)
    assert result["archiveSha256"] == stage.digest_file(source)
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("name", ["../production/file", "/opt/production/file", "C:/secret", "folder\\file"])
def test_escape_paths_are_rejected(tmp_path, name):
    source = archive(tmp_path, extras={name: b"bad"})
    with pytest.raises(stage.CandidateError):
        stage.validate_archive(source)


@pytest.mark.parametrize("name", [".env", "api/.env.local", "api/runtime/store.json",
                                  "public/generated/posters/old.jpg", "data/qa/private.json",
                                  "deploy/id_rsa", "private.key"])
def test_secret_user_state_and_historical_files_are_rejected(tmp_path, name):
    with pytest.raises(stage.CandidateError, match="forbidden"):
        stage.validate_archive(archive(tmp_path, extras={name: b"private"}))


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE])
def test_links_and_special_files_are_rejected(tmp_path, kind):
    info = tarfile.TarInfo("link")
    info.type = kind
    info.linkname = "/opt/demos/inquiry-curator/.env"
    with pytest.raises(stage.CandidateError, match="links_or_special"):
        stage.validate_archive(archive(tmp_path, members=[info]))


def test_archive_byte_tampering_is_rejected(tmp_path):
    with pytest.raises(stage.CandidateError, match="hash_mismatch"):
        stage.validate_archive(archive(tmp_path, corrupt=True))


def test_extraction_is_exclusive_and_preserves_existing_files(tmp_path):
    source = archive(tmp_path)
    target = tmp_path / "candidate"
    target.mkdir()
    stage.extract_archive(source, target)
    assert (target / "api/app/main.py").read_bytes() == b"fixture"
    with pytest.raises(FileExistsError):
        stage.extract_archive(source, target)
    assert source.exists() and (target / "api/app/main.py").read_bytes() == b"fixture"


@pytest.mark.parametrize("name", ["../production", "x;rm", "x y", "/opt/demos", "..", "", "A"])
def test_release_names_cannot_escape_or_inject_shell_commands(name):
    with pytest.raises(stage.CandidateError, match="invalid_release"):
        stage.release_name(name)


def test_production_environment_is_read_only_and_all_writable_settings_are_isolated(tmp_path):
    candidate = tmp_path / "candidate"
    original = {"DEEPSEEK_API_KEY": "secret-fixture", "STORE_PATH": "/production/store.json",
                "RAG_MODE": "bm25", "IMAGE_CACHE_DIR": "/production/images",
                "ALIYUN_TTS_OUTPUT_DIR": "/production/audio", "QREL_REVIEW_DB_PATH": "/production/reviews.db"}
    before = dict(original)
    env = stage.isolated_environment(candidate, original)
    assert original == before and env["DEEPSEEK_API_KEY"] == "secret-fixture"
    assert env["RAG_MODE"] == "hybrid" and env["API_INTERNAL_URL"] == "http://127.0.0.1:9101"
    assert env["PORT"] == "3301" and env["HOSTNAME"] == "127.0.0.1"
    for key in ("STORE_PATH", "IMAGE_CACHE_DIR", "ALIYUN_IMAGE_OUTPUT_DIR", "ALIYUN_TTS_OUTPUT_DIR",
                "QREL_REVIEW_DB_PATH", "QREL_SUGGESTION_DB_PATH", "RAG_TRACE_DIR", "RAG_MODEL_CACHE_DIR", "NPM_CONFIG_CACHE"):
        assert Path(env[key]).resolve().is_relative_to(candidate.resolve())
    assert tuple(env[key] for key in ("RAG_RETRIEVAL_TIMEOUT_SECONDS", "DEEPSEEK_FRAME_TIMEOUT_SECONDS",
                                      "DEEPSEEK_LABELS_TIMEOUT_SECONDS", "GENERATION_JOB_TIMEOUT_SECONDS")) == ("70", "55", "40", "180")


def test_planning_and_review_use_release_defaults_not_host_or_production_old_values(tmp_path, monkeypatch):
    keys = ("RAG_PLANNING_TIMEOUT_SECONDS", "DEEPSEEK_QUERY_REVIEW_THINKING")
    monkeypatch.setenv(keys[0], "8")
    monkeypatch.setenv(keys[1], "false")
    original = {keys[0]: "16", keys[1]: "false", "DEEPSEEK_API_KEY": "private-fixture"}
    before = dict(original)
    result = stage.isolated_environment(tmp_path/'candidate', original)
    assert not any(key in result for key in keys)
    assert original == before
    assert result["DEEPSEEK_API_KEY"] == "private-fixture"


def test_unconfigured_planning_and_review_are_not_pinned_by_isolation(tmp_path, monkeypatch):
    for key in ("RAG_PLANNING_TIMEOUT_SECONDS", "DEEPSEEK_QUERY_REVIEW_THINKING"):
        monkeypatch.delenv(key, raising=False)
    result = stage.isolated_environment(tmp_path/'candidate', {})
    assert "RAG_PLANNING_TIMEOUT_SECONDS" not in result
    assert "DEEPSEEK_QUERY_REVIEW_THINKING" not in result


class FakeProcess:
    def __init__(self, pid, refuse=False, exited=False):
        self.pid = pid
        self.refuse = refuse
        self.exited = exited
        self.terminated = False
        self.killed = False

    def poll(self):
        return 0 if self.exited else None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.exited = True

    def wait(self, timeout):
        if self.refuse and not self.killed:
            raise subprocess.TimeoutExpired("owned fixture", timeout)
        self.exited = True


def test_cleanup_only_terminates_given_owned_process_handles():
    api, web, existing = FakeProcess(10), FakeProcess(20, refuse=True), FakeProcess(30, exited=True)
    report = stage.stop_owned_processes([api, web, existing])
    assert api.terminated and not api.killed
    assert web.terminated and web.killed
    assert not existing.terminated and not existing.killed
    assert {row["pid"] for row in report} == {10, 20, 30}
    assert all(row["stopped"] for row in report)


def test_default_cli_does_not_attempt_ssh_or_load_credentials(tmp_path, monkeypatch, capsys):
    source = archive(tmp_path)
    monkeypatch.setattr(stage.sys, "argv", ["candidate", "--archive", str(source), "--release-name", "candidate-v6"])
    def forbidden(*args, **kwargs):
        raise AssertionError("default must not execute SSH")
    monkeypatch.setattr(stage, "execute_remote", forbidden)
    assert stage.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dryRun"] and not report["remoteExecuted"]


def test_busy_candidate_ports_fail_without_starting_or_stopping_existing_services(tmp_path, monkeypatch):
    parent = tmp_path / "candidates"
    candidate = parent / "new-release"
    candidate.mkdir(parents=True)
    source = archive(candidate)
    monkeypatch.setattr(stage, "CANDIDATE_PARENT", parent)
    monkeypatch.setattr(stage.platform, "system", lambda: "Linux")
    monkeypatch.setattr(stage, "_free_port", lambda port: False)
    monkeypatch.setattr(stage, "extract_archive", lambda *args: pytest.fail("must not extract"))
    monkeypatch.setattr(stage.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not start"))
    result = stage.remote_validate(candidate, stage.digest_file(source))
    assert not result["passed"] and result["errorCode"] == "candidate_port_already_in_use"
    assert result["ownedProcessesStopped"] == []


@pytest.mark.parametrize("backend_failure", [False, True])
def test_entire_remote_smoke_uses_only_read_requests_and_always_cleans_owned_processes(tmp_path, monkeypatch, backend_failure):
    parent = tmp_path / "candidates"
    candidate = parent / "new-release"
    candidate.mkdir(parents=True)
    source = archive(candidate)
    production = tmp_path / "production"
    (production / ".venv/bin").mkdir(parents=True)
    (production / ".venv/bin/python").write_text("existing python")
    production_env = production / ".env"
    production_env.write_text("DEEPSEEK_API_KEY=private-test-key\nSTORE_PATH=/production/store.json\n")
    original_hash = stage.digest_file(production_env)
    monkeypatch.setattr(stage, "CANDIDATE_PARENT", parent)
    monkeypatch.setattr(stage, "PRODUCTION_ROOT", production)
    monkeypatch.setattr(stage.platform, "system", lambda: "Linux")
    monkeypatch.setattr(stage, "_free_port", lambda port: True)
    commands, processes, reads = [], [], []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["cwd"] == candidate
        assert kwargs["env"]["API_INTERNAL_URL"] == "http://127.0.0.1:9101"
        return SimpleNamespace(returncode=0, stdout=json.dumps({"passed": True}))

    def start(command, **kwargs):
        commands.append(command)
        assert kwargs["env"]["STORE_PATH"] == str(candidate / "candidate-state/store.json")
        process = FakeProcess(100 + len(processes))
        processes.append(process)
        return process

    def wait(url, process):
        reads.append(url)
        if url.endswith("/health"):
            return 200, "application/json", json.dumps({"status": "ok", "retrieval": {
                "mode": "hybrid", "available": True, "collectionId": "global_open"}}).encode()
        return 200, "text/html", b'<html><script src="/_next/static/app.js"></script></html>'

    def get(url, **kwargs):
        reads.append(url)
        if "/api/collection/highlights" in url:
            if backend_failure:
                raise RuntimeError("private-upstream-content")
            return 200, "application/json", b'{"objects":[]}'
        return 200, "application/javascript", b'console.log("candidate")'

    monkeypatch.setattr(stage.subprocess, "run", run)
    monkeypatch.setattr(stage.subprocess, "Popen", start)
    monkeypatch.setattr(stage, "_wait_http", wait)
    monkeypatch.setattr(stage, "_get", get)
    result = stage.remote_validate(candidate, stage.digest_file(source))
    assert result["passed"] is (not backend_failure)
    assert processes and all(process.terminated for process in processes)
    assert all(row["stopped"] for row in result["ownedProcessesStopped"])
    assert stage.digest_file(production_env) == original_hash
    assert not any("systemctl" in command or "nginx" in command for command in commands)
    assert not any("generate" in url or "embedding" in url for url in reads)
    assert "private" not in json.dumps(result)


def test_cleanup_failure_does_not_prevent_other_owned_processes_from_stopping():
    class Unstoppable(FakeProcess):
        def terminate(self):
            raise PermissionError("private OS text")
    safe, failing = FakeProcess(1), Unstoppable(2)
    result = stage.stop_owned_processes([safe, failing])
    assert safe.terminated
    assert result[0] == {"pid": 2, "stopped": False, "errorType": "PermissionError"}
    assert result[1]["stopped"] is True


class FakeSFTP:
    def __init__(self, destination):
        self.destination = destination
        self.opened_modes = []

    def open(self, remote_file, mode):
        self.opened_modes.append(mode)
        handle = self.destination.open("xb" if mode == "wx" else mode)
        class Wrapped:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                handle.close()
            def set_pipelined(self, value):
                assert value is True
            def seek(self, value):
                handle.seek(value)
            def write(self, value):
                handle.write(value)
        return Wrapped()

    def stat(self, remote_file):
        return self.destination.stat()


def test_resume_hashes_existing_prefix_and_appends_without_truncation(tmp_path):
    source = tmp_path / "source.tar"
    source.write_bytes(b"known-prefix-remainder")
    destination = tmp_path / "remote.tar"
    destination.write_bytes(b"known-prefix")
    sftp = FakeSFTP(destination)
    before = destination.read_bytes()
    offset = stage.upload_verified_remainder(sftp, source, "/candidate/payload.tar", {
        "bytes": len(before), "sha256": hashlib.sha256(before).hexdigest(),
    })
    assert offset == len(before)
    assert destination.read_bytes() == source.read_bytes()
    assert sftp.opened_modes == ["r+b"]


@pytest.mark.parametrize("row", [
    {"bytes": 3, "sha256": "0" * 64},
    {"bytes": -1, "sha256": "0" * 64},
    {"bytes": 1000, "sha256": "0" * 64},
    {"bytes": True, "sha256": "0" * 64},
])
def test_resume_rejects_mismatch_or_invalid_size_before_writing(tmp_path, row):
    source = tmp_path / "source.tar"
    source.write_bytes(b"payload")
    destination = tmp_path / "remote.tar"
    destination.write_bytes(b"old")
    sftp = FakeSFTP(destination)
    with pytest.raises(stage.CandidateError):
        stage.upload_verified_remainder(sftp, source, "/candidate/payload.tar", row)
    assert destination.read_bytes() == b"old"
    assert sftp.opened_modes == []


def test_complete_verified_upload_is_not_written_again(tmp_path):
    source = tmp_path / "source.tar"
    source.write_bytes(b"payload")
    sftp = FakeSFTP(source)
    assert stage.upload_verified_remainder(sftp, source, "/candidate/payload.tar", {
        "bytes": source.stat().st_size, "sha256": stage.digest_file(source),
    }) == len(b"payload")
    assert sftp.opened_modes == []


def test_new_upload_requires_exclusive_creation(tmp_path):
    source = tmp_path / "source.tar"
    source.write_bytes(b"payload")
    destination = tmp_path / "remote.tar"
    sftp = FakeSFTP(destination)
    assert stage.upload_verified_remainder(sftp, source, "/candidate/payload.tar", None) == 0
    assert sftp.opened_modes == ["wx"]
    with pytest.raises(FileExistsError):
        stage.upload_verified_remainder(sftp, source, "/candidate/payload.tar", None)


def test_resume_requires_explicit_execute(tmp_path, monkeypatch, capsys):
    source = archive(tmp_path)
    monkeypatch.setattr(stage.sys, "argv", ["candidate", "--archive", str(source), "--release-name", "candidate-v6", "--resume-upload"])
    assert stage.main() == 1
    assert json.loads(capsys.readouterr().out)["errorCode"] == "resume_upload_requires_execute"


def test_upload_inventory_has_fixed_posix_root_and_rejects_extracted_payload():
    class Output(io.BytesIO):
        channel = SimpleNamespace(recv_exit_status=lambda: 0)
    commands = []
    class Client:
        def exec_command(self, command, **kwargs):
            commands.append(command)
            return None, Output(b'{"files":{},"unextracted":true}'), io.BytesIO()
    result = stage.inspect_upload(Client(), Path("/opt/demos/inquiry-curator-candidates/rc2"))
    assert result["unextracted"]
    assert "inquiry-curator-candidates" in commands[0]
    assert "stat.S_ISREG" in commands[0]
    assert "p.resolve()==p" in commands[0]
    assert "release-manifest.json" not in commands[0]  # not allowlisted once extracted


def test_manifest_equivalent_rebuild_checks_all_bytes_without_claiming_tar_upload(tmp_path):
    source = archive(tmp_path)
    target = tmp_path / "rebuild"
    target.mkdir()
    stage.extract_archive(source, target)
    digest = stage.digest_file(target / "release-manifest.json")
    result = stage.verify_manifest_tree(target, digest)
    assert result["filesVerified"] == len(REQUIRED)
    assert not result["originalTarUploadVerified"]
    (target / "api/app/main.py").write_bytes(b"tampered")
    with pytest.raises(stage.CandidateError, match="hash_or_size"):
        stage.verify_manifest_tree(target, digest)


def test_manifest_rebuild_requires_exact_manifest_identity(tmp_path):
    source = archive(tmp_path)
    target = tmp_path / "rebuild"
    target.mkdir()
    stage.extract_archive(source, target)
    with pytest.raises(stage.CandidateError, match="manifest_hash"):
        stage.verify_manifest_tree(target, "0" * 64)
