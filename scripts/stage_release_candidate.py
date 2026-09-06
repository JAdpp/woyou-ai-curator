"""Validate a release on isolated server ports without switching production.

Default is LOCAL archive validation only. --execute is required for SSH/upload.
No generation/embedding requests, systemctl calls or nginx changes are made.
The remote helper reuses production's Python environment and reads its .env,
but redirects every candidate store/cache/output path into a fresh release root.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = Path("/opt/demos/inquiry-curator")
CANDIDATE_PARENT = Path("/opt/demos/inquiry-curator-candidates")
API_PORT = 9101
WEB_PORT = 3301
MAX_FILES = 8000
MAX_BYTES = 2 * 1024 * 1024 * 1024
SCHEMA = "isolated-server-candidate-v1"


class CandidateError(RuntimeError):
    pass


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_prefix(path: Path, byte_count: int) -> str:
    if not 0 <= byte_count <= path.stat().st_size:
        raise CandidateError("remote_upload_size_invalid")
    digest = hashlib.sha256()
    remaining = byte_count
    with path.open("rb") as handle:
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise CandidateError("local_archive_changed_during_upload")
            remaining -= len(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def release_name(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,79}", value) or ".." in value:
        raise CandidateError("invalid_release_name")
    return value


def archive_member_path(name: str) -> PurePosixPath:
    if "\\" in name or "\x00" in name or ":" in name:
        raise CandidateError("invalid_archive_path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise CandidateError("archive_path_escape")
    return path


def _forbidden_payload_path(path: PurePosixPath) -> bool:
    parts = [part.casefold() for part in path.parts]
    if any(part.startswith(".env") or part in {".git", ".npmrc", "id_rsa", "id_ed25519", "credentials.json"}
           for part in parts):
        return True
    if path.suffix.casefold() in {".pem", ".key", ".pfx", ".p12"}:
        return True
    if "generated" in parts and "public" in parts:
        return True
    name = path.as_posix()
    if name.startswith("api/runtime/") and name not in {
        "api/runtime/cache", "api/runtime/cache/rag", "api/runtime/cache/filters"
    } and not name.startswith(("api/runtime/cache/rag/", "api/runtime/cache/filters/")):
        return True
    return name.startswith(("data/qa/", "artifacts/")) or "/raw/" in name


def validate_archive(archive: Path) -> dict:
    """Check every member, then every manifest byte count/hash; no extraction."""
    if not archive.is_file() or archive.is_symlink():
        raise CandidateError("archive_missing_or_symlink")
    with tarfile.open(archive, "r:*") as bundle:
        members = bundle.getmembers()
        if len(members) > MAX_FILES:
            raise CandidateError("archive_member_limit_exceeded")
        files: dict[str, tarfile.TarInfo] = {}
        directories: set[str] = set()
        total = 0
        for member in members:
            path = archive_member_path(member.name)
            name = path.as_posix()
            if name == "." and member.isdir():
                continue
            if name in files or name in directories or _forbidden_payload_path(path):
                raise CandidateError("duplicate_or_forbidden_archive_member")
            if member.isdir():
                directories.add(name)
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise CandidateError("archive_links_or_special_files_forbidden")
            if member.size < 0:
                raise CandidateError("invalid_archive_member_size")
            files[name] = member
            total += member.size
        if total > MAX_BYTES or "release-manifest.json" not in files:
            raise CandidateError("archive_size_or_manifest_invalid")
        manifest_member = files["release-manifest.json"]
        if manifest_member.size > 8 * 1024 * 1024:
            raise CandidateError("manifest_too_large")
        handle = bundle.extractfile(manifest_member)
        manifest = json.load(handle)
        if not isinstance(manifest, dict) or manifest.get("schemaVersion") != "inquiry-release-package-v1" or manifest.get("packageVerified") is not True:
            raise CandidateError("package_manifest_not_verified")
        declared = manifest.get("files")
        if not isinstance(declared, list):
            raise CandidateError("manifest_files_invalid")
        seen: set[str] = set()
        # Hash in physical archive order. Alphabetical manifest order can
        # otherwise repeatedly rewind a gzip stream hundreds of times.
        ordered = sorted(declared, key=lambda row: (
            files.get(str(row.get("path", "")), manifest_member).offset_data
            if isinstance(row, dict) else -1
        ))
        for row in ordered:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise CandidateError("manifest_file_invalid")
            name = archive_member_path(row["path"]).as_posix()
            if name in seen or name not in files or name == "release-manifest.json":
                raise CandidateError("manifest_duplicate_or_missing_member")
            seen.add(name)
            if type(row.get("bytes")) is not int or row["bytes"] != files[name].size or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256") or "")):
                raise CandidateError("manifest_file_size_or_hash_invalid")
            source = bundle.extractfile(files[name])
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != row["sha256"]:
                raise CandidateError("archive_content_hash_mismatch")
        if seen != set(files) - {"release-manifest.json"}:
            raise CandidateError("undeclared_archive_file")
        required = {"scripts/package_release.py", "scripts/release_preflight.py", "api/app/main.py",
                    ".next/standalone/server.js", "deploy/native-runtime/package.json"}
        if not required.issubset(seen):
            raise CandidateError("archive_missing_runtime_components")
    return {"archiveSha256": digest_file(archive), "payloadFiles": len(seen), "unpackedBytes": total,
            "manifestIdentity": manifest.get("identity", {}), "secretFreePathCheck": True,
            "memberHashesVerified": True}


def extract_archive(archive: Path, destination: Path) -> None:
    """Manual exclusive extraction after validation, never tar.extractall."""
    validate_archive(archive)
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle.getmembers():
            path = archive_member_path(member.name)
            if path.as_posix() == ".":
                continue
            target = root.joinpath(*path.parts)
            if not target.resolve().is_relative_to(root):
                raise CandidateError("extraction_target_escape")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)


def isolated_environment(candidate: Path, original: dict[str, str | None]) -> dict[str, str]:
    root = candidate.resolve()
    env = os.environ.copy()
    env.update({key: value for key, value in original.items() if isinstance(value, str)})
    # Planning/review behavior is part of the candidate's shipped config, not
    # the host's old process state. Never pin these to an earlier RC's values.
    # The paid smoke additionally derives its defaults from that same package.
    for key in ("RAG_PLANNING_TIMEOUT_SECONDS", "DEEPSEEK_QUERY_REVIEW_THINKING"):
        env.pop(key, None)
    env.update({
        "PYTHONPATH": str(root / "api"), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "STORE_MODE": "json", "STORE_PATH": str(root / "candidate-state/store.json"),
        "IMAGE_CACHE_DIR": str(root / "candidate-state/images"),
        "ALIYUN_IMAGE_OUTPUT_DIR": str(root / ".next/standalone/public/generated/posters"),
        "ALIYUN_TTS_OUTPUT_DIR": str(root / "candidate-state/audio"),
        "COLLECTIONS_DIR": str(root / "data/collections"), "DEFAULT_COLLECTION_ID": "global_open",
        "RAG_INDEX_DIR": str(root / "api/runtime/cache/rag"),
        "RAG_FILTER_INDEX_DIR": str(root / "api/runtime/cache/filters"),
        "RAG_MODEL_CACHE_DIR": str(root / "candidate-state/model-cache"),
        "RAG_TRACE_DIR": str(root / "candidate-state/retrieval-traces"),
        "QREL_REVIEW_DATASET_DIR": str(root / "candidate-state/no-evaluation-dataset"),
        "QREL_REVIEW_DB_PATH": str(root / "candidate-state/qrel/reviews.sqlite3"),
        "QREL_SUGGESTION_DB_PATH": str(root / "candidate-state/qrel/suggestions.sqlite3"),
        "NPM_CONFIG_CACHE": str(root / "candidate-state/npm-cache"),
        "npm_config_cache": str(root / "candidate-state/npm-cache"),
        "RAG_MODE": "hybrid", "RAG_EMBEDDING_PROVIDER": "aliyun",
        "RAG_EMBEDDING_MODEL": "qwen3.7-text-embedding", "RAG_EMBEDDING_DIMENSION": "768",
        "RAG_RERANK_ENABLED": "true", "RAG_LLM_AUDIT_ENABLED": "true",
        "RAG_STRUCTURED_FILTERS_ENABLED": "true", "RAG_RETRIEVAL_TIMEOUT_SECONDS": "70",
        "DEEPSEEK_TIMEOUT_SECONDS": "90", "DEEPSEEK_FRAME_TIMEOUT_SECONDS": "55",
        "DEEPSEEK_LABELS_TIMEOUT_SECONDS": "40", "GENERATION_JOB_TIMEOUT_SECONDS": "180",
        "GENERATION_POSTER_WAIT_SECONDS": "2", "API_INTERNAL_URL": f"http://127.0.0.1:{API_PORT}",
        "PORT": str(WEB_PORT), "HOSTNAME": "127.0.0.1", "NODE_ENV": "production",
    })
    return env


def _free_port(port: int) -> bool:
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _get(url: str, limit: int = 4 * 1024 * 1024) -> tuple[int, str, bytes]:
    # Never send local health checks through inherited corporate/provider proxies.
    with build_opener(ProxyHandler({})).open(url, timeout=8) as response:
        body = response.read(limit + 1)
        if len(body) > limit:
            raise CandidateError("http_response_limit_exceeded")
        return response.status, response.headers.get("Content-Type", ""), body


def _wait_http(url: str, process: subprocess.Popen, seconds: float = 55) -> tuple[int, str, bytes]:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CandidateError("candidate_process_exited_before_ready")
        try:
            return _get(url)
        except Exception:
            time.sleep(0.3)
    raise CandidateError("candidate_http_startup_timeout")


def stop_owned_processes(processes: list[subprocess.Popen]) -> list[dict]:
    stopped = []
    for process in reversed(processes):
        row = {"pid": process.pid, "stopped": False}
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            row["stopped"] = process.poll() is not None
        except Exception as error:
            row["errorType"] = type(error).__name__
        stopped.append(row)
    return stopped


def verify_manifest_tree(root: Path, expected_manifest_sha: str) -> dict:
    """Verify an equivalent file rebuild; never call it an uploaded tar hash."""
    manifest_path = root / "release-manifest.json"
    if manifest_path.is_symlink() or digest_file(manifest_path) != expected_manifest_sha:
        raise CandidateError("rebuilt_manifest_hash_mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("files")
    if manifest.get("schemaVersion") != "inquiry-release-package-v1" or manifest.get("packageVerified") is not True or not isinstance(rows, list) or not 1 <= len(rows) <= MAX_FILES:
        raise CandidateError("rebuilt_manifest_invalid")
    seen = set()
    total = 0
    for row in rows:
        name = archive_member_path(row["path"])
        if name.as_posix() in seen or _forbidden_payload_path(name):
            raise CandidateError("rebuilt_manifest_path_invalid")
        seen.add(name.as_posix())
        path = root.joinpath(*name.parts)
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or not path.is_file():
            raise CandidateError("rebuilt_member_path_invalid")
        if type(row.get("bytes")) is not int or path.stat().st_size != row["bytes"] or digest_file(path) != row.get("sha256"):
            raise CandidateError("rebuilt_member_hash_or_size_mismatch")
        total += row["bytes"]
    if total > MAX_BYTES or not {"scripts/package_release.py", "scripts/release_preflight.py", "api/app/main.py", ".next/standalone/server.js", "deploy/native-runtime/package.json"}.issubset(seen):
        raise CandidateError("rebuilt_runtime_components_invalid")
    return {"mode": "manifest_file_equivalent_rebuild", "manifestSha256": expected_manifest_sha,
            "filesVerified": len(seen), "bytesVerified": total, "originalTarUploadVerified": False}


def remote_validate(candidate: Path, expected_archive_sha: str | None, *, expected_manifest_sha: str | None = None) -> dict:
    root = candidate.resolve()
    if root.parent != CANDIDATE_PARENT or release_name(root.name) != root.name or not root.is_dir():
        raise CandidateError("candidate_root_outside_fixed_parent")
    if platform.system() != "Linux":
        raise CandidateError("candidate_requires_linux")
    archive = root / "payload.tar"
    if expected_manifest_sha:
        artifact = verify_manifest_tree(root, expected_manifest_sha)
    elif digest_file(archive) != expected_archive_sha:
        raise CandidateError("uploaded_archive_hash_mismatch")
    else:
        artifact = {"mode": "full_archive_upload", "archiveSha256": expected_archive_sha, "originalTarUploadVerified": True}
    report = {"schemaVersion": SCHEMA, "releaseName": root.name,
              "startedAt": datetime.now(timezone.utc).isoformat(), "paidProviderCalls": 0,
              "productionFilesModified": False, "serviceConfigurationModified": False,
              "candidateOnly": True, "artifactVerification": artifact, "checks": [], "ownedProcessesStopped": []}
    processes: list[subprocess.Popen] = []
    try:
        if not all(_free_port(port) for port in (API_PORT, WEB_PORT)):
            raise CandidateError("candidate_port_already_in_use")
        if not expected_manifest_sha:
            extract_archive(archive, root)
        python = PRODUCTION_ROOT / ".venv/bin/python"
        if not python.is_file() or not (PRODUCTION_ROOT / ".env").is_file():
            raise CandidateError("existing_production_python_or_environment_missing")
        from dotenv import dotenv_values
        env = isolated_environment(root, dotenv_values(PRODUCTION_ROOT / ".env"))
        for command, code, seconds in (
            (["npm", "install", "--prefix", str(root / "deploy/native-runtime"), "--ignore-scripts", "--no-audit", "--no-fund"], "native_npm_install", 150),
            ([str(python), str(root / "scripts/package_release.py"), "--prepare-linux-native", str(root)], "native_prepare", 30),
            (["node", "-e", "require('./.next/standalone/node_modules/sharp')"], "sharp_load", 20),
        ):
            completed = subprocess.run(command, cwd=root, env=env, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, timeout=seconds)
            report["checks"].append({"id": code, "passed": completed.returncode == 0})
            if completed.returncode:
                raise CandidateError(f"{code}_failed")
        # Validate indexes while only one Python process owns the large corpus;
        # then start API and Next, avoiding a second concurrent preflight load.
        preflight = subprocess.run([str(python), "scripts/release_preflight.py", "--expect-mode", "hybrid",
                                    "--standalone-dir", ".next/standalone"], cwd=root, env=env,
                                   capture_output=True, text=True, timeout=90)
        try:
            parsed = json.loads(preflight.stdout)
            preflight_ok = preflight.returncode == 0 and parsed.get("passed") is True
        except (ValueError, TypeError):
            preflight_ok = False
        report["checks"].append({"id": "offline_runtime_index_preflight", "passed": preflight_ok})
        if not preflight_ok:
            raise CandidateError("candidate_preflight_failed")
        api = subprocess.Popen([str(python), "-m", "uvicorn", "app.main:app", "--app-dir", "api",
                                "--host", "127.0.0.1", "--port", str(API_PORT)], cwd=root, env=env,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(api)
        status, _, body = _wait_http(f"http://127.0.0.1:{API_PORT}/health", api)
        health = json.loads(body)
        retrieval = health.get("retrieval", {})
        ready = status == 200 and health.get("status") == "ok" and retrieval.get("mode") == "hybrid" and retrieval.get("available") is True
        report["checks"].append({"id": "candidate_api_health", "passed": ready,
                                 "retrieval": {key: retrieval.get(key) for key in (
                                     "mode", "available", "collectionId", "collectionVersion", "fingerprint", "model")}})
        if not ready:
            raise CandidateError("candidate_health_not_hybrid_ready")
        status, _, body = _get(f"http://127.0.0.1:{API_PORT}/api/collection/highlights?limit=3")
        highlights = json.loads(body)
        report["checks"].append({"id": "candidate_backend_read", "passed": status == 200 and isinstance(highlights, dict),
                                 "httpStatus": status})
        web = subprocess.Popen(["node", ".next/standalone/server.js"], cwd=root, env=env,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(web)
        status, _, body = _wait_http(f"http://127.0.0.1:{WEB_PORT}/", web)
        html = body.decode("utf-8", errors="replace")
        home_ok = status == 200 and "<html" in html.lower()
        report["checks"].append({"id": "candidate_next_home", "passed": home_ok, "httpStatus": status})
        scripts = list(dict.fromkeys(re.findall(r'<script[^>]+src="(/_next/static/[^"<>]+\.js(?:\?[^"<>]*)?)"', html)))[:3]
        if not home_ok or not scripts:
            raise CandidateError("candidate_home_or_static_links_missing")
        for path in scripts:
            status, content_type, body = _get(f"http://127.0.0.1:{WEB_PORT}{path}", limit=16 * 1024 * 1024)
            report["checks"].append({"id": "candidate_next_static", "passed": status == 200 and bool(body) and "javascript" in content_type,
                                     "httpStatus": status, "path": path, "bytes": len(body)})
        report["passed"] = all(row["passed"] for row in report["checks"])
    except Exception as error:
        report["passed"] = False
        report["errorCode"] = str(error) if isinstance(error, CandidateError) else type(error).__name__
    finally:
        report["ownedProcessesStopped"] = stop_owned_processes(processes)
        if any(not row["stopped"] for row in report["ownedProcessesStopped"]):
            report["passed"] = False
        report["finishedAt"] = datetime.now(timezone.utc).isoformat()
        report["notEstablished"] = ["paid model connectivity", "actual generation", "3D/browser controls",
                                    "nginx forwarding or public cutover", "poster/TTS playback", "concurrent visitor capacity"]
    return report


def inspect_upload(client, remote: Path) -> dict:
    """Read-only guard: only an unextracted candidate can resume an upload."""
    code = """from pathlib import Path
import hashlib,json,re,stat,sys
p=Path(sys.argv[1])
assert p.parent==Path('/opt/demos/inquiry-curator-candidates')
assert not p.is_symlink() and not p.parent.is_symlink() and p.resolve()==p and p.is_dir()
files={}
for q in p.iterdir():
 assert q.name=='payload.tar' or re.fullmatch(r'candidate-validator(?:-[0-9a-f]{16})?\\.py',q.name)
 st=q.lstat(); assert stat.S_ISREG(st.st_mode)
 h=hashlib.sha256()
 with q.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''): h.update(block)
 files[q.name]={'bytes':st.st_size,'sha256':h.hexdigest()}
print(json.dumps({'files':files,'unextracted':True}))
"""
    command = " ".join(shlex.quote(value) for value in ("python3", "-c", code, remote.as_posix()))
    _, stdout, stderr = client.exec_command(command, timeout=45)
    body = stdout.read(65536)
    stderr.read(4096)
    if stdout.channel.recv_exit_status() != 0:
        raise CandidateError("candidate_not_a_resumable_upload")
    result = json.loads(body)
    if not isinstance(result, dict) or result.get("unextracted") is not True or not isinstance(result.get("files"), dict):
        raise CandidateError("invalid_upload_inventory")
    return result


def upload_verified_remainder(sftp, archive: Path, remote_file: str, existing: dict | None) -> int:
    """Append only after a byte-for-byte local prefix match; never truncate."""
    size = archive.stat().st_size
    offset = 0 if existing is None else existing.get("bytes")
    if type(offset) is not int or not 0 <= offset <= size:
        raise CandidateError("remote_upload_size_invalid")
    if existing is not None and digest_prefix(archive, offset) != existing.get("sha256"):
        raise CandidateError("remote_upload_prefix_hash_mismatch")
    if existing is not None and offset == size:
        return offset
    with archive.open("rb") as source, sftp.open(remote_file, "r+b" if existing is not None else "wx") as target:
        # Pipelined writes keep large cross-region transfers practical. The
        # final close waits for acknowledgements; a disconnect leaves only a
        # prefix, which the next explicit --resume-upload must hash again.
        target.set_pipelined(True)
        source.seek(offset)
        target.seek(offset)
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            target.write(chunk)
    if sftp.stat(remote_file).st_size != size:
        raise CandidateError("uploaded_archive_size_mismatch")
    return offset


def execute_remote(archive: Path, name: str, credentials: Path, known_hosts: Path, archive_sha: str,
                   *, resume_upload: bool = False) -> dict:
    import paramiko
    from scripts.probe_server_provider_tls import PROJECT_SERVER, load_project_credentials
    name = release_name(name)
    remote = CANDIDATE_PARENT / name
    client = paramiko.SSHClient()
    stage = "ssh_connect"
    transfer = {"archiveSha256": archive_sha, "archiveBytes": archive.stat().st_size,
                "resumeRequested": resume_upload, "remoteArchiveHashVerified": False}
    try:
        client.load_host_keys(str(known_hosts))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        username, password = load_project_credentials(credentials)
        client.connect(PROJECT_SERVER, username=username, password=password, allow_agent=False, look_for_keys=False,
                       timeout=10, banner_timeout=15, auth_timeout=20)
        password = ""
        client.get_transport().set_keepalive(15)
        # mkdir(exist_ok=False) is the allocation gate. It never runs rm/mv and
        # cannot overwrite either production or an earlier candidate release.
        allocate = (
            "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
            "assert not p.parent.is_symlink() and p.parent.resolve() == p.parent; "
            "p.parent.mkdir(exist_ok=True); p.mkdir(exist_ok=False)"
        )
        if not resume_upload:
            stage = "candidate_allocate"
            command = " ".join(shlex.quote(value) for value in ("python3", "-c", allocate, remote.as_posix()))
            _, stdout, stderr = client.exec_command(command, timeout=20)
            stdout.read(4096)
            stderr.read(4096)
            if stdout.channel.recv_exit_status() != 0:
                raise CandidateError("candidate_directory_allocation_failed")
        stage = "upload_prefix_verification"
        inventory = inspect_upload(client, remote)
        existing = inventory["files"].get("payload.tar")
        transfer["previousBytes"] = 0 if existing is None else existing["bytes"]
        helper = Path(__file__).resolve()
        helper_sha = digest_file(helper)
        helper_name = f"candidate-validator-{helper_sha[:16]}.py"
        stage = "archive_upload"
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(90)
            # Upload helper separately because the archive was already frozen
            # before this non-production validation script was introduced.
            upload_verified_remainder(sftp, archive, (remote / "payload.tar").as_posix(), existing)
            stage = "helper_upload"
            helper_existing = inventory["files"].get(helper_name)
            if helper_existing is None:
                with sftp.open((remote / helper_name).as_posix(), "wx") as handle:
                    handle.write(helper.read_bytes())
            elif helper_existing.get("sha256") != helper_sha:
                raise CandidateError("existing_helper_hash_mismatch")
        stage = "uploaded_archive_verification"
        uploaded = inspect_upload(client, remote)["files"]
        if uploaded.get("payload.tar") != {"bytes": archive.stat().st_size, "sha256": archive_sha}:
            raise CandidateError("uploaded_archive_hash_mismatch")
        if uploaded.get(helper_name, {}).get("sha256") != helper_sha:
            raise CandidateError("uploaded_helper_hash_mismatch")
        transfer["remoteArchiveHashVerified"] = True
        transfer["helperSha256"] = helper_sha
        stage = "remote_runtime_validation"
        command = " ".join(shlex.quote(value) for value in (
            (PRODUCTION_ROOT / ".venv/bin/python").as_posix(), "-B", (remote / helper_name).as_posix(),
            "--remote-validate", remote.as_posix(), "--archive-sha256", archive_sha))
        _, stdout, stderr = client.exec_command(command, timeout=480)
        body = stdout.read(1024 * 1024)
        stderr.read(65536)  # Never forward remote exception or provider strings.
        code = stdout.channel.recv_exit_status()
        report = json.loads(body)
        if not isinstance(report, dict) or report.get("schemaVersion") != SCHEMA:
            raise CandidateError("invalid_remote_validation_report")
        if code:
            report["passed"] = False
        report["transfer"] = transfer
        return report
    except Exception as error:
        return {"schemaVersion": SCHEMA, "releaseName": name, "passed": False,
                "failedStage": stage, "errorCode": str(error) if isinstance(error, CandidateError) else type(error).__name__,
                "transfer": transfer, "candidateOnly": True, "paidProviderCalls": 0,
                "productionFilesModified": False, "serviceConfigurationModified": False}
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--release-name")
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize candidate-only SSH/upload/start/stop")
    parser.add_argument("--resume-upload", action="store_true", help="With --execute, resume only an unextracted candidate whose archive prefix matches")
    parser.add_argument("--report", type=Path, help="Write a new local verification report without overwriting an earlier run")
    parser.add_argument("--credentials-file", type=Path, default=ROOT.parent / "服务器信息.txt")
    parser.add_argument("--known-hosts", type=Path, default=Path.home() / ".ssh/known_hosts")
    parser.add_argument("--remote-validate", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--archive-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--manifest-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.report is not None and args.report.exists():
            raise CandidateError("report_already_exists")
        if args.resume_upload and not args.execute:
            raise CandidateError("resume_upload_requires_execute")
        if args.remote_validate:
            if bool(args.archive_sha256) == bool(args.manifest_sha256) or not re.fullmatch(r"[0-9a-f]{64}", args.archive_sha256 or args.manifest_sha256 or ""):
                raise CandidateError("invalid_uploaded_archive_hash")
            # A terminated SSH/helper process unwinds its own subprocesses.
            def interrupted(_number, _frame):
                raise CandidateError("remote_validation_interrupted")
            for number in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGHUP", signal.SIGTERM)):
                signal.signal(number, interrupted)
            result = remote_validate(args.remote_validate, args.archive_sha256, expected_manifest_sha=args.manifest_sha256)
        else:
            if not args.archive or not args.release_name:
                parser.error("--archive and --release-name are required")
            name = release_name(args.release_name)
            checked = validate_archive(args.archive)
            if args.execute:
                result = execute_remote(args.archive, name, args.credentials_file, args.known_hosts, checked["archiveSha256"],
                                        resume_upload=args.resume_upload)
            else:
                result = {"schemaVersion": SCHEMA, "passed": True, "dryRun": True, "remoteExecuted": False,
                          "releaseName": name, "candidatePath": (CANDIDATE_PARENT / name).as_posix(),
                          "archive": checked, "paidProviderCalls": 0,
                          "boundary": "Only local archive checks; no SSH, credentials, remote files or services touched"}
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("passed") else 1
    except Exception as error:
        print(json.dumps({"schemaVersion": SCHEMA, "passed": False,
                          "errorCode": str(error) if isinstance(error, CandidateError) else type(error).__name__}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
