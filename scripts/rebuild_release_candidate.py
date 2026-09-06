"""Manifest-equivalent candidate rebuild; no server/model calls by default.

--execute allocates a NEW candidate, copies only manifest-listed unchanged
files from an explicitly named previous candidate, uploads changed files, and
checks every new SHA/size before isolated Linux/API/Next validation. This is not
an original tar upload. No production switch, package creation or paid visitor
generation is performed. On uncertain execution, use --collect-only; never
replay allocation or overwrite an interrupted candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path('/opt/demos/inquiry-curator-candidates')
SCHEMA = 'candidate-manifest-rebuild-v2'
REQUIRED = {'scripts/package_release.py', 'scripts/release_preflight.py', 'api/app/main.py',
            '.next/standalone/server.js', 'deploy/native-runtime/package.json'}


class RebuildError(RuntimeError):
    pass


def candidate_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'inquiry-v6-[0-9]{8}-rc[1-9][0-9]{0,3}', value):
        raise RebuildError('invalid_candidate_name')
    return value


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def release_dir(name: str) -> Path:
    return ROOT / 'artifacts/releases' / candidate_name(name).replace('inquiry-v6-', 'inquiry-curator-v6-', 1)


def safe_relative(value: str) -> PurePosixPath:
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value or '\x00' in value
            or PurePosixPath(value).as_posix() != value):
        raise RebuildError('invalid_manifest_path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise RebuildError('manifest_path_escape')
    parts = [part.casefold() for part in path.parts]
    if (any(part.startswith('.env') or part in {'.git', '.npmrc', 'id_rsa', 'id_ed25519', 'credentials.json'} for part in parts)
            or path.suffix.casefold() in {'.pem', '.key', '.pfx', '.p12'}
            or ('generated' in parts and 'public' in parts)
            or value.startswith(('data/qa/', 'artifacts/')) or '/raw/' in value
            or (value.startswith('api/runtime/') and not value.startswith(('api/runtime/cache/rag/', 'api/runtime/cache/filters/')))):
        raise RebuildError('forbidden_manifest_payload')
    return path


def manifest_rows(path: Path, expected_sha: str | None = None) -> dict[str, dict]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise RebuildError('invalid_manifest_file')
    if expected_sha is not None and digest(path) != expected_sha:
        raise RebuildError('manifest_hash_mismatch')
    manifest = json.loads(path.read_text(encoding='utf8'))
    rows = manifest.get('files')
    if (manifest.get('schemaVersion') != 'inquiry-release-package-v1' or manifest.get('packageVerified') is not True
            or not isinstance(rows, list) or not 1 <= len(rows) <= 8000):
        raise RebuildError('invalid_manifest_contract')
    result = {}
    total = 0
    for row in rows:
        if not isinstance(row, dict):
            raise RebuildError('invalid_manifest_row')
        name = safe_relative(row.get('path')).as_posix()
        if (name in result or type(row.get('bytes')) is not int or row['bytes'] < 0
                or not isinstance(row.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', row['sha256'])):
            raise RebuildError('invalid_manifest_row')
        result[name] = row
        total += row['bytes']
    if total > 2 * 1024 * 1024 * 1024 or not REQUIRED.issubset(result):
        raise RebuildError('manifest_components_or_size_invalid')
    return result


def regular_within(base: Path, relative: str) -> Path:
    path = base
    for part in safe_relative(relative).parts:
        path = path / part
        if path.is_symlink():
            raise RebuildError('symlink_source_forbidden')
    if not path.resolve().is_relative_to(base.resolve()) or not path.is_file():
        raise RebuildError('source_file_missing_or_outside_root')
    return path


def changed_paths(current: dict[str, dict], previous: dict[str, dict]) -> set[str]:
    return {name for name, row in current.items() if
            (row['sha256'], row['bytes']) != (previous.get(name, {}).get('sha256'), previous.get(name, {}).get('bytes'))}


def prepare_plan(name: str, source_name: str, directory: Path | None = None, source_manifest: Path | None = None) -> dict:
    name, source_name = candidate_name(name), candidate_name(source_name)
    if name == source_name:
        raise RebuildError('source_and_target_must_differ')
    directory = directory or release_dir(name)
    source_manifest = source_manifest or release_dir(source_name) / 'release-manifest.json'
    if directory.is_symlink() or directory.name != release_dir(name).name:
        raise RebuildError('release_directory_identity_mismatch')
    manifest = directory / 'release-manifest.json'
    plan = {'schemaVersion': SCHEMA, 'dryRun': True, 'remoteExecuted': False, 'paidVisitorRuns': 0,
            'method': 'manifest_file_equivalent_rebuild', 'candidate': name, 'sourceCandidate': source_name,
            'releaseDirectory': str(directory), 'sourceManifest': str(source_manifest),
            'packageReady': manifest.is_file(), 'originalTarUploadVerified': False}
    if not manifest.is_file():
        return plan
    current, previous = manifest_rows(manifest), manifest_rows(source_manifest)
    for rel, row in current.items():
        path = regular_within(directory, rel)
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise RebuildError('local_release_file_hash_mismatch')
    changed = changed_paths(current, previous)
    plan.update(manifestSha256=digest(manifest), sourceManifestSha256=digest(source_manifest),
                filesVerified=len(current), bytesVerified=sum(r['bytes'] for r in current.values()),
                changedFiles=sorted(changed), changedBytes=sum(current[r]['bytes'] for r in changed),
                copiedFileCount=len(current)-len(changed))
    return plan


def _load_stage(root: Path, helper_sha: str):
    if not isinstance(helper_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', helper_sha):
        raise RebuildError('invalid_stage_helper_hash')
    helper = regular_within(root, f'candidate-validator-{helper_sha[:16]}.py')
    if digest(helper) != helper_sha:
        raise RebuildError('stage_helper_hash_mismatch')
    spec = importlib.util.spec_from_file_location('candidate_stage', helper)
    stage = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = stage
    spec.loader.exec_module(stage)
    return stage


def remote_rebuild(root: Path, source: Path, manifest_sha: str, source_manifest_sha: str, helper_sha: str) -> dict:
    for path in (root, source):
        if (path.parent != PARENT or path.resolve() != path or path.is_symlink() or not path.is_dir()
                or path.name != candidate_name(path.name)):
            raise RebuildError('candidate_root_outside_fixed_parent')
    if root == source or (root/'release-manifest.json').exists() or (root/'candidate-compatibility.json').exists():
        raise RebuildError('candidate_already_reconstructed')
    new_manifest = regular_within(root, 'rebuild-input/manifest.json')
    current = manifest_rows(new_manifest, manifest_sha)
    previous = manifest_rows(source/'release-manifest.json', source_manifest_sha)
    changed = changed_paths(current, previous)
    stage = _load_stage(root, helper_sha)
    # Build only declared immutable payload files. User store/cache/history and
    # native install side products in the prior candidate are not enumerated.
    for relative, row in current.items():
        source_file = regular_within(root/'rebuild-input/files', relative) if relative in changed else regular_within(source, relative)
        if source_file.stat().st_size != row['bytes'] or digest(source_file) != row['sha256']:
            raise RebuildError('rebuild_source_hash_mismatch')
        target = root.joinpath(*safe_relative(relative).parts)
        if target.exists() or target.is_symlink() or target.resolve() != target or not target.resolve().is_relative_to(root):
            raise RebuildError('rebuild_target_already_exists_or_outside_root')
        target.parent.mkdir(parents=True, exist_ok=True)
        with source_file.open('rb') as handle, target.open('xb') as output:
            shutil.copyfileobj(handle, output, 1024 * 1024)
    with new_manifest.open('rb') as handle, (root/'release-manifest.json').open('xb') as output:
        shutil.copyfileobj(handle, output)
    verified = stage.verify_manifest_tree(root, manifest_sha)
    rebuild = {'method': 'manifest_file_equivalent_rebuild', 'sourceCandidate': source.name,
               'sourceManifestSha256': source_manifest_sha, 'manifestSha256': manifest_sha,
               'copiedFiles': len(current)-len(changed), 'uploadedChangedFiles': len(changed),
               'fullOriginalTarUploaded': False, 'allManifestFilesVerified': verified}
    with (root/'manifest-rebuild.json').open('x', encoding='utf8') as handle:
        json.dump(rebuild, handle, indent=2)
    def interrupted(_signal, _frame):
        raise RebuildError('candidate_validation_interrupted')
    for number in (signal.SIGTERM, signal.SIGINT, getattr(signal, 'SIGHUP', signal.SIGTERM)):
        signal.signal(number, interrupted)
    result = stage.remote_validate(root, None, expected_manifest_sha=manifest_sha)
    result.update(rebuild=rebuild, validationHelperSha256=helper_sha)
    with (root/'candidate-compatibility.json').open('x', encoding='utf8') as handle:
        json.dump(result, handle, indent=2)
    return result


def _connect(credentials: Path, known_hosts: Path):
    # Reuse the existing strict known-host/password loader. Imported only after
    # the caller explicitly chooses execution/collection, never for a plan.
    from scripts.candidate_visitor_smoke import _connect as connect
    return connect(credentials, known_hosts)


def _command(client, args, timeout=30):
    _, stdout, stderr = client.exec_command(' '.join(shlex.quote(str(arg)) for arg in args), timeout=timeout)
    body = stdout.read(1024*1024)
    stderr.read(65536)
    return stdout.channel.recv_exit_status(), body


ALLOCATE = """from pathlib import Path
import hashlib,sys
root,source=map(Path,sys.argv[1:3]);expected=sys.argv[3]
parent=Path('/opt/demos/inquiry-curator-candidates')
assert root.parent==source.parent==parent and root!=source
assert parent.resolve()==parent and not parent.is_symlink()
assert source.resolve()==source and source.is_dir() and not source.is_symlink()
manifest=source/'release-manifest.json';assert not manifest.is_symlink()
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==expected
root.mkdir(exist_ok=False)
"""


def execute_plan(plan: dict, credentials: Path, known_hosts: Path, report_path: Path) -> dict:
    if plan.get('packageReady') is not True:
        raise RebuildError('release_package_not_ready')
    if report_path.exists() or report_path.is_symlink():
        raise RebuildError('local_report_already_exists')
    root = PurePosixPath(PARENT.as_posix()) / candidate_name(plan['candidate'])
    source = PurePosixPath(PARENT.as_posix()) / candidate_name(plan['sourceCandidate'])
    directory = Path(plan['releaseDirectory'])
    helper = ROOT/'scripts/stage_release_candidate.py'
    helper_sha, protocol_sha = digest(helper), digest(Path(__file__))
    protocol_name = f'candidate-rebuild-{protocol_sha[:16]}.py'
    client = _connect(credentials, known_hosts)
    try:
        code, _ = _command(client, ['python3', '-c', ALLOCATE, root, source, plan['sourceManifestSha256']])
        if code:
            raise RebuildError('fresh_candidate_allocation_failed')
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(90)
            made = {root}
            def upload(local, remote):
                parents = []
                parent = remote.parent
                while parent not in made:
                    if parent == parent.parent or not parent.is_relative_to(root):
                        raise RebuildError('upload_path_escape')
                    parents.append(parent)
                    parent = parent.parent
                for parent in reversed(parents):
                    sftp.mkdir(parent.as_posix())
                    made.add(parent)
                with sftp.open(remote.as_posix(), 'wx') as output, local.open('rb') as handle:
                    output.set_pipelined(True)
                    shutil.copyfileobj(handle, output, 1024*1024)
            upload(directory/'release-manifest.json', root/'rebuild-input/manifest.json')
            for relative in plan['changedFiles']:
                upload(regular_within(directory, relative), root/'rebuild-input/files'/safe_relative(relative))
            upload(helper, root/f'candidate-validator-{helper_sha[:16]}.py')
            upload(Path(__file__), root/protocol_name)
        bootstrap = "import hashlib,runpy,sys;from pathlib import Path;p=Path(sys.argv[1]);expected=sys.argv[2];assert p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==expected;sys.argv=sys.argv[1:2]+sys.argv[3:];runpy.run_path(str(p),run_name='__main__')"
        code, body = _command(client, ['/opt/demos/inquiry-curator/.venv/bin/python', '-B', '-c', bootstrap,
            root/protocol_name, protocol_sha, '--remote-run', root, '--source-root', source,
            '--manifest-sha256', plan['manifestSha256'], '--source-manifest-sha256', plan['sourceManifestSha256'],
            '--stage-helper-sha256', helper_sha], timeout=480)
        result = json.loads(body)
        validate_receipt(result, plan['candidate'], plan['manifestSha256'])
        if code:
            result['passed'] = False
        save_report(report_path, result)
        return result
    finally:
        client.close()


def validate_receipt(result: dict, name: str, manifest_sha: str | None = None):
    verification = result.get('artifactVerification') or {}
    if (result.get('schemaVersion') != 'isolated-server-candidate-v1' or result.get('releaseName') != candidate_name(name)
            or verification.get('mode') != 'manifest_file_equivalent_rebuild'
            or (manifest_sha is not None and verification.get('manifestSha256') != manifest_sha)):
        raise RebuildError('invalid_remote_compatibility_receipt')


def save_report(path: Path, result: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


def collect_only(name: str, credentials: Path, known_hosts: Path, report: Path) -> dict:
    name = candidate_name(name)
    if report.exists() or report.is_symlink():
        raise RebuildError('local_report_already_exists')
    client = _connect(credentials, known_hosts)
    try:
        remote = PurePosixPath(PARENT.as_posix()) / name / 'candidate-compatibility.json'
        with client.open_sftp() as sftp, sftp.open(remote.as_posix(), 'rb') as handle:
            result = json.loads(handle.read(1024*1024))
        validate_receipt(result, name)
        save_report(report, result)
        return result
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--collect-only', action='store_true')
    parser.add_argument('--candidate')
    parser.add_argument('--source-candidate')
    parser.add_argument('--release-dir', type=Path)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--credentials-file', type=Path, default=ROOT.parent/'服务器信息.txt')
    parser.add_argument('--known-hosts', type=Path, default=Path.home()/'.ssh/known_hosts')
    parser.add_argument('--remote-run', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--source-root', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--manifest-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--source-manifest-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--stage-helper-sha256', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.remote_run:
            result = remote_rebuild(args.remote_run, args.source_root, args.manifest_sha256,
                                    args.source_manifest_sha256, args.stage_helper_sha256)
        elif not args.candidate and not (args.execute or args.collect_only):
            result = {'schemaVersion': SCHEMA, 'dryRun': True, 'remoteExecuted': False,
                      'paidVisitorRuns': 0, 'candidateRequiredForExecution': True}
        else:
            name = candidate_name(args.candidate)
            report = args.report or ROOT/'artifacts/qa/server-candidates'/name/'compatibility.json'
            if args.collect_only:
                result = collect_only(name, args.credentials_file, args.known_hosts, report)
            else:
                plan = prepare_plan(name, args.source_candidate, args.release_dir, args.source_manifest)
                result = execute_plan(plan, args.credentials_file, args.known_hosts, report) if args.execute else plan
        print(json.dumps(result, ensure_ascii=False))
        return 1 if result.get('passed') is False else 0
    except Exception as error:
        print(json.dumps({'schemaVersion': SCHEMA, 'errorType': type(error).__name__, 'paidVisitorRuns': 0,
                          'errorCode': str(error) if isinstance(error, RebuildError) else 'operation_failed',
                          'boundary': 'No overwrite/retry. Use collect-only for uncertain remote completion.'}))
        return 1


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
