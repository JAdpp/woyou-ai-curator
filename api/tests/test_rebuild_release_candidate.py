from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import rebuild_release_candidate as rebuild
from scripts import stage_release_candidate as stage

SOURCE = 'inquiry-v6-20260906-rc6'
TARGET = 'inquiry-v6-20260906-rc7'


def package(root, *, change=False):
    payload = {name: name.encode() for name in rebuild.REQUIRED}
    payload['api/app/new_helper.py'] = b'new' if change else b'old'
    rows = []
    for name, body in payload.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        rows.append({'path': name, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()})
    manifest = root / 'release-manifest.json'
    manifest.write_text(json.dumps({'schemaVersion': 'inquiry-release-package-v1', 'packageVerified': True,
                                    'files': rows}), encoding='utf8')
    return manifest


def test_default_cli_has_no_remote_side_effect(monkeypatch, capsys):
    monkeypatch.setattr(rebuild.sys, 'argv', ['rebuild'])
    monkeypatch.setattr(rebuild, '_connect', lambda *a: pytest.fail('no default SSH'))
    assert rebuild.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result['dryRun'] and not result['remoteExecuted'] and result['paidVisitorRuns'] == 0


def test_unfrozen_new_release_plan_does_not_create_package_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(rebuild, 'ROOT', tmp_path)
    result = rebuild.prepare_plan(TARGET, SOURCE)
    assert result['packageReady'] is False and result['dryRun']
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('name', ['../rc7', '/opt/demos/inquiry-curator', 'inquiry-v6-20260906-rc0',
                                   'inquiry-v6-20260906-rc7/x', 'inquiry-v6-20260906-rc7;echo'])
def test_arbitrary_candidate_names_rejected(name):
    with pytest.raises(rebuild.RebuildError, match='invalid_candidate_name'):
        rebuild.prepare_plan(name, SOURCE)


def test_target_must_not_reuse_source():
    with pytest.raises(rebuild.RebuildError, match='must_differ'):
        rebuild.prepare_plan(SOURCE, SOURCE)


def test_plan_checks_all_current_bytes_not_only_changed_files(tmp_path, monkeypatch):
    monkeypatch.setattr(rebuild, 'ROOT', tmp_path)
    package(rebuild.release_dir(SOURCE))
    package(rebuild.release_dir(TARGET), change=True)
    plan = rebuild.prepare_plan(TARGET, SOURCE)
    assert plan['changedFiles'] == ['api/app/new_helper.py']
    assert plan['filesVerified'] == 6 and plan['copiedFileCount'] == 5
    (rebuild.release_dir(TARGET)/'api/app/main.py').write_bytes(b'tampered')
    with pytest.raises(rebuild.RebuildError, match='local_release_file_hash_mismatch'):
        rebuild.prepare_plan(TARGET, SOURCE)


@pytest.mark.parametrize('path', ['../escape', '/escape', 'a\\b', './a', 'a//b', '.env', 'api/.env.local',
                                   'api/runtime/store.json', 'public/generated/poster.png', 'artifacts/qa/result.json'])
def test_manifest_rejects_escape_and_private_payload(path):
    with pytest.raises(rebuild.RebuildError):
        rebuild.safe_relative(path)


def test_duplicate_manifest_file_rejected(tmp_path):
    manifest = package(tmp_path)
    raw = json.loads(manifest.read_text())
    raw['files'].append(raw['files'][0])
    manifest.write_text(json.dumps(raw))
    with pytest.raises(rebuild.RebuildError, match='invalid_manifest_row'):
        rebuild.manifest_rows(manifest)


def setup_remote(tmp_path, monkeypatch):
    parent = tmp_path/'candidates'
    source, target = parent/SOURCE, parent/TARGET
    old_manifest = package(source)
    local = tmp_path/'local'
    new_manifest = package(local, change=True)
    changed_dir = target/'rebuild-input/files/api/app'
    changed_dir.mkdir(parents=True)
    (changed_dir/'new_helper.py').write_bytes(b'new')
    (target/'rebuild-input/manifest.json').write_bytes(new_manifest.read_bytes())
    # Prior candidate user state MUST NOT be copied even though it exists.
    (source/'candidate-state').mkdir()
    (source/'candidate-state/store.json').write_bytes(b'private history')
    monkeypatch.setattr(rebuild, 'PARENT', parent)
    monkeypatch.setattr(rebuild.signal, 'signal', lambda *a: None)
    calls = []
    def validate(root, archive_sha, **kwargs):
        calls.append(root)
        return {'schemaVersion': 'isolated-server-candidate-v1', 'releaseName': root.name, 'passed': True,
                'artifactVerification': stage.verify_manifest_tree(root, kwargs['expected_manifest_sha'])}
    monkeypatch.setattr(rebuild, '_load_stage', lambda *a: SimpleNamespace(
        verify_manifest_tree=stage.verify_manifest_tree, remote_validate=validate))
    return source, target, rebuild.digest(new_manifest), rebuild.digest(old_manifest), calls


def test_remote_rebuild_full_new_manifest_copy_is_not_hardlink_and_never_copies_old_store(tmp_path, monkeypatch):
    source, target, current, old, calls = setup_remote(tmp_path, monkeypatch)
    before = (source/'api/app/main.py').read_bytes()
    result = rebuild.remote_rebuild(target, source, current, old, 'a'*64)
    assert result['passed'] and result['rebuild']['copiedFiles'] == 5
    assert result['rebuild']['uploadedChangedFiles'] == 1
    assert result['artifactVerification']['filesVerified'] == 6
    assert result['artifactVerification']['originalTarUploadVerified'] is False
    assert calls == [target]
    assert not (target/'candidate-state/store.json').exists()
    (target/'api/app/main.py').write_bytes(b'change isolated copy')
    assert (source/'api/app/main.py').read_bytes() == before
    with pytest.raises(rebuild.RebuildError, match='already_reconstructed'):
        rebuild.remote_rebuild(target, source, current, old, 'a'*64)


@pytest.mark.parametrize('which', ['changed', 'copied', 'source_manifest'])
def test_remote_rebuild_rejects_tampered_sources_before_native_or_model(tmp_path, monkeypatch, which):
    source, target, current, old, calls = setup_remote(tmp_path, monkeypatch)
    path = {'changed': target/'rebuild-input/files/api/app/new_helper.py',
            'copied': source/'api/app/main.py', 'source_manifest': source/'release-manifest.json'}[which]
    path.write_bytes(b'tampered')
    with pytest.raises(rebuild.RebuildError):
        rebuild.remote_rebuild(target, source, current, old, 'a'*64)
    assert not calls and not (target/'candidate-compatibility.json').exists()


def test_remote_existing_file_is_preserved_and_not_overwritten(tmp_path, monkeypatch):
    source, target, current, old, calls = setup_remote(tmp_path, monkeypatch)
    path = target/'api/app/main.py'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'preserve')
    with pytest.raises(rebuild.RebuildError, match='already_exists'):
        rebuild.remote_rebuild(target, source, current, old, 'a'*64)
    assert path.read_bytes() == b'preserve' and not calls


def test_new_package_identity_cannot_be_claimed_for_old_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(rebuild, 'ROOT', tmp_path)
    package(rebuild.release_dir(SOURCE))
    with pytest.raises(rebuild.RebuildError, match='identity_mismatch'):
        rebuild.prepare_plan(TARGET, SOURCE, rebuild.release_dir(SOURCE))


def test_missing_package_execute_cannot_connect(tmp_path, monkeypatch):
    monkeypatch.setattr(rebuild, '_connect', lambda *a: pytest.fail('must not connect'))
    with pytest.raises(rebuild.RebuildError, match='not_ready'):
        rebuild.execute_plan({'packageReady': False}, tmp_path/'unread', tmp_path/'unread2', tmp_path/'report')


def test_existing_report_cannot_trigger_remote_execution(tmp_path, monkeypatch):
    report = tmp_path/'report';report.write_text('preserve')
    monkeypatch.setattr(rebuild, '_connect', lambda *a: pytest.fail('must not connect'))
    with pytest.raises(rebuild.RebuildError, match='already_exists'):
        rebuild.execute_plan({'packageReady': True}, tmp_path/'unread', tmp_path/'unread2', report)
    assert report.read_text() == 'preserve'


def test_receipt_requires_matching_candidate_and_new_manifest():
    raw = {'schemaVersion': 'isolated-server-candidate-v1', 'releaseName': TARGET,
           'artifactVerification': {'mode': 'manifest_file_equivalent_rebuild', 'manifestSha256': 'a'*64}}
    rebuild.validate_receipt(raw, TARGET, 'a'*64)
    with pytest.raises(rebuild.RebuildError): rebuild.validate_receipt(raw, SOURCE, 'a'*64)
    with pytest.raises(rebuild.RebuildError): rebuild.validate_receipt(raw, TARGET, 'b'*64)


def test_collect_only_reads_receipt_never_executes_remote_command(tmp_path, monkeypatch):
    receipt = {'schemaVersion': 'isolated-server-candidate-v1', 'releaseName': TARGET, 'passed': True,
               'artifactVerification': {'mode': 'manifest_file_equivalent_rebuild', 'manifestSha256': 'a'*64}}
    class SFTP:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def open(self, name, mode):
            assert name.endswith(TARGET+'/candidate-compatibility.json') and mode == 'rb'
            return io.BytesIO(json.dumps(receipt).encode())
    client = SimpleNamespace(open_sftp=SFTP, close=lambda: None)
    monkeypatch.setattr(rebuild, '_connect', lambda *a: client)
    monkeypatch.setattr(rebuild, '_command', lambda *a, **k: pytest.fail('collection must not execute'))
    result = rebuild.collect_only(TARGET, tmp_path/'unread', tmp_path/'unread2', tmp_path/'receipt.json')
    assert result == receipt


def test_regular_source_rejects_symlink_without_relying_on_windows_symlink_privileges(tmp_path, monkeypatch):
    file = tmp_path/'source';file.write_bytes(b'content')
    original = Path.is_symlink
    monkeypatch.setattr(Path, 'is_symlink', lambda p: p == file or original(p))
    with pytest.raises(rebuild.RebuildError, match='symlink_source_forbidden'):
        rebuild.regular_within(tmp_path, 'source')


def test_execute_uploads_only_delta_into_new_target_then_hash_bound_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(rebuild, 'ROOT', tmp_path)
    package(rebuild.release_dir(SOURCE))
    package(rebuild.release_dir(TARGET), change=True)
    helper = tmp_path/'scripts/stage_release_candidate.py'
    helper.parent.mkdir(parents=True)
    helper.write_bytes(b'fixture helper only')
    plan = rebuild.prepare_plan(TARGET, SOURCE)
    uploaded, commands, closed = {}, [], []
    class RemoteFile(io.BytesIO):
        def __init__(self, name): super().__init__();self.name=name
        def set_pipelined(self, enabled): assert enabled is True
        def close(self): uploaded[self.name]=self.getvalue();super().close()
    class SFTP:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get_channel(self): return SimpleNamespace(settimeout=lambda value: None)
        def mkdir(self, name):
            assert TARGET in name and SOURCE not in name
        def open(self, name, mode):
            assert mode == 'wx' and TARGET in name and SOURCE not in name
            return RemoteFile(name)
    client = SimpleNamespace(open_sftp=SFTP, close=lambda: closed.append(True))
    monkeypatch.setattr(rebuild, '_connect', lambda *a: client)
    def command(_client, args, **kwargs):
        commands.append(args)
        if len(commands) == 1:
            assert args[2] == rebuild.ALLOCATE
            assert args[-1] == plan['sourceManifestSha256']
            return 0, b''
        assert '--remote-run' in args and '--source-root' in args and '-B' in args
        assert args[args.index('--manifest-sha256')+1] == plan['manifestSha256']
        assert args[args.index('--stage-helper-sha256')+1] == rebuild.digest(helper)
        return 0, json.dumps({'schemaVersion': 'isolated-server-candidate-v1', 'releaseName': TARGET,
            'passed': True, 'artifactVerification': {'mode': 'manifest_file_equivalent_rebuild',
                                                    'manifestSha256': plan['manifestSha256']}}).encode()
    monkeypatch.setattr(rebuild, '_command', command)
    result = rebuild.execute_plan(plan, tmp_path/'unread', tmp_path/'unread2', tmp_path/'report.json')
    assert result['passed'] and len(commands) == 2 and closed == [True]
    assert len(uploaded) == 4  # manifest, one changed payload, validator, protocol
    assert any(name.endswith('/rebuild-input/files/api/app/new_helper.py') for name in uploaded)
    assert not any(name.endswith('/api/app/main.py') for name in uploaded)
    assert (tmp_path/'report.json').is_file()


def test_remote_manifest_has_all_required_size_hash_checks_before_copy(tmp_path, monkeypatch):
    source, target, current, old, calls = setup_remote(tmp_path, monkeypatch)
    new = target/'rebuild-input/manifest.json'
    raw = json.loads(new.read_text())
    raw['files'][0]['bytes'] = True
    new.write_text(json.dumps(raw))
    with pytest.raises(rebuild.RebuildError, match='invalid_manifest_row'):
        rebuild.remote_rebuild(target, source, rebuild.digest(new), old, 'a'*64)
    assert not calls
