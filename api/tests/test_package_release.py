from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import package_release as release


def put(root: Path, relative: str, content: str = "fixture") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    files = [".next/standalone/server.js", ".next/standalone/package.json",
             ".next/static/chunks/app.js", "public/brand/logo.svg", "api/app/main.py",
             "api/requirements.txt", "src/app/page.tsx", "deploy/nginx/demo.conf",
             "package.json", "package-lock.json", "next.config.ts", "tsconfig.json"]
    for name in files:
        put(root, name)
    put(root, ".next/standalone/node_modules/sharp/package.json", json.dumps({
        "name": "sharp", "version": "0.35.3", "optionalDependencies": {
            "@img/sharp-linux-x64": "0.35.3", "@img/sharp-libvips-linux-x64": "1.3.2"}}))
    for name in release.RELEASE_SCRIPTS:
        put(root, f"scripts/{name}")
    for name in release.COLLECTION_FILES:
        put(root, f"data/collections/global_open/{name}")
    index = put(root, "api/runtime/cache/rag/global_open/qwen/current/manifest.json", "{}")
    monkeypatch.setattr(release, "select_indexes", lambda selected_root: (
        [(index, index.relative_to(root).as_posix())], {"embedding": {"fingerprint": "fixture"}}))
    return root


def test_new_package_has_inventory_and_complete_standalone_layout(workspace, tmp_path):
    target = tmp_path / "release"
    entries, manifest = release.build_inventory(workspace, target)
    assert not target.exists()  # inventory is read-only
    assert manifest["fileCount"] == len(entries)
    assert manifest["totalBytes"] == sum(row.bytes for row in entries)
    assert manifest["nativeRuntime"]["prepared"] is False
    packaged = release.write_package(target, entries, manifest)
    assert packaged["packageVerified"] is True
    assert (target / ".next/standalone/.next/static/chunks/app.js").is_file()
    assert (target / ".next/standalone/public/brand/logo.svg").is_file()
    assert (target / "release-manifest.json").is_file()
    assert all(release.sha256_file(target / row.path) == row.sha256 for row in entries)


def test_secrets_user_state_raw_imports_generated_media_and_windows_native_are_not_packaged(workspace, tmp_path):
    sensitive = [".next/standalone/.env", ".next/standalone/.env.production",
                 ".next/standalone/node_modules/example/.env.sample", ".next/standalone/.npmrc",
                 ".next/standalone/node_modules/@img/sharp-win32-x64/lib/example.node",
                 "public/generated/posters/prior-user.jpg", "public/generated/audio/prior-user.mp3",
                 "api/runtime/store.json", "api/runtime/cache/images/private.jpg",
                 "api/runtime/logs/private.log", "data/collections/global_open/raw/raw.json",
                 "api/app/__pycache__/main.pyc", "deploy/private.key"]
    for path in sensitive:
        put(workspace, path, "sensitive fixture")
    entries, _ = release.build_inventory(workspace, tmp_path / "release")
    paths = [row.path for row in entries]
    assert not any(".env" in path or "store.json" in path or "raw.json" in path
                   or "generated/" in path or "win32" in path or "private" in path
                   or "__pycache__" in path for path in paths)
    assert all((workspace / path).exists() for path in sensitive)  # no source cleanup


def test_only_selected_complete_indexes_are_included(workspace, tmp_path):
    put(workspace, "api/runtime/cache/rag/global_open/qwen/.tmp-old/partial.npy")
    put(workspace, "api/runtime/cache/rag/global_open/minilm/old/weights.bin")
    put(workspace, "api/runtime/cache/filters/global_open/stale/filters.sqlite3")
    entries, _ = release.build_inventory(workspace, tmp_path / "release")
    indexes = [row.path for row in entries if row.section == "completed_index"]
    assert indexes == ["api/runtime/cache/rag/global_open/qwen/current/manifest.json"]


def test_existing_output_or_source_overlap_is_rejected_before_writing(workspace, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = put(existing, "keep.txt", "do not alter")
    with pytest.raises(release.PackageError, match="already_exists"):
        release.build_inventory(workspace, existing)
    assert marker.read_text() == "do not alter"
    with pytest.raises(release.PackageError, match="overlaps"):
        release.build_inventory(workspace, workspace / "api/app/new-release")


@pytest.mark.parametrize("url", ["http://localhost:8000/api", "https://127.0.0.1/api"])
def test_public_loopback_urls_fail_inventory(workspace, tmp_path, url):
    put(workspace, ".next/static/chunks/app.js", f'const api = "{url}";')
    with pytest.raises(release.PackageError, match="baked_loopback"):
        release.build_inventory(workspace, tmp_path / "release")


def test_server_loopback_default_is_not_mistaken_for_public_js(workspace, tmp_path):
    put(workspace, ".next/standalone/server.js", 'const hostname = "http://localhost:3000";')
    _, manifest = release.build_inventory(workspace, tmp_path / "release")
    assert manifest["checks"]["publicJavaScriptWithoutLoopback"] is True


def test_mutation_between_inventory_and_copy_is_detected_without_deleting_sources(workspace, tmp_path):
    target = tmp_path / "release"
    entries, manifest = release.build_inventory(workspace, target)
    put(workspace, "api/app/main.py", "changed after inventory")
    with pytest.raises(release.PackageError, match="changed_since_inventory"):
        release.write_package(target, entries, manifest)
    assert (workspace / "api/app/main.py").read_text() == "changed after inventory"
    assert not (target / "release-manifest.json").exists()


def test_unknown_native_binary_is_rejected(workspace, tmp_path):
    put(workspace, ".next/standalone/node_modules/unknown/native.node")
    with pytest.raises(release.PackageError, match="unaccounted_native"):
        release.build_inventory(workspace, tmp_path / "release")


def test_native_dependency_versions_come_from_exact_built_sharp_version(workspace, tmp_path):
    entries, manifest = release.build_inventory(workspace, tmp_path / "release")
    package = next(row for row in entries if row.path == "deploy/native-runtime/package.json")
    data = json.loads(package.content)
    assert data["dependencies"] == {"@img/sharp-linux-x64": "0.35.3", "@img/sharp-libvips-linux-x64": "1.3.2"}
    assert manifest["checks"]["targetLinuxRuntimeVerified"] is False
    source = workspace / ".next/standalone/node_modules/sharp/package.json"
    bad = json.loads(source.read_text())
    bad["optionalDependencies"]["@img/sharp-linux-x64"] = "latest"
    source.write_text(json.dumps(bad))
    with pytest.raises(release.PackageError, match="exactly_pinned"):
        release.native_runtime_plan(workspace)


def test_native_preparation_is_not_claimed_on_windows(workspace, monkeypatch):
    monkeypatch.setattr(release.platform, "system", lambda: "Windows")
    with pytest.raises(release.PackageError, match="requires_linux_x64"):
        release.prepare_linux_native(workspace)


def test_linux_native_preparation_copies_only_two_expected_packages_and_never_overwrites(tmp_path, monkeypatch):
    root = tmp_path / "release"
    packages = {"@img/sharp-linux-x64": "0.35.3", "@img/sharp-libvips-linux-x64": "1.3.2"}
    put(root, "release-manifest.json", json.dumps({"schemaVersion": release.PACKAGE_VERSION,
                                                  "packageVerified": True, "nativeRuntime": {"packages": packages}}))
    for package, version in packages.items():
        put(root, f"deploy/native-runtime/node_modules/{package}/package.json",
            json.dumps({"name": package, "version": version}))
        put(root, f"deploy/native-runtime/node_modules/{package}/lib/native.node")
    monkeypatch.setattr(release.platform, "system", lambda: "Linux")
    monkeypatch.setattr(release.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(release.platform, "libc_ver", lambda: ("glibc", "2.35"))
    result = release.prepare_linux_native(root)
    assert result["prepared"] is True and result["targetRuntimeVerified"] is False
    assert len(result["files"]) == 4
    with pytest.raises(release.PackageError, match="already_recorded"):
        release.prepare_linux_native(root)


def test_metadata_hashes_are_reproducible(workspace, tmp_path):
    first, _ = release.build_inventory(workspace, tmp_path / "release-a")
    second, _ = release.build_inventory(workspace, tmp_path / "release-b")
    assert [row.public() for row in first] == [row.public() for row in second]
