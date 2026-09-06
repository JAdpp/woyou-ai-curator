"""Create a new, secret-free release staging tree; never mutate source files.

--dry-run inventories/hashes sources without creating --output. An explicit
--report writes a new inventory report only. No mode contacts model providers.
Windows-built Next native modules are excluded; a Linux glibc x64 preparation
step remains mandatory and is NOT reported as target-runtime verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_VERSION = "inquiry-release-package-v1"
DENSE_FILES = ("manifest.json", "object_ids.json", "evidence_ids.json",
               "object_embeddings.npy", "evidence_embeddings.npy", "evidence_offsets.npy")
COLLECTION_FILES = ("manifest.json", "objects.json", "question_cards.json",
                    "regression_questions.json", "README.md", "rights_audit.md")
RELEASE_SCRIPTS = ("release_preflight.py", "package_release.py", "domains.py", "global_taxonomy.py")
LOCAL_BROWSER_URL = re.compile(rb"https?://(?:127\.0\.0\.1|localhost)(?=[:/\s\"'\\]|$)")


class PackageError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileEntry:
    path: str
    sha256: str
    bytes: int
    section: str
    source: Path | None = None
    content: bytes | None = None

    def public(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes, "section": self.section}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PackageError("invalid_json_object")
    return value


def _is_link(path: Path) -> bool:
    info = path.lstat()
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0)
                                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _excluded(relative: PurePosixPath, section: str) -> bool:
    parts = [part.casefold() for part in relative.parts]
    if any(part.startswith(".env") or part in {".git", ".npmrc", ".pypirc", "__pycache__", ".pytest_cache",
                                               "id_rsa", "id_ed25519", "id_ecdsa", "credentials.json"}
           for part in parts):
        return True
    if relative.suffix.casefold() in {".pyc", ".pyo", ".pem", ".key", ".p12", ".pfx"}:
        return True
    if section in {"public", "frontend_source"} and "generated" in parts:
        return True
    if section == "standalone" and (parts[:2] == ["public", "generated"]
                                     or parts[:2] == [".next", "cache"]):
        return True
    if any(part.startswith(("sharp-win32-", "sharp-darwin-", "swc-win32-", "swc-darwin-"))
           for part in parts):
        return True
    if section == "frontend_source" and ("tests" in parts or relative.name.endswith((".test.ts", ".test.tsx"))):
        return True
    return False


def _tree(source: Path, prefix: str, section: str) -> list[FileEntry]:
    if not source.is_dir() or _is_link(source):
        raise PackageError(f"required_source_directory_missing_or_link:{prefix}")
    result: list[FileEntry] = []
    for base, dirs, files in os.walk(source, followlinks=False):
        base_path = Path(base)
        retained = []
        for name in sorted(dirs):
            path = base_path / name
            relative = PurePosixPath(path.relative_to(source).as_posix())
            if _excluded(relative, section):
                continue
            if _is_link(path):
                raise PackageError("source_directory_link_not_supported")
            retained.append(name)
        dirs[:] = retained
        for name in sorted(files):
            path = base_path / name
            relative = PurePosixPath(path.relative_to(source).as_posix())
            if _excluded(relative, section):
                continue
            if _is_link(path) or not path.is_file():
                raise PackageError("source_file_link_not_supported")
            # Every native binary must be explicitly accounted for; do not
            # silently ship a platform-incompatible addon outside known sharp.
            if path.suffix.casefold() in {".node", ".dll", ".dylib", ".so"}:
                raise PackageError("unaccounted_native_binary_in_source")
            result.append(FileEntry(f"{prefix}/{relative}", sha256_file(path), path.stat().st_size,
                                    section, source=path))
    return result


def _file(source: Path, destination: str, section: str) -> FileEntry:
    if not source.is_file() or _is_link(source) or _excluded(PurePosixPath(destination), section):
        raise PackageError(f"required_source_file_missing_or_excluded:{destination}")
    return FileEntry(destination, sha256_file(source), source.stat().st_size, section, source=source)


def _generated(path: str, value: str, section: str = "release_metadata") -> FileEntry:
    encoded = value.encode("utf-8")
    return FileEntry(path, hashlib.sha256(encoded).hexdigest(), len(encoded), section, content=encoded)


def select_indexes(root: Path) -> tuple[list[tuple[Path, str]], dict[str, Any]]:
    """Use the runtime's read-only identity/shape loaders; never build/download."""

    sys.path.insert(0, str(root / "api"))
    from app.collections import CollectionRepository
    from app.dense_retrieval import DenseIndexManager, EmbeddingProviderSpec
    from app.structured_filters import StructuredFilterIndex, OBJECT_SEARCH_FTS5, EVIDENCE_SEARCH_FTS5

    collection_dir = root / "data/collections/global_open"
    manifest = _json(collection_dir / "manifest.json")
    actual_sha = sha256_file(collection_dir / "objects.json")
    if actual_sha != manifest.get("objectsSha256") or not manifest.get("servingReady"):
        raise PackageError("collection_snapshot_not_ready_or_hash_mismatch")
    repo = CollectionRepository(root / "data/collections", "global_open", rag_mode="bm25",
                                structured_filters_enabled=False)
    collection = repo.get("global_open")
    if len(collection.objects) != manifest.get("objectCount"):
        raise PackageError("collection_count_mismatch")
    filters = StructuredFilterIndex.for_collection(root / "api/runtime/cache/filters", collection)
    if filters.object_search_mode != OBJECT_SEARCH_FTS5 or filters.evidence_search_mode != EVIDENCE_SEARCH_FTS5:
        raise PackageError("structured_index_missing_fts5")
    # Runtime construction requires nonempty credentials, but neither host nor
    # key participates in its public fingerprint. Dummy values ensure packaging
    # never needs to read the user's .env and cannot accidentally call a host.
    spec = EmbeddingProviderSpec(provider="aliyun", model_name="qwen3.7-text-embedding", dimension=768,
                                 api_host="https://offline.invalid", api_key="offline-no-network")
    manager = DenseIndexManager(enabled=True, index_root=root / "api/runtime/cache/rag",
                                model_cache_dir=root / "api/runtime/cache/fastembed",
                                model_name=spec.model_name, provider_spec=spec,
                                provider_factory=lambda _allow_download: object())
    dense = manager.get(collection)
    if dense is None:
        raise PackageError("current_qwen_index_missing_or_inconsistent")
    sources = [(dense.path / name, (dense.path / name).relative_to(root).as_posix()) for name in DENSE_FILES]
    sources += [(filters.directory / name, (filters.directory / name).relative_to(root).as_posix())
                for name in ("manifest.json", "filters.sqlite3")]
    identity = {
        "collection": {key: manifest.get(key) for key in ("id", "version", "objectCount", "objectsSha256")},
        "embedding": {key: dense.manifest.get(key) for key in (
            "fingerprint", "provider", "model", "dimension", "queryInstructSha256", "textRecipeSha256",
            "textRecipeVersion", "objectCount", "evidenceCount")},
        "filters": {key: filters.manifest.get(key) for key in (
            "fingerprint", "formatVersion", "schemaVersion", "databaseSha256", "objectSearchMode", "evidenceSearchMode")},
    }
    return sources, identity


def native_runtime_plan(root: Path) -> dict[str, Any]:
    sharp = _json(root / ".next/standalone/node_modules/sharp/package.json")
    optional = sharp.get("optionalDependencies", {})
    packages = {key: optional.get(key) for key in ("@img/sharp-linux-x64", "@img/sharp-libvips-linux-x64")}
    if any(not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version)
           for version in packages.values()):
        raise PackageError("sharp_linux_dependencies_not_exactly_pinned")
    return {"target": "linux-x64-glibc", "sharpVersion": sharp["version"], "packages": packages,
            "prepared": False, "targetRuntimeVerified": False,
            "strategy": "Install these exact native packages in deploy/native-runtime on Linux; "
                        "copy only those packages into staged standalone via --prepare-linux-native; "
                        "then independently verify sharp load, Next server and API on that host."}


def _release_readme(native: dict[str, Any]) -> str:
    return f"""# 卧游发布包：待 Linux 运行时准备

本包由 Windows 已完成构建产生。代码、静态资源、馆藏与指定完成索引已清点；
这不是服务器已部署或跨平台运行通过的证明。Windows/macOS native 包已排除，
不得在准备 Linux sharp 依赖前直接切换现有服务。

## Linux glibc x64（不在旧服务目录执行）

在新版本暂存目录运行，Node 与项目版本兼容，系统需有中文字体和 Python 3.10+：

```bash
npm install --prefix deploy/native-runtime --ignore-scripts --no-audit --no-fund
python scripts/package_release.py --prepare-linux-native .
node -e "require('./.next/standalone/node_modules/sharp'); console.log('sharp loaded')"
```

sharp {native['sharpVersion']} 所需 native 包版本来自实际已构建 sharp 的 optionalDependencies，
不是安装最新版本。隔离 npm 目录生成的 package-lock.json 应另行保留并核对 registry integrity。
此准备操作不在项目根或 standalone 执行 npm install，不会 prune 已追踪的运行时文件。
准备器只允许 Linux glibc x64，拒绝覆盖已有目标 native 包；其他架构不能使用这条路径。

Python 用服务器隔离环境安装 api/requirements.txt；不复制 Windows 的 venv/模型缓存。
保留服务器现有 .env、store、generated、媒体目录；本包没有这些用户状态。
审核并通过独立新端口启动和 scripts/release_preflight.py 后才依既定授权切换服务。
预检还须检查海报目录映射、章节/TTS播放、3D/2D、内存与并发，不是只看首页 200。

## 更保守的跨平台备选

本包根目录另含 src、package-lock.json、package.json 与 Next/TypeScript 配置。
若目标 Linux native 路径失败，在足够内存的同架构 Linux 构建机执行 npm ci 和 npm run build，
重新形成 standalone 并补 public 与 .next/static；不要在低内存共享服务器直接跑 Next build。
重新构建的静态/服务文件需独立记录 hashes；不能沿用此 Windows 构建的 hash 清单。

## 清单边界

release-manifest.json 对每个负载文件记录相对路径、SHA-256、字节数、来源分区与模型指纹。
清单自身不做递归自哈希。Linux 准备步骤另生成 native-preparation.json 记录新增文件和锁文件 hash；
不得把 prepared 与 targetRuntimeVerified 混为一谈。未包含 raw 导入源、历史评测、图片/音频缓存、
用户 store、qrel 审阅、provider keys 或任何 .env*。
"""


def build_inventory(root: Path, output: Path) -> tuple[list[FileEntry], dict[str, Any]]:
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise PackageError("output_directory_already_exists")
    if output == root or any(output.is_relative_to(root / relative) for relative in (
        ".next", "api/app", "data", "public", "src", "scripts", "deploy", "node_modules")):
        raise PackageError("output_overlaps_release_sources")
    entries = _tree(root / ".next/standalone", ".next/standalone", "standalone")
    # Traced standalone may already include these directories after a previous
    # build; use the authoritative build/static and source/public copies once.
    entries = [row for row in entries if not row.path.startswith((
        ".next/standalone/.next/static/", ".next/standalone/public/"))]
    entries += _tree(root / ".next/static", ".next/standalone/.next/static", "static")
    entries += _tree(root / "public", ".next/standalone/public", "public")
    entries += _tree(root / "public", "public", "public")
    entries += _tree(root / "api/app", "api/app", "api_source")
    entries += _tree(root / "src", "src", "frontend_source")
    entries += _tree(root / "deploy", "deploy", "deploy")
    for relative in ("api/requirements.txt", "package.json", "package-lock.json", "next.config.ts", "tsconfig.json"):
        entries.append(_file(root / relative, relative, "source_config"))
    for relative in ("next-env.d.ts", "postcss.config.mjs", "postcss.config.js", "eslint.config.mjs"):
        if (root / relative).is_file():
            entries.append(_file(root / relative, relative, "source_config"))
    for name in RELEASE_SCRIPTS:
        entries.append(_file(root / "scripts" / name, f"scripts/{name}", "release_script"))
    for name in COLLECTION_FILES:
        path = f"data/collections/global_open/{name}"
        entries.append(_file(root / path, path, "collection"))
    indexes, identity = select_indexes(root)
    entries.extend(_file(source, target, "completed_index") for source, target in indexes)
    native = native_runtime_plan(root)
    entries += [
        _generated("deploy/native-runtime/package.json", json.dumps({
            "name": "inquiry-curator-linux-native-runtime", "version": "1.0.0", "private": True,
            "description": "Isolated native replacements only; do not install in standalone",
            "dependencies": native["packages"],
        }, indent=2) + "\n"),
        _generated("RELEASE_PACKAGE.md", _release_readme(native)),
    ]
    by_path: dict[str, FileEntry] = {}
    for entry in entries:
        if entry.path in by_path:
            raise PackageError("duplicate_release_target")
        by_path[entry.path] = entry
    required = (".next/standalone/server.js", ".next/standalone/package.json",
                "api/app/main.py", "api/requirements.txt", "scripts/release_preflight.py")
    if any(path not in by_path for path in required):
        raise PackageError("incomplete_release_components")
    browser_files = [row for row in entries if row.path.endswith(".js") and row.path.startswith((
        ".next/standalone/.next/static/", ".next/standalone/public/"))]
    if not browser_files:
        raise PackageError("no_public_javascript_to_inspect")
    if any(LOCAL_BROWSER_URL.search(row.source.read_bytes()) for row in browser_files if row.source):
        raise PackageError("public_javascript_contains_baked_loopback_url")
    entries.sort(key=lambda row: row.path)
    manifest = {"schemaVersion": PACKAGE_VERSION, "createdAt": datetime.now(timezone.utc).isoformat(),
                "sourcePlatform": platform.system(), "fileCount": len(entries),
                "totalBytes": sum(row.bytes for row in entries), "files": [row.public() for row in entries],
                "identity": identity, "nativeRuntime": native,
                "checks": {"sourceInventoryComplete": True, "publicJavaScriptWithoutLoopback": True,
                           "environmentFilesExcluded": True, "userStateExcluded": True,
                           "targetLinuxRuntimeVerified": False},
                "notIncluded": [".env*", "keys", "user store", "raw collection dumps", "historical evaluations",
                                "image/audio caches", "generated exhibits", "local MiniLM weights", "Windows native modules"]}
    return entries, manifest


def write_package(output: Path, entries: list[FileEntry], manifest: dict[str, Any]) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for row in entries:
        target = output / row.path
        if not target.resolve().is_relative_to(output):
            raise PackageError("invalid_relative_release_target")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as destination:
            if row.source:
                with row.source.open("rb") as source:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
            else:
                destination.write(row.content or b"")
        if target.stat().st_size != row.bytes or sha256_file(target) != row.sha256:
            # Leave the clearly unsuccessful new tree for inspection; never
            # delete sources or recursively clean an inferred destination.
            raise PackageError("copied_file_changed_since_inventory")
    from scripts.release_preflight import inspect_standalone
    checks = inspect_standalone(output / ".next/standalone")
    manifest = {**manifest, "stagedPreflight": checks,
                "packageVerified": all(row["status"] == "pass" for row in checks)}
    with (output / "release-manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if not manifest["packageVerified"]:
        raise PackageError("staged_preflight_failed")
    return manifest


def prepare_linux_native(release_root: Path) -> dict[str, Any]:
    """Copy exact Linux native packages after an explicit isolated npm install."""

    root = release_root.resolve()
    if platform.system() != "Linux" or platform.machine().casefold() not in {"x86_64", "amd64"}:
        raise PackageError("native_preparation_requires_linux_x64")
    if platform.libc_ver()[0].casefold() != "glibc":
        raise PackageError("native_preparation_requires_glibc")
    manifest = _json(root / "release-manifest.json")
    if manifest.get("schemaVersion") != PACKAGE_VERSION or not manifest.get("packageVerified"):
        raise PackageError("release_manifest_not_verified")
    packages = manifest["nativeRuntime"]["packages"]
    if set(packages) != {"@img/sharp-linux-x64", "@img/sharp-libvips-linux-x64"}:
        raise PackageError("unexpected_native_package")
    report = root / "native-preparation.json"
    if report.exists():
        raise PackageError("native_preparation_already_recorded")
    planned: list[tuple[Path, Path]] = []
    for package, version in packages.items():
        source = root / "deploy/native-runtime/node_modules" / package
        target = root / ".next/standalone/node_modules" / package
        if not source.resolve().is_relative_to(root) or not target.resolve().is_relative_to(root):
            raise PackageError("native_package_path_escapes_release")
        if not source.is_dir() or _is_link(source) or target.exists():
            raise PackageError("native_source_missing_or_target_exists")
        package_json = _json(source / "package.json")
        if package_json.get("name") != package or package_json.get("version") != version:
            raise PackageError("native_package_identity_mismatch")
        planned.append((source, target))
    files: list[dict[str, Any]] = []
    for source, target in planned:
        for base, dirs, names in os.walk(source, followlinks=False):
            if any(_is_link(Path(base) / name) for name in [*dirs, *names]):
                raise PackageError("native_package_link_not_supported")
            for name in sorted(names):
                path = Path(base) / name
                relative = path.relative_to(source)
                if _excluded(PurePosixPath(relative.as_posix()), "native"):
                    continue
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with path.open("rb") as reader, destination.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
                files.append({"path": destination.relative_to(root).as_posix(),
                              "sha256": sha256_file(destination), "bytes": destination.stat().st_size})
    lock = root / "deploy/native-runtime/package-lock.json"
    result = {"schemaVersion": "linux-native-preparation-v1", "prepared": True,
              "targetRuntimeVerified": False, "packages": packages, "files": files,
              "isolatedPackageLockSha256": sha256_file(lock) if lock.is_file() else None}
    with report.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New staging directory; must not already exist")
    parser.add_argument("--dry-run", action="store_true", help="Only hash/inventory; do not create --output")
    parser.add_argument("--report", type=Path, help="Optional NEW JSON inventory report (also allowed for dry-run)")
    parser.add_argument("--prepare-linux-native", type=Path, metavar="RELEASE_ROOT")
    args = parser.parse_args()
    try:
        if args.prepare_linux_native:
            if args.output or args.dry_run or args.report:
                raise PackageError("native_mode_cannot_be_combined_with_packaging")
            print(json.dumps(prepare_linux_native(args.prepare_linux_native), indent=2))
            return 0
        if not args.output:
            parser.error("--output is required unless --prepare-linux-native is used")
        if args.report and args.report.exists():
            raise PackageError("report_already_exists")
        entries, manifest = build_inventory(ROOT, args.output)
        if not args.dry_run:
            manifest = write_package(args.output, entries, manifest)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        summary = {key: value for key, value in manifest.items() if key != "files"}
        summary.update(dryRun=args.dry_run, paidProviderCalls=0, sourceMutation=False,
                       boundary="Package/native preparation is not target-Linux runtime or semantic quality verification")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        # Paths/errors may contain private data; don't dump source environment,
        # provider credentials, upstream bodies, or arbitrary exception text.
        code = str(error) if isinstance(error, PackageError) else type(error).__name__
        print(json.dumps({"schemaVersion": PACKAGE_VERSION, "passed": False,
                          "errorCode": code, "sourceMutation": False}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
