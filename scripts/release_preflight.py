"""Read-only deployment preflight; never embeds, generates, builds, or writes.

JSON on stdout deliberately contains no environment dump, credentials, request
headers, or upstream error bodies. Run on both build and serving hosts and
compare artifact hashes. Passing is a packaging/runtime gate, not a RAG quality
or browser acceptance claim. Network access occurs only with --health-url.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
LOCAL_BROWSER_URL = re.compile(rb"https?://(?:127\.0\.0\.1|localhost)(?=[:/\s\"'\\]|$)")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(check_id: str, passed: bool, **details) -> dict:
    return {"id": check_id, "status": "pass" if passed else "fail", **details}


def inspect_standalone(directory: Path) -> list[dict]:
    """Inspect a staged bundle, not the shared development build tree."""
    directory = directory.resolve()
    required = {"server.js": (directory / "server.js").is_file(),
                ".next/static": (directory / ".next/static").is_dir(),
                "public": (directory / "public").is_dir()}
    env_files = [path for path in directory.rglob(".env*") if path.is_file()]
    # Server-side Next internals legitimately contain loopback defaults. Only
    # publicly served JavaScript must be free of baked-in local API URLs.
    browser_files = list((directory / ".next/static").rglob("*.js"))
    browser_files.extend((directory / "public").rglob("*.js"))
    local_url_files = sum(bool(LOCAL_BROWSER_URL.search(path.read_bytes())) for path in browser_files)
    return [check("standalone_layout", all(required.values()), required=required),
            check("standalone_has_no_env_files", not env_files, fileCount=len(env_files)),
            check("browser_bundle_has_no_loopback_urls", bool(browser_files) and not local_url_files,
                  inspectedFiles=len(browser_files), matchingFiles=local_url_files)]


def inspect_health(url: str, expected_mode: str) -> dict:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        return check("running_health", False, errorCode="health_url_must_be_http_without_credentials_or_query")
    try:
        with urlopen(url, timeout=10) as response:
            payload = json.loads(response.read(1024 * 1024))
        retrieval = payload.get("retrieval", {})
        details = {key: retrieval.get(key) for key in (
            "mode", "available", "candidateAvailable", "collectionId", "collectionVersion", "fingerprint", "model", "version",
        )}
        # `available` describes the dense index, not baseline BM25. Shadow
        # separately exposes whether both candidate indexes are available.
        ready = bool(retrieval.get("collectionId")) and (
            expected_mode == "bm25"
            or (retrieval.get("candidateAvailable") is True if expected_mode == "shadow" else retrieval.get("available") is True)
        )
        return check("running_health", payload.get("status") == "ok" and retrieval.get("mode") == expected_mode
                     and ready, retrieval=details,
                     configured={key: payload.get(key) for key in ("modelConfigured", "posterModelConfigured", "ttsModelConfigured")})
    except Exception as error:
        # An error's text may contain a URL, key, proxy credential or response
        # body. The exception class is sufficient for this read-only gate.
        return check("running_health", False, errorCode=type(error).__name__)


def inspect_collection_and_indexes(settings) -> list[dict]:
    from app.collections import CollectionRepository
    from app.dense_retrieval import DenseIndexManager, EmbeddingProviderSpec
    from app.structured_filters import EVIDENCE_SEARCH_FTS5, OBJECT_SEARCH_FTS5, StructuredFilterIndex

    reports = []
    # This repository only parses frozen metadata. It never calls search or
    # the provider, and cannot implicitly rebuild the structured index.
    repo = CollectionRepository(settings.collections_dir, settings.default_collection_id,
                                rag_mode="bm25", structured_filters_enabled=False)
    collection = repo.get(settings.default_collection_id)
    manifest_path = settings.collections_dir / collection.id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reports.append(check("collection_frozen_identity", collection.objects_sha256 == manifest.get("objectsSha256")
                         and len(collection.objects) == manifest.get("objectCount"),
                         collectionId=collection.id, collectionVersion=collection.version,
                         objectCount=len(collection.objects), objectsSha256=collection.objects_sha256))
    if settings.rag_structured_filters_enabled:
        try:
            index = StructuredFilterIndex.for_collection(settings.rag_filter_index_dir, collection)
            reports.append(check("structured_index", index.object_search_mode == OBJECT_SEARCH_FTS5 and index.evidence_search_mode == EVIDENCE_SEARCH_FTS5,
                                 fingerprint=index.manifest["fingerprint"],
                                 databaseSha256=index.manifest["databaseSha256"],
                                 formatVersion=index.manifest["formatVersion"],
                                 objectSearchMode=index.object_search_mode, evidenceSearchMode=index.evidence_search_mode))
        except Exception as error:
            reports.append(check("structured_index", False, errorCode=type(error).__name__))
    else:
        reports.append(check("structured_index", False, errorCode="structured_filters_disabled"))
    if settings.rag_mode not in {"hybrid", "shadow"}:
        reports.append({"id": "dense_index", "status": "not_checked", "reason": "runtime_mode_is_bm25"})
        return reports
    if settings.rag_embedding_provider != "aliyun":
        reports.append(check("dense_index", False, errorCode="server_release_requires_hosted_embeddings"))
        return reports
    spec = EmbeddingProviderSpec(provider=settings.rag_embedding_provider, model_name=settings.rag_embedding_model,
                                 dimension=settings.rag_embedding_dimension,
                                 query_instruct=settings.rag_embedding_query_instruct,
                                 api_host=settings.rag_embedding_api_host, api_key=settings.rag_embedding_api_key)
    # Dummy provider is never invoked; use the actual runtime loader's source,
    # recipe, model, dimension, manifest and shape validation without any API.
    manager = DenseIndexManager(enabled=True, index_root=settings.rag_index_dir,
                                model_cache_dir=settings.rag_model_cache_dir,
                                model_name=settings.rag_embedding_model, provider_spec=spec,
                                provider_factory=lambda _allow_download: object())
    index = manager.get(collection)
    if index is None:
        reports.append(check("dense_index", False, errorCode="missing_stale_or_invalid_index"))
    else:
        artifact_hashes = {path.name: file_sha256(path) for path in sorted(index.path.iterdir()) if path.is_file()}
        reports.append(check("dense_index", True, fingerprint=index.manifest["fingerprint"],
                             objectCount=index.manifest["objectCount"], evidenceCount=index.manifest["evidenceCount"],
                             model=index.manifest["model"], dimension=index.manifest["dimension"],
                             artifactSha256=artifact_hashes,
                             boundary="Runtime identity/shape verified; compare these artifact hashes between build and serving host to verify transfer bytes."))
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-mode", choices=("bm25", "shadow", "hybrid"), default="hybrid")
    parser.add_argument("--health-url", help="Optional existing API /health URL; performs GET only, no model calls")
    parser.add_argument("--standalone-dir", type=Path, help="Optional already-staged .next/standalone tree")
    parser.add_argument("--require-cloud-assets", action="store_true", help="Require configured poster and TTS credentials and hosts")
    args = parser.parse_args()
    reports = [check("python", sys.version_info >= (3, 10), version=".".join(map(str, sys.version_info[:3])))]
    packages = {name: importlib.util.find_spec(module) is not None for name, module in (
        ("FastAPI", "fastapi"), ("httpx", "httpx"), ("Pydantic", "pydantic"), ("python-dotenv", "dotenv"),
        ("uvicorn", "uvicorn"), ("Pillow", "PIL"), ("numpy", "numpy"), ("OpenCC", "opencc"),
    )}
    reports.append(check("python_dependencies", all(packages.values()), available=packages))
    try:
        node = subprocess.run(["node", "--version"], check=True, text=True, capture_output=True, timeout=10).stdout.strip()
        reports.append(check("node", bool(re.fullmatch(r"v\d+\.\d+\.\d+", node)) and int(node[1:].split(".")[0]) >= 20, version=node))
    except Exception as error:
        reports.append(check("node", False, errorCode=type(error).__name__))
    sys.path.insert(0, str(ROOT / "api"))
    try:
        from app.config import Settings
        settings = Settings.from_env()
        reports.append(check("runtime_mode", settings.rag_mode == args.expect_mode,
                             expected=args.expect_mode, configured=settings.rag_mode,
                             embeddingProvider=settings.rag_embedding_provider,
                             rerankEnabled=settings.rag_rerank_enabled,
                             auditEnabled=settings.rag_llm_audit_enabled))
        reports.append(check("text_model_configured", bool(settings.deepseek_api_key), model=settings.deepseek_model,
                             labelsModel=settings.deepseek_labels_model))
        if args.expect_mode in {"hybrid", "shadow"}:
            reports.append(check("hybrid_services_configured", bool(settings.rag_embedding_api_key and settings.rag_embedding_api_host
                                 and settings.rag_rerank_enabled and settings.rag_llm_audit_enabled)))
        if args.require_cloud_assets:
            reports.append(check("cloud_assets_configured", bool(settings.aliyun_image_api_key and settings.aliyun_image_api_host
                                 and settings.aliyun_tts_api_key and settings.aliyun_tts_api_host)))
        reports.append(check("stage_budget", True, retrievalSeconds=settings.rag_retrieval_timeout_seconds,
                             frameSeconds=settings.deepseek_frame_timeout_seconds, labelsSeconds=settings.deepseek_labels_timeout_seconds,
                             posterWaitSeconds=settings.generation_poster_wait_seconds,
                             jobSeconds=settings.generation_job_timeout_seconds,
                             boundary="Settings validates configured ceilings; not a measured latency guarantee."))
        reports.extend(inspect_collection_and_indexes(settings))
    except Exception as error:
        reports.append(check("settings_or_collection", False, errorCode=type(error).__name__,
                             action="Inspect configuration validation locally; raw exception text is suppressed to protect secrets."))
    if args.standalone_dir:
        try:
            reports.extend(inspect_standalone(args.standalone_dir))
        except Exception as error:
            reports.append(check("standalone_inspection", False, errorCode=type(error).__name__))
    if args.health_url:
        reports.append(inspect_health(args.health_url, args.expect_mode))
    passed = all(row["status"] != "fail" for row in reports)
    print(json.dumps({"schemaVersion": "release-preflight/v1", "checkedAt": datetime.now(timezone.utc).isoformat(),
                      "passed": passed, "readOnly": True, "paidProviderCalls": 0, "checks": reports,
                      "notEstablished": ["semantic RAG quality", "actual provider connectivity or model permissions",
                                         "poster/TTS output or font readiness", "browser 3D/2D/audio usability",
                                         "server memory under concurrent visitors", "nginx admin and media-route behavior"]},
                     ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
