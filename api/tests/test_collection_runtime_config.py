from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.collections import CollectionRepository
from app.config import Settings
from app.main import create_app


def _clone_collection(base: Path, collection_id: str) -> Path:
    source = base / "cma-chinese-art"
    target = base / collection_id
    target.mkdir()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["id"] = collection_id
    manifest["name"] = f"Fixture {collection_id}"
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    (target / "objects.json").write_bytes((source / "objects.json").read_bytes())
    return target


def test_repository_uses_explicit_default_and_invalidates_file_cache(
    collections_dir: Path,
) -> None:
    selected_dir = _clone_collection(collections_dir, "global-open")
    repository = CollectionRepository(
        collections_dir, default_collection_id="global-open"
    )

    first = repository.list()
    assert repository.list() is first
    assert repository.get().id == "global-open"

    manifest_path = selected_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["name"] = "Changed global fixture name"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    refreshed = repository.list()
    assert refreshed is not first
    assert repository.get().name == "Changed global fixture name"


def test_settings_read_collection_and_cache_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEFAULT_COLLECTION_ID", " global-open ")
    monkeypatch.setenv("IMAGE_CACHE_LIMIT_MB", "384")

    settings = Settings.from_env()

    assert settings.default_collection_id == "global-open"
    assert settings.image_cache_limit_mb == 384


def test_app_wires_default_collection_and_cache_limit(collections_dir: Path) -> None:
    _clone_collection(collections_dir, "global-open")
    settings = Settings(
        app_env="test",
        collections_dir=collections_dir,
        default_collection_id="global-open",
        store_mode="memory",
        store_path=collections_dir.parent / "unused-store.json",
        image_cache_limit_mb=7,
    )

    app = create_app(settings)

    assert app.state.collections.get().id == "global-open"
    assert app.state.image_cache.limit_bytes == 7 * 1024 * 1024


def test_image_cache_limit_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_CACHE_LIMIT_MB", "0")

    with pytest.raises(ValueError, match="IMAGE_CACHE_LIMIT_MB"):
        Settings.from_env()


def test_local_baseline_rejects_unversioned_embedding_model_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "some/other-model")

    with pytest.raises(ValueError, match="local embedding baseline is pinned"):
        Settings.from_env()


def test_aliyun_embedding_and_shadow_mode_are_versioned_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_MODE", "shadow")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "aliyun")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "qwen3.7-text-embedding")
    monkeypatch.setenv("RAG_EMBEDDING_DIMENSION", "768")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only-key")
    monkeypatch.setenv(
        "ALIYUN_TEXT_API_HOST",
        "https://fixture.cn-beijing.maas.aliyuncs.com",
    )

    settings = Settings.from_env()

    assert settings.rag_mode == "shadow"
    assert settings.rag_embedding_provider == "aliyun"
    assert settings.rag_embedding_model == "qwen3.7-text-embedding"
    assert settings.rag_embedding_dimension == 768


def test_bm25_mode_does_not_construct_aliyun_retrieval_clients(
    collections_dir: Path,
) -> None:
    settings = Settings(
        app_env="test",
        collections_dir=collections_dir,
        store_mode="memory",
        store_path=collections_dir.parent / "unused-store.json",
        rag_mode="bm25",
        rag_embedding_provider="aliyun",
        rag_embedding_model="qwen3.7-text-embedding",
        rag_embedding_dimension=768,
        rag_embedding_api_key=None,
        rag_embedding_api_host=None,
        rag_rerank_enabled=True,
    )

    app = create_app(settings)

    assert app.state.collections.reranker is None
    assert app.state.collections._dense_manager.enabled is False
    assert app.state.collections._dense_manager.provider_spec.provider == "local"


def test_bm25_environment_allows_dormant_aliyun_flags_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_MODE", "bm25")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "aliyun")
    monkeypatch.setenv("RAG_RERANK_ENABLED", "true")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    monkeypatch.setenv("ALIYUN_TEXT_API_HOST", "")
    monkeypatch.setenv("ALIYUN_IMAGE_API_HOST", "")

    settings = Settings.from_env()

    assert settings.rag_embedding_api_key is None
    assert settings.rag_embedding_api_host is None
    assert settings.rag_rerank_enabled is True


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("RAG_EMBEDDING_DIMENSION", "999", "DIMENSION is not supported"),
        ("RAG_EMBEDDING_MAX_ATTEMPTS", "6", "between one and five"),
        ("RAG_EMBEDDING_QUERY_INSTRUCT", "", "INSTRUCT must not be blank"),
    ],
)
def test_enabled_aliyun_embedding_validates_provider_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv("RAG_MODE", "hybrid")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "aliyun")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only-key")
    monkeypatch.setenv(
        "ALIYUN_TEXT_API_HOST",
        "https://fixture.cn-beijing.maas.aliyuncs.com",
    )
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        Settings.from_env()


def test_enabled_reranker_validates_retry_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_MODE", "hybrid")
    monkeypatch.setenv("RAG_RERANK_ENABLED", "true")
    monkeypatch.setenv("RAG_RERANK_MAX_ATTEMPTS", "6")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only-key")
    monkeypatch.setenv(
        "ALIYUN_TEXT_API_HOST",
        "https://fixture.cn-beijing.maas.aliyuncs.com",
    )

    with pytest.raises(ValueError, match="RAG_RERANK_MAX_ATTEMPTS"):
        Settings.from_env()


def test_generation_budget_counts_retrieval_when_llm_audit_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_LLM_AUDIT_ENABLED", "false")
    monkeypatch.setenv("RAG_RETRIEVAL_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("DEEPSEEK_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("DEEPSEEK_FRAME_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("DEEPSEEK_LABELS_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("GENERATION_POSTER_WAIT_SECONDS", "2")
    monkeypatch.setenv("GENERATION_JOB_TIMEOUT_SECONDS", "150")

    with pytest.raises(ValueError, match="Generation stage timeouts"):
        Settings.from_env()
