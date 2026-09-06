from dataclasses import replace

from app.config import Settings
from app.retrieval_runtime import build_collection_repository


def test_shared_factory_preserves_settings_and_bm25_disables_paid_providers(tmp_path):
    settings = replace(
        Settings(), collections_dir=tmp_path, rag_mode="bm25",
        rag_embedding_provider="aliyun", rag_embedding_model="qwen3.7-text-embedding",
        rag_embedding_dimension=768, rag_rerank_enabled=True,
    )
    repository = build_collection_repository(settings, collection_id="test")
    assert repository.rag_mode == "bm25"
    assert repository.reranker is None
    assert repository.default_collection_id == "test"
    assert repository.rerank_top_n == settings.rag_rerank_top_n


def test_shared_factory_uses_same_hosted_provider_for_serving_and_evaluation(tmp_path, monkeypatch):
    import app.retrieval_runtime as runtime
    captured = {}
    class FakeReranker:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setattr(runtime, "AliyunTextRetrievalProvider", FakeReranker)
    settings = replace(
        Settings(), collections_dir=tmp_path, rag_mode="hybrid",
        rag_embedding_provider="aliyun", rag_embedding_model="qwen3.7-text-embedding",
        rag_embedding_dimension=768, rag_rerank_enabled=True,
        rag_embedding_api_key="test-not-a-secret", rag_embedding_api_host="https://example.test",
    )
    repository = build_collection_repository(settings)
    assert repository.rag_mode == "hybrid"
    assert isinstance(repository.reranker, FakeReranker)
    assert captured["embedding_dimension"] == 768
    assert captured["rerank_model"] == settings.rag_rerank_model
