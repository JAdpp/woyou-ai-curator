"""One retrieval runtime constructor shared by serving and evaluation."""
from __future__ import annotations

from typing import Any

from .collections import CollectionRepository
from .config import Settings
from .dense_retrieval import EmbeddingProviderSpec
from .providers.aliyun_text_retrieval import AliyunTextRetrievalProvider


def build_collection_repository(
    settings: Settings,
    *,
    rag_mode: str | None = None,
    collection_id: str | None = None,
    trace_writer: Any | None = None,
) -> CollectionRepository:
    mode = rag_mode or settings.rag_mode
    semantic = mode in {"hybrid", "shadow"}
    spec = (
        EmbeddingProviderSpec(
            provider=settings.rag_embedding_provider,
            model_name=settings.rag_embedding_model,
            dimension=settings.rag_embedding_dimension,
            api_key=settings.rag_embedding_api_key,
            api_host=settings.rag_embedding_api_host,
            query_instruct=settings.rag_embedding_query_instruct,
            timeout_seconds=settings.rag_embedding_timeout_seconds,
            max_attempts=settings.rag_embedding_max_attempts,
        ) if semantic else EmbeddingProviderSpec()
    )
    reranker = (
        AliyunTextRetrievalProvider(
            api_key=settings.rag_embedding_api_key or "",
            api_host=settings.rag_embedding_api_host or "",
            embedding_model=settings.rag_embedding_model,
            embedding_dimension=settings.rag_embedding_dimension or 768,
            query_instruct=settings.rag_embedding_query_instruct,
            rerank_model=settings.rag_rerank_model,
            rerank_instruct=settings.rag_rerank_instruct,
            timeout_seconds=settings.rag_rerank_timeout_seconds,
            max_attempts=settings.rag_rerank_max_attempts,
        ) if semantic and settings.rag_rerank_enabled else None
    )
    return CollectionRepository(
        settings.collections_dir,
        default_collection_id=collection_id or settings.default_collection_id,
        rag_mode=mode,
        dense_index_dir=settings.rag_index_dir,
        embedding_model_cache_dir=settings.rag_model_cache_dir,
        embedding_model=spec.model_name,
        embedding_provider_spec=spec,
        dense_top_k=settings.rag_dense_top_k,
        dense_min_score=settings.rag_dense_min_score,
        evidence_min_score=settings.rag_evidence_min_score,
        rrf_k=settings.rag_rrf_k,
        hybrid_max_results=settings.rag_max_results,
        reranker=reranker,
        rerank_candidate_count=settings.rag_rerank_candidate_count,
        rerank_top_n=settings.rag_rerank_top_n,
        rerank_instruct=settings.rag_rerank_instruct,
        trace_writer=trace_writer,
        structured_filters_enabled=settings.rag_structured_filters_enabled,
        structured_filter_index_dir=settings.rag_filter_index_dir,
        evidence_bm25_top_k=settings.rag_evidence_bm25_top_k,
    )
