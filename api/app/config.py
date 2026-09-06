from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .dense_retrieval import (
    DEFAULT_ALIYUN_EMBEDDING_DIMENSION,
    DEFAULT_ALIYUN_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QUERY_INSTRUCT,
    MAX_EMBEDDING_PROVIDER_ATTEMPTS,
    SUPPORTED_ALIYUN_EMBEDDING_DIMENSIONS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(PROJECT_ROOT / "api" / ".env", override=False)


def _project_path(value: str | None, default: Path) -> Path:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _optional_project_path(value: str | None) -> Path | None:
    if not value or not value.strip():
        return None
    return _project_path(value.strip(), PROJECT_ROOT)


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _bounded_float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class Settings:
    app_env: str = "development"
    collections_dir: Path = PROJECT_ROOT / "data" / "collections"
    default_collection_id: str | None = None
    store_mode: str = "json"
    store_path: Path = PROJECT_ROOT / "api" / "runtime" / "store.json"
    image_cache_limit_mb: int = 256
    image_cache_dir: Path | None = None
    # Fail safe when no environment file is present. Hybrid and shadow are
    # explicit rollout choices because they may call a hosted embedding API;
    # a bare development process must retain the keyless BM25 baseline.
    rag_mode: str = "bm25"
    rag_embedding_provider: str = "local"
    rag_embedding_model: str = DEFAULT_EMBEDDING_MODEL
    rag_embedding_dimension: int | None = None
    rag_embedding_api_key: str | None = None
    rag_embedding_api_host: str | None = None
    rag_embedding_query_instruct: str = DEFAULT_QUERY_INSTRUCT
    rag_embedding_timeout_seconds: float = 20.0
    rag_embedding_max_attempts: int = 3
    rag_index_dir: Path = PROJECT_ROOT / "api" / "runtime" / "cache" / "rag"
    rag_model_cache_dir: Path = (
        PROJECT_ROOT / "api" / "runtime" / "cache" / "fastembed"
    )
    rag_dense_top_k: int = 200
    rag_dense_min_score: float = 0.28
    rag_evidence_min_score: float = 0.30
    rag_rrf_k: int = 60
    rag_max_results: int = 250
    rag_rerank_enabled: bool = False
    rag_rerank_model: str = "qwen3-rerank"
    rag_rerank_candidate_count: int = 60
    rag_rerank_top_n: int = 24
    rag_rerank_instruct: str = (
        "Given a museum collection question, rank collection records by "
        "whether institution-supplied evidence directly helps answer it. "
        "Topical similarity alone is insufficient."
    )
    # Rerank is an optional ordering aid; one short attempt preserves enough
    # of the shared deadline for the mandatory DeepSeek evidence audit.
    rag_rerank_timeout_seconds: float = 6.0
    rag_rerank_max_attempts: int = 1
    rag_trace_dir: Path = PROJECT_ROOT / "api" / "runtime" / "traces" / "retrieval"
    rag_structured_filters_enabled: bool = True
    rag_filter_index_dir: Path = (
        PROJECT_ROOT / "api" / "runtime" / "cache" / "filters"
    )
    rag_evidence_bm25_top_k: int = 120
    # A bounded model pass audits semantic candidates before object selection.
    # If fewer than the requested number survive, it may issue at most three
    # intent-preserving catalogue queries and audit the fused candidates once.
    rag_llm_audit_enabled: bool = True
    rag_llm_audit_top_k: int = 18
    # One wall-clock budget shared by initial hybrid recall, model relevance
    # audit, batched query expansion and the final audit.  A shared deadline
    # prevents each stage from silently receiving a fresh timeout.
    rag_retrieval_timeout_seconds: float = 70.0
    # A sub-budget of retrieval, not an additional generation-stage allowance.
    rag_planning_timeout_seconds: float = 16.0
    rag_llm_audit_timeout_seconds: float = 22.0
    rag_visual_audit_enabled: bool = True
    rag_visual_audit_top_k: int = 8
    rag_visual_audit_timeout_seconds: float = 14.0
    rag_agentic_max_queries: int = 5
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_query_review_thinking: bool = False
    # The public wall-label pass is visual.  Keeping its model separate means
    # interview, frame and epilogue text stay on the stable text model while
    # only the stage that actually needs pixels uses the experimental model.
    deepseek_labels_model: str = "deepseek-v4-flash-vision-exp"
    deepseek_base_url: str = "https://api.deepseek.com"
    # Curation asks a reasoning model for a whole exhibition in one call, so
    # the ceiling is well above a typical chat round-trip.
    deepseek_timeout_seconds: float = 90.0
    # The provider timeout is a hard wall-clock ceiling.  Profile generation
    # reserves separate portions for the frame and parallel label passes so a
    # slow upstream can fall back before the visitor-visible job deadline.
    deepseek_frame_timeout_seconds: float = 55.0
    deepseek_labels_timeout_seconds: float = 40.0
    generation_job_timeout_seconds: float = 180.0
    # Poster generation starts after the final title is known.  It may use this
    # short grace period after text is ready, then continues in the background
    # instead of preventing an otherwise valid exhibition from opening.
    generation_poster_wait_seconds: float = 2.0
    aliyun_image_api_key: str | None = None
    aliyun_image_api_host: str | None = None
    aliyun_image_model: str = "qwen-image-3.0-pro"
    # Horizontal entrance poster, matching the 16:9 provider instruction and
    # the exhibition hero in both the text and 3D views.
    aliyun_image_size: str = "1536*864"
    aliyun_image_timeout_seconds: float = 90.0
    aliyun_image_output_dir: Path = PROJECT_ROOT / "public" / "generated" / "posters"
    aliyun_image_font_path: Path | None = None
    # Qwen-Audio-TTS is a separate workload from poster generation.  It may
    # use the same Model Studio key and (for the local demo) can fall back to
    # the image workspace origin, but a dedicated TTS origin always wins.
    aliyun_tts_api_key: str | None = None
    aliyun_tts_api_host: str | None = None
    aliyun_tts_model: str = "qwen-audio-3.0-tts-plus"
    aliyun_tts_voice: str = "qwen-audio-3.0-tts-plus-longyulianrong"
    aliyun_tts_instruction: str = (
        "请使用专业、克制、清晰的博物馆导览播音主持声线，语速稍慢，停连自然，避免夸张表演。"
    )
    aliyun_tts_timeout_seconds: float = 90.0
    aliyun_tts_output_dir: Path = PROJECT_ROOT / "api" / "runtime" / "cache" / "audio-guides"
    aliyun_tts_max_audio_bytes: int = 20 * 1024 * 1024
    admin_review_token: str = "development-only-admin-token"
    editor_access_token: str | None = None
    # Human relevance judgements are a development-only, single-owner
    # workflow. They never modify the frozen v1 benchmark in place.
    qrel_review_dataset_dir: Path = (
        PROJECT_ROOT / "data" / "qa" / "retrieval_eval_v1"
    )
    qrel_review_db_path: Path = (
        PROJECT_ROOT
        / "api"
        / "runtime"
        / "qrel-reviews"
        / "retrieval_eval_v1"
        / "reviews.sqlite3"
    )
    qrel_suggestion_db_path: Path = (
        PROJECT_ROOT
        / "api"
        / "runtime"
        / "qrel-suggestions"
        / "retrieval_eval_v1"
        / "suggestions.sqlite3"
    )

    @classmethod
    def from_env(cls) -> "Settings":
        store_mode = os.getenv("STORE_MODE", "json").strip().lower()
        if store_mode not in {"memory", "json"}:
            raise ValueError("STORE_MODE must be either 'memory' or 'json'")
        rag_mode = os.getenv("RAG_MODE", "bm25").strip().lower()
        if rag_mode not in {"bm25", "hybrid", "shadow"}:
            raise ValueError("RAG_MODE must be 'bm25', 'hybrid', or 'shadow'")
        rag_embedding_provider = os.getenv(
            "RAG_EMBEDDING_PROVIDER", "local"
        ).strip().lower()
        if rag_embedding_provider not in {"local", "aliyun"}:
            raise ValueError("RAG_EMBEDDING_PROVIDER must be 'local' or 'aliyun'")
        rag_embedding_model = (
            os.getenv("RAG_EMBEDDING_MODEL")
            or (
                DEFAULT_ALIYUN_EMBEDDING_MODEL
                if rag_embedding_provider == "aliyun"
                else DEFAULT_EMBEDDING_MODEL
            )
        ).strip()
        if (
            rag_embedding_provider == "local"
            and rag_embedding_model != DEFAULT_EMBEDDING_MODEL
        ):
            raise ValueError(
                "The local embedding baseline is pinned to "
                f"{DEFAULT_EMBEDDING_MODEL}; use RAG_EMBEDDING_PROVIDER=aliyun "
                "for the hosted production model"
            )

        image_api_host = (os.getenv("ALIYUN_IMAGE_API_HOST") or "").rstrip("/") or None
        tts_api_host = (os.getenv("ALIYUN_TTS_API_HOST") or "").rstrip("/") or image_api_host
        text_api_host = (
            (os.getenv("ALIYUN_TEXT_API_HOST") or "").rstrip("/")
            or image_api_host
        )
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY") or None
        rag_embedding_dimension = (
            _positive_int_env(
                "RAG_EMBEDDING_DIMENSION",
                DEFAULT_ALIYUN_EMBEDDING_DIMENSION,
            )
            if rag_embedding_provider == "aliyun"
            else None
        )
        rag_embedding_query_instruct = os.getenv(
            "RAG_EMBEDDING_QUERY_INSTRUCT", DEFAULT_QUERY_INSTRUCT
        ).strip()
        rag_embedding_max_attempts = _positive_int_env(
            "RAG_EMBEDDING_MAX_ATTEMPTS", 3
        )
        if rag_embedding_provider == "aliyun" and rag_mode != "bm25":
            if rag_embedding_dimension not in SUPPORTED_ALIYUN_EMBEDDING_DIMENSIONS:
                raise ValueError("RAG_EMBEDDING_DIMENSION is not supported")
            if not rag_embedding_query_instruct:
                raise ValueError("RAG_EMBEDDING_QUERY_INSTRUCT must not be blank")
            if rag_embedding_max_attempts > MAX_EMBEDDING_PROVIDER_ATTEMPTS:
                raise ValueError(
                    "RAG_EMBEDDING_MAX_ATTEMPTS must be between one and five"
                )
            if not dashscope_api_key or not text_api_host:
                raise ValueError(
                    "Aliyun hybrid retrieval requires DASHSCOPE_API_KEY and "
                    "ALIYUN_TEXT_API_HOST (or ALIYUN_IMAGE_API_HOST fallback)"
                )

        deepseek_timeout_seconds = _bounded_float_env(
            "DEEPSEEK_TIMEOUT_SECONDS", 90.0, 1.0, 300.0
        )
        deepseek_frame_timeout_seconds = _bounded_float_env(
            "DEEPSEEK_FRAME_TIMEOUT_SECONDS", 55.0, 1.0, 180.0
        )
        deepseek_labels_timeout_seconds = _bounded_float_env(
            "DEEPSEEK_LABELS_TIMEOUT_SECONDS", 40.0, 1.0, 180.0
        )
        rag_llm_audit_enabled = _bool_env("RAG_LLM_AUDIT_ENABLED", True)
        rag_llm_audit_timeout_seconds = _bounded_float_env(
            "RAG_LLM_AUDIT_TIMEOUT_SECONDS", 22.0, 2.0, 60.0
        )
        rag_retrieval_timeout_seconds = _bounded_float_env(
            "RAG_RETRIEVAL_TIMEOUT_SECONDS", 70.0, 5.0, 120.0
        )
        rag_rerank_enabled = _bool_env("RAG_RERANK_ENABLED", False)
        rag_rerank_candidate_count = _positive_int_env(
            "RAG_RERANK_CANDIDATE_COUNT", 60
        )
        rag_rerank_top_n = _positive_int_env("RAG_RERANK_TOP_N", 24)
        rag_rerank_model = os.getenv("RAG_RERANK_MODEL", "qwen3-rerank").strip()
        rag_rerank_instruct = os.getenv(
            "RAG_RERANK_INSTRUCT",
            (
                "Given a museum collection question, rank collection records "
                "by whether institution-supplied evidence directly helps answer "
                "it. Topical similarity alone is insufficient."
            ),
        ).strip()
        rag_rerank_max_attempts = _positive_int_env("RAG_RERANK_MAX_ATTEMPTS", 1)
        if rag_rerank_candidate_count > 500:
            raise ValueError("RAG_RERANK_CANDIDATE_COUNT must not exceed 500")
        if rag_rerank_top_n > rag_rerank_candidate_count:
            raise ValueError(
                "RAG_RERANK_TOP_N must not exceed RAG_RERANK_CANDIDATE_COUNT"
            )
        if (
            rag_mode != "bm25"
            and rag_rerank_enabled
            and (not dashscope_api_key or not text_api_host)
        ):
            raise ValueError(
                "Qwen reranking requires DASHSCOPE_API_KEY and "
                "ALIYUN_TEXT_API_HOST (or ALIYUN_IMAGE_API_HOST fallback)"
            )
        if rag_mode != "bm25" and rag_rerank_enabled:
            if not rag_rerank_model:
                raise ValueError("RAG_RERANK_MODEL must not be blank")
            if not rag_rerank_instruct:
                raise ValueError("RAG_RERANK_INSTRUCT must not be blank")
            if rag_rerank_max_attempts > MAX_EMBEDDING_PROVIDER_ATTEMPTS:
                raise ValueError("RAG_RERANK_MAX_ATTEMPTS must be between one and five")
        generation_job_timeout_seconds = _bounded_float_env(
            "GENERATION_JOB_TIMEOUT_SECONDS", 180.0, 30.0, 600.0
        )
        generation_poster_wait_seconds = _bounded_float_env(
            "GENERATION_POSTER_WAIT_SECONDS", 2.0, 0.0, 120.0
        )
        # Older local configurations reserve longer frame/label/poster stages
        # and omit the retrieval setting. Do not break startup merely because
        # the new retrieval default grew. Only the *implicit* default may use
        # the remaining budget; an explicit over-budget value still errors.
        if not os.getenv("RAG_RETRIEVAL_TIMEOUT_SECONDS"):
            available_retrieval = (
                generation_job_timeout_seconds
                - min(deepseek_timeout_seconds, deepseek_frame_timeout_seconds)
                - min(deepseek_timeout_seconds, deepseek_labels_timeout_seconds)
                - generation_poster_wait_seconds - 11.0
            )
            rag_retrieval_timeout_seconds = min(
                rag_retrieval_timeout_seconds, max(5.0, available_retrieval)
            )
        # Keep a deterministic reserve for retrieval, validation and store
        # writes. Labels run concurrently, so only one label budget is counted.
        generation_budget = (
            min(deepseek_timeout_seconds, deepseek_frame_timeout_seconds)
            + min(deepseek_timeout_seconds, deepseek_labels_timeout_seconds)
            # Initial local retrieval runs even when the optional model audit
            # is disabled, so its wall-clock ceiling always belongs here.
            + rag_retrieval_timeout_seconds
            + generation_poster_wait_seconds
            + 10.0
        )
        if generation_budget >= generation_job_timeout_seconds:
            raise ValueError(
                "Generation stage timeouts must leave at least 10 seconds before "
                "GENERATION_JOB_TIMEOUT_SECONDS"
            )

        return cls(
            image_cache_dir=_optional_project_path(os.getenv("IMAGE_CACHE_DIR")),
            app_env=os.getenv("APP_ENV", "development").strip().lower(),
            collections_dir=_project_path(
                os.getenv("COLLECTIONS_DIR"), PROJECT_ROOT / "data" / "collections"
            ),
            default_collection_id=(os.getenv("DEFAULT_COLLECTION_ID") or "").strip()
            or None,
            store_mode=store_mode,
            store_path=_project_path(
                os.getenv("STORE_PATH"), PROJECT_ROOT / "api" / "runtime" / "store.json"
            ),
            image_cache_limit_mb=_positive_int_env("IMAGE_CACHE_LIMIT_MB", 256),
            rag_mode=rag_mode,
            rag_embedding_provider=rag_embedding_provider,
            rag_embedding_model=rag_embedding_model,
            rag_embedding_dimension=rag_embedding_dimension,
            rag_embedding_api_key=dashscope_api_key,
            rag_embedding_api_host=text_api_host,
            rag_embedding_query_instruct=rag_embedding_query_instruct,
            rag_embedding_timeout_seconds=_bounded_float_env(
                "RAG_EMBEDDING_TIMEOUT_SECONDS", 20.0, 1.0, 120.0
            ),
            rag_embedding_max_attempts=rag_embedding_max_attempts,
            rag_index_dir=_project_path(
                os.getenv("RAG_INDEX_DIR"),
                PROJECT_ROOT / "api" / "runtime" / "cache" / "rag",
            ),
            rag_model_cache_dir=_project_path(
                os.getenv("RAG_MODEL_CACHE_DIR"),
                PROJECT_ROOT / "api" / "runtime" / "cache" / "fastembed",
            ),
            rag_dense_top_k=_positive_int_env("RAG_DENSE_TOP_K", 200),
            rag_dense_min_score=_bounded_float_env(
                "RAG_DENSE_MIN_SCORE", 0.28, -1.0, 1.0
            ),
            rag_evidence_min_score=_bounded_float_env(
                "RAG_EVIDENCE_MIN_SCORE", 0.30, -1.0, 1.0
            ),
            rag_rrf_k=_positive_int_env("RAG_RRF_K", 60),
            rag_max_results=_positive_int_env("RAG_MAX_RESULTS", 250),
            rag_rerank_enabled=rag_rerank_enabled,
            rag_rerank_model=rag_rerank_model,
            rag_rerank_candidate_count=rag_rerank_candidate_count,
            rag_rerank_top_n=rag_rerank_top_n,
            rag_rerank_instruct=rag_rerank_instruct,
            rag_rerank_timeout_seconds=_bounded_float_env(
                "RAG_RERANK_TIMEOUT_SECONDS", 6.0, 1.0, 60.0
            ),
            rag_rerank_max_attempts=rag_rerank_max_attempts,
            rag_trace_dir=_project_path(
                os.getenv("RAG_TRACE_DIR"),
                PROJECT_ROOT / "api" / "runtime" / "traces" / "retrieval",
            ),
            rag_structured_filters_enabled=_bool_env(
                "RAG_STRUCTURED_FILTERS_ENABLED", True
            ),
            rag_filter_index_dir=_project_path(
                os.getenv("RAG_FILTER_INDEX_DIR"),
                PROJECT_ROOT / "api" / "runtime" / "cache" / "filters",
            ),
            rag_evidence_bm25_top_k=_positive_int_env(
                "RAG_EVIDENCE_BM25_TOP_K", 120
            ),
            rag_llm_audit_enabled=rag_llm_audit_enabled,
            rag_visual_audit_enabled=_bool_env("RAG_VISUAL_AUDIT_ENABLED", True),
            rag_visual_audit_top_k=min(12, _positive_int_env("RAG_VISUAL_AUDIT_TOP_K", 8)),
            rag_visual_audit_timeout_seconds=_bounded_float_env("RAG_VISUAL_AUDIT_TIMEOUT_SECONDS", 14.0, 1.0, 30.0),
            rag_llm_audit_top_k=_positive_int_env("RAG_LLM_AUDIT_TOP_K", 18),
            rag_retrieval_timeout_seconds=rag_retrieval_timeout_seconds,
            rag_planning_timeout_seconds=_bounded_float_env("RAG_PLANNING_TIMEOUT_SECONDS", 16.0, 2.0, 70.0),
            rag_llm_audit_timeout_seconds=rag_llm_audit_timeout_seconds,
            rag_agentic_max_queries=_positive_int_env(
                "RAG_AGENTIC_MAX_QUERIES", 5
            ),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            deepseek_query_review_thinking=_bool_env("DEEPSEEK_QUERY_REVIEW_THINKING", False),
            deepseek_labels_model=os.getenv(
                "DEEPSEEK_LABELS_MODEL", "deepseek-v4-flash-vision-exp"
            ),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            deepseek_timeout_seconds=deepseek_timeout_seconds,
            deepseek_frame_timeout_seconds=deepseek_frame_timeout_seconds,
            deepseek_labels_timeout_seconds=deepseek_labels_timeout_seconds,
            generation_job_timeout_seconds=generation_job_timeout_seconds,
            generation_poster_wait_seconds=generation_poster_wait_seconds,
            aliyun_image_api_key=os.getenv("DASHSCOPE_API_KEY") or None,
            aliyun_image_api_host=image_api_host,
            aliyun_image_model=os.getenv("ALIYUN_IMAGE_MODEL", "qwen-image-3.0-pro"),
            aliyun_image_size=os.getenv("ALIYUN_IMAGE_SIZE", "1536*864"),
            aliyun_image_timeout_seconds=float(
                os.getenv("ALIYUN_IMAGE_TIMEOUT_SECONDS", "90")
            ),
            aliyun_image_output_dir=_project_path(
                os.getenv("ALIYUN_IMAGE_OUTPUT_DIR"),
                PROJECT_ROOT / "public" / "generated" / "posters",
            ),
            aliyun_image_font_path=_optional_project_path(
                os.getenv("ALIYUN_IMAGE_FONT_PATH")
            ),
            aliyun_tts_api_key=os.getenv("DASHSCOPE_API_KEY") or None,
            aliyun_tts_api_host=tts_api_host,
            aliyun_tts_model=os.getenv(
                "ALIYUN_TTS_MODEL", "qwen-audio-3.0-tts-plus"
            ),
            aliyun_tts_voice=os.getenv(
                "ALIYUN_TTS_VOICE",
                "qwen-audio-3.0-tts-plus-longyulianrong",
            ),
            aliyun_tts_instruction=os.getenv(
                "ALIYUN_TTS_INSTRUCTION",
                "请使用专业、克制、清晰的博物馆导览播音主持声线，语速稍慢，停连自然，避免夸张表演。",
            ),
            aliyun_tts_timeout_seconds=float(
                os.getenv("ALIYUN_TTS_TIMEOUT_SECONDS", "90")
            ),
            aliyun_tts_output_dir=_project_path(
                os.getenv("ALIYUN_TTS_OUTPUT_DIR"),
                PROJECT_ROOT / "api" / "runtime" / "cache" / "audio-guides",
            ),
            aliyun_tts_max_audio_bytes=_positive_int_env(
                "ALIYUN_TTS_MAX_AUDIO_BYTES", 20 * 1024 * 1024
            ),
            admin_review_token=os.getenv(
                "ADMIN_REVIEW_TOKEN", "development-only-admin-token"
            ),
            editor_access_token=os.getenv("EDITOR_ACCESS_TOKEN") or None,
            qrel_review_dataset_dir=_project_path(
                os.getenv("QREL_REVIEW_DATASET_DIR"),
                PROJECT_ROOT / "data" / "qa" / "retrieval_eval_v1",
            ),
            qrel_review_db_path=_project_path(
                os.getenv("QREL_REVIEW_DB_PATH"),
                PROJECT_ROOT
                / "api"
                / "runtime"
                / "qrel-reviews"
                / "retrieval_eval_v1"
                / "reviews.sqlite3",
            ),
            qrel_suggestion_db_path=_project_path(
                os.getenv("QREL_SUGGESTION_DB_PATH"),
                PROJECT_ROOT
                / "api"
                / "runtime"
                / "qrel-suggestions"
                / "retrieval_eval_v1"
                / "suggestions.sqlite3",
            ),
        )
