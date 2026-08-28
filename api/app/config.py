from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .dense_retrieval import DEFAULT_EMBEDDING_MODEL


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
    # Hybrid retrieval is local and keyless.  Serving only loads a versioned
    # cache built explicitly by scripts/build_dense_index.py; otherwise it
    # logs the reason and preserves the BM25-only path.
    rag_mode: str = "hybrid"
    rag_embedding_model: str = DEFAULT_EMBEDDING_MODEL
    rag_index_dir: Path = PROJECT_ROOT / "api" / "runtime" / "cache" / "rag"
    rag_model_cache_dir: Path = (
        PROJECT_ROOT / "api" / "runtime" / "cache" / "fastembed"
    )
    rag_dense_top_k: int = 200
    rag_dense_min_score: float = 0.28
    rag_evidence_min_score: float = 0.30
    rag_rrf_k: int = 60
    rag_max_results: int = 250
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-v4-flash"
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
    deepseek_frame_timeout_seconds: float = 90.0
    deepseek_labels_timeout_seconds: float = 45.0
    generation_job_timeout_seconds: float = 180.0
    # Poster generation starts after the final title is known.  It may use this
    # short grace period after text is ready, then continues in the background
    # instead of preventing an otherwise valid exhibition from opening.
    generation_poster_wait_seconds: float = 20.0
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

    @classmethod
    def from_env(cls) -> "Settings":
        store_mode = os.getenv("STORE_MODE", "json").strip().lower()
        if store_mode not in {"memory", "json"}:
            raise ValueError("STORE_MODE must be either 'memory' or 'json'")
        rag_mode = os.getenv("RAG_MODE", "hybrid").strip().lower()
        if rag_mode not in {"bm25", "hybrid"}:
            raise ValueError("RAG_MODE must be either 'bm25' or 'hybrid'")
        rag_embedding_model = (
            os.getenv("RAG_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        ).strip()
        if rag_embedding_model != DEFAULT_EMBEDDING_MODEL:
            raise ValueError(
                "RAG_EMBEDDING_MODEL is pinned for hybrid-rag-v1; rebuild and "
                "version the retrieval policy before changing models"
            )

        image_api_host = (os.getenv("ALIYUN_IMAGE_API_HOST") or "").rstrip("/") or None
        tts_api_host = (os.getenv("ALIYUN_TTS_API_HOST") or "").rstrip("/") or image_api_host

        deepseek_timeout_seconds = _bounded_float_env(
            "DEEPSEEK_TIMEOUT_SECONDS", 90.0, 1.0, 300.0
        )
        deepseek_frame_timeout_seconds = _bounded_float_env(
            "DEEPSEEK_FRAME_TIMEOUT_SECONDS", 90.0, 1.0, 180.0
        )
        deepseek_labels_timeout_seconds = _bounded_float_env(
            "DEEPSEEK_LABELS_TIMEOUT_SECONDS", 45.0, 1.0, 180.0
        )
        generation_job_timeout_seconds = _bounded_float_env(
            "GENERATION_JOB_TIMEOUT_SECONDS", 180.0, 30.0, 600.0
        )
        generation_poster_wait_seconds = _bounded_float_env(
            "GENERATION_POSTER_WAIT_SECONDS", 20.0, 0.0, 120.0
        )
        # Keep a deterministic reserve for retrieval, validation and store
        # writes. Labels run concurrently, so only one label budget is counted.
        generation_budget = (
            min(deepseek_timeout_seconds, deepseek_frame_timeout_seconds)
            + min(deepseek_timeout_seconds, deepseek_labels_timeout_seconds)
            + generation_poster_wait_seconds
            + 10.0
        )
        if generation_budget >= generation_job_timeout_seconds:
            raise ValueError(
                "Generation stage timeouts must leave at least 10 seconds before "
                "GENERATION_JOB_TIMEOUT_SECONDS"
            )

        return cls(
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
            rag_embedding_model=rag_embedding_model,
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
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
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
        )
