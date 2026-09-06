"""Build a frozen dense index for hybrid museum-object retrieval.

This is deliberately an operator command, never an API startup side effect.
The local baseline downloads FastEmbed once. The hosted Qwen path reads its
credential from ``DASHSCOPE_API_KEY`` and sends only deterministic retrieval
documents to the configured Beijing Model Studio workspace. Both paths write
the same resumable, fingerprint-addressed local matrices.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "api"))

from app.collections import CollectionRepository  # noqa: E402
from app.config import Settings  # noqa: E402
from app.dense_retrieval import EmbeddingProviderSpec, build_dense_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    defaults = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default=defaults.default_collection_id or "global_open",
        help="collection manifest id (default: configured collection)",
    )
    parser.add_argument(
        "--provider",
        choices=("local", "aliyun"),
        default=defaults.rag_embedding_provider,
        help="embedding provider",
    )
    parser.add_argument(
        "--model",
        default=defaults.rag_embedding_model,
        help="embedding model name",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=defaults.rag_embedding_dimension,
        help="hosted embedding dimension (ignored by the local baseline)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=defaults.rag_index_dir,
        help="versioned index root",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=defaults.rag_model_cache_dir,
        help="local FastEmbed model cache",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--parallelism",
        type=int,
        default=8,
        help="hosted request workers (local embeddings always use one)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace this exact fingerprint-addressed cache if it exists",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "resume a preserved .<fingerprint>.* staging directory; completed "
            "unit-normalised rows are verified before encoding continues"
        ),
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if not 1 <= args.parallelism <= 32:
        raise SystemExit("--parallelism must be between 1 and 32")
    index_dir = args.index_dir.resolve()
    model_cache_dir = args.model_cache_dir.resolve()
    free_bytes = shutil.disk_usage(index_dir.parent if index_dir.parent.exists() else PROJECT_ROOT).free
    if free_bytes < 700 * 1024 * 1024:
        raise SystemExit("at least 700 MB of free disk is required for model and index caches")

    repository = CollectionRepository(
        PROJECT_ROOT / "data" / "collections",
        default_collection_id=args.collection,
        rag_mode="bm25",
    )
    collection = repository.get(args.collection)
    defaults = Settings.from_env()
    provider_spec = EmbeddingProviderSpec(
        provider=args.provider,
        model_name=args.model,
        dimension=(args.dimension if args.provider == "aliyun" else None),
        query_instruct=defaults.rag_embedding_query_instruct,
        api_host=defaults.rag_embedding_api_host,
        api_key=defaults.rag_embedding_api_key,
        timeout_seconds=defaults.rag_embedding_timeout_seconds,
        max_attempts=defaults.rag_embedding_max_attempts,
    )
    started = time.perf_counter()
    output = build_dense_index(
        collection,
        index_root=index_dir,
        model_cache_dir=model_cache_dir,
        model_name=args.model,
        provider_spec=provider_spec,
        batch_size=args.batch_size,
        parallelism=args.parallelism,
        force=args.force,
        resume_from=args.resume_from,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    bytes_written = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(
        json.dumps(
            {
                "status": "ready",
                "path": str(output),
                "collection": manifest["collectionId"],
                "collectionVersion": manifest["collectionVersion"],
                "fingerprint": manifest["fingerprint"],
                "model": manifest["model"],
                "provider": manifest.get("provider", args.provider),
                "dimension": manifest["dimension"],
                "objectCount": manifest["objectCount"],
                "evidenceCount": manifest["evidenceCount"],
                "bytes": bytes_written,
                "seconds": round(time.perf_counter() - started, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
