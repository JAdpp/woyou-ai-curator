"""Versioned local dense-retrieval index for the hybrid collection search.

The serving API never downloads a model or builds an index implicitly.  An
operator builds the cache with ``scripts/build_dense_index.py``; the API then
loads it read-only and memory-maps the vectors.  A missing, stale, or corrupt
cache is therefore a safe and observable BM25 degradation, not an unexpected
network request during a visitor session.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Iterator, Sequence
from uuid import uuid4


logger = logging.getLogger("app.retrieval")

INDEX_FORMAT_VERSION = 1
TEXT_RECIPE_VERSION = "museum-object-and-evidence-v1"
TEXT_RECIPE_SHA256 = hashlib.sha256(
    (
        "object:title|title_original|creator|date|culture|place|type|material|"
        "classification|themes|tags|relation_facets|description|evidence;"
        "evidence:title|culture|type|chunk_text"
    ).encode("utf-8")
).hexdigest()
DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_MODEL_SOURCE = (
    "https://huggingface.co/sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_MODEL_RUNTIME_SOURCE = (
    "https://huggingface.co/qdrant/"
    "paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
)
DEFAULT_MODEL_LICENSE = "Apache-2.0"


class DenseRetrievalError(RuntimeError):
    """A build-time dense retrieval failure."""


@dataclass(frozen=True)
class DenseHit:
    object_id: str
    score: float


@dataclass(frozen=True)
class EvidenceHit:
    score: float
    evidence_ids: tuple[str, ...]
    # Per-chunk cosine lets the caller apply its raw-evidence gate before
    # treating a dense excerpt as query evidence. ``score`` remains the
    # unfiltered maximum for candidate-level thresholding/backward compatibility.
    evidence_scores: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class EvidenceDenseHit:
    object_id: str
    evidence_id: str
    score: float


@dataclass(frozen=True)
class DenseStatus:
    enabled: bool
    available: bool
    mode: str
    reason: str
    model: str
    fingerprint: str | None = None
    index_path: str | None = None


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    return compact[:96] or "embedding-model"


def _text(value: Any) -> str:
    return str(value or "").strip()


def object_embedding_text(obj: Any) -> str:
    """Build a stable, field-labelled document for object-level recall."""

    lines = [
        f"title: {_text(obj.title)}",
        f"original title: {_text(obj.title_original)}",
        f"creator: {_text(obj.creator)}",
        f"date: {_text(obj.date)}",
        f"culture: {_text(obj.culture)}",
        f"place: {_text(obj.place)}",
        f"type: {_text(obj.type)}",
        f"material: {_text(obj.material)}",
        f"classification: {_text(obj.classification)}",
        f"subjects: {'; '.join((*obj.themes, *obj.tags, *obj.relation_facets))}",
        f"institution description: {_text(obj.description)}",
    ]
    # Evidence is included for recall, while the separate evidence matrix lets
    # search verify that a dense-only candidate has a supporting excerpt.
    lines.extend(f"evidence: {_text(chunk.text)}" for chunk in obj.evidence)
    return "\n".join(line for line in lines if line.split(":", 1)[-1].strip())


def evidence_embedding_text(obj: Any, chunk: Any) -> str:
    """Embed individual institution excerpts with minimal object context."""

    return "\n".join(
        filter(
            None,
            (
                f"title: {_text(obj.title)}",
                f"culture: {_text(obj.culture)}",
                f"type: {_text(obj.type)}",
                f"evidence: {_text(chunk.text)}",
            ),
        )
    )


def collection_fingerprint(collection: Any, model_name: str) -> str:
    """Bind a cache to retrieval format, model, collection version and text."""

    digest = hashlib.sha256()
    digest.update(
        f"dense-index-v{INDEX_FORMAT_VERSION}\0{TEXT_RECIPE_VERSION}\0".encode()
    )
    digest.update(model_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(_text(collection.id).encode("utf-8"))
    digest.update(b"\0")
    digest.update(_text(collection.version).encode("utf-8"))
    declared_objects_sha = _text(getattr(collection, "objects_sha256", None))
    if declared_objects_sha:
        # Frozen serving corpora already publish a SHA-256 of objects.json.
        # Using it keeps the per-query cache signature check O(1).
        digest.update(b"\0objects-sha256\0")
        digest.update(declared_objects_sha.encode("ascii"))
        return digest.hexdigest()
    for obj in collection.objects:
        digest.update(b"\0object\0")
        digest.update(_text(obj.id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(object_embedding_text(obj).encode("utf-8"))
        for chunk in obj.evidence:
            digest.update(b"\0evidence\0")
            digest.update(_text(chunk.id).encode("utf-8"))
            digest.update(b"\0")
            digest.update(evidence_embedding_text(obj, chunk).encode("utf-8"))
    return digest.hexdigest()


class FastEmbedProvider:
    """Small adapter that keeps the optional dependency out of API startup."""

    def __init__(
        self,
        model_name: str,
        model_cache_dir: Path,
        *,
        allow_download: bool = False,
    ) -> None:
        if model_name != DEFAULT_EMBEDDING_MODEL:
            raise DenseRetrievalError(
                f"hybrid-rag-v2 supports only the pinned model {DEFAULT_EMBEDDING_MODEL!r}"
            )
        try:
            import fastembed
            import onnxruntime
            from fastembed import TextEmbedding
        except ImportError as error:  # pragma: no cover - exercised by deployment
            raise DenseRetrievalError(
                "fastembed is not installed; install api/requirements.txt"
            ) from error
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.fastembed_version = fastembed.__version__
        self.onnxruntime_version = onnxruntime.__version__
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._model = TextEmbedding(
                model_name=model_name,
                cache_dir=str(model_cache_dir),
                local_files_only=not allow_download,
            )
        except Exception as error:  # provider gives model/download-specific errors
            raise DenseRetrievalError(
                f"could not load local embedding model {model_name!r}: {error}"
            ) from error

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> Iterator[Any]:
        yield from self._model.embed(list(texts), batch_size=batch_size)

    def embed_query(self, text: str) -> Any:
        try:
            return next(iter(self._model.query_embed(text)))
        except AttributeError:
            return next(iter(self._model.embed([text])))

    def artifact_metadata(self) -> dict[str, str | None]:
        """Return the exact cached ONNX revision and model-file digest."""

        repository = self.model_cache_dir / (
            "models--qdrant--paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
        )
        revision: str | None = None
        try:
            revision = (repository / "refs" / "main").read_text(
                encoding="utf-8"
            ).strip()
        except OSError:
            pass
        candidates: list[Path] = []
        if revision:
            candidates.extend((repository / "snapshots" / revision).glob("*.onnx"))
        if not candidates:
            candidates.extend(repository.glob("snapshots/*/*.onnx"))
        artifact_sha: str | None = None
        if candidates:
            digest = hashlib.sha256()
            with min(candidates).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            artifact_sha = digest.hexdigest()
        return {
            "modelRevision": revision,
            "modelArtifactSha256": artifact_sha,
            "fastembedVersion": self.fastembed_version,
            "onnxruntimeVersion": self.onnxruntime_version,
        }


def _normalise(vector: Any) -> Any:
    import numpy as np

    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0 else array


def _batched(values: Sequence[str], batch_size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _write_matrix(
    path: Path,
    texts: Sequence[str],
    provider: FastEmbedProvider,
    *,
    batch_size: int,
    progress_label: str,
) -> int:
    """Write or resume a normalised float32 matrix without retaining it in RAM."""

    import numpy as np

    if not texts:
        raise DenseRetrievalError("cannot build a dense matrix from an empty corpus")
    if path.exists():
        matrix = np.load(path, mmap_mode="r+", allow_pickle=False)
        if (
            matrix.dtype != np.float32
            or matrix.ndim != 2
            or matrix.shape[0] != len(texts)
        ):
            raise DenseRetrievalError(
                f"resume matrix {progress_label} has an incompatible shape or dtype"
            )
        dimension = int(matrix.shape[1])
        written = 0
        # Builders write rows strictly in order. A completed row is unit
        # normalised; a torn final row or untouched preallocated row is not.
        # Resume from the first non-unit row and overwrite everything after it.
        for start in range(0, len(texts), 512):
            norms = np.linalg.norm(
                np.asarray(matrix[start : start + 512], dtype=np.float32),
                axis=1,
            )
            invalid = np.flatnonzero((norms < 0.98) | (norms > 1.02))
            if invalid.size:
                written = start + int(invalid[0])
                break
            written = min(start + 512, len(texts))
        logger.info(
            "dense index %s: resuming at %s/%s verified rows",
            progress_label,
            written,
            len(texts),
        )
    else:
        first_vector = list(provider.embed(texts[:1], 1))
        if len(first_vector) != 1:
            raise DenseRetrievalError("embedding provider returned no vectors")
        dimension = int(_normalise(first_vector[0]).shape[0])
        matrix = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(len(texts), dimension),
        )
        written = 0
    started = time.perf_counter()
    next_report = ((written // 5_000) + 1) * 5_000
    for batch in _batched(texts[written:], batch_size):
        for vector in provider.embed(batch, batch_size):
            if written >= len(texts):
                raise DenseRetrievalError("embedding provider returned extra vectors")
            normalised = _normalise(vector)
            if normalised.shape != (dimension,):
                raise DenseRetrievalError("embedding dimension changed during build")
            matrix[written] = normalised
            written += 1
        if written >= next_report or written == len(texts):
            # Periodic flush makes a killed build safely resumable.
            matrix.flush()
            logger.info(
                "dense index %s: %s/%s encoded (%.1fs)",
                progress_label,
                written,
                len(texts),
                time.perf_counter() - started,
            )
            while next_report <= written:
                next_report += 5_000
    if written != len(texts):
        raise DenseRetrievalError(
            f"embedding provider returned {written} vectors for {len(texts)} texts"
        )
    matrix.flush()
    del matrix
    return dimension


def build_dense_index(
    collection: Any,
    *,
    index_root: Path,
    model_cache_dir: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    force: bool = False,
    resume_from: Path | None = None,
) -> Path:
    """Build one immutable, fingerprint-addressed collection index."""

    fingerprint = collection_fingerprint(collection, model_name)
    target = (
        index_root
        / _slug(_text(collection.id))
        / _slug(model_name)
        / fingerprint
    )
    manifest_path = target / "manifest.json"
    if manifest_path.exists() and not force:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    if resume_from is not None:
        temporary = resume_from.resolve()
        if temporary.parent != target.parent.resolve():
            raise DenseRetrievalError(
                "resume staging directory must be inside this index fingerprint directory"
            )
        if not temporary.name.startswith(f".{fingerprint}."):
            raise DenseRetrievalError("resume staging directory fingerprint does not match")
        if not temporary.is_dir():
            raise DenseRetrievalError("resume staging directory does not exist")
    else:
        temporary = target.parent / f".{fingerprint}.tmp-{uuid4().hex}"
        temporary.mkdir(parents=False, exist_ok=False)

    provider = FastEmbedProvider(model_name, model_cache_dir, allow_download=True)
    completed = False
    try:
        objects = [obj for obj in collection.objects if obj.evidence]
        object_ids = [obj.id for obj in objects]
        object_texts = [object_embedding_text(obj) for obj in objects]
        evidence_ids: list[str] = []
        evidence_texts: list[str] = []
        evidence_offsets = [0]
        for obj in objects:
            for chunk in obj.evidence:
                evidence_ids.append(chunk.id)
                evidence_texts.append(evidence_embedding_text(obj, chunk))
            evidence_offsets.append(len(evidence_ids))

        state_path = temporary / "build_state.json"
        expected_state = {
            "formatVersion": INDEX_FORMAT_VERSION,
            "fingerprint": fingerprint,
            "collectionId": collection.id,
            "collectionVersion": collection.version,
            "objectsSha256": getattr(collection, "objects_sha256", None),
            "model": model_name,
            "textRecipeSha256": TEXT_RECIPE_SHA256,
            "objectCount": len(object_ids),
            "evidenceCount": len(evidence_ids),
        }
        if state_path.exists():
            try:
                observed_state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise DenseRetrievalError("resume build state is corrupt") from error
            if observed_state != expected_state:
                raise DenseRetrievalError("resume build state does not match collection/model")
        else:
            # A staging directory produced by the pre-resume builder has no
            # state file. The fingerprinted directory name plus matrix shape
            # checks below are the compatibility proof; write state now.
            state_path.write_text(
                json.dumps(expected_state, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        object_dimension = _write_matrix(
            temporary / "object_embeddings.npy",
            object_texts,
            provider,
            batch_size=batch_size,
            progress_label="objects",
        )
        evidence_dimension = _write_matrix(
            temporary / "evidence_embeddings.npy",
            evidence_texts,
            provider,
            batch_size=batch_size,
            progress_label="evidence",
        )
        if object_dimension != evidence_dimension:
            raise DenseRetrievalError("object and evidence embedding dimensions differ")

        import numpy as np

        np.save(
            temporary / "evidence_offsets.npy",
            np.asarray(evidence_offsets, dtype=np.int64),
            allow_pickle=False,
        )
        (temporary / "object_ids.json").write_text(
            json.dumps(object_ids, ensure_ascii=False), encoding="utf-8"
        )
        (temporary / "evidence_ids.json").write_text(
            json.dumps(evidence_ids, ensure_ascii=False), encoding="utf-8"
        )
        manifest = {
            "formatVersion": INDEX_FORMAT_VERSION,
            "textRecipeVersion": TEXT_RECIPE_VERSION,
            "textRecipeSha256": TEXT_RECIPE_SHA256,
            "collectionId": collection.id,
            "collectionVersion": collection.version,
            "objectsSha256": getattr(collection, "objects_sha256", None),
            "fingerprint": fingerprint,
            "model": model_name,
            "modelSource": DEFAULT_MODEL_SOURCE,
            "modelRuntimeSource": DEFAULT_MODEL_RUNTIME_SOURCE,
            "modelLicense": DEFAULT_MODEL_LICENSE,
            **provider.artifact_metadata(),
            "dimension": object_dimension,
            "objectCount": len(object_ids),
            "evidenceCount": len(evidence_ids),
            "normalised": True,
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        state_path.unlink(missing_ok=True)
        if target.exists():
            if not force:
                return target
            # ``target`` is an exact fingerprint-addressed cache directory,
            # never a user path.  Force rebuild only replaces this one target.
            shutil.rmtree(target)
        os.replace(temporary, target)
        completed = True
        return target
    finally:
        if not completed and temporary.exists():
            logger.error(
                "dense index build did not complete; resumable staging preserved at %s",
                temporary,
            )


class DenseIndex:
    """Read-only, memory-mapped dense index plus evidence-level reranker."""

    def __init__(self, path: Path, provider: FastEmbedProvider) -> None:
        import numpy as np

        self.path = path
        self.provider = provider
        self.manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        self.object_ids: list[str] = json.loads(
            (path / "object_ids.json").read_text(encoding="utf-8")
        )
        self.evidence_ids: list[str] = json.loads(
            (path / "evidence_ids.json").read_text(encoding="utf-8")
        )
        self.object_embeddings = np.load(
            path / "object_embeddings.npy", mmap_mode="r", allow_pickle=False
        )
        self.evidence_embeddings = np.load(
            path / "evidence_embeddings.npy", mmap_mode="r", allow_pickle=False
        )
        self.evidence_offsets = np.load(
            path / "evidence_offsets.npy", mmap_mode="r", allow_pickle=False
        )
        self.object_row = {object_id: row for row, object_id in enumerate(self.object_ids)}
        self._validate()

    def _validate(self) -> None:
        expected_objects = int(self.manifest.get("objectCount", -1))
        expected_evidence = int(self.manifest.get("evidenceCount", -1))
        dimension = int(self.manifest.get("dimension", -1))
        if self.manifest.get("formatVersion") != INDEX_FORMAT_VERSION:
            raise DenseRetrievalError("unsupported dense index format")
        if (
            len(self.object_ids) != expected_objects
            or self.object_embeddings.shape != (expected_objects, dimension)
        ):
            raise DenseRetrievalError("dense object index is inconsistent")
        if (
            len(self.evidence_ids) != expected_evidence
            or self.evidence_embeddings.shape != (expected_evidence, dimension)
            or self.evidence_offsets.shape != (expected_objects + 1,)
            or int(self.evidence_offsets[-1]) != expected_evidence
        ):
            raise DenseRetrievalError("dense evidence index is inconsistent")

    def embed_query(self, query: str) -> Any:
        vector = _normalise(self.provider.embed_query(query))
        if vector.shape != (int(self.manifest["dimension"]),):
            raise DenseRetrievalError("query embedding dimension does not match index")
        return vector

    def embed_queries(self, queries: Sequence[str]) -> list[Any]:
        """Encode several queries in one provider batch.

        Agentic query expansion commonly contributes two or three closely
        related searches.  Calling the ONNX encoder separately for each query
        repeats most of its setup cost, so the batch path deliberately uses
        ``provider.embed`` once while keeping ``embed_query`` unchanged for the
        ordinary single-search path.
        """

        query_list = list(queries)
        if not query_list:
            return []
        vectors = list(
            self.provider.embed(
                query_list,
                batch_size=min(64, len(query_list)),
            )
        )
        if len(vectors) != len(query_list):
            raise DenseRetrievalError(
                "embedding provider returned an unexpected number of query vectors"
            )
        dimension = int(self.manifest["dimension"])
        normalised = [_normalise(vector) for vector in vectors]
        if any(vector.shape != (dimension,) for vector in normalised):
            raise DenseRetrievalError("query embedding dimension does not match index")
        return normalised

    def search_vector(
        self,
        query_vector: Any,
        *,
        top_k: int,
        allowed_object_ids: set[str] | None = None,
    ) -> list[DenseHit]:
        import numpy as np

        scores = np.asarray(self.object_embeddings @ query_vector, dtype=np.float32)
        if allowed_object_ids is not None:
            allowed_rows = np.fromiter(
                (
                    row
                    for object_id in allowed_object_ids
                    if (row := self.object_row.get(object_id)) is not None
                ),
                dtype=np.int64,
            )
            if allowed_rows.size == 0:
                return []
            subset = scores[allowed_rows]
            count = min(top_k, int(subset.size))
            local = np.argpartition(subset, -count)[-count:]
            rows = allowed_rows[local]
        else:
            count = min(top_k, int(scores.size))
            if count == 0:
                return []
            rows = np.argpartition(scores, -count)[-count:]
        ordered = sorted(
            (int(row) for row in rows),
            key=lambda row: (-float(scores[row]), self.object_ids[row]),
        )
        return [
            DenseHit(self.object_ids[row], float(scores[row])) for row in ordered
        ]

    def evidence_hits(
        self,
        query_vector: Any,
        object_ids: Iterable[str],
        *,
        max_evidence_ids: int = 3,
    ) -> dict[str, EvidenceHit]:
        import numpy as np

        results: dict[str, EvidenceHit] = {}
        for object_id in object_ids:
            row = self.object_row.get(object_id)
            if row is None:
                continue
            start = int(self.evidence_offsets[row])
            stop = int(self.evidence_offsets[row + 1])
            if start >= stop:
                continue
            scores = np.asarray(
                self.evidence_embeddings[start:stop] @ query_vector,
                dtype=np.float32,
            )
            order = sorted(
                range(int(scores.size)),
                key=lambda offset: (
                    -float(scores[offset]),
                    self.evidence_ids[start + offset],
                ),
            )
            results[object_id] = EvidenceHit(
                score=float(scores[order[0]]),
                evidence_ids=tuple(
                    self.evidence_ids[start + offset]
                    for offset in order[:max_evidence_ids]
                ),
                evidence_scores=tuple(
                    (
                        self.evidence_ids[start + offset],
                        float(scores[offset]),
                    )
                    for offset in order[:max_evidence_ids]
                ),
            )
        return results

    def search_evidence_vector(
        self,
        query_vector: Any,
        *,
        top_k: int,
    ) -> list[EvidenceDenseHit]:
        """Search every institution excerpt, then map rows back to objects.

        Object documents can exceed the embedding model's 512-token window.
        Global excerpt recall prevents a relevant later evidence chunk from
        being invisible merely because it was truncated in the object vector.
        """

        import numpy as np

        scores = np.asarray(self.evidence_embeddings @ query_vector, dtype=np.float32)
        count = min(top_k, int(scores.size))
        if count == 0:
            return []
        rows = np.argpartition(scores, -count)[-count:]
        ordered = sorted(
            (int(row) for row in rows),
            key=lambda row: (-float(scores[row]), self.evidence_ids[row]),
        )
        hits: list[EvidenceDenseHit] = []
        for evidence_row in ordered:
            object_row = int(
                np.searchsorted(self.evidence_offsets, evidence_row, side="right") - 1
            )
            if not 0 <= object_row < len(self.object_ids):
                continue
            hits.append(
                EvidenceDenseHit(
                    object_id=self.object_ids[object_row],
                    evidence_id=self.evidence_ids[evidence_row],
                    score=float(scores[evidence_row]),
                )
            )
        return hits


class DenseIndexManager:
    """Lazy index loader with explicit BM25 degradation diagnostics."""

    def __init__(
        self,
        *,
        enabled: bool,
        index_root: Path,
        model_cache_dir: Path,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.enabled = enabled
        self.index_root = index_root
        self.model_cache_dir = model_cache_dir
        self.model_name = model_name
        self._loaded: dict[tuple[str, str, str], DenseIndex] = {}
        self._status: dict[tuple[str, str, str], DenseStatus] = {}
        self._warned: set[tuple[str, str, str, str]] = set()
        self._lock = RLock()

    def _collection_key(self, collection: Any) -> tuple[str, str, str]:
        return (
            _text(collection.id),
            _text(collection.version),
            collection_fingerprint(collection, self.model_name),
        )

    def expected_path(self, collection: Any) -> Path:
        fingerprint = collection_fingerprint(collection, self.model_name)
        return (
            self.index_root
            / _slug(_text(collection.id))
            / _slug(self.model_name)
            / fingerprint
        )

    def cache_signature(self, collection: Any) -> tuple[str, int, int]:
        if not self.enabled:
            return ("disabled", -1, -1)
        try:
            manifest = self.expected_path(collection) / "manifest.json"
            stat = manifest.stat()
            return (str(manifest), stat.st_mtime_ns, stat.st_size)
        except OSError:
            return ("missing", -1, -1)

    def _degraded(
        self,
        collection: Any,
        reason: str,
        *,
        diagnostic: str | None = None,
    ) -> None:
        key = self._collection_key(collection)
        status = DenseStatus(
            enabled=self.enabled,
            available=False,
            mode="bm25",
            reason=reason,
            model=self.model_name,
        )
        self._status[key] = status
        warning_key = (*key, reason)
        if warning_key not in self._warned:
            logger.warning(
                "hybrid RAG degraded to BM25 for %s: %s",
                key[0],
                diagnostic or reason,
            )
            self._warned.add(warning_key)

    def _mark_ready(self, collection: Any, index: DenseIndex) -> None:
        key = self._collection_key(collection)
        self._status[key] = DenseStatus(
            enabled=True,
            available=True,
            mode="hybrid",
            reason="versioned local dense cache loaded",
            model=self.model_name,
            fingerprint=index.path.name,
            index_path=str(index.path),
        )

    def get(self, collection: Any) -> DenseIndex | None:
        # FastEmbed constructs a sizeable ONNX session. Double-checking under
        # one manager lock prevents concurrent first requests from loading the
        # same ~220 MB model twice and briefly doubling memory.
        with self._lock:
            return self._get_locked(collection)

    def _get_locked(self, collection: Any) -> DenseIndex | None:
        key = self._collection_key(collection)
        if not self.enabled:
            self._status[key] = DenseStatus(
                enabled=False,
                available=False,
                mode="bm25",
                reason="hybrid RAG is disabled by configuration",
                model=self.model_name,
            )
            return None
        if key in self._loaded:
            return self._loaded[key]
        path = self.expected_path(collection)
        if not (path / "manifest.json").exists():
            self._degraded(
                collection,
                "versioned dense cache is missing; run scripts/build_dense_index.py",
                diagnostic=f"versioned dense cache is missing at {path}",
            )
            return None
        try:
            provider = FastEmbedProvider(
                self.model_name,
                self.model_cache_dir,
                allow_download=False,
            )
            index = DenseIndex(path, provider)
            expected_fingerprint = path.name
            if index.manifest.get("fingerprint") != expected_fingerprint:
                raise DenseRetrievalError("dense manifest fingerprint mismatch")
            expected_manifest = {
                "collectionId": collection.id,
                "collectionVersion": collection.version,
                "objectsSha256": getattr(collection, "objects_sha256", None),
                "model": self.model_name,
                "textRecipeVersion": TEXT_RECIPE_VERSION,
                "textRecipeSha256": TEXT_RECIPE_SHA256,
            }
            for field, expected in expected_manifest.items():
                if index.manifest.get(field) != expected:
                    raise DenseRetrievalError(
                        f"dense manifest {field} does not match the loaded collection"
                    )
            cached_artifact = index.manifest.get("modelArtifactSha256")
            runtime_artifact = provider.artifact_metadata().get("modelArtifactSha256")
            if cached_artifact and runtime_artifact != cached_artifact:
                raise DenseRetrievalError("local embedding model artifact changed")
        except Exception as error:
            self._degraded(
                collection,
                "dense cache or local model failed integrity validation; serving BM25",
                diagnostic=f"dense cache could not be loaded: {error}",
            )
            return None
        self._loaded[key] = index
        self._mark_ready(collection, index)
        logger.info(
            "hybrid RAG enabled for %s with %s (%s objects, %s evidence chunks)",
            collection.id,
            self.model_name,
            index.manifest["objectCount"],
            index.manifest["evidenceCount"],
        )
        return index

    def mark_query_failure(self, collection: Any, error: Exception) -> None:
        """Expose a transient query failure while allowing the next call to retry."""

        with self._lock:
            self._degraded(
                collection,
                "hybrid query failed; this request used BM25 and the next request will retry",
                diagnostic=f"hybrid query failed: {error}",
            )

    def mark_query_success(self, collection: Any, index: DenseIndex) -> None:
        with self._lock:
            self._mark_ready(collection, index)

    def status(self, collection: Any) -> DenseStatus:
        with self._lock:
            key = self._collection_key(collection)
            if key not in self._status:
                self._get_locked(collection)
            return self._status[key]
