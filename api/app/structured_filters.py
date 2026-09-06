"""Versioned, read-only SQLite filtering for museum retrieval candidates.

The database is a disposable derivative of one frozen collection snapshot. It
contains allowlisted metadata facets plus source-derived object and evidence
search documents; visitor queries are never stored. Runtime callers provide
:class:`FilterSpec` or bounded literal search text, never SQL. Query values are
bound parameters and all column/operator choices come from fixed compilers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from uuid import uuid4

from .retrieval_filters import (
    EVIDENCE_DEPTH_FILTER_VALUES,
    MAX_FILTER_YEAR,
    MIN_FILTER_YEAR,
    FilterSpec,
)
from .cultural_normalization import CULTURAL_ROUTING_VERSION


FILTER_INDEX_FORMAT_VERSION = "structured-filter-sqlite-v5"
FILTER_INDEX_SCHEMA_VERSION = 4
OBJECT_SEARCH_FTS5 = "sqlite_fts5_bm25"
OBJECT_SEARCH_UNAVAILABLE = "unavailable"
EVIDENCE_SEARCH_FTS5 = "sqlite_fts5_bm25"
EVIDENCE_SEARCH_UNAVAILABLE = "unavailable"
FACETS = frozenset(
    {"culture", "institution", "material", "object_type", "rights"}
)
_VALUE_SPLIT = re.compile(r"\s*(?:[;,；，|]|\s+/\s+)\s*")
_SAFE_SLUG = re.compile(r"[^a-z0-9._-]+")
_SEARCH_WORD = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")

# Field vocabulary normalization, not question-specific search rules. Keep
# regions distinct: a country name must not expand to an entire culture pack.
_CULTURE_EQUIVALENTS = (
    ("east_asia", "east asia", "东亚"),
    ("south_asia", "south asia", "南亚"),
    ("southeast_asia", "southeast asia", "东南亚"),
    ("west_asia_north_africa", "west asia north africa", "west asia and north africa", "西亚与北非", "西亚北非"),
    ("europe", "european", "欧洲"),
    ("africa", "african", "非洲"),
    ("americas", "the americas", "美洲"),
    ("oceania", "oceanian", "大洋洲"),
    ("china", "chinese", "中国"),
    ("japan", "japanese", "日本"),
    ("korea", "korean", "韩国", "朝鲜"),
    ("india", "indian", "印度"),
    ("egypt", "egyptian", "埃及"),
    ("iran", "iranian", "伊朗"),
    ("netherlands", "dutch", "荷兰"),
)


@lru_cache(maxsize=4096)
def _phrase_words(value: str) -> str:
    return " " + re.sub(r"[^\w]+", " ", _normalise(value)).strip() + " "


def _catalogue_phrase_contains(value: str, phrase: str) -> bool:
    # Match components of a composite medium, but not iron in ironstone or
    # gold in golden. The caller still binds literal strings, never patterns.
    return bool(phrase.strip()) and _phrase_words(phrase) in _phrase_words(value)


class StructuredFilterError(RuntimeError):
    """A derived filter index is missing, stale, corrupt or misused."""


@dataclass(frozen=True)
class FilterIndexIdentity:
    collection_id: str
    collection_version: str
    objects_sha256: str
    fingerprint: str


@dataclass(frozen=True)
class FilterBuildStats:
    object_count: int
    evidence_count: int
    object_search_mode: str
    evidence_search_mode: str


@dataclass(frozen=True)
class EvidenceSearchHit:
    object_id: str
    evidence_id: str
    score: float
    source_kind: str


@dataclass(frozen=True)
class ObjectSearchHit:
    object_id: str
    score: float


def _slug(value: str) -> str:
    slug = _SAFE_SLUG.sub("-", value.casefold()).strip("-._")
    return slug or "collection"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collection_filter_identity(collection: Any) -> FilterIndexIdentity:
    collection_id = str(getattr(collection, "id", "") or "").strip()
    collection_version = str(getattr(collection, "version", "") or "").strip()
    objects_sha256 = str(
        getattr(collection, "objects_sha256", "") or ""
    ).strip().casefold()
    if not collection_id or not collection_version:
        raise StructuredFilterError("collection id and version are required")
    if not re.fullmatch(r"[0-9a-f]{64}", objects_sha256):
        raise StructuredFilterError("collection objects SHA-256 is required")
    payload = json.dumps(
        {
            "formatVersion": FILTER_INDEX_FORMAT_VERSION,
            "culturalRoutingVersion": CULTURAL_ROUTING_VERSION,
            "collectionId": collection_id,
            "collectionVersion": collection_version,
            "objectsSha256": objects_sha256,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return FilterIndexIdentity(
        collection_id=collection_id,
        collection_version=collection_version,
        objects_sha256=objects_sha256,
        fingerprint=hashlib.sha256(payload).hexdigest(),
    )


def filter_index_path(index_root: Path, collection: Any) -> Path:
    identity = collection_filter_identity(collection)
    return index_root.resolve() / _slug(identity.collection_id) / identity.fingerprint


def _normalise(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", str(value or "")),
    ).strip().casefold()


def _fts_index_text(value: Any) -> str:
    """Pre-segment CJK runs to match the maintained lexical token recipe.

    SQLite's portable ``unicode61`` tokenizer treats a whole unspaced Chinese
    phrase as one token, while visitor queries and the fielded BM25 baseline
    use overlapping bigrams/trigrams. Store those CJK n-grams separated by
    spaces; non-CJK text remains ordinary unicode61 input.
    """

    text = _normalise(value)

    def segment(match: re.Match[str]) -> str:
        run = match.group(0)
        if len(run) == 1:
            return run
        terms = [run[index : index + 2] for index in range(len(run) - 1)]
        terms.extend(
            run[index : index + 3] for index in range(max(0, len(run) - 2))
        )
        return " ".join(terms)

    return _CJK_RUN.sub(segment, text)


def _raw_values(obj: Any, *attributes: str) -> Iterator[str]:
    for attribute in attributes:
        raw = getattr(obj, attribute, None)
        values = raw if isinstance(raw, (list, tuple, set, frozenset)) else (raw,)
        for value in values:
            text = re.sub(
                r"\s+",
                " ",
                unicodedata.normalize("NFKC", str(value or "")),
            ).strip()
            if text:
                yield text


def _facet_values(obj: Any, facet: str) -> dict[str, str]:
    attributes = {
        "culture": ("culture", "culture_display", "culture_pack_ids"),
        "institution": ("institution", "institution_id"),
        "material": ("material", "medium"),
        "object_type": ("type", "classification"),
        "rights": (
            "rights",
            "rights_uri",
            "image_license",
            "image_rights_uri",
        ),
    }[facet]
    values: dict[str, str] = {}
    for display in _raw_values(obj, *attributes):
        candidates = {display, display.replace("_", " ")}
        candidates.update(part for part in _VALUE_SPLIT.split(display) if part)
        for candidate in candidates:
            normalised = _normalise(candidate)
            if normalised:
                values.setdefault(normalised, candidate[:200])
    return values


def _object_dates(obj: Any) -> tuple[int | None, int | None]:
    earliest = getattr(obj, "date_earliest", None)
    latest = getattr(obj, "date_latest", None)
    if isinstance(earliest, bool) or not isinstance(earliest, int):
        earliest = None
    if isinstance(latest, bool) or not isinstance(latest, int):
        latest = None
    if earliest is None:
        earliest = latest
    if latest is None:
        latest = earliest
    if earliest is not None and latest is not None and earliest > latest:
        earliest, latest = latest, earliest
    return earliest, latest


def _collection_evidence_count(collection: Any) -> int:
    count = 0
    for obj in getattr(collection, "objects", ()) or ():
        for chunk in getattr(obj, "evidence", ()) or ():
            if str(getattr(chunk, "id", "") or "").strip() and str(
                getattr(chunk, "text", "") or ""
            ).strip():
                count += 1
    return count


def _database_metadata(
    identity: FilterIndexIdentity,
    stats: FilterBuildStats,
) -> dict[str, str]:
    return {
        "formatVersion": FILTER_INDEX_FORMAT_VERSION,
        "schemaVersion": str(FILTER_INDEX_SCHEMA_VERSION),
        "collectionId": identity.collection_id,
        "collectionVersion": identity.collection_version,
        "objectsSha256": identity.objects_sha256,
        "fingerprint": identity.fingerprint,
        "objectCount": str(stats.object_count),
        "evidenceCount": str(stats.evidence_count),
        "objectSearchMode": stats.object_search_mode,
        "evidenceSearchMode": stats.evidence_search_mode,
    }


def _create_evidence_fts(connection: sqlite3.Connection) -> bool:
    """Create the FTS5 table, returning False only when FTS5 is unavailable."""

    try:
        connection.execute(
            "CREATE VIRTUAL TABLE evidence_fts USING fts5("
            "text, supports, content='evidence_docs', content_rowid='rowid', "
            "tokenize='unicode61 remove_diacritics 2')"
        )
    except sqlite3.OperationalError as error:
        if "no such module: fts5" in str(error).casefold():
            return False
        raise
    return True


def _create_object_fts(connection: sqlite3.Connection) -> bool:
    """Create four-field object FTS, returning False only without FTS5."""

    try:
        connection.execute(
            "CREATE VIRTUAL TABLE object_fts USING fts5("
            "title, routing, metadata, content, "
            "content='object_docs', content_rowid='rowid', "
            "tokenize='unicode61 remove_diacritics 2')"
        )
    except sqlite3.OperationalError as error:
        if "no such module: fts5" in str(error).casefold():
            return False
        raise
    return True


def _object_search_fields(obj: Any) -> tuple[str, str, str, str]:
    """Mirror the legacy BM25 field boundaries without importing collections."""

    def joined(values: Iterable[Any]) -> str:
        return " ".join(
            re.sub(r"\s+", " ", str(value or "")).strip()
            for value in values
            if str(value or "").strip()
        )

    title = joined((getattr(obj, "title", None), getattr(obj, "title_original", None)))
    routing = joined(
        (
            *(getattr(obj, "routing_domain_ids", ()) or ()),
            *(getattr(obj, "culture_pack_ids", ()) or ()),
            getattr(obj, "culture", None),
        )
    )
    metadata = joined(
        (
            getattr(obj, "date", None),
            getattr(obj, "creator", None),
            getattr(obj, "material", None),
            getattr(obj, "place", None),
            getattr(obj, "type", None),
            getattr(obj, "classification", None),
            getattr(obj, "department", None),
            *(getattr(obj, "tags", ()) or ()),
        )
    )
    content = joined(
        (
            getattr(obj, "description", None),
            *(
                getattr(chunk, "text", None)
                for chunk in (getattr(obj, "evidence", ()) or ())
            ),
        )
    )
    return tuple(
        _fts_index_text(value) for value in (title, routing, metadata, content)
    )


def _create_database(
    path: Path,
    collection: Any,
    identity: FilterIndexIdentity,
) -> FilterBuildStats:
    objects = sorted(
        list(getattr(collection, "objects", ()) or ()),
        key=lambda obj: str(getattr(obj, "id", "")),
    )
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE objects (
                object_id TEXT PRIMARY KEY,
                date_start INTEGER,
                date_end INTEGER,
                has_image INTEGER NOT NULL CHECK (has_image IN (0, 1)),
                evidence_depth TEXT NOT NULL CHECK (evidence_depth IN ('full', 'thin'))
            ) WITHOUT ROWID;

            CREATE TABLE object_facets (
                object_id TEXT NOT NULL REFERENCES objects(object_id) ON DELETE CASCADE,
                facet TEXT NOT NULL CHECK (
                    facet IN ('culture', 'institution', 'material', 'object_type', 'rights')
                ),
                value_norm TEXT NOT NULL,
                value_display TEXT NOT NULL,
                PRIMARY KEY (object_id, facet, value_norm)
            ) WITHOUT ROWID;

            CREATE TABLE evidence_docs (
                rowid INTEGER PRIMARY KEY,
                evidence_id TEXT NOT NULL,
                object_id TEXT NOT NULL REFERENCES objects(object_id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                supports TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                UNIQUE (object_id, evidence_id)
            );

            CREATE TABLE object_docs (
                rowid INTEGER PRIMARY KEY,
                object_id TEXT NOT NULL UNIQUE REFERENCES objects(object_id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                routing TEXT NOT NULL,
                metadata TEXT NOT NULL,
                content TEXT NOT NULL
            );

            CREATE INDEX object_dates ON objects(date_start, date_end, object_id);
            CREATE INDEX object_image ON objects(has_image, object_id);
            CREATE INDEX object_depth ON objects(evidence_depth, object_id);
            CREATE INDEX facet_lookup ON object_facets(facet, value_norm, object_id);
            CREATE INDEX evidence_object ON evidence_docs(object_id, evidence_id);
            CREATE INDEX object_docs_id ON object_docs(object_id);
            """
        )
        object_fts_available = _create_object_fts(connection)
        fts5_available = _create_evidence_fts(connection)
        object_rows: list[tuple[str, int | None, int | None, int, str]] = []
        facet_rows: list[tuple[str, str, str, str]] = []
        evidence_rows: list[tuple[str, str, str, str, str]] = []
        object_document_rows: list[tuple[str, str, str, str, str]] = []
        for obj in objects:
            object_id = str(getattr(obj, "id", "") or "").strip()
            if not object_id:
                raise StructuredFilterError("every indexed object requires an id")
            date_start, date_end = _object_dates(obj)
            evidence_depth = _normalise(getattr(obj, "evidence_depth", "thin"))
            if evidence_depth not in EVIDENCE_DEPTH_FILTER_VALUES:
                evidence_depth = "thin"
            object_rows.append(
                (
                    object_id,
                    date_start,
                    date_end,
                    int(bool(getattr(obj, "image_url", None))),
                    evidence_depth,
                )
            )
            object_document_rows.append((object_id, *_object_search_fields(obj)))
            for facet in sorted(FACETS):
                facet_rows.extend(
                    (object_id, facet, normalised, display)
                    for normalised, display in sorted(
                        _facet_values(obj, facet).items()
                    )
                )
            for chunk in getattr(obj, "evidence", ()) or ():
                evidence_id = str(getattr(chunk, "id", "") or "").strip()
                text = re.sub(
                    r"\s+", " ", str(getattr(chunk, "text", "") or "")
                ).strip()
                if not evidence_id or not text:
                    continue
                evidence_rows.append(
                    (
                        evidence_id,
                        object_id,
                        _fts_index_text(text),
                        _fts_index_text(getattr(chunk, "supports", "") or ""),
                        str(getattr(chunk, "source_kind", "") or "unknown")[:80],
                    )
                )
        connection.executemany(
            "INSERT INTO objects(object_id, date_start, date_end, has_image, evidence_depth) "
            "VALUES (?, ?, ?, ?, ?)",
            object_rows,
        )
        connection.executemany(
            "INSERT INTO object_facets(object_id, facet, value_norm, value_display) "
            "VALUES (?, ?, ?, ?)",
            facet_rows,
        )
        connection.executemany(
            "INSERT INTO evidence_docs(evidence_id, object_id, text, supports, source_kind) "
            "VALUES (?, ?, ?, ?, ?)",
            evidence_rows,
        )
        connection.executemany(
            "INSERT INTO object_docs(object_id, title, routing, metadata, content) "
            "VALUES (?, ?, ?, ?, ?)",
            object_document_rows,
        )
        object_search_mode = (
            OBJECT_SEARCH_FTS5 if object_fts_available else OBJECT_SEARCH_UNAVAILABLE
        )
        evidence_search_mode = (
            EVIDENCE_SEARCH_FTS5 if fts5_available else EVIDENCE_SEARCH_UNAVAILABLE
        )
        stats = FilterBuildStats(
            object_count=len(object_rows),
            evidence_count=len(evidence_rows),
            object_search_mode=object_search_mode,
            evidence_search_mode=evidence_search_mode,
        )
        if object_fts_available:
            connection.execute("INSERT INTO object_fts(object_fts) VALUES ('rebuild')")
        if fts5_available:
            connection.execute(
                "INSERT INTO evidence_fts(evidence_fts) VALUES ('rebuild')"
            )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            _database_metadata(identity, stats).items(),
        )
        connection.execute(f"PRAGMA user_version={FILTER_INDEX_SCHEMA_VERSION}")
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise StructuredFilterError("SQLite integrity check failed")
        return stats
    except sqlite3.Error as error:
        raise StructuredFilterError(f"could not build structured filter index: {error}") from error
    finally:
        connection.close()


def build_structured_filter_index(
    collection: Any,
    *,
    index_root: Path,
    force: bool = False,
) -> Path:
    """Build one immutable, fingerprint-addressed SQLite derivative."""

    identity = collection_filter_identity(collection)
    target = filter_index_path(index_root, collection)
    if target.exists() and not force:
        StructuredFilterIndex(target, collection)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{identity.fingerprint}.tmp-{uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    completed = False
    try:
        database_path = temporary / "filters.sqlite3"
        stats = _create_database(database_path, collection, identity)
        manifest = {
            **_database_metadata(identity, stats),
            "schemaVersion": FILTER_INDEX_SCHEMA_VERSION,
            "objectCount": stats.object_count,
            "evidenceCount": stats.evidence_count,
            "objectSearchMode": stats.object_search_mode,
            "evidenceSearchMode": stats.evidence_search_mode,
            "databaseSha256": _sha256_file(database_path),
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        StructuredFilterIndex(temporary, collection)
        if target.exists():
            # The target is the exact derived fingerprint, never a caller path.
            shutil.rmtree(target)
        os.replace(temporary, target)
        completed = True
        return target
    finally:
        if not completed and temporary.exists():
            shutil.rmtree(temporary)


def _readonly_connection(path: Path) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA trusted_schema=OFF")
    connection.create_function("catalogue_phrase_contains", 2, _catalogue_phrase_contains, deterministic=True)
    return connection


def _validated_values(values: Sequence[str], *, limit: int = 8) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or len(values) > limit:
        raise StructuredFilterError("filter values must be a bounded string sequence")
    normalised: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise StructuredFilterError("filter values must be strings")
        item = _normalise(value[:100])
        if item and item not in normalised:
            normalised.append(item)
    return tuple(normalised)


def _search_query_terms(query: str | Sequence[str]) -> tuple[str, ...]:
    raw_values: Sequence[str]
    if isinstance(query, str):
        raw_values = (query,)
    elif isinstance(query, Sequence) and not isinstance(query, (bytes, bytearray)):
        raw_values = query
    else:
        raise StructuredFilterError("evidence query must be text or text tokens")
    terms: list[str] = []
    for raw in raw_values:
        if not isinstance(raw, str):
            raise StructuredFilterError("evidence query tokens must be strings")
        for match in _SEARCH_WORD.finditer(_normalise(raw)):
            term = match.group(0)[:64]
            if term and term not in terms:
                terms.append(term)
            if len(terms) >= 32:
                return tuple(terms)
    return tuple(terms)


def _chunks(values: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class StructuredFilterIndex:
    """Verified runtime handle that opens SQLite read-only for each query."""

    def __init__(self, directory: Path, collection: Any) -> None:
        self.directory = directory.resolve()
        self.database_path = self.directory / "filters.sqlite3"
        self.manifest_path = self.directory / "manifest.json"
        identity = collection_filter_identity(collection)
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StructuredFilterError("structured filter manifest is unavailable") from error
        expected = {
            "formatVersion": FILTER_INDEX_FORMAT_VERSION,
            "schemaVersion": FILTER_INDEX_SCHEMA_VERSION,
            "collectionId": identity.collection_id,
            "collectionVersion": identity.collection_version,
            "objectsSha256": identity.objects_sha256,
            "fingerprint": identity.fingerprint,
            "objectCount": len(getattr(collection, "objects", ()) or ()),
            "evidenceCount": _collection_evidence_count(collection),
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise StructuredFilterError("structured filter manifest does not match collection")
        evidence_search_mode = manifest.get("evidenceSearchMode")
        if evidence_search_mode not in {
            EVIDENCE_SEARCH_FTS5,
            EVIDENCE_SEARCH_UNAVAILABLE,
        }:
            raise StructuredFilterError("structured filter evidence search mode is invalid")
        object_search_mode = manifest.get("objectSearchMode")
        if object_search_mode not in {
            OBJECT_SEARCH_FTS5,
            OBJECT_SEARCH_UNAVAILABLE,
        }:
            raise StructuredFilterError("structured filter object search mode is invalid")
        try:
            database_sha256 = _sha256_file(self.database_path)
        except OSError as error:
            raise StructuredFilterError("structured filter database is unavailable") from error
        if manifest.get("databaseSha256") != database_sha256:
            raise StructuredFilterError("structured filter database digest mismatch")
        try:
            with _readonly_connection(self.database_path) as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                rows = dict(connection.execute("SELECT key, value FROM metadata"))
        except sqlite3.Error as error:
            raise StructuredFilterError("structured filter database is unreadable") from error
        stats = FilterBuildStats(
            object_count=expected["objectCount"],
            evidence_count=expected["evidenceCount"],
            object_search_mode=object_search_mode,
            evidence_search_mode=evidence_search_mode,
        )
        expected_metadata = _database_metadata(identity, stats)
        if version != FILTER_INDEX_SCHEMA_VERSION or rows != expected_metadata:
            raise StructuredFilterError("structured filter database metadata mismatch")
        self.manifest = manifest
        self.object_search_mode = object_search_mode
        self.evidence_search_mode = evidence_search_mode

    @classmethod
    def for_collection(cls, index_root: Path, collection: Any) -> "StructuredFilterIndex":
        return cls(filter_index_path(index_root, collection), collection)

    @staticmethod
    def _compile(filters: FilterSpec) -> tuple[str, tuple[Any, ...]]:
        clauses: list[str] = []
        parameters: list[Any] = []

        date_start = filters.date_start
        date_end = filters.date_end
        for value in (date_start, date_end):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not MIN_FILTER_YEAR <= value <= MAX_FILTER_YEAR
            ):
                raise StructuredFilterError("filter year is invalid")
        if date_start is not None and date_end is not None and date_start > date_end:
            raise StructuredFilterError("filter date range is inverted")
        if date_start is not None:
            clauses.append("o.date_end IS NOT NULL AND o.date_end >= ?")
            parameters.append(date_start)
        if date_end is not None:
            clauses.append("o.date_start IS NOT NULL AND o.date_start <= ?")
            parameters.append(date_end)
        if filters.image_required is not None and not isinstance(
            filters.image_required, bool
        ):
            raise StructuredFilterError("imageRequired must be boolean or null")
        if filters.image_required is True:
            clauses.append("o.has_image = 1")

        facet_filters = (
            ("culture", filters.cultures),
            ("institution", filters.institutions),
            ("material", filters.materials),
            ("object_type", filters.object_types),
            ("rights", filters.rights_allowed),
        )
        for facet, raw_values in facet_filters:
            values = _validated_values(raw_values)
            if not values:
                continue
            if facet == "culture":
                values = tuple(dict.fromkeys(
                    alias
                    for value in values
                    for alias in next((group for group in _CULTURE_EQUIVALENTS if value in group), (value,))
                ))
            if facet in {"material", "object_type"}:
                predicates = " OR ".join("catalogue_phrase_contains(f.value_norm, ?)" for _ in values)
                clauses.append(
                    "EXISTS (SELECT 1 FROM object_facets AS f "
                    "WHERE f.object_id = o.object_id AND f.facet = ? "
                    f"AND ({predicates}))"
                )
                parameters.append(facet)
                parameters.extend(values)
                continue
            placeholders = ", ".join("?" for _ in values)
            clauses.append(
                "EXISTS (SELECT 1 FROM object_facets AS f "
                "WHERE f.object_id = o.object_id AND f.facet = ? "
                f"AND f.value_norm IN ({placeholders}))"
            )
            parameters.append(facet)
            parameters.extend(values)

        depth = _validated_values(filters.evidence_depth, limit=2)
        if any(value not in EVIDENCE_DEPTH_FILTER_VALUES for value in depth):
            raise StructuredFilterError("evidenceDepth contains an invalid value")
        if depth:
            placeholders = ", ".join("?" for _ in depth)
            clauses.append(f"o.evidence_depth IN ({placeholders})")
            parameters.extend(depth)

        sql = "SELECT o.object_id FROM objects AS o"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY o.object_id"
        return sql, tuple(parameters)

    def filter_object_ids(
        self,
        filters: FilterSpec,
        *,
        allowed_object_ids: Iterable[str] | None = None,
    ) -> list[str]:
        """Return matching IDs; an optional candidate intersection occurs in Python."""

        if not isinstance(filters, FilterSpec):
            raise StructuredFilterError("runtime filters must be a FilterSpec")
        sql, parameters = self._compile(filters)
        try:
            with _readonly_connection(self.database_path) as connection:
                rows = [row[0] for row in connection.execute(sql, parameters)]
        except sqlite3.Error as error:
            raise StructuredFilterError("structured filter query failed") from error
        if allowed_object_ids is None:
            return rows
        allowed = {str(object_id) for object_id in allowed_object_ids}
        return [object_id for object_id in rows if object_id in allowed]

    @property
    def evidence_search_available(self) -> bool:
        return self.evidence_search_mode == EVIDENCE_SEARCH_FTS5

    @property
    def object_search_available(self) -> bool:
        return self.object_search_mode == OBJECT_SEARCH_FTS5

    def search_objects(
        self,
        query: str | Sequence[str],
        *,
        top_k: int = 60,
        allowed_object_ids: Iterable[str] | None = None,
    ) -> list[ObjectSearchHit]:
        """Run weighted object-level BM25 without scanning Python objects.

        Field weights match the maintained lexical baseline: title 5, routing
        3, metadata 2 and content 1. Caller text is reduced to bounded literal
        tokens before entering MATCH; object restrictions and limits use bound
        parameters.
        """

        if not self.object_search_available:
            raise StructuredFilterError(
                "object-level BM25 is unavailable because SQLite FTS5 was not built"
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 500:
            raise StructuredFilterError("object search top_k must be between 1 and 500")
        terms = _search_query_terms(query)
        if not terms:
            return []
        match_query = " OR ".join(f'"{term}"' for term in terms)
        allowed = (
            tuple(dict.fromkeys(str(value) for value in allowed_object_ids))
            if allowed_object_ids is not None
            else None
        )
        if allowed == ():
            return []

        base_sql = (
            "SELECT d.object_id, bm25(object_fts, 5.0, 3.0, 2.0, 1.0) AS rank "
            "FROM object_fts "
            "JOIN object_docs AS d ON d.rowid = object_fts.rowid "
            "WHERE object_fts MATCH ?"
        )
        batches: list[Sequence[str] | None] = (
            list(_chunks(allowed, 800)) if allowed is not None else [None]
        )
        hits: dict[str, ObjectSearchHit] = {}
        try:
            with _readonly_connection(self.database_path) as connection:
                for batch in batches:
                    parameters: list[Any] = [match_query]
                    sql = base_sql
                    if batch is not None:
                        placeholders = ", ".join("?" for _ in batch)
                        sql += f" AND d.object_id IN ({placeholders})"
                        parameters.extend(batch)
                    sql += " ORDER BY rank ASC, d.object_id LIMIT ?"
                    parameters.append(top_k)
                    for object_id, raw_rank in connection.execute(sql, parameters):
                        hit = ObjectSearchHit(
                            object_id=object_id,
                            score=-float(raw_rank),
                        )
                        current = hits.get(object_id)
                        if current is None or hit.score > current.score:
                            hits[object_id] = hit
        except sqlite3.Error as error:
            raise StructuredFilterError("object-level BM25 query failed") from error
        return sorted(
            hits.values(),
            key=lambda hit: (-hit.score, hit.object_id),
        )[:top_k]

    def search_evidence(
        self,
        query: str | Sequence[str],
        *,
        top_k: int = 60,
        allowed_object_ids: Iterable[str] | None = None,
    ) -> list[EvidenceSearchHit]:
        """Run independent evidence-level BM25 over source rows.

        The MATCH expression is assembled only from bounded word tokens. Object
        restrictions and limits are bound parameters. If this Python/SQLite
        build lacks FTS5, the manifest says so and this method raises explicitly
        instead of returning an empty list that could be mistaken for no hits.
        """

        if not self.evidence_search_available:
            raise StructuredFilterError(
                "evidence-level BM25 is unavailable because SQLite FTS5 was not built"
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 500:
            raise StructuredFilterError("evidence search top_k must be between 1 and 500")
        terms = _search_query_terms(query)
        if not terms:
            return []
        # Terms contain only Unicode word characters plus internal apostrophes
        # or hyphens. Quoting preserves them as literals; no raw FTS operator is
        # accepted from the caller.
        match_query = " OR ".join(f'"{term}"' for term in terms)
        allowed = (
            tuple(dict.fromkeys(str(value) for value in allowed_object_ids))
            if allowed_object_ids is not None
            else None
        )
        if allowed == ():
            return []

        base_sql = (
            "SELECT d.object_id, d.evidence_id, d.source_kind, "
            "bm25(evidence_fts, 1.0, 0.35) AS rank "
            "FROM evidence_fts "
            "JOIN evidence_docs AS d ON d.rowid = evidence_fts.rowid "
            "WHERE evidence_fts MATCH ?"
        )
        batches: list[Sequence[str] | None] = (
            list(_chunks(allowed, 800)) if allowed is not None else [None]
        )
        hits: dict[tuple[str, str], EvidenceSearchHit] = {}
        try:
            with _readonly_connection(self.database_path) as connection:
                for batch in batches:
                    parameters: list[Any] = [match_query]
                    sql = base_sql
                    if batch is not None:
                        placeholders = ", ".join("?" for _ in batch)
                        sql += f" AND d.object_id IN ({placeholders})"
                        parameters.extend(batch)
                    sql += " ORDER BY rank ASC, d.object_id, d.evidence_id LIMIT ?"
                    parameters.append(top_k)
                    for object_id, evidence_id, source_kind, raw_rank in connection.execute(
                        sql, parameters
                    ):
                        hit = EvidenceSearchHit(
                            object_id=object_id,
                            evidence_id=evidence_id,
                            score=-float(raw_rank),
                            source_kind=source_kind,
                        )
                        key = (object_id, evidence_id)
                        current = hits.get(key)
                        if current is None or hit.score > current.score:
                            hits[key] = hit
        except sqlite3.Error as error:
            raise StructuredFilterError("evidence-level BM25 query failed") from error
        return sorted(
            hits.values(),
            key=lambda hit: (-hit.score, hit.object_id, hit.evidence_id),
        )[:top_k]
