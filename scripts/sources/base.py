"""Shared plumbing for the multi-institution open-access importers.

Every adapter goes through this module so raw responses, checksums, licence
state and evidence structure stay identical across institutions. Adapters only
know how to talk to one API and how to map its fields onto ``SourceObject``.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLLECTIONS_ROOT = PROJECT_ROOT / "data" / "collections"

SCHEMA_VERSION = "3.0"
IMPORTER_VERSION = "3.0.0"
USER_AGENT = "Inquiry-Curator-Importer/2.0 (academic research demo; contact: jadppcc@gmail.com)"
REQUEST_TIMEOUT_SECONDS = 45
AIC_IMAGE_HOST = "www.artic.edu"
AIC_USER_AGENT = "Inquiry-Curator/2.0 (academic research demo; jadppcc@gmail.com)"

CC0_LICENSE = "CC0 1.0"
CC0_RIGHTS_URI = "https://creativecommons.org/publicdomain/zero/1.0/"
CC_BY_4_0_LICENSE = "CC BY 4.0"
CC_BY_4_0_RIGHTS_URI = "https://creativecommons.org/licenses/by/4.0/"

# These are source-specific import contracts, not a generic inference that the
# words "public domain" make every field in an arbitrary museum record CC0.
# All three adapters apply an institution-side open-access gate before creating
# a SourceObject.  Unknown institutions therefore remain unclassified until an
# adapter records an explicit field-level policy.
FIELD_LEVEL_OPEN_ACCESS_INSTITUTIONS = frozenset({"aic", "cma", "met"})

# Institution APIs are a shared public good; stay well under any published cap.
DEFAULT_MIN_INTERVAL_SECONDS = 0.5


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(content: bytes) -> str:
    return sha256(content).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).casefold()).strip("-")
    return slug or "item"


def encode_url(url: str) -> str:
    """Percent-encode unsafe characters in an already-formed URL.

    Institutions publish image paths containing raw spaces (the Met has files
    like ``DP-13187-007 CRT.jpg``), which urllib rejects outright. Only the path
    and query are touched, and already-encoded triplets are left alone so this
    is safe to apply repeatedly.
    """
    parts = urllib.parse.urlsplit(url)
    path = urllib.parse.quote(parts.path, safe="/%:@&=+$,~*'()!;")
    query = urllib.parse.quote(parts.query, safe="/%:@&=+$,~*'()!;?")
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, path, query, parts.fragment)
    )


def strip_html(value: Any) -> str:
    """Flatten institution HTML into plain text.

    Institution descriptions arrive as HTML fragments. They are also the main
    prompt-injection surface, so markup is removed before the text is ever
    handed to a model.
    """
    if value in (None, ""):
        return ""
    text = str(value)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    return re.sub(r"\s+", " ", text).strip()


def display_value(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, list):
        parts = [display_value(item) for item in value]
        return "; ".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("description", "title", "name", "value"):
            if value.get(key):
                return display_value(value[key])
        return ""
    return strip_html(value)


def excerpt(text: str, max_characters: int = 650) -> str:
    """Trim long institution prose at a sentence boundary where possible."""
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_characters:
        return compact
    window = compact[:max_characters]
    for terminator in (". ", "。", "; ", "! ", "? "):
        cut = window.rfind(terminator)
        if cut > max_characters * 0.6:
            return window[: cut + 1].strip()
    return window.rstrip() + "…"


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class FetchError(RuntimeError):
    pass


@dataclass
class HttpClient:
    """Rate-limited HTTP client with exponential backoff.

    ``min_interval`` is enforced per client instance, so each adapter throttles
    its own institution independently.
    """

    extra_headers: dict[str, str] = field(default_factory=dict)
    min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS
    _last_request_at: float = field(default=0.0, init=False)
    _throttle_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def _throttle(self) -> None:
        """Space requests out, correctly across threads.

        A per-thread client would give each worker its own interval, which is
        no rate limit at all — that is what got a concurrent import blocked.
        """
        with self._throttle_lock:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_request_at = time.monotonic()

    def fetch_bytes(
        self,
        url: str,
        method: str = "GET",
        attempts: int = 4,
        accept: str = "application/json",
    ) -> tuple[bytes, dict[str, str], int]:
        headers = {"User-Agent": USER_AGENT, "Accept": accept}
        headers.update(self.extra_headers)
        url = encode_url(url)
        last_error: Exception | None = None
        for attempt in range(attempts):
            self._throttle()
            try:
                request = urllib.request.Request(url, headers=headers, method=method)
                with urllib.request.urlopen(
                    request, timeout=REQUEST_TIMEOUT_SECONDS
                ) as response:
                    return (
                        response.read(),
                        dict(response.headers.items()),
                        response.status,
                    )
            except urllib.error.HTTPError as error:
                last_error = error
                # 404/400/401 mean the record genuinely is not available.
                if error.code in {400, 401, 404}:
                    raise FetchError(f"{error.code} for {url}") from error
                # 403/429 are usually rate limiting under concurrency. Back off
                # and retry; only give up once the attempts are exhausted.
                time.sleep(min(2**attempt, 12))
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error
                time.sleep(min(2**attempt, 12))
            except Exception as error:  # noqa: BLE001
                # A malformed URL (the Met publishes image paths containing raw
                # spaces) raises before any request is made. Retrying will not
                # help, and one bad record must not abort a whole import.
                raise FetchError(f"Could not request {url}: {error}") from error
        raise FetchError(f"Could not fetch {url}: {last_error}")

    def fetch_json(self, url: str) -> tuple[bytes, Any, int]:
        content, _headers, status = self.fetch_bytes(url)
        try:
            return content, json.loads(content.decode("utf-8")), status
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FetchError(f"Malformed JSON from {url}: {error}") from error

    def head(self, url: str) -> dict[str, Any]:
        """Verify an image URL resolves without downloading the payload.

        Image CDNs content-negotiate: the Met answers 406 Not Acceptable to an
        ``Accept: application/json`` request even though the image is fine, so
        image checks must advertise that they want an image.
        """
        try:
            _content, headers, status = self.fetch_bytes(
                url, method="HEAD", attempts=2, accept="image/*,*/*;q=0.8"
            )
        except Exception as error:  # noqa: BLE001 - one bad URL must not stop the sweep
            return {"checked_at": utc_now(), "ok": False, "error": str(error)[:200]}
        return {
            "checked_at": utc_now(),
            "ok": 200 <= status < 300,
            "http_status": status,
            "content_type": headers.get("Content-Type"),
            "content_length": headers.get("Content-Length"),
            "method": "HEAD",
        }


# --------------------------------------------------------------------------
# Raw snapshots
# --------------------------------------------------------------------------


class Snapshot:
    """Append-only raw response archive.

    Every institution response is written verbatim with its SHA-256 before any
    mapping happens, so a frozen collection can always be traced back to what
    the API actually returned.
    """

    def __init__(self, root: Path, source_id: str) -> None:
        self.root = root
        self.source_id = source_id
        self.id = timestamp_id()
        self.dir = root / "snapshots" / self.id / source_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict[str, Any]] = []

    def store(self, kind: str, name: str, url: str, content: bytes, status: int) -> Path:
        path = self.dir / kind / f"{slugify(name)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        self.entries.append(
            {
                "kind": kind,
                "name": name,
                "url": url,
                "httpStatus": status,
                "path": path.relative_to(self.root.parent).as_posix(),
                "sha256": sha256_bytes(content),
                "bytes": len(content),
                "accessedAt": utc_now(),
            }
        )
        return path

    def manifest(self) -> dict[str, Any]:
        return {
            "snapshotId": self.id,
            "sourceId": self.source_id,
            "createdAt": utc_now(),
            "importerVersion": IMPORTER_VERSION,
            "entryCount": len(self.entries),
            "entries": self.entries,
        }

    def finalize(self) -> None:
        write_json(self.dir / "raw_manifest.json", self.manifest())


# --------------------------------------------------------------------------
# Normalized record
# --------------------------------------------------------------------------


@dataclass
class EvidenceChunk:
    id: str
    text: str
    source_title: str
    source_url: str
    source_location: str
    supports: str
    kind: str  # tombstone | curatorial_text | context | acquisition
    license: str | None = None
    rights_uri: str | None = None
    source_kind: str | None = None

    def to_json(
        self,
        *,
        default_license: str | None = None,
        default_rights_uri: str | None = None,
    ) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "sourceTitle": self.source_title,
            "sourceUrl": self.source_url,
            "sourceLocation": self.source_location,
            "supports": self.supports,
            "kind": self.kind,
            "sourceKind": self.source_kind or evidence_source_kind(self.kind),
            "license": self.license or default_license,
            "rightsUri": self.rights_uri or default_rights_uri,
            # Institution prose is quoted verbatim, so it is not gated on a
            # separate human content review. See 01b §2.3.
            "reviewed": self.kind in {"tombstone", "curatorial_text", "acquisition"},
            "reviewStatus": "source_exact_match",
        }


@dataclass
class SourceObject:
    """One institution record mapped onto the project-wide schema."""

    id: str
    institution: str
    institution_id: str
    source_id: str
    accession_number: str
    title: str
    title_original: str | None
    date: str
    date_earliest: int | None
    date_latest: int | None
    creator: str
    medium: str
    type: str
    classification: str
    culture: str
    place: str
    description: str
    image_url: str
    image_url_large: str | None
    object_url: str
    rights: str
    rights_uri: str
    credit_line: str
    alt_text: str
    alt_text_source: str
    evidence: list[EvidenceChunk]
    source_api_url: str
    raw_record_path: str
    source_record_sha256: str
    # Additive field-level rights. ``rights`` / ``rights_uri`` stay serialized
    # for v2/v3 readers, but may no longer be interpreted as covering every
    # image and prose field in a record.
    image_license: str | None = None
    image_rights_uri: str | None = None
    metadata_license: str | None = None
    metadata_rights_uri: str | None = None
    curatorial_text_license: str | None = None
    curatorial_text_rights_uri: str | None = None
    # ``department`` is preserved as institution metadata.  Culture packs and
    # evidence domains are our own deterministic routing facets; neither is a
    # visitor-facing exhibition theme.
    department: str = ""
    culture_pack_ids: list[str] = field(default_factory=list)
    evidence_domain_ids: list[str] = field(default_factory=list)
    relation_facets: list[str] = field(default_factory=list)
    # v2 compatibility.  New imports mirror ``evidence_domain_ids`` here so
    # older runtime builds can still read the collection.
    themes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    image_validation: dict[str, Any] | None = None

    @property
    def evidence_depth(self) -> str:
        """How much independent prose backs this object.

        ``full`` objects may carry a core-evidence curatorial role; ``thin``
        objects are restricted to context and contrast roles because their
        evidence is metadata restatement rather than institution prose.
        """
        has_prose = any(chunk.kind == "curatorial_text" for chunk in self.evidence)
        if has_prose and len(self.evidence) >= 3:
            return "full"
        return "thin"

    def to_json(self) -> dict[str, Any]:
        evidence_domain_ids = self.evidence_domain_ids or self.themes
        field_rights = field_level_rights(
            institution_id=self.institution_id,
            legacy_rights=self.rights,
            legacy_rights_uri=self.rights_uri,
            has_curatorial_text=bool(self.description),
            image_license=self.image_license,
            image_rights_uri=self.image_rights_uri,
            metadata_license=self.metadata_license,
            metadata_rights_uri=self.metadata_rights_uri,
            curatorial_text_license=self.curatorial_text_license,
            curatorial_text_rights_uri=self.curatorial_text_rights_uri,
        )
        return {
            "id": self.id,
            "institution": self.institution,
            "institutionId": self.institution_id,
            "sourceId": self.source_id,
            "accessionNumber": self.accession_number,
            "title": self.title,
            "titleOriginal": self.title_original,
            "date": self.date,
            "dateEarliest": self.date_earliest,
            "dateLatest": self.date_latest,
            "creator": self.creator,
            "maker": self.creator,
            "medium": self.medium,
            "material": self.medium,
            "type": self.type,
            "classification": self.classification,
            "department": self.department,
            "culture": self.culture,
            "cultureDisplay": self.culture,
            "place": self.place,
            "description": self.description,
            "imageUrl": self.image_url,
            "imageUrlLarge": self.image_url_large,
            "objectUrl": self.object_url,
            "rights": self.rights,
            "rightsUri": self.rights_uri,
            **field_rights,
            "creditLine": self.credit_line,
            "altText": self.alt_text,
            "altTextSource": self.alt_text_source,
            "culturePackIds": self.culture_pack_ids,
            "evidenceDomainIds": evidence_domain_ids,
            "relationFacets": self.relation_facets,
            "themes": evidence_domain_ids,
            "tags": self.tags,
            "evidenceDepth": self.evidence_depth,
            "evidence": [
                chunk.to_json(
                    default_license=field_rights["metadataLicense"],
                    default_rights_uri=field_rights["metadataRightsUri"],
                )
                for chunk in self.evidence
            ],
            "sourceApiUrl": self.source_api_url,
            "rawRecordPath": self.raw_record_path,
            "sourceRecordSha256": self.source_record_sha256,
            "sourceAccessedAt": utc_now(),
            "imageValidation": self.image_validation,
        }


def evidence_source_kind(kind: str | None) -> str:
    """Describe what institution field a chunk quotes.

    ``kind`` remains the v3 evidence-depth/routing flag. ``sourceKind`` is an
    orthogonal provenance label used by rights audits and public citations.
    Adapters may override it when a legacy kind is deliberately broad (for
    example, an inscription classified as curatorial evidence).
    """

    if kind == "curatorial_text":
        return "institution_curatorial_text"
    if kind == "acquisition":
        return "institution_provenance"
    return "institution_metadata"


def _has_open_access_signal(
    institution_id: str | None,
    legacy_rights: str | None,
    legacy_rights_uri: str | None,
) -> bool:
    if (institution_id or "").casefold() not in FIELD_LEVEL_OPEN_ACCESS_INSTITUTIONS:
        return False
    signal = f"{legacy_rights or ''} {legacy_rights_uri or ''}".casefold()
    return "cc0" in signal or "public domain" in signal or "publicdomain/zero" in signal


def field_level_rights(
    *,
    institution_id: str | None,
    legacy_rights: str | None,
    legacy_rights_uri: str | None,
    has_curatorial_text: bool,
    image_license: str | None = None,
    image_rights_uri: str | None = None,
    metadata_license: str | None = None,
    metadata_rights_uri: str | None = None,
    curatorial_text_license: str | None = None,
    curatorial_text_rights_uri: str | None = None,
) -> dict[str, str | None]:
    """Resolve additive rights fields without widening unknown legacy rights.

    The fallback is deliberately institution-gated.  It exists only so frozen
    v2/v3 CMA, Met, and AIC records can be migrated without rewriting their raw
    snapshots. Explicit structured values always win.
    """

    institution = (institution_id or "").casefold()
    if _has_open_access_signal(institution, legacy_rights, legacy_rights_uri):
        image_license = image_license or CC0_LICENSE
        image_rights_uri = image_rights_uri or CC0_RIGHTS_URI
        metadata_license = metadata_license or CC0_LICENSE
        metadata_rights_uri = metadata_rights_uri or CC0_RIGHTS_URI
        if has_curatorial_text:
            if institution == "aic":
                curatorial_text_license = curatorial_text_license or CC_BY_4_0_LICENSE
                curatorial_text_rights_uri = (
                    curatorial_text_rights_uri or CC_BY_4_0_RIGHTS_URI
                )
            elif institution == "cma":
                curatorial_text_license = curatorial_text_license or CC0_LICENSE
                curatorial_text_rights_uri = (
                    curatorial_text_rights_uri or CC0_RIGHTS_URI
                )
    return {
        "imageLicense": image_license,
        "imageRightsUri": image_rights_uri,
        "metadataLicense": metadata_license,
        "metadataRightsUri": metadata_rights_uri,
        "curatorialTextLicense": curatorial_text_license,
        "curatorialTextRightsUri": curatorial_text_rights_uri,
    }


def _is_aic_description_evidence(raw: dict[str, Any]) -> bool:
    evidence_id = str(raw.get("id") or "").casefold()
    location = str(raw.get("sourceLocation") or raw.get("source_location") or "").casefold()
    return evidence_id.endswith(":description") or "curatorial description" in location


def _is_aic_short_description_evidence(raw: dict[str, Any]) -> bool:
    evidence_id = str(raw.get("id") or "").casefold()
    location = str(raw.get("sourceLocation") or raw.get("source_location") or "").casefold()
    return evidence_id.endswith(":short-description") or "short_description field" in location


def apply_rights_policy_to_json(raw: dict[str, Any]) -> dict[str, Any]:
    """Hydrate one normalized JSON record with the field-level rights contract.

    Used by serving-corpus assembly for old frozen objects.  It is idempotent,
    preserves explicit values, and never infers a license for an unknown
    institution. AIC ``description`` evidence is the sole CC BY exception;
    ``short_description`` and all other AIC artwork API fields are CC0 under
    the endpoint's licence statement.
    """

    hydrated = dict(raw)
    institution_id = str(
        hydrated.get("institutionId") or hydrated.get("institution_id") or ""
    ).casefold()
    raw_evidence = hydrated.get("evidence")
    curatorial_text_license = hydrated.get("curatorialTextLicense") or hydrated.get(
        "curatorial_text_license"
    )
    curatorial_text_rights_uri = hydrated.get(
        "curatorialTextRightsUri"
    ) or hydrated.get("curatorial_text_rights_uri")
    if institution_id == "aic" and isinstance(raw_evidence, list):
        evidence_dicts = [item for item in raw_evidence if isinstance(item, dict)]
        has_description = any(
            _is_aic_description_evidence(item) for item in evidence_dicts
        )
        has_short_description = any(
            _is_aic_short_description_evidence(item) for item in evidence_dicts
        )
        # New and rebuilt records preserve the fallback field identity. Old
        # ``:description`` records remain conservatively CC BY because their
        # pre-v4 mapper erased whether short_description supplied the prose.
        if has_short_description and not has_description:
            curatorial_text_license = curatorial_text_license or CC0_LICENSE
            curatorial_text_rights_uri = (
                curatorial_text_rights_uri or CC0_RIGHTS_URI
            )
    rights = field_level_rights(
        institution_id=institution_id,
        legacy_rights=hydrated.get("rights"),
        legacy_rights_uri=hydrated.get("rightsUri") or hydrated.get("rights_uri"),
        has_curatorial_text=bool(hydrated.get("description")),
        image_license=hydrated.get("imageLicense") or hydrated.get("image_license"),
        image_rights_uri=hydrated.get("imageRightsUri")
        or hydrated.get("image_rights_uri"),
        metadata_license=hydrated.get("metadataLicense")
        or hydrated.get("metadata_license"),
        metadata_rights_uri=hydrated.get("metadataRightsUri")
        or hydrated.get("metadata_rights_uri"),
        curatorial_text_license=curatorial_text_license,
        curatorial_text_rights_uri=curatorial_text_rights_uri,
    )
    hydrated.update(rights)

    evidence_items: list[dict[str, Any]] = []
    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            if not isinstance(item, dict):
                continue
            chunk = dict(item)
            kind = str(chunk.get("kind") or chunk.get("verification") or "")
            existing_source_kind = chunk.get("sourceKind") or chunk.get("source_kind")
            if institution_id == "aic" and str(chunk.get("id") or "").casefold().endswith(
                ":inscription"
            ):
                chunk["sourceKind"] = "institution_metadata"
            else:
                chunk["sourceKind"] = existing_source_kind or evidence_source_kind(kind)
            if institution_id == "aic" and _is_aic_description_evidence(chunk):
                if not chunk.get("license"):
                    chunk["license"] = CC_BY_4_0_LICENSE
                if not chunk.get("rightsUri") and not chunk.get("rights_uri"):
                    chunk["rightsUri"] = CC_BY_4_0_RIGHTS_URI
                chunk["sourceKind"] = "institution_curatorial_text"
            else:
                if not chunk.get("license"):
                    chunk["license"] = rights["metadataLicense"]
                if not chunk.get("rightsUri") and not chunk.get("rights_uri"):
                    chunk["rightsUri"] = rights["metadataRightsUri"]
            evidence_items.append(chunk)
        hydrated["evidence"] = evidence_items
    return hydrated


def parse_year_range(text: str) -> tuple[int | None, int | None]:
    """Best-effort year extraction from a free-text date display.

    Institutions publish dates as prose ("c. 1600–1650", "Tang dynasty
    (618–907)"). Only explicit 3–4 digit years are read; anything else stays
    ``None`` rather than being guessed.
    """
    if not text:
        return None, None
    negative = bool(re.search(r"\bB\.?C\.?E?\b", text, re.IGNORECASE))
    years = [int(match) for match in re.findall(r"\b(\d{3,4})\b", text)]
    years = [year for year in years if year <= 2100]
    if not years:
        return None, None
    earliest, latest = min(years), max(years)
    if negative:
        return -latest, -earliest
    return earliest, latest


def build_alt_text(
    title: str, date: str, medium: str, culture: str, provided: str | None
) -> tuple[str, str]:
    """Prefer institution-authored alt text; fall back to a metadata sentence."""
    cleaned = strip_html(provided or "")
    if cleaned and len(cleaned) > 25:
        return excerpt(cleaned, 300), "institution_authored"
    parts = [part for part in (title, date, culture, medium) if part]
    fallback = "，".join(parts)
    return (
        f"{fallback}。机构公开馆藏图片，未经生成或改绘。",
        "metadata_fallback",
    )


def dedupe_objects(objects: Iterable[SourceObject]) -> list[SourceObject]:
    """Drop repeated source records without erasing legitimate lookalikes.

    Generic titles such as ``Bowl`` or ``Untitled`` are common across global
    collections.  Title + maker + date therefore cannot be used as an identity
    key: it silently collapses distinct objects.  Cross-institution matching is
    retained for a future ``sameAs`` reconciliation pass; the serving corpus
    only deduplicates an institution's own stable source id.
    """
    seen: dict[tuple[str, str], SourceObject] = {}
    ordered: list[SourceObject] = []
    for obj in objects:
        key = (obj.institution_id, obj.source_id)
        if key in seen:
            # Keep whichever record carries more evidence.
            if len(obj.evidence) > len(seen[key].evidence):
                ordered[ordered.index(seen[key])] = obj
                seen[key] = obj
            continue
        seen[key] = obj
        ordered.append(obj)
    return ordered


def verify_images(
    objects: list[SourceObject],
    client: HttpClient,
    progress: Callable[[str], None] | None = None,
    workers: int = 8,
) -> list[SourceObject]:
    """HEAD every image URL and drop objects whose image does not resolve.

    A single pooled HTTP client is shared across workers so connections are
    reused.  Host-specific limiters are shared too; creating one throttled
    client per worker would multiply the advertised request rate and can get a
    public image CDN blocked.
    """
    lock = threading.Lock()
    host_locks: dict[str, threading.Lock] = {}
    host_last_request_at: dict[str, float] = {}
    aic_request_lock = threading.Lock()
    done = 0
    total = len(objects)

    # Conservative defaults for the first public sources.  They are not claims
    # about official hard limits; they are a polite project-side budget.
    host_intervals = {
        "images.metmuseum.org": 0.25,
        "collectionapi.metmuseum.org": 0.25,
        "www.artic.edu": 1.0,
        "openaccess-cdn.clevelandart.org": 0.10,
    }

    def throttle(host: str) -> None:
        with lock:
            host_lock = host_locks.setdefault(host, threading.Lock())
        with host_lock:
            interval = max(client.min_interval, host_intervals.get(host, 0.15))
            elapsed = time.monotonic() - host_last_request_at.get(host, 0.0)
            if elapsed < interval:
                time.sleep(interval - elapsed)
            host_last_request_at[host] = time.monotonic()

    pooled = httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept": "image/*,*/*;q=0.8"},
        follow_redirects=True,
        timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS),
        limits=httpx.Limits(max_connections=workers, max_keepalive_connections=workers),
    )

    def request_image(
        obj: SourceObject, host: str, request_headers: dict[str, str]
    ) -> dict[str, Any]:
        throttle(host)
        checked_at = utc_now()
        try:
            response = pooled.head(obj.image_url, headers=request_headers)
            # A small GET fallback handles image servers that reject HEAD but
            # happily serve the actual asset.  The body is streamed and never
            # downloaded in full.
            method = "HEAD"
            status_code = response.status_code
            headers = dict(response.headers)
            response.close()
            if status_code in {403, 405, 406}:
                throttle(host)
                with pooled.stream(
                    "GET",
                    obj.image_url,
                    headers={**request_headers, "Range": "bytes=0-1023"},
                ) as streamed:
                    method = "GET_RANGE"
                    status_code = streamed.status_code
                    headers = dict(streamed.headers)
            content_type = headers.get("content-type", "")
            return {
                "checked_at": checked_at,
                "ok": status_code in {200, 206}
                and content_type.casefold().startswith("image/"),
                "http_status": status_code,
                "content_type": content_type or None,
                "content_length": headers.get("content-length"),
                "method": method,
            }
        except Exception as error:  # noqa: BLE001 - one image must not abort the corpus
            return {
                "checked_at": checked_at,
                "ok": False,
                "error": str(error)[:200],
            }

    def check(obj: SourceObject) -> SourceObject:
        nonlocal done
        host = urllib.parse.urlsplit(obj.image_url).netloc.casefold()
        request_headers: dict[str, str] = {}
        if host == AIC_IMAGE_HOST:
            # AIC documents this project/contact header for API clients. It
            # also lets the IIIF edge distinguish an identified integration
            # from anonymous automation; no forged Referer, visitor URL,
            # cookie, or personal data is sent.
            request_headers = {
                "AIC-User-Agent": AIC_USER_AGENT,
            }
        # AIC explicitly asks image scrapers not to use parallel workers. Other
        # hosts keep the pooled parallel path.
        guard = aic_request_lock if host == AIC_IMAGE_HOST else nullcontext()
        with guard:
            obj.image_validation = request_image(obj, host, request_headers)
        with lock:
            done += 1
            if progress and done % 100 == 0:
                progress(f"  image check {done}/{total}")
        return obj

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            checked = list(pool.map(check, objects))
    finally:
        pooled.close()

    kept = [obj for obj in checked if (obj.image_validation or {}).get("ok")]
    if progress:
        failed = total - len(kept)
        by_host: dict[str, int] = {}
        for obj in checked:
            if not (obj.image_validation or {}).get("ok"):
                host = urllib.parse.urlsplit(obj.image_url).netloc
                by_host[host] = by_host.get(host, 0) + 1
        progress(f"  image check complete: {len(kept)} ok, {failed} unreachable")
        for host, count in sorted(by_host.items(), key=lambda pair: -pair[1]):
            progress(f"    {count:>5} unreachable from {host}")
    return kept
