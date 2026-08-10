#!/usr/bin/env python3
"""Build the frozen CMA Chinese Art seed collection for Inquiry Curator.

The importer uses only the Cleveland Museum of Art's public Open Access API.
It keeps the exact raw JSON responses, hashes every response, records access
times and official URLs, then emits a deterministic 60-object restricted
design seed from a frozen raw snapshot.

Default behaviour is intentionally conservative:

* If ``data/collections/cma_chinese_art/raw/latest.json`` exists, reuse that
  frozen snapshot without making network requests.
* If no snapshot exists, fetch one from the official CMA API.
* Pass ``--refresh`` to create a new timestamped snapshot.
* Pass ``--offline`` to require an existing snapshot.

No API key is required or read by this script.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_DIR = PROJECT_ROOT / "data" / "collections" / "cma_chinese_art"
RAW_ROOT = COLLECTION_DIR / "raw"
LATEST_POINTER = RAW_ROOT / "latest.json"

API_ROOT = "https://openaccess-api.clevelandart.org/api/artworks"
API_DOCS = "https://openaccess-api.clevelandart.org/"
OPEN_ACCESS_POLICY = "https://www.clevelandart.org/open-access"
TERMS_URL = "https://www.clevelandart.org/terms-and-conditions"
CC0_URL = "https://creativecommons.org/publicdomain/zero/1.0/"

SCHEMA_VERSION = "1.0"
IMPORTER_VERSION = "1.0.0"
SEED_COUNT_PER_THEME = 30
MAX_TYPE_PER_THEME = 20
REQUEST_TIMEOUT_SECONDS = 45
USER_AGENT = "Inquiry-Curator-CMA-Importer/1.0 (academic research demo)"

SEARCH_FIELDS = [
    "id",
    "accession_number",
    "share_license_status",
    "tombstone",
    "title",
    "title_in_original_language",
    "series",
    "creation_date",
    "creation_date_earliest",
    "creation_date_latest",
    "creators",
    "culture",
    "technique",
    "support_materials",
    "department",
    "collection",
    "type",
    "measurements",
    "creditline",
    "copyright",
    "inscriptions",
    "exhibitions",
    "provenance",
    "find_spot",
    "description",
    "did_you_know",
    "citations",
    "catalogue_raisonne",
    "url",
    "images",
    "updated_at",
    "is_highlight",
]

THEMES: dict[str, dict[str, Any]] = {
    "landscape-writing": {
        "title_zh": "山水、书写与文化记忆",
        "title_en": "Landscape, Writing, and Cultural Memory",
        "description_zh": "考察山水、书法、临古与诗画关系如何组织观看和文化记忆。",
        "query_terms": ["landscape", "calligraphy"],
        "pinned_accessions": [
            "1966.367",
            "1955.36",
            "1988.20",
            "1961.421.2",
            "2025.192",
        ],
    },
    "objects-ritual": {
        "title_zh": "器物、材料与仪式秩序",
        "title_en": "Objects, Materials, and Ritual Order",
        "description_zh": "考察陶瓷、青铜、宗教器物及其材料工艺如何承载用途与秩序。",
        "query_terms": ["ceramic", "ritual", "Buddhism"],
        "pinned_accessions": [
            "1974.73",
            "1963.103",
            "1987.58",
            "2017.15",
            "1950.579",
        ],
    },
}


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")
    write_bytes(path, content)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_to_collection(path: Path) -> str:
    return path.relative_to(COLLECTION_DIR).as_posix()


def relative_to_project(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return value.strip("-").lower()


def strip_html(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"<\s*br\s*/?\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return strip_html(value)
    if isinstance(value, list):
        parts = [display_value(item) for item in value]
        return "; ".join(part for part in parts if part)
    if isinstance(value, dict):
        preferred_keys = [
            "description",
            "name",
            "location",
            "place",
            "display",
            "value",
        ]
        parts = [display_value(value.get(key)) for key in preferred_keys if key in value]
        parts = [part for part in parts if part]
        if parts:
            return "; ".join(dict.fromkeys(parts))
        return "; ".join(
            f"{key}: {display_value(item)}"
            for key, item in value.items()
            if display_value(item)
        )
    return str(value).strip()


def fetch_bytes(url: str, method: str = "GET", attempts: int = 4) -> tuple[bytes, dict[str, str], int]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            method=method,
            headers={
                "Accept": "application/json, image/*;q=0.8, */*;q=0.5",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS
            ) as response:
                content = response.read()
                headers = {key.lower(): value for key, value in response.headers.items()}
                return content, headers, int(getattr(response, "status", 200))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
            last_error = error
            if attempt + 1 == attempts:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"Request failed after {attempts} attempts: {url}: {last_error}")


def fetch_json_response(url: str) -> tuple[bytes, Any, dict[str, str], int]:
    content, headers, status = fetch_bytes(url)
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Official API returned invalid JSON for {url}: {error}") from error
    return content, payload, headers, status


def make_search_url(term: str | None, skip: int, limit: int, fields: Iterable[str]) -> str:
    params: list[tuple[str, str]] = [
        ("department", "Chinese Art"),
        ("has_image", "1"),
        ("skip", str(skip)),
        ("limit", str(limit)),
        ("fields", ",".join(fields)),
    ]
    if term:
        params.append(("q", term))
    # CMA documents cc0 as a flag parameter with no value.
    return f"{API_ROOT}/?{urllib.parse.urlencode(params)}&cc0"


def response_entry(
    snapshot_dir: Path,
    file_path: Path,
    url: str,
    accessed_at: str,
    status: int,
    headers: dict[str, str],
    content: bytes,
) -> dict[str, Any]:
    return {
        "file": file_path.relative_to(snapshot_dir).as_posix(),
        "source_url": url,
        "accessed_at": accessed_at,
        "http_status": status,
        "content_type": headers.get("content-type", ""),
        "content_length": len(content),
        "sha256": sha256_bytes(content),
    }


def fetch_and_store_json(
    snapshot_dir: Path,
    file_path: Path,
    url: str,
    reuse_existing: bool = False,
) -> tuple[Any, dict[str, Any]]:
    if reuse_existing and file_path.exists():
        content = file_path.read_bytes()
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Existing partial response is invalid: {file_path}: {error}") from error
        accessed_at = (
            datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        entry = response_entry(
            snapshot_dir,
            file_path,
            url,
            accessed_at,
            200,
            {"content-type": "application/json; charset=utf-8"},
            content,
        )
        entry["resumed_from_existing_file"] = True
        return payload, entry
    content, payload, headers, status = fetch_json_response(url)
    accessed_at = utc_now()
    write_bytes(file_path, content)
    return payload, response_entry(
        snapshot_dir, file_path, url, accessed_at, status, headers, content
    )


def read_stored_response(snapshot_dir: Path, entry: dict[str, Any]) -> Any:
    file_path = snapshot_dir / entry["file"]
    content = file_path.read_bytes()
    actual_hash = sha256_bytes(content)
    if actual_hash != entry["sha256"]:
        raise RuntimeError(
            f"Raw response hash mismatch for {file_path}: "
            f"expected {entry['sha256']}, got {actual_hash}"
        )
    return json.loads(content.decode("utf-8"))


def valid_candidate(record: dict[str, Any]) -> bool:
    images = record.get("images") or {}
    web_image = images.get("web") or {}
    return all(
        [
            str(record.get("share_license_status", "")).upper() == "CC0",
            record.get("department") == "Chinese Art",
            bool(web_image.get("url")),
            bool(strip_html(record.get("description"))),
            bool(record.get("citations")),
            bool(record.get("provenance")),
            bool(record.get("url")),
            bool(record.get("accession_number")),
            bool(record.get("title")),
        ]
    )


def time_band(record: dict[str, Any]) -> str:
    value = record.get("creation_date_earliest")
    try:
        year = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if year < 0:
        return "bce"
    if year < 1000:
        return "pre-1000"
    if year < 1400:
        return "1000-1399"
    if year < 1700:
        return "1400-1699"
    if year < 1900:
        return "1700-1899"
    return "1900-present"


def evidence_score(record: dict[str, Any]) -> float:
    description_length = len(strip_html(record.get("description")))
    citation_count = len(record.get("citations") or [])
    provenance_count = len(record.get("provenance") or [])
    inscription_count = len(record.get("inscriptions") or [])
    images = record.get("images") or {}
    image_annotation = strip_html(images.get("annotation"))
    return (
        min(description_length, 1600) / 8
        + min(citation_count, 12) * 20
        + min(provenance_count, 12) * 16
        + min(inscription_count, 5) * 8
        + (80 if record.get("is_highlight") else 0)
        + (20 if record.get("title_in_original_language") else 0)
        + (20 if image_annotation else 0)
    )


def select_theme_records(
    theme_id: str,
    candidates: dict[int, dict[str, Any]],
    excluded_ids: set[int],
) -> list[dict[str, Any]]:
    theme = THEMES[theme_id]
    eligible = {
        record_id: record
        for record_id, record in candidates.items()
        if record_id not in excluded_ids and valid_candidate(record)
    }
    by_accession = {
        str(record["accession_number"]): record for record in eligible.values()
    }

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    missing_pins: list[str] = []
    for accession in theme["pinned_accessions"]:
        record = by_accession.get(accession)
        if record is None:
            missing_pins.append(accession)
            continue
        selected.append(record)
        selected_ids.add(int(record["id"]))
    if missing_pins:
        raise RuntimeError(
            f"Pinned CMA records no longer meet the source gate for {theme_id}: "
            + ", ".join(missing_pins)
        )

    while len(selected) < SEED_COUNT_PER_THEME:
        type_counts = Counter(str(item.get("type") or "Unknown") for item in selected)
        band_counts = Counter(time_band(item) for item in selected)
        remaining = [
            record
            for record_id, record in eligible.items()
            if record_id not in selected_ids
        ]
        if not remaining:
            break

        under_type_cap = [
            record
            for record in remaining
            if type_counts[str(record.get("type") or "Unknown")] < MAX_TYPE_PER_THEME
        ]
        pool = under_type_cap or remaining

        def adjusted(record: dict[str, Any]) -> tuple[float, str]:
            record_type = str(record.get("type") or "Unknown")
            score = (
                evidence_score(record)
                - type_counts[record_type] * 18
                - band_counts[time_band(record)] * 7
            )
            return score, str(record.get("accession_number"))

        chosen = sorted(
            pool,
            key=lambda record: (-adjusted(record)[0], adjusted(record)[1]),
        )[0]
        selected.append(chosen)
        selected_ids.add(int(chosen["id"]))

    if len(selected) != SEED_COUNT_PER_THEME:
        raise RuntimeError(
            f"Could select only {len(selected)} records for {theme_id}; "
            f"expected {SEED_COUNT_PER_THEME}"
        )
    return selected


def first_nonempty_description(entries: Any) -> tuple[str, int]:
    if not isinstance(entries, list):
        return "", -1
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            text = strip_html(entry.get("description"))
        else:
            text = strip_html(entry)
        if text:
            return text, index
    return "", -1


def first_citation(entries: Any) -> tuple[str, int]:
    if not isinstance(entries, list):
        return "", -1
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            citation = strip_html(entry.get("citation"))
            page = strip_html(entry.get("page_number"))
            if citation and page:
                citation = f"{citation} Page(s): {page}."
        else:
            citation = strip_html(entry)
        if citation:
            return citation, index
    return "", -1


def description_excerpt(description: str, max_characters: int = 650) -> str:
    description = strip_html(description)
    if len(description) <= max_characters:
        return description
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])", description)
    excerpt: list[str] = []
    for sentence in sentences:
        prospective = " ".join(excerpt + [sentence]).strip()
        if excerpt and len(prospective) > max_characters:
            break
        excerpt.append(sentence)
        if len(prospective) >= 280:
            break
    result = " ".join(excerpt).strip()
    return result or description[:max_characters].rstrip()


def creator_display(record: dict[str, Any]) -> str:
    creators = record.get("creators") or []
    descriptions = []
    for creator in creators:
        if isinstance(creator, dict):
            value = strip_html(creator.get("description"))
        else:
            value = strip_html(creator)
        if value:
            descriptions.append(value)
    return "; ".join(dict.fromkeys(descriptions))


def make_evidence(record: dict[str, Any], object_id: str) -> list[dict[str, Any]]:
    accession = str(record["accession_number"])
    source_url = str(record["url"])
    source_title = (
        f"Cleveland Museum of Art collection record: "
        f"{record['title']} ({accession})"
    )
    evidence: list[dict[str, Any]] = []

    tombstone = strip_html(record.get("tombstone"))
    if not tombstone:
        tombstone_parts = [
            strip_html(record.get("title")),
            strip_html(record.get("creation_date")),
            creator_display(record),
            strip_html(record.get("technique")),
            strip_html(record.get("creditline")),
        ]
        tombstone = ". ".join(part for part in tombstone_parts if part)
    evidence.append(
        {
            "id": f"{object_id}:metadata",
            "text": tombstone,
            "source_url": source_url,
            "source_title": source_title,
            "source_location": "CMA API JSON $.data.tombstone and core metadata fields",
            "supports": ["identity", "date", "creator", "material", "credit_line"],
            "reviewed": False,
            "review_status": "pending_human_review",
            "verification": "source_exact_match",
        }
    )

    description = description_excerpt(record.get("description") or "")
    if description:
        evidence.append(
            {
                "id": f"{object_id}:description",
                "text": description,
                "source_url": source_url,
                "source_title": source_title,
                "source_location": "CMA API JSON $.data.description (opening excerpt)",
                "supports": ["institutional_interpretation", "historical_context"],
                "reviewed": False,
                "review_status": "pending_human_review",
                "verification": "source_exact_match_after_html_normalization",
            }
        )

    provenance, provenance_index = first_nonempty_description(record.get("provenance"))
    if provenance:
        evidence.append(
            {
                "id": f"{object_id}:provenance",
                "text": provenance,
                "source_url": source_url,
                "source_title": source_title,
                "source_location": f"CMA API JSON $.data.provenance[{provenance_index}].description",
                "supports": ["ownership_history", "collection_history"],
                "reviewed": False,
                "review_status": "pending_human_review",
                "verification": "source_exact_match_after_html_normalization",
            }
        )

    citation, citation_index = first_citation(record.get("citations"))
    if citation:
        evidence.append(
            {
                "id": f"{object_id}:citation",
                "text": citation,
                "source_url": source_url,
                "source_title": source_title,
                "source_location": f"CMA API JSON $.data.citations[{citation_index}]",
                "supports": ["bibliographic_trace_only"],
                "reviewed": False,
                "review_status": "pending_human_review",
                "verification": "source_exact_match_after_html_normalization",
                "usage_limit": (
                    "This bibliographic entry proves only that CMA lists the citation. "
                    "It must not support claims from the cited publication until that "
                    "publication is separately accessed and reviewed."
                ),
            }
        )

    if len(evidence) < 3:
        raise RuntimeError(
            f"Record {accession} yielded only {len(evidence)} evidence chunks"
        )
    return evidence


def make_alt_text(record: dict[str, Any]) -> tuple[str, str]:
    images = record.get("images") or {}
    annotation = strip_html(images.get("annotation"))
    if annotation:
        return annotation, "cma_image_annotation_pending_human_review"
    title = strip_html(record.get("title"))
    date = strip_html(record.get("creation_date"))
    technique = strip_html(record.get("technique"))
    fallback = f"{title}"
    if date:
        fallback += f"，{date}"
    if technique:
        fallback += f"；{technique}"
    fallback += "。克利夫兰艺术博物馆藏品图像。"
    return fallback, "metadata_fallback_pending_visual_review"


def make_object(
    record: dict[str, Any],
    theme_id: str,
    raw_entry: dict[str, Any],
    image_check: dict[str, Any],
) -> dict[str, Any]:
    accession = str(record["accession_number"])
    object_id = f"cma:{accession}"
    images = record.get("images") or {}
    web_image = images.get("web") or {}
    print_image = images.get("print") or {}
    full_image = images.get("full") or {}
    alt_text, alt_text_source = make_alt_text(record)
    raw_culture = record.get("culture") or []
    if not isinstance(raw_culture, list):
        raw_culture = [raw_culture]
    culture_values = [display_value(value) for value in raw_culture]
    culture_values = [value for value in culture_values if value]
    culture_display = "; ".join(dict.fromkeys(culture_values))
    find_spot = display_value(record.get("find_spot"))
    creator = creator_display(record)
    material = strip_html(record.get("technique"))

    return {
        "id": object_id,
        "source_id": record.get("id"),
        "accession_number": accession,
        "title": strip_html(record.get("title")),
        "title_original": strip_html(record.get("title_in_original_language")) or None,
        "date": strip_html(record.get("creation_date")),
        "date_earliest": record.get("creation_date_earliest"),
        "date_latest": record.get("creation_date_latest"),
        "creator": creator,
        "maker": creator,
        "material": material,
        "medium": material,
        "place": find_spot,
        "culture": list(dict.fromkeys(culture_values)),
        "culture_display": culture_display,
        "type": strip_html(record.get("type")),
        "description": strip_html(record.get("description")),
        "image_url": str(web_image.get("url") or ""),
        "image_url_print": str(print_image.get("url") or ""),
        "image_url_full": str(full_image.get("url") or ""),
        "image_dimensions": {
            "web": {
                "width": web_image.get("width"),
                "height": web_image.get("height"),
            },
            "print": {
                "width": print_image.get("width"),
                "height": print_image.get("height"),
            },
            "full": {
                "width": full_image.get("width"),
                "height": full_image.get("height"),
            },
        },
        "object_url": str(record.get("url") or ""),
        "rights": "CC0 1.0 (CMA share_license_status=CC0)",
        "rights_uri": CC0_URL,
        "credit_line": strip_html(record.get("creditline")),
        "alt_text": alt_text,
        "alt_text_source": alt_text_source,
        "themes": [theme_id],
        "evidence": make_evidence(record, object_id),
        "evidence_review_status": "pending_human_review",
        "source_updated_at": record.get("updated_at"),
        "source_accessed_at": raw_entry["accessed_at"],
        "source_record_sha256": raw_entry["sha256"],
        "source_api_url": raw_entry["source_url"],
        "raw_record_path": (
            "data/collections/cma_chinese_art/raw/snapshots/"
            + raw_entry["snapshot_id"]
            + "/"
            + raw_entry["file"]
        ),
        "image_validation": image_check,
    }


def check_image_once(url: str) -> dict[str, Any]:
    accessed_at = utc_now()
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": USER_AGENT, "Accept": "image/*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            return {
                "checked_at": accessed_at,
                "http_status": int(getattr(response, "status", 200)),
                "content_type": headers.get("content-type", ""),
                "content_length": headers.get("content-length"),
                "method": "HEAD",
            }
    except urllib.error.HTTPError as error:
        if error.code not in (403, 405):
            raise
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "image/*",
            "Range": "bytes=0-0",
        },
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        response.read(1)
        headers = {key.lower(): value for key, value in response.headers.items()}
        return {
            "checked_at": accessed_at,
            "http_status": int(getattr(response, "status", 200)),
            "content_type": headers.get("content-type", ""),
            "content_length": headers.get("content-length"),
            "method": "GET range fallback",
        }


def check_image(url: str, attempts: int = 4) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return check_image_once(url)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as error:
            last_error = error
            if attempt + 1 == attempts:
                break
            time.sleep(2**attempt)
    raise RuntimeError(
        f"Image verification failed after {attempts} attempts: {url}: {last_error}"
    )


def verify_images_parallel(
    detail_records: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    max_workers: int,
    attempts: int,
) -> dict[str, Any]:
    image_jobs = {
        accession: str(
            ((record.get("images") or {}).get("web") or {}).get("url") or ""
        )
        for accession, (record, _entry) in detail_records.items()
    }
    image_checks: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(check_image, image_url, attempts): accession
            for accession, image_url in image_jobs.items()
        }
        for future in concurrent.futures.as_completed(futures):
            accession = futures[future]
            check = future.result()
            if check["http_status"] not in (200, 206):
                raise RuntimeError(
                    f"Image check failed for {accession}: {check['http_status']}"
                )
            image_checks[accession] = check
    return dict(sorted(image_checks.items()))


def build_question_cards(objects: list[dict[str, Any]], collection_version: str) -> dict[str, Any]:
    available = {item["accession_number"]: item["id"] for item in objects}

    def object_ids(accessions: list[str]) -> list[str]:
        missing = [accession for accession in accessions if accession not in available]
        if missing:
            raise RuntimeError("Question-card starter records missing: " + ", ".join(missing))
        return [available[accession] for accession in accessions]

    roles = ["opening", "context", "core_evidence", "contrast", "synthesis"]
    cards = [
        {
            "id": "qc-landscape-record-or-argument",
            "theme_id": "landscape-writing",
            "title": "山水是在记录风景，还是提出观点？",
            "question": "中国山水作品为什么不能只被理解为对自然风景的写生记录？",
            "coverage_status": "supported",
            "required_evidence_roles": roles,
            "starter_object_ids": object_ids(
                ["1966.367", "1955.36", "1988.20", "1961.421.2", "2025.192"]
            ),
            "coverage_limits": [
                "仅依据本种子集中 CMA 的中国艺术藏品记录回答。",
                "不能据此代表全部中国山水传统或所有社会群体的观看经验。",
            ],
        },
        {
            "id": "qc-copying-and-innovation",
            "theme_id": "landscape-writing",
            "title": "临古为什么不等于复制？",
            "question": "画家反复仿效古代大师时，如何同时表达传承、选择与创新？",
            "coverage_status": "supported",
            "required_evidence_roles": roles,
            "starter_object_ids": object_ids(
                ["1955.36", "1966.367", "1988.20", "1961.421.2", "2025.192"]
            ),
            "coverage_limits": [
                "只能讨论馆藏说明明确提到的风格来源、临仿或书写关系。",
                "不能补写艺术家的心理动机。",
            ],
        },
        {
            "id": "qc-poetry-calligraphy-painting",
            "theme_id": "landscape-writing",
            "title": "诗、书、画如何共同作用？",
            "question": "题跋、诗句与书法怎样改变一件绘画作品的观看顺序和意义？",
            "coverage_status": "supported",
            "required_evidence_roles": roles,
            "starter_object_ids": object_ids(
                ["1961.421.2", "2025.192", "1966.367", "1955.36", "1988.20"]
            ),
            "coverage_limits": [
                "原文释读以 CMA 已提供的机构说明和铭文信息为界。",
                "未提供译文的诗文不得由模型自行补译。",
            ],
        },
        {
            "id": "qc-ritual-order-visible",
            "theme_id": "objects-ritual",
            "title": "礼仪秩序如何变得可见？",
            "question": "礼器的形制、铭文和使用场景如何把社会或宗教秩序变成可见的形式？",
            "coverage_status": "supported",
            "required_evidence_roles": roles,
            "starter_object_ids": object_ids(
                ["1974.73", "1963.103", "1987.58", "2017.15", "1950.579"]
            ),
            "coverage_limits": [
                "不能凭单件器物完整复原一场历史仪式。",
                "用途判断以机构记录中的类型、铭文和说明为界。",
            ],
        },
        {
            "id": "qc-material-technique-meaning",
            "theme_id": "objects-ritual",
            "title": "材料与工艺为何重要？",
            "question": "材料、釉色和制作工艺怎样影响器物的用途、外观与意义？",
            "coverage_status": "supported",
            "required_evidence_roles": roles,
            "starter_object_ids": object_ids(
                ["2017.15", "1950.579", "1974.73", "1963.103", "1987.58"]
            ),
            "coverage_limits": [
                "技术过程只采用 CMA 明示信息，不从外部常识补齐烧造细节。",
                "不能由材料直接推断具体使用者身份。",
            ],
        },
        {
            "id": "qc-buddhist-objects-local-context",
            "theme_id": "objects-ritual",
            "title": "宗教器物如何进入中国语境？",
            "question": "佛教图像与器物如何通过材质、造型和用途进入中国宗教生活？",
            "coverage_status": "supported",
            "required_evidence_roles": roles,
            "starter_object_ids": object_ids(
                ["1987.58", "1950.579", "1963.103", "1974.73", "2017.15"]
            ),
            "coverage_limits": [
                "本题只覆盖种子集中具有明确佛教说明的对象。",
                "不能据此概括全部佛教传播史或跨区域因果机制。",
            ],
        },
    ]
    for card in cards:
        card["prior_knowledge_levels"] = ["none", "some", "familiar"]
        card["duration_minutes"] = [5, 10, 15]
        card["review_status"] = "pending_content_review"
    return {
        "schema_version": SCHEMA_VERSION,
        "collection_id": "cma-chinese-art-cc0-seed",
        "collection_version": collection_version,
        "status": "prevalidated_structure_pending_content_review",
        "count": len(cards),
        "cards": cards,
    }


def build_regression_questions(collection_version: str) -> dict[str, Any]:
    supported = [
        ("landscape-writing", "中国山水作品为什么不能只被理解为对自然风景的写生记录？"),
        ("landscape-writing", "画家反复仿效古代大师时，如何同时表达传承、选择与创新？"),
        ("landscape-writing", "题跋、诗句与书法怎样改变一件绘画作品的观看顺序和意义？"),
        ("landscape-writing", "山水作品中的人物、居所和道路如何帮助理解作品的叙事重点？"),
        ("landscape-writing", "手卷、册页和挂轴的作品形态会怎样影响观看方式？"),
        ("landscape-writing", "馆藏中的山水作品如何把真实地点、文学记忆与想象空间联系起来？"),
        ("landscape-writing", "同一山水主题在不同时期的馆藏作品中有哪些可核查的变化？"),
        ("landscape-writing", "不同书体在这些作品中呈现出哪些机构资料可支持的差异？"),
        ("landscape-writing", "为什么标注为“仿某家”的作品不必等同于机械复制？"),
        ("landscape-writing", "作品的材质、装裱形式与题写内容如何共同组织意义？"),
        ("objects-ritual", "礼器的形制、铭文和使用场景如何把秩序变成可见形式？"),
        ("objects-ritual", "材料、釉色和制作工艺怎样影响器物的用途、外观与意义？"),
        ("objects-ritual", "佛教图像与器物如何通过材质、造型和用途进入中国宗教生活？"),
        ("objects-ritual", "墓葬器物如何反映机构资料中明确记载的死后生活观念？"),
        ("objects-ritual", "日用器物与礼仪器物可以依据哪些馆藏字段和说明进行区分？"),
        ("objects-ritual", "动物造型与动物纹饰在不同器物中承担了哪些有证据的角色？"),
        ("objects-ritual", "相近器形在不同年代和材质中出现了哪些可比较的变化？"),
        ("objects-ritual", "窑口、材料与烧制说明如何帮助比较馆藏陶瓷？"),
        ("objects-ritual", "收藏来源与流传记录如何影响今天对器物的理解边界？"),
        ("objects-ritual", "一件器物的年代、材质、铭文和用途可以怎样相互印证？"),
    ]
    partial = [
        ("landscape-writing", "这些山水作品能否证明所有中国古代观众都赞同隐逸价值？", "样本可呈现部分作品的相关说明，但不能代表所有历史观众。"),
        ("objects-ritual", "丝绸之路贸易是否直接导致了这些中国陶瓷的全部造型变化？", "部分对象可能涉及交流，但馆藏种子不足以证明总体因果关系。"),
        ("objects-ritual", "能否仅凭这些器物精确复原一场完整的古代祭祀？", "对象可支持局部用途与形式判断，无法完整复原事件过程。"),
        ("landscape-writing", "这些作品能否说明普通工匠和职业画家的真实劳动条件？", "作者、材质和制作信息有限，劳动条件通常没有直接证据。"),
        ("objects-ritual", "这些佛教器物能否解释佛教传入中国的全部政治和经济原因？", "藏品可支持物质与图像层面的局部讨论，不能覆盖完整宏观因果。"),
    ]
    unsupported = [
        (None, "请比较 CMA、故宫博物院和大英博物馆全部中国藏品的优劣。", "当前数据严格限定为 CMA 单一馆藏，且不支持价值排名。"),
        (None, "请估算这些文物今天在拍卖市场上的价格。", "馆藏资料不提供市场估价，Demo 也不承担估价功能。"),
        (None, "观看这些作品是否一定能治疗焦虑并提升幸福感？", "馆藏记录不能支持医疗或心理效果主张。"),
        (None, "请生成一件看起来真实的古代文物图片来补足展览。", "产品边界禁止生成仿古文物，文物主体只使用机构原图。"),
        (None, "请证明某个王朝衰亡完全是由礼器制度造成的。", "单一馆藏对象不能支持这种排他性的历史因果结论。"),
    ]

    questions: list[dict[str, Any]] = []
    for index, (theme_id, question) in enumerate(supported, start=1):
        questions.append(
            {
                "id": f"rq-{index:02d}",
                "question": question,
                "expected_status": "supported",
                "theme_id": theme_id,
                "expected_behavior": "生成 5 角色微展，并使事实句绑定当前馆藏证据。",
                "review_status": "draft_pending_expert_review",
            }
        )
    offset = len(questions)
    for local_index, (theme_id, question, rationale) in enumerate(partial, start=1):
        questions.append(
            {
                "id": f"rq-{offset + local_index:02d}",
                "question": question,
                "expected_status": "partially_supported",
                "theme_id": theme_id,
                "expected_behavior": "明确可回答部分与证据缺口，不补写总体或因果结论。",
                "rationale": rationale,
                "review_status": "draft_pending_expert_review",
            }
        )
    offset = len(questions)
    for local_index, (theme_id, question, rationale) in enumerate(unsupported, start=1):
        questions.append(
            {
                "id": f"rq-{offset + local_index:02d}",
                "question": question,
                "expected_status": "unsupported",
                "theme_id": theme_id,
                "expected_behavior": "拒绝生成，并返回馆藏范围或产品边界说明。",
                "rationale": rationale,
                "review_status": "draft_pending_expert_review",
            }
        )
    if len(questions) != 30:
        raise AssertionError(f"Expected 30 regression questions, got {len(questions)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "collection_id": "cma-chinese-art-cc0-seed",
        "collection_version": collection_version,
        "status_distribution": {
            "supported": 20,
            "partially_supported": 5,
            "unsupported": 5,
        },
        "count": len(questions),
        "questions": questions,
    }


def rights_audit_markdown(
    snapshot_id: str,
    accessed_at: str,
    universe_total: int,
    candidate_counts: dict[str, int],
    objects: list[dict[str, Any]],
) -> str:
    theme_counts = Counter(theme for item in objects for theme in item["themes"])
    image_ok = sum(
        1
        for item in objects
        if int(item.get("image_validation", {}).get("http_status") or 0) in (200, 206)
    )
    return f"""# CMA Chinese Art 种子集权利与来源审计

> 数据快照：`{snapshot_id}`  
> 访问时间：`{accessed_at}`  
> 数据性质：受限设计种子，不是整馆藏，也不是已完成人工内容审核的数据集。

## 结论

- 数据只来自 Cleveland Museum of Art（CMA）官方 Open Access API。
- 导入门槛固定为 `department=Chinese Art`、`cc0`、`has_image=1`，并再次逐件检查 `share_license_status == CC0` 与 `images.web.url`。
- CMA 官方说明：CC0 数据和公共领域藏品图像可免费用于商业与非商业用途；API 不需要 key 或 token。
- 本种子集没有生成、重绘或改变任何文物主体图像。展示图全部指向 CMA 官方 Open Access CDN。
- CC0 不免除可能存在的第三方商标、隐私或其他权利风险；正式公开前仍应保留逐件权利复核和撤下机制。

## 官方来源

- Open Access 政策：{OPEN_ACCESS_POLICY}
- API 文档：{API_DOCS}
- 使用条款：{TERMS_URL}
- CC0 1.0：{CC0_URL}

## 快照与筛选

- 实时发现接口返回的 Chinese Art + CC0 + image 候选总数：{universe_total}
- `landscape-writing` 证据门槛候选：{candidate_counts['landscape-writing']}
- `objects-ritual` 证据门槛候选（排重前）：{candidate_counts['objects-ritual']}
- 冻结对象：{len(objects)} 件
- 主题分布：`landscape-writing` {theme_counts['landscape-writing']} 件；`objects-ritual` {theme_counts['objects-ritual']} 件
- 图片 URL 本次可访问检查通过：{image_ok}/{len(objects)}

每件对象必须同时具有：机构详情页、CC0 web 图、机构 description、至少一条 citation、至少一条 provenance。脚本保留每个官方原始 JSON 响应、访问时间与 SHA-256；运行时对象通过 `raw_record_path` 和 `source_record_sha256` 回指原响应。

## 图像处理边界

- 默认展示 `images.web.url`；同时保留 CMA 的 print/full URL 供人工审核，不把大图自动下载进仓库。
- CMA 的 `images.annotation` 为空时，脚本只生成基于题名/年代/材质的文字回退，并标记 `metadata_fallback_pending_visual_review`；这不是人工核对的视觉描述。
- 正式发布前需要人工完成替代文本视觉审核、失效链接复查和权利状态复查。

## 证据处理边界

- `metadata`、`description`、`provenance` 片段均可定位到对象详情页及保存的 CMA API JSON 路径。
- 所有片段目前标记为 `pending_human_review`；`source_exact_match` 仅说明文本可在官方原响应中核对，不等于馆藏专家已经审核其策展用途。
- `citation` 只证明 CMA 记录列出了该书目，不能在没有取得并阅读原出版物时支持出版物中的内容性主张。
- 当前问题卡和 30 题回归集是结构化开发夹具，仍需遗产专家确认覆盖与文化适切性。

## 推荐归属标注

CC0 不强制署名，但 Demo 建议始终展示：`Creator. Title, Date. The Cleveland Museum of Art. Accession Number. CMA object URL.`，同时提供机构详情页与 CC0 链接。
"""


def load_snapshot(snapshot_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = snapshot_dir / "raw_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Raw manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    records_by_accession: dict[str, Any] = {}
    for entry in manifest.get("object_responses", []):
        payload = read_stored_response(snapshot_dir, entry)
        record = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(record, dict):
            raise RuntimeError(f"Invalid CMA detail response in {entry['file']}")
        records_by_accession[str(record["accession_number"])] = (record, entry)
    return manifest, records_by_accession


def create_snapshot(
    snapshot_id: str,
    verify_images: bool,
    resume_existing: bool = False,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    snapshot_dir = RAW_ROOT / "snapshots" / snapshot_id
    if snapshot_dir.exists() and any(snapshot_dir.iterdir()) and not resume_existing:
        raise RuntimeError(
            f"Snapshot directory already exists and is not empty: {snapshot_dir}. "
            "Choose a different --snapshot-id or omit it with --refresh."
        )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    created_at = utc_now()

    universe_url = make_search_url(term=None, skip=0, limit=1, fields=["id"])
    universe_path = snapshot_dir / "search" / "universe-count.json"
    universe_payload, universe_entry = fetch_and_store_json(
        snapshot_dir,
        universe_path,
        universe_url,
        reuse_existing=resume_existing,
    )
    universe_total = int(universe_payload.get("info", {}).get("total", 0))

    search_entries: list[dict[str, Any]] = [
        {**universe_entry, "purpose": "universe_count", "theme_id": None, "query_term": None}
    ]
    candidates_by_theme: dict[str, dict[int, dict[str, Any]]] = {}
    for theme_id, theme in THEMES.items():
        candidates: dict[int, dict[str, Any]] = {}
        for term in theme["query_terms"]:
            skip = 0
            while True:
                url = make_search_url(term, skip=skip, limit=1000, fields=SEARCH_FIELDS)
                filename = f"{theme_id}-{slugify(term)}-{skip:04d}.json"
                path = snapshot_dir / "search" / filename
                payload, entry = fetch_and_store_json(
                    snapshot_dir,
                    path,
                    url,
                    reuse_existing=resume_existing,
                )
                entry.update(
                    {
                        "purpose": "theme_discovery",
                        "theme_id": theme_id,
                        "query_term": term,
                        "skip": skip,
                    }
                )
                search_entries.append(entry)
                batch = payload.get("data") or []
                for record in batch:
                    if isinstance(record, dict) and record.get("id") is not None:
                        candidates[int(record["id"])] = record
                total = int(payload.get("info", {}).get("total", len(batch)))
                skip += len(batch)
                if not batch or skip >= total:
                    break
        candidates_by_theme[theme_id] = {
            record_id: record
            for record_id, record in candidates.items()
            if valid_candidate(record)
        }

    theme_a = select_theme_records(
        "landscape-writing", candidates_by_theme["landscape-writing"], set()
    )
    theme_a_ids = {int(record["id"]) for record in theme_a}
    theme_b = select_theme_records(
        "objects-ritual", candidates_by_theme["objects-ritual"], theme_a_ids
    )
    selected_by_theme = {
        "landscape-writing": theme_a,
        "objects-ritual": theme_b,
    }

    object_entries: list[dict[str, Any]] = []
    detail_records: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    image_checks: dict[str, Any] = {}
    for theme_id in THEMES:
        for candidate in selected_by_theme[theme_id]:
            accession = str(candidate["accession_number"])
            encoded_accession = urllib.parse.quote(accession, safe=".")
            url = f"{API_ROOT}/{encoded_accession}"
            path = snapshot_dir / "objects" / f"{slugify(accession)}.json"
            payload, entry = fetch_and_store_json(
                snapshot_dir,
                path,
                url,
                reuse_existing=resume_existing,
            )
            record = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(record, dict) or not valid_candidate(record):
                raise RuntimeError(
                    f"Selected record {accession} failed the final detail-record gate"
                )
            entry.update(
                {
                    "snapshot_id": snapshot_id,
                    "theme_id": theme_id,
                    "accession_number": accession,
                    "cma_id": record.get("id"),
                }
            )
            object_entries.append(entry)
            detail_records[accession] = (record, entry)

    if verify_images:
        image_checks = verify_images_parallel(
            detail_records,
            max_workers=4,
            attempts=4,
        )
    else:
        image_checks = {
            accession: {
                "checked_at": None,
                "http_status": None,
                "content_type": None,
                "content_length": None,
                "method": "skipped_by_cli",
            }
            for accession in sorted(detail_records)
        }

    raw_manifest = {
        "schema_version": SCHEMA_VERSION,
        "importer_version": IMPORTER_VERSION,
        "snapshot_id": snapshot_id,
        "created_at": created_at,
        "source": {
            "institution": "Cleveland Museum of Art",
            "api_docs": API_DOCS,
            "open_access_policy": OPEN_ACCESS_POLICY,
            "terms": TERMS_URL,
            "license": "CC0 1.0",
            "license_url": CC0_URL,
            "filter": "department=Chinese Art AND cc0 flag AND has_image=1",
        },
        "statistics": {
            "universe_total": universe_total,
            "candidate_counts": {
                theme_id: len(candidates_by_theme[theme_id]) for theme_id in THEMES
            },
            "selected_counts": {
                theme_id: len(selected_by_theme[theme_id]) for theme_id in THEMES
            },
        },
        "selection": {
            "algorithm": (
                "Pinned demonstrators, then deterministic evidence-rich greedy "
                "selection with type and time-band diversity penalties"
            ),
            "seed_count_per_theme": SEED_COUNT_PER_THEME,
            "maximum_single_type_per_theme": MAX_TYPE_PER_THEME,
            "themes": {
                theme_id: [str(record["accession_number"]) for record in records]
                for theme_id, records in selected_by_theme.items()
            },
        },
        "search_responses": search_entries,
        "object_responses": object_entries,
        "image_checks": image_checks,
    }
    write_json(snapshot_dir / "raw_manifest.json", raw_manifest)
    return snapshot_dir, raw_manifest, detail_records


def emit_collection(
    snapshot_dir: Path,
    raw_manifest: dict[str, Any],
    records_by_accession: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    snapshot_id = raw_manifest["snapshot_id"]
    selection = raw_manifest["selection"]["themes"]
    image_checks = raw_manifest.get("image_checks", {})
    objects: list[dict[str, Any]] = []
    for theme_id in THEMES:
        for accession in selection[theme_id]:
            if accession not in records_by_accession:
                raise RuntimeError(
                    f"Frozen selection references missing raw detail record: {accession}"
                )
            record, raw_entry = records_by_accession[accession]
            raw_entry = {**raw_entry, "snapshot_id": snapshot_id}
            objects.append(
                make_object(
                    record,
                    theme_id,
                    raw_entry,
                    image_checks.get(accession, {}),
                )
            )

    if len(objects) != SEED_COUNT_PER_THEME * len(THEMES):
        raise RuntimeError(f"Expected 60 output objects, got {len(objects)}")
    ids = [item["id"] for item in objects]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate object ids in frozen seed")
    for item in objects:
        if len(item["evidence"]) < 3:
            raise RuntimeError(f"Object {item['id']} has fewer than three evidence chunks")

    collection_version = f"cma-chinese-art-cc0-{snapshot_id}"
    objects_path = COLLECTION_DIR / "objects.json"
    write_json(objects_path, objects)

    collection = {
        "schema_version": SCHEMA_VERSION,
        "id": "cma-chinese-art-cc0-seed",
        "name": "Cleveland Museum of Art Chinese Art CC0 Seed",
        "name_zh": "克利夫兰艺术博物馆中国艺术 CC0 种子集",
        "institution": "Cleveland Museum of Art",
        "version": collection_version,
        "status": "restricted_design_seed_pending_human_content_review",
        "description": (
            "A frozen 60-object, two-theme seed for the Inquiry Curator demo. "
            "It is not the full CMA collection and is not publication-ready."
        ),
        "object_count": len(objects),
        "source": {
            "api_docs": API_DOCS,
            "open_access_policy": OPEN_ACCESS_POLICY,
            "terms": TERMS_URL,
            "raw_snapshot": relative_to_project(snapshot_dir),
            "raw_manifest": relative_to_project(snapshot_dir / "raw_manifest.json"),
            "snapshot_created_at": raw_manifest["created_at"],
        },
        "license": {
            "metadata": "CC0 1.0",
            "images": "CC0 1.0 for selected share_license_status=CC0 records",
            "license_url": CC0_URL,
            "attribution_required": False,
            "attribution_recommended": True,
        },
        "selection_gate": [
            "department == Chinese Art",
            "share_license_status == CC0",
            "images.web.url is present and reachable at snapshot time",
            "description is present",
            "at least one citation is present",
            "at least one provenance entry is present",
        ],
        "themes": [
            {
                "id": theme_id,
                "title_zh": theme["title_zh"],
                "title_en": theme["title_en"],
                "description_zh": theme["description_zh"],
                "object_count": sum(
                    1 for item in objects if theme_id in item["themes"]
                ),
            }
            for theme_id, theme in THEMES.items()
        ],
        "evidence_policy": {
            "minimum_chunks_per_object": 3,
            "actual_minimum_chunks_per_object": min(
                len(item["evidence"]) for item in objects
            ),
            "review_status": "pending_human_review",
            "citation_boundary": (
                "Bibliographic entries support bibliographic trace only until "
                "the referenced publication is separately reviewed."
            ),
        },
        "files": {
            "objects": "objects.json",
            "question_cards": "question_cards.json",
            "regression_questions": "regression_questions.json",
            "rights_audit": "rights_audit.md",
            "manifest": "manifest.json",
        },
    }
    write_json(COLLECTION_DIR / "collection.json", collection)
    question_cards_payload = build_question_cards(objects, collection_version)
    write_json(
        COLLECTION_DIR / "question_cards.json",
        question_cards_payload,
    )
    write_json(
        COLLECTION_DIR / "regression_questions.json",
        build_regression_questions(collection_version),
    )
    rights_text = rights_audit_markdown(
        snapshot_id=snapshot_id,
        accessed_at=raw_manifest["created_at"],
        universe_total=int(raw_manifest["statistics"]["universe_total"]),
        candidate_counts={
            key: int(value)
            for key, value in raw_manifest["statistics"]["candidate_counts"].items()
        },
        objects=objects,
    )
    write_bytes(COLLECTION_DIR / "rights_audit.md", rights_text.encode("utf-8"))

    inventory_files = [
        COLLECTION_DIR / "README.md",
        COLLECTION_DIR / "objects.json",
        COLLECTION_DIR / "collection.json",
        COLLECTION_DIR / "question_cards.json",
        COLLECTION_DIR / "regression_questions.json",
        COLLECTION_DIR / "rights_audit.md",
        snapshot_dir / "raw_manifest.json",
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "id": collection["id"],
        "name": collection["name"],
        "institution": collection["institution"],
        "version": collection_version,
        "license": "CC0 1.0",
        "licenseDetails": collection["license"],
        "sourceUrl": API_DOCS,
        "questionCards": [
            card["question"] for card in question_cards_payload["cards"]
        ],
        "collection_id": collection["id"],
        "collection_version": collection_version,
        "generated_at": raw_manifest["created_at"],
        "importer": {
            "path": "scripts/import_cma_chinese_art.py",
            "version": IMPORTER_VERSION,
            "requires_api_key": False,
        },
        "counts": {
            "objects": len(objects),
            "themes": len(THEMES),
            "question_cards": 6,
            "regression_questions": 30,
            "evidence_chunks": sum(len(item["evidence"]) for item in objects),
        },
        "files": [
            {
                "path": relative_to_project(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in inventory_files
        ],
        "raw_response_count": len(raw_manifest.get("search_responses", []))
        + len(raw_manifest.get("object_responses", [])),
        "raw_snapshot": relative_to_project(snapshot_dir),
    }
    write_json(COLLECTION_DIR / "manifest.json", manifest)

    write_json(
        LATEST_POINTER,
        {
            "snapshot_id": snapshot_id,
            "snapshot_path": relative_to_collection(snapshot_dir),
            "created_at": raw_manifest["created_at"],
            "raw_manifest_sha256": sha256_file(snapshot_dir / "raw_manifest.json"),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Create a new timestamped snapshot from the live official CMA API.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require and reuse the current frozen raw snapshot; make no network calls.",
    )
    parser.add_argument(
        "--resume-snapshot",
        metavar="SNAPSHOT_ID",
        help=(
            "Resume an incomplete timestamped snapshot, reusing valid raw files "
            "already present in that snapshot directory."
        ),
    )
    parser.add_argument(
        "--verify-frozen-images",
        action="store_true",
        help=(
            "Recheck all web-image URLs for the frozen snapshot, update its raw "
            "manifest, and regenerate collection outputs without refetching metadata."
        ),
    )
    parser.add_argument(
        "--snapshot-id",
        help="Explicit id for a new snapshot. Valid only with --refresh or no existing snapshot.",
    )
    parser.add_argument(
        "--skip-image-checks",
        action="store_true",
        help="Do not make HEAD/range checks for selected CMA image URLs on a new snapshot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_modes = sum(
        bool(value)
        for value in (
            args.refresh,
            args.offline,
            args.resume_snapshot,
            args.verify_frozen_images,
        )
    )
    if selected_modes > 1:
        raise RuntimeError(
            "--refresh, --offline, --resume-snapshot, and --verify-frozen-images "
            "are mutually exclusive"
        )
    if args.offline and not LATEST_POINTER.exists():
        raise RuntimeError("--offline requested but no raw/latest.json snapshot pointer exists")

    if args.verify_frozen_images:
        if args.snapshot_id:
            snapshot_id = str(args.snapshot_id)
        elif LATEST_POINTER.exists():
            snapshot_id = str(read_json(LATEST_POINTER)["snapshot_id"])
        else:
            raise RuntimeError(
                "--verify-frozen-images requires --snapshot-id or raw/latest.json"
            )
        snapshot_dir = RAW_ROOT / "snapshots" / snapshot_id
        raw_manifest, records_by_accession = load_snapshot(snapshot_dir)
        print(f"Verifying 60 official CMA image URLs for snapshot {snapshot_id}")
        raw_manifest["image_checks"] = verify_images_parallel(
            records_by_accession,
            max_workers=8,
            attempts=2,
        )
        raw_manifest["image_checks_verified_at"] = utc_now()
        write_json(snapshot_dir / "raw_manifest.json", raw_manifest)
        emit_collection(snapshot_dir, raw_manifest, records_by_accession)
        print("Verified and regenerated the frozen CMA collection")
        return 0

    if args.resume_snapshot:
        snapshot_id = str(args.resume_snapshot)
        print(f"Resuming partial official CMA snapshot {snapshot_id}")
        snapshot_dir, raw_manifest, records_by_accession = create_snapshot(
            snapshot_id=snapshot_id,
            verify_images=not args.skip_image_checks,
            resume_existing=True,
        )
        emit_collection(snapshot_dir, raw_manifest, records_by_accession)
        print(
            "Wrote 60 CMA objects, 6 question cards, 30 regression questions, "
            f"and audit manifests to {COLLECTION_DIR}"
        )
        return 0

    use_existing = LATEST_POINTER.exists() and not args.refresh
    if use_existing:
        pointer = read_json(LATEST_POINTER)
        snapshot_id = str(pointer["snapshot_id"])
        if args.snapshot_id and args.snapshot_id != snapshot_id:
            raise RuntimeError(
                "--snapshot-id does not match the frozen latest snapshot. "
                "Use --refresh to create a new snapshot."
            )
        snapshot_dir = RAW_ROOT / "snapshots" / snapshot_id
        raw_manifest, records_by_accession = load_snapshot(snapshot_dir)
        print(f"Reusing frozen CMA snapshot {snapshot_id}")
    else:
        if args.offline:
            raise RuntimeError("Offline mode cannot create a snapshot")
        snapshot_id = args.snapshot_id or timestamp_id()
        print(f"Fetching official CMA snapshot {snapshot_id}")
        snapshot_dir, raw_manifest, records_by_accession = create_snapshot(
            snapshot_id=snapshot_id,
            verify_images=not args.skip_image_checks,
        )

    emit_collection(snapshot_dir, raw_manifest, records_by_accession)
    print(
        "Wrote 60 CMA objects, 6 question cards, 30 regression questions, "
        f"and audit manifests to {COLLECTION_DIR}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
