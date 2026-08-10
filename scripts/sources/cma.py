"""Cleveland Museum of Art Open Access adapter.

CMA publishes the richest prose of the three institutions (tombstone,
curatorial description, "did you know", provenance and citations), so its
objects are the main source of ``full`` evidence-depth records that may carry
a core-evidence curatorial role.

API: https://openaccess-api.clevelandart.org/
Licence: objects filtered to ``share_license_status == "CC0"``.
"""

from __future__ import annotations

import urllib.parse
from typing import Any, Callable

from .base import (
    EvidenceChunk,
    HttpClient,
    Snapshot,
    SourceObject,
    build_alt_text,
    display_value,
    excerpt,
    parse_year_range,
    sha256_bytes,
    strip_html,
)

INSTITUTION = "Cleveland Museum of Art"
INSTITUTION_ID = "cma"
API_ROOT = "https://openaccess-api.clevelandart.org/api/artworks"
RIGHTS = "CC0 1.0 (CMA share_license_status=CC0)"
RIGHTS_URI = "https://creativecommons.org/publicdomain/zero/1.0/"

DEPARTMENTS = ("Chinese Art", "Japanese Art", "Korean Art")
PAGE_SIZE = 100


def _search_url(department: str, skip: int) -> str:
    query = {
        "department": department,
        "cc0": "1",
        "has_image": "1",
        "limit": str(PAGE_SIZE),
        "skip": str(skip),
        "indent": "1",
    }
    return f"{API_ROOT}?{urllib.parse.urlencode(query)}"


def _creator(record: dict[str, Any]) -> str:
    creators = record.get("creators")
    if isinstance(creators, list) and creators:
        described = display_value(creators[0].get("description"))
        if described:
            return described
    return ""


def _image_url(record: dict[str, Any], size: str) -> str | None:
    images = record.get("images")
    if not isinstance(images, dict):
        return None
    entry = images.get(size)
    if isinstance(entry, dict):
        url = entry.get("url")
        if isinstance(url, str) and url.startswith("http"):
            return url
    return None


def _evidence(record: dict[str, Any], object_id: str, object_url: str) -> list[EvidenceChunk]:
    """Split the institution record into independently citable chunks.

    Every chunk quotes CMA text verbatim; nothing is paraphrased here.
    """
    chunks: list[EvidenceChunk] = []

    tombstone = strip_html(record.get("tombstone"))
    if tombstone:
        chunks.append(
            EvidenceChunk(
                id=f"{object_id}:tombstone",
                text=tombstone,
                source_title=f"CMA object record {record.get('accession_number')}",
                source_url=object_url,
                source_location="Tombstone (title, date, maker, medium, dimensions)",
                supports="题名、年代、作者、材质与尺寸等机构著录事实",
                kind="tombstone",
            )
        )

    description = strip_html(record.get("description"))
    if description:
        chunks.append(
            EvidenceChunk(
                id=f"{object_id}:description",
                text=excerpt(description),
                source_title=f"CMA curatorial description {record.get('accession_number')}",
                source_url=object_url,
                source_location="Curatorial description",
                supports="机构撰写的作品说明与背景解释",
                kind="curatorial_text",
            )
        )

    did_you_know = strip_html(record.get("did_you_know"))
    if did_you_know:
        chunks.append(
            EvidenceChunk(
                id=f"{object_id}:context",
                text=excerpt(did_you_know, 420),
                source_title=f"CMA context note {record.get('accession_number')}",
                source_url=object_url,
                source_location="Did you know",
                supports="机构补充的背景与关联信息",
                kind="curatorial_text",
            )
        )

    culture = display_value(record.get("culture"))
    collection = display_value(record.get("collection"))
    context_parts = [part for part in (culture, collection, display_value(record.get("technique"))) if part]
    if context_parts:
        chunks.append(
            EvidenceChunk(
                id=f"{object_id}:classification",
                text="；".join(context_parts),
                source_title=f"CMA classification {record.get('accession_number')}",
                source_url=object_url,
                source_location="Culture / collection / technique fields",
                supports="文化归属、馆藏分类与工艺著录",
                kind="context",
            )
        )

    credit = strip_html(record.get("creditline"))
    if credit:
        chunks.append(
            EvidenceChunk(
                id=f"{object_id}:acquisition",
                text=credit,
                source_title=f"CMA credit line {record.get('accession_number')}",
                source_url=object_url,
                source_location="Credit line",
                supports="入藏方式与归属标注",
                kind="acquisition",
            )
        )

    return chunks


def _map(record: dict[str, Any], raw_path: str, raw_sha: str, api_url: str) -> SourceObject | None:
    accession = display_value(record.get("accession_number"))
    title = strip_html(record.get("title"))
    object_url = display_value(record.get("url"))
    image_url = _image_url(record, "web")
    if not (accession and title and object_url and image_url):
        return None
    if display_value(record.get("share_license_status")).upper() != "CC0":
        return None

    object_id = f"cma:{accession}"
    date = display_value(record.get("creation_date"))
    medium = display_value(record.get("technique"))
    culture = display_value(record.get("culture"))
    evidence = _evidence(record, object_id, object_url)
    if len(evidence) < 2:
        return None

    alt_text, alt_source = build_alt_text(
        title, date, medium, culture, (record.get("images") or {}).get("annotation")
    )
    earliest = record.get("creation_date_earliest")
    latest = record.get("creation_date_latest")
    if not isinstance(earliest, int) or not isinstance(latest, int):
        earliest, latest = parse_year_range(date)

    return SourceObject(
        id=object_id,
        institution=INSTITUTION,
        institution_id=INSTITUTION_ID,
        source_id=str(record.get("id") or accession),
        accession_number=accession,
        title=title,
        title_original=strip_html(record.get("title_in_original_language")) or None,
        date=date,
        date_earliest=earliest,
        date_latest=latest,
        creator=_creator(record),
        medium=medium,
        type=display_value(record.get("type")) or "Collection object",
        classification=display_value(record.get("collection")),
        department=display_value(record.get("department")),
        culture=culture,
        place=display_value(record.get("find_spot")),
        description=excerpt(strip_html(record.get("description")), 900),
        image_url=image_url,
        image_url_large=_image_url(record, "print"),
        object_url=object_url,
        rights=RIGHTS,
        rights_uri=RIGHTS_URI,
        credit_line=display_value(record.get("creditline")),
        alt_text=alt_text,
        alt_text_source=alt_source,
        evidence=evidence,
        source_api_url=api_url,
        raw_record_path=raw_path,
        source_record_sha256=raw_sha,
        tags=[display_value(tag) for tag in (record.get("artists_tags") or []) if display_value(tag)],
    )


def fetch(
    client: HttpClient,
    snapshot: Snapshot,
    target: int,
    log: Callable[[str], None],
) -> list[SourceObject]:
    objects: list[SourceObject] = []
    seen: set[str] = set()
    for department in DEPARTMENTS:
        skip = 0
        while len(objects) < target:
            url = _search_url(department, skip)
            content, payload, status = client.fetch_json(url)
            snapshot.store("search", f"{department}-{skip:04d}", url, content, status)
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list) or not data:
                break
            for record in data:
                if not isinstance(record, dict):
                    continue
                accession = display_value(record.get("accession_number"))
                if not accession or accession in seen:
                    continue
                raw_bytes = (
                    __import__("json")
                    .dumps(record, ensure_ascii=False)
                    .encode("utf-8")
                )
                raw_path = snapshot.store(
                    "objects", accession, f"{API_ROOT}/{accession}", raw_bytes, status
                )
                mapped = _map(
                    record,
                    raw_path.relative_to(snapshot.root.parent).as_posix(),
                    sha256_bytes(raw_bytes),
                    f"{API_ROOT}/{accession}",
                )
                if mapped:
                    seen.add(accession)
                    objects.append(mapped)
            skip += PAGE_SIZE
            log(f"  CMA {department}: {len(objects)} mapped (skip={skip})")
            if len(data) < PAGE_SIZE:
                break
    return objects
