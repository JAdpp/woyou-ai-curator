"""Art Institute of Chicago adapter.

AIC contributes the best image alt text of the three institutions (roughly 96%
of Chinese public-domain records carry institution-authored ``thumbnail.alt_text``)
plus structured ``term_titles`` that feed coverage-domain assignment. Only about
a third carry curatorial prose, so many AIC records end up ``thin`` and are
restricted to context/contrast roles.

API: https://api.artic.edu/api/v1/artworks (no key; AIC-User-Agent requested)
Licence: filtered to ``is_public_domain == true``.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Any, Callable

from .base import (
    CC0_LICENSE,
    CC0_RIGHTS_URI,
    CC_BY_4_0_LICENSE,
    CC_BY_4_0_RIGHTS_URI,
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

INSTITUTION = "Art Institute of Chicago"
INSTITUTION_ID = "aic"
API_ROOT = "https://api.artic.edu/api/v1/artworks"
SEARCH_ROOT = f"{API_ROOT}/search"
IIIF_ROOT = "https://www.artic.edu/iiif/2"
RIGHTS = "Public Domain (AIC is_public_domain=true)"
RIGHTS_URI = "https://creativecommons.org/publicdomain/zero/1.0/"

# AIC files East Asian material under "Arts of Asia", not "Asian Art".
DEPARTMENT = "Arts of Asia"
ORIGINS = ("China", "Korea", "Japan")
PAGE_SIZE = 100

FIELDS = ",".join(
    (
        "id",
        "title",
        "alt_titles",
        "artist_display",
        "date_display",
        "date_start",
        "date_end",
        "place_of_origin",
        "medium_display",
        "dimensions",
        "credit_line",
        "department_title",
        "classification_title",
        "classification_titles",
        "artwork_type_title",
        "is_public_domain",
        "image_id",
        "thumbnail",
        "description",
        "short_description",
        "provenance_text",
        "inscriptions",
        "style_titles",
        "subject_titles",
        "technique_titles",
        "material_titles",
        "term_titles",
    )
)


def _search_url(origin: str, page: int) -> str:
    params = [
        ("limit", str(PAGE_SIZE)),
        ("page", str(page)),
        ("fields", FIELDS),
        ("query[bool][must][0][term][is_public_domain]", "true"),
        ("query[bool][must][1][term][department_title.keyword]", DEPARTMENT),
        ("query[bool][must][2][match][place_of_origin]", origin),
    ]
    return f"{SEARCH_ROOT}?{urllib.parse.urlencode(params)}"


def _image_url(image_id: str, width: int) -> str:
    return f"{IIIF_ROOT}/{image_id}/full/{width},/0/default.jpg"


def _description_field(
    record: dict[str, Any],
) -> tuple[str, str | None, str | None, str | None]:
    """Return prose plus the exact AIC field and its field-level licence.

    The artworks endpoint singles out only ``description`` as CC BY 4.0.
    ``short_description`` remains part of the endpoint's otherwise-CC0 data,
    so the fallback must not erase which source field supplied the prose.
    """

    description = strip_html(record.get("description"))
    if description:
        return (
            description,
            "description",
            CC_BY_4_0_LICENSE,
            CC_BY_4_0_RIGHTS_URI,
        )
    short_description = strip_html(record.get("short_description"))
    if short_description:
        return (
            short_description,
            "short_description",
            CC0_LICENSE,
            CC0_RIGHTS_URI,
        )
    return "", None, None, None


def _evidence(record: dict[str, Any], object_id: str, object_url: str) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    artwork_id = record.get("id")

    tombstone_parts = [
        display_value(record.get("title")),
        display_value(record.get("artist_display")),
        display_value(record.get("date_display")),
        display_value(record.get("medium_display")),
        display_value(record.get("dimensions")),
    ]
    tombstone = ". ".join(part for part in tombstone_parts if part)
    if tombstone:
        chunks.append(
            EvidenceChunk(
                id=f"{object_id}:tombstone",
                text=tombstone,
                source_title=f"AIC object record {artwork_id}",
                source_url=object_url,
                source_location="Tombstone (title, artist, date, medium, dimensions)",
                supports="题名、作者、年代、材质与尺寸等机构著录事实",
                kind="tombstone",
                source_kind="institution_metadata",
            )
        )

    description, description_field, description_license, description_rights_uri = (
        _description_field(record)
    )
    if description and description_field and description_license and description_rights_uri:
        is_full_description = description_field == "description"
        chunks.append(
            EvidenceChunk(
                id=(
                    f"{object_id}:description"
                    if is_full_description
                    else f"{object_id}:short-description"
                ),
                text=excerpt(description),
                source_title=(
                    f"AIC curatorial description {artwork_id}"
                    if is_full_description
                    else f"AIC short description {artwork_id}"
                ),
                source_url=object_url,
                source_location=(
                    "Curatorial description (description field)"
                    if is_full_description
                    else "Short description (short_description field)"
                ),
                supports="机构撰写的作品说明与背景解释",
                kind="curatorial_text",
                license=description_license,
                rights_uri=description_rights_uri,
                source_kind="institution_curatorial_text",
            )
        )

    inscriptions = strip_html(record.get("inscriptions"))
    if inscriptions:
        chunks.append(
            EvidenceChunk(
                id=f"{object_id}:inscription",
                text=excerpt(inscriptions, 420),
                source_title=f"AIC inscription record {artwork_id}",
                source_url=object_url,
                source_location="Inscriptions",
                supports="器物或作品上的题识、款识与铭文著录",
                kind="curatorial_text",
                # ``kind`` remains curatorial_text for evidence-depth
                # compatibility; the API field itself is CC0 metadata, not the
                # separately licensed description field.
                source_kind="institution_metadata",
            )
        )

    terms = [display_value(term) for term in (record.get("term_titles") or [])]
    context_parts = [
        display_value(record.get("place_of_origin")),
        display_value(record.get("classification_title")),
        "、".join(term for term in terms if term)[:200],
    ]
    context = "；".join(part for part in context_parts if part)
    if context:
        chunks.append(
            EvidenceChunk(
                id=f"{object_id}:classification",
                text=context,
                source_title=f"AIC classification {artwork_id}",
                source_url=object_url,
                source_location="Place of origin / classification / terms",
                supports="产地、分类与主题词著录",
                kind="context",
                source_kind="institution_metadata",
            )
        )

    provenance = strip_html(record.get("provenance_text"))
    credit = display_value(record.get("credit_line"))
    acquisition = excerpt(provenance, 420) if provenance else credit
    if acquisition:
        chunks.append(
            EvidenceChunk(
                id=f"{object_id}:acquisition",
                text=acquisition,
                source_title=f"AIC provenance / credit line {artwork_id}",
                source_url=object_url,
                source_location="Provenance and credit line",
                supports="流传经过、入藏方式与归属标注",
                kind="acquisition",
                source_kind="institution_provenance",
            )
        )

    return chunks


def _map(record: dict[str, Any], raw_path: str, raw_sha: str) -> SourceObject | None:
    artwork_id = record.get("id")
    title = strip_html(record.get("title"))
    image_id = display_value(record.get("image_id"))
    if not (artwork_id and title and image_id):
        return None
    if record.get("is_public_domain") is not True:
        return None

    object_id = f"aic:{artwork_id}"
    object_url = f"https://www.artic.edu/artworks/{artwork_id}"
    api_url = f"{API_ROOT}/{artwork_id}"
    date = display_value(record.get("date_display"))
    medium = display_value(record.get("medium_display"))
    culture = display_value(record.get("place_of_origin"))
    evidence = _evidence(record, object_id, object_url)
    if len(evidence) < 2:
        return None

    thumbnail = record.get("thumbnail")
    provided_alt = thumbnail.get("alt_text") if isinstance(thumbnail, dict) else None
    alt_text, alt_source = build_alt_text(title, date, medium, culture, provided_alt)

    date_start = record.get("date_start")
    date_end = record.get("date_end")
    if not isinstance(date_start, int) or not isinstance(date_end, int):
        date_start, date_end = parse_year_range(date)

    terms = [display_value(term) for term in (record.get("term_titles") or [])]
    description, _description_source, description_license, description_rights_uri = (
        _description_field(record)
    )
    return SourceObject(
        id=object_id,
        institution=INSTITUTION,
        institution_id=INSTITUTION_ID,
        source_id=str(artwork_id),
        accession_number=str(artwork_id),
        title=title,
        # AIC's ``alt_titles`` field has no language/kind guarantee; treating
        # its first value as an original-language title would manufacture
        # provenance the institution does not publish.
        title_original=None,
        date=date,
        date_earliest=date_start,
        date_latest=date_end,
        creator=display_value(record.get("artist_display")),
        medium=medium,
        type=display_value(record.get("artwork_type_title")) or "Collection object",
        classification=display_value(record.get("classification_title")),
        department=display_value(record.get("department_title")),
        culture=culture,
        place=culture,
        description=excerpt(description, 900),
        image_url=_image_url(image_id, 843),
        image_url_large=_image_url(image_id, 1686),
        object_url=object_url,
        rights=RIGHTS,
        rights_uri=RIGHTS_URI,
        credit_line=display_value(record.get("credit_line")),
        alt_text=alt_text,
        alt_text_source=alt_source,
        evidence=evidence,
        source_api_url=api_url,
        raw_record_path=raw_path,
        source_record_sha256=raw_sha,
        curatorial_text_license=description_license,
        curatorial_text_rights_uri=description_rights_uri,
        tags=[term for term in terms if term],
    )


def fetch(
    client: HttpClient,
    snapshot: Snapshot,
    target: int,
    log: Callable[[str], None],
) -> list[SourceObject]:
    objects: list[SourceObject] = []
    seen: set[str] = set()
    for origin in ORIGINS:
        page = 1
        while len(objects) < target:
            url = _search_url(origin, page)
            content, payload, status = client.fetch_json(url)
            snapshot.store("search", f"{origin}-{page:04d}", url, content, status)
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list) or not data:
                break
            for record in data:
                if not isinstance(record, dict):
                    continue
                artwork_id = record.get("id")
                if not artwork_id or str(artwork_id) in seen:
                    continue
                raw_bytes = json.dumps(record, ensure_ascii=False).encode("utf-8")
                raw_path = snapshot.store(
                    "objects", str(artwork_id), f"{API_ROOT}/{artwork_id}", raw_bytes, status
                )
                mapped = _map(
                    record,
                    raw_path.relative_to(snapshot.root.parent).as_posix(),
                    sha256_bytes(raw_bytes),
                )
                if mapped:
                    seen.add(str(artwork_id))
                    objects.append(mapped)
            log(f"  AIC {origin}: {len(objects)} mapped (page={page})")
            pagination = payload.get("pagination") if isinstance(payload, dict) else {}
            if page >= (pagination or {}).get("total_pages", page):
                break
            page += 1
    return objects
