"""The Metropolitan Museum of Art Open Access adapter.

The Met publishes no curatorial prose field, so every Met record is
``thin`` evidence depth: its chunks restate tombstone, cultural context and
acquisition metadata rather than quoting institution narrative. Met objects are
therefore admitted to the collection as context/contrast material and are
blocked from core-evidence curatorial roles downstream (01b §1.4).

Its compensating strength is ``tags`` carrying Getty AAT URIs, which give the
coverage-domain assignment a controlled vocabulary to work from.

API: https://collectionapi.metmuseum.org/public/collection/v1 (no key)
Licence: filtered to ``isPublicDomain == true``.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .base import (
    EvidenceChunk,
    HttpClient,
    Snapshot,
    SourceObject,
    build_alt_text,
    display_value,
    parse_year_range,
    sha256_bytes,
    strip_html,
)

INSTITUTION = "The Metropolitan Museum of Art"
INSTITUTION_ID = "met"
API_ROOT = "https://collectionapi.metmuseum.org/public/collection/v1"
RIGHTS = "CC0 1.0 (Met isPublicDomain=true)"
RIGHTS_URI = "https://creativecommons.org/publicdomain/zero/1.0/"

ASIAN_ART_DEPARTMENT_ID = 6

# The Met search API requires a query term, so the corpus is assembled from
# domain-shaped probes rather than one wildcard sweep. These deliberately span
# the coverage domains defined in scripts/domains.py.
SEARCH_TERMS = (
    # landscape / brush
    "landscape", "mountain", "river", "bamboo", "plum blossom", "album leaf",
    "hanging scroll", "handscroll", "fan painting", "literati",
    # writing
    "calligraphy", "inscription", "rubbing", "seal", "sutra",
    # ritual bronze
    "ritual vessel", "bronze", "mirror", "bell", "wine vessel",
    # ceramics
    "porcelain", "celadon", "stoneware", "blue and white", "glaze", "bowl",
    "vase", "jar", "ewer", "dish",
    # devotion
    "buddha", "bodhisattva", "guanyin", "mandala", "temple", "shrine",
    # funerary
    "tomb figure", "burial", "funerary", "mingqi",
    # court / daily life
    "jade", "lacquer", "textile", "robe", "furniture", "inkstone", "snuff bottle",
    "ornament", "screen", "box",
    # exchange
    "export", "silk road", "trade",
)


def _search_url(term: str) -> str:
    params = {
        "departmentId": str(ASIAN_ART_DEPARTMENT_ID),
        "hasImages": "true",
        "q": term,
    }
    return f"{API_ROOT}/search?{urllib.parse.urlencode(params)}"


def _artist(record: dict[str, Any]) -> str:
    name = display_value(record.get("artistDisplayName"))
    bio = display_value(record.get("artistDisplayBio"))
    if name and bio:
        return f"{name} ({bio})"
    return name


def _culture(record: dict[str, Any]) -> str:
    parts = [
        display_value(record.get("culture")),
        display_value(record.get("dynasty")),
        display_value(record.get("period")),
        display_value(record.get("reign")),
    ]
    return "，".join(dict.fromkeys(part for part in parts if part))


def _geography(record: dict[str, Any]) -> str:
    parts = [
        display_value(record.get("country")),
        display_value(record.get("region")),
        display_value(record.get("subregion")),
        display_value(record.get("city")),
        display_value(record.get("excavation")),
    ]
    return "，".join(dict.fromkeys(part for part in parts if part))


def _tags(record: dict[str, Any]) -> list[str]:
    tags = record.get("tags")
    if not isinstance(tags, list):
        return []
    return [display_value(tag.get("term")) for tag in tags if isinstance(tag, dict) and tag.get("term")]


def _evidence(record: dict[str, Any], object_id: str, object_url: str) -> list[EvidenceChunk]:
    """Build metadata-derived chunks.

    None of these is curatorial prose, which is why Met objects never reach
    ``full`` evidence depth. Each chunk still quotes Met field values verbatim
    and points back at the object page.
    """
    chunks: list[EvidenceChunk] = []
    accession = display_value(record.get("accessionNumber"))

    tombstone_parts = [
        display_value(record.get("title")),
        _artist(record),
        display_value(record.get("objectDate")),
        display_value(record.get("medium")),
        display_value(record.get("dimensions")),
    ]
    tombstone = ". ".join(part for part in tombstone_parts if part)
    if tombstone:
        chunks.append(
            EvidenceChunk(
                id=f"{object_id}:tombstone",
                text=tombstone,
                source_title=f"Met object record {accession}",
                source_url=object_url,
                source_location="Tombstone (title, artist, date, medium, dimensions)",
                supports="题名、作者、年代、材质与尺寸等机构著录事实",
                kind="tombstone",
            )
        )

    culture = _culture(record)
    geography = _geography(record)
    context_parts = [
        culture,
        geography,
        display_value(record.get("objectName")),
        display_value(record.get("classification")),
    ]
    context = "；".join(part for part in context_parts if part)
    if context:
        chunks.append(
            EvidenceChunk(
                id=f"{object_id}:classification",
                text=context,
                source_title=f"Met cultural context {accession}",
                source_url=object_url,
                source_location="Culture / dynasty / period / geography / classification fields",
                supports="文化归属、朝代分期、出土或产地与器物分类著录",
                kind="context",
            )
        )

    tags = _tags(record)
    if tags:
        chunks.append(
            EvidenceChunk(
                id=f"{object_id}:subjects",
                text="、".join(tags[:24]),
                source_title=f"Met subject tags {accession}",
                source_url=object_url,
                source_location="Subject tags (Getty AAT controlled vocabulary)",
                supports="题材与图像主题的受控词表标引",
                kind="context",
            )
        )

    credit = display_value(record.get("creditLine"))
    if credit:
        chunks.append(
            EvidenceChunk(
                id=f"{object_id}:acquisition",
                text=credit,
                source_title=f"Met credit line {accession}",
                source_url=object_url,
                source_location="Credit line",
                supports="入藏方式与归属标注",
                kind="acquisition",
            )
        )

    return chunks


def _map(record: dict[str, Any], raw_path: str, raw_sha: str) -> SourceObject | None:
    object_id_number = record.get("objectID")
    title = strip_html(record.get("title"))
    image_url = display_value(record.get("primaryImageSmall")) or display_value(
        record.get("primaryImage")
    )
    object_url = display_value(record.get("objectURL"))
    if not (object_id_number and title and image_url and object_url):
        return None
    if record.get("isPublicDomain") is not True:
        return None

    object_id = f"met:{object_id_number}"
    date = display_value(record.get("objectDate"))
    medium = display_value(record.get("medium"))
    culture = _culture(record)
    evidence = _evidence(record, object_id, object_url)
    if len(evidence) < 2:
        return None

    # The Met publishes no alt-text field, so this always falls back to a
    # metadata sentence; the flag keeps that visible downstream.
    alt_text, alt_source = build_alt_text(title, date, medium, culture, None)

    begin = record.get("objectBeginDate")
    end = record.get("objectEndDate")
    if not isinstance(begin, int) or not isinstance(end, int):
        begin, end = parse_year_range(date)

    return SourceObject(
        id=object_id,
        institution=INSTITUTION,
        institution_id=INSTITUTION_ID,
        source_id=str(object_id_number),
        accession_number=display_value(record.get("accessionNumber")) or str(object_id_number),
        title=title,
        title_original=None,
        date=date,
        date_earliest=begin,
        date_latest=end,
        creator=_artist(record),
        medium=medium,
        type=display_value(record.get("objectName")) or "Collection object",
        classification=display_value(record.get("classification")),
        department=display_value(record.get("department")),
        culture=culture,
        place=_geography(record),
        description="",
        image_url=image_url,
        image_url_large=display_value(record.get("primaryImage")) or None,
        object_url=object_url,
        rights=RIGHTS,
        rights_uri=RIGHTS_URI,
        credit_line=display_value(record.get("creditLine")),
        alt_text=alt_text,
        alt_text_source=alt_source,
        evidence=evidence,
        source_api_url=f"{API_ROOT}/objects/{object_id_number}",
        raw_record_path=raw_path,
        source_record_sha256=raw_sha,
        tags=_tags(record),
    )


def fetch(
    client: HttpClient,
    snapshot: Snapshot,
    target: int,
    log: Callable[[str], None],
) -> list[SourceObject]:
    # Phase 1: collect candidate ids from every probe, interleaved so one broad
    # term cannot crowd out the rest of the coverage domains.
    per_term: list[list[int]] = []
    for term in SEARCH_TERMS:
        url = _search_url(term)
        try:
            content, payload, status = client.fetch_json(url)
        except Exception as error:  # noqa: BLE001 - one dead probe must not stop the import
            log(f"  Met probe '{term}' failed: {error}")
            continue
        snapshot.store("search", f"term-{term}", url, content, status)
        ids = payload.get("objectIDs") if isinstance(payload, dict) else None
        per_term.append([i for i in ids if isinstance(i, int)] if isinstance(ids, list) else [])
        log(f"  Met probe '{term}': {len(per_term[-1])} ids")

    candidates: list[int] = []
    seen_ids: set[int] = set()
    for index in range(max((len(ids) for ids in per_term), default=0)):
        for ids in per_term:
            if index < len(ids) and ids[index] not in seen_ids:
                seen_ids.add(ids[index])
                candidates.append(ids[index])
        if len(candidates) >= target * 3:
            break

    # Phase 2: fetch full records concurrently until the target is met.
    objects: list[SourceObject] = []
    lock = threading.Lock()
    # Cap the work: candidates outnumber the target several times over because
    # many records turn out not to be public domain.
    batch = candidates[: int(target * 2.2)]

    def load(object_number: int) -> tuple[int, bytes, Any, int] | None:
        # One shared client, so its throttle applies globally across workers.
        try:
            content, payload, status = client.fetch_json(f"{API_ROOT}/objects/{object_number}")
        except Exception:  # noqa: BLE001 - skip records the API will not serve
            return None
        if not isinstance(payload, dict):
            return None
        return object_number, content, payload, status

    # Chunked rather than one big map: pool.map submits every task up front,
    # so a plain early-break would still pay for thousands of requests after
    # the target was already met.
    chunk_size = 120
    with ThreadPoolExecutor(max_workers=4) as pool:
        position = 0
        for offset in range(0, len(batch), chunk_size):
            if len(objects) >= target:
                break
            for result in pool.map(load, batch[offset : offset + chunk_size]):
                position += 1
                if result is None:
                    continue
                object_number, _content, payload, status = result
                with lock:
                    if len(objects) >= target:
                        continue
                    raw_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    raw_path = snapshot.store(
                        "objects",
                        str(object_number),
                        f"{API_ROOT}/objects/{object_number}",
                        raw_bytes,
                        status,
                    )
                    mapped = _map(
                        payload,
                        raw_path.relative_to(snapshot.root.parent).as_posix(),
                        sha256_bytes(raw_bytes),
                    )
                    if mapped:
                        objects.append(mapped)
            log(f"  Met: {len(objects)} mapped from {position} candidates")
    return objects
