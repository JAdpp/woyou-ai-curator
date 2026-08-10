"""Globally stratified Art Institute of Chicago open-access adapter.

The legacy :mod:`sources.aic` adapter intentionally samples East Asian art.
This adapter keeps its rights/evidence mapper, but selects public-domain
artworks with images across AIC departments.  Selection is deterministic and
two-level:

* water-fill a target across the institution's current department catalogue;
* within every department, round-robin stable culture/region and material
  buckets derived from institution metadata.

Search pages are requested with an explicit ascending id sort and sampled at
evenly spaced offsets, so neither AIC's boosted search head nor one large
department can crowd out the rest.  Every API response is stored verbatim by
``Snapshot``; each normalized object points to its aggregate response page and
stores the SHA-256 of its canonical record.  A selection receipt records the
request policy, quotas, strata, raw paths, and record hashes.

Official API guidance:
https://api.artic.edu/docs/
"""

from __future__ import annotations

import json
import math
import re
import urllib.parse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from . import aic
from .base import (
    AIC_USER_AGENT,
    SCHEMA_VERSION,
    HttpClient,
    Snapshot,
    SourceObject,
    display_value,
    sha256_bytes,
)


SOURCE_ID = "aic-global"
SAMPLING_VERSION = "aic-global-strata-v1"
DEPARTMENTS_ROOT = "https://api.artic.edu/api/v1/departments"
MAX_PAGE_SIZE = 100
# The live AIC search service currently rejects page 11 at limit=100 with
# ``403 Invalid number of results`` even though the public docs describe a
# larger deep-pagination ceiling.  Traverse the whole result set in stable
# 1,000-record id keyset windows instead of depending on that global window.
API_RESULT_WINDOW = 1_000
WINDOW_PAGES = API_RESULT_WINDOW // MAX_PAGE_SIZE
CANDIDATE_MULTIPLIER = 3
MIN_CANDIDATE_HEADROOM = 50

# A documented landing-page comparison case.  It is not exempt from any
# mapper, rights, evidence, or image-reachability gate.
REQUIRED_ARTWORK_IDS: tuple[int, ...] = (58075,)


# Ordered patterns are intentionally broad routing buckets, not assertions of
# one exclusive cultural identity.  Institution metadata remains unchanged on
# the emitted object.
_CULTURE_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "east_asia",
        (
            "china",
            "chinese",
            "japan",
            "japanese",
            "korea",
            "korean",
            "ryukyu",
            "mongolia",
            "tibet",
        ),
    ),
    (
        "south_southeast_asia",
        (
            "india",
            "indian",
            "pakistan",
            "sri lanka",
            "nepal",
            "bhutan",
            "thailand",
            "cambodia",
            "khmer",
            "vietnam",
            "myanmar",
            "burma",
            "indonesia",
            "java",
            "bali",
            "malaysia",
            "philippines",
        ),
    ),
    (
        "west_asia_north_africa",
        (
            "iran",
            "persia",
            "iraq",
            "mesopotamia",
            "syria",
            "levant",
            "anatolia",
            "turkey",
            "arabia",
            "islamic",
            "egypt",
            "morocco",
            "algeria",
            "tunisia",
            "north africa",
        ),
    ),
    (
        "africa_subsaharan",
        (
            "africa",
            "african",
            "ghana",
            "nigeria",
            "ethiopia",
            "kenya",
            "congo",
            "mali",
            "benin",
            "yoruba",
            "akan",
            "zulu",
        ),
    ),
    (
        "latin_america_caribbean",
        (
            "mexico",
            "mexican",
            "guatemala",
            "peru",
            "peruvian",
            "bolivia",
            "chile",
            "brazil",
            "argentina",
            "colombia",
            "caribbean",
            "maya",
            "aztec",
            "inca",
            "mesoamerica",
        ),
    ),
    (
        "north_america",
        (
            "united states",
            "american",
            "canada",
            "canadian",
            "alaska",
            "inuit",
            "native north america",
        ),
    ),
    (
        "oceania",
        (
            "oceania",
            "australia",
            "new zealand",
            "papua new guinea",
            "polynesia",
            "melanesia",
            "micronesia",
        ),
    ),
    (
        "europe",
        (
            "europe",
            "england",
            "britain",
            "france",
            "germany",
            "italy",
            "netherlands",
            "flanders",
            "spain",
            "portugal",
            "greece",
            "roman",
            "russia",
            "austria",
            "switzerland",
            "sweden",
            "norway",
            "denmark",
            "belgium",
            "ireland",
        ),
    ),
)


_MATERIAL_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "ceramic_glass",
        (
            "ceramic",
            "porcelain",
            "earthenware",
            "stoneware",
            "terracotta",
            "clay",
            "faience",
            "glass",
            "enamel",
        ),
    ),
    (
        "metal",
        (
            "metal",
            "bronze",
            "copper",
            "gold",
            "silver",
            "iron",
            "steel",
            "brass",
            "alloy",
            "tin",
            "lead",
        ),
    ),
    (
        "fiber_textile",
        (
            "textile",
            "fabric",
            "silk",
            "cotton",
            "wool",
            "linen",
            "fiber",
            "embroidery",
            "tapestry",
            "cloth",
            "thread",
            "weave",
        ),
    ),
    (
        "paper_paint_photo",
        (
            "paper",
            "ink",
            "pigment",
            "paint",
            "canvas",
            "parchment",
            "vellum",
            "photograph",
            "gelatin silver",
            "albumen",
        ),
    ),
    (
        "wood_plant",
        ("wood", "bamboo", "lacquer", "bark", "palm", "reed", "straw"),
    ),
    (
        "stone_mineral",
        (
            "stone",
            "marble",
            "jade",
            "rock",
            "granite",
            "limestone",
            "crystal",
            "quartz",
            "lapis",
            "gem",
        ),
    ),
    (
        "animal_derived",
        ("ivory", "bone", "horn", "shell", "leather", "hide", "feather"),
    ),
    ("mixed", ("mixed media", "mixed-media", "various materials")),
)


@dataclass(frozen=True)
class Candidate:
    """A normalized object plus auditable sampling facets."""

    obj: SourceObject
    department: str
    culture_bucket: str
    material_bucket: str

    @property
    def stratum(self) -> tuple[str, str]:
        return self.culture_bucket, self.material_bucket


def _normal_text(values: Iterable[Any]) -> str:
    return " ".join(display_value(value).casefold() for value in values if value)


def _matches(text: str, pattern: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])", text) is not None


def _culture_bucket(record: Mapping[str, Any]) -> str:
    text = _normal_text(
        (
            record.get("place_of_origin"),
            record.get("department_title"),
            *(record.get("style_titles") or []),
            *(record.get("subject_titles") or []),
        )
    )
    for bucket, patterns in _CULTURE_BUCKETS:
        if any(_matches(text, pattern) for pattern in patterns):
            return bucket
    return "other_or_unknown"


def _material_bucket(record: Mapping[str, Any]) -> str:
    text = _normal_text(
        (
            *(record.get("material_titles") or []),
            record.get("medium_display"),
            record.get("classification_title"),
            record.get("artwork_type_title"),
        )
    )
    for bucket, patterns in _MATERIAL_BUCKETS:
        if any(_matches(text, pattern) for pattern in patterns):
            return bucket
    return "other_or_unknown"


def _department_catalog_url() -> str:
    params = urllib.parse.urlencode(
        (("limit", "100"), ("page", "1"), ("fields", "id,title"))
    )
    return f"{DEPARTMENTS_ROOT}?{params}"


def _eligibility_terms(department: str | None = None) -> list[tuple[str, str]]:
    terms: list[tuple[str, str]] = [
        ("query[bool][must][0][term][is_public_domain]", "true"),
        ("query[bool][must][1][exists][field]", "image_id"),
    ]
    if department:
        terms.append(
            ("query[bool][must][2][term][department_title.keyword]", department)
        )
    return terms


def _search_url(
    department: str,
    page: int,
    limit: int,
    *,
    after_id: int | None = None,
    fields: str | None = None,
) -> str:
    params: list[tuple[str, str]] = [
        ("limit", str(min(MAX_PAGE_SIZE, max(0, limit)))),
        ("page", str(max(1, page))),
        ("fields", fields or aic.FIELDS),
        ("sort[id]", "asc"),
        *_eligibility_terms(department),
    ]
    if after_id is not None:
        params.append(("query[bool][must][3][range][id][gt]", str(after_id)))
    return f"{aic.SEARCH_ROOT}?{urllib.parse.urlencode(params)}"


def _pinned_url(artwork_id: int) -> str:
    return f"{aic.API_ROOT}/{artwork_id}?{urllib.parse.urlencode({'fields': aic.FIELDS})}"


def _pagination_total(payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    pagination = payload.get("pagination")
    if not isinstance(pagination, dict):
        return 0
    total = pagination.get("total")
    return total if isinstance(total, int) and total > 0 else 0


def _records(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, dict):
        return [data]
    if not isinstance(data, list):
        return []
    return [record for record in data if isinstance(record, dict)]


def _department_titles(payload: Any) -> list[str]:
    titles = {
        display_value(record.get("title"))
        for record in _records(payload)
        if display_value(record.get("title"))
    }
    return sorted(titles, key=lambda value: (value.casefold(), value))


def _record_bytes(record: Mapping[str, Any]) -> bytes:
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _map_record(record: dict[str, Any], raw_path: str) -> Candidate | None:
    canonical = _record_bytes(record)
    mapped = aic._map(record, raw_path, sha256_bytes(canonical))
    if mapped is None:
        return None
    return Candidate(
        obj=mapped,
        department=display_value(record.get("department_title")) or "Unknown",
        culture_bucket=_culture_bucket(record),
        material_bucket=_material_bucket(record),
    )


def _map_page(records: Iterable[dict[str, Any]], raw_path: str) -> list[Candidate]:
    mapped: list[Candidate] = []
    for record in records:
        candidate = _map_record(record, raw_path)
        if candidate is not None:
            mapped.append(candidate)
    return mapped


def _allocate_quotas(
    availability: Mapping[str, int], target: int, departments: Iterable[str] | None = None
) -> dict[str, int]:
    """Water-fill a target across departments in a stable order."""

    ordered = list(departments or sorted(availability, key=lambda value: value.casefold()))
    quotas = {department: 0 for department in ordered}
    remaining = min(
        max(0, target), sum(max(0, availability.get(department, 0)) for department in ordered)
    )
    while remaining:
        progressed = False
        for department in ordered:
            if quotas[department] >= max(0, availability.get(department, 0)):
                continue
            quotas[department] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return quotas


def _spread_page_numbers(total_pages: int, wanted_pages: int) -> list[int]:
    """Choose stable, evenly spaced 1-based pages including both ends."""

    total_pages = max(0, total_pages)
    wanted_pages = min(max(0, wanted_pages), total_pages)
    if wanted_pages == 0:
        return []
    if wanted_pages == 1:
        return [1]
    pages = {
        1 + round(index * (total_pages - 1) / (wanted_pages - 1))
        for index in range(wanted_pages)
    }
    # Rounding can theoretically collide.  Fill gaps in ascending order while
    # retaining the spread set and deterministic result.
    if len(pages) < wanted_pages:
        for page in range(1, total_pages + 1):
            pages.add(page)
            if len(pages) == wanted_pages:
                break
    return sorted(pages)


def _stratified_order(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Round-robin culture/material strata; stable id is the tie-breaker."""

    groups: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.obj.id in seen:
            continue
        seen.add(candidate.obj.id)
        groups[candidate.stratum].append(candidate)
    for group in groups.values():
        group.sort(key=lambda candidate: int(candidate.obj.source_id))

    ordered: list[Candidate] = []
    positions = {key: 0 for key in groups}
    keys = sorted(groups)
    while len(ordered) < len(seen):
        progressed = False
        for key in keys:
            position = positions[key]
            group = groups[key]
            if position >= len(group):
                continue
            ordered.append(group[position])
            positions[key] += 1
            progressed = True
        if not progressed:
            break
    return ordered


def _candidate_page_numbers(total: int, quota: int) -> list[int]:
    accessible = max(0, total)
    total_pages = math.ceil(accessible / MAX_PAGE_SIZE)
    candidate_goal = min(
        accessible,
        max(quota * CANDIDATE_MULTIPLIER, quota + MIN_CANDIDATE_HEADROOM),
    )
    wanted_pages = math.ceil(candidate_goal / MAX_PAGE_SIZE)
    return _spread_page_numbers(total_pages, wanted_pages)


def _global_page_location(global_page: int) -> tuple[int, int]:
    """Return ``(zero-based keyset window, one-based local page)``."""

    if global_page < 1:
        raise ValueError("global page must be positive")
    zero_based = global_page - 1
    return zero_based // WINDOW_PAGES, zero_based % WINDOW_PAGES + 1


def _window_cursors(
    client: HttpClient,
    snapshot: Snapshot,
    department: str,
    through_window: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Resolve exclusive id cursors for windows 1..``through_window``.

    Cursor requests only ask for ``id`` and always use local page 10.  The
    returned cursor at index ``n`` is the lower-exclusive id for keyset window
    ``n + 1``.
    """

    cursors: list[int] = []
    traversal: list[dict[str, Any]] = []
    lower_exclusive: int | None = None
    for window in range(through_window):
        url = _search_url(
            department,
            page=WINDOW_PAGES,
            limit=MAX_PAGE_SIZE,
            after_id=lower_exclusive,
            fields="id",
        )
        content, payload, status = client.fetch_json(url)
        raw_path = _store_json_response(
            snapshot,
            "cursor",
            f"{department}-window-{window:03d}-end",
            url,
            content,
            status,
        )
        records = _records(payload)
        ids = [
            record.get("id")
            for record in records
            if isinstance(record.get("id"), int)
        ]
        if not ids:
            raise RuntimeError(
                f"AIC cursor window {window} for {department} returned no ids"
            )
        cursor = max(ids)
        if lower_exclusive is not None and cursor <= lower_exclusive:
            raise RuntimeError(
                f"AIC cursor did not advance for {department}: "
                f"{cursor} <= {lower_exclusive}"
            )
        cursors.append(cursor)
        traversal.append(
            {
                "window": window,
                "lowerExclusiveId": lower_exclusive,
                "cursorId": cursor,
                "localPage": WINDOW_PAGES,
                "rawPath": raw_path,
            }
        )
        lower_exclusive = cursor
    return cursors, traversal


def _round_robin_departments(
    groups: Mapping[str, list[Candidate]], departments: Iterable[str]
) -> list[Candidate]:
    ordered_departments = list(departments)
    positions = {department: 0 for department in ordered_departments}
    selected: list[Candidate] = []
    total = sum(len(groups.get(department, [])) for department in ordered_departments)
    while len(selected) < total:
        progressed = False
        for department in ordered_departments:
            group = groups.get(department, [])
            position = positions[department]
            if position >= len(group):
                continue
            selected.append(group[position])
            positions[department] += 1
            progressed = True
        if not progressed:
            break
    return selected


def _selection_bytes(
    *,
    target: int,
    selected: list[Candidate],
    departments: list[str],
    availability: Mapping[str, int],
    quotas: Mapping[str, int],
    cursor_traversal: Mapping[str, list[dict[str, Any]]],
) -> bytes:
    culture_counts = Counter(candidate.culture_bucket for candidate in selected)
    material_counts = Counter(candidate.material_bucket for candidate in selected)
    department_counts = Counter(candidate.department for candidate in selected)
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "samplingVersion": SAMPLING_VERSION,
        "target": target,
        "selectedCount": len(selected),
        "requiredArtworkIds": list(REQUIRED_ARTWORK_IDS),
        "requiredSelectedObjectIds": [
            f"aic:{artwork_id}"
            for artwork_id in REQUIRED_ARTWORK_IDS
            if any(candidate.obj.source_id == str(artwork_id) for candidate in selected)
        ],
        "eligibility": {
            "isPublicDomain": True,
            "imageId": "exists",
            "minimumEvidenceChunks": 2,
            "remoteImageValidation": "required before serving merge",
        },
        "requestPolicy": {
            "method": "GET",
            "sort": "id asc",
            "pageSize": MAX_PAGE_SIZE,
            "apiResultWindow": API_RESULT_WINDOW,
            "windowTraversal": "range[id][gt] keyset; cursor from local page 10",
            "parallelism": 1,
            "minimumIntervalSeconds": 1.0,
            "headers": {
                "AIC-User-Agent": AIC_USER_AGENT,
                "Referer": None,
                "Cookie": None,
            },
        },
        "departments": departments,
        "departmentAvailability": dict(availability),
        "departmentQuotas": dict(quotas),
        "cursorTraversal": dict(cursor_traversal),
        "selectedByDepartment": dict(sorted(department_counts.items())),
        "selectedByCultureBucket": dict(sorted(culture_counts.items())),
        "selectedByMaterialBucket": dict(sorted(material_counts.items())),
        "selected": [
            {
                "objectId": candidate.obj.id,
                "artworkId": int(candidate.obj.source_id),
                "department": candidate.department,
                "cultureBucket": candidate.culture_bucket,
                "materialBucket": candidate.material_bucket,
                "rawRecordPath": candidate.obj.raw_record_path,
                "sourceRecordSha256": candidate.obj.source_record_sha256,
            }
            for candidate in selected
        ],
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _store_json_response(
    snapshot: Snapshot,
    kind: str,
    name: str,
    url: str,
    content: bytes,
    status: int,
) -> str:
    stored = snapshot.store(kind, name, url, content, status)
    return stored.relative_to(snapshot.root.parent).as_posix()


def fetch(
    client: HttpClient,
    snapshot: Snapshot,
    target: int,
    log: Callable[[str], None],
) -> list[SourceObject]:
    """Fetch up to ``target`` globally stratified, normalized AIC objects."""

    if target <= 0:
        return []
    if target < len(REQUIRED_ARTWORK_IDS):
        raise ValueError(
            f"target must be at least {len(REQUIRED_ARTWORK_IDS)} to include required cases"
        )

    catalog_url = _department_catalog_url()
    content, payload, status = client.fetch_json(catalog_url)
    _store_json_response(
        snapshot, "catalog", "departments", catalog_url, content, status
    )
    departments = _department_titles(payload)
    if not departments:
        raise RuntimeError("AIC department catalogue returned no usable titles")

    pinned: list[Candidate] = []
    pinned_by_department: Counter[str] = Counter()
    for artwork_id in REQUIRED_ARTWORK_IDS:
        url = _pinned_url(artwork_id)
        content, payload, status = client.fetch_json(url)
        raw_path = _store_json_response(
            snapshot, "objects", f"pinned-{artwork_id}", url, content, status
        )
        records = _records(payload)
        candidate = _map_record(records[0], raw_path) if records else None
        if candidate is None or candidate.obj.source_id != str(artwork_id):
            raise RuntimeError(
                f"required AIC artwork {artwork_id} failed public-domain/image/evidence gates"
            )
        pinned.append(candidate)
        pinned_by_department[candidate.department] += 1

    availability: dict[str, int] = {}
    for department in departments:
        url = _search_url(department, page=1, limit=0)
        content, payload, status = client.fetch_json(url)
        _store_json_response(
            snapshot, "search", f"{department}-probe", url, content, status
        )
        availability[department] = max(
            0, _pagination_total(payload) - pinned_by_department[department]
        )
        log(
            f"  AIC global probe {department}: "
            f"{availability[department]} available after required cases"
        )

    remaining_target = target - len(pinned)
    quotas = _allocate_quotas(availability, remaining_target, departments)
    selected_by_department: dict[str, list[Candidate]] = {}
    reserves_by_department: dict[str, list[Candidate]] = {}
    cursor_traversal: dict[str, list[dict[str, Any]]] = {}
    pinned_ids = {candidate.obj.id for candidate in pinned}

    for department in departments:
        quota = quotas[department]
        if quota <= 0:
            selected_by_department[department] = []
            reserves_by_department[department] = []
            continue

        candidates: list[Candidate] = []
        seen_ids: set[str] = set()
        pages = _candidate_page_numbers(
            availability[department] + pinned_by_department[department], quota
        )
        page_locations = [_global_page_location(page) for page in pages]
        through_window = max((window for window, _page in page_locations), default=0)
        cursors, traversal = _window_cursors(
            client,
            snapshot,
            department,
            through_window,
        )
        cursor_traversal[department] = traversal
        for global_page, (window, local_page) in zip(pages, page_locations):
            after_id = cursors[window - 1] if window else None
            url = _search_url(
                department,
                page=local_page,
                limit=MAX_PAGE_SIZE,
                after_id=after_id,
            )
            content, payload, status = client.fetch_json(url)
            raw_path = _store_json_response(
                snapshot,
                "search",
                (
                    f"{department}-global-page-{global_page:04d}-"
                    f"window-{window:03d}-local-{local_page:02d}"
                ),
                url,
                content,
                status,
            )
            for candidate in _map_page(_records(payload), raw_path):
                if candidate.obj.id in pinned_ids or candidate.obj.id in seen_ids:
                    continue
                seen_ids.add(candidate.obj.id)
                candidates.append(candidate)

        ordered = _stratified_order(candidates)
        selected_by_department[department] = ordered[:quota]
        reserves_by_department[department] = ordered[quota:]
        log(
            f"  AIC global {department}: quota={quota}, "
            f"selected={len(selected_by_department[department])}, "
            f"reserve={len(reserves_by_department[department])}, pages={pages}"
        )

    selected = [*pinned]
    selected.extend(_round_robin_departments(selected_by_department, departments))
    if len(selected) < target:
        selected.extend(
            _round_robin_departments(reserves_by_department, departments)[
                : target - len(selected)
            ]
        )
    selected = selected[:target]

    snapshot.store(
        "selection",
        "selected",
        "urn:woyou:global-open:aic-global-selection",
        _selection_bytes(
            target=target,
            selected=selected,
            departments=departments,
            availability=availability,
            quotas=quotas,
            cursor_traversal=cursor_traversal,
        ),
        200,
    )
    log(f"  AIC global selected {len(selected)}/{target} objects")
    return [candidate.obj for candidate in selected]
