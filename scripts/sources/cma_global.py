"""Globally stratified Cleveland Museum of Art open-access adapter.

Unlike :mod:`sources.cma`, which intentionally limits the corpus to three
East Asian departments, this adapter samples across every CMA curatorial
department.  It keeps the source contract deliberately small: callers provide
the shared HTTP client and snapshot, and receive normalized ``SourceObject``
instances through the existing CMA mapper.

Raw responses are archived one aggregate API page at a time.  Individual
records are not written as separate files; their ``raw_record_path`` points to
the page containing the record and ``source_record_sha256`` fingerprints the
canonical JSON form of that record.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Callable
from typing import Any

from . import cma
from .base import HttpClient, Snapshot, SourceObject, display_value, sha256_bytes


# Appendix B of the CMA Open Access API documentation.  Keep the official
# order: it is also the deterministic tie-breaker when the target is smaller
# than the number of non-empty departments.
DEPARTMENTS: tuple[str, ...] = (
    "African Art",
    "American Painting and Sculpture",
    "Art of the Americas",
    "Chinese Art",
    "Contemporary Art",
    "Decorative Art and Design",
    "Drawings",
    "Egyptian and Ancient Near Eastern Art",
    "European Painting and Sculpture",
    "Greek and Roman Art",
    "Indian and Southeast Asian Art",
    "Islamic Art",
    "Japanese Art",
    "Korean Art",
    "Medieval Art",
    "Modern European Painting and Sculpture",
    "Oceania",
    "Performing Arts, Music, & Film",
    "Photography",
    "Prints",
    "Textiles",
)

MAX_PAGE_SIZE = 1000
MIN_DATA_PAGE_SIZE = 50


def _search_url(department: str, skip: int, limit: int) -> str:
    """Build one stable, rights-filtered aggregate-page request."""

    query = {
        "department": department,
        "cc0": "1",
        "has_image": "1",
        "limit": str(min(MAX_PAGE_SIZE, max(1, limit))),
        "skip": str(max(0, skip)),
        # The API's default ordering can change with relevance/index updates.
        # Accession ordering gives snapshot rebuilds a stable traversal order.
        "orderby": "accession_number_sortable",
    }
    return f"{cma.API_ROOT}?{urllib.parse.urlencode(query)}"


def _total(payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    info = payload.get("info")
    if not isinstance(info, dict):
        return 0
    value = info.get("total")
    return value if isinstance(value, int) and value > 0 else 0


def _records(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    return [record for record in data if isinstance(record, dict)]


def _allocate_quotas(availability: dict[str, int], target: int) -> dict[str, int]:
    """Water-fill the target across departments in official order.

    One object is assigned to every department with remaining capacity before
    any department receives its next object.  This makes large sources such as
    Prints unable to crowd out small sources such as Oceania, while remaining
    deterministic and respecting the live capacity reported by the API.
    """

    quotas = {department: 0 for department in DEPARTMENTS}
    remaining = min(max(0, target), sum(max(0, availability.get(d, 0)) for d in DEPARTMENTS))
    while remaining:
        progressed = False
        for department in DEPARTMENTS:
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


def _record_bytes(record: dict[str, Any]) -> bytes:
    """Canonical bytes for a record embedded inside an aggregate snapshot."""

    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _map_page(
    records: list[dict[str, Any]],
    page_path: str,
) -> list[SourceObject]:
    mapped: list[SourceObject] = []
    for record in records:
        accession = display_value(record.get("accession_number"))
        if not accession:
            continue
        canonical = _record_bytes(record)
        obj = cma._map(
            record,
            page_path,
            sha256_bytes(canonical),
            f"{cma.API_ROOT}/{accession}",
        )
        if obj is not None:
            mapped.append(obj)
    return mapped


def _page_limit(quota: int) -> int:
    """Fetch modest headroom for records rejected by the shared CMA mapper."""

    headroom = max(25, quota // 4)
    return min(MAX_PAGE_SIZE, max(MIN_DATA_PAGE_SIZE, quota + headroom))


def fetch(
    client: HttpClient,
    snapshot: Snapshot,
    target: int,
    log: Callable[[str], None],
) -> list[SourceObject]:
    """Fetch up to ``target`` globally stratified, normalized CMA objects."""

    if target <= 0:
        return []

    availability: dict[str, int] = {}
    for department in DEPARTMENTS:
        url = _search_url(department, skip=0, limit=1)
        content, payload, status = client.fetch_json(url)
        snapshot.store("search", f"{department}-probe", url, content, status)
        availability[department] = _total(payload)
        log(f"  CMA global probe {department}: {availability[department]} available")

    quotas = _allocate_quotas(availability, target)
    selected_by_department: dict[str, list[SourceObject]] = {}
    reserves_by_department: dict[str, list[SourceObject]] = {}

    for department in DEPARTMENTS:
        quota = quotas[department]
        if quota <= 0:
            selected_by_department[department] = []
            reserves_by_department[department] = []
            continue

        candidates: list[SourceObject] = []
        seen_ids: set[str] = set()
        skip = 0
        page_limit = _page_limit(quota)
        available = availability[department]

        while len(candidates) < quota and skip < available:
            url = _search_url(department, skip=skip, limit=page_limit)
            content, payload, status = client.fetch_json(url)
            stored = snapshot.store(
                "search",
                f"{department}-{skip:06d}-{page_limit}",
                url,
                content,
                status,
            )
            page_path = stored.relative_to(snapshot.root.parent).as_posix()
            records = _records(payload)
            if not records:
                break
            for obj in _map_page(records, page_path):
                if obj.id in seen_ids:
                    continue
                seen_ids.add(obj.id)
                candidates.append(obj)
            skip += len(records)

        selected_by_department[department] = candidates[:quota]
        reserves_by_department[department] = candidates[quota:]
        log(
            f"  CMA global {department}: quota={quota}, "
            f"selected={len(selected_by_department[department])}, "
            f"reserve={len(reserves_by_department[department])}"
        )

    # Preserve department round-robin order in the result, rather than
    # concatenating a whole department at a time.
    selected: list[SourceObject] = []
    positions = {department: 0 for department in DEPARTMENTS}
    while len(selected) < target:
        progressed = False
        for department in DEPARTMENTS:
            group = selected_by_department[department]
            position = positions[department]
            if position >= len(group):
                continue
            selected.append(group[position])
            positions[department] += 1
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break

    # A small department can report enough raw records yet fall short after
    # rights/evidence mapping.  Deterministically fill such deficits from the
    # page-level reserves already fetched for the other departments.
    reserve_positions = {department: 0 for department in DEPARTMENTS}
    while len(selected) < target:
        progressed = False
        for department in DEPARTMENTS:
            reserve = reserves_by_department[department]
            position = reserve_positions[department]
            if position >= len(reserve):
                continue
            selected.append(reserve[position])
            reserve_positions[department] += 1
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break

    log(f"  CMA global: selected {len(selected)} of target {target}")
    return selected


def fetch_department(
    client: HttpClient,
    snapshot: Snapshot,
    department: str,
    target: int,
    log: Callable[[str], None],
) -> list[SourceObject]:
    """Fetch a deterministic supplement from one live CMA department.

    This is used when CMA changes a department label after a frozen global
    snapshot was created.  The supplement remains independently snapshotted
    and can be merged without repeating image checks for the existing corpus.
    """

    if department not in DEPARTMENTS:
        raise ValueError(f"unknown CMA department: {department}")
    if target <= 0:
        return []

    probe_url = _search_url(department, skip=0, limit=1)
    content, payload, status = client.fetch_json(probe_url)
    snapshot.store("search", f"{department}-probe", probe_url, content, status)
    available = _total(payload)
    wanted = min(target, available)
    candidates: list[SourceObject] = []
    seen_ids: set[str] = set()
    skip = 0
    while len(candidates) < wanted and skip < available:
        page_limit = min(MAX_PAGE_SIZE, max(MIN_DATA_PAGE_SIZE, wanted - len(candidates) + 100))
        url = _search_url(department, skip=skip, limit=page_limit)
        page_content, page_payload, page_status = client.fetch_json(url)
        stored = snapshot.store(
            "search",
            f"{department}-{skip:06d}-{page_limit}",
            url,
            page_content,
            page_status,
        )
        page_path = stored.relative_to(snapshot.root.parent).as_posix()
        records = _records(page_payload)
        if not records:
            break
        for obj in _map_page(records, page_path):
            if obj.id in seen_ids:
                continue
            seen_ids.add(obj.id)
            candidates.append(obj)
        skip += len(records)
    selected = candidates[:wanted]
    log(
        f"  CMA supplement {department}: selected={len(selected)}, "
        f"target={target}, available={available}"
    )
    return selected
