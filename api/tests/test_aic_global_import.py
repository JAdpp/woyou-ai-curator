from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from typing import Any

from scripts import supplement_aic_global_collection as supplement
from scripts.sources import aic_global
from scripts.sources.base import Snapshot


def _record(
    artwork_id: int,
    department: str,
    origin: str,
    materials: list[str],
    *,
    description: str | None = None,
) -> dict[str, Any]:
    return {
        "id": artwork_id,
        "title": f"Object {artwork_id}",
        "alt_titles": None,
        "artist_display": "Institution maker record",
        "date_display": "c. 1700",
        "date_start": 1690,
        "date_end": 1710,
        "place_of_origin": origin,
        "medium_display": ", ".join(materials),
        "dimensions": "10 x 10 cm",
        "credit_line": "Museum collection",
        "department_title": department,
        "classification_title": "Decorative arts",
        "classification_titles": ["decorative arts"],
        "artwork_type_title": "Vessel",
        "is_public_domain": True,
        "image_id": f"image-{artwork_id}",
        "thumbnail": {
            "alt_text": "Institution-authored description of the collection object image."
        },
        "description": description,
        "short_description": None,
        "provenance_text": None,
        "inscriptions": None,
        "style_titles": [],
        "subject_titles": [],
        "technique_titles": [],
        "material_titles": materials,
        "term_titles": ["vessel", *materials],
    }


class _FakeClient:
    def __init__(self) -> None:
        self.urls: list[str] = []
        self.by_department = {
            "Arts of Asia": [
                _record(100, "Arts of Asia", "Japan", ["silk"]),
                _record(101, "Arts of Asia", "China", ["bronze"]),
                _record(102, "Arts of Asia", "India", ["terracotta"]),
            ],
            "Painting and Sculpture of Europe": [
                _record(
                    200,
                    "Painting and Sculpture of Europe",
                    "France",
                    ["oil paint", "canvas"],
                ),
                _record(
                    201,
                    "Painting and Sculpture of Europe",
                    "Italy",
                    ["marble"],
                ),
                _record(
                    202,
                    "Painting and Sculpture of Europe",
                    "England",
                    ["silver"],
                ),
            ],
        }

    def fetch_json(self, url: str) -> tuple[bytes, Any, int]:
        self.urls.append(url)
        parsed = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path.endswith("/departments"):
            payload: dict[str, Any] = {
                "data": [
                    {"id": "PC-7", "title": "Arts of Asia"},
                    {
                        "id": "PC-10",
                        "title": "Painting and Sculpture of Europe",
                    },
                ]
            }
        elif parsed.path.endswith("/artworks/58075"):
            payload = {
                "data": _record(
                    58075,
                    "Arts of Asia",
                    "Iran",
                    ["fritware", "ceramic", "glaze"],
                    description="Institution curatorial description.",
                )
            }
        else:
            department = query[
                "query[bool][must][2][term][department_title.keyword]"
            ][0]
            records = self.by_department[department]
            limit = int(query["limit"][0])
            payload = {
                "pagination": {
                    "total": len(records),
                    "limit": limit,
                    "current_page": int(query["page"][0]),
                },
                "data": records[:limit] if limit else [],
            }
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return content, payload, 200


def test_quota_and_page_selection_are_balanced_and_spread() -> None:
    quotas = aic_global._allocate_quotas(
        {"Huge": 1000, "Small": 2, "Medium": 20},
        9,
        ["Huge", "Small", "Medium"],
    )
    assert quotas == {"Huge": 4, "Small": 2, "Medium": 3}
    assert aic_global._spread_page_numbers(100, 4) == [1, 34, 67, 100]
    assert aic_global._global_page_location(10) == (0, 10)
    assert aic_global._global_page_location(11) == (1, 1)
    assert aic_global._global_page_location(17) == (1, 7)


def test_global_fetch_pins_landing_case_and_archives_aggregate_pages(
    tmp_path: Path,
) -> None:
    collection_dir = tmp_path / "global_open"
    snapshot = Snapshot(collection_dir / "raw", aic_global.SOURCE_ID)
    client = _FakeClient()

    objects = aic_global.fetch(client, snapshot, 5, lambda _message: None)
    snapshot.finalize()

    assert len(objects) == 5
    assert objects[0].id == "aic:58075"
    assert {obj.department for obj in objects} == {
        "Arts of Asia",
        "Painting and Sculpture of Europe",
    }
    pinned = objects[0]
    assert pinned.raw_record_path.endswith("/objects/pinned-58075.json")
    assert all(
        "/search/" in obj.raw_record_path for obj in objects if obj.id != "aic:58075"
    )
    assert pinned.to_json()["curatorialTextLicense"] == "CC BY 4.0"
    description = next(
        chunk for chunk in pinned.to_json()["evidence"] if chunk["id"].endswith(":description")
    )
    assert description["license"] == "CC BY 4.0"

    selection_path = snapshot.dir / "selection" / "selected.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["requiredSelectedObjectIds"] == ["aic:58075"]
    assert selection["requestPolicy"]["headers"]["Referer"] is None
    assert selection["requestPolicy"]["headers"]["Cookie"] is None
    assert selection["selectedCount"] == 5
    assert sum(selection["selectedByDepartment"].values()) == 5

    search_urls = [url for url in client.urls if "/artworks/search" in url]
    assert any("sort%5Bid%5D=asc" in url for url in search_urls)
    assert all(
        "is_public_domain" in urllib.parse.unquote(url)
        and "exists%5D%5Bfield%5D=image_id" in url
        for url in search_urls
    )

    rebuilt, snapshot_id = supplement._rebuild_from_snapshot(
        collection_dir,
        collection_dir / "raw",
        5,
        snapshot.id,
    )
    assert snapshot_id == snapshot.id
    assert [obj.id for obj in rebuilt] == [obj.id for obj in objects]


def test_atomic_artifact_commit_keeps_unrelated_raw_and_readme(tmp_path: Path) -> None:
    collection = tmp_path / "global_open"
    stage = tmp_path / "stage" / "global_open"
    collection.mkdir(parents=True)
    stage.mkdir(parents=True)
    (collection / "raw").mkdir()
    (collection / "README.md").write_text("keep", encoding="utf-8")
    for name in supplement.GENERATED_ARTIFACTS:
        (collection / name).write_text(f"old-{name}", encoding="utf-8")
        (stage / name).write_text(f"new-{name}", encoding="utf-8")

    supplement._atomic_commit_artifacts(stage, collection)

    assert (collection / "README.md").read_text(encoding="utf-8") == "keep"
    assert (collection / "raw").is_dir()
    for name in supplement.GENERATED_ARTIFACTS:
        assert (collection / name).read_text(encoding="utf-8") == f"new-{name}"


class _KeysetClient:
    """Exercise global page 17 without ever sending local page 11+."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def fetch_json(self, url: str) -> tuple[bytes, Any, int]:
        self.urls.append(url)
        parsed = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path.endswith("/departments"):
            payload: dict[str, Any] = {
                "data": [{"id": "PC-9", "title": "Applied Arts of Europe"}]
            }
        elif parsed.path.endswith("/artworks/58075"):
            payload = {
                "data": _record(
                    58075,
                    "Applied Arts of Europe",
                    "Iran",
                    ["ceramic"],
                )
            }
        elif int(query["limit"][0]) == 0:
            payload = {"pagination": {"total": 1700}, "data": []}
        elif query["fields"][0] == "id":
            assert query["page"] == ["10"]
            assert "query[bool][must][3][range][id][gt]" not in query
            payload = {"data": [{"id": 43513}]}
        else:
            assert query["page"] == ["7"]
            assert query["query[bool][must][3][range][id][gt]"] == ["43513"]
            payload = {
                "data": [
                    _record(
                        50000,
                        "Applied Arts of Europe",
                        "France",
                        ["silver"],
                    )
                ]
            }
        content = json.dumps(payload).encode("utf-8")
        return content, payload, 200


def test_keyset_cursor_replaces_global_page_17(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(aic_global, "_candidate_page_numbers", lambda _total, _quota: [17])
    collection_dir = tmp_path / "global_open"
    snapshot = Snapshot(collection_dir / "raw", aic_global.SOURCE_ID)
    client = _KeysetClient()

    objects = aic_global.fetch(client, snapshot, 2, lambda _message: None)

    assert [obj.id for obj in objects] == ["aic:58075", "aic:50000"]
    search_urls = [
        urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        for url in client.urls
        if "/artworks/search" in url
    ]
    requested_pages = [int(query["page"][0]) for query in search_urls]
    assert max(requested_pages) == 10
    assert 17 not in requested_pages
    selection = json.loads(
        (snapshot.dir / "selection" / "selected.json").read_text(encoding="utf-8")
    )
    traversal = selection["cursorTraversal"]["Applied Arts of Europe"]
    assert traversal == [
        {
            "window": 0,
            "lowerExclusiveId": None,
            "cursorId": 43513,
            "localPage": 10,
            "rawPath": traversal[0]["rawPath"],
        }
    ]
    assert selection["requestPolicy"]["apiResultWindow"] == 1000
