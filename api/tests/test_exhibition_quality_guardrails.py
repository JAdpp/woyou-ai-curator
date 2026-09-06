from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.models import Chapter, Exhibition
from app.validator import validate_exhibition


def _generated_exhibition(
    client: TestClient, agenda_payload: dict[str, object]
) -> Exhibition:
    response = client.post(
        "/api/exhibitions/generate-sync", json={"agenda": agenda_payload}
    )
    assert response.status_code == 200, response.text
    exhibition = Exhibition.model_validate(response.json())
    assert validate_exhibition(exhibition).passed is True
    return exhibition


def _install_two_chapters(exhibition: Exhibition) -> None:
    split = max(1, len(exhibition.items) // 2)
    groups = (exhibition.items[:split], exhibition.items[split:])
    exhibition.chapters = [
        Chapter(
            id=f"quality-chapter-{index}",
            order=index,
            title=f"章节{index + 1}",
            leadIn="从对象之间的关系展开。",
            itemIds=[item.id for item in group],
        )
        for index, group in enumerate(groups)
        if group
    ]


@pytest.mark.parametrize(
    "title",
    ("四件作品之间的观看线索", "4 件展品之间的观看线索"),
)
def test_validator_blocks_wrong_chinese_or_arabic_item_count_in_public_title(
    client: TestClient,
    agenda_payload: dict[str, object],
    title: str,
) -> None:
    exhibition = _generated_exhibition(client, agenda_payload)
    assert len(exhibition.items) == 5
    exhibition.title = title

    result = validate_exhibition(exhibition)

    assert result.passed is False
    assert "PUBLIC_COPY_ITEM_COUNT_MISMATCH" in {
        issue.code for issue in result.errors
    }
    assert next(
        check for check in result.checks if check.key == "public-copy-item-counts"
    ).passed is False


def test_validator_checks_explicit_chapter_total_without_flagging_ordinals(
    client: TestClient,
    agenda_payload: dict[str, object],
) -> None:
    exhibition = _generated_exhibition(client, agenda_payload)
    _install_two_chapters(exhibition)
    chapter = exhibition.chapters[0]
    wrong_count = next(
        value
        for value in range(1, 13)
        if value not in {len(exhibition.items), len(chapter.item_ids)}
    )
    chapter.lead_in = f"本章共{wrong_count}件作品，从材料进入观看。"

    result = validate_exhibition(exhibition)
    assert "PUBLIC_COPY_ITEM_COUNT_MISMATCH" in {
        issue.code for issue in result.errors
    }

    chapter.lead_in = "第12件作品将问题带向另一个方向。"
    repaired = validate_exhibition(exhibition)
    assert "PUBLIC_COPY_ITEM_COUNT_MISMATCH" not in {
        issue.code for issue in repaired.errors
    }


def test_validator_accepts_item_counts_that_match_exhibition_or_chapter(
    client: TestClient,
    agenda_payload: dict[str, object],
) -> None:
    exhibition = _generated_exhibition(client, agenda_payload)
    _install_two_chapters(exhibition)
    exhibition.title = f"{len(exhibition.items)}件作品的观看路径"
    exhibition.subtitle = f"从{len(exhibition.items)} 件展品重读同一问题"
    for chapter in exhibition.chapters:
        chapter.title = f"{len(chapter.item_ids)}件藏品的对话"
        chapter.lead_in = f"本章共{len(chapter.item_ids)}件作品。"

    result = validate_exhibition(exhibition)

    assert "PUBLIC_COPY_ITEM_COUNT_MISMATCH" not in {
        issue.code for issue in result.errors
    }


def test_same_catalogue_titles_are_a_warning_not_proof_of_duplicate_objects(
    client: TestClient,
    agenda_payload: dict[str, object],
) -> None:
    exhibition = _generated_exhibition(client, agenda_payload)
    variants = ("Untitled", "UNTITLED", " Untitled ", "Untitled.")
    for item, title in zip(exhibition.items[:4], variants, strict=True):
        item.object.title = title
        item.object.title_original = None

    result = validate_exhibition(exhibition)

    assert result.passed is True
    assert "OVERCONCENTRATED_NORMALIZED_TITLE" in {
        issue.code for issue in result.warnings
    }
    assert next(
        check for check in result.checks if check.key == "selection-variety"
    ).passed is True


def test_validator_blocks_four_objects_explicitly_from_one_series(
    client: TestClient,
    agenda_payload: dict[str, object],
) -> None:
    exhibition = _generated_exhibition(client, agenda_payload)
    for index, item in enumerate(exhibition.items[:4], start=1):
        item.object.title = f"Plate {index}, from the series River Views"
        item.object.title_original = None

    result = validate_exhibition(exhibition)

    assert result.passed is False
    assert "OVERCONCENTRATED_OBJECT_SERIES" in {
        issue.code for issue in result.errors
    }


def test_three_repeated_titles_remain_below_the_blocking_threshold(
    client: TestClient,
    agenda_payload: dict[str, object],
) -> None:
    exhibition = _generated_exhibition(client, agenda_payload)
    for item, title in zip(
        exhibition.items[:3], ("Untitled", "UNTITLED", "Untitled."), strict=True
    ):
        item.object.title = title
        item.object.title_original = None

    result = validate_exhibition(exhibition)

    assert "OVERCONCENTRATED_NORMALIZED_TITLE" not in {
        issue.code for issue in result.errors
    }
    assert "OVERCONCENTRATED_OBJECT_SERIES" not in {
        issue.code for issue in result.errors
    }
