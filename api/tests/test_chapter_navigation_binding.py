from copy import deepcopy

import pytest

from app import curation
from .test_frame_label_independence import _Provider, _generate


def test_navigation_uses_final_translated_titles_and_actual_order_only(client):
    offline = _Provider()
    offline.configured = False
    exhibition = _generate(client, offline)
    by_id = {item.id: item for item in exhibition.items}
    for index, item in enumerate(exhibition.items):
        item.display_title = f"最终译名{index}"
    for chapter in exhibition.chapters:
        chapter.lead_in = "这里的两件欧洲作品材质都不一样"
        chapter.item_ids.reverse()
    before = deepcopy(exhibition.model_dump())
    curation.bind_chapter_navigation(exhibition)
    for chapter in exhibition.chapters:
        assert "欧洲" not in chapter.lead_in and "材质" not in chapter.lead_in
        titles = [by_id[key].display_title for key in chapter.item_ids]
        assert all(title in chapter.lead_in for title in titles)
        assert [chapter.lead_in.index(title) for title in titles] == sorted(chapter.lead_in.index(title) for title in titles)
        assert chapter.item_ids == next(row["item_ids"] for row in before["chapters"] if row["id"] == chapter.id)
    assert exhibition.items == [item.model_validate(row) for item, row in zip(exhibition.items, before["items"])]


def test_navigation_does_not_hide_unknown_stop_ids(client):
    offline = _Provider()
    offline.configured = False
    exhibition = _generate(client, offline)
    exhibition.chapters[0].item_ids.append("not-selected")
    with pytest.raises(ValueError, match="unknown"):
        curation.bind_chapter_navigation(exhibition)
