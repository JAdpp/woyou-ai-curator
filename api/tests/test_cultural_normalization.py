from __future__ import annotations

import pytest

from app.cultural_normalization import effective_culture_pack_ids


@pytest.mark.parametrize("origin", ["Western India, Gujarat", "East India", "Eastern India, Orissa", "Mughal India, Allahabad"])
def test_explicit_india_origin_corrects_department_route(origin: str) -> None:
    raw = {"culture": origin, "department": "Indian and Southeast Asian Art"}
    packs = ["europe", "southeast_asia", "africa"]
    assert effective_culture_pack_ids(raw, packs) == ["europe", "south_asia", "africa"]
    assert packs == ["europe", "southeast_asia", "africa"]


@pytest.mark.parametrize("raw", [
    {"culture": "India (?), Burma (?), Malay (?)"},
    {"culture": "India", "place": "Thailand"},
    {"culture": "India or Southeast Asia"},
    {"culture": "India or Khmer"},
])
def test_uncertain_multi_region_origin_is_preserved(raw: dict) -> None:
    assert effective_culture_pack_ids(raw, ["southeast_asia"]) == ["southeast_asia"]


@pytest.mark.parametrize("raw", [
    {"culture": "Native North America, American Indian", "department": "Indian and Southeast Asian Art"},
    {"culture": "England", "creator": "Indian artist, born in India", "maker": "India"},
    {"culture": "", "department": "Indian and Southeast Asian Art"},
    {"culture": "India ink"},
    {"culture": "British East India Company"},
    {"culture": None, "place": None},
])
def test_non_origin_words_do_not_create_a_country(raw: dict) -> None:
    assert effective_culture_pack_ids(raw, ["southeast_asia"]) == ["southeast_asia"]


def test_existing_south_asia_keeps_relative_order() -> None:
    assert effective_culture_pack_ids({"culture": "India"}, ["europe", "southeast_asia", "africa", "south_asia"]) == ["europe", "africa", "south_asia"]


def test_indian_production_for_thai_market_retains_both_regions() -> None:
    raw = {"culture": "India (Coromandel Coast), for the Thai market"}
    assert effective_culture_pack_ids(raw, ["south_asia", "southeast_asia"]) == ["south_asia", "southeast_asia"]


def test_explicit_place_can_supply_missing_route() -> None:
    assert effective_culture_pack_ids({"place": "India, Gujarat"}, []) == ["south_asia"]
