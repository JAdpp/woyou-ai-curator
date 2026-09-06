"""Derived culture routing from explicit catalogue origin fields.

This layer never changes institution metadata, source files or frozen qrels.
Its version must be recorded separately from the raw collection taxonomy.
"""

from __future__ import annotations

import re
from typing import Any


CULTURAL_ROUTING_VERSION = "controlled-origin-v2"

_INDIA = re.compile(r"\bindia\b", re.IGNORECASE)
_NON_GEOGRAPHIC_INDIA = re.compile(
    r"\bindia\s+ink\b|\beast\s+india\s+compan(?:y|ies)\b", re.IGNORECASE
)
_SOUTHEAST_ASIAN_ORIGIN = re.compile(
    r"\b(?:indonesia|burma|burmese|myanmar|thailand|thai|siam|siamese|"
    r"cambodia|cambodian|khmer|vietnam|vietnamese|viet\s+nam|laos|laotian|"
    r"malaysia|malay|malayan|java|javanese|bali|balinese|sumatra|sumatran|"
    r"philippines|philippine|filipino|singapore|brunei|timor)\b|"
    r"\bsouth[\s-]*east(?:ern)?\s+asia(?:n)?\b|\bSE\s+Asia\b",
    re.IGNORECASE,
)


def effective_culture_pack_ids(raw: dict[str, Any], pack_ids: list[str]) -> list[str]:
    """Prefer explicit India origin over a department-derived SE Asia tag.

    Only ``culture`` and ``place`` are eligible origin evidence. Artist
    nationality, titles, media and department names cannot trigger this rule.
    A record naming both India and a Southeast Asian origin stays unchanged.
    Remaining packs retain their order. If South Asia is newly added, it takes
    the place of the first removed Southeast Asia tag (or is appended when no
    such tag existed). Inputs are not mutated.
    """

    origin_fields = [raw.get(field) for field in ("culture", "place")]
    origins = [value for value in origin_fields if isinstance(value, str)]
    has_india = any(
        _INDIA.search(_NON_GEOGRAPHIC_INDIA.sub("", origin)) for origin in origins
    )
    if not has_india or any(_SOUTHEAST_ASIAN_ORIGIN.search(origin) for origin in origins):
        return list(pack_ids)

    result: list[str] = []
    has_south_asia = "south_asia" in pack_ids
    inserted = False
    for pack_id in pack_ids:
        if pack_id == "southeast_asia":
            if not has_south_asia and not inserted:
                result.append("south_asia")
                inserted = True
            continue
        result.append(pack_id)
    if not has_south_asia and not inserted:
        result.append("south_asia")
    return result
