"""Shared, data-only contracts for controlled catalogue filtering.

This module deliberately imports neither the collection repository nor the LLM
planner. Keeping the typed filter contract here lets both layers depend on the
same class without creating a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass


FILTER_KEYS = frozenset(
    {
        "dateStart",
        "dateEnd",
        "cultures",
        "institutions",
        "materials",
        "objectTypes",
        "imageRequired",
        "rightsAllowed",
        "evidenceDepth",
    }
)
EVIDENCE_DEPTH_FILTER_VALUES = frozenset({"full", "thin"})
MIN_FILTER_YEAR = -10000
MAX_FILTER_YEAR = 3000


@dataclass(frozen=True)
class FilterSpec:
    """Allowlisted constraints; populated facets are ANDed by the index."""

    date_start: int | None = None
    date_end: int | None = None
    cultures: tuple[str, ...] = ()
    institutions: tuple[str, ...] = ()
    materials: tuple[str, ...] = ()
    object_types: tuple[str, ...] = ()
    image_required: bool | None = None
    rights_allowed: tuple[str, ...] = ()
    evidence_depth: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not any(
            (
                self.date_start is not None,
                self.date_end is not None,
                self.cultures,
                self.institutions,
                self.materials,
                self.object_types,
                self.image_required is True,
                self.rights_allowed,
                self.evidence_depth,
            )
        )
