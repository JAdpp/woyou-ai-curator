from __future__ import annotations

import copy
import json
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from .models import AnalyticsResponse, EventLog, Exhibition


class ExhibitionStore:
    """Small, lock-protected store with optional JSON persistence for the P0 demo."""

    def __init__(self, mode: str = "memory", path: Path | None = None) -> None:
        self.mode = mode
        self.path = path
        self._lock = threading.RLock()
        self._exhibitions: dict[str, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        if self.mode == "json":
            if self.path is None:
                raise ValueError("A path is required for JSON store mode")
            self._load()

    def save_exhibition(self, exhibition: Exhibition) -> Exhibition:
        with self._lock:
            self._exhibitions[exhibition.id] = exhibition.model_dump(mode="json")
            self._persist()
        return self.get_exhibition(exhibition.id)

    def get_exhibition(self, exhibition_id: str) -> Exhibition:
        with self._lock:
            raw = self._exhibitions.get(exhibition_id)
            if raw is None:
                raise KeyError(exhibition_id)
            return Exhibition.model_validate(copy.deepcopy(raw))

    def get_by_slug(self, slug: str) -> Exhibition:
        with self._lock:
            for raw in self._exhibitions.values():
                if raw.get("slug") == slug and raw.get("status") == "published":
                    return Exhibition.model_validate(copy.deepcopy(raw))
        raise KeyError(slug)

    def list_exhibitions(self) -> list[Exhibition]:
        with self._lock:
            exhibitions = [
                Exhibition.model_validate(copy.deepcopy(raw))
                for raw in self._exhibitions.values()
            ]
        return sorted(exhibitions, key=lambda item: item.updated_at, reverse=True)

    def save_event(self, event: EventLog) -> EventLog:
        with self._lock:
            self._events.append(event.model_dump(mode="json"))
            self._persist()
        return EventLog.model_validate(copy.deepcopy(self._events[-1]))

    def analytics(self, collection_objects: int, reviewed_objects: int) -> AnalyticsResponse:
        with self._lock:
            event_counts = Counter(str(event.get("event")) for event in self._events)
            status_counts = Counter(
                str(exhibition.get("status")) for exhibition in self._exhibitions.values()
            )
            total_events = len(self._events)
            total_exhibitions = len(self._exhibitions)
            published = status_counts.get("published", 0)
        return AnalyticsResponse(
            event_counts=dict(sorted(event_counts.items())),
            exhibition_status_counts=dict(sorted(status_counts.items())),
            total_events=total_events,
            total_exhibitions=total_exhibitions,
            published_exhibitions=published,
            collection_objects=collection_objects,
            reviewed_objects=reviewed_objects,
            exhibitions=total_exhibitions,
            published=published,
            events=dict(sorted(event_counts.items())),
        )

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            exhibitions = payload.get("exhibitions", {})
            events = payload.get("events", [])
            if not isinstance(exhibitions, dict) or not isinstance(events, list):
                raise ValueError("Unexpected JSON store structure")
            # Validate persisted state at startup instead of returning malformed API data later.
            self._exhibitions = {
                key: Exhibition.model_validate(value).model_dump(mode="json")
                for key, value in exhibitions.items()
            }
            self._events = [EventLog.model_validate(value).model_dump(mode="json") for value in events]
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"Cannot load JSON store at {self.path}: {exc}") from exc

    def _persist(self) -> None:
        if self.mode != "json":
            return
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {"exhibitions": self._exhibitions, "events": self._events}
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_path.replace(self.path)
