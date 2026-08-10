from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from .collections import CollectionRepository, LoadedCollection
from .models import ApiModel, Exhibition, utc_now
from .store import ExhibitionStore


class EvidenceAudit(ApiModel):
    total_chunks: int
    reviewed_chunks: int
    pending_review_chunks: int
    objects_with_evidence: int
    objects_without_evidence: int
    objects_fully_reviewed: int
    objects_pending_review: int


class AssetAudit(ApiModel):
    missing_image_count: int
    missing_alt_text_count: int


class QuestionCardAudit(ApiModel):
    id: str | None = None
    title: str | None = None
    question: str
    coverage_status: str
    review_status: str
    theme_id: str | None = None
    starter_object_count: int = 0
    coverage_limits: list[str] = Field(default_factory=list)


class RegressionAudit(ApiModel):
    question_count: int
    status_distribution: dict[str, int]
    review_status_distribution: dict[str, int]


class CollectionDataAudit(ApiModel):
    collection_id: str
    name: str
    institution: str
    version: str
    source_url: str | None = None
    collection_license: str
    object_count: int
    runtime_object_count: int
    excluded_runtime_object_count: int
    theme_distribution: dict[str, int]
    culture_pack_distribution: dict[str, int] = Field(default_factory=dict)
    institution_distribution: dict[str, int] = Field(default_factory=dict)
    unrouted_object_count: int = 0
    object_rights_distribution: dict[str, int]
    evidence: EvidenceAudit
    assets: AssetAudit
    question_cards: list[QuestionCardAudit]
    question_card_coverage_status_distribution: dict[str, int]
    regression: RegressionAudit


class DataAuditReport(ApiModel):
    schema_version: str = "data-audit-v1"
    read_only: bool = True
    human_review_required: bool = True
    generated_at: datetime = Field(default_factory=utc_now)
    collections: list[CollectionDataAudit]
    review_boundary: str = (
        "This endpoint reports frozen-data review states only. It does not perform or "
        "record human evidence review and never modifies collection JSON."
    )


class CollectionVersionExport(ApiModel):
    collection_id: str
    version: str
    object_count: int


class RuntimeVersionDistributions(ApiModel):
    provider: dict[str, int]
    model: dict[str, int]
    prompt: dict[str, int]
    collection: dict[str, int]
    validator: dict[str, int]


class DescriptiveStatistics(ApiModel):
    total_events: int
    event_counts: dict[str, int]
    total_exhibitions: int
    exhibition_status_counts: dict[str, int]
    published_exhibitions: int
    collection_objects: int
    reviewed_objects: int


class ExportPrivacyBoundary(ApiModel):
    aggregate_only: bool = True
    raw_session_ids_included: bool = False
    event_parameters_included: bool = False
    agenda_fields_included: bool = False
    personal_connection_included: bool = False
    excluded_topics_included: bool = False


class DescriptiveStatisticsExport(ApiModel):
    schema_version: str = "descriptive-statistics-export-v1"
    api_version: str
    exported_at: datetime = Field(default_factory=utc_now)
    collections: list[CollectionVersionExport]
    runtime_version_distributions: RuntimeVersionDistributions
    statistics: DescriptiveStatistics
    privacy: ExportPrivacyBoundary = Field(default_factory=ExportPrivacyBoundary)
    note: str = "Aggregate descriptive product statistics only; not evidence of learning outcomes."


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(part for item in value if (part := _text(item)))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _first(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = raw.get(key)
        if value not in (None, "", []):
            return value
    return None


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _collection_directory(
    repository: CollectionRepository, collection: LoadedCollection
) -> Path | None:
    for objects_path in sorted(repository.collections_dir.glob("*/objects.json")):
        manifest = _read_json(objects_path.parent / "manifest.json", {})
        manifest_id = _text(manifest.get("id")) if isinstance(manifest, dict) else ""
        if (manifest_id or objects_path.parent.name) == collection.id:
            return objects_path.parent
    return None


def _raw_objects(directory: Path | None, collection: LoadedCollection) -> list[dict[str, Any]]:
    payload = _read_json(directory / "objects.json", []) if directory else []
    if isinstance(payload, dict):
        payload = payload.get("objects", [])
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return [item.model_dump(mode="json", by_alias=True) for item in collection.objects]


def _human_reviewed(chunk: dict[str, Any]) -> bool:
    if chunk.get("reviewed") is not True:
        return False
    status = _text(_first(chunk, "reviewStatus", "review_status")).casefold()
    return not any(marker in status for marker in ("pending", "draft", "unreviewed"))


def _question_card_audit(
    directory: Path | None, collection: LoadedCollection
) -> list[QuestionCardAudit]:
    payload = _read_json(directory / "question_cards.json", {}) if directory else {}
    raw_cards: Any = payload.get("cards", []) if isinstance(payload, dict) else payload
    cards: list[QuestionCardAudit] = []
    if isinstance(raw_cards, list):
        for raw in raw_cards:
            if not isinstance(raw, dict):
                continue
            question = _text(_first(raw, "question", "title"))
            if not question:
                continue
            cards.append(
                QuestionCardAudit(
                    id=_text(raw.get("id")) or None,
                    title=_text(raw.get("title")) or None,
                    question=question,
                    coverage_status=_text(
                        _first(raw, "coverageStatus", "coverage_status")
                    )
                    or "not_prevalidated",
                    review_status=_text(_first(raw, "reviewStatus", "review_status"))
                    or "not_recorded",
                    theme_id=_text(
                        _first(
                            raw,
                            "evidenceDomainId",
                            "evidence_domain_id",
                            "themeId",
                            "theme_id",
                        )
                    )
                    or None,
                    starter_object_count=len(
                        _list(_first(raw, "starterObjectIds", "starter_object_ids"))
                    ),
                    coverage_limits=[
                        text
                        for item in _list(
                            _first(raw, "coverageLimits", "coverage_limits")
                        )
                        if (text := _text(item))
                    ],
                )
            )
    if cards:
        return cards
    return [
        QuestionCardAudit(
            question=question,
            coverage_status="not_prevalidated",
            review_status="not_recorded",
        )
        for question in collection.question_cards
    ]


def _regression_audit(directory: Path | None) -> RegressionAudit:
    payload = _read_json(directory / "regression_questions.json", {}) if directory else {}
    raw_questions: Any = payload.get("questions", []) if isinstance(payload, dict) else payload
    status_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    question_count = 0
    if isinstance(raw_questions, list):
        for raw in raw_questions:
            if not isinstance(raw, dict):
                continue
            question_count += 1
            status_counts[
                _text(_first(raw, "expectedStatus", "expected_status")) or "not_recorded"
            ] += 1
            review_counts[
                _text(_first(raw, "reviewStatus", "review_status")) or "not_recorded"
            ] += 1
    return RegressionAudit(
        question_count=question_count,
        status_distribution=dict(sorted(status_counts.items())),
        review_status_distribution=dict(sorted(review_counts.items())),
    )


def _audit_collection(
    repository: CollectionRepository, collection: LoadedCollection
) -> CollectionDataAudit:
    directory = _collection_directory(repository, collection)
    raw_objects = _raw_objects(directory, collection)
    theme_counts: Counter[str] = Counter()
    culture_pack_counts: Counter[str] = Counter()
    institution_counts: Counter[str] = Counter()
    rights_counts: Counter[str] = Counter()
    total_chunks = 0
    reviewed_chunks = 0
    pending_chunks = 0
    objects_with_evidence = 0
    objects_fully_reviewed = 0
    objects_pending_review = 0
    missing_images = 0
    missing_alt_text = 0

    for raw in raw_objects:
        themes = [
            text
            for item in _list(
                _first(raw, "evidenceDomainIds", "evidence_domain_ids", "themes", "topics")
            )
            if (text := _text(item))
        ]
        theme_counts.update(themes or ["unclassified"])
        culture_packs = [
            text
            for item in _list(_first(raw, "culturePackIds", "culture_pack_ids"))
            if (text := _text(item))
        ]
        culture_pack_counts.update(culture_packs or ["unassigned"])
        institution_counts[
            _text(_first(raw, "institutionId", "institution_id", "institution"))
            or "unknown"
        ] += 1
        rights_counts[
            _text(
                _first(
                    raw,
                    "rights",
                    "license",
                    "share_license_status",
                    "shareLicenseStatus",
                )
            )
            or "missing"
        ] += 1
        if not _text(
            _first(raw, "imageUrl", "image_url", "primary_image", "primaryImage")
        ):
            missing_images += 1
        if not _text(
            _first(raw, "altText", "alt_text", "image_annotation", "imageAnnotation")
        ):
            missing_alt_text += 1

        evidence = [
            item
            for item in _list(_first(raw, "evidence", "evidenceChunks", "evidence_chunks"))
            if isinstance(item, dict)
        ]
        if not evidence:
            continue
        objects_with_evidence += 1
        reviewed_for_object = sum(1 for chunk in evidence if _human_reviewed(chunk))
        pending_for_object = len(evidence) - reviewed_for_object
        total_chunks += len(evidence)
        reviewed_chunks += reviewed_for_object
        pending_chunks += pending_for_object
        if pending_for_object:
            objects_pending_review += 1
        else:
            objects_fully_reviewed += 1

    cards = _question_card_audit(directory, collection)
    coverage_counts = Counter(card.coverage_status for card in cards)
    return CollectionDataAudit(
        collection_id=collection.id,
        name=collection.name,
        institution=collection.institution,
        version=collection.version,
        source_url=collection.source_url,
        collection_license=collection.license,
        object_count=len(raw_objects),
        runtime_object_count=len(collection.objects),
        excluded_runtime_object_count=max(0, len(raw_objects) - len(collection.objects)),
        theme_distribution=dict(sorted(theme_counts.items())),
        culture_pack_distribution=dict(sorted(culture_pack_counts.items())),
        institution_distribution=dict(sorted(institution_counts.items())),
        unrouted_object_count=theme_counts.get("unclassified", 0),
        object_rights_distribution=dict(sorted(rights_counts.items())),
        evidence=EvidenceAudit(
            total_chunks=total_chunks,
            reviewed_chunks=reviewed_chunks,
            pending_review_chunks=pending_chunks,
            objects_with_evidence=objects_with_evidence,
            objects_without_evidence=len(raw_objects) - objects_with_evidence,
            objects_fully_reviewed=objects_fully_reviewed,
            objects_pending_review=objects_pending_review,
        ),
        assets=AssetAudit(
            missing_image_count=missing_images,
            missing_alt_text_count=missing_alt_text,
        ),
        question_cards=cards,
        question_card_coverage_status_distribution=dict(sorted(coverage_counts.items())),
        regression=_regression_audit(directory),
    )


def build_data_audit(repository: CollectionRepository) -> DataAuditReport:
    return DataAuditReport(
        collections=[
            _audit_collection(repository, collection) for collection in repository.list()
        ]
    )


def _distribution(exhibitions: list[Exhibition], attribute: str) -> dict[str, int]:
    counts = Counter(
        _text(getattr(exhibition.versions, attribute, None)) or "not_recorded"
        for exhibition in exhibitions
    )
    return dict(sorted(counts.items()))


def build_descriptive_export(
    repository: CollectionRepository,
    store: ExhibitionStore,
    *,
    api_version: str,
) -> DescriptiveStatisticsExport:
    collections = repository.list()
    objects = [obj for collection in collections for obj in collection.objects]
    reviewed_objects = sum(
        1 for obj in objects if obj.evidence and all(chunk.reviewed for chunk in obj.evidence)
    )
    analytics = store.analytics(
        collection_objects=len(objects), reviewed_objects=reviewed_objects
    )
    exhibitions = store.list_exhibitions()
    return DescriptiveStatisticsExport(
        api_version=api_version,
        collections=[
            CollectionVersionExport(
                collection_id=collection.id,
                version=collection.version,
                object_count=len(collection.objects),
            )
            for collection in collections
        ],
        runtime_version_distributions=RuntimeVersionDistributions(
            provider=_distribution(exhibitions, "provider"),
            model=_distribution(exhibitions, "model"),
            prompt=_distribution(exhibitions, "prompt"),
            collection=_distribution(exhibitions, "collection"),
            validator=_distribution(exhibitions, "validator"),
        ),
        statistics=DescriptiveStatistics(
            total_events=analytics.total_events,
            event_counts=analytics.event_counts,
            total_exhibitions=analytics.total_exhibitions,
            exhibition_status_counts=analytics.exhibition_status_counts,
            published_exhibitions=analytics.published_exhibitions,
            collection_objects=analytics.collection_objects,
            reviewed_objects=analytics.reviewed_objects,
        ),
    )
