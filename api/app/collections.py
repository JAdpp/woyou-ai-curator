from __future__ import annotations

import json
import logging
import hashlib
import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from .models import AgendaInput, EvidenceChunk, EvidenceDepth, MuseumObject
from .dense_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    DenseIndexManager,
    DenseStatus,
)


CC0_LICENSE = "CC0 1.0"
CC0_RIGHTS_URI = "https://creativecommons.org/publicdomain/zero/1.0/"
CC_BY_4_0_LICENSE = "CC BY 4.0"
CC_BY_4_0_RIGHTS_URI = "https://creativecommons.org/licenses/by/4.0/"
FIELD_LEVEL_OPEN_ACCESS_INSTITUTIONS = frozenset({"aic", "cma", "met"})
HYBRID_RETRIEVAL_METHOD = "hybrid_bm25_dense_rrf_evidence_mmr"
HYBRID_RETRIEVAL_VERSION = "hybrid-rag-v1"
BM25_RETRIEVAL_METHOD = "fielded_bm25_hard_anchor"
BM25_RETRIEVAL_VERSION = "bm25-v1"
logger = logging.getLogger("app.retrieval")


class CollectionDataError(RuntimeError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


@dataclass(frozen=True)
class LoadedCollection:
    id: str
    name: str
    institution: str
    version: str
    objects_sha256: str | None
    license: str
    source_url: str | None
    question_cards: list[str]
    question_card_starters: dict[str, list[str]]
    question_card_limits: dict[str, list[str]]
    question_policies: list["QuestionPolicy"]
    evidence_domain_choices: dict[str, tuple[str, str]]
    culture_pack_labels: dict[str, str]
    concept_aliases: dict[str, tuple[str, ...]]
    objects: list[MuseumObject]


@dataclass(frozen=True)
class SearchResult:
    obj: MuseumObject
    score: float
    # Explainability carried with the retrieved object.  ``matched_evidence_ids``
    # is also used to put the most query-relevant institution excerpts first
    # when the object is passed to the writing model.
    matched_anchor_terms: tuple[str, ...] = ()
    matched_evidence_ids: tuple[str, ...] = ()
    field_scores: tuple[tuple[str, float], ...] = ()
    # Audit trace used by CuratorialBrief.  It deliberately records retrieval
    # channels and raw cosine components without exposing any model secret.
    retrieval_sources: tuple[str, ...] = ("bm25",)
    dense_score: float | None = None
    evidence_score: float | None = None


@dataclass(frozen=True)
class SearchDocument:
    searchable: set[str]
    title: set[str]
    routing: set[str]
    metadata: set[str]
    content: set[str]
    term_frequencies: dict[str, Counter[str]]
    field_lengths: dict[str, int]
    # Tokens whose surface form is present but whose local catalogue context
    # resolves to another sense.  They remain scoreable metadata, but cannot
    # satisfy a subject hard gate or an exclusion by themselves.
    contextual_anchor_suppressions: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SearchIndex:
    document_frequency: Counter[str]
    average_field_lengths: dict[str, float]
    document_count: int


@dataclass(frozen=True)
class QueryDocument:
    # Only frequencies for this query's handful of terms are retained.  The
    # full per-object token structures are transient, keeping the 17k-object
    # corpus from doubling the API worker's memory footprint.
    term_frequencies: dict[str, Counter[str]]
    field_lengths: dict[str, int]
    matched_anchor_terms: tuple[str, ...]


@dataclass(frozen=True)
class QueryPlan:
    scoring_tokens: set[str]
    anchor_tokens: set[str]
    anchor_groups: tuple[frozenset[str], ...]
    exact_tokens: set[str]
    browse_all: bool
    cross_cultural: bool
    # Dense recall may add candidates only for a known multilingual concept.
    # Unknown Chinese and English subjects retain the same lexical all-term
    # gate to avoid "nearest available object" answers.
    dense_fallback_allowed: bool
    # Familiar concrete entities (cat/dog/bird...) retain an exact lexical
    # subject gate even when dense retrieval is active.
    strict_anchor_groups: tuple[frozenset[str], ...]


@dataclass(frozen=True)
class QuestionPolicy:
    source: str
    question: str
    status: str
    # A corpus routing label, never a user-facing exhibition theme.
    evidence_domain_id: str | None
    starter_object_ids: list[str]
    coverage_limits: list[str]
    rationale: str | None = None
    policy_id: str | None = None


ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "object_id", "accession_number"),
    "source_id": ("sourceId", "source_id"),
    "title": ("title", "name"),
    "title_original": ("titleOriginal", "title_original"),
    "accession_number": ("accessionNumber", "accession_number"),
    "date": ("date", "creation_date"),
    "creator": ("creator", "maker", "artist"),
    "material": ("material", "technique", "medium"),
    "place": ("place", "find_spot", "findSpot"),
    "culture": ("culture", "cultures"),
    "culture_display": ("cultureDisplay", "culture_display"),
    "description": ("description", "tombstone", "summary"),
    "image_url": ("imageUrl", "image_url", "primary_image", "primaryImage"),
    "object_url": ("objectUrl", "object_url", "url", "source_url"),
    "rights": ("rights", "license", "share_license_status", "shareLicenseStatus"),
    "rights_uri": ("rightsUri", "rights_uri"),
    "image_license": ("imageLicense", "image_license"),
    "image_rights_uri": ("imageRightsUri", "image_rights_uri"),
    "metadata_license": ("metadataLicense", "metadata_license"),
    "metadata_rights_uri": ("metadataRightsUri", "metadata_rights_uri"),
    "curatorial_text_license": (
        "curatorialTextLicense",
        "curatorial_text_license",
    ),
    "curatorial_text_rights_uri": (
        "curatorialTextRightsUri",
        "curatorial_text_rights_uri",
    ),
    "alt_text": ("altText", "alt_text", "image_annotation", "imageAnnotation"),
    "themes": ("themes", "topics"),
    "evidence_domain_ids": ("evidenceDomainIds", "evidence_domain_ids"),
    "culture_pack_ids": ("culturePackIds", "culture_pack_ids"),
    "relation_facets": ("relationFacets", "relation_facets"),
    "evidence": ("evidence", "evidenceChunks", "evidence_chunks"),
}


CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "山水": ("landscape", "mountain", "river", "woodcutter", "retreat"),
    "书法": ("calligraphy", "cursive", "script", "poem", "writing"),
    "诗": ("poem", "poetry", "verse", "wang wei"),
    "题跋": ("inscription", "colophon", "calligraphy", "poem"),
    "临古": ("after", "copy", "tradition", "master", "style"),
    "仿效": ("after", "copy", "tradition", "master", "style"),
    "传承": ("tradition", "influence", "after", "style"),
    "器物": ("container", "vessel", "ceramic", "bronze", "headrest"),
    "青铜": ("bronze", "ritual vessel", "gui", "fangyou"),
    "仪式": ("ritual", "ceremonial", "mandala", "offering", "wine container", "food container"),
    "礼器": ("ritual", "ceremonial", "container", "vessel", "inscription"),
    "铭文": ("inscription", "mark", "script", "cast"),
    "釉色": ("glaze", "glazed", "ceramic", "porcelain"),
    "工艺": ("technique", "material", "cast", "carved", "glaze", "ceramic"),
    "材料": ("material", "medium", "bronze", "ceramic", "ink", "silk", "paper"),
    "青花瓷": ("blue-and-white", "cobalt", "underglaze", "porcelain"),
    "青花": ("blue-and-white", "cobalt", "underglaze"),
    "跨文化": ("cross-cultural", "transmission", "imported", "overseas", "trade"),
    "佛教": ("buddhist", "bodhisattva", "guanyin", "mandala"),
    "交流": ("exchange", "influence", "transmission", "trade", "cross-cultural"),
    "权力": ("power", "royal", "imperial", "ruler", "authority", "regalia"),
    "王权": ("royal", "king", "queen", "emperor", "court", "regalia"),
    "身份": ("identity", "status", "gender", "portrait", "dress", "body"),
    "身体": ("body", "figure", "portrait", "anatomy", "gesture"),
    "观看": ("gaze", "looking", "viewer", "viewing", "visibility", "spectator", "be seen"),
    "死亡": ("death", "funerary", "mortuary", "tomb", "burial", "afterlife"),
    "来世": ("afterlife", "funerary", "mortuary", "tomb", "burial"),
    "动物": ("animal", "lion", "dragon", "bird", "horse", "serpent"),
    "颜色": ("color", "pigment", "dye", "blue", "red", "gold", "polychrome"),
    "蓝色": ("blue", "cobalt", "indigo", "lapis", "azure"),
    "迁徙": ("migration", "mobility", "diaspora", "travel", "route", "journey"),
    "海洋": ("ocean", "sea", "maritime", "ship", "coast", "voyage"),
    "殖民": ("colonial", "colony", "empire", "imperialism"),
    "战争": ("war", "battle", "weapon", "armor", "military", "conflict"),
    "宗教": ("religion", "sacred", "devotion", "deity", "temple", "ritual"),
    "记忆": ("memory", "memorial", "commemoration", "archive", "inscription"),
    "日常": ("daily life", "domestic", "household", "clothing", "furniture", "utensil"),
    "技术": ("technology", "technique", "making", "workshop", "process", "manufacture"),
}


# Small, domain-level synonym groups cover common visitor vocabulary that
# institution catalogues express inconsistently.  These are deliberately
# concepts (animal families and familiar subjects), not object-specific title
# fixes, and are merged with aliases frozen in each collection manifest.
VISITOR_CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "猫": ("cat", "cats", "feline", "felis"),
    "猫科": ("feline family", "felidae", "lion", "tiger", "leopard", "jaguar", "panther"),
    # Include common visitor wording as explicit aliases instead of weakening
    # the single-CJK-character boundary guard below.  The guard is what keeps
    # unrelated compounds such as “猫头鹰” and “马赛克” out of animal results;
    # without these reviewed forms, however, the equally ordinary “狗狗” and
    # “小狗” are misclassified as unknown subjects.
    "狗": ("狗狗", "小狗", "犬类", "dog", "dogs", "puppy", "puppies", "canine", "hound"),
    "犬": ("狗狗", "小狗", "犬类", "dog", "dogs", "puppy", "puppies", "canine", "hound"),
    "鸟": ("bird", "birds", "avian"),
    "马": ("horse", "horses", "equine"),
    "鱼": ("fish", "fishes", "aquatic"),
    "龙": ("dragon", "dragons"),
    "狮": ("lion", "lions"),
    "虎": ("tiger", "tigers"),
}


TOKEN_STOPWORDS = {
    "and",
    "are",
    "can",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "into",
    "its",
    "only",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "was",
    "were",
    "what",
    "why",
    "with",
}


QUERY_FRAME_WORDS = {
    "artifact",
    "artifacts",
    "collection",
    "collections",
    "cultural",
    "culture",
    "cultures",
    "different",
    "exhibit",
    "exhibits",
    "find",
    "global",
    "museum",
    "museums",
    "object",
    "objects",
    "region",
    "regions",
    "show",
    "various",
}


QUERY_FRAME_PHRASES = (
    "有没有",
    "有无",
    "有哪些",
    "有什么",
    "请问",
    "帮我找",
    "给我看",
    "我想看",
    "相关的",
    "有关的",
    "关于",
    "藏品",
    "文物",
    "展品",
    "艺术品",
    "各文化地区",
    "不同文化地区",
    "各个文化",
    "不同文化",
    "多种文化",
    "各国文化",
    "各国地区",
    "不同国家",
    "多个国家",
    "世界各地",
    "各国",
    "全球",
)


CROSS_CULTURAL_PATTERNS = (
    r"各(?:个)?文化",
    r"不同文化",
    r"多种文化",
    r"跨文化",
    r"全球",
    r"各(?:个)?地区",
    r"各国(?:文化|地区)?",
    r"不同国家",
    r"多个国家",
    r"世界各地",
    r"across\s+(?:different\s+)?cultures?",
    r"different\s+cultures?",
    r"multiple\s+cultures?",
    r"around\s+the\s+world",
    r"global(?:ly)?",
)


# Chinese has no whitespace word boundary, but treating a one-character
# concept as an arbitrary substring makes 马赛克/马克思主义 look like 马 and
# 猫头鹰 look like 猫.  These grammatical/punctuation neighbours cover the
# structured visitor questions used by the interview while keeping ambiguous
# compounds conservative.  Multi-character concepts continue to use phrase
# matching.
_CJK_SINGLE_LEFT_BOUNDARIES = frozenset(
    "的地得与和或及在把被将为以于从向对看找问说讲有无是要想爱赏论谈"
)
_CJK_SINGLE_RIGHT_BOUNDARIES = frozenset(
    "的地得与和或及在是有无为之里中上下面呢吗吧啊如何怎样哪些什么"
)


def _concept_present(question: str, concept: str) -> bool:
    if not re.fullmatch(r"[\u3400-\u9fff]", concept):
        return _phrase_present(question, concept)
    for match in re.finditer(re.escape(concept), question):
        start, stop = match.span()
        left_ok = (
            start == 0
            or question[start - 1] in _CJK_SINGLE_LEFT_BOUNDARIES
            or not re.match(r"[\u3400-\u9fff]", question[start - 1])
        )
        right_ok = (
            stop == len(question)
            or question[stop] in _CJK_SINGLE_RIGHT_BOUNDARIES
            or not re.match(r"[\u3400-\u9fff]", question[stop])
        )
        if left_ok and right_ok:
            return True
    return False


BROWSE_QUERY_PATTERNS = (
    r"这些藏品.*(?:关系|联系|共同)",
    r"随便(?:看看|逛逛)",
    r"推荐(?:一些|几个|藏品|展品)",
    r"recommend\s+(?:something|anything|objects?)",
    r"surprise\s+me",
)


def _first(raw: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in raw and raw[name] not in (None, "", []):
            return raw[name]
    return default


def _string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        values = [_string(item) for item in value]
        return "; ".join(item for item in values if item) or None
    if isinstance(value, dict):
        for key in ("description", "name", "title", "value"):
            if value.get(key):
                return str(value[key]).strip()
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip() or None


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        value = [value]
    result: list[str] = []
    for item in value:
        item_text = _string(item)
        if item_text:
            result.append(item_text)
    return result


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value)
    return None


def _has_open_access_signal(
    institution_id: str, legacy_rights: str, legacy_rights_uri: str | None
) -> bool:
    if institution_id.casefold() not in FIELD_LEVEL_OPEN_ACCESS_INSTITUTIONS:
        return False
    signal = f"{legacy_rights} {legacy_rights_uri or ''}".casefold()
    return "cc0" in signal or "public domain" in signal or "publicdomain/zero" in signal


def _aic_description_field(evidence_id: str, source_location: str) -> str | None:
    evidence_key = evidence_id.casefold()
    location = source_location.casefold()
    if evidence_key.endswith(":short-description") or "short_description field" in location:
        return "short_description"
    if evidence_key.endswith(":description") or "curatorial description" in location:
        return "description"
    return None


def _aic_object_description_field(raw: dict[str, Any]) -> str | None:
    raw_evidence = _first(raw, ALIASES["evidence"], [])
    if not isinstance(raw_evidence, list):
        return None
    fallback: str | None = None
    for item in raw_evidence:
        if not isinstance(item, dict):
            continue
        evidence_id = _string(_first(item, ("id", "evidenceId", "evidence_id")))
        source_location = _string(
            _first(item, ("sourceLocation", "source_location"))
        )
        source_field = _aic_description_field(evidence_id, source_location)
        if source_field == "description":
            return source_field
        if source_field == "short_description":
            fallback = source_field
    return fallback


def _field_level_rights(
    raw: dict[str, Any], institution_id: str, legacy_rights: str
) -> dict[str, str | None]:
    """Read structured rights, with a conservative v2/v3 compatibility path."""

    legacy_rights_uri = _string(_first(raw, ALIASES["rights_uri"]))
    result = {
        "rights_uri": legacy_rights_uri,
        "image_license": _string(_first(raw, ALIASES["image_license"])),
        "image_rights_uri": _string(_first(raw, ALIASES["image_rights_uri"])),
        "metadata_license": _string(_first(raw, ALIASES["metadata_license"])),
        "metadata_rights_uri": _string(
            _first(raw, ALIASES["metadata_rights_uri"])
        ),
        "curatorial_text_license": _string(
            _first(raw, ALIASES["curatorial_text_license"])
        ),
        "curatorial_text_rights_uri": _string(
            _first(raw, ALIASES["curatorial_text_rights_uri"])
        ),
    }
    institution = institution_id.casefold()
    if not _has_open_access_signal(institution, legacy_rights, legacy_rights_uri):
        return result

    result["rights_uri"] = result["rights_uri"] or CC0_RIGHTS_URI
    result["image_license"] = result["image_license"] or CC0_LICENSE
    result["image_rights_uri"] = result["image_rights_uri"] or CC0_RIGHTS_URI
    result["metadata_license"] = result["metadata_license"] or CC0_LICENSE
    result["metadata_rights_uri"] = result["metadata_rights_uri"] or CC0_RIGHTS_URI
    has_curatorial_text = bool(_string(_first(raw, ALIASES["description"])))
    if has_curatorial_text and institution == "aic":
        description_field = _aic_object_description_field(raw)
        if description_field == "short_description":
            result["curatorial_text_license"] = (
                result["curatorial_text_license"] or CC0_LICENSE
            )
            result["curatorial_text_rights_uri"] = (
                result["curatorial_text_rights_uri"] or CC0_RIGHTS_URI
            )
        else:
            # Legacy ``:description`` records cannot reveal whether an older
            # mapper used short_description as a fallback, so retain the
            # conservative CC BY classification until rebuilt from raw data.
            result["curatorial_text_license"] = (
                result["curatorial_text_license"] or CC_BY_4_0_LICENSE
            )
            result["curatorial_text_rights_uri"] = (
                result["curatorial_text_rights_uri"] or CC_BY_4_0_RIGHTS_URI
            )
    elif has_curatorial_text and institution == "cma":
        result["curatorial_text_license"] = (
            result["curatorial_text_license"] or CC0_LICENSE
        )
        result["curatorial_text_rights_uri"] = (
            result["curatorial_text_rights_uri"] or CC0_RIGHTS_URI
        )
    return result


def _source_kind(kind: str | None) -> str:
    if kind == "curatorial_text":
        return "institution_curatorial_text"
    if kind == "acquisition":
        return "institution_provenance"
    return "institution_metadata"


def _is_aic_description(
    evidence_id: str, source_location: str, institution_id: str
) -> bool:
    return (
        institution_id.casefold() == "aic"
        and _aic_description_field(evidence_id, source_location) == "description"
    )


def _normalize_evidence(
    raw: Any,
    object_id: str,
    object_url: str,
    *,
    institution_id: str,
    metadata_license: str | None,
    metadata_rights_uri: str | None,
) -> list[EvidenceChunk]:
    if not isinstance(raw, list):
        return []
    evidence: list[EvidenceChunk] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        text = _string(_first(item, ("text", "chunk", "quote", "content")))
        if not text:
            continue
        evidence_id = _string(_first(item, ("id", "evidenceId", "evidence_id"))) or f"{object_id}-e{index + 1}"
        source_url = _string(_first(item, ("sourceUrl", "source_url", "url"))) or object_url
        source_title = _string(_first(item, ("sourceTitle", "source_title", "title"))) or f"Institution record: {object_id}"
        source_location = _string(
            _first(item, ("sourceLocation", "source_location"))
        ) or "Institution object record"
        verification = _string(_first(item, ("verification", "kind")))
        explicit_license = _string(_first(item, ("license", "licence")))
        explicit_rights_uri = _string(
            _first(item, ("rightsUri", "rights_uri", "licenseUrl", "license_url"))
        )
        if _is_aic_description(evidence_id, source_location, institution_id):
            chunk_license = explicit_license or CC_BY_4_0_LICENSE
            chunk_rights_uri = explicit_rights_uri or CC_BY_4_0_RIGHTS_URI
            source_kind = "institution_curatorial_text"
        else:
            chunk_license = explicit_license or metadata_license
            chunk_rights_uri = explicit_rights_uri or metadata_rights_uri
            explicit_source_kind = _string(
                _first(item, ("sourceKind", "source_kind"))
            )
            if institution_id.casefold() == "aic" and evidence_id.casefold().endswith(
                ":inscription"
            ):
                source_kind = "institution_metadata"
            else:
                source_kind = explicit_source_kind or _source_kind(verification)
        evidence.append(
            EvidenceChunk(
                id=evidence_id,
                text=text,
                source_url=source_url,
                source_title=source_title,
                source_location=source_location,
                supports=_string(_first(item, ("supports", "supportedFacts", "supported_facts"))) or "",
                reviewed=bool(item.get("reviewed", False)),
                review_status=_string(_first(item, ("reviewStatus", "review_status"))) or "pending",
                verification=verification,
                license=chunk_license,
                rights_uri=chunk_rights_uri,
                source_kind=source_kind,
            )
        )
    return evidence


def normalize_object(raw: dict[str, Any]) -> MuseumObject | None:
    object_id = _string(_first(raw, ALIASES["id"]))
    title = _string(_first(raw, ALIASES["title"]))
    image_url = _string(_first(raw, ALIASES["image_url"]))
    object_url = _string(_first(raw, ALIASES["object_url"]))
    rights = _string(_first(raw, ALIASES["rights"]))
    if not all((object_id, title, image_url, object_url, rights)):
        return None

    institution_id = _string(_first(raw, ("institutionId", "institution_id"))) or ""
    field_rights = _field_level_rights(raw, institution_id, rights)

    culture_values = _string_list(_first(raw, ALIASES["culture"], []))
    culture_display = _string(_first(raw, ALIASES["culture_display"]))
    creator = _string(_first(raw, ALIASES["creator"]))
    material = _string(_first(raw, ALIASES["material"]))
    date = _string(_first(raw, ALIASES["date"])) or ""
    culture = culture_display or ("; ".join(culture_values) if culture_values else "")
    evidence = _normalize_evidence(
        _first(raw, ALIASES["evidence"], []),
        object_id,
        object_url,
        institution_id=institution_id,
        metadata_license=field_rights["metadata_license"],
        metadata_rights_uri=field_rights["metadata_rights_uri"],
    )
    evidence_domain_ids = _string_list(
        _first(raw, ALIASES["evidence_domain_ids"], [])
    ) or _string_list(_first(raw, ALIASES["themes"], []))

    # Evidence depth decides whether an object may carry a core-evidence
    # curatorial role. Trust the importer's value when present; otherwise infer
    # it, so collections written before this field still behave sensibly.
    declared_depth = _string(_first(raw, ("evidenceDepth", "evidence_depth")))
    if declared_depth in {EvidenceDepth.FULL.value, EvidenceDepth.THIN.value}:
        evidence_depth = declared_depth
    else:
        has_prose = any(
            (chunk.verification == "curatorial_text" or "description" in chunk.id)
            for chunk in evidence
        )
        evidence_depth = (
            EvidenceDepth.FULL.value
            if has_prose and len(evidence) >= 3
            else EvidenceDepth.THIN.value
        )

    return MuseumObject(
        id=object_id,
        source_id=_string(_first(raw, ALIASES["source_id"])) or object_id,
        accession_number=_string(_first(raw, ALIASES["accession_number"])) or object_id,
        title=title,
        title_original=_string(_first(raw, ALIASES["title_original"])),
        date=date,
        date_earliest=_integer(_first(raw, ("dateEarliest", "date_earliest"))),
        date_latest=_integer(_first(raw, ("dateLatest", "date_latest"))),
        maker=creator or "",
        medium=material or "",
        type=_string(_first(raw, ("type", "objectType", "object_type"))) or "Collection object",
        creator=creator,
        material=material,
        place=_string(_first(raw, ALIASES["place"])),
        culture=culture,
        culture_display=culture or None,
        description=_string(_first(raw, ALIASES["description"])),
        image_url=image_url,
        image_url_large=_string(_first(raw, ("imageUrlLarge", "image_url_large"))),
        object_url=object_url,
        rights=rights,
        rights_uri=field_rights["rights_uri"],
        image_license=field_rights["image_license"],
        image_rights_uri=field_rights["image_rights_uri"],
        metadata_license=field_rights["metadata_license"],
        metadata_rights_uri=field_rights["metadata_rights_uri"],
        curatorial_text_license=field_rights["curatorial_text_license"],
        curatorial_text_rights_uri=field_rights["curatorial_text_rights_uri"],
        alt_text=_string(_first(raw, ALIASES["alt_text"])) or "",
        alt_text_source=_string(_first(raw, ("altTextSource", "alt_text_source")))
        or "metadata_fallback",
        themes=evidence_domain_ids,
        evidence_domain_ids=evidence_domain_ids,
        culture_pack_ids=_string_list(_first(raw, ALIASES["culture_pack_ids"], [])),
        relation_facets=_string_list(_first(raw, ALIASES["relation_facets"], [])),
        tags=_string_list(_first(raw, ("tags", "termTitles", "term_titles"), [])),
        evidence=evidence,
        institution=_string(_first(raw, ("institution",))) or "",
        institution_id=institution_id,
        department=_string(_first(raw, ("department", "departmentTitle", "department_title")))
        or "",
        classification=_string(_first(raw, ("classification",))) or "",
        credit_line=_string(_first(raw, ("creditLine", "credit_line"))) or "",
        evidence_depth=evidence_depth,
    )


def _token_sequence(text: str) -> list[str]:
    lowered = text.casefold()
    english = re.findall(r"[a-z0-9][a-z0-9'-]+", lowered)
    chinese_runs = re.findall(r"[\u3400-\u9fff]+", lowered)
    chinese: list[str] = []
    for run in chinese_runs:
        if len(run) == 1:
            chinese.append(run)
        else:
            chinese.extend(run[index : index + 2] for index in range(len(run) - 1))
            chinese.extend(
                run[index : index + 3] for index in range(max(0, len(run) - 2))
            )
    return [token for token in english if token not in TOKEN_STOPWORDS] + chinese


def _tokens(text: str) -> set[str]:
    return set(_token_sequence(text))


_EXPLICIT_DOG_TERMS = re.compile(
    r"\b(?:dogs?|pupp(?:y|ies)|hounds?|canids?)\b",
    re.IGNORECASE,
)
_CANINE_WORD = re.compile(r"\bcanines?\b", re.IGNORECASE)
_DENTAL_WORDS = frozenset({"tooth", "teeth", "dental", "fang", "fangs"})


def _contextual_anchor_suppressions(text: str) -> frozenset[str]:
    """Resolve a small set of high-cost catalogue homonyms conservatively.

    ``canine`` commonly denotes a dog, but museum descriptions also use it as
    the anatomical adjective in phrases such as ``leopard canine teeth``.  A
    dog query must not admit an otherwise unrelated object on that word alone.
    Suppress it only when every occurrence sits within two tokens of explicit
    dental vocabulary and the record has no independent dog signal.  Phrases
    such as ``canine companion`` therefore keep their normal dog meaning.
    """

    if not _CANINE_WORD.search(text):
        return frozenset()
    if _EXPLICIT_DOG_TERMS.search(text) or re.search(r"狗|犬(?![齿牙])", text):
        return frozenset()

    words = re.findall(r"[a-z][a-z'-]*", text.casefold())
    canine_positions = [
        index for index, word in enumerate(words) if word in {"canine", "canines"}
    ]
    if not canine_positions:
        return frozenset()
    dental_positions = {
        index for index, word in enumerate(words) if word in _DENTAL_WORDS
    }
    if dental_positions and all(
        any(abs(canine - dental) <= 2 for dental in dental_positions)
        for canine in canine_positions
    ):
        return frozenset({"canine", "canines"})
    return frozenset()


def _effective_searchable(document: SearchDocument) -> set[str]:
    return document.searchable - document.contextual_anchor_suppressions


def _normalized_question(text: str) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", text.casefold())


def _question_similarity(left: str, right: str) -> float:
    left_normalized = _normalized_question(left)
    right_normalized = _normalized_question(right)
    if not left_normalized or not right_normalized:
        return 0.0
    sequence_score = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    left_bigrams = {
        left_normalized[index : index + 2]
        for index in range(max(0, len(left_normalized) - 1))
    }
    right_bigrams = {
        right_normalized[index : index + 2]
        for index in range(max(0, len(right_normalized) - 1))
    }
    union = left_bigrams | right_bigrams
    jaccard = len(left_bigrams & right_bigrams) / len(union) if union else 0.0
    return 0.65 * sequence_score + 0.35 * jaccard


def _expanded_query_tokens(
    question: str,
    extra_aliases: dict[str, tuple[str, ...]] | None = None,
) -> set[str]:
    return _query_plan(question, extra_aliases).scoring_tokens


def _alias_catalog(
    extra_aliases: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, tuple[str, ...]]:
    aliases_by_concept = dict(CONCEPT_ALIASES)
    for concept, aliases in VISITOR_CONCEPT_ALIASES.items():
        existing = aliases_by_concept.get(concept, ())
        aliases_by_concept[concept] = tuple(dict.fromkeys((*existing, *aliases)))
    if extra_aliases:
        for concept, aliases in extra_aliases.items():
            existing = aliases_by_concept.get(concept, ())
            aliases_by_concept[concept] = tuple(
                dict.fromkeys((*existing, *aliases))
            )
    return aliases_by_concept


def _phrase_present(question: str, phrase: str) -> bool:
    if re.search(r"[\u3400-\u9fff]", phrase):
        return phrase in question
    return (
        re.search(
            rf"(?<![a-z0-9]){re.escape(phrase.casefold())}(?![a-z0-9])",
            question.casefold(),
        )
        is not None
    )


def _english_inflections(tokens: set[str]) -> set[str]:
    """Add conservative singular/plural variants for catalogue vocabulary."""

    expanded = set(tokens)
    for token in tuple(tokens):
        if not re.fullmatch(r"[a-z][a-z'-]{2,}", token):
            continue
        if token.endswith("ies") and len(token) > 4:
            expanded.add(token[:-3] + "y")
        elif token.endswith("s") and not token.endswith(("is", "ss", "ics")):
            expanded.add(token[:-1])
        elif not token.endswith(("s", "x", "z")):
            expanded.add(token + "s")
    return expanded


def _requests_cross_cultural(question: str) -> bool:
    return any(
        re.search(pattern, question, re.IGNORECASE)
        for pattern in CROSS_CULTURAL_PATTERNS
    )


def _is_browse_query(question: str) -> bool:
    return any(
        re.search(pattern, question, re.IGNORECASE)
        for pattern in BROWSE_QUERY_PATTERNS
    )


def _query_plan(
    question: str,
    extra_aliases: dict[str, tuple[str, ...]] | None = None,
) -> QueryPlan:
    """Separate a visitor's subject from request framing and collection scope.

    The former implementation treated every bigram in phrases such as
    ``有没有各文化地区的……藏品`` as equally meaningful.  That made broad request
    wording outrank the actual subject.  A query plan keeps an explicit subject
    anchor for the hard relevance gate, then uses aliases only as lexical
    expansion inside that boundary.
    """

    aliases_by_concept = _alias_catalog(extra_aliases)
    cross_cultural = _requests_cross_cultural(question)
    matched: list[tuple[str, tuple[str, ...]]] = []
    for concept, aliases in aliases_by_concept.items():
        if _concept_present(question, concept) or any(
            _phrase_present(question, alias) for alias in aliases
        ):
            matched.append((concept, aliases))

    # Cross-cultural wording controls diversification; it is not itself the
    # subject of every selected object.
    structural_concepts = {"跨文化", "交流"} if cross_cultural else set()
    thematic = [item for item in matched if item[0] not in structural_concepts]

    cleaned = question
    for phrase in QUERY_FRAME_PHRASES:
        cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(?:are there|do you have|can you find|please|across|from|about)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Split Chinese particles so an unknown subject still becomes its own
    # n-grams instead of being fused to “的藏品”.
    cleaned = re.sub(r"[的了吗呢啊吧]|地区", " ", cleaned)
    residual = _tokens(cleaned) - QUERY_FRAME_WORDS
    exact_tokens = _english_inflections(residual)

    anchor_tokens: set[str] = set()
    anchor_groups: list[frozenset[str]] = []
    strict_anchor_groups: list[frozenset[str]] = []
    scoring_tokens: set[str] = set(exact_tokens)
    for concept, aliases in thematic:
        concept_group = _tokens(concept)
        for alias in aliases:
            concept_group.update(_tokens(alias))
        concept_group = _english_inflections(concept_group)
        anchor_tokens.update(concept_group)
        anchor_groups.append(frozenset(concept_group))
        if concept in VISITOR_CONCEPT_ALIASES:
            strict_anchor_groups.append(frozenset(concept_group))
        scoring_tokens.update(concept_group)

    if not anchor_tokens:
        english_subjects = {
            token
            for token in re.findall(r"[a-z0-9][a-z0-9'-]+", cleaned.casefold())
            if token not in TOKEN_STOPWORDS and token not in QUERY_FRAME_WORDS
        }
        for token in sorted(english_subjects):
            group = frozenset(_english_inflections({token}))
            anchor_groups.append(group)
            anchor_tokens.update(group)
        chinese_subjects = {
            token for token in residual if re.search(r"[\u3400-\u9fff]", token)
        }
        if chinese_subjects:
            group = frozenset(chinese_subjects)
            anchor_groups.append(group)
            anchor_tokens.update(group)
    scoring_tokens.update(anchor_tokens)
    browse_all = _is_browse_query(question) and not thematic
    if browse_all:
        anchor_tokens.clear()
        anchor_groups.clear()
        scoring_tokens.clear()
        strict_anchor_groups.clear()
    return QueryPlan(
        scoring_tokens=scoring_tokens,
        anchor_tokens=anchor_tokens,
        anchor_groups=tuple(anchor_groups),
        exact_tokens=exact_tokens,
        browse_all=browse_all,
        cross_cultural=cross_cultural,
        # Dense-only recall is allowed for concepts with an explicit,
        # reviewable alias policy. Unknown Chinese and English subjects keep
        # the same lexical gate instead of accepting a nearest-neighbour guess.
        dense_fallback_allowed=bool(thematic),
        strict_anchor_groups=tuple(strict_anchor_groups),
    )


def _build_search_document(obj: MuseumObject) -> SearchDocument:
    field_text = {
        "title": " ".join(filter(None, [obj.title, obj.title_original])),
        "routing": " ".join(
            [*obj.routing_domain_ids, *obj.culture_pack_ids, obj.culture]
        ),
        "metadata": " ".join(
            filter(
                None,
                [
                    obj.date,
                    obj.creator,
                    obj.material,
                    obj.place,
                    obj.type,
                    obj.classification,
                    obj.department,
                    *obj.tags,
                ],
            )
        ),
        "content": " ".join(
            filter(None, [obj.description, *(item.text for item in obj.evidence)])
        ),
    }
    sequences = {name: _token_sequence(text) for name, text in field_text.items()}
    frequencies = {name: Counter(tokens) for name, tokens in sequences.items()}
    title = set(sequences["title"])
    routing = set(sequences["routing"])
    metadata = set(sequences["metadata"])
    content = set(sequences["content"])
    return SearchDocument(
        searchable=title | routing | metadata | content,
        title=title,
        routing=routing,
        metadata=metadata,
        content=content,
        term_frequencies=frequencies,
        field_lengths={name: len(tokens) for name, tokens in sequences.items()},
        contextual_anchor_suppressions=_contextual_anchor_suppressions(
            " ".join(field_text.values())
        ),
    )


def _object_score(
    obj: MuseumObject,
    query: QueryPlan,
    index: SearchIndex,
    document: QueryDocument,
) -> tuple[float, tuple[tuple[str, float], ...]]:
    if not query.scoring_tokens:
        return 0.0, ()
    weights = {"title": 5.0, "routing": 3.0, "metadata": 2.0, "content": 1.0}
    k1 = 1.35
    b = 0.72
    field_scores: list[tuple[str, float]] = []
    total = 0.0
    for field, weight in weights.items():
        frequencies = document.term_frequencies[field]
        length = document.field_lengths[field]
        average = max(index.average_field_lengths.get(field, 1.0), 1.0)
        score = 0.0
        for token in query.scoring_tokens:
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            frequency_docs = index.document_frequency.get(token, 0)
            inverse_document_frequency = math.log(
                1.0
                + (index.document_count - frequency_docs + 0.5)
                / (frequency_docs + 0.5)
            )
            saturation = frequency * (k1 + 1.0) / (
                frequency + k1 * (1.0 - b + b * length / average)
            )
            exact_boost = 1.2 if token in query.exact_tokens else 1.0
            score += inverse_document_frequency * saturation * exact_boost
        weighted = weight * score
        if weighted:
            field_scores.append((field, round(weighted, 6)))
            total += weighted
    return total, tuple(field_scores)


class CollectionRepository:
    _METADATA_FILENAMES = (
        "objects.json",
        "manifest.json",
        "question_cards.json",
        "regression_questions.json",
    )

    def __init__(
        self,
        collections_dir: Path,
        default_collection_id: str | None = None,
        *,
        rag_mode: str = "bm25",
        dense_index_dir: Path | None = None,
        embedding_model_cache_dir: Path | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        dense_top_k: int = 200,
        dense_min_score: float = 0.28,
        evidence_min_score: float = 0.30,
        rrf_k: int = 60,
        hybrid_max_results: int = 250,
        dense_manager: DenseIndexManager | None = None,
    ) -> None:
        if rag_mode not in {"bm25", "hybrid"}:
            raise ValueError("rag_mode must be either 'bm25' or 'hybrid'")
        if dense_top_k <= 0 or rrf_k <= 0 or hybrid_max_results <= 0:
            raise ValueError("dense_top_k, rrf_k and hybrid_max_results must be positive")
        if not -1.0 <= dense_min_score <= 1.0:
            raise ValueError("dense_min_score must be between -1 and 1")
        if not -1.0 <= evidence_min_score <= 1.0:
            raise ValueError("evidence_min_score must be between -1 and 1")
        self.collections_dir = collections_dir
        self.default_collection_id = default_collection_id
        self.rag_mode = rag_mode
        self.dense_top_k = dense_top_k
        self.dense_min_score = dense_min_score
        self.evidence_min_score = evidence_min_score
        self.rrf_k = rrf_k
        self.hybrid_max_results = hybrid_max_results
        runtime_root = collections_dir.parent.parent / "api" / "runtime" / "cache"
        self._dense_manager = dense_manager or DenseIndexManager(
            enabled=rag_mode == "hybrid",
            index_root=dense_index_dir or runtime_root / "rag",
            model_cache_dir=embedding_model_cache_dir or runtime_root / "fastembed",
            model_name=embedding_model,
        )
        self._cache_lock = RLock()
        self._cached_signature: tuple[tuple[str, int, int], ...] | None = None
        self._cached_collections: list[LoadedCollection] | None = None
        self._search_result_cache: OrderedDict[tuple[Any, ...], list[SearchResult]] = (
            OrderedDict()
        )

    def _file_signature(self) -> tuple[tuple[str, int, int], ...]:
        """Return a cheap signature for every file that affects collection loading."""
        if not self.collections_dir.exists():
            return ()

        entries: list[tuple[str, int, int]] = []
        for objects_path in sorted(self.collections_dir.glob("*/objects.json")):
            directory = objects_path.parent
            for filename in self._METADATA_FILENAMES:
                path = directory / filename
                try:
                    stat = path.stat()
                    entries.append(
                        (
                            path.relative_to(self.collections_dir).as_posix(),
                            stat.st_mtime_ns,
                            stat.st_size,
                        )
                    )
                except OSError:
                    entries.append(
                        (path.relative_to(self.collections_dir).as_posix(), -1, -1)
                    )
        return tuple(entries)

    def list(self) -> list[LoadedCollection]:
        signature = self._file_signature()
        with self._cache_lock:
            if (
                self._cached_collections is not None
                and signature == self._cached_signature
            ):
                return self._cached_collections

            self._search_result_cache.clear()
            collections = self._load_collections()
            confirmed_signature = self._file_signature()
            if confirmed_signature == signature:
                self._cached_signature = signature
                self._cached_collections = collections
            else:
                # A data import completed while the files were being parsed.
                # Serve this coherent read once, then force a refresh next call.
                self._cached_signature = None
                self._cached_collections = None
            return collections

    def _load_collections(self) -> list[LoadedCollection]:
        if not self.collections_dir.exists():
            return []

        collections: list[LoadedCollection] = []
        for objects_path in sorted(self.collections_dir.glob("*/objects.json")):
            directory = objects_path.parent
            manifest_path = directory / "manifest.json"
            question_cards_path = directory / "question_cards.json"
            regression_questions_path = directory / "regression_questions.json"
            try:
                objects_bytes = objects_path.read_bytes()
                actual_objects_sha256 = hashlib.sha256(objects_bytes).hexdigest()
                payload = json.loads(objects_bytes)
                manifest = (
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.exists()
                    else {}
                )
                question_payload = (
                    json.loads(question_cards_path.read_text(encoding="utf-8"))
                    if question_cards_path.exists()
                    else {}
                )
                regression_payload = (
                    json.loads(regression_questions_path.read_text(encoding="utf-8"))
                    if regression_questions_path.exists()
                    else {}
                )
            except (OSError, json.JSONDecodeError):
                continue

            declared_objects_sha256 = _string(
                _first(manifest, ("objectsSha256", "objects_sha256"))
            )
            if (
                declared_objects_sha256
                and declared_objects_sha256.casefold() != actual_objects_sha256
            ):
                logger.error(
                    "collection %s objects SHA mismatch: manifest=%s actual=%s; "
                    "dense cache will bind to the actual file",
                    manifest.get("id") or directory.name,
                    declared_objects_sha256,
                    actual_objects_sha256,
                )

            raw_objects = payload.get("objects", []) if isinstance(payload, dict) else payload
            if not isinstance(raw_objects, list):
                continue
            objects = [normalized for raw in raw_objects if isinstance(raw, dict) if (normalized := normalize_object(raw))]
            raw_cards = []
            if isinstance(question_payload, dict):
                raw_cards = _first(question_payload, ("cards", "questions", "items"), [])
            elif isinstance(question_payload, list):
                raw_cards = question_payload
            extracted_questions = []
            question_card_starters: dict[str, list[str]] = {}
            question_card_limits: dict[str, list[str]] = {}
            question_policies: list[QuestionPolicy] = []
            if isinstance(raw_cards, list):
                for card in raw_cards:
                    question = (
                        _string(_first(card, ("question", "title")))
                        if isinstance(card, dict)
                        else _string(card)
                    )
                    if question:
                        extracted_questions.append(question)
                        if isinstance(card, dict):
                            starters = _string_list(
                                _first(card, ("starterObjectIds", "starter_object_ids"), [])
                            )
                            limits = _string_list(
                                _first(card, ("coverageLimits", "coverage_limits"), [])
                            )
                            if starters:
                                question_card_starters[question] = starters
                            if limits:
                                question_card_limits[question] = limits
                            question_policies.append(
                                QuestionPolicy(
                                    source="question_card",
                                    question=question,
                                    status=_string(
                                        _first(card, ("coverageStatus", "coverage_status"))
                                    )
                                    or "supported",
                                    evidence_domain_id=_string(
                                        _first(
                                            card,
                                            (
                                                "evidenceDomainId",
                                                "evidence_domain_id",
                                                "themeId",
                                                "theme_id",
                                            ),
                                        )
                                    ),
                                    starter_object_ids=starters,
                                    coverage_limits=limits,
                                    rationale=_string(card.get("rationale")),
                                    policy_id=_string(card.get("id")),
                                )
                            )
            raw_regression_questions = (
                _first(regression_payload, ("questions", "items"), [])
                if isinstance(regression_payload, dict)
                else regression_payload
            )
            if isinstance(raw_regression_questions, list):
                for item in raw_regression_questions:
                    if not isinstance(item, dict):
                        continue
                    question = _string(item.get("question"))
                    expected_status = _string(
                        _first(item, ("expectedStatus", "expected_status"))
                    )
                    if not question or expected_status not in {
                        "supported",
                        "partially_supported",
                        "unsupported",
                    }:
                        continue
                    question_policies.append(
                        QuestionPolicy(
                            source="regression",
                            question=question,
                            status=expected_status,
                            evidence_domain_id=_string(
                                _first(
                                    item,
                                    (
                                        "evidenceDomainId",
                                        "evidence_domain_id",
                                        "themeId",
                                        "theme_id",
                                    ),
                                )
                            ),
                            starter_object_ids=[],
                            coverage_limits=[],
                            rationale=_string(item.get("rationale")),
                            policy_id=_string(item.get("id")),
                        )
                    )
            manifest_license = manifest.get("license")
            if isinstance(manifest_license, dict):
                license_parts = [
                    _string(manifest_license.get("images")),
                    _string(manifest_license.get("metadata")),
                    _string(
                        _first(
                            manifest_license,
                            ("curatorialText", "curatorial_text"),
                        )
                    ),
                    _string(_first(manifest_license, ("license_url", "licenseUrl"))),
                ]
                license_text = " | ".join(dict.fromkeys(part for part in license_parts if part))
            else:
                license_text = _string(manifest_license) or "unspecified"
            evidence_domain_choices: dict[str, tuple[str, str]] = {}
            raw_domains = manifest.get("evidenceDomains", [])
            if isinstance(raw_domains, list):
                for item in raw_domains:
                    if not isinstance(item, dict) or item.get("offerInInterview") is False:
                        continue
                    domain_id = _string(item.get("id"))
                    label = _string(item.get("label"))
                    if domain_id and label:
                        evidence_domain_choices[domain_id] = (
                            label,
                            _string(item.get("description")) or "馆藏中的一条可比较线索",
                        )
            culture_pack_labels: dict[str, str] = {}
            raw_packs = manifest.get("culturePacks", [])
            if isinstance(raw_packs, list):
                for item in raw_packs:
                    if not isinstance(item, dict):
                        continue
                    pack_id = _string(item.get("id"))
                    label = _string(item.get("label"))
                    if pack_id and label:
                        culture_pack_labels[pack_id] = label
            concept_aliases: dict[str, tuple[str, ...]] = {}
            raw_aliases = manifest.get("conceptAliases", {})
            if isinstance(raw_aliases, dict):
                for concept, aliases in raw_aliases.items():
                    concept_text = _string(concept)
                    alias_values = tuple(_string_list(aliases))
                    if concept_text and alias_values:
                        concept_aliases[concept_text] = alias_values
            collection_id = _string(manifest.get("id")) or directory.name
            collections.append(
                LoadedCollection(
                    id=collection_id,
                    name=_string(manifest.get("name")) or collection_id,
                    institution=_string(manifest.get("institution")) or "Unknown institution",
                    version=_string(manifest.get("version")) or "unversioned",
                    objects_sha256=actual_objects_sha256,
                    license=license_text,
                    source_url=_string(_first(manifest, ("sourceUrl", "source_url"))),
                    question_cards=extracted_questions
                    or _string_list(
                        _first(manifest, ("questionCards", "question_cards", "recommendedQuestions"), [])
                    ),
                    question_card_starters=question_card_starters,
                    question_card_limits=question_card_limits,
                    question_policies=question_policies,
                    evidence_domain_choices=evidence_domain_choices,
                    culture_pack_labels=culture_pack_labels,
                    concept_aliases=concept_aliases,
                    objects=objects,
                )
            )
        return collections

    def get(self, collection_id: str | None = None) -> LoadedCollection:
        collections = self.list()
        if not collections:
            raise CollectionDataError(
                "COLLECTION_NOT_FOUND",
                "No collection data was found. Add data/collections/<slug>/objects.json before generating an exhibition.",
                collectionsDir=str(self.collections_dir),
            )
        requested_id = collection_id or self.default_collection_id
        if requested_id is None:
            return collections[0]
        for collection in collections:
            if collection.id == requested_id:
                return collection
        raise CollectionDataError(
            "COLLECTION_NOT_FOUND",
            f"Collection '{requested_id}' was not found.",
            availableCollections=[collection.id for collection in collections],
            configuredDefaultCollection=self.default_collection_id,
        )

    @staticmethod
    def eligible_objects(collection: LoadedCollection) -> list[MuseumObject]:
        return [obj for obj in collection.objects if obj.evidence]

    def require_generation_ready(self, collection: LoadedCollection) -> list[MuseumObject]:
        eligible = self.eligible_objects(collection)
        if len(eligible) < 5:
            raise CollectionDataError(
                "COLLECTION_DATA_INSUFFICIENT",
                "At least five objects with image, rights, institution page, and evidence are required.",
                collectionId=collection.id,
                loadedObjectCount=len(collection.objects),
                eligibleObjectCount=len(eligible),
                requiredObjectCount=5,
            )
        return eligible

    def search(self, agenda: AgendaInput, collection: LoadedCollection) -> list[SearchResult]:
        eligible = self.require_generation_ready(collection)
        dense_signature = self._dense_manager.cache_signature(collection)
        cache_key = (
            collection.id,
            collection.version,
            # A rebuild can intentionally retain the public version string.
            # Object identity prevents an in-flight query on the old loaded
            # snapshot from repopulating the cache used by the new snapshot.
            id(collection),
            agenda.question.strip().casefold(),
            tuple(sorted({
                topic.strip().casefold()
                for topic in agenda.excluded_topics
                if topic.strip()
            })),
            self.rag_mode,
            dense_signature,
        )
        with self._cache_lock:
            cached = self._search_result_cache.get(cache_key)
            if cached is not None:
                self._search_result_cache.move_to_end(cache_key)
                return list(cached)

        query = _query_plan(agenda.question, collection.concept_aliases)
        excluded = [
            _query_plan(topic, collection.concept_aliases).anchor_tokens
            for topic in agenda.excluded_topics
        ]

        if query.browse_all and not excluded:
            results = [
                SearchResult(
                    obj=obj,
                    score=1.1 if obj.evidence_depth == EvidenceDepth.FULL.value else 1.0,
                    retrieval_sources=("browse_all",),
                )
                for obj in eligible
            ]
            with self._cache_lock:
                self._remember_search(cache_key, results)
            return results
        if not query.scoring_tokens and not query.browse_all:
            return []

        lexical_results = self._lexical_search(query, excluded, eligible)
        if self.rag_mode != "hybrid":
            with self._cache_lock:
                self._remember_search(cache_key, lexical_results)
            return lexical_results
        if not lexical_results and not query.dense_fallback_allowed:
            # Unknown subjects in either language retain the lexical refusal
            # gate; avoid loading/scanning dense matrices only to discard the
            # nearest neighbours afterwards.
            with self._cache_lock:
                self._remember_search(cache_key, [])
            return []

        dense_index = self._dense_manager.get(collection)
        if dense_index is None:
            # This path is intentionally a complete BM25 result, not a partial
            # approximation.  DenseIndexManager logs the concrete reason once.
            with self._cache_lock:
                self._remember_search(cache_key, lexical_results)
            return lexical_results

        try:
            results = self._hybrid_search(
                agenda.question,
                query,
                excluded,
                eligible,
                lexical_results,
                dense_index,
            )
        except Exception as error:
            # A query-time ONNX or cache read error must not make the collection
            # unavailable.  The warning is explicit and BM25 keeps refusal
            # semantics; no dense candidate is silently trusted.
            logger.warning(
                "hybrid RAG query degraded to BM25 for %s: %s",
                collection.id,
                error,
            )
            self._dense_manager.mark_query_failure(collection, error)
            # Do not cache a transient query failure under the hybrid key: the
            # next request must retry dense retrieval instead of silently
            # serving BM25 until the LRU entry expires.
            return lexical_results
        self._dense_manager.mark_query_success(collection, dense_index)
        with self._cache_lock:
            self._remember_search(cache_key, results)
        return results

    @staticmethod
    def _lexical_search(
        query: QueryPlan,
        excluded: list[set[str]],
        eligible: list[MuseumObject],
    ) -> list[SearchResult]:
        """The original fielded BM25 path, including its all-anchor gate."""

        document_frequency: Counter[str] = Counter()
        field_totals: Counter[str] = Counter()
        candidate_documents: list[tuple[MuseumObject, QueryDocument]] = []
        searched_document_count = 0
        for obj in eligible:
            # This full document lives for one loop iteration only.  Retaining
            # it for every object added ~400 MB to the worker and could kill the
            # API process.  Candidates keep only frequencies of query terms.
            document = _build_search_document(obj)
            effective_searchable = _effective_searchable(document)
            if any(
                topic_tokens and topic_tokens & effective_searchable
                for topic_tokens in excluded
            ):
                continue
            searched_document_count += 1
            field_totals.update(document.field_lengths)
            document_frequency.update(query.scoring_tokens & effective_searchable)
            if query.browse_all:
                candidate_documents.append(
                    (
                        obj,
                        QueryDocument(
                            term_frequencies={field: Counter() for field in document.term_frequencies},
                            field_lengths=document.field_lengths,
                            matched_anchor_terms=(),
                        ),
                    )
                )
                continue
            if query.anchor_groups and any(
                not group & effective_searchable for group in query.anchor_groups
            ):
                continue
            matched_anchor_terms = tuple(
                sorted(query.anchor_tokens & effective_searchable)
            )
            candidate_documents.append(
                (
                    obj,
                    QueryDocument(
                        term_frequencies={
                            field: Counter(
                                {
                                    token: frequency
                                    for token, frequency in frequencies.items()
                                    if token in query.scoring_tokens
                                }
                            )
                            for field, frequencies in document.term_frequencies.items()
                        },
                        field_lengths=document.field_lengths,
                        matched_anchor_terms=matched_anchor_terms,
                    ),
                )
            )

        index = SearchIndex(
            document_frequency=document_frequency,
            average_field_lengths={
                field: total / searched_document_count
                if searched_document_count
                else 0.0
                for field, total in field_totals.items()
            },
            document_count=searched_document_count,
        )
        results: list[SearchResult] = []
        for obj, document in candidate_documents:
            if query.browse_all:
                results.append(
                    SearchResult(
                        obj=obj,
                        score=1.1 if obj.evidence_depth == EvidenceDepth.FULL.value else 1.0,
                        retrieval_sources=("browse_all",),
                    )
                )
                continue
            score, field_scores = _object_score(obj, query, index, document)
            if score <= 0:
                continue
            matched_evidence_ids = tuple(
                chunk.id
                for chunk in obj.evidence
                if query.anchor_tokens & _tokens(chunk.text)
            )
            results.append(
                SearchResult(
                    obj=obj,
                    score=score,
                    matched_anchor_terms=document.matched_anchor_terms,
                    matched_evidence_ids=matched_evidence_ids,
                    field_scores=field_scores,
                )
            )
        results.sort(key=lambda result: (-result.score, result.obj.id))
        return results

    def _hybrid_search(
        self,
        question: str,
        query: QueryPlan,
        excluded: list[set[str]],
        eligible: list[MuseumObject],
        lexical_results: list[SearchResult],
        dense_index: Any,
    ) -> list[SearchResult]:
        """Fuse BM25 and dense recall, then rerank against evidence chunks.

        BM25 candidates have already passed every lexical anchor group.  Dense
        may add candidates only for a known multilingual concept and
        only when an institution evidence chunk clears the semantic floor.
        This preserves a truthful empty result for unknown subjects in either
        Chinese or English.
        """

        eligible_by_id = {obj.id: obj for obj in eligible}
        query_vector = dense_index.embed_query(question)
        object_dense_hits = dense_index.search_vector(
            query_vector,
            top_k=self.dense_top_k,
            allowed_object_ids=set(eligible_by_id),
        )
        # Search all individual evidence chunks as a second dense recall path.
        # This catches a relevant excerpt beyond the object's 512-token window.
        evidence_dense_hits = dense_index.search_evidence_vector(
            query_vector,
            top_k=self.dense_top_k * 2,
        )

        # Evidence vectors deliberately carry a little object context to make
        # catalogue fragments retrievable.  That context must never turn an
        # unrelated fragment (for example an acquisition credit) into the
        # query evidence cited by the curator.  Recheck the institution's raw
        # excerpt and its explicit ``supports`` annotation against the expanded
        # query vocabulary before an evidence id may enter the audit trace.
        # This is a query-time guard, so it does not change the frozen embedding
        # recipe or require rebuilding the index.
        query_evidence_tokens = query.anchor_tokens | query.scoring_tokens

        def raw_evidence_matches(obj: MuseumObject, evidence_id: str) -> bool:
            if not query_evidence_tokens:
                return False
            chunk = next(
                (item for item in obj.evidence if item.id == evidence_id),
                None,
            )
            if chunk is None:
                return False
            return bool(
                query_evidence_tokens
                & _tokens(" ".join(filter(None, (chunk.text, chunk.supports))))
            )

        object_dense_scores: dict[str, float] = {}
        object_dense_rank: dict[str, int] = {}
        for rank, hit in enumerate(object_dense_hits, start=1):
            if hit.score >= self.dense_min_score and hit.object_id in eligible_by_id:
                object_dense_scores[hit.object_id] = hit.score
                object_dense_rank[hit.object_id] = rank
        evidence_recall_scores: dict[str, float] = {}
        evidence_recall_rank: dict[str, int] = {}
        evidence_recall_ids: dict[str, list[str]] = {}
        for rank, hit in enumerate(evidence_dense_hits, start=1):
            if hit.score < self.evidence_min_score or hit.object_id not in eligible_by_id:
                continue
            if not raw_evidence_matches(eligible_by_id[hit.object_id], hit.evidence_id):
                continue
            evidence_recall_scores[hit.object_id] = max(
                evidence_recall_scores.get(hit.object_id, -1.0), hit.score
            )
            evidence_recall_rank.setdefault(hit.object_id, rank)
            evidence_recall_ids.setdefault(hit.object_id, []).append(hit.evidence_id)

        dense_object_ids = set(object_dense_scores) | set(evidence_recall_scores)
        dense_documents: dict[str, SearchDocument] = {}
        for object_id in tuple(dense_object_ids):
            obj = eligible_by_id[object_id]
            document = _build_search_document(obj)
            effective_searchable = _effective_searchable(document)
            if any(
                topic_tokens and topic_tokens & effective_searchable
                for topic_tokens in excluded
            ):
                dense_object_ids.discard(object_id)
                continue
            if query.strict_anchor_groups and any(
                not group & effective_searchable
                for group in query.strict_anchor_groups
            ):
                dense_object_ids.discard(object_id)
                continue
            dense_documents[object_id] = document

        # Collapse object-level and evidence-level dense rankings into one
        # dense channel before the outer BM25+dense RRF.  Evidence recall is
        # weighted slightly higher because it points to a citable institution
        # excerpt rather than only a truncated aggregate object document.
        dense_consensus = {
            object_id: (
                1.0 / (self.rrf_k + object_dense_rank[object_id])
                if object_id in object_dense_rank
                else 0.0
            )
            + (
                1.15 / (self.rrf_k + evidence_recall_rank[object_id])
                if object_id in evidence_recall_rank
                else 0.0
            )
            for object_id in dense_object_ids
        }
        dense_rank = {
            object_id: rank
            for rank, object_id in enumerate(
                sorted(
                    dense_object_ids,
                    key=lambda item: (-dense_consensus[item], item),
                ),
                start=1,
            )
        }
        dense_scores = {
            object_id: max(
                object_dense_scores.get(object_id, -1.0),
                evidence_recall_scores.get(object_id, -1.0),
            )
            for object_id in dense_object_ids
        }

        lexical_pool = lexical_results[: max(self.dense_top_k, self.hybrid_max_results)]
        lexical_by_id = {result.obj.id: result for result in lexical_pool}
        lexical_rank = {
            result.obj.id: rank for rank, result in enumerate(lexical_pool, start=1)
        }
        candidate_ids = set(lexical_by_id)
        if query.dense_fallback_allowed:
            candidate_ids.update(dense_scores)
        if not candidate_ids:
            return []

        # Ask for all of the small per-object evidence set.  The highest dense
        # row can have inherited relevance from object context; inspecting more
        # than the previous top three lets a genuinely matching description
        # survive behind an acquisition/classification fragment.
        evidence_hits = dense_index.evidence_hits(
            query_vector,
            candidate_ids,
            max_evidence_ids=5,
        )
        # Dense-only objects need evidence-level support.  BM25 objects retain
        # their lexical hard gate even when catalogue prose is very short.
        candidate_ids = {
            object_id
            for object_id in candidate_ids
            if object_id in lexical_by_id
            or (
                object_id in dense_scores
                and (evidence := evidence_hits.get(object_id)) is not None
                and evidence.score >= self.evidence_min_score
            )
        }
        if not candidate_ids:
            return []

        rrf_scores = {
            object_id: (
                (1.0 / (self.rrf_k + lexical_rank[object_id]))
                if object_id in lexical_rank
                else 0.0
            )
            + (
                (1.0 / (self.rrf_k + dense_rank[object_id]))
                if object_id in dense_rank
                else 0.0
            )
            for object_id in candidate_ids
        }
        top_rrf = max(rrf_scores.values(), default=1.0)
        top_bm25 = max((item.score for item in lexical_pool), default=1.0)

        def floor_normalise(score: float | None, floor: float) -> float:
            if score is None or score <= floor:
                return 0.0
            return min(1.0, (score - floor) / max(1.0 - floor, 1e-6))

        results: list[SearchResult] = []
        for object_id in candidate_ids:
            lexical = lexical_by_id.get(object_id)
            obj = lexical.obj if lexical else eligible_by_id[object_id]
            evidence = evidence_hits.get(object_id)
            evidence_scores = (
                evidence.evidence_scores
                if evidence and evidence.evidence_scores
                else tuple(
                    (evidence_id, evidence.score)
                    for evidence_id in (evidence.evidence_ids if evidence else ())
                )
            )
            qualified_dense_evidence = tuple(
                (evidence_id, score)
                for evidence_id, score in evidence_scores
                if score >= self.evidence_min_score
                and raw_evidence_matches(obj, evidence_id)
            )
            qualified_evidence_score = max(
                (score for _, score in qualified_dense_evidence),
                default=None,
            )
            evidence_supported = qualified_evidence_score is not None
            dense_score = dense_scores.get(object_id)
            bm25_signal = lexical.score / top_bm25 if lexical else 0.0
            dense_signal = floor_normalise(dense_score, self.dense_min_score)
            evidence_signal = floor_normalise(
                qualified_evidence_score,
                self.evidence_min_score,
            )
            rrf_signal = rrf_scores[object_id] / top_rrf if top_rrf else 0.0
            # RRF supplies the main rank consensus; direct evidence similarity
            # and the stronger original channel decide ties and weak tails.
            score = 100.0 * (
                0.55 * rrf_signal
                + 0.25 * evidence_signal
                + 0.20 * max(bm25_signal, dense_signal)
            )

            document = dense_documents.get(object_id)
            matched_anchor_terms = (
                lexical.matched_anchor_terms
                if lexical
                else tuple(
                    sorted(query.anchor_tokens & _effective_searchable(document))
                    if document
                    else ()
                )
            )
            evidence_ids = list(lexical.matched_evidence_ids if lexical else ())
            evidence_ids.extend(evidence_recall_ids.get(object_id, ()))
            if evidence_supported:
                evidence_ids.extend(
                    evidence_id for evidence_id, _ in qualified_dense_evidence
                )
            valid_ids = {chunk.id for chunk in obj.evidence}
            matched_evidence_ids = tuple(
                evidence_id
                for evidence_id in dict.fromkeys(evidence_ids)
                if evidence_id in valid_ids
            )
            sources = ["bm25"] if lexical else []
            if object_id in object_dense_rank and object_id in dense_object_ids:
                sources.append("dense_object")
            if object_id in evidence_recall_rank and object_id in dense_object_ids:
                sources.append("dense_evidence_recall")
            if evidence_supported:
                sources.append("evidence_rerank")
            fields = list(lexical.field_scores if lexical else ())
            if object_id in object_dense_scores:
                fields.append(
                    ("dense_object_cosine", round(object_dense_scores[object_id], 6))
                )
            if object_id in evidence_recall_scores:
                fields.append(
                    (
                        "dense_evidence_recall_cosine",
                        round(evidence_recall_scores[object_id], 6),
                    )
                )
            if qualified_evidence_score is not None:
                fields.append(
                    ("evidence_cosine", round(qualified_evidence_score, 6))
                )
            elif evidence:
                # Keep the candidate-level diagnostic without presenting it as
                # a query-supported institution excerpt.
                fields.append(
                    ("evidence_cosine_unverified", round(evidence.score, 6))
                )
            fields.append(("rrf", round(rrf_scores[object_id], 8)))
            results.append(
                SearchResult(
                    obj=obj,
                    score=score,
                    matched_anchor_terms=matched_anchor_terms,
                    matched_evidence_ids=matched_evidence_ids,
                    field_scores=tuple(fields),
                    retrieval_sources=tuple(sources),
                    dense_score=dense_score,
                    evidence_score=qualified_evidence_score,
                )
            )

        results.sort(key=lambda result: (-result.score, result.obj.id))
        return results[: self.hybrid_max_results]

    def retrieval_status(self, collection: LoadedCollection) -> DenseStatus:
        """Return the current retrieval mode/reason for audit and diagnostics."""

        return self._dense_manager.status(collection)

    def _remember_search(
        self,
        cache_key: tuple[Any, ...],
        results: list[SearchResult],
    ) -> None:
        self._search_result_cache[cache_key] = list(results)
        self._search_result_cache.move_to_end(cache_key)
        while len(self._search_result_cache) > 12:
            self._search_result_cache.popitem(last=False)

    @staticmethod
    def match_question_policy(
        collection: LoadedCollection, question: str
    ) -> tuple[QuestionPolicy, float] | None:
        normalized = _normalized_question(question)
        cards = [
            policy for policy in collection.question_policies if policy.source == "question_card"
        ]
        regressions = [
            policy for policy in collection.question_policies if policy.source == "regression"
        ]
        for policies in (cards, regressions):
            for policy in policies:
                if _normalized_question(policy.question) == normalized:
                    return policy, 1.0
        if cards:
            card, score = max(
                ((policy, _question_similarity(question, policy.question)) for policy in cards),
                key=lambda pair: pair[1],
            )
            if score >= 0.72:
                return card, score
        if regressions:
            regression, score = max(
                (
                    (policy, _question_similarity(question, policy.question))
                    for policy in regressions
                ),
                key=lambda pair: pair[1],
            )
            if score >= 0.9:
                return regression, score
        return None

    @staticmethod
    def recommend_questions(collection: LoadedCollection) -> list[str]:
        if collection.question_cards:
            return collection.question_cards[:3]
        themes: list[str] = []
        for obj in collection.objects:
            for theme in obj.themes:
                if theme not in themes:
                    themes.append(theme)
        if themes:
            return [f"这些藏品如何呈现“{theme}”的不同面向？" for theme in themes[:3]]
        return [
            "这些藏品在材料、用途与意义上有什么差异？",
            "机构记录能支持我们怎样理解这些藏品？",
        ]
