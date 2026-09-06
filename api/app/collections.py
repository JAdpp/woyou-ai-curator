from __future__ import annotations

import json
import logging
import hashlib
import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

from .models import AgendaInput, EvidenceChunk, EvidenceDepth, MuseumObject
from .retrieval_trace import RetrievalTraceWriter
from .retrieval_filters import FilterSpec
from .cultural_normalization import effective_culture_pack_ids
from .structured_filters import (
    EvidenceSearchHit,
    StructuredFilterError,
    StructuredFilterIndex,
    filter_index_path,
)
from .dense_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    DenseIndexManager,
    DenseStatus,
    EmbeddingProviderSpec,
)


CC0_LICENSE = "CC0 1.0"
CC0_RIGHTS_URI = "https://creativecommons.org/publicdomain/zero/1.0/"
CC_BY_4_0_LICENSE = "CC BY 4.0"
CC_BY_4_0_RIGHTS_URI = "https://creativecommons.org/licenses/by/4.0/"
FIELD_LEVEL_OPEN_ACCESS_INSTITUTIONS = frozenset({"aic", "cma", "met"})
HYBRID_RETRIEVAL_METHOD = (
    "sqlite_object_bm25_evidence_bm25_qwen_embedding_rrf_qwen_rerank"
)
HYBRID_RETRIEVAL_VERSION = "hybrid-rag-v4"
AGENTIC_HYBRID_RETRIEVAL_METHOD = (
    "fielded_bm25_evidence_bm25_dense_rrf_qwen_rerank"
)
AGENTIC_HYBRID_RETRIEVAL_VERSION = "agentic-hybrid-rag-v3"
BM25_RETRIEVAL_METHOD = "fielded_bm25_hard_anchor"
BM25_RETRIEVAL_VERSION = "bm25-v1"

# Open-vocabulary queries do not have a reviewed alias group to provide the
# lexical safety gate used by concrete subjects such as cat or emperor.  These
# calibrated floors are therefore intentionally higher than the general dense
# floors.  They are model-specific retrieval thresholds, not topic rules: any
# visitor-authored subject may pass them when the collection is semantically
# close enough, while out-of-domain nearest neighbours remain a refusal.
OPEN_QUERY_DENSE_MIN_SCORE = 0.44
OPEN_QUERY_EVIDENCE_MIN_SCORE = 0.46
logger = logging.getLogger("app.retrieval")


def _call_with_optional_deadline(
    callback: Any,
    *args: Any,
    deadline: float | None,
    **kwargs: Any,
) -> Any:
    """Pass cooperative deadlines to production adapters without breaking fakes."""

    code = getattr(callback, "__code__", None)
    if deadline is not None and code is not None:
        accepts_kwargs = bool(code.co_flags & 0x08)
        if accepts_kwargs or "deadline" in code.co_varnames:
            kwargs["deadline"] = deadline
    return callback(*args, **kwargs)


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
    # Set-level requirements are separate from this object's admission gate.
    # Only the current source-bound audit may populate these witnesses; recall
    # scores, inherited audit state and collection metadata cannot create them.
    set_witnesses: tuple[dict[str, Any], ...] = ()


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
    # Acquisition/source-history rows are normally precision hazards: a donor
    # name or credit line does not make an object topical. They become eligible
    # evidence only when the visitor explicitly asks about provenance,
    # acquisition, removal, ownership or restitution.
    allow_provenance_evidence: bool
    # Dense recall is the open-vocabulary route.  Reviewed aliases improve a
    # query but are not a whitelist: a new subject may still use embeddings.
    dense_fallback_allowed: bool
    # Open queries have no reviewed lexical concept.  They use stricter,
    # model-calibrated cosine floors and are later eligible for LLM audit.
    open_semantic_query: bool
    # Familiar concrete entities (cat/dog/bird...) retain an exact lexical
    # subject gate even when dense retrieval is active.
    strict_anchor_groups: tuple[frozenset[str], ...]
    # Depicted-person concepts such as emperor/king must identify the object's
    # subject in its title. A reign date or a donor named King is context, not
    # evidence that the object represents a ruler.
    title_anchor_groups: tuple[frozenset[str], ...]


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
    # Visitor questions about rulership rarely use the art-historical routing
    # term “王权”. Keep these ordinary Chinese forms as reviewed, strict
    # subject groups so they can reach English-language catalogues without
    # weakening the relevance gate for unknown questions.
    "皇帝": (
        "emperor",
        "empress",
        "monarch",
        "sovereign",
    ),
    "帝王": (
        "emperor",
        "empress",
        "monarch",
        "sovereign",
    ),
    "国王": (
        "king",
        "queen",
        "monarch",
        "sovereign",
    ),
    "君主": (
        "monarch",
        "sovereign",
        "emperor",
        "empress",
        "king",
        "queen",
        "ruler",
    ),
    "皇权": (
        "imperial",
        "emperor",
        "empress",
        "monarch",
        "sovereign",
        "royal",
        "ruler",
        "throne",
        "crown",
        "regalia",
    ),
}


TITLE_SUBJECT_CONCEPTS = frozenset({"皇帝", "帝王", "国王", "君主"})


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
    "告诉我们",
    "告诉我",
    "为什么",
    "能不能",
    "怎么",
    "如何",
    "是否",
    "能否",
    "分别",
    "逐个",
    "随便",
    "看看",
    "逛逛",
    "带我",
    "推荐",
    "有意思的",
    "有趣的",
    "相关的",
    "有关的",
    "关于",
    "藏品",
    "文物",
    "展品",
    "艺术品",
    "各文化地区",
    "不同文化地区",
    "不同的文化地区",
    "不同地方",
    "不同的地方",
    "各个文化",
    "不同文化",
    "不同的文化",
    "不同文明",
    "不同的文明",
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
    r"不同(?:的)?文化",
    r"不同(?:的)?文明",
    r"多种文化",
    r"跨文化",
    r"全球",
    r"各(?:个)?地区",
    r"不同(?:的)?地区",
    r"不同(?:的)?地方",
    r"各地",
    r"多个地方",
    r"几个地区",
    r"各处",
    r"各国(?:文化|地区)?",
    r"不同国家",
    r"多个国家",
    r"世界各地",
    r"相隔(?:很)?远",
    r"远隔重洋",
    r"不同(?:的)?社会",
    r"多个社会",
    r"遥远(?:的)?社会",
    r"across\s+(?:different\s+)?cultures?",
    r"different\s+cultures?",
    r"multiple\s+cultures?",
    r"around\s+the\s+world",
    r"global(?:ly)?",
)


# When a visitor asks for a cross-cultural comparison, named origins are
# coverage legs across the eventual object set. They must never become AND
# anchors that require one object to be simultaneously East Asian, European
# and American. The controlled labels mirror the collection's culture-pack
# routing vocabulary; they affect retrieval structure, not cultural claims.
CULTURAL_COMPARISON_LEG_CONCEPTS = frozenset(
    {
        "东亚",
        "中国",
        "日本",
        "朝鲜",
        "韩国",
        "南亚",
        "印度",
        "东南亚",
        "西亚",
        "中东",
        "北非",
        "伊斯兰",
        "埃及",
        "波斯",
        "欧洲",
        "非洲",
        "美洲",
        "大洋洲",
    }
)


# Explicitly naming two or more origins already expresses a cross-cultural
# comparison, even when the visitor does not also write “不同文化”.  Keep the
# groups coarse and disjoint enough that one phrase cannot satisfy two legs.
NAMED_CULTURAL_LEG_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"东亚|中国|日本|韩国|朝鲜|\b(?:east asia(?:n)?|china|chinese|japan(?:ese)?|korea(?:n)?)\b",
        r"南亚|印度|\b(?:south asia(?:n)?|india(?:n)?)\b",
        r"东南亚|越南|泰国|柬埔寨|缅甸|印尼|\b(?:southeast asia(?:n)?|vietnam(?:ese)?|thai(?:land)?|cambodia(?:n)?|myanmar|burma|indonesia(?:n)?)\b",
        r"西亚|中东|北非|伊朗|波斯|埃及|\b(?:west asia(?:n)?|middle east(?:ern)?|north africa(?:n)?|iran(?:ian)?|persia(?:n)?|egypt(?:ian)?)\b",
        r"非洲|\bafrica(?:n)?\b",
        r"美洲|墨西哥|秘鲁|\b(?:americas?|mexic(?:o|an)|peru(?:vian)?)\b",
        r"欧洲|希腊|罗马|荷兰|\b(?:europe(?:an)?|greece|greek|roman|rome|netherlands|dutch)\b",
        r"大洋洲|太平洋文化|\b(?:oceania(?:n)?|pacific island)\b",
    )
)


PROVENANCE_QUERY_PATTERN = re.compile(
    r"(?:来源史?|流传|传承|入藏|征集|购得|购买|收购|出售|"
    r"交易|捐赠|赠予|原主人|所有权|权属|合法所有|掠夺|劫掠|"
    r"战利品|远征军|殖民采集|归还|返还|遣返|"
    r"\bprovenance\b|\bacquisition\b|\bacquired\b|\bpurchas(?:e|ed)\b|"
    r"\bsold\b|\bsale\b|\bdonat(?:e|ed|ion)\b|\bbequest\b|"
    r"\bownership\b|\blegal\s+title\b|\bloot(?:ed|ing)?\b|"
    r"\bplunder(?:ed|ing)?\b|\brepatriat(?:e|ed|ion)\b|"
    r"\brestitut(?:e|ed|ion)\b)",
    re.IGNORECASE,
)


# Catalogue records describe an absent maker mark and an uncertain maker as
# different facts.  Both can nevertheless occur in visitor questions about
# the *operation* of attribution.  These state patterns therefore only decide
# whether to add museum-method retrieval vocabulary; they never turn
# ``unsigned`` into evidence for ``unknown artist`` (or the reverse).
MISSING_SIGNATURE_STATE_PATTERN = re.compile(
    r"(?:没(?:有)?|无|未|缺少|缺乏|不带|找不到)(?:任何)?(?:作者|创作者|制作者|工匠)?(?:的)?(?:署名|签名|落款|款识)|"
    r"(?:署名|签名|落款|款识)(?:缺失|不存在|不可见|未见)|"
    r"\bunsigned\b|\bwithout\s+(?:an?\s+)?(?:artist(?:'s)?\s+)?signature\b|"
    r"\bno\s+(?:known\s+)?signature\b|\bnot\s+signed\b|"
    r"\black(?:s|ing)?\s+(?:an?\s+)?signature\b",
    re.IGNORECASE,
)

UNKNOWN_CREATOR_STATE_PATTERN = re.compile(
    r"(?:作者|创作者|制作者|工匠|画家|艺术家)(?:身份)?(?:不明|未知|不可考|无法确定)|佚名|"
    r"(?:不知道|不清楚)(?:它?的?)?(?:作者|创作者|制作者|工匠|谁(?:做|画|制作|创作))|"
    r"\banonymous\b|\bunattributed\b|"
    r"\bunknown\s+(?:artist|maker|author|creator)\b|"
    r"\b(?:artist|maker|author|creator)\s+(?:unknown|unidentified)\b",
    re.IGNORECASE,
)

ATTRIBUTION_METHOD_INTENT_PATTERN = re.compile(
    r"怎么|如何|怎样|凭什么|依据什么|通过什么|"
    r"判断|鉴定|辨认|确定|识别|推断|考证|归属|"
    r"谁(?:创作|制作|做|画|写)|"
    r"\bhow\b|\bdetermin\w*\b|\bidentif\w*\b|"
    r"\battribut(?:e|ed|es|ing|ion|ions)\b|\bauthorship\b|"
    r"\bwho\s+(?:made|created|painted|wrote)\b|\bconnoisseurship\b",
    re.IGNORECASE,
)

ATTRIBUTION_CREATOR_TARGET_PATTERN = re.compile(
    r"作者|创作者|制作者|工匠|画家|艺术家|归属|"
    r"谁(?:创作|制作|做|画|写)|"
    r"\battribut(?:e|ed|es|ing|ion|ions)\b|\bauthorship\b|"
    r"\b(?:artist|maker|author|creator)\b|"
    r"\bwho\s+(?:made|created|painted|wrote)\b|\bconnoisseurship\b",
    re.IGNORECASE,
)

# High-precision catalogue vocabulary for records that discuss attribution as
# a curatorial judgement rather than merely printing a creator field.  Broad
# words such as ``artist``, ``maker`` and ``unknown`` are deliberately absent
# from the anchor group because they occur in thousands of unrelated records.
ATTRIBUTION_METHOD_ANCHOR_TERMS = frozenset(
    {
        "attribution",
        "reattributed",
        "reattribution",
        "stylistic",
        "connoisseurship",
        "authorship",
    }
)
ATTRIBUTION_METHOD_SCORING_TERMS = frozenset(
    {
        *ATTRIBUTION_METHOD_ANCHOR_TERMS,
        "follower",
        "workshop",
        "attribute",
        "attributed",
        "comparison",
        "study",
        "research",
        "technical",
        "analysis",
        "technology",
        "revealed",
        "evidence",
        "characteristic",
        "style",
        "school",
    }
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


PURE_BROWSE_QUERY_PATTERNS = (
    r"(?:我)?(?:还)?(?:没想好|没有想好)(?:(?:先|随便|带我))*(?:看看|逛逛)吧?",
    r"(?:请)?(?:随便)?(?:带我)?(?:看看|逛逛)吧?",
    r"(?:请)?(?:随便)?给我看(?:点|些)?(?:有意思|有趣|好看)?(?:的)?(?:藏品|展品|艺术品)吧?",
    r"(?:请)?(?:随便)?推荐(?:一些|几个)?(?:有意思|有趣|好看)?(?:的)?(?:藏品|展品|艺术品)?吧?",
    r"这些藏品(?:之间)?(?:有)?什么(?:关系|联系|共同点)吧?",
    r"recommend(?:something|anything|objects?)",
    r"surpriseme",
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
    place = _string(_first(raw, ALIASES["place"]))
    culture_pack_ids = _string_list(_first(raw, ALIASES["culture_pack_ids"], []))
    culture_pack_ids = effective_culture_pack_ids({"culture": culture, "place": place}, culture_pack_ids)
    # Taxonomy 1.1.0 treated AIC's broad "Arts of Asia" department as a
    # strong East-Asia signal.  Correct that frozen-data artefact at load time
    # when controlled origin fields explicitly and exclusively identify Iran;
    # future imports are fixed at the taxonomy source as well.
    controlled_origin = " ".join(filter(None, (culture, place, creator))).casefold()
    if (
        re.search(r"\b(?:iran|iranian|persia|persian)\b", controlled_origin)
        and not re.search(
            r"\b(?:china|chinese|japan|japanese|korea|korean)\b",
            controlled_origin,
        )
    ):
        culture_pack_ids = [
            "west_asia_north_africa",
            *(pack for pack in culture_pack_ids if pack not in {"east_asia", "west_asia_north_africa"}),
        ]
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
        place=place,
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
        culture_pack_ids=culture_pack_ids,
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


CATALOGUE_SPELLING_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "catalog": ("catalogue",),
    "catalogue": ("catalog",),
    "center": ("centre",),
    "centre": ("center",),
    "color": ("colour",),
    "colour": ("color",),
    "fiber": ("fibre",),
    "fibre": ("fiber",),
    "jewelry": ("jewellery",),
    "jewellery": ("jewelry",),
    "mold": ("mould",),
    "mould": ("mold",),
    "modeled": ("modelled",),
    "modelled": ("modeled",),
}


def _english_token_variants(token: str) -> set[str]:
    """Normalize orthographic form without erasing the source spelling.

    Museum records mix hyphenated compounds, open compounds, possessives and
    British/American spelling. Retaining the full token preserves precision;
    adding its constituents lets an atomic planner query such as
    ``hand-built`` match institution prose such as ``built by hand``.
    """

    variants = {token}
    parts = [part for part in token.split("-") if len(part) > 1]
    variants.update(parts)
    if token.endswith("'s") and len(token) > 3:
        variants.add(token[:-2])
    for value in tuple(variants):
        variants.update(CATALOGUE_SPELLING_EQUIVALENTS.get(value, ()))
    return variants


def _token_sequence(text: str) -> list[str]:
    lowered = text.casefold()
    raw_english = re.findall(r"[a-z0-9][a-z0-9'-]+", lowered)
    english = [
        variant
        for token in raw_english
        for variant in sorted(_english_token_variants(token))
        if variant not in TOKEN_STOPWORDS
    ]
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
    return english + chinese


def _tokens(text: str) -> set[str]:
    return set(_token_sequence(text))


ATOMIC_QUERY_FRAME_WORDS = frozenset(
    {
        "catalog",
        "catalogue",
        "evidence",
        "made",
        "method",
        "methods",
        "museum",
        "object",
        "objects",
        "process",
        "processes",
        "record",
        "records",
        "technique",
        "techniques",
    }
)


def _atomic_catalogue_query_plan(question: str, base: QueryPlan) -> QueryPlan:
    """Require every content leg in an LLM-planned atomic query.

    Visitor questions remain broad hybrid searches. A planner-generated query
    has a narrower contract: its few content words describe one evidence axis.
    Treating them as one large OR group lets ``pottery`` swamp
    ``hand-built pottery``. Separate variant groups preserve the axis while
    still accepting plurals, compounds and catalogue spelling variants.
    """

    raw_words = re.findall(r"[a-z0-9][a-z0-9'-]+", question.casefold())
    if not raw_words:
        return base
    groups: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    for raw_word in raw_words:
        parts = [part for part in raw_word.split("-") if len(part) > 1]
        for part in parts or [raw_word]:
            if part in TOKEN_STOPWORDS or part in ATOMIC_QUERY_FRAME_WORDS:
                continue
            variants = _english_token_variants(part)
            variants = _english_inflections(variants)
            catalogue_aliases = set(variants)
            for base_group in base.anchor_groups:
                if base_group.intersection(variants):
                    catalogue_aliases.update(base_group)
            group = frozenset(
                variant
                for variant in catalogue_aliases
                if variant not in TOKEN_STOPWORDS
                and variant not in ATOMIC_QUERY_FRAME_WORDS
            )
            if group and group not in seen:
                seen.add(group)
                groups.append(group)
    if len(groups) < 2:
        return base
    return replace(
        base,
        anchor_tokens=frozenset().union(*groups),
        anchor_groups=tuple(groups),
        strict_anchor_groups=tuple(groups),
    )


_EXPLICIT_DOG_TERMS = re.compile(
    r"\b(?:dogs?|pupp(?:y|ies)|hounds?|canids?)\b",
    re.IGNORECASE,
)
_CANINE_WORD = re.compile(r"\bcanines?\b", re.IGNORECASE)
_DENTAL_WORDS = frozenset({"tooth", "teeth", "dental", "fang", "fangs"})
_KING_QUEEN_WORD = re.compile(r"\b(?:kings?|queens?)\b", re.IGNORECASE)
_ACQUISITION_WORDS = frozenset(
    {"gift", "bequest", "donated", "donor", "fund", "credit", "collection"}
)
_RULERSHIP_WORDS = frozenset(
    {
        "emperor",
        "empress",
        "monarch",
        "sovereign",
        "royal",
        "ruler",
        "reign",
        "kingdom",
        "throne",
        "crown",
        "regalia",
    }
)


def _contextual_anchor_suppressions(text: str) -> frozenset[str]:
    """Resolve a small set of high-cost catalogue homonyms conservatively.

    ``canine`` commonly denotes a dog, but museum descriptions also use it as
    the anatomical adjective in phrases such as ``leopard canine teeth``.  A
    dog query must not admit an otherwise unrelated object on that word alone.
    Suppress it only when every occurrence sits within two tokens of explicit
    dental vocabulary and the record has no independent dog signal.  Phrases
    such as ``canine companion`` therefore keep their normal dog meaning.
    """

    suppressed: set[str] = set()
    words = re.findall(r"[a-z][a-z'-]*", text.casefold())
    if _CANINE_WORD.search(text) and not (
        _EXPLICIT_DOG_TERMS.search(text) or re.search(r"狗|犬(?![齿牙])", text)
    ):
        canine_positions = [
            index for index, word in enumerate(words) if word in {"canine", "canines"}
        ]
        dental_positions = {
            index for index, word in enumerate(words) if word in _DENTAL_WORDS
        }
        if dental_positions and canine_positions and all(
            any(abs(canine - dental) <= 2 for dental in dental_positions)
            for canine in canine_positions
        ):
            suppressed.update({"canine", "canines"})

    # ``King`` and ``Queen`` are also surnames in museum credit lines. A gift
    # from Ralph King is not evidence of rulership. Suppress the anchor only
    # when every occurrence is locally attached to acquisition wording and no
    # independent rulership vocabulary exists elsewhere in the record.
    if _KING_QUEEN_WORD.search(text) and not (_RULERSHIP_WORDS & set(words)):
        acquisition_positions = {
            index for index, word in enumerate(words) if word in _ACQUISITION_WORDS
        }
        for term in ("king", "kings", "queen", "queens"):
            positions = [index for index, word in enumerate(words) if word == term]
            if positions and acquisition_positions and all(
                any(abs(position - acquisition) <= 4 for acquisition in acquisition_positions)
                for position in positions
            ):
                suppressed.add(term)

    return frozenset(suppressed)


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
    if any(
        re.search(pattern, question, re.IGNORECASE)
        for pattern in CROSS_CULTURAL_PATTERNS
    ):
        return True
    return sum(bool(pattern.search(question)) for pattern in NAMED_CULTURAL_LEG_PATTERNS) >= 2


def question_requests_provenance(question: str) -> bool:
    """Whether source-history evidence is part of the visitor's subject.

    This is an evidence-scope switch, not an answerability shortcut. Allowing
    the catalogue's acquisition row into retrieval never proves consent,
    lawful title or the right remedy; the LLM audit still evaluates that
    predicate against the exact source text.
    """

    return PROVENANCE_QUERY_PATTERN.search(question) is not None


def _attribution_method_query_terms(
    question: str,
) -> tuple[frozenset[str], frozenset[str]] | None:
    """Bridge visitor phrasing to evidence-bearing attribution vocabulary.

    A rewrite is activated only when a creator-field uncertainty or missing
    mark is paired with an attribution operation (for example, ``how`` or
    ``鉴定``).  A request simply asking to see unsigned works therefore does
    not get broadened into records about connoisseurship.  State-specific
    terms are score boosts, not facts accepted by retrieval: the later LLM
    audit still has to cite an institution excerpt for every accepted object.
    """

    missing_signature = MISSING_SIGNATURE_STATE_PATTERN.search(question) is not None
    unknown_creator = UNKNOWN_CREATOR_STATE_PATTERN.search(question) is not None
    if not (missing_signature or unknown_creator):
        return None
    if ATTRIBUTION_METHOD_INTENT_PATTERN.search(question) is None:
        return None
    # A generic "how" near an unsigned object might ask about conservation,
    # display or digitisation.  Require the operation to target creator
    # identity/attribution before adding attribution-method vocabulary.
    if ATTRIBUTION_CREATOR_TARGET_PATTERN.search(question) is None:
        return None

    anchors = _english_inflections(set(ATTRIBUTION_METHOD_ANCHOR_TERMS))
    scoring = _english_inflections(set(ATTRIBUTION_METHOD_SCORING_TERMS))
    if missing_signature:
        scoring.update({"unsigned"})
    if unknown_creator:
        scoring.update({"anonymous", "unattributed", "unidentified"})
    return frozenset(anchors), frozenset(scoring)


def _is_browse_query(question: str) -> bool:
    # Browse is an explicit no-topic intent, not a phrase that can be mixed
    # into an otherwise topical question. Substring matching let requests such
    # as “随便给我看看海豚…” bypass embeddings and the evidence audit entirely.
    compact = re.sub(r"[\s，,。.!！？?、;；:：'\"]+", "", question.casefold())
    return any(
        re.fullmatch(pattern, compact, re.IGNORECASE)
        for pattern in PURE_BROWSE_QUERY_PATTERNS
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
    allow_provenance_evidence = question_requests_provenance(question)
    matched: list[tuple[str, tuple[str, ...]]] = []
    for concept, aliases in aliases_by_concept.items():
        if _concept_present(question, concept) or any(
            _phrase_present(question, alias) for alias in aliases
        ):
            matched.append((concept, aliases))

    # Cross-cultural wording controls diversification; it is not itself the
    # subject of every selected object.
    structural_concepts = (
        {"跨文化", "交流", *CULTURAL_COMPARISON_LEG_CONCEPTS}
        if cross_cultural
        else set()
    )
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
    title_anchor_groups: list[frozenset[str]] = []
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
        if concept in TITLE_SUBJECT_CONCEPTS:
            title_anchor_groups.append(frozenset(concept_group))
        scoring_tokens.update(concept_group)

    attribution_method_terms = _attribution_method_query_terms(question)
    if attribution_method_terms is not None:
        method_anchors, method_scoring = attribution_method_terms
        anchor_tokens.update(method_anchors)
        anchor_groups.append(method_anchors)
        scoring_tokens.update(method_scoring)

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
        title_anchor_groups.clear()
    return QueryPlan(
        scoring_tokens=scoring_tokens,
        anchor_tokens=anchor_tokens,
        anchor_groups=tuple(anchor_groups),
        exact_tokens=exact_tokens,
        browse_all=browse_all,
        cross_cultural=cross_cultural,
        allow_provenance_evidence=allow_provenance_evidence,
        # Embeddings, not a hand-written alias whitelist, are the fallback for
        # any non-empty visitor subject.  Unknown subjects use higher semantic
        # floors below and are audited before generation when a model is
        # configured.
        dense_fallback_allowed=bool(anchor_tokens or scoring_tokens),
        # Every topical visitor question remains semantically open at runtime.
        # Reviewed aliases add lexical precision, but never certify that the
        # rest of an arbitrary question has been understood. Browse requests
        # are the sole exception because they deliberately have no predicate.
        open_semantic_query=not browse_all,
        strict_anchor_groups=tuple(strict_anchor_groups),
        title_anchor_groups=tuple(title_anchor_groups),
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
        embedding_provider_spec: EmbeddingProviderSpec | None = None,
        dense_top_k: int = 200,
        dense_min_score: float = 0.28,
        evidence_min_score: float = 0.30,
        rrf_k: int = 60,
        hybrid_max_results: int = 250,
        dense_manager: DenseIndexManager | None = None,
        reranker: Any | None = None,
        rerank_candidate_count: int = 60,
        rerank_top_n: int = 24,
        rerank_instruct: str = "",
        trace_writer: RetrievalTraceWriter | None = None,
        structured_filters_enabled: bool = True,
        structured_filter_index_dir: Path | None = None,
        evidence_bm25_top_k: int = 120,
    ) -> None:
        if rag_mode not in {"bm25", "hybrid", "shadow"}:
            raise ValueError("rag_mode must be 'bm25', 'hybrid', or 'shadow'")
        if dense_top_k <= 0 or rrf_k <= 0 or hybrid_max_results <= 0:
            raise ValueError("dense_top_k, rrf_k and hybrid_max_results must be positive")
        if rerank_candidate_count <= 0 or rerank_top_n <= 0:
            raise ValueError("rerank candidate and output counts must be positive")
        if rerank_top_n > rerank_candidate_count or rerank_candidate_count > 500:
            raise ValueError("rerank counts must satisfy top_n <= candidates <= 500")
        if not 1 <= evidence_bm25_top_k <= 500:
            raise ValueError("evidence_bm25_top_k must be between 1 and 500")
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
        self.reranker = reranker
        self.rerank_candidate_count = rerank_candidate_count
        self.rerank_top_n = rerank_top_n
        self.rerank_instruct = rerank_instruct.strip()
        self.trace_writer = trace_writer
        runtime_root = collections_dir.parent.parent / "api" / "runtime" / "cache"
        self.structured_filters_enabled = structured_filters_enabled
        self.structured_filter_index_dir = (
            structured_filter_index_dir or runtime_root / "filters"
        )
        self.evidence_bm25_top_k = evidence_bm25_top_k
        self._structured_indexes: dict[tuple[str, str, str], StructuredFilterIndex] = {}
        self._structured_index_failures: set[tuple[str, str, str]] = set()
        self._dense_manager = dense_manager or DenseIndexManager(
            enabled=rag_mode in {"hybrid", "shadow"},
            index_root=dense_index_dir or runtime_root / "rag",
            model_cache_dir=embedding_model_cache_dir or runtime_root / "fastembed",
            model_name=embedding_model,
            provider_spec=embedding_provider_spec,
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

    def _search_cache_key(
        self,
        agenda: AgendaInput,
        collection: LoadedCollection,
        dense_signature: tuple[Any, ...],
        *,
        atomic: bool = False,
        filters: FilterSpec | None = None,
    ) -> tuple[Any, ...]:
        return (
            collection.id,
            collection.version,
            # A rebuild can intentionally retain the public version string.
            # Object identity prevents an in-flight query on the old loaded
            # snapshot from repopulating the cache used by the new snapshot.
            id(collection),
            agenda.question.strip().casefold(),
            tuple(
                sorted(
                    {
                        topic.strip().casefold()
                        for topic in agenda.excluded_topics
                        if topic.strip()
                    }
                )
            ),
            self.rag_mode,
            atomic,
            filters or FilterSpec(),
            dense_signature,
        )

    @staticmethod
    def _structured_index_key(
        collection: LoadedCollection,
    ) -> tuple[str, str, str]:
        return (
            collection.id,
            collection.version,
            str(collection.objects_sha256 or ""),
        )

    def _structured_index(
        self,
        collection: LoadedCollection,
        *,
        required: bool,
    ) -> StructuredFilterIndex | None:
        """Load one verified read-only derivative, failing closed for filters."""

        key = self._structured_index_key(collection)
        if not self.structured_filters_enabled:
            if required:
                raise CollectionDataError(
                    "STRUCTURED_FILTER_UNAVAILABLE",
                    "The query requested hard catalogue filters, but structured filtering is disabled.",
                    collectionId=collection.id,
                )
            return None
        cached = self._structured_indexes.get(key)
        if cached is not None:
            expected_directory = filter_index_path(
                self.structured_filter_index_dir,
                collection,
            ).resolve()
            if cached.directory == expected_directory:
                return cached
            # A derivative schema upgrade can intentionally keep the same
            # collection/version/SHA. Do not pin the process to the old path.
            self._structured_indexes.pop(key, None)
        if key in self._structured_index_failures:
            expected_manifest = (
                filter_index_path(self.structured_filter_index_dir, collection)
                / "manifest.json"
            )
            if expected_manifest.is_file():
                # Allow an index repaired/rebuilt while the API stays alive to
                # be verified on the next query.
                self._structured_index_failures.discard(key)
            else:
                if required:
                    raise CollectionDataError(
                        "STRUCTURED_FILTER_UNAVAILABLE",
                        "The query requested hard catalogue filters, but the verified filter index is unavailable.",
                        collectionId=collection.id,
                    )
                return None
        try:
            index = StructuredFilterIndex.for_collection(
                self.structured_filter_index_dir,
                collection,
            )
        except StructuredFilterError as error:
            self._structured_index_failures.add(key)
            logger.warning(
                "structured retrieval index unavailable for %s: %s",
                collection.id,
                error,
            )
            if required:
                raise CollectionDataError(
                    "STRUCTURED_FILTER_UNAVAILABLE",
                    "The query requested hard catalogue filters, but the verified filter index is unavailable.",
                    collectionId=collection.id,
                    reason=type(error).__name__,
                ) from error
            return None
        self._structured_indexes[key] = index
        return index

    def _retrieval_cache_signature(
        self,
        collection: LoadedCollection,
        index: StructuredFilterIndex | None,
    ) -> tuple[Any, ...]:
        structured = (
            (
                index.manifest.get("formatVersion"),
                index.manifest.get("schemaVersion"),
                index.manifest.get("databaseSha256"),
                index.manifest.get("objectSearchMode"),
                index.manifest.get("evidenceSearchMode"),
            )
            if index is not None
            else (None,)
        )
        reranker = (
            type(self.reranker).__name__ if self.reranker is not None else None,
            getattr(self.reranker, "rerank_model", None),
            self.rerank_candidate_count,
            self.rerank_top_n,
            self.rerank_instruct,
        )
        return (
            self._dense_manager.cache_signature(collection),
            structured,
            reranker,
        )

    def _eligible_with_filters(
        self,
        collection: LoadedCollection,
        eligible: list[MuseumObject],
        filters: FilterSpec,
        *,
        load_for_evidence_search: bool,
    ) -> tuple[list[MuseumObject], StructuredFilterIndex | None]:
        required = not filters.empty
        if not required and not load_for_evidence_search:
            return eligible, None
        index = self._structured_index(collection, required=required)
        if index is None:
            return eligible, None
        if filters.empty:
            return eligible, index
        try:
            allowed_ids = set(
                index.filter_object_ids(
                    filters,
                    allowed_object_ids=(obj.id for obj in eligible),
                )
            )
        except StructuredFilterError as error:
            raise CollectionDataError(
                "STRUCTURED_FILTER_FAILED",
                "The verified catalogue filter could not be applied.",
                collectionId=collection.id,
                reason=type(error).__name__,
            ) from error
        return [obj for obj in eligible if obj.id in allowed_ids], index

    def _structured_candidate_pool(
        self, eligible: list[MuseumObject], filters: FilterSpec,
        excluded: list[set[str]],
    ) -> list[SearchResult]:
        """Retain a small exact-filter set as recall, not semantic approval.

        A query can express all its criteria in typed fields and contain no
        useful lexical subject. Requiring a second lexical hit then discards
        already verified field matches. Large broad filters still use ranked
        recall; this channel never arbitrarily truncates such a set.
        """
        if filters.empty or len(eligible) > min(self.hybrid_max_results, 60):
            return []
        return [
            SearchResult(
                obj=obj, score=0.01,
                retrieval_sources=("structured_filter",),
                field_scores=(("explicit_filter_match", 1.0),),
            )
            for obj in eligible
            if not any(
                tokens and tokens & _effective_searchable(_build_search_document(obj))
                for tokens in excluded
            )
        ]

    def _object_fts_lexical_search(
        self,
        query: QueryPlan,
        excluded: list[set[str]],
        eligible: list[MuseumObject],
        index: StructuredFilterIndex | None,
        *,
        fallback_results: list[SearchResult] | None = None,
    ) -> tuple[list[SearchResult], tuple[str, ...]]:
        """Recall object candidates with verified SQLite FTS, then re-score them.

        FTS is only a fast candidate generator. The maintained Python BM25
        implementation still applies exact-title, exclusion and all-anchor
        gates to the bounded candidate set. If the derivative is unavailable,
        or a strict lexical query would otherwise be refused, retrieval falls
        back to the complete fielded-BM25 scan rather than trading correctness
        for latency silently.
        """

        def full_fallback(warning: str) -> tuple[list[SearchResult], tuple[str, ...]]:
            if fallback_results is not None:
                return list(fallback_results), (warning,)
            return self._lexical_search(query, excluded, eligible), (warning,)

        if index is None or not index.object_search_available:
            return full_fallback("object_fts_unavailable")
        try:
            indexed_count = int(index.manifest.get("objectCount", -1))
            allowed_ids: Iterable[str] | None = (
                None
                if indexed_count == len(eligible)
                else (obj.id for obj in eligible)
            )
            hits = index.search_objects(
                sorted(query.scoring_tokens),
                top_k=500,
                allowed_object_ids=allowed_ids,
            )
        except StructuredFilterError as error:
            logger.warning("object-level BM25 unavailable: %s", error)
            return full_fallback("object_fts_failed")

        hit_scores = {hit.object_id: hit.score for hit in hits}
        candidate_objects = [obj for obj in eligible if obj.id in hit_scores]
        results = self._lexical_search(query, excluded, candidate_objects)
        if not results and (
            not query.dense_fallback_allowed
            or query.strict_anchor_groups
            or query.title_anchor_groups
        ):
            # A 500-row OR pool can omit the only record satisfying every
            # strict anchor when one query term is extremely common. Preserve
            # the original refusal boundary with an observable rare fallback.
            return full_fallback("object_fts_recall_fallback")
        return (
            [
                replace(
                    result,
                    field_scores=tuple(
                        (
                            *result.field_scores,
                            ("object_fts_bm25", hit_scores[result.obj.id]),
                        )
                    ),
                    retrieval_sources=tuple(
                        dict.fromkeys((*result.retrieval_sources, "bm25_object_fts"))
                    ),
                )
                for result in results
            ],
            (),
        )

    def _trace_search(
        self,
        *,
        question: str,
        collection: LoadedCollection,
        served: list[SearchResult],
        candidate: list[SearchResult],
        started: float,
        warnings: Sequence[str] = (),
        stage_latency_ms: Mapping[str, float] | None = None,
    ) -> None:
        if self.trace_writer is None:
            return
        try:
            self.trace_writer.write(
                question=question,
                collection=collection,
                serving_mode=self.rag_mode,
                served_results=served,
                candidate_results=candidate,
                elapsed_ms=(perf_counter() - started) * 1000.0,
                dense_status=self._dense_manager.status(collection),
                warnings=warnings,
                stage_latency_ms=stage_latency_ms,
            )
        except Exception as error:  # traces must never break visitor search
            logger.warning("retrieval trace write failed: %s", error)

    def search(
        self,
        agenda: AgendaInput,
        collection: LoadedCollection,
        *,
        deadline: float | None = None,
        filters: FilterSpec | None = None,
    ) -> list[SearchResult]:
        started = perf_counter()
        stage_latency_ms: dict[str, float] = {}
        filters = filters or FilterSpec()

        def check_deadline() -> None:
            if deadline is not None and perf_counter() >= deadline:
                raise CollectionDataError(
                    "RETRIEVAL_SEARCH_TIMEOUT",
                    "Collection retrieval exceeded its wall-clock budget.",
                )

        check_deadline()
        eligible = self.require_generation_ready(collection)
        filter_started = perf_counter()
        eligible, evidence_index = self._eligible_with_filters(
            collection,
            eligible,
            filters,
            load_for_evidence_search=self.rag_mode in {"hybrid", "shadow"},
        )
        stage_latency_ms["structured_filter"] = (
            perf_counter() - filter_started
        ) * 1000.0
        dense_signature = self._retrieval_cache_signature(
            collection,
            evidence_index,
        )
        cache_key = self._search_cache_key(
            agenda,
            collection,
            dense_signature,
            filters=filters,
        )
        with self._cache_lock:
            cached = self._search_result_cache.get(cache_key)
            if cached is not None:
                self._search_result_cache.move_to_end(cache_key)
                return list(cached)

        plan_started = perf_counter()
        query = _query_plan(agenda.question, collection.concept_aliases)
        excluded = [
            _query_plan(topic, collection.concept_aliases).anchor_tokens
            for topic in agenda.excluded_topics
        ]
        stage_latency_ms["deterministic_query_plan"] = (
            perf_counter() - plan_started
        ) * 1000.0
        structured_pool = self._structured_candidate_pool(eligible, filters, excluded)

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
        if not query.scoring_tokens and not query.browse_all and not structured_pool:
            return []

        lexical_warnings: tuple[str, ...] = ()
        baseline_lexical_results: list[SearchResult] | None = None
        if self.rag_mode in {"bm25", "shadow"}:
            lexical_started = perf_counter()
            baseline_lexical_results = self._lexical_search(query, excluded, eligible)
            stage_latency_ms["object_bm25"] = (
                perf_counter() - lexical_started
            ) * 1000.0

        if self.rag_mode in {"hybrid", "shadow"}:
            object_fts_started = perf_counter()
            lexical_results, lexical_warnings = self._object_fts_lexical_search(
                query,
                excluded,
                eligible,
                evidence_index,
                fallback_results=baseline_lexical_results,
            )
            stage_latency_ms["object_fts_bm25"] = (
                perf_counter() - object_fts_started
            ) * 1000.0
        else:
            lexical_results = list(baseline_lexical_results or [])
        if structured_pool:
            lexical_ids = {result.obj.id for result in lexical_results}
            lexical_results.extend(result for result in structured_pool if result.obj.id not in lexical_ids)
            if self.rag_mode == "shadow" and baseline_lexical_results is not None:
                baseline_ids = {result.obj.id for result in baseline_lexical_results}
                baseline_lexical_results.extend(result for result in structured_pool if result.obj.id not in baseline_ids)
        check_deadline()
        if self.rag_mode not in {"hybrid", "shadow"}:
            with self._cache_lock:
                self._remember_search(cache_key, lexical_results)
            self._trace_search(
                question=agenda.question,
                collection=collection,
                served=lexical_results,
                candidate=lexical_results,
                started=started,
                stage_latency_ms=stage_latency_ms,
            )
            return lexical_results
        if not lexical_results and not query.dense_fallback_allowed:
            # Unknown subjects in either language retain the lexical refusal
            # gate; avoid loading/scanning dense matrices only to discard the
            # nearest neighbours afterwards.
            with self._cache_lock:
                self._remember_search(cache_key, [])
            return []

        dense_load_started = perf_counter()
        dense_index = self._dense_manager.get(collection)
        stage_latency_ms["dense_index_load"] = (
            perf_counter() - dense_load_started
        ) * 1000.0
        if dense_index is None:
            # This path is intentionally a complete BM25 result, not a partial
            # approximation.  DenseIndexManager logs the concrete reason once.
            served_lexical = (
                list(baseline_lexical_results)
                if self.rag_mode == "shadow"
                and baseline_lexical_results is not None
                else lexical_results
            )
            with self._cache_lock:
                self._remember_search(cache_key, served_lexical)
            self._trace_search(
                question=agenda.question,
                collection=collection,
                served=served_lexical,
                candidate=lexical_results,
                started=started,
                warnings=tuple(dict.fromkeys((*lexical_warnings, "dense_unavailable"))),
                stage_latency_ms=stage_latency_ms,
            )
            return served_lexical

        try:
            results = self._hybrid_search(
                agenda.question,
                query,
                excluded,
                eligible,
                lexical_results,
                dense_index,
                evidence_index=evidence_index,
                deadline=deadline,
                stage_latency_ms=stage_latency_ms,
            )
        except CollectionDataError:
            raise
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
            self._trace_search(
                question=agenda.question,
                collection=collection,
                served=(
                    baseline_lexical_results
                    if self.rag_mode == "shadow"
                    and baseline_lexical_results is not None
                    else lexical_results
                ),
                candidate=lexical_results,
                started=started,
                warnings=tuple(
                    dict.fromkeys((*lexical_warnings, "dense_query_failed"))
                ),
                stage_latency_ms=stage_latency_ms,
            )
            return (
                list(baseline_lexical_results)
                if self.rag_mode == "shadow"
                and baseline_lexical_results is not None
                else lexical_results
            )
        self._dense_manager.mark_query_success(collection, dense_index)
        served = (
            list(baseline_lexical_results)
            if self.rag_mode == "shadow" and baseline_lexical_results is not None
            else results
        )
        rerank_failed = any(
            "qwen_rerank_fallback" in result.retrieval_sources
            for result in results
        )
        if not rerank_failed:
            with self._cache_lock:
                self._remember_search(cache_key, served)
        self._trace_search(
            question=agenda.question,
            collection=collection,
            served=served,
            candidate=results,
            started=started,
            warnings=tuple(
                dict.fromkeys(
                    (
                        *lexical_warnings,
                        *(("rerank_failed",) if rerank_failed else ()),
                    )
                )
            ),
            stage_latency_ms=stage_latency_ms,
        )
        return served

    def search_many(
        self,
        agendas: Sequence[AgendaInput],
        collection: LoadedCollection,
        *,
        deadline: float | None = None,
        atomic: bool = False,
        filters: FilterSpec | None = None,
    ) -> list[list[SearchResult]]:
        """Search several agendas while batching lexical and dense recall.

        Each query still runs through the same lexical candidate construction,
        precision gates, evidence reranker and result cache as :meth:`search`.
        Uncached queries share one catalogue traversal for fielded BM25 and one
        embedding-provider call. Duplicate requests in the same batch share
        both their query vector and final result.
        """

        agenda_list = list(agendas)
        if not agenda_list:
            return []
        filters = filters or FilterSpec()

        def check_deadline() -> None:
            if deadline is not None and perf_counter() >= deadline:
                raise CollectionDataError(
                    "RETRIEVAL_SEARCH_TIMEOUT",
                    "Batched collection retrieval exceeded its wall-clock budget.",
                )

        eligible = self.require_generation_ready(collection)
        eligible, evidence_index = self._eligible_with_filters(
            collection,
            eligible,
            filters,
            load_for_evidence_search=self.rag_mode in {"hybrid", "shadow"},
        )
        dense_signature = self._retrieval_cache_signature(
            collection,
            evidence_index,
        )
        outputs: list[list[SearchResult] | None] = [None] * len(agenda_list)
        pending: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()

        for position, agenda in enumerate(agenda_list):
            check_deadline()
            cache_key = self._search_cache_key(
                agenda,
                collection,
                dense_signature,
                atomic=atomic,
                filters=filters,
            )
            with self._cache_lock:
                cached = self._search_result_cache.get(cache_key)
                if cached is not None:
                    self._search_result_cache.move_to_end(cache_key)
                    outputs[position] = list(cached)
                    continue

            existing = pending.get(cache_key)
            if existing is not None:
                existing["positions"].append(position)
                continue

            query = _query_plan(agenda.question, collection.concept_aliases)
            if atomic:
                query = _atomic_catalogue_query_plan(agenda.question, query)
            excluded = [
                _query_plan(topic, collection.concept_aliases).anchor_tokens
                for topic in agenda.excluded_topics
            ]

            if query.browse_all and not excluded:
                results = [
                    SearchResult(
                        obj=obj,
                        score=(
                            1.1
                            if obj.evidence_depth == EvidenceDepth.FULL.value
                            else 1.0
                        ),
                        retrieval_sources=("browse_all",),
                    )
                    for obj in eligible
                ]
                with self._cache_lock:
                    self._remember_search(cache_key, results)
                outputs[position] = list(results)
                continue
            if not query.scoring_tokens and not query.browse_all:
                outputs[position] = []
                continue

            pending[cache_key] = {
                "agenda": agenda,
                "query": query,
                "excluded": excluded,
                "positions": [position],
                "started": perf_counter(),
            }

        if not pending:
            return [list(result or []) for result in outputs]

        pending_items = list(pending.items())
        lexical_eligible = eligible
        atomic_dense_prefilter = False
        atomic_exact_prefilter = False
        atomic_dense_index: Any | None = None
        atomic_query_vectors: list[Any] = []
        if atomic and self.rag_mode in {"hybrid", "shadow"}:
            dense_index = self._dense_manager.get(collection)
            if dense_index is not None:
                try:
                    check_deadline()
                    embed_many = getattr(dense_index, "embed_queries", None)
                    if callable(embed_many):
                        query_vectors = _call_with_optional_deadline(
                            embed_many,
                            [item[1]["agenda"].question for item in pending_items],
                            deadline=deadline,
                        )
                    else:
                        query_vectors = [
                            _call_with_optional_deadline(
                                dense_index.embed_query,
                                item[1]["agenda"].question,
                                deadline=deadline,
                            )
                            for item in pending_items
                        ]
                    atomic_dense_index = dense_index
                    atomic_query_vectors = list(query_vectors)
                    check_deadline()
                    candidate_top_k = max(
                        self.dense_top_k,
                        min(self.hybrid_max_results, 400),
                    )
                    eligible_ids = {obj.id for obj in eligible}
                    candidate_ids: set[str] = set()
                    for query_vector in query_vectors:
                        candidate_ids.update(
                            hit.object_id
                            for hit in dense_index.search_vector(
                                query_vector,
                                top_k=candidate_top_k,
                                allowed_object_ids=eligible_ids,
                            )
                        )
                        candidate_ids.update(
                            hit.object_id
                            for hit in dense_index.search_evidence_vector(
                                query_vector,
                                top_k=candidate_top_k * 2,
                            )
                            if hit.object_id in eligible_ids
                        )
                    check_deadline()
                    if candidate_ids:
                        lexical_eligible = [
                            obj for obj in eligible if obj.id in candidate_ids
                        ]
                        atomic_dense_prefilter = True
                        self._dense_manager.mark_query_success(
                            collection,
                            dense_index,
                        )
                except CollectionDataError:
                    raise
                except Exception as error:
                    logger.warning(
                        "atomic semantic prefilter unavailable; scanning full "
                        "catalogue (%s: %s)",
                        type(error).__name__,
                        error,
                    )
                    self._dense_manager.mark_query_failure(collection, error)

        if atomic:
            # Embeddings can rank an exact but rare catalogue phrase very low
            # (for example a literal "tool marks" sentence). Scan raw source
            # text with compiled anchor-group expressions and union those IDs
            # with the semantic prefilter. This avoids constructing full BM25
            # documents for all 17k objects while preserving exact recall.
            # Rare exact phrases need complete rescue, while a broad exact
            # axis should not expand the semantic Top-K back into thousands of
            # objects. Once an axis exceeds this cap it is represented by the
            # dense prefilter only.
            max_exact_per_query = max(250, self.dense_top_k)
            exact_sets: list[set[str]] = [set() for _ in pending_items]
            exact_overflow = [False for _ in pending_items]
            group_patterns: list[list[re.Pattern[str]]] = []
            for _, item in pending_items:
                query: QueryPlan = item["query"]
                patterns: list[re.Pattern[str]] = []
                groups = query.strict_anchor_groups or query.anchor_groups
                # Four or more exact legs are already highly selective after
                # semantic Top-K and make a full-catalogue regex pass costly.
                # The rescue scan is for short rare phrases such as "tool
                # marks" that embeddings can rank unexpectedly low. Three-leg
                # queries remain eligible; the child deadline below prevents
                # their rescue scan from starving lexical AND scoring.
                if len(groups) > 3:
                    group_patterns.append(patterns)
                    continue
                for group in groups:
                    alternatives: list[str] = []
                    for token in sorted(group, key=lambda value: (-len(value), value)):
                        escaped = re.escape(token)
                        if re.search(r"[a-z0-9]", token, re.IGNORECASE):
                            alternatives.append(
                                rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
                            )
                        else:
                            alternatives.append(escaped)
                    if alternatives:
                        patterns.append(
                            re.compile(
                                "(?:" + "|".join(alternatives) + ")",
                                re.IGNORECASE,
                            )
                        )
                group_patterns.append(patterns)

            exact_scan_complete = True
            exact_rescue_deadline: float | None = None
            if deadline is not None:
                exact_remaining = max(0.0, deadline - perf_counter())
                lexical_reserve = min(
                    2.5,
                    max(0.75, exact_remaining * 0.40),
                )
                exact_rescue_deadline = deadline - lexical_reserve

            for obj in eligible:
                if not any(
                    patterns and not exact_overflow[index]
                    for index, patterns in enumerate(group_patterns)
                ):
                    break
                if (
                    exact_rescue_deadline is not None
                    and perf_counter() >= exact_rescue_deadline
                ):
                    # Exact rescue is an optional recall aid over the dense
                    # atomic pool. Preserve time for the mandatory lexical AND
                    # scoring instead of exhausting the shared deadline here.
                    exact_scan_complete = False
                    break
                raw_text = " ".join(
                    filter(
                        None,
                        (
                            obj.title,
                            obj.title_original,
                            obj.creator,
                            obj.culture,
                            obj.culture_display,
                            obj.place,
                            obj.date,
                            obj.material,
                            obj.type,
                            obj.classification,
                            obj.department,
                            *obj.routing_domain_ids,
                            *obj.culture_pack_ids,
                            *obj.tags,
                            *obj.relation_facets,
                            obj.description,
                            *(chunk.text for chunk in obj.evidence),
                        ),
                    )
                )
                for index, patterns in enumerate(group_patterns):
                    if exact_overflow[index] or not patterns:
                        continue
                    if all(pattern.search(raw_text) for pattern in patterns):
                        exact_sets[index].add(obj.id)
                        if len(exact_sets[index]) > max_exact_per_query:
                            exact_sets[index].clear()
                            exact_overflow[index] = True

            exact_ids = (
                set().union(
                    *(
                        ids
                        for ids, overflow in zip(
                            exact_sets,
                            exact_overflow,
                            strict=True,
                        )
                        if not overflow
                    )
                )
                if exact_scan_complete
                else set()
            )
            if exact_ids:
                if lexical_eligible is eligible:
                    candidate_ids = set(exact_ids)
                else:
                    candidate_ids = {obj.id for obj in lexical_eligible}
                    candidate_ids.update(exact_ids)
                lexical_eligible = [
                    obj for obj in eligible if obj.id in candidate_ids
                ]
                atomic_exact_prefilter = True

        lexical_batches = self._lexical_search_many(
            [
                (item["query"], item["excluded"])
                for _, item in pending_items
            ],
            lexical_eligible,
            deadline=deadline,
        )
        shadow_served_batches = [list(batch) for batch in lexical_batches]
        if (
            self.rag_mode == "shadow"
            and atomic
            and (atomic_dense_prefilter or atomic_exact_prefilter)
        ):
            # Shadow evaluation must compare against the true full-catalogue
            # BM25 baseline, not a baseline already narrowed by dense recall.
            shadow_served_batches = self._lexical_search_many(
                [
                    (item["query"], item["excluded"])
                    for _, item in pending_items
                ],
                eligible,
                deadline=deadline,
            )
        if (
            atomic
            and atomic_dense_index is not None
            and len(atomic_query_vectors) == len(pending_items)
        ):
            # An LLM planner can occasionally return an over-specified phrase
            # (for example four catalogue facets joined as one query). Atomic
            # lexical AND is the preferred precision path, but an empty or
            # singleton batch must not erase semantically direct records. Reuse
            # the already-computed query vector for a bounded evidence-backed
            # dense relaxation; every survivor still passes both source-bound
            # LLM audit stages before it can enter an exhibition.
            relaxed_batches: list[list[SearchResult]] = []
            for ((_, item), strict_results, query_vector) in zip(
                pending_items,
                lexical_batches,
                atomic_query_vectors,
                strict=True,
            ):
                if (
                    deadline is not None
                    and perf_counter() >= deadline - 0.25
                ):
                    relaxed_batches.append(strict_results)
                    continue
                try:
                    relaxed_query = _query_plan(
                        item["agenda"].question,
                        collection.concept_aliases,
                    )
                    semantic_results = self._hybrid_search(
                        item["agenda"].question,
                        relaxed_query,
                        item["excluded"],
                        eligible,
                        [],
                        atomic_dense_index,
                        query_vector=query_vector,
                        evidence_index=evidence_index,
                        deadline=deadline,
                        allow_rerank=False,
                    )
                except CollectionDataError:
                    raise
                except Exception as error:
                    logger.warning(
                        "atomic dense relaxation unavailable for %s (%s: %s)",
                        collection.id,
                        type(error).__name__,
                        error,
                    )
                    semantic_results = []
                annotated_semantic = [
                    replace(
                        result,
                        retrieval_sources=tuple(
                            dict.fromkeys(
                                (
                                    *result.retrieval_sources,
                                    "dense_atomic_relaxation",
                                )
                            )
                        ),
                    )
                    for result in semantic_results
                ]
                # Interleave both recall channels so a query-axis reserve sees
                # one semantic and one exact candidate before either channel's
                # long tail. The later evidence audit, not this ordering,
                # grants admission.
                merged: list[SearchResult] = []
                seen: set[str] = set()
                for offset in range(
                    max(len(annotated_semantic), len(strict_results))
                ):
                    for channel in (annotated_semantic, strict_results):
                        if offset >= len(channel):
                            continue
                        candidate = channel[offset]
                        if candidate.obj.id in seen:
                            continue
                        merged.append(candidate)
                        seen.add(candidate.obj.id)
                        if len(merged) >= self.hybrid_max_results:
                            break
                    if len(merged) >= self.hybrid_max_results:
                        break
                relaxed_batches.append(merged)
            lexical_batches = relaxed_batches
        if atomic_dense_prefilter or atomic_exact_prefilter:
            prefilter_sources = tuple(
                source
                for enabled, source in (
                    (atomic_dense_prefilter, "dense_atomic_prefilter"),
                    (atomic_exact_prefilter, "exact_atomic_prefilter"),
                )
                if enabled
            )
            lexical_batches = [
                [
                    replace(
                        result,
                        retrieval_sources=tuple(
                            dict.fromkeys(
                                (
                                    *result.retrieval_sources,
                                    *prefilter_sources,
                                )
                            )
                        ),
                    )
                    for result in batch
                ]
                for batch in lexical_batches
            ]
        hybrid_pending: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        for batch_index, ((cache_key, item), lexical_results) in enumerate(zip(
            pending_items,
            lexical_batches,
            strict=True,
        )):
            item["lexical_results"] = lexical_results
            item["served_lexical_results"] = shadow_served_batches[batch_index]
            # The visitor's original question already receives hybrid recall.
            # Planner-generated atomic queries are a precision complement:
            # every content leg is an explicit catalogue constraint and every
            # returned object is still checked by the source-ID LLM audit.
            # Running dense nearest-neighbour search for these expressions is
            # both redundant and harmful: it can reintroduce objects that do
            # not satisfy the atomic legs, while spending most of the bounded
            # pre-audit budget on a second embedding pass.
            if self.rag_mode == "bm25" or atomic:
                served_results = (
                    item["served_lexical_results"]
                    if self.rag_mode == "shadow"
                    else lexical_results
                )
                with self._cache_lock:
                    self._remember_search(cache_key, served_results)
                for position in item["positions"]:
                    outputs[position] = list(served_results)
                if self.rag_mode == "shadow":
                    self._trace_search(
                        question=item["agenda"].question,
                        collection=collection,
                        served=served_results,
                        candidate=lexical_results,
                        started=item["started"],
                    )
                continue
            if not lexical_results and not item["query"].dense_fallback_allowed:
                with self._cache_lock:
                    self._remember_search(cache_key, [])
                for position in item["positions"]:
                    outputs[position] = []
                continue
            hybrid_pending[cache_key] = item

        pending = hybrid_pending
        if not pending:
            return [list(result or []) for result in outputs]

        dense_index = self._dense_manager.get(collection)
        if dense_index is None:
            for cache_key, item in pending.items():
                results = item["served_lexical_results"]
                with self._cache_lock:
                    self._remember_search(cache_key, results)
                for position in item["positions"]:
                    outputs[position] = list(results)
                self._trace_search(
                    question=item["agenda"].question,
                    collection=collection,
                    served=results,
                    candidate=item["lexical_results"],
                    started=item["started"],
                    warnings=("dense_unavailable",),
                )
            return [list(result or []) for result in outputs]

        items = list(pending.items())
        check_deadline()
        try:
            query_vectors = _call_with_optional_deadline(
                dense_index.embed_queries,
                [item[1]["agenda"].question for item in items],
                deadline=deadline,
            )
            check_deadline()
            if len(query_vectors) != len(items):
                raise RuntimeError(
                    "dense index returned an unexpected number of query vectors"
                )
        except CollectionDataError:
            # A cooperative deadline is part of the public search contract,
            # not a dense-runtime failure. Never turn it into a successful
            # BM25 response after the caller's budget has already expired.
            raise
        except Exception as error:
            logger.warning(
                "batched hybrid RAG query degraded to BM25 for %s: %s",
                collection.id,
                error,
            )
            self._dense_manager.mark_query_failure(collection, error)
            for _, item in items:
                results = item["lexical_results"]
                for position in item["positions"]:
                    outputs[position] = list(results)
            # Match the single-query failure path: transient dense failures are
            # deliberately not cached, so a later request retries the encoder.
            return [list(result or []) for result in outputs]

        failures: list[Exception] = []
        for (cache_key, item), query_vector in zip(items, query_vectors, strict=True):
            check_deadline()
            try:
                results = self._hybrid_search(
                    item["agenda"].question,
                    item["query"],
                    item["excluded"],
                    eligible,
                    item["lexical_results"],
                    dense_index,
                    query_vector=query_vector,
                    evidence_index=evidence_index,
                    deadline=deadline,
                )
            except CollectionDataError:
                raise
            except Exception as error:
                logger.warning(
                    "batched hybrid RAG query degraded to BM25 for %s: %s",
                    collection.id,
                    error,
                )
                failures.append(error)
                results = item["served_lexical_results"]
                self._trace_search(
                    question=item["agenda"].question,
                    collection=collection,
                    served=results,
                    candidate=item["lexical_results"],
                    started=item["started"],
                    warnings=("dense_query_failed",),
                )
            else:
                rerank_failed = any(
                    "qwen_rerank_fallback" in result.retrieval_sources
                    for result in results
                )
                served_results = (
                    item["served_lexical_results"]
                    if self.rag_mode == "shadow"
                    else results
                )
                if not rerank_failed:
                    with self._cache_lock:
                        self._remember_search(cache_key, served_results)
                self._trace_search(
                    question=item["agenda"].question,
                    collection=collection,
                    served=served_results,
                    candidate=results,
                    started=item["started"],
                    warnings=("rerank_failed",) if rerank_failed else (),
                )
                results = served_results
            for position in item["positions"]:
                outputs[position] = list(results)

        if failures:
            self._dense_manager.mark_query_failure(collection, failures[0])
        else:
            self._dense_manager.mark_query_success(collection, dense_index)
        return [list(result or []) for result in outputs]

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
            if query.title_anchor_groups and any(
                not group & (document.title - document.contextual_anchor_suppressions)
                for group in query.title_anchor_groups
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
                if (
                    chunk.source_kind != "institution_provenance"
                    or query.allow_provenance_evidence
                )
                and query.anchor_tokens & _tokens(chunk.text)
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

    @staticmethod
    def _lexical_search_many(
        requests: Sequence[tuple[QueryPlan, list[set[str]]]],
        eligible: list[MuseumObject],
        *,
        deadline: float | None = None,
    ) -> list[list[SearchResult]]:
        """Run independent fielded-BM25 queries over one document traversal.

        Query-specific document frequencies, excluded topics, anchor gates and
        compact candidate documents deliberately remain separate. The only
        shared work is building each full :class:`SearchDocument`, which is the
        dominant cost for agentic expansion queries and previously happened up
        to three times in :meth:`search_many`.
        """

        if not requests:
            return []

        states: list[dict[str, Any]] = [
            {
                "query": query,
                "excluded": excluded,
                "document_frequency": Counter(),
                "field_totals": Counter(),
                "candidate_documents": [],
                "searched_document_count": 0,
            }
            for query, excluded in requests
        ]

        for obj in eligible:
            if deadline is not None and perf_counter() >= deadline:
                raise CollectionDataError(
                    "RETRIEVAL_SEARCH_TIMEOUT",
                    "Batched collection retrieval exceeded its wall-clock budget.",
                )
            # Keep the same memory boundary as the single-query path: the full
            # tokenised document exists for one object only. Each query retains
            # only frequencies for its own scoring terms.
            document = _build_search_document(obj)
            effective_searchable = _effective_searchable(document)
            for state in states:
                query: QueryPlan = state["query"]
                excluded: list[set[str]] = state["excluded"]
                if any(
                    topic_tokens and topic_tokens & effective_searchable
                    for topic_tokens in excluded
                ):
                    continue
                state["searched_document_count"] += 1
                state["field_totals"].update(document.field_lengths)
                state["document_frequency"].update(
                    query.scoring_tokens & effective_searchable
                )
                if query.browse_all:
                    state["candidate_documents"].append(
                        (
                            obj,
                            QueryDocument(
                                term_frequencies={
                                    field: Counter()
                                    for field in document.term_frequencies
                                },
                                field_lengths=document.field_lengths,
                                matched_anchor_terms=(),
                            ),
                        )
                    )
                    continue
                if query.anchor_groups and any(
                    not group & effective_searchable
                    for group in query.anchor_groups
                ):
                    continue
                if query.title_anchor_groups and any(
                    not group
                    & (
                        document.title
                        - document.contextual_anchor_suppressions
                    )
                    for group in query.title_anchor_groups
                ):
                    continue
                matched_anchor_terms = tuple(
                    sorted(query.anchor_tokens & effective_searchable)
                )
                state["candidate_documents"].append(
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

        batches: list[list[SearchResult]] = []
        for state in states:
            if deadline is not None and perf_counter() >= deadline:
                raise CollectionDataError(
                    "RETRIEVAL_SEARCH_TIMEOUT",
                    "Batched collection retrieval exceeded its wall-clock budget.",
                )
            query = state["query"]
            searched_document_count = state["searched_document_count"]
            index = SearchIndex(
                document_frequency=state["document_frequency"],
                average_field_lengths={
                    field: total / searched_document_count
                    if searched_document_count
                    else 0.0
                    for field, total in state["field_totals"].items()
                },
                document_count=searched_document_count,
            )
            results: list[SearchResult] = []
            for candidate_index, (obj, document) in enumerate(
                state["candidate_documents"]
            ):
                if (
                    deadline is not None
                    and candidate_index % 128 == 0
                    and perf_counter() >= deadline
                ):
                    raise CollectionDataError(
                        "RETRIEVAL_SEARCH_TIMEOUT",
                        "Batched collection retrieval exceeded its wall-clock budget.",
                    )
                if query.browse_all:
                    results.append(
                        SearchResult(
                            obj=obj,
                            score=(
                                1.1
                                if obj.evidence_depth == EvidenceDepth.FULL.value
                                else 1.0
                            ),
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
                    if (
                        chunk.source_kind != "institution_provenance"
                        or query.allow_provenance_evidence
                    )
                    and query.anchor_tokens & _tokens(chunk.text)
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
            if deadline is not None and perf_counter() >= deadline:
                raise CollectionDataError(
                    "RETRIEVAL_SEARCH_TIMEOUT",
                    "Batched collection retrieval exceeded its wall-clock budget.",
                )
            results.sort(key=lambda result: (-result.score, result.obj.id))
            batches.append(results)
        return batches

    def _hybrid_search(
        self,
        question: str,
        query: QueryPlan,
        excluded: list[set[str]],
        eligible: list[MuseumObject],
        lexical_results: list[SearchResult],
        dense_index: Any,
        *,
        query_vector: Any | None = None,
        evidence_index: StructuredFilterIndex | None = None,
        deadline: float | None = None,
        allow_rerank: bool = True,
        stage_latency_ms: dict[str, float] | None = None,
    ) -> list[SearchResult]:
        """Fuse BM25 and dense recall, then rerank against evidence chunks.

        BM25 candidates have already passed every lexical anchor group. Dense
        recall is open-vocabulary: reviewed aliases and exact-title rules remain
        precision guards for known ambiguous concepts, but they are no longer a
        whitelist of subjects the visitor is allowed to ask about. A new subject
        may enter through object and evidence embeddings when it clears the
        stricter open-query floors below.
        """

        def check_deadline() -> None:
            if deadline is not None and perf_counter() >= deadline:
                raise CollectionDataError(
                    "RETRIEVAL_SEARCH_TIMEOUT",
                    "Collection retrieval exceeded its wall-clock budget.",
                )

        check_deadline()
        stages = stage_latency_ms if stage_latency_ms is not None else {}
        eligible_by_id = {obj.id: obj for obj in eligible}
        dense_floor = (
            max(self.dense_min_score, OPEN_QUERY_DENSE_MIN_SCORE)
            if query.open_semantic_query
            else self.dense_min_score
        )
        evidence_floor = (
            max(self.evidence_min_score, OPEN_QUERY_EVIDENCE_MIN_SCORE)
            if query.open_semantic_query
            else self.evidence_min_score
        )
        if query_vector is None:
            embedding_started = perf_counter()
            query_vector = _call_with_optional_deadline(
                dense_index.embed_query,
                question,
                deadline=deadline,
            )
            stages["query_embedding"] = (
                perf_counter() - embedding_started
            ) * 1000.0
        check_deadline()
        dense_recall_started = perf_counter()
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
        stages["dense_vector_recall"] = (
            perf_counter() - dense_recall_started
        ) * 1000.0
        check_deadline()

        # Evidence vectors deliberately carry a little object context to make
        # catalogue fragments retrievable.  That context must never turn an
        # unrelated fragment (for example an acquisition credit) into the
        # query evidence cited by the curator. Provenance rows are eligible only
        # when the visitor explicitly asks about source history; even then the
        # later audit must keep documented acquisition separate from consent or
        # legal-title claims. Reviewed concrete concepts additionally retain a
        # lexical check; open-vocabulary questions may use an eligible excerpt
        # on semantic similarity alone. This query-time guard does not change
        # the frozen embedding recipe or require rebuilding the index.
        query_evidence_tokens = query.anchor_tokens | query.scoring_tokens

        def raw_evidence_matches(obj: MuseumObject, evidence_id: str) -> bool:
            chunk = next(
                (item for item in obj.evidence if item.id == evidence_id),
                None,
            )
            if chunk is None or (
                chunk.source_kind == "institution_provenance"
                and not query.allow_provenance_evidence
            ):
                return False
            lexical_match = bool(
                query_evidence_tokens
                & _tokens(" ".join(filter(None, (chunk.text, chunk.supports))))
            )
            if query.strict_anchor_groups or query.title_anchor_groups:
                return lexical_match
            return lexical_match or query.dense_fallback_allowed

        object_dense_scores: dict[str, float] = {}
        object_dense_rank: dict[str, int] = {}
        for rank, hit in enumerate(object_dense_hits, start=1):
            if hit.score >= dense_floor and hit.object_id in eligible_by_id:
                object_dense_scores[hit.object_id] = hit.score
                object_dense_rank[hit.object_id] = rank
        evidence_recall_scores: dict[str, float] = {}
        evidence_recall_rank: dict[str, int] = {}
        evidence_recall_ids: dict[str, list[str]] = {}
        for rank, hit in enumerate(evidence_dense_hits, start=1):
            if hit.score < evidence_floor or hit.object_id not in eligible_by_id:
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
            if query.title_anchor_groups and any(
                not group & (document.title - document.contextual_anchor_suppressions)
                for group in query.title_anchor_groups
            ):
                dense_object_ids.discard(object_id)
                continue
            dense_documents[object_id] = document

        # Object-level BM25 and dense recall are not enough when the decisive
        # catalogue phrase lives only inside one institution evidence row.
        # Query the verified FTS5 derivative as an independent lexical channel,
        # then re-apply the same provenance, exclusion and anchor gates used by
        # dense recall before admitting any new object ID.
        evidence_bm25_scores: dict[str, float] = {}
        evidence_bm25_rank: dict[str, int] = {}
        evidence_bm25_ids: dict[str, list[str]] = {}
        if evidence_index is not None:
            evidence_bm25_started = perf_counter()
            try:
                indexed_count = int(
                    evidence_index.manifest.get("objectCount", -1)
                )
                evidence_lexical_hits = evidence_index.search_evidence(
                    sorted(query.scoring_tokens),
                    top_k=self.evidence_bm25_top_k,
                    allowed_object_ids=(
                        None
                        if indexed_count == len(eligible_by_id)
                        else eligible_by_id
                    ),
                )
            except StructuredFilterError as error:
                logger.warning(
                    "evidence-level BM25 unavailable for %s: %s",
                    getattr(evidence_index, "manifest", {}).get(
                        "collectionId", "collection"
                    ),
                    error,
                )
                evidence_lexical_hits = []
            stages["evidence_bm25"] = (
                perf_counter() - evidence_bm25_started
            ) * 1000.0
            check_deadline()
            for hit in evidence_lexical_hits:
                obj = eligible_by_id.get(hit.object_id)
                if obj is None or not raw_evidence_matches(obj, hit.evidence_id):
                    continue
                document = dense_documents.get(hit.object_id)
                if document is None:
                    document = _build_search_document(obj)
                effective_searchable = _effective_searchable(document)
                if any(
                    topic_tokens and topic_tokens & effective_searchable
                    for topic_tokens in excluded
                ):
                    continue
                if query.strict_anchor_groups and any(
                    not group & effective_searchable
                    for group in query.strict_anchor_groups
                ):
                    continue
                if query.title_anchor_groups and any(
                    not group
                    & (document.title - document.contextual_anchor_suppressions)
                    for group in query.title_anchor_groups
                ):
                    continue
                dense_documents.setdefault(hit.object_id, document)
                evidence_bm25_scores[hit.object_id] = max(
                    evidence_bm25_scores.get(hit.object_id, 0.0),
                    hit.score,
                )
                evidence_bm25_rank.setdefault(
                    hit.object_id,
                    len(evidence_bm25_rank) + 1,
                )
                evidence_bm25_ids.setdefault(hit.object_id, []).append(
                    hit.evidence_id
                )

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
        candidate_ids.update(evidence_bm25_scores)
        if query.dense_fallback_allowed:
            candidate_ids.update(dense_scores)
        if not candidate_ids:
            return []

        # Ask for all of the small per-object evidence set.  The highest dense
        # row can have inherited relevance from object context; inspecting more
        # than the previous top three lets a genuinely matching description
        # survive behind an acquisition/classification fragment.
        evidence_rescore_started = perf_counter()
        evidence_hits = dense_index.evidence_hits(
            query_vector,
            candidate_ids,
            max_evidence_ids=5,
        )
        stages["dense_evidence_rescore"] = (
            perf_counter() - evidence_rescore_started
        ) * 1000.0
        # Dense-only objects need a non-provenance evidence row that clears the
        # semantic floor. BM25 objects retain their lexical hard gate even when
        # catalogue prose is very short.
        qualified_evidence_by_object: dict[str, tuple[tuple[str, float], ...]] = {}
        for object_id, evidence in evidence_hits.items():
            obj = eligible_by_id.get(object_id)
            if obj is None:
                continue
            scores = (
                evidence.evidence_scores
                if evidence.evidence_scores
                else tuple(
                    (evidence_id, evidence.score)
                    for evidence_id in evidence.evidence_ids
                )
            )
            qualified_evidence_by_object[object_id] = tuple(
                (evidence_id, score)
                for evidence_id, score in scores
                if score >= evidence_floor
                and raw_evidence_matches(obj, evidence_id)
            )
        candidate_ids = {
            object_id
            for object_id in candidate_ids
            if object_id in lexical_by_id
            or object_id in evidence_bm25_scores
            or (
                object_id in dense_scores
                and qualified_evidence_by_object.get(object_id)
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
            + (
                (1.15 / (self.rrf_k + evidence_bm25_rank[object_id]))
                if object_id in evidence_bm25_rank
                else 0.0
            )
            for object_id in candidate_ids
        }
        top_rrf = max(rrf_scores.values(), default=1.0)
        top_bm25 = max((item.score for item in lexical_pool), default=1.0)
        top_evidence_bm25 = max(evidence_bm25_scores.values(), default=1.0)

        def floor_normalise(score: float | None, floor: float) -> float:
            if score is None or score <= floor:
                return 0.0
            return min(1.0, (score - floor) / max(1.0 - floor, 1e-6))

        results: list[SearchResult] = []
        for object_id in candidate_ids:
            lexical = lexical_by_id.get(object_id)
            obj = lexical.obj if lexical else eligible_by_id[object_id]
            evidence = evidence_hits.get(object_id)
            qualified_dense_evidence = qualified_evidence_by_object.get(
                object_id, ()
            )
            qualified_evidence_score = max(
                (score for _, score in qualified_dense_evidence),
                default=None,
            )
            evidence_supported = qualified_evidence_score is not None
            dense_score = dense_scores.get(object_id)
            bm25_signal = lexical.score / top_bm25 if lexical else 0.0
            evidence_bm25_signal = (
                evidence_bm25_scores[object_id] / top_evidence_bm25
                if object_id in evidence_bm25_scores and top_evidence_bm25
                else 0.0
            )
            dense_signal = floor_normalise(dense_score, dense_floor)
            evidence_signal = floor_normalise(
                qualified_evidence_score,
                evidence_floor,
            )
            rrf_signal = rrf_scores[object_id] / top_rrf if top_rrf else 0.0
            # RRF supplies the main rank consensus; direct evidence similarity
            # and the stronger original channel decide ties and weak tails.
            score = 100.0 * (
                0.55 * rrf_signal
                + 0.25 * evidence_signal
                + 0.20 * max(
                    bm25_signal,
                    evidence_bm25_signal,
                    dense_signal,
                )
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
            evidence_ids.extend(evidence_bm25_ids.get(object_id, ()))
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
            sources = list(lexical.retrieval_sources if lexical else ())
            if object_id in evidence_bm25_rank:
                sources.append("bm25_evidence")
            if object_id in object_dense_rank and object_id in dense_object_ids:
                sources.append("dense_object")
            if object_id in evidence_recall_rank and object_id in dense_object_ids:
                sources.append("dense_evidence_recall")
            if query.open_semantic_query and (
                object_id in object_dense_rank or object_id in evidence_recall_rank
            ):
                sources.append("dense_open_query")
            if evidence_supported:
                sources.append("evidence_rerank")
            fields = list(lexical.field_scores if lexical else ())
            if object_id in evidence_bm25_scores:
                fields.append(
                    (
                        "evidence_bm25",
                        round(evidence_bm25_scores[object_id], 6),
                    )
                )
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
        results = results[: self.hybrid_max_results]
        if not allow_rerank:
            return results
        rerank_started = perf_counter()
        reranked = self._apply_semantic_reranker(
            question,
            results,
            deadline=deadline,
        )
        stages["qwen_rerank"] = (perf_counter() - rerank_started) * 1000.0
        return reranked

    @staticmethod
    def _rerank_document(result: SearchResult, *, limit: int = 360) -> str:
        """Build a short evidence-first rerank record under provider limits."""

        obj = result.obj
        evidence_by_id = {chunk.id: chunk for chunk in obj.evidence}
        evidence = [
            evidence_by_id[evidence_id].text
            for evidence_id in result.matched_evidence_ids
            if evidence_id in evidence_by_id
        ][:2]
        if not evidence:
            evidence = [chunk.text for chunk in obj.evidence[:2]]
        text = " | ".join(
            part
            for part in (
                f"title: {obj.title}",
                f"creator: {obj.creator or obj.maker}",
                f"date: {obj.date}",
                f"culture: {obj.culture_display or obj.culture}",
                f"type: {obj.type or obj.classification}",
                f"material: {obj.material or obj.medium}",
                "evidence: " + " ".join(evidence),
            )
            if part.split(":", 1)[-1].strip()
        )
        # Sixty candidates plus a repeated query must remain below Qwen's
        # whole-request context budget. Museum evidence is expanded only after
        # rerank, in the source-bound DeepSeek audit.
        return re.sub(r"\s+", " ", text).strip()[:limit]

    def _apply_semantic_reranker(
        self,
        question: str,
        results: list[SearchResult],
        *,
        deadline: float | None = None,
    ) -> list[SearchResult]:
        """Apply Qwen rerank between RRF recall and evidence audit.

        A reranker is an ordering aid, never an evidence authority. Failure is
        observable on every candidate and preserves the RRF order so the later
        source-quote audit still controls answerability.
        """

        if self.reranker is None or not results:
            return results
        if deadline is not None and deadline - perf_counter() < 0.5:
            return [
                replace(
                    result,
                    retrieval_sources=tuple(
                        dict.fromkeys(
                            (*result.retrieval_sources, "qwen_rerank_fallback")
                        )
                    ),
                )
                for result in results
            ]

        # Model Studio accounts for the query once per candidate and has a
        # conservative ~30k request budget. Dynamically shrink the window and
        # evidence record instead of letting perfectly valid long questions
        # fail a fixed 60 x 360-character request.
        request_budget = max(
            0,
            28_500 - len(self.rerank_instruct) - 128,
        )
        candidate_count = min(self.rerank_candidate_count, len(results))
        while candidate_count > 0:
            per_document = (
                request_budget - len(question) * candidate_count
            ) // candidate_count
            if per_document >= 120:
                break
            candidate_count -= 1
        if candidate_count <= 0:
            logger.warning("Qwen rerank skipped because the request cannot fit safely")
            return [
                replace(
                    result,
                    retrieval_sources=tuple(
                        dict.fromkeys(
                            (*result.retrieval_sources, "qwen_rerank_fallback")
                        )
                    ),
                )
                for result in results
            ]
        window = results[:candidate_count]
        document_limit = min(360, max(120, per_document))
        documents = [
            self._rerank_document(result, limit=document_limit)
            for result in window
        ]
        try:
            kwargs: dict[str, Any] = {
                "top_n": min(self.rerank_top_n, candidate_count),
                "instruct": self.rerank_instruct or None,
            }
            # The production provider accepts an absolute cooperative
            # deadline. Test doubles and older adapters remain compatible.
            batch = _call_with_optional_deadline(
                self.reranker.rerank,
                question,
                documents,
                deadline=deadline,
                **kwargs,
            )
            decisions = list(batch.results)
            if not decisions:
                raise ValueError("reranker returned no candidates")
        except Exception as error:
            logger.warning("Qwen rerank unavailable; preserving RRF order: %s", error)
            return [
                replace(
                    result,
                    retrieval_sources=tuple(
                        dict.fromkeys(
                            (*result.retrieval_sources, "qwen_rerank_fallback")
                        )
                    ),
                )
                for result in results
            ]

        top_original_score = max((result.score for result in window), default=1.0)
        reranked: list[SearchResult] = []
        selected_indices: set[int] = set()
        for decision in decisions:
            index = int(decision.index)
            if index in selected_indices or not 0 <= index < len(window):
                continue
            selected_indices.add(index)
            base = window[index]
            relevance = max(0.0, min(1.0, float(decision.relevance_score)))
            prior = base.score / top_original_score if top_original_score else 0.0
            reranked.append(
                replace(
                    base,
                    score=100.0 * (0.85 * relevance + 0.15 * prior),
                    field_scores=tuple(
                        (
                            *base.field_scores,
                            ("pre_qwen_rerank", round(base.score, 6)),
                            ("qwen_rerank", round(relevance, 6)),
                        )
                    ),
                    retrieval_sources=tuple(
                        dict.fromkeys((*base.retrieval_sources, "qwen3_rerank"))
                    ),
                )
            )
        if not reranked:
            logger.warning("Qwen rerank returned only invalid candidate indexes")
            return results

        # Keep the unselected tail available for cross-cultural quota sampling
        # and bounded query expansion. The DeepSeek audit sees only its sampled
        # top window and still requires exact source IDs and quotes.
        selected_ids = {result.obj.id for result in reranked}
        return [
            *reranked,
            *[result for result in results if result.obj.id not in selected_ids],
        ][: self.hybrid_max_results]

    def rerank_results(
        self,
        question: str,
        results: list[SearchResult],
        *,
        deadline: float | None = None,
    ) -> list[SearchResult]:
        """Rerank a fused agentic pool once before its source-bound audit."""

        if self.rag_mode != "hybrid" or self.reranker is None or not results:
            return results
        window = results[: min(self.rerank_candidate_count, len(results))]
        if window and all(
            "qwen3_rerank" in result.retrieval_sources for result in window
        ):
            return results
        return self._apply_semantic_reranker(
            question,
            results,
            deadline=deadline,
        )

    def retrieval_status(self, collection: LoadedCollection) -> DenseStatus:
        """Return the current retrieval mode/reason for audit and diagnostics."""

        return self._dense_manager.status(collection)

    def structured_retrieval_status(
        self,
        collection: LoadedCollection,
    ) -> dict[str, object]:
        """Report verified sparse derivatives used by the hybrid candidate."""

        index = self._structured_index(collection, required=False)
        if index is None:
            return {
                "available": False,
                "objectSearchAvailable": False,
                "evidenceSearchAvailable": False,
                "formatVersion": None,
            }
        return {
            "available": (
                index.object_search_available and index.evidence_search_available
            ),
            "objectSearchAvailable": index.object_search_available,
            "evidenceSearchAvailable": index.evidence_search_available,
            "formatVersion": index.manifest.get("formatVersion"),
        }

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
