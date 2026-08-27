"""Deterministic global routing taxonomy for the Woyou collection.

The taxonomy has two deliberately separate layers:

* ``CulturePack`` identifies broad cultural-geographic coverage buckets.  A
  pack is a corpus facet, not a claim that every object has one uncontested
  cultural identity and not a visitor-facing exhibition theme.
* ``GlobalDomain`` identifies cross-cultural evidence lenses such as ritual,
  material making, or exchange.  These ids can be shared by objects from many
  packs, which lets a visitor's question retrieve genuinely cross-cultural
  candidates.

All assignment is deterministic and reads institution-controlled metadata.
It never calls a model, and ASCII patterns are matched at word boundaries so
short tokens cannot leak into unrelated words (for example ``you`` must not
match ``young``).  Long institution prose is scored separately and weakly.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


ROUTING_VERSION = "1.1.1"
# Backward-compatible descriptive name used by the standalone manifest helper.
TAXONOMY_VERSION = ROUTING_VERSION


@dataclass(frozen=True)
class CulturePack:
    """A broad cultural-geographic facet used for coverage and diversity."""

    id: str
    label: str
    description: str
    patterns: tuple[str, ...]
    strong_patterns: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class GlobalDomain:
    """A cross-cultural corpus-routing lens, never a fixed exhibition theme."""

    id: str
    label: str
    description: str
    patterns: tuple[str, ...]
    strong_patterns: tuple[str, ...] = ()
    description_patterns: tuple[str, ...] = ()
    offer_in_interview: bool = True


@dataclass(frozen=True)
class RelationFacet:
    """An institution-metadata signal that may support an object relation.

    A facet is only a retrieval hint.  It never proves that two objects have a
    historical relationship; generation must still bind any such claim to the
    source evidence carried by the selected records.
    """

    id: str
    patterns: tuple[str, ...]
    strong_patterns: tuple[str, ...] = ()


# These packs are intentionally broad.  They are a serving-corpus coverage
# device, not a replacement for the institution's own culture/place fields.
CULTURE_PACKS: tuple[CulturePack, ...] = (
    CulturePack(
        id="east_asia",
        label="东亚",
        description="中国、日本、朝鲜半岛及相关文化语境。",
        patterns=(
            "east asia",
            "china",
            "chinese",
            "japan",
            "japanese",
            "korea",
            "korean",
            "ryukyu",
            "okinawa",
            "mongolia",
            "mongolian",
        ),
        # "Arts of Asia" is an institution department spanning East, South,
        # Southeast and West Asia.  Treating it as a strong East-Asia signal
        # misclassified explicitly Iranian AIC records, so only geographically
        # specific department labels receive a strong boost here.
        strong_patterns=("chinese art", "japanese art", "korean art"),
        aliases=("东亚", "中国", "日本", "韩国", "朝鲜", "蒙古"),
    ),
    CulturePack(
        id="south_asia",
        label="南亚",
        description="印度次大陆、斯里兰卡及喜马拉雅相关文化语境。",
        patterns=(
            "south asia",
            "india",
            "indian",
            "pakistan",
            "pakistani",
            "bangladesh",
            "bangladeshi",
            "sri lanka",
            "sri lankan",
            "nepal",
            "nepalese",
            "bhutan",
            "bhutanese",
            "himalayan",
            "gandhara",
            "tibet",
            "tibetan",
        ),
        strong_patterns=("indian subcontinent", "arts of india", "south asian art"),
        aliases=("南亚", "印度", "巴基斯坦", "斯里兰卡", "尼泊尔", "喜马拉雅"),
    ),
    CulturePack(
        id="southeast_asia",
        label="东南亚",
        description="东南亚大陆与岛屿区域的文化语境。",
        patterns=(
            "southeast asia",
            "thailand",
            "thai",
            "cambodia",
            "cambodian",
            "khmer",
            "vietnam",
            "vietnamese",
            "laos",
            "lao",
            "myanmar",
            "burma",
            "burmese",
            "indonesia",
            "indonesian",
            "java",
            "javanese",
            "bali",
            "balinese",
            "malaysia",
            "malaysian",
            "philippines",
            "filipino",
        ),
        strong_patterns=("southeast asian art", "arts of southeast asia"),
        aliases=("东南亚", "泰国", "柬埔寨", "越南", "缅甸", "印度尼西亚", "菲律宾"),
    ),
    CulturePack(
        id="west_asia_north_africa",
        label="西亚与北非",
        description="西亚、北非及跨区域伊斯兰文化语境。",
        patterns=(
            "west asia",
            "western asia",
            "middle east",
            "near east",
            "north africa",
            "egypt",
            "egyptian",
            "iran",
            "iranian",
            "persia",
            "persian",
            "iraq",
            "iraqi",
            "mesopotamia",
            "mesopotamian",
            "syria",
            "syrian",
            "levant",
            "levantine",
            "anatolia",
            "anatolian",
            "turkey",
            "turkish",
            "arabia",
            "arabian",
            "morocco",
            "moroccan",
            "algeria",
            "algerian",
            "tunisia",
            "tunisian",
            "islamic art",
        ),
        strong_patterns=("ancient near eastern art", "islamic art", "egyptian art"),
        aliases=("西亚", "中东", "北非", "伊斯兰", "埃及", "波斯", "伊朗", "美索不达米亚"),
    ),
    CulturePack(
        id="europe",
        label="欧洲",
        description="欧洲及古典地中海相关文化语境。",
        patterns=(
            "europe",
            "european",
            "england",
            "english",
            "britain",
            "british",
            "france",
            "french",
            "germany",
            "german",
            "italy",
            "italian",
            "netherlands",
            "dutch",
            "flanders",
            "flemish",
            "spain",
            "spanish",
            "portugal",
            "portuguese",
            "greece",
            "greek",
            "rome",
            "roman",
            "byzantine",
            "scandinavia",
            "scandinavian",
            "russia",
            "russian",
            "austria",
            "austrian",
            "switzerland",
            "swiss",
            "ireland",
            "irish",
            "belgium",
            "belgian",
            "denmark",
            "danish",
            "balkans",
            "medieval art",
        ),
        strong_patterns=("european paintings", "european decorative arts", "greek and roman art"),
        aliases=("欧洲", "英国", "法国", "德国", "意大利", "荷兰", "西班牙", "希腊", "罗马"),
    ),
    CulturePack(
        id="africa",
        label="非洲",
        description="撒哈拉以南非洲及非洲离散社群相关文化语境。",
        patterns=(
            "sub-saharan africa",
            "african art",
            "yoruba",
            "benin",
            "edo peoples",
            "akan",
            "asante",
            "kongo",
            "luba",
            "kuba",
            "dogon",
            "bamana",
            "mali",
            "malian",
            "ghana",
            "ghanaian",
            "nigeria",
            "nigerian",
            "ethiopia",
            "ethiopian",
            "kenya",
            "kenyan",
            "tanzania",
            "tanzanian",
            "south africa",
            "south african",
            "zulu",
            "xhosa",
        ),
        strong_patterns=("arts of africa", "africa oceania and the americas"),
        aliases=("非洲", "约鲁巴", "贝宁", "刚果", "埃塞俄比亚", "西非", "东非", "南部非洲"),
    ),
    CulturePack(
        id="americas",
        label="美洲",
        description="北美、中美、南美及美洲原住民相关文化语境。",
        patterns=(
            "americas",
            "north america",
            "south america",
            "latin america",
            "mesoamerica",
            "mesoamerican",
            "pre-columbian",
            "native american",
            "indigenous american",
            "maya",
            "mayan",
            "aztec",
            "mexica",
            "inca",
            "andean",
            "peru",
            "peruvian",
            "mexico",
            "mexican",
            "united states",
            "america",
            "american",
            "canada",
            "canadian",
        ),
        strong_patterns=("art of the americas", "ancient american art", "native north american art"),
        aliases=("美洲", "北美", "拉丁美洲", "中美洲", "南美", "玛雅", "阿兹特克", "印加"),
    ),
    CulturePack(
        id="oceania",
        label="大洋洲",
        description="太平洋岛屿、澳大利亚原住民与新西兰毛利文化语境。",
        patterns=(
            "oceania",
            "oceanic",
            "pacific islands",
            "polynesia",
            "polynesian",
            "melanesia",
            "melanesian",
            "micronesia",
            "micronesian",
            "maori",
            "new zealand",
            "aotearoa",
            "aboriginal australian",
            "indigenous australian",
            "papua new guinea",
            "hawaii",
            "hawaiian",
        ),
        strong_patterns=("arts of oceania", "oceanic art", "arts of africa oceania and the americas"),
        aliases=("大洋洲", "太平洋岛屿", "波利尼西亚", "美拉尼西亚", "毛利", "澳大利亚原住民"),
    ),
)


GLOBAL_DOMAINS: tuple[GlobalDomain, ...] = (
    GlobalDomain(
        id="global:nature-place",
        label="自然与地方",
        description="自然景观、动植物、建成环境与地方经验。",
        patterns=(
            "landscape", "seascape", "mountain", "river", "forest", "garden",
            "ocean", "waterfall", "botanical", "flora", "fauna", "nature",
            "environment", "cityscape", "architecture", "place",
        ),
        strong_patterns=("natural world", "landscape painting", "botanical study"),
        description_patterns=("landscape", "environment", "natural world", "sense of place"),
    ),
    GlobalDomain(
        id="global:belief-ritual",
        label="信仰与仪式",
        description="宗教信仰、礼仪实践、神圣空间与供奉对象。",
        patterns=(
            "ritual", "ceremonial", "religious", "devotional", "sacred", "worship",
            "deity", "god", "goddess", "altar", "shrine", "temple", "church",
            "mosque", "stupa", "reliquary", "icon", "buddha", "bodhisattva",
            "mandala", "offering", "pilgrimage", "amulet", "amulets",
        ),
        strong_patterns=("ritual object", "devotional image", "religious art", "sacred object"),
        description_patterns=("used in ritual", "worshipped", "devotional", "sacred", "ceremony"),
    ),
    GlobalDomain(
        id="global:death-afterlife",
        label="死亡与来世",
        description="丧葬、纪念、祖先观念与身后世界想象。",
        patterns=(
            "funerary", "mortuary", "tomb", "burial", "grave", "coffin",
            "sarcophagus", "mummy", "mummiform", "afterlife", "memorial",
            "grave goods", "ancestor shrine",
        ),
        strong_patterns=("funerary object", "burial object", "tomb figure", "grave goods"),
        description_patterns=("buried with", "from a tomb", "afterlife", "the deceased", "mortuary"),
    ),
    GlobalDomain(
        id="global:power-status",
        label="权力与身份秩序",
        description="王权、政治权威、社会等级、战争与身份标识。",
        patterns=(
            "royal", "imperial", "king", "queen", "ruler", "court", "crown",
            "throne", "heraldic", "coat of arms", "prestige", "rank", "authority",
            "armor", "armour", "weapon", "military", "war", "battle", "regalia",
            "coin", "coins", "medal",
        ),
        strong_patterns=("royal court", "imperial court", "symbol of power", "status symbol"),
        description_patterns=("political power", "social status", "high rank", "royal authority"),
    ),
    GlobalDomain(
        id="global:body-identity",
        label="身体与身份",
        description="肖像、身体表现、服饰、性别与社会身份。",
        patterns=(
            "portrait", "self-portrait", "human figure", "figure", "body", "nude",
            "dress", "costume", "garment", "robe", "headdress", "jewelry",
            "jewellery", "adornment", "mask", "gender", "identity",
        ),
        strong_patterns=("portrait painting", "self portrait", "body adornment", "ceremonial dress"),
        description_patterns=("social identity", "gender identity", "depicted wearing", "portrait of"),
    ),
    GlobalDomain(
        id="global:making-material",
        label="材料与制作",
        description="材料、技术、工艺流程与制作者知识。",
        patterns=(
            "ceramic", "porcelain", "stoneware", "earthenware", "pottery", "glaze",
            "textile", "woven", "weaving", "embroidery", "bronze", "metalwork",
            "glass", "wood", "wooden", "carved", "carving", "lacquer", "ivory",
            "jade", "printmaking", "engraving", "woodblock", "oil on canvas",
            "ink on paper", "ink on silk", "tempera", "photograph",
            "lace", "velvet", "silver", "metalwork", "basketry", "jewelry",
            "jewellery", "sculpture",
        ),
        strong_patterns=("manufacturing technique", "production process", "made from", "workshop practice"),
        description_patterns=("technique", "crafted", "manufactured", "fired in", "woven from"),
    ),
    GlobalDomain(
        id="global:text-memory",
        label="书写与记忆",
        description="文字、铭文、书籍、档案与社会记忆。",
        patterns=(
            "inscription", "inscribed", "calligraphy", "manuscript", "book", "scroll",
            "writing", "script", "alphabet", "tablet", "document", "letter", "poem",
            "poetry", "archive", "colophon", "epitaph", "stele", "rubbing",
            "bound volume",
        ),
        strong_patterns=("written text", "inscribed text", "historical document", "calligraphic work"),
        description_patterns=("records the", "commemorates", "written in", "inscription reads"),
    ),
    GlobalDomain(
        id="global:exchange-mobility",
        label="交流与流动",
        description="贸易、迁徙、旅行、殖民接触与知识传播。",
        patterns=(
            "trade", "export", "imported", "silk road", "maritime", "migration",
            "migrant", "travel", "journey", "diaspora", "exchange", "transmission",
            "merchant", "caravan", "port", "colonial", "pilgrimage", "diplomatic gift",
        ),
        strong_patterns=("cross-cultural exchange", "trade route", "made for export", "cultural transmission"),
        description_patterns=("traded between", "travelled from", "introduced to", "foreign market"),
    ),
    GlobalDomain(
        id="global:daily-life",
        label="日常生活",
        description="居家、饮食、劳动、娱乐与普通人的物质生活。",
        patterns=(
            "domestic", "household", "food", "cooking", "drinking", "bowl", "cup",
            "furniture", "chair", "table", "bed", "toy", "game", "work", "labor",
            "labour", "agriculture", "agricultural", "tool", "tools and equipment",
            "utensil", "vessel", "vessels", "everyday life",
        ),
        strong_patterns=("domestic life", "household object", "daily use", "everyday object"),
        description_patterns=("used at home", "daily life", "everyday use", "working life"),
    ),
    GlobalDomain(
        id="global:image-story",
        label="图像、观看与故事",
        description="绘画、版画、摄影、雕塑及其神话、历史与视觉叙事。",
        patterns=(
            "myth", "mythology", "mythological", "narrative", "story", "legend",
            "legendary", "epic", "hero", "heroic", "allegory", "allegorical",
            "historical scene", "biblical scene", "illustration", "theater", "theatre",
            "performance", "drama", "painting", "drawing", "print", "photograph",
            "sculpture", "poster", "visual art",
        ),
        strong_patterns=("narrative scene", "mythological scene", "visual narrative", "epic story"),
        description_patterns=("tells the story", "depicts the story", "episode from", "scene from"),
    ),
)


RELATION_FACETS: tuple[RelationFacet, ...] = (
    RelationFacet(
        id="relation:trade",
        patterns=("trade", "export", "imported", "merchant", "market", "caravan", "port"),
        strong_patterns=("made for export", "trade route", "export market"),
    ),
    RelationFacet(
        id="relation:mobility",
        patterns=("migration", "migrant", "diaspora", "travel", "journey", "pilgrimage"),
        strong_patterns=("migrated from", "travelled from", "carried along"),
    ),
    RelationFacet(
        id="relation:transmission",
        patterns=("transmission", "influence", "introduced", "adapted", "borrowed"),
        strong_patterns=("cultural transmission", "influenced by", "introduced from"),
    ),
    RelationFacet(
        id="relation:colonial-contact",
        patterns=("colonial", "colony", "colonization", "colonisation", "missionary"),
        strong_patterns=("colonial rule", "colonial encounter"),
    ),
    RelationFacet(
        id="relation:diplomatic-gift",
        patterns=("diplomatic", "tribute", "gift", "presented to"),
        strong_patterns=("diplomatic gift", "tribute gift"),
    ),
)


CULTURE_PACK_BY_ID = {pack.id: pack for pack in CULTURE_PACKS}
GLOBAL_DOMAIN_BY_ID = {domain.id: domain for domain in GLOBAL_DOMAINS}


# Chinese-first query expansion for the initial audience.  Values remain
# institution-catalogue vocabulary rather than model-authored labels.
CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    # Cultural-geographic packs
    "东亚": ("east asia", "china", "japan", "korea"),
    "中国": ("china", "chinese"),
    "日本": ("japan", "japanese"),
    "朝鲜": ("korea", "korean"),
    "韩国": ("korea", "korean"),
    "南亚": ("south asia", "india", "pakistan", "sri lanka", "nepal"),
    "印度": ("india", "indian"),
    "东南亚": ("southeast asia", "thailand", "cambodia", "vietnam", "indonesia"),
    "西亚": ("west asia", "middle east", "near east"),
    "中东": ("middle east", "west asia", "near east"),
    "北非": ("north africa", "egypt", "morocco", "algeria", "tunisia"),
    "伊斯兰": ("islamic", "islamic art", "muslim"),
    "埃及": ("egypt", "egyptian"),
    "波斯": ("persia", "persian", "iran", "iranian"),
    "欧洲": ("europe", "european"),
    "非洲": ("africa", "african", "sub-saharan africa"),
    "美洲": ("americas", "american", "mesoamerica", "andean"),
    "大洋洲": ("oceania", "oceanic", "pacific islands"),
    # Nature and place
    "自然": ("nature", "natural world", "environment"),
    "山水": ("landscape", "mountain", "river", "seascape"),
    "风景": ("landscape", "seascape", "view"),
    "环境": ("environment", "ecology", "natural world"),
    "地方": ("place", "region", "geography"),
    "动物": ("animal", "animals", "lion", "dragon", "bird", "horse"),
    "植物": ("plant", "plants", "botanical", "flower", "tree"),
    # Belief, death, and power
    "信仰": ("belief", "religious", "devotional", "worship"),
    "宗教": ("religion", "religious", "sacred", "devotional"),
    "仪式": ("ritual", "ceremonial", "ceremony", "offering"),
    "祭祀": ("ritual", "sacrifice", "altar", "offering"),
    "神": ("deity", "god", "goddess", "divine"),
    "死亡": ("death", "funerary", "mortuary", "burial", "afterlife"),
    "丧葬": ("funerary", "mortuary", "burial", "tomb"),
    "墓葬": ("tomb", "burial", "grave", "grave goods"),
    "来世": ("afterlife", "underworld", "funerary"),
    "祖先": ("ancestor", "ancestral", "ancestor shrine"),
    "权力": ("power", "authority", "royal", "imperial", "ruler"),
    "王权": ("kingship", "royal", "king", "regalia", "throne"),
    "等级": ("rank", "hierarchy", "status", "social order"),
    "战争": ("war", "battle", "military", "weapon", "armor"),
    # Body and identity
    "身体": ("body", "human figure", "figure", "nude"),
    "肖像": ("portrait", "self-portrait", "likeness"),
    "身份": ("identity", "status", "rank", "social identity"),
    "性别": ("gender", "woman", "women", "man", "men"),
    "服饰": ("dress", "costume", "garment", "robe", "clothing"),
    "面具": ("mask", "masquerade"),
    # Making and material
    "材料": ("material", "medium", "made from"),
    "工艺": ("technique", "craft", "making", "manufacture"),
    "陶瓷": ("ceramic", "ceramics", "porcelain", "pottery", "stoneware", "earthenware"),
    "青花": ("blue-and-white", "blue and white", "cobalt", "underglaze blue"),
    "纺织": ("textile", "woven", "weaving", "embroidery"),
    "金属": ("metal", "metalwork", "bronze", "silver", "gold"),
    "青铜": ("bronze", "bronze vessel", "cast bronze"),
    "玉": ("jade", "nephrite", "jadeite"),
    "玻璃": ("glass", "glassware"),
    "木雕": ("wood carving", "carved wood", "wooden sculpture"),
    "摄影": ("photograph", "photography", "photographic"),
    # Text and memory
    "文字": ("writing", "written text", "script", "inscription"),
    "书写": ("writing", "script", "calligraphy", "manuscript"),
    "书法": ("calligraphy", "script", "brush writing"),
    "铭文": ("inscription", "inscribed", "epigraph"),
    "书籍": ("book", "manuscript", "codex", "printed book"),
    "记忆": ("memory", "memorial", "commemoration", "archive"),
    "档案": ("archive", "document", "record", "historical document"),
    # Exchange and movement
    "交流": ("exchange", "transmission", "cross-cultural", "contact"),
    "跨文化": ("cross-cultural", "intercultural", "exchange", "transmission"),
    "贸易": ("trade", "merchant", "export", "imported"),
    "迁徙": ("migration", "migrant", "diaspora", "movement"),
    "旅行": ("travel", "journey", "traveler", "pilgrimage"),
    "丝路": ("silk road", "caravan", "central asia", "trade route"),
    # Daily life and narrative
    "日常": ("daily life", "everyday", "domestic", "household"),
    "家庭": ("family", "household", "domestic"),
    "饮食": ("food", "cooking", "drinking", "dining"),
    "劳动": ("work", "labor", "labour", "occupation"),
    "游戏": ("game", "play", "toy", "entertainment"),
    "神话": ("myth", "mythology", "mythological", "legend"),
    "故事": ("story", "narrative", "episode", "scene"),
    "叙事": ("narrative", "story", "visual narrative"),
    "英雄": ("hero", "heroic", "epic"),
    # Cross-domain visual prompts often used by visitors
    "蓝色": ("blue", "cobalt", "azure", "indigo", "ultramarine"),
    "红色": ("red", "vermilion", "crimson", "scarlet"),
    "金色": ("gold", "gilded", "gilt", "golden"),
}


_CULTURE_FIELDS = (
    "culture",
    "culture_display",
    "cultureDisplay",
    "place",
    "country",
    "region",
    "geography",
    "department",
    "department_title",
    "departmentTitle",
    "classification",
)

_DOMAIN_FIELDS = (
    "classification",
    "type",
    "object_type",
    "objectType",
    "medium",
    "material",
    "title",
    "title_original",
    "titleOriginal",
    "culture",
    "place",
    "tags",
    "term_titles",
    "termTitles",
)


def _value(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _flatten(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result: list[str] = []
        for key in sorted(value, key=str):
            result.extend(_flatten(value[key]))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [str(value)]


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
    return re.sub(r"\s+", " ", text).strip()


def _haystack(obj: Any, fields: tuple[str, ...]) -> str:
    parts: list[str] = []
    seen_fields: set[str] = set()
    for field in fields:
        # Several field aliases may point at the same source value.  Retaining
        # the first name is enough and prevents duplicate text affecting future
        # count-based scorers.
        if field in seen_fields:
            continue
        seen_fields.add(field)
        parts.extend(_flatten(_value(obj, field)))
    return _normalise(" ".join(parts))


@lru_cache(maxsize=None)
def _compiled_pattern(pattern: str) -> tuple[str, re.Pattern[str] | None]:
    needle = _normalise(pattern)
    if re.search(r"[a-z0-9]", needle):
        expression = rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])"
        return needle, re.compile(expression)
    return needle, None


def _matches_normalised(haystack: str, pattern: str) -> bool:
    needle, expression = _compiled_pattern(pattern)
    if not haystack or not needle:
        return False
    if expression is not None:
        return expression.search(haystack) is not None
    return needle in haystack


def matches_pattern(text: str, pattern: str) -> bool:
    """Return a Unicode-normalised, boundary-safe deterministic match."""

    return _matches_normalised(_normalise(text), pattern)


def _pattern_score(
    haystack: str,
    patterns: tuple[str, ...],
    strong_patterns: tuple[str, ...],
) -> float:
    # Callers already provide a normalized haystack.  Re-normalizing it for
    # every one of several hundred rules made a 14k rebuild CPU-bound.
    score = sum(1.0 for pattern in patterns if _matches_normalised(haystack, pattern))
    score += sum(
        3.0 for pattern in strong_patterns if _matches_normalised(haystack, pattern)
    )
    return score


def score_culture_packs(obj: Any) -> dict[str, float]:
    """Score broad packs from controlled culture/geography fields only."""

    haystack = _haystack(obj, _CULTURE_FIELDS)
    scores: dict[str, float] = {}
    for pack in CULTURE_PACKS:
        score = _pattern_score(haystack, pack.patterns, pack.strong_patterns)
        if score:
            scores[pack.id] = score
    return scores


def assign_culture_packs(obj: Any, max_packs: int = 2) -> list[str]:
    """Return up to ``max_packs`` ids, strongest first and stably tie-broken."""

    if max_packs < 1:
        return []
    ranked = sorted(score_culture_packs(obj).items(), key=lambda item: (-item[1], item[0]))
    if not ranked:
        return []
    best = ranked[0][1]
    return [pack_id for pack_id, score in ranked[:max_packs] if score >= best * 0.7]


def score_global_domains(obj: Any) -> dict[str, float]:
    """Score cross-cultural evidence lenses from controlled metadata.

    Institution description prose can add at most two points to a domain.  It
    can therefore support a controlled-field match but cannot make a chatty
    record dominate the routing result by itself.
    """

    haystack = _haystack(obj, _DOMAIN_FIELDS)
    description = _haystack(obj, ("description", "summary", "tombstone"))
    scores: dict[str, float] = {}
    for domain in GLOBAL_DOMAINS:
        score = _pattern_score(haystack, domain.patterns, domain.strong_patterns)
        if description and domain.description_patterns:
            prose_hits = sum(
                1
                for pattern in domain.description_patterns
                if _matches_normalised(description, pattern)
            )
            score += min(prose_hits * 0.5, 2.0)
        if score:
            scores[domain.id] = score
    return scores


def assign_global_domains(obj: Any, max_domains: int = 3) -> list[str]:
    """Return competitive global domain ids in deterministic rank order."""

    if max_domains < 1:
        return []
    ranked = sorted(score_global_domains(obj).items(), key=lambda item: (-item[1], item[0]))
    if not ranked:
        return []
    best = ranked[0][1]
    return [domain_id for domain_id, score in ranked[:max_domains] if score >= best * 0.5]


def assign_evidence_domains(
    obj: Any,
    culture_pack_ids: Sequence[str] | None = None,
    max_domains: int = 3,
) -> list[str]:
    """Stable importer-facing name for cross-cultural evidence assignment.

    ``culture_pack_ids`` is accepted now so pack-local rules can be added later
    without changing importer call sites.  Version 1 contains global rules only,
    therefore the argument intentionally does not alter the result.
    """

    del culture_pack_ids
    return assign_global_domains(obj, max_domains=max_domains)


def assign_relation_facets(obj: Any, max_facets: int = 3) -> list[str]:
    """Return source-grounded relation hints without inventing causal links."""

    if max_facets < 1:
        return []
    haystack = " ".join(
        filter(
            None,
            (
                _haystack(obj, _DOMAIN_FIELDS),
                _haystack(obj, ("description", "summary", "tombstone")),
            ),
        )
    )
    scores = {
        facet.id: _pattern_score(haystack, facet.patterns, facet.strong_patterns)
        for facet in RELATION_FACETS
    }
    ranked = sorted(
        ((facet_id, score) for facet_id, score in scores.items() if score),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked:
        return []
    best = ranked[0][1]
    return [facet_id for facet_id, score in ranked[:max_facets] if score >= best * 0.5]


def assign_taxonomy(obj: Any) -> dict[str, list[str]]:
    """Assign both independent layers using JSON-ready canonical field names."""

    return {
        "culturePackIds": assign_culture_packs(obj),
        "evidenceDomainIds": assign_evidence_domains(obj),
        "relationFacets": assign_relation_facets(obj),
    }


def expand_concept_aliases(text: str) -> tuple[str, ...]:
    """Return deterministic English catalogue vocabulary for Chinese concepts."""

    expanded: set[str] = set()
    normalised = _normalise(text)
    for concept, aliases in CONCEPT_ALIASES.items():
        if _normalise(concept) in normalised:
            expanded.update(_normalise(alias) for alias in aliases if alias)
    return tuple(sorted(expanded))


def culture_pack_manifest(
    distribution: Mapping[str, int],
    full_evidence_distribution: Mapping[str, int] | None = None,
    *,
    min_objects: int = 200,
    min_full_evidence: int = 20,
) -> list[dict[str, Any]]:
    """Build a stable manifest section in registry order."""

    full_evidence_distribution = full_evidence_distribution or {}
    entries: list[dict[str, Any]] = []
    for pack in CULTURE_PACKS:
        object_count = int(distribution.get(pack.id, 0))
        full_count = int(full_evidence_distribution.get(pack.id, 0))
        entries.append(
            {
                "id": pack.id,
                "label": pack.label,
                "description": pack.description,
                "objectCount": object_count,
                "fullEvidenceCount": full_count,
                "meetsMinimum": (
                    object_count >= min_objects and full_count >= min_full_evidence
                ),
                "aliases": list(pack.aliases),
                "note": "语料覆盖分面，不是单一、排他的文化身份判断。",
            }
        )
    return entries


def global_domain_manifest(
    distribution: Mapping[str, int],
    full_evidence_distribution: Mapping[str, int] | None = None,
    *,
    min_objects: int = 40,
    min_full_evidence: int = 5,
) -> list[dict[str, Any]]:
    """Build a stable global-domain manifest section in registry order."""

    full_evidence_distribution = full_evidence_distribution or {}
    entries: list[dict[str, Any]] = []
    for domain in GLOBAL_DOMAINS:
        object_count = int(distribution.get(domain.id, 0))
        full_count = int(full_evidence_distribution.get(domain.id, 0))
        entries.append(
            {
                "id": domain.id,
                "scope": "global",
                "label": domain.label,
                "description": domain.description,
                "objectCount": object_count,
                "fullEvidenceCount": full_count,
                "meetsMinimum": (
                    object_count >= min_objects and full_count >= min_full_evidence
                ),
                "offerInInterview": domain.offer_in_interview,
                "note": "跨文化语料路由标签，不是访客端固定策展主题。",
            }
        )
    return entries


def domain_manifest(
    distribution: Mapping[str, int],
    full_distribution: Mapping[str, int] | None = None,
    *,
    min_objects: int = 40,
    min_full_evidence: int = 5,
) -> list[dict[str, Any]]:
    """Stable importer-facing alias for the global evidence-domain manifest."""

    return global_domain_manifest(
        distribution,
        full_distribution,
        min_objects=min_objects,
        min_full_evidence=min_full_evidence,
    )


def taxonomy_manifest(
    culture_pack_distribution: Mapping[str, int],
    global_domain_distribution: Mapping[str, int],
    *,
    culture_pack_full_evidence: Mapping[str, int] | None = None,
    global_domain_full_evidence: Mapping[str, int] | None = None,
    min_pack_objects: int = 200,
    min_pack_full_evidence: int = 20,
    min_domain_objects: int = 40,
    min_domain_full_evidence: int = 5,
) -> dict[str, Any]:
    """Return all taxonomy metadata ready to merge into a collection manifest."""

    return {
        "taxonomyVersion": TAXONOMY_VERSION,
        "routingVersion": ROUTING_VERSION,
        "culturePacks": culture_pack_manifest(
            culture_pack_distribution,
            culture_pack_full_evidence,
            min_objects=min_pack_objects,
            min_full_evidence=min_pack_full_evidence,
        ),
        "evidenceDomains": global_domain_manifest(
            global_domain_distribution,
            global_domain_full_evidence,
            min_objects=min_domain_objects,
            min_full_evidence=min_domain_full_evidence,
        ),
        "conceptAliases": {
            concept: list(aliases) for concept, aliases in CONCEPT_ALIASES.items()
        },
        "boundaries": [
            "CulturePack 是语料覆盖分面，不是单一、排他的文化身份结论。",
            "全局证据域用于检索与覆盖审计，不是访客端预设策展主题。",
            "规则只读取机构元数据；未命中规则的合格对象应保留并标记为未路由。",
        ],
    }


__all__ = [
    "CONCEPT_ALIASES",
    "CULTURE_PACKS",
    "CULTURE_PACK_BY_ID",
    "GLOBAL_DOMAINS",
    "GLOBAL_DOMAIN_BY_ID",
    "RELATION_FACETS",
    "ROUTING_VERSION",
    "TAXONOMY_VERSION",
    "CulturePack",
    "GlobalDomain",
    "RelationFacet",
    "assign_culture_packs",
    "assign_global_domains",
    "assign_evidence_domains",
    "assign_relation_facets",
    "assign_taxonomy",
    "culture_pack_manifest",
    "domain_manifest",
    "expand_concept_aliases",
    "global_domain_manifest",
    "matches_pattern",
    "score_culture_packs",
    "score_global_domains",
    "taxonomy_manifest",
]
