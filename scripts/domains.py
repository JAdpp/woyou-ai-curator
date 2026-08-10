"""Coverage-domain assignment for the merged multi-institution collection.

These are *corpus routing labels*, not visitor-facing exhibition themes. A
visitor's theme is formed per session from the curator interview; domains only
guarantee that some slice of the corpus can actually answer it.

Assignment is deterministic and rule-based rather than embedding-clustered so a
frozen collection version reproduces byte-identically, which the regression
suite depends on. Rules read institution-controlled fields only (classification,
object type, medium, Getty AAT tags, title), never model output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Domain:
    id: str
    label: str
    description: str
    # Matched against a lowercased blob of classification/type/medium/tags/title.
    patterns: tuple[str, ...]
    # A hit here outweighs several weak pattern hits.
    strong_patterns: tuple[str, ...] = ()
    # Thematic domains (trade, funerary, inscription practice) are rarely
    # visible in controlled fields — an export bowl is classified as a bowl.
    # These patterns additionally search the curatorial description, scored
    # lower because prose also mentions what an object is *not*.
    description_patterns: tuple[str, ...] = ()


DOMAINS: tuple[Domain, ...] = (
    Domain(
        id="landscape-brush",
        label="山水与笔墨",
        description="山水、花鸟与文人绘画，及其笔法、皴染与画面结构。",
        patterns=(
            "landscape", "mountain", "river", "album leaf", "handscroll",
            "hanging scroll", "ink and color", "ink on paper", "ink on silk",
            "painting", "bamboo", "plum", "orchid", "literati", "fan",
        ),
        strong_patterns=("landscape", "literati", "ink and color on paper"),
    ),
    Domain(
        id="calligraphy-inscription",
        label="书写、题跋与铭文",
        description="书法作品、画上题跋、器物铭文与印章，及其与图像的关系。",
        patterns=(
            "calligraphy", "inscription", "colophon", "seal", "script",
            "cursive", "running script", "clerical script", "rubbing",
            "epitaph", "stele", "poem", "manuscript", "sutra",
        ),
        strong_patterns=("calligraphy", "inscription", "colophon", "rubbing", "stele"),
        description_patterns=(
            "inscription", "inscribed", "colophon", "calligraph", "signature",
            "seal of", "poem", "wrote", "brushwork", "characters",
        ),
    ),
    Domain(
        id="ritual-bronze",
        label="礼器与青铜",
        description="商周礼器、祭祀用具与其铭文、纹饰及使用场景。",
        patterns=(
            "bronze", "ritual", "ceremonial", "vessel", "gui", "ding", "zun",
            "you", "jue", "hu", "bell", "altar", "sacrific", "libation",
            "food container", "wine container", "mirror",
        ),
        strong_patterns=("ritual vessel", "ceremonial", "bronze", "libation"),
    ),
    Domain(
        id="ceramics-glaze",
        label="陶瓷与釉色",
        description="陶瓷器形、釉色与烧造工艺，含青花、青瓷、白瓷与彩瓷。",
        patterns=(
            "porcelain", "stoneware", "earthenware", "ceramic", "glaze",
            "celadon", "blue-and-white", "underglaze", "kiln", "ware",
            "bowl", "dish", "vase", "jar", "ewer", "bottle", "cup",
        ),
        strong_patterns=("porcelain", "celadon", "glaze", "blue-and-white", "stoneware"),
    ),
    Domain(
        id="buddhist-devotion",
        label="佛教造像与信仰",
        description="佛教、道教造像与法器，及其仪轨、供养与图像程式。",
        patterns=(
            "buddha", "bodhisattva", "buddhist", "guanyin", "avalokite",
            "mandala", "daoist", "taoist", "deity", "sculpture", "stupa",
            "shrine", "reliquar", "lohan", "arhat", "maitreya", "amitabha",
            "thangka", "sutra", "temple",
        ),
        strong_patterns=("buddha", "bodhisattva", "mandala", "guanyin", "buddhist"),
    ),
    Domain(
        id="funerary-afterlife",
        label="丧葬与来世观",
        description="墓葬明器、俑像与随葬品所反映的死亡观念与身后世界想象。",
        patterns=(
            "tomb", "funerary", "burial", "mingqi", "grave", "sarcoph",
            "coffin", "epitaph", "spirit", "afterlife", "tomb figure",
            "guardian", "attendant figure",
        ),
        strong_patterns=("tomb", "funerary", "burial", "mingqi"),
        description_patterns=(
            "tomb", "buried", "burial", "funerary", "afterlife", "grave goods",
            "excavated from", "interred", "deceased", "mortuary",
        ),
    ),
    Domain(
        id="court-daily-life",
        label="宫廷、服饰与日常",
        description="宫廷器用、家具、服饰纺织与日常生活用品所呈现的身份与礼制。",
        patterns=(
            "textile", "robe", "costume", "silk", "embroider", "furniture",
            "chair", "table", "screen", "lacquer", "jade", "ornament",
            "jewel", "hairpin", "belt", "snuff", "brush", "inkstone",
            "headrest", "pillow", "court", "imperial", "dragon robe",
        ),
        strong_patterns=("textile", "robe", "furniture", "lacquer", "inkstone", "imperial"),
    ),
    Domain(
        id="trade-exchange",
        label="贸易与跨文化交流",
        description="沿丝路与海路流动的器物、外销品与外来母题的本地转化。",
        patterns=(
            "export", "trade", "silk road", "sogdian", "persian", "islamic",
            "kraak", "cobalt", "imported", "foreign", "central asia",
            "camel", "hellenistic", "gandhara", "sasanian", "tribute",
        ),
        strong_patterns=("export", "silk road", "sogdian", "gandhara", "kraak", "sasanian"),
        description_patterns=(
            "export", "trade", "traded", "silk road", "imported", "foreign",
            "overseas", "sogdian", "persia", "islamic world", "cobalt",
            "central asia", "middle east", "europe", "maritime", "merchant",
            "tribute", "west asia", "gandhara",
        ),
    ),
)

DOMAIN_BY_ID = {domain.id: domain for domain in DOMAINS}

# A domain must clear this to be usable as an answerability target.
MIN_OBJECTS_PER_DOMAIN = 40


def _haystack(obj: Any) -> str:
    """Build the lowercased text the rules match against.

    Deliberately excludes long curatorial prose: descriptions mention many
    things an object is *not*, which produced noisy multi-domain hits in
    testing. Only controlled fields plus the title are read.
    """
    parts = [
        getattr(obj, "classification", ""),
        getattr(obj, "type", ""),
        getattr(obj, "medium", ""),
        getattr(obj, "title", ""),
        getattr(obj, "culture", ""),
        getattr(obj, "place", ""),
        " ".join(getattr(obj, "tags", []) or []),
    ]
    return " ".join(str(part) for part in parts if part).casefold()


def score_domains(obj: Any) -> dict[str, float]:
    haystack = _haystack(obj)
    prose = str(getattr(obj, "description", "") or "").casefold()
    scores: dict[str, float] = {}
    for domain in DOMAINS:
        score = 0.0
        for pattern in domain.patterns:
            if pattern in haystack:
                score += 1.0
        for pattern in domain.strong_patterns:
            if pattern in haystack:
                score += 2.5
        if prose:
            # Capped so a chatty description cannot outrank a controlled-field
            # match, and diminishing so one repeated word is not decisive.
            prose_hits = sum(1 for pattern in domain.description_patterns if pattern in prose)
            score += min(prose_hits * 0.8, 3.2)
        if score:
            scores[domain.id] = score
    return scores


def assign(obj: Any, max_domains: int = 2) -> list[str]:
    """Return up to ``max_domains`` domain ids, strongest first.

    Objects that match nothing return ``[]`` and are dropped by the importer:
    an object no domain can route to cannot be retrieved, so keeping it only
    inflates the headline count.
    """
    scores = score_domains(obj)
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    best = ranked[0][1]
    # Second domain only if it is genuinely competitive, to avoid every ceramic
    # also being filed under trade.
    kept = [domain_id for domain_id, score in ranked[:max_domains] if score >= best * 0.6]
    return kept


def domain_manifest(distribution: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {
            "id": domain.id,
            "label": domain.label,
            "description": domain.description,
            "objectCount": distribution.get(domain.id, 0),
            "meetsMinimum": distribution.get(domain.id, 0) >= MIN_OBJECTS_PER_DOMAIN,
            "note": "语料路由标签，不是访客端固定策展主题。",
        }
        for domain in DOMAINS
    ]


def slug_to_label(domain_id: str) -> str:
    domain = DOMAIN_BY_ID.get(domain_id)
    return domain.label if domain else domain_id


_CJK = re.compile(r"[㐀-鿿]")


def is_chinese(text: str) -> bool:
    return bool(_CJK.search(text or ""))
