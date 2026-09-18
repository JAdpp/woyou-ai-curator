"""What the curator shows a visitor who arrives without a question.

"I don't know what to ask" is the most common honest opening, and answering it
with "I'll build the room around 'I don't know what to ask'" is a joke. A person
in that spot is shown around first: what the collection is, and a few objects
worth stopping at, each carrying a question the collection is known to answer.

The entries are hand-picked, not generated. On 2026-09-18 every Chinese
question went through the public interview, hybrid retrieval and the evidence
audit against ``global_open`` (``scripts/probe_visitor_retrieval.py``) and came
back with 6-10 accepted objects, where five are needed; the cat and market
questions had also produced full exhibitions before. That is why the interview
skips the evidence negotiation for them. The English wordings mirror the
Chinese but were not probed separately. Change a question and it must be
probed again. An entry whose hero object is missing from the loaded
collection is dropped rather than shown as a broken card.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import i18n
from .collections import LoadedCollection

FEATURED_PREFIX = "__featured__:"


@dataclass(frozen=True)
class FeaturedEntry:
    id: str
    object_id: str
    # Card headline in the visitor's own voice; also what their chat bubble
    # shows once picked, so it stays under the 60-character answer label cap.
    title_zh: str
    title_en: str
    # The retrieval question. Longer than the title where the extra words are
    # what made the question answerable in the first place.
    question_zh: str
    question_en: str
    # Institution title kept as catalogued; place, period and material are ours.
    caption_zh: str
    caption_en: str
    # How the curator names the starting point in one breath ("这抹蓝").
    hook_zh: str
    hook_en: str
    # What choosing it will do to the visit. No specific object is promised:
    # retrieval and the evidence audit still decide the final five.
    plan_zh: str
    plan_en: str

    def title(self, language: str) -> str:
        return i18n.pick(language, self.title_zh, self.title_en)

    def question(self, language: str) -> str:
        return i18n.pick(language, self.question_zh, self.question_en)

    def caption(self, language: str) -> str:
        return i18n.pick(language, self.caption_zh, self.caption_en)

    def hook(self, language: str) -> str:
        return i18n.pick(language, self.hook_zh, self.hook_en)

    def plan(self, language: str) -> str:
        return i18n.pick(language, self.plan_zh, self.plan_en)


FEATURED_ENTRIES: tuple[FeaturedEntry, ...] = (
    FeaturedEntry(
        id="blue",
        object_id="cma:1939.205",
        # Worded as looking, not "why did the blue spread". On 2026-09-18 the
        # landing page's causal wording and a materials comparison both failed
        # query-plan review (invalid_v4_evidence_scope); that check was fixed
        # the next day, but only this wording has been probed end to end. It
        # returned Chinese, Vietnamese and Iranian pieces.
        title_zh="不同地方的陶瓷上，蓝色花纹画成了什么样？",
        title_en="How is blue painted on ceramics across cultures?",
        question_zh="我想看看不同地方的陶瓷上，蓝色花纹都画成了什么样子，比如中国、越南和伊朗的蓝彩器物。",
        question_en=(
            "I'd like to see how blue patterns were painted on ceramics from different "
            "places, such as Chinese, Vietnamese and Iranian blue-decorated wares."
        ),
        caption_zh="Bottle Vase with Dragons and Waves · 景德镇 · 1736–1795",
        caption_en="Bottle Vase with Dragons and Waves · Jingdezhen · 1736–95",
        hook_zh="这抹蓝",
        hook_en="that blue",
        plan_zh="我会把不同地方的蓝彩器物放在一起，看同样的蓝被画成了什么不同的花纹。",
        plan_en="I'll set blue-decorated pieces from different places side by side and see what different patterns the same blue became.",
    ),
    FeaturedEntry(
        id="cat",
        object_id="cma:1942.776",
        title_zh="猫在不同文化里，是什么样的存在？",
        title_en="What was the cat to different cultures?",
        question_zh="猫在不同文化里，分别是什么样的存在？",
        question_en="What kind of presence did the cat have in different cultures?",
        caption_zh="Cat · 古埃及晚期 · 青铜与金",
        caption_en="Cat · Egypt, Late Period · bronze and gold",
        hook_zh="猫",
        hook_en="the cat",
        plan_zh="同样是猫，在不同文化的器物和画里扮演的角色可能完全不同，我们一件件看。",
        plan_en="The same animal can play quite different parts in different cultures' objects and pictures, so we'll take them one at a time.",
    ),
    FeaturedEntry(
        id="landscape",
        object_id="cma:1952.285",
        title_zh="中国山水画，画的只是风景吗？",
        title_en="Is a Chinese landscape painting only scenery?",
        question_zh="中国山水画为什么不能只被理解为对自然风景的写生记录？",
        question_en="Why is a Chinese landscape painting more than a record of scenery?",
        caption_zh="Landscape with Flying Geese · 南宋 · 绢本册页",
        caption_en="Landscape with Flying Geese · Southern Song · album leaf on silk",
        hook_zh="山水",
        hook_en="landscape",
        plan_zh="我会挑几幅山水放在一起，看它们画的是眼前的风景，还是别的东西。",
        plan_en="I'll put a few landscapes together and ask whether they show the view in front of the painter or something else.",
    ),
    FeaturedEntry(
        id="market",
        object_id="cma:2020.113",
        title_zh="以前的人赶集、逛市场是什么样？",
        title_en="What were markets like in the past?",
        question_zh="以前的人赶集是什么样？我想看看市场、摊贩和街头买卖的画面。",
        question_en="What were markets like in the past? I'd like to see stalls, sellers and street trading.",
        caption_zh="Fishmarket · 法国 · 1902 · 布面油画",
        caption_en="Fishmarket · France · 1902 · oil on canvas",
        hook_zh="市集",
        hook_en="the market",
        plan_zh="我会从馆藏里找摊贩和买主挤在一起的画面，看不同地方的买卖是什么样子。",
        plan_en="I'll look for pictures of sellers and buyers crowded together, and see what trading looked like in different places.",
    ),
)

# Visitor-facing names for the source museums, keyed by institution id.
_INSTITUTION_NAMES_ZH = {
    "cma": "克利夫兰艺术博物馆",
    "met": "大都会艺术博物馆",
    "aic": "芝加哥艺术博物馆",
}


def featured_for(collection: LoadedCollection) -> list[FeaturedEntry]:
    """Entries whose hero object this collection can actually show."""

    showable = {obj.id for obj in collection.objects if obj.image_url}
    return [entry for entry in FEATURED_ENTRIES if entry.object_id in showable]


def entry_by_id(entry_id: str) -> FeaturedEntry | None:
    return next((entry for entry in FEATURED_ENTRIES if entry.id == entry_id), None)


def entry_for_question(question: str | None, language: str) -> FeaturedEntry | None:
    """The entry whose retrieval question this is, if the visitor kept it."""

    text = (question or "").strip()
    if not text:
        return None
    return next(
        (entry for entry in FEATURED_ENTRIES if entry.question(language) == text),
        None,
    )


def _join(items: list[str], language: str) -> str:
    if len(items) <= 1:
        return "".join(items)
    if language == "en":
        return ", ".join(items[:-1]) + " and " + items[-1]
    return "、".join(items[:-1]) + "和" + items[-1]


def collection_intro(collection: LoadedCollection, language: str = "zh") -> str:
    """One or two sentences about what is actually in the collection.

    Every number and name is counted from the loaded objects, so the intro
    stays true when the collection is rebuilt.
    """

    institutions: dict[str, int] = {}
    institution_names: dict[str, str] = {}
    packs: dict[str, int] = {}
    for obj in collection.objects:
        key = obj.institution_id or obj.institution
        if key:
            institutions[key] = institutions.get(key, 0) + 1
            institution_names[key] = obj.institution or key
        for pack_id in obj.culture_pack_ids:
            packs[pack_id] = packs.get(pack_id, 0) + 1

    ranked_institutions = sorted(institutions, key=lambda key: -institutions[key])
    if language == "en":
        museum_names = [institution_names[key] for key in ranked_institutions]
    else:
        museum_names = [
            _INSTITUTION_NAMES_ZH.get(key, institution_names[key])
            for key in ranked_institutions
        ]
    ranked_packs = [
        pack_id for pack_id in sorted(packs, key=lambda pack_id: -packs[pack_id])
    ]
    if language == "en":
        pack_names = [
            i18n.CULTURE_PACK_LABELS_EN.get(pack_id, pack_id) for pack_id in ranked_packs
        ]
    else:
        pack_names = [
            collection.culture_pack_labels.get(pack_id, pack_id) for pack_id in ranked_packs
        ]

    count = len(collection.objects)
    if language == "en":
        first = f"The collection holds {count:,} open-access objects"
        first += f" from {_join(museum_names, 'en')}." if museum_names else "."
        if len(pack_names) >= 3:
            first += (
                f" {_join(pack_names[:2], 'en')} are the best represented, and there are "
                f"also pieces from {_join(pack_names[2:], 'en')}."
            )
        return first
    first = f"我们的馆藏一共有 {count:,} 件开放藏品"
    first += f"，来自{_join(museum_names, 'zh')}。" if museum_names else "。"
    if len(pack_names) >= 3:
        first += f"{_join(pack_names[:2], 'zh')}的最多，{_join(pack_names[2:], 'zh')}的也都有。"
    return first
