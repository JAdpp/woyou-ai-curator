"""Conservative linguistic cue for an underspecified opening, without I/O.

This is not a semantic question classifier or a topic whitelist. It only
recognises generic/deictic referents and degree-modified generic nouns with no
explicit head ("the most ... exhibition", "that kind of thing"). A specific
object, place, use or compound question remains untouched. Wider ambiguity
still belongs to the evidence-aware planner, not ever-growing topic rules.
"""
from __future__ import annotations

import re


def needs_scope_clarification(text: str) -> bool:
    value = re.sub(r"[\s，,。.!！?？‘’“”\"']+", "", text.strip())
    if not value or len(value) > 90:
        return False
    value = re.sub(
        r"^(?:(?:能不能|想要|我们|给我|帮我|麻烦|希望|看看|找找|推荐|安排|展示|一座|一个|一些|几个|几件|一点|请|我|想|看|找|做|挑|个|点))+",
        "", value,
    )
    value = re.sub(r"(?:就行|就好|吧|吗|呢)$", "", value)
    generic = r"(?:东西|作品|展品|展览|物件|对象)"
    deictic = r"(?:这样|那样|这种|那种|这一类|那一类|这类|那类|这些|那些)"
    patterns = (
        rf"{deictic}的?{generic}",
        # The 的 must directly precede a generic noun (or its demonstrative):
        # "最古老的陶瓷" and "最真实的人像作品" therefore stay concrete.
        rf"(?:最|更|比较|很|特别|非常|有点|有些).{{1,16}}?的(?:{deictic})?{generic}",
        rf"(?:以前|过去|从前|当时|那时).{{0,12}}?的{deictic}{generic}",
    )
    return any(re.fullmatch(pattern, value) for pattern in patterns)


# An opening turn that carries no subject at all: "I don't know what to ask",
# "you choose", "what have you got", a bare greeting. Storing one of these as
# the visitor's question made the curator promise to build a room around "I
# don't know what to ask", so they are recognised and answered with the
# collection instead. The cues must account for the whole utterance; one
# content word left over ("我不知道青花瓷怎么烧的") keeps it a real question.
_UNDECIDED_ZH: dict[str, str] = {
    "meta": r"你是谁|你是(?:干|做)什么的|你(?:能|可以|会)(?:做|干)?(?:什么|啥)|(?:这|这个)是(?:什么|干嘛的|做什么的)|怎么(?:用|玩|开始)",
    "delegate": (
        r"(?:你|您|彦远)?(?:来|帮我|替我|给我|帮忙)?(?:推荐|决定|挑|安排|做主|看着办|拿主意)"
        r"(?:一下|一个|几个|几件|一些|点)?(?:给我|我)?(?:看|看看)?(?:什么|啥|哪些|哪件)?(?:好|的)?"
        r"|(?:你|您|彦远)(?:来)?(?:定|选)"
    ),
    "collection": (
        r"(?:你们|你们馆|馆里|这里|这儿|你这|你这里|卧游|馆藏|博物馆里?|馆)?(?:都|一共|现在)?(?:有|收藏了?|收了)"
        r"(?:什么|啥|哪些|多少)(?:好看的|好玩的|有意思的|有趣的|值得看的)?(?:藏品|展品|东西|宝贝|作品|馆藏)?"
        r"|(?:介绍|讲讲|说说)(?:一下|下)?(?:你们|你|馆藏|藏品|这里|卧游|自己)?的?(?:馆藏|藏品)?"
    ),
    "unsure": (
        r"(?:还|也|真的|完全|暂时|其实)*"
        r"(?:不知道|不晓得|不太知道|不清楚|不确定|没想好|想不好|说不好|想不出来?|想不到|想不起来|说不上来"
        r"|没头绪|没有头绪|没主意|没有主意|没(?:什么|啥|有)?(?:想法|特别想(?:看|了解)的|特别的)"
        r"|没什么|没啥|没有|不懂|不太懂|不太清楚|不太确定)"
        r"(?:(?:应该|该|要|可以|能|想)?(?:问|看|了解|聊|说|选|找)?(?:你|您)?(?:些|点)?"
        r"(?:什么|啥|哪些|哪个|哪方面的?)?(?:问题|东西)?)?"
        r"(?:(?:从|在)?(?:哪|哪里|哪儿|何)(?:开始|问起|看起|说起|入手|下手))?"
    ),
    "browse": (
        r"(?:随便|随意|都行|都可以|都好|无所谓|看心情|什么都行|什么都可以|啥都行|都ok)"
        r"(?:带我)?(?:看看|逛逛|转转|瞧瞧|聊聊)?(?:就行|就好|而已)?"
        r"|(?:想|先|想先)?(?:带我)?(?:随便)?(?:看看|逛逛|转转|瞧瞧)(?:就行|就好|而已)?"
    ),
    "greeting": r"你好|您好|嗨|嘿|在吗|在不在|早上好|下午好|晚上好|hi|hello|hey",
}
_UNDECIDED_EN: dict[str, str] = {
    "meta": r"who are you|what are you|what (?:can|do) you do|how does (?:this|it) work|what is this",
    "delegate": (
        r"you (?:choose|pick|decide|recommend)|surprise me|up to you|your (?:call|choice|pick)"
        r"|(?:please )?(?:recommend|suggest) (?:something|anything|me something)(?: for me)?"
        r"|what (?:do|would) you (?:recommend|suggest)"
    ),
    "collection": (
        r"what (?:do you have|have you got|is in (?:the|your) collection|can i see|is here|objects do you have)"
        r"|(?:tell me|show me) (?:about )?(?:the|your) collection|introduce (?:the|your) collection"
    ),
    "unsure": (
        r"(?:i |i'm |im |i am )?(?:really |honestly )?"
        r"(?:(?:do not|don't|dont) know|not sure|no idea|no clue|not really|nothing(?: in particular| specific)?"
        r"|no preference|none|nope|no|nah)"
        r"(?: what (?:to|i should|i want to) (?:ask|see|look at|say|choose)| where to (?:start|begin)| yet)?"
        r"(?: really)?"
    ),
    "browse": r"anything(?: is fine| works| goes)?|whatever|(?:i'm |im |i am )?just (?:browsing|looking(?: around)?)",
    "greeting": r"(?:hi|hello|hey|good (?:morning|afternoon|evening))(?: there| yanyuan)?",
}
# The most informative cue decides how the curator answers: "你好，我不知道看
# 什么" is an unsure visitor who happens to be polite.
_UNDECIDED_PRIORITY = ("meta", "delegate", "collection", "unsure", "browse", "greeting")
_UNDECIDED_PARTICLES = re.compile(r"[啊呀吧呢嘛哈诶欸哦噢耶哇啦呗咯喔嗯额呃]")
_UNDECIDED_ZH_FILLER = re.compile(
    r"我|我也|也|还|就|那|那就|还是|所以|而且|但是|可是|反正|说实话|老实说|那个|这个|就是|其实|真的|暂时|好的|好"
)
_UNDECIDED_EN_FILLER = re.compile(
    r"um|uh|well|so|ok|okay|hmm|honestly|and|but|then|yeah|yes|really|just|i|i'm|im"
)
_UNDECIDED_ZH_CUES = tuple((kind, re.compile(pattern)) for kind, pattern in _UNDECIDED_ZH.items())
_UNDECIDED_EN_CUES = tuple((kind, re.compile(pattern)) for kind, pattern in _UNDECIDED_EN.items())


def undecided_opening_kind(text: str) -> str | None:
    """Name what an opening that carries no subject is doing, if it is one.

    Returns one of ``meta``, ``delegate``, ``collection``, ``unsure``,
    ``browse`` or ``greeting``; ``None`` means the text names something and is
    a real answer. Punctuation-only or filler-only text counts as ``unsure``:
    there is nothing in it to curate from.
    """

    raw = text.strip().casefold().replace("’", "'")
    if len(raw) > 60:
        return None
    english = bool(re.search(r"[a-z]", raw)) and not re.search(r"[一-鿿]", raw)
    kinds: list[str] = []
    if english:
        clauses = re.split(r"[^\w\s']+", raw)
        for clause in clauses:
            words = clause.split()
            if not words:
                continue
            found = _segment(words, " ", _UNDECIDED_EN_CUES, _UNDECIDED_EN_FILLER)
            if found is None:
                return None
            kinds.extend(found)
    else:
        value = re.sub(r"哈[喽啰]", "你好", raw)
        value = _UNDECIDED_PARTICLES.sub("", value)
        for clause in re.split(r"[\s，,。.!！?？、；;：:…~～\-—‘’“”\"'()（）]+", value):
            if not clause:
                continue
            found = _segment(list(clause), "", _UNDECIDED_ZH_CUES, _UNDECIDED_ZH_FILLER)
            if found is None:
                return None
            kinds.extend(found)
    for kind in _UNDECIDED_PRIORITY:
        if kind in kinds:
            return kind
    return "unsure"


def _segment(
    units: list[str],
    joiner: str,
    cues: tuple[tuple[str, re.Pattern[str]], ...],
    filler: re.Pattern[str],
) -> list[str] | None:
    """Cover a clause entirely with cues and fillers, or report that it can't be.

    Dynamic programming over unit boundaries rather than one big regex: a
    single pattern with repeated optional groups backtracks exponentially on a
    long near-miss, and this runs on untrusted visitor text. Here every regex
    only ever sees one bounded slice. English units are words so a cue never
    matches half of one ("no" inside "notes").
    """

    size = len(units)
    reached: list[list[str] | None] = [None] * (size + 1)
    reached[0] = []
    for begin in range(size):
        so_far = reached[begin]
        if so_far is None:
            continue
        for end in range(begin + 1, size + 1):
            if reached[end] is not None:
                continue
            piece = joiner.join(units[begin:end])
            if filler.fullmatch(piece):
                reached[end] = so_far
                continue
            kind = next((name for name, cue in cues if cue.fullmatch(piece)), None)
            if kind is not None:
                reached[end] = [*so_far, kind]
    return reached[size]
