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
