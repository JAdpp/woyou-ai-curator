from __future__ import annotations

import json
import re

from app.query_plan_review import (
    V4_SEARCH_KEYS,
    parse_query_plan_review_v4,
    query_plan_review_prompt,
    query_plan_review_prompt_compact_zh,
)
from app.retrieval_agent import RetrievalQueryPlan


def _skeleton():
    prompt = query_plan_review_prompt_compact_zh()
    return json.loads(prompt[prompt.index("\n{") + 1:])


def test_compact_prompt_is_selected_only_for_chinese_and_within_requested_length():
    compact = query_plan_review_prompt_compact_zh()
    assert compact == query_plan_review_prompt() == query_plan_review_prompt("zh")
    assert compact != query_plan_review_prompt("en")
    assert query_plan_review_prompt("en").startswith("Independently compile")
    prose = compact.split("\n{", 1)[0]
    assert 1200 <= len(re.findall(r"[\u4e00-\u9fff]", prose)) <= 1800


def test_compact_skeleton_has_exact_v4_top_search_and_intent_variant_keys():
    output = _skeleton()
    assert set(output) == {"searchPlan", "intents"}
    assert set(output["searchPlan"]) == V4_SEARCH_KEYS
    base = {"id", "sourceQuote", "type", "text"}
    expected = {
        "admission": base | {"evidenceScope"},
        "set_witness": base | {"evidenceScope", "minWitnesses", "quantifierOrigin"},
        "editorial": base,
        "preference": base,
    }
    assert {row["type"] for row in output["intents"]} == set(expected)
    assert len({row["id"] for row in output["intents"]}) == 4
    for row in output["intents"]:
        assert set(row) == expected[row["type"]]
        assert all(value is not None for value in row.values())
    assert not {"mandatoryPerObjectPredicates", "exhibitionSetRequirements", "visualPredicateIds",
                "selectionRationaleConstraints", "editorialConstraints"} & set(output["searchPlan"])


def test_compact_skeleton_uses_the_existing_compiler_not_a_new_schema():
    output = _skeleton()
    # Turn the placeholder quotations into a synthetic schema fixture only.
    # This is not a model invocation or evidence of real semantic compliance.
    question = "；".join(row["sourceQuote"] for row in output["intents"])
    draft = RetrievalQueryPlan(valid=True, in_collection_scope=True, search_queries=("fixture",),
                               semantic_query="fixture query", mandatory_predicates=("fixture class",),
                               evidence_mode="visual_observation")
    outcome = parse_query_plan_review_v4(output, question=question, draft=draft)
    assert outcome.reviewed
    assert len(outcome.plan.mandatory_predicates) == 1
    assert outcome.plan.visual_predicate_ids == ()  # The sample admission is record-bound.
    assert len(outcome.plan.exhibition_set_requirements) == 1
    assert len(outcome.plan.editorial_constraints) == len(outcome.plan.selection_constraints) == 1
    assert outcome.diagnostics["semanticFaithfulnessProven"] is False


def test_compact_prompt_preserves_general_intent_and_evidence_boundaries():
    prompt = query_plan_review_prompt_compact_zh()
    for text in ("只根据 visitorQuestion", "不是需求、证据、标准答案", "每条条件只使用一种证据权限",
                 "OR", "AND", "同一件对象", "不是保证两组都必须找到例子",
                 "只有原题明确要求每组都有例子", "对全部相关文案", "所提供记录未能确认",
                 "地区或文化来源、材质身份、年代", "不能标为纯可见", "不要按每个分类重复复制",
                 "不能要求同一件属于全部类别", "非适用的键必须省略，不要填 null",
                 "hardFilters 必须原样复制 recallDraft.hardFilters", "minWitnesses 只能为1至3",
                 "不可照抄", "可见部分A位于可见部分B之上"):
        assert text in prompt
    for topic in ("猫", "狗", "扇", "香器", "香水瓶", "市场", "雨伞"):
        assert topic not in prompt
    assert not re.search(r"\b(?:cat|dog|fan|incense|umbrella|market)\b", prompt, re.I)
