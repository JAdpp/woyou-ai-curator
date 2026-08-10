import assert from "node:assert/strict";
import test from "node:test";
import {
  ApiError,
  normalizeEpilogueChatResponse,
  resolveEpilogueChatUrl,
} from "./api";

test("epilogue chat URL encodes the exhibition id as one path segment", () => {
  const url = new URL(resolveEpilogueChatUrl("展览/一"));
  assert.equal(url.pathname, "/api/exhibitions/%E5%B1%95%E8%A7%88%2F%E4%B8%80/epilogue-chat");
});

test("normalizes the backend answer and snake-case evidence response", () => {
  const response = normalizeEpilogueChatResponse({
    answer: "可以从这件器物的使用情境继续想。",
    citations: [{
      item_id: "item-7",
      evidence_ids: ["evidence-2", "evidence-3"],
    }],
    suggested_prompts: ["它与另一件展品有什么差别？", "这种解释还有别的可能吗？"],
    mode: "deepseek",
    notice: "模型基于本展材料作答。",
  });

  assert.equal(response.message, "可以从这件器物的使用情境继续想。");
  assert.deepEqual(response.citations, [{
    itemId: "item-7",
    objectId: undefined,
    evidenceIds: ["evidence-2", "evidence-3"],
    label: undefined,
  }]);
  assert.deepEqual(response.suggestedReplies, [
    "它与另一件展品有什么差别？",
    "这种解释还有别的可能吗？",
  ]);
  assert.equal(response.mode, "deepseek");
  assert.equal(response.notice, "模型基于本展材料作答。");
});

test("accepts the API camel-case response contract", () => {
  const response = normalizeEpilogueChatResponse({
    message: "回应",
    citations: [{ objectId: "object-9", evidenceIds: ["ev-1"], label: "展品九" }],
    suggestedPrompts: ["继续", "换一件展品比较"],
  });
  assert.equal(response.message, "回应");
  assert.equal(response.citations[0]?.objectId, "object-9");
  assert.deepEqual(response.suggestedReplies, ["继续", "换一件展品比较"]);
});

test("rejects an empty or unreadable model response", () => {
  assert.throws(
    () => normalizeEpilogueChatResponse({ answer: "  " }),
    (error: unknown) => error instanceof ApiError && error.status === 502,
  );
});
