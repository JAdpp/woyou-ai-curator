import assert from "node:assert/strict";
import test from "node:test";
import {
  EMPTY_EPILOGUE_CONVERSATION,
  epilogueConversationReducer,
  toEpilogueChatHistory,
  trimEpilogueDraft,
} from "./epilogueConversationState";

test("trims a visitor draft and rejects whitespace in the reducer", () => {
  assert.equal(trimEpilogueDraft("  我想到了一件事。\n"), "我想到了一件事。");
  const state = epilogueConversationReducer(EMPTY_EPILOGUE_CONVERSATION, {
    type: "append_turn",
    id: "visitor-empty",
    role: "user",
    content: "   ",
  });
  assert.equal(state, EMPTY_EPILOGUE_CONVERSATION);
});

test("keeps a selected opening question visible but out of API history", () => {
  const state = epilogueConversationReducer(EMPTY_EPILOGUE_CONVERSATION, {
    type: "select_open_question",
    id: "opening-1",
    question: "这件作品改变了你对仪式的看法吗？",
  });

  assert.equal(state.turns[0]?.role, "assistant");
  assert.equal(
    state.turns[0]?.content,
    "如果你愿意，我们可以从这则题笺继续：这件作品改变了你对仪式的看法吗？",
  );
  assert.equal(state.selectedOpenQuestion, "这件作品改变了你对仪式的看法吗？");
  assert.deepEqual(toEpilogueChatHistory(state.turns), []);
});

test("exports only completed visitor and curator pairs after the opening", () => {
  let state = epilogueConversationReducer(EMPTY_EPILOGUE_CONVERSATION, {
    type: "select_open_question",
    id: "opening-1",
    question: "你会怎样回答？",
  });
  state = epilogueConversationReducer(state, {
    type: "append_turn",
    id: "visitor-1",
    role: "user",
    content: "我会先看材料。",
  });
  state = epilogueConversationReducer(state, {
    type: "append_turn",
    id: "assistant-1",
    role: "assistant",
    content: "那你最信任哪一条材料？",
  });

  assert.deepEqual(toEpilogueChatHistory(state.turns), [
    { role: "user", content: "我会先看材料。" },
    { role: "assistant", content: "那你最信任哪一条材料？" },
  ]);
});

test("does not replace the opening anchor after a visitor has answered", () => {
  const answered = {
    turns: [{ id: "visitor-1", role: "user" as const, content: "我的回答" }],
    selectedOpenQuestion: null,
    suggestedReplies: [],
  };
  const state = epilogueConversationReducer(answered, {
    type: "select_open_question",
    id: "opening-2",
    question: "另一则题笺",
  });
  assert.equal(state, answered);
});

test("keeps only the four most recent completed pairs for the API", () => {
  const turns = Array.from({ length: 10 }, (_, index) => ({
    id: `turn-${index}`,
    role: index % 2 === 0 ? "user" as const : "assistant" as const,
    content: `turn ${index}`,
  }));
  const history = toEpilogueChatHistory(turns);
  assert.equal(history.length, 8);
  assert.deepEqual(history[0], { role: "user", content: "turn 2" });
  assert.deepEqual(history[7], { role: "assistant", content: "turn 9" });
});
