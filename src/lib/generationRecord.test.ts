import assert from "node:assert/strict";
import test from "node:test";

import {
  generationFallbackNotice,
  generationProviderLabel,
} from "./generationRecord";

test("deterministic fallback discloses the actual text route and configured-model boundary", () => {
  assert.equal(
    generationProviderLabel("deterministic_fallback"),
    "deterministic_fallback",
  );
  assert.equal(
    generationFallbackNotice("deterministic_fallback"),
    "DeepSeek 框架阶段超时，策展文本使用确定性回退；模型字段仅表示配置模型。",
  );
});

test("successful model records do not show a fallback warning", () => {
  assert.equal(generationProviderLabel("deepseek"), "deepseek");
  assert.equal(generationFallbackNotice("deepseek"), null);
  assert.equal(generationProviderLabel(undefined), "旧记录未保存");
});
