import assert from "node:assert/strict";
import test from "node:test";
import { resolveAudioGuideUrl } from "./api";

test("audio guide URL encodes exhibition and chapter references", () => {
  const url = new URL(resolveAudioGuideUrl("展览/一", {
    kind: "chapter",
    ref: "第一章 & 序",
  }));

  assert.equal(url.pathname, "/api/exhibitions/%E5%B1%95%E8%A7%88%2F%E4%B8%80/audio-guide");
  assert.equal(url.searchParams.get("kind"), "chapter");
  assert.equal(url.searchParams.get("ref"), "第一章 & 序");
});

test("lobby and epilogue URLs never send an irrelevant ref", () => {
  for (const kind of ["lobby", "epilogue"] as const) {
    const url = new URL(resolveAudioGuideUrl("exhibition-1", { kind, ref: "ignored" }));
    assert.equal(url.searchParams.get("kind"), kind);
    assert.equal(url.searchParams.has("ref"), false);
  }
});

