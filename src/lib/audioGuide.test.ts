import assert from "node:assert/strict";
import test from "node:test";
import {
  API_BASE_URL,
  resolveApiAssetUrl,
  resolveAudioGuideUrl,
  resolveObjectImageUrl,
} from "./api";

/* These run under node, where `window` is undefined, so every call below takes
   the server-rendering path — the one that produced the broken images. */

/** Browser-bound URLs are relative, so resolve them against a stand-in origin. */
const VISITOR_ORIGIN = "https://visitor.example";
const asVisitorSees = (path: string) => new URL(path, VISITOR_ORIGIN);

test("audio guide URL encodes exhibition and chapter references", () => {
  const url = asVisitorSees(resolveAudioGuideUrl("展览/一", {
    kind: "chapter",
    ref: "第一章 & 序",
  }));

  assert.equal(url.pathname, "/api/exhibitions/%E5%B1%95%E8%A7%88%2F%E4%B8%80/audio-guide");
  assert.equal(url.searchParams.get("kind"), "chapter");
  assert.equal(url.searchParams.get("ref"), "第一章 & 序");
});

test("lobby and epilogue URLs never send an irrelevant ref", () => {
  for (const kind of ["lobby", "epilogue"] as const) {
    const url = asVisitorSees(resolveAudioGuideUrl("exhibition-1", { kind, ref: "ignored" }));
    assert.equal(url.searchParams.get("kind"), kind);
    assert.equal(url.searchParams.has("ref"), false);
  }
});

test("the server still fetches the API over loopback", () => {
  // The split is deliberate and must survive: this base is for requests the
  // server makes itself, and it is the only one allowed to be absolute.
  assert.equal(API_BASE_URL, "http://127.0.0.1:9001");
});

test("URLs the browser resolves never carry the server's loopback base", () => {
  // Rendered into HTML, a loopback URL points at the visitor's own machine.
  // Nothing listens there, so the image fails before a request leaves the
  // device and no trace of it reaches the server log.
  const browserBound = [
    resolveObjectImageUrl("cma:1939.205", 512),
    resolveObjectImageUrl("met:37451"),
    resolveAudioGuideUrl("exhibition-1", { kind: "lobby" }),
    resolveApiAssetUrl("/generated/posters/p1.webp"),
    resolveApiAssetUrl("generated/posters/p2.webp"),
  ];

  for (const url of browserBound) {
    assert.ok(
      url.startsWith("/"),
      `expected a root-relative URL for the browser, got ${url}`,
    );
    assert.doesNotMatch(
      url,
      /127\.0\.0\.1|localhost|:9001/,
      `a server-only address leaked into a browser URL: ${url}`,
    );
  }
});

test("an already-absolute asset URL is passed through untouched", () => {
  for (const absolute of [
    "https://openaccess-cdn.example/x.jpg",
    "data:image/webp;base64,AAAA",
    "blob:https://visitor.example/abc",
  ]) {
    assert.equal(resolveApiAssetUrl(absolute), absolute);
  }
});

test("object image URL keeps the id in one path segment", () => {
  const url = asVisitorSees(resolveObjectImageUrl("cma:1939.205", 512));
  assert.equal(url.pathname, "/api/images/cma%3A1939.205");
  assert.equal(url.searchParams.get("w"), "512");
});
