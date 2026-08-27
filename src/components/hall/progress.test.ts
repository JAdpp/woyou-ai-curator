import assert from "node:assert/strict";
import test from "node:test";
import type { TourStop } from "./layout";
import {
  clampStopIndex,
  decideInitialHallMode,
  exhibitionViewHref,
  parseHallSession,
  stopAnchor,
  stopIndexFromUrl,
} from "./progress";

const stops = [
  { kind: "lobby", label: "Lobby" },
  { kind: "chapter", chapterId: "one", label: "Chapter" },
  { kind: "artwork", itemId: "object:42", label: "Object" },
  { kind: "epilogue", label: "Epilogue" },
] as TourStop[];

test("clampStopIndex rejects invalid and out-of-range progress", () => {
  assert.equal(clampStopIndex(stops.length, "nope"), 0);
  assert.equal(clampStopIndex(stops.length, -4), 0);
  assert.equal(clampStopIndex(stops.length, 99), 3);
  assert.equal(clampStopIndex(stops.length, 2.8), 2);
});

test("semantic anchors restore the matching chapter, object, or epilogue", () => {
  assert.equal(stopAnchor(stops[2]), "item-object:42");
  assert.equal(
    stopIndexFromUrl(stops, new URL("https://example.test/e/x?stop=1#item-object%3A42")),
    2,
  );
  assert.equal(
    stopIndexFromUrl(stops, new URL("https://example.test/e/x#epilogue")),
    3,
  );
});

test("query progress is used when no semantic anchor is present", () => {
  assert.equal(stopIndexFromUrl(stops, new URL("https://example.test/e/x?stop=2")), 2);
  assert.equal(stopIndexFromUrl(stops, new URL("https://example.test/e/x?stop=bad"), 1), 1);
});

test("view links preserve unrelated query parameters and encode progress", () => {
  assert.equal(
    exhibitionViewHref("https://example.test/exhibitions/x?source=share", "text", 2, stops[2]),
    "/exhibitions/x?source=share&view=text&stop=2#item-object%3A42",
  );
});

test("session state is parsed defensively", () => {
  assert.deepEqual(
    parseHallSession('{"stopIndex":99,"lastMode":"2d","autoEntered":true}', stops.length),
    { stopIndex: 3, lastMode: "2d", autoEntered: true },
  );
  assert.equal(parseHallSession("not-json", stops.length), null);
  assert.equal(parseHallSession('{"lastMode":"text"}', stops.length), null);
});

test("desktop WebGL auto-enters once while touch defaults to 2D", () => {
  assert.deepEqual(decideInitialHallMode({
    forceTextView: false,
    requestedView: null,
    webglAvailable: true,
    touchPrimary: false,
    session: null,
  }), { enter: true, autoEntered: true });
  assert.deepEqual(decideInitialHallMode({
    forceTextView: false,
    requestedView: null,
    webglAvailable: true,
    touchPrimary: true,
    session: null,
  }), { enter: false, autoEntered: false });
});

test("explicit text is permanent and an explicit 3D choice can override touch default", () => {
  assert.equal(decideInitialHallMode({
    forceTextView: true,
    requestedView: "3d",
    webglAvailable: true,
    touchPrimary: false,
    session: null,
  }).enter, false);
  assert.equal(decideInitialHallMode({
    forceTextView: false,
    requestedView: "3d",
    webglAvailable: true,
    touchPrimary: true,
    session: null,
  }).enter, true);
});

test("a saved 2D choice is not overwritten by desktop auto-entry", () => {
  assert.equal(decideInitialHallMode({
    forceTextView: false,
    requestedView: null,
    webglAvailable: true,
    touchPrimary: false,
    session: { stopIndex: 2, lastMode: "2d", autoEntered: true },
  }).enter, false);
});
