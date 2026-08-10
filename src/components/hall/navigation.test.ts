import assert from "node:assert/strict";
import test from "node:test";
import type { HallLayout, RoomLayout, TourStop } from "./layout";
import {
  buildGuidedRoute,
  createNavigationRoute,
  routeFlightDuration,
  sampleNavigationRoute,
  type NavigationPoint,
} from "./navigation";

function room(index: number, zStart: number, zEnd: number): RoomLayout {
  return {
    chapter: {
      id: `chapter-${index}`,
      order: index,
      title: `Chapter ${index}`,
      leadIn: "",
      itemIds: [],
    },
    index,
    width: 10,
    depth: zStart - zEnd,
    center: [0, 2.3, (zStart + zEnd) / 2],
    zStart,
    zEnd,
    entryViewpoint: [0, 1.6, zStart - 1.5],
    artworks: [],
    exitZ: index < 2 ? zEnd : null,
  };
}

function layout(): HallLayout {
  const rooms = [room(0, 0, -8), room(1, -8, -16), room(2, -16, -24)];
  return {
    rooms,
    artworks: [],
    stops: [],
    totalDepth: 24,
    lobbyViewpoint: [0, 1.6, 5.5],
    epilogueViewpoint: [0, 1.6, -25],
    design: {
      spaceForm: "hall",
      wallColor: "#eeeeee",
      floorColor: "#111111",
      accentColor: "#a4512c",
      ceilingColor: "#f4f0e8",
      lightTemperature: 4000,
      lightIntensity: 1,
      mood: "neutral",
      frameStyle: "thin_dark",
      pacing: "ant",
      typography: "serif",
    },
  };
}

function artworkStop(chapterIndex: number, point: NavigationPoint): TourStop {
  return {
    kind: "artwork",
    position: point,
    target: [chapterIndex % 2 ? -5 : 5, 1.8, point[2]],
    chapterId: `chapter-${chapterIndex}`,
    itemId: `item-${chapterIndex}`,
    label: "Artwork",
  };
}

test("keeps observation-stop movement direct inside one chapter section", () => {
  const route = buildGuidedRoute([2, 1.6, -3], artworkStop(0, [-2, 1.6, -6]), layout());
  assert.deepEqual(route.points, [
    [2, 1.6, -3],
    [-2, 1.6, -6],
  ]);
});

test("returns to the centre line and crosses every forward chapter threshold", () => {
  const route = buildGuidedRoute([2.5, 1.6, -5], artworkStop(2, [-2, 1.6, -20]), layout());

  assert.deepEqual(route.points[1], [0, 1.6, -5]);
  assert.ok(route.points.some((point) => point[2] === -8), "first chapter boundary is visited");
  assert.ok(route.points.some((point) => point[2] === -9.5), "second chapter entrance is visited");
  assert.ok(route.points.some((point) => point[2] === -16), "target chapter boundary is visited");
  assert.ok(route.points.some((point) => point[2] === -17.5), "target chapter entrance is visited");

  const centreLineZ = route.points.slice(1, -1).map((point) => point[2]);
  assert.deepEqual(centreLineZ, [...centreLineZ].sort((a, b) => b - a));
  assert.ok(route.points.slice(1, -1).every((point) => point[0] === 0));
});

test("orders thresholds correctly when navigating back to an earlier chapter", () => {
  const route = buildGuidedRoute([-2.5, 1.6, -21], artworkStop(0, [2, 1.6, -4]), layout());

  assert.deepEqual(route.points[1], [0, 1.6, -21]);
  const centreLineZ = route.points.slice(1, -1).map((point) => point[2]);
  assert.deepEqual(centreLineZ, [...centreLineZ].sort((a, b) => a - b));
  assert.ok(centreLineZ.includes(-16));
  assert.ok(centreLineZ.includes(-9.5));
  assert.ok(centreLineZ.includes(-8));
});

test("routes from the lobby through the first chapter entrance", () => {
  const route = buildGuidedRoute([1, 1.6, 5.5], artworkStop(0, [-2, 1.6, -4]), layout());
  assert.deepEqual(route.points[1], [0, 1.6, 5.5]);
  assert.ok(route.points.some((point) => point[2] === 0));
  assert.ok(route.points.some((point) => point[2] === -1.5));
});

test("samples by accumulated distance rather than equal time per segment", () => {
  const route = createNavigationRoute([
    [0, 0, 0],
    [10, 0, 0],
    [10, 0, -2],
  ]);
  assert.equal(route.totalLength, 12);
  assert.deepEqual(sampleNavigationRoute(route, 6), [6, 0, 0]);
  assert.deepEqual(sampleNavigationRoute(route, 11), [10, 0, -1]);
  assert.deepEqual(sampleNavigationRoute(route, 99), [10, 0, -2]);
});

test("deduplicates coincident boundaries in a continuous layout", () => {
  const route = createNavigationRoute([
    [0, 1.6, 0],
    [0, 1.6, -8],
    [0, 1.6, -8],
    [0, 1.6, -10],
  ]);
  assert.equal(route.points.length, 3);
  assert.equal(route.totalLength, 10);
});

test("keeps the existing pacing duration for a short route", () => {
  assert.equal(routeFlightDuration(1.6, 4), 1.6);
});

test("extends a long route at approximately four scene units per second", () => {
  assert.equal(routeFlightDuration(1.6, 16), 4);
});

test("caps long guided flights at six seconds", () => {
  assert.equal(routeFlightDuration(2.4, 100), 6);
});
