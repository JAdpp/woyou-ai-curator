import assert from "node:assert/strict";
import test from "node:test";
import type { HallLayout, Obstacle, RoomLayout, TourStop } from "./layout";
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
    screen: { z: zEnd + 3, width: 4, height: 4.3, thickness: 0.32, heroItemId: null },
  };
}

function layout(obstacles: Obstacle[] = []): HallLayout {
  const rooms = [room(0, 0, -8), room(1, -8, -16), room(2, -16, -24)];
  return {
    epilogueZone: { zStart: -24, zEnd: -32 },
    endZ: -32,
    obstacles,
    title: "T",
    language: "zh",
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

/** Every point the camera passes, sampled every 5 cm along the route. */
function sampled(route: ReturnType<typeof buildGuidedRoute>): NavigationPoint[] {
  const points: NavigationPoint[] = [];
  for (let travelled = 0; travelled <= route.totalLength; travelled += 0.05) {
    points.push(sampleNavigationRoute(route, travelled));
  }
  return points;
}

function passesThrough(route: ReturnType<typeof buildGuidedRoute>, obstacle: Obstacle): boolean {
  return sampled(route).some(([x, , z]) =>
    x > obstacle.x0 && x < obstacle.x1 && z > obstacle.z0 && z < obstacle.z1);
}

const screenWall: Obstacle = { x0: -2, x1: 2, z0: -5.16, z1: -4.84, kind: "screen" };

test("a flight down the axis goes round a screen wall instead of through it", () => {
  const route = buildGuidedRoute([0, 1.6, -1], artworkStop(1, [0, 1.6, -12]), layout([screenWall]));
  assert.equal(passesThrough(route, screenWall), false);
  assert.ok(route.points.some(([x]) => Math.abs(x) > 2), "the route never stepped aside");
});

test("crossing behind a screen within a chapter takes the nearer side", () => {
  const route = buildGuidedRoute([3, 1.6, -2], artworkStop(0, [2.5, 1.6, -7]), layout([screenWall]));
  assert.equal(passesThrough(route, screenWall), false);
  assert.ok(route.points.every(([x]) => x > 0), "the detour swung to the far side");
});

test("a stop placed just in front of a case is still reached directly", () => {
  const displayCase: Obstacle = { x0: -0.8, x1: 0.8, z0: -3.25, z1: -2.75, kind: "case" };
  const route = buildGuidedRoute([0, 1.6, -1], artworkStop(0, [0, 1.6, -2.4]), layout([displayCase]));
  assert.deepEqual(route.points[route.points.length - 1], [0, 1.6, -2.4]);
  assert.equal(passesThrough(route, displayCase), false);
});
