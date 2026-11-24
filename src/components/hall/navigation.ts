import { walkableBounds, type HallLayout, type Obstacle, type TourStop } from "./layout";

export type NavigationPoint = [number, number, number];

export interface NavigationRoute {
  points: NavigationPoint[];
  /** Length of each segment, in the same order as `points.slice(1)`. */
  segmentLengths: number[];
  /** Sum of all segment lengths. */
  totalLength: number;
}

const POINT_EPSILON = 0.001;

/**
 * Keep familiar short flights, but give long routes enough time to remain
 * legible. Four scene units are approximately one second of guided travel;
 * the cap avoids making a deliberate jump across the exhibition feel stuck.
 */
export function routeFlightDuration(baseDuration: number, totalLength: number): number {
  const safeBase = Math.max(0, baseDuration);
  const distanceDuration = Math.max(0, totalLength) / 4;
  return Math.min(6, Math.max(safeBase, distanceDuration));
}

function distance(a: NavigationPoint, b: NavigationPoint): number {
  return Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
}

function samePoint(a: NavigationPoint, b: NavigationPoint): boolean {
  return distance(a, b) <= POINT_EPSILON;
}

function containingRoomIndex(layout: HallLayout, z: number): number {
  return layout.rooms.findIndex((room) => z <= room.zStart && z >= room.zEnd);
}

function destinationRoomIndex(layout: HallLayout, stop: TourStop): number {
  if (stop.chapterId) {
    const chapterIndex = layout.rooms.findIndex((room) => room.chapter.id === stop.chapterId);
    if (chapterIndex >= 0) return chapterIndex;
  }
  return containingRoomIndex(layout, stop.position[2]);
}

/**
 * Build an axis-safe guided route through the exhibition.
 *
 * Movements inside one chapter stay direct. Entering the hall or crossing a
 * chapter boundary first brings the camera back to the centre line, then
 * visits every room threshold and chapter entrance that lies between the
 * current position and the destination. That keeps a flight inside the
 * architectural circulation route instead of cutting diagonally through a
 * divider or across an artwork wall.
 */
export function buildGuidedRoute(
  from: NavigationPoint,
  stop: TourStop,
  layout: HallLayout,
): NavigationRoute {
  const destination: NavigationPoint = [...stop.position];
  const fromRoomIndex = containingRoomIndex(layout, from[2]);
  const toRoomIndex = destinationRoomIndex(layout, stop);
  const sameChapterSection = fromRoomIndex >= 0 && fromRoomIndex === toRoomIndex;

  if (sameChapterSection || layout.rooms.length === 0) {
    return createNavigationRoute(avoidObstacles([from, destination], layout));
  }

  const travellingForward = destination[2] < from[2];
  const between = (z: number) =>
    travellingForward
      ? z < from[2] - POINT_EPSILON && z > destination[2] + POINT_EPSILON
      : z > from[2] + POINT_EPSILON && z < destination[2] - POINT_EPSILON;

  const centreLineCandidates: NavigationPoint[] = [];
  for (const room of layout.rooms) {
    // Preserve both sides of a threshold when an older layout contains a
    // physical gap. In the continuous layout they coincide and are deduped.
    centreLineCandidates.push(
      [0, from[1], room.zStart],
      [0, from[1], room.entryViewpoint[2]],
      [0, from[1], room.zEnd],
    );
  }

  centreLineCandidates.sort((a, b) =>
    travellingForward ? b[2] - a[2] : a[2] - b[2],
  );

  const points: NavigationPoint[] = [from];
  // Always leave a side-wall observation point perpendicular to the wall
  // before advancing along the hall's longitudinal axis.
  points.push([0, from[1], from[2]]);
  points.push(...centreLineCandidates.filter((point) => between(point[2])));
  points.push(destination);

  return createNavigationRoute(avoidObstacles(points, layout));
}

/** Clearance kept between the camera's path and any freestanding footprint. */
const OBSTACLE_MARGIN = 0.45;
const MAX_DETOUR_DEPTH = 8;

type Box = { x0: number; x1: number; z0: number; z1: number };

function expanded(obstacle: Obstacle, margin: number): Box {
  return {
    x0: obstacle.x0 - margin,
    x1: obstacle.x1 + margin,
    z0: obstacle.z0 - margin,
    z1: obstacle.z1 + margin,
  };
}

function inside(point: NavigationPoint, box: Box): boolean {
  return point[0] > box.x0 && point[0] < box.x1 && point[2] > box.z0 && point[2] < box.z1;
}

/** Parameter along a→b where the segment enters the box, or null if it misses. */
function segmentEntry(a: NavigationPoint, b: NavigationPoint, box: Box): number | null {
  let enter = 0;
  let leave = 1;
  const axes: [number, number, number, number][] = [
    [a[0], b[0] - a[0], box.x0, box.x1],
    [a[2], b[2] - a[2], box.z0, box.z1],
  ];
  for (const [origin, delta, low, high] of axes) {
    if (Math.abs(delta) < POINT_EPSILON) {
      if (origin <= low || origin >= high) return null;
      continue;
    }
    let near = (low - origin) / delta;
    let far = (high - origin) / delta;
    if (near > far) [near, far] = [far, near];
    enter = Math.max(enter, near);
    leave = Math.min(leave, far);
    if (enter >= leave) return null;
  }
  return enter;
}

function pathLength(points: NavigationPoint[]): number {
  let total = 0;
  for (let index = 1; index < points.length; index += 1) total += distance(points[index - 1], points[index]);
  return total;
}

/**
 * The two corners that take a→b round the box on its shorter side. Travel
 * mostly along the hall passes beside the box; travel mostly across it
 * passes in front of or behind it. A side that would put the camera into a
 * wall is never chosen while the other is open.
 */
function detourCorners(
  a: NavigationPoint,
  b: NavigationPoint,
  box: Box,
  layout: HallLayout,
): NavigationPoint[] {
  const y = a[1];
  const alongHall = Math.abs(b[2] - a[2]) >= Math.abs(b[0] - a[0]);
  const candidates: NavigationPoint[][] = [];
  if (alongHall) {
    const [zNear, zFar] = a[2] > b[2] ? [box.z1, box.z0] : [box.z0, box.z1];
    for (const x of [box.x0, box.x1]) candidates.push([[x, y, zNear], [x, y, zFar]]);
  } else {
    const [xNear, xFar] = a[0] < b[0] ? [box.x0, box.x1] : [box.x1, box.x0];
    for (const z of [box.z1, box.z0]) candidates.push([[xNear, y, z], [xFar, y, z]]);
  }
  const walkable = (corner: NavigationPoint) => {
    const bounds = walkableBounds(layout, corner[2]);
    return corner[0] >= bounds.minX && corner[0] <= bounds.maxX;
  };
  const ranked = candidates
    .map((corners) => ({ corners, open: corners.every(walkable), length: pathLength([a, ...corners, b]) }))
    .sort((left, right) => Number(right.open) - Number(left.open) || left.length - right.length);
  return ranked[0].corners;
}

function detourSegment(
  a: NavigationPoint,
  b: NavigationPoint,
  layout: HallLayout,
  depth: number,
): NavigationPoint[] {
  if (depth >= MAX_DETOUR_DEPTH) return [b];
  let first: { box: Box; entry: number } | null = null;
  for (const obstacle of layout.obstacles ?? []) {
    const box = expanded(obstacle, OBSTACLE_MARGIN);
    // A stop placed deliberately close to a case is still reached.
    if (inside(a, box) || inside(b, box)) continue;
    const entry = segmentEntry(a, b, box);
    if (entry !== null && (!first || entry < first.entry)) first = { box, entry };
  }
  if (!first) return [b];
  const result: NavigationPoint[] = [];
  let previous = a;
  for (const waypoint of [...detourCorners(a, b, first.box, layout), b]) {
    result.push(...detourSegment(previous, waypoint, layout, depth + 1));
    previous = waypoint;
  }
  return result;
}

/**
 * Bend a waypoint route round screen walls and cases. The hall's
 * centre line now carries a screen wall in every chapter, so without this a
 * guided flight would pass straight through the key object.
 */
export function avoidObstacles(points: NavigationPoint[], layout: HallLayout): NavigationPoint[] {
  const obstacles = layout.obstacles ?? [];
  if (!points.length || !obstacles.length) return points;
  // A centre-line waypoint that falls inside a footprint's clearance (the
  // "step back to the axis" point beside a case, say) is dropped; the
  // detour round that footprint replaces it.
  const boxes = obstacles.map((obstacle) => expanded(obstacle, OBSTACLE_MARGIN));
  const kept = points.filter((point, index) =>
    index === 0 || index === points.length - 1 || !boxes.some((box) => inside(point, box)));
  const result: NavigationPoint[] = [kept[0]];
  for (let index = 1; index < kept.length; index += 1) {
    result.push(...detourSegment(result[result.length - 1], kept[index], layout, 0));
  }
  return result;
}

/** Create a measured route while removing adjacent duplicate waypoints. */
export function createNavigationRoute(points: NavigationPoint[]): NavigationRoute {
  const deduped: NavigationPoint[] = [];
  for (const point of points) {
    const copy: NavigationPoint = [...point];
    if (!deduped.length || !samePoint(deduped[deduped.length - 1], copy)) {
      deduped.push(copy);
    }
  }

  // A route always needs a sampleable point, even for a zero-distance cut.
  if (!deduped.length) deduped.push([0, 0, 0]);

  const segmentLengths = deduped.slice(1).map((point, index) => distance(deduped[index], point));
  const totalLength = segmentLengths.reduce((sum, length) => sum + length, 0);
  return { points: deduped, segmentLengths, totalLength };
}

/**
 * Sample a route by cumulative travelled distance, not by waypoint count.
 * This gives the camera one continuous velocity through every threshold.
 */
export function sampleNavigationRoute(
  route: NavigationRoute,
  travelledDistance: number,
): NavigationPoint {
  if (route.points.length === 1 || route.totalLength <= POINT_EPSILON) {
    return [...route.points[route.points.length - 1]];
  }

  const clampedDistance = Math.max(0, Math.min(route.totalLength, travelledDistance));
  let consumed = 0;

  for (let index = 0; index < route.segmentLengths.length; index += 1) {
    const segmentLength = route.segmentLengths[index];
    const segmentEnd = consumed + segmentLength;
    if (clampedDistance <= segmentEnd || index === route.segmentLengths.length - 1) {
      const localProgress = segmentLength <= POINT_EPSILON
        ? 1
        : (clampedDistance - consumed) / segmentLength;
      const start = route.points[index];
      const end = route.points[index + 1];
      return [
        start[0] + (end[0] - start[0]) * localProgress,
        start[1] + (end[1] - start[1]) * localProgress,
        start[2] + (end[2] - start[2]) * localProgress,
      ];
    }
    consumed = segmentEnd;
  }

  return [...route.points[route.points.length - 1]];
}
