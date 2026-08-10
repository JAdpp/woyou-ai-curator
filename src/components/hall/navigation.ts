import type { HallLayout, TourStop } from "./layout";

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
    return createNavigationRoute([from, destination]);
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

  return createNavigationRoute(points);
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
