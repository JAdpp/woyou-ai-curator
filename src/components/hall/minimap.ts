import { DOORWAY_WIDTH, type HallLayout } from "./layout";

/**
 * Floor-plan projection for the hall's minimap.
 *
 * The route runs down -Z, so the plan is drawn lying on its side: the entrance
 * is on the left and the visit moves right. Facing right, a visitor's left
 * hand is up, so the left wall (x < 0) is the top edge of the plan. Pure
 * arithmetic, kept apart from React and three.js like `layout.ts`.
 */

/**
 * Where the camera is and which way it faces, written every frame from inside
 * the canvas and read by the minimap outside it, without a React render.
 */
export interface CameraPose {
  x: number;
  z: number;
  forwardX: number;
  forwardZ: number;
}

export interface PlanRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PlanRoom extends PlanRect {
  chapterId: string;
  index: number;
}

export interface PlanArtwork extends PlanRect {
  itemId: string;
  tourIndex: number;
  side: "left" | "right";
}

export interface HallPlan {
  width: number;
  height: number;
  /** Plan units per metre. */
  scale: number;
  hall: PlanRect;
  approach: PlanRect;
  rooms: PlanRoom[];
  artworks: PlanArtwork[];
  project: (x: number, z: number) => [number, number];
  /** Plan-space heading, in degrees, for a world-space forward vector. */
  heading: (forwardX: number, forwardZ: number) => number;
}

/** How far beyond the entrance door the plan reaches, to include the lobby. */
const APPROACH_SHOWN = 6;
/** Artworks are drawn as short bars lying along their wall. */
const ARTWORK_DEPTH = 0.55;

export function buildHallPlan(
  layout: HallLayout,
  width: number,
  height: number,
  padding = 6,
): HallPlan {
  const firstRoom = layout.rooms[0];
  const lastRoom = layout.rooms[layout.rooms.length - 1];
  const hallWidth = firstRoom?.width ?? 11;
  const zEntrance = firstRoom?.zStart ?? 0;
  const zEnd = lastRoom?.zEnd ?? -8;
  const zMax = Math.max(zEntrance + APPROACH_SHOWN, layout.lobbyViewpoint[2] + 0.5);
  const span = Math.max(1, zMax - zEnd);

  const scale = Math.max(
    0.0001,
    Math.min((width - padding * 2) / span, (height - padding * 2) / hallWidth),
  );
  const offsetX = (width - span * scale) / 2;
  const offsetY = (height - hallWidth * scale) / 2;

  const project = (x: number, z: number): [number, number] => [
    offsetX + (zMax - z) * scale,
    offsetY + (x + hallWidth / 2) * scale,
  ];

  const rectBetween = (x0: number, z0: number, x1: number, z1: number): PlanRect => {
    const [ax, ay] = project(x0, z0);
    const [bx, by] = project(x1, z1);
    return {
      x: Math.min(ax, bx),
      y: Math.min(ay, by),
      width: Math.abs(bx - ax),
      height: Math.abs(by - ay),
    };
  };

  const half = hallWidth / 2;
  const hall = rectBetween(-half, zEntrance, half, zEnd);
  const approach = rectBetween(-DOORWAY_WIDTH / 2, zMax, DOORWAY_WIDTH / 2, zEntrance);
  const rooms = layout.rooms.map((room) => ({
    ...rectBetween(-half, room.zStart, half, room.zEnd),
    chapterId: room.chapter.id,
    index: room.index,
  }));

  const artworks = layout.artworks.map((placement) => {
    const along = placement.width / 2;
    const z = placement.position[2];
    const wallX = placement.side === "left" ? -half : half;
    const innerX = placement.side === "left" ? -half + ARTWORK_DEPTH : half - ARTWORK_DEPTH;
    return {
      ...rectBetween(wallX, z + along, innerX, z - along),
      itemId: placement.itemId,
      tourIndex: placement.tourIndex,
      side: placement.side,
    };
  });

  // World forward (fx, fz) maps to plan (-fz, fx): -Z is plan-right, -X is up.
  const heading = (forwardX: number, forwardZ: number) =>
    (Math.atan2(forwardX, -forwardZ) * 180) / Math.PI;

  return { width, height, scale, hall, approach, rooms, artworks, project, heading };
}
