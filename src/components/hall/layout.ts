import type { Chapter, Exhibition, ExhibitionItem, SpaceDesignSpec } from "@/lib/types";

/**
 * Pure geometry for the generated hall.
 *
 * Kept free of three.js so the layout can be reasoned about and tested on its
 * own: the renderer just draws whatever this produces.
 *
 * The plan is one continuous route down -Z. Each chapter is a zone with a
 * freestanding screen wall (对景墙) across its axis carrying the chapter's key
 * object, so the view down the hall always ends on something. The narrative
 * maps onto space: objects that lead up to the key object hang on the side
 * walls approaching the screen, objects that follow it hang behind the
 * screen, and the visitor walks round the screen to continue. After the last
 * chapter a short closing room ends on a
 * retrospective wall (回望墙) showing every object of the visit.
 */

export const WALL_HEIGHT = 4.6;
export const DOORWAY_WIDTH = 2.4;
export const DOORWAY_HEIGHT = 3;
/** Eye height for both camera modes. */
export const EYE_HEIGHT = 1.6;
/** How far in front of an artwork's surface the guided camera parks. */
export const VIEW_DISTANCE = 3.1;
/** Centre height of every hung object. */
export const HANG_HEIGHT = EYE_HEIGHT + 0.28;
/** Freestanding screen walls stop short of the ceiling so they read as free. */
export const SCREEN_HEIGHT = 4.3;
export const SCREEN_THICKNESS = 0.32;
/** Mounts stand this far off the surface behind them. */
export const MOUNT_OFFSET = 0.06;

const SIDE_SPACING = 3.1;
/** From a threshold to the first side-wall object. */
const ZONE_EDGE = 2.0;
/** Side-wall objects keep this far along the axis from a screen wall. */
const SCREEN_CLEARANCE = 1.9;
/** Threshold to screen: room for the framed view and the viewer. */
const FRONT_MIN = 7.2;
/** Screen to the next threshold: enough to walk round and see what follows. */
const BACK_MIN = 3.6;
const EPILOGUE_DEPTH = 8;
const WALKING_INSET = 0.55;
const ENTRANCE_APPROACH_DEPTH = 7;
/** How deep a case on a plinth projects in front of its wall. */
export const CASE_DEPTH = 0.78;
export interface ArtworkPlacement {
  itemId: string;
  item: ExhibitionItem;
  chapterId: string;
  /** Index across the whole exhibition, matching the guided tour order. */
  tourIndex: number;
  /** Centre of the artwork's hanging surface. */
  position: [number, number, number];
  /** Y rotation so the artwork faces the visitor. */
  rotationY: number;
  /** Where the guided camera stops to look at it. */
  viewpoint: [number, number, number];
  /** The box the object is fitted into; the image keeps its own aspect. */
  width: number;
  height: number;
  /** Which surface it hangs on. */
  side: "left" | "right" | "screen";
}

export interface ScreenWall {
  /** Centre of the wall along the route. */
  z: number;
  width: number;
  height: number;
  thickness: number;
  /** The chapter's key object, on the face toward the chapter entrance. */
  heroItemId: string | null;
}

/** A footprint the visitor and the guided camera must go round. */
export interface Obstacle {
  x0: number;
  x1: number;
  z0: number;
  z1: number;
  kind: "screen" | "case";
}

export interface RoomLayout {
  chapter: Chapter;
  index: number;
  width: number;
  depth: number;
  /** Room centre. */
  center: [number, number, number];
  zStart: number;
  zEnd: number;
  /** Camera stop for the chapter: just inside, facing the screen wall. */
  entryViewpoint: [number, number, number];
  artworks: ArtworkPlacement[];
  /** Narrative threshold to the next section, if any. */
  exitZ: number | null;
  screen: ScreenWall;
}

export interface HallLayout {
  rooms: RoomLayout[];
  artworks: ArtworkPlacement[];
  /** Ordered camera stops: lobby, chapter entries, artworks, epilogue. */
  stops: TourStop[];
  totalDepth: number;
  lobbyViewpoint: [number, number, number];
  epilogueViewpoint: [number, number, number];
  design: SpaceDesignSpec;
  /** The closing room after the last chapter. */
  epilogueZone: { zStart: number; zEnd: number };
  /** The final wall of the hall: the retrospective wall. */
  endZ: number;
  obstacles: Obstacle[];
  /** For the wall graphics; the overlay carries the accessible text. */
  title: string;
  language: "zh" | "en";
}

export type TourStopKind = "lobby" | "chapter" | "artwork" | "epilogue";

export interface TourStop {
  kind: TourStopKind;
  /** Where the camera sits. */
  position: [number, number, number];
  /** What it looks at. */
  target: [number, number, number];
  chapterId?: string;
  itemId?: string;
  label: string;
}

/** Aspect-aware artwork dimensions, clamped so nothing dwarfs the room. */
function artworkSize(item: ExhibitionItem): { width: number; height: number } {
  // The object type bounds the box; the renderer fits the actual image's
  // aspect ratio inside it once the image has loaded.
  const type = `${item.object.type} ${item.object.medium}`.toLowerCase();
  if (type.includes("handscroll")) return { width: 2.6, height: 1.1 };
  if (type.includes("hanging scroll") || type.includes("scroll")) {
    return { width: 1.1, height: 2.3 };
  }
  if (type.includes("album")) return { width: 1.5, height: 1.2 };
  if (isVitrineObject(item)) {
    // Three-dimensional objects are shown as a vitrine card rather than a
    // fabricated 3D model — none of the institutions publish scan data.
    return { width: 1.35, height: 1.7 };
  }
  return { width: 1.6, height: 1.8 };
}

/** True when an object should be presented in a case rather than on the wall. */
export function isVitrineObject(item: ExhibitionItem): boolean {
  const type = `${item.object.type} ${item.object.classification ?? ""} ${item.object.medium}`.toLowerCase();
  return /vessel|ceramic|porcelain|bronze|jade|sculpture|metalwork|lacquer|stoneware|earthenware|figure/.test(
    type,
  );
}

/** Mount surround beyond the fitted box, used to size the screen wall. */
function mountExtraWidth(item: ExhibitionItem): number {
  const type = `${item.object.type} ${item.object.medium}`.toLowerCase();
  if (type.includes("handscroll")) return 0.6;
  if (isVitrineObject(item)) return 0.45;
  return 0.42;
}

// The key object of a chapter: what the curatorial frame calls core evidence,
// then whatever opens or closes the argument.
const HERO_ROLE_ORDER: ExhibitionItem["role"][] = [
  "core_evidence",
  "opening",
  "synthesis",
  "contrast",
  "context",
];

export function chapterHero(items: ExhibitionItem[]): ExhibitionItem | null {
  for (const role of HERO_ROLE_ORDER) {
    const match = items.find((item) => item.role === role);
    if (match) return match;
  }
  return items[0] ?? null;
}

export function buildHallLayout(exhibition: Exhibition): HallLayout {
  const design = exhibition.spaceDesign;
  const itemsById = new Map(exhibition.items.map((item) => [item.id, item]));

  // The space form changes the proportions of one continuous hall. Chapters
  // share this width; they do not become disconnected room templates.
  const widthByForm: Record<SpaceDesignSpec["spaceForm"], number> = {
    hall: 11,
    cloister: 12.5,
    corridor: 8,
    pavilion: 13,
  };
  const roomWidth = widthByForm[design.spaceForm] ?? 11;
  const halfWidth = roomWidth / 2;
  // Side-wall viewpoints stand on this line.
  const sideViewX = halfWidth - VIEW_DISTANCE;

  const chapters = exhibition.chapters.length
    ? exhibition.chapters
    : // Degenerate fallback: one narrative zone holding everything.
      [
        {
          id: "single",
          order: 0,
          title: exhibition.title,
          leadIn: "",
          itemIds: exhibition.items.map((item) => item.id),
        } as Chapter,
      ];

  const rooms: RoomLayout[] = [];
  const artworks: ArtworkPlacement[] = [];
  const obstacles: Obstacle[] = [];
  let cursorZ = 0;
  let tourIndex = 0;

  const sidePlacement = (
    item: ExhibitionItem,
    chapterId: string,
    side: "left" | "right",
    z: number,
  ): ArtworkPlacement => {
    const { width, height } = artworkSize(item);
    const wallX = side === "left" ? -halfWidth : halfWidth;
    // Nudge off the wall so the frame never z-fights with the plaster.
    const x = side === "left" ? wallX + MOUNT_OFFSET : wallX - MOUNT_OFFSET;
    if (isVitrineObject(item)) {
      obstacles.push({
        x0: side === "left" ? -halfWidth : halfWidth - CASE_DEPTH,
        x1: side === "left" ? -halfWidth + CASE_DEPTH : halfWidth,
        z0: z - width / 2 - 0.25,
        z1: z + width / 2 + 0.25,
        kind: "case",
      });
    }
    return {
      itemId: item.id,
      item,
      chapterId,
      tourIndex: 0,
      position: [x, HANG_HEIGHT, z],
      rotationY: side === "left" ? Math.PI / 2 : -Math.PI / 2,
      viewpoint: [side === "left" ? x + VIEW_DISTANCE : x - VIEW_DISTANCE, EYE_HEIGHT, z],
      width,
      height,
      side,
    };
  };

  chapters.forEach((chapter, roomIndex) => {
    const items = chapter.itemIds
      .map((id) => itemsById.get(id))
      .filter((item): item is ExhibitionItem => Boolean(item));
    const hero = chapterHero(items);
    const heroIndex = hero ? items.indexOf(hero) : -1;
    // Narrative before the key object leads up to it; after it, follows on.
    const before = heroIndex >= 0 ? items.slice(0, heroIndex) : items;
    const after = heroIndex >= 0 ? items.slice(heroIndex + 1) : [];
    const perSideBefore = Math.ceil(before.length / 2);
    const perSideAfter = Math.ceil(after.length / 2);

    const frontLength = Math.max(
      FRONT_MIN,
      ZONE_EDGE + Math.max(0, perSideBefore - 1) * SIDE_SPACING + SCREEN_CLEARANCE + SCREEN_THICKNESS / 2,
    );
    const backLength = after.length
      ? Math.max(
          BACK_MIN,
          SCREEN_THICKNESS / 2 + SCREEN_CLEARANCE + (perSideAfter - 1) * SIDE_SPACING + ZONE_EDGE,
        )
      : BACK_MIN;
    const depth = frontLength + backLength;
    const zStart = cursorZ;
    const zEnd = cursorZ - depth;
    const screenZ = zStart - frontLength;
    const screenFace = screenZ + SCREEN_THICKNESS / 2;

    const heroSize = hero ? artworkSize(hero) : { width: 1.6, height: 1.8 };
    const screenWidth = Math.min(
      roomWidth - 2 * 2.1,
      Math.max(3.6, roomWidth * 0.42, heroSize.width + (hero ? mountExtraWidth(hero) : 0) + 0.5),
    );
    const screen: ScreenWall = {
      z: screenZ,
      width: screenWidth,
      height: SCREEN_HEIGHT,
      thickness: SCREEN_THICKNESS,
      heroItemId: hero?.id ?? null,
    };
    const heroIsCase = hero ? isVitrineObject(hero) : false;
    obstacles.push({
      x0: -screenWidth / 2,
      x1: screenWidth / 2,
      z0: screenZ - SCREEN_THICKNESS / 2,
      z1: heroIsCase ? screenFace + CASE_DEPTH : screenFace,
      kind: "screen",
    });

    // Placement order is narrative order: zig-zag pairs before the screen,
    // the key object on the screen, then pairs behind it.
    const placements: ArtworkPlacement[] = [];
    before.forEach((item, index) => {
      const z = zStart - ZONE_EDGE - Math.floor(index / 2) * SIDE_SPACING;
      placements.push(sidePlacement(item, chapter.id, index % 2 === 0 ? "left" : "right", z));
    });
    if (hero) {
      const { width, height } = heroSize;
      const surfaceZ = screenFace + MOUNT_OFFSET;
      placements.push({
        itemId: hero.id,
        item: hero,
        chapterId: chapter.id,
        tourIndex: 0,
        position: [0, HANG_HEIGHT, surfaceZ],
        rotationY: 0,
        viewpoint: [0, EYE_HEIGHT, surfaceZ + VIEW_DISTANCE],
        width,
        height,
        side: "screen",
      });
    }
    after.forEach((item, index) => {
      const z = screenZ - SCREEN_THICKNESS / 2 - SCREEN_CLEARANCE - Math.floor(index / 2) * SIDE_SPACING;
      placements.push(sidePlacement(item, chapter.id, index % 2 === 0 ? "left" : "right", z));
    });
    for (const placement of placements) {
      placement.tourIndex = tourIndex++;
      artworks.push(placement);
    }

    rooms.push({
      chapter,
      index: roomIndex,
      width: roomWidth,
      depth,
      center: [0, WALL_HEIGHT / 2, (zStart + zEnd) / 2],
      zStart,
      zEnd,
      entryViewpoint: [0, EYE_HEIGHT, zStart - 0.6],
      artworks: placements,
      exitZ: roomIndex < chapters.length - 1 ? zEnd : null,
      screen,
    });

    // The next chapter begins exactly where this one ends. The renderer marks
    // the transition with piers and a lintel, never a closed wall or a gap.
    cursorZ = zEnd;
  });

  const lastRoom = rooms[rooms.length - 1];
  const epilogueZone = { zStart: lastRoom.zEnd, zEnd: lastRoom.zEnd - EPILOGUE_DEPTH };
  const endZ = epilogueZone.zEnd;
  const epilogueViewpoint: [number, number, number] = [0, EYE_HEIGHT, endZ + 7.2];

  const stops: TourStop[] = [
    {
      kind: "lobby",
      position: [0, EYE_HEIGHT, 5.5],
      target: [0, EYE_HEIGHT + 0.6, 0],
      label: exhibition.title,
    },
  ];

  rooms.forEach((room) => {
    stops.push({
      kind: "chapter",
      position: room.entryViewpoint,
      // Slightly above the key object, so the title band reads too.
      target: [0, EYE_HEIGHT + 0.7, room.screen.z],
      chapterId: room.chapter.id,
      label: room.chapter.title,
    });
    room.artworks.forEach((placement) => {
      stops.push({
        kind: "artwork",
        position: placement.viewpoint,
        target: placement.position,
        chapterId: room.chapter.id,
        itemId: placement.itemId,
        label: placement.item.object.title,
      });
    });
  });

  stops.push({
    kind: "epilogue",
    position: epilogueViewpoint,
    target: [0, EYE_HEIGHT + 0.7, endZ],
    label: exhibition.visitorProfile?.language === "en" ? "Closing" : "结语",
  });

  return {
    rooms,
    artworks,
    stops,
    totalDepth: rooms[0].zStart - endZ + ENTRANCE_APPROACH_DEPTH,
    lobbyViewpoint: [0, EYE_HEIGHT, 5.5],
    epilogueViewpoint,
    design,
    epilogueZone,
    endZ,
    obstacles,
    title: exhibition.title,
    language: exhibition.visitorProfile?.language === "en" ? "en" : "zh",
  };
}

/** Camera flight time in seconds, from the model-chosen pacing. */
export function flightDuration(pacing: SpaceDesignSpec["pacing"]): number {
  switch (pacing) {
    case "ant":
      return 2.4;
    case "grasshopper":
      return 1.0;
    default:
      return 1.6;
  }
}

/**
 * Axis-aligned walkable bounds for free-walk collision.
 *
 * Every chapter boundary remains the full hall width. Only the exterior
 * approach in front of the entrance is narrowed to the doorway. The global
 * minZ sits inside the retrospective wall, so a visitor can never step
 * through it. Freestanding walls and cases are handled separately
 * by `resolveObstacles`.
 */
export function walkableBounds(
  layout: HallLayout,
  z: number,
): { minX: number; maxX: number; minZ: number; maxZ: number } {
  const firstRoom = layout.rooms[0];
  if (!firstRoom) {
    return { minX: -5, maxX: 5, minZ: -10, maxZ: 10 };
  }

  const outsideEntrance = z > firstRoom.zStart;
  const halfWidth = outsideEntrance
    ? DOORWAY_WIDTH / 2 - 0.12
    : firstRoom.width / 2 - WALKING_INSET;
  return {
    minX: -halfWidth,
    maxX: halfWidth,
    minZ: layout.endZ + WALKING_INSET,
    maxZ: firstRoom.zStart + ENTRANCE_APPROACH_DEPTH,
  };
}

function insideObstacle(x: number, z: number, obstacle: Obstacle, radius: number): boolean {
  return (
    x > obstacle.x0 - radius &&
    x < obstacle.x1 + radius &&
    z > obstacle.z0 - radius &&
    z < obstacle.z1 + radius
  );
}

/**
 * Keep a walking visitor out of freestanding walls and cases.
 *
 * Blocked moves slide along whichever axis is still free, so walking into a
 * screen wall at an angle carries the visitor round it instead of stopping
 * dead. Pure, so it can be tested without a renderer.
 */
export function resolveObstacles(
  from: { x: number; z: number },
  to: { x: number; z: number },
  obstacles: readonly Obstacle[],
  radius = 0.32,
): { x: number; z: number } {
  const blocked = (x: number, z: number) =>
    obstacles.some((obstacle) => insideObstacle(x, z, obstacle, radius));
  if (!blocked(to.x, to.z)) return to;
  if (!blocked(to.x, from.z)) return { x: to.x, z: from.z };
  if (!blocked(from.x, to.z)) return { x: from.x, z: to.z };
  return { x: from.x, z: from.z };
}
