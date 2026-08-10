import type { Chapter, Exhibition, ExhibitionItem, SpaceDesignSpec } from "@/lib/types";

/**
 * Pure geometry for the generated hall.
 *
 * Kept free of three.js so the layout can be reasoned about and tested on its
 * own: the renderer just draws whatever this produces. Chapters are contiguous
 * narrative zones along one shared -Z exhibition line, with artworks
 * alternating between the side walls.
 */

export const WALL_HEIGHT = 4.6;
export const DOORWAY_WIDTH = 2.4;
export const DOORWAY_HEIGHT = 3;
/** Eye height for both camera modes. */
export const EYE_HEIGHT = 1.6;
/** How far in front of a wall the guided camera parks. */
export const VIEW_DISTANCE = 3.1;

const WALL_MARGIN = 3.2;
const ARTWORK_SPACING = 3.4;
const WALKING_INSET = 0.55;
const ENTRANCE_APPROACH_DEPTH = 7;

export interface ArtworkPlacement {
  itemId: string;
  item: ExhibitionItem;
  chapterId: string;
  /** Index across the whole exhibition, matching the guided tour order. */
  tourIndex: number;
  /** Centre of the artwork on the wall. */
  position: [number, number, number];
  /** Y rotation so the artwork faces into the room. */
  rotationY: number;
  /** Where the guided camera stops to look at it. */
  viewpoint: [number, number, number];
  width: number;
  height: number;
  side: "left" | "right";
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
  /** Camera stop for the chapter title card. */
  entryViewpoint: [number, number, number];
  artworks: ArtworkPlacement[];
  /** Narrative threshold to the next section, if any. */
  exitZ: number | null;
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
  // Institution metadata rarely carries reliable pixel dimensions, so the
  // object type is a better hint at orientation than a guessed aspect ratio.
  const type = `${item.object.type} ${item.object.medium}`.toLowerCase();
  if (type.includes("handscroll")) return { width: 2.6, height: 1.1 };
  if (type.includes("hanging scroll") || type.includes("scroll")) {
    return { width: 1.1, height: 2.3 };
  }
  if (type.includes("album")) return { width: 1.5, height: 1.2 };
  if (
    type.includes("vessel") ||
    type.includes("ceramic") ||
    type.includes("porcelain") ||
    type.includes("bronze") ||
    type.includes("jade") ||
    type.includes("sculpture") ||
    type.includes("metalwork")
  ) {
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
  let cursorZ = 0;
  let tourIndex = 0;

  chapters.forEach((chapter, roomIndex) => {
    const items = chapter.itemIds
      .map((id) => itemsById.get(id))
      .filter((item): item is ExhibitionItem => Boolean(item));

    // Depth grows with how many artworks have to fit down each side wall.
    const perSide = Math.ceil(items.length / 2);
    const depth = Math.max(8, WALL_MARGIN * 2 + Math.max(0, perSide - 1) * ARTWORK_SPACING);
    const zStart = cursorZ;
    const zEnd = cursorZ - depth;
    const centerZ = (zStart + zEnd) / 2;

    const placements: ArtworkPlacement[] = [];
    let leftCount = 0;
    let rightCount = 0;

    items.forEach((item, index) => {
      const side: "left" | "right" = index % 2 === 0 ? "left" : "right";
      const slot = side === "left" ? leftCount++ : rightCount++;
      const z = zStart - WALL_MARGIN - slot * ARTWORK_SPACING;
      const { width, height } = artworkSize(item);
      const wallX = side === "left" ? -roomWidth / 2 : roomWidth / 2;
      // Nudge off the wall so the frame never z-fights with the plaster.
      const x = side === "left" ? wallX + 0.06 : wallX - 0.06;
      const y = EYE_HEIGHT + 0.28;

      const placement: ArtworkPlacement = {
        itemId: item.id,
        item,
        chapterId: chapter.id,
        tourIndex: tourIndex++,
        position: [x, y, z],
        rotationY: side === "left" ? Math.PI / 2 : -Math.PI / 2,
        viewpoint: [
          side === "left" ? wallX + VIEW_DISTANCE : wallX - VIEW_DISTANCE,
          EYE_HEIGHT,
          z,
        ],
        width,
        height,
        side,
      };
      placements.push(placement);
      artworks.push(placement);
    });

    rooms.push({
      chapter,
      index: roomIndex,
      width: roomWidth,
      depth,
      center: [0, WALL_HEIGHT / 2, centerZ],
      zStart,
      zEnd,
      entryViewpoint: [0, EYE_HEIGHT, zStart - 1.8],
      artworks: placements,
      exitZ: roomIndex < chapters.length - 1 ? zEnd : null,
    });

    // The next chapter begins exactly where this one ends. The renderer marks
    // the transition with light and a threshold, never a wall or a gap.
    cursorZ = zEnd;
  });

  const lastRoom = rooms[rooms.length - 1];
  const epilogueViewpoint: [number, number, number] = lastRoom
    ? [0, EYE_HEIGHT, lastRoom.zEnd + Math.min(2.4, lastRoom.depth / 2)]
    : [0, EYE_HEIGHT, -3.5];

  const stops: TourStop[] = [
    {
      kind: "lobby",
      position: [0, EYE_HEIGHT, 5.5],
      target: [0, EYE_HEIGHT + 0.4, 0],
      label: exhibition.title,
    },
  ];

  rooms.forEach((room) => {
    stops.push({
      kind: "chapter",
      position: room.entryViewpoint,
      target: [0, EYE_HEIGHT + 0.3, room.zEnd],
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
    target: [0, EYE_HEIGHT + 0.3, lastRoom ? lastRoom.zEnd + 0.2 : -5.8],
    label: "结语",
  });

  const hallDepth = lastRoom ? rooms[0].zStart - lastRoom.zEnd : 0;

  return {
    rooms,
    artworks,
    stops,
    totalDepth: hallDepth + ENTRANCE_APPROACH_DEPTH,
    lobbyViewpoint: [0, EYE_HEIGHT, 5.5],
    epilogueViewpoint,
    design,
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
 * minZ sits inside the final wall, so a visitor can never step through it.
 */
export function walkableBounds(
  layout: HallLayout,
  z: number,
): { minX: number; maxX: number; minZ: number; maxZ: number } {
  const firstRoom = layout.rooms[0];
  const lastRoom = layout.rooms[layout.rooms.length - 1];
  if (!firstRoom || !lastRoom) {
    return { minX: -5, maxX: 5, minZ: -10, maxZ: 10 };
  }

  const room =
    layout.rooms.find((candidate) => z <= candidate.zStart && z >= candidate.zEnd) ??
    layout.rooms.reduce((closest, candidate) => {
      const closestDistance = Math.min(
        Math.abs(z - closest.zStart),
        Math.abs(z - closest.zEnd),
      );
      const candidateDistance = Math.min(
        Math.abs(z - candidate.zStart),
        Math.abs(z - candidate.zEnd),
      );
      return candidateDistance < closestDistance ? candidate : closest;
    }, firstRoom);

  const outsideEntrance = z > firstRoom.zStart;
  const halfWidth = outsideEntrance
    ? DOORWAY_WIDTH / 2 - 0.12
    : room.width / 2 - WALKING_INSET;
  return {
    minX: -halfWidth,
    maxX: halfWidth,
    minZ: lastRoom.zEnd + WALKING_INSET,
    maxZ: firstRoom.zStart + ENTRANCE_APPROACH_DEPTH,
  };
}
