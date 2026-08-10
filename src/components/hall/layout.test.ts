import { strict as assert } from "node:assert";
import test from "node:test";
import type { Exhibition, ExhibitionItem } from "@/lib/types";
import {
  buildHallLayout,
  walkableBounds,
  DOORWAY_WIDTH,
  EYE_HEIGHT,
  isVitrineObject,
} from "./layout";

/**
 * Hall geometry is pure, so it can be checked without a GPU. These guard the
 * properties a visitor would actually notice if they broke: unreachable
 * artworks, broken chapter seams, or walls they can walk through.
 */

function makeItem(id: string, type = "Painting"): ExhibitionItem {
  return {
    id,
    object: {
      id: `obj-${id}`,
      accessionNumber: id,
      title: `Object ${id}`,
      date: "1400",
      medium: "ink on silk",
      type,
      imageUrl: "https://example.test/i.jpg",
      objectUrl: "https://example.test/o",
      rights: "CC0",
      altText: "alt",
      themes: [],
      evidence: [],
    },
    role: "core_evidence",
    roleLabel: "核心证据",
    subQuestion: "q",
    whySelected: "w",
    relation: "r",
    labelSentences: [],
    alternatives: [],
  } as unknown as ExhibitionItem;
}

function makeExhibition(itemCount: number, chapterCount: number): Exhibition {
  const items = Array.from({ length: itemCount }, (_unused, index) => makeItem(`i${index}`));
  const perChapter = Math.ceil(itemCount / chapterCount);
  const chapters = Array.from({ length: chapterCount }, (_unused, index) => ({
    id: `c${index}`,
    order: index,
    title: `Chapter ${index}`,
    leadIn: "lead",
    itemIds: items.slice(index * perChapter, (index + 1) * perChapter).map((item) => item.id),
  }));
  return {
    id: "ex",
    title: "T",
    question: "q",
    curatorialThesis: "t",
    coreAnswer: "a",
    subQuestions: ["a", "b"],
    items,
    chapters: chapters.filter((chapter) => chapter.itemIds.length > 0),
    epilogue: { text: "e", openQuestions: [], materialBoundary: [] },
    spaceDesign: {
      spaceForm: "hall",
      wallColor: "#e8e3da",
      floorColor: "#4a4038",
      accentColor: "#8c6b3f",
      ceilingColor: "#f2efe9",
      lightTemperature: 3400,
      lightIntensity: 0.85,
      mood: "warm_dim",
      frameStyle: "thin_dark",
      pacing: "butterfly",
      typography: "serif",
    },
    coverageLimits: [],
    status: "ready",
    versions: { model: "m", prompt: "p", collection: "c", validator: "v" },
    validation: { passed: true, checks: [], blockingIssues: [], checkedAt: "" },
    updatedAt: "",
  } as unknown as Exhibition;
}

test("every item gets exactly one placement and one tour stop", () => {
  for (const [items, chapters] of [
    [5, 2],
    [8, 3],
    [12, 4],
  ] as const) {
    const layout = buildHallLayout(makeExhibition(items, chapters));
    assert.equal(layout.artworks.length, items, `${items} items lost a placement`);

    const placed = new Set(layout.artworks.map((artwork) => artwork.itemId));
    assert.equal(placed.size, items, "an item was placed twice");

    const artworkStops = layout.stops.filter((stop) => stop.kind === "artwork");
    assert.equal(artworkStops.length, items, "an item is unreachable from the tour");

    // Lobby first, epilogue last — the tour must have a beginning and an end.
    assert.equal(layout.stops[0].kind, "lobby");
    assert.equal(layout.stops[layout.stops.length - 1].kind, "epilogue");
  }
});

test("chapter zones meet exactly on one continuous exhibition line", () => {
  const layout = buildHallLayout(makeExhibition(12, 4));
  for (let index = 1; index < layout.rooms.length; index += 1) {
    const previous = layout.rooms[index - 1];
    const current = layout.rooms[index];
    assert.equal(previous.zEnd, current.zStart, `chapter ${index} has a spatial gap`);
    assert.equal(previous.exitZ, current.zStart, `chapter ${index} threshold is misaligned`);
    assert.equal(previous.width, current.width, "a shared side wall changes width mid-route");
  }
});

test("artworks sit inside their room and face into it", () => {
  const layout = buildHallLayout(makeExhibition(8, 3));
  for (const room of layout.rooms) {
    for (const artwork of room.artworks) {
      const [x, y, z] = artwork.position;
      assert.ok(
        Math.abs(x) <= room.width / 2 + 0.1,
        "artwork hangs outside the room width",
      );
      assert.ok(z <= room.zStart && z >= room.zEnd, "artwork sits outside the room depth");
      assert.ok(y > 0 && y < 4.6, "artwork hangs outside the wall height");
      // Left wall faces +x, right wall faces -x.
      assert.equal(artwork.rotationY, artwork.side === "left" ? Math.PI / 2 : -Math.PI / 2);
    }
  }
});

test("guided viewpoints stand in front of the artwork at eye height", () => {
  const layout = buildHallLayout(makeExhibition(8, 3));
  for (const artwork of layout.artworks) {
    const [viewX, viewY, viewZ] = artwork.viewpoint;
    const [artX, , artZ] = artwork.position;
    assert.equal(viewY, EYE_HEIGHT);
    assert.equal(viewZ, artZ, "viewpoint is not aligned with the artwork");
    // The camera must be on the room side of the wall, not inside it.
    assert.ok(
      artwork.side === "left" ? viewX > artX : viewX < artX,
      "viewpoint is behind the wall",
    );
  }
});

test("chapter boundaries stay full-width while only the exterior entrance narrows", () => {
  const layout = buildHallLayout(makeExhibition(8, 3));
  for (const room of layout.rooms) {
    const midZ = (room.zStart + room.zEnd) / 2;
    const bounds = walkableBounds(layout, midZ);
    assert.ok(bounds.maxX < room.width / 2, "bounds allow walking into the wall");
    assert.ok(bounds.minX > -room.width / 2, "bounds allow walking into the wall");
  }

  for (let index = 1; index < layout.rooms.length; index += 1) {
    const boundaryZ = layout.rooms[index].zStart;
    const before = walkableBounds(layout, boundaryZ + 0.02);
    const at = walkableBounds(layout, boundaryZ);
    const after = walkableBounds(layout, boundaryZ - 0.02);
    assert.equal(before.maxX, at.maxX, "chapter threshold narrows on approach");
    assert.equal(after.maxX, at.maxX, "chapter threshold narrows after crossing");
  }

  const firstRoom = layout.rooms[0];
  const justInside = walkableBounds(layout, firstRoom.zStart - 0.02);
  const justOutside = walkableBounds(layout, firstRoom.zStart + 0.02);
  assert.ok(justInside.maxX > DOORWAY_WIDTH / 2, "hall interior is still doorway-width");
  assert.ok(
    justOutside.maxX < DOORWAY_WIDTH / 2,
    "exterior approach does not narrow to the entrance",
  );
});

test("the final wall is a hard walking boundary", () => {
  const layout = buildHallLayout(makeExhibition(8, 3));
  const lastRoom = layout.rooms[layout.rooms.length - 1];
  for (const probeZ of [lastRoom.zEnd + 1, lastRoom.zEnd, lastRoom.zEnd - 10]) {
    const bounds = walkableBounds(layout, probeZ);
    assert.ok(bounds.minZ > lastRoom.zEnd, "bounds allow crossing the final wall");
  }
});

test("every guided stop and artwork viewpoint is reachable", () => {
  const layout = buildHallLayout(makeExhibition(12, 4));

  for (const stop of layout.stops) {
    const [x, , z] = stop.position;
    const bounds = walkableBounds(layout, z);
    assert.ok(x >= bounds.minX && x <= bounds.maxX, `${stop.label} is outside the route width`);
    assert.ok(z >= bounds.minZ && z <= bounds.maxZ, `${stop.label} is beyond an end wall`);
  }

  for (const artwork of layout.artworks) {
    const [x, , z] = artwork.viewpoint;
    const bounds = walkableBounds(layout, z);
    assert.ok(x >= bounds.minX && x <= bounds.maxX, `${artwork.itemId} viewpoint hits a wall`);
    assert.ok(z >= bounds.minZ && z <= bounds.maxZ, `${artwork.itemId} viewpoint leaves the hall`);
  }
});

test("a degenerate exhibition with no chapters still builds one room", () => {
  const exhibition = makeExhibition(5, 1);
  exhibition.chapters = [];
  const layout = buildHallLayout(exhibition);
  assert.equal(layout.rooms.length, 1);
  assert.equal(layout.artworks.length, 5);
});

test("three-dimensional objects are shown in a vitrine, not as a flat picture", () => {
  assert.equal(isVitrineObject(makeItem("a", "Vessel")), true);
  assert.equal(isVitrineObject(makeItem("b", "Bronze")), true);
  assert.equal(isVitrineObject(makeItem("c", "Painting")), false);
});
