import { strict as assert } from "node:assert";
import test from "node:test";
import type { Exhibition, ExhibitionItem } from "@/lib/types";
import { buildHallLayout } from "./layout";
import { buildHallPlan } from "./minimap";

function makeExhibition(itemCount: number, chapterCount: number): Exhibition {
  const items = Array.from({ length: itemCount }, (_unused, index) => ({
    id: `i${index}`,
    object: { id: `obj-${index}`, title: `Object ${index}`, type: "Painting", medium: "ink" },
    role: "context",
    labelSentences: [],
  })) as unknown as ExhibitionItem[];
  const perChapter = Math.ceil(itemCount / chapterCount);
  return {
    id: "ex",
    title: "T",
    items,
    chapters: Array.from({ length: chapterCount }, (_unused, index) => ({
      id: `c${index}`,
      order: index,
      title: `Chapter ${index}`,
      leadIn: "",
      itemIds: items.slice(index * perChapter, (index + 1) * perChapter).map((item) => item.id),
    })),
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
  } as unknown as Exhibition;
}

const inside = (rect: { x: number; y: number; width: number; height: number }, w: number, h: number) =>
  rect.x >= -0.001 && rect.y >= -0.001 && rect.x + rect.width <= w + 0.001 && rect.y + rect.height <= h + 0.001;

test("the whole route, lobby included, fits inside the plan", () => {
  for (const [items, chapters] of [[5, 2], [8, 3], [12, 4]]) {
    const layout = buildHallLayout(makeExhibition(items, chapters));
    const plan = buildHallPlan(layout, 240, 84);
    assert.ok(inside(plan.hall, 240, 84));
    assert.ok(inside(plan.approach, 240, 84));
    for (const artwork of plan.artworks) assert.ok(inside(artwork, 240, 84));
    const [lobbyX, lobbyY] = plan.project(layout.lobbyViewpoint[0], layout.lobbyViewpoint[2]);
    assert.ok(lobbyX >= 0 && lobbyX <= 240 && lobbyY >= 0 && lobbyY <= 84);
    assert.equal(plan.rooms.length, layout.rooms.length);
    assert.equal(plan.artworks.length, layout.artworks.length);
  }
});

test("the visit runs left to right and the left wall is the top edge", () => {
  const layout = buildHallLayout(makeExhibition(8, 3));
  const plan = buildHallPlan(layout, 240, 84);
  const [entranceX] = plan.project(0, layout.rooms[0].zStart);
  const [endX] = plan.project(0, layout.rooms[layout.rooms.length - 1].zEnd);
  assert.ok(entranceX < endX);
  for (let index = 1; index < plan.rooms.length; index += 1) {
    assert.ok(plan.rooms[index].x > plan.rooms[index - 1].x);
  }
  const left = plan.artworks.find((artwork) => artwork.side === "left")!;
  const right = plan.artworks.find((artwork) => artwork.side === "right")!;
  assert.ok(left.y < plan.height / 2 && right.y > plan.height / 2);
});

test("heading points along the plan: forward is right, turning left is up", () => {
  const plan = buildHallPlan(buildHallLayout(makeExhibition(5, 2)), 240, 84);
  assert.equal(Math.round(plan.heading(0, -1)), 0);
  // Facing the left wall (-X) is up on the plan, i.e. -90 degrees in SVG.
  assert.equal(Math.round(plan.heading(-1, 0)), -90);
  assert.equal(Math.round(plan.heading(1, 0)), 90);
});
