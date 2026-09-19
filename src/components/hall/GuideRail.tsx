"use client";

import { useEffect, useMemo, useRef, type RefObject } from "react";
import type { Exhibition } from "@/lib/types";
import { resolveObjectImageUrl } from "@/lib/api";
import { fill } from "@/lib/i18n";
import { useLanguage } from "@/lib/useLanguage";
import { publicObjectMetadata, publicObjectTitle } from "@/lib/localizedMetadata";
import { DOORWAY_WIDTH, type HallLayout } from "./layout";
import { buildHallPlan, type CameraPose } from "./minimap";
import styles from "./hall.module.css";

const PLAN_WIDTH = 256;
const PLAN_HEIGHT = 92;

/**
 * The floor plan. Drawn once from the layout; only the "you are here" marker
 * moves, and it is moved by writing its transform directly on an animation
 * frame rather than through React state, sixty times a second.
 *
 * The plan is a pointer shortcut. Everything on it is also in the object list
 * below, which is the keyboard and screen-reader route, so the drawing itself
 * stays out of the accessibility tree.
 */
function HallPlan({
  layout,
  activeItemId,
  activeChapterId,
  poseRef,
  titles,
  stopByItem,
  onGoTo,
}: {
  layout: HallLayout;
  activeItemId: string | null;
  activeChapterId: string | null;
  poseRef: RefObject<CameraPose>;
  titles: Map<string, string>;
  stopByItem: Map<string, number>;
  onGoTo: (index: number) => void;
}) {
  const { t: copy } = useLanguage();
  const t = copy.hall;
  const plan = useMemo(() => buildHallPlan(layout, PLAN_WIDTH, PLAN_HEIGHT, 8), [layout]);
  const markerRef = useRef<SVGGElement>(null);

  useEffect(() => {
    let frame = 0;
    let last = "";
    const follow = () => {
      const pose = poseRef.current;
      const marker = markerRef.current;
      if (pose && marker) {
        const [x, y] = plan.project(pose.x, pose.z);
        const angle = plan.heading(pose.forwardX, pose.forwardZ);
        const transform = `translate(${x.toFixed(1)} ${y.toFixed(1)}) rotate(${angle.toFixed(0)})`;
        if (transform !== last) {
          marker.setAttribute("transform", transform);
          last = transform;
        }
      }
      frame = window.requestAnimationFrame(follow);
    };
    frame = window.requestAnimationFrame(follow);
    return () => window.cancelAnimationFrame(frame);
  }, [plan, poseRef]);

  const [, doorTop] = plan.project(-DOORWAY_WIDTH / 2, 0);
  const [, doorBottom] = plan.project(DOORWAY_WIDTH / 2, 0);
  const { hall } = plan;
  // Outline with a gap where the entrance door is.
  const outline = [
    `M ${hall.x} ${doorTop}`,
    `V ${hall.y}`,
    `H ${hall.x + hall.width}`,
    `V ${hall.y + hall.height}`,
    `H ${hall.x}`,
    `V ${doorBottom}`,
  ].join(" ");

  return (
    <figure className={styles.planFigure}>
      <svg
        className={styles.plan}
        viewBox={`0 0 ${PLAN_WIDTH} ${PLAN_HEIGHT}`}
        aria-hidden="true"
        focusable="false"
      >
        <rect
          className={styles.planApproach}
          x={plan.approach.x}
          y={plan.approach.y}
          width={plan.approach.width}
          height={plan.approach.height}
        />
        {plan.rooms.map((room) => (
          <g key={room.chapterId}>
            <rect
              className={room.chapterId === activeChapterId ? styles.planRoomActive : styles.planRoom}
              x={room.x}
              y={room.y}
              width={room.width}
              height={room.height}
            />
            <text
              className={styles.planRoomNumber}
              x={room.x + room.width / 2}
              y={room.y + room.height / 2 + 3.5}
              textAnchor="middle"
            >
              {room.index + 1}
            </text>
          </g>
        ))}
        {plan.rooms.slice(1).map((room) => (
          <line
            key={`divider-${room.chapterId}`}
            className={styles.planDivider}
            x1={room.x}
            x2={room.x}
            y1={room.y}
            y2={room.y + room.height}
          />
        ))}
        <path className={styles.planWalls} d={outline} />
        {plan.artworks.map((artwork) => {
          const stop = stopByItem.get(artwork.itemId);
          return (
            <g
              key={artwork.itemId}
              className={styles.planArtwork}
              data-active={artwork.itemId === activeItemId ? "true" : "false"}
              onClick={() => {
                if (stop !== undefined) onGoTo(stop);
              }}
            >
              <title>{titles.get(artwork.itemId) ?? ""}</title>
              {/* Generous invisible target around a three-pixel mark. */}
              <rect
                className={styles.planHitArea}
                x={artwork.x - 4}
                y={artwork.y - 4}
                width={artwork.width + 8}
                height={artwork.height + 8}
              />
              <rect x={artwork.x} y={artwork.y} width={artwork.width} height={artwork.height} rx={0.8} />
            </g>
          );
        })}
        <g ref={markerRef} className={styles.planMarker}>
          <path d="M0 0 L15 -7.5 A 16.8 16.8 0 0 1 15 7.5 Z" className={styles.planCone} />
          <circle r={3.4} className={styles.planDot} />
        </g>
      </svg>
      <figcaption>
        <span className={styles.planKey} aria-hidden="true" />
        {t.youAreHere}
        <span className={styles.planHint}>{t.planHint}</span>
      </figcaption>
    </figure>
  );
}

/**
 * Left-hand guide: the exhibition's name, where the visitor is on a floor
 * plan, and every stop as a list that can be jumped to. On narrow screens the
 * same panel opens as a drawer from the top bar.
 */
export function GuideRail({
  exhibition,
  layout,
  stopIndex,
  visited,
  poseRef,
  contentLang,
  listOpen,
  drawerOpen,
  onToggleList,
  onCloseDrawer,
  onGoTo,
}: {
  exhibition: Exhibition;
  layout: HallLayout;
  stopIndex: number;
  visited: ReadonlySet<number>;
  poseRef: RefObject<CameraPose>;
  contentLang: string;
  listOpen: boolean;
  drawerOpen: boolean;
  onToggleList: () => void;
  onCloseDrawer: () => void;
  onGoTo: (index: number) => void;
}) {
  const { t: copy } = useLanguage();
  const t = copy.hall;
  const listRef = useRef<HTMLElement>(null);
  const stop = layout.stops[stopIndex] ?? layout.stops[0];
  const epilogueIndex = layout.stops.length - 1;

  const { stopByItem, stopByChapter, titles } = useMemo(() => {
    const byItem = new Map<string, number>();
    const byChapter = new Map<string, number>();
    layout.stops.forEach((candidate, index) => {
      if (candidate.kind === "artwork" && candidate.itemId) byItem.set(candidate.itemId, index);
      if (candidate.kind === "chapter" && candidate.chapterId) byChapter.set(candidate.chapterId, index);
    });
    const names = new Map(
      exhibition.items.map((item) => [item.id, publicObjectTitle(item, contentLang)]),
    );
    return { stopByItem: byItem, stopByChapter: byChapter, titles: names };
  }, [contentLang, exhibition.items, layout.stops]);

  const activeRoom = stop.chapterId
    ? layout.rooms.find((room) => room.chapter.id === stop.chapterId) ?? null
    : null;
  const place = stop.kind === "lobby"
    ? t.lobbyStop
    : stop.kind === "epilogue"
      ? t.epilogueStop
      : activeRoom
        ? `${fill(t.partShort, { n: activeRoom.index + 1 })} · ${activeRoom.chapter.title}`
        : "";
  const listVisible = listOpen || drawerOpen;

  // Keep the current stop comfortably in view inside the list: when it is
  // near either edge, bring it to the middle. Only the list scrolls, never the
  // page or the fixed hall around it.
  useEffect(() => {
    const list = listRef.current;
    if (!list || !listVisible) return;
    const current = list.querySelector<HTMLElement>("[aria-current='step']");
    if (!current) return;
    const listBox = list.getBoundingClientRect();
    const itemBox = current.getBoundingClientRect();
    const margin = Math.min(72, listBox.height / 4);
    if (itemBox.top < listBox.top + margin || itemBox.bottom > listBox.bottom - margin) {
      list.scrollTop += itemBox.top + itemBox.height / 2 - (listBox.top + listBox.height / 2);
    }
  }, [listVisible, stopIndex]);

  const current = (index: number) => (index === stopIndex ? "step" : undefined);
  const seen = (index: number) => (visited.has(index) ? "true" : "false");

  return (
    <aside
      className={styles.guide}
      data-drawer={drawerOpen ? "open" : "closed"}
      aria-label={t.guideLabel}
    >
      <div className={styles.guideHead}>
        <div>
          <p className={styles.guideEyebrow}>
            {copy.brand.productName} · {t.guideLabel}
          </p>
          <p className={styles.guideTitle} lang={contentLang}>{exhibition.title}</p>
          <p className={styles.guidePlace} lang={contentLang}>
            {fill(t.nowAt, { place })}
          </p>
        </div>
        <button type="button" className={styles.guideClose} onClick={onCloseDrawer} aria-label={t.closeGuide}>
          ×
        </button>
      </div>

      <HallPlan
        layout={layout}
        activeItemId={stop.kind === "artwork" ? stop.itemId ?? null : null}
        activeChapterId={stop.chapterId ?? null}
        poseRef={poseRef}
        titles={titles}
        stopByItem={stopByItem}
        onGoTo={onGoTo}
      />

      <button
        type="button"
        className={styles.guideListToggle}
        onClick={onToggleList}
        aria-expanded={listVisible}
        aria-controls="hall-guide-list"
      >
        <span>{t.objectList}</span>
        <small>{listOpen ? t.hideList : t.showList}</small>
      </button>

      <nav
        id="hall-guide-list"
        ref={listRef}
        className={styles.guideList}
        aria-label={t.objectList}
        hidden={!listVisible}
      >
        <ol>
          <li>
            <button
              type="button"
              className={styles.guideStop}
              aria-current={current(0)}
              data-visited={seen(0)}
              onClick={() => onGoTo(0)}
            >
              {t.lobbyStop}
            </button>
          </li>
          {layout.rooms.map((room) => {
            const chapterStop = stopByChapter.get(room.chapter.id);
            return (
              <li key={room.chapter.id}>
                <button
                  type="button"
                  className={styles.guideChapter}
                  aria-current={chapterStop === undefined ? undefined : current(chapterStop)}
                  data-in-chapter={stop.chapterId === room.chapter.id ? "true" : "false"}
                  onClick={() => {
                    if (chapterStop !== undefined) onGoTo(chapterStop);
                  }}
                >
                  <small>{fill(t.partShort, { n: room.index + 1 })}</small>
                  <span lang={contentLang}>{room.chapter.title}</span>
                </button>
                <ol>
                  {room.artworks.map((placement) => {
                    const index = stopByItem.get(placement.itemId);
                    if (index === undefined) return null;
                    const meta = publicObjectMetadata(placement.item, contentLang)[0];
                    return (
                      <li key={placement.itemId}>
                        <button
                          type="button"
                          className={styles.guideItem}
                          aria-current={current(index)}
                          data-visited={seen(index)}
                          onClick={() => onGoTo(index)}
                        >
                          <span className={styles.guideNumber} aria-hidden="true">
                            {String(placement.tourIndex + 1).padStart(2, "0")}
                          </span>
                          <span className={styles.guideThumb} aria-hidden="true">
                            {/* eslint-disable-next-line @next/next/no-img-element -- collection image proxy, sized server-side */}
                            <img
                              src={resolveObjectImageUrl(placement.item.object.id, 512)}
                              alt=""
                              loading="lazy"
                              decoding="async"
                              onError={(event) => { event.currentTarget.hidden = true; }}
                            />
                          </span>
                          <span className={styles.guideItemText}>
                            <strong lang={contentLang}>{titles.get(placement.itemId)}</strong>
                            {meta && <small lang={contentLang}>{meta}</small>}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ol>
              </li>
            );
          })}
          <li>
            <button
              type="button"
              className={styles.guideStop}
              aria-current={current(epilogueIndex)}
              data-visited={seen(epilogueIndex)}
              onClick={() => onGoTo(epilogueIndex)}
            >
              {t.epilogueStop}
            </button>
          </li>
        </ol>
      </nav>
    </aside>
  );
}
