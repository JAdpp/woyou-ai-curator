import type { TourStop } from "./layout";

export type ExhibitionViewMode = "3d" | "2d" | "text";

export interface HallSessionState {
  stopIndex: number;
  lastMode: "3d" | "2d";
  autoEntered: boolean;
}

export interface InitialHallModeInput {
  forceTextView: boolean;
  requestedView: string | null;
  webglAvailable: boolean;
  touchPrimary: boolean;
  session: HallSessionState | null;
}

export function decideInitialHallMode(input: InitialHallModeInput): {
  enter: boolean;
  autoEntered: boolean;
} {
  const { forceTextView, requestedView, webglAvailable, touchPrimary, session } = input;
  if (forceTextView) return { enter: false, autoEntered: session?.autoEntered ?? false };
  if (requestedView === "3d") {
    return { enter: webglAvailable, autoEntered: session?.autoEntered ?? false };
  }
  if (requestedView === "2d") {
    return { enter: false, autoEntered: session?.autoEntered ?? false };
  }
  if (session?.autoEntered) {
    return {
      enter: session.lastMode === "3d" && webglAvailable && !touchPrimary,
      autoEntered: true,
    };
  }
  const enter = webglAvailable && !touchPrimary;
  return { enter, autoEntered: enter };
}

export function clampStopIndex(stopCount: number, candidate: unknown): number {
  if (stopCount <= 0) return 0;
  const parsed = typeof candidate === "number" ? candidate : Number(candidate);
  if (!Number.isFinite(parsed)) return 0;
  return Math.max(0, Math.min(stopCount - 1, Math.trunc(parsed)));
}

export function stopAnchor(stop: TourStop | undefined): string {
  if (!stop) return "";
  if (stop.kind === "artwork" && stop.itemId) return `item-${stop.itemId}`;
  if (stop.kind === "chapter" && stop.chapterId) return `chapter-${stop.chapterId}`;
  if (stop.kind === "epilogue") return "epilogue";
  return "";
}

function stopIndexFromHash(stops: TourStop[], rawHash: string): number | null {
  let hash = rawHash.replace(/^#/, "");
  try {
    hash = decodeURIComponent(hash);
  } catch {
    // A malformed fragment should not prevent the exhibition from opening.
  }
  if (!hash) return null;

  const index = stops.findIndex((stop) => stopAnchor(stop) === hash);
  return index >= 0 ? index : null;
}

/** Resolve progress from a shareable URL, preferring its semantic anchor. */
export function stopIndexFromUrl(stops: TourStop[], url: URL, fallback = 0): number {
  const anchored = stopIndexFromHash(stops, url.hash);
  if (anchored !== null) return anchored;

  const queryValue = url.searchParams.get("stop");
  if (queryValue !== null && /^\d+$/.test(queryValue)) {
    return clampStopIndex(stops.length, queryValue);
  }
  return clampStopIndex(stops.length, fallback);
}

/**
 * Keep mode and progress in the address bar without navigating or reloading.
 * `text` is deliberately distinct from the reversible in-page `2d` mode.
 */
export function exhibitionViewHref(
  currentHref: string,
  mode: ExhibitionViewMode,
  stopIndex: number,
  stop: TourStop | undefined,
): string {
  const url = new URL(currentHref);
  url.searchParams.set("view", mode);
  url.searchParams.set("stop", String(Math.max(0, Math.trunc(stopIndex))));
  const anchor = stopAnchor(stop);
  url.hash = anchor ? `#${encodeURIComponent(anchor)}` : "";
  return `${url.pathname}${url.search}${url.hash}`;
}

export function parseHallSession(raw: string | null, stopCount: number): HallSessionState | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<HallSessionState>;
    if (parsed.lastMode !== "3d" && parsed.lastMode !== "2d") return null;
    return {
      stopIndex: clampStopIndex(stopCount, parsed.stopIndex),
      lastMode: parsed.lastMode,
      autoEntered: parsed.autoEntered === true,
    };
  } catch {
    return null;
  }
}
