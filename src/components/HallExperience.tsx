"use client";

import dynamic from "next/dynamic";
import {
  Component,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { Exhibition } from "@/lib/types";
import { logEvent } from "@/lib/api";
import { useIsTouchPrimary, useWebglCapability } from "@/lib/useClientCapability";
import { ExhibitionView2D } from "./ExhibitionView2D";
import { buildHallLayout } from "./hall/layout";
import {
  clampStopIndex,
  decideInitialHallMode,
  exhibitionViewHref,
  parseHallSession,
  stopIndexFromUrl,
  type HallSessionState,
} from "./hall/progress";

// three.js is ~600 KB gzipped and cannot render on the server; keep it out of
// the initial bundle so the text version stays fast.
const Hall3D = dynamic(() => import("./hall/Hall3D").then((module) => module.Hall3D), {
  ssr: false,
  loading: () => (
    <div
      style={{
        position: "fixed",
        inset: 0,
        display: "grid",
        placeItems: "center",
        background: "#14110d",
        color: "#f3efe6",
        fontFamily: "ui-serif, serif",
      }}
    >
      <p>正在搭建展厅…</p>
    </div>
  ),
});

const HALL_SESSION_PREFIX = "woyou.hall.";

class HallRuntimeBoundary extends Component<
  { children: ReactNode; onFailure: (error: unknown) => void },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: unknown) {
    this.props.onFailure(error);
  }

  render() {
    return this.state.failed ? null : this.props.children;
  }
}

export function HallExperience({
  exhibition,
  forceTextView = false,
}: {
  exhibition: Exhibition;
  forceTextView?: boolean;
}) {
  const capability = useWebglCapability();
  const touchPrimary = useIsTouchPrimary();
  const layout = useMemo(() => buildHallLayout(exhibition), [exhibition]);
  const [inHall, setInHall] = useState(false);
  const [currentStopIndex, setCurrentStopIndex] = useState(0);
  const [hallFailed, setHallFailed] = useState(false);
  const [hallAttempt, setHallAttempt] = useState(0);
  const [restoreTextFocus, setRestoreTextFocus] = useState(forceTextView);
  const entryResolvedRef = useRef(false);
  const autoEnteredRef = useRef(false);
  const sessionKey = `${HALL_SESSION_PREFIX}${exhibition.id}`;

  const persist = useCallback((lastMode: "3d" | "2d", stopIndex: number) => {
    const state: HallSessionState = {
      stopIndex: clampStopIndex(layout.stops.length, stopIndex),
      lastMode,
      autoEntered: autoEnteredRef.current,
    };
    try {
      window.sessionStorage.setItem(sessionKey, JSON.stringify(state));
    } catch {
      // Private browsing can block storage; the URL still preserves progress.
    }
  }, [layout.stops.length, sessionKey]);

  const replaceViewUrl = useCallback((mode: "3d" | "2d" | "text", stopIndex: number) => {
    if (typeof window === "undefined") return;
    const safeIndex = clampStopIndex(layout.stops.length, stopIndex);
    const href = exhibitionViewHref(
      window.location.href,
      forceTextView ? "text" : mode,
      safeIndex,
      layout.stops[safeIndex],
    );
    window.history.replaceState(window.history.state, "", href);
  }, [forceTextView, layout.stops]);

  // Resolve the entry mode once. A desktop with usable WebGL enters 3D on its
  // first visit; a coarse-pointer device starts in 2D. An explicit `view=text`
  // is permanent and never gets overridden by capability detection.
  useEffect(() => {
    if (capability === "checking" || entryResolvedRef.current) return;
    entryResolvedRef.current = true;

    const session = (() => {
      try {
        return parseHallSession(window.sessionStorage.getItem(sessionKey), layout.stops.length);
      } catch {
        return null;
      }
    })();
    const url = new URL(window.location.href);
    const restoredStop = stopIndexFromUrl(layout.stops, url, session?.stopIndex ?? 0);
    const requestedView = url.searchParams.get("view");
    const decision = decideInitialHallMode({
      forceTextView,
      requestedView,
      webglAvailable: capability === "ok",
      touchPrimary,
      session,
    });
    const enter = decision.enter;
    autoEnteredRef.current = decision.autoEntered;

    setCurrentStopIndex(restoredStop);
    setInHall(enter);
    setRestoreTextFocus(
      !enter && restoredStop > 0 && Boolean(forceTextView || requestedView === "2d" || session),
    );
    persist(enter ? "3d" : "2d", restoredStop);
    replaceViewUrl(forceTextView ? "text" : enter ? "3d" : "2d", restoredStop);
    void logEvent(enter ? "hall_entered" : "hall_fallback_2d", exhibition.id, {
      mode: enter ? "3d" : "text",
      reason: forceTextView
        ? "explicit_text_view"
        : capability !== "ok"
          ? "webgl_unavailable"
          : touchPrimary
            ? "coarse_pointer_default"
            : "saved_2d_preference",
    });
  }, [
    capability,
    exhibition.id,
    forceTextView,
    layout.stops,
    persist,
    replaceViewUrl,
    sessionKey,
    touchPrimary,
  ]);

  const updateProgress = useCallback((stopIndex: number) => {
    const safeIndex = clampStopIndex(layout.stops.length, stopIndex);
    setCurrentStopIndex(safeIndex);
    const mode = inHall ? "3d" : "2d";
    persist(mode, safeIndex);
    replaceViewUrl(mode, safeIndex);
  }, [inHall, layout.stops.length, persist, replaceViewUrl]);

  const leaveHall = useCallback(() => {
    setInHall(false);
    setRestoreTextFocus(true);
    persist("2d", currentStopIndex);
    replaceViewUrl("2d", currentStopIndex);
    void logEvent("hall_exited", exhibition.id, { stopIndex: currentStopIndex });
  }, [currentStopIndex, exhibition.id, persist, replaceViewUrl]);

  const enterHall = useCallback(() => {
    if (capability !== "ok" || forceTextView) return;
    setHallFailed(false);
    setRestoreTextFocus(false);
    setHallAttempt((attempt) => attempt + 1);
    setInHall(true);
    persist("3d", currentStopIndex);
    replaceViewUrl("3d", currentStopIndex);
    void logEvent("hall_entered", exhibition.id, {
      mode: "3d",
      stopIndex: currentStopIndex,
      source: "visitor_choice",
    });
  }, [capability, currentStopIndex, exhibition.id, forceTextView, persist, replaceViewUrl]);

  const fallbackFromHall = useCallback((error: unknown) => {
    setHallFailed(true);
    setInHall(false);
    setRestoreTextFocus(true);
    persist("2d", currentStopIndex);
    replaceViewUrl("2d", currentStopIndex);
    void logEvent("hall_fallback_2d", exhibition.id, {
      mode: "text",
      reason: "hall_runtime_error",
      errorType: error instanceof Error ? error.name : "unknown",
    });
  }, [currentStopIndex, exhibition.id, persist, replaceViewUrl]);

  if (inHall && capability === "ok") {
    return (
      <HallRuntimeBoundary key={hallAttempt} onFailure={fallbackFromHall}>
        <Hall3D
          exhibition={exhibition}
          initialStopIndex={currentStopIndex}
          onStopChange={updateProgress}
          onRuntimeFailure={fallbackFromHall}
          onExit={leaveHall}
        />
      </HallRuntimeBoundary>
    );
  }

  return (
    <ExhibitionView2D
      exhibition={exhibition}
      initialStopIndex={currentStopIndex}
      restoreInitialFocus={restoreTextFocus}
      onActiveStopChange={updateProgress}
      hallRuntimeFailed={hallFailed}
      webglAvailable={capability === "ok"}
      onEnterHall={forceTextView ? undefined : enterHall}
    />
  );
}
