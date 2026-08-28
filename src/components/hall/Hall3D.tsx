"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, type RootState } from "@react-three/fiber";
import type { Exhibition } from "@/lib/types";
import { logEvent } from "@/lib/api";
import { useIsTouchPrimary, usePrefersReducedMotion } from "@/lib/useClientCapability";
import { buildHallLayout, EYE_HEIGHT, type TourStop } from "./layout";
import { clampStopIndex } from "./progress";
import { FreeWalkCamera, GuidedCamera, type JoystickState } from "./Controls";
import { HallScene } from "./Scene";
import { HallOverlay } from "./HallOverlay";
import { TouchJoystick } from "./TouchJoystick";
import styles from "./hall.module.css";

export type HallMode = "guided" | "free";

/**
 * The generated 3D hall.
 *
 * All text lives in an HTML overlay rather than in the scene: it stays crisp,
 * it needs no CJK font atlas, and it can be read by assistive tech that cannot
 * see into a canvas. The canvas draws space, artworks and light only.
 */
export function Hall3D({
  exhibition,
  initialStopIndex = 0,
  onStopChange,
  onRuntimeFailure,
  onExit,
}: {
  exhibition: Exhibition;
  initialStopIndex?: number;
  onStopChange?: (stopIndex: number) => void;
  onRuntimeFailure?: (error: unknown) => void;
  onExit: () => void;
}) {
  const layout = useMemo(() => buildHallLayout(exhibition), [exhibition]);
  const [stopIndex, setStopIndex] = useState(() =>
    clampStopIndex(layout.stops.length, initialStopIndex),
  );
  const [mode, setMode] = useState<HallMode>("guided");
  const [pointerLocked, setPointerLocked] = useState(false);
  const [freeLookUsed, setFreeLookUsed] = useState(false);
  const reduceMotion = usePrefersReducedMotion();
  const touchPrimary = useIsTouchPrimary();
  const [cameraArrived, setCameraArrived] = useState(reduceMotion);
  const joystick = useRef<JoystickState>({ x: 0, y: 0 });
  const visitedRef = useRef<Set<string>>(new Set());
  const canvasCleanupRef = useRef<(() => void) | null>(null);
  const runtimeFailureRef = useRef(onRuntimeFailure);
  const effectiveMode: HallMode = touchPrimary ? "guided" : mode;

  const stop: TourStop = layout.stops[stopIndex] ?? layout.stops[0];

  const goTo = useCallback(
    (index: number) => {
      const clamped = Math.max(0, Math.min(layout.stops.length - 1, index));
      if (clamped === stopIndex) return;
      setCameraArrived(false);
      setStopIndex(clamped);
      onStopChange?.(clamped);
      const next = layout.stops[clamped];
      if (next?.kind === "chapter" && next.chapterId && !visitedRef.current.has(next.chapterId)) {
        visitedRef.current.add(next.chapterId);
        logEvent("chapter_entered", exhibition.id, { chapterId: next.chapterId });
      }
      if (next?.kind === "artwork" && next.itemId) {
        logEvent("object_viewed", exhibition.id, { itemId: next.itemId });
      }
      if (next?.kind === "epilogue") {
        logEvent("epilogue_reached", exhibition.id);
      }
    },
    [exhibition.id, layout.stops, onStopChange, stopIndex],
  );

  // Switching to guided from free walk snaps to the nearest stop rather than
  // restarting the tour, so the visitor never loses their place.
  const enterGuided = useCallback(() => {
    if (mode === "guided") return;
    setMode("guided");
    setPointerLocked(false);
    setFreeLookUsed(false);
    setCameraArrived(false);
  }, [mode]);

  const enterFree = useCallback(() => {
    if (touchPrimary) return;
    setPointerLocked(false);
    setFreeLookUsed(false);
    setMode("free");
    setCameraArrived(false);
    logEvent("free_walk_entered", exhibition.id);
  }, [exhibition.id, touchPrimary]);

  const handleLookInteraction = useCallback(() => {
    setFreeLookUsed(true);
  }, []);

  useEffect(() => () => canvasCleanupRef.current?.(), []);

  useEffect(() => {
    runtimeFailureRef.current = onRuntimeFailure;
  }, [onRuntimeFailure]);

  const handleCanvasCreated = useCallback(({ gl }: RootState) => {
    canvasCleanupRef.current?.();
    const canvas = gl.domElement;
    const handleContextLost = (event: Event) => {
      event.preventDefault();
      runtimeFailureRef.current?.(new Error("WebGL context lost"));
    };
    canvas.addEventListener("webglcontextlost", handleContextLost, { once: true });
    canvasCleanupRef.current = () => {
      canvas.removeEventListener("webglcontextlost", handleContextLost);
    };
  }, []);

  useEffect(() => {
    if (effectiveMode !== "guided") return;
    const onKey = (event: KeyboardEvent) => {
      const target = event.target;
      const interactiveTarget = target instanceof HTMLElement
        && (target.isContentEditable
          || Boolean(target.closest("button, a, input, textarea, select, summary, [role='button'], [role='slider']")));
      if (event.defaultPrevented || interactiveTarget) return;
      if (event.key === "Tab" && !event.shiftKey) {
        // Tab is the documented mode toggle; let shift+Tab keep normal focus
        // traversal so the overlay stays keyboard-navigable.
        return;
      }
      if (event.key === "ArrowRight" || event.key === " " || event.key === "PageDown") {
        event.preventDefault();
        goTo(stopIndex + 1);
      } else if (event.key === "ArrowLeft" || event.key === "PageUp") {
        event.preventDefault();
        goTo(stopIndex - 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [effectiveMode, stopIndex, goTo]);

  const activeItemId = stop.kind === "artwork" ? stop.itemId ?? null : null;

  return (
    <div
      className={styles.hallRoot}
      data-mode={effectiveMode}
      data-pointer-locked={pointerLocked ? "true" : "false"}
    >
      <Canvas
        className={styles.canvas}
        aria-hidden="true"
        shadows={false}
        dpr={[1, 1.75]}
        gl={{ antialias: true, powerPreference: "high-performance" }}
        camera={{ fov: 58, near: 0.1, far: 120, position: [0, EYE_HEIGHT, 6] }}
        onCreated={handleCanvasCreated}
      >
        <Suspense fallback={null}>
          <HallScene
            layout={layout}
            activeItemId={activeItemId}
            onSelectItem={(itemId) => {
              const index = layout.stops.findIndex((candidate) => candidate.itemId === itemId);
              if (index >= 0) {
                if (effectiveMode === "free") enterGuided();
                goTo(index);
              }
            }}
          />
          {effectiveMode === "guided" ? (
            <GuidedCamera
              layout={layout}
              stop={stop}
              pacing={layout.design.pacing}
              reduceMotion={reduceMotion}
              onArrive={() => setCameraArrived(true)}
            />
          ) : (
            <FreeWalkCamera
              layout={layout}
              joystick={joystick}
              enabled
              onLockChange={setPointerLocked}
              onLookInteraction={handleLookInteraction}
            />
          )}
        </Suspense>
      </Canvas>

      <HallOverlay
        exhibition={exhibition}
        layout={layout}
        stop={stop}
        stopIndex={stopIndex}
        mode={effectiveMode}
        cameraArrived={cameraArrived}
        pointerLocked={pointerLocked}
        freeLookUsed={freeLookUsed}
        reduceMotion={reduceMotion}
        freeWalkAvailable={!touchPrimary}
        onGoTo={goTo}
        onEnterGuided={enterGuided}
        onEnterFree={enterFree}
        onExit={onExit}
      />

      {effectiveMode === "free" && <TouchJoystick stateRef={joystick} />}
    </div>
  );
}
