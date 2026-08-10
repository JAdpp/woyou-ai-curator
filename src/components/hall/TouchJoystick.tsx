"use client";

import { useRef, useState } from "react";
import { useIsTouchPrimary } from "@/lib/useClientCapability";
import type { JoystickState } from "./Controls";
import styles from "./hall.module.css";

const RADIUS = 52;

/**
 * Virtual movement stick for touch devices.
 *
 * Pointer lock does not exist on mobile, so free walk there needs an on-screen
 * control. Writes straight into a ref so dragging never triggers a React
 * re-render inside the animation loop.
 */
export function TouchJoystick({ stateRef }: { stateRef: React.RefObject<JoystickState> }) {
  const baseRef = useRef<HTMLDivElement>(null);
  const [knob, setKnob] = useState({ x: 0, y: 0 });
  const activePointer = useRef<number | null>(null);
  // Only shown where touch is the primary input; pointer lock covers desktop.
  const visible = useIsTouchPrimary();

  if (!visible) return null;

  const reset = () => {
    activePointer.current = null;
    setKnob({ x: 0, y: 0 });
    stateRef.current = { x: 0, y: 0 };
  };

  const update = (event: React.PointerEvent<HTMLDivElement>) => {
    const base = baseRef.current;
    if (!base) return;
    const rect = base.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    let dx = event.clientX - centerX;
    let dy = event.clientY - centerY;
    const distance = Math.hypot(dx, dy);
    if (distance > RADIUS) {
      dx = (dx / distance) * RADIUS;
      dy = (dy / distance) * RADIUS;
    }
    setKnob({ x: dx, y: dy });
    stateRef.current = { x: dx / RADIUS, y: dy / RADIUS };
  };

  return (
    <div
      ref={baseRef}
      className={styles.joystickBase}
      role="application"
      aria-label="移动摇杆"
      onPointerDown={(event) => {
        activePointer.current = event.pointerId;
        event.currentTarget.setPointerCapture(event.pointerId);
        update(event);
      }}
      onPointerMove={(event) => {
        if (activePointer.current !== event.pointerId) return;
        update(event);
      }}
      onPointerUp={reset}
      onPointerCancel={reset}
    >
      <div className={styles.joystickKnob} style={{ transform: `translate(${knob.x}px, ${knob.y}px)` }} />
    </div>
  );
}
