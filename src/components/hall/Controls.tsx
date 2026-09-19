"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { useFrame, useThree } from "@react-three/fiber";
import { EYE_HEIGHT, flightDuration, walkableBounds, type HallLayout, type TourStop } from "./layout";
import type { CameraPose } from "./minimap";
import {
  accumulatePointerTravel,
  DRAG_GESTURE_DISTANCE,
  nextLookAngles,
} from "./look";
import {
  buildGuidedRoute,
  routeFlightDuration,
  sampleNavigationRoute,
  type NavigationRoute,
} from "./navigation";

/**
 * Guided camera.
 *
 * Flies to the current tour stop with an eased interpolation, or cuts straight
 * there when the visitor has asked for reduced motion — camera flight is the
 * single most nausea-inducing thing in a walkable scene.
 */
export function GuidedCamera({
  layout,
  stop,
  pacing,
  reduceMotion,
  onArrive,
}: {
  layout: HallLayout;
  stop: TourStop;
  pacing: HallLayout["design"]["pacing"];
  reduceMotion: boolean;
  onArrive?: () => void;
}) {
  const { camera } = useThree();
  const from = useRef(new THREE.Vector3());
  const fromTarget = useRef(new THREE.Vector3());
  const elapsed = useRef(0);
  const arrived = useRef(false);
  const currentTarget = useRef(new THREE.Vector3(...stop.target));
  const route = useRef<NavigationRoute | null>(null);
  const flightSeconds = useRef(reduceMotion ? 0 : flightDuration(pacing));
  // Reused each frame instead of allocating two vectors per tick.
  const scratch = useRef({
    destination: new THREE.Vector3(),
    target: new THREE.Vector3(),
  });

  useEffect(() => {
    from.current.copy(camera.position);
    fromTarget.current.copy(currentTarget.current);
    const nextRoute = buildGuidedRoute(
      [camera.position.x, camera.position.y, camera.position.z],
      stop,
      layout,
    );
    route.current = nextRoute;
    flightSeconds.current = reduceMotion
      ? 0
      : routeFlightDuration(flightDuration(pacing), nextRoute.totalLength);
    elapsed.current = 0;
    arrived.current = false;
  }, [stop, layout, camera, pacing, reduceMotion]);

  useFrame((_state, delta) => {
    if (arrived.current) return;
    const activeRoute = route.current;
    const destination = scratch.current.destination.set(...stop.position);
    const target = scratch.current.target.set(...stop.target);

    const duration = flightSeconds.current;
    if (duration <= 0) {
      camera.position.copy(destination);
      currentTarget.current.copy(target);
      camera.lookAt(target);
      arrived.current = true;
      onArrive?.();
      return;
    }

    elapsed.current += delta;
    const progress = Math.min(1, elapsed.current / duration);
    // Smoothstep: no sudden start or stop, which is what makes flight readable.
    const eased = progress * progress * (3 - 2 * progress);

    if (activeRoute && activeRoute.totalLength > 0) {
      const travelled = activeRoute.totalLength * eased;
      camera.position.set(...sampleNavigationRoute(activeRoute, travelled));

      // Look a little way down the circulation path while travelling, then
      // turn toward the object only on approach. This prevents the camera
      // staring sideways through several rooms while following the doorway.
      const lookAhead = sampleNavigationRoute(
        activeRoute,
        Math.min(activeRoute.totalLength, travelled + 1.8),
      );
      const guideTarget = scratch.current.destination.set(...lookAhead);
      guideTarget.y = camera.position.y;
      const startTurn = Math.min(1, progress / 0.15);
      currentTarget.current.lerpVectors(fromTarget.current, guideTarget, startTurn);
      const finalTurn = Math.max(0, Math.min(1, (progress - 0.72) / 0.28));
      const easedFinalTurn = finalTurn * finalTurn * (3 - 2 * finalTurn);
      currentTarget.current.lerp(target, easedFinalTurn);
    } else {
      camera.position.lerpVectors(from.current, destination, eased);
      currentTarget.current.lerpVectors(fromTarget.current, target, eased);
    }
    camera.lookAt(currentTarget.current);

    if (progress >= 1) {
      arrived.current = true;
      onArrive?.();
    }
  });

  return null;
}

const MOVE_SPEED = 3.2;
const KEY_BINDINGS: Record<string, "forward" | "back" | "left" | "right"> = {
  KeyW: "forward",
  ArrowUp: "forward",
  KeyS: "back",
  ArrowDown: "back",
  KeyA: "left",
  ArrowLeft: "left",
  KeyD: "right",
  ArrowRight: "right",
};

export interface JoystickState {
  x: number;
  y: number;
}

/**
 * Free-walk camera.
 *
 * Pointer lock plus WASD on desktop, virtual joystick on touch. Movement is
 * clamped to the walkable bounds of the room the visitor is in, so there is no
 * walking through walls and no getting lost outside the hall. Eye height is
 * fixed and there is no run modifier or head bob — all three are common
 * motion-sickness triggers and none of them serve a gallery.
 */
export function FreeWalkCamera({
  layout,
  joystick,
  enabled,
  onLockChange,
  onLookInteraction,
}: {
  layout: HallLayout;
  joystick: React.RefObject<JoystickState>;
  enabled: boolean;
  onLockChange: (locked: boolean) => void;
  onLookInteraction: () => void;
}) {
  const camera = useThree((state) => state.camera);
  const gl = useThree((state) => state.gl);
  const get = useThree((state) => state.get);
  const setEvents = useThree((state) => state.setEvents);
  const pressed = useRef<Set<string>>(new Set());
  // Scratch vectors reused every frame. They live in refs rather than memos
  // because they are mutated in the render loop, and allocating three vectors
  // per frame would churn the GC at 60 fps.
  const scratch = useRef({
    direction: new THREE.Vector3(),
    forward: new THREE.Vector3(),
    right: new THREE.Vector3(),
    up: new THREE.Vector3(0, 1, 0),
    lookEuler: new THREE.Euler(0, 0, 0, "YXZ"),
  });

  useEffect(() => {
    if (!enabled) return;
    const down = (event: KeyboardEvent) => {
      const action = KEY_BINDINGS[event.code];
      if (action) {
        pressed.current.add(action);
        event.preventDefault();
      }
    };
    const up = (event: KeyboardEvent) => {
      const action = KEY_BINDINGS[event.code];
      if (action) pressed.current.delete(action);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    const held = pressed.current;
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      held.clear();
    };
  }, [enabled]);

  useEffect(() => {
    if (!enabled) return;

    const canvas = gl.domElement;
    const ownerDocument = canvas.ownerDocument;
    const pointerLockAvailable =
      typeof canvas.requestPointerLock === "function"
      && typeof ownerDocument.exitPointerLock === "function";
    let activePointer: number | null = null;
    let lastX = 0;
    let lastY = 0;
    let pointerTravel = 0;
    let dragged = false;
    let raycasterCentered = false;
    const previousCompute = get().events.compute;

    const restoreRaycaster = () => {
      if (!raycasterCentered) return;
      setEvents({ compute: previousCompute });
      raycasterCentered = false;
    };

    const centerRaycaster = () => {
      if (raycasterCentered) return;
      setEvents({
        compute(event, state) {
          const offsetX = state.size.width / 2;
          const offsetY = state.size.height / 2;
          state.pointer.set(
            offsetX / state.size.width * 2 - 1,
            -(offsetY / state.size.height) * 2 + 1,
          );
          state.raycaster.setFromCamera(state.pointer, state.camera);
        },
      });
      raycasterCentered = true;
    };

    const rotate = (deltaX: number, deltaY: number) => {
      if (deltaX === 0 && deltaY === 0) return;
      const euler = scratch.current.lookEuler;
      euler.setFromQuaternion(camera.quaternion, "YXZ");
      const next = nextLookAngles(
        { yaw: euler.y, pitch: euler.x },
        deltaX,
        deltaY,
      );
      euler.set(next.pitch, next.yaw, 0, "YXZ");
      camera.quaternion.setFromEuler(euler);
    };

    const finishDrag = (event?: PointerEvent) => {
      if (event && activePointer !== event.pointerId) return;
      const pointerId = activePointer;
      activePointer = null;
      if (pointerId !== null && canvas.hasPointerCapture?.(pointerId)) {
        canvas.releasePointerCapture(pointerId);
      }
    };

    const onPointerDown = (event: PointerEvent) => {
      if (event.button !== 0 || ownerDocument.pointerLockElement === canvas) return;
      activePointer = event.pointerId;
      lastX = event.clientX;
      lastY = event.clientY;
      pointerTravel = 0;
      dragged = false;
      canvas.setPointerCapture?.(event.pointerId);
      event.preventDefault();
    };

    const onPointerMove = (event: PointerEvent) => {
      if (activePointer !== event.pointerId || ownerDocument.pointerLockElement === canvas) return;
      const deltaX = event.clientX - lastX;
      const deltaY = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;
      pointerTravel = accumulatePointerTravel(pointerTravel, deltaX, deltaY);
      if (pointerTravel >= DRAG_GESTURE_DISTANCE) {
        dragged = true;
        onLookInteraction();
      }
      rotate(deltaX, deltaY);
      event.preventDefault();
    };

    const onPointerUp = (event: PointerEvent) => {
      const shouldLock =
        activePointer === event.pointerId
        && !dragged
        && event.pointerType !== "touch"
        && pointerLockAvailable;
      finishDrag(event);
      if (!shouldLock) return;
      try {
        const lockResult = canvas.requestPointerLock();
        if (lockResult && typeof lockResult.catch === "function") {
          void lockResult.catch(() => onLockChange(false));
        }
      } catch {
        onLockChange(false);
      }
    };

    const onMouseMove = (event: MouseEvent) => {
      if (ownerDocument.pointerLockElement !== canvas) return;
      rotate(event.movementX, event.movementY);
      if (event.movementX !== 0 || event.movementY !== 0) onLookInteraction();
    };

    const onPointerLockChange = () => {
      const locked = ownerDocument.pointerLockElement === canvas;
      if (locked) centerRaycaster();
      else restoreRaycaster();
      onLockChange(locked);
      if (locked) activePointer = null;
    };

    const onPointerLockError = () => {
      restoreRaycaster();
      onLockChange(false);
    };

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", finishDrag);
    canvas.addEventListener("lostpointercapture", finishDrag);
    ownerDocument.addEventListener("mousemove", onMouseMove);
    ownerDocument.addEventListener("pointerlockchange", onPointerLockChange);
    ownerDocument.addEventListener("pointerlockerror", onPointerLockError);

    return () => {
      finishDrag();
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointercancel", finishDrag);
      canvas.removeEventListener("lostpointercapture", finishDrag);
      ownerDocument.removeEventListener("mousemove", onMouseMove);
      ownerDocument.removeEventListener("pointerlockchange", onPointerLockChange);
      ownerDocument.removeEventListener("pointerlockerror", onPointerLockError);
      if (ownerDocument.pointerLockElement === canvas) ownerDocument.exitPointerLock?.();
      restoreRaycaster();
      onLockChange(false);
    };
  }, [camera, enabled, get, gl, onLockChange, onLookInteraction, setEvents]);

  useFrame((_state, delta) => {
    if (!enabled) return;

    const { direction, forward, right, up } = scratch.current;
    direction.set(0, 0, 0);
    if (pressed.current.has("forward")) direction.z -= 1;
    if (pressed.current.has("back")) direction.z += 1;
    if (pressed.current.has("left")) direction.x -= 1;
    if (pressed.current.has("right")) direction.x += 1;

    const stick = joystick.current;
    if (stick && (stick.x !== 0 || stick.y !== 0)) {
      direction.x += stick.x;
      direction.z += stick.y;
    }

    if (direction.lengthSq() === 0) return;
    direction.normalize();

    camera.getWorldDirection(forward);
    forward.y = 0;
    forward.normalize();
    right.crossVectors(forward, up).normalize();

    const step = MOVE_SPEED * Math.min(delta, 0.05);
    const next = camera.position.clone();
    next.addScaledVector(forward, -direction.z * step);
    next.addScaledVector(right, direction.x * step);

    const bounds = walkableBounds(layout, next.z);
    next.x = Math.max(bounds.minX, Math.min(bounds.maxX, next.x));
    next.z = Math.max(bounds.minZ, Math.min(bounds.maxZ, next.z));
    next.y = EYE_HEIGHT;

    camera.position.copy(next);
  });

  return null;
}

/**
 * Copies the camera's floor position and heading into a plain object each
 * frame. The minimap reads it on its own animation frame, so following the
 * visitor never re-renders React.
 */
export function CameraProbe({ poseRef }: { poseRef: React.RefObject<CameraPose> }) {
  const camera = useThree((state) => state.camera);
  const forward = useRef(new THREE.Vector3());

  useFrame(() => {
    const pose = poseRef.current;
    if (!pose) return;
    camera.getWorldDirection(forward.current);
    pose.x = camera.position.x;
    pose.z = camera.position.z;
    pose.forwardX = forward.current.x;
    pose.forwardZ = forward.current.z;
  });

  return null;
}
