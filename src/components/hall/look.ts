export interface LookAngles {
  yaw: number;
  pitch: number;
}

/** A little below vertical keeps free look comfortable and prevents flips. */
export const MAX_FREE_LOOK_PITCH = Math.PI * (83 / 180);
export const FREE_LOOK_SENSITIVITY = 0.0035;
export const DRAG_GESTURE_DISTANCE = 4;

export function accumulatePointerTravel(current: number, deltaX: number, deltaY: number) {
  return current + Math.hypot(deltaX, deltaY);
}

export function nextLookAngles(
  current: LookAngles,
  deltaX: number,
  deltaY: number,
  sensitivity = FREE_LOOK_SENSITIVITY,
): LookAngles {
  const nextPitch = current.pitch - deltaY * sensitivity;
  return {
    yaw: current.yaw - deltaX * sensitivity,
    pitch: Math.max(-MAX_FREE_LOOK_PITCH, Math.min(MAX_FREE_LOOK_PITCH, nextPitch)),
  };
}
