import assert from "node:assert/strict";
import test from "node:test";
import * as THREE from "three";
import {
  accumulatePointerTravel,
  DRAG_GESTURE_DISTANCE,
  FREE_LOOK_SENSITIVITY,
  MAX_FREE_LOOK_PITCH,
  nextLookAngles,
} from "./look";

const closeTo = (actual: number, expected: number) => {
  assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} should be close to ${expected}`);
};

test("horizontal dragging changes yaw without changing pitch", () => {
  const next = nextLookAngles({ yaw: 0.25, pitch: -0.1 }, 100, 0);
  closeTo(next.yaw, 0.25 - 100 * FREE_LOOK_SENSITIVITY);
  closeTo(next.pitch, -0.1);
});

test("vertical dragging changes pitch without changing yaw", () => {
  const next = nextLookAngles({ yaw: 0.25, pitch: -0.1 }, 0, -40);
  closeTo(next.yaw, 0.25);
  closeTo(next.pitch, -0.1 + 40 * FREE_LOOK_SENSITIVITY);
});

test("pitch is clamped before the camera can flip", () => {
  assert.equal(nextLookAngles({ yaw: 0, pitch: 0 }, 0, -10000).pitch, MAX_FREE_LOOK_PITCH);
  assert.equal(nextLookAngles({ yaw: 0, pitch: 0 }, 0, 10000).pitch, -MAX_FREE_LOOK_PITCH);
});

test("zero pointer movement preserves the current view", () => {
  assert.deepEqual(nextLookAngles({ yaw: 1.2, pitch: -0.4 }, 0, 0), {
    yaw: 1.2,
    pitch: -0.4,
  });
});

test("many one-pixel moves still become a drag rather than a click", () => {
  let travelled = 0;
  for (let index = 0; index < DRAG_GESTURE_DISTANCE; index += 1) {
    travelled = accumulatePointerTravel(travelled, 1, 0);
  }
  assert.ok(travelled >= DRAG_GESTURE_DISTANCE);
});

test("walking direction follows the yaw produced by free look", () => {
  const angles = nextLookAngles({ yaw: 0, pitch: 0 }, -Math.PI / (2 * FREE_LOOK_SENSITIVITY), 0);
  const forward = new THREE.Vector3(0, 0, -1).applyEuler(
    new THREE.Euler(angles.pitch, angles.yaw, 0, "YXZ"),
  );

  closeTo(forward.x, -1);
  closeTo(forward.z, 0);
});
