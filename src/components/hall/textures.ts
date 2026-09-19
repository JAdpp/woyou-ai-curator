import * as THREE from "three";

/**
 * Procedural surface textures for the hall, drawn once on a canvas.
 *
 * Nothing is fetched: the only images a visitor downloads are the collection
 * objects themselves. The maps are near-white greyscale so the model-chosen
 * colours (`SpaceDesignSpec`) still decide the palette; the texture only adds
 * the variation that stops a wall or floor reading as a flat plane.
 */

/** Real-world size, in metres, one repeat of the floor texture covers. */
export const FLOOR_TILE_METRES = 4;
/** Real-world size, in metres, one repeat of the plaster texture covers. */
export const PLASTER_TILE_METRES = 3;

const cache = new Map<string, THREE.CanvasTexture>();

// Seeded so the hall looks the same on every visit and every reload.
function seeded(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeCanvas(size: number) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("2D canvas unavailable");
  return { canvas, context };
}

function grey(value: number, alpha = 1) {
  const level = Math.round(Math.max(0, Math.min(1, value)) * 255);
  return `rgba(${level}, ${level}, ${level}, ${alpha})`;
}

function finish(key: string, canvas: HTMLCanvasElement, repeat: boolean) {
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  if (repeat) {
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
  }
  cache.set(key, texture);
  return texture;
}

/** Relative luminance of a CSS colour, 0..1. */
export function luminance(color: string): number {
  const { r, g, b } = new THREE.Color(color);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * Floor: long boards running with the route on a dark floor, large stone
 * slabs on a light one. Boards run along texture V, which the rotated floor
 * plane maps onto the walking direction.
 */
export function floorTexture(kind: "boards" | "stone"): THREE.CanvasTexture {
  const key = `floor:${kind}`;
  const cached = cache.get(key);
  if (cached) return cached;

  const size = 1024;
  const pxPerMetre = size / FLOOR_TILE_METRES;
  const random = seeded(kind === "boards" ? 11 : 29);
  const { canvas, context } = makeCanvas(size);
  context.fillStyle = grey(1);
  context.fillRect(0, 0, size, size);

  if (kind === "boards") {
    const boardWidth = Math.round(0.2 * pxPerMetre);
    for (let x = 0; x < size; x += boardWidth) {
      let y = -Math.round(random() * 2 * pxPerMetre);
      while (y < size) {
        const length = Math.round((1.4 + random() * 1.6) * pxPerMetre);
        const tone = 0.84 + random() * 0.16;
        context.fillStyle = grey(tone);
        context.fillRect(x, y, boardWidth, length);
        // Grain: faint lines along the board.
        for (let line = 0; line < 7; line += 1) {
          const gx = x + 2 + random() * (boardWidth - 4);
          context.strokeStyle = grey(tone - 0.08 - random() * 0.06, 0.35);
          context.lineWidth = 0.6 + random() * 0.8;
          context.beginPath();
          context.moveTo(gx, y);
          context.bezierCurveTo(
            gx + (random() - 0.5) * 3, y + length * 0.33,
            gx + (random() - 0.5) * 3, y + length * 0.66,
            gx + (random() - 0.5) * 2, y + length,
          );
          context.stroke();
        }
        context.fillStyle = grey(0.55, 0.55);
        context.fillRect(x, y, boardWidth, 2);
        y += length;
      }
      context.fillStyle = grey(0.55, 0.6);
      context.fillRect(x, 0, 2, size);
    }
  } else {
    const slabWidth = pxPerMetre * 1.0;
    const slabLength = pxPerMetre * 2.0;
    for (let row = 0; row * slabLength < size; row += 1) {
      const offset = row % 2 === 0 ? 0 : slabWidth / 2;
      for (let x = -slabWidth; x < size + slabWidth; x += slabWidth) {
        const left = x + offset;
        const top = row * slabLength;
        context.fillStyle = grey(0.9 + random() * 0.1);
        context.fillRect(left, top, slabWidth, slabLength);
        // Soft mottling within each slab.
        for (let spot = 0; spot < 14; spot += 1) {
          const radius = 12 + random() * 46;
          const cx = left + random() * slabWidth;
          const cy = top + random() * slabLength;
          const gradient = context.createRadialGradient(cx, cy, 0, cx, cy, radius);
          gradient.addColorStop(0, grey(0.82, 0.16));
          gradient.addColorStop(1, grey(0.82, 0));
          context.fillStyle = gradient;
          context.fillRect(cx - radius, cy - radius, radius * 2, radius * 2);
        }
        context.strokeStyle = grey(0.62, 0.7);
        context.lineWidth = 2;
        context.strokeRect(left + 1, top + 1, slabWidth - 2, slabLength - 2);
      }
    }
  }

  // A fine speckle over everything keeps large areas from banding.
  const speckle = context.getImageData(0, 0, size, size);
  for (let index = 0; index < speckle.data.length; index += 4) {
    const noise = (random() - 0.5) * 10;
    speckle.data[index] += noise;
    speckle.data[index + 1] += noise;
    speckle.data[index + 2] += noise;
  }
  context.putImageData(speckle, 0, 0);
  return finish(key, canvas, true);
}

/** Wall plaster: barely-there mottling, enough to catch the picture lights. */
export function plasterTexture(): THREE.CanvasTexture {
  const key = "plaster";
  const cached = cache.get(key);
  if (cached) return cached;
  const size = 512;
  const random = seeded(7);
  const { canvas, context } = makeCanvas(size);
  context.fillStyle = grey(1);
  context.fillRect(0, 0, size, size);
  for (let spot = 0; spot < 90; spot += 1) {
    const radius = 20 + random() * 90;
    const cx = random() * size;
    const cy = random() * size;
    const gradient = context.createRadialGradient(cx, cy, 0, cx, cy, radius);
    gradient.addColorStop(0, grey(0.9, 0.07));
    gradient.addColorStop(1, grey(0.9, 0));
    context.fillStyle = gradient;
    // Drawn at every wrapped position so the tile repeats without seams.
    for (const dx of [-size, 0, size]) {
      for (const dy of [-size, 0, size]) {
        context.fillRect(cx - radius + dx, cy - radius + dy, radius * 2, radius * 2);
      }
    }
  }
  const pixels = context.getImageData(0, 0, size, size);
  for (let index = 0; index < pixels.data.length; index += 4) {
    const noise = (random() - 0.5) * 6;
    pixels.data[index] += noise;
    pixels.data[index + 1] += noise;
    pixels.data[index + 2] += noise;
  }
  context.putImageData(pixels, 0, 0);
  return finish(key, canvas, true);
}

/**
 * A soft-edged rectangle, used as a drop shadow behind frames, a contact
 * shadow under plinths and, in white, as the glow behind the current object.
 * Drawn with canvas shadows rather than `ctx.filter`, which older Safari
 * ignores.
 */
export function softRectTexture(tone: "dark" | "light"): THREE.CanvasTexture {
  const key = `soft:${tone}`;
  const cached = cache.get(key);
  if (cached) return cached;
  const size = 256;
  const { canvas, context } = makeCanvas(size);
  const inset = size * 0.2;
  const colour = tone === "dark" ? "rgba(0, 0, 0, 1)" : "rgba(255, 255, 255, 1)";
  context.shadowColor = colour;
  context.shadowBlur = size * 0.11;
  // Draw the shape off-canvas and let only its blurred shadow land inside.
  context.shadowOffsetX = size * 2;
  context.fillStyle = colour;
  context.fillRect(inset - size * 2, inset, size - inset * 2, size - inset * 2);
  return finish(key, canvas, false);
}
