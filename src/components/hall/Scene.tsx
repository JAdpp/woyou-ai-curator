"use client";

import { Component, Suspense, useMemo, useState, type ReactNode } from "react";
import * as THREE from "three";
import type { ThreeElements } from "@react-three/fiber";
import { Environment, Lightformer, MeshReflectorMaterial, useTexture } from "@react-three/drei";
import { resolveObjectImageUrl } from "@/lib/api";
import type { ExhibitionItem, SpaceDesignSpec } from "@/lib/types";
import {
  DOORWAY_HEIGHT,
  DOORWAY_WIDTH,
  MOUNT_OFFSET,
  WALL_HEIGHT,
  type ArtworkPlacement,
  type HallLayout,
  type RoomLayout,
  isVitrineObject,
} from "./layout";
import {
  FLOOR_TILE_METRES,
  PLASTER_TILE_METRES,
  floorTexture,
  luminance,
  plasterTexture,
  softRectTexture,
  wallTextTexture,
} from "./textures";

/** "high" adds the reflective floor; touch devices and slow GPUs get "low". */
export type HallQuality = "high" | "low";

/** Convert a colour temperature in kelvin to an approximate RGB tint. */
function kelvinToColor(kelvin: number): THREE.Color {
  const temperature = Math.max(2200, Math.min(6500, kelvin)) / 100;
  let red: number;
  let green: number;
  let blue: number;
  if (temperature <= 66) {
    red = 255;
    green = 99.47 * Math.log(temperature) - 161.12;
    blue = temperature <= 19 ? 0 : 138.52 * Math.log(temperature - 10) - 305.04;
  } else {
    red = 329.7 * Math.pow(temperature - 60, -0.1332);
    green = 288.12 * Math.pow(temperature - 60, -0.0755);
    blue = 255;
  }
  return new THREE.Color(
    Math.min(255, Math.max(0, red)) / 255,
    Math.min(255, Math.max(0, green)) / 255,
    Math.min(255, Math.max(0, blue)) / 255,
  );
}

const MOOD_LIGHT: Record<SpaceDesignSpec["mood"], { ambient: number; spot: number }> = {
  warm_dim: { ambient: 0.85, spot: 1 },
  neutral: { ambient: 1, spot: 1 },
  cool_bright: { ambient: 1.25, spot: 0.9 },
  dramatic: { ambient: 0.65, spot: 1.2 },
};

/**
 * Every colour in the hall is derived from the model-chosen design, so a
 * generated exhibition keeps its own palette; this only adds the darker and
 * lighter steps that real architecture has (trim, reveals, ceiling, silk).
 */
interface HallPalette {
  light: THREE.Color;
  wall: THREE.Color;
  floor: THREE.Color;
  ceiling: THREE.Color;
  accent: THREE.Color;
  trim: THREE.Color;
  portal: THREE.Color;
  /** Screen walls: the wall colour leaning toward the accent, so they read as feature walls. */
  feature: THREE.Color;
  plinth: THREE.Color;
  mat: THREE.Color;
  silk: THREE.Color;
  brocade: THREE.Color;
  wood: THREE.Color;
  endWall: THREE.Color;
  fog: THREE.Color;
  void: THREE.Color;
  /** Lettering colours that read on the feature and end walls. */
  ink: string;
  inkAccent: string;
  floorKind: "boards" | "stone";
  ambient: number;
  spot: number;
}

function buildPalette(design: SpaceDesignSpec): HallPalette {
  const light = kelvinToColor(design.lightTemperature);
  const wall = new THREE.Color(design.wallColor);
  const floor = new THREE.Color(design.floorColor);
  const accent = new THREE.Color(design.accentColor);
  const mood = MOOD_LIGHT[design.mood] ?? MOOD_LIGHT.neutral;
  const intensity = Math.max(0.4, Math.min(1.4, design.lightIntensity || 0.85));
  const feature = wall.clone().lerp(accent, 0.16).multiplyScalar(0.95);
  const darkFeature = luminance(`#${feature.getHexString()}`) < 0.45;
  const accentInk = darkFeature
    ? accent.clone().lerp(new THREE.Color("#ffffff"), 0.45)
    : accent.clone().lerp(new THREE.Color("#000000"), 0.6);
  return {
    light,
    wall,
    floor,
    accent,
    feature,
    ceiling: new THREE.Color(design.ceilingColor).multiplyScalar(0.58),
    trim: floor.clone().multiplyScalar(0.62),
    portal: wall.clone().multiplyScalar(0.9),
    plinth: wall.clone().lerp(new THREE.Color("#ffffff"), 0.35),
    mat: new THREE.Color("#efe9dc").lerp(wall, 0.25),
    silk: new THREE.Color("#e9e1cc").lerp(accent, 0.12),
    brocade: accent.clone().lerp(new THREE.Color("#3a2c1c"), 0.55),
    wood: new THREE.Color("#4a3321"),
    endWall: wall.clone().lerp(accent, 0.12),
    fog: wall.clone().multiplyScalar(0.3).lerp(light, 0.08),
    void: wall.clone().multiplyScalar(0.12),
    ink: darkFeature ? "#f3ede2" : "#2a231b",
    inkAccent: `#${accentInk.getHexString()}`,
    floorKind: luminance(design.floorColor) > 0.55 ? "stone" : "boards",
    ambient: intensity * mood.ambient,
    spot: intensity * mood.spot,
  };
}

/** A shared texture repeated over a surface of the given size in metres. */
function useRepeated(texture: THREE.Texture, width: number, height: number, tile: number) {
  return useMemo(() => {
    const copy = texture.clone();
    copy.repeat.set(width / tile, height / tile);
    copy.needsUpdate = true;
    return copy;
  }, [texture, width, height, tile]);
}

const ZH_NUMERALS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"];
const EN_NUMERALS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"];

function partLabel(index: number, language: "zh" | "en"): string {
  const n = index + 1;
  if (language === "en") return `PART ${EN_NUMERALS[n]?.toUpperCase() ?? n}`;
  return `第${ZH_NUMERALS[n] ?? n}部分`;
}

/**
 * A museum CDN failure must not tear down the entire hall. Texture hooks throw
 * through Suspense after a rejected request, so each object gets its own error
 * boundary and can fall back to an honest neutral surface while the route
 * remains walkable. The HTML overlay still exposes the title and metadata.
 */
class ArtworkTextureBoundary extends Component<
  { children: ReactNode; fallback: ReactNode; textureUrl: string },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }

  componentDidCatch() {
    // suspend-react retains rejected loader entries. Clear only this image so
    // leaving and re-entering the hall can try the institution CDN again.
    useTexture.clear(this.props.textureUrl);
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

function configureTexture(loaded: THREE.Texture | THREE.Texture[]) {
  const applyTo = (candidate: THREE.Texture) => {
    candidate.colorSpace = THREE.SRGBColorSpace;
    candidate.anisotropy = 8;
    candidate.needsUpdate = true;
  };
  if (Array.isArray(loaded)) loaded.forEach(applyTo);
  else applyTo(loaded);
}

/**
 * Loads an object image and hands it to `render`. Configured in the load
 * callback: drei caches textures by URL, so the artwork, its thumbnail on the
 * retrospective wall and a re-entry all share one upload.
 */
function WithObjectTexture({
  url,
  render,
}: {
  url: string;
  render: (texture: THREE.Texture | null) => ReactNode;
}) {
  return (
    <ArtworkTextureBoundary key={url} textureUrl={url} fallback={render(null)}>
      <Suspense fallback={render(null)}>
        <LoadedTexture url={url} render={render} />
      </Suspense>
    </ArtworkTextureBoundary>
  );
}

function LoadedTexture({ url, render }: { url: string; render: (texture: THREE.Texture) => ReactNode }) {
  const texture = useTexture(url, configureTexture);
  return <>{render(texture)}</>;
}

/**
 * The image's own proportions fitted inside the slot's box. Until it has
 * loaded (or if it fails) the box itself is used.
 */
function fitInside(texture: THREE.Texture | null, maxWidth: number, maxHeight: number) {
  const image = texture?.image as { width?: number; height?: number } | undefined;
  if (!image?.width || !image?.height) return { width: maxWidth, height: maxHeight };
  const aspect = image.width / image.height;
  return aspect > maxWidth / maxHeight
    ? { width: maxWidth, height: maxWidth / aspect }
    : { width: maxHeight * aspect, height: maxHeight };
}

type Mount = "vitrine" | "hanging_scroll" | "handscroll" | "framed";

/** How an object is presented. Scrolls get a mount whatever the hall style. */
function mountFor(item: ExhibitionItem): Mount {
  if (isVitrineObject(item)) return "vitrine";
  const text = `${item.object.type} ${item.object.classification ?? ""} ${item.object.medium}`.toLowerCase();
  if (text.includes("handscroll")) return "handscroll";
  if (text.includes("hanging scroll") || text.includes("scroll")) return "hanging_scroll";
  return "framed";
}

const FRAME_STYLE: Record<
  SpaceDesignSpec["frameStyle"],
  { border: number; depth: number; mat: number; color: string; metalness: number; roughness: number }
> = {
  thin_dark: { border: 0.045, depth: 0.06, mat: 0.13, color: "#1d1813", metalness: 0.15, roughness: 0.55 },
  wide_gold: { border: 0.13, depth: 0.1, mat: 0.07, color: "#b08c4c", metalness: 0.85, roughness: 0.32 },
  scroll_hanging: { border: 0.04, depth: 0.05, mat: 0.11, color: "#7a5a3a", metalness: 0.05, roughness: 0.7 },
  vitrine: { border: 0.045, depth: 0.06, mat: 0.13, color: "#2a2620", metalness: 0.35, roughness: 0.45 },
};

/** Where every mount sits relative to the surface behind it (see layout.ts). */
const WALL_OFFSET = MOUNT_OFFSET;

function SoftShadow({
  width,
  height,
  position,
  opacity,
  rotation,
}: {
  width: number;
  height: number;
  position: [number, number, number];
  opacity: number;
  rotation?: [number, number, number];
}) {
  const texture = useMemo(() => softRectTexture("dark"), []);
  return (
    <mesh position={position} rotation={rotation} raycast={() => null}>
      {/* The texture's blurred edge occupies its outer fifth. */}
      <planeGeometry args={[width / 0.6, height / 0.6]} />
      <meshBasicMaterial map={texture} transparent opacity={opacity} depthWrite={false} toneMapped={false} />
    </mesh>
  );
}

function ArtworkImage({
  width,
  height,
  texture,
  palette,
  onSelect,
  z,
}: {
  width: number;
  height: number;
  texture: THREE.Texture | null;
  palette: HallPalette;
  onSelect: () => void;
  z: number;
}) {
  return (
    <mesh
      position={[0, 0, z]}
      onClick={(event) => {
        event.stopPropagation();
        onSelect();
      }}
      onPointerOver={(event) => {
        event.stopPropagation();
        document.body.style.cursor = "pointer";
      }}
      onPointerOut={() => {
        document.body.style.cursor = "auto";
      }}
    >
      <planeGeometry args={[width, height]} />
      {texture ? (
        // Unlit and not tone-mapped: the institution's photograph keeps its colours.
        <meshBasicMaterial map={texture} toneMapped={false} />
      ) : (
        <meshStandardMaterial
          name="museum-image-unavailable"
          color={palette.mat}
          emissive={palette.accent}
          emissiveIntensity={0.08}
          roughness={1}
        />
      )}
    </mesh>
  );
}

interface MountProps {
  placement: ArtworkPlacement;
  design: SpaceDesignSpec;
  palette: HallPalette;
  texture: THREE.Texture | null;
  onSelect: () => void;
}

/** A frame built from four bevelled rails, sized to the image, with a recessed mat. */
function FramedMount({ placement, design, palette, texture, onSelect }: MountProps) {
  const style = FRAME_STYLE[design.frameStyle] ?? FRAME_STYLE.thin_dark;
  const { width, height } = fitInside(texture, placement.width, placement.height);
  const innerW = width + style.mat * 2;
  const innerH = height + style.mat * 2;
  const outerW = innerW + style.border * 2;
  const outerH = innerH + style.border * 2;
  const railZ = -WALL_OFFSET + style.depth / 2;
  const matZ = -WALL_OFFSET + 0.012;
  const frameMaterial = (
    <meshStandardMaterial color={style.color} metalness={style.metalness} roughness={style.roughness} />
  );
  const matColor = design.frameStyle === "scroll_hanging" ? palette.silk : palette.mat;

  return (
    <group>
      <SoftShadow width={outerW} height={outerH} position={[0.04, -0.07, -WALL_OFFSET + 0.004]} opacity={0.42} />
      {/* Rails: top, bottom, left, right. */}
      <mesh position={[0, (innerH + style.border) / 2, railZ]}>
        <boxGeometry args={[outerW, style.border, style.depth]} />
        {frameMaterial}
      </mesh>
      <mesh position={[0, -(innerH + style.border) / 2, railZ]}>
        <boxGeometry args={[outerW, style.border, style.depth]} />
        {frameMaterial}
      </mesh>
      {[-1, 1].map((side) => (
        <mesh key={side} position={[side * (innerW + style.border) / 2, 0, railZ]}>
          <boxGeometry args={[style.border, innerH, style.depth]} />
          {frameMaterial}
        </mesh>
      ))}
      <mesh position={[0, 0, matZ]}>
        <planeGeometry args={[innerW, innerH]} />
        <meshStandardMaterial color={matColor} roughness={0.95} />
      </mesh>
      <ArtworkImage width={width} height={height} texture={texture} palette={palette} onSelect={onSelect} z={matZ + 0.004} />
    </group>
  );
}

/**
 * A hanging-scroll mount: silk surround, darker brocade bands above and below,
 * a thin top rod and a weighted roller with knobs. Proportions follow the
 * common Chinese format rather than any one object.
 */
function HangingScrollMount({ placement, palette, texture, onSelect }: MountProps) {
  const { width, height } = fitInside(texture, placement.width, placement.height);
  const side = 0.08;
  const top = 0.42;
  const bottom = 0.24;
  const mountW = width + side * 2;
  const mountH = height + top + bottom;
  const centerY = (top - bottom) / 2;
  const topY = height / 2 + top;
  const bottomY = -height / 2 - bottom;
  const woodMaterial = <meshStandardMaterial color={palette.wood} roughness={0.55} />;

  return (
    <group>
      <SoftShadow width={mountW} height={mountH} position={[0.03, centerY - 0.06, -WALL_OFFSET + 0.004]} opacity={0.36} />
      <mesh position={[0, centerY, -0.012]}>
        <planeGeometry args={[mountW, mountH]} />
        <meshStandardMaterial color={palette.silk} roughness={0.9} />
      </mesh>
      <mesh position={[0, topY - 0.16, -0.008]}>
        <planeGeometry args={[mountW, 0.3]} />
        <meshStandardMaterial color={palette.brocade} roughness={0.85} />
      </mesh>
      <mesh position={[0, bottomY + 0.09, -0.008]}>
        <planeGeometry args={[mountW, 0.16]} />
        <meshStandardMaterial color={palette.brocade} roughness={0.85} />
      </mesh>
      <mesh position={[0, topY, 0.004]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.016, 0.016, mountW + 0.04, 12]} />
        {woodMaterial}
      </mesh>
      <mesh position={[0, bottomY, 0.012]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.028, 0.028, mountW + 0.06, 16]} />
        {woodMaterial}
      </mesh>
      {[-1, 1].map((end) => (
        <mesh key={end} position={[end * (mountW / 2 + 0.07), bottomY, 0.012]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.042, 0.042, 0.08, 16]} />
          <meshStandardMaterial color={palette.brocade} metalness={0.3} roughness={0.5} />
        </mesh>
      ))}
      {/* Hanging cord from the rod to a hook under the picture rail. */}
      {[-1, 1].map((end) => (
        <mesh
          key={`cord-${end}`}
          position={[end * mountW * 0.16, topY + 0.14, -0.02]}
          rotation={[0, 0, end * 0.55]}
          raycast={() => null}
        >
          <cylinderGeometry args={[0.004, 0.004, 0.34, 6]} />
          <meshStandardMaterial color={palette.brocade} roughness={0.8} />
        </mesh>
      ))}
      <ArtworkImage width={width} height={height} texture={texture} palette={palette} onSelect={onSelect} z={-0.004} />
    </group>
  );
}

/** A handscroll shown opened flat, with a roller at each end. */
function HandscrollMount({ placement, palette, texture, onSelect }: MountProps) {
  const { width, height } = fitInside(texture, placement.width, placement.height);
  const mountW = width + 0.5;
  const mountH = height + 0.18;
  return (
    <group>
      <SoftShadow width={mountW} height={mountH} position={[0.03, -0.06, -WALL_OFFSET + 0.004]} opacity={0.36} />
      <mesh position={[0, 0, -0.012]}>
        <planeGeometry args={[mountW, mountH]} />
        <meshStandardMaterial color={palette.silk} roughness={0.9} />
      </mesh>
      <mesh position={[-(mountW / 2) + 0.12, 0, -0.008]}>
        <planeGeometry args={[0.2, mountH]} />
        <meshStandardMaterial color={palette.brocade} roughness={0.85} />
      </mesh>
      {[-1, 1].map((end) => (
        <mesh key={end} position={[end * (mountW / 2 + 0.02), 0, 0.012]}>
          <cylinderGeometry args={[0.032, 0.032, mountH + 0.08, 16]} />
          <meshStandardMaterial color={palette.wood} roughness={0.55} />
        </mesh>
      ))}
      <ArtworkImage width={width} height={height} texture={texture} palette={palette} onSelect={onSelect} z={-0.004} />
    </group>
  );
}

/**
 * Three-dimensional objects: the institution's photograph stands at the back
 * of a glass case on a plinth. No 3D model is fabricated — none of the
 * institutions publish scan data, so the photograph is what we can show. The
 * case keeps the slot's size; only the photograph takes its own proportions.
 */
function VitrineMount({ placement, palette, texture, onSelect }: MountProps) {
  const { width, height } = placement;
  const image = fitInside(texture, width, height);
  const centerY = placement.position[1];
  const caseDepth = 0.72;
  const caseZ = -WALL_OFFSET + caseDepth / 2;
  const plinthTop = -height / 2 - 0.08;
  const plinthHeight = centerY + plinthTop;
  // The case closes over the photograph, so a wide, low object does not sit
  // at the bottom of a tall empty case.
  const caseTop = plinthTop + 0.05 + image.height + 0.2;
  const caseHeight = caseTop - plinthTop;
  const caseW = width + 0.34;

  return (
    <group>
      {/* Plinth, from the floor to the base of the case. */}
      <mesh position={[0, plinthTop - plinthHeight / 2, caseZ]}>
        <boxGeometry args={[caseW + 0.08, plinthHeight, caseDepth + 0.06]} />
        <meshStandardMaterial color={palette.plinth} roughness={0.85} />
      </mesh>
      <mesh position={[0, -centerY + 0.05, caseZ]}>
        <boxGeometry args={[caseW + 0.1, 0.1, caseDepth + 0.08]} />
        <meshStandardMaterial color={palette.trim} roughness={0.7} />
      </mesh>
      <SoftShadow
        width={caseW + 0.3}
        height={caseDepth + 0.3}
        position={[0, -centerY + 0.004, caseZ + 0.08]}
        rotation={[-Math.PI / 2, 0, 0]}
        opacity={0.5}
      />
      {/* Back panel the photograph stands against. */}
      <mesh position={[0, (caseTop + plinthTop) / 2, -WALL_OFFSET + 0.012]}>
        <planeGeometry args={[caseW - 0.04, caseHeight - 0.02]} />
        <meshStandardMaterial color={palette.mat} roughness={0.95} />
      </mesh>
      {/* The photograph stands on the plinth, not floating mid-case. */}
      <group position={[0, plinthTop + 0.05 + image.height / 2, 0]}>
        <ArtworkImage
          width={image.width}
          height={image.height}
          texture={texture}
          palette={palette}
          onSelect={onSelect}
          z={-WALL_OFFSET + 0.03}
        />
      </group>
      {/* Glass. Never a click target, or it would swallow clicks on the image. */}
      <mesh position={[0, (caseTop + plinthTop) / 2, caseZ + 0.006]} raycast={() => null} renderOrder={2}>
        <boxGeometry args={[caseW, caseHeight, caseDepth - 0.012]} />
        <meshPhysicalMaterial
          color="#ffffff"
          transparent
          opacity={0.1}
          roughness={0.04}
          metalness={0}
          envMapIntensity={1.6}
          depthWrite={false}
        />
      </mesh>
      {/* A lit hood, as real cases have, and slim posts so the glass reads. */}
      <mesh position={[0, caseTop + 0.03, caseZ]} raycast={() => null}>
        <boxGeometry args={[caseW + 0.02, 0.06, caseDepth + 0.02]} />
        <meshStandardMaterial
          color={palette.plinth}
          emissive={palette.light}
          emissiveIntensity={0.28}
          roughness={0.8}
        />
      </mesh>
      {[-1, 1].map((edge) => (
        <mesh
          key={edge}
          position={[edge * (caseW / 2), (caseTop + plinthTop) / 2, caseZ + caseDepth / 2 - 0.006]}
          raycast={() => null}
        >
          <boxGeometry args={[0.014, caseHeight, 0.014]} />
          <meshStandardMaterial color={palette.trim} metalness={0.5} roughness={0.35} />
        </mesh>
      ))}
    </group>
  );
}

/**
 * A spot light's target must sit in the scene graph for its world matrix to
 * update; one that is only given a position aims at the world origin. Parent
 * it to the object instead.
 */
function AimedSpotLight({
  target,
  ...props
}: Omit<ThreeElements["spotLight"], "target"> & { target: [number, number, number] }) {
  const [aim] = useState(() => new THREE.Object3D());
  return (
    <>
      <primitive object={aim} position={target} />
      <spotLight target={aim} {...props} />
    </>
  );
}

function Artwork({
  placement,
  design,
  palette,
  active,
  onSelect,
}: Omit<MountProps, "texture"> & { active: boolean }) {
  const mount = mountFor(placement.item);
  const glow = useMemo(() => softRectTexture("light"), []);
  const centerY = placement.position[1];
  // The picture light hangs from the ceiling, aimed at the object.
  const lampY = WALL_HEIGHT - 0.42 - centerY;
  const lampZ = 1.28;
  const aim = Math.atan2(-lampZ, -lampY);
  const outline = mount === "vitrine" ? 0.34 : 0.26;
  // Mostly the lamp's own colour with a hint of the accent: a saturated
  // accent (a red, say) turns the wall behind the object into a coloured blob.
  const glowColor = useMemo(() => palette.light.clone().lerp(palette.accent, 0.3), [palette]);
  // The key object of a chapter gets the stronger light, as it would in a gallery.
  const keyLight = placement.side === "screen" ? 1.35 : 1;
  // Textures come from our own proxy: institution CDNs do not all send CORS
  // headers, and WebGL cannot sample a tainted image.
  const textureUrl = resolveObjectImageUrl(placement.item.object.id, 1024);

  return (
    <group position={placement.position} rotation={[0, placement.rotationY, 0]}>
      <WithObjectTexture
        url={textureUrl}
        render={(texture) => {
          const props = { placement, design, palette, texture, onSelect };
          if (mount === "vitrine") return <VitrineMount {...props} />;
          if (mount === "hanging_scroll") return <HangingScrollMount {...props} />;
          if (mount === "handscroll") return <HandscrollMount {...props} />;
          return <FramedMount {...props} />;
        }}
      />

      {/* The current stop: a soft wash of light on the wall behind it. */}
      {active && (
        <mesh position={[0, mount === "vitrine" ? 0.05 : 0, -WALL_OFFSET + 0.002]} raycast={() => null}>
          <planeGeometry args={[(placement.width + outline * 2) / 0.6, (placement.height + outline * 2) / 0.6]} />
          <meshBasicMaterial
            map={glow}
            color={glowColor}
            transparent
            opacity={0.32}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      )}

      {/* Track light: housing, stem to the ceiling, and the light itself. */}
      <group position={[0, lampY, lampZ]}>
        <mesh rotation={[aim, 0, 0]} raycast={() => null}>
          <cylinderGeometry args={[0.05, 0.036, 0.17, 14]} />
          <meshStandardMaterial color="#1c1a17" metalness={0.5} roughness={0.4} />
        </mesh>
        <mesh position={[0, 0.21, 0]} raycast={() => null}>
          <cylinderGeometry args={[0.008, 0.008, 0.42, 6]} />
          <meshStandardMaterial color="#1c1a17" metalness={0.5} roughness={0.4} />
        </mesh>
      </group>
      <AimedSpotLight
        position={[0, lampY - 0.06, lampZ - 0.03]}
        target={[0, 0, 0]}
        angle={0.5}
        penumbra={0.85}
        intensity={(active ? 14 : 8) * keyLight * palette.spot}
        distance={9}
        decay={1.6}
        color={palette.light}
      />
    </group>
  );
}

/**
 * One physical shell for the whole visit. Chapter zones never draw their own
 * floor, ceiling, side walls, or end caps, so there are no seams or phantom
 * walls between consecutive parts of the argument.
 */
function HallShell({
  layout,
  palette,
  quality,
}: {
  layout: HallLayout;
  palette: HallPalette;
  quality: HallQuality;
}) {
  const firstRoom = layout.rooms[0];
  const width = firstRoom?.width ?? 11;
  const zStart = firstRoom?.zStart ?? 0;
  const zEnd = layout.endZ;
  const depth = zStart - zEnd;
  const centerZ = (zStart + zEnd) / 2;
  const halfWidth = width / 2;
  const entranceSideWidth = (width - DOORWAY_WIDTH) / 2;
  const approachDepth = Math.max(1, layout.lobbyViewpoint[2] - zStart + 1.5);
  const facadeWidth = Math.min(width - 1.2, 4.8);
  const facadeTitle = useMemo(
    () => wallTextTexture([{ text: layout.title, size: 0.34, color: palette.ink, weight: 600 }], facadeWidth / 0.9),
    [facadeWidth, layout.title, palette.ink],
  );

  const floorMap = useRepeated(floorTexture(palette.floorKind), width, depth, FLOOR_TILE_METRES);
  const approachMap = useRepeated(
    floorTexture(palette.floorKind),
    DOORWAY_WIDTH + 0.8,
    approachDepth,
    FLOOR_TILE_METRES,
  );
  const sideWallMap = useRepeated(plasterTexture(), depth, WALL_HEIGHT, PLASTER_TILE_METRES);
  const endWallMap = useRepeated(plasterTexture(), width, WALL_HEIGHT, PLASTER_TILE_METRES);

  if (!firstRoom) return null;

  const wallMaterial = (map: THREE.Texture, color = palette.wall) => (
    <meshStandardMaterial color={color} map={map} roughness={0.93} side={THREE.DoubleSide} />
  );
  // Picture-light track positions, shared with the artworks.
  const trackX = halfWidth - WALL_OFFSET - 1.28;

  return (
    <group>
      {/* A single continuous floor from prologue to epilogue. */}
      <mesh position={[0, 0, centerZ]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[width, depth]} />
        {quality === "high" ? (
          <MeshReflectorMaterial
            map={floorMap}
            color={palette.floor}
            resolution={512}
            blur={[420, 140]}
            mixBlur={1}
            mixStrength={palette.floorKind === "stone" ? 0.9 : 1.4}
            mirror={0}
            depthScale={0.4}
            minDepthThreshold={0.4}
            maxDepthThreshold={1.2}
            roughness={0.82}
            metalness={0.04}
          />
        ) : (
          <meshStandardMaterial map={floorMap} color={palette.floor} roughness={0.72} />
        )}
      </mesh>

      <mesh position={[0, WALL_HEIGHT, centerZ]} rotation={[Math.PI / 2, 0, 0]}>
        <planeGeometry args={[width, depth]} />
        <meshStandardMaterial color={palette.ceiling} roughness={1} />
      </mesh>

      {/* Light coves along both edges of the ceiling. */}
      {[-1, 1].map((side) => (
        <mesh
          key={`cove-${side}`}
          position={[side * (halfWidth - 0.32), WALL_HEIGHT - 0.012, centerZ]}
          rotation={[Math.PI / 2, 0, 0]}
          raycast={() => null}
        >
          <planeGeometry args={[0.1, depth - 0.2]} />
          <meshBasicMaterial color={palette.light} toneMapped={false} />
        </mesh>
      ))}

      {/* Picture-light tracks the side-wall lamps hang from. */}
      {[-1, 1].map((side) => (
        <mesh key={`track-${side}`} position={[side * trackX, WALL_HEIGHT - 0.02, centerZ]} raycast={() => null}>
          <boxGeometry args={[0.045, 0.035, depth - 0.3]} />
          <meshStandardMaterial color="#1c1a17" metalness={0.5} roughness={0.45} />
        </mesh>
      ))}

      {/* One uninterrupted pair of side walls, with skirting and a shadow gap. */}
      {[-1, 1].map((side) => (
        <group key={side}>
          <mesh
            position={[side * halfWidth, WALL_HEIGHT / 2, centerZ]}
            rotation={[0, -side * (Math.PI / 2), 0]}
          >
            <planeGeometry args={[depth, WALL_HEIGHT]} />
            {wallMaterial(sideWallMap)}
          </mesh>
          <mesh position={[side * (halfWidth - 0.015), 0.07, centerZ]} raycast={() => null}>
            <boxGeometry args={[0.03, 0.14, depth]} />
            <meshStandardMaterial color={palette.trim} roughness={0.6} />
          </mesh>
          <mesh position={[side * (halfWidth - 0.004), WALL_HEIGHT - 0.05, centerZ]} raycast={() => null}>
            <boxGeometry args={[0.008, 0.035, depth]} />
            <meshBasicMaterial color={palette.void} />
          </mesh>
        </group>
      ))}

      {/* The only opening wall is the entrance, split around a real doorway. */}
      <group position={[0, 0, zStart]}>
        {[-1, 1].map((side) => (
          <mesh
            key={side}
            position={[side * (DOORWAY_WIDTH / 2 + entranceSideWidth / 2), WALL_HEIGHT / 2, 0]}
          >
            <planeGeometry args={[entranceSideWidth, WALL_HEIGHT]} />
            {wallMaterial(endWallMap)}
          </mesh>
        ))}
        <mesh position={[0, DOORWAY_HEIGHT + (WALL_HEIGHT - DOORWAY_HEIGHT) / 2, 0]}>
          <planeGeometry args={[DOORWAY_WIDTH, WALL_HEIGHT - DOORWAY_HEIGHT]} />
          {wallMaterial(endWallMap)}
        </mesh>
        {/* A deep reveal makes the doorway read as a threshold, not a hole. */}
        {[-1, 1].map((side) => (
          <mesh key={`jamb-${side}`} position={[side * (DOORWAY_WIDTH / 2 + 0.09), DOORWAY_HEIGHT / 2, 0]}>
            <boxGeometry args={[0.18, DOORWAY_HEIGHT, 0.34]} />
            <meshStandardMaterial color={palette.portal} roughness={0.85} />
          </mesh>
        ))}
        <mesh position={[0, DOORWAY_HEIGHT + 0.09, 0]}>
          <boxGeometry args={[DOORWAY_WIDTH + 0.36, 0.18, 0.34]} />
          <meshStandardMaterial color={palette.portal} roughness={0.85} />
        </mesh>
        {/* The exhibition's name over the door, read from the lobby. */}
        <mesh position={[0, DOORWAY_HEIGHT + 0.78, 0.012]} raycast={() => null}>
          <planeGeometry args={[facadeWidth, 0.9]} />
          <meshBasicMaterial map={facadeTitle} transparent toneMapped={false} depthWrite={false} />
        </mesh>
        <AimedSpotLight
          position={[0, WALL_HEIGHT - 0.2, 3.2]}
          target={[0, DOORWAY_HEIGHT + 0.5, 0]}
          angle={0.7}
          penumbra={1}
          intensity={8 * palette.spot}
          distance={9}
          decay={1.5}
          color={palette.light}
        />
      </group>

      {/* A narrow approach makes the exterior lobby viewpoint spatially legible. */}
      <mesh position={[0, 0, zStart + approachDepth / 2]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[DOORWAY_WIDTH + 0.8, approachDepth]} />
        <meshStandardMaterial map={approachMap} color={palette.floor} roughness={0.8} />
      </mesh>
    </group>
  );
}

/** Piers and a lintel frame the next part like a gallery opening. */
function Portal({ z, width, palette }: { z: number; width: number; palette: HallPalette }) {
  const halfWidth = width / 2;
  const pier = 0.46;
  const pierDepth = 0.5;
  const lintel = 0.62;
  return (
    <group position={[0, 0, z]}>
      {/* A brass inlay marks the threshold without interrupting the aisle. */}
      <mesh position={[0, 0.006, -0.05]} raycast={() => null}>
        <boxGeometry args={[width - 1.2, 0.012, 0.05]} />
        <meshStandardMaterial color={palette.accent} metalness={0.7} roughness={0.35} />
      </mesh>
      {[-1, 1].map((side) => (
        <mesh key={side} position={[side * (halfWidth - pier / 2), WALL_HEIGHT / 2, 0]}>
          <boxGeometry args={[pier, WALL_HEIGHT, pierDepth]} />
          <meshStandardMaterial color={palette.portal} roughness={0.88} />
        </mesh>
      ))}
      <mesh position={[0, WALL_HEIGHT - lintel / 2, 0]}>
        <boxGeometry args={[width, lintel, pierDepth]} />
        <meshStandardMaterial color={palette.portal} roughness={0.88} />
      </mesh>
      {[-1, 1].map((face) => (
        <mesh
          key={`inlay-${face}`}
          position={[0, WALL_HEIGHT - lintel + 0.049, face * (pierDepth / 2 + 0.002)]}
          raycast={() => null}
        >
          <planeGeometry args={[width - pier * 2, 0.03]} />
          <meshStandardMaterial
            color={palette.accent}
            emissive={palette.accent}
            emissiveIntensity={0.25}
            metalness={0.6}
            roughness={0.4}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
    </group>
  );
}

/**
 * The chapter's screen wall (对景墙): a freestanding feature wall across the
 * axis, so the view down every section ends on its key object instead of an
 * empty wall. The section title is lettered along the top, the way a gallery
 * names a room. Its back carries the part numeral, seen when looking back.
 */
function ScreenWall({ room, layout, palette }: { room: RoomLayout; layout: HallLayout; palette: HallPalette }) {
  const { screen } = room;
  const bandWidth = screen.width - 0.4;
  const bandHeight = 0.56;
  const title = useMemo(
    () =>
      wallTextTexture(
        [
          {
            text: partLabel(room.index, layout.language),
            size: 0.2,
            color: palette.inkAccent,
            weight: 700,
            tracking: 0.32,
            family: "sans",
          },
          { text: room.chapter.title, size: 0.42, color: palette.ink, weight: 600 },
        ],
        bandWidth / bandHeight,
      ),
    [bandWidth, layout.language, palette.ink, palette.inkAccent, room.chapter.title, room.index],
  );
  const numeral = useMemo(
    () =>
      wallTextTexture(
        [{ text: String(room.index + 1).padStart(2, "0"), size: 0.78, color: palette.inkAccent, weight: 300 }],
        1,
      ),
    [palette.inkAccent, room.index],
  );
  const map = useRepeated(plasterTexture(), screen.width, screen.height, PLASTER_TILE_METRES);

  return (
    <group position={[0, 0, screen.z]}>
      {/* Raised on a recessed shadow plinth so it reads as freestanding. */}
      <mesh position={[0, 0.06, 0]} raycast={() => null}>
        <boxGeometry args={[screen.width - 0.08, 0.12, screen.thickness - 0.06]} />
        <meshStandardMaterial color={palette.void} roughness={0.9} />
      </mesh>
      <mesh position={[0, 0.12 + (screen.height - 0.12) / 2, 0]}>
        <boxGeometry args={[screen.width, screen.height - 0.12, screen.thickness]} />
        <meshStandardMaterial color={palette.feature} map={map} roughness={0.92} />
      </mesh>
      <mesh position={[0, screen.height - 0.03 - bandHeight / 2, screen.thickness / 2 + 0.004]} raycast={() => null}>
        <planeGeometry args={[bandWidth, bandHeight]} />
        <meshBasicMaterial map={title} transparent toneMapped={false} depthWrite={false} />
      </mesh>
      <mesh
        position={[0, 2.2, -screen.thickness / 2 - 0.004]}
        rotation={[0, Math.PI, 0]}
        raycast={() => null}
      >
        <planeGeometry args={[1.5, 1.5]} />
        <meshBasicMaterial map={numeral} transparent opacity={0.55} toneMapped={false} depthWrite={false} />
      </mesh>
      <SoftShadow
        width={screen.width + 0.2}
        height={screen.thickness + 0.5}
        position={[0, 0.004, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        opacity={0.45}
      />
    </group>
  );
}

/** Lighting and the free-standing furniture of one chapter. */
function ChapterZone({
  room,
  layout,
  palette,
}: {
  room: RoomLayout;
  layout: HallLayout;
  palette: HallPalette;
}) {
  return (
    <group>
      {room.index > 0 && <Portal z={room.zStart} width={room.width} palette={palette} />}
      {/* The entrance threshold is the doorway; it still gets the inlay. */}
      {room.index === 0 && (
        <mesh position={[0, 0.006, room.zStart - 0.05]} raycast={() => null}>
          <boxGeometry args={[DOORWAY_WIDTH, 0.012, 0.05]} />
          <meshStandardMaterial color={palette.accent} metalness={0.7} roughness={0.35} />
        </mesh>
      )}
      <ScreenWall room={room} layout={layout} palette={palette} />

      {/* A bounded pool of light gives each section its own atmosphere. */}
      <pointLight
        position={[0, WALL_HEIGHT - 0.7, (room.zStart + room.screen.z) / 2]}
        intensity={10 * palette.ambient}
        distance={Math.max(7, Math.min(13, room.depth * 0.8))}
        decay={1.7}
        color={palette.light}
      />
    </group>
  );
}

/** One square thumbnail on the retrospective wall; it also jumps to its object. */
function RetrospectiveTile({
  placement,
  x,
  y,
  size,
  palette,
  onSelect,
}: {
  placement: ArtworkPlacement;
  x: number;
  y: number;
  size: number;
  palette: HallPalette;
  onSelect: () => void;
}) {
  const number = useMemo(
    () =>
      wallTextTexture(
        [{ text: String(placement.tourIndex + 1).padStart(2, "0"), size: 0.6, color: palette.ink, weight: 500, family: "sans" }],
        2,
      ),
    [palette.ink, placement.tourIndex],
  );
  const inner = size - 0.14;
  return (
    <group position={[x, y, 0]}>
      <mesh position={[0, 0, 0.02]}>
        <boxGeometry args={[size, size, 0.04]} />
        <meshStandardMaterial color="#1d1813" roughness={0.55} metalness={0.15} />
      </mesh>
      <mesh position={[0, 0, 0.041]}>
        <planeGeometry args={[size - 0.05, size - 0.05]} />
        <meshStandardMaterial color={palette.mat} roughness={0.95} />
      </mesh>
      <WithObjectTexture
        url={resolveObjectImageUrl(placement.item.object.id, 1024)}
        render={(texture) => {
          const fitted = fitInside(texture, inner, inner);
          return (
            <ArtworkImage
              width={fitted.width}
              height={fitted.height}
              texture={texture}
              palette={palette}
              onSelect={onSelect}
              z={0.044}
            />
          );
        }}
      />
      <mesh position={[0, -size / 2 - 0.11, 0.004]} raycast={() => null}>
        <planeGeometry args={[0.3, 0.15]} />
        <meshBasicMaterial map={number} transparent toneMapped={false} depthWrite={false} />
      </mesh>
    </group>
  );
}

/**
 * The closing room: a portal, then the retrospective wall (回望墙) with every
 * object of the visit in tour order under the exhibition's name. Each
 * thumbnail is a way back to its object.
 */
function EpilogueRoom({
  layout,
  palette,
  onSelectItem,
}: {
  layout: HallLayout;
  palette: HallPalette;
  onSelectItem: (itemId: string) => void;
}) {
  const width = layout.rooms[0]?.width ?? 11;
  const { epilogueZone, endZ, artworks } = layout;
  const count = artworks.length;
  const rows = count > 7 ? 2 : 1;
  const perRow = Math.ceil(count / rows);
  const size = Math.min(0.72, (width - 2.4) / (perRow * 1.3));
  const gap = size * 0.3;
  const rowYs = rows === 1 ? [1.85] : [2.25, 1.3];
  const headingWidth = Math.min(width - 2, 5);
  const heading = useMemo(
    () =>
      wallTextTexture(
        [
          {
            text: layout.language === "en" ? "LOOKING BACK" : "回望",
            size: 0.16,
            color: palette.inkAccent,
            weight: 500,
            tracking: 0.4,
            family: "sans",
          },
          { text: layout.title, size: 0.36, color: palette.ink, weight: 600 },
        ],
        headingWidth / 0.9,
      ),
    [headingWidth, layout.language, layout.title, palette.ink, palette.inkAccent],
  );
  const endWallMap = useRepeated(plasterTexture(), width, WALL_HEIGHT, PLASTER_TILE_METRES);

  return (
    <group>
      <Portal z={epilogueZone.zStart} width={width} palette={palette} />
      <mesh position={[0, WALL_HEIGHT / 2, endZ]}>
        <planeGeometry args={[width, WALL_HEIGHT]} />
        <meshStandardMaterial color={palette.endWall} map={endWallMap} roughness={0.93} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, 3.55, endZ + 0.01]} raycast={() => null}>
        <planeGeometry args={[headingWidth, 0.9]} />
        <meshBasicMaterial map={heading} transparent toneMapped={false} depthWrite={false} />
      </mesh>
      <group position={[0, 0, endZ + 0.01]}>
        {artworks.map((placement, index) => {
          const row = Math.floor(index / perRow);
          const inRow = row === rows - 1 ? count - perRow * (rows - 1) : perRow;
          const column = index - row * perRow;
          const rowWidth = inRow * size + (inRow - 1) * gap;
          return (
            <RetrospectiveTile
              key={placement.itemId}
              placement={placement}
              x={-rowWidth / 2 + size / 2 + column * (size + gap)}
              y={rowYs[row]}
              size={size}
              palette={palette}
              onSelect={() => onSelectItem(placement.itemId)}
            />
          );
        })}
      </group>
      <AimedSpotLight
        position={[0, WALL_HEIGHT - 0.3, endZ + 4]}
        target={[0, 2.1, endZ]}
        angle={0.8}
        penumbra={1}
        intensity={12 * palette.spot}
        distance={12}
        decay={1.5}
        color={palette.light}
      />
      <pointLight
        position={[0, WALL_HEIGHT - 0.7, (epilogueZone.zStart + endZ) / 2]}
        intensity={8 * palette.ambient}
        distance={9}
        decay={1.7}
        color={palette.light}
      />
    </group>
  );
}

/**
 * Image-based lighting built from a few emissive panels rather than a
 * downloaded HDR: it gives frames, glass and the floor something to reflect
 * without a network request.
 */
function HallEnvironment({ palette }: { palette: HallPalette }) {
  return (
    <Environment resolution={128} frames={1} environmentIntensity={0.42}>
      <color attach="background" args={[palette.void]} />
      {[-3, 3].map((x) => (
        <Lightformer
          key={x}
          form="rect"
          intensity={2.2}
          color={palette.light}
          position={[x, 4, 0]}
          rotation-x={Math.PI / 2}
          scale={[0.6, 60, 1]}
        />
      ))}
      {[-1, 1].map((side) => (
        <Lightformer
          key={`wall-${side}`}
          form="rect"
          intensity={0.5}
          color={palette.wall}
          position={[side * 6, 1.8, 0]}
          rotation-y={-side * (Math.PI / 2)}
          scale={[60, 3.5, 1]}
        />
      ))}
      <Lightformer
        form="rect"
        intensity={0.35}
        color={palette.floor}
        position={[0, -1.5, 0]}
        rotation-x={-Math.PI / 2}
        scale={[14, 60, 1]}
      />
    </Environment>
  );
}

export function HallScene({
  layout,
  activeItemId,
  onSelectItem,
  quality = "high",
}: {
  layout: HallLayout;
  activeItemId: string | null;
  onSelectItem: (itemId: string) => void;
  quality?: HallQuality;
}) {
  const { design } = layout;
  const palette = useMemo(() => buildPalette(design), [design]);

  return (
    <>
      <color attach="background" args={[palette.void]} />
      <fog attach="fog" args={[palette.fog, 18, 64]} />
      <ambientLight intensity={0.22 * palette.ambient} color={palette.light} />
      <hemisphereLight intensity={0.28 * palette.ambient} color={palette.light} groundColor={palette.floor} />
      <HallEnvironment palette={palette} />

      <HallShell layout={layout} palette={palette} quality={quality} />

      {layout.rooms.map((room) => (
        <ChapterZone key={room.chapter.id} room={room} layout={layout} palette={palette} />
      ))}

      <EpilogueRoom layout={layout} palette={palette} onSelectItem={onSelectItem} />

      {layout.artworks.map((placement) => (
        <Artwork
          key={placement.itemId}
          placement={placement}
          design={design}
          palette={palette}
          active={placement.itemId === activeItemId}
          onSelect={() => onSelectItem(placement.itemId)}
        />
      ))}
    </>
  );
}
