"use client";

import { Component, Suspense, type ReactNode } from "react";
import * as THREE from "three";
import { useTexture } from "@react-three/drei";
import { resolveObjectImageUrl } from "@/lib/api";
import type { SpaceDesignSpec } from "@/lib/types";
import {
  DOORWAY_HEIGHT,
  DOORWAY_WIDTH,
  WALL_HEIGHT,
  type ArtworkPlacement,
  type HallLayout,
  type RoomLayout,
  isVitrineObject,
} from "./layout";

/** Frame profile per model-chosen style: [depth, border, colour]. */
const FRAME_PROFILE: Record<
  SpaceDesignSpec["frameStyle"],
  { depth: number; border: number; color: string; metalness: number; roughness: number }
> = {
  thin_dark: { depth: 0.06, border: 0.05, color: "#241f19", metalness: 0.1, roughness: 0.7 },
  wide_gold: { depth: 0.12, border: 0.16, color: "#a98244", metalness: 0.75, roughness: 0.35 },
  scroll_hanging: { depth: 0.04, border: 0.11, color: "#8a7c62", metalness: 0.05, roughness: 0.85 },
  vitrine: { depth: 0.08, border: 0.04, color: "#3b372f", metalness: 0.4, roughness: 0.45 },
};

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

function ArtworkUnavailableMaterial({
  design,
}: {
  design: SpaceDesignSpec;
}) {
  return (
    <meshStandardMaterial
      name="museum-image-unavailable"
      color={design.wallColor}
      emissive={design.accentColor}
      emissiveIntensity={0.08}
      roughness={1}
    />
  );
}

function ArtworkImageMaterial({ textureUrl }: { textureUrl: string }) {
  // Configured in the load callback rather than by mutating the returned
  // texture: drei caches textures across components, so this runs once per
  // image instead of on every render of every artwork that shares it.
  const texture = useTexture(textureUrl, (loaded) => {
    const applyTo = (candidate: THREE.Texture) => {
      candidate.colorSpace = THREE.SRGBColorSpace;
      candidate.anisotropy = 4;
      candidate.needsUpdate = true;
    };
    if (Array.isArray(loaded)) loaded.forEach(applyTo);
    else applyTo(loaded);
  });

  return <meshBasicMaterial map={texture} toneMapped={false} />;
}

function Artwork({
  placement,
  design,
  active,
  onSelect,
}: {
  placement: ArtworkPlacement;
  design: SpaceDesignSpec;
  active: boolean;
  onSelect: () => void;
}) {
  const vitrine = isVitrineObject(placement.item);
  const style = vitrine ? "vitrine" : design.frameStyle;
  const profile = FRAME_PROFILE[style];
  const { width, height } = placement;
  // Textures come from our own proxy: institution CDNs do not all send CORS
  // headers, and WebGL cannot sample a tainted image.
  const textureUrl = resolveObjectImageUrl(placement.item.object.id, 1024);

  return (
    <group position={placement.position} rotation={[0, placement.rotationY, 0]}>
      {/* Frame */}
      <mesh position={[0, 0, -profile.depth / 2]} castShadow>
        <boxGeometry args={[width + profile.border * 2, height + profile.border * 2, profile.depth]} />
        <meshStandardMaterial
          color={profile.color}
          metalness={profile.metalness}
          roughness={profile.roughness}
        />
      </mesh>

      {/* The institution image itself; never generated or redrawn. */}
      <mesh
        position={[0, 0, 0.01]}
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
        <ArtworkTextureBoundary
          key={textureUrl}
          textureUrl={textureUrl}
          fallback={<ArtworkUnavailableMaterial design={design} />}
        >
          <Suspense fallback={<ArtworkUnavailableMaterial design={design} />}>
            <ArtworkImageMaterial textureUrl={textureUrl} />
          </Suspense>
        </ArtworkTextureBoundary>
      </mesh>

      {/* Highlight ring when the item is the current tour stop. */}
      {active && (
        <mesh position={[0, 0, -profile.depth - 0.01]}>
          <planeGeometry args={[width + profile.border * 4, height + profile.border * 4]} />
          <meshBasicMaterial color={design.accentColor} transparent opacity={0.55} />
        </mesh>
      )}

      {/* Picture light */}
      <spotLight
        position={[0, height / 2 + 0.9, 1.1]}
        target-position={[0, 0, 0]}
        angle={0.62}
        penumbra={0.75}
        intensity={active ? 5.5 : 3.2}
        distance={7}
        color={kelvinToColor(design.lightTemperature)}
      />

      {/* Vitrine plinth for three-dimensional objects. */}
      {vitrine && (
        <mesh position={[0, -height / 2 - 0.42, 0.18]}>
          <boxGeometry args={[width + 0.3, 0.8, 0.5]} />
          <meshStandardMaterial color={design.floorColor} roughness={0.6} />
        </mesh>
      )}
    </group>
  );
}

/**
 * One physical shell for the whole visit. Chapter zones never draw their own
 * floor, ceiling, side walls, or end caps, so there are no seams or phantom
 * walls between consecutive parts of the argument.
 */
function HallShell({ layout, design }: { layout: HallLayout; design: SpaceDesignSpec }) {
  const firstRoom = layout.rooms[0];
  const lastRoom = layout.rooms[layout.rooms.length - 1];
  if (!firstRoom || !lastRoom) return null;

  const width = firstRoom.width;
  const halfWidth = width / 2;
  const zStart = firstRoom.zStart;
  const zEnd = lastRoom.zEnd;
  const depth = zStart - zEnd;
  const centerZ = (zStart + zEnd) / 2;
  const entranceSideWidth = (width - DOORWAY_WIDTH) / 2;
  const approachDepth = Math.max(1, layout.lobbyViewpoint[2] - zStart + 1.5);

  return (
    <group>
      {/* A single continuous floor and ceiling from prologue to epilogue. */}
      <mesh position={[0, 0, centerZ]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[width, depth]} />
        <meshStandardMaterial color={design.floorColor} roughness={0.92} />
      </mesh>
      <mesh position={[0, WALL_HEIGHT, centerZ]} rotation={[Math.PI / 2, 0, 0]}>
        <planeGeometry args={[width, depth]} />
        <meshStandardMaterial color={design.ceilingColor} roughness={1} />
      </mesh>

      {/* One uninterrupted pair of side walls. */}
      {[-1, 1].map((side) => (
        <mesh
          key={side}
          position={[side * halfWidth, WALL_HEIGHT / 2, centerZ]}
          rotation={[0, -side * (Math.PI / 2), 0]}
          receiveShadow
        >
          <planeGeometry args={[depth, WALL_HEIGHT]} />
          <meshStandardMaterial color={design.wallColor} roughness={0.95} side={THREE.DoubleSide} />
        </mesh>
      ))}

      {/* The only opening wall is the entrance, split around a real doorway. */}
      <group position={[0, 0, zStart]}>
        {[-1, 1].map((side) => (
          <mesh
            key={side}
            position={[side * (DOORWAY_WIDTH / 2 + entranceSideWidth / 2), WALL_HEIGHT / 2, 0]}
          >
            <planeGeometry args={[entranceSideWidth, WALL_HEIGHT]} />
            <meshStandardMaterial color={design.wallColor} roughness={0.95} side={THREE.DoubleSide} />
          </mesh>
        ))}
        <mesh position={[0, DOORWAY_HEIGHT + (WALL_HEIGHT - DOORWAY_HEIGHT) / 2, 0]}>
          <planeGeometry args={[DOORWAY_WIDTH, WALL_HEIGHT - DOORWAY_HEIGHT]} />
          <meshStandardMaterial color={design.wallColor} roughness={0.95} side={THREE.DoubleSide} />
        </mesh>
      </group>

      {/* A narrow approach makes the exterior lobby viewpoint spatially legible. */}
      <mesh
        position={[0, 0, zStart + approachDepth / 2]}
        rotation={[-Math.PI / 2, 0, 0]}
        receiveShadow
      >
        <planeGeometry args={[DOORWAY_WIDTH + 0.8, approachDepth]} />
        <meshStandardMaterial color={design.floorColor} roughness={0.94} />
      </mesh>

      {/* The epilogue has one final wall; no intermediate chapter has one. */}
      <mesh position={[0, WALL_HEIGHT / 2, zEnd]}>
        <planeGeometry args={[width, WALL_HEIGHT]} />
        <meshStandardMaterial color={design.wallColor} roughness={0.95} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

/** Non-blocking spatial punctuation for one chapter of the continuous route. */
function ChapterZone({ room, design }: { room: RoomLayout; design: SpaceDesignSpec }) {
  const { width, depth, center } = room;
  const halfWidth = width / 2;
  const isInteriorBoundary = room.index > 0;

  return (
    <group>
      {/* A flush threshold changes chapter without interrupting the aisle. */}
      <mesh position={[0, 0.018, room.zStart - 0.08]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[width - 0.7, 0.14]} />
        <meshBasicMaterial
          color={design.accentColor}
          transparent
          opacity={0.72}
          depthWrite={false}
        />
      </mesh>

      {/* Wall-edge posts and an overhead beam read as a portal but leave the
          full central path open at visitor height. */}
      {isInteriorBoundary && (
        <group position={[0, 0, room.zStart]}>
          {[-1, 1].map((side) => (
            <mesh key={side} position={[side * (halfWidth - 0.07), WALL_HEIGHT / 2, 0]}>
              <boxGeometry args={[0.14, WALL_HEIGHT, 0.16]} />
              <meshStandardMaterial
                color={design.accentColor}
                roughness={0.7}
                transparent
                opacity={0.62}
              />
            </mesh>
          ))}
          <mesh position={[0, WALL_HEIGHT - 0.09, 0]}>
            <boxGeometry args={[width, 0.18, 0.16]} />
            <meshStandardMaterial
              color={design.accentColor}
              roughness={0.7}
              transparent
              opacity={0.62}
            />
          </mesh>
        </group>
      )}

      {/* A bounded pool of warm light gives each narrative section its own
          atmosphere without introducing another enclosing room. */}
      <pointLight
        position={[0, WALL_HEIGHT - 0.55, center[2]]}
        intensity={design.lightIntensity * 11}
        distance={Math.max(7, Math.min(13, depth * 0.95))}
        decay={1.7}
        color={kelvinToColor(design.lightTemperature)}
      />
    </group>
  );
}

export function HallScene({
  layout,
  activeItemId,
  onSelectItem,
}: {
  layout: HallLayout;
  activeItemId: string | null;
  onSelectItem: (itemId: string) => void;
}) {
  const { design } = layout;

  return (
    <>
      <color attach="background" args={[design.floorColor]} />
      <fog attach="fog" args={[design.floorColor, 12, 46]} />
      <ambientLight intensity={design.lightIntensity * 0.55} color={kelvinToColor(design.lightTemperature)} />
      <hemisphereLight
        intensity={design.lightIntensity * 0.4}
        color={design.ceilingColor}
        groundColor={design.floorColor}
      />

      <HallShell layout={layout} design={design} />

      {layout.rooms.map((room) => (
        <ChapterZone key={room.chapter.id} room={room} design={design} />
      ))}

      {layout.artworks.map((placement) => (
        <Artwork
          key={placement.itemId}
          placement={placement}
          design={design}
          active={placement.itemId === activeItemId}
          onSelect={() => onSelectItem(placement.itemId)}
        />
      ))}
    </>
  );
}
