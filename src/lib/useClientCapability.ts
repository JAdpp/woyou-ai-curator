"use client";

import { useCallback, useSyncExternalStore } from "react";

/**
 * Client-only capability probes.
 *
 * These read browser APIs that do not exist during SSR. Doing that with
 * `useState` + `useEffect` causes a cascading render on every mount, so they go
 * through `useSyncExternalStore` instead: the server snapshot is the safe
 * default and the client snapshot is read during hydration.
 */

const noopSubscribe = () => () => {};

/** Track a media query, re-rendering when it changes. */
export function useMediaQuery(query: string, serverValue = false): boolean {
  const subscribe = useCallback(
    (onChange: () => void) => {
      if (typeof window === "undefined") return () => {};
      const list = window.matchMedia(query);
      list.addEventListener("change", onChange);
      return () => list.removeEventListener("change", onChange);
    },
    [query],
  );

  const getSnapshot = useCallback(() => {
    if (typeof window === "undefined") return serverValue;
    return window.matchMedia(query).matches;
  }, [query, serverValue]);

  return useSyncExternalStore(subscribe, getSnapshot, () => serverValue);
}

export function usePrefersReducedMotion(): boolean {
  return useMediaQuery("(prefers-reduced-motion: reduce)");
}

export function useIsTouchPrimary(): boolean {
  return useMediaQuery("(pointer: coarse)");
}

/**
 * Read a value that only exists on the client, once, without an effect.
 *
 * `read` must be stable and cheap — it is called on every render.
 */
export function useClientValue<T>(read: () => T, serverValue: T): T {
  return useSyncExternalStore(noopSubscribe, read, () => serverValue);
}

export type WebglCapability = "checking" | "ok" | "unavailable";

let cachedWebgl: WebglCapability | null = null;

/**
 * Probe for a usable WebGL context and enough hardware to run the hall.
 *
 * A device that creates a context but then renders at 8 fps is a worse outcome
 * than the text version, so core count is part of the test. The result is
 * cached: creating throwaway contexts on every render is expensive and can
 * exhaust the browser's context pool.
 */
export function detectWebgl(): WebglCapability {
  if (typeof window === "undefined") return "checking";
  if (cachedWebgl !== null) return cachedWebgl;
  try {
    const canvas = document.createElement("canvas");
    const context =
      canvas.getContext("webgl2") ??
      canvas.getContext("webgl") ??
      canvas.getContext("experimental-webgl");
    if (!context) {
      cachedWebgl = "unavailable";
      return cachedWebgl;
    }
    // Release the probe context immediately.
    const lose = (context as WebGLRenderingContext).getExtension("WEBGL_lose_context");
    lose?.loseContext();

    const cores = navigator.hardwareConcurrency;
    cachedWebgl = typeof cores === "number" && cores > 0 && cores < 4 ? "unavailable" : "ok";
  } catch {
    cachedWebgl = "unavailable";
  }
  return cachedWebgl;
}

export function useWebglCapability(): WebglCapability {
  return useClientValue(detectWebgl, "checking");
}

export function useSpeechSynthesisSupported(): boolean {
  return useClientValue(
    () => typeof window !== "undefined" && "speechSynthesis" in window,
    false,
  );
}
