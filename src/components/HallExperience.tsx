"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import type { Exhibition } from "@/lib/types";
import { logEvent } from "@/lib/api";
import { useWebglCapability } from "@/lib/useClientCapability";
import { ExhibitionView2D } from "./ExhibitionView2D";

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

export function HallExperience({
  exhibition,
  forceTextView = false,
}: {
  exhibition: Exhibition;
  forceTextView?: boolean;
}) {
  const capability = useWebglCapability();
  // Enter the hall only when it will actually work and the visitor has not
  // explicitly asked for the text version.
  const shouldEnter = capability === "ok" && !forceTextView;
  const [inHall, setInHall] = useState(shouldEnter);

  useEffect(() => {
    if (capability === "checking") return;
    logEvent(capability === "ok" ? "hall_entered" : "hall_fallback_2d", exhibition.id, {
      mode: capability === "ok" && !forceTextView ? "3d" : "text",
    });
  }, [capability, exhibition.id, forceTextView]);

  if (inHall && capability === "ok") {
    return (
      <Hall3D
        exhibition={exhibition}
        onExit={() => {
          setInHall(false);
          logEvent("hall_exited", exhibition.id);
        }}
      />
    );
  }

  return (
    <ExhibitionView2D
      exhibition={exhibition}
      webglAvailable={capability === "ok"}
      onEnterHall={() => {
        setInHall(true);
        logEvent("hall_entered", exhibition.id, { mode: "3d" });
      }}
    />
  );
}
