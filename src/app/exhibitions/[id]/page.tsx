import Link from "next/link";
import { HallExperience } from "@/components/HallExperience";
import { getExhibition } from "@/lib/api";
import type { Exhibition } from "@/lib/types";

export default async function ExhibitionPage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ view?: string }>;
}) {
  const { id } = await params;
  const { view } = await searchParams;
  let exhibition: Exhibition | null = null;

  try {
    exhibition = await getExhibition(id);
  } catch {
    // Fall through to the unavailable state below.
  }

  if (exhibition) {
    // `?view=text` is the permanent accessible entry point linked from inside
    // the hall, so it must never be overridden by the WebGL probe.
    return <HallExperience exhibition={exhibition} forceTextView={view === "text"} />;
  }

  return (
    <main style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 24 }}>
      <section style={{ maxWidth: 560, padding: 36, background: "#fbfcfa", border: "1px solid #cbd5d1" }}>
        <p style={{ fontFamily: "monospace", color: "#b44532", fontSize: 11, marginBottom: 12 }}>
          EXHIBITION UNAVAILABLE
        </p>
        <h1 style={{ fontFamily: "serif", fontSize: 34, marginBottom: 14 }}>找不到这场展览</h1>
        <p style={{ color: "#42545a", lineHeight: 1.8, marginBottom: 24 }}>
          本地 API 可能尚未启动，或展览记录已经清除。生成的展览保存在后端运行状态中，而不是浏览器临时内存。
        </p>
        <Link href="/" style={{ color: "#176b70", borderBottom: "1px solid currentColor" }}>
          返回并重新策展
        </Link>
      </section>
    </main>
  );
}
