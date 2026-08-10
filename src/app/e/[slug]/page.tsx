import Link from "next/link";
import { ExhibitionView2D } from "@/components/ExhibitionView2D";
import { getPublishedExhibition } from "@/lib/api";
import type { Exhibition } from "@/lib/types";

/**
 * Read-only share link.
 *
 * Always renders the text version: a shared link is opened cold, often on a
 * phone, often from a chat app's in-app browser, and should show the exhibition
 * immediately rather than negotiating a WebGL context.
 */
export default async function SharedExhibitionPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  let exhibition: Exhibition | null = null;

  try {
    exhibition = await getPublishedExhibition(slug);
  } catch {
    // Render the unavailable state below, outside the error boundary.
  }

  if (exhibition) return <ExhibitionView2D exhibition={exhibition} />;

  return (
    <main style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 24 }}>
      <section style={{ maxWidth: 560, padding: 36, background: "#fbfcfa", border: "1px solid #cbd5d1" }}>
        <p style={{ fontFamily: "monospace", color: "#176b70", fontSize: 11, marginBottom: 12 }}>
          SHARED EXHIBITION
        </p>
        <h1 style={{ fontFamily: "serif", fontSize: 34, marginBottom: 14 }}>这个展览链接打不开</h1>
        <p style={{ color: "#42545a", lineHeight: 1.8, marginBottom: 24 }}>
          展览可能已经撤下，或本地 API 尚未启动。
        </p>
        <Link href="/" style={{ color: "#176b70", borderBottom: "1px solid currentColor" }}>
          去生成一座属于你的展厅
        </Link>
      </section>
    </main>
  );
}
