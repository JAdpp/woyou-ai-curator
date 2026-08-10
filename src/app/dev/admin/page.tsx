import { notFound } from "next/navigation";
import { AdminDashboard } from "@/components/AdminDashboard";

/**
 * Internal data/review tooling.
 *
 * Moved off the visitor path (01b §2.3): the demo is for the public, and a
 * review queue is not part of what they came for. Kept because it is still the
 * fastest way to inspect generated exhibitions and the collection audit while
 * developing, but it never renders in a production build.
 */
export default function DevAdminPage() {
  if (process.env.NODE_ENV === "production") notFound();
  return <AdminDashboard />;
}
