import { notFound } from "next/navigation";
import { QrelReviewWorkbench } from "@/components/qrel-review/QrelReviewWorkbench";

/** Human qrel annotation must never be exposed on the visitor deployment. */
export default function DevQrelReviewPage() {
  if (process.env.NODE_ENV === "production") notFound();
  return <QrelReviewWorkbench />;
}
