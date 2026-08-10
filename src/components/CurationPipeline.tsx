"use client";

import { useEffect, useRef, useState } from "react";
import { followJob, startCuration } from "@/lib/api";
import type { GenerationJob } from "@/lib/types";
import { CURATOR_TITLE } from "@/lib/brand";
import { CollectionPuzzle } from "./CollectionPuzzle";
import styles from "./chat.module.css";

const STATUS_GLYPH: Record<string, string> = {
  pending: "○",
  running: "◐",
  done: "✓",
  failed: "!",
  skipped: "—",
};

/**
 * The visible curation todo list.
 *
 * Every step reports a concrete finding when it lands ("selected 12 of 43
 * candidates", "3 narrative segments: …"), which is the point of showing the pipeline at
 * all — a spinner with fake percentages tells the visitor nothing.
 */
export function CurationPipeline({
  interviewId,
  onComplete,
  onError,
}: {
  interviewId: string;
  onComplete: (exhibitionId: string) => void;
  onError: (message: string) => void;
}) {
  const [job, setJob] = useState<GenerationJob | null>(null);
  // React's development double-invoke must not start two jobs — that would be
  // two model calls per visitor. The request is cached by interview id and the
  // second run re-subscribes to the same promise, rather than being skipped:
  // skipping it would strand the first run's result behind its own cleanup
  // flag and leave the pipeline showing "正在启动…" forever.
  const pending = useRef<{ interviewId: string; job: Promise<GenerationJob> } | null>(null);

  useEffect(() => {
    let stopFollowing: (() => void) | null = null;
    let cancelled = false;

    if (pending.current?.interviewId !== interviewId) {
      pending.current = { interviewId, job: startCuration(interviewId) };
    }

    pending.current.job
      .then((created) => {
        if (cancelled) return;
        setJob(created);
        stopFollowing = followJob(created.id, (update) => {
          setJob(update);
          if (update.status === "completed" && update.exhibitionId) {
            onComplete(update.exhibitionId);
          } else if (update.status === "failed") {
            onError(update.error || "策展没有完成，请重试。");
          }
        });
      })
      .catch((startError) => {
        if (cancelled) return;
        onError(startError instanceof Error ? startError.message : "策展没有启动，请重试。");
      });

    return () => {
      cancelled = true;
      stopFollowing?.();
    };
  }, [interviewId, onComplete, onError]);

  const steps = job?.steps ?? [];

  return (
    <section className={styles.pipelineShell} aria-label="策展生成进度">
      <div className={styles.pipelineLayout}>
        <div className={styles.pipelineProcess} aria-live="polite" aria-busy={job?.status !== "completed"}>
          <header>
            <h2>{CURATOR_TITLE}正在为你的问题策展</h2>
            <p className={styles.pipelineStage}>{job?.stage || "正在启动…"}</p>
          </header>

          <ol className={styles.stepList}>
            {steps.map((step, index) => (
              <li key={step.key} className={styles[`step_${step.status}`] ?? undefined}>
                <span className={styles.stepGlyph} aria-hidden="true">
                  {step.status === "pending" ? String(index + 1).padStart(2, "0") : STATUS_GLYPH[step.status]}
                </span>
                <div>
                  <strong>{step.title}</strong>
                  <small>{step.finding || step.detail}</small>
                </div>
              </li>
            ))}
          </ol>

          <div className={styles.progressTrack} role="progressbar" aria-valuenow={job?.progress ?? 0} aria-valuemin={0} aria-valuemax={100}>
            <div className={styles.progressFill} style={{ width: `${job?.progress ?? 0}%` }} />
          </div>
        </div>

        <CollectionPuzzle />
      </div>
    </section>
  );
}
