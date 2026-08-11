"use client";

import { useEffect, useRef, useState } from "react";
import { followJob, startCuration } from "@/lib/api";
import type { GenerationJob } from "@/lib/types";
import { useLanguage } from "@/lib/useLanguage";
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
  const { t } = useLanguage();
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
            onError(update.error || t.pipeline.failed);
          }
        });
      })
      .catch((startError) => {
        if (cancelled) return;
        onError(startError instanceof Error ? startError.message : t.pipeline.notStarted);
      });

    return () => {
      cancelled = true;
      stopFollowing?.();
    };
  }, [interviewId, onComplete, onError]);

  const steps = job?.steps ?? [];

  return (
    <section className={styles.pipelineShell} aria-label={t.pipeline.ariaLabel}>
      <div className={styles.pipelineLayout}>
        <div className={styles.pipelineProcess} aria-live="polite" aria-busy={job?.status !== "completed"}>
          <header>
            <h2>{t.pipeline.heading}</h2>
            {/* The stage text comes from the job and is written by the server in
                the language the interview ran in, so it is not translated here. */}
            <p className={styles.pipelineStage}>{job?.stage || t.pipeline.starting}</p>
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
