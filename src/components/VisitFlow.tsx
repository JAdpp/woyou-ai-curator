"use client";

import { useCallback, useReducer } from "react";
import { useRouter } from "next/navigation";
import { CuratorChat } from "./CuratorChat";
import { CurationPipeline } from "./CurationPipeline";
import { Landing } from "./Landing";
import { CuratorAvatar } from "./CuratorAvatar";
import type { GenerationJob, InterviewState, PriorKnowledge } from "@/lib/types";
import { fill } from "@/lib/i18n";
import { useLanguage } from "@/lib/useLanguage";
import {
  INITIAL_VISIT_FLOW_STATE,
  visitFlowReducer,
} from "@/lib/visitFlowState";
import styles from "./chat.module.css";

function visitInquiry(interview: InterviewState): string {
  const profile = interview.profile;
  return (
    profile.freeFormQuestion?.trim() ||
    profile.openQuestion?.trim() ||
    profile.curiosityLabel.trim()
  );
}

function VisitBrief({
  interview,
  onConfirm,
  onRevise,
}: {
  interview: InterviewState;
  onConfirm: () => void;
  onRevise: () => void;
}) {
  const { language, t } = useLanguage();
  const profile = interview.profile;
  const knowledge: Record<PriorKnowledge, string> = {
    none: t.visitBrief.knowledgeNone,
    some: t.visitBrief.knowledgeSome,
    familiar: t.visitBrief.knowledgeFamiliar,
  };
  const inquiry = visitInquiry(interview);
  const showDirection = Boolean(
    profile.curiosityLabel.trim() && profile.curiosityLabel.trim() !== inquiry,
  );

  return (
    <section className={styles.briefShell} aria-label={t.visitBrief.ariaLabel}>
      <div className={styles.briefIdentity}>
        <CuratorAvatar size="lg" />
        <div>
          <p className={styles.briefEyebrow}>{t.visitBrief.eyebrow}</p>
          <h2>{t.visitBrief.heading}</h2>
        </div>
      </div>
      <p className={styles.briefIntro}>{t.visitBrief.intro}</p>
      <dl className={styles.briefDetails}>
        <div className={styles.briefInquiry}>
          <dt>{t.visitBrief.inquiry}</dt>
          <dd>{inquiry}</dd>
        </div>
        {showDirection && (
          <div>
            <dt>{t.visitBrief.direction}</dt>
            <dd>{profile.curiosityLabel}</dd>
          </div>
        )}
        <div>
          <dt>{t.visitBrief.priorKnowledge}</dt>
          <dd>{knowledge[profile.priorKnowledge]}</dd>
        </div>
        <div>
          <dt>{t.visitBrief.duration}</dt>
          <dd>{fill(t.visitBrief.minutes, { n: profile.durationMinutes })}</dd>
        </div>
        <div>
          <dt>{t.visitBrief.exclusions}</dt>
          <dd>
            {profile.excludedTopics.length > 0
              ? profile.excludedTopics.join(language === "en" ? ", " : "、")
              : t.visitBrief.noExclusions}
          </dd>
        </div>
      </dl>
      <div className={styles.briefActions}>
        <button type="button" className={styles.primaryAction} onClick={onConfirm}>
          {t.visitBrief.confirm}
        </button>
        <button type="button" className={styles.textAction} onClick={onRevise}>
          {t.visitBrief.revise}
        </button>
      </div>
    </section>
  );
}

function CurationRecovery({
  interview,
  failedJob,
  onRetry,
  onRevise,
  onHome,
}: {
  interview: InterviewState;
  failedJob: GenerationJob | null;
  onRetry: () => void;
  onRevise: () => void;
  onHome: () => void;
}) {
  const { t } = useLanguage();
  const inquiry = visitInquiry(interview);
  const finishedSteps = failedJob?.steps.filter(
    (step) => step.status === "done" || step.status === "skipped",
  ).length ?? 0;

  return (
    <section className={styles.recoveryShell} aria-label={t.recovery.ariaLabel}>
      <div className={styles.recoveryIdentity}>
        <CuratorAvatar size="lg" />
        <div>
          <p className={styles.briefEyebrow}>{t.recovery.eyebrow}</p>
          <h2>{t.recovery.heading}</h2>
        </div>
      </div>
      <p className={styles.recoveryBody}>{t.recovery.body}</p>
      <p className={styles.retainedInquiry}>
        <span aria-hidden="true">✓</span>
        <span>
          <strong>{t.recovery.retained}</strong>
          {inquiry}
        </span>
      </p>
      {failedJob && failedJob.steps.length > 0 && (
        <p className={styles.recoveryProgress}>
          {fill(t.recovery.progressRetained, {
            done: finishedSteps,
            total: failedJob.steps.length,
          })}
        </p>
      )}
      <div className={styles.recoveryActions}>
        <button type="button" className={styles.primaryAction} onClick={onRetry} autoFocus>
          {t.recovery.retry}
        </button>
        <button type="button" className={styles.secondaryAction} onClick={onRevise}>
          {t.recovery.revise}
        </button>
        <button type="button" className={styles.textAction} onClick={onHome}>
          {t.recovery.home}
        </button>
      </div>
    </section>
  );
}

/** Visitor entry flow: landing → interview → brief → curation → hall. */
export function VisitFlow() {
  const router = useRouter();
  const [flow, dispatch] = useReducer(visitFlowReducer, INITIAL_VISIT_FLOW_STATE);

  const handleCurated = useCallback(
    (exhibitionId: string) => {
      router.push(`/exhibitions/${exhibitionId}`);
    },
    [router],
  );

  const handleFailure = useCallback((job: GenerationJob | null) => {
    dispatch({ type: "curation_failed", job });
  }, []);

  if (flow.stage === "landing") {
    return <Landing onStart={() => dispatch({ type: "start_interview" })} />;
  }

  if (flow.stage === "brief" && flow.interview) {
    return (
      <VisitBrief
        interview={flow.interview}
        onConfirm={() => dispatch({ type: "confirm_brief" })}
        onRevise={() => dispatch({ type: "adjust_answers" })}
      />
    );
  }

  if (flow.stage === "curating" && flow.interview) {
    return (
      <CurationPipeline
        key={`${flow.interview.id}:${flow.curationAttempt}`}
        interviewId={flow.interview.id}
        onComplete={handleCurated}
        onError={handleFailure}
      />
    );
  }

  if (flow.stage === "failed" && flow.interview) {
    return (
      <CurationRecovery
        interview={flow.interview}
        failedJob={flow.failedJob}
        onRetry={() => dispatch({ type: "retry_curation" })}
        onRevise={() => dispatch({ type: "adjust_answers" })}
        onHome={() => dispatch({ type: "return_home" })}
      />
    );
  }

  return (
    <CuratorChat
      onComplete={(interview) => dispatch({ type: "complete_interview", interview })}
      onSkip={(interview) => dispatch({ type: "skip_to_curation", interview })}
    />
  );
}
