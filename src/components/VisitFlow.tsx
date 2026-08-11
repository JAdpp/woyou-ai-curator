"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { CuratorChat } from "./CuratorChat";
import { CurationPipeline } from "./CurationPipeline";
import { Landing } from "./Landing";
import type { InterviewState } from "@/lib/types";
import { useLanguage } from "@/lib/useLanguage";
import styles from "./chat.module.css";

type Stage = "landing" | "interview" | "curating";

/**
 * Visitor entry flow: landing → curator interview → visible curation → hall.
 *
 * The exhibition itself lives on its own route, so a visitor can reload or
 * share the hall without replaying the interview.
 */
export function VisitFlow() {
  const { language, t } = useLanguage();
  const router = useRouter();
  const [stage, setStage] = useState<Stage>("landing");
  const [interviewId, setInterviewId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleComplete = useCallback((state: InterviewState) => {
    setInterviewId(state.id);
    setStage("curating");
  }, []);

  const handleCurated = useCallback(
    (exhibitionId: string) => {
      router.push(`/exhibitions/${exhibitionId}`);
    },
    [router],
  );

  const handleError = useCallback((message: string) => {
    setError(message);
    setStage("interview");
  }, []);

  if (stage === "landing") {
    return <Landing onStart={() => setStage("interview")} />;
  }

  if (stage === "curating" && interviewId) {
    return (
      <CurationPipeline
        interviewId={interviewId}
        onComplete={handleCurated}
        onError={handleError}
      />
    );
  }

  return (
    <>
      {error && (
        <div className={styles.chatError} role="alert" style={{ margin: "1rem auto", maxWidth: "44rem" }}>
          <p>{error}</p>
        </div>
      )}
      <CuratorChat
        onComplete={handleComplete}
        onSkip={() => {
          // "Skip the interview" still needs a session, so start one and run
          // straight through with server defaults.
          setError(null);
          void (async () => {
            const { startInterview } = await import("@/lib/api");
            try {
              const state = await startInterview(language);
              setInterviewId(state.id);
              setStage("curating");
            } catch {
              setError(t.pipeline.defaultFailed);
            }
          })();
        }}
      />
    </>
  );
}
