"use client";

import { useEffect, useRef, useState } from "react";
import { answerInterview, startInterview } from "@/lib/api";
import type { InterviewAnswerInput, InterviewState } from "@/lib/types";
import { fill } from "@/lib/i18n";
import { useLanguage } from "@/lib/useLanguage";
import { CuratorAvatar } from "./CuratorAvatar";
import styles from "./chat.module.css";

function CuratorSpeaker({ label }: { label: string }) {
  return (
    <span className={[styles.bubbleSpeaker, styles.curatorSpeaker].join(" ")}>
      <CuratorAvatar size="xs" />
      <span>{label}</span>
    </span>
  );
}

/**
 * The curator interview, rendered as a server-driven chat. The server owns the
 * question order; the client owns visible waiting, focus, and recovery around
 * those transitions.
 */
export function CuratorChat({
  onComplete,
  onSkip,
}: {
  onComplete: (state: InterviewState) => void;
  onSkip: (state: InterviewState) => void;
}) {
  const { language, t } = useLanguage();
  const curatorLine = `${t.brand.curatorName} · ${t.brand.curatorRole}`;
  const [state, setState] = useState<InterviewState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [freeText, setFreeText] = useState("");
  const [multiSelected, setMultiSelected] = useState<string[]>([]);
  const threadRef = useRef<HTMLDivElement>(null);
  const questionRef = useRef<HTMLDivElement>(null);
  const startedRef = useRef(false);
  const question = state?.nextQuestion ?? null;
  const questionKey = question
    ? `${question.id}:${question.step}:${question.prompt}`
    : "";

  useEffect(() => {
    // React 19 strict mode double-invokes effects; one session is enough.
    if (startedRef.current) return;
    startedRef.current = true;
    startInterview(language)
      .then(setState)
      .catch(() => setError(t.chat.startFailed));
    // Language is fixed in the profile created by this call. The header below
    // displays that fact instead of offering a misleading mid-interview toggle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const thread = threadRef.current;
    if (!thread) return;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    thread.scrollTo({ top: thread.scrollHeight, behavior: reduceMotion ? "auto" : "smooth" });
  }, [state]);

  useEffect(() => {
    if (!questionKey) return;
    const frame = window.requestAnimationFrame(() => {
      questionRef.current?.focus({ preventScroll: true });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [questionKey]);

  async function send(answer: InterviewAnswerInput) {
    if (!state || busy) return;
    setBusy(true);
    setError(null);
    try {
      const next = await answerInterview(state.id, answer);
      setState(next);
      setFreeText("");
      setMultiSelected([]);
      if (next.complete) onComplete(next);
    } catch {
      // Provider and validation internals belong in logs, not in visitor copy.
      setError(t.chat.errorGeneric);
    } finally {
      setBusy(false);
    }
  }

  const directQuestionFirst = question?.id === "curiosity" && question.allowFreeText;
  const hasExplicitNoneOption = question?.options.some((option) => option.value === "none") ?? false;
  const freeTextId = question ? `curator-free-text-${question.id}` : "curator-free-text";

  const freeTextControl = question?.allowFreeText ? (
    <form
      className={styles.freeTextBlock}
      onSubmit={(event) => {
        event.preventDefault();
        if (!freeText.trim()) return;
        void send({ questionId: question.id, freeText: freeText.trim() });
      }}
    >
      <label htmlFor={freeTextId}>
        {directQuestionFirst ? t.chat.askDirectly : t.chat.freeTextLabel}
      </label>
      <div className={styles.freeTextRow}>
        <input
          id={freeTextId}
          value={freeText}
          onChange={(event) => setFreeText(event.target.value)}
          placeholder={question.freeTextPlaceholder ?? t.chat.freeTextPlaceholder}
          maxLength={question.id === "open_question" ? 300 : 500}
          disabled={busy}
        />
        <button type="submit" disabled={busy || !freeText.trim()}>
          {t.chat.send}
        </button>
      </div>
    </form>
  ) : null;

  const optionControls = question && question.options.length > 0 ? (
    <>
      <div className={styles.optionGrid} role="group" aria-label={t.chat.answerGroup}>
        {question.options.map((option) => {
          const selected = multiSelected.includes(option.value);
          return (
            <button
              key={option.value}
              type="button"
              className={selected ? styles.optionSelected : styles.option}
              aria-pressed={question.multiSelect ? selected : undefined}
              disabled={busy}
              onClick={() => {
                if (question.multiSelect) {
                  // "Nothing in particular" is a complete, exclusive answer,
                  // not one more topic to add before pressing Confirm.
                  if (option.value === "none") {
                    setMultiSelected([]);
                    void send({ questionId: question.id, value: option.value });
                    return;
                  }
                  setMultiSelected((current) =>
                    current.includes(option.value)
                      ? current.filter((value) => value !== option.value)
                      : [...current.filter((value) => value !== "none"), option.value],
                  );
                  return;
                }
                void send({ questionId: question.id, value: option.value });
              }}
            >
              <strong>{option.label}</strong>
              {option.hint && <small>{option.hint}</small>}
            </button>
          );
        })}
      </div>
      {question.multiSelect && (
        <button
          type="button"
          className={styles.primaryAction}
          disabled={busy}
          onClick={() => void send({ questionId: question.id, value: multiSelected.join(",") })}
        >
          {multiSelected.length > 0
            ? fill(t.chat.confirmCount, { n: multiSelected.length })
            : t.chat.confirmNone}
        </button>
      )}
    </>
  ) : null;

  return (
    <section className={styles.chatShell} aria-label={t.chat.ariaLabel}>
      <header className={styles.chatHeader}>
        <div className={styles.chatIdentity}>
          <CuratorAvatar size="md" />
          <div>
            <p className={styles.curatorIdentity}>{curatorLine}</p>
            <h2>{t.chat.heading}</h2>
          </div>
        </div>
        <div className={styles.chatHeaderMeta}>
          {question && (
            <span
              className={styles.progressPill}
              aria-label={fill(t.chat.stepOf, { step: question.step, total: question.totalSteps })}
            >
              {question.step} / {question.totalSteps}
            </span>
          )}
          <span
            className={styles.languageLocked}
            title={t.chat.languageLockedHint}
            aria-label={`${t.chat.languageLocked}. ${t.chat.languageLockedHint}`}
          >
            {t.chat.languageLocked}
          </span>
        </div>
      </header>

      <div className={styles.thread} ref={threadRef} role="log" aria-live="polite">
        {state?.transcript.map((turn) => (
          <div key={`${turn.questionId}-${turn.answeredAt}`}>
            <div className={styles.curatorBubble}>
              <CuratorSpeaker label={curatorLine} />
              {turn.prompt.split("\n").map((line, index) => (
                <p key={index}>{line}</p>
              ))}
            </div>
            <div className={styles.visitorBubble}>
              <span className={styles.bubbleSpeaker}>{t.chat.you}</span>
              {turn.skipped ? t.chat.skipped : turn.answerLabel || turn.freeText || turn.answerValue}
            </div>
            {turn.curatorReply && (
              <div className={styles.curatorBubble}>
                <CuratorSpeaker label={curatorLine} />
                <p>{turn.curatorReply}</p>
              </div>
            )}
          </div>
        ))}

        {question && (
          <div
            className={[styles.curatorBubble, styles.currentQuestion].join(" ")}
            ref={questionRef}
            tabIndex={-1}
          >
            <CuratorSpeaker label={curatorLine} />
            {question.prompt.split("\n").map((line, index) => (
              <p key={index}>{line}</p>
            ))}
          </div>
        )}

        {!state && !error && (
          <div className={styles.curatorBubble}>
            <CuratorSpeaker label={curatorLine} />
            <p>{t.chat.connecting}</p>
          </div>
        )}
      </div>

      {question && (
        <div className={styles.answerArea} aria-busy={busy}>
          {busy && (
            <p className={styles.thinkingStatus} role="status" aria-live="polite">
              <span aria-hidden="true">◐</span>
              {t.chat.thinking}
            </p>
          )}

          {directQuestionFirst && freeTextControl}
          {directQuestionFirst && optionControls ? (
            <details className={styles.suggestionDetails}>
              <summary>{t.chat.browseSuggestions}</summary>
              <div className={styles.suggestionOptions}>{optionControls}</div>
            </details>
          ) : (
            optionControls
          )}
          {!directQuestionFirst && freeTextControl}

          <div className={styles.secondaryRow}>
            {question.skippable && !(question.multiSelect && hasExplicitNoneOption) && (
              <button
                type="button"
                disabled={busy}
                onClick={() => void send({ questionId: question.id, skipped: true })}
              >
                {t.chat.skipQuestion}
              </button>
            )}
            {question.step < question.totalSteps && state && (
              <button type="button" onClick={() => onSkip(state)} disabled={busy}>
                {state.transcript.length > 0 ? t.chat.continueWithAnswers : t.chat.skipInterview}
              </button>
            )}
          </div>
        </div>
      )}

      {error && (
        <div className={styles.chatError} role="alert">
          <strong>{t.chat.errorTitle}</strong>
          <p>{error}</p>
        </div>
      )}

      <div className={styles.privacyNote}>
        <strong>{t.chat.privacyTitle}</strong>
        <p>{t.chat.privacyBody}</p>
      </div>
    </section>
  );
}
