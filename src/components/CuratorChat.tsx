"use client";

import { useEffect, useRef, useState } from "react";
import { answerInterview, startInterview } from "@/lib/api";
import type { InterviewAnswerInput, InterviewState } from "@/lib/types";
import { fill } from "@/lib/i18n";
import { useLanguage } from "@/lib/useLanguage";
import { CuratorAvatar } from "./CuratorAvatar";
import { LanguageToggle } from "./LanguageToggle";
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
 * The curator interview, rendered as a chat.
 *
 * The conversation is driven entirely by server state: the backend decides
 * which question comes next and which options exist, so the visitor can never
 * be offered a theme the collection cannot route. This component only renders
 * turns and posts answers.
 */
export function CuratorChat({
  onComplete,
  onSkip,
}: {
  onComplete: (state: InterviewState) => void;
  onSkip: () => void;
}) {
  const { language, setLanguage, t } = useLanguage();
  const curatorLine = `${t.brand.curatorName} · ${t.brand.curatorRole}`;
  const [state, setState] = useState<InterviewState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [freeText, setFreeText] = useState("");
  const [multiSelected, setMultiSelected] = useState<string[]>([]);
  const threadRef = useRef<HTMLDivElement>(null);
  const startedRef = useRef(false);

  useEffect(() => {
    // React 19 strict mode double-invokes effects; one session is enough.
    if (startedRef.current) return;
    startedRef.current = true;
    startInterview(language)
      .then(setState)
      .catch((startError) =>
        setError(startError instanceof Error ? startError.message : t.chat.startFailed),
      );
    // Started once, in whatever language was active then. The profile carries
    // that language onward, so switching later changes the interface but not
    // the interview already under way.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const thread = threadRef.current;
    if (!thread) return;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    thread.scrollTo({ top: thread.scrollHeight, behavior: reduceMotion ? "auto" : "smooth" });
  }, [state]);

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
    } catch (answerError) {
      setError(answerError instanceof Error ? answerError.message : t.chat.errorGeneric);
    } finally {
      setBusy(false);
    }
  }

  const question = state?.nextQuestion ?? null;

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
          <LanguageToggle
            language={language}
            onChange={setLanguage}
            label={t.header.languageGroup}
          />
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
          <div className={styles.curatorBubble}>
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
        <div className={styles.answerArea}>
          {question.options.length > 0 && (
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
                        setMultiSelected((current) =>
                          current.includes(option.value)
                            ? current.filter((value) => value !== option.value)
                            : [...current, option.value],
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
          )}

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

          {question.allowFreeText && (
            <form
              className={styles.freeTextRow}
              onSubmit={(event) => {
                event.preventDefault();
                if (!freeText.trim()) return;
                void send({ questionId: question.id, freeText: freeText.trim() });
              }}
            >
              <input
                value={freeText}
                onChange={(event) => setFreeText(event.target.value)}
                placeholder={question.freeTextPlaceholder ?? t.chat.freeTextPlaceholder}
                maxLength={200}
                disabled={busy}
                aria-label={t.chat.freeTextLabel}
              />
              <button type="submit" disabled={busy || !freeText.trim()}>
                {t.chat.send}
              </button>
            </form>
          )}

          <div className={styles.secondaryRow}>
            {question.skippable && (
              <button
                type="button"
                disabled={busy}
                onClick={() => void send({ questionId: question.id, skipped: true })}
              >
                {t.chat.skipQuestion}
              </button>
            )}
            <button type="button" onClick={onSkip} disabled={busy}>
              {t.chat.skipInterview}
            </button>
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
