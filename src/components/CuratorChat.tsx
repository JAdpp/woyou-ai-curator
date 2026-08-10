"use client";

import { useEffect, useRef, useState } from "react";
import { answerInterview, startInterview } from "@/lib/api";
import type { InterviewAnswerInput, InterviewState } from "@/lib/types";
import { CURATOR_NAME, CURATOR_TITLE } from "@/lib/brand";
import styles from "./chat.module.css";

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
    startInterview()
      .then(setState)
      .catch((startError) =>
        setError(startError instanceof Error ? startError.message : `${CURATOR_NAME}没能接上话，请刷新重试。`),
      );
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
      setError(answerError instanceof Error ? answerError.message : "这一步没有记录下来，请重试。");
    } finally {
      setBusy(false);
    }
  }

  const question = state?.nextQuestion ?? null;

  return (
    <section className={styles.chatShell} aria-label={`与${CURATOR_TITLE}对话`}>
      <header className={styles.chatHeader}>
        <div>
          <p className={styles.curatorIdentity}>{CURATOR_NAME} · AI 策展人</p>
          <h2>先说说你想了解什么</h2>
        </div>
        {question && (
          <span className={styles.progressPill} aria-label={`第 ${question.step} 步，共 ${question.totalSteps} 步`}>
            {question.step} / {question.totalSteps}
          </span>
        )}
      </header>

      <div className={styles.thread} ref={threadRef} role="log" aria-live="polite">
        {state?.transcript.map((turn) => (
          <div key={`${turn.questionId}-${turn.answeredAt}`}>
            <div className={styles.curatorBubble}>
              <span className={styles.bubbleSpeaker}>{CURATOR_NAME} · AI 策展人</span>
              {turn.prompt.split("\n").map((line, index) => (
                <p key={index}>{line}</p>
              ))}
            </div>
            <div className={styles.visitorBubble}>
              <span className={styles.bubbleSpeaker}>你</span>
              {turn.skipped ? "（跳过）" : turn.answerLabel || turn.freeText || turn.answerValue}
            </div>
          </div>
        ))}

        {question && (
          <div className={styles.curatorBubble}>
            <span className={styles.bubbleSpeaker}>{CURATOR_NAME} · AI 策展人</span>
            {question.prompt.split("\n").map((line, index) => (
              <p key={index}>{line}</p>
            ))}
          </div>
        )}

        {!state && !error && (
          <div className={styles.curatorBubble}>
            <span className={styles.bubbleSpeaker}>{CURATOR_NAME} · AI 策展人</span>
            <p>正在接通…</p>
          </div>
        )}
      </div>

      {question && (
        <div className={styles.answerArea}>
          <div className={styles.optionGrid} role="group" aria-label="可选回答">
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

          {question.multiSelect && (
            <button
              type="button"
              className={styles.primaryAction}
              disabled={busy}
              onClick={() => void send({ questionId: question.id, value: multiSelected.join(",") })}
            >
              {multiSelected.length > 0 ? `确认 ${multiSelected.length} 项` : "都可以"}
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
                placeholder={question.freeTextPlaceholder ?? "直接说说你想弄懂什么…"}
                maxLength={200}
                disabled={busy}
                aria-label="自由输入"
              />
              <button type="submit" disabled={busy || !freeText.trim()}>
                发送
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
                跳过这题
              </button>
            )}
            <button type="button" onClick={onSkip} disabled={busy}>
              跳过访谈，用默认设置
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className={styles.chatError} role="alert">
          <strong>这一步没有记录下来</strong>
          <p>{error}</p>
        </div>
      )}

      <div className={styles.privacyNote}>
        <strong>关于这次对话</strong>
        <p>
          访谈内容只用于这次策展。启用云模型时，你输入的文字会发送给已配置的模型服务；请不要填写个人敏感信息。
        </p>
      </div>
    </section>
  );
}
