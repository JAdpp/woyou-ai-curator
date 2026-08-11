"use client";

import {
  type FormEvent,
  type KeyboardEvent,
  useEffect,
  useId,
  useReducer,
  useRef,
  useState,
} from "react";
import { logEvent, sendEpilogueChatMessage } from "@/lib/api";
import { fill } from "@/lib/i18n";
import { useLanguage } from "@/lib/useLanguage";
import type { EpilogueChatRequest } from "@/lib/types";
import {
  EMPTY_EPILOGUE_CONVERSATION,
  epilogueConversationReducer,
  toEpilogueChatHistory,
  trimEpilogueDraft,
} from "./epilogueConversationState";
import { CuratorAvatar } from "./CuratorAvatar";
import styles from "./EpilogueConversation.module.css";

type ConversationContext = "page" | "hall";

interface EpilogueConversationProps {
  exhibitionId: string;
  openQuestions: string[];
  context?: ConversationContext;
}

interface PendingAttempt extends EpilogueChatRequest {
  id: string;
}

export function EpilogueConversation({
  exhibitionId,
  openQuestions,
  context = "page",
}: EpilogueConversationProps) {
  const { t: copy } = useLanguage();
  const t = copy.epilogue;
  const [open, setOpen] = useState(false);
  const [skipped, setSkipped] = useState(false);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [validationMessage, setValidationMessage] = useState<string | null>(null);
  const [lastAttempt, setLastAttempt] = useState<PendingAttempt | null>(null);
  const [conversation, dispatch] = useReducer(
    epilogueConversationReducer,
    EMPTY_EPILOGUE_CONVERSATION,
  );
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const idCounterRef = useRef(0);
  const openedLoggedRef = useRef(false);
  const formHintId = useId();
  const validationId = useId();

  const nextId = (prefix: string) => {
    idCounterRef.current += 1;
    return `${prefix}-${idCounterRef.current}`;
  };

  useEffect(() => () => controllerRef.current?.abort(), []);

  useEffect(() => {
    if (!open) return;
    inputRef.current?.focus({ preventScroll: true });
  }, [open, conversation.selectedOpenQuestion]);

  useEffect(() => {
    if (!open || conversation.turns.length === 0) return;
    endRef.current?.scrollIntoView({ block: "nearest" });
  }, [conversation.turns.length, open, pending]);

  const openConversation = () => {
    setOpen(true);
    setSkipped(false);
    if (!openedLoggedRef.current) {
      openedLoggedRef.current = true;
      void logEvent("epilogue_chat_opened", exhibitionId, {
        source: "epilogue",
        turnCount: conversation.turns.length,
      });
    }
  };

  const closeConversation = (reason: "collapse" | "skip") => {
    if (reason === "skip") {
      controllerRef.current?.abort();
      setPending(false);
      setSkipped(true);
    }
    setOpen(false);
    void logEvent("epilogue_chat_closed", exhibitionId, {
      source: "epilogue",
      reason,
      turnCount: conversation.turns.length,
    });
  };

  const clearConversation = () => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setPending(false);
    setDraft("");
    setError(null);
    setValidationMessage(null);
    setLastAttempt(null);
    dispatch({ type: "clear" });
    inputRef.current?.focus();
  };

  const performAttempt = async (attempt: PendingAttempt) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setPending(true);
    setError(null);
    setValidationMessage(null);
    setDraft("");
    dispatch({ type: "set_suggestions", replies: [] });

    try {
      const response = await sendEpilogueChatMessage(
        exhibitionId,
        {
          message: attempt.message,
          history: attempt.history,
          openQuestion: attempt.openQuestion,
        },
        controller.signal,
      );
      dispatch({
        type: "append_turn",
        id: attempt.id,
        role: "user",
        content: attempt.message,
      });
      dispatch({
        type: "append_turn",
        id: nextId("assistant"),
        role: "assistant",
        content: response.message,
        citations: response.citations,
        notice: response.notice || (response.mode === "local_fallback"
          ? t.localFallback
          : undefined),
      });
      dispatch({ type: "set_suggestions", replies: response.suggestedReplies });
      setLastAttempt(null);
      void logEvent("epilogue_chat_message_sent", exhibitionId, {
        source: "epilogue",
        turnCount: attempt.history.length + 2,
      });
    } catch (caught: unknown) {
      if (caught instanceof DOMException && caught.name === "AbortError") {
        if (controllerRef.current === controller) setDraft(attempt.message);
        return;
      }
      setDraft(attempt.message);
      setError(t.sendFailed);
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
        setPending(false);
      }
    }
  };

  const submitDraft = (event?: FormEvent<HTMLFormElement>) => {
    event?.preventDefault();
    if (pending) return;
    const message = trimEpilogueDraft(draft);
    if (!message) {
      setValidationMessage(t.needText);
      inputRef.current?.focus();
      return;
    }
    const attempt: PendingAttempt = {
      id: nextId("visitor"),
      message,
      history: toEpilogueChatHistory(conversation.turns),
      ...(conversation.selectedOpenQuestion
        ? { openQuestion: conversation.selectedOpenQuestion }
        : {}),
    };
    setLastAttempt(attempt);
    void performAttempt(attempt);
  };

  const handleInputKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    event.stopPropagation();
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };

  const chooseOpeningQuestion = (question: string) => {
    dispatch({ type: "select_open_question", question, id: nextId("opening") });
    setError(null);
    setValidationMessage(null);
  };

  const hasVisitorTurn = conversation.turns.some((turn) => turn.role === "user");

  if (!open) {
    return (
      <section className={styles.root} data-context={context} data-state="compact">
        <span className={styles.folioMark} aria-hidden="true">{t.folioMark}</span>
        <div className={styles.compactIdentity}>
          <CuratorAvatar size="sm" />
          <div className={styles.compactCopy}>
            <p className={styles.kicker}>{t.kicker}</p>
            <h3>{skipped ? t.skippedTitle : t.openTitle}</h3>
            <p>
              {skipped
                ? t.skippedBody
                : fill(t.openBody, { n: Math.max(openQuestions.length, 1) })}
            </p>
          </div>
        </div>
        <div className={styles.compactActions}>
          <button type="button" className={styles.primaryButton} onClick={openConversation}>
            {skipped ? t.reopen : t.start}
          </button>
          {!skipped && (
            <button type="button" className={styles.textButton} onClick={() => closeConversation("skip")}>
              {t.skipNow}
            </button>
          )}
        </div>
      </section>
    );
  }

  return (
    <section className={styles.root} data-context={context} data-state="open" aria-label={t.sectionLabel}>
      <span className={styles.folioMark} aria-hidden="true">{t.folioMark}</span>
      <header className={styles.header}>
        <div className={styles.headerIdentity}>
          <CuratorAvatar size="sm" />
          <div>
            <p className={styles.kicker}>{t.kicker}</p>
            <h3>{t.liveTitle}</h3>
          </div>
        </div>
        <div className={styles.headerActions}>
          {conversation.turns.length > 0 && (
            <button type="button" className={styles.textButton} onClick={clearConversation} disabled={pending}>
              {t.clear}
            </button>
          )}
          <button type="button" className={styles.textButton} onClick={() => closeConversation("collapse")}>
            {t.collapse}
          </button>
        </div>
      </header>

      <p className={styles.intro}>
        {t.noScoring}
      </p>

      {conversation.turns.length === 0 && (
        <p className={styles.emptyNote}>{t.emptyNote}</p>
      )}

      <div className={styles.transcript} role="log" aria-live="polite" aria-relevant="additions text">
        {conversation.turns.map((turn) => (
          <article className={styles.turn} data-role={turn.role} key={turn.id}>
            {turn.role === "assistant" ? (
              <div className={styles.speaker}>
                <CuratorAvatar size="xs" />
                <span>{t.curator}</span>
              </div>
            ) : (
              <p className={styles.speaker}>{t.visitor}</p>
            )}
            <div className={styles.turnBody}>
              <p>{turn.content}</p>
              {turn.notice && <p className={styles.serviceNotice}>{turn.notice}</p>}
              {turn.citations && turn.citations.length > 0 && (
                <details className={styles.citations}>
                  <summary>{t.citationsSummary}</summary>
                  <ul>
                    {turn.citations.map((citation, index) => (
                      <li key={`${citation.itemId || citation.objectId || "evidence"}-${index}`}>
                        {citation.label || citation.itemId || citation.objectId || fill(t.citationFallback, { n: citation.evidenceIds?.length ?? 0 })}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          </article>
        ))}
        {pending && lastAttempt && (
          <article className={styles.turn} data-role="user">
            <p className={styles.speaker}>{t.visitor}</p>
            <div className={styles.turnBody}><p>{lastAttempt.message}</p></div>
          </article>
        )}
        {pending && (
          <div className={styles.pendingTurn} role="status">
            <span className={styles.pendingSpeaker}>
              <CuratorAvatar size="xs" />
              <span>{t.curator}</span>
            </span>
            <p>{t.thinking}</p>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {!hasVisitorTurn && openQuestions.length > 0 && (
        <fieldset className={styles.starters}>
          <legend>{conversation.selectedOpenQuestion ? t.changeNote : t.pickNote}</legend>
          {openQuestions.map((question) => (
            <button
              type="button"
              key={question}
              aria-pressed={conversation.selectedOpenQuestion === question}
              onClick={() => chooseOpeningQuestion(question)}
            >
              {question}
            </button>
          ))}
        </fieldset>
      )}

      {conversation.suggestedReplies.length > 0 && !pending && (
        <div className={styles.suggestions} aria-label={t.suggestionsLabel}>
          <span>{t.suggestionsLead}</span>
          {conversation.suggestedReplies.map((reply) => (
            <button
              type="button"
              key={reply}
              onClick={() => {
                setDraft(reply);
                setValidationMessage(null);
                inputRef.current?.focus();
              }}
            >
              {reply}
            </button>
          ))}
        </div>
      )}

      {error && (
        <div className={styles.error} role="alert">
          <span>{error}</span>
          {lastAttempt && (
            <button type="button" onClick={() => void performAttempt(lastAttempt)} disabled={pending}>
              {t.retry}
            </button>
          )}
        </div>
      )}

      <form className={styles.form} onSubmit={submitDraft} onKeyDown={(event) => event.stopPropagation()}>
        <label htmlFor={`${formHintId}-input`}>{t.inputLabel}</label>
        <textarea
          ref={inputRef}
          id={`${formHintId}-input`}
          value={draft}
          rows={context === "hall" ? 2 : 3}
          maxLength={800}
          placeholder={conversation.selectedOpenQuestion ? t.placeholderWithNote : t.placeholderOpen}
          aria-describedby={`${formHintId} ${validationMessage ? validationId : ""}`.trim()}
          aria-invalid={validationMessage ? true : undefined}
          disabled={pending}
          onChange={(event) => {
            setDraft(event.target.value);
            if (validationMessage && event.target.value.trim()) setValidationMessage(null);
          }}
          onKeyDown={handleInputKeyDown}
        />
        <div className={styles.formFooter}>
          <p id={formHintId}>
            {t.privacy}
          </p>
          <button
            type="submit"
            className={styles.primaryButton}
            disabled={pending || trimEpilogueDraft(draft).length === 0}
          >
            {pending ? t.sending : t.send}
          </button>
        </div>
        {validationMessage && <p className={styles.validation} id={validationId}>{validationMessage}</p>}
      </form>

      <button type="button" className={styles.skipButton} onClick={() => closeConversation("skip")}>
        {t.optOut}
      </button>
    </section>
  );
}
