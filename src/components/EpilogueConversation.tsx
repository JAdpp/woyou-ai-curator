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
import type { EpilogueChatRequest } from "@/lib/types";
import {
  EMPTY_EPILOGUE_CONVERSATION,
  epilogueConversationReducer,
  toEpilogueChatHistory,
  trimEpilogueDraft,
} from "./epilogueConversationState";
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
          ? "当前回应由本展馆藏材料的本地回退生成，未调用实时模型。"
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
      setError("彦远暂时没能接上这句话。你可以重试，或稍后再继续。");
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
      setValidationMessage("写下一点想法后再发送。");
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
        <span className={styles.folioMark} aria-hidden="true">题笺</span>
        <div className={styles.compactCopy}>
          <p className={styles.kicker}>可选讨论 · AI 策展人彦远</p>
          <h3>{skipped ? "题笺替你留在这里" : "如果你还想把一个念头说完"}</h3>
          <p>
            {skipped
              ? "不用现在回答；想回来时，彦远仍会从这场展览接着聊。"
              : `彦远留下了 ${Math.max(openQuestions.length, 1)} 则展后题笺，也听你自由发问。`}
          </p>
        </div>
        <div className={styles.compactActions}>
          <button type="button" className={styles.primaryButton} onClick={openConversation}>
            {skipped ? "重新打开题笺" : "与彦远聊一会儿"}
          </button>
          {!skipped && (
            <button type="button" className={styles.textButton} onClick={() => closeConversation("skip")}>
              先跳过
            </button>
          )}
        </div>
      </section>
    );
  }

  return (
    <section className={styles.root} data-context={context} data-state="open" aria-label="展后题笺">
      <span className={styles.folioMark} aria-hidden="true">题笺</span>
      <header className={styles.header}>
        <div>
          <p className={styles.kicker}>可选讨论 · AI 策展人彦远</p>
          <h3>把展览带出展厅</h3>
        </div>
        <div className={styles.headerActions}>
          {conversation.turns.length > 0 && (
            <button type="button" className={styles.textButton} onClick={clearConversation} disabled={pending}>
              清空
            </button>
          )}
          <button type="button" className={styles.textButton} onClick={() => closeConversation("collapse")}>
            收起
          </button>
        </div>
      </header>

      <p className={styles.intro}>
        这里没有评分，也没有标准答案。你可以回应一则题笺，提出异议，或从自己的经验谈起。
      </p>

      {conversation.turns.length === 0 && (
        <p className={styles.emptyNote}>彦远在等你选择一则题笺，也可以直接写下自己的问题。</p>
      )}

      <div className={styles.transcript} role="log" aria-live="polite" aria-relevant="additions text">
        {conversation.turns.map((turn) => (
          <article className={styles.turn} data-role={turn.role} key={turn.id}>
            <p className={styles.speaker}>{turn.role === "assistant" ? "彦远" : "访客"}</p>
            <div className={styles.turnBody}>
              <p>{turn.content}</p>
              {turn.notice && <p className={styles.serviceNotice}>{turn.notice}</p>}
              {turn.citations && turn.citations.length > 0 && (
                <details className={styles.citations}>
                  <summary>与本展材料的关联</summary>
                  <ul>
                    {turn.citations.map((citation, index) => (
                      <li key={`${citation.itemId || citation.objectId || "evidence"}-${index}`}>
                        {citation.label || citation.itemId || citation.objectId || `${citation.evidenceIds?.length ?? 0} 条馆藏依据`}
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
            <p className={styles.speaker}>访客</p>
            <div className={styles.turnBody}><p>{lastAttempt.message}</p></div>
          </article>
        )}
        {pending && (
          <div className={styles.pendingTurn} role="status">
            <span>彦远</span>
            <p>正在回看这场展览里的线索……</p>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {!hasVisitorTurn && openQuestions.length > 0 && (
        <fieldset className={styles.starters}>
          <legend>{conversation.selectedOpenQuestion ? "换一则题笺" : "从一则题笺开始"}</legend>
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
        <div className={styles.suggestions} aria-label="可以继续谈的方向">
          <span>可以接着谈</span>
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
              重试
            </button>
          )}
        </div>
      )}

      <form className={styles.form} onSubmit={submitDraft} onKeyDown={(event) => event.stopPropagation()}>
        <label htmlFor={`${formHintId}-input`}>写给彦远</label>
        <textarea
          ref={inputRef}
          id={`${formHintId}-input`}
          value={draft}
          rows={context === "hall" ? 2 : 3}
          maxLength={800}
          placeholder={conversation.selectedOpenQuestion ? "写下你的想法……" : "你注意到了什么？也可以提出不同看法。"}
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
            访客发言会发送给模型生成回应，但不写入展览记录或统计事件。
          </p>
          <button
            type="submit"
            className={styles.primaryButton}
            disabled={pending || trimEpilogueDraft(draft).length === 0}
          >
            {pending ? "彦远正在回应" : "发送"}
          </button>
        </div>
        {validationMessage && <p className={styles.validation} id={validationId}>{validationMessage}</p>}
      </form>

      <button type="button" className={styles.skipButton} onClick={() => closeConversation("skip")}>
        暂不参与这次讨论
      </button>
    </section>
  );
}
