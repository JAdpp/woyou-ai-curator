"use client";

import Link from "next/link";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ApiError,
  finalizeQrelQuestion,
  getQrelReviewQuestion,
  listQrelReviewQuestions,
  resolveObjectImageUrl,
  saveQrelJudgment,
} from "@/lib/api";
import {
  aiSuggestionDecision,
  filterQrelQuestions,
  isHighRiskAiSuggestion,
  materialAiRiskFlags,
  manualDispositionForSuggestion,
  nextCandidateIndex,
  parseQrelReviewShortcut,
  reviewProgressPercent,
  reviewedCandidateCount,
  validateQrelJudgmentDraft,
  type QrelQuestionFilterState,
} from "@/lib/qrelReviewState";
import type {
  EvidenceChunk,
  QrelAiCandidateSuggestion,
  QrelExpectedAnswerability,
  QrelEvidenceVerdict,
  QrelHumanJudgment,
  QrelHumanDisposition,
  QrelRelevance,
  QrelReviewCandidate,
  QrelReviewQuestionDetail,
  QrelReviewQuestionsResponse,
} from "@/lib/types";
import styles from "./qrelReview.module.css";

interface JudgmentDraft {
  relevance: QrelRelevance | null;
  evidenceVerdict: QrelEvidenceVerdict;
  supportingEvidenceIds: string[];
  note: string;
  revision: number | null;
  acceptedSuggestionId: string | null;
  disposition: QrelHumanDisposition;
}

interface FinalizationDraft {
  expectedAnswerability: QrelExpectedAnswerability;
  note: string;
  revision: number | null;
  acceptedSuggestionId: string | null;
  disposition: QrelHumanDisposition;
}

type SaveState = "idle" | "dirty" | "saving" | "saved" | "conflict" | "error";

const EMPTY_DRAFT: JudgmentDraft = {
  relevance: null,
  evidenceVerdict: "uncertain",
  supportingEvidenceIds: [],
  note: "",
  revision: null,
  acceptedSuggestionId: null,
  disposition: "human_only",
};

const DEFAULT_FILTERS: QrelQuestionFilterState = {
  status: "all",
  category: "",
  search: "",
  ai: "all",
};

const DEFAULT_FINALIZATION: FinalizationDraft = {
  expectedAnswerability: "partially_supported",
  note: "",
  revision: null,
  acceptedSuggestionId: null,
  disposition: "human_only",
};

const STATUS_LABELS = {
  unreviewed: "未开始",
  in_progress: "审阅中",
  complete: "已定稿",
  conflict: "有冲突",
} as const;

const RELEVANCE_OPTIONS: Array<{
  value: QrelRelevance;
  title: string;
  detail: string;
  key: string;
}> = [
  { value: 3, title: "高度相关", detail: "直接回答问题，可作为核心证据", key: "3" },
  { value: 2, title: "相关", detail: "支持问题的一部分或一个文化切面", key: "2" },
  { value: 1, title: "弱相关", detail: "有联系，但不足以承担论证", key: "1" },
  { value: 0, title: "不相关", detail: "不应进入该问题的相关集合", key: "0" },
];

const EVIDENCE_OPTIONS: Array<{
  value: QrelEvidenceVerdict;
  label: string;
}> = [
  { value: "supports", label: "证据充分" },
  { value: "insufficient", label: "证据不足" },
  { value: "contradicts", label: "证据冲突" },
  { value: "uncertain", label: "尚不确定" },
  { value: "not_applicable", label: "不适用" },
];

const RELEVANCE_LABELS: Record<QrelRelevance, string> = {
  0: "不相关",
  1: "弱相关",
  2: "相关",
  3: "高度相关",
};

const EVIDENCE_LABELS: Record<QrelEvidenceVerdict, string> = {
  supports: "证据充分",
  insufficient: "证据不足",
  contradicts: "证据冲突",
  uncertain: "尚不确定",
  not_applicable: "不适用",
};

const CONFIDENCE_LABELS = {
  high: "较高一致性",
  medium: "中等一致性",
  low: "低置信，建议优先核对",
} as const;

function confidenceLabel(suggestion: QrelAiCandidateSuggestion) {
  if (
    suggestion.confidenceBand === "low" &&
    suggestion.relevance === 0 &&
    suggestion.evidenceVerdict === "not_applicable"
  ) {
    return "明确不相关（AI 暂定）";
  }
  return CONFIDENCE_LABELS[suggestion.confidenceBand];
}

function requestId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `review-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function draftFromJudgment(judgment?: QrelHumanJudgment | null): JudgmentDraft {
  if (!judgment) return { ...EMPTY_DRAFT };
  return {
    relevance: judgment.relevance,
    evidenceVerdict: judgment.evidenceVerdict,
    supportingEvidenceIds: [...judgment.supportingEvidenceIds],
    note: judgment.note,
    revision: judgment.revision,
    acceptedSuggestionId: judgment.acceptedSuggestionId ?? null,
    disposition: judgment.disposition ?? "human_only",
  };
}

function draftSignature(draft: JudgmentDraft) {
  if (draft.relevance === null) return null;
  return JSON.stringify({
    relevance: draft.relevance,
    evidenceVerdict: draft.evidenceVerdict,
    supportingEvidenceIds: [...draft.supportingEvidenceIds].sort(),
    note: draft.note.trim(),
    acceptedSuggestionId: draft.acceptedSuggestionId,
    disposition: draft.disposition,
  });
}

function mergeEvidence(candidate: QrelReviewCandidate): EvidenceChunk[] {
  const byId = new Map<string, EvidenceChunk>();
  for (const chunk of [...candidate.object.evidence, ...candidate.evidence]) {
    if (!byId.has(chunk.id)) byId.set(chunk.id, chunk);
  }
  return [...byId.values()];
}

function readableError(error: unknown) {
  return error instanceof Error ? error.message : "审核服务暂时不可用";
}

function percentLabel(reviewed: number, total: number) {
  return `${reviewed.toLocaleString("zh-CN")} / ${total.toLocaleString("zh-CN")}`;
}

function answerabilityLabel(value: QrelExpectedAnswerability) {
  return {
    supported: "证据充分，可回答",
    partially_supported: "部分可回答",
    unsupported: "证据不足，不可回答",
  }[value];
}

function readableRiskFlag(value: string) {
  if (value.startsWith("image_fetch_failed:")) return "图像读取失败";
  if (value.startsWith("vision_review_failed:")) return "视觉复核失败";
  const labels: Record<string, string> = {
    low_confidence: "低置信",
    visual_review_required: "仍需视觉复核",
    vision_cache_unavailable: "视觉服务不可用",
    missing_cultural_leg: "文化语境未覆盖",
    cultural_leg_gap: "文化切面不足",
    pending_evidence: "引用证据待复核",
    weak_evidence: "证据支撑较弱",
    model_disagreement: "多次判断不一致",
    image_unavailable: "图像输入不可用",
    metadata_incomplete: "核心元数据不完整",
    validation_repaired: "输出经过修复",
  };
  return labels[value] ?? value.replaceAll("_", " ");
}

export function QrelReviewWorkbench() {
  const [loadingList, setLoadingList] = useState(true);
  const [listData, setListData] = useState<QrelReviewQuestionsResponse | null>(null);
  const [filters, setFilters] = useState<QrelQuestionFilterState>(DEFAULT_FILTERS);
  const [selectedQueryId, setSelectedQueryId] = useState<string | null>(null);
  const [detail, setDetail] = useState<QrelReviewQuestionDetail | null>(null);
  const [candidateIndex, setCandidateIndex] = useState(0);
  const [draft, setDraft] = useState<JudgmentDraft>({ ...EMPTY_DRAFT });
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [imageSource, setImageSource] = useState<"proxy" | "source" | "missing">("proxy");
  const [finalizing, setFinalizing] = useState(false);
  const [finalizationDraft, setFinalizationDraft] = useState<FinalizationDraft>(DEFAULT_FINALIZATION);
  const noteRef = useRef<HTMLTextAreaElement>(null);
  const lastSavedSignature = useRef<string | null>(null);
  const saveInFlight = useRef(false);

  const visibleQuestions = useMemo(
    () => filterQrelQuestions(listData?.questions ?? [], filters),
    [filters, listData?.questions],
  );
  const activeQueryId = visibleQuestions.some((question) => question.queryId === selectedQueryId)
    ? selectedQueryId
    : visibleQuestions[0]?.queryId ?? null;
  const activeDetail = detail?.question.queryId === activeQueryId ? detail : null;
  const candidate = activeDetail?.candidates[candidateIndex] ?? null;
  const evidence = useMemo(() => (candidate ? mergeEvidence(candidate) : []), [candidate]);
  const localReviewedCount = activeDetail ? reviewedCandidateCount(activeDetail.candidates) : 0;
  const allCandidatesReviewed = Boolean(
    activeDetail && activeDetail.candidates.length > 0 && localReviewedCount === activeDetail.candidates.length,
  );
  const busy = saveState === "saving" || finalizing;
  const navigationBlocked = busy || saveState === "dirty" || saveState === "conflict" || saveState === "error";
  const draftValidation = validateQrelJudgmentDraft(draft);
  const aiSuggestion = candidate?.aiSuggestion ?? null;
  const aiSuggestionHighRisk = isHighRiskAiSuggestion(aiSuggestion);
  const aiSuggestedEvidenceIds = useMemo(
    () => new Set(aiSuggestion?.supportingEvidenceIds ?? []),
    [aiSuggestion?.supportingEvidenceIds],
  );
  const aiSuggestedTotal = useMemo(
    () => (listData?.questions ?? []).reduce(
      (count, question) => count + (question.suggestedCandidateCount ?? 0),
      0,
    ),
    [listData?.questions],
  );
  const aiHighRiskTotal = useMemo(
    () => (listData?.questions ?? []).reduce(
      (count, question) => count + (question.aiHighRiskCount ?? 0),
      0,
    ),
    [listData?.questions],
  );

  const asManualDraft = useCallback((nextDraft: JudgmentDraft): JudgmentDraft => ({
    ...nextDraft,
    acceptedSuggestionId: aiSuggestion?.suggestionId ?? null,
    disposition: manualDispositionForSuggestion(aiSuggestion),
  }), [aiSuggestion]);

  const updateQuestionSummary = useCallback(
    (queryId: string, reviewedCount: number, status?: QrelReviewQuestionDetail["question"]["status"]) => {
      setListData((current) => current
        ? {
            ...current,
            questions: current.questions.map((question) => question.queryId === queryId
              ? {
                  ...question,
                  reviewedCandidateCount: reviewedCount,
                  status: status ?? (reviewedCount > 0 ? "in_progress" : question.status),
                }
              : question),
          }
        : current);
    },
    [],
  );

  const applyQuestionDetail = useCallback((response: QrelReviewQuestionDetail, preferredObjectId?: string) => {
    setDetail(response);
    const firstUnreviewed = response.candidates.findIndex((item) => !item.judgment);
    const preferredIndex = preferredObjectId
      ? response.candidates.findIndex((item) => item.objectId === preferredObjectId)
      : -1;
    const nextIndex = preferredIndex >= 0 ? preferredIndex : firstUnreviewed >= 0 ? firstUnreviewed : 0;
    setCandidateIndex(nextIndex);
    const nextDraft = draftFromJudgment(response.candidates[nextIndex]?.judgment);
    setDraft(nextDraft);
    lastSavedSignature.current = draftSignature(nextDraft);
    setSaveState(response.candidates[nextIndex]?.judgment ? "saved" : "idle");
    setImageSource("proxy");
    setFinalizationDraft(response.finalization
      ? {
          expectedAnswerability: response.finalization.expectedAnswerability,
          note: response.finalization.note,
          revision: response.finalization.revision,
          acceptedSuggestionId: response.finalization.acceptedSuggestionId ?? null,
          disposition: response.finalization.disposition ?? "human_only",
        }
      : {
          expectedAnswerability: response.question.expectedAnswerability === "supported" ||
            response.question.expectedAnswerability === "partially_supported" ||
            response.question.expectedAnswerability === "unsupported"
            ? response.question.expectedAnswerability
            : "partially_supported",
          note: "",
          revision: null,
          acceptedSuggestionId: null,
          disposition: "human_only",
        });
  }, []);

  const applyQuestionList = useCallback((response: QrelReviewQuestionsResponse) => {
    setListData(response);
    setSelectedQueryId((current) => current ?? response.questions[0]?.queryId ?? null);
    setLoadingList(false);
  }, []);

  const retryQuestionList = async () => {
    setLoadingList(true);
    setError(null);
    try {
      applyQuestionList(await listQrelReviewQuestions());
    } catch (loadError) {
      setLoadingList(false);
      setError(`问题队列载入失败：${readableError(loadError)}`);
    }
  };

  useEffect(() => {
    const controller = new AbortController();
    void listQrelReviewQuestions({}, { signal: controller.signal })
      .then(applyQuestionList)
      .catch((loadError: unknown) => {
        if (loadError instanceof DOMException && loadError.name === "AbortError") return;
        setLoadingList(false);
        setError(`问题队列载入失败：${readableError(loadError)}`);
      });
    return () => controller.abort();
  }, [applyQuestionList]);

  useEffect(() => {
    if (!activeQueryId) return;
    const controller = new AbortController();
    void getQrelReviewQuestion(activeQueryId, {
      signal: controller.signal,
    }).then(applyQuestionDetail).catch((loadError: unknown) => {
      if (loadError instanceof DOMException && loadError.name === "AbortError") return;
      setError(readableError(loadError));
    });
    return () => controller.abort();
  }, [activeQueryId, applyQuestionDetail]);

  const saveDraft = useCallback(async (nextDraft: JudgmentDraft) => {
    if (!activeDetail || !candidate || nextDraft.relevance === null || busy || saveInFlight.current) return;
    if (validateQrelJudgmentDraft(nextDraft)) {
      setSaveState("dirty");
      return;
    }
    const signature = draftSignature(nextDraft);
    if (signature === lastSavedSignature.current) {
      setSaveState("saved");
      setError(null);
      return;
    }
    setSaveState("saving");
    saveInFlight.current = true;
    setError(null);
    try {
      const response = await saveQrelJudgment(
        activeDetail.question.queryId,
        candidate.objectId,
        {
          relevance: nextDraft.relevance,
          evidenceVerdict: nextDraft.evidenceVerdict,
          supportingEvidenceIds: nextDraft.supportingEvidenceIds,
          note: nextDraft.note.trim(),
          expectedRevision: nextDraft.revision,
          acceptedSuggestionId: nextDraft.acceptedSuggestionId,
          disposition: nextDraft.disposition,
          requestId: requestId(),
        },
      );
      lastSavedSignature.current = signature;
      setDraft((current) => ({ ...current, revision: response.judgment.revision }));
      setDetail((current) => current
        ? {
            ...current,
            progress: response.progress,
            question: { ...current.question, status: "in_progress" },
            candidates: current.candidates.map((item) => item.objectId === candidate.objectId
              ? { ...item, judgment: response.judgment }
              : item),
          }
        : current);
      const updatedReviewed = activeDetail.candidates.reduce(
        (count, item) => count + (item.objectId === candidate.objectId || item.judgment ? 1 : 0),
        0,
      );
      updateQuestionSummary(activeDetail.question.queryId, updatedReviewed);
      setListData((current) => current ? { ...current, progress: response.progress } : current);
      setSaveState("saved");
    } catch (saveError) {
      if (saveError instanceof ApiError && saveError.status === 409) {
        setSaveState("conflict");
        setError("该候选已被另一个窗口更新。请重新载入后再判断");
      } else {
        setSaveState("error");
        setError(readableError(saveError));
      }
    } finally {
      saveInFlight.current = false;
    }
  }, [activeDetail, busy, candidate, updateQuestionSummary]);

  useEffect(() => {
    if (saveState !== "dirty" || draft.relevance === null) return;
    const timer = window.setTimeout(() => void saveDraft(draft), 750);
    return () => window.clearTimeout(timer);
  }, [draft, saveDraft, saveState]);

  const commitDraft = (nextDraft: JudgmentDraft) => {
    setDraft(nextDraft);
    setSaveState("dirty");
    void saveDraft(nextDraft);
  };

  const chooseRelevance = useCallback((value: QrelRelevance) => {
    if (busy || saveState === "conflict") return;
    commitDraft({
      ...asManualDraft(draft),
      relevance: value,
      evidenceVerdict: draft.evidenceVerdict === "supports" && value < 2
        ? (value === 0 ? "not_applicable" : "insufficient")
        : draft.evidenceVerdict,
    });
  // `commitDraft` deliberately captures the current object and revision.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [asManualDraft, busy, draft, saveState, saveDraft]);

  const chooseEvidenceVerdict = (value: QrelEvidenceVerdict) => {
    if (busy || saveState === "conflict") return;
    const nextDraft = asManualDraft({ ...draft, evidenceVerdict: value });
    setDraft(nextDraft);
    if (nextDraft.relevance !== null) {
      setSaveState("dirty");
      void saveDraft(nextDraft);
    }
  };

  const toggleSupportingEvidence = (evidenceId: string) => {
    if (busy || saveState === "conflict") return;
    const selected = draft.supportingEvidenceIds.includes(evidenceId);
    const nextDraft = asManualDraft({
      ...draft,
      supportingEvidenceIds: selected
        ? draft.supportingEvidenceIds.filter((id) => id !== evidenceId)
        : [...draft.supportingEvidenceIds, evidenceId],
    });
    setDraft(nextDraft);
    if (nextDraft.relevance !== null) {
      setSaveState("dirty");
      void saveDraft(nextDraft);
    }
  };

  const adoptAiSuggestion = () => {
    if (!aiSuggestion || busy || saveState === "conflict") return;
    const suggestionDecision = aiSuggestionDecision(aiSuggestion);
    const nextDraft: JudgmentDraft = {
      ...draft,
      ...suggestionDecision,
      note: draft.note,
      acceptedSuggestionId: aiSuggestion.suggestionId,
      disposition: "accepted",
    };
    commitDraft(nextDraft);
  };

  const moveCandidate = useCallback((direction: 1 | -1) => {
    if (!activeDetail || navigationBlocked) return;
    const nextIndex = nextCandidateIndex(activeDetail.candidates, candidateIndex, direction);
    const nextDraft = draftFromJudgment(activeDetail.candidates[nextIndex]?.judgment);
    setCandidateIndex(nextIndex);
    setDraft(nextDraft);
    lastSavedSignature.current = draftSignature(nextDraft);
    setSaveState(activeDetail.candidates[nextIndex]?.judgment ? "saved" : "idle");
    setImageSource("proxy");
  }, [activeDetail, candidateIndex, navigationBlocked]);

  const discardDraft = () => {
    if (busy) return;
    const savedDraft = draftFromJudgment(candidate?.judgment);
    setDraft(savedDraft);
    lastSavedSignature.current = draftSignature(savedDraft);
    setSaveState(candidate?.judgment ? "saved" : "idle");
    setError(null);
  };

  const reloadQuestion = useCallback(() => {
    if (!activeQueryId) return;
    setError(null);
    setSaveState("idle");
    setDetail(null);
    void getQrelReviewQuestion(activeQueryId)
      .then((response) => applyQuestionDetail(response, candidate?.objectId))
      .catch((loadError: unknown) => setError(readableError(loadError)));
  }, [activeQueryId, applyQuestionDetail, candidate?.objectId]);

  const finalizeQuestion = async () => {
    if (!activeDetail || !allCandidatesReviewed || busy) return;
    setFinalizing(true);
    setError(null);
    try {
      const response = await finalizeQrelQuestion(
        activeDetail.question.queryId,
        {
          expectedAnswerability: finalizationDraft.expectedAnswerability,
          note: finalizationDraft.note.trim(),
          expectedRevision: finalizationDraft.revision,
          acceptedSuggestionId: finalizationDraft.acceptedSuggestionId,
          disposition: finalizationDraft.disposition,
          requestId: requestId(),
        },
      );
      setDetail((current) => current
        ? { ...current, progress: response.progress, question: { ...current.question, status: response.status } }
        : current);
      setListData((current) => current ? { ...current, progress: response.progress } : current);
      setFinalizationDraft((current) => ({ ...current, revision: response.revision }));
      updateQuestionSummary(activeDetail.question.queryId, activeDetail.candidates.length, response.status);
    } catch (finalizeError) {
      setError(readableError(finalizeError));
    } finally {
      setFinalizing(false);
    }
  };

  useEffect(() => {
    const handleKeyboard = (event: globalThis.KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const action = parseQrelReviewShortcut(
        event.key,
        event.target as HTMLElement | null,
        event.isComposing,
      );
      if (!action) return;
      event.preventDefault();
      if (action.kind === "relevance") chooseRelevance(action.value);
      if (action.kind === "previous_candidate") moveCandidate(-1);
      if (action.kind === "next_candidate") moveCandidate(1);
      if (action.kind === "focus_note") noteRef.current?.focus();
    };
    window.addEventListener("keydown", handleKeyboard);
    return () => window.removeEventListener("keydown", handleKeyboard);
  }, [chooseRelevance, moveCandidate]);

  const progress = listData?.progress;
  const globalPercent = reviewProgressPercent(
    progress?.reviewedCandidates ?? 0,
    progress?.totalCandidates ?? 0,
  );

  return (
    <div className={styles.workbench}>
      <header className={styles.workbenchHeader}>
        <div className={styles.identity}>
          <Link href="/dev/admin" aria-label="返回内部审阅">卧游 / 内部</Link>
          <span aria-hidden="true">/</span>
          <strong>相关性审核</strong>
        </div>
        <div className={styles.globalProgress} aria-label={`候选审核进度 ${globalPercent}%`}>
          <span>全库进度</span>
          <div className={styles.progressTrack}><i style={{ width: `${globalPercent}%` }} /></div>
          <b>{percentLabel(progress?.reviewedCandidates ?? 0, progress?.totalCandidates ?? 0)}</b>
        </div>
        <div className={styles.aiCoverageSummary}>
          <span>AI 暂定覆盖 <strong>{aiSuggestedTotal.toLocaleString("zh-CN")}</strong></span>
          <span data-risk={aiHighRiskTotal > 0}>高风险 <strong>{aiHighRiskTotal.toLocaleString("zh-CN")}</strong></span>
          <small>人工确认 {progress?.humanReviewedCandidates ?? progress?.reviewedCandidates ?? 0} · 授权代审 {progress?.delegatedReviewedCandidates ?? 0}</small>
        </div>
      </header>

      {error && (
        <div className={styles.alert} role="alert">
          <span>{error}</span>
          {!listData
            ? <button type="button" onClick={retryQuestionList}>重试载入</button>
            : (saveState === "conflict" || !detail) && <button type="button" onClick={reloadQuestion}>重新载入</button>}
        </div>
      )}

      <main className={styles.reviewGrid}>
        <aside className={styles.questionRail} aria-label="待审问题">
          <header>
            <div>
              <span>问题队列</span>
              <strong>{visibleQuestions.length}</strong>
            </div>
            <small>{listData?.benchmarkId ?? "retrieval_eval_v1"}</small>
          </header>
          <div className={styles.filters}>
            <label>
              <span className="sr-only">搜索问题</span>
              <input
                type="search"
                value={filters.search}
                onChange={(event) => setFilters((current) => ({ ...current, search: event.target.value }))}
                placeholder="搜索问题或编号"
                disabled={loadingList || navigationBlocked}
              />
            </label>
            <div>
              <label>
                <span className="sr-only">状态</span>
                <select
                  value={filters.status}
                  disabled={loadingList || navigationBlocked}
                  onChange={(event) => setFilters((current) => ({
                    ...current,
                    status: event.target.value as QrelQuestionFilterState["status"],
                  }))}
                >
                  <option value="all">全部状态</option>
                  <option value="unreviewed">未开始</option>
                  <option value="in_progress">审阅中</option>
                  <option value="complete">已定稿</option>
                  <option value="conflict">有冲突</option>
                </select>
              </label>
              <label>
                <span className="sr-only">类别</span>
                <select
                  value={filters.category}
                  disabled={loadingList || navigationBlocked}
                  onChange={(event) => setFilters((current) => ({ ...current, category: event.target.value }))}
                >
                  <option value="">全部类别</option>
                  {(listData?.categories ?? []).map((category) => (
                    <option value={category} key={category}>{category.replaceAll("_", " ")}</option>
                  ))}
                </select>
              </label>
            </div>
            <label>
              <span className="sr-only">AI 建议状态</span>
              <select
                value={filters.ai}
                disabled={loadingList || navigationBlocked}
                onChange={(event) => setFilters((current) => ({
                  ...current,
                  ai: event.target.value as QrelQuestionFilterState["ai"],
                }))}
              >
                <option value="all">全部 AI 建议状态</option>
                <option value="covered">AI 已覆盖全部候选</option>
                <option value="uncovered">AI 尚未完整覆盖</option>
                <option value="high_risk">含高风险 AI 判断</option>
              </select>
            </label>
          </div>
          <ol className={styles.questionList}>
            {visibleQuestions.map((question, index) => {
              const percent = reviewProgressPercent(question.reviewedCandidateCount, question.candidateCount);
              return (
                <li key={question.queryId}>
                  <button
                    type="button"
                    data-active={question.queryId === activeQueryId}
                    data-status={question.status}
                    onClick={() => !navigationBlocked && setSelectedQueryId(question.queryId)}
                    disabled={navigationBlocked}
                  >
                    <span className={styles.questionIndex}>{String(index + 1).padStart(3, "0")}</span>
                    <span className={styles.questionCopy}>{question.question}</span>
                    <span className={styles.questionFoot}>
                      <i style={{ width: `${percent}%` }} />
                      <em>{STATUS_LABELS[question.status]} · {question.reviewedCandidateCount}/{question.candidateCount}</em>
                    </span>
                    <span className={styles.questionAiMeta} data-risk={(question.aiHighRiskCount ?? 0) > 0}>
                      AI {question.suggestedCandidateCount ?? 0}/{question.candidateCount}
                      {(question.aiHighRiskCount ?? 0) > 0 && ` · 高风险 ${question.aiHighRiskCount}`}
                    </span>
                  </button>
                </li>
              );
            })}
          </ol>
        </aside>

        <section className={styles.evidenceDesk} aria-live="polite">
          {!detail || !candidate ? (
            <div className={styles.loadingPlate}>
              <span className={styles.loader} aria-hidden="true" />
              <p>{loadingList ? "正在载入审核问题…" : visibleQuestions.length === 0 ? "当前筛选没有问题" : "正在展开藏品与证据…"}</p>
            </div>
          ) : (
            <>
              <header className={styles.queryHeader}>
                <div>
                  <span>{detail.question.queryId} · {detail.question.category.replaceAll("_", " ")}</span>
                  <h1>{detail.question.question}</h1>
                  {detail.question.requiredCulturalLegs.length > 0 && (
                    <p>预期文化切面：{detail.question.requiredCulturalLegs.join(" · ")}</p>
                  )}
                </div>
                <div className={styles.queryStatus} data-status={detail.question.status}>
                  <span>{STATUS_LABELS[detail.question.status]}</span>
                  <b>{localReviewedCount}/{detail.candidates.length}</b>
                </div>
              </header>

              <nav className={styles.candidateNavigator} aria-label="候选藏品导航">
                <button type="button" onClick={() => moveCandidate(-1)} disabled={navigationBlocked || detail.candidates.length < 2}>
                  <span aria-hidden="true">←</span> 上一件
                </button>
                <div>
                  <span>候选藏品</span>
                  <strong>{candidateIndex + 1} / {detail.candidates.length}</strong>
                  <em>
                    {candidate.judgment
                      ? candidate.judgment.reviewOrigin === "delegated_ai"
                        ? "已完成授权 AI 代审"
                        : candidate.judgment.disposition === "accepted"
                        ? "AI 建议已由你确认"
                        : candidate.aiSuggestion
                          ? "已由你判断，未直接采用 AI"
                          : "候选，已由你判断"
                      : candidate.aiSuggestion
                        ? "AI 已暂定，待你确认"
                        : "候选，待你判断"}
                  </em>
                </div>
                <button type="button" onClick={() => moveCandidate(1)} disabled={navigationBlocked || detail.candidates.length < 2}>
                  下一件 <span aria-hidden="true">→</span>
                </button>
              </nav>

              <article className={styles.objectSheet}>
                <figure className={styles.imagePlate}>
                  {imageSource === "missing" ? (
                    <div className={styles.imageUnavailable} role="img" aria-label={`${candidate.object.title}的图像暂时无法载入`}>
                      <span>图像暂时无法载入</span>
                      <small>仍可依据馆藏元数据与来源证据完成判断</small>
                    </div>
                  ) : (
                    /* Institution hosts vary; a failed proxy gets one source-URL fallback. */
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img
                      key={`${candidate.objectId}-${imageSource}`}
                      src={imageSource === "source" ? candidate.object.imageUrl : resolveObjectImageUrl(candidate.objectId, 1024)}
                      alt={candidate.object.altText || candidate.object.title}
                      onError={() => setImageSource((current) => current === "proxy" && candidate.object.imageUrl ? "source" : "missing")}
                    />
                  )}
                  <figcaption>
                    <span>{candidate.object.institution || "来源机构未标明"}</span>
                    <span>{candidate.object.imageLicense || candidate.object.rights || "图像许可待核对"}</span>
                  </figcaption>
                </figure>

                <div className={styles.objectRecord}>
                  <p className={styles.recordNumber}>{candidate.object.accessionNumber || candidate.object.id}</p>
                  <h2>{candidate.object.title}</h2>
                  {candidate.object.titleOriginal && candidate.object.titleOriginal !== candidate.object.title && (
                    <p className={styles.originalTitle}>{candidate.object.titleOriginal}</p>
                  )}
                  <dl>
                    <div><dt>作者 / 制作者</dt><dd>{candidate.object.creator || candidate.object.maker || "未详"}</dd></div>
                    <div><dt>年代</dt><dd>{candidate.object.date || "未详"}</dd></div>
                    <div><dt>文化 / 地域</dt><dd>{candidate.object.culture || "未详"}</dd></div>
                    <div><dt>材质</dt><dd>{candidate.object.medium || "未详"}</dd></div>
                    <div><dt>类别</dt><dd>{candidate.object.type || candidate.object.classification || "未详"}</dd></div>
                    <div><dt>馆藏</dt><dd>{candidate.object.institution || "未详"}</dd></div>
                  </dl>
                  {candidate.object.description && <p className={styles.description}>{candidate.object.description}</p>}
                  <div className={styles.objectFoot}>
                    <span>{candidate.culturalLegs.length > 0 ? candidate.culturalLegs.join(" · ") : "文化语境尚未路由"}</span>
                    {candidate.object.objectUrl && (
                      <a href={candidate.object.objectUrl} target="_blank" rel="noreferrer">查看机构原始记录 ↗</a>
                    )}
                  </div>
                </div>
              </article>

              <section className={styles.evidenceSection} aria-labelledby="evidence-title">
                <header>
                  <div>
                    <span>02 / SOURCE EVIDENCE</span>
                    <h2 id="evidence-title">全部可用证据</h2>
                  </div>
                  <p>勾选真正支撑本次判断的片段。证据内容保持机构原文，不在审核界面自动改写。</p>
                </header>
                {evidence.length === 0 ? (
                  <div className={styles.noEvidence}>当前冻结记录没有证据片段，请将证据结论标为“不足”或“不适用”</div>
                ) : (
                  <ol className={styles.evidenceList}>
                    {evidence.map((chunk, index) => {
                      const selected = draft.supportingEvidenceIds.includes(chunk.id);
                      return (
                        <li
                          key={chunk.id}
                          data-selected={selected}
                          data-ai-suggested={aiSuggestedEvidenceIds.has(chunk.id)}
                        >
                          <label>
                            <input
                              type="checkbox"
                              checked={selected}
                              onChange={() => toggleSupportingEvidence(chunk.id)}
                              disabled={busy || saveState === "conflict"}
                            />
                            <span className={styles.evidenceOrdinal}>{String(index + 1).padStart(2, "0")}</span>
                            <span className={styles.evidenceBody}>
                              <strong>
                                {chunk.sourceTitle}
                                {aiSuggestedEvidenceIds.has(chunk.id) && <em>AI 建议引用</em>}
                              </strong>
                              <span>{chunk.text}</span>
                              <small>{chunk.sourceLocation} · {chunk.reviewStatus === "reviewed" ? "来源字段已核对" : "来源字段待核对"}</small>
                            </span>
                          </label>
                          {chunk.sourceUrl && <a href={chunk.sourceUrl} target="_blank" rel="noreferrer">原始来源</a>}
                        </li>
                      );
                    })}
                  </ol>
                )}
              </section>
            </>
          )}
        </section>

        <aside className={styles.actionDock} aria-label="审核操作" aria-busy={busy}>
          <AiSuggestionCard
            candidate={candidate}
            highRisk={aiSuggestionHighRisk}
            busy={busy || saveState === "conflict"}
            invalid={aiSuggestion ? validateQrelJudgmentDraft(aiSuggestion) : null}
            onAdopt={adoptAiSuggestion}
          />

          <header>
            <span>{candidate?.judgment?.reviewOrigin === "delegated_ai" ? "代审结论 · 可修改" : "你的判断"}</span>
            <SaveIndicator state={saveState} />
          </header>

          <fieldset disabled={!candidate || busy || saveState === "conflict"}>
            <legend>相关性等级</legend>
            <div className={styles.relevanceButtons}>
              {RELEVANCE_OPTIONS.map((option) => (
                <button
                  type="button"
                  key={option.value}
                  data-value={option.value}
                  data-selected={draft.relevance === option.value}
                  onClick={() => chooseRelevance(option.value)}
                  aria-keyshortcuts={option.key}
                  aria-pressed={draft.relevance === option.value}
                >
                  <kbd>{option.key}</kbd>
                  <span><strong>{option.title}</strong><small>{option.detail}</small></span>
                </button>
              ))}
            </div>
          </fieldset>

          <fieldset disabled={!candidate || busy || saveState === "conflict"}>
            <legend>证据结论</legend>
            <div className={styles.verdictButtons}>
              {EVIDENCE_OPTIONS.map((option) => (
                <button
                  type="button"
                  key={option.value}
                  data-selected={draft.evidenceVerdict === option.value}
                  onClick={() => chooseEvidenceVerdict(option.value)}
                  aria-pressed={draft.evidenceVerdict === option.value}
                >
                  {option.label}
                </button>
              ))}
            </div>
            {(draft.relevance !== null || saveState === "dirty") && draftValidation && (
              <p className={styles.validationMessage} role="status">{draftValidation}</p>
            )}
          </fieldset>

          <label className={styles.noteField}>
            <span>审阅备注 <kbd>N</kbd></span>
            <textarea
              ref={noteRef}
              value={draft.note}
              onChange={(event) => {
                setDraft((current) => asManualDraft({ ...current, note: event.target.value }));
                setSaveState("dirty");
              }}
              placeholder="仅记录需要复核的歧义、排除理由或来源问题"
              maxLength={2000}
              disabled={!candidate || busy || saveState === "conflict"}
            />
            <small>{draft.note.length}/2000 · 停止输入后自动保存</small>
          </label>

          {saveState === "conflict" && (
            <button className={styles.reloadButton} type="button" onClick={reloadQuestion}>放弃本地版本并重新载入</button>
          )}
          {saveState === "error" && (
            <button className={styles.reloadButton} type="button" onClick={() => void saveDraft(draft)}>重试保存当前判断</button>
          )}
          {(saveState === "dirty" || saveState === "error") && (
            <button className={styles.reloadButton} type="button" onClick={discardDraft} disabled={busy}>撤销本次未保存修改</button>
          )}

          <div className={styles.dockNavigation}>
            <button type="button" onClick={() => moveCandidate(-1)} disabled={navigationBlocked || !candidate} aria-keyshortcuts="ArrowLeft">← 上一件</button>
            <button type="button" onClick={() => moveCandidate(1)} disabled={navigationBlocked || !candidate} aria-keyshortcuts="ArrowRight">下一件 →</button>
          </div>

          <div className={styles.finalizeBlock} data-ready={allCandidatesReviewed}>
            <div>
              <span>本题审核</span>
              <strong>{localReviewedCount}/{detail?.candidates.length ?? 0}</strong>
            </div>
            {detail?.aiQuestionSuggestion && (
              <section className={styles.aiQuestionSuggestion}>
                <span>{detail.finalization?.reviewOrigin === "delegated_ai" ? "问题级代审已完成" : "AI 问题级建议"}</span>
                <strong>{answerabilityLabel(detail.aiQuestionSuggestion.expectedAnswerability)}</strong>
                {detail.aiQuestionSuggestion.note && <p>{detail.aiQuestionSuggestion.note}</p>}
                <button
                  type="button"
                  onClick={() => setFinalizationDraft((current) => ({
                    ...current,
                    expectedAnswerability: detail.aiQuestionSuggestion?.expectedAnswerability ?? current.expectedAnswerability,
                    note: detail.aiQuestionSuggestion?.note ?? current.note,
                    acceptedSuggestionId: detail.aiQuestionSuggestion?.suggestionId ?? null,
                    disposition: "accepted",
                  }))}
                  disabled={!allCandidatesReviewed || navigationBlocked}
                >
                  填入定稿表单（仍需确认）
                </button>
              </section>
            )}
            <label>
              <span>问题总体可回答性</span>
              <select
                value={finalizationDraft.expectedAnswerability}
                onChange={(event) => setFinalizationDraft((current) => ({
                  ...current,
                  expectedAnswerability: event.target.value as QrelExpectedAnswerability,
                  disposition: current.acceptedSuggestionId ? "modified" : "human_only",
                }))}
                disabled={!allCandidatesReviewed || navigationBlocked}
              >
                <option value="supported">证据充分，可回答</option>
                <option value="partially_supported">部分可回答</option>
                <option value="unsupported">证据不足，不可回答</option>
              </select>
            </label>
            <label>
              <span>定稿说明</span>
              <textarea
                value={finalizationDraft.note}
                onChange={(event) => setFinalizationDraft((current) => ({ ...current, note: event.target.value, disposition: current.acceptedSuggestionId ? "modified" : "human_only" }))}
                placeholder="概括证据覆盖边界或定稿依据"
                maxLength={4000}
                disabled={!allCandidatesReviewed || navigationBlocked}
              />
            </label>
            <small>并发保护版本：{finalizationDraft.revision ?? "首次定稿"}</small>
            <button
              type="button"
              onClick={finalizeQuestion}
              disabled={!allCandidatesReviewed || navigationBlocked}
            >
              {finalizing ? "正在保存定稿…" : detail?.question.status === "complete" ? "更新本题定稿" : "确认本题定稿"}
            </button>
          </div>

          <p className={styles.shortcutHint}>
            <kbd>0–3</kbd> 判断相关性 · <kbd>←</kbd><kbd>→</kbd> 切换藏品 · 输入框与中文输入法组合期间自动停用快捷键
          </p>
        </aside>
      </main>
    </div>
  );
}

function SaveIndicator({ state }: { state: SaveState }) {
  const label = {
    idle: "尚未判断",
    dirty: "等待保存",
    saving: "正在保存",
    saved: "已自动保存",
    conflict: "版本冲突",
    error: "保存失败",
  }[state];
  return <span className={styles.saveIndicator} data-state={state} role="status">{label}</span>;
}

function AiSuggestionCard({
  candidate,
  highRisk,
  busy,
  invalid,
  onAdopt,
}: {
  candidate: QrelReviewCandidate | null;
  highRisk: boolean;
  busy: boolean;
  invalid: string | null;
  onAdopt: () => void;
}) {
  const suggestion = candidate?.aiSuggestion;
  if (!candidate) {
    return <section className={styles.aiSuggestionCard} data-empty="true">正在读取 AI 暂定建议…</section>;
  }
  if (!suggestion) {
    return (
      <section className={styles.aiSuggestionCard} data-empty="true">
        <span>AI 暂定建议</span>
        <strong>当前候选尚无 AI 建议</strong>
        <p>你仍可直接完成人工判断；缺少 AI 建议不会阻断审核。</p>
      </section>
    );
  }

  const relationship = candidate.judgment?.reviewOrigin === "delegated_ai"
    ? "已按你的授权完成 AI 代审，可查看或修改下方结论"
    : candidate.judgment?.disposition === "accepted"
    ? "AI 建议已由你确认"
    : candidate.judgment
      ? "你已作出独立或修改后的判断"
      : "AI 暂定，尚未由你确认";
  const substantiveRiskFlags = materialAiRiskFlags(suggestion);

  return (
    <section className={styles.aiSuggestionCard} data-risk={highRisk} aria-labelledby="ai-suggestion-title">
      <header>
        <div>
          <span>AI PROVISIONAL</span>
          <h2 id="ai-suggestion-title">AI 暂定建议</h2>
        </div>
        <strong data-confidence={suggestion.confidenceBand}>
          {confidenceLabel(suggestion)}
        </strong>
      </header>
      <p className={styles.aiBoundary}>{relationship}。</p>
      <dl className={styles.aiDecision}>
        <div><dt>相关性</dt><dd>{RELEVANCE_LABELS[suggestion.relevance]}</dd></div>
        <div><dt>证据结论</dt><dd>{EVIDENCE_LABELS[suggestion.evidenceVerdict]}</dd></div>
        <div><dt>建议引用</dt><dd>{suggestion.supportingEvidenceIds.length} 条</dd></div>
      </dl>
      {suggestion.note && (
        <div className={styles.aiNote}>
          <span>AI 说明，不是馆方证据</span>
          <p>{suggestion.note}</p>
        </div>
      )}
      {substantiveRiskFlags.length > 0 && (
        <div className={styles.riskFlags} aria-label="AI 判断风险">
          {substantiveRiskFlags.map((flag) => <span key={flag}>{readableRiskFlag(flag)}</span>)}
        </div>
      )}
      {invalid && <p className={styles.aiInvalid}>该建议未通过人工保存规则：{invalid}</p>}
      <button
        className={styles.adoptSuggestion}
        type="button"
        onClick={onAdopt}
        disabled={busy || Boolean(invalid)}
      >
        采用并确认 AI 建议
      </button>
      <details className={styles.aiProvenance}>
        <summary>查看 AI 建议来源</summary>
        <dl>
          <div><dt>模型</dt><dd>{suggestion.provider} / {suggestion.model}</dd></div>
          <div><dt>提示词</dt><dd>{suggestion.promptVersion} · {suggestion.promptHash.slice(0, 12)}</dd></div>
          <div><dt>输入模态</dt><dd>{suggestion.modalities.join(" · ") || "未记录"}</dd></div>
          <div><dt>校验</dt><dd>{suggestion.validationStatus}</dd></div>
          <div><dt>记录标记</dt><dd>{suggestion.riskFlags.join(" · ") || "无"}</dd></div>
          <div><dt>运行</dt><dd>{suggestion.suggestionRunId}</dd></div>
          <div><dt>生成时间</dt><dd>{new Date(suggestion.generatedAt).toLocaleString("zh-CN")}</dd></div>
          <div><dt>输入摘要</dt><dd>{suggestion.inputHash.slice(0, 16)}</dd></div>
        </dl>
      </details>
    </section>
  );
}
