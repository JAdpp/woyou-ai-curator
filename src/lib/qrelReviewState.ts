import type {
  QrelAiCandidateSuggestion,
  QrelEvidenceVerdict,
  QrelRelevance,
  QrelReviewCandidate,
  QrelReviewQuestionSummary,
  QrelReviewStatus,
  QrelHumanDisposition,
} from "./types";

export type QrelQuestionStatusFilter = QrelReviewStatus | "all";
export type QrelAiCoverageFilter = "all" | "covered" | "uncovered" | "high_risk";

export interface QrelQuestionFilterState {
  status: QrelQuestionStatusFilter;
  category: string;
  search: string;
  ai: QrelAiCoverageFilter;
}

export type QrelReviewShortcut =
  | { kind: "relevance"; value: 0 | 1 | 2 | 3 }
  | { kind: "previous_candidate" }
  | { kind: "next_candidate" }
  | { kind: "focus_note" };

export interface ShortcutTargetLike {
  tagName?: string | null;
  isContentEditable?: boolean;
}

export interface QrelJudgmentDraftLike {
  relevance: QrelRelevance | null;
  evidenceVerdict: QrelEvidenceVerdict;
  supportingEvidenceIds: string[];
}

export function manualDispositionForSuggestion(
  suggestion?: QrelAiCandidateSuggestion | null,
): QrelHumanDisposition {
  return suggestion ? "modified" : "human_only";
}

export function aiSuggestionDecision(suggestion: QrelAiCandidateSuggestion) {
  return {
    relevance: suggestion.relevance,
    evidenceVerdict: suggestion.evidenceVerdict,
    supportingEvidenceIds: [...suggestion.supportingEvidenceIds],
  };
}

// These flags describe the AI-review workflow, not uncertainty in the
// suggestion itself. Keep this allow-list intentionally narrow: unknown flags
// are not automatically promoted to "high risk" below.
const AI_WORKFLOW_ONLY_FLAGS = new Set([
  "ai_draft",
  "needs_human_confirmation",
  "vision_reviewed",
  "text_reviewed",
  "metadata_reviewed",
  "evidence_reviewed",
]);

const MATERIAL_AI_RISK_FLAGS = new Set([
  "low_confidence",
  "visual_review_required",
  "visual_confirmation_recommended",
  "image_unavailable",
  "vision_cache_unavailable",
  "pending_evidence",
  "weak_evidence",
  "insufficient_evidence",
  "unsupported_evidence",
  "evidence_boundary",
  "evidence_boundary_risk",
  "visual_relation_unverified",
  "missing_cultural_leg",
  "cultural_leg_gap",
  "model_disagreement",
  "metadata_incomplete",
  "validation_repaired",
  "vision_support_without_owned_evidence",
  "lexical_only",
  "partial",
  "insufficient",
]);

const MATERIAL_AI_RISK_MARKERS = [
  "_failed",
  "_failure",
  "_error",
  "_unavailable",
  "_timeout",
  "_invalid",
  "_mismatch",
  "_unverified",
  "_insufficient",
  "_unsupported",
  "_incomplete",
  "_disagreement",
  "_repaired",
  "_fallback",
  "evidence_boundary",
];

const MATERIAL_VALIDATION_MARKERS = [
  "failed",
  "failure",
  "error",
  "invalid",
  "rejected",
  "unavailable",
  "timeout",
  "repaired",
  "fallback",
  "not_validated",
  "unvalidated",
];

function normalizeAiRiskValue(value: string) {
  return value.trim().toLocaleLowerCase("en-US").replace(/[\s-]+/g, "_");
}

export function materialAiRiskFlags(suggestion?: QrelAiCandidateSuggestion | null) {
  if (!suggestion) return [];
  const confidentNegative = suggestion.relevance === 0 &&
    suggestion.evidenceVerdict === "not_applicable";
  const normalizedFlags = new Set(suggestion.riskFlags.map(normalizeAiRiskValue));
  const visionReviewed = normalizedFlags.has("vision_reviewed");
  return suggestion.riskFlags.filter((flag) => {
    const normalized = normalizeAiRiskValue(flag);
    if (!normalized || AI_WORKFLOW_ONLY_FLAGS.has(normalized)) return false;
    if (normalized === "low_confidence" && confidentNegative) return false;
    if (normalized === "visual_review_required" && visionReviewed) return false;
    return MATERIAL_AI_RISK_FLAGS.has(normalized) ||
      MATERIAL_AI_RISK_MARKERS.some((marker) => normalized.includes(marker));
  });
}

export function hasMaterialAiValidationFailure(validationStatus: string) {
  const normalized = normalizeAiRiskValue(validationStatus);
  if (!normalized) return true;
  return MATERIAL_VALIDATION_MARKERS.some((marker) => normalized.includes(marker));
}

export function isHighRiskAiSuggestion(suggestion?: QrelAiCandidateSuggestion | null) {
  if (!suggestion) return false;
  const confidentNegative = suggestion.relevance === 0 &&
    suggestion.evidenceVerdict === "not_applicable";
  return (suggestion.confidenceBand === "low" && !confidentNegative) ||
    materialAiRiskFlags(suggestion).length > 0 ||
    hasMaterialAiValidationFailure(suggestion.validationStatus);
}

export function validateQrelJudgmentDraft(draft: QrelJudgmentDraftLike) {
  if (draft.relevance === null) return "请先判断相关性等级";
  if (draft.evidenceVerdict === "supports" && draft.relevance < 2) {
    return "“证据充分”只适用于相关或高度相关的候选";
  }
  if (draft.evidenceVerdict === "supports" && draft.supportingEvidenceIds.length === 0) {
    return "选择“证据充分”时，至少勾选一条支撑证据";
  }
  return null;
}

export function filterQrelQuestions(
  questions: QrelReviewQuestionSummary[],
  filters: QrelQuestionFilterState,
) {
  const search = filters.search.trim().toLocaleLowerCase("zh-CN");
  return questions.filter((question) => {
    const suggestedCount = question.suggestedCandidateCount ?? 0;
    const highRiskCount = question.aiHighRiskCount ?? 0;
    if (filters.status !== "all" && question.status !== filters.status) return false;
    if (filters.category && question.category !== filters.category) return false;
    if (filters.ai === "covered" && suggestedCount < question.candidateCount) return false;
    if (filters.ai === "uncovered" && suggestedCount >= question.candidateCount) return false;
    if (filters.ai === "high_risk" && highRiskCount <= 0) return false;
    if (!search) return true;
    return `${question.queryId} ${question.question} ${question.category}`
      .toLocaleLowerCase("zh-CN")
      .includes(search);
  });
}

export function clampCandidateIndex(candidateCount: number, index: number) {
  if (candidateCount <= 0) return 0;
  return Math.min(Math.max(index, 0), candidateCount - 1);
}

/**
 * Previous/next always means the adjacent card, including reviewed cards.
 */
export function nextCandidateIndex(
  candidates: QrelReviewCandidate[],
  currentIndex: number,
  direction: 1 | -1 = 1,
) {
  if (candidates.length === 0) return 0;
  const start = clampCandidateIndex(candidates.length, currentIndex);
  return (start + direction + candidates.length) % candidates.length;
}

export function reviewedCandidateCount(candidates: QrelReviewCandidate[]) {
  return candidates.reduce((count, candidate) => count + (candidate.judgment ? 1 : 0), 0);
}

export function reviewProgressPercent(reviewed: number, total: number) {
  if (total <= 0) return 0;
  return Math.round((Math.min(Math.max(reviewed, 0), total) / total) * 100);
}

export function isTextEntryTarget(target: ShortcutTargetLike | null) {
  const tagName = target?.tagName?.toUpperCase();
  return Boolean(
    target?.isContentEditable ||
    tagName === "INPUT" ||
    tagName === "TEXTAREA" ||
    tagName === "SELECT",
  );
}

export function parseQrelReviewShortcut(
  key: string,
  target: ShortcutTargetLike | null,
  isComposing = false,
): QrelReviewShortcut | null {
  if (isComposing || isTextEntryTarget(target)) return null;
  if (key === "0" || key === "1" || key === "2" || key === "3") {
    return { kind: "relevance", value: Number(key) as 0 | 1 | 2 | 3 };
  }
  if (key === "ArrowLeft") return { kind: "previous_candidate" };
  if (key === "ArrowRight") return { kind: "next_candidate" };
  if (key.toLocaleLowerCase("en-US") === "n") return { kind: "focus_note" };
  return null;
}
