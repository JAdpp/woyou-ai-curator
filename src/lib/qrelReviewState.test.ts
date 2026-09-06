import assert from "node:assert/strict";
import test from "node:test";
import type { QrelReviewCandidate, QrelReviewQuestionSummary } from "./types";
import {
  clampCandidateIndex,
  filterQrelQuestions,
  aiSuggestionDecision,
  isTextEntryTarget,
  hasMaterialAiValidationFailure,
  isHighRiskAiSuggestion,
  materialAiRiskFlags,
  manualDispositionForSuggestion,
  nextCandidateIndex,
  parseQrelReviewShortcut,
  reviewProgressPercent,
  reviewedCandidateCount,
  validateQrelJudgmentDraft,
} from "./qrelReviewState";

const questions: QrelReviewQuestionSummary[] = [
  {
    queryId: "001-open-theme",
    question: "窗、帘与屏风怎样安排可见与不可见？",
    category: "open_theme",
    judgmentMode: "pooled_silver",
    requiredCulturalLegs: [],
    candidateCount: 2,
    reviewedCandidateCount: 0,
    suggestedCandidateCount: 0,
    aiHighRiskCount: 0,
    status: "unreviewed",
  },
  {
    queryId: "002-cross-cultural",
    question: "不同文化中的守护动物",
    category: "cross_cultural",
    judgmentMode: "pooled_silver",
    requiredCulturalLegs: ["east_asia", "africa"],
    candidateCount: 2,
    reviewedCandidateCount: 1,
    suggestedCandidateCount: 2,
    aiHighRiskCount: 1,
    status: "in_progress",
  },
];

function candidate(id: string, reviewed: boolean): QrelReviewCandidate {
  return {
    objectId: id,
    culturalLegs: [],
    evidence: [],
    object: {
      id,
      accessionNumber: id,
      title: id,
      date: "",
      medium: "",
      type: "",
      imageUrl: "",
      objectUrl: "",
      rights: "",
      altText: id,
      themes: [],
      evidence: [],
    },
    judgment: reviewed
      ? {
          relevance: 2,
          evidenceVerdict: "supports",
          supportingEvidenceIds: [],
          note: "",
          revision: 1,
          updatedAt: "2026-09-01T00:00:00Z",
        }
      : null,
  };
}

test("question filtering combines status, category and normalized search", () => {
  assert.deepEqual(
    filterQrelQuestions(questions, {
      status: "in_progress",
      category: "cross_cultural",
      search: "守护",
      ai: "all",
    }).map((question) => question.queryId),
    ["002-cross-cultural"],
  );
  assert.equal(filterQrelQuestions(questions, { status: "all", category: "", search: "001", ai: "all" }).length, 1);
  assert.deepEqual(
    filterQrelQuestions(questions, { status: "all", category: "", search: "", ai: "high_risk" })
      .map((question) => question.queryId),
    ["002-cross-cultural"],
  );
  assert.deepEqual(
    filterQrelQuestions(questions, { status: "all", category: "", search: "", ai: "uncovered" })
      .map((question) => question.queryId),
    ["001-open-theme"],
  );
});

test("candidate navigation visits adjacent reviewed cards and wraps", () => {
  const candidates = [candidate("a", true), candidate("b", true), candidate("c", false)];
  assert.equal(nextCandidateIndex(candidates, 0, 1), 1);
  assert.equal(nextCandidateIndex(candidates, 2, -1), 1);
  assert.equal(nextCandidateIndex(candidates, 0, -1), 2);
  assert.equal(nextCandidateIndex(candidates.map((item) => ({ ...item, judgment: candidates[0]?.judgment })), 2, 1), 0);
});

test("progress and candidate index helpers stay bounded", () => {
  const candidates = [candidate("a", true), candidate("b", false)];
  assert.equal(reviewedCandidateCount(candidates), 1);
  assert.equal(reviewProgressPercent(1, 2), 50);
  assert.equal(reviewProgressPercent(12, 10), 100);
  assert.equal(clampCandidateIndex(2, -5), 0);
  assert.equal(clampCandidateIndex(2, 99), 1);
});

test("shortcuts are disabled during text entry and IME composition", () => {
  assert.deepEqual(parseQrelReviewShortcut("3", { tagName: "DIV" }), { kind: "relevance", value: 3 });
  assert.deepEqual(parseQrelReviewShortcut("ArrowRight", null), { kind: "next_candidate" });
  assert.equal(parseQrelReviewShortcut("2", { tagName: "TEXTAREA" }), null);
  assert.equal(parseQrelReviewShortcut("2", { tagName: "DIV" }, true), null);
  assert.equal(isTextEntryTarget({ tagName: "span", isContentEditable: true }), true);
});

test("supporting verdict requires both a relevant grade and cited evidence", () => {
  assert.equal(validateQrelJudgmentDraft({ relevance: 1, evidenceVerdict: "supports", supportingEvidenceIds: ["e1"] }), "“证据充分”只适用于相关或高度相关的候选");
  assert.equal(validateQrelJudgmentDraft({ relevance: 3, evidenceVerdict: "supports", supportingEvidenceIds: [] }), "选择“证据充分”时，至少勾选一条支撑证据");
  assert.equal(validateQrelJudgmentDraft({ relevance: 2, evidenceVerdict: "supports", supportingEvidenceIds: ["e1"] }), null);
});

test("AI suggestions remain provisional and manual edits retain their relationship", () => {
  const suggestion = {
    suggestionId: "suggestion-1",
    suggestionRunId: "run-1",
    relevance: 3 as const,
    evidenceVerdict: "supports" as const,
    supportingEvidenceIds: ["e1"],
    note: "AI-only rationale",
    confidenceBand: "low" as const,
    riskFlags: ["missing_cultural_leg"],
    modalities: ["metadata", "evidence"],
    provider: "provider",
    model: "model",
    promptVersion: "v1",
    promptHash: "prompt-hash",
    inputHash: "input-hash",
    validationStatus: "validated",
    generatedAt: "2026-09-01T00:00:00Z",
  };

  assert.deepEqual(aiSuggestionDecision(suggestion), {
    relevance: 3,
    evidenceVerdict: "supports",
    supportingEvidenceIds: ["e1"],
  });
  assert.equal(manualDispositionForSuggestion(suggestion), "modified");
  assert.equal(manualDispositionForSuggestion(null), "human_only");
  assert.equal(isHighRiskAiSuggestion(suggestion), true);
  assert.equal(reviewedCandidateCount([{ ...candidate("ai-only", false), aiSuggestion: suggestion }]), 0);

  const routineSuggestion = {
    ...suggestion,
    confidenceBand: "medium" as const,
    riskFlags: ["ai_draft", "needs_human_confirmation", "vision_reviewed"],
    validationStatus: "validated_text_only",
  };
  assert.deepEqual(materialAiRiskFlags(routineSuggestion), []);
  assert.equal(hasMaterialAiValidationFailure(routineSuggestion.validationStatus), false);
  assert.equal(isHighRiskAiSuggestion(routineSuggestion), false);

  const failedVisionSuggestion = {
    ...routineSuggestion,
    riskFlags: ["vision_reviewed", "image_fetch_failed:upstream_http_error"],
  };
  assert.deepEqual(materialAiRiskFlags(failedVisionSuggestion), ["image_fetch_failed:upstream_http_error"]);
  assert.equal(isHighRiskAiSuggestion(failedVisionSuggestion), true);

  const reviewedConfidentNegative = {
    ...routineSuggestion,
    relevance: 0 as const,
    evidenceVerdict: "not_applicable" as const,
    confidenceBand: "low" as const,
    riskFlags: ["low_confidence", "visual_review_required", "vision_reviewed"],
  };
  assert.deepEqual(materialAiRiskFlags(reviewedConfidentNegative), []);
  assert.equal(isHighRiskAiSuggestion(reviewedConfidentNegative), false);

  const visuallyUnverified = {
    ...reviewedConfidentNegative,
    riskFlags: ["low_confidence", "visual_review_required"],
  };
  assert.deepEqual(materialAiRiskFlags(visuallyUnverified), ["visual_review_required"]);
  assert.equal(isHighRiskAiSuggestion(visuallyUnverified), true);

  const unownedVisualSupport = {
    ...routineSuggestion,
    riskFlags: ["vision_reviewed", "vision_support_without_owned_evidence"],
  };
  assert.deepEqual(
    materialAiRiskFlags(unownedVisualSupport),
    ["vision_support_without_owned_evidence"],
  );
  assert.equal(isHighRiskAiSuggestion(unownedVisualSupport), true);

  assert.equal(
    isHighRiskAiSuggestion({ ...routineSuggestion, validationStatus: "validation_failed" }),
    true,
  );
});
