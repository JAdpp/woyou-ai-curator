import { strict as assert } from "node:assert";
import test from "node:test";
import type { CuratorialBrief, ExhibitionItem } from "./types";
import {
  buildCuratorialEvidenceIndex,
  findCuratorialDecision,
  unresolvedEthicsMessages,
} from "./curatorialBrief";

const brief: CuratorialBrief = {
  schemaVersion: "curatorial-brief/v1",
  status: "deterministic",
  generatedAt: "2026-08-09T00:00:00Z",
  visitorInquiry: "猫在不同文化中意味着什么？",
  audience: {
    motivation: "explorer",
    priorKnowledge: "none",
    durationMinutes: 5,
    excludedTopics: [],
    voice: "plain",
    density: "short",
  },
  bigIdea: { id: "big-idea", text: "策展判断", evidenceIds: ["ev-1"], confidence: "provisional" },
  keyMessages: [],
  criticalQuestions: [],
  objects: [{
    itemId: "item-1",
    objectId: "object-1",
    role: "opening",
    selectionRationale: "与问题直接相关",
    relation: "作为入口",
    evidenceIds: ["ev-1"],
  }],
  excludedCandidates: [],
  ethics: {
    provenanceStatus: "not_reviewed",
    provenanceNotes: [],
    culturalSensitivityStatus: "not_reviewed",
    culturalSensitivity: [],
    communityReviewRequired: null,
    communityReviewStatus: "not_assessed",
    communityReviewNotes: [],
  },
  interpretationPolicy: {
    factPolicy: "机构记录保持原文",
    inferencePolicy: "系统解释单独标示",
    uncertaintyPolicy: "证据不足时保留不确定性",
    externalKnowledgeAllowed: false,
  },
  evaluationTargets: [],
  retrieval: {
    method: "bm25",
    version: "test",
    candidateCount: 12,
    selectedCount: 5,
  },
};

const item = {
  id: "item-1",
  displayTitle: "猫",
  object: {
    id: "object-1",
    title: "Cat",
    evidence: [{
      id: "ev-1",
      sourceTitle: "Museum record",
      sourceUrl: "https://museum.example/object-1",
      sourceLocation: "Description",
    }],
  },
} as ExhibitionItem;

test("resolves curatorial evidence to its object without claiming semantic proof", () => {
  const references = buildCuratorialEvidenceIndex([item]);
  assert.deepEqual(references["ev-1"], {
    id: "ev-1",
    itemId: "item-1",
    objectId: "object-1",
    objectTitle: "猫",
    sourceTitle: "Museum record",
    sourceUrl: "https://museum.example/object-1",
    sourceLocation: "Description",
  });
});

test("matches a curatorial decision by item or collection object id", () => {
  assert.equal(findCuratorialDecision(brief, item)?.selectionRationale, "与问题直接相关");
  assert.equal(
    findCuratorialDecision(brief, { ...item, id: "replacement-item" })?.objectId,
    "object-1",
  );
});

test("never turns missing ethics metadata into a cleared status", () => {
  const messages = unresolvedEthicsMessages(brief);
  assert.ok(messages.includes("来源史尚未审核"));
  assert.ok(messages.some((message) => message.includes("不等于已经排除风险")));
  assert.ok(messages.some((message) => message.includes("尚未评估")));
});
