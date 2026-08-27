import assert from "node:assert/strict";
import test from "node:test";
import type { GenerationJob, InterviewState } from "./types";
import { INITIAL_VISIT_FLOW_STATE, visitFlowReducer } from "./visitFlowState";

const interview: InterviewState = {
  id: "interview-1",
  complete: true,
  profile: {
    curiosityLabel: "跨文化中的狗",
    freeFormQuestion: "狗为什么既是伙伴，也是守护者？",
    motivation: "explorer",
    priorKnowledge: "some",
    durationMinutes: 10,
    excludedTopics: [],
  },
  transcript: [],
  nextQuestion: null,
};

const failedJob: GenerationJob = {
  id: "job-1",
  status: "failed",
  progress: 25,
  stage: "确定展览主题",
  steps: [],
  error: "internal provider detail that must not be rendered",
};

test("a failed curation retains its interview and job for recovery", () => {
  const brief = visitFlowReducer(INITIAL_VISIT_FLOW_STATE, {
    type: "complete_interview",
    interview,
  });
  const curating = visitFlowReducer(brief, { type: "confirm_brief" });
  const failed = visitFlowReducer(curating, {
    type: "curation_failed",
    job: failedJob,
  });

  assert.equal(failed.stage, "failed");
  assert.equal(failed.interview?.id, interview.id);
  assert.equal(failed.failedJob?.id, failedJob.id);
});

test("retry keeps the interview and increments the attempt", () => {
  const failed = {
    stage: "failed" as const,
    interview,
    failedJob,
    curationAttempt: 2,
  };
  const retried = visitFlowReducer(failed, { type: "retry_curation" });

  assert.equal(retried.stage, "curating");
  assert.equal(retried.interview?.id, interview.id);
  assert.equal(retried.failedJob, null);
  assert.equal(retried.curationAttempt, 3);
});

test("adjusting answers deliberately starts a fresh interview", () => {
  const failed = {
    stage: "failed" as const,
    interview,
    failedJob,
    curationAttempt: 1,
  };
  const adjusted = visitFlowReducer(failed, { type: "adjust_answers" });

  assert.equal(adjusted.stage, "interview");
  assert.equal(adjusted.interview, null);
  assert.equal(adjusted.failedJob, null);
});
