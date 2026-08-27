import type { GenerationJob, InterviewState } from "./types";

export type VisitStage = "landing" | "interview" | "brief" | "curating" | "failed";

export interface VisitFlowState {
  stage: VisitStage;
  interview: InterviewState | null;
  failedJob: GenerationJob | null;
  curationAttempt: number;
}

export type VisitFlowAction =
  | { type: "start_interview" }
  | { type: "complete_interview"; interview: InterviewState }
  | { type: "skip_to_curation"; interview: InterviewState }
  | { type: "confirm_brief" }
  | { type: "curation_failed"; job: GenerationJob | null }
  | { type: "retry_curation" }
  | { type: "adjust_answers" }
  | { type: "return_home" };

export const INITIAL_VISIT_FLOW_STATE: VisitFlowState = {
  stage: "landing",
  interview: null,
  failedJob: null,
  curationAttempt: 0,
};

/**
 * Keep recovery transitions explicit. In particular, a failed generation must
 * retain the completed interview and failed job until the visitor chooses to
 * discard them by returning home or starting a fresh interview.
 */
export function visitFlowReducer(
  state: VisitFlowState,
  action: VisitFlowAction,
): VisitFlowState {
  switch (action.type) {
    case "start_interview":
    case "adjust_answers":
      return {
        ...state,
        stage: "interview",
        interview: null,
        failedJob: null,
      };
    case "complete_interview":
      return {
        ...state,
        stage: "brief",
        interview: action.interview,
        failedJob: null,
      };
    case "skip_to_curation":
      return {
        ...state,
        stage: "curating",
        interview: action.interview,
        failedJob: null,
      };
    case "confirm_brief":
      if (!state.interview) return state;
      return { ...state, stage: "curating", failedJob: null };
    case "curation_failed":
      if (!state.interview) return state;
      return { ...state, stage: "failed", failedJob: action.job };
    case "retry_curation":
      if (!state.interview) return state;
      return {
        ...state,
        stage: "curating",
        failedJob: null,
        curationAttempt: state.curationAttempt + 1,
      };
    case "return_home":
      return {
        ...INITIAL_VISIT_FLOW_STATE,
        curationAttempt: state.curationAttempt,
      };
  }
}
