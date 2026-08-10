import type {
  EpilogueChatCitation,
  EpilogueChatRole,
  EpilogueChatTurn,
} from "@/lib/types";

export interface EpilogueConversationTurn extends EpilogueChatTurn {
  id: string;
  citations?: EpilogueChatCitation[];
  openingQuestion?: boolean;
  notice?: string;
}

export interface EpilogueConversationState {
  turns: EpilogueConversationTurn[];
  selectedOpenQuestion: string | null;
  suggestedReplies: string[];
}

export type EpilogueConversationAction =
  | { type: "select_open_question"; question: string; id: string }
  | { type: "append_turn"; id: string; role: EpilogueChatRole; content: string; citations?: EpilogueChatCitation[]; notice?: string }
  | { type: "set_suggestions"; replies: string[] }
  | { type: "clear" };

export const EMPTY_EPILOGUE_CONVERSATION: EpilogueConversationState = {
  turns: [],
  selectedOpenQuestion: null,
  suggestedReplies: [],
};

export function trimEpilogueDraft(value: string) {
  return value.trim();
}

export function epilogueConversationReducer(
  state: EpilogueConversationState,
  action: EpilogueConversationAction,
): EpilogueConversationState {
  switch (action.type) {
    case "select_open_question": {
      const question = trimEpilogueDraft(action.question);
      if (!question || state.turns.some((turn) => turn.role === "user")) return state;
      return {
        turns: [{
          id: action.id,
          role: "assistant",
          content: `如果你愿意，我们可以从这则题笺继续：${question}`,
          openingQuestion: true,
        }],
        selectedOpenQuestion: question,
        suggestedReplies: [],
      };
    }
    case "append_turn": {
      const content = trimEpilogueDraft(action.content);
      if (!content) return state;
      return {
        ...state,
        turns: [...state.turns, {
          id: action.id,
          role: action.role,
          content,
          citations: action.citations,
          notice: action.notice,
        }],
      };
    }
    case "set_suggestions":
      return { ...state, suggestedReplies: action.replies };
    case "clear":
      return EMPTY_EPILOGUE_CONVERSATION;
    default:
      return state;
  }
}

export function toEpilogueChatHistory(
  turns: EpilogueConversationTurn[],
): EpilogueChatTurn[] {
  // The selected curatorial prompt is sent through `openQuestion`, not as a
  // synthetic assistant turn. Server history therefore remains completed
  // user/assistant pairs and cannot begin on the assistant role.
  return turns
    .filter((turn) => !turn.openingQuestion)
    .map(({ role, content }) => ({ role, content }))
    .slice(-8);
}
