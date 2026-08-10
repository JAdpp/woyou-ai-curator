import type {
  CuratorialBrief,
  CuratorialObjectDecision,
  ExhibitionItem,
} from "./types";

export const CURATORIAL_CONFIDENCE_LABELS = {
  supported: "有馆藏记录可追溯",
  provisional: "暂定解释",
  uncertain: "证据仍不足",
} as const;

export const PROVENANCE_STATUS_LABELS = {
  not_reviewed: "来源史尚未审核",
  unknown: "当前馆藏记录不足以判断来源史状态",
  partial: "来源史记录不完整",
  documented: "馆方记录可查，仍待人工核对",
} as const;

export const EVALUATION_METHOD_LABELS = {
  visitor_prompt: "访客讨论",
  comprehension_check: "理解测试",
  expert_review: "专家审核",
} as const;

export const CULTURAL_SENSITIVITY_STATUS_LABELS = {
  not_reviewed: "文化敏感性尚未审核",
  unknown: "当前记录不足以判断文化敏感性",
  no_flags_after_review: "复核后未记录文化敏感性标记",
  flags_present: "存在文化敏感性标记",
} as const;

export const COMMUNITY_REVIEW_STATUS_LABELS = {
  not_assessed: "相关社群审阅需求尚未评估",
  not_required: "记录为不需要相关社群审阅",
  required: "需要相关社群审阅，尚未记录完成",
  completed: "已记录完成相关社群审阅",
} as const;

export interface CuratorialEvidenceReference {
  id: string;
  itemId: string;
  objectId: string;
  objectTitle: string;
  sourceTitle: string;
  sourceUrl: string;
  sourceLocation: string;
}

/** Resolve claim citations without implying that an ID proves its claim. */
export function buildCuratorialEvidenceIndex(items: ExhibitionItem[]) {
  const references: Record<string, CuratorialEvidenceReference> = {};
  for (const item of items) {
    for (const evidence of item.object.evidence) {
      references[evidence.id] = {
        id: evidence.id,
        itemId: item.id,
        objectId: item.object.id,
        objectTitle: item.displayTitle || item.object.titleOriginal || item.object.title,
        sourceTitle: evidence.sourceTitle,
        sourceUrl: evidence.sourceUrl,
        sourceLocation: evidence.sourceLocation,
      };
    }
  }
  return references;
}

export function findCuratorialDecision(
  brief: CuratorialBrief | null | undefined,
  item: ExhibitionItem,
): CuratorialObjectDecision | undefined {
  return brief?.objects.find(
    (decision) => decision.itemId === item.id || decision.objectId === item.object.id,
  );
}

export function unresolvedEthicsMessages(brief: CuratorialBrief) {
  const messages: string[] = [];
  if (brief.ethics.provenanceStatus !== "documented") {
    messages.push(PROVENANCE_STATUS_LABELS[brief.ethics.provenanceStatus]);
  }
  if (brief.ethics.culturalSensitivity.length === 0) {
    if (brief.ethics.culturalSensitivityStatus === "no_flags_after_review") {
      messages.push("复核记录未发现文化敏感性标记");
    } else {
      messages.push("未记录文化敏感性标记；这不等于已经排除风险");
    }
  }
  if (brief.ethics.communityReviewStatus === "completed") {
    messages.push("已记录完成相关社群审阅；请结合审阅说明核对");
  } else if (brief.ethics.communityReviewRequired || brief.ethics.communityReviewStatus === "required") {
    messages.push("系统标记为需要相关社群审阅，尚未记录完成");
  } else if (brief.ethics.communityReviewStatus === "not_required") {
    messages.push("记录为不需要相关社群审阅");
  } else {
    messages.push("相关社群审阅需求尚未评估");
  }
  return messages;
}
