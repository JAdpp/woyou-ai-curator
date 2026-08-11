import type { Language } from "./i18n";
import type {
  CuratorialBrief,
  CuratorialObjectDecision,
  ExhibitionItem,
} from "./types";

/* Ethics and confidence vocabulary.
   These read as verdicts, so the English is written to be as guarded as the
   Chinese: "recorded as" and "not yet" rather than anything that would sound
   like a clearance the record does not actually give. */

type Bilingual = Record<Language, string>;

const bi = (zh: string, en: string): Bilingual => ({ zh, en });

export const CURATORIAL_CONFIDENCE_LABELS = {
  supported: bi("有馆藏记录可追溯", "Traceable to a collection record"),
  provisional: bi("暂定解释", "Provisional reading"),
  uncertain: bi("证据仍不足", "Evidence still insufficient"),
} as const;

export const PROVENANCE_STATUS_LABELS = {
  not_reviewed: bi("来源史尚未审核", "Provenance not yet reviewed"),
  unknown: bi(
    "当前馆藏记录不足以判断来源史状态",
    "The collection record is not enough to judge provenance status",
  ),
  partial: bi("来源史记录不完整", "Provenance record incomplete"),
  documented: bi(
    "馆方记录可查，仍待人工核对",
    "Documented by the institution; still awaiting human verification",
  ),
} as const;

export const EVALUATION_METHOD_LABELS = {
  visitor_prompt: bi("访客讨论", "Visitor discussion"),
  comprehension_check: bi("理解测试", "Comprehension check"),
  expert_review: bi("专家审核", "Expert review"),
} as const;

export const CULTURAL_SENSITIVITY_STATUS_LABELS = {
  not_reviewed: bi("文化敏感性尚未审核", "Cultural sensitivity not yet reviewed"),
  unknown: bi(
    "当前记录不足以判断文化敏感性",
    "The record is not enough to judge cultural sensitivity",
  ),
  no_flags_after_review: bi(
    "复核后未记录文化敏感性标记",
    "No cultural-sensitivity flags recorded after review",
  ),
  flags_present: bi("存在文化敏感性标记", "Cultural-sensitivity flags present"),
} as const;

export const COMMUNITY_REVIEW_STATUS_LABELS = {
  not_assessed: bi(
    "相关社群审阅需求尚未评估",
    "Whether community review is needed has not been assessed",
  ),
  not_required: bi(
    "记录为不需要相关社群审阅",
    "Recorded as not requiring community review",
  ),
  required: bi(
    "需要相关社群审阅，尚未记录完成",
    "Community review is required and is not recorded as complete",
  ),
  completed: bi(
    "已记录完成相关社群审阅",
    "Community review is recorded as complete",
  ),
} as const;

/** Read one bilingual label. */
export function label(entry: Bilingual, language: Language): string {
  return entry[language];
}

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

export function unresolvedEthicsMessages(
  brief: CuratorialBrief,
  language: Language = "zh",
) {
  const messages: string[] = [];
  const say = (zh: string, en: string) => messages.push(language === "en" ? en : zh);

  if (brief.ethics.provenanceStatus !== "documented") {
    messages.push(PROVENANCE_STATUS_LABELS[brief.ethics.provenanceStatus][language]);
  }
  if (brief.ethics.culturalSensitivity.length === 0) {
    if (brief.ethics.culturalSensitivityStatus === "no_flags_after_review") {
      say(
        "复核记录未发现文化敏感性标记",
        "The review record found no cultural-sensitivity flags",
      );
    } else {
      say(
        "未记录文化敏感性标记；这不等于已经排除风险",
        "No cultural-sensitivity flags are recorded, which is not the same as having ruled any out",
      );
    }
  }
  if (brief.ethics.communityReviewStatus === "completed") {
    say(
      "已记录完成相关社群审阅；请结合审阅说明核对",
      "Community review is recorded as complete; read it together with the review notes",
    );
  } else if (brief.ethics.communityReviewRequired || brief.ethics.communityReviewStatus === "required") {
    say(
      "系统标记为需要相关社群审阅，尚未记录完成",
      "Flagged as requiring community review, which is not recorded as complete",
    );
  } else if (brief.ethics.communityReviewStatus === "not_required") {
    say("记录为不需要相关社群审阅", "Recorded as not requiring community review");
  } else {
    say(
      "相关社群审阅需求尚未评估",
      "Whether community review is needed has not been assessed",
    );
  }
  return messages;
}
