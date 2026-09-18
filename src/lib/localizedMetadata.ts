import type { ExhibitionItem } from "./types";

const CJK = /[\u3400-\u9fff]/;
const LATIN = /[A-Za-z]/;

function safeChinese(value?: string | null): string {
  const text = value?.trim() ?? "";
  return text && CJK.test(text) && !LATIN.test(text) ? text : "";
}

export function publicObjectTitle(item: ExhibitionItem, language: string): string {
  if (!language.startsWith("zh")) {
    return item.displayTitle || item.object.titleOriginal || item.object.title;
  }
  // A heading has to name the object. Without a checked Chinese title the
  // institution's own title is the honest fallback; a placeholder such as
  // "这件展品" named nothing.
  return (
    safeChinese(item.displayTitle) ||
    safeChinese(item.object.titleOriginal) ||
    item.object.titleOriginal ||
    item.object.title
  );
}

// The curatorial roles ("核心证据", "对照／其他声音") are how the pipeline
// arranges an argument, not words a visitor uses. The tag above a label says
// what the stop is for in plain terms; the internal label stays in the record.
const ROLE_TAGS: Record<ExhibitionItem["role"], { zh: string; en: string }> = {
  opening: { zh: "开场", en: "Opening" },
  context: { zh: "背景", en: "Background" },
  core_evidence: { zh: "重点", en: "Key piece" },
  contrast: { zh: "换个角度", en: "Another angle" },
  synthesis: { zh: "收尾", en: "Closing" },
};

export function visitorRoleTag(item: ExhibitionItem, language: string): string {
  const tag = ROLE_TAGS[item.role];
  if (!tag) return item.roleLabel;
  return language.startsWith("zh") ? tag.zh : tag.en;
}

/** What a Chinese voice says for the object: never an English title read aloud. */
export function spokenObjectTitle(item: ExhibitionItem, language: string): string {
  if (!language.startsWith("zh")) return publicObjectTitle(item, language);
  return safeChinese(item.displayTitle) || safeChinese(item.object.titleOriginal) || "这件展品";
}

export function publicObjectMetadata(item: ExhibitionItem, language: string): string[] {
  if (!language.startsWith("zh")) {
    return [
      item.object.maker || item.object.creator,
      item.object.date,
      item.object.medium,
      item.object.culture,
    ].filter(
      (value): value is string => Boolean(value),
    );
  }
  const localized = item.localizedMetadata;
  return [localized?.creator, localized?.date, localized?.medium, localized?.culture]
    .map(safeChinese)
    .filter(Boolean);
}

export function publicInstitutionName(item: ExhibitionItem, language: string): string {
  if (!language.startsWith("zh")) return item.object.institution ?? "";
  return safeChinese(item.localizedMetadata?.institution);
}
