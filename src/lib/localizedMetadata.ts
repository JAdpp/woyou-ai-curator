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
