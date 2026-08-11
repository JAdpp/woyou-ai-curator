"use client";

import type { Language } from "@/lib/i18n";
import styles from "./siteHeader.module.css";

/**
 * 中 / EN, defaulting to Chinese.
 *
 * Switching changes the interface only. An exhibition's labels and curatorial
 * prose are written once, at generation time, in the language the visit was
 * curated in; see the notice in the exhibition view.
 */
export function LanguageToggle({
  language,
  onChange,
  label,
}: {
  language: Language;
  onChange: (next: Language) => void;
  label: string;
}) {
  return (
    <div className={styles.languageSwitch} role="group" aria-label={label}>
      <button
        type="button"
        aria-pressed={language === "zh"}
        onClick={() => onChange("zh")}
      >
        中文
      </button>
      <button
        type="button"
        aria-pressed={language === "en"}
        onClick={() => onChange("en")}
      >
        EN
      </button>
    </div>
  );
}
