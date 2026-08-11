"use client";

import { useCallback, useLayoutEffect, useState } from "react";
import {
  applyLanguage,
  copy,
  readStoredLanguage,
  type Language,
} from "./i18n";

/**
 * The client half of the language switch.
 *
 * The initial state comes from a lazy initialiser reading the same
 * `localStorage` key the root layout's boot script reads, so React's first
 * render already agrees with the DOM the script produced — no hydration
 * mismatch, and no flash of the wrong language.
 */
export function useLanguage() {
  const [language, setLanguage] = useState<Language>(readStoredLanguage);

  useLayoutEffect(() => {
    // React's development remount strips attributes it does not own from
    // <html>, clearing what the boot script set. Re-applying is a no-op in
    // production and keeps development honest.
    const stored = readStoredLanguage();
    setLanguage(stored);
    applyLanguage(stored);
  }, []);

  const choose = useCallback((next: Language) => {
    setLanguage(next);
    applyLanguage(next);
  }, []);

  return { language, setLanguage: choose, t: copy(language) };
}
