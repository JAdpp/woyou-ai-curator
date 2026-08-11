"use client";

import { useCallback, useLayoutEffect, useSyncExternalStore } from "react";
import {
  applyLanguage,
  copy,
  DEFAULT_LANGUAGE,
  readStoredLanguage,
  type Language,
} from "./i18n";

/**
 * The client half of the language switch.
 *
 * The choice is one setting for the whole document, so it is held in one store
 * rather than in each caller's `useState`. That mattered the moment the switch
 * moved into the site header: with per-component state, clicking it re-rendered
 * the header into English and left the page under it in Chinese.
 *
 * `applyLanguage` still writes `<html data-lang>` and `localStorage`; this only
 * adds the subscription that tells every reader the value changed.
 */
let current: Language = DEFAULT_LANGUAGE;
const listeners = new Set<() => void>();

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function getSnapshot(): Language {
  return current;
}

function getServerSnapshot(): Language {
  // The server has no stored preference; the boot script and the layout both
  // start from the default, so agreeing with them here avoids a mismatch.
  return DEFAULT_LANGUAGE;
}

function setStoredLanguage(next: Language): void {
  if (current === next) return;
  current = next;
  for (const listener of listeners) listener();
}

export function useLanguage() {
  const language = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  useLayoutEffect(() => {
    // React's development remount strips attributes it does not own from
    // <html>, clearing what the boot script set. Re-applying is a no-op in
    // production and keeps development honest.
    const stored = readStoredLanguage();
    setStoredLanguage(stored);
    applyLanguage(stored);
  }, []);

  const choose = useCallback((next: Language) => {
    setStoredLanguage(next);
    applyLanguage(next);
  }, []);

  return { language, setLanguage: choose, t: copy(language) };
}
