/* No "use client": the root layout is a Server Component and needs the boot
   script constant from here. The hook that owns the state lives in
   `useLanguage.ts`, which is the client half. */

/**
 * Site language.
 *
 * Deliberately not Next's locale routing. That approach puts the locale in the
 * path and redirects on Accept-Language, which would change every URL the demo
 * has already been handed out on, for a two-language toggle that defaults to
 * Chinese. This is client state instead, persisted and applied before paint by
 * the inline script in the root layout (see the "preventing flash before
 * hydration" guide: the lazy initialiser below reads the same source the script
 * does, so React's first render matches the DOM).
 */
export type Language = "zh" | "en";

export const LANGUAGE_STORAGE_KEY = "woyou.language";
export const DEFAULT_LANGUAGE: Language = "zh";

/** Runs in the browser before React; kept in sync with `readStoredLanguage`. */
export const LANGUAGE_BOOT_SCRIPT =
  `(function(){try{var l=localStorage.getItem(${JSON.stringify(LANGUAGE_STORAGE_KEY)});` +
  `if(l==="en"||l==="zh"){document.documentElement.setAttribute("data-lang",l);` +
  `document.documentElement.lang=(l==="en"?"en":"zh-CN")}}catch(e){}})()`;

export function readStoredLanguage(): Language {
  if (typeof window === "undefined") return DEFAULT_LANGUAGE;
  try {
    return window.localStorage.getItem(LANGUAGE_STORAGE_KEY) === "en" ? "en" : DEFAULT_LANGUAGE;
  } catch {
    // Blocked storage is not a reason to fail; fall back to the default.
    return DEFAULT_LANGUAGE;
  }
}

/** Apply a chosen language to the document and remember it. */
export function applyLanguage(next: Language): void {
  document.documentElement.setAttribute("data-lang", next);
  document.documentElement.lang = next === "en" ? "en" : "zh-CN";
  try {
    window.localStorage.setItem(LANGUAGE_STORAGE_KEY, next);
  } catch {
    // Losing the preference is survivable; blocking the switch is not.
  }
}

type Entry = { zh: string; en: string };

function line(zh: string, en: string): Entry {
  return { zh, en };
}

/**
 * Visitor-facing copy.
 *
 * The admin dashboard is not here on purpose: it is blocked at nginx and is not
 * a visitor surface, so translating it would be work nobody reads.
 */
const COPY = {
  brand: {
    productName: line("卧游", "Woyou"),
    productNameLatin: line("Woyou", "卧游"),
    curatorName: line("彦远", "Yanyuan"),
    curatorTitle: line("AI 策展人彦远", "Yanyuan, the AI curator"),
    curatorRole: line("AI 策展人", "AI curator"),
    tagline: line(
      "和 AI 策展人彦远聊聊，把问题变成展览",
      "Talk to Yanyuan, and turn your question into an exhibition",
    ),
    collectionScope: line(
      "涵盖全球多种文化的开放馆藏",
      "open collections spanning many cultures",
    ),
  },
  header: {
    home: line("首页", "Home"),
    howItWorks: line("怎么使用", "How it works"),
    sources: line("馆藏来源", "Collections"),
    internalTool: line("内部工具", "Internal"),
    nav: line("主导航", "Main navigation"),
    languageGroup: line("语言 / Language", "语言 / Language"),
  },
  chat: {
    heading: line("先说说你想了解什么", "Tell me what you'd like to look into"),
    ariaLabel: line("与 AI 策展人对话", "Conversation with the AI curator"),
    connecting: line("正在接通…", "Connecting…"),
    you: line("你", "You"),
    skipped: line("（跳过）", "(skipped)"),
    answerGroup: line("可选回答", "Answer options"),
    confirmCount: line("确认 {n} 项", "Confirm {n}"),
    confirmNone: line("都可以", "Anything is fine"),
    freeTextLabel: line("自由输入", "Free text"),
    freeTextPlaceholder: line("直接说说你想弄懂什么…", "Just say what you want to understand…"),
    send: line("发送", "Send"),
    skipQuestion: line("跳过这题", "Skip this one"),
    skipInterview: line("跳过访谈，用默认设置", "Skip the interview, use defaults"),
    stepOf: line("第 {step} 步，共 {total} 步", "Step {step} of {total}"),
    errorTitle: line("这一步没有记录下来", "That step wasn't recorded"),
    errorGeneric: line("这一步没有记录下来，请重试。", "That step wasn't recorded. Please try again."),
    startFailed: line("彦远没能接上话，请刷新重试。", "Yanyuan couldn't pick up. Please refresh and retry."),
    privacyTitle: line("关于这次对话", "About this conversation"),
    privacyBody: line(
      "访谈内容只用于这次策展。启用云模型时，你输入的文字会发送给已配置的模型服务；请不要填写个人敏感信息。",
      "What you say here is used only to curate this visit. With the cloud model enabled, your text is sent to the configured model service — please don't enter anything sensitive.",
    ),
  },
  pipeline: {
    ariaLabel: line("策展生成进度", "Curation progress"),
    heading: line("彦远正在为你的问题策展", "Yanyuan is curating for your question"),
    starting: line("正在启动…", "Starting…"),
    failed: line("策展没有完成，请重试。", "The curation didn't finish. Please try again."),
    notStarted: line("策展没有启动，请重试。", "The curation didn't start. Please try again."),
    defaultFailed: line("没能启动默认策展，请重试。", "Couldn't start the default curation. Please try again."),
  },
  exhibition: {
    languageNoticeTitle: line("语言提示", "A note on language"),
    // Shown when the reader's language differs from the language the exhibition
    // was written in. Labels are generated once and stored, so they cannot be
    // switched after the fact without regenerating the exhibition.
    languageNoticeBody: line(
      "这场展览是用英文策展与撰写的，展签与策展词保持原文；界面已切换为中文。要得到中文展签，请用中文重新策展一次。",
      "This exhibition was curated and written in Chinese, and its labels and curatorial text stay in the language they were written in. The interface is in English. For English labels, curate a new visit in English.",
    ),
  },
} as const;

export type Copy = typeof COPY;

/** Resolve the whole catalogue for one language. */
export function copy(language: Language) {
  const resolve = <T extends Record<string, Entry>>(section: T) =>
    Object.fromEntries(
      Object.entries(section).map(([key, entry]) => [key, entry[language]]),
    ) as { [K in keyof T]: string };

  return {
    brand: resolve(COPY.brand),
    header: resolve(COPY.header),
    chat: resolve(COPY.chat),
    pipeline: resolve(COPY.pipeline),
    exhibition: resolve(COPY.exhibition),
  };
}

/** Fill `{name}` placeholders. */
export function fill(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (match, key) =>
    key in values ? String(values[key]) : match,
  );
}
