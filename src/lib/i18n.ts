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
  puzzle: {
    licenceFallback: line("许可见机构记录", "Licence per the institutional record"),
    unavailableLabel: line("馆藏拼图暂不可用", "Collection puzzle unavailable"),
    unavailableTitle: line("馆藏拼图暂时没有接上", "The collection puzzle didn't connect"),
    unavailableBody: line(
      "彦远仍在继续策展，这不会影响展览生成。",
      "Yanyuan is still curating; this doesn't affect the exhibition.",
    ),
    heading: line("等待时，拼一件馆藏", "While you wait, piece one back together"),
    instructions: line(
      "点击空格旁的图块，或聚焦拼图后使用方向键。",
      "Click a tile next to the gap, or focus the puzzle and use the arrow keys.",
    ),
    done: line("完成", "Done"),
    moves: line("{n} 步", "{n} moves"),
    boardLabel: line(
      "三乘三馆藏图像滑块拼图；方向键移动空格",
      "Three-by-three sliding puzzle of a collection image; arrow keys move the gap",
    ),
    tileLabel: line("拼图第 {n} 片", "Puzzle tile {n}"),
    tileMovable: line("，可移入空格", ", can slide into the gap"),
    loadingImage: line("正在取一件馆藏图像…", "Fetching a collection image…"),
    imageFailed: line(
      "这张图没有载入，可换一件继续。",
      "That image didn't load; try another one.",
    ),
    solved: line("拼好了。彦远仍在整理展线。", "Solved. Yanyuan is still laying out the route."),
    credit: line(
      "馆藏图像 · {licence}。拼图是独立的等待互动；策展仍依据访谈与馆藏资料进行。",
      "Collection image · {licence}. The puzzle is just something to do while waiting; the curation still runs on the interview and the collection records.",
    ),
    reset: line("重置拼图", "Reset puzzle"),
    swap: line("换一件藏品", "Another object"),
  },
  epilogue: {
    localFallback: line(
      "当前回应由本展馆藏材料的本地回退生成，未调用实时模型。",
      "This reply came from a local fallback over this exhibition's own material; no live model was called.",
    ),
    sendFailed: line(
      "彦远暂时没能接上这句话。你可以重试，或稍后再继续。",
      "Yanyuan couldn't pick that up. Try again, or come back to it later.",
    ),
    needText: line("写下一点想法后再发送。", "Write something before sending."),
    folioMark: line("题笺", "Note"),
    kicker: line("可选讨论 · AI 策展人彦远", "Optional discussion · Yanyuan, the AI curator"),
    skippedTitle: line("题笺替你留在这里", "The notes will stay here for you"),
    openTitle: line("如果你还想把一个念头说完", "If you'd like to finish a thought"),
    skippedBody: line(
      "不用现在回答；想回来时，彦远仍会从这场展览接着聊。",
      "No need to answer now; when you come back, Yanyuan will pick up from this exhibition.",
    ),
    openBody: line(
      "彦远留下了 {n} 则展后题笺，也听你自由发问。",
      "Yanyuan left {n} closing note(s), and will take any question of your own.",
    ),
    reopen: line("重新打开题笺", "Reopen the notes"),
    start: line("与彦远聊一会儿", "Talk with Yanyuan"),
    skipNow: line("先跳过", "Skip for now"),
    sectionLabel: line("展后题笺", "Closing notes"),
    liveTitle: line("把展览带出展厅", "Take the exhibition out of the hall"),
    clear: line("清空", "Clear"),
    collapse: line("收起", "Collapse"),
    noScoring: line(
      "这里没有评分，也没有标准答案。你可以回应一则题笺，提出异议，或从自己的经验谈起。",
      "Nothing here is scored and there is no right answer. Reply to a note, disagree with it, or start from your own experience.",
    ),
    emptyNote: line(
      "彦远在等你选择一则题笺，也可以直接写下自己的问题。",
      "Yanyuan is waiting for you to pick a note, or to write a question of your own.",
    ),
    curator: line("彦远", "Yanyuan"),
    visitor: line("访客", "Visitor"),
    citationsSummary: line("与本展材料的关联", "How this ties to the exhibition's material"),
    citationFallback: line("{n} 条馆藏依据", "{n} collection reference(s)"),
    thinking: line("正在回看这场展览里的线索……", "Looking back through this exhibition…"),
    changeNote: line("换一则题笺", "Pick another note"),
    pickNote: line("从一则题笺开始", "Start from a note"),
    suggestionsLabel: line("可以继续谈的方向", "Ways to carry this on"),
    suggestionsLead: line("可以接着谈", "You could go on with"),
    retry: line("重试", "Retry"),
    inputLabel: line("写给彦远", "Write to Yanyuan"),
    placeholderWithNote: line("写下你的想法……", "Write what you think…"),
    placeholderOpen: line(
      "你注意到了什么？也可以提出不同看法。",
      "What did you notice? A different view is welcome too.",
    ),
    privacy: line(
      "访客发言会发送给模型生成回应，但不写入展览记录或统计事件。",
      "What you write is sent to the model to compose a reply, but is not written into the exhibition record or the analytics events.",
    ),
    sending: line("彦远正在回应", "Yanyuan is replying"),
    send: line("发送", "Send"),
    optOut: line("暂不参与这次讨论", "Not this time"),
  },
  view: {
    sentenceInstitutionFact: line("机构记录", "Institutional record"),
    sentenceSystemInference: line("系统推断", "System inference"),
    sentenceUncertain: line("仍不确定", "Still uncertain"),
    sourceMetadata: line("机构元数据", "Institutional metadata"),
    sourceCuratorialText: line("机构说明", "Institutional description"),
    sourceProvenance: line("机构来源记录", "Institutional provenance record"),
    sourceFieldFallback: line("字段来源", "Field source"),
    noEvidence: line("尚未绑定可定位的馆藏记录", "No locatable collection record attached yet"),
    linkedRecords: line("关联的馆藏记录", "Linked collection records"),
    unresolvedRecords: line("{n} 条记录暂无法定位", "{n} record(s) cannot be located"),
    briefLabel: line("策展依据", "Curatorial basis"),
    briefAction: line("查看论证、选物与边界", "See the argument, the selection and its limits"),
    briefDisclosure: line(
      "以下是彦远根据本展馆藏记录形成的策展判断，不是来源机构原话。引用表示可以返回相关记录核对，不代表该判断已经由机构或专家确认。",
      "What follows is Yanyuan's curatorial judgement, formed from this exhibition's collection records. It is not the wording of the holding institutions. A citation means you can go back to the record and check; it does not mean the judgement has been confirmed by an institution or a specialist.",
    ),
    briefArgument: line("论证线索", "The argument"),
    briefSelection: line("选物边界", "Selection and its limits"),
    briefSelectionCount: line(
      "本展记录了 {n} 件入选判断",
      "This exhibition records {n} selection decision(s)",
    ),
    briefExcludedSuffix: line(
      "；另有 {n} 件候选未进入最终动线。",
      "; {n} further candidate(s) did not enter the final route.",
    ),
    briefNoExcluded: line(
      "。公开版本不包含访客画像与排除候选记录。",
      ". The public version omits the visitor profile and the excluded-candidate record.",
    ),
    briefEthics: line("来源与伦理状态", "Provenance and ethics status"),
    briefOpen: line("这份策展仍留下的问题", "What this curation leaves open"),
    briefModelRefined: line("模型整理，规则校验", "Model-drafted, rule-validated"),
    briefRuleGenerated: line("规则生成", "Rule-generated"),
    briefRetrieval: line("检索", "Retrieval"),
    briefExternalKnowledge: line("外部知识", "External knowledge"),
    briefExternalAllowed: line("允许", "allowed"),
    briefExternalUnused: line("未使用", "not used"),
    image: line("图片", "Image"),
    institutionPage: line("机构页", "Institution page"),
    tombstoneOnly: line("仅著录信息", "Catalogue data only"),
    whySelected: line("为什么选它？", "Why this object?"),
    systemJudgement: line("系统策展判断", "System curatorial judgement"),
    relationPrefix: line("与前后展品的关系：", "Relation to the objects before and after: "),
    collapseSources: line("收起来源", "Hide sources"),
    expandSources: line("来源（{n} 条）", "Sources ({n})"),
    baseMetadata: line("基础元数据", "Core metadata"),
    perFieldSeeRecord: line("逐字段见机构记录", "Per field, see the institutional record"),
    institutionDescription: line("机构描述", "Institutional description"),
    legacyRights: line("旧版综合声明", "Legacy combined rights statement"),
    posterAlt: line("AI 生成的展览视觉", "AI-generated exhibition visual"),
    posterCollageAlt: line(
      "本展所选馆藏公开图像拼贴，用作入口视觉回退",
      "Collage of this exhibition's open collection images, used as the entrance visual fallback",
    ),
    posterUnavailable: line("主题画面暂不可用", "The key image is unavailable"),
    posterCaptionAi: line(
      "AI 生成展览海报；主题画面由模型生成，标题由系统精确排版",
      "AI-generated exhibition poster: the key image comes from a model, the title is set by the system",
    ),
    posterCaptionCollage: line(
      "馆藏公开图像拼贴与系统排版回退；图像许可见各展品来源",
      "A collage of open collection images with system typesetting; image licences are listed with each object",
    ),
    skipToContent: line("跳到展览内容", "Skip to the exhibition"),
    eyebrow: line("彦远为你策展 · 文字版", "Curated for you by Yanyuan · text version"),
    enterHall: line("进入 3D 展厅", "Enter the 3D hall"),
    curateAgain: line("重新策展", "Curate again"),
    noWebgl: line(
      "当前设备不支持 3D 展厅（缺少 WebGL 或显卡能力不足），已为你打开完整的文字版本。",
      "This device can't run the 3D hall (no WebGL, or the graphics capability is insufficient), so the full text version is open instead.",
    ),
    toc: line("展览目录", "Contents"),
    epilogue: line("结语", "In closing"),
    chapterOf: line("第 {n} 部分 / 共 {total}", "Part {n} of {total}"),
    materialLimits: line("这场展览的材料边界", "What this exhibition's material cannot do"),
    generationRecord: line("生成记录", "Generation record"),
    actualPath: line("实际路径", "Actual path"),
    configuredModel: line("配置模型", "Configured model"),
    promptVersion: line("提示", "Prompt"),
    collection: line("馆藏", "Collection"),
    validator: line("检查器", "Validator"),
    footer: line(
      "本展由彦远生成。展品与说明可追溯到机构公开馆藏；策展解释不代表来源机构立场。",
      "This exhibition was generated by Yanyuan. Objects and their catalogue data trace back to open institutional collections; the curatorial reading is not the position of the holding institutions.",
    ),
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
    puzzle: resolve(COPY.puzzle),
    epilogue: resolve(COPY.epilogue),
    view: resolve(COPY.view),
    exhibition: resolve(COPY.exhibition),
  };
}

/** Fill `{name}` placeholders. */
export function fill(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (match, key) =>
    key in values ? String(values[key]) : match,
  );
}
