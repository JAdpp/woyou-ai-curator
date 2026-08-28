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
    askDirectly: line("直接告诉彦远你的问题", "Ask Yanyuan your question directly"),
    browseSuggestions: line("还没想好？看看馆藏可以展开的方向", "Need inspiration? Browse directions the collection can support"),
    confirmCount: line("确认 {n} 项", "Confirm {n}"),
    confirmNone: line("都可以", "Anything is fine"),
    freeTextLabel: line("自由输入", "Free text"),
    freeTextPlaceholder: line("直接说说你想弄懂什么…", "Just say what you want to understand…"),
    send: line("发送", "Send"),
    skipQuestion: line("跳过这题", "Skip this one"),
    skipInterview: line("跳过访谈，用默认设置", "Skip the interview, use defaults"),
    continueWithAnswers: line("用已有回答开始策展", "Curate with my answers so far"),
    stepOf: line("第 {step} 步，共 {total} 步", "Step {step} of {total}"),
    errorTitle: line("这一步没有记录下来", "That step wasn't recorded"),
    errorGeneric: line("这一步没有记录下来，请重试。", "That step wasn't recorded. Please try again."),
    startFailed: line("彦远没能接上话，请刷新重试。", "Yanyuan couldn't pick up. Please refresh and retry."),
    thinking: line("彦远正在理解你的回答…", "Yanyuan is considering your answer…"),
    languageLocked: line("中文访谈", "English interview"),
    languageLockedHint: line(
      "策展文字会沿用访谈开始时的语言。如需切换，请刷新页面，在首页切换后重新开始。",
      "The exhibition will use the language selected when this interview began. To switch, refresh and choose a language on the home page before starting again.",
    ),
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
    progressLabel: line("策展进度", "Curation progress"),
  },
  visitBrief: {
    ariaLabel: line("确认本次参观设定", "Confirm this visit"),
    eyebrow: line("开始策展前", "Before curation begins"),
    heading: line("看看彦远理解得对不对", "Check what Yanyuan understood"),
    intro: line(
      "这些设定会决定展览的问题、篇幅和讲述方式。若有偏差，可以重新调整回答。",
      "These choices shape the exhibition's question, length, and voice. If anything is off, you can revise your answers.",
    ),
    inquiry: line("你想弄懂", "Your inquiry"),
    direction: line("馆藏方向", "Collection direction"),
    priorKnowledge: line("了解程度", "Familiarity"),
    duration: line("参观时长", "Visit length"),
    exclusions: line("希望避开", "Topics to avoid"),
    noExclusions: line("没有特别需要避开的内容", "No topics specified"),
    minutes: line("{n} 分钟", "{n} minutes"),
    knowledgeNone: line("第一次接触", "New to the subject"),
    knowledgeSome: line("了解一些", "Some familiarity"),
    knowledgeFamiliar: line("比较熟悉", "Already familiar"),
    confirm: line("开始策展", "Begin curation"),
    revise: line("修改回答（重新访谈）", "Revise answers (restart interview)"),
  },
  recovery: {
    ariaLabel: line("策展失败后的恢复选项", "Curation recovery options"),
    eyebrow: line("这次没有生成完成", "This curation did not finish"),
    heading: line("你的问题和参观设定都还在", "Your inquiry and visit settings are safe"),
    body: line(
      "彦远暂时没能完成展览。你可以直接重试，不必再回答一遍；也可以调整问题后重新开始。",
      "Yanyuan couldn't finish the exhibition this time. You can retry without repeating the interview, or revise your inquiry and start again.",
    ),
    retained: line("已保留本次访谈", "This interview has been retained"),
    progressRetained: line(
      "上次已完成 {done} / {total} 个步骤",
      "The previous attempt completed {done} of {total} steps",
    ),
    retry: line("重试本次策展", "Retry this curation"),
    revise: line("调整问题，重新访谈", "Revise the inquiry and interview"),
    home: line("返回首页", "Return home"),
  },
  landing: {
    microcopy: line(
      "无需注册 · 访谈可整段跳过 · 同时支持 3D 与 2D 网页版",
      "No sign-up · the interview can be skipped entirely · 3D and 2D web versions",
    ),
    caseOneIndex: line("案例一 · 色彩与技术", "Case one · Colour and technique"),
    caseTwoIndex: line("案例二 · 跨文化动物", "Case two · An animal across cultures"),
    mode3dCaption: line(
      "连续展线 · 逐站镜头 · 随时自由行走",
      "One continuous route · stop-by-stop camera · walk freely at any time",
    ),
    mode2dCaption: line(
      "完整展签 · 机构来源 · 键盘与读屏可用",
      "Full labels · institutional sources · keyboard and screen-reader ready",
    ),
    sampleFieldLicence: line(
      "材质字段 CC0 1.0 · 馆方作品说明 CC BY 4.0",
      "Material field CC0 1.0 · institutional description CC BY 4.0",
    ),
    epigraph: line("澄怀观道，卧以游之。", "Clear the mind to see the way; travel it lying down."),
    epigraphCite: line("— 宗炳《画山水序》，五世纪", "— Zong Bing, Preface on Landscape Painting, 5th c."),
    heroTitle: line(
      "把你对文明的好奇，变成一座只为你搭的展厅。",
      "Turn your curiosity about civilisation into an exhibition hall built just for you.",
    ),
    heroBody: line(
      "“AI 策展人彦远”会从涵盖全球多种文化的开放馆藏中检索、比较，并为你组织主题、章节与导览。",
      "Yanyuan, the AI curator, searches and compares across open collections spanning many cultures, then organises a theme, its chapters and a route through them for you.",
    ),
    heroCta: line("和“AI 策展人彦远”聊聊", "Talk to Yanyuan, the AI curator"),
    corpusLive: line(
      "当前接入 {objects} 件开放馆藏，来自 {museums} 家博物馆；涵盖全球多种文化，许可逐字段记录。",
      "Currently {objects} open collection objects from {museums} museums, spanning many cultures, with rights recorded field by field.",
    ),
    corpusLoading: line("正在读取当前冻结馆藏。", "Reading the current frozen collection."),
    corpusOffline: line("馆藏数据暂未连接。", "The collection data is not connected."),

    blueCaseTitle: line(
      "相似的蓝，为什么会出现在中国瓷瓶、越南盘与伊朗陶器上？",
      "Why does a similar blue turn up on a Chinese vase, a Vietnamese plate and an Iranian dish?",
    ),
    blueCaseLead: line(
      "这个案例使用三件真实开放馆藏，展示“彦远”如何从访客问题提出可核查的策展命题。",
      "Three real open-collection objects, showing how Yanyuan turns a visitor's question into a checkable curatorial proposition.",
    ),
    visitorQuestion: line("访客的问题", "The visitor's question"),
    blueCaseQuestion: line(
      "“我只知道青花瓷。蓝色是不是从中国传到世界各地的？”",
      "“All I know is Chinese blue-and-white. Did the blue spread from China to everywhere else?”",
    ),
    blueCaseNote: line(
      "系统先保留疑问，不把访客的猜测直接当成结论。",
      "The system holds the question open rather than treating the guess as a finding.",
    ),
    blueCaseEvidenceLabel: line("蓝色案例中的三件馆藏证据", "Three collection objects in the blue case"),
    caseJudgement: line("案例中的策展判断", "The curatorial judgement"),
    blueCaseJudgement: line(
      "相似的蓝色并不自动证明一条单向传播路线。把色彩与材料著录、胎釉、器形和年代放在一起，才能追问技术如何被不同地区重新制作和使用。",
      "A similar blue does not by itself prove a one-way route of transmission. Only by putting the colour and material records, the body and glaze, the form and the dating side by side can you ask how a technique was remade and reused in different places.",
    ),
    blueCaseFooter: line(
      "卡片中的题名、年代与材料来自三家博物馆的原始著录；联系与解释另行标记。",
      "Titles, dates and materials on these cards come from the three museums' own records; connections and readings are marked separately.",
    ),

    catCaseTitle: line(
      "同样是猫，为什么会走进安第斯陶碗、埃及青铜像与荷兰寓言画？",
      "Why does the cat turn up in an Andean bowl, an Egyptian bronze and a Dutch fable painting?",
    ),
    catCaseLead: line(
      "这个案例来自此前生成的“猫咪的千面形象”展览：共享一个动物主题，不等于共享一种象征意义。",
      "From an exhibition generated earlier, “The Many Faces of the Cat”: sharing an animal does not mean sharing a symbolism.",
    ),
    catCaseQuestion: line(
      "“各个文化里都有猫，它们是不是都代表神秘和好运？”",
      "“Cats appear in every culture — do they all stand for mystery and good luck?”",
    ),
    catCaseNote: line(
      "系统先检索“猫”本身，再比较器物用途、材料、时代与馆方说明。",
      "The system retrieves the cat itself first, then compares each object's use, material, period and institutional description.",
    ),
    catCaseEvidenceLabel: line("猫主题案例中的三件馆藏证据", "Three collection objects in the cat case"),
    catCaseJudgement: line(
      "不能用一个“猫的象征意义”覆盖不同文化。猫进入容器装饰、小型金属像与寓言画的方式各不相同；比较应从这些可见差异和机构记录开始。",
      "No single “meaning of the cat” covers these cultures. A cat enters vessel ornament, a small bronze figure and a moral fable in quite different ways; the comparison should start from those visible differences and from the institutional record.",
    ),
    catCaseFooter: line(
      "三件展品的图片、基础元数据与馆方说明均按 CMA 的 CC0 开放记录使用。",
      "Images, core metadata and institutional descriptions for all three are used under the Cleveland Museum of Art's CC0 open records.",
    ),
    swapForMine: line("换成我的好奇", "Use my own curiosity"),

    featureTitle: line("AI策展人如何把问题变成展览", "How the AI curator turns a question into an exhibition"),
    featureLead: line(
      "“彦远”先了解访客，再检索和比较馆藏、安排展线，并为每件展品保留机构来源与字段许可。",
      "Yanyuan gets to know the visitor first, then searches and compares the collection, lays out a route, and keeps each object's institutional source and per-field licence attached.",
    ),
    featureInterviewKicker: line("访谈", "Interview"),
    featureInterviewTitle: line("先弄清你想怎么看", "First, how you want to look"),
    featureInterviewBody: line(
      "兴趣、参观倾向、熟悉程度、可用时间与不想看到的内容都会改变展品数量、解释深度和导览节奏；陌生题目会给出可选方向，也可直接跳过整段访谈。",
      "Your interest, the kind of visit you want, how familiar you are, how long you have and what you would rather avoid all change the number of objects, the depth of explanation and the pace of the route. Unfamiliar subjects come with suggested directions, and the whole interview can be skipped.",
    ),
    featureInterviewDiagram: line(
      "访谈示意：AI 策展人“彦远”询问哪一种好奇最接近今天想看的，访客回答想知道一种颜色怎样穿过不同文化，并选择第一次接触、约十分钟和不同地区。",
      "Interview diagram: Yanyuan asks which curiosity comes closest to today's; the visitor answers that they want to know how one colour travelled across cultures, and picks first-time, about ten minutes, and different regions.",
    ),
    featureInterviewAsk: line("哪一种好奇最接近你今天想看的？", "Which curiosity is closest to what you want today?"),
    featureInterviewReply: line("我想知道一种颜色怎样穿过不同文化。", "I want to know how one colour travelled across cultures."),
    chipFirstTime: line("第一次接触", "First time"),
    chipTenMinutes: line("约 10 分钟", "About 10 min"),
    chipRegions: line("想看不同地区", "Different regions"),

    featureSelectKicker: line("选择与编排", "Selection and sequencing"),
    featureSelectTitle: line("主题按你的问题形成", "The theme forms around your question"),
    featureSelectBody: line(
      "“彦远”会从{scope}馆藏中检索、比较、排除，再形成主题、章节和展品角色。",
      "Yanyuan searches, compares and rules out across {scope} of the collection, then forms the theme, the chapters and each object's role.",
    ),
    featureSelectScopeCount: line("当前 {n} 件", "the current {n} objects"),
    featureSelectScopeUnknown: line("当前可用", "what is currently available"),
    featureSelectDiagram: line(
      "策展进度示意：依次理解问题、寻找证据、组织展线，并搭建采用连续动线的展厅。",
      "Curation diagram: understand the question, find evidence, organise the route, then build a hall with one continuous path.",
    ),
    stepUnderstand: line("理解问题", "Understand the question"),
    stepUnderstandDetail: line("已提取颜色、流动、跨文化", "Extracted: colour, movement, cross-cultural"),
    stepEvidence: line("寻找证据", "Find evidence"),
    stepEvidenceDetail: line("比较年代、材料与产地", "Comparing dates, materials and places of origin"),
    stepRoute: line("组织展线", "Organise the route"),
    stepRouteDetail: line("保留一件反例或限制", "Keeping one counter-example or limit"),
    stepBuild: line("搭建展厅", "Build the hall"),
    stepBuildDetail: line(
      "用连续动线连接章节、导览与主题海报",
      "One continuous path linking chapters, narration and the theme poster",
    ),

    featureVisitKicker: line("参观方式", "Ways to visit"),
    featureVisitTitle: line("展品被放进一条空间叙事", "The objects sit in a spatial narrative"),
    featureVisitBody: line(
      "可开启千问 AI 合成的“彦远”专业讲述，并用镜头逐站推进；想停下来时可以随时切换自由行走，也可直接打开完整 2D 网页版。",
      "You can turn on Yanyuan's studio narration, synthesised by Qwen, and let the camera move stop by stop; switch to walking freely whenever you want to linger, or open the full 2D web version instead.",
    ),
    featureVisitDiagram: line(
      "同一展览的两种参观方式：左侧为带真实馆藏图像和连续动线的 3D 导览，右侧为包含题名、展品图、展签与来源的 2D 网页版。",
      "Two ways through the same exhibition: on the left a 3D guided route with real collection images and one continuous path, on the right a 2D web version with title, images, labels and sources.",
    ),
    mode3d: line("3D 导览", "3D route"),
    mode2d: line("2D 网页版", "2D web version"),
    modeSampleTitle: line("相似的蓝", "A similar blue"),

    featureSourceKicker: line("来源与许可", "Source and licence"),
    featureSourceTitle: line("每件展品都保留来源", "Every object keeps its source"),
    featureSourceBody: line(
      "机构原文、系统转述与策展推断分开显示。题名、年代、材料、权利声明和机构对象页随展品保留，便于核对。",
      "Institutional wording, system paraphrase and curatorial inference are shown apart from one another. Title, date, material, rights statement and the institution's own object page stay with the object so they can be checked.",
    ),
    featureSourceDiagram: line(
      "展签来源面板示意：机构著录与策展解释分开显示，来源为芝加哥艺术博物馆；所示材质字段使用 CC0 1.0，馆方作品说明使用 CC BY 4.0。",
      "Label source panel: institutional record and curatorial reading shown separately, source the Art Institute of Chicago; the material field under CC0 1.0, the institution's description under CC BY 4.0.",
    ),
    institutionalRecord: line("机构著录", "Institutional record"),
    curatorialReading: line("策展解释", "Curatorial reading"),
    featureSourceReadingSample: line(
      "这件器物提供了另一种蓝白装饰的材料路径。",
      "This object offers another material route to blue-and-white decoration.",
    ),
    source: line("来源", "Source"),
    fieldLicence: line("字段许可", "Field licence"),

    methodKicker: line("AGENT 架构 × 策展 HARNESS", "AGENT ARCHITECTURE × CURATORIAL HARNESS"),
    methodTitle: line("一位 AI 策展人，一套受控的策展 Harness", "One AI curator, one controlled curation harness"),
    methodLead: line(
      "“彦远”不是靠一个提示词临场编展。访谈、检索、策展论证、展签与呈现被拆成可观察、可验证、可回退的阶段；模型负责提出解释，系统负责约束它能看什么、能写什么，以及什么结果可以进入展厅。",
      "Yanyuan does not improvise an exhibition from one prompt. Interview, retrieval, curatorial argument, labels and presentation are split into observable, testable and recoverable stages: the model proposes readings, while the system controls what it may see, what it may write and what is allowed into the hall.",
    ),
    agentArchitectureTitle: line("Agent 怎样把问题推进成展览", "How the agent advances a question into an exhibition"),
    agentArchitectureLead: line(
      "这是受控的单 Agent 工作流：确定性程序掌握流程，语言模型只在被授权的环节写作。",
      "This is a controlled single-agent workflow: deterministic code owns the process, and the language model writes only inside authorised stages.",
    ),
    agentStageInterview: line("结构化访谈", "Structured interview"),
    agentStageInterviewBody: line(
      "状态机建立兴趣、熟悉度、参观时长与回避项；模型不能擅自改变访谈顺序。",
      "A state machine builds interest, familiarity, visit length and exclusions; the model cannot rewrite the interview order.",
    ),
    agentStageRetrieve: line("可回答性与选物", "Answerability and selection"),
    agentStageRetrieveBody: line(
      "先判断馆藏能否支撑问题，再用混合 RAG 与跨文化多样性重排选择候选。",
      "The collection is checked for support first, then hybrid RAG and cross-cultural diversity ranking select candidates.",
    ),
    agentStageBrief: line("策展合同", "Curatorial contract"),
    agentStageBriefBody: line(
      "在写标题和展签前，先固定 Big Idea、分论点、对象角色、证据 ID 与审阅状态。",
      "Before titles or labels, the big idea, sub-arguments, object roles, evidence IDs and review status are fixed.",
    ),
    agentStageCompose: line("分阶段写作", "Staged composition"),
    agentStageComposeBody: line(
      "DeepSeek 先形成展览框架，再由视觉模型逐件看图、并发撰写展签；最终题名确定后，海报走独立旁路。",
      "DeepSeek forms the exhibition frame first; a vision model then reads each object image and writes labels in parallel, while the settled title starts the poster sidecar.",
    ),
    agentStageValidate: line("验证后入场", "Validate before entry"),
    agentStageValidateBody: line(
      "结构、角色、证据深度与逐句引用通过检查后，展览才进入可参观状态。",
      "Only after structure, roles, evidence depth and sentence-level citations pass their checks can the exhibition become visitable.",
    ),
    harnessTitle: line("Harness 怎样约束与接住模型", "How the harness constrains and catches the model"),
    harnessLead: line(
      "Harness 不是另一个模型，而是包围每次模型调用的合同、预算、校验、回退与留痕。",
      "The harness is not another model. It is the contracts, budgets, validation, fallbacks and trace around every model call.",
    ),
    harnessContract: line("输入输出合同", "Input and output contracts"),
    harnessContractBody: line(
      "模型只接收本轮可见的馆方证据摘要，并且只能返回受类型约束的 JSON。",
      "The model sees only the institutional evidence excerpts exposed for that call and may return only typed JSON.",
    ),
    harnessTransaction: line("事务性证据门", "Transactional evidence gate"),
    harnessTransactionBody: line(
      "隐藏证据、伪造 ID、跨对象借证或半写入结果会被整段拒绝，不进入展览记录。",
      "Hidden evidence, invented IDs, cross-object borrowing and partially applied output are rejected as a unit and never enter the exhibition record.",
    ),
    harnessFallback: line("阶段预算与诚实回退", "Stage budgets and honest fallback"),
    harnessFallbackBody: line(
      "框架和展签各有墙钟预算；超时改用已标注的确定性文本，海报或语音失败不抹掉已经成立的展览。",
      "Frame and label stages have wall-clock budgets. Timeouts switch to disclosed deterministic text, while a poster or voice failure never erases an otherwise valid exhibition.",
    ),
    harnessTrace: line("过程与版本留痕", "Process and version trace"),
    harnessTraceBody: line(
      "页面显示每一步的真实发现；记录保留实际 provider、模型、检索、馆藏、提示词与验证器版本。",
      "The page shows what each stage actually found, while the record keeps the real provider, model, retrieval, collection, prompt and validator versions.",
    ),
    methodCoreTitle: line("支撑这条链路的四项核心技术", "Four core technologies supporting the chain"),
    method01: line("混合 RAG", "Hybrid RAG"),
    method01Body: line(
      "字段检索与多语向量检索经 RRF 融合，再按证据片段和文化差异重排；主体硬门控减少答非所问。",
      "Field search and multilingual vector search are fused with RRF, then re-ranked on evidence excerpts and cultural difference; a hard subject gate cuts down on answers to a question nobody asked.",
    ),
    method02Body: line(
      "每场展览记录 Big Idea、关键问题、对象角色、入选与排除理由；来源史和文化敏感性未审状态不会被伪装成“已通过”。",
      "Every exhibition records its big idea, critical questions, object roles, and the reasons for inclusion and exclusion. An unreviewed provenance or cultural-sensitivity status is never dressed up as “cleared”.",
    ),
    method03: line("证据约束写作", "Evidence-bound writing"),
    method03Body: line(
      "馆方事实、AI 策展解释与不确定说法分层；核心判断绑定同一对象的证据，证据较薄的对象不承担核心证据角色。",
      "Institutional fact, curatorial reading and uncertain claim are kept in separate layers. A core judgement is bound to evidence from the same object, and thinly evidenced objects never carry a core evidential role.",
    ),
    method04: line("多模态呈现", "Multimodal presentation"),
    method04Body: line(
      "Qwen Image 3 优先生成主题主视觉，中文由排版器精确合成；千问 TTS 提供专业声线导览，服务不可用时明确回退设备语音。同一展览结构同时进入 3D 与 2D 网页版。",
      "Qwen Image 3 generates the key visual, with any Chinese set precisely by a typesetter rather than the model; Qwen TTS supplies the studio voice, falling back visibly to the device voice when unavailable. The same exhibition structure drives both the 3D hall and the 2D web version.",
    ),
    methodBoundary: line(
      "这是一套受控的单 Agent 编排，不是任意自主行动的多 Agent 网络。可追溯不等于专业审阅；来源史、文化敏感性与相关社群审阅状态会如实保留。",
      "This is a controlled single-agent orchestration, not a network of freely acting autonomous agents. Traceable is not the same as professionally reviewed; provenance, cultural sensitivity and community-review status are reported as they actually stand.",
    ),

    collectionLive: line(
      "当前接入 {objects} 件开放馆藏，来自 {museums} 家博物馆。",
      "Currently {objects} open collection objects from {museums} museums.",
    ),
    collectionLiveSub: line(
      "语料涵盖全球多种文化；可按地区、年代、媒材和主题线索浏览，或直接向“彦远”提问。",
      "The corpus spans many cultures. Browse by region, period, medium or theme, or simply put a question to Yanyuan.",
    ),
    collectionLoading: line("开放馆藏目录正在载入。", "The open collection catalogue is loading."),
    collectionLoadingSub: line(
      "实际件数、来源机构与可探索线索会在连接后出现。",
      "Object counts, holding institutions and the threads you can follow appear once it connects.",
    ),
    collectionOffline: line("馆藏目录暂时未连接。", "The collection catalogue is not connected."),
    collectionOfflineSub: line(
      "请确认本地 API 已启动；馆藏恢复后才能生成展览。",
      "Check that the API is running; an exhibition can only be generated once the collection is back.",
    ),
    collectionLoadingNote: line(
      "正在读取当前冻结馆藏，缩略图与实际件数稍后出现。",
      "Reading the current frozen collection; thumbnails and counts follow shortly.",
    ),
    collectionOfflineNote: line(
      "馆藏接口暂未连接。启动本地 API 后刷新本页，再开始访谈。",
      "The collection endpoint is not connected. Start the API, refresh this page, then begin the interview.",
    ),
    collectionEmpty: line(
      "当前冻结馆藏没有可用于展览的对象，请先检查馆藏配置。",
      "The frozen collection holds no objects usable for an exhibition; check the collection configuration first.",
    ),
    mainTypes: line("主要门类：", "Main categories: "),
    typeWithCount: line("{label}（{count}）", "{label} ({count})"),
    collectionNote: line(
      "这里展示当前馆藏覆盖的主题线索。“彦远”会根据你的问题重新确定主题和章节；材料不足时会说明缺口，并推荐当前馆藏可支持的相邻方向。",
      "These are the threads the current collection covers. Yanyuan settles the theme and chapters against your question; where the material runs short it says so, and suggests a neighbouring direction the collection can actually support.",
    ),

    howTitle: line("怎么使用", "How to use it"),
    howTalkTitle: line("聊几句", "Say a few words"),
    howTalkBody: line(
      "“彦远”会问你的兴趣、熟悉程度、参观时长和回避内容；也可以整段跳过。问题选项会根据当前馆藏动态调整。",
      "Yanyuan asks about your interest, familiarity, how long you have and what to avoid — or skip the whole thing. The options adjust to what the collection currently holds.",
    ),
    howWatchTitle: line("看它策展", "Watch it curate"),
    howWatchBody: line(
      "页面会显示检索数量、入选展品、章节安排和展签生成状态。",
      "The page shows how many objects were searched, which were selected, how the chapters fell out, and the state of each label.",
    ),
    howEnterTitle: line("走进去", "Walk in"),
    howEnterBody: line(
      "可按导览逐站参观，也可切换到自由行走。每件展品都可查看博物馆原始记录。",
      "Follow the route stop by stop, or switch to walking freely. Every object opens onto the museum's own record.",
    ),

    boundaryTitle: line("馆藏、AI 与数据边界", "Collection, AI, and data limits"),
    boundaryObjectsTitle: line("展品来自博物馆开放馆藏", "The objects come from open museum collections"),
    boundaryObjectsBody: line(
      "展览只从下方来源机构中选取经逐件权利筛选的 CC0 / Public Domain 公开馆藏图像。不生成、不改画任何文物。",
      "Exhibitions draw only on CC0 / Public Domain collection images from the institutions listed below, screened for rights object by object. No artefact is ever generated or repainted.",
    ),
    boundaryTextTitle: line("策展文本由 AI 生成", "The curatorial text is AI-generated"),
    boundaryTextBody: line(
      "机构原文直接引用并标明出处，系统推断另作标记，每件都能点开核对。",
      "Institutional wording is quoted with its source; system inference is marked as such; both can be opened and checked on every object.",
    ),
    boundaryHallTitle: line("展厅与入口海报由 AI 生成", "The hall and entrance poster are AI-generated"),
    boundaryHallBody: line(
      "空间、灯光与配色按你的访谈结果生成；图像模型生成主题主视觉，中文标题由系统精确排版。生图失败时改用馆藏图像与同一排版模板，不影响展览。",
      "Space, lighting and colour follow from your interview. An image model makes the key visual, while any Chinese title is set precisely by the system rather than drawn. If generation fails, collection images and the same layout take over without affecting the exhibition.",
    ),
    boundaryDemoTitle: line("这是研究型 Demo", "This is a research demo"),
    boundaryDemoBody: line(
      "系统会用随机会话标识记录必要的参观事件，用于内部汇总；统计导出不包含原始会话 ID、访谈答案或排除项。结果不构成学习成效证明，也不替代专业策展。",
      "Visit events are recorded against a random session identifier for internal aggregation; the statistics export contains no raw session IDs, interview answers or exclusions. Nothing here evidences learning outcomes, and none of it replaces professional curation.",
    ),

    creditsTitle: line("馆藏来源与致谢", "Collections and credits"),
    creditsBody: line(
      "本版使用 CMA、The Met 与 AIC 的开放馆藏。卧游保存机构著录、对象页与逐字段许可，并仅将明确开放的图片用于展览。",
      "This build uses the open collections of the Cleveland Museum of Art, The Metropolitan Museum of Art and the Art Institute of Chicago. Woyou keeps each institutional record, object page and per-field licence, and shows only images that are explicitly open.",
    ),
    creditsLinkLabel: line(
      "查看 {museum} 的开放获取说明（新窗口）",
      "Open-access policy for {museum} (new window)",
    ),
    licencePerObject: line("许可逐件记录", "Licence recorded per object"),
    creditsObjectLine: line("{count} 件 · 图片{licence}", "{count} objects · images {licence}"),
    creditsDisclaimer: line(
      "机构标识当前仅用于内部研究演示中的数据来源致谢；相关商标归各机构所有，不属于馆藏开放许可，也不表示这些机构对卧游提供赞助或背书。公开发布前应按各机构品牌条款另行确认许可，或改用纯文字来源铭牌。",
      "Institutional marks appear here solely to credit data sources in an internal research demo. The trademarks belong to those institutions, fall outside the open-collection licences, and imply no sponsorship or endorsement of Woyou. Before any public release, clear them under each institution's brand terms or switch to a text-only source credit.",
    ),
    footerNames: line(
      "名出宗炳「卧以游之」；“AI 策展人彦远”名出张彦远《历代名画记》",
      "“Woyou” after Zong Bing's “travel it lying down”; Yanyuan after Zhang Yanyuan's Record of Famous Painters of Successive Dynasties",
    ),
  },
  hall: {
    posterAlt: line("AI 生成的展览海报", "AI-generated exhibition poster"),
    posterFallbackAlt: line("{title}，馆藏公开图像回退", "{title}, open collection image fallback"),
    labelOf: line("展签：{title}", "Label: {title}"),
    moveLabelAria: line(
      "移动展签；可拖动，或使用方向键微调，按 Shift 加速",
      "Move the label: drag it, or nudge with the arrow keys; hold Shift to move faster",
    ),
    moveLabelTitle: line("拖动展签；方向键也可以移动", "Drag the label; arrow keys work too"),
    moveLabel: line("移动展签", "Move label"),
    shrink: line("缩小展签", "Shrink label"),
    enlarge: line("放大展签", "Enlarge label"),
    expandLabel: line("展开展签", "Expand label"),
    collapseLabel: line("收起展签", "Collapse label"),
    expand: line("展开", "Expand"),
    collapse: line("收起", "Collapse"),
    depthTagTitle: line(
      "该机构未提供策展说明字段，展签只使用著录信息",
      "This institution supplies no curatorial description field, so the label uses catalogue data only",
    ),
    depthTag: line("仅著录信息", "Catalogue data only"),
    hideSources: line("收起来源", "Hide sources"),
    sources: line("来源", "Sources"),
    fieldSource: line("字段来源", "Field source"),
    pausedFreeWalk: line("⏸ 自由行走时暂停", "⏸ Paused while walking freely"),
    pausedCamera: line("⏸ 镜头移动时暂停", "⏸ Paused while the camera moves"),
    guidePreparing: line("⏳ AI 讲述准备中", "⏳ Preparing the AI narration"),
    guideSpeaking: line("🔊 彦远讲述中", "🔊 Yanyuan is speaking"),
    deviceSpeaking: line("🔊 设备语音讲述中", "🔊 Device voice speaking"),
    retryGuide: line("↻ 重试专业讲述", "↻ Retry the studio narration"),
    playGuide: line("🎙 彦远专业讲述（AI 合成）", "🎙 Yanyuan, studio narration (AI-synthesised)"),
    guidePreparingLong: line(
      "千问 AI 正在准备专业播音导览…",
      "Qwen is preparing the studio narration…",
    ),
    leaveHall: line("← 离开展厅", "← Leave the hall"),
    accessibleVersion: line("无障碍版本", "Accessible version"),
    noAudio: line("当前浏览器不支持音频播放", "This browser cannot play audio"),
    resumeInGuided: line("切回导览模式后继续播放", "Switch back to guided mode to continue"),
    playOnArrival: line(
      "镜头到站后可以播放专业讲述",
      "The studio narration can play once the camera arrives",
    ),
    modeGroup: line("参观方式", "How to visit"),
    guided: line("导览", "Guided"),
    freeWalk: line("自由行走", "Walk freely"),
    curatedBy: line("AI 策展人彦远为你策展", "Curated for you by Yanyuan, the AI curator"),
    segments: line("叙事区段", "Segments"),
    objects: line("展品", "Objects"),
    estimated: line("预计", "About"),
    minutes: line("{n} 分钟", "{n} min"),
    startVisit: line("开始参观 →", "Start the visit →"),
    chapterOf: line("第 {n} 部分 / 共 {total}", "Part {n} of {total}"),
    epilogueEyebrow: line("结语", "In closing"),
    epilogueTitle: line("AI 策展人彦远的结语", "Yanyuan's closing remarks"),
    materialLimits: line("这场展览的材料边界", "What this exhibition's material cannot do"),
    replay: line("从头再看", "Watch again"),
    endVisit: line("结束参观", "End the visit"),
    dragToLook: line("按住画面拖动即可转向", "Drag on the view to look around"),
    dragWalkHint: line(
      "WASD 或方向键移动；支持的浏览器也可单击画面捕获鼠标，按 Esc 释放",
      "Move with WASD or the arrow keys; supported browsers can also capture the pointer on click, released with Esc",
    ),
    progressNav: line("参观进度", "Visit progress"),
    previous: line("← 上一处", "← Previous"),
    next: line("下一处 →", "Next →"),
    motionNote: line(
      "已按系统设置关闭镜头飞行动画。",
      "Camera fly-through is off, following your system motion setting.",
    ),
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
    sentenceVisualObservation: line("图像观察", "Image observation"),
    sentenceSystemInference: line("系统推断", "System inference"),
    sentenceUncertain: line("仍不确定", "Still uncertain"),
    sourceMetadata: line("机构元数据", "Institutional metadata"),
    sourceCuratorialText: line("机构说明", "Institutional description"),
    sourceProvenance: line("机构来源记录", "Institutional provenance record"),
    sourceCollectionImage: line("馆藏图像", "Collection image"),
    sourceCollectionImageDescription: line(
      "这一来源记录对应送入视觉模型的馆藏图像，只支持颜色、形制、构图与表面状态等可见观察。",
      "This source is the collection image supplied to the vision model. It supports visible colour, form, composition and surface condition only.",
    ),
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
    configuredModel: line("展览框架模型", "Exhibition frame model"),
    configuredLabelsModel: line("展签视觉模型", "Visual label model"),
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
    visitBrief: resolve(COPY.visitBrief),
    recovery: resolve(COPY.recovery),
    landing: resolve(COPY.landing),
    hall: resolve(COPY.hall),
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
