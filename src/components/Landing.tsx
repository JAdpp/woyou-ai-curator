"use client";

import { useEffect, useState } from "react";
import { getCollectionHighlights, resolveObjectImageUrl } from "@/lib/api";
import type { CollectionHighlights } from "@/lib/types";
import {
  COLLECTION_SCOPE,
  CURATOR_NAME,
  PRODUCT_NAME,
  PRODUCT_NAME_LATIN,
} from "@/lib/brand";
import styles from "./landing.module.css";

const CC0_URL = "https://creativecommons.org/publicdomain/zero/1.0/";
const CC_BY_4_URL = "https://creativecommons.org/licenses/by/4.0/";
const QUOTED_CURATOR_NAME = `“${CURATOR_NAME}”`;
const QUOTED_CURATOR_TITLE = `AI 策展人${QUOTED_CURATOR_NAME}`;
const WALL_ROW_COUNT = 4;
const WALL_ITEMS_PER_ROW = 8;

interface CaseObject {
  id: string;
  place: string;
  date: string;
  title: string;
  material: string;
  institution: string;
  objectUrl: string;
  rights: ReadonlyArray<{
    scope: string;
    license: string;
    url: string;
  }>;
}

const BLUE_CASE_OBJECTS = [
  {
    id: "cma:1939.205",
    place: "中国 · 景德镇",
    date: "1736–1795",
    title: "Bottle Vase with Dragons and Waves",
    material: "釉下钴蓝与铜红彩瓷",
    institution: "Cleveland Museum of Art",
    objectUrl: "https://www.clevelandart.org/art/1939.205",
    rights: [
      { scope: "所示图片与基础元数据", license: "CC0 1.0", url: CC0_URL },
    ],
  },
  {
    id: "met:37451",
    place: "越南",
    date: "15 世纪",
    title: "Plate with Peonies",
    material: "釉下钴蓝彩炻器",
    institution: "The Metropolitan Museum of Art",
    objectUrl: "https://www.metmuseum.org/art/collection/search/37451",
    rights: [
      { scope: "所示图片与基础元数据", license: "CC0 1.0", url: CC0_URL },
    ],
  },
  {
    id: "aic:58075",
    place: "伊朗",
    date: "17 世纪",
    title: "Dish with Floral and Animal Decoration",
    material: "蓝、黑彩釉下装饰熔块陶",
    institution: "Art Institute of Chicago",
    objectUrl: "https://www.artic.edu/artworks/58075",
    rights: [
      { scope: "所示图片与基础元数据", license: "CC0 1.0", url: CC0_URL },
      { scope: "馆方作品说明", license: "CC BY 4.0", url: CC_BY_4_URL },
    ],
  },
] as const satisfies ReadonlyArray<CaseObject>;

const CAT_CASE_OBJECTS = [
  {
    id: "cma:2021.130",
    place: "安第斯中部 · 帕拉卡斯",
    date: "公元前 500–400 年",
    title: "Square Bowl with Pampas Cats",
    material: "陶器与烧成后彩绘",
    institution: "Cleveland Museum of Art",
    objectUrl: "https://www.clevelandart.org/art/2021.130",
    rights: [
      { scope: "所示图片、基础元数据与馆方说明", license: "CC0 1.0", url: CC0_URL },
    ],
  },
  {
    id: "cma:1942.776",
    place: "埃及 · 晚期王朝",
    date: "公元前 664–30 年",
    title: "Cat",
    material: "青铜与金",
    institution: "Cleveland Museum of Art",
    objectUrl: "https://www.clevelandart.org/art/1942.776",
    rights: [
      { scope: "所示图片、基础元数据与馆方说明", license: "CC0 1.0", url: CC0_URL },
    ],
  },
  {
    id: "cma:1979.82",
    place: "荷兰",
    date: "约 1670 年代",
    title: "The Monkey and the Cat",
    material: "布面油画",
    institution: "Cleveland Museum of Art",
    objectUrl: "https://www.clevelandart.org/art/1979.82",
    rights: [
      { scope: "所示图片、基础元数据与馆方说明", license: "CC0 1.0", url: CC0_URL },
    ],
  },
] as const satisfies ReadonlyArray<CaseObject>;

const MUSEUM_CREDITS = [
  {
    id: "cma",
    name: "Cleveland Museum of Art",
    shortName: "CMA",
    href: "https://www.clevelandart.org/open-access",
    logo: "/institutions/cma.svg",
  },
  {
    id: "met",
    name: "The Metropolitan Museum of Art",
    shortName: "The Met",
    href: "https://www.metmuseum.org/policies/image-resources",
    logo: "/institutions/met.svg",
  },
  {
    id: "aic",
    name: "Art Institute of Chicago",
    shortName: "AIC",
    href: "https://www.artic.edu/open-access/open-access-images",
    logo: "/institutions/aic.svg",
  },
] as const;

function buildWallRows(items: CollectionHighlights["items"]) {
  if (items.length === 0) return [];
  return Array.from({ length: WALL_ROW_COUNT }, (_, rowIndex) =>
    Array.from(
      { length: WALL_ITEMS_PER_ROW },
      (_, itemIndex) => items[(rowIndex * WALL_ITEMS_PER_ROW + itemIndex) % items.length],
    ),
  );
}

function CaseEvidenceLine({
  items,
  ariaLabel,
}: {
  items: ReadonlyArray<CaseObject>;
  ariaLabel: string;
}) {
  return (
    <ol className={styles.evidenceLine} aria-label={ariaLabel}>
      {items.map((item, index) => (
        <li key={item.id}>
          <div className={styles.caseImageWrap} data-fallback="图像暂不可用">
            {/* eslint-disable-next-line @next/next/no-img-element -- object proxy provides fixed, licensed collection media */}
            <img
              src={resolveObjectImageUrl(item.id, 512)}
              alt={`${item.title}，${item.institution}`}
              loading="lazy"
              onError={(event) => { event.currentTarget.hidden = true; }}
            />
            <span>{String(index + 1).padStart(2, "0")}</span>
          </div>
          <p className={styles.casePlace}>{item.place} · {item.date}</p>
          <h3>{item.title}</h3>
          <p className={styles.caseMaterial}>{item.material}</p>
          <small>{item.institution}</small>
          <a
            className={styles.caseSource}
            href={item.objectUrl}
            target="_blank"
            rel="noopener noreferrer"
            aria-label={`查看 ${item.title} 的机构原始记录（新窗口）`}
          >
            查看机构原始记录 <span aria-hidden="true">↗</span>{/* deslop-ignore 15 -- semantic external-link indicator */}
          </a>
          <small className={styles.caseLicense}>
            {item.rights.map((right, rightIndex) => (
              <span key={`${item.id}-${right.scope}`}>
                {rightIndex > 0 && " · "}
                {right.scope}{" "}
                <a
                  href={right.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={`${right.scope}权利说明：${right.license}（新窗口）`}
                >
                  {right.license}
                </a>
              </span>
            ))}
          </small>
        </li>
      ))}
    </ol>
  );
}

/**
 * Entry page.
 *
 * The hero is a wall of real objects from the loaded collection rather than
 * stock decoration — the product's whole claim is that it builds halls out of
 * actual open museum holdings, so the landing should be made of them too. It
 * degrades to a plain gradient if the API is not up.
 */
export function Landing({ onStart }: { onStart: () => void }) {
  const [highlights, setHighlights] = useState<CollectionHighlights | null>(null);
  const [collectionStatus, setCollectionStatus] = useState<"loading" | "ready" | "error">(
    "loading",
  );

  useEffect(() => {
    let cancelled = false;
    getCollectionHighlights(28)
      .then((data) => {
        if (!cancelled) {
          setHighlights(data);
          setCollectionStatus("ready");
        }
      })
      .catch(() => {
        // The wall is decorative; a cold API must not block the page.
        if (!cancelled) setCollectionStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const objectCount = highlights?.objectCount ?? null;
  const institutions = highlights?.institutions ?? [];
  const domains = highlights?.domains ?? [];
  const wallRows = buildWallRows(highlights?.items ?? []);

  return (
    <div className={styles.page}>
      <section className={styles.hero}>
        <div className={styles.wall} aria-hidden="true">
          {wallRows.map((row, rowIndex) => (
            <div className={styles.wallRow} key={`wall-row-${rowIndex}`}>
              <div className={styles.wallTrack}>
                {[0, 1].map((copyIndex) => (
                  <div className={styles.wallSet} key={`wall-row-${rowIndex}-copy-${copyIndex}`}>
                    {row.map((item, itemIndex) => (
                      // eslint-disable-next-line @next/next/no-img-element -- decorative wall, fixed by the collection image proxy
                      <img
                        key={`${copyIndex}-${item.id}-${itemIndex}`}
                        src={resolveObjectImageUrl(item.id, 512)}
                        alt=""
                        loading={copyIndex === 0 && rowIndex < 2 ? "eager" : "lazy"}
                        onError={(event) => { event.currentTarget.hidden = true; }}
                      />
                    ))}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
        <div className={styles.wallVeil} aria-hidden="true" />

        <div className={styles.heroInner}>
          <p className={styles.epigraph}>
            <span>澄怀观道，卧以游之。</span>
            <cite>— 宗炳《画山水序》，五世纪</cite>
          </p>

          <h1 className={styles.wordmark}>
            {PRODUCT_NAME}
            <span className={styles.latin}>{PRODUCT_NAME_LATIN}</span>
          </h1>

          <p className={styles.lede}>
            把你的好奇，变成一座只为你搭的展厅。
            <br />
            {QUOTED_CURATOR_TITLE}会从{COLLECTION_SCOPE}中检索、比较，并为你组织主题、章节与导览。
          </p>

          <button type="button" className={styles.cta} onClick={onStart}>
            和{QUOTED_CURATOR_TITLE}聊聊
            <span aria-hidden="true">→</span>
          </button>

          <p className={styles.microcopy}>
            无需注册 · 访谈可整段跳过 · 同时支持 3D 与 2D 网页版
          </p>
        </div>

        <p className={styles.corpusLine} aria-live="polite" aria-atomic="true">
          {objectCount !== null ? (
            <>
              当前接入 <strong>{objectCount.toLocaleString()}</strong> 件开放馆藏，来自
              <strong>{institutions.length}</strong> 家博物馆；涵盖全球多种文化，许可逐字段记录。
            </>
          ) : collectionStatus === "loading" ? (
            "正在读取当前冻结馆藏。"
          ) : (
            "馆藏数据暂未连接。"
          )}
        </p>
      </section>

      <main>
        <section className={styles.caseStudy} aria-labelledby="landing-case-title">
          <article className={styles.caseExample}>
            <div className={styles.caseHeading}>
              <div>
                <span className={styles.caseIndex}>案例一 · 色彩与技术</span>
                <h2 id="landing-case-title">相似的蓝，为什么会出现在中国瓷瓶、越南盘与伊朗陶器上？</h2>
              </div>
              <p>
                这个案例使用三件真实开放馆藏，展示{QUOTED_CURATOR_NAME}如何从访客问题提出可核查的策展命题。
              </p>
            </div>

            <div className={styles.caseFrame}>
              <div className={styles.caseQuestion}>
                <span>访客的问题</span>
                <p>“我只知道青花瓷。蓝色是不是从中国传到世界各地的？”</p>
                <small>系统先保留疑问，不把访客的猜测直接当成结论。</small>
              </div>

              <CaseEvidenceLine
                items={BLUE_CASE_OBJECTS}
                ariaLabel="蓝色案例中的三件馆藏证据"
              />

              <div className={styles.caseArgument}>
                <span>案例中的策展判断</span>
                <p>
                  相似的蓝色并不自动证明一条单向传播路线。把色彩与材料著录、胎釉、器形和年代放在一起，
                  才能追问技术如何被不同地区重新制作和使用。
                </p>
                <small>卡片中的题名、年代与材料来自三家博物馆的原始著录；联系与解释另行标记。</small>
              </div>
            </div>
          </article>

          <article className={styles.caseExample} aria-labelledby="landing-cat-case-title">
            <div className={styles.caseHeading}>
              <div>
                <span className={styles.caseIndex}>案例二 · 跨文化动物</span>
                <h2 id="landing-cat-case-title">同样是猫，为什么会走进安第斯陶碗、埃及青铜像与荷兰寓言画？</h2>
              </div>
              <p>
                这个案例来自此前生成的“猫咪的千面形象”展览：共享一个动物主题，不等于共享一种象征意义。
              </p>
            </div>

            <div className={styles.caseFrame}>
              <div className={styles.caseQuestion}>
                <span>访客的问题</span>
                <p>“各个文化里都有猫，它们是不是都代表神秘和好运？”</p>
                <small>系统先检索“猫”本身，再比较器物用途、材料、时代与馆方说明。</small>
              </div>

              <CaseEvidenceLine
                items={CAT_CASE_OBJECTS}
                ariaLabel="猫主题案例中的三件馆藏证据"
              />

              <div className={styles.caseArgument}>
                <span>案例中的策展判断</span>
                <p>
                  不能用一个“猫的象征意义”覆盖不同文化。猫进入容器装饰、小型金属像与寓言画的方式各不相同；
                  比较应从这些可见差异和机构记录开始。
                </p>
                <small>三件展品的图片、基础元数据与馆方说明均按 CMA 的 CC0 开放记录使用。</small>
              </div>
            </div>
          </article>

          <button type="button" className={styles.caseCta} onClick={onStart}>
            换成我的好奇
            <span aria-hidden="true">→</span>
          </button>
        </section>

        <section className={styles.features} aria-labelledby="feature-title">
          <div className={styles.featureIntro}>
            <h2 id="feature-title">AI策展人如何把问题变成展览</h2>
            <p>{QUOTED_CURATOR_NAME}先了解访客，再检索和比较馆藏、安排展线，并为每件展品保留机构来源与字段许可。</p>
          </div>

          <div className={styles.featureLedger}>
            <article>
              <div className={styles.featureCopy}>
                <span>访谈</span>
                <h3>先弄清你想怎么看</h3>
                <p>兴趣、参观倾向、熟悉程度、可用时间与不想看到的内容都会改变展品数量、解释深度和导览节奏；陌生题目会给出可选方向，也可直接跳过整段访谈。</p>
              </div>
              <div
                className={styles.chatDemo}
                role="img"
                aria-label="访谈示意：AI 策展人“彦远”询问哪一种好奇最接近今天想看的，访客回答想知道一种颜色怎样穿过不同文化，并选择第一次接触、约十分钟和不同地区。"
              >
                <p className={styles.agentBubble}>哪一种好奇最接近你今天想看的？</p>
                <p className={styles.visitorBubble}>我想知道一种颜色怎样穿过不同文化。</p>
                <div><span>第一次接触</span><span>约 10 分钟</span><span>想看不同地区</span></div>
              </div>
            </article>

            <article>
              <div className={styles.featureCopy}>
                <span>选择与编排</span>
                <h3>主题按你的问题形成</h3>
                <p>{QUOTED_CURATOR_NAME}会从{objectCount !== null ? `当前 ${objectCount.toLocaleString()} 件` : "当前可用"}馆藏中检索、比较、排除，再形成主题、章节和展品角色。</p>
              </div>
              <div
                className={styles.pipelineDemo}
                role="img"
                aria-label="策展进度示意：依次理解问题、寻找证据、组织展线，并搭建采用连续动线的展厅。"
              >
                {[
                  ["理解问题", "已提取颜色、流动、跨文化"],
                  ["寻找证据", "比较年代、材料与产地"],
                  ["组织展线", "保留一件反例或限制"],
                  ["搭建展厅", "用连续动线连接章节、导览与主题海报"],
                ].map(([label, finding], index) => (
                  <div key={label}>
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <p><strong>{label}</strong><small>{finding}</small></p>
                  </div>
                ))}
              </div>
            </article>

            <article>
              <div className={styles.featureCopy}>
                <span>参观方式</span>
                <h3>展品被放进一条空间叙事</h3>
                <p>可开启千问 AI 合成的{QUOTED_CURATOR_NAME}专业讲述，并用镜头逐站推进；想停下来时可以随时切换自由行走，也可直接打开完整 2D 网页版。</p>
              </div>
              <div
                className={styles.visitModes}
                role="img"
                aria-label="同一展览的两种参观方式：左侧为带真实馆藏图像和连续动线的 3D 导览，右侧为包含题名、展品图、展签与来源的 2D 网页版。"
              >
                <div className={styles.mode3d} aria-hidden="true">
                  <span className={styles.modeLabel}>3D 导览</span>
                  <div className={styles.corridorPreview}>
                    <span className={styles.corridorPath} />
                    {BLUE_CASE_OBJECTS.map((item) => (
                      // eslint-disable-next-line @next/next/no-img-element -- small functional preview from the same licensed proxy
                      <img
                        key={`hall-${item.id}`}
                        src={resolveObjectImageUrl(item.id, 512)}
                        alt=""
                        loading="lazy"
                        onError={(event) => { event.currentTarget.hidden = true; }}
                      />
                    ))}
                    <span className={styles.cameraPoint} />
                  </div>
                  <small>连续展线 · 逐站镜头 · 随时自由行走</small>
                </div>
                <div className={styles.mode2d} aria-hidden="true">
                  <span className={styles.modeLabel}>2D 网页版</span>
                  <div className={styles.browserPreview}>
                    <div className={styles.browserBar}><i /><i /><i /></div>
                    <div className={styles.browserSheet}>
                      <strong>相似的蓝</strong>
                      {/* eslint-disable-next-line @next/next/no-img-element -- small functional preview from the same licensed proxy */}
                      <img
                        src={resolveObjectImageUrl(BLUE_CASE_OBJECTS[0].id, 512)}
                        alt=""
                        loading="lazy"
                        onError={(event) => { event.currentTarget.hidden = true; }}
                      />
                      <span /><span /><span />
                    </div>
                  </div>
                  <small>完整展签 · 机构来源 · 键盘与读屏可用</small>
                </div>
              </div>
            </article>

            <article>
              <div className={styles.featureCopy}>
                <span>来源与许可</span>
                <h3>每件展品都保留来源</h3>
                <p>机构原文、系统转述与策展推断分开显示。题名、年代、材料、权利声明和机构对象页随展品保留，便于核对。</p>
              </div>
              <div
                className={styles.labelDemo}
                role="img"
                aria-label="展签来源面板示意：机构著录与策展解释分开显示，来源为芝加哥艺术博物馆；所示材质字段使用 CC0 1.0，馆方作品说明使用 CC BY 4.0。"
              >
                <p><span>机构著录</span>Fritware painted in blue and black over white slip…</p>
                <p><span>策展解释</span>这件器物提供了另一种蓝白装饰的材料路径。</p>
                <dl>
                  <div>
                    <dt>来源</dt>
                    <dd>Art Institute of Chicago</dd>
                  </div>
                  <div>
                    <dt>字段许可</dt>
                    <dd>材质字段 CC0 1.0 · 馆方作品说明 CC BY 4.0</dd>
                  </div>
                </dl>
              </div>
            </article>
          </div>
        </section>

        <section className={styles.method} aria-labelledby="method-title">
          <div className={styles.methodIntro}>
            <div>
              <span className={styles.methodKicker}>AI 技术 × 策展方法</span>
              <h2 id="method-title">AI 技术，服务于可检查的策展判断</h2>
            </div>
            <p>
              {QUOTED_CURATOR_NAME}不是先写故事、再寻找插图。它先从馆方著录与证据片段中召回候选，
              再把命题、分论点、对象角色和材料边界写进版本化策展简报。
            </p>
          </div>

          <dl className={styles.methodLedger}>
            <div>
              <dt><span>01</span>混合 RAG</dt>
              <dd>字段检索与多语向量检索经 RRF 融合，再按证据片段和文化差异重排；主体硬门控减少答非所问。</dd>
            </div>
            <div>
              <dt><span>02</span>Curatorial Brief</dt>
              <dd>每场展览记录 Big Idea、关键问题、对象角色、入选与排除理由；来源史和文化敏感性未审状态不会被伪装成“已通过”。</dd>
            </div>
            <div>
              <dt><span>03</span>证据约束写作</dt>
              <dd>馆方事实、AI 策展解释与不确定说法分层；核心判断绑定同一对象的证据，证据较薄的对象不承担核心证据角色。</dd>
            </div>
            <div>
              <dt><span>04</span>多模态呈现</dt>
              <dd>Qwen Image 3 优先生成主题主视觉，中文由排版器精确合成；千问 TTS 提供专业声线导览，服务不可用时明确回退设备语音。同一展览结构同时进入 3D 与 2D 网页版。</dd>
            </div>
          </dl>

          <p className={styles.methodBoundary}>
            可追溯不等于专业审阅。卧游仍是内部研究 Demo；来源史、文化敏感性与相关社群审阅状态会如实保留。
          </p>
        </section>

        <section
          className={styles.collection}
          aria-labelledby="collection-title"
          aria-busy={collectionStatus === "loading"}
        >
          <h2 id="collection-title" className={styles.collectionLead}>
            {objectCount !== null ? (
              <>
                当前接入 {objectCount.toLocaleString()} 件开放馆藏，来自 {institutions.length} 家博物馆。
                <span>语料涵盖全球多种文化；可按地区、年代、媒材和主题线索浏览，或直接向{QUOTED_CURATOR_NAME}提问。</span>
              </>
            ) : collectionStatus === "loading" ? (
              <>
                开放馆藏目录正在载入。
                <span>实际件数、来源机构与可探索线索会在连接后出现。</span>
              </>
            ) : (
              <>
                馆藏目录暂时未连接。
                <span>请确认本地 API 已启动；馆藏恢复后才能生成展览。</span>
              </>
            )}
          </h2>

          {collectionStatus !== "ready" && (
            <p className={styles.collectionLoadNote} role="status">
              {collectionStatus === "loading"
                ? "正在读取当前冻结馆藏，缩略图与实际件数稍后出现。"
                : "馆藏接口暂未连接。启动本地 API 后刷新本页，再开始访谈。"}
            </p>
          )}

          {collectionStatus === "ready" && objectCount === 0 && (
            <p className={styles.collectionLoadNote} role="status">
              当前冻结馆藏没有可用于展览的对象，请先检查馆藏配置。
            </p>
          )}

          {domains.length > 0 && (
            <ul className={styles.domainGrid}>
              {domains.map((domain) => (
              <li key={domain.id}>
                <div className={styles.domainThumbs} aria-hidden="true">
                  {domain.samples.map((sample) => (
                    // eslint-disable-next-line @next/next/no-img-element -- thumbnail strip, sized by CSS
                    <img
                      key={sample.id}
                      src={resolveObjectImageUrl(sample.id, 512)}
                      alt=""
                      loading="lazy"
                      onError={(event) => { event.currentTarget.hidden = true; }}
                    />
                  ))}
                </div>
                <div className={styles.domainText}>
                  <h3>
                    {domain.label}
                    <span className={styles.domainCount}>{domain.objectCount}</span>
                  </h3>
                  <p>{domain.hint}</p>
                </div>
              </li>
              ))}
            </ul>
          )}

          {(highlights?.topTypes?.length ?? 0) > 0 && (
            <p className={styles.typeLine}>
              主要门类：
              {highlights!.topTypes
                .slice(0, 6)
                .map((type) => `${type.label}（${type.count}）`)
                .join(" · ")}
            </p>
          )}

          {domains.length > 0 && (
            <p className={styles.collectionNote}>
              这里展示当前馆藏覆盖的主题线索。{QUOTED_CURATOR_NAME}会根据你的问题重新确定主题和章节；
              材料不足时会说明缺口，并推荐当前馆藏可支持的相邻方向。
            </p>
          )}
        </section>

        <section id="how-it-works" className={styles.how} aria-labelledby="how-title">
          <h2 id="how-title" className={styles.sectionTitle}>怎么使用</h2>
          <ol>
            <li>
              {/* deslop-ignore-next-line 30 -- a real, ordered visit sequence */}
              <span className={styles.step}>01</span>
              <h3>聊几句</h3>
              <p>
                {QUOTED_CURATOR_NAME}会问你的兴趣、熟悉程度、参观时长和回避内容；也可以整段跳过。
                问题选项会根据当前馆藏动态调整。
              </p>
            </li>
            <li>
              {/* deslop-ignore-next-line 30 -- a real, ordered visit sequence */}
              <span className={styles.step}>02</span>
              <h3>看它策展</h3>
              <p>
                页面会显示检索数量、入选展品、章节安排和展签生成状态。
              </p>
            </li>
            <li>
              {/* deslop-ignore-next-line 30 -- a real, ordered visit sequence */}
              <span className={styles.step}>03</span>
              <h3>走进去</h3>
              <p>
                可按导览逐站参观，也可切换到自由行走。每件展品都可查看博物馆原始记录。
              </p>
            </li>
          </ol>
        </section>

        <section className={styles.boundary} aria-labelledby="boundary-title">
          <h2 id="boundary-title" className={styles.sectionTitle}>馆藏、AI 与数据边界</h2>
          <dl>
            <div>
              <dt>展品来自博物馆开放馆藏</dt>
              <dd>
                展览只从下方来源机构中选取经逐件权利筛选的 CC0 / Public Domain 公开馆藏图像。
                不生成、不改画任何文物。
              </dd>
            </div>
            <div>
              <dt>策展文本由 AI 生成</dt>
              <dd>机构原文直接引用并标明出处，系统推断另作标记，每件都能点开核对。</dd>
            </div>
            <div>
              <dt>展厅与入口海报由 AI 生成</dt>
              <dd>空间、灯光与配色按你的访谈结果生成；图像模型生成主题主视觉，中文标题由系统精确排版。生图失败时改用馆藏图像与同一排版模板，不影响展览。</dd>
            </div>
            <div>
              <dt>这是研究型 Demo</dt>
              <dd>系统会用随机会话标识记录必要的参观事件，用于内部汇总；统计导出不包含原始会话 ID、访谈答案或排除项。结果不构成学习成效证明，也不替代专业策展。</dd>
            </div>
          </dl>
        </section>

        <section id="sources" className={styles.credits} aria-labelledby="credits-title">
          <div className={styles.creditsHeading}>
            <h2 id="credits-title">馆藏来源与致谢</h2>
            <p>本版使用 CMA、The Met 与 AIC 的开放馆藏。卧游保存机构著录、对象页与逐字段许可，并仅将明确开放的图片用于展览。</p>
          </div>
          <ul className={styles.logoList}>
            {MUSEUM_CREDITS.map((museum) => {
              const summary = highlights?.institutionSummaries?.find(
                (entry) => entry.id === museum.id,
              );
              return (
                <li key={museum.name}>
                  <a
                    href={museum.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label={`查看 ${museum.name} 的开放获取说明（新窗口）`}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element -- locally archived institution mark for source acknowledgement */}
                    <img
                      src={museum.logo}
                      alt=""
                      onError={(event) => { event.currentTarget.hidden = true; }}
                    />
                    <span>
                      <strong>{museum.shortName}</strong>
                      <small>{museum.name}</small>
                      {summary && (
                        <small>
                          {summary.objectCount.toLocaleString()} 件 · 图片
                          {summary.imageLicenses.join(" / ") || "许可逐件记录"}
                        </small>
                      )}
                    </span>
                  </a>
                </li>
              );
            })}
          </ul>
          <p className={styles.trademarkNote}>
            机构标识当前仅用于内部研究演示中的数据来源致谢；相关商标归各机构所有，不属于馆藏开放许可，
            也不表示这些机构对卧游提供赞助或背书。公开发布前应按各机构品牌条款另行确认许可，或改用纯文字来源铭牌。
          </p>
        </section>
      </main>

      <footer className={styles.footer}>
        <span>
          {PRODUCT_NAME} · {PRODUCT_NAME_LATIN}
        </span>
        <span className={styles.footerNote}>
          名出宗炳「卧以游之」；{QUOTED_CURATOR_TITLE}名出张彦远《历代名画记》
        </span>
      </footer>
    </div>
  );
}
