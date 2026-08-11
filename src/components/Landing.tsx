"use client";

import { useEffect, useState } from "react";
import { getCollectionHighlights, resolveObjectImageUrl } from "@/lib/api";
import type { CollectionHighlights } from "@/lib/types";
import { PRODUCT_NAME, PRODUCT_NAME_LATIN } from "@/lib/brand";
import { fill, type Language } from "@/lib/i18n";
import { useLanguage } from "@/lib/useLanguage";
import { SiteHeader } from "./SiteHeader";
import styles from "./landing.module.css";

const CC0_URL = "https://creativecommons.org/publicdomain/zero/1.0/";
const CC_BY_4_URL = "https://creativecommons.org/licenses/by/4.0/";
/* Place, material and rights wording on the case cards are ours, not the
   institutions'; titles and institution names stay exactly as catalogued. */
type Bi = Record<Language, string>;
const bi = (zh: string, en: string): Bi => ({ zh, en });
const WALL_ROW_COUNT = 4;
const WALL_ITEMS_PER_ROW = 8;

interface CaseObject {
  id: string;
  place: Bi;
  date: string | Bi;
  title: string;
  material: Bi;
  institution: string;
  objectUrl: string;
  rights: ReadonlyArray<{
    scope: Bi;
    license: string;
    url: string;
  }>;
}

const BLUE_CASE_OBJECTS = [
  {
    id: "cma:1939.205",
    place: bi("中国 · 景德镇", "China · Jingdezhen"),
    date: "1736–1795",
    title: "Bottle Vase with Dragons and Waves",
    material: bi("釉下钴蓝与铜红彩瓷", "Porcelain with underglaze cobalt blue and copper red"),
    institution: "Cleveland Museum of Art",
    objectUrl: "https://www.clevelandart.org/art/1939.205",
    rights: [
      { scope: bi("所示图片与基础元数据", "Image shown and core metadata"), license: "CC0 1.0", url: CC0_URL },
    ],
  },
  {
    id: "met:37451",
    place: bi("越南", "Vietnam"),
    date: bi("15 世纪", "15th century"),
    title: "Plate with Peonies",
    material: bi("釉下钴蓝彩炻器", "Stoneware with underglaze cobalt blue"),
    institution: "The Metropolitan Museum of Art",
    objectUrl: "https://www.metmuseum.org/art/collection/search/37451",
    rights: [
      { scope: bi("所示图片与基础元数据", "Image shown and core metadata"), license: "CC0 1.0", url: CC0_URL },
    ],
  },
  {
    id: "aic:58075",
    place: bi("伊朗", "Iran"),
    date: bi("17 世纪", "17th century"),
    title: "Dish with Floral and Animal Decoration",
    material: bi("蓝、黑彩釉下装饰熔块陶", "Fritware with underglaze blue and black decoration"),
    institution: "Art Institute of Chicago",
    objectUrl: "https://www.artic.edu/artworks/58075",
    rights: [
      { scope: bi("所示图片与基础元数据", "Image shown and core metadata"), license: "CC0 1.0", url: CC0_URL },
      { scope: bi("馆方作品说明", "Institutional description"), license: "CC BY 4.0", url: CC_BY_4_URL },
    ],
  },
] as const satisfies ReadonlyArray<CaseObject>;

const CAT_CASE_OBJECTS = [
  {
    id: "cma:2021.130",
    place: bi("安第斯中部 · 帕拉卡斯", "Central Andes · Paracas"),
    date: bi("公元前 500–400 年", "500–400 BCE"),
    title: "Square Bowl with Pampas Cats",
    material: bi("陶器与烧成后彩绘", "Ceramic with post-fired paint"),
    institution: "Cleveland Museum of Art",
    objectUrl: "https://www.clevelandart.org/art/2021.130",
    rights: [
      { scope: bi("所示图片、基础元数据与馆方说明", "Image shown, core metadata and institutional description"), license: "CC0 1.0", url: CC0_URL },
    ],
  },
  {
    id: "cma:1942.776",
    place: bi("埃及 · 晚期王朝", "Egypt · Late Period"),
    date: bi("公元前 664–30 年", "664–30 BCE"),
    title: "Cat",
    material: bi("青铜与金", "Bronze and gold"),
    institution: "Cleveland Museum of Art",
    objectUrl: "https://www.clevelandart.org/art/1942.776",
    rights: [
      { scope: bi("所示图片、基础元数据与馆方说明", "Image shown, core metadata and institutional description"), license: "CC0 1.0", url: CC0_URL },
    ],
  },
  {
    id: "cma:1979.82",
    place: bi("荷兰", "Netherlands"),
    date: bi("约 1670 年代", "about the 1670s"),
    title: "The Monkey and the Cat",
    material: bi("布面油画", "Oil on canvas"),
    institution: "Cleveland Museum of Art",
    objectUrl: "https://www.clevelandart.org/art/1979.82",
    rights: [
      { scope: bi("所示图片、基础元数据与馆方说明", "Image shown, core metadata and institutional description"), license: "CC0 1.0", url: CC0_URL },
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
  language,
}: {
  items: ReadonlyArray<CaseObject>;
  ariaLabel: string;
  language: Language;
}) {
  const { t: copy } = useLanguage();
  const t = copy.landing;
  const read = (value: string | Bi) =>
    typeof value === "string" ? value : value[language];
  return (
    <ol className={styles.evidenceLine} aria-label={ariaLabel}>
      {items.map((item, index) => (
        <li key={item.id}>
          <div className={styles.caseImageWrap} data-fallback="">
            {/* eslint-disable-next-line @next/next/no-img-element -- object proxy provides fixed, licensed collection media */}
            <img
              src={resolveObjectImageUrl(item.id, 512)}
              alt={`${item.title} · ${item.institution}`}
              loading="lazy"
              onError={(event) => { event.currentTarget.hidden = true; }}
            />
            <span>{String(index + 1).padStart(2, "0")}</span>
          </div>
          <p className={styles.casePlace}>{read(item.place)} · {read(item.date)}</p>
          <h3>{item.title}</h3>
          <p className={styles.caseMaterial}>{read(item.material)}</p>
          <small>{item.institution}</small>
          <a
            className={styles.caseSource}
            href={item.objectUrl}
            target="_blank"
            rel="noopener noreferrer"
            aria-label={`${t.source}: ${item.title} (new window)`}
          >
            {t.source} <span aria-hidden="true">↗</span>{/* deslop-ignore 15 -- semantic external-link indicator */}
          </a>
          <small className={styles.caseLicense}>
            {item.rights.map((right, rightIndex) => (
              <span key={`${item.id}-${rightIndex}`}>
                {rightIndex > 0 && " · "}
                {read(right.scope)}{" "}
                <a
                  href={right.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={`${read(right.scope)}: ${right.license} (new window)`}
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
  const { language, t: copy } = useLanguage();
  const t = copy.landing;
  const [highlights, setHighlights] = useState<CollectionHighlights | null>(null);
  const [collectionStatus, setCollectionStatus] = useState<"loading" | "ready" | "error">(
    "loading",
  );

  useEffect(() => {
    let cancelled = false;
    getCollectionHighlights(28, language)
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
    // Domain labels and hints are written by the backend in the requested
    // language, so a switch has to refetch. Without this the section kept the
    // language of the first render — which is always the default, since the
    // stored preference only arrives in a layout effect.
  }, [language]);

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

        <SiteHeader tone="onDark" compact internalLabel={false} />

        <div className={styles.heroInner}>
          <p className={styles.epigraph}>
            <span>{t.epigraph}</span>
            <cite>{t.epigraphCite}</cite>
          </p>

          <h1 className={styles.wordmark}>
            {PRODUCT_NAME}
            <span className={styles.latin}>{PRODUCT_NAME_LATIN}</span>
          </h1>

          <p className={styles.lede}>
            {t.heroTitle}
            <br />
            {t.heroBody}
          </p>

          <button type="button" className={styles.cta} onClick={onStart}>
            {t.heroCta}
            <span aria-hidden="true">→</span>
          </button>

          <p className={styles.microcopy}>
            {t.microcopy}
          </p>
        </div>

        <p className={styles.corpusLine} aria-live="polite" aria-atomic="true">
          {objectCount !== null ? (
            <>
              {fill(t.corpusLive, {
                objects: objectCount.toLocaleString(),
                museums: institutions.length,
              })}
            </>
          ) : collectionStatus === "loading" ? (
            t.corpusLoading
          ) : (
            t.corpusOffline
          )}
        </p>
      </section>

      <main>
        <section className={styles.caseStudy} aria-labelledby="landing-case-title">
          <article className={styles.caseExample}>
            <div className={styles.caseHeading}>
              <div>
                <span className={styles.caseIndex}>{t.caseOneIndex}</span>
                <h2 id="landing-case-title">{t.blueCaseTitle}</h2>
              </div>
              <p>
                {t.blueCaseLead}
              </p>
            </div>

            <div className={styles.caseFrame}>
              <div className={styles.caseQuestion}>
                <span>{t.visitorQuestion}</span>
                <p>{t.blueCaseQuestion}</p>
                <small>{t.blueCaseNote}</small>
              </div>

              <CaseEvidenceLine
                items={BLUE_CASE_OBJECTS}
                ariaLabel={t.blueCaseEvidenceLabel}
                language={language}
              />

              <div className={styles.caseArgument}>
                <span>{t.caseJudgement}</span>
                <p>{t.blueCaseJudgement}</p>
                <small>{t.blueCaseFooter}</small>
              </div>
            </div>
          </article>

          <article className={styles.caseExample} aria-labelledby="landing-cat-case-title">
            <div className={styles.caseHeading}>
              <div>
                <span className={styles.caseIndex}>{t.caseTwoIndex}</span>
                <h2 id="landing-cat-case-title">{t.catCaseTitle}</h2>
              </div>
              <p>
                {t.catCaseLead}
              </p>
            </div>

            <div className={styles.caseFrame}>
              <div className={styles.caseQuestion}>
                <span>{t.visitorQuestion}</span>
                <p>{t.catCaseQuestion}</p>
                <small>{t.catCaseNote}</small>
              </div>

              <CaseEvidenceLine
                items={CAT_CASE_OBJECTS}
                ariaLabel={t.catCaseEvidenceLabel}
                language={language}
              />

              <div className={styles.caseArgument}>
                <span>{t.caseJudgement}</span>
                <p>{t.catCaseJudgement}</p>
                <small>{t.catCaseFooter}</small>
              </div>
            </div>
          </article>

          <button type="button" className={styles.caseCta} onClick={onStart}>
            {t.swapForMine}
            <span aria-hidden="true">→</span>
          </button>
        </section>

        <section className={styles.features} aria-labelledby="feature-title">
          <div className={styles.featureIntro}>
            <h2 id="feature-title">{t.featureTitle}</h2>
            <p>{t.featureLead}</p>
          </div>

          <div className={styles.featureLedger}>
            <article>
              <div className={styles.featureCopy}>
                <span>{t.featureInterviewKicker}</span>
                <h3>{t.featureInterviewTitle}</h3>
                <p>{t.featureInterviewBody}</p>
              </div>
              <div
                className={styles.chatDemo}
                role="img"
                aria-label={t.featureInterviewDiagram}
              >
                <p className={styles.agentBubble}>{t.featureInterviewAsk}</p>
                <p className={styles.visitorBubble}>{t.featureInterviewReply}</p>
                <div><span>{t.chipFirstTime}</span><span>{t.chipTenMinutes}</span><span>{t.chipRegions}</span></div>
              </div>
            </article>

            <article>
              <div className={styles.featureCopy}>
                <span>{t.featureSelectKicker}</span>
                <h3>{t.featureSelectTitle}</h3>
                <p>
                  {fill(t.featureSelectBody, {
                    scope:
                      objectCount !== null
                        ? fill(t.featureSelectScopeCount, { n: objectCount.toLocaleString() })
                        : t.featureSelectScopeUnknown,
                  })}
                </p>
              </div>
              <div
                className={styles.pipelineDemo}
                role="img"
                aria-label={t.featureSelectDiagram}
              >
                {[
                  [t.stepUnderstand, t.stepUnderstandDetail],
                  [t.stepEvidence, t.stepEvidenceDetail],
                  [t.stepRoute, t.stepRouteDetail],
                  [t.stepBuild, t.stepBuildDetail],
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
                <span>{t.featureVisitKicker}</span>
                <h3>{t.featureVisitTitle}</h3>
                <p>{t.featureVisitBody}</p>
              </div>
              <div
                className={styles.visitModes}
                role="img"
                aria-label={t.featureVisitDiagram}
              >
                <div className={styles.mode3d} aria-hidden="true">
                  <span className={styles.modeLabel}>{t.mode3d}</span>
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
                  <small>{t.mode3dCaption}</small>
                </div>
                <div className={styles.mode2d} aria-hidden="true">
                  <span className={styles.modeLabel}>{t.mode2d}</span>
                  <div className={styles.browserPreview}>
                    <div className={styles.browserBar}><i /><i /><i /></div>
                    <div className={styles.browserSheet}>
                      <strong>{t.modeSampleTitle}</strong>
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
                  <small>{t.mode2dCaption}</small>
                </div>
              </div>
            </article>

            <article>
              <div className={styles.featureCopy}>
                <span>{t.featureSourceKicker}</span>
                <h3>{t.featureSourceTitle}</h3>
                <p>{t.featureSourceBody}</p>
              </div>
              <div
                className={styles.labelDemo}
                role="img"
                aria-label={t.featureSourceDiagram}
              >
                <p><span>{t.institutionalRecord}</span>Fritware painted in blue and black over white slip…</p>
                <p><span>{t.curatorialReading}</span>{t.featureSourceReadingSample}</p>
                <dl>
                  <div>
                    <dt>{t.source}</dt>
                    <dd>Art Institute of Chicago</dd>
                  </div>
                  <div>
                    <dt>{t.fieldLicence}</dt>
                    <dd>{t.sampleFieldLicence}</dd>
                  </div>
                </dl>
              </div>
            </article>
          </div>
        </section>

        <section className={styles.method} aria-labelledby="method-title">
          <div className={styles.methodIntro}>
            <div>
              <span className={styles.methodKicker}>{t.methodKicker}</span>
              <h2 id="method-title">{t.methodTitle}</h2>
            </div>
            <p>
              {t.methodLead}
            </p>
          </div>

          <dl className={styles.methodLedger}>
            <div>
              <dt><span>01</span>{t.method01}</dt>
              <dd>{t.method01Body}</dd>
            </div>
            <div>
              <dt><span>02</span>Curatorial Brief</dt>
              <dd>{t.method02Body}</dd>
            </div>
            <div>
              <dt><span>03</span>{t.method03}</dt>
              <dd>{t.method03Body}</dd>
            </div>
            <div>
              <dt><span>04</span>{t.method04}</dt>
              <dd>{t.method04Body}</dd>
            </div>
          </dl>

          <p className={styles.methodBoundary}>
            {t.methodBoundary}
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
                {fill(t.collectionLive, {
                  objects: objectCount.toLocaleString(),
                  museums: institutions.length,
                })}
                <span>{t.collectionLiveSub}</span>
              </>
            ) : collectionStatus === "loading" ? (
              <>
                {t.collectionLoading}
                <span>{t.collectionLoadingSub}</span>
              </>
            ) : (
              <>
                {t.collectionOffline}
                <span>{t.collectionOfflineSub}</span>
              </>
            )}
          </h2>

          {collectionStatus !== "ready" && (
            <p className={styles.collectionLoadNote} role="status">
              {collectionStatus === "loading"
                ? t.collectionLoadingNote
                : t.collectionOfflineNote}
            </p>
          )}

          {collectionStatus === "ready" && objectCount === 0 && (
            <p className={styles.collectionLoadNote} role="status">
              {t.collectionEmpty}
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
              {t.mainTypes}
              {highlights!.topTypes
                .slice(0, 6)
                .map((type) => fill(t.typeWithCount, { label: type.label, count: type.count }))
                .join(" · ")}
            </p>
          )}

          {domains.length > 0 && (
            <p className={styles.collectionNote}>
              {t.collectionNote}
            </p>
          )}
        </section>

        <section id="how-it-works" className={styles.how} aria-labelledby="how-title">
          <h2 id="how-title" className={styles.sectionTitle}>{t.howTitle}</h2>
          <ol>
            <li>
              {/* deslop-ignore-next-line 30 -- a real, ordered visit sequence */}
              <span className={styles.step}>01</span>
              <h3>{t.howTalkTitle}</h3>
              <p>
                {t.howTalkBody}
              </p>
            </li>
            <li>
              {/* deslop-ignore-next-line 30 -- a real, ordered visit sequence */}
              <span className={styles.step}>02</span>
              <h3>{t.howWatchTitle}</h3>
              <p>
                {t.howWatchBody}
              </p>
            </li>
            <li>
              {/* deslop-ignore-next-line 30 -- a real, ordered visit sequence */}
              <span className={styles.step}>03</span>
              <h3>{t.howEnterTitle}</h3>
              <p>
                {t.howEnterBody}
              </p>
            </li>
          </ol>
        </section>

        <section className={styles.boundary} aria-labelledby="boundary-title">
          <h2 id="boundary-title" className={styles.sectionTitle}>{t.boundaryTitle}</h2>
          <dl>
            <div>
              <dt>{t.boundaryObjectsTitle}</dt>
              <dd>
                {t.boundaryObjectsBody}
              </dd>
            </div>
            <div>
              <dt>{t.boundaryTextTitle}</dt>
              <dd>{t.boundaryTextBody}</dd>
            </div>
            <div>
              <dt>{t.boundaryHallTitle}</dt>
              <dd>{t.boundaryHallBody}</dd>
            </div>
            <div>
              <dt>{t.boundaryDemoTitle}</dt>
              <dd>{t.boundaryDemoBody}</dd>
            </div>
          </dl>
        </section>

        <section id="sources" className={styles.credits} aria-labelledby="credits-title">
          <div className={styles.creditsHeading}>
            <h2 id="credits-title">{t.creditsTitle}</h2>
            <p>{t.creditsBody}</p>
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
                    aria-label={fill(t.creditsLinkLabel, { museum: museum.name })}
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
                          {fill(t.creditsObjectLine, {
                            count: summary.objectCount.toLocaleString(),
                            licence:
                              summary.imageLicenses.join(" / ") || t.licencePerObject,
                          })}
                        </small>
                      )}
                    </span>
                  </a>
                </li>
              );
            })}
          </ul>
          <p className={styles.trademarkNote}>
            {t.creditsDisclaimer}
          </p>
        </section>
      </main>

      <footer className={styles.footer}>
        <span>
          {PRODUCT_NAME} · {PRODUCT_NAME_LATIN}
        </span>
        <span className={styles.footerNote}>
          {t.footerNames}
        </span>
      </footer>
    </div>
  );
}
