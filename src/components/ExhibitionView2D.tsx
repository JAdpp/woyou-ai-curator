"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import type {
  CuratorialBriefClaim,
  CuratorialObjectDecision,
  Exhibition,
  ExhibitionItem,
  LabelSentence,
} from "@/lib/types";
import { logEvent, resolveApiAssetUrl, resolveObjectImageUrl } from "@/lib/api";
import { fill } from "@/lib/i18n";
import { useLanguage } from "@/lib/useLanguage";
import {
  buildCuratorialEvidenceIndex,
  CURATORIAL_CONFIDENCE_LABELS,
  findCuratorialDecision,
  PROVENANCE_STATUS_LABELS,
  unresolvedEthicsMessages,
  type CuratorialEvidenceReference,
} from "@/lib/curatorialBrief";
import {
  generationFallbackNotice,
  generationProviderLabel,
} from "@/lib/generationRecord";
import { getImageLicenseLabel, getLegacyRightsLabel } from "@/lib/rights";
import {
  publicInstitutionName,
  publicObjectMetadata,
  publicObjectTitle,
  visitorRoleTag,
} from "@/lib/localizedMetadata";
import { EpilogueConversation } from "./EpilogueConversation";
import { buildHallLayout } from "./hall/layout";
import { clampStopIndex, stopAnchor } from "./hall/progress";
import styles from "./view2d.module.css";

type ViewCopy = ReturnType<typeof useLanguage>["t"]["view"];

function typeLabel(type: LabelSentence["type"], t: ViewCopy): string {
  if (type === "institution_fact") return t.sentenceInstitutionFact;
  if (type === "visual_observation") return t.sentenceVisualObservation;
  if (type === "uncertain") return t.sentenceUncertain;
  return t.sentenceSystemInference;
}

function sourceKindLabel(kind: string, t: ViewCopy): string {
  if (kind === "institution_metadata") return t.sourceMetadata;
  if (kind === "institution_curatorial_text") return t.sourceCuratorialText;
  if (kind === "institution_provenance") return t.sourceProvenance;
  if (kind === "collection_image") return t.sourceCollectionImage;
  return "";
}

type CuratorialEvidenceIndex = Record<string, CuratorialEvidenceReference>;

function EvidenceLinks({
  evidenceIds,
  evidenceIndex,
}: {
  evidenceIds: string[];
  evidenceIndex: CuratorialEvidenceIndex;
}) {
  const { t: copy } = useLanguage();
  const t = copy.view;
  const references = evidenceIds
    .map((id) => evidenceIndex[id])
    .filter((reference): reference is CuratorialEvidenceReference => Boolean(reference));
  const unresolvedCount = evidenceIds.length - references.length;

  if (references.length === 0) {
    return <span className={styles.noEvidence}>{t.noEvidence}</span>;
  }

  return (
    <span className={styles.evidenceLinks} aria-label={t.linkedRecords}>
      {references.map((reference, index) => (
        <a key={reference.id} href={`#item-${reference.itemId}`}>
          {String(index + 1).padStart(2, "0")} · {reference.objectTitle} · {reference.sourceLocation}
        </a>
      ))}
      {unresolvedCount > 0 && <span>{fill(t.unresolvedRecords, { n: unresolvedCount })}</span>}
    </span>
  );
}

function BriefClaim({
  claim,
  evidenceIndex,
}: {
  claim: CuratorialBriefClaim;
  evidenceIndex: CuratorialEvidenceIndex;
}) {
  const { language } = useLanguage();
  return (
    <article className={styles.briefClaim} data-confidence={claim.confidence}>
      <div>
        <p>{claim.text}</p>
        <span>{CURATORIAL_CONFIDENCE_LABELS[claim.confidence][language]}</span>
      </div>
      <EvidenceLinks evidenceIds={claim.evidenceIds} evidenceIndex={evidenceIndex} />
    </article>
  );
}

function CuratorialBriefPanel({ exhibition }: { exhibition: Exhibition }) {
  const { language, t: copy } = useLanguage();
  const t = copy.view;
  const brief = exhibition.curatorialBrief;
  if (!brief) return null;

  const evidenceIndex = buildCuratorialEvidenceIndex(exhibition.items);
  const provenanceLabel = PROVENANCE_STATUS_LABELS[brief.ethics.provenanceStatus][language];
  const ethicsMessages = unresolvedEthicsMessages(brief, language).filter(
    (message) => message !== provenanceLabel,
  );
  const excludedCandidates = brief.excludedCandidates;

  return (
    <section className={styles.briefSection} aria-labelledby="curatorial-brief-title">
      <details className={styles.briefDetails}>
        <summary>
          <span className={styles.briefSummaryLabel} id="curatorial-brief-title">{t.briefLabel}</span>
          <strong>{brief.bigIdea.text}</strong>
          <span className={styles.briefSummaryAction}>{t.briefAction}</span>
        </summary>

        <div className={styles.briefBody}>
          <p className={styles.briefDisclosure}>
            {t.briefDisclosure}
          </p>

          <section className={styles.briefArgument} aria-labelledby="brief-argument-title">
            <h3 id="brief-argument-title">{t.briefArgument}</h3>
            <BriefClaim claim={brief.bigIdea} evidenceIndex={evidenceIndex} />
            {brief.keyMessages.map((claim) => (
              <BriefClaim key={claim.id} claim={claim} evidenceIndex={evidenceIndex} />
            ))}
          </section>

          <div className={styles.briefColumns}>
            <section>
              <h3>{t.briefSelection}</h3>
              <p>
                {fill(t.briefSelectionCount, { n: brief.objects.length })}
                {excludedCandidates
                  ? fill(t.briefExcludedSuffix, { n: excludedCandidates.length })
                  : t.briefNoExcluded}
              </p>
              {excludedCandidates && excludedCandidates.length > 0 && (
                <ul>
                  {excludedCandidates.slice(0, 2).map((candidate) => (
                    <li key={candidate.objectId}>{candidate.title}: {candidate.reason}</li>
                  ))}
                </ul>
              )}
            </section>

            <section className={styles.briefEthics}>
              <h3>{t.briefEthics}</h3>
              <p className={styles.reviewPending}>
                {provenanceLabel}
              </p>
              <ul>
                {ethicsMessages.map((message) => <li key={message}>{message}</li>)}
                {brief.ethics.culturalSensitivity.map((note) => <li key={note}>{note}</li>)}
              </ul>
            </section>
          </div>

          {brief.criticalQuestions.length > 0 && (
            <section className={styles.briefQuestions}>
              <h3>{t.briefOpen}</h3>
              <ul>
                {brief.criticalQuestions.map((question) => <li key={question}>{question}</li>)}
              </ul>
            </section>
          )}

          <p className={styles.briefVersion}>
            {brief.schemaVersion} ·{" "}
            {brief.status === "model_refined" ? t.briefModelRefined : t.briefRuleGenerated} ·{" "}
            {t.briefRetrieval} {brief.retrieval.method} {brief.retrieval.selectedCount}/{brief.retrieval.candidateCount} ·{" "}
            {t.briefExternalKnowledge}{" "}
            {brief.interpretationPolicy.externalKnowledgeAllowed
              ? t.briefExternalAllowed
              : t.briefExternalUnused}
          </p>
        </div>
      </details>
    </section>
  );
}

/**
 * Accessible text version of the exhibition.
 *
 * This is not a degraded mode — it is where keyboard and screen-reader users
 * get the full exhibition, because a WebGL canvas is opaque to assistive tech.
 * It is also the fallback when WebGL is unavailable, and what search engines
 * and link previews see.
 */
function ItemBlock({
  item,
  index,
  exhibitionId,
  decision,
  evidenceIndex,
  tourStopIndex,
  contentLang,
}: {
  item: ExhibitionItem;
  index: number;
  exhibitionId: string;
  decision?: CuratorialObjectDecision;
  evidenceIndex: CuratorialEvidenceIndex;
  tourStopIndex: number;
  contentLang: string;
}) {
  const { language, t: copy } = useLanguage();
  const t = copy.view;
  const [sourceOpen, setSourceOpen] = useState(false);
  const imageLicenseLabel = getImageLicenseLabel(item.object);
  const legacyRightsLabel = getLegacyRightsLabel(item.object);
  const publicTitle = publicObjectTitle(item, contentLang);
  const publicMetadata = publicObjectMetadata(item, contentLang);
  const institutionName = publicInstitutionName(item, contentLang);

  return (
    <article
      className={styles.item}
      id={`item-${item.id}`}
      data-tour-stop={tourStopIndex}
      tabIndex={-1}
    >
      <div className={styles.itemVisual}>
        {/* Institution-hosted image, served via our proxy at texture size.
            Never generated or redrawn. */}
        <Image
          src={resolveObjectImageUrl(item.object.id, 1024)}
          alt={item.object.altText}
          width={900}
          height={900}
          sizes="(max-width: 900px) 90vw, 440px"
          unoptimized
        />
        <p className={styles.altTextSource}>
          {item.object.altTextSource === "institution_authored"
            ? language === "en"
              ? "Image description supplied by the holding institution."
              : "图像替代文字由来源机构提供。"
            : language === "en"
              ? "Alternative text is assembled from catalogue fields; it is not a visually verified description."
              : "替代文字由题名、年代等著录字段组成，并非经视觉核验的图像描述。"}
        </p>
        <p className={styles.credit}>
          <span>{t.image}: {imageLicenseLabel}</span>
          <a
            href={item.object.objectUrl}
            target="_blank"
            rel="noreferrer"
            onClick={() => logEvent("institution_page_opened", exhibitionId, { itemId: item.id })}
          >
            {institutionName || (contentLang.startsWith("zh") ? t.institutionPage : item.object.institution) || t.institutionPage} ↗
          </a>
        </p>
      </div>

      <div className={styles.itemText}>
        <p className={styles.itemMeta}>
          <span className={styles.roleTag}>{visitorRoleTag(item, contentLang)}</span>
          <span className={styles.itemNumber}>{String(index + 1).padStart(2, "0")}</span>
          {item.object.evidenceDepth === "thin" && (
            <span className={styles.depthTag}>{t.tombstoneOnly}</span>
          )}
        </p>
        <h3 lang={contentLang}>{publicTitle}</h3>
        {item.object.title !== publicTitle && (
          <p className={styles.original}>{item.object.title}</p>
        )}
        <p className={styles.tombstone}>
          {publicMetadata.length > 0
            ? publicMetadata.join(" · ")
            : contentLang.startsWith("zh")
              ? "中文著录暂缺，馆方原文见来源"
              : ""}
        </p>

        <div className={styles.label} lang={contentLang}>
          {item.labelSentences.map((sentence) => (
            <p key={sentence.id} data-type={sentence.type}>
              {sentence.text}
              <span className={styles.sentenceType}>{typeLabel(sentence.type, t)}</span>
            </p>
          ))}
        </div>

        <details
          className={styles.why}
          onToggle={(event) =>
            event.currentTarget.open && logEvent("why_selected_opened", exhibitionId, { itemId: item.id })
          }
        >
          <summary>
            {t.whySelected}
            <span className={styles.systemJudgement}>{t.systemJudgement}</span>
          </summary>
          <p>{decision?.selectionRationale || item.whySelected}</p>
          <p className={styles.relation}>{t.relationPrefix}{decision?.relation || item.relation}</p>
          {decision && (
            <EvidenceLinks evidenceIds={decision.evidenceIds} evidenceIndex={evidenceIndex} />
          )}
        </details>

        <button
          type="button"
          className={styles.sourceToggle}
          aria-expanded={sourceOpen}
          onClick={() => {
            setSourceOpen((open) => !open);
            if (!sourceOpen) logEvent("source_opened", exhibitionId, { itemId: item.id });
          }}
        >
          {sourceOpen ? t.collapseSources : fill(t.expandSources, { n: item.object.evidence.length })}
        </button>

        {sourceOpen && (
          <div className={styles.sources}>
            {item.object.evidence.map((chunk) => (
              <div key={chunk.id}>
                <h4>{chunk.sourceLocation}</h4>
                <p>{chunk.sourceKind === "collection_image" ? t.sourceCollectionImageDescription : chunk.text}</p>
                <small>{chunk.sourceTitle}</small>
                {(chunk.license || chunk.sourceKind) && (
                  <p className={styles.sourceRights}>
                    <span>{sourceKindLabel(chunk.sourceKind || "", t) || chunk.sourceKind || t.sourceFieldFallback}</span>
                    {chunk.license && chunk.rightsUri ? (
                      <a href={chunk.rightsUri} target="_blank" rel="noreferrer">{chunk.license} ↗</a>
                    ) : (
                      <span>{chunk.license}</span>
                    )}
                  </p>
                )}
              </div>
            ))}
            <dl className={styles.fieldRights}>
              <div><dt>{t.image}</dt><dd>{item.object.imageLicense && item.object.imageRightsUri ? <a href={item.object.imageRightsUri} target="_blank" rel="noreferrer">{imageLicenseLabel} ↗</a> : imageLicenseLabel}</dd></div>
              <div><dt>{t.baseMetadata}</dt><dd>{item.object.metadataRightsUri ? <a href={item.object.metadataRightsUri} target="_blank" rel="noreferrer">{item.object.metadataLicense} ↗</a> : item.object.metadataLicense || t.perFieldSeeRecord}</dd></div>
              {item.object.curatorialTextLicense && (
                <div><dt>{t.institutionDescription}</dt><dd>{item.object.curatorialTextRightsUri ? <a href={item.object.curatorialTextRightsUri} target="_blank" rel="noreferrer">{item.object.curatorialTextLicense} ↗</a> : item.object.curatorialTextLicense}</dd></div>
              )}
              {legacyRightsLabel && (
                <div><dt>{t.legacyRights}</dt><dd>{item.object.rightsUri ? <a href={item.object.rightsUri} target="_blank" rel="noreferrer">{legacyRightsLabel} ↗</a> : legacyRightsLabel}</dd></div>
              )}
            </dl>
          </div>
        )}
      </div>
    </article>
  );
}

function ExhibitionPosterVisual({
  exhibition,
  posterUrl,
}: {
  exhibition: Exhibition;
  posterUrl: string | null;
}) {
  const { t: copy } = useLanguage();
  const t = copy.view;
  const brandLine = `${copy.brand.productName} · ${copy.brand.curatorTitle}`;
  const [posterLoaded, setPosterLoaded] = useState(false);
  const [posterFailed, setPosterFailed] = useState(false);
  const posterImageRef = useRef<HTMLImageElement>(null);
  const failureLoggedRef = useRef(false);
  const fallbackItems = exhibition.items.slice(0, 4);
  const showGeneratedPoster = Boolean(posterUrl && posterLoaded && !posterFailed);

  const handlePosterFailure = useCallback(() => {
    setPosterFailed(true);
    setPosterLoaded(false);
    if (!failureLoggedRef.current) {
      failureLoggedRef.current = true;
      void logEvent("poster_load_failed", exhibition.id, { posterUrl });
    }
  }, [exhibition.id, posterUrl]);

  // A cached image can finish before React hydrates, so onLoad/onError alone
  // cannot decide whether to reveal the generated poster.
  useEffect(() => {
    const image = posterImageRef.current;
    if (!posterUrl || !image?.complete) return;
    if (image.naturalWidth > 0) setPosterLoaded(true);
    else handlePosterFailure();
  }, [handlePosterFailure, posterUrl]);

  return (
    <figure className={styles.poster}>
      <div className={styles.posterVisual}>
        {posterUrl && !posterFailed && (
          // eslint-disable-next-line @next/next/no-img-element -- generated asset is served by the API, with an explicit collection-image fallback
          <img
            ref={posterImageRef}
            className={styles.posterImage}
            data-loaded={posterLoaded ? "true" : "false"}
            src={posterUrl}
            alt={exhibition.poster?.altText ?? t.posterAlt}
            aria-hidden={posterLoaded ? undefined : true}
            onLoad={() => setPosterLoaded(true)}
            onError={handlePosterFailure}
          />
        )}

        {!showGeneratedPoster && fallbackItems.length > 0 && (
          <div
            className={styles.posterCollage}
            role="img"
            aria-label={t.posterCollageAlt}
          >
            {fallbackItems.map((item, index) => (
              <Image
                key={item.id}
                src={resolveObjectImageUrl(item.object.id, 512)}
                alt=""
                width={360}
                height={360}
                sizes="136px"
                loading={index === 0 ? "eager" : "lazy"}
                unoptimized
              />
            ))}
          </div>
        )}

        {!showGeneratedPoster && fallbackItems.length === 0 && (
          <div className={styles.posterUnavailable}>{t.posterUnavailable}</div>
        )}

        {!showGeneratedPoster && (
          <div className={styles.posterFallbackCopy} aria-hidden="true">
            <small>{brandLine}</small>
            <strong>{exhibition.title}</strong>
            {exhibition.subtitle && <span>{exhibition.subtitle}</span>}
          </div>
        )}
      </div>
      <figcaption>
        {showGeneratedPoster
          ? t.posterCaptionAi
          : t.posterCaptionCollage}
      </figcaption>
    </figure>
  );
}

export function ExhibitionView2D({
  exhibition,
  onEnterHall,
  webglAvailable,
  initialStopIndex = 0,
  restoreInitialFocus = false,
  onActiveStopChange,
  hallRuntimeFailed = false,
}: {
  exhibition: Exhibition;
  onEnterHall?: () => void;
  webglAvailable?: boolean;
  initialStopIndex?: number;
  restoreInitialFocus?: boolean;
  onActiveStopChange?: (stopIndex: number) => void;
  hallRuntimeFailed?: boolean;
}) {
  const { language, t: copy } = useLanguage();
  const t = copy.view;
  // Labels and curatorial prose were written once, when the exhibition was
  // generated. Switching the interface afterwards cannot retranslate them, so
  // say so rather than wrapping one language in the other and hoping.
  const writtenIn = exhibition.visitorProfile?.language ?? "zh";
  const languageMismatch = writtenIn !== language;
  const itemsById = new Map(exhibition.items.map((item) => [item.id, item]));
  const posterUrl = exhibition.poster?.backgroundUrl
    ? resolveApiAssetUrl(exhibition.poster.backgroundUrl)
    : null;
  const evidenceIndex = buildCuratorialEvidenceIndex(exhibition.items);
  const generationNotice = generationFallbackNotice(exhibition.versions.provider);
  const layout = useMemo(() => buildHallLayout(exhibition), [exhibition]);
  const safeInitialStopIndex = clampStopIndex(layout.stops.length, initialStopIndex);
  const mainRef = useRef<HTMLElement>(null);
  const pageRef = useRef<HTMLDivElement>(null);
  const lastVisibleStopRef = useRef(safeInitialStopIndex);
  const restoredFocusRef = useRef(false);
  const contentLang = writtenIn === "en" ? "en" : "zh-CN";
  const chapterStopIndex = useMemo(() => {
    const index = new Map<string, number>();
    layout.stops.forEach((stop, stopIndex) => {
      if (stop.kind === "chapter" && stop.chapterId) index.set(stop.chapterId, stopIndex);
    });
    return index;
  }, [layout.stops]);
  const itemStopIndex = useMemo(() => {
    const index = new Map<string, number>();
    layout.stops.forEach((stop, stopIndex) => {
      if (stop.kind === "artwork" && stop.itemId) index.set(stop.itemId, stopIndex);
    });
    return index;
  }, [layout.stops]);
  const epilogueStopIndex = layout.stops.findIndex((stop) => stop.kind === "epilogue");
  let runningIndex = 0;

  // Returning from 3D lands on the same chapter or object and places keyboard
  // focus there. Stop zero is left alone so a fresh 2D visit keeps the normal
  // browser focus order.
  useEffect(() => {
    if (!restoreInitialFocus || restoredFocusRef.current || safeInitialStopIndex <= 0) return;
    const anchor = stopAnchor(layout.stops[safeInitialStopIndex]);
    if (!anchor) return;
    restoredFocusRef.current = true;
    const frame = window.requestAnimationFrame(() => {
      const target = document.getElementById(anchor);
      target?.scrollIntoView({ block: "start" });
      target?.focus({ preventScroll: true });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [exhibition.id, layout.stops, restoreInitialFocus, safeInitialStopIndex]);

  // A lightweight scroll spy keeps the 3D resume point aligned with what the
  // reader is actually looking at in the long-form page.
  useEffect(() => {
    const root = pageRef.current;
    if (!root || !onActiveStopChange) return;
    let frame = 0;
    const update = () => {
      frame = 0;
      const nodes = Array.from(root.querySelectorAll<HTMLElement>("[data-tour-stop]"));
      if (nodes.length === 0) return;
      const readingLine = window.innerHeight * 0.34;
      let active = nodes[0];
      for (const node of nodes) {
        if (node.getBoundingClientRect().top <= readingLine) active = node;
        else break;
      }
      const next = clampStopIndex(layout.stops.length, active.dataset.tourStop);
      if (next === lastVisibleStopRef.current) return;
      lastVisibleStopRef.current = next;
      onActiveStopChange(next);
    };
    const schedule = () => {
      if (!frame) frame = window.requestAnimationFrame(update);
    };
    if (restoreInitialFocus && safeInitialStopIndex > 0) schedule();
    else update();
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [layout.stops.length, onActiveStopChange, restoreInitialFocus, safeInitialStopIndex]);

  return (
    <div className={styles.page} ref={pageRef}>
      <a
        href="#exhibition-body"
        className={styles.skipLink}
        onClick={(event) => {
          event.preventDefault();
          mainRef.current?.focus({ preventScroll: true });
          mainRef.current?.scrollIntoView({ block: "start" });
        }}
      >
        {t.skipToContent}
      </a>

      <header className={styles.hero} data-tour-stop={0}>
        <ExhibitionPosterVisual
          key={posterUrl ?? "collection-collage"}
          exhibition={exhibition}
          posterUrl={posterUrl}
        />
        <div>
          <p className={styles.eyebrow}>{t.eyebrow}</p>
          {languageMismatch && (
            <p className={styles.languageNotice} role="note">
              <strong>{copy.exhibition.languageNoticeTitle}</strong>
              {copy.exhibition.languageNoticeBody}
            </p>
          )}
          <h1 lang={contentLang}>{exhibition.title}</h1>
          {exhibition.subtitle && <p className={styles.subtitle} lang={contentLang}>{exhibition.subtitle}</p>}
          <p className={styles.thesis} lang={contentLang}>{exhibition.curatorialThesis}</p>
          <p className={styles.coreAnswer} lang={contentLang}>{exhibition.coreAnswer}</p>

          <div className={styles.heroActions}>
            {onEnterHall && webglAvailable && (
              <button type="button" className={styles.primary} onClick={onEnterHall}>
                {t.enterHall}
              </button>
            )}
            <Link href="/">{t.curateAgain}</Link>
          </div>

          {onEnterHall && !webglAvailable && (
            <p className={styles.webglNote}>
              {t.noWebgl}
            </p>
          )}
          {onEnterHall && hallRuntimeFailed && webglAvailable && (
            <p className={styles.webglNote} role="alert">
              {language === "en"
                ? "The 3D hall could not finish loading, so this exhibition has returned to the complete 2D version. Your place is preserved; you can try the hall again."
                : "3D 展厅没有完整载入，已回到同一展览的完整 2D 版本。当前位置已保留，你也可以再次尝试进入。"}
            </p>
          )}
        </div>
      </header>

      <CuratorialBriefPanel exhibition={exhibition} />

      <nav className={styles.toc} aria-label={t.toc}>
        <ol>
          {exhibition.chapters.map((chapter) => (
            <li key={chapter.id}>
              <a href={`#chapter-${chapter.id}`}>{chapter.title}</a>
            </li>
          ))}
          <li>
            <a href="#epilogue">{t.epilogue}</a>
          </li>
        </ol>
      </nav>

      <main id="exhibition-body" ref={mainRef} tabIndex={-1}>
        {exhibition.chapters.map((chapter) => (
          <section
            key={chapter.id}
            className={styles.chapter}
            id={`chapter-${chapter.id}`}
            data-tour-stop={chapterStopIndex.get(chapter.id) ?? 0}
            tabIndex={-1}
          >
            <header className={styles.chapterHead}>
              <p className={styles.eyebrow}>
                {fill(t.chapterOf, { n: chapter.order + 1, total: exhibition.chapters.length })}
              </p>
              <h2 lang={contentLang}>{chapter.title}</h2>
              <p lang={contentLang}>{chapter.leadIn}</p>
            </header>
            {chapter.itemIds.map((itemId) => {
              const item = itemsById.get(itemId);
              if (!item) return null;
              return (
                <ItemBlock
                  key={item.id}
                  item={item}
                  index={runningIndex++}
                  exhibitionId={exhibition.id}
                  decision={findCuratorialDecision(exhibition.curatorialBrief, item)}
                  evidenceIndex={evidenceIndex}
                  tourStopIndex={itemStopIndex.get(item.id) ?? 0}
                  contentLang={contentLang}
                />
              );
            })}
          </section>
        ))}

        <section
          className={styles.epilogue}
          id="epilogue"
          data-tour-stop={epilogueStopIndex >= 0 ? epilogueStopIndex : layout.stops.length - 1}
          tabIndex={-1}
        >
          <h2>{t.epilogue}</h2>
          <p className={styles.epilogueText} lang={contentLang}>{exhibition.epilogue.text}</p>
          <EpilogueConversation
            exhibitionId={exhibition.id}
            openQuestions={exhibition.epilogue.openQuestions}
          />
          <h2>{t.materialLimits}</h2>
          <ul className={styles.boundary}>
            {exhibition.epilogue.materialBoundary.map((limit) => (
              <li key={limit}>{limit}</li>
            ))}
          </ul>
        </section>

        <section className={styles.versions}>
          <h2>{t.generationRecord}</h2>
          <dl>
            <div>
              <dt>{t.actualPath}</dt>
              <dd>{generationProviderLabel(exhibition.versions.provider)}</dd>
            </div>
            <div>
              <dt>{t.configuredModel}</dt>
              <dd>{exhibition.versions.model}</dd>
            </div>
            {exhibition.versions.labelsModel && (
              <div>
                <dt>{t.configuredLabelsModel}</dt>
                <dd>{exhibition.versions.labelsModel}</dd>
              </div>
            )}
            <div>
              <dt>{t.promptVersion}</dt>
              <dd>{exhibition.versions.prompt}</dd>
            </div>
            <div>
              <dt>{t.collection}</dt>
              <dd>{exhibition.versions.collection}</dd>
            </div>
            <div>
              <dt>{t.validator}</dt>
              <dd>{exhibition.versions.validator}</dd>
            </div>
          </dl>
          {generationNotice && (
            <p className={styles.generationFallback} role="note">
              {generationNotice}
            </p>
          )}
          <p>
            {t.footer}
          </p>
        </section>
      </main>
    </div>
  );
}
