"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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
import { CURATOR_TITLE } from "@/lib/brand";
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
import { EpilogueConversation } from "./EpilogueConversation";
import styles from "./view2d.module.css";

const TYPE_LABELS: Record<LabelSentence["type"], string> = {
  institution_fact: "机构记录",
  system_inference: "系统推断",
  uncertain: "仍不确定",
};

const SOURCE_KIND_LABELS: Record<string, string> = {
  institution_metadata: "机构元数据",
  institution_curatorial_text: "机构说明",
  institution_provenance: "机构来源记录",
};

type CuratorialEvidenceIndex = Record<string, CuratorialEvidenceReference>;

function EvidenceLinks({
  evidenceIds,
  evidenceIndex,
}: {
  evidenceIds: string[];
  evidenceIndex: CuratorialEvidenceIndex;
}) {
  const references = evidenceIds
    .map((id) => evidenceIndex[id])
    .filter((reference): reference is CuratorialEvidenceReference => Boolean(reference));
  const unresolvedCount = evidenceIds.length - references.length;

  if (references.length === 0) {
    return <span className={styles.noEvidence}>尚未绑定可定位的馆藏记录</span>;
  }

  return (
    <span className={styles.evidenceLinks} aria-label="关联的馆藏记录">
      {references.map((reference, index) => (
        <a key={reference.id} href={`#item-${reference.itemId}`}>
          {String(index + 1).padStart(2, "0")} · {reference.objectTitle} · {reference.sourceLocation}
        </a>
      ))}
      {unresolvedCount > 0 && <span>{unresolvedCount} 条记录暂无法定位</span>}
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
  return (
    <article className={styles.briefClaim} data-confidence={claim.confidence}>
      <div>
        <p>{claim.text}</p>
        <span>{CURATORIAL_CONFIDENCE_LABELS[claim.confidence]}</span>
      </div>
      <EvidenceLinks evidenceIds={claim.evidenceIds} evidenceIndex={evidenceIndex} />
    </article>
  );
}

function CuratorialBriefPanel({ exhibition }: { exhibition: Exhibition }) {
  const brief = exhibition.curatorialBrief;
  if (!brief) return null;

  const evidenceIndex = buildCuratorialEvidenceIndex(exhibition.items);
  const provenanceLabel = PROVENANCE_STATUS_LABELS[brief.ethics.provenanceStatus];
  const ethicsMessages = unresolvedEthicsMessages(brief).filter(
    (message) => message !== provenanceLabel,
  );
  const excludedCandidates = brief.excludedCandidates;

  return (
    <section className={styles.briefSection} aria-labelledby="curatorial-brief-title">
      <details className={styles.briefDetails}>
        <summary>
          <span className={styles.briefSummaryLabel} id="curatorial-brief-title">策展依据</span>
          <strong>{brief.bigIdea.text}</strong>
          <span className={styles.briefSummaryAction}>查看论证、选物与边界</span>
        </summary>

        <div className={styles.briefBody}>
          <p className={styles.briefDisclosure}>
            以下是彦远根据本展馆藏记录形成的策展判断，不是来源机构原话。引用表示可以返回相关记录核对，不代表该判断已经由机构或专家确认。
          </p>

          <section className={styles.briefArgument} aria-labelledby="brief-argument-title">
            <h3 id="brief-argument-title">论证线索</h3>
            <BriefClaim claim={brief.bigIdea} evidenceIndex={evidenceIndex} />
            {brief.keyMessages.map((claim) => (
              <BriefClaim key={claim.id} claim={claim} evidenceIndex={evidenceIndex} />
            ))}
          </section>

          <div className={styles.briefColumns}>
            <section>
              <h3>选物边界</h3>
              <p>
                本展记录了 <strong>{brief.objects.length}</strong> 件入选判断
                {excludedCandidates
                  ? <>；另有 <strong>{excludedCandidates.length}</strong> 件候选未进入最终动线。</>
                  : "。公开版本不包含访客画像与排除候选记录。"}
              </p>
              {excludedCandidates && excludedCandidates.length > 0 && (
                <ul>
                  {excludedCandidates.slice(0, 2).map((candidate) => (
                    <li key={candidate.objectId}>{candidate.title}：{candidate.reason}</li>
                  ))}
                </ul>
              )}
            </section>

            <section className={styles.briefEthics}>
              <h3>来源与伦理状态</h3>
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
              <h3>这份策展仍留下的问题</h3>
              <ul>
                {brief.criticalQuestions.map((question) => <li key={question}>{question}</li>)}
              </ul>
            </section>
          )}

          <p className={styles.briefVersion}>
            {brief.schemaVersion} · {brief.status === "model_refined" ? "模型整理，规则校验" : "规则生成"} ·
            检索 {brief.retrieval.method} {brief.retrieval.selectedCount}/{brief.retrieval.candidateCount} ·
            外部知识{brief.interpretationPolicy.externalKnowledgeAllowed ? "允许" : "未使用"}
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
}: {
  item: ExhibitionItem;
  index: number;
  exhibitionId: string;
  decision?: CuratorialObjectDecision;
  evidenceIndex: CuratorialEvidenceIndex;
}) {
  const [sourceOpen, setSourceOpen] = useState(false);
  const imageLicenseLabel = getImageLicenseLabel(item.object);
  const legacyRightsLabel = getLegacyRightsLabel(item.object);

  return (
    <article className={styles.item} id={`item-${item.id}`}>
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
        <p className={styles.credit}>
          <span>图片：{imageLicenseLabel}</span>
          <a
            href={item.object.objectUrl}
            target="_blank"
            rel="noreferrer"
            onClick={() => logEvent("institution_page_opened", exhibitionId, { itemId: item.id })}
          >
            {item.object.institution || "机构页"} ↗
          </a>
        </p>
      </div>

      <div className={styles.itemText}>
        <p className={styles.itemMeta}>
          <span className={styles.roleTag}>{item.roleLabel}</span>
          <span className={styles.itemNumber}>{String(index + 1).padStart(2, "0")}</span>
          {item.object.evidenceDepth === "thin" && (
            <span className={styles.depthTag}>仅著录信息</span>
          )}
        </p>
        <h3>{item.displayTitle || item.object.titleOriginal || item.object.title}</h3>
        <p className={styles.original}>{item.object.title}</p>
        <p className={styles.tombstone}>
          {[item.object.date, item.object.medium, item.object.culture].filter(Boolean).join(" · ")}
        </p>

        <div className={styles.label}>
          {item.labelSentences.map((sentence) => (
            <p key={sentence.id} data-type={sentence.type}>
              {sentence.text}
              <span className={styles.sentenceType}>{TYPE_LABELS[sentence.type]}</span>
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
            为什么选它？
            <span className={styles.systemJudgement}>系统策展判断</span>
          </summary>
          <p>{decision?.selectionRationale || item.whySelected}</p>
          <p className={styles.relation}>与前后展品的关系：{decision?.relation || item.relation}</p>
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
          {sourceOpen ? "收起来源" : `来源（${item.object.evidence.length} 条）`}
        </button>

        {sourceOpen && (
          <div className={styles.sources}>
            {item.object.evidence.map((chunk) => (
              <div key={chunk.id}>
                <h4>{chunk.sourceLocation}</h4>
                <p>{chunk.text}</p>
                <small>{chunk.sourceTitle}</small>
                {(chunk.license || chunk.sourceKind) && (
                  <p className={styles.sourceRights}>
                    <span>{SOURCE_KIND_LABELS[chunk.sourceKind || ""] || chunk.sourceKind || "字段来源"}</span>
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
              <div><dt>图片</dt><dd>{item.object.imageLicense && item.object.imageRightsUri ? <a href={item.object.imageRightsUri} target="_blank" rel="noreferrer">{imageLicenseLabel} ↗</a> : imageLicenseLabel}</dd></div>
              <div><dt>基础元数据</dt><dd>{item.object.metadataRightsUri ? <a href={item.object.metadataRightsUri} target="_blank" rel="noreferrer">{item.object.metadataLicense} ↗</a> : item.object.metadataLicense || "逐字段见机构记录"}</dd></div>
              {item.object.curatorialTextLicense && (
                <div><dt>机构描述</dt><dd>{item.object.curatorialTextRightsUri ? <a href={item.object.curatorialTextRightsUri} target="_blank" rel="noreferrer">{item.object.curatorialTextLicense} ↗</a> : item.object.curatorialTextLicense}</dd></div>
              )}
              {legacyRightsLabel && (
                <div><dt>旧版综合声明</dt><dd>{item.object.rightsUri ? <a href={item.object.rightsUri} target="_blank" rel="noreferrer">{legacyRightsLabel} ↗</a> : legacyRightsLabel}</dd></div>
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
            alt={exhibition.poster?.altText ?? "AI 生成的展览视觉"}
            aria-hidden={posterLoaded ? undefined : true}
            onLoad={() => setPosterLoaded(true)}
            onError={handlePosterFailure}
          />
        )}

        {!showGeneratedPoster && fallbackItems.length > 0 && (
          <div
            className={styles.posterCollage}
            role="img"
            aria-label="本展所选馆藏公开图像拼贴，用作入口视觉回退"
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
          <div className={styles.posterUnavailable}>主题画面暂不可用</div>
        )}

        {!showGeneratedPoster && (
          <div className={styles.posterFallbackCopy} aria-hidden="true">
            <small>卧游 · AI 策展人彦远</small>
            <strong>{exhibition.title}</strong>
            {exhibition.subtitle && <span>{exhibition.subtitle}</span>}
          </div>
        )}
      </div>
      <figcaption>
        {showGeneratedPoster
          ? "AI 生成展览海报；主题画面由模型生成，标题由系统精确排版"
          : "馆藏公开图像拼贴与系统排版回退；图像许可见各展品来源"}
      </figcaption>
    </figure>
  );
}

export function ExhibitionView2D({
  exhibition,
  onEnterHall,
  webglAvailable,
}: {
  exhibition: Exhibition;
  onEnterHall?: () => void;
  webglAvailable?: boolean;
}) {
  const itemsById = new Map(exhibition.items.map((item) => [item.id, item]));
  const posterUrl = exhibition.poster?.backgroundUrl
    ? resolveApiAssetUrl(exhibition.poster.backgroundUrl)
    : null;
  const evidenceIndex = buildCuratorialEvidenceIndex(exhibition.items);
  const generationNotice = generationFallbackNotice(exhibition.versions.provider);
  let runningIndex = 0;

  return (
    <div className={styles.page}>
      <a href="#exhibition-body" className={styles.skipLink}>
        跳到展览内容
      </a>

      <header className={styles.hero}>
        <ExhibitionPosterVisual
          key={posterUrl ?? "collection-collage"}
          exhibition={exhibition}
          posterUrl={posterUrl}
        />
        <div>
          <p className={styles.eyebrow}>{CURATOR_TITLE}为你策展 · 文字版</p>
          <h1>{exhibition.title}</h1>
          {exhibition.subtitle && <p className={styles.subtitle}>{exhibition.subtitle}</p>}
          <p className={styles.thesis}>{exhibition.curatorialThesis}</p>
          <p className={styles.coreAnswer}>{exhibition.coreAnswer}</p>

          <div className={styles.heroActions}>
            {onEnterHall && webglAvailable && (
              <button type="button" className={styles.primary} onClick={onEnterHall}>
                进入 3D 展厅
              </button>
            )}
            <Link href="/">重新策展</Link>
          </div>

          {onEnterHall && !webglAvailable && (
            <p className={styles.webglNote}>
              当前设备不支持 3D 展厅（缺少 WebGL 或显卡能力不足），已为你打开完整的文字版本。
            </p>
          )}
        </div>
      </header>

      <CuratorialBriefPanel exhibition={exhibition} />

      <nav className={styles.toc} aria-label="展览目录">
        <ol>
          {exhibition.chapters.map((chapter) => (
            <li key={chapter.id}>
              <a href={`#chapter-${chapter.id}`}>{chapter.title}</a>
            </li>
          ))}
          <li>
            <a href="#epilogue">结语</a>
          </li>
        </ol>
      </nav>

      <main id="exhibition-body">
        {exhibition.chapters.map((chapter) => (
          <section key={chapter.id} className={styles.chapter} id={`chapter-${chapter.id}`}>
            <header className={styles.chapterHead}>
              <p className={styles.eyebrow}>
                第 {chapter.order + 1} 部分 / 共 {exhibition.chapters.length}
              </p>
              <h2>{chapter.title}</h2>
              <p>{chapter.leadIn}</p>
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
                />
              );
            })}
          </section>
        ))}

        <section className={styles.epilogue} id="epilogue">
          <h2>结语</h2>
          <p className={styles.epilogueText}>{exhibition.epilogue.text}</p>
          <EpilogueConversation
            exhibitionId={exhibition.id}
            openQuestions={exhibition.epilogue.openQuestions}
          />
          <h2>这场展览的材料边界</h2>
          <ul className={styles.boundary}>
            {exhibition.epilogue.materialBoundary.map((limit) => (
              <li key={limit}>{limit}</li>
            ))}
          </ul>
        </section>

        <section className={styles.versions}>
          <h2>生成记录</h2>
          <dl>
            <div>
              <dt>实际路径</dt>
              <dd>{generationProviderLabel(exhibition.versions.provider)}</dd>
            </div>
            <div>
              <dt>配置模型</dt>
              <dd>{exhibition.versions.model}</dd>
            </div>
            <div>
              <dt>提示</dt>
              <dd>{exhibition.versions.prompt}</dd>
            </div>
            <div>
              <dt>馆藏</dt>
              <dd>{exhibition.versions.collection}</dd>
            </div>
            <div>
              <dt>检查器</dt>
              <dd>{exhibition.versions.validator}</dd>
            </div>
          </dl>
          {generationNotice && (
            <p className={styles.generationFallback} role="note">
              {generationNotice}
            </p>
          )}
          <p>
            本展由 {CURATOR_TITLE} 生成。展品与说明可追溯到机构公开馆藏；策展解释不代表来源机构立场。
          </p>
        </section>
      </main>
    </div>
  );
}
