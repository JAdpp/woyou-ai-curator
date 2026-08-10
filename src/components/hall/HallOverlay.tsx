"use client";

import {
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { Exhibition, ExhibitionItem, LabelSentence } from "@/lib/types";
import { logEvent, resolveApiAssetUrl, resolveObjectImageUrl } from "@/lib/api";
import { getImageLicenseLabel, getLegacyRightsLabel } from "@/lib/rights";
import { EpilogueConversation } from "../EpilogueConversation";
import type { HallLayout, TourStop } from "./layout";
import { useAudioGuide } from "./useAudioGuide";
import styles from "./hall.module.css";

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

type PanelPosition = { x: number; y: number };
type PanelSize = { width: number; height: number };

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(value, minimum), Math.max(minimum, maximum));
}

function clampPanelPosition(
  x: number,
  y: number,
  width: number,
  height: number,
  viewportWidth: number,
  viewportHeight: number,
): PanelPosition {
  return {
    x: clamp(x, 8, viewportWidth - width - 8),
    y: clamp(y, 8, viewportHeight - height - 8),
  };
}

function LobbyPoster({ exhibition }: { exhibition: Exhibition }) {
  const generatedUrl = exhibition.poster?.backgroundUrl
    ? resolveApiAssetUrl(exhibition.poster.backgroundUrl)
    : null;
  const fallbackObject = exhibition.items[0]?.object ?? null;
  const [source, setSource] = useState<"generated" | "collection" | "none">(
    generatedUrl ? "generated" : fallbackObject ? "collection" : "none",
  );

  const imageUrl = source === "generated"
    ? generatedUrl
    : source === "collection" && fallbackObject
      ? resolveObjectImageUrl(fallbackObject.id, 1024)
      : null;

  if (!imageUrl) return null;

  const isGenerated = source === "generated";
  return (
    <figure className={styles.lobbyPoster}>
      <div className={styles.posterFrame} data-source={source}>
        {/* eslint-disable-next-line @next/next/no-img-element -- generated and proxied museum imagery are runtime URLs */}
        <img
          src={imageUrl}
          alt={
            isGenerated
              ? exhibition.poster?.altText ?? "AI 生成的展览海报"
              : `${fallbackObject?.title ?? exhibition.title}，馆藏公开图像回退`
          }
          onError={() => {
            if (source === "generated" && fallbackObject) setSource("collection");
            else setSource("none");
          }}
        />
        {!isGenerated && (
          <div className={styles.posterFallbackCopy} aria-hidden="true">
            <small>卧游 · AI 策展人彦远</small>
            <strong>{exhibition.title}</strong>
            {exhibition.subtitle && <span>{exhibition.subtitle}</span>}
          </div>
        )}
      </div>
      <figcaption className={styles.posterCaption}>
        {isGenerated
          ? "AI 生成主题画面 · 系统精确排版"
          : "生图暂不可用 · 馆藏公开图像与系统排版回退"}
      </figcaption>
    </figure>
  );
}

/**
 * Label card for one artwork.
 *
 * Split out and keyed by item id so its expanded/collapsed source panel is
 * reset by remounting rather than by an effect writing state on every stop.
 */
function ArtworkLabel({
  item,
  exhibitionId,
}: {
  item: ExhibitionItem;
  exhibitionId: string;
}) {
  const [sourceOpen, setSourceOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [position, setPosition] = useState<PanelPosition | null>(null);
  const [size, setSize] = useState<PanelSize | null>(null);
  const panelRef = useRef<HTMLElement>(null);
  const dragState = useRef<{
    pointerId: number;
    originX: number;
    originY: number;
    panelX: number;
    panelY: number;
    width: number;
    height: number;
  } | null>(null);
  const imageLicenseLabel = getImageLicenseLabel(item.object);
  const legacyRightsLabel = getLegacyRightsLabel(item.object);

  const clampPosition = (x: number, y: number, width: number, height: number) =>
    clampPanelPosition(x, y, width, height, window.innerWidth, window.innerHeight);

  useLayoutEffect(() => {
    const keepPanelOnScreen = () => {
      if (!panelRef.current) return;
      const rect = panelRef.current.getBoundingClientRect();
      setPosition((current) => {
        if (!current) return current;
        const next = clampPanelPosition(
          current.x,
          current.y,
          rect.width,
          rect.height,
          window.innerWidth,
          window.innerHeight,
        );
        return next.x === current.x && next.y === current.y ? current : next;
      });
    };

    keepPanelOnScreen();
    window.addEventListener("resize", keepPanelOnScreen);
    return () => window.removeEventListener("resize", keepPanelOnScreen);
  }, [collapsed, size]);

  const startDrag = (event: ReactPointerEvent<HTMLButtonElement>) => {
    if (event.button !== 0 || !panelRef.current) return;
    const rect = panelRef.current.getBoundingClientRect();
    dragState.current = {
      pointerId: event.pointerId,
      originX: event.clientX,
      originY: event.clientY,
      panelX: rect.left,
      panelY: rect.top,
      width: rect.width,
      height: rect.height,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const moveDrag = (event: ReactPointerEvent<HTMLButtonElement>) => {
    const drag = dragState.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    setPosition(clampPosition(
      drag.panelX + event.clientX - drag.originX,
      drag.panelY + event.clientY - drag.originY,
      drag.width,
      drag.height,
    ));
  };

  const stopDrag = (event: ReactPointerEvent<HTMLButtonElement>) => {
    if (dragState.current?.pointerId !== event.pointerId) return;
    dragState.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const moveWithKeyboard = (event: KeyboardEvent<HTMLButtonElement>) => {
    const directions: Record<string, [number, number]> = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, -1],
      ArrowDown: [0, 1],
    };
    const direction = directions[event.key];
    if (!direction || !panelRef.current) return;
    event.preventDefault();
    event.stopPropagation();
    const rect = panelRef.current.getBoundingClientRect();
    const step = event.shiftKey ? 32 : 10;
    setPosition(clampPosition(
      rect.left + direction[0] * step,
      rect.top + direction[1] * step,
      rect.width,
      rect.height,
    ));
  };

  const adjustSize = (delta: number) => {
    if (!panelRef.current) return;
    const rect = panelRef.current.getBoundingClientRect();
    const width = clamp(rect.width + delta, 260, Math.min(560, window.innerWidth - 16));
    const height = clamp(rect.height + delta, 180, window.innerHeight - 16);
    setSize({ width, height });
    if (position) setPosition(clampPosition(position.x, position.y, width, height));
  };

  const panelStyle: CSSProperties = {
    ...(position ? { left: position.x, top: position.y } : {}),
    ...(size && !collapsed ? { width: size.width, height: size.height } : {}),
  };

  return (
    <section
      ref={panelRef}
      className={styles.labelCard}
      style={panelStyle}
      data-positioned={position ? "true" : "false"}
      data-collapsed={collapsed ? "true" : "false"}
      aria-label={`展签：${item.displayTitle || item.object.title}`}
    >
      <div className={styles.labelWindowBar}>
        <button
          type="button"
          className={styles.dragHandle}
          onPointerDown={startDrag}
          onPointerMove={moveDrag}
          onPointerUp={stopDrag}
          onPointerCancel={stopDrag}
          onKeyDown={moveWithKeyboard}
          aria-label="移动展签；可拖动，或使用方向键微调，按 Shift 加速"
          title="拖动展签；方向键也可以移动"
        >
          <span aria-hidden="true">⠿</span>
          <span>{collapsed ? item.displayTitle || item.object.title : "移动展签"}</span>
        </button>
        <div className={styles.labelWindowActions}>
          {!collapsed && (
            <>
              <button type="button" onClick={() => adjustSize(-48)} aria-label="缩小展签">−</button>
              <button type="button" onClick={() => adjustSize(48)} aria-label="放大展签">＋</button>
            </>
          )}
          <button
            type="button"
            onClick={() => setCollapsed((value) => !value)}
            aria-expanded={!collapsed}
            aria-label={collapsed ? "展开展签" : "收起展签"}
          >
            {collapsed ? "展开" : "收起"}
          </button>
        </div>
      </div>

      {!collapsed && (
        <div className={styles.labelContent}>
      <div className={styles.labelHead}>
        <span className={styles.roleTag}>{item.roleLabel}</span>
        {item.object.evidenceDepth === "thin" && (
          <span className={styles.depthTag} title="该机构未提供策展说明字段，展签只使用著录信息">
            仅著录信息
          </span>
        )}
      </div>
      {/* Chinese first: institutions catalogue in English, but the exhibition
          is read in Chinese. The original stays visible underneath and in the
          source panel, so nothing is hidden. */}
      <h2>{item.displayTitle || item.object.titleOriginal || item.object.title}</h2>
      <p className={styles.originalTitle}>{item.object.title}</p>
      <p className={styles.tombstone}>
        {[item.object.date, item.object.medium, item.object.culture].filter(Boolean).join(" · ")}
      </p>

      <div className={styles.labelBody}>
        {item.labelSentences.map((sentence) => (
          <p key={sentence.id} data-type={sentence.type}>
            {sentence.text}
            <span className={styles.sentenceType}>{TYPE_LABELS[sentence.type]}</span>
          </p>
        ))}
      </div>

      <div className={styles.labelActions}>
        <button
          type="button"
          onClick={() => {
            setSourceOpen((open) => !open);
            if (!sourceOpen) logEvent("source_opened", exhibitionId, { itemId: item.id });
          }}
          aria-expanded={sourceOpen}
        >
          {sourceOpen ? "收起来源" : "来源"}
        </button>
        <a
          href={item.object.objectUrl}
          target="_blank"
          rel="noreferrer"
          onClick={() => logEvent("institution_page_opened", exhibitionId, { itemId: item.id })}
        >
          {item.object.institution || "机构页"} ↗
        </a>
      </div>

      {sourceOpen && (
        <div className={styles.sourcePanel}>
          {item.object.evidence.map((chunk) => (
            <article key={chunk.id}>
              <h3>{chunk.sourceLocation}</h3>
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
            </article>
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
      )}
    </section>
  );
}

/**
 * All hall text, as DOM on top of the canvas.
 *
 * Keeping it out of the scene means it stays selectable, sharp at any DPI, and
 * legible to assistive tech — a canvas is opaque to screen readers, which is
 * why the accessible-version link is permanent rather than a fallback.
 */
export function HallOverlay({
  exhibition,
  layout,
  stop,
  stopIndex,
  mode,
  cameraArrived = true,
  pointerLocked,
  reduceMotion,
  onGoTo,
  onEnterGuided,
  onEnterFree,
  onExit,
}: {
  exhibition: Exhibition;
  layout: HallLayout;
  stop: TourStop;
  stopIndex: number;
  mode: "guided" | "free";
  /** Narration waits until the guided camera has reached this stop. */
  cameraArrived?: boolean;
  pointerLocked: boolean;
  reduceMotion: boolean;
  onGoTo: (index: number) => void;
  onEnterGuided: () => void;
  onEnterFree: () => void;
  onExit: () => void;
}) {
  const itemsById = useMemo(
    () => new Map(exhibition.items.map((item) => [item.id, item])),
    [exhibition.items],
  );
  const chaptersById = useMemo(
    () => new Map(exhibition.chapters.map((chapter) => [chapter.id, chapter])),
    [exhibition.chapters],
  );

  const item: ExhibitionItem | null = stop.itemId ? itemsById.get(stop.itemId) ?? null : null;
  const chapter = stop.chapterId ? chaptersById.get(stop.chapterId) ?? null : null;

  const audioSegments = useMemo(() => {
    const describe = (candidate: TourStop) => {
      const candidateChapter = candidate.chapterId
        ? chaptersById.get(candidate.chapterId) ?? null
        : null;
      const candidateItem = candidate.itemId
        ? itemsById.get(candidate.itemId) ?? null
        : null;
      let text = "";

      if (candidate.kind === "lobby") {
        text = `${exhibition.title}。${exhibition.subtitle || ""}${exhibition.curatorialThesis}`;
      } else if (candidate.kind === "chapter" && candidateChapter) {
        text = `${candidateChapter.title}。${candidateChapter.leadIn}`;
      } else if (candidate.kind === "artwork" && candidateItem) {
        const spoken = candidateItem.labelSentences
          .filter((sentence) => sentence.type !== "institution_fact")
          .map((sentence) => sentence.text);
        // The device fallback reads the Chinese label, never the English
        // institution record. Qwen's text is reconstructed by the backend.
        text = `${candidateItem.displayTitle || candidateItem.object.title}。${spoken.join(" ")}`;
      } else if (candidate.kind === "epilogue") {
        text = exhibition.epilogue.text;
      }

      return {
        kind: candidate.kind,
        ref: candidate.kind === "chapter"
          ? candidate.chapterId
          : candidate.kind === "artwork"
            ? candidate.itemId
            : undefined,
        text,
      };
    };

    const nextStop = layout.stops[stopIndex + 1] ?? null;
    return {
      current: describe(stop),
      next: nextStop ? describe(nextStop) : null,
    };
  }, [chaptersById, exhibition, itemsById, layout.stops, stop, stopIndex]);

  const audio = useAudioGuide({
    exhibitionId: exhibition.id,
    segment: audioSegments.current,
    nextSegment: audioSegments.next,
    active: mode === "guided" && cameraArrived,
  });

  const atStart = stopIndex === 0;
  const atEnd = stopIndex >= layout.stops.length - 1;
  const audioIsActive =
    audio.status === "preparing" ||
    audio.status === "playing" ||
    (audio.status === "fallback" && audio.fallbackSpeaking);
  const audioButtonLabel = mode === "free"
    ? "⏸ 自由行走时暂停"
    : !cameraArrived
      ? "⏸ 镜头移动时暂停"
    : audio.status === "preparing"
      ? "⏳ AI 讲述准备中"
      : audio.status === "playing"
        ? "🔊 彦远讲述中"
        : audio.status === "fallback" && audio.fallbackSpeaking
          ? "🔊 设备语音讲述中"
          : audio.status === "error"
            ? "↻ 重试专业讲述"
            : "🎙 彦远专业讲述（AI 合成）";
  const audioStatusMessage = audio.message ?? (
    audio.status === "preparing"
      ? "千问 AI 正在准备专业播音导览…"
      : audio.status === "playing"
        ? "千问 AI 合成 · 专业播音声线"
        : null
  );
  return (
    <div className={styles.overlay} data-mode={mode}>
      {/* ------------------------------------------------------- top bar */}
      <header className={styles.topBar}>
        <button type="button" className={styles.ghostButton} onClick={onExit}>
          ← 离开展厅
        </button>
        <div className={styles.topRight}>
          <a className={styles.accessibleLink} href={`/exhibitions/${exhibition.id}?view=text`}>
            无障碍版本
          </a>
          <div className={styles.audioControl}>
            <button
              type="button"
              className={styles.ghostButton}
              aria-pressed={audioIsActive}
              aria-busy={audio.status === "preparing"}
              onClick={audio.toggle}
              disabled={!audio.supported || mode === "free" || !cameraArrived}
              title={
                !audio.supported
                  ? "当前浏览器不支持音频播放"
                  : mode === "free"
                    ? "切回导览模式后继续播放"
                    : !cameraArrived
                      ? "镜头到站后可以播放专业讲述"
                    : "千问 TTS AI 合成 · 专业播音声线"
              }
            >
              {audioButtonLabel}
            </button>
            {audioStatusMessage && (
              <span
                className={styles.audioStatus}
                role="status"
                aria-live="polite"
                data-state={audio.status}
              >
                {audioStatusMessage}
              </span>
            )}
          </div>
          <div className={styles.modeToggle} role="group" aria-label="参观方式">
            <button
              type="button"
              aria-pressed={mode === "guided"}
              onClick={onEnterGuided}
              className={mode === "guided" ? styles.modeActive : undefined}
            >
              导览
            </button>
            <button
              type="button"
              aria-pressed={mode === "free"}
              onClick={onEnterFree}
              className={mode === "free" ? styles.modeActive : undefined}
            >
              自由行走
            </button>
          </div>
        </div>
      </header>

      {/* ------------------------------------------------------- content */}
      {mode === "guided" && (
        <>
          {stop.kind === "lobby" && (
            <section className={styles.lobbyCard}>
              <LobbyPoster
                key={exhibition.poster?.backgroundUrl ?? "collection-fallback"}
                exhibition={exhibition}
              />
              <div>
                <span className={styles.eyebrow}>AI 策展人彦远为你策展</span>
                <h1>{exhibition.title}</h1>
                {exhibition.subtitle && <p className={styles.subtitle}>{exhibition.subtitle}</p>}
                <p className={styles.thesis}>{exhibition.curatorialThesis}</p>
                <dl className={styles.lobbyStats}>
                  <div>
                    <dt>叙事区段</dt>
                    <dd>{exhibition.chapters.length}</dd>
                  </div>
                  <div>
                    <dt>展品</dt>
                    <dd>{exhibition.items.length}</dd>
                  </div>
                  <div>
                    <dt>预计</dt>
                    <dd>{exhibition.visitorProfile?.durationMinutes ?? 10} 分钟</dd>
                  </div>
                </dl>
                <button type="button" className={styles.primaryAction} onClick={() => onGoTo(1)}>
                  开始参观 →
                </button>
              </div>
            </section>
          )}

          {stop.kind === "chapter" && chapter && (
            <section className={styles.chapterCard}>
              <span className={styles.eyebrow}>
                第 {chapter.order + 1} 部分 / 共 {exhibition.chapters.length}
              </span>
              <h2>{chapter.title}</h2>
              <p>{chapter.leadIn}</p>
            </section>
          )}

          {stop.kind === "artwork" && item && (
            // Keyed by item so the source panel closes when the visitor moves
            // on, without an effect resetting derived state.
            <ArtworkLabel key={item.id} item={item} exhibitionId={exhibition.id} />
          )}

          {stop.kind === "epilogue" && (
            <section className={styles.epilogueCard}>
              <span className={styles.eyebrow}>结语</span>
              <h2>AI 策展人彦远的结语</h2>
              <p className={styles.epilogueText}>{exhibition.epilogue.text}</p>
              <EpilogueConversation
                exhibitionId={exhibition.id}
                openQuestions={exhibition.epilogue.openQuestions}
                context="hall"
              />
              <details className={styles.boundaryDetails}>
                <summary>这场展览的材料边界</summary>
                <ul>
                  {exhibition.epilogue.materialBoundary.map((limit) => (
                    <li key={limit}>{limit}</li>
                  ))}
                </ul>
              </details>
              <div className={styles.epilogueActions}>
                <button type="button" onClick={() => onGoTo(0)}>
                  从头再看
                </button>
                <button type="button" className={styles.primaryAction} onClick={onExit}>
                  结束参观
                </button>
              </div>
            </section>
          )}
        </>
      )}

      {mode === "free" && !pointerLocked && (
        <div className={styles.freeWalkPrompt}>
          <p>点击画面开始自由行走</p>
          <small>桌面用 WASD 或方向键，触屏用左下角摇杆；随时可以切回导览</small>
        </div>
      )}

      {/* ------------------------------------------------------ bottom bar */}
      {mode === "guided" && (
        <nav className={styles.bottomBar} aria-label="参观进度">
          <button type="button" onClick={() => onGoTo(stopIndex - 1)} disabled={atStart}>
            ← 上一处
          </button>
          <ol className={styles.stopTrack}>
            {layout.stops.map((candidate, index) => (
              <li key={`${candidate.kind}-${index}`}>
                <button
                  type="button"
                  data-kind={candidate.kind}
                  aria-current={index === stopIndex ? "step" : undefined}
                  aria-label={candidate.label}
                  className={index === stopIndex ? styles.stopActive : styles.stopDot}
                  onClick={() => onGoTo(index)}
                />
              </li>
            ))}
          </ol>
          <button type="button" onClick={() => onGoTo(stopIndex + 1)} disabled={atEnd}>
            下一处 →
          </button>
        </nav>
      )}

      {reduceMotion && mode === "guided" && (
        <p className={styles.motionNote}>已按系统设置关闭镜头飞行动画。</p>
      )}
    </div>
  );
}
