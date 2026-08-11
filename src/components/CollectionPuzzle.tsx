"use client";

import { useEffect, useState, type CSSProperties, type KeyboardEvent } from "react";
import { getCollectionHighlights, resolveObjectImageUrl } from "@/lib/api";
import type { CollectionHighlights } from "@/lib/types";
import {
  EMPTY_TILE,
  PUZZLE_SIZE,
  isPuzzleSolved,
  movePuzzleEmpty,
  movePuzzleTile,
  scramblePuzzle,
  type PuzzleDirection,
} from "./slidingPuzzle";
import { fill } from "@/lib/i18n";
import { useLanguage } from "@/lib/useLanguage";
import styles from "./puzzle.module.css";

const KEY_DIRECTIONS: Partial<Record<string, PuzzleDirection>> = {
  ArrowUp: "up",
  ArrowDown: "down",
  ArrowLeft: "left",
  ArrowRight: "right",
};

export function CollectionPuzzle() {
  const { t: copy } = useLanguage();
  const t = copy.puzzle;
  const [highlights, setHighlights] = useState<CollectionHighlights | null>(null);
  const [loadError, setLoadError] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const [imageReady, setImageReady] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);
  const [initialBoard, setInitialBoard] = useState(() => scramblePuzzle());
  const [board, setBoard] = useState(initialBoard);
  const [moves, setMoves] = useState(0);

  useEffect(() => {
    let cancelled = false;
    getCollectionHighlights(2)
      .then((result) => {
        if (!cancelled) setHighlights(result);
      })
      .catch(() => {
        if (!cancelled) setLoadError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const items = highlights?.items.slice(0, 2) ?? [];
  const item = items[activeIndex] ?? null;
  const imageUrl = item ? resolveObjectImageUrl(item.id, 1024) : null;
  const completed = isPuzzleSolved(board);
  const emptyIndex = board.indexOf(EMPTY_TILE);
  const institutionSummary = item
    ? highlights?.institutionSummaries.find((summary) => summary.name === item.institution)
    : null;
  const imageLicense = institutionSummary?.imageLicenses.length
    ? institutionSummary.imageLicenses.join("、")
    : t.licenceFallback;

  function startBoard(nextBoard = scramblePuzzle()) {
    setInitialBoard(nextBoard);
    setBoard(nextBoard);
    setMoves(0);
  }

  function applyMove(nextBoard: number[]) {
    if (nextBoard === board || completed) return;
    setBoard(nextBoard);
    setMoves((value) => value + 1);
  }

  function changeArtwork() {
    if (items.length < 2) return;
    setActiveIndex((index) => (index + 1) % items.length);
    setImageReady(false);
    setImageFailed(false);
    startBoard();
  }

  function handleGridKey(event: KeyboardEvent<HTMLDivElement>) {
    const direction = KEY_DIRECTIONS[event.key];
    if (!direction) return;
    event.preventDefault();
    applyMove(movePuzzleEmpty(board, direction));
  }

  if (loadError || (highlights && items.length === 0)) {
    return (
      <aside className={styles.unavailable} aria-label={t.unavailableLabel}>
        <strong>{t.unavailableTitle}</strong>
        <p>{t.unavailableBody}</p>
      </aside>
    );
  }

  return (
    <aside className={styles.puzzle} aria-labelledby="collection-puzzle-title">
      <header className={styles.header}>
        <div>
          <h2 id="collection-puzzle-title">{t.heading}</h2>
          <p>{t.instructions}</p>
        </div>
        <span aria-live="polite">{completed ? t.done : fill(t.moves, { n: moves })}</span>
      </header>

      <div
        className={styles.grid}
        data-ready={imageReady && !imageFailed ? "true" : "false"}
        data-complete={completed ? "true" : "false"}
        role="group"
        aria-label={t.boardLabel}
        tabIndex={0}
        onKeyDown={handleGridKey}
      >
        {imageUrl && !imageFailed && (
          // eslint-disable-next-line @next/next/no-img-element -- this hidden image preloads the proxy asset used by CSS tiles
          <img
            className={styles.preload}
            src={imageUrl}
            alt=""
            aria-hidden="true"
            onLoad={() => setImageReady(true)}
            onError={() => {
              setImageReady(false);
              setImageFailed(true);
            }}
          />
        )}

        {board.map((tile, position) => {
          if (tile === EMPTY_TILE) {
            return <span className={styles.blank} key="empty" aria-hidden="true" />;
          }

          const sourceIndex = tile - 1;
          const sourceRow = Math.floor(sourceIndex / PUZZLE_SIZE);
          const sourceColumn = sourceIndex % PUZZLE_SIZE;
          const movable = Math.abs(position - emptyIndex) === PUZZLE_SIZE
            || (Math.floor(position / PUZZLE_SIZE) === Math.floor(emptyIndex / PUZZLE_SIZE)
              && Math.abs(position - emptyIndex) === 1);
          const tileStyle: CSSProperties = imageReady && imageUrl && !imageFailed
            ? {
                backgroundImage: `url("${imageUrl}")`,
                backgroundPosition: `${(sourceColumn / (PUZZLE_SIZE - 1)) * 100}% ${(sourceRow / (PUZZLE_SIZE - 1)) * 100}%`,
              }
            : {};

          return (
            <button
              type="button"
              key={tile}
              className={styles.tile}
              style={tileStyle}
              data-movable={movable ? "true" : "false"}
              aria-disabled={!movable || completed}
              aria-label={`${fill(t.tileLabel, { n: tile })}${movable && !completed ? t.tileMovable : ""}`}
              onClick={() => applyMove(movePuzzleTile(board, position))}
            >
              <span>{tile}</span>
            </button>
          );
        })}

        {!imageReady && !imageFailed && <p className={styles.loading}>{t.loadingImage}</p>}
        {imageFailed && <p className={styles.loading}>{t.imageFailed}</p>}
        {completed && <p className={styles.complete}>{t.solved}</p>}
      </div>

      {item && (
        <div className={styles.objectMeta}>
          <strong>{item.title}</strong>
          <span>{item.institution}</span>
          <small>{fill(t.credit, { licence: imageLicense })}</small>
        </div>
      )}

      <div className={styles.actions}>
        <button type="button" onClick={() => {
          setBoard(initialBoard);
          setMoves(0);
        }}>
          {t.reset}
        </button>
        <button type="button" onClick={changeArtwork} disabled={items.length < 2}>
          {t.swap}
        </button>
      </div>
    </aside>
  );
}
