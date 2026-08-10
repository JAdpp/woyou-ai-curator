export const PUZZLE_SIZE = 3;
export const EMPTY_TILE = 0;
export const SOLVED_PUZZLE = [1, 2, 3, 4, 5, 6, 7, 8, EMPTY_TILE] as const;

export type PuzzleDirection = "up" | "down" | "left" | "right";

function swap(board: readonly number[], first: number, second: number): number[] {
  const next = [...board];
  [next[first], next[second]] = [next[second], next[first]];
  return next;
}

export function adjacentIndices(index: number): number[] {
  const row = Math.floor(index / PUZZLE_SIZE);
  const column = index % PUZZLE_SIZE;
  const adjacent: number[] = [];

  if (row > 0) adjacent.push(index - PUZZLE_SIZE);
  if (column < PUZZLE_SIZE - 1) adjacent.push(index + 1);
  if (row < PUZZLE_SIZE - 1) adjacent.push(index + PUZZLE_SIZE);
  if (column > 0) adjacent.push(index - 1);

  return adjacent;
}

export function isPuzzleSolved(board: readonly number[]): boolean {
  return SOLVED_PUZZLE.every((tile, index) => board[index] === tile);
}

/** Move a numbered tile into the empty square, when the two are adjacent. */
export function movePuzzleTile(board: readonly number[], tileIndex: number): number[] {
  const emptyIndex = board.indexOf(EMPTY_TILE);
  if (!adjacentIndices(emptyIndex).includes(tileIndex)) return board as number[];
  return swap(board, emptyIndex, tileIndex);
}

/** Direction keys move the empty square in the named direction. */
export function movePuzzleEmpty(
  board: readonly number[],
  direction: PuzzleDirection,
): number[] {
  const emptyIndex = board.indexOf(EMPTY_TILE);
  const row = Math.floor(emptyIndex / PUZZLE_SIZE);
  const column = emptyIndex % PUZZLE_SIZE;
  let target = -1;

  if (direction === "up" && row > 0) target = emptyIndex - PUZZLE_SIZE;
  if (direction === "down" && row < PUZZLE_SIZE - 1) target = emptyIndex + PUZZLE_SIZE;
  if (direction === "left" && column > 0) target = emptyIndex - 1;
  if (direction === "right" && column < PUZZLE_SIZE - 1) target = emptyIndex + 1;

  return target < 0 ? (board as number[]) : swap(board, emptyIndex, target);
}

/**
 * Produce a solvable board by walking from the solved state through legal
 * moves. An injectable random source keeps the helper deterministic in tests.
 */
export function scramblePuzzle(
  random: () => number = Math.random,
  moveCount = 72,
): number[] {
  let board: readonly number[] = SOLVED_PUZZLE;
  let previousEmpty = -1;

  for (let move = 0; move < moveCount; move += 1) {
    const emptyIndex = board.indexOf(EMPTY_TILE);
    const candidates = adjacentIndices(emptyIndex).filter((index) => index !== previousEmpty);
    const pool = candidates.length > 0 ? candidates : adjacentIndices(emptyIndex);
    const choice = Math.min(pool.length - 1, Math.floor(random() * pool.length));
    previousEmpty = emptyIndex;
    board = swap(board, emptyIndex, pool[choice]);
  }

  if (isPuzzleSolved(board)) {
    const emptyIndex = board.indexOf(EMPTY_TILE);
    board = swap(board, emptyIndex, adjacentIndices(emptyIndex)[0]);
  }

  return [...board];
}
