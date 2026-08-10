import assert from "node:assert/strict";
import test from "node:test";
import {
  EMPTY_TILE,
  SOLVED_PUZZLE,
  adjacentIndices,
  isPuzzleSolved,
  movePuzzleEmpty,
  movePuzzleTile,
  scramblePuzzle,
} from "./slidingPuzzle";

test("the solved board and corner adjacency are stable", () => {
  assert.equal(isPuzzleSolved(SOLVED_PUZZLE), true);
  assert.deepEqual(adjacentIndices(0), [1, 3]);
  assert.deepEqual(adjacentIndices(8), [5, 7]);
});

test("only a tile next to the empty square can move", () => {
  const board = [...SOLVED_PUZZLE];
  assert.deepEqual(movePuzzleTile(board, 7), [1, 2, 3, 4, 5, 6, 7, 0, 8]);
  assert.equal(movePuzzleTile(board, 0), board);
});

test("direction keys move the empty square without changing the tile set", () => {
  const moved = movePuzzleEmpty(SOLVED_PUZZLE, "up");
  assert.deepEqual(moved, [1, 2, 3, 4, 5, 0, 7, 8, 6]);
  assert.deepEqual([...moved].sort((a, b) => a - b), [EMPTY_TILE, 1, 2, 3, 4, 5, 6, 7, 8]);
  assert.equal(movePuzzleEmpty(SOLVED_PUZZLE, "right"), SOLVED_PUZZLE);
});

test("scrambling always returns a solvable-looking non-solved permutation", () => {
  const board = scramblePuzzle(() => 0.25, 40);
  assert.equal(isPuzzleSolved(board), false);
  assert.deepEqual([...board].sort((a, b) => a - b), [EMPTY_TILE, 1, 2, 3, 4, 5, 6, 7, 8]);
});
