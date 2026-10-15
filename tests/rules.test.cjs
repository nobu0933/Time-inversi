const { test } = require('node:test');
const assert = require('node:assert/strict');
const R = require('../src/game/rules.js');
test('initial position and legal opening; no mutation or illegal placement', () => {
  const board = R.initial();
  assert.deepEqual(R.moves(board, 1), [8, 13, 22, 27]);
  assert.equal(R.play(board, 0, 1), null);
  assert.equal(R.play(board, 14, 1), null);
  const move = R.play(board, 22, 1);
  assert.deepEqual(move.captured, [21]);
  assert.equal(move.board.filter(x => x === 1).length, 4);
  assert.equal(move.player, -1);
  assert.equal(board[22], 0);
});
test('captures all eight directions and never wraps across board edges', () => {
  const board = Array(36).fill(0), index = 14;
  for (const [dx, dy] of [[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]]) {
    board[index + dy * 6 + dx] = -1;
    board[index + dy * 12 + dx * 2] = 1;
  }
  assert.equal(R.flips(board, index, 1).length, 8);
  const edge = Array(36).fill(0); edge[6] = -1; edge[7] = 1;
  assert.deepEqual(R.flips(edge, 5, 1), []);
});
test('automatic pass and termination with unfilled squares', () => {
  const board = Array(36).fill(1);
  board[0] = 0; board[1] = -1; board[30] = 0; board[31] = -1;
  const move = R.play(board, 0, 1);
  assert.equal(move.passed, true); assert.equal(move.player, 1); assert.equal(move.ended, false);
  const end = R.play(move.board, 30, 1);
  assert.equal(end.ended, true);
  const sparse = Array(36).fill(0); sparse[1] = -1; sparse[2] = 1;
  assert.equal(R.play(sparse, 0, 1).ended, true);
});
test('complete games preserve stone count and terminate correctly', () => {
  for (let seed = 1; seed <= 40; seed++) {
    let board = R.initial(), player = 1, count = 4, ended = false, random = seed;
    while (!ended) {
      random = (random * 1664525 + 1013904223) >>> 0;
      const moves = R.moves(board, player);
      assert.ok(moves.length);
      const move = R.play(board, moves[random % moves.length], player);
      board = move.board; player = move.player; ended = move.ended;
      assert.equal(board.filter(Boolean).length, ++count);
      assert.ok(count <= 36);
    }
    assert.equal(R.moves(board, 1).length + R.moves(board, -1).length, 0);
  }
});
