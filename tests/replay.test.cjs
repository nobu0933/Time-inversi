const { test } = require('node:test');
const assert = require('node:assert/strict');
const R = require('../src/game/rules.js');
const Replay = require('../src/game/replay.js');

test('replay keeps placement, disappearance, board colors, and clock in order', () => {
  let game = R.createGame();
  game.usedCorners = [0, 5, 30, 35];
  const first = R.advance(game, R.moves(game.board, game.player)[0]);
  const firstFrames = Replay.framesForAdvance(game, first);
  assert.equal(firstFrames.length, 2);
  assert.equal(firstFrames[0].board[first.index], R.BLACK);
  assert.equal(firstFrames[0].origins[first.index], 1);
  assert.deepEqual(R.clockReading(firstFrames[0]), { forward: 1, reverse: 0, start: 0, needle: 1, futureEnd: 1 });
  assert.deepEqual(firstFrames.at(-1).board, first.state.board);
  assert.notStrictEqual(firstFrames.at(-1).board, first.state.board);

  game = first.state;
  const second = R.advance(game, R.moves(game.board, game.player)[0]);
  const frames = Replay.framesForAdvance(game, second);
  assert.equal(frames.length, 2);
  assert.equal(frames[0].board[second.index], R.WHITE);
  assert.equal(R.clockReading(frames[0]).forward, 2);
  assert.equal(R.clockReading(frames[1]).forward, 2);
  const inverted = R.invert(second.state);
  assert.notEqual(inverted.removed, null);
  assert.equal(R.clockReading(inverted.state).forward, 1);
  assert.equal(inverted.state.board[inverted.removed], 0);
});

test('corner replay includes a frame before its forced inversion', () => {
  const game = R.createGame();
  const board = Array(36).fill(0);
  board[2] = board[14] = R.BLACK;
  board[1] = board[7] = R.WHITE;
  game.board = board;
  const move = R.advance(game, 0);
  assert.ok(move.cornerInversion);
  const frames = Replay.framesForAdvance(game, move);
  assert.equal(frames.length, 3);
  assert.equal(frames[0].board[0], R.BLACK);
  assert.equal(frames[1].direction, 1);
  assert.equal(frames[2].direction, -1);
  assert.ok(frames.every(frame => frame.usedCorners.includes(0)));
});
