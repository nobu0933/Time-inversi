const { test } = require('node:test');
const assert = require('node:assert/strict');
const R = require('../src/game/rules.js');
function step(state, index) {
  const result = R.advance(state, index);
  assert.ok(result, `Expected valid turn at ${index}`);
  return result;
}
function fixture(board, player = 1, direction = 1) {
  return { ...R.createGame(), board, player, direction, usedCorners: [0, 5, 30, 35] };
}
test('inversion removes B immediately without changing the next player; C removes A', () => {
  let s = step(R.createGame(), 8).state;
  s = step(s, 7).state;
  const before = JSON.stringify(s);
  const inversion = R.invert(s);
  assert.equal(JSON.stringify(s), before);
  assert.equal(inversion.removed, 7);
  assert.equal(inversion.removedColor, -1);
  assert.equal(inversion.state.player, 1);
  assert.equal(inversion.state.board[7], 0);
  assert.equal(inversion.state.board[14], -1);
  assert.deepEqual(inversion.state.history[-1], []);
  const c = step(inversion.state, 13);
  assert.equal(c.removed, 8);
  assert.equal(c.state.direction, -1);
  assert.deepEqual(c.state.history[1], []);
  assert.equal(c.state.player, -1);
  assert.equal(c.state.origins[13], -1);
});
test('reverse-to-forward immediately removes reverse stone without another placement', () => {
  let s = R.createGame();
  s.usedCorners = [0, 5, 30, 35];
  for (let i = 0; i < 4; i++) s = step(s, R.moves(s.board, s.player)[0]).state;
  s = R.invert(s).state;
  const index = R.moves(s.board, s.player)[0];
  s = step(s, index).state;
  const currentPlayer = s.player;
  const move = R.invert(s);
  assert.equal(move.removed, index);
  assert.equal(move.state.board[index], 0);
  assert.equal(move.state.origins[index], 0);
  assert.equal(move.state.player, currentPlayer);
  assert.equal(move.state.direction, 1);
});
test('pass history consumes only one slot on inversion and preserves player', () => {
  const board = Array(36).fill(1);
  board[0] = board[30] = 0; board[1] = board[31] = -1;
  let s = fixture(board, -1);
  s = step(s, null).state;
  s = step(s, 0).state;
  const inversion = R.invert(s);
  assert.equal(inversion.removed, 0);
  assert.equal(inversion.state.player, -1);
  assert.equal(inversion.state.direction, -1);
  const pass = step(inversion.state, null);
  assert.equal(pass.removed, null);
  assert.equal(pass.state.direction, -1);
  assert.equal(pass.state.board[0], 0);
  const trailingPass = fixture(R.initial());
  trailingPass.history[1] = [8, null];
  trailingPass.board[8] = 1; trailingPass.origins[8] = 1;
  const onlyPass = R.invert(trailingPass);
  assert.equal(onlyPass.removed, null);
  assert.equal(onlyPass.state.board[8], 1);
  assert.deepEqual(onlyPass.state.history[1], [8]);
});
test('inversion handles empty history, last history entry, and elimination without forced direction change', () => {
  const empty = R.invert(R.createGame());
  assert.equal(empty.state.direction, -1);
  assert.equal(empty.state.player, 1);
  assert.deepEqual(empty.state.board, R.initial());
  const last = R.invert(step(R.createGame(), 8).state);
  assert.equal(last.removed, 8);
  assert.equal(last.state.direction, -1);
  assert.equal(last.state.player, -1);
  const next = step(last.state, R.moves(last.state.board, last.state.player)[0]);
  assert.equal(next.state.direction, -1);
  assert.deepEqual(next.state.history[1], []);
  const board = Array(36).fill(0); board[0] = 1; board[35] = -1;
  const s = fixture(board);
  s.history[1] = [35]; s.origins[35] = 1;
  const end = R.invert(s);
  assert.equal(end.state.ended, true);
  assert.equal(end.state.board[35], 0);
  assert.equal(R.invert(end.state), null);
});
test('full-board check occurs after disappearing stone and not before', () => {
  const board = Array(36).fill(1);
  board[0] = 0; board[1] = board[10] = -1;
  const s = fixture(board);
  s.origins[5] = s.origins[31] = -1;
  s.history[-1] = [31, 5];
  const turn = step(s, 0);
  assert.equal(turn.state.board[5], 0);
  assert.equal(turn.state.ended, false);
  const ordinary = step(fixture(board), 0);
  assert.equal(ordinary.state.ended, true);
});
test('removing last stone of a color ends the game after a pass', () => {
  const board = Array(36).fill(0); board[0] = 1; board[35] = -1;
  const s = fixture(board);
  s.origins[35] = -1; s.history[-1] = [35];
  const turn = step(s, null);
  assert.equal(turn.removed, 35);
  assert.equal(turn.state.ended, true);
  assert.equal(R.advance(turn.state, null), null);
});
test('both blocked continues over pass history while opposite stones remain', () => {
  const board = Array(36).fill(0); board[0] = 1; board[35] = -1;
  const s = fixture(board);
  s.origins[35] = -1; s.history[-1] = [35, null];
  const first = step(s, null);
  assert.equal(first.removed, null);
  assert.equal(first.state.ended, false);
  assert.equal(step(first.state, null).state.ended, true);
  assert.equal(step(fixture(board), null).state.ended, true);
});
test('forward continues with no opposite history; illegal pass/move rejected', () => {
  const s = R.createGame();
  assert.equal(R.advance(s, null), null);
  assert.equal(R.advance(s, 0), null);
  const turn = step(s, 8);
  assert.equal(turn.state.direction, 1);
  assert.equal(turn.removed, null);
  assert.equal(turn.state.direction, 1);
});
test('repeated inversions preserve placement identity and one removal per turn', () => {
  for (let seed = 1; seed <= 30; seed++) {
    let s = R.createGame(), random = seed;
    for (let t = 0; t < 200 && !s.ended; t++) {
      random = (random * 1664525 + 1013904223) >>> 0;
      if (random % 5 === 0) s = R.invert(s).state;
      if (s.ended) break;
      const legal = R.moves(s.board, s.player);
      const index = legal.length ? legal[random % legal.length] : null;
      const count = s.board.filter(Boolean).length;
      const turn = step(s, index);
      s = turn.state;
      assert.equal(s.board.filter(Boolean).length, count + (index === null ? 0 : 1) - (turn.removed === null ? 0 : 1)
        - (turn.cornerInversion?.removed != null ? 1 : 0));
      const tracked = new Set();
      for (const direction of [1, -1]) for (const i of s.history[direction]) {
        if (i === null) continue;
        assert.ok(!tracked.has(i)); tracked.add(i);
        assert.ok(s.board[i]); assert.equal(s.origins[i], direction);
      }
      s.origins.forEach((direction, i) => { if (direction) assert.ok(tracked.has(i)); });
    }
  }
});

