const { test } = require('node:test');
const assert = require('node:assert/strict');
const R = require('../src/game/rules.js');
function step(state, index) {
  const result = R.advance(state, index);
  assert.ok(result, `Expected valid turn at ${index}`);
  return result;
}
function fixture(board, player = 1, direction = 1) {
  return { ...R.createGame(), board, player, direction };
}
test('A B inversion C removes B; D removes A and forces forward', () => {
  const initial = R.createGame();
  let s = step(initial, 8).state;
  s = step(s, 7).state;
  const before = JSON.stringify(s);
  const c = step(R.invert(s), 6);
  assert.equal(JSON.stringify(s), before);
  assert.equal(c.removed, 7);
  assert.equal(c.removedColor, 1); // B was flipped by C before disappearing.
  assert.equal(c.state.board[7], 0);
  assert.equal(c.state.board[14], -1); // B's original capture is not undone.
  assert.equal(c.state.origins[6], -1);
  assert.equal(c.state.origins[8], 1);
  const d = step(c.state, 2);
  assert.equal(d.removed, 8);
  assert.equal(d.forced, true);
  assert.equal(d.state.direction, 1);
  assert.equal(d.state.origins[8], 0);
  assert.ok(R.moves(d.state.board, d.state.player).includes(8));
  assert.deepEqual(initial.board, R.initial());
  assert.ok(initial.origins.every(value => value === 0));
});
test('manual reverse-to-forward removes latest reverse stone; direction survives color flips', () => {
  let s = step(R.createGame(), 8).state;
  s = step(s, 7).state;
  s = step(R.invert(s), 6).state;
  const d = step(R.invert(s), 2);
  assert.equal(d.removed, 6);
  assert.equal(d.state.origins[6], 0);
  assert.equal(d.state.origins[2], 1);
  assert.equal(d.state.board[8], -1);
  assert.equal(d.state.origins[8], 1);
  assert.equal(d.state.direction, 1);
});
test('pass consumes time; returning to its history slot removes nothing', () => {
  const board = Array(36).fill(1);
  board[0] = board[30] = 0; board[1] = board[31] = -1;
  let s = fixture(board, -1);
  s = step(s, null).state;
  assert.deepEqual(s.history[1], [null]);
  s = step(s, 0).state;
  const c = step(R.invert(s), null);
  assert.equal(c.removed, 0);
  assert.equal(c.state.direction, -1);
  const d = step(c.state, 30);
  assert.equal(d.removed, null);
  assert.equal(d.forced, true);
  assert.equal(d.state.board[0], 0);
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
  assert.equal(turn.forced, false);
});
test('repeated inversions preserve placement identity and one removal per turn', () => {
  for (let seed = 1; seed <= 30; seed++) {
    let s = R.createGame(), random = seed;
    for (let t = 0; t < 200 && !s.ended; t++) {
      random = (random * 1664525 + 1013904223) >>> 0;
      if (random % 5 === 0) s = R.invert(s);
      const legal = R.moves(s.board, s.player);
      const index = legal.length ? legal[random % legal.length] : null;
      const count = s.board.filter(Boolean).length;
      const turn = step(s, index);
      s = turn.state;
      assert.equal(s.board.filter(Boolean).length, count + (index === null ? 0 : 1) - (turn.removed === null ? 0 : 1));
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
