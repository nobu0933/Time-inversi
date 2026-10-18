const { test } = require('node:test');
const assert = require('node:assert/strict');
const R = require('../src/game/rules.js');

test('clock begins at twelve and completes one 32-step revolution', () => {
  const start = R.createGame();
  assert.deepEqual(R.clockReading(start), { forward: 0, reverse: 0, start: 0, needle: 0, futureEnd: 0 });
  const full = { ...start, origins: Array(32).fill(1) };
  assert.deepEqual(R.clockReading(full), { forward: 32, reverse: 0, start: 0, needle: 32, futureEnd: 32 });
});

test('reverse sector retains its future edge while needle moves back past twelve', () => {
  let state = R.createGame();
  state.usedCorners = [0, 5, 30, 35];
  for (let i = 0; i < 4; i++) state = R.advance(state, R.moves(state.board, state.player)[0]).state;
  assert.deepEqual(R.clockReading(state), { forward: 4, reverse: 0, start: 0, needle: 4, futureEnd: 4 });
  state = R.invert(state).state;
  assert.deepEqual(R.clockReading(state), { forward: 3, reverse: 0, start: 0, needle: 3, futureEnd: 3 });
  let crossedTwelve = false;
  for (let i = 0; i < 20 && !state.ended; i++) {
    const before = R.clockReading(state), moves = R.moves(state.board, state.player);
    const turn = R.advance(state, moves[0] ?? null);
    state = turn.state;
    const after = R.clockReading(state);
    if (before.forward === 0 && after.reverse > before.reverse) {
      crossedTwelve = true;
      assert.equal(after.start, before.start - 1);
      assert.equal(after.futureEnd, before.futureEnd);
      assert.equal(after.needle, after.start);
      break;
    }
  }
  assert.ok(crossedTwelve);
  const beforeInversion = R.clockReading(state);
  const inversion = R.invert(state);
  const afterInversion = R.clockReading(inversion.state);
  assert.equal(afterInversion.start, beforeInversion.start + 1);
  assert.equal(afterInversion.futureEnd, beforeInversion.futureEnd);
});

test('forward pass without a disappearing stone leaves both wedges unchanged', () => {
  const board = Array(36).fill(1);
  board[0] = board[30] = 0; board[1] = board[31] = -1;
  const state = { ...R.createGame(), board, player: -1,
    history: { '1': [8, 9], '-1': [] }, usedCorners: [0, 5, 30, 35] };
  const before = R.clockReading(state);
  const turn = R.advance(state, null);
  const after = R.clockReading(turn.state);
  assert.equal(turn.removed, null);
  assert.equal(after.start, before.start);
  assert.equal(after.needle, before.needle);
  assert.equal(after.forward, before.forward);
  assert.equal(turn.state.board.filter(Boolean).length, board.filter(Boolean).length);
});

test('reverse pass with no disappearing stone leaves both wedges unchanged', () => {
  const board = Array(36).fill(0); board[0] = 1; board[35] = -1;
  const state = { ...R.createGame(), board, player: 1, direction: -1,
    origins: Array(36).fill(0).map((_, index) => index === 35 ? -1 : 0),
    history: { '1': [], '-1': [35] }, clockOrigin: -1,
    usedCorners: [0, 5, 30, 35] };
  const before = R.clockReading(state);
  const turn = R.advance(state, null);
  const after = R.clockReading(turn.state);
  assert.equal(after.start, before.start);
  assert.equal(after.needle, before.needle);
  assert.equal(after.futureEnd, before.futureEnd);
});

test('a pass that removes a forward stone changes the endpoint, not the orange start', () => {
  const board = Array(36).fill(0); board[0] = 1; board[35] = -1;
  const state = { ...R.createGame(), board, player: 1, direction: -1,
    origins: Array(36).fill(0).map((_, index) => index === 0 ? 1 : 0),
    history: { '1': [0], '-1': [] }, usedCorners: [0, 5, 30, 35] };
  const before = R.clockReading(state);
  const turn = R.advance(state, null);
  const after = R.clockReading(turn.state);
  assert.equal(turn.removed, 0);
  assert.equal(after.start, before.start);
  assert.equal(after.needle, before.needle - 1);
  assert.equal(after.forward, before.forward - 1);
});

test('forward pass removing a reverse stone keeps the orange arc fixed', () => {
  const board = Array(36).fill(0); board[0] = 1; board[35] = -1;
  const state = { ...R.createGame(), board, player: 1,
    origins: Array(36).fill(0).map((_, index) => index === 35 ? -1 : 0),
    history: { '1': [], '-1': [35] }, usedCorners: [0, 5, 30, 35] };
  const before = R.clockReading(state);
  const turn = R.advance(state, null);
  const after = R.clockReading(turn.state);
  assert.equal(turn.removed, 35);
  assert.equal(after.start, before.start);
  assert.equal(after.needle, before.needle);
  assert.equal(after.reverse, before.reverse - 1);
});

test('inverting back before crossing twelve keeps the orange start at twelve', () => {
  let state = R.createGame();
  state.usedCorners = [0, 5, 30, 35];
  for (let i = 0; i < 4; i++) state = R.advance(state, R.moves(state.board, state.player)[0]).state;
  state = R.invert(state).state;
  state = R.advance(state, R.moves(state.board, state.player)[0]).state;
  assert.equal(R.clockReading(state).start, 0);
  assert.equal(R.clockReading(state).reverse, 1);

  state = R.invert(state).state;
  const reading = R.clockReading(state);
  assert.equal(reading.start, 0);
  assert.equal(reading.needle, reading.forward);
  assert.equal(reading.reverse, 0);
});

test('a pass and later inversion cannot move the oldest orange start forward', () => {
  const board = Array(36).fill(1);
  board[0] = board[30] = 0; board[1] = board[31] = -1;
  const state = { ...R.createGame(), board, player: -1, direction: -1,
    origins: Array(36).fill(0).map((_, index) => index === 35 ? -1 : 0),
    history: { '1': [], '-1': [35] }, clockOrigin: -2, clockAnchor: -2,
    usedCorners: [0, 5, 30, 35] };
  const passed = R.advance(state, null).state;
  assert.equal(R.clockReading(passed).start, -2);
  let current = R.invert(passed).state; // The pass has no stone to remove.
  assert.equal(R.clockReading(current).start, -2);
  current = R.invert(current).state;
  current = R.invert(current).state; // A later blue stone disappears.
  assert.equal(R.clockReading(current).start, -2);
  assert.equal(current.clockAnchor, -2);
});

test('a new inversion farther in the past updates the orange start once', () => {
  const board = Array(36).fill(0); board[0] = 1; board[35] = -1;
  const state = { ...R.createGame(), board, direction: -1,
    origins: Array(36).fill(0).map((_, index) => index === 35 ? -1 : 0),
    history: { '1': [], '-1': [35] }, clockOrigin: -4, clockAnchor: -2,
    usedCorners: [0, 5, 30, 35] };
  const inverted = R.invert(state).state;
  assert.equal(R.clockReading(inverted).start, -3);
  assert.equal(inverted.clockAnchor, -3);
});

test('ordinary turns preserve the oldest clock anchor', () => {
  const state = { ...R.createGame(), clockAnchor: -3, clockOrigin: -3 };
  const next = R.advance(state, R.moves(state.board, state.player)[0]).state;
  assert.equal(next.clockAnchor, -3);
});
