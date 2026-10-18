const { test } = require('node:test');
const assert = require('node:assert/strict');
const R = require('../src/game/rules.js');
const corners = [[0,1,2], [5,4,3], [30,31,32], [35,34,33]];
for (const [corner, adjacent, anchor] of corners) {
  test(`corner ${corner} inverts once and stays spent after disappearance`, () => {
    const s = R.createGame();
    s.board[adjacent] = -1; s.board[anchor] = 1;
    s.board[9] = 1; s.origins[9] = 1; s.history[1] = [9];
    const before = JSON.stringify(s);
    const move = R.advance(s, corner);
    assert.equal(JSON.stringify(s), before);
    assert.equal(move.cornerInversion.removed, corner);
    assert.equal(move.state.direction, -1);
    assert.equal(move.state.player, -1);
    assert.equal(move.state.board[corner], 0);
    assert.equal(move.state.board[adjacent], 1);
    assert.deepEqual(move.state.usedCorners, [corner]);
    const repeat = { ...move.state, board: move.state.board.slice(), player: 1 };
    repeat.board[adjacent] = -1;
    const again = R.advance(repeat, corner);
    assert.equal(again.cornerInversion, null);
    assert.equal(again.state.board[corner], 1);
    assert.deepEqual(again.state.usedCorners, [corner]);
    assert.deepEqual(R.createGame().usedCorners, []);
  });
}
test('reverse corner removes pending forward stone then itself, returning forward', () => {
  const s = R.createGame(); s.direction = -1;
  s.board[1] = -1; s.board[2] = 1;
  s.board[9] = 1; s.origins[9] = 1; s.history[1] = [9];
  const m = R.advance(s, 0);
  assert.equal(m.removed, 9);
  assert.equal(m.cornerInversion.removed, 0);
  assert.equal(m.state.direction, 1);
  assert.equal(m.state.player, -1);
  assert.deepEqual(m.state.usedCorners, [0]);
});
test('corner inversion occurs before full-board termination', () => {
  const s = R.createGame(); s.board.fill(1);
  s.board[0] = 0; s.board[1] = s.board[10] = -1;
  s.origins[9] = 1; s.history[1] = [9];
  const m = R.advance(s, 0);
  assert.equal(m.state.board[0], 0);
  assert.equal(m.state.ended, false);
  assert.equal(m.cornerInversion.removed, 0);
});
test('invalid corner placement does not consume its trigger', () => {
  const s = R.createGame();
  assert.equal(R.advance(s, 0), null);
  assert.deepEqual(s.usedCorners, []);
});
