(function (root) {
  'use strict';
  const SIZE = 6, BLACK = 1, WHITE = -1;
  const directions = [-1, 0, 1].flatMap(y => [-1, 0, 1].filter(x => x || y).map(x => [x, y]));
  function initial() {
    const board = Array(SIZE * SIZE).fill(0);
    board[14] = board[21] = WHITE;
    board[15] = board[20] = BLACK;
    return board;
  }
  function flips(board, index, player) {
    if (!Number.isInteger(index) || index < 0 || index >= board.length || board[index]) return [];
    const result = [], col = index % SIZE, row = Math.floor(index / SIZE);
    for (const [dx, dy] of directions) {
      let x = col + dx, y = row + dy;
      const ray = [];
      while (x >= 0 && x < SIZE && y >= 0 && y < SIZE && board[y * SIZE + x] === -player) {
        ray.push(y * SIZE + x); x += dx; y += dy;
      }
      if (ray.length && x >= 0 && x < SIZE && y >= 0 && y < SIZE && board[y * SIZE + x] === player) result.push(...ray);
    }
    return result;
  }
  function moves(board, player) { return board.flatMap((_, i) => flips(board, i, player).length ? [i] : []); }
  function play(board, index, player) {
    const captured = flips(board, index, player);
    if (!captured.length) return null;
    const next = board.slice();
    for (const i of [index, ...captured]) next[i] = player;
    const opponentCanPlay = moves(next, -player).length > 0;
    const ended = !opponentCanPlay && !moves(next, player).length;
    return { board: next, captured, player: opponentCanPlay ? -player : player, passed: !opponentCanPlay && !ended, ended };
  }
  // Placement identity and time direction are independent of the stone's current color.
  // Each direction owns a stack of still-existing turns, including empty pass turns.
  function createGame() {
    return { board: initial(), origins: Array(SIZE * SIZE).fill(0), player: BLACK,
      direction: 1, history: { '1': [], '-1': [] }, ended: false };
  }
  function invert(state) {
    return state.ended ? state : { ...state, direction: -state.direction };
  }
  function advance(state, index = null) {
    if (state.ended) return null;
    const captured = index === null ? [] : flips(state.board, index, state.player);
    if (index === null ? moves(state.board, state.player).length > 0 : !captured.length) return null;
    const board = state.board.slice(), origins = state.origins.slice();
    const history = { '1': state.history[1].slice(), '-1': state.history[-1].slice() };
    const placedDirection = state.direction;
    if (index !== null) {
      board[index] = state.player; origins[index] = placedDirection;
      for (const i of captured) board[i] = state.player;
    }
    history[placedDirection].push(index);
    // Pop exactly one turn: a pass must not skip to the preceding placement.
    const previous = history[-placedDirection].pop();
    const removed = previous == null ? null : previous;
    const removedColor = removed === null ? 0 : board[removed];
    if (removed !== null) { board[removed] = 0; origins[removed] = 0; }
    const forced = placedDirection === -1 && history[1].length === 0;
    const direction = forced ? 1 : placedDirection;
    const black = board.filter(value => value === BLACK).length;
    const white = board.filter(value => value === WHITE).length;
    const stalled = !moves(board, BLACK).length && !moves(board, WHITE).length;
    const ended = black === 0 || white === 0 || black + white === SIZE * SIZE
      || (stalled && !origins.includes(-direction));
    return { state: { board, origins, history, player: -state.player, direction, ended },
      index, captured, removed, removedColor, passed: index === null, forced, placedDirection };
  }
  const api = { SIZE, BLACK, WHITE, initial, flips, moves, play, createGame, invert, advance };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Reversi = api;
})(globalThis);
