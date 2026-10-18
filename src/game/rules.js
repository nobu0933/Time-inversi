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
      direction: 1, history: { '1': [], '-1': [] }, clockOrigin: 0, clockAnchor: 0,
      usedCorners: [], ended: false };
  }
  function invert(state) {
    if (state.ended) return null;
    const result = rewind({ ...state, board: state.board.slice(), origins: state.origins.slice(),
      history: { '1': state.history[1].slice(), '-1': state.history[-1].slice() },
      direction: -state.direction });
    if (result.state.direction === 1) {
      // Keep the oldest reverse-to-forward turning point as the orange sector's start.
      // A removed reverse stone moves the live boundary, but never moves that start forward.
      const turningPoint = result.state.clockOrigin + (result.removed !== null ? 1 : 0);
      const anchor = Math.min(state.clockAnchor ?? 0, turningPoint);
      result.state.clockAnchor = anchor;
      result.state.clockOrigin = anchor;
    }
    return result;
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
    const clockOrigin = placedDirection === -1 && index !== null
      && state.history[1].every(turn => turn === null)
      ? state.clockOrigin - 1 : state.clockOrigin;
    const cornerTriggered = [0, SIZE - 1, SIZE * (SIZE - 1), SIZE * SIZE - 1].includes(index)
      && !state.usedCorners.includes(index);
    const usedCorners = cornerTriggered ? [...state.usedCorners, index] : state.usedCorners.slice();
    const turn = rewind({ board, origins, history, clockOrigin, clockAnchor: state.clockAnchor, usedCorners, player: -state.player, direction: placedDirection });
    const cornerInversion = cornerTriggered ? invert({ ...turn.state, ended: false }) : null;
    return { ...turn, state: cornerInversion ? cornerInversion.state : turn.state,
      beforeInversion: cornerTriggered ? turn.state : null, cornerInversion,
      index, captured, passed: index === null, placedDirection };
  }
  function rewind(state) {
    const { board, origins, history, player } = state;
    const placedDirection = state.direction;
    // Pop exactly one turn: a pass must not skip to the preceding placement.
    const previous = history[-placedDirection].pop();
    const removed = previous == null ? null : previous;
    const removedColor = removed === null ? 0 : board[removed];
    if (removed !== null) { board[removed] = 0; origins[removed] = 0; }
    const direction = placedDirection;
    const black = board.filter(value => value === BLACK).length;
    const white = board.filter(value => value === WHITE).length;
    const stalled = !moves(board, BLACK).length && !moves(board, WHITE).length;
    const ended = black === 0 || white === 0 || black + white === SIZE * SIZE
      || (stalled && !origins.includes(-direction));
    return { state: { ...state, board, origins, history, player, direction, ended },
      removed, removedColor };
  }
  function clockReading(state) {
    const forward = state.origins.filter(origin => origin === 1).length;
    const reverse = state.origins.filter(origin => origin === -1).length;
    const start = state.clockOrigin;
    return { forward, reverse, start, needle: start + forward, futureEnd: start + forward + reverse };
  }
  const api = { SIZE, BLACK, WHITE, initial, flips, moves, play, createGame, invert, advance, clockReading };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Reversi = api;
})(globalThis);
