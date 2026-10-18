(function (root) {
  'use strict';

  function snapshot(state) {
    return { ...state, board: state.board.slice(), origins: state.origins.slice(),
      history: { '1': state.history[1].slice(), '-1': state.history[-1].slice() },
      usedCorners: state.usedCorners.slice() };
  }

  function framesForAdvance(before, move) {
    const frames = [];
    if (move.index !== null) {
      const placed = snapshot(before);
      placed.board[move.index] = before.player;
      placed.origins[move.index] = move.placedDirection;
      for (const index of move.captured) placed.board[index] = before.player;
      placed.history[move.placedDirection].push(move.index);
      if (move.placedDirection === -1 && before.history[1].every(index => index === null))
        placed.clockOrigin--;
      if (move.beforeInversion) placed.usedCorners = move.beforeInversion.usedCorners.slice();
      placed.player = -before.player;
      placed.ended = false;
      frames.push(placed);
    }
    if (move.beforeInversion) frames.push(snapshot(move.beforeInversion));
    frames.push(snapshot(move.state));
    return frames;
  }

  const api = { snapshot, framesForAdvance };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ReplayTimeline = api;
})(globalThis);
