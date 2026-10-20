(function (root) {
  'use strict';
  const R = typeof module !== 'undefined' && module.exports ? require('../game/rules.js') : root.Reversi;
  const key = s => JSON.stringify([s.board, s.origins, s.player, s.direction, s.history[1], s.history[-1], [...s.usedCorners].sort(), s.ended]);
  const choose = (items, random) => items[Math.floor(random() * items.length)];
  class Search {
    constructor(network, random = Math.random) { this.network = network; this.random = random; this.nodes = new Map(); }
    expand(state) {
      const { policy, value } = this.network.predict(state);
      const moves = R.moves(state.board, state.player), legal = moves.length ? moves : [36];
      this.nodes.set(key(state), { prior: policy, legal, visits: new Int32Array(37), values: new Float64Array(37), count: 0 });
      return value;
    }
    visit(state, depth = 0) {
      if (state.ended) return Math.sign(state.board.reduce((sum, color) => sum + color, 0)) * state.player;
      if (depth >= 512) throw new Error('MID search exceeded depth limit');
      const node = this.nodes.get(key(state));
      if (!node) return this.expand(state);
      const scale = 1.5 * Math.sqrt(1 + node.count);
      let best = -Infinity, choices = [];
      for (const a of node.legal) {
        const q = node.visits[a] ? node.values[a] / node.visits[a] : 0;
        const score = q + scale * node.prior[a] / (1 + node.visits[a]);
        if (score > best + 1e-12) { best = score; choices = [a]; }
        else if (Math.abs(score - best) <= 1e-12) choices.push(a);
      }
      const action = choose(choices, this.random);
      const next = R.advance(state, action === 36 ? null : action).state;
      let value = this.visit(next, depth + 1);
      if (next.player !== state.player) value = -value;
      node.visits[action]++; node.count++; node.values[action] += value;
      return value;
    }
    search(state, simulations = 64) {
      if (state.ended || !Number.isInteger(simulations) || simulations < 1) throw new Error('Invalid MID search');
      if (!this.nodes.has(key(state))) this.expand(state);
      for (let i = 0; i < simulations; i++) this.visit(state);
      const node = this.nodes.get(key(state)), max = Math.max(...node.visits);
      return { action: choose(node.legal.filter(a => node.visits[a] === max), this.random),
        policy: Array.from(node.visits, n => n / node.count) };
    }
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = Search;
  else root.TimeInversiSearch = Search;
})(globalThis);
