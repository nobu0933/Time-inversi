/* Float32 inference for the trained residual CNN + full-history GRU.
 * Runs in a worker; no Python, remote API or third-party runtime is required. */
(function (root) {
  'use strict';
  const R = typeof module !== 'undefined' && module.exports ? require('../game/rules.js') : root.Reversi;
  const relu = x => { for (let i = 0; i < x.length; i++) x[i] = Math.max(0, x[i]); return x; };
  const sigmoid = x => 1 / (1 + Math.exp(-x));
  class Network {
    constructor(data) {
      if (data.format !== 'time-inversi-browser-v1') throw new Error('Unsupported MID model');
      this.config = data.config;
      this.weights = {};
      for (const [name, tensor] of Object.entries(data.tensors)) {
        const binary = atob(tensor.data);
        const bytes = Uint8Array.from(binary, x => x.charCodeAt(0));
        const view = new DataView(bytes.buffer);
        const values = new Float32Array(bytes.length / 4);
        for (let i = 0; i < values.length; i++) values[i] = view.getFloat32(i * 4, true);
        this.weights[name] = values;
      }
    }
    linear(x, name) {
      const w = this.weights[name + '.weight'], b = this.weights[name + '.bias'];
      const output = new Float32Array(b.length);
      for (let o = 0; o < b.length; o++) {
        let sum = b[o];
        for (let i = 0; i < x.length; i++) sum += w[o * x.length + i] * x[i];
        output[o] = sum;
      }
      return output;
    }
    conv(x, name) {
      const w = this.weights[name + '.weight'];
      const channels = x.length / 36, width = channels * 9, outChannels = w.length / width;
      // Each spatial patch is contiguous for the dot products below.
      const patches = new Float32Array(36 * width);
      for (let p = 0; p < 36; p++) {
        const row = Math.floor(p / 6), col = p % 6;
        for (let c = 0; c < channels; c++) {
          for (let dy = 0; dy < 3; dy++) for (let dx = 0; dx < 3; dx++) {
            const y = row + dy - 1, z = col + dx - 1;
            if (y >= 0 && y < 6 && z >= 0 && z < 6)
              patches[p * width + c * 9 + dy * 3 + dx] = x[c * 36 + y * 6 + z];
          }
        }
      }
      const output = new Float32Array(outChannels * 36);
      for (let c = 0; c < outChannels; c++) for (let p = 0; p < 36; p++) {
        let sum = 0;
        const wi = c * width, xi = p * width;
        for (let i = 0; i < width; i++) sum += w[wi + i] * patches[xi + i];
        output[c * 36 + p] = sum;
      }
      return output;
    }
    norm(x, name) {
      let mean = 0, variance = 0;
      for (const value of x) mean += value;
      mean /= x.length;
      for (const value of x) variance += (value - mean) ** 2;
      const scale = 1 / Math.sqrt(variance / x.length + 1e-5);
      const w = this.weights[name + '.weight'], b = this.weights[name + '.bias'];
      for (let i = 0; i < x.length; i++) x[i] = (x[i] - mean) * scale * w[Math.floor(i / 36)] + b[Math.floor(i / 36)];
      return x;
    }
    history(turns) {
      const h = this.config.history_hidden, w = this.weights;
      let hidden = new Float32Array(h);
      for (const token of turns.length ? turns : [38]) {
        const id = token === null ? 36 : token;
        const input = new Float32Array(3 * h), recurrent = new Float32Array(3 * h);
        for (let o = 0; o < 3 * h; o++) {
          let a = w['history.bias_ih_l0'][o], b = w['history.bias_hh_l0'][o];
          for (let i = 0; i < 32; i++) a += w['history.weight_ih_l0'][o * 32 + i] * w['embedding.weight'][id * 32 + i];
          for (let i = 0; i < h; i++) b += w['history.weight_hh_l0'][o * h + i] * hidden[i];
          input[o] = a; recurrent[o] = b;
        }
        const next = new Float32Array(h);
        for (let i = 0; i < h; i++) {
          const reset = sigmoid(input[i] + recurrent[i]);
          const update = sigmoid(input[h + i] + recurrent[h + i]);
          const candidate = Math.tanh(input[2 * h + i] + reset * recurrent[2 * h + i]);
          next[i] = (1 - update) * candidate + update * hidden[i];
        }
        hidden = next;
      }
      return hidden;
    }
    predict(state) {
      if (state.ended) throw new Error('Cannot predict a finished game');
      const legalMoves = R.moves(state.board, state.player), legal = legalMoves.length ? legalMoves : [36];
      const input = new Float32Array(8 * 36);
      for (let i = 0; i < 36; i++) {
        input[i] = +(state.board[i] === state.player);
        input[36 + i] = +(state.board[i] === -state.player);
        input[72 + i] = +(state.origins[i] === 1);
        input[108 + i] = +(state.origins[i] === -1);
        input[144 + i] = +state.usedCorners.includes(i);
        input[180 + i] = +legalMoves.includes(i);
        input[216 + i] = +(state.direction === 1);
        input[252 + i] = +(state.player === 1);
      }
      let x = relu(this.norm(this.conv(input, 'board.0'), 'board.1'));
      for (let i = 0; i < this.config.blocks; i++) {
        const name = `board.${3 + i}.layers`;
        let residual = relu(this.norm(this.conv(x, name + '.0'), name + '.1'));
        residual = this.norm(this.conv(residual, name + '.3'), name + '.4');
        for (let j = 0; j < x.length; j++) residual[j] += x[j];
        x = relu(residual);
      }
      x = relu(this.linear(x, `board.${4 + this.config.blocks}`));
      const combined = new Float32Array(128 + 2 * this.config.history_hidden);
      combined.set(x);
      combined.set(this.history(state.history[1]), 128);
      combined.set(this.history(state.history[-1]), 128 + this.config.history_hidden);
      const logits = this.linear(combined, 'policy');
      const value = Math.tanh(this.linear(relu(this.linear(combined, 'value.0')), 'value.2')[0]);
      const maximum = Math.max(...legal.map(i => logits[i]));
      const policy = new Float32Array(37);
      let total = 0;
      for (const i of legal) { policy[i] = Math.exp(logits[i] - maximum); total += policy[i]; }
      for (const i of legal) policy[i] /= total;
      return { policy, value };
    }
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = Network;
  else root.TimeInversiNetwork = Network;
})(globalThis);
