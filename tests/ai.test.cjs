const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const R = require('../src/game/rules.js');
const Search = require('../src/ai/search.js');
const uniform = { predict: () => ({ policy: new Float32Array(37).fill(1 / 37), value: 0 }) };

test('browser MCTS propagates winning terminal values and supports passes', () => {
  const state = R.createGame();
  state.board.fill(1); state.board[0] = 0; state.board[1] = -1; state.usedCorners = [0,5,30,35];
  const search = new Search(uniform, () => .5);
  assert.equal(search.search(state, 8).action, 0);
  const node = [...search.nodes.values()][0];
  assert.equal(node.values[0] / node.visits[0], 1);
  state.player = -1;
  assert.equal(new Search(uniform).search(state, 4).action, 36);
});

test('browser MCTS keys include both histories, passes and used corners', () => {
  const search = new Search(uniform);
  const state = R.createGame(); search.search(state, 1);
  const size = search.nodes.size;
  state.history[1].push(null); search.search(state, 1);
  assert.ok(search.nodes.size > size);
});

test('worker client uses relative URLs, reuses loaded model and cancels stale results', async () => {
  const workers = [];
  class Worker {
    constructor(url) { this.url = url; workers.push(this); }
    postMessage(data) { this.sent = data; }
    terminate() { this.stopped = true; }
  }
  const window = {};
  vm.runInNewContext(fs.readFileSync(require.resolve('../src/ai/client.js'), 'utf8'), {
    window, document: { currentScript: { src: 'https://example.test/my-game/src/ai/client.js' } },
    Worker, URL, DOMException
  });
  const controller = new AbortController();
  const first = window.ProAI.choose(R.createGame(), { signal: controller.signal });
  assert.equal(workers[0].url.href, 'https://example.test/my-game/src/ai/worker.js');
  assert.equal(workers[0].sent.mode, 'MID');
  const cancelled = assert.rejects(first, { name: 'AbortError' });
  controller.abort(); await cancelled;
  assert.equal(workers[0].stopped, true);
  const next = window.ProAI.choose(R.createGame(), { mode: 'PRO' });
  assert.equal(workers[1].sent.mode, 'PRO');
  workers[0].onmessage({ data: { id: workers[0].sent.id, action: 27 } });
  workers[1].onmessage({ data: { id: workers[1].sent.id, action: 8 } });
  assert.equal((await next).action, 8);
  const again = window.ProAI.choose(R.createGame());
  assert.equal(workers.length, 2);
  workers[1].onmessage({ data: { id: workers[1].sent.id, action: 13 } });
  assert.equal((await again).action, 13);
});

test('worker loads MID and PRO weights separately and reuses each model', () => {
  const imports = [], responses = [];
  const context = {
    self: { postMessage: message => responses.push(message) },
    importScripts(...paths) {
      imports.push(...paths);
      if (paths.length === 1) context.TimeInversiWeights = { mode: paths[0].includes('pro-model') ? 'PRO' : 'MID' };
    },
    TimeInversiNetwork: class { constructor(weights) { this.mode = weights.mode; } },
    TimeInversiSearch: class {
      constructor(network) { this.network = network; }
      search() { return { action: this.network.mode === 'PRO' ? 13 : 8 }; }
    }
  };
  vm.runInNewContext(fs.readFileSync(require.resolve('../src/ai/worker.js'), 'utf8'), context);
  for (const [id, mode] of [[1, 'MID'], [2, 'PRO'], [3, 'MID']])
    context.self.onmessage({ data: { id, mode, state: R.createGame() } });
  assert.deepEqual(responses.map(result => result.action), [8, 13, 8]);
  assert.equal(imports.filter(path => path.includes('model.js')).length, 2);
  assert.ok(imports.includes('../../assets/ai/model.js'));
  assert.ok(imports.includes('../../assets/ai/pro-model.js'));
});
