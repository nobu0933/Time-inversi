const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const R = require('../src/game/rules.js');
const Replay = require('../src/game/replay.js');

// A small DOM and virtual clock exercise UI scheduling without a browser or network.
function app(initial = R.createGame()) {
  class Element {
    constructor() {
      this.children = []; this.attrs = {}; this.events = {}; this.hidden = false;
      this.style = { setProperty() {}, removeProperty() {} };
      const classes = new Set();
      this.classList = { add: (...v) => v.forEach(x => classes.add(x)), remove: (...v) => v.forEach(x => classes.delete(x)),
        toggle: (v, enabled) => enabled ? classes.add(v) : classes.delete(v), contains: v => classes.has(v) };
    }
    set innerHTML(value) { this.children = [new Element()]; this.children[0].className = 'body'; }
    setAttribute(key, value) { this.attrs[key] = value; }
    append(child) { this.children.push(child); }
    replaceChildren(...children) { this.children = children; }
    querySelector(selector) { return this.children.find(x => x.className === selector.slice(1)) || this.children[0]; }
    querySelectorAll() { return []; }
    addEventListener(type, fn) { (this.events[type] ||= []).push(fn); }
    emit(type, event = {}) { return (this.events[type] || []).map(fn => fn({ detail: 0, stopPropagation() {}, ...event })); }
    showModal() { this.open = true; }
    close(value) { this.returnValue = value; this.open = false; this.emit('close'); }
  }
  const ids = Object.fromEntries(['cells','log','new-game','pass-overlay','bgm','replay','time-clock','title','cpu-white','cpu-black','white-count','black-count'].map(id => [id, new Element()]));
  ids.bgm.append(new Element()); ids['pass-overlay'].hidden = true; ids.replay.hidden = true;
  const document = { querySelector: selector => ids[selector.slice(1)], createElement: () => new Element(), body: new Element(), documentElement: new Element(), addEventListener() {} };
  let now = 0, next = 0; const timers = new Map();
  const context = { window: { Reversi: { ...R, createGame: () => Replay.snapshot(initial) }, ReplayTimeline: Replay, addEventListener() {} }, document,
    matchMedia: () => ({ matches: false }), setTimeout: (fn, ms) => { timers.set(++next, { fn, at: now + ms }); return next; }, clearTimeout: id => timers.delete(id), console };
  vm.runInNewContext(fs.readFileSync(require.resolve('../src/ui/game.js'), 'utf8'), context);
  async function tick(ms) {
    const end = now + ms;
    for (let steps = 0; steps < 1000; steps++) {
      const entry = [...timers].sort((a,b) => a[1].at - b[1].at)[0];
      if (!entry || entry[1].at > end) break;
      now = entry[1].at; timers.delete(entry[0]); entry[1].fn();
      for (let i = 0; i < 15; i++) await Promise.resolve();
    }
    now = end;
  }
  return { ids, document, tick, click: id => ids[id].emit('click'), move: index => ids.cells.children[index].emit('click') };
}

test('white CPU plays after black; BGM switches its supplied icon', async () => {
  const ui = app();
  assert.equal(ui.ids.log.textContent, "Black's turn");
  ui.click('bgm'); assert.equal(ui.ids.bgm.children[0].src, 'assets/icon/bgm-off.svg');
  ui.click('bgm'); assert.equal(ui.ids.bgm.children[0].src, 'assets/icon/bgm-on.svg');
  ui.move(8); await ui.tick(500);
  assert.equal(ui.ids.log.textContent, "White's turn");
  await ui.tick(1150);
  assert.equal(ui.ids.log.textContent, "Black's turn");
});

test('new game during an animation cancels stale work and confirmation pauses CPU', async () => {
  const ui = app();
  ui.move(8); ui.click('title'); ui.ids['new-game'].close('yes');
  await ui.tick(2000);
  assert.equal(ui.ids.log.textContent, "Black's turn");
  assert.match(ui.ids.cells.children[8].attrs['aria-label'], /empty/);
  ui.click('cpu-black'); ui.click('title'); await ui.tick(2000);
  assert.match(ui.ids.cells.children[8].attrs['aria-label'], /empty/);
  ui.ids['new-game'].close('no'); await ui.tick(1150);
  assert.equal(ui.ids.log.textContent, "White's turn");
});

test('pass disc stays for 800 ms and disappears before the next turn', async () => {
  const state = R.createGame(); state.player = R.WHITE;
  state.board = Array(36).fill(R.BLACK);
  state.board[0] = state.board[30] = 0; state.board[1] = state.board[31] = R.WHITE;
  const ui = app(state);
  await ui.tick(350); assert.equal(ui.ids['pass-overlay'].hidden, false);
  await ui.tick(799); assert.equal(ui.ids['pass-overlay'].hidden, false);
  await ui.tick(1); assert.equal(ui.ids['pass-overlay'].hidden, true);
  assert.equal(ui.ids.log.textContent, "Black's turn");
});

test('corner inversion switches tint, pulses once, and announces inversion', async () => {
  const state = R.createGame(); state.board.fill(0);
  state.board[2] = state.board[14] = R.BLACK;
  state.board[1] = state.board[7] = R.WHITE;
  const ui = app(state); ui.move(0); await ui.tick(500);
  assert.equal(ui.document.body.classList.contains('reverse-time'), true);
  assert.equal(ui.document.body.classList.contains('time-pulse'), true);
  assert.equal(ui.ids.log.textContent, 'Time inversion');
  await ui.tick(360); assert.equal(ui.document.body.classList.contains('time-pulse'), false);
});

test('finished replay preserves the final English result and can run again', async () => {
  const state = R.createGame(); state.board.fill(R.BLACK);
  state.board[0] = 0; state.board[1] = R.WHITE; state.usedCorners = [0,5,30,35];
  const ui = app(state); ui.move(0); await ui.tick(500);
  assert.equal(ui.ids.replay.hidden, false);
  assert.equal(ui.ids.log.textContent, 'Black wins!');
  assert.equal(ui.ids['black-count'].textContent, '36');
  assert.equal(ui.ids['white-count'].textContent, '0');
  assert.equal(ui.ids['black-count'].hidden, false);
  ui.click('replay'); assert.equal(ui.ids.replay.disabled, true);
  await ui.tick(500);
  assert.equal(ui.ids.log.textContent, 'Black wins!');
  assert.equal(ui.ids.replay.disabled, false);
  ui.click('replay'); await ui.tick(500);
  assert.equal(ui.ids.log.textContent, 'Black wins!');
  assert.equal(ui.ids['black-count'].textContent, '36');
  ui.click('title'); ui.ids['new-game'].close('yes');
  assert.equal(ui.ids['black-count'].hidden, true);
  assert.equal(ui.ids['white-count'].hidden, true);
});
