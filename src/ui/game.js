(() => {
  'use strict';
  const R = window.Reversi;
  const Replay = window.ReplayTimeline;
  const grid = document.querySelector('#cells');
  const log = document.querySelector('#log');
  const dialog = document.querySelector('#new-game');
  const passOverlay = document.querySelector('#pass-overlay');
  const bgm = document.querySelector('#bgm');
  const replay = document.querySelector('#replay');
  const timeClock = document.querySelector('#time-clock');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const corners = new Set([0, R.SIZE - 1, R.SIZE * (R.SIZE - 1), R.SIZE * R.SIZE - 1]);
  let game = R.createGame();
  let replayFrames = [Replay.snapshot(game)], epoch = 0, replaying = false;
  const cpu = { '1': 'OFF', '-1': 'EZY' };
  let bgmOn = true;
  const cancelled = Symbol('cancelled');
  let { board, player, ended } = game;
  let busy = false, hover = -1, turnTimer, logTimer, proRequest;
  let pointer = null, target = -1;
  const cursor = piece(player, 'cursor-piece');
  cursor.hidden = true;
  document.body.append(cursor);
  const name = value => value === R.BLACK ? 'Black' : 'White';
  const cells = board.map((_, index) => {
    const cell = document.createElement('button');
    cell.type = 'button'; cell.className = 'cell';
    cell.addEventListener('focus', () => { hover = index; preview(); });
    cell.addEventListener('blur', () => { hover = -1; preview(); });
    cell.addEventListener('click', event => {
      if (event.detail === 0 || event.pointerType === 'touch') {
        event.stopPropagation(); place(index);
      }
    });
    cell.addEventListener('keydown', event => {
      const delta = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -6, ArrowDown: 6 }[event.key];
      if (delta) { event.preventDefault(); cells[(index + delta + 36) % 36].focus(); }
    });
    grid.append(cell); return cell;
  });
  function piece(color, extra = '') {
    const element = document.createElement('span');
    element.className = `piece ${color === R.WHITE ? 'white' : 'black'} ${extra}`;
    element.setAttribute('aria-hidden', 'true');
    element.innerHTML = '<span class="body"></span>';
    return element;
  }
  function preview() {
    grid.querySelectorAll('.landing-shadow').forEach(element => element.remove());
    target = -1;
    const overControls = pointer && document.elementFromPoint(pointer.x, pointer.y)?.closest('.control');
    cursor.hidden = !pointer || busy || ended || cpu[player] !== 'OFF' || dialog.open || Boolean(overControls);
    document.body.classList.toggle('piece-pointer', Boolean(pointer) && !busy && !ended && cpu[player] === 'OFF' && !dialog.open);
    if (busy || ended || cpu[player] !== 'OFF' || dialog.open) return;
    const legal = R.moves(board, player);
    if (pointer) {
      cursor.classList.toggle('white', player === R.WHITE);
      const sample = cells[0];
      cursor.style.width = `${sample.clientWidth * .73}px`;
      cursor.style.height = `${sample.clientHeight * .73}px`;
      cursor.style.left = `${pointer.x}px`;
      cursor.style.top = `${pointer.y}px`;
      let nearest = Infinity;
      for (const index of legal) {
        const rect = cells[index].getBoundingClientRect();
        const distance = Math.hypot(pointer.x - rect.x - rect.width / 2, pointer.y - rect.y - rect.height / 2);
        if (distance <= Math.min(rect.width, rect.height) * .8 && distance < nearest) {
          nearest = distance; target = index;
        }
      }
    } else if (legal.includes(hover)) target = hover;
    if (target >= 0) {
      const shadow = document.createElement('span');
      shadow.className = 'piece landing-shadow';
      shadow.setAttribute('aria-hidden', 'true');
      cells[target].append(shadow);
    }
  }
  document.addEventListener('pointermove', event => {
    if (event.pointerType === 'touch') return;
    pointer = { x: event.clientX, y: event.clientY }; preview();
  });
  document.addEventListener('click', event => {
    if (event.button !== 0 || event.detail === 0 || event.target.closest('.control')) return;
    if (event.pointerType === 'touch') return;
    pointer = { x: event.clientX, y: event.clientY }; preview();
    if (target >= 0) place(target);
  });
  function clearPointer() { pointer = null; preview(); }
  document.documentElement.addEventListener('pointerleave', clearPointer);
  window.addEventListener('blur', clearPointer);
  window.addEventListener('resize', preview);
  document.addEventListener('keydown', event => {
    if (['Tab', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) clearPointer();
  });
  document.addEventListener('pointerdown', event => { if (event.pointerType === 'touch') clearPointer(); });
  function updateTimeShades(history = game.history) {
    cells.forEach(cell => {
      cell.style.removeProperty('--time-strength');
      cell.classList.remove('nearest-time', 'second-time');
    });
    for (const direction of [1, -1]) {
      const turns = history[direction];
      // Anchor to the nearest surviving stone, not a trailing pass. Passes
      // between stones still count as distance along the timeline.
      const nearest = turns.findLastIndex(index => index !== null);
      const second = turns.findLastIndex((index, turn) => turn < nearest && index !== null);
      for (let turn = 0; turn <= nearest; turn++) {
        const index = turns[turn];
        if (index === null) continue;
        const distance = nearest - turn;
        cells[index].classList.toggle('nearest-time', distance === 0);
        cells[index].classList.toggle('second-time', turn === second);
        const strength = 100 / (1 + distance * .24);
        cells[index].style.setProperty('--time-strength', `${strength.toFixed(3)}%`);
      }
    }
  }
  function render() {
    document.body.classList.toggle('reverse-time', game.direction === -1);
    const clock = R.clockReading(game);
    timeClock.style.setProperty('--clock-start', `${clock.start * 11.25}deg`);
    timeClock.style.setProperty('--orange-end', `${clock.forward * 11.25}deg`);
    timeClock.style.setProperty('--blue-end', `${(clock.forward + clock.reverse) * 11.25}deg`);
    timeClock.style.setProperty('--hand-angle', `${clock.needle * 11.25}deg`);
    timeClock.setAttribute('aria-label', `${game.direction === 1 ? 'Forward' : 'Reverse'} time. ${clock.forward} forward stones, ${clock.reverse} reverse stones`);
    const legal = busy || ended ? [] : R.moves(board, player);
    cells.forEach((cell, index) => {
      cell.replaceChildren();
      cell.classList.toggle('legal', legal.includes(index));
      const activeCorner = corners.has(index) && !game.usedCorners.includes(index) && !board[index];
      cell.classList.toggle('inversion-corner', activeCorner);
      cell.classList.toggle('forward-stone', game.origins[index] === 1);
      cell.classList.toggle('reverse-stone', game.origins[index] === -1);
      cell.setAttribute('aria-disabled', String(!legal.includes(index)));
      cell.setAttribute('aria-label', `Row ${Math.floor(index / 6) + 1}, column ${index % 6 + 1}: ${board[index] ? name(board[index]) : 'empty'}${activeCorner ? ', time inversion corner' : ''}`);
      if (board[index]) cell.append(piece(board[index]));
      if (activeCorner) {
        const icon = document.createElement('img');
        icon.className = 'corner-icon';
        icon.src = game.direction === 1 ? 'assets/icon/time-inversion-forward.svg' : 'assets/icon/time-inversion-reverse.svg';
        icon.alt = '';
        icon.setAttribute('aria-hidden', 'true');
        cell.append(icon);
      }
    });
    updateTimeShades();
    preview();

  }
  function setLog(text) { log.textContent = text; }
  function updateLog() {
    if (replaying) return;
    const blackCount = document.querySelector('#black-count');
    const whiteCount = document.querySelector('#white-count');
    blackCount.hidden = whiteCount.hidden = !ended;
    if (!ended) { setLog(`${name(player)}'s turn`); return; }
    const black = board.filter(value => value === R.BLACK).length;
    const white = board.filter(value => value === R.WHITE).length;
    blackCount.textContent = String(black);
    whiteCount.textContent = String(white);
    blackCount.setAttribute('aria-label', `Black: ${black} stones`);
    whiteCount.setAttribute('aria-label', `White: ${white} stones`);
    setLog(black === white ? 'Draw!' : `${name(black > white ? R.BLACK : R.WHITE)} wins!`);
  }
  function scheduleTurn() {
    clearTimeout(turnTimer);
    proRequest?.abort(); proRequest = null;
    if (busy || ended || dialog.open || replaying) return;
    const legal = R.moves(board, player);
    if (!legal.length) turnTimer = setTimeout(() => place(null, 'auto'), 350);
    else if (cpu[player] === 'EZY') turnTimer = setTimeout(() => {
      const moves = R.moves(board, player);
      if (cpu[player] === 'EZY' && moves.length) place(moves[Math.floor(Math.random() * moves.length)], 'auto');
    }, 650);
    else if (cpu[player] === 'MID') turnTimer = setTimeout(requestMidMove, 350);
  }
  async function requestMidMove() {
    const run = epoch, position = game, controller = new AbortController();
    proRequest = controller;
    try {
      const result = await window.ProAI.choose(position, { signal: controller.signal });
      if (run !== epoch || position !== game || cpu[player] !== 'MID' || dialog.open || replaying) return;
      const moves = R.moves(board, player);
      if (!Number.isInteger(result.action) || !moves.includes(result.action)) throw new Error('MID returned an illegal move');
      place(result.action, 'auto');
    } catch (error) {
      if (error.name !== 'AbortError' && run === epoch && position === game && cpu[player] === 'MID') {
        console.error(error);
        setLog('MID unavailable — reload the page');
      }
    } finally { if (proRequest === controller) proRequest = null; }
  }
  async function pause(ms, run, motion = false) {
    await new Promise(resolve => setTimeout(resolve, motion && reduced.matches ? 0 : ms));
    if (run !== epoch) throw cancelled;
  }
  async function pulse(run) {
    document.body.classList.remove('time-pulse');
    void document.body.offsetWidth;
    document.body.classList.add('time-pulse');
    await pause(360, run, true);
    document.body.classList.remove('time-pulse');
  }
  async function showPass(run) {
    const disc = piece(player, 'pass-disc');
    const label = document.createElement('span');
    label.className = 'pass-label'; label.textContent = 'PASS';
    disc.querySelector('.body').append(label);
    passOverlay.replaceChildren(disc); passOverlay.hidden = false;
    await pause(800, run);
    passOverlay.hidden = true;
  }
  async function place(index, source = 'human') {
    if (busy || ended || dialog.open || (source === 'human' && cpu[player] !== 'OFF')) return;
    const run = epoch;
    const before = game;
    const move = R.advance(game, index);
    if (!move) return;
    const frames = Replay.framesForAdvance(before, move);
    const origin = source === 'human' && pointer && !cursor.hidden ? cursor.getBoundingClientRect() : null;
    busy = true;
    clearTimeout(turnTimer); clearTimeout(logTimer);
    grid.setAttribute('aria-busy', 'true');
    render();
    try {
      if (index === null) await showPass(run);
      else {
        const placed = piece(player, 'placing');
        cells[index].classList.add(move.placedDirection === 1 ? 'forward-stone' : 'reverse-stone');
        cells[index].append(placed);
        if (origin) {
          const destination = placed.getBoundingClientRect();
          placed.style.setProperty('--drop-x', `${origin.left - destination.left}px`);
          placed.style.setProperty('--drop-y', `${origin.top - destination.top}px`);
        }
        await pause(240, run, true);
        updateTimeShades({ ...game.history, [move.placedDirection]: [...game.history[move.placedDirection], index] });
      }
      for (const i of move.captured) {
        const disc = piece(player, 'turning');
        disc.style.setProperty('--old-face', player === R.BLACK ? 'var(--white)' : 'var(--black)');
        disc.style.setProperty('--old-side', player === R.BLACK ? 'var(--white-side)' : 'var(--black-side)');
        cells[i].replaceChildren(disc);
      }
      if (move.captured.length) await pause(260, run, true);
      await animateRemoval(move, run);
      if (move.cornerInversion) {
        game = { ...move.beforeInversion, direction: -move.placedDirection, ended: false };
        board = game.board;
        render(); setLog('Time inversion');
        await pulse(run);
        await animateRemoval(move.cornerInversion, run);
      }
      if (run !== epoch) return;
      game = move.state;
      replayFrames.push(...frames);
      ({ board, player, ended } = game);
      busy = false;
      grid.setAttribute('aria-busy', 'false');
      replay.hidden = !ended;
      render();
      if (move.cornerInversion && !ended) {
        setLog('Time inversion');
        logTimer = setTimeout(() => { if (run === epoch) updateLog(); }, 500);
      } else updateLog();
      scheduleTurn();
    } catch (error) { if (error !== cancelled) throw error; }
  }
  async function animateRemoval(move, run) {
    if (move.removed !== null) {
      cells[move.removed].replaceChildren(piece(move.removedColor, 'vanishing'));
      await pause(300, run, true);
      cells[move.removed].replaceChildren();
      cells[move.removed].classList.remove('forward-stone', 'reverse-stone');
    }
  }
  replay.addEventListener('click', async event => {
    event.stopPropagation();
    if (!ended || busy || dialog.open) return;
    const finalGame = game, run = epoch;
    busy = true; replaying = true; replay.disabled = true;
    grid.setAttribute('aria-busy', 'true');
    try {
      for (const frame of replayFrames) {
        const changed = game.direction !== frame.direction;
        game = frame;
        ({ board, player, ended } = game);
        render();
        if (changed) await pulse(run);
        await pause(145, run);
      }
      game = finalGame;
      ({ board, player, ended } = game);
      busy = false; replaying = false; replay.disabled = false;
      grid.setAttribute('aria-busy', 'false'); render();
    } catch (error) { if (error !== cancelled) throw error; }
  });
  function newGame() {
    epoch++;
    proRequest?.abort(); proRequest = null;
    clearTimeout(turnTimer); clearTimeout(logTimer);
    game = R.createGame(); ({ board, player, ended } = game); hover = -1;
    replayFrames = [Replay.snapshot(game)];
    busy = false; replaying = false; replay.disabled = false; replay.hidden = true;
    passOverlay.hidden = true;
    document.body.classList.remove('time-pulse');
    grid.setAttribute('aria-busy', 'false');
    render(); updateLog(); scheduleTurn();
  }
  document.querySelector('#title').addEventListener('click', () => {
    clearTimeout(turnTimer);
    proRequest?.abort(); proRequest = null;
    dialog.returnValue = ''; dialog.showModal(); preview();
  });
  dialog.addEventListener('close', () => {
    if (dialog.returnValue === 'yes') newGame();
    else { preview(); scheduleTurn(); }
  });
  bgm.addEventListener('click', () => {
    bgmOn = !bgmOn;
    bgm.setAttribute('aria-pressed', String(bgmOn));
    bgm.title = `BGM ${bgmOn ? 'on' : 'off'}`;
    bgm.querySelector('img').src = `assets/icon/bgm-${bgmOn ? 'on' : 'off'}.svg`;
  });
  for (const color of [R.BLACK, R.WHITE]) {
    const button = document.querySelector(`#cpu-${name(color).toLowerCase()}`);
    button.addEventListener('click', () => {
      const modes = ['OFF', 'EZY', 'MID', 'PRO'];
      cpu[color] = modes[(modes.indexOf(cpu[color]) + 1) % modes.length];
      button.textContent = cpu[color];
      button.setAttribute('data-mode', cpu[color]);
      button.setAttribute('aria-pressed', String(cpu[color] !== 'OFF'));
      button.setAttribute('aria-label', `${name(color)} CPU: ${cpu[color]}${cpu[color] === 'PRO' ? ', not available yet' : ''}`);
      button.title = `${name(color)} CPU: ${{ OFF: 'human', EZY: 'random', MID: 'browser AI', PRO: 'not available yet' }[cpu[color]]}`;
      if (log.textContent === 'MID unavailable — reload the page') updateLog();
      preview(); scheduleTurn();
    });
  }
  render(); updateLog(); scheduleTurn();
})();
