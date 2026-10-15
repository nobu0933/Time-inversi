(() => {
  'use strict';
  const R = window.Reversi;
  const grid = document.querySelector('#cells');
  const announcement = document.querySelector('#announcement');
  const notice = document.querySelector('#notice');
  const result = document.querySelector('#result');
  const inversion = document.querySelector('#inversion');
  const directionLabel = document.querySelector('#time-direction');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let game = R.createGame();
  let { board, player, ended } = game;
  let busy = false, hover = -1, noticeTimer, passTimer;
  let pointer = null, target = -1;
  const cursor = piece(player, 'cursor-piece');
  cursor.hidden = true;
  document.body.append(cursor);
  const name = value => value === R.BLACK ? '黒' : '白';
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
    const overControls = pointer && document.elementFromPoint(pointer.x, pointer.y)?.closest('.time-controls, #result');
    cursor.hidden = !pointer || busy || ended || Boolean(overControls);
    document.body.classList.toggle('piece-pointer', Boolean(pointer) && !ended);
    if (busy || ended) return;
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
    if (event.button !== 0 || event.detail === 0 || event.target.closest('#result, .time-controls')) return;
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
  function render() {
    inversion.disabled = busy || ended;
    inversion.setAttribute('aria-pressed', String(game.direction === -1));
    directionLabel.textContent = game.direction === 1 ? '順行 →' : '← 逆行';
    document.querySelector('.time-controls').classList.toggle('reverse', game.direction === -1);
    const legal = busy || ended ? [] : R.moves(board, player);
    cells.forEach((cell, index) => {
      cell.replaceChildren();
      cell.classList.toggle('legal', legal.includes(index));
      cell.classList.toggle('forward-stone', game.origins[index] === 1);
      cell.classList.toggle('reverse-stone', game.origins[index] === -1);
      cell.setAttribute('aria-disabled', String(!legal.includes(index)));
      cell.setAttribute('aria-label', `${Math.floor(index / 6) + 1}行${index % 6 + 1}列、${board[index] ? name(board[index]) + (game.origins[index] === 1 ? '、順行駒' : game.origins[index] === -1 ? '、逆行駒' : '') : legal.includes(index) ? name(player) + 'を置けます' : '空き'}`);
      if (board[index]) cell.append(piece(board[index]));
    });
    preview();
    if (!busy && !ended) announcement.textContent = `${game.direction === 1 ? '順行' : '逆行'}、${name(player)}の手番です。`;
  }
  function showNotice(text) {
    clearTimeout(noticeTimer);
    notice.textContent = text; announcement.textContent = text; notice.hidden = false;
    noticeTimer = setTimeout(() => { notice.hidden = true; }, 2600);
  }
  function schedulePass() {
    clearTimeout(passTimer);
    if (busy || ended || R.moves(board, player).length) return;
    showNotice(`${name(player)}は置ける場所がないためパスします。`);
    // Leave a between-turn interval in which TIME INVERSION remains available.
    passTimer = setTimeout(() => place(null), 1100);
  }
  inversion.addEventListener('click', event => {
    event.stopPropagation();
    if (busy || ended) return;
    game = R.invert(game);
    render();
    showNotice(game.direction === 1 ? '時間反転：順行' : '時間反転：逆行');
    schedulePass();
  });
  const wait = ms => new Promise(resolve => setTimeout(resolve, reduced.matches ? 0 : ms));
  async function place(index) {
    if (busy || ended) return;
    const move = R.advance(game, index);
    if (!move) return;
    const origin = pointer && !cursor.hidden ? cursor.getBoundingClientRect() : null;
    busy = true;
    clearTimeout(passTimer);
    clearTimeout(noticeTimer); notice.hidden = true;
    grid.setAttribute('aria-busy', 'true');
    render();
    if (index !== null) {
      const placed = piece(player, 'placing');
      cells[index].classList.add(move.placedDirection === 1 ? 'forward-stone' : 'reverse-stone');
      cells[index].append(placed);
      if (origin) {
        const destination = placed.getBoundingClientRect();
        placed.style.setProperty('--drop-x', `${origin.left - destination.left}px`);
        placed.style.setProperty('--drop-y', `${origin.top - destination.top}px`);
      }
      await wait(240);
    }
    for (const i of move.captured) {
      const disc = piece(player, 'turning');
      disc.style.setProperty('--old-face', player === R.BLACK ? 'var(--white)' : 'var(--black)');
      disc.style.setProperty('--old-side', player === R.BLACK ? 'var(--white-side)' : 'var(--black-side)');
      cells[i].replaceChildren(disc);
    }
    if (move.captured.length) await wait(260);
    if (move.removed !== null) {
      cells[move.removed].replaceChildren(piece(move.removedColor, 'vanishing'));
      await wait(300);
      cells[move.removed].replaceChildren();
      cells[move.removed].classList.remove('forward-stone', 'reverse-stone');
    }
    game = move.state;
    ({ board, player, ended } = game);
    busy = false;
    grid.setAttribute('aria-busy', 'false');
    render();
    if (move.forced) showNotice('順行の履歴を戻りきったため、順行に戻ります。');
    else if (move.passed) showNotice(`${name(-player)}はパス。${name(player)}の手番です。`);
    if (ended) {
      const black = board.filter(value => value === R.BLACK).length;
      const white = board.filter(value => value === R.WHITE).length;
      const text = `黒 ${black} ― ${white} 白　${black === white ? '引き分け' : name(black > white ? R.BLACK : R.WHITE) + 'の勝ち'}`;
      document.querySelector('#result-text').textContent = text;
      announcement.textContent = text; result.hidden = false;
      clearTimeout(noticeTimer); notice.hidden = true;
    }
    schedulePass();
  }
  document.querySelector('#restart').addEventListener('click', () => {
    clearTimeout(passTimer); clearTimeout(noticeTimer);
    game = R.createGame(); ({ board, player, ended } = game); hover = -1;
    result.hidden = true; notice.hidden = true; render(); cells[0].focus();
  });
  render();
})();
