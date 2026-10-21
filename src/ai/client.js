(() => {
  'use strict';
  // Resolve relative to this script, including project subpaths and itch.io embeds.
  const workerURL = new URL('worker.js', document.currentScript.src);
  let worker, sequence = 0, pending;
  function stop(error) {
    worker?.terminate(); worker = null;
    if (pending) { const task = pending; pending = null; task.cleanup(); task.reject(error); }
  }
  window.ProAI = {
    choose(state, { signal, mode = 'MID' } = {}) {
      if (pending) stop(new DOMException('Superseded', 'AbortError'));
      if (signal?.aborted) return Promise.reject(new DOMException('Aborted', 'AbortError'));
      return new Promise((resolve, reject) => {
        const id = ++sequence;
        const abort = () => stop(new DOMException('Aborted', 'AbortError'));
        pending = { id, resolve, reject, cleanup: () => signal?.removeEventListener('abort', abort) };
        signal?.addEventListener('abort', abort, { once: true });
        try {
          if (!worker) {
            worker = new Worker(workerURL);
            worker.onmessage = ({ data }) => {
              if (!pending || pending.id !== data.id) return;
              const task = pending; pending = null; task.cleanup();
              if (data.error) task.reject(new Error(data.error));
              else task.resolve({ action: data.action });
            };
            worker.onerror = event => { event.preventDefault(); stop(new Error('Unable to load browser AI model')); };
          }
          worker.postMessage({ id, state, mode });
        } catch (error) { stop(error); }
      });
    }
  };
})();
