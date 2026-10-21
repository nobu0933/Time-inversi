'use strict';
importScripts('../game/rules.js', 'network.js', 'search.js');
const modelFiles = { MID: '../../assets/ai/model.js', PRO: '../../assets/ai/pro-model.js' };
const networks = {};
self.onmessage = ({ data }) => {
  try {
    const mode = data.mode || 'MID';
    if (!Object.hasOwn(modelFiles, mode)) throw new Error('Unknown CPU mode');
    if (!networks[mode]) {
      importScripts(modelFiles[mode]);
      networks[mode] = new TimeInversiNetwork(globalThis.TimeInversiWeights);
      delete globalThis.TimeInversiWeights;
    }
    // A fresh tree keeps memory bounded and matches the Python serving mode.
    const result = new TimeInversiSearch(networks[mode]).search(data.state, 64);
    self.postMessage({ id: data.id, action: result.action });
  } catch (error) { self.postMessage({ id: data.id, error: error.message }); }
};
