'use strict';
importScripts('../game/rules.js', 'network.js', 'search.js', '../../assets/ai/model.js');
const network = new TimeInversiNetwork(globalThis.TimeInversiWeights);
delete globalThis.TimeInversiWeights;
self.onmessage = ({ data }) => {
  try {
    // A fresh tree keeps memory bounded and matches the Python serving mode.
    const result = new TimeInversiSearch(network).search(data.state, 64);
    self.postMessage({ id: data.id, action: result.action });
  } catch (error) { self.postMessage({ id: data.id, error: error.message }); }
};
