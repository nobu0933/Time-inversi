const fs = require('node:fs');
const path = require('node:path');
require(path.resolve(process.argv[2]));
const Network = require('../../src/ai/network.js');
const network = new Network(globalThis.TimeInversiWeights);
const states = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(states.map(state => {
  const { policy, value } = network.predict(state);
  return { policy: Array.from(policy), value };
})));
