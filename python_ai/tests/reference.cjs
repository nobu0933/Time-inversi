// Read-only oracle: use the actual browser rules, not a second JS implementation.
const fs = require('node:fs');
const R = require('../../src/game/rules.js');
const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
const output = cases.map(test => {
  let state = test.initial;
  return test.commands.map(command => {
    const turn = command === 'invert' ? R.invert(state) : R.advance(state, command);
    if (turn) state = turn.state;
    return turn;
  });
});
process.stdout.write(JSON.stringify(output));
