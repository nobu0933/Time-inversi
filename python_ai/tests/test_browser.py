from dataclasses import replace
import json
from pathlib import Path
import random
import shutil
import subprocess
import unittest

import numpy as np
import torch

from time_inversi.checkpoint import load
from time_inversi.rules import create_game, legal_actions, step

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'python_ai/runs/experiment01/latest.pt'
PRO_CHECKPOINT = ROOT / 'python_ai/runs/pro-mixed-from-202/iteration-307.pt'


@unittest.skipUnless(shutil.which('node') and CHECKPOINT.exists(), 'Browser parity needs Node.js and the original checkpoint')
class BrowserInferenceTests(unittest.TestCase):
    def _compare_model(self, checkpoint, asset):
        torch.set_num_threads(2)
        model, _ = load(checkpoint)
        rng = random.Random(701)
        states = [create_game()]
        for _ in range(4):
            state = create_game()
            while not state.ended:
                if len(states) < 100 and (state.direction == -1 or rng.random() < .4):
                    states.append(state)
                state = step(state, rng.choice(legal_actions(state)))
        # Explicitly exercise pass tokens, empty histories and long, differently sized sequences.
        states.extend([replace(create_game(), forward=(None,) * 60 + (8, None), reverse=(13, None)),
                       replace(create_game(), reverse=(None,) * 51)])
        blocked = replace(create_game(), board=(1,) * 35 + (0,), used_corners=(0, 5, 30, 35))
        states.append(blocked)
        self.assertTrue(any(s.direction == -1 for s in states))
        process = subprocess.run(['node', str(Path(__file__).with_name('browser_inference.cjs')),
                                  str(ROOT / asset)],
                                 input=json.dumps([s.to_dict() for s in states]), text=True,
                                 capture_output=True, check=True)
        predictions = json.loads(process.stdout)
        self.assertEqual(len(predictions), len(states))
        errors = []
        for state, actual in zip(states, predictions):
            policy, value = model.predict(state)
            np.testing.assert_allclose(actual['policy'], policy, atol=2e-5, rtol=2e-5)
            self.assertAlmostEqual(actual['value'], value, delta=2e-5)
            errors.append(max(np.max(np.abs(policy - actual['policy'])), abs(value - actual['value'])))
        print(f'{asset}: {len(states)} states, maximum error {max(errors):.8g}')

    def test_shipped_mid_matches_python_with_complete_histories(self):
        self._compare_model(CHECKPOINT, 'assets/ai/model.js')

    @unittest.skipUnless(PRO_CHECKPOINT.exists(), 'PRO parity needs the local training checkpoint')
    def test_shipped_pro_matches_python_with_complete_histories(self):
        self._compare_model(PRO_CHECKPOINT, 'assets/ai/pro-model.js')
