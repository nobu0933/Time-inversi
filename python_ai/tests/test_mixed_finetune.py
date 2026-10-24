import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install requirements.txt to test mixed fine-tuning')
class MixedFineTuneTests(unittest.TestCase):
    def test_varied_opening_games_train_only_candidate_turns(self):
        from time_inversi.mcts import UniformEvaluator
        from time_inversi.mixed_finetune import opening, play
        from time_inversi.rules import BLACK, WHITE, legal_actions

        model = UniformEvaluator()
        start, actions = opening(np.random.default_rng(123), 4, 8)
        self.assertGreaterEqual(len(actions), 4)
        self_samples, self_row = play(model, None, np.random.default_rng(456), None,
                                      start, actions, 2, 12, 512, True)
        self.assertEqual(len(self_samples), self_row['turns_after_opening'])
        self.assertEqual(self_row['opening_actions'], actions)
        for color in (BLACK, WHITE):
            samples, row = play(model, model, np.random.default_rng(321), color,
                                start, actions, 2, 12, 512, True)
            self.assertTrue(samples)
            self.assertEqual(len(samples), row['samples'])
            self.assertLessEqual(len(samples), row['turns_after_opening'])
            self.assertEqual(row['opening_actions'], actions)
            for sample in samples:
                self.assertEqual(sample.state.player, color)
                self.assertEqual(sample.value, row['result'])
                self.assertAlmostEqual(float(sample.policy.sum()), 1)
                self.assertTrue(set(np.flatnonzero(sample.policy)).issubset(
                    set(legal_actions(sample.state))))

    def test_prepare_train_evaluate_and_resume_to_target(self):
        from time_inversi.buffer import ReplayBuffer
        from time_inversi.checkpoint import load, save
        from time_inversi.mixed_finetune import main
        from time_inversi.network import NetworkConfig, PolicyValueNet

        torch.set_num_threads(1)
        scratch = Path(__file__).resolve().parents[1] / '.tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            model = PolicyValueNet(NetworkConfig(8, 1, 8))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0002)
            inputs = {}
            for name, iteration in (('base', 147), ('mid', 100), ('pro100', 100),
                                    ('pro122', 122)):
                path = root / (name + '.pt')
                save(path, model, optimizer, ReplayBuffer(100),
                     np.random.default_rng(iteration), iteration, {})
                inputs[name] = path
            output = root / 'mixed'
            command = ['--base', str(inputs['base']), '--mid', str(inputs['mid']),
                       '--pro100', str(inputs['pro100']), '--pro122', str(inputs['pro122']),
                       '--output', str(output), '--target-iteration', '148',
                       '--self-play-games', '0', '--opening-probability', '1',
                       '--simulations', '1', '--batch-size', '4', '--updates', '1',
                       '--buffer-size', '100', '--eval-every', '1',
                       '--eval-standard-games', '2', '--eval-openings', '1',
                       '--threads', '1', '--device', 'cpu']
            main(command + ['--prepare-only'])
            _, prepared = load(output / 'latest.pt')
            _, base = load(inputs['base'])
            self.assertEqual(prepared['iteration'], 147)
            self.assertEqual(prepared['replay']['samples'], [])
            self.assertEqual(prepared['training_config']['learning_rate'], .0001)
            for key, weight in base['model'].items():
                self.assertTrue(torch.equal(weight, prepared['model'][key]))
            main(command)
            _, trained = load(output / 'latest.pt')
            self.assertEqual(trained['iteration'], 148)
            self.assertGreater(len(trained['replay']['samples']), 0)
            rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual({name: sum(game['type'] == name for game in rows[0]['games'])
                              for name in ('mid', 'pro100', 'pro122')},
                             {'mid': 2, 'pro100': 2, 'pro122': 2})
            self.assertTrue(all(game['opening_actions'] for game in rows[0]['games']))
            self.assertEqual(rows[0]['updates'], 1)
            for name in ('mid', 'pro100', 'pro122'):
                self.assertEqual(rows[0]['evaluation'][name]['standard']['games'], 2)
                self.assertEqual(rows[0]['evaluation'][name]['varied']['games'], 2)
            self.assertTrue((output / 'evaluation-148.json').is_file())
            self.assertTrue((output / 'iteration-148.pt').is_file())
            main(command)
            self.assertEqual(load(output / 'latest.pt')[1]['iteration'], 148)
            self.assertEqual(len((output / 'metrics.jsonl').read_text().splitlines()), 1)
            pending = {key: value for key, value in rows[0].items()
                       if key not in ('evaluation', 'evaluation_seconds', 'seconds')}
            (output / 'metrics.jsonl').unlink()
            (output / 'evaluation-148.json').unlink()
            (output / 'iteration-148.pt').unlink()
            (output / 'pending.json').write_text(json.dumps(pending), encoding='utf-8')
            main(command)
            recovered = json.loads((output / 'metrics.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(recovered['iteration'], 148)
            self.assertIn('evaluation', recovered)
            self.assertTrue((output / 'iteration-148.pt').is_file())
            with (output / 'mid.pt').open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Frozen checkpoint changed'):
                main(command)


if __name__ == '__main__':
    unittest.main()
