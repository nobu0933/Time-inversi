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


@unittest.skipIf(torch is None, 'Install requirements.txt to test fine-tuning')
class FineTuneTests(unittest.TestCase):
    def test_evaluation_interval_counts_additional_iterations(self):
        from time_inversi.finetune import evaluation_due

        self.assertFalse(evaluation_due(102, 100, 3))
        self.assertTrue(evaluation_due(103, 100, 3))
        self.assertTrue(evaluation_due(106, 100, 3))
        self.assertFalse(evaluation_due(103, 100, 0))

    def test_float32_visit_fractions_are_normalized_before_sampling(self):
        from time_inversi.finetune import sample_policy

        policy = np.array([1 / 3] * 3, dtype=np.float32)
        choices = [sample_policy(np.random.default_rng(seed), policy) for seed in range(10)]
        self.assertTrue(all(0 <= choice < 3 for choice in choices))

    def test_frozen_games_train_only_candidate_turns(self):
        from time_inversi.finetune import frozen_game
        from time_inversi.mcts import UniformEvaluator
        from time_inversi.rules import BLACK, WHITE, legal_actions

        for color in (BLACK, WHITE):
            samples, record = frozen_game(UniformEvaluator(), UniformEvaluator(),
                                          np.random.default_rng(44), color, simulations=2)
            self.assertTrue(samples)
            self.assertEqual(record['samples'], len(samples))
            self.assertEqual(record['color'], color)
            self.assertLessEqual(len(samples), record['turns'])
            for sample in samples:
                self.assertEqual(sample.state.player, color)
                self.assertEqual(sample.value, record['result'])
                self.assertAlmostEqual(float(sample.policy.sum()), 1)
                self.assertTrue(set(np.flatnonzero(sample.policy)).issubset(
                    set(legal_actions(sample.state))))

    def test_new_run_keeps_frozen_opponents_and_resumes(self):
        from time_inversi.buffer import ReplayBuffer
        from time_inversi.checkpoint import load, save
        from time_inversi.finetune import main
        from time_inversi.network import NetworkConfig, PolicyValueNet

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            model = PolicyValueNet(NetworkConfig(8, 1, 8))
            optimizer = torch.optim.AdamW(model.parameters())
            base, mid, output = root / 'base.pt', root / 'mid.pt', root / 'final'
            save(base, model, optimizer, ReplayBuffer(1000), np.random.default_rng(1), 100, {})
            save(mid, model, optimizer, ReplayBuffer(1000), np.random.default_rng(2), 100, {})
            settings = ['--base', str(base), '--mid', str(mid), '--output', str(output),
                        '--iterations', '1', '--self-play-games', '0', '--mid-games', '2',
                        '--anchor-games', '2', '--simulations', '1', '--batch-size', '4',
                        '--updates', '1', '--eval-every', '1', '--eval-games', '2',
                        '--threads', '1', '--device', 'cpu']
            main(settings)
            _, first = load(output / 'latest.pt')
            self.assertEqual(first['iteration'], 101)
            self.assertEqual(first['training_config']['mode'], 'frozen-opponent-finetune-v1')
            self.assertTrue((output / 'iteration-101.pt').is_file())
            self.assertEqual(load(output / 'pro100.pt')[1]['iteration'], 100)
            self.assertEqual(load(base)[1]['iteration'], 100)
            self.assertGreater(len(first['replay']['samples']), 0)
            main(['--resume', str(output / 'latest.pt'), '--iterations', '1',
                  '--threads', '1', '--device', 'cpu'])
            _, second = load(output / 'latest.pt')
            self.assertEqual(second['iteration'], 102)
            self.assertTrue((output / 'iteration-102.pt').is_file())
            self.assertEqual(len((output / 'metrics.jsonl').read_text().splitlines()), 2)
            continuous = root / 'continuous'
            direct = settings.copy()
            direct[direct.index(str(output))] = str(continuous)
            direct[direct.index('1')] = '2'
            main(direct)
            _, expected = load(continuous / 'latest.pt')
            self.assertEqual(expected['replay'], second['replay'])
            self.assertEqual(expected['numpy_rng'], second['numpy_rng'])
            for name, weight in expected['model'].items():
                self.assertTrue(torch.equal(weight, second['model'][name]), name)

    def test_self_play_record_is_json_serializable(self):
        from time_inversi.buffer import ReplayBuffer
        from time_inversi.checkpoint import save
        from time_inversi.finetune import main
        from time_inversi.network import NetworkConfig, PolicyValueNet

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            model = PolicyValueNet(NetworkConfig(8, 1, 8))
            optimizer = torch.optim.AdamW(model.parameters())
            base, mid, output = root / 'base.pt', root / 'mid.pt', root / 'out'
            for path in (base, mid):
                save(path, model, optimizer, ReplayBuffer(1000), np.random.default_rng(1), 100, {})
            main(['--base', str(base), '--mid', str(mid), '--output', str(output),
                  '--iterations', '1', '--self-play-games', '1', '--mid-games', '0',
                  '--anchor-games', '0', '--simulations', '2', '--batch-size', '4',
                  '--updates', '1', '--eval-every', '0', '--threads', '1', '--device', 'cpu'])
            record = json.loads((output / 'metrics.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(record['iteration'], 101)
            self.assertEqual(record['games'][0]['type'], 'self')
            self.assertIsInstance(record['games'][0]['passes'], int)


if __name__ == '__main__':
    unittest.main()
