from dataclasses import replace
from pathlib import Path
import tempfile
import contextlib
import io
import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install requirements.txt to test actual learning')
class LearningTests(unittest.TestCase):
    def test_complete_training_resume_matches_uninterrupted(self):
        from time_inversi.train import main
        from time_inversi.checkpoint import load
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            settings = ['--self-play-games', '1', '--simulations', '2', '--batch-size', '4',
                        '--updates', '1', '--channels', '8', '--blocks', '1', '--history-hidden', '8',
                        '--eval-every', '1', '--eval-games', '2', '--threads', '1', '--device', 'cpu',
                        '--buffer-size', '400', '--learning-rate', '0.002', '--seed', '42']
            main(settings + ['--iterations', '2', '--output', str(root / 'continuous')])
            main(settings + ['--iterations', '1', '--output', str(root / 'split')])
            path = root / 'split' / 'latest.pt'
            main(['--resume', str(path), '--iterations', '1', '--device', 'cpu'])
            _, expected = load(root / 'continuous' / 'latest.pt')
            _, actual = load(path)
            self.assertEqual(actual['iteration'], 2)
            self.assertEqual(expected['replay'], actual['replay'])
            self.assertEqual(expected['numpy_rng'], actual['numpy_rng'])
            self.assertTrue(torch.equal(expected['torch_rng'], actual['torch_rng']))
            for name, weight in expected['model'].items():
                self.assertTrue(torch.equal(weight, actual['model'][name]), name)
            for key in ('simulations', 'batch_size', 'channels', 'learning_rate', 'seed'):
                self.assertEqual(expected['training_config'][key], actual['training_config'][key])
            import json
            logs = [json.loads(line) for line in (root / 'split' / 'metrics.jsonl').read_text().splitlines()]
            reference = [json.loads(line) for line in (root / 'continuous' / 'metrics.jsonl').read_text().splitlines()]
            for a, b in zip(logs, reference):
                a.pop('seconds'); b.pop('seconds')
                self.assertEqual(a, b)
            with contextlib.redirect_stderr(io.StringIO()):
                for extra in (['--channels', '16'], ['--learning-rate', '.01'],
                              ['--output', str(root / 'continuous')]):
                    with self.assertRaises(SystemExit):
                        main(['--resume', str(path), '--iterations', '1'] + extra)

    def test_rejects_invalid_value_targets(self):
        from time_inversi.network import NetworkConfig, PolicyValueNet
        from time_inversi.rules import create_game
        from time_inversi.selfplay import Sample
        from time_inversi.train import train_batch
        model = PolicyValueNet(NetworkConfig(8, 1, 8))
        optimizer = torch.optim.AdamW(model.parameters())
        policy, _ = model.predict(create_game())
        for value in (float('nan'), float('inf'), 2.0, -2.0):
            with self.assertRaises(ValueError):
                train_batch(model, optimizer, [Sample(create_game(), policy, value)])

    def test_optimizer_checkpoint_resume_and_full_history(self):
        from time_inversi.network import NetworkConfig, PolicyValueNet
        from time_inversi.rules import create_game
        from time_inversi.selfplay import Sample
        from time_inversi.buffer import ReplayBuffer
        from time_inversi.train import train_batch
        from time_inversi.checkpoint import save, load
        torch.set_num_threads(1)
        torch.manual_seed(1)
        rng = np.random.default_rng(1)
        model = PolicyValueNet(NetworkConfig(8, 1, 8))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        state = replace(create_game(), forward=(None,) * 50)
        policy, _ = model.predict(state)
        samples = [Sample(state, policy, 1.0), Sample(create_game(), policy, -1.0)]
        buffer = ReplayBuffer(20)
        buffer.add(samples)
        before = model.policy.weight.detach().clone()
        stats = train_batch(model, optimizer, samples)
        self.assertTrue(np.isfinite(stats['loss']))
        self.assertFalse(torch.equal(before, model.policy.weight))
        prediction = model.predict(state)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'latest.pt'
            save(path, model, optimizer, buffer, rng, 4, {})
            restored, data = load(path)
            self.assertEqual(data['iteration'], 4)
            self.assertEqual(len(ReplayBuffer.from_dict(data['replay'])), 2)
            np.testing.assert_allclose(restored.predict(state)[0], prediction[0], atol=1e-7)
            restored_optimizer = torch.optim.AdamW(restored.parameters())
            restored_optimizer.load_state_dict(data['optimizer'])
            # Identical next update confirms optimizer state resumes, not just weights.
            train_batch(model, optimizer, samples)
            train_batch(restored, restored_optimizer, samples)
            np.testing.assert_allclose(restored.predict(state)[0], model.predict(state)[0], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
