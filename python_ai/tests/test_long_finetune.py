import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install requirements.txt to test long fine-tuning')
class LongFineTuneTests(unittest.TestCase):
    def build_source(self, root):
        from time_inversi.buffer import ReplayBuffer
        from time_inversi.checkpoint import save
        from time_inversi.long_finetune import digest
        from time_inversi.network import NetworkConfig, PolicyValueNet

        pilot = root / 'pilot'
        pilot.mkdir()
        model = PolicyValueNet(NetworkConfig(8, 1, 8))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0002)
        sum(param.sum() for param in model.parameters()).backward()
        optimizer.step()
        replay = ReplayBuffer(1000)
        for filename in ('pro100.pt', 'mid.pt'):
            save(pilot / filename, model, optimizer, replay, np.random.default_rng(1), 100, {})
        config = dict(mode='frozen-opponent-finetune-v1', base_iteration=100,
                      pro100_sha256=digest(pilot / 'pro100.pt'), mid_sha256=digest(pilot / 'mid.pt'),
                      self_play_games=0, mid_games=2, anchor_games=2, simulations=1,
                      temperature_moves=2, max_moves=512, batch_size=4, updates=1,
                      learning_rate=.0002, weight_decay=.0001, eval_every=3,
                      eval_games=2, seed=20261022)
        source = pilot / 'iteration-102.pt'
        save(source, model, optimizer, replay, np.random.default_rng(2), 102, config)
        return source

    def test_prepare_then_train_to_target_without_overshoot(self):
        from time_inversi.checkpoint import load
        from time_inversi.long_finetune import main

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            source = self.build_source(root)
            output = root / 'long'
            cmd = ['--source', str(source), '--output', str(output),
                   '--target-iteration', '103', '--threads', '1', '--device', 'cpu']
            main(cmd + ['--prepare-only'])
            _, original = load(source)
            _, prepared = load(output / 'latest.pt')
            self.assertEqual(prepared['iteration'], 102)
            self.assertEqual(prepared['replay'], original['replay'])
            self.assertEqual(prepared['optimizer']['state'].keys(), original['optimizer']['state'].keys())
            self.assertEqual(prepared['training_config']['base_iteration'], 102)
            self.assertEqual(prepared['training_config']['eval_every'], 5)
            self.assertEqual(prepared['training_config']['eval_games'], 10)
            self.assertNotEqual(prepared['numpy_rng'], original['numpy_rng'])
            for key, value in original['model'].items():
                self.assertTrue(torch.equal(value, prepared['model'][key]))
            main(cmd)
            self.assertEqual(load(output / 'latest.pt')[1]['iteration'], 103)
            self.assertEqual(len((output / 'metrics.jsonl').read_text().splitlines()), 1)
            main(cmd)
            self.assertEqual(load(output / 'latest.pt')[1]['iteration'], 103)
            self.assertEqual(len((output / 'metrics.jsonl').read_text().splitlines()), 1)

    def test_refuses_changed_source_and_frozen_reference(self):
        from time_inversi.long_finetune import main

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            source = self.build_source(root)
            output = root / 'long'
            cmd = ['--source', str(source), '--output', str(output), '--prepare-only', '--threads', '1']
            main(cmd)
            with (output / 'mid.pt').open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Frozen reference'):
                main(cmd)

    def test_recovers_evaluation_after_checkpoint_without_duplicate(self):
        from time_inversi.buffer import ReplayBuffer
        from time_inversi.checkpoint import load, save
        from time_inversi.long_finetune import main, recover_pending_evaluation

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            source = self.build_source(root)
            output = root / 'long'
            main(['--source', str(source), '--output', str(output),
                  '--prepare-only', '--threads', '1'])
            model, data = load(output / 'latest.pt')
            optimizer = torch.optim.AdamW(model.parameters())
            optimizer.load_state_dict(data['optimizer'])
            save(output / 'latest.pt', model, optimizer, ReplayBuffer.from_dict(data['replay']),
                 np.random.default_rng(9), 107, data['training_config'])
            (output / 'metrics.jsonl').write_text('{"iteration": 106}\n{"iteration": 107', encoding='utf-8')
            with patch('time_inversi.long_finetune.compare', return_value={'wins': 5}) as compare:
                recover_pending_evaluation(output, 107, torch.device('cpu'))
                self.assertEqual(compare.call_count, 2)
                recover_pending_evaluation(output, 107, torch.device('cpu'))
                self.assertEqual(compare.call_count, 2)
            rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['iteration'] for row in rows], [106, 107])
            self.assertEqual(rows[-1]['evaluation']['mid']['wins'], 5)
            self.assertTrue((output / 'iteration-107.pt').is_file())


if __name__ == '__main__':
    unittest.main()
