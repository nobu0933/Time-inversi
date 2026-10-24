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
class FineTune202Tests(unittest.TestCase):
    def test_prepare_train_evaluate_resume_and_budget(self):
        from time_inversi.buffer import ReplayBuffer
        from time_inversi.checkpoint import load, save
        from time_inversi.finetune_202 import main
        from time_inversi.network import NetworkConfig, PolicyValueNet

        torch.set_num_threads(1)
        scratch = Path(__file__).resolve().parents[1] / '.tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder, contextlib.redirect_stdout(io.StringIO()) as stream:
                root = Path(folder)
                model = PolicyValueNet(NetworkConfig(8, 1, 8))
                optimizer = torch.optim.AdamW(model.parameters())
                paths = {}
                for name, iteration in (('base', 202), ('mid', 100), ('pro100', 100),
                                        ('pro122', 122), ('pro207', 207)):
                    path = root / (name + '.pt')
                    save(path, model, optimizer, ReplayBuffer(100),
                         np.random.default_rng(iteration), iteration, {})
                    paths[name] = path
                output = root / 'run'
                command = ['--base', str(paths['base']), '--mid', str(paths['mid']),
                           '--pro100', str(paths['pro100']), '--pro122', str(paths['pro122']),
                           '--pro207', str(paths['pro207']), '--output', str(output),
                           '--target-iteration', '203', '--max-hours', '1',
                           '--self-play-games', '0', '--mid-games', '2',
                           '--pro100-games', '2', '--pro122-games', '2',
                           '--pro207-games', '2', '--opening-probability', '1',
                           '--simulations', '1', '--batch-size', '4', '--updates', '1',
                           '--buffer-size', '100', '--eval-every', '1',
                           '--eval-standard-games', '2', '--eval-openings', '1',
                           '--threads', '1', '--device', 'cpu']
                main(command + ['--prepare-only'])
                _, prepared = load(output / 'latest.pt')
                self.assertEqual(prepared['iteration'], 202)
                self.assertEqual(prepared['replay']['samples'], [])
                self.assertEqual(prepared['training_config']['learning_rate'], .00005)
                _, base = load(paths['base'])
                for key, weight in base['model'].items():
                    self.assertTrue(torch.equal(weight, prepared['model'][key]))
                main(command)
                _, trained = load(output / 'latest.pt')
                self.assertEqual(trained['iteration'], 203)
                rows = [json.loads(line) for line in
                        (output / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertEqual({name: sum(game['type'] == name for game in rows[0]['games'])
                                  for name in ('mid', 'pro100', 'pro122', 'pro207')},
                                 dict(mid=2, pro100=2, pro122=2, pro207=2))
                self.assertTrue(all(game['opening_actions'] for game in rows[0]['games']))
                for name in ('mid', 'pro100', 'pro122', 'pro207'):
                    self.assertEqual(rows[0]['evaluation'][name]['standard']['games'], 2)
                    self.assertEqual(rows[0]['evaluation'][name]['varied']['games'], 2)
                self.assertTrue((output / 'iteration-203.pt').is_file())
                main(command)
                self.assertEqual(len((output / 'metrics.jsonl').read_text().splitlines()), 1)
                main(command + ['--target-iteration', '204', '--max-hours', '.00001'])
                self.assertEqual(load(output / 'latest.pt')[1]['iteration'], 203)
                self.assertIn('Compute budget reached', stream.getvalue())


if __name__ == '__main__':
    unittest.main()
