"""Measure actual self-play, optimizer and evaluation time without changing saved models."""
import argparse
from datetime import datetime
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .buffer import ReplayBuffer
from .checkpoint import device_for
from .evaluate import evaluate
from .network import NetworkConfig, PolicyValueNet
from .rules import create_game
from .selfplay import self_play
from .train import train_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=3)
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--init-model', type=Path,
                        help='Measure from an Oracle distilled checkpoint instead of random weights')
    args = parser.parse_args()
    if min(args.games, args.simulations, args.updates, args.batch_size, args.threads) < 1:
        parser.error('Counts must be positive')
    device = device_for(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    initial_weights = None
    if args.init_model:
        initial_weights = torch.load(args.init_model, map_location='cpu', weights_only=True)
        if (initial_weights.get('format') != 'time-inversi-ordinary-pretrain-v1' or
                initial_weights.get('teacher_type') != 'oracle'):
            parser.error('--init-model requires an Oracle distilled checkpoint')
    model = PolicyValueNet(NetworkConfig(**initial_weights['network_config'])
                           if initial_weights else NetworkConfig()).to(device)
    if initial_weights:
        model.load_state_dict(initial_weights['model'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    replay = ReplayBuffer(50000)

    def now():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return perf_counter()

    for _ in range(3):
        model.predict(create_game())
    games = []
    for i in range(args.games):
        start = now()
        samples, _ = self_play(model, rng, simulations=args.simulations)
        elapsed = now() - start
        replay.add(samples)
        games.append(dict(seconds=elapsed, turns=len(samples)))
        print(f'Self-play {i + 1}/{args.games}: {elapsed:.2f}s, {len(samples)} turns', flush=True)
    if len(replay) < args.batch_size:
        raise RuntimeError('Not enough samples for requested batch; increase --games')
    start = now()
    for _ in range(args.updates):
        train_batch(model, optimizer, replay.sample(args.batch_size, rng))
    update_seconds = now() - start
    print(f'{args.updates} updates: {update_seconds:.2f}s', flush=True)
    evaluations = {}
    for opponent in ('random', 'greedy'):
        start = now()
        result = evaluate(model, games=2, simulations=args.simulations, opponent=opponent,
                          seed=args.seed + 1000)
        evaluations[opponent] = dict(seconds=now() - start, turns=[x['turns'] for x in result['results']])
        print(f'Evaluate {opponent}, 2 games: {evaluations[opponent]["seconds"]:.2f}s', flush=True)

    # Standard training: ten games + 100 updates; each fifth iteration adds 20 games per opponent.
    ordinary = float(np.mean([g['seconds'] for g in games])) * 10 + update_seconds * 100 / args.updates
    evaluation = sum(x['seconds'] for x in evaluations.values()) * 10
    report = dict(created=datetime.now().isoformat(), python=platform.python_version(),
                  torch=str(torch.__version__), processor=platform.processor(), device=str(device),
                  config={key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
                  network=model.config_dict(), selfplay=games,
                  update_seconds=update_seconds, evaluation=evaluations,
                  estimates_seconds=dict(ordinary_iteration=ordinary, evaluation_iteration=ordinary + evaluation,
                                         hundred_iterations=100 * ordinary + 20 * evaluation),
                  caveat='Estimate from a short sample; excludes checkpoint saving/startup. '
                         'Game length, history length, full replay buffer and machine load can change timings. '
                         'This is not an estimate of time to reach a given playing strength.')
    output = Path(__file__).resolve().parents[1] / 'runs' / 'benchmarks'
    output.mkdir(parents=True, exist_ok=True)
    path = output / (datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.json')
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    print(f'Report: {path}', flush=True)


if __name__ == '__main__':
    main()
