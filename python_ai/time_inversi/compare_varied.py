"""Compare two checkpoints on unseen paired random openings, with resume support."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

from .checkpoint import device_for, load
from .mixed_finetune import opening, play
from .rules import BLACK, step, create_game


def file_hash(path):
    digest = sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_from(actions):
    state = create_game()
    for action in actions:
        state = step(state, action)
    return state


def excluded_states(metrics_path, evaluation_path):
    keys = set()
    for line in metrics_path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            for game in json.loads(line)['games']:
                if game['opening_actions']:
                    keys.add(state_from(game['opening_actions']).key())
    evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
    for group in evaluation['results'].values():
        for game in group['varied']['results']:
            keys.add(state_from(game['opening_actions']).key())
    return keys


def starts(seed, count, excluded):
    rng = np.random.default_rng(seed)
    chosen = []
    seen = set(excluded)
    for _ in range(100000):
        if len(chosen) == count:
            break
        state, actions = opening(rng, 4, 8)
        if state.key() not in seen:
            seen.add(state.key())
            chosen.append((state, actions))
    if len(chosen) != count:
        raise RuntimeError(f'Could generate only {len(chosen)} distinct unseen starts')
    return chosen


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--opponent', type=Path, required=True)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--openings', type=int, default=60)
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--seed', type=int, default=62024)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.openings < 1 or args.simulations < 1 or args.threads < 1:
        parser.error('openings, simulations and threads must be positive')
    torch.set_num_threads(args.threads)
    config = dict(candidate=str(args.candidate.resolve()),
                  opponent=str(args.opponent.resolve()),
                  candidate_sha256=file_hash(args.candidate),
                  opponent_sha256=file_hash(args.opponent),
                  metrics_sha256=file_hash(args.metrics),
                  evaluation_sha256=file_hash(args.evaluation),
                  openings=args.openings, simulations=args.simulations, seed=args.seed)
    selected = starts(args.seed, args.openings,
                      excluded_states(args.metrics, args.evaluation))
    if args.output.exists():
        data = json.loads(args.output.read_text(encoding='utf-8'))
        if data['config'] != config:
            raise ValueError('Existing results use different inputs or settings')
        rows = data['results']
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data = dict(config=config, opening_actions=[actions for _, actions in selected],
                    results=[])
        rows = data['results']
        atomic_json(args.output, data)
    if data['opening_actions'] != [actions for _, actions in selected]:
        raise ValueError('Saved openings differ from generated openings')
    if len(rows) > args.openings * 2:
        raise ValueError('Saved result count exceeds planned games')
    candidate, _ = load(args.candidate, device_for(args.device))
    opponent, _ = load(args.opponent, device_for(args.device))
    for game_index in range(len(rows), args.openings * 2):
        opening_index, color_index = divmod(game_index, 2)
        state, actions = selected[opening_index]
        color = BLACK if color_index == 0 else -BLACK
        rng = np.random.default_rng(args.seed + 100000 + game_index)
        start = time.perf_counter()
        _, row = play(candidate, opponent, rng, color, state, actions,
                      args.simulations, 0, 512, False)
        row.update(game=game_index + 1, opening_index=opening_index,
                   seconds=time.perf_counter() - start)
        rows.append(row)
        atomic_json(args.output, data)
        wins = sum(r['result'] == 1 for r in rows)
        draws = sum(r['result'] == 0 for r in rows)
        print(f'{len(rows)}/{args.openings * 2}: 202 vs 207 '
              f'{wins}-{draws}-{len(rows) - wins - draws}', flush=True)


if __name__ == '__main__':
    main()
